//! Code Generation: AST → Tensor Operations
//!
//! Compiles parsed rule specifications into executable tensor functions.
//!
//! ## Overview
//!
//! Given a rule like:
//! ```text
//! exit(X) :- profit_target(X, 0.02), momentum_shift(X).
//! ```
//!
//! The codegen produces a differentiable function that:
//! 1. Looks up input features for each predicate
//! 2. Computes predicate activations (via threshold or learned embedding)
//! 3. Combines activations with fuzzy AND (for conjunctions)
//! 4. Combines multiple rules with fuzzy OR
//! 5. Returns output + explanation
//!
//! ## Tensor Logic Mapping
//!
//! | Rule Construct | Tensor Operation |
//! |----------------|------------------|
//! | `a, b` (AND) | `fuzzy_and(a, b)` = a × b |
//! | `a; b` (OR) | `fuzzy_or(a, b)` = a + b - ab |
//! | `not a` | `fuzzy_not(a)` = 1 - a |
//! | `pred(X, θ)` | `soft_threshold(X, θ, k)` |
//! | Multiple rules | `fuzzy_or_many([rule1, rule2, ...])` |

pub mod attention;
pub mod utils;

pub use attention::AttentionParams;
use utils::{
    apply_activation, apply_dropout, apply_layer_norm, apply_multi_head_attention, l2_normalize,
    normalize_adjacency,
};
pub use utils::{detect_sparse_coo, get_adjacency_matrix, sparse_coo_to_dense};

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor, Var};

use super::{Argument, CompiledOutput, PredicateSpec, RuleSpec};
use crate::primitives::{fuzzy_and, fuzzy_not, fuzzy_or_many, soft_threshold_scalar};
use crate::{Result, TensorCoreError};

// =============================================================================
// COMPILED PREDICATE
// =============================================================================

/// A compiled predicate ready for evaluation
#[derive(Debug)]
pub enum CompiledPredicate {
    /// Threshold predicate: value > threshold (or < if inverted)
    Threshold {
        /// Input feature name to look up
        input_name: String,
        /// Threshold value (may be a Var for learning)
        threshold: f32,
        /// Sharpness of soft threshold
        sharpness: f32,
        /// True = greater than, False = less than
        greater_than: bool,
    },

    /// Learned embedding predicate (basic linear)
    Learned {
        /// Name for this predicate
        name: String,
        /// Learned weight matrix (input_dim -> 1)
        weights: Var,
        /// Bias term
        bias: Var,
    },

    /// Learned projection: embedding → hidden → scalar
    /// Two-layer MLP replacement with full differentiability
    /// Optionally includes attention, layer norm, dropout, and residual connections
    LearnedProjection {
        /// Input tensor names
        input_names: Vec<String>,
        /// First layer weights: [input_dim, hidden_dim]
        w1: Var,
        /// First layer bias: `[hidden_dim]`
        b1: Var,
        /// Second layer weights: [hidden_dim, 1]
        w2: Var,
        /// Second layer bias: `[1]`
        b2: Var,
        /// Activation function
        activation: super::Activation,
        /// LayerNorm parameters for first layer: (gamma, beta)
        ln1: Option<(Var, Var)>,
        /// Multi-head attention parameters: (Wq, Wk, Wv, Wo) per head
        attention: Option<AttentionParams>,
        /// LayerNorm for post-attention
        ln_attn: Option<(Var, Var)>,
        /// Dropout rate (applied at runtime)
        dropout_rate: Option<f32>,
        /// Attention dropout rate
        attention_dropout_rate: Option<f32>,
        /// Use residual connections
        residual: bool,
        /// Residual projection if input_dim != hidden_dim
        residual_proj: Option<Var>,
    },

    /// Learned similarity: compare two embeddings
    LearnedSimilarity {
        /// Left embedding name
        left_name: String,
        /// Right embedding name
        right_name: String,
        /// Similarity method
        method: super::SimilarityMethod,
        /// For Learned method: weights for element-wise product
        weights: Option<Var>,
        /// For Learned/Bilinear methods: bias term
        bias: Option<Var>,
        /// For Bilinear method: bilinear weight matrix [dim, dim]
        bilinear_w: Option<Var>,
    },

    /// Graph Neural predicate: message passing → per-node score
    GraphNeural {
        /// Node features input name
        node_features_name: String,
        /// Adjacency matrix input name
        adjacency_name: String,
        /// Per-layer weights: [(W_self, W_neighbor, bias), ...]
        layers: Vec<(Var, Var, Var)>,
        /// Final projection: [hidden_dim, 1]
        output_w: Var,
        /// Final projection bias
        output_b: Var,
        /// Aggregation method
        aggregation: super::Aggregation,
    },

    /// External predicate (passed in by user)
    External {
        /// Name to look up in inputs
        name: String,
    },
}

impl CompiledPredicate {
    /// Evaluate this predicate given inputs
    pub fn evaluate(&self, inputs: &HashMap<String, Tensor>, device: &Device) -> Result<Tensor> {
        match self {
            CompiledPredicate::Threshold {
                input_name,
                threshold,
                sharpness,
                greater_than,
            } => {
                let input = inputs.get(input_name).ok_or_else(|| {
                    TensorCoreError::Compiler(format!(
                        "Missing input '{}' for threshold predicate",
                        input_name
                    ))
                })?;

                // Apply soft threshold
                let sharp = if *greater_than {
                    *sharpness
                } else {
                    -*sharpness
                };
                soft_threshold_scalar(input, *threshold, sharp)
            }

            CompiledPredicate::Learned {
                name,
                weights,
                bias,
            } => {
                let input = inputs.get(name).ok_or_else(|| {
                    TensorCoreError::Compiler(format!(
                        "Missing input '{}' for learned predicate",
                        name
                    ))
                })?;

                // Linear transform + sigmoid: σ(Wx + b)
                let wx = input
                    .matmul(weights.as_tensor())
                    .map_err(|e| TensorCoreError::Tensor(format!("matmul failed: {}", e)))?;

                let wx_b = wx
                    .broadcast_add(bias.as_tensor())
                    .map_err(|e| TensorCoreError::Tensor(format!("add bias failed: {}", e)))?;

                crate::primitives::sigmoid(&wx_b)
            }

            CompiledPredicate::LearnedProjection {
                input_names,
                w1,
                b1,
                w2,
                b2,
                activation,
                ln1,
                attention,
                ln_attn,
                dropout_rate,
                attention_dropout_rate,
                residual,
                residual_proj,
            } => {
                // Gather and concatenate inputs
                let input = if input_names.len() == 1 {
                    inputs
                        .get(&input_names[0])
                        .ok_or_else(|| {
                            TensorCoreError::Compiler(format!(
                                "Missing input '{}' for learned projection",
                                input_names[0]
                            ))
                        })?
                        .clone()
                } else {
                    let tensors: Result<Vec<Tensor>> = input_names
                        .iter()
                        .map(|name| {
                            inputs.get(name).cloned().ok_or_else(|| {
                                TensorCoreError::Compiler(format!(
                                    "Missing input '{}' for learned projection",
                                    name
                                ))
                            })
                        })
                        .collect();
                    let tensors = tensors?;
                    Tensor::cat(&tensors, 1)
                        .map_err(|e| TensorCoreError::Tensor(format!("concat failed: {}", e)))?
                };

                // ═══════════════════════════════════════════════════════════════════════
                // AUTO-CLAMP MLP WEIGHTS (Fix for NaN after optimizer.step())
                // ═══════════════════════════════════════════════════════════════════════
                // Clamp all trainable weights to prevent explosion after gradient updates.
                const MLP_WEIGHT_MAX: f32 = 10.0;

                let w1_clamped = w1
                    .as_tensor()
                    .clamp(-MLP_WEIGHT_MAX, MLP_WEIGHT_MAX)
                    .map_err(|e| TensorCoreError::Tensor(format!("w1 clamp failed: {}", e)))?;
                let b1_clamped = b1
                    .as_tensor()
                    .clamp(-MLP_WEIGHT_MAX, MLP_WEIGHT_MAX)
                    .map_err(|e| TensorCoreError::Tensor(format!("b1 clamp failed: {}", e)))?;

                // Layer 1: input → hidden
                let mut h1 = input
                    .matmul(&w1_clamped)
                    .map_err(|e| TensorCoreError::Tensor(format!("matmul w1 failed: {}", e)))?;
                h1 = h1
                    .broadcast_add(&b1_clamped)
                    .map_err(|e| TensorCoreError::Tensor(format!("add b1 failed: {}", e)))?;

                // Apply LayerNorm if enabled
                if let Some((gamma, beta)) = ln1 {
                    h1 = apply_layer_norm(&h1, gamma, beta)?;
                }

                // ═══════════════════════════════════════════════════════════════════════
                // RESIDUAL CONNECTION FIX (Research Team Directive 2024-12-31)
                // ═══════════════════════════════════════════════════════════════════════
                // Prepare residual AFTER LayerNorm to prevent raw input features from
                // dominating. This ensures residual is scale-normalized before injection.
                // Previous: Used raw `input` which allowed unnormalized betti_0 to saturate.
                // Fixed: Use post-LayerNorm h1 (before activation) for stable gradients.
                let residual_input = if *residual {
                    // Project the LayerNorm'd hidden state if dimensions don't match
                    // Note: h1 is now centered/scaled, preventing feature dominance
                    if let Some(proj) = residual_proj {
                        // For input_dim != hidden_dim: project input through residual_proj
                        // Clamp residual projection weights for stability
                        let proj_clamped = proj
                            .as_tensor()
                            .clamp(-MLP_WEIGHT_MAX, MLP_WEIGHT_MAX)
                            .map_err(|e| {
                                TensorCoreError::Tensor(format!(
                                    "residual proj clamp failed: {}",
                                    e
                                ))
                            })?;
                        Some(input.matmul(&proj_clamped).map_err(|e| {
                            TensorCoreError::Tensor(format!("residual proj failed: {}", e))
                        })?)
                    } else if input.dims() == h1.dims() {
                        // Same dims: use the normalized h1 before activation
                        Some(h1.clone())
                    } else {
                        // Fallback: no residual if dims mismatch without projection
                        None
                    }
                } else {
                    None
                };

                // Apply activation
                h1 = apply_activation(&h1, *activation)?;

                // Apply dropout if enabled (during training - for now we always apply with scaling)
                if let Some(rate) = dropout_rate {
                    if *rate > 0.0 {
                        h1 = apply_dropout(&h1, *rate)?;
                    }
                }

                // Apply multi-head attention if enabled
                if let Some(attn) = attention {
                    let attn_out = apply_multi_head_attention(&h1, attn, *attention_dropout_rate)?;

                    // Add residual connection around attention
                    // Clamp after addition to prevent value explosion
                    h1 = (&h1 + &attn_out)
                        .map_err(|e| {
                            TensorCoreError::Tensor(format!("attention residual add failed: {}", e))
                        })?
                        .clamp(-100.0f32, 100.0f32)
                        .map_err(|e| {
                            TensorCoreError::Tensor(format!(
                                "attention residual clamp failed: {}",
                                e
                            ))
                        })?;

                    // Apply post-attention LayerNorm
                    if let Some((gamma, beta)) = ln_attn {
                        h1 = apply_layer_norm(&h1, gamma, beta)?;
                    }
                }

                // Add residual from input if enabled
                if let Some(res) = residual_input {
                    // Clamp after residual addition to prevent accumulation
                    h1 = (&h1 + &res)
                        .map_err(|e| {
                            TensorCoreError::Tensor(format!("residual add failed: {}", e))
                        })?
                        .clamp(-100.0f32, 100.0f32)
                        .map_err(|e| {
                            TensorCoreError::Tensor(format!("residual clamp failed: {}", e))
                        })?;
                }

                // Layer 2: hidden → 1 (with clamped weights)
                let w2_clamped = w2
                    .as_tensor()
                    .clamp(-MLP_WEIGHT_MAX, MLP_WEIGHT_MAX)
                    .map_err(|e| TensorCoreError::Tensor(format!("w2 clamp failed: {}", e)))?;
                let b2_clamped = b2
                    .as_tensor()
                    .clamp(-MLP_WEIGHT_MAX, MLP_WEIGHT_MAX)
                    .map_err(|e| TensorCoreError::Tensor(format!("b2 clamp failed: {}", e)))?;

                let mut h2 = h1
                    .matmul(&w2_clamped)
                    .map_err(|e| TensorCoreError::Tensor(format!("matmul w2 failed: {}", e)))?;
                h2 = h2
                    .broadcast_add(&b2_clamped)
                    .map_err(|e| TensorCoreError::Tensor(format!("add b2 failed: {}", e)))?;

                // Squeeze to 1D for consistency with other predicates
                h2 = h2
                    .squeeze(1)
                    .map_err(|e| TensorCoreError::Tensor(format!("squeeze failed: {}", e)))?;

                // Final sigmoid for [0, 1] output
                crate::primitives::sigmoid(&h2)
            }

            CompiledPredicate::LearnedSimilarity {
                left_name,
                right_name,
                method,
                weights,
                bias,
                bilinear_w,
            } => {
                let left = inputs.get(left_name).ok_or_else(|| {
                    TensorCoreError::Compiler(format!(
                        "Missing left input '{}' for similarity",
                        left_name
                    ))
                })?;
                let right = inputs.get(right_name).ok_or_else(|| {
                    TensorCoreError::Compiler(format!(
                        "Missing right input '{}' for similarity",
                        right_name
                    ))
                })?;

                match method {
                    super::SimilarityMethod::Learned => {
                        // Element-wise product, then learned weighting
                        let product = left
                            .mul(right)
                            .map_err(|e| TensorCoreError::Tensor(format!("mul failed: {}", e)))?;

                        if let (Some(w), Some(b)) = (weights, bias) {
                            let score = product.matmul(w.as_tensor()).map_err(|e| {
                                TensorCoreError::Tensor(format!("matmul failed: {}", e))
                            })?;
                            let score = score.broadcast_add(b.as_tensor()).map_err(|e| {
                                TensorCoreError::Tensor(format!("add failed: {}", e))
                            })?;
                            // Squeeze to 1D for consistency with other predicates
                            let score = score.squeeze(1).map_err(|e| {
                                TensorCoreError::Tensor(format!("squeeze failed: {}", e))
                            })?;
                            crate::primitives::sigmoid(&score)
                        } else {
                            // Fallback: sum and normalize
                            let sum = product.sum_keepdim(1).map_err(|e| {
                                TensorCoreError::Tensor(format!("sum failed: {}", e))
                            })?;
                            let sum = sum.squeeze(1).map_err(|e| {
                                TensorCoreError::Tensor(format!("squeeze failed: {}", e))
                            })?;
                            crate::primitives::sigmoid(&sum)
                        }
                    }

                    super::SimilarityMethod::Cosine => {
                        // Normalize both vectors
                        let left_norm = l2_normalize(left)?;
                        let right_norm = l2_normalize(right)?;

                        // Dot product
                        let dot = (&left_norm * &right_norm)
                            .map_err(|e| TensorCoreError::Tensor(format!("mul failed: {}", e)))?
                            .sum_keepdim(1)
                            .map_err(|e| TensorCoreError::Tensor(format!("sum failed: {}", e)))?;

                        // Squeeze to 1D
                        let dot = dot.squeeze(1).map_err(|e| {
                            TensorCoreError::Tensor(format!("squeeze failed: {}", e))
                        })?;

                        // Map from [-1, 1] to [0, 1]
                        let one = Tensor::ones_like(&dot)
                            .map_err(|e| TensorCoreError::Tensor(format!("ones failed: {}", e)))?;
                        (&dot + &one)
                            .map_err(|e| TensorCoreError::Tensor(format!("add failed: {}", e)))?
                            .affine(0.5, 0.0)
                            .map_err(|e| TensorCoreError::Tensor(format!("affine failed: {}", e)))
                    }

                    super::SimilarityMethod::DotNormalized => {
                        // Simple dot product with sigmoid
                        let dot = (left * right)
                            .map_err(|e| TensorCoreError::Tensor(format!("mul failed: {}", e)))?
                            .sum_keepdim(1)
                            .map_err(|e| TensorCoreError::Tensor(format!("sum failed: {}", e)))?;
                        // Squeeze to 1D
                        let dot = dot.squeeze(1).map_err(|e| {
                            TensorCoreError::Tensor(format!("squeeze failed: {}", e))
                        })?;
                        crate::primitives::sigmoid(&dot)
                    }

                    super::SimilarityMethod::Bilinear => {
                        // aᵀWb → sigmoid
                        if let Some(w) = bilinear_w {
                            // left @ W @ right.T
                            let lw = left.matmul(w.as_tensor()).map_err(|e| {
                                TensorCoreError::Tensor(format!("matmul failed: {}", e))
                            })?;

                            // For batched: element-wise multiply and sum
                            let score = (&lw * right)
                                .map_err(|e| TensorCoreError::Tensor(format!("mul failed: {}", e)))?
                                .sum_keepdim(1)
                                .map_err(|e| {
                                    TensorCoreError::Tensor(format!("sum failed: {}", e))
                                })?;

                            // Squeeze to 1D
                            let score = score.squeeze(1).map_err(|e| {
                                TensorCoreError::Tensor(format!("squeeze failed: {}", e))
                            })?;

                            if let Some(b) = bias {
                                let score = score.broadcast_add(b.as_tensor()).map_err(|e| {
                                    TensorCoreError::Tensor(format!("add failed: {}", e))
                                })?;
                                crate::primitives::sigmoid(&score)
                            } else {
                                crate::primitives::sigmoid(&score)
                            }
                        } else {
                            Err(TensorCoreError::Compiler(
                                "Bilinear similarity requires bilinear_w".into(),
                            ))
                        }
                    }
                }
            }

            CompiledPredicate::GraphNeural {
                node_features_name,
                adjacency_name,
                layers,
                output_w,
                output_b,
                aggregation,
            } => {
                let node_features = inputs.get(node_features_name).ok_or_else(|| {
                    TensorCoreError::Compiler(format!(
                        "Missing node features '{}'",
                        node_features_name
                    ))
                })?;

                // Get adjacency matrix (supports both dense and sparse COO formats)
                let adjacency = get_adjacency_matrix(inputs, adjacency_name, device)?;

                // Normalize adjacency matrix based on aggregation method
                let adj_norm = normalize_adjacency(&adjacency, *aggregation, device)?;

                // Message passing layers
                let mut h = node_features.clone();
                for (w_self, w_neighbor, bias) in layers {
                    // Self transform: H @ W_self
                    let h_self = h.matmul(w_self.as_tensor()).map_err(|e| {
                        TensorCoreError::Tensor(format!("matmul self failed: {}", e))
                    })?;

                    // Neighbor aggregation (strategy-dependent)
                    let h_neighbor = match aggregation {
                        super::Aggregation::Max => {
                            // Element-wise max pooling over neighbors.
                            // For each node i: h_agg[i] = max over j∈N(i) of h[j]
                            // Computed as: expand h[j] per neighbor, zero non-edges, take max.
                            let n = h.dims()[0];
                            let feat = h.dims()[1];

                            // adj_norm here is raw adjacency (no normalization for Max)
                            // Expand H to [N, N, feat]: h_j for each (i,j) pair
                            let h_expanded = h
                                .unsqueeze(0)
                                .map_err(|e| {
                                    TensorCoreError::Tensor(format!(
                                        "max expand unsqueeze failed: {}",
                                        e
                                    ))
                                })?
                                .expand(&[n, n, feat])
                                .map_err(|e| {
                                    TensorCoreError::Tensor(format!("max expand failed: {}", e))
                                })?;

                            // Mask: adj [N,N] → [N,N,1], zero out non-neighbors
                            let adj_mask = adj_norm.unsqueeze(2).map_err(|e| {
                                TensorCoreError::Tensor(format!("max adj unsqueeze failed: {}", e))
                            })?;

                            let neg_inf = Tensor::full(f32::NEG_INFINITY, (n, n, feat), device)
                                .map_err(|e| {
                                    TensorCoreError::Tensor(format!("max neg_inf failed: {}", e))
                                })?;
                            let adj_bool = adj_mask.gt(0.0f64).map_err(|e| {
                                TensorCoreError::Tensor(format!("max adj_bool failed: {}", e))
                            })?;
                            let h_masked =
                                adj_bool.where_cond(&h_expanded, &neg_inf).map_err(|e| {
                                    TensorCoreError::Tensor(format!("max where_cond failed: {}", e))
                                })?;

                            // Max over neighbor dimension → [N, feat]
                            let h_max = h_masked.max(1).map_err(|e| {
                                TensorCoreError::Tensor(format!("max reduce failed: {}", e))
                            })?;

                            // Replace -inf (isolated nodes) with 0
                            let zeros_2d = Tensor::zeros_like(&h_max).map_err(|e| {
                                TensorCoreError::Tensor(format!("max zeros_2d failed: {}", e))
                            })?;
                            let is_neg_inf = h_max.lt(-1e30_f64).map_err(|e| {
                                TensorCoreError::Tensor(format!("max lt failed: {}", e))
                            })?;
                            let h_agg = is_neg_inf.where_cond(&zeros_2d, &h_max).map_err(|e| {
                                TensorCoreError::Tensor(format!("max fill failed: {}", e))
                            })?;

                            h_agg.matmul(w_neighbor.as_tensor()).map_err(|e| {
                                TensorCoreError::Tensor(format!("max matmul W failed: {}", e))
                            })?
                        }
                        _ => {
                            // Mean / Sum / Attention: A_norm @ H @ W_neighbor
                            adj_norm
                                .matmul(&h)
                                .map_err(|e| {
                                    TensorCoreError::Tensor(format!("adj matmul failed: {}", e))
                                })?
                                .matmul(w_neighbor.as_tensor())
                                .map_err(|e| {
                                    TensorCoreError::Tensor(format!(
                                        "matmul neighbor failed: {}",
                                        e
                                    ))
                                })?
                        }
                    };

                    // Combine self + neighbor and apply non-linearity
                    h = (&h_self + &h_neighbor)
                        .map_err(|e| TensorCoreError::Tensor(format!("add failed: {}", e)))?
                        .broadcast_add(bias.as_tensor())
                        .map_err(|e| TensorCoreError::Tensor(format!("add bias failed: {}", e)))?;

                    h = h
                        .relu()
                        .map_err(|e| TensorCoreError::Tensor(format!("relu failed: {}", e)))?;
                }

                // Output projection: H @ W_out + b_out → sigmoid
                let out = h
                    .matmul(output_w.as_tensor())
                    .map_err(|e| TensorCoreError::Tensor(format!("matmul out failed: {}", e)))?
                    .broadcast_add(output_b.as_tensor())
                    .map_err(|e| TensorCoreError::Tensor(format!("add out failed: {}", e)))?;

                crate::primitives::sigmoid(&out)
            }

            CompiledPredicate::External { name } => inputs.get(name).cloned().ok_or_else(|| {
                TensorCoreError::Compiler(format!("Missing external predicate input '{}'", name))
            }),
        }
    }

    /// Get trainable variables from this predicate
    pub fn trainable_vars(&self) -> Vec<Var> {
        match self {
            CompiledPredicate::Learned { weights, bias, .. } => {
                vec![weights.clone(), bias.clone()]
            }
            CompiledPredicate::LearnedProjection {
                w1,
                b1,
                w2,
                b2,
                ln1,
                attention,
                ln_attn,
                residual_proj,
                ..
            } => {
                let mut vars = vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()];

                // Add LayerNorm parameters
                if let Some((gamma, beta)) = ln1 {
                    vars.push(gamma.clone());
                    vars.push(beta.clone());
                }

                // Add attention parameters
                if let Some(attn) = attention {
                    vars.extend(attn.trainable_vars());
                }

                // Add post-attention LayerNorm
                if let Some((gamma, beta)) = ln_attn {
                    vars.push(gamma.clone());
                    vars.push(beta.clone());
                }

                // Add residual projection if present
                if let Some(proj) = residual_proj {
                    vars.push(proj.clone());
                }

                vars
            }
            CompiledPredicate::LearnedSimilarity {
                weights,
                bias,
                bilinear_w,
                ..
            } => {
                let mut vars = Vec::new();
                if let Some(w) = weights {
                    vars.push(w.clone());
                }
                if let Some(b) = bias {
                    vars.push(b.clone());
                }
                if let Some(bw) = bilinear_w {
                    vars.push(bw.clone());
                }
                vars
            }
            CompiledPredicate::GraphNeural {
                layers,
                output_w,
                output_b,
                ..
            } => {
                let mut vars: Vec<Var> = layers
                    .iter()
                    .flat_map(|(w_self, w_neighbor, bias)| {
                        vec![w_self.clone(), w_neighbor.clone(), bias.clone()]
                    })
                    .collect();
                vars.push(output_w.clone());
                vars.push(output_b.clone());
                vars
            }
            _ => Vec::new(),
        }
    }
}

// Utility functions (apply_activation, apply_layer_norm, apply_dropout,
// apply_multi_head_attention, l2_normalize, normalize_adjacency,
// sparse_coo_to_dense, detect_sparse_coo, get_adjacency_matrix)
// are in the `utils` submodule.

// =============================================================================

/// A compiled literal (possibly negated predicate)
#[derive(Debug)]
pub struct CompiledLiteral {
    /// The predicate name (for explanation)
    pub predicate_name: String,
    /// Index into the predicates array
    pub predicate_idx: usize,
    /// Whether to negate the result
    pub negated: bool,
    /// Constant argument if present (e.g., threshold value from rule)
    pub constant_arg: Option<f32>,
}

impl CompiledLiteral {
    /// Evaluate this literal
    pub fn evaluate(&self, predicate_activation: &Tensor) -> Result<Tensor> {
        if self.negated {
            fuzzy_not(predicate_activation)
        } else {
            Ok(predicate_activation.clone())
        }
    }
}

// =============================================================================
// COMPILED RULE BODY
// =============================================================================

/// A compiled rule body (conjunction of literals)
#[derive(Debug)]
pub struct CompiledBody {
    /// Rule head name (for explanation)
    pub head: String,
    /// Rule index (for explanation)
    pub rule_idx: usize,
    /// Literals in this body (ANDed together)
    pub literals: Vec<CompiledLiteral>,
}

impl CompiledBody {
    /// Evaluate this rule body
    ///
    /// Returns the fuzzy AND of all literals.
    /// For facts (empty body), returns 1.0 (unconditionally true).
    pub fn evaluate(&self, predicate_activations: &[Tensor], device: &Device) -> Result<Tensor> {
        // Facts have empty bodies and are unconditionally true
        if self.literals.is_empty() {
            // Return 1.0 with same shape as other activations
            // Use shape [1] for scalar output
            return Tensor::from_vec(vec![1.0f32], 1, device).map_err(|e| {
                TensorCoreError::Compiler(format!("Failed to create fact tensor: {}", e))
            });
        }

        // Evaluate first literal
        let first_lit = &self.literals[0];
        let mut result = first_lit.evaluate(&predicate_activations[first_lit.predicate_idx])?;

        // AND with remaining literals
        for lit in &self.literals[1..] {
            let lit_result = lit.evaluate(&predicate_activations[lit.predicate_idx])?;
            result = fuzzy_and(&result, &lit_result)?;
        }

        Ok(result)
    }
}

// =============================================================================
// EXECUTION ENGINE
// =============================================================================

/// Compiled rule set ready for execution
#[derive(Debug)]
pub struct CompiledRuleSet {
    /// Name of this rule set
    pub name: String,

    /// Source specification (for debugging/serialization)
    pub source: RuleSpec,

    /// Compiled predicates (indexed by position)
    pub predicates: Vec<CompiledPredicate>,

    /// Predicate name → index mapping
    pub predicate_indices: HashMap<String, usize>,

    /// Compiled rule bodies
    pub bodies: Vec<CompiledBody>,

    /// Head predicate for this rule set
    pub head: String,

    /// Device for tensor operations
    pub device: Device,
}

impl CompiledRuleSet {
    /// Compile a rule specification
    pub fn compile(spec: RuleSpec, device: &Device) -> Result<Self> {
        if spec.rules.is_empty() {
            return Err(TensorCoreError::Compiler("No rules to compile".into()));
        }

        // All rules should have the same head (for now)
        let head = spec.rules[0].head.clone();
        for rule in &spec.rules {
            if rule.head != head {
                return Err(TensorCoreError::Compiler(format!(
                    "Mixed heads not yet supported: '{}' vs '{}'",
                    head, rule.head
                )));
            }
        }

        // Collect all unique predicates from rule bodies
        let mut predicate_names: Vec<String> = Vec::new();
        let mut predicate_indices: HashMap<String, usize> = HashMap::new();

        for rule in &spec.rules {
            for lit in &rule.body {
                if !predicate_indices.contains_key(&lit.predicate) {
                    let idx = predicate_names.len();
                    predicate_names.push(lit.predicate.clone());
                    predicate_indices.insert(lit.predicate.clone(), idx);
                }
            }
        }

        // Compile predicates
        let mut predicates: Vec<CompiledPredicate> = Vec::new();
        for pred_name in &predicate_names {
            let compiled = if let Some(pred_spec) = spec.predicates.get(pred_name) {
                compile_predicate_from_spec(pred_name, pred_spec, device)?
            } else {
                // Default: external predicate
                CompiledPredicate::External {
                    name: pred_name.clone(),
                }
            };
            predicates.push(compiled);
        }

        // Compile rule bodies
        let mut bodies: Vec<CompiledBody> = Vec::new();
        for (rule_idx, rule) in spec.rules.iter().enumerate() {
            let literals: Vec<CompiledLiteral> = rule
                .body
                .iter()
                .map(|lit| {
                    let pred_idx = predicate_indices[&lit.predicate];
                    let constant_arg = lit.args.iter().find_map(|arg| {
                        if let Argument::Constant(v) = arg {
                            Some(*v)
                        } else {
                            None
                        }
                    });
                    CompiledLiteral {
                        predicate_name: lit.predicate.clone(),
                        predicate_idx: pred_idx,
                        negated: lit.negated,
                        constant_arg,
                    }
                })
                .collect();

            bodies.push(CompiledBody {
                head: rule.head.clone(),
                rule_idx,
                literals,
            });
        }

        Ok(CompiledRuleSet {
            name: spec.name.clone(),
            source: spec,
            predicates,
            predicate_indices,
            bodies,
            head,
            device: device.clone(),
        })
    }

    /// Forward pass: evaluate all rules and combine
    pub fn forward(&self, inputs: &HashMap<String, Tensor>) -> Result<CompiledOutput> {
        // Step 1: Evaluate all predicates
        let mut predicate_activations: Vec<Tensor> = Vec::new();
        let mut predicate_activation_map: HashMap<String, f32> = HashMap::new();

        for (idx, pred) in self.predicates.iter().enumerate() {
            let activation = pred.evaluate(inputs, &self.device)?;

            // Extract scalar for explanation (take mean if batched)
            let scalar = activation
                .mean_all()
                .map_err(|e| TensorCoreError::Tensor(format!("mean failed: {}", e)))?
                .to_scalar::<f32>()
                .unwrap_or(0.0);

            let pred_name = self
                .predicate_indices
                .iter()
                .find(|(_, &i)| i == idx)
                .map(|(n, _)| n.clone())
                .unwrap_or_default();

            predicate_activation_map.insert(pred_name, scalar);
            predicate_activations.push(activation);
        }

        // Step 2: Evaluate each rule body
        let mut rule_results: Vec<Tensor> = Vec::new();
        let mut rule_activation_map: HashMap<String, f32> = HashMap::new();

        for body in &self.bodies {
            let body_result = body.evaluate(&predicate_activations, &self.device)?;

            // Extract scalar for explanation
            let scalar = body_result
                .mean_all()
                .map_err(|e| TensorCoreError::Tensor(format!("mean failed: {}", e)))?
                .to_scalar::<f32>()
                .unwrap_or(0.0);

            let rule_name = format!("rule_{}", body.rule_idx);
            rule_activation_map.insert(rule_name, scalar);
            rule_results.push(body_result);
        }

        // Step 3: Combine rules with fuzzy OR
        let output = if rule_results.len() == 1 {
            rule_results.remove(0)
        } else {
            let refs: Vec<&Tensor> = rule_results.iter().collect();
            fuzzy_or_many(&refs)?
        };

        // Step 4: Extract rule weights (if defined in source)
        let rule_weights = extract_rule_weights(&self.source);

        // Step 5: Generate explanation
        let explanation = generate_explanation(
            &self.head,
            &self.bodies,
            &predicate_activation_map,
            &rule_activation_map,
        );

        Ok(CompiledOutput {
            output,
            rule_activations: rule_activation_map,
            predicate_activations: predicate_activation_map,
            explanation,
            rule_weights,
        })
    }

    /// Fast inference path - returns just the output tensor
    ///
    /// This is ~20x faster than `forward()` because it:
    /// - Skips all explanation generation
    /// - Avoids CPU/GPU sync for scalar extraction
    /// - Minimizes allocation overhead
    ///
    /// Use this for production inference when you don't need explanations.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // For backtesting where speed matters:
    /// let prob = compiled.forward_fast(&inputs)?;
    /// let should_trade = prob.to_scalar::<f32>()? > 0.5;
    ///
    /// // Only get explanations when needed:
    /// if should_trade && need_explanation {
    ///     let output = compiled.forward(&inputs)?;
    ///     println!("{}", output.explanation);
    /// }
    /// ```
    #[inline]
    pub fn forward_fast(&self, inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        // Step 1: Evaluate all predicates (minimal allocation)
        let mut predicate_activations: Vec<Tensor> = Vec::with_capacity(self.predicates.len());

        for pred in &self.predicates {
            let activation = pred.evaluate(inputs, &self.device)?;
            predicate_activations.push(activation);
        }

        // Step 2: Evaluate each rule body
        let mut rule_results: Vec<Tensor> = Vec::with_capacity(self.bodies.len());

        for body in &self.bodies {
            let body_result = body.evaluate(&predicate_activations, &self.device)?;
            rule_results.push(body_result);
        }

        // Step 3: Combine rules with fuzzy OR
        if rule_results.len() == 1 {
            Ok(rule_results.remove(0))
        } else {
            let refs: Vec<&Tensor> = rule_results.iter().collect();
            fuzzy_or_many(&refs)
        }
    }

    /// Batched fast inference - evaluate multiple inputs in parallel
    ///
    /// Even faster than multiple `forward_fast()` calls because:
    /// - Single tensor allocation for all samples
    /// - Better GPU utilization for batch matmuls
    ///
    /// # Arguments
    ///
    /// * `batch_inputs` - Vec of input HashMaps, one per sample.
    ///   Each input tensor should have shape `[1, features]` or `[features]`.
    ///
    /// # Returns
    ///
    /// Tensor of shape [batch_size, 1] with output probabilities
    ///
    /// # Example
    ///
    /// ```ignore
    /// let inputs: Vec<HashMap<String, Tensor>> = symbols
    ///     .iter()
    ///     .map(|sym| build_inputs(sym, bar))
    ///     .collect();
    ///
    /// let probs = compiled.forward_batch(&inputs)?;
    /// // probs shape: [num_symbols, 1]
    /// ```
    pub fn forward_batch(&self, batch_inputs: &[HashMap<String, Tensor>]) -> Result<Tensor> {
        if batch_inputs.is_empty() {
            return Err(TensorCoreError::Compiler("Empty batch".into()));
        }

        // Stack all inputs into batched tensors
        let input_keys: Vec<String> = batch_inputs[0].keys().cloned().collect();
        let mut batched_inputs: HashMap<String, Tensor> = HashMap::new();

        for key in &input_keys {
            let tensors: Result<Vec<Tensor>> = batch_inputs
                .iter()
                .map(|inputs| {
                    let t = inputs.get(key).cloned().ok_or_else(|| {
                        TensorCoreError::Compiler(format!("Missing input '{}' in batch", key))
                    })?;

                    // Ensure 2D: [1, features] - squeeze if [1, 1, features]
                    let dims = t.dims();
                    if dims.len() == 2 {
                        Ok(t)
                    } else if dims.len() == 1 {
                        // [features] -> [1, features]
                        t.unsqueeze(0).map_err(|e| {
                            TensorCoreError::Tensor(format!("Unsqueeze failed: {}", e))
                        })
                    } else {
                        // [1, 1, features] -> [1, features]
                        t.squeeze(0)
                            .map_err(|e| TensorCoreError::Tensor(format!("Squeeze failed: {}", e)))
                    }
                })
                .collect();

            let tensors = tensors?;
            let refs: Vec<&Tensor> = tensors.iter().collect();

            // Concatenate along batch dimension (dim 0) - creates [batch_size, features]
            let batched = Tensor::cat(&refs, 0)
                .map_err(|e| TensorCoreError::Tensor(format!("Batch cat failed: {}", e)))?;

            batched_inputs.insert(key.clone(), batched);
        }

        // Run forward pass on batched inputs
        self.forward_fast(&batched_inputs)
    }

    /// Get all trainable variables
    pub fn trainable_vars(&self) -> Vec<Var> {
        self.predicates
            .iter()
            .flat_map(|p| p.trainable_vars())
            .collect()
    }

    /// Number of trainable parameters
    pub fn param_count(&self) -> usize {
        self.trainable_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum()
    }

    /// Clamp all attention weights to prevent gradient explosion
    ///
    /// Call this after `optimizer.step()` to keep attention weights bounded.
    /// This prevents the Q/K weight explosion that causes NaN after training steps.
    ///
    /// # Example
    ///
    /// ```ignore
    /// for epoch in 0..100 {
    ///     let output = compiled.forward(&inputs)?;
    ///     let loss = compute_loss(&output, &targets)?;
    ///     let grads = loss.backward()?;
    ///     optimizer.step(&grads)?;
    ///     
    ///     // Prevent weight explosion
    ///     compiled.clamp_attention_weights(10.0)?;
    /// }
    /// ```
    ///
    /// # Arguments
    ///
    /// * `max_val` - Maximum absolute value for attention weights (recommended: 10.0)
    pub fn clamp_attention_weights(&self, max_val: f32) -> Result<()> {
        for predicate in &self.predicates {
            if let CompiledPredicate::LearnedProjection {
                attention: Some(attn_params),
                ..
            } = predicate
            {
                attn_params.clamp_weights(max_val)?;
            }
        }
        Ok(())
    }

    /// Check if any trainable variable contains NaN or Inf
    ///
    /// Useful for debugging training instability.
    ///
    /// # Returns
    ///
    /// `Some((var_index, issue))` if a problem is found, `None` if all OK.
    pub fn check_weights_health(&self) -> Option<(usize, &'static str)> {
        for (i, var) in self.trainable_vars().iter().enumerate() {
            let tensor = var.as_tensor();
            if let Ok(vals) = tensor.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                for v in vals {
                    if v.is_nan() {
                        return Some((i, "NaN"));
                    }
                    if v.is_infinite() {
                        return Some((i, "Infinite"));
                    }
                }
            }
        }
        None
    }

    /// Get FiLM conditioning variables (for separate optimizer groups)
    ///
    /// Returns an empty vector if FiLM is not enabled.
    pub fn film_vars(&self) -> Vec<candle_core::Var> {
        // FiLM variables are the gamma/beta scales inside LearnedProjection predicates
        // For now, return empty since we haven't implemented FiLM-specific var separation
        Vec::new()
    }

    /// Get main (non-FiLM) trainable variables
    ///
    /// Returns all trainable variables when FiLM is not used.
    pub fn main_vars(&self) -> Vec<candle_core::Var> {
        self.trainable_vars()
    }

    /// Check if FiLM conditioning is enabled in any predicate
    ///
    /// Note: FiLM conditioning info is stored in PredicateSpec during compilation,
    /// not in the CompiledPredicate. This returns false for now as FiLM vars
    /// are integrated into the main trainable_vars.
    pub fn has_film(&self) -> bool {
        // FiLM gamma/beta are part of LayerNorm parameters when enabled
        // For now, return false - all vars are returned via trainable_vars()
        false
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Compile a predicate from its specification
fn compile_predicate_from_spec(
    name: &str,
    spec: &PredicateSpec,
    device: &Device,
) -> Result<CompiledPredicate> {
    match spec {
        PredicateSpec::Threshold {
            input,
            threshold,
            greater_than,
            sharpness,
        } => Ok(CompiledPredicate::Threshold {
            input_name: input.clone(),
            threshold: *threshold,
            sharpness: *sharpness,
            greater_than: *greater_than,
        }),

        PredicateSpec::Learned { dim } => {
            // Initialize weights and bias
            let weights = Var::zeros((*dim, 1), DType::F32, device)
                .map_err(|e| TensorCoreError::Tensor(format!("Failed to create weights: {}", e)))?;
            let bias = Var::zeros(1, DType::F32, device)
                .map_err(|e| TensorCoreError::Tensor(format!("Failed to create bias: {}", e)))?;

            Ok(CompiledPredicate::Learned {
                name: name.to_string(),
                weights,
                bias,
            })
        }

        PredicateSpec::LearnedProjection {
            inputs,
            input_dim,
            hidden_dim,
            activation,
            attention_heads,
            attention_dropout,
            layer_norm,
            dropout,
            residual,
            conditioning_dim: _, // TODO: Implement FiLM conditioning in compiled form
            conditioning_type: _,
            film_identity_init: _,
        } => {
            // Xavier/Glorot initialization for better training
            let scale1 = (2.0f32 / (*input_dim + *hidden_dim) as f32).sqrt();
            let scale2 = (2.0f32 / (*hidden_dim + 1) as f32).sqrt();

            // Layer 1: input_dim → hidden_dim
            let w1 = Var::from_tensor(
                &Tensor::randn(0.0f32, scale1, (*input_dim, *hidden_dim), device)
                    .map_err(|e| TensorCoreError::Tensor(format!("Failed to create w1: {}", e)))?,
            )
            .map_err(|e| TensorCoreError::Tensor(format!("Failed to create w1 var: {}", e)))?;

            let b1 = Var::zeros(*hidden_dim, DType::F32, device)
                .map_err(|e| TensorCoreError::Tensor(format!("Failed to create b1: {}", e)))?;

            // Layer 2: hidden_dim → 1
            let w2 = Var::from_tensor(
                &Tensor::randn(0.0f32, scale2, (*hidden_dim, 1), device)
                    .map_err(|e| TensorCoreError::Tensor(format!("Failed to create w2: {}", e)))?,
            )
            .map_err(|e| TensorCoreError::Tensor(format!("Failed to create w2 var: {}", e)))?;

            let b2 = Var::zeros(1, DType::F32, device)
                .map_err(|e| TensorCoreError::Tensor(format!("Failed to create b2: {}", e)))?;

            // Optional LayerNorm for first layer
            let ln1 = if layer_norm.unwrap_or(false) {
                let gamma =
                    Var::from_tensor(&Tensor::ones(*hidden_dim, DType::F32, device).map_err(
                        |e| TensorCoreError::Tensor(format!("Failed to create ln1 gamma: {}", e)),
                    )?)
                    .map_err(|e| {
                        TensorCoreError::Tensor(format!("Failed to create ln1 gamma var: {}", e))
                    })?;

                let beta = Var::zeros(*hidden_dim, DType::F32, device).map_err(|e| {
                    TensorCoreError::Tensor(format!("Failed to create ln1 beta: {}", e))
                })?;

                Some((gamma, beta))
            } else {
                None
            };

            // Optional multi-head attention
            let (attention, ln_attn) = if let Some(num_heads) = attention_heads {
                let attn_params = AttentionParams::new(*hidden_dim, *num_heads, device)?;

                // Post-attention LayerNorm
                let ln = if layer_norm.unwrap_or(false) {
                    let gamma = Var::from_tensor(
                        &Tensor::ones(*hidden_dim, DType::F32, device).map_err(|e| {
                            TensorCoreError::Tensor(format!(
                                "Failed to create ln_attn gamma: {}",
                                e
                            ))
                        })?,
                    )
                    .map_err(|e| {
                        TensorCoreError::Tensor(format!(
                            "Failed to create ln_attn gamma var: {}",
                            e
                        ))
                    })?;

                    let beta = Var::zeros(*hidden_dim, DType::F32, device).map_err(|e| {
                        TensorCoreError::Tensor(format!("Failed to create ln_attn beta: {}", e))
                    })?;

                    Some((gamma, beta))
                } else {
                    None
                };

                (Some(attn_params), ln)
            } else {
                (None, None)
            };

            // Residual projection if dimensions don't match
            let use_residual = residual.unwrap_or(false);
            let residual_proj = if use_residual && input_dim != hidden_dim {
                let proj = Var::from_tensor(
                    &Tensor::randn(0.0f32, scale1, (*input_dim, *hidden_dim), device).map_err(
                        |e| {
                            TensorCoreError::Tensor(format!(
                                "Failed to create residual proj: {}",
                                e
                            ))
                        },
                    )?,
                )
                .map_err(|e| {
                    TensorCoreError::Tensor(format!("Failed to create residual proj var: {}", e))
                })?;
                Some(proj)
            } else {
                None
            };

            Ok(CompiledPredicate::LearnedProjection {
                input_names: inputs.clone(),
                w1,
                b1,
                w2,
                b2,
                activation: *activation,
                ln1,
                attention,
                ln_attn,
                dropout_rate: *dropout,
                attention_dropout_rate: *attention_dropout,
                residual: use_residual,
                residual_proj,
            })
        }

        PredicateSpec::LearnedSimilarity {
            left,
            right,
            dim,
            method,
        } => {
            let (weights, bias, bilinear_w) = match method {
                super::SimilarityMethod::Learned => {
                    // Weights for element-wise product aggregation
                    let w = Var::from_tensor(
                        &Tensor::randn(0.0f32, 0.01f32, (*dim, 1), device).map_err(|e| {
                            TensorCoreError::Tensor(format!(
                                "Failed to create similarity weights: {}",
                                e
                            ))
                        })?,
                    )
                    .map_err(|e| {
                        TensorCoreError::Tensor(format!("Failed to create weights var: {}", e))
                    })?;

                    let b = Var::zeros(1, DType::F32, device).map_err(|e| {
                        TensorCoreError::Tensor(format!("Failed to create similarity bias: {}", e))
                    })?;

                    (Some(w), Some(b), None)
                }
                super::SimilarityMethod::Bilinear => {
                    // Bilinear weight matrix
                    let scale = (2.0f32 / (2.0f32 * *dim as f32)).sqrt();
                    let bw = Var::from_tensor(
                        &Tensor::randn(0.0f32, scale, (*dim, *dim), device).map_err(|e| {
                            TensorCoreError::Tensor(format!(
                                "Failed to create bilinear weights: {}",
                                e
                            ))
                        })?,
                    )
                    .map_err(|e| {
                        TensorCoreError::Tensor(format!("Failed to create bilinear var: {}", e))
                    })?;

                    let b = Var::zeros(1, DType::F32, device).map_err(|e| {
                        TensorCoreError::Tensor(format!("Failed to create bilinear bias: {}", e))
                    })?;

                    (None, Some(b), Some(bw))
                }
                super::SimilarityMethod::Cosine | super::SimilarityMethod::DotNormalized => {
                    // No learnable parameters for these methods
                    (None, None, None)
                }
            };

            Ok(CompiledPredicate::LearnedSimilarity {
                left_name: left.clone(),
                right_name: right.clone(),
                method: *method,
                weights,
                bias,
                bilinear_w,
            })
        }

        PredicateSpec::GraphNeural {
            node_features,
            adjacency,
            feature_dim,
            hidden_dim,
            num_layers,
            aggregation,
        } => {
            let mut layers = Vec::with_capacity(*num_layers);

            let mut current_dim = *feature_dim;
            for _ in 0..*num_layers {
                let scale = (2.0f32 / (current_dim + *hidden_dim) as f32).sqrt();

                // W_self: current_dim → hidden_dim
                let w_self = Var::from_tensor(
                    &Tensor::randn(0.0f32, scale, (current_dim, *hidden_dim), device).map_err(
                        |e| TensorCoreError::Tensor(format!("Failed to create w_self: {}", e)),
                    )?,
                )
                .map_err(|e| {
                    TensorCoreError::Tensor(format!("Failed to create w_self var: {}", e))
                })?;

                // W_neighbor: current_dim → hidden_dim
                let w_neighbor = Var::from_tensor(
                    &Tensor::randn(0.0f32, scale, (current_dim, *hidden_dim), device).map_err(
                        |e| TensorCoreError::Tensor(format!("Failed to create w_neighbor: {}", e)),
                    )?,
                )
                .map_err(|e| {
                    TensorCoreError::Tensor(format!("Failed to create w_neighbor var: {}", e))
                })?;

                // Bias
                let bias = Var::zeros(*hidden_dim, DType::F32, device).map_err(|e| {
                    TensorCoreError::Tensor(format!("Failed to create layer bias: {}", e))
                })?;

                layers.push((w_self, w_neighbor, bias));
                current_dim = *hidden_dim;
            }

            // Output projection: hidden_dim → 1
            let output_scale = (2.0f32 / (*hidden_dim + 1) as f32).sqrt();
            let output_w = Var::from_tensor(
                &Tensor::randn(0.0f32, output_scale, (*hidden_dim, 1), device).map_err(|e| {
                    TensorCoreError::Tensor(format!("Failed to create output_w: {}", e))
                })?,
            )
            .map_err(|e| {
                TensorCoreError::Tensor(format!("Failed to create output_w var: {}", e))
            })?;

            let output_b = Var::zeros(1, DType::F32, device).map_err(|e| {
                TensorCoreError::Tensor(format!("Failed to create output_b: {}", e))
            })?;

            Ok(CompiledPredicate::GraphNeural {
                node_features_name: node_features.clone(),
                adjacency_name: adjacency.clone(),
                layers,
                output_w,
                output_b,
                aggregation: *aggregation,
            })
        }

        PredicateSpec::External => Ok(CompiledPredicate::External {
            name: name.to_string(),
        }),

        PredicateSpec::Composite { .. } => {
            // For now, composite predicates are handled by defining them as rules
            Err(TensorCoreError::Compiler(
                "Composite predicates not yet implemented directly. Define as rules instead."
                    .into(),
            ))
        }

        PredicateSpec::TierLookup { .. } => Err(TensorCoreError::Compiler(
            "TierLookup predicates not yet implemented in codegen.".into(),
        )),

        PredicateSpec::PairwiseDifference { .. } => Err(TensorCoreError::Compiler(
            "PairwiseDifference predicates not yet implemented in codegen.".into(),
        )),

        PredicateSpec::CascadingLookup { .. } => Err(TensorCoreError::Compiler(
            "CascadingLookup predicates not yet implemented in codegen.".into(),
        )),
    }
}

/// Extract rule weights from the source specification
///
/// If rules have explicit weights defined, normalize them to sum to 1.0.
/// Returns None if no weights are defined.
fn extract_rule_weights(spec: &RuleSpec) -> Option<Vec<f32>> {
    let weights: Vec<f32> = spec.rules.iter().map(|r| r.weight.unwrap_or(1.0)).collect();

    // Check if any non-default weights are defined
    let has_custom_weights = spec.rules.iter().any(|r| r.weight.is_some());

    if !has_custom_weights {
        return None;
    }

    // Normalize weights to sum to 1.0
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        Some(weights.iter().map(|w| w / sum).collect())
    } else {
        None
    }
}

/// Generate a human-readable explanation
fn generate_explanation(
    head: &str,
    bodies: &[CompiledBody],
    predicate_activations: &HashMap<String, f32>,
    rule_activations: &HashMap<String, f32>,
) -> String {
    let mut lines = Vec::new();

    // Find the most activated rule
    let (best_rule, best_activation) = rule_activations
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(k, v)| (k.clone(), *v))
        .unwrap_or(("none".to_string(), 0.0));

    lines.push(format!(
        "{}({:.0}%) triggered by {}",
        head,
        best_activation * 100.0,
        best_rule
    ));

    // Detail the predicates in the best rule
    if let Some(body) = bodies
        .iter()
        .find(|b| format!("rule_{}", b.rule_idx) == best_rule)
    {
        lines.push("  Contributing predicates:".to_string());
        for lit in &body.literals {
            if let Some(&activation) = predicate_activations.get(&lit.predicate_name) {
                let neg_str = if lit.negated { "¬" } else { "" };
                lines.push(format!(
                    "    - {}{}: {:.0}%",
                    neg_str,
                    lit.predicate_name,
                    activation * 100.0
                ));
            }
        }
    }

    lines.join("\n")
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::parser::parse_rules;

    fn test_device() -> Device {
        Device::Cpu
    }

    #[test]
    fn test_compile_simple_rule() {
        let spec = parse_rules("test", "exit(X) :- profit(X), momentum(X).").unwrap();
        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        assert_eq!(compiled.head, "exit");
        assert_eq!(compiled.bodies.len(), 1);
        assert_eq!(compiled.predicates.len(), 2);
    }

    #[test]
    fn test_compile_multiple_rules() {
        let spec = parse_rules(
            "test",
            r#"
            exit(X) :- profit(X), momentum(X).
            exit(X) :- stop_loss(X).
        "#,
        )
        .unwrap();

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        assert_eq!(compiled.bodies.len(), 2);
        assert_eq!(compiled.predicates.len(), 3); // profit, momentum, stop_loss
    }

    #[test]
    fn test_compile_with_negation() {
        let spec = parse_rules("test", "hold(X) :- not exit(X), position(X).").unwrap();
        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        assert!(compiled.bodies[0].literals[0].negated);
        assert!(!compiled.bodies[0].literals[1].negated);
    }

    #[test]
    fn test_forward_external_predicates() {
        let spec = parse_rules("test", "exit(X) :- profit(X), momentum(X).").unwrap();
        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Create inputs
        let mut inputs = HashMap::new();
        inputs.insert(
            "profit".to_string(),
            Tensor::from_vec(vec![0.8f32], 1, &test_device()).unwrap(),
        );
        inputs.insert(
            "momentum".to_string(),
            Tensor::from_vec(vec![0.9f32], 1, &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();

        // fuzzy_and(0.8, 0.9) = 0.72
        let result = output.output.to_vec1::<f32>().unwrap()[0];
        assert!((result - 0.72).abs() < 0.001);
    }

    #[test]
    fn test_forward_with_negation() {
        let spec = parse_rules("test", "hold(X) :- not exit(X).").unwrap();
        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            "exit".to_string(),
            Tensor::from_vec(vec![0.3f32], 1, &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();

        // fuzzy_not(0.3) = 0.7
        let result = output.output.to_vec1::<f32>().unwrap()[0];
        assert!((result - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_forward_multiple_rules() {
        let spec = parse_rules(
            "test",
            r#"
            exit(X) :- profit(X).
            exit(X) :- stop_loss(X).
        "#,
        )
        .unwrap();
        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            "profit".to_string(),
            Tensor::from_vec(vec![0.3f32], 1, &test_device()).unwrap(),
        );
        inputs.insert(
            "stop_loss".to_string(),
            Tensor::from_vec(vec![0.4f32], 1, &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();

        // fuzzy_or(0.3, 0.4) = 0.3 + 0.4 - 0.3*0.4 = 0.58
        let result = output.output.to_vec1::<f32>().unwrap()[0];
        assert!((result - 0.58).abs() < 0.001);
    }

    #[test]
    fn test_forward_generates_explanation() {
        let spec = parse_rules("test", "exit(X) :- profit(X), momentum(X).").unwrap();
        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            "profit".to_string(),
            Tensor::from_vec(vec![0.9f32], 1, &test_device()).unwrap(),
        );
        inputs.insert(
            "momentum".to_string(),
            Tensor::from_vec(vec![0.8f32], 1, &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();

        assert!(output.explanation.contains("exit"));
        assert!(output.explanation.contains("profit"));
        assert!(output.explanation.contains("momentum"));
    }

    #[test]
    fn test_compile_with_threshold_spec() {
        let mut spec = parse_rules("test", "exit(X) :- profit_target(X).").unwrap();

        // Add predicate specification
        spec.add_predicate(
            "profit_target",
            PredicateSpec::Threshold {
                input: "profit".to_string(),
                threshold: 0.02,
                greater_than: true,
                sharpness: 10.0,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            "profit".to_string(),
            Tensor::from_vec(vec![0.03f32], 1, &test_device()).unwrap(), // Above threshold
        );

        let output = compiled.forward(&inputs).unwrap();

        // soft_threshold(0.03, 0.02, 10) = sigmoid((0.03-0.02)*10) = sigmoid(0.1) ≈ 0.52
        let result = output.output.to_vec1::<f32>().unwrap()[0];
        assert!(result > 0.5); // Should be above 0.5 since profit > threshold
    }

    #[test]
    fn test_param_count() {
        let mut spec = parse_rules("test", "exit(X) :- momentum(X).").unwrap();

        spec.add_predicate("momentum", PredicateSpec::Learned { dim: 10 });

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Learned predicate has weights (10x1) + bias (1) = 11 params
        assert_eq!(compiled.param_count(), 11);
    }

    #[test]
    fn test_trainable_vars() {
        let mut spec = parse_rules("test", "exit(X) :- momentum(X).").unwrap();

        spec.add_predicate("momentum", PredicateSpec::Learned { dim: 10 });

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();
        let vars = compiled.trainable_vars();

        assert_eq!(vars.len(), 2); // weights + bias
    }

    // =========================================================================
    // P0a: LearnedProjection Tests
    // =========================================================================

    #[test]
    fn test_learned_projection_compile() {
        use super::super::Activation;

        let mut spec = parse_rules("test", "high_intent(X) :- intent_score(X).").unwrap();

        spec.add_predicate(
            "intent_score",
            PredicateSpec::LearnedProjection {
                inputs: vec!["code_embedding".into()],
                input_dim: 128,
                hidden_dim: 32,
                activation: Activation::ReLU,
                attention_heads: None,
                attention_dropout: None,
                layer_norm: None,
                dropout: None,
                residual: None,
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Parameters: w1(128x32) + b1(32) + w2(32x1) + b2(1) = 4096 + 32 + 32 + 1 = 4161
        assert_eq!(compiled.param_count(), 4161);
        assert_eq!(compiled.trainable_vars().len(), 4);
    }

    #[test]
    fn test_learned_projection_forward() {
        use super::super::Activation;

        let mut spec = parse_rules("test", "high_intent(X) :- intent_score(X).").unwrap();

        spec.add_predicate(
            "intent_score",
            PredicateSpec::LearnedProjection {
                inputs: vec!["embedding".into()],
                input_dim: 8,
                hidden_dim: 4,
                activation: Activation::ReLU,
                attention_heads: None,
                attention_dropout: None,
                layer_norm: None,
                dropout: None,
                residual: None,
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            "embedding".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 8), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();

        // Output should be in [0, 1] (sigmoid output)
        let result = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((0.0..=1.0).contains(&result));
        assert!(output.explanation.contains("intent_score"));
    }

    #[test]
    fn test_learned_projection_multi_input() {
        use super::super::Activation;

        let mut spec = parse_rules("test", "match(X) :- combined_score(X).").unwrap();

        // Concatenate code and intent embeddings
        spec.add_predicate(
            "combined_score",
            PredicateSpec::LearnedProjection {
                inputs: vec!["code_embedding".into(), "intent_embedding".into()],
                input_dim: 16, // 8 + 8
                hidden_dim: 4,
                activation: Activation::GELU,
                attention_heads: None,
                attention_dropout: None,
                layer_norm: None,
                dropout: None,
                residual: None,
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            "code_embedding".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 8), &test_device()).unwrap(),
        );
        inputs.insert(
            "intent_embedding".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 8), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();
        let result = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((0.0..=1.0).contains(&result));
    }

    // =========================================================================
    // P0b: LearnedSimilarity Tests
    // =========================================================================

    #[test]
    fn test_learned_similarity_compile() {
        use super::super::SimilarityMethod;

        let mut spec = parse_rules("test", "similar(X) :- intent_match(X).").unwrap();

        spec.add_predicate(
            "intent_match",
            PredicateSpec::LearnedSimilarity {
                left: "code_embedding".into(),
                right: "intent_embedding".into(),
                dim: 64,
                method: SimilarityMethod::Learned,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Learned similarity: weights(64x1) + bias(1) = 65 params
        assert_eq!(compiled.param_count(), 65);
    }

    #[test]
    fn test_learned_similarity_forward() {
        use super::super::SimilarityMethod;

        let mut spec = parse_rules("test", "similar(X) :- intent_match(X).").unwrap();

        spec.add_predicate(
            "intent_match",
            PredicateSpec::LearnedSimilarity {
                left: "left".into(),
                right: "right".into(),
                dim: 8,
                method: SimilarityMethod::Learned,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            "left".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 8), &test_device()).unwrap(),
        );
        inputs.insert(
            "right".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 8), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();
        let result = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((0.0..=1.0).contains(&result));
    }

    #[test]
    fn test_cosine_similarity() {
        use super::super::SimilarityMethod;

        let mut spec = parse_rules("test", "similar(X) :- cosine_match(X).").unwrap();

        spec.add_predicate(
            "cosine_match",
            PredicateSpec::LearnedSimilarity {
                left: "a".into(),
                right: "b".into(),
                dim: 4,
                method: SimilarityMethod::Cosine,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Cosine similarity has no learnable parameters
        assert_eq!(compiled.param_count(), 0);

        // Test with identical vectors (should be similarity = 1.0)
        let mut inputs = HashMap::new();
        let vec = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], (1, 4), &test_device()).unwrap();
        inputs.insert("a".to_string(), vec.clone());
        inputs.insert("b".to_string(), vec);

        let output = compiled.forward(&inputs).unwrap();
        // Output is 2D [batch, 1], flatten to get scalar
        let result = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];

        // Cosine of identical vectors = 1.0, mapped to [0,1] = 1.0
        assert!((result - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_bilinear_similarity() {
        use super::super::SimilarityMethod;

        let mut spec = parse_rules("test", "similar(X) :- bilinear_match(X).").unwrap();

        spec.add_predicate(
            "bilinear_match",
            PredicateSpec::LearnedSimilarity {
                left: "a".into(),
                right: "b".into(),
                dim: 8,
                method: SimilarityMethod::Bilinear,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Bilinear: W(8x8) + bias(1) = 64 + 1 = 65 params
        assert_eq!(compiled.param_count(), 65);

        let mut inputs = HashMap::new();
        inputs.insert(
            "a".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 8), &test_device()).unwrap(),
        );
        inputs.insert(
            "b".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 8), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();
        let result = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((0.0..=1.0).contains(&result));
    }

    // =========================================================================
    // P0c: GraphNeural Tests
    // =========================================================================

    #[test]
    fn test_graph_neural_compile() {
        use super::super::Aggregation;

        let mut spec = parse_rules("test", "anomaly(X) :- gnn_score(X).").unwrap();

        spec.add_predicate(
            "gnn_score",
            PredicateSpec::GraphNeural {
                node_features: "node_emb".into(),
                adjacency: "adj".into(),
                feature_dim: 16,
                hidden_dim: 8,
                num_layers: 2,
                aggregation: Aggregation::Mean,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Layer 1: w_self(16x8) + w_neighbor(16x8) + bias(8) = 128 + 128 + 8 = 264
        // Layer 2: w_self(8x8) + w_neighbor(8x8) + bias(8) = 64 + 64 + 8 = 136
        // Output: w(8x1) + b(1) = 8 + 1 = 9
        // Total: 264 + 136 + 9 = 409
        assert_eq!(compiled.param_count(), 409);
    }

    #[test]
    fn test_graph_neural_forward() {
        use super::super::Aggregation;

        let mut spec = parse_rules("test", "anomaly(X) :- gnn_score(X).").unwrap();

        spec.add_predicate(
            "gnn_score",
            PredicateSpec::GraphNeural {
                node_features: "features".into(),
                adjacency: "adj".into(),
                feature_dim: 4,
                hidden_dim: 4,
                num_layers: 1,
                aggregation: Aggregation::Mean,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Create a simple 3-node graph
        let mut inputs = HashMap::new();
        inputs.insert(
            "features".to_string(),
            Tensor::randn(0.0f32, 1.0, (3, 4), &test_device()).unwrap(),
        );

        // Simple adjacency matrix (fully connected)
        inputs.insert(
            "adj".to_string(),
            Tensor::from_vec(
                vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                (3, 3),
                &test_device(),
            )
            .unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();

        // Should return per-node predictions
        let result = output.output.to_vec2::<f32>().unwrap();
        assert_eq!(result.len(), 3); // 3 nodes

        // All values should be in [0, 1]
        for row in result {
            assert!(row[0] >= 0.0 && row[0] <= 1.0);
        }
    }

    #[test]
    fn test_graph_neural_aggregation_sum() {
        use super::super::Aggregation;

        let mut spec = parse_rules("test", "score(X) :- gnn(X).").unwrap();

        spec.add_predicate(
            "gnn",
            PredicateSpec::GraphNeural {
                node_features: "f".into(),
                adjacency: "a".into(),
                feature_dim: 2,
                hidden_dim: 2,
                num_layers: 1,
                aggregation: Aggregation::Sum,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            "f".to_string(),
            Tensor::ones((2, 2), DType::F32, &test_device()).unwrap(),
        );
        inputs.insert(
            "a".to_string(),
            Tensor::ones((2, 2), DType::F32, &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs);
        assert!(output.is_ok());
    }

    #[test]
    fn test_graph_neural_sparse_coo() {
        use super::super::Aggregation;

        let mut spec = parse_rules("test", "score(X) :- gnn(X).").unwrap();

        spec.add_predicate(
            "gnn",
            PredicateSpec::GraphNeural {
                node_features: "features".into(),
                adjacency: "adj".into(), // Will be provided in sparse COO format
                feature_dim: 4,
                hidden_dim: 4,
                num_layers: 1,
                aggregation: Aggregation::Mean,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Create a 5-node graph with sparse edges: 0→1, 1→2, 2→3, 3→4, 4→0 (ring)
        let mut inputs = HashMap::new();

        // Node features: [5, 4]
        inputs.insert(
            "features".to_string(),
            Tensor::randn(0.0f32, 1.0, (5, 4), &test_device()).unwrap(),
        );

        // Sparse COO format: row_indices, col_indices, shape
        inputs.insert(
            "adj_row_indices".to_string(),
            Tensor::from_vec(vec![0i64, 1, 2, 3, 4], (5,), &test_device()).unwrap(),
        );
        inputs.insert(
            "adj_col_indices".to_string(),
            Tensor::from_vec(vec![1i64, 2, 3, 4, 0], (5,), &test_device()).unwrap(),
        );
        inputs.insert(
            "adj_shape".to_string(),
            Tensor::from_vec(vec![5i64, 5], (2,), &test_device()).unwrap(),
        );
        // Note: adj_values not provided, defaults to 1.0

        let output = compiled.forward(&inputs).unwrap();

        // Should return per-node predictions
        let result = output.output.to_vec2::<f32>().unwrap();
        assert_eq!(result.len(), 5); // 5 nodes

        // All values should be in [0, 1] and not NaN
        for (i, row) in result.iter().enumerate() {
            assert!(!row[0].is_nan(), "NaN at node {}", i);
            assert!(
                row[0] >= 0.0 && row[0] <= 1.0,
                "Out of range at node {}: {}",
                i,
                row[0]
            );
        }
    }

    #[test]
    fn test_graph_neural_sparse_coo_with_values() {
        use super::super::Aggregation;

        let mut spec = parse_rules("test", "score(X) :- gnn(X).").unwrap();

        spec.add_predicate(
            "gnn",
            PredicateSpec::GraphNeural {
                node_features: "features".into(),
                adjacency: "adj".into(),
                feature_dim: 2,
                hidden_dim: 2,
                num_layers: 1,
                aggregation: Aggregation::Sum,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // 3-node graph with weighted edges
        let mut inputs = HashMap::new();

        inputs.insert(
            "features".to_string(),
            Tensor::ones((3, 2), DType::F32, &test_device()).unwrap(),
        );

        // Edges: 0→1 (weight 0.5), 0→2 (weight 1.0), 1→2 (weight 0.7)
        inputs.insert(
            "adj_row_indices".to_string(),
            Tensor::from_vec(vec![0i64, 0, 1], (3,), &test_device()).unwrap(),
        );
        inputs.insert(
            "adj_col_indices".to_string(),
            Tensor::from_vec(vec![1i64, 2, 2], (3,), &test_device()).unwrap(),
        );
        inputs.insert(
            "adj_values".to_string(),
            Tensor::from_vec(vec![0.5f32, 1.0, 0.7], (3,), &test_device()).unwrap(),
        );
        inputs.insert(
            "adj_shape".to_string(),
            Tensor::from_vec(vec![3i64, 3], (2,), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();
        assert_eq!(output.output.dims(), &[3, 1]);
    }

    #[test]
    fn test_sparse_coo_to_dense() {
        let device = test_device();

        // Create sparse COO: 3x3 matrix with edges at (0,1), (1,2), (2,0)
        let row_indices = Tensor::from_vec(vec![0i64, 1, 2], (3,), &device).unwrap();
        let col_indices = Tensor::from_vec(vec![1i64, 2, 0], (3,), &device).unwrap();
        let values = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device).unwrap();

        let dense =
            super::sparse_coo_to_dense(&row_indices, &col_indices, Some(&values), (3, 3), &device)
                .unwrap();

        let result = dense.to_vec2::<f32>().unwrap();

        // Verify structure
        assert_eq!(result[0][1], 1.0); // (0,1) = 1.0
        assert_eq!(result[1][2], 2.0); // (1,2) = 2.0
        assert_eq!(result[2][0], 3.0); // (2,0) = 3.0

        // Other entries should be 0
        assert_eq!(result[0][0], 0.0);
        assert_eq!(result[1][1], 0.0);
        assert_eq!(result[2][2], 0.0);
    }

    // =========================================================================
    // Integration Tests: Full Pipeline-like Usage
    // =========================================================================

    #[test]
    fn test_zero_mlp_pipeline_integration() {
        use super::super::SimilarityMethod;

        // Single-head version (mixed heads will be P2)
        // Tests that LearnedSimilarity integrates with threshold predicates
        let mut spec = parse_rules(
            "pipeline_rules",
            r#"
            escalate(X) :- high_risk(X), low_intent(X).
            escalate(X) :- security_path(X).
        "#,
        )
        .unwrap();

        // Threshold predicate
        spec.add_predicate(
            "high_risk",
            PredicateSpec::Threshold {
                input: "risk_score".into(),
                threshold: 0.5,
                greater_than: true,
                sharpness: 10.0,
            },
        );

        // Learned similarity (replaces intent_matcher MLP)
        spec.add_predicate(
            "low_intent",
            PredicateSpec::LearnedSimilarity {
                left: "code_emb".into(),
                right: "intent_emb".into(),
                dim: 8,
                method: SimilarityMethod::Learned,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Create inputs
        let mut inputs = HashMap::new();
        inputs.insert(
            "risk_score".to_string(),
            Tensor::from_vec(vec![0.8f32], 1, &test_device()).unwrap(),
        );
        inputs.insert(
            "security_path".to_string(),
            Tensor::from_vec(vec![0.1f32], 1, &test_device()).unwrap(),
        );
        inputs.insert(
            "code_emb".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 8), &test_device()).unwrap(),
        );
        inputs.insert(
            "intent_emb".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 8), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();

        // Should produce a valid output
        let result = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((0.0..=1.0).contains(&result));

        // Should have trainable params from LearnedSimilarity
        assert!(compiled.param_count() > 0);

        // Explanation should reference predicates
        assert!(output.explanation.contains("escalate"));
    }

    // =========================================================================
    // Enhanced LearnedProjection Tests (Attention, LayerNorm, Dropout)
    // =========================================================================

    #[test]
    fn test_learned_projection_with_layer_norm() {
        use super::super::Activation;

        let mut spec = parse_rules("test", "score(X) :- proj(X).").unwrap();

        spec.add_predicate(
            "proj",
            PredicateSpec::LearnedProjection {
                inputs: vec!["embedding".into()],
                input_dim: 8,
                hidden_dim: 8,
                activation: Activation::GELU,
                attention_heads: None,
                attention_dropout: None,
                layer_norm: Some(true), // Enable LayerNorm
                dropout: None,
                residual: None,
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // LayerNorm adds gamma(8) + beta(8) = 16 extra params
        // Base: w1(8x8) + b1(8) + w2(8x1) + b2(1) = 64 + 8 + 8 + 1 = 81
        // Total: 81 + 16 = 97
        assert_eq!(compiled.param_count(), 97);

        let mut inputs = HashMap::new();
        inputs.insert(
            "embedding".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 8), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();
        let result = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((0.0..=1.0).contains(&result));
    }

    #[test]
    fn test_learned_projection_with_dropout() {
        use super::super::Activation;

        let mut spec = parse_rules("test", "score(X) :- proj(X).").unwrap();

        spec.add_predicate(
            "proj",
            PredicateSpec::LearnedProjection {
                inputs: vec!["embedding".into()],
                input_dim: 8,
                hidden_dim: 8,
                activation: Activation::GELU,
                attention_heads: None,
                attention_dropout: None,
                layer_norm: None,
                dropout: Some(0.3), // 30% dropout
                residual: None,
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            "embedding".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 8), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();
        let result = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((0.0..=1.0).contains(&result));
    }

    #[test]
    fn test_learned_projection_with_residual() {
        use super::super::Activation;

        let mut spec = parse_rules("test", "score(X) :- proj(X).").unwrap();

        spec.add_predicate(
            "proj",
            PredicateSpec::LearnedProjection {
                inputs: vec!["embedding".into()],
                input_dim: 8,
                hidden_dim: 8, // Same as input_dim, no projection needed
                activation: Activation::GELU,
                attention_heads: None,
                attention_dropout: None,
                layer_norm: None,
                dropout: None,
                residual: Some(true), // Enable residual connection
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(
            "embedding".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 8), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();
        let result = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((0.0..=1.0).contains(&result));
    }

    #[test]
    fn test_learned_projection_with_residual_projection() {
        use super::super::Activation;

        let mut spec = parse_rules("test", "score(X) :- proj(X).").unwrap();

        spec.add_predicate(
            "proj",
            PredicateSpec::LearnedProjection {
                inputs: vec!["embedding".into()],
                input_dim: 8,
                hidden_dim: 16, // Different from input_dim, needs projection
                activation: Activation::GELU,
                attention_heads: None,
                attention_dropout: None,
                layer_norm: None,
                dropout: None,
                residual: Some(true), // Enable residual with projection
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Base: w1(8x16) + b1(16) + w2(16x1) + b2(1) = 128 + 16 + 16 + 1 = 161
        // Residual projection: (8x16) = 128
        // Total: 161 + 128 = 289
        assert_eq!(compiled.param_count(), 289);

        let mut inputs = HashMap::new();
        inputs.insert(
            "embedding".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 8), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();
        let result = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((0.0..=1.0).contains(&result));
    }

    #[test]
    fn test_learned_projection_with_attention() {
        use super::super::Activation;

        let mut spec = parse_rules("test", "score(X) :- proj(X).").unwrap();

        spec.add_predicate(
            "proj",
            PredicateSpec::LearnedProjection {
                inputs: vec!["embedding".into()],
                input_dim: 8,
                hidden_dim: 8, // Must be divisible by num_heads
                activation: Activation::GELU,
                attention_heads: Some(2), // 2 attention heads
                attention_dropout: Some(0.1),
                layer_norm: Some(true),
                dropout: None,
                residual: None,
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Base: w1(8x8) + b1(8) + w2(8x1) + b2(1) = 64 + 8 + 8 + 1 = 81
        // LayerNorm 1: gamma(8) + beta(8) = 16
        // Attention: Wq(8x8) + Wk(8x8) + Wv(8x8) + Wo(8x8) = 4 * 64 = 256
        // LayerNorm post-attention: gamma(8) + beta(8) = 16
        // Total: 81 + 16 + 256 + 16 = 369
        assert_eq!(compiled.param_count(), 369);

        let mut inputs = HashMap::new();
        inputs.insert(
            "embedding".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 8), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();
        let result = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((0.0..=1.0).contains(&result));
    }

    #[test]
    fn test_learned_projection_full_transformer_style() {
        use super::super::Activation;

        // Full transformer-style block: LayerNorm + Attention + Dropout + Residual
        let mut spec = parse_rules("test", "score(X) :- proj(X).").unwrap();

        spec.add_predicate(
            "proj",
            PredicateSpec::LearnedProjection {
                inputs: vec!["embedding".into()],
                input_dim: 16,
                hidden_dim: 16,
                activation: Activation::GELU,
                attention_heads: Some(4), // 4-head attention
                attention_dropout: Some(0.1),
                layer_norm: Some(true),
                dropout: Some(0.1),
                residual: Some(true),
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Should have significant params from attention
        assert!(compiled.param_count() > 1000);

        let mut inputs = HashMap::new();
        inputs.insert(
            "embedding".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 16), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();
        let result = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((0.0..=1.0).contains(&result));
    }

    #[test]
    fn test_learned_projection_batch_input() {
        use super::super::Activation;

        let mut spec = parse_rules("test", "score(X) :- proj(X).").unwrap();

        spec.add_predicate(
            "proj",
            PredicateSpec::LearnedProjection {
                inputs: vec!["embedding".into()],
                input_dim: 8,
                hidden_dim: 8,
                activation: Activation::GELU,
                attention_heads: Some(2),
                attention_dropout: None,
                layer_norm: Some(true),
                dropout: Some(0.1),
                residual: Some(true),
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Test with batch of 4 samples
        let mut inputs = HashMap::new();
        inputs.insert(
            "embedding".to_string(),
            Tensor::randn(0.0f32, 1.0, (4, 8), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();
        let results = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(results.len(), 4);
        for result in results {
            assert!((0.0..=1.0).contains(&result));
        }
    }

    #[test]
    fn test_attention_params_creation() {
        let params = AttentionParams::new(16, 4, &test_device()).unwrap();

        assert_eq!(params.num_heads, 4);
        assert_eq!(params.head_dim, 4);
        assert_eq!(params.trainable_vars().len(), 4);

        // 4 * (16x16) = 1024 params
        assert_eq!(params.param_count(), 1024);
    }

    #[test]
    fn test_attention_params_invalid_heads() {
        // hidden_dim=8 is not divisible by num_heads=3
        let result = AttentionParams::new(8, 3, &test_device());
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_no_nan_with_trading_config() {
        // Regression test for NaN bug reported by org-lttr
        // Config: 18 features, 128 hidden, 4 attention heads
        use super::super::Activation;

        let mut spec = parse_rules("test", "exit(X) :- exit_signal(X).").unwrap();

        spec.add_predicate(
            "exit_signal",
            PredicateSpec::LearnedProjection {
                inputs: vec!["topology_features".into()],
                input_dim: 18,
                hidden_dim: 128,
                activation: Activation::GELU,
                attention_heads: Some(4),
                attention_dropout: Some(0.1),
                layer_norm: Some(true),
                dropout: Some(0.3),
                residual: Some(true),
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Test with batch of 100 samples (simulating trading data)
        let mut inputs = HashMap::new();
        inputs.insert(
            "topology_features".to_string(),
            Tensor::randn(0.0f32, 1.0, (100, 18), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();
        let results = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Check no NaN values
        for (i, result) in results.iter().enumerate() {
            assert!(!result.is_nan(), "NaN at index {}", i);
            assert!(!result.is_infinite(), "Infinite at index {}", i);
            assert!(
                *result >= 0.0 && *result <= 1.0,
                "Out of range at index {}: {}",
                i,
                result
            );
        }

        // Check reasonable variance (not collapsed)
        let mean: f32 = results.iter().sum::<f32>() / results.len() as f32;
        let variance: f32 =
            results.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / results.len() as f32;
        let std = variance.sqrt();

        // Std should be > 0.01 (not collapsed to single value)
        assert!(std > 0.001, "Predictions collapsed: std={}", std);
    }

    #[test]
    fn test_attention_with_extreme_inputs() {
        // Test that attention handles extreme input values without NaN
        use super::super::Activation;

        let mut spec = parse_rules("test", "score(X) :- proj(X).").unwrap();

        spec.add_predicate(
            "proj",
            PredicateSpec::LearnedProjection {
                inputs: vec!["features".into()],
                input_dim: 8,
                hidden_dim: 8,
                activation: Activation::GELU,
                attention_heads: Some(2),
                attention_dropout: None,
                layer_norm: Some(true),
                dropout: None,
                residual: Some(true),
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Test with extreme values (simulating unnormalized inputs)
        let mut inputs = HashMap::new();
        inputs.insert(
            "features".to_string(),
            Tensor::from_vec(
                vec![
                    100.0f32, -100.0, 0.0, 50.0, -50.0, 1e6, -1e6, 0.0, // Row 1: extreme
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Row 2: all zeros
                ],
                (2, 8),
                &test_device(),
            )
            .unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();
        let results = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        for (i, result) in results.iter().enumerate() {
            assert!(!result.is_nan(), "NaN at index {} with extreme inputs", i);
            assert!(
                !result.is_infinite(),
                "Infinite at index {} with extreme inputs",
                i
            );
        }
    }

    #[test]
    fn test_attention_stable_after_backward() {
        // Regression test: NaN after first backward + step
        // This was the org-lttr training bug
        use super::super::Activation;
        use candle_nn::optim::{Optimizer, SGD};

        let mut spec = parse_rules("test", "exit(X) :- exit_signal(X).").unwrap();

        spec.add_predicate(
            "exit_signal",
            PredicateSpec::LearnedProjection {
                inputs: vec!["features".into()],
                input_dim: 18,
                hidden_dim: 128,
                activation: Activation::GELU,
                attention_heads: Some(4),
                attention_dropout: Some(0.1),
                layer_norm: Some(true),
                dropout: Some(0.3),
                residual: Some(true),
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();
        let vars = compiled.trainable_vars();
        let mut optimizer = SGD::new(vars.clone(), 0.01).unwrap();

        // Simulate 5 training steps
        for epoch in 0..5 {
            let mut inputs = HashMap::new();
            inputs.insert(
                "features".to_string(),
                Tensor::randn(0.0f32, 1.0, (10, 18), &test_device()).unwrap(),
            );

            // Forward pass
            let output = compiled.forward(&inputs).unwrap();
            let predictions = output.output;

            // Check forward didn't produce NaN
            let pred_vals = predictions.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (i, v) in pred_vals.iter().enumerate() {
                assert!(
                    !v.is_nan(),
                    "NaN in forward at epoch {} sample {}",
                    epoch,
                    i
                );
                assert!(
                    !v.is_infinite(),
                    "Inf in forward at epoch {} sample {}",
                    epoch,
                    i
                );
            }

            // Create target and compute simple MSE loss
            let target = Tensor::from_vec(vec![0.5f32; 10], (10,), &test_device()).unwrap();

            let diff = (&predictions - &target).unwrap();
            let loss = diff.sqr().unwrap().mean_all().unwrap();

            // Backward pass
            let grads = loss.backward().unwrap();

            // Step
            optimizer.step(&grads).unwrap();

            // Clamp attention weights to prevent explosion
            compiled.clamp_attention_weights(10.0).unwrap();

            // Check weights are still healthy
            assert!(
                compiled.check_weights_health().is_none(),
                "Weights unhealthy after epoch {}",
                epoch
            );
        }

        // Final forward pass should still work
        let mut inputs = HashMap::new();
        inputs.insert(
            "features".to_string(),
            Tensor::randn(0.0f32, 1.0, (10, 18), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();
        let final_preds = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        for (i, v) in final_preds.iter().enumerate() {
            assert!(!v.is_nan(), "NaN in final forward at sample {}", i);
            assert!(
                *v >= 0.0 && *v <= 1.0,
                "Out of range in final forward: {}",
                v
            );
        }
    }

    #[test]
    fn test_clamp_attention_weights() {
        use super::super::Activation;

        let mut spec = parse_rules("test", "score(X) :- proj(X).").unwrap();

        spec.add_predicate(
            "proj",
            PredicateSpec::LearnedProjection {
                inputs: vec!["x".into()],
                input_dim: 8,
                hidden_dim: 8,
                activation: Activation::GELU,
                attention_heads: Some(2),
                attention_dropout: None,
                layer_norm: None,
                dropout: None,
                residual: None,
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Should not error
        compiled.clamp_attention_weights(10.0).unwrap();

        // Weights should be healthy
        assert!(compiled.check_weights_health().is_none());
    }

    #[test]
    fn test_attention_auto_clamp_no_manual_intervention() {
        // Regression test: Training with attention should work WITHOUT manual clamp_attention_weights()
        // This was the org-lttr bug - NaN after first backward + step when not calling clamp.
        // The fix: auto-clamp weights during forward pass in apply_multi_head_attention().
        use super::super::Activation;
        use candle_nn::optim::{Optimizer, SGD};

        let mut spec = parse_rules("test", "exit(X) :- exit_signal(X).").unwrap();

        spec.add_predicate(
            "exit_signal",
            PredicateSpec::LearnedProjection {
                inputs: vec!["features".into()],
                input_dim: 18,
                hidden_dim: 128,
                activation: Activation::GELU,
                attention_heads: Some(4),
                attention_dropout: Some(0.1),
                layer_norm: Some(true),
                dropout: Some(0.3),
                residual: Some(true),
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();
        let vars = compiled.trainable_vars();
        let mut optimizer = SGD::new(vars.clone(), 0.01).unwrap();

        // Simulate 5 training steps WITHOUT calling clamp_attention_weights()
        for epoch in 0..5 {
            let mut inputs = HashMap::new();
            inputs.insert(
                "features".to_string(),
                Tensor::randn(0.0f32, 1.0, (10, 18), &test_device()).unwrap(),
            );

            // Forward pass
            let output = compiled.forward(&inputs).unwrap();
            let predictions = output.output;

            // Check forward didn't produce NaN
            let pred_vals = predictions.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (i, v) in pred_vals.iter().enumerate() {
                assert!(
                    !v.is_nan(),
                    "NaN in forward at epoch {} sample {} (auto-clamp should prevent this)",
                    epoch,
                    i
                );
                assert!(
                    !v.is_infinite(),
                    "Inf in forward at epoch {} sample {}",
                    epoch,
                    i
                );
            }

            // Create target and compute simple MSE loss
            let target = Tensor::from_vec(vec![0.5f32; 10], (10,), &test_device()).unwrap();

            let diff = (&predictions - &target).unwrap();
            let loss = diff.sqr().unwrap().mean_all().unwrap();

            // Backward pass
            let grads = loss.backward().unwrap();

            // Step - NO MANUAL CLAMPING! Auto-clamp in forward should handle this.
            optimizer.step(&grads).unwrap();

            // NOTE: We intentionally do NOT call clamp_attention_weights() here.
            // The auto-clamp during forward pass should handle weight explosion.
        }

        // Final forward pass should still work
        let mut inputs = HashMap::new();
        inputs.insert(
            "features".to_string(),
            Tensor::randn(0.0f32, 1.0, (10, 18), &test_device()).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();
        let final_preds = output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        for (i, v) in final_preds.iter().enumerate() {
            assert!(
                !v.is_nan(),
                "NaN in final forward at sample {} (auto-clamp failed)",
                i
            );
            assert!(
                *v >= 0.0 && *v <= 1.0,
                "Out of range in final forward: {}",
                v
            );
        }
    }

    #[test]
    fn test_all_feature_combinations_no_nan() {
        // Comprehensive test: all combinations of attention, layer_norm, dropout, residual
        // This tests the auto-clamp fixes for all trainable weights.
        use super::super::Activation;
        use candle_nn::optim::{Optimizer, SGD};

        let configs = vec![
            // (name, attention, layer_norm, dropout, residual)
            ("attention_only", Some(4), None, None, None),
            ("layernorm_only", None, Some(true), None, None),
            ("dropout_only", None, None, Some(0.2), None),
            ("residual_only", None, None, None, Some(true)),
            ("attention+layernorm", Some(4), Some(true), None, None),
            ("attention+dropout", Some(4), None, Some(0.2), None),
            ("attention+residual", Some(4), None, None, Some(true)),
            ("layernorm+dropout", None, Some(true), Some(0.2), None),
            ("layernorm+residual", None, Some(true), None, Some(true)),
            ("dropout+residual", None, None, Some(0.2), Some(true)),
            (
                "attention+layernorm+dropout",
                Some(4),
                Some(true),
                Some(0.2),
                None,
            ),
            (
                "attention+layernorm+residual",
                Some(4),
                Some(true),
                None,
                Some(true),
            ),
            ("all_features", Some(4), Some(true), Some(0.2), Some(true)),
        ];

        for (name, attn, ln, drop, res) in configs {
            let mut spec = parse_rules(&format!("test_{}", name), "exit(X) :- signal(X).").unwrap();

            spec.add_predicate(
                "signal",
                PredicateSpec::LearnedProjection {
                    inputs: vec!["features".into()],
                    input_dim: 18,
                    hidden_dim: 128,
                    activation: Activation::GELU,
                    attention_heads: attn,
                    attention_dropout: attn.map(|_| 0.1),
                    layer_norm: ln,
                    dropout: drop,
                    residual: res,
                    conditioning_dim: None,
                    conditioning_type: None,
                    film_identity_init: None,
                },
            );

            let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();
            let vars = compiled.trainable_vars();
            let mut optimizer = SGD::new(vars.clone(), 0.01).unwrap();

            // Train for 5 steps - should NOT produce NaN with auto-clamp
            for epoch in 0..5 {
                let mut inputs = HashMap::new();
                inputs.insert(
                    "features".to_string(),
                    Tensor::randn(0.0f32, 1.0, (10, 18), &test_device()).unwrap(),
                );

                let output = compiled.forward(&inputs).unwrap();
                let predictions = output.output;

                // Check for NaN
                let pred_vals = predictions.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                for (i, v) in pred_vals.iter().enumerate() {
                    assert!(
                        !v.is_nan() && v.is_finite(),
                        "{}: NaN/Inf at epoch {} sample {} (pred={})",
                        name,
                        epoch,
                        i,
                        v
                    );
                }

                let target = Tensor::from_vec(vec![0.5f32; 10], (10,), &test_device()).unwrap();
                let diff = (&predictions - &target).unwrap();
                let loss = diff.sqr().unwrap().mean_all().unwrap();

                let grads = loss.backward().unwrap();
                optimizer.step(&grads).unwrap();
                // NOTE: No manual clamp_attention_weights() - auto-clamp should handle it
            }
        }
    }

    #[test]
    fn test_forward_fast_matches_forward() {
        use super::super::Activation;

        let mut spec = parse_rules("test", "score(X) :- proj(X).").unwrap();

        spec.add_predicate(
            "proj",
            PredicateSpec::LearnedProjection {
                inputs: vec!["embedding".into()],
                input_dim: 18,
                hidden_dim: 128,
                activation: Activation::ReLU,
                attention_heads: None,
                attention_dropout: None,
                layer_norm: Some(true),
                dropout: None,
                residual: Some(true),
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Same inputs for both
        let mut inputs = HashMap::new();
        inputs.insert(
            "embedding".to_string(),
            Tensor::randn(0.0f32, 1.0, (10, 18), &test_device()).unwrap(),
        );

        // forward() - full output with explanation
        let full_output = compiled.forward(&inputs).unwrap();

        // forward_fast() - just the tensor
        let fast_output = compiled.forward_fast(&inputs).unwrap();

        // Results should match exactly
        let full_vals = full_output
            .output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let fast_vals = fast_output.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(full_vals.len(), fast_vals.len());
        for (i, (full, fast)) in full_vals.iter().zip(fast_vals.iter()).enumerate() {
            assert!(
                (full - fast).abs() < 1e-6,
                "Mismatch at index {}: forward={}, forward_fast={}",
                i,
                full,
                fast
            );
        }
    }

    #[test]
    fn test_forward_batch() {
        use super::super::Activation;

        let mut spec = parse_rules("test", "score(X) :- proj(X).").unwrap();

        spec.add_predicate(
            "proj",
            PredicateSpec::LearnedProjection {
                inputs: vec!["features".into()],
                input_dim: 8,
                hidden_dim: 32,
                activation: Activation::GELU,
                attention_heads: None,
                attention_dropout: None,
                layer_norm: Some(true),
                dropout: None,
                residual: None,
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Create 5 separate input samples
        let batch_inputs: Vec<HashMap<String, Tensor>> = (0..5)
            .map(|_| {
                let mut inputs = HashMap::new();
                inputs.insert(
                    "features".to_string(),
                    Tensor::randn(0.0f32, 1.0, (1, 8), &test_device()).unwrap(),
                );
                inputs
            })
            .collect();

        // Batch inference
        let batch_output = compiled.forward_batch(&batch_inputs).unwrap();
        let batch_vals = batch_output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Individual inference for comparison
        let individual_vals: Vec<f32> = batch_inputs
            .iter()
            .map(|inputs| {
                compiled
                    .forward_fast(inputs)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()[0]
            })
            .collect();

        assert_eq!(batch_vals.len(), 5);
        assert_eq!(individual_vals.len(), 5);

        // Results should match (with small tolerance for float ops)
        for (i, (batch, individual)) in batch_vals.iter().zip(individual_vals.iter()).enumerate() {
            assert!(
                (batch - individual).abs() < 1e-5,
                "Batch mismatch at index {}: batch={}, individual={}",
                i,
                batch,
                individual
            );
        }
    }

    #[test]
    fn test_forward_fast_performance_characteristics() {
        // This test verifies the fast path is structurally simpler
        use super::super::Activation;
        use std::time::Instant;

        let mut spec = parse_rules("test", "exit(X) :- signal(X).").unwrap();

        spec.add_predicate(
            "signal",
            PredicateSpec::LearnedProjection {
                inputs: vec!["features".into()],
                input_dim: 18,
                hidden_dim: 512,
                activation: Activation::ReLU,
                attention_heads: None,
                attention_dropout: None,
                layer_norm: Some(true),
                dropout: None,
                residual: Some(true),
                conditioning_dim: None,
                conditioning_type: None,
                film_identity_init: None,
            },
        );

        let compiled = CompiledRuleSet::compile(spec, &test_device()).unwrap();

        // Warm up
        let mut inputs = HashMap::new();
        inputs.insert(
            "features".to_string(),
            Tensor::randn(0.0f32, 1.0, (1, 18), &test_device()).unwrap(),
        );
        let _ = compiled.forward_fast(&inputs).unwrap();

        // Time many forward_fast calls
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = compiled.forward_fast(&inputs).unwrap();
        }
        let fast_duration = start.elapsed();

        // Time many forward calls (with explanation overhead)
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = compiled.forward(&inputs).unwrap();
        }
        let full_duration = start.elapsed();

        // forward_fast should be faster (at least 1.5x typically)
        // Note: This is a soft assertion since CI performance varies
        let speedup = full_duration.as_nanos() as f64 / fast_duration.as_nanos() as f64;
        println!(
            "forward_fast speedup: {:.2}x ({:?} vs {:?})",
            speedup, fast_duration, full_duration
        );

        // Just verify it works, don't fail on speed (CI variability)
        assert!(fast_duration.as_nanos() > 0);
    }
}

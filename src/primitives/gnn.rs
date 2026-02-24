//! Graph Neural Network primitives
//!
//! Standalone, composable GNN operations that can be used independently of
//! the rule compiler's `GraphNeural` predicate type. Compose arbitrary GNN
//! architectures and feed their outputs into the rule compiler as external
//! predicate inputs.
//!
//! ## Layer types
//!
//! | Function | Architecture | Paper |
//! |---|---|---|
//! | [`sage_layer`] | GraphSAGE — separate self/neighbor transforms | Hamilton et al. 2017 |
//! | [`gcn_layer`] | GCN — symmetric normalized aggregation | Kipf & Welling 2017 |
//! | [`gat_layer`] | GAT — learned attention over neighbor edges | Veličković et al. 2018 |
//! | [`message_passing`] | Generic — plug in any aggregation function | — |
//!
//! ## Adjacency formats
//!
//! All layers accept dense `[N, N]` adjacency matrices or sparse COO format
//! via [`crate::compiler::codegen::get_adjacency_matrix`].
//!
//! ## Example
//!
//! ```rust,ignore
//! use an_tensor_compiler::primitives::gnn::*;
//! use an_tensor_compiler::prelude::*;
//!
//! // Build a 2-layer GNN
//! let device = best_device();
//! let (n_nodes, feat_dim, hidden_dim, out_dim) = (20, 16, 32, 8);
//!
//! let node_features = Tensor::randn(0.0f32, 1.0, (n_nodes, feat_dim), &device)?;
//! let adj = random_adjacency(n_nodes, &device)?; // your adjacency matrix
//!
//! // Layer 1: GraphSAGE
//! let w1 = Var::from_tensor(&Tensor::randn(0.0f32, 0.1, (feat_dim * 2, hidden_dim), &device)?)?;
//! let b1 = Var::zeros(hidden_dim, DType::F32, &device)?;
//! let h1 = sage_layer(&node_features, &adj, &w1, &b1, true)?;
//!
//! // Layer 2: GCN with symmetric normalization
//! let w2 = Var::from_tensor(&Tensor::randn(0.0f32, 0.1, (hidden_dim, out_dim), &device)?)?;
//! let b2 = Var::zeros(out_dim, DType::F32, &device)?;
//! let h2 = gcn_layer(&h1, &adj, &w2, &b2, true)?;
//!
//! // Use h2 as input to rule compiler
//! inputs.insert("graph_features".into(), h2);
//! ```

use crate::{Result, TensorCoreError};
use candle_core::{DType, Tensor, Var};

// =============================================================================
// Normalization
// =============================================================================

/// Row-normalize adjacency matrix: `D⁻¹ A`
///
/// Standard normalization for GraphSAGE-style mean aggregation.
/// Each row sums to 1.0 (or 0.0 for isolated nodes).
pub fn row_normalize(adj: &Tensor) -> Result<Tensor> {
    let degree = adj
        .sum_keepdim(1)
        .map_err(|e| TensorCoreError::Tensor(format!("row_normalize degree failed: {}", e)))?;

    // Add epsilon to prevent division by zero for isolated nodes
    let degree = degree
        .affine(1.0, 1e-8)
        .map_err(|e| TensorCoreError::Tensor(format!("row_normalize eps failed: {}", e)))?;

    adj.broadcast_div(&degree)
        .map_err(|e| TensorCoreError::Tensor(format!("row_normalize div failed: {}", e)))
}

/// Symmetric normalization of adjacency matrix: `D⁻¹/² (A + I) D⁻¹/²`
///
/// GCN-style normalization (Kipf & Welling 2017). Adding self-loops (`+ I`)
/// before normalization is standard practice — set `add_self_loops = true`.
///
/// Compared to row normalization, symmetric normalization provides better
/// gradient flow and is empirically more stable for deep GCNs.
pub fn symmetric_normalize(adj: &Tensor, add_self_loops: bool) -> Result<Tensor> {
    let n = adj.dims()[0];
    let device = adj.device();

    // Optionally add self-loops: A' = A + I
    let adj = if add_self_loops {
        let eye = Tensor::eye(n, DType::F32, device)
            .map_err(|e| TensorCoreError::Tensor(format!("eye failed: {}", e)))?;
        (adj + &eye).map_err(|e| TensorCoreError::Tensor(format!("add self-loop failed: {}", e)))?
    } else {
        adj.clone()
    };

    // Degree: D[i] = sum of row i
    let degree = adj
        .sum_keepdim(1)
        .map_err(|e| TensorCoreError::Tensor(format!("sym_norm degree failed: {}", e)))?;

    // D⁻¹/²: element-wise reciprocal square root
    let d_inv_sqrt = degree
        .affine(1.0, 1e-8)
        .map_err(|e| TensorCoreError::Tensor(format!("sym_norm eps failed: {}", e)))?
        .sqrt()
        .map_err(|e| TensorCoreError::Tensor(format!("sym_norm sqrt failed: {}", e)))?
        .recip()
        .map_err(|e| TensorCoreError::Tensor(format!("sym_norm recip failed: {}", e)))?;

    // D⁻¹/² A D⁻¹/² = d_inv_sqrt * A * d_inv_sqrt (broadcast row × col)
    let left = adj
        .broadcast_mul(&d_inv_sqrt)
        .map_err(|e| TensorCoreError::Tensor(format!("sym_norm left mul failed: {}", e)))?;

    let d_inv_sqrt_col = d_inv_sqrt
        .transpose(0, 1)
        .map_err(|e| TensorCoreError::Tensor(format!("sym_norm transpose failed: {}", e)))?;

    left.broadcast_mul(&d_inv_sqrt_col)
        .map_err(|e| TensorCoreError::Tensor(format!("sym_norm right mul failed: {}", e)))
}

// =============================================================================
// GraphSAGE layer
// =============================================================================

/// GraphSAGE message passing layer (Hamilton et al. 2017)
///
/// Separates self-transform and neighbor-aggregation with independent weight
/// matrices, then concatenates and projects:
///
/// ```text
/// h_agg = mean(A_norm @ H)                     — mean neighbor aggregation
/// h_cat = concat(H, h_agg)                     — concatenate self + neighbors
/// h_out = ReLU(h_cat @ W + b)                  — project and activate
/// ```
///
/// # Arguments
/// - `h`: Node features `[N, feat_dim]`
/// - `adj`: Adjacency matrix `[N, N]` (raw, will be row-normalized internally)
/// - `w`: Weight matrix `[feat_dim * 2, out_dim]` (self + neighbor concat)
/// - `b`: Bias `[out_dim]`
/// - `apply_relu`: Whether to apply ReLU activation (disable for final layer)
///
/// # Returns
/// Node embeddings `[N, out_dim]`
pub fn sage_layer(h: &Tensor, adj: &Tensor, w: &Var, b: &Var, apply_relu: bool) -> Result<Tensor> {
    let adj_norm = row_normalize(adj)?;

    // Mean neighbor aggregation: A_norm @ H → [N, feat_dim]
    let h_neighbor = adj_norm
        .matmul(h)
        .map_err(|e| TensorCoreError::Tensor(format!("sage matmul neighbor failed: {}", e)))?;

    // Concatenate self features and neighbor features: [N, feat_dim * 2]
    let h_cat = Tensor::cat(&[h, &h_neighbor], 1)
        .map_err(|e| TensorCoreError::Tensor(format!("sage cat failed: {}", e)))?;

    // Project: [N, feat_dim * 2] @ [feat_dim * 2, out_dim] → [N, out_dim]
    let out = h_cat
        .matmul(w.as_tensor())
        .map_err(|e| TensorCoreError::Tensor(format!("sage matmul W failed: {}", e)))?
        .broadcast_add(b.as_tensor())
        .map_err(|e| TensorCoreError::Tensor(format!("sage add bias failed: {}", e)))?;

    if apply_relu {
        out.relu()
            .map_err(|e| TensorCoreError::Tensor(format!("sage relu failed: {}", e)))
    } else {
        Ok(out)
    }
}

// =============================================================================
// GCN layer
// =============================================================================

/// GCN message passing layer (Kipf & Welling 2017)
///
/// Uses symmetric normalization `D⁻¹/²(A+I)D⁻¹/²` for better gradient flow:
///
/// ```text
/// A_hat = D⁻¹/²(A + I)D⁻¹/²              — symmetric normalized adjacency
/// h_out = ReLU(A_hat @ H @ W + b)          — aggregate, transform, activate
/// ```
///
/// Compared to GraphSAGE, GCN uses a single weight matrix and symmetric
/// normalization, which empirically works well for semi-supervised tasks.
///
/// # Arguments
/// - `h`: Node features `[N, feat_dim]`
/// - `adj`: Adjacency matrix `[N, N]` (self-loops will be added internally)
/// - `w`: Weight matrix `[feat_dim, out_dim]`
/// - `b`: Bias `[out_dim]`
/// - `apply_relu`: Whether to apply ReLU (disable for final layer)
///
/// # Returns
/// Node embeddings `[N, out_dim]`
pub fn gcn_layer(h: &Tensor, adj: &Tensor, w: &Var, b: &Var, apply_relu: bool) -> Result<Tensor> {
    // Symmetric normalization with self-loops (standard GCN)
    let adj_hat = symmetric_normalize(adj, true)?;

    // A_hat @ H @ W: aggregate then project
    let out = adj_hat
        .matmul(h)
        .map_err(|e| TensorCoreError::Tensor(format!("gcn matmul A@H failed: {}", e)))?
        .matmul(w.as_tensor())
        .map_err(|e| TensorCoreError::Tensor(format!("gcn matmul W failed: {}", e)))?
        .broadcast_add(b.as_tensor())
        .map_err(|e| TensorCoreError::Tensor(format!("gcn add bias failed: {}", e)))?;

    if apply_relu {
        out.relu()
            .map_err(|e| TensorCoreError::Tensor(format!("gcn relu failed: {}", e)))
    } else {
        Ok(out)
    }
}

// =============================================================================
// GAT layer
// =============================================================================

/// GAT message passing layer (Veličković et al. 2018)
///
/// Learns per-edge attention weights from node feature pairs:
///
/// ```text
/// z_i = H @ W                                    — linear transform
/// e_ij = LeakyReLU([z_i || z_j] @ a)             — edge attention score
/// α_ij = softmax_j(e_ij) * adj_ij                — masked, normalized weights
/// h_out = σ(Σ_j α_ij * z_j)                      — weighted aggregation
/// ```
///
/// The attention mechanism allows the model to learn which neighbors are
/// most relevant for each node, rather than treating all neighbors equally.
///
/// # Arguments
/// - `h`: Node features `[N, feat_dim]`
/// - `adj`: Adjacency matrix `[N, N]` (used as mask — 0 entries are excluded)
/// - `w`: Node transform `[feat_dim, out_dim]`
/// - `a`: Attention vector `[out_dim * 2]`
/// - `b`: Bias `[out_dim]`
/// - `apply_activation`: Whether to apply sigmoid output activation
/// - `leaky_relu_slope`: Negative slope for LeakyReLU (default: 0.2)
///
/// # Returns
/// Node embeddings `[N, out_dim]`
pub fn gat_layer(
    h: &Tensor,
    adj: &Tensor,
    w: &Var,
    a: &Var,
    b: &Var,
    apply_activation: bool,
    leaky_relu_slope: f64,
) -> Result<Tensor> {
    let (n, _) = h
        .dims2()
        .map_err(|e| TensorCoreError::Tensor(format!("gat dims2 failed: {}", e)))?;
    let device = h.device();

    // Linear transform: [N, feat_dim] → [N, out_dim]
    let z = h
        .matmul(w.as_tensor())
        .map_err(|e| TensorCoreError::Tensor(format!("gat matmul W failed: {}", e)))?;

    let out_dim = z.dims()[1];

    // Compute attention scores for all pairs: e_ij = LeakyReLU([z_i || z_j] @ a)
    // z_i repeated across columns: [N, 1, out_dim] → [N, N, out_dim]
    // z_j repeated across rows:    [1, N, out_dim] → [N, N, out_dim]
    let z_i = z
        .unsqueeze(1)
        .map_err(|e| TensorCoreError::Tensor(format!("gat unsqueeze i failed: {}", e)))?
        .expand(&[n, n, out_dim])
        .map_err(|e| TensorCoreError::Tensor(format!("gat expand i failed: {}", e)))?;

    let z_j = z
        .unsqueeze(0)
        .map_err(|e| TensorCoreError::Tensor(format!("gat unsqueeze j failed: {}", e)))?
        .expand(&[n, n, out_dim])
        .map_err(|e| TensorCoreError::Tensor(format!("gat expand j failed: {}", e)))?;

    // Concatenate: [N, N, out_dim * 2]
    let z_cat = Tensor::cat(&[&z_i, &z_j], 2)
        .map_err(|e| TensorCoreError::Tensor(format!("gat cat failed: {}", e)))?;

    // Attention: [N, N, out_dim * 2] @ [out_dim * 2] → [N, N]
    let a_vec = a
        .as_tensor()
        .unsqueeze(1)
        .map_err(|e| TensorCoreError::Tensor(format!("gat unsqueeze a failed: {}", e)))?;

    let e_raw = z_cat
        .reshape((n * n, out_dim * 2))
        .map_err(|e| TensorCoreError::Tensor(format!("gat reshape failed: {}", e)))?
        .matmul(&a_vec)
        .map_err(|e| TensorCoreError::Tensor(format!("gat matmul a failed: {}", e)))?
        .reshape((n, n))
        .map_err(|e| TensorCoreError::Tensor(format!("gat reshape e failed: {}", e)))?;

    // LeakyReLU: max(slope * x, x)
    let zeros = Tensor::zeros_like(&e_raw)
        .map_err(|e| TensorCoreError::Tensor(format!("gat zeros failed: {}", e)))?;
    let positive = e_raw
        .maximum(&zeros)
        .map_err(|e| TensorCoreError::Tensor(format!("gat maximum failed: {}", e)))?;
    let negative = e_raw
        .minimum(&zeros)
        .map_err(|e| TensorCoreError::Tensor(format!("gat minimum failed: {}", e)))?;
    let leaky_neg = negative
        .affine(leaky_relu_slope, 0.0)
        .map_err(|e| TensorCoreError::Tensor(format!("gat leaky scale failed: {}", e)))?;
    let e = (&positive + &leaky_neg)
        .map_err(|e| TensorCoreError::Tensor(format!("gat leaky add failed: {}", e)))?;

    // Mask with adjacency (set non-edges to -inf before softmax)
    let neg_inf = Tensor::full(f32::NEG_INFINITY, (n, n), device)
        .map_err(|e| TensorCoreError::Tensor(format!("gat neg_inf failed: {}", e)))?;
    let adj_bool = adj
        .gt(0.0f64)
        .map_err(|e| TensorCoreError::Tensor(format!("gat adj_bool failed: {}", e)))?;
    let e_masked = adj_bool
        .where_cond(&e, &neg_inf)
        .map_err(|e| TensorCoreError::Tensor(format!("gat where_cond failed: {}", e)))?;

    // Softmax over neighbors (row-wise): attention weights [N, N]
    let alpha = candle_nn::ops::softmax(&e_masked, 1)
        .map_err(|e| TensorCoreError::Tensor(format!("gat softmax failed: {}", e)))?;

    // Weighted aggregation: [N, N] @ [N, out_dim] → [N, out_dim]
    let out = alpha
        .matmul(&z)
        .map_err(|e| TensorCoreError::Tensor(format!("gat matmul alpha@z failed: {}", e)))?
        .broadcast_add(b.as_tensor())
        .map_err(|e| TensorCoreError::Tensor(format!("gat add bias failed: {}", e)))?;

    if apply_activation {
        crate::primitives::sigmoid(&out)
    } else {
        Ok(out)
    }
}

// =============================================================================
// Multi-head GAT
// =============================================================================

/// Multi-head GAT layer — runs `num_heads` independent attention heads
///
/// Each head attends to different aspects of the neighborhood. Outputs are
/// concatenated (for intermediate layers) or averaged (for final layer).
///
/// # Arguments
/// - `h`: Node features `[N, feat_dim]`
/// - `adj`: Adjacency matrix `[N, N]`
/// - `ws`: Weight matrices — one per head, each `[feat_dim, head_dim]`
/// - `as_`: Attention vectors — one per head, each `[head_dim * 2]`
/// - `bs`: Biases — one per head, each `[head_dim]`
/// - `concat`: If true, concatenate heads `[N, head_dim * num_heads]`;
///   if false, average them `[N, head_dim]`
/// - `leaky_relu_slope`: Negative slope for LeakyReLU (default: 0.2)
///
/// # Returns
/// Node embeddings `[N, head_dim * num_heads]` (concat) or `[N, head_dim]` (average)
pub fn multi_head_gat(
    h: &Tensor,
    adj: &Tensor,
    ws: &[Var],
    as_: &[Var],
    bs: &[Var],
    concat: bool,
    leaky_relu_slope: f64,
) -> Result<Tensor> {
    if ws.is_empty() || ws.len() != as_.len() || ws.len() != bs.len() {
        return Err(TensorCoreError::Tensor(
            "multi_head_gat: ws, as_, and bs must be non-empty and equal length".into(),
        ));
    }

    let head_outputs: Result<Vec<Tensor>> = ws
        .iter()
        .zip(as_.iter())
        .zip(bs.iter())
        .map(|((w, a), b)| gat_layer(h, adj, w, a, b, false, leaky_relu_slope))
        .collect();
    let head_outputs = head_outputs?;

    if concat {
        // Concatenate along feature dimension: [N, head_dim * num_heads]
        let refs: Vec<&Tensor> = head_outputs.iter().collect();
        Tensor::cat(&refs, 1)
            .map_err(|e| TensorCoreError::Tensor(format!("multi_head_gat cat failed: {}", e)))
    } else {
        // Average heads: [N, head_dim]
        let n_heads = head_outputs.len() as f64;
        let refs: Vec<&Tensor> = head_outputs.iter().collect();
        let sum = Tensor::stack(&refs, 0)
            .map_err(|e| TensorCoreError::Tensor(format!("multi_head_gat stack failed: {}", e)))?
            .sum(0)
            .map_err(|e| TensorCoreError::Tensor(format!("multi_head_gat sum failed: {}", e)))?;
        sum.affine(1.0 / n_heads, 0.0)
            .map_err(|e| TensorCoreError::Tensor(format!("multi_head_gat scale failed: {}", e)))
    }
}

// =============================================================================
// Generic message passing
// =============================================================================

/// Generic message passing layer
///
/// Applies a user-provided aggregation function to neighbor features, then
/// combines with self features through a linear transform.
///
/// ```text
/// h_agg = aggregate_fn(adj, H)                  — custom aggregation
/// h_out = activation(concat(H, h_agg) @ W + b) — transform and activate
/// ```
///
/// Use this when GraphSAGE, GCN, and GAT don't fit your use case.
/// The `aggregate_fn` receives `(adjacency, node_features)` and returns
/// aggregated node features of the same shape as `node_features`.
pub fn message_passing<F>(
    h: &Tensor,
    adj: &Tensor,
    w: &Var,
    b: &Var,
    aggregate_fn: F,
    apply_relu: bool,
) -> Result<Tensor>
where
    F: Fn(&Tensor, &Tensor) -> Result<Tensor>,
{
    let h_agg = aggregate_fn(adj, h)?;

    let h_cat = Tensor::cat(&[h, &h_agg], 1)
        .map_err(|e| TensorCoreError::Tensor(format!("message_passing cat failed: {}", e)))?;

    let out = h_cat
        .matmul(w.as_tensor())
        .map_err(|e| TensorCoreError::Tensor(format!("message_passing matmul failed: {}", e)))?
        .broadcast_add(b.as_tensor())
        .map_err(|e| TensorCoreError::Tensor(format!("message_passing bias failed: {}", e)))?;

    if apply_relu {
        out.relu()
            .map_err(|e| TensorCoreError::Tensor(format!("message_passing relu failed: {}", e)))
    } else {
        Ok(out)
    }
}

// =============================================================================
// Adjacency utilities
// =============================================================================

/// Global mean pooling: average node features into a graph-level embedding
///
/// Reduces `[N, feat_dim]` to `[feat_dim]` by averaging across nodes.
/// Standard readout for graph classification tasks.
pub fn global_mean_pool(h: &Tensor) -> Result<Tensor> {
    h.mean(0)
        .map_err(|e| TensorCoreError::Tensor(format!("global_mean_pool failed: {}", e)))
}

/// Global max pooling: max node features into a graph-level embedding
///
/// Reduces `[N, feat_dim]` to `[feat_dim]` by taking the max across nodes.
/// Captures the most activated feature across the whole graph.
pub fn global_max_pool(h: &Tensor) -> Result<Tensor> {
    h.max(0)
        .map_err(|e| TensorCoreError::Tensor(format!("global_max_pool failed: {}", e)))
}

/// Global sum pooling: sum node features into a graph-level embedding
pub fn global_sum_pool(h: &Tensor) -> Result<Tensor> {
    h.sum(0)
        .map_err(|e| TensorCoreError::Tensor(format!("global_sum_pool failed: {}", e)))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn device() -> Device {
        Device::Cpu
    }

    fn small_graph() -> (Tensor, Tensor) {
        // 5-node graph, 3 features per node
        let h = Tensor::randn(0.0f32, 1.0, (5, 3), &device()).unwrap();
        // Simple chain adjacency: 0-1-2-3-4
        let adj_data: Vec<f32> = vec![
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let adj = Tensor::from_vec(adj_data, (5, 5), &device()).unwrap();
        (h, adj)
    }

    #[test]
    fn test_row_normalize() {
        let (_, adj) = small_graph();
        let normed = row_normalize(&adj).unwrap();
        let row_sums: Vec<f32> = normed.sum(1).unwrap().to_vec1().unwrap();
        // Each row (that has neighbors) should sum to ~1.0
        for (i, &sum) in row_sums.iter().enumerate() {
            if sum > 0.0 {
                assert!((sum - 1.0).abs() < 0.01, "row {} sum was {}", i, sum);
            }
        }
    }

    #[test]
    fn test_symmetric_normalize() {
        let (_, adj) = small_graph();
        let normed = symmetric_normalize(&adj, true).unwrap();
        // Should be square and same shape
        assert_eq!(normed.dims(), &[5, 5]);
        // Values should be in reasonable range (not exploding)
        let vals: Vec<f32> = normed.flatten_all().unwrap().to_vec1().unwrap();
        for v in vals {
            assert!(v.is_finite(), "Non-finite value in symmetric normalization");
            assert!(v >= 0.0, "Negative value in symmetric normalization");
        }
    }

    #[test]
    fn test_sage_layer() {
        let (h, adj) = small_graph();
        let in_dim = 3;
        let out_dim = 8;

        // W: [feat_dim * 2, out_dim] (SAGE concatenates self + neighbor)
        let w = Var::from_tensor(
            &Tensor::randn(0.0f32, 0.1, (in_dim * 2, out_dim), &device()).unwrap(),
        )
        .unwrap();
        let b = Var::zeros(out_dim, DType::F32, &device()).unwrap();

        let out = sage_layer(&h, &adj, &w, &b, true).unwrap();
        assert_eq!(out.dims(), &[5, out_dim]);

        // Verify all values are finite and non-negative (ReLU applied)
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for v in &vals {
            assert!(v.is_finite());
            assert!(*v >= 0.0, "ReLU should produce non-negative values");
        }
    }

    #[test]
    fn test_gcn_layer() {
        let (h, adj) = small_graph();
        let in_dim = 3;
        let out_dim = 8;

        // W: [feat_dim, out_dim] (GCN uses single weight matrix)
        let w =
            Var::from_tensor(&Tensor::randn(0.0f32, 0.1, (in_dim, out_dim), &device()).unwrap())
                .unwrap();
        let b = Var::zeros(out_dim, DType::F32, &device()).unwrap();

        let out = gcn_layer(&h, &adj, &w, &b, true).unwrap();
        assert_eq!(out.dims(), &[5, out_dim]);

        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for v in &vals {
            assert!(v.is_finite());
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn test_gat_layer() {
        let (h, adj) = small_graph();
        let in_dim = 3;
        let out_dim = 4;

        let w =
            Var::from_tensor(&Tensor::randn(0.0f32, 0.1, (in_dim, out_dim), &device()).unwrap())
                .unwrap();
        let a = Var::from_tensor(&Tensor::randn(0.0f32, 0.1, (out_dim * 2,), &device()).unwrap())
            .unwrap();
        let b = Var::zeros(out_dim, DType::F32, &device()).unwrap();

        let out = gat_layer(&h, &adj, &w, &a, &b, false, 0.2).unwrap();
        assert_eq!(out.dims(), &[5, out_dim]);

        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for v in &vals {
            assert!(v.is_finite(), "GAT produced non-finite value: {}", v);
        }
    }

    #[test]
    fn test_multi_head_gat_concat() {
        let (h, adj) = small_graph();
        let in_dim = 3;
        let head_dim = 4;
        let num_heads = 2;

        let ws: Vec<Var> = (0..num_heads)
            .map(|_| {
                Var::from_tensor(
                    &Tensor::randn(0.0f32, 0.1, (in_dim, head_dim), &device()).unwrap(),
                )
                .unwrap()
            })
            .collect();
        let as_: Vec<Var> = (0..num_heads)
            .map(|_| {
                Var::from_tensor(&Tensor::randn(0.0f32, 0.1, (head_dim * 2,), &device()).unwrap())
                    .unwrap()
            })
            .collect();
        let bs: Vec<Var> = (0..num_heads)
            .map(|_| Var::zeros(head_dim, DType::F32, &device()).unwrap())
            .collect();

        let out = multi_head_gat(&h, &adj, &ws, &as_, &bs, true, 0.2).unwrap();
        // Concat: [5, head_dim * num_heads]
        assert_eq!(out.dims(), &[5, head_dim * num_heads]);
    }

    #[test]
    fn test_multi_head_gat_average() {
        let (h, adj) = small_graph();
        let in_dim = 3;
        let head_dim = 4;
        let num_heads = 3;

        let ws: Vec<Var> = (0..num_heads)
            .map(|_| {
                Var::from_tensor(
                    &Tensor::randn(0.0f32, 0.1, (in_dim, head_dim), &device()).unwrap(),
                )
                .unwrap()
            })
            .collect();
        let as_: Vec<Var> = (0..num_heads)
            .map(|_| {
                Var::from_tensor(&Tensor::randn(0.0f32, 0.1, (head_dim * 2,), &device()).unwrap())
                    .unwrap()
            })
            .collect();
        let bs: Vec<Var> = (0..num_heads)
            .map(|_| Var::zeros(head_dim, DType::F32, &device()).unwrap())
            .collect();

        let out = multi_head_gat(&h, &adj, &ws, &as_, &bs, false, 0.2).unwrap();
        // Average: [5, head_dim]
        assert_eq!(out.dims(), &[5, head_dim]);
    }

    #[test]
    fn test_global_pooling() {
        let (h, _) = small_graph();

        let mean = global_mean_pool(&h).unwrap();
        let max = global_max_pool(&h).unwrap();
        let sum = global_sum_pool(&h).unwrap();

        assert_eq!(mean.dims(), &[3]);
        assert_eq!(max.dims(), &[3]);
        assert_eq!(sum.dims(), &[3]);
    }

    #[test]
    fn test_two_layer_gnn_pipeline() {
        let device = device();
        let (n, feat, hidden, out) = (10, 8, 16, 4);
        let h = Tensor::randn(0.0f32, 1.0, (n, feat), &device).unwrap();
        let adj_data: Vec<f32> = (0..n * n)
            .map(|i| if (i / n) != (i % n) { 0.2f32 } else { 0.0 })
            .collect();
        let adj = Tensor::from_vec(adj_data, (n, n), &device).unwrap();

        // Layer 1: GraphSAGE
        let w1 =
            Var::from_tensor(&Tensor::randn(0.0f32, 0.1, (feat * 2, hidden), &device).unwrap())
                .unwrap();
        let b1 = Var::zeros(hidden, DType::F32, &device).unwrap();
        let h1 = sage_layer(&h, &adj, &w1, &b1, true).unwrap();
        assert_eq!(h1.dims(), &[n, hidden]);

        // Layer 2: GCN
        let w2 =
            Var::from_tensor(&Tensor::randn(0.0f32, 0.1, (hidden, out), &device).unwrap()).unwrap();
        let b2 = Var::zeros(out, DType::F32, &device).unwrap();
        let h2 = gcn_layer(&h1, &adj, &w2, &b2, false).unwrap();
        assert_eq!(h2.dims(), &[n, out]);

        // Graph readout
        let graph_embedding = global_mean_pool(&h2).unwrap();
        assert_eq!(graph_embedding.dims(), &[out]);
    }
}

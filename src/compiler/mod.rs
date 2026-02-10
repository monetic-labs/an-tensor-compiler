//! TensorRuleCompiler
//!
//! Compiles symbolic rule specifications into differentiable tensor equations.
//!
//! ## Overview
//!
//! The compiler transforms human-readable rule specifications:
//!
//! ```text
//! exit(X) ← profit_target(X, 0.02) ∧ momentum_shift(X)
//! exit(X) ← stop_loss(X, -0.01)
//! exit(X) ← regime_change(X) ∧ ¬bullish(X)
//! ```
//!
//! Into differentiable tensor functions that:
//! 1. Accept input tensors (market data, embeddings, features)
//! 2. Compute predicate activations via learned thresholds
//! 3. Combine with fuzzy logic operators
//! 4. Produce outputs with full gradient flow
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
//! │  Rule Spec     │ ──▶ │   Parser       │ ──▶ │   AST          │
//! │  (text/DSL)    │     │   ✅ Week 2    │     │                │
//! └────────────────┘     └────────────────┘     └────────────────┘
//!                                                      │
//!                                                      ▼
//! ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
//! │  Executable    │ ◀── │   Codegen      │ ◀── │   IR           │
//! │  (Tensor fn)   │     │   📋 Week 3    │     │                │
//! └────────────────┘     └────────────────┘     └────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use an_tensor_compiler::compiler::RuleSpec;
//!
//! // Parse rules from DSL
//! let spec = RuleSpec::parse("exit_rules", r#"
//!     exit(X) :- profit_target(X, 0.02), momentum_shift(X).
//!     exit(X) :- stop_loss(X, -0.01).
//!     exit(X) :- regime_change(X), not bullish(X).
//! "#)?;
//!
//! assert_eq!(spec.rules.len(), 3);
//! ```

pub mod codegen;
pub mod parser;
pub mod validation;
pub mod inference;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// Re-export key codegen types
pub use codegen::CompiledRuleSet;

// Re-export inference types for high-performance batched inference
pub use inference::{InputBuffer, BatchedInputBuffer, InferenceContext, BatchedInferenceContext};

/// A rule specification in symbolic form
///
/// Rules follow Prolog-like syntax:
/// - Head ← Body
/// - Body is a conjunction of literals
/// - Literals can be negated with ¬
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleSpec {
    /// Name of this rule set (e.g., "exit_rules_v2")
    pub name: String,

    /// Version for tracking
    pub version: String,

    /// Individual rules
    pub rules: Vec<Rule>,

    /// Predicate definitions (what predicates mean)
    pub predicates: HashMap<String, PredicateSpec>,
}

impl RuleSpec {
    /// Create a new empty rule specification
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: "1.0.0".into(),
            rules: Vec::new(),
            predicates: HashMap::new(),
        }
    }

    /// Add a rule
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Add a predicate definition
    pub fn add_predicate(&mut self, name: impl Into<String>, spec: PredicateSpec) {
        self.predicates.insert(name.into(), spec);
    }

    /// Parse from a DSL string
    ///
    /// # Example
    ///
    /// ```ignore
    /// let spec = RuleSpec::parse("exit_rules", r#"
    ///     exit(X) :- profit_target(X, 0.02), momentum_shift(X).
    ///     exit(X) :- stop_loss(X, -0.01).
    /// "#)?;
    /// ```
    ///
    /// # Supported Syntax
    ///
    /// - Prolog-style: `head(X) :- body1(X), body2(X).`
    /// - Unicode: `head(X) ← body1(X) ∧ body2(X)`
    /// - Negation: `not pred(X)` or `¬pred(X)`
    /// - Disjunction: `a(X); b(X)` or `a(X) ∨ b(X)` (expands to multiple rules)
    pub fn parse(name: &str, source: &str) -> crate::Result<Self> {
        parser::parse_rules(name, source)
    }

    /// Parse from a DSL string (legacy single-arg version for backwards compat)
    #[deprecated(since = "0.2.0", note = "Use parse(name, source) instead")]
    pub fn parse_source(source: &str) -> crate::Result<Self> {
        parser::parse_rules("unnamed", source)
    }

    /// Parse and validate in one step
    ///
    /// This is the recommended way to load rules as it ensures
    /// semantic correctness in addition to syntactic correctness.
    pub fn parse_and_validate(name: &str, source: &str) -> crate::Result<Self> {
        let spec = parser::parse_rules(name, source)?;
        validation::validate_strict(&spec)?;
        Ok(spec)
    }

    /// Validate this rule specification
    ///
    /// Returns a list of validation errors (empty if valid).
    pub fn validate(&self) -> Vec<validation::ValidationError> {
        validation::validate(self)
    }

    /// Validate strictly, returning an error if any issues found
    pub fn validate_strict(&self) -> crate::Result<()> {
        validation::validate_strict(self)
    }

    /// Get all predicate names used in this specification
    pub fn predicate_names(&self) -> std::collections::HashSet<String> {
        validation::extract_predicates(self)
    }

    /// Get predicate arities (name -> argument count)
    pub fn predicate_arities(&self) -> HashMap<String, usize> {
        validation::predicate_arities(self)
    }
}

/// A single rule: head ← body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    /// Head predicate (what this rule concludes)
    pub head: String,

    /// Body literals (conjunction of conditions)
    pub body: Vec<Literal>,

    /// Learned weight for this rule (importance/confidence)
    pub weight: Option<f32>,
}

impl Rule {
    /// Create a new rule
    pub fn new(head: impl Into<String>, body: Vec<Literal>) -> Self {
        Self {
            head: head.into(),
            body,
            weight: None,
        }
    }

    /// Set the weight
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = Some(weight);
        self
    }
}

/// A literal in a rule body
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Literal {
    /// Predicate name
    pub predicate: String,

    /// Arguments (variable names or constants)
    pub args: Vec<Argument>,

    /// Whether this literal is negated
    pub negated: bool,
}

impl Literal {
    /// Create a positive literal
    pub fn positive(predicate: impl Into<String>, args: Vec<Argument>) -> Self {
        Self {
            predicate: predicate.into(),
            args,
            negated: false,
        }
    }

    /// Create a negated literal
    pub fn negated(predicate: impl Into<String>, args: Vec<Argument>) -> Self {
        Self {
            predicate: predicate.into(),
            args,
            negated: true,
        }
    }
}

/// An argument to a predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Argument {
    /// Variable (e.g., X)
    Variable(String),

    /// Constant value
    Constant(f32),

    /// String constant
    StringConstant(String),
}

/// Specification of how a predicate is computed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredicateSpec {
    /// Threshold comparison: predicate(X) = X > threshold
    Threshold {
        /// Input feature name
        input: String,
        /// Threshold value (can be learned)
        threshold: f32,
        /// Comparison direction (true = greater than)
        greater_than: bool,
        /// Sharpness of soft threshold
        sharpness: f32,
    },

    /// Learned embedding similarity (basic linear: Wx + b → sigmoid)
    Learned {
        /// Embedding dimension
        dim: usize,
    },

    /// Learned projection: high-dim embedding → scalar predicate
    /// 
    /// Replaces MLPs for embedding-to-predicate conversion.
    /// Architecture: input → Linear(hidden_dim) → [LayerNorm] → activation → [Dropout] → 
    ///               [Attention] → Linear(1) → sigmoid
    /// 
    /// # Example
    /// ```ignore
    /// PredicateSpec::LearnedProjection {
    ///     inputs: vec!["code_embedding".into()],
    ///     input_dim: 1536,
    ///     hidden_dim: 128,
    ///     activation: Activation::GELU,
    ///     // Optional enhancements:
    ///     attention_heads: Some(4),
    ///     attention_dropout: Some(0.1),
    ///     layer_norm: Some(true),
    ///     dropout: Some(0.3),
    ///     residual: Some(true),
    /// }
    /// ```
    LearnedProjection {
        /// Input tensor names (concatenated if multiple)
        inputs: Vec<String>,
        /// Total input dimension
        input_dim: usize,
        /// Hidden layer dimension
        hidden_dim: usize,
        /// Activation function between layers
        activation: Activation,
        /// Number of attention heads (None = no attention)
        #[serde(default)]
        attention_heads: Option<usize>,
        /// Dropout rate for attention (only used if attention_heads is set)
        #[serde(default)]
        attention_dropout: Option<f32>,
        /// Apply LayerNorm after each linear layer
        #[serde(default)]
        layer_norm: Option<bool>,
        /// Dropout rate after activations (0.0-1.0)
        #[serde(default)]
        dropout: Option<f32>,
        /// Add residual/skip connections (requires input_dim == hidden_dim or projection)
        #[serde(default)]
        residual: Option<bool>,
        
        /// FiLM Conditioning: dimension of conditioning vector (e.g., 6 for regime vector)
        /// When set, the conditioning input modulates hidden activations via scale/shift
        #[serde(default)]
        conditioning_dim: Option<usize>,
        
        /// How to apply conditioning (FiLM = Feature-wise Linear Modulation)
        /// FiLM: gamma * hidden + beta, where gamma/beta are projected from conditioning
        #[serde(default)]
        conditioning_type: Option<ConditioningType>,
        
        /// Initialize FiLM layers to identity transform (gamma projection → 0, beta projection → 0)
        /// When true: output = 1.0 * hidden + 0.0 (identity at initialization)
        /// Recommended for training stability - lets model learn modulation from data
        #[serde(default)]
        film_identity_init: Option<bool>,
    },

    /// Learned similarity: compare two embeddings → similarity score
    /// 
    /// Replaces MLPs for embedding comparison.
    /// 
    /// # Example
    /// ```ignore
    /// PredicateSpec::LearnedSimilarity {
    ///     left: "code_embedding".into(),
    ///     right: "intent_embedding".into(),
    ///     dim: 1536,
    ///     method: SimilarityMethod::Learned,
    /// }
    /// ```
    LearnedSimilarity {
        /// Left embedding tensor name
        left: String,
        /// Right embedding tensor name
        right: String,
        /// Embedding dimension
        dim: usize,
        /// Similarity computation method
        method: SimilarityMethod,
    },

    /// Graph Neural predicate: message passing over graph → per-node predicate
    /// 
    /// Implements GraphSAGE-style message passing as Tensor Logic.
    /// 
    /// # Example
    /// ```ignore
    /// PredicateSpec::GraphNeural {
    ///     node_features: "code_embedding".into(),
    ///     adjacency: "ast_adjacency".into(),
    ///     feature_dim: 1536,
    ///     hidden_dim: 128,
    ///     num_layers: 2,
    ///     aggregation: Aggregation::Mean,
    /// }
    /// ```
    GraphNeural {
        /// Node feature tensor name
        node_features: String,
        /// Adjacency matrix tensor name (sparse or dense)
        adjacency: String,
        /// Input feature dimension per node
        feature_dim: usize,
        /// Hidden dimension for message passing
        hidden_dim: usize,
        /// Number of message passing layers
        num_layers: usize,
        /// Aggregation method for neighbor messages
        aggregation: Aggregation,
    },

    /// Composite of other predicates
    Composite {
        /// How predicates are combined
        operator: CompositeOp,
        /// Child predicates
        children: Vec<String>,
    },
    
    /// Tier lookup: Multi-threshold step function for utilization-based preferences
    /// 
    /// Maps an input value to one of several output values based on thresholds.
    /// Common for utilization tiers, prestige levels, risk bands, etc.
    /// 
    /// # Example
    /// ```ignore
    /// // Utilization-based provider preference
    /// // < 50%: 1.5 (strong preference)
    /// // 50-80%: 1.2 (moderate)
    /// // 80-100%: 1.0 (neutral)
    /// // 100-150%: 0.8 (in overage)
    /// // > 150%: 0.6 (deep overage)
    /// PredicateSpec::TierLookup {
    ///     input: "utilization_pct".into(),
    ///     thresholds: vec![50.0, 80.0, 100.0, 150.0],
    ///     values: vec![1.5, 1.2, 1.0, 0.8, 0.6],
    ///     sharpness: 5.0,
    ///     interpolate: false,  // Step function
    /// }
    /// ```
    TierLookup {
        /// Input tensor name
        input: String,
        /// Threshold boundaries (N thresholds create N+1 tiers)
        thresholds: Vec<f32>,
        /// Output values for each tier (must be thresholds.len() + 1)
        values: Vec<f32>,
        /// Sharpness of transitions (higher = sharper step)
        #[serde(default = "default_sharpness")]
        sharpness: f32,
        /// If true, linearly interpolate between tiers; if false, step function
        #[serde(default)]
        interpolate: bool,
    },
    
    /// Pairwise difference: Compare all pairs for arbitrage/gap detection
    /// 
    /// Computes pairwise differences between elements and identifies
    /// significant gaps (e.g., cost arbitrage between providers).
    /// 
    /// # Example
    /// ```ignore
    /// // Detect corridor arbitrage when cost differs > 20% between providers
    /// PredicateSpec::PairwiseDifference {
    ///     input: "corridor_costs".into(),
    ///     threshold_pct: 20.0,
    ///     sharpness: 10.0,
    ///     output_mode: PairwiseOutput::MaxDifference,
    /// }
    /// ```
    PairwiseDifference {
        /// Input tensor name (shape [N] for N items to compare)
        input: String,
        /// Threshold percentage for significant difference
        threshold_pct: f32,
        /// Sharpness of threshold (higher = sharper cutoff)
        #[serde(default = "default_sharpness")]
        sharpness: f32,
        /// What to output from the pairwise comparison
        #[serde(default)]
        output_mode: PairwiseOutput,
    },

    /// Cascading lookup with priority-ordered fallback chain
    /// 
    /// For cost schedule lookup with wildcard fallback patterns.
    /// Given multiple candidate values and their existence masks,
    /// selects the first one that exists with priority weighting.
    /// 
    /// # Example
    /// ```ignore
    /// // Cost schedule lookup: exact → source_wild → dest_wild → global
    /// PredicateSpec::CascadingLookup {
    ///     candidates: vec![
    ///         "exact_schedule".into(),    // Full match (provider, endpoint, src, dst)
    ///         "src_wild".into(),          // (provider, endpoint, src, *)
    ///         "dst_wild".into(),          // (provider, endpoint, *, dst)
    ///         "global".into(),            // (provider, endpoint, *, *)
    ///     ],
    ///     exists: vec![
    ///         "exact_exists".into(),
    ///         "src_wild_exists".into(),
    ///         "dst_wild_exists".into(),
    ///         "global_exists".into(),
    ///     ],
    ///     priorities: vec![1.0, 0.9, 0.9, 0.8],
    /// }
    /// ```
    CascadingLookup {
        /// Candidate value tensor names (in priority order)
        candidates: Vec<String>,
        /// Existence mask tensor names (matching candidates order)
        /// Each should be a scalar or broadcastable boolean/fuzzy value
        exists: Vec<String>,
        /// Priority weights for each level (used for output weighting)
        priorities: Vec<f32>,
    },

    /// External (computed by user code, not compiled)
    External,
}

/// Output mode for pairwise difference predicate
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum PairwiseOutput {
    /// Output the maximum difference as a ratio (0-1 scaled by threshold)
    #[default]
    MaxDifference,
    /// Output 1.0 if any pair exceeds threshold, 0.0 otherwise
    AnyAboveThreshold,
    /// Output count of pairs above threshold (normalized by total pairs)
    CountAboveThreshold,
    /// Output index of best (lowest) item
    BestIndex,
    /// Output the full pairwise difference matrix [P, P] as percentage differences
    /// Useful for arbitrage detection where you need all provider comparisons
    AllPairs,
}

fn default_sharpness() -> f32 {
    10.0
}

/// Activation function for learned predicates
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum Activation {
    /// Rectified Linear Unit: max(0, x)
    #[default]
    ReLU,
    /// Gaussian Error Linear Unit
    GELU,
    /// Sigmoid: 1 / (1 + e^-x)
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// No activation (identity)
    None,
}

impl Activation {
    /// Apply this activation function to a tensor
    ///
    /// Routes to the corresponding implementation in [`crate::primitives`].
    pub fn apply(&self, tensor: &candle_core::Tensor) -> crate::Result<candle_core::Tensor> {
        match self {
            Self::ReLU => crate::primitives::relu(tensor),
            Self::GELU => crate::primitives::gelu(tensor),
            Self::Sigmoid => crate::primitives::sigmoid(tensor),
            Self::Tanh => crate::primitives::tanh(tensor),
            Self::None => Ok(tensor.clone()),
        }
    }
}

/// Conditioning type for FiLM and other modulation methods
/// 
/// FiLM (Feature-wise Linear Modulation) allows a conditioning vector
/// to modulate hidden activations via learned scale (gamma) and shift (beta):
///   output = gamma * hidden + beta
/// 
/// This enables regime-aware processing where the same features are
/// interpreted differently based on market regime.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum ConditioningType {
    /// Feature-wise Linear Modulation: gamma * x + beta
    /// Conditioning vector is projected to gamma and beta vectors
    /// gamma = Linear(cond_dim → hidden_dim) + 1.0 (centered at identity)
    /// beta = Linear(cond_dim → hidden_dim)
    #[default]
    FiLM,
    /// Additive only: x + beta (no scaling)
    Additive,
    /// Multiplicative only: gamma * x (no shift)
    Multiplicative,
    /// Gating: sigmoid(proj(cond)) * x (soft attention gate)
    Gating,
}

/// Similarity computation method
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum SimilarityMethod {
    /// Learned similarity: W·(a⊙b) + bias → sigmoid
    /// Most flexible, learns what "similar" means
    #[default]
    Learned,
    /// Cosine similarity: a·b / (|a||b|) → [0,1]
    Cosine,
    /// Normalized dot product
    DotNormalized,
    /// Bilinear: aᵀWb → sigmoid
    Bilinear,
}

/// Aggregation method for graph neural predicates
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum Aggregation {
    /// Mean of neighbor features (SAGE default)
    #[default]
    Mean,
    /// Sum of neighbor features (GCN-style)
    Sum,
    /// Max pooling over neighbors
    Max,
    /// Attention-weighted aggregation (GAT-style)
    Attention,
}

/// How to combine predicates in a composite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositeOp {
    /// Fuzzy AND (product t-norm)
    And,
    /// Fuzzy OR (probabilistic sum)
    Or,
    /// Fuzzy NOT (complement)
    Not,
}

/// A compiled rule ready for execution
///
/// The compiled rule is a differentiable function that:
/// 1. Takes input tensors
/// 2. Computes predicate activations
/// 3. Combines via fuzzy logic
/// 4. Returns output with gradient support
///
/// This is a thin wrapper around `CompiledRuleSet` for backwards compatibility.
#[derive(Debug)]
pub struct CompiledRule {
    /// Name of this compiled rule set
    pub name: String,

    /// Source specification (retained for debugging and serialization)
    pub source: RuleSpec,

    /// Total number of trainable parameters across all predicates
    pub param_count: usize,

    /// The actual compiled rule set (Arc for weight sharing with BatchedInferenceContext)
    inner: std::sync::Arc<codegen::CompiledRuleSet>,
}

impl CompiledRule {
    /// Compile a rule specification
    ///
    /// # Example
    ///
    /// ```ignore
    /// let spec = RuleSpec::parse("exit_rules", r#"
    ///     exit(X) :- profit_target(X, 0.02), momentum_shift(X).
    ///     exit(X) :- stop_loss(X, -0.01).
    /// "#)?;
    ///
    /// let compiled = CompiledRule::compile(spec)?;
    /// let output = compiled.forward(&inputs)?;
    /// ```
    pub fn compile(spec: RuleSpec) -> crate::Result<Self> {
        let device = crate::primitives::best_device();
        Self::compile_on_device(spec, &device)
    }

    /// Compile a rule specification on a specific device
    pub fn compile_on_device(spec: RuleSpec, device: &candle_core::Device) -> crate::Result<Self> {
        let inner = codegen::CompiledRuleSet::compile(spec.clone(), device)?;
        let param_count = inner.param_count();

        Ok(Self {
            name: spec.name.clone(),
            source: spec,
            param_count,
            inner: std::sync::Arc::new(inner),
        })
    }

    /// Forward pass: evaluate rules and return output with explanation
    ///
    /// # Arguments
    ///
    /// * `inputs` - Map of predicate/feature names to input tensors
    ///
    /// # Returns
    ///
    /// * `CompiledOutput` containing:
    ///   - `output`: The main result tensor
    ///   - `rule_activations`: Per-rule activation values
    ///   - `predicate_activations`: Per-predicate activation values
    ///   - `explanation`: Human-readable explanation
    pub fn forward(&self, inputs: &HashMap<String, candle_core::Tensor>) -> crate::Result<CompiledOutput> {
        self.inner.forward(inputs)
    }
    
    /// Fast forward pass: only returns output tensor, skips explanations
    /// 
    /// ~20x faster than `forward()` for hot paths. Use when you only need
    /// the output tensor and don't need rule activations or explanations.
    /// 
    /// # Example
    /// ```ignore
    /// // BEFORE (slow - ~300ms):
    /// let output = compiled.forward(&inputs)?;
    /// let prob = output.output.to_scalar::<f32>()?;
    /// 
    /// // AFTER (fast - ~15ms):
    /// let prob = compiled.forward_fast(&inputs)?.to_scalar::<f32>()?;
    /// ```
    #[inline]
    pub fn forward_fast(&self, inputs: &HashMap<String, candle_core::Tensor>) -> crate::Result<candle_core::Tensor> {
        self.inner.forward_fast(inputs)
    }
    
    /// Batch forward pass: evaluate multiple input sets in one GPU call
    /// 
    /// For backtesting with many symbols, this provides massive speedup
    /// by batching all evaluations into a single GPU kernel launch.
    /// 
    /// # Arguments
    /// * `batch_inputs` - Vector of input maps, one per sample
    /// 
    /// # Returns
    /// * Tensor of shape [batch_size, output_dim]
    /// 
    /// # Example
    /// ```ignore
    /// // Build inputs for all 102 symbols at this bar
    /// let batch_inputs: Vec<HashMap<String, Tensor>> = symbols
    ///     .iter()
    ///     .map(|sym| build_inputs_for_symbol(sym, bar))
    ///     .collect();
    /// 
    /// // One GPU call for all symbols
    /// let probs = compiled.forward_batch(&batch_inputs)?;
    /// ```
    #[inline]
    pub fn forward_batch(&self, batch_inputs: &[HashMap<String, candle_core::Tensor>]) -> crate::Result<candle_core::Tensor> {
        self.inner.forward_batch(batch_inputs)
    }
    
    /// Get the inner CompiledRuleSet reference
    /// 
    /// Use this when you need read-only access to the compiled rules.
    pub fn inner(&self) -> &codegen::CompiledRuleSet {
        &self.inner
    }
    
    /// Get a shared Arc reference to the inner CompiledRuleSet
    /// 
    /// Use this for weight sharing with `BatchedInferenceContext`:
    /// ```ignore
    /// // In TradingTLEngine
    /// pub fn get_exit_rules_arc(&self) -> Option<Arc<CompiledRuleSet>> {
    ///     self.exit_rules.as_ref().map(|r| r.inner_arc())
    /// }
    /// 
    /// // Create BatchedInferenceContext with shared weights
    /// let ctx = BatchedInferenceContext::new(
    ///     rules.inner_arc(),  // Same weights as original CompiledRule
    ///     &input_names,
    ///     batch_size,
    ///     &device,
    /// )?;
    /// ```
    /// 
    /// This enables both single-sample and batched inference to share
    /// the same trained weights in memory.
    pub fn inner_arc(&self) -> std::sync::Arc<codegen::CompiledRuleSet> {
        std::sync::Arc::clone(&self.inner)
    }
    
    /// Consume the CompiledRule and return the inner Arc<CompiledRuleSet>
    /// 
    /// Use with `BatchedInferenceContext::new()` for ownership transfer:
    /// ```ignore
    /// let rules = CompiledRule::compile(spec)?;
    /// let ctx = BatchedInferenceContext::new(rules.into_inner(), ...)?;
    /// ```
    pub fn into_inner(self) -> std::sync::Arc<codegen::CompiledRuleSet> {
        self.inner
    }

    /// Get trainable variables for optimization
    pub fn trainable_vars(&self) -> Vec<candle_core::Var> {
        self.inner.trainable_vars()
    }
    
    /// Get only FiLM conditioning variables (for separate learning rate)
    /// 
    /// FiLM layers often need 10x lower learning rate than main network.
    /// Use with `main_vars()` to create separate optimizer groups.
    /// 
    /// # Example
    /// ```ignore
    /// let film_vars = compiled.film_vars();
    /// let main_vars = compiled.main_vars();
    /// 
    /// // Create separate optimizers with different LRs
    /// let mut film_opt = Optimizer::adam(film_vars, 0.00003)?;  // 10x lower
    /// let mut main_opt = Optimizer::adam(main_vars, 0.0003)?;
    /// 
    /// // Training loop
    /// for batch in batches {
    ///     let grads = loss.backward()?;
    ///     safe_optimizer_step(&mut main_opt, &grads, &main_vars, 1.0, 0.0003)?;
    ///     safe_optimizer_step(&mut film_opt, &grads, &film_vars, 1.0, 0.00003)?;
    /// }
    /// ```
    pub fn film_vars(&self) -> Vec<candle_core::Var> {
        self.inner.film_vars()
    }
    
    /// Get main network variables (excludes FiLM conditioning)
    pub fn main_vars(&self) -> Vec<candle_core::Var> {
        self.inner.main_vars()
    }
    
    /// Check if this rule uses FiLM conditioning
    pub fn has_film(&self) -> bool {
        self.inner.has_film()
    }

    /// Get the head predicate name
    pub fn head(&self) -> &str {
        &self.inner.head
    }

    /// Get number of rules
    pub fn rule_count(&self) -> usize {
        self.inner.bodies.len()
    }

    /// Get predicate names used in this rule set
    pub fn predicate_names(&self) -> Vec<String> {
        self.inner.predicate_indices.keys().cloned().collect()
    }

    /// Save learned parameters to a safetensors file
    ///
    /// Saves all trainable variables (thresholds, projection weights, similarity weights,
    /// GNN layers) to a single file for persistence.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // After training
    /// compiled_rule.save("models/pipeline_rules_v1.safetensors")?;
    ///
    /// // Later, load and continue
    /// let mut loaded = CompiledRule::compile(spec)?;
    /// loaded.load("models/pipeline_rules_v1.safetensors")?;
    /// ```
    pub fn save(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        let vars = self.trainable_vars();
        if vars.is_empty() {
            return Err(crate::TensorCoreError::Compiler(
                "No trainable variables to save".into(),
            ));
        }

        // Collect tensors with named keys into HashMap
        let tensors: std::collections::HashMap<String, candle_core::Tensor> = vars
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("param_{}", i), v.as_tensor().clone()))
            .collect();

        // Serialize to file using candle's safetensors API
        candle_core::safetensors::save(&tensors, path.as_ref())
            .map_err(|e| crate::TensorCoreError::Serialization(format!("Failed to save: {}", e)))?;

        Ok(())
    }

    /// Load learned parameters from a safetensors file
    ///
    /// Loads trainable variables from a previously saved checkpoint.
    /// The rule specification must match (same predicates, same dimensions).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut compiled = CompiledRule::compile(spec)?;
    /// compiled.load("models/pipeline_rules_v1.safetensors")?;
    /// // Now using learned weights
    /// ```
    pub fn load(&mut self, path: impl AsRef<Path>) -> crate::Result<()> {
        let data = std::fs::read(path.as_ref())
            .map_err(|e| crate::TensorCoreError::Serialization(format!("Failed to read file: {}", e)))?;

        // Use candle's safetensors loader which handles dtype conversion properly
        let tensors = candle_core::safetensors::load_buffer(&data, &self.inner.device)
            .map_err(|e| crate::TensorCoreError::Serialization(format!("Failed to deserialize: {}", e)))?;

        let vars = self.inner.trainable_vars();
        
        for (i, var) in vars.iter().enumerate() {
            let name = format!("param_{}", i);
            
            let loaded_tensor = tensors.get(&name)
                .ok_or_else(|| crate::TensorCoreError::Serialization(
                    format!("Missing parameter '{}'", name)
                ))?;

            // Update the variable
            var.set(loaded_tensor)
                .map_err(|e| crate::TensorCoreError::Tensor(format!("Failed to set var: {}", e)))?;
        }

        Ok(())
    }

    /// Save with metadata (version, spec hash, training info)
    ///
    /// Extended save that includes metadata for version tracking
    /// and compatibility checking.
    pub fn save_with_metadata(
        &self,
        path: impl AsRef<Path>,
        metadata: RuleCheckpointMetadata,
    ) -> crate::Result<()> {
        let vars = self.trainable_vars();
        if vars.is_empty() {
            return Err(crate::TensorCoreError::Compiler(
                "No trainable variables to save".into(),
            ));
        }

        // Collect tensors with named keys into HashMap
        let tensors: std::collections::HashMap<String, candle_core::Tensor> = vars
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("param_{}", i), v.as_tensor().clone()))
            .collect();

        // Save tensors using candle's safetensors API
        candle_core::safetensors::save(&tensors, path.as_ref())
            .map_err(|e| crate::TensorCoreError::Serialization(format!("Failed to save: {}", e)))?;

        // Save metadata as sidecar JSON file
        let meta_path = path.as_ref().with_extension("meta.json");
        let meta_json = serde_json::json!({
            "name": metadata.name,
            "version": metadata.version,
            "spec_hash": metadata.spec_hash,
            "namespace": metadata.namespace,
            "trained_at": metadata.trained_at.to_rfc3339(),
            "param_count": self.param_count,
        });
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta_json).unwrap())
            .map_err(|e| crate::TensorCoreError::Serialization(format!("Failed to save metadata: {}", e)))?;

        Ok(())
    }

    /// Load and return metadata from a checkpoint
    ///
    /// Reads the sidecar `.meta.json` file created by `save_with_metadata`.
    /// Returns an error if the metadata file doesn't exist.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let meta = CompiledRule::load_metadata("models/pipeline_rules_v1.safetensors")?;
    /// println!("Version: {}", meta.version);
    /// ```
    pub fn load_metadata(path: impl AsRef<Path>) -> crate::Result<RuleCheckpointMetadata> {
        let meta_path = path.as_ref().with_extension("meta.json");

        let meta_json = std::fs::read_to_string(&meta_path)
            .map_err(|e| crate::TensorCoreError::Serialization(
                format!("Failed to read metadata file '{}': {}", meta_path.display(), e)
            ))?;

        let meta_value: serde_json::Value = serde_json::from_str(&meta_json)
            .map_err(|e| crate::TensorCoreError::Serialization(format!("Invalid metadata JSON: {}", e)))?;

        let get_str = |key: &str| -> String {
            meta_value.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };

        let trained_at = meta_value.get("trained_at")
            .and_then(|v| v.as_str())
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt: chrono::DateTime<chrono::FixedOffset>| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(chrono::Utc::now);

        Ok(RuleCheckpointMetadata {
            name: get_str("name"),
            version: get_str("version"),
            spec_hash: get_str("spec_hash"),
            namespace: meta_value.get("namespace").and_then(|v| v.as_str()).map(|s| s.to_string()),
            trained_at,
        })
    }

    /// Clamp attention weights to prevent gradient explosion
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
    ///     optimizer.step(&loss.backward()?)?;
    ///     
    ///     // Prevent weight explosion
    ///     compiled.clamp_attention_weights(10.0)?;
    /// }
    /// ```
    ///
    /// # Arguments
    ///
    /// * `max_val` - Maximum absolute value for attention weights (recommended: 10.0)
    pub fn clamp_attention_weights(&self, max_val: f32) -> crate::Result<()> {
        self.inner.clamp_attention_weights(max_val)
    }

    /// Check if any trainable variable contains NaN or Inf
    ///
    /// Useful for debugging training instability.
    ///
    /// # Returns
    ///
    /// `Some((var_index, issue))` if a problem is found, `None` if all OK.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some((idx, issue)) = compiled.check_weights_health() {
    ///     eprintln!("Warning: {} detected at weight index {}", issue, idx);
    /// }
    /// ```
    pub fn check_weights_health(&self) -> Option<(usize, &'static str)> {
        self.inner.check_weights_health()
    }
}

/// Metadata for rule checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCheckpointMetadata {
    /// Rule set name
    pub name: String,
    
    /// Version string (e.g., "v1.2.3")
    pub version: String,
    
    /// Hash of the rule specification (for compatibility checking)
    pub spec_hash: String,
    
    /// Namespace (for federated predicates)
    pub namespace: Option<String>,
    
    /// When this checkpoint was created
    pub trained_at: chrono::DateTime<chrono::Utc>,
}

impl RuleCheckpointMetadata {
    /// Create metadata for a compiled rule
    pub fn from_compiled(rule: &CompiledRule, version: &str) -> Self {
        Self {
            name: rule.name.clone(),
            version: version.to_string(),
            spec_hash: compute_spec_hash(&rule.source),
            namespace: None,
            trained_at: chrono::Utc::now(),
        }
    }

    /// Create metadata with namespace
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }
}

/// Compute a hash of a rule specification for compatibility checking
fn compute_spec_hash(spec: &RuleSpec) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    
    // Hash the essential structure
    spec.name.hash(&mut hasher);
    spec.rules.len().hash(&mut hasher);
    
    for rule in &spec.rules {
        rule.head.hash(&mut hasher);
        rule.body.len().hash(&mut hasher);
        for lit in &rule.body {
            lit.predicate.hash(&mut hasher);
            lit.negated.hash(&mut hasher);
        }
    }
    
    for (name, _spec) in &spec.predicates {
        name.hash(&mut hasher);
    }
    
    format!("{:016x}", hasher.finish())
}

/// A multi-head compiled rule set
///
/// Supports rule files with multiple different head predicates.
/// Each head is compiled separately and can be evaluated independently
/// or all at once.
///
/// # Example
///
/// ```ignore
/// let spec = RuleSpec::parse("policy", r#"
///     needs_human(X) :- critical_file(X).
///     high_quorum(X) :- high_risk(X), security_file(X).
///     escalate(X) :- needs_human(X), urgent(X).
/// "#)?;
///
/// let multi = MultiHeadCompiledRule::compile(spec)?;
///
/// // Evaluate all heads at once
/// let outputs = multi.forward_all(&inputs)?;
/// let needs_human = outputs.get("needs_human").unwrap();
/// let high_quorum = outputs.get("high_quorum").unwrap();
///
/// // Or evaluate a specific head
/// let escalate_output = multi.forward_head("escalate", &inputs)?;
/// ```
#[derive(Debug)]
pub struct MultiHeadCompiledRule {
    /// Name of this multi-head rule set
    pub name: String,

    /// Source specification
    pub source: RuleSpec,

    /// Compiled rule sets per head predicate
    heads: HashMap<String, codegen::CompiledRuleSet>,

    /// Device for tensor operations
    device: candle_core::Device,
}

impl MultiHeadCompiledRule {
    /// Compile a rule specification with multiple heads
    ///
    /// Groups rules by their head predicate and compiles each group separately.
    pub fn compile(spec: RuleSpec) -> crate::Result<Self> {
        let device = crate::primitives::best_device();
        Self::compile_on_device(spec, &device)
    }

    /// Compile on a specific device
    pub fn compile_on_device(spec: RuleSpec, device: &candle_core::Device) -> crate::Result<Self> {
        if spec.rules.is_empty() {
            return Err(crate::TensorCoreError::Compiler("No rules to compile".into()));
        }

        // Group rules by head
        let mut by_head: HashMap<String, Vec<Rule>> = HashMap::new();
        for rule in &spec.rules {
            by_head.entry(rule.head.clone()).or_default().push(rule.clone());
        }

        // Compile each head separately
        let mut heads = HashMap::new();
        for (head, rules) in by_head {
            let sub_spec = RuleSpec {
                name: format!("{}_{}", spec.name, head),
                version: spec.version.clone(),
                rules,
                predicates: spec.predicates.clone(),
            };
            let compiled = codegen::CompiledRuleSet::compile(sub_spec, device)?;
            heads.insert(head, compiled);
        }

        Ok(Self {
            name: spec.name.clone(),
            source: spec,
            heads,
            device: device.clone(),
        })
    }

    /// Forward pass for all heads at once
    ///
    /// Returns a map of head predicate → output.
    pub fn forward_all(
        &self,
        inputs: &HashMap<String, candle_core::Tensor>,
    ) -> crate::Result<HashMap<String, CompiledOutput>> {
        let mut outputs = HashMap::new();
        for (head, compiled) in &self.heads {
            let output = compiled.forward(inputs)?;
            outputs.insert(head.clone(), output);
        }
        Ok(outputs)
    }

    /// Forward pass for a specific head
    pub fn forward_head(
        &self,
        head: &str,
        inputs: &HashMap<String, candle_core::Tensor>,
    ) -> crate::Result<CompiledOutput> {
        let compiled = self.heads.get(head).ok_or_else(|| {
            crate::TensorCoreError::Compiler(format!("Unknown head predicate: {}", head))
        })?;
        compiled.forward(inputs)
    }

    /// Get all head predicate names
    pub fn head_names(&self) -> Vec<String> {
        self.heads.keys().cloned().collect()
    }

    /// Get number of heads
    pub fn head_count(&self) -> usize {
        self.heads.len()
    }

    /// Get trainable variables from all heads
    pub fn trainable_vars(&self) -> Vec<candle_core::Var> {
        self.heads.values().flat_map(|h| h.trainable_vars()).collect()
    }
    
    /// Get FiLM conditioning variables from all heads (for separate LR)
    pub fn film_vars(&self) -> Vec<candle_core::Var> {
        self.heads.values().flat_map(|h| h.film_vars()).collect()
    }
    
    /// Get main network variables from all heads (excludes FiLM)
    pub fn main_vars(&self) -> Vec<candle_core::Var> {
        self.heads.values().flat_map(|h| h.main_vars()).collect()
    }
    
    /// Check if any head uses FiLM conditioning
    pub fn has_film(&self) -> bool {
        self.heads.values().any(|h| h.has_film())
    }

    /// Total parameter count across all heads
    pub fn param_count(&self) -> usize {
        self.trainable_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum()
    }

    /// Get a specific head's compiled rule set
    pub fn get_head(&self, head: &str) -> Option<&codegen::CompiledRuleSet> {
        self.heads.get(head)
    }

    /// Save all heads to a directory (one file per head)
    pub fn save_all(&self, dir: impl AsRef<Path>) -> crate::Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)
            .map_err(|e| crate::TensorCoreError::Serialization(format!("Failed to create dir: {}", e)))?;

        for (head, compiled) in &self.heads {
            let path = dir.join(format!("{}.safetensors", head));
            
            let vars = compiled.trainable_vars();
            if vars.is_empty() {
                continue; // Skip heads with no trainable params
            }

            let tensors: std::collections::HashMap<String, candle_core::Tensor> = vars
                .iter()
                .enumerate()
                .map(|(i, v)| (format!("param_{}", i), v.as_tensor().clone()))
                .collect();

            candle_core::safetensors::save(&tensors, &path)
                .map_err(|e| crate::TensorCoreError::Serialization(format!("Failed to save {}: {}", head, e)))?;
        }

        Ok(())
    }

    /// Load all heads from a directory
    pub fn load_all(&mut self, dir: impl AsRef<Path>) -> crate::Result<()> {
        let dir = dir.as_ref();

        for (head, compiled) in &self.heads {
            let path = dir.join(format!("{}.safetensors", head));
            
            if !path.exists() {
                continue; // Skip missing files
            }

            let data = std::fs::read(&path)
                .map_err(|e| crate::TensorCoreError::Serialization(format!("Failed to read {}: {}", head, e)))?;

            // Use candle's safetensors loader
            let tensors = candle_core::safetensors::load_buffer(&data, &self.device)
                .map_err(|e| crate::TensorCoreError::Serialization(format!("Failed to deserialize {}: {}", head, e)))?;

            let vars = compiled.trainable_vars();
            for (i, var) in vars.iter().enumerate() {
                let name = format!("param_{}", i);
                
                if let Some(loaded_tensor) = tensors.get(&name) {
                    var.set(loaded_tensor)
                        .map_err(|e| crate::TensorCoreError::Tensor(format!("Failed to set var: {}", e)))?;
                }
            }
        }

        Ok(())
    }
}

/// Output of a compiled rule forward pass
#[derive(Debug)]
pub struct CompiledOutput {
    /// Main output tensor
    pub output: candle_core::Tensor,

    /// Individual rule activations (for explanation)
    pub rule_activations: HashMap<String, f32>,

    /// Predicate activations (for explanation)
    pub predicate_activations: HashMap<String, f32>,

    /// Human-readable explanation
    pub explanation: String,

    /// Learned rule weights (normalized importance of each rule)
    /// 
    /// For disjunctive rules (same head), these weights indicate
    /// the relative importance of each rule after training.
    /// Normalized to sum to 1.0.
    pub rule_weights: Option<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_spec_creation() {
        let mut spec = RuleSpec::new("exit_rules");

        spec.add_rule(Rule::new(
            "exit",
            vec![
                Literal::positive("profit_target", vec![
                    Argument::Variable("X".into()),
                    Argument::Constant(0.02),
                ]),
                Literal::positive("momentum_shift", vec![
                    Argument::Variable("X".into()),
                ]),
            ],
        ));

        spec.add_predicate("profit_target", PredicateSpec::Threshold {
            input: "unrealized_pnl".into(),
            threshold: 0.02,
            greater_than: true,
            sharpness: 10.0,
        });

        assert_eq!(spec.rules.len(), 1);
        assert_eq!(spec.predicates.len(), 1);
    }

    #[test]
    fn test_rule_with_negation() {
        let rule = Rule::new(
            "exit",
            vec![
                Literal::positive("regime_change", vec![Argument::Variable("X".into())]),
                Literal::negated("bullish", vec![Argument::Variable("X".into())]),
            ],
        );

        assert!(!rule.body[0].negated);
        assert!(rule.body[1].negated);
    }

    #[test]
    fn test_parse_simple_rule() {
        let result = RuleSpec::parse("test", "exit(X) :- profit(X, 0.02).");
        assert!(result.is_ok());
        let spec = result.unwrap();
        assert_eq!(spec.rules.len(), 1);
        assert_eq!(spec.rules[0].head, "exit");
    }

    #[test]
    fn test_parse_unicode_syntax() {
        let result = RuleSpec::parse("test", "exit(X) ← profit(X, 0.02) ∧ momentum(X)");
        assert!(result.is_ok());
        let spec = result.unwrap();
        assert_eq!(spec.rules.len(), 1);
        assert_eq!(spec.rules[0].body.len(), 2);
    }

    #[test]
    fn test_parse_multiple_rules() {
        let spec = RuleSpec::parse("exit_rules", r#"
            exit(X) :- profit_target(X, 0.02), momentum_shift(X).
            exit(X) :- stop_loss(X, -0.01).
            exit(X) :- regime_change(X), not bullish(X).
        "#).unwrap();

        assert_eq!(spec.name, "exit_rules");
        assert_eq!(spec.rules.len(), 3);
        
        // Third rule should have negation
        assert!(spec.rules[2].body[1].negated);
    }

    #[test]
    fn test_parse_disjunction() {
        let spec = RuleSpec::parse("test", "escalate(X) :- low_intent(X); high_risk(X).").unwrap();
        // Disjunction expands to multiple rules
        assert_eq!(spec.rules.len(), 2);
    }

    #[test]
    fn test_compile_and_forward() {
        let spec = RuleSpec::parse("test", "exit(X) :- profit(X), momentum(X).").unwrap();
        let compiled = CompiledRule::compile(spec).unwrap();

        // Create inputs
        let mut inputs = std::collections::HashMap::new();
        inputs.insert(
            "profit".to_string(),
            candle_core::Tensor::from_vec(vec![0.8f32], 1, &candle_core::Device::Cpu).unwrap(),
        );
        inputs.insert(
            "momentum".to_string(),
            candle_core::Tensor::from_vec(vec![0.9f32], 1, &candle_core::Device::Cpu).unwrap(),
        );

        let output = compiled.forward(&inputs).unwrap();

        // fuzzy_and(0.8, 0.9) = 0.72
        let result = output.output.to_vec1::<f32>().unwrap()[0];
        assert!((result - 0.72).abs() < 0.001);
        assert!(!output.explanation.is_empty());
    }

    // =========================================================================
    // P1: Save/Load Tests
    // =========================================================================

    #[test]
    fn test_save_load_basic() {
        // Create a compiled rule with learnable parameters
        let mut spec = RuleSpec::parse("test", "exit(X) :- momentum(X).").unwrap();
        spec.add_predicate("momentum", PredicateSpec::Learned { dim: 4 });
        
        let compiled = CompiledRule::compile(spec.clone()).unwrap();
        assert!(compiled.param_count > 0);

        // Save to temp file
        let temp_path = std::env::temp_dir().join("test_save_load.safetensors");
        compiled.save(&temp_path).unwrap();

        // Verify file exists
        assert!(temp_path.exists());

        // Load into a new compiled rule
        let mut loaded = CompiledRule::compile(spec).unwrap();
        loaded.load(&temp_path).unwrap();

        // Verify parameter counts match
        assert_eq!(compiled.param_count, loaded.param_count);

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_save_load_preserves_weights() {
        // Create a rule with learned projection
        let mut spec = RuleSpec::parse("test", "score(X) :- proj(X).").unwrap();
        spec.add_predicate("proj", PredicateSpec::LearnedProjection {
            inputs: vec!["embedding".into()],
            input_dim: 4,
            hidden_dim: 2,
            activation: Activation::ReLU,
            attention_heads: None,
            attention_dropout: None,
            layer_norm: None,
            dropout: None,
            residual: None,
            conditioning_dim: None,
            conditioning_type: None,
                film_identity_init: None,
        });
        
        let compiled = CompiledRule::compile(spec.clone()).unwrap();

        // Create test input
        let mut inputs = std::collections::HashMap::new();
        inputs.insert(
            "embedding".to_string(),
            candle_core::Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &candle_core::Device::Cpu).unwrap(),
        );

        // Get output before save
        let output_before = compiled.forward(&inputs).unwrap();
        let result_before = output_before.output.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];

        // Save
        let temp_path = std::env::temp_dir().join("test_preserve_weights.safetensors");
        compiled.save(&temp_path).unwrap();

        // Load into new instance
        let mut loaded = CompiledRule::compile(spec).unwrap();
        loaded.load(&temp_path).unwrap();

        // Get output after load
        let output_after = loaded.forward(&inputs).unwrap();
        let result_after = output_after.output.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];

        // Results should be identical
        assert!((result_before - result_after).abs() < 1e-6);

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_save_with_metadata() {
        let mut spec = RuleSpec::parse("pipeline_rules", "risk(X) :- score(X).").unwrap();
        spec.add_predicate("score", PredicateSpec::Learned { dim: 4 });
        
        let compiled = CompiledRule::compile(spec).unwrap();

        let metadata = RuleCheckpointMetadata::from_compiled(&compiled, "v1.0.0")
            .with_namespace("PIPELINE");

        let temp_path = std::env::temp_dir().join("test_metadata.safetensors");
        compiled.save_with_metadata(&temp_path, metadata).unwrap();

        // Load metadata back
        let loaded_meta = CompiledRule::load_metadata(&temp_path).unwrap();
        
        assert_eq!(loaded_meta.name, "pipeline_rules");
        assert_eq!(loaded_meta.version, "v1.0.0");
        assert_eq!(loaded_meta.namespace, Some("PIPELINE".to_string()));
        assert!(!loaded_meta.spec_hash.is_empty());

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_spec_hash_consistency() {
        let spec1 = RuleSpec::parse("test", "a(X) :- b(X), c(X).").unwrap();
        let spec2 = RuleSpec::parse("test", "a(X) :- b(X), c(X).").unwrap();
        let spec3 = RuleSpec::parse("test", "a(X) :- b(X).").unwrap();

        let hash1 = compute_spec_hash(&spec1);
        let hash2 = compute_spec_hash(&spec2);
        let hash3 = compute_spec_hash(&spec3);

        // Same spec should produce same hash
        assert_eq!(hash1, hash2);
        
        // Different spec should produce different hash
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_save_no_params_error() {
        // Rule with only external predicates (no learnable params)
        let spec = RuleSpec::parse("test", "a(X) :- b(X).").unwrap();
        let compiled = CompiledRule::compile(spec).unwrap();
        
        assert_eq!(compiled.param_count, 0);
        
        let temp_path = std::env::temp_dir().join("test_no_params.safetensors");
        let result = compiled.save(&temp_path);
        
        // Should fail because no trainable variables
        assert!(result.is_err());
    }

    // =========================================================================
    // P2a: Multi-Head Tests
    // =========================================================================

    #[test]
    fn test_multi_head_compile() {
        let spec = RuleSpec::parse("policy", r#"
            needs_human(X) :- critical_file(X).
            high_quorum(X) :- high_risk(X), security_file(X).
            escalate(X) :- needs_human(X), urgent(X).
        "#).unwrap();

        let multi = MultiHeadCompiledRule::compile(spec).unwrap();
        
        assert_eq!(multi.head_count(), 3);
        assert!(multi.head_names().contains(&"needs_human".to_string()));
        assert!(multi.head_names().contains(&"high_quorum".to_string()));
        assert!(multi.head_names().contains(&"escalate".to_string()));
    }

    #[test]
    fn test_multi_head_forward_all() {
        let spec = RuleSpec::parse("test", r#"
            a(X) :- x(X).
            b(X) :- y(X).
        "#).unwrap();

        let multi = MultiHeadCompiledRule::compile(spec).unwrap();

        let mut inputs = std::collections::HashMap::new();
        inputs.insert(
            "x".to_string(),
            candle_core::Tensor::from_vec(vec![0.7f32], 1, &candle_core::Device::Cpu).unwrap(),
        );
        inputs.insert(
            "y".to_string(),
            candle_core::Tensor::from_vec(vec![0.3f32], 1, &candle_core::Device::Cpu).unwrap(),
        );

        let outputs = multi.forward_all(&inputs).unwrap();
        
        assert_eq!(outputs.len(), 2);
        
        let a_result = outputs.get("a").unwrap().output.to_vec1::<f32>().unwrap()[0];
        let b_result = outputs.get("b").unwrap().output.to_vec1::<f32>().unwrap()[0];
        
        assert!((a_result - 0.7).abs() < 0.001);
        assert!((b_result - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_multi_head_forward_single() {
        let spec = RuleSpec::parse("test", r#"
            a(X) :- x(X).
            b(X) :- y(X).
        "#).unwrap();

        let multi = MultiHeadCompiledRule::compile(spec).unwrap();

        let mut inputs = std::collections::HashMap::new();
        inputs.insert(
            "x".to_string(),
            candle_core::Tensor::from_vec(vec![0.8f32], 1, &candle_core::Device::Cpu).unwrap(),
        );
        inputs.insert(
            "y".to_string(),
            candle_core::Tensor::from_vec(vec![0.2f32], 1, &candle_core::Device::Cpu).unwrap(),
        );

        // Only evaluate head "a"
        let output = multi.forward_head("a", &inputs).unwrap();
        let result = output.output.to_vec1::<f32>().unwrap()[0];
        
        assert!((result - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_multi_head_with_shared_predicates() {
        // Multiple heads using the same predicates
        let mut spec = RuleSpec::parse("test", r#"
            risk(X) :- score(X).
            alert(X) :- score(X), urgent(X).
        "#).unwrap();
        
        spec.add_predicate("score", PredicateSpec::Threshold {
            input: "value".into(),
            threshold: 0.5,
            greater_than: true,
            sharpness: 10.0,
        });

        let multi = MultiHeadCompiledRule::compile(spec).unwrap();
        
        let mut inputs = std::collections::HashMap::new();
        inputs.insert(
            "value".to_string(),
            candle_core::Tensor::from_vec(vec![0.8f32], 1, &candle_core::Device::Cpu).unwrap(),
        );
        inputs.insert(
            "urgent".to_string(),
            candle_core::Tensor::from_vec(vec![0.9f32], 1, &candle_core::Device::Cpu).unwrap(),
        );

        let outputs = multi.forward_all(&inputs).unwrap();
        
        // Both should use the same threshold predicate
        let risk = outputs.get("risk").unwrap().output.to_vec1::<f32>().unwrap()[0];
        let alert = outputs.get("alert").unwrap().output.to_vec1::<f32>().unwrap()[0];
        
        assert!(risk > 0.5);  // 0.8 > 0.5 threshold
        assert!(alert > 0.0); // Combined with urgent
    }

    #[test]
    fn test_multi_head_trainable_vars() {
        let mut spec = RuleSpec::parse("test", r#"
            a(X) :- learned_a(X).
            b(X) :- learned_b(X).
        "#).unwrap();
        
        spec.add_predicate("learned_a", PredicateSpec::Learned { dim: 4 });
        spec.add_predicate("learned_b", PredicateSpec::Learned { dim: 4 });

        let multi = MultiHeadCompiledRule::compile(spec).unwrap();
        
        // Each head has 1 learned predicate with dim 4: weights(4x1) + bias(1) = 5 params
        // Total: 10 params (but predicates are duplicated per head in current impl)
        assert!(multi.param_count() > 0);
        assert!(multi.trainable_vars().len() > 0);
    }

    #[test]
    fn test_multi_head_unknown_head_error() {
        let spec = RuleSpec::parse("test", "a(X) :- b(X).").unwrap();
        let multi = MultiHeadCompiledRule::compile(spec).unwrap();

        let inputs = std::collections::HashMap::new();
        let result = multi.forward_head("nonexistent", &inputs);
        
        assert!(result.is_err());
    }
}



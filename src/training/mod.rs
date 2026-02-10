//! Training Infrastructure
//!
//! Experience replay, optimization, and learning utilities.
//!
//! ## Overview
//!
//! This module provides training infrastructure that can be shared
//! across all Tensor Logic engines:
//!
//! - **Experience Replay**: Buffer management and prioritized sampling
//! - **Optimizers**: Wrapped candle-nn optimizers (AdamW, SGD)
//! - **Metrics**: Loss tracking and convergence monitoring
//!
//! ## Example
//!
//! ```ignore
//! use an_tensor_compiler::training::*;
//!
//! // Create optimizer
//! let optimizer = Optimizer::adam(vars, 0.001)?;
//!
//! // Training loop
//! for batch in replay_buffer.sample(32) {
//!     let loss = model.forward(&batch.features)?;
//!     let grads = loss.backward()?;
//!     optimizer.step(&grads)?;
//! }
//! ```

use crate::namespace::NamespaceId;
use candle_core::Var;
use candle_nn::optim::Optimizer as CandleOptimizer;
use serde::{Deserialize, Serialize};

// =============================================================================
// Types for an-ecosystem integration
// =============================================================================

/// Training metrics published after epochs
///
/// Used by an-ecosystem to publish to CRDT at `tensor/metrics/{namespace}/latest`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Namespace this training run belongs to
    pub namespace: NamespaceId,

    /// Current epoch number
    pub epoch: u64,

    /// Loss value for this epoch
    pub loss: f32,

    /// Accuracy (if applicable)
    pub accuracy: Option<f32>,

    /// Number of samples trained on
    pub samples_trained: usize,

    /// Duration of this epoch in milliseconds
    pub duration_ms: u64,
}

impl TrainingMetrics {
    /// Create new training metrics
    pub fn new(namespace: NamespaceId, epoch: u64, loss: f32) -> Self {
        Self {
            namespace,
            epoch,
            loss,
            accuracy: None,
            samples_trained: 0,
            duration_ms: 0,
        }
    }

    /// Builder: set accuracy
    pub fn with_accuracy(mut self, accuracy: f32) -> Self {
        self.accuracy = Some(accuracy);
        self
    }

    /// Builder: set samples trained
    pub fn with_samples(mut self, samples: usize) -> Self {
        self.samples_trained = samples;
        self
    }

    /// Builder: set duration
    pub fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }
}

/// Predicate-level metrics for detailed monitoring
///
/// Provides per-predicate breakdown for debugging and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateMetrics {
    /// Predicate name
    pub name: String,

    /// Namespace this predicate belongs to
    pub namespace: NamespaceId,

    /// Average activation value (0.0 to 1.0)
    pub mean_activation: f32,

    /// Activation standard deviation
    pub activation_std: f32,

    /// Gradient norm (if training)
    pub gradient_norm: Option<f32>,

    /// Number of evaluations
    pub eval_count: u64,
}

/// Outcome types for training signal
///
/// Matches the feedback types used by an-ecosystem HTTP API.
/// Used to provide supervised signal for threshold/predicate learning.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "PascalCase")]
pub enum TrainingOutcome {
    /// Deployed successfully, no issues
    Success,

    /// Deployed but failed in production
    FailedInProduction,

    /// Had to rollback after deployment
    RolledBack,

    /// Escalated to human, and human confirmed issue (correct escalation)
    EscalatedCorrectly,

    /// Escalated to human, but human said it was fine (unnecessary escalation)
    EscalatedUnnecessarily,

    /// Blocked deployment, and it would have been bad (correct rejection)
    RejectedCorrectly,

    /// Blocked deployment, but it would have been fine (incorrect rejection)
    RejectedIncorrectly,
}

impl TrainingOutcome {
    /// Returns training signal strength
    ///
    /// - Positive values (0.0 to 1.0): Good outcomes, reinforce current behavior
    /// - Negative values (-1.0 to 0.0): Bad outcomes, adjust thresholds/predicates
    ///
    /// Signal magnitudes:
    /// - ±1.0: Strongest signal (success vs production failure)
    /// - ±0.8: Strong signal (correct rejection/escalation vs rollback)
    /// - ±0.5: Moderate signal (unnecessary escalation)
    pub fn signal(&self) -> f32 {
        match self {
            Self::Success => 1.0,
            Self::EscalatedCorrectly => 0.8,
            Self::RejectedCorrectly => 0.8,
            Self::EscalatedUnnecessarily => -0.5,
            Self::RejectedIncorrectly => -0.8,
            Self::RolledBack => -0.8,
            Self::FailedInProduction => -1.0,
        }
    }

    /// Whether this outcome is positive (model made correct decision)
    pub fn is_positive(&self) -> bool {
        self.signal() > 0.0
    }

    /// Whether this outcome is negative (model made incorrect decision)
    pub fn is_negative(&self) -> bool {
        self.signal() < 0.0
    }

    /// Get all possible outcomes
    pub fn all() -> &'static [TrainingOutcome] {
        &[
            Self::Success,
            Self::FailedInProduction,
            Self::RolledBack,
            Self::EscalatedCorrectly,
            Self::EscalatedUnnecessarily,
            Self::RejectedCorrectly,
            Self::RejectedIncorrectly,
        ]
    }
}

impl std::fmt::Display for TrainingOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Success => write!(f, "Success"),
            Self::FailedInProduction => write!(f, "FailedInProduction"),
            Self::RolledBack => write!(f, "RolledBack"),
            Self::EscalatedCorrectly => write!(f, "EscalatedCorrectly"),
            Self::EscalatedUnnecessarily => write!(f, "EscalatedUnnecessarily"),
            Self::RejectedCorrectly => write!(f, "RejectedCorrectly"),
            Self::RejectedIncorrectly => write!(f, "RejectedIncorrectly"),
        }
    }
}

// =============================================================================
// Optimizers
// =============================================================================

/// Wrapper around candle-nn's AdamW optimizer
pub struct AdamOptimizer {
    inner: candle_nn::optim::AdamW,
}

impl AdamOptimizer {
    /// Create a new Adam optimizer
    pub fn new(vars: Vec<Var>, learning_rate: f64) -> crate::Result<Self> {
        let params = candle_nn::optim::ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        };

        let inner = <candle_nn::optim::AdamW as CandleOptimizer>::new(vars, params)
            .map_err(|e| crate::TensorCoreError::Tensor(format!("AdamW init failed: {}", e)))?;

        Ok(Self { inner })
    }

    /// Take a gradient step
    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> crate::Result<()> {
        CandleOptimizer::step(&mut self.inner, grads)
            .map_err(|e| crate::TensorCoreError::Tensor(format!("Adam step failed: {}", e)))
    }

    /// Get the learning rate
    pub fn learning_rate(&self) -> f64 {
        CandleOptimizer::learning_rate(&self.inner)
    }

    /// Set the learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        CandleOptimizer::set_learning_rate(&mut self.inner, lr)
    }
}

/// Wrapper around candle-nn's SGD optimizer
pub struct SGDOptimizer {
    inner: candle_nn::optim::SGD,
}

impl SGDOptimizer {
    /// Create a new SGD optimizer
    pub fn new(vars: Vec<Var>, learning_rate: f64) -> crate::Result<Self> {
        let inner = <candle_nn::optim::SGD as CandleOptimizer>::new(vars, learning_rate)
            .map_err(|e| crate::TensorCoreError::Tensor(format!("SGD init failed: {}", e)))?;

        Ok(Self { inner })
    }

    /// Take a gradient step
    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> crate::Result<()> {
        CandleOptimizer::step(&mut self.inner, grads)
            .map_err(|e| crate::TensorCoreError::Tensor(format!("SGD step failed: {}", e)))
    }

    /// Get the learning rate
    pub fn learning_rate(&self) -> f64 {
        CandleOptimizer::learning_rate(&self.inner)
    }

    /// Set the learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        CandleOptimizer::set_learning_rate(&mut self.inner, lr)
    }
}

/// Optimizer enum for flexibility
pub enum Optimizer {
    /// Stochastic Gradient Descent optimizer
    SGD(SGDOptimizer),
    /// Adam optimizer with weight decay (AdamW)
    Adam(AdamOptimizer),
}

impl Optimizer {
    /// Create SGD optimizer
    pub fn sgd(vars: Vec<Var>, learning_rate: f64) -> crate::Result<Self> {
        Ok(Self::SGD(SGDOptimizer::new(vars, learning_rate)?))
    }

    /// Create Adam optimizer
    pub fn adam(vars: Vec<Var>, learning_rate: f64) -> crate::Result<Self> {
        Ok(Self::Adam(AdamOptimizer::new(vars, learning_rate)?))
    }

    /// Take a gradient step
    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> crate::Result<()> {
        match self {
            Self::SGD(opt) => opt.step(grads),
            Self::Adam(opt) => opt.step(grads),
        }
    }

    /// Get the learning rate
    pub fn learning_rate(&self) -> f64 {
        match self {
            Self::SGD(opt) => opt.learning_rate(),
            Self::Adam(opt) => opt.learning_rate(),
        }
    }

    /// Set the learning rate
    pub fn set_learning_rate(&mut self, lr: f64) {
        match self {
            Self::SGD(opt) => opt.set_learning_rate(lr),
            Self::Adam(opt) => opt.set_learning_rate(lr),
        }
    }
}

// =============================================================================
// Variable Groups (for FiLM and other specialized layers)
// =============================================================================

/// A group of variables with shared hyperparameters
///
/// Use for different learning rates on different parts of the network.
/// Common pattern: FiLM layers need 10x lower LR than main network.
#[derive(Debug, Clone)]
pub struct VarGroup {
    /// Variables in this group
    pub vars: Vec<Var>,
    /// Learning rate for this group
    pub learning_rate: f64,
    /// Weight decay (L2 regularization) for this group
    pub weight_decay: f64,
    /// Group name (for logging)
    pub name: String,
}

impl VarGroup {
    /// Create a new variable group
    pub fn new(name: impl Into<String>, vars: Vec<Var>, learning_rate: f64) -> Self {
        Self {
            vars,
            learning_rate,
            weight_decay: 0.0,
            name: name.into(),
        }
    }

    /// Add weight decay (L2 regularization)
    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

/// Optimizer with multiple variable groups, each with its own hyperparameters
///
/// # Example
///
/// ```ignore
/// use an_tensor_compiler::training::{VarGroup, GroupedOptimizer};
///
/// let main_vars = compiled.main_vars();
/// let film_vars = compiled.film_vars();
///
/// let groups = vec![
///     VarGroup::new("main", main_vars, 0.0003).with_weight_decay(1e-4),
///     VarGroup::new("film", film_vars, 0.00003).with_weight_decay(1e-3),
/// ];
///
/// let mut optimizer = GroupedOptimizer::adam(groups)?;
///
/// // Training loop
/// for batch in batches {
///     let loss = forward_and_loss(&batch)?;
///     let grads = loss.backward()?;
///     optimizer.step(&grads)?;
/// }
/// ```
pub struct GroupedOptimizer {
    groups: Vec<(VarGroup, candle_nn::optim::AdamW)>,
}

impl GroupedOptimizer {
    /// Create a grouped Adam optimizer
    pub fn adam(groups: Vec<VarGroup>) -> crate::Result<Self> {
        let mut opt_groups = Vec::new();

        for group in groups {
            let params = candle_nn::optim::ParamsAdamW {
                lr: group.learning_rate,
                weight_decay: group.weight_decay,
                ..Default::default()
            };

            let optimizer =
                <candle_nn::optim::AdamW as CandleOptimizer>::new(group.vars.clone(), params)
                    .map_err(|e| {
                        crate::TensorCoreError::Tensor(format!(
                            "AdamW init failed for group '{}': {}",
                            group.name, e
                        ))
                    })?;

            opt_groups.push((group, optimizer));
        }

        Ok(Self { groups: opt_groups })
    }

    /// Take a gradient step on all groups
    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> crate::Result<()> {
        for (group, opt) in &mut self.groups {
            CandleOptimizer::step(opt, grads).map_err(|e| {
                crate::TensorCoreError::Training(format!(
                    "Step failed for group '{}': {}",
                    group.name, e
                ))
            })?;
        }
        Ok(())
    }

    /// Safe step with gradient clipping (recommended for attention/FiLM)
    pub fn safe_step(
        &mut self,
        grads: &candle_core::backprop::GradStore,
        max_grad_norm: f32,
    ) -> crate::Result<()> {
        for (group, opt) in &mut self.groups {
            safe_optimizer_step(opt, grads, &group.vars, max_grad_norm, group.learning_rate)?;
        }
        Ok(())
    }

    /// Get learning rate for a named group
    pub fn learning_rate(&self, group_name: &str) -> Option<f64> {
        self.groups
            .iter()
            .find(|(g, _)| g.name == group_name)
            .map(|(g, _)| g.learning_rate)
    }

    /// Set learning rate for a named group
    pub fn set_learning_rate(&mut self, group_name: &str, lr: f64) {
        for (group, opt) in &mut self.groups {
            if group.name == group_name {
                group.learning_rate = lr;
                CandleOptimizer::set_learning_rate(opt, lr);
            }
        }
    }

    /// Get all group names
    pub fn group_names(&self) -> Vec<&str> {
        self.groups.iter().map(|(g, _)| g.name.as_str()).collect()
    }
}

/// Configuration for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Learning rate
    pub learning_rate: f64,

    /// Batch size for training
    pub batch_size: usize,

    /// Minimum samples before training
    pub min_samples: usize,

    /// Training frequency (every N outcomes)
    pub train_every: usize,

    /// Gradient clipping threshold
    pub grad_clip: f32,

    /// Use AdamW (true) or SGD (false)
    pub use_adam: bool,

    /// Weight decay for regularization
    pub weight_decay: f64,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            min_samples: 100,
            train_every: 10,
            grad_clip: 1.0,
            use_adam: true,
            weight_decay: 0.01,
        }
    }
}

/// Result of a training step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Loss value
    pub loss: f32,

    /// Number of samples processed
    pub samples: usize,

    /// Gradient norms (for monitoring)
    pub grad_norms: Vec<f32>,

    /// Training step number
    pub step: usize,
}

/// Training metrics tracker
#[derive(Debug, Default)]
pub struct MetricsTracker {
    /// Loss history
    losses: Vec<f32>,

    /// Best loss seen
    best_loss: Option<f32>,

    /// Steps since improvement
    steps_since_improvement: usize,
}

impl MetricsTracker {
    /// Create a new metrics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a training result
    pub fn record(&mut self, result: &TrainingResult) {
        self.losses.push(result.loss);

        if self.best_loss.is_none() || result.loss < self.best_loss.unwrap() {
            self.best_loss = Some(result.loss);
            self.steps_since_improvement = 0;
        } else {
            self.steps_since_improvement += 1;
        }
    }

    /// Get average loss over last N steps
    pub fn average_loss(&self, n: usize) -> Option<f32> {
        if self.losses.is_empty() {
            return None;
        }

        let start = self.losses.len().saturating_sub(n);
        let slice = &self.losses[start..];
        Some(slice.iter().sum::<f32>() / slice.len() as f32)
    }

    /// Check if training has converged (no improvement for N steps)
    pub fn has_converged(&self, patience: usize) -> bool {
        self.steps_since_improvement >= patience
    }

    /// Get best loss
    pub fn best_loss(&self) -> Option<f32> {
        self.best_loss
    }

    /// Get total training steps
    pub fn total_steps(&self) -> usize {
        self.losses.len()
    }
}

// =============================================================================
// Gradient Utilities
// =============================================================================

/// Safe optimizer step with gradient clipping
///
/// **This is the recommended way to train models with attention.**
/// It clips gradients before applying them to prevent NaN weights.
///
/// When gradients contain NaN/Inf, the step is skipped and an error is returned.
/// When gradient norm exceeds `max_grad_norm`, gradients are scaled down before
/// applying the update.
///
/// # Arguments
///
/// * `optimizer` - Any Candle optimizer (Adam, SGD, etc.)
/// * `grads` - Gradient store from `loss.backward()`
/// * `vars` - Trainable variables (from `compiled.trainable_vars()`)
/// * `max_grad_norm` - Maximum allowed gradient norm (recommended: 1.0 for attention)
/// * `learning_rate` - Learning rate for manual update when clipping
///
/// # Example
///
/// ```ignore
/// use an_tensor_compiler::training::safe_optimizer_step;
/// use an_tensor_compiler::compiler::CompiledRule;
///
/// let mut optimizer = Optimizer::adam(vars.clone(), 0.001)?;
///
/// for epoch in 0..100 {
///     let output = compiled.forward(&inputs)?;
///     let loss = compute_loss(&output)?;
///     let grads = loss.backward()?;
///     
///     // Use safe step instead of optimizer.step()
///     match safe_optimizer_step(&mut optimizer, &grads, &vars, 1.0, 0.001) {
///         Ok(()) => {},
///         Err(e) if e.to_string().contains("NaN") => {
///             warn!("Skipping step due to NaN gradients");
///             continue;
///         }
///         Err(e) => return Err(e),
///     }
///     
///     // Clamp weights after step to prevent drift
///     compiled.clamp_attention_weights(10.0)?;
/// }
/// ```
pub fn safe_optimizer_step<O: candle_nn::optim::Optimizer>(
    optimizer: &mut O,
    grads: &candle_core::backprop::GradStore,
    vars: &[candle_core::Var],
    max_grad_norm: f32,
    learning_rate: f64,
) -> crate::Result<()> {
    // Compute total gradient norm
    let mut total_sq_norm = 0.0f32;
    let mut has_nan = false;

    for var in vars {
        if let Some(grad) = grads.get(var.as_tensor()) {
            // Check for NaN in gradient
            if let Ok(vals) = grad.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                for v in &vals {
                    if v.is_nan() || v.is_infinite() {
                        has_nan = true;
                        break;
                    }
                }
            }

            if has_nan {
                break;
            }

            let sq_norm = grad
                .sqr()
                .and_then(|t| t.sum_all())
                .and_then(|t| t.to_scalar::<f32>())
                .unwrap_or(0.0);
            total_sq_norm += sq_norm;
        }
    }

    // If gradients contain NaN, skip this step entirely
    if has_nan {
        return Err(crate::TensorCoreError::Training(
            "NaN detected in gradients - skipping step".into(),
        ));
    }

    let total_norm = total_sq_norm.sqrt();

    // If gradient norm is too large, scale down manually
    if total_norm > max_grad_norm && total_norm > 0.0 {
        let scale = max_grad_norm / total_norm;

        // Manually apply scaled gradient update
        for var in vars {
            if let Some(grad) = grads.get(var.as_tensor()) {
                let scaled_grad = grad
                    .affine(scale as f64, 0.0)
                    .map_err(|e| crate::TensorCoreError::Tensor(format!("scale failed: {}", e)))?;

                let current = var.as_tensor().clone();
                let updated = (&current
                    - &scaled_grad.affine(learning_rate, 0.0).map_err(|e| {
                        crate::TensorCoreError::Tensor(format!("lr scale failed: {}", e))
                    })?)
                    .map_err(|e| crate::TensorCoreError::Tensor(format!("sub failed: {}", e)))?;

                var.set(&updated)
                    .map_err(|e| crate::TensorCoreError::Tensor(format!("set failed: {}", e)))?;
            }
        }

        Ok(())
    } else {
        // Normal step
        optimizer
            .step(grads)
            .map_err(|e| crate::TensorCoreError::Training(format!("optimizer step failed: {}", e)))
    }
}

/// Safe optimizer step with gradient clipping (using optimizer's learning rate)
///
/// Convenience version of `safe_optimizer_step` that uses a default learning rate of 0.001
/// when clipping is needed. Use the full version if you need a different LR.
pub fn safe_optimizer_step_default<O: candle_nn::optim::Optimizer>(
    optimizer: &mut O,
    grads: &candle_core::backprop::GradStore,
    vars: &[candle_core::Var],
    max_grad_norm: f32,
) -> crate::Result<()> {
    safe_optimizer_step(optimizer, grads, vars, max_grad_norm, 0.001)
}

/// Compute the total L2 norm of all gradients
///
/// Useful for monitoring gradient explosion during training.
pub fn compute_grad_norm(
    grads: &candle_core::backprop::GradStore,
    vars: &[candle_core::Var],
) -> crate::Result<f32> {
    let mut total_sq_norm = 0.0f32;

    for var in vars {
        if let Some(grad) = grads.get(var.as_tensor()) {
            let sq_norm = grad
                .sqr()
                .and_then(|t| t.sum_all())
                .and_then(|t| t.to_scalar::<f32>())
                .unwrap_or(0.0);
            total_sq_norm += sq_norm;
        }
    }

    Ok(total_sq_norm.sqrt())
}

/// Check if any gradient contains NaN or Inf
///
/// Returns true if gradients are healthy (no NaN/Inf).
pub fn check_gradients_health(
    grads: &candle_core::backprop::GradStore,
    vars: &[candle_core::Var],
) -> bool {
    for var in vars {
        if let Some(grad) = grads.get(var.as_tensor()) {
            if let Ok(vals) = grad.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                for v in &vals {
                    if v.is_nan() || v.is_infinite() {
                        return false;
                    }
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_config_default() {
        let config = LearningConfig::default();
        assert_eq!(config.learning_rate, 0.001);
        assert!(config.use_adam);
    }

    #[test]
    fn test_training_outcome_signals() {
        // Positive outcomes
        assert!(TrainingOutcome::Success.signal() > 0.0);
        assert!(TrainingOutcome::EscalatedCorrectly.signal() > 0.0);
        assert!(TrainingOutcome::RejectedCorrectly.signal() > 0.0);

        // Negative outcomes
        assert!(TrainingOutcome::FailedInProduction.signal() < 0.0);
        assert!(TrainingOutcome::RolledBack.signal() < 0.0);
        assert!(TrainingOutcome::EscalatedUnnecessarily.signal() < 0.0);
        assert!(TrainingOutcome::RejectedIncorrectly.signal() < 0.0);

        // Strongest signals
        assert_eq!(TrainingOutcome::Success.signal(), 1.0);
        assert_eq!(TrainingOutcome::FailedInProduction.signal(), -1.0);
    }

    #[test]
    fn test_training_outcome_predicates() {
        assert!(TrainingOutcome::Success.is_positive());
        assert!(!TrainingOutcome::Success.is_negative());

        assert!(TrainingOutcome::FailedInProduction.is_negative());
        assert!(!TrainingOutcome::FailedInProduction.is_positive());
    }

    #[test]
    fn test_training_outcome_all() {
        let all = TrainingOutcome::all();
        assert_eq!(all.len(), 7);
    }

    #[test]
    fn test_training_outcome_display() {
        assert_eq!(TrainingOutcome::Success.to_string(), "Success");
        assert_eq!(
            TrainingOutcome::FailedInProduction.to_string(),
            "FailedInProduction"
        );
    }

    #[test]
    fn test_training_outcome_serde() {
        let outcome = TrainingOutcome::EscalatedCorrectly;
        let json = serde_json::to_string(&outcome).unwrap();
        assert_eq!(json, "\"EscalatedCorrectly\"");

        let parsed: TrainingOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, outcome);
    }

    #[test]
    fn test_training_metrics_builder() {
        use crate::namespace::PIPELINE;

        let metrics = TrainingMetrics::new(PIPELINE, 10, 0.15)
            .with_accuracy(0.92)
            .with_samples(1000)
            .with_duration_ms(5000);

        assert_eq!(metrics.namespace, PIPELINE);
        assert_eq!(metrics.epoch, 10);
        assert!((metrics.loss - 0.15).abs() < 0.001);
        assert_eq!(metrics.accuracy, Some(0.92));
        assert_eq!(metrics.samples_trained, 1000);
        assert_eq!(metrics.duration_ms, 5000);
    }

    #[test]
    fn test_training_metrics_serde() {
        use crate::namespace::TRADING;

        let metrics = TrainingMetrics::new(TRADING, 5, 0.25).with_accuracy(0.88);

        let json = serde_json::to_string(&metrics).unwrap();
        let parsed: TrainingMetrics = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.namespace, TRADING);
        assert_eq!(parsed.epoch, 5);
        assert_eq!(parsed.accuracy, Some(0.88));
    }

    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new();

        tracker.record(&TrainingResult {
            loss: 1.0,
            samples: 32,
            grad_norms: vec![],
            step: 0,
        });

        tracker.record(&TrainingResult {
            loss: 0.5,
            samples: 32,
            grad_norms: vec![],
            step: 1,
        });

        assert_eq!(tracker.best_loss(), Some(0.5));
        assert_eq!(tracker.total_steps(), 2);
        assert!((tracker.average_loss(2).unwrap() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_convergence_detection() {
        let mut tracker = MetricsTracker::new();

        // Improving
        for i in 0..5 {
            tracker.record(&TrainingResult {
                loss: 1.0 - i as f32 * 0.1,
                samples: 32,
                grad_norms: vec![],
                step: i,
            });
        }
        assert!(!tracker.has_converged(3));

        // Plateau
        for i in 5..10 {
            tracker.record(&TrainingResult {
                loss: 0.7, // Same loss, no improvement
                samples: 32,
                grad_norms: vec![],
                step: i,
            });
        }
        assert!(tracker.has_converged(3));
    }
}

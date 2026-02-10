#![warn(missing_docs)]

//! # an-tensor-compiler
//!
//! Differentiable neuro-symbolic compiler: symbolic logic rules → GPU-accelerated
//! tensor equations with full gradient flow.
//!
//! ## Overview
//!
//! This crate compiles human-readable symbolic rules into differentiable tensor
//! functions that run on GPU with automatic differentiation. It bridges symbolic AI
//! (logic, rules, constraints) with subsymbolic AI (neural networks, gradient descent).
//!
//! Core capabilities:
//!
//! - **Compiler**: Prolog-like rule DSL → differentiable tensor equations
//! - **Primitives**: Fuzzy logic operators with gradient flow
//! - **Holographic**: Distributed representations via holographic reduced representations
//! - **CRDTs**: Conflict-free tensor synchronization across devices
//! - **Namespace**: Gradient isolation between domains
//! - **Federation**: Cross-domain parameter sharing and merge algorithms
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use an_tensor_compiler::prelude::*;
//!
//! // Write rules in Prolog-like syntax
//! let spec = RuleSpec::parse("policy_rules", r#"
//!     escalate(X) :- high_risk(X), not approved(X).
//!     escalate(X) :- critical_resource(X), low_confidence(X).
//!     approve(X) :- all_checks_pass(X), high_confidence(X).
//! "#)?;
//!
//! // Compile to differentiable tensor function
//! let compiled = CompiledRule::compile(spec)?;
//!
//! // Forward pass with gradient flow
//! let output = compiled.forward(&inputs)?;
//! println!("Decision: {:.3}", output.output.to_scalar::<f32>()?);
//! println!("Explanation: {}", output.explanation);
//! ```
//!
//! ## Feature Flags
//!
//! - `metal`: Apple Metal GPU acceleration (M1/M2/M3/M4)
//! - `cuda`: NVIDIA CUDA GPU acceleration

pub mod compiler;
pub mod crdt;
pub mod federation;
pub mod holographic;
pub mod namespace;
pub mod primitives;
pub mod training;

// Re-export candle types for convenience
pub use candle_core::{DType, Device, Tensor, Var};

/// Error types for tensor compiler operations
#[derive(Debug, thiserror::Error)]
pub enum TensorCoreError {
    /// A tensor operation (matmul, element-wise, etc.) failed
    #[error("Tensor operation failed: {0}")]
    Tensor(String),

    /// Invalid configuration was provided
    #[error("Invalid configuration: {0}")]
    Config(String),

    /// A namespace operation failed (gradient isolation boundary violation, etc.)
    #[error("Namespace error: {0}")]
    Namespace(String),

    /// Rule compilation failed (parsing, validation, or codegen error)
    #[error("Compiler error: {0}")]
    Compiler(String),

    /// A federation operation failed (merge, sync, transport)
    #[error("Federation error: {0}")]
    Federation(String),

    /// Serialization or deserialization failed (safetensors, JSON, etc.)
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// A training operation failed (optimizer step, gradient computation)
    #[error("Training error: {0}")]
    Training(String),

    /// An I/O operation failed
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// An underlying candle tensor operation failed
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Result type alias for tensor compiler operations
pub type Result<T> = std::result::Result<T, TensorCoreError>;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{DType, Device, Tensor, Var};
    pub use crate::{Result, TensorCoreError};

    // Primitives
    pub use crate::primitives::{
        // Device selection
        best_device,
        binary_cross_entropy,
        // Tensor operations
        cosine_similarity,
        cpu_device,
        cuda_available,
        // Fuzzy logic operators
        fuzzy_and,
        fuzzy_and_many,
        fuzzy_implies,
        fuzzy_not,
        fuzzy_or,
        fuzzy_or_many,
        gelu,
        gpu_available,
        gpu_disabled,
        hard_threshold,
        // Additional activations
        leaky_relu,
        metal_available,
        mse_loss,
        relu,
        sigmoid,
        soft_threshold,
        soft_threshold_scalar,
        softmax,
        tanh,
        thread_local_device,
        weighted_rule_combination,
        with_gpu_sync,
    };

    // Namespace
    pub use crate::namespace::{
        NamespaceId, NamespaceRegistry, NamespacedTensor, CHAT, PIPELINE, TRADING,
    };

    // Compiler
    pub use crate::compiler::{CompiledRule, RuleSpec};

    // Federation
    pub use crate::federation::{FederationConfig, MergeStrategy};

    // Training utilities
    pub use crate::training::{
        check_gradients_health, compute_grad_norm, safe_optimizer_step, TrainingMetrics,
        TrainingOutcome,
    };

    // Holographic
    pub use crate::holographic::{bind, project, superimpose, unbind};
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;

        let device = best_device();
        assert!(
            matches!(device, Device::Cpu)
                || matches!(device, Device::Metal(_))
                || matches!(device, Device::Cuda(_))
        );
    }
}

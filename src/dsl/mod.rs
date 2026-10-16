//! # DSL Module
//!
//! Domain-Specific Language absorption layer for ecosystem projects.
//!
//! This module provides parsers and compilers for external DSLs (YAML/JSON)
//! that compile to tensor execution graphs. The design follows the absorption
//! strategy: take 80-90% of syntax straight, tensorize 10-20% of semantics.
//!
//! ## DSL Families
//!
//! - [`workflow`]: Workflow orchestration (Saga/DAG/Linear/Chain patterns)
//! - [`tool`]: Tool pattern matching and strategy learning
//! - [`context`]: Context management with temporal decay
//! - [`executor`]: Execution traits for tensor-native operations
//!
//! ## Architecture
//!
//! ```text
//! YAML/JSON Input
//!       │
//!       ▼
//! ┌─────────────┐
//! │   Parser    │  ← Absorb syntax straight
//! └─────────────┘
//!       │
//!       ▼
//! ┌─────────────┐
//! │  Validator  │  ← Semantic checks
//! └─────────────┘
//!       │
//!       ▼
//! ┌─────────────┐
//! │  Compiler   │  ← Tensorize execution
//! └─────────────┘
//!       │
//!       ▼
//! ┌─────────────┐
//! │ TensorGraph │  ← GPU-ready execution
//! └─────────────┘
//! ```
//!
//! ## Namespaces
//!
//! All DSL outputs use hierarchical namespace paths:
//! - `an-tensor-compiler.workflows.compiled.*`
//! - `an-tensor-compiler.tools.patterns.*`
//! - `an-tensor-compiler.context.dynamic.*`
//!
//! ## Soft-ification
//!
//! Hard concepts are converted to differentiable tensors:
//! - IDs → Hash embeddings
//! - Enums → One-hot tensors
//! - Conditionals → Soft gates (sigmoid)

pub mod common;
pub mod context;
pub mod executor;
pub mod tool;
pub mod workflow;

pub use common::{DslError, DslResult};
pub use context::{CoreContext, DecayAlgorithm, DynamicNode};
pub use executor::{
    ActivationContext, DefaultContextManager, DefaultPatternMatcher, DefaultWorkflowExecutor,
    PatternMatch, TensorBatch, TensorContextManager, TensorPatternMatcher, TensorWorkflowExecutor,
    ToolOutcome, WorkflowResult,
};
pub use tool::{ExecutionStrategy, ToolPattern, ToolSpec};
pub use workflow::{TensorGraph, WorkflowPattern, WorkflowSpec};

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_exports() {
        // Verify module structure compiles
        // Just importing the types to ensure they're accessible
        use super::{CoreContext, DslError, ToolPattern, WorkflowSpec};
        let _ = std::any::type_name::<WorkflowSpec>();
        let _ = std::any::type_name::<ToolPattern>();
        let _ = std::any::type_name::<CoreContext>();
        let _ = std::any::type_name::<DslError>();
    }
}

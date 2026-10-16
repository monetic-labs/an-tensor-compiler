//! # Interaction Primitives
//!
//! Shared predicates and rules for conversational AI across all organisms.
//!
//! This module provides tensor-native building blocks for:
//! - **Safety**: Message filtering, content moderation
//! - **Context**: Freshness, relevance, drift detection  
//! - **Intent**: Recognition, clarity assessment
//! - **Quality**: Response coherence, user satisfaction
//!
//! ## Design Philosophy
//!
//! Interaction primitives are **composable building blocks**, not complete solutions.
//! Each organism combines these with domain-specific rules to build their
//! conversational AI systems.
//!
//! ## Example
//!
//! ```ignore
//! use an_tensor_compiler::interaction::prelude::*;
//!
//! // Create interaction context
//! let ctx = InteractionContext::new(&device)?;
//!
//! // Register standard safety predicates
//! ctx.register_standard_predicates()?;
//!
//! // Compile safety rules
//! let rules = ctx.compile_safety_rules()?;
//!
//! // Evaluate message
//! let mut inputs = HashMap::new();
//! inputs.insert("content_embedding".into(), message_embedding);
//! let output = rules.forward(&inputs)?;
//!
//! if output.is_safe() {
//!     process_message(&message);
//! }
//! ```
//!
//! ## Namespace Convention
//!
//! ```text
//! an-tensor-compiler.interaction.{org}.{domain}
//! ├── safety.chat              # Chat safety predicates
//! ├── safety.pipeline          # Pipeline safety predicates
//! ├── context.chat             # Context management
//! ├── intent.chat              # Intent recognition
//! └── quality.shared           # Cross-org quality metrics
//! ```

mod context;
mod outcomes;
mod predicates;
mod rules;

pub use context::*;
pub use outcomes::*;
pub use predicates::*;
pub use rules::*;

/// Prelude for convenient imports
pub mod prelude {
    pub use super::{
        ContextSpec, IntentSpec, InteractionContext, InteractionOutcome, InteractionPredicate,
        InteractionResult, QualitySpec, SafetySpec,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[test]
    fn test_interaction_context_creation() {
        let device = Device::Cpu;
        let ctx = InteractionContext::new(device);
        assert!(ctx.is_ok());
    }

    #[test]
    fn test_standard_predicates_registration() {
        let device = Device::Cpu;
        let mut ctx = InteractionContext::new(device).unwrap();
        assert!(ctx.register_standard_predicates().is_ok());
        assert!(ctx.predicate_count() > 0);
    }
}

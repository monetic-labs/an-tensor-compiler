//! # CounterOps Primitives
//!
//! Security, defense, and offense capabilities as tensor operations.
//!
//! This module provides tensor-native building blocks for:
//! - **Threat Detection**: Anomaly scoring, attack pattern recognition
//! - **Defense**: Rate limiting, injection detection, defensive mode triggers
//! - **Offense**: Pattern analysis, proactive detection (for authorized use)
//!
//! ## Design Philosophy
//!
//! Security operations are expressed as differentiable predicates that can
//! learn from attack patterns and false positives over time. This enables
//! adaptive security that improves with experience.
//!
//! ## Example
//!
//! ```ignore
//! use an_tensor_compiler::counterops::prelude::*;
//!
//! // Create security context
//! let ctx = SecurityContext::new(&device)?;
//!
//! // Register threat detection predicates
//! ctx.register_standard_predicates()?;
//!
//! // Compile defense rules
//! let rules = ctx.compile_defense_rules()?;
//!
//! // Evaluate request
//! let mut inputs = HashMap::new();
//! inputs.insert("security_features".into(), request_features);
//! let output = rules.forward(&inputs)?;
//!
//! if output.threat_level() > 0.8 {
//!     enter_defense_mode();
//! }
//! ```
//!
//! ## Namespace Convention
//!
//! ```text
//! an-tensor-compiler.counterops.{org}.{domain}
//! ├── threat.shared            # Threat detection (all orgs)
//! ├── defense.pipeline         # Defense predicates
//! ├── offense.trading          # Market attack/defense
//! └── anomaly.shared           # Anomaly detection
//! ```

mod context;
mod detection;
mod outcomes;
mod predicates;
mod rules;

pub use context::*;
pub use detection::*;
pub use outcomes::*;
pub use predicates::*;
pub use rules::*;

/// Prelude for convenient imports
pub mod prelude {
    pub use super::{
        AnomalyScore, AnomalySpec, DefenseSpec, InjectionResult, SecurityContext, SecurityOutcome,
        ThreatAssessment, ThreatPredicate, ThreatSpec,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[test]
    fn test_security_context_creation() {
        let device = Device::Cpu;
        let ctx = SecurityContext::new(device);
        assert!(ctx.is_ok());
    }

    #[test]
    fn test_standard_predicates_registration() {
        let device = Device::Cpu;
        let mut ctx = SecurityContext::new(device).unwrap();
        assert!(ctx.register_standard_predicates().is_ok());
        assert!(ctx.predicate_count() > 0);
    }
}

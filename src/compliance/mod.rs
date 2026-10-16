//! # Compliance Primitives
//!
//! Standardized reporting, privacy, and regulatory compliance.
//!
//! This module provides tensor-native building blocks for:
//! - **Privacy**: PII detection, anonymization verification, consent tracking
//! - **Retention**: Data lifecycle management, deletion policies
//! - **Audit**: Compliance reporting, trail generation
//! - **Regulation**: GDPR, SOX, and other regulatory frameworks
//!
//! ## Design Philosophy
//!
//! Compliance operations are expressed as differentiable predicates that can
//! learn from labeled examples. This enables:
//! - Improved PII detection accuracy over time
//! - Reduced false positives in content filtering
//! - Adaptive consent tracking
//!
//! ## Example
//!
//! ```ignore
//! use an_tensor_compiler::compliance::prelude::*;
//!
//! // Create compliance context for GDPR
//! let ctx = ComplianceContext::new(&device, Regulation::GDPR)?;
//!
//! // Register privacy predicates
//! ctx.register_standard_predicates()?;
//!
//! // Compile privacy rules
//! let rules = ctx.compile_privacy_rules()?;
//!
//! // Check document for PII
//! let mut inputs = HashMap::new();
//! inputs.insert("content_embedding".into(), document_embedding);
//! let output = rules.forward(&inputs)?;
//!
//! if output.pii_detected() {
//!     redact_document(&document);
//! }
//! ```
//!
//! ## Namespace Convention
//!
//! ```text
//! an-tensor-compiler.compliance.{org}.{regulation}
//! ├── privacy.shared           # PII detection, consent
//! ├── gdpr.eu-ops              # GDPR-specific rules
//! ├── sox.us-finance           # SOX compliance
//! ├── audit.shared             # Audit trail generation
//! └── retention.shared         # Data retention policies
//! ```

mod context;
mod outcomes;
mod predicates;
mod regulations;
mod reporting;
mod rules;

pub use context::*;
pub use outcomes::*;
pub use predicates::*;
pub use regulations::*;
pub use reporting::*;
pub use rules::*;

/// Prelude for convenient imports
pub mod prelude {
    pub use super::{
        AuditSpec, ComplianceContext, ComplianceOutcome, ComplianceReport, ComplianceResult,
        PiiType, PrivacyPredicate, PrivacySpec, Regulation, RetentionSpec,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[test]
    fn test_compliance_context_creation() {
        let device = Device::Cpu;
        let ctx = ComplianceContext::new(device, Regulation::GDPR);
        assert!(ctx.is_ok());
    }

    #[test]
    fn test_standard_predicates_registration() {
        let device = Device::Cpu;
        let mut ctx = ComplianceContext::new(device, Regulation::GDPR).unwrap();
        assert!(ctx.register_standard_predicates().is_ok());
        assert!(ctx.predicate_count() > 0);
    }
}

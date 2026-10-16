//! Compliance Predicate Specifications
//!
//! Defines the standard predicates for regulatory compliance.

use crate::compiler::{Activation, PredicateSpec, SimilarityMethod};

/// PII (Personally Identifiable Information) types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PiiType {
    /// Names (first, last, full)
    Name,
    /// Email addresses
    Email,
    /// Phone numbers
    Phone,
    /// Physical addresses
    Address,
    /// Social security numbers
    SSN,
    /// Credit card numbers
    CreditCard,
    /// Dates of birth
    DateOfBirth,
    /// IP addresses
    IPAddress,
    /// Biometric data
    Biometric,
    /// Health information (HIPAA)
    Health,
    /// Financial information
    Financial,
    /// Government IDs (passport, driver's license)
    GovernmentID,
}

impl PiiType {
    /// All PII types
    pub fn all() -> &'static [PiiType] {
        &[
            Self::Name,
            Self::Email,
            Self::Phone,
            Self::Address,
            Self::SSN,
            Self::CreditCard,
            Self::DateOfBirth,
            Self::IPAddress,
            Self::Biometric,
            Self::Health,
            Self::Financial,
            Self::GovernmentID,
        ]
    }

    /// GDPR-relevant PII types
    pub fn gdpr_relevant() -> &'static [PiiType] {
        &[
            Self::Name,
            Self::Email,
            Self::Phone,
            Self::Address,
            Self::DateOfBirth,
            Self::IPAddress,
            Self::Biometric,
            Self::Health,
            Self::Financial,
            Self::GovernmentID,
        ]
    }

    /// High-sensitivity PII types
    pub fn high_sensitivity() -> &'static [PiiType] {
        &[
            Self::SSN,
            Self::CreditCard,
            Self::Biometric,
            Self::Health,
            Self::GovernmentID,
        ]
    }

    /// Name as string
    pub fn name(&self) -> &'static str {
        match self {
            Self::Name => "name",
            Self::Email => "email",
            Self::Phone => "phone",
            Self::Address => "address",
            Self::SSN => "ssn",
            Self::CreditCard => "credit_card",
            Self::DateOfBirth => "date_of_birth",
            Self::IPAddress => "ip_address",
            Self::Biometric => "biometric",
            Self::Health => "health",
            Self::Financial => "financial",
            Self::GovernmentID => "government_id",
        }
    }
}

/// Standard privacy/compliance predicates
#[derive(Debug, Clone)]
pub enum PrivacyPredicate {
    /// PII detection in content
    PiiDetected,
    /// User consent is valid
    ConsentValid,
    /// Data past retention period
    RetentionExceeded,
    /// Audit trail is complete
    AuditComplete,
    /// Data is properly anonymized
    AnonymizationSufficient,
    /// Data is in correct jurisdiction
    JurisdictionMatch,
    /// User authorized for access
    AccessAuthorized,
    /// Sensitive data detected
    SensitiveData,
    /// Custom predicate with spec
    Custom(String, PredicateSpec),
}

impl PrivacyPredicate {
    /// Get the predicate name
    pub fn name(&self) -> &str {
        match self {
            Self::PiiDetected => "pii_detected",
            Self::ConsentValid => "consent_valid",
            Self::RetentionExceeded => "retention_exceeded",
            Self::AuditComplete => "audit_complete",
            Self::AnonymizationSufficient => "anonymization_sufficient",
            Self::JurisdictionMatch => "jurisdiction_match",
            Self::AccessAuthorized => "access_authorized",
            Self::SensitiveData => "sensitive_data",
            Self::Custom(name, _) => name,
        }
    }

    /// Get the default PredicateSpec for this predicate
    pub fn default_spec(&self) -> PredicateSpec {
        match self {
            Self::PiiDetected => Self::pii_detected_spec(1536),
            Self::ConsentValid => Self::consent_valid_spec(),
            Self::RetentionExceeded => Self::retention_exceeded_spec(),
            Self::AuditComplete => Self::audit_complete_spec(),
            Self::AnonymizationSufficient => Self::anonymization_sufficient_spec(1536),
            Self::JurisdictionMatch => Self::jurisdiction_match_spec(),
            Self::AccessAuthorized => Self::access_authorized_spec(64),
            Self::SensitiveData => Self::sensitive_data_spec(1536),
            Self::Custom(_, spec) => spec.clone(),
        }
    }

    /// PII detection: Does content contain personally identifiable information?
    ///
    /// Uses LearnedProjection with attention to detect various PII types
    /// in content embeddings.
    pub fn pii_detected_spec(embedding_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["content_embedding".into()],
            input_dim: embedding_dim,
            hidden_dim: 512,
            activation: Activation::GELU,
            attention_heads: Some(8), // Multi-aspect PII detection
            attention_dropout: Some(0.1),
            layer_norm: Some(true),
            dropout: Some(0.15),
            residual: Some(true),
            conditioning_dim: None,
            conditioning_type: None,
            film_identity_init: None,
        }
    }

    /// Consent valid: Is user consent current and applicable?
    ///
    /// Threshold on consent timestamp age.
    pub fn consent_valid_spec() -> PredicateSpec {
        PredicateSpec::Threshold {
            input: "consent_age_days".into(),
            threshold: 365.0,    // 1 year typical consent validity
            greater_than: false, // Valid if age < threshold
            sharpness: 5.0,
        }
    }

    /// Retention exceeded: Has data exceeded its retention period?
    ///
    /// Threshold on data age.
    pub fn retention_exceeded_spec() -> PredicateSpec {
        PredicateSpec::Threshold {
            input: "data_age_days".into(),
            threshold: 2555.0,  // 7 years (typical legal retention)
            greater_than: true, // Exceeded if age > threshold
            sharpness: 3.0,
        }
    }

    /// Audit complete: Is the audit trail complete?
    ///
    /// Threshold on audit completeness score.
    pub fn audit_complete_spec() -> PredicateSpec {
        PredicateSpec::Threshold {
            input: "audit_score".into(),
            threshold: 0.95, // 95% completeness required
            greater_than: true,
            sharpness: 20.0,
        }
    }

    /// Anonymization sufficient: Is data properly anonymized?
    ///
    /// Uses LearnedSimilarity to compare anonymized content against
    /// original patterns to detect residual PII.
    pub fn anonymization_sufficient_spec(embedding_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedSimilarity {
            left: "anonymized_embedding".into(),
            right: "pii_patterns".into(),
            dim: embedding_dim,
            method: SimilarityMethod::Learned,
        }
    }

    /// Jurisdiction match: Is data in the correct geographic jurisdiction?
    ///
    /// Threshold on jurisdiction code match.
    pub fn jurisdiction_match_spec() -> PredicateSpec {
        PredicateSpec::Threshold {
            input: "jurisdiction_match_score".into(),
            threshold: 0.5,
            greater_than: true,
            sharpness: 10.0,
        }
    }

    /// Access authorized: Is user authorized to access this resource?
    ///
    /// Uses LearnedProjection on combined user/resource features.
    pub fn access_authorized_spec(feature_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["user_role".into(), "resource_type".into()],
            input_dim: feature_dim,
            hidden_dim: 32,
            activation: Activation::ReLU,
            attention_heads: None,
            attention_dropout: None,
            layer_norm: None,
            dropout: Some(0.1),
            residual: None,
            conditioning_dim: None,
            conditioning_type: None,
            film_identity_init: None,
        }
    }

    /// Sensitive data: Does content contain sensitive categories?
    ///
    /// Uses LearnedProjection to detect sensitive data categories
    /// (health, financial, etc.)
    pub fn sensitive_data_spec(embedding_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["content_embedding".into()],
            input_dim: embedding_dim,
            hidden_dim: 256,
            activation: Activation::GELU,
            attention_heads: Some(4),
            attention_dropout: Some(0.1),
            layer_norm: Some(true),
            dropout: Some(0.2),
            residual: Some(true),
            conditioning_dim: None,
            conditioning_type: None,
            film_identity_init: None,
        }
    }
}

/// Privacy specification for customization
#[derive(Debug, Clone)]
pub struct PrivacySpec {
    /// Embedding dimension for content analysis
    pub embedding_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Number of attention heads (None = no attention)
    pub attention_heads: Option<usize>,
    /// Dropout rate
    pub dropout: f32,
    /// PII types to detect
    pub pii_types: Vec<PiiType>,
}

impl Default for PrivacySpec {
    fn default() -> Self {
        Self {
            embedding_dim: 1536,
            hidden_dim: 512,
            attention_heads: Some(8),
            dropout: 0.15,
            pii_types: PiiType::all().to_vec(),
        }
    }
}

impl PrivacySpec {
    /// Convert to PredicateSpec
    pub fn to_predicate_spec(&self) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["content_embedding".into()],
            input_dim: self.embedding_dim,
            hidden_dim: self.hidden_dim,
            activation: Activation::GELU,
            attention_heads: self.attention_heads,
            attention_dropout: Some(0.1),
            layer_norm: Some(true),
            dropout: Some(self.dropout),
            residual: Some(true),
            conditioning_dim: None,
            conditioning_type: None,
            film_identity_init: None,
        }
    }
}

/// Retention specification for customization
#[derive(Debug, Clone)]
pub struct RetentionSpec {
    /// Retention period in days
    pub retention_days: f32,
    /// Sharpness of threshold transition
    pub sharpness: f32,
}

impl Default for RetentionSpec {
    fn default() -> Self {
        Self {
            retention_days: 2555.0, // 7 years
            sharpness: 3.0,
        }
    }
}

impl RetentionSpec {
    /// Convert to PredicateSpec
    pub fn to_predicate_spec(&self) -> PredicateSpec {
        PredicateSpec::Threshold {
            input: "data_age_days".into(),
            threshold: self.retention_days,
            greater_than: true,
            sharpness: self.sharpness,
        }
    }
}

/// Audit specification for customization
#[derive(Debug, Clone)]
pub struct AuditSpec {
    /// Minimum completeness threshold (0.0-1.0)
    pub completeness_threshold: f32,
    /// Sharpness of threshold transition
    pub sharpness: f32,
}

impl Default for AuditSpec {
    fn default() -> Self {
        Self {
            completeness_threshold: 0.95,
            sharpness: 20.0,
        }
    }
}

impl AuditSpec {
    /// Convert to PredicateSpec
    pub fn to_predicate_spec(&self) -> PredicateSpec {
        PredicateSpec::Threshold {
            input: "audit_score".into(),
            threshold: self.completeness_threshold,
            greater_than: true,
            sharpness: self.sharpness,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pii_types() {
        assert_eq!(PiiType::all().len(), 12);
        assert!(!PiiType::gdpr_relevant().is_empty());
        assert!(PiiType::high_sensitivity().contains(&PiiType::SSN));
    }

    #[test]
    fn test_pii_type_names() {
        assert_eq!(PiiType::Email.name(), "email");
        assert_eq!(PiiType::CreditCard.name(), "credit_card");
    }

    #[test]
    fn test_predicate_names() {
        assert_eq!(PrivacyPredicate::PiiDetected.name(), "pii_detected");
        assert_eq!(PrivacyPredicate::ConsentValid.name(), "consent_valid");
        assert_eq!(
            PrivacyPredicate::RetentionExceeded.name(),
            "retention_exceeded"
        );
    }

    #[test]
    fn test_default_specs() {
        let spec = PrivacyPredicate::PiiDetected.default_spec();
        match spec {
            PredicateSpec::LearnedProjection { hidden_dim, .. } => {
                assert_eq!(hidden_dim, 512);
            }
            _ => panic!("Expected LearnedProjection"),
        }
    }

    #[test]
    fn test_privacy_spec_customization() {
        let spec = PrivacySpec {
            embedding_dim: 768,
            hidden_dim: 256,
            attention_heads: Some(4),
            dropout: 0.2,
            pii_types: PiiType::high_sensitivity().to_vec(),
        };

        let pred_spec = spec.to_predicate_spec();
        match pred_spec {
            PredicateSpec::LearnedProjection {
                input_dim,
                hidden_dim,
                attention_heads,
                ..
            } => {
                assert_eq!(input_dim, 768);
                assert_eq!(hidden_dim, 256);
                assert_eq!(attention_heads, Some(4));
            }
            _ => panic!("Expected LearnedProjection"),
        }
    }

    #[test]
    fn test_retention_spec_defaults() {
        let spec = RetentionSpec::default();
        assert_eq!(spec.retention_days, 2555.0);
    }

    #[test]
    fn test_audit_spec_defaults() {
        let spec = AuditSpec::default();
        assert_eq!(spec.completeness_threshold, 0.95);
    }
}

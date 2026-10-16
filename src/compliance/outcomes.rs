//! Compliance Outcomes
//!
//! Training signals for compliance predicates.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Outcome types for compliance training
///
/// These outcomes provide training signal for the compliance predicates.
/// Organisms record outcomes after processing documents, which feeds back
/// into predicate learning.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "PascalCase")]
pub enum ComplianceOutcome {
    // === Correct Compliance Actions ===
    /// Data was correctly identified as compliant
    Compliant,
    /// Violation was correctly detected and prevented
    ViolationPrevented,
    /// PII was correctly detected and redacted
    PiiRedacted,
    /// Data was correctly deleted per retention policy
    DataDeleted,
    /// Audit was completed successfully
    AuditPassed,

    // === Over-Compliance (False Positives) ===
    /// Non-PII was incorrectly flagged as PII
    OverRedaction,
    /// Compliant data was incorrectly flagged
    FalseViolation,
    /// Data was deleted prematurely
    PrematureDeletion,

    // === Under-Compliance (False Negatives) ===
    /// PII was missed during detection
    PiiMissed,
    /// Violation was not detected
    ViolationMissed,
    /// Retention period was not enforced
    RetentionViolated,
    /// Audit failed due to missing records
    AuditFailed,
}

impl ComplianceOutcome {
    /// Training signal strength
    ///
    /// - Positive values (0.0 to 1.0): Good outcomes, reinforce behavior
    /// - Negative values (-1.0 to 0.0): Bad outcomes, adjust predicates
    pub fn signal(&self) -> f32 {
        match self {
            // Strong positive - correct compliance
            Self::Compliant => 0.8,
            Self::ViolationPrevented => 1.0,
            Self::PiiRedacted => 1.0,
            Self::DataDeleted => 0.9,
            Self::AuditPassed => 0.9,

            // Moderate negative - over-compliance (annoying but safe)
            Self::OverRedaction => -0.4,
            Self::FalseViolation => -0.5,
            Self::PrematureDeletion => -0.6,

            // Strong negative - under-compliance (real risk)
            Self::PiiMissed => -0.9,
            Self::ViolationMissed => -1.0,
            Self::RetentionViolated => -0.8,
            Self::AuditFailed => -0.9,
        }
    }

    /// Whether this outcome is positive
    pub fn is_positive(&self) -> bool {
        self.signal() > 0.0
    }

    /// Whether this outcome is negative
    pub fn is_negative(&self) -> bool {
        self.signal() < 0.0
    }

    /// Whether this outcome represents a compliance failure
    pub fn is_compliance_failure(&self) -> bool {
        matches!(
            self,
            Self::PiiMissed | Self::ViolationMissed | Self::RetentionViolated | Self::AuditFailed
        )
    }

    /// Whether this outcome represents over-compliance
    pub fn is_over_compliance(&self) -> bool {
        matches!(
            self,
            Self::OverRedaction | Self::FalseViolation | Self::PrematureDeletion
        )
    }

    /// Category of this outcome
    pub fn category(&self) -> OutcomeCategory {
        match self {
            Self::Compliant
            | Self::ViolationPrevented
            | Self::PiiRedacted
            | Self::DataDeleted
            | Self::AuditPassed => OutcomeCategory::TruePositive,

            Self::OverRedaction | Self::FalseViolation | Self::PrematureDeletion => {
                OutcomeCategory::FalsePositive
            }

            Self::PiiMissed
            | Self::ViolationMissed
            | Self::RetentionViolated
            | Self::AuditFailed => OutcomeCategory::FalseNegative,
        }
    }

    /// Get all outcomes in a category
    pub fn in_category(category: OutcomeCategory) -> Vec<ComplianceOutcome> {
        Self::all()
            .iter()
            .filter(|o| o.category() == category)
            .copied()
            .collect()
    }

    /// All possible outcomes
    pub fn all() -> &'static [ComplianceOutcome] {
        &[
            Self::Compliant,
            Self::ViolationPrevented,
            Self::PiiRedacted,
            Self::DataDeleted,
            Self::AuditPassed,
            Self::OverRedaction,
            Self::FalseViolation,
            Self::PrematureDeletion,
            Self::PiiMissed,
            Self::ViolationMissed,
            Self::RetentionViolated,
            Self::AuditFailed,
        ]
    }
}

impl std::fmt::Display for ComplianceOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compliant => write!(f, "Compliant"),
            Self::ViolationPrevented => write!(f, "ViolationPrevented"),
            Self::PiiRedacted => write!(f, "PiiRedacted"),
            Self::DataDeleted => write!(f, "DataDeleted"),
            Self::AuditPassed => write!(f, "AuditPassed"),
            Self::OverRedaction => write!(f, "OverRedaction"),
            Self::FalseViolation => write!(f, "FalseViolation"),
            Self::PrematureDeletion => write!(f, "PrematureDeletion"),
            Self::PiiMissed => write!(f, "PiiMissed"),
            Self::ViolationMissed => write!(f, "ViolationMissed"),
            Self::RetentionViolated => write!(f, "RetentionViolated"),
            Self::AuditFailed => write!(f, "AuditFailed"),
        }
    }
}

/// Outcome categories (confusion matrix quadrants)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutcomeCategory {
    /// True Positive: Correctly identified compliance issue
    TruePositive,
    /// False Positive: Incorrectly flagged as issue
    FalsePositive,
    /// False Negative: Missed a real compliance issue
    FalseNegative,
}

/// Result of a compliance evaluation
#[derive(Debug, Clone)]
pub struct ComplianceResult {
    /// Is the data/action compliant?
    pub is_compliant: bool,
    /// PII detection score (0.0 = no PII, 1.0 = definite PII)
    pub pii_score: f32,
    /// Consent validity score
    pub consent_score: f32,
    /// Retention status score
    pub retention_score: f32,
    /// Audit completeness score
    pub audit_score: f32,
    /// List of detected issues
    pub issues: Vec<String>,
    /// Explanation for the decision
    pub explanation: String,
    /// Individual predicate activations
    pub predicate_activations: HashMap<String, f32>,
}

impl ComplianceResult {
    /// Check if PII was detected
    pub fn pii_detected(&self) -> bool {
        self.pii_score > 0.5
    }

    /// Check if consent is valid
    pub fn consent_valid(&self) -> bool {
        self.consent_score > 0.5
    }

    /// Check if retention period exceeded
    pub fn retention_exceeded(&self) -> bool {
        self.retention_score > 0.5
    }

    /// Check if audit is complete
    pub fn audit_complete(&self) -> bool {
        self.audit_score > 0.5
    }

    /// Check if any issues were found
    pub fn has_issues(&self) -> bool {
        !self.issues.is_empty()
    }
}

impl Default for ComplianceResult {
    fn default() -> Self {
        Self {
            is_compliant: true,
            pii_score: 0.0,
            consent_score: 1.0,
            retention_score: 0.0,
            audit_score: 1.0,
            issues: Vec::new(),
            explanation: String::new(),
            predicate_activations: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outcome_signals() {
        // Positive outcomes
        assert!(ComplianceOutcome::ViolationPrevented.signal() > 0.0);
        assert!(ComplianceOutcome::PiiRedacted.signal() > 0.0);
        assert!(ComplianceOutcome::AuditPassed.signal() > 0.0);

        // Negative outcomes
        assert!(ComplianceOutcome::PiiMissed.signal() < 0.0);
        assert!(ComplianceOutcome::ViolationMissed.signal() < 0.0);
        assert!(ComplianceOutcome::OverRedaction.signal() < 0.0);
    }

    #[test]
    fn test_compliance_failures() {
        assert!(ComplianceOutcome::PiiMissed.is_compliance_failure());
        assert!(ComplianceOutcome::ViolationMissed.is_compliance_failure());
        assert!(!ComplianceOutcome::OverRedaction.is_compliance_failure());
        assert!(!ComplianceOutcome::ViolationPrevented.is_compliance_failure());
    }

    #[test]
    fn test_over_compliance() {
        assert!(ComplianceOutcome::OverRedaction.is_over_compliance());
        assert!(ComplianceOutcome::FalseViolation.is_over_compliance());
        assert!(!ComplianceOutcome::PiiMissed.is_over_compliance());
    }

    #[test]
    fn test_outcome_categories() {
        assert_eq!(
            ComplianceOutcome::ViolationPrevented.category(),
            OutcomeCategory::TruePositive
        );
        assert_eq!(
            ComplianceOutcome::OverRedaction.category(),
            OutcomeCategory::FalsePositive
        );
        assert_eq!(
            ComplianceOutcome::PiiMissed.category(),
            OutcomeCategory::FalseNegative
        );
    }

    #[test]
    fn test_in_category() {
        let true_positives = ComplianceOutcome::in_category(OutcomeCategory::TruePositive);
        assert_eq!(true_positives.len(), 5);
        assert!(true_positives.contains(&ComplianceOutcome::ViolationPrevented));
    }

    #[test]
    fn test_all_outcomes() {
        let all = ComplianceOutcome::all();
        assert_eq!(all.len(), 12);
    }

    #[test]
    fn test_outcome_serde() {
        let outcome = ComplianceOutcome::PiiRedacted;
        let json = serde_json::to_string(&outcome).unwrap();
        assert_eq!(json, "\"PiiRedacted\"");

        let parsed: ComplianceOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, outcome);
    }

    #[test]
    fn test_compliance_result_default() {
        let result = ComplianceResult::default();
        assert!(result.is_compliant);
        assert!(!result.pii_detected());
        assert!(result.consent_valid());
        assert!(!result.has_issues());
    }

    #[test]
    fn test_compliance_result_with_issues() {
        let result = ComplianceResult {
            is_compliant: false,
            pii_score: 0.9,
            consent_score: 0.3,
            retention_score: 0.1,
            audit_score: 0.95,
            issues: vec!["PII detected".to_string(), "Consent expired".to_string()],
            explanation: "Data contains PII without valid consent".to_string(),
            predicate_activations: HashMap::new(),
        };

        assert!(!result.is_compliant);
        assert!(result.pii_detected());
        assert!(!result.consent_valid());
        assert!(result.has_issues());
        assert_eq!(result.issues.len(), 2);
    }
}

//! Security Outcomes
//!
//! Training signals for security predicates.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Outcome types for security training
///
/// These outcomes provide training signal for the security predicates.
/// Organisms record outcomes after processing requests, which feeds back
/// into predicate learning.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "PascalCase")]
pub enum SecurityOutcome {
    // === True Positives (Correct Detections) ===
    /// Threat was detected and mitigated
    ThreatMitigated,
    /// Attack was blocked
    AttackBlocked,
    /// Anomaly was correctly flagged
    AnomalyConfirmed,
    /// Data exfiltration was prevented
    ExfilPrevented,

    // === False Positives (Unnecessary Actions) ===
    /// False alarm - no actual threat
    FalseAlarm,
    /// Legitimate request was blocked
    LegitimateBlocked,

    // === False Negatives (Missed Threats) ===
    /// Attack succeeded (detection failure)
    AttackSucceeded,
    /// Data breach occurred
    DataBreached,
    /// Anomaly was missed
    AnomalyMissed,

    // === True Negatives (Correct Passes) ===
    /// Request was correctly allowed
    RequestAllowed,
}

impl SecurityOutcome {
    /// Training signal strength
    ///
    /// - Positive values (0.0 to 1.0): Good outcomes, reinforce behavior
    /// - Negative values (-1.0 to 0.0): Bad outcomes, adjust predicates
    pub fn signal(&self) -> f32 {
        match self {
            // Strong positive - correct security actions
            Self::ThreatMitigated => 1.0,
            Self::AttackBlocked => 1.0,
            Self::ExfilPrevented => 1.0,
            Self::AnomalyConfirmed => 0.9,
            Self::RequestAllowed => 0.7,

            // Moderate negative - false positives
            Self::FalseAlarm => -0.5,
            Self::LegitimateBlocked => -0.7,

            // Strong negative - security failures
            Self::AnomalyMissed => -0.8,
            Self::AttackSucceeded => -1.0,
            Self::DataBreached => -1.0,
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

    /// Whether this outcome represents a security failure
    pub fn is_security_failure(&self) -> bool {
        matches!(
            self,
            Self::AttackSucceeded | Self::DataBreached | Self::AnomalyMissed
        )
    }

    /// Category of this outcome
    pub fn category(&self) -> OutcomeCategory {
        match self {
            Self::ThreatMitigated
            | Self::AttackBlocked
            | Self::AnomalyConfirmed
            | Self::ExfilPrevented => OutcomeCategory::TruePositive,

            Self::FalseAlarm | Self::LegitimateBlocked => OutcomeCategory::FalsePositive,

            Self::AttackSucceeded | Self::DataBreached | Self::AnomalyMissed => {
                OutcomeCategory::FalseNegative
            }

            Self::RequestAllowed => OutcomeCategory::TrueNegative,
        }
    }

    /// Get all outcomes in a category
    pub fn in_category(category: OutcomeCategory) -> Vec<SecurityOutcome> {
        Self::all()
            .iter()
            .filter(|o| o.category() == category)
            .copied()
            .collect()
    }

    /// All possible outcomes
    pub fn all() -> &'static [SecurityOutcome] {
        &[
            Self::ThreatMitigated,
            Self::AttackBlocked,
            Self::AnomalyConfirmed,
            Self::ExfilPrevented,
            Self::FalseAlarm,
            Self::LegitimateBlocked,
            Self::AttackSucceeded,
            Self::DataBreached,
            Self::AnomalyMissed,
            Self::RequestAllowed,
        ]
    }
}

impl std::fmt::Display for SecurityOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ThreatMitigated => write!(f, "ThreatMitigated"),
            Self::AttackBlocked => write!(f, "AttackBlocked"),
            Self::AnomalyConfirmed => write!(f, "AnomalyConfirmed"),
            Self::ExfilPrevented => write!(f, "ExfilPrevented"),
            Self::FalseAlarm => write!(f, "FalseAlarm"),
            Self::LegitimateBlocked => write!(f, "LegitimateBlocked"),
            Self::AttackSucceeded => write!(f, "AttackSucceeded"),
            Self::DataBreached => write!(f, "DataBreached"),
            Self::AnomalyMissed => write!(f, "AnomalyMissed"),
            Self::RequestAllowed => write!(f, "RequestAllowed"),
        }
    }
}

/// Outcome categories (confusion matrix quadrants)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutcomeCategory {
    /// True Positive: Correctly detected a threat
    TruePositive,
    /// False Positive: Flagged when there was no threat
    FalsePositive,
    /// False Negative: Missed an actual threat
    FalseNegative,
    /// True Negative: Correctly allowed a safe request
    TrueNegative,
}

/// Result of a threat assessment
#[derive(Debug, Clone)]
pub struct ThreatAssessment {
    /// Overall threat level (0.0 = safe, 1.0 = critical)
    pub threat_level: f32,
    /// Anomaly score (0.0 = normal, 1.0 = highly anomalous)
    pub anomaly_score: f32,
    /// Whether to block the request
    pub should_block: bool,
    /// Whether to enter defense mode
    pub enter_defense_mode: bool,
    /// Specific threat types detected
    pub threats_detected: Vec<String>,
    /// Explanation for the assessment
    pub explanation: String,
    /// Individual predicate activations
    pub predicate_activations: HashMap<String, f32>,
}

impl ThreatAssessment {
    /// Check if the request should be allowed
    pub fn is_allowed(&self) -> bool {
        !self.should_block
    }

    /// Check if immediate action is required
    pub fn requires_action(&self) -> bool {
        self.should_block || self.enter_defense_mode || !self.threats_detected.is_empty()
    }

    /// Severity level as a string
    pub fn severity(&self) -> &'static str {
        if self.threat_level > 0.9 {
            "CRITICAL"
        } else if self.threat_level > 0.7 {
            "HIGH"
        } else if self.threat_level > 0.5 {
            "MEDIUM"
        } else if self.threat_level > 0.3 {
            "LOW"
        } else {
            "NONE"
        }
    }
}

impl Default for ThreatAssessment {
    fn default() -> Self {
        Self {
            threat_level: 0.0,
            anomaly_score: 0.0,
            should_block: false,
            enter_defense_mode: false,
            threats_detected: Vec::new(),
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
        assert!(SecurityOutcome::ThreatMitigated.signal() > 0.0);
        assert!(SecurityOutcome::AttackBlocked.signal() > 0.0);
        assert!(SecurityOutcome::RequestAllowed.signal() > 0.0);

        // Negative outcomes
        assert!(SecurityOutcome::AttackSucceeded.signal() < 0.0);
        assert!(SecurityOutcome::FalseAlarm.signal() < 0.0);
        assert!(SecurityOutcome::DataBreached.signal() < 0.0);
    }

    #[test]
    fn test_security_failures() {
        assert!(SecurityOutcome::AttackSucceeded.is_security_failure());
        assert!(SecurityOutcome::DataBreached.is_security_failure());
        assert!(!SecurityOutcome::FalseAlarm.is_security_failure());
        assert!(!SecurityOutcome::ThreatMitigated.is_security_failure());
    }

    #[test]
    fn test_outcome_categories() {
        assert_eq!(
            SecurityOutcome::ThreatMitigated.category(),
            OutcomeCategory::TruePositive
        );
        assert_eq!(
            SecurityOutcome::FalseAlarm.category(),
            OutcomeCategory::FalsePositive
        );
        assert_eq!(
            SecurityOutcome::AttackSucceeded.category(),
            OutcomeCategory::FalseNegative
        );
        assert_eq!(
            SecurityOutcome::RequestAllowed.category(),
            OutcomeCategory::TrueNegative
        );
    }

    #[test]
    fn test_in_category() {
        let true_positives = SecurityOutcome::in_category(OutcomeCategory::TruePositive);
        assert_eq!(true_positives.len(), 4);
        assert!(true_positives.contains(&SecurityOutcome::ThreatMitigated));
    }

    #[test]
    fn test_all_outcomes() {
        let all = SecurityOutcome::all();
        assert_eq!(all.len(), 10);
    }

    #[test]
    fn test_outcome_serde() {
        let outcome = SecurityOutcome::AttackBlocked;
        let json = serde_json::to_string(&outcome).unwrap();
        assert_eq!(json, "\"AttackBlocked\"");

        let parsed: SecurityOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, outcome);
    }

    #[test]
    fn test_threat_assessment_severity() {
        let mut assessment = ThreatAssessment::default();
        assert_eq!(assessment.severity(), "NONE");

        assessment.threat_level = 0.4;
        assert_eq!(assessment.severity(), "LOW");

        assessment.threat_level = 0.6;
        assert_eq!(assessment.severity(), "MEDIUM");

        assessment.threat_level = 0.8;
        assert_eq!(assessment.severity(), "HIGH");

        assessment.threat_level = 0.95;
        assert_eq!(assessment.severity(), "CRITICAL");
    }

    #[test]
    fn test_threat_assessment_allowed() {
        let mut assessment = ThreatAssessment::default();
        assert!(assessment.is_allowed());

        assessment.should_block = true;
        assert!(!assessment.is_allowed());
    }
}

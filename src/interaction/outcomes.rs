//! Interaction Outcomes
//!
//! Training signals for interaction predicates.

use serde::{Deserialize, Serialize};

/// Outcome types for interaction training
///
/// These outcomes provide training signal for the interaction predicates.
/// Organisms record outcomes after processing messages, which feeds back
/// into predicate learning.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "PascalCase")]
pub enum InteractionOutcome {
    // === Safety Outcomes ===
    /// Message was safe and processed successfully
    SafeResponse,
    /// Unsafe message was correctly blocked
    UnsafeBlocked,
    /// Safe message was incorrectly blocked (false positive)
    SafeMessageBlocked,
    /// Unsafe message was incorrectly passed (false negative)
    UnsafeMessagePassed,

    // === Escalation Outcomes ===
    /// Escalation was needed and correctly triggered
    EscalationCorrect,
    /// Escalation was not needed but was triggered
    EscalationUnnecessary,
    /// Escalation was needed but not triggered
    EscalationMissed,

    // === Quality Outcomes ===
    /// User satisfied with response
    UserSatisfied,
    /// User dissatisfied with response
    UserDissatisfied,
    /// User requested clarification
    ClarificationNeeded,

    // === Context Outcomes ===
    /// Context was fresh and relevant
    ContextRelevant,
    /// Context was stale and refresh was needed
    ContextStale,
}

impl InteractionOutcome {
    /// Training signal strength
    ///
    /// - Positive values (0.0 to 1.0): Good outcomes, reinforce behavior
    /// - Negative values (-1.0 to 0.0): Bad outcomes, adjust predicates
    pub fn signal(&self) -> f32 {
        match self {
            // Strong positive
            Self::SafeResponse => 1.0,
            Self::UnsafeBlocked => 1.0,
            Self::EscalationCorrect => 0.9,
            Self::UserSatisfied => 0.8,
            Self::ContextRelevant => 0.7,

            // Moderate negative
            Self::EscalationUnnecessary => -0.5,
            Self::ClarificationNeeded => -0.3,
            Self::ContextStale => -0.4,
            Self::UserDissatisfied => -0.6,

            // Strong negative
            Self::SafeMessageBlocked => -0.8,
            Self::EscalationMissed => -0.9,
            Self::UnsafeMessagePassed => -1.0,
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

    /// Category of this outcome
    pub fn category(&self) -> OutcomeCategory {
        match self {
            Self::SafeResponse
            | Self::UnsafeBlocked
            | Self::SafeMessageBlocked
            | Self::UnsafeMessagePassed => OutcomeCategory::Safety,

            Self::EscalationCorrect | Self::EscalationUnnecessary | Self::EscalationMissed => {
                OutcomeCategory::Escalation
            }

            Self::UserSatisfied | Self::UserDissatisfied | Self::ClarificationNeeded => {
                OutcomeCategory::Quality
            }

            Self::ContextRelevant | Self::ContextStale => OutcomeCategory::Context,
        }
    }

    /// Get all outcomes in a category
    pub fn in_category(category: OutcomeCategory) -> Vec<InteractionOutcome> {
        Self::all()
            .iter()
            .filter(|o| o.category() == category)
            .copied()
            .collect()
    }

    /// All possible outcomes
    pub fn all() -> &'static [InteractionOutcome] {
        &[
            Self::SafeResponse,
            Self::UnsafeBlocked,
            Self::SafeMessageBlocked,
            Self::UnsafeMessagePassed,
            Self::EscalationCorrect,
            Self::EscalationUnnecessary,
            Self::EscalationMissed,
            Self::UserSatisfied,
            Self::UserDissatisfied,
            Self::ClarificationNeeded,
            Self::ContextRelevant,
            Self::ContextStale,
        ]
    }
}

impl std::fmt::Display for InteractionOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SafeResponse => write!(f, "SafeResponse"),
            Self::UnsafeBlocked => write!(f, "UnsafeBlocked"),
            Self::SafeMessageBlocked => write!(f, "SafeMessageBlocked"),
            Self::UnsafeMessagePassed => write!(f, "UnsafeMessagePassed"),
            Self::EscalationCorrect => write!(f, "EscalationCorrect"),
            Self::EscalationUnnecessary => write!(f, "EscalationUnnecessary"),
            Self::EscalationMissed => write!(f, "EscalationMissed"),
            Self::UserSatisfied => write!(f, "UserSatisfied"),
            Self::UserDissatisfied => write!(f, "UserDissatisfied"),
            Self::ClarificationNeeded => write!(f, "ClarificationNeeded"),
            Self::ContextRelevant => write!(f, "ContextRelevant"),
            Self::ContextStale => write!(f, "ContextStale"),
        }
    }
}

/// Outcome categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutcomeCategory {
    /// Safety-related outcomes
    Safety,
    /// Escalation-related outcomes
    Escalation,
    /// Quality-related outcomes
    Quality,
    /// Context-related outcomes
    Context,
}

/// Result of an interaction evaluation
#[derive(Debug, Clone)]
pub struct InteractionResult {
    /// Overall safety score (0.0 = unsafe, 1.0 = safe)
    pub safety_score: f32,
    /// Whether the message should be processed
    pub should_process: bool,
    /// Whether escalation is recommended
    pub should_escalate: bool,
    /// Whether context refresh is needed
    pub needs_refresh: bool,
    /// Explanation for the decision
    pub explanation: String,
    /// Individual predicate activations
    pub predicate_activations: std::collections::HashMap<String, f32>,
}

impl InteractionResult {
    /// Check if the message is safe to process
    pub fn is_safe(&self) -> bool {
        self.safety_score > 0.5 && self.should_process
    }

    /// Check if action is needed (escalation or refresh)
    pub fn needs_action(&self) -> bool {
        self.should_escalate || self.needs_refresh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outcome_signals() {
        // Positive outcomes
        assert!(InteractionOutcome::SafeResponse.signal() > 0.0);
        assert!(InteractionOutcome::UnsafeBlocked.signal() > 0.0);
        assert!(InteractionOutcome::EscalationCorrect.signal() > 0.0);

        // Negative outcomes
        assert!(InteractionOutcome::UnsafeMessagePassed.signal() < 0.0);
        assert!(InteractionOutcome::SafeMessageBlocked.signal() < 0.0);
        assert!(InteractionOutcome::EscalationMissed.signal() < 0.0);
    }

    #[test]
    fn test_outcome_categories() {
        assert_eq!(
            InteractionOutcome::SafeResponse.category(),
            OutcomeCategory::Safety
        );
        assert_eq!(
            InteractionOutcome::EscalationCorrect.category(),
            OutcomeCategory::Escalation
        );
        assert_eq!(
            InteractionOutcome::UserSatisfied.category(),
            OutcomeCategory::Quality
        );
        assert_eq!(
            InteractionOutcome::ContextStale.category(),
            OutcomeCategory::Context
        );
    }

    #[test]
    fn test_in_category() {
        let safety_outcomes = InteractionOutcome::in_category(OutcomeCategory::Safety);
        assert_eq!(safety_outcomes.len(), 4);
        assert!(safety_outcomes.contains(&InteractionOutcome::SafeResponse));
    }

    #[test]
    fn test_all_outcomes() {
        let all = InteractionOutcome::all();
        assert_eq!(all.len(), 12);
    }

    #[test]
    fn test_outcome_display() {
        assert_eq!(InteractionOutcome::SafeResponse.to_string(), "SafeResponse");
    }

    #[test]
    fn test_outcome_serde() {
        let outcome = InteractionOutcome::EscalationCorrect;
        let json = serde_json::to_string(&outcome).unwrap();
        assert_eq!(json, "\"EscalationCorrect\"");

        let parsed: InteractionOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, outcome);
    }

    #[test]
    fn test_interaction_result() {
        let result = InteractionResult {
            safety_score: 0.9,
            should_process: true,
            should_escalate: false,
            needs_refresh: false,
            explanation: "All checks passed".to_string(),
            predicate_activations: std::collections::HashMap::new(),
        };

        assert!(result.is_safe());
        assert!(!result.needs_action());
    }
}

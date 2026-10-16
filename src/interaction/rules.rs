//! Interaction Rule Templates
//!
//! Pre-defined rule templates for common interaction patterns.
//!
//! ## Single-Head vs Multi-Head Rules
//!
//! - **Single-head rules**: Return one decision (e.g., `safe_to_respond`). Use with `CompiledRule`.
//! - **Multi-head rules**: Return multiple independent decisions. Use with `MultiHeadCompiledRule`.
//!
//! Most templates provide both options:
//! - `SafetyRules::BASIC_GATE` → single head
//! - `SafetyRules::MULTI_HEAD` → multiple heads

use crate::compiler::RuleSpec;
use crate::Result;

/// Safety rule templates
pub struct SafetyRules;

impl SafetyRules {
    /// Basic safety gate: block unsafe messages
    ///
    /// ```prolog
    /// safe_to_respond(X) :- message_safe(X), not blocked_user(X).
    /// ```
    pub const BASIC_GATE: &'static str = r#"
        safe_to_respond(X) :- message_safe(X), not blocked_user(X).
    "#;

    /// Comprehensive safety with escalation
    ///
    /// ```prolog
    /// safe_to_respond(X) :- message_safe(X), not blocked_user(X), not escalation_needed(X).
    /// needs_review(X) :- not message_safe(X).
    /// needs_review(X) :- escalation_needed(X).
    /// ```
    pub const WITH_ESCALATION: &'static str = r#"
        safe_to_respond(X) :- message_safe(X), not blocked_user(X), not escalation_needed(X).
        needs_review(X) :- not message_safe(X).
        needs_review(X) :- escalation_needed(X).
    "#;

    /// Parse basic safety gate rules
    pub fn parse_basic() -> Result<RuleSpec> {
        RuleSpec::parse("safety_basic", Self::BASIC_GATE)
    }

    /// Parse comprehensive safety rules (single-head version)
    pub fn parse_with_escalation() -> Result<RuleSpec> {
        RuleSpec::parse("safety_escalation", Self::WITH_ESCALATION)
    }

    /// Multi-head safety rules
    ///
    /// Returns independent decisions for each concern:
    /// - `safe_to_respond`: Primary safety gate
    /// - `needs_review`: Requires human review
    /// - `blocked`: Should not respond at all
    ///
    /// ```prolog
    /// safe_to_respond(X) :- message_safe(X), not blocked_user(X), not escalation_needed(X).
    /// needs_review(X) :- not message_safe(X), not blocked_user(X).
    /// needs_review(X) :- escalation_needed(X).
    /// blocked(X) :- blocked_user(X).
    /// blocked(X) :- severe_violation(X).
    /// ```
    pub const MULTI_HEAD: &'static str = r#"
        safe_to_respond(X) :- message_safe(X), not blocked_user(X), not escalation_needed(X).
        needs_review(X) :- not message_safe(X), not blocked_user(X).
        needs_review(X) :- escalation_needed(X).
        blocked(X) :- blocked_user(X).
        blocked(X) :- severe_violation(X).
    "#;

    /// Parse multi-head safety rules (use with `MultiHeadCompiledRule`)
    pub fn parse_multi_head() -> Result<RuleSpec> {
        RuleSpec::parse("safety_multi", Self::MULTI_HEAD)
    }
}

/// Context management rule templates
pub struct ContextRules;

impl ContextRules {
    /// Context refresh logic
    ///
    /// ```prolog
    /// needs_refresh(X) :- not context_fresh(X), not recently_refreshed(X).
    /// use_cached(X) :- context_fresh(X).
    /// ```
    pub const REFRESH_LOGIC: &'static str = r#"
        needs_refresh(X) :- not context_fresh(X), not recently_refreshed(X).
        use_cached(X) :- context_fresh(X).
    "#;

    /// Topic continuity rules
    ///
    /// ```prolog
    /// continue_topic(X) :- not topic_drift(X), context_fresh(X).
    /// start_new_topic(X) :- topic_drift(X).
    /// start_new_topic(X) :- not context_fresh(X).
    /// ```
    pub const TOPIC_CONTINUITY: &'static str = r#"
        continue_topic(X) :- not topic_drift(X), context_fresh(X).
        start_new_topic(X) :- topic_drift(X).
        start_new_topic(X) :- not context_fresh(X).
    "#;

    /// Parse context refresh rules
    pub fn parse_refresh() -> Result<RuleSpec> {
        RuleSpec::parse("context_refresh", Self::REFRESH_LOGIC)
    }

    /// Parse topic continuity rules
    pub fn parse_topic_continuity() -> Result<RuleSpec> {
        RuleSpec::parse("context_topic", Self::TOPIC_CONTINUITY)
    }

    /// Multi-head context rules
    ///
    /// Returns independent decisions:
    /// - `needs_refresh`: Context needs refreshing
    /// - `use_cached`: Safe to use cached context
    /// - `continue_topic`: Stay on current topic
    /// - `start_new_topic`: Begin new topic
    ///
    /// ```prolog
    /// needs_refresh(X) :- not context_fresh(X), not recently_refreshed(X).
    /// use_cached(X) :- context_fresh(X).
    /// continue_topic(X) :- not topic_drift(X), context_fresh(X).
    /// start_new_topic(X) :- topic_drift(X).
    /// start_new_topic(X) :- not context_fresh(X).
    /// ```
    pub const MULTI_HEAD: &'static str = r#"
        needs_refresh(X) :- not context_fresh(X), not recently_refreshed(X).
        use_cached(X) :- context_fresh(X).
        continue_topic(X) :- not topic_drift(X), context_fresh(X).
        start_new_topic(X) :- topic_drift(X).
        start_new_topic(X) :- not context_fresh(X).
    "#;

    /// Parse multi-head context rules (use with `MultiHeadCompiledRule`)
    pub fn parse_multi_head() -> Result<RuleSpec> {
        RuleSpec::parse("context_multi", Self::MULTI_HEAD)
    }
}

/// Escalation rule templates
pub struct EscalationRules;

impl EscalationRules {
    /// Basic escalation logic
    ///
    /// ```prolog
    /// escalate(X) :- escalation_needed(X), human_available(X).
    /// queue_for_review(X) :- escalation_needed(X), not human_available(X).
    /// ```
    pub const BASIC: &'static str = r#"
        escalate(X) :- escalation_needed(X), human_available(X).
        queue_for_review(X) :- escalation_needed(X), not human_available(X).
    "#;

    /// Frustration-aware escalation
    ///
    /// ```prolog
    /// escalate(X) :- escalation_needed(X), human_available(X).
    /// escalate(X) :- user_frustrated(X), repeated_failure(X).
    /// queue_for_review(X) :- escalation_needed(X), not human_available(X).
    /// queue_for_review(X) :- user_frustrated(X), not human_available(X).
    /// ```
    pub const FRUSTRATION_AWARE: &'static str = r#"
        escalate(X) :- escalation_needed(X), human_available(X).
        escalate(X) :- user_frustrated(X), repeated_failure(X).
        queue_for_review(X) :- escalation_needed(X), not human_available(X).
        queue_for_review(X) :- user_frustrated(X), not human_available(X).
    "#;

    /// Parse basic escalation rules
    pub fn parse_basic() -> Result<RuleSpec> {
        RuleSpec::parse("escalation_basic", Self::BASIC)
    }

    /// Parse frustration-aware escalation rules
    pub fn parse_frustration_aware() -> Result<RuleSpec> {
        RuleSpec::parse("escalation_frustration", Self::FRUSTRATION_AWARE)
    }

    /// Multi-head escalation rules
    ///
    /// Returns independent decisions:
    /// - `escalate`: Immediately escalate to human
    /// - `queue_for_review`: Queue for later review
    /// - `auto_resolve`: Can be resolved automatically
    ///
    /// ```prolog
    /// escalate(X) :- escalation_needed(X), human_available(X).
    /// escalate(X) :- user_frustrated(X), repeated_failure(X), human_available(X).
    /// queue_for_review(X) :- escalation_needed(X), not human_available(X).
    /// queue_for_review(X) :- user_frustrated(X), not human_available(X).
    /// auto_resolve(X) :- not escalation_needed(X), not user_frustrated(X).
    /// ```
    pub const MULTI_HEAD: &'static str = r#"
        escalate(X) :- escalation_needed(X), human_available(X).
        escalate(X) :- user_frustrated(X), repeated_failure(X), human_available(X).
        queue_for_review(X) :- escalation_needed(X), not human_available(X).
        queue_for_review(X) :- user_frustrated(X), not human_available(X).
        auto_resolve(X) :- not escalation_needed(X), not user_frustrated(X).
    "#;

    /// Parse multi-head escalation rules (use with `MultiHeadCompiledRule`)
    pub fn parse_multi_head() -> Result<RuleSpec> {
        RuleSpec::parse("escalation_multi", Self::MULTI_HEAD)
    }
}

/// Quality assurance rule templates
pub struct QualityRules;

impl QualityRules {
    /// Response quality gate
    ///
    /// ```prolog
    /// good_response(X) :- response_coherent(X), intent_clear(X).
    /// needs_improvement(X) :- not response_coherent(X).
    /// needs_clarification(X) :- not intent_clear(X).
    /// ```
    pub const RESPONSE_QUALITY: &'static str = r#"
        good_response(X) :- response_coherent(X), intent_clear(X).
        needs_improvement(X) :- not response_coherent(X).
        needs_clarification(X) :- not intent_clear(X).
    "#;

    /// Parse response quality rules
    pub fn parse_response_quality() -> Result<RuleSpec> {
        RuleSpec::parse("quality_response", Self::RESPONSE_QUALITY)
    }

    /// Multi-head quality rules
    ///
    /// Returns independent decisions:
    /// - `good_response`: Response is high quality
    /// - `needs_improvement`: Response needs work
    /// - `needs_clarification`: User intent unclear
    ///
    /// ```prolog
    /// good_response(X) :- response_coherent(X), intent_clear(X).
    /// needs_improvement(X) :- not response_coherent(X).
    /// needs_clarification(X) :- not intent_clear(X).
    /// ```
    pub const MULTI_HEAD: &'static str = r#"
        good_response(X) :- response_coherent(X), intent_clear(X).
        needs_improvement(X) :- not response_coherent(X).
        needs_clarification(X) :- not intent_clear(X).
    "#;

    /// Parse multi-head quality rules (use with `MultiHeadCompiledRule`)
    pub fn parse_multi_head() -> Result<RuleSpec> {
        RuleSpec::parse("quality_multi", Self::MULTI_HEAD)
    }
}

/// Complete interaction rule set combining all templates
pub struct InteractionRuleSet;

impl InteractionRuleSet {
    /// Full interaction rules with single head (proceed/not proceed decision)
    ///
    /// This ruleset combines safety, context, and quality checks into a
    /// unified `proceed` decision. Use the individual rule templates
    /// (SafetyRules, ContextRules, etc.) when you need separate decisions.
    pub const COMPLETE: &'static str = r#"
        proceed(X) :- message_safe(X), not blocked_user(X), not escalation_needed(X), 
                      context_fresh(X), response_coherent(X), intent_clear(X).
        proceed(X) :- message_safe(X), not blocked_user(X), not escalation_needed(X),
                      not context_fresh(X), response_coherent(X), intent_clear(X), 
                      refresh_available(X).
    "#;

    /// Parse complete interaction rules
    pub fn parse_complete() -> Result<RuleSpec> {
        RuleSpec::parse("interaction_complete", Self::COMPLETE)
    }

    /// Full interaction rules with multiple heads
    ///
    /// Returns independent decisions for comprehensive interaction handling:
    /// - `proceed`: Safe to generate response
    /// - `needs_review`: Requires human oversight
    /// - `escalate`: Immediate escalation needed
    /// - `blocked`: Do not respond
    /// - `needs_clarification`: Ask user for clarification
    ///
    /// Use with `MultiHeadCompiledRule` for full multi-output capability.
    pub const MULTI_HEAD: &'static str = r#"
        proceed(X) :- message_safe(X), not blocked_user(X), not escalation_needed(X), 
                      context_fresh(X), response_coherent(X), intent_clear(X).
        proceed(X) :- message_safe(X), not blocked_user(X), not escalation_needed(X),
                      not context_fresh(X), response_coherent(X), intent_clear(X), 
                      refresh_available(X).
        needs_review(X) :- not message_safe(X), not blocked_user(X).
        needs_review(X) :- escalation_needed(X), not blocked_user(X).
        escalate(X) :- escalation_needed(X), human_available(X).
        escalate(X) :- user_frustrated(X), repeated_failure(X).
        blocked(X) :- blocked_user(X).
        blocked(X) :- severe_violation(X).
        needs_clarification(X) :- not intent_clear(X), message_safe(X).
    "#;

    /// Parse multi-head interaction rules (use with `MultiHeadCompiledRule`)
    pub fn parse_multi_head() -> Result<RuleSpec> {
        RuleSpec::parse("interaction_multi", Self::MULTI_HEAD)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_safety_basic() {
        let spec = SafetyRules::parse_basic();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_safety_escalation() {
        let spec = SafetyRules::parse_with_escalation();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_context_refresh() {
        let spec = ContextRules::parse_refresh();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_topic_continuity() {
        let spec = ContextRules::parse_topic_continuity();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_escalation_basic() {
        let spec = EscalationRules::parse_basic();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_escalation_frustration() {
        let spec = EscalationRules::parse_frustration_aware();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_quality_response() {
        let spec = QualityRules::parse_response_quality();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_complete_ruleset() {
        let spec = InteractionRuleSet::parse_complete();
        assert!(spec.is_ok());
    }

    // Multi-head parsing tests
    #[test]
    fn test_parse_safety_multi_head() {
        let spec = SafetyRules::parse_multi_head();
        assert!(spec.is_ok());
        let spec = spec.unwrap();
        // Should have 3 heads: safe_to_respond, needs_review, blocked
        let heads: std::collections::HashSet<_> =
            spec.rules.iter().map(|r| r.head.as_str()).collect();
        assert!(heads.contains("safe_to_respond"));
        assert!(heads.contains("needs_review"));
        assert!(heads.contains("blocked"));
    }

    #[test]
    fn test_parse_context_multi_head() {
        let spec = ContextRules::parse_multi_head();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_escalation_multi_head() {
        let spec = EscalationRules::parse_multi_head();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_quality_multi_head() {
        let spec = QualityRules::parse_multi_head();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_interaction_multi_head() {
        let spec = InteractionRuleSet::parse_multi_head();
        assert!(spec.is_ok());
        let spec = spec.unwrap();
        // Should have 5 heads
        let heads: std::collections::HashSet<_> =
            spec.rules.iter().map(|r| r.head.as_str()).collect();
        assert!(heads.contains("proceed"));
        assert!(heads.contains("needs_review"));
        assert!(heads.contains("escalate"));
        assert!(heads.contains("blocked"));
        assert!(heads.contains("needs_clarification"));
    }
}

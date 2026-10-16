//! F-1 demo fixture (Slice 4): a safety-rules spec exercising every L0
//! outcome, margin collapse, and a sort finding. Shared by the integration
//! test and the runnable example so both prove the same behavior.
//!
//! Local builders duplicate `property::fixtures` deliberately: fixtures are
//! `#[cfg(test)]` while the demo ships in the example binary.

use crate::compiler::{Argument, Literal, PredicateSpec, Rule, RuleSpec};

use super::margin::MarginConfig;
use super::property::{L0Declarations, TriggerDecl};
use super::sort::SortPolicy;

/// Everything the F-1 pipeline needs, preassembled for the demo.
pub struct DemoInputs {
    /// Demo rule spec.
    pub spec: RuleSpec,
    /// L0 declarations (exclusions, triggers).
    pub decls: L0Declarations,
    /// Margin propagation config.
    pub margin_config: MarginConfig,
    /// Sort-check policy.
    pub sort_policy: SortPolicy,
    /// Predicate under demo.
    pub predicate_id: String,
    /// Model version string.
    pub model_version: String,
    /// Corpus reference string.
    pub corpus_ref: String,
}

fn threshold() -> PredicateSpec {
    PredicateSpec::Threshold {
        input: "x".to_string(),
        threshold: 0.5,
        greater_than: true,
        sharpness: 10.0,
    }
}

fn literal(predicate: &str) -> Literal {
    Literal::positive(predicate, vec![Argument::Variable("X".to_string())])
}

fn negated(predicate: &str) -> Literal {
    Literal::negated(predicate, vec![Argument::Variable("X".to_string())])
}

fn rule(head: &str, body: Vec<Literal>) -> Rule {
    Rule::new(head, body)
}

/// The demo safety-rules spec. Intentional findings (asserted by the
/// integration test, not incidental):
///
/// - Rule 3 (`blocked :- flagged, ¬flagged`): contradictory body → BodySat
///   + HeadCoverage disproved.
/// - Rule 5 (`escalate :- vip_override`): unguarded path → TriggerPresence
///   disproved with a countermodel.
/// - The 5-leaf conjunction and 2-rule heads at ε=0.1: margin collapse →
///   `safe_to_respond`, `escalate` unresolvable, `blocked` unresolvable on
///   its true side.
/// - `tier` (TierLookup over `[1.5, 0.8]`) in literal position: one Score
///   finding (warn).
pub fn safety_spec() -> DemoInputs {
    let mut spec = RuleSpec::new("demo_safety_rules");
    for name in [
        "low_toxicity",
        "has_consent",
        "reviewed_clean",
        "uncertain_intent",
        "flagged",
        "high_risk",
        "verified",
        "fresh",
        "calm",
    ] {
        spec.add_predicate(name.to_string(), threshold());
    }
    spec.add_predicate("vip_override", PredicateSpec::External);
    spec.add_predicate(
        "tier",
        PredicateSpec::TierLookup {
            input: "utilization".to_string(),
            thresholds: vec![50.0],
            values: vec![1.5, 0.8],
            sharpness: 5.0,
            interpolate: false,
        },
    );

    spec.add_rule(rule(
        "safe_to_respond",
        vec![literal("low_toxicity"), literal("has_consent")],
    ));
    spec.add_rule(rule("safe_to_respond", vec![literal("reviewed_clean")]));
    spec.add_rule(rule("needs_review", vec![literal("uncertain_intent")]));
    spec.add_rule(rule(
        "blocked",
        vec![literal("flagged"), negated("flagged")],
    ));
    spec.add_rule(rule(
        "escalate",
        vec![
            literal("low_toxicity"),
            literal("high_risk"),
            literal("verified"),
            literal("fresh"),
            literal("calm"),
        ],
    ));
    spec.add_rule(rule("escalate", vec![literal("vip_override")]));
    spec.add_rule(rule("routed", vec![literal("tier")]));

    DemoInputs {
        spec,
        decls: L0Declarations {
            exclusions: vec![vec!["safe_to_respond".to_string(), "blocked".to_string()]],
            triggers: vec![TriggerDecl {
                head: "escalate".to_string(),
                triggers: vec!["high_risk".to_string()],
            }],
        },
        margin_config: MarginConfig::uniform(0.1),
        sort_policy: SortPolicy::default(),
        predicate_id: "demo_safety_rules".to_string(),
        model_version: "tl-0.4.2".to_string(),
        corpus_ref: "corpus_demo_v1".to_string(),
    }
}

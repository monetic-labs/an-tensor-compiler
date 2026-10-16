//! Output-sort checking (Slice 2).
//!
//! Predicate outputs are multi-sorted (PPL finding F-B): fuzzy combinators
//! assume `[0,1]` truth, but several predicate types produce scores, counts,
//! or indices. [`sort_check`] verifies that every literal-position predicate
//! resolves to [`OutputSort::Truth`], so a count of `3.0` can never silently
//! flow into `fuzzy_and`.
//!
//! Sorts resolve recursively over predicates *and* heads: a head's sort is
//! `Truth` iff every literal in its transitive body closure resolves to
//! `Truth` (fuzzy ops preserve `Truth`; anything else poisons it).
//! Reference cycles resolve to [`OutputSort::Unknown`] and are reported.
//! Names unknown to the spec resolve to `Unknown` and are *skipped* here —
//! undefined references belong to L0.6, except inside [`PredicateSpec::Composite`]
//! children, which rule-body validation cannot see (those are reported here).

use std::collections::BTreeSet;

use crate::compiler::{PairwiseOutput, PredicateSpec, RuleSpec};

/// Output sorts. Only [`OutputSort::Truth`] may appear in literal position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputSort {
    /// `[0,1]`-valued truth degree: combinable under t-norms/s-norms.
    Truth,
    /// Real-valued score (preference weight, magnitude, difference).
    Score,
    /// Positional output (argmax-style index).
    Index,
    /// Cardinal output (a count, possibly normalized).
    Count,
    /// Unresolvable: unknown name or reference cycle.
    Unknown,
}

/// Finding severity. Strictness is a policy knob (decision D2b):
/// warn-then-deny, default warn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortSeverity {
    /// Report but allow.
    Warn,
    /// Reject the spec.
    Deny,
}

/// Where a non-`Truth` sort was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SortLocation {
    /// A rule-body literal must be `Truth`-sorted.
    Literal {
        /// Index into the spec's rule list.
        rule_idx: usize,
        /// Offending predicate name.
        predicate: String,
    },
    /// A composite child must resolve (rule-body validation cannot see it).
    CompositeChild {
        /// Composite predicate name.
        composite: String,
        /// Unresolved child name.
        child: String,
    },
}

/// One sort finding: a non-`Truth` sort where `Truth` is required.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortFinding {
    /// Where the bad sort was found.
    pub location: SortLocation,
    /// The offending sort.
    pub sort: OutputSort,
    /// Warn or deny.
    pub severity: SortSeverity,
    /// Human-readable explanation.
    pub message: String,
}

/// Sort policy: when `deny_non_truth_literals` is set, literal-position
/// findings report [`SortSeverity::Deny`]; otherwise [`SortSeverity::Warn`].
#[derive(Debug, Clone, Copy)]
pub struct SortPolicy {
    /// Deny (vs warn) non-`Truth` literal positions.
    pub deny_non_truth_literals: bool,
}

impl Default for SortPolicy {
    /// D2b default: warn, with denial after a migration window.
    fn default() -> Self {
        Self {
            deny_non_truth_literals: false,
        }
    }
}

/// Sort-check `spec` under the default (warn) policy.
pub fn sort_check(spec: &RuleSpec) -> Vec<SortFinding> {
    sort_check_with_policy(spec, &SortPolicy::default())
}

/// Sort-check `spec` under `policy`.
pub fn sort_check_with_policy(spec: &RuleSpec, policy: &SortPolicy) -> Vec<SortFinding> {
    let mut findings = Vec::new();
    let severity = if policy.deny_non_truth_literals {
        SortSeverity::Deny
    } else {
        SortSeverity::Warn
    };

    // Literal positions: every rule body, in order (deterministic).
    for (rule_idx, rule) in spec.rules.iter().enumerate() {
        for lit in &rule.body {
            match sort_of_name(spec, &lit.predicate) {
                OutputSort::Truth | OutputSort::Unknown => {}
                sort => findings.push(SortFinding {
                    location: SortLocation::Literal {
                        rule_idx,
                        predicate: lit.predicate.clone(),
                    },
                    sort,
                    severity,
                    message: format!(
                        "rule {rule_idx}: literal '{}' has sort {sort:?}, expected Truth",
                        lit.predicate
                    ),
                }),
            }
        }
    }

    // Composite children: must resolve (invisible to rule-body validation).
    // Predicate names iterated sorted for determinism.
    let mut names: Vec<&String> = spec.predicates.keys().collect();
    names.sort();
    for name in names {
        if let PredicateSpec::Composite { children, .. } = &spec.predicates[name] {
            for child in children {
                let defined = spec.predicates.contains_key(child);
                let headed = spec.rules.iter().any(|rule| rule.head == *child);
                if !defined && !headed {
                    findings.push(SortFinding {
                        location: SortLocation::CompositeChild {
                            composite: name.clone(),
                            child: child.clone(),
                        },
                        sort: OutputSort::Unknown,
                        severity,
                        message: format!("composite '{name}' references unknown '{child}'"),
                    });
                }
            }
        }
    }

    findings
}

/// Resolve the output sort of any name: defined predicates via the sort
/// table, heads via transitive body closure, anything else `Unknown`.
/// Cycles resolve to `Unknown` (reported by the caller as a finding when
/// they appear where `Truth` is required).
pub fn sort_of_name(spec: &RuleSpec, name: &str) -> OutputSort {
    sort_of_name_guarded(spec, name, &mut BTreeSet::new())
}

fn sort_of_name_guarded(
    spec: &RuleSpec,
    name: &str,
    visiting: &mut BTreeSet<String>,
) -> OutputSort {
    if !visiting.insert(name.to_string()) {
        return OutputSort::Unknown;
    }
    let sort = if let Some(predicate) = spec.predicates.get(name) {
        sort_of_predicate(spec, predicate, visiting)
    } else if spec.rules.iter().any(|rule| rule.head == name) {
        // Head: Truth iff every literal in every body is transitively Truth.
        let mut head_sort = OutputSort::Truth;
        for rule in spec.rules.iter().filter(|rule| rule.head == name) {
            for lit in &rule.body {
                match sort_of_name_guarded(spec, &lit.predicate, visiting) {
                    OutputSort::Truth => {}
                    other => {
                        head_sort = other;
                        break;
                    }
                }
            }
            if head_sort != OutputSort::Truth {
                break;
            }
        }
        head_sort
    } else {
        OutputSort::Unknown
    };
    visiting.remove(name);
    sort
}

/// The per-variant sort table. Operators are sort-transparent: AND/OR/NOT
/// over `Truth` stay `Truth` — only the children's sorts matter.
fn sort_of_predicate(
    spec: &RuleSpec,
    predicate: &PredicateSpec,
    visiting: &mut BTreeSet<String>,
) -> OutputSort {
    // typescript-exhaustive-switch analog: this match must stay exhaustive
    // over PredicateSpec — a new variant with an unconsidered output shape
    // must fail closed here, not default to Truth.
    match predicate {
        PredicateSpec::Threshold { .. }
        | PredicateSpec::Learned { .. }
        | PredicateSpec::LearnedProjection { .. }
        | PredicateSpec::LearnedSimilarity { .. }
        | PredicateSpec::GraphNeural { .. } => OutputSort::Truth,
        // External delivers Truth by caller contract (runtime-checked —
        // PPL L1 External row), so it sorts as Truth statically.
        PredicateSpec::External => OutputSort::Truth,
        PredicateSpec::Composite { children, .. } => {
            let mut composite_sort = OutputSort::Truth;
            for child in children {
                match sort_of_name_guarded(spec, child, visiting) {
                    OutputSort::Truth => {}
                    other => {
                        composite_sort = other;
                        break;
                    }
                }
            }
            composite_sort
        }
        PredicateSpec::TierLookup { values, .. } => {
            if values.iter().all(|value| (0.0..=1.0).contains(value)) {
                OutputSort::Truth
            } else {
                OutputSort::Score
            }
        }
        PredicateSpec::PairwiseDifference { output_mode, .. } => match output_mode {
            PairwiseOutput::AnyAboveThreshold => OutputSort::Truth,
            PairwiseOutput::MaxDifference | PairwiseOutput::AllPairs => OutputSort::Score,
            PairwiseOutput::CountAboveThreshold => OutputSort::Count,
            PairwiseOutput::BestIndex => OutputSort::Index,
        },
        PredicateSpec::CascadingLookup { .. } => OutputSort::Score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{Literal, Rule};
    use crate::governance::property::fixtures::{literal, rule, threshold_predicate};

    fn spec_with_predicates(names: &[&str]) -> RuleSpec {
        let mut spec = RuleSpec::new("sorts");
        for name in names {
            spec.add_predicate(name.to_string(), threshold_predicate());
        }
        spec
    }

    fn tier_lookup(values: Vec<f32>) -> PredicateSpec {
        PredicateSpec::TierLookup {
            input: "x".to_string(),
            thresholds: vec![50.0],
            values,
            sharpness: 5.0,
            interpolate: false,
        }
    }

    fn pairwise(mode: PairwiseOutput) -> PredicateSpec {
        PredicateSpec::PairwiseDifference {
            input: "costs".to_string(),
            threshold_pct: 20.0,
            sharpness: 10.0,
            output_mode: mode,
        }
    }

    #[test]
    fn truth_literals_pass_silently() {
        let mut spec = spec_with_predicates(&["p", "q"]);
        spec.add_rule(rule("h", vec![literal("p"), literal("q")]));
        assert!(sort_check(&spec).is_empty());
    }

    #[test]
    fn score_in_literal_position_is_found() {
        // TierLookup with out-of-[0,1] values is a Score, not Truth.
        let mut spec = spec_with_predicates(&["p"]);
        spec.add_predicate("tier", tier_lookup(vec![1.5, 0.8]));
        spec.add_rule(rule("h", vec![literal("p"), literal("tier")]));
        let findings = sort_check(&spec);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].sort, OutputSort::Score);
        assert_eq!(findings[0].severity, SortSeverity::Warn);
        assert!(findings[0].message.contains("'tier'"));
    }

    #[test]
    fn tier_lookup_with_unit_values_sorts_truth() {
        let mut spec = spec_with_predicates(&["p"]);
        spec.add_predicate("tier", tier_lookup(vec![0.9, 0.2]));
        spec.add_rule(rule("h", vec![literal("tier")]));
        assert!(sort_check(&spec).is_empty());
    }

    #[test]
    fn pairwise_modes_sort_by_variant() {
        assert_eq!(
            sort_of_name(&spec_with_predicates(&[]), "missing"),
            OutputSort::Unknown
        );
        let mut spec = RuleSpec::new("pairwise");
        spec.add_predicate("any", pairwise(PairwiseOutput::AnyAboveThreshold));
        spec.add_predicate("max", pairwise(PairwiseOutput::MaxDifference));
        spec.add_predicate("count", pairwise(PairwiseOutput::CountAboveThreshold));
        spec.add_predicate("best", pairwise(PairwiseOutput::BestIndex));
        spec.add_predicate("all", pairwise(PairwiseOutput::AllPairs));
        assert_eq!(sort_of_name(&spec, "any"), OutputSort::Truth);
        assert_eq!(sort_of_name(&spec, "max"), OutputSort::Score);
        assert_eq!(sort_of_name(&spec, "count"), OutputSort::Count);
        assert_eq!(sort_of_name(&spec, "best"), OutputSort::Index);
        assert_eq!(sort_of_name(&spec, "all"), OutputSort::Score);

        spec.add_rule(Rule::new(
            "h",
            vec![literal("any"), literal("best"), literal("count")],
        ));
        let findings = sort_check(&spec);
        assert_eq!(findings.len(), 2);
    }

    #[test]
    fn external_sorts_truth_by_contract() {
        let mut spec = spec_with_predicates(&["p"]);
        spec.add_predicate("ext", PredicateSpec::External);
        spec.add_rule(rule("h", vec![literal("ext")]));
        assert!(sort_check(&spec).is_empty());
    }

    #[test]
    fn composite_inherits_non_truth_children() {
        let mut spec = spec_with_predicates(&["p"]);
        spec.add_predicate("tier", tier_lookup(vec![1.5, 0.8]));
        spec.add_predicate(
            "combo",
            PredicateSpec::Composite {
                operator: crate::compiler::CompositeOp::And,
                children: vec!["p".to_string(), "tier".to_string()],
            },
        );
        assert_eq!(sort_of_name(&spec, "combo"), OutputSort::Score);
        spec.add_rule(rule("h", vec![literal("combo")]));
        let findings = sort_check(&spec);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].sort, OutputSort::Score);
    }

    #[test]
    fn composite_unknown_child_is_reported() {
        let mut spec = spec_with_predicates(&["p"]);
        spec.add_predicate(
            "combo",
            PredicateSpec::Composite {
                operator: crate::compiler::CompositeOp::Or,
                children: vec!["p".to_string(), "ghost".to_string()],
            },
        );
        spec.add_rule(rule("h", vec![literal("p")]));
        let findings = sort_check(&spec);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].sort, OutputSort::Unknown);
        assert!(matches!(
            findings[0].location,
            SortLocation::CompositeChild { .. }
        ));
    }

    #[test]
    fn head_sort_follows_transitive_closure() {
        // stage :- tier(score). esc :- stage. esc's literal use flags Score.
        let mut spec = spec_with_predicates(&["p"]);
        spec.add_predicate("tier", tier_lookup(vec![1.5, 0.8]));
        spec.add_rule(rule("stage", vec![literal("tier")]));
        spec.add_rule(rule("esc", vec![literal("stage")]));
        assert_eq!(sort_of_name(&spec, "esc"), OutputSort::Score);
        let findings = sort_check(&spec);
        // stage's body literal (tier) + esc's body literal (stage): 2 findings.
        assert_eq!(findings.len(), 2);
    }

    #[test]
    fn unknown_literals_belong_to_l0_6_not_sort_check() {
        let mut spec = spec_with_predicates(&["p"]);
        spec.add_rule(rule("h", vec![literal("ghost")]));
        // Sort-check stays silent; L0.6 (definedness) owns this finding.
        assert!(sort_check(&spec).is_empty());
    }

    #[test]
    fn deny_policy_escalates_severity() {
        let mut spec = spec_with_predicates(&["p"]);
        spec.add_predicate("tier", tier_lookup(vec![1.5, 0.8]));
        spec.add_rule(rule("h", vec![literal("tier")]));
        let policy = SortPolicy {
            deny_non_truth_literals: true,
        };
        let findings = sort_check_with_policy(&spec, &policy);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].severity, SortSeverity::Deny);
    }

    #[test]
    fn reference_cycle_resolves_unknown() {
        let mut spec = spec_with_predicates(&["p"]);
        spec.add_predicate(
            "a",
            PredicateSpec::Composite {
                operator: crate::compiler::CompositeOp::And,
                children: vec!["b".to_string()],
            },
        );
        spec.add_predicate(
            "b",
            PredicateSpec::Composite {
                operator: crate::compiler::CompositeOp::And,
                children: vec!["a".to_string()],
            },
        );
        assert_eq!(sort_of_name(&spec, "a"), OutputSort::Unknown);
        spec.add_rule(rule("h", vec![literal("p")]));
        // Cycle members unused by rules: no literal findings (Unknown at the
        // literal position would also skip — L0.6-adjacent, out of scope).
        assert!(sort_check(&spec).is_empty());
    }

    #[test]
    fn literal_position_uses_predicate_not_argument() {
        // Args are erased propositionally; sorts key on the predicate name.
        let mut spec = spec_with_predicates(&["p"]);
        spec.add_predicate("tier", tier_lookup(vec![1.5, 0.8]));
        spec.add_rule(Rule::new(
            "h",
            vec![Literal::positive(
                "tier",
                vec![crate::compiler::Argument::Constant(50.0)],
            )],
        ));
        assert_eq!(sort_check(&spec).len(), 1);
    }
}

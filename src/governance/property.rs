//! L0 property catalog (Slice 1).
//!
//! [`derive_l0`] auto-derives structural properties from a [`RuleSpec`]
//! (PPL v0.1 §4). Fail-grade properties feed Verdicts; lint-grade properties
//! feed `refine` suggestions and never feed Verdicts. Deciding the properties
//! lives in [`crate::governance::checker`]; this module only derives.
//!
//! Polarity convention (uniform across grades): `holds == true` means
//! good/no-action-needed. For lints, `holds == false` means the pattern was
//! detected and feeds `refine`. Grade routes the action (verdict vs refine),
//! never the polarity.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::compiler::RuleSpec;

/// Declared governance inputs: exclusion sets (L0.4) and trigger sets (L0.5).
/// Everything else in L0 derives without author input.
#[derive(Debug, Clone, Default)]
pub struct L0Declarations {
    /// Mutually exclusive head sets: at most one head in each set may hold.
    pub exclusions: Vec<Vec<String>>,
    /// Per-head trigger requirements: every activation path for `head` must
    /// pass through at least one of `triggers` (defined predicates or heads).
    pub triggers: Vec<TriggerDecl>,
}

/// One trigger requirement: `head` implies the disjunction of `triggers`.
#[derive(Debug, Clone)]
pub struct TriggerDecl {
    /// Head the requirement applies to.
    pub head: String,
    /// Predicates/heads at least one activation path must pass through.
    pub triggers: Vec<String>,
}

/// Fail-grade feeds Verdicts; lint-grade feeds `refine`, never Verdicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Grade {
    /// Feeds Verdicts.
    Fail,
    /// Feeds refine suggestions, never Verdicts.
    Lint,
}

/// One L0 property. See PPL v0.1 §4 for the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Property {
    /// Which catalog entry this is.
    pub kind: PropertyKind,
    /// Human-readable statement.
    pub description: String,
}

/// The L0.1–L0.9 catalog, plus `InvalidDeclaration` (the L0.6 family extended
/// to declaration inputs: a declaration naming unknown heads/predicates is a
/// fail-grade finding, not a silent skip).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyKind {
    /// L0.1: rule `rule_idx`'s body is satisfiable.
    BodySat {
        /// Index into the spec's rule list.
        rule_idx: usize,
    },
    /// L0.2: `head` is satisfiable (not vacuously false).
    HeadCoverage {
        /// Head predicate name.
        head: String,
    },
    /// L0.3: `head` is falsifiable (not a tautology).
    HeadNontrivial {
        /// Head predicate name.
        head: String,
    },
    /// L0.4 (declared): heads in `set` are pairwise exclusive.
    Exclusion {
        /// Mutually exclusive head names.
        set: Vec<String>,
    },
    /// L0.5 (declared): `head` implies the disjunction of `triggers`.
    TriggerPresence {
        /// Head predicate name.
        head: String,
        /// Required trigger names.
        triggers: Vec<String>,
    },
    /// L0.6: the existing validator reports no errors (imported, $0 solver).
    DefinedAritySafety,
    /// L0.7: rule `rule_idx`'s declared weight is finite.
    WeightFinite {
        /// Index into the spec's rule list.
        rule_idx: usize,
    },
    /// L0.6 family: a declaration is malformed (unknown names, degenerate
    /// sets). Fails by construction — the derivation IS the finding.
    InvalidDeclaration {
        /// What is malformed.
        message: String,
    },
    /// L0.8 (lint): rule `subsumed_idx`'s body implies rule
    /// `subsuming_idx`'s body (redundancy).
    Subsumption {
        /// Head predicate name.
        head: String,
        /// Index of the redundant rule.
        subsumed_idx: usize,
        /// Index of the subsuming rule.
        subsuming_idx: usize,
    },
    /// L0.9 (lint): predicate `name` is defined but never referenced.
    UnusedPredicate {
        /// Unreferenced predicate name.
        name: String,
    },
    /// L0.9 (lint): two rules share a head and an identical body.
    DuplicateRule {
        /// Head predicate name.
        head: String,
        /// Index of the first occurrence.
        first_idx: usize,
        /// Index of the duplicate.
        second_idx: usize,
    },
}

impl Property {
    /// Grade routes the action (verdict vs refine), never the polarity.
    pub fn grade(&self) -> Grade {
        match &self.kind {
            PropertyKind::Subsumption { .. }
            | PropertyKind::UnusedPredicate { .. }
            | PropertyKind::DuplicateRule { .. } => Grade::Lint,
            _ => Grade::Fail,
        }
    }
}

/// Derive every L0 property for `spec` (+ declared sets). Pure function;
/// deterministic order: rules by index, heads sorted, declarations as given.
pub fn derive_l0(spec: &RuleSpec, decls: &L0Declarations) -> Vec<Property> {
    let mut properties = Vec::new();

    // L0.1: every rule body satisfiable.
    for (rule_idx, rule) in spec.rules.iter().enumerate() {
        properties.push(Property {
            kind: PropertyKind::BodySat { rule_idx },
            description: format!("rule {rule_idx} ({}): body is satisfiable", rule.head),
        });
    }

    // Heads in sorted order — deterministic across runs.
    let heads: BTreeSet<&str> = spec.rules.iter().map(|rule| rule.head.as_str()).collect();

    // L0.2 + L0.3: every head covered and nontrivial.
    for head in &heads {
        properties.push(Property {
            kind: PropertyKind::HeadCoverage {
                head: head.to_string(),
            },
            description: format!("head '{head}': satisfiable (not vacuously false)"),
        });
        properties.push(Property {
            kind: PropertyKind::HeadNontrivial {
                head: head.to_string(),
            },
            description: format!("head '{head}': falsifiable (not a tautology)"),
        });
    }

    // L0.4: declared exclusion sets (validated, never silently skipped).
    for set in &decls.exclusions {
        let mut distinct: Vec<String> = Vec::new();
        for name in set {
            if !distinct.contains(name) {
                distinct.push(name.clone());
            }
        }
        let unknown: Vec<&String> = distinct
            .iter()
            .filter(|name| !heads.contains(name.as_str()))
            .collect();
        if !unknown.is_empty() {
            properties.push(invalid_declaration(format!(
                "exclusion set {set:?} names unknown head(s): {unknown:?}"
            )));
        } else if distinct.len() < 2 {
            properties.push(invalid_declaration(format!(
                "exclusion set {set:?} needs at least 2 distinct heads"
            )));
        } else {
            properties.push(Property {
                kind: PropertyKind::Exclusion { set: distinct },
                description: format!("exclusion: at most one of {set:?} holds"),
            });
        }
    }

    // L0.5: declared trigger sets (validated the same way). A trigger needs
    // executable meaning: a head (defined by rules) or a defined predicate
    // (has a kernel). Referenced-but-undefined names are rejected — and
    // separately flagged by L0.6.
    let defined: BTreeSet<&str> = spec.predicates.keys().map(String::as_str).collect();
    for decl in &decls.triggers {
        if !heads.contains(decl.head.as_str()) {
            properties.push(invalid_declaration(format!(
                "trigger requirement names unknown head '{}'",
                decl.head
            )));
            continue;
        }
        let unknown: Vec<&String> = decl
            .triggers
            .iter()
            .filter(|name| !heads.contains(name.as_str()) && !defined.contains(name.as_str()))
            .collect();
        if !unknown.is_empty() {
            properties.push(invalid_declaration(format!(
                "trigger requirement for '{}' names unknown predicate(s): {unknown:?}",
                decl.head
            )));
            continue;
        }
        if decl.triggers.is_empty() {
            properties.push(invalid_declaration(format!(
                "trigger requirement for '{}' lists no triggers",
                decl.head
            )));
            continue;
        }
        properties.push(Property {
            kind: PropertyKind::TriggerPresence {
                head: decl.head.clone(),
                triggers: decl.triggers.clone(),
            },
            description: format!(
                "trigger presence: '{}' implies ({})",
                decl.head,
                decl.triggers.join(" | ")
            ),
        });
    }

    // L0.6: validator import (exactly one property; the checker runs it).
    properties.push(Property {
        kind: PropertyKind::DefinedAritySafety,
        description: String::from(
            "validator: defined predicates, consistent arity, no reserved words",
        ),
    });

    // L0.7: finiteness for rules that declare weights.
    for (rule_idx, rule) in spec.rules.iter().enumerate() {
        if rule.weight.is_some() {
            properties.push(Property {
                kind: PropertyKind::WeightFinite { rule_idx },
                description: format!("rule {rule_idx}: declared weight is finite"),
            });
        }
    }

    // L0.8: ordered subsumption pairs per head (rule_idx order is stable).
    for head in &heads {
        let indices: Vec<usize> = spec
            .rules
            .iter()
            .enumerate()
            .filter(|(_, rule)| rule.head == *head)
            .map(|(i, _)| i)
            .collect();
        for &subsumed_idx in &indices {
            for &subsuming_idx in &indices {
                if subsumed_idx == subsuming_idx {
                    continue;
                }
                properties.push(Property {
                    kind: PropertyKind::Subsumption {
                        head: head.to_string(),
                        subsumed_idx,
                        subsuming_idx,
                    },
                    description: format!(
                        "subsumption: rule {subsumed_idx} is not redundant given rule {subsuming_idx}"
                    ),
                });
            }
        }
    }

    // L0.9: unused defined predicates.
    let referenced: BTreeSet<&str> = spec
        .rules
        .iter()
        .flat_map(|rule| rule.body.iter().map(|lit| lit.predicate.as_str()))
        .collect();
    let mut defined: Vec<&String> = spec.predicates.keys().collect();
    defined.sort();
    for name in defined {
        if !referenced.contains(name.as_str()) {
            properties.push(Property {
                kind: PropertyKind::UnusedPredicate { name: name.clone() },
                description: format!("predicate '{name}' is defined but never referenced"),
            });
        }
    }

    // L0.9: duplicate rules (same head, identical literal multiset).
    // Args are ignored: the compiler erases them (propositional evaluation),
    // so arg-differing bodies with identical literal shapes ARE duplicates
    // at execution. The fingerprint sorts (predicate, negated) pairs.
    for head in &heads {
        let indices: Vec<usize> = spec
            .rules
            .iter()
            .enumerate()
            .filter(|(_, rule)| rule.head == *head)
            .map(|(i, _)| i)
            .collect();
        for (a_pos, &first_idx) in indices.iter().enumerate() {
            for &second_idx in &indices[a_pos + 1..] {
                if body_fingerprint(&spec.rules[first_idx])
                    == body_fingerprint(&spec.rules[second_idx])
                {
                    properties.push(Property {
                        kind: PropertyKind::DuplicateRule {
                            head: head.to_string(),
                            first_idx,
                            second_idx,
                        },
                        description: format!(
                            "rules {first_idx} and {second_idx} (head '{head}') have identical bodies"
                        ),
                    });
                }
            }
        }
    }

    properties
}

fn invalid_declaration(message: String) -> Property {
    Property {
        kind: PropertyKind::InvalidDeclaration {
            message: message.clone(),
        },
        description: format!("invalid declaration: {message}"),
    }
}

fn body_fingerprint(rule: &crate::compiler::Rule) -> Vec<(&str, bool)> {
    let mut fingerprint: Vec<(&str, bool)> = rule
        .body
        .iter()
        .map(|lit| (lit.predicate.as_str(), lit.negated))
        .collect();
    fingerprint.sort();
    fingerprint
}

/// Test fixtures shared by governance test suites (derivation + checking
/// speak the same fixture language).
#[cfg(test)]
pub(crate) mod fixtures {
    use crate::compiler::{Argument, Literal, PredicateSpec, Rule};

    pub(crate) fn threshold_predicate() -> PredicateSpec {
        PredicateSpec::Threshold {
            input: "x".to_string(),
            threshold: 0.5,
            greater_than: true,
            sharpness: 10.0,
        }
    }

    pub(crate) fn literal(predicate: &str) -> Literal {
        Literal::positive(predicate, vec![Argument::Variable("X".to_string())])
    }

    pub(crate) fn negated_literal(predicate: &str) -> Literal {
        Literal::negated(predicate, vec![Argument::Variable("X".to_string())])
    }

    pub(crate) fn rule(head: &str, body: Vec<Literal>) -> Rule {
        Rule::new(head, body)
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;
    use crate::compiler::{Argument, Literal, Rule};

    fn empty_decls() -> L0Declarations {
        L0Declarations::default()
    }

    #[test]
    fn derives_body_coverage_and_nontrivial_per_rule_and_head() {
        let mut spec = RuleSpec::new("shapes");
        spec.add_rule(rule("h", vec![literal("p")]));
        spec.add_rule(rule("h", vec![literal("q")]));
        spec.add_predicate("p", threshold_predicate());
        spec.add_predicate("q", threshold_predicate());

        let kinds: Vec<PropertyKind> = derive_l0(&spec, &empty_decls())
            .iter()
            .map(|p| p.kind.clone())
            .collect();
        assert!(kinds.contains(&PropertyKind::BodySat { rule_idx: 0 }));
        assert!(kinds.contains(&PropertyKind::BodySat { rule_idx: 1 }));
        assert!(kinds.contains(&PropertyKind::HeadCoverage {
            head: "h".to_string()
        }));
        assert!(kinds.contains(&PropertyKind::HeadNontrivial {
            head: "h".to_string()
        }));
        assert!(kinds.contains(&PropertyKind::DefinedAritySafety));
    }

    #[test]
    fn weight_properties_derive_only_for_weighted_rules() {
        let mut spec = RuleSpec::new("weights");
        spec.add_rule(rule("h", vec![literal("p")]).with_weight(0.7));
        spec.add_rule(rule("h", vec![literal("q")]));
        spec.add_predicate("p", threshold_predicate());
        spec.add_predicate("q", threshold_predicate());

        let kinds: Vec<PropertyKind> = derive_l0(&spec, &empty_decls())
            .iter()
            .map(|p| p.kind.clone())
            .collect();
        assert!(kinds.contains(&PropertyKind::WeightFinite { rule_idx: 0 }));
        assert!(!kinds.contains(&PropertyKind::WeightFinite { rule_idx: 1 }));
    }

    #[test]
    fn unknown_exclusion_head_becomes_invalid_declaration() {
        let mut spec = RuleSpec::new("excl");
        spec.add_rule(rule("a", vec![literal("p")]));
        spec.add_predicate("p", threshold_predicate());
        let decls = L0Declarations {
            exclusions: vec![vec!["a".to_string(), "ghost".to_string()]],
            triggers: vec![],
        };

        let properties = derive_l0(&spec, &decls);
        assert!(
            properties.iter().any(|p| matches!(
                &p.kind,
                PropertyKind::InvalidDeclaration { message }
                if message.contains("ghost")
            )),
            "expected an InvalidDeclaration naming 'ghost'"
        );
        assert!(
            !properties
                .iter()
                .any(|p| matches!(p.kind, PropertyKind::Exclusion { .. })),
            "no Exclusion property must derive from a bad declaration"
        );
    }

    #[test]
    fn degenerate_exclusion_set_becomes_invalid_declaration() {
        let mut spec = RuleSpec::new("excl");
        spec.add_rule(rule("a", vec![literal("p")]));
        spec.add_predicate("p", threshold_predicate());
        let decls = L0Declarations {
            exclusions: vec![vec!["a".to_string(), "a".to_string()]],
            triggers: vec![],
        };
        assert!(derive_l0(&spec, &decls)
            .iter()
            .any(|p| matches!(&p.kind, PropertyKind::InvalidDeclaration { .. })));
    }

    #[test]
    fn unused_predicate_derives_and_referenced_does_not() {
        let mut spec = RuleSpec::new("unused");
        spec.add_rule(rule("h", vec![literal("p")]));
        spec.add_predicate("p", threshold_predicate());
        spec.add_predicate("orphan", threshold_predicate());

        let properties = derive_l0(&spec, &empty_decls());
        assert!(properties.iter().any(|p| matches!(
            &p.kind,
            PropertyKind::UnusedPredicate { name } if name == "orphan"
        )));
        assert!(!properties.iter().any(|p| matches!(
            &p.kind,
            PropertyKind::UnusedPredicate { name } if name == "p"
        )));
    }

    #[test]
    fn duplicate_rules_detected_args_ignored() {
        // Same literal shape, different arg spellings: duplicates at execution
        // because the compiler erases args (propositional evaluation).
        let mut spec = RuleSpec::new("dup");
        spec.add_rule(rule("h", vec![literal("p")]));
        spec.add_rule(Rule::new(
            "h",
            vec![Literal::positive(
                "p",
                vec![Argument::Variable("Y".to_string()), Argument::Constant(1.0)],
            )],
        ));
        spec.add_predicate("p", threshold_predicate());

        assert!(derive_l0(&spec, &empty_decls()).iter().any(|p| matches!(
            &p.kind,
            PropertyKind::DuplicateRule {
                first_idx: 0,
                second_idx: 1,
                ..
            }
        )));
    }

    #[test]
    fn lint_grades_never_feed_verdicts() {
        // Grade routing is the verdict/lint firewall: fail-grade kinds must
        // never report Lint, lint-grade kinds must never report Fail.
        let mut spec = RuleSpec::new("grades");
        spec.add_rule(rule("h", vec![literal("p"), literal("q")]));
        spec.add_rule(rule("h", vec![literal("p")]));
        spec.add_predicate("p", threshold_predicate());
        spec.add_predicate("q", threshold_predicate());
        spec.add_predicate("orphan", threshold_predicate());

        for property in derive_l0(&spec, &empty_decls()) {
            match &property.kind {
                PropertyKind::Subsumption { .. }
                | PropertyKind::UnusedPredicate { .. }
                | PropertyKind::DuplicateRule { .. } => {
                    assert_eq!(property.grade(), Grade::Lint, "{:?}", property.kind)
                }
                _ => assert_eq!(property.grade(), Grade::Fail, "{:?}", property.kind),
            }
        }
    }
}

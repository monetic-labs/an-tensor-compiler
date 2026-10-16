//! Margin propagation calculus (Slice 2).
//!
//! Closed-form, solver-free: given leaf margins, compute per-head output
//! margins through the rule DAG (PPL v0.1 §5). All formulas assume the
//! governance threshold τ = 0.5 (decision D1): the hard reading commutes
//! with negation only there, and NOT swaps margin sides exactly.
//!
//! ## Semantics
//!
//! A leaf margin `(ε⁺, ε⁻)` promises: when the predicate is decided-true,
//! soft ≥ 0.5+ε⁺; when decided-false, soft ≤ 0.5−ε⁻. (Symmetric input
//! margins are the common case; the calculus tracks sides separately
//! because AND/OR shrink opposite sides.)
//!
//! - AND (product), true-side: `ε_out⁺ = Π(0.5+εᵢ⁺) − 0.5` — shrinks
//!   exponentially in body length (the t-norm gap, PPL finding F-A).
//! - AND, false-side: `ε_out⁻ = min εᵢ⁻` — preserved (a single false
//!   literal caps the product regardless of the rest).
//! - OR (probabilistic s-norm): mirror image — true-side preserved
//!   (`min εᵢ⁺`), false-side shrinks (`Π(0.5+εᵢ⁻) − 0.5`).
//! - NOT: sides swap exactly (`ε_out⁺ = ε_in⁻`, vice versa).
//!
//! Literals naming intermediate heads use the head's computed margin, so
//! heads evaluate in dependency order. Dependency cycles fall back to leaf
//! defaults for cycle members and are reported in [`MarginReport::notes`]
//! (margins are advisory — triage ranking, abstention guidance — so a cycle
//! degrades that head's report instead of failing propagation).
//!
//! Where `ε_out ≤ 0` on a side, the head is **unresolvable-at-margin** on
//! that side: the abstention band widens across the threshold side it
//! touches (see `sample::band_decision` for the exact clear/abstain rule —
//! two-sided exclusion, not "nothing ever clears").

use std::collections::{BTreeMap, BTreeSet};

use crate::compiler::RuleSpec;

/// Output margins for one head: guaranteed clearance outside the abstention
/// band per side. Non-positive on a side means unresolvable-at-margin there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeMargin {
    /// Clearance above τ when decided-true (`soft ≥ 0.5 + eps_true`).
    pub eps_true: f64,
    /// Clearance below τ when decided-false (`soft ≤ 0.5 − eps_false`).
    pub eps_false: f64,
    /// True when dependency-cycle fallback supplied this head's inputs.
    pub cyclic: bool,
}

impl NodeMargin {
    /// Symmetric margins (the common leaf case).
    pub fn symmetric(eps: f64) -> Self {
        Self {
            eps_true: eps,
            eps_false: eps,
            cyclic: false,
        }
    }

    /// True-side resolvable at margin.
    pub fn resolvable_true(&self) -> bool {
        self.eps_true > 0.0
    }

    /// False-side resolvable at margin.
    pub fn resolvable_false(&self) -> bool {
        self.eps_false > 0.0
    }

    /// Unresolvable on at least one side → abstention region.
    pub fn unresolvable(&self) -> bool {
        !self.resolvable_true() || !self.resolvable_false()
    }

    /// NOT swaps sides exactly (requires τ = 0.5).
    fn negated(self) -> Self {
        Self {
            eps_true: self.eps_false,
            eps_false: self.eps_true,
            cyclic: self.cyclic,
        }
    }
}

/// Leaf-margin configuration: a default plus per-predicate overrides.
/// (Calibrator-fed ε lands with O4; F-1 uses configured constants.)
#[derive(Debug, Clone)]
pub struct MarginConfig {
    /// Default symmetric leaf margin.
    pub default_eps: f64,
    /// Per-predicate overrides (heads and leaves alike).
    pub overrides: BTreeMap<String, NodeMargin>,
}

impl MarginConfig {
    /// Uniform symmetric margins everywhere.
    pub fn uniform(default_eps: f64) -> Self {
        Self {
            default_eps,
            overrides: BTreeMap::new(),
        }
    }

    fn leaf_margin(&self, name: &str) -> NodeMargin {
        self.overrides
            .get(name)
            .copied()
            .unwrap_or(NodeMargin::symmetric(self.default_eps))
    }
}

/// Propagation output: per-head margins plus advisory notes (cycles,
/// unresolvable heads). Heads keyed by name (sorted); notes in
/// deterministic order.
#[derive(Debug, Clone)]
pub struct MarginReport {
    /// Per-head output margins.
    pub heads: BTreeMap<String, NodeMargin>,
    /// Advisory notes: dependency cycles, unresolvable heads.
    pub notes: Vec<String>,
}

/// Propagate leaf margins through `spec` to per-head output margins.
/// Pure function; deterministic for identical inputs.
pub fn propagate_margins(spec: &RuleSpec, config: &MarginConfig) -> MarginReport {
    // Heads in dependency order (dependencies first); cycles trail with notes.
    let order = head_order(spec);
    let mut heads: BTreeMap<String, NodeMargin> = BTreeMap::new();
    let mut notes = Vec::new();

    for head in &order.sorted {
        heads.insert(head.clone(), head_margin(spec, head, &heads, config));
    }
    for head in &order.cyclic {
        // Cycle fallback: members evaluate against leaf defaults rather than
        // each other's uncomputed margins; flagged, never silent.
        notes.push(format!(
            "dependency cycle involving '{head}': inputs fall back to leaf defaults"
        ));
        let mut margin = head_margin(spec, head, &heads, config);
        margin.cyclic = true;
        heads.insert(head.clone(), margin);
    }

    let mut unresolvable: Vec<&String> = heads
        .iter()
        .filter(|(_, margin)| margin.unresolvable())
        .map(|(head, _)| head)
        .collect();
    unresolvable.sort();
    for head in unresolvable {
        let margin = &heads[head];
        notes.push(format!(
            "head '{head}' unresolvable-at-margin (ε⁺={:.4}, ε⁻={:.4})",
            margin.eps_true, margin.eps_false
        ));
    }

    MarginReport { heads, notes }
}

/// Margin of one head: OR over its rule-body margins.
fn head_margin(
    spec: &RuleSpec,
    head: &str,
    computed: &BTreeMap<String, NodeMargin>,
    config: &MarginConfig,
) -> NodeMargin {
    let bodies: Vec<NodeMargin> = spec
        .rules
        .iter()
        .filter(|rule| rule.head == head)
        .map(|rule| body_margin(rule, computed, config))
        .collect();
    or_margins(&bodies)
}

/// Margin of one rule body: AND over its literal margins. A literal naming
/// an already-computed head uses that head's margin; otherwise the leaf
/// margin (default or override). An empty body is vacuously true with zero
/// margins (no evidence either way — it contributes no clearance).
fn body_margin(
    rule: &crate::compiler::Rule,
    computed: &BTreeMap<String, NodeMargin>,
    config: &MarginConfig,
) -> NodeMargin {
    if rule.body.is_empty() {
        return NodeMargin::symmetric(0.0);
    }
    let literals: Vec<NodeMargin> = rule
        .body
        .iter()
        .map(|lit| {
            let base = computed
                .get(&lit.predicate)
                .copied()
                .unwrap_or_else(|| config.leaf_margin(&lit.predicate));
            if lit.negated {
                base.negated()
            } else {
                base
            }
        })
        .collect();
    and_margins(&literals)
}

/// AND (product t-norm): true-side shrinks multiplicatively, false-side is
/// the weakest false-side among the conjuncts.
fn and_margins(literals: &[NodeMargin]) -> NodeMargin {
    let eps_true = literals
        .iter()
        .map(|lit| 0.5 + lit.eps_true)
        .product::<f64>()
        - 0.5;
    let eps_false = literals
        .iter()
        .map(|lit| lit.eps_false)
        .fold(f64::INFINITY, f64::min);
    let cyclic = literals.iter().any(|lit| lit.cyclic);
    NodeMargin {
        eps_true,
        eps_false: if eps_false.is_infinite() {
            0.0
        } else {
            eps_false
        },
        cyclic,
    }
}

/// OR (probabilistic s-norm): mirror of AND — true-side is the weakest
/// true-side, false-side shrinks multiplicatively.
fn or_margins(terms: &[NodeMargin]) -> NodeMargin {
    if terms.is_empty() {
        return NodeMargin::symmetric(0.0);
    }
    let eps_true = terms
        .iter()
        .map(|term| term.eps_true)
        .fold(f64::INFINITY, f64::min);
    let eps_false = terms
        .iter()
        .map(|term| 0.5 + term.eps_false)
        .product::<f64>()
        - 0.5;
    let cyclic = terms.iter().any(|term| term.cyclic);
    NodeMargin {
        eps_true: if eps_true.is_infinite() {
            0.0
        } else {
            eps_true
        },
        eps_false,
        cyclic,
    }
}

/// Dependency order over heads: `sorted` has dependencies before dependents
/// (Kahn's algorithm over head→head references, deterministic via BTreeSet);
/// `cyclic` holds the leftovers when references cycle.
struct HeadOrder {
    sorted: Vec<String>,
    cyclic: Vec<String>,
}

fn head_order(spec: &RuleSpec) -> HeadOrder {
    // Edge head → dependency for every literal naming another head.
    let heads: BTreeSet<&str> = spec.rules.iter().map(|rule| rule.head.as_str()).collect();
    let mut dependencies: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for rule in &spec.rules {
        let entry = dependencies.entry(rule.head.as_str()).or_default();
        for lit in &rule.body {
            if heads.contains(lit.predicate.as_str()) && lit.predicate != rule.head {
                entry.insert(lit.predicate.as_str());
            }
        }
    }
    // Self-reference is a length-1 cycle, not a dependency.
    let mut cyclic: BTreeSet<&str> = BTreeSet::new();
    for rule in &spec.rules {
        if rule.body.iter().any(|lit| lit.predicate == rule.head) {
            cyclic.insert(rule.head.as_str());
        }
    }

    // Kahn's algorithm; BTreeSets keep every choice deterministic.
    let mut sorted = Vec::new();
    let mut emitted: BTreeSet<&str> = BTreeSet::new();
    loop {
        let ready: Vec<&str> = dependencies
            .iter()
            .filter(|(head, deps)| {
                !emitted.contains(**head)
                    && !cyclic.contains(**head)
                    && deps.iter().all(|dep| emitted.contains(dep))
            })
            .map(|(head, _)| *head)
            .collect();
        if ready.is_empty() {
            break;
        }
        for head in ready {
            emitted.insert(head);
            sorted.push(head.to_string());
        }
    }
    // Leftovers (minus the emitted) are cycle members, sorted.
    let mut leftover: Vec<String> = heads
        .iter()
        .filter(|head| !emitted.contains(**head))
        .map(ToString::to_string)
        .collect();
    leftover.sort();
    HeadOrder {
        sorted,
        cyclic: leftover,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::governance::property::fixtures::{
        literal, negated_literal, rule, threshold_predicate,
    };

    fn spec_with_predicates(names: &[&str]) -> RuleSpec {
        let mut spec = RuleSpec::new("margins");
        for name in names {
            spec.add_predicate(name.to_string(), threshold_predicate());
        }
        spec
    }

    fn approx_eq(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "actual {actual} != expected {expected}"
        );
    }

    #[test]
    fn and_true_side_shrinks_exponentially() {
        // Three leaves at ε=0.1: 0.6³ − 0.5 = −0.284 → unresolvable.
        let mut spec = spec_with_predicates(&["a", "b", "c"]);
        spec.add_rule(rule("h", vec![literal("a"), literal("b"), literal("c")]));
        let report = propagate_margins(&spec, &MarginConfig::uniform(0.1));
        let margin = &report.heads["h"];
        approx_eq(margin.eps_true, 0.6f64.powi(3) - 0.5);
        assert!(!margin.resolvable_true());
        assert!(margin.unresolvable());
        assert!(report
            .notes
            .iter()
            .any(|note| note.contains("unresolvable-at-margin")));
    }

    #[test]
    fn and_false_side_preserves_weakest_margin() {
        // AND with leaves at ε=0.1 and ε=0.3: false-side is min = 0.1.
        let mut spec = spec_with_predicates(&["a", "b"]);
        spec.add_rule(rule("h", vec![literal("a"), literal("b")]));
        let mut config = MarginConfig::uniform(0.1);
        config
            .overrides
            .insert("b".to_string(), NodeMargin::symmetric(0.3));
        let report = propagate_margins(&spec, &config);
        approx_eq(report.heads["h"].eps_false, 0.1);
    }

    #[test]
    fn or_mirrors_and() {
        // h :- a. h :- b. with ε=0.1: true-side preserved (0.1),
        // false-side shrinks (0.6² − 0.5 = −0.14).
        let mut spec = spec_with_predicates(&["a", "b"]);
        spec.add_rule(rule("h", vec![literal("a")]));
        spec.add_rule(rule("h", vec![literal("b")]));
        let report = propagate_margins(&spec, &MarginConfig::uniform(0.1));
        approx_eq(report.heads["h"].eps_true, 0.1);
        approx_eq(report.heads["h"].eps_false, 0.6f64.powi(2) - 0.5);
    }

    #[test]
    fn not_swaps_sides_exactly() {
        let mut spec = spec_with_predicates(&["a"]);
        spec.add_rule(rule("h", vec![negated_literal("a")]));
        let mut config = MarginConfig::uniform(0.0);
        config.overrides.insert(
            "a".to_string(),
            NodeMargin {
                eps_true: 0.2,
                eps_false: 0.4,
                cyclic: false,
            },
        );
        let report = propagate_margins(&spec, &config);
        // Body is a single negated literal: head = NOT a.
        approx_eq(report.heads["h"].eps_true, 0.4);
        approx_eq(report.heads["h"].eps_false, 0.2);
    }

    #[test]
    fn confident_short_body_stays_resolvable() {
        // Five leaves at 0.9 (ε=0.4): 0.9⁵ − 0.5 ≈ 0.09 > 0.
        let mut spec = spec_with_predicates(&["a", "b", "c", "d", "e"]);
        spec.add_rule(rule(
            "h",
            vec![
                literal("a"),
                literal("b"),
                literal("c"),
                literal("d"),
                literal("e"),
            ],
        ));
        let report = propagate_margins(&spec, &MarginConfig::uniform(0.4));
        approx_eq(report.heads["h"].eps_true, 0.9f64.powi(5) - 0.5);
        assert!(report.heads["h"].resolvable_true());
    }

    #[test]
    fn intermediate_heads_chain_in_dependency_order() {
        // stage :- low (ε=0.3). esc :- stage. esc inherits stage's margin.
        let mut spec = spec_with_predicates(&["low"]);
        spec.add_rule(rule("stage", vec![literal("low")]));
        spec.add_rule(rule("esc", vec![literal("stage")]));
        let report = propagate_margins(&spec, &MarginConfig::uniform(0.3));
        approx_eq(report.heads["stage"].eps_true, 0.3);
        approx_eq(report.heads["esc"].eps_true, 0.3);
        assert!(!report.heads["esc"].cyclic);
        assert!(report.notes.iter().all(|note| !note.contains("cycle")));
    }

    #[test]
    fn dependency_cycle_falls_back_and_flags() {
        // a :- b. b :- a. Both members fall back to leaf defaults, flagged.
        let mut spec = RuleSpec::new("cycle");
        spec.add_rule(rule("a", vec![literal("b")]));
        spec.add_rule(rule("b", vec![literal("a")]));
        let report = propagate_margins(&spec, &MarginConfig::uniform(0.25));
        assert!(report.heads["a"].cyclic);
        assert!(report.heads["b"].cyclic);
        approx_eq(report.heads["a"].eps_true, 0.25);
        assert!(report.notes.iter().any(|note| note.contains("cycle")));
    }

    #[test]
    fn deterministic_across_runs() {
        let mut spec = spec_with_predicates(&["z", "a", "m"]);
        spec.add_rule(rule("h", vec![literal("z"), literal("a")]));
        spec.add_rule(rule("g", vec![negated_literal("m")]));
        let first = propagate_margins(&spec, &MarginConfig::uniform(0.15));
        let second = propagate_margins(&spec, &MarginConfig::uniform(0.15));
        assert_eq!(format!("{:?}", first.heads), format!("{:?}", second.heads));
        assert_eq!(first.notes, second.notes);
    }
}

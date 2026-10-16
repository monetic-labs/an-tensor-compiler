//! Static verification for tensor-logic rules (governance core)
//!
//! Demonstrates: parse → add predicates → derive L0 → check → margins → sorts
//!
//! Run with:
//! ```bash
//! cargo run --example governance_demo
//! ```

use an_tensor_compiler::compiler::{PredicateSpec, RuleSpec};
use an_tensor_compiler::governance::{
    check_property, derive_l0, emit_smtlib, propagate_margins, sort_check, BoolSkeleton,
    L0Declarations, MarginConfig,
};
use an_tensor_compiler::{Result, TensorCoreError};

fn threshold(input: &str, theta: f32) -> PredicateSpec {
    PredicateSpec::Threshold {
        input: input.into(),
        threshold: theta,
        greater_than: true,
        sharpness: 10.0,
    }
}

fn main() -> Result<()> {
    // 1. Parse rules and declare their predicate groundings
    let mut spec = RuleSpec::parse(
        "policy_rules",
        r#"
        escalate(X) :- high_risk(X), not approved(X).
        escalate(X) :- critical_resource(X), low_confidence(X).
    "#,
    )?;
    spec.add_predicate("high_risk", threshold("risk_score", 0.7));
    spec.add_predicate("approved", threshold("approval_score", 0.5));
    spec.add_predicate("critical_resource", threshold("criticality", 0.8));
    spec.add_predicate("low_confidence", threshold("confidence", 0.4));

    // 2. Derive L0 structural properties (fully automatic, no spec-writing)
    let properties = derive_l0(&spec, &L0Declarations::default());
    println!("Derived {} L0 properties", properties.len());

    // 3. Check each property over the boolean skeleton
    let skeleton = BoolSkeleton::from_spec(&spec);
    for property in &properties {
        let result = check_property(&skeleton, &spec, property)
            .map_err(|e| TensorCoreError::Compiler(format!("check failed: {e:?}")))?;
        println!("  {:?} → holds={}", property.kind, result.holds);
    }

    // 4. Propagate margins: how much soft/hard disagreement can the rules take?
    //    Narrow leaf margins reproduce Finding F-A (product AND does not commute
    //    with thresholding): the head is honestly unresolvable. Wide margins clear.
    for eps in [0.1, 0.4] {
        let report = propagate_margins(&spec, &MarginConfig::uniform(eps));
        println!("\n--- Margins (leaf eps = {eps}) ---");
        for (head, margin) in &report.heads {
            println!("  {head}: {margin:?}");
        }
        for note in &report.notes {
            println!("  note: {note}");
        }
    }

    // 5. Sort-check: every literal position must be Truth-sorted
    let findings = sort_check(&spec);
    println!("\nSort findings: {}", findings.len());
    for finding in &findings {
        println!("  {finding:?}");
    }

    // 6. Emit SMT-LIB for one property (pipe to z3/cvc5 externally)
    if let Some(first) = properties.first() {
        let smtlib = emit_smtlib(&skeleton, first)
            .map_err(|e| TensorCoreError::Compiler(format!("SMT emission failed: {e:?}")))?;
        println!("\n--- SMT-LIB (first property, {} bytes) ---", smtlib.len());
        for line in smtlib.lines().take(8) {
            println!("  {line}");
        }
    }

    Ok(())
}

//! Basic rule compilation and evaluation
//!
//! Demonstrates: parse → compile → forward → explain
//!
//! Run with:
//! ```bash
//! cargo run --example basic_rules
//! ```

use an_tensor_compiler::prelude::*;
use std::collections::HashMap;

fn main() -> Result<()> {
    // 1. Parse rules from a Prolog-like DSL
    //
    // Rules with the same head predicate are compiled as disjunctions:
    //   escalate(X) if (high_risk AND NOT approved) OR (critical AND low_confidence)
    let spec = RuleSpec::parse(
        "policy_rules",
        r#"
        escalate(X) :- high_risk(X), not approved(X).
        escalate(X) :- critical_resource(X), low_confidence(X).
    "#,
    )?;

    println!("Parsed {} rules", spec.rules.len());
    println!("Predicates used: {:?}", spec.predicate_names());

    // 2. Compile to differentiable tensor function
    let compiled = CompiledRule::compile(spec)?;
    println!(
        "Compiled '{}' with {} params",
        compiled.name, compiled.param_count
    );

    // 3. Create inputs (fuzzy truth values in [0, 1])
    let device = best_device();
    let mut inputs = HashMap::new();
    inputs.insert(
        "high_risk".to_string(),
        Tensor::from_vec(vec![0.9f32], 1, &device)?,
    );
    inputs.insert(
        "approved".to_string(),
        Tensor::from_vec(vec![0.2f32], 1, &device)?,
    );
    inputs.insert(
        "critical_resource".to_string(),
        Tensor::from_vec(vec![0.5f32], 1, &device)?,
    );
    inputs.insert(
        "low_confidence".to_string(),
        Tensor::from_vec(vec![0.7f32], 1, &device)?,
    );

    // 4. Forward pass with full explanation
    let output = compiled.forward(&inputs)?;
    let result = output.output.to_vec1::<f32>()?[0];

    println!("\n--- Output ---");
    println!("Decision score: {:.3}", result);
    println!("Explanation: {}", output.explanation);
    println!("Rule activations: {:?}", output.rule_activations);
    println!("Predicate activations: {:?}", output.predicate_activations);

    // 5. Fast forward (no explanation, ~20x faster)
    let fast_result = compiled.forward_fast(&inputs)?;
    let fast_val = fast_result.to_vec1::<f32>()?[0];
    println!("\nFast forward: {:.3}", fast_val);

    Ok(())
}

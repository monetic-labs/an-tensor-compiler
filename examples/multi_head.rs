//! Multi-head rules with shared predicates
//!
//! Demonstrates: multiple head predicates compiled and evaluated independently
//!
//! Run with:
//! ```bash
//! cargo run --example multi_head
//! ```

use an_tensor_compiler::compiler::MultiHeadCompiledRule;
use an_tensor_compiler::prelude::*;
use std::collections::HashMap;

fn main() -> Result<()> {
    let device = best_device();

    // 1. Define rules with multiple heads sharing predicates
    let spec = RuleSpec::parse(
        "access_policy",
        r#"
        needs_review(X) :- high_risk(X).
        needs_review(X) :- critical_resource(X), not approved(X).
        auto_approve(X) :- low_risk(X), all_checks_pass(X).
        escalate(X) :- needs_review(X), urgent(X).
    "#,
    )?;

    println!(
        "Parsed {} rules with heads: {:?}",
        spec.rules.len(),
        spec.predicate_names()
    );

    // 2. Compile as multi-head
    let multi = MultiHeadCompiledRule::compile_on_device(spec, &device)?;
    println!(
        "Compiled {} heads: {:?}",
        multi.head_count(),
        multi.head_names()
    );

    // 3. Create inputs
    let mut inputs = HashMap::new();
    inputs.insert(
        "high_risk".to_string(),
        Tensor::from_vec(vec![0.85f32], 1, &device)?,
    );
    inputs.insert(
        "critical_resource".to_string(),
        Tensor::from_vec(vec![0.6f32], 1, &device)?,
    );
    inputs.insert(
        "approved".to_string(),
        Tensor::from_vec(vec![0.1f32], 1, &device)?,
    );
    inputs.insert(
        "low_risk".to_string(),
        Tensor::from_vec(vec![0.15f32], 1, &device)?,
    );
    inputs.insert(
        "all_checks_pass".to_string(),
        Tensor::from_vec(vec![0.95f32], 1, &device)?,
    );
    inputs.insert(
        "urgent".to_string(),
        Tensor::from_vec(vec![0.7f32], 1, &device)?,
    );
    inputs.insert(
        "needs_review".to_string(),
        Tensor::from_vec(vec![0.85f32], 1, &device)?,
    );

    // 4. Evaluate all heads at once
    let outputs = multi.forward_all(&inputs)?;
    println!("\n--- All Heads ---");
    for (head, output) in &outputs {
        let val = output.output.to_vec1::<f32>()?[0];
        println!("  {}: {:.3} — {}", head, val, output.explanation);
    }

    // 5. Or evaluate a single head
    let review = multi.forward_head("needs_review", &inputs)?;
    println!("\n--- Single Head ---");
    println!("needs_review: {:.3}", review.output.to_vec1::<f32>()?[0]);

    Ok(())
}

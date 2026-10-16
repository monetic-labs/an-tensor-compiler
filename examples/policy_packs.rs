//! Pre-built policy packs: interaction safety predicates + rules
//!
//! Demonstrates: register standard predicates → compile rules → forward.
//! Uses random vectors as stand-in embeddings (swap in any real embedder).
//!
//! Run with:
//! ```bash
//! cargo run --example policy_packs
//! ```

use an_tensor_compiler::interaction::InteractionContext;
use an_tensor_compiler::{Device, Result, Tensor};
use std::collections::HashMap;

fn rand_batch(dim: usize, device: &Device) -> Result<Tensor> {
    Ok(Tensor::randn(0f32, 1f32, (1, dim), device)?)
}

fn main() -> Result<()> {
    let device = Device::Cpu;

    // 1. Register the standard interaction predicates
    let mut ctx = InteractionContext::new(device.clone())?;
    ctx.register_standard_predicates()?;
    println!("Registered {} predicates", ctx.predicate_count());

    // 2. Compile the safety gate:
    //    safe_to_respond(X) :- message_safe(X), not blocked_user(X).
    let rules = ctx.compile_safety_rules()?;
    println!("Compiled '{}'", rules.name);

    // 3. Build inputs (random stand-ins with the shapes the specs declare)
    let mut inputs = HashMap::new();
    inputs.insert("content_embedding".to_string(), rand_batch(1536, &device)?);
    inputs.insert("intent_embedding".to_string(), rand_batch(1536, &device)?);
    inputs.insert("context_embedding".to_string(), rand_batch(1536, &device)?);
    inputs.insert(
        "context_age_minutes".to_string(),
        Tensor::from_vec(vec![12.0f32], 1, &device)?,
    );
    inputs.insert("query_embedding".to_string(), rand_batch(1536, &device)?);
    inputs.insert("response_embedding".to_string(), rand_batch(1536, &device)?);
    inputs.insert("window_embedding".to_string(), rand_batch(1536, &device)?);
    inputs.insert("risk_features".to_string(), rand_batch(32, &device)?);
    inputs.insert("sentiment_features".to_string(), rand_batch(64, &device)?);
    inputs.insert(
        "blocked_user".to_string(),
        Tensor::from_vec(vec![0.0f32], 1, &device)?,
    );
    // External predicates (rule-referenced, caller-supplied): a clean user
    // with no violations.
    inputs.insert(
        "severe_violation".to_string(),
        Tensor::from_vec(vec![0.0f32], 1, &device)?,
    );

    // 4. Forward: single-head safety gate
    let output = rules.forward(&inputs)?;
    let score = output.output.to_vec1::<f32>()?[0];
    println!("\nsafe_to_respond: {score:.3}");
    println!("Explanation: {}", output.explanation);

    // 5. Multi-head: independent decisions per concern
    let multi = ctx.compile_safety_multi_head()?;
    let outputs = multi.forward_all(&inputs)?;
    println!("\n--- Multi-head ---");
    let mut heads: Vec<_> = outputs.keys().collect();
    heads.sort();
    for head in heads {
        let value = outputs[head].output.to_vec1::<f32>()?[0];
        println!("  {head}: {value:.3}");
    }

    Ok(())
}

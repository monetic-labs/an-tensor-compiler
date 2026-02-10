//! Learned projection predicates with save/load
//!
//! Demonstrates: LearnedProjection predicate → compile → forward → save → load
//!
//! Run with:
//! ```bash
//! cargo run --example learned_projection
//! ```

use an_tensor_compiler::prelude::*;
use an_tensor_compiler::compiler::{PredicateSpec, Activation};
use std::collections::HashMap;

fn main() -> Result<()> {
    let device = best_device();

    // 1. Define rules with a learned projection predicate
    let mut spec = RuleSpec::parse("risk_scoring", r#"
        high_risk(X) :- risk_projection(X).
    "#)?;

    // Register a learned projection: embedding → hidden → sigmoid
    spec.add_predicate("risk_projection", PredicateSpec::LearnedProjection {
        inputs: vec!["features".into()],
        input_dim: 8,
        hidden_dim: 4,
        activation: Activation::GELU,
        attention_heads: None,
        attention_dropout: None,
        layer_norm: None,
        dropout: None,
        residual: None,
        conditioning_dim: None,
        conditioning_type: None,
        film_identity_init: None,
    });

    // 2. Compile
    let compiled = CompiledRule::compile_on_device(spec.clone(), &device)?;
    println!("Compiled with {} trainable parameters", compiled.param_count);

    // 3. Forward pass with random features
    let mut inputs = HashMap::new();
    inputs.insert("features".to_string(),
        Tensor::randn(0.0f32, 1.0, (1, 8), &device)?);

    let output = compiled.forward(&inputs)?;
    let result = output.output.flatten_all()?.to_vec1::<f32>()?[0];
    println!("Risk score: {:.4}", result);
    println!("Explanation: {}", output.explanation);

    // 4. Save learned weights
    let save_path = std::env::temp_dir().join("risk_model.safetensors");
    compiled.save(&save_path)?;
    println!("\nSaved to: {}", save_path.display());

    // 5. Load into a fresh instance
    let mut loaded = CompiledRule::compile_on_device(spec, &device)?;
    loaded.load(&save_path)?;

    // Verify identical output
    let loaded_output = loaded.forward(&inputs)?;
    let loaded_result = loaded_output.output.flatten_all()?.to_vec1::<f32>()?[0];
    println!("Loaded model score: {:.4} (should match {:.4})", loaded_result, result);
    assert!((result - loaded_result).abs() < 1e-6);

    // Cleanup
    std::fs::remove_file(&save_path).ok();
    println!("\nSave/load roundtrip verified.");

    Ok(())
}

//! Training loop with gradient descent
//!
//! Demonstrates: compile → forward → loss → backward → optimizer step
//!
//! Run with:
//! ```bash
//! cargo run --example training_loop
//! ```

use an_tensor_compiler::prelude::*;
use an_tensor_compiler::compiler::{PredicateSpec, Activation};
use an_tensor_compiler::training::{safe_optimizer_step, compute_grad_norm, check_gradients_health};
use candle_nn::optim::{AdamW, ParamsAdamW, Optimizer};
use std::collections::HashMap;

fn main() -> Result<()> {
    let device = best_device();

    // 1. Define a learnable rule
    let mut spec = RuleSpec::parse("classifier", r#"
        positive(X) :- score(X).
    "#)?;

    spec.add_predicate("score", PredicateSpec::LearnedProjection {
        inputs: vec!["features".into()],
        input_dim: 4,
        hidden_dim: 8,
        activation: Activation::ReLU,
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
    let compiled = CompiledRule::compile_on_device(spec, &device)?;
    let vars = compiled.trainable_vars();
    println!("Training with {} parameters ({} vars)", compiled.param_count, vars.len());

    // 3. Create optimizer
    let params = ParamsAdamW {
        lr: 0.01,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(vars.clone(), params)
        .map_err(|e| TensorCoreError::Tensor(format!("Optimizer init: {}", e)))?;

    // 4. Training loop
    println!("\n--- Training ---");
    for epoch in 0..50 {
        // Generate random features and target
        let features = Tensor::randn(0.0f32, 1.0, (1, 4), &device)?;
        // Target: positive when first feature > 0
        let first_feat = features.narrow(1, 0, 1)?.squeeze(1)?;
        let target = sigmoid(&(&first_feat * Tensor::new(&[5.0f32], &device)?)?)?;

        // Forward pass
        let mut inputs = HashMap::new();
        inputs.insert("features".to_string(), features);
        let output = compiled.forward_fast(&inputs)?;
        let pred = output.flatten_all()?;

        // Compute loss
        let loss = binary_cross_entropy(&pred, &target)?;

        // Backward pass
        let grads = loss.backward()
            .map_err(|e| TensorCoreError::Tensor(format!("backward: {}", e)))?;

        // Check gradient health
        if !check_gradients_health(&grads, &vars) {
            println!("  epoch {}: skipping (NaN gradients)", epoch);
            continue;
        }

        let grad_norm = compute_grad_norm(&grads, &vars)?;

        // Safe optimizer step with gradient clipping
        match safe_optimizer_step(&mut optimizer, &grads, &vars, 1.0, 0.01) {
            Ok(()) => {},
            Err(e) => {
                println!("  epoch {}: step failed: {}", epoch, e);
                continue;
            }
        }

        if epoch % 10 == 0 {
            let loss_val = loss.to_scalar::<f32>()?;
            println!("  epoch {:3}: loss={:.4}, grad_norm={:.4}", epoch, loss_val, grad_norm);
        }
    }

    // 5. Final evaluation
    let test_features = Tensor::randn(0.0f32, 1.0, (1, 4), &device)?;
    let mut test_inputs = HashMap::new();
    test_inputs.insert("features".to_string(), test_features);

    let final_output = compiled.forward(&test_inputs)?;
    let final_val = final_output.output.flatten_all()?.to_vec1::<f32>()?[0];
    println!("\nFinal prediction: {:.4}", final_val);
    println!("Explanation: {}", final_output.explanation);

    // Check weights health
    match compiled.check_weights_health() {
        Some((idx, issue)) => println!("Warning: {} at weight index {}", issue, idx),
        None => println!("All weights healthy."),
    }

    Ok(())
}

//! Fuzzy Logic Operators as Tensor Operations
//!
//! Differentiable fuzzy logic operators that form the foundation of Tensor Logic.
//!
//! ## Fuzzy Logic Semantics
//!
//! In classical logic, truth values are binary: {0, 1}.
//! In fuzzy logic, truth values are continuous: [0, 1].
//!
//! This allows gradients to flow through logical operations,
//! enabling learning of thresholds and rule weights.
//!
//! ## T-norms and S-norms
//!
//! We use the **product t-norm** for AND and **probabilistic sum** for OR:
//!
//! | Operation | Classical | Fuzzy (our choice) | Alternative |
//! |-----------|-----------|-------------------|-------------|
//! | AND(a,b)  | a ∧ b     | a × b             | min(a,b)    |
//! | OR(a,b)   | a ∨ b     | a + b - ab        | max(a,b)    |
//! | NOT(a)    | ¬a        | 1 - a             | 1 - a       |
//!
//! The product t-norm is preferred because:
//! - Gradients are non-zero everywhere (unlike min/max)
//! - Matches probabilistic interpretation (independence assumption)
//! - Numerical stability (no discontinuities)

use candle_core::{DType, Tensor};
use crate::{Result, TensorCoreError};

/// Fuzzy AND using product t-norm: AND(a, b) = a × b
///
/// # Properties
/// - AND(1, 1) = 1
/// - AND(0, x) = 0
/// - AND(0.5, 0.5) = 0.25
/// - Gradient: ∂AND/∂a = b, ∂AND/∂b = a
///
/// # Example
/// ```ignore
/// let a = Tensor::from_vec(vec![0.8f32], 1, &device)?;
/// let b = Tensor::from_vec(vec![0.9f32], 1, &device)?;
/// let result = fuzzy_and(&a, &b)?;
/// // result ≈ 0.72
/// ```
pub fn fuzzy_and(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.mul(b)
        .map_err(|e| TensorCoreError::Tensor(format!("fuzzy_and failed: {}", e)))
}

/// Fuzzy OR using probabilistic sum: OR(a, b) = a + b - ab
///
/// This is the dual of the product t-norm, also known as the
/// algebraic sum or probabilistic sum.
///
/// # Properties
/// - OR(0, 0) = 0
/// - OR(1, x) = 1
/// - OR(0.5, 0.5) = 0.75
/// - Gradient: ∂OR/∂a = 1 - b, ∂OR/∂b = 1 - a
pub fn fuzzy_or(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let sum = (a + b)
        .map_err(|e| TensorCoreError::Tensor(format!("fuzzy_or add failed: {}", e)))?;
    let product = a.mul(b)
        .map_err(|e| TensorCoreError::Tensor(format!("fuzzy_or mul failed: {}", e)))?;
    (sum - product)
        .map_err(|e| TensorCoreError::Tensor(format!("fuzzy_or sub failed: {}", e)))
}

/// Fuzzy NOT: NOT(a) = 1 - a
///
/// # Properties
/// - NOT(0) = 1
/// - NOT(1) = 0
/// - NOT(0.5) = 0.5
/// - Gradient: ∂NOT/∂a = -1
pub fn fuzzy_not(a: &Tensor) -> Result<Tensor> {
    // Use f32 to match input dtype (F32 inputs should stay F32)
    let one = Tensor::ones_like(a)
        .map_err(|e| TensorCoreError::Tensor(format!("fuzzy_not ones_like failed: {}", e)))?;
    (&one - a)
        .map_err(|e| TensorCoreError::Tensor(format!("fuzzy_not failed: {}", e)))
}

/// Fuzzy IMPLIES: a → b ≡ ¬a ∨ b
///
/// Using our definitions:
/// IMPLIES(a, b) = OR(NOT(a), b) = (1-a) + b - (1-a)b = 1 - a + ab
///
/// # Properties
/// - IMPLIES(0, x) = 1 (false implies anything)
/// - IMPLIES(1, 1) = 1
/// - IMPLIES(1, 0) = 0 (true cannot imply false)
pub fn fuzzy_implies(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let not_a = fuzzy_not(a)?;
    fuzzy_or(&not_a, b)
}

/// Multi-way AND: AND(a, b, c, ...) = a × b × c × ...
///
/// More efficient than chaining binary AND operations.
///
/// # Example
/// ```ignore
/// let terms = vec![&high_confidence, &low_risk, &has_tests];
/// let all_good = fuzzy_and_many(&terms)?;
/// ```
pub fn fuzzy_and_many(terms: &[&Tensor]) -> Result<Tensor> {
    if terms.is_empty() {
        return Err(TensorCoreError::Config(
            "fuzzy_and_many requires at least one term".into()
        ));
    }

    let mut result = terms[0].clone();
    for term in &terms[1..] {
        result = result.mul(term)
            .map_err(|e| TensorCoreError::Tensor(format!("fuzzy_and_many failed: {}", e)))?;
    }
    Ok(result)
}

/// Multi-way OR: OR(a, b, c, ...) = 1 - (1-a)(1-b)(1-c)...
///
/// This is equivalent to chaining binary OR but computed directly.
///
/// # Derivation
/// OR(a,b,c) = 1 - (1-a)(1-b)(1-c)
///
/// # Example
/// ```ignore
/// let terms = vec![&low_intent, &high_risk, &security_file, &many_warnings];
/// let should_escalate = fuzzy_or_many(&terms)?;
/// ```
pub fn fuzzy_or_many(terms: &[&Tensor]) -> Result<Tensor> {
    if terms.is_empty() {
        return Err(TensorCoreError::Config(
            "fuzzy_or_many requires at least one term".into()
        ));
    }

    // Compute 1 - ∏(1 - xᵢ)
    let mut product_of_complements = Tensor::ones_like(terms[0])
        .map_err(|e| TensorCoreError::Tensor(format!("ones_like failed: {}", e)))?;

    for term in terms {
        let complement = fuzzy_not(term)?;
        product_of_complements = product_of_complements.mul(&complement)
            .map_err(|e| TensorCoreError::Tensor(format!("fuzzy_or_many mul failed: {}", e)))?;
    }

    fuzzy_not(&product_of_complements)
}

/// Soft threshold: convert continuous value to fuzzy truth value
///
/// ```text
/// soft_threshold(x, θ, k) = σ((x - θ) × k)
/// ```
///
/// # Parameters
/// - `x`: Input tensor
/// - `threshold`: The threshold value (can be a Tensor for learned thresholds)
/// - `sharpness`: How sharp the transition is
///   - Positive: "greater than" semantics (x > θ → 1)
///   - Negative: "less than" semantics (x < θ → 1)
///   - |k| ≈ 1: gradual transition
///   - |k| ≈ 10: sharp transition
///   - |k| → ∞: approaches hard threshold
///
/// # Example
/// ```ignore
/// // "is high risk" = risk > 0.8
/// let high_risk = soft_threshold(&risk, &threshold_0_8, 10.0)?;
///
/// // "is low confidence" = confidence < 0.5
/// let low_confidence = soft_threshold(&confidence, &threshold_0_5, -10.0)?;
/// ```
pub fn soft_threshold(x: &Tensor, threshold: &Tensor, sharpness: f32) -> Result<Tensor> {
    let diff = (x - threshold)
        .map_err(|e| TensorCoreError::Tensor(format!("soft_threshold diff failed: {}", e)))?;

    // Use affine transformation to scale: y = sharpness * x + 0
    // This preserves dtype and is more efficient than scalar multiplication
    let scaled = diff.affine(sharpness as f64, 0.0)
        .map_err(|e| TensorCoreError::Tensor(format!("soft_threshold scale failed: {}", e)))?;

    super::activations::sigmoid(&scaled)
}

/// Soft threshold with scalar threshold value (convenience function)
pub fn soft_threshold_scalar(x: &Tensor, threshold: f32, sharpness: f32) -> Result<Tensor> {
    // Create threshold tensor matching input dtype
    let threshold_tensor = Tensor::new(&[threshold], x.device())
        .map_err(|e| TensorCoreError::Tensor(format!("soft_threshold_scalar threshold tensor failed: {}", e)))?;

    // Broadcast threshold to match input shape if needed
    let threshold_broadcast = threshold_tensor.broadcast_as(x.shape())
        .map_err(|e| TensorCoreError::Tensor(format!("soft_threshold_scalar broadcast failed: {}", e)))?;

    soft_threshold(x, &threshold_broadcast, sharpness)
}

/// Hard threshold (for comparison/testing only - not differentiable)
///
/// Returns 1.0 where x > threshold, 0.0 otherwise.
/// **Warning**: Gradients are zero everywhere, not useful for learning.
pub fn hard_threshold(x: &Tensor, threshold: f32) -> Result<Tensor> {
    x.ge(threshold as f64)
        .map_err(|e| TensorCoreError::Tensor(format!("hard_threshold failed: {}", e)))?
        .to_dtype(DType::F32)
        .map_err(|e| TensorCoreError::Tensor(format!("hard_threshold dtype failed: {}", e)))
}

/// Weighted combination of rule activations
///
/// Given rule activations [r₁, r₂, ..., rₙ] and weights [w₁, w₂, ..., wₙ],
/// computes: σ(Σᵢ wᵢ × rᵢ)
///
/// This allows rules to have different importance levels.
pub fn weighted_rule_combination(
    activations: &Tensor,  // [batch, num_rules]
    weights: &Tensor,      // [num_rules]
) -> Result<Tensor> {
    let weighted = activations.matmul(&weights.unsqueeze(1)
        .map_err(|e| TensorCoreError::Tensor(format!("unsqueeze failed: {}", e)))?)
        .map_err(|e| TensorCoreError::Tensor(format!("matmul failed: {}", e)))?;

    let squeezed = weighted.squeeze(1)
        .map_err(|e| TensorCoreError::Tensor(format!("squeeze failed: {}", e)))?;

    super::activations::sigmoid(&squeezed)
}

/// Asymmetric MSE loss for reconciliation learning
///
/// Penalizes underprediction (actual > predicted) more heavily than overprediction.
/// This is critical for cost budgeting where surprise costs are worse than budget headroom.
///
/// # Arguments
/// * `predicted` - Predicted values tensor
/// * `actual` - Actual (ground truth) values tensor
/// * `underprediction_weight` - Penalty weight for underprediction (default: 2.0)
/// * `overprediction_weight` - Penalty weight for overprediction (default: 0.5)
///
/// # Returns
/// Scalar mean asymmetric loss
///
/// # Example
/// ```ignore
/// // Actual costs were higher than predicted (bad - budget surprise)
/// let predicted = Tensor::new(&[100.0f32], &device)?;
/// let actual = Tensor::new(&[120.0f32], &device)?;
/// let loss = asymmetric_mse_loss(&predicted, &actual, 2.0, 0.5)?;
/// // Loss = (120-100)² × 2.0 = 800.0
/// ```
pub fn asymmetric_mse_loss(
    predicted: &Tensor,
    actual: &Tensor,
    underprediction_weight: f32,
    overprediction_weight: f32,
) -> Result<Tensor> {
    // Error: actual - predicted
    let errors = actual.sub(predicted)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_mse_loss sub failed: {}", e)))?;
    let squared = errors.sqr()
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_mse_loss sqr failed: {}", e)))?;
    
    // Create asymmetric weights based on error sign
    let zeros = Tensor::zeros_like(&errors)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_mse_loss zeros failed: {}", e)))?;
    let positive_mask = errors.gt(&zeros)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_mse_loss gt failed: {}", e)))?;
    
    // Create constant tensors by multiplying ones_like by the weight
    let ones = Tensor::ones_like(&errors)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_mse_loss ones failed: {}", e)))?;
    let weight_under = ones.affine(underprediction_weight as f64, 0.0)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_mse_loss weight_under failed: {}", e)))?;
    let weight_over = ones.affine(overprediction_weight as f64, 0.0)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_mse_loss weight_over failed: {}", e)))?;
    
    let weights = positive_mask.where_cond(&weight_under, &weight_over)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_mse_loss where_cond failed: {}", e)))?;
    
    // Weighted loss
    let asymmetric_loss = squared.mul(&weights)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_mse_loss mul failed: {}", e)))?;
    asymmetric_loss.mean_all()
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_mse_loss mean failed: {}", e)))
}

/// Per-element asymmetric loss (for analysis, not reduction)
///
/// Returns per-element weighted squared errors without reducing to scalar.
/// Useful for analyzing per-corridor or per-schedule variance.
pub fn asymmetric_loss_per_element(
    predicted: &Tensor,
    actual: &Tensor,
    underprediction_weight: f32,
    overprediction_weight: f32,
) -> Result<Tensor> {
    let errors = actual.sub(predicted)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_loss_per_element sub failed: {}", e)))?;
    let squared = errors.sqr()
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_loss_per_element sqr failed: {}", e)))?;
    
    let zeros = Tensor::zeros_like(&errors)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_loss_per_element zeros failed: {}", e)))?;
    let positive_mask = errors.gt(&zeros)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_loss_per_element gt failed: {}", e)))?;
    
    let ones = Tensor::ones_like(&errors)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_loss_per_element ones failed: {}", e)))?;
    let weight_under = ones.affine(underprediction_weight as f64, 0.0)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_loss_per_element weight_under failed: {}", e)))?;
    let weight_over = ones.affine(overprediction_weight as f64, 0.0)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_loss_per_element weight_over failed: {}", e)))?;
    
    let weights = positive_mask.where_cond(&weight_under, &weight_over)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_loss_per_element where_cond failed: {}", e)))?;
    
    squared.mul(&weights)
        .map_err(|e| TensorCoreError::Tensor(format!("asymmetric_loss_per_element mul failed: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn device() -> Device {
        Device::Cpu
    }

    fn tensor(vals: &[f32]) -> Tensor {
        Tensor::from_vec(vals.to_vec(), vals.len(), &device()).unwrap()
    }

    fn scalar(t: &Tensor) -> f32 {
        t.to_vec1::<f32>().unwrap()[0]
    }

    #[test]
    fn test_fuzzy_and() {
        let a = tensor(&[0.8]);
        let b = tensor(&[0.9]);
        let result = fuzzy_and(&a, &b).unwrap();
        assert!((scalar(&result) - 0.72).abs() < 0.001);

        // AND(1, 1) = 1
        let ones = tensor(&[1.0]);
        let result = fuzzy_and(&ones, &ones).unwrap();
        assert!((scalar(&result) - 1.0).abs() < 0.001);

        // AND(0, x) = 0
        let zero = tensor(&[0.0]);
        let result = fuzzy_and(&zero, &a).unwrap();
        assert!((scalar(&result) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_fuzzy_or() {
        let a = tensor(&[0.3]);
        let b = tensor(&[0.4]);
        let result = fuzzy_or(&a, &b).unwrap();
        // 0.3 + 0.4 - 0.12 = 0.58
        assert!((scalar(&result) - 0.58).abs() < 0.001);

        // OR(0, 0) = 0
        let zeros = tensor(&[0.0]);
        let result = fuzzy_or(&zeros, &zeros).unwrap();
        assert!((scalar(&result) - 0.0).abs() < 0.001);

        // OR(1, x) = 1
        let one = tensor(&[1.0]);
        let result = fuzzy_or(&one, &a).unwrap();
        assert!((scalar(&result) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_fuzzy_not() {
        let a = tensor(&[0.3]);
        let result = fuzzy_not(&a).unwrap();
        assert!((scalar(&result) - 0.7).abs() < 0.001);

        // NOT(0) = 1
        let zero = tensor(&[0.0]);
        let result = fuzzy_not(&zero).unwrap();
        assert!((scalar(&result) - 1.0).abs() < 0.001);

        // NOT(1) = 0
        let one = tensor(&[1.0]);
        let result = fuzzy_not(&one).unwrap();
        assert!((scalar(&result) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_fuzzy_implies() {
        // IMPLIES(0, x) = 1 (false implies anything)
        let zero = tensor(&[0.0]);
        let x = tensor(&[0.5]);
        let result = fuzzy_implies(&zero, &x).unwrap();
        assert!((scalar(&result) - 1.0).abs() < 0.001);

        // IMPLIES(1, 1) = 1
        let one = tensor(&[1.0]);
        let result = fuzzy_implies(&one, &one).unwrap();
        assert!((scalar(&result) - 1.0).abs() < 0.001);

        // IMPLIES(1, 0) = 0
        let result = fuzzy_implies(&one, &zero).unwrap();
        assert!((scalar(&result) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_fuzzy_or_many() {
        let a = tensor(&[0.3]);
        let b = tensor(&[0.4]);
        let c = tensor(&[0.2]);

        let result = fuzzy_or_many(&[&a, &b, &c]).unwrap();

        // 1 - (1-0.3)(1-0.4)(1-0.2) = 1 - 0.7*0.6*0.8 = 1 - 0.336 = 0.664
        assert!((scalar(&result) - 0.664).abs() < 0.001);
    }

    #[test]
    fn test_fuzzy_and_many() {
        let a = tensor(&[0.8]);
        let b = tensor(&[0.9]);
        let c = tensor(&[0.7]);

        let result = fuzzy_and_many(&[&a, &b, &c]).unwrap();

        // 0.8 * 0.9 * 0.7 = 0.504
        assert!((scalar(&result) - 0.504).abs() < 0.001);
    }

    #[test]
    fn test_soft_threshold_greater_than() {
        // "is high risk" = risk > 0.8, with sharpness 10
        let risk = tensor(&[0.9]);
        let threshold = tensor(&[0.8]);

        let result = soft_threshold(&risk, &threshold, 10.0).unwrap();
        // (0.9 - 0.8) * 10 = 1.0, sigmoid(1.0) ≈ 0.731
        assert!(scalar(&result) > 0.7);

        let low_risk = tensor(&[0.5]);
        let result = soft_threshold(&low_risk, &threshold, 10.0).unwrap();
        // (0.5 - 0.8) * 10 = -3.0, sigmoid(-3.0) ≈ 0.047
        assert!(scalar(&result) < 0.1);
    }

    #[test]
    fn test_soft_threshold_less_than() {
        // "is low confidence" = confidence < 0.5, use negative sharpness
        let confidence = tensor(&[0.3]);
        let threshold = tensor(&[0.5]);

        let result = soft_threshold(&confidence, &threshold, -10.0).unwrap();
        // (0.3 - 0.5) * -10 = 2.0, sigmoid(2.0) ≈ 0.88
        assert!(scalar(&result) > 0.8);

        let high_conf = tensor(&[0.9]);
        let result = soft_threshold(&high_conf, &threshold, -10.0).unwrap();
        // (0.9 - 0.5) * -10 = -4.0, sigmoid(-4.0) ≈ 0.018
        assert!(scalar(&result) < 0.1);
    }

    #[test]
    fn test_de_morgan() {
        // Verify De Morgan's laws hold in fuzzy logic
        let a = tensor(&[0.6]);
        let b = tensor(&[0.4]);

        // ¬(a ∧ b) = ¬a ∨ ¬b
        let lhs = fuzzy_not(&fuzzy_and(&a, &b).unwrap()).unwrap();
        let rhs = fuzzy_or(&fuzzy_not(&a).unwrap(), &fuzzy_not(&b).unwrap()).unwrap();
        assert!((scalar(&lhs) - scalar(&rhs)).abs() < 0.001);

        // ¬(a ∨ b) = ¬a ∧ ¬b
        let lhs = fuzzy_not(&fuzzy_or(&a, &b).unwrap()).unwrap();
        let rhs = fuzzy_and(&fuzzy_not(&a).unwrap(), &fuzzy_not(&b).unwrap()).unwrap();
        assert!((scalar(&lhs) - scalar(&rhs)).abs() < 0.001);
    }

    // =========================================================================
    // Dtype preservation regression tests (Issue: F32/F64 mismatch)
    // =========================================================================

    #[test]
    fn test_dtype_preservation_fuzzy_not() {
        // Regression test: fuzzy_not should preserve F32 dtype
        let a = tensor(&[0.6]);
        assert_eq!(a.dtype(), DType::F32);
        
        let result = fuzzy_not(&a).unwrap();
        assert_eq!(result.dtype(), DType::F32, "fuzzy_not should preserve F32 dtype");
    }

    #[test]
    fn test_dtype_preservation_fuzzy_or_many() {
        // Regression test: fuzzy_or_many should preserve F32 dtype
        let a = tensor(&[0.3]);
        let b = tensor(&[0.4]);
        let c = tensor(&[0.2]);
        
        let result = fuzzy_or_many(&[&a, &b, &c]).unwrap();
        assert_eq!(result.dtype(), DType::F32, "fuzzy_or_many should preserve F32 dtype");
    }

    #[test]
    fn test_dtype_preservation_soft_threshold() {
        // Regression test: soft_threshold should preserve F32 dtype
        let x = tensor(&[0.9]);
        let threshold = tensor(&[0.5]);
        
        let result = soft_threshold(&x, &threshold, 10.0).unwrap();
        assert_eq!(result.dtype(), DType::F32, "soft_threshold should preserve F32 dtype");
    }

    #[test]
    fn test_dtype_preservation_soft_threshold_scalar() {
        // Regression test: soft_threshold_scalar should preserve F32 dtype
        let x = tensor(&[0.9]);
        
        let result = soft_threshold_scalar(&x, 0.5, 10.0).unwrap();
        assert_eq!(result.dtype(), DType::F32, "soft_threshold_scalar should preserve F32 dtype");
    }

    #[test]
    fn test_dtype_chain_no_promotion() {
        // Regression test: chained operations should not promote to F64
        let a = tensor(&[0.8]);
        let b = tensor(&[0.6]);
        let threshold = tensor(&[0.5]);
        
        // Chain multiple operations
        let and_result = fuzzy_and(&a, &b).unwrap();
        let not_result = fuzzy_not(&and_result).unwrap();
        let thresh_result = soft_threshold(&not_result, &threshold, 5.0).unwrap();
        
        assert_eq!(thresh_result.dtype(), DType::F32, "Chained ops should preserve F32");
    }

    // =========================================================================
    // Asymmetric Loss Tests
    // =========================================================================

    #[test]
    fn test_asymmetric_mse_loss_underprediction() {
        // Underprediction: actual > predicted (2× penalty)
        let predicted = tensor(&[100.0]);
        let actual = tensor(&[120.0]);
        
        let loss = asymmetric_mse_loss(&predicted, &actual, 2.0, 0.5).unwrap();
        // mean_all returns a scalar (rank 0)
        let result = loss.to_scalar::<f32>().unwrap();
        
        // error = 20, squared = 400, weighted = 400 × 2.0 = 800
        assert!((result - 800.0).abs() < 0.1, "Expected 800, got {}", result);
    }

    #[test]
    fn test_asymmetric_mse_loss_overprediction() {
        // Overprediction: actual < predicted (0.5× penalty)
        let predicted = tensor(&[120.0]);
        let actual = tensor(&[100.0]);
        
        let loss = asymmetric_mse_loss(&predicted, &actual, 2.0, 0.5).unwrap();
        let result = loss.to_scalar::<f32>().unwrap();
        
        // error = -20, squared = 400, weighted = 400 × 0.5 = 200
        assert!((result - 200.0).abs() < 0.1, "Expected 200, got {}", result);
    }

    #[test]
    fn test_asymmetric_mse_loss_mixed() {
        // Mixed: some over, some under
        let predicted = tensor(&[100.0, 100.0]);
        let actual = tensor(&[120.0, 80.0]);  // +20 under, -20 over
        
        let loss = asymmetric_mse_loss(&predicted, &actual, 2.0, 0.5).unwrap();
        let result = loss.to_scalar::<f32>().unwrap();
        
        // under: 400 × 2.0 = 800
        // over: 400 × 0.5 = 200
        // mean = (800 + 200) / 2 = 500
        assert!((result - 500.0).abs() < 0.1, "Expected 500, got {}", result);
    }

    #[test]
    fn test_asymmetric_loss_per_element() {
        let predicted = tensor(&[100.0, 100.0]);
        let actual = tensor(&[120.0, 80.0]);
        
        let losses = asymmetric_loss_per_element(&predicted, &actual, 2.0, 0.5).unwrap();
        let result = losses.to_vec1::<f32>().unwrap();
        
        // First: underprediction → 400 × 2.0 = 800
        assert!((result[0] - 800.0).abs() < 0.1, "Expected 800, got {}", result[0]);
        
        // Second: overprediction → 400 × 0.5 = 200
        assert!((result[1] - 200.0).abs() < 0.1, "Expected 200, got {}", result[1]);
    }
}




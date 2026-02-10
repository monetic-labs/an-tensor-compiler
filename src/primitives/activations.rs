//! Activation Functions
//!
//! Standard neural network activation functions used in Tensor Logic.

use candle_core::Tensor;
use crate::{Result, TensorCoreError};

/// Sigmoid activation: σ(x) = 1 / (1 + e^(-x))
///
/// Maps any real number to (0, 1). Used for:
/// - Converting logits to probabilities
/// - Soft thresholding in fuzzy logic
pub fn sigmoid(tensor: &Tensor) -> Result<Tensor> {
    let neg = tensor.neg()
        .map_err(|e| TensorCoreError::Tensor(format!("sigmoid neg failed: {}", e)))?;
    let exp_neg = neg.exp()
        .map_err(|e| TensorCoreError::Tensor(format!("sigmoid exp failed: {}", e)))?;
    let one_plus = (exp_neg + 1.0)
        .map_err(|e| TensorCoreError::Tensor(format!("sigmoid add failed: {}", e)))?;
    one_plus.recip()
        .map_err(|e| TensorCoreError::Tensor(format!("sigmoid recip failed: {}", e)))
}

/// Softmax activation along a dimension
///
/// Converts a vector of real numbers to a probability distribution.
pub fn softmax(tensor: &Tensor, dim: usize) -> Result<Tensor> {
    candle_nn::ops::softmax(tensor, dim)
        .map_err(|e| TensorCoreError::Tensor(format!("Softmax failed: {}", e)))
}

/// ReLU activation: max(0, x)
///
/// Rectified Linear Unit - the most common activation in deep learning.
pub fn relu(tensor: &Tensor) -> Result<Tensor> {
    tensor.relu()
        .map_err(|e| TensorCoreError::Tensor(format!("ReLU failed: {}", e)))
}

/// Leaky ReLU activation: max(αx, x) where α is typically 0.01
pub fn leaky_relu(tensor: &Tensor, negative_slope: f64) -> Result<Tensor> {
    // leaky_relu(x) = max(αx, x) = x if x > 0 else αx
    let zeros = Tensor::zeros_like(tensor)
        .map_err(|e| TensorCoreError::Tensor(format!("zeros_like failed: {}", e)))?;

    let positive = tensor.maximum(&zeros)
        .map_err(|e| TensorCoreError::Tensor(format!("maximum failed: {}", e)))?;

    let negative = tensor.minimum(&zeros)
        .map_err(|e| TensorCoreError::Tensor(format!("minimum failed: {}", e)))?;

    let scaled_negative = (negative * negative_slope)
        .map_err(|e| TensorCoreError::Tensor(format!("scale failed: {}", e)))?;

    (positive + scaled_negative)
        .map_err(|e| TensorCoreError::Tensor(format!("leaky_relu add failed: {}", e)))
}

/// Tanh activation: (e^x - e^(-x)) / (e^x + e^(-x))
///
/// Maps any real number to (-1, 1).
pub fn tanh(tensor: &Tensor) -> Result<Tensor> {
    tensor.tanh()
        .map_err(|e| TensorCoreError::Tensor(format!("tanh failed: {}", e)))
}

/// GELU activation: x * Φ(x) where Φ is the CDF of standard normal
///
/// Gaussian Error Linear Unit - used in transformers.
/// Approximation: x * σ(1.702 * x)
pub fn gelu(tensor: &Tensor) -> Result<Tensor> {
    // GELU approximation: x * sigmoid(1.702 * x)
    let scaled = (tensor * 1.702)
        .map_err(|e| TensorCoreError::Tensor(format!("GELU scale failed: {}", e)))?;
    let sig = sigmoid(&scaled)?;
    tensor.mul(&sig)
        .map_err(|e| TensorCoreError::Tensor(format!("GELU mul failed: {}", e)))
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

    #[test]
    fn test_sigmoid() {
        let t = tensor(&[0.0]);
        let s = sigmoid(&t).unwrap();
        let val = s.to_vec1::<f32>().unwrap()[0];

        // sigmoid(0) = 0.5
        assert!((val - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_sigmoid_extremes() {
        let t_large = tensor(&[100.0]);
        let s_large = sigmoid(&t_large).unwrap();
        let val_large = s_large.to_vec1::<f32>().unwrap()[0];
        assert!((val_large - 1.0).abs() < 0.001);

        let t_small = tensor(&[-100.0]);
        let s_small = sigmoid(&t_small).unwrap();
        let val_small = s_small.to_vec1::<f32>().unwrap()[0];
        assert!(val_small.abs() < 0.001);
    }

    #[test]
    fn test_relu() {
        let t = tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = relu(&t).unwrap();
        let vals = r.to_vec1::<f32>().unwrap();

        assert_eq!(vals, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_leaky_relu() {
        let t = tensor(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = leaky_relu(&t, 0.1).unwrap();
        let vals = r.to_vec1::<f32>().unwrap();

        // Negative values scaled by 0.1
        assert!((vals[0] - (-0.2)).abs() < 0.001);
        assert!((vals[1] - (-0.1)).abs() < 0.001);
        assert!(vals[2].abs() < 0.001);
        assert!((vals[3] - 1.0).abs() < 0.001);
        assert!((vals[4] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_tanh() {
        let t = tensor(&[0.0]);
        let result = tanh(&t).unwrap();
        let val = result.to_vec1::<f32>().unwrap()[0];

        // tanh(0) = 0
        assert!(val.abs() < 0.001);
    }

    #[test]
    fn test_softmax() {
        let t = tensor(&[1.0, 2.0, 3.0]);
        let t = t.unsqueeze(0).unwrap(); // [1, 3]
        let s = softmax(&t, 1).unwrap();
        let vals = s.squeeze(0).unwrap().to_vec1::<f32>().unwrap();

        // Softmax should sum to 1
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // Larger input should have larger probability
        assert!(vals[2] > vals[1]);
        assert!(vals[1] > vals[0]);
    }
}


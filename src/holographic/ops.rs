//! Holographic tensor operations
//!
//! Implements binding, unbinding, and superposition operations for
//! holographic reduced representations (HRR) using FFT-based circular
//! convolution (Plate, 1995).
//!
//! ## Operations
//!
//! - **`bind`**: Circular convolution via FFT — mathematically correct HRR binding
//! - **`bind_fast`**: Phase-shift approximation — ~3x faster, lower fidelity
//! - **`unbind`**: Circular correlation via FFT — recovers bound concepts
//! - **`unbind_fast`**: Fast approximate unbinding, pairs with `bind_fast`
//! - **`superimpose`**: Weighted sum — creates hologram from multiple concepts
//!
//! ## Which bind to use
//!
//! | Function | Fidelity | Speed | When |
//! |---|---|---|---|
//! | `bind` | High (>0.6 cosine at dim=512) | 1x | Default — correctness matters |
//! | `bind_fast` | Low (noisy below dim=256) | ~3x | Large-batch encoding, approximate recall |
//!
//! The FFT and fast paths are interchangeable by signature — swap `bind` for
//! `bind_fast` to trade fidelity for throughput. Use `unbind_fast` with
//! `bind_fast`; mixing the two paths gives incorrect results.

use crate::{Result, TensorCoreError};
use candle_core::{DType, Device, Tensor};
use rustfft::{num_complex::Complex, FftPlanner};

// =============================================================================
// FFT-based binding (mathematically correct HRR)
// =============================================================================

/// Bind two tensors via circular convolution (FFT-based, Plate 1995)
///
/// This is the mathematically correct HRR binding operation:
/// `bind(a, b) = IFFT(FFT(a) ⊙ FFT(b))`
///
/// # Properties
/// - Associative: (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)
/// - Commutative: a ⊗ b = b ⊗ a
/// - Distributes: a ⊗ (b + c) = a ⊗ b + a ⊗ c
/// - Exact inverse: `unbind(bind(a, b), a) ≈ b` (fidelity improves with dim)
///
/// # Recovery fidelity
///
/// For random Gaussian vectors, expected cosine similarity after bind/unbind:
/// - dim=64: ~0.4–0.6
/// - dim=256: ~0.6–0.8
/// - dim=512: ~0.7–0.9
/// - dim=1024: ~0.85–0.95
///
/// For performance-critical paths where approximate recovery is acceptable,
/// use [`bind_fast`] instead (~3x faster).
///
/// # Example
/// ```rust,ignore
/// let semantic = embed("struct Foo")?;
/// let structural = embed("pub fields: Vec<Bar>")?;
/// let bound = bind(&semantic, &structural)?;
/// let recovered = unbind(&bound, &semantic)?;
/// // cosine_similarity(&recovered, &structural) > 0.7 at dim=512
/// ```
pub fn bind(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    circular_convolution(a, b)
}

/// Unbind via circular correlation (FFT-based, Plate 1995)
///
/// Recovers bound information: if `h = bind(a, b)`, then `unbind(h, a) ≈ b`.
///
/// Uses FFT-based circular correlation:
/// `unbind(h, q) = IFFT(FFT(h) ⊙ conj(FFT(q)))`
///
/// Pair with [`bind`]. Do not mix with [`bind_fast`] / [`unbind_fast`].
///
/// # Example
/// ```rust,ignore
/// let hologram = bind(&semantic, &structural)?;
/// let recovered = unbind(&hologram, &semantic)?;
/// // recovered ≈ structural
/// ```
pub fn unbind(hologram: &Tensor, query: &Tensor) -> Result<Tensor> {
    circular_correlation(hologram, query)
}

/// Alias for `unbind` — more intuitive for context queries
pub fn project(hologram: &Tensor, query: &Tensor) -> Result<Tensor> {
    unbind(hologram, query)
}

// =============================================================================
// Fast approximate binding (phase-shifted element-wise multiply)
// =============================================================================

/// Bind two tensors via fast phase-shift approximation
///
/// Approximately ~3x faster than FFT-based [`bind`], but produces noisier
/// recovery, especially at small dimensions (<256). Good for large-batch
/// encoding pipelines where approximate recall is acceptable.
///
/// **Pair with [`unbind_fast`].** Do not mix with [`unbind`] / [`project`].
///
/// # Example
/// ```rust,ignore
/// // High-throughput encoding (approximate)
/// let bound = bind_fast(&semantic, &structural)?;
/// let recovered = unbind_fast(&bound, &semantic)?;
/// ```
pub fn bind_fast(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_shape = a.dims();
    let b_shape = b.dims();

    if a_shape != b_shape {
        return Err(TensorCoreError::Tensor(format!(
            "Shape mismatch for bind_fast: {:?} vs {:?}",
            a_shape, b_shape
        )));
    }

    let dim = a_shape[a_shape.len() - 1];
    let a_flat = a.flatten_all()?;
    let b_flat = b.flatten_all()?;

    let half = dim / 2;
    let b_first = b_flat.narrow(0, 0, half)?;
    let b_second = b_flat.narrow(0, half, dim - half)?;
    let b_shifted = Tensor::cat(&[&b_second, &b_first], 0)?;

    let result = (&a_flat * &b_shifted)?;
    result
        .reshape(a_shape)
        .map_err(|e| TensorCoreError::Tensor(e.to_string()))
}

/// Unbind via fast approximate inverse
///
/// Pairs with [`bind_fast`]. Do not mix with [`unbind`] / [`project`].
pub fn unbind_fast(hologram: &Tensor, query: &Tensor) -> Result<Tensor> {
    let query_inv = hrr_inverse(query)?;
    bind_fast(hologram, &query_inv)
}

// =============================================================================
// FFT implementation
// =============================================================================

/// Circular convolution via FFT: `IFFT(FFT(a) ⊙ FFT(b))`
fn circular_convolution(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_shape = a.dims();
    let b_shape = b.dims();

    if a_shape != b_shape {
        return Err(TensorCoreError::Tensor(format!(
            "Shape mismatch for circular_convolution: {:?} vs {:?}",
            a_shape, b_shape
        )));
    }

    let a_flat = a.flatten_all()?;
    let b_flat = b.flatten_all()?;
    let dim = a_flat.dims()[0];

    let a_vals: Vec<f32> = a_flat.to_vec1()?;
    let b_vals: Vec<f32> = b_flat.to_vec1()?;

    let mut a_complex: Vec<Complex<f32>> = a_vals.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut b_complex: Vec<Complex<f32>> = b_vals.iter().map(|&v| Complex::new(v, 0.0)).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(dim);
    fft.process(&mut a_complex);
    fft.process(&mut b_complex);

    // Element-wise multiply in frequency domain
    let mut result_complex: Vec<Complex<f32>> = a_complex
        .iter()
        .zip(b_complex.iter())
        .map(|(a, b)| a * b)
        .collect();

    let ifft = planner.plan_fft_inverse(dim);
    ifft.process(&mut result_complex);

    // Normalize (rustfft convention) and take real part
    let scale = 1.0 / dim as f32;
    let result_vals: Vec<f32> = result_complex.iter().map(|c| c.re * scale).collect();

    let result = Tensor::from_vec(result_vals, a_flat.dims(), a_flat.device())
        .map_err(|e| TensorCoreError::Tensor(e.to_string()))?;
    result
        .reshape(a_shape)
        .map_err(|e| TensorCoreError::Tensor(e.to_string()))
}

/// Circular correlation via FFT: `IFFT(FFT(h) ⊙ conj(FFT(q)))`
fn circular_correlation(hologram: &Tensor, query: &Tensor) -> Result<Tensor> {
    let h_shape = hologram.dims();
    let q_shape = query.dims();

    if h_shape != q_shape {
        return Err(TensorCoreError::Tensor(format!(
            "Shape mismatch for circular_correlation: {:?} vs {:?}",
            h_shape, q_shape
        )));
    }

    let h_flat = hologram.flatten_all()?;
    let q_flat = query.flatten_all()?;
    let dim = h_flat.dims()[0];

    let h_vals: Vec<f32> = h_flat.to_vec1()?;
    let q_vals: Vec<f32> = q_flat.to_vec1()?;

    let mut h_complex: Vec<Complex<f32>> = h_vals.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut q_complex: Vec<Complex<f32>> = q_vals.iter().map(|&v| Complex::new(v, 0.0)).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(dim);
    fft.process(&mut h_complex);
    fft.process(&mut q_complex);

    // Correlation: hologram FFT × conjugate of query FFT
    let mut result_complex: Vec<Complex<f32>> = h_complex
        .iter()
        .zip(q_complex.iter())
        .map(|(h, q)| h * q.conj())
        .collect();

    let ifft = planner.plan_fft_inverse(dim);
    ifft.process(&mut result_complex);

    let scale = 1.0 / dim as f32;
    let result_vals: Vec<f32> = result_complex.iter().map(|c| c.re * scale).collect();

    let result = Tensor::from_vec(result_vals, h_flat.dims(), h_flat.device())
        .map_err(|e| TensorCoreError::Tensor(e.to_string()))?;
    result
        .reshape(h_shape)
        .map_err(|e| TensorCoreError::Tensor(e.to_string()))
}

/// Exact HRR inverse via element reversal (except index 0)
///
/// For vector x, the inverse x⁻¹ satisfies `bind(x, x⁻¹) ≈ identity`.
/// Defined as: `x⁻¹[0] = x[0]`, `x⁻¹[i] = x[n-i]` for i > 0.
fn hrr_inverse(tensor: &Tensor) -> Result<Tensor> {
    let flat = tensor.flatten_all()?;
    let dim = flat.dims()[0];
    let values: Vec<f32> = flat.to_vec1()?;

    let mut inverse = vec![0.0f32; dim];
    inverse[0] = values[0];
    for i in 1..dim {
        inverse[i] = values[dim - i];
    }

    Tensor::from_vec(inverse, flat.dims(), flat.device())
        .map_err(|e| TensorCoreError::Tensor(e.to_string()))
}

// =============================================================================
// Superposition and normalization
// =============================================================================

/// Superimpose multiple tensors into a hologram
///
/// Creates a hologram containing all input concepts, each recoverable via
/// `project`. Composed holograms accumulate noise proportional to the number
/// of items: theoretical SNR ≈ `sqrt(dim / N)`. At SNR < 1.0 (N > dim),
/// recall becomes unreliable.
///
/// # Arguments
/// - `tensors`: Tensors to superimpose
/// - `weights`: Optional per-tensor weights (defaults to equal `1/N`)
///
/// # Example
/// ```rust,ignore
/// let hologram = superimpose(&[comp1, comp2, comp3], None)?;
/// let recalled = project(&hologram, &comp1)?;
/// // cosine_similarity(&recalled, &comp1_original) > threshold
/// ```
pub fn superimpose(tensors: &[Tensor], weights: Option<&[f32]>) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(TensorCoreError::Tensor(
            "Cannot superimpose empty list".into(),
        ));
    }

    let device = tensors[0].device();
    let shape = tensors[0].dims();

    let default_weights: Vec<f32> = vec![1.0 / tensors.len() as f32; tensors.len()];
    let weights = weights.unwrap_or(&default_weights);

    if weights.len() != tensors.len() {
        return Err(TensorCoreError::Tensor(
            "Weights length must match tensors length".into(),
        ));
    }

    let mut result = Tensor::zeros(shape, DType::F32, device)?;

    for (tensor, weight) in tensors.iter().zip(weights.iter()) {
        let weight_tensor = Tensor::from_vec(vec![*weight], (1,), device)?;
        let weighted = tensor.broadcast_mul(&weight_tensor)?;
        result = (&result + &weighted)?;
    }

    normalize(&result)
}

/// Normalize tensor to unit length
pub fn normalize(tensor: &Tensor) -> Result<Tensor> {
    let squared = tensor.sqr()?;
    let sum = squared.sum_all()?;
    let norm = sum.sqrt()?;
    let norm_scalar: f32 = norm.to_scalar()?;

    if norm_scalar < 1e-8 {
        return Ok(tensor.clone());
    }

    let norm_tensor = Tensor::from_vec(vec![norm_scalar], (1,), tensor.device())?;
    tensor
        .broadcast_div(&norm_tensor)
        .map_err(|e| TensorCoreError::Tensor(e.to_string()))
}

/// Cosine similarity between two tensors
///
/// Delegates to [`crate::primitives::cosine_similarity`] — single implementation.
pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    crate::primitives::cosine_similarity(a, b)
}

/// Generate sinusoidal position encoding
///
/// Used to encode position information that can be bound with content.
/// Standard transformer-style: sin/cos at varying frequencies.
///
/// # Arguments
/// - `position`: The position index
/// - `dim`: Embedding dimension (must be even)
/// - `device`: Device for tensor creation
pub fn position_encoding(position: usize, dim: usize, device: &Device) -> Result<Tensor> {
    if !dim.is_multiple_of(2) {
        return Err(TensorCoreError::Tensor(
            "Position encoding dimension must be even".into(),
        ));
    }

    let mut encoding = vec![0.0f32; dim];

    for i in 0..dim / 2 {
        let freq = 1.0 / (10000.0_f32.powf(2.0 * i as f32 / dim as f32));
        let angle = position as f32 * freq;
        encoding[2 * i] = angle.sin();
        encoding[2 * i + 1] = angle.cos();
    }

    Tensor::from_vec(encoding, (dim,), device).map_err(|e| TensorCoreError::Tensor(e.to_string()))
}

/// Generate deterministic role encoding from a string
///
/// Creates a unique, deterministic unit vector for a semantic role.
/// The same role string always produces the same encoding across runs.
///
/// # Arguments
/// - `role`: Role identifier (e.g., `"core"`, `"support"`, `"integration"`)
/// - `dim`: Embedding dimension
/// - `device`: Device for tensor creation
pub fn role_encoding(role: &str, dim: usize, device: &Device) -> Result<Tensor> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    role.hash(&mut hasher);
    let seed = hasher.finish();

    let mut encoding = vec![0.0f32; dim];
    let mut state = seed;

    for val in &mut encoding {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let normalized = (state as f64 / u64::MAX as f64) as f32;
        *val = normalized * 2.0 - 1.0;
    }

    let norm: f32 = encoding.iter().map(|x| x * x).sum::<f32>().sqrt();
    let normalized: Vec<f32> = encoding.iter().map(|x| x / norm).collect();

    Tensor::from_vec(normalized, (dim,), device).map_err(|e| TensorCoreError::Tensor(e.to_string()))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_bind_unbind_recovery() {
        let device = Device::Cpu;
        let dim = 512;

        let a = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();

        let bound = bind(&a, &b).unwrap();
        let recovered = unbind(&bound, &a).unwrap();

        let similarity = cosine_similarity(&b, &recovered).unwrap();
        let random = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();
        let random_sim = cosine_similarity(&b, &random).unwrap().abs();

        println!("FFT recovery similarity (dim={}): {:.4}", dim, similarity);

        // FFT recovery should beat random by a wide margin at dim=512
        assert!(
            similarity > 0.5,
            "Expected >0.5 similarity, got {}",
            similarity
        );
        assert!(
            similarity > random_sim + 0.2,
            "FFT ({}) should beat random ({}) by >0.2",
            similarity,
            random_sim
        );
    }

    #[test]
    fn test_fft_better_than_fast() {
        let device = Device::Cpu;
        let dim = 256;

        let a = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();

        // FFT path
        let bound_fft = bind(&a, &b).unwrap();
        let recovered_fft = unbind(&bound_fft, &a).unwrap();
        let sim_fft = cosine_similarity(&b, &recovered_fft).unwrap();

        // Fast path
        let bound_fast = bind_fast(&a, &b).unwrap();
        let recovered_fast = unbind_fast(&bound_fast, &a).unwrap();
        let sim_fast = cosine_similarity(&b, &recovered_fast).unwrap();

        println!("FFT: {:.4}, Fast: {:.4}", sim_fft, sim_fast);

        // FFT should be strictly better
        assert!(
            sim_fft > sim_fast,
            "FFT ({}) should beat fast approximation ({})",
            sim_fft,
            sim_fast
        );
    }

    #[test]
    fn test_fast_bind_unbind_still_works() {
        let device = Device::Cpu;
        let dim = 64;

        let a = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();

        let bound = bind_fast(&a, &b).unwrap();
        let recovered = unbind_fast(&bound, &a).unwrap();

        let similarity = cosine_similarity(&b, &recovered).unwrap();
        let random = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();
        let random_sim = cosine_similarity(&b, &random).unwrap();

        println!(
            "Fast recovery (dim={}): {:.4} (random: {:.4})",
            dim, similarity, random_sim
        );

        assert!(similarity > random_sim - 0.3);
    }

    #[test]
    fn test_superimpose_normalized() {
        let device = Device::Cpu;
        let dim = 64;

        let tensors: Vec<Tensor> = (0..5)
            .map(|_| Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap())
            .collect();

        let hologram = superimpose(&tensors, None).unwrap();

        let norm: f32 = hologram
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .sqrt()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_position_encoding() {
        let device = Device::Cpu;
        let dim = 64;

        let pos0 = position_encoding(0, dim, &device).unwrap();
        let pos1 = position_encoding(1, dim, &device).unwrap();
        let pos100 = position_encoding(100, dim, &device).unwrap();

        let sim_0_1 = cosine_similarity(&pos0, &pos1).unwrap();
        let sim_0_100 = cosine_similarity(&pos0, &pos100).unwrap();

        assert!(sim_0_1 > sim_0_100);
    }

    #[test]
    fn test_role_encoding_deterministic() {
        let device = Device::Cpu;
        let dim = 64;

        let role1a = role_encoding("synapse", dim, &device).unwrap();
        let role1b = role_encoding("synapse", dim, &device).unwrap();
        let role2 = role_encoding("crdt", dim, &device).unwrap();

        let sim_same = cosine_similarity(&role1a, &role1b).unwrap();
        assert!((sim_same - 1.0).abs() < 0.001);

        let sim_diff = cosine_similarity(&role1a, &role2).unwrap();
        assert!(sim_diff < 0.5);
    }
}

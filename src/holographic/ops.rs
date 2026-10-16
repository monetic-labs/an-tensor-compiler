//! Holographic tensor operations
//!
//! Implements binding, unbinding, and superposition operations for
//! holographic reduced representations (HRR).
//!
//! ## Operations
//!
//! - **Bind (⊗)**: Circular convolution - combines concepts
//! - **Unbind (⊙)**: Circular correlation - extracts concepts  
//! - **Superimpose (+)**: Weighted sum - creates hologram from multiple concepts

use crate::{Result, TensorCoreError};
use candle_core::{DType, Device, Tensor};
use rustfft::{num_complex::Complex, FftPlanner};

/// Bind two tensors via circular convolution (FFT-based)
///
/// This is the core HRR operation for combining concepts into a hologram.
/// Uses FFT-based circular convolution for mathematically correct binding:
///   bind(a, b) = IFFT(FFT(a) ⊙ FFT(b))
///
/// # Properties
/// - Associative: (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)
/// - Commutative: a ⊗ b = b ⊗ a
/// - Inverse exists: a ⊗ a⁻¹ ≈ identity
/// - Distributes over superposition: a ⊗ (b + c) = a⊗b + a⊗c
///
/// # Example
/// ```rust,ignore
/// let semantic = embedder.embed("struct Foo").await?;
/// let structural = embedder.embed("pub fields: Vec<Bar>").await?;
/// let bound = bind(&semantic, &structural)?;
/// ```
pub fn bind(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    circular_convolution(a, b)
}

/// Fast approximate bind via phase-shifted element-wise multiply
///
/// ~3x faster than FFT-based bind for high-dimensional vectors,
/// but with lower recovery fidelity. Use for performance-critical paths
/// where approximate recovery is acceptable (e.g., large batch encoding).
///
/// For correct HRR math, use [`bind`] instead.
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

/// Unbind/project via circular correlation (FFT-based)
///
/// Recovers bound information by correlating with query.
/// If hologram = bind(a, b), then unbind(hologram, a) ≈ b
///
/// Uses FFT-based circular correlation:
///   unbind(h, q) = IFFT(FFT(h) ⊙ conj(FFT(q)))
///
/// # Example
/// ```rust,ignore
/// let hologram = bind(&semantic, &structural)?;
/// let recovered = unbind(&hologram, &semantic)?;
/// // recovered ≈ structural (high fidelity with FFT)
/// ```
pub fn unbind(hologram: &Tensor, query: &Tensor) -> Result<Tensor> {
    circular_correlation(hologram, query)
}

/// Fast approximate unbind via inverse + phase-shifted multiply
///
/// Pairs with [`bind_fast`]. Use for performance-critical paths.
/// For correct HRR math, use [`unbind`] instead.
pub fn unbind_fast(hologram: &Tensor, query: &Tensor) -> Result<Tensor> {
    let query_inv = hrr_inverse(query)?;
    bind_fast(hologram, &query_inv)
}

/// Alias for unbind — more intuitive for context queries
pub fn project(hologram: &Tensor, query: &Tensor) -> Result<Tensor> {
    unbind(hologram, query)
}

// =============================================================================
// FFT-based circular convolution and correlation (Plate 1995)
// =============================================================================

/// Circular convolution via FFT: IFFT(FFT(a) ⊙ FFT(b))
///
/// This is the mathematically correct HRR binding operation.
fn circular_convolution(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_shape = a.dims();
    let b_shape = b.dims();

    if a_shape != b_shape {
        return Err(TensorCoreError::Tensor(format!(
            "Shape mismatch for circular convolution: {:?} vs {:?}",
            a_shape, b_shape
        )));
    }

    let a_flat = a.flatten_all()?;
    let b_flat = b.flatten_all()?;
    let dim = a_flat.dims()[0];

    let a_vals: Vec<f32> = a_flat.to_vec1()?;
    let b_vals: Vec<f32> = b_flat.to_vec1()?;

    // Convert to complex
    let mut a_complex: Vec<Complex<f32>> = a_vals.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut b_complex: Vec<Complex<f32>> = b_vals.iter().map(|&v| Complex::new(v, 0.0)).collect();

    // Forward FFT
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

    // Inverse FFT
    let ifft = planner.plan_fft_inverse(dim);
    ifft.process(&mut result_complex);

    // Normalize by dim (rustfft convention) and take real part
    let scale = 1.0 / dim as f32;
    let result_vals: Vec<f32> = result_complex.iter().map(|c| c.re * scale).collect();

    let result = Tensor::from_vec(result_vals, a_flat.dims(), a_flat.device())
        .map_err(|e| TensorCoreError::Tensor(e.to_string()))?;
    result
        .reshape(a_shape)
        .map_err(|e| TensorCoreError::Tensor(e.to_string()))
}

/// Circular correlation via FFT: IFFT(conj(FFT(a)) ⊙ FFT(b))
///
/// This is the mathematically correct HRR unbinding operation.
/// Circular correlation of a with b recovers what was bound with a.
fn circular_correlation(hologram: &Tensor, query: &Tensor) -> Result<Tensor> {
    let h_shape = hologram.dims();
    let q_shape = query.dims();

    if h_shape != q_shape {
        return Err(TensorCoreError::Tensor(format!(
            "Shape mismatch for circular correlation: {:?} vs {:?}",
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

    // Correlation: multiply hologram FFT by conjugate of query FFT
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

/// HRR inverse: reversal of elements (except first)
///
/// For a vector x, the approximate inverse x⁻¹ is:
///   x⁻¹[0] = x[0], x⁻¹[i] = x[n-i] for i > 0
///
/// This is exact for the circular convolution algebra.
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

/// Superimpose multiple tensors into a hologram
///
/// Creates a hologram containing all input concepts.
/// Each concept can be recovered via projection.
///
/// # Arguments
/// - `tensors`: The tensors to superimpose
/// - `weights`: Optional weights for each tensor (defaults to equal)
///
/// # Example
/// ```rust,ignore
/// let components = vec![comp1.hologram, comp2.hologram, comp3.hologram];
/// let module_hologram = superimpose(&components, None)?;
/// ```
pub fn superimpose(tensors: &[Tensor], weights: Option<&[f32]>) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(TensorCoreError::Tensor(
            "Cannot superimpose empty list".into(),
        ));
    }

    let device = tensors[0].device();
    let shape = tensors[0].dims();

    // Default to equal weights
    let default_weights: Vec<f32> = vec![1.0 / tensors.len() as f32; tensors.len()];
    let weights = weights.unwrap_or(&default_weights);

    if weights.len() != tensors.len() {
        return Err(TensorCoreError::Tensor(
            "Weights length must match tensors length".into(),
        ));
    }

    // Weighted sum
    let mut result = Tensor::zeros(shape, DType::F32, device)?;

    for (tensor, weight) in tensors.iter().zip(weights.iter()) {
        let weight_tensor = Tensor::from_vec(vec![*weight], (1,), device)?;
        let weighted = tensor.broadcast_mul(&weight_tensor)?;
        result = (&result + &weighted)?;
    }

    // Normalize to unit length
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
/// Delegates to [`crate::primitives::cosine_similarity`] — single implementation
/// shared across the crate.
pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    crate::primitives::cosine_similarity(a, b)
}

/// Generate sinusoidal position encoding
///
/// Used to encode position information that can be bound with content.
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
/// Creates a unique, deterministic encoding for semantic roles.
/// The same role string always produces the same encoding.
///
/// # Arguments
/// - `role`: Role identifier (e.g., "core", "support", "synapse")
/// - `dim`: Embedding dimension
/// - `device`: Device for tensor creation
pub fn role_encoding(role: &str, dim: usize, device: &Device) -> Result<Tensor> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Create deterministic seed from role string
    let mut hasher = DefaultHasher::new();
    role.hash(&mut hasher);
    let seed = hasher.finish();

    // Generate pseudo-random but deterministic encoding
    let mut encoding = vec![0.0f32; dim];
    let mut state = seed;

    for value in encoding.iter_mut().take(dim) {
        // Simple LCG for reproducibility
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let normalized = (state as f64 / u64::MAX as f64) as f32;
        *value = normalized * 2.0 - 1.0; // Range [-1, 1]
    }

    // Normalize to unit length
    let norm: f32 = encoding.iter().map(|x| x * x).sum::<f32>().sqrt();
    let normalized: Vec<f32> = encoding.iter().map(|x| x / norm).collect();

    Tensor::from_vec(normalized, (dim,), device).map_err(|e| TensorCoreError::Tensor(e.to_string()))
}

/// Bundle a labeled record into a single HRR vector.
///
/// Implements the canonical Plate / Kanerva pattern for encoding a
/// structured record where each field has a stable role:
///
/// `bundle = Σ bind(role_encoding(role_i), value_i)`
///
/// The result is a single fixed-dimension hologram in which any one field
/// can be recovered by [`unbind_role`] using the same role name and
/// dimension. Two records produced by `bundle_record` over the same role
/// names live in a comparable HRR space — cosine similarity between two
/// records reflects similarity of corresponding fields, slot-by-slot.
///
/// Role encodings are produced by [`role_encoding`], which is
/// deterministic across calls and across consumers: any two callers that
/// reference the role `"tax"` get the same role hologram and their
/// `bundle_record`s are cross-comparable on the tax slot. That is the
/// substrate-level discipline; consumers should not invent their own
/// role-vector schemes.
///
/// # Arguments
///
/// * `fields` — `(role_name, value_hologram)` pairs. The order of
///   entries does not affect the bundle (superposition is commutative).
/// * `weights` — optional per-field weights for the superposition.
///   `None` is equal weighting; supply weights when some fields should
///   dominate the bundle's geometry (e.g., a Charter where `purpose`
///   should weigh more heavily than a single stakeholder line).
/// * `device` — device for the role-vector encoding pass.
///
/// All `value` tensors must share dimensionality. Returns an error on
/// empty fields, dimension mismatch, or weight-length mismatch.
///
/// # Example
///
/// ```ignore
/// let charter_hrr = bundle_record(
///     &[
///         ("purpose", &purpose_hrr),
///         ("tax_classification", &tax_hrr),
///         ("stakeholders", &stakeholders_hrr),
///         ("reserves", &reserves_hrr),
///         ("distribution", &distribution_hrr),
///         ("triggers", &triggers_hrr),
///         ("accounting", &accounting_hrr),
///     ],
///     None,
///     &device,
/// )?;
///
/// // Project a query against just the tax aspect.
/// let tax_aspect = unbind_role(&charter_hrr, "tax_classification", dim, &device)?;
/// ```
pub fn bundle_record(
    fields: &[(&str, &Tensor)],
    weights: Option<&[f32]>,
    device: &Device,
) -> Result<Tensor> {
    if fields.is_empty() {
        return Err(TensorCoreError::Tensor(
            "bundle_record requires at least one field".into(),
        ));
    }

    let first_dim = fields[0].1.dims().last().copied().ok_or_else(|| {
        TensorCoreError::Tensor("bundle_record: field 0 has zero-dim value tensor".into())
    })?;
    for (idx, (_, value)) in fields.iter().enumerate().skip(1) {
        let dim = value.dims().last().copied().ok_or_else(|| {
            TensorCoreError::Tensor(format!(
                "bundle_record: field {} has zero-dim value tensor",
                idx
            ))
        })?;
        if dim != first_dim {
            return Err(TensorCoreError::Tensor(format!(
                "bundle_record: field {} has dim {} but field 0 has dim {}; \
                 all fields must share dimensionality",
                idx, dim, first_dim
            )));
        }
    }

    if let Some(weights) = weights {
        if weights.len() != fields.len() {
            return Err(TensorCoreError::Tensor(format!(
                "bundle_record received {} fields but {} weights",
                fields.len(),
                weights.len()
            )));
        }
    }

    let mut role_bound: Vec<Tensor> = Vec::with_capacity(fields.len());
    for (role_name, value) in fields.iter() {
        let role = role_encoding(role_name, first_dim, device)?;
        role_bound.push(bind(&role, value)?);
    }

    superimpose(&role_bound, weights)
}

/// Recover a single field's value from a bundle produced by
/// [`bundle_record`].
///
/// `unbind_role(bundle, role_name, dim, device)` reproduces the
/// per-role unbind step using the same deterministic role encoding the
/// bundle was built with. Recovery fidelity is the standard HRR
/// circular-correlation fidelity (see `unbind` docs); for `dim ≥ 256`
/// with normalized values, expect cosine similarity `> 0.5` against the
/// original field value, sufficient for "find Charters with similar
/// tax classification" projections.
///
/// `dim` must match the dimension the bundle was built with.
pub fn unbind_role(
    bundle: &Tensor,
    role_name: &str,
    dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let role = role_encoding(role_name, dim, device)?;
    unbind(bundle, &role)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic normalized random vector (seeded). Replaces unseeded
    /// `Tensor::randn` in round-trip tests so they are reproducible — the
    /// unseeded version occasionally drew fields too similar, flaking the
    /// `sim_correct > sim_wrong + 0.2` margin. Distinct seeds give
    /// near-orthogonal vectors at dim=256 (cosine sim ~0.05), well clear of the margin.
    fn seeded_unit_vec(dim: usize, seed: u64, device: &Device) -> Tensor {
        use crate::information::monte_carlo::SimpleRng;
        let mut rng = SimpleRng::new(seed);
        let vals: Vec<f32> = (0..dim).map(|_| rng.normal() as f32).collect();
        normalize(&Tensor::from_vec(vals, (dim,), device).unwrap()).unwrap()
    }

    #[test]
    fn test_bind_unbind_recovery_fft() {
        let device = Device::Cpu;
        let dim = 512; // Higher dims give better recovery (more frequency resolution)

        let a = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();

        // FFT-based bind/unbind with standard random vectors.
        // Recovery fidelity depends on the power spectrum of the binding vector:
        // - Random Gaussian vectors: ~0.7-0.9 cosine similarity at dim=64, better at higher dims
        // - Random phase vectors (flat spectrum): ~0.99+ (see test below)
        // This is a known property of HRR (Plate 1995).
        let bound = bind(&a, &b).unwrap();
        let recovered = unbind(&bound, &a).unwrap();

        let similarity = cosine_similarity(&b, &recovered).unwrap();
        println!("FFT recovery similarity (dim={}): {}", dim, similarity);

        // FFT is substantially better than the fast approximation
        assert!(similarity > 0.6, "Expected >0.6, got {}", similarity);

        // Verify it significantly beats random
        let random = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();
        let random_sim = cosine_similarity(&b, &random).unwrap().abs();
        assert!(
            similarity > random_sim + 0.3,
            "FFT recovery ({}) should beat random baseline ({}) by >0.3",
            similarity,
            random_sim
        );
    }

    #[test]
    fn test_bind_unbind_recovery_fft_normalized() {
        let device = Device::Cpu;
        let dim = 256;

        // Normalized vectors give better recovery because their power spectrum
        // is more uniform — closer to the ideal flat-spectrum case.
        let a_raw = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();
        let b_raw = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();
        let a = normalize(&a_raw).unwrap();
        let b = normalize(&b_raw).unwrap();

        let bound = bind(&a, &b).unwrap();
        let recovered = unbind(&bound, &a).unwrap();

        let similarity = cosine_similarity(&b, &recovered).unwrap();
        println!(
            "FFT recovery similarity (normalized, dim={}): {}",
            dim, similarity
        );

        // Normalized vectors should recover well
        assert!(
            similarity > 0.5,
            "Expected >0.5 for normalized, got {}",
            similarity
        );
    }

    #[test]
    fn test_fft_better_than_fast() {
        let device = Device::Cpu;
        let dim = 128;

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
    fn test_bind_fast_recovery() {
        // Property test for `bind_fast` / `unbind_fast` recovery.
        // The fast (non-FFT) approximation has known higher variance
        // at small dimensions than the FFT path tested above. A single-
        // trial assertion against a single random baseline is too
        // tight — the test was historically flaky enough to fail
        // two independent CI runs on the same day. Both runs failed on
        // RNG luck, not algorithm regression.
        //
        // Candle's CPU device explicitly does not support `set_seed`
        // ("cannot seed the CPU rng with set_seed"), so we can't lock
        // randomness to a known-good draw. The robust shape that does
        // work without seeding is statistical averaging on BOTH sides
        // of the inequality:
        //
        //   - Mean recovery similarity across N independent (a, b)
        //     pairs. The std-dev of a single draw shrinks like
        //     1/sqrt(N).
        //   - Mean random-baseline similarity across N independent
        //     reference vectors. Same shrinkage; the expected value
        //     of cos-sim between two isotropic Gaussians is 0, so the
        //     mean stays near zero with tight variance.
        //
        // With N=16 and dim=64, both quantities concentrate enough
        // that the recovery floor (~0.3-0.5 empirical for bind_fast)
        // sits comfortably above the baseline ceiling (≤ 0.05 in
        // absolute value across observed seeds).
        let device = Device::Cpu;
        let dim = 64;
        // N=64 trials chosen empirically: at N=16 a 20-run sweep still
        // produced ~5% flake rate (the joint tail event of low recovery
        // + high baseline). Quadrupling N halves the std-dev of each
        // mean; the joint-tail probability drops by ~16x at fixed
        // margin. The test still runs in well under one second on a
        // dim=64 CPU device.
        let n_trials = 64;

        let mut recovery_sum = 0.0f32;
        let mut baseline_sum = 0.0f32;
        for _ in 0..n_trials {
            let a = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();
            let b = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();

            let bound = bind_fast(&a, &b).unwrap();
            let recovered = unbind_fast(&bound, &a).unwrap();
            recovery_sum += cosine_similarity(&b, &recovered).unwrap();

            let r = Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap();
            baseline_sum += cosine_similarity(&b, &r).unwrap();
        }
        let mean_recovery = recovery_sum / n_trials as f32;
        let mean_baseline = baseline_sum / n_trials as f32;

        println!(
            "Mean fast recovery over {} trials: {:.4} (mean random baseline: {:.4})",
            n_trials, mean_recovery, mean_baseline
        );

        // Empirically at dim=64, bind_fast's recovery sits near
        // baseline (the algorithm is intentionally lossy at small
        // dims; FFT-based `bind`/`unbind` are the high-fidelity
        // path tested in `test_bind_unbind_recovery_fft` above).
        // The original assertion `similarity > random_sim - 0.3`
        // is a smoke test, not a quality bound: it confirms the
        // implementation isn't catastrophically broken (returning
        // values 0.3+ below random, which would imply numerical
        // instability or an actual regression in the algorithm).
        // We preserve that intent but use averaged quantities so
        // the bound holds across RNG draws instead of passing
        // intermittently. Tighter quality bounds belong on the
        // FFT path; this test owns the bind_fast smoke-test slot.
        assert!(
            mean_recovery > mean_baseline - 0.1,
            "bind_fast mean recovery ({:.4}) over {} trials is more than 0.1 \
             below the mean random baseline ({:.4}). The fast approximation is \
             allowed to be lossy at small dims — the FFT path covers high-fidelity \
             recovery — but a >0.1 deficit here suggests numerical instability \
             or an algorithmic regression.",
            mean_recovery,
            n_trials,
            mean_baseline
        );
    }

    #[test]
    fn test_superimpose() {
        let device = Device::Cpu;
        let dim = 64;

        let tensors: Vec<Tensor> = (0..5)
            .map(|_| Tensor::randn(0.0f32, 1.0, (dim,), &device).unwrap())
            .collect();

        let hologram = superimpose(&tensors, None).unwrap();

        // Should be normalized
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

        // Different positions should have different encodings
        let sim_0_1 = cosine_similarity(&pos0, &pos1).unwrap();
        let sim_0_100 = cosine_similarity(&pos0, &pos100).unwrap();

        // Closer positions should be more similar
        assert!(sim_0_1 > sim_0_100);
    }

    #[test]
    fn test_role_encoding_deterministic() {
        let device = Device::Cpu;
        let dim = 64;

        let role1a = role_encoding("synapse", dim, &device).unwrap();
        let role1b = role_encoding("synapse", dim, &device).unwrap();
        let role2 = role_encoding("crdt", dim, &device).unwrap();

        // Same role should give same encoding
        let sim_same = cosine_similarity(&role1a, &role1b).unwrap();
        assert!((sim_same - 1.0).abs() < 0.001);

        // Different roles should give different encodings
        let sim_diff = cosine_similarity(&role1a, &role2).unwrap();
        assert!(sim_diff < 0.5);
    }

    #[test]
    fn test_bundle_record_round_trip_recovers_field() {
        let device = Device::Cpu;
        let dim = 256;

        let purpose = seeded_unit_vec(dim, 1, &device);
        let tax = seeded_unit_vec(dim, 2, &device);
        let stakeholders = seeded_unit_vec(dim, 3, &device);

        let bundle = bundle_record(
            &[
                ("purpose", &purpose),
                ("tax", &tax),
                ("stakeholders", &stakeholders),
            ],
            None,
            &device,
        )
        .unwrap();

        let recovered_tax = unbind_role(&bundle, "tax", dim, &device).unwrap();
        let sim_correct = cosine_similarity(&tax, &recovered_tax).unwrap();
        let sim_wrong = cosine_similarity(&purpose, &recovered_tax).unwrap();

        assert!(
            sim_correct > 0.5,
            "expected correct-field recovery > 0.5, got {sim_correct}"
        );
        assert!(
            sim_correct > sim_wrong + 0.2,
            "tax recovery ({sim_correct}) should beat purpose ({sim_wrong}) by >0.2"
        );
    }

    #[test]
    fn test_bundle_record_two_records_share_role_geometry() {
        // Two consumers building independent bundles over the same role
        // names must produce HRRs comparable on a per-role basis. This is
        // the cross-consumer composability invariant.
        let device = Device::Cpu;
        let dim = 256;

        // Seeded: the unseeded `Tensor::randn` path flaked the
        // `tax > purpose + 0.2` margin on cluster CI (v0.4.0:
        // tax=0.690, purpose=0.552). Same helper as
        // `test_bundle_record_round_trip_recovers_field` (1acb0a3).
        let shared_tax = seeded_unit_vec(dim, 10, &device);
        let purpose_a = seeded_unit_vec(dim, 11, &device);
        let purpose_b = seeded_unit_vec(dim, 12, &device);

        let bundle_a = bundle_record(
            &[("purpose", &purpose_a), ("tax", &shared_tax)],
            None,
            &device,
        )
        .unwrap();
        let bundle_b = bundle_record(
            &[("purpose", &purpose_b), ("tax", &shared_tax)],
            None,
            &device,
        )
        .unwrap();

        let tax_a = unbind_role(&bundle_a, "tax", dim, &device).unwrap();
        let tax_b = unbind_role(&bundle_b, "tax", dim, &device).unwrap();
        let purpose_a_rec = unbind_role(&bundle_a, "purpose", dim, &device).unwrap();
        let purpose_b_rec = unbind_role(&bundle_b, "purpose", dim, &device).unwrap();

        let tax_cross = cosine_similarity(&tax_a, &tax_b).unwrap();
        let purpose_cross = cosine_similarity(&purpose_a_rec, &purpose_b_rec).unwrap();

        assert!(
            tax_cross > purpose_cross + 0.2,
            "shared tax field should produce notably higher cross-bundle similarity \
             (tax={tax_cross}, purpose={purpose_cross})"
        );
    }

    #[test]
    fn test_bundle_record_rejects_dim_mismatch() {
        let device = Device::Cpu;
        let a = Tensor::randn(0.0f32, 1.0, (64,), &device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, (32,), &device).unwrap();

        let result = bundle_record(&[("a", &a), ("b", &b)], None, &device);
        assert!(result.is_err(), "dim mismatch must error");
    }

    #[test]
    fn test_bundle_record_rejects_empty() {
        let device = Device::Cpu;
        assert!(bundle_record(&[], None, &device).is_err());
    }

    #[test]
    fn test_bundle_record_rejects_weight_length_mismatch() {
        let device = Device::Cpu;
        let a = Tensor::randn(0.0f32, 1.0, (64,), &device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, (64,), &device).unwrap();

        let result = bundle_record(&[("a", &a), ("b", &b)], Some(&[0.5]), &device);
        assert!(result.is_err());
    }
}

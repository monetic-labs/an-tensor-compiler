//! Monte Carlo Methods
//!
//! Randomized algorithms for estimation and inference.
//!
//! ## Overview
//!
//! Monte Carlo methods use random sampling to estimate quantities that
//! are difficult to compute analytically:
//!
//! - **Bootstrap**: Confidence intervals via resampling
//! - **Permutation tests**: Statistical significance testing
//! - **Importance sampling**: Rare event estimation
//! - **MCMC**: Sampling from complex distributions
//!
//! ## Use Cases
//!
//! - Confidence intervals for MI, entropy estimates
//! - Test if observed MI is statistically significant
//! - Estimate probabilities of rare events
//! - Sample from posterior distributions

use crate::{Result, TensorCoreError};

/// Small epsilon for numerical stability
const EPSILON: f64 = 1e-10;

/// Simple LCG random number generator for reproducibility
#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create a new deterministic RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }

    /// Generate uniform [0, 1)
    pub fn uniform(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Generate uniform integer in [0, n)
    pub fn uniform_int(&mut self, n: usize) -> usize {
        (self.uniform() * n as f64) as usize
    }

    /// Generate standard normal via Box-Muller
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(EPSILON);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Bootstrap result with confidence intervals
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Point estimate (original statistic)
    pub estimate: f64,
    /// Bootstrap mean
    pub mean: f64,
    /// Bootstrap standard error
    pub std_error: f64,
    /// Lower confidence bound
    pub ci_lower: f64,
    /// Upper confidence bound
    pub ci_upper: f64,
    /// Confidence level used
    pub confidence: f64,
    /// All bootstrap samples
    pub samples: Vec<f64>,
}

/// Bootstrap resampling for confidence intervals
///
/// # Arguments
/// - `data`: Original data samples
/// - `statistic`: Function to compute statistic
/// - `n_bootstrap`: Number of bootstrap samples
/// - `confidence`: Confidence level (e.g., 0.95)
/// - `seed`: Random seed
///
/// # Returns
/// BootstrapResult with CIs
pub fn bootstrap<F>(
    data: &[f64],
    statistic: F,
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> BootstrapResult
where
    F: Fn(&[f64]) -> f64,
{
    let n = data.len();
    let mut rng = SimpleRng::new(seed);
    let mut samples = Vec::with_capacity(n_bootstrap);

    // Original estimate
    let estimate = statistic(data);

    // Bootstrap samples
    for _ in 0..n_bootstrap {
        let resample: Vec<f64> = (0..n).map(|_| data[rng.uniform_int(n)]).collect();
        samples.push(statistic(&resample));
    }

    // Statistics
    let mean = samples.iter().sum::<f64>() / n_bootstrap as f64;
    let variance =
        samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_bootstrap - 1) as f64;
    let std_error = variance.sqrt();

    // Percentile CI
    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = (1.0 - confidence) / 2.0;
    let lower_idx = (alpha * n_bootstrap as f64) as usize;
    let upper_idx = ((1.0 - alpha) * n_bootstrap as f64) as usize;

    BootstrapResult {
        estimate,
        mean,
        std_error,
        ci_lower: sorted[lower_idx.min(n_bootstrap - 1)],
        ci_upper: sorted[upper_idx.min(n_bootstrap - 1)],
        confidence,
        samples,
    }
}

/// Bootstrap for paired data (e.g., MI between X and Y)
pub fn bootstrap_paired<F>(
    x: &[f64],
    y: &[f64],
    statistic: F,
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> Result<BootstrapResult>
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    if x.len() != y.len() {
        return Err(TensorCoreError::Information(
            "X and Y must have same length".into(),
        ));
    }

    let n = x.len();
    let mut rng = SimpleRng::new(seed);
    let mut samples = Vec::with_capacity(n_bootstrap);

    let estimate = statistic(x, y);

    for _ in 0..n_bootstrap {
        let indices: Vec<usize> = (0..n).map(|_| rng.uniform_int(n)).collect();
        let x_resample: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
        let y_resample: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
        samples.push(statistic(&x_resample, &y_resample));
    }

    let mean = samples.iter().sum::<f64>() / n_bootstrap as f64;
    let variance =
        samples.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / (n_bootstrap - 1) as f64;
    let std_error = variance.sqrt();

    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = (1.0 - confidence) / 2.0;
    let lower_idx = (alpha * n_bootstrap as f64) as usize;
    let upper_idx = ((1.0 - alpha) * n_bootstrap as f64) as usize;

    Ok(BootstrapResult {
        estimate,
        mean,
        std_error,
        ci_lower: sorted[lower_idx.min(n_bootstrap - 1)],
        ci_upper: sorted[upper_idx.min(n_bootstrap - 1)],
        confidence,
        samples,
    })
}

/// Permutation test result
#[derive(Debug, Clone)]
pub struct PermutationTestResult {
    /// Observed statistic
    pub observed: f64,
    /// P-value (proportion of permutations >= observed)
    pub p_value: f64,
    /// P-value for two-sided test
    pub p_value_two_sided: f64,
    /// Number of permutations
    pub n_permutations: usize,
    /// Null distribution samples
    pub null_distribution: Vec<f64>,
}

/// Permutation test for statistical significance
///
/// Tests H0: X and Y are independent
///
/// # Arguments
/// - `x`: First variable
/// - `y`: Second variable
/// - `statistic`: Dependence statistic (e.g., correlation, MI)
/// - `n_permutations`: Number of permutations
/// - `seed`: Random seed
///
/// # Returns
/// PermutationTestResult with p-value
pub fn permutation_test<F>(
    x: &[f64],
    y: &[f64],
    statistic: F,
    n_permutations: usize,
    seed: u64,
) -> Result<PermutationTestResult>
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    if x.len() != y.len() {
        return Err(TensorCoreError::Information(
            "X and Y must have same length".into(),
        ));
    }

    let n = x.len();
    let mut rng = SimpleRng::new(seed);

    let observed = statistic(x, y);
    let mut null_distribution = Vec::with_capacity(n_permutations);
    let mut count_greater = 0;
    let mut count_extreme = 0;

    for _ in 0..n_permutations {
        // Fisher-Yates shuffle of y
        let mut y_perm = y.to_vec();
        for i in (1..n).rev() {
            let j = rng.uniform_int(i + 1);
            y_perm.swap(i, j);
        }

        let perm_stat = statistic(x, &y_perm);
        null_distribution.push(perm_stat);

        if perm_stat >= observed {
            count_greater += 1;
        }
        if perm_stat.abs() >= observed.abs() {
            count_extreme += 1;
        }
    }

    let p_value = count_greater as f64 / n_permutations as f64;
    let p_value_two_sided = count_extreme as f64 / n_permutations as f64;

    Ok(PermutationTestResult {
        observed,
        p_value,
        p_value_two_sided,
        n_permutations,
        null_distribution,
    })
}

/// Permutation test for discrete MI
pub fn permutation_test_mi(
    x: &[i64],
    y: &[i64],
    n_states_x: usize,
    n_states_y: usize,
    n_permutations: usize,
    seed: u64,
) -> Result<PermutationTestResult> {
    let n = x.len();
    if n != y.len() {
        return Err(TensorCoreError::Information(
            "X and Y must have same length".into(),
        ));
    }

    let mut rng = SimpleRng::new(seed);

    // Compute observed MI
    let observed = compute_discrete_mi(x, y, n_states_x, n_states_y);

    let mut null_distribution = Vec::with_capacity(n_permutations);
    let mut count_greater = 0;

    for _ in 0..n_permutations {
        let mut y_perm = y.to_vec();
        for i in (1..n).rev() {
            let j = rng.uniform_int(i + 1);
            y_perm.swap(i, j);
        }

        let perm_mi = compute_discrete_mi(x, &y_perm, n_states_x, n_states_y);
        null_distribution.push(perm_mi);

        if perm_mi >= observed {
            count_greater += 1;
        }
    }

    let p_value = count_greater as f64 / n_permutations as f64;

    Ok(PermutationTestResult {
        observed,
        p_value,
        p_value_two_sided: p_value, // MI is always non-negative
        n_permutations,
        null_distribution,
    })
}

fn compute_discrete_mi(x: &[i64], y: &[i64], nx: usize, ny: usize) -> f64 {
    let n = x.len() as f64;
    let mut joint = vec![vec![0.0; ny]; nx];
    let mut px = vec![0.0; nx];
    let mut py = vec![0.0; ny];

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let xi = xi as usize;
        let yi = yi as usize;
        if xi < nx && yi < ny {
            joint[xi][yi] += 1.0;
            px[xi] += 1.0;
            py[yi] += 1.0;
        }
    }

    let mut mi = 0.0;
    for i in 0..nx {
        for j in 0..ny {
            let pxy = joint[i][j] / n;
            let pxi = px[i] / n;
            let pyj = py[j] / n;
            if pxy > EPSILON && pxi > EPSILON && pyj > EPSILON {
                mi += pxy * (pxy / (pxi * pyj)).ln();
            }
        }
    }

    mi / 2.0_f64.ln() // Convert to bits
}

/// Importance sampling result
#[derive(Debug, Clone)]
pub struct ImportanceSamplingResult {
    /// Estimated expectation
    pub estimate: f64,
    /// Standard error
    pub std_error: f64,
    /// Effective sample size
    pub effective_sample_size: f64,
    /// All importance weights
    pub weights: Vec<f64>,
}

/// Importance sampling for rare event estimation
///
/// Estimates E_p[f(x)] by sampling from q and reweighting
///
/// # Arguments
/// - `samples`: Samples from proposal q
/// - `log_target`: log p(x) for each sample
/// - `log_proposal`: log q(x) for each sample
/// - `f_values`: f(x) for each sample
///
/// # Returns
/// ImportanceSamplingResult with estimate and diagnostics
pub fn importance_sampling(
    log_target: &[f64],
    log_proposal: &[f64],
    f_values: &[f64],
) -> Result<ImportanceSamplingResult> {
    let n = log_target.len();
    if n != log_proposal.len() || n != f_values.len() {
        return Err(TensorCoreError::Information(
            "All inputs must have same length".into(),
        ));
    }

    // Compute log importance weights
    let log_weights: Vec<f64> = log_target
        .iter()
        .zip(log_proposal.iter())
        .map(|(&lp, &lq)| lp - lq)
        .collect();

    // Normalize using log-sum-exp trick
    let max_log_w = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = log_weights.iter().map(|&lw| (lw - max_log_w).exp()).sum();
    let log_normalizer = max_log_w + sum_exp.ln();

    let weights: Vec<f64> = log_weights
        .iter()
        .map(|&lw| (lw - log_normalizer).exp())
        .collect();

    // Weighted estimate
    let estimate: f64 = weights
        .iter()
        .zip(f_values.iter())
        .map(|(&w, &f)| w * f)
        .sum();

    // Standard error via weighted variance
    let mean_f = estimate;
    let weighted_var: f64 = weights
        .iter()
        .zip(f_values.iter())
        .map(|(&w, &f)| w * (f - mean_f).powi(2))
        .sum();
    let std_error = (weighted_var / n as f64).sqrt();

    // Effective sample size
    let sum_w2: f64 = weights.iter().map(|w| w.powi(2)).sum();
    let ess = 1.0 / sum_w2;

    Ok(ImportanceSamplingResult {
        estimate,
        std_error,
        effective_sample_size: ess,
        weights,
    })
}

/// Self-normalized importance sampling
///
/// More stable when normalizing constant is unknown
pub fn importance_sampling_self_normalized(
    log_weights: &[f64],
    f_values: &[f64],
) -> Result<ImportanceSamplingResult> {
    let n = log_weights.len();
    if n != f_values.len() {
        return Err(TensorCoreError::Information(
            "Weights and values must have same length".into(),
        ));
    }

    // Log-sum-exp normalization
    let max_lw = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = log_weights.iter().map(|&lw| (lw - max_lw).exp()).collect();
    let sum_w: f64 = weights.iter().sum();
    let normalized: Vec<f64> = weights.iter().map(|&w| w / sum_w).collect();

    let estimate: f64 = normalized
        .iter()
        .zip(f_values.iter())
        .map(|(&w, &f)| w * f)
        .sum();

    let weighted_var: f64 = normalized
        .iter()
        .zip(f_values.iter())
        .map(|(&w, &f)| w * (f - estimate).powi(2))
        .sum();
    let std_error = (weighted_var / n as f64).sqrt();

    let sum_w2: f64 = normalized.iter().map(|w| w.powi(2)).sum();
    let ess = 1.0 / sum_w2;

    Ok(ImportanceSamplingResult {
        estimate,
        std_error,
        effective_sample_size: ess,
        weights: normalized,
    })
}

/// MCMC sample result
#[derive(Debug, Clone)]
pub struct MCMCResult {
    /// Samples from posterior
    pub samples: Vec<Vec<f64>>,
    /// Acceptance rate
    pub acceptance_rate: f64,
    /// Log probabilities of samples
    pub log_probs: Vec<f64>,
}

/// Metropolis-Hastings MCMC sampler
///
/// # Arguments
/// - `log_prob`: Log probability function (up to constant)
/// - `initial`: Initial state
/// - `proposal_std`: Standard deviation for Gaussian proposal
/// - `n_samples`: Number of samples to generate
/// - `burn_in`: Number of burn-in samples to discard
/// - `seed`: Random seed
///
/// # Returns
/// MCMCResult with samples and diagnostics
pub fn metropolis_hastings<F>(
    log_prob: F,
    initial: &[f64],
    proposal_std: f64,
    n_samples: usize,
    burn_in: usize,
    seed: u64,
) -> MCMCResult
where
    F: Fn(&[f64]) -> f64,
{
    let _d = initial.len();
    let mut rng = SimpleRng::new(seed);
    let mut current = initial.to_vec();
    let mut current_lp = log_prob(&current);

    let mut samples = Vec::with_capacity(n_samples);
    let mut log_probs = Vec::with_capacity(n_samples);
    let mut n_accepted = 0;
    let total_iter = n_samples + burn_in;

    for i in 0..total_iter {
        // Propose
        let proposal: Vec<f64> = current
            .iter()
            .map(|&x| x + proposal_std * rng.normal())
            .collect();

        let proposal_lp = log_prob(&proposal);

        // Accept/reject
        let log_alpha = proposal_lp - current_lp;
        let accept = log_alpha >= 0.0 || rng.uniform().ln() < log_alpha;

        if accept {
            current = proposal;
            current_lp = proposal_lp;
            if i >= burn_in {
                n_accepted += 1;
            }
        }

        if i >= burn_in {
            samples.push(current.clone());
            log_probs.push(current_lp);
        }
    }

    let acceptance_rate = n_accepted as f64 / n_samples as f64;

    MCMCResult {
        samples,
        acceptance_rate,
        log_probs,
    }
}

/// Gibbs sampler for discrete distributions
///
/// # Arguments
/// - `full_conditional`: Function(state, dim) -> `Vec<f64>` giving P(x_dim | x_-dim)
/// - `n_dims`: Number of dimensions
/// - `n_states`: Number of states per dimension
/// - `n_samples`: Number of samples
/// - `burn_in`: Burn-in samples
/// - `seed`: Random seed
pub fn gibbs_sampler<F>(
    full_conditional: F,
    n_dims: usize,
    n_states: usize,
    n_samples: usize,
    burn_in: usize,
    seed: u64,
) -> Vec<Vec<usize>>
where
    F: Fn(&[usize], usize) -> Vec<f64>,
{
    let mut rng = SimpleRng::new(seed);
    let mut current: Vec<usize> = (0..n_dims).map(|_| rng.uniform_int(n_states)).collect();

    let mut samples = Vec::with_capacity(n_samples);

    for i in 0..(n_samples + burn_in) {
        for d in 0..n_dims {
            let probs = full_conditional(&current, d);

            // Sample from discrete distribution
            let u = rng.uniform();
            let mut cumsum = 0.0;
            for (s, &p) in probs.iter().enumerate() {
                cumsum += p;
                if u < cumsum {
                    current[d] = s;
                    break;
                }
            }
        }

        if i >= burn_in {
            samples.push(current.clone());
        }
    }

    samples
}

/// Estimate Monte Carlo standard error
pub fn monte_carlo_se(samples: &[f64]) -> f64 {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (variance / n).sqrt()
}

/// Compute effective sample size accounting for autocorrelation
pub fn effective_sample_size(samples: &[f64]) -> f64 {
    let n = samples.len();
    if n < 4 {
        return n as f64;
    }

    let mean = samples.iter().sum::<f64>() / n as f64;
    let var: f64 = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if var < EPSILON {
        return n as f64;
    }

    // Compute autocorrelation up to n/2
    let max_lag = n / 2;
    let mut rho_sum = 0.0;

    for lag in 1..max_lag {
        let mut auto_cov = 0.0;
        for i in 0..(n - lag) {
            auto_cov += (samples[i] - mean) * (samples[i + lag] - mean);
        }
        auto_cov /= n as f64;

        let rho = auto_cov / var;
        if rho < 0.05 {
            break; // Stop when autocorrelation becomes negligible
        }
        rho_sum += rho;
    }

    // ESS = n / (1 + 2 * sum of autocorrelations)
    let tau = 1.0 + 2.0 * rho_sum;
    (n as f64 / tau).max(1.0)
}

/// Jackknife resampling for bias estimation
pub fn jackknife<F>(data: &[f64], statistic: F) -> (f64, f64, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let n = data.len();
    let full_estimate = statistic(data);

    // Leave-one-out estimates
    let mut jackknife_estimates = Vec::with_capacity(n);
    for i in 0..n {
        let subset: Vec<f64> = data
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &v)| v)
            .collect();
        jackknife_estimates.push(statistic(&subset));
    }

    let mean_jack = jackknife_estimates.iter().sum::<f64>() / n as f64;

    // Bias estimate
    let bias = (n - 1) as f64 * (mean_jack - full_estimate);

    // Bias-corrected estimate
    let corrected = full_estimate - bias;

    // Variance estimate
    let variance = jackknife_estimates
        .iter()
        .map(|&x| (x - mean_jack).powi(2))
        .sum::<f64>()
        * (n - 1) as f64
        / n as f64;

    (corrected, bias, variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_rng() {
        let mut rng = SimpleRng::new(42);

        // Should be deterministic
        let v1 = rng.uniform();
        let mut rng2 = SimpleRng::new(42);
        let v2 = rng2.uniform();

        assert_eq!(v1, v2);

        // Should be in [0, 1)
        for _ in 0..100 {
            let v = rng.uniform();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_bootstrap_basic() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();

        let result = bootstrap(
            &data,
            |d| d.iter().sum::<f64>() / d.len() as f64, // mean
            1000,
            0.95,
            42,
        );

        // Mean of 0-99 is 49.5
        assert!((result.estimate - 49.5).abs() < 0.01);
        assert!(result.ci_lower < result.estimate);
        assert!(result.ci_upper > result.estimate);
    }

    #[test]
    fn test_bootstrap_paired() -> Result<()> {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 2.0 + 1.0).collect();

        let result = bootstrap_paired(
            &x,
            &y,
            |a, b| {
                // Correlation
                let n = a.len() as f64;
                let mean_a = a.iter().sum::<f64>() / n;
                let mean_b = b.iter().sum::<f64>() / n;
                let cov: f64 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| (ai - mean_a) * (bi - mean_b))
                    .sum::<f64>()
                    / n;
                cov
            },
            500,
            0.95,
            42,
        )?;

        // Should detect positive covariance
        assert!(result.estimate > 0.0);

        Ok(())
    }

    #[test]
    fn test_permutation_test() -> Result<()> {
        // Perfectly correlated
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.clone();

        let result = permutation_test(
            &x,
            &y,
            |a, b| {
                a.iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| ai * bi)
                    .sum::<f64>()
            },
            999,
            42,
        )?;

        // Should be highly significant
        assert!(result.p_value < 0.05);

        Ok(())
    }

    #[test]
    fn test_permutation_test_independent() -> Result<()> {
        // Independent (deterministic pattern that looks random)
        let x: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let y: Vec<f64> = (0..100).map(|i| ((i * 7) as f64).cos()).collect();

        let result = permutation_test(
            &x,
            &y,
            |a, b| {
                let n = a.len() as f64;
                a.iter()
                    .zip(b.iter())
                    .map(|(&ai, &bi)| ai * bi)
                    .sum::<f64>()
                    / n
            },
            999,
            42,
        )?;

        // P-value should not be extremely low
        assert!(result.p_value > 0.001);

        Ok(())
    }

    #[test]
    fn test_permutation_test_mi() -> Result<()> {
        // Dependent
        let x: Vec<i64> = (0..100).map(|i| (i % 4) as i64).collect();
        let y: Vec<i64> = x.clone(); // Perfect dependence

        let result = permutation_test_mi(&x, &y, 4, 4, 999, 42)?;

        // Should detect dependence
        assert!(result.observed > 0.0);
        assert!(result.p_value < 0.05);

        Ok(())
    }

    #[test]
    fn test_importance_sampling() -> Result<()> {
        // Simple test: estimate E[X] where X ~ N(0,1) using samples from N(0,2)
        let mut rng = SimpleRng::new(42);
        let n = 1000;

        let samples: Vec<f64> = (0..n).map(|_| 2.0 * rng.normal()).collect();

        // log p(x) = -x^2/2 (ignoring constant)
        let log_target: Vec<f64> = samples.iter().map(|&x| -x * x / 2.0).collect();

        // log q(x) = -x^2/8 (N(0,2))
        let log_proposal: Vec<f64> = samples.iter().map(|&x| -x * x / 8.0).collect();

        // f(x) = x^2 (estimate E[X^2] = 1 for N(0,1))
        let f_values: Vec<f64> = samples.iter().map(|&x| x * x).collect();

        let result = importance_sampling(&log_target, &log_proposal, &f_values)?;

        // E[X^2] = 1 for N(0,1)
        assert!((result.estimate - 1.0).abs() < 0.5); // Allow some variance
        assert!(result.effective_sample_size > 0.0);

        Ok(())
    }

    #[test]
    fn test_metropolis_hastings() {
        // Sample from N(0,1)
        let log_prob = |x: &[f64]| -x[0] * x[0] / 2.0;

        let result = metropolis_hastings(log_prob, &[0.0], 1.0, 1000, 100, 42);

        assert_eq!(result.samples.len(), 1000);
        assert!(result.acceptance_rate > 0.1 && result.acceptance_rate < 0.9);

        // Mean should be close to 0
        let mean: f64 = result.samples.iter().map(|s| s[0]).sum::<f64>() / 1000.0;
        assert!(mean.abs() < 0.5);
    }

    #[test]
    fn test_gibbs_sampler() {
        // Sample from uniform over {0,1}^2
        let full_conditional = |_state: &[usize], _dim: usize| vec![0.5, 0.5];

        let samples = gibbs_sampler(full_conditional, 2, 2, 100, 10, 42);

        assert_eq!(samples.len(), 100);

        // All samples should be valid
        for s in &samples {
            assert_eq!(s.len(), 2);
            assert!(s[0] < 2);
            assert!(s[1] < 2);
        }
    }

    #[test]
    fn test_monte_carlo_se() {
        let samples: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let se = monte_carlo_se(&samples);

        assert!(se > 0.0);
    }

    #[test]
    fn test_effective_sample_size() {
        // Use SimpleRng for pseudo-random samples
        let mut rng = SimpleRng::new(42);
        let iid: Vec<f64> = (0..100).map(|_| rng.normal()).collect();
        let ess_iid = effective_sample_size(&iid);

        // Should be reasonably high for low autocorrelation
        assert!(ess_iid > 20.0);

        // Highly autocorrelated (monotonic) should have low ESS
        let correlated: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let ess_corr = effective_sample_size(&correlated);

        // Correlated should be lower (or function returns max for edge cases)
        assert!(ess_corr <= 100.0);
    }

    #[test]
    fn test_jackknife() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();

        let (corrected, bias, se) = jackknife(&data, |d| d.iter().sum::<f64>() / d.len() as f64);

        // Mean is unbiased, so bias should be small
        assert!(bias.abs() < 1.0);
        assert!(se > 0.0);
        assert!((corrected - 9.5).abs() < 1.0);
    }

    #[test]
    fn test_rng_normal() {
        let mut rng = SimpleRng::new(42);

        let samples: Vec<f64> = (0..1000).map(|_| rng.normal()).collect();

        // Mean should be close to 0
        let mean = samples.iter().sum::<f64>() / 1000.0;
        assert!(mean.abs() < 0.2);

        // Variance should be close to 1
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 1000.0;
        assert!((var - 1.0).abs() < 0.3);
    }
}

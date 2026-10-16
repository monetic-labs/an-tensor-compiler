//! Lyapunov Exponent Computation
//!
//! GPU-accelerated chaos detection via Lyapunov exponents for time series.
//!
//! # Overview
//!
//! Lyapunov exponents measure the rate of separation of infinitesimally close
//! trajectories in a dynamical system. Positive exponents indicate chaos (exponential
//! divergence), while negative exponents indicate stability (convergence).
//!
//! # Algorithm: Rosenstein et al. (1993)
//!
//! The largest Lyapunov exponent (λ₁) is computed using the Rosenstein algorithm:
//!
//! 1. **Embed** the time series in m-dimensional phase space using delay embedding:
//!    `X(t) = [x(t), x(t+τ), x(t+2τ), ..., x(t+(m-1)τ)]`
//!
//! 2. **Find nearest neighbors** for each embedded point (excluding temporal neighbors
//!    within a mean period to avoid false neighbors).
//!
//! 3. **Track divergence** of nearest neighbor pairs over time.
//!
//! 4. **Estimate λ₁** from the slope of log(divergence) vs. time.
//!
//! # Trading Application
//!
//! For regime detection in financial time series:
//! - **High λ₁ (>0)**: Chaotic regime, high uncertainty, potential large moves
//! - **λ₁ ≈ 0**: Noisy/random walk regime
//! - **Low λ₁ (<0)**: Stable/mean-reverting regime
//!
//! # References
//!
//! - Rosenstein, Collins, De Luca (1993): "A practical method for calculating largest
//!   Lyapunov exponents from small data sets"
//! - arXiv:1703.04385: TDA + Lyapunov for crash prediction
//! - Frontiers fphy.2021.572216: Chaos analysis in neural networks

use crate::{Result, TensorCoreError};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

/// Lyapunov exponent analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LyapunovFeatures {
    /// Largest Lyapunov exponent (λ₁) per symbol
    pub largest_exponent: Vec<f64>,

    /// Lyapunov spectrum (if computed) - top k exponents per symbol
    pub spectrum: Option<Vec<Vec<f64>>>,

    /// Divergence curves per symbol: log(d(t)) vs t
    #[serde(skip)]
    pub divergence_curves: Vec<Vec<f64>>,

    /// Chaos classification per symbol
    pub chaos_regime: Vec<ChaosRegime>,

    /// Computation metadata
    pub metadata: LyapunovMetadata,
}

impl LyapunovFeatures {
    /// Get mean largest exponent across all symbols.
    pub fn mean_exponent(&self) -> f64 {
        if self.largest_exponent.is_empty() {
            0.0
        } else {
            self.largest_exponent.iter().sum::<f64>() / self.largest_exponent.len() as f64
        }
    }

    /// Get max largest exponent across all symbols.
    pub fn max_exponent(&self) -> f64 {
        self.largest_exponent
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Fraction of symbols in chaotic regime.
    pub fn chaos_fraction(&self) -> f64 {
        if self.chaos_regime.is_empty() {
            0.0
        } else {
            let chaotic = self
                .chaos_regime
                .iter()
                .filter(|r| matches!(r, ChaosRegime::Chaotic))
                .count();
            chaotic as f64 / self.chaos_regime.len() as f64
        }
    }

    /// Feature vector for ML: [mean_λ, max_λ, chaos_fraction, entropy]
    pub fn feature_vector(&self) -> Vec<f64> {
        let mean = self.mean_exponent();
        let max = self.max_exponent();
        let chaos_frac = self.chaos_fraction();

        // Entropy of exponent distribution
        let entropy = self.exponent_entropy();

        vec![mean, max, chaos_frac, entropy]
    }

    /// Shannon entropy of Lyapunov exponent distribution.
    fn exponent_entropy(&self) -> f64 {
        if self.largest_exponent.is_empty() {
            return 0.0;
        }

        // Bin exponents into categories
        let mut counts = [0usize; 5]; // very_neg, neg, zero, pos, very_pos
        for &exp in &self.largest_exponent {
            let idx = if exp < -0.5 {
                0
            } else if exp < -0.1 {
                1
            } else if exp < 0.1 {
                2
            } else if exp < 0.5 {
                3
            } else {
                4
            };
            counts[idx] += 1;
        }

        let n = self.largest_exponent.len() as f64;
        let mut entropy = 0.0;
        for &c in &counts {
            if c > 0 {
                let p = c as f64 / n;
                entropy -= p * p.ln();
            }
        }
        entropy
    }
}

/// Chaos regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChaosRegime {
    /// λ₁ < -0.1: Stable, mean-reverting
    Stable,
    /// -0.1 ≤ λ₁ ≤ 0.1: Random walk / noisy
    RandomWalk,
    /// λ₁ > 0.1: Chaotic, exponential divergence
    Chaotic,
}

impl ChaosRegime {
    /// Classify from Lyapunov exponent value.
    pub fn from_exponent(lambda: f64) -> Self {
        if lambda < -0.1 {
            ChaosRegime::Stable
        } else if lambda > 0.1 {
            ChaosRegime::Chaotic
        } else {
            ChaosRegime::RandomWalk
        }
    }

    /// Get numeric code for ML.
    pub fn as_code(&self) -> i32 {
        match self {
            ChaosRegime::Stable => -1,
            ChaosRegime::RandomWalk => 0,
            ChaosRegime::Chaotic => 1,
        }
    }
}

/// Metadata about Lyapunov computation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LyapunovMetadata {
    /// Number of symbols processed
    pub n_symbols: usize,
    /// Time series length
    pub n_timestamps: usize,
    /// Embedding dimension used
    pub embedding_dim: usize,
    /// Time delay used
    pub time_delay: usize,
    /// Mean period for temporal exclusion
    pub mean_period: usize,
    /// Max evolution time for divergence tracking
    pub max_evolution: usize,
    /// Computation time in milliseconds
    pub compute_time_ms: u64,
}

/// Configuration for Lyapunov computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LyapunovConfig {
    /// Embedding dimension (default: 10)
    /// Higher = more phase space structure captured, but needs more data
    pub embedding_dim: usize,

    /// Time delay for embedding (default: 1)
    /// Often set to first minimum of autocorrelation or mutual information
    pub time_delay: usize,

    /// Mean period for temporal exclusion (default: 10)
    /// Nearest neighbors within this distance in time are excluded
    pub mean_period: usize,

    /// Maximum evolution time for tracking divergence (default: 50)
    pub max_evolution: usize,

    /// Minimum data points required (default: 100)
    pub min_data_points: usize,

    /// Number of nearest neighbors to average (default: 1)
    /// Using k>1 can reduce noise
    pub k_neighbors: usize,
}

impl Default for LyapunovConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 10,
            time_delay: 1,
            mean_period: 10,
            max_evolution: 50,
            min_data_points: 100,
            k_neighbors: 1,
        }
    }
}

impl LyapunovConfig {
    /// Create config optimized for financial time series.
    pub fn for_finance() -> Self {
        Self {
            embedding_dim: 7,    // Lower for noisy data
            time_delay: 1,       // Daily/bar data
            mean_period: 5,      // ~1 week
            max_evolution: 20,   // ~1 month lookahead
            min_data_points: 50, // Allow smaller windows
            k_neighbors: 3,      // Average to reduce noise
        }
    }

    /// Create config for high-frequency data.
    pub fn for_hft() -> Self {
        Self {
            embedding_dim: 5,
            time_delay: 1,
            mean_period: 10,
            max_evolution: 100,
            min_data_points: 200,
            k_neighbors: 5,
        }
    }
}

/// Lyapunov exponent computer with GPU acceleration.
///
/// # Example
///
/// ```ignore
/// use an_tensor_compiler::topology::lyapunov::{LyapunovComputer, LyapunovConfig};
///
/// let computer = LyapunovComputer::new(Device::Cpu, LyapunovConfig::for_finance());
/// let features = computer.compute(&prices)?;
///
/// println!("Largest λ₁: {:.4}", features.mean_exponent());
/// println!("Chaos fraction: {:.2}%", features.chaos_fraction() * 100.0);
/// ```
pub struct LyapunovComputer {
    _device: Device,
    config: LyapunovConfig,
}

impl LyapunovComputer {
    /// Create a new Lyapunov computer.
    pub fn new(device: Device, config: LyapunovConfig) -> Self {
        Self {
            _device: device,
            config,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults(device: Device) -> Self {
        Self::new(device, LyapunovConfig::default())
    }

    /// Create with CPU and default config.
    pub fn cpu() -> Self {
        Self::new(Device::Cpu, LyapunovConfig::default())
    }

    /// Create optimized for financial data.
    pub fn for_finance(device: Device) -> Self {
        Self::new(device, LyapunovConfig::for_finance())
    }

    /// Get configuration.
    pub fn config(&self) -> &LyapunovConfig {
        &self.config
    }

    /// Compute Lyapunov features from price matrix.
    ///
    /// # Arguments
    ///
    /// * `data` — Price or return tensor, shape [n_symbols, n_timestamps]
    ///
    /// # Returns
    ///
    /// LyapunovFeatures with largest exponent per symbol.
    pub fn compute(&self, data: &Tensor) -> Result<LyapunovFeatures> {
        let start = std::time::Instant::now();

        let shape = data.dims();
        if shape.len() != 2 {
            return Err(TensorCoreError::Tensor(format!(
                "Data must be 2D [symbols, timestamps], got {:?}",
                shape
            )));
        }

        let n_symbols = shape[0];
        let n_timestamps = shape[1];

        // Check minimum data requirements
        let min_required = self.config.embedding_dim * self.config.time_delay
            + self.config.max_evolution
            + self.config.mean_period;

        if n_timestamps < min_required {
            return Err(TensorCoreError::Tensor(format!(
                "Need at least {} timestamps for Lyapunov, got {}",
                min_required, n_timestamps
            )));
        }

        // Convert to returns for stationarity
        let returns = self.compute_returns(data)?;
        let ret_timestamps = returns.dim(1)?;

        // Move to CPU for algorithm (involves nearest neighbor search)
        let returns_cpu = returns.to_device(&Device::Cpu)?;
        let returns_vec: Vec<f64> = returns_cpu
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|x| x as f64)
            .collect();

        let mut largest_exponents = Vec::with_capacity(n_symbols);
        let mut divergence_curves = Vec::with_capacity(n_symbols);
        let mut chaos_regimes = Vec::with_capacity(n_symbols);

        for sym in 0..n_symbols {
            let start_idx = sym * ret_timestamps;
            let end_idx = start_idx + ret_timestamps;
            let series: Vec<f64> = returns_vec[start_idx..end_idx]
                .iter()
                .map(|&x| {
                    if x.is_nan() || x.is_infinite() {
                        0.0
                    } else {
                        x
                    }
                })
                .collect();

            let (lambda, divergence) = self.rosenstein_largest(&series)?;

            largest_exponents.push(lambda);
            divergence_curves.push(divergence);
            chaos_regimes.push(ChaosRegime::from_exponent(lambda));
        }

        let elapsed = start.elapsed();

        Ok(LyapunovFeatures {
            largest_exponent: largest_exponents,
            spectrum: None,
            divergence_curves,
            chaos_regime: chaos_regimes,
            metadata: LyapunovMetadata {
                n_symbols,
                n_timestamps: ret_timestamps,
                embedding_dim: self.config.embedding_dim,
                time_delay: self.config.time_delay,
                mean_period: self.config.mean_period,
                max_evolution: self.config.max_evolution,
                compute_time_ms: elapsed.as_millis() as u64,
            },
        })
    }

    /// Compute Lyapunov for a single time series (convenience method).
    pub fn compute_single(&self, series: &[f64]) -> Result<(f64, ChaosRegime)> {
        let min_required = self.config.embedding_dim * self.config.time_delay
            + self.config.max_evolution
            + self.config.mean_period;

        if series.len() < min_required {
            return Err(TensorCoreError::Tensor(format!(
                "Need at least {} points for Lyapunov, got {}",
                min_required,
                series.len()
            )));
        }

        let (lambda, _) = self.rosenstein_largest(series)?;
        let regime = ChaosRegime::from_exponent(lambda);
        Ok((lambda, regime))
    }

    /// Rosenstein algorithm for largest Lyapunov exponent.
    ///
    /// 1. Embed time series in m-dimensional phase space
    /// 2. Find nearest neighbors (excluding temporal neighbors)
    /// 3. Track divergence over time
    /// 4. Estimate λ₁ from slope of log(divergence)
    fn rosenstein_largest(&self, series: &[f64]) -> Result<(f64, Vec<f64>)> {
        let n = series.len();
        let m = self.config.embedding_dim;
        let tau = self.config.time_delay;
        let mean_period = self.config.mean_period;
        let max_iter = self.config.max_evolution;
        let k = self.config.k_neighbors;

        // Number of embedded vectors
        let n_embedded = n.saturating_sub((m - 1) * tau);
        if n_embedded < mean_period + max_iter {
            return Err(TensorCoreError::Tensor(
                "Series too short for embedding parameters".to_string(),
            ));
        }

        // Step 1: Create delay embedding
        let embedded = self.delay_embed(series, m, tau);

        // Step 2: Find k nearest neighbors for each point
        // (excluding points within mean_period time steps)
        let neighbors = self.find_nearest_neighbors(&embedded, k, mean_period);

        // Step 3: Track divergence over time
        // d_i(j) = ||X(t_i + j) - X(t_{nn(i)} + j)||
        let max_track = max_iter.min(n_embedded.saturating_sub(1));
        let mut mean_log_divergence = vec![0.0; max_track];
        let mut counts = vec![0usize; max_track];

        for (i, neighbor_list) in neighbors.iter().enumerate() {
            for &nn_idx in neighbor_list {
                // Track divergence from step 0 to max_track
                for j in 0..max_track {
                    let future_i = i + j;
                    let future_nn = nn_idx + j;

                    if future_i < embedded.len() && future_nn < embedded.len() {
                        let dist = euclidean_distance(&embedded[future_i], &embedded[future_nn]);
                        if dist > 1e-15 {
                            mean_log_divergence[j] += dist.ln();
                            counts[j] += 1;
                        }
                    }
                }
            }
        }

        // Average the log divergences
        for j in 0..max_track {
            if counts[j] > 0 {
                mean_log_divergence[j] /= counts[j] as f64;
            }
        }

        // Step 4: Estimate λ₁ from slope using linear regression
        // on the linear region of log(d) vs t
        let lambda = self.estimate_slope(&mean_log_divergence);

        Ok((lambda, mean_log_divergence))
    }

    /// Create delay embedding of time series.
    ///
    /// X(t) = [x(t), x(t+τ), x(t+2τ), ..., x(t+(m-1)τ)]
    fn delay_embed(&self, series: &[f64], m: usize, tau: usize) -> Vec<Vec<f64>> {
        let n = series.len();
        let n_vectors = n.saturating_sub((m - 1) * tau);

        let mut embedded = Vec::with_capacity(n_vectors);
        for i in 0..n_vectors {
            let mut vec = Vec::with_capacity(m);
            for dim in 0..m {
                let idx = i + dim * tau;
                if idx < n {
                    vec.push(series[idx]);
                }
            }
            if vec.len() == m {
                embedded.push(vec);
            }
        }
        embedded
    }

    /// Find k nearest neighbors for each embedded point.
    ///
    /// Excludes neighbors within mean_period time steps (temporal neighbors).
    fn find_nearest_neighbors(
        &self,
        embedded: &[Vec<f64>],
        k: usize,
        mean_period: usize,
    ) -> Vec<Vec<usize>> {
        let n = embedded.len();
        let mut neighbors = Vec::with_capacity(n);

        for i in 0..n {
            // Compute distances to all valid neighbors
            let mut distances: Vec<(usize, f64)> = Vec::new();

            for j in 0..n {
                // Exclude temporal neighbors (within mean_period)
                let time_sep = i.abs_diff(j);
                if time_sep > mean_period {
                    let dist = euclidean_distance(&embedded[i], &embedded[j]);
                    distances.push((j, dist));
                }
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let nn: Vec<usize> = distances.iter().take(k).map(|(idx, _)| *idx).collect();
            neighbors.push(nn);
        }

        neighbors
    }

    /// Estimate slope from divergence curve using robust linear regression.
    ///
    /// Uses the linear region (early part) of log(d) vs t.
    fn estimate_slope(&self, log_divergence: &[f64]) -> f64 {
        if log_divergence.len() < 3 {
            return 0.0;
        }

        // Use first 50% of points for slope estimation (linear region)
        let n_fit = (log_divergence.len() / 2).max(3);

        // Simple linear regression: y = a + b*x
        // b = Cov(x,y) / Var(x)
        let x_mean = (n_fit as f64 - 1.0) / 2.0;
        let y_mean: f64 = log_divergence[..n_fit].iter().sum::<f64>() / n_fit as f64;

        let mut cov_xy = 0.0;
        let mut var_x = 0.0;

        for (i, &y) in log_divergence[..n_fit].iter().enumerate() {
            let x = i as f64;
            cov_xy += (x - x_mean) * (y - y_mean);
            var_x += (x - x_mean).powi(2);
        }

        if var_x < 1e-15 {
            return 0.0;
        }

        cov_xy / var_x
    }

    /// Compute log returns from prices.
    fn compute_returns(&self, prices: &Tensor) -> Result<Tensor> {
        let n_timestamps = prices.dim(1)?;

        let p_t = prices.narrow(1, 1, n_timestamps - 1)?;
        let p_t1 = prices.narrow(1, 0, n_timestamps - 1)?;

        let ratio = (&p_t / &p_t1)?;
        ratio
            .log()
            .map_err(|e| TensorCoreError::Tensor(format!("Log returns failed: {}", e)))
    }
}

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ============================================================================
// Hybrid Lyapunov Features (combining with topology)
// ============================================================================

/// Hybrid chaos features combining Lyapunov with topological invariants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridChaosFeatures {
    /// Lyapunov exponent
    pub lyapunov: f64,

    /// Betti-adjusted Lyapunov: λ₁ * (1 + β₁/10)
    pub betti_adjusted_lyapunov: f64,

    /// Chaos-rigidity ratio: λ₁ / (rigidity + ε)
    pub chaos_rigidity_ratio: f64,

    /// Composite chaos score [0, 1]
    pub chaos_score: f64,

    /// Regime classification
    pub regime: ChaosRegime,
}

impl HybridChaosFeatures {
    /// Compute hybrid features from Lyapunov and topology.
    pub fn compute(lyapunov: f64, betti_1: usize, rigidity: f64) -> Self {
        // Betti-adjusted: more loops amplify chaos detection
        let betti_adjusted = lyapunov * (1.0 + betti_1 as f64 / 10.0);

        // Chaos/rigidity ratio: high chaos + low rigidity = regime break
        let chaos_rig_ratio = lyapunov / (rigidity + 0.01);

        // Composite score: combine into [0, 1]
        // sigmoid(lyapunov * 5) gives smooth mapping
        let raw_score = lyapunov * 5.0;
        let chaos_score = 1.0 / (1.0 + (-raw_score).exp());

        let regime = ChaosRegime::from_exponent(lyapunov);

        Self {
            lyapunov,
            betti_adjusted_lyapunov: betti_adjusted,
            chaos_rigidity_ratio: chaos_rig_ratio,
            chaos_score,
            regime,
        }
    }

    /// Feature vector for ML.
    pub fn feature_vector(&self) -> Vec<f64> {
        vec![
            self.lyapunov,
            self.betti_adjusted_lyapunov,
            self.chaos_rigidity_ratio,
            self.chaos_score,
            self.regime.as_code() as f64,
        ]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate Lorenz system data (known chaotic, λ₁ ≈ 0.9).
    ///
    /// # Deprecation Notice
    ///
    /// This inline implementation is deprecated. For new code, use:
    /// ```ignore
    /// use an_tensor_compiler::dynamics::{LorenzSystem, Vec3};
    /// let lorenz = LorenzSystem::classic();
    /// let series = lorenz.trajectory_1d(Vec3::new(1.0, 1.0, 1.0), dt, n);
    /// ```
    ///
    /// The `dynamics::LorenzSystem` provides the canonical implementation with
    /// analytical Jacobians for Lyapunov spectrum computation.
    #[deprecated(
        since = "0.5.0",
        note = "Use dynamics::LorenzSystem::trajectory_1d() instead"
    )]
    #[allow(dead_code)]
    fn lorenz_series_inline(n: usize, dt: f64) -> Vec<f64> {
        let sigma = 10.0;
        let rho = 28.0;
        let beta = 8.0 / 3.0;

        let mut x = 1.0;
        let mut y = 1.0;
        let mut z = 1.0;

        let mut series = Vec::with_capacity(n);
        for _ in 0..n {
            let dx = sigma * (y - x);
            let dy = x * (rho - z) - y;
            let dz = x * y - beta * z;

            x += dx * dt;
            y += dy * dt;
            z += dz * dt;

            series.push(x);
        }
        series
    }

    /// Generate Lorenz system data with a minimal inline integrator.
    ///
    /// NOTE (public port): private tests use `dynamics::LorenzSystem`. The
    /// dynamics module is not part of the public crate, so tests integrate the
    /// classic Lorenz system (σ=10, ρ=28, β=8/3) directly with Euler steps.
    fn lorenz_series(n: usize, dt: f64) -> Vec<f64> {
        let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
        let (mut x, mut y, mut z) = (1.0, 1.0, 1.0);
        let mut series = Vec::with_capacity(n);
        for _ in 0..n {
            let dx = sigma * (y - x);
            let dy = x * (rho - z) - y;
            let dz = x * y - beta * z;
            x += dx * dt;
            y += dy * dt;
            z += dz * dt;
            series.push(x);
        }
        series
    }

    /// Generate sine wave (periodic, λ₁ ≈ 0).
    fn sine_series(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * 0.1).sin()).collect()
    }

    /// Generate random walk (λ₁ ≈ 0).
    #[allow(dead_code)]
    fn random_walk(n: usize) -> Vec<f64> {
        let mut rng_state = 42u64;
        let mut value = 0.0;
        let mut series = Vec::with_capacity(n);

        for _ in 0..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let step = ((rng_state >> 32) as f64 / u32::MAX as f64 - 0.5) * 0.1;
            value += step;
            series.push(value);
        }
        series
    }

    #[test]
    fn test_lyapunov_config_creation() {
        let config = LyapunovConfig::default();
        assert_eq!(config.embedding_dim, 10);

        let finance = LyapunovConfig::for_finance();
        assert_eq!(finance.embedding_dim, 7);
    }

    #[test]
    fn test_delay_embedding() {
        let computer = LyapunovComputer::cpu();
        let series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let embedded = computer.delay_embed(&series, 3, 1);

        assert_eq!(embedded.len(), 4); // 6 - (3-1)*1 = 4
        assert_eq!(embedded[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(embedded[1], vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_lorenz_is_chaotic() {
        let series = lorenz_series(2000, 0.01);

        let config = LyapunovConfig {
            embedding_dim: 5,
            time_delay: 10,
            mean_period: 20,
            max_evolution: 100,
            min_data_points: 100,
            k_neighbors: 1,
        };
        let computer = LyapunovComputer::new(Device::Cpu, config);

        let (lambda, _) = computer.rosenstein_largest(&series).unwrap();

        // Lorenz has positive largest exponent (chaotic)
        assert!(
            lambda > 0.0,
            "Lorenz should be chaotic, got λ₁ = {}",
            lambda
        );
    }

    #[test]
    fn test_sine_is_periodic() {
        let series = sine_series(1000);

        let config = LyapunovConfig {
            embedding_dim: 3,
            time_delay: 5,
            mean_period: 10,
            max_evolution: 50,
            min_data_points: 50,
            k_neighbors: 1,
        };
        let computer = LyapunovComputer::new(Device::Cpu, config);

        let (lambda, _) = computer.rosenstein_largest(&series).unwrap();

        // Sine wave should have λ₁ ≈ 0 (periodic/stable)
        assert!(
            lambda.abs() < 0.3,
            "Sine should be stable, got λ₁ = {}",
            lambda
        );
    }

    #[test]
    fn test_chaos_regime_classification() {
        assert_eq!(ChaosRegime::from_exponent(-0.5), ChaosRegime::Stable);
        assert_eq!(ChaosRegime::from_exponent(0.0), ChaosRegime::RandomWalk);
        assert_eq!(ChaosRegime::from_exponent(0.5), ChaosRegime::Chaotic);
    }

    #[test]
    fn test_compute_from_tensor() {
        let config = LyapunovConfig {
            embedding_dim: 3,
            time_delay: 1,
            mean_period: 5,
            max_evolution: 20,
            min_data_points: 30,
            k_neighbors: 1,
        };
        let computer = LyapunovComputer::new(Device::Cpu, config);

        // Create price-like tensor (2 symbols, 100 timestamps)
        let prices: Vec<f32> = (0..200)
            .map(|i| 100.0 + (i as f32 * 0.1).sin() * 5.0)
            .collect();
        let tensor = Tensor::from_vec(prices, (2, 100), &Device::Cpu).unwrap();

        let features = computer.compute(&tensor).unwrap();

        assert_eq!(features.largest_exponent.len(), 2);
        assert_eq!(features.chaos_regime.len(), 2);
        assert!(features.metadata.compute_time_ms < 5000);
    }

    #[test]
    fn test_lyapunov_features() {
        let features = LyapunovFeatures {
            largest_exponent: vec![0.5, -0.2, 0.3, 0.0],
            spectrum: None,
            divergence_curves: vec![],
            chaos_regime: vec![
                ChaosRegime::Chaotic,
                ChaosRegime::Stable,
                ChaosRegime::Chaotic,
                ChaosRegime::RandomWalk,
            ],
            metadata: LyapunovMetadata::default(),
        };

        assert!((features.mean_exponent() - 0.15).abs() < 0.001);
        assert!((features.max_exponent() - 0.5).abs() < 0.001);
        assert!((features.chaos_fraction() - 0.5).abs() < 0.001);

        let fv = features.feature_vector();
        assert_eq!(fv.len(), 4);
    }

    #[test]
    fn test_hybrid_chaos_features() {
        let hybrid = HybridChaosFeatures::compute(0.5, 5, 0.3);

        assert!(hybrid.betti_adjusted_lyapunov > hybrid.lyapunov);
        assert!(hybrid.chaos_score > 0.5);
        assert_eq!(hybrid.regime, ChaosRegime::Chaotic);

        let fv = hybrid.feature_vector();
        assert_eq!(fv.len(), 5);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];

        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-10);
    }
}

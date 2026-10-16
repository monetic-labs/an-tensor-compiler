//! Streaming/Online Information Theory
//!
//! Incremental updates for real-time information-theoretic computations.
//!
//! ## Overview
//!
//! Traditional information measures require full data. These streaming variants
//! maintain statistics for O(1) or O(log n) updates:
//!
//! - **StreamingEntropy**: Online entropy estimation
//! - **StreamingMI**: Online mutual information
//! - **StreamingDigest**: Summary statistics for distributions
//!
//! ## Use Cases
//!
//! - Real-time regime monitoring
//! - Low-latency trading signals
//! - Continuous drift detection
//! - Live feature importance tracking

use crate::{Result, TensorCoreError};
use candle_core::{Device, Tensor};
use std::collections::{HashMap, VecDeque};

/// Small epsilon for numerical stability
const EPSILON: f64 = 1e-10;

/// Streaming entropy estimator with sliding window
#[derive(Debug, Clone)]
pub struct StreamingEntropy {
    /// Sliding window of observations
    window: VecDeque<i64>,
    /// Maximum window size
    window_size: usize,
    /// Symbol counts in current window
    counts: HashMap<i64, usize>,
    /// Total count (should equal window.len())
    total: usize,
    /// Cached entropy value
    cached_entropy: Option<f64>,
}

impl StreamingEntropy {
    /// Create new streaming entropy estimator
    ///
    /// # Arguments
    /// - `window_size`: Size of sliding window
    pub fn new(window_size: usize) -> Self {
        StreamingEntropy {
            window: VecDeque::with_capacity(window_size),
            window_size,
            counts: HashMap::new(),
            total: 0,
            cached_entropy: None,
        }
    }

    /// Update with new observation - O(1) amortized
    ///
    /// # Arguments
    /// - `symbol`: New symbol to add
    pub fn update(&mut self, symbol: i64) {
        // If window is full, remove oldest
        if self.window.len() >= self.window_size {
            if let Some(old) = self.window.pop_front() {
                if let Some(count) = self.counts.get_mut(&old) {
                    *count -= 1;
                    if *count == 0 {
                        self.counts.remove(&old);
                    }
                }
                self.total -= 1;
            }
        }

        // Add new symbol
        self.window.push_back(symbol);
        *self.counts.entry(symbol).or_insert(0) += 1;
        self.total += 1;

        // Invalidate cache
        self.cached_entropy = None;
    }

    /// Update with batch of observations
    pub fn update_batch(&mut self, symbols: &[i64]) {
        for &s in symbols {
            self.update(s);
        }
    }

    /// Get current entropy estimate (in nats) - O(k) where k is unique symbols
    pub fn get(&mut self) -> f64 {
        if let Some(h) = self.cached_entropy {
            return h;
        }

        if self.total == 0 {
            return 0.0;
        }

        let n = self.total as f64;
        let mut entropy = 0.0;

        for &count in self.counts.values() {
            if count > 0 {
                let p = count as f64 / n;
                entropy -= p * p.ln();
            }
        }

        self.cached_entropy = Some(entropy);
        entropy
    }

    /// Get entropy in bits
    pub fn get_bits(&mut self) -> f64 {
        self.get() / 2.0_f64.ln()
    }

    /// Get normalized entropy [0, 1]
    pub fn get_normalized(&mut self) -> f64 {
        let h = self.get();
        let k = self.counts.len();
        if k <= 1 {
            return 0.0;
        }
        let h_max = (k as f64).ln();
        h / h_max.max(EPSILON)
    }

    /// Get current probability distribution
    pub fn get_distribution(&self) -> HashMap<i64, f64> {
        let n = self.total as f64;
        self.counts
            .iter()
            .map(|(&k, &v)| (k, v as f64 / n.max(EPSILON)))
            .collect()
    }

    /// Get number of unique symbols
    pub fn n_unique(&self) -> usize {
        self.counts.len()
    }

    /// Get current window fill ratio
    pub fn fill_ratio(&self) -> f64 {
        self.window.len() as f64 / self.window_size as f64
    }

    /// Reset the estimator
    pub fn reset(&mut self) {
        self.window.clear();
        self.counts.clear();
        self.total = 0;
        self.cached_entropy = None;
    }

    /// Convert current distribution to tensor
    pub fn to_tensor(&self, n_symbols: usize, device: &Device) -> Result<Tensor> {
        let mut probs = vec![0.0f32; n_symbols];
        let n = self.total as f64;

        for (&symbol, &count) in &self.counts {
            if (symbol as usize) < n_symbols {
                probs[symbol as usize] = (count as f64 / n.max(EPSILON)) as f32;
            }
        }

        Tensor::from_vec(probs, (n_symbols,), device)
            .map_err(|e| TensorCoreError::Information(e.to_string()))
    }
}

/// Streaming mutual information estimator
#[derive(Debug, Clone)]
pub struct StreamingMI {
    /// Window of (x, y) pairs
    window: VecDeque<(i64, i64)>,
    /// Maximum window size
    window_size: usize,
    /// Joint counts
    joint_counts: HashMap<(i64, i64), usize>,
    /// Marginal X counts
    x_counts: HashMap<i64, usize>,
    /// Marginal Y counts
    y_counts: HashMap<i64, usize>,
    /// Total count
    total: usize,
    /// Cached MI
    cached_mi: Option<f64>,
}

impl StreamingMI {
    /// Create new streaming MI estimator
    pub fn new(window_size: usize) -> Self {
        StreamingMI {
            window: VecDeque::with_capacity(window_size),
            window_size,
            joint_counts: HashMap::new(),
            x_counts: HashMap::new(),
            y_counts: HashMap::new(),
            total: 0,
            cached_mi: None,
        }
    }

    /// Update with new (x, y) pair - O(1) amortized
    pub fn update(&mut self, x: i64, y: i64) {
        // Remove oldest if full
        if self.window.len() >= self.window_size {
            if let Some((old_x, old_y)) = self.window.pop_front() {
                Self::decrement(&mut self.joint_counts, (old_x, old_y));
                Self::decrement_single(&mut self.x_counts, old_x);
                Self::decrement_single(&mut self.y_counts, old_y);
                self.total -= 1;
            }
        }

        // Add new pair
        self.window.push_back((x, y));
        *self.joint_counts.entry((x, y)).or_insert(0) += 1;
        *self.x_counts.entry(x).or_insert(0) += 1;
        *self.y_counts.entry(y).or_insert(0) += 1;
        self.total += 1;

        self.cached_mi = None;
    }

    fn decrement(map: &mut HashMap<(i64, i64), usize>, key: (i64, i64)) {
        if let Some(count) = map.get_mut(&key) {
            *count -= 1;
            if *count == 0 {
                map.remove(&key);
            }
        }
    }

    fn decrement_single(map: &mut HashMap<i64, usize>, key: i64) {
        if let Some(count) = map.get_mut(&key) {
            *count -= 1;
            if *count == 0 {
                map.remove(&key);
            }
        }
    }

    /// Update with batch of pairs
    pub fn update_batch(&mut self, pairs: &[(i64, i64)]) {
        for &(x, y) in pairs {
            self.update(x, y);
        }
    }

    /// Get current MI estimate (in nats) - O(|joint|)
    pub fn get(&mut self) -> f64 {
        if let Some(mi) = self.cached_mi {
            return mi;
        }

        if self.total == 0 {
            return 0.0;
        }

        let n = self.total as f64;
        let mut mi = 0.0;

        for (&(x, y), &count) in &self.joint_counts {
            let p_xy = count as f64 / n;
            let p_x = *self.x_counts.get(&x).unwrap_or(&0) as f64 / n;
            let p_y = *self.y_counts.get(&y).unwrap_or(&0) as f64 / n;

            if p_xy > EPSILON && p_x > EPSILON && p_y > EPSILON {
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }

        self.cached_mi = Some(mi.max(0.0));
        mi.max(0.0)
    }

    /// Get MI in bits
    pub fn get_bits(&mut self) -> f64 {
        self.get() / 2.0_f64.ln()
    }

    /// Get normalized MI [0, 1]
    pub fn get_normalized(&mut self) -> f64 {
        let mi = self.get();

        // Compute marginal entropies
        let n = self.total as f64;

        let h_x: f64 = self
            .x_counts
            .values()
            .map(|&c| {
                let p = c as f64 / n;
                if p > EPSILON {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();

        let h_y: f64 = self
            .y_counts
            .values()
            .map(|&c| {
                let p = c as f64 / n;
                if p > EPSILON {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();

        let h_max = h_x.min(h_y);
        if h_max < EPSILON {
            return 0.0;
        }

        (mi / h_max).min(1.0)
    }

    /// Reset the estimator
    pub fn reset(&mut self) {
        self.window.clear();
        self.joint_counts.clear();
        self.x_counts.clear();
        self.y_counts.clear();
        self.total = 0;
        self.cached_mi = None;
    }

    /// Get fill ratio
    pub fn fill_ratio(&self) -> f64 {
        self.window.len() as f64 / self.window_size as f64
    }
}

/// Streaming divergence detector for distribution drift
#[derive(Debug, Clone)]
pub struct StreamingDrift {
    /// Reference distribution (fixed)
    reference: HashMap<i64, f64>,
    /// Current window distribution
    current: StreamingEntropy,
    /// Divergence threshold for alert
    threshold: f64,
}

impl StreamingDrift {
    /// Create new drift detector
    ///
    /// # Arguments
    /// - `reference_counts`: Reference distribution counts
    /// - `window_size`: Sliding window for current distribution
    /// - `threshold`: JS divergence threshold for alerting
    pub fn new(reference_counts: &HashMap<i64, usize>, window_size: usize, threshold: f64) -> Self {
        let total: usize = reference_counts.values().sum();
        let reference: HashMap<i64, f64> = reference_counts
            .iter()
            .map(|(&k, &v)| (k, v as f64 / total as f64))
            .collect();

        StreamingDrift {
            reference,
            current: StreamingEntropy::new(window_size),
            threshold,
        }
    }

    /// Update with new observation
    pub fn update(&mut self, symbol: i64) {
        self.current.update(symbol);
    }

    /// Get current JS divergence from reference
    pub fn divergence(&self) -> f64 {
        let current_dist = self.current.get_distribution();

        // Get all symbols
        let mut all_symbols: Vec<i64> = self
            .reference
            .keys()
            .chain(current_dist.keys())
            .copied()
            .collect();
        all_symbols.sort();
        all_symbols.dedup();

        // Compute JS divergence
        let mut js = 0.0;
        for &s in &all_symbols {
            let p = *self.reference.get(&s).unwrap_or(&EPSILON);
            let q = *current_dist.get(&s).unwrap_or(&EPSILON);
            let m = (p + q) / 2.0;

            if p > EPSILON && m > EPSILON {
                js += 0.5 * p * (p / m).ln();
            }
            if q > EPSILON && m > EPSILON {
                js += 0.5 * q * (q / m).ln();
            }
        }

        js / 2.0_f64.ln() // Convert to bits
    }

    /// Check if drift detected
    pub fn is_drifting(&self) -> bool {
        self.divergence() > self.threshold
    }

    /// Get drift status with details
    pub fn status(&self) -> DriftStatus {
        let div = self.divergence();
        DriftStatus {
            divergence: div,
            threshold: self.threshold,
            is_drifting: div > self.threshold,
            fill_ratio: self.current.fill_ratio(),
        }
    }

    /// Update reference from current window
    pub fn update_reference(&mut self) {
        self.reference = self.current.get_distribution();
    }
}

/// Drift detection status
#[derive(Debug, Clone)]
pub struct DriftStatus {
    /// Current JS divergence
    pub divergence: f64,
    /// Alert threshold
    pub threshold: f64,
    /// Whether drift is detected
    pub is_drifting: bool,
    /// Window fill ratio
    pub fill_ratio: f64,
}

/// Streaming statistics for continuous values
#[derive(Debug, Clone)]
pub struct StreamingStats {
    /// Sliding window
    window: VecDeque<f64>,
    /// Window size
    window_size: usize,
    /// Running sum
    sum: f64,
    /// Running sum of squares
    sum_sq: f64,
    /// Running min
    min: f64,
    /// Running max
    max: f64,
}

impl StreamingStats {
    /// Create new streaming stats
    pub fn new(window_size: usize) -> Self {
        StreamingStats {
            window: VecDeque::with_capacity(window_size),
            window_size,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    /// Update with new value - O(1) for mean/variance, O(n) for min/max recalc
    pub fn update(&mut self, value: f64) {
        // Remove oldest if full
        if self.window.len() >= self.window_size {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }

        // Add new
        self.window.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;

        // Update min/max (lazy - only if needed)
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// Get current mean
    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        self.sum / self.window.len() as f64
    }

    /// Get current variance
    pub fn variance(&self) -> f64 {
        if self.window.len() < 2 {
            return 0.0;
        }
        let n = self.window.len() as f64;
        let mean = self.sum / n;
        (self.sum_sq / n - mean * mean).max(0.0)
    }

    /// Get current standard deviation
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get current min (may need recalculation)
    pub fn min(&mut self) -> f64 {
        // Recalculate if window changed and old min was removed
        if self.window.is_empty() {
            return 0.0;
        }
        if !self.window.contains(&self.min) {
            self.min = self.window.iter().cloned().fold(f64::MAX, f64::min);
        }
        self.min
    }

    /// Get current max (may need recalculation)
    pub fn max(&mut self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        if !self.window.contains(&self.max) {
            self.max = self.window.iter().cloned().fold(f64::MIN, f64::max);
        }
        self.max
    }

    /// Get estimated SNR (mean² / variance)
    pub fn snr(&self) -> f64 {
        let var = self.variance();
        if var < EPSILON {
            return f64::MAX;
        }
        self.mean().powi(2) / var
    }

    /// Get z-score for a value
    pub fn z_score(&self, value: f64) -> f64 {
        let std = self.std();
        if std < EPSILON {
            return 0.0;
        }
        (value - self.mean()) / std
    }

    /// Reset stats
    pub fn reset(&mut self) {
        self.window.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.min = f64::MAX;
        self.max = f64::MIN;
    }

    /// Get fill ratio
    pub fn fill_ratio(&self) -> f64 {
        self.window.len() as f64 / self.window_size as f64
    }
}

/// Streaming surprise/novelty detector
#[derive(Debug, Clone)]
pub struct StreamingNovelty {
    /// Entropy tracker
    entropy: StreamingEntropy,
    /// Stats tracker for surprise values
    surprise_stats: StreamingStats,
    /// Number of symbols
    n_symbols: usize,
    /// Adaptation rate
    alpha: f64,
    /// Symbol probabilities (smoothed)
    probs: Vec<f64>,
}

impl StreamingNovelty {
    /// Create new novelty detector
    ///
    /// # Arguments
    /// - `n_symbols`: Number of possible symbols
    /// - `window_size`: Entropy window size
    /// - `alpha`: Probability adaptation rate (0-1)
    pub fn new(n_symbols: usize, window_size: usize, alpha: f64) -> Self {
        StreamingNovelty {
            entropy: StreamingEntropy::new(window_size),
            surprise_stats: StreamingStats::new(window_size),
            n_symbols,
            alpha: alpha.clamp(0.0, 1.0),
            probs: vec![1.0 / n_symbols as f64; n_symbols],
        }
    }

    /// Update with new symbol, return surprise value
    pub fn update(&mut self, symbol: i64) -> f64 {
        let idx = symbol as usize;

        // Get surprise before updating
        let surprise = if idx < self.n_symbols && self.probs[idx] > EPSILON {
            -self.probs[idx].log2()
        } else {
            (self.n_symbols as f64).log2() // Max surprise
        };

        // Update probability estimates
        for i in 0..self.n_symbols {
            if i == idx {
                self.probs[i] = (1.0 - self.alpha) * self.probs[i] + self.alpha;
            } else {
                self.probs[i] *= 1.0 - self.alpha;
            }
        }

        // Normalize
        let sum: f64 = self.probs.iter().sum();
        self.probs.iter_mut().for_each(|p| *p /= sum);

        // Update trackers
        self.entropy.update(symbol);
        self.surprise_stats.update(surprise);

        surprise
    }

    /// Get current entropy
    pub fn entropy(&mut self) -> f64 {
        self.entropy.get()
    }

    /// Get entropy in bits
    pub fn entropy_bits(&mut self) -> f64 {
        self.entropy.get_bits()
    }

    /// Get z-score of last surprise (how unusual was it?)
    pub fn novelty_z_score(&self, surprise: f64) -> f64 {
        let mean = self.surprise_stats.mean();
        let std = self.surprise_stats.std();
        if std < EPSILON {
            return 0.0;
        }
        (surprise - mean) / std
    }

    /// Check if symbol is anomalous (z-score > threshold)
    pub fn is_anomaly(&self, surprise: f64, z_threshold: f64) -> bool {
        self.novelty_z_score(surprise) > z_threshold
    }

    /// Get current probability estimates
    pub fn probabilities(&self) -> &[f64] {
        &self.probs
    }

    /// Reset detector
    pub fn reset(&mut self) {
        self.entropy.reset();
        self.surprise_stats.reset();
        self.probs = vec![1.0 / self.n_symbols as f64; self.n_symbols];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_entropy_basic() {
        let mut se = StreamingEntropy::new(100);

        // Uniform symbols
        for i in 0..100 {
            se.update(i % 4);
        }

        let h = se.get_bits();
        // Should be close to log2(4) = 2 bits
        assert!(h > 1.5 && h < 2.5);
    }

    #[test]
    fn test_streaming_entropy_sliding() {
        let mut se = StreamingEntropy::new(10);

        // Fill with all 0s
        for _ in 0..10 {
            se.update(0);
        }

        let h1 = se.get();
        assert!(h1 < 0.01); // Zero entropy for constant

        // Now add some 1s
        for _ in 0..5 {
            se.update(1);
        }

        let h2 = se.get();
        assert!(h2 > h1); // Entropy should increase
    }

    #[test]
    fn test_streaming_entropy_normalized() {
        let mut se = StreamingEntropy::new(100);

        // Add 4 symbols equally
        for i in 0..100 {
            se.update(i % 4);
        }

        let h_norm = se.get_normalized();
        assert!(h_norm > 0.9 && h_norm <= 1.0);
    }

    #[test]
    fn test_streaming_mi_basic() {
        let mut smi = StreamingMI::new(100);

        // Perfect correlation: x == y
        for i in 0..100 {
            smi.update(i % 4, i % 4);
        }

        let mi = smi.get();
        assert!(mi > 1.0); // High MI

        let nmi = smi.get_normalized();
        assert!(nmi > 0.9); // Close to 1
    }

    #[test]
    fn test_streaming_mi_independent() {
        let mut smi = StreamingMI::new(200);

        // Pseudo-independent: x and y cycle differently
        for i in 0..200 {
            smi.update(i % 4, (i * 7) % 5);
        }

        let mi = smi.get();
        // Should be low (not exactly 0 due to finite sample)
        assert!(mi < 0.5);
    }

    #[test]
    fn test_streaming_drift_detection() {
        let mut reference = HashMap::new();
        reference.insert(0, 50);
        reference.insert(1, 50);

        let mut drift = StreamingDrift::new(&reference, 100, 0.1);

        // Add samples matching reference
        for i in 0..100 {
            drift.update(i % 2);
        }

        assert!(!drift.is_drifting());

        // Now add skewed data
        for _ in 0..100 {
            drift.update(0); // All 0s
        }

        assert!(drift.is_drifting());
    }

    #[test]
    fn test_streaming_stats_basic() {
        let mut ss = StreamingStats::new(100);

        for i in 0..100 {
            ss.update(i as f64);
        }

        let mean = ss.mean();
        let var = ss.variance();

        // Mean of 0-99 is 49.5
        assert!((mean - 49.5).abs() < 0.1);

        // Variance should be positive
        assert!(var > 0.0);
    }

    #[test]
    fn test_streaming_stats_sliding() {
        let mut ss = StreamingStats::new(10);

        for i in 0..10 {
            ss.update(i as f64);
        }

        let mean1 = ss.mean();

        // Add more values that shift the window
        for i in 10..20 {
            ss.update(i as f64);
        }

        let mean2 = ss.mean();

        // Mean should shift higher
        assert!(mean2 > mean1);
    }

    #[test]
    fn test_streaming_novelty_basic() {
        let mut sn = StreamingNovelty::new(4, 100, 0.1);

        // Add mostly 0s
        for _ in 0..50 {
            sn.update(0);
        }

        // Add a rare symbol - should be surprising
        let surprise = sn.update(3);
        assert!(surprise > 1.0); // High surprise

        // Common symbol - low surprise
        let surprise2 = sn.update(0);
        assert!(surprise2 < surprise);
    }

    #[test]
    fn test_streaming_novelty_anomaly() {
        let mut sn = StreamingNovelty::new(4, 100, 0.1);

        // Establish baseline
        for i in 0..100 {
            sn.update(i % 2); // Just 0s and 1s
        }

        // Rare symbol
        let surprise = sn.update(3);
        let is_anom = sn.is_anomaly(surprise, 2.0);

        // Should be flagged as anomaly
        assert!(is_anom || surprise > 3.0);
    }

    #[test]
    fn test_streaming_entropy_reset() {
        let mut se = StreamingEntropy::new(50);

        for i in 0..50 {
            se.update(i % 4);
        }

        assert!(se.n_unique() > 0);

        se.reset();

        assert_eq!(se.n_unique(), 0);
        assert_eq!(se.fill_ratio(), 0.0);
    }

    #[test]
    fn test_streaming_mi_reset() {
        let mut smi = StreamingMI::new(50);

        for i in 0..50 {
            smi.update(i % 4, i % 4);
        }

        let mi1 = smi.get();
        assert!(mi1 > 0.0);

        smi.reset();

        assert_eq!(smi.fill_ratio(), 0.0);
    }

    #[test]
    fn test_drift_status() {
        let mut reference = HashMap::new();
        reference.insert(0, 50);
        reference.insert(1, 50);

        let mut drift = StreamingDrift::new(&reference, 50, 0.2);

        for i in 0..50 {
            drift.update(i % 2);
        }

        let status = drift.status();

        assert!(status.fill_ratio > 0.9);
        assert!(status.threshold == 0.2);
    }

    #[test]
    fn test_to_tensor() -> Result<()> {
        let device = Device::Cpu;
        let mut se = StreamingEntropy::new(50);

        for i in 0..50 {
            se.update(i % 3);
        }

        let tensor = se.to_tensor(4, &device)?;

        assert_eq!(tensor.dims(), &[4]);

        // Sum should be ~1 (distribution)
        let sum: f32 = tensor.sum_all()?.to_scalar()?;
        assert!((sum - 1.0).abs() < 0.01);

        Ok(())
    }
}

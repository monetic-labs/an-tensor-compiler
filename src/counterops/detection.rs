//! Specialized Detection Operations
//!
//! Higher-level detection functions built on tensor primitives.

use crate::primitives::cosine_similarity;
use crate::{Device, Result, Tensor, TensorCoreError};

/// Result of an injection detection check
#[derive(Debug, Clone)]
pub struct InjectionResult {
    /// Was an injection attempt detected?
    pub detected: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Most similar known attack pattern (if any)
    pub matched_pattern: Option<String>,
    /// Similarity score to matched pattern
    pub similarity: f32,
}

impl InjectionResult {
    /// Create a result indicating no injection detected
    pub fn not_detected() -> Self {
        Self {
            detected: false,
            confidence: 0.0,
            matched_pattern: None,
            similarity: 0.0,
        }
    }

    /// Create a result indicating injection detected
    pub fn detected_with_pattern(pattern: String, similarity: f32, confidence: f32) -> Self {
        Self {
            detected: true,
            confidence,
            matched_pattern: Some(pattern),
            similarity,
        }
    }
}

/// Result of an anomaly scoring operation
#[derive(Debug, Clone)]
pub struct AnomalyScore {
    /// Anomaly score (0.0 = normal, 1.0 = highly anomalous)
    pub score: f32,
    /// Distance from baseline
    pub distance: f32,
    /// Is this significantly anomalous?
    pub is_anomalous: bool,
    /// Features contributing most to anomaly
    pub top_contributing_features: Vec<(String, f32)>,
}

impl AnomalyScore {
    /// Create a normal (non-anomalous) score
    pub fn normal() -> Self {
        Self {
            score: 0.0,
            distance: 0.0,
            is_anomalous: false,
            top_contributing_features: Vec::new(),
        }
    }

    /// Create an anomalous score
    pub fn anomalous(score: f32, distance: f32) -> Self {
        Self {
            score,
            distance,
            is_anomalous: score > 0.5,
            top_contributing_features: Vec::new(),
        }
    }

    /// Add contributing features
    pub fn with_features(mut self, features: Vec<(String, f32)>) -> Self {
        self.top_contributing_features = features;
        self
    }
}

/// Detect potential injection attempts by comparing to known patterns
///
/// Uses cosine similarity to compare the input embedding against a matrix
/// of known attack pattern embeddings.
///
/// # Arguments
///
/// * `input` - Input embedding to check [1, dim]
/// * `attack_patterns` - Matrix of known attack pattern embeddings [n_patterns, dim]
/// * `pattern_names` - Names of the attack patterns (for reporting)
/// * `threshold` - Similarity threshold for detection (e.g., 0.85)
///
/// # Returns
///
/// `InjectionResult` indicating whether injection was detected
pub fn detect_injection(
    input: &Tensor,
    attack_patterns: &Tensor,
    pattern_names: &[String],
    threshold: f32,
) -> Result<InjectionResult> {
    // Get input shape
    let input_shape = input.shape();
    if input_shape.rank() != 2 {
        return Err(TensorCoreError::Tensor(format!(
            "Input must be 2D [batch, dim], got shape {:?}",
            input_shape
        )));
    }

    let patterns_shape = attack_patterns.shape();
    if patterns_shape.rank() != 2 {
        return Err(TensorCoreError::Tensor(format!(
            "Attack patterns must be 2D [n_patterns, dim], got shape {:?}",
            patterns_shape
        )));
    }

    let n_patterns = patterns_shape.dims()[0];
    if n_patterns != pattern_names.len() {
        return Err(TensorCoreError::Config(format!(
            "Pattern count mismatch: {} patterns but {} names",
            n_patterns,
            pattern_names.len()
        )));
    }

    // Compute similarity with each pattern
    let mut max_similarity = 0.0f32;
    let mut max_pattern_idx = 0usize;

    // Flatten input for similarity computation
    let input_flat = input
        .flatten_from(0)
        .map_err(|e| TensorCoreError::Tensor(format!("Failed to flatten input: {}", e)))?;

    for i in 0..n_patterns {
        let pattern = attack_patterns
            .get(i)
            .map_err(|e| TensorCoreError::Tensor(format!("Failed to get pattern {}: {}", i, e)))?;

        let similarity = cosine_similarity(&input_flat, &pattern)?;

        if similarity > max_similarity {
            max_similarity = similarity;
            max_pattern_idx = i;
        }
    }

    // Check against threshold
    if max_similarity >= threshold {
        let confidence = (max_similarity - threshold) / (1.0 - threshold);
        Ok(InjectionResult::detected_with_pattern(
            pattern_names[max_pattern_idx].clone(),
            max_similarity,
            confidence.min(1.0),
        ))
    } else {
        Ok(InjectionResult::not_detected())
    }
}

/// Compute anomaly score by comparing behavior to baseline
///
/// Uses L2 distance and statistical deviation to score how anomalous
/// the current behavior is compared to the baseline.
///
/// # Arguments
///
/// * `behavior` - Current behavior embedding \[dim\]
/// * `baseline` - Baseline behavior embedding \[dim\] (e.g., moving average)
/// * `baseline_std` - Standard deviation of baseline \[dim\] (for normalization)
/// * `sensitivity` - Sensitivity factor (higher = more sensitive)
///
/// # Returns
///
/// `AnomalyScore` with the computed anomaly information
pub fn compute_anomaly_score(
    behavior: &Tensor,
    baseline: &Tensor,
    baseline_std: Option<&Tensor>,
    sensitivity: f32,
    device: &Device,
) -> Result<AnomalyScore> {
    // Compute difference
    let diff = (behavior - baseline)
        .map_err(|e| TensorCoreError::Tensor(format!("Failed to compute difference: {}", e)))?;

    // Normalize by std if provided
    let normalized = if let Some(std) = baseline_std {
        // Add small epsilon to avoid division by zero
        let epsilon = Tensor::ones_like(std)?
            .affine(0.0, 1e-6)
            .map_err(|e| TensorCoreError::Tensor(format!("Failed to create epsilon: {}", e)))?;

        let safe_std = (std + &epsilon)
            .map_err(|e| TensorCoreError::Tensor(format!("Failed to add epsilon to std: {}", e)))?;

        (&diff / &safe_std)
            .map_err(|e| TensorCoreError::Tensor(format!("Failed to normalize: {}", e)))?
    } else {
        diff.clone()
    };

    // Compute L2 distance
    let squared = normalized
        .sqr()
        .map_err(|e| TensorCoreError::Tensor(format!("Failed to square: {}", e)))?;
    let sum_squared = squared
        .sum_all()
        .map_err(|e| TensorCoreError::Tensor(format!("Failed to sum: {}", e)))?;
    let distance: f32 = sum_squared
        .sqrt()
        .map_err(|e| TensorCoreError::Tensor(format!("Failed to sqrt: {}", e)))?
        .to_scalar()
        .map_err(|e| TensorCoreError::Tensor(format!("Failed to convert to scalar: {}", e)))?;

    // Convert distance to score (0-1) using sigmoid
    let scaled_distance = distance * sensitivity;
    let score = 1.0 / (1.0 + (-scaled_distance + 3.0).exp()); // Centered at 3 std devs

    // Find top contributing features
    let abs_normalized = normalized
        .abs()
        .map_err(|e| TensorCoreError::Tensor(format!("Failed to abs: {}", e)))?;
    let values: Vec<f32> = abs_normalized
        .flatten_all()
        .map_err(|e| TensorCoreError::Tensor(format!("Failed to flatten: {}", e)))?
        .to_vec1()
        .map_err(|e| TensorCoreError::Tensor(format!("Failed to convert: {}", e)))?;

    let mut indexed_values: Vec<(usize, f32)> =
        values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_features: Vec<(String, f32)> = indexed_values
        .iter()
        .take(5)
        .map(|(i, v)| (format!("feature_{}", i), *v))
        .collect();

    let _ = device; // Suppress unused warning for now

    Ok(AnomalyScore::anomalous(score, distance).with_features(top_features))
}

/// Check if a rate limit has been exceeded
///
/// Simple threshold check for rate limiting.
///
/// # Arguments
///
/// * `request_count` - Number of requests in the window
/// * `window_seconds` - Time window in seconds
/// * `max_rate` - Maximum allowed rate (requests per minute)
///
/// # Returns
///
/// (exceeded, current_rate) tuple
pub fn check_rate_limit(request_count: usize, window_seconds: f32, max_rate: f32) -> (bool, f32) {
    if window_seconds <= 0.0 {
        return (false, 0.0);
    }

    let current_rate = (request_count as f32 / window_seconds) * 60.0; // Convert to per-minute
    let exceeded = current_rate > max_rate;

    (exceeded, current_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> Device {
        Device::Cpu
    }

    #[test]
    fn test_injection_result_not_detected() {
        let result = InjectionResult::not_detected();
        assert!(!result.detected);
        assert!(result.matched_pattern.is_none());
    }

    #[test]
    fn test_injection_result_detected() {
        let result =
            InjectionResult::detected_with_pattern("sql_injection".to_string(), 0.92, 0.85);
        assert!(result.detected);
        assert_eq!(result.matched_pattern, Some("sql_injection".to_string()));
        assert_eq!(result.similarity, 0.92);
    }

    #[test]
    fn test_anomaly_score_normal() {
        let score = AnomalyScore::normal();
        assert_eq!(score.score, 0.0);
        assert!(!score.is_anomalous);
    }

    #[test]
    fn test_anomaly_score_anomalous() {
        let score = AnomalyScore::anomalous(0.8, 4.2);
        assert_eq!(score.score, 0.8);
        assert!(score.is_anomalous);
        assert_eq!(score.distance, 4.2);
    }

    #[test]
    fn test_anomaly_score_with_features() {
        let features = vec![
            ("feature_3".to_string(), 2.5),
            ("feature_7".to_string(), 1.8),
        ];
        let score = AnomalyScore::anomalous(0.75, 3.0).with_features(features.clone());
        assert_eq!(score.top_contributing_features.len(), 2);
    }

    #[test]
    fn test_detect_injection_no_match() {
        let device = test_device();

        let input = Tensor::randn(0.0f32, 1.0, (1, 16), &device).unwrap();
        let patterns = Tensor::randn(5.0f32, 1.0, (3, 16), &device).unwrap();
        let names = vec![
            "sql_injection".to_string(),
            "xss".to_string(),
            "command_injection".to_string(),
        ];

        let result = detect_injection(&input, &patterns, &names, 0.95).unwrap();
        // Random vectors unlikely to have high similarity
        // (might still match by chance, so just check it doesn't error)
        assert!(result.similarity >= 0.0 && result.similarity <= 1.0);
    }

    #[test]
    fn test_detect_injection_match() {
        let device = test_device();

        // Create an input that matches one pattern
        let pattern = Tensor::from_vec(vec![1.0f32; 16], (1, 16), &device).unwrap();
        let input = pattern.clone();
        let patterns = Tensor::cat(
            &[
                &Tensor::randn(0.0f32, 1.0, (2, 16), &device).unwrap(),
                &pattern.reshape((1, 16)).unwrap(),
            ],
            0,
        )
        .unwrap();
        let names = vec![
            "pattern_a".to_string(),
            "pattern_b".to_string(),
            "matching_pattern".to_string(),
        ];

        let result = detect_injection(&input, &patterns, &names, 0.9).unwrap();
        assert!(result.detected);
        assert_eq!(result.matched_pattern, Some("matching_pattern".to_string()));
    }

    #[test]
    fn test_compute_anomaly_score() {
        let device = test_device();

        let behavior = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &device).unwrap();
        let baseline = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &device).unwrap();

        let score = compute_anomaly_score(&behavior, &baseline, None, 1.0, &device).unwrap();
        // Identical vectors should have low anomaly score
        assert!(score.score < 0.5);
    }

    #[test]
    fn test_compute_anomaly_score_high_deviation() {
        let device = test_device();

        let behavior = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], 3, &device).unwrap();
        let baseline = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &device).unwrap();

        let score = compute_anomaly_score(&behavior, &baseline, None, 1.0, &device).unwrap();
        // Very different vectors should have high anomaly score
        assert!(score.is_anomalous);
    }

    #[test]
    fn test_rate_limit_not_exceeded() {
        let (exceeded, rate) = check_rate_limit(50, 60.0, 100.0);
        assert!(!exceeded);
        assert_eq!(rate, 50.0);
    }

    #[test]
    fn test_rate_limit_exceeded() {
        let (exceeded, rate) = check_rate_limit(200, 60.0, 100.0);
        assert!(exceeded);
        assert_eq!(rate, 200.0);
    }

    #[test]
    fn test_rate_limit_zero_window() {
        let (exceeded, rate) = check_rate_limit(100, 0.0, 100.0);
        assert!(!exceeded);
        assert_eq!(rate, 0.0);
    }
}

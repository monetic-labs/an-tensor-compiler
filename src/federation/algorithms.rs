//! Federation Merge Algorithms
//!
//! Pure functions for merging parameter tensors across namespaces.
//! These algorithms don't depend on networking - they're pure tensor operations.
//!
//! ## Algorithms
//!
//! | Algorithm | Description | When to Use |
//! |-----------|-------------|-------------|
//! | `weighted_average` | Average weighted by example count | Default, most stable |
//! | `fedavg` | FedAvg algorithm (McMahan et al. 2017) | Standard FL approach |
//! | `median` | Coordinate-wise median | Robust to outliers |
//! | `trimmed_mean` | Remove extremes, then average | Byzantine-tolerant |

use candle_core::{DType, Tensor};
use std::collections::HashMap;

use crate::namespace::NamespaceId;
use crate::{Result, TensorCoreError};

// =============================================================================
// PARTICIPANT UPDATE
// =============================================================================

/// A parameter update from a single namespace/participant
#[derive(Debug, Clone)]
pub struct ParticipantUpdate {
    /// Source namespace
    pub namespace: NamespaceId,
    
    /// Parameter tensor
    pub params: Tensor,
    
    /// Number of training examples this update is based on
    pub example_count: usize,
    
    /// Accuracy/loss on validation data (optional, for weighting)
    pub accuracy: Option<f32>,
}

impl ParticipantUpdate {
    /// Create a new participant update
    pub fn new(namespace: NamespaceId, params: Tensor, example_count: usize) -> Self {
        Self {
            namespace,
            params,
            example_count,
            accuracy: None,
        }
    }

    /// Create with accuracy
    pub fn with_accuracy(mut self, accuracy: f32) -> Self {
        self.accuracy = Some(accuracy);
        self
    }
}

// =============================================================================
// MERGE RESULT
// =============================================================================

/// Result of a merge operation
#[derive(Debug)]
pub struct MergeResult {
    /// Merged parameter tensor
    pub merged_params: Tensor,
    
    /// Per-namespace contribution weights (normalized)
    pub contribution_weights: HashMap<NamespaceId, f32>,
    
    /// Total examples used
    pub total_examples: usize,
    
    /// Number of participants
    pub participant_count: usize,
}

// =============================================================================
// WEIGHTED AVERAGE
// =============================================================================

/// Weighted average merge: params_merged = Σ(wᵢ × paramsᵢ)
///
/// Weight is proportional to example count:
///   wᵢ = example_countᵢ / Σ example_counts
///
/// # Arguments
/// * `updates` - Parameter updates from each namespace
///
/// # Example
/// ```ignore
/// let merged = weighted_average(&[
///     ParticipantUpdate::new(TRADING, trading_params, 1000),
///     ParticipantUpdate::new(PIPELINE, pipeline_params, 500),
/// ])?;
/// // Trading contributes 2/3, Pipeline contributes 1/3
/// ```
pub fn weighted_average(updates: &[ParticipantUpdate]) -> Result<MergeResult> {
    if updates.is_empty() {
        return Err(TensorCoreError::Federation(
            "weighted_average requires at least one update".into(),
        ));
    }

    // Calculate total examples
    let total_examples: usize = updates.iter().map(|u| u.example_count).sum();
    if total_examples == 0 {
        return Err(TensorCoreError::Federation(
            "Total example count is zero".into(),
        ));
    }

    // Calculate weights
    let mut contribution_weights = HashMap::new();
    for update in updates {
        let weight = update.example_count as f32 / total_examples as f32;
        contribution_weights.insert(update.namespace, weight);
    }

    // Compute weighted sum
    let device = updates[0].params.device();
    let shape = updates[0].params.dims();
    
    let mut merged = Tensor::zeros(shape, DType::F32, device)
        .map_err(|e| TensorCoreError::Tensor(format!("zeros failed: {}", e)))?;

    for update in updates {
        let weight = contribution_weights[&update.namespace] as f64;
        let weighted = (&update.params * weight)
            .map_err(|e| TensorCoreError::Tensor(format!("scale failed: {}", e)))?;
        merged = (&merged + &weighted)
            .map_err(|e| TensorCoreError::Tensor(format!("add failed: {}", e)))?;
    }

    Ok(MergeResult {
        merged_params: merged,
        contribution_weights,
        total_examples,
        participant_count: updates.len(),
    })
}

// =============================================================================
// FEDAVG (Federated Averaging)
// =============================================================================

/// FedAvg: Federated Averaging (McMahan et al. 2017)
///
/// The canonical federated learning algorithm:
/// 1. Each participant trains locally for E epochs
/// 2. Send model updates (delta from global model)
/// 3. Server averages updates weighted by data quantity
/// 4. Broadcast new global model
///
/// This function implements step 3 - the server-side averaging.
///
/// # Difference from weighted_average
/// 
/// FedAvg expects **deltas** (change from previous global model),
/// while weighted_average works on full parameters.
///
/// # Arguments
/// * `global_params` - Current global model parameters
/// * `updates` - Deltas from each participant (params - global)
///
/// # Example
/// ```ignore
/// // Each participant computes: delta = local_params - global_params
/// let merged = fedavg(&global_params, &[
///     ParticipantUpdate::new(TRADING, trading_delta, 1000),
///     ParticipantUpdate::new(PIPELINE, pipeline_delta, 500),
/// ])?;
/// // merged = global + weighted_average(deltas)
/// ```
pub fn fedavg(global_params: &Tensor, updates: &[ParticipantUpdate]) -> Result<MergeResult> {
    if updates.is_empty() {
        return Err(TensorCoreError::Federation(
            "fedavg requires at least one update".into(),
        ));
    }

    // First, compute weighted average of deltas
    let delta_result = weighted_average(updates)?;

    // Apply averaged delta to global params
    let merged_params = (global_params + &delta_result.merged_params)
        .map_err(|e| TensorCoreError::Tensor(format!("add global failed: {}", e)))?;

    Ok(MergeResult {
        merged_params,
        contribution_weights: delta_result.contribution_weights,
        total_examples: delta_result.total_examples,
        participant_count: delta_result.participant_count,
    })
}

// =============================================================================
// COORDINATE-WISE MEDIAN
// =============================================================================

/// Coordinate-wise median: for each parameter, take the median across participants
///
/// More robust to outliers than averaging. Good when some namespaces
/// might have corrupted or adversarial updates.
///
/// # Note
/// Requires at least 3 participants to be meaningful.
pub fn median(updates: &[ParticipantUpdate]) -> Result<MergeResult> {
    if updates.is_empty() {
        return Err(TensorCoreError::Federation(
            "median requires at least one update".into(),
        ));
    }

    if updates.len() == 1 {
        // Single update - just return it
        return Ok(MergeResult {
            merged_params: updates[0].params.clone(),
            contribution_weights: [(updates[0].namespace, 1.0)].into_iter().collect(),
            total_examples: updates[0].example_count,
            participant_count: 1,
        });
    }

    let shape = updates[0].params.dims();
    let n = updates.len();

    // For coordinate-wise median, we need to find the median value at each position
    // We'll do this by extracting values, sorting, and taking the middle
    
    // Flatten all params
    let flat_params: Vec<Vec<f32>> = updates
        .iter()
        .map(|u| {
            u.params
                .flatten_all()
                .map_err(|e| TensorCoreError::Tensor(format!("flatten failed: {}", e)))?
                .to_vec1::<f32>()
                .map_err(|e| TensorCoreError::Tensor(format!("to_vec failed: {}", e)))
        })
        .collect::<Result<Vec<_>>>()?;

    let n_params = flat_params[0].len();
    let mut median_values = Vec::with_capacity(n_params);

    // For each parameter position, find the median across all participants
    for i in 0..n_params {
        let mut values: Vec<f32> = flat_params.iter().map(|p| p[i]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let median_val = if n % 2 == 1 {
            values[n / 2]
        } else {
            (values[n / 2 - 1] + values[n / 2]) / 2.0
        };
        median_values.push(median_val);
    }

    let device = updates[0].params.device();
    let merged_flat = Tensor::from_vec(median_values, n_params, device)
        .map_err(|e| TensorCoreError::Tensor(format!("from_vec failed: {}", e)))?;
    
    let merged_params = merged_flat
        .reshape(shape)
        .map_err(|e| TensorCoreError::Tensor(format!("reshape failed: {}", e)))?;

    // Equal contribution for median
    let weight = 1.0 / updates.len() as f32;
    let contribution_weights: HashMap<NamespaceId, f32> = updates
        .iter()
        .map(|u| (u.namespace, weight))
        .collect();

    let total_examples: usize = updates.iter().map(|u| u.example_count).sum();

    Ok(MergeResult {
        merged_params,
        contribution_weights,
        total_examples,
        participant_count: updates.len(),
    })
}

// =============================================================================
// TRIMMED MEAN
// =============================================================================

/// Trimmed mean: remove top and bottom k% before averaging
///
/// Byzantine-tolerant: can handle up to k% malicious participants.
///
/// # Arguments
/// * `updates` - Parameter updates from each namespace
/// * `trim_fraction` - Fraction to trim from each end (e.g., 0.1 = remove 10% highest and lowest)
///
/// # Example
/// ```ignore
/// let merged = trimmed_mean(&updates, 0.1)?; // Trim 10% from each end
/// ```
pub fn trimmed_mean(updates: &[ParticipantUpdate], trim_fraction: f32) -> Result<MergeResult> {
    if updates.is_empty() {
        return Err(TensorCoreError::Federation(
            "trimmed_mean requires at least one update".into(),
        ));
    }

    if trim_fraction < 0.0 || trim_fraction >= 0.5 {
        return Err(TensorCoreError::Federation(
            "trim_fraction must be in [0, 0.5)".into(),
        ));
    }

    let n = updates.len();
    let trim_count = (n as f32 * trim_fraction).floor() as usize;
    
    // If we'd trim everything, just do regular average
    if n <= 2 * trim_count {
        return weighted_average(updates);
    }

    let shape = updates[0].params.dims();
    let keep_count = n - 2 * trim_count;

    // Flatten all params
    let flat_params: Vec<Vec<f32>> = updates
        .iter()
        .map(|u| {
            u.params
                .flatten_all()
                .map_err(|e| TensorCoreError::Tensor(format!("flatten failed: {}", e)))?
                .to_vec1::<f32>()
                .map_err(|e| TensorCoreError::Tensor(format!("to_vec failed: {}", e)))
        })
        .collect::<Result<Vec<_>>>()?;

    let n_params = flat_params[0].len();
    let mut trimmed_mean_values = Vec::with_capacity(n_params);

    // For each parameter position, compute trimmed mean
    for i in 0..n_params {
        let mut values: Vec<f32> = flat_params.iter().map(|p| p[i]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take middle portion and average
        let trimmed: Vec<f32> = values[trim_count..(n - trim_count)].to_vec();
        let mean_val: f32 = trimmed.iter().sum::<f32>() / trimmed.len() as f32;
        trimmed_mean_values.push(mean_val);
    }

    let device = updates[0].params.device();
    let merged_flat = Tensor::from_vec(trimmed_mean_values, n_params, device)
        .map_err(|e| TensorCoreError::Tensor(format!("from_vec failed: {}", e)))?;
    
    let merged_params = merged_flat
        .reshape(shape)
        .map_err(|e| TensorCoreError::Tensor(format!("reshape failed: {}", e)))?;

    // Weights are less meaningful for trimmed mean
    let weight = 1.0 / keep_count as f32;
    let contribution_weights: HashMap<NamespaceId, f32> = updates
        .iter()
        .map(|u| (u.namespace, weight))
        .collect();

    let total_examples: usize = updates.iter().map(|u| u.example_count).sum();

    Ok(MergeResult {
        merged_params,
        contribution_weights,
        total_examples,
        participant_count: updates.len(),
    })
}

// =============================================================================
// ACCURACY-WEIGHTED AVERAGE
// =============================================================================

/// Average weighted by validation accuracy instead of example count
///
/// Gives more weight to namespaces that perform better on held-out data.
///
/// # Arguments
/// * `updates` - Updates with accuracy field set
pub fn accuracy_weighted(updates: &[ParticipantUpdate]) -> Result<MergeResult> {
    if updates.is_empty() {
        return Err(TensorCoreError::Federation(
            "accuracy_weighted requires at least one update".into(),
        ));
    }

    // Check that all updates have accuracy
    for update in updates {
        if update.accuracy.is_none() {
            return Err(TensorCoreError::Federation(format!(
                "Namespace {} missing accuracy for accuracy_weighted merge",
                update.namespace
            )));
        }
    }

    // Use accuracy as weight (higher accuracy = more weight)
    let total_accuracy: f32 = updates.iter().map(|u| u.accuracy.unwrap()).sum();
    if total_accuracy <= 0.0 {
        return Err(TensorCoreError::Federation(
            "Total accuracy is zero or negative".into(),
        ));
    }

    let mut contribution_weights = HashMap::new();
    for update in updates {
        let weight = update.accuracy.unwrap() / total_accuracy;
        contribution_weights.insert(update.namespace, weight);
    }

    let device = updates[0].params.device();
    let shape = updates[0].params.dims();

    let mut merged = Tensor::zeros(shape, DType::F32, device)
        .map_err(|e| TensorCoreError::Tensor(format!("zeros failed: {}", e)))?;

    for update in updates {
        let weight = contribution_weights[&update.namespace] as f64;
        let weighted = (&update.params * weight)
            .map_err(|e| TensorCoreError::Tensor(format!("scale failed: {}", e)))?;
        merged = (&merged + &weighted)
            .map_err(|e| TensorCoreError::Tensor(format!("add failed: {}", e)))?;
    }

    let total_examples: usize = updates.iter().map(|u| u.example_count).sum();

    Ok(MergeResult {
        merged_params: merged,
        contribution_weights,
        total_examples,
        participant_count: updates.len(),
    })
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::namespace::{TRADING, PIPELINE, CHAT};
    use candle_core::Device;

    fn device() -> Device {
        Device::Cpu
    }

    fn tensor(vals: &[f32]) -> Tensor {
        Tensor::from_vec(vals.to_vec(), vals.len(), &device()).unwrap()
    }

    #[test]
    fn test_weighted_average_equal_weights() {
        let updates = vec![
            ParticipantUpdate::new(TRADING, tensor(&[1.0, 2.0, 3.0]), 100),
            ParticipantUpdate::new(PIPELINE, tensor(&[3.0, 4.0, 5.0]), 100),
        ];

        let result = weighted_average(&updates).unwrap();
        let vals = result.merged_params.to_vec1::<f32>().unwrap();

        // Equal weights: (1+3)/2=2, (2+4)/2=3, (3+5)/2=4
        assert!((vals[0] - 2.0).abs() < 0.001);
        assert!((vals[1] - 3.0).abs() < 0.001);
        assert!((vals[2] - 4.0).abs() < 0.001);

        assert_eq!(result.contribution_weights[&TRADING], 0.5);
        assert_eq!(result.contribution_weights[&PIPELINE], 0.5);
    }

    #[test]
    fn test_weighted_average_unequal_weights() {
        let updates = vec![
            ParticipantUpdate::new(TRADING, tensor(&[1.0, 2.0]), 300), // 75%
            ParticipantUpdate::new(PIPELINE, tensor(&[5.0, 6.0]), 100), // 25%
        ];

        let result = weighted_average(&updates).unwrap();
        let vals = result.merged_params.to_vec1::<f32>().unwrap();

        // 0.75 * 1 + 0.25 * 5 = 2.0
        // 0.75 * 2 + 0.25 * 6 = 3.0
        assert!((vals[0] - 2.0).abs() < 0.001);
        assert!((vals[1] - 3.0).abs() < 0.001);

        assert!((result.contribution_weights[&TRADING] - 0.75).abs() < 0.001);
        assert!((result.contribution_weights[&PIPELINE] - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_fedavg() {
        let global = tensor(&[10.0, 20.0]);
        let updates = vec![
            ParticipantUpdate::new(TRADING, tensor(&[1.0, 2.0]), 100), // delta
            ParticipantUpdate::new(PIPELINE, tensor(&[3.0, 4.0]), 100), // delta
        ];

        let result = fedavg(&global, &updates).unwrap();
        let vals = result.merged_params.to_vec1::<f32>().unwrap();

        // Average delta: (1+3)/2=2, (2+4)/2=3
        // Merged: 10+2=12, 20+3=23
        assert!((vals[0] - 12.0).abs() < 0.001);
        assert!((vals[1] - 23.0).abs() < 0.001);
    }

    #[test]
    fn test_median_odd() {
        let updates = vec![
            ParticipantUpdate::new(TRADING, tensor(&[1.0, 10.0]), 100),
            ParticipantUpdate::new(PIPELINE, tensor(&[5.0, 50.0]), 100),
            ParticipantUpdate::new(CHAT, tensor(&[3.0, 30.0]), 100),
        ];

        let result = median(&updates).unwrap();
        let vals = result.merged_params.to_vec1::<f32>().unwrap();

        // Sorted: [1,3,5] and [10,30,50], median is middle: 3, 30
        assert!((vals[0] - 3.0).abs() < 0.001);
        assert!((vals[1] - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_trimmed_mean() {
        // 5 participants, trim 20% = trim 1 from each end
        let updates = vec![
            ParticipantUpdate::new(0, tensor(&[1.0]), 100),  // min - trimmed
            ParticipantUpdate::new(1, tensor(&[2.0]), 100),
            ParticipantUpdate::new(2, tensor(&[3.0]), 100),
            ParticipantUpdate::new(3, tensor(&[4.0]), 100),
            ParticipantUpdate::new(4, tensor(&[100.0]), 100), // max - trimmed
        ];

        let result = trimmed_mean(&updates, 0.2).unwrap();
        let vals = result.merged_params.to_vec1::<f32>().unwrap();

        // After trimming 1 and 100: mean of [2,3,4] = 3
        assert!((vals[0] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_accuracy_weighted() {
        let updates = vec![
            ParticipantUpdate::new(TRADING, tensor(&[1.0, 2.0]), 100)
                .with_accuracy(0.9), // High accuracy
            ParticipantUpdate::new(PIPELINE, tensor(&[5.0, 6.0]), 100)
                .with_accuracy(0.1), // Low accuracy
        ];

        let result = accuracy_weighted(&updates).unwrap();
        let vals = result.merged_params.to_vec1::<f32>().unwrap();

        // 0.9/1.0 * [1,2] + 0.1/1.0 * [5,6] = [0.9*1+0.1*5, 0.9*2+0.1*6]
        // = [0.9+0.5, 1.8+0.6] = [1.4, 2.4]
        assert!((vals[0] - 1.4).abs() < 0.001);
        assert!((vals[1] - 2.4).abs() < 0.001);
    }

    #[test]
    fn test_weighted_average_single() {
        let updates = vec![
            ParticipantUpdate::new(TRADING, tensor(&[1.0, 2.0, 3.0]), 100),
        ];

        let result = weighted_average(&updates).unwrap();
        let vals = result.merged_params.to_vec1::<f32>().unwrap();

        assert!((vals[0] - 1.0).abs() < 0.001);
        assert!((vals[1] - 2.0).abs() < 0.001);
        assert!((vals[2] - 3.0).abs() < 0.001);
        assert_eq!(result.contribution_weights[&TRADING], 1.0);
    }

    #[test]
    fn test_weighted_average_empty() {
        let updates: Vec<ParticipantUpdate> = vec![];
        assert!(weighted_average(&updates).is_err());
    }

    #[test]
    fn test_three_namespace_federation() {
        let updates = vec![
            ParticipantUpdate::new(TRADING, tensor(&[1.0]), 500),   // 50%
            ParticipantUpdate::new(PIPELINE, tensor(&[2.0]), 300),  // 30%
            ParticipantUpdate::new(CHAT, tensor(&[3.0]), 200),      // 20%
        ];

        let result = weighted_average(&updates).unwrap();
        let vals = result.merged_params.to_vec1::<f32>().unwrap();

        // 0.5*1 + 0.3*2 + 0.2*3 = 0.5 + 0.6 + 0.6 = 1.7
        assert!((vals[0] - 1.7).abs() < 0.001);
        assert_eq!(result.participant_count, 3);
        assert_eq!(result.total_examples, 1000);
    }
}


//! Federation Layer
//!
//! Cross-domain predicate synchronization and parameter sharing.
//!
//! ## Overview
//!
//! Federation enables learning across namespace boundaries while
//! maintaining gradient isolation. Key capabilities:
//!
//! - **Predicate Sync**: Share learned concept embeddings
//! - **Replay Merge**: Combine experience buffers across domains
//! - **Weighted Fusion**: Parameter averaging with domain weights
//!
//! ## Architecture Split
//!
//! **This module provides** (owned by an-tensor-core):
//! - Types: `FederationConfig`, `MergeStrategy`, `FederatedPredicate`
//! - Algorithms: `weighted_average()`, `fedavg()` merge functions
//! - Traits: `SyncTransport` trait for transport abstraction
//!
//! **Transport implementations live elsewhere** (an-integration / DevOps):
//! - gRPC transport implementation
//! - Redis pub/sub transport implementation
//! - Scheduling, orchestration, monitoring
//!
//! ## Example
//!
//! ```ignore
//! use an_tensor_compiler::federation::*;
//!
//! // Core provides types and algorithms
//! let config = FederationConfig {
//!     shared_predicates: vec!["regime_change".into()],
//!     merge_strategy: MergeStrategy::WeightedAverage,
//!     ..Default::default()
//! };
//!
//! // Use merge algorithms directly
//! let merged = weighted_average(&[
//!     ParticipantUpdate::new(TRADING, trading_params, 1000),
//!     ParticipantUpdate::new(PIPELINE, pipeline_params, 500),
//! ])?;
//! ```
//!
//! ## Status
//!
//! | Component | Status | Owner |
//! |-----------|--------|-------|
//! | Types (FederationConfig, etc.) | ✅ Complete | an-tensor-core |
//! | Merge algorithms | ✅ Complete | an-tensor-core |
//! | SyncTransport trait | ✅ Complete | an-tensor-core |
//! | Transport impls | 📋 Week 6 | an-integration / DevOps |
//! | Orchestration | 📋 Week 6 | DevOps |

pub mod algorithms;

// Re-export algorithm types and functions
pub use algorithms::{
    weighted_average, fedavg, median, trimmed_mean, accuracy_weighted,
    ParticipantUpdate, MergeResult,
};

use crate::namespace::NamespaceId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Configuration for federation between namespaces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationConfig {
    /// Which predicates to share across namespaces
    pub shared_predicates: Vec<String>,

    /// How to merge predicate updates
    pub merge_strategy: MergeStrategy,

    /// How often to sync (0 = manual only)
    #[serde(with = "humantime_serde")]
    pub sync_interval: Duration,

    /// Minimum confidence before sharing a predicate
    pub min_confidence: f32,

    /// Maximum staleness before considering a predicate outdated
    #[serde(with = "humantime_serde")]
    pub max_staleness: Duration,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            shared_predicates: Vec::new(),
            merge_strategy: MergeStrategy::WeightedAverage,
            sync_interval: Duration::from_secs(3600), // 1 hour
            min_confidence: 0.7,
            max_staleness: Duration::from_secs(86400), // 24 hours
        }
    }
}

/// How to merge parameter updates from multiple namespaces
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Weighted average by outcome count
    /// weight_ns = outcomes_ns / total_outcomes
    WeightedAverage,

    /// Take most recent update (last writer wins)
    LastWriter,

    /// Keep all versions, vote at inference time
    Ensemble,

    /// Federated averaging (FedAvg algorithm)
    FedAvg,
}

/// Statistics about a federated predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedPredicateStats {
    /// Predicate name
    pub name: String,

    /// Per-namespace statistics
    pub namespace_stats: HashMap<NamespaceId, NamespacePredicateStats>,

    /// Last sync time
    pub last_sync: Option<chrono::DateTime<chrono::Utc>>,

    /// Current merge version
    pub version: u64,
}

/// Per-namespace statistics for a predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespacePredicateStats {
    /// Number of training examples
    pub example_count: usize,

    /// Average confidence on held-out data
    pub accuracy: f32,

    /// Last update time
    pub last_update: chrono::DateTime<chrono::Utc>,

    /// Weight in federation (computed from example_count)
    pub federation_weight: f32,
}

/// A federated predicate that can be shared across namespaces
#[derive(Debug)]
pub struct FederatedPredicate {
    /// Predicate name
    pub name: String,

    /// Merged parameters (shared across all namespaces)
    merged_params: Option<candle_core::Tensor>,

    /// Per-namespace parameters (before merge)
    namespace_params: HashMap<NamespaceId, candle_core::Tensor>,

    /// Per-namespace example counts
    namespace_examples: HashMap<NamespaceId, usize>,

    /// Per-namespace accuracy scores
    namespace_accuracy: HashMap<NamespaceId, f32>,

    /// Contribution weights by namespace (after last merge)
    contribution_weights: HashMap<NamespaceId, f32>,

    /// Statistics
    stats: FederatedPredicateStats,
}

impl FederatedPredicate {
    /// Create a new federated predicate
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            name: name.clone(),
            merged_params: None,
            namespace_params: HashMap::new(),
            namespace_examples: HashMap::new(),
            namespace_accuracy: HashMap::new(),
            contribution_weights: HashMap::new(),
            stats: FederatedPredicateStats {
                name,
                namespace_stats: HashMap::new(),
                last_sync: None,
                version: 0,
            },
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &FederatedPredicateStats {
        &self.stats
    }

    /// Get contribution weights from last merge
    pub fn contribution_weights(&self) -> &HashMap<NamespaceId, f32> {
        &self.contribution_weights
    }

    /// Get merged parameters
    pub fn merged_params(&self) -> Option<&candle_core::Tensor> {
        self.merged_params.as_ref()
    }

    /// Register parameters from a namespace
    pub fn register_params(
        &mut self,
        namespace: NamespaceId,
        params: candle_core::Tensor,
        example_count: usize,
        accuracy: Option<f32>,
    ) {
        self.namespace_params.insert(namespace, params);
        self.namespace_examples.insert(namespace, example_count);
        if let Some(acc) = accuracy {
            self.namespace_accuracy.insert(namespace, acc);
        }
    }

    /// Merge parameters from all registered namespaces
    pub fn merge(&mut self, strategy: MergeStrategy) -> crate::Result<MergeResult> {
        if self.namespace_params.is_empty() {
            return Err(crate::TensorCoreError::Federation(
                "No namespace params registered".into(),
            ));
        }

        // Build participant updates
        let updates: Vec<ParticipantUpdate> = self
            .namespace_params
            .iter()
            .map(|(&ns, params)| {
                let mut update = ParticipantUpdate::new(
                    ns,
                    params.clone(),
                    *self.namespace_examples.get(&ns).unwrap_or(&1),
                );
                if let Some(&acc) = self.namespace_accuracy.get(&ns) {
                    update = update.with_accuracy(acc);
                }
                update
            })
            .collect();

        // Apply merge strategy
        let result = match strategy {
            MergeStrategy::WeightedAverage => weighted_average(&updates)?,
            MergeStrategy::FedAvg => {
                // For FedAvg without a global, treat as weighted average
                weighted_average(&updates)?
            }
            MergeStrategy::LastWriter => {
                // Take most recent (just use first for now - should track timestamps)
                weighted_average(&[updates.into_iter().next().unwrap()])?
            }
            MergeStrategy::Ensemble => {
                // For ensemble, we'd keep all - for now, just average
                weighted_average(&updates)?
            }
        };

        // Store merged result
        self.merged_params = Some(result.merged_params.clone());
        self.contribution_weights = result.contribution_weights.clone();
        self.stats.last_sync = Some(chrono::Utc::now());
        self.stats.version += 1;

        // Update namespace stats
        for (ns, &weight) in &result.contribution_weights {
            self.stats.namespace_stats.insert(
                *ns,
                NamespacePredicateStats {
                    example_count: *self.namespace_examples.get(ns).unwrap_or(&0),
                    accuracy: *self.namespace_accuracy.get(ns).unwrap_or(&0.0),
                    last_update: chrono::Utc::now(),
                    federation_weight: weight,
                },
            );
        }

        Ok(result)
    }
}

// =============================================================================
// TRANSPORT TRAIT (Implementation lives in an-integration / DevOps)
// =============================================================================

/// Trait for federation transport implementations
///
/// **This trait is defined here but implemented elsewhere.**
///
/// Implementations might include:
/// - `GrpcTransport` in an-integration
/// - `RedisTransport` in an-integration
/// - `LocalTransport` for testing
///
/// # Example Implementation
///
/// ```ignore
/// // In an-integration/src/tensor_sync/grpc.rs
/// use an_tensor_compiler::federation::SyncTransport;
///
/// pub struct GrpcTransport {
///     endpoint: String,
///     client: TensorSyncClient,
/// }
///
/// #[async_trait]
/// impl SyncTransport for GrpcTransport {
///     async fn push_update(&self, update: PredicateUpdate) -> Result<()> {
///         self.client.push(update.into()).await?;
///         Ok(())
///     }
///
///     async fn pull_updates(&self, since: DateTime<Utc>) -> Result<Vec<PredicateUpdate>> {
///         let response = self.client.pull(since).await?;
///         Ok(response.updates.into_iter().map(Into::into).collect())
///     }
/// }
/// ```
#[async_trait::async_trait]
pub trait SyncTransport: Send + Sync {
    /// Push a predicate update to the sync layer
    async fn push_update(&self, update: PredicateUpdate) -> anyhow::Result<()>;

    /// Pull predicate updates since a given timestamp
    async fn pull_updates(
        &self,
        since: chrono::DateTime<chrono::Utc>,
    ) -> anyhow::Result<Vec<PredicateUpdate>>;

    /// Check if the transport is healthy
    async fn health_check(&self) -> anyhow::Result<bool>;
}

/// A predicate update for synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateUpdate {
    /// Predicate name
    pub predicate_name: String,

    /// Source namespace
    pub source_namespace: NamespaceId,

    /// Serialized embedding (safetensors format)
    pub embedding_bytes: Vec<u8>,

    /// Number of training examples this update is based on
    pub example_count: usize,

    /// Accuracy on held-out data
    pub accuracy: f32,

    /// When this update was created
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Version number
    pub version: u64,
}

/// No-op transport for testing and local-only mode
pub struct LocalTransport;

#[async_trait::async_trait]
impl SyncTransport for LocalTransport {
    async fn push_update(&self, _update: PredicateUpdate) -> anyhow::Result<()> {
        // No-op: local mode doesn't sync
        Ok(())
    }

    async fn pull_updates(
        &self,
        _since: chrono::DateTime<chrono::Utc>,
    ) -> anyhow::Result<Vec<PredicateUpdate>> {
        // No-op: local mode has no remote updates
        Ok(Vec::new())
    }

    async fn health_check(&self) -> anyhow::Result<bool> {
        Ok(true)
    }
}

// =============================================================================
// FEDERATION MANAGER
// =============================================================================

/// Federation manager for coordinating cross-namespace learning
#[derive(Debug)]
pub struct FederationManager {
    /// Configuration
    config: FederationConfig,

    /// Managed predicates
    predicates: HashMap<String, FederatedPredicate>,
}

impl FederationManager {
    /// Create a new federation manager
    pub fn new(config: FederationConfig) -> Self {
        Self {
            config,
            predicates: HashMap::new(),
        }
    }

    /// Register a predicate for federation
    pub fn register_predicate(&mut self, name: impl Into<String>) {
        let name = name.into();
        if !self.predicates.contains_key(&name) {
            self.predicates.insert(name.clone(), FederatedPredicate::new(&name));
        }
    }

    /// Get a federated predicate
    pub fn get_predicate(&self, name: &str) -> Option<&FederatedPredicate> {
        self.predicates.get(name)
    }

    /// Get a mutable federated predicate
    pub fn get_predicate_mut(&mut self, name: &str) -> Option<&mut FederatedPredicate> {
        self.predicates.get_mut(name)
    }

    /// Register parameters from a namespace for a predicate
    pub fn register_namespace_params(
        &mut self,
        predicate_name: &str,
        namespace: NamespaceId,
        params: candle_core::Tensor,
        example_count: usize,
        accuracy: Option<f32>,
    ) -> crate::Result<()> {
        let predicate = self.predicates.get_mut(predicate_name).ok_or_else(|| {
            crate::TensorCoreError::Federation(format!(
                "Predicate '{}' not registered",
                predicate_name
            ))
        })?;

        predicate.register_params(namespace, params, example_count, accuracy);
        Ok(())
    }

    /// Sync a single predicate using the configured merge strategy
    pub fn sync_predicate(&mut self, name: &str) -> crate::Result<PredicateSyncStatus> {
        let strategy = self.config.merge_strategy;
        let min_confidence = self.config.min_confidence;

        let predicate = self.predicates.get_mut(name).ok_or_else(|| {
            crate::TensorCoreError::Federation(format!("Predicate '{}' not registered", name))
        })?;

        // Check if we have enough data
        if predicate.namespace_params.is_empty() {
            return Ok(PredicateSyncStatus::Skipped {
                reason: "No namespace params registered".into(),
            });
        }

        // Check minimum confidence (if accuracy data available)
        let has_low_confidence = predicate
            .namespace_accuracy
            .values()
            .any(|&acc| acc < min_confidence);

        if has_low_confidence && !predicate.namespace_accuracy.is_empty() {
            return Ok(PredicateSyncStatus::Skipped {
                reason: format!("Some namespaces below min_confidence {}", min_confidence),
            });
        }

        // Perform merge
        match predicate.merge(strategy) {
            Ok(result) => Ok(PredicateSyncStatus::Success {
                contributors: result.contribution_weights.keys().cloned().collect(),
                new_version: predicate.stats.version,
            }),
            Err(e) => Ok(PredicateSyncStatus::Failed {
                error: e.to_string(),
            }),
        }
    }

    /// Sync all registered predicates
    pub fn sync_all_predicates(&mut self) -> crate::Result<SyncReport> {
        let start = std::time::Instant::now();
        let mut predicate_status = HashMap::new();
        let mut predicates_synced = 0;

        let predicate_names: Vec<String> = self.predicates.keys().cloned().collect();

        for name in predicate_names {
            let status = self.sync_predicate(&name)?;
            if matches!(status, PredicateSyncStatus::Success { .. }) {
                predicates_synced += 1;
            }
            predicate_status.insert(name, status);
        }

        Ok(SyncReport {
            predicates_synced,
            predicate_status,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Sync all predicates (async version with transport)
    pub async fn sync_all(&mut self) -> crate::Result<SyncReport> {
        // For local-only mode, just call the sync function directly
        self.sync_all_predicates()
    }

    /// Sync with a specific transport (for remote federation)
    pub async fn sync_with_transport(
        &mut self,
        transport: &dyn SyncTransport,
    ) -> crate::Result<SyncReport> {
        // Check transport health
        let healthy = transport
            .health_check()
            .await
            .map_err(|e| crate::TensorCoreError::Federation(format!("Transport error: {}", e)))?;

        if !healthy {
            return Err(crate::TensorCoreError::Federation(
                "Transport health check failed".into(),
            ));
        }

        // For now, just do local sync
        // Full implementation would:
        // 1. Pull remote updates via transport
        // 2. Merge with local params
        // 3. Push merged result via transport
        self.sync_all_predicates()
    }

    /// Get configuration
    pub fn config(&self) -> &FederationConfig {
        &self.config
    }

    /// Get all predicate names
    pub fn predicate_names(&self) -> Vec<String> {
        self.predicates.keys().cloned().collect()
    }
}

/// Report from a federation sync operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncReport {
    /// Number of predicates synced
    pub predicates_synced: usize,

    /// Per-predicate sync status
    pub predicate_status: HashMap<String, PredicateSyncStatus>,

    /// Total time taken
    pub duration_ms: u64,
}

/// Status of syncing a single predicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredicateSyncStatus {
    /// Successfully synced with merged parameters
    Success {
        /// Namespaces that contributed to the merge
        contributors: Vec<NamespaceId>,
        /// New version number after merge
        new_version: u64,
    },

    /// Skipped due to insufficient data or confidence
    Skipped {
        /// Reason the sync was skipped
        reason: String,
    },

    /// Sync failed with an error
    Failed {
        /// Error description
        error: String,
    },
}

// For serde with Duration
mod humantime_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error> {
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::namespace::{TRADING, PIPELINE};
    use candle_core::{Device, Tensor};

    fn device() -> Device {
        Device::Cpu
    }

    fn tensor(vals: &[f32]) -> Tensor {
        Tensor::from_vec(vals.to_vec(), vals.len(), &device()).unwrap()
    }

    #[test]
    fn test_federation_config_default() {
        let config = FederationConfig::default();
        assert!(config.shared_predicates.is_empty());
        assert_eq!(config.merge_strategy, MergeStrategy::WeightedAverage);
        assert_eq!(config.sync_interval.as_secs(), 3600);
    }

    #[test]
    fn test_federation_manager() {
        let config = FederationConfig {
            shared_predicates: vec!["regime_change".into()],
            ..Default::default()
        };

        let mut manager = FederationManager::new(config);
        manager.register_predicate("regime_change");

        assert!(manager.get_predicate("regime_change").is_some());
        assert!(manager.get_predicate("nonexistent").is_none());
    }

    #[test]
    fn test_federated_predicate_merge() {
        let mut pred = FederatedPredicate::new("test_predicate");

        // Register params from two namespaces
        pred.register_params(TRADING, tensor(&[1.0, 2.0, 3.0]), 1000, Some(0.9));
        pred.register_params(PIPELINE, tensor(&[3.0, 4.0, 5.0]), 1000, Some(0.8));

        // Merge with weighted average
        let result = pred.merge(MergeStrategy::WeightedAverage).unwrap();

        // Equal weights (same example count): (1+3)/2=2, (2+4)/2=3, (3+5)/2=4
        let vals = result.merged_params.to_vec1::<f32>().unwrap();
        assert!((vals[0] - 2.0).abs() < 0.001);
        assert!((vals[1] - 3.0).abs() < 0.001);
        assert!((vals[2] - 4.0).abs() < 0.001);

        // Check merged params are stored
        assert!(pred.merged_params().is_some());
        assert_eq!(pred.stats().version, 1);
    }

    #[test]
    fn test_federation_manager_full_sync() {
        let config = FederationConfig {
            shared_predicates: vec!["regime_change".into()],
            merge_strategy: MergeStrategy::WeightedAverage,
            min_confidence: 0.5,
            ..Default::default()
        };

        let mut manager = FederationManager::new(config);
        manager.register_predicate("regime_change");

        // Register params from namespaces
        manager
            .register_namespace_params(
                "regime_change",
                TRADING,
                tensor(&[1.0, 2.0]),
                500,
                Some(0.9),
            )
            .unwrap();
        manager
            .register_namespace_params(
                "regime_change",
                PIPELINE,
                tensor(&[3.0, 4.0]),
                500,
                Some(0.8),
            )
            .unwrap();

        // Sync all
        let report = manager.sync_all_predicates().unwrap();
        assert_eq!(report.predicates_synced, 1);

        // Check the predicate was synced
        let pred = manager.get_predicate("regime_change").unwrap();
        let merged = pred.merged_params().unwrap().to_vec1::<f32>().unwrap();
        assert!((merged[0] - 2.0).abs() < 0.001);
        assert!((merged[1] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_sync_skipped_no_params() {
        let config = FederationConfig::default();
        let mut manager = FederationManager::new(config);
        manager.register_predicate("empty_pred");

        let status = manager.sync_predicate("empty_pred").unwrap();
        assert!(matches!(status, PredicateSyncStatus::Skipped { .. }));
    }

    #[test]
    fn test_sync_skipped_low_confidence() {
        let config = FederationConfig {
            min_confidence: 0.9, // High threshold
            ..Default::default()
        };
        let mut manager = FederationManager::new(config);
        manager.register_predicate("low_conf");

        manager
            .register_namespace_params(
                "low_conf",
                TRADING,
                tensor(&[1.0]),
                100,
                Some(0.5), // Below threshold
            )
            .unwrap();

        let status = manager.sync_predicate("low_conf").unwrap();
        assert!(matches!(status, PredicateSyncStatus::Skipped { .. }));
    }

    #[test]
    fn test_weighted_average_integration() {
        let updates = vec![
            ParticipantUpdate::new(TRADING, tensor(&[2.0, 4.0]), 300),
            ParticipantUpdate::new(PIPELINE, tensor(&[4.0, 8.0]), 100),
        ];

        let result = weighted_average(&updates).unwrap();
        let vals = result.merged_params.to_vec1::<f32>().unwrap();

        // Trading: 75%, Pipeline: 25%
        // 0.75*2 + 0.25*4 = 2.5
        // 0.75*4 + 0.25*8 = 5.0
        assert!((vals[0] - 2.5).abs() < 0.001);
        assert!((vals[1] - 5.0).abs() < 0.001);
    }
}


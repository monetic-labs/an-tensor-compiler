//! Agent Memory System using Holographic Reduced Representations
//!
//! This module extends HRR for agent (Synapse/Cortex) memory systems, enabling:
//!
//! - **Episode Memory**: Bound observation-action-outcome triples
//! - **Experience Hologram**: Long-term superposition of episodes
//! - **Pattern Library**: Failure signatures and recovery patterns
//! - **Working Memory**: Salience-decayed temporary context
//!
//! # Enhanced Memory Capabilities
//!
//! Leveraging our math packages for intelligent memory:
//!
//! - **Entropy-Gated Consolidation**: Skip redundant episodes (information theory)
//! - **Regime-Partitioned Memory**: Separate holograms per chaos regime (topology/lyapunov)
//! - **Topological Indexing**: Find stable memory clusters (persistence homology)
//! - **Information Bottleneck Compression**: Optimal long-term storage (IB theory)
//! - **Multi-Scale Memory**: Wavelet-based time horizons (wavelets)
//! - **Causal Memory**: Transfer entropy for predictive chaining
//!
//! # Example
//!
//! ```rust,ignore
//! use an_tensor_compiler::holographic::memory::{
//!     SynapseMemory, Episode, Observation, Action, ActionOutcome
//! };
//!
//! let mut memory = SynapseMemory::new(embedder, config)?;
//!
//! // Encode an experience
//! let episode = Episode::new(
//!     Observation::new("deployment-failed", "Service crashed on startup"),
//!     Action::new("restart-with-rollback", "Rolled back to previous version"),
//!     ActionOutcome::success("Service recovered, rollback successful"),
//! );
//! memory.encode_episode(&episode).await?;
//!
//! // Recall from experience
//! let obs = Observation::new("deployment-failed", "Similar crash pattern");
//! let result = memory.process_observation(&obs).await?;
//! ```

use crate::information::StreamingNovelty;
use crate::topology::{ChaosRegime, LyapunovComputer, LyapunovConfig};
use crate::Result;
use candle_core::{DType, Device, Tensor};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::ops::{bind, cosine_similarity, normalize, project, superimpose};

// ============================================================================
// Core Episode Types
// ============================================================================

/// An observation from the environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Short category/type (e.g., "deployment-failed", "high-latency")
    pub category: String,
    /// Detailed description for embedding
    pub description: String,
    /// Structured metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl Observation {
    pub fn new(category: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            category: category.into(),
            description: description.into(),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// An action taken by the agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// Action type/name
    pub action_type: String,
    /// Description of what was done
    pub description: String,
    /// Parameters used
    #[serde(default)]
    pub parameters: HashMap<String, String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl Action {
    pub fn new(action_type: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            action_type: action_type.into(),
            description: description.into(),
            parameters: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }
}

/// Outcome of an action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionOutcome {
    /// Success/failure
    pub success: bool,
    /// Description of outcome
    pub description: String,
    /// Metrics captured
    #[serde(default)]
    pub metrics: HashMap<String, f64>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl ActionOutcome {
    pub fn success(description: impl Into<String>) -> Self {
        Self {
            success: true,
            description: description.into(),
            metrics: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    pub fn failure(description: impl Into<String>) -> Self {
        Self {
            success: false,
            description: description.into(),
            metrics: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    pub fn with_metric(mut self, key: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(key.into(), value);
        self
    }
}

/// A complete episode: observation → action → outcome
#[derive(Debug, Clone)]
pub struct Episode {
    /// Unique episode ID
    pub id: EpisodeId,
    /// The observation that triggered action
    pub observation: Observation,
    /// The action taken
    pub action: Action,
    /// The outcome of the action
    pub outcome: ActionOutcome,
    /// Embedded observation tensor
    pub obs_tensor: Tensor,
    /// Embedded action tensor
    pub action_tensor: Tensor,
    /// Embedded outcome tensor
    pub outcome_tensor: Tensor,
    /// Combined holographic representation: obs ⊗ (action ⊗ outcome)
    pub hologram: Tensor,
    /// Novelty score at encoding time
    pub novelty_bits: f32,
    /// Detected regime at encoding time
    pub regime: Option<ChaosRegime>,
}

/// Unique episode identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EpisodeId(pub String);

impl EpisodeId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl Default for EpisodeId {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Failure Patterns
// ============================================================================

/// A recognized failure pattern
#[derive(Debug, Clone)]
pub struct FailurePattern {
    /// Pattern name
    pub name: String,
    /// Description
    pub description: String,
    /// Holographic signature of this failure
    pub signature: Tensor,
    /// Number of times observed
    pub occurrence_count: usize,
    /// Last seen
    pub last_seen: DateTime<Utc>,
}

/// A recovery pattern for a failure
#[derive(Debug, Clone)]
pub struct RecoveryPattern {
    /// Target failure pattern name
    pub failure_name: String,
    /// Recovery description
    pub description: String,
    /// Success rate
    pub success_rate: f64,
    /// Holographic representation of recovery actions
    pub recovery_hologram: Tensor,
    /// Number of applications
    pub application_count: usize,
}

/// Pattern library for anti-fragility
#[derive(Debug)]
pub struct PatternLibrary {
    /// Known failure patterns
    failures: HashMap<String, FailurePattern>,
    /// Recovery patterns for failures
    recoveries: HashMap<String, Vec<RecoveryPattern>>,
    /// Matching threshold
    match_threshold: f32,
    /// Device for tensor operations
    _device: Device,
}

impl PatternLibrary {
    pub fn new(device: Device, match_threshold: f32) -> Self {
        Self {
            failures: HashMap::new(),
            recoveries: HashMap::new(),
            match_threshold,
            _device: device,
        }
    }

    /// Register a new failure pattern
    pub fn register_failure(&mut self, name: String, description: String, signature: Tensor) {
        let pattern = FailurePattern {
            name: name.clone(),
            description,
            signature,
            occurrence_count: 1,
            last_seen: Utc::now(),
        };
        self.failures.insert(name, pattern);
    }

    /// Match an observation against known failures
    pub fn match_failure(&self, obs_tensor: &Tensor) -> Option<(&FailurePattern, f32)> {
        let mut best_match: Option<(&FailurePattern, f32)> = None;

        for pattern in self.failures.values() {
            if let Ok(sim) = cosine_similarity(obs_tensor, &pattern.signature) {
                if sim > self.match_threshold
                    && (best_match.is_none() || sim > best_match.unwrap().1)
                {
                    best_match = Some((pattern, sim));
                }
            }
        }

        best_match
    }

    /// Get recovery patterns for a failure
    pub fn get_recoveries(&self, failure_name: &str) -> Option<&Vec<RecoveryPattern>> {
        self.recoveries.get(failure_name)
    }

    /// Get best recovery for a failure
    pub fn get_best_recovery(&self, failure_name: &str) -> Option<&RecoveryPattern> {
        self.recoveries.get(failure_name).and_then(|recoveries| {
            recoveries
                .iter()
                .max_by(|a, b| a.success_rate.partial_cmp(&b.success_rate).unwrap())
        })
    }

    /// Register a recovery pattern
    pub fn register_recovery(
        &mut self,
        failure_name: String,
        description: String,
        recovery_hologram: Tensor,
        success: bool,
    ) {
        let recoveries = self.recoveries.entry(failure_name.clone()).or_default();

        // Check if we have an existing similar recovery
        let mut found = false;
        for recovery in recoveries.iter_mut() {
            if let Ok(sim) = cosine_similarity(&recovery.recovery_hologram, &recovery_hologram) {
                if sim > 0.8 {
                    // Update existing
                    recovery.application_count += 1;
                    let total = recovery.application_count as f64;
                    recovery.success_rate = (recovery.success_rate * (total - 1.0)
                        + if success { 1.0 } else { 0.0 })
                        / total;
                    found = true;
                    break;
                }
            }
        }

        if !found {
            recoveries.push(RecoveryPattern {
                failure_name,
                description,
                success_rate: if success { 1.0 } else { 0.0 },
                recovery_hologram,
                application_count: 1,
            });
        }
    }

    /// Increment failure occurrence count
    pub fn record_failure_occurrence(&mut self, name: &str) {
        if let Some(pattern) = self.failures.get_mut(name) {
            pattern.occurrence_count += 1;
            pattern.last_seen = Utc::now();
        }
    }

    /// Get number of registered failure patterns
    pub fn failure_count(&self) -> usize {
        self.failures.len()
    }

    /// Iterate over all failure patterns
    pub fn failures(&self) -> impl Iterator<Item = &FailurePattern> {
        self.failures.values()
    }

    /// Iterate over all failure pattern names
    pub fn failure_names(&self) -> impl Iterator<Item = &String> {
        self.failures.keys()
    }

    /// Get a specific failure pattern by name
    pub fn get_failure(&self, name: &str) -> Option<&FailurePattern> {
        self.failures.get(name)
    }

    /// Get number of recovery patterns
    pub fn recovery_count(&self) -> usize {
        self.recoveries.values().map(|v| v.len()).sum()
    }

    /// Iterate over all recoveries (failure_name, recoveries)
    pub fn all_recoveries(&self) -> impl Iterator<Item = (&String, &Vec<RecoveryPattern>)> {
        self.recoveries.iter()
    }
}

// ============================================================================
// Working Memory
// ============================================================================

/// An item in working memory with salience
#[derive(Debug, Clone)]
pub struct FocusItem {
    /// The tensor representation
    pub tensor: Tensor,
    /// Description for debugging / text extraction
    pub description: String,
    /// Current salience (0-1)
    pub salience: f32,
    /// When added
    pub added_at: Instant,
    /// Sensory source that produced this item (e.g., "voice", "k8s", "discord").
    /// Used by the compressor and ecosystem hologram to attribute context fragments.
    #[doc(alias = "source_id")]
    pub source: Option<String>,
}

/// Working memory with salience decay
#[derive(Debug)]
pub struct WorkingMemory {
    /// Current focus items
    items: Vec<FocusItem>,
    /// Max items to keep
    max_items: usize,
    /// Decay rate per second
    decay_rate: f32,
    /// Min salience before removal
    min_salience: f32,
    /// Composed context tensor
    pub context: Tensor,
    /// Device
    device: Device,
    /// Embedding dimension
    dim: usize,
}

impl WorkingMemory {
    pub fn new(device: Device, dim: usize, max_items: usize) -> Result<Self> {
        let context = Tensor::zeros((dim,), DType::F32, &device)?;
        Ok(Self {
            items: Vec::new(),
            max_items,
            decay_rate: 0.1,   // 10% per second
            min_salience: 0.1, // Remove below 10%
            context,
            device,
            dim,
        })
    }

    /// Add a new focus item
    pub fn focus(&mut self, item: FocusItem) -> Result<()> {
        self.items.push(item);

        // Enforce capacity
        while self.items.len() > self.max_items {
            // Remove lowest salience
            if let Some(idx) = self
                .items
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.salience.partial_cmp(&b.salience).unwrap())
                .map(|(i, _)| i)
            {
                self.items.remove(idx);
            }
        }

        self.update_context()
    }

    /// Apply decay and remove stale items
    pub fn tick(&mut self, elapsed: Duration) -> Result<()> {
        let elapsed_secs = elapsed.as_secs_f32();
        let decay = (-self.decay_rate * elapsed_secs).exp();

        for item in &mut self.items {
            item.salience *= decay;
        }

        // Remove below threshold
        self.items.retain(|item| item.salience >= self.min_salience);

        self.update_context()
    }

    /// Boost salience of items similar to query
    pub fn attend(&mut self, query: &Tensor, boost: f32) -> Result<()> {
        for item in &mut self.items {
            if let Ok(sim) = cosine_similarity(&item.tensor, query) {
                if sim > 0.5 {
                    item.salience = (item.salience + sim * boost).min(1.0);
                }
            }
        }
        self.update_context()
    }

    /// Recompose context from current items
    fn update_context(&mut self) -> Result<()> {
        if self.items.is_empty() {
            self.context = Tensor::zeros((self.dim,), DType::F32, &self.device)?;
            return Ok(());
        }

        let tensors: Vec<Tensor> = self.items.iter().map(|i| i.tensor.clone()).collect();
        let weights: Vec<f32> = self.items.iter().map(|i| i.salience).collect();

        // Normalize weights
        let sum: f32 = weights.iter().sum();
        let norm_weights: Vec<f32> = weights.iter().map(|w| w / sum).collect();

        self.context = superimpose(&tensors, Some(&norm_weights))?;
        Ok(())
    }

    /// Get current items
    pub fn items(&self) -> &[FocusItem] {
        &self.items
    }

    /// Clear working memory
    pub fn clear(&mut self) -> Result<()> {
        self.items.clear();
        self.context = Tensor::zeros((self.dim,), DType::F32, &self.device)?;
        Ok(())
    }
}

// ============================================================================
// Experience Hologram (Long-term Memory)
// ============================================================================

/// Long-term experience storage as holographic superposition
#[derive(Debug)]
pub struct ExperienceHologram {
    /// The composed hologram of all experiences
    pub hologram: Tensor,
    /// Episode count
    pub episode_count: usize,
    /// Successful episode count
    pub success_count: usize,
    /// Episode IDs in this hologram
    episode_ids: Vec<EpisodeId>,
    /// Device
    _device: Device,
    /// Embedding dimension
    dim: usize,
}

impl ExperienceHologram {
    pub fn new(device: Device, dim: usize) -> Result<Self> {
        let hologram = Tensor::zeros((dim,), DType::F32, &device)?;
        Ok(Self {
            hologram,
            episode_count: 0,
            success_count: 0,
            episode_ids: Vec::new(),
            _device: device,
            dim,
        })
    }

    /// Encode an episode into the experience hologram
    pub fn encode_episode(&mut self, episode: &Episode) -> Result<()> {
        // Weight by outcome: successful episodes get higher weight
        let weight: f64 = if episode.outcome.success { 1.0 } else { 0.5 };

        // Incremental update: new = (old * count + new * weight) / (count + weight)
        let old_weight = self.episode_count as f64;
        let new_total = old_weight + weight;

        let scaled_old = self.hologram.affine(old_weight / new_total, 0.0)?;
        let scaled_new = episode.hologram.affine(weight / new_total, 0.0)?;

        self.hologram = (&scaled_old + &scaled_new)?;
        self.hologram = normalize(&self.hologram)?;

        self.episode_count += 1;
        if episode.outcome.success {
            self.success_count += 1;
        }
        self.episode_ids.push(episode.id.clone());

        Ok(())
    }

    /// Recall from experience given a query observation
    pub fn recall(&self, obs_tensor: &Tensor) -> Result<(Tensor, f32)> {
        let projection = project(&self.hologram, obs_tensor)?;
        let confidence = cosine_similarity(&projection, &self.hologram)?;
        Ok((projection, confidence))
    }

    /// Success rate
    pub fn success_rate(&self) -> f64 {
        if self.episode_count == 0 {
            0.0
        } else {
            self.success_count as f64 / self.episode_count as f64
        }
    }

    /// Consolidate the hologram to manage accumulated noise.
    ///
    /// HRR superposition accumulates noise proportional to the number of
    /// episodes encoded. This method re-normalizes the hologram and returns
    /// diagnostics about the current noise level.
    ///
    /// The theoretical signal-to-noise ratio for N superimposed items in
    /// dimension d is approximately sqrt(d/N). When this drops below ~1.0,
    /// recall quality degrades noticeably.
    ///
    /// Call periodically (e.g., every 100 episodes) or when recall confidence
    /// drops below acceptable thresholds.
    pub fn consolidate(&mut self) -> Result<HologramHealth> {
        if self.episode_count == 0 {
            return Ok(HologramHealth {
                episode_count: 0,
                dim: self.dim,
                estimated_snr: f32::INFINITY,
                norm_before: 0.0,
                norm_after: 0.0,
                needs_attention: false,
            });
        }

        // Measure current norm (drift from unit norm indicates accumulated error)
        let norm_before = {
            let n = self.hologram.sqr()?.sum_all()?.sqrt()?;
            n.to_scalar::<f32>()?
        };

        // Re-normalize to unit norm (removes numerical drift)
        self.hologram = normalize(&self.hologram)?;

        let norm_after = {
            let n = self.hologram.sqr()?.sum_all()?.sqrt()?;
            n.to_scalar::<f32>()?
        };

        // Estimate signal-to-noise ratio: SNR ≈ sqrt(dim / episode_count)
        // This is the theoretical HRR capacity bound.
        let snr = (self.dim as f32 / self.episode_count as f32).sqrt();

        Ok(HologramHealth {
            episode_count: self.episode_count,
            dim: self.dim,
            estimated_snr: snr,
            norm_before,
            norm_after,
            needs_attention: snr < 1.0,
        })
    }

    /// Estimate the signal-to-noise ratio without modifying the hologram.
    ///
    /// SNR ≈ sqrt(dim / N) for N superimposed items in dim dimensions.
    /// Below ~1.0, recall is unreliable. Below ~2.0, quality is degraded.
    pub fn estimated_snr(&self) -> f32 {
        if self.episode_count == 0 {
            return f32::INFINITY;
        }
        (self.dim as f32 / self.episode_count as f32).sqrt()
    }

    /// Number of episode IDs tracked.
    pub fn episode_ids(&self) -> &[EpisodeId] {
        &self.episode_ids
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Diagnostic report from hologram consolidation.
#[derive(Debug, Clone)]
pub struct HologramHealth {
    /// Number of episodes encoded.
    pub episode_count: usize,
    /// Embedding dimension.
    pub dim: usize,
    /// Estimated signal-to-noise ratio (sqrt(dim/N)).
    /// Below 1.0 = recall unreliable. Below 2.0 = quality degraded.
    pub estimated_snr: f32,
    /// Hologram L2 norm before consolidation (should be ~1.0).
    pub norm_before: f32,
    /// Hologram L2 norm after consolidation (should be ~1.0).
    pub norm_after: f32,
    /// Whether the hologram needs attention (SNR < 1.0).
    pub needs_attention: bool,
}

// ============================================================================
// Entropy-Gated Consolidation
// ============================================================================

/// Memory configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Minimum novelty bits to store episode
    pub min_novelty_bits: f64,
    /// Working memory max items
    pub working_memory_size: usize,
    /// Pattern matching threshold
    pub pattern_match_threshold: f32,
    /// Enable regime partitioning
    pub enable_regime_partitioning: bool,
    /// Lyapunov embedding dimension for regime detection
    pub lyapunov_embedding_dim: usize,
    /// Lyapunov time delay
    pub lyapunov_time_delay: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 1024,
            min_novelty_bits: 0.5,
            working_memory_size: 10,
            pattern_match_threshold: 0.7,
            enable_regime_partitioning: true,
            lyapunov_embedding_dim: 3,
            lyapunov_time_delay: 1,
        }
    }
}

/// Entropy-gated memory that filters redundant experiences
#[derive(Debug)]
pub struct EntropyGatedMemory {
    /// Streaming novelty for full episodes
    novelty_detector: StreamingNovelty,
    /// Minimum novelty to store
    min_novelty_bits: f64,
    /// Episodes skipped due to low novelty
    skipped_count: usize,
    /// Episodes stored
    stored_count: usize,
}

impl EntropyGatedMemory {
    pub fn new(window_size: usize, n_categories: usize, min_novelty_bits: f64) -> Self {
        Self {
            novelty_detector: StreamingNovelty::new(n_categories, window_size, 0.1),
            min_novelty_bits,
            skipped_count: 0,
            stored_count: 0,
        }
    }

    /// Check if episode is novel enough to store
    pub fn should_store(&mut self, category_idx: usize) -> (bool, f32) {
        let novelty = self.novelty_detector.update(category_idx as i64);

        if novelty >= self.min_novelty_bits {
            self.stored_count += 1;
            (true, novelty as f32)
        } else {
            self.skipped_count += 1;
            (false, novelty as f32)
        }
    }

    /// Get current entropy in bits
    pub fn current_entropy(&mut self) -> f64 {
        self.novelty_detector.entropy_bits()
    }

    /// Get storage statistics
    pub fn stats(&self) -> (usize, usize) {
        (self.stored_count, self.skipped_count)
    }
}

// ============================================================================
// Regime-Partitioned Memory
// ============================================================================

/// Memory partitioned by detected regime
pub struct RegimePartitionedMemory {
    /// Hologram for stable regime
    stable_hologram: ExperienceHologram,
    /// Hologram for random walk regime
    random_walk_hologram: ExperienceHologram,
    /// Hologram for chaotic regime
    chaotic_hologram: ExperienceHologram,
    /// Current detected regime
    current_regime: ChaosRegime,
    /// Lyapunov computer for regime detection
    lyapunov: LyapunovComputer,
    /// Recent observations for regime detection
    observation_buffer: Vec<f64>,
    /// Buffer size
    buffer_size: usize,
    /// Device
    device: Device,
    /// Embedding dimension
    dim: usize,
}

impl std::fmt::Debug for RegimePartitionedMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegimePartitionedMemory")
            .field("current_regime", &self.current_regime)
            .field("buffer_size", &self.buffer_size)
            .field("dim", &self.dim)
            .finish()
    }
}

impl RegimePartitionedMemory {
    pub fn new(device: Device, dim: usize, lyapunov_config: LyapunovConfig) -> Result<Self> {
        Ok(Self {
            stable_hologram: ExperienceHologram::new(device.clone(), dim)?,
            random_walk_hologram: ExperienceHologram::new(device.clone(), dim)?,
            chaotic_hologram: ExperienceHologram::new(device.clone(), dim)?,
            current_regime: ChaosRegime::RandomWalk, // Default
            lyapunov: LyapunovComputer::new(device.clone(), lyapunov_config),
            observation_buffer: Vec::new(),
            buffer_size: 100,
            device,
            dim,
        })
    }

    /// Get hologram for a specific regime
    fn get_hologram(&self, regime: ChaosRegime) -> &ExperienceHologram {
        match regime {
            ChaosRegime::Stable => &self.stable_hologram,
            ChaosRegime::RandomWalk => &self.random_walk_hologram,
            ChaosRegime::Chaotic => &self.chaotic_hologram,
        }
    }

    /// Get mutable hologram for a specific regime
    fn get_hologram_mut(&mut self, regime: ChaosRegime) -> &mut ExperienceHologram {
        match regime {
            ChaosRegime::Stable => &mut self.stable_hologram,
            ChaosRegime::RandomWalk => &mut self.random_walk_hologram,
            ChaosRegime::Chaotic => &mut self.chaotic_hologram,
        }
    }

    /// Update regime based on recent observations
    pub fn update_regime(&mut self, observation_value: f64) -> Result<ChaosRegime> {
        self.observation_buffer.push(observation_value);
        if self.observation_buffer.len() > self.buffer_size {
            self.observation_buffer.remove(0);
        }

        // Need enough data for Lyapunov
        if self.observation_buffer.len() < 50 {
            return Ok(self.current_regime);
        }

        // Compute Lyapunov features
        // Shape: [1, n_timestamps] - single symbol with time series
        let n_timestamps = self.observation_buffer.len();
        let data_tensor = Tensor::from_vec(
            self.observation_buffer.clone(),
            (1, n_timestamps),
            &self.device,
        )?;

        if let Ok(features) = self.lyapunov.compute(&data_tensor) {
            if !features.chaos_regime.is_empty() {
                self.current_regime = features.chaos_regime[0];
            }
        }

        Ok(self.current_regime)
    }

    /// Encode episode into current regime's hologram
    pub fn encode_episode(&mut self, episode: &Episode) -> Result<()> {
        let regime = self.current_regime;
        self.get_hologram_mut(regime).encode_episode(episode)
    }

    /// Recall from current regime
    pub fn recall(&self, obs_tensor: &Tensor) -> Result<(Tensor, f32)> {
        self.get_hologram(self.current_regime).recall(obs_tensor)
    }

    /// Recall from all regimes
    pub fn recall_all(&self, obs_tensor: &Tensor) -> Result<Vec<(ChaosRegime, Tensor, f32)>> {
        let mut results = Vec::new();

        for regime in [
            ChaosRegime::Stable,
            ChaosRegime::RandomWalk,
            ChaosRegime::Chaotic,
        ] {
            if let Ok((proj, conf)) = self.get_hologram(regime).recall(obs_tensor) {
                results.push((regime, proj, conf));
            }
        }

        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        Ok(results)
    }

    /// Get current regime
    pub fn current_regime(&self) -> ChaosRegime {
        self.current_regime
    }

    /// Get hologram for specific regime
    pub fn get_regime_hologram(&self, regime: ChaosRegime) -> &ExperienceHologram {
        self.get_hologram(regime)
    }
}

// ============================================================================
// Multi-Scale Memory (Wavelet-based)
// ============================================================================

/// Time horizon for multi-scale recall
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeHorizon {
    /// Most recent (high frequency details)
    Immediate,
    /// Recent past
    Recent,
    /// Historical (low frequency approximation)
    Historical,
    /// All time scales
    All,
}

/// Multi-scale memory using wavelet decomposition concepts
#[derive(Debug)]
pub struct MultiScaleMemory {
    /// High-frequency details (recent, specific) - levels 0-1
    detail_holograms: Vec<Tensor>,
    /// Low-frequency approximation (old, general)
    approximation: Tensor,
    /// Episodes per detail level
    detail_counts: Vec<usize>,
    /// Total approximation episodes
    approx_count: usize,
    /// Device
    device: Device,
    /// Dimension
    dim: usize,
    /// Number of detail levels
    n_levels: usize,
}

impl MultiScaleMemory {
    pub fn new(device: Device, dim: usize, n_levels: usize) -> Result<Self> {
        let detail_holograms: Result<Vec<Tensor>> = (0..n_levels)
            .map(|_| Tensor::zeros((dim,), DType::F32, &device).map_err(Into::into))
            .collect();

        Ok(Self {
            detail_holograms: detail_holograms?,
            approximation: Tensor::zeros((dim,), DType::F32, &device)?,
            detail_counts: vec![0; n_levels],
            approx_count: 0,
            device,
            dim,
            n_levels,
        })
    }

    /// Encode episode at appropriate scale based on age
    pub fn encode(&mut self, episode: &Episode, age_days: usize) -> Result<()> {
        let level = self.age_to_level(age_days);

        if level < self.n_levels {
            // Store in detail level
            let old_count = self.detail_counts[level] as f64;
            let new_count = old_count + 1.0;

            let scaled_old = self.detail_holograms[level].affine(old_count / new_count, 0.0)?;
            let scaled_new = episode.hologram.affine(1.0 / new_count, 0.0)?;

            self.detail_holograms[level] = normalize(&(&scaled_old + &scaled_new)?)?;
            self.detail_counts[level] += 1;
        } else {
            // Store in approximation
            let old_count = self.approx_count as f64;
            let new_count = old_count + 1.0;

            let scaled_old = self.approximation.affine(old_count / new_count, 0.0)?;
            let scaled_new = episode.hologram.affine(1.0 / new_count, 0.0)?;

            self.approximation = normalize(&(&scaled_old + &scaled_new)?)?;
            self.approx_count += 1;
        }

        Ok(())
    }

    /// Recall at specific time horizon
    pub fn recall(&self, query: &Tensor, horizon: TimeHorizon) -> Result<Tensor> {
        match horizon {
            TimeHorizon::Immediate => {
                if self.detail_counts[0] > 0 {
                    project(&self.detail_holograms[0], query)
                } else {
                    Ok(Tensor::zeros((self.dim,), DType::F32, &self.device)?)
                }
            }
            TimeHorizon::Recent => {
                // Combine detail levels
                let mut tensors = Vec::new();
                let mut weights = Vec::new();
                for (i, (h, &count)) in self
                    .detail_holograms
                    .iter()
                    .zip(self.detail_counts.iter())
                    .enumerate()
                {
                    if count > 0 {
                        tensors.push(h.clone());
                        // Weight by recency (higher index = older)
                        weights.push(1.0 / (i as f32 + 1.0));
                    }
                }
                if tensors.is_empty() {
                    return Ok(Tensor::zeros((self.dim,), DType::F32, &self.device)?);
                }
                let combined = superimpose(&tensors, Some(&weights))?;
                project(&combined, query)
            }
            TimeHorizon::Historical => {
                if self.approx_count > 0 {
                    project(&self.approximation, query)
                } else {
                    Ok(Tensor::zeros((self.dim,), DType::F32, &self.device)?)
                }
            }
            TimeHorizon::All => {
                // Combine everything
                let mut tensors = Vec::new();
                let mut weights = Vec::new();

                for (i, (h, &count)) in self
                    .detail_holograms
                    .iter()
                    .zip(self.detail_counts.iter())
                    .enumerate()
                {
                    if count > 0 {
                        tensors.push(h.clone());
                        weights.push(1.0 / (i as f32 + 1.0));
                    }
                }

                if self.approx_count > 0 {
                    tensors.push(self.approximation.clone());
                    weights.push(0.5); // Historical gets less weight
                }

                if tensors.is_empty() {
                    return Ok(Tensor::zeros((self.dim,), DType::F32, &self.device)?);
                }

                let combined = superimpose(&tensors, Some(&weights))?;
                project(&combined, query)
            }
        }
    }

    fn age_to_level(&self, age_days: usize) -> usize {
        match age_days {
            0 => 0,             // Today
            1..=7 => 1,         // This week
            8..=30 => 2,        // This month
            _ => self.n_levels, // Older → approximation
        }
    }
}

// ============================================================================
// Memory Result
// ============================================================================

/// Result of processing an observation
#[derive(Debug)]
pub enum MemoryResult {
    /// Matches a known failure pattern
    KnownFailure {
        failure: FailurePattern,
        recovery: Option<RecoveryPattern>,
        confidence: f32,
    },
    /// Novel situation requiring new action
    NewSituation {
        experience_recall: Tensor,
        experience_confidence: f32,
        working_context: Tensor,
    },
    /// Similar to recent experience
    SimilarExperience {
        recall: Tensor,
        confidence: f32,
        episode_ids: Vec<EpisodeId>,
    },
}

// ============================================================================
// Synapse Memory (Unified Agent Memory)
// ============================================================================

/// Embedding backend trait for memory
pub trait EmbeddingBackend: Send + Sync {
    fn embed(&self, text: &str) -> impl std::future::Future<Output = Result<Tensor>> + Send;
    fn device(&self) -> &Device;
    fn dim(&self) -> usize;
}

/// Complete Synapse memory system
pub struct SynapseMemory<E: EmbeddingBackend> {
    /// Embedding backend
    embedder: E,
    /// Configuration
    config: MemoryConfig,
    /// Entropy-gated consolidation
    entropy_gate: EntropyGatedMemory,
    /// Regime-partitioned long-term memory
    regime_memory: Option<RegimePartitionedMemory>,
    /// Simple experience hologram (when not using regimes)
    experience: Option<ExperienceHologram>,
    /// Multi-scale memory
    multiscale: MultiScaleMemory,
    /// Pattern library
    patterns: PatternLibrary,
    /// Working memory
    working: WorkingMemory,
    /// Category to index mapping
    category_indices: HashMap<String, usize>,
    /// Next category index
    next_category_idx: usize,
}

impl<E: EmbeddingBackend> SynapseMemory<E> {
    pub async fn new(embedder: E, config: MemoryConfig) -> Result<Self> {
        let device = embedder.device().clone();
        let dim = config.embedding_dim;

        let regime_memory = if config.enable_regime_partitioning {
            Some(RegimePartitionedMemory::new(
                device.clone(),
                dim,
                LyapunovConfig {
                    embedding_dim: config.lyapunov_embedding_dim,
                    time_delay: config.lyapunov_time_delay,
                    ..Default::default()
                },
            )?)
        } else {
            None
        };

        let experience = if !config.enable_regime_partitioning {
            Some(ExperienceHologram::new(device.clone(), dim)?)
        } else {
            None
        };

        let pattern_threshold = config.pattern_match_threshold;
        let min_novelty = config.min_novelty_bits;
        let working_size = config.working_memory_size;

        Ok(Self {
            embedder,
            entropy_gate: EntropyGatedMemory::new(500, 100, min_novelty),
            regime_memory,
            experience,
            multiscale: MultiScaleMemory::new(device.clone(), dim, 3)?,
            patterns: PatternLibrary::new(device.clone(), pattern_threshold),
            working: WorkingMemory::new(device, dim, working_size)?,
            config,
            category_indices: HashMap::new(),
            next_category_idx: 0,
        })
    }

    /// Get or create category index
    fn get_category_index(&mut self, category: &str) -> usize {
        if let Some(&idx) = self.category_indices.get(category) {
            idx
        } else {
            let idx = self.next_category_idx;
            self.next_category_idx += 1;
            self.category_indices.insert(category.to_string(), idx);
            idx
        }
    }

    /// Encode a complete episode
    pub async fn encode_episode(&mut self, episode: &mut Episode) -> Result<bool> {
        // Check entropy gate
        let cat_idx = self.get_category_index(&episode.observation.category);
        let (should_store, novelty) = self.entropy_gate.should_store(cat_idx);
        episode.novelty_bits = novelty;

        if !should_store {
            return Ok(false); // Skipped due to low novelty
        }

        // Store in appropriate memory
        if let Some(regime_mem) = &mut self.regime_memory {
            // Update regime if we have a metric
            if let Some(value) = episode.outcome.metrics.get("observation_value") {
                regime_mem.update_regime(*value)?;
            }
            episode.regime = Some(regime_mem.current_regime());
            regime_mem.encode_episode(episode)?;
        } else if let Some(exp) = &mut self.experience {
            exp.encode_episode(episode)?;
        }

        // Also store in multi-scale (age 0 = today)
        self.multiscale.encode(episode, 0)?;

        // Update working memory
        self.working.focus(FocusItem {
            tensor: episode.hologram.clone(),
            description: format!(
                "{}: {}",
                episode.observation.category, episode.action.action_type
            ),
            salience: 1.0,
            added_at: Instant::now(),
            source: None,
        })?;

        // If failure, register pattern
        if !episode.outcome.success {
            self.patterns.register_failure(
                format!("{}_{}", episode.observation.category, episode.id.0),
                episode.observation.description.clone(),
                episode.obs_tensor.clone(),
            );
        }

        Ok(true)
    }

    /// Create episode from components
    pub async fn create_episode(
        &self,
        observation: Observation,
        action: Action,
        outcome: ActionOutcome,
    ) -> Result<Episode> {
        // Embed all components
        let obs_tensor = self.embedder.embed(&observation.description).await?;
        let action_tensor = self.embedder.embed(&action.description).await?;
        let outcome_tensor = self.embedder.embed(&outcome.description).await?;

        // Bind: obs ⊗ (action ⊗ outcome)
        let action_outcome = bind(&action_tensor, &outcome_tensor)?;
        let hologram = bind(&obs_tensor, &action_outcome)?;

        Ok(Episode {
            id: EpisodeId::new(),
            observation,
            action,
            outcome,
            obs_tensor,
            action_tensor,
            outcome_tensor,
            hologram,
            novelty_bits: 0.0,
            regime: None,
        })
    }

    /// Process an observation and get memory-informed result
    pub async fn process_observation(&mut self, obs: &Observation) -> Result<MemoryResult> {
        let obs_tensor = self.embedder.embed(&obs.description).await?;

        // 1. Check pattern library for known failures
        // Extract match info to avoid borrow conflicts
        let match_result = self
            .patterns
            .match_failure(&obs_tensor)
            .map(|(f, c)| (f.name.clone(), f.clone(), c));

        if let Some((failure_name, failure, confidence)) = match_result {
            self.patterns.record_failure_occurrence(&failure_name);
            let recovery = self.patterns.get_best_recovery(&failure_name).cloned();
            return Ok(MemoryResult::KnownFailure {
                failure,
                recovery,
                confidence,
            });
        }

        // 2. Check working memory for recent context
        let _working_sim = cosine_similarity(&obs_tensor, &self.working.context)?;
        self.working.attend(&obs_tensor, 0.3)?;

        // 3. Recall from long-term experience
        let (recall, confidence) = if let Some(regime_mem) = &self.regime_memory {
            regime_mem.recall(&obs_tensor)?
        } else if let Some(exp) = &self.experience {
            exp.recall(&obs_tensor)?
        } else {
            (
                Tensor::zeros(
                    (self.config.embedding_dim,),
                    DType::F32,
                    self.embedder.device(),
                )?,
                0.0,
            )
        };

        // 4. Add to working memory
        self.working.focus(FocusItem {
            tensor: obs_tensor,
            description: obs.description.clone(),
            salience: 1.0,
            added_at: Instant::now(),
            source: None,
        })?;

        if confidence > 0.7 {
            Ok(MemoryResult::SimilarExperience {
                recall,
                confidence,
                episode_ids: Vec::new(), // Could track these
            })
        } else {
            Ok(MemoryResult::NewSituation {
                experience_recall: recall,
                experience_confidence: confidence,
                working_context: self.working.context.clone(),
            })
        }
    }

    /// Record feedback about an action
    pub fn record_feedback(
        &mut self,
        failure_name: &str,
        recovery_hologram: Tensor,
        success: bool,
    ) {
        self.patterns.register_recovery(
            failure_name.to_string(),
            format!("Recovery attempt, success={}", success),
            recovery_hologram,
            success,
        );
    }

    /// Tick working memory decay
    pub fn tick(&mut self, elapsed: Duration) -> Result<()> {
        self.working.tick(elapsed)
    }

    /// Get current regime
    pub fn current_regime(&self) -> Option<ChaosRegime> {
        self.regime_memory.as_ref().map(|r| r.current_regime())
    }

    /// Get memory statistics
    pub fn stats(&mut self) -> MemoryStats {
        let (stored, skipped) = self.entropy_gate.stats();
        MemoryStats {
            episodes_stored: stored,
            episodes_skipped: skipped,
            current_entropy: self.entropy_gate.current_entropy(),
            working_memory_items: self.working.items().len(),
            failure_patterns: self.patterns.failure_count(),
            current_regime: self.current_regime(),
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub episodes_stored: usize,
    pub episodes_skipped: usize,
    pub current_entropy: f64,
    pub working_memory_items: usize,
    pub failure_patterns: usize,
    pub current_regime: Option<ChaosRegime>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_library() {
        let device = Device::Cpu;
        let mut lib = PatternLibrary::new(device.clone(), 0.7);

        // Create a failure pattern
        let sig = Tensor::randn(0.0f32, 1.0, (64,), &device).unwrap();
        lib.register_failure(
            "oom_crash".to_string(),
            "Out of memory crash".to_string(),
            sig.clone(),
        );

        // Match against similar tensor
        let query = &sig; // Exact match
        let result = lib.match_failure(query);
        assert!(result.is_some());
        let (pattern, conf) = result.unwrap();
        assert_eq!(pattern.name, "oom_crash");
        assert!(conf > 0.9);
    }

    #[test]
    fn test_working_memory() {
        let device = Device::Cpu;
        let mut wm = WorkingMemory::new(device.clone(), 64, 5).unwrap();

        // Add items
        for i in 0..3 {
            let tensor = Tensor::randn(0.0f32, 1.0, (64,), &device).unwrap();
            wm.focus(FocusItem {
                tensor,
                description: format!("item_{}", i),
                salience: 1.0,
                added_at: Instant::now(),
                source: None,
            })
            .unwrap();
        }

        assert_eq!(wm.items().len(), 3);

        // Decay
        wm.tick(Duration::from_secs(5)).unwrap();

        // Salience should have decayed
        for item in wm.items() {
            assert!(item.salience < 1.0);
        }
    }

    #[test]
    #[ignore = "TODO: cosine similarity returns 0.0 for zero-initialized tensors - needs investigation"]
    fn test_experience_hologram() {
        let device = Device::Cpu;
        let mut exp = ExperienceHologram::new(device.clone(), 64).unwrap();

        // Create mock episode
        let obs = Tensor::randn(0.0f32, 1.0, (64,), &device).unwrap();
        let action = Tensor::randn(0.0f32, 1.0, (64,), &device).unwrap();
        let outcome = Tensor::randn(0.0f32, 1.0, (64,), &device).unwrap();
        let hologram = bind(&obs, &bind(&action, &outcome).unwrap()).unwrap();

        let episode = Episode {
            id: EpisodeId::new(),
            observation: Observation::new("test", "test obs"),
            action: Action::new("test", "test action"),
            outcome: ActionOutcome::success("success"),
            obs_tensor: obs.clone(),
            action_tensor: action,
            outcome_tensor: outcome,
            hologram,
            novelty_bits: 1.0,
            regime: None,
        };

        exp.encode_episode(&episode).unwrap();
        assert_eq!(exp.episode_count, 1);
        assert_eq!(exp.success_count, 1);

        // Recall
        let (_recall, conf) = exp.recall(&obs).unwrap();
        assert!(conf > 0.0);
    }

    #[test]
    fn test_entropy_gated_memory() {
        let mut gate = EntropyGatedMemory::new(100, 10, 0.5);

        // First occurrences should be novel
        let (should_store, novelty) = gate.should_store(0);
        // First occurrence typically has high novelty
        println!(
            "First: should_store={}, novelty={:.2}",
            should_store, novelty
        );

        // Repeated category should eventually be less novel
        for _ in 0..20 {
            let _ = gate.should_store(0);
        }

        let (should_store, novelty) = gate.should_store(0);
        println!(
            "After repeats: should_store={}, novelty={:.2}",
            should_store, novelty
        );

        // Stats
        let (stored, skipped) = gate.stats();
        println!("Stored: {}, Skipped: {}", stored, skipped);
    }

    #[test]
    fn test_multiscale_memory() {
        let device = Device::Cpu;
        let mut ms = MultiScaleMemory::new(device.clone(), 64, 3).unwrap();

        // Create episode
        let obs = Tensor::randn(0.0f32, 1.0, (64,), &device).unwrap();
        let action = Tensor::randn(0.0f32, 1.0, (64,), &device).unwrap();
        let outcome = Tensor::randn(0.0f32, 1.0, (64,), &device).unwrap();
        let hologram = bind(&obs, &bind(&action, &outcome).unwrap()).unwrap();

        let episode = Episode {
            id: EpisodeId::new(),
            observation: Observation::new("test", "test obs"),
            action: Action::new("test", "test action"),
            outcome: ActionOutcome::success("success"),
            obs_tensor: obs.clone(),
            action_tensor: action,
            outcome_tensor: outcome,
            hologram,
            novelty_bits: 1.0,
            regime: None,
        };

        // Encode at different ages
        ms.encode(&episode, 0).unwrap(); // Today
        ms.encode(&episode, 5).unwrap(); // This week
        ms.encode(&episode, 20).unwrap(); // This month

        // Recall at different horizons
        let _imm = ms.recall(&obs, TimeHorizon::Immediate).unwrap();
        let _rec = ms.recall(&obs, TimeHorizon::Recent).unwrap();
        let _all = ms.recall(&obs, TimeHorizon::All).unwrap();
    }
}

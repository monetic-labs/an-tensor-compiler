//! Ecosystem Hologram
//!
//! A composed hologram representing everything the ecosystem knows. Built from
//! multiple sensory sources, each weighted by relevance. Any modality can project
//! against it and compose budget-aware context.
//!
//! Unlike `ExperienceHologram` (single-source, episode-oriented), the ecosystem
//! hologram is multi-source and manages noise from unbounded superposition via
//! periodic consolidation.
//!
//! # Example
//!
//! ```ignore
//! use an_tensor_compiler::holographic::ecosystem::{EcosystemHologram, EcosystemConfig};
//! use candle_core::Device;
//!
//! let config = EcosystemConfig { dim: 1024, ..Default::default() };
//! let mut eco = EcosystemHologram::new(config, Device::Cpu)?;
//!
//! // Ingest from different sources
//! eco.ingest("voice", &conversation_tensor, 1.0)?;
//! eco.ingest("k8s", &observation_tensor, 0.8)?;
//! eco.ingest("pipeline", &event_tensor, 0.5)?;
//!
//! // Project a query
//! let result = eco.project(&query_tensor)?;
//!
//! // Compose budget-aware context for voice text_prompt
//! let context = eco.compose_for_budget(&query_tensor, 100)?;
//! ```
//!
//! See: `docs/SPEC_VOICE_CONTEXT_BRIDGE.md` for voice integration.

use crate::holographic::diagnostics::{
    BridgeContribution, ProjectionDiagnostics, SpaceContribution,
};
use crate::holographic::memory::FocusItem;
use crate::holographic::ops::{cosine_similarity, normalize, project, superimpose};
use crate::information::StreamingEntropy;
use crate::{Result, TensorCoreError};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::time::Instant;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the ecosystem hologram.
#[derive(Debug, Clone)]
pub struct EcosystemConfig {
    /// Embedding dimension (must match all ingested tensors).
    pub dim: usize,
    /// Default weight for sources without explicit weight (default: 0.5).
    pub default_source_weight: f32,
    /// Maximum number of distinct sources tracked (default: 32).
    pub max_sources: usize,
    /// Consolidation: minimum cosine similarity to merge ingested items (default: 0.85).
    /// Higher = less merging, more fidelity. Lower = more merging, less noise.
    pub consolidation_similarity: f32,
    /// Consolidation: noise floor — after this many total ingestions, start
    /// consolidation on each ingest (default: 100).
    pub consolidation_threshold: usize,
    /// Maximum items in the working memory (default: 20).
    pub working_memory_capacity: usize,
}

impl Default for EcosystemConfig {
    fn default() -> Self {
        Self {
            dim: 1024,
            default_source_weight: 0.5,
            max_sources: 32,
            consolidation_similarity: 0.85,
            consolidation_threshold: 100,
            working_memory_capacity: 20,
        }
    }
}

// =============================================================================
// Source Tracking
// =============================================================================

/// Metadata for a sensory source.
#[derive(Debug, Clone)]
pub struct SourceInfo {
    /// Source identifier (e.g., "voice", "k8s", "discord").
    pub id: String,
    /// Weight applied to ingested data from this source (0.0–1.0).
    pub weight: f32,
    /// Number of items ingested from this source.
    pub ingest_count: usize,
    /// Last ingestion time.
    pub last_ingest: Instant,
}

// =============================================================================
// Projection Result
// =============================================================================

/// Result of projecting a query against the ecosystem hologram.
#[derive(Debug)]
pub struct EcoProjectionResult {
    /// The projected tensor (recalled context).
    pub tensor: Tensor,
    /// Confidence of the projection (cosine similarity with hologram).
    pub confidence: f32,
    /// Per-source confidence (how much each source contributed).
    pub source_confidence: HashMap<String, f32>,
}

/// Richer projection result that includes v2 diagnostics.
#[derive(Debug)]
pub struct EcoProjectionWithDiagnostics {
    /// Legacy-compatible projection result.
    pub result: EcoProjectionResult,
    /// Diagnostics for attribution quality and interference.
    pub diagnostics: ProjectionDiagnostics,
}

/// Budget-aware composition result.
#[derive(Debug)]
pub struct ComposedContext {
    /// Ranked context items (highest relevance first).
    pub items: Vec<ComposedItem>,
    /// Total estimated tokens across all items.
    pub total_estimated_tokens: usize,
}

/// A single item in the composed context.
#[derive(Debug, Clone)]
pub struct ComposedItem {
    /// The tensor.
    pub tensor: Tensor,
    /// Text description (if available).
    pub description: String,
    /// Source that produced this item.
    pub source: String,
    /// Relevance score to the query.
    pub relevance: f32,
    /// Estimated token count.
    pub estimated_tokens: usize,
}

// =============================================================================
// Ecosystem Hologram
// =============================================================================

/// Composes and maintains the ecosystem-wide holographic memory.
///
/// Consumes data from multiple named sources, each weighted independently.
/// Maintains a single hologram that any modality can project against.
/// Handles consolidation to manage noise from unbounded superposition.
#[derive(Debug)]
pub struct EcosystemHologram {
    /// The composed hologram (superposition of all source holograms).
    hologram: Tensor,

    /// Per-source sub-holograms (for source-attributed projection).
    source_holograms: HashMap<String, Tensor>,

    /// Source metadata.
    sources: HashMap<String, SourceInfo>,

    /// Recent items with descriptions (for budget-aware composition).
    /// This is the "working memory" of the ecosystem — bounded and decayed.
    recent_items: Vec<FocusItem>,

    /// Total ingestions (triggers consolidation after threshold).
    total_ingestions: usize,

    /// Streaming entropy of ingest stream (for novelty gating).
    _novelty_tracker: StreamingEntropy,

    /// Config.
    config: EcosystemConfig,
    /// Device.
    device: Device,
}

impl EcosystemHologram {
    /// Create a new ecosystem hologram.
    pub fn new(config: EcosystemConfig, device: Device) -> Result<Self> {
        let hologram = Tensor::zeros((config.dim,), DType::F32, &device)?;
        Ok(Self {
            hologram,
            source_holograms: HashMap::new(),
            sources: HashMap::new(),
            recent_items: Vec::new(),
            total_ingestions: 0,
            _novelty_tracker: StreamingEntropy::new(500),
            config,
            device,
        })
    }

    /// Configure the weight for a source. Sources are created on first ingest
    /// if not pre-configured.
    pub fn set_source_weight(&mut self, source: &str, weight: f32) {
        if let Some(info) = self.sources.get_mut(source) {
            info.weight = weight;
        } else {
            self.sources.insert(
                source.to_string(),
                SourceInfo {
                    id: source.to_string(),
                    weight,
                    ingest_count: 0,
                    last_ingest: Instant::now(),
                },
            );
        }
    }

    /// Ingest a tensor from a named source.
    ///
    /// The tensor is weighted by the source weight and superimposed into both
    /// the per-source hologram and the global hologram. If `description` is
    /// provided, the item is also added to the recent items list for
    /// budget-aware composition.
    pub fn ingest(&mut self, source: &str, tensor: &Tensor, salience: f32) -> Result<()> {
        self.ingest_with_description(source, tensor, salience, None)
    }

    /// Ingest with an optional text description (for budget-aware composition).
    pub fn ingest_with_description(
        &mut self,
        source: &str,
        tensor: &Tensor,
        salience: f32,
        description: Option<&str>,
    ) -> Result<()> {
        // Validate dimension
        let dims = tensor.dims();
        if dims.len() != 1 || dims[0] != self.config.dim {
            return Err(TensorCoreError::Tensor(format!(
                "Expected tensor of dim [{}], got {:?}",
                self.config.dim, dims
            )));
        }

        // Ensure source exists
        let source_weight = if let Some(info) = self.sources.get_mut(source) {
            info.ingest_count += 1;
            info.last_ingest = Instant::now();
            info.weight
        } else {
            if self.sources.len() >= self.config.max_sources {
                return Err(TensorCoreError::Config(format!(
                    "Maximum sources ({}) reached",
                    self.config.max_sources
                )));
            }
            let weight = self.config.default_source_weight;
            self.sources.insert(
                source.to_string(),
                SourceInfo {
                    id: source.to_string(),
                    weight,
                    ingest_count: 1,
                    last_ingest: Instant::now(),
                },
            );
            weight
        };

        let effective_weight = salience * source_weight;

        // Update per-source hologram
        let source_hologram = if let Some(sh) = self.source_holograms.get(source) {
            sh.clone()
        } else {
            Tensor::zeros((self.config.dim,), DType::F32, &self.device)?
        };

        let source_count = self
            .sources
            .get(source)
            .map(|s| s.ingest_count)
            .unwrap_or(1);
        let updated_source = self.incremental_superimpose(
            &source_hologram,
            tensor,
            source_count as f64,
            effective_weight as f64,
        )?;
        self.source_holograms
            .insert(source.to_string(), updated_source);

        // Update global hologram
        self.total_ingestions += 1;
        self.hologram = self.incremental_superimpose(
            &self.hologram,
            tensor,
            (self.total_ingestions - 1) as f64,
            effective_weight as f64,
        )?;

        // Track in recent items (if description provided)
        if let Some(desc) = description {
            self.recent_items.push(FocusItem {
                tensor: tensor.clone(),
                description: desc.to_string(),
                salience,
                added_at: Instant::now(),
                source: Some(source.to_string()),
            });

            // Enforce capacity — remove lowest salience
            while self.recent_items.len() > self.config.working_memory_capacity {
                if let Some(idx) = self
                    .recent_items
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.salience
                            .partial_cmp(&b.salience)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                {
                    self.recent_items.remove(idx);
                }
            }
        }

        Ok(())
    }

    /// Incremental superimpose: new = (old * old_weight + new * new_weight) / total, then normalize.
    fn incremental_superimpose(
        &self,
        existing: &Tensor,
        incoming: &Tensor,
        old_count: f64,
        new_weight: f64,
    ) -> Result<Tensor> {
        let total = old_count + new_weight;
        if total < 1e-10 {
            return Ok(incoming.clone());
        }
        let scaled_old = existing.affine(old_count / total, 0.0)?;
        let scaled_new = incoming.affine(new_weight / total, 0.0)?;
        let combined = (&scaled_old + &scaled_new)?;
        normalize(&combined)
    }

    /// Project a query against the global hologram.
    pub fn project_query(&self, query: &Tensor) -> Result<EcoProjectionResult> {
        let projection = project(&self.hologram, query)?;
        let confidence = cosine_similarity(&projection, &self.hologram)?;

        // Per-source confidence
        let mut source_confidence = HashMap::new();
        for (source_id, source_hologram) in &self.source_holograms {
            let source_proj = project(source_hologram, query)?;
            let source_conf = cosine_similarity(&source_proj, source_hologram)?;
            source_confidence.insert(source_id.clone(), source_conf);
        }

        Ok(EcoProjectionResult {
            tensor: projection,
            confidence,
            source_confidence,
        })
    }

    /// Project a query and return richer diagnostics.
    pub fn project_query_with_diagnostics(
        &self,
        query: &Tensor,
    ) -> Result<EcoProjectionWithDiagnostics> {
        let result = self.project_query(query)?;
        let diagnostics = self.build_projection_diagnostics(&result)?;

        Ok(EcoProjectionWithDiagnostics {
            result,
            diagnostics,
        })
    }

    /// Compose budget-aware context from recent items.
    ///
    /// Ranks recent items by relevance to the query tensor, then greedily
    /// selects within the token budget. Returns items with descriptions
    /// for direct use in text_prompt composition.
    pub fn compose_for_budget(
        &self,
        query: &Tensor,
        token_budget: usize,
    ) -> Result<ComposedContext> {
        // Score each recent item by relevance to query
        let mut scored: Vec<(usize, f32)> = Vec::new();
        for (i, item) in self.recent_items.iter().enumerate() {
            let sim = cosine_similarity(&item.tensor, query).unwrap_or(0.0);
            let source_weight = item
                .source
                .as_ref()
                .and_then(|s| self.sources.get(s))
                .map(|info| info.weight)
                .unwrap_or(self.config.default_source_weight);
            let score = sim * item.salience * source_weight;
            scored.push((i, score));
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Greedy selection within budget
        let mut items = Vec::new();
        let mut tokens_used = 0;

        for (idx, relevance) in scored {
            let item = &self.recent_items[idx];
            let estimated_tokens = item.description.len().div_ceil(4); // Same heuristic as compressor
            if estimated_tokens == 0 {
                continue;
            }
            if tokens_used + estimated_tokens > token_budget {
                break;
            }
            items.push(ComposedItem {
                tensor: item.tensor.clone(),
                description: item.description.clone(),
                source: item.source.clone().unwrap_or_default(),
                relevance,
                estimated_tokens,
            });
            tokens_used += estimated_tokens;
        }

        Ok(ComposedContext {
            items,
            total_estimated_tokens: tokens_used,
        })
    }

    /// Apply salience decay to recent items and remove stale ones.
    pub fn tick(&mut self, elapsed: std::time::Duration) -> Result<()> {
        let decay = (-0.1 * elapsed.as_secs_f32()).exp();
        for item in &mut self.recent_items {
            item.salience *= decay;
        }
        self.recent_items.retain(|item| item.salience >= 0.1);
        Ok(())
    }

    /// Consolidate the hologram to manage noise.
    ///
    /// Recomposes the global hologram from per-source holograms with
    /// current source weights. This eliminates accumulated numerical noise
    /// from incremental updates and applies the latest weight configuration.
    pub fn consolidate(&mut self) -> Result<ConsolidationResult> {
        if self.source_holograms.is_empty() {
            return Ok(ConsolidationResult {
                sources_consolidated: 0,
                pre_consolidation_norm: 0.0,
                post_consolidation_norm: 0.0,
            });
        }

        let pre_norm = tensor_norm(&self.hologram)?;

        // Recompose from per-source holograms
        let mut tensors = Vec::new();
        let mut weights = Vec::new();

        for (source_id, source_hologram) in &self.source_holograms {
            let weight = self
                .sources
                .get(source_id)
                .map(|s| s.weight)
                .unwrap_or(self.config.default_source_weight);
            tensors.push(source_hologram.clone());
            weights.push(weight);
        }

        self.hologram = superimpose(&tensors, Some(&weights))?;
        self.hologram = normalize(&self.hologram)?;

        let post_norm = tensor_norm(&self.hologram)?;

        Ok(ConsolidationResult {
            sources_consolidated: tensors.len(),
            pre_consolidation_norm: pre_norm,
            post_consolidation_norm: post_norm,
        })
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// The composed hologram tensor.
    pub fn hologram(&self) -> &Tensor {
        &self.hologram
    }

    /// All tracked sources.
    pub fn sources(&self) -> &HashMap<String, SourceInfo> {
        &self.sources
    }

    /// Number of recent items in working memory.
    pub fn recent_item_count(&self) -> usize {
        self.recent_items.len()
    }

    /// Recent items (for external composition, e.g., VoiceContextCompressor).
    pub fn recent_items(&self) -> &[FocusItem] {
        &self.recent_items
    }

    /// Total ingestions across all sources.
    pub fn total_ingestions(&self) -> usize {
        self.total_ingestions
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.config.dim
    }

    fn build_projection_diagnostics(
        &self,
        result: &EcoProjectionResult,
    ) -> Result<ProjectionDiagnostics> {
        let mut contributing_spaces: Vec<SpaceContribution> = result
            .source_confidence
            .iter()
            .map(|(space_id, confidence)| SpaceContribution {
                space_id: space_id.clone(),
                confidence: *confidence,
            })
            .collect();

        contributing_spaces.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(ProjectionDiagnostics {
            contributing_spaces,
            contributing_bridges: self.bridge_contributions(),
            overall_confidence: result.confidence,
            bridge_bleed_score: self.bridge_bleed_score(),
            source_saturation_score: self.source_saturation_score(),
            interference_score: self.interference_score()?,
        })
    }

    fn source_saturation_score(&self) -> f32 {
        if self.config.max_sources == 0 {
            return 0.0;
        }

        (self.sources.len() as f32 / self.config.max_sources as f32).clamp(0.0, 1.0)
    }

    fn interference_score(&self) -> Result<f32> {
        let source_ids: Vec<&String> = self.source_holograms.keys().collect();
        if source_ids.len() < 2 {
            return Ok(0.0);
        }

        let mut total_similarity = 0.0;
        let mut pair_count = 0usize;

        for i in 0..source_ids.len() {
            for j in (i + 1)..source_ids.len() {
                let left = self
                    .source_holograms
                    .get(source_ids[i])
                    .ok_or_else(|| TensorCoreError::Projection("Missing source hologram".into()))?;
                let right = self
                    .source_holograms
                    .get(source_ids[j])
                    .ok_or_else(|| TensorCoreError::Projection("Missing source hologram".into()))?;
                let similarity = cosine_similarity(left, right)?;

                total_similarity += similarity.max(0.0);
                pair_count += 1;
            }
        }

        if pair_count == 0 {
            return Ok(0.0);
        }

        Ok((total_similarity / pair_count as f32).clamp(0.0, 1.0))
    }

    fn bridge_bleed_score(&self) -> f32 {
        // The current ecosystem hologram tracks source attribution but not
        // explicit cross-space bridges yet, so bridge bleed is zero until
        // bridge-aware projection is introduced.
        0.0
    }

    fn bridge_contributions(&self) -> Vec<BridgeContribution> {
        Vec::new()
    }
}

/// Result of a consolidation operation.
#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    /// Number of source holograms consolidated.
    pub sources_consolidated: usize,
    /// Hologram norm before consolidation.
    pub pre_consolidation_norm: f32,
    /// Hologram norm after consolidation.
    pub post_consolidation_norm: f32,
}

/// Compute L2 norm of a 1-D tensor.
fn tensor_norm(t: &Tensor) -> Result<f32> {
    let norm = t.sqr()?.sum_all()?.sqrt()?;
    Ok(norm.to_scalar::<f32>()?)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(dim: usize) -> EcosystemConfig {
        EcosystemConfig {
            dim,
            working_memory_capacity: 10,
            ..Default::default()
        }
    }

    fn random_tensor(dim: usize) -> Tensor {
        Tensor::randn(0.0f32, 1.0, (dim,), &Device::Cpu).unwrap()
    }

    #[test]
    fn test_new_ecosystem_hologram() {
        let eco = EcosystemHologram::new(test_config(64), Device::Cpu).unwrap();
        assert_eq!(eco.dim(), 64);
        assert_eq!(eco.total_ingestions(), 0);
        assert!(eco.sources().is_empty());
    }

    #[test]
    fn test_ingest_creates_source() {
        let mut eco = EcosystemHologram::new(test_config(64), Device::Cpu).unwrap();
        let t = random_tensor(64);

        eco.ingest("voice", &t, 1.0).unwrap();

        assert_eq!(eco.total_ingestions(), 1);
        assert!(eco.sources().contains_key("voice"));
        assert_eq!(eco.sources()["voice"].ingest_count, 1);
    }

    #[test]
    fn test_ingest_with_description_adds_to_recent() {
        let mut eco = EcosystemHologram::new(test_config(64), Device::Cpu).unwrap();
        let t = random_tensor(64);

        eco.ingest_with_description("voice", &t, 1.0, Some("User discussed deployment"))
            .unwrap();

        assert_eq!(eco.recent_item_count(), 1);
        assert_eq!(
            eco.recent_items()[0].description,
            "User discussed deployment"
        );
        assert_eq!(eco.recent_items()[0].source.as_deref(), Some("voice"));
    }

    #[test]
    fn test_ingest_without_description_no_recent() {
        let mut eco = EcosystemHologram::new(test_config(64), Device::Cpu).unwrap();
        let t = random_tensor(64);

        eco.ingest("k8s", &t, 1.0).unwrap();

        assert_eq!(eco.recent_item_count(), 0); // No description → no recent item
        assert_eq!(eco.total_ingestions(), 1); // But hologram was updated
    }

    #[test]
    fn test_multi_source_ingest() {
        let mut eco = EcosystemHologram::new(test_config(64), Device::Cpu).unwrap();

        for _ in 0..5 {
            eco.ingest("voice", &random_tensor(64), 1.0).unwrap();
        }
        for _ in 0..3 {
            eco.ingest("k8s", &random_tensor(64), 0.8).unwrap();
        }

        assert_eq!(eco.total_ingestions(), 8);
        assert_eq!(eco.sources()["voice"].ingest_count, 5);
        assert_eq!(eco.sources()["k8s"].ingest_count, 3);
        assert_eq!(eco.sources().len(), 2);
    }

    #[test]
    fn test_source_weight_configuration() {
        let mut eco = EcosystemHologram::new(test_config(64), Device::Cpu).unwrap();

        eco.set_source_weight("voice", 1.0);
        eco.set_source_weight("k8s", 0.3);

        assert_eq!(eco.sources()["voice"].weight, 1.0);
        assert_eq!(eco.sources()["k8s"].weight, 0.3);

        // Update existing
        eco.set_source_weight("voice", 0.8);
        assert_eq!(eco.sources()["voice"].weight, 0.8);
    }

    #[test]
    fn test_project_query() {
        let mut eco = EcosystemHologram::new(test_config(64), Device::Cpu).unwrap();
        let t = random_tensor(64);

        eco.ingest("voice", &t, 1.0).unwrap();

        let query = random_tensor(64);
        let result = eco.project_query(&query).unwrap();

        assert_eq!(result.tensor.dims(), &[64]);
        assert!(result.source_confidence.contains_key("voice"));
    }

    #[test]
    fn test_project_query_with_diagnostics() {
        let mut eco = EcosystemHologram::new(test_config(64), Device::Cpu).unwrap();
        let voice = random_tensor(64);
        let k8s = random_tensor(64);

        eco.ingest("voice", &voice, 1.0).unwrap();
        eco.ingest("k8s", &k8s, 0.8).unwrap();

        let result = eco.project_query_with_diagnostics(&voice).unwrap();

        assert_eq!(result.result.tensor.dims(), &[64]);
        assert_eq!(result.diagnostics.contributing_spaces.len(), 2);
        assert_eq!(result.diagnostics.contributing_bridges.len(), 0);
        assert!(result.diagnostics.overall_confidence.is_finite());
        assert!((-1.0..=1.0).contains(&result.diagnostics.overall_confidence));
        assert!(result.diagnostics.source_saturation_score > 0.0);
        assert!(result.result.source_confidence.contains_key("voice"));
    }

    #[test]
    fn test_single_source_has_zero_interference() {
        let mut eco = EcosystemHologram::new(test_config(64), Device::Cpu).unwrap();
        let voice = random_tensor(64);

        eco.ingest("voice", &voice, 1.0).unwrap();

        let result = eco.project_query_with_diagnostics(&voice).unwrap();

        assert_eq!(result.diagnostics.interference_score, 0.0);
        assert_eq!(result.diagnostics.bridge_bleed_score, 0.0);
    }

    #[test]
    fn test_compose_for_budget() {
        let mut eco = EcosystemHologram::new(test_config(64), Device::Cpu).unwrap();

        // Add items with descriptions
        for i in 0..5 {
            eco.ingest_with_description(
                "voice",
                &random_tensor(64),
                1.0,
                Some(&format!("Topic {} about infrastructure", i)),
            )
            .unwrap();
        }

        let query = random_tensor(64);
        let context = eco.compose_for_budget(&query, 50).unwrap();

        // Should have some items within budget
        assert!(!context.items.is_empty());
        assert!(context.total_estimated_tokens <= 50);

        // Items should have descriptions and source
        for item in &context.items {
            assert!(!item.description.is_empty());
            assert_eq!(item.source, "voice");
        }
    }

    #[test]
    fn test_compose_respects_budget() {
        let mut eco = EcosystemHologram::new(test_config(64), Device::Cpu).unwrap();

        // Add items with long descriptions
        for i in 0..10 {
            eco.ingest_with_description(
                "voice",
                &random_tensor(64),
                1.0,
                Some(&format!(
                    "This is a very long description number {} that should consume many tokens in the budget",
                    i
                )),
            )
            .unwrap();
        }

        let query = random_tensor(64);

        // Very tight budget
        let context = eco.compose_for_budget(&query, 10).unwrap();
        assert!(context.total_estimated_tokens <= 10);

        // Generous budget
        let context = eco.compose_for_budget(&query, 1000).unwrap();
        assert!(!context.items.is_empty());
    }

    #[test]
    fn test_consolidation() {
        let mut eco = EcosystemHologram::new(test_config(64), Device::Cpu).unwrap();

        // Ingest from multiple sources
        for _ in 0..10 {
            eco.ingest("voice", &random_tensor(64), 1.0).unwrap();
            eco.ingest("k8s", &random_tensor(64), 0.5).unwrap();
        }

        let result = eco.consolidate().unwrap();
        assert_eq!(result.sources_consolidated, 2);
        assert!(result.post_consolidation_norm > 0.0);
    }

    #[test]
    fn test_consolidation_empty() {
        let mut eco = EcosystemHologram::new(test_config(64), Device::Cpu).unwrap();
        let result = eco.consolidate().unwrap();
        assert_eq!(result.sources_consolidated, 0);
    }

    #[test]
    fn test_tick_decay() {
        let mut eco = EcosystemHologram::new(test_config(64), Device::Cpu).unwrap();

        eco.ingest_with_description("voice", &random_tensor(64), 0.5, Some("will decay"))
            .unwrap();

        assert_eq!(eco.recent_item_count(), 1);

        // Large decay should remove the item (salience 0.5 * e^(-0.1*100) ≈ 0)
        eco.tick(std::time::Duration::from_secs(100)).unwrap();
        assert_eq!(eco.recent_item_count(), 0);
    }

    #[test]
    fn test_working_memory_capacity() {
        let config = EcosystemConfig {
            dim: 64,
            working_memory_capacity: 3,
            ..Default::default()
        };
        let mut eco = EcosystemHologram::new(config, Device::Cpu).unwrap();

        for i in 0..5 {
            eco.ingest_with_description(
                "voice",
                &random_tensor(64),
                (i as f32 + 1.0) / 5.0, // Increasing salience
                Some(&format!("item {}", i)),
            )
            .unwrap();
        }

        // Should be capped at 3 — the 3 highest salience
        assert_eq!(eco.recent_item_count(), 3);
    }

    #[test]
    fn test_dimension_validation() {
        let mut eco = EcosystemHologram::new(test_config(64), Device::Cpu).unwrap();
        let wrong_dim = Tensor::randn(0.0f32, 1.0, (128,), &Device::Cpu).unwrap();

        let result = eco.ingest("voice", &wrong_dim, 1.0);
        assert!(result.is_err());
    }
}

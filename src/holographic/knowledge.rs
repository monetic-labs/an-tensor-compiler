//! HRR knowledge-space primitives and boundary-enforced projection semantics.
//!
//! These types generalize the schema layer above the HRR algebra so that code,
//! documents, business records, workflows, telemetry, and other datasets can
//! share the same memory substrate without inheriting code-specific structure.

use crate::holographic::bridges::BridgeBinding;
use crate::holographic::ops::superimpose;
use crate::holographic::policy::{ProjectionMode, ProjectionPolicy};
use crate::{Result, TensorCoreError};
use candle_core::Tensor;
use chrono::{DateTime, Utc};
use std::collections::{HashMap, HashSet, VecDeque};

/// Neutral tensor container for a named knowledge space.
#[derive(Debug, Clone)]
pub struct KnowledgeSpaceTensor {
    /// Stable identifier for the space.
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Description of the space contents and intent.
    pub description: String,
    /// Broad category for the knowledge in this space.
    pub kind: KnowledgeSpaceKind,
    /// Composed hologram for the space.
    pub hologram: Tensor,
    /// Items contained in the space.
    pub items: Vec<KnowledgeItemRef>,
    /// Boundary behavior for projection into and out of the space.
    pub boundaries: SpaceBoundaryPolicy,
    /// Shared provenance metadata for the space.
    pub provenance: SpaceProvenance,
    /// Freshness and temporal policy for the space.
    pub temporal: SpaceTemporalPolicy,
    /// Default salience behavior for items in the space.
    pub salience: SalienceDefaults,
    /// Geometric cell-locality index over this space's items (HRR-G2).
    /// `None` for spaces built without geometric indexing or with too few
    /// items; projection falls back to linear scan over `items`.
    pub geometric_index: Option<crate::holographic::GeometricIndex>,
    /// Extension metadata for non-core annotations.
    pub metadata: HashMap<String, String>,
}

/// Neutral top-level view over one or more knowledge spaces.
#[derive(Debug, Clone)]
pub struct KnowledgeSystemView {
    /// Stable identifier for the system view.
    pub id: String,
    /// Human-readable system name.
    pub name: String,
    /// Description of the system and its spaces.
    pub description: String,
    /// Composed hologram for the system.
    pub hologram: Tensor,
    /// Spaces participating in the system.
    pub spaces: Vec<KnowledgeSpaceTensor>,
    /// Explicit bridges available between spaces.
    pub bridges: Vec<BridgeBinding>,
    /// Extension metadata for non-core annotations.
    pub metadata: HashMap<String, String>,
}

impl KnowledgeSystemView {
    /// Get a space by its identifier.
    pub fn get_space(&self, id: &str) -> Option<&KnowledgeSpaceTensor> {
        self.spaces.iter().find(|space| space.id == id)
    }

    /// Get a bridge by its identifier.
    pub fn get_bridge(&self, id: &str) -> Option<&BridgeBinding> {
        self.bridges.iter().find(|bridge| bridge.id == id)
    }

    /// Inspect bridge availability for a given projection policy without
    /// running an actual projection.
    ///
    /// The substrate enforces a strict semantics for
    /// [`ProjectionMode::Bridged`]: the projection errors when its
    /// allowed-bridge set does not connect at least two spaces in the
    /// view. That strictness is correct — `Bridged` declares a contract
    /// with the corpus that bridges are present and meaningful — but it
    /// is hostile to consumers that cannot tell a priori whether a given
    /// view was built with the bridges their policy assumes.
    ///
    /// `inspect_bridges` exposes the substrate's own pre-flight view:
    /// which bridge ids the policy declares, which of those exist as
    /// active bridges in this view, which spaces are reachable from the
    /// policy's start space within the policy's hop budget, and a
    /// `viable` flag answering "would `Bridged` succeed against this
    /// view?" without paying the query-encode + projection cost.
    ///
    /// Consumers that build their policies under uncertainty (e.g. mixed
    /// corpus maturity across deployments) should call this before
    /// dispatching `ProjectionMode::Bridged`; on a non-viable result, the
    /// honest fallback is `ProjectionMode::MultiSpace` over the same
    /// allowed spaces. The substrate intentionally does not silently
    /// downgrade — `Bridged` and `MultiSpace` mean different things and
    /// the consumer should know which it got.
    ///
    /// Returns availability for non-`Bridged` policies as well; spaces
    /// reachable in those modes is just the intersection of `allowed_spaces`
    /// with present spaces and `bridges` fields are empty.
    pub fn inspect_bridges(&self, policy: &ProjectionPolicy) -> BridgeAvailability {
        let active_bridge_ids: HashSet<&str> = self
            .bridges
            .iter()
            .filter(|bridge| bridge.active)
            .map(|bridge| bridge.id.as_str())
            .collect();

        let declared_bridges: Vec<String> = policy.allowed_bridges.clone();
        let active_bridges: Vec<String> = policy
            .allowed_bridges
            .iter()
            .filter(|id| active_bridge_ids.contains(id.as_str()))
            .cloned()
            .collect();
        let missing_bridges: Vec<String> = policy
            .allowed_bridges
            .iter()
            .filter(|id| !active_bridge_ids.contains(id.as_str()))
            .cloned()
            .collect();

        let present_space_ids: HashSet<&str> = self.spaces.iter().map(|s| s.id.as_str()).collect();
        let allowed_space_ids: HashSet<&str> =
            policy.allowed_spaces.iter().map(String::as_str).collect();
        let missing_spaces: Vec<String> = policy
            .allowed_spaces
            .iter()
            .filter(|id| !present_space_ids.contains(id.as_str()))
            .cloned()
            .collect();

        let (reachable_spaces, unreachable_allowed_spaces) = match policy.mode {
            ProjectionMode::Bridged => {
                if policy.allowed_spaces.is_empty() {
                    (Vec::new(), Vec::new())
                } else {
                    let active_allowed_bridge_ids: HashSet<&str> =
                        active_bridges.iter().map(String::as_str).collect();
                    let start_space_id = policy.allowed_spaces[0].clone();
                    let mut reached: HashSet<String> = HashSet::new();
                    let mut queue: VecDeque<(String, usize)> = VecDeque::new();
                    if present_space_ids.contains(start_space_id.as_str()) {
                        queue.push_back((start_space_id.clone(), 0));
                        reached.insert(start_space_id);
                    }

                    while let Some((space_id, hops)) = queue.pop_front() {
                        if hops >= policy.max_space_hops {
                            continue;
                        }
                        for bridge in &self.bridges {
                            if !bridge.active
                                || bridge.from_space != space_id
                                || !active_allowed_bridge_ids.contains(bridge.id.as_str())
                                || !allowed_space_ids.contains(bridge.to_space.as_str())
                            {
                                continue;
                            }
                            if reached.insert(bridge.to_space.clone()) {
                                queue.push_back((bridge.to_space.clone(), hops + 1));
                            }
                        }
                    }

                    let mut reachable: Vec<String> = reached.iter().cloned().collect();
                    reachable.sort();
                    let unreachable: Vec<String> = policy
                        .allowed_spaces
                        .iter()
                        .filter(|id| !reached.contains(id.as_str()))
                        .cloned()
                        .collect();
                    (reachable, unreachable)
                }
            }
            ProjectionMode::SingleSpace | ProjectionMode::MultiSpace | ProjectionMode::Auto => {
                let mut reachable: Vec<String> = policy
                    .allowed_spaces
                    .iter()
                    .filter(|id| present_space_ids.contains(id.as_str()))
                    .cloned()
                    .collect();
                reachable.sort();
                let unreachable: Vec<String> = policy
                    .allowed_spaces
                    .iter()
                    .filter(|id| !present_space_ids.contains(id.as_str()))
                    .cloned()
                    .collect();
                (reachable, unreachable)
            }
        };

        let viable = match policy.mode {
            ProjectionMode::Bridged => reachable_spaces.len() >= 2,
            ProjectionMode::SingleSpace => reachable_spaces.len() == 1,
            ProjectionMode::MultiSpace => reachable_spaces.len() >= 2,
            ProjectionMode::Auto => !self.spaces.is_empty(),
        };

        BridgeAvailability {
            mode: policy.mode,
            declared_bridges,
            active_bridges,
            missing_bridges,
            reachable_spaces,
            unreachable_allowed_spaces,
            missing_spaces,
            viable,
        }
    }

    /// Compose multiple system views into a single view at query time.
    ///
    /// This is the substrate-level mechanism for assembling a runtime
    /// projection target from independently-loaded corpora — for example, a
    /// shared deployment-wide view plus one or more consumer-scoped views.
    /// The composed view exposes the union of spaces and bridges; the
    /// composed hologram is the (optionally weighted) superposition of the
    /// inputs' holograms, which means policy-driven projection (per-space
    /// weights, allowed spaces, bridge gating) continues to apply normally.
    ///
    /// Composition is governed by HRR algebra, not by any consumer ontology:
    ///
    /// - All inputs must share the same hologram dimensionality. Mixing
    ///   different embedding geometries in one composition is mathematically
    ///   unsound (different spaces would be incomparable under cosine
    ///   similarity), so it returns an error rather than silently producing
    ///   wrong scores.
    /// - Space and bridge ids must be unique across inputs. Two inputs
    ///   contributing the same id is treated as a configuration error and
    ///   returned as an error rather than resolved by precedence — the
    ///   substrate refuses to guess which definition the consumer meant.
    /// - `weights`, when supplied, must be the same length as `views` and
    ///   contain only finite, non-negative values. When omitted the inputs
    ///   are superimposed with equal weight. Per-space attenuation at
    ///   projection time is a separate concern handled by
    ///   [`crate::holographic::ProjectionPolicy::space_weights`].
    /// - Metadata maps from each input are merged into the composed view.
    ///   Conflicting keys are an error so the composition cannot silently
    ///   drop or overwrite consumer-supplied annotations.
    ///
    /// **Cross-scope bridges**: a bridge declared on one input view that
    /// targets a space living on a different input view composes
    /// naturally — both views participate in the resulting composition,
    /// so the bridge's `from_space`/`to_space` resolve as expected during
    /// projection. This is deliberate: it lets a consumer-scoped corpus
    /// declare a bridge against a shared-corpus concept without any
    /// substrate-side cross-scope-bridge type. Bridge gating
    /// (`allowed_bridge_ids`, hop limits, confidence penalty) continues
    /// to apply through `ProjectionPolicy` exactly as for in-view bridges.
    pub fn compose(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        views: &[&KnowledgeSystemView],
        weights: Option<&[f32]>,
    ) -> Result<Self> {
        if views.is_empty() {
            return Err(TensorCoreError::Projection(
                "KnowledgeSystemView::compose requires at least one input view".into(),
            ));
        }

        if let Some(weights) = weights {
            if weights.len() != views.len() {
                return Err(TensorCoreError::Projection(format!(
                    "KnowledgeSystemView::compose received {} views but {} weights",
                    views.len(),
                    weights.len()
                )));
            }
            for (idx, weight) in weights.iter().enumerate() {
                if !weight.is_finite() || *weight < 0.0 {
                    return Err(TensorCoreError::Projection(format!(
                        "KnowledgeSystemView::compose weight[{}] must be finite and \
                         non-negative, got {}",
                        idx, weight
                    )));
                }
            }
        }

        let dim_dims = views[0].hologram.dims().first().copied().ok_or_else(|| {
            TensorCoreError::Projection(
                "KnowledgeSystemView::compose: input view 0 has a zero-dim hologram".into(),
            )
        })?;
        for (idx, view) in views.iter().enumerate().skip(1) {
            let dim = view.hologram.dims().first().copied().ok_or_else(|| {
                TensorCoreError::Projection(format!(
                    "KnowledgeSystemView::compose: input view {} has a zero-dim hologram",
                    idx
                ))
            })?;
            if dim != dim_dims {
                return Err(TensorCoreError::Projection(format!(
                    "KnowledgeSystemView::compose: input view {} has dim {} but view 0 \
                     has dim {}; cross-encoding composition is not supported",
                    idx, dim, dim_dims
                )));
            }
        }

        let holograms: Vec<Tensor> = views.iter().map(|view| view.hologram.clone()).collect();
        let composed_hologram = superimpose(&holograms, weights)?;

        let mut spaces: Vec<KnowledgeSpaceTensor> = Vec::new();
        let mut bridges: Vec<BridgeBinding> = Vec::new();
        let mut seen_space_ids: HashSet<String> = HashSet::new();
        let mut seen_bridge_ids: HashSet<String> = HashSet::new();
        let mut composed_metadata: HashMap<String, String> = HashMap::new();

        for (view_idx, view) in views.iter().enumerate() {
            for space in &view.spaces {
                if !seen_space_ids.insert(space.id.clone()) {
                    return Err(TensorCoreError::Projection(format!(
                        "KnowledgeSystemView::compose: duplicate space id '{}' across \
                         input views (input {} re-introduces it). Resolve in the \
                         consumer-side composition step, not the substrate.",
                        space.id, view_idx
                    )));
                }
                spaces.push(space.clone());
            }

            for bridge in &view.bridges {
                if !seen_bridge_ids.insert(bridge.id.clone()) {
                    return Err(TensorCoreError::Projection(format!(
                        "KnowledgeSystemView::compose: duplicate bridge id '{}' across \
                         input views (input {} re-introduces it). Resolve in the \
                         consumer-side composition step, not the substrate.",
                        bridge.id, view_idx
                    )));
                }
                bridges.push(bridge.clone());
            }

            for (key, value) in &view.metadata {
                if let Some(existing) = composed_metadata.get(key) {
                    if existing != value {
                        return Err(TensorCoreError::Projection(format!(
                            "KnowledgeSystemView::compose: conflicting metadata for key \
                             '{}' across input views ('{}' vs '{}'). Resolve in the \
                             consumer-side composition step, not the substrate.",
                            key, existing, value
                        )));
                    }
                } else {
                    composed_metadata.insert(key.clone(), value.clone());
                }
            }
        }

        Ok(Self {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            hologram: composed_hologram,
            spaces,
            bridges,
            metadata: composed_metadata,
        })
    }
}

/// Broad category of knowledge represented in a space.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KnowledgeSpaceKind {
    Code,
    Business,
    Document,
    Workflow,
    Telemetry,
    UserState,
    Conversation,
    External,
    Custom(String),
}

/// Neutral reference to a space item.
#[derive(Debug, Clone)]
pub struct KnowledgeItemRef {
    /// Stable identifier for the item.
    pub id: String,
    /// High-level item category.
    pub item_type: KnowledgeItemType,
    /// HRR representation for the item.
    pub hologram: Tensor,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Provenance metadata for the item.
    pub provenance: ItemProvenance,
}

/// Category of item stored inside a knowledge space.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KnowledgeItemType {
    Component,
    Entity,
    Relation,
    Document,
    Record,
    Event,
    Summary,
    Custom(String),
}

/// Boundary rules for a knowledge space.
#[derive(Debug, Clone, Default)]
pub struct SpaceBoundaryPolicy {
    /// Whether projection may leave this space without an explicit bridge.
    pub allow_unbridged_projection: bool,
    /// Space IDs that may project into this space directly.
    pub allowed_inbound_spaces: Vec<String>,
    /// Space IDs this space may project into directly.
    pub allowed_outbound_spaces: Vec<String>,
    /// Bridge IDs allowed for this space.
    pub allowed_bridge_ids: Vec<String>,
}

/// Shared provenance metadata for a knowledge space.
#[derive(Debug, Clone, Default)]
pub struct SpaceProvenance {
    /// Source identifiers contributing to the space.
    pub sources: Vec<String>,
    /// Declared authority for the space, when relevant.
    pub authority: Option<String>,
    /// Refresh cadence string (for example `5m`, `1h`, `daily`).
    pub refresh_cadence: Option<String>,
    /// Encoding or ingestion policy identifier.
    pub encoding_policy: Option<String>,
    /// Last successful refresh time.
    pub updated_at: Option<DateTime<Utc>>,
}

/// Provenance metadata for a single item.
#[derive(Debug, Clone, Default)]
pub struct ItemProvenance {
    /// Primary source identifier for the item.
    pub source_id: String,
    /// Declared authority for the item, if any.
    pub authority: Option<String>,
    /// Collection timestamp.
    pub collected_at: Option<DateTime<Utc>>,
    /// Last update timestamp.
    pub updated_at: Option<DateTime<Utc>>,
    /// Confidence in the item contents.
    pub confidence: Option<f32>,
    /// Prior artifacts or source identifiers contributing to this item.
    pub lineage: Vec<String>,
}

/// Temporal state attached to a knowledge item.
#[derive(Debug, Clone, Default)]
pub struct TemporalState {
    /// Observation timestamp for the item.
    pub observed_at: Option<DateTime<Utc>>,
    /// Start of the item's effective interval.
    pub effective_from: Option<DateTime<Utc>>,
    /// End of the item's effective interval.
    pub effective_to: Option<DateTime<Utc>>,
    /// Expiration timestamp when the item should be considered stale.
    pub expires_at: Option<DateTime<Utc>>,
}

/// Temporal policy for a knowledge space.
#[derive(Debug, Clone, Default)]
pub struct SpaceTemporalPolicy {
    /// Freshness window used to decide if data is stale.
    pub freshness_window_seconds: Option<u64>,
    /// Half-life for salience decay.
    pub decay_half_life_seconds: Option<u64>,
    /// Temporal resolution string, if the space is time-structured.
    pub temporal_resolution: Option<String>,
}

/// Default salience behavior for a knowledge space.
#[derive(Debug, Clone)]
pub struct SalienceDefaults {
    /// Default ingestion salience.
    pub default_salience: f32,
    /// Optional boost applied to fresh items.
    pub freshness_boost: f32,
    /// Optional penalty applied to stale items.
    pub staleness_penalty: f32,
}

impl Default for SalienceDefaults {
    fn default() -> Self {
        Self {
            default_salience: 1.0,
            freshness_boost: 0.0,
            staleness_penalty: 0.0,
        }
    }
}

/// Pre-flight bridge-availability report for a [`ProjectionPolicy`] over a
/// [`KnowledgeSystemView`]. Returned by
/// [`KnowledgeSystemView::inspect_bridges`].
///
/// Lets consumers detect whether `ProjectionMode::Bridged` would succeed
/// against the current view *before* paying the query-encode + projection
/// cost, and surfaces precisely which bridges/spaces the policy assumed
/// but the view does not provide. The substrate intentionally does not
/// silently downgrade `Bridged` to `MultiSpace` on absent bridges; this
/// report is the substrate's contract for letting the consumer make that
/// choice with full visibility.
#[derive(Debug, Clone, PartialEq)]
pub struct BridgeAvailability {
    /// The projection mode this report was computed for.
    pub mode: ProjectionMode,
    /// All bridge ids the policy declared as allowed.
    pub declared_bridges: Vec<String>,
    /// Subset of `declared_bridges` that exist as active bridges in the view.
    pub active_bridges: Vec<String>,
    /// Subset of `declared_bridges` that are absent or inactive in the view.
    pub missing_bridges: Vec<String>,
    /// Spaces reachable from the policy's start space (for `Bridged`) or
    /// simply present-and-allowed (for other modes), respecting the
    /// policy's hop budget.
    pub reachable_spaces: Vec<String>,
    /// Spaces in `policy.allowed_spaces` that are not reachable under
    /// the policy's mode + bridges + hop budget.
    pub unreachable_allowed_spaces: Vec<String>,
    /// Spaces in `policy.allowed_spaces` that simply do not exist in
    /// the view at all (subset of `unreachable_allowed_spaces`).
    pub missing_spaces: Vec<String>,
    /// Whether projecting under this policy against this view would
    /// satisfy the mode's reachability contract. The substrate does not
    /// promise it would return useful results — only that it would not
    /// fail with a reachability error.
    pub viable: bool,
}

impl BridgeAvailability {
    /// True when the policy declared at least one bridge but the view
    /// carries none of them. The canonical "corpus build did not
    /// materialize the bridges this policy assumes" signal.
    pub fn no_declared_bridges_active(&self) -> bool {
        !self.declared_bridges.is_empty() && self.active_bridges.is_empty()
    }
}

#[cfg(test)]
mod compose_tests {
    use super::*;
    use crate::holographic::bridges::{BridgeBinding, BridgeBindingType, BridgeProjectionPolicy};
    use crate::holographic::ops::normalize;
    use candle_core::Device;

    fn space(id: &str, dim: usize, device: &Device) -> KnowledgeSpaceTensor {
        let hologram = normalize(&Tensor::randn(0.0f32, 1.0, (dim,), device).unwrap()).unwrap();
        KnowledgeSpaceTensor {
            id: id.to_string(),
            name: id.to_string(),
            description: format!("{id} space"),
            kind: KnowledgeSpaceKind::Custom("test".into()),
            hologram,
            items: Vec::new(),
            boundaries: SpaceBoundaryPolicy::default(),
            provenance: SpaceProvenance::default(),
            temporal: SpaceTemporalPolicy::default(),
            salience: SalienceDefaults::default(),
            geometric_index: None,
            metadata: HashMap::new(),
        }
    }

    fn view(
        id: &str,
        dim: usize,
        device: &Device,
        spaces: Vec<KnowledgeSpaceTensor>,
    ) -> KnowledgeSystemView {
        let hologram = if spaces.is_empty() {
            normalize(&Tensor::randn(0.0f32, 1.0, (dim,), device).unwrap()).unwrap()
        } else {
            crate::holographic::ops::superimpose(
                &spaces
                    .iter()
                    .map(|s| s.hologram.clone())
                    .collect::<Vec<_>>(),
                None,
            )
            .unwrap()
        };
        KnowledgeSystemView {
            id: id.to_string(),
            name: id.to_string(),
            description: format!("{id} view"),
            hologram,
            spaces,
            bridges: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn compose_unions_spaces_and_metadata_across_views() {
        let device = Device::Cpu;
        let dim = 64;
        let shared = view("shared", dim, &device, vec![space("a", dim, &device)]);
        let mut tenant = view("tenant", dim, &device, vec![space("b", dim, &device)]);
        tenant.metadata.insert("tenant_kind".into(), "test".into());

        let composed =
            KnowledgeSystemView::compose("composed", "composed", "test", &[&shared, &tenant], None)
                .unwrap();

        assert_eq!(composed.spaces.len(), 2);
        assert!(composed.get_space("a").is_some());
        assert!(composed.get_space("b").is_some());
        assert_eq!(
            composed.metadata.get("tenant_kind").map(String::as_str),
            Some("test")
        );
    }

    #[test]
    fn compose_rejects_duplicate_space_ids() {
        let device = Device::Cpu;
        let dim = 64;
        let shared = view("shared", dim, &device, vec![space("dup", dim, &device)]);
        let tenant = view("tenant", dim, &device, vec![space("dup", dim, &device)]);

        let err = KnowledgeSystemView::compose("c", "c", "c", &[&shared, &tenant], None)
            .expect_err("duplicate space id must error");
        let msg = format!("{err}");
        assert!(msg.contains("duplicate space id"), "got: {msg}");
    }

    #[test]
    fn compose_rejects_duplicate_bridge_ids() {
        let device = Device::Cpu;
        let dim = 64;
        let mut a = view("shared", dim, &device, vec![space("x", dim, &device)]);
        let mut b = view("tenant", dim, &device, vec![space("y", dim, &device)]);
        let bridge = BridgeBinding {
            id: "bridge::dup".into(),
            from_space: "x".into(),
            to_space: "y".into(),
            binding_type: BridgeBindingType::Custom("test".into()),
            policy: BridgeProjectionPolicy {
                allow_automatic_projection: true,
                require_attribution: true,
                confidence_penalty: 0.0,
                max_traversals: 1,
            },
            provenance: ItemProvenance::default(),
            active: true,
        };
        a.bridges.push(bridge.clone());
        b.bridges.push(bridge);

        let err = KnowledgeSystemView::compose("c", "c", "c", &[&a, &b], None)
            .expect_err("duplicate bridge id must error");
        assert!(format!("{err}").contains("duplicate bridge id"));
    }

    #[test]
    fn compose_rejects_dim_mismatch() {
        let device = Device::Cpu;
        let a = view("a", 64, &device, vec![space("a", 64, &device)]);
        let b = view("b", 32, &device, vec![space("b", 32, &device)]);

        let err = KnowledgeSystemView::compose("c", "c", "c", &[&a, &b], None)
            .expect_err("dim mismatch must error");
        assert!(format!("{err}").contains("dim"));
    }

    #[test]
    fn compose_rejects_weight_length_mismatch() {
        let device = Device::Cpu;
        let dim = 64;
        let a = view("a", dim, &device, vec![space("a", dim, &device)]);
        let b = view("b", dim, &device, vec![space("b", dim, &device)]);

        let err = KnowledgeSystemView::compose("c", "c", "c", &[&a, &b], Some(&[0.5]))
            .expect_err("weight length mismatch must error");
        assert!(format!("{err}").contains("weights"));
    }

    #[test]
    fn compose_rejects_negative_or_nan_weights() {
        let device = Device::Cpu;
        let dim = 64;
        let a = view("a", dim, &device, vec![space("a", dim, &device)]);
        let b = view("b", dim, &device, vec![space("b", dim, &device)]);

        assert!(
            KnowledgeSystemView::compose("c", "c", "c", &[&a, &b], Some(&[1.0, -0.1])).is_err()
        );
        assert!(
            KnowledgeSystemView::compose("c", "c", "c", &[&a, &b], Some(&[1.0, f32::NAN])).is_err()
        );
    }

    fn bridged_view_with_one_active_bridge(dim: usize, device: &Device) -> KnowledgeSystemView {
        use crate::holographic::bridges::{BridgeBindingType, BridgeProjectionPolicy};

        let mut a = view(
            "workflows",
            dim,
            device,
            vec![space("workflows", dim, device)],
        );
        let b = view(
            "capabilities",
            dim,
            device,
            vec![space("capabilities", dim, device)],
        );

        let bridge = BridgeBinding {
            id: "capabilities_to_workflows".into(),
            from_space: "workflows".into(),
            to_space: "capabilities".into(),
            binding_type: BridgeBindingType::Custom("acme".into()),
            policy: BridgeProjectionPolicy {
                allow_automatic_projection: true,
                require_attribution: true,
                confidence_penalty: 0.1,
                max_traversals: 1,
            },
            provenance: ItemProvenance::default(),
            active: true,
        };
        a.bridges.push(bridge);

        KnowledgeSystemView::compose("composed", "composed", "test", &[&a, &b], None).unwrap()
    }

    #[test]
    fn inspect_bridges_reports_no_active_bridges_when_corpus_is_bare() {
        let device = Device::Cpu;
        let dim = 64;
        let view_no_bridges = KnowledgeSystemView::compose(
            "composed",
            "composed",
            "test",
            &[
                &view(
                    "workflows",
                    dim,
                    &device,
                    vec![space("workflows", dim, &device)],
                ),
                &view(
                    "capabilities",
                    dim,
                    &device,
                    vec![space("capabilities", dim, &device)],
                ),
                &view("docs", dim, &device, vec![space("docs", dim, &device)]),
            ],
            None,
        )
        .unwrap();

        let policy = ProjectionPolicy::bridged(
            vec!["workflows".into(), "capabilities".into(), "docs".into()],
            vec![
                "capabilities_to_workflows".into(),
                "docs_to_workflows".into(),
            ],
        );

        let availability = view_no_bridges.inspect_bridges(&policy);

        assert_eq!(availability.mode, ProjectionMode::Bridged);
        assert_eq!(
            availability.declared_bridges,
            vec![
                "capabilities_to_workflows".to_string(),
                "docs_to_workflows".to_string()
            ]
        );
        assert!(availability.active_bridges.is_empty());
        assert_eq!(availability.missing_bridges.len(), 2);
        assert!(availability.no_declared_bridges_active());
        assert!(
            !availability.viable,
            "Bridged with zero active bridges must report viable=false"
        );
    }

    #[test]
    fn inspect_bridges_reports_partial_coverage() {
        let device = Device::Cpu;
        let dim = 64;
        let composed = bridged_view_with_one_active_bridge(dim, &device);

        let policy = ProjectionPolicy::bridged(
            vec!["workflows".into(), "capabilities".into()],
            vec![
                "capabilities_to_workflows".into(),
                "docs_to_workflows".into(),
            ],
        );

        let availability = composed.inspect_bridges(&policy);

        assert_eq!(
            availability.active_bridges,
            vec!["capabilities_to_workflows".to_string()]
        );
        assert_eq!(
            availability.missing_bridges,
            vec!["docs_to_workflows".to_string()]
        );
        assert!(
            availability.viable,
            "viable should be true once at least one bridge connects two allowed spaces"
        );
    }

    #[test]
    fn inspect_bridges_reports_missing_spaces() {
        let device = Device::Cpu;
        let dim = 64;
        let composed = bridged_view_with_one_active_bridge(dim, &device);

        let policy = ProjectionPolicy::bridged(
            vec!["workflows".into(), "capabilities".into(), "absent".into()],
            vec!["capabilities_to_workflows".into()],
        );

        let availability = composed.inspect_bridges(&policy);

        assert_eq!(availability.missing_spaces, vec!["absent".to_string()]);
        assert!(availability
            .unreachable_allowed_spaces
            .iter()
            .any(|s| s == "absent"));
    }

    #[test]
    fn inspect_bridges_for_multi_space_ignores_bridges() {
        let device = Device::Cpu;
        let dim = 64;
        let composed = bridged_view_with_one_active_bridge(dim, &device);

        let policy = ProjectionPolicy::multi_space(vec!["workflows".into(), "capabilities".into()]);

        let availability = composed.inspect_bridges(&policy);

        assert_eq!(availability.mode, ProjectionMode::MultiSpace);
        assert!(availability.viable);
        assert!(availability.declared_bridges.is_empty());
        assert!(availability.active_bridges.is_empty());
    }

    #[test]
    fn compose_rejects_conflicting_metadata() {
        let device = Device::Cpu;
        let dim = 64;
        let mut a = view("a", dim, &device, vec![space("a", dim, &device)]);
        let mut b = view("b", dim, &device, vec![space("b", dim, &device)]);
        a.metadata.insert("policy".into(), "strict".into());
        b.metadata.insert("policy".into(), "lenient".into());

        let err = KnowledgeSystemView::compose("c", "c", "c", &[&a, &b], None)
            .expect_err("conflicting metadata must error");
        assert!(format!("{err}").contains("conflicting metadata"));
    }
}

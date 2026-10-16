//! Context manifest parsing
//!
//! Parses the `contexts.toml` file that defines an organism's
//! bounded contexts and their relationships.

use crate::holographic::bridges::{BridgeBinding, BridgeBindingType, BridgeProjectionPolicy};
use crate::holographic::knowledge::{ItemProvenance, KnowledgeSpaceKind};
use crate::{Result, TensorCoreError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Full context manifest (parsed from contexts.toml)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextManifest {
    /// Organism info
    pub organism: OrganismInfo,

    /// Encoding configuration
    pub encoding: EncodingInfo,

    /// Bounded contexts
    #[serde(default)]
    pub contexts: Vec<ContextInfo>,

    /// Boundary rules
    #[serde(default)]
    pub boundaries: HashMap<String, Vec<String>>,

    /// Federation configuration
    #[serde(default)]
    pub federation: FederationInfo,

    /// History configuration
    #[serde(default)]
    pub history: HistoryInfo,

    /// Codegen configuration
    #[serde(default)]
    pub codegen: CodegenInfo,
}

/// Neutral top-level identity info for a generalized knowledge system.
pub type SystemInfo = OrganismInfo;

/// Generalized manifest for knowledge spaces, bridges, and projection defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeManifest {
    /// Top-level system metadata.
    pub system: SystemInfo,
    /// Shared encoding configuration.
    pub encoding: EncodingInfo,
    /// Declared knowledge spaces.
    #[serde(default)]
    pub spaces: Vec<KnowledgeSpaceInfo>,
    /// Explicit cross-space bridges.
    #[serde(default)]
    pub bridges: Vec<BridgeInfo>,
    /// Default projection behavior for the system.
    #[serde(default)]
    pub projection: ProjectionDefaults,
    /// Holographic sigil — per-QIP watermark and ASCII seal. `None` on
    /// manifests written before sigils were introduced; `Some` with
    /// empty fields when the compiler could not derive a sigil (e.g. a
    /// program with no spaces).
    #[serde(default)]
    pub sigil: Option<SigilInfo>,
}

impl KnowledgeManifest {
    /// Load a neutral knowledge manifest from file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(TensorCoreError::Io)?;

        toml::from_str(&content)
            .map_err(|e| TensorCoreError::Config(format!("Failed to parse manifest: {}", e)))
    }

    /// Write this manifest as pretty TOML. Pair of [`Self::load`].
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| TensorCoreError::Config(format!("Failed to serialize manifest: {}", e)))?;
        std::fs::write(path.as_ref(), content).map_err(TensorCoreError::Io)?;
        Ok(())
    }

    /// Find a neutral manifest in a project directory.
    ///
    /// Looks for:
    /// 1. `.an-ecosystem/knowledge.toml`
    /// 2. `knowledge.toml`
    /// 3. `.knowledge-manifest.toml`
    pub fn find(project_root: impl AsRef<Path>) -> Option<Self> {
        let root = project_root.as_ref();

        let candidates = [
            root.join(".an-ecosystem/knowledge.toml"),
            root.join("knowledge.toml"),
            root.join(".knowledge-manifest.toml"),
        ];

        for path in candidates {
            if path.exists() {
                if let Ok(manifest) = Self::load(&path) {
                    return Some(manifest);
                }
            }
        }

        None
    }

    /// Get space info by name.
    pub fn get_space(&self, id: &str) -> Option<&KnowledgeSpaceInfo> {
        self.spaces.iter().find(|space| space.id == id)
    }

    /// Return all declared space identifiers.
    pub fn space_ids(&self) -> Vec<&str> {
        self.spaces.iter().map(|space| space.id.as_str()).collect()
    }

    /// Materialize the manifest's `BridgeInfo` entries into runtime
    /// `BridgeBinding`s using substrate-blessed defaults.
    ///
    /// This is the canonical path for any consumer that loads a
    /// `KnowledgeManifest` and assembles a `KnowledgeSystemView` —
    /// inference servers, in-process projectors, anything else. Going
    /// through this method ensures the runtime bridges interpret
    /// declared-in-manifest as "active and traversable" by default,
    /// avoiding the trap of constructing `BridgeBinding`s by hand and
    /// landing on `BridgeProjectionPolicy::default()`'s
    /// `allow_automatic_projection: false` posture (which is correct
    /// for programmatically-constructed bridges with no consumer
    /// intent attached, but wrong for manifest-declared ones).
    pub fn to_bridge_bindings(&self) -> Vec<BridgeBinding> {
        self.bridges.iter().map(BridgeBinding::from).collect()
    }

    /// Explicit migration path from the legacy context manifest model.
    ///
    /// This is intentionally opt-in so the v2 manifest surface does not
    /// silently inherit legacy semantics.
    pub fn from_context_manifest(value: &ContextManifest) -> Self {
        let spaces = value
            .contexts
            .iter()
            .map(|context| KnowledgeSpaceInfo::from_context(value, context))
            .collect();

        let bridges = value
            .federation
            .imports
            .iter()
            .map(|import| BridgeInfo {
                id: format!("federation::{}", import.from),
                from_space: import.from.clone(),
                to_space: value.organism.id.clone(),
                binding_type: "federation_import".to_string(),
                enabled: true,
                metadata: HashMap::from([("items".to_string(), import.items.join(","))]),
                ..BridgeInfo::default()
            })
            .collect();

        Self {
            system: value.organism.clone(),
            encoding: value.encoding.clone(),
            spaces,
            bridges,
            projection: ProjectionDefaults::default(),
            sigil: None,
        }
    }
}

impl ContextManifest {
    /// Load manifest from file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(TensorCoreError::Io)?;

        toml::from_str(&content)
            .map_err(|e| TensorCoreError::Config(format!("Failed to parse manifest: {}", e)))
    }

    /// Find manifest in a project directory
    ///
    /// Looks for:
    /// 1. `.an-ecosystem/contexts.toml`
    /// 2. `contexts.toml`
    /// 3. `.context-manifest.toml`
    pub fn find(project_root: impl AsRef<Path>) -> Option<Self> {
        let root = project_root.as_ref();

        let candidates = [
            root.join(".an-ecosystem/contexts.toml"),
            root.join("contexts.toml"),
            root.join(".context-manifest.toml"),
        ];

        for path in candidates {
            if path.exists() {
                if let Ok(manifest) = Self::load(&path) {
                    return Some(manifest);
                }
            }
        }

        None
    }

    /// Get context info by name
    pub fn get_context(&self, name: &str) -> Option<&ContextInfo> {
        self.contexts.iter().find(|c| c.name == name)
    }

    /// Check if a dependency is allowed
    pub fn can_depend(&self, from: &str, to: &str) -> bool {
        self.boundaries
            .get(from)
            .map(|deps| deps.contains(&to.to_string()))
            .unwrap_or(false)
    }

    /// Get all context names
    pub fn context_names(&self) -> Vec<&str> {
        self.contexts.iter().map(|c| c.name.as_str()).collect()
    }
}

/// Organism metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismInfo {
    /// Unique identifier (e.g., "an-ecosystem")
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Description
    pub description: String,

    /// Version
    #[serde(default = "default_version")]
    pub version: String,
}

fn default_version() -> String {
    "0.1.0".to_string()
}

/// Provenance of the embedder used to compile a memory program.
///
/// Records the exact identity of the embedder so a cold-start agent
/// (or a query-time runtime) can verify it is projecting queries into the
/// same geometry the corpus was compiled against. Without this, two corpora
/// compiled against two different snapshots of the same model id would have
/// indistinguishable `[encoding]` tables but incompatible holograms — a
/// silent geometry mismatch.
///
/// All fields are optional and default to empty so that manifests written
/// before this table existed still parse. A runtime that finds
/// `content_hash` empty MUST treat the embedder identity as unverified and
/// SHOULD refuse to project queries (or warn loudly) rather than rank
/// against a possibly-mismatched geometry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbedderProvenance {
    /// Content hash of the embedder weights, namespaced
    /// (e.g. `"sha256://8aaa29b9..."`). Sourced from the inference
    /// server's provenance record at compile time.
    #[serde(default)]
    pub content_hash: String,
    /// Canonical source URI for the weights. Valid forms:
    /// - Hugging Face repo id
    /// - local `file://` path (machine-local warm/hydrate tree)
    /// - beacon URL
    /// - program-relative sibling `+embedder/<id>` (Procedure 2: weights
    ///   next to the four `memory_program_spec/5` files; portable, not a
    ///   host path). See `MEMORY_PROGRAM_SPEC.md` §5.
    ///
    /// A cold-start agent uses this to fetch or load the weights when no
    /// embedder is already in the runtime pool.
    #[serde(default)]
    pub source_uri: String,
    /// On-disk format of the weights (e.g. `"safetensors_dir"`).
    #[serde(default)]
    pub format: String,
    /// Model architecture family (e.g. `"xlm-roberta"`). Cheap to record
    /// and lets a cold-start agent pick a compatible runtime without
    /// downloading the weights first.
    #[serde(default)]
    pub arch: String,
    /// Total byte size of the weights payload. Informational; useful for
    /// cold-start agents deciding between fetch-and-run vs. hit-the-beacon.
    #[serde(default)]
    pub byte_size: u64,
    /// The inference-server URL the compiler embedded against. A
    /// networked cold-start agent can hit this beacon to obtain query
    /// capability (embedder-as-a-service) without shipping weights. Empty
    /// when the program was compiled without a reachable server (e.g. an
    /// offline rebuild from a sibling weights bundle).
    #[serde(default)]
    pub beacon_url: String,
}

/// Holographic sigil — a per-QIP watermark and visual seal.
///
/// Every compiled QIP carries a sigil: a deterministic token
/// (`beacon_sigil`, e.g. `sigil://82f9273cee3669b8`) derived from the
/// program's URN and corpus checksum, plus an ASCII art `seal` that is
/// mathematically derived from the program's space holograms.
///
/// The token serves as a watermark for beacon attribution: when a
/// networked cold-start agent hits the beacon to project a query, the
/// sigil identifies which compiled QIP the query is targeting. The seal
/// is a fingerprint you can read at a glance — two QIPs with different
/// content produce different holograms, therefore different seals.
///
/// Both fields are optional and default to empty so manifests written
/// before sigils were introduced still parse. A runtime that finds
/// `beacon_sigil` empty SHOULD treat the program as unattributed at the
/// beacon (no watermark to track).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SigilInfo {
    /// Deterministic per-QIP token (e.g. `sigil://82f9273cee3669b8`).
    /// SHA-256 of `<urn>:<corpus_checksum>`, first 16 hex digits. Used
    /// as a watermark for beacon attribution.
    #[serde(default)]
    pub beacon_sigil: String,
    /// Holographic ASCII seal — a bilaterally-symmetric crest derived
    /// from the program's space holograms. The crown row is the system
    /// hologram; each body row is one space. Deterministic from the
    /// holograms, so two QIPs with different content produce different
    /// seals. Empty when the program has no spaces (nothing to render).
    #[serde(default)]
    pub seal: String,
}

/// Encoding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingInfo {
    /// Embedding model to use
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,

    /// Embedding dimension
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,

    /// Structural encoding dimension
    #[serde(default = "default_structural_dim")]
    pub structural_dim: usize,

    /// Position encoding type
    #[serde(default = "default_position_encoding")]
    pub position_encoding: String,

    /// Provenance of the embedder that compiled this program. `None` on
    /// manifests written before provenance was recorded; `Some` with empty
    /// fields when the compiler could not resolve the embedder's identity
    /// (e.g. an unreachable beacon with no local weights cache).
    #[serde(default)]
    pub embedder_provenance: EmbedderProvenance,
}

impl Default for EncodingInfo {
    fn default() -> Self {
        Self {
            embedding_model: default_embedding_model(),
            embedding_dim: default_embedding_dim(),
            structural_dim: default_structural_dim(),
            position_encoding: default_position_encoding(),
            embedder_provenance: EmbedderProvenance::default(),
        }
    }
}

fn default_embedding_model() -> String {
    "arctic-embed-l".to_string()
}
fn default_embedding_dim() -> usize {
    1024
}
fn default_structural_dim() -> usize {
    256
}
fn default_position_encoding() -> String {
    "sinusoidal".to_string()
}

/// Bounded context configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextInfo {
    /// Context name (e.g., "synapse")
    pub name: String,

    /// Path relative to project root
    pub path: String,

    /// Description
    #[serde(default)]
    pub description: String,

    /// Role in organism
    #[serde(default = "default_role")]
    pub role: String,

    /// Exports configuration
    #[serde(default)]
    pub exports: Option<ExportsInfo>,

    /// Patterns configuration
    #[serde(default)]
    pub patterns: Option<PatternsInfo>,
}

/// Generalized knowledge-space manifest entry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeSpaceInfo {
    /// Stable identifier for the space.
    pub id: String,
    /// Human-readable name for the space.
    pub name: String,
    /// Serialized kind string.
    pub kind: String,
    /// Optional path for file-backed spaces.
    #[serde(default)]
    pub path: Option<String>,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// Boundary policy for the space.
    #[serde(default)]
    pub boundaries: Option<SpaceBoundaryInfo>,
    /// Provenance policy for the space.
    #[serde(default)]
    pub provenance: Option<SpaceProvenanceInfo>,
    /// Temporal policy for the space.
    #[serde(default)]
    pub temporal: Option<SpaceTemporalInfo>,
    /// Salience defaults for the space.
    #[serde(default)]
    pub salience: Option<SalienceInfo>,
    /// Extension metadata for non-core annotations.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl KnowledgeSpaceInfo {
    fn from_context(manifest: &ContextManifest, context: &ContextInfo) -> Self {
        let allowed_outbound_spaces = manifest
            .boundaries
            .get(&context.name)
            .cloned()
            .unwrap_or_default();

        let allowed_inbound_spaces = manifest
            .boundaries
            .iter()
            .filter_map(|(from, deps)| {
                if deps.iter().any(|dep| dep == &context.name) {
                    Some(from.clone())
                } else {
                    None
                }
            })
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert("context_role".to_string(), context.role.clone());

        if let Some(exports) = &context.exports {
            metadata.insert("exports_public".to_string(), exports.public.join(","));
        }

        if let Some(patterns) = &context.patterns {
            metadata.insert("patterns_required".to_string(), patterns.required.join(","));
            metadata.insert("patterns_detected".to_string(), patterns.detected.join(","));
        }

        Self {
            id: context.name.clone(),
            name: context.name.clone(),
            kind: knowledge_space_kind_name(KnowledgeSpaceKind::Code).to_string(),
            path: Some(context.path.clone()),
            description: context.description.clone(),
            boundaries: Some(SpaceBoundaryInfo {
                allow_unbridged_projection: false,
                allowed_inbound_spaces,
                allowed_outbound_spaces,
                allowed_bridge_ids: Vec::new(),
            }),
            provenance: None,
            temporal: None,
            salience: None,
            metadata,
        }
    }
}

fn default_role() -> String {
    "utility".to_string()
}

/// Public exports for a context
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExportsInfo {
    /// Public items
    #[serde(default)]
    pub public: Vec<String>,
}

/// Pattern configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatternsInfo {
    /// Required patterns
    #[serde(default)]
    pub required: Vec<String>,

    /// Detected patterns (auto-populated)
    #[serde(default)]
    pub detected: Vec<String>,
}

/// Boundary configuration for a generalized knowledge space.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpaceBoundaryInfo {
    /// Whether projection may leave this space without an explicit bridge.
    #[serde(default)]
    pub allow_unbridged_projection: bool,
    /// Spaces that may project into this space directly.
    #[serde(default)]
    pub allowed_inbound_spaces: Vec<String>,
    /// Spaces this space may project into directly.
    #[serde(default)]
    pub allowed_outbound_spaces: Vec<String>,
    /// Allowed bridge identifiers.
    #[serde(default)]
    pub allowed_bridge_ids: Vec<String>,
}

/// Provenance policy for a knowledge space.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpaceProvenanceInfo {
    /// Source identifiers contributing to the space.
    #[serde(default)]
    pub sources: Vec<String>,
    /// Declared authority for the space.
    #[serde(default)]
    pub authority: Option<String>,
    /// Refresh cadence string.
    #[serde(default)]
    pub refresh_cadence: Option<String>,
    /// Encoding policy identifier.
    #[serde(default)]
    pub encoding_policy: Option<String>,
}

/// Temporal policy for a knowledge space.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpaceTemporalInfo {
    /// Freshness window in seconds.
    #[serde(default)]
    pub freshness_window_seconds: Option<u64>,
    /// Decay half-life in seconds.
    #[serde(default)]
    pub decay_half_life_seconds: Option<u64>,
    /// Temporal resolution string.
    #[serde(default)]
    pub temporal_resolution: Option<String>,
}

/// Salience defaults for a knowledge space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalienceInfo {
    /// Default ingestion salience.
    #[serde(default = "default_salience")]
    pub default_salience: f32,
    /// Freshness boost for recently updated items.
    #[serde(default)]
    pub freshness_boost: f32,
    /// Penalty for stale items.
    #[serde(default)]
    pub staleness_penalty: f32,
}

impl Default for SalienceInfo {
    fn default() -> Self {
        Self {
            default_salience: default_salience(),
            freshness_boost: 0.0,
            staleness_penalty: 0.0,
        }
    }
}

fn default_salience() -> f32 {
    1.0
}

/// Federation configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FederationInfo {
    /// Allowed external organisms
    #[serde(default)]
    pub allowed: Vec<String>,

    /// Specific imports
    #[serde(default)]
    pub imports: Vec<FederationImport>,
}

/// A single federation import
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationImport {
    /// Source organism
    pub from: String,

    /// Imported items
    #[serde(default)]
    pub items: Vec<String>,
}

/// Bridge configuration connecting two knowledge spaces.
///
/// `BridgeInfo` is the *manifest* (TOML-friendly) shape of a bridge. The
/// runtime shape consumed by the projector is
/// [`crate::holographic::BridgeBinding`]; conversion is provided by
/// [`KnowledgeManifest::to_bridge_bindings`] (and equivalently by the
/// `From<&BridgeInfo> for BridgeBinding` impl). Manifest-shape fields
/// default to substrate-blessed values that preserve the "if a consumer
/// declared this bridge, it should be traversable at projection time"
/// reading; consumers that want to override (e.g. confidence_penalty,
/// max_traversals) supply the optional fields below in their TOML.
///
/// `BridgeInfo::default()` is implemented to match the
/// `#[serde(default = "...")]` defaults so the substrate-blessed
/// reading applies in both serde and code-path construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeInfo {
    /// Stable bridge identifier.
    pub id: String,
    /// Source space identifier.
    pub from_space: String,
    /// Target space identifier.
    pub to_space: String,
    /// Binding category. Recognized values are
    /// `semantic_equivalence`, `implementation_mapping`,
    /// `policy_applies_to`, `workflow_depends_on`, `derived_from`;
    /// any other value is preserved as
    /// `BridgeBindingType::Custom(...)` at runtime.
    pub binding_type: String,
    /// Whether the bridge is enabled. Becomes
    /// `BridgeBinding.active` at runtime; non-enabled bridges never
    /// participate in projection traversal.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Whether projection may traverse this bridge automatically.
    ///
    /// Defaults to `true` for manifest-declared bridges. The substrate's
    /// reading is: declaring a bridge in a manifest is an explicit
    /// signal that the consumer wants projection to use it; the
    /// runtime-side `BridgeProjectionPolicy::default()` is `false`
    /// because that default applies to bridges constructed
    /// programmatically without consumer intent. Override to `false`
    /// in TOML for bridges that should be present but inactive
    /// (e.g. for staged rollouts or audit-only bridges).
    #[serde(default = "default_allow_automatic_projection")]
    pub allow_automatic_projection: bool,
    /// Whether attribution is required when this bridge contributes
    /// to a projection. Defaults to `true`; mirror of
    /// `BridgeProjectionPolicy::require_attribution`.
    #[serde(default = "default_require_attribution_bridge")]
    pub require_attribution: bool,
    /// Confidence discount applied each time the bridge is traversed.
    /// Defaults to `0.0`; consumers that derive bridges
    /// programmatically (co-occurrence ingest, embedding similarity)
    /// typically set `0.1–0.2` to keep `bridge_bleed_score`
    /// meaningful.
    #[serde(default)]
    pub confidence_penalty: f32,
    /// Maximum number of times this bridge may be traversed within
    /// one projection. Defaults to `1`.
    #[serde(default = "default_max_traversals")]
    pub max_traversals: usize,
    /// Extension metadata. Free-form annotations the substrate
    /// preserves but does not interpret.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

fn default_enabled() -> bool {
    true
}

fn default_allow_automatic_projection() -> bool {
    true
}

fn default_require_attribution_bridge() -> bool {
    true
}

fn default_max_traversals() -> usize {
    1
}

impl Default for BridgeInfo {
    fn default() -> Self {
        Self {
            id: String::new(),
            from_space: String::new(),
            to_space: String::new(),
            binding_type: String::new(),
            enabled: default_enabled(),
            allow_automatic_projection: default_allow_automatic_projection(),
            require_attribution: default_require_attribution_bridge(),
            confidence_penalty: 0.0,
            max_traversals: default_max_traversals(),
            metadata: HashMap::new(),
        }
    }
}

/// Parse a manifest-shape `binding_type` string into the runtime enum.
///
/// Recognized canonical strings map to their corresponding
/// [`BridgeBindingType`] variant; anything else is preserved as
/// `Custom(...)` so consumer-specific binding categories survive the
/// transform without loss.
pub fn parse_binding_type(value: &str) -> BridgeBindingType {
    match value {
        "semantic_equivalence" => BridgeBindingType::SemanticEquivalence,
        "implementation_mapping" => BridgeBindingType::ImplementationMapping,
        "policy_applies_to" => BridgeBindingType::PolicyAppliesTo,
        "workflow_depends_on" => BridgeBindingType::WorkflowDependsOn,
        "derived_from" => BridgeBindingType::DerivedFrom,
        other => BridgeBindingType::Custom(other.to_string()),
    }
}

impl From<&BridgeInfo> for BridgeBinding {
    /// Materialize a manifest-shape `BridgeInfo` into the runtime
    /// `BridgeBinding` using substrate-blessed defaults.
    ///
    /// Defaults preserve the "declared = traversable" reading:
    /// `active = enabled`, `policy.allow_automatic_projection` follows
    /// the manifest's optional field (default `true` for declared
    /// bridges), `policy.confidence_penalty` and `policy.max_traversals`
    /// from the manifest if present (else `0.0` / `1`).
    /// `provenance` is left default; consumers that want richer
    /// provenance should construct `BridgeBinding` directly rather
    /// than going through this convenience.
    fn from(info: &BridgeInfo) -> Self {
        BridgeBinding {
            id: info.id.clone(),
            from_space: info.from_space.clone(),
            to_space: info.to_space.clone(),
            binding_type: parse_binding_type(&info.binding_type),
            policy: BridgeProjectionPolicy {
                allow_automatic_projection: info.allow_automatic_projection,
                require_attribution: info.require_attribution,
                confidence_penalty: info.confidence_penalty,
                max_traversals: info.max_traversals,
            },
            provenance: ItemProvenance::default(),
            active: info.enabled,
        }
    }
}

/// Default projection behavior for a knowledge manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionDefaults {
    /// Default projection mode name.
    #[serde(default = "default_projection_mode")]
    pub mode: String,
    /// Whether attribution is required by default.
    #[serde(default = "default_require_attribution")]
    pub require_attribution: bool,
    /// Maximum number of space hops allowed by default.
    #[serde(default = "default_max_space_hops")]
    pub max_space_hops: usize,
    /// Minimum confidence threshold, if any.
    #[serde(default)]
    pub min_confidence: Option<f32>,
}

impl Default for ProjectionDefaults {
    fn default() -> Self {
        Self {
            mode: default_projection_mode(),
            require_attribution: default_require_attribution(),
            max_space_hops: default_max_space_hops(),
            min_confidence: None,
        }
    }
}

fn default_projection_mode() -> String {
    "auto".to_string()
}

fn default_require_attribution() -> bool {
    true
}

fn default_max_space_hops() -> usize {
    1
}

/// History configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryInfo {
    /// Number of commits to track
    #[serde(default = "default_history_depth")]
    pub depth: usize,

    /// Patterns to track
    #[serde(default)]
    pub track_patterns: Vec<String>,
}

impl Default for HistoryInfo {
    fn default() -> Self {
        Self {
            depth: default_history_depth(),
            track_patterns: vec!["bug-fix".into(), "feature".into(), "refactor".into()],
        }
    }
}

fn default_history_depth() -> usize {
    100
}

/// Codegen configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodegenInfo {
    /// Primary model for code generation
    #[serde(default = "default_primary_model")]
    pub primary_model: String,

    /// Fallback model
    #[serde(default = "default_fallback_model")]
    pub fallback_model: String,

    /// Minimum coherence score
    #[serde(default = "default_min_coherence")]
    pub min_coherence: f32,

    /// Minimum style match
    #[serde(default = "default_min_style")]
    pub min_style_match: f32,

    /// Require tests for generated code
    #[serde(default = "default_require_tests")]
    pub require_tests: bool,

    /// Require lint pass
    #[serde(default = "default_require_lint")]
    pub require_lint_pass: bool,

    /// Maximum diff lines
    #[serde(default = "default_max_diff")]
    pub max_diff_lines: usize,
}

impl Default for CodegenInfo {
    fn default() -> Self {
        Self {
            primary_model: default_primary_model(),
            fallback_model: default_fallback_model(),
            min_coherence: default_min_coherence(),
            min_style_match: default_min_style(),
            require_tests: default_require_tests(),
            require_lint_pass: default_require_lint(),
            max_diff_lines: default_max_diff(),
        }
    }
}

fn default_primary_model() -> String {
    "deepseek-coder-v2".to_string()
}
fn default_fallback_model() -> String {
    "qwen25-coder-7b".to_string()
}
fn default_min_coherence() -> f32 {
    0.7
}
fn default_min_style() -> f32 {
    0.6
}
fn default_require_tests() -> bool {
    true
}
fn default_require_lint() -> bool {
    true
}
fn default_max_diff() -> usize {
    200
}

fn knowledge_space_kind_name(kind: KnowledgeSpaceKind) -> &'static str {
    match kind {
        KnowledgeSpaceKind::Code => "code",
        KnowledgeSpaceKind::Business => "business",
        KnowledgeSpaceKind::Document => "document",
        KnowledgeSpaceKind::Workflow => "workflow",
        KnowledgeSpaceKind::Telemetry => "telemetry",
        KnowledgeSpaceKind::UserState => "user_state",
        KnowledgeSpaceKind::Conversation => "conversation",
        KnowledgeSpaceKind::External => "external",
        KnowledgeSpaceKind::Custom(_) => "custom",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_manifest() {
        let toml = r#"
[organism]
id = "test-org"
name = "Test Organism"
description = "A test organism"

[encoding]
embedding_model = "arctic-embed-l"
embedding_dim = 1024

[[contexts]]
name = "core"
path = "crates/core"
description = "Core functionality"
role = "core"

[[contexts]]
name = "util"
path = "crates/util"
role = "utility"

[boundaries]
core = []
util = ["core"]
"#;

        let manifest: ContextManifest = toml::from_str(toml).unwrap();

        assert_eq!(manifest.organism.id, "test-org");
        assert_eq!(manifest.contexts.len(), 2);
        assert!(manifest.can_depend("util", "core"));
        assert!(!manifest.can_depend("core", "util"));
    }

    #[test]
    fn test_parse_knowledge_manifest() {
        let toml = r#"
[system]
id = "test-knowledge"
name = "Test Knowledge"
description = "A generalized knowledge manifest"

[encoding]
embedding_model = "arctic-embed-l"

[[spaces]]
id = "business"
name = "Business Knowledge"
kind = "business"
description = "Business entities and relations"

[spaces.boundaries]
allowed_outbound_spaces = ["code"]

[[spaces]]
id = "code"
name = "Code Knowledge"
kind = "code"
path = "src"

[[bridges]]
id = "business_to_code"
from_space = "business"
to_space = "code"
binding_type = "implementation_mapping"
"#;

        let manifest: KnowledgeManifest = toml::from_str(toml).unwrap();

        assert_eq!(manifest.system.id, "test-knowledge");
        assert_eq!(manifest.spaces.len(), 2);
        assert_eq!(manifest.bridges.len(), 1);
        assert_eq!(manifest.projection.mode, "auto");
    }

    #[test]
    fn test_context_manifest_migrates_to_knowledge_manifest() {
        let context_manifest = ContextManifest {
            organism: OrganismInfo {
                id: "org-test".into(),
                name: "Org Test".into(),
                description: "Adapter test".into(),
                version: "0.1.0".into(),
            },
            encoding: EncodingInfo::default(),
            contexts: vec![
                ContextInfo {
                    name: "core".into(),
                    path: "crates/core".into(),
                    description: "Core".into(),
                    role: "core".into(),
                    exports: Some(ExportsInfo {
                        public: vec!["Api".into()],
                    }),
                    patterns: Some(PatternsInfo {
                        required: vec!["async".into()],
                        detected: vec!["error-handling".into()],
                    }),
                },
                ContextInfo {
                    name: "util".into(),
                    path: "crates/util".into(),
                    description: String::new(),
                    role: "utility".into(),
                    exports: None,
                    patterns: None,
                },
            ],
            boundaries: HashMap::from([("util".to_string(), vec!["core".to_string()])]),
            federation: FederationInfo::default(),
            history: HistoryInfo::default(),
            codegen: CodegenInfo::default(),
        };

        let manifest = KnowledgeManifest::from_context_manifest(&context_manifest);

        assert_eq!(manifest.system.id, "org-test");
        assert_eq!(manifest.spaces.len(), 2);
        assert_eq!(manifest.spaces[0].kind, "code");
        assert_eq!(
            manifest.spaces[1]
                .boundaries
                .as_ref()
                .unwrap()
                .allowed_outbound_spaces,
            vec!["core".to_string()]
        );
        assert_eq!(
            manifest.spaces[0].metadata.get("exports_public").unwrap(),
            "Api"
        );
        assert_eq!(
            manifest.spaces[0].metadata.get("context_role").unwrap(),
            "core"
        );
    }

    #[test]
    fn test_parse_binding_type_canonical_strings() {
        assert_eq!(
            parse_binding_type("semantic_equivalence"),
            BridgeBindingType::SemanticEquivalence
        );
        assert_eq!(
            parse_binding_type("implementation_mapping"),
            BridgeBindingType::ImplementationMapping
        );
        assert_eq!(
            parse_binding_type("policy_applies_to"),
            BridgeBindingType::PolicyAppliesTo
        );
        assert_eq!(
            parse_binding_type("workflow_depends_on"),
            BridgeBindingType::WorkflowDependsOn
        );
        assert_eq!(
            parse_binding_type("derived_from"),
            BridgeBindingType::DerivedFrom
        );
    }

    #[test]
    fn test_parse_binding_type_unknown_preserved_as_custom() {
        let parsed = parse_binding_type("acme_specific_thing");
        match parsed {
            BridgeBindingType::Custom(s) => assert_eq!(s, "acme_specific_thing"),
            other => panic!("expected Custom, got {other:?}"),
        }
    }

    #[test]
    fn test_bridge_info_to_binding_uses_substrate_blessed_defaults() {
        // The bare-minimum BridgeInfo a consumer would emit: id, spaces,
        // type, enabled. All policy fields fall through to substrate
        // defaults. Critical correctness invariant: the resulting
        // binding must be active AND traversable, NOT inert.
        let info = BridgeInfo {
            id: "capabilities_to_workflows".into(),
            from_space: "capabilities".into(),
            to_space: "workflows".into(),
            binding_type: "workflow_depends_on".into(),
            enabled: true,
            allow_automatic_projection: default_allow_automatic_projection(),
            require_attribution: default_require_attribution_bridge(),
            confidence_penalty: 0.0,
            max_traversals: default_max_traversals(),
            metadata: HashMap::new(),
        };

        let binding: BridgeBinding = (&info).into();

        assert!(binding.active, "declared+enabled bridge must be active");
        assert!(
            binding.policy.allow_automatic_projection,
            "manifest-declared bridge must default to traversable; \
             this is the substrate's blessed reading of declared = traversable"
        );
        assert!(binding.policy.require_attribution);
        assert_eq!(binding.policy.confidence_penalty, 0.0);
        assert_eq!(binding.policy.max_traversals, 1);
        assert_eq!(binding.binding_type, BridgeBindingType::WorkflowDependsOn);
    }

    #[test]
    fn test_bridge_info_disabled_lands_inactive() {
        // Override path: a consumer that wants a present-but-inactive
        // bridge sets enabled=false. Result: binding.active=false.
        let mut info = BridgeInfo {
            id: "staged_bridge".into(),
            from_space: "a".into(),
            to_space: "b".into(),
            binding_type: "derived_from".into(),
            enabled: false,
            allow_automatic_projection: default_allow_automatic_projection(),
            require_attribution: default_require_attribution_bridge(),
            confidence_penalty: 0.0,
            max_traversals: default_max_traversals(),
            metadata: HashMap::new(),
        };
        let binding: BridgeBinding = (&info).into();
        assert!(!binding.active);

        info.enabled = true;
        info.allow_automatic_projection = false;
        let binding: BridgeBinding = (&info).into();
        assert!(binding.active, "enabled flag controls .active");
        assert!(
            !binding.policy.allow_automatic_projection,
            "the consumer override of allow_automatic_projection \
             must round-trip independently of enabled"
        );
    }

    #[test]
    fn test_bridge_info_carries_consumer_supplied_penalty_and_traversals() {
        let info = BridgeInfo {
            id: "co_occurrence_derived".into(),
            from_space: "capabilities".into(),
            to_space: "workflows".into(),
            binding_type: "workflow_depends_on".into(),
            enabled: true,
            allow_automatic_projection: true,
            require_attribution: true,
            confidence_penalty: 0.15,
            max_traversals: 2,
            metadata: HashMap::new(),
        };
        let binding: BridgeBinding = (&info).into();
        assert!((binding.policy.confidence_penalty - 0.15).abs() < 1e-6);
        assert_eq!(binding.policy.max_traversals, 2);
    }

    #[test]
    fn test_to_bridge_bindings_preserves_order_and_count() {
        let toml = r#"
[system]
id = "test"
name = "Test"
description = "test"

[encoding]

[[bridges]]
id = "intent_to_business"
from_space = "intent"
to_space = "business"
binding_type = "derived_from"

[[bridges]]
id = "capabilities_to_workflows"
from_space = "capabilities"
to_space = "workflows"
binding_type = "workflow_depends_on"
confidence_penalty = 0.1

[[bridges]]
id = "docs_to_workflows"
from_space = "docs"
to_space = "workflows"
binding_type = "policy_applies_to"
"#;
        let manifest: KnowledgeManifest = toml::from_str(toml).unwrap();
        let bindings = manifest.to_bridge_bindings();

        assert_eq!(bindings.len(), 3);
        assert_eq!(bindings[0].id, "intent_to_business");
        assert_eq!(bindings[1].id, "capabilities_to_workflows");
        assert_eq!(bindings[2].id, "docs_to_workflows");

        // Critical: the bare manifest declarations must round-trip
        // into traversable runtime bindings without any consumer code
        // touching policy fields.
        assert!(
            bindings.iter().all(|b| b.active),
            "all enabled-by-default bridges must be active"
        );
        assert!(
            bindings.iter().all(|b| b.policy.allow_automatic_projection),
            "all default-policy bridges must be auto-traversable"
        );

        // The one bridge with explicit confidence_penalty propagated.
        assert!((bindings[1].policy.confidence_penalty - 0.1).abs() < 1e-6);
        // The other two fall through to default 0.0.
        assert_eq!(bindings[0].policy.confidence_penalty, 0.0);
        assert_eq!(bindings[2].policy.confidence_penalty, 0.0);
    }

    #[test]
    fn test_existing_minimal_toml_round_trips_under_added_fields() {
        // Backward compat: old TOML manifests written before policy
        // fields were added must still parse, with the new fields
        // taking their substrate-blessed defaults.
        let toml = r#"
[system]
id = "test"
name = "Test"
description = "test"

[encoding]

[[bridges]]
id = "legacy_bridge"
from_space = "a"
to_space = "b"
binding_type = "semantic_equivalence"
"#;
        let manifest: KnowledgeManifest = toml::from_str(toml).unwrap();
        let info = &manifest.bridges[0];

        assert!(info.allow_automatic_projection);
        assert!(info.require_attribution);
        assert_eq!(info.confidence_penalty, 0.0);
        assert_eq!(info.max_traversals, 1);

        let binding: BridgeBinding = info.into();
        assert!(binding.active);
        assert!(binding.policy.allow_automatic_projection);
    }

    #[test]
    fn test_find_knowledge_manifest_does_not_fall_back_to_context_manifest() {
        let temp_dir = tempfile::tempdir().unwrap();
        let context_path = temp_dir.path().join("contexts.toml");
        std::fs::write(
            &context_path,
            r#"
[organism]
id = "legacy-only"
name = "Legacy Only"
description = "Legacy manifest present"
"#,
        )
        .unwrap();

        assert!(KnowledgeManifest::find(temp_dir.path()).is_none());
    }
}

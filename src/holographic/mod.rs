//! # Holographic Memory System
//!
//! Tensor-based knowledge representation using holographic reduced representations.
//! Instead of retrieving context on-demand (RAG), this module maintains
//! persistent holographic representations of codebases, documents, and knowledge spaces.
//!
//! ## Key Concepts
//!
//! - **Holographic Storage**: Every piece contains information about the whole
//! - **Hierarchical Composition**: Tensors compose upward (module → context → organism)
//! - **Projection Over Retrieval**: Context emerges via tensor operations, not search
//! - **Incremental Updates**: Tensors update on code changes, not rebuilt
//! - **Shared Knowledge Spaces**: The same HRR substrate can represent code,
//!   documents, workflows, business knowledge, and other datasets
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    HOLOGRAPHIC MEMORY SYSTEM                      │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Corpus → Embeddings → Bind → Compose → KnowledgeSystemView    │
//! │                                                                 │
//! │  Query → Embed → Project (⊙) → Relevant Context                │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use an_tensor_compiler::holographic::{bind, bundle_record, project, superimpose};
//!
//! // Bind role-filler pairs and bundle a labeled record
//! let record = bundle_record(&[("role", &role_tensor, "filler", &filler_tensor)])?;
//!
//! // Superimpose many records into one fixed-size hologram
//! let hologram = superimpose(&[record_a, record_b], None)?;
//!
//! // Project a query: recall is a tensor op, not a search
//! let result = project(&hologram, &query)?;
//! ```
//!
//! Ported from `an-tensor-core` v0.5.0 (`holographic` module, portable subset).
//! Excluded from this port (they require the private inference/IO stack):
//! `encoding`, `projection`, `manager`, `conversation`, `daemon_encoder`,
//! `validation`, `runtime_advance`, `cache`.

pub mod bridges;
pub mod composite;
pub mod diagnostics;
pub mod ecosystem;
pub mod entities;
pub mod geometric;
pub mod hypothesis;
pub mod knowledge;
pub mod manifest;
pub mod memory;
pub mod multi_resolution;
pub mod ops;
pub mod persistence;
pub mod policy;
pub mod query_protocol;
pub mod recovery;
pub mod spatial;
pub mod storage;
pub mod types;

// --- Re-exports ---
pub use bridges::{BridgeBinding, BridgeBindingType, BridgeProjectionPolicy};
pub use composite::{compose_binary_relation, compose_relation_with_transform};
pub use diagnostics::{BridgeContribution, ProjectionDiagnostics, SpaceContribution};
pub use ecosystem::{
    ComposedContext, ComposedItem, ConsolidationResult, EcoProjectionResult,
    EcoProjectionWithDiagnostics, EcosystemConfig, EcosystemHologram, SourceInfo,
};
pub use entities::{
    AttributeTensor, EntityTensor, RelationParticipant, RelationTensor, RoleTensor,
};
pub use geometric::{GeometricIndex, DEFAULT_N_CLASSES, DEFAULT_N_SUB_REFS};
pub use hypothesis::{CandidateRule, HypothesisBank};
pub use knowledge::{
    BridgeAvailability, ItemProvenance, KnowledgeItemRef, KnowledgeItemType, KnowledgeSpaceKind,
    KnowledgeSpaceTensor, KnowledgeSystemView, SalienceDefaults, SpaceBoundaryPolicy,
    SpaceProvenance, SpaceTemporalPolicy, TemporalState,
};
pub use manifest::{
    parse_binding_type, BridgeInfo, CodegenInfo, ContextInfo, ContextManifest, EmbedderProvenance,
    EncodingInfo, ExportsInfo, FederationImport, FederationInfo, HistoryInfo, KnowledgeManifest,
    KnowledgeSpaceInfo, OrganismInfo, PatternsInfo, ProjectionDefaults, SalienceInfo, SigilInfo,
    SpaceBoundaryInfo, SpaceProvenanceInfo, SpaceTemporalInfo, SystemInfo,
};
pub use memory::{
    Action,
    ActionOutcome,
    EmbeddingBackend,
    // Enhanced memory systems
    EntropyGatedMemory,
    Episode,
    EpisodeId,
    // Experience storage
    ExperienceHologram,
    // Pattern types
    FailurePattern,
    // Working memory
    FocusItem,
    HologramHealth,
    MemoryConfig,
    MemoryResult,
    MemoryStats,
    MultiScaleMemory,
    // Core episode types
    Observation,
    PatternLibrary,
    RecoveryPattern,
    RegimePartitionedMemory,
    // Unified memory
    SynapseMemory,
    TimeHorizon,
    WorkingMemory,
};
pub use multi_resolution::{
    EscalationPolicy, EscalationReason, MultiResolutionProgram, MultiResolutionStats,
    ResolutionItem, ResolutionLayer, ResolutionSpace, ResolutionTier,
};
pub use ops::{
    bind, bind_fast, bundle_record, cosine_similarity, normalize, position_encoding, project,
    role_encoding, superimpose, unbind, unbind_fast, unbind_role,
};
pub use persistence::{HrrArtifact, TensorPayload};
pub use policy::{ProjectionMode, ProjectionPolicy};
pub use query_protocol::{HrrEmbedder, HrrItemScore, HrrQuery, HrrQueryResult, HrrSpaceScore};
pub use recovery::{BindChainProbe, ProbeResult};
pub use spatial::{LatticeEncoder, LatticeNormalization, LatticeSpec};
pub use storage::{CacheEntry, CacheStats, ComponentCache, IncrementalState, TensorStore};
pub use types::{
    BoundaryRules, BoundaryViolation, BoundedContextTensor, CoherenceResult, ComponentId,
    ComponentKind, ComponentTensor, ContextRole, FederationBinding, HistoryTensor, ModuleTensor,
    OrganismTensor, ProjectionResult,
};

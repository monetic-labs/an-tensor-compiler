//! Cross-space bridge primitives for holographic memory.
//!
//! Bridge bindings model intentional, inspectable connections between knowledge
//! spaces so projection can cross boundaries without accidental bleed.

use crate::holographic::knowledge::ItemProvenance;

/// Explicit cross-space binding used for intentional projection between spaces.
#[derive(Debug, Clone)]
pub struct BridgeBinding {
    /// Stable identifier for the bridge.
    pub id: String,
    /// Source space identifier.
    pub from_space: String,
    /// Target space identifier.
    pub to_space: String,
    /// Broad binding category.
    pub binding_type: BridgeBindingType,
    /// Projection behavior allowed for the bridge.
    pub policy: BridgeProjectionPolicy,
    /// Provenance for bridge construction.
    pub provenance: ItemProvenance,
    /// Whether the bridge is currently active.
    pub active: bool,
}

/// Generic categories of cross-space binding.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BridgeBindingType {
    /// Both ends mean the same thing.
    SemanticEquivalence,
    /// One end implements the other.
    ImplementationMapping,
    /// A policy governing a target.
    PolicyAppliesTo,
    /// A workflow depending on another space.
    WorkflowDependsOn,
    /// Derived or summarized content.
    DerivedFrom,
    /// Caller-defined category tag.
    Custom(String),
}

/// Projection behavior for a bridge.
#[derive(Debug, Clone)]
pub struct BridgeProjectionPolicy {
    /// Whether projection may traverse this bridge automatically.
    pub allow_automatic_projection: bool,
    /// Whether attribution is required when this bridge contributes.
    pub require_attribution: bool,
    /// Maximum confidence discount applied when traversing the bridge.
    pub confidence_penalty: f32,
    /// Maximum number of times the bridge may be traversed within one projection.
    pub max_traversals: usize,
}

impl Default for BridgeProjectionPolicy {
    fn default() -> Self {
        Self {
            allow_automatic_projection: false,
            require_attribution: true,
            confidence_penalty: 0.0,
            max_traversals: 1,
        }
    }
}

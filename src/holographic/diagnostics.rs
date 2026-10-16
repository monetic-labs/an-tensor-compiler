//! Diagnostics for mixed-space holographic projection.
//!
//! These types provide a shared observability surface for attribution quality,
//! interference, and bridge behavior when projection spans multiple spaces.
//! Keep ProjectionDiagnostics fields aligned with mixed-space probe outputs.

/// Diagnostics emitted for a projection over one or more knowledge spaces.
#[derive(Debug, Clone, Default)]
pub struct ProjectionDiagnostics {
    /// Space-level contributors to the final projection.
    pub contributing_spaces: Vec<SpaceContribution>,
    /// Bridge-level contributors to the final projection.
    pub contributing_bridges: Vec<BridgeContribution>,
    /// Overall confidence for the projection.
    pub overall_confidence: f32,
    /// Degree of unintended bridge bleed in the result.
    pub bridge_bleed_score: f32,
    /// Degree of source saturation in the contributing spaces.
    pub source_saturation_score: f32,
    /// Overall interference score across mixed sources.
    pub interference_score: f32,
}

/// Contribution from a single knowledge space.
#[derive(Debug, Clone)]
pub struct SpaceContribution {
    /// Contributing space identifier.
    pub space_id: String,
    /// Confidence associated with the contribution.
    pub confidence: f32,
}

/// Contribution from a single bridge.
#[derive(Debug, Clone)]
pub struct BridgeContribution {
    /// Contributing bridge identifier.
    pub bridge_id: String,
    /// Confidence associated with the contribution.
    pub confidence: f32,
}

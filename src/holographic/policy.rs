//! Projection and boundary policy for generalized holographic spaces.

use crate::{Result, TensorCoreError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Generalized policy for projecting across one or more knowledge spaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionPolicy {
    /// Broad projection mode.
    pub mode: ProjectionMode,
    /// Spaces that may participate in the projection.
    pub allowed_spaces: Vec<String>,
    /// Bridges that may participate in the projection.
    pub allowed_bridges: Vec<String>,
    /// Whether attribution is required in the result.
    pub require_attribution: bool,
    /// Maximum number of bridge hops permitted.
    pub max_space_hops: usize,
    /// Minimum acceptable contribution confidence.
    pub min_confidence: Option<f32>,
    /// Optional per-space weight multipliers applied to space and item scores
    /// at projection time. Entries are keyed by space id; spaces absent from
    /// this map use a weight of `1.0`. Weights must be `>= 0.0`. Final scores
    /// remain clamped to `[0.0, 1.0]`.
    ///
    /// This is the substrate-level mechanism for compositions like
    /// "weight tenant-owned space at 0.7, shared space at 0.3" within a
    /// multi-space projection over a composed [`crate::holographic::KnowledgeSystemView`].
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub space_weights: HashMap<String, f32>,
    /// HRR-G2 geometric retrieval radius. When a space carries a
    /// [`GeometricIndex`](crate::holographic::GeometricIndex), item scoring
    /// preselects candidates in the query cell + neighbors out to this radius
    /// before cosine reranking. `0` means "cell only" (no neighbors); the
    /// default of `1` includes adjacent classes and sub-cells. Set to `None`
    /// to disable geometric retrieval and force a linear scan.
    #[serde(default = "default_geometric_radius")]
    pub geometric_radius: Option<usize>,
}

fn default_geometric_radius() -> Option<usize> {
    Some(1)
}

impl Default for ProjectionPolicy {
    fn default() -> Self {
        Self {
            mode: ProjectionMode::Auto,
            allowed_spaces: Vec::new(),
            allowed_bridges: Vec::new(),
            require_attribution: true,
            max_space_hops: 1,
            min_confidence: None,
            space_weights: HashMap::new(),
            geometric_radius: default_geometric_radius(),
        }
    }
}

impl ProjectionPolicy {
    /// Create a policy with the given mode and default safeguards.
    pub fn new(mode: ProjectionMode) -> Self {
        Self {
            mode,
            ..Self::default()
        }
    }

    /// Create a policy scoped to a single knowledge space.
    pub fn single_space(space_id: impl Into<String>) -> Self {
        Self {
            mode: ProjectionMode::SingleSpace,
            allowed_spaces: vec![space_id.into()],
            ..Self::default()
        }
    }

    /// Create a policy spanning multiple spaces without explicit bridge traversal.
    pub fn multi_space(space_ids: Vec<String>) -> Self {
        Self {
            mode: ProjectionMode::MultiSpace,
            allowed_spaces: space_ids,
            ..Self::default()
        }
    }

    /// Create a bridge-aware policy across one or more spaces.
    pub fn bridged(space_ids: Vec<String>, bridge_ids: Vec<String>) -> Self {
        Self {
            mode: ProjectionMode::Bridged,
            allowed_spaces: space_ids,
            allowed_bridges: bridge_ids,
            ..Self::default()
        }
    }

    /// Add an allowed space to the policy.
    pub fn with_allowed_space(mut self, space_id: impl Into<String>) -> Self {
        self.allowed_spaces.push(space_id.into());
        self
    }

    /// Add an allowed bridge to the policy.
    pub fn with_allowed_bridge(mut self, bridge_id: impl Into<String>) -> Self {
        self.allowed_bridges.push(bridge_id.into());
        self
    }

    /// Override attribution requirements.
    pub fn requiring_attribution(mut self, require_attribution: bool) -> Self {
        self.require_attribution = require_attribution;
        self
    }

    /// Set the maximum number of bridge hops allowed.
    pub fn with_max_space_hops(mut self, max_space_hops: usize) -> Self {
        self.max_space_hops = max_space_hops;
        self
    }

    /// Set the minimum contribution confidence threshold.
    pub fn with_min_confidence(mut self, min_confidence: Option<f32>) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Set a single space's weight multiplier.
    pub fn with_space_weight(mut self, space_id: impl Into<String>, weight: f32) -> Self {
        self.space_weights.insert(space_id.into(), weight);
        self
    }

    /// Replace the full per-space weight map.
    pub fn with_space_weights(mut self, weights: HashMap<String, f32>) -> Self {
        self.space_weights = weights;
        self
    }

    /// Override the HRR-G2 geometric retrieval radius. `None` disables
    /// geometric retrieval and forces a linear scan; `Some(0)` uses the query
    /// cell only; `Some(r)` includes neighbors out to radius `r`.
    pub fn with_geometric_radius(mut self, radius: Option<usize>) -> Self {
        self.geometric_radius = radius;
        self
    }

    /// Resolved weight for a space id (defaults to `1.0` when absent).
    pub fn space_weight(&self, space_id: &str) -> f32 {
        self.space_weights.get(space_id).copied().unwrap_or(1.0)
    }

    /// Check whether the policy crosses more than one space.
    pub fn is_cross_space(&self) -> bool {
        self.allowed_spaces.len() > 1
            || matches!(
                self.mode,
                ProjectionMode::MultiSpace | ProjectionMode::Bridged
            )
    }

    /// Check whether the policy requires explicit bridge traversal.
    pub fn requires_bridges(&self) -> bool {
        matches!(self.mode, ProjectionMode::Bridged)
    }

    /// Validate that the policy is internally consistent.
    pub fn validate(&self) -> Result<()> {
        if let Some(min_confidence) = self.min_confidence {
            if !(0.0..=1.0).contains(&min_confidence) {
                return Err(TensorCoreError::Projection(format!(
                    "ProjectionPolicy.min_confidence must be in [0, 1], got {}",
                    min_confidence
                )));
            }
        }

        for (space_id, weight) in &self.space_weights {
            if !weight.is_finite() || *weight < 0.0 {
                return Err(TensorCoreError::Projection(format!(
                    "ProjectionPolicy.space_weights[{}] must be a finite, non-negative \
                     number, got {}",
                    space_id, weight
                )));
            }
        }

        match self.mode {
            ProjectionMode::SingleSpace => {
                if self.allowed_spaces.len() > 1 {
                    return Err(TensorCoreError::Projection(
                        "ProjectionMode::SingleSpace cannot allow more than one space".into(),
                    ));
                }
                if !self.allowed_bridges.is_empty() {
                    return Err(TensorCoreError::Projection(
                        "ProjectionMode::SingleSpace cannot declare bridges".into(),
                    ));
                }
            }
            ProjectionMode::MultiSpace => {
                if self.allowed_spaces.len() < 2 {
                    return Err(TensorCoreError::Projection(
                        "ProjectionMode::MultiSpace requires at least two allowed spaces".into(),
                    ));
                }
                if !self.allowed_bridges.is_empty() {
                    return Err(TensorCoreError::Projection(
                        "ProjectionMode::MultiSpace should not declare explicit bridges".into(),
                    ));
                }
            }
            ProjectionMode::Bridged => {
                if self.allowed_bridges.is_empty() {
                    return Err(TensorCoreError::Projection(
                        "ProjectionMode::Bridged requires at least one allowed bridge".into(),
                    ));
                }
                if self.max_space_hops == 0 {
                    return Err(TensorCoreError::Projection(
                        "ProjectionMode::Bridged requires max_space_hops >= 1".into(),
                    ));
                }
            }
            ProjectionMode::Auto => {}
        }

        Ok(())
    }
}

/// High-level projection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ProjectionMode {
    /// Project within one space only.
    SingleSpace,
    /// Project across spaces without bridges.
    MultiSpace,
    /// Traverse explicit bridges.
    Bridged,
    /// Choose automatically from the query.
    #[default]
    Auto,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_space_policy_validates() {
        let policy = ProjectionPolicy::single_space("code");
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn test_multi_space_policy_requires_multiple_spaces() {
        let policy = ProjectionPolicy::multi_space(vec!["code".to_string()]);
        assert!(policy.validate().is_err());
    }

    #[test]
    fn test_bridged_policy_requires_bridge() {
        let policy = ProjectionPolicy::new(ProjectionMode::Bridged)
            .with_allowed_space("business")
            .with_allowed_space("code");
        assert!(policy.validate().is_err());
    }

    #[test]
    fn test_min_confidence_must_be_normalized() {
        let policy = ProjectionPolicy::single_space("code").with_min_confidence(Some(1.5));
        assert!(policy.validate().is_err());
    }

    #[test]
    fn test_space_weight_defaults_to_one_when_absent() {
        let policy =
            ProjectionPolicy::multi_space(vec!["a".into(), "b".into()]).with_space_weight("a", 0.7);
        assert_eq!(policy.space_weight("a"), 0.7);
        assert_eq!(policy.space_weight("b"), 1.0);
        assert_eq!(policy.space_weight("missing"), 1.0);
    }

    #[test]
    fn test_negative_space_weight_is_rejected() {
        let policy = ProjectionPolicy::multi_space(vec!["a".into(), "b".into()])
            .with_space_weight("a", -0.1);
        assert!(policy.validate().is_err());
    }

    #[test]
    fn test_non_finite_space_weight_is_rejected() {
        let policy = ProjectionPolicy::multi_space(vec!["a".into(), "b".into()])
            .with_space_weight("a", f32::NAN);
        assert!(policy.validate().is_err());
    }
}

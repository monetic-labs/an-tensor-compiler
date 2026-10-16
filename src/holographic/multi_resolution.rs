//! Multi-Resolution Holograms — L1/L2 dimension hierarchy (D1).
//!
//! Per `HRR_INTELLIGENCE_PLATFORM_SPEC.md` D1: the 1024 vs 65536 tension is
//! resolved by maintaining two holograms per memory program, like a CPU cache
//! hierarchy:
//!
//! - **L1 (1024-dim):** fast query, low compute, used for common projections
//! - **L2 (65536-dim):** deep reasoning, high fidelity, used when L1
//!   confidence is low or interference is high
//!
//! This lets the system be fast for common queries and deep for complex ones,
//! without forcing a single dimension. It resolves:
//! - HRR-G3 (cross-dim federation): cells with only L1 can federate L1; cells
//!   with L2 federate L2; cross-tier queries escalate internally.
//! - HRR-G4 (structural_dim unused): structural bindings live in L1 (low-dim,
//!   fast — tree position, module hierarchy). Semantic bindings live in L2
//!   (high-dim, rich — meaning, relationships).

use serde::{Deserialize, Serialize};

// ============================================================================
// Resolution tier — L1 (fast) / L2 (deep)
// ============================================================================

/// A hologram resolution tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ResolutionTier {
    /// L1 — fast query, low compute. Default: 1024-dim. Structural bindings
    /// (tree position, module hierarchy) live here.
    L1,
    /// L2 — deep reasoning, high fidelity. Default: 65536-dim. Semantic
    /// bindings (meaning, relationships) live here.
    L2,
}

impl ResolutionTier {
    /// Default dimension for each tier.
    pub fn default_dim(&self) -> usize {
        match self {
            Self::L1 => 1024,
            Self::L2 => 65536,
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::L1 => "L1 (fast, 1024-dim)",
            Self::L2 => "L2 (deep, 65536-dim)",
        }
    }

    /// Parse from a string ("l1", "l2", "L1", "L2").
    pub fn from_str_lossy(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "l1" => Some(Self::L1),
            "l2" => Some(Self::L2),
            _ => None,
        }
    }
}

// ============================================================================
// Escalation policy — when to use the deep path
// ============================================================================

/// When to escalate from L1 to L2.
///
/// Per D1.1: L1 is the fast path; L2 is the deep path used when L1 confidence
/// is low or interference is high. The escalation policy defines the
/// thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationPolicy {
    /// Escalate if L1 overall confidence is below this (0–1). Default: 0.4.
    pub min_confidence: f32,
    /// Escalate if L1 interference is above this (0–1). Default: 0.5.
    pub max_interference: f32,
    /// Whether to escalate on bridge bleed (cross-space signal leaking).
    /// Default: true — bridge bleed indicates the L1 projection is noisy.
    pub escalate_on_bridge_bleed: bool,
    /// Bridge bleed threshold for escalation (0–1). Default: 0.3.
    pub max_bridge_bleed: f32,
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        Self {
            min_confidence: 0.4,
            max_interference: 0.5,
            escalate_on_bridge_bleed: true,
            max_bridge_bleed: 0.3,
        }
    }
}

impl EscalationPolicy {
    /// Whether a probe result at L1 warrants escalation to L2.
    ///
    /// Returns the reason for escalation (or None if L1 is sufficient).
    pub fn should_escalate(
        &self,
        confidence: f32,
        interference: f32,
        bridge_bleed: f32,
    ) -> Option<EscalationReason> {
        if confidence < self.min_confidence {
            return Some(EscalationReason::LowConfidence {
                observed: confidence,
                threshold: self.min_confidence,
            });
        }
        if interference > self.max_interference {
            return Some(EscalationReason::HighInterference {
                observed: interference,
                threshold: self.max_interference,
            });
        }
        if self.escalate_on_bridge_bleed && bridge_bleed > self.max_bridge_bleed {
            return Some(EscalationReason::HighBridgeBleed {
                observed: bridge_bleed,
                threshold: self.max_bridge_bleed,
            });
        }
        None
    }
}

/// Why the system escalated from L1 to L2.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EscalationReason {
    /// L1 confidence was below the threshold.
    LowConfidence {
        /// Observed confidence.
        observed: f32,
        /// Threshold it fell below.
        threshold: f32,
    },
    /// L1 interference was above the threshold (semantic space crowded).
    HighInterference {
        /// Observed interference.
        observed: f32,
        /// Threshold it exceeded.
        threshold: f32,
    },
    /// L1 bridge bleed was above the threshold (cross-space signal leaking).
    HighBridgeBleed {
        /// Observed bleed.
        observed: f32,
        /// Threshold it exceeded.
        threshold: f32,
    },
}

impl EscalationReason {
    /// Stable snake_case label for logs and metrics.
    pub fn label(&self) -> &'static str {
        match self {
            Self::LowConfidence { .. } => "low_confidence",
            Self::HighInterference { .. } => "high_interference",
            Self::HighBridgeBleed { .. } => "high_bridge_bleed",
        }
    }
}

// ============================================================================
// Multi-resolution compiled program — carries L1 + L2 holograms
// ============================================================================

/// A compiled memory program with multi-resolution holograms (D1).
///
/// L1 (1024-dim) is always present — it's the fast path. L2 (65536-dim) is
/// optional — not every program needs the deep path. When L2 is present, the
/// query protocol uses the [`EscalationPolicy`] to decide whether to project
/// against L1 only (fast) or escalate to L2 (deep).
///
/// This is a portable type — it uses `Vec<f32>` instead of `Tensor` so it can
/// be serialized and federated without Candle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiResolutionProgram {
    /// URN identifying this program.
    pub urn: String,
    /// L1 holograms (always present, 1024-dim).
    pub l1: ResolutionLayer,
    /// L2 holograms (optional, 65536-dim). None when the program is L1-only.
    pub l2: Option<ResolutionLayer>,
    /// Escalation policy for L1→L2 decisions.
    pub escalation: EscalationPolicy,
    /// Compilation statistics.
    pub stats: MultiResolutionStats,
}

/// A single resolution layer (L1 or L2) — space holograms + item holograms
/// at a fixed dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionLayer {
    /// Which tier this is.
    pub tier: ResolutionTier,
    /// Dimension of the holograms in this layer.
    pub dimension: usize,
    /// Space holograms — superposition of all item vectors in each space.
    pub spaces: Vec<ResolutionSpace>,
    /// Individual item holograms (for item-level ranking).
    pub items: Vec<ResolutionItem>,
}

/// A space hologram at a specific resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionSpace {
    /// Space identifier.
    pub space_id: String,
    /// Superposed space hologram.
    pub hologram: Vec<f32>,
    /// Items folded into the hologram.
    pub item_count: usize,
}

/// An item hologram at a specific resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionItem {
    /// Item identifier.
    pub id: String,
    /// Owning space id.
    pub space_id: String,
    /// Item hologram.
    pub hologram: Vec<f32>,
    /// What kind of content (code, document, etc.) — mirrors DataSourceItem.kind.
    pub kind: String,
}

/// Statistics for a multi-resolution compilation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiResolutionStats {
    /// Items compiled across all spaces.
    pub total_items: usize,
    /// Spaces compiled.
    pub total_spaces: usize,
    /// L1 hologram dimension.
    pub l1_dimension: usize,
    /// L2 dimension, if an L2 layer was built.
    pub l2_dimension: Option<usize>,
    /// Embedder identity string.
    pub embedder: String,
    /// Compilation timestamp.
    pub compiled_at: chrono::DateTime<chrono::Utc>,
}

impl MultiResolutionProgram {
    /// Whether this program has an L2 layer.
    pub fn has_l2(&self) -> bool {
        self.l2.is_some()
    }

    /// Get the L2 layer, or None if not present.
    pub fn l2(&self) -> Option<&ResolutionLayer> {
        self.l2.as_ref()
    }

    /// The dimension of the L1 layer.
    pub fn l1_dimension(&self) -> usize {
        self.l1.dimension
    }

    /// The dimension of the L2 layer, or None if not present.
    pub fn l2_dimension(&self) -> Option<usize> {
        self.l2.as_ref().map(|l| l.dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_tier_defaults() {
        assert_eq!(ResolutionTier::L1.default_dim(), 1024);
        assert_eq!(ResolutionTier::L2.default_dim(), 65536);
    }

    #[test]
    fn resolution_tier_parse() {
        assert_eq!(
            ResolutionTier::from_str_lossy("l1"),
            Some(ResolutionTier::L1)
        );
        assert_eq!(
            ResolutionTier::from_str_lossy("L2"),
            Some(ResolutionTier::L2)
        );
        assert_eq!(ResolutionTier::from_str_lossy("l3"), None);
    }

    #[test]
    fn escalation_on_low_confidence() {
        let policy = EscalationPolicy::default();
        let reason = policy.should_escalate(0.2, 0.1, 0.05);
        assert!(matches!(
            reason,
            Some(EscalationReason::LowConfidence { .. })
        ));
    }

    #[test]
    fn escalation_on_high_interference() {
        let policy = EscalationPolicy::default();
        // Confidence is OK (0.6 > 0.4) but interference is high (0.7 > 0.5).
        let reason = policy.should_escalate(0.6, 0.7, 0.05);
        assert!(matches!(
            reason,
            Some(EscalationReason::HighInterference { .. })
        ));
    }

    #[test]
    fn escalation_on_high_bridge_bleed() {
        let policy = EscalationPolicy::default();
        // Confidence and interference OK, but bridge bleed high (0.5 > 0.3).
        let reason = policy.should_escalate(0.6, 0.2, 0.5);
        assert!(matches!(
            reason,
            Some(EscalationReason::HighBridgeBleed { .. })
        ));
    }

    #[test]
    fn no_escalation_when_l1_is_sufficient() {
        let policy = EscalationPolicy::default();
        // All metrics within thresholds.
        let reason = policy.should_escalate(0.8, 0.2, 0.1);
        assert!(reason.is_none());
    }

    #[test]
    fn escalation_disabled_for_bridge_bleed() {
        let policy = EscalationPolicy {
            escalate_on_bridge_bleed: false,
            ..Default::default()
        };
        // High bridge bleed but escalation disabled — should not escalate.
        let reason = policy.should_escalate(0.8, 0.2, 0.9);
        assert!(reason.is_none());
    }

    #[test]
    fn multi_resolution_program_l2_detection() {
        let l1_layer = ResolutionLayer {
            tier: ResolutionTier::L1,
            dimension: 1024,
            spaces: vec![],
            items: vec![],
        };
        // L1-only program.
        let program_l1_only = MultiResolutionProgram {
            urn: "test#v1".to_string(),
            l1: l1_layer.clone(),
            l2: None,
            escalation: EscalationPolicy::default(),
            stats: MultiResolutionStats {
                total_items: 0,
                total_spaces: 0,
                l1_dimension: 1024,
                l2_dimension: None,
                embedder: "hash".to_string(),
                compiled_at: chrono::Utc::now(),
            },
        };
        assert!(!program_l1_only.has_l2());
        assert_eq!(program_l1_only.l2_dimension(), None);

        // Program with L2.
        let l2_layer = ResolutionLayer {
            tier: ResolutionTier::L2,
            dimension: 65536,
            spaces: vec![],
            items: vec![],
        };
        let program_with_l2 = MultiResolutionProgram {
            urn: "test#v1".to_string(),
            l1: l1_layer,
            l2: Some(l2_layer),
            escalation: EscalationPolicy::default(),
            stats: MultiResolutionStats {
                total_items: 0,
                total_spaces: 0,
                l1_dimension: 1024,
                l2_dimension: Some(65536),
                embedder: "hash".to_string(),
                compiled_at: chrono::Utc::now(),
            },
        };
        assert!(program_with_l2.has_l2());
        assert_eq!(program_with_l2.l2_dimension(), Some(65536));
    }
}

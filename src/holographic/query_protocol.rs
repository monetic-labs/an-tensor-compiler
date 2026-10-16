//! HRR Query Protocol — standard interface for any LLM to query HRR corpora.
//!
//! Per `HRR_INTELLIGENCE_PLATFORM_SPEC.md` P2: lifts HRR projection out of the
//! inference-server into a general protocol any LLM client can implement. The
//! types use `Vec<f32>` instead of `Tensor` for portability — no Candle
//! dependency required.
//!
//! This module is always available (not gated behind the `io` feature) so any
//! LLM client, federation node, or external tool can implement the query
//! protocol without requiring the full tensor backend.

use crate::holographic::diagnostics::{
    BridgeContribution, ProjectionDiagnostics, SpaceContribution,
};
use crate::holographic::policy::ProjectionPolicy;
use crate::Result;

// ============================================================================
// HrrEmbedder — any embedder that produces a fixed-dimension vector
// (P2.1: Standard embedding interface)
// ============================================================================

/// Any embedder that produces a fixed-dimension vector can be a query frontend.
///
/// Reference implementations (in the inference-server and compiler crate):
/// - `ArcticEmbedL` (1024-dim, current default) — wraps the pool's embedder
/// - `HashEncoder` (any-dim, no GPU) — the fallback for dev/testing
/// - `ExternalEmbedder` (wraps any HTTP-based embedding API)
///
/// The protocol doesn't care which embedder produced the query vector — it only
/// cares that the dimension matches the hologram.
#[async_trait::async_trait]
pub trait HrrEmbedder: Send + Sync {
    /// The dimension of the vectors this embedder produces.
    fn dim(&self) -> usize;

    /// Embed a text string into a fixed-dimension vector.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed a batch of text strings (for efficiency). Default implementation
    /// iterates; implementations with batch APIs should override.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }
}

// ============================================================================
// HrrQuery / HrrQueryResult — the standard query interface
// (P2.2: Standard query interface)
// ============================================================================

/// A query against an HRR memory program.
///
/// The `query_vector` is pre-encoded by any [`HrrEmbedder`] — the protocol is
/// embedder-agnostic. The `program_urn` identifies which compiled memory
/// program to project against.
#[derive(Debug, Clone)]
pub struct HrrQuery {
    /// URN identifying the memory program to query
    /// (e.g. `hrr:program:acme/code-context#v1`).
    pub program_urn: String,
    /// Pre-encoded query vector from any [`HrrEmbedder`]. The dimension must
    /// match the program's hologram dimension.
    pub query_vector: Vec<f32>,
    /// Projection policy (bridged, local, hybrid — see [`ProjectionPolicy`]).
    pub policy: ProjectionPolicy,
    /// Maximum number of items to return.
    pub max_results: usize,
}

/// The result of an HRR query — items, spaces, and diagnostics.
///
/// The [`ProjectionDiagnostics`] are machine-readable reasoning inputs for the
/// LLM (per E1: Think→Probe→Refine). They are not just human-readable
/// diagnostics — they tell the LLM where the interference is, which bridges
/// contributed, and how confident the result is.
#[derive(Debug, Clone)]
pub struct HrrQueryResult {
    /// The highest-scoring space (if any).
    pub primary_space: Option<String>,
    /// Spaces that contributed to the projection, ranked by confidence.
    pub spaces: Vec<HrrSpaceScore>,
    /// Items that matched the query, ranked by confidence.
    pub items: Vec<HrrItemScore>,
    /// Projection diagnostics — machine-readable reasoning inputs (E1).
    pub diagnostics: ProjectionDiagnostics,
    /// Spaces that contributed to the projection (from the substrate).
    pub contributing_spaces: Vec<SpaceContribution>,
    /// Bridges that were traversed during projection (from the substrate).
    pub contributing_bridges: Vec<BridgeContribution>,
}

/// A scored item in the query result.
#[derive(Debug, Clone)]
pub struct HrrItemScore {
    /// Item identifier (from the manifest).
    pub id: String,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
}

/// A scored space in the query result.
#[derive(Debug, Clone)]
pub struct HrrSpaceScore {
    /// Space identifier (from the manifest).
    pub id: String,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
}

impl HrrQueryResult {
    /// Get the top N items.
    pub fn top_items(&self, n: usize) -> &[HrrItemScore] {
        &self.items[..n.min(self.items.len())]
    }

    /// The primary space (highest scored).
    pub fn primary_space(&self) -> Option<&str> {
        self.spaces.first().map(|s| s.id.as_str())
    }
}

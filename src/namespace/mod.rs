//! Namespace-Based Compartmentalization
//!
//! Provides gradient isolation between different Tensor Logic domains.
//! Each namespace has its own parameter space and gradients don't cross
//! boundaries unless explicitly federated.
//!
//! ## Two-Layer Namespace Model
//!
//! Namespaces operate at two levels:
//!
//! ### 1. Root Namespace (NamespaceId = u32)
//!
//! The gradient isolation boundary. Used for efficient tensor indexing.
//! Well-known roots: `TRADING`, `PIPELINE`, `CHAT`.
//!
//! ### 2. Hierarchical Path (NamespacePath)
//!
//! Dot-separated paths for fine-grained organization within a root namespace.
//! Used for CRDT keys, human discovery, and avoiding matrix pollution.
//!
//! ```text
//! an-ecosystem.devops.pipeline.metrics
//! └──────┬────┘└──┬──┘└──┬───┘└──┬───┘
//!     owner    domain  sub    data-type
//! ```
//!
//! ## Design Principles
//!
//! From research team guidance on avoiding namespace pollution:
//!
//! 1. **Ownership First**: Prefix by project/team (`an-*` or `org-*`)
//! 2. **Domain Grouping**: Subnamespaces by functional domain (devops, trading)
//! 3. **Lifecycle Separation**: Keep volatile/stable data in different paths
//! 4. **Depth Limits**: 2-4 levels max for human intuition
//! 5. **Human-Centric**: Readable paths that work as query selectors
//!
//! ## Example
//!
//! ```ignore
//! use an_tensor_compiler::namespace::*;
//!
//! // Root namespace for gradient isolation
//! let root = PIPELINE;
//!
//! // Hierarchical path for CRDT organization
//! let path = NamespacePath::new("an-ecosystem.devops.pipeline.metrics")?;
//! assert_eq!(path.owner(), "an-ecosystem");
//! assert_eq!(path.domain(), Some("devops"));
//! assert_eq!(path.root_namespace(), PIPELINE);
//!
//! // Convert to CRDT key
//! let crdt_key = path.to_crdt_key("latest");
//! // "tensor/an-ecosystem/devops/pipeline/metrics/latest"
//! ```

use candle_core::Tensor;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Namespace identifier (u32 for efficient tensor indexing)
///
/// # Allocation Strategy
///
/// - **0-99**: Reserved for well-known namespaces (example defaults provided below)
/// - **100+**: Dynamic allocation via `NamespaceRegistry::register()`
///
/// # Defining Your Own Namespaces
///
/// The pre-defined constants (`TRADING`, `PIPELINE`, `CHAT`) are provided as
/// examples for common domain patterns. You can:
///
/// 1. **Use them directly** if your domains map naturally
/// 2. **Register custom domains** via [`NamespaceRegistry::register()`] (IDs start at 100)
/// 3. **Define your own constants** in the reserved range (0-99) for performance-critical paths
///
/// ```rust,ignore
/// use an_tensor_compiler::namespace::NamespaceId;
///
/// // Option 1: Use the provided defaults
/// use an_tensor_compiler::namespace::PIPELINE;
///
/// // Option 2: Define your own well-known namespace
/// pub const INFERENCE: NamespaceId = 10;
/// pub const TRAINING: NamespaceId = 11;
///
/// // Option 3: Dynamic registration
/// let mut registry = NamespaceRegistry::new();
/// let my_domain = registry.register("my-custom-domain");
/// ```
pub type NamespaceId = u32;

/// Reserved range for well-known namespaces (0-99)
pub const RESERVED_NAMESPACE_MAX: NamespaceId = 99;

/// First ID available for dynamic allocation
pub const DYNAMIC_NAMESPACE_START: NamespaceId = 100;

/// Example well-known namespace: Trading / real-time decision engine
///
/// Pre-defined for convenience. See [`NamespaceId`] docs for how to define your own.
pub const TRADING: NamespaceId = 0;

/// Example well-known namespace: Pipeline / CI/CD / batch processing
///
/// Pre-defined for convenience. See [`NamespaceId`] docs for how to define your own.
pub const PIPELINE: NamespaceId = 1;

/// Example well-known namespace: Chat / conversational agent
///
/// Pre-defined for convenience. See [`NamespaceId`] docs for how to define your own.
pub const CHAT: NamespaceId = 2;

// =============================================================================
// CRDT String Conversion Helpers
// =============================================================================
//
// These functions convert between internal u32 NamespaceIds and the UPPERCASE
// string format used in CRDT keys (e.g., "tensor/metrics/PIPELINE/latest").
//
// The u32 representation is kept internally for efficient tensor indexing.
// String conversion happens at the boundary when interacting with CRDT.

/// Convert namespace ID to CRDT-compatible string
///
/// Returns UPPERCASE string for use in CRDT keys.
///
/// # Example
/// ```
/// use an_tensor_compiler::namespace::{namespace_to_string, PIPELINE};
///
/// assert_eq!(namespace_to_string(PIPELINE), "PIPELINE");
/// ```
pub fn namespace_to_string(ns: NamespaceId) -> &'static str {
    match ns {
        TRADING => "TRADING",
        PIPELINE => "PIPELINE",
        CHAT => "CHAT",
        _ => "UNKNOWN",
    }
}

/// Parse namespace from CRDT string
///
/// Accepts UPPERCASE strings from CRDT keys.
///
/// # Example
/// ```
/// use an_tensor_compiler::namespace::{namespace_from_string, TRADING};
///
/// assert_eq!(namespace_from_string("TRADING"), Some(TRADING));
/// assert_eq!(namespace_from_string("invalid"), None);
/// ```
pub fn namespace_from_string(s: &str) -> Option<NamespaceId> {
    match s {
        "TRADING" => Some(TRADING),
        "PIPELINE" => Some(PIPELINE),
        "CHAT" => Some(CHAT),
        _ => None,
    }
}

/// Get all well-known namespaces as (id, string) pairs
///
/// Useful for iteration and validation.
///
/// # Example
/// ```
/// use an_tensor_compiler::namespace::all_namespaces;
///
/// for (id, name) in all_namespaces() {
///     println!("Namespace {}: {}", id, name);
/// }
/// ```
pub fn all_namespaces() -> &'static [(NamespaceId, &'static str)] {
    &[
        (TRADING, "TRADING"),
        (PIPELINE, "PIPELINE"),
        (CHAT, "CHAT"),
    ]
}

// =============================================================================
// Hierarchical Namespace Paths
// =============================================================================
//
// Hierarchical paths provide fine-grained organization within root namespaces.
// Format: {owner}.{domain}.{subdomain}.{data-type}
//
// Examples:
//   an-ecosystem.devops.pipeline.metrics
//   org-lttr.trading.regime.states
//   an-agent.chat.sessions.embeddings
//
// Guidelines (from research team):
// - Max depth: 2-4 levels for human intuition
// - Owner prefix: an-* (foundation) or org-* (partners)
// - Domain grouping: devops, trading, monitoring, etc.
// - Lifecycle separation: Keep volatile (.states) separate from stable (.configs)

/// Owner type for namespace paths
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OwnerType {
    /// Foundation team (an-*)
    Foundation,
    /// Partner organization (org-*)
    Organization,
    /// Unknown/custom owner
    Custom,
}

/// A hierarchical namespace path for fine-grained organization
///
/// Provides human-readable, queryable paths for CRDT keys while maintaining
/// compatibility with the u32 NamespaceId for tensor operations.
///
/// # Path Structure
///
/// ```text
/// {owner}.{domain}.{subdomain}.{data-type}
///
/// an-ecosystem.devops.pipeline.metrics
/// └─────┬────┘└──┬──┘└───┬───┘└──┬───┘
///    owner    domain   sub    type
/// ```
///
/// # Guidelines
///
/// - **Owner** (required): `an-*` for foundation, `org-*` for partners
/// - **Domain** (recommended): Functional area like `devops`, `trading`, `monitoring`
/// - **Subdomain**: Specific system or feature
/// - **Data-type**: Kind of data: `metrics`, `states`, `configs`, `models`
///
/// # Anti-Pollution Rules
///
/// 1. Never mix volatile (`states`) with stable (`configs`) at same level
/// 2. Keep different domains separate (don't let trading leak into devops)
/// 3. Use lifecycle-aware suffixes: `.states` for ephemeral, `.models` for persistent
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamespacePath {
    /// The full dot-separated path
    path: String,
    /// Parsed segments
    segments: Vec<String>,
}

impl NamespacePath {
    /// Maximum recommended depth for namespace paths
    pub const MAX_RECOMMENDED_DEPTH: usize = 4;

    /// Create a new namespace path from a dot-separated string
    ///
    /// # Validation
    ///
    /// - Must have at least one segment (owner)
    /// - Segments must be non-empty and contain only valid chars
    /// - Warns (but allows) paths deeper than 4 levels
    ///
    /// # Example
    ///
    /// ```
    /// use an_tensor_compiler::namespace::NamespacePath;
    ///
    /// let path = NamespacePath::new("an-ecosystem.devops.pipeline").unwrap();
    /// assert_eq!(path.owner(), "an-ecosystem");
    /// assert_eq!(path.depth(), 3);
    /// ```
    pub fn new(path: &str) -> crate::Result<Self> {
        let segments: Vec<String> = path.split('.').map(|s| s.to_string()).collect();

        // Validate: at least owner required
        if segments.is_empty() || segments[0].is_empty() {
            return Err(crate::TensorCoreError::Namespace(
                "Namespace path must have at least an owner segment".into(),
            ));
        }

        // Validate: all segments non-empty
        for (i, seg) in segments.iter().enumerate() {
            if seg.is_empty() {
                return Err(crate::TensorCoreError::Namespace(format!(
                    "Empty segment at position {} in path '{}'",
                    i, path
                )));
            }

            // Validate characters: alphanumeric, dash, underscore
            if !seg
                .chars()
                .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
            {
                return Err(crate::TensorCoreError::Namespace(format!(
                    "Invalid characters in segment '{}' - use alphanumeric, dash, or underscore",
                    seg
                )));
            }
        }

        Ok(Self {
            path: path.to_string(),
            segments,
        })
    }

    /// Create a path from individual segments
    ///
    /// # Example
    ///
    /// ```
    /// use an_tensor_compiler::namespace::NamespacePath;
    ///
    /// let path = NamespacePath::from_segments(&["an-ecosystem", "devops", "pipeline"]).unwrap();
    /// assert_eq!(path.as_str(), "an-ecosystem.devops.pipeline");
    /// ```
    pub fn from_segments(segments: &[&str]) -> crate::Result<Self> {
        Self::new(&segments.join("."))
    }

    /// Get the full path string
    pub fn as_str(&self) -> &str {
        &self.path
    }

    /// Get path segments
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// Get the owner segment (first segment)
    ///
    /// This identifies the team/project that owns this namespace.
    pub fn owner(&self) -> &str {
        &self.segments[0]
    }

    /// Get the owner type
    pub fn owner_type(&self) -> OwnerType {
        let owner = self.owner();
        if owner.starts_with("an-") {
            OwnerType::Foundation
        } else if owner.starts_with("org-") {
            OwnerType::Organization
        } else {
            OwnerType::Custom
        }
    }

    /// Get the domain segment (second segment, if present)
    ///
    /// The domain groups related functionality: devops, trading, monitoring, etc.
    pub fn domain(&self) -> Option<&str> {
        self.segments.get(1).map(|s| s.as_str())
    }

    /// Get the subdomain segment (third segment, if present)
    pub fn subdomain(&self) -> Option<&str> {
        self.segments.get(2).map(|s| s.as_str())
    }

    /// Get the data-type segment (fourth segment, if present)
    ///
    /// Common data types: metrics, states, configs, models, embeddings
    pub fn data_type(&self) -> Option<&str> {
        self.segments.get(3).map(|s| s.as_str())
    }

    /// Get the depth of this path (number of segments)
    pub fn depth(&self) -> usize {
        self.segments.len()
    }

    /// Check if path exceeds recommended depth
    pub fn exceeds_recommended_depth(&self) -> bool {
        self.depth() > Self::MAX_RECOMMENDED_DEPTH
    }

    /// Extract version from the last segment if present
    ///
    /// Recognizes patterns like `.v2`, `.v1_0`, `.v2_1_3`
    ///
    /// # Example
    ///
    /// ```
    /// use an_tensor_compiler::namespace::NamespacePath;
    ///
    /// let path = NamespacePath::new("an-ecosystem.tensor.regime.v2").unwrap();
    /// assert_eq!(path.version(), Some("v2"));
    ///
    /// let path = NamespacePath::new("an-ecosystem.tensor.regime.states").unwrap();
    /// assert_eq!(path.version(), None);
    /// ```
    pub fn version(&self) -> Option<&str> {
        self.segments.last().and_then(|seg| {
            if seg.starts_with('v') && seg.len() > 1 {
                // Check if rest is numeric (with optional underscores for minor/patch)
                let rest = &seg[1..];
                if rest.chars().all(|c| c.is_ascii_digit() || c == '_') {
                    Some(seg.as_str())
                } else {
                    None
                }
            } else {
                None
            }
        })
    }

    /// Create a versioned variant of this path
    ///
    /// # Example
    ///
    /// ```
    /// use an_tensor_compiler::namespace::NamespacePath;
    ///
    /// let path = NamespacePath::new("an-ecosystem.tensor.regime").unwrap();
    /// let v2 = path.with_version(2, None, None).unwrap();
    /// assert_eq!(v2.as_str(), "an-ecosystem.tensor.regime.v2");
    ///
    /// let v2_1 = path.with_version(2, Some(1), None).unwrap();
    /// assert_eq!(v2_1.as_str(), "an-ecosystem.tensor.regime.v2_1");
    /// ```
    pub fn with_version(
        &self,
        major: u32,
        minor: Option<u32>,
        patch: Option<u32>,
    ) -> crate::Result<Self> {
        let version_str = match (minor, patch) {
            (Some(min), Some(pat)) => format!("v{}_{}{}", major, min, pat),
            (Some(min), None) => format!("v{}_{}", major, min),
            (None, _) => format!("v{}", major),
        };
        self.child(&version_str)
    }

    /// Parse semantic version from path if present
    ///
    /// Returns (major, minor, patch) tuple.
    ///
    /// # Example
    ///
    /// ```
    /// use an_tensor_compiler::namespace::NamespacePath;
    ///
    /// let path = NamespacePath::new("an-ecosystem.tensor.regime.v2_1_3").unwrap();
    /// assert_eq!(path.parse_version(), Some((2, Some(1), Some(3))));
    /// ```
    pub fn parse_version(&self) -> Option<(u32, Option<u32>, Option<u32>)> {
        self.version().and_then(|v| {
            let v = v.strip_prefix('v')?;
            let parts: Vec<&str> = v.split('_').collect();
            let major = parts.first()?.parse().ok()?;
            let minor = parts.get(1).and_then(|s| s.parse().ok());
            let patch = parts.get(2).and_then(|s| s.parse().ok());
            Some((major, minor, patch))
        })
    }

    /// Get the base path without version suffix
    ///
    /// # Example
    ///
    /// ```
    /// use an_tensor_compiler::namespace::NamespacePath;
    ///
    /// let path = NamespacePath::new("an-ecosystem.tensor.regime.v2").unwrap();
    /// let base = path.without_version().unwrap();
    /// assert_eq!(base.as_str(), "an-ecosystem.tensor.regime");
    /// ```
    pub fn without_version(&self) -> Option<Self> {
        if self.version().is_some() && self.segments.len() > 1 {
            let base_segments: Vec<&str> = self.segments[..self.segments.len() - 1]
                .iter()
                .map(|s| s.as_str())
                .collect();
            Self::from_segments(&base_segments).ok()
        } else {
            None
        }
    }

    /// Infer the root NamespaceId from the owner
    ///
    /// Maps well-known owners to their root namespaces.
    pub fn root_namespace(&self) -> Option<NamespaceId> {
        match self.owner() {
            "org-lttr" | "an-trading" => Some(TRADING),
            "an-ecosystem" | "an-pipeline" => Some(PIPELINE),
            "an-agent" | "an-chat" => Some(CHAT),
            _ => None,
        }
    }

    /// Check if this path is a child of another path
    ///
    /// # Example
    ///
    /// ```
    /// use an_tensor_compiler::namespace::NamespacePath;
    ///
    /// let parent = NamespacePath::new("an-ecosystem.devops").unwrap();
    /// let child = NamespacePath::new("an-ecosystem.devops.pipeline").unwrap();
    /// assert!(child.is_child_of(&parent));
    /// ```
    pub fn is_child_of(&self, parent: &NamespacePath) -> bool {
        if self.segments.len() <= parent.segments.len() {
            return false;
        }
        self.segments
            .iter()
            .zip(parent.segments.iter())
            .all(|(a, b)| a == b)
    }

    /// Create a child path by appending a segment
    ///
    /// # Example
    ///
    /// ```
    /// use an_tensor_compiler::namespace::NamespacePath;
    ///
    /// let parent = NamespacePath::new("an-ecosystem.devops").unwrap();
    /// let child = parent.child("pipeline").unwrap();
    /// assert_eq!(child.as_str(), "an-ecosystem.devops.pipeline");
    /// ```
    pub fn child(&self, segment: &str) -> crate::Result<Self> {
        Self::new(&format!("{}.{}", self.path, segment))
    }

    /// Convert to a CRDT key with a suffix
    ///
    /// # Example
    ///
    /// ```
    /// use an_tensor_compiler::namespace::NamespacePath;
    ///
    /// let path = NamespacePath::new("an-ecosystem.devops.pipeline.metrics").unwrap();
    /// let key = path.to_crdt_key("latest");
    /// assert_eq!(key, "tensor/an-ecosystem/devops/pipeline/metrics/latest");
    /// ```
    pub fn to_crdt_key(&self, suffix: &str) -> String {
        let path_part = self.segments.join("/");
        format!("tensor/{}/{}", path_part, suffix)
    }

    /// Parse a CRDT key back into a NamespacePath
    ///
    /// # Example
    ///
    /// ```
    /// use an_tensor_compiler::namespace::NamespacePath;
    ///
    /// let (path, suffix) = NamespacePath::from_crdt_key("tensor/an-ecosystem/devops/pipeline/latest").unwrap();
    /// assert_eq!(path.as_str(), "an-ecosystem.devops.pipeline");
    /// assert_eq!(suffix, "latest");
    /// ```
    pub fn from_crdt_key(key: &str) -> crate::Result<(Self, String)> {
        let key = key.strip_prefix("tensor/").ok_or_else(|| {
            crate::TensorCoreError::Namespace("CRDT key must start with 'tensor/'".into())
        })?;

        let parts: Vec<&str> = key.split('/').collect();
        if parts.len() < 2 {
            return Err(crate::TensorCoreError::Namespace(
                "CRDT key must have at least owner and suffix".into(),
            ));
        }

        // Last part is the suffix, rest is the path
        let suffix = parts.last().unwrap().to_string();
        let path_parts = &parts[..parts.len() - 1];

        let path = Self::new(&path_parts.join("."))?;
        Ok((path, suffix))
    }
}

impl std::fmt::Display for NamespacePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.path)
    }
}

impl std::str::FromStr for NamespacePath {
    type Err = crate::TensorCoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

/// Well-known data type suffixes for namespace paths
pub mod data_types {
    /// Numeric metrics (loss, accuracy, latency)
    pub const METRICS: &str = "metrics";
    /// Volatile runtime states
    pub const STATES: &str = "states";
    /// Stable configuration tensors
    pub const CONFIGS: &str = "configs";
    /// Trained model weights
    pub const MODELS: &str = "models";
    /// Embedding vectors
    pub const EMBEDDINGS: &str = "embeddings";
    /// Training checkpoints
    pub const CHECKPOINTS: &str = "checkpoints";
    /// Logs and audit trails
    pub const LOGS: &str = "logs";
}

/// Well-known domain prefixes
pub mod domains {
    /// DevOps: CI/CD, builds, deploys
    pub const DEVOPS: &str = "devops";
    /// Trading: Market data, positions, regimes
    pub const TRADING: &str = "trading";
    /// Monitoring: Health checks, alerts
    pub const MONITORING: &str = "monitoring";
    /// Security: Auth, permissions, scans
    pub const SECURITY: &str = "security";
    /// Integration: External service connectors
    pub const INTEGRATION: &str = "integration";
    /// Meta: Cross-domain discovery, schemas
    pub const META: &str = "meta";
}

// =============================================================================
// Namespace Metadata (Mini Data Catalog)
// =============================================================================
//
// Rich metadata for namespace paths, enabling discovery and debugging.
// Stored at {path}.meta in CRDT for human-friendly exploration.

/// Metadata for a namespace path
///
/// Provides a mini data catalog for namespace discovery and debugging.
/// Store at `{path}.meta` keys in CRDT.
///
/// # Example
///
/// ```
/// use an_tensor_compiler::namespace::{NamespaceMeta, NamespacePath};
///
/// let path = NamespacePath::new("an-ecosystem.devops.pipeline.metrics").unwrap();
/// let meta = NamespaceMeta::new(&path)
///     .with_description("CI/CD pipeline build and deploy metrics")
///     .with_owner_contact("infra-team@example.com")
///     .with_schema_fingerprint("sha256:abc123...")
///     .with_tensor_shape(&[1000, 64]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceMeta {
    /// The path this metadata describes
    pub path: String,

    /// Human-readable description
    pub description: Option<String>,

    /// Contact for the owner (email, slack channel, etc.)
    pub owner_contact: Option<String>,

    /// Schema fingerprint for detecting breaking changes
    ///
    /// When tensor dimensions or semantics change, bump this.
    /// Format: "sha256:{hash}" or "v{major}.{minor}"
    pub schema_fingerprint: Option<String>,

    /// Expected tensor shape (for validation)
    pub tensor_shape: Option<Vec<usize>>,

    /// Tensor dtype (F32, F64, etc.)
    pub tensor_dtype: Option<String>,

    /// Last recompute timestamp (ISO 8601)
    pub last_recompute: Option<String>,

    /// Recompute frequency hint (e.g., "per-epoch", "daily", "on-demand")
    pub recompute_frequency: Option<String>,

    /// Data coverage stats (e.g., "95% of trades", "all pipelines")
    pub coverage: Option<String>,

    /// Upstream dependencies (paths this data derives from)
    pub upstream: Vec<String>,

    /// Downstream consumers (paths that depend on this)
    pub downstream: Vec<String>,

    /// Additional custom metadata
    pub custom: HashMap<String, String>,
}

impl NamespaceMeta {
    /// Create metadata for a namespace path
    pub fn new(path: &NamespacePath) -> Self {
        Self {
            path: path.as_str().to_string(),
            description: None,
            owner_contact: None,
            schema_fingerprint: None,
            tensor_shape: None,
            tensor_dtype: None,
            last_recompute: None,
            recompute_frequency: None,
            coverage: None,
            upstream: Vec::new(),
            downstream: Vec::new(),
            custom: HashMap::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Set owner contact
    pub fn with_owner_contact(mut self, contact: &str) -> Self {
        self.owner_contact = Some(contact.to_string());
        self
    }

    /// Set schema fingerprint
    ///
    /// Use semantic versioning (v2, v2.1) or content hash (sha256:abc...)
    pub fn with_schema_fingerprint(mut self, fingerprint: &str) -> Self {
        self.schema_fingerprint = Some(fingerprint.to_string());
        self
    }

    /// Set expected tensor shape
    pub fn with_tensor_shape(mut self, shape: &[usize]) -> Self {
        self.tensor_shape = Some(shape.to_vec());
        self
    }

    /// Set tensor dtype
    pub fn with_tensor_dtype(mut self, dtype: &str) -> Self {
        self.tensor_dtype = Some(dtype.to_string());
        self
    }

    /// Set last recompute timestamp
    pub fn with_last_recompute(mut self, timestamp: &str) -> Self {
        self.last_recompute = Some(timestamp.to_string());
        self
    }

    /// Set recompute frequency hint
    pub fn with_recompute_frequency(mut self, freq: &str) -> Self {
        self.recompute_frequency = Some(freq.to_string());
        self
    }

    /// Set coverage description
    pub fn with_coverage(mut self, coverage: &str) -> Self {
        self.coverage = Some(coverage.to_string());
        self
    }

    /// Add upstream dependency
    pub fn with_upstream(mut self, path: &str) -> Self {
        self.upstream.push(path.to_string());
        self
    }

    /// Add downstream consumer
    pub fn with_downstream(mut self, path: &str) -> Self {
        self.downstream.push(path.to_string());
        self
    }

    /// Add custom metadata
    pub fn with_custom(mut self, key: &str, value: &str) -> Self {
        self.custom.insert(key.to_string(), value.to_string());
        self
    }

    /// Generate CRDT key for this metadata
    pub fn to_crdt_key(&self) -> String {
        format!("tensor/{}/meta", self.path.replace('.', "/"))
    }

    /// Check if schema fingerprint indicates a breaking change
    ///
    /// Compares major version if using semantic versioning
    pub fn is_compatible_with(&self, other: &NamespaceMeta) -> bool {
        match (&self.schema_fingerprint, &other.schema_fingerprint) {
            (Some(a), Some(b)) => {
                // If both use semantic versioning, compare major
                if a.starts_with('v') && b.starts_with('v') {
                    let parse_major = |s: &str| -> Option<u32> {
                        s.strip_prefix('v')
                            .and_then(|rest| rest.split('_').next())
                            .and_then(|major| major.split('.').next())
                            .and_then(|m| m.parse().ok())
                    };
                    parse_major(a) == parse_major(b)
                } else {
                    // Hash comparison - must match exactly
                    a == b
                }
            }
            // If either is missing, assume compatible (no schema enforcement)
            _ => true,
        }
    }
}

// =============================================================================
// Namespace Registry Trait (for CRDT backing)
// =============================================================================

/// Registration result from a namespace registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceRegistration {
    /// The assigned namespace ID
    pub id: NamespaceId,
    /// Whether this was a new registration (true) or existing (false)
    pub newly_registered: bool,
}

/// Trait for namespace registry implementations
///
/// This trait allows different backing stores for namespace registration:
/// - `LocalNamespaceRegistry`: In-memory, single-process (default)
/// - `CrdtNamespaceRegistry`: CRDT-backed, distributed (implemented by an-ecosystem)
///
/// # CRDT Implementation Guide
///
/// For an-ecosystem to implement a CRDT-backed registry:
///
/// ```ignore
/// use an_tensor_compiler::namespace::{NamespaceRegistryProvider, NamespaceId, NamespaceRegistration};
///
/// pub struct CrdtNamespaceRegistry {
///     connection: StateConnection,
///     cache: RwLock<HashMap<String, NamespaceId>>,
/// }
///
/// impl NamespaceRegistryProvider for CrdtNamespaceRegistry {
///     fn register(&mut self, name: &str) -> NamespaceRegistration {
///         // 1. Check local cache
///         // 2. If not found, check CRDT: tensor/namespaces/{name}
///         // 3. If not in CRDT, claim next ID atomically
///         // 4. Store in CRDT and local cache
///         // 5. Return registration result
///     }
///     // ... other methods
/// }
/// ```
pub trait NamespaceRegistryProvider: Send + Sync {
    /// Register a new namespace (or return existing ID if already registered)
    ///
    /// Returns the namespace ID and whether it was newly registered.
    fn register(&mut self, name: &str) -> NamespaceRegistration;

    /// Get namespace ID by name
    fn get(&self, name: &str) -> Option<NamespaceId>;

    /// Get namespace name by ID
    fn name(&self, id: NamespaceId) -> Option<String>;

    /// Get all registered namespaces
    fn all(&self) -> Vec<(String, NamespaceId)>;

    /// Number of registered namespaces
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if a namespace ID is in the reserved range (0-99)
    fn is_reserved(&self, id: NamespaceId) -> bool {
        id <= RESERVED_NAMESPACE_MAX
    }

    /// Check if a namespace ID is dynamically allocated (100+)
    fn is_dynamic(&self, id: NamespaceId) -> bool {
        id >= DYNAMIC_NAMESPACE_START
    }
}

/// Local in-memory namespace registry
///
/// Default implementation for single-process use and testing.
/// For distributed systems, use a CRDT-backed implementation.
#[derive(Debug)]
pub struct LocalNamespaceRegistry {
    /// Name to ID mapping
    namespaces: HashMap<String, NamespaceId>,

    /// ID to name mapping (for reverse lookup)
    names: HashMap<NamespaceId, String>,

    /// Next available ID for dynamic allocation
    next_id: NamespaceId,
}

impl LocalNamespaceRegistry {
    /// Create a new registry with well-known namespaces pre-registered
    pub fn new() -> Self {
        let mut namespaces = HashMap::new();
        let mut names = HashMap::new();

        // Register well-known namespaces
        namespaces.insert("trading".to_string(), TRADING);
        namespaces.insert("pipeline".to_string(), PIPELINE);
        namespaces.insert("chat".to_string(), CHAT);

        names.insert(TRADING, "trading".to_string());
        names.insert(PIPELINE, "pipeline".to_string());
        names.insert(CHAT, "chat".to_string());

        Self {
            namespaces,
            names,
            next_id: DYNAMIC_NAMESPACE_START,
        }
    }

    /// Get the next ID that would be allocated
    pub fn peek_next_id(&self) -> NamespaceId {
        self.next_id
    }
}

impl Default for LocalNamespaceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl NamespaceRegistryProvider for LocalNamespaceRegistry {
    fn register(&mut self, name: &str) -> NamespaceRegistration {
        if let Some(&id) = self.namespaces.get(name) {
            return NamespaceRegistration {
                id,
                newly_registered: false,
            };
        }

        let id = self.next_id;
        self.next_id += 1;
        self.namespaces.insert(name.to_string(), id);
        self.names.insert(id, name.to_string());

        NamespaceRegistration {
            id,
            newly_registered: true,
        }
    }

    fn get(&self, name: &str) -> Option<NamespaceId> {
        self.namespaces.get(name).copied()
    }

    fn name(&self, id: NamespaceId) -> Option<String> {
        self.names.get(&id).cloned()
    }

    fn all(&self) -> Vec<(String, NamespaceId)> {
        self.namespaces
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    fn len(&self) -> usize {
        self.namespaces.len()
    }
}

/// Type alias for backward compatibility
///
/// Use `LocalNamespaceRegistry` directly for new code.
pub type NamespaceRegistry = LocalNamespaceRegistry;

/// A tensor tagged with its namespace
///
/// Provides gradient isolation - operations between tensors in different
/// namespaces require explicit federation.
#[derive(Debug, Clone)]
pub struct NamespacedTensor {
    /// The underlying tensor
    tensor: Tensor,

    /// Namespace this tensor belongs to
    namespace: NamespaceId,
}

impl NamespacedTensor {
    /// Create a new namespaced tensor
    pub fn new(tensor: Tensor, namespace: NamespaceId) -> Self {
        Self { tensor, namespace }
    }

    /// Get the underlying tensor
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Get the namespace ID
    pub fn namespace(&self) -> NamespaceId {
        self.namespace
    }

    /// Consume and return the underlying tensor
    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }

    /// Apply a unary operation (stays in same namespace)
    pub fn map<F>(&self, op: F) -> crate::Result<Self>
    where
        F: FnOnce(&Tensor) -> candle_core::Result<Tensor>,
    {
        let result = op(&self.tensor)
            .map_err(|e| crate::TensorCoreError::Tensor(format!("map failed: {}", e)))?;
        Ok(Self::new(result, self.namespace))
    }

    /// Combine with another tensor in the SAME namespace
    ///
    /// Returns error if namespaces don't match (use federation for cross-namespace ops)
    pub fn combine<F>(&self, other: &Self, op: F) -> crate::Result<Self>
    where
        F: FnOnce(&Tensor, &Tensor) -> candle_core::Result<Tensor>,
    {
        if self.namespace != other.namespace {
            return Err(crate::TensorCoreError::Namespace(format!(
                "Cannot combine tensors from different namespaces ({} vs {}). Use federation for cross-namespace operations.",
                self.namespace, other.namespace
            )));
        }

        let result = op(&self.tensor, &other.tensor)
            .map_err(|e| crate::TensorCoreError::Tensor(format!("combine failed: {}", e)))?;
        Ok(Self::new(result, self.namespace))
    }

    /// Check if this tensor can be combined with another (same namespace)
    pub fn can_combine(&self, other: &Self) -> bool {
        self.namespace == other.namespace
    }
}

/// Thread-safe shared namespace registry (local implementation)
pub type SharedNamespaceRegistry = Arc<RwLock<LocalNamespaceRegistry>>;

/// Thread-safe shared namespace registry (trait object for any implementation)
pub type SharedDynNamespaceRegistry = Arc<RwLock<Box<dyn NamespaceRegistryProvider>>>;

/// Create a new shared namespace registry (local implementation)
pub fn shared_registry() -> SharedNamespaceRegistry {
    Arc::new(RwLock::new(LocalNamespaceRegistry::new()))
}

/// Create a shared registry from any implementation
pub fn shared_registry_from<T: NamespaceRegistryProvider + 'static>(
    registry: T,
) -> SharedDynNamespaceRegistry {
    Arc::new(RwLock::new(Box::new(registry)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_well_known_namespaces() {
        let registry = NamespaceRegistry::new();

        assert_eq!(registry.get("trading"), Some(TRADING));
        assert_eq!(registry.get("pipeline"), Some(PIPELINE));
        assert_eq!(registry.get("chat"), Some(CHAT));
    }

    #[test]
    fn test_namespace_to_string() {
        assert_eq!(namespace_to_string(TRADING), "TRADING");
        assert_eq!(namespace_to_string(PIPELINE), "PIPELINE");
        assert_eq!(namespace_to_string(CHAT), "CHAT");
        assert_eq!(namespace_to_string(999), "UNKNOWN");
    }

    #[test]
    fn test_namespace_from_string() {
        assert_eq!(namespace_from_string("TRADING"), Some(TRADING));
        assert_eq!(namespace_from_string("PIPELINE"), Some(PIPELINE));
        assert_eq!(namespace_from_string("CHAT"), Some(CHAT));
        assert_eq!(namespace_from_string("trading"), None); // Case-sensitive
        assert_eq!(namespace_from_string("invalid"), None);
    }

    #[test]
    fn test_all_namespaces() {
        let all = all_namespaces();
        assert_eq!(all.len(), 3);

        // Verify roundtrip for all well-known namespaces
        for (id, name) in all {
            assert_eq!(namespace_to_string(*id), *name);
            assert_eq!(namespace_from_string(name), Some(*id));
        }
    }

    #[test]
    fn test_namespace_string_roundtrip() {
        // All well-known namespaces should roundtrip
        for ns in [TRADING, PIPELINE, CHAT] {
            let s = namespace_to_string(ns);
            let parsed = namespace_from_string(s);
            assert_eq!(parsed, Some(ns));
        }
    }

    #[test]
    fn test_register_custom_namespace() {
        let mut registry = NamespaceRegistry::new();

        let result = registry.register("custom_engine");
        assert!(result.id >= DYNAMIC_NAMESPACE_START); // Dynamic allocation starts at 100
        assert!(result.newly_registered);

        // Re-registering returns same ID, not newly registered
        let result2 = registry.register("custom_engine");
        assert_eq!(result.id, result2.id);
        assert!(!result2.newly_registered);
    }

    #[test]
    fn test_namespace_name_lookup() {
        let registry = NamespaceRegistry::new();

        assert_eq!(registry.name(TRADING), Some("trading".to_string()));
        assert_eq!(registry.name(PIPELINE), Some("pipeline".to_string()));
        assert_eq!(registry.name(CHAT), Some("chat".to_string()));
        assert_eq!(registry.name(999), None);
    }

    #[test]
    fn test_namespaced_tensor() {
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &device).unwrap();

        let nt = NamespacedTensor::new(t, TRADING);
        assert_eq!(nt.namespace(), TRADING);
        assert_eq!(nt.tensor().dims(), &[3]);
    }

    #[test]
    fn test_namespaced_tensor_map() {
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &device).unwrap();
        let nt = NamespacedTensor::new(t, PIPELINE);

        let squared = nt.map(|t| t.sqr()).unwrap();
        assert_eq!(squared.namespace(), PIPELINE); // Same namespace

        let vals = squared.tensor().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 1.0).abs() < 0.001);
        assert!((vals[1] - 4.0).abs() < 0.001);
        assert!((vals[2] - 9.0).abs() < 0.001);
    }

    #[test]
    fn test_namespaced_tensor_combine_same_ns() {
        let device = Device::Cpu;
        let a = Tensor::from_vec(vec![1.0f32, 2.0], 2, &device).unwrap();
        let b = Tensor::from_vec(vec![3.0f32, 4.0], 2, &device).unwrap();

        let na = NamespacedTensor::new(a, TRADING);
        let nb = NamespacedTensor::new(b, TRADING);

        let sum = na.combine(&nb, |x, y| x + y).unwrap();
        assert_eq!(sum.namespace(), TRADING);

        let vals = sum.tensor().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 4.0).abs() < 0.001);
        assert!((vals[1] - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_namespaced_tensor_combine_different_ns() {
        let device = Device::Cpu;
        let a = Tensor::from_vec(vec![1.0f32], 1, &device).unwrap();
        let b = Tensor::from_vec(vec![2.0f32], 1, &device).unwrap();

        let na = NamespacedTensor::new(a, TRADING);
        let nb = NamespacedTensor::new(b, PIPELINE); // Different namespace!

        let result = na.combine(&nb, |x, y| x + y);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("federation"));
    }

    #[test]
    fn test_shared_registry() {
        let registry = shared_registry();

        {
            let mut w = registry.write();
            w.register("test_ns");
        }

        {
            let r = registry.read();
            assert!(r.get("test_ns").is_some());
        }
    }

    // =========================================================================
    // NamespacePath tests
    // =========================================================================

    #[test]
    fn test_namespace_path_basic() {
        let path = NamespacePath::new("an-ecosystem.devops.pipeline").unwrap();
        assert_eq!(path.as_str(), "an-ecosystem.devops.pipeline");
        assert_eq!(path.depth(), 3);
        assert!(!path.exceeds_recommended_depth());
    }

    #[test]
    fn test_namespace_path_segments() {
        let path = NamespacePath::new("an-ecosystem.devops.pipeline.metrics").unwrap();
        assert_eq!(path.owner(), "an-ecosystem");
        assert_eq!(path.domain(), Some("devops"));
        assert_eq!(path.subdomain(), Some("pipeline"));
        assert_eq!(path.data_type(), Some("metrics"));
    }

    #[test]
    fn test_namespace_path_owner_type() {
        let foundation = NamespacePath::new("an-ecosystem.devops").unwrap();
        assert_eq!(foundation.owner_type(), OwnerType::Foundation);

        let org = NamespacePath::new("org-lttr.trading").unwrap();
        assert_eq!(org.owner_type(), OwnerType::Organization);

        let custom = NamespacePath::new("custom.domain").unwrap();
        assert_eq!(custom.owner_type(), OwnerType::Custom);
    }

    #[test]
    fn test_namespace_path_root_namespace() {
        let pipeline = NamespacePath::new("an-ecosystem.devops").unwrap();
        assert_eq!(pipeline.root_namespace(), Some(PIPELINE));

        let trading = NamespacePath::new("org-lttr.trading.regime").unwrap();
        assert_eq!(trading.root_namespace(), Some(TRADING));

        let chat = NamespacePath::new("an-agent.sessions").unwrap();
        assert_eq!(chat.root_namespace(), Some(CHAT));

        let unknown = NamespacePath::new("unknown.domain").unwrap();
        assert_eq!(unknown.root_namespace(), None);
    }

    #[test]
    fn test_namespace_path_child() {
        let parent = NamespacePath::new("an-ecosystem.devops").unwrap();
        let child = parent.child("pipeline").unwrap();
        assert_eq!(child.as_str(), "an-ecosystem.devops.pipeline");
    }

    #[test]
    fn test_namespace_path_is_child_of() {
        let parent = NamespacePath::new("an-ecosystem.devops").unwrap();
        let child = NamespacePath::new("an-ecosystem.devops.pipeline").unwrap();
        let grandchild = NamespacePath::new("an-ecosystem.devops.pipeline.metrics").unwrap();
        let unrelated = NamespacePath::new("org-lttr.trading").unwrap();

        assert!(child.is_child_of(&parent));
        assert!(grandchild.is_child_of(&parent));
        assert!(grandchild.is_child_of(&child));
        assert!(!parent.is_child_of(&child)); // Parent is not child of child
        assert!(!unrelated.is_child_of(&parent)); // Different owner
    }

    #[test]
    fn test_namespace_path_to_crdt_key() {
        let path = NamespacePath::new("an-ecosystem.devops.pipeline.metrics").unwrap();
        let key = path.to_crdt_key("latest");
        assert_eq!(key, "tensor/an-ecosystem/devops/pipeline/metrics/latest");
    }

    #[test]
    fn test_namespace_path_from_crdt_key() {
        let (path, suffix) =
            NamespacePath::from_crdt_key("tensor/an-ecosystem/devops/pipeline/latest").unwrap();
        assert_eq!(path.as_str(), "an-ecosystem.devops.pipeline");
        assert_eq!(suffix, "latest");
    }

    #[test]
    fn test_namespace_path_crdt_roundtrip() {
        let original = NamespacePath::new("an-ecosystem.devops.pipeline.metrics").unwrap();
        let key = original.to_crdt_key("epoch_5");
        let (parsed, suffix) = NamespacePath::from_crdt_key(&key).unwrap();
        assert_eq!(parsed.as_str(), original.as_str());
        assert_eq!(suffix, "epoch_5");
    }

    #[test]
    fn test_namespace_path_from_segments() {
        let path =
            NamespacePath::from_segments(&["an-ecosystem", "devops", "pipeline", "metrics"])
                .unwrap();
        assert_eq!(path.as_str(), "an-ecosystem.devops.pipeline.metrics");
    }

    #[test]
    fn test_namespace_path_validation_empty() {
        assert!(NamespacePath::new("").is_err());
        assert!(NamespacePath::new(".").is_err());
        assert!(NamespacePath::new("an-ecosystem..pipeline").is_err());
    }

    #[test]
    fn test_namespace_path_validation_invalid_chars() {
        assert!(NamespacePath::new("an-ecosystem/devops").is_err()); // slash
        assert!(NamespacePath::new("an ecosystem.devops").is_err()); // space
        assert!(NamespacePath::new("an-ecosystem.devops!").is_err()); // special
    }

    #[test]
    fn test_namespace_path_depth_warning() {
        let deep = NamespacePath::new("a.b.c.d.e.f").unwrap(); // 6 levels
        assert!(deep.exceeds_recommended_depth());

        let ok = NamespacePath::new("a.b.c.d").unwrap(); // 4 levels
        assert!(!ok.exceeds_recommended_depth());
    }

    #[test]
    fn test_namespace_path_serde() {
        let path = NamespacePath::new("an-ecosystem.devops.pipeline").unwrap();
        let json = serde_json::to_string(&path).unwrap();
        let parsed: NamespacePath = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.as_str(), path.as_str());
    }

    #[test]
    fn test_namespace_path_display() {
        let path = NamespacePath::new("an-ecosystem.devops.pipeline").unwrap();
        assert_eq!(format!("{}", path), "an-ecosystem.devops.pipeline");
    }

    #[test]
    fn test_namespace_path_from_str() {
        use std::str::FromStr;
        let path = NamespacePath::from_str("an-ecosystem.devops").unwrap();
        assert_eq!(path.owner(), "an-ecosystem");
    }

    // =========================================================================
    // Version tests
    // =========================================================================

    #[test]
    fn test_namespace_path_version_detection() {
        let v2 = NamespacePath::new("an-ecosystem.tensor.regime.v2").unwrap();
        assert_eq!(v2.version(), Some("v2"));

        let v2_1 = NamespacePath::new("an-ecosystem.tensor.regime.v2_1").unwrap();
        assert_eq!(v2_1.version(), Some("v2_1"));

        let no_version = NamespacePath::new("an-ecosystem.tensor.regime.states").unwrap();
        assert_eq!(no_version.version(), None);
    }

    #[test]
    fn test_namespace_path_version_parsing() {
        let v2 = NamespacePath::new("an-ecosystem.tensor.regime.v2").unwrap();
        assert_eq!(v2.parse_version(), Some((2, None, None)));

        let v2_1 = NamespacePath::new("an-ecosystem.tensor.regime.v2_1").unwrap();
        assert_eq!(v2_1.parse_version(), Some((2, Some(1), None)));

        let v2_1_3 = NamespacePath::new("an-ecosystem.tensor.regime.v2_1_3").unwrap();
        assert_eq!(v2_1_3.parse_version(), Some((2, Some(1), Some(3))));
    }

    #[test]
    fn test_namespace_path_with_version() {
        let base = NamespacePath::new("an-ecosystem.tensor.regime").unwrap();

        let v2 = base.with_version(2, None, None).unwrap();
        assert_eq!(v2.as_str(), "an-ecosystem.tensor.regime.v2");

        let v2_1 = base.with_version(2, Some(1), None).unwrap();
        assert_eq!(v2_1.as_str(), "an-ecosystem.tensor.regime.v2_1");
    }

    #[test]
    fn test_namespace_path_without_version() {
        let versioned = NamespacePath::new("an-ecosystem.tensor.regime.v2").unwrap();
        let base = versioned.without_version().unwrap();
        assert_eq!(base.as_str(), "an-ecosystem.tensor.regime");

        // Non-versioned path returns None
        let no_ver = NamespacePath::new("an-ecosystem.tensor.regime.states").unwrap();
        assert!(no_ver.without_version().is_none());
    }

    // =========================================================================
    // NamespaceMeta tests
    // =========================================================================

    #[test]
    fn test_namespace_meta_builder() {
        let path = NamespacePath::new("an-ecosystem.devops.pipeline.metrics").unwrap();
        let meta = NamespaceMeta::new(&path)
            .with_description("Pipeline build metrics")
            .with_owner_contact("infra@example.com")
            .with_schema_fingerprint("v2")
            .with_tensor_shape(&[1000, 64])
            .with_tensor_dtype("F32")
            .with_recompute_frequency("per-epoch")
            .with_coverage("all pipelines")
            .with_upstream("an-ecosystem.devops.pipeline.configs")
            .with_custom("retention_days", "30");

        assert_eq!(meta.description, Some("Pipeline build metrics".to_string()));
        assert_eq!(meta.owner_contact, Some("infra@example.com".to_string()));
        assert_eq!(meta.schema_fingerprint, Some("v2".to_string()));
        assert_eq!(meta.tensor_shape, Some(vec![1000, 64]));
        assert_eq!(meta.tensor_dtype, Some("F32".to_string()));
        assert_eq!(meta.upstream.len(), 1);
        assert_eq!(meta.custom.get("retention_days"), Some(&"30".to_string()));
    }

    #[test]
    fn test_namespace_meta_crdt_key() {
        let path = NamespacePath::new("an-ecosystem.devops.pipeline").unwrap();
        let meta = NamespaceMeta::new(&path);
        assert_eq!(meta.to_crdt_key(), "tensor/an-ecosystem/devops/pipeline/meta");
    }

    #[test]
    fn test_namespace_meta_compatibility() {
        let path = NamespacePath::new("an-ecosystem.tensor.regime").unwrap();

        let v1 = NamespaceMeta::new(&path).with_schema_fingerprint("v1");
        let v1_also = NamespaceMeta::new(&path).with_schema_fingerprint("v1_1");
        let v2 = NamespaceMeta::new(&path).with_schema_fingerprint("v2");

        // Same major version = compatible
        assert!(v1.is_compatible_with(&v1_also));

        // Different major version = incompatible
        assert!(!v1.is_compatible_with(&v2));

        // Missing fingerprint = assume compatible
        let no_schema = NamespaceMeta::new(&path);
        assert!(v1.is_compatible_with(&no_schema));
    }

    #[test]
    fn test_namespace_meta_serde() {
        let path = NamespacePath::new("an-ecosystem.devops.pipeline").unwrap();
        let meta = NamespaceMeta::new(&path)
            .with_description("Test")
            .with_schema_fingerprint("v1");

        let json = serde_json::to_string(&meta).unwrap();
        let parsed: NamespaceMeta = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.description, meta.description);
        assert_eq!(parsed.schema_fingerprint, meta.schema_fingerprint);
    }

    #[test]
    fn test_reserved_namespace_constants() {
        assert!(TRADING < RESERVED_NAMESPACE_MAX);
        assert!(PIPELINE < RESERVED_NAMESPACE_MAX);
        assert!(CHAT < RESERVED_NAMESPACE_MAX);
        assert_eq!(DYNAMIC_NAMESPACE_START, 100);
    }

    #[test]
    fn test_namespace_registry_provider_trait() {
        // Test that LocalNamespaceRegistry implements the trait
        let mut registry: Box<dyn NamespaceRegistryProvider> =
            Box::new(LocalNamespaceRegistry::new());

        // Well-known namespaces work
        assert_eq!(registry.get("trading"), Some(TRADING));
        assert!(registry.is_reserved(TRADING));

        // Dynamic registration works
        let result = registry.register("org-myland");
        assert!(result.newly_registered);
        assert!(registry.is_dynamic(result.id));
        assert_eq!(result.id, DYNAMIC_NAMESPACE_START);

        // Lookup works
        assert_eq!(registry.get("org-myland"), Some(result.id));
        assert_eq!(registry.name(result.id), Some("org-myland".to_string()));
    }

    #[test]
    fn test_namespace_registration_result() {
        let mut registry = LocalNamespaceRegistry::new();

        // First registration is new
        let first = registry.register("new-team");
        assert!(first.newly_registered);
        assert_eq!(first.id, 100);

        // Second registration is not new
        let second = registry.register("new-team");
        assert!(!second.newly_registered);
        assert_eq!(second.id, first.id);

        // Third team gets next ID
        let third = registry.register("another-team");
        assert!(third.newly_registered);
        assert_eq!(third.id, 101);
    }

    #[test]
    fn test_shared_dyn_registry() {
        let registry = shared_registry_from(LocalNamespaceRegistry::new());

        {
            let mut w = registry.write();
            let result = w.register("test-dyn");
            assert!(result.newly_registered);
        }

        {
            let r = registry.read();
            assert!(r.get("test-dyn").is_some());
        }
    }
}




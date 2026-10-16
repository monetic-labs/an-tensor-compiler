//! Core types for holographic context representation
//!
//! Defines the tensor types for components, modules, bounded contexts,
//! and organisms in the hierarchical code representation.

use crate::holographic::bridges::{BridgeBinding, BridgeBindingType, BridgeProjectionPolicy};
use crate::holographic::knowledge::{
    ItemProvenance, KnowledgeItemRef, KnowledgeItemType, KnowledgeSpaceKind, KnowledgeSpaceTensor,
    KnowledgeSystemView, SalienceDefaults, SpaceBoundaryPolicy, SpaceProvenance,
    SpaceTemporalPolicy,
};
use candle_core::Tensor;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;

// ============================================================================
// Component Level (Atomic)
// ============================================================================

/// Kind of source code component
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentKind {
    /// struct definition
    Struct,
    /// enum definition
    Enum,
    /// trait definition
    Trait,
    /// impl block
    Impl,
    /// function or method
    Function,
    /// const item
    Const,
    /// static item
    Static,
    /// macro definition
    Macro,
    /// module declaration
    Module,
    /// type alias
    TypeAlias,
}

impl ComponentKind {
    /// Weight for composition (more important = higher weight)
    pub fn importance_weight(&self) -> f32 {
        match self {
            Self::Trait => 1.0,     // Traits define interfaces
            Self::Struct => 0.9,    // Core data structures
            Self::Enum => 0.9,      // Core types
            Self::Function => 0.7,  // Behavior
            Self::Impl => 0.6,      // Implementation details
            Self::Macro => 0.5,     // Metaprogramming
            Self::Const => 0.3,     // Constants
            Self::Static => 0.3,    // Statics
            Self::TypeAlias => 0.4, // Type aliases
            Self::Module => 0.2,    // Module structure
        }
    }
}

/// Unique identifier for a code component
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComponentId {
    /// Module path (e.g., "synapse::data_sync")
    pub module: String,
    /// Component name (e.g., "SyncDecision")  
    pub name: String,
    /// Kind of component
    pub kind: ComponentKind,
}

impl ComponentId {
    /// Identifier from module path, component name, and kind.
    pub fn new(module: impl Into<String>, name: impl Into<String>, kind: ComponentKind) -> Self {
        Self {
            module: module.into(),
            name: name.into(),
            kind,
        }
    }

    /// Full qualified path
    pub fn full_path(&self) -> String {
        format!("{}::{}", self.module, self.name)
    }
}

impl std::fmt::Display for ComponentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.full_path())
    }
}

/// Tensor representation of a code component
#[derive(Debug, Clone)]
pub struct ComponentTensor {
    /// Unique identifier
    pub id: ComponentId,

    /// Source file location
    pub file: PathBuf,
    /// First source line (1-based).
    pub line_start: usize,
    /// Last source line (inclusive).
    pub line_end: usize,

    /// Semantic embedding (what it does) \[dim\]
    pub semantic: Tensor,

    /// Structural embedding (AST shape) \[dim\]
    pub structural: Tensor,

    /// Signature embedding (types, parameters) \[dim\]
    pub signature: Tensor,

    /// Documentation embedding \[dim\]
    pub documentation: Option<Tensor>,

    /// Combined holographic representation \[dim\]
    pub hologram: Tensor,

    /// Visibility
    pub is_public: bool,
}

// ============================================================================
// Module Level
// ============================================================================

/// Tensor representation of a module (file)
#[derive(Debug, Clone)]
pub struct ModuleTensor {
    /// Module path (e.g., "synapse::data_sync")
    pub path: String,

    /// File path
    pub file: PathBuf,

    /// Components in this module
    pub components: Vec<ComponentTensor>,

    /// Composed hologram \[dim\]
    pub hologram: Tensor,

    /// Public exports
    pub exports: Vec<ComponentId>,

    /// Imports from other modules
    pub imports: Vec<String>,
}

impl ModuleTensor {
    /// Get a component by name
    pub fn get_component(&self, name: &str) -> Option<&ComponentTensor> {
        self.components.iter().find(|c| c.id.name == name)
    }

    /// Get public components only
    pub fn public_components(&self) -> impl Iterator<Item = &ComponentTensor> {
        self.components.iter().filter(|c| c.is_public)
    }
}

// ============================================================================
// Bounded Context Level
// ============================================================================

/// Role a context plays in an organism
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextRole {
    /// Core functionality - central to the organism
    Core,
    /// Supporting infrastructure
    Support,
    /// Utility code
    Utility,
    /// External integrations
    Integration,
}

impl ContextRole {
    /// Weight for composition
    pub fn weight(&self) -> f32 {
        match self {
            Self::Core => 1.0,
            Self::Support => 0.7,
            Self::Integration => 0.5,
            Self::Utility => 0.4,
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

impl FromStr for ContextRole {
    type Err = ();

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("core") {
            Ok(Self::Core)
        } else if s.eq_ignore_ascii_case("support") {
            Ok(Self::Support)
        } else if s.eq_ignore_ascii_case("utility") {
            Ok(Self::Utility)
        } else if s.eq_ignore_ascii_case("integration") {
            Ok(Self::Integration)
        } else {
            Err(())
        }
    }
}

/// Tensor representation of a bounded context
#[derive(Debug, Clone)]
pub struct BoundedContextTensor {
    /// Context name
    pub name: String,

    /// Root path
    pub path: PathBuf,

    /// Description
    pub description: String,

    /// Role in organism
    pub role: ContextRole,

    /// Modules in this context
    pub modules: Vec<ModuleTensor>,

    /// Composed hologram \[dim\]
    pub hologram: Tensor,

    /// Public API exports
    pub api: Vec<ComponentId>,

    /// Detected patterns (e.g., "async-trait", "error-handling")
    pub patterns: Vec<String>,

    /// Internal module dependencies
    pub dependencies: Vec<(String, String)>,
}

impl BoundedContextTensor {
    /// Get a module by path
    pub fn get_module(&self, path: &str) -> Option<&ModuleTensor> {
        self.modules.iter().find(|m| m.path == path)
    }

    /// Find components matching a pattern
    pub fn find_components(&self, pattern: &str) -> Vec<&ComponentTensor> {
        self.modules
            .iter()
            .flat_map(|m| m.components.iter())
            .filter(|c| c.id.name.contains(pattern) || c.id.module.contains(pattern))
            .collect()
    }
}

impl From<&BoundedContextTensor> for KnowledgeSpaceTensor {
    fn from(value: &BoundedContextTensor) -> Self {
        let mut items = Vec::new();

        for module in &value.modules {
            items.push(KnowledgeItemRef {
                id: format!("module::{}", module.path),
                item_type: KnowledgeItemType::Custom("module".to_string()),
                hologram: module.hologram.clone(),
                description: Some(module.path.clone()),
                provenance: ItemProvenance {
                    source_id: value.name.clone(),
                    lineage: vec![module.file.display().to_string()],
                    ..ItemProvenance::default()
                },
            });

            for component in &module.components {
                items.push(KnowledgeItemRef {
                    id: component.id.full_path(),
                    item_type: KnowledgeItemType::Component,
                    hologram: component.hologram.clone(),
                    description: Some(component.id.full_path()),
                    provenance: ItemProvenance {
                        source_id: value.name.clone(),
                        lineage: vec![
                            module.path.clone(),
                            component.file.display().to_string(),
                            format!("{}:{}", component.line_start, component.line_end),
                        ],
                        ..ItemProvenance::default()
                    },
                });
            }
        }

        let mut metadata = HashMap::new();
        metadata.insert(
            "context_role".to_string(),
            context_role_name(value.role).to_string(),
        );
        metadata.insert("path".to_string(), value.path.display().to_string());
        metadata.insert("module_count".to_string(), value.modules.len().to_string());
        metadata.insert("api_export_count".to_string(), value.api.len().to_string());
        metadata.insert(
            "dependency_edge_count".to_string(),
            value.dependencies.len().to_string(),
        );

        if !value.patterns.is_empty() {
            metadata.insert("patterns".to_string(), value.patterns.join(","));
        }

        Self {
            id: value.name.clone(),
            name: value.name.clone(),
            description: value.description.clone(),
            kind: KnowledgeSpaceKind::Code,
            hologram: value.hologram.clone(),
            items,
            boundaries: SpaceBoundaryPolicy::default(),
            provenance: SpaceProvenance {
                sources: vec![value.path.display().to_string()],
                ..SpaceProvenance::default()
            },
            temporal: SpaceTemporalPolicy::default(),
            salience: SalienceDefaults::default(),
            geometric_index: None,
            metadata,
        }
    }
}

// ============================================================================
// Organism Level
// ============================================================================

/// Rules for bounded context dependencies
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BoundaryRules {
    /// Map of context -> allowed dependencies
    #[serde(flatten)]
    pub allowed: HashMap<String, Vec<String>>,
}

impl BoundaryRules {
    /// Check if a dependency is allowed
    pub fn can_depend(&self, from: &str, to: &str) -> bool {
        self.allowed
            .get(from)
            .map(|deps| deps.contains(&to.to_string()))
            .unwrap_or(false)
    }

    /// Add a dependency rule
    pub fn allow(&mut self, from: impl Into<String>, to: impl Into<String>) {
        self.allowed.entry(from.into()).or_default().push(to.into());
    }
}

/// Federation binding to external organism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationBinding {
    /// External organism ID
    pub organism: String,
    /// Imported items
    pub imports: Vec<String>,
}

/// Tensor representation of commit history
#[derive(Debug, Clone)]
pub struct CommitTensor {
    /// Commit hash.
    pub hash: String,
    /// Commit message.
    pub message: String,
    /// Commit timestamp.
    pub timestamp: DateTime<Utc>,
    /// Tensor encoding of the diff
    pub diff_tensor: Tensor,
    /// Category (bug-fix, feature, refactor)
    pub category: Option<String>,
}

/// History tensor for tracking changes and patterns
#[derive(Debug, Clone)]
pub struct HistoryTensor {
    /// Recent commits with tensors
    pub commits: Vec<CommitTensor>,
    /// Aggregated bug fix patterns
    pub bug_fix_pattern: Option<Tensor>,
    /// Aggregated feature patterns  
    pub feature_pattern: Option<Tensor>,
    /// Aggregated refactoring patterns
    pub refactor_pattern: Option<Tensor>,
}

impl HistoryTensor {
    /// Find commits similar to a query
    pub fn find_similar(&self, query: &Tensor, top_k: usize) -> Vec<(&CommitTensor, f32)> {
        use super::ops::cosine_similarity;

        let mut scored: Vec<_> = self
            .commits
            .iter()
            .filter_map(|c| {
                cosine_similarity(query, &c.diff_tensor)
                    .ok()
                    .map(|score| (c, score))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }
}

/// Tensor representation of an organism (project)
#[derive(Debug, Clone)]
pub struct OrganismTensor {
    /// Project identifier
    pub project_id: String,

    /// Human-readable name
    pub name: String,

    /// Description
    pub description: String,

    /// Bounded contexts
    pub contexts: Vec<BoundedContextTensor>,

    /// Composed hologram \[dim\]
    pub hologram: Tensor,

    /// Boundary rules
    pub boundaries: BoundaryRules,

    /// Federation (dependencies on other organisms)
    pub federation: Vec<FederationBinding>,

    /// History tensor
    pub history: Option<HistoryTensor>,

    /// Last update timestamp.
    pub updated_at: DateTime<Utc>,
    /// Head commit hash at build time.
    pub commit: String,
    /// Embedding dimension of the holograms.
    pub embedding_dim: usize,
}

impl OrganismTensor {
    /// Get a bounded context by name
    pub fn get_context(&self, name: &str) -> Option<&BoundedContextTensor> {
        self.contexts.iter().find(|c| c.name == name)
    }

    /// Adapt this organism into neutral knowledge spaces.
    pub fn as_knowledge_spaces(&self) -> Vec<KnowledgeSpaceTensor> {
        self.contexts
            .iter()
            .map(|context| {
                let mut space = KnowledgeSpaceTensor::from(context);
                space.boundaries = SpaceBoundaryPolicy {
                    allow_unbridged_projection: false,
                    allowed_inbound_spaces: self
                        .boundaries
                        .allowed
                        .iter()
                        .filter_map(|(from, deps)| {
                            if deps.iter().any(|dep| dep == &context.name) {
                                Some(from.clone())
                            } else {
                                None
                            }
                        })
                        .collect(),
                    allowed_outbound_spaces: self
                        .boundaries
                        .allowed
                        .get(&context.name)
                        .cloned()
                        .unwrap_or_default(),
                    allowed_bridge_ids: self
                        .federation
                        .iter()
                        .map(|binding| format!("federation::{}", binding.organism))
                        .collect(),
                };
                space.provenance.updated_at = Some(self.updated_at);
                space
            })
            .collect()
    }

    /// Adapt this organism into a neutral multi-space system view.
    pub fn as_knowledge_system(&self) -> KnowledgeSystemView {
        let mut metadata = HashMap::new();
        metadata.insert("commit".to_string(), self.commit.clone());
        metadata.insert("embedding_dim".to_string(), self.embedding_dim.to_string());
        metadata.insert("context_count".to_string(), self.contexts.len().to_string());
        metadata.insert(
            "federation_binding_count".to_string(),
            self.federation.len().to_string(),
        );
        metadata.insert(
            "history_enabled".to_string(),
            self.history.is_some().to_string(),
        );

        KnowledgeSystemView {
            id: self.project_id.clone(),
            name: self.name.clone(),
            description: self.description.clone(),
            hologram: self.hologram.clone(),
            spaces: self.as_knowledge_spaces(),
            bridges: self
                .federation
                .iter()
                .map(|binding| BridgeBinding {
                    id: format!("federation::{}", binding.organism),
                    from_space: self.project_id.clone(),
                    to_space: binding.organism.clone(),
                    binding_type: BridgeBindingType::DerivedFrom,
                    policy: BridgeProjectionPolicy::default(),
                    provenance: ItemProvenance {
                        source_id: self.project_id.clone(),
                        lineage: binding.imports.clone(),
                        ..ItemProvenance::default()
                    },
                    active: true,
                })
                .collect(),
            metadata,
        }
    }

    /// Project a query to find relevant context
    pub fn project(&self, query_tensor: &Tensor) -> crate::Result<ProjectionResult> {
        use super::ops::{cosine_similarity, project as tensor_project};

        // Project against organism hologram
        let projection = tensor_project(&self.hologram, query_tensor)?;

        // Score each context
        let mut context_scores: Vec<(String, f32)> = self
            .contexts
            .iter()
            .filter_map(|ctx| {
                tensor_project(&ctx.hologram, query_tensor)
                    .ok()
                    .and_then(|ctx_proj| cosine_similarity(&projection, &ctx_proj).ok())
                    .map(|score| (ctx.name.clone(), score))
            })
            .filter(|(_, score)| *score > 0.1)
            .collect();

        context_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Score components in relevant contexts
        let mut component_scores: Vec<(ComponentId, f32)> = Vec::new();

        for (ctx_name, _) in context_scores.iter().take(3) {
            if let Some(ctx) = self.get_context(ctx_name) {
                for module in &ctx.modules {
                    for comp in &module.components {
                        if let Ok(comp_proj) = tensor_project(&comp.hologram, query_tensor) {
                            if let Ok(score) = cosine_similarity(&projection, &comp_proj) {
                                if score > 0.2 {
                                    component_scores.push((comp.id.clone(), score));
                                }
                            }
                        }
                    }
                }
            }
        }

        component_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        component_scores.truncate(20);

        Ok(ProjectionResult {
            query_tensor: query_tensor.clone(),
            contexts: context_scores,
            components: component_scores,
            projection,
        })
    }

    /// Check boundary violations in generated code
    pub fn check_boundaries(
        &self,
        from_context: &str,
        used_items: &[String],
    ) -> Vec<BoundaryViolation> {
        let mut violations = Vec::new();

        for item in used_items {
            // Find which context the item belongs to
            for ctx in &self.contexts {
                let item_in_ctx = ctx.modules.iter().any(|m| {
                    m.path.contains(item) || m.components.iter().any(|c| c.id.name == *item)
                });

                if item_in_ctx
                    && ctx.name != from_context
                    && !self.boundaries.can_depend(from_context, &ctx.name)
                {
                    violations.push(BoundaryViolation {
                        from_context: from_context.to_string(),
                        to_context: ctx.name.clone(),
                        component: item.clone(),
                        reason: format!(
                            "Context '{}' cannot depend on '{}' (not in boundary rules)",
                            from_context, ctx.name
                        ),
                    });
                }
            }
        }

        violations
    }
}

// ============================================================================
// Query Results
// ============================================================================

/// Result of projecting a query into context space
#[derive(Debug)]
pub struct ProjectionResult {
    /// Query embedding
    pub query_tensor: Tensor,

    /// Relevant contexts with scores
    pub contexts: Vec<(String, f32)>,

    /// Relevant components with scores
    pub components: Vec<(ComponentId, f32)>,

    /// The projected tensor
    pub projection: Tensor,
}

impl ProjectionResult {
    /// Get top N components
    pub fn top_components(&self, n: usize) -> &[(ComponentId, f32)] {
        &self.components[..n.min(self.components.len())]
    }

    /// Get primary context (highest scored)
    pub fn primary_context(&self) -> Option<&str> {
        self.contexts.first().map(|(name, _)| name.as_str())
    }
}

/// Result of coherence checking
#[derive(Debug)]
pub struct CoherenceResult {
    /// Overall coherence score (0-1)
    pub overall: f32,

    /// Pattern coherence (does it match existing patterns?)
    pub pattern_score: f32,

    /// Style coherence (does it match coding style?)
    pub style_score: f32,

    /// Boundary violations
    pub boundary_violations: Vec<BoundaryViolation>,
}

impl CoherenceResult {
    /// Is the result acceptable?
    pub fn is_acceptable(&self, min_coherence: f32) -> bool {
        self.overall >= min_coherence && self.boundary_violations.is_empty()
    }
}

/// A boundary violation in generated code
#[derive(Debug, Clone)]
pub struct BoundaryViolation {
    /// Context the access originates from.
    pub from_context: String,
    /// Context being accessed.
    pub to_context: String,
    /// Component performing the access.
    pub component: String,
    /// Why the access violates the boundary.
    pub reason: String,
}

impl std::fmt::Display for BoundaryViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -> {} via '{}': {}",
            self.from_context, self.to_context, self.component, self.reason
        )
    }
}

fn context_role_name(role: ContextRole) -> &'static str {
    match role {
        ContextRole::Core => "core",
        ContextRole::Support => "support",
        ContextRole::Utility => "utility",
        ContextRole::Integration => "integration",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn zeros(dim: usize) -> Tensor {
        Tensor::zeros((dim,), candle_core::DType::F32, &Device::Cpu).unwrap()
    }

    fn sample_component() -> ComponentTensor {
        ComponentTensor {
            id: ComponentId::new("crate::core", "Thing", ComponentKind::Struct),
            file: PathBuf::from("src/core.rs"),
            line_start: 1,
            line_end: 10,
            semantic: zeros(8),
            structural: zeros(8),
            signature: zeros(8),
            documentation: None,
            hologram: zeros(8),
            is_public: true,
        }
    }

    fn sample_context() -> BoundedContextTensor {
        let component = sample_component();
        let module = ModuleTensor {
            path: "crate::core".to_string(),
            file: PathBuf::from("src/core.rs"),
            components: vec![component.clone()],
            hologram: zeros(8),
            exports: vec![component.id.clone()],
            imports: vec!["crate::util".to_string()],
        };

        BoundedContextTensor {
            name: "core".to_string(),
            path: PathBuf::from("crates/core"),
            description: "Core context".to_string(),
            role: ContextRole::Core,
            modules: vec![module],
            hologram: zeros(8),
            api: vec![component.id],
            patterns: vec!["async".to_string()],
            dependencies: vec![("crate::core".to_string(), "crate::util".to_string())],
        }
    }

    #[test]
    fn test_bounded_context_adapts_to_knowledge_space() {
        let context = sample_context();
        let space = KnowledgeSpaceTensor::from(&context);

        assert_eq!(space.id, "core");
        assert_eq!(space.kind, KnowledgeSpaceKind::Code);
        assert_eq!(space.items.len(), 2);
        assert_eq!(space.metadata.get("context_role").unwrap(), "core");
        assert_eq!(space.metadata.get("module_count").unwrap(), "1");
    }

    #[test]
    fn test_organism_adapts_to_knowledge_system() {
        let context = sample_context();
        let mut boundaries = BoundaryRules::default();
        boundaries.allow("core", "util");
        boundaries.allow("util", "core");

        let organism = OrganismTensor {
            project_id: "proj".to_string(),
            name: "Project".to_string(),
            description: "Test project".to_string(),
            contexts: vec![context],
            hologram: zeros(8),
            boundaries,
            federation: vec![FederationBinding {
                organism: "ext-org".to_string(),
                imports: vec!["shared::Thing".to_string()],
            }],
            history: None,
            updated_at: Utc::now(),
            commit: "abc123".to_string(),
            embedding_dim: 8,
        };

        let system = organism.as_knowledge_system();

        assert_eq!(system.id, "proj");
        assert_eq!(system.spaces.len(), 1);
        assert_eq!(
            system.spaces[0].boundaries.allowed_outbound_spaces,
            vec!["util".to_string()]
        );
        assert_eq!(
            system.spaces[0].boundaries.allowed_bridge_ids,
            vec!["federation::ext-org".to_string()]
        );
        assert_eq!(system.bridges.len(), 1);
        assert_eq!(system.metadata.get("commit").unwrap(), "abc123");
    }
}

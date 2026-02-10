//! Core types for holographic context representation
//!
//! Defines the tensor types for components, modules, bounded contexts,
//! and organisms in the hierarchical code representation.

use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use chrono::{DateTime, Utc};

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
            Self::Trait => 1.0,      // Traits define interfaces
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
    /// Create a new component identifier
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
    /// Starting line number in source file
    pub line_start: usize,
    /// Ending line number in source file
    pub line_end: usize,
    
    /// Semantic embedding (what it does) [dim]
    pub semantic: Tensor,
    
    /// Structural embedding (AST shape) [dim]
    pub structural: Tensor,
    
    /// Signature embedding (types, parameters) [dim]
    pub signature: Tensor,
    
    /// Documentation embedding [dim]
    pub documentation: Option<Tensor>,
    
    /// Combined holographic representation [dim]
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
    
    /// Composed hologram [dim]
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
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "core" => Some(Self::Core),
            "support" => Some(Self::Support),
            "utility" => Some(Self::Utility),
            "integration" => Some(Self::Integration),
            _ => None,
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
    
    /// Composed hologram [dim]
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
        self.modules.iter()
            .flat_map(|m| m.components.iter())
            .filter(|c| c.id.name.contains(pattern) || c.id.module.contains(pattern))
            .collect()
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
        self.allowed.get(from)
            .map(|deps| deps.contains(&to.to_string()))
            .unwrap_or(false)
    }
    
    /// Add a dependency rule
    pub fn allow(&mut self, from: impl Into<String>, to: impl Into<String>) {
        self.allowed
            .entry(from.into())
            .or_default()
            .push(to.into());
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
    /// Git commit hash
    pub hash: String,
    /// Commit message
    pub message: String,
    /// Commit timestamp
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
        
        let mut scored: Vec<_> = self.commits.iter()
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
    
    /// Composed hologram [dim]
    pub hologram: Tensor,
    
    /// Boundary rules
    pub boundaries: BoundaryRules,
    
    /// Federation (dependencies on other organisms)
    pub federation: Vec<FederationBinding>,
    
    /// History tensor
    pub history: Option<HistoryTensor>,
    
    /// When this organism tensor was last updated
    pub updated_at: DateTime<Utc>,
    /// Git commit hash this tensor was built from
    pub commit: String,
    /// Embedding dimension used for all holograms
    pub embedding_dim: usize,
}

impl OrganismTensor {
    /// Get a bounded context by name
    pub fn get_context(&self, name: &str) -> Option<&BoundedContextTensor> {
        self.contexts.iter().find(|c| c.name == name)
    }
    
    /// Project a query to find relevant context
    pub fn project(&self, query_tensor: &Tensor) -> crate::Result<ProjectionResult> {
        use super::ops::{project as tensor_project, cosine_similarity};
        
        // Project against organism hologram
        let projection = tensor_project(&self.hologram, query_tensor)?;
        
        // Score each context
        let mut context_scores: Vec<(String, f32)> = self.contexts.iter()
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
    pub fn check_boundaries(&self, from_context: &str, used_items: &[String]) -> Vec<BoundaryViolation> {
        let mut violations = Vec::new();
        
        for item in used_items {
            // Find which context the item belongs to
            for ctx in &self.contexts {
                let item_in_ctx = ctx.modules.iter()
                    .any(|m| m.path.contains(item) || 
                         m.components.iter().any(|c| c.id.name == *item));
                
                if item_in_ctx && ctx.name != from_context {
                    if !self.boundaries.can_depend(from_context, &ctx.name) {
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
    /// Source bounded context that has the invalid dependency
    pub from_context: String,
    /// Target bounded context that was illegally depended upon
    pub to_context: String,
    /// Component that caused the violation
    pub component: String,
    /// Human-readable explanation of why this is a violation
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

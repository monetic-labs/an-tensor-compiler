//! Context manifest parsing
//!
//! Parses the `contexts.toml` file that defines an organism's
//! bounded contexts and their relationships.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use crate::{Result, TensorCoreError};

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

impl ContextManifest {
    /// Load manifest from file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(TensorCoreError::Io)?;
        
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
        self.boundaries.get(from)
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
}

impl Default for EncodingInfo {
    fn default() -> Self {
        Self {
            embedding_model: default_embedding_model(),
            embedding_dim: default_embedding_dim(),
            structural_dim: default_structural_dim(),
            position_encoding: default_position_encoding(),
        }
    }
}

fn default_embedding_model() -> String { "arctic-embed-l".to_string() }
fn default_embedding_dim() -> usize { 1024 }
fn default_structural_dim() -> usize { 256 }
fn default_position_encoding() -> String { "sinusoidal".to_string() }

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

fn default_role() -> String { "utility".to_string() }

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

fn default_history_depth() -> usize { 100 }

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

fn default_primary_model() -> String { "deepseek-coder-v2".to_string() }
fn default_fallback_model() -> String { "qwen25-coder-7b".to_string() }
fn default_min_coherence() -> f32 { 0.7 }
fn default_min_style() -> f32 { 0.6 }
fn default_require_tests() -> bool { true }
fn default_require_lint() -> bool { true }
fn default_max_diff() -> usize { 200 }

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
}

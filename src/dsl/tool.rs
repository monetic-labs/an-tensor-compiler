//! # Tool Pattern DSL
//!
//! Absorbs an agent's tool pattern DSL for tensor-native pattern matching.
//!
//! ## Absorption Strategy (80% straight, 20% tensorized)
//!
//! **Taken straight:**
//! - Tool definitions (interface, inputs, outputs, validation)
//! - Pattern structure (example_call, confidence, template, exemplars)
//!
//! **Tensorized:**
//! - `ExecutionStrategy` enums → RL policy tensors
//! - Pattern matching → embedding similarity
//! - Alternative tools → dynamic discovery via embeddings

use super::common::{dsl_namespace_path, namespaces, soft_id, DslError, DslResult};
use crate::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Tool Specification Types (Absorbed from an agent)
// =============================================================================

/// Tool definition with interface and validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Stable identifier for this tool.
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Functional category.
    pub category: ToolCategory,
    /// Tool version string.
    pub version: String,
    /// What the tool does.
    pub description: String,
    /// Declared inputs and outputs.
    pub interface: ToolInterface,
    /// Optional input validation rules.
    #[serde(default)]
    pub validation: Option<ValidationConfig>,
}

/// Tool categories.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ToolCategory {
    /// Remote API invocation.
    APICall,
    /// File read/write/list operations.
    FileSystem,
    /// Database queries and mutations.
    Database,
    /// Local computation.
    Compute,
    /// Messaging and notifications.
    Communication,
    /// Anything else.
    Custom,
}

/// Tool input/output interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInterface {
    /// Accepted input parameters.
    pub inputs: Vec<ParamSpec>,
    /// Produced output parameters.
    pub outputs: Vec<ParamSpec>,
}

/// Parameter specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamSpec {
    /// Parameter name.
    pub name: String,
    /// Expected value type.
    pub param_type: ParamType,
    /// Whether the caller must supply it.
    pub required: bool,
    /// What the parameter means.
    pub description: String,
}

/// Parameter types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ParamType {
    /// Text value.
    String,
    /// Numeric value.
    Number,
    /// True/false value.
    Boolean,
    /// Ordered list value.
    Array,
    /// Keyed object value.
    Object,
}

/// Validation configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationConfig {
    /// Rules keyed by input parameter name.
    #[serde(default)]
    pub input_validation: HashMap<String, Vec<ValidationRule>>,
}

/// Validation rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "rule_type", rename_all = "PascalCase")]
pub enum ValidationRule {
    /// Minimum string length.
    MinLength {
        /// Required minimum length.
        value: usize,
        /// Message on violation.
        error_message: String,
    },
    /// Maximum string length.
    MaxLength {
        /// Allowed maximum length.
        value: usize,
        /// Message on violation.
        error_message: String,
    },
    /// Regex pattern match.
    Pattern {
        /// Pattern the value must match.
        value: String,
        /// Message on violation.
        error_message: String,
    },
    /// Numeric range check.
    Range {
        /// Inclusive lower bound.
        min: f64,
        /// Inclusive upper bound.
        max: f64,
        /// Message on violation.
        error_message: String,
    },
}

// =============================================================================
// Tool Pattern Types (Learned from executions)
// =============================================================================

/// A learned tool usage pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPattern {
    /// Stable identifier for this pattern.
    pub id: String,
    /// Owning context id.
    pub context_id: String,
    /// Tool this pattern applies to.
    pub tool: String,
    /// Use case the pattern captures.
    pub use_case: String,
    /// Canonical example invocation.
    pub example_call: ExampleCall,

    // Success metrics
    /// Observed successes.
    pub success_count: u64,
    /// Observed failures.
    pub failure_count: u64,
    /// Confidence score derived from outcomes.
    pub confidence: f64,

    // Strategy
    /// How the tool should be executed.
    pub execution_strategy: ExecutionStrategy,

    // Alternatives
    /// Fallback tools with their own track records.
    #[serde(default)]
    pub alternative_tools: Vec<AlternativeTool>,

    // Prototype features
    /// Abstracted call template, if generalized.
    #[serde(default)]
    pub template: Option<ToolTemplate>,
    /// Template variables.
    #[serde(default)]
    pub variables: Vec<TemplateVariable>,
    /// Concrete usage exemplars.
    #[serde(default)]
    pub exemplars: Vec<Exemplar>,

    // Metadata
    /// Contexts this pattern applies to.
    #[serde(default)]
    pub applies_to: Vec<String>,
    /// When to prefer this pattern.
    #[serde(default)]
    pub when_to_use: Option<String>,
    /// Situations where this pattern fails.
    #[serde(default)]
    pub when_not_to_use: Vec<String>,
}

/// Example tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleCall {
    /// Tool being invoked.
    pub tool: String,
    /// Parameter values by name.
    pub params: HashMap<String, serde_json::Value>,
}

/// Execution strategy for tool patterns.
/// Tensorized: converted to RL policy tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "PascalCase")]
pub enum ExecutionStrategy {
    /// Clear winner (95%+ success)
    Single,
    /// Try primary, fallback if needed
    SequentialFallback {
        /// Observed primary success rate.
        primary_success_rate: f64,
        /// Observed rate of fallback use.
        fallback_needed_rate: f64,
    },
    /// Race multiple tools
    ParallelFirstSuccess {
        /// Whether the tools are interchangeable.
        tools_comparable: bool,
        /// Whether latency dominates the choice.
        speed_critical: bool,
    },
    /// Still learning
    Exploring {
        /// Attempts observed so far.
        attempts: u32,
    },
}

impl Default for ExecutionStrategy {
    fn default() -> Self {
        Self::Exploring { attempts: 0 }
    }
}

/// Alternative tool for fallback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeTool {
    /// Fallback tool name.
    pub tool: String,
    /// Canonical example invocation.
    pub example_call: ExampleCall,
    /// Observed successes.
    pub success_count: u64,
    /// Observed failures.
    pub failure_count: u64,
    /// Mean latency in milliseconds, if measured.
    #[serde(default)]
    pub avg_latency_ms: Option<u64>,
    /// Times used as a fallback.
    #[serde(default)]
    pub used_as_fallback: u64,
    /// Success rate when used as a fallback.
    #[serde(default)]
    pub fallback_success_rate: f64,
}

/// Template for pattern abstraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolTemplate {
    /// Tool being invoked.
    pub tool: String,
    /// Templated parameter values by name.
    pub params: HashMap<String, serde_json::Value>,
}

/// Template variable definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateVariable {
    /// Variable name as it appears in the template.
    pub name: String,
    /// What the variable captures.
    pub description: String,
    /// Observed example values.
    pub example_values: Vec<String>,
    /// Whether the template requires it.
    pub required: bool,
}

/// Concrete exemplar of pattern usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exemplar {
    /// Context the exemplar was observed in.
    pub context: String,
    /// Concrete variable bindings.
    pub concrete_values: HashMap<String, String>,
    /// Outcome of the call.
    pub result: ExemplarResult,
}

/// Outcome of an exemplar call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExemplarResult {
    /// Whether the call succeeded.
    pub success: bool,
    /// Observed latency in milliseconds, if measured.
    #[serde(default)]
    pub latency_ms: Option<u64>,
}

// =============================================================================
// Tensor Embeddings for Patterns
// =============================================================================

/// Embedded tool pattern for similarity matching.
#[derive(Debug)]
pub struct EmbeddedPattern {
    /// Pattern ID
    pub id: String,
    /// Pattern embedding vector
    pub embedding: Tensor,
    /// Confidence as scalar tensor
    pub confidence: Tensor,
    /// Strategy as one-hot tensor
    pub strategy: Tensor,
}

impl ToolPattern {
    /// Parse from YAML string.
    pub fn from_yaml(yaml: &str) -> DslResult<Self> {
        #[derive(Deserialize)]
        struct YamlDoc {
            pattern: ToolPattern,
        }
        let doc: YamlDoc = serde_yaml::from_str(yaml)?;
        Ok(doc.pattern)
    }

    /// Parse from JSON string.
    pub fn from_json(json: &str) -> DslResult<Self> {
        #[derive(Deserialize)]
        struct JsonDoc {
            pattern: ToolPattern,
        }
        let doc: JsonDoc = serde_json::from_str(json)?;
        Ok(doc.pattern)
    }

    /// Get namespace path for this pattern.
    pub fn namespace_path(&self) -> String {
        dsl_namespace_path(namespaces::TOOLS, &self.context_id, &self.id)
    }

    /// Embed this pattern to a tensor representation.
    pub fn embed(&self, device: &Device) -> DslResult<EmbeddedPattern> {
        // Create ID-based embedding
        let id_embedding = soft_id(&self.id, 32, device)?;

        // Create use-case embedding
        let use_case_embedding = soft_id(&self.use_case, 32, device)?;

        // Create tool embedding
        let tool_embedding = soft_id(&self.tool, 32, device)?;

        // Concatenate embeddings
        let embedding = Tensor::cat(&[&id_embedding, &use_case_embedding, &tool_embedding], 0)?;

        // Confidence as tensor
        let confidence = Tensor::from_vec(vec![self.confidence as f32], 1, device)?;

        // Strategy as one-hot
        let strategy = self.execution_strategy.to_tensor(device)?;

        Ok(EmbeddedPattern {
            id: self.id.clone(),
            embedding,
            confidence,
            strategy,
        })
    }

    /// Compute success rate.
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.5 // Unknown
        } else {
            self.success_count as f64 / total as f64
        }
    }
}

impl ExecutionStrategy {
    /// Convert strategy to one-hot tensor.
    pub fn to_tensor(&self, device: &Device) -> DslResult<Tensor> {
        let idx = match self {
            ExecutionStrategy::Single => 0,
            ExecutionStrategy::SequentialFallback { .. } => 1,
            ExecutionStrategy::ParallelFirstSuccess { .. } => 2,
            ExecutionStrategy::Exploring { .. } => 3,
        };
        super::common::one_hot(idx, 4, device)
    }

    /// Get strategy index (for RL).
    pub fn index(&self) -> usize {
        match self {
            ExecutionStrategy::Single => 0,
            ExecutionStrategy::SequentialFallback { .. } => 1,
            ExecutionStrategy::ParallelFirstSuccess { .. } => 2,
            ExecutionStrategy::Exploring { .. } => 3,
        }
    }
}

// =============================================================================
// Pattern Matching
// =============================================================================

/// Find similar patterns via embedding cosine similarity.
pub fn find_similar_patterns(
    query: &Tensor,
    patterns: &[EmbeddedPattern],
    k: usize,
) -> DslResult<Vec<(usize, f32)>> {
    let mut similarities: Vec<(usize, f32)> = Vec::with_capacity(patterns.len());

    for (i, pattern) in patterns.iter().enumerate() {
        let sim_val = crate::primitives::cosine_similarity(query, &pattern.embedding)
            .map_err(|e| DslError::Execution(e.to_string()))?;
        similarities.push((i, sim_val));
    }

    // Sort by similarity descending
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top k
    similarities.truncate(k);

    Ok(similarities)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_PATTERN_YAML: &str = r#"
pattern:
  id: "pattern-tool-abc123"
  context_id: "ctx_devops"
  tool: "terminal.execute_command"
  use_case: "Create DigitalOcean VPC"
  example_call:
    tool: "terminal.execute_command"
    params:
      command: "doctl vpc create --name my-vpc --region nyc1"
  success_count: 47
  failure_count: 2
  confidence: 0.92
  execution_strategy:
    type: SequentialFallback
    primary_success_rate: 0.96
    fallback_needed_rate: 0.04
  applies_to: ["cloud-infrastructure", "networking"]
  when_to_use: "Creating isolated network environments"
"#;

    #[test]
    fn test_parse_pattern_yaml() {
        let pattern = ToolPattern::from_yaml(SAMPLE_PATTERN_YAML).unwrap();
        assert_eq!(pattern.id, "pattern-tool-abc123");
        assert_eq!(pattern.confidence, 0.92);
        assert_eq!(pattern.success_count, 47);
    }

    #[test]
    fn test_success_rate() {
        let pattern = ToolPattern::from_yaml(SAMPLE_PATTERN_YAML).unwrap();
        let rate = pattern.success_rate();
        assert!((rate - 0.959).abs() < 0.01); // 47 / (47 + 2) ≈ 0.959
    }

    #[test]
    fn test_embed_pattern() {
        let pattern = ToolPattern::from_yaml(SAMPLE_PATTERN_YAML).unwrap();
        let device = Device::Cpu;
        let embedded = pattern.embed(&device).unwrap();

        // Check embedding dimensions: 32 + 32 + 32 = 96
        assert_eq!(embedded.embedding.dims(), &[96]);
        assert_eq!(embedded.confidence.dims(), &[1]);
        assert_eq!(embedded.strategy.dims(), &[4]);
    }

    #[test]
    fn test_strategy_to_tensor() {
        let device = Device::Cpu;

        let single = ExecutionStrategy::Single;
        let vec: Vec<f32> = single.to_tensor(&device).unwrap().to_vec1().unwrap();
        assert_eq!(vec, vec![1.0, 0.0, 0.0, 0.0]);

        let fallback = ExecutionStrategy::SequentialFallback {
            primary_success_rate: 0.9,
            fallback_needed_rate: 0.1,
        };
        let vec: Vec<f32> = fallback.to_tensor(&device).unwrap().to_vec1().unwrap();
        assert_eq!(vec, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_namespace_path() {
        let pattern = ToolPattern::from_yaml(SAMPLE_PATTERN_YAML).unwrap();
        assert_eq!(
            pattern.namespace_path(),
            "an-tensor-compiler.tools.patterns.ctx_devops.pattern-tool-abc123"
        );
    }
}

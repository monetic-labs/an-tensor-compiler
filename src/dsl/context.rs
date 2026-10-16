//! # Context & Learning DSL
//!
//! Absorbs an agent's context DSL for tensor-native knowledge management.
//!
//! ## Absorption Strategy (70% straight, 30% tensorized)
//!
//! **Taken straight:**
//! - Core context structure (mission, goals, principles)
//! - Dynamic node structure (decay metadata, relations)
//!
//! **Tensorized:**
//! - Decay algorithms → temporal tensor ops
//! - Promotion → learned confidence gates
//! - Graph relations → GNN edges

use super::common::{dsl_namespace_path, namespaces, soft_id, DslResult};
use crate::{Device, Tensor};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Core Context Types (Human-defined)
// =============================================================================

/// Core context definition (mission, goals, principles).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreContext {
    pub id: String,
    pub name: String,
    pub context_type: ContextType,
    pub status: ContextStatus,
    pub mission: Mission,
    #[serde(default)]
    pub goals: Vec<Goal>,
    #[serde(default)]
    pub principles: Vec<Principle>,
    #[serde(default)]
    pub measurables: Vec<Measurable>,
}

/// Context type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContextType {
    Organization,
    Project,
    Team,
    Individual,
}

/// Context status.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContextStatus {
    Active,
    Inactive,
    Archived,
}

/// Mission statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mission {
    pub statement: String,
    #[serde(default)]
    pub approved_by: Option<String>,
}

/// Goal definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: String,
    pub name: String,
    pub target: f64,
    pub unit: String,
    pub priority: u32,
    pub status: GoalStatus,
    #[serde(default)]
    pub deadline: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum GoalStatus {
    Active,
    Achieved,
    Failed,
    Paused,
}

/// Principle with weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Principle {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Weight for soft constraint tensor (0-10)
    pub weight: f64,
}

/// Measurable metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurable {
    pub id: String,
    pub name: String,
    pub formula: String,
    pub threshold: f64,
    pub alert_on: AlertCondition,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AlertCondition {
    AboveThreshold,
    BelowThreshold,
    OutsideRange,
}

// =============================================================================
// Dynamic Context Types (Agent-created, decays)
// =============================================================================

/// Dynamic context node that decays over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicNode {
    pub id: String,
    pub context_id: String,
    pub node_type: String,
    pub created_by: String,

    // Decay metadata
    pub access_count: u64,
    pub last_accessed: DateTime<Utc>,
    pub decay_weight: f64,
    pub decay_algorithm: DecayAlgorithm,

    // Content
    pub data: HashMap<String, serde_json::Value>,

    // Graph structure
    #[serde(default)]
    pub related_to: Vec<String>,

    // References (deprecated: use CRDT keys)
    #[serde(default)]
    pub references: Vec<Reference>,
}

/// Reference to external data (deprecated).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    #[serde(rename = "type")]
    pub ref_type: String,
    #[serde(default)]
    pub table: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
}

/// Decay algorithms - all implemented as tensor ops.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "PascalCase")]
pub enum DecayAlgorithm {
    Fibonacci {
        initial_weight: f64,
        decay_rate: f64,
    },
    Logarithmic {
        initial_weight: f64,
        decay_constant: f64,
    },
    Quadratic {
        initial_weight: f64,
        acceleration: f64,
    },
    AccessBased {
        base_decay: f64,
        access_boost: f64,
    },
}

impl Default for DecayAlgorithm {
    fn default() -> Self {
        Self::AccessBased {
            base_decay: 0.01,
            access_boost: 0.1,
        }
    }
}

// =============================================================================
// Learned Principle Types (Promoted from patterns)
// =============================================================================

/// A principle learned from successful patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedPrinciple {
    pub id: String,
    pub context_id: String,
    pub name: String,
    pub description: String,
    /// Learned weight (0-10)
    pub weight: f64,

    // Provenance
    pub learned_from: Vec<String>,
    pub source_pattern_type: PatternType,
    pub success_rate: f64,
    pub total_uses: u64,

    // Governance (tensorized to confidence threshold)
    pub approval_status: ApprovalStatus,
    #[serde(default)]
    pub approved_by: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PatternType {
    Tool,
    Workflow,
    Context,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Rejected,
    Revoked,
}

// =============================================================================
// Tensor Operations for Context
// =============================================================================

impl DecayAlgorithm {
    /// Compute decay weight as tensor operation.
    ///
    /// # Arguments
    /// * `age_seconds` - Time since creation
    /// * `access_count` - Number of accesses
    /// * `device` - Tensor device
    pub fn compute_weight(
        &self,
        age_seconds: f64,
        access_count: u64,
        device: &Device,
    ) -> DslResult<Tensor> {
        let weight = match self {
            DecayAlgorithm::Fibonacci {
                initial_weight,
                decay_rate,
            } => {
                // Fibonacci-like decay: w(t) = initial * decay^(fib_index(t))
                let fib_index = (age_seconds / 86400.0).log2().max(1.0); // Days → fib index
                initial_weight * decay_rate.powf(fib_index)
            }
            DecayAlgorithm::Logarithmic {
                initial_weight,
                decay_constant,
            } => {
                // w(t) = initial / (1 + decay * ln(1 + t))
                let log_age = (1.0 + age_seconds).ln();
                initial_weight / (1.0 + decay_constant * log_age)
            }
            DecayAlgorithm::Quadratic {
                initial_weight,
                acceleration,
            } => {
                // w(t) = initial * max(0, 1 - acceleration * t^2)
                let decay = acceleration * age_seconds * age_seconds;
                initial_weight * (1.0 - decay).max(0.0)
            }
            DecayAlgorithm::AccessBased {
                base_decay,
                access_boost,
            } => {
                // w(t) = max(0, 1 - base_decay * t + access_boost * access_count)
                let decay_term = base_decay * age_seconds;
                let boost_term = access_boost * access_count as f64;
                (1.0 - decay_term + boost_term).clamp(0.0, 1.0)
            }
        };

        Ok(Tensor::from_vec(vec![weight as f32], 1, device)?)
    }

    /// Get algorithm index for one-hot encoding.
    pub fn index(&self) -> usize {
        match self {
            DecayAlgorithm::Fibonacci { .. } => 0,
            DecayAlgorithm::Logarithmic { .. } => 1,
            DecayAlgorithm::Quadratic { .. } => 2,
            DecayAlgorithm::AccessBased { .. } => 3,
        }
    }
}

impl CoreContext {
    /// Parse from YAML string.
    pub fn from_yaml(yaml: &str) -> DslResult<Self> {
        #[derive(Deserialize)]
        struct YamlDoc {
            core_context: CoreContext,
        }
        let doc: YamlDoc = serde_yaml::from_str(yaml)?;
        Ok(doc.core_context)
    }

    /// Get namespace path.
    pub fn namespace_path(&self) -> String {
        dsl_namespace_path(namespaces::CONTEXT, "core", &self.id)
    }

    /// Embed mission statement to tensor.
    pub fn embed_mission(&self, device: &Device) -> DslResult<Tensor> {
        soft_id(&self.mission.statement, 128, device)
    }

    /// Get principles as weight tensor.
    pub fn principles_weights(&self, device: &Device) -> DslResult<Tensor> {
        let weights: Vec<f32> = self
            .principles
            .iter()
            .map(|p| (p.weight / 10.0) as f32) // Normalize to [0, 1]
            .collect();

        if weights.is_empty() {
            return Ok(Tensor::zeros(1, candle_core::DType::F32, device)?);
        }

        Ok(Tensor::from_vec(weights.clone(), weights.len(), device)?)
    }

    /// Get goals as constraint tensors.
    pub fn goals_constraints(&self, device: &Device) -> DslResult<Vec<(String, Tensor)>> {
        let mut constraints = Vec::new();

        for goal in &self.goals {
            // Each goal becomes a threshold constraint
            let target = Tensor::from_vec(vec![goal.target as f32], 1, device)?;
            constraints.push((goal.id.clone(), target));
        }

        Ok(constraints)
    }
}

impl DynamicNode {
    /// Parse from YAML string.
    pub fn from_yaml(yaml: &str) -> DslResult<Self> {
        #[derive(Deserialize)]
        struct YamlDoc {
            dynamic_node: DynamicNode,
        }
        let doc: YamlDoc = serde_yaml::from_str(yaml)?;
        Ok(doc.dynamic_node)
    }

    /// Get namespace path.
    pub fn namespace_path(&self) -> String {
        dsl_namespace_path(namespaces::CONTEXT, "dynamic", &self.id)
    }

    /// Compute current weight after decay.
    pub fn current_weight(&self, now: DateTime<Utc>, device: &Device) -> DslResult<Tensor> {
        let age_seconds = (now - self.last_accessed).num_seconds().max(0) as f64;
        self.decay_algorithm
            .compute_weight(age_seconds, self.access_count, device)
    }

    /// Get relation edges for GNN.
    pub fn relation_edges(&self) -> Vec<(String, String)> {
        self.related_to
            .iter()
            .map(|target| (self.id.clone(), target.clone()))
            .collect()
    }
}

impl LearnedPrinciple {
    /// Evaluate if principle should be promoted based on confidence gate.
    ///
    /// Returns true if success_rate exceeds learned threshold.
    pub fn should_promote(&self, threshold: f64) -> bool {
        self.success_rate >= threshold && self.total_uses >= 10
    }

    /// Get confidence as tensor.
    pub fn confidence_tensor(&self, device: &Device) -> DslResult<Tensor> {
        Ok(Tensor::from_vec(vec![self.success_rate as f32], 1, device)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CONTEXT_YAML: &str = r#"
core_context:
  id: "ctx_devops"
  name: "DevOps Operations"
  context_type: Project
  status: Active
  mission:
    statement: "Automate infrastructure with reliability and cost efficiency"
    approved_by: "team-lead"
  goals:
    - id: "g1"
      name: "Deployment Success Rate"
      target: 99.5
      unit: "percent"
      priority: 1
      status: Active
  principles:
    - id: "p1"
      name: "Immutable Infrastructure"
      description: "Never modify running instances"
      weight: 9.0
    - id: "p2"
      name: "Least Privilege"
      description: "Minimal permissions"
      weight: 8.5
"#;

    const SAMPLE_NODE_YAML: &str = r#"
dynamic_node:
  id: "node_12345"
  context_id: "ctx_devops"
  node_type: "learned_pattern"
  created_by: "agent-123"
  access_count: 15
  last_accessed: "2025-12-24T10:00:00Z"
  decay_weight: 0.85
  decay_algorithm:
    type: AccessBased
    base_decay: 0.01
    access_boost: 0.1
  data:
    insight: "VPC creation faster via CLI"
  related_to: ["node_12340", "node_12342"]
"#;

    #[test]
    fn test_parse_core_context() {
        let ctx = CoreContext::from_yaml(SAMPLE_CONTEXT_YAML).unwrap();
        assert_eq!(ctx.id, "ctx_devops");
        assert_eq!(ctx.context_type, ContextType::Project);
        assert_eq!(ctx.principles.len(), 2);
        assert_eq!(ctx.goals.len(), 1);
    }

    #[test]
    fn test_embed_mission() {
        let ctx = CoreContext::from_yaml(SAMPLE_CONTEXT_YAML).unwrap();
        let device = Device::Cpu;
        let embedding = ctx.embed_mission(&device).unwrap();
        assert_eq!(embedding.dims(), &[128]);
    }

    #[test]
    fn test_principles_weights() {
        let ctx = CoreContext::from_yaml(SAMPLE_CONTEXT_YAML).unwrap();
        let device = Device::Cpu;
        let weights = ctx.principles_weights(&device).unwrap();
        assert_eq!(weights.dims(), &[2]);

        let vec: Vec<f32> = weights.to_vec1().unwrap();
        assert!((vec[0] - 0.9).abs() < 0.01); // 9.0 / 10
        assert!((vec[1] - 0.85).abs() < 0.01); // 8.5 / 10
    }

    #[test]
    fn test_parse_dynamic_node() {
        let node = DynamicNode::from_yaml(SAMPLE_NODE_YAML).unwrap();
        assert_eq!(node.id, "node_12345");
        assert_eq!(node.access_count, 15);
        assert_eq!(node.related_to.len(), 2);
    }

    #[test]
    fn test_decay_access_based() {
        let algo = DecayAlgorithm::AccessBased {
            base_decay: 0.00001,
            access_boost: 0.05,
        };
        let device = Device::Cpu;

        // With high access count, weight should be boosted
        let weight = algo.compute_weight(3600.0, 10, &device).unwrap();
        let val: f32 = weight.to_vec1().unwrap()[0];
        assert!(val > 0.5); // Access boost should keep weight high
    }

    #[test]
    fn test_decay_logarithmic() {
        let algo = DecayAlgorithm::Logarithmic {
            initial_weight: 1.0,
            decay_constant: 0.1,
        };
        let device = Device::Cpu;

        let weight = algo.compute_weight(86400.0, 0, &device).unwrap();
        let val: f32 = weight.to_vec1().unwrap()[0];
        assert!(val > 0.0 && val < 1.0); // Should decay but not fully
    }

    #[test]
    fn test_namespace_path() {
        let ctx = CoreContext::from_yaml(SAMPLE_CONTEXT_YAML).unwrap();
        assert_eq!(
            ctx.namespace_path(),
            "an-tensor-compiler.context.core.ctx_devops"
        );

        let node = DynamicNode::from_yaml(SAMPLE_NODE_YAML).unwrap();
        assert_eq!(
            node.namespace_path(),
            "an-tensor-compiler.context.dynamic.node_12345"
        );
    }
}

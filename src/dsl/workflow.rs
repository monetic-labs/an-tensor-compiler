//! # Workflow Orchestration DSL
//!
//! Absorbs an agent's workflow DSL for tensor-native execution.
//!
//! ## Patterns Supported
//! - **Saga**: Compensatable sequences with rollback
//! - **DAG**: Parallel execution with dependencies
//! - **Linear**: Sequential execution
//! - **Chain**: Meta-orchestration of workflows
//!
//! ## Absorption Strategy (90% straight, 10% tensorized)
//!
//! **Taken straight:**
//! - YAML structure (patterns, steps, dependencies)
//! - Compensation strategies as mask patterns
//!
//! **Tensorized:**
//! - `agents: []` → `op: tensor_*` (direct tensor ops)
//! - `execute_if` → soft gates (sigmoid)
//! - Timeouts → GPU scheduler hints

use super::common::{
    dsl_namespace_path, namespaces, one_hot, soft_id, CompensationMask, DslError, DslResult,
    TensorNode, TensorOp,
};
use crate::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Workflow Specification Types (Absorbed from an agent)
// =============================================================================

/// Top-level workflow specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowSpec {
    /// Stable identifier for this workflow.
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Execution pattern.
    pub pattern: WorkflowPattern,
    /// Pattern-specific definition.
    pub definition: WorkflowDefinition,
}

/// Workflow execution patterns.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum WorkflowPattern {
    /// Steps with compensation on failure.
    Saga,
    /// Dependency-ordered directed acyclic graph.
    DAG,
    /// Strictly sequential steps.
    Linear,
}

/// Pattern-specific workflow definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "PascalCase")]
pub enum WorkflowDefinition {
    /// Saga steps with a compensation strategy.
    Saga {
        /// Ordered steps.
        steps: Vec<WorkflowStep>,
        /// How to compensate on failure.
        compensation_strategy: CompensationStrategy,
    },
    /// DAG nodes executed in dependency order.
    DAG {
        /// Graph nodes.
        nodes: Vec<WorkflowStep>,
    },
    /// Strictly sequential steps.
    Linear {
        /// Ordered steps.
        steps: Vec<WorkflowStep>,
    },
}

/// A single step/node in a workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    /// Stable identifier for this step.
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Tensor operation to execute (replaces legacy `agents` field)
    #[serde(default)]
    pub op: Option<TensorOp>,
    /// Legacy agent refs (deprecated, converted to ops)
    #[serde(default)]
    pub agents: Vec<String>,
    /// Whether the step may run alongside siblings.
    #[serde(default)]
    pub parallelizable: bool,
    /// Ids of steps that must complete first.
    #[serde(default)]
    pub dependencies: Vec<String>,
    /// Per-step timeout in seconds, if any.
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
    /// Compensation to run if a later step fails.
    #[serde(default)]
    pub compensation: Option<CompensationConfig>,
}

/// Compensation configuration for saga steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompensationConfig {
    /// Tensor op for compensation (replaces legacy `agents`)
    #[serde(default)]
    pub op: Option<TensorOp>,
    /// Legacy agent refs (deprecated)
    #[serde(default)]
    pub agents: Vec<String>,
    /// Times to retry compensation before giving up.
    #[serde(default)]
    pub max_retries: u32,
}

/// Compensation strategies (map directly to mask patterns).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
#[derive(Default)]
pub enum CompensationStrategy {
    /// Compensate all completed steps
    #[default]
    RollbackAll,
    /// Only compensate steps that depend on failed step
    RollbackDependents,
    /// Try to repair instead of rollback
    ForwardRecovery,
}

// =============================================================================
// Workflow Chain Types (Meta-orchestration)
// =============================================================================

/// Chain of workflows for meta-orchestration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowChain {
    /// Stable identifier for this chain.
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// How the chained workflows execute.
    pub chain_pattern: ChainPattern,
    /// Workflows in chain order.
    pub workflows: Vec<ChainedWorkflow>,
}

/// Chain execution patterns.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum ChainPattern {
    /// Run workflows one after another.
    Sequential,
    /// Run each workflow only if its condition holds.
    Conditional,
    /// Run all workflows concurrently.
    Parallel,
    /// Repeat the chain until exit.
    Loop,
}

/// A workflow in a chain with execution conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainedWorkflow {
    /// Referenced workflow id.
    pub workflow_id: String,
    /// Zero-based position in the chain.
    pub position: u32,
    /// Optional gate condition.
    #[serde(default)]
    pub execute_if: Option<ExecuteCondition>,
}

/// Execution condition for chained workflows.
/// These are tensorized to soft gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "PascalCase")]
pub enum ExecuteCondition {
    /// String equality check.
    Equals {
        /// Field to read.
        field: String,
        /// Expected value.
        value: String,
    },
    /// Numeric greater-than check.
    GreaterThan {
        /// Field to read.
        field: String,
        /// Threshold value.
        value: f64,
    },
    /// Numeric less-than check.
    LessThan {
        /// Field to read.
        field: String,
        /// Threshold value.
        value: f64,
    },
    /// All sub-conditions hold.
    And {
        /// Sub-conditions.
        conditions: Vec<ExecuteCondition>,
    },
    /// Any sub-condition holds.
    Or {
        /// Sub-conditions.
        conditions: Vec<ExecuteCondition>,
    },
}

// =============================================================================
// Tensor Graph Compilation
// =============================================================================

/// A compiled tensor execution graph.
#[derive(Debug)]
pub struct TensorGraph {
    /// Workflow ID
    pub id: String,
    /// Nodes in topological order
    pub nodes: Vec<TensorNode>,
    /// Sparse adjacency matrix [n, n]
    pub adjacency: Tensor,
    /// Pattern type
    pub pattern: WorkflowPattern,
    /// Compensation strategy (for Saga)
    pub compensation_strategy: Option<CompensationStrategy>,
    /// Execution device
    device: Device,
}

impl TensorGraph {
    /// Compile a workflow spec to a tensor graph.
    pub fn compile(spec: &WorkflowSpec, device: &Device) -> DslResult<Self> {
        let steps = match &spec.definition {
            WorkflowDefinition::Saga { steps, .. } => steps,
            WorkflowDefinition::DAG { nodes } => nodes,
            WorkflowDefinition::Linear { steps } => steps,
        };

        let compensation_strategy = match &spec.definition {
            WorkflowDefinition::Saga {
                compensation_strategy,
                ..
            } => Some(*compensation_strategy),
            _ => None,
        };

        // Build node index
        let node_index: HashMap<&str, usize> = steps
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id.as_str(), i))
            .collect();

        // Build adjacency matrix
        let n = steps.len();
        let mut adj = vec![0.0f32; n * n];
        for (i, step) in steps.iter().enumerate() {
            for dep in &step.dependencies {
                if let Some(&j) = node_index.get(dep.as_str()) {
                    adj[j * n + i] = 1.0; // Edge from dep to step
                }
            }
        }
        let adjacency = Tensor::from_vec(adj, (n, n), device)?;

        // Build tensor nodes
        let mut nodes = Vec::with_capacity(n);
        for step in steps {
            let embedding = soft_id(&step.id, 64, device)?;

            // Convert legacy agents to tensor ops
            let op = step.op.clone().unwrap_or_else(|| {
                if !step.agents.is_empty() {
                    // Legacy: convert agent call to custom op
                    TensorOp::Custom {
                        name: format!("agent_call_{}", step.agents.join("_")),
                        params: HashMap::new(),
                    }
                } else {
                    TensorOp::Identity
                }
            });

            nodes.push(TensorNode {
                id: step.id.clone(),
                embedding: Some(embedding),
                op,
                dependencies: step.dependencies.clone(),
                parallelizable: step.parallelizable,
                gate: None, // No conditional for individual steps
            });
        }

        Ok(Self {
            id: spec.id.clone(),
            nodes,
            adjacency,
            pattern: spec.pattern,
            compensation_strategy,
            device: device.clone(),
        })
    }

    /// Get the namespace path for this compiled graph.
    pub fn namespace_path(&self) -> String {
        dsl_namespace_path(namespaces::WORKFLOWS, "compiled", &self.id)
    }

    /// Get number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get nodes that can run in parallel (no unmet dependencies).
    ///
    /// Delegates the predecessor-completion check to
    /// [`scheduling::ready_set`](crate::scheduling::ready_set) (the canonical
    /// implementation) and then applies this graph's `parallelizable` filter on
    /// top. Do not re-implement the dep check inline — it lives in `scheduling`.
    pub fn ready_nodes(&self, completed: &[bool]) -> Vec<usize> {
        let edges = self.predecessor_edges();
        let ready = crate::scheduling::ready_set(&edges, completed);

        // Filter to parallelizable nodes for batching (TensorGraph-specific).
        if ready.len() > 1 {
            let parallelizable: Vec<usize> = ready
                .iter()
                .cloned()
                .filter(|&i| self.nodes[i].parallelizable)
                .collect();
            if !parallelizable.is_empty() {
                return parallelizable;
            }
        }

        ready
    }

    /// Build the index-based predecessor edge list (`edges[i]` = predecessor
    /// indices of node `i`) from this graph's string-ID `dependencies`.
    ///
    /// This is the bridge between `TensorGraph`'s string-ID node model and the
    /// index-based [`scheduling`](crate::scheduling) primitives.
    fn predecessor_edges(&self) -> Vec<Vec<usize>> {
        let id_to_idx: HashMap<&str, usize> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id.as_str(), i))
            .collect();
        self.nodes
            .iter()
            .map(|node| {
                node.dependencies
                    .iter()
                    .filter_map(|dep| id_to_idx.get(dep.as_str()).copied())
                    .collect()
            })
            .collect()
    }

    /// Generate compensation mask for a failed node.
    pub fn compensation_mask(&self, failed_idx: usize) -> DslResult<CompensationMask> {
        let strategy = self.compensation_strategy.unwrap_or_default();
        let node_ids: Vec<String> = self.nodes.iter().map(|n| n.id.clone()).collect();

        match strategy {
            CompensationStrategy::RollbackAll => {
                CompensationMask::rollback_all(node_ids, &self.device)
            }
            CompensationStrategy::RollbackDependents => CompensationMask::rollback_dependents(
                node_ids,
                failed_idx,
                &self.adjacency,
                &self.device,
            ),
            CompensationStrategy::ForwardRecovery => {
                // For forward recovery, we don't rollback - just mark failed
                let mut mask_vec = vec![0.0f32; self.nodes.len()];
                mask_vec[failed_idx] = 1.0;
                let mask = Tensor::from_vec(mask_vec, self.nodes.len(), &self.device)?;
                Ok(CompensationMask { mask, node_ids })
            }
        }
    }

    /// Get compensation strategy as one-hot tensor.
    pub fn compensation_strategy_tensor(&self) -> DslResult<Option<Tensor>> {
        match self.compensation_strategy {
            Some(CompensationStrategy::RollbackAll) => Ok(Some(one_hot(0, 3, &self.device)?)),
            Some(CompensationStrategy::RollbackDependents) => {
                Ok(Some(one_hot(1, 3, &self.device)?))
            }
            Some(CompensationStrategy::ForwardRecovery) => Ok(Some(one_hot(2, 3, &self.device)?)),
            None => Ok(None),
        }
    }
}

impl WorkflowSpec {
    /// Parse from YAML string.
    pub fn from_yaml(yaml: &str) -> DslResult<Self> {
        // Extract the workflow from the document
        #[derive(Deserialize)]
        struct YamlDoc {
            workflow: WorkflowSpec,
        }
        let doc: YamlDoc = serde_yaml::from_str(yaml)?;
        Ok(doc.workflow)
    }

    /// Parse from JSON string.
    pub fn from_json(json: &str) -> DslResult<Self> {
        #[derive(Deserialize)]
        struct JsonDoc {
            workflow: WorkflowSpec,
        }
        let doc: JsonDoc = serde_json::from_str(json)?;
        Ok(doc.workflow)
    }

    /// Validate the workflow spec.
    pub fn validate(&self) -> DslResult<()> {
        let steps = match &self.definition {
            WorkflowDefinition::Saga { steps, .. } => steps,
            WorkflowDefinition::DAG { nodes } => nodes,
            WorkflowDefinition::Linear { steps } => steps,
        };

        // Check for duplicate IDs
        let mut seen = std::collections::HashSet::new();
        for step in steps {
            if !seen.insert(&step.id) {
                return Err(DslError::Validation(format!(
                    "Duplicate step ID: {}",
                    step.id
                )));
            }
        }

        // Check dependencies exist
        let step_ids: std::collections::HashSet<&str> =
            steps.iter().map(|s| s.id.as_str()).collect();

        for step in steps {
            for dep in &step.dependencies {
                if !step_ids.contains(dep.as_str()) {
                    return Err(DslError::Validation(format!(
                        "Step '{}' depends on unknown step '{}'",
                        step.id, dep
                    )));
                }
            }
        }

        // Check for circular dependencies (DAG property)
        if self.pattern == WorkflowPattern::DAG {
            self.validate_acyclic(steps)?;
        }

        Ok(())
    }

    fn validate_acyclic(&self, steps: &[WorkflowStep]) -> DslResult<()> {
        // Kahn's algorithm for topological sort
        let n = steps.len();
        let node_index: HashMap<&str, usize> = steps
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id.as_str(), i))
            .collect();

        let mut in_degree = vec![0usize; n];
        for step in steps {
            for dep in &step.dependencies {
                if let Some(&i) = node_index.get(dep.as_str()) {
                    let step_idx = node_index[step.id.as_str()];
                    in_degree[step_idx] += 1;
                    let _ = i; // Suppress warning
                }
            }
        }

        let mut queue: Vec<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, &d)| d == 0)
            .map(|(i, _)| i)
            .collect();

        let mut visited = 0;
        while let Some(node) = queue.pop() {
            visited += 1;
            let step = &steps[node];
            for other_step in steps {
                if other_step.dependencies.contains(&step.id) {
                    let other_idx = node_index[other_step.id.as_str()];
                    in_degree[other_idx] -= 1;
                    if in_degree[other_idx] == 0 {
                        queue.push(other_idx);
                    }
                }
            }
        }

        if visited != n {
            return Err(DslError::Validation(
                "Circular dependency detected in DAG".into(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_DAG_YAML: &str = r#"
workflow:
  id: "wf_parallel_deploy"
  name: "ParallelDeploy"
  pattern: DAG
  definition:
    type: DAG
    nodes:
      - id: "build_frontend"
        name: "Build Frontend"
        op:
          type: custom
          name: "build"
          params: {}
        dependencies: []
        parallelizable: true
      - id: "build_backend"
        name: "Build Backend"
        op:
          type: custom
          name: "build"
          params: {}
        dependencies: []
        parallelizable: true
      - id: "deploy"
        name: "Deploy Both"
        op:
          type: custom
          name: "deploy"
          params: {}
        dependencies: ["build_frontend", "build_backend"]
        parallelizable: false
"#;

    const SAMPLE_SAGA_YAML: &str = r#"
workflow:
  id: "wf_regime_fusion"
  name: "RegimeFusion"
  pattern: Saga
  definition:
    type: Saga
    steps:
      - id: "load_data"
        name: "Load Historical Data"
        op:
          type: tensor_load
          path: "acme.trading.regime.history"
        dependencies: []
        parallelizable: true
        compensation:
          op:
            type: identity
          max_retries: 3
      - id: "fuse"
        name: "Fuse Vectors"
        op:
          type: rule_forward
          rule_name: "regime_fuse"
        dependencies: ["load_data"]
        parallelizable: false
    compensation_strategy: RollbackDependents
"#;

    #[test]
    fn test_parse_dag_yaml() {
        let spec = WorkflowSpec::from_yaml(SAMPLE_DAG_YAML).unwrap();
        assert_eq!(spec.id, "wf_parallel_deploy");
        assert_eq!(spec.pattern, WorkflowPattern::DAG);

        if let WorkflowDefinition::DAG { nodes } = &spec.definition {
            assert_eq!(nodes.len(), 3);
        } else {
            panic!("Expected DAG definition");
        }
    }

    #[test]
    fn test_parse_saga_yaml() {
        let spec = WorkflowSpec::from_yaml(SAMPLE_SAGA_YAML).unwrap();
        assert_eq!(spec.id, "wf_regime_fusion");
        assert_eq!(spec.pattern, WorkflowPattern::Saga);

        if let WorkflowDefinition::Saga {
            steps,
            compensation_strategy,
        } = &spec.definition
        {
            assert_eq!(steps.len(), 2);
            assert_eq!(
                *compensation_strategy,
                CompensationStrategy::RollbackDependents
            );
        } else {
            panic!("Expected Saga definition");
        }
    }

    #[test]
    fn test_validate_dag() {
        let spec = WorkflowSpec::from_yaml(SAMPLE_DAG_YAML).unwrap();
        spec.validate().unwrap();
    }

    #[test]
    fn test_validate_circular_dependency() {
        let yaml = r#"
workflow:
  id: "wf_circular"
  name: "Circular"
  pattern: DAG
  definition:
    type: DAG
    nodes:
      - id: "a"
        name: "A"
        dependencies: ["b"]
      - id: "b"
        name: "B"
        dependencies: ["a"]
"#;
        let spec = WorkflowSpec::from_yaml(yaml).unwrap();
        let result = spec.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Circular"));
    }

    #[test]
    fn test_compile_to_tensor_graph() {
        let spec = WorkflowSpec::from_yaml(SAMPLE_DAG_YAML).unwrap();
        let device = Device::Cpu;
        let graph = TensorGraph::compile(&spec, &device).unwrap();

        assert_eq!(graph.len(), 3);
        assert_eq!(graph.pattern, WorkflowPattern::DAG);

        // Check adjacency matrix shape
        assert_eq!(graph.adjacency.dims(), &[3, 3]);
    }

    #[test]
    fn test_ready_nodes() {
        let spec = WorkflowSpec::from_yaml(SAMPLE_DAG_YAML).unwrap();
        let device = Device::Cpu;
        let graph = TensorGraph::compile(&spec, &device).unwrap();

        // Initially, build_frontend and build_backend should be ready
        let ready = graph.ready_nodes(&[false, false, false]);
        assert_eq!(ready.len(), 2);

        // After both builds complete, deploy should be ready
        let ready = graph.ready_nodes(&[true, true, false]);
        assert_eq!(ready.len(), 1);
        assert_eq!(graph.nodes[ready[0]].id, "deploy");
    }

    #[test]
    fn test_compensation_mask_rollback_all() {
        let spec = WorkflowSpec::from_yaml(SAMPLE_SAGA_YAML).unwrap();
        let device = Device::Cpu;
        let mut graph = TensorGraph::compile(&spec, &device).unwrap();
        graph.compensation_strategy = Some(CompensationStrategy::RollbackAll);

        let mask = graph.compensation_mask(0).unwrap();
        let vec: Vec<f32> = mask.mask.to_vec1().unwrap();
        assert_eq!(vec, vec![1.0, 1.0]); // All nodes marked
    }

    #[test]
    fn test_namespace_path() {
        let spec = WorkflowSpec::from_yaml(SAMPLE_DAG_YAML).unwrap();
        let device = Device::Cpu;
        let graph = TensorGraph::compile(&spec, &device).unwrap();

        assert_eq!(
            graph.namespace_path(),
            "an-tensor-compiler.workflows.compiled.wf_parallel_deploy"
        );
    }
}

//! # DSL Executor Traits
//!
//! Defines the execution interfaces for tensor-native DSL operations.
//!
//! These traits abstract the tensor execution layer, allowing:
//! - GPU-accelerated workflow execution
//! - Learnable pattern matching
//! - Differentiable context management
//!
//! ## Traits
//!
//! - [`TensorWorkflowExecutor`]: Compile and execute workflow DAGs
//! - [`TensorPatternMatcher`]: Embed and match tool patterns
//! - [`TensorContextManager`]: Manage context decay and activation

use super::common::{DslError, DslResult, TensorNode, TensorOp};
use super::context::{DynamicNode, LearnedPrinciple};
use super::tool::{EmbeddedPattern, ExecutionStrategy, ToolPattern};
use super::workflow::{TensorGraph, WorkflowSpec};
use crate::compiler::CompiledRuleSet;
use crate::{Device, Tensor, Var};
use async_trait::async_trait;
use candle_core::IndexOp;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Arc;

// =============================================================================
// Workflow Executor
// =============================================================================

/// Batch of tensor inputs for workflow execution.
#[derive(Debug, Clone)]
pub struct TensorBatch {
    /// Named tensors (input name → tensor)
    pub inputs: HashMap<String, Tensor>,
    /// Batch size
    pub batch_size: usize,
}

impl TensorBatch {
    /// Create a new batch with inputs.
    pub fn new(inputs: HashMap<String, Tensor>, batch_size: usize) -> Self {
        Self { inputs, batch_size }
    }

    /// Create an empty batch.
    pub fn empty() -> Self {
        Self {
            inputs: HashMap::new(),
            batch_size: 0,
        }
    }

    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.inputs.get(name)
    }

    /// Insert a tensor.
    pub fn insert(&mut self, name: String, tensor: Tensor) {
        self.inputs.insert(name, tensor);
    }
}

/// Result of workflow execution.
#[derive(Debug)]
pub struct WorkflowResult {
    /// Output tensors from each completed node
    pub node_outputs: HashMap<String, Tensor>,
    /// Execution trace for debugging
    pub trace: Vec<ExecutionStep>,
    /// Whether execution completed successfully
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// A single step in the execution trace.
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    /// Node ID
    pub node_id: String,
    /// Operation executed
    pub op: TensorOp,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Success status
    pub success: bool,
}

/// Executor for tensor-compiled workflows.
///
/// Implements the workflow execution interface from an agent spec:
/// - Compile DAG to tensor graph
/// - Execute in parallel batches
/// - Handle compensation on failure
///
/// # Example
///
/// ```ignore
/// let executor = DefaultWorkflowExecutor::new(&device);
/// let graph = executor.compile(&spec)?;
/// let result = executor.execute(&graph, &inputs).await?;
/// if !result.success {
///     executor.compensate(&graph, &result.trace).await?;
/// }
/// ```
#[async_trait]
pub trait TensorWorkflowExecutor: Send + Sync {
    /// Compile a workflow spec to a tensor graph.
    fn compile(&self, spec: &WorkflowSpec) -> DslResult<TensorGraph>;

    /// Execute the workflow with batch inputs.
    async fn execute(&self, graph: &TensorGraph, inputs: &TensorBatch)
        -> DslResult<WorkflowResult>;

    /// Execute compensation for a failed workflow.
    async fn compensate(&self, graph: &TensorGraph, trace: &[ExecutionStep]) -> DslResult<()>;
}

/// Default implementation of workflow executor.
///
/// Supports executing tensor operations including TL rule forwards.
/// Register compiled rules via `register_rule()` before executing workflows
/// that reference them.
pub struct DefaultWorkflowExecutor {
    device: Device,
    /// Registry of compiled TL rules by name
    rules: HashMap<String, Arc<CompiledRuleSet>>,
}

impl DefaultWorkflowExecutor {
    /// Create a new executor for the given device.
    pub fn new(device: Device) -> Self {
        Self {
            device,
            rules: HashMap::new(),
        }
    }

    /// Register a compiled rule set for use in workflows.
    ///
    /// Rules must be registered before executing workflows that
    /// contain `TensorOp::RuleForward` referencing them.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let spec = RuleSpec::parse("exit_rules", "exit(X) :- profit(X).")?;
    /// let compiled = CompiledRuleSet::compile(spec, &device)?;
    /// executor.register_rule("exit_rules", compiled);
    /// ```
    pub fn register_rule(&mut self, name: impl Into<String>, rule: CompiledRuleSet) {
        self.rules.insert(name.into(), Arc::new(rule));
    }

    /// Register a rule from an Arc (for sharing across executors).
    pub fn register_rule_arc(&mut self, name: impl Into<String>, rule: Arc<CompiledRuleSet>) {
        self.rules.insert(name.into(), rule);
    }

    /// Get a registered rule by name.
    pub fn get_rule(&self, name: &str) -> Option<Arc<CompiledRuleSet>> {
        self.rules.get(name).cloned()
    }

    /// List all registered rule names.
    pub fn rule_names(&self) -> Vec<String> {
        self.rules.keys().cloned().collect()
    }
}

#[async_trait]
impl TensorWorkflowExecutor for DefaultWorkflowExecutor {
    fn compile(&self, spec: &WorkflowSpec) -> DslResult<TensorGraph> {
        spec.validate()?;
        TensorGraph::compile(spec, &self.device)
    }

    async fn execute(
        &self,
        graph: &TensorGraph,
        inputs: &TensorBatch,
    ) -> DslResult<WorkflowResult> {
        let mut node_outputs: HashMap<String, Tensor> = HashMap::new();
        let mut trace: Vec<ExecutionStep> = Vec::new();
        let mut completed = vec![false; graph.len()];

        // Process nodes in dependency order
        loop {
            let ready = graph.ready_nodes(&completed);
            if ready.is_empty() {
                break;
            }

            // Execute ready nodes (potentially in parallel)
            for node_idx in ready {
                let node = &graph.nodes[node_idx];
                let start = std::time::Instant::now();

                // Execute the node's operation
                let result = self.execute_node(node, inputs, &node_outputs).await;
                let duration_ms = start.elapsed().as_millis() as u64;

                match result {
                    Ok(output) => {
                        node_outputs.insert(node.id.clone(), output);
                        completed[node_idx] = true;
                        trace.push(ExecutionStep {
                            node_id: node.id.clone(),
                            op: node.op.clone(),
                            duration_ms,
                            success: true,
                        });
                    }
                    Err(e) => {
                        trace.push(ExecutionStep {
                            node_id: node.id.clone(),
                            op: node.op.clone(),
                            duration_ms,
                            success: false,
                        });
                        return Ok(WorkflowResult {
                            node_outputs,
                            trace,
                            success: false,
                            error: Some(e.to_string()),
                        });
                    }
                }
            }
        }

        Ok(WorkflowResult {
            node_outputs,
            trace,
            success: true,
            error: None,
        })
    }

    async fn compensate(&self, graph: &TensorGraph, trace: &[ExecutionStep]) -> DslResult<()> {
        // Find the first failed step
        let failed_idx = trace
            .iter()
            .position(|s| !s.success)
            .ok_or_else(|| DslError::Execution("No failed step to compensate".into()))?;

        let failed_node_id = &trace[failed_idx].node_id;
        let node_idx = graph
            .nodes
            .iter()
            .position(|n| n.id == *failed_node_id)
            .ok_or_else(|| DslError::Execution("Failed node not found in graph".into()))?;

        // Generate compensation mask
        let mask = graph.compensation_mask(node_idx)?;

        // Execute compensation for masked nodes (in reverse order)
        for (i, step) in trace.iter().enumerate().rev() {
            let mask_val: f32 = mask.mask.i(i)?.to_vec0()?;
            if mask_val > 0.5 && step.success {
                // This node needs compensation
                tracing::info!("Compensating node: {}", step.node_id);
                // In a real implementation, we'd execute the compensation op here
            }
        }

        Ok(())
    }
}

impl DefaultWorkflowExecutor {
    async fn execute_node(
        &self,
        node: &TensorNode,
        inputs: &TensorBatch,
        prior_outputs: &HashMap<String, Tensor>,
    ) -> DslResult<Tensor> {
        match &node.op {
            TensorOp::TensorLoad { path } => {
                // Load from inputs or prior outputs
                inputs
                    .get(path)
                    .or_else(|| prior_outputs.get(path))
                    .cloned()
                    .ok_or_else(|| DslError::Execution(format!("Tensor not found: {}", path)))
            }
            TensorOp::RuleForward { rule_name } => {
                // Look up the compiled rule and execute it
                let rule = self.rules.get(rule_name).ok_or_else(|| {
                    DslError::Execution(format!(
                        "Rule '{}' not found. Register it via executor.register_rule() first.",
                        rule_name
                    ))
                })?;

                // Convert prior outputs to the format expected by CompiledRuleSet
                let mut rule_inputs: HashMap<String, Tensor> = inputs.inputs.clone();
                rule_inputs.extend(prior_outputs.clone());

                // Execute the rule forward pass
                let output = rule
                    .forward(&rule_inputs)
                    .map_err(|e| DslError::Execution(format!("Rule forward failed: {}", e)))?;

                Ok(output.output)
            }
            TensorOp::Custom { name, params: _ } => {
                // Custom ops would be registered handlers
                tracing::debug!("Executing custom op: {}", name);
                Ok(Tensor::ones(1, candle_core::DType::F32, &self.device)?)
            }
            TensorOp::Identity => {
                // Return a unit tensor
                Ok(Tensor::ones(1, candle_core::DType::F32, &self.device)?)
            }
        }
    }
}

// =============================================================================
// Pattern Matcher
// =============================================================================

/// Result of pattern matching.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// Matched pattern ID
    pub pattern_id: String,
    /// Similarity score (0-1)
    pub similarity: f32,
    /// Confidence of the match
    pub confidence: f32,
}

/// Outcome of a tool execution for strategy updates.
#[derive(Debug, Clone, Copy)]
pub enum ToolOutcome {
    /// Tool executed successfully
    Success { latency_ms: u64 },
    /// Tool failed
    Failure { recoverable: bool },
    /// Tool timed out
    Timeout,
}

/// Learnable pattern matcher for tool patterns.
///
/// Implements the pattern matching interface from an agent spec:
/// - Embed patterns to vector space
/// - Find similar patterns via cosine similarity
/// - Update execution strategies based on outcomes
///
/// # Example
///
/// ```ignore
/// let matcher = DefaultPatternMatcher::new(&device);
/// let embedded = matcher.embed(&pattern)?;
/// let similar = matcher.find_similar(&embedded.embedding, 5).await?;
/// matcher.update_strategy("pattern_123", ToolOutcome::Success { latency_ms: 100 }).await?;
/// ```
#[async_trait]
pub trait TensorPatternMatcher: Send + Sync {
    /// Embed a tool pattern to a tensor representation.
    fn embed(&self, pattern: &ToolPattern) -> DslResult<EmbeddedPattern>;

    /// Find patterns similar to the given embedding.
    async fn find_similar(&self, embedding: &Tensor, k: usize) -> DslResult<Vec<PatternMatch>>;

    /// Update execution strategy based on outcome.
    async fn update_strategy(
        &self,
        pattern_id: &str,
        outcome: ToolOutcome,
    ) -> DslResult<ExecutionStrategy>;

    /// Get all embedded patterns (for batch operations).
    fn all_patterns(&self) -> &[EmbeddedPattern];
}

/// Default implementation of pattern matcher.
pub struct DefaultPatternMatcher {
    device: Device,
    patterns: Vec<EmbeddedPattern>,
    /// Learnable strategy weights (for RL-based selection)
    strategy_weights: Option<Var>,
}

impl DefaultPatternMatcher {
    /// Create a new pattern matcher.
    pub fn new(device: Device) -> Self {
        Self {
            device,
            patterns: Vec::new(),
            strategy_weights: None,
        }
    }

    /// Register a pattern for matching.
    pub fn register(&mut self, pattern: &ToolPattern) -> DslResult<()> {
        let embedded = pattern.embed(&self.device)?;
        self.patterns.push(embedded);
        Ok(())
    }

    /// Initialize learnable strategy weights.
    pub fn init_strategy_weights(&mut self, num_strategies: usize) -> DslResult<()> {
        let weights = Tensor::ones(num_strategies, candle_core::DType::F32, &self.device)?;
        self.strategy_weights = Some(Var::from_tensor(&weights)?);
        Ok(())
    }

    /// Get trainable variables for optimization.
    pub fn trainable_vars(&self) -> Vec<Var> {
        self.strategy_weights.iter().cloned().collect()
    }
}

#[async_trait]
impl TensorPatternMatcher for DefaultPatternMatcher {
    fn embed(&self, pattern: &ToolPattern) -> DslResult<EmbeddedPattern> {
        pattern.embed(&self.device)
    }

    async fn find_similar(&self, embedding: &Tensor, k: usize) -> DslResult<Vec<PatternMatch>> {
        let mut matches = Vec::new();

        for pattern in &self.patterns {
            let sim = crate::primitives::cosine_similarity(embedding, &pattern.embedding)
                .map_err(|e| DslError::Execution(e.to_string()))?;
            let confidence: f32 = pattern.confidence.to_vec1()?[0];

            matches.push(PatternMatch {
                pattern_id: pattern.id.clone(),
                similarity: sim,
                confidence,
            });
        }

        // Sort by similarity descending
        matches.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(k);

        Ok(matches)
    }

    async fn update_strategy(
        &self,
        pattern_id: &str,
        outcome: ToolOutcome,
    ) -> DslResult<ExecutionStrategy> {
        // Find the pattern
        let pattern = self
            .patterns
            .iter()
            .find(|p| p.id == pattern_id)
            .ok_or_else(|| DslError::Execution(format!("Pattern not found: {}", pattern_id)))?;

        // Get current strategy from one-hot tensor
        let strategy_vec: Vec<f32> = pattern.strategy.to_vec1()?;
        let current_idx = strategy_vec.iter().position(|&v| v > 0.5).unwrap_or(3); // Default to Exploring

        // Update strategy based on outcome
        let new_strategy = match outcome {
            ToolOutcome::Success { .. } => {
                // Success reinforces current strategy or promotes to Single
                if current_idx == 3 {
                    // From Exploring → SequentialFallback
                    ExecutionStrategy::SequentialFallback {
                        primary_success_rate: 0.8,
                        fallback_needed_rate: 0.2,
                    }
                } else if current_idx == 1 {
                    // High success in Fallback → Single
                    ExecutionStrategy::Single
                } else {
                    // Stay at current
                    ExecutionStrategy::Single
                }
            }
            ToolOutcome::Failure { recoverable } => {
                if recoverable {
                    // Demote to fallback
                    ExecutionStrategy::SequentialFallback {
                        primary_success_rate: 0.5,
                        fallback_needed_rate: 0.5,
                    }
                } else {
                    // Explore alternatives
                    ExecutionStrategy::Exploring { attempts: 0 }
                }
            }
            ToolOutcome::Timeout => {
                // Try parallel for speed
                ExecutionStrategy::ParallelFirstSuccess {
                    tools_comparable: true,
                    speed_critical: true,
                }
            }
        };

        Ok(new_strategy)
    }

    fn all_patterns(&self) -> &[EmbeddedPattern] {
        &self.patterns
    }
}

// =============================================================================
// Context Manager
// =============================================================================

/// Context for activation propagation.
#[derive(Debug, Clone)]
pub struct ActivationContext {
    /// Current activation strength
    pub activation: f32,
    /// Decay factor per hop
    pub decay_per_hop: f32,
    /// Maximum hops to propagate
    pub max_hops: usize,
    /// Minimum activation threshold
    pub min_activation: f32,
}

impl Default for ActivationContext {
    fn default() -> Self {
        Self {
            activation: 1.0,
            decay_per_hop: 0.5,
            max_hops: 3,
            min_activation: 0.1,
        }
    }
}

/// Manager for tensor-native context operations.
///
/// Implements the context management interface from an agent spec:
/// - Apply decay to dynamic nodes
/// - Propagate activation through knowledge graph
/// - Evaluate pattern promotion to principles
///
/// # Example
///
/// ```ignore
/// let manager = DefaultContextManager::new(&device);
/// manager.decay_step("ctx_devops", Utc::now()).await?;
/// let activated = manager.propagate_activation("node_123", &ctx).await?;
/// let promoted = manager.evaluate_promotion(&pattern).await?;
/// ```
#[async_trait]
pub trait TensorContextManager: Send + Sync {
    /// Apply decay to all dynamic nodes in a context.
    async fn decay_step(&self, context_id: &str, now: DateTime<Utc>) -> DslResult<Vec<String>>;

    /// Propagate activation from a node through the knowledge graph.
    async fn propagate_activation(
        &self,
        node_id: &str,
        context: &ActivationContext,
    ) -> DslResult<HashMap<String, f32>>;

    /// Evaluate if a pattern should be promoted to a principle.
    async fn evaluate_promotion(
        &self,
        pattern: &ToolPattern,
        threshold: f32,
    ) -> DslResult<Option<LearnedPrinciple>>;
}

/// Default implementation of context manager.
pub struct DefaultContextManager {
    device: Device,
    /// Registered dynamic nodes by ID
    nodes: HashMap<String, DynamicNode>,
    /// Adjacency list for graph structure
    edges: HashMap<String, Vec<String>>,
    /// Promotion threshold (default 0.9)
    #[allow(dead_code)]
    default_threshold: f32,
}

impl DefaultContextManager {
    /// Create a new context manager.
    pub fn new(device: Device) -> Self {
        Self {
            device,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            default_threshold: 0.9,
        }
    }

    /// Register a dynamic node.
    pub fn register_node(&mut self, node: DynamicNode) {
        // Add edges from this node's relations
        for target in &node.related_to {
            self.edges
                .entry(node.id.clone())
                .or_default()
                .push(target.clone());
        }
        self.nodes.insert(node.id.clone(), node);
    }

    /// Get all nodes in a context.
    pub fn nodes_in_context(&self, context_id: &str) -> Vec<&DynamicNode> {
        self.nodes
            .values()
            .filter(|n| n.context_id == context_id)
            .collect()
    }
}

#[async_trait]
impl TensorContextManager for DefaultContextManager {
    async fn decay_step(&self, context_id: &str, now: DateTime<Utc>) -> DslResult<Vec<String>> {
        let mut pruned = Vec::new();

        for node in self.nodes_in_context(context_id) {
            let weight = node.current_weight(now, &self.device)?;
            let weight_val: f32 = weight.to_vec1()?[0];

            if weight_val < 0.01 {
                // Node has decayed below threshold - mark for pruning
                pruned.push(node.id.clone());
            }
        }

        Ok(pruned)
    }

    async fn propagate_activation(
        &self,
        node_id: &str,
        context: &ActivationContext,
    ) -> DslResult<HashMap<String, f32>> {
        let mut activations: HashMap<String, f32> = HashMap::new();
        let mut to_visit: Vec<(String, f32, usize)> =
            vec![(node_id.to_string(), context.activation, 0)];

        while let Some((current_id, activation, hop)) = to_visit.pop() {
            if hop >= context.max_hops || activation < context.min_activation {
                continue;
            }

            // Record activation
            let entry = activations.entry(current_id.clone()).or_insert(0.0);
            *entry = entry.max(activation);

            // Propagate to neighbors
            if let Some(neighbors) = self.edges.get(&current_id) {
                for neighbor in neighbors {
                    let new_activation = activation * context.decay_per_hop;
                    if new_activation >= context.min_activation {
                        to_visit.push((neighbor.clone(), new_activation, hop + 1));
                    }
                }
            }
        }

        Ok(activations)
    }

    async fn evaluate_promotion(
        &self,
        pattern: &ToolPattern,
        threshold: f32,
    ) -> DslResult<Option<LearnedPrinciple>> {
        let success_rate = pattern.success_rate();
        let total_uses = pattern.success_count + pattern.failure_count;

        // Check promotion criteria
        if success_rate >= threshold as f64 && total_uses >= 10 {
            Ok(Some(LearnedPrinciple {
                id: format!("principle-from-{}", pattern.id),
                context_id: pattern.context_id.clone(),
                name: format!("Learned: {}", pattern.use_case),
                description: pattern.when_to_use.clone().unwrap_or_default(),
                weight: (success_rate * 10.0).min(10.0),
                learned_from: vec![pattern.id.clone()],
                source_pattern_type: super::context::PatternType::Tool,
                success_rate,
                total_uses,
                approval_status: super::context::ApprovalStatus::Pending,
                approved_by: None,
            }))
        } else {
            Ok(None)
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_batch() {
        let device = Device::Cpu;
        let mut batch = TensorBatch::empty();
        batch.insert(
            "input".into(),
            Tensor::ones(10, candle_core::DType::F32, &device).unwrap(),
        );

        assert_eq!(batch.inputs.len(), 1);
        assert!(batch.get("input").is_some());
        assert!(batch.get("missing").is_none());
    }

    #[test]
    fn test_workflow_executor_compile() {
        let device = Device::Cpu;
        let executor = DefaultWorkflowExecutor::new(device);

        let yaml = r#"
workflow:
  id: "wf_test"
  name: "Test Workflow"
  pattern: DAG
  definition:
    type: DAG
    nodes:
      - id: "step1"
        name: "Step 1"
        dependencies: []
      - id: "step2"
        name: "Step 2"
        dependencies: ["step1"]
"#;
        let spec = WorkflowSpec::from_yaml(yaml).unwrap();
        let graph = executor.compile(&spec).unwrap();

        assert_eq!(graph.len(), 2);
    }

    #[tokio::test]
    async fn test_workflow_executor_execute() {
        let device = Device::Cpu;
        let executor = DefaultWorkflowExecutor::new(device.clone());

        let yaml = r#"
workflow:
  id: "wf_test"
  name: "Test"
  pattern: Linear
  definition:
    type: Linear
    steps:
      - id: "only_step"
        name: "Only Step"
        op:
          type: identity
        dependencies: []
"#;
        let spec = WorkflowSpec::from_yaml(yaml).unwrap();
        let graph = executor.compile(&spec).unwrap();
        let inputs = TensorBatch::empty();

        let result = executor.execute(&graph, &inputs).await.unwrap();
        assert!(result.success);
        assert_eq!(result.trace.len(), 1);
    }

    #[test]
    fn test_pattern_matcher_register() {
        let device = Device::Cpu;
        let mut matcher = DefaultPatternMatcher::new(device);

        let yaml = r#"
pattern:
  id: "test_pattern"
  context_id: "ctx_test"
  tool: "test.tool"
  use_case: "Test"
  example_call:
    tool: "test.tool"
    params: {}
  success_count: 10
  failure_count: 1
  confidence: 0.9
  execution_strategy:
    type: Single
"#;
        let pattern = ToolPattern::from_yaml(yaml).unwrap();
        matcher.register(&pattern).unwrap();

        assert_eq!(matcher.all_patterns().len(), 1);
    }

    #[tokio::test]
    async fn test_pattern_matcher_find_similar() {
        let device = Device::Cpu;
        let mut matcher = DefaultPatternMatcher::new(device.clone());

        let yaml = r#"
pattern:
  id: "pattern1"
  context_id: "ctx"
  tool: "tool1"
  use_case: "Use case 1"
  example_call:
    tool: "tool1"
    params: {}
  success_count: 10
  failure_count: 0
  confidence: 0.95
  execution_strategy:
    type: Single
"#;
        let pattern = ToolPattern::from_yaml(yaml).unwrap();
        matcher.register(&pattern).unwrap();

        let query = pattern.embed(&device).unwrap();
        let matches = matcher.find_similar(&query.embedding, 5).await.unwrap();

        assert_eq!(matches.len(), 1);
        assert!(matches[0].similarity > 0.99); // Should match itself
    }

    #[tokio::test]
    async fn test_update_strategy_success() {
        let device = Device::Cpu;
        let mut matcher = DefaultPatternMatcher::new(device);

        let yaml = r#"
pattern:
  id: "pattern1"
  context_id: "ctx"
  tool: "tool1"
  use_case: "Test"
  example_call:
    tool: "tool1"
    params: {}
  success_count: 5
  failure_count: 0
  confidence: 0.9
  execution_strategy:
    type: Exploring
    attempts: 5
"#;
        let pattern = ToolPattern::from_yaml(yaml).unwrap();
        matcher.register(&pattern).unwrap();

        let new_strategy = matcher
            .update_strategy("pattern1", ToolOutcome::Success { latency_ms: 100 })
            .await
            .unwrap();

        // Should promote from Exploring to SequentialFallback
        match new_strategy {
            ExecutionStrategy::SequentialFallback { .. } => (),
            _ => panic!("Expected SequentialFallback after success from Exploring"),
        }
    }

    #[test]
    fn test_context_manager_register_node() {
        let device = Device::Cpu;
        let mut manager = DefaultContextManager::new(device);

        let yaml = r#"
dynamic_node:
  id: "node1"
  context_id: "ctx_test"
  node_type: "test"
  created_by: "test"
  access_count: 5
  last_accessed: "2025-12-25T12:00:00Z"
  decay_weight: 0.9
  decay_algorithm:
    type: AccessBased
    base_decay: 0.01
    access_boost: 0.1
  data: {}
  related_to: ["node2"]
"#;
        let node = DynamicNode::from_yaml(yaml).unwrap();
        manager.register_node(node);

        assert_eq!(manager.nodes_in_context("ctx_test").len(), 1);
        assert!(manager.edges.contains_key("node1"));
    }

    #[tokio::test]
    async fn test_propagate_activation() {
        let device = Device::Cpu;
        let mut manager = DefaultContextManager::new(device);

        // Create a chain: node1 -> node2 -> node3
        let nodes = vec![
            ("node1", vec!["node2"]),
            ("node2", vec!["node3"]),
            ("node3", vec![]),
        ];

        for (id, related) in nodes {
            let related_strings: Vec<String> = related.iter().map(|s| s.to_string()).collect();
            manager.nodes.insert(
                id.to_string(),
                DynamicNode {
                    id: id.to_string(),
                    context_id: "ctx".to_string(),
                    node_type: "test".to_string(),
                    created_by: "test".to_string(),
                    access_count: 1,
                    last_accessed: Utc::now(),
                    decay_weight: 1.0,
                    decay_algorithm: crate::dsl::context::DecayAlgorithm::default(),
                    data: HashMap::new(),
                    related_to: related_strings.clone(),
                    references: vec![],
                },
            );
            if !related.is_empty() {
                manager.edges.insert(id.to_string(), related_strings);
            }
        }

        let ctx = ActivationContext {
            activation: 1.0,
            decay_per_hop: 0.5,
            max_hops: 3,
            min_activation: 0.1,
        };

        let activations = manager.propagate_activation("node1", &ctx).await.unwrap();

        assert!(activations.contains_key("node1"));
        assert!(activations.contains_key("node2"));
        assert!(activations.contains_key("node3"));
        assert!((activations["node1"] - 1.0).abs() < 0.01);
        assert!((activations["node2"] - 0.5).abs() < 0.01);
        assert!((activations["node3"] - 0.25).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_evaluate_promotion_success() {
        let device = Device::Cpu;
        let manager = DefaultContextManager::new(device);

        let yaml = r#"
pattern:
  id: "high_success_pattern"
  context_id: "ctx"
  tool: "tool"
  use_case: "Highly successful pattern"
  example_call:
    tool: "tool"
    params: {}
  success_count: 95
  failure_count: 5
  confidence: 0.95
  execution_strategy:
    type: Single
  when_to_use: "When you need reliable results"
"#;
        let pattern = ToolPattern::from_yaml(yaml).unwrap();
        let promoted = manager.evaluate_promotion(&pattern, 0.9).await.unwrap();

        assert!(promoted.is_some());
        let principle = promoted.unwrap();
        assert_eq!(principle.context_id, "ctx");
        assert!(principle.success_rate >= 0.9);
    }

    #[tokio::test]
    async fn test_evaluate_promotion_failure() {
        let device = Device::Cpu;
        let manager = DefaultContextManager::new(device);

        let yaml = r#"
pattern:
  id: "low_success_pattern"
  context_id: "ctx"
  tool: "tool"
  use_case: "Low success pattern"
  example_call:
    tool: "tool"
    params: {}
  success_count: 5
  failure_count: 5
  confidence: 0.5
  execution_strategy:
    type: Exploring
    attempts: 10
"#;
        let pattern = ToolPattern::from_yaml(yaml).unwrap();
        let promoted = manager.evaluate_promotion(&pattern, 0.9).await.unwrap();

        assert!(promoted.is_none());
    }
}

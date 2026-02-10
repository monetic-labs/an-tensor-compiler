//! # CRDT Tensor Operations
//!
//! Conflict-free replicated data type semantics for multi-device tensor sync.
//!
//! ## Key Types
//!
//! - [`TensorCrdt`]: Trait for CRDT-compatible tensor types
//! - [`TensorMergeStrategy`]: Predefined merge strategies
//! - [`TensorDelta`]: Incremental sync deltas
//! - [`VectorClock`]: Causality tracking
//!
//! ## Usage
//!
//! ```rust,ignore
//! use an_tensor_compiler::crdt::{TensorCrdt, VectorClock, TensorDelta};
//!
//! // Sync between devices
//! let delta = device_a.delta_since(last_sync_version);
//! device_b.apply_delta(&delta)?;
//!
//! // Or full merge
//! device_a.merge(&device_b)?;
//! ```

use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{Result, TensorCoreError};

// ============================================================================
// Merge Strategies
// ============================================================================

/// Merge strategies for different tensor types.
///
/// Each strategy defines how to combine two versions of the same tensor.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum TensorMergeStrategy {
    /// Last writer wins (by timestamp).
    ///
    /// Simple but may lose concurrent updates.
    LastWriterWins,

    /// Element-wise maximum.
    ///
    /// Good for attention tensors where we want to preserve
    /// the highest attention any device has given to a context.
    ElementMax,

    /// Weighted average by freshness.
    ///
    /// Newer updates get more weight.
    WeightedAverage {
        /// Decay factor for older values (0.0-1.0)
        decay: f32,
    },

    /// Superimpose (add weighted).
    ///
    /// For holographic tensors where concepts stack.
    Superimpose,

    /// Union of sparse elements.
    ///
    /// For pattern libraries where we want all patterns.
    Union,

    /// Custom merge function identifier.
    ///
    /// Organisms register custom mergers.
    Custom {
        /// Identifier for registered merge function
        merger_id: String,
    },
}

impl Default for TensorMergeStrategy {
    fn default() -> Self {
        Self::LastWriterWins
    }
}

// ============================================================================
// Vector Clock
// ============================================================================

/// Simple vector clock for causality tracking.
///
/// Each node maintains a counter that increments on local updates.
/// Comparing clocks determines happens-before relationships.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct VectorClock {
    /// Map from node ID to logical clock value
    pub clocks: HashMap<String, u64>,
}

impl VectorClock {
    /// Create a new empty vector clock
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with initial node
    pub fn with_node(node_id: impl Into<String>) -> Self {
        let mut clocks = HashMap::new();
        clocks.insert(node_id.into(), 0);
        Self { clocks }
    }

    /// Increment the clock for a node
    pub fn increment(&mut self, node_id: &str) {
        *self.clocks.entry(node_id.to_string()).or_insert(0) += 1;
    }

    /// Get the clock value for a node
    pub fn get(&self, node_id: &str) -> u64 {
        self.clocks.get(node_id).copied().unwrap_or(0)
    }

    /// Merge with another vector clock (take max of each)
    pub fn merge(&mut self, other: &VectorClock) {
        for (node, &clock) in &other.clocks {
            let entry = self.clocks.entry(node.clone()).or_insert(0);
            *entry = (*entry).max(clock);
        }
    }

    /// Check if this clock happens-before another.
    ///
    /// Returns true if all components of self <= other.
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        self.clocks.iter().all(|(node, &clock)| {
            other.clocks.get(node).copied().unwrap_or(0) >= clock
        })
    }

    /// Check if clocks are concurrent (neither happens-before the other)
    pub fn concurrent_with(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }

    /// Get total clock value (sum of all components)
    pub fn total(&self) -> u64 {
        self.clocks.values().sum()
    }

    /// Check if clock is empty (no updates)
    pub fn is_empty(&self) -> bool {
        self.clocks.is_empty() || self.clocks.values().all(|&v| v == 0)
    }
}

// ============================================================================
// Tensor Delta
// ============================================================================

/// Delta for incremental sync between nodes.
///
/// Instead of sending full tensors, send only the changes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorDelta {
    /// Source version (vector clock before delta)
    pub from_clock: VectorClock,
    /// Target version (vector clock after delta)
    pub to_clock: VectorClock,
    /// Operations to apply
    pub operations: Vec<TensorOp>,
    /// Source node ID
    pub source_node: String,
    /// Timestamp when delta was created
    pub timestamp: i64,
}

impl TensorDelta {
    /// Create a new delta
    pub fn new(
        source_node: impl Into<String>,
        from_clock: VectorClock,
        to_clock: VectorClock,
    ) -> Self {
        Self {
            from_clock,
            to_clock,
            operations: Vec::new(),
            source_node: source_node.into(),
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    /// Add an operation to the delta
    pub fn add_op(&mut self, op: TensorOp) {
        self.operations.push(op);
    }

    /// Check if delta is empty
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Number of operations in delta
    pub fn len(&self) -> usize {
        self.operations.len()
    }
}

/// Tensor operations for delta application.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum TensorOp {
    /// Blend entire tensor with weight.
    Blend {
        /// Tensor data (flattened)
        tensor: Vec<f32>,
        /// Shape
        shape: Vec<usize>,
        /// Blend weight (0.0-1.0)
        weight: f32,
    },

    /// Add to holographic superposition.
    Superimpose {
        /// Tensor data (flattened)
        tensor: Vec<f32>,
        /// Shape
        shape: Vec<usize>,
        /// Superposition weight
        weight: f32,
    },

    /// Scatter update (sparse).
    Scatter {
        /// Indices to update
        indices: Vec<usize>,
        /// Values to set
        values: Vec<f32>,
    },

    /// Update context attention.
    AttentionUpdate {
        /// Context path
        context: String,
        /// New attention value
        attention: f32,
        /// Merge strategy
        strategy: AttentionMerge,
    },

    /// Append to trajectory.
    TrajectoryAppend {
        /// Serialized projection point
        point_json: String,
    },

    /// Set a scalar value.
    SetScalar {
        /// Field name
        field: String,
        /// Value
        value: f64,
    },

    /// Increment a counter.
    Increment {
        /// Field name
        field: String,
        /// Amount to increment
        amount: i64,
    },
}

/// How to merge attention values
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionMerge {
    /// Take the maximum
    Max,
    /// Take the latest
    Latest,
    /// Average
    Average,
    /// Sum (for counting)
    Sum,
}

// ============================================================================
// CRDT Trait
// ============================================================================

/// Trait for CRDT-compatible tensor types.
///
/// Implements merge semantics for multi-device synchronization.
pub trait TensorCrdt: Sized {
    /// Get merge strategy for this type
    fn merge_strategy(&self) -> TensorMergeStrategy;

    /// Merge with another instance.
    ///
    /// Result should be deterministic regardless of merge order.
    fn merge(&mut self, other: &Self) -> Result<()>;

    /// Compute delta since a given version.
    ///
    /// Returns None if no changes since that version.
    fn delta_since(&self, since_clock: &VectorClock) -> Option<TensorDelta>;

    /// Apply delta from another node.
    fn apply_delta(&mut self, delta: &TensorDelta) -> Result<()>;

    /// Get current vector clock.
    fn vector_clock(&self) -> &VectorClock;

    /// Get mutable vector clock.
    fn vector_clock_mut(&mut self) -> &mut VectorClock;

    /// Record a local update (increments clock).
    fn record_update(&mut self, node_id: &str) {
        self.vector_clock_mut().increment(node_id);
    }
}

// ============================================================================
// CRDT Helpers
// ============================================================================

/// Apply element-wise maximum merge to two tensors.
pub fn tensor_element_max(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dims() != b.dims() {
        return Err(TensorCoreError::Tensor(format!(
            "Shape mismatch in element_max: {:?} vs {:?}",
            a.dims(),
            b.dims()
        )));
    }
    a.maximum(b).map_err(Into::into)
}

/// Apply weighted blend to two tensors.
pub fn tensor_blend(a: &Tensor, b: &Tensor, b_weight: f32) -> Result<Tensor> {
    if a.dims() != b.dims() {
        return Err(TensorCoreError::Tensor(format!(
            "Shape mismatch in blend: {:?} vs {:?}",
            a.dims(),
            b.dims()
        )));
    }
    let a_weight = 1.0 - b_weight;
    let result = (a * a_weight as f64)?.add(&(b * b_weight as f64)?)?;
    Ok(result)
}

/// Apply superimpose (weighted addition) to tensors.
pub fn tensor_superimpose(base: &Tensor, add: &Tensor, weight: f32) -> Result<Tensor> {
    if base.dims() != add.dims() {
        return Err(TensorCoreError::Tensor(format!(
            "Shape mismatch in superimpose: {:?} vs {:?}",
            base.dims(),
            add.dims()
        )));
    }
    base.add(&(add * weight as f64)?).map_err(Into::into)
}

// ============================================================================
// CRDT State Container
// ============================================================================

/// Generic CRDT container for any serializable state.
///
/// Wraps state with vector clock and merge strategy.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrdtState<T> {
    /// The actual state
    pub state: T,
    /// Vector clock
    pub clock: VectorClock,
    /// Merge strategy
    pub strategy: TensorMergeStrategy,
    /// Node ID
    pub node_id: String,
    /// Last update timestamp
    pub last_update: i64,
}

impl<T: Clone> CrdtState<T> {
    /// Create new CRDT state
    pub fn new(state: T, node_id: impl Into<String>, strategy: TensorMergeStrategy) -> Self {
        let node_id = node_id.into();
        Self {
            state,
            clock: VectorClock::with_node(&node_id),
            strategy,
            node_id,
            last_update: chrono::Utc::now().timestamp(),
        }
    }

    /// Update the state, incrementing the clock
    pub fn update(&mut self, new_state: T) {
        self.state = new_state;
        self.clock.increment(&self.node_id);
        self.last_update = chrono::Utc::now().timestamp();
    }

    /// Get the state
    pub fn get(&self) -> &T {
        &self.state
    }
}

// ============================================================================
// Attention Map CRDT
// ============================================================================

/// CRDT for context attention maps.
///
/// Supports multi-device sync of attention distributions.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AttentionMapCrdt {
    /// Attention values per context
    pub attention: HashMap<String, f32>,
    /// Vector clock
    pub clock: VectorClock,
    /// Node ID
    pub node_id: String,
    /// Per-context last-update timestamps
    pub timestamps: HashMap<String, i64>,
}

impl AttentionMapCrdt {
    /// Create new attention map CRDT
    pub fn new(node_id: impl Into<String>) -> Self {
        let node_id = node_id.into();
        Self {
            attention: HashMap::new(),
            clock: VectorClock::with_node(&node_id),
            node_id,
            timestamps: HashMap::new(),
        }
    }

    /// Update attention for a context
    pub fn update(&mut self, context: impl Into<String>, attention: f32) {
        let context = context.into();
        self.attention.insert(context.clone(), attention);
        self.timestamps.insert(context, chrono::Utc::now().timestamp());
        self.clock.increment(&self.node_id);
    }

    /// Get attention for a context
    pub fn get(&self, context: &str) -> f32 {
        self.attention.get(context).copied().unwrap_or(0.0)
    }

    /// Merge with another attention map (element-wise max)
    pub fn merge_max(&mut self, other: &AttentionMapCrdt) {
        for (context, &attention) in &other.attention {
            let current = self.attention.entry(context.clone()).or_insert(0.0);
            *current = current.max(attention);
            
            // Update timestamp to latest
            let other_ts = other.timestamps.get(context).copied().unwrap_or(0);
            let current_ts = self.timestamps.entry(context.clone()).or_insert(0);
            *current_ts = (*current_ts).max(other_ts);
        }
        self.clock.merge(&other.clock);
    }

    /// Merge with another attention map (last-writer-wins per context)
    pub fn merge_lww(&mut self, other: &AttentionMapCrdt) {
        for (context, &attention) in &other.attention {
            let other_ts = other.timestamps.get(context).copied().unwrap_or(0);
            let current_ts = self.timestamps.get(context).copied().unwrap_or(0);
            
            if other_ts > current_ts {
                self.attention.insert(context.clone(), attention);
                self.timestamps.insert(context.clone(), other_ts);
            }
        }
        self.clock.merge(&other.clock);
    }

    /// Get all contexts with attention above threshold
    pub fn above_threshold(&self, threshold: f32) -> Vec<(&str, f32)> {
        self.attention
            .iter()
            .filter(|(_, &v)| v > threshold)
            .map(|(k, &v)| (k.as_str(), v))
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_vector_clock_ordering() {
        let mut clock_a = VectorClock::new();
        let mut clock_b = VectorClock::new();
        
        // Empty clocks: both happen-before each other (vacuously true)
        // So they're not concurrent by our definition
        
        // A advances
        clock_a.increment("node_a");
        clock_a.increment("node_a");
        
        // B advances on different node
        clock_b.increment("node_b");
        
        // They're concurrent (neither has seen the other's updates)
        // A doesn't happen-before B (A has node_a=2, B has node_a=0)
        // B doesn't happen-before A (B has node_b=1, A has node_b=0)
        assert!(!clock_a.happens_before(&clock_b));
        assert!(!clock_b.happens_before(&clock_a));
        
        // B merges A (B now has both node_a=2 and node_b=1)
        clock_b.merge(&clock_a);
        
        // Now A happens-before B (A's node_a=2 <= B's node_a=2)
        assert!(clock_a.happens_before(&clock_b));
        assert!(!clock_b.happens_before(&clock_a));
    }

    #[test]
    fn test_tensor_delta() {
        let from = VectorClock::with_node("node_a");
        let mut to = from.clone();
        to.increment("node_a");
        
        let mut delta = TensorDelta::new("node_a", from, to);
        
        delta.add_op(TensorOp::AttentionUpdate {
            context: "synapse".into(),
            attention: 0.8,
            strategy: AttentionMerge::Max,
        });
        
        assert_eq!(delta.len(), 1);
        assert!(!delta.is_empty());
    }

    #[test]
    fn test_attention_map_crdt() {
        let mut map_a = AttentionMapCrdt::new("device_a");
        let mut map_b = AttentionMapCrdt::new("device_b");
        
        // Concurrent updates
        map_a.update("context1", 0.5);
        map_b.update("context1", 0.8);
        map_b.update("context2", 0.3);
        
        // Merge with max
        map_a.merge_max(&map_b);
        
        assert_eq!(map_a.get("context1"), 0.8); // Max wins
        assert_eq!(map_a.get("context2"), 0.3); // New from B
    }

    #[test]
    fn test_tensor_element_max() {
        let a = Tensor::from_slice(&[1.0f32, 5.0, 3.0], &[3], &Device::Cpu).unwrap();
        let b = Tensor::from_slice(&[2.0f32, 1.0, 4.0], &[3], &Device::Cpu).unwrap();
        
        let result = tensor_element_max(&a, &b).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();
        
        assert_eq!(values, vec![2.0, 5.0, 4.0]);
    }

    #[test]
    fn test_tensor_blend() {
        let a = Tensor::from_slice(&[1.0f32, 1.0], &[2], &Device::Cpu).unwrap();
        let b = Tensor::from_slice(&[3.0f32, 3.0], &[2], &Device::Cpu).unwrap();
        
        let result = tensor_blend(&a, &b, 0.5).unwrap();
        let values: Vec<f32> = result.to_vec1().unwrap();
        
        // 0.5 * 1.0 + 0.5 * 3.0 = 2.0
        assert!((values[0] - 2.0).abs() < 0.01);
        assert!((values[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_crdt_state() {
        let mut state = CrdtState::new(42i32, "node_a", TensorMergeStrategy::LastWriterWins);
        
        assert_eq!(*state.get(), 42);
        assert_eq!(state.clock.get("node_a"), 0);
        
        state.update(100);
        
        assert_eq!(*state.get(), 100);
        assert_eq!(state.clock.get("node_a"), 1);
    }
}

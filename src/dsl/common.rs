//! Common types and utilities for DSL absorption.
//!
//! This module provides shared infrastructure for all DSL families:
//! - Error types
//! - Soft-ification utilities (IDs → embeddings, enums → one-hot)
//! - Tensor graph primitives

use crate::{Device, Tensor, TensorCoreError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// DSL-specific error type
#[derive(Debug, thiserror::Error)]
pub enum DslError {
    /// Spec text failed to parse.
    #[error("Parse error: {0}")]
    Parse(String),

    /// Parsed spec failed validation.
    #[error("Validation error: {0}")]
    Validation(String),

    /// Valid spec failed tensor-graph compilation.
    #[error("Compilation error: {0}")]
    Compilation(String),

    /// Compiled graph failed at runtime.
    #[error("Execution error: {0}")]
    Execution(String),

    /// YAML deserialization failure.
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// JSON deserialization failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Underlying tensor operation failure.
    #[error("Tensor error: {0}")]
    Tensor(#[from] candle_core::Error),
}

impl From<DslError> for TensorCoreError {
    fn from(e: DslError) -> Self {
        TensorCoreError::Compiler(e.to_string())
    }
}

impl From<TensorCoreError> for DslError {
    fn from(e: TensorCoreError) -> Self {
        DslError::Execution(e.to_string())
    }
}

/// Result type for DSL operations
pub type DslResult<T> = std::result::Result<T, DslError>;

// =============================================================================
// Soft-ification Utilities
// =============================================================================

/// Convert a string ID to a hash-based embedding.
///
/// This allows gradient flow through ID-based lookups by converting
/// discrete IDs to continuous embeddings.
///
/// # Arguments
/// * `id` - The string ID to embed
/// * `dim` - Embedding dimension
/// * `device` - Tensor device
///
/// # Example
/// ```ignore
/// let embedding = soft_id("step_load_data", 64, &device)?;
/// ```
pub fn soft_id(id: &str, dim: usize, device: &Device) -> DslResult<Tensor> {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(id.as_bytes());
    let hash = hasher.finalize();

    // Use hash bytes to seed embedding
    let mut embedding = vec![0.0f32; dim];
    for (i, &byte) in hash.iter().take(dim).enumerate() {
        // Normalize to [-1, 1]
        embedding[i] = (byte as f32 / 128.0) - 1.0;
    }

    // L2 normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }

    Ok(Tensor::from_vec(embedding, dim, device)?)
}

/// Convert a one-hot index to a tensor.
///
/// # Arguments
/// * `index` - The active index
/// * `num_classes` - Total number of classes
/// * `device` - Tensor device
pub fn one_hot(index: usize, num_classes: usize, device: &Device) -> DslResult<Tensor> {
    let mut vec = vec![0.0f32; num_classes];
    if index < num_classes {
        vec[index] = 1.0;
    }
    Ok(Tensor::from_vec(vec, num_classes, device)?)
}

/// Soft gate for conditional execution.
///
/// Converts a hard boolean condition to a differentiable gate.
/// Uses sigmoid for smooth gradient flow.
///
/// # Arguments
/// * `condition` - Tensor of condition scores
/// * `sharpness` - How "hard" the gate is (higher = more binary)
pub fn soft_gate(condition: &Tensor, sharpness: f32) -> DslResult<Tensor> {
    // sigmoid(condition * sharpness)
    let scaled = (condition * sharpness as f64)?;
    Ok(crate::primitives::sigmoid(&scaled)?)
}

// =============================================================================
// Tensor Graph Primitives
// =============================================================================

/// A node in a tensor execution graph.
#[derive(Debug, Clone)]
pub struct TensorNode {
    /// Unique node ID
    pub id: String,
    /// Node embedding (for soft lookups)
    pub embedding: Option<Tensor>,
    /// Operation to execute
    pub op: TensorOp,
    /// Input dependencies (node IDs)
    pub dependencies: Vec<String>,
    /// Whether this node can run in parallel with siblings
    pub parallelizable: bool,
    /// Soft gate for conditional execution (optional)
    pub gate: Option<Tensor>,
}

/// Operations that can be executed in a tensor graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TensorOp {
    /// Load a tensor from namespace path
    TensorLoad {
        /// Namespace path to load.
        path: String,
    },
    /// Apply a compiled rule
    RuleForward {
        /// Name of the rule to run.
        rule_name: String,
    },
    /// Custom tensor operation
    Custom {
        /// Operation name.
        name: String,
        /// String parameters by key.
        params: HashMap<String, String>,
    },
    /// No-op (for structural nodes)
    Identity,
}

/// Compensation mask for rollback operations.
#[derive(Debug, Clone)]
pub struct CompensationMask {
    /// Which nodes to rollback (1.0 = rollback, 0.0 = keep)
    pub mask: Tensor,
    /// Node IDs in mask order
    pub node_ids: Vec<String>,
}

impl CompensationMask {
    /// Create a "rollback all" mask
    pub fn rollback_all(node_ids: Vec<String>, device: &Device) -> DslResult<Self> {
        let n = node_ids.len();
        let mask = Tensor::ones(n, candle_core::DType::F32, device)?;
        Ok(Self { mask, node_ids })
    }

    /// Create a "rollback dependents" mask from failed node
    pub fn rollback_dependents(
        node_ids: Vec<String>,
        failed_idx: usize,
        adjacency: &Tensor,
        device: &Device,
    ) -> DslResult<Self> {
        // Compute reachability from failed node
        // mask[i] = 1.0 if node i depends on failed node
        let n = node_ids.len();
        let mut reachable = vec![0.0f32; n];
        reachable[failed_idx] = 1.0;

        // BFS through adjacency matrix
        let adj_vec: Vec<f32> = adjacency.flatten_all()?.to_vec1()?;
        for _ in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if adj_vec[i * n + j] > 0.5 && reachable[i] > 0.5 {
                        reachable[j] = 1.0;
                    }
                }
            }
        }

        let mask = Tensor::from_vec(reachable, n, device)?;
        Ok(Self { mask, node_ids })
    }
}

// =============================================================================
// Namespace Helpers
// =============================================================================

/// DSL namespace roots
pub mod namespaces {
    /// Workflow DSL namespace root
    pub const WORKFLOWS: &str = "an-tensor-compiler.workflows";
    /// Tool patterns namespace root
    pub const TOOLS: &str = "an-tensor-compiler.tools.patterns";
    /// Context namespace root
    pub const CONTEXT: &str = "an-tensor-compiler.context";
}

/// Build a full namespace path for a DSL artifact
pub fn dsl_namespace_path(root: &str, artifact_type: &str, id: &str) -> String {
    format!("{}.{}.{}", root, artifact_type, id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_id_deterministic() {
        let device = Device::Cpu;
        let e1 = soft_id("test_id", 32, &device).unwrap();
        let e2 = soft_id("test_id", 32, &device).unwrap();

        let v1: Vec<f32> = e1.to_vec1().unwrap();
        let v2: Vec<f32> = e2.to_vec1().unwrap();

        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_soft_id_different_ids() {
        let device = Device::Cpu;
        let e1 = soft_id("id_a", 32, &device).unwrap();
        let e2 = soft_id("id_b", 32, &device).unwrap();

        let v1: Vec<f32> = e1.to_vec1().unwrap();
        let v2: Vec<f32> = e2.to_vec1().unwrap();

        // Should be different
        let same = v1.iter().zip(v2.iter()).all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(!same);
    }

    #[test]
    fn test_one_hot() {
        let device = Device::Cpu;
        let oh = one_hot(2, 5, &device).unwrap();
        let vec: Vec<f32> = oh.to_vec1().unwrap();
        assert_eq!(vec, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_compensation_mask_rollback_all() {
        let device = Device::Cpu;
        let nodes = vec!["a".into(), "b".into(), "c".into()];
        let mask = CompensationMask::rollback_all(nodes.clone(), &device).unwrap();

        let vec: Vec<f32> = mask.mask.to_vec1().unwrap();
        assert_eq!(vec, vec![1.0, 1.0, 1.0]);
        assert_eq!(mask.node_ids, nodes);
    }

    #[test]
    fn test_namespace_path() {
        let path = dsl_namespace_path(namespaces::WORKFLOWS, "compiled", "wf_regime_fusion");
        assert_eq!(
            path,
            "an-tensor-compiler.workflows.compiled.wf_regime_fusion"
        );
    }
}

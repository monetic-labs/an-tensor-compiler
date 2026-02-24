//! Tensor Logic Primitives
//!
//! Core tensor operations and differentiable fuzzy logic operators.
//!
//! ## Thread Safety
//!
//! Metal GPU command buffers are **not thread-safe**. When using parallel workloads
//! (e.g., with rayon), use the thread-safe device selection options:
//!
//! - Set `AN_TENSOR_NO_GPU=1` to force CPU mode
//! - Use [`thread_local_device`] for per-thread devices
//! - Use [`with_gpu_sync`] to serialize GPU access
//!
//! ## Submodules
//!
//! - Fuzzy logic operators (AND, OR, NOT, IMPLIES)
//! - Core tensor operations (matmul, cosine similarity)
//! - Activation functions (sigmoid, softmax, relu)
//! - [`gnn`]: Graph neural network layers (GraphSAGE, GCN, GAT)

mod activations;
mod fuzzy_ops;
pub mod gnn;
mod tensor_ops;

// Re-export all primitives at module level
pub use activations::*;
pub use fuzzy_ops::*;
pub use tensor_ops::*;

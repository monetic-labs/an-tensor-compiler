//! Tensor Logic Primitives
//!
//! Core tensor operations and differentiable fuzzy logic operators.
//!
//! ## Thread Safety
//!
//! Metal GPU command buffers are **not thread-safe**. When using parallel workloads
//! (e.g., with rayon), see [`tensor_ops`] for thread-safe device selection options:
//!
//! - Set `AN_TENSOR_NO_GPU=1` to force CPU mode
//! - Use [`thread_local_device`] for per-thread devices
//! - Use [`with_gpu_sync`] to serialize GPU access
//!
//! ## Submodules
//!
//! - [`fuzzy_ops`]: Fuzzy logic operators (AND, OR, NOT, IMPLIES)
//! - [`tensor_ops`]: Core tensor operations (matmul, cosine similarity)
//! - [`activations`]: Activation functions (sigmoid, softmax, relu)

mod fuzzy_ops;
mod tensor_ops;
mod activations;

// Re-export all primitives at module level
pub use fuzzy_ops::*;
pub use tensor_ops::*;
pub use activations::*;




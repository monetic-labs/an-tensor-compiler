//! # Holographic Reduced Representations
//!
//! Tensor-based distributed representations using holographic reduced
//! representations (HRR). Every piece of a hologram contains information
//! about the whole — this isn't metaphor, it's the mathematical property
//! that makes compositional distributed memory work.
//!
//! ## Key Concepts
//!
//! - **Bind (⊗)**: Circular convolution — combines two concepts into one
//! - **Unbind (⊙)**: Circular correlation — extracts a concept from a hologram
//! - **Superimpose (+)**: Weighted sum — creates a hologram from multiple concepts
//! - **Hierarchical Composition**: Tensors compose upward (component → module → context → organism)
//!
//! ## Architecture
//!
//! ```text
//! Component → Module → BoundedContext → Organism
//!     ↓          ↓           ↓              ↓
//!   hologram   hologram    hologram       hologram
//!     ↑          ↑           ↑              ↑
//!   bind()    superimpose()  ...          superimpose()
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use an_tensor_compiler::holographic::*;
//!
//! // Bind two concepts
//! let semantic = embed("struct Foo")?;
//! let structural = embed("pub fields: Vec<Bar>")?;
//! let bound = bind(&semantic, &structural)?;
//!
//! // Recover one concept from the binding
//! let recovered = unbind(&bound, &semantic)?;
//! // recovered ≈ structural
//!
//! // Compose many concepts into a hologram
//! let hologram = superimpose(&[comp1, comp2, comp3], None)?;
//!
//! // Project a query to find relevant context
//! let result = organism.project(&query_tensor)?;
//! ```
//!
//! ## Reference
//!
//! Tony Plate, "Holographic Reduced Representations" (1995)

pub mod manifest;
pub mod ops;
pub mod storage;
pub mod types;

// Core operations
pub use ops::{
    // FFT-based (mathematically correct, Plate 1995)
    bind,
    // Fast approximation (~3x faster, lower fidelity — pair together)
    bind_fast,
    cosine_similarity,
    normalize,
    position_encoding,
    project,
    role_encoding,
    // Composition and utilities
    superimpose,
    unbind,
    unbind_fast,
};

// Type hierarchy
pub use types::{
    BoundaryRules, BoundaryViolation, BoundedContextTensor, CoherenceResult, ComponentId,
    ComponentKind, ComponentTensor, ContextRole, FederationBinding, HistoryTensor, ModuleTensor,
    OrganismTensor, ProjectionResult,
};

// Manifest for project structure
pub use manifest::{ContextInfo, ContextManifest, EncodingInfo, OrganismInfo};

// Storage and persistence
pub use storage::{CacheEntry, CacheStats, ComponentCache, IncrementalState, TensorStore};

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-10

### Added

- **Compiler**: Rule DSL parser supporting Prolog-style (`:-`), Unicode (`←`, `∧`, `¬`), facts, disjunction, and comments
- **Compiler**: 8 predicate types — Threshold, Learned, LearnedProjection, LearnedSimilarity, GraphNeural, TierLookup, PairwiseDifference, CascadingLookup
- **Compiler**: FiLM (Feature-wise Linear Modulation) conditioning with identity initialization
- **Compiler**: Multi-head compiled rules for rule sets with multiple head predicates
- **Compiler**: High-performance inference contexts — `InferenceContext` (single) and `BatchedInferenceContext` (batched)
- **Compiler**: `forward_fast()` for ~20x faster inference when explanations are not needed
- **Compiler**: Save/load learned parameters via safetensors format
- **Compiler**: Rule validation (arity consistency, reserved words, variable binding)
- **Primitives**: Differentiable fuzzy logic operators (AND, OR, NOT, IMPLIES) using product t-norm
- **Primitives**: Multi-way AND/OR, soft/hard thresholds, weighted rule combination
- **Primitives**: Asymmetric MSE loss for reconciliation learning
- **Primitives**: Activation functions — sigmoid, softmax, relu, leaky_relu, tanh, gelu
- **Primitives**: Thread-safe GPU device management with `AN_TENSOR_NO_GPU`, `thread_local_device()`, `with_gpu_sync()`
- **Namespace**: Gradient isolation between domains via `NamespaceId` and `NamespacedTensor`
- **Namespace**: Hierarchical namespace paths with CRDT key conversion
- **Namespace**: Namespace metadata for discovery and schema compatibility
- **Federation**: Cross-domain parameter merge algorithms — weighted average, FedAvg, coordinate-wise median, trimmed mean, accuracy-weighted
- **Federation**: `SyncTransport` trait for pluggable transport implementations
- **Federation**: `FederationManager` for coordinating cross-namespace learning
- **Holographic**: Holographic reduced representations (HRR) — bind, unbind, superimpose, project
- **Holographic**: Position and role encodings for compositional distributed memory
- **Holographic**: Organism/context/module/component tensor hierarchy
- **Holographic**: Context manifest parsing from TOML
- **Holographic**: Tensor storage with safetensors persistence and component caching
- **Holographic**: Incremental encoding via git change detection
- **CRDT**: Conflict-free tensor sync primitives — vector clocks, deltas, merge strategies
- **CRDT**: Attention map CRDT with max and last-writer-wins merge
- **Training**: AdamW and SGD optimizer wrappers
- **Training**: `GroupedOptimizer` for per-group learning rates (e.g., FiLM vs main network)
- **Training**: `safe_optimizer_step()` with NaN detection and gradient clipping
- **Training**: Gradient health monitoring and convergence detection
- **Training**: `TrainingOutcome` enum for supervised feedback signals
- **GPU**: Metal (Apple Silicon) and CUDA feature flags

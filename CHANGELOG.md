# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-15

Major sync with eight months of `an-tensor-core` development: test suite grows
from 225 to 597 lib tests, and the crate gains six new modules.

### Added

- **Governance core** (new `governance` module): auto-derived L0 structural
  properties (`derive_l0`), in-house boolean-skeleton checker (`check_property`)
  plus SMT-LIB emission (`emit_smtlib`), closed-form margin propagation
  (`propagate_margins` with honest unresolvable-at-margin verdicts), output
  sort system (`sort_check`: `Truth` vs `Score` vs `Index` vs `Count`),
  canonical bytes, and an ECE calibrator. See `examples/governance_demo.rs`.
- **Policy packs** (new `interaction`, `counterops`, `compliance` modules):
  pre-built predicate sets with single-head and multi-head rule templates for
  chat safety/intent/quality, threat/anomaly/defense, and PII/GDPR/HIPAA/SOX
  compliance. See `examples/policy_packs.rs`.
- **Workflow DSL** (new `dsl` + `scheduling` modules): compile agent-style
  workflow DAGs to tensor execution (`TensorWorkflowExecutor` with batching
  and compensation), tool-pattern matching (`TensorPatternMatcher`), context
  management (`TensorContextManager`), and DAG utilities (`ready_set`,
  `topological_levels`, `critical_path`, `makespan_with_concurrency`).
- **HRR v2** (`holographic`): neutral knowledge-space primitives
  (`KnowledgeSpaceTensor`, `EntityTensor`, `RelationTensor`), explicit
  `BridgeBinding` cross-space projection with `ProjectionPolicy`,
  embedder-agnostic query protocol (`HrrEmbedder`, `HrrQuery`, `HrrQueryResult`
  with projection diagnostics), `EmbedderProvenance` (content hash + source
  URI geometry verification), multi-resolution holograms with confidence
  escalation, geometric index for sublinear retrieval, labeled-record
  encoding (`bundle_record` / `unbind_role`), and full memory systems
  (`ExperienceHologram`, `PatternLibrary`, `WorkingMemory`,
  entropy-gated/ regime-partitioned/multi-scale memory, `SynapseMemory`).
  See `examples/hrr_memory.rs`.
- **Math toolkit** (new `information`, `topology` modules): streaming
  entropy/MI/drift/novelty, Monte Carlo statistics (bootstrap, permutation
  tests, importance sampling, Metropolis-Hastings, Gibbs, jackknife), and
  Lyapunov-exponent regime detection (`LyapunovComputer`, `ChaosRegime`).
- **Compiler**: `predicate_builders` for canonical declarative invariants
  (non-negativity, ranges, tolerance bands, monotone history, allowed-set
  distance); sparse-COO adjacency detection for graph predicates.
- **Training**: `LearningConfig`, `MetricsTracker`, `Optimizer` training
  telemetry types.

### Changed

- **Checkpoint format (migration note)**: `save_with_metadata` now embeds
  metadata in the safetensors `__metadata__` header instead of writing a
  sidecar `.meta.json` file, and `load_metadata` reads it back from the
  header. Re-save v0.1.0 checkpoints with v0.2.0 to migrate; the old sidecar
  files are not read. Saving also now stages tensors through CPU, fixing
  saves of GPU-resident parameters.
- Rule-body evaluation flattens predicate activations before `fuzzy_and`,
  so mixed-rank predicate outputs (e.g. `Threshold` vs `LearnedProjection`)
  conjunction-cleanly instead of failing on shape mismatch.
- `codegen` consolidated from three files to one; no public API change.
- New `TensorCoreError` variants: `Validation`, `Projection`, `Governance`,
  `Information`.

### Fixed

- `graph_neural` and record-encoding test failures; N=64 averaging for the
  `bind_fast` recovery property test; checksum serialization errors in
  holographic storage are no longer swallowed.

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

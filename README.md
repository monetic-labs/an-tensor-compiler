# an-tensor-compiler

[![Crates.io](https://img.shields.io/crates/v/an-tensor-compiler.svg)](https://crates.io/crates/an-tensor-compiler)
[![Documentation](https://docs.rs/an-tensor-compiler/badge.svg)](https://docs.rs/an-tensor-compiler)
[![CI](https://github.com/monetic-labs/an-tensor-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/monetic-labs/an-tensor-compiler/actions)
[![License](https://img.shields.io/crates/l/an-tensor-compiler.svg)](LICENSE)

A differentiable neuro-symbolic compiler. Write symbolic logic rules, get tensor equations with full gradient flow.

Rules become functions. Functions learn from data. Data improves rules. The loop closes.

## What This Does

You write rules in a Prolog-like DSL:

```prolog
escalate(X) :- high_risk(X), not approved(X).
escalate(X) :- critical_resource(X), low_confidence(X).
```

The compiler transforms them into differentiable tensor operations:

```text
rule₁ = fuzzy_and(high_risk, fuzzy_not(approved))     — product t-norm
rule₂ = fuzzy_and(critical_resource, low_confidence)   — product t-norm
output = fuzzy_or(rule₁, rule₂)                        — probabilistic sum
```

Every operation has gradients. Thresholds, projections, similarity functions, graph structures — all trainable via standard backpropagation. The symbolic structure gives you interpretability. The tensor substrate gives you learnability. You don't sacrifice either.

## Quick Start

```rust
use an_tensor_compiler::prelude::*;
use std::collections::HashMap;

// Parse rules
let spec = RuleSpec::parse("policy", r#"
    escalate(X) :- high_risk(X), not approved(X).
    escalate(X) :- critical_resource(X), low_confidence(X).
"#)?;

// Compile to differentiable function
let compiled = CompiledRule::compile(spec)?;

// Evaluate with fuzzy inputs (values in [0, 1])
let device = best_device();
let mut inputs = HashMap::new();
inputs.insert("high_risk".into(), Tensor::from_vec(vec![0.9f32], 1, &device)?);
inputs.insert("approved".into(), Tensor::from_vec(vec![0.2f32], 1, &device)?);
inputs.insert("critical_resource".into(), Tensor::from_vec(vec![0.5f32], 1, &device)?);
inputs.insert("low_confidence".into(), Tensor::from_vec(vec![0.7f32], 1, &device)?);

let output = compiled.forward(&inputs)?;
println!("Score: {:.3}", output.output.to_vec1::<f32>()?[0]);
println!("Why: {}", output.explanation);
```

See [`examples/`](examples/) for complete runnable demos: basic rules, learned projections, multi-head policies, training loops, governance checks, holographic memory, and policy packs.

## Installation

```toml
[dependencies]
an-tensor-compiler = "0.2"
```

### GPU Acceleration

```toml
# Apple Silicon (M1/M2/M3/M4)
an-tensor-compiler = { version = "0.2", features = ["metal"] }

# NVIDIA GPU
an-tensor-compiler = { version = "0.2", features = ["cuda"] }
```

### Requirements

- **Rust** 1.90+ (edition 2021)
- **CPU**: Works out of the box, no additional dependencies
- **Metal**: macOS with Apple Silicon, Xcode command line tools
- **CUDA**: NVIDIA GPU with CUDA toolkit installed
- **Optional**: `git` on PATH (for incremental holographic encoding)

## Why GPUs Change the Design

When your logic is differentiable tensors and your state lives in conflict-free replicated data structures, the computational model is fundamentally different from traditional software.

In a conventional system, logic is branches and state is rows in a database. Parallelism means careful locking. Scaling means sharding.

In this model, logic is matrix multiplication and state is tensor merge operations. Your "if-then" rules are literally `matmul → activation → multiply`. Your "database" is a CRDT that converges without coordination. This means:

**Inference is GPU-native.** A compiled rule evaluating 100 inputs isn't a loop — it's a single batched `matmul`. The GPU doesn't just make it faster; it makes the architecture viable for real-time workloads where you need thousands of decisions per second with full gradient tracking.

**Training is backpropagation.** Every threshold, every projection weight, every similarity function is a differentiable parameter. When you get feedback ("that escalation was unnecessary"), the gradient flows back through the entire rule chain and adjusts every learned parameter. On GPU, this takes milliseconds instead of seconds.

**Concurrency is a different problem.** GPU command buffers are not thread-safe in the way CPU memory is. When multiple threads need tensor operations, you face a choice the compiler makes explicit:

```rust
// Option 1: Force CPU for parallel read-heavy workloads
// Safe, simple, good for fan-out evaluation
std::env::set_var("AN_TENSOR_NO_GPU", "1");

// Option 2: Thread-local devices
// Each thread gets its own GPU context — avoids Metal command buffer conflicts
let device = thread_local_device();

// Option 3: Serialized GPU access
// When you need GPU acceleration but have concurrent callers
with_gpu_sync(|| {
    // GPU operations are serialized here
    compiled.forward_fast(&inputs)
});
```

The right choice depends on your workload shape. Read-heavy fan-out (evaluate rules across many contexts) favors CPU parallelism. Compute-heavy inference (large learned projections, batched evaluation) favors GPU with explicit synchronization. Batch everything if you can — `BatchedInferenceContext` processes hundreds of evaluations in a single GPU call.

## Architecture

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│                              an-tensor-compiler                             │
├──────────┬──────────┬──────────┬───────────┬───────────┬────────┬───────────┤
│ compiler │primitives│ namespace│ federation│holographic│  crdt  │governance │
│          │          │          │           │           │        │           │
│ DSL →    │ fuzzy    │ gradient │ cross-    │ HRR       │conflict│ L0 checks │
│ tensor   │ logic    │ isolation│ domain    │ memory +  │-free   │ margins + │
│ functions│ ops      │ boundaries│ merge    │ knowledge │ sync   │ sorts     │
│          │          │          │           │ spaces    │        │           │
├──────────┴──────────┴──────────┴───────────┴───────────┴────────┴───────────┤
│ interaction │ counterops │ compliance │ dsl │ information │ topology │sched. │
│             │            │            │     │             │          │       │
│ chat safety │ threat +   │ PII/GDPR/  │work-│ streaming   │ Lyapunov │ DAG   │
│ intent +    │ anomaly +  │ HIPAA/SOX  │flow │ entropy +   │ regimes  │ topo  │
│ quality     │ defense    │ packs      │ DSL │ MC stats    │          │ utils │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Compiler

Parse → validate → compile → execute. The rule DSL supports Prolog-style (`:-`), Unicode (`←`, `∧`, `¬`), facts, disjunction, and comments. Eight predicate types cover the spectrum from simple thresholds to graph neural networks:

| Predicate Type | What It Does | Learnable Parameters |
|---|---|---|
| `Threshold` | Soft comparison (x > θ) | Threshold value, sharpness |
| `Learned` | Linear → sigmoid | Weight matrix, bias |
| `LearnedProjection` | MLP with optional attention, LayerNorm, FiLM | Full network weights |
| `LearnedSimilarity` | Compare two embeddings | Similarity weights |
| `GraphNeural` | Message passing over graph structure | GNN layer weights |
| `TierLookup` | Multi-threshold step function | Tier boundaries |
| `PairwiseDifference` | Detect gaps between items | Threshold percentage |
| `CascadingLookup` | Priority-ordered fallback chain | Priority weights |
| `Composite` | Named AND/OR/NOT over children (reserved; express as rules for now) | None |
| `External` | Caller-supplied truth value | None (provided at runtime) |

Think of it as: Prolog gives you the structure. Tensors give you the substrate. Gradients close the loop.

The `predicate_builders` module provides canonical constructors for common declarative invariants (non-negativity, ranges, tolerance bands, monotone history, allowed-set distance) as single `Threshold` specs that compose cleanly under `fuzzy_and`.

### Primitives

Differentiable fuzzy logic operators form the computational foundation. The product t-norm (`AND(a,b) = a × b`) and probabilistic sum (`OR(a,b) = a + b - ab`) were chosen because gradients are non-zero everywhere — unlike min/max, which have zero gradients at the boundary. This is what makes the rules actually learnable.

### Namespace

Gradient isolation between domains. Tensors in different namespaces cannot be combined without explicit federation — this prevents gradient pollution where learning in one domain silently corrupts parameters in another. Think of it as bounded contexts for tensor operations.

### Federation

When isolated domains need to share what they've learned, federation provides the merge. Five algorithms are included: weighted average, FedAvg (McMahan et al. 2017), coordinate-wise median, trimmed mean, and accuracy-weighted averaging. The `SyncTransport` trait abstracts the network layer — implement it for gRPC, Redis, or whatever your infrastructure uses.

### Holographic Memory

This is where things get interesting.

Holographic Reduced Representations (Plate, 1995) give you a mathematical property that biological memory has: every piece of a hologram contains information about the whole. When you `bind` two concepts via circular convolution, the result is a single vector the same size as the inputs. When you `superimpose` many bound pairs, you get a fixed-size hologram that encodes all of them. When you `project` a query against that hologram, you recover the relevant concepts.

```rust
use an_tensor_compiler::holographic::*;

// Bind concepts into a hologram
let bound = bind(&semantic_embedding, &structural_embedding)?;

// Superimpose many concepts into a fixed-size representation
let hologram = superimpose(&[comp1, comp2, comp3, comp4], None)?;

// Query the hologram — recovers relevant concepts
let result = project(&hologram, &query)?;
```

This isn't an embedding database. There's no index, no retrieval pipeline, no nearest-neighbor search. The hologram *is* the memory, and projection *is* the query — both are tensor operations with gradients. The entire hierarchy (component → module → context → organism) composes upward through superposition.

The practical consequence: you can represent an entire codebase, knowledge graph, or document corpus as a single tensor, and query it with a matrix multiply.

Beyond the core ops, the module now includes HRR v2: neutral `KnowledgeSpaceTensor` / `EntityTensor` / `RelationTensor` primitives, explicit `BridgeBinding` cross-space projection with policy control, the embedder-agnostic `HrrEmbedder` + `HrrQuery` query protocol, `EmbedderProvenance` (content hash + source URI, so a query can verify it is projecting into the geometry the corpus was compiled against), multi-resolution holograms (fast L1 / deep L2 with confidence escalation), a geometric index for sublinear retrieval, and full memory systems — `ExperienceHologram`, `PatternLibrary` (failure signatures + recovery rates), `WorkingMemory` (salience decay), entropy-gated consolidation, Lyapunov regime partitioning, and multi-scale horizons — unified behind `SynapseMemory` with a caller-supplied embedder.

### CRDTs

Conflict-free replicated data types for tensor synchronization across devices. When multiple nodes update the same tensor concurrently, CRDTs guarantee convergence without coordination — no locks, no consensus protocol, no conflict resolution logic. You get eventual consistency by mathematical construction.

```rust
use an_tensor_compiler::crdt::*;

// Each device maintains its own state with a vector clock
let mut device_a = AttentionMapCrdt::new("device_a");
let mut device_b = AttentionMapCrdt::new("device_b");

// Concurrent updates — no coordination needed
device_a.update("context_1", 0.8);
device_b.update("context_1", 0.6);
device_b.update("context_2", 0.9);

// Merge deterministically (order doesn't matter)
device_a.merge_max(&device_b);
// context_1 = 0.8 (max wins), context_2 = 0.9 (new from b)
```

The merge strategies are tensor-aware: `ElementMax` for attention maps (preserve strongest signal), `Superimpose` for holographic memory (concepts stack), `WeightedAverage` with decay for evolving state. Vector clocks track causality so you know which updates have been seen.

This matters because the systems you build with differentiable logic are inherently distributed. Models train on different data. Devices see different contexts. Federation merges learned parameters. CRDTs are the substrate that makes all of this converge without a coordinator.

### Governance

Static verification for your rules, before you train them. Given a `RuleSpec`, the governance core auto-derives L0 structural properties (body satisfiability, head coverage/non-triviality, exclusion sets, trigger presence, definedness) and checks them over the boolean skeleton with the in-house F0 checker — or emits SMT-LIB for external solvers. A closed-form margin calculus propagates leaf margins through the rule DAG; where the output margin goes non-positive, the head is honestly marked *unresolvable-at-margin* instead of silently flipping (product-AND does not commute with thresholding — five leaves at 0.8 give 0.33). A sort system (`Truth` vs `Score` vs `Index` vs `Count`) keeps counts and magnitudes from silently flowing into probabilistic combinators. See `examples/governance_demo.rs`.

### Policy Packs

Batteries-included predicate sets with ready-made rules: `interaction` (chat safety, intent clarity, context freshness, coherence, topic drift, escalation, frustration), `counterops` (threat level, anomaly scoring, injection detection, exfiltration risk, defense mode, auth anomalies), and `compliance` (PII detection, GDPR/HIPAA/SOX configurations, retention, breach detection, audit reporting). Each pack ships single-head and multi-head rule templates. See `examples/policy_packs.rs`.

### Workflow DSL

The `dsl` module compiles agent-style workflow DAGs, tool patterns, and context graphs into tensor execution: `TensorWorkflowExecutor` (compile DAG, batched execute, compensate), `TensorPatternMatcher` (embed patterns, similarity retrieval, strategy routing), and `TensorContextManager` (temporal decay, activation propagation, promotion gates) — plus a `scheduling` module with DAG readiness, topological levels, critical path, and concurrency-aware makespan.

### Math Toolkit

`information` (streaming entropy/MI/drift/novelty, Monte Carlo statistics: bootstrap, permutation tests, importance sampling, MCMC, jackknife) and `topology` (Lyapunov-exponent regime detection) support the memory systems above and are useful standalone.

## Training

Compiled rules are trainable via standard gradient descent:

```rust
use candle_nn::optim::{AdamW, ParamsAdamW};

let compiled = CompiledRule::compile(spec)?;
let vars = compiled.trainable_vars();
let params = ParamsAdamW { lr: 0.01, ..Default::default() };
let mut optimizer = AdamW::new(vars.clone(), params)
    .map_err(|e| TensorCoreError::Tensor(format!("optimizer: {e}")))?;

for batch in data {
    let output = compiled.forward_fast(&batch.inputs)?;
    let loss = binary_cross_entropy(&output, &batch.target)?;
    let grads = loss
        .backward()
        .map_err(|e| TensorCoreError::Tensor(format!("backward: {e}")))?;

    // Safe step with gradient clipping (recommended for attention)
    safe_optimizer_step(&mut optimizer, &grads, &vars, 1.0, 0.01)?;
    compiled.clamp_attention_weights(10.0)?;
}

// Save learned parameters
compiled.save("model.safetensors")?;
```

See `examples/training_loop.rs` for the complete runnable version.

FiLM conditioning (Feature-wise Linear Modulation) lets a conditioning vector modulate hidden activations — useful when the same rules should behave differently based on context. Separate learning rate groups (`GroupedOptimizer`) prevent FiLM layers from being overwhelmed by the main network's gradients.

## Feature Flags

| Flag | Effect |
|---|---|
| *(default)* | CPU-only, no additional dependencies |
| `metal` | Apple Metal GPU acceleration (M1/M2/M3/M4) |
| `cuda` | NVIDIA CUDA GPU acceleration |

## Intellectual Lineage

The core ideas draw from three research threads:

- **Pedro Domingos** — Unifying logical and statistical AI. The insight that symbolic rules can be compiled into differentiable tensor operations without losing interpretability or learnability — from kernel machines (arXiv:2012.00152) to the full logic↔tensor isomorphism (*"Tensor Logic"*, arXiv:2510.12269), which this crate implements.
- **Lotfi Zadeh** — Fuzzy logic (1965). Continuous truth values in \[0, 1\] instead of {0, 1}, enabling gradient flow through logical operations.
- **Tony Plate** — Holographic Reduced Representations (1995). Circular convolution binding for distributed compositional memory where every piece contains information about the whole.

See [PHILOSOPHY.md](PHILOSOPHY.md) for the full discussion.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests are welcome. The compiler is designed with clear extension points — new predicate types, new similarity methods, new aggregation strategies, new merge algorithms.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

This is strong copyleft: if you build on this crate and distribute the result,
you must release your complete source under the same terms. Corporations and
proprietary products are welcome to use it — on those terms.

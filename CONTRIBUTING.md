# Contributing to an-tensor-compiler

Pull requests are welcome. We review, we decide. If your change reveals something we couldn't see, it gets in.

## Getting Started

### Build

```bash
# Default (CPU only)
cargo build

# With Apple Metal GPU support
cargo build --features metal

# With NVIDIA CUDA GPU support
cargo build --features cuda
```

### Test

```bash
cargo test
```

### Lint

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
```

## Architecture

The crate is organized into 7 modules:

| Module | Purpose |
|---|---|
| `compiler/` | Rule DSL → differentiable tensor functions |
| `primitives/` | Fuzzy logic operators, activations, device management |
| `namespace/` | Gradient isolation between domains |
| `federation/` | Cross-domain parameter merging |
| `holographic/` | Holographic reduced representations for distributed memory |
| `crdt/` | Conflict-free tensor synchronization |
| `training/` | Optimizers, gradient utilities, metrics |

### Key Extension Points

**Adding a new predicate type:**

1. Add a variant to `PredicateSpec` in `src/compiler/mod.rs`
2. Add compilation logic in `src/compiler/codegen.rs` (in the `compile_predicate` match)
3. Add forward pass logic in the `forward_predicate` match
4. Add tests

**Adding a new merge algorithm:**

1. Add the function in `src/federation/algorithms.rs`
2. Re-export in `src/federation/mod.rs`
3. Optionally add a `MergeStrategy` variant
4. Add tests

**Adding a new activation function:**

1. Add the function in `src/primitives/activations.rs`
2. Add a variant to `Activation` in `src/compiler/mod.rs`
3. Handle it in `Activation::apply()`
4. Re-export in the prelude if it's commonly used
5. Add tests

## Code Style

- Run `cargo fmt` before committing
- All public items should have doc comments (`///`)
- Tests live in `#[cfg(test)] mod tests` at the bottom of each file
- Error messages should include the operation that failed (e.g., `"fuzzy_and failed: ..."`)

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure `cargo test`, `cargo fmt --check`, and `cargo clippy` all pass
5. Open a PR with a clear description of what and why

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

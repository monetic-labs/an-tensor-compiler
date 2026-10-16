//! # Streaming Information Theory
//!
//! Online information-theoretic primitives over observation streams: entropy,
//! mutual information, drift detection, and novelty scoring. Used by holographic
//! memory for entropy-gated consolidation (only store surprising experiences).
//!
//! Ported from `an-tensor-core` v0.5.0 (`information::streaming`).

pub mod monte_carlo;
pub mod streaming;

pub use monte_carlo::{
    bootstrap, bootstrap_paired, effective_sample_size, gibbs_sampler, importance_sampling,
    importance_sampling_self_normalized, jackknife, metropolis_hastings, monte_carlo_se,
    permutation_test, permutation_test_mi, BootstrapResult, ImportanceSamplingResult, MCMCResult,
    PermutationTestResult, SimpleRng,
};
pub use streaming::{
    DriftStatus, StreamingDrift, StreamingEntropy, StreamingMI, StreamingNovelty, StreamingStats,
};

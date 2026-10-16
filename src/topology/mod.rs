//! # Dynamical Topology
//!
//! Lyapunov-exponent regime detection for observation streams: classify dynamics
//! as stable, random-walk, or chaotic so memory and policy can adapt per regime.
//! Used by holographic memory for regime-partitioned storage and recall.
//!
//! Ported from `an-tensor-compiler` v0.5.0 (`topology::lyapunov`).

pub mod lyapunov;

pub use lyapunov::{
    ChaosRegime, HybridChaosFeatures, LyapunovComputer, LyapunovConfig, LyapunovFeatures,
    LyapunovMetadata,
};

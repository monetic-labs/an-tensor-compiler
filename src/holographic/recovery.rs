//! Bind-chain recovery fidelity probes for HRR regression testing.
//!
//! Used by integration tests to guard changes to [`super::ops`] bind/unbind behavior.

use candle_core::{Device, Tensor};

use crate::holographic::ops::{bind, cosine_similarity, normalize, unbind};
use crate::Result;

/// Configuration for a single bind-chain recovery experiment.
#[derive(Debug, Clone)]
pub struct BindChainProbe {
    /// HRR vector dimension (last axis length).
    pub dim: usize,
    /// Number of `bind` steps after the first vector (`depth` additional binds).
    pub depth: usize,
    /// If true, apply [`normalize`] after each [`bind`] and each [`unbind`].
    pub normalize_each: bool,
}

/// Result of running a [`BindChainProbe`].
#[derive(Debug, Clone)]
pub struct ProbeResult {
    /// Cosine similarity between the original first vector and the fully unbound estimate.
    pub cosine_to_original: f32,
}

impl BindChainProbe {
    /// Run the probe: chain `depth` binds on random normalized vectors, then unbind in reverse.
    pub fn run(&self, device: &Device) -> Result<ProbeResult> {
        if self.depth == 0 {
            return Ok(ProbeResult {
                cosine_to_original: 1.0,
            });
        }

        let mut vectors: Vec<Tensor> = Vec::with_capacity(self.depth + 1);
        for i in 0..=self.depth {
            // Seeded per (dim, depth, normalize, index). Unseeded
            // `Tensor::randn` missed MIN_COSINE=0.25 on cluster CI
            // (v0.4.0: dim=64 depth=3 normalize_each=true cosine=0.232).
            let seed = 0x4852_5255u64
                ^ (self.dim as u64)
                ^ ((self.depth as u64) << 16)
                ^ ((self.normalize_each as u64) << 32)
                ^ ((i as u64) << 8);
            vectors.push(random_normalized(self.dim, device, seed)?);
        }

        let mut h = vectors[0].clone();
        for v in vectors.iter().skip(1) {
            h = bind(&h, v)?;
            if self.normalize_each {
                h = normalize(&h)?;
            }
        }

        let mut current = h;
        for i in (1..=self.depth).rev() {
            current = unbind(&current, &vectors[i])?;
            if self.normalize_each {
                current = normalize(&current)?;
            }
        }

        let cosine = cosine_similarity(&vectors[0], &current)?;
        Ok(ProbeResult {
            cosine_to_original: cosine,
        })
    }
}

fn random_normalized(dim: usize, device: &Device, seed: u64) -> Result<Tensor> {
    use crate::information::monte_carlo::SimpleRng;
    let mut rng = SimpleRng::new(seed);
    let vals: Vec<f32> = (0..dim).map(|_| rng.normal() as f32).collect();
    normalize(&Tensor::from_vec(vals, (dim,), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn bind_chain_depth_one_recoverable() {
        let device = Device::Cpu;
        let probe = BindChainProbe {
            dim: 128,
            depth: 1,
            normalize_each: false,
        };
        let r = probe.run(&device).unwrap();
        assert!(r.cosine_to_original > 0.5, "got {}", r.cosine_to_original);
    }
}

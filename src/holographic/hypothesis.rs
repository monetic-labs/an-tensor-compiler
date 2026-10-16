//! Candidate rule storage and scoring without evolutionary semantics.
//!
//! Population updates, fitness, and domain predicates live outside HRR (e.g. `an-cortex` / HETS).

use candle_core::Tensor;

use crate::holographic::ops::{cosine_similarity, normalize, project, superimpose};
use crate::Result;

/// A single candidate rule as an HRR vector plus stable id.
#[derive(Debug, Clone)]
pub struct CandidateRule {
    pub id: String,
    pub hologram: Tensor,
}

impl CandidateRule {
    pub fn new(id: impl Into<String>, hologram: Tensor) -> Result<Self> {
        Ok(Self {
            id: id.into(),
            hologram: normalize(&hologram)?,
        })
    }
}

/// Indexed candidate rules with optional superposed summary hologram.
#[derive(Debug, Clone)]
pub struct HypothesisBank {
    pub candidates: Vec<CandidateRule>,
    pub superposed: Option<Tensor>,
}

impl HypothesisBank {
    pub fn from_candidates(candidates: Vec<CandidateRule>) -> Result<Self> {
        Ok(Self {
            candidates,
            superposed: None,
        })
    }

    /// Build normalized superposition of all candidate holograms (uniform weights).
    pub fn with_superposition(mut self) -> Result<Self> {
        if self.candidates.is_empty() {
            self.superposed = None;
            return Ok(self);
        }
        let tensors: Vec<Tensor> = self.candidates.iter().map(|c| c.hologram.clone()).collect();
        self.superposed = Some(normalize(&superimpose(&tensors, None)?)?);
        Ok(self)
    }

    /// Rank candidates by cosine similarity between `query` and each hologram.
    pub fn score_by_cosine(&self, query: &Tensor) -> Result<Vec<(String, f32)>> {
        let q = normalize(query)?;
        let mut out: Vec<(String, f32)> = Vec::with_capacity(self.candidates.len());
        for c in &self.candidates {
            let s = cosine_similarity(&q, &c.hologram)?;
            out.push((c.id.clone(), s));
        }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(out)
    }

    /// Cosine between normalized `query` and the superposed bank hologram, if built.
    pub fn score_superposed_similarity(&self, query: &Tensor) -> Result<Option<f32>> {
        let Some(ref holo) = self.superposed else {
            return Ok(None);
        };
        let q = normalize(query)?;
        Ok(Some(cosine_similarity(&q, holo)?))
    }

    /// Project query through the superposed hologram (associative unbind) and score
    /// cosine match to each candidate (diagnostic; not a domain predicate).
    pub fn score_by_project_superposed(&self, query: &Tensor) -> Result<Vec<(String, f32)>> {
        let Some(ref holo) = self.superposed else {
            return Ok(vec![]);
        };
        let q = normalize(query)?;
        let mut out = Vec::with_capacity(self.candidates.len());
        for c in &self.candidates {
            let projected = project(holo, &q)?;
            let s = cosine_similarity(&projected, &c.hologram)?;
            out.push((c.id.clone(), s));
        }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn hypothesis_bank_ranks_by_cosine() {
        let device = Device::Cpu;
        let a = CandidateRule::new(
            "a",
            Tensor::ones((32,), candle_core::DType::F32, &device).unwrap(),
        )
        .unwrap();
        let b = CandidateRule::new(
            "b",
            Tensor::zeros((32,), candle_core::DType::F32, &device).unwrap(),
        )
        .unwrap();
        let bank = HypothesisBank::from_candidates(vec![a, b]).unwrap();
        let q = Tensor::ones((32,), candle_core::DType::F32, &device).unwrap();
        let scores = bank.score_by_cosine(&q).unwrap();
        assert_eq!(scores[0].0, "a");
    }
}

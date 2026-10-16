//! Domain-neutral lattice (grid) HRR encoding.
//!
//! Binds deterministic position keys to caller-supplied content tensors. ARC-specific
//! color rules or task semantics belong in application layers (e.g. `an-cortex`).

use candle_core::{Device, Tensor};

use crate::holographic::ops::{bind, normalize, role_encoding, superimpose};
use crate::Result;

/// How normalization is applied when building cell holograms or the full grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatticeNormalization {
    /// No normalization beyond content normalization in [`LatticeEncoder::encode_cell`].
    None,
    /// Normalize each cell hologram before optional superposition.
    PerCell,
    /// Normalize only the final superposed grid hologram.
    WholeGrid,
}

/// Layout of a 2D lattice. Rows and columns are zero-based indices.
#[derive(Debug, Clone)]
pub struct LatticeSpec {
    /// Row count (zero-based indices).
    pub rows: usize,
    /// Column count (zero-based indices).
    pub cols: usize,
    /// Number of discrete channel slots (caller-defined semantics); used for documentation
    /// and optional validation when using [`LatticeEncoder::encode_grid_superposition`].
    pub channel_slots: usize,
}

impl LatticeSpec {
    /// Total cells (`rows * cols`).
    pub fn cell_count(&self) -> usize {
        self.rows * self.cols
    }
}

/// Encodes spatial structure into HRR vectors using deterministic position keys.
pub struct LatticeEncoder {
    dim: usize,
    device: Device,
    spec: LatticeSpec,
    norm: LatticeNormalization,
}

impl LatticeEncoder {
    /// Encoder for `spec` at `dim` dimensions with the given normalization.
    pub fn new(dim: usize, device: &Device, spec: LatticeSpec, norm: LatticeNormalization) -> Self {
        Self {
            dim,
            device: device.clone(),
            spec,
            norm,
        }
    }

    /// Hologram dimension.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Lattice layout.
    pub fn spec(&self) -> &LatticeSpec {
        &self.spec
    }

    /// Deterministic hologram for a cell address (no content).
    pub fn cell_position_hologram(&self, row: usize, col: usize) -> Result<Tensor> {
        let key = format!("lattice:cell:{row}:{col}");
        role_encoding(&key, self.dim, &self.device)
    }

    /// Bind position to normalized `content` (single cell hologram).
    pub fn encode_cell(&self, row: usize, col: usize, content: &Tensor) -> Result<Tensor> {
        let pos = self.cell_position_hologram(row, col)?;
        let c = normalize(content)?;
        let h = bind(&pos, &c)?;
        match self.norm {
            LatticeNormalization::PerCell => normalize(&h),
            _ => Ok(h),
        }
    }

    /// Row-major list of cell contents; length must equal `rows * cols`.
    pub fn encode_grid_superposition(&self, row_major_contents: &[Tensor]) -> Result<Tensor> {
        let n = self.spec.cell_count();
        if row_major_contents.len() != n {
            return Err(crate::TensorCoreError::Tensor(format!(
                "encode_grid_superposition: expected {} cells, got {}",
                n,
                row_major_contents.len()
            )));
        }

        let mut pieces = Vec::with_capacity(n);
        for row in 0..self.spec.rows {
            for col in 0..self.spec.cols {
                let idx = row * self.spec.cols + col;
                let h = self.encode_cell(row, col, &row_major_contents[idx])?;
                pieces.push(h);
            }
        }

        let grid = superimpose(&pieces, None)?;
        match self.norm {
            LatticeNormalization::WholeGrid | LatticeNormalization::PerCell => normalize(&grid),
            LatticeNormalization::None => Ok(grid),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holographic::ops::cosine_similarity;

    #[test]
    fn small_grid_superposition_is_finite() {
        let device = Device::Cpu;
        let spec = LatticeSpec {
            rows: 2,
            cols: 2,
            channel_slots: 4,
        };
        let enc = LatticeEncoder::new(128, &device, spec, LatticeNormalization::WholeGrid);
        let cells: Vec<Tensor> = (0..4)
            .map(|_| Tensor::randn(0.0f32, 1.0, (128,), &device).unwrap())
            .collect();
        let grid = enc.encode_grid_superposition(&cells).unwrap();
        let t = grid.flatten_all().unwrap();
        assert!(t.sum_all().unwrap().to_scalar::<f32>().unwrap().is_finite());
    }

    #[test]
    fn same_cell_position_is_deterministic() {
        let device = Device::Cpu;
        let spec = LatticeSpec {
            rows: 3,
            cols: 3,
            channel_slots: 1,
        };
        let enc = LatticeEncoder::new(64, &device, spec, LatticeNormalization::None);
        let a = enc.cell_position_hologram(1, 2).unwrap();
        let b = enc.cell_position_hologram(1, 2).unwrap();
        let s = cosine_similarity(&a, &b).unwrap();
        assert!((s - 1.0).abs() < 1e-4);
    }
}

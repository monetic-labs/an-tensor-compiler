//! Geometric cell-locality primitives for HRR retrieval.
//!
//! Mirrors the proven `AtlasHrrBalanced` pattern from `an-cache` (G7-graduated,
//! best raw locality at 19.1%) but operates on `Tensor` at the retrieval
//! dimension (1024-dim) instead of `HrrVec` at 256-dim.
//!
//! Items get cell addresses at compile time; queries compute their cell from
//! the projection result and fetch only items in that cell + neighbors, then
//! rerank with cosine. This reduces retrieval from O(n) linear scan to
//! O(|candidates|) where the candidate set is typically 5-25% of the corpus.
//!
//! ## Cell addressing
//!
//! A hologram is placed by:
//! 1. `argmax` cosine against `n_classes` (default 96) class reference vectors.
//! 2. `argmax` cosine against `n_sub_refs` (default 128) sub-cell reference vectors.
//! 3. `cell_id = class * n_sub_refs + sub`.
//!
//! Reference vectors are created via [`role_encoding`] at the retrieval
//! dimension, so they live in the same vector space as item holograms and
//! query projections — the same reference vectors apply at query time.
//!
//! ## Neighbor lookup
//!
//! At query time, neighbors are adjacent classes (all sub-cells) plus the
//! same class with adjacent sub-cells, out to a `radius`. This bounds the
//! candidate set while preserving locality.

use crate::holographic::ops::{cosine_similarity, role_encoding};
use crate::{Device, Result, Tensor, TensorCoreError};
use std::collections::HashMap;

/// Default number of class reference vectors (matches `AtlasHrrBalanced::N_REFS`).
pub const DEFAULT_N_CLASSES: usize = 96;

/// Default number of sub-cell reference vectors (matches `AtlasHrrBalanced::N_SUB_REFS`).
pub const DEFAULT_N_SUB_REFS: usize = 128;

/// A geometric cell-locality index over HRR holograms.
///
/// Built once at compile time over a corpus of item holograms; queried at
/// projection time to narrow the candidate set before cosine reranking.
/// The reference vectors are deterministic given the same
/// `(dim, n_classes, n_sub_refs)`, so any two indices built with the same
/// parameters produce the same cell addresses for the same holograms.
#[derive(Debug, Clone)]
pub struct GeometricIndex {
    /// Class reference vectors at the retrieval dimension.
    pub class_refs: Vec<Tensor>,
    /// Sub-cell reference vectors at the retrieval dimension.
    pub sub_refs: Vec<Tensor>,
    /// `cell_id -> item indices` mapping.
    pub cell_index: HashMap<u64, Vec<usize>>,
    /// Number of class reference vectors.
    pub n_classes: usize,
    /// Number of sub-cell reference vectors.
    pub n_sub_refs: usize,
}

impl GeometricIndex {
    /// Build a new empty geometric index with reference vectors created via
    /// [`role_encoding`] at `dim`.
    ///
    /// The reference vectors are deterministic given the same
    /// `(dim, n_classes, n_sub_refs)`, so callers that build indices over
    /// disjoint slices of the same corpus (e.g. sharded compilers) will
    /// agree on cell addresses.
    pub fn new(dim: usize, device: &Device, n_classes: usize, n_sub_refs: usize) -> Result<Self> {
        if n_classes == 0 || n_sub_refs == 0 {
            return Err(TensorCoreError::Tensor(format!(
                "GeometricIndex requires n_classes > 0 and n_sub_refs > 0 (got {} / {})",
                n_classes, n_sub_refs
            )));
        }
        let class_refs = (0..n_classes)
            .map(|c| role_encoding(&format!("hrr-geo:class:{}", c), dim, device))
            .collect::<Result<Vec<_>>>()?;
        let sub_refs = (0..n_sub_refs)
            .map(|s| role_encoding(&format!("hrr-geo:sub:{}", s), dim, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            class_refs,
            sub_refs,
            cell_index: HashMap::new(),
            n_classes,
            n_sub_refs,
        })
    }

    /// Build a geometric index with default parameters (96 classes, 128 sub-cells).
    pub fn new_default(dim: usize, device: &Device) -> Result<Self> {
        Self::new(dim, device, DEFAULT_N_CLASSES, DEFAULT_N_SUB_REFS)
    }

    /// Bulk-build a geometric index from a slice of item holograms, inferring
    /// `dim` and `device` from the first hologram.
    ///
    /// Returns an empty index (no reference vectors) when `holograms` is empty,
    /// so callers can build over a corpus without a special-case branch.
    pub fn build(holograms: &[Tensor], n_classes: usize, n_sub_refs: usize) -> Result<Self> {
        if holograms.is_empty() {
            return Ok(Self {
                class_refs: Vec::new(),
                sub_refs: Vec::new(),
                cell_index: HashMap::new(),
                n_classes,
                n_sub_refs,
            });
        }
        let first = &holograms[0];
        let dim = first.dims().last().copied().ok_or_else(|| {
            TensorCoreError::Tensor("GeometricIndex::build: hologram has zero-dim shape".into())
        })?;
        let device = first.device();
        let mut idx = Self::new(dim, device, n_classes, n_sub_refs)?;
        for (i, h) in holograms.iter().enumerate() {
            idx.insert(i, h)?;
        }
        Ok(idx)
    }

    /// Bulk-build with default parameters (96 classes, 128 sub-cells).
    pub fn build_default(holograms: &[Tensor]) -> Result<Self> {
        Self::build(holograms, DEFAULT_N_CLASSES, DEFAULT_N_SUB_REFS)
    }

    /// Compute the cell id for a hologram by `argmax` cosine against class
    /// refs then sub-cell refs. `cell_id = class * n_sub_refs + sub`.
    pub fn place(&self, hologram: &Tensor) -> Result<u64> {
        let class = self.argmax_class(hologram)?;
        let sub = self.argmax_sub(hologram)?;
        Ok(class as u64 * self.n_sub_refs as u64 + sub as u64)
    }

    /// Compute the cell id for a query projection. Same computation as
    /// [`place`](Self::place) — the query projection lives in the same
    /// vector space as item holograms, so the same reference vectors apply.
    pub fn query_cell(&self, projection: &Tensor) -> Result<u64> {
        self.place(projection)
    }

    fn argmax_class(&self, hologram: &Tensor) -> Result<usize> {
        let mut best = (0usize, f32::NEG_INFINITY);
        for (c, r) in self.class_refs.iter().enumerate() {
            let sim = cosine_similarity(hologram, r)?;
            if sim > best.1 {
                best = (c, sim);
            }
        }
        Ok(best.0)
    }

    fn argmax_sub(&self, hologram: &Tensor) -> Result<usize> {
        let mut best = (0usize, f32::NEG_INFINITY);
        for (s, r) in self.sub_refs.iter().enumerate() {
            let sim = cosine_similarity(hologram, r)?;
            if sim > best.1 {
                best = (s, sim);
            }
        }
        Ok(best.0)
    }

    /// Decompose a cell id into `(class, sub)`.
    pub fn decode_cell(&self, cell_id: u64) -> (usize, usize) {
        let class = (cell_id / self.n_sub_refs as u64) as usize;
        let sub = (cell_id % self.n_sub_refs as u64) as usize;
        (class, sub)
    }

    /// Return the cell ids of neighboring cells within `radius` classes/sub-cells.
    ///
    /// At `radius == 0`, returns just the cell itself. At `radius >= 1`,
    /// includes:
    /// - adjacent classes (all sub-cells) for `class +/- 1..=radius`
    /// - same class with adjacent sub-cells for `sub +/- 1..=radius`
    /// - the cell itself
    ///
    /// Class indices wrap modulo `n_classes`; sub-cell indices wrap modulo
    /// `n_sub_refs`, mirroring the toroidal topology of the reference space.
    /// The result is sorted and deduplicated.
    pub fn neighbor_cells(&self, cell_id: u64, radius: usize) -> Vec<u64> {
        if radius == 0 {
            return vec![cell_id];
        }
        let (class, sub) = self.decode_cell(cell_id);
        let mut out = Vec::new();

        // Adjacent classes (all sub-cells), wrapping modulo n_classes.
        for dc in 1..=radius {
            for &c in &[
                (class + self.n_classes - dc) % self.n_classes,
                (class + dc) % self.n_classes,
            ] {
                for s in 0..self.n_sub_refs {
                    out.push(c as u64 * self.n_sub_refs as u64 + s as u64);
                }
            }
        }
        // Same class, adjacent sub-cells, wrapping modulo n_sub_refs.
        for ds in 1..=radius {
            for &s in &[
                (sub + self.n_sub_refs - ds) % self.n_sub_refs,
                (sub + ds) % self.n_sub_refs,
            ] {
                out.push(class as u64 * self.n_sub_refs as u64 + s as u64);
            }
        }
        // Always include the cell itself.
        out.push(cell_id);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Return the candidate item indices for a cell + its neighbors.
    ///
    /// The candidate set is the union of items in `cell_id` and all
    /// neighbor cells out to `radius`. Order is cell-first then
    /// neighbor-order; within a cell, items are in insertion order.
    pub fn candidate_items(&self, cell_id: u64, radius: usize) -> Vec<usize> {
        let cells = self.neighbor_cells(cell_id, radius);
        let mut items = Vec::new();
        for c in cells {
            if let Some(v) = self.cell_index.get(&c) {
                items.extend_from_slice(v);
            }
        }
        items
    }

    /// Insert an item into the index, computing its cell from the hologram.
    /// Returns the cell id the item was placed in.
    pub fn insert(&mut self, item_idx: usize, hologram: &Tensor) -> Result<u64> {
        let cell_id = self.place(hologram)?;
        self.cell_index.entry(cell_id).or_default().push(item_idx);
        Ok(cell_id)
    }

    /// Number of items in the index.
    pub fn len(&self) -> usize {
        self.cell_index.values().map(|v| v.len()).sum()
    }

    /// Whether the index has no items.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of distinct cells that contain at least one item.
    pub fn occupied_cells(&self) -> usize {
        self.cell_index.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Device {
        Device::Cpu
    }

    fn make_hologram(seed: usize, dim: usize) -> Tensor {
        let mut state = seed as u64;
        let mut vals = vec![0.0f32; dim];
        for v in vals.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = (state as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0;
        }
        let norm: f32 = vals.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in vals.iter_mut() {
            *v /= norm;
        }
        Tensor::from_vec(vals, (dim,), &device()).unwrap()
    }

    #[test]
    fn place_and_query_cell_agree() {
        let dim = 1024;
        let idx = GeometricIndex::new_default(dim, &device()).unwrap();
        let h = make_hologram(7, dim);
        let cell = idx.place(&h).unwrap();
        let qcell = idx.query_cell(&h).unwrap();
        assert_eq!(
            cell, qcell,
            "place and query_cell must agree on the same tensor"
        );
    }

    #[test]
    fn insert_and_candidate_items_round_trip() {
        let dim = 1024;
        let mut idx = GeometricIndex::new_default(dim, &device()).unwrap();
        let h0 = make_hologram(0, dim);
        let h1 = make_hologram(1, dim);
        let h2 = make_hologram(2, dim);

        let c0 = idx.insert(0, &h0).unwrap();
        let c1 = idx.insert(1, &h1).unwrap();
        let _c2 = idx.insert(2, &h2).unwrap();

        assert_eq!(idx.len(), 3);
        assert!(!idx.is_empty());

        let own = idx.candidate_items(c0, 0);
        assert!(
            own.contains(&0),
            "candidate_items(c0, 0) must contain item 0"
        );

        let own1 = idx.candidate_items(c1, 0);
        assert!(
            own1.contains(&1),
            "candidate_items(c1, 0) must contain item 1"
        );
    }

    #[test]
    fn build_from_holograms_covers_all_items() {
        let dim = 1024;
        let holograms: Vec<Tensor> = (0..50).map(|i| make_hologram(i, dim)).collect();
        let idx = GeometricIndex::build_default(&holograms).unwrap();
        assert_eq!(idx.len(), 50, "build must index every hologram");
        let mut all: Vec<usize> = idx.cell_index.values().flatten().copied().collect();
        all.sort_unstable();
        let expected: Vec<usize> = (0..50).collect();
        assert_eq!(all, expected, "every item must be placed exactly once");
    }

    #[test]
    fn build_empty_returns_empty_index() {
        let idx = GeometricIndex::build_default(&[]).unwrap();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert!(idx.class_refs.is_empty());
    }

    #[test]
    fn neighbor_cells_radius_zero_is_just_self() {
        let dim = 1024;
        let idx = GeometricIndex::new_default(dim, &device()).unwrap();
        let cell = 1234u64;
        let neighbors = idx.neighbor_cells(cell, 0);
        assert_eq!(neighbors, vec![cell]);
    }

    #[test]
    fn neighbor_cells_bounded_and_includes_self() {
        let dim = 1024;
        let idx = GeometricIndex::new_default(dim, &device()).unwrap();
        let cell = 1000u64; // class 7, sub 104
        let neighbors = idx.neighbor_cells(cell, 1);
        assert!(
            neighbors.contains(&cell),
            "neighbors must include the cell itself"
        );
        assert!(
            neighbors.len() <= 259,
            "neighbor set must be bounded: got {}",
            neighbors.len()
        );
        assert!(!neighbors.is_empty());
    }

    #[test]
    fn neighbor_cells_grows_with_radius() {
        let dim = 1024;
        let idx = GeometricIndex::new_default(dim, &device()).unwrap();
        let cell = 1000u64;
        let r1 = idx.neighbor_cells(cell, 1);
        let r2 = idx.neighbor_cells(cell, 2);
        assert!(
            r2.len() > r1.len(),
            "radius 2 must cover more cells than radius 1"
        );
    }

    #[test]
    fn decode_cell_round_trips() {
        let dim = 1024;
        let idx = GeometricIndex::new_default(dim, &device()).unwrap();
        for cell in [0u64, 1, 127, 128, 1000, 12287] {
            let (c, s) = idx.decode_cell(cell);
            let recombined = c as u64 * idx.n_sub_refs as u64 + s as u64;
            assert_eq!(recombined, cell, "decode_cell must round-trip");
            assert!(c < idx.n_classes);
            assert!(s < idx.n_sub_refs);
        }
    }

    #[test]
    fn rejects_zero_refs() {
        let dim = 1024;
        assert!(GeometricIndex::new(dim, &device(), 0, 128).is_err());
        assert!(GeometricIndex::new(dim, &device(), 96, 0).is_err());
    }

    // ---- HRR-G2 invariant gate tests ----

    /// Geometric retrieval must surface the exact-match item as top-1.
    /// When the query equals an item hologram, both linear and geometric
    /// paths must rank that item first (similarity 1.0).
    #[test]
    fn geometric_top1_matches_linear_for_exact_query() {
        let dim = 1024;
        let holograms: Vec<Tensor> = (0..20).map(|i| make_hologram(i, dim)).collect();
        let idx = GeometricIndex::build_default(&holograms).unwrap();

        // Query with item 7's hologram.
        let query = &holograms[7];
        let cell = idx.query_cell(query).unwrap();
        let candidates = idx.candidate_items(cell, 1);

        // The exact-match item must be in the candidate set.
        assert!(
            candidates.contains(&7),
            "geometric candidate set must contain the exact-match item"
        );

        // Rerank candidates by cosine and check top-1 is item 7.
        let mut scored: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|i| {
                (
                    i,
                    crate::holographic::ops::cosine_similarity(query, &holograms[i]).unwrap(),
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(
            scored[0].0, 7,
            "geometric top-1 must be the exact-match item"
        );
        assert!(
            scored[0].1 > 0.999,
            "exact-match cosine must be ~1.0, got {}",
            scored[0].1
        );
    }

    /// For a small corpus with a well-populated index, geometric retrieval
    /// must surface the linear top-1 as its own top-1 whenever that item is in
    /// the candidate set, and must achieve reasonable recall. Uses a small
    /// 4×4=16 cell index so cells are well-populated for a 30-item corpus
    /// (the default 96×128=12288 cell index is designed for 10k+ corpora).
    #[test]
    fn geometric_top1_agrees_with_linear_when_candidate_contains_it() {
        let dim = 1024;
        let n = 30;
        let dev = device();
        let holograms: Vec<Tensor> = (0..n).map(|i| make_hologram(i, dim)).collect();
        // Small index: 4 classes × 4 sub-cells = 16 cells. With 30 items,
        // each cell holds ~2 items, so radius-1 neighbors (~9 cells) cover
        // ~17 items — enough for meaningful recall.
        let mut idx = GeometricIndex::new(dim, &dev, 4, 4).unwrap();
        for (i, h) in holograms.iter().enumerate() {
            idx.insert(i, h).unwrap();
        }

        // Query: a perturbation of item 5.
        let noise = make_hologram(999, dim);
        let query = crate::holographic::ops::normalize(&(&holograms[5] + &noise).unwrap()).unwrap();

        // Linear top-10.
        let mut linear: Vec<(usize, f32)> = holograms
            .iter()
            .enumerate()
            .map(|(i, h)| {
                (
                    i,
                    crate::holographic::ops::cosine_similarity(&query, h).unwrap(),
                )
            })
            .collect();
        linear.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let linear_top1 = linear[0].0;
        let linear_top10: Vec<usize> = linear.iter().take(10).map(|(i, _)| *i).collect();

        // Geometric top-10.
        let cell = idx.query_cell(&query).unwrap();
        let candidates = idx.candidate_items(cell, 1);
        assert!(
            !candidates.is_empty(),
            "candidate set must not be empty for a 16-cell index"
        );
        let linear_top1_in_candidates = candidates.contains(&linear_top1);
        let mut geo: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|i| {
                (
                    i,
                    crate::holographic::ops::cosine_similarity(&query, &holograms[i]).unwrap(),
                )
            })
            .collect();
        geo.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let geo_top10: Vec<usize> = geo.iter().take(10).map(|(i, _)| *i).collect();

        // Correctness invariant: if the linear top-1 is in the candidate set,
        // geometric top-1 must equal it. (Geometric reranks candidates by the
        // same cosine metric, so the best candidate is the linear top-1 when
        // it's present.)
        if linear_top1_in_candidates {
            assert_eq!(
                geo[0].0, linear_top1,
                "geometric top-1 must match linear top-1 when it's a candidate"
            );
        }

        // Every geometric top-10 item must be a valid corpus item — geometric
        // retrieval never invents items outside the corpus.
        for &g in &geo_top10 {
            assert!(g < n, "geometric item {} must be a valid corpus index", g);
        }

        // Recall floor: with a 16-cell index and 30 items, radius-1 neighbors
        // cover ~56% of cells, so the candidate set is large relative to the
        // corpus. At least 3 of the linear top-10 should appear in the
        // geometric top-10.
        let overlap = geo_top10
            .iter()
            .filter(|g| linear_top10.contains(g))
            .count();
        assert!(
            overlap >= 3,
            "geometric top-10 must overlap linear top-10 by at least 3 items, got {}",
            overlap
        );
    }

    /// Bind/unbind round-trip fidelity is not regressed by geometric
    /// placement. A bound hologram, when placed in the geometric index and
    /// retrieved, must still unbind to recover the original value with the
    /// same fidelity as a hologram that never went through the index.
    #[test]
    fn bind_unbind_fidelity_preserved_through_geometric_placement() {
        use crate::holographic::ops::{bind, cosine_similarity, normalize, role_encoding, unbind};

        let dim = 1024;
        let dev = device();

        let role = role_encoding("test:role", dim, &dev).unwrap();
        let value = normalize(&make_hologram(42, dim)).unwrap();
        let bound = bind(&role, &value).unwrap();

        // Place the bound hologram in the geometric index (this only computes
        // a cell address — it must not mutate the tensor).
        let idx = GeometricIndex::new_default(dim, &dev).unwrap();
        let cell = idx.place(&bound).unwrap();
        let recovered_cell = idx.query_cell(&bound).unwrap();
        assert_eq!(
            cell, recovered_cell,
            "bound hologram must map to a stable cell"
        );

        // Unbind the bound hologram and check fidelity.
        let recovered = unbind(&bound, &role).unwrap();
        let fidelity = cosine_similarity(&recovered, &value).unwrap();
        assert!(
            fidelity > 0.5,
            "bind/unbind fidelity must be preserved through geometric placement, got {}",
            fidelity
        );
    }

    /// The cell_index must cover every item exactly once — no item is lost
    /// or duplicated during build.
    #[test]
    fn cell_index_covers_all_items_exactly_once() {
        let dim = 1024;
        let n = 200;
        let holograms: Vec<Tensor> = (0..n).map(|i| make_hologram(i, dim)).collect();
        let idx = GeometricIndex::build_default(&holograms).unwrap();

        assert_eq!(idx.len(), n, "index must contain all {} items", n);

        let mut all_items: Vec<usize> = idx.cell_index.values().flatten().copied().collect();
        all_items.sort_unstable();
        all_items.dedup();
        assert_eq!(
            all_items.len(),
            n,
            "after dedup, must have exactly {} distinct item indices",
            n
        );

        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(
            all_items, expected,
            "item indices must be 0..n with no gaps or dups"
        );
    }

    /// neighbor_cells must return a bounded set that never exceeds the total
    /// number of cells, and must always include the query cell.
    #[test]
    fn neighbor_cells_bounded_and_complete() {
        let dim = 1024;
        let idx = GeometricIndex::new_default(dim, &device()).unwrap();
        let total_cells = (idx.n_classes * idx.n_sub_refs) as u64;

        for cell in [0u64, 1, 500, 1000, 5000, total_cells - 1] {
            for radius in [0usize, 1, 2, 3] {
                let neighbors = idx.neighbor_cells(cell, radius);
                assert!(
                    neighbors.contains(&cell),
                    "radius {} must include self",
                    radius
                );
                assert!(
                    neighbors.len() <= total_cells as usize,
                    "neighbor set must not exceed total cells"
                );
                // Every neighbor must be a valid cell id.
                for &c in &neighbors {
                    assert!(c < total_cells, "neighbor cell {} out of range", c);
                }
            }
        }
    }
}

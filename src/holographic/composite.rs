//! Composite concept helpers: object–relation patterns on top of [`super::entities`].
//!
//! Intentionally avoids HETS vocabulary (evolution, fitness, generations).

use candle_core::{Device, Tensor};

use crate::holographic::entities::{EntityTensor, RelationTensor, RoleTensor};
use crate::holographic::ops::{bind, normalize, superimpose};
use crate::Result;

/// Build a binary [`RelationTensor`] between two entities (wrapper for discoverability).
pub fn compose_binary_relation(
    id: impl Into<String>,
    relation_type: impl Into<String>,
    left: &EntityTensor,
    left_role: impl Into<String>,
    right: &EntityTensor,
    right_role: impl Into<String>,
    device: &Device,
) -> Result<RelationTensor> {
    RelationTensor::binary(
        id,
        relation_type,
        left,
        left_role,
        right,
        right_role,
        device,
    )
}

/// Superimpose a relation hologram with `bind(operator_role, transform_content)` for
/// operator–transform structure (e.g. learned edit or functor application in cortex).
pub fn compose_relation_with_transform(
    relation: &RelationTensor,
    transform_content: &Tensor,
    operator_role: impl Into<String>,
    device: &Device,
) -> Result<Tensor> {
    let dim = relation.dimension();
    let role = RoleTensor::encode(operator_role, dim, device)?;
    let t = normalize(transform_content)?;
    let op_part = bind(&role.hologram, &t)?;
    let rel = normalize(&relation.hologram)?;
    superimpose(&[rel, op_part], Some(&[0.5f32, 0.5f32]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holographic::ops::cosine_similarity;
    use candle_core::Device;

    fn entity(id: &str, dim: usize, device: &Device) -> EntityTensor {
        let h = Tensor::randn(0.0f32, 1.0, (dim,), device).unwrap();
        EntityTensor::new(id, "thing", h).unwrap()
    }

    #[test]
    fn compose_binary_matches_direct_binary() {
        let device = Device::Cpu;
        let left = entity("a", 128, &device);
        let right = entity("b", 128, &device);
        let r1 = compose_binary_relation("r1", "links", &left, "l", &right, "r", &device).unwrap();
        let r2 = RelationTensor::binary("r1", "links", &left, "l", &right, "r", &device).unwrap();
        let s = cosine_similarity(&r1.hologram, &r2.hologram).unwrap();
        assert!((s - 1.0).abs() < 1e-4);
    }

    #[test]
    fn compose_with_transform_is_normalized_length() {
        let device = Device::Cpu;
        let left = entity("a", 64, &device);
        let right = entity("b", 64, &device);
        let rel = RelationTensor::binary("r", "rel", &left, "l", &right, "r", &device).unwrap();
        let xf = Tensor::randn(0.0f32, 1.0, (64,), &device).unwrap();
        let composed = compose_relation_with_transform(&rel, &xf, "transform_op", &device).unwrap();
        let n = composed.flatten_all().unwrap();
        let norm = (n.sqr().unwrap().sum_all().unwrap().sqrt().unwrap())
            .to_scalar::<f32>()
            .unwrap();
        assert!(norm.is_finite() && norm > 0.0);
    }
}

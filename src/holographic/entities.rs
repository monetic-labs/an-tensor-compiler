//! HRR-native entity and relation primitives.
//!
//! These types model entities, attributes, roles, and typed relations as shared
//! holographic representations. They are intentionally distinct from HETS
//! entities, which add adaptive and evolutionary semantics on top.

use candle_core::{Device, Tensor};

use crate::holographic::knowledge::{ItemProvenance, TemporalState};
use crate::holographic::ops::{bind, normalize, role_encoding, superimpose, unbind};
use crate::Result;

/// HRR-native entity representation for shared knowledge modeling.
#[derive(Debug, Clone)]
pub struct EntityTensor {
    /// Stable identifier for the entity.
    pub id: String,
    /// Broad entity type such as `service`, `customer`, or `document`.
    pub entity_type: String,
    /// Roles attached directly to the entity.
    pub roles: Vec<RoleTensor>,
    /// Structured attributes attached to the entity.
    pub attributes: Vec<AttributeTensor>,
    /// Composed hologram for the entity.
    pub hologram: Tensor,
    /// Provenance metadata for the entity.
    pub provenance: ItemProvenance,
    /// Temporal state for freshness and effective-dating.
    pub temporal: TemporalState,
}

impl EntityTensor {
    /// Create a new entity from a precomputed hologram.
    pub fn new(
        id: impl Into<String>,
        entity_type: impl Into<String>,
        hologram: Tensor,
    ) -> Result<Self> {
        Ok(Self {
            id: id.into(),
            entity_type: entity_type.into(),
            roles: Vec::new(),
            attributes: Vec::new(),
            hologram: normalize(&hologram)?,
            provenance: ItemProvenance::default(),
            temporal: TemporalState::default(),
        })
    }

    /// Return the entity HRR dimension.
    pub fn dimension(&self) -> usize {
        self.hologram.dims()[self.hologram.dims().len() - 1]
    }

    /// Attach a role to the entity.
    pub fn with_role(mut self, role: RoleTensor) -> Self {
        self.roles.push(role);
        self
    }

    /// Attach an attribute to the entity.
    pub fn with_attribute(mut self, attribute: AttributeTensor) -> Self {
        self.attributes.push(attribute);
        self
    }

    /// Override provenance metadata.
    pub fn with_provenance(mut self, provenance: ItemProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    /// Override temporal metadata.
    pub fn with_temporal(mut self, temporal: TemporalState) -> Self {
        self.temporal = temporal;
        self
    }

    /// Bind the entity hologram to a role for relation participation.
    pub fn bind_in_role(&self, role: &RoleTensor) -> Result<Tensor> {
        Self::bind_hologram_in_role(&self.hologram, role)
    }

    /// Bind an arbitrary entity hologram into a role.
    pub fn bind_hologram_in_role(entity_hologram: &Tensor, role: &RoleTensor) -> Result<Tensor> {
        bind(&role.hologram, entity_hologram)
    }

    /// Recover an entity hologram from a role binding.
    pub fn recover_from_role_binding(binding: &Tensor, role: &RoleTensor) -> Result<Tensor> {
        unbind(binding, &role.hologram)
    }
}

/// Tensor-native attribute attached to an entity or relation.
#[derive(Debug, Clone)]
pub struct AttributeTensor {
    /// Attribute key.
    pub key: String,
    /// HRR value representation.
    pub value_hologram: Tensor,
    /// Optional confidence value for uncertain attributes.
    pub confidence: Option<f32>,
    /// Optional provenance tag for attribute-specific lineage.
    pub provenance: Option<String>,
}

impl AttributeTensor {
    /// Create a new tensor-native attribute.
    pub fn new(key: impl Into<String>, value_hologram: Tensor) -> Result<Self> {
        Ok(Self {
            key: key.into(),
            value_hologram: normalize(&value_hologram)?,
            confidence: None,
            provenance: None,
        })
    }

    /// Set an optional confidence value.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Set an optional provenance tag.
    pub fn with_provenance(mut self, provenance: impl Into<String>) -> Self {
        self.provenance = Some(provenance.into());
        self
    }
}

/// Role encoding used by entities and relation participants.
#[derive(Debug, Clone)]
pub struct RoleTensor {
    /// Role name.
    pub name: String,
    /// HRR encoding for the role.
    pub hologram: Tensor,
}

impl RoleTensor {
    /// Deterministically encode a role from its name.
    pub fn encode(name: impl Into<String>, dim: usize, device: &Device) -> Result<Self> {
        let name = name.into();
        let hologram = role_encoding(&name, dim, device)?;
        Ok(Self { name, hologram })
    }
}

/// HRR-native typed relation representation.
#[derive(Debug, Clone)]
pub struct RelationTensor {
    /// Stable identifier for the relation.
    pub id: String,
    /// Relation type such as `implements`, `depends_on`, or `applies_to`.
    pub relation_type: String,
    /// Participants taking part in the relation.
    pub participants: Vec<RelationParticipant>,
    /// Role-bound participant encodings used for reversible projection.
    pub participant_bindings: Vec<Tensor>,
    /// Composed hologram for the relation.
    pub hologram: Tensor,
    /// Provenance metadata for the relation.
    pub provenance: ItemProvenance,
    /// Temporal state for freshness and effective-dating.
    pub temporal: TemporalState,
    /// Optional confidence value for inferred relations.
    pub confidence: Option<f32>,
}

impl RelationTensor {
    /// Create a relation from precomputed participant bindings and hologram.
    pub fn new(
        id: impl Into<String>,
        relation_type: impl Into<String>,
        participants: Vec<RelationParticipant>,
        participant_bindings: Vec<Tensor>,
        hologram: Tensor,
    ) -> Result<Self> {
        Ok(Self {
            id: id.into(),
            relation_type: relation_type.into(),
            participants,
            participant_bindings,
            hologram: normalize(&hologram)?,
            provenance: ItemProvenance::default(),
            temporal: TemporalState::default(),
            confidence: None,
        })
    }

    /// Return the relation HRR dimension.
    pub fn dimension(&self) -> usize {
        self.hologram.dims()[self.hologram.dims().len() - 1]
    }

    /// Build a binary relation with deterministic role and relation encodings.
    pub fn binary(
        id: impl Into<String>,
        relation_type: impl Into<String>,
        left: &EntityTensor,
        left_role: impl Into<String>,
        right: &EntityTensor,
        right_role: impl Into<String>,
        device: &Device,
    ) -> Result<Self> {
        let relation_type = relation_type.into();
        let dim = left.dimension();
        let right_dim = right.dimension();
        if dim != right_dim {
            return Err(crate::TensorCoreError::Tensor(format!(
                "Entity dimension mismatch for relation: {} vs {}",
                dim, right_dim
            )));
        }

        let left_role = RoleTensor::encode(left_role, dim, device)?;
        let right_role = RoleTensor::encode(right_role, dim, device)?;
        let participants = vec![
            RelationParticipant::new(left.id.clone(), left_role.clone()),
            RelationParticipant::new(right.id.clone(), right_role.clone()),
        ];

        let left_binding = left.bind_in_role(&left_role)?;
        let right_binding = right.bind_in_role(&right_role)?;
        let relation_encoding = role_encoding(&relation_type, dim, device)?;
        let participant_bindings = vec![left_binding.clone(), right_binding.clone()];
        let hologram = superimpose(&[relation_encoding, left_binding, right_binding], None)?;

        Self::new(
            id,
            relation_type,
            participants,
            participant_bindings,
            hologram,
        )
    }

    /// Recover a participant hologram from a stored participant binding.
    pub fn recover_participant(&self, index: usize) -> Result<Tensor> {
        let binding = self.participant_bindings.get(index).ok_or_else(|| {
            crate::TensorCoreError::Tensor(format!(
                "Participant binding index {} out of range",
                index
            ))
        })?;
        let participant = self.participants.get(index).ok_or_else(|| {
            crate::TensorCoreError::Tensor(format!("Participant index {} out of range", index))
        })?;

        EntityTensor::recover_from_role_binding(binding, &participant.role)
    }

    /// Set optional confidence for an inferred relation.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Override provenance metadata.
    pub fn with_provenance(mut self, provenance: ItemProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    /// Override temporal metadata.
    pub fn with_temporal(mut self, temporal: TemporalState) -> Self {
        self.temporal = temporal;
        self
    }
}

/// A single participant in a relation.
#[derive(Debug, Clone)]
pub struct RelationParticipant {
    /// Referenced entity ID.
    pub entity_id: String,
    /// Role played by the entity in this relation.
    pub role: RoleTensor,
}

impl RelationParticipant {
    /// Create a relation participant with a specific role.
    pub fn new(entity_id: impl Into<String>, role: RoleTensor) -> Self {
        Self {
            entity_id: entity_id.into(),
            role,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holographic::ops::cosine_similarity;

    fn random_entity(id: &str, entity_type: &str, dim: usize, device: &Device) -> EntityTensor {
        let hologram = Tensor::randn(0.0f32, 1.0, (dim,), device).unwrap();
        EntityTensor::new(id, entity_type, hologram).unwrap()
    }

    #[test]
    fn test_role_tensor_encode_is_deterministic() {
        let device = Device::Cpu;
        let role_a = RoleTensor::encode("owner", 64, &device).unwrap();
        let role_b = RoleTensor::encode("owner", 64, &device).unwrap();

        let similarity = cosine_similarity(&role_a.hologram, &role_b.hologram).unwrap();
        assert!((similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_entity_role_binding_is_recoverable() {
        let device = Device::Cpu;
        let entity = random_entity("entity-1", "service", 256, &device);
        let role = RoleTensor::encode("subject", 256, &device).unwrap();

        let binding = entity.bind_in_role(&role).unwrap();
        let recovered = EntityTensor::recover_from_role_binding(&binding, &role).unwrap();
        let similarity = cosine_similarity(&entity.hologram, &recovered).unwrap();

        assert!(
            similarity > 0.5,
            "Expected >0.5 recovery similarity, got {}",
            similarity
        );
    }

    #[test]
    fn test_binary_relation_recovers_participants() {
        let device = Device::Cpu;
        let left = random_entity("capability-1", "capability", 256, &device);
        let right = random_entity("impl-1", "implementation", 256, &device);

        let relation = RelationTensor::binary(
            "relation-1",
            "implements",
            &left,
            "subject",
            &right,
            "object",
            &device,
        )
        .unwrap();

        let recovered_left = relation.recover_participant(0).unwrap();
        let recovered_right = relation.recover_participant(1).unwrap();
        let left_similarity = cosine_similarity(&left.hologram, &recovered_left).unwrap();
        let right_similarity = cosine_similarity(&right.hologram, &recovered_right).unwrap();

        assert_eq!(relation.participants.len(), 2);
        assert_eq!(relation.participant_bindings.len(), 2);
        assert!(
            left_similarity > 0.5,
            "Expected >0.5 left recovery, got {}",
            left_similarity
        );
        assert!(
            right_similarity > 0.5,
            "Expected >0.5 right recovery, got {}",
            right_similarity
        );
    }
}

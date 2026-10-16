//! Versioned interchange for named tensor payloads (HRR artifacts).
//!
//! Full `HetsEntity` or closure snapshots are out of scope; serialize those at HETS / cortex.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

use crate::TensorCoreError;

/// Flat tensor payload suitable for JSON and checksums.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorPayload {
    /// Row-major tensor shape.
    pub shape: Vec<usize>,
    /// Flattened row-major values.
    pub data: Vec<f32>,
}

/// Versioned bundle of tensors + metadata for checkpoints (not HETS entities).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HrrArtifact {
    /// Format tag; must equal `FORMAT_VERSION`.
    pub format_version: String,
    /// Hologram dimension of the payloads.
    pub dim: usize,
    /// Whether consumers should normalize on load.
    pub normalize_outputs: bool,
    /// Named tensor payloads.
    pub tensors: BTreeMap<String, TensorPayload>,
    /// Hex checksum over canonical JSON, if sealed.
    pub checksum_sha256: Option<String>,
}

impl HrrArtifact {
    /// Current format tag written by `new`.
    pub const FORMAT_VERSION: &'static str = "hrr_artifact/1";

    /// Empty artifact at the current format version.
    pub fn new(dim: usize, normalize_outputs: bool) -> Self {
        Self {
            format_version: Self::FORMAT_VERSION.to_string(),
            dim,
            normalize_outputs,
            tensors: BTreeMap::new(),
            checksum_sha256: None,
        }
    }

    /// Insert or replace a named payload.
    pub fn insert_tensor(&mut self, name: impl Into<String>, payload: TensorPayload) {
        self.tensors.insert(name.into(), payload);
    }

    /// Compute checksum over canonical JSON of (version, dim, normalize_outputs, tensors).
    pub fn compute_checksum(&self) -> String {
        let s = Self {
            format_version: self.format_version.clone(),
            dim: self.dim,
            normalize_outputs: self.normalize_outputs,
            tensors: self.tensors.clone(),
            checksum_sha256: None,
        };
        let json = serde_json::to_string(&s).unwrap_or_else(|_| "{}".to_string());
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Seal with a checksum over current contents (builder style).
    pub fn with_checksum(mut self) -> Self {
        self.checksum_sha256 = Some(self.compute_checksum());
        self
    }

    /// Verify the sealed checksum against current contents.
    pub fn verify_checksum(&self) -> Result<(), TensorCoreError> {
        let Some(ref expected) = self.checksum_sha256 else {
            return Err(TensorCoreError::Tensor(
                "HrrArtifact has no checksum to verify".into(),
            ));
        };
        let mut probe = self.clone();
        probe.checksum_sha256 = None;
        let actual = probe.compute_checksum();
        if actual != *expected {
            return Err(TensorCoreError::Tensor(format!(
                "checksum mismatch: expected {expected}, got {actual}"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_json_and_checksum() {
        let mut a = HrrArtifact::new(64, true);
        a.insert_tensor(
            "test",
            TensorPayload {
                shape: vec![64],
                data: vec![1.0f32; 64],
            },
        );
        let a = a.with_checksum();
        let json = serde_json::to_string(&a).unwrap();
        let b: HrrArtifact = serde_json::from_str(&json).unwrap();
        b.verify_checksum().unwrap();
    }
}

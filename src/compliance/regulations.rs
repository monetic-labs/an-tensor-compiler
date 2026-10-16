//! Regulatory Framework Definitions
//!
//! Defines supported regulations and their requirements.

use serde::{Deserialize, Serialize};

/// Supported regulatory frameworks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Regulation {
    /// General Data Protection Regulation (EU)
    GDPR,
    /// California Consumer Privacy Act
    CCPA,
    /// Sarbanes-Oxley Act (financial)
    SOX,
    /// Health Insurance Portability and Accountability Act
    HIPAA,
    /// Payment Card Industry Data Security Standard
    PCIDSS,
    /// General/custom compliance requirements
    General,
}

impl Regulation {
    /// All supported regulations
    pub fn all() -> &'static [Regulation] {
        &[
            Self::GDPR,
            Self::CCPA,
            Self::SOX,
            Self::HIPAA,
            Self::PCIDSS,
            Self::General,
        ]
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::GDPR => "General Data Protection Regulation",
            Self::CCPA => "California Consumer Privacy Act",
            Self::SOX => "Sarbanes-Oxley Act",
            Self::HIPAA => "Health Insurance Portability and Accountability Act",
            Self::PCIDSS => "Payment Card Industry Data Security Standard",
            Self::General => "General Compliance",
        }
    }

    /// Short code
    pub fn code(&self) -> &'static str {
        match self {
            Self::GDPR => "GDPR",
            Self::CCPA => "CCPA",
            Self::SOX => "SOX",
            Self::HIPAA => "HIPAA",
            Self::PCIDSS => "PCI-DSS",
            Self::General => "GENERAL",
        }
    }

    /// Default retention period in days
    pub fn default_retention_days(&self) -> f32 {
        match self {
            Self::GDPR => 1095.0,    // 3 years (typical)
            Self::CCPA => 365.0,     // 1 year minimum
            Self::SOX => 2555.0,     // 7 years
            Self::HIPAA => 2190.0,   // 6 years
            Self::PCIDSS => 365.0,   // 1 year minimum
            Self::General => 2555.0, // 7 years (conservative)
        }
    }

    /// Required consent validity period in days
    pub fn consent_validity_days(&self) -> f32 {
        match self {
            Self::GDPR => 365.0,           // Annual renewal common
            Self::CCPA => 365.0,           // Annual opt-out window
            Self::SOX => f32::INFINITY,    // N/A
            Self::HIPAA => 365.0,          // Annual authorization
            Self::PCIDSS => f32::INFINITY, // N/A
            Self::General => 365.0,
        }
    }

    /// Requires explicit consent for data processing
    pub fn requires_explicit_consent(&self) -> bool {
        match self {
            Self::GDPR => true,
            Self::CCPA => false, // Opt-out model
            Self::SOX => false,
            Self::HIPAA => true,
            Self::PCIDSS => false,
            Self::General => true,
        }
    }

    /// Requires data localization (geographic restrictions)
    pub fn requires_data_localization(&self) -> bool {
        match self {
            Self::GDPR => true, // EU data stays in EU (with exceptions)
            Self::CCPA => false,
            Self::SOX => false,
            Self::HIPAA => false,
            Self::PCIDSS => false,
            Self::General => false,
        }
    }

    /// Requires breach notification within N hours (None if no requirement)
    pub fn breach_notification_hours(&self) -> Option<u32> {
        match self {
            Self::GDPR => Some(72),
            Self::CCPA => None, // "Expedient" but no specific time
            Self::SOX => None,
            Self::HIPAA => Some(60 * 24), // 60 days
            Self::PCIDSS => None,
            Self::General => None,
        }
    }

    /// Jurisdictions covered by this regulation
    pub fn jurisdictions(&self) -> &'static [&'static str] {
        match self {
            Self::GDPR => &["EU", "EEA", "UK"],
            Self::CCPA => &["US-CA"],
            Self::SOX => &["US"],
            Self::HIPAA => &["US"],
            Self::PCIDSS => &["GLOBAL"],
            Self::General => &["GLOBAL"],
        }
    }
}

impl std::fmt::Display for Regulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code())
    }
}

/// Requirement severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequirementLevel {
    /// Must be implemented
    Mandatory,
    /// Should be implemented
    Recommended,
    /// Optional implementation
    Optional,
}

/// A specific compliance requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Requirement {
    /// Unique identifier
    pub id: String,
    /// Regulation this belongs to
    pub regulation: Regulation,
    /// Human-readable description
    pub description: String,
    /// Severity level
    pub level: RequirementLevel,
    /// Associated predicate (if any)
    pub predicate: Option<String>,
}

impl Requirement {
    /// GDPR Article 17 - Right to Erasure
    pub fn gdpr_right_to_erasure() -> Self {
        Self {
            id: "GDPR-17".to_string(),
            regulation: Regulation::GDPR,
            description: "Right to erasure ('right to be forgotten')".to_string(),
            level: RequirementLevel::Mandatory,
            predicate: Some("retention_exceeded".to_string()),
        }
    }

    /// GDPR Article 6 - Lawful Basis
    pub fn gdpr_lawful_basis() -> Self {
        Self {
            id: "GDPR-6".to_string(),
            regulation: Regulation::GDPR,
            description: "Lawful basis for processing".to_string(),
            level: RequirementLevel::Mandatory,
            predicate: Some("consent_valid".to_string()),
        }
    }

    /// HIPAA Privacy Rule - Minimum Necessary
    pub fn hipaa_minimum_necessary() -> Self {
        Self {
            id: "HIPAA-MIN".to_string(),
            regulation: Regulation::HIPAA,
            description: "Minimum necessary use and disclosure".to_string(),
            level: RequirementLevel::Mandatory,
            predicate: Some("access_authorized".to_string()),
        }
    }

    /// SOX Section 802 - Document Retention
    pub fn sox_document_retention() -> Self {
        Self {
            id: "SOX-802".to_string(),
            regulation: Regulation::SOX,
            description: "Retention of audit workpapers".to_string(),
            level: RequirementLevel::Mandatory,
            predicate: Some("audit_complete".to_string()),
        }
    }

    /// PCI-DSS Requirement 3 - Protect Cardholder Data
    pub fn pci_protect_data() -> Self {
        Self {
            id: "PCI-3".to_string(),
            regulation: Regulation::PCIDSS,
            description: "Protect stored cardholder data".to_string(),
            level: RequirementLevel::Mandatory,
            predicate: Some("pii_detected".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regulation_names() {
        assert_eq!(
            Regulation::GDPR.name(),
            "General Data Protection Regulation"
        );
        assert_eq!(Regulation::SOX.code(), "SOX");
    }

    #[test]
    fn test_regulation_retention_days() {
        assert_eq!(Regulation::SOX.default_retention_days(), 2555.0);
        assert!(
            Regulation::GDPR.default_retention_days() < Regulation::SOX.default_retention_days()
        );
    }

    #[test]
    fn test_regulation_consent_requirements() {
        assert!(Regulation::GDPR.requires_explicit_consent());
        assert!(!Regulation::CCPA.requires_explicit_consent());
    }

    #[test]
    fn test_regulation_data_localization() {
        assert!(Regulation::GDPR.requires_data_localization());
        assert!(!Regulation::SOX.requires_data_localization());
    }

    #[test]
    fn test_regulation_breach_notification() {
        assert_eq!(Regulation::GDPR.breach_notification_hours(), Some(72));
        assert!(Regulation::CCPA.breach_notification_hours().is_none());
    }

    #[test]
    fn test_regulation_jurisdictions() {
        assert!(Regulation::GDPR.jurisdictions().contains(&"EU"));
        assert!(Regulation::CCPA.jurisdictions().contains(&"US-CA"));
    }

    #[test]
    fn test_regulation_display() {
        assert_eq!(format!("{}", Regulation::HIPAA), "HIPAA");
    }

    #[test]
    fn test_regulation_serde() {
        let reg = Regulation::GDPR;
        let json = serde_json::to_string(&reg).unwrap();
        assert_eq!(json, "\"GDPR\"");

        let parsed: Regulation = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, reg);
    }

    #[test]
    fn test_requirement_gdpr() {
        let req = Requirement::gdpr_right_to_erasure();
        assert_eq!(req.regulation, Regulation::GDPR);
        assert_eq!(req.level, RequirementLevel::Mandatory);
    }
}

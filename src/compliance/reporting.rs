//! Compliance Reporting
//!
//! Report generation and export for regulatory requirements.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::regulations::Regulation;
use crate::namespace::NamespacePath;

/// Compliance report for a namespace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Regulation being reported on
    pub regulation: Regulation,
    /// Namespace scope
    pub namespace: NamespacePath,
    /// Report version (for format compatibility)
    pub version: String,
    /// Summary metrics
    pub summary: ComplianceSummary,
    /// Individual findings
    pub findings: Vec<ComplianceFinding>,
    /// Recommended actions
    pub recommendations: Vec<Recommendation>,
}

impl ComplianceReport {
    /// Create a new compliance report
    pub fn new(regulation: Regulation, namespace: NamespacePath) -> Self {
        Self {
            generated_at: Utc::now(),
            regulation,
            namespace,
            version: "1.0".to_string(),
            summary: ComplianceSummary::default(),
            findings: Vec::new(),
            recommendations: Vec::new(),
        }
    }

    /// Add a finding
    pub fn add_finding(&mut self, finding: ComplianceFinding) {
        // Update summary based on severity
        match finding.severity {
            Severity::Critical => self.summary.critical_count += 1,
            Severity::High => self.summary.high_count += 1,
            Severity::Medium => self.summary.medium_count += 1,
            Severity::Low => self.summary.low_count += 1,
            Severity::Info => self.summary.info_count += 1,
        }
        self.summary.total_findings += 1;
        self.findings.push(finding);
    }

    /// Add a recommendation
    pub fn add_recommendation(&mut self, recommendation: Recommendation) {
        self.recommendations.push(recommendation);
    }

    /// Calculate overall compliance score (0.0-1.0)
    pub fn compliance_score(&self) -> f32 {
        if self.summary.total_findings == 0 {
            return 1.0;
        }

        // Weighted score: critical=50, high=20, medium=5, low=1, info=0
        let penalty = (self.summary.critical_count * 50
            + self.summary.high_count * 20
            + self.summary.medium_count * 5
            + self.summary.low_count) as f32;

        // Score decreases with findings, minimum 0
        (1.0 - penalty / 100.0).max(0.0)
    }

    /// Is the organization compliant (no critical/high findings)?
    pub fn is_compliant(&self) -> bool {
        self.summary.critical_count == 0 && self.summary.high_count == 0
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export to compact JSON
    pub fn to_json_compact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// Summary metrics for a compliance report
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComplianceSummary {
    /// Total number of findings
    pub total_findings: usize,
    /// Critical severity count
    pub critical_count: usize,
    /// High severity count
    pub high_count: usize,
    /// Medium severity count
    pub medium_count: usize,
    /// Low severity count
    pub low_count: usize,
    /// Informational count
    pub info_count: usize,
    /// Number of records analyzed
    pub records_analyzed: usize,
    /// Number of PII detections
    pub pii_detections: usize,
    /// Number of consent issues
    pub consent_issues: usize,
    /// Number of retention issues
    pub retention_issues: usize,
}

/// Individual compliance finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceFinding {
    /// Unique identifier
    pub id: String,
    /// Finding type
    pub finding_type: FindingType,
    /// Severity level
    pub severity: Severity,
    /// Human-readable title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Affected resource (if applicable)
    pub resource: Option<String>,
    /// Regulation requirement reference
    pub requirement_ref: Option<String>,
    /// Remediation steps
    pub remediation: Option<String>,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
}

impl ComplianceFinding {
    /// Create a new finding
    pub fn new(
        finding_type: FindingType,
        severity: Severity,
        title: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            finding_type,
            severity,
            title: title.into(),
            description: description.into(),
            resource: None,
            requirement_ref: None,
            remediation: None,
            detected_at: Utc::now(),
        }
    }

    /// Set the affected resource
    pub fn with_resource(mut self, resource: impl Into<String>) -> Self {
        self.resource = Some(resource.into());
        self
    }

    /// Set the requirement reference
    pub fn with_requirement(mut self, requirement: impl Into<String>) -> Self {
        self.requirement_ref = Some(requirement.into());
        self
    }

    /// Set remediation steps
    pub fn with_remediation(mut self, remediation: impl Into<String>) -> Self {
        self.remediation = Some(remediation.into());
        self
    }
}

/// Type of compliance finding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingType {
    /// PII detected without proper handling
    PiiExposure,
    /// Consent missing or expired
    ConsentViolation,
    /// Data retained beyond allowed period
    RetentionViolation,
    /// Access control failure
    AccessViolation,
    /// Audit trail incomplete
    AuditGap,
    /// Data not properly anonymized
    AnonymizationFailure,
    /// Jurisdiction requirement violation
    JurisdictionViolation,
    /// Configuration issue
    ConfigurationIssue,
    /// Other/custom finding type
    Other,
}

/// Severity level for findings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Critical: Immediate action required
    Critical,
    /// High: Urgent attention needed
    High,
    /// Medium: Should be addressed soon
    Medium,
    /// Low: Address when convenient
    Low,
    /// Info: Informational only
    Info,
}

impl Severity {
    /// Get numeric priority (1 = highest)
    pub fn priority(&self) -> u8 {
        match self {
            Self::Critical => 1,
            Self::High => 2,
            Self::Medium => 3,
            Self::Low => 4,
            Self::Info => 5,
        }
    }
}

/// Recommended action for compliance improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Priority (1 = highest)
    pub priority: u8,
    /// Action title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Related finding IDs
    pub related_findings: Vec<String>,
    /// Estimated effort (optional)
    pub estimated_effort: Option<String>,
    /// Category
    pub category: RecommendationCategory,
}

impl Recommendation {
    /// Create a new recommendation
    pub fn new(
        priority: u8,
        title: impl Into<String>,
        description: impl Into<String>,
        category: RecommendationCategory,
    ) -> Self {
        Self {
            priority,
            title: title.into(),
            description: description.into(),
            related_findings: Vec::new(),
            estimated_effort: None,
            category,
        }
    }

    /// Add related findings
    pub fn with_findings(mut self, finding_ids: Vec<String>) -> Self {
        self.related_findings = finding_ids;
        self
    }

    /// Set estimated effort
    pub fn with_effort(mut self, effort: impl Into<String>) -> Self {
        self.estimated_effort = Some(effort.into());
        self
    }
}

/// Category of recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendationCategory {
    /// Process improvement
    Process,
    /// Technical fix
    Technical,
    /// Training/education
    Training,
    /// Policy update
    Policy,
    /// Configuration change
    Configuration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compliance_report_creation() {
        let report =
            ComplianceReport::new(Regulation::GDPR, NamespacePath::new("org.domain").unwrap());
        assert_eq!(report.regulation, Regulation::GDPR);
        assert!(report.is_compliant());
        assert_eq!(report.compliance_score(), 1.0);
    }

    #[test]
    fn test_add_finding() {
        let mut report =
            ComplianceReport::new(Regulation::GDPR, NamespacePath::new("org.domain").unwrap());

        let finding = ComplianceFinding::new(
            FindingType::PiiExposure,
            Severity::High,
            "PII Found",
            "Personal information detected in logs",
        );

        report.add_finding(finding);

        assert_eq!(report.summary.total_findings, 1);
        assert_eq!(report.summary.high_count, 1);
        assert!(!report.is_compliant());
    }

    #[test]
    fn test_compliance_score() {
        let mut report =
            ComplianceReport::new(Regulation::SOX, NamespacePath::new("org.domain").unwrap());

        // Add some findings
        report.add_finding(ComplianceFinding::new(
            FindingType::AuditGap,
            Severity::Medium,
            "Gap",
            "Description",
        ));
        report.add_finding(ComplianceFinding::new(
            FindingType::ConfigurationIssue,
            Severity::Low,
            "Config",
            "Description",
        ));

        // Score should be reduced but still above 0
        let score = report.compliance_score();
        assert!(score < 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_critical_finding_fails_compliance() {
        let mut report =
            ComplianceReport::new(Regulation::HIPAA, NamespacePath::new("org.domain").unwrap());

        report.add_finding(ComplianceFinding::new(
            FindingType::PiiExposure,
            Severity::Critical,
            "PHI Breach",
            "Protected health information exposed",
        ));

        assert!(!report.is_compliant());
        // Critical finding causes 50 point penalty, so score = max(0, 1 - 50/100) = 0.5
        // Since we want it to fail compliance, score should be <= 0.5
        assert!(report.compliance_score() <= 0.5);
    }

    #[test]
    fn test_finding_builder() {
        let finding = ComplianceFinding::new(
            FindingType::ConsentViolation,
            Severity::Medium,
            "Expired Consent",
            "User consent has expired",
        )
        .with_resource("user_123")
        .with_requirement("GDPR-6")
        .with_remediation("Request consent renewal");

        assert_eq!(finding.resource, Some("user_123".to_string()));
        assert_eq!(finding.requirement_ref, Some("GDPR-6".to_string()));
        assert!(finding.remediation.is_some());
    }

    #[test]
    fn test_recommendation_builder() {
        let rec = Recommendation::new(
            1,
            "Implement PII Detection",
            "Add automated PII detection to data pipeline",
            RecommendationCategory::Technical,
        )
        .with_findings(vec!["finding_1".to_string()])
        .with_effort("2 weeks");

        assert_eq!(rec.priority, 1);
        assert_eq!(rec.related_findings.len(), 1);
        assert_eq!(rec.estimated_effort, Some("2 weeks".to_string()));
    }

    #[test]
    fn test_severity_priority() {
        assert!(Severity::Critical.priority() < Severity::High.priority());
        assert!(Severity::High.priority() < Severity::Medium.priority());
        assert!(Severity::Medium.priority() < Severity::Low.priority());
        assert!(Severity::Low.priority() < Severity::Info.priority());
    }

    #[test]
    fn test_report_json_export() {
        let report =
            ComplianceReport::new(Regulation::CCPA, NamespacePath::new("org.domain").unwrap());

        let json = report.to_json().unwrap();
        assert!(json.contains("CCPA"));
        assert!(json.contains("generated_at"));
    }
}

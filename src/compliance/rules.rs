//! Compliance Rule Templates
//!
//! Pre-defined rule templates for regulatory compliance.
//!
//! ## Single-Head vs Multi-Head Rules
//!
//! - **Single-head rules**: Return one decision (e.g., `must_redact`). Use with `CompiledRule`.
//! - **Multi-head rules**: Return multiple independent decisions. Use with `MultiHeadCompiledRule`.
//!
//! Most templates provide both options:
//! - `PrivacyRules::REDACTION` → single head
//! - `PrivacyRules::MULTI_HEAD` → multiple heads

use crate::compiler::RuleSpec;
use crate::Result;

/// Privacy rule templates
pub struct PrivacyRules;

impl PrivacyRules {
    /// Basic redaction rules (single head: must_redact)
    ///
    /// ```prolog
    /// must_redact(X) :- pii_detected(X), not consent_valid(X).
    /// must_redact(X) :- pii_detected(X), consent_valid(X), sensitive_data(X).
    /// ```
    pub const REDACTION: &'static str = r#"
        must_redact(X) :- pii_detected(X), not consent_valid(X).
        must_redact(X) :- pii_detected(X), consent_valid(X), sensitive_data(X).
    "#;

    /// Consent-based access rules (single head: allow_access)
    ///
    /// ```prolog
    /// allow_access(X) :- access_authorized(X), consent_valid(X), jurisdiction_match(X).
    /// allow_access(X) :- access_authorized(X), consent_valid(X), not requires_jurisdiction(X).
    /// ```
    pub const CONSENT_ACCESS: &'static str = r#"
        allow_access(X) :- access_authorized(X), consent_valid(X), jurisdiction_match(X).
        allow_access(X) :- access_authorized(X), consent_valid(X), not requires_jurisdiction(X).
    "#;

    /// Parse redaction rules
    pub fn parse_redaction() -> Result<RuleSpec> {
        RuleSpec::parse("privacy_redaction", Self::REDACTION)
    }

    /// Parse consent access rules
    pub fn parse_consent_access() -> Result<RuleSpec> {
        RuleSpec::parse("privacy_consent", Self::CONSENT_ACCESS)
    }

    /// Multi-head privacy rules
    ///
    /// Returns independent decisions:
    /// - `must_redact`: Data must be redacted
    /// - `flag_for_review`: Needs privacy review
    /// - `allow_access`: Access is permitted
    /// - `require_consent`: Consent must be obtained
    ///
    /// ```prolog
    /// must_redact(X) :- pii_detected(X), not consent_valid(X).
    /// must_redact(X) :- pii_detected(X), sensitive_data(X).
    /// flag_for_review(X) :- pii_detected(X), consent_valid(X), not sensitive_data(X).
    /// allow_access(X) :- access_authorized(X), consent_valid(X), jurisdiction_match(X).
    /// require_consent(X) :- pii_detected(X), not consent_valid(X), not blocked(X).
    /// ```
    pub const MULTI_HEAD: &'static str = r#"
        must_redact(X) :- pii_detected(X), not consent_valid(X).
        must_redact(X) :- pii_detected(X), sensitive_data(X).
        flag_for_review(X) :- pii_detected(X), consent_valid(X), not sensitive_data(X).
        allow_access(X) :- access_authorized(X), consent_valid(X), jurisdiction_match(X).
        require_consent(X) :- pii_detected(X), not consent_valid(X), not blocked(X).
    "#;

    /// Parse multi-head privacy rules (use with `MultiHeadCompiledRule`)
    pub fn parse_multi_head() -> Result<RuleSpec> {
        RuleSpec::parse("privacy_multi", Self::MULTI_HEAD)
    }
}

/// Retention rule templates
pub struct RetentionRules;

impl RetentionRules {
    /// Data deletion rules (single head: must_delete)
    ///
    /// ```prolog
    /// must_delete(X) :- retention_exceeded(X), not legal_hold(X).
    /// must_delete(X) :- retention_exceeded(X), legal_hold(X), hold_expired(X).
    /// ```
    pub const DELETION: &'static str = r#"
        must_delete(X) :- retention_exceeded(X), not legal_hold(X).
        must_delete(X) :- retention_exceeded(X), legal_hold(X), hold_expired(X).
    "#;

    /// Data archival rules (single head: archive)
    ///
    /// ```prolog
    /// archive(X) :- not retention_exceeded(X), not recently_accessed(X),
    ///               not frequently_accessed(X).
    /// archive(X) :- not retention_exceeded(X), frequently_accessed(X),
    ///               archive_requested(X).
    /// ```
    pub const ARCHIVAL: &'static str = r#"
        archive(X) :- not retention_exceeded(X), not recently_accessed(X), 
                      not frequently_accessed(X).
        archive(X) :- not retention_exceeded(X), frequently_accessed(X), 
                      archive_requested(X).
    "#;

    /// Parse deletion rules
    pub fn parse_deletion() -> Result<RuleSpec> {
        RuleSpec::parse("retention_deletion", Self::DELETION)
    }

    /// Parse archival rules
    pub fn parse_archival() -> Result<RuleSpec> {
        RuleSpec::parse("retention_archival", Self::ARCHIVAL)
    }

    /// Multi-head retention rules
    ///
    /// Returns independent decisions:
    /// - `must_delete`: Data must be deleted
    /// - `archive`: Data should be archived
    /// - `keep_active`: Data should remain active
    ///
    /// ```prolog
    /// must_delete(X) :- retention_exceeded(X), not legal_hold(X).
    /// must_delete(X) :- retention_exceeded(X), legal_hold(X), hold_expired(X).
    /// archive(X) :- not retention_exceeded(X), not recently_accessed(X), not frequently_accessed(X).
    /// keep_active(X) :- not retention_exceeded(X), frequently_accessed(X).
    /// keep_active(X) :- not retention_exceeded(X), recently_accessed(X).
    /// ```
    pub const MULTI_HEAD: &'static str = r#"
        must_delete(X) :- retention_exceeded(X), not legal_hold(X).
        must_delete(X) :- retention_exceeded(X), legal_hold(X), hold_expired(X).
        archive(X) :- not retention_exceeded(X), not recently_accessed(X), not frequently_accessed(X).
        keep_active(X) :- not retention_exceeded(X), frequently_accessed(X).
        keep_active(X) :- not retention_exceeded(X), recently_accessed(X).
    "#;

    /// Parse multi-head retention rules (use with `MultiHeadCompiledRule`)
    pub fn parse_multi_head() -> Result<RuleSpec> {
        RuleSpec::parse("retention_multi", Self::MULTI_HEAD)
    }
}

/// Audit rule templates
pub struct AuditRules;

impl AuditRules {
    /// Audit readiness rules (single head: ready_for_audit)
    ///
    /// ```prolog
    /// ready_for_audit(X) :- audit_complete(X), documentation_current(X).
    /// ready_for_audit(X) :- audit_complete(X), not requires_documentation(X).
    /// ```
    pub const READINESS: &'static str = r#"
        ready_for_audit(X) :- audit_complete(X), documentation_current(X).
        ready_for_audit(X) :- audit_complete(X), not requires_documentation(X).
    "#;

    /// Parse audit readiness rules
    pub fn parse_readiness() -> Result<RuleSpec> {
        RuleSpec::parse("audit_readiness", Self::READINESS)
    }
}

/// Access control rule templates  
pub struct AccessRules;

impl AccessRules {
    /// Resource access rules (single head: allow_resource)
    ///
    /// ```prolog
    /// allow_resource(X) :- access_authorized(X), jurisdiction_match(X),
    ///                      not suspended(X).
    /// allow_resource(X) :- access_authorized(X), not requires_jurisdiction(X),
    ///                      not suspended(X).
    /// ```
    pub const RESOURCE_ACCESS: &'static str = r#"
        allow_resource(X) :- access_authorized(X), jurisdiction_match(X), 
                            not suspended(X).
        allow_resource(X) :- access_authorized(X), not requires_jurisdiction(X),
                            not suspended(X).
    "#;

    /// Parse resource access rules
    pub fn parse_resource_access() -> Result<RuleSpec> {
        RuleSpec::parse("access_resource", Self::RESOURCE_ACCESS)
    }
}

/// Complete compliance rule set
pub struct ComplianceRuleSet;

impl ComplianceRuleSet {
    /// Unified compliance decision
    ///
    /// Combines privacy, retention, and access checks into a single
    /// `compliant` decision.
    pub const COMPLETE: &'static str = r#"
        compliant(X) :- not pii_detected(X), not retention_exceeded(X), 
                        audit_complete(X).
        compliant(X) :- pii_detected(X), consent_valid(X), 
                        not retention_exceeded(X), audit_complete(X),
                        jurisdiction_match(X).
    "#;

    /// Parse complete compliance rules
    pub fn parse_complete() -> Result<RuleSpec> {
        RuleSpec::parse("compliance_complete", Self::COMPLETE)
    }

    /// Multi-head compliance rules
    ///
    /// Returns independent decisions for comprehensive compliance:
    /// - `compliant`: Overall compliance status
    /// - `must_redact`: Data must be redacted
    /// - `must_delete`: Data must be deleted
    /// - `requires_audit`: Audit action required
    /// - `breach_detected`: Potential breach detected
    ///
    /// Use with `MultiHeadCompiledRule` for full multi-output capability.
    pub const MULTI_HEAD: &'static str = r#"
        compliant(X) :- not pii_detected(X), not retention_exceeded(X), audit_complete(X).
        compliant(X) :- pii_detected(X), consent_valid(X), not retention_exceeded(X), 
                        audit_complete(X), jurisdiction_match(X).
        must_redact(X) :- pii_detected(X), not consent_valid(X).
        must_redact(X) :- pii_detected(X), sensitive_data(X).
        must_delete(X) :- retention_exceeded(X), not legal_hold(X).
        requires_audit(X) :- not audit_complete(X), access_authorized(X).
        requires_audit(X) :- pii_detected(X), not audit_complete(X).
        breach_detected(X) :- pii_detected(X), not consent_valid(X), access_authorized(X).
        breach_detected(X) :- retention_exceeded(X), not legal_hold(X), not deleted(X).
    "#;

    /// Parse multi-head compliance rules (use with `MultiHeadCompiledRule`)
    pub fn parse_multi_head() -> Result<RuleSpec> {
        RuleSpec::parse("compliance_multi", Self::MULTI_HEAD)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_privacy_redaction() {
        let spec = PrivacyRules::parse_redaction();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_privacy_consent() {
        let spec = PrivacyRules::parse_consent_access();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_retention_deletion() {
        let spec = RetentionRules::parse_deletion();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_retention_archival() {
        let spec = RetentionRules::parse_archival();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_audit_readiness() {
        let spec = AuditRules::parse_readiness();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_access_resource() {
        let spec = AccessRules::parse_resource_access();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_compliance_complete() {
        let spec = ComplianceRuleSet::parse_complete();
        assert!(spec.is_ok());
    }

    // Multi-head parsing tests
    #[test]
    fn test_parse_privacy_multi_head() {
        let spec = PrivacyRules::parse_multi_head();
        assert!(spec.is_ok());
        let spec = spec.unwrap();
        // Should have 4 heads: must_redact, flag_for_review, allow_access, require_consent
        let heads: std::collections::HashSet<_> =
            spec.rules.iter().map(|r| r.head.as_str()).collect();
        assert!(heads.contains("must_redact"));
        assert!(heads.contains("flag_for_review"));
        assert!(heads.contains("allow_access"));
        assert!(heads.contains("require_consent"));
    }

    #[test]
    fn test_parse_retention_multi_head() {
        let spec = RetentionRules::parse_multi_head();
        assert!(spec.is_ok());
        let spec = spec.unwrap();
        // Should have 3 heads: must_delete, archive, keep_active
        let heads: std::collections::HashSet<_> =
            spec.rules.iter().map(|r| r.head.as_str()).collect();
        assert!(heads.contains("must_delete"));
        assert!(heads.contains("archive"));
        assert!(heads.contains("keep_active"));
    }

    #[test]
    fn test_parse_compliance_multi_head() {
        let spec = ComplianceRuleSet::parse_multi_head();
        assert!(spec.is_ok());
        let spec = spec.unwrap();
        // Should have 5 heads
        let heads: std::collections::HashSet<_> =
            spec.rules.iter().map(|r| r.head.as_str()).collect();
        assert!(heads.contains("compliant"));
        assert!(heads.contains("must_redact"));
        assert!(heads.contains("must_delete"));
        assert!(heads.contains("requires_audit"));
        assert!(heads.contains("breach_detected"));
    }
}

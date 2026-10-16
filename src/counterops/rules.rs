//! CounterOps Rule Templates
//!
//! Pre-defined rule templates for security operations.

use crate::compiler::RuleSpec;
use crate::Result;

/// Defense rule templates
pub struct DefenseRules;

impl DefenseRules {
    /// Basic threat response: block on high threat
    ///
    /// ```prolog
    /// block_request(X) :- threat_level(X), not verified_admin(X).
    /// block_request(X) :- rate_limit_exceeded(X).
    /// ```
    pub const BASIC_BLOCK: &'static str = r#"
        block_request(X) :- threat_level(X), not verified_admin(X).
        block_request(X) :- rate_limit_exceeded(X).
    "#;

    /// Defense mode triggers
    ///
    /// ```prolog
    /// enter_defense_mode(X) :- threat_level(X), defense_mode(X).
    /// enter_defense_mode(X) :- anomaly_score(X), not known_safe_pattern(X).
    /// ```
    pub const DEFENSE_TRIGGERS: &'static str = r#"
        enter_defense_mode(X) :- threat_level(X), defense_mode(X).
        enter_defense_mode(X) :- anomaly_score(X), not known_safe_pattern(X).
    "#;

    /// Parse basic blocking rules
    pub fn parse_basic_block() -> Result<RuleSpec> {
        RuleSpec::parse("defense_block", Self::BASIC_BLOCK)
    }

    /// Parse defense mode triggers
    pub fn parse_defense_triggers() -> Result<RuleSpec> {
        RuleSpec::parse("defense_triggers", Self::DEFENSE_TRIGGERS)
    }
}

/// Attack detection rule templates
pub struct DetectionRules;

impl DetectionRules {
    /// Injection detection rules
    ///
    /// ```prolog
    /// block_for_injection(X) :- injection_attempt(X), not verified_admin(X).
    /// flag_for_review(X) :- injection_attempt(X), verified_admin(X).
    /// ```
    pub const INJECTION: &'static str = r#"
        block_for_injection(X) :- injection_attempt(X), not verified_admin(X).
        flag_for_review(X) :- injection_attempt(X), verified_admin(X).
    "#;

    /// Anomaly response rules
    ///
    /// ```prolog
    /// investigate(X) :- anomaly_score(X), not scheduled_change(X), business_hours(X).
    /// alert_on_call(X) :- anomaly_score(X), not scheduled_change(X), not business_hours(X).
    /// ```
    pub const ANOMALY_RESPONSE: &'static str = r#"
        investigate(X) :- anomaly_score(X), not scheduled_change(X), business_hours(X).
        alert_on_call(X) :- anomaly_score(X), not scheduled_change(X), not business_hours(X).
    "#;

    /// Data protection rules
    ///
    /// ```prolog
    /// block_exfil(X) :- data_exfil_risk(X), contains_sensitive(X).
    /// flag_exfil(X) :- data_exfil_risk(X), not contains_sensitive(X).
    /// ```
    pub const DATA_PROTECTION: &'static str = r#"
        block_exfil(X) :- data_exfil_risk(X), contains_sensitive(X).
        flag_exfil(X) :- data_exfil_risk(X), not contains_sensitive(X).
    "#;

    /// Parse injection detection rules
    pub fn parse_injection() -> Result<RuleSpec> {
        RuleSpec::parse("detection_injection", Self::INJECTION)
    }

    /// Parse anomaly response rules
    pub fn parse_anomaly_response() -> Result<RuleSpec> {
        RuleSpec::parse("detection_anomaly", Self::ANOMALY_RESPONSE)
    }

    /// Parse data protection rules
    pub fn parse_data_protection() -> Result<RuleSpec> {
        RuleSpec::parse("detection_data", Self::DATA_PROTECTION)
    }
}

/// Authentication rule templates
pub struct AuthRules;

impl AuthRules {
    /// Authentication anomaly rules
    ///
    /// ```prolog
    /// require_mfa(X) :- auth_anomaly(X), not mfa_verified(X).
    /// block_auth(X) :- auth_anomaly(X), repeated_failure(X).
    /// ```
    pub const AUTH_ANOMALY: &'static str = r#"
        require_mfa(X) :- auth_anomaly(X), not mfa_verified(X).
        block_auth(X) :- auth_anomaly(X), repeated_failure(X).
    "#;

    /// Parse authentication anomaly rules
    pub fn parse_auth_anomaly() -> Result<RuleSpec> {
        RuleSpec::parse("auth_anomaly", Self::AUTH_ANOMALY)
    }
}

/// Complete security rule set
pub struct SecurityRuleSet;

impl SecurityRuleSet {
    /// Unified security decision rules
    ///
    /// Combines threat detection, defense triggers, and response into
    /// a single `allow_request` decision.
    pub const COMPLETE: &'static str = r#"
        allow_request(X) :- not threat_level(X), not anomaly_score(X), 
                           not injection_attempt(X), not rate_limit_exceeded(X).
        allow_request(X) :- verified_admin(X), not data_exfil_risk(X).
    "#;

    /// Parse complete security rules
    pub fn parse_complete() -> Result<RuleSpec> {
        RuleSpec::parse("security_complete", Self::COMPLETE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_defense_basic() {
        let spec = DefenseRules::parse_basic_block();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_defense_triggers() {
        let spec = DefenseRules::parse_defense_triggers();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_injection() {
        let spec = DetectionRules::parse_injection();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_anomaly_response() {
        let spec = DetectionRules::parse_anomaly_response();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_data_protection() {
        let spec = DetectionRules::parse_data_protection();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_auth_anomaly() {
        let spec = AuthRules::parse_auth_anomaly();
        assert!(spec.is_ok());
    }

    #[test]
    fn test_parse_security_complete() {
        let spec = SecurityRuleSet::parse_complete();
        assert!(spec.is_ok());
    }
}

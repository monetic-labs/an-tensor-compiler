//! Compliance Context
//!
//! Manages compliance predicates and rules for regulatory requirements.
//!
//! ## Single-Head vs Multi-Head
//!
//! - Use `compile_*_rules()` methods for simple single-output decisions (`CompiledRule`)
//! - Use `compile_*_multi_head()` methods for multiple independent outputs (`MultiHeadCompiledRule`)

use std::collections::HashMap;

use crate::compiler::{CompiledRule, MultiHeadCompiledRule, PredicateSpec, RuleSpec};
use crate::{Device, Result, TensorCoreError};

use super::predicates::PrivacyPredicate;
use super::regulations::Regulation;
use super::rules::{AccessRules, AuditRules, ComplianceRuleSet, PrivacyRules, RetentionRules};

/// Compliance context managing predicates and rules
///
/// This is the primary entry point for using compliance primitives.
/// It manages predicate registration, rule compilation, and provides
/// convenient methods for common compliance patterns.
///
/// ## Usage
///
/// ```rust,ignore
/// use an_tensor_compiler::compliance::prelude::*;
///
/// // Single-head (simple)
/// let mut ctx = ComplianceContext::new(Device::Cpu, Regulation::GDPR)?;
/// ctx.register_standard_predicates()?;
/// let rules = ctx.compile_privacy_rules()?;
/// let output = rules.forward(&inputs)?;
///
/// // Multi-head (full)
/// let rules = ctx.compile_privacy_multi_head()?;
/// let outputs = rules.forward_all(&inputs)?;
/// let must_redact = outputs.get("must_redact");
/// let allow_access = outputs.get("allow_access");
/// ```
#[derive(Debug)]
pub struct ComplianceContext {
    /// Device for tensor operations
    device: Device,
    /// Regulatory framework
    regulation: Regulation,
    /// Registered predicates
    predicates: HashMap<String, PredicateSpec>,
    /// Compiled single-head rules (lazily populated)
    compiled_rules: HashMap<String, CompiledRule>,
    /// Compiled multi-head rules (lazily populated)
    compiled_multi_head: HashMap<String, MultiHeadCompiledRule>,
}

impl ComplianceContext {
    /// Create a new compliance context for a specific regulation
    pub fn new(device: Device, regulation: Regulation) -> Result<Self> {
        Ok(Self {
            device,
            regulation,
            predicates: HashMap::new(),
            compiled_rules: HashMap::new(),
            compiled_multi_head: HashMap::new(),
        })
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the regulation
    pub fn regulation(&self) -> Regulation {
        self.regulation
    }

    /// Register a predicate
    pub fn register_predicate(&mut self, name: &str, spec: PredicateSpec) -> Result<()> {
        self.predicates.insert(name.to_string(), spec);
        Ok(())
    }

    /// Register a standard privacy predicate
    pub fn register_standard(&mut self, predicate: PrivacyPredicate) -> Result<()> {
        let name = predicate.name().to_string();
        let spec = predicate.default_spec();
        self.predicates.insert(name, spec);
        Ok(())
    }

    /// Register all standard predicates
    ///
    /// Registers:
    /// - `pii_detected`: PII detection in content
    /// - `consent_valid`: User consent is valid
    /// - `retention_exceeded`: Data past retention period
    /// - `audit_complete`: Audit trail is complete
    /// - `anonymization_sufficient`: Data is properly anonymized
    /// - `jurisdiction_match`: Data is in correct jurisdiction
    /// - `access_authorized`: User authorized for access
    /// - `sensitive_data`: Sensitive data detected
    pub fn register_standard_predicates(&mut self) -> Result<()> {
        self.register_standard(PrivacyPredicate::PiiDetected)?;
        self.register_standard(PrivacyPredicate::ConsentValid)?;
        self.register_standard(PrivacyPredicate::RetentionExceeded)?;
        self.register_standard(PrivacyPredicate::AuditComplete)?;
        self.register_standard(PrivacyPredicate::AnonymizationSufficient)?;
        self.register_standard(PrivacyPredicate::JurisdictionMatch)?;
        self.register_standard(PrivacyPredicate::AccessAuthorized)?;
        self.register_standard(PrivacyPredicate::SensitiveData)?;
        Ok(())
    }

    /// Get the number of registered predicates
    pub fn predicate_count(&self) -> usize {
        self.predicates.len()
    }

    /// Get a registered predicate spec
    pub fn get_predicate(&self, name: &str) -> Option<&PredicateSpec> {
        self.predicates.get(name)
    }

    /// Compile privacy/redaction rules
    pub fn compile_privacy_rules(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("privacy") {
            let mut spec = PrivacyRules::parse_redaction()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules.insert("privacy".to_string(), compiled);
        }

        self.compiled_rules
            .get("privacy")
            .ok_or_else(|| TensorCoreError::Compiler("Privacy rules not found".into()))
    }

    /// Compile consent-based access rules
    pub fn compile_consent_rules(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("consent") {
            let mut spec = PrivacyRules::parse_consent_access()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules.insert("consent".to_string(), compiled);
        }

        self.compiled_rules
            .get("consent")
            .ok_or_else(|| TensorCoreError::Compiler("Consent rules not found".into()))
    }

    /// Compile retention/deletion rules
    pub fn compile_retention_rules(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("retention") {
            let mut spec = RetentionRules::parse_deletion()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules
                .insert("retention".to_string(), compiled);
        }

        self.compiled_rules
            .get("retention")
            .ok_or_else(|| TensorCoreError::Compiler("Retention rules not found".into()))
    }

    /// Compile audit readiness rules
    pub fn compile_audit_rules(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("audit") {
            let mut spec = AuditRules::parse_readiness()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules.insert("audit".to_string(), compiled);
        }

        self.compiled_rules
            .get("audit")
            .ok_or_else(|| TensorCoreError::Compiler("Audit rules not found".into()))
    }

    /// Compile access control rules
    pub fn compile_access_rules(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("access") {
            let mut spec = AccessRules::parse_resource_access()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules.insert("access".to_string(), compiled);
        }

        self.compiled_rules
            .get("access")
            .ok_or_else(|| TensorCoreError::Compiler("Access rules not found".into()))
    }

    /// Compile complete compliance rules
    ///
    /// Combines all compliance checks into a unified compliant/non-compliant decision.
    pub fn compile_complete(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("complete") {
            let mut spec = ComplianceRuleSet::parse_complete()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules.insert("complete".to_string(), compiled);
        }

        self.compiled_rules
            .get("complete")
            .ok_or_else(|| TensorCoreError::Compiler("Complete rules not found".into()))
    }

    // =========================================================================
    // Multi-Head Compilation Methods
    // =========================================================================

    /// Compile privacy rules with multiple independent outputs
    ///
    /// Returns a `MultiHeadCompiledRule` with:
    /// - `must_redact`: Data must be redacted
    /// - `flag_for_review`: Needs privacy review
    /// - `allow_access`: Access is permitted
    /// - `require_consent`: Consent must be obtained
    pub fn compile_privacy_multi_head(&mut self) -> Result<&MultiHeadCompiledRule> {
        if !self.compiled_multi_head.contains_key("privacy") {
            let mut spec = PrivacyRules::parse_multi_head()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = MultiHeadCompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_multi_head
                .insert("privacy".to_string(), compiled);
        }

        self.compiled_multi_head
            .get("privacy")
            .ok_or_else(|| TensorCoreError::Compiler("Multi-head privacy rules not found".into()))
    }

    /// Compile retention rules with multiple independent outputs
    ///
    /// Returns a `MultiHeadCompiledRule` with:
    /// - `must_delete`: Data must be deleted
    /// - `archive`: Data should be archived
    /// - `keep_active`: Data should remain active
    pub fn compile_retention_multi_head(&mut self) -> Result<&MultiHeadCompiledRule> {
        if !self.compiled_multi_head.contains_key("retention") {
            let mut spec = RetentionRules::parse_multi_head()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = MultiHeadCompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_multi_head
                .insert("retention".to_string(), compiled);
        }

        self.compiled_multi_head
            .get("retention")
            .ok_or_else(|| TensorCoreError::Compiler("Multi-head retention rules not found".into()))
    }

    /// Compile complete compliance rules with multiple independent outputs
    ///
    /// Returns a `MultiHeadCompiledRule` with:
    /// - `compliant`: Overall compliance status
    /// - `must_redact`: Data must be redacted
    /// - `must_delete`: Data must be deleted
    /// - `requires_audit`: Audit action required
    /// - `breach_detected`: Potential breach detected
    pub fn compile_complete_multi_head(&mut self) -> Result<&MultiHeadCompiledRule> {
        if !self.compiled_multi_head.contains_key("complete") {
            let mut spec = ComplianceRuleSet::parse_multi_head()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = MultiHeadCompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_multi_head
                .insert("complete".to_string(), compiled);
        }

        self.compiled_multi_head
            .get("complete")
            .ok_or_else(|| TensorCoreError::Compiler("Multi-head complete rules not found".into()))
    }

    /// Get a compiled multi-head rule by name
    pub fn get_multi_head(&self, name: &str) -> Option<&MultiHeadCompiledRule> {
        self.compiled_multi_head.get(name)
    }

    /// Compile custom rules
    ///
    /// Allows organisms to define their own rules while reusing
    /// registered predicates.
    pub fn compile_custom(&mut self, name: &str, rules: &str) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key(name) {
            let mut spec = RuleSpec::parse(name, rules)?;

            for (pred_name, pred_spec) in &self.predicates {
                spec.add_predicate(pred_name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules.insert(name.to_string(), compiled);
        }

        self.compiled_rules
            .get(name)
            .ok_or_else(|| TensorCoreError::Compiler(format!("Custom rules '{}' not found", name)))
    }

    /// Get compiled rule by name
    pub fn get_compiled(&self, name: &str) -> Option<&CompiledRule> {
        self.compiled_rules.get(name)
    }

    /// Clear all compiled rules (for recompilation after predicate changes)
    pub fn clear_compiled(&mut self) {
        self.compiled_rules.clear();
        self.compiled_multi_head.clear();
    }

    /// Get all trainable variables across all compiled rules
    pub fn trainable_vars(&self) -> Vec<crate::Var> {
        let single_head_vars = self
            .compiled_rules
            .values()
            .flat_map(|rule| rule.trainable_vars());

        let multi_head_vars = self
            .compiled_multi_head
            .values()
            .flat_map(|rule| rule.trainable_vars());

        single_head_vars.chain(multi_head_vars).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = ComplianceContext::new(Device::Cpu, Regulation::GDPR);
        assert!(ctx.is_ok());
        assert_eq!(ctx.unwrap().regulation(), Regulation::GDPR);
    }

    #[test]
    fn test_register_standard_predicates() {
        let mut ctx = ComplianceContext::new(Device::Cpu, Regulation::GDPR).unwrap();
        assert!(ctx.register_standard_predicates().is_ok());
        assert_eq!(ctx.predicate_count(), 8);
    }

    #[test]
    fn test_get_predicate() {
        let mut ctx = ComplianceContext::new(Device::Cpu, Regulation::GDPR).unwrap();
        ctx.register_standard_predicates().unwrap();

        assert!(ctx.get_predicate("pii_detected").is_some());
        assert!(ctx.get_predicate("nonexistent").is_none());
    }

    #[test]
    fn test_compile_privacy_rules() {
        let mut ctx = ComplianceContext::new(Device::Cpu, Regulation::GDPR).unwrap();
        ctx.register_standard_predicates().unwrap();

        let result = ctx.compile_privacy_rules();
        if let Err(e) = &result {
            eprintln!("Compile privacy rules error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_complete() {
        let mut ctx = ComplianceContext::new(Device::Cpu, Regulation::GDPR).unwrap();
        ctx.register_standard_predicates().unwrap();

        let result = ctx.compile_complete();
        if let Err(e) = &result {
            eprintln!("Compile complete error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_custom() {
        let mut ctx = ComplianceContext::new(Device::Cpu, Regulation::SOX).unwrap();
        ctx.register_standard_predicates().unwrap();

        let custom_rules = r#"
            my_rule(X) :- pii_detected(X), consent_valid(X).
        "#;

        let result = ctx.compile_custom("my_custom", custom_rules);
        assert!(result.is_ok());
    }

    #[test]
    fn test_clear_compiled() {
        let mut ctx = ComplianceContext::new(Device::Cpu, Regulation::HIPAA).unwrap();
        ctx.register_standard_predicates().unwrap();
        ctx.compile_privacy_rules().unwrap();

        assert!(ctx.get_compiled("privacy").is_some());

        ctx.clear_compiled();

        assert!(ctx.get_compiled("privacy").is_none());
    }

    #[test]
    fn test_different_regulations() {
        for regulation in Regulation::all() {
            let mut ctx = ComplianceContext::new(Device::Cpu, *regulation).unwrap();
            ctx.register_standard_predicates().unwrap();
            assert_eq!(ctx.regulation(), *regulation);
        }
    }

    // Multi-head compilation tests
    #[test]
    fn test_compile_privacy_multi_head() {
        let mut ctx = ComplianceContext::new(Device::Cpu, Regulation::GDPR).unwrap();
        ctx.register_standard_predicates().unwrap();

        let result = ctx.compile_privacy_multi_head();
        if let Err(e) = &result {
            eprintln!("Compile privacy multi-head error: {:?}", e);
        }
        assert!(result.is_ok());

        // Verify it's cached
        assert!(ctx.get_multi_head("privacy").is_some());
    }

    #[test]
    fn test_compile_retention_multi_head() {
        let mut ctx = ComplianceContext::new(Device::Cpu, Regulation::GDPR).unwrap();
        ctx.register_standard_predicates().unwrap();

        let result = ctx.compile_retention_multi_head();
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_complete_multi_head() {
        let mut ctx = ComplianceContext::new(Device::Cpu, Regulation::GDPR).unwrap();
        ctx.register_standard_predicates().unwrap();

        let result = ctx.compile_complete_multi_head();
        if let Err(e) = &result {
            eprintln!("Compile complete multi-head error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_clear_compiled_includes_multi_head() {
        let mut ctx = ComplianceContext::new(Device::Cpu, Regulation::HIPAA).unwrap();
        ctx.register_standard_predicates().unwrap();
        ctx.compile_privacy_rules().unwrap();
        ctx.compile_privacy_multi_head().unwrap();

        assert!(ctx.get_compiled("privacy").is_some());
        assert!(ctx.get_multi_head("privacy").is_some());

        ctx.clear_compiled();

        assert!(ctx.get_compiled("privacy").is_none());
        assert!(ctx.get_multi_head("privacy").is_none());
    }
}

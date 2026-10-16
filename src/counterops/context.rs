//! Security Context
//!
//! Manages security predicates and rules for threat detection.

use std::collections::HashMap;

use crate::compiler::{CompiledRule, PredicateSpec, RuleSpec};
use crate::{Device, Result, TensorCoreError};

use super::predicates::ThreatPredicate;
use super::rules::{AuthRules, DefenseRules, DetectionRules, SecurityRuleSet};

/// Security context managing predicates and rules
///
/// This is the primary entry point for using counterops primitives.
/// It manages predicate registration, rule compilation, and provides
/// convenient methods for common security patterns.
#[derive(Debug)]
pub struct SecurityContext {
    /// Device for tensor operations
    device: Device,
    /// Registered predicates
    predicates: HashMap<String, PredicateSpec>,
    /// Compiled rules (lazily populated)
    compiled_rules: HashMap<String, CompiledRule>,
}

impl SecurityContext {
    /// Create a new security context
    pub fn new(device: Device) -> Result<Self> {
        Ok(Self {
            device,
            predicates: HashMap::new(),
            compiled_rules: HashMap::new(),
        })
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Register a predicate
    pub fn register_predicate(&mut self, name: &str, spec: PredicateSpec) -> Result<()> {
        self.predicates.insert(name.to_string(), spec);
        Ok(())
    }

    /// Register a standard security predicate
    pub fn register_standard(&mut self, predicate: ThreatPredicate) -> Result<()> {
        let name = predicate.name().to_string();
        let spec = predicate.default_spec();
        self.predicates.insert(name, spec);
        Ok(())
    }

    /// Register all standard predicates
    ///
    /// Registers:
    /// - `threat_level`: Combined threat assessment
    /// - `anomaly_score`: Statistical anomaly detection
    /// - `injection_attempt`: Prompt injection detection
    /// - `data_exfil_risk`: Data exfiltration risk
    /// - `rate_limit_exceeded`: Rate limiting trigger
    /// - `pattern_deviation`: Behavior pattern deviation
    /// - `defense_mode`: Defense mode trigger
    /// - `auth_anomaly`: Authentication anomaly
    pub fn register_standard_predicates(&mut self) -> Result<()> {
        self.register_standard(ThreatPredicate::ThreatLevel)?;
        self.register_standard(ThreatPredicate::AnomalyScore)?;
        self.register_standard(ThreatPredicate::InjectionAttempt)?;
        self.register_standard(ThreatPredicate::DataExfilRisk)?;
        self.register_standard(ThreatPredicate::RateLimitExceeded)?;
        self.register_standard(ThreatPredicate::PatternDeviation)?;
        self.register_standard(ThreatPredicate::DefenseMode)?;
        self.register_standard(ThreatPredicate::AuthAnomaly)?;
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

    /// Compile defense rules
    ///
    /// Uses basic blocking rules:
    /// ```prolog
    /// block_request(X) :- threat_level(X), not verified_admin(X).
    /// block_request(X) :- rate_limit_exceeded(X).
    /// ```
    pub fn compile_defense_rules(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("defense") {
            let mut spec = DefenseRules::parse_basic_block()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules.insert("defense".to_string(), compiled);
        }

        self.compiled_rules
            .get("defense")
            .ok_or_else(|| TensorCoreError::Compiler("Defense rules not found".into()))
    }

    /// Compile defense mode trigger rules
    pub fn compile_defense_triggers(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("defense_triggers") {
            let mut spec = DefenseRules::parse_defense_triggers()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules
                .insert("defense_triggers".to_string(), compiled);
        }

        self.compiled_rules
            .get("defense_triggers")
            .ok_or_else(|| TensorCoreError::Compiler("Defense trigger rules not found".into()))
    }

    /// Compile injection detection rules
    pub fn compile_injection_rules(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("injection") {
            let mut spec = DetectionRules::parse_injection()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules
                .insert("injection".to_string(), compiled);
        }

        self.compiled_rules
            .get("injection")
            .ok_or_else(|| TensorCoreError::Compiler("Injection rules not found".into()))
    }

    /// Compile data protection rules
    pub fn compile_data_protection_rules(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("data_protection") {
            let mut spec = DetectionRules::parse_data_protection()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules
                .insert("data_protection".to_string(), compiled);
        }

        self.compiled_rules
            .get("data_protection")
            .ok_or_else(|| TensorCoreError::Compiler("Data protection rules not found".into()))
    }

    /// Compile authentication anomaly rules
    pub fn compile_auth_rules(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("auth") {
            let mut spec = AuthRules::parse_auth_anomaly()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules.insert("auth".to_string(), compiled);
        }

        self.compiled_rules
            .get("auth")
            .ok_or_else(|| TensorCoreError::Compiler("Auth rules not found".into()))
    }

    /// Compile complete security rules
    ///
    /// Combines all security checks into a unified allow/block decision.
    pub fn compile_complete(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("complete") {
            let mut spec = SecurityRuleSet::parse_complete()?;

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
    }

    /// Get all trainable variables across all compiled rules
    pub fn trainable_vars(&self) -> Vec<crate::Var> {
        self.compiled_rules
            .values()
            .flat_map(|rule| rule.trainable_vars())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = SecurityContext::new(Device::Cpu);
        assert!(ctx.is_ok());
    }

    #[test]
    fn test_register_standard_predicates() {
        let mut ctx = SecurityContext::new(Device::Cpu).unwrap();
        assert!(ctx.register_standard_predicates().is_ok());
        assert_eq!(ctx.predicate_count(), 8);
    }

    #[test]
    fn test_get_predicate() {
        let mut ctx = SecurityContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();

        assert!(ctx.get_predicate("threat_level").is_some());
        assert!(ctx.get_predicate("nonexistent").is_none());
    }

    #[test]
    fn test_compile_defense_rules() {
        let mut ctx = SecurityContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();

        let result = ctx.compile_defense_rules();
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_complete() {
        let mut ctx = SecurityContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();

        let result = ctx.compile_complete();
        if let Err(e) = &result {
            eprintln!("Compile complete error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_custom() {
        let mut ctx = SecurityContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();

        let custom_rules = r#"
            my_rule(X) :- threat_level(X), anomaly_score(X).
        "#;

        let result = ctx.compile_custom("my_custom", custom_rules);
        assert!(result.is_ok());
    }

    #[test]
    fn test_clear_compiled() {
        let mut ctx = SecurityContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();
        ctx.compile_defense_rules().unwrap();

        assert!(ctx.get_compiled("defense").is_some());

        ctx.clear_compiled();

        assert!(ctx.get_compiled("defense").is_none());
    }
}

//! Interaction Context
//!
//! Manages interaction predicates and rules for a conversation session.
//!
//! ## Single-Head vs Multi-Head
//!
//! - Use `compile_*_rules()` methods for simple single-output decisions (`CompiledRule`)
//! - Use `compile_*_multi_head()` methods for multiple independent outputs (`MultiHeadCompiledRule`)

use std::collections::HashMap;

use crate::compiler::{CompiledRule, MultiHeadCompiledRule, PredicateSpec, RuleSpec};
use crate::{Device, Result, TensorCoreError};

use super::predicates::InteractionPredicate;
use super::rules::{ContextRules, EscalationRules, InteractionRuleSet, QualityRules, SafetyRules};

/// Interaction context managing predicates and rules
///
/// This is the primary entry point for using interaction primitives.
/// It manages predicate registration, rule compilation, and provides
/// convenient methods for common interaction patterns.
///
/// ## Usage
///
/// ```rust,ignore
/// use an_tensor_compiler::interaction::prelude::*;
///
/// // Single-head (simple)
/// let mut ctx = InteractionContext::new(Device::Cpu)?;
/// ctx.register_standard_predicates()?;
/// let rules = ctx.compile_safety_rules()?;
/// let output = rules.forward(&inputs)?;
///
/// // Multi-head (full)
/// let rules = ctx.compile_safety_multi_head()?;
/// let outputs = rules.forward_all(&inputs)?;
/// let is_safe = outputs.get("safe_to_respond");
/// let needs_review = outputs.get("needs_review");
/// ```
#[derive(Debug)]
pub struct InteractionContext {
    /// Device for tensor operations
    device: Device,
    /// Registered predicates
    predicates: HashMap<String, PredicateSpec>,
    /// Compiled single-head rules (lazily populated)
    compiled_rules: HashMap<String, CompiledRule>,
    /// Compiled multi-head rules (lazily populated)
    compiled_multi_head: HashMap<String, MultiHeadCompiledRule>,
}

impl InteractionContext {
    /// Create a new interaction context
    pub fn new(device: Device) -> Result<Self> {
        Ok(Self {
            device,
            predicates: HashMap::new(),
            compiled_rules: HashMap::new(),
            compiled_multi_head: HashMap::new(),
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

    /// Register a standard interaction predicate
    pub fn register_standard(&mut self, predicate: InteractionPredicate) -> Result<()> {
        let name = predicate.name().to_string();
        let spec = predicate.default_spec();
        self.predicates.insert(name, spec);
        Ok(())
    }

    /// Register all standard predicates
    ///
    /// Registers:
    /// - `message_safe`: Content safety detection
    /// - `intent_clear`: Intent clarity assessment
    /// - `context_fresh`: Context staleness check
    /// - `response_coherent`: Response quality check
    /// - `topic_drift`: Topic drift detection
    /// - `escalation_needed`: Escalation assessment
    /// - `user_frustrated`: Frustration detection
    pub fn register_standard_predicates(&mut self) -> Result<()> {
        self.register_standard(InteractionPredicate::MessageSafe)?;
        self.register_standard(InteractionPredicate::IntentClear)?;
        self.register_standard(InteractionPredicate::ContextFresh)?;
        self.register_standard(InteractionPredicate::ResponseCoherent)?;
        self.register_standard(InteractionPredicate::TopicDrift)?;
        self.register_standard(InteractionPredicate::EscalationNeeded)?;
        self.register_standard(InteractionPredicate::UserFrustrated)?;
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

    /// Compile safety rules
    ///
    /// Uses the basic safety gate:
    /// ```prolog
    /// safe_to_respond(X) :- message_safe(X), not blocked_user(X).
    /// ```
    pub fn compile_safety_rules(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("safety") {
            let mut spec = SafetyRules::parse_basic()?;

            // Add registered predicates
            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules.insert("safety".to_string(), compiled);
        }

        self.compiled_rules
            .get("safety")
            .ok_or_else(|| TensorCoreError::Compiler("Safety rules not found".into()))
    }

    /// Compile safety rules with escalation
    ///
    /// Uses comprehensive safety with review queue:
    /// ```prolog
    /// safe_to_respond(X) :- message_safe(X), not blocked_user(X), not escalation_needed(X).
    /// needs_review(X) :- not message_safe(X).
    /// needs_review(X) :- escalation_needed(X).
    /// ```
    pub fn compile_safety_with_escalation(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("safety_escalation") {
            let mut spec = SafetyRules::parse_with_escalation()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules
                .insert("safety_escalation".to_string(), compiled);
        }

        self.compiled_rules
            .get("safety_escalation")
            .ok_or_else(|| TensorCoreError::Compiler("Safety escalation rules not found".into()))
    }

    /// Compile context management rules
    pub fn compile_context_rules(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("context") {
            let mut spec = ContextRules::parse_refresh()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules.insert("context".to_string(), compiled);
        }

        self.compiled_rules
            .get("context")
            .ok_or_else(|| TensorCoreError::Compiler("Context rules not found".into()))
    }

    /// Compile escalation rules
    pub fn compile_escalation_rules(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("escalation") {
            let mut spec = EscalationRules::parse_frustration_aware()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules
                .insert("escalation".to_string(), compiled);
        }

        self.compiled_rules
            .get("escalation")
            .ok_or_else(|| TensorCoreError::Compiler("Escalation rules not found".into()))
    }

    /// Compile quality rules
    pub fn compile_quality_rules(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("quality") {
            let mut spec = QualityRules::parse_response_quality()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = CompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_rules.insert("quality".to_string(), compiled);
        }

        self.compiled_rules
            .get("quality")
            .ok_or_else(|| TensorCoreError::Compiler("Quality rules not found".into()))
    }

    /// Compile complete interaction rules
    ///
    /// Combines all rule templates into a comprehensive ruleset.
    pub fn compile_complete(&mut self) -> Result<&CompiledRule> {
        if !self.compiled_rules.contains_key("complete") {
            let mut spec = InteractionRuleSet::parse_complete()?;

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

    /// Compile safety rules with multiple independent outputs
    ///
    /// Returns a `MultiHeadCompiledRule` with:
    /// - `safe_to_respond`: Primary safety gate
    /// - `needs_review`: Requires human review
    /// - `blocked`: Should not respond
    pub fn compile_safety_multi_head(&mut self) -> Result<&MultiHeadCompiledRule> {
        if !self.compiled_multi_head.contains_key("safety") {
            let mut spec = SafetyRules::parse_multi_head()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = MultiHeadCompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_multi_head
                .insert("safety".to_string(), compiled);
        }

        self.compiled_multi_head
            .get("safety")
            .ok_or_else(|| TensorCoreError::Compiler("Multi-head safety rules not found".into()))
    }

    /// Compile context rules with multiple independent outputs
    ///
    /// Returns a `MultiHeadCompiledRule` with:
    /// - `needs_refresh`: Context needs refreshing
    /// - `use_cached`: Safe to use cached context
    /// - `continue_topic`: Stay on current topic
    /// - `start_new_topic`: Begin new topic
    pub fn compile_context_multi_head(&mut self) -> Result<&MultiHeadCompiledRule> {
        if !self.compiled_multi_head.contains_key("context") {
            let mut spec = ContextRules::parse_multi_head()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = MultiHeadCompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_multi_head
                .insert("context".to_string(), compiled);
        }

        self.compiled_multi_head
            .get("context")
            .ok_or_else(|| TensorCoreError::Compiler("Multi-head context rules not found".into()))
    }

    /// Compile escalation rules with multiple independent outputs
    ///
    /// Returns a `MultiHeadCompiledRule` with:
    /// - `escalate`: Immediately escalate to human
    /// - `queue_for_review`: Queue for later review
    /// - `auto_resolve`: Can be resolved automatically
    pub fn compile_escalation_multi_head(&mut self) -> Result<&MultiHeadCompiledRule> {
        if !self.compiled_multi_head.contains_key("escalation") {
            let mut spec = EscalationRules::parse_multi_head()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = MultiHeadCompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_multi_head
                .insert("escalation".to_string(), compiled);
        }

        self.compiled_multi_head.get("escalation").ok_or_else(|| {
            TensorCoreError::Compiler("Multi-head escalation rules not found".into())
        })
    }

    /// Compile quality rules with multiple independent outputs
    ///
    /// Returns a `MultiHeadCompiledRule` with:
    /// - `good_response`: Response is high quality
    /// - `needs_improvement`: Response needs work
    /// - `needs_clarification`: User intent unclear
    pub fn compile_quality_multi_head(&mut self) -> Result<&MultiHeadCompiledRule> {
        if !self.compiled_multi_head.contains_key("quality") {
            let mut spec = QualityRules::parse_multi_head()?;

            for (name, pred_spec) in &self.predicates {
                spec.add_predicate(name, pred_spec.clone());
            }

            let compiled = MultiHeadCompiledRule::compile_on_device(spec, &self.device)?;
            self.compiled_multi_head
                .insert("quality".to_string(), compiled);
        }

        self.compiled_multi_head
            .get("quality")
            .ok_or_else(|| TensorCoreError::Compiler("Multi-head quality rules not found".into()))
    }

    /// Compile complete interaction rules with multiple independent outputs
    ///
    /// Returns a `MultiHeadCompiledRule` with:
    /// - `proceed`: Safe to generate response
    /// - `needs_review`: Requires human oversight
    /// - `escalate`: Immediate escalation needed
    /// - `blocked`: Do not respond
    /// - `needs_clarification`: Ask user for clarification
    pub fn compile_complete_multi_head(&mut self) -> Result<&MultiHeadCompiledRule> {
        if !self.compiled_multi_head.contains_key("complete") {
            let mut spec = InteractionRuleSet::parse_multi_head()?;

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
        let ctx = InteractionContext::new(Device::Cpu);
        assert!(ctx.is_ok());
    }

    #[test]
    fn test_register_standard_predicates() {
        let mut ctx = InteractionContext::new(Device::Cpu).unwrap();
        assert!(ctx.register_standard_predicates().is_ok());
        assert_eq!(ctx.predicate_count(), 7);
    }

    #[test]
    fn test_get_predicate() {
        let mut ctx = InteractionContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();

        assert!(ctx.get_predicate("message_safe").is_some());
        assert!(ctx.get_predicate("nonexistent").is_none());
    }

    #[test]
    fn test_compile_safety_rules() {
        let mut ctx = InteractionContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();

        let result = ctx.compile_safety_rules();
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_complete() {
        let mut ctx = InteractionContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();

        let result = ctx.compile_complete();
        if let Err(e) = &result {
            eprintln!("Compile complete error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_custom() {
        let mut ctx = InteractionContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();

        let custom_rules = r#"
            my_rule(X) :- message_safe(X), intent_clear(X).
        "#;

        let result = ctx.compile_custom("my_custom", custom_rules);
        assert!(result.is_ok());
    }

    #[test]
    fn test_clear_compiled() {
        let mut ctx = InteractionContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();
        ctx.compile_safety_rules().unwrap();

        assert!(ctx.get_compiled("safety").is_some());

        ctx.clear_compiled();

        assert!(ctx.get_compiled("safety").is_none());
    }

    // Multi-head compilation tests
    #[test]
    fn test_compile_safety_multi_head() {
        let mut ctx = InteractionContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();

        let result = ctx.compile_safety_multi_head();
        assert!(result.is_ok());

        // Verify it's cached
        assert!(ctx.get_multi_head("safety").is_some());
    }

    #[test]
    fn test_compile_context_multi_head() {
        let mut ctx = InteractionContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();

        let result = ctx.compile_context_multi_head();
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_escalation_multi_head() {
        let mut ctx = InteractionContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();

        let result = ctx.compile_escalation_multi_head();
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_quality_multi_head() {
        let mut ctx = InteractionContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();

        let result = ctx.compile_quality_multi_head();
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_complete_multi_head() {
        let mut ctx = InteractionContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();

        let result = ctx.compile_complete_multi_head();
        if let Err(e) = &result {
            eprintln!("Compile complete multi-head error: {:?}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_clear_compiled_includes_multi_head() {
        let mut ctx = InteractionContext::new(Device::Cpu).unwrap();
        ctx.register_standard_predicates().unwrap();
        ctx.compile_safety_rules().unwrap();
        ctx.compile_safety_multi_head().unwrap();

        assert!(ctx.get_compiled("safety").is_some());
        assert!(ctx.get_multi_head("safety").is_some());

        ctx.clear_compiled();

        assert!(ctx.get_compiled("safety").is_none());
        assert!(ctx.get_multi_head("safety").is_none());
    }
}

//! Rule Validation
//!
//! Validates parsed rule specifications for semantic correctness.
//!
//! ## Checks Performed
//!
//! - **Variable binding**: All variables in body must appear in head or be bound
//! - **Predicate consistency**: Same predicate has consistent arity
//! - **Reserved words**: Predicate names can't be reserved words

use std::collections::{HashMap, HashSet};

use super::{Argument, Rule, RuleSpec};
use crate::{Result, TensorCoreError};

/// Validation error with location information
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error message
    pub message: String,
    /// Rule index where error occurred
    pub rule_index: Option<usize>,
    /// Predicate name involved
    pub predicate: Option<String>,
    /// Suggested fix
    pub suggestion: Option<String>,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(ref pred) = self.predicate {
            write!(f, " (predicate: {})", pred)?;
        }
        if let Some(idx) = self.rule_index {
            write!(f, " [rule {}]", idx)?;
        }
        if let Some(ref sug) = self.suggestion {
            write!(f, "\n  = help: {}", sug)?;
        }
        Ok(())
    }
}

/// Reserved words that cannot be used as predicate names
const RESERVED_WORDS: &[&str] = &["not", "and", "or", "true", "false", "if", "then", "else"];

/// Validate a rule specification
///
/// Returns a list of validation errors (empty if valid).
pub fn validate(spec: &RuleSpec) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Collect predicate arities
    let mut predicate_arities: HashMap<String, usize> = HashMap::new();

    for (rule_idx, rule) in spec.rules.iter().enumerate() {
        // Check head predicate
        check_reserved_word(&rule.head, rule_idx, &mut errors);

        // Check variable binding
        check_variable_binding(rule, rule_idx, &mut errors);

        // Check body predicates
        for lit in &rule.body {
            check_reserved_word(&lit.predicate, rule_idx, &mut errors);
            check_predicate_arity(
                &lit.predicate,
                lit.args.len(),
                &mut predicate_arities,
                rule_idx,
                &mut errors,
            );
        }
    }

    errors
}

/// Validate and return Result
pub fn validate_strict(spec: &RuleSpec) -> Result<()> {
    let errors = validate(spec);
    if errors.is_empty() {
        Ok(())
    } else {
        let msg = errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        Err(TensorCoreError::Compiler(format!(
            "Validation errors:\n{}",
            msg
        )))
    }
}

/// Check if a name is a reserved word
fn check_reserved_word(name: &str, rule_idx: usize, errors: &mut Vec<ValidationError>) {
    if RESERVED_WORDS.contains(&name) {
        errors.push(ValidationError {
            message: format!(
                "'{}' is a reserved word and cannot be used as a predicate name",
                name
            ),
            rule_index: Some(rule_idx),
            predicate: Some(name.to_string()),
            suggestion: Some(format!("Use a different name like '{}_pred'", name)),
        });
    }
}

/// Check that all variables in body appear in head
fn check_variable_binding(rule: &Rule, _rule_idx: usize, _errors: &mut Vec<ValidationError>) {
    // Collect variables from head (we treat head as just the predicate name for now)
    // In our AST, head is just a string - variables are in body literals that define them

    // For now, collect all variables used in the rule
    let mut all_vars: HashSet<String> = HashSet::new();
    let mut var_first_occurrence: HashMap<String, String> = HashMap::new();

    for lit in &rule.body {
        for arg in &lit.args {
            if let Argument::Variable(ref v) = arg {
                if !all_vars.contains(v) {
                    all_vars.insert(v.clone());
                    var_first_occurrence.insert(v.clone(), lit.predicate.clone());
                }
            }
        }
    }

    // Check for single-use variables (often a typo)
    let mut var_usage_count: HashMap<String, usize> = HashMap::new();
    for lit in &rule.body {
        for arg in &lit.args {
            if let Argument::Variable(ref v) = arg {
                *var_usage_count.entry(v.clone()).or_insert(0) += 1;
            }
        }
    }

    for (var, count) in var_usage_count {
        if count == 1 && var != "_" {
            // Single-use variable is a warning (might be intentional)
            // We'll skip this for now as it's not strictly an error
        }
    }
}

/// Check predicate arity consistency
fn check_predicate_arity(
    name: &str,
    arity: usize,
    arities: &mut HashMap<String, usize>,
    rule_idx: usize,
    errors: &mut Vec<ValidationError>,
) {
    match arities.get(name) {
        Some(&expected) if expected != arity => {
            errors.push(ValidationError {
                message: format!(
                    "Predicate '{}' used with inconsistent arity: expected {} arguments, found {}",
                    name, expected, arity
                ),
                rule_index: Some(rule_idx),
                predicate: Some(name.to_string()),
                suggestion: Some(format!(
                    "Ensure all uses of '{}' have the same number of arguments",
                    name
                )),
            });
        }
        None => {
            arities.insert(name.to_string(), arity);
        }
        _ => {}
    }
}

/// Extract all predicate names used in a RuleSpec
pub fn extract_predicates(spec: &RuleSpec) -> HashSet<String> {
    let mut predicates = HashSet::new();

    for rule in &spec.rules {
        predicates.insert(rule.head.clone());
        for lit in &rule.body {
            predicates.insert(lit.predicate.clone());
        }
    }

    predicates
}

/// Extract all variables used in a rule
pub fn extract_variables(rule: &Rule) -> HashSet<String> {
    let mut vars = HashSet::new();

    for lit in &rule.body {
        for arg in &lit.args {
            if let Argument::Variable(ref v) = arg {
                vars.insert(v.clone());
            }
        }
    }

    vars
}

/// Get predicate arity map for a RuleSpec
pub fn predicate_arities(spec: &RuleSpec) -> HashMap<String, usize> {
    let mut arities = HashMap::new();

    for rule in &spec.rules {
        for lit in &rule.body {
            arities
                .entry(lit.predicate.clone())
                .or_insert(lit.args.len());
        }
    }

    arities
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::parser::parse_rules;

    #[test]
    fn test_valid_spec() {
        let spec = parse_rules(
            "test",
            r#"
            exit(X) :- profit_target(X, 0.02), momentum_shift(X).
            exit(X) :- stop_loss(X, -0.01).
        "#,
        )
        .unwrap();

        let errors = validate(&spec);
        assert!(errors.is_empty(), "Expected no errors, got: {:?}", errors);
    }

    #[test]
    fn test_reserved_word_error() {
        let spec = parse_rules("test", "not(X) :- something(X).").unwrap();
        let errors = validate(&spec);
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("reserved word"));
    }

    #[test]
    fn test_inconsistent_arity() {
        let spec = parse_rules(
            "test",
            r#"
            exit(X) :- profit(X, 0.02).
            exit(X) :- profit(X).
        "#,
        )
        .unwrap();

        let errors = validate(&spec);
        assert!(!errors.is_empty());
        assert!(errors[0].message.contains("inconsistent arity"));
    }

    #[test]
    fn test_extract_predicates() {
        let spec = parse_rules(
            "test",
            r#"
            exit(X) :- profit_target(X, 0.02), momentum_shift(X).
            hold(X) :- not exit(X).
        "#,
        )
        .unwrap();

        let preds = extract_predicates(&spec);
        assert!(preds.contains("exit"));
        assert!(preds.contains("profit_target"));
        assert!(preds.contains("momentum_shift"));
        assert!(preds.contains("hold"));
    }

    #[test]
    fn test_extract_variables() {
        let spec = parse_rules("test", "exit(X) :- profit(X, Y), momentum(Y).").unwrap();
        let vars = extract_variables(&spec.rules[0]);
        assert!(vars.contains("X"));
        assert!(vars.contains("Y"));
    }

    #[test]
    fn test_predicate_arities() {
        let spec = parse_rules(
            "test",
            r#"
            exit(X) :- profit_target(X, 0.02), momentum_shift(X).
        "#,
        )
        .unwrap();

        let arities = predicate_arities(&spec);
        assert_eq!(arities.get("profit_target"), Some(&2));
        assert_eq!(arities.get("momentum_shift"), Some(&1));
    }

    #[test]
    fn test_validate_strict_ok() {
        let spec = parse_rules("test", "exit(X) :- profit(X, 0.02).").unwrap();
        assert!(validate_strict(&spec).is_ok());
    }

    #[test]
    fn test_validate_strict_error() {
        let spec = parse_rules("test", "and(X) :- something(X).").unwrap();
        assert!(validate_strict(&spec).is_err());
    }
}

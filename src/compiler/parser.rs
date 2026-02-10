//! Rule DSL Parser
//!
//! Parses Tensor Logic rule specifications from text into AST.
//!
//! ## Supported Syntax
//!
//! ### Facts (unconditional truths)
//! ```text
//! panel_access(chat).
//! authenticated.
//! ```
//!
//! ### Prolog-style rules (ASCII)
//! ```text
//! exit(X) :- profit_target(X, 0.02), momentum_shift(X).
//! escalate(X) :- low_intent(X); high_risk(X).
//! hold(X) :- not exit(X), position_open(X).
//! ```
//!
//! ### Unicode style
//! ```text
//! exit(X) ← profit_target(X, 0.02) ∧ momentum_shift(X)
//! escalate(X) ← low_intent(X) ∨ high_risk(X)
//! hold(X) ← ¬exit(X) ∧ position_open(X)
//! ```
//!
//! ### Atom arguments
//! ```text
//! panel_access(chat).
//! route_access(bridge) :- has_kyc.
//! ```

use nom::{
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{char, multispace0, multispace1},
    combinator::{map, opt, recognize, value},
    multi::{many0, separated_list1},
    number::complete::recognize_float,
    sequence::{delimited, pair, preceded, terminated, tuple},
    IResult,
};

use super::{Argument, Literal, Rule, RuleSpec};
use crate::{Result, TensorCoreError};

// =============================================================================
// TOKEN PARSERS
// =============================================================================

/// Parse whitespace and comments
fn ws(input: &str) -> IResult<&str, ()> {
    value(
        (),
        many0(alt((
            value((), multispace1),
            value((), preceded(char('%'), take_while(|c| c != '\n'))),
        ))),
    )(input)
}

/// Parse optional whitespace
fn ws0(input: &str) -> IResult<&str, ()> {
    value((), multispace0)(input)
}

/// Parse an atom (lowercase identifier)
fn atom(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        take_while1(|c: char| c.is_ascii_lowercase()),
        take_while(|c: char| c.is_alphanumeric() || c == '_'),
    ))(input)
}

/// Parse a variable (uppercase identifier)
fn variable(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        take_while1(|c: char| c.is_ascii_uppercase()),
        take_while(|c: char| c.is_alphanumeric() || c == '_'),
    ))(input)
}

/// Parse a number (integer or float, possibly negative)
fn number(input: &str) -> IResult<&str, f32> {
    map(recognize_float, |s: &str| s.parse::<f32>().unwrap_or(0.0))(input)
}

/// Parse a string literal
fn string_literal(input: &str) -> IResult<&str, &str> {
    delimited(char('"'), take_while(|c| c != '"'), char('"'))(input)
}

// =============================================================================
// OPERATOR PARSERS
// =============================================================================

/// Parse implication operator: :- or <- or ←
fn impl_op(input: &str) -> IResult<&str, ()> {
    value((), alt((tag(":-"), tag("<-"), tag("←"))))(input)
}

/// Parse AND operator: , or & or ∧ or "and"
fn and_op(input: &str) -> IResult<&str, ()> {
    value((), alt((tag("∧"), tag("and"), tag("&"), tag(","))))(input)
}

/// Parse OR operator: ; or | or ∨ or "or"
fn or_op(input: &str) -> IResult<&str, ()> {
    value((), alt((tag("∨"), tag("or"), tag("|"), tag(";"))))(input)
}

/// Parse NOT operator: not or ! or ~ or ¬
fn not_op(input: &str) -> IResult<&str, ()> {
    value(
        (),
        alt((
            tag("¬"),
            terminated(tag("not"), multispace1),
            tag("!"),
            tag("~"),
        )),
    )(input)
}

// =============================================================================
// ARGUMENT PARSERS
// =============================================================================

/// Parse an argument (variable, number, string, or atom)
///
/// Arguments can be:
/// - Variables: X, Position, MyVar (uppercase start)
/// - Numbers: 0.02, -0.01, 42
/// - Strings: "hello", "chat"
/// - Atoms: chat, transfer, bridge (lowercase, treated as string constants)
fn argument(input: &str) -> IResult<&str, Argument> {
    alt((
        map(variable, |s: &str| Argument::Variable(s.to_string())),
        map(number, Argument::Constant),
        map(string_literal, |s: &str| {
            Argument::StringConstant(s.to_string())
        }),
        // Atoms are lowercase identifiers, treated as string constants
        map(atom, |s: &str| Argument::StringConstant(s.to_string())),
    ))(input)
}

/// Parse argument list: (arg1, arg2, ...)
fn argument_list(input: &str) -> IResult<&str, Vec<Argument>> {
    delimited(
        tuple((char('('), ws0)),
        separated_list1(tuple((ws0, char(','), ws0)), argument),
        tuple((ws0, char(')'))),
    )(input)
}

// =============================================================================
// PREDICATE AND LITERAL PARSERS
// =============================================================================

/// Parse a predicate: atom(args) or atom() or just atom
///
/// Supports:
/// - `predicate(X, Y)` — with arguments
/// - `predicate()` — empty argument list
/// - `predicate` — no parentheses (treated as empty args)
fn predicate(input: &str) -> IResult<&str, (&str, Vec<Argument>)> {
    pair(
        atom,
        alt((
            argument_list,
            value(Vec::new(), tuple((char('('), ws0, char(')')))),
            // No parentheses = empty argument list
            value(Vec::new(), |i| Ok((i, ()))),
        )),
    )(input)
}

/// Parse a literal (possibly negated predicate)
fn literal(input: &str) -> IResult<&str, Literal> {
    let (input, negated) = opt(not_op)(input)?;
    let (input, _) = ws0(input)?;
    let (input, (pred_name, args)) = predicate(input)?;

    Ok((
        input,
        Literal {
            predicate: pred_name.to_string(),
            args,
            negated: negated.is_some(),
        },
    ))
}

// =============================================================================
// BODY PARSERS (conjunction and disjunction)
// =============================================================================

/// Parse a conjunction (AND of literals)
fn conjunction(input: &str) -> IResult<&str, Vec<Literal>> {
    separated_list1(tuple((ws0, and_op, ws0)), literal)(input)
}

/// Parse a disjunction (OR of conjunctions)
/// Returns `Vec<Vec<Literal>>` where outer vec is OR, inner vec is AND
fn disjunction(input: &str) -> IResult<&str, Vec<Vec<Literal>>> {
    separated_list1(tuple((ws0, or_op, ws0)), conjunction)(input)
}

// =============================================================================
// RULE PARSER
// =============================================================================

/// Parse a single rule: head :- body. OR a fact: head.
///
/// Supports:
/// - Rules: `head :- body.` or `head <- body.`
/// - Facts: `head.` (unconditionally true)
fn rule(input: &str) -> IResult<&str, Vec<Rule>> {
    let (input, _) = ws(input)?;
    let (input, (head_name, head_args)) = predicate(input)?;
    let (input, _) = ws0(input)?;

    // Check if there's an implication operator
    let (input, has_body) = opt(impl_op)(input)?;

    let (input, body_disjunctions) = if has_body.is_some() {
        // Parse the body after :-
        let (input, _) = ws0(input)?;
        let (input, body) = disjunction(input)?;
        (input, body)
    } else {
        // Fact: no body, treated as unconditionally true
        // We create an empty body which will evaluate to 1.0
        (input, vec![vec![]])
    };

    let (input, _) = ws0(input)?;
    let (input, _) = opt(char('.'))(input)?; // Optional period

    // Convert disjunction to multiple rules (disjunctive normal form)
    // For facts, body_disjunctions is [[]] which creates one rule with empty body
    let rules: Vec<Rule> = if body_disjunctions.is_empty()
        || (body_disjunctions.len() == 1 && body_disjunctions[0].is_empty())
    {
        // Fact: single rule with empty body
        vec![Rule {
            head: head_name.to_string(),
            body: vec![],
            weight: None,
        }]
    } else {
        body_disjunctions
            .into_iter()
            .map(|body| Rule {
                head: head_name.to_string(),
                body,
                weight: None,
            })
            .collect()
    };

    // Store head args in the rule if needed (for parameterized facts)
    // For now, args are stored in the head string if needed
    let _ = head_args; // Suppress unused warning

    Ok((input, rules))
}

/// Parse multiple rules
fn rules(input: &str) -> IResult<&str, Vec<Rule>> {
    let (input, rule_groups) = many0(rule)(input)?;
    let (input, _) = ws(input)?;

    let all_rules: Vec<Rule> = rule_groups.into_iter().flatten().collect();
    Ok((input, all_rules))
}

// =============================================================================
// PUBLIC API
// =============================================================================

/// Parse a rule specification from a string
///
/// # Example
///
/// ```ignore
/// use an_tensor_compiler::compiler::parser::parse_rules;
///
/// let source = r#"
///     exit(X) :- profit_target(X, 0.02), momentum_shift(X).
///     exit(X) :- stop_loss(X, -0.01).
/// "#;
///
/// let spec = parse_rules("exit_rules", source)?;
/// assert_eq!(spec.rules.len(), 2);
/// ```
pub fn parse_rules(name: &str, source: &str) -> Result<RuleSpec> {
    match rules(source) {
        Ok((remaining, parsed_rules)) => {
            // Check for unparsed input
            let remaining = remaining.trim();
            if !remaining.is_empty() {
                return Err(TensorCoreError::Compiler(format!(
                    "Unexpected input after rules: '{}'",
                    &remaining[..remaining.len().min(50)]
                )));
            }

            Ok(RuleSpec {
                name: name.to_string(),
                version: "1.0.0".to_string(),
                rules: parsed_rules,
                predicates: std::collections::HashMap::new(),
            })
        }
        Err(e) => Err(TensorCoreError::Compiler(format!("Parse error: {:?}", e))),
    }
}

/// Parse a single rule from a string (convenience function)
pub fn parse_single_rule(source: &str) -> Result<Rule> {
    match rule(source.trim()) {
        Ok((_, mut rules)) => {
            if rules.is_empty() {
                Err(TensorCoreError::Compiler("No rule found".into()))
            } else if rules.len() > 1 {
                // Disjunction was expanded - return first, caller can handle
                Ok(rules.remove(0))
            } else {
                Ok(rules.remove(0))
            }
        }
        Err(e) => Err(TensorCoreError::Compiler(format!("Parse error: {:?}", e))),
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_atom() {
        let (remaining, result) = atom("profit_target").unwrap();
        assert_eq!(result, "profit_target");
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_variable() {
        let (remaining, result) = variable("X").unwrap();
        assert_eq!(result, "X");
        assert_eq!(remaining, "");

        let (remaining, result) = variable("Position").unwrap();
        assert_eq!(result, "Position");
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_number() {
        let (_, result) = number("0.02").unwrap();
        assert!((result - 0.02).abs() < 0.001);

        let (_, result) = number("-0.01").unwrap();
        assert!((result - (-0.01)).abs() < 0.001);

        let (_, result) = number("42").unwrap();
        assert!((result - 42.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_argument() {
        let (_, arg) = argument("X").unwrap();
        assert!(matches!(arg, Argument::Variable(ref s) if s == "X"));

        let (_, arg) = argument("0.02").unwrap();
        assert!(matches!(arg, Argument::Constant(v) if (v - 0.02).abs() < 0.001));

        let (_, arg) = argument("\"hello\"").unwrap();
        assert!(matches!(arg, Argument::StringConstant(ref s) if s == "hello"));
    }

    #[test]
    fn test_parse_predicate() {
        let (_, (name, args)) = predicate("profit_target(X, 0.02)").unwrap();
        assert_eq!(name, "profit_target");
        assert_eq!(args.len(), 2);
        assert!(matches!(&args[0], Argument::Variable(s) if s == "X"));
        assert!(matches!(args[1], Argument::Constant(v) if (v - 0.02).abs() < 0.001));
    }

    #[test]
    fn test_parse_literal_positive() {
        let (_, lit) = literal("profit_target(X, 0.02)").unwrap();
        assert_eq!(lit.predicate, "profit_target");
        assert!(!lit.negated);
    }

    #[test]
    fn test_parse_literal_negated() {
        // Various negation syntaxes
        let (_, lit) = literal("not bullish(X)").unwrap();
        assert_eq!(lit.predicate, "bullish");
        assert!(lit.negated);

        let (_, lit) = literal("¬bullish(X)").unwrap();
        assert!(lit.negated);

        let (_, lit) = literal("!bullish(X)").unwrap();
        assert!(lit.negated);
    }

    #[test]
    fn test_parse_conjunction() {
        let (_, lits) = conjunction("profit_target(X, 0.02), momentum_shift(X)").unwrap();
        assert_eq!(lits.len(), 2);
        assert_eq!(lits[0].predicate, "profit_target");
        assert_eq!(lits[1].predicate, "momentum_shift");
    }

    #[test]
    fn test_parse_conjunction_unicode() {
        let (_, lits) = conjunction("profit_target(X, 0.02) ∧ momentum_shift(X)").unwrap();
        assert_eq!(lits.len(), 2);
    }

    #[test]
    fn test_parse_simple_rule() {
        let source = "exit(X) :- profit_target(X, 0.02), momentum_shift(X).";
        let (_, rules) = rule(source).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].head, "exit");
        assert_eq!(rules[0].body.len(), 2);
    }

    #[test]
    fn test_parse_rule_unicode() {
        let source = "exit(X) ← profit_target(X, 0.02) ∧ momentum_shift(X)";
        let (_, rules) = rule(source).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].head, "exit");
        assert_eq!(rules[0].body.len(), 2);
    }

    #[test]
    fn test_parse_rule_with_negation() {
        let source = "hold(X) :- not exit(X), position_open(X).";
        let (_, rules) = rule(source).unwrap();
        assert_eq!(rules.len(), 1);
        assert!(rules[0].body[0].negated);
        assert!(!rules[0].body[1].negated);
    }

    #[test]
    fn test_parse_disjunction_expands_to_rules() {
        let source = "escalate(X) :- low_intent(X); high_risk(X).";
        let (_, rules) = rule(source).unwrap();
        // Disjunction expands to 2 rules
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].body[0].predicate, "low_intent");
        assert_eq!(rules[1].body[0].predicate, "high_risk");
    }

    #[test]
    fn test_parse_multiple_rules() {
        let source = r#"
            exit(X) :- profit_target(X, 0.02), momentum_shift(X).
            exit(X) :- stop_loss(X, -0.01).
            hold(X) :- not exit(X).
        "#;
        let spec = parse_rules("test", source).unwrap();
        assert_eq!(spec.rules.len(), 3);
    }

    #[test]
    fn test_parse_rules_with_comments() {
        let source = r#"
            % This is a comment
            exit(X) :- profit_target(X, 0.02).
            % Another comment
            hold(X) :- not exit(X).
        "#;
        let spec = parse_rules("test", source).unwrap();
        assert_eq!(spec.rules.len(), 2);
    }

    #[test]
    fn test_parse_complex_rule() {
        let source = "escalate(X) :- low_intent(X), high_risk(X); security_file(X).";
        let (_, rules) = rule(source).unwrap();
        // (low_intent AND high_risk) OR security_file = 2 rules
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].body.len(), 2); // low_intent, high_risk
        assert_eq!(rules[1].body.len(), 1); // security_file
    }

    #[test]
    fn test_parse_rules_full_spec() {
        let spec = parse_rules(
            "exit_rules",
            r#"
            exit(X) :- profit_target(X, 0.02), momentum_shift(X).
            exit(X) :- stop_loss(X, -0.01).
            exit(X) :- regime_change(X), not bullish(X).
        "#,
        )
        .unwrap();

        assert_eq!(spec.name, "exit_rules");
        assert_eq!(spec.rules.len(), 3);

        // Check first rule
        assert_eq!(spec.rules[0].head, "exit");
        assert_eq!(spec.rules[0].body.len(), 2);
        assert_eq!(spec.rules[0].body[0].predicate, "profit_target");

        // Check third rule has negation
        assert!(spec.rules[2].body[1].negated);
        assert_eq!(spec.rules[2].body[1].predicate, "bullish");
    }

    // =========================================================================
    // NEW: Fact support tests
    // =========================================================================

    #[test]
    fn test_parse_simple_fact() {
        let source = "panel_access(chat).";
        let spec = parse_rules("facts", source).unwrap();
        assert_eq!(spec.rules.len(), 1);
        assert_eq!(spec.rules[0].head, "panel_access");
        assert!(spec.rules[0].body.is_empty()); // Fact has no body
    }

    #[test]
    fn test_parse_fact_without_args() {
        let source = "authenticated.";
        let spec = parse_rules("facts", source).unwrap();
        assert_eq!(spec.rules.len(), 1);
        assert_eq!(spec.rules[0].head, "authenticated");
        assert!(spec.rules[0].body.is_empty());
    }

    #[test]
    fn test_parse_multiple_facts() {
        let source = r#"
            panel_access(chat).
            panel_access(settings).
            panel_access(accounts).
        "#;
        let spec = parse_rules("facts", source).unwrap();
        assert_eq!(spec.rules.len(), 3);
        assert!(spec.rules.iter().all(|r| r.head == "panel_access"));
        assert!(spec.rules.iter().all(|r| r.body.is_empty()));
    }

    #[test]
    fn test_parse_facts_and_rules_mixed() {
        let source = r#"
            panel_access(chat).
            panel_access(settings).
            panel_access(transfer) :- has_basic_kyc.
            panel_access(analytics) :- has_enhanced_kyc.
        "#;
        let spec = parse_rules("mixed", source).unwrap();
        assert_eq!(spec.rules.len(), 4);

        // First two are facts (empty body)
        assert!(spec.rules[0].body.is_empty());
        assert!(spec.rules[1].body.is_empty());

        // Last two are rules (have body)
        assert_eq!(spec.rules[2].body.len(), 1);
        assert_eq!(spec.rules[2].body[0].predicate, "has_basic_kyc");
        assert_eq!(spec.rules[3].body.len(), 1);
        assert_eq!(spec.rules[3].body[0].predicate, "has_enhanced_kyc");
    }

    // =========================================================================
    // NEW: Atom argument tests
    // =========================================================================

    #[test]
    fn test_parse_atom_argument() {
        let (_, arg) = argument("chat").unwrap();
        assert!(matches!(arg, Argument::StringConstant(ref s) if s == "chat"));
    }

    #[test]
    fn test_parse_predicate_with_atom_arg() {
        let (_, (name, args)) = predicate("panel_access(chat)").unwrap();
        assert_eq!(name, "panel_access");
        assert_eq!(args.len(), 1);
        assert!(matches!(&args[0], Argument::StringConstant(s) if s == "chat"));
    }

    #[test]
    fn test_parse_rule_with_atom_args() {
        let source = "route_access(bridge) :- has_kyc.";
        let spec = parse_rules("atoms", source).unwrap();
        assert_eq!(spec.rules.len(), 1);
        assert_eq!(spec.rules[0].head, "route_access");
        assert_eq!(spec.rules[0].body.len(), 1);
        assert_eq!(spec.rules[0].body[0].predicate, "has_kyc");
    }

    // =========================================================================
    // NEW: Predicate without parentheses
    // =========================================================================

    #[test]
    fn test_parse_predicate_no_parens() {
        let (_, (name, args)) = predicate("authenticated").unwrap();
        assert_eq!(name, "authenticated");
        assert!(args.is_empty());
    }

    #[test]
    fn test_parse_rule_predicates_no_parens() {
        let source = "panel_transfer :- has_basic_kyc.";
        let spec = parse_rules("no_parens", source).unwrap();
        assert_eq!(spec.rules.len(), 1);
        assert_eq!(spec.rules[0].head, "panel_transfer");
        assert_eq!(spec.rules[0].body[0].predicate, "has_basic_kyc");
    }

    // =========================================================================
    // NEW: Full org-monetic style rules
    // =========================================================================

    #[test]
    fn test_parse_monetic_panel_rules() {
        let source = r#"
            % Universal panels
            panel_access(chat).
            panel_access(settings).
            panel_access(accounts).
            
            % Conditional panels
            panel_access(transfer) :- has_basic_kyc.
            panel_access(transfer) :- has_payment_prestige.
            panel_access(analytics) :- has_enhanced_kyc.
            panel_access(analytics) :- high_prestige.
            
            % Admin panels
            panel_access(partner_management) :- is_admin.
            panel_access(customer_management) :- is_admin.
        "#;

        let spec = parse_rules("monetic_panels", source).unwrap();
        assert_eq!(spec.rules.len(), 9);

        // All should have head "panel_access"
        assert!(spec.rules.iter().all(|r| r.head == "panel_access"));

        // First 3 are facts
        assert!(spec.rules[0].body.is_empty());
        assert!(spec.rules[1].body.is_empty());
        assert!(spec.rules[2].body.is_empty());

        // Rest are rules
        assert!(!spec.rules[3].body.is_empty());
    }
}

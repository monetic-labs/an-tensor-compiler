//! F0 boolean-skeleton checker (Slice 1).
//!
//! [`BoolSkeleton`] erases a [`RuleSpec`] to propositional logic — exact for
//! the current compiler, whose `CompiledLiteral` carries no bindings (one
//! truth value per predicate name per forward pass). [`check_property`]
//! decides each L0 property by brute force over at most [`F0_MAX_ATOMS`]
//! atoms; beyond the cap it errors loudly instead of approximating.
//! [`emit_smtlib`] renders the same query as SMT-LIB v2 (golden-tested,
//! never executed in F-1 — the runner lands with the solver slice).
//!
//! Skeleton and spec passed to [`check_property`] must derive from the same
//! `RuleSpec`; mismatch surfaces as [`CheckerError::UnknownHead`] /
//! [`CheckerError::UnknownRule`, never silent.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::compiler::RuleSpec;

use super::property::{Property, PropertyKind};

/// Brute-force atom cap (2^24 assignments worst case). The cap documents the
/// cliff and forces the solver slice — F0 never silently approximates.
pub const F0_MAX_ATOMS: usize = 24;

/// F0 checker failure: input too large or skeleton/spec mismatch.
#[derive(Debug, thiserror::Error)]
pub enum CheckerError {
    /// Too many atoms for brute force.
    #[error("atom cap exceeded: {atoms} atoms > cap {cap}; F0 brute force refuses to approximate")]
    AtomCapExceeded {
        /// Atoms in the skeleton.
        atoms: usize,
        /// Enforced cap.
        cap: usize,
    },

    /// Head missing from the skeleton.
    #[error("unknown head '{0}' (skeleton and spec must derive from the same RuleSpec)")]
    UnknownHead(String),

    /// Rule index missing from the skeleton.
    #[error("unknown rule index {0} (skeleton and spec must derive from the same RuleSpec)")]
    UnknownRule(usize),
}

/// One conjunct of a rule body: possibly-negated atom reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lit {
    /// Atom index into the skeleton.
    pub atom: usize,
    /// Whether the literal is negated.
    pub negated: bool,
}

/// Propositional erasure of a `RuleSpec`: sorted atoms, rule bodies as
/// conjunctions, heads as rule-index groups. A head name used as a literal is
/// bound to its definition (the OR of its bodies) via an internal
/// `definitions_hold` check,
/// conjoined into every solver query and SMT emission.
#[derive(Debug, Clone)]
pub struct BoolSkeleton {
    atoms: Vec<String>,
    atom_idx: HashMap<String, usize>,
    bodies: Vec<Vec<Lit>>,
    heads: BTreeMap<String, Vec<usize>>,
}

impl BoolSkeleton {
    /// Erase `spec` to its boolean skeleton. Atom order is fully sorted
    /// (defined predicates ∪ referenced literals), so skeletons — and every
    /// downstream enumeration, emission, and address — are deterministic.
    pub fn from_spec(spec: &RuleSpec) -> Self {
        let mut names: BTreeSet<&str> = spec.predicates.keys().map(String::as_str).collect();
        names.extend(
            spec.rules
                .iter()
                .flat_map(|rule| rule.body.iter().map(|lit| lit.predicate.as_str())),
        );
        let atoms: Vec<String> = names.iter().map(ToString::to_string).collect();
        let atom_idx: HashMap<String, usize> = atoms
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.clone(), idx))
            .collect();
        let bodies: Vec<Vec<Lit>> = spec
            .rules
            .iter()
            .map(|rule| {
                rule.body
                    .iter()
                    .map(|lit| Lit {
                        atom: atom_idx[&lit.predicate],
                        negated: lit.negated,
                    })
                    .collect()
            })
            .collect();
        let mut heads: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (rule_idx, rule) in spec.rules.iter().enumerate() {
            heads.entry(rule.head.clone()).or_default().push(rule_idx);
        }
        Self {
            atoms,
            atom_idx,
            bodies,
            heads,
        }
    }

    /// Atom count (the brute-force exponent).
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    /// Atom name by index.
    pub fn atom_name(&self, idx: usize) -> Option<&str> {
        self.atoms.get(idx).map(String::as_str)
    }
}

/// A complete truth assignment, index-aligned with [`BoolSkeleton`] atoms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assignment(pub Vec<bool>);

impl Assignment {
    /// `name=true/false` pairs in atom order — for witnesses,
    /// counterexamples, and golden tests.
    pub fn fmt_named(&self, skeleton: &BoolSkeleton) -> String {
        self.0
            .iter()
            .enumerate()
            .map(|(idx, value)| format!("{}={value}", skeleton.atoms[idx]))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Decision on one property. `holds == true` is uniform-good across grades
/// (see `property.rs` polarity convention).
#[derive(Debug, Clone)]
pub struct PropertyResult {
    /// True = good/no-action-needed.
    pub holds: bool,
    /// Actionable model: satisfying assignment for SAT properties,
    /// countermodel for violated UNSAT properties, `None` for syntactic
    /// properties and unfired lints.
    pub witness: Option<Assignment>,
    /// Human/machine-readable evidence: validation errors, values, findings.
    pub notes: Vec<String>,
}

/// Decide `property` over `skeleton` (+ `spec` for syntactic properties).
pub fn check_property(
    skeleton: &BoolSkeleton,
    spec: &RuleSpec,
    property: &Property,
) -> Result<PropertyResult, CheckerError> {
    match &property.kind {
        PropertyKind::DefinedAritySafety => {
            let mut notes: Vec<String> = spec.validate().iter().map(ToString::to_string).collect();
            // PPL-native definedness half: the validator covers syntax-level
            // checks (reserved words, binding, arity) only. Every literal
            // must resolve to a predicate definition or a head.
            let head_names: BTreeSet<&str> =
                spec.rules.iter().map(|rule| rule.head.as_str()).collect();
            let mut undefined: BTreeSet<(usize, &str)> = BTreeSet::new();
            for (rule_idx, rule) in spec.rules.iter().enumerate() {
                for lit in &rule.body {
                    if !spec.predicates.contains_key(&lit.predicate)
                        && !head_names.contains(lit.predicate.as_str())
                    {
                        undefined.insert((rule_idx, lit.predicate.as_str()));
                    }
                }
            }
            for (rule_idx, name) in &undefined {
                notes.push(format!(
                    "rule {rule_idx} references undefined predicate '{name}'"
                ));
            }
            Ok(PropertyResult {
                holds: notes.is_empty(),
                witness: None,
                notes,
            })
        }
        PropertyKind::WeightFinite { rule_idx } => {
            let rule = spec
                .rules
                .get(*rule_idx)
                .ok_or(CheckerError::UnknownRule(*rule_idx))?;
            let holds = rule.weight.map(f32::is_finite).unwrap_or(true);
            Ok(PropertyResult {
                holds,
                witness: None,
                notes: if holds {
                    vec![]
                } else {
                    vec![format!(
                        "rule {rule_idx} weight is not finite: {:?}",
                        rule.weight
                    )]
                },
            })
        }
        PropertyKind::InvalidDeclaration { message } => Ok(PropertyResult {
            holds: false,
            witness: None,
            notes: vec![message.clone()],
        }),
        PropertyKind::UnusedPredicate { name } => Ok(PropertyResult {
            holds: false,
            witness: None,
            notes: vec![format!(
                "predicate '{name}' is defined but never referenced"
            )],
        }),
        PropertyKind::DuplicateRule {
            head,
            first_idx,
            second_idx,
        } => Ok(PropertyResult {
            holds: false,
            witness: None,
            notes: vec![format!(
                "rules {first_idx} and {second_idx} (head '{head}') have identical bodies"
            )],
        }),
        solver_kind => {
            if skeleton.atom_count() > F0_MAX_ATOMS {
                return Err(CheckerError::AtomCapExceeded {
                    atoms: skeleton.atom_count(),
                    cap: F0_MAX_ATOMS,
                });
            }
            check_solver_property(skeleton, solver_kind)
        }
    }
}

fn check_solver_property(
    skeleton: &BoolSkeleton,
    kind: &PropertyKind,
) -> Result<PropertyResult, CheckerError> {
    match kind {
        PropertyKind::BodySat { rule_idx } => {
            let body = skeleton
                .bodies
                .get(*rule_idx)
                .ok_or(CheckerError::UnknownRule(*rule_idx))?;
            match first_model(skeleton, |assign| eval_body(body, assign)) {
                Some(model) => Ok(PropertyResult {
                    holds: true,
                    witness: Some(Assignment(model)),
                    notes: vec![],
                }),
                None => Ok(PropertyResult {
                    holds: false,
                    witness: None,
                    notes: vec![format!(
                        "rule {rule_idx} body unsatisfiable: {}",
                        conflict_pairs(skeleton, body)
                    )],
                }),
            }
        }
        PropertyKind::HeadCoverage { head } => {
            let rules = skeleton
                .heads
                .get(head)
                .ok_or_else(|| CheckerError::UnknownHead(head.clone()))?;
            match first_model(skeleton, |assign| eval_head(skeleton, rules, assign)) {
                Some(model) => Ok(PropertyResult {
                    holds: true,
                    witness: Some(Assignment(model)),
                    notes: vec![],
                }),
                None => Ok(PropertyResult {
                    holds: false,
                    witness: None,
                    notes: vec![format!(
                        "head '{head}' is vacuously false: no rule body satisfiable"
                    )],
                }),
            }
        }
        PropertyKind::HeadNontrivial { head } => {
            let rules = skeleton
                .heads
                .get(head)
                .ok_or_else(|| CheckerError::UnknownHead(head.clone()))?;
            match first_model(skeleton, |assign| !eval_head(skeleton, rules, assign)) {
                Some(model) => Ok(PropertyResult {
                    holds: true,
                    witness: Some(Assignment(model)),
                    notes: vec![],
                }),
                None => Ok(PropertyResult {
                    holds: false,
                    witness: None,
                    notes: vec![format!(
                        "head '{head}' is a tautology: true under every assignment"
                    )],
                }),
            }
        }
        PropertyKind::Exclusion { set } => {
            // Query seeks a countermodel (all heads true); holds iff UNSAT.
            let rules_per_head: Vec<&Vec<usize>> = set
                .iter()
                .map(|head| {
                    skeleton
                        .heads
                        .get(head)
                        .ok_or_else(|| CheckerError::UnknownHead(head.clone()))
                })
                .collect::<Result<_, _>>()?;
            match first_model(skeleton, |assign| {
                rules_per_head
                    .iter()
                    .all(|rules| eval_head(skeleton, rules, assign))
            }) {
                Some(countermodel) => Ok(PropertyResult {
                    holds: false,
                    witness: Some(Assignment(countermodel)),
                    notes: vec![format!(
                        "exclusion violated: {set:?} can hold simultaneously"
                    )],
                }),
                None => Ok(PropertyResult {
                    holds: true,
                    witness: None,
                    notes: vec![],
                }),
            }
        }
        PropertyKind::TriggerPresence { head, triggers } => {
            // Query seeks a countermodel (head true, every trigger false).
            let head_rules = skeleton
                .heads
                .get(head)
                .ok_or_else(|| CheckerError::UnknownHead(head.clone()))?;
            // Pre-resolve triggers so skeleton/spec mismatch errors before
            // enumeration instead of being masked inside the closure.
            let mut resolved = Vec::with_capacity(triggers.len());
            for trigger in triggers {
                if let Some(rules) = skeleton.heads.get(trigger) {
                    resolved.push(TriggerRef::Head(rules.clone()));
                } else if let Some(&idx) = skeleton.atom_idx.get(trigger) {
                    resolved.push(TriggerRef::Atom(idx));
                } else {
                    return Err(CheckerError::UnknownHead(trigger.clone()));
                }
            }
            match first_model(skeleton, |assign| {
                // Empty trigger lists (rejected at derive; handled here for
                // direct constructors) mean the head must be unsatisfiable.
                let any_trigger = if resolved.is_empty() {
                    false
                } else {
                    resolved.iter().any(|trigger| match trigger {
                        TriggerRef::Head(rules) => eval_head(skeleton, rules, assign),
                        TriggerRef::Atom(idx) => assign[*idx],
                    })
                };
                eval_head(skeleton, head_rules, assign) && !any_trigger
            }) {
                Some(countermodel) => Ok(PropertyResult {
                    holds: false,
                    witness: Some(Assignment(countermodel)),
                    notes: vec![format!(
                        "head '{head}' can activate while every trigger in {triggers:?} is false"
                    )],
                }),
                None => Ok(PropertyResult {
                    holds: true,
                    witness: None,
                    notes: vec![],
                }),
            }
        }
        PropertyKind::Subsumption {
            subsumed_idx,
            subsuming_idx,
            ..
        } => {
            // Query seeks a distinguishing assignment; SAT means NOT subsumed
            // (clean). A fired lint carries no witness — the finding is the note.
            let subsumed = skeleton
                .bodies
                .get(*subsumed_idx)
                .ok_or(CheckerError::UnknownRule(*subsumed_idx))?;
            let subsuming = skeleton
                .bodies
                .get(*subsuming_idx)
                .ok_or(CheckerError::UnknownRule(*subsuming_idx))?;
            match first_model(skeleton, |assign| {
                eval_body(subsumed, assign) && !eval_body(subsuming, assign)
            }) {
                Some(_) => Ok(PropertyResult {
                    holds: true,
                    witness: None,
                    notes: vec![],
                }),
                None => Ok(PropertyResult {
                    holds: false,
                    witness: None,
                    notes: vec![format!(
                        "rule {subsumed_idx} is subsumed by rule {subsuming_idx} (redundant)"
                    )],
                }),
            }
        }
        _ => unreachable!("syntactic kinds are decided in check_property"),
    }
}

/// A trigger resolved against the skeleton: intermediate heads evaluate as
/// the OR of their bodies, leaf predicates read their atom.
enum TriggerRef {
    Head(Vec<usize>),
    Atom(usize),
}

fn eval_body(body: &[Lit], assignment: &[bool]) -> bool {
    // Empty conjunction is vacuously true.
    body.iter().all(|lit| assignment[lit.atom] != lit.negated)
}

fn eval_head(skeleton: &BoolSkeleton, rule_idxs: &[usize], assignment: &[bool]) -> bool {
    rule_idxs
        .iter()
        .any(|&idx| eval_body(&skeleton.bodies[idx], assignment))
}

/// Head definitions: a head name used as a literal is bound to its definition
/// (the OR of its bodies). Conjoined into every solver query so a name means
/// one thing throughout the skeleton. Heads never referenced as literals
/// impose nothing.
fn definitions_hold(skeleton: &BoolSkeleton, assignment: &[bool]) -> bool {
    skeleton
        .heads
        .iter()
        .all(|(head, rules)| match skeleton.atom_idx.get(head) {
            None => true,
            Some(&idx) => assignment[idx] == eval_head(skeleton, rules, assignment),
        })
}

/// First satisfying assignment in mask order (atom `i` = bit `i` of the
/// mask), or `None` if UNSAT. Head definitions ([`definitions_hold`]) are
/// conjoined here so every query — present and future — respects them.
/// Callers enforce [`F0_MAX_ATOMS`] first.
fn first_model(
    skeleton: &BoolSkeleton,
    mut holds_under: impl FnMut(&[bool]) -> bool,
) -> Option<Vec<bool>> {
    debug_assert!(
        skeleton.atom_count() <= F0_MAX_ATOMS,
        "brute force beyond the cap — check_property must gate first"
    );
    let total = 1usize << skeleton.atom_count();
    let mut assignment = vec![false; skeleton.atom_count()];
    for mask in 0..total {
        for (idx, slot) in assignment.iter_mut().enumerate() {
            *slot = ((mask >> idx) & 1) == 1;
        }
        if definitions_hold(skeleton, &assignment) && holds_under(&assignment) {
            return Some(assignment.clone());
        }
    }
    None
}

/// For an UNSAT conjunction: every atom appearing in both polarities. A pure
/// conjunction is UNSAT iff this set is non-empty, so the report is complete.
fn conflict_pairs(skeleton: &BoolSkeleton, body: &[Lit]) -> String {
    let mut positive = std::collections::BTreeSet::new();
    let mut negated = std::collections::BTreeSet::new();
    for lit in body {
        let name = skeleton.atoms[lit.atom].clone();
        if lit.negated {
            negated.insert(name);
        } else {
            positive.insert(name);
        }
    }
    let conflicts: Vec<String> = positive.intersection(&negated).cloned().collect();
    if conflicts.is_empty() {
        String::from("no single-atom conflict (unexpected for a pure conjunction)")
    } else {
        format!("{} appears both plain and negated", conflicts.join(", "))
    }
}

// ---------------------------------------------------------------------------
// SMT-LIB emission (golden-tested, never executed in F-1)
// ---------------------------------------------------------------------------

/// Render `property` as an SMT-LIB v2 script over `skeleton`'s atoms.
/// Solver kinds emit declarations + one assertion + `(check-sat)`; syntactic
/// kinds emit a comment only (emitting a bogus query would be dishonest
/// output). Polarities are stated in the header comment so the future runner
/// cannot misread them.
pub fn emit_smtlib(skeleton: &BoolSkeleton, property: &Property) -> Result<String, CheckerError> {
    let mut out = String::new();
    out.push_str(&format!(
        "; PPL {} — {}\n",
        kind_tag(&property.kind),
        property.description
    ));
    match &property.kind {
        PropertyKind::DefinedAritySafety
        | PropertyKind::WeightFinite { .. }
        | PropertyKind::InvalidDeclaration { .. }
        | PropertyKind::UnusedPredicate { .. }
        | PropertyKind::DuplicateRule { .. } => {
            out.push_str("; decided syntactically — no SMT query\n");
            return Ok(out);
        }
        _ => {}
    }
    out.push_str("(set-logic QF_UF)\n");
    for (idx, name) in skeleton.atoms.iter().enumerate() {
        out.push_str(&format!(
            "(declare-const {} Bool) ; {name}\n",
            smt_atom(idx, name)
        ));
    }
    let (query, polarity) = smt_query(skeleton, &property.kind)?;
    let definitions = smt_definitions(skeleton);
    let (formula, polarity) = if definitions.is_empty() {
        (query, polarity)
    } else {
        (
            format!("(and {} {query}", definitions.join(" ")),
            format!("{polarity} (head definitions conjoined)"),
        )
    };
    out.push_str(&format!("; {polarity}\n(assert {formula})\n(check-sat)\n"));
    Ok(out)
}

/// Definitional constraints for heads referenced as literals, mirroring
/// [`definitions_hold`]: each binds the head's atom to the OR of its bodies.
fn smt_definitions(skeleton: &BoolSkeleton) -> Vec<String> {
    let mut definitions = Vec::new();
    for head in skeleton.heads.keys() {
        if let Some(&idx) = skeleton.atom_idx.get(head) {
            let head_or = smt_head_or(skeleton, head).expect("head from skeleton map");
            definitions.push(format!("(= {} {head_or})", smt_atom(idx, head)));
        }
    }
    definitions
}

fn smt_query(
    skeleton: &BoolSkeleton,
    kind: &PropertyKind,
) -> Result<(String, String), CheckerError> {
    match kind {
        PropertyKind::BodySat { rule_idx } => {
            let body = skeleton
                .bodies
                .get(*rule_idx)
                .ok_or(CheckerError::UnknownRule(*rule_idx))?;
            Ok((smt_body(skeleton, body), String::from("holds iff SAT")))
        }
        PropertyKind::HeadCoverage { head } => Ok((
            smt_head_or(skeleton, head)?,
            String::from("holds iff SAT (query seeks a satisfying assignment)"),
        )),
        PropertyKind::HeadNontrivial { head } => Ok((
            format!("(not {})", smt_head_or(skeleton, head)?),
            String::from("holds iff SAT (query seeks a falsifying assignment)"),
        )),
        PropertyKind::Exclusion { set } => {
            let mut parts = Vec::with_capacity(set.len());
            for head in set {
                parts.push(smt_head_or(skeleton, head)?);
            }
            Ok((
                format!("(and {})", parts.join(" ")),
                String::from("holds iff UNSAT (query seeks a countermodel)"),
            ))
        }
        PropertyKind::TriggerPresence { head, triggers } => {
            let mut trigger_parts = Vec::with_capacity(triggers.len());
            for trigger in triggers {
                trigger_parts.push(smt_name(skeleton, trigger)?);
            }
            Ok((
                format!(
                    "(and {} (not (or {})))",
                    smt_head_or(skeleton, head)?,
                    trigger_parts.join(" ")
                ),
                String::from("holds iff UNSAT (query seeks activation avoiding all triggers)"),
            ))
        }
        PropertyKind::Subsumption {
            subsumed_idx,
            subsuming_idx,
            ..
        } => {
            let subsumed = skeleton
                .bodies
                .get(*subsumed_idx)
                .ok_or(CheckerError::UnknownRule(*subsumed_idx))?;
            let subsuming = skeleton
                .bodies
                .get(*subsuming_idx)
                .ok_or(CheckerError::UnknownRule(*subsuming_idx))?;
            Ok((
                format!(
                    "(and {} (not {}))",
                    smt_body(skeleton, subsumed),
                    smt_body(skeleton, subsuming)
                ),
                String::from("holds iff SAT (query seeks a distinguishing assignment)"),
            ))
        }
        _ => unreachable!("syntactic kinds return early in emit_smtlib"),
    }
}

fn smt_body(skeleton: &BoolSkeleton, body: &[Lit]) -> String {
    if body.is_empty() {
        return String::from("true");
    }
    let parts: Vec<String> = body
        .iter()
        .map(|lit| {
            let atom = smt_atom(lit.atom, &skeleton.atoms[lit.atom]);
            if lit.negated {
                format!("(not {atom})")
            } else {
                atom
            }
        })
        .collect();
    if parts.len() == 1 {
        parts.into_iter().next().expect("single part")
    } else {
        format!("(and {})", parts.join(" "))
    }
}

fn smt_head_or(skeleton: &BoolSkeleton, head: &str) -> Result<String, CheckerError> {
    let rules = skeleton
        .heads
        .get(head)
        .ok_or_else(|| CheckerError::UnknownHead(head.to_string()))?;
    if rules.is_empty() {
        return Ok(String::from("false"));
    }
    let parts: Vec<String> = rules
        .iter()
        .map(|&idx| smt_body(skeleton, &skeleton.bodies[idx]))
        .collect();
    if parts.len() == 1 {
        Ok(parts.into_iter().next().expect("single part"))
    } else {
        Ok(format!("(or {})", parts.join(" ")))
    }
}

fn smt_name(skeleton: &BoolSkeleton, name: &str) -> Result<String, CheckerError> {
    if skeleton.heads.contains_key(name) {
        return smt_head_or(skeleton, name);
    }
    skeleton
        .atom_idx
        .get(name)
        .map(|&idx| smt_atom(idx, &skeleton.atoms[idx]))
        .ok_or_else(|| CheckerError::UnknownHead(name.to_string()))
}

/// SMT-safe atom symbol: `p_<sanitized>_<idx>`. The `p_` prefix guarantees a
/// leading letter; the idx suffix guarantees uniqueness after sanitization.
fn smt_atom(idx: usize, name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let stem = if sanitized.is_empty() {
        "atom".to_string()
    } else {
        sanitized
    };
    format!("p_{stem}_{idx}")
}

fn kind_tag(kind: &PropertyKind) -> &'static str {
    match kind {
        PropertyKind::BodySat { .. } => "L0.1 body_sat",
        PropertyKind::HeadCoverage { .. } => "L0.2 head_coverage",
        PropertyKind::HeadNontrivial { .. } => "L0.3 head_nontrivial",
        PropertyKind::Exclusion { .. } => "L0.4 exclusion",
        PropertyKind::TriggerPresence { .. } => "L0.5 trigger_presence",
        PropertyKind::DefinedAritySafety => "L0.6 defined_arity_safety",
        PropertyKind::WeightFinite { .. } => "L0.7 weight_finite",
        PropertyKind::InvalidDeclaration { .. } => "L0.6x invalid_declaration",
        PropertyKind::Subsumption { .. } => "L0.8 subsumption",
        PropertyKind::UnusedPredicate { .. } => "L0.9 unused_predicate",
        PropertyKind::DuplicateRule { .. } => "L0.9 duplicate_rule",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{Argument, Literal, Rule};
    use crate::governance::property::{
        fixtures::{literal, negated_literal, rule, threshold_predicate},
        L0Declarations, Property, PropertyKind,
    };

    fn check_all(spec: &RuleSpec, decls: &L0Declarations) -> Vec<(Property, PropertyResult)> {
        let skeleton = BoolSkeleton::from_spec(spec);
        crate::governance::property::derive_l0(spec, decls)
            .iter()
            .map(|property| {
                let result = check_property(&skeleton, spec, property).expect("within atom cap");
                (property.clone(), result)
            })
            .collect()
    }

    fn find(
        results: &[(Property, PropertyResult)],
        mut matches: impl FnMut(&PropertyKind) -> bool,
    ) -> &PropertyResult {
        results
            .iter()
            .find(|(property, _)| matches(&property.kind))
            .map(|(_, result)| result)
            .expect("expected property present")
    }

    fn spec_with_predicates(names: &[&str]) -> RuleSpec {
        let mut spec = RuleSpec::new("fixture");
        for name in names {
            spec.add_predicate(name.to_string(), threshold_predicate());
        }
        spec
    }

    #[test]
    fn body_sat_detects_contradiction_and_names_it() {
        let mut spec = spec_with_predicates(&["p"]);
        spec.add_rule(rule("h", vec![literal("p"), negated_literal("p")]));
        let results = check_all(&spec, &L0Declarations::default());

        let body = find(&results, |kind| {
            matches!(kind, PropertyKind::BodySat { .. })
        });
        assert!(!body.holds);
        assert!(body.notes.iter().any(|note| note.contains('p')));
        let coverage = find(&results, |kind| {
            matches!(kind, PropertyKind::HeadCoverage { .. })
        });
        assert!(!coverage.holds, "vacuously false head must fail coverage");
    }

    #[test]
    fn head_nontrivial_fails_on_tautology() {
        let mut spec = spec_with_predicates(&["p"]);
        spec.add_rule(rule("h", vec![literal("p")]));
        spec.add_rule(rule("h", vec![negated_literal("p")]));
        let results = check_all(&spec, &L0Declarations::default());

        let nontrivial = find(&results, |kind| {
            matches!(kind, PropertyKind::HeadNontrivial { .. })
        });
        assert!(!nontrivial.holds);
        assert!(nontrivial
            .notes
            .iter()
            .any(|note| note.contains("tautology")));
        let coverage = find(&results, |kind| {
            matches!(kind, PropertyKind::HeadCoverage { .. })
        });
        assert!(coverage.holds);
    }

    #[test]
    fn exclusion_holds_when_disjoint_fails_when_overlapping() {
        // Disjoint: a :- p. b :- ¬p. → holds.
        let mut disjoint = spec_with_predicates(&["p"]);
        disjoint.add_rule(rule("a", vec![literal("p")]));
        disjoint.add_rule(rule("b", vec![negated_literal("p")]));
        let decls = L0Declarations {
            exclusions: vec![vec!["a".to_string(), "b".to_string()]],
            triggers: vec![],
        };
        let results = check_all(&disjoint, &decls);
        let holds = find(&results, |kind| {
            matches!(kind, PropertyKind::Exclusion { .. })
        });
        assert!(holds.holds);
        assert!(holds.witness.is_none());

        // Overlapping: a :- p. b :- p. → fails with countermodel p=true.
        let mut overlapping = spec_with_predicates(&["p"]);
        overlapping.add_rule(rule("a", vec![literal("p")]));
        overlapping.add_rule(rule("b", vec![literal("p")]));
        let results = check_all(&overlapping, &decls);
        let violated = find(&results, |kind| {
            matches!(kind, PropertyKind::Exclusion { .. })
        });
        assert!(!violated.holds);
        let witness = violated.witness.as_ref().expect("countermodel");
        let skeleton = BoolSkeleton::from_spec(&overlapping);
        assert!(witness.fmt_named(&skeleton).contains("p=true"));
    }

    #[test]
    fn trigger_presence_catches_unguarded_path() {
        // esc :- low. esc :- other. triggers(esc, [low, high]) → fails: the
        // `other` path activates esc with every trigger false.
        let mut spec = spec_with_predicates(&["low", "high", "other"]);
        spec.add_rule(rule("esc", vec![literal("low")]));
        spec.add_rule(rule("esc", vec![literal("other")]));
        let decls = L0Declarations {
            exclusions: vec![],
            triggers: vec![crate::governance::property::TriggerDecl {
                head: "esc".to_string(),
                triggers: vec!["low".to_string(), "high".to_string()],
            }],
        };
        let results = check_all(&spec, &decls);
        let violated = find(&results, |kind| {
            matches!(kind, PropertyKind::TriggerPresence { .. })
        });
        assert!(!violated.holds);
        let witness = violated.witness.as_ref().expect("countermodel");
        let skeleton = BoolSkeleton::from_spec(&spec);
        let named = witness.fmt_named(&skeleton);
        assert!(named.contains("low=false"), "{named}");
        assert!(named.contains("high=false"), "{named}");
        assert!(named.contains("other=true"), "{named}");

        // Guarded: esc :- low. esc :- high. → holds.
        let mut guarded = spec_with_predicates(&["low", "high"]);
        guarded.add_rule(rule("esc", vec![literal("low")]));
        guarded.add_rule(rule("esc", vec![literal("high")]));
        let results = check_all(&guarded, &decls);
        let holds = find(&results, |kind| {
            matches!(kind, PropertyKind::TriggerPresence { .. })
        });
        assert!(holds.holds);
    }

    #[test]
    fn trigger_may_name_an_intermediate_head() {
        // esc :- stage. stage :- low. triggers(esc, [stage]) → holds: the
        // trigger evaluates as the OR of stage's bodies.
        let mut spec = spec_with_predicates(&["low"]);
        spec.add_rule(rule("stage", vec![literal("low")]));
        spec.add_rule(rule("esc", vec![literal("stage")]));
        let decls = L0Declarations {
            exclusions: vec![],
            triggers: vec![crate::governance::property::TriggerDecl {
                head: "esc".to_string(),
                triggers: vec!["stage".to_string()],
            }],
        };
        let results = check_all(&spec, &decls);
        let holds = find(&results, |kind| {
            matches!(kind, PropertyKind::TriggerPresence { .. })
        });
        assert!(holds.holds);
    }

    #[test]
    fn defined_arity_safety_imports_arity_errors() {
        let mut spec = RuleSpec::new("arity");
        spec.add_rule(Rule::new(
            "h",
            vec![Literal::positive(
                "p",
                vec![Argument::Variable("X".to_string())],
            )],
        ));
        spec.add_rule(Rule::new(
            "h",
            vec![Literal::positive(
                "p",
                vec![
                    Argument::Variable("X".to_string()),
                    Argument::Variable("Y".to_string()),
                ],
            )],
        ));
        spec.add_predicate("p", threshold_predicate());
        let results = check_all(&spec, &L0Declarations::default());
        let imported = find(&results, |kind| {
            matches!(kind, PropertyKind::DefinedAritySafety)
        });
        assert!(!imported.holds);
        assert!(imported.notes.iter().any(|note| note.contains("arity")));
    }

    #[test]
    fn weight_finite_flags_nan_only() {
        let mut spec = spec_with_predicates(&["p", "q"]);
        spec.add_rule(rule("h", vec![literal("p")]).with_weight(f32::NAN));
        spec.add_rule(rule("h", vec![literal("q")]).with_weight(0.5));
        let results = check_all(&spec, &L0Declarations::default());

        let nan = find(&results, |kind| {
            matches!(kind, PropertyKind::WeightFinite { rule_idx: 0 })
        });
        assert!(!nan.holds);
        let finite = find(&results, |kind| {
            matches!(kind, PropertyKind::WeightFinite { rule_idx: 1 })
        });
        assert!(finite.holds);
    }

    #[test]
    fn subsumption_lint_fires_on_redundancy_clears_otherwise() {
        // rule 0 (p ∧ q) implies rule 1 (p): 0 subsumed by 1 fires, 1 by 0 clears.
        let mut spec = spec_with_predicates(&["p", "q"]);
        spec.add_rule(rule("h", vec![literal("p"), literal("q")]));
        spec.add_rule(rule("h", vec![literal("p")]));
        let results = check_all(&spec, &L0Declarations::default());

        let fired = find(&results, |kind| {
            matches!(
                kind,
                PropertyKind::Subsumption {
                    subsumed_idx: 0,
                    subsuming_idx: 1,
                    ..
                }
            )
        });
        assert!(!fired.holds);
        assert!(fired.witness.is_none());
        let clear = find(&results, |kind| {
            matches!(
                kind,
                PropertyKind::Subsumption {
                    subsumed_idx: 1,
                    subsuming_idx: 0,
                    ..
                }
            )
        });
        assert!(clear.holds);
    }

    #[test]
    fn atom_cap_errors_loudly_while_syntactic_properties_skip_it() {
        let names: Vec<String> = (0..25).map(|i| format!("p{i:02}")).collect();
        let mut spec = RuleSpec::new("wide");
        for name in &names {
            spec.add_predicate(name.clone(), threshold_predicate());
        }
        spec.add_rule(Rule::new("h", vec![literal(&names[0])]).with_weight(0.5));
        let skeleton = BoolSkeleton::from_spec(&spec);
        assert_eq!(skeleton.atom_count(), 25);

        let properties = crate::governance::property::derive_l0(&spec, &L0Declarations::default());
        let body_sat = properties
            .iter()
            .find(|p| matches!(p.kind, PropertyKind::BodySat { .. }))
            .expect("BodySat derived");
        assert!(matches!(
            check_property(&skeleton, &spec, body_sat),
            Err(CheckerError::AtomCapExceeded {
                atoms: 25,
                cap: F0_MAX_ATOMS
            })
        ));
        // Syntactic properties never enumerate: no cap applies.
        let weight = properties
            .iter()
            .find(|p| matches!(p.kind, PropertyKind::WeightFinite { .. }))
            .expect("WeightFinite derived");
        assert!(check_property(&skeleton, &spec, weight).unwrap().holds);
    }

    #[test]
    fn smtlib_golden_body_sat() {
        let mut spec = spec_with_predicates(&["p", "q"]);
        spec.add_rule(rule("h", vec![literal("p"), negated_literal("q")]));
        let skeleton = BoolSkeleton::from_spec(&spec);
        let properties = crate::governance::property::derive_l0(&spec, &L0Declarations::default());
        let body_sat = properties
            .iter()
            .find(|p| matches!(p.kind, PropertyKind::BodySat { .. }))
            .expect("BodySat derived");
        assert_eq!(
            emit_smtlib(&skeleton, body_sat).unwrap(),
            "; PPL L0.1 body_sat — rule 0 (h): body is satisfiable\n\
             (set-logic QF_UF)\n\
             (declare-const p_p_0 Bool) ; p\n\
             (declare-const p_q_1 Bool) ; q\n\
             ; holds iff SAT\n\
             (assert (and p_p_0 (not p_q_1)))\n\
             (check-sat)\n"
        );
    }

    #[test]
    fn smtlib_states_polarity_and_skips_syntactic_kinds() {
        let mut spec = spec_with_predicates(&["p"]);
        spec.add_rule(rule("a", vec![literal("p")]));
        spec.add_rule(rule("b", vec![literal("p")]).with_weight(0.5));
        let decls = L0Declarations {
            exclusions: vec![vec!["a".to_string(), "b".to_string()]],
            triggers: vec![],
        };
        let skeleton = BoolSkeleton::from_spec(&spec);
        let properties = crate::governance::property::derive_l0(&spec, &decls);
        let exclusion = properties
            .iter()
            .find(|p| matches!(p.kind, PropertyKind::Exclusion { .. }))
            .expect("Exclusion derived");
        let emitted = emit_smtlib(&skeleton, exclusion).unwrap();
        assert!(emitted.contains("holds iff UNSAT (query seeks a countermodel)"));
        assert!(emitted.contains("(check-sat)"));

        let weight = properties
            .iter()
            .find(|p| matches!(p.kind, PropertyKind::WeightFinite { .. }))
            .expect("WeightFinite derived");
        let emitted = emit_smtlib(&skeleton, weight).unwrap();
        assert!(emitted.contains("no SMT query"));
        assert!(!emitted.contains("(check-sat)"));
    }

    #[test]
    fn skeleton_erases_args_to_one_atom_per_name() {
        let mut spec = RuleSpec::new("erasure");
        spec.add_rule(Rule::new(
            "h",
            vec![Literal::positive(
                "p",
                vec![Argument::Variable("X".to_string())],
            )],
        ));
        spec.add_rule(Rule::new(
            "h",
            vec![Literal::positive(
                "p",
                vec![Argument::Variable("Y".to_string()), Argument::Constant(1.0)],
            )],
        ));
        spec.add_predicate("p", threshold_predicate());
        let skeleton = BoolSkeleton::from_spec(&spec);
        assert_eq!(skeleton.atom_count(), 1);
        assert_eq!(skeleton.atom_name(0), Some("p"));
    }

    #[test]
    fn smtlib_conjoins_head_definitions() {
        // esc :- stage (head used as a literal): the emission must bind the
        // head's atom to its definition, mirroring definitions_hold.
        let mut spec = spec_with_predicates(&["low"]);
        spec.add_rule(rule("stage", vec![literal("low")]));
        spec.add_rule(rule("esc", vec![literal("stage")]));
        let skeleton = BoolSkeleton::from_spec(&spec);
        let properties = crate::governance::property::derive_l0(&spec, &L0Declarations::default());
        let coverage = properties
            .iter()
            .find(|p| matches!(p.kind, PropertyKind::HeadCoverage { .. }))
            .expect("HeadCoverage derived");
        let emitted = emit_smtlib(&skeleton, coverage).unwrap();
        // Atoms sorted: low(0), stage(1); stage's definition is low.
        assert!(
            emitted.contains("(= p_stage_1 p_low_0)"),
            "missing head definition:\n{emitted}"
        );
        assert!(emitted.contains("(head definitions conjoined)"));
    }
}

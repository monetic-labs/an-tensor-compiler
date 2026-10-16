//! # Predicate Governance Core
//!
//! Static verification for tensor-logic predicates: derive properties from a
//! [`crate::compiler::RuleSpec`], check them over the boolean skeleton, propagate
//! margins through the rule DAG, and sort-check predicate outputs.
//!
//! This is the solver-free core of the Predicate Property Language (PPL):
//!
//! - [`property`]: L0 structural properties auto-derived from `RuleSpec`
//!   (`body_sat`, `head_coverage`, exclusion sets, trigger presence, …).
//! - [`checker`]: in-house boolean-skeleton (F0) checker plus SMT-LIB emission
//!   for external solvers.
//! - [`margin`]: closed-form margin propagation calculus. Where the output
//!   margin goes non-positive, the head is marked unresolvable-at-margin and
//!   joins the abstention region.
//! - [`sort`]: output sort system (`Truth` vs `Score` vs `Index` vs `Count`)
//!   with a sort-check over literal positions.
//! - [`canonical`]: stable canonical bytes for specs and properties.
//! - [`calibrate`]: expected-calibration-error (ECE) calibrator feeding leaf margins.
//! - [`demo`]: runnable end-to-end demo wiring the above together.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use an_tensor_compiler::governance::{derive_l0, check_property, propagate_margins};
//! use an_tensor_compiler::compiler::RuleSpec;
//!
//! let spec = RuleSpec::parse("policy", "escalate(X) :- high_risk(X), not approved(X).")?;
//! let properties = derive_l0(&spec, &Default::default());
//! for property in &properties {
//!     let result = check_property(property)?;
//!     println!("{:?}: {:?}", property.kind, result);
//! }
//! ```
//!
//! Ported from `an-tensor-compiler` v0.5.0 (`governance` module, solver-free subset).
//! Excluded from this port (they require the private content-addressed artifact
//! backend): `artifact`, `ledger`, `stamp`, `promotion`, `revoke`, `eval`, `queue`,
//! `sample`, `shadow`, `pipeline`, `triage`, `triage_queue`.

pub mod calibrate;
pub mod canonical;
pub mod checker;
pub mod demo;
pub mod margin;
pub mod property;
pub mod sort;

pub use calibrate::{Calibrator, EceHistory};
pub use canonical::{
    canonical_json, canonical_spec_bytes, sorted_predicates, to_canonical_bytes, CanonicalError,
    FiniteF32, FiniteF64,
};
pub use checker::{
    check_property, emit_smtlib, Assignment, BoolSkeleton, CheckerError, Lit, PropertyResult,
};
pub use demo::{safety_spec, DemoInputs};
pub use margin::{propagate_margins, MarginConfig, MarginReport, NodeMargin};
pub use property::{derive_l0, Grade, L0Declarations, Property, PropertyKind, TriggerDecl};
pub use sort::{
    sort_check, sort_check_with_policy, sort_of_name, OutputSort, SortFinding, SortLocation,
    SortPolicy, SortSeverity,
};

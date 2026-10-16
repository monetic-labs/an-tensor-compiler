//! Builders for common declarative invariants → [`PredicateSpec`].
//!
//! Most policy-shaped HetsEntity consumers (Charters, route playbooks,
//! provider-policy entities, governance documents) need the same handful
//! of invariant patterns:
//!
//! - a value (or a sum of values) equals a target within tolerance
//! - a value is non-negative
//! - a value lies in `[min, max]`
//! - a value is greater-than-or-equal-to a predecessor (monotone history)
//! - a categorical "fingerprint" lies within an allowed-set distance
//!
//! Each of these can be expressed as a single
//! [`PredicateSpec::Threshold`] over a *consumer-pre-computed* scalar
//! input. That is the discipline this module enforces: the substrate
//! does not own how to *compute* "the sum of stakeholder claims" or
//! "the distance from the allowed-set fingerprint" — that is consumer
//! domain logic. The substrate provides canonical thresholding so that
//! once the consumer pre-computes its scalar feature, the predicate's
//! output is a smooth value in `[0, 1]` that interacts correctly with
//! [`crate::primitives::fuzzy_and`] and friends in conjunctions.
//!
//! # Why not Composite predicates?
//!
//! [`PredicateSpec::Composite`] declares an AND/OR/NOT over *named*
//! children. In-process evaluation does not yet resolve those names —
//! Composite evaluations fall through to a neutral `0.5`. Until that
//! gap closes, the substrate-honest path
//! for "A AND B" is to build `A` and `B` as separate `PredicateSpec`s,
//! evaluate them independently against the same `inputs` map, and
//! combine the resulting scalars with [`crate::primitives::fuzzy_and`]
//! in consumer code. These builders return single
//! `PredicateSpec::Threshold`s precisely so that pattern is the
//! straightforward one.
//!
//! # Sharpness
//!
//! Every builder accepts a `sharpness` argument that controls how
//! quickly the soft threshold transitions from `0` to `1` near the
//! boundary. The default
//! ([`DEFAULT_PREDICATE_SHARPNESS`]) is calibrated so that
//! "near-misses" of a tolerance bound get a graded score (which is
//! desirable for `fuzzy_and` over many invariants — a single
//! near-miss does not collapse the whole conjunction to zero) while
//! clear violations approach zero quickly. Override only if you have
//! a profile-specific reason.

use crate::compiler::PredicateSpec;

/// Default soft-threshold sharpness for invariant builders.
///
/// At `sharpness = 10.0`, a value `0.1` away from the threshold scores
/// roughly `0.27` (clearly violating); a value `0.01` away scores
/// roughly `0.48` (graded near-miss); at the threshold scores `0.5`.
/// This is the regime that interacts cleanly with `fuzzy_and` over
/// flat conjunctions of invariants.
pub const DEFAULT_PREDICATE_SHARPNESS: f32 = 10.0;

/// Predicate that evaluates near `1.0` when `inputs[input] >= 0.0` and
/// degrades smoothly toward `0.0` as the value goes negative.
///
/// Canonical usage: a non-negativity invariant over a pre-computed
/// scalar. Example: `non_negative("reserve_floor")` for the
/// "reserves must be non-negative" rule.
pub fn non_negative(input: impl Into<String>) -> PredicateSpec {
    PredicateSpec::Threshold {
        input: input.into(),
        threshold: 0.0,
        greater_than: true,
        sharpness: DEFAULT_PREDICATE_SHARPNESS,
    }
}

/// Predicate that evaluates near `1.0` when `inputs[input] >= threshold`
/// and degrades smoothly below it.
pub fn at_least(input: impl Into<String>, threshold: f32) -> PredicateSpec {
    PredicateSpec::Threshold {
        input: input.into(),
        threshold,
        greater_than: true,
        sharpness: DEFAULT_PREDICATE_SHARPNESS,
    }
}

/// Predicate that evaluates near `1.0` when `inputs[input] <= threshold`
/// and degrades smoothly above it.
pub fn at_most(input: impl Into<String>, threshold: f32) -> PredicateSpec {
    PredicateSpec::Threshold {
        input: input.into(),
        threshold,
        greater_than: false,
        sharpness: DEFAULT_PREDICATE_SHARPNESS,
    }
}

/// Predicate that evaluates near `1.0` when `inputs[input]` is within
/// `tolerance` of `target`, falling off as the absolute difference grows.
///
/// The consumer is expected to pre-compute the scalar
/// `signed_excess = value - target` and put it under `inputs[input]`.
/// The predicate fires when `|signed_excess| <= tolerance`.
///
/// Canonical usage: "stakeholder claims for pool P sum to 100%". The
/// consumer computes `claim_sum_excess = stakeholder_claims_sum - 1.0`
/// and the predicate fires when that is near zero.
///
/// Returns a `PredicateSpec::Threshold` with `greater_than = false` and
/// the threshold set to `tolerance`, evaluating the absolute-distance
/// invariant via `|excess| <= tolerance`. The consumer must put the
/// *absolute* value under `inputs[input]`; this builder does not
/// pre-compute `abs()` because the substrate does not own how the
/// scalar is derived. (Most callers will pre-compute `abs(value -
/// target)` once and use that as the input feature.)
pub fn approximately(input: impl Into<String>, tolerance: f32) -> PredicateSpec {
    PredicateSpec::Threshold {
        input: input.into(),
        threshold: tolerance,
        greater_than: false,
        sharpness: DEFAULT_PREDICATE_SHARPNESS,
    }
}

/// Predicate that evaluates near `1.0` when
/// `inputs[input] >= predecessor_value` and degrades smoothly below.
///
/// The consumer is expected to pre-compute the scalar
/// `delta = current_value - predecessor_value` and put it under
/// `inputs[input]`. The predicate fires when `delta >= 0`.
///
/// Canonical usage: "amendment effective_date must be ≥ predecessor's
/// effective_date" — the consumer subtracts predecessor from current
/// (in any time unit, including i64 unix seconds normalized to f32
/// days) and the predicate fires non-negative.
pub fn monotone_ge_predecessor(input: impl Into<String>) -> PredicateSpec {
    non_negative(input)
}

/// Predicate that evaluates near `1.0` when
/// `inputs[input] <= max_distance` and degrades smoothly above it.
///
/// Intended for "categorical membership" invariants encoded as
/// distance-from-allowed-set. The consumer pre-computes a scalar
/// distance (e.g., minimum cosine distance from any allowed-set
/// member's HRR) and the predicate fires when that distance is below
/// `max_distance`.
///
/// Canonical usage: "tax classification must be in allowed set for
/// entity kind + jurisdiction." The consumer holds an allowed-set
/// fingerprint per `(kind, jurisdiction)` pair and pre-computes the
/// distance from the candidate's tax-aspect vector; the predicate
/// gates on that.
pub fn distance_within(input: impl Into<String>, max_distance: f32) -> PredicateSpec {
    PredicateSpec::Threshold {
        input: input.into(),
        threshold: max_distance,
        greater_than: false,
        sharpness: DEFAULT_PREDICATE_SHARPNESS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::PredicateSpec;
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;

    fn scalar(value: f32) -> Tensor {
        Tensor::from_vec(vec![value], 1, &Device::Cpu).unwrap()
    }

    // NOTE (public port): private tests evaluate via `hets::PredicateEvaluator`.
    // HETS is not part of the public crate, so tests evaluate Threshold specs
    // directly with the same semantics as `CompiledPredicate::Threshold`
    // (soft threshold with sign-flipped sharpness when `greater_than` is false).
    fn eval(pred: &PredicateSpec, inputs: HashMap<String, Tensor>) -> f32 {
        match pred {
            PredicateSpec::Threshold {
                input,
                threshold,
                greater_than,
                sharpness,
            } => {
                let x = inputs.get(input).unwrap();
                let sharp = if *greater_than {
                    *sharpness
                } else {
                    -*sharpness
                };
                crate::primitives::soft_threshold_scalar(x, *threshold, sharp)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()[0]
            }
            other => panic!("builders only produce Threshold specs, got {other:?}"),
        }
    }

    #[test]
    fn non_negative_fires_for_zero_and_positive() {
        let pred = non_negative("reserve");
        assert!(eval(&pred, HashMap::from([("reserve".into(), scalar(0.0))])) >= 0.5);
        assert!(eval(&pred, HashMap::from([("reserve".into(), scalar(1.0))])) > 0.95);
        assert!(eval(&pred, HashMap::from([("reserve".into(), scalar(-1.0))])) < 0.05);
    }

    #[test]
    fn at_least_at_most_complement_each_other() {
        let lower = at_least("v", 5.0);
        let upper = at_most("v", 10.0);

        let mid = HashMap::from([("v".into(), scalar(7.5))]);
        assert!(eval(&lower, mid.clone()) > 0.95);
        assert!(eval(&upper, mid) > 0.95);

        let too_low = HashMap::from([("v".into(), scalar(2.0))]);
        assert!(eval(&lower, too_low.clone()) < 0.05);
        assert!(eval(&upper, too_low) > 0.95);

        let too_high = HashMap::from([("v".into(), scalar(15.0))]);
        assert!(eval(&lower, too_high.clone()) > 0.95);
        assert!(eval(&upper, too_high) < 0.05);
    }

    #[test]
    fn approximately_fires_within_tolerance() {
        // At default unit-scale sharpness=10, tight tolerances produce a
        // graded soft-threshold near boundary rather than a crisp 0/1.
        // Consumers that need crisper transitions for tight tolerances
        // should override sharpness; see module docs.
        let pred = approximately("abs_excess", 0.01);
        let well_inside = eval(&pred, HashMap::from([("abs_excess".into(), scalar(0.0))]));
        let well_outside = eval(&pred, HashMap::from([("abs_excess".into(), scalar(0.5))]));

        assert!(
            well_inside > well_outside + 0.4,
            "perfect value should score notably higher than gross violation \
             (in={well_inside}, out={well_outside})"
        );
        assert!(well_outside < 0.1, "gross violation should approach 0");
    }

    #[test]
    fn distance_within_fires_below_threshold() {
        let pred = distance_within("dist", 0.3);
        // For unit-scale max_distance, default sharpness gives crisp fire / no-fire.
        assert!(eval(&pred, HashMap::from([("dist".into(), scalar(0.05))])) > 0.9);
        assert!(eval(&pred, HashMap::from([("dist".into(), scalar(0.6))])) < 0.1);
    }

    #[test]
    fn monotone_ge_predecessor_fires_on_non_decreasing_delta() {
        let pred = monotone_ge_predecessor("delta");
        assert!(eval(&pred, HashMap::from([("delta".into(), scalar(1.0))])) > 0.95);
        assert!(eval(&pred, HashMap::from([("delta".into(), scalar(0.0))])) >= 0.5);
        assert!(eval(&pred, HashMap::from([("delta".into(), scalar(-0.5))])) < 0.1);
    }

    #[test]
    fn fuzzy_and_pattern_for_flat_conjunction() {
        // Demonstrates the substrate-honest pattern for "A AND B AND C":
        // evaluate each Threshold independently, then combine with
        // fuzzy_and externally. This keeps the conjunction
        // mathematically correct without depending on the
        // PredicateSpec::Composite evaluator gap.
        //
        // Inputs are scaled to be unit-scale so default sharpness is
        // appropriate. Tight-tolerance invariants would normalize their
        // pre-computed scalar (e.g. claim_excess in basis points, not
        // raw fraction) to land in the same regime.
        use crate::primitives::fuzzy_and;

        let claim_excess = at_most("claim_excess_units", 1.0); // <= 1 unit of tolerance
        let reserve = non_negative("reserve_floor_units");
        let amendment_monotone = monotone_ge_predecessor("amendment_delta_units");

        let inputs_clean = HashMap::from([
            ("claim_excess_units".into(), scalar(0.0)),
            ("reserve_floor_units".into(), scalar(50.0)),
            ("amendment_delta_units".into(), scalar(2.0)),
        ]);

        let s1 = eval(&claim_excess, inputs_clean.clone());
        let s2 = eval(&reserve, inputs_clean.clone());
        let s3 = eval(&amendment_monotone, inputs_clean);

        let conj = fuzzy_and(&scalar(s1), &scalar(s2)).unwrap();
        let conj = fuzzy_and(&conj, &scalar(s3)).unwrap();
        let combined: f32 = conj.to_vec1::<f32>().unwrap()[0];

        assert!(
            combined > 0.85,
            "all-clean unit-scale conjunction should be ~1, got {combined}"
        );

        let inputs_violated = HashMap::from([
            ("claim_excess_units".into(), scalar(5.0)), // way over tolerance
            ("reserve_floor_units".into(), scalar(50.0)),
            ("amendment_delta_units".into(), scalar(2.0)),
        ]);
        let s1v = eval(&claim_excess, inputs_violated.clone());
        let s2v = eval(&reserve, inputs_violated.clone());
        let s3v = eval(&amendment_monotone, inputs_violated);
        let conj_v = fuzzy_and(&scalar(s1v), &scalar(s2v)).unwrap();
        let conj_v = fuzzy_and(&conj_v, &scalar(s3v)).unwrap();
        let combined_v: f32 = conj_v.to_vec1::<f32>().unwrap()[0];

        assert!(
            combined_v < 0.1,
            "violated invariant must collapse the conjunction, got {combined_v}"
        );
    }
}

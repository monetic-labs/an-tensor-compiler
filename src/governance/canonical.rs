//! Canonical bytes for governance artifacts (Slice 0).
//!
//! Every governed artifact canonicalizes before addressing: same meaning,
//! same bytes, same `an-addr` — on every machine, on every run. The rules:
//!
//! 1. Object keys sorted recursively (insertion order is meaningless).
//! 2. Array order preserved (order is meaning).
//! 3. Integers preserved exactly (`1u64` stays `1`, never `1.0`).
//! 4. Floats via f64-widened shortest roundtrip (deterministic; exact-bit
//!    sensitive — `0.1f32` and `0.1f64` canonicalize differently, by design).
//! 5. Non-finite floats fail closed. `serde_json` maps NaN/±Inf to `null`
//!    silently at serialization time, which would alias distinct specs — so
//!    governance schemas hold floats behind [`FiniteF32`]/[`FiniteF64`],
//!    whose `Serialize` impls reject non-finite values before that mapping.
//!
//! [`sorted_predicates`] enforces the scope-doc determinism rule
//! (`SCOPE_F1 §4`): `RuleSpec.predicates` is a `HashMap` with randomized
//! iteration, so every derivation/emission path sorts predicate names first.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::compiler::{PredicateSpec, RuleSpec};
use crate::TensorCoreError;

/// Canonicalization failure.
#[derive(Debug, thiserror::Error)]
pub enum CanonicalError {
    /// A non-finite float reached canonical bytes. Unreachable for parsed
    /// `Value`s (`serde_json` cannot represent non-finite) — kept as
    /// defense-in-depth. The real fail-closed point is [`FiniteF32`] /
    /// [`FiniteF64`] at serialization time.
    #[error("non-finite float at {path}: {value}")]
    NonFiniteFloat { path: String, value: String },

    /// `Serialize` → `Value` conversion failed (includes `Finite*`
    /// rejections, which surface here with their message).
    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl From<CanonicalError> for TensorCoreError {
    fn from(e: CanonicalError) -> Self {
        TensorCoreError::Governance(e.to_string())
    }
}

/// An f32 guaranteed finite. Governance schemas use this (and [`FiniteF64`])
/// for all floats so non-finite values error at serialization instead of
/// silently becoming JSON `null`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteF32(f32);

/// An f64 guaranteed finite. See [`FiniteF32`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteF64(f64);

impl FiniteF32 {
    /// Checked constructor: `Err` on NaN/±Inf.
    pub fn new(v: f32) -> Result<Self, CanonicalError> {
        if v.is_finite() {
            Ok(Self(v))
        } else {
            Err(CanonicalError::NonFiniteFloat {
                path: String::from("$"),
                value: v.to_string(),
            })
        }
    }

    /// The wrapped value (always finite).
    pub fn get(self) -> f32 {
        self.0
    }
}

impl FiniteF64 {
    /// Checked constructor: `Err` on NaN/±Inf.
    pub fn new(v: f64) -> Result<Self, CanonicalError> {
        if v.is_finite() {
            Ok(Self(v))
        } else {
            Err(CanonicalError::NonFiniteFloat {
                path: String::from("$"),
                value: v.to_string(),
            })
        }
    }

    /// The wrapped value (always finite).
    pub fn get(self) -> f64 {
        self.0
    }
}

impl Serialize for FiniteF32 {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if !self.0.is_finite() {
            return Err(serde::ser::Error::custom(
                "non-finite f32 in canonical artifact",
            ));
        }
        s.serialize_f32(self.0)
    }
}

impl Serialize for FiniteF64 {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if !self.0.is_finite() {
            return Err(serde::ser::Error::custom(
                "non-finite f64 in canonical artifact",
            ));
        }
        s.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for FiniteF32 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = f32::deserialize(d)?;
        FiniteF32::new(v).map_err(serde::de::Error::custom)
    }
}

impl<'de> Deserialize<'de> for FiniteF64 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = f64::deserialize(d)?;
        FiniteF64::new(v).map_err(serde::de::Error::custom)
    }
}

/// Canonical bytes for a JSON value: sorted keys, preserved ints, f64-widened
/// shortest-roundtrip floats. Deterministic across processes and machines.
pub fn canonical_json(value: &Value) -> Result<Vec<u8>, CanonicalError> {
    let sorted = sort_value(value, "$")?;
    // `serde_json::to_vec` renders f64 via Ryu shortest roundtrip, which is
    // deterministic for identical inputs — the property the verdict cache
    // depends on (same spec+property bytes, same addr, no re-solve).
    Ok(serde_json::to_vec(&sorted)?)
}

/// Convenience: any serializable value to canonical bytes.
///
/// Caller discipline: all floats reachable from `T` must sit behind
/// [`FiniteF32`]/[`FiniteF64`]. A bare `f32`/`f64` field carrying NaN would be
/// silently mapped to `null` by `serde_json::to_value` *before* this function
/// ever sees it — the newtypes are what make that failure loud.
pub fn to_canonical_bytes<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, CanonicalError> {
    let v = serde_json::to_value(value).map_err(CanonicalError::Serialization)?;
    canonical_json(&v)
}

fn sort_value(value: &Value, path: &str) -> Result<Value, CanonicalError> {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = serde_json::Map::with_capacity(map.len());
            for key in keys {
                let child_path = format!("{path}.{key}");
                out.insert(key.clone(), sort_value(&map[key], &child_path)?);
            }
            Ok(Value::Object(out))
        }
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(i, item)| sort_value(item, &format!("{path}[{i}]")))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        Value::Number(number) => {
            // Integers pass through untouched: 1u64 must stay "1", never "1.0".
            if number.is_i64() || number.is_u64() {
                return Ok(Value::Number(number.clone()));
            }
            let float = number
                .as_f64()
                .expect("non-integer JSON number is f64-representable");
            if !float.is_finite() {
                return Err(CanonicalError::NonFiniteFloat {
                    path: path.to_string(),
                    value: number.to_string(),
                });
            }
            // f64-widened shortest roundtrip. Widening is exact, so identical
            // f32 inputs always yield identical bytes; f32 and f64 spellings
            // of "the same" decimal intentionally do not (exactness over
            // prettiness — the verdict cache compares identical specs).
            Ok(Value::Number(
                serde_json::Number::from_f64(float).expect("finite checked above"),
            ))
        }
        scalar => Ok(scalar.clone()),
    }
}

/// Predicates of `spec` in sorted name order (`SCOPE_F1 §4`).
///
/// `RuleSpec.predicates` is a `HashMap`: iteration order is randomized per
/// process, so any derivation, emission, or canonicalization path that walks
/// it directly would produce run-dependent bytes and a verdict cache that
/// never hits. All such paths go through here.
pub fn sorted_predicates(spec: &RuleSpec) -> Vec<(&str, &PredicateSpec)> {
    let mut entries: Vec<(&str, &PredicateSpec)> = spec
        .predicates
        .iter()
        .map(|(name, pred)| (name.as_str(), pred))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries
}

/// Canonical bytes for a whole `RuleSpec`: the JSON form plus a sorted
/// sidecar of non-finite float paths. `serde_json` maps NaN/±Inf to `null`
/// silently, which would alias distinct specs (e.g. weight `None` vs
/// `Some(NaN)`) to one addr and corrupt the verdict cache — the sidecar
/// keeps canonical bytes injective. L0.7 still reports NaN weights as
/// disproved properties; addressing stays sound underneath it.
pub fn canonical_spec_bytes(spec: &RuleSpec) -> Result<Vec<u8>, CanonicalError> {
    let value = serde_json::to_value(spec).map_err(CanonicalError::Serialization)?;
    let paths = non_finite_paths(spec);
    let mut wrapper = serde_json::Map::with_capacity(2);
    wrapper.insert(
        "non_finite".to_string(),
        Value::Array(paths.into_iter().map(Value::String).collect()),
    );
    wrapper.insert("spec".to_string(), value);
    canonical_json(&Value::Object(wrapper))
}

/// JSON paths of every non-finite float in `spec` (sorted). Exhaustive over
/// `PredicateSpec` — a new float-bearing variant fails to compile here until
/// classified (fail-closed at compile time, not silent at runtime).
fn non_finite_paths(spec: &RuleSpec) -> Vec<String> {
    fn push_if_non_finite(paths: &mut Vec<String>, path: String, value: f32) {
        if !value.is_finite() {
            paths.push(path);
        }
    }

    let mut paths = Vec::new();
    for (rule_idx, rule) in spec.rules.iter().enumerate() {
        if let Some(weight) = rule.weight {
            push_if_non_finite(&mut paths, format!("rules[{rule_idx}].weight"), weight);
        }
        for (lit_idx, lit) in rule.body.iter().enumerate() {
            for (arg_idx, arg) in lit.args.iter().enumerate() {
                if let crate::compiler::Argument::Constant(value) = arg {
                    push_if_non_finite(
                        &mut paths,
                        format!("rules[{rule_idx}].body[{lit_idx}].args[{arg_idx}]"),
                        *value,
                    );
                }
            }
        }
    }
    let mut names: Vec<&String> = spec.predicates.keys().collect();
    names.sort();
    for name in names {
        let base = format!("predicates[\"{name}\"]");
        match &spec.predicates[name] {
            PredicateSpec::Threshold {
                threshold,
                sharpness,
                ..
            } => {
                push_if_non_finite(&mut paths, format!("{base}.threshold"), *threshold);
                push_if_non_finite(&mut paths, format!("{base}.sharpness"), *sharpness);
            }
            PredicateSpec::Learned { .. }
            | PredicateSpec::LearnedSimilarity { .. }
            | PredicateSpec::GraphNeural { .. }
            | PredicateSpec::Composite { .. }
            | PredicateSpec::External => {}
            PredicateSpec::LearnedProjection {
                attention_dropout,
                dropout,
                ..
            } => {
                if let Some(value) = attention_dropout {
                    push_if_non_finite(&mut paths, format!("{base}.attention_dropout"), *value);
                }
                if let Some(value) = dropout {
                    push_if_non_finite(&mut paths, format!("{base}.dropout"), *value);
                }
            }
            PredicateSpec::TierLookup {
                thresholds,
                values,
                sharpness,
                ..
            } => {
                for (idx, value) in thresholds.iter().enumerate() {
                    push_if_non_finite(&mut paths, format!("{base}.thresholds[{idx}]"), *value);
                }
                for (idx, value) in values.iter().enumerate() {
                    push_if_non_finite(&mut paths, format!("{base}.values[{idx}]"), *value);
                }
                push_if_non_finite(&mut paths, format!("{base}.sharpness"), *sharpness);
            }
            PredicateSpec::PairwiseDifference {
                threshold_pct,
                sharpness,
                ..
            } => {
                push_if_non_finite(&mut paths, format!("{base}.threshold_pct"), *threshold_pct);
                push_if_non_finite(&mut paths, format!("{base}.sharpness"), *sharpness);
            }
            PredicateSpec::CascadingLookup { priorities, .. } => {
                for (idx, value) in priorities.iter().enumerate() {
                    push_if_non_finite(&mut paths, format!("{base}.priorities[{idx}]"), *value);
                }
            }
        }
    }
    paths.sort();
    paths
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::{Argument, Literal, PredicateSpec, Rule};
    use serde_json::json;

    #[test]
    fn key_order_does_not_affect_bytes() {
        let mut first = serde_json::Map::new();
        first.insert("b".to_string(), json!(2));
        first.insert("a".to_string(), json!(1));
        let mut second = serde_json::Map::new();
        second.insert("a".to_string(), json!(1));
        second.insert("b".to_string(), json!(2));
        assert_eq!(
            canonical_json(&Value::Object(first)).unwrap(),
            canonical_json(&Value::Object(second)).unwrap()
        );
    }

    #[test]
    fn nested_objects_sort_recursively() {
        let value = json!({"z": {"b": 1, "a": 2}, "a": 0});
        assert_eq!(
            canonical_json(&value).unwrap(),
            br#"{"a":0,"z":{"a":2,"b":1}}"#
        );
    }

    #[test]
    fn array_order_is_meaning_and_preserved() {
        let forward = json!({"items": [1, 2, 3]});
        let backward = json!({"items": [3, 2, 1]});
        assert_ne!(
            canonical_json(&forward).unwrap(),
            canonical_json(&backward).unwrap()
        );
    }

    #[test]
    fn integers_pass_through_unfloated() {
        assert_eq!(canonical_json(&json!(1u64)).unwrap(), b"1");
        assert_eq!(canonical_json(&json!(-7i64)).unwrap(), b"-7");
    }

    #[test]
    fn identical_f32_inputs_yield_identical_bytes() {
        let a = json!(0.1f32);
        let b = serde_json::to_value(0.1f32).unwrap();
        assert_eq!(canonical_json(&a).unwrap(), canonical_json(&b).unwrap());
    }

    #[test]
    fn f32_and_f64_spellings_differ_by_design() {
        // Exactness over prettiness: 0.1f32 widens to 0.10000000149011612,
        // which must not alias 0.1f64. The verdict cache compares identical
        // specs, so this never costs a hit — it only prevents a false one.
        let f32_bytes = canonical_json(&serde_json::to_value(0.1f32).unwrap()).unwrap();
        let f64_bytes = canonical_json(&json!(0.1f64)).unwrap();
        assert_ne!(f32_bytes, f64_bytes);
        assert_eq!(f64_bytes, b"0.1");
    }

    #[test]
    fn finite_rejects_nan_on_serialize() {
        // `FiniteF32::new` fails closed, so a NaN can only arrive via an
        // already-broken value — exercise the Serialize guard directly.
        let broken = FiniteF32(f32::NAN);
        assert!(serde_json::to_value(broken).is_err());
        assert!(FiniteF32::new(f32::NAN).is_err());
        assert!(FiniteF32::new(f32::INFINITY).is_err());
        assert!(FiniteF64::new(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn finite_rejects_null_and_nan_on_deserialize() {
        assert!(serde_json::from_value::<FiniteF32>(Value::Null).is_err());
        assert!(serde_json::from_value::<FiniteF32>(json!("0.5")).is_err());
        assert_eq!(
            serde_json::from_value::<FiniteF32>(json!(0.5))
                .unwrap()
                .get(),
            0.5
        );
    }

    #[test]
    fn to_canonical_bytes_roundtrips_a_struct() {
        #[derive(Serialize)]
        struct Threshold {
            input: String,
            threshold: FiniteF32,
        }
        let bytes = to_canonical_bytes(&Threshold {
            input: "score".to_string(),
            threshold: FiniteF32::new(0.5).unwrap(),
        })
        .unwrap();
        assert_eq!(bytes, br#"{"input":"score","threshold":0.5}"#);
    }

    #[test]
    fn sorted_predicates_returns_name_order() {
        let mut spec = RuleSpec::new("ordering");
        for name in ["zebra", "apple", "mango"] {
            spec.add_predicate(
                name,
                PredicateSpec::Threshold {
                    input: "x".to_string(),
                    threshold: 0.5,
                    greater_than: true,
                    sharpness: 10.0,
                },
            );
        }
        let names: Vec<&str> = sorted_predicates(&spec).iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["apple", "mango", "zebra"]);
    }

    #[test]
    fn spec_bytes_distinguish_nan_from_none() {
        // serde_json maps NaN to null silently; the non_finite sidecar keeps
        // canonical bytes injective (None-weight vs NaN-weight must not alias
        // to one addr — verdict-cache soundness depends on it).
        let mut clean = RuleSpec::new("nan");
        clean.add_rule(Rule::new("h", vec![]));
        let mut nan = RuleSpec::new("nan");
        nan.add_rule(Rule::new("h", vec![]).with_weight(f32::NAN));
        assert_ne!(
            canonical_spec_bytes(&clean).unwrap(),
            canonical_spec_bytes(&nan).unwrap()
        );
        // ...and identical specs (NaN included) hash identically.
        let mut nan_again = RuleSpec::new("nan");
        nan_again.add_rule(Rule::new("h", vec![]).with_weight(f32::NAN));
        assert_eq!(
            canonical_spec_bytes(&nan).unwrap(),
            canonical_spec_bytes(&nan_again).unwrap()
        );
    }

    #[test]
    fn non_finite_paths_name_predicate_floats() {
        let mut spec = RuleSpec::new("paths");
        spec.add_predicate(
            "tier",
            PredicateSpec::TierLookup {
                input: "x".to_string(),
                thresholds: vec![50.0, f32::INFINITY],
                values: vec![1.0, 0.5],
                sharpness: 5.0,
                interpolate: false,
            },
        );
        spec.add_rule(Rule::new(
            "h",
            vec![Literal::positive(
                "tier",
                vec![Argument::Variable("X".to_string())],
            )],
        ));
        let bytes = canonical_spec_bytes(&spec).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            value["non_finite"],
            serde_json::json!(["predicates[\"tier\"].thresholds[1]"])
        );
    }
}

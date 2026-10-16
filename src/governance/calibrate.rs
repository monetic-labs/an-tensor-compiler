//! O4-lite calibrator (F-2 Slice 6).
//!
//! The F-1.5 assay assumed uniform leaf ε=0.1 because no measured
//! calibration existed — and the QIP readout showed why that is load-bearing:
//! hot scores (0.99 medians) collapse any abstention band to nothing. This
//! module measures per-predicate miscalibration (binned ECE) and maps it to
//! measured leaf ε: `ε_measured = ε_config × (1 − ECE)`.
//!
//! Design points worth restating:
//!
//! - Meter, not model: binning math only, no ML deps. The autonomous
//!   training tracker (`autonomous::training`) carries the same math behind
//!   a candle device; this is the governance-suitable extraction — 20 lines
//!   of binning with none of the weight.
//! - Below `min_samples`, the configured ε (prior) holds: start from prior,
//!   tune on traffic. Measured values take over only with evidence.
//! - The mapping is monotone and bounded: ECE=0 keeps full configured ε;
//!   ECE→1 drives ε→0, so a shattered leaf claims no clearance, its bands
//!   swallow every reading, and its evals abstain. Fail-safe, exactly as
//!   the hot-scores lesson demands.
//! - Confidence convention: [`Calibrator::confidence_from_soft`] maps a
//!   soft value to distance-from-threshold (`|soft − 0.5| × 2`). Success is
//!   caller-asserted — and source quality matters: agreement-fed histories
//!   measure layer *consistency*, not truth calibration. Sealed labels
//!   (slice 9) and human stamps (slice 8) supersede agreement as success
//!   sources when they arrive; until then the meter says what it sees.
//! - The meter never breaks the eval path: out-of-range confidences clamp
//!   (debug-asserted, rounding artifacts are not worth failing capture
//!   over), histories are bounded per predicate, and all sharing is via
//!   interior mutability so capture takes `&Calibrator`.

use std::collections::{BTreeMap, HashMap, VecDeque};

use parking_lot::RwLock;

use super::margin::{MarginConfig, NodeMargin};

/// One predicate's calibration history: bounded (confidence, outcome) samples.
#[derive(Debug, Clone)]
pub struct EceHistory {
    samples: VecDeque<(f32, bool)>,
    cap: usize,
}

impl EceHistory {
    /// Empty history keeping at most `cap` samples (oldest evicted first).
    pub fn new(cap: usize) -> Self {
        Self {
            samples: VecDeque::new(),
            cap: cap.max(1),
        }
    }

    /// Record one sample. Confidence is clamped to [0, 1] (see module docs).
    pub fn record(&mut self, confidence: f64, success: bool) {
        debug_assert!(
            (0.0..=1.0).contains(&confidence),
            "confidence {confidence} outside [0, 1]"
        );
        let clamped = (confidence as f32).clamp(0.0, 1.0);
        if self.samples.len() >= self.cap {
            self.samples.pop_front();
        }
        self.samples.push_back((clamped, success));
    }

    /// Binned expected calibration error: Σ_bins weight × |avg_conf − acc|.
    /// Always in [0, 1]; 0.0 on empty histories (no evidence, no error).
    pub fn ece(&self, n_bins: usize) -> f64 {
        if self.samples.is_empty() || n_bins == 0 {
            return 0.0;
        }
        let mut bins: Vec<Vec<(f32, bool)>> = vec![Vec::new(); n_bins];
        for &(conf, success) in &self.samples {
            let idx = ((conf * n_bins as f32) as usize).min(n_bins - 1);
            bins[idx].push((conf, success));
        }
        let total = self.samples.len() as f64;
        bins.iter()
            .filter(|bin| !bin.is_empty())
            .map(|bin| {
                let avg_conf = bin.iter().map(|(c, _)| *c as f64).sum::<f64>() / bin.len() as f64;
                let accuracy = bin.iter().filter(|(_, s)| *s).count() as f64 / bin.len() as f64;
                (bin.len() as f64 / total) * (avg_conf - accuracy).abs()
            })
            .sum()
    }

    /// Number of recorded samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether no samples have been recorded.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Per-predicate ECE meter with an ECE→ε mapping.
///
/// Clone is cheap-ish (shares nothing — histories clone); the capture path
/// shares one instance by reference instead.
#[derive(Debug)]
pub struct Calibrator {
    histories: RwLock<HashMap<String, EceHistory>>,
    /// Prior leaf ε (the F-1 configured constant) — holds until a predicate
    /// accumulates `min_samples`.
    config_eps: f64,
    min_samples: usize,
    bins: usize,
    cap: usize,
}

impl Calibrator {
    /// A calibrator over `config_eps` priors: `min_samples` per predicate
    /// before measured values take over, `bins` ECE bins, `cap` samples
    /// retained per predicate.
    pub fn new(config_eps: f64, min_samples: usize, bins: usize, cap: usize) -> Self {
        Self {
            histories: RwLock::new(HashMap::new()),
            config_eps,
            min_samples: min_samples.max(1),
            bins: bins.max(1),
            cap: cap.max(1),
        }
    }

    /// Slice-6 provisional hyperparameters: ε=0.1 prior (the F-1.5 assay
    /// assumption, kept until DRI rulings say otherwise), 50-sample minimum,
    /// 10 bins, 10k retained samples per predicate.
    pub fn provisional() -> Self {
        Self::new(0.1, 50, 10, 10_000)
    }

    /// Soft value → confidence: distance from τ=0.5 scaled to [0, 1].
    pub fn confidence_from_soft(soft: f64) -> f64 {
        ((soft - 0.5).abs() * 2.0).clamp(0.0, 1.0)
    }

    /// Record one (confidence, outcome) sample for `predicate`.
    pub fn record(&self, predicate: &str, confidence: f64, success: bool) {
        let mut histories = self.histories.write();
        histories
            .entry(predicate.to_string())
            .or_insert_with(|| EceHistory::new(self.cap))
            .record(confidence, success);
    }

    /// Measured ECE for `predicate`, or `None` below `min_samples`
    /// (no evidence — the prior holds).
    pub fn ece(&self, predicate: &str) -> Option<f64> {
        let histories = self.histories.read();
        let history = histories.get(predicate)?;
        if history.len() < self.min_samples {
            return None;
        }
        Some(history.ece(self.bins))
    }

    /// Leaf margin for `predicate`: measured `ε_config × (1 − ECE)` with
    /// sufficient samples, else the configured prior. Symmetric — per-side
    /// calibration arrives with richer outcome data.
    pub fn leaf_margin(&self, predicate: &str) -> NodeMargin {
        match self.ece(predicate) {
            Some(ece) => NodeMargin::symmetric(self.config_eps * (1.0 - ece).max(0.0)),
            None => NodeMargin::symmetric(self.config_eps),
        }
    }

    /// The O4 wiring, concretely: clone `base` with measured leaf margins
    /// installed as per-predicate overrides (predicates below `min_samples`
    /// keep the base values). Feed the result to margin propagation and the
    /// abstention band consumes measured ε.
    pub fn apply_to_margin_config(&self, base: &MarginConfig) -> MarginConfig {
        let mut overrides: BTreeMap<String, NodeMargin> = base.overrides.clone();
        let predicates: Vec<String> = self.histories.read().keys().cloned().collect();
        for predicate in predicates {
            if self.ece(&predicate).is_some() {
                overrides.insert(predicate.clone(), self.leaf_margin(&predicate));
            }
        }
        MarginConfig {
            default_eps: base.default_eps,
            overrides,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibrator() -> Calibrator {
        Calibrator::new(0.1, 10, 10, 1000)
    }

    #[test]
    fn perfect_calibration_measures_zero() {
        let mut history = EceHistory::new(100);
        for _ in 0..10 {
            history.record(1.0, true);
            history.record(0.0, false);
        }
        assert!(history.ece(10) < 1e-9, "ece = {}", history.ece(10));
    }

    #[test]
    fn hot_overconfidence_measures_high() {
        // The QIP lesson in miniature: 0.99 confidence at 50% accuracy.
        let mut history = EceHistory::new(100);
        for i in 0..20 {
            history.record(0.99, i % 2 == 0);
        }
        let ece = history.ece(10);
        assert!((ece - 0.49).abs() < 1e-6, "expected ~0.49, got {ece}");
    }

    #[test]
    fn empty_history_measures_zero() {
        assert_eq!(EceHistory::new(100).ece(10), 0.0);
    }

    #[test]
    fn history_is_bounded() {
        let mut history = EceHistory::new(8);
        for i in 0..20 {
            history.record(0.5, i % 2 == 0);
        }
        assert_eq!(history.len(), 8);
    }

    #[test]
    fn prior_holds_below_min_samples() {
        let cal = calibrator();
        assert_eq!(cal.ece("p"), None);
        cal.record("p", 0.99, false);
        assert_eq!(cal.ece("p"), None, "one sample is not evidence");
        assert_eq!(cal.leaf_margin("p"), NodeMargin::symmetric(0.1));
    }

    #[test]
    fn measured_eps_shrinks_with_ece() {
        let cal = calibrator();
        for i in 0..10 {
            cal.record("hot", 0.99, i % 2 == 0);
        }
        let ece = cal.ece("hot").expect("ten samples suffice");
        assert!(ece > 0.4, "ece = {ece}");
        let measured = cal.leaf_margin("hot");
        assert_eq!(measured, NodeMargin::symmetric(0.1 * (1.0 - ece)));
        assert!(
            measured.eps_true < 0.1 && measured.eps_true > 0.0,
            "shrunk but positive: {:?}",
            measured
        );
    }

    #[test]
    fn well_calibrated_keeps_full_eps() {
        let cal = calibrator();
        for _ in 0..5 {
            cal.record("good", 1.0, true);
            cal.record("good", 0.0, false);
        }
        let measured = cal.leaf_margin("good");
        assert!((measured.eps_true - 0.1).abs() < 1e-9, "{measured:?}");
    }

    #[test]
    fn confidence_from_soft_convention() {
        assert_eq!(Calibrator::confidence_from_soft(0.5), 0.0);
        assert_eq!(Calibrator::confidence_from_soft(1.0), 1.0);
        assert_eq!(Calibrator::confidence_from_soft(0.0), 1.0);
        assert!((Calibrator::confidence_from_soft(0.82) - 0.64).abs() < 1e-9);
    }

    #[test]
    fn apply_installs_measured_overrides_only() {
        let cal = calibrator();
        for i in 0..10 {
            cal.record("hot", 0.99, i % 2 == 0);
        }
        let out = cal.apply_to_margin_config(&MarginConfig::uniform(0.1));
        assert_eq!(out.default_eps, 0.1);
        let hot = out.overrides.get("hot").expect("measured override");
        assert!(hot.eps_true < 0.1, "{hot:?}");
        assert!(
            !out.overrides.contains_key("never_seen"),
            "unmeasured predicates keep base values"
        );
    }
}

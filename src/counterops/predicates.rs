//! CounterOps Predicate Specifications
//!
//! Defines the standard predicates for security operations.

use crate::compiler::{Activation, Aggregation, PredicateSpec, SimilarityMethod};

/// Standard security/counterops predicates
#[derive(Debug, Clone)]
pub enum ThreatPredicate {
    /// Overall threat level assessment
    ThreatLevel,
    /// Statistical anomaly detection
    AnomalyScore,
    /// Prompt injection detection
    InjectionAttempt,
    /// Data exfiltration risk
    DataExfilRisk,
    /// Rate limiting trigger
    RateLimitExceeded,
    /// Behavior pattern deviation (graph-based)
    PatternDeviation,
    /// Should enter defensive mode?
    DefenseMode,
    /// Authentication anomaly
    AuthAnomaly,
    /// Custom predicate with spec
    Custom(String, PredicateSpec),
}

impl ThreatPredicate {
    /// Get the predicate name
    pub fn name(&self) -> &str {
        match self {
            Self::ThreatLevel => "threat_level",
            Self::AnomalyScore => "anomaly_score",
            Self::InjectionAttempt => "injection_attempt",
            Self::DataExfilRisk => "data_exfil_risk",
            Self::RateLimitExceeded => "rate_limit_exceeded",
            Self::PatternDeviation => "pattern_deviation",
            Self::DefenseMode => "defense_mode",
            Self::AuthAnomaly => "auth_anomaly",
            Self::Custom(name, _) => name,
        }
    }

    /// Get the default PredicateSpec for this predicate
    pub fn default_spec(&self) -> PredicateSpec {
        match self {
            Self::ThreatLevel => Self::threat_level_spec(64),
            Self::AnomalyScore => Self::anomaly_score_spec(128),
            Self::InjectionAttempt => Self::injection_attempt_spec(1536),
            Self::DataExfilRisk => Self::data_exfil_risk_spec(256),
            Self::RateLimitExceeded => Self::rate_limit_exceeded_spec(),
            Self::PatternDeviation => Self::pattern_deviation_spec(64),
            Self::DefenseMode => Self::defense_mode_spec(32),
            Self::AuthAnomaly => Self::auth_anomaly_spec(48),
            Self::Custom(_, spec) => spec.clone(),
        }
    }

    /// Threat level: Combined threat assessment
    ///
    /// Uses LearnedProjection with attention to aggregate multiple
    /// security signals into a unified threat score (0.0 = safe, 1.0 = critical).
    pub fn threat_level_spec(feature_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["security_features".into()],
            input_dim: feature_dim,
            hidden_dim: 128,
            activation: Activation::ReLU,
            attention_heads: Some(4),
            attention_dropout: Some(0.1),
            layer_norm: Some(true),
            dropout: Some(0.2),
            residual: Some(true),
            conditioning_dim: None,
            conditioning_type: None,
            film_identity_init: None,
        }
    }

    /// Anomaly score: Statistical deviation from normal behavior
    ///
    /// Uses LearnedProjection on behavioral embeddings to detect
    /// unusual patterns that may indicate an attack.
    pub fn anomaly_score_spec(embedding_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["behavior_embedding".into()],
            input_dim: embedding_dim,
            hidden_dim: 64,
            activation: Activation::ReLU,
            attention_heads: None,
            attention_dropout: None,
            layer_norm: Some(true),
            dropout: Some(0.1),
            residual: None,
            conditioning_dim: None,
            conditioning_type: None,
            film_identity_init: None,
        }
    }

    /// Injection attempt: Prompt injection detection
    ///
    /// Uses LearnedSimilarity to compare input against known
    /// attack pattern embeddings.
    pub fn injection_attempt_spec(embedding_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedSimilarity {
            left: "input_embedding".into(),
            right: "attack_patterns".into(),
            dim: embedding_dim,
            method: SimilarityMethod::Learned,
        }
    }

    /// Data exfiltration risk: Output analysis for data leaks
    ///
    /// Uses LearnedProjection on output content analysis to detect
    /// potential data exfiltration attempts.
    pub fn data_exfil_risk_spec(feature_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["output_analysis".into()],
            input_dim: feature_dim,
            hidden_dim: 128,
            activation: Activation::GELU,
            attention_heads: Some(2),
            attention_dropout: Some(0.1),
            layer_norm: Some(true),
            dropout: Some(0.15),
            residual: Some(true),
            conditioning_dim: None,
            conditioning_type: None,
            film_identity_init: None,
        }
    }

    /// Rate limit exceeded: Simple threshold on request rate
    pub fn rate_limit_exceeded_spec() -> PredicateSpec {
        PredicateSpec::Threshold {
            input: "request_rate".into(),
            threshold: 100.0, // requests per minute
            greater_than: true,
            sharpness: 10.0,
        }
    }

    /// Pattern deviation: Graph-based behavior analysis
    ///
    /// Uses GraphNeural to detect deviations from expected
    /// behavioral patterns by analyzing the action graph.
    pub fn pattern_deviation_spec(feature_dim: usize) -> PredicateSpec {
        PredicateSpec::GraphNeural {
            node_features: "action_features".into(),
            adjacency: "action_graph".into(),
            feature_dim,
            hidden_dim: 32,
            num_layers: 2,
            aggregation: Aggregation::Mean,
        }
    }

    /// Defense mode: Combined signal to trigger defensive posture
    ///
    /// Uses LearnedProjection on aggregated security signals to
    /// determine if defensive mode should be activated.
    pub fn defense_mode_spec(feature_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["combined_signals".into()],
            input_dim: feature_dim,
            hidden_dim: 64,
            activation: Activation::ReLU,
            attention_heads: None,
            attention_dropout: None,
            layer_norm: Some(true),
            dropout: Some(0.1),
            residual: None,
            conditioning_dim: None,
            conditioning_type: None,
            film_identity_init: None,
        }
    }

    /// Authentication anomaly: Unusual auth patterns
    ///
    /// Uses LearnedProjection on authentication features to detect
    /// suspicious login patterns.
    pub fn auth_anomaly_spec(feature_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["auth_features".into()],
            input_dim: feature_dim,
            hidden_dim: 32,
            activation: Activation::ReLU,
            attention_heads: None,
            attention_dropout: None,
            layer_norm: None,
            dropout: Some(0.1),
            residual: None,
            conditioning_dim: None,
            conditioning_type: None,
            film_identity_init: None,
        }
    }
}

/// Threat detection specification for customization
#[derive(Debug, Clone)]
pub struct ThreatSpec {
    /// Feature dimension for security signals
    pub feature_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Number of attention heads (None = no attention)
    pub attention_heads: Option<usize>,
    /// Dropout rate
    pub dropout: f32,
}

impl Default for ThreatSpec {
    fn default() -> Self {
        Self {
            feature_dim: 64,
            hidden_dim: 128,
            attention_heads: Some(4),
            dropout: 0.2,
        }
    }
}

impl ThreatSpec {
    /// Convert to PredicateSpec
    pub fn to_predicate_spec(&self) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["security_features".into()],
            input_dim: self.feature_dim,
            hidden_dim: self.hidden_dim,
            activation: Activation::ReLU,
            attention_heads: self.attention_heads,
            attention_dropout: Some(0.1),
            layer_norm: Some(true),
            dropout: Some(self.dropout),
            residual: Some(true),
            conditioning_dim: None,
            conditioning_type: None,
            film_identity_init: None,
        }
    }
}

/// Anomaly detection specification for customization
#[derive(Debug, Clone)]
pub struct AnomalySpec {
    /// Embedding dimension for behavior vectors
    pub embedding_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Sensitivity factor (higher = more sensitive)
    pub sensitivity: f32,
}

impl Default for AnomalySpec {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            hidden_dim: 64,
            sensitivity: 1.0,
        }
    }
}

impl AnomalySpec {
    /// Convert to PredicateSpec
    pub fn to_predicate_spec(&self) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["behavior_embedding".into()],
            input_dim: self.embedding_dim,
            hidden_dim: self.hidden_dim,
            activation: Activation::ReLU,
            attention_heads: None,
            attention_dropout: None,
            layer_norm: Some(true),
            dropout: Some(0.1),
            residual: None,
            conditioning_dim: None,
            conditioning_type: None,
            film_identity_init: None,
        }
    }
}

/// Defense mode specification for customization
#[derive(Debug, Clone)]
pub struct DefenseSpec {
    /// Number of input signals
    pub signal_count: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Threshold for triggering defense mode (0.0-1.0)
    pub trigger_threshold: f32,
}

impl Default for DefenseSpec {
    fn default() -> Self {
        Self {
            signal_count: 32,
            hidden_dim: 64,
            trigger_threshold: 0.7,
        }
    }
}

impl DefenseSpec {
    /// Convert to PredicateSpec
    pub fn to_predicate_spec(&self) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["combined_signals".into()],
            input_dim: self.signal_count,
            hidden_dim: self.hidden_dim,
            activation: Activation::ReLU,
            attention_heads: None,
            attention_dropout: None,
            layer_norm: Some(true),
            dropout: Some(0.1),
            residual: None,
            conditioning_dim: None,
            conditioning_type: None,
            film_identity_init: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predicate_names() {
        assert_eq!(ThreatPredicate::ThreatLevel.name(), "threat_level");
        assert_eq!(ThreatPredicate::AnomalyScore.name(), "anomaly_score");
        assert_eq!(
            ThreatPredicate::InjectionAttempt.name(),
            "injection_attempt"
        );
    }

    #[test]
    fn test_default_specs() {
        let spec = ThreatPredicate::ThreatLevel.default_spec();
        match spec {
            PredicateSpec::LearnedProjection { hidden_dim, .. } => {
                assert_eq!(hidden_dim, 128);
            }
            _ => panic!("Expected LearnedProjection"),
        }
    }

    #[test]
    fn test_pattern_deviation_uses_gnn() {
        let spec = ThreatPredicate::PatternDeviation.default_spec();
        match spec {
            PredicateSpec::GraphNeural { num_layers, .. } => {
                assert_eq!(num_layers, 2);
            }
            _ => panic!("Expected GraphNeural"),
        }
    }

    #[test]
    fn test_threat_spec_customization() {
        let spec = ThreatSpec {
            feature_dim: 128,
            hidden_dim: 256,
            attention_heads: Some(8),
            dropout: 0.3,
        };

        let pred_spec = spec.to_predicate_spec();
        match pred_spec {
            PredicateSpec::LearnedProjection {
                input_dim,
                hidden_dim,
                attention_heads,
                ..
            } => {
                assert_eq!(input_dim, 128);
                assert_eq!(hidden_dim, 256);
                assert_eq!(attention_heads, Some(8));
            }
            _ => panic!("Expected LearnedProjection"),
        }
    }

    #[test]
    fn test_defense_spec_defaults() {
        let spec = DefenseSpec::default();
        assert_eq!(spec.trigger_threshold, 0.7);
    }
}

//! Interaction Predicate Specifications
//!
//! Defines the standard predicates for conversational AI.

use crate::compiler::{Activation, PredicateSpec, SimilarityMethod};

/// Standard interaction predicates
#[derive(Debug, Clone)]
pub enum InteractionPredicate {
    /// Message safety predicate
    MessageSafe,
    /// Intent clarity predicate
    IntentClear,
    /// Context freshness predicate
    ContextFresh,
    /// Response coherence predicate
    ResponseCoherent,
    /// Topic drift detection
    TopicDrift,
    /// Escalation needed predicate
    EscalationNeeded,
    /// User frustration detection
    UserFrustrated,
    /// Custom predicate with spec
    Custom(String, PredicateSpec),
}

impl InteractionPredicate {
    /// Get the predicate name
    pub fn name(&self) -> &str {
        match self {
            Self::MessageSafe => "message_safe",
            Self::IntentClear => "intent_clear",
            Self::ContextFresh => "context_fresh",
            Self::ResponseCoherent => "response_coherent",
            Self::TopicDrift => "topic_drift",
            Self::EscalationNeeded => "escalation_needed",
            Self::UserFrustrated => "user_frustrated",
            Self::Custom(name, _) => name,
        }
    }

    /// Get the default PredicateSpec for this predicate
    pub fn default_spec(&self) -> PredicateSpec {
        match self {
            Self::MessageSafe => Self::message_safe_spec(1536),
            Self::IntentClear => Self::intent_clear_spec(1536),
            Self::ContextFresh => Self::context_fresh_spec(),
            Self::ResponseCoherent => Self::response_coherent_spec(1536),
            Self::TopicDrift => Self::topic_drift_spec(1536),
            Self::EscalationNeeded => Self::escalation_needed_spec(32),
            Self::UserFrustrated => Self::user_frustrated_spec(64),
            Self::Custom(_, spec) => spec.clone(),
        }
    }

    /// Message safety: Is this message safe to process?
    ///
    /// Uses LearnedProjection with attention to detect:
    /// - Harmful content
    /// - Prompt injection attempts
    /// - Personally identifiable information
    /// - Policy violations
    pub fn message_safe_spec(embedding_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["content_embedding".into()],
            input_dim: embedding_dim,
            hidden_dim: 256,
            activation: Activation::GELU,
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

    /// Intent clarity: Is the user's intent well-defined?
    ///
    /// Uses LearnedSimilarity to compare:
    /// - User's stated intent
    /// - Context of the conversation
    ///
    /// High similarity indicates clear intent.
    pub fn intent_clear_spec(embedding_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedSimilarity {
            left: "intent_embedding".into(),
            right: "context_embedding".into(),
            dim: embedding_dim,
            method: SimilarityMethod::Learned,
        }
    }

    /// Context freshness: Is the conversation context still relevant?
    ///
    /// Simple threshold on last access time.
    /// Configurable staleness threshold (default: 30 minutes).
    pub fn context_fresh_spec() -> PredicateSpec {
        PredicateSpec::Threshold {
            input: "context_age_minutes".into(),
            threshold: 30.0,
            greater_than: false, // Fresh if age < threshold
            sharpness: 5.0,
        }
    }

    /// Response coherence: Does the response match the query?
    ///
    /// Uses LearnedSimilarity with bilinear method for
    /// asymmetric query-response matching.
    pub fn response_coherent_spec(embedding_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedSimilarity {
            left: "query_embedding".into(),
            right: "response_embedding".into(),
            dim: embedding_dim,
            method: SimilarityMethod::Bilinear,
        }
    }

    /// Topic drift: Has the conversation drifted off-topic?
    ///
    /// Uses LearnedProjection on conversation window embedding
    /// to detect significant topic changes.
    pub fn topic_drift_spec(embedding_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["window_embedding".into()],
            input_dim: embedding_dim,
            hidden_dim: 128,
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

    /// Escalation needed: Should a human be involved?
    ///
    /// Uses LearnedProjection on risk features including:
    /// - Uncertainty score
    /// - Complexity estimate
    /// - Previous failure count
    /// - Sensitivity indicators
    pub fn escalation_needed_spec(feature_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["risk_features".into()],
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

    /// User frustrated: Is the user showing frustration signals?
    ///
    /// Uses LearnedProjection on sentiment features including:
    /// - Sentiment scores
    /// - Message length patterns
    /// - Repetition indicators
    /// - Explicit frustration markers
    pub fn user_frustrated_spec(feature_dim: usize) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["sentiment_features".into()],
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

/// Safety specification for customization
#[derive(Debug, Clone)]
pub struct SafetySpec {
    /// Embedding dimension for content analysis
    pub embedding_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Number of attention heads (None = no attention)
    pub attention_heads: Option<usize>,
    /// Dropout rate
    pub dropout: f32,
    /// Enable residual connections
    pub residual: bool,
}

impl Default for SafetySpec {
    fn default() -> Self {
        Self {
            embedding_dim: 1536,
            hidden_dim: 256,
            attention_heads: Some(4),
            dropout: 0.2,
            residual: true,
        }
    }
}

impl SafetySpec {
    /// Convert to PredicateSpec
    pub fn to_predicate_spec(&self) -> PredicateSpec {
        PredicateSpec::LearnedProjection {
            inputs: vec!["content_embedding".into()],
            input_dim: self.embedding_dim,
            hidden_dim: self.hidden_dim,
            activation: Activation::GELU,
            attention_heads: self.attention_heads,
            attention_dropout: Some(0.1),
            layer_norm: Some(true),
            dropout: Some(self.dropout),
            residual: Some(self.residual),
            conditioning_dim: None,
            conditioning_type: None,
            film_identity_init: None,
        }
    }
}

/// Context specification for customization
#[derive(Debug, Clone)]
pub struct ContextSpec {
    /// Staleness threshold in minutes
    pub freshness_threshold_minutes: f32,
    /// Sharpness of threshold transition
    pub sharpness: f32,
}

impl Default for ContextSpec {
    fn default() -> Self {
        Self {
            freshness_threshold_minutes: 30.0,
            sharpness: 5.0,
        }
    }
}

impl ContextSpec {
    /// Convert to PredicateSpec
    pub fn to_predicate_spec(&self) -> PredicateSpec {
        PredicateSpec::Threshold {
            input: "context_age_minutes".into(),
            threshold: self.freshness_threshold_minutes,
            greater_than: false,
            sharpness: self.sharpness,
        }
    }
}

/// Intent specification for customization
#[derive(Debug, Clone)]
pub struct IntentSpec {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Similarity method
    pub method: SimilarityMethod,
}

impl Default for IntentSpec {
    fn default() -> Self {
        Self {
            embedding_dim: 1536,
            method: SimilarityMethod::Learned,
        }
    }
}

impl IntentSpec {
    /// Convert to PredicateSpec
    pub fn to_predicate_spec(&self) -> PredicateSpec {
        PredicateSpec::LearnedSimilarity {
            left: "intent_embedding".into(),
            right: "context_embedding".into(),
            dim: self.embedding_dim,
            method: self.method,
        }
    }
}

/// Quality specification for customization
#[derive(Debug, Clone)]
pub struct QualitySpec {
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Similarity method for coherence
    pub coherence_method: SimilarityMethod,
    /// Enable topic drift detection
    pub detect_drift: bool,
}

impl Default for QualitySpec {
    fn default() -> Self {
        Self {
            embedding_dim: 1536,
            coherence_method: SimilarityMethod::Bilinear,
            detect_drift: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predicate_names() {
        assert_eq!(InteractionPredicate::MessageSafe.name(), "message_safe");
        assert_eq!(InteractionPredicate::IntentClear.name(), "intent_clear");
        assert_eq!(InteractionPredicate::ContextFresh.name(), "context_fresh");
    }

    #[test]
    fn test_default_specs() {
        let spec = InteractionPredicate::MessageSafe.default_spec();
        match spec {
            PredicateSpec::LearnedProjection { hidden_dim, .. } => {
                assert_eq!(hidden_dim, 256);
            }
            _ => panic!("Expected LearnedProjection"),
        }
    }

    #[test]
    fn test_safety_spec_customization() {
        let spec = SafetySpec {
            embedding_dim: 768,
            hidden_dim: 128,
            attention_heads: Some(2),
            dropout: 0.1,
            residual: false,
        };

        let pred_spec = spec.to_predicate_spec();
        match pred_spec {
            PredicateSpec::LearnedProjection {
                input_dim,
                hidden_dim,
                attention_heads,
                ..
            } => {
                assert_eq!(input_dim, 768);
                assert_eq!(hidden_dim, 128);
                assert_eq!(attention_heads, Some(2));
            }
            _ => panic!("Expected LearnedProjection"),
        }
    }

    #[test]
    fn test_context_spec_defaults() {
        let spec = ContextSpec::default();
        assert_eq!(spec.freshness_threshold_minutes, 30.0);
    }
}

//! Multi-head attention parameters for learned projections
//!
//! Implements scaled dot-product attention with proper initialization
//! for stable training from the first backward pass.

use candle_core::{Device, Tensor, Var};
use crate::{Result, TensorCoreError};

/// Parameters for multi-head attention
#[derive(Debug)]
pub struct AttentionParams {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head (hidden_dim / num_heads)
    pub head_dim: usize,
    /// Query projection: [hidden_dim, hidden_dim]
    pub wq: Var,
    /// Key projection: [hidden_dim, hidden_dim]
    pub wk: Var,
    /// Value projection: [hidden_dim, hidden_dim]
    pub wv: Var,
    /// Output projection: [hidden_dim, hidden_dim]
    pub wo: Var,
}

impl AttentionParams {
    /// Create new attention parameters with scaled dot-product initialization
    /// 
    /// Uses standard transformer initialization for stable gradients:
    /// - Q, K: scaled by 1/sqrt(head_dim) (keeps Var[QK^T] ≈ 1 after scaling)
    /// - V, O: Xavier initialization
    /// 
    /// This matches PyTorch's nn.MultiheadAttention default and provides
    /// stable gradients from the first backward pass.
    pub fn new(hidden_dim: usize, num_heads: usize, device: &Device) -> Result<Self> {
        if !hidden_dim.is_multiple_of(num_heads) {
            return Err(TensorCoreError::Compiler(format!(
                "hidden_dim ({}) must be divisible by num_heads ({})",
                hidden_dim, num_heads
            )));
        }
        
        let head_dim = hidden_dim / num_heads;
        
        // Standard scaled dot-product initialization for Q and K
        let qk_scale = 1.0f32 / (head_dim as f32).sqrt();
        
        // Xavier scale for V and O
        let vo_scale = (2.0f32 / (hidden_dim * 2) as f32).sqrt();
        
        let wq = Var::from_tensor(
            &Tensor::randn(0.0f32, qk_scale, (hidden_dim, hidden_dim), device)
                .map_err(|e| TensorCoreError::Tensor(format!("Failed to create Wq: {}", e)))?,
        ).map_err(|e| TensorCoreError::Tensor(format!("Failed to create Wq var: {}", e)))?;
        
        let wk = Var::from_tensor(
            &Tensor::randn(0.0f32, qk_scale, (hidden_dim, hidden_dim), device)
                .map_err(|e| TensorCoreError::Tensor(format!("Failed to create Wk: {}", e)))?,
        ).map_err(|e| TensorCoreError::Tensor(format!("Failed to create Wk var: {}", e)))?;
        
        let wv = Var::from_tensor(
            &Tensor::randn(0.0f32, vo_scale, (hidden_dim, hidden_dim), device)
                .map_err(|e| TensorCoreError::Tensor(format!("Failed to create Wv: {}", e)))?,
        ).map_err(|e| TensorCoreError::Tensor(format!("Failed to create Wv var: {}", e)))?;
        
        let wo = Var::from_tensor(
            &Tensor::randn(0.0f32, vo_scale, (hidden_dim, hidden_dim), device)
                .map_err(|e| TensorCoreError::Tensor(format!("Failed to create Wo: {}", e)))?,
        ).map_err(|e| TensorCoreError::Tensor(format!("Failed to create Wo var: {}", e)))?;
        
        Ok(Self {
            num_heads,
            head_dim,
            wq,
            wk,
            wv,
            wo,
        })
    }
    
    /// Clamp attention weights to prevent gradient explosion
    /// 
    /// Call this after optimizer.step() to keep weights bounded.
    /// This prevents the Q/K weight explosion that causes NaN after training steps.
    pub fn clamp_weights(&self, max_val: f32) -> Result<()> {
        let clamp_var = |v: &Var| -> Result<()> {
            let clamped = v.as_tensor()
                .clamp(-max_val, max_val)
                .map_err(|e| TensorCoreError::Tensor(format!("clamp failed: {}", e)))?;
            v.set(&clamped)
                .map_err(|e| TensorCoreError::Tensor(format!("set failed: {}", e)))?;
            Ok(())
        };
        
        clamp_var(&self.wq)?;
        clamp_var(&self.wk)?;
        clamp_var(&self.wv)?;
        clamp_var(&self.wo)?;
        Ok(())
    }
    
    /// Get all trainable variables
    pub fn trainable_vars(&self) -> Vec<Var> {
        vec![
            self.wq.clone(),
            self.wk.clone(),
            self.wv.clone(),
            self.wo.clone(),
        ]
    }
    
    /// Total parameter count
    pub fn param_count(&self) -> usize {
        4 * self.wq.as_tensor().elem_count()
    }
}

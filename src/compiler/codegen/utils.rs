//! Codegen utility functions
//!
//! Layer normalization, dropout, multi-head attention forward pass,
//! normalization, and sparse COO support.

use std::collections::HashMap;

use candle_core::{Device, Tensor, Var, D};
use crate::{Result, TensorCoreError};

use super::attention::AttentionParams;

/// Apply activation function
///
/// Delegates to [`super::super::Activation::apply()`] for a single implementation.
pub(crate) fn apply_activation(x: &Tensor, activation: super::super::Activation) -> Result<Tensor> {
    activation.apply(x)
}

/// Apply Layer Normalization
/// 
/// LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
/// 
/// This function auto-clamps gamma/beta weights and normalized values to prevent
/// NaN from weight explosion after gradient updates.
pub(crate) fn apply_layer_norm(x: &Tensor, gamma: &Var, beta: &Var) -> Result<Tensor> {
    let eps = 1e-4;
    
    const LAYERNORM_WEIGHT_MAX: f32 = 10.0;
    
    let gamma_clamped = gamma.as_tensor()
        .clamp(-LAYERNORM_WEIGHT_MAX, LAYERNORM_WEIGHT_MAX)
        .map_err(|e| TensorCoreError::Tensor(format!("layer_norm gamma clamp failed: {}", e)))?;
    let beta_clamped = beta.as_tensor()
        .clamp(-LAYERNORM_WEIGHT_MAX, LAYERNORM_WEIGHT_MAX)
        .map_err(|e| TensorCoreError::Tensor(format!("layer_norm beta clamp failed: {}", e)))?;
    
    let mean = x
        .mean_keepdim(1)
        .map_err(|e| TensorCoreError::Tensor(format!("layer_norm mean failed: {}", e)))?;
    
    let x_centered = x
        .broadcast_sub(&mean)
        .map_err(|e| TensorCoreError::Tensor(format!("layer_norm center failed: {}", e)))?;
    
    let var = x_centered
        .sqr()
        .map_err(|e| TensorCoreError::Tensor(format!("layer_norm sqr failed: {}", e)))?
        .mean_keepdim(1)
        .map_err(|e| TensorCoreError::Tensor(format!("layer_norm var mean failed: {}", e)))?;
    
    let std = var
        .affine(1.0, eps)
        .map_err(|e| TensorCoreError::Tensor(format!("layer_norm eps failed: {}", e)))?
        .sqrt()
        .map_err(|e| TensorCoreError::Tensor(format!("layer_norm sqrt failed: {}", e)))?
        .clamp(eps.sqrt(), f32::MAX)
        .map_err(|e| TensorCoreError::Tensor(format!("layer_norm std clamp failed: {}", e)))?;
    
    let normalized = x_centered
        .broadcast_div(&std)
        .map_err(|e| TensorCoreError::Tensor(format!("layer_norm div failed: {}", e)))?
        .clamp(-10.0f32, 10.0f32)
        .map_err(|e| TensorCoreError::Tensor(format!("layer_norm normalized clamp failed: {}", e)))?;
    
    let scaled = normalized
        .broadcast_mul(&gamma_clamped)
        .map_err(|e| TensorCoreError::Tensor(format!("layer_norm scale failed: {}", e)))?;
    
    scaled
        .broadcast_add(&beta_clamped)
        .map_err(|e| TensorCoreError::Tensor(format!("layer_norm shift failed: {}", e)))
}

/// Apply dropout with inverted scaling
/// 
/// During training: randomly zero elements and scale remaining by 1/(1-p)
/// Clamps outputs to prevent NaN propagation from extreme values.
pub(crate) fn apply_dropout(x: &Tensor, rate: f32) -> Result<Tensor> {
    if rate <= 0.0 || rate >= 1.0 {
        return Ok(x.clone());
    }
    
    let rate = rate.clamp(0.01, 0.99);
    let shape = x.shape();
    let device = x.device();
    
    let rand = Tensor::rand(0.0f32, 1.0f32, shape, device)
        .map_err(|e| TensorCoreError::Tensor(format!("dropout rand failed: {}", e)))?;
    
    let threshold = Tensor::full(rate, shape, device)
        .map_err(|e| TensorCoreError::Tensor(format!("dropout threshold failed: {}", e)))?;
    
    let diff = (&rand - &threshold)
        .map_err(|e| TensorCoreError::Tensor(format!("dropout diff failed: {}", e)))?;
    
    let mask = diff
        .affine(100.0, 0.0)
        .map_err(|e| TensorCoreError::Tensor(format!("dropout sharpen failed: {}", e)))?
        .clamp(-50.0f32, 50.0f32)
        .map_err(|e| TensorCoreError::Tensor(format!("dropout sharpen clamp failed: {}", e)))?;
    let mask = crate::primitives::sigmoid(&mask)?;
    
    let scale = 1.0 / (1.0 - rate);
    
    let masked = (x * &mask)
        .map_err(|e| TensorCoreError::Tensor(format!("dropout mul failed: {}", e)))?;
    
    masked
        .affine(scale as f64, 0.0)
        .map_err(|e| TensorCoreError::Tensor(format!("dropout scale failed: {}", e)))?
        .clamp(-100.0f32, 100.0f32)
        .map_err(|e| TensorCoreError::Tensor(format!("dropout output clamp failed: {}", e)))
}

/// Metal-compatible softmax over last dimension
fn softmax_last_dim_metal_compat(x: &Tensor) -> Result<Tensor> {
    let clamped = x
        .clamp(-100.0f32, 100.0f32)
        .map_err(|e| TensorCoreError::Tensor(format!("softmax clamp failed: {}", e)))?;
    
    let max_vals = clamped
        .max_keepdim(D::Minus1)
        .map_err(|e| TensorCoreError::Tensor(format!("softmax max failed: {}", e)))?;
    
    let shifted = clamped
        .broadcast_sub(&max_vals)
        .map_err(|e| TensorCoreError::Tensor(format!("softmax shift failed: {}", e)))?;
    
    let exp_vals = shifted
        .exp()
        .map_err(|e| TensorCoreError::Tensor(format!("softmax exp failed: {}", e)))?;
    
    let sum_vals = exp_vals
        .sum_keepdim(D::Minus1)
        .map_err(|e| TensorCoreError::Tensor(format!("softmax sum failed: {}", e)))?;
    
    let sum_vals = sum_vals
        .affine(1.0, 1e-10)
        .map_err(|e| TensorCoreError::Tensor(format!("softmax epsilon failed: {}", e)))?;
    
    exp_vals
        .broadcast_div(&sum_vals)
        .map_err(|e| TensorCoreError::Tensor(format!("softmax div failed: {}", e)))
}

/// Apply multi-head self-attention
/// 
/// MultiHeadAttention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
/// 
/// Auto-clamps attention weights to prevent NaN from weight explosion.
pub(crate) fn apply_multi_head_attention(
    x: &Tensor,
    params: &AttentionParams,
    dropout_rate: Option<f32>,
) -> Result<Tensor> {
    let (batch_size, _seq_len) = {
        let shape = x.dims();
        if shape.len() == 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1, shape[0])
        } else {
            return Err(TensorCoreError::Tensor(format!(
                "Unexpected input shape for attention: {:?}", shape
            )));
        }
    };
    
    let hidden_dim = params.num_heads * params.head_dim;
    
    const ATTENTION_WEIGHT_MAX: f32 = 10.0;
    
    let wq_clamped = params.wq.as_tensor()
        .clamp(-ATTENTION_WEIGHT_MAX, ATTENTION_WEIGHT_MAX)
        .map_err(|e| TensorCoreError::Tensor(format!("attention Wq clamp failed: {}", e)))?;
    let wk_clamped = params.wk.as_tensor()
        .clamp(-ATTENTION_WEIGHT_MAX, ATTENTION_WEIGHT_MAX)
        .map_err(|e| TensorCoreError::Tensor(format!("attention Wk clamp failed: {}", e)))?;
    let wv_clamped = params.wv.as_tensor()
        .clamp(-ATTENTION_WEIGHT_MAX, ATTENTION_WEIGHT_MAX)
        .map_err(|e| TensorCoreError::Tensor(format!("attention Wv clamp failed: {}", e)))?;
    let wo_clamped = params.wo.as_tensor()
        .clamp(-ATTENTION_WEIGHT_MAX, ATTENTION_WEIGHT_MAX)
        .map_err(|e| TensorCoreError::Tensor(format!("attention Wo clamp failed: {}", e)))?;
    
    let q = x
        .matmul(&wq_clamped)
        .map_err(|e| TensorCoreError::Tensor(format!("attention Q projection failed: {}", e)))?;
    let k = x
        .matmul(&wk_clamped)
        .map_err(|e| TensorCoreError::Tensor(format!("attention K projection failed: {}", e)))?;
    let v = x
        .matmul(&wv_clamped)
        .map_err(|e| TensorCoreError::Tensor(format!("attention V projection failed: {}", e)))?;
    
    let q = q
        .reshape((batch_size, params.num_heads, params.head_dim))
        .map_err(|e| TensorCoreError::Tensor(format!("attention Q reshape failed: {}", e)))?;
    let k = k
        .reshape((batch_size, params.num_heads, params.head_dim))
        .map_err(|e| TensorCoreError::Tensor(format!("attention K reshape failed: {}", e)))?;
    let v = v
        .reshape((batch_size, params.num_heads, params.head_dim))
        .map_err(|e| TensorCoreError::Tensor(format!("attention V reshape failed: {}", e)))?;
    
    let scale = (params.head_dim as f32).sqrt();
    
    let k_t = k
        .transpose(1, 2)
        .map_err(|e| TensorCoreError::Tensor(format!("attention K transpose failed: {}", e)))?;
    
    let mut scores = q
        .matmul(&k_t)
        .map_err(|e| TensorCoreError::Tensor(format!("attention scores matmul failed: {}", e)))?;
    
    scores = scores
        .affine(1.0 / scale as f64, 0.0)
        .map_err(|e| TensorCoreError::Tensor(format!("attention scale failed: {}", e)))?;
    
    scores = scores
        .clamp(-50.0f32, 50.0f32)
        .map_err(|e| TensorCoreError::Tensor(format!("attention score clamp failed: {}", e)))?;
    
    let attn_weights = softmax_last_dim_metal_compat(&scores)?;
    
    let attn_weights = if let Some(rate) = dropout_rate {
        if rate > 0.0 {
            apply_dropout(&attn_weights, rate)?
        } else {
            attn_weights
        }
    } else {
        attn_weights
    };
    
    let attn_output = attn_weights
        .matmul(&v)
        .map_err(|e| TensorCoreError::Tensor(format!("attention output matmul failed: {}", e)))?;
    
    let attn_output = attn_output
        .reshape((batch_size, hidden_dim))
        .map_err(|e| TensorCoreError::Tensor(format!("attention output reshape failed: {}", e)))?;
    
    attn_output
        .matmul(&wo_clamped)
        .map_err(|e| TensorCoreError::Tensor(format!("attention output projection failed: {}", e)))
}

/// L2 normalize a tensor along the last dimension
pub(crate) fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let norm = x
        .sqr()
        .map_err(|e| TensorCoreError::Tensor(format!("sqr failed: {}", e)))?
        .sum_keepdim(1)
        .map_err(|e| TensorCoreError::Tensor(format!("sum failed: {}", e)))?
        .sqrt()
        .map_err(|e| TensorCoreError::Tensor(format!("sqrt failed: {}", e)))?;
    
    let eps = 1e-8;
    let norm = norm
        .affine(1.0, eps)
        .map_err(|e| TensorCoreError::Tensor(format!("affine failed: {}", e)))?;
    
    x.broadcast_div(&norm)
        .map_err(|e| TensorCoreError::Tensor(format!("div failed: {}", e)))
}

/// Normalize adjacency matrix based on aggregation method
pub(crate) fn normalize_adjacency(
    adj: &Tensor,
    aggregation: super::super::Aggregation,
    _device: &Device,
) -> Result<Tensor> {
    match aggregation {
        super::super::Aggregation::Mean | super::super::Aggregation::Attention => {
            let degree = adj
                .sum_keepdim(1)
                .map_err(|e| TensorCoreError::Tensor(format!("sum failed: {}", e)))?;
            let degree = degree
                .affine(1.0, 1e-8)
                .map_err(|e| TensorCoreError::Tensor(format!("affine failed: {}", e)))?;
            adj.broadcast_div(&degree)
                .map_err(|e| TensorCoreError::Tensor(format!("div failed: {}", e)))
        }
        super::super::Aggregation::Sum | super::super::Aggregation::Max => {
            Ok(adj.clone())
        }
    }
}

// =============================================================================
// SPARSE COO SUPPORT
// =============================================================================

/// Convert sparse COO format adjacency to dense matrix
///
/// Sparse COO format expects inputs as a HashMap with keys:
/// - `{name}_row_indices`: [nnz] i64 tensor of row indices
/// - `{name}_col_indices`: [nnz] i64 tensor of column indices  
/// - `{name}_values`: [nnz] f32 tensor of edge values (or None for all 1s)
/// - `{name}_shape`: [2] i64 tensor with (num_rows, num_cols)
pub fn sparse_coo_to_dense(
    row_indices: &Tensor,
    col_indices: &Tensor,
    values: Option<&Tensor>,
    shape: (usize, usize),
    device: &Device,
) -> Result<Tensor> {
    let (num_rows, num_cols) = shape;
    
    let rows = row_indices
        .flatten_all()
        .map_err(|e| TensorCoreError::Tensor(format!("flatten rows failed: {}", e)))?
        .to_vec1::<i64>()
        .map_err(|e| TensorCoreError::Tensor(format!("to_vec rows failed: {}", e)))?;
    
    let cols = col_indices
        .flatten_all()
        .map_err(|e| TensorCoreError::Tensor(format!("flatten cols failed: {}", e)))?
        .to_vec1::<i64>()
        .map_err(|e| TensorCoreError::Tensor(format!("to_vec cols failed: {}", e)))?;
    
    let vals: Vec<f32> = if let Some(v) = values {
        v.flatten_all()
            .map_err(|e| TensorCoreError::Tensor(format!("flatten values failed: {}", e)))?
            .to_vec1::<f32>()
            .map_err(|e| TensorCoreError::Tensor(format!("to_vec values failed: {}", e)))?
    } else {
        vec![1.0f32; rows.len()]
    };
    
    if rows.len() != cols.len() || rows.len() != vals.len() {
        return Err(TensorCoreError::Tensor(format!(
            "Sparse COO dimension mismatch: rows={}, cols={}, values={}",
            rows.len(), cols.len(), vals.len()
        )));
    }
    
    let mut dense = vec![0.0f32; num_rows * num_cols];
    for ((&r, &c), &v) in rows.iter().zip(cols.iter()).zip(vals.iter()) {
        let r = r as usize;
        let c = c as usize;
        if r < num_rows && c < num_cols {
            dense[r * num_cols + c] = v;
        }
    }
    
    Tensor::from_vec(dense, (num_rows, num_cols), device)
        .map_err(|e| TensorCoreError::Tensor(format!("create dense tensor failed: {}", e)))
}

/// Check if sparse COO inputs are present for a given adjacency name
pub fn detect_sparse_coo<'a>(
    inputs: &'a HashMap<String, Tensor>,
    adjacency_name: &str,
) -> Option<(&'a Tensor, &'a Tensor, Option<&'a Tensor>, (usize, usize))> {
    let row_key = format!("{}_row_indices", adjacency_name);
    let col_key = format!("{}_col_indices", adjacency_name);
    let shape_key = format!("{}_shape", adjacency_name);
    let values_key = format!("{}_values", adjacency_name);
    
    let row_indices = inputs.get(&row_key)?;
    let col_indices = inputs.get(&col_key)?;
    let shape_tensor = inputs.get(&shape_key)?;
    let values = inputs.get(&values_key);
    
    let shape_vec = shape_tensor
        .flatten_all()
        .ok()?
        .to_vec1::<i64>()
        .ok()?;
    
    if shape_vec.len() != 2 {
        return None;
    }
    
    let shape = (shape_vec[0] as usize, shape_vec[1] as usize);
    
    Some((row_indices, col_indices, values, shape))
}

/// Get adjacency matrix, handling both dense and sparse COO formats
pub fn get_adjacency_matrix(
    inputs: &HashMap<String, Tensor>,
    adjacency_name: &str,
    device: &Device,
) -> Result<Tensor> {
    if let Some((row_indices, col_indices, values, shape)) = detect_sparse_coo(inputs, adjacency_name) {
        return sparse_coo_to_dense(row_indices, col_indices, values, shape, device);
    }
    
    inputs.get(adjacency_name).cloned().ok_or_else(|| {
        TensorCoreError::Compiler(format!(
            "Missing adjacency matrix '{}'. Provide either:\n  \
             - Dense: inputs[\"{}\"] = [n, n] tensor\n  \
             - Sparse COO: inputs[\"{}_row_indices\"], inputs[\"{}_col_indices\"], inputs[\"{}_shape\"]",
            adjacency_name, adjacency_name, adjacency_name, adjacency_name, adjacency_name
        ))
    })
}

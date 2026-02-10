//! Core Tensor Operations
//!
//! Fundamental tensor operations used across all Tensor Logic components.
//!
//! ## Thread Safety
//!
//! Metal GPU command buffers are **not thread-safe** for concurrent encoding.
//! When sharing tensors across threads (e.g., with rayon), you have several options:
//!
//! ### Option 1: Force CPU mode (recommended for parallel read-only workloads)
//! ```bash
//! export AN_TENSOR_NO_GPU=1
//! ```
//!
//! ### Option 2: Use thread-local devices
//! ```rust,ignore
//! use an_tensor_compiler::primitives::thread_local_device;
//!
//! rayon::scope(|s| {
//!     for _ in 0..16 {
//!         s.spawn(|_| {
//!             let device = thread_local_device(); // Each thread gets its own device
//!             // ... tensor operations ...
//!         });
//!     }
//! });
//! ```
//!
//! ### Option 3: Synchronize GPU access
//! ```rust,ignore
//! use an_tensor_compiler::primitives::with_gpu_sync;
//!
//! with_gpu_sync(|| {
//!     // GPU operations are serialized here
//! });
//! ```

use crate::{Result, TensorCoreError};
use candle_core::{Device, Tensor};
use parking_lot::Mutex;
use std::cell::RefCell;
use std::sync::OnceLock;
use tracing::info;

// ============================================================================
// GPU Synchronization
// ============================================================================

/// Global mutex for serializing GPU command buffer access when needed.
///
/// Use `with_gpu_sync()` to safely execute GPU operations from multiple threads.
static GPU_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

fn gpu_mutex() -> &'static Mutex<()> {
    GPU_MUTEX.get_or_init(|| Mutex::new(()))
}

/// Execute a closure with synchronized GPU access.
///
/// This serializes GPU command buffer encoding to avoid Metal thread safety issues.
/// Use this when you need GPU acceleration from multiple threads but can tolerate
/// serialized execution.
///
/// # Example
/// ```rust,ignore
/// use an_tensor_compiler::primitives::with_gpu_sync;
///
/// let results: Vec<_> = (0..16)
///     .into_par_iter()
///     .map(|i| {
///         with_gpu_sync(|| {
///             // Safe GPU operations here
///             tensor.matmul(&other)?
///         })
///     })
///     .collect();
/// ```
pub fn with_gpu_sync<T, F: FnOnce() -> T>(f: F) -> T {
    let _guard = gpu_mutex().lock();
    f()
}

// ============================================================================
// Environment-controlled Device Selection
// ============================================================================

/// Check if GPU is disabled via environment variable.
///
/// Set `AN_TENSOR_NO_GPU=1` to force CPU-only mode. This is recommended for
/// parallel workloads where multiple threads access shared tensor data.
pub fn gpu_disabled() -> bool {
    std::env::var("AN_TENSOR_NO_GPU")
        .map(|v| !v.is_empty() && v != "0" && v.to_lowercase() != "false")
        .unwrap_or(false)
}

/// Get the best available device for tensor operations
///
/// Priority:
/// 1. Check `AN_TENSOR_NO_GPU` env var (forces CPU if set)
/// 2. Metal (Apple Silicon - M1/M2/M3)
/// 3. CUDA (NVIDIA GPUs)
/// 4. CPU (fallback)
///
/// # Thread Safety Warning
///
/// Metal devices are **not thread-safe** for concurrent command buffer encoding.
/// If you're sharing tensors across threads, either:
/// - Set `AN_TENSOR_NO_GPU=1` to force CPU mode
/// - Use `thread_local_device()` for per-thread devices
/// - Use `with_gpu_sync()` to serialize GPU access
///
/// # Example
/// ```ignore
/// let device = best_device();
/// let tensor = Tensor::zeros((10, 10), DType::F32, &device)?;
/// ```
pub fn best_device() -> Device {
    // Check for forced CPU mode
    if gpu_disabled() {
        info!("💻 Using CPU device (AN_TENSOR_NO_GPU set)");
        return Device::Cpu;
    }

    // Try Metal first (for M3/M3 Ultra)
    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            info!("🍎 Using Metal device (Apple Silicon)");
            return device;
        }
    }

    // Try CUDA
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            info!("🟢 Using CUDA device (NVIDIA GPU)");
            return device;
        }
    }

    // Fallback to CPU
    info!("💻 Using CPU device");
    Device::Cpu
}

/// Force CPU device, ignoring GPU availability.
///
/// Use this when you need guaranteed thread-safe tensor operations,
/// such as in parallel simulations or when sharing manifolds across threads.
///
/// # Example
/// ```rust,ignore
/// // Load manifold once with CPU device for thread-safe parallel access
/// let device = cpu_device();
/// let manifold = TensorMarketManifold::load_with_device(path, &device)?;
/// let manifold = Arc::new(manifold);
///
/// // Safe parallel access
/// rayon::scope(|s| {
///     for _ in 0..16 {
///         let m = Arc::clone(&manifold);
///         s.spawn(move |_| {
///             let data = m.get_close_by_idx(sym_idx, ts_idx); // No GPU conflicts
///         });
///     }
/// });
/// ```
pub fn cpu_device() -> Device {
    Device::Cpu
}

// ============================================================================
// Thread-Local Device for Parallel Workloads
// ============================================================================

thread_local! {
    /// Thread-local device instance.
    ///
    /// Each thread gets its own device to avoid Metal command buffer conflicts.
    /// For CPU, this is always safe. For GPU, each thread gets its own Metal device
    /// with separate command buffers (if available).
    static THREAD_LOCAL_DEVICE: RefCell<Option<Device>> = const { RefCell::new(None) };
}

/// Get a thread-local device for the current thread.
///
/// This is the recommended approach for parallel workloads where each thread
/// needs its own tensor operations. Each thread gets its own device instance,
/// avoiding Metal command buffer conflicts.
///
/// **Note**: When `AN_TENSOR_NO_GPU=1` is set, all threads use CPU.
///
/// # Example
/// ```rust,ignore
/// use an_tensor_compiler::primitives::thread_local_device;
/// use rayon::prelude::*;
///
/// let results: Vec<_> = (0..16)
///     .into_par_iter()
///     .map(|i| {
///         let device = thread_local_device();
///         let tensor = Tensor::randn(0f32, 1.0, (100, 100), &device)?;
///         tensor.sum_all()?.to_scalar::<f32>()
///     })
///     .collect();
/// ```
pub fn thread_local_device() -> Device {
    THREAD_LOCAL_DEVICE.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            // Initialize device for this thread
            let device = if gpu_disabled() {
                Device::Cpu
            } else {
                // Try GPU, but with thread-specific initialization
                #[cfg(feature = "metal")]
                {
                    // Note: Metal devices on the same GPU share command queue state.
                    // For truly parallel GPU workloads, CPU is safer.
                    // But we try Metal in case the user wants it.
                    if let Ok(device) = Device::new_metal(0) {
                        // Log only on first thread to avoid spam
                        static LOGGED: std::sync::atomic::AtomicBool =
                            std::sync::atomic::AtomicBool::new(false);
                        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                            info!("🍎 Thread-local Metal device (Apple Silicon)");
                        }
                        device
                    } else {
                        Device::Cpu
                    }
                }
                #[cfg(all(feature = "cuda", not(feature = "metal")))]
                {
                    if let Ok(device) = Device::new_cuda(0) {
                        device
                    } else {
                        Device::Cpu
                    }
                }
                #[cfg(not(any(feature = "metal", feature = "cuda")))]
                {
                    Device::Cpu
                }
            };
            *opt = Some(device);
        }
        opt.as_ref().unwrap().clone()
    })
}

// ============================================================================
// Device Availability Checks
// ============================================================================

/// Check if Metal is available (respects AN_TENSOR_NO_GPU)
#[cfg(feature = "metal")]
pub fn metal_available() -> bool {
    !gpu_disabled() && Device::new_metal(0).is_ok()
}

/// Check if Metal is available (always false when `metal` feature is not enabled)
#[cfg(not(feature = "metal"))]
pub fn metal_available() -> bool {
    false
}

/// Check if CUDA is available (respects AN_TENSOR_NO_GPU)
#[cfg(feature = "cuda")]
pub fn cuda_available() -> bool {
    !gpu_disabled() && Device::new_cuda(0).is_ok()
}

/// Check if CUDA is available (always false when `cuda` feature is not enabled)
#[cfg(not(feature = "cuda"))]
pub fn cuda_available() -> bool {
    false
}

/// Check if any GPU is available and enabled
pub fn gpu_available() -> bool {
    !gpu_disabled() && (metal_available() || cuda_available())
}

/// Compute cosine similarity between two 1D tensors
///
/// Returns a scalar in [-1, 1] representing the angle between vectors.
pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    let dot = a
        .mul(b)?
        .sum_all()?
        .to_scalar::<f32>()
        .map_err(|e| TensorCoreError::Tensor(format!("Dot product failed: {}", e)))?;

    let norm_a = a
        .sqr()?
        .sum_all()?
        .sqrt()?
        .to_scalar::<f32>()
        .map_err(|e| TensorCoreError::Tensor(format!("Norm a failed: {}", e)))?;

    let norm_b = b
        .sqr()?
        .sum_all()?
        .sqrt()?
        .to_scalar::<f32>()
        .map_err(|e| TensorCoreError::Tensor(format!("Norm b failed: {}", e)))?;

    if norm_a > 1e-8 && norm_b > 1e-8 {
        Ok(dot / (norm_a * norm_b))
    } else {
        Ok(0.0)
    }
}

/// Binary cross-entropy loss
///
/// BCE = -[y * log(p) + (1-y) * log(1-p)]
pub fn binary_cross_entropy(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    let eps = 1e-7f32;
    let pred_clamped = pred
        .clamp(eps, 1.0 - eps)
        .map_err(|e| TensorCoreError::Tensor(format!("Clamp failed: {}", e)))?;
    let log_p = pred_clamped
        .log()
        .map_err(|e| TensorCoreError::Tensor(format!("Log failed: {}", e)))?;

    // Use ones_like to preserve dtype (avoid F64 promotion from 1.0 literal)
    let ones = Tensor::ones_like(&pred_clamped)
        .map_err(|e| TensorCoreError::Tensor(format!("ones_like failed: {}", e)))?;
    let one_minus_pred = (&ones - &pred_clamped)
        .map_err(|e| TensorCoreError::Tensor(format!("1-pred failed: {}", e)))?;
    let log_1_p = one_minus_pred
        .log()
        .map_err(|e| TensorCoreError::Tensor(format!("Log 1-p failed: {}", e)))?;

    let term1 = target
        .mul(&log_p)
        .map_err(|e| TensorCoreError::Tensor(format!("BCE term1 failed: {}", e)))?;

    let ones_target = Tensor::ones_like(target)
        .map_err(|e| TensorCoreError::Tensor(format!("ones_like target failed: {}", e)))?;
    let one_minus_target = (&ones_target - target)
        .map_err(|e| TensorCoreError::Tensor(format!("1-target failed: {}", e)))?;
    let term2 = one_minus_target
        .mul(&log_1_p)
        .map_err(|e| TensorCoreError::Tensor(format!("BCE term2 failed: {}", e)))?;

    let loss = (term1 + term2)?;
    let neg_loss = loss
        .neg()
        .map_err(|e| TensorCoreError::Tensor(format!("Neg failed: {}", e)))?;
    neg_loss
        .mean_all()
        .map_err(|e| TensorCoreError::Tensor(format!("BCE mean failed: {}", e)))
}

/// Mean squared error loss
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    let diff =
        (pred - target).map_err(|e| TensorCoreError::Tensor(format!("MSE diff failed: {}", e)))?;
    diff.sqr()?
        .mean_all()
        .map_err(|e| TensorCoreError::Tensor(format!("MSE failed: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_best_device() {
        let device = best_device();
        // Should return a valid device
        assert!(matches!(
            device,
            Device::Cpu | Device::Metal(_) | Device::Cuda(_)
        ));
    }

    #[test]
    fn test_cpu_device() {
        let device = cpu_device();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_thread_local_device() {
        // Thread-local device should work on main thread
        let device = thread_local_device();
        assert!(matches!(
            device,
            Device::Cpu | Device::Metal(_) | Device::Cuda(_)
        ));

        // Same thread should get same device type
        let device2 = thread_local_device();
        assert!(matches!(
            device2,
            Device::Cpu | Device::Metal(_) | Device::Cuda(_)
        ));
    }

    #[test]
    fn test_gpu_sync() {
        // Test that with_gpu_sync works for basic operations
        let result = with_gpu_sync(|| {
            let device = cpu_device();
            let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &device).unwrap();
            tensor.sum_all().unwrap().to_scalar::<f32>().unwrap()
        });
        assert!((result - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_gpu_disabled_flag() {
        // Note: This test may behave differently based on env var
        // Just verify the function doesn't panic
        let _disabled = gpu_disabled();
    }

    #[test]
    fn test_gpu_available() {
        // Just verify it doesn't panic
        let _available = gpu_available();
    }

    /// Test parallel tensor access with CPU device (thread-safe)
    ///
    /// This simulates the pattern from the bug report where multiple threads
    /// read from shared tensor data.
    #[test]
    fn test_parallel_tensor_access_cpu() {
        use std::thread;

        // Create shared tensor data on CPU (thread-safe)
        let device = cpu_device();
        let tensor_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let tensor = Arc::new(Tensor::from_vec(tensor_data, (10, 100), &device).unwrap());

        // Spawn parallel threads accessing the shared tensor
        let handles: Vec<_> = (0..16)
            .map(|i| {
                let t = Arc::clone(&tensor);
                thread::spawn(move || {
                    // Each thread reads from shared tensor
                    let row = t.narrow(0, i % 10, 1).unwrap();

                    row.sum_all().unwrap().to_scalar::<f32>().unwrap()
                })
            })
            .collect();

        // Collect results
        let results: Vec<f32> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        assert_eq!(results.len(), 16);
        // All results should be valid numbers
        for r in &results {
            assert!(r.is_finite());
        }
    }

    /// Test parallel operations with thread-local devices
    #[test]
    fn test_parallel_thread_local_devices() {
        use std::thread;

        let handles: Vec<_> = (0..8)
            .map(|i| {
                thread::spawn(move || {
                    let device = thread_local_device();
                    // Each thread creates its own tensor on its device
                    let tensor = Tensor::from_vec(vec![i as f32; 100], 100, &device).unwrap();
                    tensor.sum_all().unwrap().to_scalar::<f32>().unwrap()
                })
            })
            .collect();

        let results: Vec<f32> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        assert_eq!(results.len(), 8);
        for (i, r) in results.iter().enumerate() {
            let expected = (i as f32) * 100.0;
            assert!(
                (r - expected).abs() < 0.001,
                "Expected {} but got {}",
                expected,
                r
            );
        }
    }

    /// Test synchronized GPU access pattern
    #[test]
    fn test_parallel_gpu_sync() {
        use std::thread;

        let handles: Vec<_> = (0..4)
            .map(|i| {
                thread::spawn(move || {
                    with_gpu_sync(|| {
                        // Inside gpu_sync, safe to use any device
                        let device = cpu_device(); // Using CPU for test reliability
                        let tensor = Tensor::from_vec(vec![i as f32; 10], 10, &device).unwrap();
                        tensor.sum_all().unwrap().to_scalar::<f32>().unwrap()
                    })
                })
            })
            .collect();

        let results: Vec<f32> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        assert_eq!(results.len(), 4);
        for (i, r) in results.iter().enumerate() {
            let expected = (i as f32) * 10.0;
            assert!((r - expected).abs() < 0.001);
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let device = Device::Cpu;

        // Same vector = similarity 1.0
        let a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0], 3, &device).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, 0.0, 0.0], 3, &device).unwrap();
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 0.001);

        // Orthogonal vectors = similarity 0.0
        let c = Tensor::from_vec(vec![0.0f32, 1.0, 0.0], 3, &device).unwrap();
        let sim_ortho = cosine_similarity(&a, &c).unwrap();
        assert!(sim_ortho.abs() < 0.001);

        // Opposite vectors = similarity -1.0
        let d = Tensor::from_vec(vec![-1.0f32, 0.0, 0.0], 3, &device).unwrap();
        let sim_opp = cosine_similarity(&a, &d).unwrap();
        assert!((sim_opp + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_mse_loss() {
        let device = Device::Cpu;
        let pred = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &device).unwrap();
        let target = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &device).unwrap();

        let loss = mse_loss(&pred, &target).unwrap();
        let loss_val = loss.to_scalar::<f32>().unwrap();

        // Perfect prediction = 0 loss
        assert!(loss_val.abs() < 0.001);
    }

    #[test]
    fn test_bce_loss() {
        let device = Device::Cpu;
        let pred = Tensor::from_vec(vec![0.9f32], 1, &device).unwrap();
        let target = Tensor::from_vec(vec![1.0f32], 1, &device).unwrap();

        let loss = binary_cross_entropy(&pred, &target).unwrap();
        let loss_val = loss.to_scalar::<f32>().unwrap();

        // High confidence correct prediction = low loss
        assert!(loss_val < 0.2);
    }
}

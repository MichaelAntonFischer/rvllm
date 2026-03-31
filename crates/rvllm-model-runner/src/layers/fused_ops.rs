//! Fused CUDA kernel dispatch for performance-critical operation pairs.
//!
//! These fused operations eliminate intermediate GPU memory round-trips:
//!
//! - `fused_residual_rmsnorm`: residual add + RMSNorm in one kernel launch
//! - `embedding_gather`: GPU-side embedding lookup (no DtoH/HtoD)
//! - `add_bias`: in-place bias addition on GPU
//! - `add_tensors`: element-wise tensor addition on GPU
//!
//! All gated behind `#[cfg(feature = "cuda")]`.

#[cfg(feature = "cuda")]
mod inner {
    use std::sync::Arc;

    use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
    use tracing::trace;

    use rvllm_core::prelude::{LLMError, Result};

    /// Fused residual addition + RMS normalization.
    ///
    /// Computes:
    ///   residual[i] = input[i] + add[i]
    ///   output[i] = rmsnorm(residual, weight, eps)[i]
    ///
    /// in a single kernel launch, saving one full memory traversal of the
    /// hidden state tensor (typically 4096+ floats per token).
    ///
    /// Both `residual` (the writeback) and `output` (the normalized result)
    /// are returned so the caller can use the residual for the next add.
    pub fn fused_residual_rmsnorm(
        stream: &Arc<CudaStream>,
        kernel: &CudaFunction,
        input: &CudaSlice<f32>,
        add: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        eps: f32,
        num_tokens: usize,
        hidden_size: usize,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
        let n = num_tokens * hidden_size;

        // Safety: kernel writes all n elements to both output and residual
        let output = unsafe { stream.alloc::<f32>(n) }
            .map_err(|e| LLMError::GpuError(format!("fused_res_rmsnorm output alloc: {e}")))?;
        let residual = unsafe { stream.alloc::<f32>(n) }
            .map_err(|e| LLMError::GpuError(format!("fused_res_rmsnorm residual alloc: {e}")))?;

        let block_dim = hidden_size.min(1024) as u32;
        let shared_mem = block_dim * std::mem::size_of::<f32>() as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: shared_mem,
        };

        // SAFETY: All slices are valid device memory. Grid covers all tokens,
        // block covers hidden_size with stride loop. Shared memory is sized
        // for the reduction.
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(&output)
                .arg(&residual)
                .arg(input)
                .arg(add)
                .arg(weight)
                .arg(&eps)
                .arg(&(hidden_size as i32))
                .launch(cfg)
                .map_err(|e| LLMError::GpuError(format!("fused_residual_rmsnorm launch: {e}")))?;
        }

        trace!(num_tokens, hidden_size, "fused_residual_rmsnorm launched");
        Ok((output, residual))
    }

    /// GPU-side embedding gather. Avoids the DtoH (embed table) + HtoD (gathered rows)
    /// round-trip that dominates embedding lookup latency.
    ///
    /// The embed table stays on GPU; token IDs are uploaded once as i32.
    pub fn embedding_gather(
        stream: &Arc<CudaStream>,
        kernel: &CudaFunction,
        embed_table: &CudaSlice<f32>,
        token_ids: &[u32],
        hidden_size: usize,
        vocab_size: usize,
    ) -> Result<CudaSlice<f32>> {
        let num_tokens = token_ids.len();
        let output_len = num_tokens * hidden_size;

        // Safety: embedding gather kernel writes all output_len elements
        let output = unsafe { stream.alloc::<f32>(output_len) }
            .map_err(|e| LLMError::GpuError(format!("embed_gather output alloc: {e}")))?;

        // Upload token IDs (small transfer, typically < 4KB even for batch=1024)
        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        let token_ids_gpu = stream
            .clone_htod(&token_ids_i32)
            .map_err(|e| LLMError::GpuError(format!("embed_gather token_ids upload: {e}")))?;

        let block_dim = hidden_size.min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: embed_table has vocab_size * hidden_size elements.
        // output has num_tokens * hidden_size elements. token_ids_gpu has
        // num_tokens elements. Grid covers all tokens.
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(&output)
                .arg(embed_table)
                .arg(&token_ids_gpu)
                .arg(&(hidden_size as i32))
                .arg(&(vocab_size as i32))
                .launch(cfg)
                .map_err(|e| LLMError::GpuError(format!("embedding_gather launch: {e}")))?;
        }

        trace!(num_tokens, hidden_size, "embedding_gather launched");
        Ok(output)
    }

    /// In-place bias addition on GPU.
    ///
    /// tensor[token, d] += bias[d] for all tokens.
    ///
    /// Replaces the DtoH + CPU add + HtoD pattern in GpuWorker::add_bias_gpu.
    pub fn add_bias_gpu(
        stream: &Arc<CudaStream>,
        kernel: &CudaFunction,
        tensor: &mut CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        num_tokens: usize,
        dim: usize,
    ) -> Result<()> {
        let block_dim = dim.min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: tensor has num_tokens * dim elements, bias has dim elements.
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(tensor)
                .arg(bias)
                .arg(&(dim as i32))
                .launch(cfg)
                .map_err(|e| LLMError::GpuError(format!("add_bias launch: {e}")))?;
        }

        trace!(num_tokens, dim, "add_bias launched");
        Ok(())
    }

    /// Element-wise tensor addition on GPU: output = a + b.
    ///
    /// Replaces the DtoH + CPU add + HtoD pattern in GpuWorker::add_tensors_gpu.
    pub fn add_tensors_gpu(
        stream: &Arc<CudaStream>,
        kernel: &CudaFunction,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
    ) -> Result<CudaSlice<f32>> {
        // Safety: add kernel writes all n elements
        let output = unsafe { stream.alloc::<f32>(n) }
            .map_err(|e| LLMError::GpuError(format!("add_tensors alloc: {e}")))?;

        let threads = 256u32;
        let blocks = (n as u32 + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: a, b, output all have n elements.
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(&output)
                .arg(a)
                .arg(b)
                .arg(&(n as i32))
                .launch(cfg)
                .map_err(|e| LLMError::GpuError(format!("add_kernel launch: {e}")))?;
        }

        trace!(n, "add_tensors launched");
        Ok(output)
    }

    /// In-place tensor addition on GPU: a += b.
    pub fn add_inplace_gpu(
        stream: &Arc<CudaStream>,
        kernel: &CudaFunction,
        a: &mut CudaSlice<f32>,
        b: &CudaSlice<f32>,
        n: usize,
    ) -> Result<()> {
        let threads = 256u32;
        let blocks = (n as u32 + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: a and b both have n elements. Per-element operation, no aliasing.
        unsafe {
            stream
                .launch_builder(kernel)
                .arg(a)
                .arg(b)
                .arg(&(n as i32))
                .launch(cfg)
                .map_err(|e| LLMError::GpuError(format!("add_inplace launch: {e}")))?;
        }

        trace!(n, "add_inplace launched");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub use inner::*;

#[cfg(test)]
mod tests {
    #[test]
    fn module_compiles() {
        // Compile-only check under mock-gpu.
    }
}

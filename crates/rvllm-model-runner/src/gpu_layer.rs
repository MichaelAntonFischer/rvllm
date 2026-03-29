//! GPU Transformer Layer -- one complete transformer block on CUDA (f16 only).
//!
//! All code is gated behind `#[cfg(feature = "cuda")]`.

#[cfg(feature = "cuda")]
mod inner {
    use std::sync::Arc;

    use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtrMut, DeviceSlice, LaunchConfig, PushKernelArg};
    use half::f16;
    use tracing::{info, trace};

    use rvllm_core::error::{LLMError, Result};
    use rvllm_gpu::cublas::CublasHandle;
    use rvllm_gpu::kernel_loader::KernelLoader;

    /// Configuration for a single transformer layer.
    #[derive(Debug, Clone)]
    pub struct GpuLayerConfig {
        pub hidden_size: usize,
        pub num_heads: usize,
        pub num_kv_heads: usize,
        pub head_dim: usize,
        pub intermediate_size: usize,
        pub rms_norm_eps: f32,
        pub layer_idx: usize,
    }

    /// Weight references for a single transformer layer (all f16).
    ///
    /// All slices live on GPU and are owned by the GpuModelWeights container.
    /// This struct borrows them for the duration of a forward pass.
    pub struct GpuLayerWeights<'a> {
        // Pre-attention norm
        pub input_layernorm: &'a CudaSlice<f16>,
        // Attention projections
        pub q_proj: &'a CudaSlice<f16>,
        pub k_proj: &'a CudaSlice<f16>,
        pub v_proj: &'a CudaSlice<f16>,
        pub o_proj: &'a CudaSlice<f16>,
        // Post-attention norm
        pub post_attention_layernorm: &'a CudaSlice<f16>,
        // MLP weights
        pub gate_proj: &'a CudaSlice<f16>,
        pub up_proj: &'a CudaSlice<f16>,
        pub down_proj: &'a CudaSlice<f16>,
        /// Fused QKV weight: [q_dim + kv_dim + kv_dim, hidden]. None if not fused.
        pub fused_qkv: Option<&'a CudaSlice<f16>>,
        /// Fused gate+up weight: [intermediate*2, hidden]. None if not fused.
        pub fused_gate_up: Option<&'a CudaSlice<f16>>,
        /// Fused QKV bias (f16). None if model has no QKV bias.
        pub qkv_bias: Option<&'a CudaSlice<f16>>,
    }

    /// Metadata needed for a single layer forward pass.
    pub struct GpuLayerInput<'a> {
        /// Hidden states entering this layer, shape [num_tokens, hidden_size] (f16).
        pub hidden_states: &'a CudaSlice<f16>,
        /// Position ids for RoPE, shape [num_tokens]. Kernels expect int*.
        pub positions: CudaView<'a, i32>,
        /// KV cache key block for this layer (f16), shape [num_blocks, block_size, num_kv_heads, head_dim].
        pub key_cache: &'a CudaSlice<f16>,
        /// KV cache value block for this layer (f16), shape [num_blocks, block_size, num_kv_heads, head_dim].
        pub value_cache: &'a CudaSlice<f16>,
        /// Block table mapping sequence positions to cache blocks, shape [num_seqs, max_blocks_per_seq].
        pub block_tables: CudaView<'a, i32>,
        /// Context length for each sequence, shape [num_seqs].
        pub context_lens: CudaView<'a, i32>,
        /// Slot mapping for cache writes during prefill, shape [num_tokens].
        pub slot_mapping: CudaView<'a, i32>,
        /// Number of tokens in the batch.
        pub num_tokens: usize,
        /// Number of sequences in the batch.
        pub num_seqs: usize,
        /// Maximum context length across sequences.
        pub max_context_len: u32,
        /// Block size for paged attention.
        pub block_size: usize,
        /// True during prefill (prompt processing), false during decode.
        pub is_prefill: bool,
        /// Per-sequence query token start positions: [num_seqs + 1] with sentinel.
        /// Built from actual query token counts, NOT context_lens.
        pub seq_start_pos: CudaView<'a, i32>,
        /// RoPE cos table on GPU: [max_position, head_dim/2].
        pub rope_cos: &'a CudaSlice<f32>,
        /// RoPE sin table on GPU: [max_position, head_dim/2].
        pub rope_sin: &'a CudaSlice<f32>,
    }

    /// One complete GPU transformer layer.
    ///
    /// Holds references to the kernel loader and cuBLAS handle;
    /// weights are passed in per-call via `GpuLayerWeights`.
    pub struct GpuTransformerLayer {
        config: GpuLayerConfig,
        stream: Arc<CudaStream>,
        loader: Arc<KernelLoader>,
    }

    impl GpuTransformerLayer {
        pub fn new(config: GpuLayerConfig, stream: Arc<CudaStream>, loader: Arc<KernelLoader>) -> Self {
            Self { config, stream, loader }
        }

        /// Execute a full transformer layer forward pass.
        ///
        /// Returns the output hidden states as a new CudaSlice<f32> of shape
        /// [num_tokens, hidden_size]. The caller is responsible for using this
        /// as input to the next layer.
        /// Fully f16 forward pass -- ZERO type casts within the layer.
        /// All inputs, intermediates, and outputs are f16.
        //   B) qkv [T*(Q+2K)] dead at step 5 -> reuse for gate_up [T*2I] at step 9
        //      (qkv_dim ~ H+2*H/GQA_ratio, gate_up = 2*I; for llama 2I > qkv, but
        //       qkv buf could be a prefix if gate_up scratch is sized to max(qkv,2I))
        //   C) attn_out [T*Q] dead at step 7 -> reuse for fused/silu [T*I] at step 9
        //      (I > Q typically, so need max(Q,I) sizing)
        //   D) attn_proj [T*H] dead at step 8 -> reuse for mlp_out [T*H] at step 10
        //      (exact same size)
        //   E) normed2 [T*H] dead at step 9 -> reuse for mlp_out [T*H] at step 10
        //      (exact same size, alternative to D)
        //
        // LOW-HANGING FRUIT (same size, zero waste):
        //   1. attn_proj -> mlp_out  [both T*H, saves 1 alloc]
        //   2. normed -> attn_out    [T*H vs T*Q, same when Q=H, saves 1 alloc]
        //   3. normed2 is an alternative to attn_proj for mlp_out reuse
        //
        // With a single scratch buffer sized to max(T*qkv_dim, T*2I, T*H):
        //   Could eliminate 4-5 allocs, reducing 10 -> 5-6 per layer.
        // ==================================================================
        /// Full f16 forward with cross-layer fusion support.
        /// When `prev_mlp_out` is Some, fuses the previous layer's residual add
        /// with this layer's pre-attention RMSNorm (1 kernel instead of 2).
        /// Returns (residual, mlp_out) for the NEXT layer to fuse.
        pub fn forward(
            &self,
            input: &GpuLayerInput<'_>,
            weights: &GpuLayerWeights<'_>,
            blas: &CublasHandle,
            prev_mlp_out: Option<&CudaSlice<f16>>,
            lt: Option<&crate::CublasLtRef>,
        ) -> Result<(CudaSlice<f16>, CudaSlice<f16>)> {
            let cfg = &self.config;
            let num_tokens = input.num_tokens;
            let hidden = cfg.hidden_size;
            let num_heads = cfg.num_heads;
            let num_kv_heads = cfg.num_kv_heads;
            let head_dim = cfg.head_dim;
            let intermediate = cfg.intermediate_size;

            let hidden_f16 = input.hidden_states;
            let norm_w = weights.input_layernorm;
            let post_norm_w = weights.post_attention_layernorm;
            let dbg = cfg.layer_idx < 1 && std::env::var("RVLLM_DEBUG").is_ok();
            let dbg_dump = |label: &str, buf: &CudaSlice<f16>, stream: &Arc<CudaStream>| {
                if let Ok(vals) = stream.clone_dtoh(buf) {
                    let first5: Vec<f32> = vals.iter().take(5).map(|v| v.to_f32()).collect();
                    let last5: Vec<f32> = vals.iter().rev().take(5).rev().map(|v| v.to_f32()).collect();
                    let nan = vals.iter().any(|v| v.to_f32().is_nan());
                    let max = vals.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max);
                    let mean = vals.iter().map(|v| v.to_f32()).sum::<f32>() / vals.len() as f32;
                    info!("DEBUG L0 {label}: first5={first5:?} last5={last5:?} max={max:.4} mean={mean:.6} nan={nan} len={}", vals.len());
                }
            };

            // 1. Pre-attention RMSNorm f16 (fused with previous layer's residual add if available)
            let (normed, fused_residual) = if let Some(prev_mlp) = prev_mlp_out {
                // Fuse: hidden + prev_mlp -> norm in one kernel
                let (n, r) = Self::fused_residual_rmsnorm_f16(
                    &self.stream, &self.loader,
                    hidden_f16, prev_mlp, norm_w,
                    cfg.rms_norm_eps, num_tokens, hidden,
                )?;
                (n, Some(r))
            } else {
                // First layer: standalone norm
                let n = Self::rms_norm_f16(
                    &self.stream, &self.loader,
                    hidden_f16, norm_w,
                    cfg.rms_norm_eps, num_tokens, hidden,
                )?;
                (n, None)
            };
            // The residual stream: either the fused result or the original hidden_states
            let residual_ref = fused_residual.as_ref().unwrap_or(hidden_f16);
            if dbg { dbg_dump("normed", &normed, &self.stream); }

            // 2. QKV projections: hgemm f16 x f16 -> f16
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let qkv_dim = q_dim + kv_dim + kv_dim;

            // Fused QKV GEMM outputs [T, qkv_dim] where each row is [Q_t, K_t, V_t].
            // The downstream split (slice at q_end, k_end) assumes [all_Q | all_K | all_V].
            // These layouts match only when T=1. For T>1 (prefill), use unfused path.
            let mut qkv = if let Some(fused_qkv) = weights.fused_qkv {
                if num_tokens == 1 {
                    // T=1: [1, qkv_dim] = [Q | K | V], split is correct
                    Self::hgemm_dispatch(&self.stream, blas, lt, &normed, fused_qkv, num_tokens, qkv_dim, hidden, &self.loader)?
                } else {
                    // T>1: fused GEMM gives [T, qkv_dim] with interleaved Q/K/V per row.
                    // Use individual weight projections for correct [all_Q | all_K | all_V] layout.
                    let mut qkv_buf = unsafe { self.stream.alloc::<f16>(num_tokens * qkv_dim) }
                        .map_err(|e| LLMError::GpuError(format!("qkv f16 alloc: {e}")))?;
                    let q_end_t = num_tokens * q_dim;
                    let k_end_t = q_end_t + num_tokens * kv_dim;
                    {
                        let mut q_dst = qkv_buf.slice_mut(..q_end_t);
                        blas.hgemm_into(num_tokens, q_dim, hidden, 1.0, &normed, weights.q_proj, 0.0, &mut q_dst)?;
                    }
                    {
                        let mut k_dst = qkv_buf.slice_mut(q_end_t..k_end_t);
                        blas.hgemm_into(num_tokens, kv_dim, hidden, 1.0, &normed, weights.k_proj, 0.0, &mut k_dst)?;
                    }
                    {
                        let mut v_dst = qkv_buf.slice_mut(k_end_t..);
                        blas.hgemm_into(num_tokens, kv_dim, hidden, 1.0, &normed, weights.v_proj, 0.0, &mut v_dst)?;
                    }
                    qkv_buf
                }
            } else {
                let mut qkv_buf = unsafe { self.stream.alloc::<f16>(num_tokens * qkv_dim) }
                    .map_err(|e| LLMError::GpuError(format!("qkv f16 alloc: {e}")))?;
                let q_end = num_tokens * q_dim;
                let k_end = q_end + num_tokens * kv_dim;
                {
                    let mut q_dst = qkv_buf.slice_mut(..q_end);
                    blas.hgemm_into(num_tokens, q_dim, hidden, 1.0, &normed, weights.q_proj, 0.0, &mut q_dst)?;
                }
                {
                    let mut k_dst = qkv_buf.slice_mut(q_end..k_end);
                    blas.hgemm_into(num_tokens, kv_dim, hidden, 1.0, &normed, weights.k_proj, 0.0, &mut k_dst)?;
                }
                {
                    let mut v_dst = qkv_buf.slice_mut(k_end..);
                    blas.hgemm_into(num_tokens, kv_dim, hidden, 1.0, &normed, weights.v_proj, 0.0, &mut v_dst)?;
                }
                qkv_buf
            };

            if dbg { dbg_dump("qkv", &qkv, &self.stream); }

            // 3. Apply fused QKV bias (f16) in-place
            let q_end = num_tokens * q_dim;
            let k_end = q_end + num_tokens * kv_dim;
            if let Some(bias) = weights.qkv_bias {
                if num_tokens == 1 {
                    // T=1: layout is [Q, K, V] per row, fused bias matches
                    let mut qkv_view = qkv.slice_mut(..num_tokens * qkv_dim);
                    Self::add_bias_f16_view(&self.stream, &self.loader, &mut qkv_view, bias, num_tokens, qkv_dim)?;
                } else {
                    // T>1: layout is [all_Q | all_K | all_V], apply bias per section.
                    // Copy bias slices to temp CudaSlice (bias is tiny, ~2KB).
                    let mut q_bias_buf = self.stream.alloc_zeros::<f16>(q_dim)
                        .map_err(|e| LLMError::GpuError(format!("q_bias alloc: {e}")))?;
                    self.stream.memcpy_dtod(&bias.slice(..q_dim), &mut q_bias_buf)
                        .map_err(|e| LLMError::GpuError(format!("q_bias copy: {e}")))?;
                    let mut k_bias_buf = self.stream.alloc_zeros::<f16>(kv_dim)
                        .map_err(|e| LLMError::GpuError(format!("k_bias alloc: {e}")))?;
                    self.stream.memcpy_dtod(&bias.slice(q_dim..q_dim + kv_dim), &mut k_bias_buf)
                        .map_err(|e| LLMError::GpuError(format!("k_bias copy: {e}")))?;
                    let mut v_bias_buf = self.stream.alloc_zeros::<f16>(kv_dim)
                        .map_err(|e| LLMError::GpuError(format!("v_bias alloc: {e}")))?;
                    self.stream.memcpy_dtod(&bias.slice(q_dim + kv_dim..qkv_dim), &mut v_bias_buf)
                        .map_err(|e| LLMError::GpuError(format!("v_bias copy: {e}")))?;
                    {
                        let mut q_view = qkv.slice_mut(..q_end);
                        Self::add_bias_f16_view(&self.stream, &self.loader, &mut q_view, &q_bias_buf, num_tokens, q_dim)?;
                    }
                    {
                        let mut k_view = qkv.slice_mut(q_end..k_end);
                        Self::add_bias_f16_view(&self.stream, &self.loader, &mut k_view, &k_bias_buf, num_tokens, kv_dim)?;
                    }
                    {
                        let mut v_view = qkv.slice_mut(k_end..);
                        Self::add_bias_f16_view(&self.stream, &self.loader, &mut v_view, &v_bias_buf, num_tokens, kv_dim)?;
                    }
                }
            }

            // 4. RoPE f16 in-place on Q and K regions (split_at_mut avoids double borrow)
            {
                let (mut q_part, mut kv_part) = qkv.split_at_mut(q_end);
                let mut k_view = kv_part.slice_mut(..num_tokens * kv_dim);
                Self::apply_rotary_embedding_f16_views(
                    &self.stream, &self.loader,
                    &mut q_part, &mut k_view,
                    &input.positions, input.rope_cos, input.rope_sin,
                    num_tokens, num_heads, num_kv_heads, head_dim,
                )?;
            }

            // 5. KV cache write: f16 K/V -> f16 cache (pure copy, no conversion)
            {
                let k_view = qkv.slice(q_end..k_end);
                let v_view = qkv.slice(k_end..);
                Self::cache_write_f16_views(
                    &self.stream, &self.loader,
                    &k_view, &v_view,
                    input.key_cache, input.value_cache, &input.slot_mapping,
                    num_tokens, num_kv_heads, head_dim,
                )?;
            }

            // 6. Attention: f16 Q, f16 KV cache, f16 output
            let attn_out = if input.is_prefill {
                // Prefill uses f32 Q kernel (prefill is one-shot, not perf-critical)
                // Fall back to f32 path for prefill: cast Q f16->f32, run f32 prefill, cast back
                let cast_f16_f32 = self.loader.get_func("cast_fp", "cast_f16_to_f32_kernel")
                    .map_err(|e| LLMError::GpuError(format!("load cast f16->f32: {e}")))?;
                let cast_f32_f16 = self.loader.get_func("cast_fp", "cast_f32_to_f16_kernel")
                    .map_err(|e| LLMError::GpuError(format!("load cast f32->f16: {e}")))?;
                let q_f16 = qkv.slice(..q_end);
                let q_f32 = Self::cast_f16_to_f32(&self.stream, &q_f16, q_end, &cast_f16_f32)?;
                let attn_f32 = Self::prefill_attention(
                    &self.stream, &self.loader,
                    &q_f32.as_view(),
                    input.key_cache, input.value_cache,
                    &input.block_tables, &input.context_lens, &input.seq_start_pos,
                    num_tokens, input.num_seqs, num_heads, num_kv_heads, head_dim,
                    input.max_context_len, input.block_size,
                )?;
                Self::cast_f32_to_f16(&self.stream, &attn_f32, num_tokens * num_heads * head_dim, &cast_f32_f16)?
            } else {
                Self::decode_attention_f16io(
                    &self.stream, &self.loader,
                    &qkv.slice(..q_end),
                    input.key_cache, input.value_cache,
                    &input.block_tables, &input.context_lens,
                    num_tokens, input.num_seqs, num_heads, num_kv_heads, head_dim,
                    input.max_context_len, input.block_size,
                )?
            };

            if dbg { dbg_dump("attn_out", &attn_out, &self.stream); }

            // 7. Output projection: hgemm f16
            let attn_proj = Self::hgemm_dispatch(&self.stream, blas, lt, &attn_out, weights.o_proj, num_tokens, hidden, q_dim, &self.loader)?;
            if dbg { dbg_dump("o_proj", &attn_proj, &self.stream); }

            // 8. Fused residual + post-attention RMSNorm f16
            let (normed2, residual) = Self::fused_residual_rmsnorm_f16(
                &self.stream, &self.loader,
                residual_ref, &attn_proj, post_norm_w,
                cfg.rms_norm_eps, num_tokens, hidden,
            )?;

            // 9. MLP: gate+up -> fused_silu_mul_f16 -> down, all f16
            let fused = if let Some(fused_gate_up) = weights.fused_gate_up {
                if num_tokens == 1 {
                    // T=1: [1, 2*I] = [gate | up], split is correct
                    let gate_up = Self::hgemm_dispatch(&self.stream, blas, lt, &normed2, fused_gate_up, num_tokens, intermediate * 2, hidden, &self.loader)?;
                    let n = num_tokens * intermediate;
                    Self::fused_silu_mul_f16_split(&self.stream, &self.loader, &gate_up, n)?
                } else {
                    // T>1: fused gives [T, 2*I] interleaved. Use individual weights.
                    let gate = Self::hgemm_dispatch(&self.stream, blas, lt, &normed2, weights.gate_proj, num_tokens, intermediate, hidden, &self.loader)?;
                    let up = Self::hgemm_dispatch(&self.stream, blas, lt, &normed2, weights.up_proj, num_tokens, intermediate, hidden, &self.loader)?;
                    Self::fused_silu_mul_f16(&self.stream, &self.loader, &gate, &up, num_tokens * intermediate)?
                }
            } else {
                let gate = Self::hgemm_dispatch(&self.stream, blas, lt, &normed2, weights.gate_proj, num_tokens, intermediate, hidden, &self.loader)?;
                let up = Self::hgemm_dispatch(&self.stream, blas, lt, &normed2, weights.up_proj, num_tokens, intermediate, hidden, &self.loader)?;
                Self::fused_silu_mul_f16(&self.stream, &self.loader, &gate, &up, num_tokens * intermediate)?
            };

            // 10. Down projection: hgemm f16
            let mlp_out = Self::hgemm_dispatch(&self.stream, blas, lt, &fused, weights.down_proj, num_tokens, hidden, intermediate, &self.loader)?;

            // 11. Return (residual, mlp_out) -- the add is fused into the NEXT layer's norm
            Ok((residual, mlp_out))
        }

        // ===================================================================
        // f16 dispatch helpers
        // ===================================================================

        /// RMSNorm f16: f16 input, f16 weight, f16 output.
        fn rms_norm_f16(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            input: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
        ) -> Result<CudaSlice<f16>> {
            let n = num_tokens * hidden_size;
            // Safety: kernel writes all num_tokens * hidden_size elements
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("rms_norm_f16 alloc: {e}")))?;
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };
            let kernel = loader.get_func("rms_norm_f16", "rms_norm_f16_kernel")?;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(input)
                    .arg(weight)
                    .arg(&eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rms_norm_f16 launch: {e}")))?;
            }
            Ok(output)
        }

        /// In-place RMSNorm f16: normalizes `input` directly, no output allocation.
        /// Safe because the kernel uses __syncthreads() between the read pass and
        /// write pass within each block (single-token-per-block launch).
        fn rms_norm_f16_inplace(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            input: &mut CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
        ) -> Result<()> {
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };
            let kernel = loader.get_func("rms_norm_f16", "rms_norm_f16_kernel")?;
            unsafe {
                let (raw_ptr, _guard) = DevicePtrMut::device_ptr_mut(input, stream);
                stream.launch_builder(&kernel)
                    .arg(&raw_ptr)  // output (same ptr)
                    .arg(&raw_ptr)  // input  (same ptr)
                    .arg(weight)
                    .arg(&eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rms_norm_f16_inplace launch: {e}")))?;
            }
            Ok(())
        }

        /// hgemm with output allocation: f16 x f16 -> f16.
        fn hgemm_alloc(
            stream: &Arc<CudaStream>,
            blas: &CublasHandle,
            input: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<CudaSlice<f16>> {
            // Safety: hgemm with beta=0 writes all m*n elements
            let mut output = unsafe { stream.alloc::<f16>(m * n) }
                .map_err(|e| LLMError::GpuError(format!("hgemm_alloc: {e}")))?;
            blas.hgemm(m, n, k, f16::ONE, input, weight, f16::ZERO, &mut output)?;
            Ok(output)
        }

        /// hgemm dispatch: uses cublasLt for M<=32 (split-K), falls back to standard cuBLAS.
        /// GEMM dispatch: custom GEMV for M=1, cublasLt for M<=32, cuBLAS for larger.
        fn hgemm_dispatch(
            stream: &Arc<CudaStream>,
            blas: &CublasHandle,
            lt: Option<&crate::CublasLtRef>,
            input: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            m: usize,
            n: usize,
            k: usize,
            loader: &KernelLoader,
        ) -> Result<CudaSlice<f16>> {
            let mut output = unsafe { stream.alloc::<f16>(m * n) }
                .map_err(|e| LLMError::GpuError(format!("hgemm_dispatch: {e}")))?;

            // M=1: custom GEMV kernel (vectorized half2, warp shuffle reduction)
            if m == 1 {
                if let Ok(kernel) = loader.get_func("gemv_f16", "gemv_f16_kernel") {
                    let cfg = LaunchConfig {
                        grid_dim: (n as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    unsafe {
                        stream.launch_builder(&kernel)
                            .arg(&mut output)
                            .arg(weight)
                            .arg(input)
                            .arg(&(n as i32))
                            .arg(&(k as i32))
                            .launch(cfg)
                            .map_err(|e| LLMError::GpuError(format!("gemv_f16 launch: {e}")))?;
                    }
                    return Ok(output);
                }
                // Fallthrough to cuBLAS if kernel not loaded
            }

            #[cfg(feature = "cublaslt")]
            if let Some(lt_ops) = lt {
                if m <= rvllm_gpu::cublaslt_ops::CUBLASLT_M_THRESHOLD {
                    lt_ops.hgemm_a_bt(m, n, k, 1.0, input, weight, 0.0, &mut output)?;
                    return Ok(output);
                }
            }
            blas.hgemm(m, n, k, f16::ONE, input, weight, f16::ZERO, &mut output)?;
            Ok(output)
        }

        /// Add bias f16 in-place on a CudaViewMut<f16>.
        fn add_bias_f16_view(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            tensor: &mut CudaViewMut<'_, f16>,
            bias: &CudaSlice<f16>,
            num_tokens: usize,
            dim: usize,
        ) -> Result<()> {
            let kernel = loader.get_func("add_bias_f16", "add_bias_f16_kernel")?;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(tensor)
                    .arg(bias)
                    .arg(&(dim as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("add_bias_f16 launch: {e}")))?;
            }
            Ok(())
        }

        /// RoPE f16 in-place on CudaViewMut<f16> Q/K slices.
        fn apply_rotary_embedding_f16_views(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &mut CudaViewMut<'_, f16>,
            k: &mut CudaViewMut<'_, f16>,
            positions: &CudaView<'_, i32>,
            rope_cos: &CudaSlice<f32>,
            rope_sin: &CudaSlice<f32>,
            num_tokens: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<()> {
            if num_tokens == 0 { return Ok(()); }
            let kernel = loader.get_func("rotary_embedding_f16", "rotary_embedding_f16_kernel")?;
            let half_dim = head_dim / 2;
            let grid_y = num_heads.max(num_kv_heads) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, grid_y, 1),
                block_dim: (half_dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(q).arg(k)
                    .arg(rope_cos).arg(rope_sin).arg(positions)
                    .arg(&(num_tokens as i32)).arg(&(num_heads as i32))
                    .arg(&(num_kv_heads as i32)).arg(&(head_dim as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rope_f16 launch: {e}")))?;
            }
            Ok(())
        }

        /// Cache write f16: f16 K/V input -> f16 paged cache (pure copy).
        fn cache_write_f16_views(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            k: &CudaView<'_, f16>,
            v: &CudaView<'_, f16>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            slot_mapping: &CudaView<'_, i32>,
            num_tokens: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<()> {
            let kv_dim = num_kv_heads * head_dim;
            let kernel = loader.get_func("reshape_and_cache_f16", "reshape_and_cache_f16io_kernel")?;
            let threads = kv_dim.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(key_cache).arg(value_cache)
                    .arg(k).arg(v)
                    .arg(slot_mapping)
                    .arg(&(num_tokens as i32))
                    .arg(&(num_kv_heads as i32))
                    .arg(&(head_dim as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("cache_write_f16 launch: {e}")))?;
            }
            Ok(())
        }

        /// FA2 prefill attention: f32 Q, f16 KV cache, f32 output.
        /// The only f32 touch point in the forward pass (prefill is one-shot).
        #[allow(clippy::too_many_arguments)]
        fn prefill_attention(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &CudaView<'_, f32>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            block_tables: &CudaView<'_, i32>,
            context_lens: &CudaView<'_, i32>,
            seq_start_pos: &CudaView<'_, i32>,
            num_tokens: usize,
            num_seqs: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            max_context_len: u32,
            block_size: usize,
        ) -> Result<CudaSlice<f32>> {
            let out_len = num_tokens * num_heads * head_dim;
            let output = stream
                .alloc_zeros::<f32>(out_len)
                .map_err(|e| LLMError::GpuError(format!("prefill_attn alloc: {e}")))?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();

            const FA2_BC: usize = 64;
            const FA2_THREADS: u32 = 128;
            let shared_mem_bytes = ((2 * FA2_BC * head_dim + FA2_BC + (FA2_THREADS as usize / 32))
                * std::mem::size_of::<f32>()) as u32;

            let kernel = loader.get_func("flash_attention", "flash_attention_2_f16kv_kernel")?;

            if num_seqs == 0 {
                return Err(LLMError::GpuError("prefill_attention: num_seqs == 0".into()));
            }

            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_heads as u32, 1),
                block_dim: (FA2_THREADS, 1, 1),
                shared_mem_bytes,
            };

            let max_blocks_per_seq = (DeviceSlice::len(block_tables) / num_seqs) as i32;

            if shared_mem_bytes > 49152 {
                kernel.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                ).map_err(|e| LLMError::GpuError(format!("prefill FA2 set max shared mem: {e}")))?;
            }

            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&output)
                    .arg(q)
                    .arg(key_cache)
                    .arg(value_cache)
                    .arg(block_tables)
                    .arg(context_lens)
                    .arg(seq_start_pos)
                    .arg(&scale)
                    .arg(&(num_heads as i32))
                    .arg(&(num_kv_heads as i32))
                    .arg(&(head_dim as i32))
                    .arg(&(block_size as i32))
                    .arg(&(max_context_len as i32))
                    .arg(&max_blocks_per_seq)
                    .arg(&(num_tokens as i32))
                    .arg(&1i32) // causal
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("prefill FA2 launch: {e}")))?;
            }

            Ok(output)
        }

        /// FA2 decode attention with f16 I/O: f16 Q, f16 KV cache, f16 output.
        #[allow(clippy::too_many_arguments)]
        fn decode_attention_f16io(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &CudaView<'_, f16>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            block_tables: &CudaView<'_, i32>,
            context_lens: &CudaView<'_, i32>,
            num_tokens: usize,
            num_seqs: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            max_context_len: u32,
            block_size: usize,
        ) -> Result<CudaSlice<f16>> {
            let out_len = num_tokens * num_heads * head_dim;
            // Safety: attention kernel writes all out_len elements
            let mut output = unsafe { stream.alloc::<f16>(out_len) }
                .map_err(|e| LLMError::GpuError(format!("decode_attn_f16io alloc: {e}")))?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();

            const FA2_BC: usize = 64;
            const FA2_THREADS: u32 = 128;
            let shared_mem_bytes = ((2 * FA2_BC * head_dim + FA2_BC + (FA2_THREADS as usize / 32))
                * std::mem::size_of::<f32>()) as u32;

            let kernel = loader.get_func("flash_attention", "flash_attention_2_decode_f16io_kernel")?;

            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_heads as u32, 1),
                block_dim: (FA2_THREADS, 1, 1),
                shared_mem_bytes,
            };

            if shared_mem_bytes > 49152 {
                kernel.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                ).map_err(|e| LLMError::GpuError(format!("decode FA2 f16io set max shared mem: {e}")))?;
            }

            let p_num_heads = num_heads as i32;
            let p_num_kv_heads = num_kv_heads as i32;
            let p_head_dim = head_dim as i32;
            let p_block_size = block_size as i32;
            let p_max_blocks = (block_tables.len() / num_seqs.max(1)) as i32;

            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(q)
                    .arg(key_cache).arg(value_cache)
                    .arg(block_tables).arg(context_lens)
                    .arg(&scale)
                    .arg(&p_num_heads).arg(&p_num_kv_heads)
                    .arg(&p_head_dim).arg(&p_block_size)
                    .arg(&p_max_blocks)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("FA2 decode f16io launch: {e}")))?;
            }
            Ok(output)
        }

        /// Fused residual add + RMSNorm, all f16.
        /// Returns (normed_output, residual) both f16.
        pub fn fused_residual_rmsnorm_f16(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            input: &CudaSlice<f16>,
            add: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
        ) -> Result<(CudaSlice<f16>, CudaSlice<f16>)> {
            let n = num_tokens * hidden_size;
            // Safety: kernel writes all n elements to both output and residual
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("fused_rn_f16 output alloc: {e}")))?;
            let mut residual = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("fused_rn_f16 residual alloc: {e}")))?;
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };
            let kernel = loader.get_func("fused_residual_rmsnorm_f16", "fused_residual_rmsnorm_f16_kernel")?;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(&mut residual)
                    .arg(input)
                    .arg(add)
                    .arg(weight)
                    .arg(&eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("fused_rn_f16 launch: {e}")))?;
            }
            Ok((output, residual))
        }

        /// Fused SiLU * mul, f16 variant.
        fn fused_silu_mul_f16(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            gate: &CudaSlice<f16>,
            up: &CudaSlice<f16>,
            n: usize,
        ) -> Result<CudaSlice<f16>> {
            // Safety: element-wise kernel writes all n elements
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_f16 alloc: {e}")))?;
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let kernel = loader.get_func("activation_f16", "fused_silu_mul_f16_kernel")?;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(gate).arg(up)
                    .arg(&(n as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_f16 launch: {e}")))?;
            }
            Ok(output)
        }

        /// Fused SiLU*mul on a contiguous [gate || up] buffer, f16 variant.
        fn fused_silu_mul_f16_split(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            gate_up: &CudaSlice<f16>,
            n: usize,
        ) -> Result<CudaSlice<f16>> {
            let gate_view = gate_up.slice(..n);
            let up_view = gate_up.slice(n..n * 2);
            // Safety: element-wise kernel writes all n elements
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_f16_split alloc: {e}")))?;
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let kernel = loader.get_func("activation_f16", "fused_silu_mul_f16_kernel")?;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(&gate_view).arg(&up_view)
                    .arg(&(n as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_f16_split launch: {e}")))?;
            }
            Ok(output)
        }

        /// Element-wise tensor addition f16: out = a + b.
        fn add_tensors_f16(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            a: &CudaSlice<f16>,
            b: &CudaSlice<f16>,
            n: usize,
        ) -> Result<CudaSlice<f16>> {
            // Safety: element-wise kernel writes all n elements
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("add_tensors_f16 alloc: {e}")))?;
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let kernel = loader.get_func("add_bias_f16", "add_f16_kernel")?;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(a).arg(b)
                    .arg(&(n as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("add_f16 launch: {e}")))?;
            }
            Ok(output)
        }

        /// Cast f16 -> f32 (used only for prefill fallback).
        fn cast_f16_to_f32(
            stream: &Arc<CudaStream>,
            input: &CudaView<'_, f16>,
            n: usize,
            kernel: &CudaFunction,
        ) -> Result<CudaSlice<f32>> {
            // Safety: cast kernel writes all n elements
            let mut output = unsafe { stream.alloc::<f32>(n) }
                .map_err(|e| LLMError::GpuError(format!("cast_f16_f32 alloc: {e}")))?;
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream.launch_builder(kernel)
                    .arg(&mut output)
                    .arg(input)
                    .arg(&(n as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("cast_f16_f32 launch: {e}")))?;
            }
            Ok(output)
        }

        /// Cast f32 -> f16 (used only for prefill fallback).
        fn cast_f32_to_f16(
            stream: &Arc<CudaStream>,
            input: &CudaSlice<f32>,
            n: usize,
            kernel: &CudaFunction,
        ) -> Result<CudaSlice<f16>> {
            // Safety: cast kernel writes all n elements
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("cast_f32_f16 alloc: {e}")))?;
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream.launch_builder(kernel)
                    .arg(&mut output)
                    .arg(input)
                    .arg(&(n as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("cast_f32_f16 launch: {e}")))?;
            }
            Ok(output)
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::*;

#[cfg(test)]
mod tests {
    // Tests run under default features (mock-gpu), so we verify the module
    // compiles but the CUDA types are not exposed.
    #[test]
    fn module_compiles_without_cuda() {
        // Under mock-gpu the `inner` module is not compiled.
        // This test confirms that the crate still builds cleanly.
        assert!(true);
    }
}

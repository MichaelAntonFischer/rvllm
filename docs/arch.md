# rvLLM Architecture: Full Forward Pass Trace

This documents the complete call stack from HTTP request to token output,
with every GPU kernel and data movement annotated. All model data is f16
end-to-end. f32 exists only for RoPE lookup tables, the rms_norm epsilon
scalar, and the final logits output at the argmax/sampling boundary.

---

## 1. Thread Model

```
[tokio async runtime]
  |
  |-- HTTP server (axum) receives /v1/completions
  |-- AsyncGpuEngine: schedules sequences, sends GpuWork::Step
  |
  v
[gpu-step OS thread]  <-- dedicated, owns all CUDA state
  |
  |-- GpuLLMEngine::step()
  |     |-- Scheduler: pick sequences, build SequenceGroupMetadata
  |     |-- GpuWorker::execute() or execute_with_overlap()
  |     |     |-- Build ModelInput from metadata
  |     |     |-- GpuModelRunner forward pass (all GPU work here)
  |     |     |-- DtoH copy of token IDs
  |     |-- Process outputs, update sequences
  |
  v
[tokio async runtime]
  |-- Stream tokens back to HTTP client
```

The GPU thread communicates via `std::sync::mpsc` (work in) and
`tokio::sync::mpsc` (results out). No spin-wait -- the async side
does `result_rx.recv().await`.

---

## 2. Weight Loading

```
GpuWorker::load_weights(model_path)
  |
  |-- gpu_loader::load_weights_to_gpu(path, stream)
  |     |-- mmap safetensors file(s)
  |     |-- For each tensor:
  |     |     F16 on disk  -> memcpy to host Vec<f16> -> clone_htod -> CudaSlice<f16>
  |     |     BF16 on disk -> bf16->f32->f16 per elem -> clone_htod -> CudaSlice<f16>
  |     |     F32 on disk  -> f32->f16 per elem       -> clone_htod -> CudaSlice<f16>
  |     |-- Returns HashMap<String, CudaSlice<f16>>
  |
  |-- Store raw_weight_map for GpuModelRunner construction
  |-- Build local LayerWeights structs (all CudaSlice<f16>)
```

**GpuModelWeights** (in rvllm-model-loader): single HashMap<String, CudaSlice<f16>>.
No dual f32/f16 storage. `get()` returns `&CudaSlice<f16>`.

---

## 3. Runner Construction & Weight Fusion

```
GpuWorker::init_cache(num_gpu_blocks, num_cpu_blocks)
  |
  |-- CudaCacheEngine::new()  -- allocates f16 KV cache for all layers
  |
  |-- GpuModelRunner::new(weights, cache, blas, loader, config, device, stream)
  |     |-- embed_tokens    = weights.get("model.embed_tokens.weight").clone()    [f16]
  |     |-- final_norm_weight = weights.get("model.norm.weight").clone()          [f16]
  |     |-- lm_head_weight  = weights.get("lm_head.weight").clone()              [f16]
  |     |-- Precompute RoPE cos/sin tables in f32 on host, upload to GPU         [f32]
  |     |-- Create GpuTransformerLayer for each layer
  |
  |-- runner.fuse_weights()
        |-- For each layer:
        |     |-- Fuse QKV: DtoD copy q_proj||k_proj||v_proj -> fused_qkv        [f16]
        |     |-- Fuse gate+up: DtoD copy gate_proj||up_proj -> fused_gate_up     [f16]
        |     |-- Fuse QKV bias: DtoD copy q_bias||k_bias||v_bias -> fused_bias   [f16]
        |-- alloc_scratch(): pre-allocate reusable f16 buffers (128 tokens max)
```

No f32->f16 casting at fusion time. Weights are already f16 from the loader.
Fusion is pure DtoD memcpy (concatenation on GPU).

---

## 4. Forward Pass: Decode (Hot Path)

Entry point depends on whether CUDA graphs are active:

```
Without graphs:  GpuModelRunner::forward_gpu_only(N, N, max_ctx, false)
With graphs:     CudaGraph::replay(stream)  [replays captured forward_gpu_only]
Standalone:      GpuModelRunner::forward_ex(token_ids, positions, meta, false, true)
```

All three converge on the same kernel sequence. Here is the full trace
for a single decode step with N tokens:

### 4a. Metadata Upload (1 HtoD)

```
upload_metadata(token_ids, positions, attn_meta)
  |-- Pack 6 arrays into 1 contiguous i32 buffer on host:
  |     [token_ids | positions | context_lens | block_tables | slot_mapping | seq_start_pos]
  |-- Single memcpy_htod into reusable GPU buffer (stable pointer for graphs)
```

### 4b. Embedding Lookup (1 kernel)

```
embedding_lookup_from_meta(num_tokens)
  |-- Kernel: embedding_gather_f16_kernel
  |     Input:  embed_tokens [vocab_size, hidden] f16, token_ids [N] i32
  |     Output: hidden_states [N, hidden] f16
  |-- Grid: (N, 1, 1), Block: (min(hidden, 1024), 1, 1)
```

### 4c. Transformer Layers (repeated num_layers times)

Each layer executes this sequence. Cross-layer fusion: the previous
layer's mlp_out is added to hidden_states inside the NEXT layer's
RMSNorm (fused_residual_rmsnorm), avoiding a separate add kernel.

```
layer.forward(input, weights, blas, prev_mlp_out, cublaslt)
  |
  |-- [1] Pre-attention RMSNorm
  |     First layer:  rms_norm_f16_kernel(hidden, norm_w)           -> normed [N,H] f16
  |     Other layers: fused_residual_rmsnorm_f16_kernel(hidden,     -> (normed, residual) [N,H] f16
  |                     prev_mlp_out, norm_w)
  |     Internal math: f32 accumulation for variance, f16 I/O
  |
  |-- [2] QKV Projection (1 GEMM)
  |     hgemm_dispatch: routes based on M:
  |       M=1:    gemv_f16_kernel (custom, half2 vectorized, warp shuffle)
  |       M<=32:  cublasLt hgemm (split-K)
  |       M>32:   cuBLAS hgemm
  |     Input:  normed [N, H] f16  x  fused_qkv [Q+2K, H] f16
  |     Output: qkv [N, Q+2K] f16
  |
  |-- [3] QKV Bias (1 kernel, optional -- Qwen2.5 has this)
  |     add_bias_f16_kernel(qkv, fused_bias)
  |     In-place on qkv buffer
  |
  |-- [4] RoPE (1 kernel)
  |     rotary_embedding_f16_kernel
  |     In-place on Q and K regions of qkv buffer (split_at_mut)
  |     Reads f32 cos/sin tables, applies to f16 Q/K
  |     Grid: (N, max(num_heads, num_kv_heads), 1)
  |
  |-- [5] KV Cache Write (1 kernel)
  |     reshape_and_cache_f16io_kernel
  |     f16 K/V from qkv buffer -> f16 paged cache (pure copy, no conversion)
  |     Indexed by slot_mapping
  |
  |-- [6] Attention (1 kernel)
  |     Decode: flash_attention_2_decode_f16io_kernel
  |       f16 Q (from qkv), f16 KV cache -> f16 output
  |       Internal: f32 dot products, f32 softmax, f16 output
  |       Grid: (num_seqs, num_heads, 1), Block: (128, 1, 1)
  |     Prefill: cast Q f16->f32, flash_attention_2_f16kv_kernel, cast output f32->f16
  |       (one-shot, not perf-critical)
  |
  |-- [7] Output Projection (1 GEMM)
  |     hgemm_dispatch: attn_out [N, Q] f16  x  o_proj [H, Q] f16 -> [N, H] f16
  |
  |-- [8] Fused Residual + Post-Attention RMSNorm (1 kernel)
  |     fused_residual_rmsnorm_f16_kernel(residual, attn_proj, post_norm_w)
  |     -> (normed2, new_residual)
  |
  |-- [9] MLP Gate+Up (1 GEMM)
  |     hgemm_dispatch: normed2 [N, H] f16  x  fused_gate_up [2*I, H] f16 -> [N, 2*I] f16
  |
  |-- [10] Fused SiLU * Mul (1 kernel)
  |     fused_silu_mul_f16_kernel on the [gate || up] buffer
  |     gate = buf[0..N*I], up = buf[N*I..N*2I]
  |     Output: [N, I] f16
  |
  |-- [11] MLP Down Projection (1 GEMM)
  |     hgemm_dispatch: silu_out [N, I] f16  x  down_proj [H, I] f16 -> mlp_out [N, H] f16
  |
  |-- Return (residual, mlp_out)
  |     The add of residual + mlp_out is DEFERRED to the next layer's
  |     fused_residual_rmsnorm (step 1), saving one kernel.
```

### 4d. Final Norm (1 kernel)

```
After last layer, fuse the final residual add with the model's final RMSNorm:

fused_residual_rmsnorm_f16_kernel(hidden, last_mlp_out, final_norm_weight)
  -> normed_f16 [N, H] f16
```

### 4e. LM Head + Token Selection

**Single-token greedy (decode hot path):**
```
gpu_fused_lm_head_argmax_f16_hidden(normed_f16, lm_head_weight)
  |-- Cast normed f16 -> f32 (1 kernel, hidden_size elements only)
  |-- fused_lm_head_argmax_f16_kernel:
  |     Each block: dot(hidden_f32, lm_head_row_f16) -> partial max
  |     Reduction: argmax across blocks
  |     Output: 1 x i32 token ID
```

**Multi-token or non-greedy:**
```
CudaLinearLayer::forward_f16_in(normed_f16, lm_head_weight)
  |-- hgemm_f32_output: f16 x f16 -> f32 logits [N, vocab] via cublasGemmEx
  |-- gpu_argmax or DtoH for sampling
```

### 4f. Output (1 DtoH)

```
read_graph_output_async(num_tokens, pinned_buffer)
  |-- memcpy_dtoh: graph_output [N] i32 -> pinned host buffer
  |-- (async: stream not synced yet)

sync_stream()
  |-- cuStreamSynchronize
  |-- Read token IDs from pinned buffer
```

---

## 5. Kernel Count Per Decode Step

For N=1 greedy decode with fused weights and no QKV bias:

| Phase | Kernels | Notes |
|-------|---------|-------|
| Metadata upload | 0 | 1 memcpy_htod |
| Embedding | 1 | embedding_gather_f16 |
| Per layer (x L) | 7 | norm + GEMM + RoPE + cache_write + attn + GEMM + fused_resid_norm ... |
| ... continued | +3 | ... + GEMM + silu_mul + GEMM = 10 per layer |
| Final norm | 1 | fused_residual_rmsnorm_f16 |
| LM head + argmax | 2 | cast_f16_f32 + fused_lm_head_argmax_f16 |
| DtoH | 0 | 1 memcpy_dtoh |

**Total: 4 + 10L kernels** (L = num_layers).
For Qwen2.5-1.5B (28 layers): **284 kernels per decode step**.

With CUDA graphs: 1 cuGraphLaunch replaces all 284.

---

## 6. Data Types Summary

| Data | Type | Location |
|------|------|----------|
| Model weights (all) | f16 | GPU, GpuModelWeights |
| Fused QKV weights | f16 | GPU, runner.fused_qkv_weights |
| Fused gate+up weights | f16 | GPU, runner.fused_gate_up_weights |
| Fused QKV bias | f16 | GPU, runner.fused_qkv_bias |
| Embedding table | f16 | GPU, runner.embed_tokens |
| Final norm weight | f16 | GPU, runner.final_norm_weight |
| LM head weight | f16 | GPU, runner.lm_head_weight |
| Hidden states | f16 | GPU, per-step |
| KV cache | f16 | GPU, CudaCacheEngine |
| RoPE cos/sin tables | f32 | GPU, runner.rope_cos/sin |
| RMSNorm epsilon | f32 | Scalar parameter |
| Logits (LM head output) | f32 | GPU, at sampling boundary |
| Token IDs (argmax output) | i32 | GPU -> pinned host |
| Metadata (positions, etc.) | i32 | Host -> GPU, packed buffer |

---

## 7. CUDA Graph Dispatch (gpu_forward_ex)

CUDA graphs capture the kernel sequence from `forward_gpu_only()`:
embedding -> all layers -> final norm -> LM head -> argmax -> dtod copy to output buffer.

**Dispatch logic (per decode step):**
```
gpu_forward_ex(model_input, greedy_only=true)
  |
  |-- Not decode or not greedy? -> raw_gpu_forward_ex (no graphs)
  |
  |-- Graph exists for exact batch size?
  |     YES -> gpu_forward_ex_graphed_exact:
  |              upload_metadata()     [memcpy_htod, stable pointers]
  |              graph.replay()        [cuGraphLaunch, ~0.01ms]
  |              read_graph_output()   [sync DtoH, N x i32]
  |              return TokenIds
  |
  |-- Past warmup (>3 calls) and not yet attempted?
  |     YES -> try_capture_graph_exact:
  |              upload_metadata()     [warmup data]
  |              forward_gpu_only()    [warmup run, triggers lazy init]
  |              stream.synchronize()  [drain all pending ops]
  |              upload_metadata()     [re-upload for capture]
  |              begin_capture_on()    [cuStreamBeginCapture]
  |              forward_gpu_only()    [captured!]
  |              end_capture_on()      [cuStreamEndCapture -> CudaGraph]
  |              pool.insert(graph)    [cache for future replay]
  |              read_graph_output()   [return first result]
  |     FAIL -> warn, mark attempted, fall through
  |
  |-- Fallback: raw_gpu_forward_ex (pre-warmup or capture failure)
```

Graphs are keyed by exact batch size (no padding). Each unique N
encountered during decode gets its own graph after warmup. Typical
steady-state: 1-3 graphs (N=1 for single-sequence, N=batch_size for
concurrent requests).

**Requirements for graph capture:**
- All GPU pointers must be stable (no per-step allocation)
- `upload_metadata()` writes to stable pointers (reusable packed buffer)
- `block_tables` padded to `graph_max_blocks` width (stable shape)
- cuBLAS workspace pre-allocated via `prepare_for_graph_capture()`
- cudarc event tracking disabled (prevents cross-phase dependencies)
- Non-default stream (stream 0 doesn't support cuStreamBeginCapture)
- Warmup run before capture (lazy kernel compilation must complete first)
- Stream sync between warmup and capture (drain async allocs/frees)

---

## 8. File Map

```
rvllm-server/src/main.rs           HTTP server, routes to engine
rvllm-engine/src/async_gpu_engine.rs  Async wrapper, GPU thread spawn
rvllm-engine/src/gpu_engine.rs      Scheduler + worker orchestration
rvllm-worker/src/gpu_worker.rs      Weight loading, runner construction, execute()
rvllm-worker/src/graph_runner.rs    CUDA graph pool, capture/replay dispatch
rvllm-model-runner/src/gpu_runner.rs  Forward pass orchestrator (embed->layers->norm->lm_head)
rvllm-model-runner/src/gpu_layer.rs   Single transformer layer (all kernels)
rvllm-model-loader/src/gpu_loader.rs  Safetensors -> f16 GPU upload
rvllm-model-loader/src/gpu_weights.rs f16 weight container
rvllm-gpu/src/cublas.rs             cuBLAS/cublasLt wrappers (hgemm, sgemm, gemv)
rvllm-gpu/src/cuda_graph.rs         CudaGraph/CudaGraphPool types
rvllm-gpu/src/kernel_loader.rs      PTX loading and kernel function lookup
rvllm-kv-cache/src/engine_cuda.rs   f16 paged KV cache allocation
kernels/*.cu                        All CUDA kernels (f16 I/O, f32 internal math)
```

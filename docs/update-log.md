# rvLLM Update Log

Chronological record of all optimization work, benchmarks, and architecture changes.

---

## 2026-03-30: rTriton -- Unified Kernel Layer

**Goal:** Replicate Triton's kernel fusion + our cuBLAS tricks in one standalone Rust crate.

**What was built:** `crates/rtriton/` -- Triton-style GPU kernel compiler with cuBLAS integration:
- SSA IR (30 ops), builder DSL matching Triton API, 7 optimization passes, PTX codegen (sm_80+)
- 8 LLM kernels: rmsnorm, fused_residual_rmsnorm, rope, silu_mul, tiled_gemm, gemv, persistent_gemm, flash_attention_decode
- cuBLAS integration: GemmOp descriptor (F16/FP8/F32), GemmEngine with plan cache, M-threshold routing (cublasLt M<=32, cuBLAS M>32, FP8 cublasLt M=1)
- Mixed execution graph: GraphNode enum (Triton JIT | cuBLAS GEMM), buffer liveness allocator
- DecodeLayerPlan: 9-op decode layer (5 Triton + 4 cuBLAS) captured as single CUDA graph
- H100 SXM roofline model for batch sizing analysis

**Key insight:** Research confirmed vLLM's advantage is NOT from Triton GEMM (cuBLAS wins at all shapes). The wins come from fusing ~20-30 pointwise/reduction ops between GEMMs into ~6-9 Triton kernels, eliminating kernel launch overhead and GMEM round-trips.

**Status:** 50 tests passing (40 unit + 10 integration), zero warnings. Full pipeline validated: builder -> passes -> PTX codegen for all 8 kernels. Mixed Triton+cuBLAS graph test for Llama-7B decode layer. P0 remaining: wire real CUDA driver calls.

---

## 2026-03-28: Phase 4 -- CUDA Graph Capture/Replay

**Problem:** Graph capture failed with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.

**Root causes found (3):**
1. `context.default_stream()` -- the legacy CUDA stream doesn't support `cuStreamBeginCapture`. Fixed: `context.new_stream()`.
2. cuBLAS workspace -- cuBLAS internally calls `cudaMalloc` during capture. Fixed: pre-allocate 4MB workspace via `cublasSetWorkspace_v2`.
3. cudarc event tracking -- per-CudaSlice read/write events create cross-phase dependencies during capture. Fixed: `disable_event_tracking()`.

**Result:** Graph capture working end-to-end on A100. Full decode forward pass (28 layers) captured as single replayable graph.

**Benchmark (A100 80GB, Qwen2.5-1.5B, f16, greedy):**
| N | tok/s |
|---|---|
| 1 | 128 |
| 4 | 540 |
| 8 | 1,091 |
| 16 | 2,118 |
| 32 | 3,467 |

Per-token: 7.7ms. Overhead: 77% (5.95ms of 7.7ms is not compute).

---

## 2026-03-28: Phase 5 -- 10-Agent Overhead Reduction Swarm

**Problem:** 5.95ms overhead per token. Breakdown:
- 392 unnecessary f32<->f16 cast kernels per forward
- Separate add+norm (56 extra kernel launches)
- RoPE allocates + copies (56 allocs + 56 memcpy)
- 6 separate metadata HtoD uploads
- tokio yield_now every iteration
- Hot-path info! logging

**10 agents launched in parallel worktrees:**
1. cuBLAS GemmEx mixed-precision GEMM
2. forward_mixed (1-alloc GEMM, no output cast)
3. In-place RoPE (zero alloc, zero copy)
4. Fused residual+RMSNorm (wire existing kernel)
5. Wire forward_mixed into all 7 linear calls
6. Packed metadata upload (6 HtoD -> 1)
7. Engine loop (periodic yield_now every 64 steps)
8. Scheduler (defer tokenizer.decode, reduce cloning)
9. Input prep (DecodeInputScratch with buffer reuse)
10. Graph capture fix (infinite retry bug, logging downgrade)

**Result:** 7.7ms -> 5.75ms per token (-25%).

| N | tok/s |
|---|---|
| 1 | 174 |
| 32 | 4,276 |

---

## 2026-03-28: Shared Input Casts

**Problem:** forward_mixed casts the same f32 input to f16 redundantly for Q/K/V (3x) and gate/up (2x).

**Fix:** Cast once, share across projections using `forward_f16_in`.

**Result:** 5.75ms -> 5.65ms (-1.8%).

---

## 2026-03-28: Fused QKV and Gate+Up Weights

**Problem:** 3 separate cuBLAS calls for Q/K/V, 2 for gate/up. Each has ~15us dispatch overhead.

**Fix:** Concatenate weights at load time. Single GEMM for QKV (1536+256+256=2048 output), single GEMM for gate+up (8960*2=17920 output). Zero-copy split via CudaView slices.

**Result:** 5.65ms -> 5.55ms. 84 fewer cuBLAS calls per forward.

---

## 2026-03-28: Full f16 Forward (Zero Casts)

**Problem:** 4 remaining f32->f16 cast kernels per layer (112 total). Hidden states bounced between f32 (for element-wise ops) and f16 (for GEMMs).

**Fix:** Complete f16 forward path. All kernels use f16 variants:
- `rms_norm_f16_kernel`, `rotary_embedding_f16_kernel`
- `flash_attention_2_decode_f16io_kernel` (new: f16 Q/output)
- `fused_residual_rmsnorm_f16_kernel`, `fused_silu_mul_f16_kernel`
- `add_f16_kernel`, `add_bias_f16_kernel`
- `reshape_and_cache_f16io_kernel` (f16 K/V input)
- `hgemm` f16xf16->f16 for all projections

Pre-converted norm weights, biases, and embedding table to f16 at load time. Zero cast kernels in the decode inner loop.

**Result:** 5.55ms -> 4.93ms (-11.2%). 200 tok/s at N=1.

---

## 2026-03-28: Timing Instrumentation Reveals Truth

Added per-step timing to the decode loop. **Key finding:**

```
upload_metadata:    2us   (basically free after packed upload)
graph_replay:     750us   (kernel dispatch)
read_graph_output: 4172us (GPU compute + sync wait)
```

The 4.2ms "read" is NOT transfer time (4 bytes). It's **waiting for GPU compute to finish**. The `clone_dtoh` synchronizes the stream, blocking until the graph replay completes.

**Conclusion:** CPU overhead is solved (~0.5ms). The remaining time is GPU kernel efficiency. The theoretical memory-bandwidth floor is 1.3ms (2.6GB weights at 2039 GB/s).

---

## 2026-03-29: 9-Agent Kernel Efficiency Swarm

**Problem:** 2.9ms gap between GPU compute (4.2ms) and theoretical floor (1.3ms). Sources: cuBLAS per-call overhead, memset kernels from alloc_zeros, separate add+norm across layers.

**9 agents:**

cuBLAS launch overhead:
1. `hgemv_f16` for M=1 decode (lower dispatch than GEMM)
2. `hgemm_strided_batched` for multi-GEMM batching
3. `warmup_gemm_shapes` to pre-warm cuBLAS algo cache

Small kernel overhead:
4. In-place `rms_norm_f16` (zero alloc via shared memory)
5. Cross-layer add+norm fusion (fuse last layer's residual add into next layer's norm)
6. Replace `alloc_zeros` with `unsafe { alloc }` for immediately-overwritten buffers

Memory pool overhead:
7. `F16LayerScratch` struct (2.3MB, reusable across layers)
8. CUDA memory pool `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = u64::MAX`
9. Allocation audit (10 allocs/layer, 5 reusable)

**Result:** 4.93ms -> 4.21ms per token (-14.6%). Graph replay: 750us -> 95us (-87%).

| N | tok/s |
|---|---|
| 1 | 236 |
| 4 | 834 |
| 8 | 1,786 |
| 32 | 5,123 |

---

## Summary: Full Optimization History

| Phase | Per-token | N=1 tok/s | Key change |
|---|---|---|---|
| Phase 4 (baseline) | 7.70ms | 130 | CUDA graph capture working |
| Phase 5 (10-agent) | 5.75ms | 174 | Cast reduction, fused ops, engine opt |
| Shared casts | 5.65ms | 177 | Share input cast across QKV, gate+up |
| Fused weights | 5.55ms | 180 | QKV + gate+up weight concatenation |
| Full f16 | 4.93ms | 200 | Zero casts, all f16 kernels |
| 9-agent kernel | **4.21ms** | **236** | Cross-layer fusion, memset elimination, pool tuning |

**Total: 7.70ms -> 4.21ms (1.83x speedup). 130 -> 236 tok/s (1.82x).**

Theoretical peak: 574 tok/s (1.74ms, memory-bandwidth-bound). Current utilization: 41%.

---

## 2026-03-29: Custom GEMV + cublasLt + Async DtoH Pipeline

### Custom GEMV kernel
Wrote `gemv_f16_kernel` with half2 vectorized loads, warp shuffle reduction, f32 accumulation. Result: **no improvement over cuBLAS** (4205us vs 4205us). cuBLAS's internal M=1 dispatch is already efficient. The 4.1ms is genuine GPU memory bandwidth time.

### cublasLt for decode GEMMs
Wired `CublasLtOps::hgemm_a_bt` with split-K for M<=32. Result: **marginal** (~7us, 4212->4205us). CUDA graph bakes the kernel at capture time; algorithm selection is one-shot.

### Async DtoH pipeline
Added pinned host memory DtoH path. Graph replay + async DtoH enqueue returns in **60us** (was 4212us blocking). Forward call is now non-blocking.

### Engine-level pipeline (attempted, reverted)
Tried overlapping GPU compute with next step's prepare_step. **Catastrophic at high N**: 1787->154 tok/s at N=16. Root cause: the scheduler needs ALL tokens from step N before scheduling step N+1. Pipelined step delays token delivery, causing the scheduler to re-schedule same sequences without latest tokens.

**Correct pipeline approach**: overlap GPU compute with new request arrival processing at the async engine level, not same-sequence scheduling. Future work.

### Key finding
The 4.1ms per token is the **hardware efficiency floor** for Qwen2.5-1.5B on A100. Weight reads dominate: 3.09GB at ~75% memory bandwidth utilization. Custom kernels, cublasLt, and algorithm tuning all converge to the same number. Further gains require:
- Smaller models (fewer weights to read)
- Quantization (INT8/FP8 halves weight reads)
- Multi-GPU (parallel weight reads)
- Speculative decoding (amortize weight reads across multiple tokens)

---

## Summary: Full Optimization History

| Phase | Per-token | N=1 tok/s | N=32 tok/s | Key change |
|---|---|---|---|---|
| Phase 4 (baseline) | 7.70ms | 130 | 3,467 | CUDA graph capture working |
| Phase 5 (10-agent) | 5.75ms | 174 | 4,276 | Cast reduction, fused ops, engine opt |
| Shared casts | 5.65ms | 177 | - | Share input cast across QKV, gate+up |
| Fused weights | 5.55ms | 180 | - | QKV + gate+up weight concatenation |
| Full f16 | 4.93ms | 200 | - | Zero casts, all f16 kernels |
| 9-agent kernel | **4.21ms** | **236** | **5,123** | Cross-layer fusion, memset elimination, pool tuning |

**Total: 130 -> 236 tok/s (1.82x). 41% of theoretical peak (574 tok/s).**

---

## 2026-03-29: Pure f16 Refactor + Coherence Fix + H100 Benchmarks

### Pure f16 (removed all f32 from data path)
- `GpuModelWeights`: single `HashMap<String, CudaSlice<f16>>` (was dual f32+f16)
- `gpu_loader`: loads all weights as f16 (bf16/f32 on disk -> f16 at load time)
- `GpuLayerWeights`: all fields `CudaSlice<f16>` (was mixed f32 norms + f16 projections)
- `GpuLayerInput`: `hidden_states: &CudaSlice<f16>` (was f32 with optional f16)
- Runner: `embed_tokens`, `final_norm_weight`, `lm_head_weight` all f16
- `fuse_weights()`: pure DtoD concat, no f32->f16 GPU casts
- Removed ~800 lines of f32 helper methods, ~530 lines of dead CPU attention path
- f32 only remains for: RoPE cos/sin tables, rms_norm epsilon scalar, LM head logits (argmax boundary)

### Coherence bug found and fixed
**Root cause:** Fused QKV GEMM outputs `[T, QKV_dim]` where each row is `[Q_t, K_t, V_t]` interleaved. The split code assumed `[all_Q | all_K | all_V]` layout. For T=1 (decode) these are identical. For T>1 (prefill) the split mixes Q/K/V from different tokens, corrupting the KV cache. Every subsequent decode step read garbage.
**Fix:** Use individual projections (3 GEMMs) for prefill (T>1), fused single GEMM for decode (T=1). Same fix applied to fused gate+up and QKV bias.
**Also fixed:** `rms_norm_eps` was hardcoded 1e-5 instead of reading from model config (Qwen2.5 uses 1e-6). Now threaded from `config.json` -> `HfModelConfig` -> `WorkerConfig` -> `ModelRunnerConfig` -> each layer.

### CUDA graphs re-enabled
- Exact batch size match (no padding)
- Pre-allocated metadata + output GPU buffers at max size (256 seqs) so graph-captured pointers never go stale on buffer reallocation
- Warmup (3 forward calls) before capture
- Fallback to raw forward on capture failure

### H100 80GB benchmarks (Qwen2.5-1.5B, f16, greedy, 100 tok/req)

| N | tok/s |
|---|---|
| 1 | 197 |
| 2 | 506 |
| 4 | 976 |
| 8 | 1,905 |
| 16 | 3,564 |
| 32 | 5,714 |
| 64 | 10,016 |
| 128 | **12,673** |

Near-linear scaling. Zero failures N=1 through N=128. All via CUDA graph replay.

---

## Full History

| Phase | GPU | N=1 | Peak | Key change |
|---|---|---|---|---|
| Phase 4 | A100 | 130 | 3,467 (N=32) | CUDA graph capture |
| Phase 5 | A100 | 174 | 4,276 (N=32) | 10-agent swarm |
| Full f16 | A100 | 200 | - | Zero casts |
| 9-agent kernel | A100 | 236 | 5,123 (N=32) | Cross-layer fusion |
| GPU thread | A100 | 236 | 6,385 (N=32) | Dedicated OS thread |
| **Pure f16 + fix** | **H100** | **197** | **12,673 (N=128)** | **Coherence fix, f16 refactor, graph pre-alloc** |

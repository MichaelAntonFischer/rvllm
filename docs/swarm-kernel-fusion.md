# Swarm: Kernel Fusion -- Close the vLLM Gap

## Status Quo

- **3,034 tok/s** single-seq decode after CUDA graph + cublasLt fixes (commit 029f116ac)
- **0.67x vLLM** (vLLM: ~4,530 tok/s equivalent). Gap narrowed from 4.2x to 1.5x.
- Remaining bottleneck: **~224 kernel launches per decode step** with intermediate HBM round-trips between every kernel

## Architecture Overview

The T=1 decode path in `gpu_layer.rs:175-357` already has dispatch code for 4 fused "cute" kernels:

| Kernel Function | Module Name | Fallback Path |
|---|---|---|
| `fused_cute_add_norm_qkv_gemv` | `fused_add_norm_qkv` | fused_residual_rmsnorm + hgemm_dispatch |
| `fused_cute_norm_qkv_gemv` | `fused_norm_qkv` | rms_norm_f16 + hgemm_dispatch |
| `fused_cute_add_norm_gateup_gemv` | `fused_add_norm_gateup` | fused_residual_rmsnorm + hgemm_dispatch |
| `fused_cute_silu_down_gemv` | `fused_silu_down` | fused_silu_mul_f16_split + hgemm_dispatch |

**These kernels don't exist yet.** The `loader.get_func()` calls fail silently and every layer falls through to the unfused path. The dispatch code, launch configs, argument orders, and shared memory calculations are already wired in `gpu_layer.rs`. The agents just need to write the .cu files.

## What Exists Already

- `codegen.rs` in `rvllm-fusion` has CUDA C generators for all 4 patterns (RMSNormGemv, SiLUElemMulGemv, ElemAddRMSNorm, ElemAddRMSNormGemv) -- these are reference implementations
- `fused_norm_gemv.cu` -- RMSNorm + single GEMV (close but different function name/signature)
- `fused_silu_down.cu` -- SiLU*up + Down GEMV (close but different function name/signature)
- `fused_residual_rmsnorm_f16.cu` -- Add + RMSNorm (working, actively used)
- `fused_rope_cache.cu` -- RoPE + KV cache write (working for M=1)
- `flash_attention_3.cu` -- FA3 decode, 256 threads, f16 I/O (working)

## Kernel Count: Current vs Target

| | Current (fallback) | Target (all fused) |
|---|---|---|
| Kernels per layer | 8-10 | 5 |
| Total 28 layers | 224-280 | 140 |
| + outer (embed, lm_head) | ~230-286 | ~146 |
| Allocs per layer | ~6 | ~3 |

With fusions: 5 kernels/layer = (1) add+norm+QKV_GEMV, (2) bias, (3) RoPE+cache, (4) attention, (5) add+norm+gateup + silu+down (pipelined as 2 kernels but could be 1 if we fuse attention output with MLP).

## Target

Match vLLM single-seq: **~4,500 tok/s** (1.0x parity)
Stretch goal: **~6,000 tok/s** (1.3x vLLM)

---

## Agent Assignments

### Agent 1: fused_cute_add_norm_qkv_gemv kernel

**Branch:** `fusion/add-norm-qkv`
**Files:** `kernels/fused_add_norm_qkv_gemv.cu`

Write the 3-way fused kernel: residual_add + RMSNorm + GEMV producing QKV output.

**Signature** (must match gpu_layer.rs:192-197):
```c
extern "C" __global__ void fused_cute_add_norm_qkv_gemv(
    __half* output,          // [qkv_dim] -- interleaved QKV
    __half* residual_out,    // [hidden_size] -- residual for next layer
    const __half* input,     // [hidden_size] -- previous layer residual
    const __half* add_vec,   // [hidden_size] -- previous MLP output
    const __half* norm_weight, // [hidden_size]
    const __half* proj_weight, // [qkv_dim, hidden_size] -- fused QKV weight
    float eps,
    int hidden_size,
    int qkv_dim
)
```

**Launch config** (from gpu_layer.rs):
- Grid: `((qkv_dim + rpb - 1) / rpb, 1, 1)` where rpb=8
- Block: `(256, 1, 1)`
- Shared mem: `hidden_size * 4 + 8 * 4` bytes

**Algorithm:**
1. Each block handles `rpb` (8) output elements
2. Phase 1: All blocks redundantly compute residual_add + RMSNorm into shared memory (hidden_size floats)
3. Block 0 writes residual_out
4. Phase 2: Each block computes 8 dot products against 8 rows of proj_weight
5. Use half2 vectorized loads for weight rows
6. Warp-shuffle reduction for dot products

**Reference:** `codegen.rs:emit_elemadd_rmsnorm_gemv()` has the pattern. Adapt for rpb=8 (multiple outputs per block).

**Qwen2.5-1.5B dimensions:** hidden=1536, qkv_dim=2048 (Q=1536 + K=256 + V=256)

### Agent 2: fused_cute_norm_qkv_gemv kernel

**Branch:** `fusion/norm-qkv`
**Files:** Same file as Agent 1 OR `kernels/fused_norm_qkv_gemv.cu`

First-layer variant: RMSNorm + GEMV (no residual add). Same signature minus `add_vec` and `residual_out`.

**Signature** (from gpu_layer.rs:220-224):
```c
extern "C" __global__ void fused_cute_norm_qkv_gemv(
    __half* output,            // [qkv_dim]
    const __half* input,       // [hidden_size]
    const __half* norm_weight, // [hidden_size]
    const __half* proj_weight, // [qkv_dim, hidden_size]
    float eps,
    int hidden_size,
    int qkv_dim
)
```

**Launch config:** Same as Agent 1.

This is simpler -- skip the add step. Can share 90% of the code with Agent 1.

### Agent 3: fused_cute_add_norm_gateup_gemv kernel

**Branch:** `fusion/add-norm-gateup`
**Files:** `kernels/fused_add_norm_gateup_gemv.cu`

Same pattern as Agent 1 but for MLP: residual_add + RMSNorm + GEMV producing gate_up output.

**Signature** (from gpu_layer.rs:310-315):
```c
extern "C" __global__ void fused_cute_add_norm_gateup_gemv(
    __half* output,            // [gate_up_dim] = [2 * intermediate]
    __half* residual_out,      // [hidden_size]
    const __half* input,       // [hidden_size] -- residual
    const __half* add_vec,     // [hidden_size] -- O projection output
    const __half* norm_weight, // [hidden_size]
    const __half* proj_weight, // [gate_up_dim, hidden_size] -- fused gate+up weight
    float eps,
    int hidden_size,
    int gate_up_dim
)
```

**Launch config:** Same pattern, grid based on gate_up_dim.

**Qwen2.5-1.5B:** hidden=1536, intermediate=8960, gate_up_dim=17920. That's 17920/8 = 2240 blocks. Fine on A100 (108 SMs, waves of ~21).

### Agent 4: fused_cute_silu_down_gemv kernel

**Branch:** `fusion/silu-down`
**Files:** `kernels/fused_silu_down_gemv.cu`

Fused SiLU(gate)*up + Down projection GEMV.

**Signature** (from gpu_layer.rs:325-329):
```c
extern "C" __global__ void fused_cute_silu_down_gemv(
    __half* output,        // [hidden_size]
    const __half* gate,    // [intermediate_size] (slice of gate_up)
    const __half* up,      // [intermediate_size] (slice of gate_up)
    const __half* weight,  // [hidden_size, intermediate_size] -- down_proj
    int hidden_size,
    int intermediate_size
)
```

**Launch config:**
- Grid: `((hidden_size + 7) / 8, 1, 1)`
- Block: `(256, 1, 1)`
- Shared mem: `8 * 4` bytes

**Algorithm:**
1. Each block computes 8 output elements
2. For each output row: dot(silu(gate) * up, weight_row)
3. SiLU computed inline in registers -- no intermediate buffer
4. Half2 vectorized loads for gate, up, weight

**Reference:** `codegen.rs:emit_silu_elemmul_gemv()` and existing `fused_silu_down.cu`

### Agent 5: Kernel registration + build

**Branch:** `fusion/register`
**Files:** `crates/rvllm-gpu/src/kernel_loader.rs`, build scripts

Register all 4 new kernel files in KERNEL_FUNCTIONS:
```rust
("fused_add_norm_qkv_gemv", &["fused_cute_add_norm_qkv_gemv"]),
("fused_norm_qkv_gemv", &["fused_cute_norm_qkv_gemv"]),
("fused_add_norm_gateup_gemv", &["fused_cute_add_norm_gateup_gemv"]),
// fused_silu_down already registered but wrong function name -- update:
("fused_silu_down", &["fused_silu_down_f16_kernel", "fused_silu_down_bias_f16_kernel", "fused_cute_silu_down_gemv"]),
```

Also check that gpu_layer.rs module names match:
- `loader.get_func("fused_add_norm_qkv", ...)` -- module is `"fused_add_norm_qkv"` but we're registering `"fused_add_norm_qkv_gemv"`. Fix the module name in either kernel_loader.rs or gpu_layer.rs to match.

Verify the PTX compilation: add the new .cu files to Makefile/build.rs kernel compilation list.

### Agent 6: GQA-optimized FA3 decode attention

**Branch:** `fusion/gqa-attention`
**Files:** `kernels/flash_attention_3.cu`

Current FA3 decode kernel loads KV cache independently for each of the 12 query heads. Qwen2.5-1.5B has 12 query heads sharing 2 KV heads (6:1 GQA ratio). The KV data is loaded 6x redundantly.

**Optimization:**
- Change grid from `(num_seqs, num_heads)` to `(num_seqs, num_kv_heads)`
- Each block processes ALL 6 query heads that share one KV head
- Load KV tile once into shared memory
- Compute 6 independent QK dot products + softmax + PV accumulations
- Write 6 output rows

**Expected gain:** ~3-5x reduction in KV cache bandwidth for attention phase. At our 512-token benchmark standard, this is already significant. At 2048+ context, even more so.

**Risk:** Higher register pressure (6 query accumulators). May need to reduce BC (tile size) or use register spilling. Profile both approaches.

### Agent 7: Prefill path kernel count reduction

**Branch:** `fusion/prefill-opt`
**Files:** `crates/rvllm-model-runner/src/gpu_layer.rs` (lines 361-537)

The prefill path (T>1) still uses separate kernels for everything. Optimize:

1. **Wire existing fused_residual_rmsnorm_f16 into prefill path** -- currently only used in decode fallback
2. **Fused QKV for N>64** -- current code does 3 separate GEMMs for N>64. Use a single fused GEMM to [N, qkv_dim] then slice (the weight already exists as `weights.fused_qkv`)
3. **Use silu_mul_interleaved** for all N (currently only N<=64)
4. **Eliminate redundant allocs** -- reuse F16LayerScratch buffers in prefill too

Target: reduce prefill from 12-14 kernels/layer to 8-9 kernels/layer.

### Agent 8: Integration validation + benchmark harness

**Branch:** `fusion/integration`
**Files:** `bench/`, test scripts

After all agents merge:

1. **Coherence test:** Run model with all fusions enabled, verify output matches unfused baseline
   - Generate 100 tokens with temperature=0
   - Compare token-for-token with baseline (no fused kernels loaded)
   - Any mismatch = bug in a fused kernel

2. **Per-kernel benchmark:** Time each fused kernel vs its unfused equivalent
   - fused_cute_add_norm_qkv_gemv vs (fused_residual_rmsnorm + hgemm)
   - fused_cute_silu_down_gemv vs (silu_mul + hgemm)
   - Measure: kernel time (nsec), memory bandwidth (GB/s), arithmetic intensity

3. **End-to-end benchmark:** Full decode throughput at N=1,4,8,16,32
   - Compare with/without fusion kernels
   - Compare with vLLM 0.18 on same hardware

4. **CUDA graph re-capture test:** Verify graphs capture correctly with new kernel sequence
   - The graph captures a different kernel sequence now (fewer launches)
   - Pre-capture at startup must work with fused path

## Merge Order

```
Phase 1 (parallel, no conflicts):
  Agent 1 + Agent 2  (new .cu files)
  Agent 3            (new .cu file)
  Agent 4            (new .cu file)
  Agent 5            (kernel_loader.rs registration)

Phase 2 (after Phase 1):
  Agent 6            (modifies flash_attention_3.cu)
  Agent 7            (modifies gpu_layer.rs prefill path)

Phase 3 (after all):
  Agent 8            (validation + benchmarks)
```

## Key Dimensions (Qwen2.5-1.5B)

| Parameter | Value |
|---|---|
| hidden_size | 1536 |
| intermediate_size | 8960 |
| num_heads | 12 |
| num_kv_heads | 2 |
| head_dim | 128 |
| q_dim | 1536 (12 * 128) |
| kv_dim | 256 (2 * 128) |
| qkv_dim | 2048 (1536 + 256 + 256) |
| gate_up_dim | 17920 (2 * 8960) |
| num_layers | 28 |

## Shared Memory Budget

A100: 164KB configurable shared memory per SM (up to 99KB per block with opt-in).

For hidden=1536:
- `hidden_size * 4 + 8 * 4` = 6176 bytes (well under 48KB default limit)
- No need for `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` override

For larger models (hidden=4096+):
- `4096 * 4 + 8 * 4` = 16416 bytes (still under 48KB)
- Only models with hidden >= 12288 would need the override (which gpu_layer.rs already handles)

## Verification Command

```bash
# On A100/H100:
make kernels CUDA_ARCH=sm_80  # or sm_90 for H100
cargo build --release --features cuda

# Coherence check
curl localhost:8080/v1/completions -d '{"model":"Qwen2.5-1.5B","prompt":"The capital of France is","max_tokens":20,"temperature":0}'

# Benchmark
./target/release/rvllm benchmark --model Qwen2.5-1.5B --n "1,4,8,16,32"
```

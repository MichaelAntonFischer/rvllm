// SM-DAG persistent decode kernel for M=1 decode.
// Executes an ENTIRE transformer layer in ONE kernel launch using
// atomic point-to-point synchronization between phases (no grid.sync()).
// Improvements over original persistent_layer_decode:
//   - Atomic sync: ~3-5x lower sync latency than grid.sync()
//   - Parallelized RoPE: all blocks participate (vs block 0 only)
//   - L2 weight prefetch: idle blocks prefetch O-proj weights during attention
//
// Requires: cooperative launch (cuLaunchCooperativeKernel) for co-residency
// Compile: nvcc -cubin -arch=sm_90 -O3 --use_fast_math -rdc=true
//
// Grid: (256, 1, 1) -- all blocks must be co-resident
// Block: (256, 1, 1)
// Shared: max(hidden_size * 4 + 32, FA_BC * head_dim * 4 + MAX_HPG * FA_BC * 4 + 32)
//
// Phases:
//   1: Add + RMSNorm + QKV GEMV
//   2: Bias + RoPE + KV cache write (parallelized across blocks)
//   3: GQA flash attention decode (idle blocks prefetch O-proj into L2)
//   4: O-proj GEMV
//   5: Add + RMSNorm + GateUp GEMV
//   6: SiLU + Down GEMV

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <float.h>

// Atomic point-to-point sync (replaces grid.sync())
// Each phase signals completion via atomicAdd on a counter.
// Consumer phases spin-wait until the counter reaches the expected value.
// ~3-5x lower latency than grid.sync() on H100.
__device__ __forceinline__ void dag_signal(volatile int* flags, int phase) {
    __threadfence();
    if (threadIdx.x == 0) atomicAdd((int*)&flags[phase], 1);
}

__device__ __forceinline__ void dag_wait(volatile int* flags, int phase, int expected) {
    if (threadIdx.x == 0) {
        while (atomicAdd((int*)&flags[phase], 0) < expected) {}
    }
    __syncthreads();
}

#define DAG_SYNC_QKV    0  // Phase 1 -> Phase 2
#define DAG_SYNC_ROPE   1  // Phase 2 -> Phase 3
#define DAG_SYNC_ATTN   2  // Phase 3 -> Phase 4
#define DAG_SYNC_OPROJ  3  // Phase 4 -> Phase 5
#define DAG_SYNC_LOAD5  4  // Phase 5 race avoidance
#define DAG_SYNC_GATEUP 5  // Phase 5 -> Phase 6
#define DAG_NUM_SYNCS   6

#define PLD_THREADS 256
#define PLD_WARPS 8
#define PLD_RPB 8
#define PLD_FA_BC 64
#define PLD_FA_MAX_HPG 8

// ---- Device helpers ----

__device__ __forceinline__ float pld_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float pld_warp_xor_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float pld_silu(float x) {
    return x / (1.0f + expf(-x));
}

// ---- Shared subroutines ----

// Load prev_residual + prev_mlp (or just prev_residual for layer 0) into s_normed,
// compute RMSNorm, write residual_out from block 0, apply norm weights.
// Returns rms_scale in s_scratch[0].
__device__ void pld_add_rmsnorm(
    float* s_normed, float* s_scratch,
    __half* __restrict__ residual_out,
    const __half* __restrict__ prev_residual,
    const __half* __restrict__ prev_mlp,  // NULL for layer 0
    const __half* __restrict__ norm_w,
    float eps, int hidden_size
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int h2 = hidden_size / 2;

    float local_ss = 0.0f;

    if (prev_mlp != nullptr) {
        const half2* r2 = (const half2*)prev_residual;
        const half2* m2 = (const half2*)prev_mlp;
        for (int i = tid; i < h2; i += PLD_THREADS) {
            half2 a = r2[i];
            half2 b = m2[i];
            float v0 = __half2float(a.x) + __half2float(b.x);
            float v1 = __half2float(a.y) + __half2float(b.y);
            s_normed[i * 2]     = v0;
            s_normed[i * 2 + 1] = v1;
            local_ss += v0 * v0 + v1 * v1;
        }
        if ((hidden_size & 1) && tid == 0) {
            float v = __half2float(prev_residual[hidden_size - 1]) + __half2float(prev_mlp[hidden_size - 1]);
            s_normed[hidden_size - 1] = v;
            local_ss += v * v;
        }
    } else {
        const half2* r2 = (const half2*)prev_residual;
        for (int i = tid; i < h2; i += PLD_THREADS) {
            half2 a = r2[i];
            float v0 = __half2float(a.x);
            float v1 = __half2float(a.y);
            s_normed[i * 2]     = v0;
            s_normed[i * 2 + 1] = v1;
            local_ss += v0 * v0 + v1 * v1;
        }
        if ((hidden_size & 1) && tid == 0) {
            float v = __half2float(prev_residual[hidden_size - 1]);
            s_normed[hidden_size - 1] = v;
            local_ss += v * v;
        }
    }

    // Warp reduction
    local_ss = pld_warp_reduce_sum(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < PLD_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = pld_warp_reduce_sum(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    // Block 0 writes pre-norm residual
    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += PLD_THREADS)
            residual_out[i] = __float2half(s_normed[i]);
    }

    // Apply norm weights
    for (int i = tid; i < hidden_size; i += PLD_THREADS)
        s_normed[i] = s_normed[i] * __half2float(norm_w[i]) * rms_scale;
    __syncthreads();
}

// GEMV with grid-stride loop: each block computes RPB rows per iteration,
// loops over all assigned rows. Output written to global memory.
__device__ void pld_gemv_grid_stride(
    __half* __restrict__ output,
    const float* s_normed,
    float* s_scratch,
    const __half* __restrict__ weight,  // [out_dim, hidden_size]
    int hidden_size, int out_dim
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int h2 = hidden_size / 2;

    for (int base = blockIdx.x * PLD_RPB; base < out_dim; base += gridDim.x * PLD_RPB) {
        int rows_here = min(PLD_RPB, out_dim - base);

        for (int r = 0; r < rows_here; r++) {
            int row = base + r;
            const half2* w2 = (const half2*)(weight + (long long)row * hidden_size);
            float acc = 0.0f;
            for (int i = tid; i < h2; i += PLD_THREADS) {
                half2 w = w2[i];
                acc += __half2float(w.x) * s_normed[i * 2];
                acc += __half2float(w.y) * s_normed[i * 2 + 1];
            }
            if ((hidden_size & 1) && tid == 0) {
                const __half* w_row = weight + (long long)row * hidden_size;
                acc += __half2float(w_row[hidden_size - 1]) * s_normed[hidden_size - 1];
            }

            acc = pld_warp_reduce_sum(acc);
            if (lane_id == 0) s_scratch[warp_id] = acc;
            __syncthreads();

            if (warp_id == 0) {
                float val = (lane_id < PLD_WARPS) ? s_scratch[lane_id] : 0.0f;
                val = pld_warp_reduce_sum(val);
                if (lane_id == 0) output[row] = __float2half(val);
            }
            if (r + 1 < rows_here) __syncthreads();
        }
    }
}

// SiLU-gated down-projection GEMV with grid-stride loop.
// Each warp handles one row, 8 rows per block iteration.
__device__ void pld_silu_down_gemv_grid_stride(
    __half* __restrict__ output,
    const __half* __restrict__ gateup,   // [gate_up_dim] interleaved [gate|up]
    const __half* __restrict__ weight,   // [hidden_size, intermediate_size]
    int hidden_size, int intermediate_size
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const __half* gate = gateup;
    const __half* up = gateup + intermediate_size;

    for (int base = blockIdx.x * PLD_RPB; base < hidden_size; base += gridDim.x * PLD_RPB) {
        int row = base + warp_id;
        if (row >= hidden_size) continue;

        const half2* w2 = (const half2*)(weight + (long long)row * intermediate_size);
        const half2* g2 = (const half2*)gate;
        const half2* u2 = (const half2*)up;
        const int k2 = intermediate_size / 2;
        float acc = 0.0f;

        for (int i = lane_id; i < k2; i += 32) {
            half2 g = g2[i];
            half2 u = u2[i];
            half2 w = w2[i];
            float g0 = __half2float(g.x), g1 = __half2float(g.y);
            float u0 = __half2float(u.x), u1 = __half2float(u.y);
            float w0 = __half2float(w.x), w1 = __half2float(w.y);
            acc += pld_silu(g0) * u0 * w0;
            acc += pld_silu(g1) * u1 * w1;
        }
        if ((intermediate_size & 1) && lane_id == 0) {
            int last = intermediate_size - 1;
            float g = __half2float(gate[last]);
            acc += pld_silu(g) * __half2float(up[last]) * __half2float(weight[(long long)row * intermediate_size + last]);
        }

        acc = pld_warp_reduce_sum(acc);
        if (lane_id == 0) output[row] = __float2half(acc);
    }
}

// ============================================================================
// MAIN PERSISTENT LAYER KERNEL
// ============================================================================
extern "C" __global__ void __launch_bounds__(PLD_THREADS, 2)
persistent_layer_decode_f16(
    // Outputs
    __half* __restrict__ mlp_out,
    __half* __restrict__ residual_out,
    // Inputs
    const __half* __restrict__ prev_residual,
    const __half* __restrict__ prev_mlp,       // NULL for layer 0
    // Attention I/O
    __half* __restrict__ key_cache,
    __half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const int* __restrict__ positions,
    const int* __restrict__ slot_mapping,
    const float* __restrict__ rope_cos,        // [max_pos, head_dim/2]
    const float* __restrict__ rope_sin,
    // Weights
    const __half* __restrict__ norm_w,
    const __half* __restrict__ qkv_weight,     // [qkv_dim, hidden]
    const __half* __restrict__ qkv_bias,       // [qkv_dim] or NULL
    const __half* __restrict__ o_weight,       // [hidden, q_dim]
    const __half* __restrict__ post_norm_w,
    const __half* __restrict__ gateup_weight,  // [gate_up_dim, hidden]
    const __half* __restrict__ down_weight,    // [hidden, intermediate]
    // Scratch buffers (global, persistent across layers)
    __half* __restrict__ qkv_scratch,          // [qkv_dim]
    __half* __restrict__ attn_scratch,         // [q_dim]
    __half* __restrict__ oproj_scratch,        // [hidden]
    __half* __restrict__ gateup_scratch,       // [gate_up_dim]
    // Config
    float eps,
    float attn_scale,
    int hidden_size,
    int q_dim,
    int kv_dim,
    int qkv_dim,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int intermediate_size,
    int gate_up_dim,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq,
    int* sync_flags               // [DAG_NUM_SYNCS] atomic completion counters, zeroed before launch
) {
    const int tid = threadIdx.x;

    // Dynamic shared memory: [hidden_size] floats + [8] floats scratch
    extern __shared__ float smem[];
    float* s_hidden = smem;
    float* s_scratch = smem + hidden_size;

    // ====================================================================
    // PHASE 1: Add + RMSNorm + QKV GEMV
    // ====================================================================
    pld_add_rmsnorm(s_hidden, s_scratch, residual_out,
                    prev_residual, prev_mlp, norm_w, eps, hidden_size);

    // GEMV: normed_hidden @ qkv_weight^T -> qkv_scratch[qkv_dim]
    pld_gemv_grid_stride(qkv_scratch, s_hidden, s_scratch,
                         qkv_weight, hidden_size, qkv_dim);

    dag_signal(sync_flags, DAG_SYNC_QKV);
    dag_wait(sync_flags, DAG_SYNC_QKV, gridDim.x);

    // ====================================================================
    // PHASE 2: QKV bias + RoPE + KV cache write (parallelized across blocks)
    //   Each block handles one head: Q heads, K heads (+ cache), V heads (+ cache).
    //   Total active blocks = num_heads + 2 * num_kv_heads.
    // ====================================================================
    {
        const int half_dim = head_dim / 2;
        const int pos = positions[0];
        const int slot = slot_mapping[0];
        const int total_heads = num_heads + 2 * num_kv_heads;
        const int bid = (int)blockIdx.x;

        if (bid < num_heads) {
            // Q head: add bias + RoPE
            int h = bid;
            __half* q_head = qkv_scratch + h * head_dim;
            if (qkv_bias != nullptr) {
                const __half* bias_head = qkv_bias + h * head_dim;
                for (int i = tid; i < head_dim; i += PLD_THREADS)
                    q_head[i] = __float2half(__half2float(q_head[i]) + __half2float(bias_head[i]));
            }
            for (int i = tid; i < half_dim; i += PLD_THREADS) {
                float cos_val = rope_cos[pos * half_dim + i];
                float sin_val = rope_sin[pos * half_dim + i];
                float q0 = __half2float(q_head[2 * i]);
                float q1 = __half2float(q_head[2 * i + 1]);
                q_head[2 * i]     = __float2half(q0 * cos_val - q1 * sin_val);
                q_head[2 * i + 1] = __float2half(q0 * sin_val + q1 * cos_val);
            }
        } else if (bid < num_heads + num_kv_heads) {
            // K head: add bias + RoPE + cache write
            int h = bid - num_heads;
            __half* k_head = qkv_scratch + q_dim + h * head_dim;
            if (qkv_bias != nullptr) {
                const __half* bias_head = qkv_bias + q_dim + h * head_dim;
                for (int i = tid; i < head_dim; i += PLD_THREADS)
                    k_head[i] = __float2half(__half2float(k_head[i]) + __half2float(bias_head[i]));
            }
            for (int i = tid; i < half_dim; i += PLD_THREADS) {
                float cos_val = rope_cos[pos * half_dim + i];
                float sin_val = rope_sin[pos * half_dim + i];
                float k0 = __half2float(k_head[2 * i]);
                float k1 = __half2float(k_head[2 * i + 1]);
                float k0_rot = k0 * cos_val - k1 * sin_val;
                float k1_rot = k0 * sin_val + k1 * cos_val;
                k_head[2 * i]     = __float2half(k0_rot);
                k_head[2 * i + 1] = __float2half(k1_rot);
                if (slot >= 0) {
                    int cache_off = (slot * num_kv_heads + h) * head_dim;
                    key_cache[cache_off + 2 * i]     = __float2half(k0_rot);
                    key_cache[cache_off + 2 * i + 1] = __float2half(k1_rot);
                }
            }
        } else if (bid < total_heads) {
            // V head: add bias + cache write (no RoPE)
            int h = bid - num_heads - num_kv_heads;
            __half* v_head = qkv_scratch + q_dim + kv_dim + h * head_dim;
            if (qkv_bias != nullptr) {
                const __half* bias_head = qkv_bias + q_dim + kv_dim + h * head_dim;
                for (int i = tid; i < head_dim; i += PLD_THREADS)
                    v_head[i] = __float2half(__half2float(v_head[i]) + __half2float(bias_head[i]));
            }
            if (slot >= 0) {
                int cache_off = (slot * num_kv_heads + h) * head_dim;
                for (int i = tid; i < head_dim; i += PLD_THREADS)
                    value_cache[cache_off + i] = v_head[i];
            }
        }
        // Blocks >= total_heads: idle during RoPE phase
    }

    // All blocks signal RoPE completion (even idle ones) so all can proceed
    dag_signal(sync_flags, DAG_SYNC_ROPE);
    dag_wait(sync_flags, DAG_SYNC_ROPE, gridDim.x);

    // ====================================================================
    // PHASE 3: GQA Flash Attention Decode
    //   Blocks [0, num_kv_heads) each handle one KV head group.
    //   Other blocks idle (attention is bandwidth-bound, not compute-bound).
    // ====================================================================
    if ((int)blockIdx.x < num_kv_heads) {
        const int kv_head_idx = blockIdx.x;
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        const int context_len = context_lens[0];
        const int heads_per_group = num_heads / num_kv_heads;
        const int num_tiles = (context_len + PLD_FA_BC - 1) / PLD_FA_BC;
        const int dims_per_thread = (head_dim + PLD_THREADS - 1) / PLD_THREADS;

        // Reuse shared memory for attention:
        //   s_kv:     [BC * head_dim] floats
        //   s_scores: [MAX_HPG * BC] floats
        //   s_warp:   [WARPS] floats
        float* s_kv     = smem;
        float* s_scores = smem + PLD_FA_BC * head_dim;
        float* s_warp_a = s_scores + PLD_FA_MAX_HPG * PLD_FA_BC;

        // Per-head softmax state and accumulators
        float head_row_max[PLD_FA_MAX_HPG];
        float head_row_sum[PLD_FA_MAX_HPG];
        float head_acc[PLD_FA_MAX_HPG][4];

        for (int g = 0; g < heads_per_group && g < PLD_FA_MAX_HPG; g++) {
            head_row_max[g] = -FLT_MAX;
            head_row_sum[g] = 0.0f;
            #pragma unroll
            for (int r = 0; r < 4; r++) head_acc[g][r] = 0.0f;
        }

        float q_reg[4];

        for (int tile = 0; tile < num_tiles; tile++) {
            const int tile_start = tile * PLD_FA_BC;
            const int tile_len = min(PLD_FA_BC, context_len - tile_start);

            // Load K tile
            {
                const int total_h2 = (tile_len * head_dim) / 2;
                for (int idx = tid; idx < total_h2; idx += PLD_THREADS) {
                    int elem = idx * 2;
                    int t = elem / head_dim;
                    int d = elem % head_dim;
                    int kv_pos = tile_start + t;
                    int page_idx = kv_pos / block_size;
                    int page_off = kv_pos % block_size;
                    int phys_block = block_tables[page_idx];
                    int base = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d;
                    __half2 h2 = *reinterpret_cast<const __half2*>(&key_cache[base]);
                    s_kv[t * head_dim + d]     = __half2float(h2.x);
                    s_kv[t * head_dim + d + 1] = __half2float(h2.y);
                }
                int total_elems = tile_len * head_dim;
                if ((total_elems & 1) && tid == 0) {
                    int e = total_elems - 1;
                    int t = e / head_dim, d = e % head_dim;
                    int kv_pos = tile_start + t;
                    int pi = kv_pos / block_size, po = kv_pos % block_size;
                    int pb = block_tables[pi];
                    s_kv[t * head_dim + d] = __half2float(key_cache[((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d]);
                }
            }
            __syncthreads();

            // For each query head: QK^T + online softmax
            for (int g = 0; g < heads_per_group && g < PLD_FA_MAX_HPG; g++) {
                int head_idx = kv_head_idx * heads_per_group + g;
                float* g_scores = s_scores + g * PLD_FA_BC;

                // Load Q from qkv_scratch
                int q_base = head_idx * head_dim;
                #pragma unroll
                for (int r = 0; r < dims_per_thread && r < 4; r++) {
                    int d = tid + r * PLD_THREADS;
                    q_reg[r] = (d < head_dim) ? (__half2float(qkv_scratch[q_base + d]) * attn_scale) : 0.0f;
                }

                for (int t = 0; t < tile_len; t++) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int r = 0; r < dims_per_thread && r < 4; r++) {
                        int d = tid + r * PLD_THREADS;
                        if (d < head_dim) dot += q_reg[r] * s_kv[t * head_dim + d];
                    }
                    dot = pld_warp_xor_sum(dot);
                    if (lane_id == 0) s_warp_a[warp_id] = dot;
                    __syncthreads();
                    if (tid == 0) {
                        float total = 0.0f;
                        for (int w = 0; w < PLD_WARPS; w++) total += s_warp_a[w];
                        g_scores[t] = total;
                    }
                    __syncthreads();
                }

                // Online softmax
                float tile_max = -FLT_MAX;
                if (tid == 0) {
                    for (int t = 0; t < tile_len; t++) tile_max = fmaxf(tile_max, g_scores[t]);
                    s_warp_a[0] = tile_max;
                }
                __syncthreads();
                tile_max = s_warp_a[0];
                __syncthreads();

                float prev_max = head_row_max[g];
                float new_max = fmaxf(prev_max, tile_max);
                if (new_max > prev_max && prev_max > -FLT_MAX) {
                    float correction = expf(prev_max - new_max);
                    #pragma unroll
                    for (int r = 0; r < dims_per_thread && r < 4; r++) head_acc[g][r] *= correction;
                    head_row_sum[g] *= correction;
                }
                head_row_max[g] = new_max;

                if (tid == 0) {
                    float tsum = 0.0f;
                    for (int t = 0; t < tile_len; t++) {
                        float v = expf(g_scores[t] - new_max);
                        g_scores[t] = v;
                        tsum += v;
                    }
                    s_warp_a[0] = tsum;
                }
                __syncthreads();
                head_row_sum[g] += s_warp_a[0];
                __syncthreads();
            }

            // Load V tile (reuse s_kv)
            {
                const int total_h2 = (tile_len * head_dim) / 2;
                for (int idx = tid; idx < total_h2; idx += PLD_THREADS) {
                    int elem = idx * 2;
                    int t = elem / head_dim;
                    int d = elem % head_dim;
                    int kv_pos = tile_start + t;
                    int page_idx = kv_pos / block_size;
                    int page_off = kv_pos % block_size;
                    int phys_block = block_tables[page_idx];
                    int base = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d;
                    __half2 h2 = *reinterpret_cast<const __half2*>(&value_cache[base]);
                    s_kv[t * head_dim + d]     = __half2float(h2.x);
                    s_kv[t * head_dim + d + 1] = __half2float(h2.y);
                }
                int total_elems = tile_len * head_dim;
                if ((total_elems & 1) && tid == 0) {
                    int e = total_elems - 1;
                    int t = e / head_dim, d = e % head_dim;
                    int kv_pos = tile_start + t;
                    int pi = kv_pos / block_size, po = kv_pos % block_size;
                    int pb = block_tables[pi];
                    s_kv[t * head_dim + d] = __half2float(value_cache[((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d]);
                }
            }
            __syncthreads();

            // Accumulate P @ V
            for (int g = 0; g < heads_per_group && g < PLD_FA_MAX_HPG; g++) {
                float* g_scores = s_scores + g * PLD_FA_BC;
                #pragma unroll
                for (int r = 0; r < dims_per_thread && r < 4; r++) {
                    int d = tid + r * PLD_THREADS;
                    if (d < head_dim) {
                        float v_acc = 0.0f;
                        for (int t = 0; t < tile_len; t++)
                            v_acc += g_scores[t] * s_kv[t * head_dim + d];
                        head_acc[g][r] += v_acc;
                    }
                }
            }
            __syncthreads();
        }

        // Write attention output to attn_scratch
        for (int g = 0; g < heads_per_group && g < PLD_FA_MAX_HPG; g++) {
            int head_idx = kv_head_idx * heads_per_group + g;
            float inv = (head_row_sum[g] > 0.0f) ? (1.0f / head_row_sum[g]) : 0.0f;
            int out_base = head_idx * head_dim;
            #pragma unroll
            for (int r = 0; r < dims_per_thread && r < 4; r++) {
                int d = tid + r * PLD_THREADS;
                if (d < head_dim)
                    attn_scratch[out_base + d] = __float2half(head_acc[g][r] * inv);
            }
        }
    } else {
        // Non-attention blocks: prefetch O-proj weights into L2 cache
        // O-proj weight: [hidden_size, q_dim] f16
        // On H100: L2 is 50MB, O-proj for 7B model is ~25MB -- fits entirely
        const int total_h = hidden_size * q_dim;
        const int stride = ((int)gridDim.x - num_kv_heads) * PLD_THREADS;
        const int my_offset = ((int)blockIdx.x - num_kv_heads) * PLD_THREADS + tid;
        const __half* w_ptr = o_weight;
        volatile __half dummy;
        for (int i = my_offset; i < total_h; i += stride) {
            dummy = w_ptr[i];
        }
    }

    // All blocks signal attention completion (attention blocks after compute,
    // idle blocks after prefetch), then all wait
    dag_signal(sync_flags, DAG_SYNC_ATTN);
    dag_wait(sync_flags, DAG_SYNC_ATTN, gridDim.x);

    // ====================================================================
    // PHASE 4: O-proj GEMV: attn_scratch[q_dim] @ o_weight -> oproj_scratch[hidden]
    // ====================================================================
    {
        // Load attn_scratch into shared memory for dot products
        const int h2_q = q_dim / 2;
        const half2* attn2 = (const half2*)attn_scratch;
        for (int i = tid; i < h2_q; i += PLD_THREADS) {
            half2 a = attn2[i];
            s_hidden[i * 2]     = __half2float(a.x);
            s_hidden[i * 2 + 1] = __half2float(a.y);
        }
        if ((q_dim & 1) && tid == 0) {
            s_hidden[q_dim - 1] = __half2float(attn_scratch[q_dim - 1]);
        }
        __syncthreads();

        // O-proj GEMV using s_hidden (contains attn output in f32)
        // o_weight is [hidden, q_dim], so we need q_dim as the inner dimension
        for (int base = blockIdx.x * PLD_RPB; base < hidden_size; base += gridDim.x * PLD_RPB) {
            int rows_here = min(PLD_RPB, hidden_size - base);
            const int warp_id = tid / 32;
            const int lane_id = tid % 32;

            for (int r = 0; r < rows_here; r++) {
                int row = base + r;
                const half2* w2 = (const half2*)(o_weight + (long long)row * q_dim);
                float acc = 0.0f;
                for (int i = tid; i < h2_q; i += PLD_THREADS) {
                    half2 w = w2[i];
                    acc += __half2float(w.x) * s_hidden[i * 2];
                    acc += __half2float(w.y) * s_hidden[i * 2 + 1];
                }
                if ((q_dim & 1) && tid == 0) {
                    const __half* w_row = o_weight + (long long)row * q_dim;
                    acc += __half2float(w_row[q_dim - 1]) * s_hidden[q_dim - 1];
                }

                acc = pld_warp_reduce_sum(acc);
                if (lane_id == 0) s_scratch[warp_id] = acc;
                __syncthreads();

                if (warp_id == 0) {
                    float val = (lane_id < PLD_WARPS) ? s_scratch[lane_id] : 0.0f;
                    val = pld_warp_reduce_sum(val);
                    if (lane_id == 0) oproj_scratch[row] = __float2half(val);
                }
                if (r + 1 < rows_here) __syncthreads();
            }
        }
    }

    dag_signal(sync_flags, DAG_SYNC_OPROJ);
    dag_wait(sync_flags, DAG_SYNC_OPROJ, gridDim.x);

    // ====================================================================
    // PHASE 5: Add + RMSNorm + GateUp GEMV
    //   residual_out (from Phase 1) + oproj_scratch -> new residual
    //   RMSNorm -> gateup GEMV
    // ====================================================================
    // RACE AVOIDANCE: residual_out is both input and output here.
    // All blocks must finish loading it into smem before block 0 overwrites it.
    // We inline the add+norm with a DAG sync between load and write.
    {
        const int h2 = hidden_size / 2;
        const half2* r2 = (const half2*)residual_out;
        const half2* o2 = (const half2*)oproj_scratch;
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;

        float local_ss = 0.0f;
        for (int i = tid; i < h2; i += PLD_THREADS) {
            half2 a = r2[i];
            half2 b = o2[i];
            float v0 = __half2float(a.x) + __half2float(b.x);
            float v1 = __half2float(a.y) + __half2float(b.y);
            s_hidden[i * 2]     = v0;
            s_hidden[i * 2 + 1] = v1;
            local_ss += v0 * v0 + v1 * v1;
        }
        if ((hidden_size & 1) && tid == 0) {
            float v = __half2float(residual_out[hidden_size - 1]) + __half2float(oproj_scratch[hidden_size - 1]);
            s_hidden[hidden_size - 1] = v;
            local_ss += v * v;
        }

        // Warp reduction
        local_ss = pld_warp_reduce_sum(local_ss);
        if (lane_id == 0) s_scratch[warp_id] = local_ss;
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane_id < PLD_WARPS) ? s_scratch[lane_id] : 0.0f;
            val = pld_warp_reduce_sum(val);
            if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
        }
        __syncthreads();

        float rms_scale = s_scratch[0];

        // DAG sync: all blocks have loaded residual_out into smem.
        // Now safe for block 0 to overwrite it.
        dag_signal(sync_flags, DAG_SYNC_LOAD5);
        dag_wait(sync_flags, DAG_SYNC_LOAD5, gridDim.x);

        if (blockIdx.x == 0) {
            for (int i = tid; i < hidden_size; i += PLD_THREADS)
                residual_out[i] = __float2half(s_hidden[i]);
        }

        // Apply norm weights
        for (int i = tid; i < hidden_size; i += PLD_THREADS)
            s_hidden[i] = s_hidden[i] * __half2float(post_norm_w[i]) * rms_scale;
        __syncthreads();
    }

    // GateUp GEMV
    pld_gemv_grid_stride(gateup_scratch, s_hidden, s_scratch,
                         gateup_weight, hidden_size, gate_up_dim);

    dag_signal(sync_flags, DAG_SYNC_GATEUP);
    dag_wait(sync_flags, DAG_SYNC_GATEUP, gridDim.x);

    // ====================================================================
    // PHASE 6: SiLU + Down GEMV
    //   silu(gate) * up @ down_weight -> mlp_out[hidden]
    // ====================================================================
    pld_silu_down_gemv_grid_stride(mlp_out, gateup_scratch, down_weight,
                                   hidden_size, intermediate_size);

    // No final sync needed -- caller waits for kernel completion.
}

// ============================================================================
// First-layer variant: no prev_mlp addition (prev_mlp is NULL).
// This is the same kernel -- caller passes NULL for prev_mlp.
// The pld_add_rmsnorm function handles the NULL case.
// ============================================================================
// (No separate kernel needed -- pass prev_mlp=nullptr)

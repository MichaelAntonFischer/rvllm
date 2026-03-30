// Optimized decode attention kernel: 256 threads, vectorized half2 loads,
// warp-parallel reductions. Fits in 48KB shared memory (no set_attribute needed).
//
// Improvements over FA2 (128 threads, scalar loads, block_reduce_sum):
//   - 2x threads (256 vs 128): better memory latency hiding
//   - half2 vectorized loads: 2x global memory bandwidth
//   - Warp-parallel dot products: less sync overhead
//   - K/V share same shared memory buffer (V loaded after K consumed)
//
// Launch: grid(num_seqs, num_heads), block(256)
// Shared: BC * head_dim * sizeof(float) + BC * sizeof(float) + WARPS * sizeof(float)
//       = 64 * 128 * 4 + 64 * 4 + 8 * 4 = 33,056 bytes (fits in 48KB)

#include <float.h>
#include <cuda_fp16.h>

#define FA3_BC 64
#define FA3_THREADS 256
#define FA3_WARPS 8

__device__ __forceinline__ float fa3_warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

extern "C"
__global__ void __launch_bounds__(FA3_THREADS, 2)
flash_attention_3_decode_f16io_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ query,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq
) {
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane_id  = tid % 32;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int kv_head_idx = (num_kv_heads == num_heads)
        ? head_idx
        : (head_idx / (num_heads / num_kv_heads));

    const int num_tiles = (context_len + FA3_BC - 1) / FA3_BC;

    // Shared memory: one KV tile buffer (reused for K then V) + scores + warp reduce
    extern __shared__ float smem[];
    float* s_kv     = smem;                          // [BC * head_dim]
    float* s_score  = smem + FA3_BC * head_dim;      // [BC]
    float* s_warp   = s_score + FA3_BC;              // [WARPS]

    // Load Q into registers (persistent)
    const int dims_per_thread = (head_dim + FA3_THREADS - 1) / FA3_THREADS;
    float q_reg[4];
    const int q_base = (seq_idx * num_heads + head_idx) * head_dim;
    #pragma unroll
    for (int r = 0; r < dims_per_thread && r < 4; r++) {
        int d = tid + r * FA3_THREADS;
        q_reg[r] = (d < head_dim) ? (__half2float(query[q_base + d]) * scale) : 0.0f;
    }

    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float acc[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) acc[r] = 0.0f;

    for (int tile = 0; tile < num_tiles; tile++) {
        const int tile_start = tile * FA3_BC;
        const int tile_len = min(FA3_BC, context_len - tile_start);

        // ---- Load K tile (half2 vectorized) ----
        {
            const int total_h2 = (tile_len * head_dim) / 2;
            for (int idx = tid; idx < total_h2; idx += FA3_THREADS) {
                int elem = idx * 2;
                int t = elem / head_dim;
                int d = elem % head_dim;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
                int base = ((phys_block * block_size + page_off) * num_kv_heads + kv_head_idx) * head_dim + d;
                __half2 h2 = *reinterpret_cast<const __half2*>(&key_cache[base]);
                s_kv[t * head_dim + d]     = __half2float(h2.x);
                s_kv[t * head_dim + d + 1] = __half2float(h2.y);
            }
            // Handle remainder
            int total_elems = tile_len * head_dim;
            if ((total_elems & 1) && tid == 0) {
                int e = total_elems - 1;
                int t = e / head_dim, d = e % head_dim;
                int kv_pos = tile_start + t;
                int pi = kv_pos / block_size, po = kv_pos % block_size;
                int pb = block_tables[seq_idx * max_blocks_per_seq + pi];
                s_kv[t * head_dim + d] = __half2float(key_cache[((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d]);
            }
        }
        __syncthreads();

        // ---- Q @ K^T (warp-parallel dot products) ----
        // All threads participate in every iteration to avoid syncthreads deadlock.
        for (int t = 0; t < tile_len; t++) {
            float dot = 0.0f;
            #pragma unroll
            for (int r = 0; r < dims_per_thread && r < 4; r++) {
                int d = tid + r * FA3_THREADS;
                if (d < head_dim) dot += q_reg[r] * s_kv[t * head_dim + d];
            }
            // Intra-warp reduction (no sync needed)
            dot = fa3_warp_sum(dot);
            // Cross-warp: lane 0 of each warp writes partial sum
            if (lane_id == 0) s_warp[warp_id] = dot;
            __syncthreads();
            // Thread 0 sums across warps (8 adds, negligible)
            if (tid == 0) {
                float total = 0.0f;
                for (int w = 0; w < FA3_WARPS; w++) total += s_warp[w];
                s_score[t] = total;
            }
            __syncthreads();
        }

        // ---- Online softmax ----
        float tile_max = -FLT_MAX;
        if (tid == 0) {
            for (int t = 0; t < tile_len; t++) tile_max = fmaxf(tile_max, s_score[t]);
            s_warp[0] = tile_max;
        }
        __syncthreads();
        tile_max = s_warp[0];
        __syncthreads();

        float prev_max = row_max;
        float new_max = fmaxf(row_max, tile_max);
        if (new_max > prev_max && prev_max > -FLT_MAX) {
            float correction = expf(prev_max - new_max);
            #pragma unroll
            for (int r = 0; r < dims_per_thread && r < 4; r++) acc[r] *= correction;
            row_sum *= correction;
        }
        row_max = new_max;

        if (tid == 0) {
            float tsum = 0.0f;
            for (int t = 0; t < tile_len; t++) {
                float v = expf(s_score[t] - row_max);
                s_score[t] = v;
                tsum += v;
            }
            s_warp[0] = tsum;
        }
        __syncthreads();
        row_sum += s_warp[0];
        __syncthreads();

        // ---- Load V tile (reuse s_kv, K is consumed) ----
        {
            const int total_h2 = (tile_len * head_dim) / 2;
            for (int idx = tid; idx < total_h2; idx += FA3_THREADS) {
                int elem = idx * 2;
                int t = elem / head_dim;
                int d = elem % head_dim;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
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
                int pb = block_tables[seq_idx * max_blocks_per_seq + pi];
                s_kv[t * head_dim + d] = __half2float(value_cache[((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d]);
            }
        }
        __syncthreads();

        // ---- Accumulate P @ V ----
        #pragma unroll
        for (int r = 0; r < dims_per_thread && r < 4; r++) {
            int d = tid + r * FA3_THREADS;
            if (d < head_dim) {
                float v_acc = 0.0f;
                for (int t = 0; t < tile_len; t++)
                    v_acc += s_score[t] * s_kv[t * head_dim + d];
                acc[r] += v_acc;
            }
        }
        __syncthreads();
    }

    // ---- Write output (f16) ----
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    int out_base = (seq_idx * num_heads + head_idx) * head_dim;
    #pragma unroll
    for (int r = 0; r < dims_per_thread && r < 4; r++) {
        int d = tid + r * FA3_THREADS;
        if (d < head_dim)
            output[out_base + d] = __float2half(acc[r] * inv_sum);
    }
}

// GQA-optimized decode attention: one block per KV head, processes all query heads
// in the group, loading each KV tile only once.
// Grid: (num_seqs, num_kv_heads, 1), Block: (256, 1, 1)
//
// Shared memory layout:
//   s_kv:     [BC * head_dim] floats  -- K/V tile buffer (reused)
//   s_scores: [MAX_HPG * BC] floats   -- per-head softmax weights (all heads stored)
//   s_warp:   [WARPS] floats          -- warp reduction scratch
//
// Strategy per tile:
//   1. Load K tile once into s_kv
//   2. For each query head g: compute QK^T, online softmax, store weights in s_scores[g*BC..]
//   3. Load V tile once into s_kv (overwrites K)
//   4. For each query head g: accumulate P@V using stored weights from s_scores[g*BC..]

#define FA3_GQA_MAX_HPG 8

extern "C"
__global__ void __launch_bounds__(FA3_THREADS, 1)
flash_attention_3_decode_gqa_f16io_kernel(
    __half* __restrict__ output,
    const __half* __restrict__ query,
    const __half* __restrict__ key_cache,
    const __half* __restrict__ value_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    float scale,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_context_len,
    int max_blocks_per_seq
) {
    const int seq_idx     = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int tid         = threadIdx.x;
    const int warp_id     = tid / 32;
    const int lane_id     = tid % 32;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    const int heads_per_group = num_heads / num_kv_heads;
    const int num_tiles = (context_len + FA3_BC - 1) / FA3_BC;
    const int dims_per_thread = (head_dim + FA3_THREADS - 1) / FA3_THREADS;

    extern __shared__ float smem[];
    float* s_kv     = smem;                                             // [BC * head_dim]
    float* s_scores = smem + FA3_BC * head_dim;                         // [MAX_HPG * BC]
    float* s_warp   = s_scores + FA3_GQA_MAX_HPG * FA3_BC;             // [WARPS]

    // Per-head online softmax state and output accumulators (in registers).
    float head_row_max[FA3_GQA_MAX_HPG];
    float head_row_sum[FA3_GQA_MAX_HPG];
    float head_acc[FA3_GQA_MAX_HPG][4];

    for (int g = 0; g < heads_per_group && g < FA3_GQA_MAX_HPG; g++) {
        head_row_max[g] = -FLT_MAX;
        head_row_sum[g] = 0.0f;
        #pragma unroll
        for (int r = 0; r < 4; r++) head_acc[g][r] = 0.0f;
    }

    float q_reg[4];

    for (int tile = 0; tile < num_tiles; tile++) {
        const int tile_start = tile * FA3_BC;
        const int tile_len = min(FA3_BC, context_len - tile_start);

        // ---- Load K tile ONCE (half2 vectorized) ----
        {
            const int total_h2 = (tile_len * head_dim) / 2;
            for (int idx = tid; idx < total_h2; idx += FA3_THREADS) {
                int elem = idx * 2;
                int t = elem / head_dim;
                int d = elem % head_dim;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
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
                int pb = block_tables[seq_idx * max_blocks_per_seq + pi];
                s_kv[t * head_dim + d] = __half2float(key_cache[((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d]);
            }
        }
        __syncthreads();

        // ---- For each query head: QK^T + online softmax, store weights ----
        for (int g = 0; g < heads_per_group && g < FA3_GQA_MAX_HPG; g++) {
            int head_idx = kv_head_idx * heads_per_group + g;
            float* g_scores = s_scores + g * FA3_BC;

            // Load Q for this head
            int q_base = (seq_idx * num_heads + head_idx) * head_dim;
            #pragma unroll
            for (int r = 0; r < dims_per_thread && r < 4; r++) {
                int d = tid + r * FA3_THREADS;
                q_reg[r] = (d < head_dim) ? (__half2float(query[q_base + d]) * scale) : 0.0f;
            }

            // Q @ K^T (warp-parallel dot products) -> raw scores into g_scores
            for (int t = 0; t < tile_len; t++) {
                float dot = 0.0f;
                #pragma unroll
                for (int r = 0; r < dims_per_thread && r < 4; r++) {
                    int d = tid + r * FA3_THREADS;
                    if (d < head_dim) dot += q_reg[r] * s_kv[t * head_dim + d];
                }
                dot = fa3_warp_sum(dot);
                if (lane_id == 0) s_warp[warp_id] = dot;
                __syncthreads();
                if (tid == 0) {
                    float total = 0.0f;
                    for (int w = 0; w < FA3_WARPS; w++) total += s_warp[w];
                    g_scores[t] = total;
                }
                __syncthreads();
            }

            // Online softmax: find tile max, rescale previous accumulators, compute weights
            float tile_max = -FLT_MAX;
            if (tid == 0) {
                for (int t = 0; t < tile_len; t++) tile_max = fmaxf(tile_max, g_scores[t]);
                s_warp[0] = tile_max;
            }
            __syncthreads();
            tile_max = s_warp[0];
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

            // Exponentiate scores and compute tile sum
            if (tid == 0) {
                float tsum = 0.0f;
                for (int t = 0; t < tile_len; t++) {
                    float v = expf(g_scores[t] - new_max);
                    g_scores[t] = v;
                    tsum += v;
                }
                s_warp[0] = tsum;
            }
            __syncthreads();
            head_row_sum[g] += s_warp[0];
            __syncthreads();
        }

        // ---- Load V tile ONCE (reuse s_kv, K is consumed) ----
        {
            const int total_h2 = (tile_len * head_dim) / 2;
            for (int idx = tid; idx < total_h2; idx += FA3_THREADS) {
                int elem = idx * 2;
                int t = elem / head_dim;
                int d = elem % head_dim;
                int kv_pos = tile_start + t;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int phys_block = block_tables[seq_idx * max_blocks_per_seq + page_idx];
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
                int pb = block_tables[seq_idx * max_blocks_per_seq + pi];
                s_kv[t * head_dim + d] = __half2float(value_cache[((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d]);
            }
        }
        __syncthreads();

        // ---- For each query head: accumulate P @ V using stored weights ----
        for (int g = 0; g < heads_per_group && g < FA3_GQA_MAX_HPG; g++) {
            float* g_scores = s_scores + g * FA3_BC;
            #pragma unroll
            for (int r = 0; r < dims_per_thread && r < 4; r++) {
                int d = tid + r * FA3_THREADS;
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

    // ---- Write output for all heads in the group ----
    for (int g = 0; g < heads_per_group && g < FA3_GQA_MAX_HPG; g++) {
        int head_idx = kv_head_idx * heads_per_group + g;
        float inv = (head_row_sum[g] > 0.0f) ? (1.0f / head_row_sum[g]) : 0.0f;
        int out_base = (seq_idx * num_heads + head_idx) * head_dim;
        #pragma unroll
        for (int r = 0; r < dims_per_thread && r < 4; r++) {
            int d = tid + r * FA3_THREADS;
            if (d < head_dim)
                output[out_base + d] = __float2half(head_acc[g][r] * inv);
        }
    }
}

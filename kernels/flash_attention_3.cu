// FlashAttention-3 decode kernel with f16 shared memory.
//
// Improvements over previous FA3 (f32 smem):
//   - f16 shared memory: halves smem footprint, freeing L1 cache
//   - half2 vectorized loads: same as before
//   - Same warp-parallel reductions, same tile sizes
//
// The f16->f32 conversion happens at compute time (register), not at load time
// (shared memory). This halves smem bandwidth and lets the L1 cache serve more
// data concurrently.
//
// Shared memory layout (f16):
//   s_kv[BC * head_dim] half   -- K then V tile (reused)
//   s_score[BC] float          -- attention weights
//   s_warp[WARPS] float        -- warp reduction scratch
//
// Total: 64*128*2 + 64*4 + 8*4 = 16,384 + 288 = 16,672 bytes
// (half the old f32 kernel's 33,056 bytes!)
//
// Launch: grid(num_seqs, num_heads), block(256)

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

    // Shared memory: f16 KV tile + f32 scores + f32 warp scratch
    extern __shared__ char smem_raw[];
    __half* s_kv    = (__half*)smem_raw;                              // [BC * head_dim]
    float* s_score  = (float*)(s_kv + FA3_BC * head_dim);            // [BC]
    float* s_warp   = s_score + FA3_BC;                               // [WARPS]

    // Load Q into registers (persistent across all tiles)
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

        // ---- Load K tile (half2 vectorized, stays as f16 in smem) ----
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
                s_kv[t * head_dim + d]     = h2.x;
                s_kv[t * head_dim + d + 1] = h2.y;
            }
            int total_elems = tile_len * head_dim;
            if ((total_elems & 1) && tid == 0) {
                int e = total_elems - 1;
                int t = e / head_dim, d = e % head_dim;
                int kv_pos = tile_start + t;
                int pi = kv_pos / block_size, po = kv_pos % block_size;
                int pb = block_tables[seq_idx * max_blocks_per_seq + pi];
                s_kv[t * head_dim + d] = key_cache[((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d];
            }
        }
        __syncthreads();

        // ---- Q @ K^T (warp-parallel dot products, f16 smem reads) ----
        for (int t = 0; t < tile_len; t++) {
            float dot = 0.0f;
            #pragma unroll
            for (int r = 0; r < dims_per_thread && r < 4; r++) {
                int d = tid + r * FA3_THREADS;
                if (d < head_dim) dot += q_reg[r] * __half2float(s_kv[t * head_dim + d]);
            }
            dot = fa3_warp_sum(dot);
            if (lane_id == 0) s_warp[warp_id] = dot;
            __syncthreads();
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
                s_kv[t * head_dim + d]     = h2.x;
                s_kv[t * head_dim + d + 1] = h2.y;
            }
            int total_elems = tile_len * head_dim;
            if ((total_elems & 1) && tid == 0) {
                int e = total_elems - 1;
                int t = e / head_dim, d = e % head_dim;
                int kv_pos = tile_start + t;
                int pi = kv_pos / block_size, po = kv_pos % block_size;
                int pb = block_tables[seq_idx * max_blocks_per_seq + pi];
                s_kv[t * head_dim + d] = value_cache[((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d];
            }
        }
        __syncthreads();

        // ---- Accumulate P @ V (f16 V reads from smem) ----
        #pragma unroll
        for (int r = 0; r < dims_per_thread && r < 4; r++) {
            int d = tid + r * FA3_THREADS;
            if (d < head_dim) {
                float v_acc = 0.0f;
                for (int t = 0; t < tile_len; t++)
                    v_acc += s_score[t] * __half2float(s_kv[t * head_dim + d]);
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

// ---- GQA-optimized decode kernel ----
// One block per KV head, all query heads in the group share KV loads.
// f16 shared memory, same tile sizes as before.

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

    // Shared memory: f16 KV tile + f32 per-head scores + f32 warp scratch
    extern __shared__ char smem_raw[];
    __half* s_kv     = (__half*)smem_raw;                                       // [BC * head_dim]
    float* s_scores  = (float*)(s_kv + FA3_BC * head_dim);                     // [MAX_HPG * BC]
    float* s_warp    = s_scores + FA3_GQA_MAX_HPG * FA3_BC;                    // [WARPS]

    // Per-head state in registers
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

        // ---- Load K tile ONCE (half2 vectorized, f16 in smem) ----
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
                s_kv[t * head_dim + d]     = h2.x;
                s_kv[t * head_dim + d + 1] = h2.y;
            }
            int total_elems = tile_len * head_dim;
            if ((total_elems & 1) && tid == 0) {
                int e = total_elems - 1;
                int t = e / head_dim, d = e % head_dim;
                int kv_pos = tile_start + t;
                int pi = kv_pos / block_size, po = kv_pos % block_size;
                int pb = block_tables[seq_idx * max_blocks_per_seq + pi];
                s_kv[t * head_dim + d] = key_cache[((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d];
            }
        }
        __syncthreads();

        // ---- For each query head: QK^T + online softmax ----
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

            // Q @ K^T (f16 K from smem)
            for (int t = 0; t < tile_len; t++) {
                float dot = 0.0f;
                #pragma unroll
                for (int r = 0; r < dims_per_thread && r < 4; r++) {
                    int d = tid + r * FA3_THREADS;
                    if (d < head_dim) dot += q_reg[r] * __half2float(s_kv[t * head_dim + d]);
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

            // Online softmax
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

        // ---- Load V tile ONCE (reuse s_kv, K consumed) ----
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
                s_kv[t * head_dim + d]     = h2.x;
                s_kv[t * head_dim + d + 1] = h2.y;
            }
            int total_elems = tile_len * head_dim;
            if ((total_elems & 1) && tid == 0) {
                int e = total_elems - 1;
                int t = e / head_dim, d = e % head_dim;
                int kv_pos = tile_start + t;
                int pi = kv_pos / block_size, po = kv_pos % block_size;
                int pb = block_tables[seq_idx * max_blocks_per_seq + pi];
                s_kv[t * head_dim + d] = value_cache[((pb * block_size + po) * num_kv_heads + kv_head_idx) * head_dim + d];
            }
        }
        __syncthreads();

        // ---- For each query head: accumulate P @ V ----
        for (int g = 0; g < heads_per_group && g < FA3_GQA_MAX_HPG; g++) {
            float* g_scores = s_scores + g * FA3_BC;
            #pragma unroll
            for (int r = 0; r < dims_per_thread && r < 4; r++) {
                int d = tid + r * FA3_THREADS;
                if (d < head_dim) {
                    float v_acc = 0.0f;
                    for (int t = 0; t < tile_len; t++)
                        v_acc += g_scores[t] * __half2float(s_kv[t * head_dim + d]);
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

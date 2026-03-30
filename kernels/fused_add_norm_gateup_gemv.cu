// Fused residual-add + RMSNorm + GEMV kernel for gate_up projection (M=1 decode).
//
// Eliminates intermediate buffers by computing:
//   residual_out = input + add_vec
//   normed = rmsnorm(residual_out, norm_weight, eps)
//   output[j] = dot(proj_weight[j, :], normed)
//
// Each block handles rpb=8 output rows. All blocks redundantly compute the
// residual add + RMSNorm into shared memory, then each block computes 8 dot
// products against consecutive weight rows.
//
// Launch config:
//   Grid:  ((gate_up_dim + 7) / 8, 1, 1)
//   Block: (256, 1, 1)
//   Shared mem: hidden_size * sizeof(float) + 8 * sizeof(float)
//
// For Qwen2.5-1.5B: hidden=1536, gate_up_dim=17920, 2240 blocks.

#include <cuda_fp16.h>

#define THREADS 256
#define ROWS_PER_BLOCK 8

__device__ __forceinline__ float warp_reduce_sum_angv(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_add_norm_gateup_gemv(
    __half* __restrict__ output,           // [gate_up_dim]
    __half* __restrict__ residual_out,     // [hidden_size]
    const __half* __restrict__ input,      // [hidden_size] -- residual
    const __half* __restrict__ add_vec,    // [hidden_size] -- O projection output
    const __half* __restrict__ norm_weight, // [hidden_size]
    const __half* __restrict__ proj_weight, // [gate_up_dim, hidden_size] row-major
    float eps,
    int hidden_size,
    int gate_up_dim
) {
    const int block_row_base = blockIdx.x * ROWS_PER_BLOCK;
    if (block_row_base >= gate_up_dim) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    // Shared memory layout:
    //   [0 .. hidden_size-1]  : normed hidden state (f32)
    //   [hidden_size .. hidden_size+7] : warp partial sums scratch
    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_warp   = smem + hidden_size;

    // ---- Phase 1: Residual add + RMSNorm ----
    // All blocks redundantly compute this. Cheap for hidden_size=1536.

    const int h2 = hidden_size / 2;
    const half2* in2  = (const half2*)input;
    const half2* add2 = (const half2*)add_vec;

    float local_ss = 0.0f;
    for (int i = tid; i < h2; i += THREADS) {
        half2 a = in2[i];
        half2 b = add2[i];
        float v0 = __half2float(a.x) + __half2float(b.x);
        float v1 = __half2float(a.y) + __half2float(b.y);
        s_normed[i * 2]     = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    // Handle odd hidden_size
    if ((hidden_size & 1) && tid == 0) {
        int last = hidden_size - 1;
        float v = __half2float(input[last]) + __half2float(add_vec[last]);
        s_normed[last] = v;
        local_ss += v * v;
    }

    // Warp-level reduction of sum-of-squares
    local_ss = warp_reduce_sum_angv(local_ss);
    if (lane_id == 0) s_warp[warp_id] = local_ss;
    __syncthreads();

    // Cross-warp reduction in first warp
    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
        val = warp_reduce_sum_angv(val);
        if (lane_id == 0) {
            s_warp[0] = rsqrtf(val / (float)hidden_size + eps);
        }
    }
    __syncthreads();

    float rms_scale = s_warp[0];

    // Block 0 writes residual_out (only one block to avoid races)
    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += THREADS) {
            residual_out[i] = __float2half(s_normed[i]);
        }
    }

    // Apply norm weights in-place in smem
    for (int i = tid; i < hidden_size; i += THREADS) {
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    }
    __syncthreads();

    // ---- Phase 2: GEMV -- rpb=8 dot products per block ----

    // Compute how many rows this block actually handles (last block may have fewer)
    const int rows_this_block = min(ROWS_PER_BLOCK, gate_up_dim - block_row_base);

    // Each thread accumulates rpb=8 dot products
    float acc[ROWS_PER_BLOCK];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) acc[r] = 0.0f;

    for (int i = tid; i < h2; i += THREADS) {
        float sn0 = s_normed[i * 2];
        float sn1 = s_normed[i * 2 + 1];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            if (r < rows_this_block) {
                const half2* w2 = (const half2*)(proj_weight + (long long)(block_row_base + r) * hidden_size);
                half2 w = w2[i];
                acc[r] += __half2float(w.x) * sn0 + __half2float(w.y) * sn1;
            }
        }
    }

    // Handle odd hidden_size
    if ((hidden_size & 1) && tid == 0) {
        int last = hidden_size - 1;
        float sn = s_normed[last];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            if (r < rows_this_block) {
                const __half* w_row = proj_weight + (long long)(block_row_base + r) * hidden_size;
                acc[r] += __half2float(w_row[last]) * sn;
            }
        }
    }

    // Reduce each row's dot product: warp shuffle then cross-warp via shared mem
    // Serialize across rows reusing the 8-float scratch space
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        if (r >= rows_this_block) break;

        float val = warp_reduce_sum_angv(acc[r]);
        if (lane_id == 0) s_warp[warp_id] = val;
        __syncthreads();

        if (warp_id == 0) {
            float v = (lane_id < NUM_WARPS) ? s_warp[lane_id] : 0.0f;
            v = warp_reduce_sum_angv(v);
            if (lane_id == 0) {
                output[block_row_base + r] = __float2half(v);
            }
        }
        __syncthreads();
    }
}

// Fused residual-add + RMSNorm + QKV GEMV kernel for M=1 decode.
// f16 I/O, f32 accumulation. Each block handles RPB=8 output rows.
//
// Launch config:
//   Grid:  ((qkv_dim + RPB - 1) / RPB, 1, 1)
//   Block: (256, 1, 1)
//   Shared mem: hidden_size * sizeof(float) + RPB * sizeof(float)
//
// For Qwen2.5-1.5B: hidden=1536, qkv_dim=2048 -> 256 blocks, 8 rows each.

#include <cuda_fp16.h>

#define THREADS 256
#define RPB 8

__device__ __forceinline__ float warp_reduce_sum_anqg(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// --------------------------------------------------------------------------
// Fused add + RMSNorm + QKV GEMV (layers 1..N where residual add is needed)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_add_norm_qkv_gemv(
    __half* __restrict__ output,          // [qkv_dim]
    __half* __restrict__ residual_out,    // [hidden_size]
    const __half* __restrict__ input,     // [hidden_size]
    const __half* __restrict__ add_vec,   // [hidden_size]
    const __half* __restrict__ norm_weight, // [hidden_size]
    const __half* __restrict__ proj_weight, // [qkv_dim, hidden_size]
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    // Shared memory: [hidden_size] floats for normed vector + [RPB] floats for scratch
    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    // Phase 1: residual add -> shared mem, compute sum-of-squares
    float local_ss = 0.0f;
    const int h2 = hidden_size / 2;
    const half2* in2 = (const half2*)input;
    const half2* add2 = (const half2*)add_vec;

    for (int i = tid; i < h2; i += THREADS) {
        half2 a = in2[i];
        half2 b = add2[i];
        float v0 = __half2float(a.x) + __half2float(b.x);
        float v1 = __half2float(a.y) + __half2float(b.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        int last = hidden_size - 1;
        float v = __half2float(input[last]) + __half2float(add_vec[last]);
        s_normed[last] = v;
        local_ss += v * v;
    }

    // Warp reduction of sum-of-squares
    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    // Block 0 writes residual_out (pre-norm residual)
    if (blockIdx.x == 0) {
        for (int i = tid; i < hidden_size; i += THREADS) {
            residual_out[i] = __float2half(s_normed[i]);
        }
    }

    // Apply RMSNorm: multiply by norm_weight and rms_scale in shared mem
    for (int i = tid; i < hidden_size; i += THREADS) {
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    }
    __syncthreads();

    // Phase 2: GEMV -- each block computes RPB dot products
    for (int r = 0; r < RPB; r++) {
        int row = block_base + r;
        if (row >= qkv_dim) break;

        const __half* w_row = proj_weight + (long long)row * hidden_size;
        float acc = 0.0f;

        const half2* w2 = (const half2*)w_row;
        for (int i = tid; i < h2; i += THREADS) {
            half2 w = w2[i];
            int base = i * 2;
            acc += __half2float(w.x) * s_normed[base];
            acc += __half2float(w.y) * s_normed[base + 1];
        }
        if ((hidden_size & 1) && tid == 0) {
            int last = hidden_size - 1;
            acc += __half2float(w_row[last]) * s_normed[last];
        }

        // Warp reduction
        acc = warp_reduce_sum_anqg(acc);
        if (lane_id == 0) s_scratch[warp_id] = acc;
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
            val = warp_reduce_sum_anqg(val);
            if (lane_id == 0) output[row] = __float2half(val);
        }
        // Need sync before next iteration reuses s_scratch
        if (r + 1 < RPB) __syncthreads();
    }
}

// --------------------------------------------------------------------------
// First-layer variant: RMSNorm + QKV GEMV (no residual add)
// --------------------------------------------------------------------------
extern "C"
__global__ void __launch_bounds__(THREADS)
fused_cute_norm_qkv_gemv(
    __half* __restrict__ output,          // [qkv_dim]
    const __half* __restrict__ input,     // [hidden_size]
    const __half* __restrict__ norm_weight, // [hidden_size]
    const __half* __restrict__ proj_weight, // [qkv_dim, hidden_size]
    float eps,
    int hidden_size,
    int qkv_dim
) {
    const int block_base = blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int NUM_WARPS = THREADS / 32;

    extern __shared__ float smem[];
    float* s_normed = smem;
    float* s_scratch = smem + hidden_size;

    // Phase 1: Load input into shared mem, compute sum-of-squares
    float local_ss = 0.0f;
    const int h2 = hidden_size / 2;
    const half2* in2 = (const half2*)input;

    for (int i = tid; i < h2; i += THREADS) {
        half2 a = in2[i];
        float v0 = __half2float(a.x);
        float v1 = __half2float(a.y);
        s_normed[i * 2] = v0;
        s_normed[i * 2 + 1] = v1;
        local_ss += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        int last = hidden_size - 1;
        float v = __half2float(input[last]);
        s_normed[last] = v;
        local_ss += v * v;
    }

    local_ss = warp_reduce_sum_anqg(local_ss);
    if (lane_id == 0) s_scratch[warp_id] = local_ss;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
        val = warp_reduce_sum_anqg(val);
        if (lane_id == 0) s_scratch[0] = rsqrtf(val / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_scale = s_scratch[0];

    // Apply RMSNorm
    for (int i = tid; i < hidden_size; i += THREADS) {
        s_normed[i] = s_normed[i] * __half2float(norm_weight[i]) * rms_scale;
    }
    __syncthreads();

    // Phase 2: GEMV -- RPB rows per block
    for (int r = 0; r < RPB; r++) {
        int row = block_base + r;
        if (row >= qkv_dim) break;

        const __half* w_row = proj_weight + (long long)row * hidden_size;
        float acc = 0.0f;

        const half2* w2 = (const half2*)w_row;
        for (int i = tid; i < h2; i += THREADS) {
            half2 w = w2[i];
            int base = i * 2;
            acc += __half2float(w.x) * s_normed[base];
            acc += __half2float(w.y) * s_normed[base + 1];
        }
        if ((hidden_size & 1) && tid == 0) {
            int last = hidden_size - 1;
            acc += __half2float(w_row[last]) * s_normed[last];
        }

        acc = warp_reduce_sum_anqg(acc);
        if (lane_id == 0) s_scratch[warp_id] = acc;
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? s_scratch[lane_id] : 0.0f;
            val = warp_reduce_sum_anqg(val);
            if (lane_id == 0) output[row] = __float2half(val);
        }
        if (r + 1 < RPB) __syncthreads();
    }
}

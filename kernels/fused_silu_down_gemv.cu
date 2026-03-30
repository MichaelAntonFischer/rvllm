// Fused SiLU*Mul + Down-projection GEMV (f16 I/O, f32 accumulation).
//
// Each block computes RPB=8 output rows. With 256 threads and 8 rows,
// each row is handled by one warp (32 threads). This gives 8x fewer blocks
// than the 1-row-per-block variant, improving occupancy and L2 reuse of
// gate/up vectors across the 8 rows sharing a block.
//
//   output[r] = sum_i( silu(gate[i]) * up[i] * weight[r, i] )
//
// Launch config:
//   Grid:  ((hidden_size + 7) / 8, 1, 1)
//   Block: (256, 1, 1)
//   Shared memory: RPB * sizeof(float) = 32 bytes

#include <cuda_fp16.h>

#define CUTE_SILU_THREADS 256
#define CUTE_SILU_RPB 8
#define CUTE_SILU_WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum_csg(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float silu_f32_csg(float x) {
    return x / (1.0f + expf(-x));
}

extern "C"
__global__ void __launch_bounds__(CUTE_SILU_THREADS)
fused_cute_silu_down_gemv(
    __half* __restrict__ output,         // [hidden_size]
    const __half* __restrict__ gate,     // [intermediate_size]
    const __half* __restrict__ up,       // [intermediate_size]
    const __half* __restrict__ weight,   // [hidden_size, intermediate_size]
    int hidden_size,
    int intermediate_size
) {
    // 8 rows per block, 1 warp per row
    const int base_row = blockIdx.x * CUTE_SILU_RPB;
    const int warp_id = threadIdx.x / CUTE_SILU_WARP_SIZE;
    const int lane_id = threadIdx.x % CUTE_SILU_WARP_SIZE;
    const int row = base_row + warp_id;

    if (row >= hidden_size) return;

    const __half* w_row = weight + (long long)row * intermediate_size;
    float acc = 0.0f;

    // Vectorized half2 loads: 2 elements per iteration per lane
    const int k2 = intermediate_size / 2;
    const half2* gate2 = (const half2*)gate;
    const half2* up2   = (const half2*)up;
    const half2* w2    = (const half2*)w_row;

    for (int i = lane_id; i < k2; i += CUTE_SILU_WARP_SIZE) {
        half2 g = gate2[i];
        half2 u = up2[i];
        half2 w = w2[i];

        float g0 = __half2float(g.x);
        float g1 = __half2float(g.y);
        float u0 = __half2float(u.x);
        float u1 = __half2float(u.y);
        float w0 = __half2float(w.x);
        float w1 = __half2float(w.y);

        acc += silu_f32_csg(g0) * u0 * w0;
        acc += silu_f32_csg(g1) * u1 * w1;
    }

    // Handle odd intermediate_size
    if ((intermediate_size & 1) && lane_id == 0) {
        int last = intermediate_size - 1;
        float g = __half2float(gate[last]);
        acc += silu_f32_csg(g) * __half2float(up[last]) * __half2float(w_row[last]);
    }

    // Warp-level reduction (no shared memory needed within a warp)
    acc = warp_reduce_sum_csg(acc);

    // Lane 0 of each warp writes the result
    if (lane_id == 0) {
        output[row] = __float2half(acc);
    }
}

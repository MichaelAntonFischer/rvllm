// Standalone FP8 GEMV kernel benchmark for GB10 (sm_121).
//
// Measures actual memory bandwidth and tok/s potential for each kernel variant.
// Compile: nvcc -O3 -arch=sm_121 bench_fp8_gemv.cu -o bench_fp8_gemv
//
// Qwen3.5-27B layer dimensions:
//   hidden_size = 3584, intermediate_size = 18944
//   num_attention_heads = 24, num_kv_heads = 8 (GQA 3:1)
//   head_dim = 256 (query 256, note: gated output doubles Q-proj)
//   QKV sizes: Q = 3584 x 12288, K = 3584 x 2048, V = 3584 x 2048
//   Gate+Up: 3584 x 37888 (18944*2), Down: 18944 x 3584
//   Total FP8 per layer: ~200 MB
//   58 layers → ~11.6 GB FP8 weights (not 27 GB — rest is embeddings/norms)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// FP8 E4M3 → f32 conversion (branchless bit manipulation)
// ============================================================================
__device__ __forceinline__ float fp8e4m3_to_float(unsigned char val) {
    unsigned int s = (val >> 7) & 1u;
    unsigned int e = (val >> 3) & 0xFu;
    unsigned int m = val & 0x7u;
    unsigned int f32_bits = (s << 31) | ((e + 120u) << 23) | (m << 20);
    unsigned int is_normal = (e != 0u) & ((e != 0xFu) | (m != 0x7u));
    f32_bits &= (unsigned int)(-(int)is_normal);
    return __uint_as_float(f32_bits);
}

// ============================================================================
// Kernel 1: Blockwise v2 — SM_121 safe (scalar loads, LUT)
// Current production kernel for GB10.
// ============================================================================
extern "C"
__global__ void fp8_gemv_blockwise_v2_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scale,
    const float* __restrict__ input,
    int M, int N, int K,
    int num_col_blocks
) {
    int n = blockIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int BLOCK_DIM = 256;

    __shared__ float fp8_lut[256];
    fp8_lut[threadIdx.x] = fp8e4m3_to_float((unsigned char)threadIdx.x);
    __syncthreads();

    int scale_row = n >> 7;
    const unsigned char* w_row = weight + (long long)n * K;
    const float* x_row = input + (long long)m * K;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int k = threadIdx.x * 8; k + 7 < K; k += BLOCK_DIM * 8) {
        unsigned int lo4 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k));
        unsigned int hi4 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k + 4));

        int sc0 = k >> 7;
        float s0 = __ldg(&scale[scale_row * num_col_blocks + sc0]);
        int sc4 = (k + 4) >> 7;
        float s4 = (sc4 != sc0) ? __ldg(&scale[scale_row * num_col_blocks + sc4]) : s0;

        acc0 += fp8_lut[lo4 & 0xFFu]         * s0 * __ldg(x_row + k);
        acc0 += fp8_lut[(lo4 >> 8) & 0xFFu]  * s0 * __ldg(x_row + k + 1);
        acc0 += fp8_lut[(lo4 >> 16) & 0xFFu] * s0 * __ldg(x_row + k + 2);
        acc0 += fp8_lut[(lo4 >> 24) & 0xFFu] * s0 * __ldg(x_row + k + 3);
        acc1 += fp8_lut[hi4 & 0xFFu]         * s4 * __ldg(x_row + k + 4);
        acc1 += fp8_lut[(hi4 >> 8) & 0xFFu]  * s4 * __ldg(x_row + k + 5);
        acc1 += fp8_lut[(hi4 >> 16) & 0xFFu] * s4 * __ldg(x_row + k + 6);
        acc1 += fp8_lut[(hi4 >> 24) & 0xFFu] * s4 * __ldg(x_row + k + 7);
    }

    float acc = acc0 + acc1;

    {
        int aligned_k = (K / 8) * 8;
        for (int kr = aligned_k + threadIdx.x; kr < K; kr += BLOCK_DIM) {
            int sc = kr >> 7;
            float s = __ldg(&scale[scale_row * num_col_blocks + sc]);
            acc += fp8_lut[__ldg(w_row + kr)] * s * __ldg(x_row + kr);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    __shared__ float warp_sums[BLOCK_DIM / 32];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < (BLOCK_DIM / 32)) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = (BLOCK_DIM / 64); offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        if (lane == 0) output[(long long)m * N + n] = acc;
    }
}

// ============================================================================
// Kernel 2: Wider work per thread — 16 FP8 per iter, 4 accumulators
// Attempt to better saturate memory bandwidth by increasing work per thread.
// ============================================================================
extern "C"
__global__ void fp8_gemv_wide_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scale,
    const float* __restrict__ input,
    int M, int N, int K,
    int num_col_blocks
) {
    int n = blockIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const int BLOCK_DIM = 256;

    __shared__ float fp8_lut[256];
    fp8_lut[threadIdx.x] = fp8e4m3_to_float((unsigned char)threadIdx.x);
    __syncthreads();

    int scale_row = n >> 7;
    const unsigned char* w_row = weight + (long long)n * K;
    const float* x_row = input + (long long)m * K;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // 16 FP8 per iteration: 4x uint32 weight reads + 4x float4 input reads
    for (int k = threadIdx.x * 16; k + 15 < K; k += BLOCK_DIM * 16) {
        unsigned int w0 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k));
        unsigned int w1 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k + 4));
        unsigned int w2 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k + 8));
        unsigned int w3 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k + 12));

        int sc0 = k >> 7;
        float s = __ldg(&scale[scale_row * num_col_blocks + sc0]);

        // Check for scale boundary (every 128 elements)
        // 16 elements span at most 2 scale blocks
        int sc_end = (k + 15) >> 7;
        float s2 = (sc_end != sc0) ? __ldg(&scale[scale_row * num_col_blocks + sc_end]) : s;
        int boundary = (sc0 + 1) << 7;  // first element of next scale block

        // Which of the 4 groups of 4 crosses the boundary?
        float s_g0 = s, s_g1 = s, s_g2 = s, s_g3 = s;
        if (k + 4  >= boundary) s_g1 = s2;
        if (k + 8  >= boundary) s_g2 = s2;
        if (k + 12 >= boundary) s_g3 = s2;

        acc0 += fp8_lut[w0 & 0xFFu]         * s_g0 * __ldg(x_row + k);
        acc0 += fp8_lut[(w0 >> 8) & 0xFFu]  * s_g0 * __ldg(x_row + k + 1);
        acc0 += fp8_lut[(w0 >> 16) & 0xFFu] * s_g0 * __ldg(x_row + k + 2);
        acc0 += fp8_lut[(w0 >> 24) & 0xFFu] * s_g0 * __ldg(x_row + k + 3);

        acc1 += fp8_lut[w1 & 0xFFu]         * s_g1 * __ldg(x_row + k + 4);
        acc1 += fp8_lut[(w1 >> 8) & 0xFFu]  * s_g1 * __ldg(x_row + k + 5);
        acc1 += fp8_lut[(w1 >> 16) & 0xFFu] * s_g1 * __ldg(x_row + k + 6);
        acc1 += fp8_lut[(w1 >> 24) & 0xFFu] * s_g1 * __ldg(x_row + k + 7);

        acc2 += fp8_lut[w2 & 0xFFu]         * s_g2 * __ldg(x_row + k + 8);
        acc2 += fp8_lut[(w2 >> 8) & 0xFFu]  * s_g2 * __ldg(x_row + k + 9);
        acc2 += fp8_lut[(w2 >> 16) & 0xFFu] * s_g2 * __ldg(x_row + k + 10);
        acc2 += fp8_lut[(w2 >> 24) & 0xFFu] * s_g2 * __ldg(x_row + k + 11);

        acc3 += fp8_lut[w3 & 0xFFu]         * s_g3 * __ldg(x_row + k + 12);
        acc3 += fp8_lut[(w3 >> 8) & 0xFFu]  * s_g3 * __ldg(x_row + k + 13);
        acc3 += fp8_lut[(w3 >> 16) & 0xFFu] * s_g3 * __ldg(x_row + k + 14);
        acc3 += fp8_lut[(w3 >> 24) & 0xFFu] * s_g3 * __ldg(x_row + k + 15);
    }

    float acc = acc0 + acc1 + acc2 + acc3;

    {
        int aligned_k = (K / 16) * 16;
        for (int kr = aligned_k + threadIdx.x; kr < K; kr += BLOCK_DIM) {
            int sc = kr >> 7;
            float s = __ldg(&scale[scale_row * num_col_blocks + sc]);
            acc += fp8_lut[__ldg(w_row + kr)] * s * __ldg(x_row + kr);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    __shared__ float warp_sums[BLOCK_DIM / 32];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < (BLOCK_DIM / 32)) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = (BLOCK_DIM / 64); offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        if (lane == 0) output[(long long)m * N + n] = acc;
    }
}

// ============================================================================
// Kernel 3: Multi-row — each block computes ROWS_PER_BLOCK output rows.
// Reduces grid size, amortizes LUT build, and improves L2 weight reuse.
// ============================================================================
#define ROWS_PER_BLOCK 4

extern "C"
__global__ void fp8_gemv_multirow_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scale,
    const float* __restrict__ input,
    int M, int N, int K,
    int num_col_blocks
) {
    int n_base = blockIdx.x * ROWS_PER_BLOCK;
    int m = blockIdx.y;
    if (m >= M) return;

    const int BLOCK_DIM = 256;

    __shared__ float fp8_lut[256];
    fp8_lut[threadIdx.x] = fp8e4m3_to_float((unsigned char)threadIdx.x);
    __syncthreads();

    const float* x_row = input + (long long)m * K;

    // Each thread accumulates for ROWS_PER_BLOCK output rows
    float acc[ROWS_PER_BLOCK];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) acc[r] = 0.0f;

    for (int k = threadIdx.x * 8; k + 7 < K; k += BLOCK_DIM * 8) {
        // Load input once, reuse across all rows
        float x0 = __ldg(x_row + k);
        float x1 = __ldg(x_row + k + 1);
        float x2 = __ldg(x_row + k + 2);
        float x3 = __ldg(x_row + k + 3);
        float x4 = __ldg(x_row + k + 4);
        float x5 = __ldg(x_row + k + 5);
        float x6 = __ldg(x_row + k + 6);
        float x7 = __ldg(x_row + k + 7);

        #pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            int n = n_base + r;
            if (n >= N) break;

            const unsigned char* w_row = weight + (long long)n * K;
            int scale_row = n >> 7;

            unsigned int lo4 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k));
            unsigned int hi4 = __ldg(reinterpret_cast<const unsigned int*>(w_row + k + 4));

            int sc0 = k >> 7;
            float s0 = __ldg(&scale[scale_row * num_col_blocks + sc0]);
            int sc4 = (k + 4) >> 7;
            float s4 = (sc4 != sc0) ? __ldg(&scale[scale_row * num_col_blocks + sc4]) : s0;

            acc[r] += fp8_lut[lo4 & 0xFFu]         * s0 * x0;
            acc[r] += fp8_lut[(lo4 >> 8) & 0xFFu]  * s0 * x1;
            acc[r] += fp8_lut[(lo4 >> 16) & 0xFFu] * s0 * x2;
            acc[r] += fp8_lut[(lo4 >> 24) & 0xFFu] * s0 * x3;
            acc[r] += fp8_lut[hi4 & 0xFFu]         * s4 * x4;
            acc[r] += fp8_lut[(hi4 >> 8) & 0xFFu]  * s4 * x5;
            acc[r] += fp8_lut[(hi4 >> 16) & 0xFFu] * s4 * x6;
            acc[r] += fp8_lut[(hi4 >> 24) & 0xFFu] * s4 * x7;
        }
    }

    // Reduce and write each row
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int n = n_base + r;
        if (n >= N) break;

        float a = acc[r];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xffffffff, a, offset);

        __shared__ float warp_sums[BLOCK_DIM / 32];
        int warp = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        if (lane == 0) warp_sums[warp] = a;
        __syncthreads();

        if (warp == 0) {
            a = (lane < (BLOCK_DIM / 32)) ? warp_sums[lane] : 0.0f;
            #pragma unroll
            for (int offset = (BLOCK_DIM / 64); offset > 0; offset >>= 1)
                a += __shfl_down_sync(0xffffffff, a, offset);
            if (lane == 0) output[(long long)m * N + n] = a;
        }
        __syncthreads();  // barrier before reusing warp_sums
    }
}

// ============================================================================
// Host benchmark harness
// ============================================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

struct GemvShape {
    const char* name;
    int N;
    int K;
};

void bench_kernel(
    const char* kernel_name,
    void (*launcher)(float*, const unsigned char*, const float*, const float*,
                     int, int, int, int, dim3, dim3),
    int N, int K, int warmup, int iters
) {
    int M = 1;
    int num_col_blocks = (K + 127) / 128;
    int num_row_blocks = (N + 127) / 128;
    size_t w_bytes = (size_t)N * K;
    size_t s_bytes = (size_t)num_row_blocks * num_col_blocks * sizeof(float);
    size_t x_bytes = (size_t)M * K * sizeof(float);
    size_t o_bytes = (size_t)M * N * sizeof(float);

    unsigned char *d_w; float *d_s, *d_x, *d_o;
    CHECK_CUDA(cudaMalloc(&d_w, w_bytes));
    CHECK_CUDA(cudaMalloc(&d_s, s_bytes));
    CHECK_CUDA(cudaMalloc(&d_x, x_bytes));
    CHECK_CUDA(cudaMalloc(&d_o, o_bytes));

    // Fill with random data
    unsigned char* h_w = (unsigned char*)malloc(w_bytes);
    float* h_s = (float*)malloc(s_bytes);
    float* h_x = (float*)malloc(x_bytes);
    for (size_t i = 0; i < w_bytes; i++) h_w[i] = rand() & 0xFF;
    for (size_t i = 0; i < s_bytes / sizeof(float); i++) h_s[i] = 1.0f;
    for (int i = 0; i < M * K; i++) h_x[i] = 0.01f * (rand() % 100);

    CHECK_CUDA(cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_s, h_s, s_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice));

    // Determine grid/block
    dim3 block(256);
    dim3 grid(N, M);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        launcher(d_o, d_w, d_s, d_x, M, N, K, num_col_blocks, grid, block);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed runs
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        launcher(d_o, d_w, d_s, d_x, M, N, K, num_col_blocks, grid, block);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;

    // Bandwidth: weight bytes read per call (dominant term for M=1)
    double weight_bytes = (double)N * K;  // FP8 = 1 byte each
    double input_bytes = (double)K * sizeof(float);  // input vector
    double total_bytes = weight_bytes + input_bytes;
    double bw_gb_s = (total_bytes / (avg_ms * 1e-3)) / 1e9;

    printf("  %-30s  N=%-6d K=%-6d  %8.3f ms  %7.1f GB/s  %7.1f GElem/s\n",
           kernel_name, N, K, avg_ms, bw_gb_s, (weight_bytes / (avg_ms * 1e-3)) / 1e9);

    free(h_w); free(h_s); free(h_x);
    cudaFree(d_w); cudaFree(d_s); cudaFree(d_x); cudaFree(d_o);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

// ============================================================================
// Kernel 4: WPR (warp-per-row) — 8 warps per block, 32 threads per row.
// Production kernel from fp8_gemv.cu. Uses u64 loads.
// ============================================================================
extern "C"
__global__ void fp8_gemv_wpr_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scale,
    const float* __restrict__ input,
    int M, int N, int K,
    int num_col_blocks
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * 8 + warp;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    int scale_row = n >> 7;
    const unsigned char* w_row = weight + (long long)n * K;
    const float* x_row = input + (long long)m * K;

    float acc0 = 0.0f, acc1 = 0.0f;

    for (int k = lane * 8; k + 7 < K; k += 256) {
        unsigned long long w8 = __ldg(reinterpret_cast<const unsigned long long*>(w_row + k));
        unsigned int lo4 = (unsigned int)(w8);
        unsigned int hi4 = (unsigned int)(w8 >> 32);

        unsigned long long x01 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k));
        unsigned long long x23 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 2));
        unsigned long long x45 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 4));
        unsigned long long x67 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 6));

        int sc0 = k >> 7;
        float s0 = __ldg(&scale[scale_row * num_col_blocks + sc0]);
        int sc4 = (k + 4) >> 7;
        float s4 = (sc4 != sc0) ? __ldg(&scale[scale_row * num_col_blocks + sc4]) : s0;

        acc0 += fp8e4m3_to_float(lo4 & 0xFFu)         * s0 * __uint_as_float((unsigned int)(x01));
        acc0 += fp8e4m3_to_float((lo4 >> 8) & 0xFFu)  * s0 * __uint_as_float((unsigned int)(x01 >> 32));
        acc0 += fp8e4m3_to_float((lo4 >> 16) & 0xFFu) * s0 * __uint_as_float((unsigned int)(x23));
        acc0 += fp8e4m3_to_float((lo4 >> 24) & 0xFFu) * s0 * __uint_as_float((unsigned int)(x23 >> 32));
        acc1 += fp8e4m3_to_float(hi4 & 0xFFu)         * s4 * __uint_as_float((unsigned int)(x45));
        acc1 += fp8e4m3_to_float((hi4 >> 8) & 0xFFu)  * s4 * __uint_as_float((unsigned int)(x45 >> 32));
        acc1 += fp8e4m3_to_float((hi4 >> 16) & 0xFFu) * s4 * __uint_as_float((unsigned int)(x67));
        acc1 += fp8e4m3_to_float((hi4 >> 24) & 0xFFu) * s4 * __uint_as_float((unsigned int)(x67 >> 32));
    }

    float acc = acc0 + acc1;
    {
        int aligned_k = (K / 8) * 8;
        for (int kr = aligned_k + lane; kr < K; kr += 32) {
            int sc = kr >> 7;
            float s = __ldg(&scale[scale_row * num_col_blocks + sc]);
            acc += fp8e4m3_to_float(__ldg(w_row + kr)) * s * __ldg(x_row + kr);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    if (lane == 0) output[(long long)m * N + n] = acc;
}

// ============================================================================
// Kernel 5: WPR + Native CVT — hardware FP8 conversion (3 insn vs 24)
// Previously caused clock drops due to power bug. Should be fastest now.
// ============================================================================
__device__ __forceinline__ void fp8x2_to_f32_bench(unsigned short packed_fp8x2,
                                                    float& f0, float& f1) {
    unsigned int f16x2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(packed_fp8x2));
    unsigned short lo = (unsigned short)(f16x2);
    unsigned short hi = (unsigned short)(f16x2 >> 16);
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
}

extern "C"
__global__ void fp8_gemv_wpr_native_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ scale,
    const float* __restrict__ input,
    int M, int N, int K,
    int num_col_blocks
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * 8 + warp;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    int scale_row = n >> 7;
    const unsigned char* w_row = weight + (long long)n * K;
    const float* x_row = input + (long long)m * K;

    float acc0 = 0.0f, acc1 = 0.0f;

    for (int k = lane * 8; k + 7 < K; k += 256) {
        unsigned long long w8 = __ldg(reinterpret_cast<const unsigned long long*>(w_row + k));
        unsigned long long x01 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k));
        unsigned long long x23 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 2));
        unsigned long long x45 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 4));
        unsigned long long x67 = __ldg(reinterpret_cast<const unsigned long long*>(x_row + k + 6));

        int sc0 = k >> 7;
        float s0 = __ldg(&scale[scale_row * num_col_blocks + sc0]);
        int sc4 = (k + 4) >> 7;
        float s4 = (sc4 != sc0) ? __ldg(&scale[scale_row * num_col_blocks + sc4]) : s0;

        float w0, w1, w2, w3, w4, w5, w6, w7;
        fp8x2_to_f32_bench((unsigned short)(w8),       w0, w1);
        fp8x2_to_f32_bench((unsigned short)(w8 >> 16), w2, w3);
        fp8x2_to_f32_bench((unsigned short)(w8 >> 32), w4, w5);
        fp8x2_to_f32_bench((unsigned short)(w8 >> 48), w6, w7);

        acc0 += w0 * s0 * __uint_as_float((unsigned int)(x01));
        acc0 += w1 * s0 * __uint_as_float((unsigned int)(x01 >> 32));
        acc0 += w2 * s0 * __uint_as_float((unsigned int)(x23));
        acc0 += w3 * s0 * __uint_as_float((unsigned int)(x23 >> 32));
        acc1 += w4 * s4 * __uint_as_float((unsigned int)(x45));
        acc1 += w5 * s4 * __uint_as_float((unsigned int)(x45 >> 32));
        acc1 += w6 * s4 * __uint_as_float((unsigned int)(x67));
        acc1 += w7 * s4 * __uint_as_float((unsigned int)(x67 >> 32));
    }

    float acc = acc0 + acc1;
    {
        int aligned_k = (K / 8) * 8;
        for (int kr = aligned_k + lane; kr < K; kr += 32) {
            int sc = kr >> 7;
            float s = __ldg(&scale[scale_row * num_col_blocks + sc]);
            acc += fp8e4m3_to_float(__ldg(w_row + kr)) * s * __ldg(x_row + kr);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    if (lane == 0) output[(long long)m * N + n] = acc;
}

// Launcher wrappers
void launch_v2(float* o, const unsigned char* w, const float* s, const float* x,
               int M, int N, int K, int ncb, dim3 grid, dim3 block) {
    fp8_gemv_blockwise_v2_kernel<<<grid, block>>>(o, w, s, x, M, N, K, ncb);
}

void launch_wide(float* o, const unsigned char* w, const float* s, const float* x,
                 int M, int N, int K, int ncb, dim3 grid, dim3 block) {
    fp8_gemv_wide_kernel<<<grid, block>>>(o, w, s, x, M, N, K, ncb);
}

void launch_multirow(float* o, const unsigned char* w, const float* s, const float* x,
                     int M, int N, int K, int ncb, dim3 grid, dim3 block) {
    dim3 mr_grid((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1);
    fp8_gemv_multirow_kernel<<<mr_grid, block>>>(o, w, s, x, 1, N, K, ncb);
}

void launch_wpr(float* o, const unsigned char* w, const float* s, const float* x,
                int M, int N, int K, int ncb, dim3 grid, dim3 block) {
    dim3 wpr_grid(((N + 7) / 8), 1);
    fp8_gemv_wpr_kernel<<<wpr_grid, block>>>(o, w, s, x, 1, N, K, ncb);
}

void launch_wpr_native(float* o, const unsigned char* w, const float* s, const float* x,
                       int M, int N, int K, int ncb, dim3 grid, dim3 block) {
    dim3 wpr_grid(((N + 7) / 8), 1);
    fp8_gemv_wpr_native_kernel<<<wpr_grid, block>>>(o, w, s, x, 1, N, K, ncb);
}

int main() {
    int dev;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&dev));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s (sm_%d%d), %d SMs, %.1f GB memory\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);
    // GB10 uses unified LPDDR5X — no separate memory clock in cudaDeviceProp
    // Theoretical peak: 273 GB/s (8-channel LPDDR5X @ 8533 MT/s)
    double peak_bw = 273.0;
    printf("LPDDR5X theoretical peak BW: %.1f GB/s (shared CPU+GPU)\n\n", peak_bw);

    // Qwen3.5-27B representative layer shapes (M=1 decode)
    GemvShape shapes[] = {
        // Attention projections
        {"QProj (gated)",    12288, 3584},  // Q-proj with output gate
        {"KProj",             2048, 3584},  // K-proj
        {"VProj",             2048, 3584},  // V-proj
        {"OProj",             3584, 6144},  // O-proj (24 heads * 256)
        // MLP
        {"GateUp (fused)",   37888, 3584},  // gate + up projection
        {"Down",              3584, 18944}, // down projection
        // Synthetic large shapes for bandwidth ceiling
        {"Synthetic 8Kx8K",   8192, 8192},
        {"Synthetic 16Kx4K", 16384, 4096},
    };
    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);

    int warmup = 20;
    int iters = 200;

    printf("=== FP8 GEMV Benchmark (M=1 decode, %d warmup, %d iters) ===\n\n", warmup, iters);

    for (int i = 0; i < n_shapes; i++) {
        printf("[%s]\n", shapes[i].name);
        bench_kernel("v2 (256t/row, LUT)",     launch_v2,
                     shapes[i].N, shapes[i].K, warmup, iters);
        bench_kernel("multirow (4 rows/blk)",  launch_multirow,
                     shapes[i].N, shapes[i].K, warmup, iters);
        bench_kernel("WPR (8 warps, ALU)",     launch_wpr,
                     shapes[i].N, shapes[i].K, warmup, iters);
        bench_kernel("WPR+native (hw CVT)",    launch_wpr_native,
                     shapes[i].N, shapes[i].K, warmup, iters);
        printf("\n");
    }

    // Full-layer simulation: sum up all GEMVs in one transformer layer
    printf("=== Full Layer Simulation (1 decode step) ===\n");
    // Per layer: QKV + OProj + GateUp + Down
    // Total FP8 bytes per layer: 12288*3584 + 2048*3584 + 2048*3584 + 3584*6144 + 37888*3584 + 3584*18944
    long long bytes_per_layer =
        (long long)12288*3584 + (long long)2048*3584 + (long long)2048*3584 +
        (long long)3584*6144 + (long long)37888*3584 + (long long)3584*18944;
    int num_layers = 58;
    double total_gb = (double)bytes_per_layer * num_layers / 1e9;
    printf("Weight bytes/layer: %.2f MB, total %d layers: %.2f GB\n",
           bytes_per_layer / 1e6, num_layers, total_gb);
    printf("Theoretical tok/s at peak BW (%.0f GB/s): %.1f\n",
           peak_bw, peak_bw / total_gb);
    printf("Theoretical tok/s at 80%% BW: %.1f\n", peak_bw * 0.8 / total_gb);

    return 0;
}

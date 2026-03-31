#!/usr/bin/env python3
"""Benchmark cuBLAS vs Triton persistent matmul for rvllm GEMM shapes."""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def persistent_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


def triton_mm(A, B, NUM_SMS=132):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    config = dict(
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8, NUM_SMS=NUM_SMS, num_stages=3, num_warps=8,
    )
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),)
    persistent_matmul_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        **config,
    )
    return C


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Qwen2.5-7B shapes: C[M,N] = A[M,K] @ B[K,N]
    shapes = [
        (1, 18944, 3584, "gate_up M=1"),
        (1, 3584, 9472, "down M=1"),
        (1, 7168, 3584, "qkv M=1"),
        (32, 18944, 3584, "gate_up M=32"),
        (64, 18944, 3584, "gate_up M=64"),
        (128, 18944, 3584, "gate_up M=128"),
    ]

    print(f"{'shape':<20} {'cuBLAS':>10} {'Triton':>10} {'ratio':>8}")
    print("-" * 52)

    for M, N, K, label in shapes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)

        # cuBLAS warmup + bench
        for _ in range(10):
            torch.mm(A, B)
        torch.cuda.synchronize()
        iters = 200
        t0 = time.perf_counter()
        for _ in range(iters):
            torch.mm(A, B)
        torch.cuda.synchronize()
        cublas_us = (time.perf_counter() - t0) / iters * 1e6

        # Triton warmup + bench
        for _ in range(10):
            triton_mm(A, B)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            triton_mm(A, B)
        torch.cuda.synchronize()
        triton_us = (time.perf_counter() - t0) / iters * 1e6

        ratio = triton_us / cublas_us
        print(f"{label:<20} {cublas_us:>8.1f}us {triton_us:>8.1f}us {ratio:>7.2f}x")


if __name__ == "__main__":
    main()

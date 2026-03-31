#!/usr/bin/env python3
"""Compile Triton persistent matmul kernel to PTX/cubin for rvllm.

Generates shape-specialized GEMM kernels that replace cuBLAS for the
linear layer projections. Uses the same algorithm as vLLM's
batch_invariant.py persistent matmul.

Usage:
    python3 kernels/compile_triton_matmul.py --arch sm_90 --out kernels/sm_90/
"""

import argparse
import os
import triton
import triton.language as tl
import torch


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
    """Persistent tiled GEMM: C[M,N] = A[M,K] @ B[K,N]

    Each program instance loops over multiple output tiles.
    Uses tensor core tl.dot for the tile multiply.
    """
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        # Swizzled tile index for L2 locality
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
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


def compile_for_shape(M, N, K, num_sms=132, outdir=".", arch="sm_90"):
    """Compile the persistent matmul kernel for specific M, N, K."""

    # fp16 config (same as vLLM)
    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "NUM_SMS": num_sms,
        "num_stages": 3,
        "num_warps": 8,
    }

    # Create dummy tensors to trigger compilation
    a = torch.empty((M, K), device='cuda', dtype=torch.float16)
    b = torch.empty((K, N), device='cuda', dtype=torch.float16)
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)

    def grid(META):
        return (min(META["NUM_SMS"], triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),)

    # Warmup / compile
    persistent_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        **config,
    )

    # Extract compiled kernel
    key = persistent_matmul_kernel.cache[0]  # first (and only) variant
    compiled = list(persistent_matmul_kernel.cache.values())[0]

    # Get the PTX or cubin
    name = f"triton_matmul_{M}x{N}x{K}"
    asm = compiled.asm

    if 'ptx' in asm:
        ptx_path = os.path.join(outdir, f"{name}.ptx")
        with open(ptx_path, 'w') as f:
            f.write(asm['ptx'])
        print(f"Wrote {ptx_path}")
    if 'cubin' in asm:
        cubin_path = os.path.join(outdir, f"{name}.cubin")
        with open(cubin_path, 'wb') as f:
            f.write(asm['cubin'])
        print(f"Wrote {cubin_path}")

    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="sm_90")
    parser.add_argument("--out", default="kernels/sm_90/")
    parser.add_argument("--sms", type=int, default=132, help="Number of SMs (132 for H100)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Qwen2.5-7B shapes (M=1 for decode, but we pad to tile size internally)
    # C = A @ B^T where A=[M,K], B=[N,K] -> C=[M,N]
    # For row-major weight stored as [N,K], B is already [N,K], so B^T=[K,N]
    # Triton kernel does C[M,N] = A[M,K] @ B[K,N]
    # So we pass weight transposed.
    shapes = [
        # (M, N, K) -- note: these are the GEMM shapes after transposing weight
        # QKV projection: input[1, 3584] @ weight[3584, 7168] -> [1, 7168]
        (1, 7168, 3584),
        # O projection: input[1, 3584] @ weight[3584, 3584] -> [1, 3584]
        (1, 3584, 3584),
        # Gate+Up: input[1, 3584] @ weight[3584, 18944] -> [1, 18944]
        (1, 18944, 3584),
        # Down: input[1, 9472] @ weight[9472, 3584] -> [1, 3584]
        (1, 3584, 9472),
        # Batched decode shapes
        (8, 7168, 3584),
        (8, 18944, 3584),
        (8, 3584, 9472),
        (32, 7168, 3584),
        (32, 18944, 3584),
        (32, 3584, 9472),
        (64, 7168, 3584),
        (64, 18944, 3584),
        (64, 3584, 9472),
        (128, 7168, 3584),
        (128, 18944, 3584),
        (128, 3584, 9472),
    ]

    for M, N, K in shapes:
        try:
            name = compile_for_shape(M, N, K, num_sms=args.sms, outdir=args.out, arch=args.arch)
            print(f"  {name}: M={M}, N={N}, K={K}")
        except Exception as e:
            print(f"  FAILED M={M} N={N} K={K}: {e}")


if __name__ == "__main__":
    main()

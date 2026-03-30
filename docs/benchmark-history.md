# Benchmark History

All results greedy decoding, 32 tokens/request unless noted.

## Phase 5 (2026-03-30) -- Kernel fusion swarm (H100 SXM 80GB)

Direct engine benchmark (no HTTP). Fused kernels: add+norm+QKV GEMV, add+norm+gateup GEMV,
silu+down GEMV, GQA-optimized FA3 attention. Prefill uses fused QKV/gateup for all N.

| N | tok/s | vs Phase 4 (A100) | Notes |
|---|---|---|---|
| 1 | 240 | 1.88x | Fused kernels + cublasLt split-K |
| 4 | 1,201 | 2.22x | |
| 8 | 2,328 | 2.13x | |
| 16 | 4,229 | 2.00x | |
| 32 | 8,575 | 2.47x | |
| 64 | 15,812 | 3.89x | GQA attention 6x less KV bandwidth |
| 128 | 26,161 | 4.11x | |
| 256 | 40,714 | 4.89x | |

### Qwen2.5-7B f16 (H100 SXM 80GB, direct engine)

| N | tok/s | wall_ms |
|---|---|---|
| 1 | 108 | 296 |
| 4 | 544 | 235 |
| 8 | 1,073 | 238 |
| 16 | 2,019 | 253 |
| 32 | 3,911 | 261 |
| 64 | 7,300 | 280 |
| 128 | 12,624 | 324 |

N=256 hits KV cache limits at 0.9 gpu-memory-utilization for 7B.

Note: Phase 4 was on A100, Phase 5 on H100. H100 has ~2x raw bandwidth and ~3x tensor
core FLOPS vs A100. The per-hardware improvement from fusion alone is ~1.5-2x.

## Phase 4 (2026-03-28) -- CUDA graph + cublasLt (A100 80GB SXM4)

Measured with concurrent Python HTTP requests after graph capture fix.

| N | tok/s | ms/tok | Notes |
|---|---|---|---|
| 1 | 128 | 7.7 | 22.7% mem BW utilization |
| 4 | 540 | - | |
| 8 | 1,091 | - | |
| 16 | 2,118 | - | |
| 32 | 3,467 | - | |

Per-token overhead: 5.95ms (77% of total), theoretical peak 574 tok/s.

## Phase 3 (earlier) -- Sampling + attention backend

Previous head-to-head numbers (measured with bench/run.sh batched harness, not reproducible with current code):

| N | rvLLM (tok/s) | vLLM 0.18 (tok/s) |
|---|---|---|
| 1 | 117 | 69 |
| 4 | 882 | 256 |
| 8 | 1,213 | 517 |
| 16 | 1,391 | 1,060 |
| 32 | 1,434 | 1,943 |
| 48 | 3,918 | 2,887 |
| 64 | 4,796 | 3,828 |
| 96 | 5,965 | 5,197 |
| 128 | 7,380 | 6,400 |
| 256 | 9,905 | 9,437 |
| 512 | 10,291 | 10,771 |
| 768 | 10,235 | -- |
| 1024 | 10,051 | 12,740 |

Note: These numbers included optimizations (fused QKV, fused gate+up, vectorized float4, packed HtoD, pre-alloc buffers) that were lost in subsequent code changes and are being re-implemented in Phase 5.

## Phase 2 -- FP16 inference

- 8,339 tok/s peak at N=768
- Matched vLLM at N=48-128

## Phase 1 -- FP32 baseline

- 3,191 tok/s peak at N=512
- 86 tok/s single-sequence

## B200 Results (FP32, earlier)

| N | Tokens | Wall time | tok/s |
|---|---|---|---|
| 1 | 32 | 279ms | 114 |
| 64 | 2,048 | 798ms | 2,566 |
| 256 | 8,192 | 2,106ms | 3,889 |
| 768 | 24,576 | 6,227ms | 3,946 |
| 4,096 | 131,072 | 34,002ms | 3,854 |

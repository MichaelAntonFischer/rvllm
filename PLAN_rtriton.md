# rTriton: Rust Triton Implementation Plan

## What This Is

A standalone Rust crate (`crates/rtriton/`) that replicates Triton's GPU kernel compiler 1:1.
Takes block-level kernel definitions written in a Rust builder DSL, optimizes them, and emits
PTX/CUBIN. Includes pre-built LLM inference kernels matching what vLLM's torch.compile generates.

## Crate Structure

```
crates/rtriton/
  Cargo.toml
  src/
    lib.rs              -- module hub
    ir.rs               -- core IR types (SSA values, ops, blocks, functions, types)
    builder.rs          -- Rust DSL for writing Triton-style kernels
    passes.rs           -- optimization passes (fusion, coalescing, smem alloc, pipelining)
    codegen.rs          -- IR -> PTX text emission
    runtime.rs          -- kernel loading, caching, launching (cfg cuda)
    autotune.rs         -- config search + persistent cache
    graph.rs            -- CUDA graph capture/replay + buffer pool
    kernels/
      mod.rs            -- kernel library index
      rmsnorm.rs        -- RMSNorm + fused residual_add+rmsnorm
      rope.rs           -- RoPE + KV cache write
      fused_attention.rs -- Flash Attention decode (single-query)
      silu_mul.rs       -- SiLU * mul for MLP gating
      gemm.rs           -- tiled GEMM, GEMV, persistent GEMM
      gemm_dispatch.rs  -- shape-based routing (M=1 -> GEMV, etc.)
```

## Dependencies

- `half` (f16)
- `cudarc` (CUDA driver API) -- optional, behind `cuda` feature
- `serde` + `serde_json` (autotune cache)
- `parking_lot` (thread-safe caches)
- `tracing` (logging, matches workspace)
- `thiserror` (error types)

## Module Specs

### 1. ir.rs -- Core IR

Types:
- `ScalarType`: F16, BF16, F32, F64, F8E4M3, F8E5M2, I8, I16, I32, I64, Bool
- `Type`: Scalar(ScalarType), Tensor { scalar: ScalarType, shape: Vec<u32> }, Pointer { pointee: ScalarType, address_space: AddrSpace }
- `AddrSpace`: Global, Shared, Generic
- `ValueId`: newtype u32, SSA value identifier
- `Value`: ValueId + Type
- `Op` enum (~30 variants):
  - Memory: Load { ptr, mask, other }, Store { ptr, val, mask }
  - Matmul: Dot { a, b, acc }
  - Binary: Add, Sub, Mul, Div, Rem, Max, Min, And, Or, Xor, Shl, Shr
  - Unary: Neg, Abs, Exp, Log, Sqrt, Rsqrt, Exp2, Log2
  - Compare: Eq, Ne, Lt, Le, Gt, Ge
  - Select { cond, true_val, false_val }
  - Cast { val, to_dtype }
  - Shape: Splat, Broadcast, ExpandDims, Reshape, Trans
  - Reduce { val, axis, op: ReduceOp }  (ReduceOp: Sum, Max, Min)
  - Index: MakeRange { start, end }, GetProgramId { axis }
  - Atomic: AtomicAdd { ptr, val, mask }
  - Async: AsyncCopy, AsyncCommit, AsyncWait { count }
  - Barrier
  - Constant { value: ConstVal }
  - AddPtr { ptr, offset }
- `Instruction`: { result: Option<ValueId>, op: Op }
- `Block`: { id: u32, args: Vec<Value>, instructions: Vec<Instruction> }
- `Function`: { name: String, args: Vec<Value>, blocks: Vec<Block>, constexprs: HashMap<String, u32> }
- `Module`: { functions: Vec<Function> }

### 2. builder.rs -- Frontend DSL

`KernelBuilder` struct with methods:
- `new(name)` -> Self
- `arg_ptr(name, dtype)` -> Value (kernel pointer arg)
- `arg_scalar(name, scalar_type)` -> Value (kernel scalar arg)
- `constexpr(name, default)` -> Value (compile-time constant)
- `program_id(axis)` -> Value
- `arange(start, end)` -> Value (1D range tensor)
- `load(ptr, mask, other)` -> Value
- `store(ptr, val, mask)`
- `dot(a, b, acc)` -> Value
- `add/sub/mul/div/maximum/minimum(a, b)` -> Value
- `exp/log/sqrt/rsqrt/exp2/log2/neg/abs(x)` -> Value
- `reduce_sum/reduce_max/reduce_min(val, axis)` -> Value
- `where_(cond, true_val, false_val)` -> Value
- `cast(val, dtype)` -> Value
- `broadcast_to(val, shape)` -> Value
- `reshape(val, shape)` -> Value
- `atomic_add(ptr, val, mask)` -> Value
- `add_ptr(ptr, offset)` -> Value
- `splat(val, shape)` -> Value
- `build()` -> Function

Auto SSA numbering, shape inference, broadcasting rules.

### 3. passes.rs -- Optimization Passes

```rust
trait Pass { fn name(&self) -> &str; fn run(&self, module: &mut Module); }
struct PassManager { passes: Vec<Box<dyn Pass>> }
```

Passes:
1. **FusionPass** -- merge consecutive elementwise ops with single-use edges
2. **CoalescePass** -- ensure N-dim (fastest varying) maps to threadIdx.x
3. **SmemAllocPass** -- insert shared memory allocs for Dot operands, compute sizes
4. **PipelinePass** -- software pipelining for num_stages > 1 (async copy + barriers)
5. **WarpSchedulePass** -- map block tiles to warp tiles
6. **DeadCodeElimPass** -- remove unused values

### 4. codegen.rs -- PTX Emission

`PtxCodegen` struct:
- Walks Function, emits PTX text
- Register allocation: per-type counters, placeholder pattern (declare at end)
- Kernel entry: `.visible .entry name(.param ...)`
- Global loads/stores: `ld.global`, `st.global` with predication for masks
- Shared memory: `.shared .align 16 .b8 smem[N]`
- Async: `cp.async.ca.shared.global`, `cp.async.wait_group`
- Math: `add.f16x2`, `mul.f16x2`, `fma.rn.f16x2`, `ex2.approx.f32`, `rsqrt.approx.f32`
- MMA: `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`
- Reductions: `shfl.sync.bfly.b32` + shared memory
- Thread/block IDs: `%tid.x`, `%ctaid.x`
- `compile_ptx_to_cubin(ptx, arch) -> Result<Vec<u8>>` via `ptxas`
- Target: sm_80, sm_89, sm_90

### 5. runtime.rs -- Kernel Loading & Launch (cfg cuda)

- `KernelArg` enum: DevicePtr(u64), I32(i32), U32(u32), F32(f32), U64(u64)
- `CompiledKernel`: ptx, cubin, name, smem, regs, block_dim, grid_fn, config
- `KernelCache`: thread-safe HashMap<(name, config) -> CUmodule/CUfunction>
- `launch(kernel, stream, args, problem_shape)` -> cuLaunchKernel
- `JitFunction`: compile-on-first-call + autotune + cache

### 6. autotune.rs -- Autotuner

- `Config`: { block_m, block_n, block_k, num_warps, num_stages, split_k, extras: HashMap }
- `Autotune`: candidates + kernel builder -> benchmark each -> return best
- `PersistentCache`: JSON at ~/.cache/rtriton/, keyed by (kernel, shape, device)
- `default_gemm_configs()`, `default_elementwise_configs()`, `default_reduction_configs()`
- `heuristic_config(shape, sm_count)` -- no-benchmark fast pick

### 7. graph.rs -- CUDA Graph Capture/Replay

- `BufferHandle`: u32 index into buffer pool
- `KernelCall`: function_id, launch_config, args (with buffer handles), read/write sets
- `KernelGraph`: ordered Vec<KernelCall> + Vec<BufferInfo> + allocation plan
- `GraphBuilder`: begin_capture -> allocate_buffer/launch_kernel -> end_capture
- `BufferPool`: single large alloc, subdivided by liveness analysis + interval coloring
- `CudaGraphWrapper`: cuStreamBeginCapture/EndCapture, cuGraphInstantiate, cuGraphLaunch
- `GraphExecutor`: first call captures + instantiates, subsequent calls update ptrs + replay

### 8. kernels/ -- Pre-built LLM Kernels

Each kernel is a Rust function `fn build_X(b: &mut KernelBuilder)` that constructs the IR.

**rmsnorm.rs**: per-row variance reduction, rsqrt, elementwise mul. Fused variant with residual add.
**rope.rs**: complex pair rotation on q/k. Grid over M*heads, tile over head_dim/2.
**fused_attention.rs**: flash attention decode. Per-head, iterate KV blocks, online softmax.
**silu_mul.rs**: split interleaved gate_up, sigmoid*gate*up.
**gemm.rs**: tiled GEMM (smem + MMA), GEMV (warp-parallel K-reduce + float4), persistent GEMM.
**gemm_dispatch.rs**: M=1->GEMV, M<=4->batch GEMV, M>=16->tiled GEMM, large->persistent.

## Build & Verify

```bash
# Must compile (possibly with mock-gpu on Mac):
cargo check -p rtriton
# With CUDA:
cargo check -p rtriton --features cuda
# Run examples (PTX generation, no GPU needed):
cargo run -p rtriton --example rmsnorm
cargo run -p rtriton --example llm_decode_layer
```

## Integration With rvllm

Future step (NOT in this PR): replace hand-written CUDA kernels in `rvllm-model-runner` with
rTriton JIT kernels. The decode forward pass becomes:

```
Layer N (T=1 decode, 9 kernel launches, 0 cudaMalloc):
  1. [fused] residual_add + rmsnorm        -- rTriton kernel
  2. [cublas] QKV GEMV                      -- cuBLAS (faster at M=1)
  3. [fused] RoPE + KV cache write          -- rTriton kernel
  4. [fused] Flash Attention decode          -- rTriton kernel
  5. [cublas] O-proj GEMV                   -- cuBLAS
  6. [fused] residual_add + rmsnorm         -- rTriton kernel
  7. [cublas] Gate+Up GEMV                  -- cuBLAS
  8. [fused] SiLU * mul                     -- rTriton kernel
  9. [cublas] Down GEMV                     -- cuBLAS
Captured as CUDA graph, replayed with cuGraphLaunch.
```

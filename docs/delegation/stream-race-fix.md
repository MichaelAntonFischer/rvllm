# Stream Race Fix -- 8-Agent Investigation

## Problem
Output is coherent ONLY when debug probes force `dtoh_sync_copy` (which implicitly synchronizes all CUDA streams). Without probes, output degrades to token 0 (`!`).

The forward pass has 4 `device.synchronize()` calls per layer in `gpu_layer.rs` plus 1 before LM head in `gpu_runner.rs`. These should be sufficient but aren't.

## Key facts
- cuBLAS is created via `CudaBlas::new(device)` -- no explicit stream binding
- Custom kernels launch via `device.get_func().launch()` -- uses default stream (stream 0)
- Final RMSNorm uses `CudaRMSNorm::forward(..., &self.stream)` with a FORKED stream
- `device.synchronize()` syncs ALL streams on the device
- The probes call `device.dtoh_sync_copy()` which also syncs the device

## Files involved
- `crates/rvllm-gpu/src/cublas.rs` -- cuBLAS wrapper, sgemm/hgemm/sgemm_nn
- `crates/rvllm-model-runner/src/gpu_runner.rs` -- forward(), embedding, final RMSNorm, LM head
- `crates/rvllm-model-runner/src/gpu_layer.rs` -- layer forward, 4 syncs per layer
- `crates/rvllm-model-runner/src/layers/norm_cuda.rs` -- CudaRMSNorm uses launch_on_stream
- `crates/rvllm-model-runner/src/layers/linear_cuda.rs` -- CudaLinearLayer uses cuBLAS
- `crates/rvllm-gpu/src/stream.rs` -- GpuStream wrapper

## Agent assignments

### Agent 1: cuBLAS stream audit
- READ: `crates/rvllm-gpu/src/cublas.rs`, cudarc 0.12.1 CudaBlas source
- QUESTION: Does `CudaBlas::new(device)` bind to stream 0 or a private stream?
- QUESTION: Does cudarc's `Gemm::gemm()` use the handle's stream or the default?
- OUTPUT: Which stream cuBLAS operations execute on

### Agent 2: Kernel launch stream audit
- READ: `crates/rvllm-model-runner/src/gpu_layer.rs` (all kernel launches)
- QUESTION: Which stream does `device.get_func(module, name).launch(cfg, args)` use?
- QUESTION: Which stream does `kernel.launch_on_stream(stream, cfg, args)` use?
- OUTPUT: Stream used by each kernel launch in the layer forward path

### Agent 3: CudaRMSNorm stream analysis
- READ: `crates/rvllm-model-runner/src/layers/norm_cuda.rs`
- READ: `crates/rvllm-model-runner/src/gpu_runner.rs` (final RMSNorm call)
- QUESTION: Why does CudaRMSNorm use `launch_on_stream` with a forked stream?
- QUESTION: Is the forked stream the same as the default stream?
- OUTPUT: Whether the final RMSNorm is the only stream mismatch or if per-layer RMSNorm also has one

### Agent 4: Sync placement analysis
- READ: `crates/rvllm-model-runner/src/gpu_layer.rs` (lines with synchronize)
- READ: `crates/rvllm-model-runner/src/gpu_runner.rs` (sync before LM head)
- QUESTION: Are the 5 syncs placed at the correct boundaries?
- QUESTION: Is there a missing sync between any two operations that use different streams?
- OUTPUT: Map of all stream transitions and whether each has a sync

### Agent 5: Probe effect analysis
- READ: `crates/rvllm-model-runner/src/gpu_runner.rs` (all eprintln/PROBE blocks)
- QUESTION: What CUDA operations do the probes trigger?
- QUESTION: `dtoh_sync_copy` syncs all streams -- which specific stream transition does this accidentally fix?
- OUTPUT: The exact operation boundary where the probe sync makes the difference

### Agent 6: cudarc stream internals
- READ: cudarc 0.12.1 source on remote: /root/.cargo/registry/src/*/cudarc-0.12.1/src/
- QUESTION: Does `CudaDevice::fork_default_stream()` create a new stream?
- QUESTION: Does `CudaFunction::launch()` (no stream arg) use stream 0?
- QUESTION: Does `CudaBlas` internally call `cublasSetStream`?
- ACCESS: ssh -p 16806 root@ssh4.vast.ai to read cudarc source
- OUTPUT: Definitive answer on which streams are used by which cudarc APIs

### Agent 7: Fix proposal -- bind cuBLAS to default stream
- READ: all findings from agents 1-6
- PROPOSE: Add `cublasSetStream(handle, stream_0)` after creating CudaBlas
- OR: Make all kernel launches use the same stream as cuBLAS
- OR: Remove the forked stream entirely (use stream 0 everywhere)
- OUTPUT: Concrete code change with file:line references

### Agent 8: Fix proposal -- remove all syncs and use single stream
- READ: all findings from agents 1-6
- PROPOSE: Remove the forked stream from GpuModelRunner
- PROPOSE: Change CudaRMSNorm to use `launch()` instead of `launch_on_stream()`
- PROPOSE: Remove all 5 `device.synchronize()` calls
- OUTPUT: Concrete code change that eliminates stream mismatches entirely

## Rules
1. Agents 1-6 are READ-ONLY
2. Agents 7-8 propose changes but DO NOT edit files
3. No agent touches the remote server
4. No benchmarking
5. Report exact file:line references

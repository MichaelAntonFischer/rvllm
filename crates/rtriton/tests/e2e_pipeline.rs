// End-to-end test: builder -> IR -> passes -> codegen -> PTX for every kernel.
// Also tests the full decode layer graph (mixed Triton + cuBLAS nodes).

use rtriton::builder::KernelBuilder;
use rtriton::codegen::{PtxCodegen, SmArch};
use rtriton::passes::PassManager;
use rtriton::cublas_gemm::{DecodeLayerPlan, H100Config};
use rtriton::graph::{GraphBuilder, GraphExecutor, LaunchConfig, KernelArg, GraphNode};

fn compile_kernel(name: &str, build_fn: fn(&mut KernelBuilder)) -> String {
    let mut b = KernelBuilder::new(name);
    build_fn(&mut b);
    let mut func = b.build();

    let pm = PassManager::default_pipeline();
    pm.run(&mut func);

    let ptx = PtxCodegen::compile(&func, SmArch::Sm80).unwrap();

    // Basic sanity checks on PTX output
    assert!(ptx.contains(".version 7.8"), "{name}: missing PTX version");
    assert!(ptx.contains(".target sm_80"), "{name}: missing target");
    assert!(ptx.contains(&format!(".visible .entry {name}")), "{name}: missing entry point");
    assert!(ptx.contains("ret;") || ptx.contains("st.global"), "{name}: no store or ret");
    assert!(!ptx.contains("REG_COUNT_"), "{name}: unfixed register placeholder");

    ptx
}

#[test]
fn rmsnorm_full_pipeline() {
    let ptx = compile_kernel("rmsnorm", rtriton::kernels::rmsnorm::build_rmsnorm);
    // Should have load, rsqrt, store
    assert!(ptx.contains("ld.global"), "rmsnorm: missing load");
    assert!(ptx.contains("rsqrt.approx"), "rmsnorm: missing rsqrt");
    assert!(ptx.contains("st.global"), "rmsnorm: missing store");
    // Should have warp shuffle reduction for sum(x^2)
    assert!(ptx.contains("shfl.sync.bfly"), "rmsnorm: missing warp shuffle");
}

#[test]
fn fused_residual_rmsnorm_full_pipeline() {
    let ptx = compile_kernel(
        "fused_residual_rmsnorm",
        rtriton::kernels::rmsnorm::build_fused_residual_rmsnorm,
    );
    // Should have at least 2 stores (new_residual + normed output)
    let store_count = ptx.matches("st.global").count();
    assert!(store_count >= 2, "fused_residual: expected >=2 stores, got {store_count}");
}

#[test]
fn rope_full_pipeline() {
    let ptx = compile_kernel("rope", rtriton::kernels::rope::build_rope);
    // RoPE does sin/cos rotation: loads, mul, sub, add, stores
    assert!(ptx.contains("ld.global"), "rope: missing load");
    assert!(ptx.contains("sub.f32"), "rope: missing sub");
    assert!(ptx.contains("st.global"), "rope: missing store");
}

#[test]
fn silu_mul_full_pipeline() {
    let ptx = compile_kernel("silu_mul", rtriton::kernels::silu_mul::build_silu_mul);
    // SiLU = x * sigmoid(x) = x * 1/(1+exp(-x))
    // Should have exp (via ex2.approx), div, mul
    assert!(ptx.contains("ex2.approx"), "silu_mul: missing exp2 (for sigmoid)");
    assert!(ptx.contains("div.rn.f32"), "silu_mul: missing div (for sigmoid)");
}

#[test]
fn tiled_gemm_full_pipeline() {
    let ptx = compile_kernel("tiled_gemm", rtriton::kernels::gemm::build_tiled_gemm);
    // Should have dot (MMA placeholder for now)
    assert!(ptx.contains("mma.sync") || ptx.contains("TODO: real MMA"),
        "tiled_gemm: missing MMA comment/instruction");
}

#[test]
fn gemv_full_pipeline() {
    let ptx = compile_kernel("gemv", rtriton::kernels::gemm::build_gemv);
    assert!(ptx.contains("shfl.sync.bfly"), "gemv: missing warp shuffle reduction");
}

#[test]
fn persistent_gemm_full_pipeline() {
    let ptx = compile_kernel("persistent_gemm", rtriton::kernels::gemm::build_persistent_gemm);
    assert!(ptx.contains("mma.sync") || ptx.contains("TODO: real MMA"),
        "persistent_gemm: missing MMA");
}

#[test]
fn flash_attention_decode_full_pipeline() {
    let ptx = compile_kernel(
        "flash_attention_decode",
        rtriton::kernels::fused_attention::build_flash_attention_decode,
    );
    // Online softmax: exp, max, sum reduction
    assert!(ptx.contains("ex2.approx"), "attention: missing exp");
    assert!(ptx.contains("max.f32"), "attention: missing max");
    assert!(ptx.contains("shfl.sync.bfly"), "attention: missing warp shuffle");
}

// ---------------------------------------------------------------------------
// Full decode layer graph test
// ---------------------------------------------------------------------------

#[test]
fn decode_layer_graph_llama_7b() {
    // Build a full decode layer graph for Llama-2 7B at c=128
    let plan = DecodeLayerPlan::build(128, 4096, 4096, 12288, 11008, 22016, false);

    let mut gb = GraphBuilder::new();

    // Allocate buffers matching the decode data flow
    let hidden = gb.allocate_buffer(128 * 4096 * 2);      // f16 hidden state
    let normed = gb.allocate_buffer(128 * 4096 * 2);
    let qkv = gb.allocate_buffer(128 * 12288 * 2);
    let attn_out = gb.allocate_buffer(128 * 4096 * 2);
    let normed2 = gb.allocate_buffer(128 * 4096 * 2);
    let gate_up = gb.allocate_buffer(128 * 22016 * 2);
    let mlp_out = gb.allocate_buffer(128 * 11008 * 2);
    let down_out = gb.allocate_buffer(128 * 4096 * 2);

    gb.mark_input(hidden);
    gb.mark_output(down_out);

    let tcfg = LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1), smem_bytes: 0 };

    for op in &plan.ops {
        match op {
            rtriton::cublas_gemm::LayerOp::TritonKernel { name } => {
                match name.as_str() {
                    "fused_residual_rmsnorm" => {
                        gb.launch_kernel(name, tcfg.clone(), vec![], vec![hidden], vec![normed]);
                    }
                    "rope_kv_write" => {
                        gb.launch_kernel(name, tcfg.clone(), vec![], vec![qkv], vec![qkv]);
                    }
                    "flash_attention_decode" => {
                        gb.launch_kernel(name, tcfg.clone(), vec![], vec![qkv], vec![attn_out]);
                    }
                    "silu_mul" => {
                        gb.launch_kernel(name, tcfg.clone(), vec![], vec![gate_up], vec![mlp_out]);
                    }
                    _ => {}
                }
            }
            rtriton::cublas_gemm::LayerOp::CublasGemm(gemm) => {
                match gemm.name.as_str() {
                    "qkv_proj" => {
                        gb.launch_cublas(
                            gemm.clone(), KernelArg::Buffer(normed),
                            KernelArg::ExternalPtr(0), KernelArg::Buffer(qkv),
                            vec![normed], vec![qkv],
                        );
                    }
                    "o_proj" => {
                        gb.launch_cublas(
                            gemm.clone(), KernelArg::Buffer(attn_out),
                            KernelArg::ExternalPtr(0), KernelArg::Buffer(normed2),
                            vec![attn_out], vec![normed2],
                        );
                    }
                    "gate_up_proj" => {
                        gb.launch_cublas(
                            gemm.clone(), KernelArg::Buffer(normed2),
                            KernelArg::ExternalPtr(0), KernelArg::Buffer(gate_up),
                            vec![normed2], vec![gate_up],
                        );
                    }
                    "down_proj" => {
                        gb.launch_cublas(
                            gemm.clone(), KernelArg::Buffer(mlp_out),
                            KernelArg::ExternalPtr(0), KernelArg::Buffer(down_out),
                            vec![mlp_out], vec![down_out],
                        );
                    }
                    _ => {}
                }
            }
        }
    }

    let graph = gb.build();
    assert_eq!(graph.nodes.len(), 9);

    // Count Triton vs cuBLAS nodes
    let triton_count = graph.nodes.iter().filter(|n| matches!(n, GraphNode::Triton(_))).count();
    let cublas_count = graph.nodes.iter().filter(|n| matches!(n, GraphNode::Cublas(_))).count();
    assert_eq!(triton_count, 5);
    assert_eq!(cublas_count, 4);

    // Buffer allocation should reuse memory for non-overlapping lifetimes
    let alloc = graph.compute_allocation_plan();
    let naive_total: usize = graph.buffers.iter().map(|b| b.size_bytes).sum();
    println!("naive total: {} bytes, optimized: {} bytes, saved: {:.0}%",
        naive_total, alloc.total_bytes,
        (1.0 - alloc.total_bytes as f64 / naive_total as f64) * 100.0);
    assert!(alloc.total_bytes < naive_total, "allocator should save memory");

    let exec = GraphExecutor::new(graph);
    exec.execute_mock();
}

#[test]
fn h100_batch_analysis() {
    let h100 = H100Config::sxm();

    // At what batch size does QKV projection become compute-bound?
    let threshold = h100.compute_bound_batch(12288, 4096, rtriton::cublas_gemm::GemmDtype::F16);
    println!("QKV proj f16 compute-bound at M >= {threshold}");
    assert!(threshold > 1);
    assert!(threshold <= 512);

    // FP8 should become compute-bound at lower M (2x throughput, same bandwidth)
    let threshold_fp8 = h100.compute_bound_batch(12288, 4096, rtriton::cublas_gemm::GemmDtype::Fp8E4M3);
    println!("QKV proj fp8 compute-bound at M >= {threshold_fp8}");
    // FP8 has half the bytes but 2x the peak flops, so threshold should be lower
    assert!(threshold_fp8 <= threshold);
}

// rTriton cuBLAS GEMM integration.
//
// Brings our cuBLAS tricks (FP8 plan cache, autotuning, graph workspace,
// warmup, M-threshold routing) into rTriton so the graph system can
// capture mixed Triton+cuBLAS decode passes as a single CUDA graph.

use std::collections::HashMap;

/// GEMM data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmDtype {
    F16,
    F32,
    Fp8E4M3,
}

impl GemmDtype {
    pub fn element_bytes(&self) -> usize {
        match self {
            GemmDtype::F16 => 2,
            GemmDtype::F32 => 4,
            GemmDtype::Fp8E4M3 => 1,
        }
    }
}

/// A GEMM operation descriptor -- everything needed to launch a cuBLAS call.
/// Row-major: C[m,n] = alpha * A[m,k] @ B[n,k]^T + beta * C[m,n]
#[derive(Debug, Clone)]
pub struct GemmOp {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub alpha: f32,
    pub beta: f32,
    pub input_dtype: GemmDtype,
    pub output_dtype: GemmDtype,
    pub name: String,
}

impl GemmOp {
    /// Standard f16 HGEMM for decode/prefill.
    pub fn hgemm(name: &str, m: usize, n: usize, k: usize) -> Self {
        Self {
            m, n, k,
            alpha: 1.0,
            beta: 0.0,
            input_dtype: GemmDtype::F16,
            output_dtype: GemmDtype::F16,
            name: name.to_owned(),
        }
    }

    /// FP8 GEMM: fp8 inputs, f16 output. Best for M=1 decode via cublasLt.
    pub fn fp8_gemm(name: &str, m: usize, n: usize, k: usize) -> Self {
        Self {
            m, n, k,
            alpha: 1.0,
            beta: 0.0,
            input_dtype: GemmDtype::Fp8E4M3,
            output_dtype: GemmDtype::F16,
            name: name.to_owned(),
        }
    }

    /// HGEMM with f32 accumulation output (eliminates post-GEMM cast).
    pub fn hgemm_f32out(name: &str, m: usize, n: usize, k: usize) -> Self {
        Self {
            m, n, k,
            alpha: 1.0,
            beta: 0.0,
            input_dtype: GemmDtype::F16,
            output_dtype: GemmDtype::F32,
            name: name.to_owned(),
        }
    }

    pub fn shape_key(&self) -> (usize, usize, usize) {
        (self.m, self.n, self.k)
    }

    /// Bytes read/written for roofline analysis.
    pub fn bytes_transferred(&self) -> usize {
        let in_bytes = self.input_dtype.element_bytes();
        let out_bytes = self.output_dtype.element_bytes();
        // A[m,k] + B[n,k] reads + C[m,n] write
        self.m * self.k * in_bytes + self.n * self.k * in_bytes + self.m * self.n * out_bytes
    }

    /// FLOPs (2*M*N*K for GEMM).
    pub fn flops(&self) -> usize {
        2 * self.m * self.n * self.k
    }

    /// Arithmetic intensity (flops/byte). Tells you if mem-bound or compute-bound.
    pub fn arithmetic_intensity(&self) -> f64 {
        self.flops() as f64 / self.bytes_transferred() as f64
    }
}

/// M-threshold: use cublasLt for M <= this (decode path, split-K heuristics).
/// Standard cuBLAS for M > this (prefill, lower overhead).
pub const CUBLASLT_M_THRESHOLD: usize = 32;

/// Graph workspace size. cuBLAS needs pre-allocated workspace inside graph capture.
pub const GRAPH_WORKSPACE_BYTES: usize = 4 * 1024 * 1024;

/// FP8 workspace for cublasLt split-K heuristics.
pub const FP8_WORKSPACE_BYTES: usize = 4 * 1024 * 1024;

/// Autotuning constants.
pub const WARMUP_ITERS: usize = 3;
pub const BENCH_ITERS: usize = 10;
pub const MAX_ALGOS: usize = 32;

/// Cached autotuned algorithm result.
#[derive(Debug, Clone)]
pub struct AutotunedAlgo {
    pub workspace_size: usize,
    pub time_us: f64,
    // On GPU: also stores cublasLtMatmulAlgo_t (opaque, behind cfg)
}

/// GEMM engine state (plan cache, autotuner results, workspace).
/// This is the rTriton-owned version of what rvllm-gpu splits across
/// CublasHandle + CublasLtOps + CublasAutotuner.
pub struct GemmEngine {
    /// Autotuned results keyed by (m, n, k).
    autotuned: HashMap<(usize, usize, usize), AutotunedAlgo>,
    /// Which shapes have been warmed up (cuBLAS internal algo cache populated).
    warmed_shapes: Vec<(usize, usize, usize)>,
}

impl GemmEngine {
    pub fn new() -> Self {
        Self {
            autotuned: HashMap::new(),
            warmed_shapes: Vec::new(),
        }
    }

    /// Whether cublasLt should be used for this shape (vs standard cuBLAS).
    pub fn should_use_lt(&self, m: usize) -> bool {
        m <= CUBLASLT_M_THRESHOLD
    }

    /// Register an autotuned result for a shape.
    pub fn set_autotuned(&mut self, m: usize, n: usize, k: usize, algo: AutotunedAlgo) {
        self.autotuned.insert((m, n, k), algo);
    }

    /// Get autotuned result for a shape.
    pub fn get_autotuned(&self, m: usize, n: usize, k: usize) -> Option<&AutotunedAlgo> {
        self.autotuned.get(&(m, n, k))
    }

    /// Max workspace across all autotuned algos.
    pub fn max_workspace(&self) -> usize {
        self.autotuned.values().map(|a| a.workspace_size).max().unwrap_or(0)
    }

    /// Mark shapes as warmed.
    pub fn mark_warmed(&mut self, shapes: &[(usize, usize, usize)]) {
        self.warmed_shapes.extend_from_slice(shapes);
    }

    pub fn is_warmed(&self, m: usize, n: usize, k: usize) -> bool {
        self.warmed_shapes.contains(&(m, n, k))
    }

    /// All standard GEMM shapes for a transformer model at given batch sizes.
    /// Returns (m, n, k) tuples for QKV, O-proj, gate_up, down at each batch size.
    pub fn model_shapes(
        batch_sizes: &[usize],
        hidden: usize,
        q_dim: usize,
        qkv_dim: usize,
        intermediate: usize,
        gate_up_dim: usize,
    ) -> Vec<(usize, usize, usize)> {
        let mut shapes = Vec::new();
        for &m in batch_sizes {
            shapes.push((m, qkv_dim, hidden));      // QKV projection
            shapes.push((m, hidden, q_dim));          // O-proj
            shapes.push((m, gate_up_dim, hidden));    // gate_up
            shapes.push((m, hidden, intermediate));   // down
        }
        shapes
    }
}

impl Default for GemmEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Decode layer plan -- the full T=1 (or batched) decode as a sequence of ops
// ---------------------------------------------------------------------------

/// One step in a decode layer execution plan.
#[derive(Debug, Clone)]
pub enum LayerOp {
    /// rTriton JIT kernel (fused pointwise/reduction).
    TritonKernel {
        name: String,
    },
    /// cuBLAS GEMM with all our tricks.
    CublasGemm(GemmOp),
}

/// A complete decode layer plan: the 9-operation sequence that gets
/// captured as a single CUDA graph.
///
/// For c=64 or c=128 batch decode on H100 SXM:
///   [Triton] fused_residual_rmsnorm
///   [cuBLAS] QKV GEMV/GEMM        (M=batch_size)
///   [Triton] RoPE + KV cache write
///   [Triton] Flash Attention Decode
///   [cuBLAS] O-proj GEMV/GEMM
///   [Triton] fused_residual_rmsnorm
///   [cuBLAS] gate_up GEMV/GEMM
///   [Triton] SiLU * mul
///   [cuBLAS] down GEMV/GEMM
pub struct DecodeLayerPlan {
    pub ops: Vec<LayerOp>,
    pub batch_size: usize,
}

impl DecodeLayerPlan {
    /// Build the decode layer plan for a given model config and batch size.
    ///
    /// `use_fp8`: whether to use FP8 GEMMs (single-stream only, not batched).
    pub fn build(
        batch_size: usize,
        hidden: usize,
        q_dim: usize,
        qkv_dim: usize,
        intermediate: usize,
        gate_up_dim: usize,
        use_fp8: bool,
    ) -> Self {
        let gemm = |name: &str, n: usize, k: usize| -> LayerOp {
            if use_fp8 && batch_size == 1 {
                LayerOp::CublasGemm(GemmOp::fp8_gemm(name, batch_size, n, k))
            } else {
                LayerOp::CublasGemm(GemmOp::hgemm(name, batch_size, n, k))
            }
        };

        let ops = vec![
            LayerOp::TritonKernel { name: "fused_residual_rmsnorm".into() },
            gemm("qkv_proj", qkv_dim, hidden),
            LayerOp::TritonKernel { name: "rope_kv_write".into() },
            LayerOp::TritonKernel { name: "flash_attention_decode".into() },
            gemm("o_proj", hidden, q_dim),
            LayerOp::TritonKernel { name: "fused_residual_rmsnorm".into() },
            gemm("gate_up_proj", gate_up_dim, hidden),
            LayerOp::TritonKernel { name: "silu_mul".into() },
            gemm("down_proj", hidden, intermediate),
        ];

        Self { ops, batch_size }
    }

    /// All GEMM shapes in this plan (for warmup/autotuning).
    pub fn gemm_shapes(&self) -> Vec<(usize, usize, usize)> {
        self.ops.iter().filter_map(|op| {
            if let LayerOp::CublasGemm(g) = op {
                Some(g.shape_key())
            } else {
                None
            }
        }).collect()
    }

    /// Total bytes transferred per layer (all ops combined).
    pub fn total_bytes(&self) -> usize {
        self.ops.iter().map(|op| {
            match op {
                LayerOp::CublasGemm(g) => g.bytes_transferred(),
                LayerOp::TritonKernel { .. } => 0, // negligible vs GEMMs
            }
        }).sum()
    }

    /// Total FLOPs per layer.
    pub fn total_flops(&self) -> usize {
        self.ops.iter().map(|op| {
            match op {
                LayerOp::CublasGemm(g) => g.flops(),
                LayerOp::TritonKernel { .. } => 0,
            }
        }).sum()
    }

    /// Print the plan.
    pub fn dump(&self) {
        println!("DecodeLayerPlan (batch={})", self.batch_size);
        for (i, op) in self.ops.iter().enumerate() {
            match op {
                LayerOp::TritonKernel { name } => {
                    println!("  [{}] [Triton] {}", i, name);
                }
                LayerOp::CublasGemm(g) => {
                    let ai = g.arithmetic_intensity();
                    let bound = if ai < 100.0 { "mem-bound" } else { "compute-bound" };
                    println!(
                        "  [{}] [cuBLAS] {} M={} N={} K={} {:?}->{:?} AI={:.1} ({})",
                        i, g.name, g.m, g.n, g.k,
                        g.input_dtype, g.output_dtype,
                        ai, bound,
                    );
                }
            }
        }
        let gb = self.total_bytes() as f64 / 1e9;
        let tflops = self.total_flops() as f64 / 1e12;
        println!("  total: {:.3} GB, {:.3} TFLOP", gb, tflops);
    }
}

// ---------------------------------------------------------------------------
// H100 SXM batching limits
// ---------------------------------------------------------------------------

/// H100 SXM specs for batch sizing.
pub struct H100Config {
    /// HBM3 bandwidth (bytes/sec).
    pub mem_bw: f64,
    /// FP16 tensor core throughput (FLOP/sec).
    pub fp16_tflops: f64,
    /// FP8 tensor core throughput (FLOP/sec).
    pub fp8_tflops: f64,
    /// Number of SMs.
    pub num_sm: u32,
    /// HBM capacity (bytes).
    pub hbm_bytes: usize,
}

impl H100Config {
    pub fn sxm() -> Self {
        Self {
            mem_bw: 3.35e12,       // 3.35 TB/s
            fp16_tflops: 989.5e12, // ~990 TFLOP/s
            fp8_tflops: 1979e12,   // ~1.98 PFLOP/s
            num_sm: 132,
            hbm_bytes: 80 * 1024 * 1024 * 1024, // 80 GB
        }
    }

    /// Max batch size before a GEMM becomes compute-bound (AI > peak flops/bw).
    /// At this point, adding more batch doesn't help -- you're saturating tensor cores.
    pub fn compute_bound_batch(&self, n: usize, k: usize, dtype: GemmDtype) -> usize {
        let peak = match dtype {
            GemmDtype::Fp8E4M3 => self.fp8_tflops,
            _ => self.fp16_tflops,
        };
        let elem = dtype.element_bytes();
        // AI = 2*m*n*k / (m*k*e + n*k*e + m*n*e_out)
        // For large m: AI ~ 2*n*k / (k*e + n*e + n*e_out) ~ 2*n*k / ((k+2n)*e)
        // Compute-bound when AI > peak/bw
        let threshold = peak / self.mem_bw;
        // Solve: 2*m*n*k / ((m*k + n*k + m*n)*e) = threshold
        // 2*m*n*k = threshold * e * (m*k + n*k + m*n)
        // For m: 2*m*n*k = threshold*e*m*k + threshold*e*n*k + threshold*e*m*n
        // m*(2*n*k - threshold*e*k - threshold*e*n) = threshold*e*n*k
        let te = threshold * elem as f64;
        let denom = 2.0 * n as f64 * k as f64 - te * k as f64 - te * n as f64;
        if denom <= 0.0 {
            return 1; // always compute-bound at this shape
        }
        let m = te * n as f64 * k as f64 / denom;
        m.ceil() as usize
    }
}

impl Default for H100Config {
    fn default() -> Self {
        Self::sxm()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_op_basics() {
        let op = GemmOp::hgemm("qkv", 1, 12288, 4096);
        assert_eq!(op.flops(), 2 * 1 * 12288 * 4096);
        assert_eq!(op.shape_key(), (1, 12288, 4096));
        assert!(op.arithmetic_intensity() < 10.0); // M=1 is always mem-bound
    }

    #[test]
    fn gemm_op_fp8() {
        let op = GemmOp::fp8_gemm("qkv", 1, 12288, 4096);
        assert_eq!(op.input_dtype, GemmDtype::Fp8E4M3);
        assert_eq!(op.output_dtype, GemmDtype::F16);
        // FP8 reads half the bytes of F16 for A and B
        let f16_op = GemmOp::hgemm("qkv", 1, 12288, 4096);
        assert!(op.bytes_transferred() < f16_op.bytes_transferred());
    }

    #[test]
    fn decode_layer_plan_llama_7b() {
        // Llama-2 7B config
        let plan = DecodeLayerPlan::build(
            1,          // batch_size
            4096,       // hidden
            4096,       // q_dim (num_heads * head_dim)
            12288,      // qkv_dim (q + k + v)
            11008,      // intermediate
            22016,      // gate_up_dim (2 * intermediate)
            false,      // use_fp8
        );
        assert_eq!(plan.ops.len(), 9);
        assert_eq!(plan.gemm_shapes().len(), 4);
        plan.dump();
    }

    #[test]
    fn decode_layer_plan_batch128() {
        let plan = DecodeLayerPlan::build(
            128,        // c=128 batch
            4096,
            4096,
            12288,
            11008,
            22016,
            false,
        );
        // At c=128, GEMMs are M=128 -- compute-bound on H100
        for op in &plan.ops {
            if let LayerOp::CublasGemm(g) = op {
                let ai = g.arithmetic_intensity();
                assert!(ai > 10.0, "{} AI={:.1} too low for M=128", g.name, ai);
            }
        }
    }

    #[test]
    fn h100_compute_bound_threshold() {
        let h100 = H100Config::sxm();
        // At M=1, N=4096, K=4096 with f16 -- should be mem-bound
        let threshold = h100.compute_bound_batch(4096, 4096, GemmDtype::F16);
        assert!(threshold > 1, "threshold={threshold}, M=1 should be mem-bound");
        // Large shapes should become compute-bound at reasonable batch
        assert!(threshold < 512, "threshold={threshold}, should be <512 for 4096x4096");
    }

    #[test]
    fn model_shapes_coverage() {
        let shapes = GemmEngine::model_shapes(
            &[1, 64, 128],
            4096,   // hidden
            4096,   // q_dim
            12288,  // qkv_dim
            11008,  // intermediate
            22016,  // gate_up_dim
        );
        // 3 batch sizes * 4 projections = 12 shapes
        assert_eq!(shapes.len(), 12);
    }

    #[test]
    fn engine_lt_routing() {
        let engine = GemmEngine::new();
        assert!(engine.should_use_lt(1));
        assert!(engine.should_use_lt(32));
        assert!(!engine.should_use_lt(33));
        assert!(!engine.should_use_lt(128));
    }
}

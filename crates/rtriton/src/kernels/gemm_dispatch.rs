/// GEMM dispatch: select the best strategy based on problem shape.
///
/// Routes to either rTriton JIT GEMM kernels or cuBLAS (with our tricks:
/// FP8 plan cache, autotuning, graph workspace, M-threshold routing).

use crate::cublas_gemm::{GemmOp, CUBLASLT_M_THRESHOLD};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmStrategy {
    /// M=1, rTriton GEMV kernel (PTX JIT).
    Gemv,
    /// M=2..4, batched GEMV approach.
    SmallBatchGemv,
    /// M>=16, standard tiled GEMM (rTriton PTX JIT).
    TiledGemm,
    /// Very large problems, persistent GEMM (stream-K style).
    PersistentGemm,
    /// M<=32, cuBLAS via cublasLt (split-K, autotuned algo).
    CublasLt,
    /// M>32, standard cuBLAS (tensor-op auto-select).
    CublasStandard,
    /// FP8 via cublasLt (cached plan, best for M=1 single-stream).
    CublasFp8,
}

/// Select GEMM strategy based on M, N, K dimensions.
///
/// Decision tree:
///   M=1 + fp8 available -> CublasFp8 (best single-token decode)
///   M<=32              -> CublasLt  (split-K heuristics win at small M)
///   M>32               -> CublasStandard (lower overhead for prefill)
///
/// rTriton JIT GEMMs (Gemv, TiledGemm, PersistentGemm) are available
/// but currently routed to only when prefer_triton=true, since cuBLAS
/// beats hand-written PTX at all shapes for GEMM.
pub fn select_strategy(m: usize, _n: usize, _k: usize, fp8: bool, prefer_triton: bool) -> GemmStrategy {
    if prefer_triton {
        if m == 1 {
            return GemmStrategy::Gemv;
        }
        if m <= 4 {
            return GemmStrategy::SmallBatchGemv;
        }
        let tiles = ((m + 127) / 128) * ((_n + 127) / 128);
        if tiles >= 256 && m * _n * _k > 128 * 128 * 4096 {
            return GemmStrategy::PersistentGemm;
        }
        return GemmStrategy::TiledGemm;
    }

    // cuBLAS path (default -- proven faster than Triton for all GEMM shapes)
    if m == 1 && fp8 {
        return GemmStrategy::CublasFp8;
    }
    if m <= CUBLASLT_M_THRESHOLD {
        return GemmStrategy::CublasLt;
    }
    GemmStrategy::CublasStandard
}

/// Create a GemmOp for the selected strategy.
pub fn make_gemm_op(name: &str, m: usize, n: usize, k: usize, strategy: GemmStrategy) -> GemmOp {
    match strategy {
        GemmStrategy::CublasFp8 => GemmOp::fp8_gemm(name, m, n, k),
        _ => GemmOp::hgemm(name, m, n, k),
    }
}

/// Return default autotune configs for rTriton JIT strategies.
pub fn default_configs_for(strategy: &GemmStrategy) -> Vec<crate::autotune::Config> {
    match strategy {
        GemmStrategy::Gemv | GemmStrategy::SmallBatchGemv => {
            let mut cfgs = Vec::new();
            for &bn in &[64u32, 128, 256] {
                for &bk in &[32u32, 64, 128] {
                    cfgs.push(crate::autotune::Config::new(1, bn, bk, 4, 1));
                }
            }
            cfgs
        }
        GemmStrategy::TiledGemm => crate::autotune::default_gemm_configs(),
        GemmStrategy::PersistentGemm => {
            let mut cfgs = Vec::new();
            for &bm in &[128u32, 256] {
                for &bn in &[128u32, 256] {
                    for &bk in &[32u32, 64] {
                        for &nw in &[4u32, 8] {
                            let c = crate::autotune::Config::new(bm, bn, bk, nw, 3);
                            if c.smem_bytes(2) <= 48 * 1024 {
                                cfgs.push(c);
                            }
                        }
                    }
                }
            }
            cfgs
        }
        // cuBLAS strategies don't use Triton configs
        GemmStrategy::CublasLt | GemmStrategy::CublasStandard | GemmStrategy::CublasFp8 => {
            vec![]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cublas_gemm::GemmDtype;

    #[test]
    fn dispatch_m1_fp8() {
        assert_eq!(select_strategy(1, 4096, 4096, true, false), GemmStrategy::CublasFp8);
    }

    #[test]
    fn dispatch_m1_f16() {
        assert_eq!(select_strategy(1, 4096, 4096, false, false), GemmStrategy::CublasLt);
    }

    #[test]
    fn dispatch_m32_cublas_lt() {
        assert_eq!(select_strategy(32, 4096, 4096, false, false), GemmStrategy::CublasLt);
    }

    #[test]
    fn dispatch_m64_cublas_standard() {
        assert_eq!(select_strategy(64, 4096, 4096, false, false), GemmStrategy::CublasStandard);
    }

    #[test]
    fn dispatch_m128_cublas_standard() {
        assert_eq!(select_strategy(128, 4096, 4096, false, false), GemmStrategy::CublasStandard);
    }

    #[test]
    fn dispatch_triton_gemv() {
        assert_eq!(select_strategy(1, 4096, 4096, false, true), GemmStrategy::Gemv);
    }

    #[test]
    fn dispatch_triton_persistent() {
        assert_eq!(select_strategy(8192, 8192, 4096, false, true), GemmStrategy::PersistentGemm);
    }

    #[test]
    fn make_fp8_op() {
        let op = make_gemm_op("qkv", 1, 12288, 4096, GemmStrategy::CublasFp8);
        assert_eq!(op.input_dtype, GemmDtype::Fp8E4M3);
    }

    #[test]
    fn make_hgemm_op() {
        let op = make_gemm_op("qkv", 128, 12288, 4096, GemmStrategy::CublasStandard);
        assert_eq!(op.input_dtype, GemmDtype::F16);
    }
}

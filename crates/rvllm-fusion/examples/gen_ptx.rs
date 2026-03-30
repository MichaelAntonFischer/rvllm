//! Generate JIT PTX files for all 4 fused kernel patterns at a model's shapes.
//!
//! Usage:
//!   cargo run --example gen_ptx -p rvllm-fusion
//!   cargo run --example gen_ptx -p rvllm-fusion -- --arch sm_90 --hidden 3584 --intermediate 18944
//!
//! Environment overrides:
//!   ARCH=sm_90  PTX_DIR=kernels/sm_90  HIDDEN=3584  INTERMEDIATE=18944

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use rvllm_fusion::dispatch::ModelShapes;
use rvllm_fusion::ir::{Dtype, FusedKernel, FusionOp};
use rvllm_fusion::ptx_emit::compile_fused_kernel_ptx;

fn make_kernel(ops: Vec<FusionOp>, out_shape: Vec<usize>) -> FusedKernel {
    FusedKernel {
        node_ids: (0..ops.len()).collect(),
        ops,
        output_shape: out_shape,
        dtype: Dtype::F16,
    }
}

struct Config {
    arch: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    ptx_dir: PathBuf,
}

impl Config {
    fn from_args() -> Self {
        let args: Vec<String> = env::args().collect();

        let mut arch = env::var("ARCH").unwrap_or_else(|_| "sm_90".into());
        let mut hidden: usize = env::var("HIDDEN").ok().and_then(|v| v.parse().ok()).unwrap_or(3584);
        let mut intermediate: usize = env::var("INTERMEDIATE").ok().and_then(|v| v.parse().ok()).unwrap_or(18944);
        let mut num_heads: usize = env::var("NUM_HEADS").ok().and_then(|v| v.parse().ok()).unwrap_or(28);
        let mut num_kv_heads: usize = env::var("NUM_KV_HEADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        let mut head_dim: usize = env::var("HEAD_DIM").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
        let mut vocab_size: usize = env::var("VOCAB_SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(152064);
        let mut eps: f32 = env::var("RMS_EPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-6);
        let mut ptx_dir: Option<String> = env::var("PTX_DIR").ok();

        // Parse CLI args (override env)
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--arch" if i + 1 < args.len() => { arch = args[i + 1].clone(); i += 2; }
                "--hidden" if i + 1 < args.len() => { hidden = args[i + 1].parse().unwrap(); i += 2; }
                "--intermediate" if i + 1 < args.len() => { intermediate = args[i + 1].parse().unwrap(); i += 2; }
                "--num-heads" if i + 1 < args.len() => { num_heads = args[i + 1].parse().unwrap(); i += 2; }
                "--num-kv-heads" if i + 1 < args.len() => { num_kv_heads = args[i + 1].parse().unwrap(); i += 2; }
                "--head-dim" if i + 1 < args.len() => { head_dim = args[i + 1].parse().unwrap(); i += 2; }
                "--vocab-size" if i + 1 < args.len() => { vocab_size = args[i + 1].parse().unwrap(); i += 2; }
                "--eps" if i + 1 < args.len() => { eps = args[i + 1].parse().unwrap(); i += 2; }
                "--ptx-dir" if i + 1 < args.len() => { ptx_dir = Some(args[i + 1].clone()); i += 2; }
                "--help" | "-h" => {
                    eprintln!("Usage: gen_ptx [--arch sm_XX] [--hidden N] [--intermediate N] [--ptx-dir DIR]");
                    eprintln!("  Defaults: Qwen2.5-7B shapes, sm_90");
                    std::process::exit(0);
                }
                _ => { i += 1; }
            }
        }

        let dir = ptx_dir.unwrap_or_else(|| format!("kernels/{}", arch));

        Config {
            arch,
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            rms_norm_eps: eps,
            ptx_dir: PathBuf::from(dir),
        }
    }

    fn model_shapes(&self) -> ModelShapes {
        ModelShapes {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            vocab_size: self.vocab_size,
            rms_norm_eps: self.rms_norm_eps,
        }
    }
}

struct PatternDef {
    name: &'static str,
    ops: Vec<FusionOp>,
    out_shape_fn: fn(&Config) -> Vec<usize>,
}

fn patterns() -> Vec<PatternDef> {
    vec![
        PatternDef {
            name: "rmsnorm_gemv",
            ops: vec![FusionOp::RMSNorm { eps: 1e-6 }, FusionOp::Gemv],
            // QKV projection: hidden -> qkv_dim
            out_shape_fn: |c| vec![1, c.num_heads * c.head_dim + 2 * c.num_kv_heads * c.head_dim],
        },
        PatternDef {
            name: "silu_elemmul_gemv",
            ops: vec![FusionOp::SiLU, FusionOp::ElemMul, FusionOp::Gemv],
            // Down projection: intermediate -> hidden
            out_shape_fn: |c| vec![1, c.hidden_size],
        },
        PatternDef {
            name: "elemadd_rmsnorm",
            ops: vec![FusionOp::ElemAdd, FusionOp::RMSNorm { eps: 1e-6 }],
            out_shape_fn: |c| vec![1, c.hidden_size],
        },
        PatternDef {
            name: "elemadd_rmsnorm_gemv",
            ops: vec![FusionOp::ElemAdd, FusionOp::RMSNorm { eps: 1e-6 }, FusionOp::Gemv],
            // Gate+up projection: hidden -> intermediate*2
            out_shape_fn: |c| vec![1, c.intermediate_size * 2],
        },
    ]
}

fn main() {
    let cfg = Config::from_args();
    let shapes = cfg.model_shapes();

    println!("JIT PTX Generator");
    println!("  arch:          {}", cfg.arch);
    println!("  hidden:        {}", cfg.hidden_size);
    println!("  intermediate:  {}", cfg.intermediate_size);
    println!("  heads:         {}/{} (dim={})", cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    println!("  vocab:         {}", cfg.vocab_size);
    println!("  eps:           {}", cfg.rms_norm_eps);
    println!("  output dir:    {}", cfg.ptx_dir.display());
    println!();

    fs::create_dir_all(&cfg.ptx_dir).expect("failed to create PTX output directory");

    let mut total_bytes = 0usize;
    let mut generated = 0u32;
    let mut failed = 0u32;
    let t_total = Instant::now();

    for pat in patterns() {
        let out_shape = (pat.out_shape_fn)(&cfg);
        let kernel = make_kernel(pat.ops, out_shape);

        let t0 = Instant::now();
        match compile_fused_kernel_ptx(&kernel, &shapes, &cfg.arch) {
            Some(ptx) => {
                let ptx_bytes = ptx.as_bytes();
                let filename = format!("jit_{}.ptx", pat.name);
                let path = cfg.ptx_dir.join(&filename);
                fs::write(&path, ptx_bytes).expect("failed to write PTX file");

                let size = ptx_bytes.len();
                total_bytes += size;
                generated += 1;

                println!(
                    "  [OK] {:<30} {:>6} bytes  ({:.1}ms)",
                    filename,
                    size,
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }
            None => {
                failed += 1;
                println!("  [SKIP] {:<28} unsupported pattern", pat.name);
            }
        }
    }

    println!();
    println!("Summary: {} generated, {} skipped, {} bytes total in {:.1}ms",
        generated, failed, total_bytes, t_total.elapsed().as_secs_f64() * 1000.0);

    if generated == 0 {
        eprintln!("ERROR: no PTX files generated");
        std::process::exit(1);
    }
}

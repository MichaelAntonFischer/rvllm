// rTriton PTX code generator
//
// Walks the IR and emits PTX assembly text targeting sm_80+.

use crate::ir::*;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

#[derive(Debug, Clone, Copy)]
pub enum SmArch {
    Sm80,
    Sm89,
    Sm90,
}

impl SmArch {
    fn target(&self) -> &'static str {
        match self {
            SmArch::Sm80 => "sm_80",
            SmArch::Sm89 => "sm_89",
            SmArch::Sm90 => "sm_90",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    #[error("unsupported op: {0}")]
    UnsupportedOp(String),
    #[error("ptxas failed: {0}")]
    PtxasFailed(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

// Register classes mirroring PTX register types.
#[derive(Debug, Clone, Copy)]
enum RegClass {
    R,  // .u32 / .s32
    Rd, // .u64 (pointers)
    F,  // .f32
    H,  // .f16
    P,  // .pred
}

impl RegClass {
    fn prefix(&self) -> &'static str {
        match self {
            RegClass::R => "%r",
            RegClass::Rd => "%rd",
            RegClass::F => "%f",
            RegClass::H => "%h",
            RegClass::P => "%p",
        }
    }
    #[allow(dead_code)]
    fn decl_type(&self) -> &'static str {
        match self {
            RegClass::R => ".u32",
            RegClass::Rd => ".u64",
            RegClass::F => ".f32",
            RegClass::H => ".f16",
            RegClass::P => ".pred",
        }
    }
}

pub struct PtxCodegen {
    out: String,
    reg_counters: [u32; 5], // R, Rd, F, H, P
    reg_map: HashMap<ValueId, (RegClass, u32)>,
    indent: usize,
}

impl PtxCodegen {
    fn new() -> Self {
        Self {
            out: String::with_capacity(8192),
            reg_counters: [0; 5],
            reg_map: HashMap::new(),
            indent: 0,
        }
    }

    fn class_idx(c: RegClass) -> usize {
        match c {
            RegClass::R => 0,
            RegClass::Rd => 1,
            RegClass::F => 2,
            RegClass::H => 3,
            RegClass::P => 4,
        }
    }

    fn alloc_reg(&mut self, class: RegClass) -> (RegClass, u32) {
        let idx = Self::class_idx(class);
        let n = self.reg_counters[idx];
        self.reg_counters[idx] += 1;
        (class, n)
    }

    fn assign(&mut self, vid: ValueId, class: RegClass) -> String {
        let reg = self.alloc_reg(class);
        self.reg_map.insert(vid, reg);
        self.reg_name(reg)
    }

    fn reg_name(&self, reg: (RegClass, u32)) -> String {
        format!("{}{}", reg.0.prefix(), reg.1)
    }

    fn get(&self, vid: ValueId) -> String {
        match self.reg_map.get(&vid) {
            Some(r) => self.reg_name(*r),
            None => format!("/*UNK:{:?}*/", vid),
        }
    }

    fn get_class(&self, vid: ValueId) -> RegClass {
        self.reg_map.get(&vid).map_or(RegClass::F, |r| r.0)
    }

    fn emit_line(&mut self, line: &str) {
        for _ in 0..self.indent {
            self.out.push('\t');
        }
        self.out.push_str(line);
        self.out.push('\n');
    }

    fn emit(&mut self, s: &str) {
        self.out.push_str(s);
    }

    pub fn compile(func: &Function, arch: SmArch) -> Result<String, CodegenError> {
        let mut cg = PtxCodegen::new();
        cg.emit_header(arch);
        cg.emit_kernel(func)?;
        cg.fixup_reg_decls();
        Ok(cg.out)
    }

    fn emit_header(&mut self, arch: SmArch) {
        let _ = writeln!(self.out, ".version 7.8");
        let _ = writeln!(self.out, ".target {}", arch.target());
        let _ = writeln!(self.out, ".address_size 64");
        let _ = writeln!(self.out);
    }

    fn emit_kernel(&mut self, func: &Function) -> Result<(), CodegenError> {
        // Collect args (Op::Arg instructions)
        let args: Vec<_> = func.body.instructions.iter().filter_map(|inst| {
            if let Op::Arg { name, dtype, is_ptr } = &inst.op {
                Some((inst.result, name.clone(), *dtype, *is_ptr))
            } else {
                None
            }
        }).collect();

        // Entry signature
        self.emit(&format!(".visible .entry {}(\n", func.name));
        for (i, (_, name, dtype, is_ptr)) in args.iter().enumerate() {
            let ty = if *is_ptr {
                ".u64"
            } else {
                match dtype {
                    ScalarType::F32 => ".f32",
                    ScalarType::I32 | ScalarType::U32 => ".u32",
                    ScalarType::I64 | ScalarType::U64 => ".u64",
                    _ => ".u32",
                }
            };
            let comma = if i + 1 < args.len() { "," } else { "" };
            let _ = writeln!(self.out, "\t.param {} param_{}{}",  ty, name, comma);
        }
        self.emit(")\n{\n");
        self.indent = 1;

        // Register declaration placeholders (replaced at end)
        self.emit_line(".reg .u32 %r<REG_COUNT_R>;");
        self.emit_line(".reg .u64 %rd<REG_COUNT_RD>;");
        self.emit_line(".reg .f32 %f<REG_COUNT_F>;");
        self.emit_line(".reg .f16 %h<REG_COUNT_H>;");
        self.emit_line(".reg .pred %p<REG_COUNT_P>;");
        self.emit_line("");

        // Load params into registers
        for (result, name, dtype, is_ptr) in &args {
            if let Some(vid) = result {
                if *is_ptr {
                    let r = self.assign(*vid, RegClass::Rd);
                    self.emit_line(&format!("ld.param.u64 {}, [param_{}];", r, name));
                } else {
                    match dtype {
                        ScalarType::F32 => {
                            let r = self.assign(*vid, RegClass::F);
                            self.emit_line(&format!("ld.param.f32 {}, [param_{}];", r, name));
                        }
                        ScalarType::I64 | ScalarType::U64 => {
                            let r = self.assign(*vid, RegClass::Rd);
                            self.emit_line(&format!("ld.param.u64 {}, [param_{}];", r, name));
                        }
                        _ => {
                            let r = self.assign(*vid, RegClass::R);
                            self.emit_line(&format!("ld.param.u32 {}, [param_{}];", r, name));
                        }
                    }
                }
            }
        }
        self.emit_line("");

        // Emit each instruction
        for inst in &func.body.instructions {
            self.emit_instruction(inst)?;
        }

        self.indent = 0;
        self.emit("}\n");
        Ok(())
    }

    fn emit_instruction(&mut self, inst: &Instruction) -> Result<(), CodegenError> {
        match &inst.op {
            // Already handled in param loading
            Op::Arg { .. } => {}
            Op::ConstExpr { name, default } => {
                if let Some(vid) = inst.result {
                    let r = self.assign(vid, RegClass::R);
                    self.emit_line(&format!("mov.u32 {}, {};  // constexpr {}", r, default, name));
                }
            }
            Op::ProgramId { axis } => {
                if let Some(vid) = inst.result {
                    let r = self.assign(vid, RegClass::R);
                    let dim = match axis { 0 => "x", 1 => "y", _ => "z" };
                    self.emit_line(&format!("mov.u32 {}, %ctaid.{};", r, dim));
                }
            }
            Op::Arange { start, .. } => {
                if let Some(vid) = inst.result {
                    let r = self.assign(vid, RegClass::R);
                    let tmp = self.alloc_reg(RegClass::R);
                    let tmp_name = self.reg_name(tmp);
                    self.emit_line(&format!("mov.u32 {}, %tid.x;", tmp_name));
                    if *start != 0 {
                        self.emit_line(&format!("add.u32 {}, {}, {};", r, tmp_name, start));
                    } else {
                        self.emit_line(&format!("mov.u32 {}, {};", r, tmp_name));
                    }
                }
            }
            Op::ConstantF32 { val } => {
                if let Some(vid) = inst.result {
                    let r = self.assign(vid, RegClass::F);
                    let bits = val.to_bits();
                    self.emit_line(&format!("mov.f32 {}, 0f{:08X};", r, bits));
                }
            }
            Op::ConstantI32 { val } => {
                if let Some(vid) = inst.result {
                    let r = self.assign(vid, RegClass::R);
                    self.emit_line(&format!("mov.u32 {}, {};", r, val));
                }
            }

            // Binary arithmetic (default to f32)
            Op::Add { a, b } => self.emit_binary_f32("add", inst.result, *a, *b),
            Op::Sub { a, b } => self.emit_binary_f32("sub", inst.result, *a, *b),
            Op::Mul { a, b } => self.emit_binary_f32("mul", inst.result, *a, *b),
            Op::Div { a, b } => self.emit_binary_f32("div.rn", inst.result, *a, *b),
            Op::Maximum { a, b } => self.emit_binary_f32("max", inst.result, *a, *b),
            Op::Minimum { a, b } => self.emit_binary_f32("min", inst.result, *a, *b),

            // Unary math
            Op::Neg { x } => {
                if let Some(vid) = inst.result {
                    let r = self.assign(vid, RegClass::F);
                    self.emit_line(&format!("neg.f32 {}, {};", r, self.get(*x)));
                }
            }
            Op::Abs { x } => {
                if let Some(vid) = inst.result {
                    let r = self.assign(vid, RegClass::F);
                    self.emit_line(&format!("abs.f32 {}, {};", r, self.get(*x)));
                }
            }
            Op::Exp { x } => {
                // exp(x) = 2^(x * log2(e))
                if let Some(vid) = inst.result {
                    let tmp = self.alloc_reg(RegClass::F);
                    let tmp_n = self.reg_name(tmp);
                    let r = self.assign(vid, RegClass::F);
                    // log2(e) = 1.4426950408... = 0x3FB8AA3B
                    self.emit_line(&format!("mul.f32 {}, {}, 0f3FB8AA3B;", tmp_n, self.get(*x)));
                    self.emit_line(&format!("ex2.approx.f32 {}, {};", r, tmp_n));
                }
            }
            Op::Log { x } => {
                // log(x) = ln(2) * log2(x)
                if let Some(vid) = inst.result {
                    let tmp = self.alloc_reg(RegClass::F);
                    let tmp_n = self.reg_name(tmp);
                    let r = self.assign(vid, RegClass::F);
                    self.emit_line(&format!("lg2.approx.f32 {}, {};", tmp_n, self.get(*x)));
                    // ln(2) = 0.6931471805... = 0x3F317218
                    self.emit_line(&format!("mul.f32 {}, {}, 0f3F317218;", r, tmp_n));
                }
            }
            Op::Sqrt { x } => {
                if let Some(vid) = inst.result {
                    let r = self.assign(vid, RegClass::F);
                    self.emit_line(&format!("sqrt.rn.f32 {}, {};", r, self.get(*x)));
                }
            }
            Op::Rsqrt { x } => {
                if let Some(vid) = inst.result {
                    let r = self.assign(vid, RegClass::F);
                    self.emit_line(&format!("rsqrt.approx.f32 {}, {};", r, self.get(*x)));
                }
            }

            // Comparisons
            Op::CmpLt { a, b } => self.emit_setp("lt", inst.result, *a, *b),
            Op::CmpGt { a, b } => self.emit_setp("gt", inst.result, *a, *b),
            Op::CmpEq { a, b } => self.emit_setp("eq", inst.result, *a, *b),

            // Select
            Op::Where { cond, true_val, false_val } => {
                if let Some(vid) = inst.result {
                    let r = self.assign(vid, RegClass::F);
                    self.emit_line(&format!(
                        "selp.f32 {}, {}, {}, {};",
                        r, self.get(*true_val), self.get(*false_val), self.get(*cond)
                    ));
                }
            }

            // Memory
            Op::Load { ptr, mask, .. } => {
                if let Some(vid) = inst.result {
                    let r = self.assign(vid, RegClass::F);
                    let ptr_r = self.get(*ptr);
                    if let Some(m) = mask {
                        let pred = self.get(*m);
                        self.emit_line(&format!("@{} ld.global.f32 {}, [{}];", pred, r, ptr_r));
                    } else {
                        self.emit_line(&format!("ld.global.f32 {}, [{}];", r, ptr_r));
                    }
                }
            }
            Op::Store { ptr, val, mask } => {
                let ptr_r = self.get(*ptr);
                let val_r = self.get(*val);
                if let Some(m) = mask {
                    let pred = self.get(*m);
                    self.emit_line(&format!("@{} st.global.f32 [{}], {};", pred, ptr_r, val_r));
                } else {
                    self.emit_line(&format!("st.global.f32 [{}], {};", ptr_r, val_r));
                }
            }
            Op::AddPtr { ptr, offset } => {
                if let Some(vid) = inst.result {
                    let r = self.assign(vid, RegClass::Rd);
                    let ptr_r = self.get(*ptr);
                    // Widen offset (u32) to u64 and multiply by 4 (f32 stride)
                    let wide = self.alloc_reg(RegClass::Rd);
                    let wide_n = self.reg_name(wide);
                    let off_r = self.get(*offset);
                    self.emit_line(&format!("mul.wide.u32 {}, {}, 4;", wide_n, off_r));
                    self.emit_line(&format!("add.u64 {}, {}, {};", r, ptr_r, wide_n));
                }
            }
            Op::AtomicAdd { ptr, val, .. } => {
                if let Some(vid) = inst.result {
                    let r = self.assign(vid, RegClass::F);
                    self.emit_line(&format!(
                        "atom.global.add.f32 {}, [{}], {};",
                        r, self.get(*ptr), self.get(*val)
                    ));
                }
            }

            // Matmul
            Op::Dot { a, b, acc } => {
                if let Some(vid) = inst.result {
                    let r = self.assign(vid, RegClass::F);
                    self.emit_line(&format!(
                        "// mma.sync.aligned.m16n8k16 {} = dot({}, {}, {})",
                        r, self.get(*a), self.get(*b), self.get(*acc)
                    ));
                    self.emit_line(&format!("mov.f32 {}, {};  // TODO: real MMA", r, self.get(*acc)));
                }
            }

            // Reductions (warp shuffle tree)
            Op::ReduceSum { val, .. } => {
                if let Some(vid) = inst.result {
                    let src = self.get(*val);
                    let r = self.assign(vid, RegClass::F);
                    self.emit_line(&format!("mov.f32 {}, {};", r, src));
                    // 5-round butterfly reduction (warp size 32)
                    for dist in [16, 8, 4, 2, 1] {
                        let tmp = self.alloc_reg(RegClass::F);
                        let tmp_n = self.reg_name(tmp);
                        self.emit_line(&format!(
                            "shfl.sync.bfly.b32 {}, {}, {}, 0x1f, 0xffffffff;",
                            tmp_n, r, dist
                        ));
                        self.emit_line(&format!("add.f32 {}, {}, {};", r, r, tmp_n));
                    }
                }
            }
            Op::ReduceMax { val, .. } => {
                if let Some(vid) = inst.result {
                    let src = self.get(*val);
                    let r = self.assign(vid, RegClass::F);
                    self.emit_line(&format!("mov.f32 {}, {};", r, src));
                    for dist in [16, 8, 4, 2, 1] {
                        let tmp = self.alloc_reg(RegClass::F);
                        let tmp_n = self.reg_name(tmp);
                        self.emit_line(&format!(
                            "shfl.sync.bfly.b32 {}, {}, {}, 0x1f, 0xffffffff;",
                            tmp_n, r, dist
                        ));
                        self.emit_line(&format!("max.f32 {}, {}, {};", r, r, tmp_n));
                    }
                }
            }

            // Cast
            Op::Cast { val, to } => {
                if let Some(vid) = inst.result {
                    let src = self.get(*val);
                    let src_class = self.get_class(*val);
                    let (dst_class, cvt) = match (src_class, to) {
                        (RegClass::F, ScalarType::F16) => (RegClass::H, "cvt.rn.f16.f32"),
                        (RegClass::H, ScalarType::F32) => (RegClass::F, "cvt.f32.f16"),
                        (RegClass::R, ScalarType::F32) => (RegClass::F, "cvt.rn.f32.s32"),
                        (RegClass::F, ScalarType::I32) => (RegClass::R, "cvt.rzi.s32.f32"),
                        _ => (RegClass::F, "mov.b32"), // fallback
                    };
                    let r = self.assign(vid, dst_class);
                    self.emit_line(&format!("{} {}, {};", cvt, r, src));
                }
            }

            // Shape ops -- no-op in PTX, just alias the register
            Op::Splat { val, .. } | Op::BroadcastTo { val, .. }
            | Op::Reshape { val, .. } | Op::ExpandDims { val, .. } => {
                if let Some(vid) = inst.result {
                    if let Some(r) = self.reg_map.get(val) {
                        self.reg_map.insert(vid, *r);
                    }
                }
            }

            // Async pipeline
            Op::AsyncCopy { dst, src, size } => {
                self.emit_line(&format!(
                    "cp.async.ca.shared.global [{}], [{}], {};",
                    self.get(*dst), self.get(*src), size
                ));
            }
            Op::AsyncCommit => {
                self.emit_line("cp.async.commit_group;");
            }
            Op::AsyncWait { count } => {
                self.emit_line(&format!("cp.async.wait_group {};", count));
            }
            Op::Barrier => {
                self.emit_line("bar.sync 0;");
            }

            Op::Comment { text } => {
                self.emit_line(&format!("// {}", text));
            }
        }
        Ok(())
    }

    fn emit_binary_f32(&mut self, op: &str, result: Option<ValueId>, a: ValueId, b: ValueId) {
        if let Some(vid) = result {
            let ra = self.get(a);
            let rb = self.get(b);
            let r = self.assign(vid, RegClass::F);
            self.emit_line(&format!("{}.f32 {}, {}, {};", op, r, ra, rb));
        }
    }

    fn emit_setp(&mut self, cmp: &str, result: Option<ValueId>, a: ValueId, b: ValueId) {
        if let Some(vid) = result {
            let ra = self.get(a);
            let rb = self.get(b);
            let r = self.assign(vid, RegClass::P);
            self.emit_line(&format!("setp.{}.f32 {}, {}, {};", cmp, r, ra, rb));
        }
    }

    fn fixup_reg_decls(&mut self) {
        let replacements = [
            ("REG_COUNT_R", self.reg_counters[0]),
            ("REG_COUNT_RD", self.reg_counters[1]),
            ("REG_COUNT_F", self.reg_counters[2]),
            ("REG_COUNT_H", self.reg_counters[3]),
            ("REG_COUNT_P", self.reg_counters[4]),
        ];
        for (placeholder, count) in replacements {
            // +1 because PTX register ranges are exclusive
            self.out = self.out.replace(placeholder, &(count + 1).to_string());
        }
    }

    /// Compile PTX text to CUBIN by shelling out to ptxas.
    pub fn compile_ptx_to_cubin(ptx: &str, arch: SmArch) -> Result<Vec<u8>, CodegenError> {
        let tmp_dir = std::env::temp_dir();
        let ptx_path = tmp_dir.join("rtriton_kernel.ptx");
        let cubin_path = tmp_dir.join("rtriton_kernel.cubin");

        std::fs::write(&ptx_path, ptx)?;

        let output = std::process::Command::new("ptxas")
            .arg("--gpu-name")
            .arg(arch.target())
            .arg("-o")
            .arg(&cubin_path)
            .arg(&ptx_path)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CodegenError::PtxasFailed(stderr.into_owned()));
        }

        let cubin = std::fs::read(&cubin_path)?;
        // Clean up
        let _ = std::fs::remove_file(&ptx_path);
        let _ = std::fs::remove_file(&cubin_path);
        Ok(cubin)
    }
}


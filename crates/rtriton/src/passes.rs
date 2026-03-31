// rTriton optimization passes

use crate::ir::{Function, Instruction, Op, ScalarType, ValueId};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Pass trait + manager
// ---------------------------------------------------------------------------

pub trait Pass {
    fn name(&self) -> &str;
    fn run(&self, func: &mut Function);
}

pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    pub fn run(&self, func: &mut Function) {
        for pass in &self.passes {
            tracing::debug!("running pass: {}", pass.name());
            pass.run(func);
        }
    }

    /// Build the default optimization pipeline (all passes in recommended order).
    pub fn default_pipeline() -> Self {
        let mut pm = Self::new();
        pm.add(Box::new(ConstantFoldPass));
        pm.add(Box::new(DeadCodeElimPass));
        pm.add(Box::new(FusionPass));
        pm.add(Box::new(CoalescePass));
        pm.add(Box::new(SmemAllocPass));
        pm.add(Box::new(PipelinePass));
        // DCE again after pipeline may have introduced dead values
        pm.add(Box::new(DeadCodeElimPass));
        pm
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 1. DeadCodeElimPass
// ---------------------------------------------------------------------------

pub struct DeadCodeElimPass;

impl Pass for DeadCodeElimPass {
    fn name(&self) -> &str {
        "dead-code-elim"
    }

    fn run(&self, func: &mut Function) {
        // Collect all ValueIds that are used as operands anywhere.
        let mut used: HashSet<ValueId> = HashSet::new();
        for inst in &func.body.instructions {
            for op in inst.op.operands() {
                used.insert(op);
            }
        }

        // Walk in reverse, removing instructions whose result is unused
        // and that have no side effects. When we remove an instruction,
        // its operands may become unused too, so we iterate to fixpoint.
        let mut changed = true;
        while changed {
            changed = false;
            // Recompute used set each iteration (cheap enough for IR sizes we care about).
            used.clear();
            for inst in &func.body.instructions {
                for op in inst.op.operands() {
                    used.insert(op);
                }
            }

            let before = func.body.instructions.len();
            func.body.instructions.retain(|inst| {
                if inst.op.has_side_effects() {
                    return true;
                }
                match inst.result {
                    Some(vid) => used.contains(&vid),
                    // No result and no side effects -- dead
                    None => false,
                }
            });
            if func.body.instructions.len() < before {
                changed = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. FusionPass
// ---------------------------------------------------------------------------

pub struct FusionPass;

impl Pass for FusionPass {
    fn name(&self) -> &str {
        "fusion"
    }

    fn run(&self, func: &mut Function) {
        // Build use-count map: how many instructions consume each ValueId.
        let mut use_count: HashMap<ValueId, usize> = HashMap::new();
        for inst in &func.body.instructions {
            for op in inst.op.operands() {
                *use_count.entry(op).or_insert(0) += 1;
            }
        }

        // Walk consecutive pairs. If both are elementwise and the producer has
        // exactly one consumer (the next instruction), assign them the same
        // fusion group.
        let mut group_id: u32 = 0;
        let mut current_group: Option<u32> = None;

        let len = func.body.instructions.len();
        for i in 0..len {
            let is_ew = func.body.instructions[i].op.is_elementwise();
            if !is_ew {
                current_group = None;
                continue;
            }

            // Check if this instruction's result feeds exactly into the next
            // elementwise instruction (single consumer).
            let can_fuse_with_next = if i + 1 < len {
                let next_is_ew = func.body.instructions[i + 1].op.is_elementwise();
                if let (true, Some(result_id)) = (next_is_ew, func.body.instructions[i].result) {
                    use_count.get(&result_id).copied().unwrap_or(0) == 1
                } else {
                    false
                }
            } else {
                false
            };

            // Assign group to current instruction.
            let gid = match current_group {
                Some(g) => g,
                None => {
                    group_id += 1;
                    group_id
                }
            };
            func.body.instructions[i]
                .annotations
                .insert("fusion_group".to_string(), gid.to_string());

            if can_fuse_with_next {
                current_group = Some(gid);
            } else {
                current_group = None;
            }
        }

        // Tag the last instruction in a group that didn't get handled by the
        // lookahead above.
        // (Already handled: each instruction in a chain gets the group id.)
    }
}

// ---------------------------------------------------------------------------
// 3. CoalescePass
// ---------------------------------------------------------------------------

pub struct CoalescePass;

impl Pass for CoalescePass {
    fn name(&self) -> &str {
        "coalesce"
    }

    fn run(&self, func: &mut Function) {
        // Build a map from ValueId -> Op for lookups.
        let val_to_op: HashMap<ValueId, &Op> = func
            .body
            .instructions
            .iter()
            .filter_map(|inst| inst.result.map(|r| (r, &inst.op)))
            .collect();

        // For each Load/Store, check if the pointer comes from an AddPtr whose
        // offset was derived from an Arange (which maps to threadIdx on the
        // innermost dimension).
        let coalesced_indices: Vec<usize> = func
            .body
            .instructions
            .iter()
            .enumerate()
            .filter_map(|(idx, inst)| {
                let ptr_id = match &inst.op {
                    Op::Load { ptr, .. } => Some(*ptr),
                    Op::Store { ptr, .. } => Some(*ptr),
                    _ => None,
                }?;

                // The pointer should be an AddPtr.
                let offset_id = match val_to_op.get(&ptr_id)? {
                    Op::AddPtr { offset, .. } => Some(*offset),
                    _ => None,
                }?;

                // The offset (or a value in its chain) should originate from Arange,
                // which represents threadIdx-based indexing on the innermost dim.
                if is_derived_from_arange(offset_id, &val_to_op) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        for idx in coalesced_indices {
            func.body.instructions[idx]
                .annotations
                .insert("coalesced".to_string(), "true".to_string());
        }
    }
}

/// Recursively check if a value is derived (transitively) from an Arange op.
fn is_derived_from_arange(vid: ValueId, val_to_op: &HashMap<ValueId, &Op>) -> bool {
    match val_to_op.get(&vid) {
        Some(Op::Arange { .. }) => true,
        Some(op) => {
            // Chase through elementwise / shape ops
            op.operands().iter().any(|&o| is_derived_from_arange(o, val_to_op))
        }
        None => false,
    }
}

// ---------------------------------------------------------------------------
// 4. SmemAllocPass
// ---------------------------------------------------------------------------

pub struct SmemAllocPass;

impl Pass for SmemAllocPass {
    fn name(&self) -> &str {
        "smem-alloc"
    }

    fn run(&self, func: &mut Function) {
        let block_m = func.get_constexpr("BLOCK_M").unwrap_or(128) as usize;
        let block_k = func.get_constexpr("BLOCK_K").unwrap_or(64) as usize;
        let block_n = func.get_constexpr("BLOCK_N").unwrap_or(128) as usize;
        let num_stages = func.get_constexpr("num_stages").unwrap_or(3) as usize;

        // Determine dtype size from Dot operands by chasing through the IR.
        // Read phase: collect (index, smem_bytes) pairs.
        let val_to_op: HashMap<ValueId, Op> = func
            .body
            .instructions
            .iter()
            .filter_map(|inst| inst.result.map(|r| (r, inst.op.clone())))
            .collect();

        let smem_annotations: Vec<(usize, usize)> = func
            .body
            .instructions
            .iter()
            .enumerate()
            .filter_map(|(idx, inst)| {
                if let Op::Dot { a, b, .. } = &inst.op {
                    let dtype_size_a = resolve_dtype_size(*a, &val_to_op);
                    let dtype_size_b = resolve_dtype_size(*b, &val_to_op);
                    let smem_a = block_m * block_k * dtype_size_a;
                    let smem_b = block_k * block_n * dtype_size_b;
                    let total_smem = (smem_a + smem_b) * num_stages;
                    Some((idx, total_smem))
                } else {
                    None
                }
            })
            .collect();

        // Write phase: apply annotations.
        for (idx, total_smem) in &smem_annotations {
            func.body.instructions[*idx]
                .annotations
                .insert("smem_bytes".to_string(), total_smem.to_string());
        }

        // Also store at function level for codegen.
        if let Some(total) = func
            .body
            .instructions
            .iter()
            .filter_map(|i| i.annotations.get("smem_bytes"))
            .filter_map(|s| s.parse::<usize>().ok())
            .max()
        {
            func.annotations
                .insert("smem_bytes".to_string(), total.to_string());
        }
    }
}

/// Chase a value back through Loads/Casts to find the underlying dtype size.
fn resolve_dtype_size(vid: ValueId, val_to_op: &HashMap<ValueId, Op>) -> usize {
    match val_to_op.get(&vid) {
        Some(Op::Load { .. }) => {
            // Look for a Cast feeding the Dot -- the cast target tells us the
            // compute dtype, but for smem allocation we want the load dtype.
            // Default to 2 bytes (fp16/bf16) which is the common case.
            2
        }
        Some(Op::Cast { to, .. }) => scalar_size(*to),
        _ => 2, // default fp16
    }
}

fn scalar_size(ty: ScalarType) -> usize {
    match ty {
        ScalarType::F32 | ScalarType::I32 | ScalarType::U32 => 4,
        ScalarType::F16 | ScalarType::BF16 => 2,
        ScalarType::F8E4M3 | ScalarType::F8E5M2 => 1,
        ScalarType::I64 | ScalarType::U64 => 8,
        ScalarType::I1 => 1,
    }
}

// ---------------------------------------------------------------------------
// 5. PipelinePass
// ---------------------------------------------------------------------------

pub struct PipelinePass;

impl Pass for PipelinePass {
    fn name(&self) -> &str {
        "pipeline"
    }

    fn run(&self, func: &mut Function) {
        let num_stages = func.get_constexpr("num_stages").unwrap_or(1);
        if num_stages <= 1 {
            return;
        }

        // Find Load instructions that feed into a Dot. For each such Load,
        // we transform it into an async-copy pipeline:
        //   AsyncCopy (prefetch next tile into smem)
        //   AsyncCommit
        //   ... (other work)
        //   AsyncWait(num_stages - 2)
        //   Barrier
        //   Load (from smem -- the original load is now from shared)

        // Build producer map: ValueId -> instruction index.
        let val_to_idx: HashMap<ValueId, usize> = func
            .body
            .instructions
            .iter()
            .enumerate()
            .filter_map(|(i, inst)| inst.result.map(|r| (r, i)))
            .collect();

        // Find Dot instructions and their Load operands.
        let dot_load_pairs: Vec<(usize, Vec<usize>)> = func
            .body
            .instructions
            .iter()
            .enumerate()
            .filter_map(|(dot_idx, inst)| {
                if let Op::Dot { a, b, .. } = &inst.op {
                    let load_indices: Vec<usize> = [*a, *b]
                        .iter()
                        .filter_map(|vid| val_to_idx.get(vid).copied())
                        .filter(|&idx| matches!(func.body.instructions[idx].op, Op::Load { .. }))
                        .collect();
                    if !load_indices.is_empty() {
                        return Some((dot_idx, load_indices));
                    }
                }
                None
            })
            .collect();

        if dot_load_pairs.is_empty() {
            return;
        }

        // Insert pipeline instructions. We work backwards to keep indices stable.
        let wait_count = if num_stages >= 2 { num_stages - 2 } else { 0 };

        for (dot_idx, load_indices) in dot_load_pairs.into_iter().rev() {
            // Insert AsyncWait + Barrier right before the Dot.
            let barrier = Instruction {
                result: None,
                op: Op::Barrier,
                annotations: HashMap::new(),
            };
            let wait = Instruction {
                result: None,
                op: Op::AsyncWait { count: wait_count },
                annotations: HashMap::new(),
            };
            func.body.instructions.insert(dot_idx, barrier);
            func.body.instructions.insert(dot_idx, wait);

            // Insert AsyncCopy + AsyncCommit right after each Load that feeds the Dot.
            for &load_idx in load_indices.iter().rev() {
                let (src_id, dst_id) = match &func.body.instructions[load_idx].op {
                    Op::Load { ptr, .. } => {
                        // src = the global pointer, dst = a new smem pointer (we reuse
                        // the load's result as the dst placeholder -- codegen resolves this)
                        let dst = func.body.instructions[load_idx]
                            .result
                            .unwrap_or(ValueId(0));
                        (*ptr, dst)
                    }
                    _ => continue,
                };

                let commit = Instruction {
                    result: None,
                    op: Op::AsyncCommit,
                    annotations: HashMap::new(),
                };
                let copy = Instruction {
                    result: None,
                    op: Op::AsyncCopy {
                        dst: dst_id,
                        src: src_id,
                        size: 0, // size resolved by codegen from tile dimensions
                    },
                    annotations: HashMap::new(),
                };
                // Insert after the load instruction.
                let insert_pos = load_idx + 1;
                func.body.instructions.insert(insert_pos, commit);
                func.body.instructions.insert(insert_pos, copy);
            }
        }

        func.annotations
            .insert("pipelined".to_string(), "true".to_string());
        func.annotations
            .insert("num_stages".to_string(), num_stages.to_string());
    }
}

// ---------------------------------------------------------------------------
// 6. ConstantFoldPass
// ---------------------------------------------------------------------------

pub struct ConstantFoldPass;

impl Pass for ConstantFoldPass {
    fn name(&self) -> &str {
        "constant-fold"
    }

    fn run(&self, func: &mut Function) {
        // Build map: ValueId -> constant value (if the instruction is a constant).
        let mut constants: HashMap<ValueId, ConstValue> = HashMap::new();
        for inst in &func.body.instructions {
            if let Some(vid) = inst.result {
                match &inst.op {
                    Op::ConstantF32 { val } => {
                        constants.insert(vid, ConstValue::F32(*val));
                    }
                    Op::ConstantI32 { val } => {
                        constants.insert(vid, ConstValue::I32(*val));
                    }
                    _ => {}
                }
            }
        }

        // Replace binary ops on two constants with a single constant.
        let mut replacements: Vec<(usize, Op)> = Vec::new();

        for (idx, inst) in func.body.instructions.iter().enumerate() {
            let folded = match &inst.op {
                Op::Add { a, b } => fold_binary(*a, *b, &constants, |x, y| x + y, |x, y| x + y),
                Op::Sub { a, b } => fold_binary(*a, *b, &constants, |x, y| x - y, |x, y| x - y),
                Op::Mul { a, b } => fold_binary(*a, *b, &constants, |x, y| x * y, |x, y| x * y),
                Op::Div { a, b } => {
                    fold_binary(*a, *b, &constants, |x, y| x / y, |x, y| {
                        if y != 0 { x / y } else { 0 }
                    })
                }
                Op::Maximum { a, b } => {
                    fold_binary(*a, *b, &constants, f32::max, std::cmp::max)
                }
                Op::Minimum { a, b } => {
                    fold_binary(*a, *b, &constants, f32::min, std::cmp::min)
                }
                _ => None,
            };

            if let Some(new_op) = folded {
                replacements.push((idx, new_op));
            }
        }

        for (idx, new_op) in replacements {
            // Update the constant map with the newly folded value.
            if let Some(vid) = func.body.instructions[idx].result {
                match &new_op {
                    Op::ConstantF32 { val } => {
                        constants.insert(vid, ConstValue::F32(*val));
                    }
                    Op::ConstantI32 { val } => {
                        constants.insert(vid, ConstValue::I32(*val));
                    }
                    _ => {}
                }
            }
            func.body.instructions[idx].op = new_op;
        }
    }
}

#[derive(Clone, Copy)]
enum ConstValue {
    F32(f32),
    I32(i32),
}

fn fold_binary(
    a: ValueId,
    b: ValueId,
    constants: &HashMap<ValueId, ConstValue>,
    f_f32: impl Fn(f32, f32) -> f32,
    f_i32: impl Fn(i32, i32) -> i32,
) -> Option<Op> {
    let ca = constants.get(&a)?;
    let cb = constants.get(&b)?;
    match (ca, cb) {
        (ConstValue::F32(x), ConstValue::F32(y)) => {
            Some(Op::ConstantF32 { val: f_f32(*x, *y) })
        }
        (ConstValue::I32(x), ConstValue::I32(y)) => {
            Some(Op::ConstantI32 { val: f_i32(*x, *y) })
        }
        // Mixed types: promote i32 to f32.
        (ConstValue::F32(x), ConstValue::I32(y)) => {
            Some(Op::ConstantF32 { val: f_f32(*x, *y as f32) })
        }
        (ConstValue::I32(x), ConstValue::F32(y)) => {
            Some(Op::ConstantF32 { val: f_f32(*x as f32, *y) })
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::*;

    fn make_test_func() -> Function {
        Function {
            name: "test_kernel".to_string(),
            args: vec![],
            constexprs: vec![],
            body: Block {
                instructions: vec![],
            },
            next_value_id: 0,
            annotations: HashMap::new(),
        }
    }

    fn push_inst(func: &mut Function, op: Op) -> Option<ValueId> {
        let has_result = !op.has_side_effects();
        let result = if has_result {
            Some(func.alloc_value())
        } else {
            None
        };
        func.body.instructions.push(Instruction {
            result,
            op,
            annotations: HashMap::new(),
        });
        result
    }

    #[test]
    fn test_dead_code_elim() {
        let mut func = make_test_func();
        let c1 = push_inst(&mut func, Op::ConstantI32 { val: 42 }).unwrap();
        let c2 = push_inst(&mut func, Op::ConstantI32 { val: 10 }).unwrap();
        // c3 uses c1 and c2
        let _c3 = push_inst(&mut func, Op::Add { a: c1, b: c2 }).unwrap();
        // c4 is dead -- nobody uses it
        let _c4 = push_inst(&mut func, Op::ConstantI32 { val: 999 }).unwrap();

        assert_eq!(func.body.instructions.len(), 4);
        DeadCodeElimPass.run(&mut func);
        // c3 is also dead (not used by any side-effecting op), and transitively
        // c1 and c2 become dead too. All should be eliminated.
        assert_eq!(func.body.instructions.len(), 0);
    }

    #[test]
    fn test_dead_code_preserves_stores() {
        let mut func = make_test_func();
        let c1 = push_inst(&mut func, Op::ConstantI32 { val: 42 }).unwrap();
        let ptr = push_inst(&mut func, Op::ConstantI32 { val: 0 }).unwrap();
        // Store has side effects, should be kept along with its operands.
        push_inst(
            &mut func,
            Op::Store {
                ptr,
                val: c1,
                mask: None,
            },
        );

        assert_eq!(func.body.instructions.len(), 3);
        DeadCodeElimPass.run(&mut func);
        assert_eq!(func.body.instructions.len(), 3);
    }

    #[test]
    fn test_constant_fold() {
        let mut func = make_test_func();
        let c1 = push_inst(&mut func, Op::ConstantI32 { val: 10 }).unwrap();
        let c2 = push_inst(&mut func, Op::ConstantI32 { val: 20 }).unwrap();
        let _sum = push_inst(&mut func, Op::Add { a: c1, b: c2 }).unwrap();

        ConstantFoldPass.run(&mut func);

        // The Add should have been replaced with ConstantI32 { val: 30 }.
        match &func.body.instructions[2].op {
            Op::ConstantI32 { val } => assert_eq!(*val, 30),
            other => panic!("expected ConstantI32, got {:?}", other),
        }
    }

    #[test]
    fn test_constant_fold_f32() {
        let mut func = make_test_func();
        let c1 = push_inst(&mut func, Op::ConstantF32 { val: 2.5 }).unwrap();
        let c2 = push_inst(&mut func, Op::ConstantF32 { val: 4.0 }).unwrap();
        let _prod = push_inst(&mut func, Op::Mul { a: c1, b: c2 }).unwrap();

        ConstantFoldPass.run(&mut func);

        match &func.body.instructions[2].op {
            Op::ConstantF32 { val } => assert!((val - 10.0).abs() < 1e-6),
            other => panic!("expected ConstantF32, got {:?}", other),
        }
    }

    #[test]
    fn test_fusion_consecutive_elementwise() {
        let mut func = make_test_func();
        let a = push_inst(&mut func, Op::ConstantF32 { val: 1.0 }).unwrap();
        let b = push_inst(&mut func, Op::ConstantF32 { val: 2.0 }).unwrap();
        // Three consecutive elementwise ops forming a chain.
        let c = push_inst(&mut func, Op::Add { a, b }).unwrap();
        let d = push_inst(&mut func, Op::Mul { a: c, b: a }).unwrap();
        let _e = push_inst(&mut func, Op::Neg { x: d }).unwrap();

        FusionPass.run(&mut func);

        // Instructions 2,3,4 (Add, Mul, Neg) should share the same fusion group.
        let g2 = func.body.instructions[2].annotations.get("fusion_group");
        let g3 = func.body.instructions[3].annotations.get("fusion_group");
        let g4 = func.body.instructions[4].annotations.get("fusion_group");
        assert!(g2.is_some());
        assert_eq!(g2, g3);
        assert_eq!(g3, g4);
    }

    #[test]
    fn test_pass_manager_default() {
        let pm = PassManager::default_pipeline();
        assert!(pm.passes.len() >= 6);
    }
}

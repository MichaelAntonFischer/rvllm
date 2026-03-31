// rTriton builder DSL

use std::collections::HashMap;
use crate::ir::*;

pub struct KernelBuilder {
    name: String,
    next_id: u32,
    instructions: Vec<Instruction>,
    args: Vec<ArgInfo>,
    constexprs: Vec<ConstExprInfo>,
}

impl KernelBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            next_id: 0,
            instructions: Vec::new(),
            args: Vec::new(),
            constexprs: Vec::new(),
        }
    }

    fn fresh_id(&mut self) -> ValueId {
        let id = ValueId(self.next_id);
        self.next_id += 1;
        id
    }

    fn emit(&mut self, op: Op) -> ValueId {
        let id = self.fresh_id();
        self.instructions.push(Instruction { result: Some(id), op, annotations: Default::default() });
        id
    }

    fn emit_void(&mut self, op: Op) {
        self.instructions.push(Instruction { result: None, op, annotations: Default::default() });
    }

    // --- Kernel arguments ---

    pub fn arg_ptr(&mut self, name: &str, dtype: ScalarType) -> ValueId {
        self.args.push(ArgInfo { name: name.to_owned(), dtype, is_ptr: true });
        self.emit(Op::Arg { name: name.to_owned(), dtype, is_ptr: true })
    }

    pub fn arg_i32(&mut self, name: &str) -> ValueId {
        self.args.push(ArgInfo { name: name.to_owned(), dtype: ScalarType::I32, is_ptr: false });
        self.emit(Op::Arg { name: name.to_owned(), dtype: ScalarType::I32, is_ptr: false })
    }

    pub fn arg_f32(&mut self, name: &str) -> ValueId {
        self.args.push(ArgInfo { name: name.to_owned(), dtype: ScalarType::F32, is_ptr: false });
        self.emit(Op::Arg { name: name.to_owned(), dtype: ScalarType::F32, is_ptr: false })
    }

    // --- Compile-time constants ---

    pub fn constexpr(&mut self, name: &str, default: u32) -> ValueId {
        self.constexprs.push(ConstExprInfo { name: name.to_owned(), default });
        self.emit(Op::ConstExpr { name: name.to_owned(), default })
    }

    // --- Triton primitives ---

    pub fn program_id(&mut self, axis: u32) -> ValueId {
        self.emit(Op::ProgramId { axis })
    }

    pub fn arange(&mut self, start: i32, end: i32) -> ValueId {
        self.emit(Op::Arange { start, end })
    }

    pub fn constant_f32(&mut self, val: f32) -> ValueId {
        self.emit(Op::ConstantF32 { val })
    }

    pub fn constant_i32(&mut self, val: i32) -> ValueId {
        self.emit(Op::ConstantI32 { val })
    }

    // --- Memory ---

    pub fn load(&mut self, ptr: ValueId, mask: Option<ValueId>, other: Option<ValueId>) -> ValueId {
        self.emit(Op::Load { ptr, mask, other })
    }

    pub fn store(&mut self, ptr: ValueId, val: ValueId, mask: Option<ValueId>) {
        self.emit_void(Op::Store { ptr, val, mask });
    }

    pub fn add_ptr(&mut self, ptr: ValueId, offset: ValueId) -> ValueId {
        self.emit(Op::AddPtr { ptr, offset })
    }

    pub fn atomic_add(&mut self, ptr: ValueId, val: ValueId, mask: Option<ValueId>) -> ValueId {
        self.emit(Op::AtomicAdd { ptr, val, mask })
    }

    // --- Matmul ---

    pub fn dot(&mut self, a: ValueId, b: ValueId, acc: ValueId) -> ValueId {
        self.emit(Op::Dot { a, b, acc })
    }

    // --- Binary ops ---

    pub fn add(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.emit(Op::Add { a, b })
    }

    pub fn sub(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.emit(Op::Sub { a, b })
    }

    pub fn mul(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.emit(Op::Mul { a, b })
    }

    pub fn div(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.emit(Op::Div { a, b })
    }

    pub fn maximum(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.emit(Op::Maximum { a, b })
    }

    pub fn minimum(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.emit(Op::Minimum { a, b })
    }

    // --- Unary math ---

    pub fn exp(&mut self, x: ValueId) -> ValueId {
        self.emit(Op::Exp { x })
    }

    pub fn log(&mut self, x: ValueId) -> ValueId {
        self.emit(Op::Log { x })
    }

    pub fn sqrt(&mut self, x: ValueId) -> ValueId {
        self.emit(Op::Sqrt { x })
    }

    pub fn rsqrt(&mut self, x: ValueId) -> ValueId {
        self.emit(Op::Rsqrt { x })
    }

    pub fn neg(&mut self, x: ValueId) -> ValueId {
        self.emit(Op::Neg { x })
    }

    pub fn abs(&mut self, x: ValueId) -> ValueId {
        self.emit(Op::Abs { x })
    }

    // --- Compound (lowered to primitives) ---

    pub fn sigmoid(&mut self, x: ValueId) -> ValueId {
        let neg_x = self.neg(x);
        let exp_neg = self.exp(neg_x);
        let one = self.constant_f32(1.0);
        let denom = self.add(one, exp_neg);
        self.div(one, denom)
    }

    // --- Comparison ---

    pub fn cmp_lt(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.emit(Op::CmpLt { a, b })
    }

    pub fn cmp_gt(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.emit(Op::CmpGt { a, b })
    }

    pub fn cmp_eq(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.emit(Op::CmpEq { a, b })
    }

    // --- Select ---

    pub fn where_(&mut self, cond: ValueId, true_val: ValueId, false_val: ValueId) -> ValueId {
        self.emit(Op::Where { cond, true_val, false_val })
    }

    // --- Reduce ---

    pub fn reduce_sum(&mut self, val: ValueId, axis: u32) -> ValueId {
        self.emit(Op::ReduceSum { val, axis })
    }

    pub fn reduce_max(&mut self, val: ValueId, axis: u32) -> ValueId {
        self.emit(Op::ReduceMax { val, axis })
    }

    // --- Shape ---

    pub fn cast(&mut self, val: ValueId, to: ScalarType) -> ValueId {
        self.emit(Op::Cast { val, to })
    }

    pub fn broadcast_to(&mut self, val: ValueId, shape: &[u32]) -> ValueId {
        self.emit(Op::BroadcastTo { val, shape: shape.to_vec() })
    }

    pub fn reshape(&mut self, val: ValueId, shape: &[u32]) -> ValueId {
        self.emit(Op::Reshape { val, shape: shape.to_vec() })
    }

    pub fn splat(&mut self, val: ValueId, shape: &[u32]) -> ValueId {
        self.emit(Op::Splat { val, shape: shape.to_vec() })
    }

    pub fn expand_dims(&mut self, val: ValueId, axis: u32) -> ValueId {
        self.emit(Op::ExpandDims { val, axis })
    }

    // --- Build ---

    pub fn build(self) -> Function {
        Function {
            name: self.name,
            args: self.args,
            constexprs: self.constexprs,
            body: Block { instructions: self.instructions },
            next_value_id: self.next_id,
            annotations: HashMap::new(),
        }
    }
}

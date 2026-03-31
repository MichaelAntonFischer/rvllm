// rTriton IR -- core types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Opaque handle to an SSA value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValueId(pub u32);

/// Scalar element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarType {
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    I32,
    I64,
    U32,
    U64,
    I1,
}

/// Operation tag -- one per IR node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Op {
    // Kernel arguments
    Arg { name: String, dtype: ScalarType, is_ptr: bool },
    ConstExpr { name: String, default: u32 },

    // Triton built-ins
    ProgramId { axis: u32 },
    Arange { start: i32, end: i32 },
    ConstantF32 { val: f32 },
    ConstantI32 { val: i32 },

    // Memory
    Load { ptr: ValueId, mask: Option<ValueId>, other: Option<ValueId> },
    Store { ptr: ValueId, val: ValueId, mask: Option<ValueId> },
    AddPtr { ptr: ValueId, offset: ValueId },
    AtomicAdd { ptr: ValueId, val: ValueId, mask: Option<ValueId> },

    // Matmul
    Dot { a: ValueId, b: ValueId, acc: ValueId },

    // Binary
    Add { a: ValueId, b: ValueId },
    Sub { a: ValueId, b: ValueId },
    Mul { a: ValueId, b: ValueId },
    Div { a: ValueId, b: ValueId },
    Maximum { a: ValueId, b: ValueId },
    Minimum { a: ValueId, b: ValueId },

    // Unary
    Exp { x: ValueId },
    Log { x: ValueId },
    Sqrt { x: ValueId },
    Rsqrt { x: ValueId },
    Neg { x: ValueId },
    Abs { x: ValueId },

    // Comparison (result is I1 / mask)
    CmpLt { a: ValueId, b: ValueId },
    CmpGt { a: ValueId, b: ValueId },
    CmpEq { a: ValueId, b: ValueId },

    // Select
    Where { cond: ValueId, true_val: ValueId, false_val: ValueId },

    // Reduce
    ReduceSum { val: ValueId, axis: u32 },
    ReduceMax { val: ValueId, axis: u32 },

    // Shape manipulation
    Cast { val: ValueId, to: ScalarType },
    BroadcastTo { val: ValueId, shape: Vec<u32> },
    Reshape { val: ValueId, shape: Vec<u32> },
    Splat { val: ValueId, shape: Vec<u32> },
    ExpandDims { val: ValueId, axis: u32 },

    // Async pipeline ops
    AsyncCopy { dst: ValueId, src: ValueId, size: u32 },
    AsyncCommit,
    AsyncWait { count: u32 },
    Barrier,

    // Annotation (no-op, carries metadata for codegen)
    Comment { text: String },
}

impl Op {
    /// Returns all ValueId operands referenced by this op.
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            Op::Arg { .. } | Op::ConstExpr { .. } | Op::ProgramId { .. }
            | Op::Arange { .. } | Op::ConstantF32 { .. } | Op::ConstantI32 { .. }
            | Op::AsyncCommit | Op::AsyncWait { .. } | Op::Barrier
            | Op::Comment { .. } => vec![],

            Op::Load { ptr, mask, other } => {
                let mut v = vec![*ptr];
                if let Some(m) = mask { v.push(*m); }
                if let Some(o) = other { v.push(*o); }
                v
            }
            Op::Store { ptr, val, mask } => {
                let mut v = vec![*ptr, *val];
                if let Some(m) = mask { v.push(*m); }
                v
            }
            Op::AddPtr { ptr, offset } => vec![*ptr, *offset],
            Op::AtomicAdd { ptr, val, mask } => {
                let mut v = vec![*ptr, *val];
                if let Some(m) = mask { v.push(*m); }
                v
            }
            Op::Dot { a, b, acc } => vec![*a, *b, *acc],

            Op::Add { a, b } | Op::Sub { a, b } | Op::Mul { a, b }
            | Op::Div { a, b } | Op::Maximum { a, b } | Op::Minimum { a, b }
            | Op::CmpLt { a, b } | Op::CmpGt { a, b } | Op::CmpEq { a, b } => vec![*a, *b],

            Op::Exp { x } | Op::Log { x } | Op::Sqrt { x } | Op::Rsqrt { x }
            | Op::Neg { x } | Op::Abs { x } => vec![*x],

            Op::Where { cond, true_val, false_val } => vec![*cond, *true_val, *false_val],

            Op::ReduceSum { val, .. } | Op::ReduceMax { val, .. }
            | Op::Cast { val, .. } | Op::BroadcastTo { val, .. }
            | Op::Reshape { val, .. } | Op::Splat { val, .. }
            | Op::ExpandDims { val, .. } => vec![*val],

            Op::AsyncCopy { dst, src, .. } => vec![*dst, *src],
        }
    }

    /// Whether this op has side effects (cannot be dead-code eliminated).
    pub fn has_side_effects(&self) -> bool {
        matches!(
            self,
            Op::Store { .. } | Op::AtomicAdd { .. } | Op::AsyncCopy { .. }
            | Op::AsyncCommit | Op::AsyncWait { .. } | Op::Barrier
            | Op::Comment { .. }
        )
    }

    /// Whether this op is elementwise (candidate for fusion).
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            Op::Add { .. } | Op::Sub { .. } | Op::Mul { .. } | Op::Div { .. }
            | Op::Maximum { .. } | Op::Minimum { .. }
            | Op::Exp { .. } | Op::Log { .. } | Op::Sqrt { .. } | Op::Rsqrt { .. }
            | Op::Neg { .. } | Op::Abs { .. }
            | Op::Cast { .. } | Op::Where { .. }
            | Op::CmpLt { .. } | Op::CmpGt { .. } | Op::CmpEq { .. }
        )
    }

    /// Whether this is a constant op.
    pub fn is_constant(&self) -> bool {
        matches!(self, Op::ConstantF32 { .. } | Op::ConstantI32 { .. })
    }
}

/// Single SSA instruction: op + optional result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instruction {
    pub result: Option<ValueId>,
    pub op: Op,
    /// Pass-set annotations (e.g. "coalesced", "fusion_group", "smem_bytes")
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub annotations: HashMap<String, String>,
}

/// A basic block: linear sequence of instructions (no control flow yet).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub instructions: Vec<Instruction>,
}

/// Kernel-level argument metadata (for launch signature).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgInfo {
    pub name: String,
    pub dtype: ScalarType,
    pub is_ptr: bool,
}

/// Compile-time constant parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstExprInfo {
    pub name: String,
    pub default: u32,
}

/// Top-level function (one GPU kernel).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub args: Vec<ArgInfo>,
    pub constexprs: Vec<ConstExprInfo>,
    pub body: Block,
    /// Next available ValueId counter.
    #[serde(default)]
    pub next_value_id: u32,
    /// Function-level annotations from passes.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub annotations: HashMap<String, String>,
}

impl Function {
    /// Allocate a fresh ValueId.
    pub fn alloc_value(&mut self) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    /// Look up a constexpr by name.
    pub fn get_constexpr(&self, name: &str) -> Option<u32> {
        self.constexprs.iter().find(|c| c.name == name).map(|c| c.default)
    }
}

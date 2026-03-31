use crate::ir::ScalarType;
use crate::builder::KernelBuilder;


/// Fused SiLU * mul for MLP gating.
///
/// Input:  gate_up[M, 2*intermediate] (gate and up interleaved or concatenated)
/// Output: out[M, intermediate] = silu(gate) * up
///
/// Grid: one program per BLOCK_SIZE chunk of the output.
pub fn build_silu_mul(b: &mut KernelBuilder) {
    let gate_ptr = b.arg_ptr("gate_ptr", ScalarType::F32);
    let up_ptr = b.arg_ptr("up_ptr", ScalarType::F32);
    let out_ptr = b.arg_ptr("out_ptr", ScalarType::F32);
    let n_elems = b.arg_i32("n_elems"); // total output elements (M * intermediate)
    let _block = b.constexpr("BLOCK_SIZE", 1024);

    let pid = b.program_id(0);
    let block_size = b.constant_i32(1024);
    let base = b.mul(pid, block_size);
    let offs = b.arange(0, 1024);
    let idx = b.add(base, offs);
    let mask = b.cmp_lt(idx, n_elems);

    let zero = b.constant_f32(0.0);

    // Load gate and up
    let gate_addr = b.add_ptr(gate_ptr, idx);
    let up_addr = b.add_ptr(up_ptr, idx);
    let gate = b.load(gate_addr, Some(mask), Some(zero));
    let up = b.load(up_addr, Some(mask), Some(zero));

    // silu(gate) = gate * sigmoid(gate)
    let sig = b.sigmoid(gate);
    let silu = b.mul(gate, sig);

    // out = silu(gate) * up
    let result = b.mul(silu, up);

    let out_addr = b.add_ptr(out_ptr, idx);
    b.store(out_addr, result, Some(mask));

}

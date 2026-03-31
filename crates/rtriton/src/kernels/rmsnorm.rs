use crate::ir::ScalarType;
use crate::builder::KernelBuilder;


/// RMSNorm: out[row] = x[row] * weight / sqrt(mean(x[row]^2) + eps)
///
/// Grid: one program per row.
pub fn build_rmsnorm(b: &mut KernelBuilder) {
    let x_ptr = b.arg_ptr("x_ptr", ScalarType::F32);
    let w_ptr = b.arg_ptr("weight_ptr", ScalarType::F32);
    let out_ptr = b.arg_ptr("out_ptr", ScalarType::F32);
    let n_cols = b.arg_i32("n_cols");
    let eps = b.arg_f32("eps");
    let block_size = b.constexpr("BLOCK_SIZE", 1024);

    let pid = b.program_id(0); // row index
    let offs = b.arange(0, 1024); // [0..BLOCK_SIZE)

    // mask = offs < n_cols
    let mask = b.cmp_lt(offs, n_cols);

    // x = load(x_ptr + pid * n_cols + offs)
    let row_start = b.mul(pid, n_cols);
    let x_off = b.add(row_start, offs);
    let x_addr = b.add_ptr(x_ptr, x_off);
    let zero = b.constant_f32(0.0);
    let x = b.load(x_addr, Some(mask), Some(zero));

    // variance = sum(x^2) / n_cols
    let x_sq = b.mul(x, x);
    let sum_sq = b.reduce_sum(x_sq, 0);
    let n_cols_f = b.cast(n_cols, ScalarType::F32);
    let var = b.div(sum_sq, n_cols_f);

    // rstd = rsqrt(var + eps)
    let var_eps = b.add(var, eps);
    let rstd = b.rsqrt(var_eps);

    // w = load(weight_ptr + offs)
    let w_addr = b.add_ptr(w_ptr, offs);
    let w = b.load(w_addr, Some(mask), Some(zero));

    // out = x * rstd * w
    let normed = b.mul(x, rstd);
    let result = b.mul(normed, w);

    // store
    let out_addr = b.add_ptr(out_ptr, x_off);
    b.store(out_addr, result, Some(mask));

    let _ = (block_size, eps); // used as constexpr/arg
}

/// Fused residual_add + RMSNorm:
///   new_residual = x + residual
///   normed = rmsnorm(new_residual, weight, eps)
/// Writes both new_residual and normed.
///
/// Grid: one program per row.
pub fn build_fused_residual_rmsnorm(b: &mut KernelBuilder) {
    let x_ptr = b.arg_ptr("x_ptr", ScalarType::F32);
    let res_ptr = b.arg_ptr("residual_ptr", ScalarType::F32);
    let w_ptr = b.arg_ptr("weight_ptr", ScalarType::F32);
    let normed_ptr = b.arg_ptr("normed_ptr", ScalarType::F32);
    let res_out_ptr = b.arg_ptr("residual_out_ptr", ScalarType::F32);
    let n_cols = b.arg_i32("n_cols");
    let eps = b.arg_f32("eps");
    let _block_size = b.constexpr("BLOCK_SIZE", 1024);

    let pid = b.program_id(0);
    let offs = b.arange(0, 1024);
    let mask = b.cmp_lt(offs, n_cols);
    let zero = b.constant_f32(0.0);

    let row_start = b.mul(pid, n_cols);
    let idx = b.add(row_start, offs);

    // Load x and residual
    let x_addr = b.add_ptr(x_ptr, idx);
    let x = b.load(x_addr, Some(mask), Some(zero));
    let res_addr = b.add_ptr(res_ptr, idx);
    let residual = b.load(res_addr, Some(mask), Some(zero));

    // new_residual = x + residual
    let new_res = b.add(x, residual);

    // Store new residual
    let res_out_addr = b.add_ptr(res_out_ptr, idx);
    b.store(res_out_addr, new_res, Some(mask));

    // RMSNorm on new_residual
    let sq = b.mul(new_res, new_res);
    let sum_sq = b.reduce_sum(sq, 0);
    let n_f = b.cast(n_cols, ScalarType::F32);
    let var = b.div(sum_sq, n_f);
    let var_eps = b.add(var, eps);
    let rstd = b.rsqrt(var_eps);

    let w_addr = b.add_ptr(w_ptr, offs);
    let w = b.load(w_addr, Some(mask), Some(zero));

    let normed = b.mul(new_res, rstd);
    let result = b.mul(normed, w);

    let out_addr = b.add_ptr(normed_ptr, idx);
    b.store(out_addr, result, Some(mask));

}

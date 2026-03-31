use crate::ir::ScalarType;
use crate::builder::KernelBuilder;


/// Rotary Position Embedding: apply complex rotation to Q and K.
///
/// Grid: one program per (token, head). Tiles over head_dim/2 pairs.
/// For each pair (x[2i], x[2i+1]):
///   x_new[2i]   = x[2i] * cos[i] - x[2i+1] * sin[i]
///   x_new[2i+1] = x[2i] * sin[i] + x[2i+1] * cos[i]
pub fn build_rope(b: &mut KernelBuilder) {
    let q_ptr = b.arg_ptr("q_ptr", ScalarType::F32);
    let k_ptr = b.arg_ptr("k_ptr", ScalarType::F32);
    let cos_ptr = b.arg_ptr("cos_ptr", ScalarType::F32);
    let sin_ptr = b.arg_ptr("sin_ptr", ScalarType::F32);
    let num_heads = b.arg_i32("num_heads");
    let num_kv_heads = b.arg_i32("num_kv_heads");
    let head_dim = b.arg_i32("head_dim");
    let _block = b.constexpr("BLOCK_SIZE", 64);

    let pid_token = b.program_id(0);
    let pid_head = b.program_id(1);

    let offs = b.arange(0, 64); // pair index within head_dim/2
    let two = b.constant_i32(2);
    let half_dim = b.div(head_dim, two);
    let mask = b.cmp_lt(offs, half_dim);

    // Even and odd indices
    let even_idx = b.mul(offs, two);
    let one = b.constant_i32(1);
    let odd_idx = b.add(even_idx, one);

    // Q base offset: (pid_token * num_heads + pid_head) * head_dim
    let q_head_off = b.mul(pid_token, num_heads);
    let q_head_off = b.add(q_head_off, pid_head);
    let q_base = b.mul(q_head_off, head_dim);
    let q_even = b.add(q_base, even_idx);
    let q_odd = b.add(q_base, odd_idx);

    let zero = b.constant_f32(0.0);

    // Load Q pairs
    let q_even_addr = b.add_ptr(q_ptr, q_even);
    let q_odd_addr = b.add_ptr(q_ptr, q_odd);
    let qe = b.load(q_even_addr, Some(mask), Some(zero));
    let qo = b.load(q_odd_addr, Some(mask), Some(zero));

    // Load cos/sin (indexed by pair position)
    let cos_addr = b.add_ptr(cos_ptr, offs);
    let sin_addr = b.add_ptr(sin_ptr, offs);
    let cos = b.load(cos_addr, Some(mask), Some(zero));
    let sin = b.load(sin_addr, Some(mask), Some(zero));

    // Rotate Q
    let qe_cos = b.mul(qe, cos);
    let qo_sin = b.mul(qo, sin);
    let new_qe = b.sub(qe_cos, qo_sin);

    let qe_sin = b.mul(qe, sin);
    let qo_cos = b.mul(qo, cos);
    let new_qo = b.add(qe_sin, qo_cos);

    // Store rotated Q
    b.store(q_even_addr, new_qe, Some(mask));
    b.store(q_odd_addr, new_qo, Some(mask));

    // Same for K, using num_kv_heads
    let k_head_off = b.mul(pid_token, num_kv_heads);
    let k_head_off = b.add(k_head_off, pid_head);
    let k_base = b.mul(k_head_off, head_dim);
    let k_even = b.add(k_base, even_idx);
    let k_odd = b.add(k_base, odd_idx);

    // Only rotate K if pid_head < num_kv_heads (GQA support)
    let kv_mask = b.cmp_lt(pid_head, num_kv_heads);

    let ke_addr = b.add_ptr(k_ptr, k_even);
    let ko_addr = b.add_ptr(k_ptr, k_odd);
    let ke = b.load(ke_addr, Some(kv_mask), Some(zero));
    let ko = b.load(ko_addr, Some(kv_mask), Some(zero));

    let ke_cos = b.mul(ke, cos);
    let ko_sin = b.mul(ko, sin);
    let new_ke = b.sub(ke_cos, ko_sin);

    let ke_sin = b.mul(ke, sin);
    let ko_cos = b.mul(ko, cos);
    let new_ko = b.add(ke_sin, ko_cos);

    b.store(ke_addr, new_ke, Some(kv_mask));
    b.store(ko_addr, new_ko, Some(kv_mask));

}

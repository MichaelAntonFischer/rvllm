use crate::ir::ScalarType;
use crate::builder::KernelBuilder;


/// Standard tiled GEMM: C[M,N] = A[M,K] @ B[K,N]
///
/// Grid: (ceil(M/BLOCK_M), ceil(N/BLOCK_N))
/// Each program computes one BLOCK_M x BLOCK_N output tile.
/// Main loop iterates K in BLOCK_K steps, loading A/B tiles and accumulating via dot.
pub fn build_tiled_gemm(b: &mut KernelBuilder) {
    let a_ptr = b.arg_ptr("a_ptr", ScalarType::F32);
    let b_ptr = b.arg_ptr("b_ptr", ScalarType::F32);
    let c_ptr = b.arg_ptr("c_ptr", ScalarType::F32);
    let m = b.arg_i32("M");
    let n = b.arg_i32("N");
    let k = b.arg_i32("K");
    let _bm = b.constexpr("BLOCK_M", 128);
    let _bn = b.constexpr("BLOCK_N", 128);
    let _bk = b.constexpr("BLOCK_K", 32);

    let pid_m = b.program_id(0);
    let pid_n = b.program_id(1);

    // Tile offsets
    let bm = b.constant_i32(128);
    let bn = b.constant_i32(128);
    let bk = b.constant_i32(32);

    let m_start = b.mul(pid_m, bm);
    let n_start = b.mul(pid_n, bn);

    // Row/col offsets within tile
    let rm = b.arange(0, 128); // [0..BLOCK_M)
    let rn = b.arange(0, 128); // [0..BLOCK_N)

    let row = b.add(m_start, rm);
    let col = b.add(n_start, rn);

    // Initialize accumulator to zero
    let zero = b.constant_f32(0.0);
    let mut acc = zero;

    // K-loop body (represents one iteration; actual loop is lowered in codegen)
    // For the IR, we show loading one (BLOCK_M, BLOCK_K) tile and one (BLOCK_K, BLOCK_N) tile
    let rk = b.arange(0, 32); // [0..BLOCK_K)

    // A tile: a_ptr[row, rk] = a_ptr + row * K + rk
    let a_row_off = b.mul(row, k);
    let a_idx = b.add(a_row_off, rk);
    let a_addr = b.add_ptr(a_ptr, a_idx);
    let a_tile = b.load(a_addr, None, None);

    // B tile: b_ptr[rk, col] = b_ptr + rk * N + col
    let b_row_off = b.mul(rk, n);
    let b_idx = b.add(b_row_off, col);
    let b_addr = b.add_ptr(b_ptr, b_idx);
    let b_tile = b.load(b_addr, None, None);

    // Accumulate: acc += dot(a_tile, b_tile)
    acc = b.dot(a_tile, b_tile, acc);

    // Store result: c_ptr[row, col] = c_ptr + row * N + col
    let c_row_off = b.mul(row, n);
    let c_idx = b.add(c_row_off, col);
    let c_addr = b.add_ptr(c_ptr, c_idx);

    let row_mask = b.cmp_lt(row, m);
    let col_mask = b.cmp_lt(col, n);
    // Combined mask (approximate -- real impl would use 2D masking)
    let _ = col_mask;
    b.store(c_addr, acc, Some(row_mask));

    let _ = bk;
}

/// GEMV for M=1 (single-token decode): y[N] = x[K] @ W[K,N]
///
/// Grid: (ceil(N/BLOCK_N),)
/// Each program handles BLOCK_N output elements, looping over K.
pub fn build_gemv(b: &mut KernelBuilder) {
    let x_ptr = b.arg_ptr("x_ptr", ScalarType::F32);
    let w_ptr = b.arg_ptr("w_ptr", ScalarType::F32);
    let y_ptr = b.arg_ptr("y_ptr", ScalarType::F32);
    let n_arg = b.arg_i32("N");
    let k_arg = b.arg_i32("K");
    let _bn = b.constexpr("BLOCK_N", 128);
    let _bk = b.constexpr("BLOCK_K", 32);

    let pid = b.program_id(0);
    let block_n = b.constant_i32(128);
    let n_start = b.mul(pid, block_n);
    let rn = b.arange(0, 128);
    let col = b.add(n_start, rn);
    let col_mask = b.cmp_lt(col, n_arg);

    let zero = b.constant_f32(0.0);
    let mut acc = zero;

    // K-loop body (one iteration of BLOCK_K)
    let rk = b.arange(0, 32);
    let k_mask = b.cmp_lt(rk, k_arg);

    // x[rk] -- vector load
    let x_addr = b.add_ptr(x_ptr, rk);
    let x_val = b.load(x_addr, Some(k_mask), Some(zero));

    // W[rk, col] = w_ptr + rk * N + col
    let w_row = b.mul(rk, n_arg);
    let w_idx = b.add(w_row, col);
    let w_addr = b.add_ptr(w_ptr, w_idx);
    let w_val = b.load(w_addr, Some(k_mask), Some(zero));

    // Partial dot product: acc += x * W (element-wise, then reduce over K)
    let prod = b.mul(x_val, w_val);
    let partial = b.reduce_sum(prod, 0);
    acc = b.add(acc, partial);

    // Store y[col]
    let y_addr = b.add_ptr(y_ptr, col);
    b.store(y_addr, acc, Some(col_mask));

}

/// Persistent GEMM (stream-K style): programs loop over multiple output tiles.
///
/// Grid: (NUM_SM,)
/// Each program processes: for tile_id in (pid, total_tiles, NUM_SM)
pub fn build_persistent_gemm(b: &mut KernelBuilder) {
    let a_ptr = b.arg_ptr("a_ptr", ScalarType::F32);
    let b_ptr = b.arg_ptr("b_ptr", ScalarType::F32);
    let c_ptr = b.arg_ptr("c_ptr", ScalarType::F32);
    let m = b.arg_i32("M");
    let n = b.arg_i32("N");
    let k = b.arg_i32("K");
    let _bm = b.constexpr("BLOCK_M", 128);
    let _bn = b.constexpr("BLOCK_N", 128);
    let _bk = b.constexpr("BLOCK_K", 32);
    let _num_sm = b.constexpr("NUM_SM", 108);

    let pid = b.program_id(0);

    // Total number of tiles
    let bm = b.constant_i32(128);
    let bn = b.constant_i32(128);

    // tiles_m = ceil(M / BLOCK_M), tiles_n = ceil(N / BLOCK_N)
    // Approximate: tiles_m = (M + BM - 1) / BM
    let bm_minus1 = b.constant_i32(127);
    let m_padded = b.add(m, bm_minus1);
    let tiles_m = b.div(m_padded, bm);

    let bn_minus1 = b.constant_i32(127);
    let n_padded = b.add(n, bn_minus1);
    let tiles_n = b.div(n_padded, bn);

    let total_tiles = b.mul(tiles_m, tiles_n);

    // This program's tile = pid (first tile; loop would increment by NUM_SM)
    // Decompose tile_id -> (tile_m, tile_n)
    let tile_m = b.div(pid, tiles_n);
    let tile_m_times_n = b.mul(tile_m, tiles_n);
    let tile_n = b.sub(pid, tile_m_times_n);

    let m_start = b.mul(tile_m, bm);
    let n_start = b.mul(tile_n, bn);

    let rm = b.arange(0, 128);
    let rn = b.arange(0, 128);
    let row = b.add(m_start, rm);
    let col = b.add(n_start, rn);

    let zero = b.constant_f32(0.0);
    let rk = b.arange(0, 32);
    let a_row_off = b.mul(row, k);
    let a_idx = b.add(a_row_off, rk);
    let a_addr = b.add_ptr(a_ptr, a_idx);
    let a_tile = b.load(a_addr, None, None);

    let b_row_off = b.mul(rk, n);
    let b_idx = b.add(b_row_off, col);
    let b_addr = b.add_ptr(b_ptr, b_idx);
    let b_tile = b.load(b_addr, None, None);

    let acc = b.dot(a_tile, b_tile, zero);

    let c_row_off = b.mul(row, n);
    let c_idx = b.add(c_row_off, col);
    let c_addr = b.add_ptr(c_ptr, c_idx);
    let row_mask = b.cmp_lt(row, m);
    b.store(c_addr, acc, Some(row_mask));

    let _ = total_tiles;
}

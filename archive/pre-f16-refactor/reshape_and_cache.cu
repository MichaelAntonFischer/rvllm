// Reshape and cache kernel: scatter per-token K/V into paged cache.
//
// For each token, copies its KV vector into the paged cache at the position
// given by slot_mapping[token_idx].
//
// Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
// Input layout: [num_tokens, num_kv_heads, head_dim]
// slot_mapping:  [num_tokens] -- each entry is (block_idx * block_size + block_offset)
//
// Launch config:
//   Grid:  (num_tokens, 1, 1)
//   Block: (min(num_kv_heads * head_dim, 1024), 1, 1)
//   Shared memory: none

extern "C"
__global__ void reshape_and_cache_kernel(
    float* __restrict__ key_cache,        // [num_blocks, block_size, num_kv_heads, head_dim]
    float* __restrict__ value_cache,      // [num_blocks, block_size, num_kv_heads, head_dim]
    const float* __restrict__ key,        // [num_tokens, num_kv_heads, head_dim]
    const float* __restrict__ value,      // [num_tokens, num_kv_heads, head_dim]
    const int* __restrict__ slot_mapping, // [num_tokens]
    int num_tokens,
    int num_kv_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int tid = threadIdx.x;
    const int kv_dim = num_kv_heads * head_dim;
    const int slot = slot_mapping[token_idx];

    // slot = block_idx * block_size + block_offset
    // Cache element index: slot * kv_dim + element_within_kv
    const int cache_offset = slot * kv_dim;
    const int src_offset = token_idx * kv_dim;

    for (int i = tid; i < kv_dim; i += blockDim.x) {
        key_cache[cache_offset + i] = key[src_offset + i];
        value_cache[cache_offset + i] = value[src_offset + i];
    }
}

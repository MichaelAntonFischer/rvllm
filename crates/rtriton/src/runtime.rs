// rTriton runtime: kernel loading, caching, launching via CUDA driver API.
// Only compiled with cfg(feature = "cuda").

use std::collections::HashMap;
use parking_lot::Mutex;

#[derive(Debug, Clone)]
pub enum KernelArg {
    DevicePtr(u64),
    I32(i32),
    U32(u32),
    F32(f32),
    U64(u64),
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("CUDA error: {0}")]
    Cuda(String),
    #[error("kernel not found: {0}")]
    NotFound(String),
    #[error("launch error: {0}")]
    Launch(String),
}

pub struct CompiledKernel {
    pub name: String,
    pub ptx: String,
    pub smem_bytes: u32,
    pub block_dim: (u32, u32, u32),
    pub grid_fn: Box<dyn Fn(&[usize]) -> (u32, u32, u32) + Send + Sync>,
}

struct CachedEntry {
    _ptx: String,
    // With real CUDA: CUmodule + CUfunction handles
}

pub struct KernelCache {
    cache: Mutex<HashMap<String, CachedEntry>>,
}

impl KernelCache {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for KernelCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Launch a compiled kernel.
///
/// Real CUDA path (TODO):
/// 1. Load PTX into CUmodule (cached)
/// 2. Get CUfunction by name
/// 3. Pack args into void** array
/// 4. cuLaunchKernel(func, grid, block, args, smem, stream)
pub fn launch(
    cache: &KernelCache,
    kernel: &CompiledKernel,
    args: &[KernelArg],
    problem_shape: &[usize],
) -> Result<(), RuntimeError> {
    let grid = (kernel.grid_fn)(problem_shape);

    // Ensure kernel is cached
    {
        let mut c = cache.cache.lock();
        if !c.contains_key(&kernel.name) {
            c.insert(
                kernel.name.clone(),
                CachedEntry {
                    _ptx: kernel.ptx.clone(),
                },
            );
        }
    }

    tracing::debug!(
        kernel = %kernel.name,
        grid = ?grid,
        block = ?kernel.block_dim,
        smem = kernel.smem_bytes,
        n_args = args.len(),
        "rtriton::launch (stub -- no GPU call)"
    );

    Ok(())
}

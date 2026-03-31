use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

const SMEM_LIMIT: usize = 48 * 1024;

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub num_warps: u32,
    pub num_stages: u32,
    pub split_k: u32,
    pub extras: BTreeMap<String, u32>,
}

pub struct TuneResult {
    pub best_config: Config,
    pub best_time_us: f64,
    pub all_results: Vec<(Config, f64)>,
}

pub struct PersistentCache {
    path: PathBuf,
    entries: HashMap<String, Config>,
}

impl Config {
    pub fn new(block_m: u32, block_n: u32, block_k: u32, num_warps: u32, num_stages: u32) -> Self {
        Self {
            block_m,
            block_n,
            block_k,
            num_warps,
            num_stages,
            split_k: 1,
            extras: BTreeMap::new(),
        }
    }

    /// Shared memory bytes: (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * dtype_size * num_stages
    pub fn smem_bytes(&self, dtype_size: usize) -> usize {
        let tiles = (self.block_m as usize * self.block_k as usize)
            + (self.block_k as usize * self.block_n as usize);
        tiles * dtype_size * self.num_stages as usize
    }
}

impl PersistentCache {
    pub fn load_or_create(path: &Path) -> Self {
        let entries = if path.exists() {
            match std::fs::read_to_string(path) {
                Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
                Err(_) => HashMap::new(),
            }
        } else {
            HashMap::new()
        };
        Self {
            path: path.to_path_buf(),
            entries,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Config> {
        self.entries.get(key)
    }

    pub fn insert(&mut self, key: &str, config: Config) {
        self.entries.insert(key.to_string(), config);
    }

    /// Atomic write: tmp file + rename to avoid partial reads.
    pub fn save(&self) -> Result<(), std::io::Error> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.entries)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, json.as_bytes())?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    pub fn cache_key(kernel_name: &str, shape: &[usize], device: &str) -> String {
        use std::fmt::Write;
        let mut key = String::new();
        let _ = write!(key, "{}:", kernel_name);
        for (i, s) in shape.iter().enumerate() {
            if i > 0 {
                key.push('x');
            }
            let _ = write!(key, "{}", s);
        }
        let _ = write!(key, ":{}", device);
        key
    }

    pub fn default_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        PathBuf::from(home)
            .join(".cache")
            .join("rtriton")
            .join("autotune.json")
    }
}

/// ~50 GEMM configs after pruning.
/// Iterates BLOCK_M in [16,32,64,128], BLOCK_N in [32,64,128], BLOCK_K in [16,32,64],
/// num_warps in [2,4,8], num_stages in [2,3,4].
/// Prunes configs where smem > 48KB (fp16) or BLOCK_M*BLOCK_N/32 < num_warps.
pub fn default_gemm_configs() -> Vec<Config> {
    let block_ms = [16u32, 32, 64, 128];
    let block_ns = [32u32, 64, 128];
    let block_ks = [16u32, 32, 64];
    let warps = [2u32, 4, 8];
    let stages = [2u32, 3, 4];
    let dtype_size = 2; // fp16

    let mut configs = Vec::with_capacity(64);
    for &bm in &block_ms {
        for &bn in &block_ns {
            for &bk in &block_ks {
                for &nw in &warps {
                    // Tiles per block must be >= num_warps
                    let tiles_per_block = (bm * bn) / 32;
                    if tiles_per_block < nw {
                        continue;
                    }
                    for &ns in &stages {
                        let cfg = Config::new(bm, bn, bk, nw, ns);
                        if cfg.smem_bytes(dtype_size) > SMEM_LIMIT {
                            continue;
                        }
                        configs.push(cfg);
                    }
                }
            }
        }
    }
    configs
}

/// ~4 elementwise configs: varying BLOCK_SIZE.
pub fn default_elementwise_configs() -> Vec<Config> {
    [256, 512, 1024, 2048]
        .iter()
        .map(|&bs| {
            let mut cfg = Config::new(bs, 1, 1, 4, 1);
            cfg.extras.insert("BLOCK_SIZE".into(), bs);
            cfg
        })
        .collect()
}

/// ~4 reduction configs: varying block dimensions.
pub fn default_reduction_configs() -> Vec<Config> {
    [(256, 4), (512, 4), (1024, 8), (2048, 8)]
        .iter()
        .map(|&(bs, nw)| {
            let mut cfg = Config::new(bs, 1, 1, nw, 1);
            cfg.extras.insert("BLOCK_SIZE".into(), bs);
            cfg
        })
        .collect()
}

fn round_up_pow2(v: usize) -> u32 {
    if v == 0 {
        return 1;
    }
    let v = v as u64;
    // next power of two
    1u32 << (64 - (v - 1).leading_zeros())
}

/// Heuristic GEMM config selection (no benchmarking).
pub fn heuristic_gemm_config(m: usize, n: usize, k: usize, _sm_count: u32) -> Config {
    let block_m = round_up_pow2(m).min(128);
    let block_n = round_up_pow2(n).min(128);
    let block_k = 32u32;
    let num_warps = 4u32;
    let dtype_size = 2usize; // fp16

    // smem per stage = (BM*BK + BK*BN) * dtype
    let smem_per_stage =
        (block_m as usize * block_k as usize + block_k as usize * block_n as usize) * dtype_size;
    let num_stages = if smem_per_stage == 0 {
        1u32
    } else {
        (SMEM_LIMIT / smem_per_stage).min(4).max(1) as u32
    };

    let _ = k; // k informs split_k in future
    Config::new(block_m, block_n, block_k, num_warps, num_stages)
}

/// Heuristic elementwise config selection.
pub fn heuristic_elementwise_config(numel: usize) -> Config {
    let bs = if numel <= 1024 {
        256
    } else if numel <= 65536 {
        512
    } else if numel <= 1_048_576 {
        1024
    } else {
        2048
    };
    let mut cfg = Config::new(bs, 1, 1, 4, 1);
    cfg.extras.insert("BLOCK_SIZE".into(), bs);
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smem_bytes_correct() {
        let c = Config::new(128, 128, 32, 4, 3);
        // (128*32 + 32*128) * 2 * 3 = 8192 * 2 * 3 = 49152
        assert_eq!(c.smem_bytes(2), 49152);
    }

    #[test]
    fn gemm_configs_nonempty_and_within_smem() {
        let cfgs = default_gemm_configs();
        assert!(!cfgs.is_empty());
        for c in &cfgs {
            assert!(c.smem_bytes(2) <= SMEM_LIMIT);
            assert!((c.block_m * c.block_n) / 32 >= c.num_warps);
        }
    }

    #[test]
    fn gemm_configs_count_reasonable() {
        let cfgs = default_gemm_configs();
        // 4*3*3*3*3 = 324 combos, smem pruning removes ~45
        assert!(cfgs.len() >= 200 && cfgs.len() <= 324, "got {}", cfgs.len());
    }

    #[test]
    fn heuristic_gemm_basic() {
        let c = heuristic_gemm_config(64, 64, 512, 80);
        assert_eq!(c.block_m, 64);
        assert_eq!(c.block_n, 64);
        assert_eq!(c.block_k, 32);
        assert!(c.smem_bytes(2) <= SMEM_LIMIT);
    }

    #[test]
    fn heuristic_gemm_small() {
        let c = heuristic_gemm_config(1, 4096, 4096, 80);
        assert_eq!(c.block_m, 1);
        assert_eq!(c.block_n, 128);
    }

    #[test]
    fn elementwise_configs() {
        let cfgs = default_elementwise_configs();
        assert_eq!(cfgs.len(), 4);
        assert!(cfgs[0].extras.contains_key("BLOCK_SIZE"));
    }

    #[test]
    fn reduction_configs() {
        let cfgs = default_reduction_configs();
        assert_eq!(cfgs.len(), 4);
    }

    #[test]
    fn cache_key_format() {
        let key = PersistentCache::cache_key("matmul_fp16", &[4096, 4096, 4096], "cuda:0");
        assert_eq!(key, "matmul_fp16:4096x4096x4096:cuda:0");
    }

    #[test]
    fn round_up_pow2_cases() {
        assert_eq!(round_up_pow2(1), 1);
        assert_eq!(round_up_pow2(3), 4);
        assert_eq!(round_up_pow2(64), 64);
        assert_eq!(round_up_pow2(65), 128);
        assert_eq!(round_up_pow2(0), 1);
    }

    #[test]
    fn persistent_cache_roundtrip() {
        let dir = std::env::temp_dir().join("rtriton_test_cache");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("autotune.json");

        let mut cache = PersistentCache::load_or_create(&path);
        let cfg = Config::new(128, 128, 32, 4, 3);
        cache.insert("test_key", cfg.clone());
        cache.save().unwrap();

        let cache2 = PersistentCache::load_or_create(&path);
        assert_eq!(cache2.get("test_key"), Some(&cfg));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn heuristic_elementwise_small() {
        let c = heuristic_elementwise_config(512);
        assert_eq!(*c.extras.get("BLOCK_SIZE").unwrap(), 256);
    }

    #[test]
    fn heuristic_elementwise_large() {
        let c = heuristic_elementwise_config(10_000_000);
        assert_eq!(*c.extras.get("BLOCK_SIZE").unwrap(), 2048);
    }
}

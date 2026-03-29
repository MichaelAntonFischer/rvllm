//! GPU model weights container backed by CUDA device memory (f16 only).
//!
//! Holds all model weight tensors as `CudaSlice<f16>` on a single device,
//! with shape metadata for downstream layers to query dimensions.

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::HashMap;
    use std::sync::Arc;

    use cudarc::driver::{CudaSlice, CudaStream, DeviceSlice as _};
    use half::f16;
    use rvllm_core::error::{LLMError, Result};
    use tracing::debug;

    /// Container holding all model weights as f16 CUDA device buffers.
    pub struct GpuModelWeights {
        weights: HashMap<String, CudaSlice<f16>>,
        shapes: HashMap<String, Vec<usize>>,
    }

    impl GpuModelWeights {
        /// Build from pre-loaded f16 weight maps.
        pub fn new(
            weights: HashMap<String, CudaSlice<f16>>,
            shapes: HashMap<String, Vec<usize>>,
        ) -> Self {
            debug!(num_weights = weights.len(), "GpuModelWeights created (f16)");
            Self { weights, shapes }
        }

        /// Build an empty container.
        pub fn empty() -> Self {
            Self {
                weights: HashMap::new(),
                shapes: HashMap::new(),
            }
        }

        /// Insert a single f16 weight tensor with its shape.
        pub fn insert(&mut self, name: String, data: CudaSlice<f16>, shape: Vec<usize>) {
            self.shapes.insert(name.clone(), shape);
            self.weights.insert(name, data);
        }

        /// Look up a weight by name.
        pub fn get(&self, name: &str) -> Option<&CudaSlice<f16>> {
            self.weights.get(name)
        }

        /// Look up a weight by name, returning an error if missing.
        pub fn require(&self, name: &str) -> Result<&CudaSlice<f16>> {
            self.weights
                .get(name)
                .ok_or_else(|| LLMError::GpuError(format!("weight not found: {}", name)))
        }

        /// Look up the shape of a weight by name.
        pub fn shape(&self, name: &str) -> Option<&[usize]> {
            self.shapes.get(name).map(|v| v.as_slice())
        }

        /// Look up shape, returning an error if missing.
        pub fn require_shape(&self, name: &str) -> Result<&[usize]> {
            self.shapes
                .get(name)
                .map(|v| v.as_slice())
                .ok_or_else(|| LLMError::GpuError(format!("shape not found: {}", name)))
        }

        /// Number of weight tensors stored.
        pub fn num_weights(&self) -> usize {
            self.weights.len()
        }

        /// Iterate over all weight names.
        pub fn names(&self) -> impl Iterator<Item = &str> {
            self.weights.keys().map(|s| s.as_str())
        }

        /// Check whether a weight exists.
        pub fn contains(&self, name: &str) -> bool {
            self.weights.contains_key(name)
        }

        /// Total GPU memory used by all weight buffers, in bytes.
        pub fn total_bytes(&self) -> usize {
            self.weights
                .values()
                .map(|s| s.len() * std::mem::size_of::<f16>())
                .sum()
        }

        /// Build from a host-side f16 weight map by uploading each tensor to GPU.
        pub fn from_host(
            host_weights: HashMap<String, Vec<f16>>,
            shapes: HashMap<String, Vec<usize>>,
            stream: &Arc<CudaStream>,
        ) -> Result<Self> {
            let mut gpu_weights = HashMap::with_capacity(host_weights.len());
            for (name, data) in host_weights {
                let slice = stream.clone_htod(&data).map_err(|e| {
                    LLMError::GpuError(format!("htod copy failed for {}: {}", name, e))
                })?;
                gpu_weights.insert(name, slice);
            }
            debug!(
                num_weights = gpu_weights.len(),
                "GpuModelWeights uploaded from host (f16)"
            );
            Ok(Self {
                weights: gpu_weights,
                shapes,
            })
        }

        /// Consume the container and return the underlying maps.
        pub fn into_parts(self) -> (HashMap<String, CudaSlice<f16>>, HashMap<String, Vec<usize>>) {
            (self.weights, self.shapes)
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::GpuModelWeights;

#[cfg(test)]
mod tests {
    #[test]
    fn module_compiles() {
        assert!(true);
    }
}

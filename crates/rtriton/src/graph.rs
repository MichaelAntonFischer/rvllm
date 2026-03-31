// rTriton CUDA graph capture/replay
//
// Data structures and allocation planning for multi-kernel execution graphs.
// Supports mixed Triton JIT kernels and cuBLAS GEMMs in a single graph.
// No actual CUDA calls here -- those live in runtime.rs behind cfg(feature = "cuda").

use std::collections::HashMap;
use crate::cublas_gemm::GemmOp;

const ALIGN: usize = 256;

fn align_up(x: usize) -> usize {
    (x + ALIGN - 1) & !(ALIGN - 1)
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub u32);

#[derive(Debug, Clone)]
pub struct BufferInfo {
    pub handle: BufferHandle,
    pub size_bytes: usize,
    pub first_write: u32,
    pub last_read: u32,
}

#[derive(Debug, Clone)]
pub enum KernelArg {
    Buffer(BufferHandle),
    ExternalPtr(u64),
    I32(i32),
    U32(u32),
    F32(f32),
    U64(u64),
}

#[derive(Debug, Clone)]
pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub smem_bytes: u32,
}

#[derive(Debug, Clone)]
pub struct KernelCall {
    pub kernel_name: String,
    pub config: LaunchConfig,
    pub args: Vec<KernelArg>,
    pub reads: Vec<BufferHandle>,
    pub writes: Vec<BufferHandle>,
}

/// A cuBLAS GEMM call in the graph (uses our cuBLAS tricks: plan cache, autotuned algo).
#[derive(Debug, Clone)]
pub struct CublasCall {
    pub gemm: GemmOp,
    pub a_ptr: KernelArg,  // input activations
    pub b_ptr: KernelArg,  // weights
    pub c_ptr: KernelArg,  // output
    pub reads: Vec<BufferHandle>,
    pub writes: Vec<BufferHandle>,
}

/// A node in the execution graph -- either a Triton JIT kernel or a cuBLAS GEMM.
/// Both get captured into the same CUDA graph for zero launch overhead.
#[derive(Debug, Clone)]
pub enum GraphNode {
    Triton(KernelCall),
    Cublas(CublasCall),
}

impl GraphNode {
    pub fn name(&self) -> &str {
        match self {
            GraphNode::Triton(k) => &k.kernel_name,
            GraphNode::Cublas(c) => &c.gemm.name,
        }
    }

    pub fn reads(&self) -> &[BufferHandle] {
        match self {
            GraphNode::Triton(k) => &k.reads,
            GraphNode::Cublas(c) => &c.reads,
        }
    }

    pub fn writes(&self) -> &[BufferHandle] {
        match self {
            GraphNode::Triton(k) => &k.writes,
            GraphNode::Cublas(c) => &c.writes,
        }
    }
}

// ---------------------------------------------------------------------------
// KernelGraph
// ---------------------------------------------------------------------------

pub struct KernelGraph {
    pub nodes: Vec<GraphNode>,
    pub buffers: Vec<BufferInfo>,
    pub input_buffers: Vec<BufferHandle>,
    pub output_buffers: Vec<BufferHandle>,
}

impl KernelGraph {
    /// Greedy interval-coloring allocator.
    ///
    /// Sort buffers by first_write, then assign each to the lowest offset where it
    /// doesn't overlap any already-placed buffer whose liveness intersects.
    pub fn compute_allocation_plan(&self) -> AllocationPlan {
        // Sort buffer indices by first_write (stable, ties broken by index).
        let mut order: Vec<usize> = (0..self.buffers.len()).collect();
        order.sort_by_key(|&i| self.buffers[i].first_write);

        // Placed intervals: (offset, aligned_size, first_write, last_read).
        let mut placed: Vec<(usize, usize, u32, u32)> = Vec::with_capacity(self.buffers.len());
        let mut offsets: HashMap<BufferHandle, usize> = HashMap::new();
        let mut total_bytes: usize = 0;

        for &idx in &order {
            let buf = &self.buffers[idx];
            let size = align_up(buf.size_bytes);
            let fw = buf.first_write;
            let lr = buf.last_read;

            // Collect occupied intervals that overlap this buffer's lifetime.
            // Two intervals overlap iff !(a.last < b.first || b.last < a.first).
            let mut occupied: Vec<(usize, usize)> = placed
                .iter()
                .filter(|&&(_, _, pfw, plr)| !(plr < fw || lr < pfw))
                .map(|&(off, sz, _, _)| (off, sz))
                .collect();
            occupied.sort_by_key(|&(off, _)| off);

            // Find first gap.
            let mut offset = 0usize;
            for &(occ_off, occ_sz) in &occupied {
                if offset + size <= occ_off {
                    break;
                }
                let end = occ_off + occ_sz;
                if end > offset {
                    offset = end;
                }
            }

            offsets.insert(buf.handle, offset);
            placed.push((offset, size, fw, lr));

            let need = offset + size;
            if need > total_bytes {
                total_bytes = need;
            }
        }

        AllocationPlan {
            total_bytes,
            offsets,
        }
    }
}

// ---------------------------------------------------------------------------
// AllocationPlan
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct AllocationPlan {
    pub total_bytes: usize,
    pub offsets: HashMap<BufferHandle, usize>,
}

// ---------------------------------------------------------------------------
// GraphBuilder
// ---------------------------------------------------------------------------

pub struct GraphBuilder {
    nodes: Vec<GraphNode>,
    buffers: Vec<BufferInfo>,
    next_buffer: u32,
    next_node: u32,
    input_buffers: Vec<BufferHandle>,
    output_buffers: Vec<BufferHandle>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            buffers: Vec::new(),
            next_buffer: 0,
            next_node: 0,
            input_buffers: Vec::new(),
            output_buffers: Vec::new(),
        }
    }

    pub fn allocate_buffer(&mut self, size_bytes: usize) -> BufferHandle {
        let id = self.next_buffer;
        self.next_buffer += 1;
        let handle = BufferHandle(id);
        self.buffers.push(BufferInfo {
            handle,
            size_bytes,
            first_write: u32::MAX,
            last_read: 0,
        });
        handle
    }

    pub fn mark_input(&mut self, h: BufferHandle) {
        self.input_buffers.push(h);
    }

    pub fn mark_output(&mut self, h: BufferHandle) {
        self.output_buffers.push(h);
    }

    fn update_liveness(&mut self, reads: &[BufferHandle], writes: &[BufferHandle]) {
        let kid = self.next_node;
        for &h in reads {
            if let Some(buf) = self.buffers.iter_mut().find(|b| b.handle == h) {
                if kid < buf.first_write {
                    buf.first_write = kid;
                }
                if kid > buf.last_read {
                    buf.last_read = kid;
                }
            }
        }
        for &h in writes {
            if let Some(buf) = self.buffers.iter_mut().find(|b| b.handle == h) {
                if kid < buf.first_write {
                    buf.first_write = kid;
                }
                if kid > buf.last_read {
                    buf.last_read = kid;
                }
            }
        }
    }

    /// Add a Triton JIT kernel to the graph.
    pub fn launch_kernel(
        &mut self,
        name: &str,
        config: LaunchConfig,
        args: Vec<KernelArg>,
        reads: Vec<BufferHandle>,
        writes: Vec<BufferHandle>,
    ) {
        self.update_liveness(&reads, &writes);
        self.next_node += 1;

        self.nodes.push(GraphNode::Triton(KernelCall {
            kernel_name: name.to_string(),
            config,
            args,
            reads,
            writes,
        }));
    }

    /// Add a cuBLAS GEMM to the graph.
    pub fn launch_cublas(
        &mut self,
        gemm: GemmOp,
        a_ptr: KernelArg,
        b_ptr: KernelArg,
        c_ptr: KernelArg,
        reads: Vec<BufferHandle>,
        writes: Vec<BufferHandle>,
    ) {
        self.update_liveness(&reads, &writes);
        self.next_node += 1;

        self.nodes.push(GraphNode::Cublas(CublasCall {
            gemm,
            a_ptr,
            b_ptr,
            c_ptr,
            reads,
            writes,
        }));
    }

    pub fn build(self) -> KernelGraph {
        KernelGraph {
            nodes: self.nodes,
            buffers: self.buffers,
            input_buffers: self.input_buffers,
            output_buffers: self.output_buffers,
        }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GraphExecutor
// ---------------------------------------------------------------------------

pub struct GraphExecutor {
    graph: KernelGraph,
    plan: AllocationPlan,
}

impl GraphExecutor {
    pub fn new(graph: KernelGraph) -> Self {
        let plan = graph.compute_allocation_plan();
        Self { graph, plan }
    }

    pub fn plan(&self) -> &AllocationPlan {
        &self.plan
    }

    pub fn graph(&self) -> &KernelGraph {
        &self.graph
    }

    /// Print the execution graph. For testing without a GPU.
    pub fn execute_mock(&self) {
        println!(
            "GraphExecutor: {} nodes, {} buffers, {} bytes total",
            self.graph.nodes.len(),
            self.graph.buffers.len(),
            self.plan.total_bytes,
        );
        for (i, node) in self.graph.nodes.iter().enumerate() {
            match node {
                GraphNode::Triton(call) => {
                    let grid = call.config.grid;
                    let block = call.config.block;
                    println!(
                        "  [{}] [Triton] {} grid=({},{},{}) block=({},{},{}) smem={} args={} reads={:?} writes={:?}",
                        i, call.kernel_name,
                        grid.0, grid.1, grid.2,
                        block.0, block.1, block.2,
                        call.config.smem_bytes,
                        call.args.len(),
                        call.reads.iter().map(|h| h.0).collect::<Vec<_>>(),
                        call.writes.iter().map(|h| h.0).collect::<Vec<_>>(),
                    );
                }
                GraphNode::Cublas(call) => {
                    println!(
                        "  [{}] [cuBLAS] {} M={} N={} K={} {:?}->{:?} reads={:?} writes={:?}",
                        i, call.gemm.name,
                        call.gemm.m, call.gemm.n, call.gemm.k,
                        call.gemm.input_dtype, call.gemm.output_dtype,
                        call.reads.iter().map(|h| h.0).collect::<Vec<_>>(),
                        call.writes.iter().map(|h| h.0).collect::<Vec<_>>(),
                    );
                }
            }
        }
        for buf in &self.graph.buffers {
            let offset = self.plan.offsets.get(&buf.handle).copied().unwrap_or(0);
            println!(
                "  buf[{}]: {} bytes @ offset {} (live [{}, {}])",
                buf.handle.0, buf.size_bytes, offset, buf.first_write, buf.last_read,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(gx: u32, bx: u32) -> LaunchConfig {
        LaunchConfig {
            grid: (gx, 1, 1),
            block: (bx, 1, 1),
            smem_bytes: 0,
        }
    }

    #[test]
    fn single_kernel() {
        let mut gb = GraphBuilder::new();
        let a = gb.allocate_buffer(1024);
        let b = gb.allocate_buffer(1024);
        gb.mark_input(a);
        gb.mark_output(b);
        gb.launch_kernel(
            "copy",
            cfg(1, 256),
            vec![KernelArg::Buffer(a), KernelArg::Buffer(b)],
            vec![a],
            vec![b],
        );
        let graph = gb.build();
        let plan = graph.compute_allocation_plan();
        // Two buffers both live at kernel 0 => can't overlap.
        assert!(plan.total_bytes >= 2 * align_up(1024));
        assert_ne!(plan.offsets[&a], plan.offsets[&b]);
    }

    #[test]
    fn non_overlapping_reuse() {
        // buf0 written at k0, read at k0
        // buf1 written at k1, read at k1
        // They don't overlap => can share the same offset.
        let mut gb = GraphBuilder::new();
        let buf0 = gb.allocate_buffer(4096);
        let buf1 = gb.allocate_buffer(4096);

        gb.launch_kernel(
            "k0",
            cfg(1, 128),
            vec![KernelArg::Buffer(buf0)],
            vec![],
            vec![buf0],
        );
        gb.launch_kernel(
            "k1",
            cfg(1, 128),
            vec![KernelArg::Buffer(buf1)],
            vec![buf0], // read buf0 here extends its life
            vec![buf1],
        );
        // buf0: first_write=0, last_read=1 (read in k1)
        // buf1: first_write=1, last_read=1
        // They overlap at kernel 1 => must NOT share.
        let graph = gb.build();
        let plan = graph.compute_allocation_plan();
        assert!(plan.total_bytes >= 2 * align_up(4096));
    }

    #[test]
    fn truly_disjoint_reuse() {
        // buf0: written k0, read k0 only
        // buf1: written k1, read k1 only
        // Disjoint lifetimes => CAN share memory.
        let mut gb = GraphBuilder::new();
        let buf0 = gb.allocate_buffer(4096);
        let buf1 = gb.allocate_buffer(4096);

        gb.launch_kernel(
            "k0",
            cfg(1, 128),
            vec![KernelArg::Buffer(buf0)],
            vec![],
            vec![buf0],
        );
        gb.launch_kernel(
            "k1",
            cfg(1, 128),
            vec![KernelArg::Buffer(buf1)],
            vec![],
            vec![buf1],
        );
        let graph = gb.build();
        let plan = graph.compute_allocation_plan();
        // Disjoint: buf0 live [0,0], buf1 live [1,1]. Can overlap.
        assert_eq!(plan.total_bytes, align_up(4096));
        assert_eq!(plan.offsets[&buf0], plan.offsets[&buf1]);
    }

    #[test]
    fn alignment() {
        let mut gb = GraphBuilder::new();
        let buf0 = gb.allocate_buffer(100); // not aligned
        let buf1 = gb.allocate_buffer(200);

        // Both live at k0 => forced apart
        gb.launch_kernel(
            "k0",
            cfg(1, 64),
            vec![KernelArg::Buffer(buf0), KernelArg::Buffer(buf1)],
            vec![buf0],
            vec![buf1],
        );
        let graph = gb.build();
        let plan = graph.compute_allocation_plan();
        for &off in plan.offsets.values() {
            assert_eq!(off % ALIGN, 0, "offset {} not {}-byte aligned", off, ALIGN);
        }
    }

    #[test]
    fn mock_execution() {
        let mut gb = GraphBuilder::new();
        let a = gb.allocate_buffer(512);
        gb.launch_kernel(
            "relu",
            cfg(4, 256),
            vec![KernelArg::Buffer(a), KernelArg::I32(42)],
            vec![a],
            vec![a],
        );
        let graph = gb.build();
        let exec = GraphExecutor::new(graph);
        exec.execute_mock(); // just ensure it doesn't panic
    }

    #[test]
    fn mixed_triton_cublas_graph() {
        use crate::cublas_gemm::GemmOp;

        let mut gb = GraphBuilder::new();
        // Simulate one decode layer: norm -> gemm -> norm -> gemm
        let hidden = gb.allocate_buffer(4096 * 2); // f16 hidden state
        let normed = gb.allocate_buffer(4096 * 2);
        let qkv = gb.allocate_buffer(12288 * 2);
        let attn_out = gb.allocate_buffer(4096 * 2);

        gb.mark_input(hidden);
        gb.mark_output(attn_out);

        // [Triton] fused_residual_rmsnorm
        gb.launch_kernel(
            "fused_residual_rmsnorm",
            cfg(1, 256),
            vec![KernelArg::Buffer(hidden), KernelArg::Buffer(normed)],
            vec![hidden],
            vec![normed],
        );

        // [cuBLAS] QKV projection
        gb.launch_cublas(
            GemmOp::hgemm("qkv_proj", 1, 12288, 4096),
            KernelArg::Buffer(normed),
            KernelArg::ExternalPtr(0), // weight ptr (external)
            KernelArg::Buffer(qkv),
            vec![normed],
            vec![qkv],
        );

        // [Triton] RoPE
        gb.launch_kernel(
            "rope",
            cfg(1, 256),
            vec![KernelArg::Buffer(qkv)],
            vec![qkv],
            vec![qkv],
        );

        // [cuBLAS] O-proj
        gb.launch_cublas(
            GemmOp::hgemm("o_proj", 1, 4096, 4096),
            KernelArg::Buffer(qkv),
            KernelArg::ExternalPtr(0),
            KernelArg::Buffer(attn_out),
            vec![qkv],
            vec![attn_out],
        );

        let graph = gb.build();
        assert_eq!(graph.nodes.len(), 4);

        // Verify node types
        assert!(matches!(&graph.nodes[0], GraphNode::Triton(_)));
        assert!(matches!(&graph.nodes[1], GraphNode::Cublas(_)));
        assert!(matches!(&graph.nodes[2], GraphNode::Triton(_)));
        assert!(matches!(&graph.nodes[3], GraphNode::Cublas(_)));

        // normed buffer: live [0,1], qkv: live [1,3], so normed and attn_out can share
        let plan = graph.compute_allocation_plan();
        assert!(plan.total_bytes > 0);

        let exec = GraphExecutor::new(graph);
        exec.execute_mock();
    }
}

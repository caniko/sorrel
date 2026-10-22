//! sorrel-render: concrete vertex types and statically-compiled mapping
//! functions. These are the "tight, inlined loops" that turn backend data
//! into the flat arrays consumed by the GPU pipelines.

pub mod buffers;
pub mod pipelines;
pub mod vertex;

pub use buffers::{
    apply_preproc_cpu, build_scatter_vertices, build_trace_vertices, build_trace_vertices_cfg,
    build_trace_vertices_cfg_shanked, build_trace_vertices_gpu, build_trace_vertices_gpu_shanked,
    GpuTracePreproc, TracePreproc,
};
pub use pipelines::{ScatterPipeline, TracePipeline};
pub use vertex::{ScatterVertex, TraceVertex};

//! sorrel-render: concrete vertex types and statically-compiled mapping
//! functions. These are the "tight, inlined loops" that turn backend data
//! into the flat arrays consumed by the GPU pipelines.

pub mod buffers;
pub mod pipelines;
pub mod vertex;

pub use buffers::{build_scatter_vertices, build_trace_vertices};
pub use pipelines::{ScatterPipeline, TracePipeline};
pub use vertex::{ScatterVertex, TraceVertex};

//! sorrel-compute: generic, vectorisable kernels.
//!
//! Functions are parameterised over numeric types so the compiler can emit
//! SIMD-friendly inner loops per call site.

pub mod binning;
pub mod lttb;

pub use binning::{histogram_u64, isi_histogram};
pub use lttb::lttb_downsample;

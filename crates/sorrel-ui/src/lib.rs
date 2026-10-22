//! sorrel-ui: egui widgets bound to a generic `Session<P>`.

// wgpu's `Device: Send` proof nests deeper than the default 128 on current
// nightlies (future-compat lint `recursion_depth_exceeding_limit`).
#![recursion_limit = "256"]

pub mod app;
pub mod intent;
pub mod selection;
pub mod views;
pub mod wgpu_raster;
pub mod wgpu_trace;
pub mod widgets;

pub use app::SorrelApp;
pub use intent::Intent;
pub use selection::SelectionSet;
pub use views::{FeatureViewState, RasterCache, WaveformCache};
pub use wgpu_raster::RasterPipeline;
pub use wgpu_trace::{TraceCallback, TracePipeline};

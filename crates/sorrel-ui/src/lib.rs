//! sorrel-ui: egui widgets bound to a generic `Session<P>`.

pub mod app;
pub mod intent;
pub mod selection;
pub mod views;
pub mod widgets;
pub mod wgpu_raster;
pub mod wgpu_trace;

pub use app::SorrelApp;
pub use intent::Intent;
pub use selection::SelectionSet;
pub use views::{FeatureViewState, RasterCache, WaveformCache};
pub use wgpu_raster::RasterPipeline;
pub use wgpu_trace::{TraceCallback, TracePipeline};

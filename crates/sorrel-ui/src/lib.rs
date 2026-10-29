//! sorrel-ui: egui widgets bound to a generic `Session<P>`.

// wgpu's `Device: Send` proof nests deeper than the default 128 on current
// nightlies (future-compat lint `recursion_depth_exceeding_limit`).
#![recursion_limit = "256"]

pub mod amplitude_view;
pub mod app;
pub mod cluster_scatter;
pub mod cluster_view;
pub mod correlogram_view;
pub mod feature_view;
pub mod history_view;
pub mod intent;
pub mod isi_view;
pub mod jobs;
pub mod lasso;
pub mod layout;
pub mod panzoom;
pub mod prefs;
pub mod selection;
pub mod shell;
pub mod similarity_view;
pub mod snippets;
pub mod spike_attr_view;
pub mod template_feature_view;
pub mod template_grid;
pub mod theme;
pub mod trace_image;
pub mod views;
pub mod waveform_view;
pub mod wgpu_raster;
pub mod wgpu_scatter;
pub mod wgpu_trace;
pub mod widgets;

pub use app::SorrelApp;
pub use feature_view::FeatureViewState;
pub use intent::Intent;
pub use jobs::{Job, JobCtx, Memo};
pub use layout::Pane;
pub use prefs::UiPrefs;
pub use selection::SelectionSet;
pub use shell::{CloseState, OpenRequest, Opened, Opener, Shell, Tab};
pub use theme::{cluster_colour, selection_colour, Theme};
pub use views::{RasterCache, WaveformCache};
pub use wgpu_raster::RasterPipeline;
pub use wgpu_scatter::ScatterPipeline;
pub use wgpu_trace::{TraceCallback, TracePipeline};

//! Concrete pipeline structs. V1 holds CPU-side buffers; the wgpu device,
//! shader, and bind-group plumbing belongs in the UI integration once egui's
//! `paint_callback` hook is in use.

use crate::vertex::{ScatterVertex, TraceVertex};

pub struct TracePipeline {
    pub vertices: Vec<TraceVertex>,
    pub n_channels: u32,
}

impl TracePipeline {
    pub const fn new(n_channels: u32) -> Self {
        Self {
            vertices: Vec::new(),
            n_channels,
        }
    }
    pub fn upload(&mut self, vertices: Vec<TraceVertex>) {
        self.vertices = vertices;
    }
}

pub struct ScatterPipeline {
    pub vertices: Vec<ScatterVertex>,
}

impl ScatterPipeline {
    pub const fn new() -> Self {
        Self {
            vertices: Vec::new(),
        }
    }
    pub fn upload(&mut self, vertices: Vec<ScatterVertex>) {
        self.vertices = vertices;
    }
}

impl Default for ScatterPipeline {
    fn default() -> Self {
        Self::new()
    }
}

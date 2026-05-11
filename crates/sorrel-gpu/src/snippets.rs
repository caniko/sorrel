//! GPU column-mean over a `(n_spikes, snippet_len)` matrix — the spike
//! template (mean waveform) used in waveform views and split previews.
//!
//! Mirrors [`sorrel_compute::mean_snippet`] but flattens the input to one
//! contiguous row-major buffer instead of a `Vec<Vec<f32>>`, which is also
//! the layout the rest of the pipeline (snippet extraction → GPU upload)
//! wants anyway.

use crate::{block_on_map, GpuContext};
use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;

const SHADER: &str = include_str!("shaders/mean_snippet.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct Params {
    n_spikes: u32,
    snippet_len: u32,
}

const PARAMS_SIZE: u64 = std::mem::size_of::<Params>() as u64;

pub struct GpuMeanSnippet {
    ctx: GpuContext,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
}

impl GpuMeanSnippet {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sorrel-gpu.snippets.shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sorrel-gpu.snippets.bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(PARAMS_SIZE),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sorrel-gpu.snippets.layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sorrel-gpu.snippets.pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sorrel-gpu.snippets.params"),
            size: PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            ctx: ctx.clone(),
            pipeline,
            bgl,
            params_buf,
        }
    }

    /// Compute the column-mean of `snippets` (row-major,
    /// `n_spikes × snippet_len`). Returns the mean as a `Vec<f32>` of length
    /// `snippet_len`. Empty input returns an empty `Vec`.
    pub fn run(&self, snippets: &[f32], snippet_len: u32) -> Result<Vec<f32>, GpuMeanSnippetError> {
        if snippet_len == 0 || snippets.is_empty() {
            return Ok(Vec::new());
        }
        if (snippets.len() as u64).checked_rem(snippet_len as u64) != Some(0) {
            return Err(GpuMeanSnippetError::ShapeMismatch);
        }
        let n_spikes = (snippets.len() / snippet_len as usize) as u32;
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let in_bytes = std::mem::size_of_val(snippets) as u64;
        let out_bytes = (snippet_len as usize * std::mem::size_of::<f32>()) as u64;

        let in_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sorrel-gpu.snippets.in"),
            size: in_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&in_buf, 0, bytemuck::cast_slice(snippets));

        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sorrel-gpu.snippets.out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = Params {
            n_spikes,
            snippet_len,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sorrel-gpu.snippets.bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sorrel-gpu.snippets.readback"),
            size: out_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let workgroups = snippet_len.div_ceil(64);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sorrel-gpu.snippets.enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sorrel-gpu.snippets.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, out_bytes);
        queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        block_on_map(device, slice).map_err(|e| GpuMeanSnippetError::Readback(e.to_string()))?;
        let out: Vec<f32> = {
            let view = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&view).to_vec()
        };
        readback.unmap();
        Ok(out)
    }
}

#[derive(Debug)]
pub enum GpuMeanSnippetError {
    ShapeMismatch,
    Readback(String),
}

impl std::fmt::Display for GpuMeanSnippetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch => write!(f, "snippets length not a multiple of snippet_len"),
            Self::Readback(e) => write!(f, "readback failed: {e}"),
        }
    }
}
impl std::error::Error for GpuMeanSnippetError {}

#[cfg(test)]
mod tests {
    use super::*;
    use sorrel_compute::mean_snippet;

    fn ctx() -> Option<crate::test_support::TestGpuContext> {
        crate::test_support::ctx()
    }

    #[test]
    fn matches_cpu_mean_snippet() {
        let Some(ctx) = ctx() else { return };
        let gpu = GpuMeanSnippet::new(&ctx);
        let snippet_len = 80u32;
        let n_spikes = 257usize;
        let mut flat = Vec::with_capacity(n_spikes * snippet_len as usize);
        let mut nested: Vec<Vec<f32>> = Vec::with_capacity(n_spikes);
        for s in 0..n_spikes {
            let row: Vec<f32> = (0..snippet_len)
                .map(|i| ((s as i32 * 13 + i as i32 * 7) % 91 - 45) as f32)
                .collect();
            flat.extend_from_slice(&row);
            nested.push(row);
        }
        let g = gpu.run(&flat, snippet_len).unwrap();
        let c = mean_snippet(&nested);
        assert_eq!(g.len(), c.len());
        for (i, (a, b)) in g.iter().zip(c.iter()).enumerate() {
            assert!((a - b).abs() < 1e-3, "i={i}: gpu={a} cpu={b}");
        }
    }

    #[test]
    fn empty_returns_empty() {
        let Some(ctx) = ctx() else { return };
        let gpu = GpuMeanSnippet::new(&ctx);
        let v = gpu.run(&[], 80).unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn single_snippet_equals_input() {
        let Some(ctx) = ctx() else { return };
        let gpu = GpuMeanSnippet::new(&ctx);
        let s: Vec<f32> = vec![1.0, -2.0, 3.5, -4.25, 5.125];
        let m = gpu.run(&s, s.len() as u32).unwrap();
        for (a, b) in m.iter().zip(s.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

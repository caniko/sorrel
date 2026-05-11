//! GPU CMR (common median referencing).
//!
//! Mirrors [`sorrel_compute::subtract_channel_median`] but launches one
//! workgroup per time step and bitonic-sorts the row in workgroup-local
//! memory. Practical channel-count cap is 512 (Neuropixels = 384) — beyond
//! that the caller should fall back to CPU.

use crate::{block_on_map, GpuContext};
use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;

const SHADER: &str = include_str!("shaders/cmr_median.wgsl");
pub const MAX_CHANNELS: u32 = 512;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct Params {
    n_channels: u32,
    n_time_steps: u32,
    _pad0: u32,
    _pad1: u32,
}

const PARAMS_SIZE: u64 = std::mem::size_of::<Params>() as u64;

pub struct GpuCmr {
    ctx: GpuContext,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
}

impl GpuCmr {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sorrel-gpu.cmr.shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sorrel-gpu.cmr.bgl"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sorrel-gpu.cmr.layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sorrel-gpu.cmr.pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sorrel-gpu.cmr.params"),
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

    /// Run CMR in-place on `samples` (time-major, `n_channels` per row).
    /// Returns `Err` if `n_channels > MAX_CHANNELS` so the caller can route
    /// to the CPU path.
    pub fn run(&self, samples: &mut [f32], n_channels: u32) -> Result<(), GpuCmrError> {
        if n_channels == 0 || samples.is_empty() {
            return Ok(());
        }
        if n_channels > MAX_CHANNELS {
            return Err(GpuCmrError::TooManyChannels(n_channels));
        }
        if (samples.len() as u64).checked_rem(n_channels as u64) != Some(0) {
            return Err(GpuCmrError::ShapeMismatch);
        }
        let n_time = (samples.len() / n_channels as usize) as u32;
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let byte_len = std::mem::size_of_val(samples) as u64;
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sorrel-gpu.cmr.data"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&storage, 0, bytemuck::cast_slice(samples));

        let params = Params {
            n_channels,
            n_time_steps: n_time,
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sorrel-gpu.cmr.bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: storage.as_entire_binding(),
                },
            ],
        });

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sorrel-gpu.cmr.readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sorrel-gpu.cmr.enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sorrel-gpu.cmr.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_time, 1, 1);
        }
        enc.copy_buffer_to_buffer(&storage, 0, &readback, 0, byte_len);
        queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        block_on_map(device, slice).map_err(|e| GpuCmrError::Readback(e.to_string()))?;
        {
            let view = slice.get_mapped_range();
            samples.copy_from_slice(bytemuck::cast_slice(&view));
        }
        readback.unmap();
        Ok(())
    }
}

#[derive(Debug)]
pub enum GpuCmrError {
    TooManyChannels(u32),
    ShapeMismatch,
    Readback(String),
}

impl std::fmt::Display for GpuCmrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyChannels(n) => {
                write!(f, "GPU CMR limit is {MAX_CHANNELS} channels, got {n}")
            }
            Self::ShapeMismatch => write!(f, "samples length not a multiple of n_channels"),
            Self::Readback(e) => write!(f, "readback failed: {e}"),
        }
    }
}
impl std::error::Error for GpuCmrError {}

#[cfg(test)]
mod tests {
    use super::*;
    use sorrel_compute::subtract_channel_median;

    fn ctx() -> Option<crate::test_support::TestGpuContext> {
        crate::test_support::ctx()
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol, "i={i}: {x} vs {y}");
        }
    }

    #[test]
    fn matches_cpu_for_odd_channel_count() {
        let Some(ctx) = ctx() else { return };
        let cmr = GpuCmr::new(&ctx);
        let mut gpu_in: Vec<f32> = (0..(7 * 32))
            .map(|i| ((i * 31) % 197) as f32 - 100.0)
            .collect();
        let mut cpu_in = gpu_in.clone();
        cmr.run(&mut gpu_in, 7).unwrap();
        subtract_channel_median(&mut cpu_in, 7);
        approx_eq(&gpu_in, &cpu_in, 1e-4);
    }

    #[test]
    fn matches_cpu_for_even_channel_count() {
        let Some(ctx) = ctx() else { return };
        let cmr = GpuCmr::new(&ctx);
        let mut gpu_in: Vec<f32> = (0..(8 * 64))
            .map(|i| ((i * 11) % 233) as f32 - 100.0)
            .collect();
        let mut cpu_in = gpu_in.clone();
        cmr.run(&mut gpu_in, 8).unwrap();
        subtract_channel_median(&mut cpu_in, 8);
        approx_eq(&gpu_in, &cpu_in, 1e-4);
    }

    #[test]
    fn matches_cpu_for_neuropixels_channel_count() {
        let Some(ctx) = ctx() else { return };
        let cmr = GpuCmr::new(&ctx);
        let nc = 384u32;
        let nt = 64usize;
        let mut gpu_in: Vec<f32> = (0..(nc as usize * nt))
            .map(|i| ((i * 37) % 8191) as f32 - 4000.0)
            .collect();
        let mut cpu_in = gpu_in.clone();
        cmr.run(&mut gpu_in, nc).unwrap();
        subtract_channel_median(&mut cpu_in, nc as usize);
        approx_eq(&gpu_in, &cpu_in, 1e-3);
    }

    #[test]
    fn rejects_too_many_channels() {
        let Some(ctx) = ctx() else { return };
        let cmr = GpuCmr::new(&ctx);
        let mut buf = vec![0.0f32; 1024];
        let r = cmr.run(&mut buf, 1024);
        assert!(matches!(r, Err(GpuCmrError::TooManyChannels(1024))));
    }

    #[test]
    fn empty_input_is_a_noop() {
        let Some(ctx) = ctx() else { return };
        let cmr = GpuCmr::new(&ctx);
        let mut buf: Vec<f32> = Vec::new();
        cmr.run(&mut buf, 4).unwrap();
        assert!(buf.is_empty());
    }
}

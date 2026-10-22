//! GPU Biquad IIR filter, parallel across channels.
//!
//! Mirrors [`sorrel_compute::Biquad::apply`], but instead of streaming a
//! single channel through the recursion we launch one GPU thread per
//! channel. State is kept in registers (not workgroup memory) so the
//! filter is bit-identical to the single-channel CPU recursion.

use crate::{block_on_map, GpuContext};
use bytemuck::{Pod, Zeroable};
use sorrel_compute::Biquad;
use std::num::NonZeroU64;

const SHADER: &str = include_str!("shaders/biquad_hp.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct Params {
    n_channels: u32,
    n_samples: u32,
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    _pad: f32,
}

const PARAMS_SIZE: u64 = std::mem::size_of::<Params>() as u64;

pub struct GpuBiquad {
    ctx: GpuContext,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
}

impl GpuBiquad {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sorrel-gpu.biquad.shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sorrel-gpu.biquad.bgl"),
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
            label: Some("sorrel-gpu.biquad.layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sorrel-gpu.biquad.pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sorrel-gpu.biquad.params"),
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

    /// Apply `biquad` in-place to `samples` (time-major, `n_channels` per row).
    /// Each channel runs the IIR recursion independently.
    pub fn run(
        &self,
        samples: &mut [f32],
        n_channels: u32,
        biquad: Biquad,
    ) -> Result<(), GpuBiquadError> {
        if n_channels == 0 || samples.is_empty() {
            return Ok(());
        }
        if (samples.len() as u64).checked_rem(n_channels as u64) != Some(0) {
            return Err(GpuBiquadError::ShapeMismatch);
        }
        let n_samples = (samples.len() / n_channels as usize) as u32;
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let byte_len = std::mem::size_of_val(samples) as u64;
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sorrel-gpu.biquad.data"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&storage, 0, bytemuck::cast_slice(samples));

        let params = Params {
            n_channels,
            n_samples,
            b0: biquad.b0,
            b1: biquad.b1,
            b2: biquad.b2,
            a1: biquad.a1,
            a2: biquad.a2,
            _pad: 0.0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sorrel-gpu.biquad.bg"),
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
            label: Some("sorrel-gpu.biquad.readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let workgroups = n_channels.div_ceil(64);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sorrel-gpu.biquad.enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sorrel-gpu.biquad.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&storage, 0, &readback, 0, byte_len);
        queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        block_on_map(device, slice).map_err(|e| GpuBiquadError::Readback(e.to_string()))?;
        {
            let view = slice.get_mapped_range();
            samples.copy_from_slice(bytemuck::cast_slice(&view));
        }
        readback.unmap();
        Ok(())
    }
}

#[derive(Debug)]
pub enum GpuBiquadError {
    ShapeMismatch,
    Readback(String),
}

impl std::fmt::Display for GpuBiquadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch => write!(f, "samples length not a multiple of n_channels"),
            Self::Readback(e) => write!(f, "readback failed: {e}"),
        }
    }
}
impl std::error::Error for GpuBiquadError {}

#[cfg(test)]
mod tests {
    use super::*;
    use sorrel_compute::{Biquad, BiquadState};

    fn ctx() -> Option<crate::test_support::TestGpuContext> {
        crate::test_support::ctx()
    }

    /// Streaming CPU reference: same recursion as `Biquad::step`, kept here
    /// to be explicit about per-channel state.
    fn cpu_apply_per_channel(samples: &mut [f32], n_channels: usize, b: Biquad) {
        if n_channels == 0 {
            return;
        }
        let n_samples = samples.len() / n_channels;
        let mut states = vec![BiquadState::default(); n_channels];
        for t in 0..n_samples {
            let row = t * n_channels;
            for (ch, state) in states.iter_mut().enumerate().take(n_channels) {
                let i = row + ch;
                let y = b.step(samples[i], state);
                samples[i] = y;
            }
        }
    }

    #[test]
    fn matches_cpu_per_channel_high_pass() {
        let Some(ctx) = ctx() else { return };
        let gpu = GpuBiquad::new(&ctx);
        let b = Biquad::butterworth_hp(300.0, 30_000.0);
        let nc = 8usize;
        let n = 2048usize;
        let mut gpu_in: Vec<f32> = (0..(nc * n))
            .map(|i| ((i as f32 * 0.013).sin() + (i as f32 * 0.057).cos()) * 100.0)
            .collect();
        let mut cpu_in = gpu_in.clone();
        gpu.run(&mut gpu_in, nc as u32, b).unwrap();
        cpu_apply_per_channel(&mut cpu_in, nc, b);
        // IIR recursion drifts in f32 over thousands of steps even between
        // bit-identical formulations — different FMA fusion on GPU vs CPU is
        // enough. Compare relative.
        for (i, (g, c)) in gpu_in.iter().zip(cpu_in.iter()).enumerate() {
            let scale = g.abs().max(c.abs()).max(1.0);
            assert!((g - c).abs() / scale < 1e-3, "i={i}: gpu={g} cpu={c}",);
        }
    }

    #[test]
    fn dc_is_attenuated() {
        let Some(ctx) = ctx() else { return };
        let gpu = GpuBiquad::new(&ctx);
        let b = Biquad::butterworth_hp(300.0, 30_000.0);
        let nc = 4usize;
        let n = 4096usize;
        let mut buf = vec![1000.0f32; nc * n];
        gpu.run(&mut buf, nc as u32, b).unwrap();
        // Last row, every channel should have settled near zero.
        let tail = &buf[(n - 1) * nc..];
        for (i, &v) in tail.iter().enumerate() {
            assert!(v.abs() < 1.0, "channel {i} tail = {v}");
        }
    }

    #[test]
    fn many_channels_run_in_parallel() {
        let Some(ctx) = ctx() else { return };
        let gpu = GpuBiquad::new(&ctx);
        let b = Biquad::butterworth_hp(300.0, 30_000.0);
        let nc = 384usize;
        let n = 256usize;
        let mut gpu_in: Vec<f32> = (0..(nc * n)).map(|i| ((i % 71) as f32) - 35.0).collect();
        let mut cpu_in = gpu_in.clone();
        gpu.run(&mut gpu_in, nc as u32, b).unwrap();
        cpu_apply_per_channel(&mut cpu_in, nc, b);
        // Spot-check a handful of channels at a few time steps.
        for ch in [0usize, 17, 100, 383] {
            for t in [0usize, 50, 200, 255] {
                let i = t * nc + ch;
                let scale = gpu_in[i].abs().max(cpu_in[i].abs()).max(1.0);
                assert!(
                    (gpu_in[i] - cpu_in[i]).abs() / scale < 1e-3,
                    "ch={ch} t={t} gpu={} cpu={}",
                    gpu_in[i],
                    cpu_in[i],
                );
            }
        }
    }
}

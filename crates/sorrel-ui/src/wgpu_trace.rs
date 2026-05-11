//! GPU-side trace renderer.
//!
//! The CPU still does LTTB downsampling each frame ([`sorrel_render::build_trace_vertices`]),
//! producing a `Vec<TraceVertex>` arranged as `points × n_channels`, channel-major.
//! The GPU then turns that into one line strip per channel via N draw calls
//! into a single shared vertex buffer. Per-channel y-offset and amplitude
//! scaling happen in the vertex shader so the CPU never iterates pixels.
//!
//! This is the path that lets the trace view scale to Neuropixels-grade
//! `n_channels × points_per_channel` without choking the egui painter.

use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::{self, RenderState};
use sorrel_render::TraceVertex;
use std::num::NonZeroU64;

/// Uniforms uploaded once per frame. Layout matches the WGSL `Uniforms`
/// struct verbatim — keep them in sync.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct TraceUniforms {
    t0: f32,
    t_span_inv: f32,
    n_channels_inv: f32,
    amp_scale: f32,
    color: [f32; 4],
}

const VERTEX_SIZE: u64 = std::mem::size_of::<TraceVertex>() as u64;
const UNIFORMS_SIZE: u64 = std::mem::size_of::<TraceUniforms>() as u64;
const INITIAL_VERTEX_CAPACITY: u64 = 1024 * 64; // 64k vertices ≈ 768 KB

/// GPU resources for the trace pipeline. Lives in the wgpu renderer's
/// `CallbackResources` map and is shared across every paint callback.
pub struct TracePipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    /// Capacity of `vertex_buffer` in bytes; grown on demand.
    vertex_capacity_bytes: u64,
}

impl TracePipeline {
    /// Install the pipeline into the egui-wgpu renderer's callback resources.
    /// Idempotent: a second call replaces the previous pipeline (safe but
    /// usually unnecessary).
    pub fn install(rs: &RenderState) {
        let pipeline = Self::build(&rs.device, rs.target_format);
        let mut renderer = rs.renderer.write();
        renderer.callback_resources.insert(pipeline);
    }

    fn build(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sorrel.trace.shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_WGSL.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sorrel.trace.uniforms"),
            size: UNIFORMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sorrel.trace.bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(UNIFORMS_SIZE),
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sorrel.trace.bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sorrel.trace.layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sorrel.trace.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: VERTEX_SIZE,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            shader_location: 0,
                            offset: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            shader_location: 1,
                            offset: 8,
                            format: wgpu::VertexFormat::Uint32,
                        },
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sorrel.trace.verts"),
            size: INITIAL_VERTEX_CAPACITY * VERTEX_SIZE,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group,
            vertex_buffer,
            uniform_buffer,
            vertex_capacity_bytes: INITIAL_VERTEX_CAPACITY * VERTEX_SIZE,
        }
    }

    /// Upload vertex + uniform data for the next frame. Grows the vertex
    /// buffer if the new data doesn't fit.
    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[TraceVertex],
        uniforms: TraceUniforms,
    ) {
        let bytes = bytemuck::cast_slice(vertices);
        let needed = bytes.len() as u64;
        if needed > self.vertex_capacity_bytes {
            // Grow geometrically so frequent upsizes don't churn the GPU.
            let mut new_cap = self.vertex_capacity_bytes.max(VERTEX_SIZE);
            while new_cap < needed {
                new_cap *= 2;
            }
            self.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sorrel.trace.verts"),
                size: new_cap,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertex_capacity_bytes = new_cap;
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.vertex_buffer, 0, bytes);
        }
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'static>, n_channels: u32, points_per_channel: u32) {
        if n_channels == 0 || points_per_channel < 2 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        // One LineStrip draw per channel — wgpu has no primitive restart in 0.20.
        for ch in 0..n_channels {
            let start = ch * points_per_channel;
            let end = start + points_per_channel;
            pass.draw(start..end, 0..1);
        }
    }
}

/// Per-frame paint callback. Owns the data the GPU needs for *this* frame
/// only; `TracePipeline` (in `CallbackResources`) owns the persistent
/// shader/buffer state.
pub struct TraceCallback {
    pub vertices: Vec<TraceVertex>,
    pub n_channels: u32,
    pub points_per_channel: u32,
    pub t0: f32,
    pub t_span: f32,
    pub amp_scale: f32,
    pub color: [f32; 4],
}

impl egui_wgpu::CallbackTrait for TraceCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(pipeline) = callback_resources.get_mut::<TracePipeline>() else {
            log::warn!("TracePipeline missing from callback resources — skipping prepare");
            return Vec::new();
        };
        let uniforms = TraceUniforms {
            t0: self.t0,
            t_span_inv: if self.t_span > 0.0 {
                1.0 / self.t_span
            } else {
                0.0
            },
            n_channels_inv: if self.n_channels > 0 {
                1.0 / self.n_channels as f32
            } else {
                0.0
            },
            amp_scale: self.amp_scale,
            color: self.color,
        };
        pipeline.upload(device, queue, &self.vertices, uniforms);
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(pipeline) = callback_resources.get::<TracePipeline>() else {
            return;
        };
        pipeline.draw(render_pass, self.n_channels, self.points_per_channel);
    }
}

const SHADER_WGSL: &str = r#"
struct VertexIn {
    @location(0) pos: vec2<f32>,
    @location(1) channel: u32,
};

struct Uniforms {
    t0: f32,
    t_span_inv: f32,
    n_channels_inv: f32,
    amp_scale: f32,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
};

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    let x_norm = (input.pos.x - u.t0) * u.t_span_inv;
    let x_clip = x_norm * 2.0 - 1.0;
    // y_center for channel ch in clip space (top = +1, bottom = -1):
    //   row 0 should sit near the top, last row near the bottom.
    let row = f32(input.channel) + 0.5;
    let y_center = 1.0 - row * 2.0 * u.n_channels_inv;
    let y_clip = y_center - input.pos.y * u.amp_scale;
    var out: VertexOut;
    out.clip_pos = vec4<f32>(x_clip, y_clip, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return u.color;
}
"#;

//! GPU-side raster renderer.
//!
//! One instanced point per spike, packed into a single vertex buffer:
//! `(time_seconds, cluster_row, rgba_u32)`. The vertex shader maps
//! `(time, cluster_row)` into clip space using a uniform viewport, and
//! the fragment shader outputs the unpacked colour. PointList topology
//! draws 1-pixel dots — same primitive phy uses internally.
//!
//! At Neuropixels-grade spike counts (~10–50 M total spikes) the point
//! buffer dominates memory: 12 B/vertex × 50 M ≈ 600 MB. We expect callers
//! to cache the vertex buffer and only rebuild on curation changes.

use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::{self, RenderState};
use std::num::NonZeroU64;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct RasterVertex {
    /// `pos.x` is time in seconds, `pos.y` is cluster row index.
    pub pos: [f32; 2],
    /// Packed `0xRRGGBBAA`.
    pub color: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct RasterUniforms {
    t0: f32,
    t_span_inv: f32,
    n_rows_inv: f32,
    /// Padding so the struct is a multiple of 16 bytes (WGSL std140).
    _pad: f32,
}

const VERTEX_SIZE: u64 = std::mem::size_of::<RasterVertex>() as u64;
const UNIFORMS_SIZE: u64 = std::mem::size_of::<RasterUniforms>() as u64;
const INITIAL_VERTEX_CAPACITY: u64 = 1024 * 64;

pub struct RasterPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    vertex_capacity_bytes: u64,
}

impl RasterPipeline {
    pub fn install(rs: &RenderState) {
        let pipeline = Self::build(&rs.device, rs.target_format);
        let mut renderer = rs.renderer.write();
        renderer.callback_resources.insert(pipeline);
    }

    fn build(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sorrel.raster.shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_WGSL.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sorrel.raster.uniforms"),
            size: UNIFORMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sorrel.raster.bgl"),
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
            label: Some("sorrel.raster.bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sorrel.raster.layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sorrel.raster.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
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
                topology: wgpu::PrimitiveTopology::PointList,
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
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sorrel.raster.verts"),
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

    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[RasterVertex],
        uniforms: RasterUniforms,
    ) {
        let bytes = bytemuck::cast_slice(vertices);
        let needed = bytes.len() as u64;
        if needed > self.vertex_capacity_bytes {
            let mut new_cap = self.vertex_capacity_bytes.max(VERTEX_SIZE);
            while new_cap < needed {
                new_cap *= 2;
            }
            self.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sorrel.raster.verts"),
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

    fn draw<'rp>(&'rp self, pass: &mut wgpu::RenderPass<'rp>, n_vertices: u32) {
        if n_vertices == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..n_vertices, 0..1);
    }
}

/// Per-frame raster paint callback. Uploads vertices on `prepare` and
/// issues a single `draw(0..n_vertices)` on `paint`.
pub struct RasterCallback {
    pub vertices: Vec<RasterVertex>,
    pub n_vertices: u32,
    pub t0: f32,
    pub t_span: f32,
    pub n_rows: u32,
}

impl egui_wgpu::CallbackTrait for RasterCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(pipeline) = callback_resources.get_mut::<RasterPipeline>() else {
            log::warn!("RasterPipeline missing from callback resources — skipping prepare");
            return Vec::new();
        };
        let uniforms = RasterUniforms {
            t0: self.t0,
            t_span_inv: if self.t_span > 0.0 {
                1.0 / self.t_span
            } else {
                0.0
            },
            n_rows_inv: if self.n_rows > 0 {
                1.0 / self.n_rows as f32
            } else {
                0.0
            },
            _pad: 0.0,
        };
        pipeline.upload(device, queue, &self.vertices, uniforms);
        Vec::new()
    }

    fn paint<'a>(
        &'a self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'a>,
        callback_resources: &'a egui_wgpu::CallbackResources,
    ) {
        let Some(pipeline) = callback_resources.get::<RasterPipeline>() else {
            return;
        };
        pipeline.draw(render_pass, self.n_vertices);
    }
}

/// Pack `(r, g, b, a)` bytes into a vertex `u32`. Endian-correct on the
/// shader side because we read it back via `unpack4x8unorm`.
#[inline]
pub fn pack_rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    u32::from_le_bytes([r, g, b, a])
}

const SHADER_WGSL: &str = r#"
struct VertexIn {
    @location(0) pos: vec2<f32>,
    @location(1) color: u32,
};

struct Uniforms {
    t0: f32,
    t_span_inv: f32,
    n_rows_inv: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    let x_norm = (input.pos.x - u.t0) * u.t_span_inv;
    let x_clip = x_norm * 2.0 - 1.0;
    // y_center: row 0 near the top (+1), last row near the bottom (-1).
    let row = input.pos.y + 0.5;
    let y_clip = 1.0 - row * 2.0 * u.n_rows_inv;
    var out: VertexOut;
    out.clip_pos = vec4<f32>(x_clip, y_clip, 0.0, 1.0);
    out.color = unpack4x8unorm(input.color);
    return out;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    return input.color;
}
"#;

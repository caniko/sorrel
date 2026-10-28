//! GPU scatter renderer for the point-cloud views (Features first).
//!
//! Unlike [`crate::wgpu_raster`], several scatters draw in one frame (the
//! FeatureView grid alone has 16 cells), so every callback owns a slot —
//! vertex buffer, uniforms, bind group — keyed by a caller-chosen id.
//! Vertices hold data coordinates; the per-frame uniforms carry the data
//! window (pan/zoom) and the marker size, so neither needs a re-upload.
//! Each point is an instanced quad shaded as a disc, which gives phy's
//! adjustable marker size (a `PointList` is fixed at one pixel).

use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::{self, RenderState};
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct ScatterVertex {
    /// Data coordinates, mapped through [`ScatterCallback::x`] / `y`.
    pub pos: [f32; 2],
    /// Packed with [`crate::wgpu_raster::pack_rgba`].
    pub color: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct ScatterUniforms {
    lo: [f32; 2],
    inv_span: [f32; 2],
    /// Marker radius in clip units per axis.
    radius: [f32; 2],
    _pad: [f32; 2],
}

const VERTEX_SIZE: u64 = std::mem::size_of::<ScatterVertex>() as u64;
const UNIFORMS_SIZE: u64 = std::mem::size_of::<ScatterUniforms>() as u64;

struct Slot {
    vertex_buffer: wgpu::Buffer,
    capacity_bytes: u64,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    generation: Option<u64>,
}

pub struct ScatterPipeline {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    // ponytail: slots are never freed; there is one per scatter cell on
    // screen (a few dozen). Evict by last-used frame if views multiply.
    slots: HashMap<u64, Slot>,
}

impl ScatterPipeline {
    pub fn install(rs: &RenderState) {
        let pipeline = Self::build(&rs.device, rs.target_format);
        rs.renderer.write().callback_resources.insert(pipeline);
    }

    fn build(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sorrel.scatter.shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sorrel.scatter.bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(UNIFORMS_SIZE),
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sorrel.scatter.layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sorrel.scatter.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: VERTEX_SIZE,
                    step_mode: wgpu::VertexStepMode::Instance,
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
            primitive: wgpu::PrimitiveState::default(),
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
        Self {
            pipeline,
            bgl,
            slots: HashMap::new(),
        }
    }

    fn slot(&mut self, device: &wgpu::Device, id: u64) -> &mut Slot {
        let bgl = &self.bgl;
        self.slots.entry(id).or_insert_with(|| {
            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sorrel.scatter.uniforms"),
                size: UNIFORMS_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sorrel.scatter.bg"),
                layout: bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });
            Slot {
                vertex_buffer: vertex_buffer(device, VERTEX_SIZE * 1024),
                capacity_bytes: VERTEX_SIZE * 1024,
                uniform_buffer,
                bind_group,
                generation: None,
            }
        })
    }
}

fn vertex_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sorrel.scatter.verts"),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// One scatter plot. `id` names its GPU slot and must be unique among the
/// scatters on screen; `generation` changes whenever `vertices` do.
pub struct ScatterCallback {
    pub id: u64,
    pub vertices: Arc<Vec<ScatterVertex>>,
    pub generation: u64,
    /// Data range shown along each axis: `(lo, hi)`.
    pub x: (f32, f32),
    pub y: (f32, f32),
    /// Marker radius in points, and the plot rect size in points.
    pub radius: f32,
    pub size: egui::Vec2,
}

impl egui_wgpu::CallbackTrait for ScatterCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(pipeline) = resources.get_mut::<ScatterPipeline>() else {
            log::warn!("ScatterPipeline missing from callback resources — skipping prepare");
            return Vec::new();
        };
        let slot = pipeline.slot(device, self.id);
        if slot.generation != Some(self.generation) {
            let bytes: &[u8] = bytemuck::cast_slice(&self.vertices);
            let needed = bytes.len() as u64;
            if needed > slot.capacity_bytes {
                slot.capacity_bytes = needed.next_power_of_two();
                slot.vertex_buffer = vertex_buffer(device, slot.capacity_bytes);
            }
            if !bytes.is_empty() {
                queue.write_buffer(&slot.vertex_buffer, 0, bytes);
            }
            slot.generation = Some(self.generation);
        }
        let inv = |(lo, hi): (f32, f32)| if hi > lo { 1.0 / (hi - lo) } else { 0.0 };
        let uniforms = ScatterUniforms {
            lo: [self.x.0, self.y.0],
            inv_span: [inv(self.x), inv(self.y)],
            radius: [
                2.0 * self.radius / self.size.x.max(1.0),
                2.0 * self.radius / self.size.y.max(1.0),
            ],
            _pad: [0.0; 2],
        };
        queue.write_buffer(&slot.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(pipeline) = resources.get::<ScatterPipeline>() else {
            return;
        };
        let Some(slot) = pipeline.slots.get(&self.id) else {
            return;
        };
        if self.vertices.is_empty() {
            return;
        }
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &slot.bind_group, &[]);
        pass.set_vertex_buffer(0, slot.vertex_buffer.slice(..));
        pass.draw(0..6, 0..self.vertices.len() as u32);
    }
}

const SHADER_WGSL: &str = r#"
struct Uniforms {
    lo: vec2<f32>,
    inv_span: vec2<f32>,
    radius: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) pos: vec2<f32>,
    @location(1) color: u32,
) -> VertexOut {
    var corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    );
    let corner = corners[vi];
    let centre = (pos - u.lo) * u.inv_span * 2.0 - 1.0;
    var out: VertexOut;
    out.clip_pos = vec4<f32>(centre + corner * u.radius, 0.0, 1.0);
    out.color = unpack4x8unorm(color);
    out.local = corner;
    return out;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    if dot(input.local, input.local) > 1.0 {
        discard;
    }
    return input.color;
}
"#;

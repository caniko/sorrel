//! Generic GPU 3-D renderer (points, lines, translucent triangles) behind an
//! orbit camera. It knows nothing about any one view: callers normalise their
//! data into the cube `[-1, 1]^3`, fill a [`Scene3d`] and paint a
//! [`Scene3dCallback`] into a rect.
//!
//! Like [`crate::wgpu_scatter`], every callback owns a slot keyed by a caller
//! id. egui's own pass has no depth buffer, so the scene is drawn in
//! `prepare` into a per-slot offscreen colour + `Depth32Float` target (sized
//! to the rect in physical pixels) and `paint` blits it into egui's pass.
//! Everything is premultiplied alpha, matching egui, and the target clears to
//! transparent so whatever the caller painted underneath shows through.
//! The target is only redrawn when the scene, camera or size change.
//!
//! Draw order: points (instanced camera-facing discs, depth test + write),
//! then lines, then triangles (depth test, no write, alpha-blended).

use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::{self, RenderState};
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

/// A disc in the scene. `color` is packed with [`crate::wgpu_raster::pack_rgba`].
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct Point3 {
    pub pos: [f32; 3],
    pub color: u32,
}

/// A line or triangle corner (same layout as [`Point3`]).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct Vert3 {
    pub pos: [f32; 3],
    pub color: u32,
}

#[derive(Debug, Default)]
pub struct Scene3d {
    pub points: Arc<[Point3]>,
    /// `LineList`: consecutive pairs.
    pub lines: Arc<[Vert3]>,
    /// `TriangleList`: consecutive triples, drawn double-sided.
    pub tris: Arc<[Vert3]>,
}

const FOV_Y: f32 = std::f32::consts::FRAC_PI_4;
const NEAR: f32 = 0.05;
const FAR: f32 = 100.0;

/// Orbit camera around `target`; `yaw`/`pitch` in radians, +Y up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Orbit {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: [f32; 3],
}

impl Default for Orbit {
    fn default() -> Self {
        Self {
            yaw: 0.6,
            pitch: 0.4,
            distance: 4.0,
            target: [0.0; 3],
        }
    }
}

type V3 = [f32; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V3) -> V3 {
    let l = dot(a, a).sqrt().max(1e-12);
    [a[0] / l, a[1] / l, a[2] / l]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

impl Orbit {
    fn eye(&self) -> V3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let t = self.target;
        [
            t[0] + self.distance * cp * sy,
            t[1] + self.distance * sp,
            t[2] + self.distance * cp * cy,
        ]
    }

    /// Camera right and up axes in world space.
    fn axes(&self) -> (V3, V3) {
        let f = norm(sub(self.target, self.eye()));
        let s = norm(cross(f, [0.0, 1.0, 0.0]));
        (s, cross(s, f))
    }

    /// Perspective × look-at, column-major (WGSL `mat4x4` order), depth 0..1.
    pub fn view_proj(&self, aspect: f32) -> [[f32; 4]; 4] {
        let eye = self.eye();
        let f = norm(sub(self.target, eye));
        let (s, u) = self.axes();
        let view = [
            [s[0], u[0], -f[0], 0.0],
            [s[1], u[1], -f[1], 0.0],
            [s[2], u[2], -f[2], 0.0],
            [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
        ];
        let t = 1.0 / (FOV_Y / 2.0).tan();
        let proj = [
            [t / aspect.max(1e-6), 0.0, 0.0, 0.0],
            [0.0, t, 0.0, 0.0],
            [0.0, 0.0, FAR / (NEAR - FAR), -1.0],
            [0.0, 0.0, NEAR * FAR / (NEAR - FAR), 0.0],
        ];
        // proj * view, column by column.
        let mut m = [[0.0; 4]; 4];
        for (c, col) in view.iter().enumerate() {
            for r in 0..4 {
                m[c][r] = (0..4).map(|k| proj[k][r] * col[k]).sum();
            }
        }
        m
    }

    /// Screen position of `p` when the scene is drawn into `rect`; `None`
    /// behind the camera. Lets egui paint labels over the render.
    pub fn project(&self, p: [f32; 3], rect: egui::Rect) -> Option<egui::Pos2> {
        let m = self.view_proj(rect.width() / rect.height().max(1e-6));
        let c = |r: usize| m[0][r] * p[0] + m[1][r] * p[1] + m[2][r] * p[2] + m[3][r];
        let w = c(3);
        if w < NEAR {
            return None;
        }
        Some(egui::pos2(
            rect.left() + (c(0) / w * 0.5 + 0.5) * rect.width(),
            rect.top() + (0.5 - c(1) / w * 0.5) * rect.height(),
        ))
    }

    /// Drag orbits, Shift/middle-drag pans, wheel zooms, double-click resets.
    /// Drags with Ctrl or Alt held are ignored. Returns whether the camera
    /// changed.
    pub fn interact(&mut self, resp: &egui::Response) -> bool {
        let before = *self;
        if resp.double_clicked() {
            *self = Self::default();
            return *self != before;
        }
        let mods = resp.ctx.input(|i| i.modifiers);
        let d = resp.drag_delta();
        if !(mods.ctrl || mods.alt) && d != egui::Vec2::ZERO {
            if mods.shift || resp.dragged_by(egui::PointerButton::Middle) {
                // World units per screen pixel at the target's depth.
                let k = 2.0 * self.distance * (FOV_Y / 2.0).tan() / resp.rect.height().max(1.0);
                let (r, u) = self.axes();
                for i in 0..3 {
                    self.target[i] += (-d.x * r[i] + d.y * u[i]) * k;
                }
            } else if resp.dragged_by(egui::PointerButton::Primary) {
                self.yaw -= d.x * 0.01;
                let lim = 89f32.to_radians();
                self.pitch = (self.pitch + d.y * 0.01).clamp(-lim, lim);
            }
        }
        if resp.hovered() {
            let scroll = resp.ctx.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.distance = (self.distance * (-scroll * 0.002).exp()).clamp(0.3, 30.0);
            }
        }
        *self != before
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    size_px: [f32; 2],
    radius_px: f32,
    _pad: f32,
}

const UNIFORMS_SIZE: u64 = std::mem::size_of::<Uniforms>() as u64;
const VERT_SIZE: u64 = std::mem::size_of::<Vert3>() as u64;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

struct Buf {
    buf: wgpu::Buffer,
    cap: u64,
}

impl Buf {
    fn new(device: &wgpu::Device, cap: u64) -> Self {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sorrel.scene3d.verts"),
            size: cap,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buf, cap }
    }

    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) {
        if bytes.len() as u64 > self.cap {
            *self = Self::new(device, (bytes.len() as u64).next_power_of_two());
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.buf, 0, bytes);
        }
    }
}

/// Offscreen colour + depth for one slot, and the blit bind group over it.
struct Target {
    size: [u32; 2],
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    blit_bg: wgpu::BindGroup,
}

struct Slot {
    bufs: [Buf; 3],
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    target: Option<Target>,
    generation: Option<u64>,
    /// What the target currently shows; skips the redraw when unchanged.
    drawn: Option<(u64, Uniforms)>,
}

pub struct Scene3dPipeline {
    format: wgpu::TextureFormat,
    /// Points, lines, triangles.
    pipelines: [wgpu::RenderPipeline; 3],
    bgl: wgpu::BindGroupLayout,
    blit: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    // ponytail: slots are never freed; there is one per 3-D view on screen.
    // Evict by last-used frame if views multiply.
    slots: HashMap<u64, Slot>,
}

impl Scene3dPipeline {
    pub fn install(rs: &RenderState) {
        let pipeline = Self::new(&rs.device, rs.target_format);
        rs.renderer.write().callback_resources.insert(pipeline);
    }

    /// `format` is the target the blit draws into (egui's surface format).
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sorrel.scene3d.shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sorrel.scene3d.bgl"),
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
            label: Some("sorrel.scene3d.layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let attributes = [
            wgpu::VertexAttribute {
                shader_location: 0,
                offset: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                shader_location: 1,
                offset: 12,
                format: wgpu::VertexFormat::Uint32,
            },
        ];
        let make = |vs: &str, topology, step_mode, write: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sorrel.scene3d.pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: VERT_SIZE,
                        step_mode,
                        attributes: &attributes,
                    }],
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(write),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        use wgpu::{PrimitiveTopology as T, VertexStepMode as S};
        let pipelines = [
            make("vs_point", T::TriangleList, S::Instance, true),
            make("vs_vert", T::LineList, S::Vertex, false),
            make("vs_vert", T::TriangleList, S::Vertex, false),
        ];

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sorrel.scene3d.blit_shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sorrel.scene3d.blit_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sorrel.scene3d.blit_layout"),
            bind_group_layouts: &[Some(&blit_bgl)],
            immediate_size: 0,
        });
        let blit = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sorrel.scene3d.blit"),
            layout: Some(&blit_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_blit"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_blit"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sorrel.scene3d.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            format,
            pipelines,
            bgl,
            blit,
            blit_bgl,
            sampler,
            slots: HashMap::new(),
        }
    }

    /// The slot's offscreen colour texture (tests read it back).
    pub fn color_texture(&self, id: u64) -> Option<&wgpu::Texture> {
        Some(&self.slots.get(&id)?.target.as_ref()?.color)
    }

    /// Upload `cb`'s scene if its generation changed, (re)size the offscreen
    /// target, and draw into it on `encoder` unless it already shows exactly
    /// this. The path `prepare` takes; also callable without egui.
    pub fn render_offscreen(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        cb: &Scene3dCallback,
    ) {
        let bgl = &self.bgl;
        let slot = self.slots.entry(cb.id).or_insert_with(|| {
            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sorrel.scene3d.uniforms"),
                size: UNIFORMS_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sorrel.scene3d.bg"),
                layout: bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });
            Slot {
                bufs: [(); 3].map(|_| Buf::new(device, VERT_SIZE * 256)),
                uniform_buffer,
                bind_group,
                target: None,
                generation: None,
                drawn: None,
            }
        });
        let scene = &cb.scene;
        if slot.generation != Some(cb.generation) {
            let bytes: [&[u8]; 3] = [
                bytemuck::cast_slice(&scene.points),
                bytemuck::cast_slice(&scene.lines),
                bytemuck::cast_slice(&scene.tris),
            ];
            for (buf, b) in slot.bufs.iter_mut().zip(bytes) {
                buf.upload(device, queue, b);
            }
            slot.generation = Some(cb.generation);
        }

        let max = device.limits().max_texture_dimension_2d;
        let size = cb.size_px.map(|s| s.clamp(1, max));
        if slot.target.as_ref().map(|t| t.size) != Some(size) {
            let extent = wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            };
            let tex = |label, format, usage| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | usage,
                    view_formats: &[],
                })
            };
            let color = tex(
                "sorrel.scene3d.color",
                self.format,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            );
            let color_view = color.create_view(&Default::default());
            let depth_view = tex(
                "sorrel.scene3d.depth",
                DEPTH_FORMAT,
                wgpu::TextureUsages::empty(),
            )
            .create_view(&Default::default());
            let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sorrel.scene3d.blit_bg"),
                layout: &self.blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            slot.target = Some(Target {
                size,
                color,
                color_view,
                depth_view,
                blit_bg,
            });
            slot.drawn = None;
        }

        let uniforms = Uniforms {
            view_proj: cb.view_proj,
            size_px: size.map(|s| s as f32),
            radius_px: cb.point_radius_px,
            _pad: 0.0,
        };
        if slot.drawn == Some((cb.generation, uniforms)) {
            return;
        }
        slot.drawn = Some((cb.generation, uniforms));
        queue.write_buffer(&slot.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        let target = slot.target.as_ref().expect("target created above");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sorrel.scene3d.pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &slot.bind_group, &[]);
        let counts = [scene.points.len(), scene.lines.len(), scene.tris.len()];
        for (i, &n) in counts.iter().enumerate() {
            if n == 0 {
                continue;
            }
            pass.set_pipeline(&self.pipelines[i]);
            pass.set_vertex_buffer(0, slot.bufs[i].buf.slice(..));
            if i == 0 {
                pass.draw(0..6, 0..n as u32);
            } else {
                pass.draw(0..n as u32, 0..1);
            }
        }
    }
}

/// One 3-D view. `id` names its GPU slot and must be unique among the views
/// on screen; `generation` changes whenever `scene` does. Camera and point
/// size changes need no re-upload.
pub struct Scene3dCallback {
    pub id: u64,
    pub scene: Arc<Scene3d>,
    pub generation: u64,
    /// From [`Orbit::view_proj`] with the rect's aspect ratio.
    pub view_proj: [[f32; 4]; 4],
    /// Disc radius in physical pixels.
    pub point_radius_px: f32,
    /// Offscreen size: the rect in physical pixels (`rect.size() *
    /// pixels_per_point`, rounded).
    pub size_px: [u32; 2],
}

impl egui_wgpu::CallbackTrait for Scene3dCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(pipeline) = resources.get_mut::<Scene3dPipeline>() else {
            log::warn!("Scene3dPipeline missing from callback resources — skipping prepare");
            return Vec::new();
        };
        pipeline.render_offscreen(device, queue, encoder, self);
        Vec::new()
    }

    /// egui has already set the pass viewport to the callback rect, so the
    /// fullscreen triangle lands exactly on it (the scissor is the clip rect).
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(pipeline) = resources.get::<Scene3dPipeline>() else {
            return;
        };
        let Some(target) = pipeline.slots.get(&self.id).and_then(|s| s.target.as_ref()) else {
            return;
        };
        pass.set_pipeline(&pipeline.blit);
        pass.set_bind_group(0, &target.blit_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

const SHADER_WGSL: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    size_px: vec2<f32>,
    radius_px: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local: vec2<f32>,
};

fn corner_of(vi: u32) -> vec2<f32> {
    var corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    );
    return corners[vi];
}

// One instanced quad per point, offset in clip space so it stays
// `radius_px` wide at any depth.
@vertex
fn vs_point(
    @builtin(vertex_index) vi: u32,
    @location(0) pos: vec3<f32>,
    @location(1) color: u32,
) -> VertexOut {
    let corner = corner_of(vi);
    let c = u.view_proj * vec4<f32>(pos, 1.0);
    var out: VertexOut;
    out.clip_pos = vec4<f32>(c.xy + corner * u.radius_px * 2.0 / u.size_px * c.w, c.zw);
    out.color = unpack4x8unorm(color);
    out.local = corner;
    return out;
}

@vertex
fn vs_vert(@location(0) pos: vec3<f32>, @location(1) color: u32) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = u.view_proj * vec4<f32>(pos, 1.0);
    out.color = unpack4x8unorm(color);
    out.local = vec2<f32>(0.0, 0.0);
    return out;
}

// Discs and lines share this; lines and triangles have local == 0.
@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    if dot(input.local, input.local) > 1.0 {
        discard;
    }
    return vec4<f32>(input.color.rgb * input.color.a, input.color.a);
}

"#;

// Fullscreen triangle sampling the offscreen colour.
const BLIT_WGSL: &str = r#"
@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_samp: sampler;

struct BlitOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_blit(@builtin(vertex_index) vi: u32) -> BlitOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var out: BlitOut;
    out.pos = vec4<f32>(p[vi], 0.0, 1.0);
    out.uv = vec2<f32>(p[vi].x * 0.5 + 0.5, 0.5 - p[vi].y * 0.5);
    return out;
}

@fragment
fn fs_blit(input: BlitOut) -> @location(0) vec4<f32> {
    return textureSample(scene_tex, scene_samp, input.uv);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit() {
        let rect = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(200.0, 100.0));
        for o in [
            Orbit::default(),
            Orbit {
                yaw: -2.0,
                pitch: 1.0,
                distance: 2.0,
                target: [0.3, -0.2, 0.5],
            },
        ] {
            let c = o.project(o.target, rect).expect("target is in front");
            assert!((c - rect.center()).length() < 1e-3, "{c:?}");
            // A point behind the camera does not project.
            let behind = [
                o.target[0] + 2.0 * (o.eye()[0] - o.target[0]),
                o.target[1] + 2.0 * (o.eye()[1] - o.target[1]),
                o.target[2] + 2.0 * (o.eye()[2] - o.target[2]),
            ];
            assert!(o.project(behind, rect).is_none());
        }
        // +X is to the right of the target at yaw 0.
        let o = Orbit {
            yaw: 0.0,
            pitch: 0.0,
            ..Orbit::default()
        };
        assert!(o.project([0.5, 0.0, 0.0], rect).unwrap().x > rect.center().x);
        assert!(o.project([0.0, 0.5, 0.0], rect).unwrap().y < rect.center().y);
    }
}

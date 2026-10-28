//! GPU per-unit spatiotemporal matched filter — the inner loop of spike
//! rescue. Mirrors [`sorrel_compute::rescue::matched_filter_reference`].
//!
//! For a `n_time × n_chan` row-major chunk (already high-passed and CMR'd),
//! per-unit filters `U × C × nt` over each unit's `C` channels and one
//! circular `shift` per channel, computes
//! `out[u][t] = Σ_c Σ_k filters[u][c][k] · data[(t + k + shifts[ch]) mod n_time][ch]`
//! with `ch = chans[u][c]`. The output is `U × n_time`, unit-major. The wrap
//! is what makes channel-shifted null replicates possible; the real pass
//! passes zero shifts and the caller discards the last `nt - 1` samples (or
//! overlaps chunks).

use crate::{block_on_map, GpuContext};
use bytemuck::{Pod, Zeroable};
use std::num::NonZeroU64;

const SHADER: &str = include_str!("shaders/matched_filter.wgsl");
const WORKGROUP: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct Params {
    n_time: u32,
    n_chan: u32,
    nt: u32,
    c_per_unit: u32,
    n_units: u32,
}

const PARAMS_SIZE: u64 = std::mem::size_of::<Params>() as u64;

/// Per-unit filters: `filters` is `U × c_per_unit × nt` (channel-major per
/// unit, as `sorrel_compute::rescue::matched_filter_weights` returns) and
/// `chans` the `U × c_per_unit` channel each filter row reads.
#[derive(Clone, Copy, Debug)]
pub struct MatchedFilterBank<'a> {
    pub filters: &'a [f32],
    pub chans: &'a [u32],
    pub nt: u32,
    pub c_per_unit: u32,
}

pub struct GpuMatchedFilter {
    ctx: GpuContext,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

fn layout_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: (binding == 0)
                .then(|| NonZeroU64::new(PARAMS_SIZE))
                .flatten(),
        },
        count: None,
    }
}

impl GpuMatchedFilter {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = &ctx.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sorrel-gpu.matched_filter.shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sorrel-gpu.matched_filter.bgl"),
            entries: &[
                layout_entry(0, wgpu::BufferBindingType::Uniform),
                layout_entry(1, ro),
                layout_entry(2, ro),
                layout_entry(3, ro),
                layout_entry(4, ro),
                layout_entry(5, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sorrel-gpu.matched_filter.layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sorrel-gpu.matched_filter.pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            ctx: ctx.clone(),
            pipeline,
            bgl,
        }
    }

    /// Run the filter bank. `data` is `n_time × n_chan` row-major (`n_time`
    /// is inferred), `bank` the per-unit filters, `shifts` one circular
    /// shift per channel (any sign or size; reduced mod `n_time`). Returns
    /// `U × n_time` scores, unit-major. Units are dispatched in chunks that
    /// fit the device's buffer and dispatch limits, so any `U` works as long
    /// as `data` and one unit fit.
    pub fn run(
        &self,
        data: &[f32],
        n_chan: u32,
        bank: &MatchedFilterBank<'_>,
        shifts: &[i32],
    ) -> Result<Vec<f32>, GpuMatchedFilterError> {
        use GpuMatchedFilterError as E;
        let MatchedFilterBank {
            filters,
            chans,
            nt,
            c_per_unit,
        } = *bank;
        if n_chan == 0 || nt == 0 || c_per_unit == 0 || data.is_empty() {
            return Ok(Vec::new());
        }
        let shape = |m: &str| Err(E::ShapeMismatch(m.into()));
        if !data.len().is_multiple_of(n_chan as usize) {
            return shape("data length not a multiple of n_chan");
        }
        if !chans.len().is_multiple_of(c_per_unit as usize) {
            return shape("chans length not a multiple of c_per_unit");
        }
        let n_time = (data.len() / n_chan as usize) as u32;
        let n_units = chans.len() / c_per_unit as usize;
        if filters.len() != n_units * c_per_unit as usize * nt as usize {
            return shape("filters length != U * c_per_unit * nt");
        }
        if shifts.len() != n_chan as usize {
            return shape("shifts length != n_chan");
        }
        if chans.iter().any(|&c| c >= n_chan) {
            return shape("channel index >= n_chan");
        }
        if n_units == 0 {
            return Ok(Vec::new());
        }
        let shifts: Vec<u32> = shifts
            .iter()
            .map(|&s| i64::from(s).rem_euclid(i64::from(n_time)) as u32)
            .collect();

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let limits = device.limits();
        let max_bytes = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        crate::check_limits(
            device,
            std::mem::size_of_val(data) as u64,
            n_time.div_ceil(WORKGROUP),
        )
        .map_err(E::Limits)?;

        // Units per dispatch: bounded by the output / filter buffer size and
        // by the y-dimension dispatch limit.
        let out_per_unit = u64::from(n_time) * 4;
        let filt_per_unit = u64::from(c_per_unit) * u64::from(nt) * 4;
        let per_chunk = (max_bytes / out_per_unit)
            .min(max_bytes / filt_per_unit)
            .min(u64::from(limits.max_compute_workgroups_per_dimension))
            .min(n_units as u64) as usize;
        if per_chunk == 0 {
            return Err(E::Limits(
                "one unit exceeds the storage buffer limit".into(),
            ));
        }

        let storage = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let upload = |label, bytes: &[u8]| {
            let b = storage(label, bytes.len() as u64);
            queue.write_buffer(&b, 0, bytes);
            b
        };
        let data_buf = upload("sorrel-gpu.matched_filter.data", bytemuck::cast_slice(data));
        let shifts_buf = upload(
            "sorrel-gpu.matched_filter.shifts",
            bytemuck::cast_slice(&shifts),
        );

        let mut out = Vec::with_capacity(n_units * n_time as usize);
        for u0 in (0..n_units).step_by(per_chunk) {
            let u1 = (u0 + per_chunk).min(n_units);
            let units = (u1 - u0) as u32;
            let (cw, fw) = (c_per_unit as usize, nt as usize);
            let chans_buf = upload(
                "sorrel-gpu.matched_filter.chans",
                bytemuck::cast_slice(&chans[u0 * cw..u1 * cw]),
            );
            let filters_buf = upload(
                "sorrel-gpu.matched_filter.filters",
                bytemuck::cast_slice(&filters[u0 * cw * fw..u1 * cw * fw]),
            );
            let out_bytes = u64::from(units) * out_per_unit;
            let out_buf = storage("sorrel-gpu.matched_filter.out", out_bytes);
            let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sorrel-gpu.matched_filter.params"),
                size: PARAMS_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(
                &params_buf,
                0,
                bytemuck::bytes_of(&Params {
                    n_time,
                    n_chan,
                    nt,
                    c_per_unit,
                    n_units: units,
                }),
            );
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sorrel-gpu.matched_filter.bg"),
                layout: &self.bgl,
                entries: &[
                    (0, &params_buf),
                    (1, &data_buf),
                    (2, &filters_buf),
                    (3, &chans_buf),
                    (4, &shifts_buf),
                    (5, &out_buf),
                ]
                .map(|(binding, buf)| wgpu::BindGroupEntry {
                    binding,
                    resource: buf.as_entire_binding(),
                }),
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sorrel-gpu.matched_filter.readback"),
                size: out_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sorrel-gpu.matched_filter.enc"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sorrel-gpu.matched_filter.pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(n_time.div_ceil(WORKGROUP), units, 1);
            }
            enc.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, out_bytes);
            queue.submit(Some(enc.finish()));

            let slice = readback.slice(..);
            block_on_map(device, slice).map_err(|e| E::Readback(e.to_string()))?;
            out.extend_from_slice(bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()));
            readback.unmap();
        }
        Ok(out)
    }
}

#[derive(Debug)]
pub enum GpuMatchedFilterError {
    ShapeMismatch(String),
    Limits(String),
    Readback(String),
}

impl std::fmt::Display for GpuMatchedFilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch(e) => write!(f, "shape mismatch: {e}"),
            Self::Limits(e) => write!(f, "{e}"),
            Self::Readback(e) => write!(f, "readback failed: {e}"),
        }
    }
}
impl std::error::Error for GpuMatchedFilterError {}

#[cfg(test)]
mod tests {
    use super::*;
    use sorrel_compute::rescue::matched_filter_reference;

    fn rnd(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    struct Case {
        data: Vec<f32>,
        filters: Vec<f32>,
        chans: Vec<u32>,
        shifts: Vec<i32>,
        n_chan: usize,
        nt: usize,
        c: usize,
    }

    fn case(n_time: usize, n_chan: usize, nt: usize, c: usize, units: usize, shift: bool) -> Case {
        let mut s = 0x9e3779b97f4a7c15;
        Case {
            data: (0..n_time * n_chan).map(|_| rnd(&mut s)).collect(),
            filters: (0..units * c * nt).map(|_| rnd(&mut s)).collect(),
            chans: (0..units * c)
                .map(|i| ((i * 7 + i / c) % n_chan) as u32)
                .collect(),
            shifts: (0..n_chan)
                .map(|ch| if shift { (ch as i32 * 37) - 100 } else { 0 })
                .collect(),
            n_chan,
            nt,
            c,
        }
    }

    impl Case {
        fn bank(&self) -> MatchedFilterBank<'_> {
            MatchedFilterBank {
                filters: &self.filters,
                chans: &self.chans,
                nt: self.nt as u32,
                c_per_unit: self.c as u32,
            }
        }
    }

    fn check(gpu: &GpuMatchedFilter, k: &Case) {
        let got = gpu
            .run(&k.data, k.n_chan as u32, &k.bank(), &k.shifts)
            .unwrap();
        let want = matched_filter_reference(
            &k.data, k.n_chan, &k.filters, &k.chans, k.nt, k.c, &k.shifts,
        );
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(1.0_f32, |m, v| m.max(v.abs()));
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 1e-4 * scale, "i={i}: gpu={g} cpu={w}");
        }
    }

    #[test]
    fn matches_cpu_reference() {
        let Some(ctx) = crate::test_support::ctx() else {
            return;
        };
        let gpu = GpuMatchedFilter::new(&ctx);
        // Odd sizes: n_time not a workgroup multiple, shifts exercise the wrap.
        check(&gpu, &case(1000, 24, 21, 6, 9, false));
        check(&gpu, &case(1000, 24, 21, 6, 9, true));
    }

    fn bank<'a>(filters: &'a [f32], chans: &'a [u32]) -> MatchedFilterBank<'a> {
        MatchedFilterBank {
            filters,
            chans,
            nt: 3,
            c_per_unit: 2,
        }
    }

    #[test]
    fn rejects_bad_shapes_and_handles_empty() {
        let Some(ctx) = crate::test_support::ctx() else {
            return;
        };
        let gpu = GpuMatchedFilter::new(&ctx);
        assert!(gpu
            .run(&[], 4, &bank(&[], &[]), &[0; 4])
            .unwrap()
            .is_empty());
        let r = gpu.run(&[0.0; 40], 4, &bank(&[0.0; 5], &[0, 1]), &[0; 4]);
        assert!(matches!(r, Err(GpuMatchedFilterError::ShapeMismatch(_))));
        let r = gpu.run(&[0.0; 40], 4, &bank(&[0.0; 6], &[0, 9]), &[0; 4]);
        assert!(matches!(r, Err(GpuMatchedFilterError::ShapeMismatch(_))));
    }

    /// Rough throughput probe at Neuropixels scale; run manually with
    /// `cargo test -p sorrel-gpu --release -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_neuropixels_scale() {
        let Some(ctx) = crate::test_support::ctx() else {
            return;
        };
        let gpu = GpuMatchedFilter::new(&ctx);
        for c in [16usize, 32] {
            let k = case(60_000, 384, 61, c, 300, false);
            let t0 = std::time::Instant::now();
            let out = gpu.run(&k.data, 384, &k.bank(), &k.shifts).unwrap();
            let dt = t0.elapsed().as_secs_f64();
            let gmac = (60_000.0 * 300.0 * c as f64 * 61.0) / 1e9;
            eprintln!(
                "C={c}: {dt:.3}s per 2 s chunk, {:.0} GMAC/s ({} outputs)",
                gmac / dt,
                out.len()
            );
        }
    }
}

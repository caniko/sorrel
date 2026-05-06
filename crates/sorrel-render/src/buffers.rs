use crate::vertex::{ScatterVertex, TraceVertex};
use sorrel_compute::lttb_downsample;
use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider, SampleIndex};

/// Build a flat trace vertex buffer for a window. Generic over the backend so
/// the producer-consumer chain (mmap → LTTB → vertex) is monomorphised.
pub fn build_trace_vertices<P: DataProvider>(
    session: &Session<P>,
    start: SampleIndex,
    len: u32,
    target_points_per_channel: usize,
) -> Vec<TraceVertex> {
    let provider = &session.provider;
    let slice = provider.trace(start, len);
    let nc = slice.n_channels as usize;
    if nc == 0 || slice.samples.is_empty() {
        return Vec::new();
    }

    let n_samples = slice.samples.len() / nc;
    let x_step = 1.0f32 / provider.sample_rate();
    let x_origin = start as f32 * x_step;

    let mut out: Vec<TraceVertex> = Vec::with_capacity(target_points_per_channel * nc);

    // Walk channels in the outer loop so the inner LTTB sees a contiguous,
    // strided iterator the optimiser can collapse.
    for ch in 0..nc {
        let mut tmp: Vec<i16> = Vec::with_capacity(n_samples);
        let mut p = ch;
        while p < slice.samples.len() {
            tmp.push(slice.samples[p]);
            p += nc;
        }
        let pts = lttb_downsample::<i16>(&tmp, target_points_per_channel, x_origin, x_step);
        out.extend(pts.into_iter().map(|[x, y]| TraceVertex {
            pos: [x, y],
            channel: ch as u32,
        }));
    }
    out
}

/// Build a flat scatter buffer of `(time, cluster_y)` points, one per spike,
/// over the given clusters.
pub fn build_scatter_vertices<P: DataProvider>(
    session: &Session<P>,
    clusters: &[ClusterId],
) -> Vec<ScatterVertex> {
    let provider = &session.provider;
    let inv_sr = 1.0f32 / provider.sample_rate();
    let mut out = Vec::new();
    for (row, &c) in clusters.iter().enumerate() {
        let y = row as f32;
        for &t in provider.spike_times(c) {
            out.push(ScatterVertex {
                pos: [t as f32 * inv_sr, y],
                cluster: c,
            });
        }
    }
    out
}

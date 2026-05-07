use crate::vertex::{ScatterVertex, TraceVertex};
use sorrel_compute::{lttb_downsample, subtract_channel_median, Biquad, BiquadState};
use sorrel_data::Session;
use sorrel_gpu::{GpuBiquad, GpuCmr, GpuContext};
use sorrel_io::{ClusterId, DataProvider, SampleIndex, TraceSamples};

/// Optional pre-processing applied to the trace window before LTTB.
///
/// `Off` means the on-disk mmap is the truth — this is correct when
/// `params.py:hp_filtered = True` and the recording is already CMR'd, where
/// the extra stages would be redundant.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum TracePreproc {
    #[default]
    Off,
    /// HP filter only at the given cutoff in Hz.
    Hp(f32),
    /// Per-time-step channel median subtraction only.
    Cmr,
    /// CMR followed by HP filter at the given cutoff in Hz.
    HpAndCmr(f32),
}

impl TracePreproc {
    /// phy's default HP cutoff for the spike band.
    pub const HP_DEFAULT_HZ: f32 = 300.0;

    /// HP cutoff in Hz, if HP is enabled.
    pub const fn hp(self) -> Option<f32> {
        match self {
            Self::Hp(hz) | Self::HpAndCmr(hz) => Some(hz),
            Self::Off | Self::Cmr => None,
        }
    }

    pub const fn has_cmr(self) -> bool {
        matches!(self, Self::Cmr | Self::HpAndCmr(_))
    }

    /// True when no preprocessing applies — render code can take the
    /// dtype-native fast path.
    pub const fn is_off(self) -> bool {
        matches!(self, Self::Off)
    }

    /// Build from independent HP / CMR controls (e.g. two UI checkboxes).
    pub const fn from_flags(hp: Option<f32>, cmr: bool) -> Self {
        match (hp, cmr) {
            (None, false) => Self::Off,
            (Some(hz), false) => Self::Hp(hz),
            (None, true) => Self::Cmr,
            (Some(hz), true) => Self::HpAndCmr(hz),
        }
    }
}

/// Build a flat trace vertex buffer for a window. Generic over the backend so
/// the producer-consumer chain (mmap → LTTB → vertex) is monomorphised.
pub fn build_trace_vertices<P: DataProvider>(
    session: &Session<P>,
    start: SampleIndex,
    len: u32,
    target_points_per_channel: usize,
) -> Vec<TraceVertex> {
    build_trace_vertices_cfg(
        session,
        start,
        len,
        target_points_per_channel,
        TracePreproc::Off,
    )
}

/// Like [`build_trace_vertices`] but applies optional HP filter / CMR
/// pre-processing per [`TracePreproc`] before running LTTB.
pub fn build_trace_vertices_cfg<P: DataProvider>(
    session: &Session<P>,
    start: SampleIndex,
    len: u32,
    target_points_per_channel: usize,
    cfg: TracePreproc,
) -> Vec<TraceVertex> {
    let provider = &session.provider;
    let slice = provider.trace(start, len);
    let nc = slice.n_channels as usize;
    if nc == 0 || slice.samples.is_empty() {
        return Vec::new();
    }

    let x_step = 1.0f32 / provider.sample_rate();
    let x_origin = start.as_f32() * x_step;
    let cap = target_points_per_channel * nc;

    if cfg.is_off() {
        // Fast path: dispatch on the native dtype, do no preprocessing.
        return match slice.samples {
            TraceSamples::I16(s) => {
                lttb_per_channel(s, nc, target_points_per_channel, x_origin, x_step, cap)
            }
            TraceSamples::U16(s) => {
                let centred: Vec<f32> =
                    s.iter().map(|&v| v as f32 - i16::MAX as f32).collect();
                lttb_per_channel(&centred, nc, target_points_per_channel, x_origin, x_step, cap)
            }
            TraceSamples::I32(s) => {
                let f: Vec<f32> = s.iter().map(|&v| v as f32).collect();
                lttb_per_channel(&f, nc, target_points_per_channel, x_origin, x_step, cap)
            }
            TraceSamples::F32(s) => {
                lttb_per_channel(s, nc, target_points_per_channel, x_origin, x_step, cap)
            }
        };
    }

    // Slow path: convert to f32, run CMR + per-channel HP, then LTTB.
    let mut buf = slice.samples.to_f32_centred();
    apply_preproc_cpu(&mut buf, nc, cfg, provider.sample_rate());
    lttb_per_channel(&buf, nc, target_points_per_channel, x_origin, x_step, cap)
}

/// GPU-accelerated counterpart of [`build_trace_vertices_cfg`].
///
/// Uses GPU compute kernels for CMR (median across channels) and the HP
/// Biquad filter when both apply and `n_channels ≤ 512`. LTTB stays on the
/// CPU — it's already cheap relative to the 384-channel preprocessing.
/// Falls back to the CPU path on any GPU error so a transient device hiccup
/// never blanks the trace view.
pub fn build_trace_vertices_gpu<P: DataProvider>(
    session: &Session<P>,
    start: SampleIndex,
    len: u32,
    target_points_per_channel: usize,
    cfg: TracePreproc,
    gpu: &GpuTracePreproc,
) -> Vec<TraceVertex> {
    let provider = &session.provider;
    let slice = provider.trace(start, len);
    let nc = slice.n_channels as usize;
    if nc == 0 || slice.samples.is_empty() {
        return Vec::new();
    }

    let x_step = 1.0f32 / provider.sample_rate();
    let x_origin = start.as_f32() * x_step;
    let cap = target_points_per_channel * nc;

    if cfg.is_off() {
        // Same fast path as CPU: skip preprocessing entirely.
        return build_trace_vertices_cfg(
            session,
            start,
            len,
            target_points_per_channel,
            cfg,
        );
    }

    let mut buf = slice.samples.to_f32_centred();
    let sr = provider.sample_rate();
    if !gpu.apply(&mut buf, slice.n_channels, cfg, sr) {
        // GPU returned an error — finish on CPU so the user still sees the
        // trace.
        apply_preproc_cpu(&mut buf, nc, cfg, sr);
    }
    lttb_per_channel(&buf, nc, target_points_per_channel, x_origin, x_step, cap)
}

/// Reusable GPU preprocessor for trace windows. Owns one CMR pipeline and
/// one Biquad pipeline so we don't recompile shaders every frame.
pub struct GpuTracePreproc {
    cmr: GpuCmr,
    biquad: GpuBiquad,
}

impl GpuTracePreproc {
    pub fn new(ctx: &GpuContext) -> Self {
        Self {
            cmr: GpuCmr::new(ctx),
            biquad: GpuBiquad::new(ctx),
        }
    }

    /// Apply CMR + HP filter on the GPU. Returns `false` if any stage fails
    /// — the caller is expected to retry on CPU.
    pub fn apply(
        &self,
        buf: &mut [f32],
        n_channels: u32,
        cfg: TracePreproc,
        sample_rate: f32,
    ) -> bool {
        if cfg.has_cmr() {
            if let Err(e) = self.cmr.run(buf, n_channels) {
                log::warn!("GPU CMR failed, falling back to CPU: {e}");
                return false;
            }
        }
        if let Some(hz) = cfg.hp() {
            let b = Biquad::butterworth_hp(hz, sample_rate);
            if let Err(e) = self.biquad.run(buf, n_channels, b) {
                log::warn!("GPU Biquad failed, falling back to CPU: {e}");
                return false;
            }
        }
        true
    }
}

fn apply_preproc_cpu(buf: &mut [f32], nc: usize, cfg: TracePreproc, sample_rate: f32) {
    if cfg.has_cmr() {
        subtract_channel_median(buf, nc);
    }
    if let Some(hz) = cfg.hp() {
        let f = Biquad::butterworth_hp(hz, sample_rate);
        let mut states = vec![BiquadState::default(); nc];
        let n_samples = buf.len() / nc;
        for t in 0..n_samples {
            let base = t * nc;
            for ch in 0..nc {
                let i = base + ch;
                let y = f.step(buf[i], &mut states[ch]);
                buf[i] = y;
            }
        }
    }
}

fn lttb_per_channel<S>(
    samples: &[S],
    nc: usize,
    target: usize,
    x_origin: f32,
    x_step: f32,
    cap: usize,
) -> Vec<TraceVertex>
where
    S: Copy + Into<f32>,
{
    let n_samples = samples.len() / nc;
    let mut out = Vec::with_capacity(cap);
    let mut tmp: Vec<S> = Vec::with_capacity(n_samples);
    for ch in 0..nc {
        tmp.clear();
        let mut p = ch;
        while p < samples.len() {
            tmp.push(samples[p]);
            p += nc;
        }
        let pts = lttb_downsample::<S>(&tmp, target, x_origin, x_step);
        out.extend(pts.into_iter().map(|[x, y]| TraceVertex {
            pos: [x, y],
            channel: ch as u32,
        }));
    }
    out
}

/// Build a flat scatter buffer of `(time, cluster_y)` points, one per spike,
/// over the given clusters. Reads from the live, post-curation cluster index
/// via `Session::spike_times`.
pub fn build_scatter_vertices<P: DataProvider>(
    session: &Session<P>,
    clusters: &[ClusterId],
) -> Vec<ScatterVertex> {
    let inv_sr = 1.0f32 / session.provider.sample_rate();
    let mut out = Vec::new();
    for (row, &c) in clusters.iter().enumerate() {
        let y = row as f32;
        for &t in session.spike_times(c) {
            out.push(ScatterVertex {
                pos: [t.as_f32() * inv_sr, y],
                cluster: c.0,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sorrel_data::SqliteJournal;
    use sorrel_io::{ClusterId, DataProvider, SampleIndex, TraceSlice};

    struct MockProvider {
        spikes: Vec<Vec<SampleIndex>>,
        trace: Vec<i16>,
        n_channels: u32,
        sr: f32,
    }

    impl DataProvider for MockProvider {
        type Label = u8;
        fn sample_rate(&self) -> f32 { self.sr }
        fn n_channels(&self) -> u32 { self.n_channels }
        fn n_samples(&self) -> SampleIndex {
            SampleIndex((self.trace.len() / self.n_channels.max(1) as usize) as u64)
        }
        fn n_clusters(&self) -> u32 { self.spikes.len() as u32 }
        fn spike_times(&self, cluster: ClusterId) -> &[SampleIndex] {
            self.spikes.get(cluster.idx()).map(Vec::as_slice).unwrap_or(&[])
        }
        fn trace(&self, start: SampleIndex, len: u32) -> TraceSlice<'_> {
            let nc = self.n_channels as usize;
            let s = (start.idx() * nc).min(self.trace.len());
            let e = (s + len as usize * nc).min(self.trace.len());
            TraceSlice {
                start,
                n_channels: self.n_channels,
                samples: TraceSamples::I16(&self.trace[s..e]),
            }
        }
        fn initial_labels(&self) -> Vec<u8> { vec![0u8; self.spikes.len()] }
    }

    fn session_with(provider: MockProvider) -> (Session<MockProvider>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let j = SqliteJournal::open(&dir.path().join("j.sqlite")).unwrap();
        (Session::new(provider, j), dir)
    }

    #[test]
    fn scatter_vertices_one_per_spike_with_correct_row_and_time() {
        let provider = MockProvider {
            spikes: vec![vec![SampleIndex(0), SampleIndex(100), SampleIndex(200)], vec![SampleIndex(50)]],
            trace: vec![],
            n_channels: 1,
            sr: 1000.0,
        };
        let (session, _d) = session_with(provider);
        let v = build_scatter_vertices(&session, &[ClusterId(0), ClusterId(1)]);
        assert_eq!(v.len(), 4);

        // Cluster 0 (row 0): 3 spikes.
        assert_eq!(v[0].cluster, 0);
        assert_eq!(v[0].pos, [0.0, 0.0]);
        assert_eq!(v[1].pos, [0.1, 0.0]);
        assert_eq!(v[2].pos, [0.2, 0.0]);
        // Cluster 1 (row 1): 1 spike at t = 50/1000 = 0.05.
        assert_eq!(v[3].cluster, 1);
        assert_eq!(v[3].pos, [0.05, 1.0]);
    }

    #[test]
    fn scatter_vertices_skips_unknown_cluster_ids() {
        let provider = MockProvider {
            spikes: vec![vec![SampleIndex(1), SampleIndex(2)]],
            trace: vec![],
            n_channels: 1,
            sr: 100.0,
        };
        let (session, _d) = session_with(provider);
        let v = build_scatter_vertices(&session, &[ClusterId(0), ClusterId(99)]);
        assert_eq!(v.len(), 2);
        assert!(v.iter().all(|x| x.cluster == 0));
    }

    #[test]
    fn trace_vertices_emit_per_channel_when_downsampling() {
        let n_channels = 3u32;
        let n_samples = 200usize;
        let mut trace = Vec::with_capacity(n_channels as usize * n_samples);
        for s in 0..n_samples {
            for ch in 0..n_channels as usize {
                trace.push(((s as i32) * (ch as i32 + 1)) as i16);
            }
        }
        let provider = MockProvider {
            spikes: vec![],
            trace,
            n_channels,
            sr: 1000.0,
        };
        let (session, _d) = session_with(provider);
        let target = 32usize;
        let v = build_trace_vertices(&session, SampleIndex(0), n_samples as u32, target);
        assert_eq!(v.len(), target * n_channels as usize);
        // Channels are emitted in outer-loop order: first `target` verts are channel 0, etc.
        assert_eq!(v[0].channel, 0);
        assert_eq!(v[target].channel, 1);
        assert_eq!(v[2 * target].channel, 2);
    }

    #[test]
    fn trace_vertices_empty_for_zero_channels_or_no_data() {
        let (session, _d) = session_with(MockProvider {
            spikes: vec![],
            trace: vec![],
            n_channels: 0,
            sr: 1000.0,
        });
        assert!(build_trace_vertices(&session, SampleIndex(0), 100, 32).is_empty());
    }

    #[test]
    fn trace_config_off_matches_legacy_no_arg_call() {
        let n_channels = 2u32;
        let n_samples = 128usize;
        let mut trace = Vec::with_capacity(n_channels as usize * n_samples);
        for s in 0..n_samples {
            for ch in 0..n_channels as usize {
                trace.push(((s * 5 + ch) as i16).wrapping_mul(7));
            }
        }
        let provider = MockProvider {
            spikes: vec![],
            trace,
            n_channels,
            sr: 1000.0,
        };
        let (session, _d) = session_with(provider);

        let a = build_trace_vertices(&session, SampleIndex(0), n_samples as u32, 32);
        let b = build_trace_vertices_cfg(&session, SampleIndex(0), n_samples as u32, 32, TracePreproc::Off);
        assert_eq!(a.len(), b.len());
        for (av, bv) in a.iter().zip(b.iter()) {
            assert_eq!(av.channel, bv.channel);
            assert_eq!(av.pos, bv.pos);
        }
    }

    #[test]
    fn trace_config_with_cmr_zeros_identical_channels() {
        // When every channel reads the same value at each time step, CMR
        // should null the trace entirely. Vertex y values land at 0.
        let n_channels = 3u32;
        let n_samples = 128usize;
        let mut trace = Vec::with_capacity(n_channels as usize * n_samples);
        for s in 0..n_samples {
            for _ch in 0..n_channels as usize {
                trace.push(s as i16);
            }
        }
        let (session, _d) = session_with(MockProvider {
            spikes: vec![],
            trace,
            n_channels,
            sr: 1000.0,
        });

        let cfg = TracePreproc::Cmr;
        let v = build_trace_vertices_cfg(&session, SampleIndex(0), n_samples as u32, 32, cfg);
        for vert in &v {
            assert!(
                vert.pos[1].abs() < 1e-3,
                "expected ~0 after CMR; got {}",
                vert.pos[1]
            );
        }
    }

    #[test]
    fn trace_config_with_hp_filter_attenuates_dc() {
        // Constant signal + HP filter -> tail amplitude << input amplitude.
        let n_channels = 1u32;
        let n_samples = 4096usize;
        let trace: Vec<i16> = vec![1000; n_samples];
        let (session, _d) = session_with(MockProvider {
            spikes: vec![],
            trace,
            n_channels,
            sr: 30_000.0,
        });

        let cfg_off = TracePreproc::Off;
        let cfg_hp = TracePreproc::Hp(300.0);
        let v_off = build_trace_vertices_cfg(&session, SampleIndex(0), n_samples as u32, 64, cfg_off);
        let v_hp = build_trace_vertices_cfg(&session, SampleIndex(0), n_samples as u32, 64, cfg_hp);

        // Last vertex is sampled near the end of the window — settled.
        let off_tail = v_off.last().unwrap().pos[1].abs();
        let hp_tail = v_hp.last().unwrap().pos[1].abs();
        assert!(off_tail > 100.0, "raw tail magnitude {off_tail} too small");
        assert!(hp_tail < 1.0, "HP tail magnitude {hp_tail} not attenuated");
    }

    #[test]
    fn scatter_vertices_empty_when_no_clusters_passed() {
        let provider = MockProvider {
            spikes: vec![vec![SampleIndex(1), SampleIndex(2), SampleIndex(3)]],
            trace: vec![],
            n_channels: 1,
            sr: 1000.0,
        };
        let (session, _d) = session_with(provider);
        assert!(build_scatter_vertices(&session, &[]).is_empty());
    }

    #[test]
    fn scatter_vertices_y_equals_row_index() {
        let provider = MockProvider {
            spikes: vec![vec![SampleIndex(10)], vec![SampleIndex(20)], vec![SampleIndex(30)]],
            trace: vec![],
            n_channels: 1,
            sr: 1000.0,
        };
        let (session, _d) = session_with(provider);
        let v = build_scatter_vertices(&session, &[ClusterId(0), ClusterId(1), ClusterId(2)]);
        assert_eq!(v[0].pos[1], 0.0);
        assert_eq!(v[1].pos[1], 1.0);
        assert_eq!(v[2].pos[1], 2.0);
    }

    #[test]
    fn scatter_vertices_time_in_seconds_via_sample_rate() {
        let provider = MockProvider {
            spikes: vec![vec![SampleIndex(3000)]], // = 0.1 s at 30 kHz
            trace: vec![],
            n_channels: 1,
            sr: 30_000.0,
        };
        let (session, _d) = session_with(provider);
        let v = build_scatter_vertices(&session, &[ClusterId(0)]);
        assert_eq!(v.len(), 1);
        assert!((v[0].pos[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn trace_config_hp_plus_cmr_chains_both_stages() {
        // Two channels, one constant + slow drift + identical across both.
        // CMR cancels the channel-mean; HP wipes any residual DC.
        let n_channels = 2u32;
        let n_samples = 4096usize;
        let mut trace = Vec::with_capacity(n_channels as usize * n_samples);
        for s in 0..n_samples {
            let v = 200 + (s / 10) as i16;
            for _ch in 0..n_channels as usize {
                trace.push(v);
            }
        }
        let (session, _d) = session_with(MockProvider {
            spikes: vec![],
            trace,
            n_channels,
            sr: 30_000.0,
        });
        let cfg = TracePreproc::HpAndCmr(300.0);
        let v = build_trace_vertices_cfg(&session, SampleIndex(0), n_samples as u32, 64, cfg);
        for vert in v.iter().skip(40) {
            assert!(
                vert.pos[1].abs() < 0.5,
                "settled tail after HP+CMR should be ~0; got {}",
                vert.pos[1]
            );
        }
    }
}

//! `sorrel rescue`: matched-filter recovery of spikes a sorter left out of
//! known units, with an empirical-null FDR, a second-best margin, ridge-
//! predicted Kilosort features, and a virtual hybrid-injection recall test.
//!
//! Spike-time convention: the spike time of an accepted spike sits at column
//! `PRE` of the unit's template window, so a detection at window start `w`
//! is spike time `w + PRE`. That keeps whatever alignment the sorter used.
//!
//! The recording is streamed in chunks (never written back): preprocess
//! (CMR + HP, like the trace view), optionally inject hybrid spikes, subtract
//! every accepted spike, then score all units with one matched-filter pass
//! plus `replicates` channel-shifted null passes.

use super::{save_session, seeded, OpenCtx, Seed, SessionVisitor};
use anyhow::{bail, ensure, Result};
use clap::Args;
use sorrel_compute::linalg::Sym;
use sorrel_compute::rescue::{
    apply_margin, apply_refractory, estimate_noise_cov_inv, fdr_threshold, isi_violation_fraction,
    matched_filter_reference, matched_filter_weights, pick_peaks, ridge_fit, ridge_predict,
    ridge_r2, Fdr, RidgeModel,
};
use sorrel_data::command::SpikeBatch;
use sorrel_data::session::ApplyPhyLabel;
use sorrel_data::{CurationCommand, Session};
use sorrel_gpu::{GpuContext, GpuMatchedFilter, MatchedFilterBank};
use sorrel_io::kilosort::PhyLabel;
use sorrel_io::{ClusterId, DataProvider, HasGeometry, SampleIndex};
use sorrel_render::{apply_preproc_cpu, extract_snippet, GpuTracePreproc, TracePreproc};

/// Template window length and the column holding the spike time.
const NT: usize = 61;
const PRE: usize = 20;
/// Accepted spikes sampled per unit for templates, PCs and the ridge fits.
const MAX_TRAIN: usize = 200;
const MIN_SPIKES: usize = 20;
const N_PCS: usize = 4;
/// Detections and accepted-spike subtraction share this scale range.
const SCALE: (f32, f32) = (0.5, 2.0);
#[cfg(test)]
mod tests;

const NOISE_WINDOWS: usize = 64;
const NOISE_LEN: usize = 200;

#[derive(Args, Debug, Clone)]
pub struct RescueArgs {
    /// Target units: good | all | comma-separated cluster ids. Every
    /// non-noise unit competes either way
    #[arg(long, value_name = "SET", default_value = "good")]
    pub units: String,
    /// Target false-discovery rate per unit
    #[arg(long, default_value_t = 0.05, value_parser = parse_q)]
    pub q: f32,
    /// A detection must beat the best competitor's explained energy by this fraction
    #[arg(long, default_value_t = 0.2)]
    pub margin: f32,
    /// Channel-shifted null replicates
    #[arg(long, default_value_t = 3)]
    pub replicates: usize,
    /// Template drift blocks
    #[arg(long, value_name = "S", default_value_t = 60.0)]
    pub block_seconds: f32,
    /// Streaming chunk length (a 384-channel chunk must fit one GPU buffer)
    #[arg(long, value_name = "S", default_value_t = 2.0)]
    pub chunk_seconds: f32,
    /// Channels per unit filter
    #[arg(long, default_value_t = 16)]
    pub channels_per_unit: usize,
    /// Virtual hybrid test: inject FRAC x n_spikes synthetic spikes per
    /// target unit (never written) and report recall. Implies a dry run
    #[arg(long, value_name = "FRAC")]
    pub hybrid_frac: Option<f32>,
    /// Emit JSON instead of a table
    #[arg(long)]
    pub json: bool,
    /// Journal the additions as one undoable batch (AddSpikes + review:rescue tags)
    #[arg(long)]
    pub apply: bool,
    /// With --apply, also write the phy artifacts (same as `apply --save`)
    #[arg(long, requires = "apply")]
    pub save: bool,
    /// RNG seed (null shifts, hybrid injections)
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
    /// Skip the GPU and use the CPU reference (slow; for tests)
    #[arg(long, hide = true)]
    pub cpu: bool,
}

/// `--q` must be a proper false-discovery rate: 0 < q < 1 (NaN fails too).
fn parse_q(s: &str) -> Result<f32, String> {
    let q: f32 = s.parse().map_err(|e| format!("{e}"))?;
    if q > 0.0 && q < 1.0 {
        Ok(q)
    } else {
        Err(format!("--q must satisfy 0 < q < 1, got {s}"))
    }
}

// ---- small helpers ---------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Up to `want` random window starts in `[lo, hi)` whose spike (the start
/// shifted by `PRE`) is at least `gap` from every time in `existing`
/// (ascending) and from every start already in `chosen` (ascending; extended
/// with the new ones). Returns the new starts, sorted.
fn place_injections(
    want: usize,
    (lo, hi): (u64, u64),
    gap: u64,
    existing: &[u64],
    chosen: &mut Vec<u64>,
    rng: &mut Rng,
) -> Vec<u64> {
    let mut v = vec![];
    for _ in 0..want * 20 {
        if v.len() >= want {
            break;
        }
        let ws = lo + rng.below(hi.saturating_sub(lo));
        let t = ws + PRE as u64;
        let i = existing.partition_point(|&e| e + gap <= t);
        let j = chosen.partition_point(|&c| c + gap <= ws);
        if existing.get(i).is_none_or(|&e| e >= t + gap)
            && chosen.get(j).is_none_or(|&c| c >= ws + gap)
        {
            chosen.insert(j, ws);
            v.push(ws);
        }
    }
    v.sort_unstable();
    v
}

fn clamp_scale(x: f32) -> f32 {
    if x.is_nan() {
        1.0
    } else {
        x.clamp(SCALE.0, SCALE.1)
    }
}

enum Kind {
    Amp,
    Pc,
    Tf,
    Attr(String),
}

/// One Kilosort quantity predicted for new spikes: `(kind, width)`.
type Groups = Vec<(Kind, usize)>;

fn groups_of<P: DataProvider>(s: &Session<P>) -> Groups {
    let ci = s.cluster_index();
    let (n_pcs, n_ch) = ci.pc_shape();
    let mut g: Groups = vec![(Kind::Amp, 1)];
    if n_pcs * n_ch > 0 {
        g.push((Kind::Pc, n_pcs * n_ch));
    }
    if ci.has_template_features() {
        // The width is global, so read it off the first spike that has a
        // row rather than assuming cluster 0 is non-empty.
        let w = (0..ci.n_spikes() as u32)
            .find_map(|i| ci.template_feature_row(i))
            .map_or(0, <[f32]>::len);
        if w > 0 {
            g.push((Kind::Tf, w));
        }
    }
    for (name, cols) in ci.spike_attribute_names() {
        g.push((Kind::Attr(name.to_string()), cols));
    }
    g
}

fn y_row<P: DataProvider>(s: &Session<P>, k: &Kind, c: ClusterId, i: usize) -> Option<Vec<f32>> {
    let ci = s.cluster_index();
    let g = *ci.global_indices(c).get(i)?;
    match k {
        Kind::Amp => s.spike_amplitudes(c).get(i).map(|&a| vec![a]),
        Kind::Pc => ci.pc_feature_for(g).map(<[f32]>::to_vec),
        Kind::Tf => ci.template_feature_row(g).map(<[f32]>::to_vec),
        Kind::Attr(n) => ci.spike_attribute(n, g).map(<[f32]>::to_vec),
    }
}

// ---- per-unit model --------------------------------------------------------

struct Ridge {
    model: RidgeModel,
    r2: f32,
}

struct Unit {
    id: ClusterId,
    target: bool,
    primary: u32,
    /// Accepted spike times, ascending.
    times: Vec<u64>,
    chans: Vec<u32>,
    /// `C x NT` channel-major raw-space (preprocessed) templates.
    global: Vec<f32>,
    blocks: Vec<Option<Vec<f32>>>,
    cov_inv: Vec<f32>,
    // Targets only.
    mean: Vec<f32>,
    loads: Vec<Vec<f32>>,
    scales: Vec<f32>,
    models: Vec<Ridge>,
}

impl Unit {
    fn tmpl(&self, b: usize) -> &[f32] {
        self.blocks
            .get(b)
            .and_then(|t| t.as_deref())
            .unwrap_or(&self.global)
    }

    /// The window on this unit's channels, channel-major, from an
    /// interleaved `buf` (`nc` channels) at window start `wl`.
    fn window(&self, buf: &[f32], nc: usize, wl: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.chans.len() * NT);
        for &ch in &self.chans {
            v.extend((0..NT).map(|k| buf[(wl + k) * nc + ch as usize]));
        }
        v
    }

    /// `[scale, PC scores...]`: the Sorrel feature vector the ridge fits use.
    fn feat(&self, v: &[f32], b: usize) -> Vec<f32> {
        let m = self.tmpl(b);
        let mut x = vec![clamp_scale(dot(v, m) / dot(m, m).max(f32::MIN_POSITIVE))];
        let c: Vec<f32> = v.iter().zip(&self.mean).map(|(a, m)| a - m).collect();
        x.extend(self.loads.iter().map(|l| dot(l, &c)));
        x
    }
}

struct Ctx<'a> {
    nc: usize,
    cfg: TracePreproc,
    shanks: &'a [u32],
    c: usize,
    block: usize,
    noise: Vec<f32>,
    noise_n: usize,
}

/// Template, channels, noise precision and (targets) feature models of one unit.
fn train<P: DataProvider>(
    s: &Session<P>,
    cx: &Ctx,
    id: ClusterId,
    target: bool,
    groups: &Groups,
) -> Option<Unit> {
    let times: Vec<u64> = s.spike_times(id).iter().map(|t| t.0).collect();
    let tpl = s.spike_templates(id);
    // ponytail: one template per unit (its primary); a merged unit's other
    // template shapes are neither modelled nor subtracted.
    let primary = s.primary_template(id)?;
    let cand: Vec<usize> = (0..times.len())
        .filter(|&i| tpl.get(i) == Some(&primary))
        .collect();
    if cand.len() < MIN_SPIKES {
        return None;
    }
    let step = cand.len().div_ceil(MAX_TRAIN);
    let (nc, c) = (cx.nc, cx.c);
    let snips: Vec<(usize, Vec<f32>)> = cand
        .iter()
        .step_by(step)
        .filter_map(|&i| {
            let v = extract_snippet(
                &s.provider,
                SampleIndex(times[i]),
                PRE as u32,
                (NT - PRE - 1) as u32,
                cx.cfg,
                cx.shanks,
            )?;
            Some((i, v))
        })
        .collect();
    if snips.len() < MIN_SPIKES {
        return None;
    }
    let n = snips.len();
    let mut mean_full = vec![0f32; NT * nc];
    for (_, v) in &snips {
        for (m, x) in mean_full.iter_mut().zip(v) {
            *m += x / n as f32;
        }
    }
    let energy = |ch: usize| (0..NT).map(|k| mean_full[k * nc + ch].powi(2)).sum::<f32>();
    let mut order: Vec<usize> = (0..nc).collect();
    order.sort_by(|&a, &b| energy(b).total_cmp(&energy(a)));
    let chans: Vec<u32> = order[..c].iter().map(|&x| x as u32).collect();

    // Channel-major windows on the unit's channels, plus the block of each.
    let vs: Vec<Vec<f32>> = snips
        .iter()
        .map(|(_, v)| {
            chans
                .iter()
                .flat_map(|&ch| (0..NT).map(move |k| v[k * nc + ch as usize]))
                .collect()
        })
        .collect();
    let blk: Vec<usize> = snips
        .iter()
        .map(|(i, _)| times[*i] as usize / cx.block)
        .collect();
    let avg = |sel: &[usize]| -> Vec<f32> {
        let mut m = vec![0f32; c * NT];
        for &j in sel {
            for (a, x) in m.iter_mut().zip(&vs[j]) {
                *a += x / sel.len() as f32;
            }
        }
        m
    };
    let all: Vec<usize> = (0..n).collect();
    let global = avg(&all);
    // ponytail: one template per block; blocks with < MIN_SPIKES use the global one.
    let n_blocks = blk.iter().max().map_or(0, |&b| b + 1);
    let blocks = (0..n_blocks)
        .map(|b| {
            let sel: Vec<usize> = (0..n).filter(|&j| blk[j] == b).collect();
            (sel.len() >= MIN_SPIKES).then(|| avg(&sel))
        })
        .collect();

    let mut sub = Vec::with_capacity(cx.noise_n * c);
    for r in 0..cx.noise_n {
        sub.extend(chans.iter().map(|&ch| cx.noise[r * nc + ch as usize]));
    }
    let cov_inv = estimate_noise_cov_inv(&sub, cx.noise_n, c, 0.01)?;

    let mut u = Unit {
        id,
        target,
        primary,
        times,
        chans,
        global,
        blocks,
        cov_inv,
        mean: vec![],
        loads: vec![],
        scales: vec![],
        models: vec![],
    };
    if !target {
        return Some(u);
    }

    // PCA by the dual (n x n Gram) trick: the window is C*NT-dim, n <= 200.
    let d = c * NT;
    u.mean = (0..d)
        .map(|k| vs.iter().map(|v| v[k]).sum::<f32>() / n as f32)
        .collect();
    let xc: Vec<Vec<f32>> = vs
        .iter()
        .map(|v| v.iter().zip(&u.mean).map(|(a, m)| a - m).collect())
        .collect();
    let mut gram = Sym::zeros(n);
    for i in 0..n {
        for j in i..n {
            let g = f64::from(dot(&xc[i], &xc[j]));
            gram.set(i, j, g);
            gram.set(j, i, g);
        }
    }
    let tr = (0..n)
        .map(|i| gram.get(i, i))
        .sum::<f64>()
        .max(f64::MIN_POSITIVE);
    gram.data.iter_mut().for_each(|g| *g /= tr);
    let k = N_PCS.min(n - 1);
    for ev in gram.top_eigenvectors(k) {
        let mut l = vec![0f32; d];
        for (w, row) in ev.iter().zip(&xc) {
            for (a, x) in l.iter_mut().zip(row) {
                *a += *w as f32 * x;
            }
        }
        let nrm = dot(&l, &l).sqrt().max(f32::MIN_POSITIVE);
        l.iter_mut().for_each(|a| *a /= nrm);
        u.loads.push(l);
    }
    let xs: Vec<Vec<f32>> = (0..n).map(|j| u.feat(&vs[j], blk[j])).collect();
    u.scales = xs.iter().map(|x| x[0]).collect();
    let p = 1 + u.loads.len();
    for (kind, w) in groups {
        let (mut x, mut y, mut rows) = (vec![], vec![], 0);
        for (j, (i, _)) in snips.iter().enumerate() {
            if let Some(r) = y_row(s, kind, id, *i).filter(|r| r.len() == *w) {
                x.extend_from_slice(&xs[j]);
                y.extend(r);
                rows += 1;
            }
        }
        if rows < p + 3 {
            return None;
        }
        let model = ridge_fit(&x, &y, rows, p, *w, 1e-4)?;
        let r2s = ridge_r2(&model, &x, &y);
        let r2 = r2s.iter().sum::<f32>() / r2s.len() as f32;
        u.models.push(Ridge { model, r2 });
    }
    Some(u)
}

// ---- results ---------------------------------------------------------------

struct Cand {
    /// Absolute window start.
    t: u64,
    de: f32,
    x: Vec<f32>,
}

pub struct UnitReport {
    pub id: u32,
    pub existing: usize,
    pub added: usize,
    pub theta: Option<f32>,
    pub false_positives: f32,
    pub isi_before: f32,
    pub isi_after: f32,
    pub min_r2: f32,
    pub note: &'static str,
    /// Hybrid recall: `[overall, low, mid, high]` amplitude tercile, and #injections.
    pub recall: Option<([f32; 4], usize)>,
}

pub struct Outcome {
    pub units: Vec<UnitReport>,
    pub cmd: Option<CurationCommand>,
    pub added: usize,
    /// Detected spike times per target unit (read by the tests).
    #[cfg_attr(not(test), allow(dead_code))]
    pub added_times: Vec<(u32, Vec<u64>)>,
    /// Usable units that could not be trained; detections near their spikes are rejected.
    pub unmodelled: Vec<u32>,
}

fn parse_targets<P: DataProvider<Label = PhyLabel>>(
    s: &Session<P>,
    spec: &str,
    usable: &[ClusterId],
) -> Result<Vec<ClusterId>> {
    Ok(match spec {
        "good" => usable
            .iter()
            .copied()
            .filter(|&c| s.label(c) == Some(PhyLabel::Good))
            .collect(),
        "all" => usable.to_vec(),
        ids => {
            let mut v = vec![];
            for t in ids.split(',') {
                let id: u32 = t.trim().parse()?;
                let c = ClusterId::new(id);
                ensure!(
                    usable.contains(&c),
                    "cluster {id} is not a usable non-noise unit"
                );
                v.push(c);
            }
            v
        }
    })
}

/// Run the whole pipeline on an opened (seeded) session.
pub fn run<P: DataProvider<Label = PhyLabel>>(
    s: &Session<P>,
    a: &RescueArgs,
    shanks: &[u32],
) -> Result<Outcome> {
    ensure!(
        s.provider.phy_dir().is_some(),
        "rescue needs a phy/Kilosort directory"
    );
    ensure!(a.replicates >= 1, "--replicates must be at least 1");
    let (nc, sr) = (s.provider.n_channels() as usize, s.provider.sample_rate());
    let n_samples = s.provider.n_samples().get() as usize;
    ensure!(nc > 0 && n_samples > 2000, "no raw recording to scan");
    ensure!(
        s.provider.trace(SampleIndex(0), 8).samples.len() == 8 * nc,
        "raw recording is not readable"
    );
    let ci = s.cluster_index();
    ensure!(
        ci.spike_amplitudes_global().len() == ci.n_spikes()
            && ci.spike_templates_global().len() == ci.n_spikes(),
        "rescue needs amplitudes.npy and spike_templates.npy"
    );
    let hp_done = s.provider.raw_file().is_some_and(|r| r.hp_filtered);
    let cfg = if hp_done {
        TracePreproc::Cmr
    } else {
        TracePreproc::HpAndCmr(TracePreproc::HP_DEFAULT_HZ)
    };
    let settle = cfg.hp().map_or(0, |hz| (1.6 * sr / hz).ceil() as usize);
    let c = a.channels_per_unit.clamp(1, nc);
    let block = ((a.block_seconds * sr) as usize).max(NT * 4);
    let chunk = ((a.chunk_seconds * sr) as usize).clamp(NT * 4, block);
    let refr = (0.0015 * sr).ceil() as u64;
    let mut rng = Rng::new(a.seed);

    let usable: Vec<ClusterId> = (0..s.n_clusters())
        .map(ClusterId::new)
        .filter(|&u| s.label(u) != Some(PhyLabel::Noise) && s.spike_times(u).len() >= MIN_SPIKES)
        .collect();
    let target_ids = parse_targets(s, &a.units, &usable)?;
    ensure!(!target_ids.is_empty(), "no target units");

    // Spike-free preprocessed samples shared by every unit's covariance.
    let all = ci.all_spike_times();
    let mut noise = vec![];
    let mut noise_n = 0;
    for _ in 0..4000 {
        if noise_n >= NOISE_WINDOWS * NOISE_LEN {
            break;
        }
        let ws = 300 + rng.below((n_samples - NOISE_LEN - 600) as u64);
        let i = all.partition_point(|t| t.0 + (NT as u64) < ws);
        if all
            .get(i)
            .is_some_and(|t| t.0 <= ws + (NOISE_LEN + NT) as u64)
        {
            continue;
        }
        if let Some(v) = extract_snippet(
            &s.provider,
            SampleIndex(ws + PRE as u64),
            PRE as u32,
            (NOISE_LEN - PRE - 1) as u32,
            cfg,
            shanks,
        ) {
            noise.extend(v);
            noise_n += NOISE_LEN;
        }
    }
    ensure!(
        noise_n >= 4 * NOISE_LEN,
        "could not find spike-free noise windows"
    );
    let cx = Ctx {
        nc,
        cfg,
        shanks,
        c,
        block,
        noise,
        noise_n,
    };

    let groups = groups_of(s);
    let mut skipped = vec![];
    let units: Vec<Unit> = usable
        .iter()
        .filter_map(|&id| {
            let u = train(s, &cx, id, target_ids.contains(&id), &groups);
            if u.is_none() && target_ids.contains(&id) {
                skipped.push(id.0);
            }
            u
        })
        .collect();
    let n_u = units.len();
    ensure!(
        units.iter().any(|u| u.target),
        "no target unit could be modelled (units {skipped:?})"
    );
    // Usable units that could not be trained (e.g. a merged unit with too few
    // spikes on its primary template) neither compete nor get subtracted, so
    // detections near their spikes are rejected instead and the units are reported.
    let unmodelled: Vec<u32> = usable
        .iter()
        .filter(|&&id| !units.iter().any(|u| u.id == id))
        .map(|id| id.0)
        .collect();
    let mut blocked: Vec<u64> = unmodelled
        .iter()
        .flat_map(|&id| s.spike_times(ClusterId::new(id)).iter().map(|t| t.0))
        .collect();
    blocked.sort_unstable();

    // Hybrid injections: (window start, scale), sorted, >= 2 ms from every
    // existing spike and from every other injection (across targets too).
    let gap = (0.002 * sr) as u64;
    let all_t: Vec<u64> = all.iter().map(|t| t.0).collect();
    let mut chosen: Vec<u64> = vec![];
    let inj: Vec<Vec<(u64, f32)>> = units
        .iter()
        .map(|u| {
            let Some(frac) = a.hybrid_frac.filter(|_| u.target) else {
                return vec![];
            };
            let want = (frac * u.times.len() as f32).round() as usize;
            let (lo, hi) = (settle + 400, n_samples - NT - 400);
            let v = place_injections(
                want,
                (lo as u64, hi as u64),
                gap,
                &all_t,
                &mut chosen,
                &mut rng,
            );
            v.into_iter()
                .map(|ws| {
                    let sc = u.scales.get(rng.below(u.scales.len() as u64) as usize);
                    (ws, clamp_scale(sc.copied().unwrap_or(1.0)))
                })
                .collect()
        })
        .collect();

    let gctx = if a.cpu {
        None
    } else {
        GpuContext::headless()
            .inspect_err(|e| eprintln!("no GPU ({e}); the CPU matched filter is ~100x slower"))
            .ok()
    };
    let mf = gctx.as_ref().map(GpuMatchedFilter::new);
    // A backend that rejects the preprocessing kernels falls back to the CPU.
    let pre = gctx.as_ref().and_then(|g| {
        GpuTracePreproc::new(g)
            .inspect_err(|e| eprintln!("GPU preprocessing unavailable ({e}); using the CPU"))
            .ok()
    });
    let chans_flat: Vec<u32> = units.iter().flat_map(|u| u.chans.iter().copied()).collect();

    let mut real: Vec<Vec<Cand>> = (0..n_u).map(|_| vec![]).collect();
    let mut null: Vec<Vec<f32>> = (0..n_u).map(|_| vec![]).collect();
    let tail = NT + NT / 2;
    let n_blocks = n_samples.div_ceil(block);
    for b in 0..n_blocks {
        let (b_lo, b_hi) = (b * block, ((b + 1) * block).min(n_samples));
        let mut filt = vec![0f32; n_u * c * NT];
        let mut hs = vec![0f32; n_u];
        for (ui, u) in units.iter().enumerate() {
            let (f, h) = matched_filter_weights(u.tmpl(b), &u.cov_inv);
            filt[ui * c * NT..(ui + 1) * c * NT].copy_from_slice(&f);
            hs[ui] = h;
        }
        let inv_h: Vec<f32> = hs
            .iter()
            .map(|&h| if h > 0.0 { 1.0 / h } else { 0.0 })
            .collect();
        let bank = MatchedFilterBank {
            filters: &filt,
            chans: &chans_flat,
            nt: NT as u32,
            c_per_unit: c as u32,
        };
        let mut st = b_lo;
        while st < b_hi {
            let len = chunk.min(b_hi - st);
            let b0 = st.saturating_sub(settle);
            let margin = st - b0;
            let end = (st + len + tail).min(n_samples);
            let slice = s.provider.trace(SampleIndex(b0 as u64), (end - b0) as u32);
            let mut buf = slice.samples.to_f32_centred();
            let n_time = buf.len() / nc;
            st += len;
            if n_time < margin + NT + 1 {
                continue;
            }
            let ok = pre
                .as_ref()
                .is_some_and(|g| g.apply_shanked(&mut buf, nc as u32, cfg, sr, shanks));
            if !ok {
                apply_preproc_cpu(&mut buf, nc, cfg, sr, shanks);
            }
            // ponytail: hybrid spikes are added after preprocessing (HP is
            // linear, CMR nearly so) instead of into the raw samples.
            for (ui, u) in units.iter().enumerate() {
                let i0 = inj[ui].partition_point(|&(w, _)| (w as usize) < b0);
                for &(w, sc) in inj[ui][i0..]
                    .iter()
                    .take_while(|&&(w, _)| w as usize + NT <= b0 + n_time)
                {
                    let wl = w as usize - b0;
                    let m = u.tmpl(b);
                    for (ci_, &ch) in u.chans.iter().enumerate() {
                        for k in 0..NT {
                            buf[(wl + k) * nc + ch as usize] += sc * m[ci_ * NT + k];
                        }
                    }
                }
            }
            // ponytail: accepted spikes are removed with one least-squares
            // scale (clamped to SCALE) times the unit's template; overlaps
            // are subtracted in sequence, not jointly.
            for u in &units {
                let i0 = u.times.partition_point(|&t| (t as usize) < b0 + PRE);
                for &t in u.times[i0..]
                    .iter()
                    .take_while(|&&t| t as usize - PRE + NT <= b0 + n_time)
                {
                    let wl = t as usize - PRE - b0;
                    let m = u.tmpl(b);
                    let sc = clamp_scale(
                        dot(&u.window(&buf, nc, wl), m) / dot(m, m).max(f32::MIN_POSITIVE),
                    );
                    for (ci_, &ch) in u.chans.iter().enumerate() {
                        for k in 0..NT {
                            buf[(wl + k) * nc + ch as usize] -= sc * m[ci_ * NT + k];
                        }
                    }
                }
            }

            let lo_s = margin.saturating_sub(NT / 2);
            let hi_s = (margin + len + NT / 2 + 1).min(n_time - NT + 1);
            for pass in 0..=a.replicates {
                let shifts: Vec<i32> = (0..nc)
                    .map(|_| {
                        if pass == 0 {
                            0
                        } else {
                            1 + rng.below(n_time as u64 - 1) as i32
                        }
                    })
                    .collect();
                let scores = match mf.as_ref().map(|m| m.run(&buf, nc as u32, &bank, &shifts)) {
                    Some(Ok(v)) => v,
                    other => {
                        if let Some(Err(e)) = other {
                            log::warn!("GPU matched filter failed, using the CPU reference: {e}");
                        }
                        matched_filter_reference(&buf, nc, &filt, &chans_flat, NT, c, &shifts)
                    }
                };
                // Best and second-best explained energy over all units.
                let mut best = vec![0f32; n_time];
                let mut second = vec![0f32; n_time];
                let mut best_u = vec![u32::MAX; n_time];
                for (ui, &ih) in inv_h.iter().enumerate() {
                    for (t, &g) in scores[ui * n_time..(ui + 1) * n_time].iter().enumerate() {
                        if g > 0.0 {
                            let e = g * g * ih;
                            if e > best[t] {
                                second[t] = best[t];
                                best[t] = e;
                                best_u[t] = ui as u32;
                            } else if e > second[t] {
                                second[t] = e;
                            }
                        }
                    }
                }
                for (ui, u) in units.iter().enumerate().filter(|(_, u)| u.target) {
                    let sc = &scores[ui * n_time + lo_s..ui * n_time + hi_s];
                    let mut peaks = pick_peaks(sc, hs[ui], NT, SCALE);
                    peaks.iter_mut().for_each(|p| p.t += lo_s as u64);
                    peaks.retain(|p| (margin as u64..(margin + len) as u64).contains(&p.t));
                    let comp = |t: u64| {
                        let t = t as usize;
                        if t >= n_time {
                            0.0
                        } else if best_u[t] == ui as u32 {
                            second[t]
                        } else {
                            best[t]
                        }
                    };
                    let peaks = apply_margin(&peaks, NT, comp, a.margin);
                    let (e0, e1) = (b0 + PRE, b0 + n_time + PRE);
                    let i0 = u.times.partition_point(|&t| (t as usize) < e0);
                    let i1 = u.times.partition_point(|&t| (t as usize) < e1);
                    let (j0, j1) = (
                        blocked.partition_point(|&t| (t as usize) < e0),
                        blocked.partition_point(|&t| (t as usize) < e1),
                    );
                    let mut existing: Vec<u64> = u.times[i0..i1]
                        .iter()
                        .chain(&blocked[j0..j1])
                        .map(|&t| t - PRE as u64 - b0 as u64)
                        .collect();
                    existing.sort_unstable();
                    let peaks = apply_refractory(&peaks, &existing, refr);
                    if pass == 0 {
                        for p in peaks {
                            let wl = p.t as usize;
                            let x = u.feat(&u.window(&buf, nc, wl), b);
                            real[ui].push(Cand {
                                t: b0 as u64 + p.t,
                                de: p.delta_e,
                                x,
                            });
                        }
                    } else {
                        null[ui].extend(peaks.iter().map(|p| p.delta_e));
                    }
                }
            }
        }
    }

    // ---- thresholds, ISI guard, recall, batch --------------------------------
    let tol = (0.0005 * sr) as u64;
    let mut reports = vec![];
    let mut kept_all: Vec<(usize, Vec<Cand>)> = vec![];
    for (ui, u) in units.iter().enumerate().filter(|(_, u)| u.target) {
        let de: Vec<f32> = real[ui].iter().map(|c| c.de).collect();
        let fdr: Option<Fdr> = fdr_threshold(&de, &null[ui], a.replicates, a.q);
        let mut kept: Vec<Cand> = match fdr {
            Some(f) => std::mem::take(&mut real[ui])
                .into_iter()
                .filter(|c| c.de >= f.theta)
                .collect(),
            None => vec![],
        };
        let isi_before = isi_violation_fraction(&u.times, refr);
        let mut merged: Vec<u64> = u.times.clone();
        merged.extend(kept.iter().map(|c| c.t + PRE as u64));
        merged.sort_unstable();
        let isi_after = isi_violation_fraction(&merged, refr);
        let mut note = if fdr.is_none() {
            "no threshold meets q"
        } else {
            ""
        };
        if !kept.is_empty() && isi_after > 2.0 * isi_before.max(0.005) {
            kept.clear();
            note = "dropped: ISI violations more than doubled";
        }
        let hits = |t: u64| kept.iter().any(|c| (c.t + PRE as u64).abs_diff(t) <= tol);
        let recall = (!inj[ui].is_empty()).then(|| {
            let mut sc: Vec<f32> = inj[ui].iter().map(|x| x.1).collect();
            sc.sort_by(f32::total_cmp);
            let n = sc.len();
            let (mut got, mut tot) = ([0usize; 3], [0usize; 3]);
            for &(w, s_) in &inj[ui] {
                let tercile = (sc.partition_point(|&x| x < s_) * 3 / n).min(2);
                tot[tercile] += 1;
                got[tercile] += usize::from(hits(w + PRE as u64));
            }
            let r = |g: usize, t: usize| {
                if t == 0 {
                    f32::NAN
                } else {
                    g as f32 / t as f32
                }
            };
            (
                [
                    r(got.iter().sum(), n),
                    r(got[0], tot[0]),
                    r(got[1], tot[1]),
                    r(got[2], tot[2]),
                ],
                n,
            )
        });
        let r2s: Vec<f32> = u.models.iter().map(|m| m.r2).collect();
        reports.push(UnitReport {
            id: u.id.0,
            existing: u.times.len(),
            added: kept.len(),
            theta: fdr.map(|f| f.theta),
            false_positives: if kept.is_empty() {
                0.0
            } else {
                fdr.map_or(0.0, |f| f.false_positives)
            },
            isi_before,
            isi_after: if kept.is_empty() {
                isi_before
            } else {
                isi_after
            },
            min_r2: r2s.iter().copied().fold(f32::INFINITY, f32::min),
            note,
            recall,
        });
        kept_all.push((ui, kept));
    }
    let added: usize = kept_all.iter().map(|k| k.1.len()).sum();
    let added_times = kept_all
        .iter()
        .map(|(ui, k)| {
            (
                units[*ui].id.0,
                k.iter().map(|c| c.t + PRE as u64).collect(),
            )
        })
        .collect();

    let cmd = (a.apply && a.hybrid_frac.is_none() && added > 0)
        .then(|| build_cmd(s, &units, &groups, &kept_all))
        .transpose()?;
    Ok(Outcome {
        units: reports,
        cmd,
        added,
        added_times,
        unmodelled,
    })
}

/// One `AddSpikes` (all units merged, times sorted) plus a `review:rescue`
/// tag per touched cluster, as a single batch.
fn build_cmd<P: DataProvider<Label = PhyLabel>>(
    s: &Session<P>,
    units: &[Unit],
    groups: &Groups,
    kept: &[(usize, Vec<Cand>)],
) -> Result<CurationCommand> {
    let mut rows: Vec<(u64, usize, usize)> = vec![];
    // preds[unit slot][group] = n x width
    let mut preds: Vec<Vec<Vec<f32>>> = vec![];
    for (slot, (ui, k)) in kept.iter().enumerate() {
        let x: Vec<f32> = k.iter().flat_map(|c| c.x.iter().copied()).collect();
        preds.push(
            units[*ui]
                .models
                .iter()
                .map(|m| ridge_predict(&m.model, &x))
                .collect(),
        );
        rows.extend(
            k.iter()
                .enumerate()
                .map(|(j, c)| (c.t + PRE as u64, slot, j)),
        );
    }
    rows.sort_by_key(|r| r.0);
    let mut b = SpikeBatch {
        attributes: groups
            .iter()
            .filter_map(|(k, w)| match k {
                Kind::Attr(n) => Some((n.clone(), *w as u32, vec![])),
                _ => None,
            })
            .collect(),
        ..Default::default()
    };
    for &(t, slot, j) in &rows {
        let u = &units[kept[slot].0];
        b.times.push(t);
        b.clusters.push(u.id);
        b.templates.push(u.primary);
        let mut attr = 0;
        for (g, (kind, w)) in groups.iter().enumerate() {
            let v = &preds[slot][g][j * w..(j + 1) * w];
            match kind {
                Kind::Amp => b.amplitudes.push(v[0]),
                Kind::Pc => b.pc_features.extend_from_slice(v),
                Kind::Tf => b.template_features.extend_from_slice(v),
                Kind::Attr(_) => {
                    b.attributes[attr].2.extend_from_slice(v);
                    attr += 1;
                }
            }
        }
    }
    let mut cmds = vec![CurationCommand::AddSpikes(b)];
    for (ui, _) in kept.iter().filter(|(_, k)| !k.is_empty()) {
        let id = units[*ui].id;
        let mut tags = s.tags(id).to_vec();
        if !tags.iter().any(|t| t == "review:rescue") {
            tags.push("review:rescue".into());
        }
        cmds.push(CurationCommand::Tag { cluster: id, tags });
    }
    CurationCommand::batch(cmds)
        .ok_or_else(|| anyhow::anyhow!("could not batch the rescue commands"))
}

// ---- CLI -------------------------------------------------------------------

pub struct Rescue(pub RescueArgs);

fn pct(v: f32) -> String {
    if v.is_nan() {
        "-".into()
    } else {
        format!("{:.1}%", 100.0 * v)
    }
}

impl SessionVisitor for Rescue {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        let a = self.0;
        if a.hybrid_frac.is_some() && (a.apply || a.save) {
            log::warn!("--hybrid-frac is a dry run; ignoring --apply/--save");
        }
        let mut session = seeded(session, seed)?;
        let out = run(&session, &a, &ctx.shanks)?;
        if a.json {
            let units: Vec<_> = out
                .units
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "unit": r.id, "existing": r.existing, "added": r.added,
                        "theta": r.theta, "est_false_positives": r.false_positives,
                        "isi_before": r.isi_before, "isi_after": r.isi_after,
                        "min_r2": r.min_r2, "note": r.note,
                        "hybrid": r.recall.map(|(v, n)| serde_json::json!({
                            "injected": n, "recall": v[0],
                            "recall_by_amplitude_tercile": [v[1], v[2], v[3]],
                        })),
                    })
                })
                .collect();
            let doc = serde_json::json!({
                "added": out.added, "applied": out.cmd.is_some(), "units": units,
                "unmodelled_units": out.unmodelled,
            });
            serde_json::to_writer_pretty(std::io::stdout().lock(), &doc)?;
            println!();
        } else {
            println!("unit\texisting\tadded\ttheta\test_fp\tisi_before\tisi_after\tmin_r2\tnote");
            for r in &out.units {
                println!(
                    "{}\t{}\t{}\t{}\t{:.2}\t{}\t{}\t{:.3}\t{}",
                    r.id,
                    r.existing,
                    r.added,
                    r.theta.map_or("-".into(), |t| format!("{t:.1}")),
                    r.false_positives,
                    pct(r.isi_before),
                    pct(r.isi_after),
                    r.min_r2,
                    r.note
                );
            }
            if out.units.iter().any(|r| r.recall.is_some()) {
                println!("hybrid recall (unit, injected, overall, low/mid/high amplitude tercile)");
                for (r, (v, n)) in out.units.iter().filter_map(|r| r.recall.map(|x| (r, x))) {
                    println!(
                        "{}\t{}\t{}\t{}/{}/{}",
                        r.id,
                        n,
                        pct(v[0]),
                        pct(v[1]),
                        pct(v[2]),
                        pct(v[3])
                    );
                }
            }
            if !out.unmodelled.is_empty() {
                println!(
                    "{} units could not be modelled (detections near their spikes are rejected): {:?}",
                    out.unmodelled.len(),
                    out.unmodelled
                );
            }
            println!(
                "{} spikes found{}",
                out.added,
                if out.cmd.is_some() { "" } else { " (dry run)" }
            );
        }
        if let Some(cmd) = out.cmd {
            session.dispatch(cmd)?;
            if a.save {
                save_session(&mut session, &ctx)?;
            }
        } else if a.save && !a.json && a.hybrid_frac.is_none() {
            bail!("--save needs --apply and at least one added spike");
        }
        Ok(())
    }
}

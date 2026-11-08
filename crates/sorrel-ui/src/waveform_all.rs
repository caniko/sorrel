//! Background extraction of *every* spike for the waveform pane.
//!
//! The legacy [`crate::waveform_view::WaveformCache::ensure`] path is capped
//! at `n_spikes` and stays synchronous because Probe/Trace only need a peak
//! channel and a small overlay. The waveform pane's default (`AllSpikes`)
//! needs the whole cluster: the mean must be over every valid spike, and the
//! raw traces must not silently be a 100-spike sample.
//!
//! [`run_all`] streams the extraction in prefetched batches. Pass 1
//! accumulates the mean and the true min/max over every valid spike and is
//! published as soon as it finishes (exact full-set mean, no raster yet), so
//! `M` mean-only never waits for the density images. Pass 2 rasterizes
//! **every** valid spike into a small fixed-size density image (so a million
//! spike cluster does not create a million egui polylines). Counts make
//! failures explicit: `total`, `valid`, `invalid` (edge/missing), `drawn`, and
//! `raster_failed` (never silently replaced by a sample).
//!
//! Everything runs on a `crate::jobs::Job`-style worker: the UI never
//! extracts or tessellates all spikes. Cancellation is checked per spike and
//! per batch; a dropped [`AllJob`] or [`AllCache::cancel`] stops a curation
//! write from waiting on a stale read lock.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider, SampleIndex, TraceSamples};
use sorrel_render::{apply_preproc_cpu, snippet_window, TracePreproc};

use crate::jobs::JobCtx;

/// Fixed density-image size per cluster/channel. Content is normalized to
/// [`AllResult::raster_lo`]/[`AllResult::raster_hi`], so the image is reused
/// while the pane is resized; the painter only scales it.
pub(crate) const RASTER_SIZE: [usize; 2] = [512, 256];

/// Windows prefetched per provider round trip.
const PREFETCH_BATCH: usize = 256;

/// Below this many spikes the mean pass stays on the worker thread; the
/// rayon fan-out costs more than it saves for small clusters and tests.
const PARALLEL_EXTRACT_MIN: usize = 4_096;

/// Spikes per parallel mean task. Chunk results are merged in index order, so
/// the published mean/extrema are deterministic for a given input.
const EXTRACT_CHUNK: usize = 4_096;

/// Below this many valid spikes the raster pass stays on the worker thread;
/// the rayon fan-out costs more than it saves for small clusters and tests.
const PARALLEL_RASTER_MIN: usize = 4_096;

/// Spikes per parallel raster task. Each task owns one 512×256×channels
/// density image; merged images are exactly equal because saturating u8 adds
/// are commutative.
const RASTER_CHUNK: usize = 4_096;

/// Wall-clock throttle for progress repaints from the worker.
const REPAINT_EVERY: Duration = Duration::from_millis(50);

const STAGE_EXTRACT: u8 = 0;
pub(crate) const STAGE_RASTER: u8 = 1;

/// Everything an all-spike request is keyed on. The density range is derived
/// from the data itself (see [`run_all`]), so it is deliberately not part of
/// the key: zoom/scale changes never restart extraction.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AllKey {
    pub clusters: Vec<ClusterId>,
    pub pre: u32,
    pub post: u32,
    pub cfg: TracePreproc,
    pub shanks: Vec<u32>,
    pub render_channels: Vec<usize>,
    pub n_channels: u32,
    pub n_samples: u64,
    pub sample_rate: f32,
    pub version: u64,
}

/// Atomic progress shared by the worker and the UI. No locks: the pane only
/// does relaxed loads once a frame.
#[derive(Debug)]
pub(crate) struct AllProgress {
    total: AtomicUsize,
    done: AtomicUsize,
    valid: AtomicUsize,
    invalid: AtomicUsize,
    out_of_range: AtomicUsize,
    missing: AtomicUsize,
    drawn: AtomicUsize,
    raster_done: AtomicUsize,
    raster_total: AtomicUsize,
    stage: AtomicU8,
    clusters_done: AtomicUsize,
    n_clusters: AtomicUsize,
}

impl AllProgress {
    pub fn new(total: usize, n_clusters: usize) -> Self {
        Self {
            total: AtomicUsize::new(total),
            done: AtomicUsize::new(0),
            valid: AtomicUsize::new(0),
            invalid: AtomicUsize::new(0),
            out_of_range: AtomicUsize::new(0),
            missing: AtomicUsize::new(0),
            drawn: AtomicUsize::new(0),
            raster_done: AtomicUsize::new(0),
            raster_total: AtomicUsize::new(0),
            stage: AtomicU8::new(STAGE_EXTRACT),
            clusters_done: AtomicUsize::new(0),
            n_clusters: AtomicUsize::new(n_clusters),
        }
    }

    pub fn snapshot(&self) -> ProgressSnapshot {
        ProgressSnapshot {
            total: self.total.load(Ordering::Relaxed),
            done: self.done.load(Ordering::Relaxed),
            valid: self.valid.load(Ordering::Relaxed),
            invalid: self.invalid.load(Ordering::Relaxed),
            out_of_range: self.out_of_range.load(Ordering::Relaxed),
            missing: self.missing.load(Ordering::Relaxed),
            drawn: self.drawn.load(Ordering::Relaxed),
            raster_done: self.raster_done.load(Ordering::Relaxed),
            raster_total: self.raster_total.load(Ordering::Relaxed),
            stage: self.stage.load(Ordering::Relaxed),
            clusters_done: self.clusters_done.load(Ordering::Relaxed),
            n_clusters: self.n_clusters.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ProgressSnapshot {
    pub total: usize,
    pub done: usize,
    pub valid: usize,
    pub invalid: usize,
    pub out_of_range: usize,
    pub missing: usize,
    pub drawn: usize,
    pub raster_done: usize,
    pub raster_total: usize,
    pub stage: u8,
    pub clusters_done: usize,
    pub n_clusters: usize,
}

/// Per-cluster output. `mean` and `min`/`max` cover every channel; `raster`
/// has one density image per entry of [`AllResult::render_channels`].
#[derive(Clone, Debug)]
pub(crate) struct AllCluster {
    pub total: usize,
    pub valid: usize,
    pub invalid: usize,
    pub out_of_range: usize,
    pub missing: usize,
    pub drawn: usize,
    pub mean: Vec<Vec<f32>>,
    pub min: Vec<f32>,
    pub max: Vec<f32>,
    pub raster: Vec<Vec<u8>>,
}

impl AllCluster {
    fn empty(nc: usize, snippet_len: usize) -> Self {
        Self {
            total: 0,
            valid: 0,
            invalid: 0,
            out_of_range: 0,
            missing: 0,
            drawn: 0,
            mean: vec![vec![0.0; snippet_len]; nc],
            min: vec![0.0; nc],
            max: vec![0.0; nc],
            // Allocated only for non-empty clusters that will rasterize
            // traces; an empty cluster costs no density memory.
            raster: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AllResult {
    pub clusters: Vec<AllCluster>,
    pub render_channels: Vec<usize>,
    pub raster_lo: f32,
    pub raster_hi: f32,
    pub total: usize,
    pub valid: usize,
    pub invalid: usize,
    pub out_of_range: usize,
    pub missing: usize,
    pub drawn: usize,
    /// Valid spikes that could not be re-extracted for the density image.
    /// Never silently replaced by a sample; the pane reports this.
    pub raster_failed: usize,
    /// True only once pass 2 has rasterized the complete valid set. A
    /// provisional result carries the exact all-spike mean and extrema but no
    /// density images yet, so the pane can show the full mean immediately.
    pub raster_complete: bool,
    /// Some trace bytes were still in flight when extraction ended. The
    /// result is usable (counts are honest) and the cache retries shortly.
    pub incomplete: bool,
    pub cancelled: bool,
    pub elapsed_ms: u64,
    /// Set by [`AllCache`] when published; texture keys use it.
    pub generation: u64,
}

impl AllResult {
    pub fn cluster(&self, slot: usize) -> Option<&AllCluster> {
        self.clusters.get(slot)
    }
}

/// One message from the streaming worker: the exact mean as soon as pass 1
/// ends, then the completed result with density images.
enum AllUpdate {
    Provisional(Box<AllResult>),
    Final(Box<AllResult>),
}

/// Last result plus the one in-flight job. Kept next to `WaveformCache` so
/// the legacy sampled `ensure` cannot cancel or restart it on every paint.
#[derive(Default)]
pub(crate) struct AllCache {
    requested: Option<AllKey>,
    data: Option<(AllKey, AllResult)>,
    job: Option<AllJob>,
    progress: Option<Arc<AllProgress>>,
    generation: u64,
    started: Option<Instant>,
    retry_at: Option<Instant>,
}

/// A streaming worker. Dropping it sets the cancel flag; the worker polls it
/// per spike and releases the session read lock promptly, so a curation write
/// never waits for a whole cluster.
struct AllJob {
    key: AllKey,
    rx: Receiver<AllUpdate>,
    cancel: Arc<AtomicBool>,
}

impl Drop for AllJob {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl AllCache {
    pub fn requested(&self) -> Option<&AllKey> {
        self.requested.as_ref()
    }

    /// The current result (provisional exact mean or the finished result),
    /// never one computed for another selection/key.
    pub fn current(&self) -> Option<&AllResult> {
        let requested = self.requested.as_ref()?;
        self.data
            .as_ref()
            .filter(|(key, _)| key == requested)
            .map(|(_, data)| data)
    }

    pub fn is_loading(&self) -> bool {
        self.job.is_some()
    }

    pub fn started(&self) -> Option<Instant> {
        self.started
    }

    pub fn progress(&self) -> Option<ProgressSnapshot> {
        self.progress.as_ref().map(|progress| progress.snapshot())
    }

    /// Stop the worker without discarding the last complete result (which is
    /// key-checked, so it can never masquerade as a newer selection).
    pub fn cancel(&mut self) {
        self.job = None;
        self.progress = None;
    }

    /// Stop and forget everything. Used when preprocessing/channel mapping
    /// changes, where the old result is no longer just stale but unfaithful.
    pub fn invalidate(&mut self) {
        self.cancel();
        self.requested = None;
        self.data = None;
        self.retry_at = None;
        self.started = None;
    }

    fn begin(&mut self, key: &AllKey) {
        if self.requested.as_ref() == Some(key) {
            return;
        }
        self.job = None;
        self.progress = None;
        self.started = None;
        self.retry_at = None;
        self.requested = Some(key.clone());
        if self.data.as_ref().is_some_and(|(old, _)| old != key) {
            self.data = None;
        }
    }

    fn store(&mut self, key: &AllKey, mut result: AllResult) {
        self.generation = self.generation.wrapping_add(1);
        result.generation = self.generation;
        self.data = Some((key.clone(), result));
    }

    fn poll(&mut self, key: &AllKey) {
        let current_job = self.job.as_ref().is_some_and(|job| job.key == *key);
        if !current_job {
            self.job = None;
            self.progress = None;
        }
        if self.job.is_none() {
            return;
        }
        let mut updates = Vec::new();
        let mut disconnected = false;
        {
            let job = self.job.as_ref().expect("checked above");
            loop {
                match job.rx.try_recv() {
                    Ok(update) => updates.push(update),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        let mut finished = false;
        for update in updates {
            match update {
                AllUpdate::Provisional(result) => {
                    self.store(key, *result);
                }
                AllUpdate::Final(result) => {
                    finished = true;
                    if !result.cancelled {
                        self.retry_at = result
                            .incomplete
                            .then(|| Instant::now() + Duration::from_millis(500));
                        self.store(key, *result);
                    }
                }
            }
        }
        if finished || disconnected {
            self.job = None;
            self.progress = None;
        }
    }

    fn spawn<P: DataProvider>(&mut self, key: &AllKey, jobs: &JobCtx<Session<P>>, total: usize)
    where
        Session<P>: Send + Sync + 'static,
    {
        let progress = Arc::new(AllProgress::new(total, key.clusters.len()));
        let worker_progress = Arc::clone(&progress);
        let worker_key = key.clone();
        let egui = jobs.egui.clone();
        let session = Arc::clone(&jobs.session);
        let repaint = jobs.repaint();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);
        let (tx, rx) = channel();
        self.started = Some(Instant::now());
        self.retry_at = None;
        self.progress = Some(Arc::clone(&progress));
        std::thread::Builder::new()
            .name("sorrel-waveform-all".into())
            .spawn(move || {
                let guard = crate::jobs::read(&session);
                if worker_cancel.load(Ordering::Relaxed) {
                    return;
                }
                let send = |update: AllUpdate| tx.send(update).is_ok();
                let result = run_all_emit(
                    &guard,
                    &worker_key,
                    &worker_progress,
                    &worker_cancel,
                    Some(&egui),
                    &mut |provisional| send(AllUpdate::Provisional(Box::new(provisional))),
                );
                drop(guard);
                if !worker_cancel.load(Ordering::Relaxed) && !result.cancelled {
                    let _ = tx.send(AllUpdate::Final(Box::new(result)));
                    repaint();
                }
            })
            .expect("spawn waveform worker thread");
        self.job = Some(AllJob {
            key: key.clone(),
            rx,
            cancel,
        });
    }

    /// Ensure a result for `key`, starting/reusing the worker as needed.
    /// `total` is the session's spike count for the selection, for progress.
    pub fn ensure<P: DataProvider>(&mut self, key: &AllKey, jobs: &JobCtx<Session<P>>, total: usize)
    where
        Session<P>: Send + Sync + 'static,
    {
        self.begin(key);
        self.poll(key);
        let retry_due = self.retry_at.is_some_and(|at| Instant::now() >= at);
        let needs_job = match self.current() {
            // A provisional result still needs the raster pass.
            Some(result) if result.raster_complete => result.incomplete && retry_due,
            Some(_) => true,
            None => true,
        };
        if needs_job && self.job.is_none() {
            self.spawn::<P>(key, jobs, total);
        }
    }
}

/// Reusable extraction and median scratch. Both passes process the same
/// channel layout, so one allocation per worker instead of two `Vec`s per
/// spike (millions of snippets on a real cluster).
struct SnipScratch {
    /// `[sample_in_snippet * nc + channel]`, baseline-subtracted.
    buf: Vec<f32>,
    /// One channel's samples, reused by the median selection.
    median: Vec<f32>,
}

impl SnipScratch {
    fn new(nc: usize, snippet_len: usize) -> Self {
        Self {
            buf: Vec::with_capacity(nc * snippet_len),
            median: Vec::with_capacity(snippet_len),
        }
    }

    /// Same window and dtype centring as `sorrel_render::extract_snippet`,
    /// but into `self.buf` (only the post-margin samples when no preproc is
    /// needed). Returns false when the window is short/not resident.
    #[allow(clippy::too_many_arguments)]
    fn extract<P: DataProvider>(
        &mut self,
        provider: &P,
        t: SampleIndex,
        pre: u32,
        post: u32,
        cfg: TracePreproc,
        shanks: &[u32],
        nc: usize,
        snippet_len: usize,
    ) -> bool {
        let sr = provider.sample_rate();
        let (start, len, margin) = snippet_window(sr, t, pre, post, cfg);
        let slice = provider.trace(start, len);
        if nc == 0 || slice.samples.len() != len as usize * nc {
            self.buf.clear();
            return false;
        }
        let margin_values = margin as usize * nc;
        self.buf.clear();
        let borrow = slice.samples.borrow();
        if cfg.is_off() {
            // The settles-trimmed part is never used without preproc.
            match borrow {
                TraceSamples::I16(s) => self
                    .buf
                    .extend(s[margin_values..].iter().map(|&v| v as f32)),
                TraceSamples::U16(s) => self.buf.extend(
                    s[margin_values..]
                        .iter()
                        .map(|&v| v as f32 - i16::MAX as f32),
                ),
                TraceSamples::I32(s) => self
                    .buf
                    .extend(s[margin_values..].iter().map(|&v| v as f32)),
                TraceSamples::F32(s) => self.buf.extend_from_slice(&s[margin_values..]),
                TraceSamples::OwnedI16(v) => self
                    .buf
                    .extend(v[margin_values..].iter().map(|&v| v as f32)),
            }
        } else {
            match borrow {
                TraceSamples::I16(s) => self.buf.extend(s.iter().map(|&v| v as f32)),
                TraceSamples::U16(s) => self
                    .buf
                    .extend(s.iter().map(|&v| v as f32 - i16::MAX as f32)),
                TraceSamples::I32(s) => self.buf.extend(s.iter().map(|&v| v as f32)),
                TraceSamples::F32(s) => self.buf.extend_from_slice(s),
                TraceSamples::OwnedI16(v) => self.buf.extend(v.iter().map(|&v| v as f32)),
            }
            apply_preproc_cpu(&mut self.buf, nc, cfg, sr, shanks);
            if margin_values > 0 && margin_values < self.buf.len() {
                self.buf.copy_within(margin_values.., 0);
                let keep = self.buf.len() - margin_values;
                self.buf.truncate(keep);
            }
        }
        self.buf.len() == snippet_len * nc
    }
}

/// Subtract each display channel's upper median. Exact same value as
/// [`subtract_median_baseline`] for the channels the pane draws; channels
/// that are never rendered are deliberately not touched, which keeps a
/// 384-channel probe from paying for 384 median selections per spike. The
/// finite check in the caller still covers every channel, so validity is
/// unchanged.
fn subtract_render_medians(
    buf: &mut [f32],
    nc: usize,
    snippet_len: usize,
    render_channels: &[usize],
    scratch: &mut Vec<f32>,
) {
    if nc == 0 || snippet_len == 0 {
        return;
    }
    for &ch in render_channels {
        if ch >= nc {
            continue;
        }
        scratch.clear();
        scratch.extend((0..snippet_len).map(|t| buf[t * nc + ch]));
        let n = scratch.len();
        if n == 0 {
            continue;
        }
        let (_, med, _) = scratch.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
        let med = *med;
        for t in 0..snippet_len {
            buf[t * nc + ch] -= med;
        }
    }
}

/// Partial pass-1 result for one contiguous spike chunk. Merged in chunk
/// order for a deterministic mean.
#[derive(Default)]
struct Pass1Chunk {
    sums: Vec<Vec<f64>>,
    min: Vec<f32>,
    max: Vec<f32>,
    valid: usize,
    invalid: usize,
    out_of_range: usize,
    missing: usize,
    cancelled: bool,
}

/// Extract a chunk of spikes for the full-set mean and true extrema. Fills
/// the matching slice of the valid bitmask. Used by both the sequential and
/// parallel path so counting/validity cannot drift.
#[allow(clippy::too_many_arguments)]
fn extract_pass1_chunk<P: DataProvider>(
    session: &Session<P>,
    key: &AllKey,
    times: &[SampleIndex],
    valid_out: &mut [bool],
    nc: usize,
    snippet_len: usize,
    sr: f32,
    n_samples: u64,
    progress: &AllProgress,
    cancel: &AtomicBool,
) -> Pass1Chunk {
    let render_channels = &key.render_channels;
    let mut out = Pass1Chunk {
        sums: vec![vec![0.0f64; snippet_len]; nc],
        min: vec![f32::INFINITY; nc],
        max: vec![f32::NEG_INFINITY; nc],
        ..Default::default()
    };
    let mut scratch = SnipScratch::new(nc, snippet_len);
    let mut local = 0usize;
    for batch in times.chunks(PREFETCH_BATCH) {
        if cancel.load(Ordering::Relaxed) {
            out.cancelled = true;
            return out;
        }
        let windows: Vec<(SampleIndex, u32)> = batch
            .iter()
            .map(|&t| {
                let (start, len, _) = snippet_window(sr, t, key.pre, key.post, key.cfg);
                (start, len)
            })
            .collect();
        session.provider.prefetch(&windows);
        for &t in batch {
            if cancel.load(Ordering::Relaxed) {
                out.cancelled = true;
                return out;
            }
            progress.done.fetch_add(1, Ordering::Relaxed);
            let (start, len, _) = snippet_window(sr, t, key.pre, key.post, key.cfg);
            if t.0 < key.pre as u64 || start.0.saturating_add(len as u64) > n_samples {
                out.invalid += 1;
                out.out_of_range += 1;
                progress.invalid.fetch_add(1, Ordering::Relaxed);
                progress.out_of_range.fetch_add(1, Ordering::Relaxed);
                local += 1;
                continue;
            }
            if !scratch.extract(
                &session.provider,
                t,
                key.pre,
                key.post,
                key.cfg,
                &key.shanks,
                nc,
                snippet_len,
            ) {
                out.invalid += 1;
                out.missing += 1;
                progress.invalid.fetch_add(1, Ordering::Relaxed);
                progress.missing.fetch_add(1, Ordering::Relaxed);
                local += 1;
                continue;
            }
            subtract_render_medians(
                &mut scratch.buf,
                nc,
                snippet_len,
                render_channels,
                &mut scratch.median,
            );
            if !scratch.buf.iter().all(|v| v.is_finite()) {
                out.invalid += 1;
                out.missing += 1;
                progress.invalid.fetch_add(1, Ordering::Relaxed);
                progress.missing.fetch_add(1, Ordering::Relaxed);
                local += 1;
                continue;
            }
            valid_out[local] = true;
            out.valid += 1;
            progress.valid.fetch_add(1, Ordering::Relaxed);

            // Only the shared display channels need a mean/min/max: the
            // channel set is part of the work key, so changing it starts a
            // fresh request instead of paying for all 384 probe channels on
            // every spike.
            for &ch in render_channels {
                if ch >= nc {
                    continue;
                }
                let (mut ch_min, mut ch_max) = (out.min[ch], out.max[ch]);
                let sum_row = &mut out.sums[ch];
                for (sample, &value) in scratch.buf[ch..]
                    .iter()
                    .step_by(nc)
                    .take(snippet_len)
                    .enumerate()
                {
                    sum_row[sample] += value as f64;
                    if value < ch_min {
                        ch_min = value;
                    }
                    if value > ch_max {
                        ch_max = value;
                    }
                }
                out.min[ch] = ch_min;
                out.max[ch] = ch_max;
            }
            local += 1;
        }
    }
    out
}

/// Merge chunk results in their original order.
fn merge_pass1(mut chunks: Vec<Pass1Chunk>, nc: usize, snippet_len: usize) -> Pass1Chunk {
    let mut merged = Pass1Chunk {
        sums: vec![vec![0.0f64; snippet_len]; nc],
        min: vec![f32::INFINITY; nc],
        max: vec![f32::NEG_INFINITY; nc],
        ..Default::default()
    };
    for chunk in chunks.drain(..) {
        for (row, chunk_row) in merged.sums.iter_mut().zip(&chunk.sums) {
            for (sum, value) in row.iter_mut().zip(chunk_row) {
                *sum += value;
            }
        }
        for (current, value) in merged.min.iter_mut().zip(&chunk.min) {
            *current = current.min(*value);
        }
        for (current, value) in merged.max.iter_mut().zip(&chunk.max) {
            *current = current.max(*value);
        }
        merged.valid += chunk.valid;
        merged.invalid += chunk.invalid;
        merged.out_of_range += chunk.out_of_range;
        merged.missing += chunk.missing;
        merged.cancelled |= chunk.cancelled;
    }
    merged
}

#[cfg(test)]
pub(crate) fn run_all<P: DataProvider>(
    session: &Session<P>,
    key: &AllKey,
    progress: &AllProgress,
    cancel: &AtomicBool,
    egui: Option<&egui::Context>,
) -> AllResult {
    run_all_emit(session, key, progress, cancel, egui, &mut |_| true)
}

/// [`run_all`] with a synchronous progress publication hook. The exact
/// all-spike mean is emitted after pass 1; `emit` returns `false` when the
/// receiver is gone, which stops the worker like a cancellation.
fn run_all_emit<P: DataProvider>(
    session: &Session<P>,
    key: &AllKey,
    progress: &AllProgress,
    cancel: &AtomicBool,
    egui: Option<&egui::Context>,
    emit: &mut dyn FnMut(AllResult) -> bool,
) -> AllResult {
    let started = Instant::now();
    let nc = key.n_channels as usize;
    let snippet_len = (key.pre + key.post + 1) as usize;
    let render_channels = &key.render_channels;
    let n_samples = session.provider.n_samples().0;
    let sr = session.provider.sample_rate();
    let mut last_repaint = Instant::now();

    let total: usize = key
        .clusters
        .iter()
        .map(|&cluster| session.spike_times(cluster).len())
        .sum();
    progress.total.store(total, Ordering::Relaxed);

    let mut out = Vec::with_capacity(key.clusters.len());
    // A compact bit per spike keeps pass 2 on exactly the same valid set as
    // the mean, even if remote trace availability changes between passes.
    let mut valid_spikes = Vec::with_capacity(key.clusters.len());
    let (mut all_total, mut all_valid, mut all_invalid) = (0usize, 0usize, 0usize);
    let (mut all_edge, mut all_missing) = (0usize, 0usize);

    // Pass 1: every valid spike for the mean and the true amplitude range.
    for &cluster in &key.clusters {
        let times = session.spike_times(cluster);
        let mut counts = AllCluster::empty(nc, snippet_len);
        counts.total = times.len();
        valid_spikes.push(vec![false; times.len()]);
        let valid_mask = valid_spikes.last_mut().expect("just added");
        all_total += times.len();
        if times.is_empty() || nc == 0 || snippet_len < 3 {
            out.push(counts);
            progress.clusters_done.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        let stats = if times.len() >= PARALLEL_EXTRACT_MIN {
            use rayon::prelude::*;
            let chunks: Vec<Pass1Chunk> = times
                .par_chunks(EXTRACT_CHUNK)
                .zip(valid_mask.par_chunks_mut(EXTRACT_CHUNK))
                .map(|(chunk, mask)| {
                    extract_pass1_chunk(
                        session,
                        key,
                        chunk,
                        mask,
                        nc,
                        snippet_len,
                        sr,
                        n_samples,
                        progress,
                        cancel,
                    )
                })
                .collect();
            merge_pass1(chunks, nc, snippet_len)
        } else {
            extract_pass1_chunk(
                session,
                key,
                times,
                valid_mask,
                nc,
                snippet_len,
                sr,
                n_samples,
                progress,
                cancel,
            )
        };
        if stats.cancelled || cancel.load(Ordering::Relaxed) {
            return cancelled(key, started, all_total);
        }
        counts.valid = stats.valid;
        counts.invalid = stats.invalid;
        counts.out_of_range = stats.out_of_range;
        counts.missing = stats.missing;
        let divisor = counts.valid.max(1) as f64;
        counts.mean = stats
            .sums
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|value| (value / divisor) as f32)
                    .collect()
            })
            .collect();
        let mut min = stats.min;
        let mut max = stats.max;
        for (low, high) in min.iter_mut().zip(max.iter_mut()) {
            if !low.is_finite() {
                *low = 0.0;
            }
            if !high.is_finite() {
                *high = 0.0;
            }
        }
        counts.min = min;
        counts.max = max;
        all_valid += counts.valid;
        all_invalid += counts.invalid;
        all_edge += counts.out_of_range;
        all_missing += counts.missing;
        out.push(counts);
        progress.clusters_done.fetch_add(1, Ordering::Relaxed);
    }

    if cancel.load(Ordering::Relaxed) {
        return cancelled(key, started, all_total);
    }

    // The density image is normalized to the true streamed min/max (with the
    // same 8% padding as the view), so no valid spike is clipped just because
    // the legacy 100-spike preview missed its extremes.
    let (raster_lo, raster_hi) = data_raster_range(&out, render_channels);
    progress.stage.store(STAGE_RASTER, Ordering::Relaxed);
    progress.raster_total.store(all_total, Ordering::Relaxed);

    // Publish the exact full-set mean before touching the raster pass. This
    // is the whole pass-1 result (means, extrema and honest counts); only the
    // density images are missing.
    let provisional = AllResult {
        clusters: out.clone(),
        render_channels: render_channels.clone(),
        raster_lo,
        raster_hi,
        total: all_total,
        valid: all_valid,
        invalid: all_invalid,
        out_of_range: all_edge,
        missing: all_missing,
        drawn: 0,
        raster_failed: 0,
        raster_complete: false,
        incomplete: false,
        cancelled: false,
        elapsed_ms: started.elapsed().as_millis() as u64,
        generation: 0,
    };
    if !emit(provisional) {
        return cancelled(key, started, all_total);
    }
    if cancel.load(Ordering::Relaxed) {
        return cancelled(key, started, all_total);
    }

    // Pass 2: rasterize EVERY valid spike. No draw budget or sample fallback:
    // `drawn` must equal `valid` for a complete result. If a spike cannot be
    // re-extracted, it is reported as `raster_failed` (and retried when trace
    // bytes were still in flight) rather than silently dropped.
    let mut all_drawn = 0usize;
    let mut raster_failed = 0usize;
    for (slot, &cluster) in key.clusters.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return cancelled(key, started, all_total);
        }
        let times = session.spike_times(cluster);
        if times.is_empty() || out[slot].valid == 0 || nc == 0 || snippet_len < 3 {
            progress
                .raster_done
                .fetch_add(times.len(), Ordering::Relaxed);
            continue;
        }
        let mut raster = new_raster(render_channels.len());
        let drawn = rasterize_slot(
            session,
            key,
            times,
            &valid_spikes[slot],
            raster_lo,
            raster_hi,
            progress,
            cancel,
            &mut raster,
        );
        let expected = out[slot].valid;
        out[slot].drawn = drawn;
        out[slot].raster = raster;
        all_drawn += drawn;
        raster_failed += expected.saturating_sub(drawn);
        progress.drawn.fetch_add(drawn, Ordering::Relaxed);
        if egui.is_some_and(|_| last_repaint.elapsed() >= REPAINT_EVERY) {
            last_repaint = Instant::now();
            if let Some(ctx) = egui {
                ctx.request_repaint();
            }
        }
    }

    if cancel.load(Ordering::Relaxed) {
        return cancelled(key, started, all_total);
    }
    let incomplete =
        (all_missing > 0 || raster_failed > 0) && session.provider.pending_fetches() > 0;
    AllResult {
        clusters: out,
        render_channels: render_channels.clone(),
        raster_lo,
        raster_hi,
        total: all_total,
        valid: all_valid,
        invalid: all_invalid,
        out_of_range: all_edge,
        missing: all_missing,
        drawn: all_drawn,
        raster_failed,
        raster_complete: true,
        incomplete,
        cancelled: false,
        elapsed_ms: started.elapsed().as_millis() as u64,
        generation: 0,
    }
}

/// Same padding as the view's [`super::pad_range`]: include zero, then 8%.
fn data_raster_range(clusters: &[AllCluster], render_channels: &[usize]) -> (f32, f32) {
    let (mut lo, mut hi) = (0.0f32, 0.0f32);
    for cluster in clusters {
        for &ch in render_channels {
            if let (Some(&a), Some(&b)) = (cluster.min.get(ch), cluster.max.get(ch)) {
                lo = lo.min(a);
                hi = hi.max(b);
            }
        }
    }
    if !lo.is_finite() || !hi.is_finite() || hi - lo < 1e-3 {
        return (-1.0, 1.0);
    }
    let pad = (hi - lo) * 0.08;
    (lo - pad, hi + pad)
}

fn cancelled(key: &AllKey, started: Instant, total: usize) -> AllResult {
    AllResult {
        clusters: Vec::new(),
        render_channels: key.render_channels.clone(),
        raster_lo: -1.0,
        raster_hi: 1.0,
        total,
        valid: 0,
        invalid: 0,
        out_of_range: 0,
        missing: 0,
        drawn: 0,
        raster_failed: 0,
        raster_complete: false,
        incomplete: false,
        cancelled: true,
        elapsed_ms: started.elapsed().as_millis() as u64,
        generation: 0,
    }
}

fn new_raster(n_render: usize) -> Vec<Vec<u8>> {
    vec![vec![0u8; RASTER_SIZE[0] * RASTER_SIZE[1]]; n_render]
}

/// Rasterize one cluster's whole valid set, in parallel chunks above
/// [`PARALLEL_RASTER_MIN`]. Returns the number of spikes that contributed.
///
/// Every task owns a local density image; merging is a saturating `u8` add,
/// which is associative and commutative, so the published image is identical
/// to a sequential pass. Progress and cancellation stay atomic.
#[allow(clippy::too_many_arguments)]
fn rasterize_slot<P: DataProvider>(
    session: &Session<P>,
    key: &AllKey,
    times: &[SampleIndex],
    valid_mask: &[bool],
    raster_lo: f32,
    raster_hi: f32,
    progress: &AllProgress,
    cancel: &AtomicBool,
    raster: &mut Vec<Vec<u8>>,
) -> usize
where
    Session<P>: Sync,
{
    let nc = key.n_channels as usize;
    let snippet_len = (key.pre + key.post + 1) as usize;
    let n_samples = session.provider.n_samples().0;
    let sr = session.provider.sample_rate();
    let render_channels = &key.render_channels;
    let init = || {
        (
            new_raster(render_channels.len()),
            SnipScratch::new(nc, snippet_len),
            0usize,
        )
    };

    let (merged, _, drawn) = if times.len() >= PARALLEL_RASTER_MIN {
        use rayon::prelude::*;
        times
            .par_chunks(RASTER_CHUNK)
            .enumerate()
            .fold(init, |mut acc, (chunk_index, chunk)| {
                acc.2 += rasterize_batch_into(
                    session,
                    key,
                    chunk,
                    chunk_index * RASTER_CHUNK,
                    valid_mask,
                    &mut acc.0,
                    &mut acc.1,
                    raster_lo,
                    raster_hi,
                    sr,
                    nc,
                    snippet_len,
                    n_samples,
                    progress,
                    cancel,
                );
                acc
            })
            .reduce(init, |mut left, right| {
                merge_raster(&mut left.0, &right.0);
                left.2 += right.2;
                left
            })
    } else {
        let (mut image, mut scratch, _) = init();
        let drawn = rasterize_batch_into(
            session,
            key,
            times,
            0,
            valid_mask,
            &mut image,
            &mut scratch,
            raster_lo,
            raster_hi,
            sr,
            nc,
            snippet_len,
            n_samples,
            progress,
            cancel,
        );
        (image, scratch, drawn)
    };
    *raster = merged;
    drawn
}

#[inline]
fn merge_raster(dst: &mut [Vec<u8>], src: &[Vec<u8>]) {
    for (target, source) in dst.iter_mut().zip(src) {
        for (a, &b) in target.iter_mut().zip(source) {
            *a = a.saturating_add(b);
        }
    }
}

/// Process `times[base_index..]` into a task-local density image. Called by
/// both the sequential and the parallel path so the counting and raster
/// semantics cannot drift.
#[allow(clippy::too_many_arguments)]
fn rasterize_batch_into<P: DataProvider>(
    session: &Session<P>,
    key: &AllKey,
    times: &[SampleIndex],
    base_index: usize,
    valid_mask: &[bool],
    raster: &mut [Vec<u8>],
    scratch: &mut SnipScratch,
    raster_lo: f32,
    raster_hi: f32,
    sr: f32,
    nc: usize,
    snippet_len: usize,
    n_samples: u64,
    progress: &AllProgress,
    cancel: &AtomicBool,
) -> usize {
    let render_channels = &key.render_channels;
    let mut drawn = 0usize;
    let mut base = base_index;
    for batch in times.chunks(PREFETCH_BATCH) {
        if cancel.load(Ordering::Relaxed) {
            return drawn;
        }
        let windows: Vec<(SampleIndex, u32)> = batch
            .iter()
            .map(|&t| {
                let (start, len, _) = snippet_window(sr, t, key.pre, key.post, key.cfg);
                (start, len)
            })
            .collect();
        session.provider.prefetch(&windows);
        for (offset, &t) in batch.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return drawn;
            }
            if !valid_mask.get(base + offset).copied().unwrap_or(false) {
                continue;
            }
            let (start, len, _) = snippet_window(sr, t, key.pre, key.post, key.cfg);
            if start.0.saturating_add(len as u64) > n_samples {
                continue;
            }
            if !scratch.extract(
                &session.provider,
                t,
                key.pre,
                key.post,
                key.cfg,
                &key.shanks,
                nc,
                snippet_len,
            ) {
                continue;
            }
            subtract_render_medians(
                &mut scratch.buf,
                nc,
                snippet_len,
                render_channels,
                &mut scratch.median,
            );
            if !scratch.buf.iter().all(|v| v.is_finite()) {
                continue;
            }
            for (r, &ch) in render_channels.iter().enumerate() {
                if ch < nc {
                    rasterize_trace(
                        &mut raster[r],
                        &scratch.buf,
                        nc,
                        snippet_len,
                        ch,
                        raster_lo,
                        raster_hi,
                    );
                }
            }
            drawn += 1;
        }
        progress
            .raster_done
            .fetch_add(batch.len(), Ordering::Relaxed);
        base += batch.len();
    }
    drawn
}

/// Add one preprocessed snippet's channel `ch` to a density image. Pixel row
/// 0 is `raster_hi`; columns span the snippet uniformly. Coverage saturates,
/// so dense clouds stay visible without any per-spike egui geometry.
///
/// The segment is walked **per destination column**: a column covers the
/// vertical span the trace crosses there and gets one 55-point stamp per
/// scanned row. This is the same coverage the former per-pixel DDA produced,
/// but without four saturating bilinear updates for every ~1-px step along
/// the line, which made a million-spike cluster spend minutes in floating
/// point splats.
fn rasterize_trace(
    alpha: &mut [u8],
    snip: &[f32],
    nc: usize,
    snippet_len: usize,
    ch: usize,
    lo: f32,
    hi: f32,
) {
    let [w, h] = RASTER_SIZE;
    if snippet_len < 2 || hi <= lo || alpha.len() < w * h {
        return;
    }
    let y_scale = (h.saturating_sub(1)) as f32 / (hi - lo);
    let y_of = |value: f32| ((hi - value) * y_scale).clamp(0.0, (h - 1) as f32);
    // The sample index maps across the full texture width; the last sample
    // lands on the right edge, so the final column is clamped to w - 1.
    let x_step = w as f32 / (snippet_len - 1) as f32;
    let mut y0 = y_of(snip[ch]);
    for t in 1..snippet_len {
        let y1 = y_of(snip[t * nc + ch]);
        draw_segment(
            alpha,
            w,
            h,
            (t - 1) as f32 * x_step,
            y0,
            t as f32 * x_step,
            y1,
        );
        y0 = y1;
    }
}

/// Coverage stamp for one x column: `weight` is split across every row the
/// segment crosses, so steep sections do not lose density (and leave no
/// gaps) while shallow ones still antialias between two rows.
fn draw_segment(alpha: &mut [u8], w: usize, h: usize, x0: f32, y0: f32, x1: f32, y1: f32) {
    if !x0.is_finite() || !x1.is_finite() || !y0.is_finite() || !y1.is_finite() || x1 <= x0 {
        return;
    }
    const WEIGHT: f32 = 55.0;
    let dx = x1 - x0;
    let slope = (y1 - y0) / dx;
    let c0 = x0.floor().max(0.0) as usize;
    let c1 = (x1.ceil() as isize - 1).max(c0 as isize) as usize;
    let c1 = c1.min(w - 1);
    for x in c0..=c1 {
        let ax = (x as f32).max(x0);
        let bx = ((x + 1) as f32).min(x1);
        if bx <= ax {
            continue;
        }
        // Fraction of this pixel column covered by the segment. Normal
        // columns are exactly 1; only the two ends can be partial.
        let frac = (bx - ax).min(1.0);
        let ya = y0 + slope * (ax - x0);
        let yb = y0 + slope * (bx - x0);
        add_column(alpha, w, h, x, ya, yb, WEIGHT * frac);
    }
}

#[inline]
fn add_column(alpha: &mut [u8], w: usize, h: usize, x: usize, ya: f32, yb: f32, weight: f32) {
    let (mut lo, mut hi) = if ya <= yb { (ya, yb) } else { (yb, ya) };
    lo = lo.clamp(0.0, (h - 1) as f32);
    hi = hi.clamp(0.0, (h - 1) as f32);
    let span = hi - lo;
    if span < 1.0 {
        // A shallow crossing: one bilinear stamp, exactly as a single DDA
        // step used to add.
        let y = (lo + hi) * 0.5;
        let y0 = y.floor();
        let fy = y - y0;
        let row = y0 as usize;
        add_pixel(alpha, w, x, row, weight * (1.0 - fy));
        if fy > 0.0 && row + 1 < h {
            add_pixel(alpha, w, x, row + 1, weight * fy);
        }
        return;
    }
    // Spread `weight * span` over the integer rows the segment crosses, so
    // a steep jump gets the same per-row density a DDA walk would.
    let per_row = weight * span / (hi.floor() - lo.floor() + 1.0).max(1.0);
    let first = lo.floor() as usize;
    let last = (hi.floor() as usize).min(h - 1);
    for row in first..=last {
        add_pixel(alpha, w, x, row, per_row);
    }
}

#[inline]
fn add_pixel(alpha: &mut [u8], w: usize, x: usize, y: usize, weight: f32) {
    if weight <= 0.0 {
        return;
    }
    let index = y * w + x;
    if let Some(pixel) = alpha.get_mut(index) {
        *pixel = (*pixel as f32 + weight).min(255.0) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waveform_view::subtract_median_baseline;
    use sorrel_data::Journal;
    use sorrel_io::TraceSlice;

    const SR: f32 = 30_000.0;
    const NC: usize = 2;

    struct StubProvider {
        trace: Vec<f32>,
        spikes: Vec<Vec<SampleIndex>>,
        n_samples: u64,
        sr: f32,
    }

    impl DataProvider for StubProvider {
        type Label = u8;
        fn sample_rate(&self) -> f32 {
            self.sr
        }
        fn n_channels(&self) -> u32 {
            NC as u32
        }
        fn n_samples(&self) -> SampleIndex {
            SampleIndex(self.n_samples)
        }
        fn n_clusters(&self) -> u32 {
            self.spikes.len() as u32
        }
        fn spike_times(&self, cluster: ClusterId) -> &[SampleIndex] {
            self.spikes
                .get(cluster.0 as usize)
                .map(Vec::as_slice)
                .unwrap_or(&[])
        }
        fn trace(&self, start: SampleIndex, len: u32) -> TraceSlice<'_> {
            let begin = start.0 as usize * NC;
            let end = begin.saturating_add(len as usize * NC);
            let samples = self.trace.get(begin..end).map_or(&[][..], |s| s);
            TraceSlice {
                start,
                n_channels: NC as u32,
                samples: TraceSamples::F32(samples),
            }
        }
        fn initial_labels(&self) -> Vec<Self::Label> {
            vec![0; self.spikes.len()]
        }
    }

    fn provider_with_spikes(n: usize) -> StubProvider {
        provider_with_spikes_at(n, SR, 200)
    }

    fn provider_with_spikes_at(n: usize, sr: f32, spacing: usize) -> StubProvider {
        let mut trace = vec![0.0f32; n * spacing * NC];
        let mut spikes = Vec::with_capacity(n);
        // One narrow, signed pulse per spike; amplitudes 10..=10*n so the
        // all-spike mean differs sharply from a stride-based legacy sample.
        let kernel = [0.25f32, 0.5, 1.0, 0.5, 0.25];
        for i in 0..n {
            let t = i * spacing + spacing / 2;
            spikes.push(SampleIndex(t as u64));
            let amplitude = 10.0 * (i + 1) as f32;
            for (k, amp) in kernel.iter().enumerate() {
                let sample = t + k - 2;
                trace[sample * NC] = *amp * amplitude;
                trace[sample * NC + 1] = -*amp * amplitude;
            }
        }
        StubProvider {
            trace,
            spikes: vec![spikes],
            n_samples: (n * spacing) as u64,
            sr,
        }
    }

    fn session(provider: StubProvider) -> (Session<StubProvider>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open_or_create(&dir.path().join("journal"), 0, 1).unwrap();
        (Session::new(provider, journal), dir)
    }

    fn key_at(selection: &[ClusterId], version: u64, sr: f32) -> AllKey {
        AllKey {
            clusters: selection.to_vec(),
            pre: super::super::waveform_window_samples(sr),
            post: super::super::waveform_window_samples(sr),
            cfg: TracePreproc::Off,
            shanks: Vec::new(),
            render_channels: vec![0, 1],
            n_channels: NC as u32,
            n_samples: 0,
            sample_rate: sr,
            version,
        }
    }

    fn all_key<P: DataProvider>(session: &Session<P>, selection: &[ClusterId]) -> AllKey {
        let mut key = key_at(selection, session.version(), session.provider.sample_rate());
        key.n_channels = session.provider.n_channels();
        key.n_samples = session.provider.n_samples().0;
        key
    }

    #[test]
    fn every_spike_contributes_to_the_mean_not_the_legacy_hundred_sample() {
        let (session, _dir) = session(provider_with_spikes(250));
        let key = all_key(&session, &[ClusterId(0)]);
        let progress = AllProgress::new(250, 1);
        let cancel = AtomicBool::new(false);
        let result = run_all(&session, &key, &progress, &cancel, None);
        assert!(!result.cancelled && !result.incomplete);
        let cluster = &result.clusters[0];
        assert_eq!(cluster.total, 250);
        assert_eq!(cluster.valid, 250);
        assert_eq!(cluster.invalid, 0);
        assert_eq!(cluster.drawn, 250, "every valid trace is drawn");
        let peak = cluster.mean[0].iter().copied().fold(0.0f32, f32::max);
        assert!(
            (peak - 1255.0).abs() < 0.01,
            "mean of 10..=2500 must be 1255, got {peak}"
        );
        // A 100-spike stride sample of 250 has a very different mean.
        let sample_stride = 250 / 100;
        let sampled_peak: f32 = (0..250)
            .step_by(sample_stride)
            .take(100)
            .map(|i| 10.0 * (i + 1) as f32)
            .sum::<f32>()
            / 100.0;
        assert!(
            (sampled_peak - peak).abs() > 200.0,
            "the all-spike mean cannot be the legacy sample ({sampled_peak} vs {peak})"
        );
        assert_eq!(result.valid, 250);
        assert_eq!(progress.snapshot().done, 250);
    }

    #[test]
    fn optimized_extraction_reproduces_the_reference_mean_and_extrema_exactly() {
        // Above EXTRACT_CHUNK, so this exercises the parallel merge too.
        let n = 5_000;
        let provider = provider_with_spikes(n);
        let expected_mean = {
            let mut sum = vec![vec![0.0f64; 61]; NC];
            let mut min = vec![f32::INFINITY; NC];
            let mut max = vec![f32::NEG_INFINITY; NC];
            let nc = NC;
            for &t in &provider.spikes[0] {
                let mut snip =
                    sorrel_render::extract_snippet(&provider, t, 30, 30, TracePreproc::Off, &[])
                        .unwrap();
                subtract_median_baseline(&mut snip, nc);
                let channels = sum.iter_mut().zip(min.iter_mut()).zip(max.iter_mut());
                for (ch, ((row, low), high)) in channels.enumerate() {
                    for (s, &v) in snip[ch..].iter().step_by(nc).take(61).enumerate() {
                        row[s] += v as f64;
                        *low = low.min(v);
                        *high = high.max(v);
                    }
                }
            }
            let n = provider.spikes[0].len() as f64;
            (
                sum.into_iter()
                    .map(|row| row.into_iter().map(|v| (v / n) as f32).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                min,
                max,
            )
        };
        let (session, _dir) = session(provider);
        let key = all_key(&session, &[ClusterId(0)]);
        let result = run_all(
            &session,
            &key,
            &AllProgress::new(n, 1),
            &AtomicBool::new(false),
            None,
        );
        assert_eq!(
            result.clusters[0].mean, expected_mean.0,
            "bit-identical mean"
        );
        assert_eq!(result.clusters[0].min, expected_mean.1, "exact extrema");
        assert_eq!(result.clusters[0].max, expected_mean.2);
    }

    #[test]
    fn edge_and_missing_windows_are_reported_but_do_not_stop_the_mean() {
        let mut provider = provider_with_spikes(4);
        // Both recording edges are invalid, not shifted to another centre.
        provider.spikes[0].insert(0, SampleIndex(0));
        provider.spikes[0].push(SampleIndex(provider.n_samples - 1));
        let expected_total = provider.spikes[0].len();
        let (session, _dir) = session(provider);
        let key = all_key(&session, &[ClusterId(0)]);
        let progress = AllProgress::new(expected_total, 1);
        let result = run_all(
            &session,
            &key,
            &AllProgress::new(0, 0),
            &AtomicBool::new(false),
            None,
        );
        let cluster = &result.clusters[0];
        assert_eq!(cluster.total, expected_total);
        assert_eq!(cluster.valid, 4);
        assert_eq!(cluster.invalid, 2);
        assert_eq!(cluster.out_of_range, 2);
        assert_eq!(result.drawn, cluster.valid);
        let _ = progress;
    }

    #[test]
    fn every_valid_spike_is_rasterized_even_beyond_eight_thousand() {
        // Short synthetic snippets (1 sample pre/post at 1 kHz) keep the
        // two-pass worker fast while proving there is no draw budget.
        let n = 8_500;
        let (session, _dir) = session(provider_with_spikes_at(n, 1_000.0, 20));
        let key = all_key(&session, &[ClusterId(0)]);
        let result = run_all(
            &session,
            &key,
            &AllProgress::new(n, 1),
            &AtomicBool::new(false),
            None,
        );
        let cluster = &result.clusters[0];
        assert_eq!(cluster.total, n);
        assert_eq!(cluster.valid, n);
        assert_eq!(
            cluster.drawn, n,
            "ALL valid spikes are rasterized, not an 8,000-spike draw sample"
        );
        assert_eq!(result.drawn, n);
        assert_eq!(result.valid, n);
        assert_eq!(result.raster_failed, 0);
        assert!(result.raster_lo < 0.0 && result.raster_hi > 0.0);
    }

    #[test]
    fn a_provisional_result_carries_the_exact_mean_before_the_raster_finishes() {
        let (session, _dir) = session(provider_with_spikes(300));
        let key = all_key(&session, &[ClusterId(0)]);
        let mut updates = Vec::new();
        let result = run_all_emit(
            &session,
            &key,
            &AllProgress::new(300, 1),
            &AtomicBool::new(false),
            None,
            &mut |provisional| {
                updates.push(provisional);
                true
            },
        );
        assert_eq!(updates.len(), 1, "pass 1 publishes one mean-ready update");
        let provisional = &updates[0];
        assert!(!provisional.raster_complete);
        assert_eq!((provisional.valid, provisional.drawn), (300, 0));
        assert_eq!(provisional.clusters[0].mean, result.clusters[0].mean);
        assert_eq!(provisional.clusters[0].min, result.clusters[0].min);
        assert_eq!(provisional.clusters[0].max, result.clusters[0].max);
        assert!(result.raster_complete);
        assert_eq!(result.drawn, 300);
    }

    #[test]
    fn nonfinite_raw_is_invalid_and_cannot_poison_the_full_set_mean() {
        let mut provider = provider_with_spikes(4);
        provider.trace[100 * NC] = f32::NAN;
        let (session, _dir) = session(provider);
        let key = all_key(&session, &[ClusterId(0)]);
        let result = run_all(
            &session,
            &key,
            &AllProgress::new(4, 1),
            &AtomicBool::new(false),
            None,
        );
        assert_eq!(
            (result.total, result.valid, result.drawn, result.missing),
            (4, 3, 3, 1)
        );
        assert!(result.clusters[0]
            .mean
            .iter()
            .flatten()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn changing_trace_availability_cannot_substitute_a_different_spike_for_the_mean_set() {
        struct ChangingTrace {
            inner: StubProvider,
            calls: AtomicUsize,
        }
        impl DataProvider for ChangingTrace {
            type Label = u8;
            fn sample_rate(&self) -> f32 {
                self.inner.sample_rate()
            }
            fn n_channels(&self) -> u32 {
                self.inner.n_channels()
            }
            fn n_samples(&self) -> SampleIndex {
                self.inner.n_samples()
            }
            fn n_clusters(&self) -> u32 {
                self.inner.n_clusters()
            }
            fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
                self.inner.spike_times(c)
            }
            fn initial_labels(&self) -> Vec<u8> {
                self.inner.initial_labels()
            }
            fn trace(&self, start: SampleIndex, len: u32) -> TraceSlice<'_> {
                let call = self.calls.fetch_add(1, Ordering::Relaxed);
                // Spike A is initially absent, but becomes available later.
                // Spike B disappears after contributing to the mean.
                if call == 0 || (call >= 2 && start.0 == 300 - 30) {
                    TraceSlice {
                        start,
                        n_channels: NC as u32,
                        samples: TraceSamples::F32(&[]),
                    }
                } else {
                    self.inner.trace(start, len)
                }
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open_or_create(&dir.path().join("journal"), 0, 1).unwrap();
        let session = Session::new(
            ChangingTrace {
                inner: provider_with_spikes(2),
                calls: AtomicUsize::new(0),
            },
            journal,
        );
        let mut key = key_at(&[ClusterId(0)], session.version(), SR);
        key.n_samples = session.provider.n_samples().0;
        let result = run_all(
            &session,
            &key,
            &AllProgress::new(2, 1),
            &AtomicBool::new(false),
            None,
        );
        assert_eq!(
            (result.valid, result.drawn, result.raster_failed),
            (1, 0, 1)
        );
        assert_eq!(
            session.provider.calls.load(Ordering::Relaxed),
            3,
            "do not rasterize spike A in place of B"
        );
    }

    #[test]
    fn a_cancelled_worker_returns_no_publishable_result() {
        let (session, _dir) = session(provider_with_spikes(20));
        let key = all_key(&session, &[ClusterId(0)]);
        let cancel = AtomicBool::new(true);
        let result = run_all(&session, &key, &AllProgress::new(20, 1), &cancel, None);
        assert!(result.cancelled);
        assert!(result.clusters.is_empty());
    }

    #[test]
    fn cancelling_during_the_raster_pass_stops_before_the_next_spike() {
        struct CancelAtRaster {
            inner: StubProvider,
            progress: Arc<AllProgress>,
            cancel: Arc<AtomicBool>,
        }
        impl DataProvider for CancelAtRaster {
            type Label = u8;
            fn sample_rate(&self) -> f32 {
                self.inner.sample_rate()
            }
            fn n_channels(&self) -> u32 {
                self.inner.n_channels()
            }
            fn n_samples(&self) -> SampleIndex {
                self.inner.n_samples()
            }
            fn n_clusters(&self) -> u32 {
                1
            }
            fn spike_times(&self, _c: ClusterId) -> &[SampleIndex] {
                &self.inner.spikes[0]
            }
            fn initial_labels(&self) -> Vec<u8> {
                vec![0]
            }
            fn trace(&self, start: SampleIndex, len: u32) -> TraceSlice<'_> {
                if self.progress.snapshot().stage == STAGE_RASTER {
                    self.cancel.store(true, Ordering::Relaxed);
                }
                self.inner.trace(start, len)
            }
        }
        let progress = Arc::new(AllProgress::new(64, 1));
        let cancel = Arc::new(AtomicBool::new(false));
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open_or_create(&dir.path().join("journal"), 0, 1).unwrap();
        let session = Session::new(
            CancelAtRaster {
                inner: provider_with_spikes(64),
                progress: Arc::clone(&progress),
                cancel: Arc::clone(&cancel),
            },
            journal,
        );
        let key = all_key(&session, &[ClusterId(0)]);
        let result = run_all(&session, &key, &progress, &cancel, None);
        assert!(result.cancelled);
        assert!(result.clusters.is_empty(), "no partial raster is published");
    }

    #[test]
    fn a_late_result_for_an_old_selection_is_never_published_as_current() {
        let mut a = key_at(&[ClusterId(0)], 1, SR);
        a.n_samples = 100_000;
        let mut b = key_at(&[ClusterId(1)], 1, SR);
        b.n_samples = 100_000;
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let mut cache = AllCache {
            requested: Some(a.clone()),
            job: Some(AllJob {
                key: a,
                rx,
                cancel: Arc::clone(&cancel),
            }),
            ..Default::default()
        };
        // Selection changed while the old worker was still running.
        cache.begin(&b);
        assert!(cache.job.is_none(), "the old job is dropped");
        assert!(
            cancel.load(Ordering::Relaxed),
            "dropping the old job must cancel its worker"
        );
        // A result the old worker sent before noticing must not land.
        let late = AllResult {
            clusters: Vec::new(),
            render_channels: vec![0],
            raster_lo: -1.0,
            raster_hi: 1.0,
            total: 7,
            valid: 7,
            invalid: 0,
            out_of_range: 0,
            missing: 0,
            drawn: 7,
            raster_failed: 0,
            raster_complete: true,
            incomplete: false,
            cancelled: false,
            elapsed_ms: 0,
            generation: 0,
        };
        let _ = tx.send(AllUpdate::Final(Box::new(late)));
        cache.poll(&b);
        assert!(
            cache.current().is_none(),
            "the old selection's result must not be shown as the new one"
        );
        assert_eq!(cache.requested(), Some(&b));
    }

    #[test]
    fn rasterization_covers_every_sample_column() {
        let mut alpha = vec![0u8; RASTER_SIZE[0] * RASTER_SIZE[1]];
        // Flat and then a spike; every column must receive some coverage.
        let mut snip = vec![0.0f32; 9];
        snip[4] = 0.5;
        rasterize_trace(&mut alpha, &snip, 1, 9, 0, -1.0, 1.0);
        for x in 0..RASTER_SIZE[0] {
            assert!(
                (0..RASTER_SIZE[1]).any(|y| alpha[y * RASTER_SIZE[0] + x] > 0),
                "column {x} was not rasterized"
            );
        }
    }

    #[test]
    fn a_steep_segment_leaves_no_gap_between_its_endpoints() {
        let mut alpha = vec![0u8; RASTER_SIZE[0] * RASTER_SIZE[1]];
        // Nine samples span eight 64-column intervals; the first interval
        // jumps from the top to the bottom, so each column crosses ~4 rows.
        let mut snip = vec![0.0f32; 9];
        snip[0] = 1.0;
        snip[1] = -1.0;
        rasterize_trace(&mut alpha, &snip, 1, 9, 0, -1.0, 1.0);
        let rows: Vec<usize> = (0..RASTER_SIZE[1])
            .filter(|&y| (0..64).any(|x| alpha[y * RASTER_SIZE[0] + x] > 0))
            .collect();
        assert!(
            rows.len() >= RASTER_SIZE[1] - 2,
            "vertical segment covered {} of {} rows",
            rows.len(),
            RASTER_SIZE[1]
        );
        assert!(rows.windows(2).all(|w| w[1] == w[0] + 1), "no gaps");
    }

    /// Real-recording benchmark. `SORREL_WAVEFORM_BENCH_ROOT` must point at a
    /// phy/Kilosort directory whose `params.py` names the raw recording (the
    /// recording itself is only ever opened read-only). `SORREL_WAVEFORM_BENCH_
    /// SPIKES` optionally truncates the cluster for a quick run; the default
    /// processes every spike. Run with:
    ///
    /// ```text
    /// cargo test -p sorrel-ui --release --lib \
    ///   waveform_all::tests::bench_real_recording -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "reads a real recording; set SORREL_WAVEFORM_BENCH_ROOT"]
    fn bench_real_recording() {
        use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider, PhyLabel};
        use std::path::{Path, PathBuf};

        struct SizedProvider {
            inner: KilosortProvider,
            times: Vec<SampleIndex>,
        }

        impl DataProvider for SizedProvider {
            type Label = PhyLabel;
            fn sample_rate(&self) -> f32 {
                self.inner.sample_rate()
            }
            fn n_channels(&self) -> u32 {
                self.inner.n_channels()
            }
            fn n_samples(&self) -> SampleIndex {
                self.inner.n_samples()
            }
            fn n_clusters(&self) -> u32 {
                1
            }
            fn spike_times(&self, _cluster: ClusterId) -> &[SampleIndex] {
                &self.times
            }
            fn trace(&self, start: SampleIndex, len: u32) -> TraceSlice<'_> {
                self.inner.trace(start, len)
            }
            fn initial_labels(&self) -> Vec<Self::Label> {
                vec![PhyLabel::default()]
            }
        }

        let Some(root) = std::env::var_os("SORREL_WAVEFORM_BENCH_ROOT").map(PathBuf::from) else {
            eprintln!("skipping: set SORREL_WAVEFORM_BENCH_ROOT to a sorter output directory");
            return;
        };
        let tmp = tempfile::tempdir().unwrap();
        for name in ["params.py", "spike_times.npy", "spike_clusters.npy"] {
            std::fs::copy(root.join(name), tmp.path().join(name)).unwrap();
        }
        let params = KilosortOpenParams {
            dat_path: Some(Path::new(&root).join("recording.dat")),
            ..Default::default()
        };
        let inner = KilosortProvider::open(tmp.path(), params).unwrap();
        let all_times = inner.spike_times(ClusterId(0)).to_vec();
        let limit = std::env::var("SORREL_WAVEFORM_BENCH_SPIKES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(all_times.len())
            .min(all_times.len());
        let times = all_times[..limit].to_vec();
        eprintln!(
            "benchmark: {} spikes ({} total in cluster 0), {} channels, {} samples",
            times.len(),
            all_times.len(),
            inner.n_channels(),
            inner.n_samples().0
        );
        let provider = SizedProvider { inner, times };
        let journal = Journal::open_or_create(&tmp.path().join("journal"), 0, 1).unwrap();
        let session = Session::new(provider, journal);
        let key = AllKey {
            clusters: vec![ClusterId(0)],
            pre: super::super::waveform_window_samples(session.provider.sample_rate()),
            post: super::super::waveform_window_samples(session.provider.sample_rate()),
            cfg: TracePreproc::Off,
            shanks: Vec::new(),
            render_channels: (0..session.provider.n_channels() as usize).collect(),
            n_channels: session.provider.n_channels(),
            n_samples: session.provider.n_samples().0,
            sample_rate: session.provider.sample_rate(),
            version: session.version(),
        };
        let progress = AllProgress::new(limit, 1);
        let cancel = AtomicBool::new(false);
        let started = Instant::now();
        let mut mean_ready = None;
        let result = run_all_emit(
            &session,
            &key,
            &progress,
            &cancel,
            None,
            &mut |provisional| {
                mean_ready = Some(started.elapsed());
                eprintln!(
                    "benchmark: exact mean ready after {} ms ({} valid)",
                    provisional.elapsed_ms, provisional.valid
                );
                true
            },
        );
        let elapsed = started.elapsed();
        eprintln!(
            "benchmark: {} valid, {} drawn, {} raster_failed, {} ms total ({:.0} spikes/s), mean ready {:?}",
            result.valid,
            result.drawn,
            result.raster_failed,
            elapsed.as_millis(),
            result.valid as f64 / elapsed.as_secs_f64().max(1e-9),
            mean_ready,
        );
        assert_eq!(result.valid, limit);
        assert_eq!(result.drawn, limit);
    }
}

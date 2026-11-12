//! phy WaveformView: spikes, means or templates per channel at probe positions.
//!
//! One shared channel set follows the primary cluster's peak, four channels
//! in all by default, on the same shank. Modes, overlap and the
//! scaling keys follow phy v2.1's `WaveformView`. Parameters live in
//! [`WaveformParams`] so they persist with the other prefs.

use egui::{Color32, Pos2, Sense, Stroke, Ui};
use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider, SampleIndex};
use sorrel_render::{extract_snippet, snippet_window, TracePreproc};

use crate::feature_view::{ChannelEdit, FeatureLink};
use crate::layout::ViewAction;
use crate::prefs::{WaveformMode, WaveformParams};
use crate::theme::{selection_colour, Theme};
use crate::views::{alloc_plot, grid_colour, nice_ticks, pad_range, tick_label, TICK_PX};

/// Templates drawn per cluster in `Templates` mode (a heavily merged cluster
/// can span hundreds); the rest are dropped, smallest share first.
// ponytail: fixed cap, make it a param if someone needs the long tail.
const MAX_TEMPLATES: usize = 16;
/// phy `_overlap_transform` ratio of waveform width to the gap between slots.
const SLOT_K: f32 = 8.0;

// ---- Waveform extraction & cache ---------------------------------------

/// Window-around-spike used by `WaveformCache`. Picks ±1 ms by default.
pub(crate) fn waveform_window_samples(sr: f32) -> u32 {
    (sr.max(1.0) * 0.001).round() as u32
}

/// phy v2.1 baseline: subtract each channel's median over the snippet
/// (`snip` is `[sample * nc + channel]`).
pub(crate) fn subtract_median_baseline(snip: &mut [f32], nc: usize) {
    let n = snip.len().checked_div(nc).unwrap_or(0);
    if n == 0 {
        return;
    }
    let mut col = vec![0.0f32; n];
    for ch in 0..nc {
        for (t, c) in col.iter_mut().enumerate() {
            *c = snip[t * nc + ch];
        }
        let (_, med, _) = col.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
        let med = *med;
        for t in 0..n {
            snip[t * nc + ch] -= med;
        }
    }
}

/// Cached snippet extraction keyed on `(selection, pre, post, preproc)`.
/// Snippets go through the trace pane's HP/CMR setting (see
/// [`WaveformCache::set_preproc`]) so waveforms match the trace. Spike
/// extraction walks the trace mmap N times per spike, which is fast in the
/// steady state thanks to the page cache, but recomputing it every paint
/// stalls the egui thread on cluster switches.
#[derive(Debug)]
pub struct WaveformCache {
    key: Option<WaveformKey>,
    cfg: TracePreproc,
    /// Spikes sampled per cluster (phy `n_spikes_waveforms`).
    n_spikes: usize,
    /// Shank id per channel for per-shank CMR (empty = global).
    shanks: Vec<u32>,
    /// `[cluster][spike][sample_in_snippet * n_channels + channel]`
    snippets: Vec<Vec<Vec<f32>>>,
    /// `[cluster][channel][sample_in_snippet]`
    means: Vec<Vec<Vec<f32>>>,
    /// `[cluster]` -> `(template id, share of the cluster's spikes)`, biggest
    /// share first; empty when spike templates are not seeded.
    templates: Vec<Vec<(u32, f32)>>,
    /// y range frozen on first draw (phy v2.1 `data_bounds or ...`):
    /// `(mode, n_channels, lo, hi)`. Resets on mode change and `R`.
    frozen: Option<(WaveformMode, usize, f32, f32)>,
    /// Snippets of the lassoed / k-means spikes the Features pane links in.
    linked: Option<LinkedSnippets>,
    /// Snippets of the spike brush's split what-if part / ISI violators.
    brushed: Option<LinkedSnippets>,
    /// Snippet of the spike hovered in another view, drawn bold.
    hovered: Option<LinkedSnippets>,
    /// The split what-if: `(cluster slot, inside means, outside means)`,
    /// `[channel][sample]`, with the key they were computed for.
    whatif: Option<(WhatIfKey, usize, WhatIfMeans)>,
    /// Merge ghost: its mean through the same extraction path, as a
    /// one-cluster cache of its own (the main key stays the selection's).
    ghost: Option<Box<WaveformCache>>,
    /// Whether the ghost is drawn this frame.
    ghost_on: bool,
}

/// `[inside, outside]`, each `[channel][sample]`.
type WhatIfMeans = [Vec<Vec<f32>>; 2];

type WhatIfKey = (
    Option<WaveformKey>,
    (crate::brush::BrushSource, u64, usize, usize),
    ClusterId,
);

/// Snippets for `FeatureLink::spikes`, extracted once per link identity.
#[derive(Debug)]
struct LinkedSnippets {
    spikes: Vec<(ClusterId, std::sync::Arc<[u32]>)>,
    pre: u32,
    post: u32,
    cfg: TracePreproc,
    version: u64,
    /// `[spike][sample_in_snippet * n_channels + channel]`
    snips: Vec<Vec<f32>>,
}

/// Linked spikes extracted per cluster.
// ponytail: the Features pane already caps its list at 100; this repeats the
// guard so a stray huge list cannot stall a paint. Extraction is synchronous
// like the main cache (same cost: <= 100 snippets per cluster, once per link);
// move it onto a crate::jobs worker if lists ever grow.
const MAX_LINKED_PER_CLUSTER: usize = 100;

/// The link's spikes, unless made against an older curation version
/// (Features not redrawn since): the indices now name other spikes.
fn link_spikes(link: &FeatureLink, version: u64) -> &[(ClusterId, std::sync::Arc<[u32]>)] {
    if link.version == version {
        &link.spikes
    } else {
        &[]
    }
}

/// At most `cap` of `idx`, spread over the whole list (a stride would keep
/// only the first `cap` of 101..=2*cap entries).
fn spread(idx: &[u32], cap: usize) -> impl Iterator<Item = u32> + '_ {
    let take = idx.len().min(cap);
    (0..take).map(move |i| idx[i * idx.len() / take])
}

impl Default for WaveformCache {
    fn default() -> Self {
        Self {
            key: None,
            cfg: TracePreproc::default(),
            n_spikes: 100,
            shanks: Vec::new(),
            snippets: Vec::new(),
            means: Vec::new(),
            templates: Vec::new(),
            frozen: None,
            linked: None,
            brushed: None,
            hovered: None,
            whatif: None,
            ghost: None,
            ghost_on: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WaveformKey {
    clusters: Vec<ClusterId>,
    pre: u32,
    post: u32,
    cfg: TracePreproc,
    n_channels: u32,
    n_spikes: usize,
    /// Curation version: merges/splits/undo change which spikes a cluster
    /// owns, so the snippets must be re-extracted.
    version: u64,
}

impl WaveformCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Preprocess snippets like the trace pane; re-extracts on change.
    pub(crate) fn set_preproc(&mut self, cfg: TracePreproc, shanks: &[u32]) {
        if self.shanks != shanks {
            self.shanks = shanks.to_vec();
            self.key = None;
            self.linked = None;
            self.brushed = None;
            self.hovered = None;
        }
        self.cfg = cfg;
    }

    /// Spikes per cluster for the next [`ensure`](Self::ensure); a change
    /// invalidates the key.
    pub(crate) fn set_n_spikes(&mut self, n: usize) {
        self.n_spikes = n.max(1);
    }

    fn whatif_on(&self) -> bool {
        self.whatif.is_some()
    }

    /// The ghost cluster's `[channel][sample]` mean while it is drawn.
    fn ghost_mean(&self) -> Option<&[Vec<f32>]> {
        self.ghost
            .as_ref()
            .filter(|_| self.ghost_on)
            .and_then(|g| g.means.first())
            .map(Vec::as_slice)
    }

    /// Peak channel (argmax of |mean|) of the cluster at `slot` in the
    /// selection the cache was built for. `None` until built.
    pub fn peak_channel(&self, slot: usize) -> Option<usize> {
        let means = self.means.get(slot)?;
        let mut best: Option<(usize, f32)> = None;
        for (ch, mean) in means.iter().enumerate() {
            let peak = mean.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            match best {
                Some((_, b)) if peak <= b => {}
                _ => best = Some((ch, peak)),
            }
        }
        best.filter(|&(_, p)| p > 0.0).map(|(ch, _)| ch)
    }

    /// Extract snippets for `selection` if the key changed. Public to the
    /// crate so the trace and probe panes can share one extraction.
    pub(crate) fn ensure<P: DataProvider>(
        &mut self,
        session: &Session<P>,
        selection: &[ClusterId],
        pre: u32,
        post: u32,
    ) {
        let key = WaveformKey {
            clusters: selection.to_vec(),
            pre,
            post,
            cfg: self.cfg,
            n_channels: session.provider.n_channels(),
            n_spikes: self.n_spikes,
            version: session.version(),
        };
        if self.key.as_ref() == Some(&key) {
            return;
        }
        let snippet_len = (pre + post + 1) as usize;
        let nc = key.n_channels as usize;
        let max_spikes = self.n_spikes.max(1);

        // Pick the snippet windows first and prefetch them in one call, so
        // a remote provider needs one round trip instead of one per spike
        // (a no-op for local, memory-mapped backends).
        let spikes: Vec<Vec<SampleIndex>> = selection
            .iter()
            .map(|&c| {
                let times = session.spike_times(c);
                if times.is_empty() || nc == 0 || snippet_len < 3 {
                    return Vec::new();
                }
                let stride = (times.len() / max_spikes).max(1);
                times
                    .iter()
                    .step_by(stride)
                    .take(max_spikes)
                    .copied()
                    .collect()
            })
            .collect();
        let sr = session.provider.sample_rate();
        let windows: Vec<(SampleIndex, u32)> = spikes
            .iter()
            .flatten()
            .map(|&t| {
                let (start, len, _) = snippet_window(sr, t, pre, post, key.cfg);
                (start, len)
            })
            .collect();
        session.provider.prefetch(&windows);

        let mut snippets: Vec<Vec<Vec<f32>>> = Vec::with_capacity(selection.len());
        for cluster_spikes in &spikes {
            let bucket = cluster_spikes
                .iter()
                .filter_map(|&t| {
                    let mut s =
                        extract_snippet(&session.provider, t, pre, post, key.cfg, &self.shanks)?;
                    subtract_median_baseline(&mut s, nc);
                    Some(s)
                })
                .collect();
            snippets.push(bucket);
        }
        // Bytes still in flight (remote guest): show what we have but keep
        // the key stale so the extraction reruns once they land.
        let incomplete = session.provider.pending_fetches() > 0;

        // Per-cluster mean waveform per channel.
        let mut means: Vec<Vec<Vec<f32>>> = Vec::with_capacity(selection.len());
        for cluster_snips in &snippets {
            means.push(mean_of(cluster_snips.iter(), nc, snippet_len));
        }

        // Templates each cluster uses, by share of its spikes.
        self.templates = selection
            .iter()
            .map(|&c| {
                let ts = session.spike_templates(c);
                let mut counts: std::collections::HashMap<u32, usize> = Default::default();
                for &t in ts {
                    *counts.entry(t).or_insert(0) += 1;
                }
                let mut v: Vec<(u32, usize)> = counts.into_iter().collect();
                v.sort_by_key(|&(t, n)| (std::cmp::Reverse(n), t));
                v.truncate(MAX_TEMPLATES);
                v.into_iter()
                    .map(|(t, n)| (t, n as f32 / ts.len() as f32))
                    .collect()
            })
            .collect();

        if !incomplete {
            self.key = Some(key);
        }
        self.snippets = snippets;
        self.means = means;
    }

    /// Extract the linked spikes' snippets when the link (compared by `Arc`
    /// identity), window, preprocessing or curation version changed. Local
    /// indices address the cluster's time-sorted bucket, as in
    /// `CurationCommand::Split`.
    pub(crate) fn ensure_linked<P: DataProvider>(
        &mut self,
        session: &Session<P>,
        link: &FeatureLink,
        pre: u32,
        post: u32,
    ) {
        Self::ensure_snips(
            &mut self.linked,
            self.cfg,
            &self.shanks,
            session,
            link_spikes(link, session.version()),
            pre,
            post,
        );
    }

    /// Same extraction for the spike brush's what-if part or violators.
    fn ensure_brushed<P: DataProvider>(
        &mut self,
        session: &Session<P>,
        spikes: &[(ClusterId, std::sync::Arc<[u32]>)],
        pre: u32,
        post: u32,
    ) {
        Self::ensure_snips(
            &mut self.brushed,
            self.cfg,
            &self.shanks,
            session,
            spikes,
            pre,
            post,
        );
    }

    /// The hovered spike (a one-spike list, or none) through the same path.
    fn ensure_hovered<P: DataProvider>(
        &mut self,
        session: &Session<P>,
        spike: Option<(ClusterId, u32)>,
        pre: u32,
        post: u32,
    ) {
        // `ensure_snips` compares the lists by `Arc` identity: keep the one
        // already held while the same spike stays hovered.
        let held = self.hovered.as_ref().map(|l| l.spikes.clone());
        let spikes: Vec<_> = match (spike, held) {
            (Some((c, l)), Some(h)) if h.iter().any(|(hc, hi)| *hc == c && **hi == [l]) => h,
            (Some((c, l)), _) => vec![(c, std::sync::Arc::from([l]))],
            (None, _) => Vec::new(),
        };
        Self::ensure_snips(
            &mut self.hovered,
            self.cfg,
            &self.shanks,
            session,
            &spikes,
            pre,
            post,
        );
    }

    fn ensure_snips<P: DataProvider>(
        slot: &mut Option<LinkedSnippets>,
        cfg: TracePreproc,
        shanks: &[u32],
        session: &Session<P>,
        spikes: &[(ClusterId, std::sync::Arc<[u32]>)],
        pre: u32,
        post: u32,
    ) {
        let version = session.version();
        if let Some(l) = &*slot {
            let same = l.spikes.len() == spikes.len()
                && l.spikes
                    .iter()
                    .zip(spikes)
                    .all(|(a, b)| a.0 == b.0 && std::sync::Arc::ptr_eq(&a.1, &b.1));
            if same && l.pre == pre && l.post == post && l.cfg == cfg && l.version == version {
                return;
            }
        }
        let nc = session.provider.n_channels() as usize;
        let times: Vec<SampleIndex> = spikes
            .iter()
            .flat_map(|(c, idx)| {
                let bucket = session.spike_times(*c);
                // Evenly thinned: a brush can hold thousands.
                spread(idx, MAX_LINKED_PER_CLUSTER)
                    .filter_map(move |i| bucket.get(i as usize).copied())
            })
            .collect();
        let sr = session.provider.sample_rate();
        let windows: Vec<(SampleIndex, u32)> = times
            .iter()
            .map(|&t| {
                let (start, len, _) = snippet_window(sr, t, pre, post, cfg);
                (start, len)
            })
            .collect();
        session.provider.prefetch(&windows);
        let snips = times
            .iter()
            .filter_map(|&t| {
                let mut s = extract_snippet(&session.provider, t, pre, post, cfg, shanks)?;
                subtract_median_baseline(&mut s, nc);
                Some(s)
            })
            .collect();
        // Bytes still in flight (remote guest): keep what we have but leave
        // the link identity empty so the extraction reruns once they land.
        let incomplete = session.provider.pending_fetches() > 0;
        *slot = Some(LinkedSnippets {
            spikes: if incomplete {
                Vec::new()
            } else {
                spikes.to_vec()
            },
            pre,
            post,
            cfg,
            version,
            snips,
        });
    }

    /// The split what-if's inside / outside means for `cluster` (selection
    /// `slot`), from the brushed snippets and the main cache's sample.
    /// Outside = the cluster's sampled spikes that are not brushed; the
    /// sample is `step_by(stride)` of the bucket, so snippet `j` is local
    /// `j * stride` (ponytail: a snippet whose extraction failed shifts the
    /// rest by one; harmless for a mean over <= 100 spikes).
    fn ensure_whatif<P: DataProvider>(
        &mut self,
        session: &Session<P>,
        brush: &crate::brush::SpikeBrush,
        cluster: ClusterId,
        slot: usize,
        nc: usize,
        snippet_len: usize,
    ) {
        let key = (self.key.clone(), brush.key(), cluster);
        if self.whatif.as_ref().is_some_and(|w| w.0 == key) {
            return;
        }
        let inside = self.brushed.iter().flat_map(|l| &l.snips);
        let stride = (session.spike_times(cluster).len() / self.n_spikes.max(1)).max(1);
        let outside = self
            .snippets
            .get(slot)
            .into_iter()
            .flatten()
            .enumerate()
            .filter(|(j, _)| !brush.contains(cluster, (j * stride) as u32));
        let means = [
            mean_of(inside, nc, snippet_len),
            mean_of(outside.map(|(_, s)| s), nc, snippet_len),
        ];
        self.whatif = Some((key, slot, means));
    }

    /// The merge ghost's `[channel][sample]` mean, extracted once per
    /// (cluster, window, curation version) like any selected cluster.
    fn ensure_ghost<P: DataProvider>(
        &mut self,
        session: &Session<P>,
        c: Option<ClusterId>,
        pre: u32,
        post: u32,
    ) {
        self.ghost_on = c.is_some();
        let Some(c) = c else { return };
        let g = self.ghost.get_or_insert_with(Default::default);
        (g.cfg, g.n_spikes) = (self.cfg, self.n_spikes);
        if g.shanks != self.shanks {
            g.shanks = self.shanks.clone();
            g.key = None;
        }
        g.ensure(session, &[c], pre, post);
    }

    /// Frozen y range for `mode`, computing it with `f` the first time (and
    /// after a mode change or reset).
    fn frozen_range(
        &mut self,
        mode: WaveformMode,
        nc: usize,
        f: impl FnOnce(&Self) -> (f32, f32),
    ) -> (f32, f32) {
        match self.frozen {
            Some((m, n, lo, hi)) if m == mode && n == nc => (lo, hi),
            _ => {
                let (lo, hi) = f(self);
                self.frozen = Some((mode, nc, lo, hi));
                (lo, hi)
            }
        }
    }
}

/// Per-channel peak amplitude across all clusters' means. Used to rank
/// channels for the multi-channel layout.
pub(crate) fn channel_peak_abs(means: &[Vec<Vec<f32>>], n_channels: usize) -> Vec<f32> {
    let mut peaks = vec![0.0f32; n_channels];
    for cluster_means in means {
        for (ch, mean) in cluster_means.iter().enumerate() {
            if ch >= peaks.len() {
                continue;
            }
            for &v in mean {
                let a = v.abs();
                if a > peaks[ch] {
                    peaks[ch] = a;
                }
            }
        }
    }
    peaks
}

// ---- Pure layout helpers -------------------------------------------------

/// phylib `get_closest_channels(.., n)`: the `n` channels nearest `peak`,
/// `peak` itself included (distance 0), on its shank (an empty or
/// mismatched `shanks` means one shank), sorted.
pub(crate) fn nearest_channels(
    peak: usize,
    geometry: &[[f32; 2]],
    shanks: &[u32],
    n: usize,
) -> Vec<usize> {
    let one_shank = shanks.len() != geometry.len();
    let d = |c: usize| {
        let (a, b) = (geometry[c], geometry[peak]);
        (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
    };
    let mut others: Vec<usize> = (0..geometry.len())
        .filter(|&c| c != peak && (one_shank || shanks[c] == shanks[peak]))
        .collect();
    others.sort_by(|&a, &b| d(a).total_cmp(&d(b))); // stable: ties by index
    others.truncate(n.saturating_sub(1));
    others.push(peak);
    others.sort_unstable();
    others
}

/// One channel set for the whole selection. Adding a comparison cluster must
/// not add another cluster's neighbours and silently expand a tetrode view.
fn display_channels(
    n_channels: usize,
    peak: usize,
    geometry: &[[f32; 2]],
    shanks: &[u32],
    limit: usize,
) -> Vec<usize> {
    if n_channels == 0 {
        return Vec::new();
    }
    let peak = peak.min(n_channels - 1);
    let limit = limit.clamp(1, n_channels);
    if geometry.len() == n_channels {
        return nearest_channels(peak, geometry, shanks, limit);
    }
    // Without positions there is no physical-neighbour inference to make.
    // Keep the primary peak, then channels in acquisition order, respecting
    // an explicit channel group if one was supplied.
    let mut channels = vec![peak];
    channels
        .extend((0..n_channels).filter(|&ch| {
            ch != peak && (shanks.len() != n_channels || shanks[ch] == shanks[peak])
        }));
    channels.truncate(limit);
    channels.sort_unstable();
    channels
}

/// Peak channel of a template (`n_samples × n_channels`, flat): argmax |v|.
fn template_peak(flat: &[f32], n_ch: usize) -> Option<usize> {
    let (ch, v) = flat
        .iter()
        .enumerate()
        .map(|(i, v)| (i % n_ch.max(1), v.abs()))
        .fold((0, 0.0f32), |b, c| if c.1 > b.1 { c } else { b });
    (v > 0.0).then_some(ch)
}

/// phy `_overlap_transform`: the `[start, end]` fraction of a box's width for
/// slot `slot` of `n` side-by-side clusters, with a gap of `width / SLOT_K`.
pub(crate) fn overlap_slot(slot: usize, n: usize) -> (f32, f32) {
    if n <= 1 {
        return (0.0, 1.0);
    }
    let w = 1.0 / (n as f32 + (n - 1) as f32 / SLOT_K);
    let a = slot as f32 * w * (1.0 + 1.0 / SLOT_K);
    (a, a + w)
}

/// `w` / `shift+w`: next mode, skipping `Templates` without templates.
pub(crate) fn next_mode(mode: WaveformMode, dir: i32, has_templates: bool) -> WaveformMode {
    use WaveformMode::*;
    const ORDER: [WaveformMode; 3] = [Waveforms, Mean, Templates];
    let mut i = ORDER.iter().position(|&m| m == mode).unwrap_or(0) as i32;
    loop {
        i = (i + dir).rem_euclid(3);
        if has_templates || ORDER[i as usize] != Templates {
            return ORDER[i as usize];
        }
    }
}

/// Consume one exact `key` press (ctrl = command), so nothing downstream
/// (the global keymap) sees it.
/// Pane-local keys (only called while the pane is hovered). Returns true
/// when `R` asked for a reset of the scaling.
fn handle_keys(ctx: &egui::Context, p: &mut WaveformParams, has_templates: bool) -> bool {
    use crate::shortcuts::{Action, Scope, WaveformAction as W};
    const STEP: f32 = 1.1;
    let mut reset = false;
    let actions = crate::settings::consume(ctx, Scope::Waveforms);
    let has = |a| actions.contains(&Action::Waveform(a));
    if has(W::Previous) {
        p.mode = next_mode(p.mode, -1, has_templates);
    }
    if has(W::Next) {
        p.mode = next_mode(p.mode, 1, has_templates);
    }
    // phy `toggle_mean_waveforms`: mean-only <-> spikes with the mean on top.
    if has(W::Mean) {
        p.mode = if p.mode == WaveformMode::Mean {
            WaveformMode::Waveforms
        } else {
            WaveformMode::Mean
        };
    }
    if has(W::Overlap) {
        p.overlap = !p.overlap;
    }
    if has(W::Labels) {
        p.show_labels = !p.show_labels;
    }
    let scale =
        |v: &mut f32, up: bool| *v = (*v * if up { STEP } else { 1.0 / STEP }).clamp(0.05, 50.0);
    for (amplitude, spread, up) in [
        (W::AmplitudeUp, W::SpreadUp, true),
        (W::AmplitudeDown, W::SpreadDown, false),
    ] {
        if has(amplitude) {
            scale(&mut p.amp_scale, up);
        }
        if has(spread) {
            scale(&mut p.probe_scale[1], up);
        }
    }
    for (box_width, spread, up) in [
        (W::BoxWider, W::SpreadWider, true),
        (W::BoxNarrower, W::SpreadNarrower, false),
    ] {
        if has(box_width) {
            scale(&mut p.box_scale[0], up);
        }
        if has(spread) {
            scale(&mut p.probe_scale[0], up);
        }
    }
    if has(W::Reset) {
        p.box_scale = [1.0; 2];
        p.probe_scale = [1.0; 2];
        p.amp_scale = 1.0;
        reset = true;
    }
    reset
}

/// How the Features link marks one channel box.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BoxMark {
    /// `A` / `B` badge: the first / second channel of the link's set.
    pub badge: Option<char>,
    /// Member of the channel set (outlined).
    pub in_set: bool,
    /// One of the focused feature plot's channels (strong outline).
    pub highlighted: bool,
    /// Some channels are highlighted and this is not one of them.
    pub dimmed: bool,
}

pub(crate) fn box_mark(link: &FeatureLink, ch: u32) -> BoxMark {
    let highlighted = link.highlight.contains(&ch);
    BoxMark {
        badge: match link.set.iter().position(|&c| c == ch) {
            Some(0) => Some('A'),
            Some(1) => Some('B'),
            _ => None,
        },
        in_set: link.set.contains(&ch),
        highlighted,
        dimmed: !link.highlight.is_empty() && !highlighted,
    }
}

/// Toolbar hint: the channel set and the gestures that edit it.
/// What the Features overlays on the highlighted channels mean.
fn link_legend(link: &FeatureLink, pc_shapes: bool) -> String {
    let mut parts = vec!["outlined: the focused feature plot's channels (others dimmed)"];
    if !link.spikes.is_empty() {
        parts.push("thin white / black: the lassoed or k-means spikes");
    }
    if pc_shapes && !link.basis.is_empty() {
        parts.push("dashed: PC1 shape · dotted: PC2 shape");
    }
    parts.join("\n")
}

pub(crate) fn link_hint(set: &[u32]) -> String {
    let gestures = "Shift+click toggles · Ctrl+click A · Ctrl+right-click B";
    if set.is_empty() {
        return gestures.into();
    }
    let chans: Vec<String> = set.iter().map(u32::to_string).collect();
    format!("set ch {} · {gestures}", chans.join(" · "))
}

/// Per-channel mean of `snips` (`[sample * nc + channel]` each); zeros when
/// there are none.
fn mean_of<'a>(
    snips: impl Iterator<Item = &'a Vec<f32>>,
    nc: usize,
    snippet_len: usize,
) -> Vec<Vec<f32>> {
    let mut sum = vec![vec![0.0f64; snippet_len]; nc];
    let mut count = 0usize;
    for snip in snips {
        count += 1;
        for (t, channel_values) in snip.chunks_exact(nc.max(1)).enumerate().take(snippet_len) {
            for (ch, channel_sum) in sum.iter_mut().enumerate() {
                channel_sum[t] += channel_values[ch] as f64;
            }
        }
    }
    let count = count.max(1) as f64;
    sum.into_iter()
        .map(|row| row.into_iter().map(|v| (v / count) as f32).collect())
        .collect()
}

// ---- Drawing -----------------------------------------------------------

/// Multi-cluster waveform view. Shows all selected clusters on one shared
/// channel set (four channels by default), at their probe positions or in a
/// simple grid when no geometry is available. Caches the extracted
/// snippets so cluster-switching stalls happen at most once per change.
/// Clicks on a channel box queue a [`ViewAction::Channel`]: Shift+click
/// toggles it in the Features channel set, Ctrl+click / Ctrl+right-click set
/// channel A / B. `link` is the Features pane's cross-talk state (see
/// [`FeatureLink`]): it marks the set, dims all but the focused plot's
/// channels, and adds the linked spikes and PC shapes.
#[allow(clippy::too_many_arguments)]
pub fn waveform_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    geometry: &[[f32; 2]],
    shanks: &[u32],
    cache: &mut WaveformCache,
    params: &mut WaveformParams,
    link: &FeatureLink,
    brushes: &crate::brush::Brushes,
    actions: &mut Vec<ViewAction>,
) {
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }

    let sr = session.provider.sample_rate().max(1.0);
    let pre = waveform_window_samples(sr);
    let post = pre;
    let snippet_len = (pre + post + 1) as usize;
    let nc = session.provider.n_channels() as usize;
    if nc == 0 || snippet_len < 3 {
        ui.label("no channels");
        return;
    }

    // Pane-local keys, consumed before anything downstream can see them; the
    // global keymap binds none of these chords (plain m/w/o/r, ctrl+l, ctrl
    // and shift + arrows), so there is no clash even for unhovered panes.
    let has_templates = session.has_template_waveforms();
    if ui.rect_contains_pointer(ui.max_rect()) && !ui.ctx().egui_wants_keyboard_input() {
        let pinch = ui.input(|i| i.zoom_delta());
        if pinch != 1.0 {
            // egui reports Ctrl+wheel as a zoom gesture.
            params.box_scale[1] = (params.box_scale[1] * pinch).clamp(0.05, 50.0);
        }
        if handle_keys(ui.ctx(), params, has_templates) {
            cache.frozen = None;
        }
    }
    let mode = if params.mode == WaveformMode::Templates && !has_templates {
        WaveformMode::Waveforms
    } else {
        params.mode
    };

    cache.set_n_spikes(params.n_spikes);
    cache.ensure(session, selection, pre, post);
    cache.ensure_linked(session, link, pre, post);
    // Spike brush: the split what-if's inside part (lasso / k-means) or the
    // ISI violators of every selected cluster, drawn as ink; and the merge
    // ghost's mean.
    // Only a brush of this curation version: its indices are bucket
    // positions (`Brushes::sync` drops older ones at the start of a frame).
    let current = brushes.synced.is_none_or(|v| v == session.version());
    let brush = brushes
        .spikes
        .as_ref()
        .filter(|b| b.version == session.version());
    let whatif =
        crate::isi_view::split_target(brushes, selection, |c| session.spike_times(c).len())
            .filter(|_| brush.is_some());
    let brushed: Vec<_> = match (brush, &whatif) {
        (_, Some((c, idx))) => vec![(*c, std::sync::Arc::clone(idx))],
        (Some(b), None) if b.source == crate::brush::BrushSource::IsiViolators => b
            .spikes
            .iter()
            .filter(|(c, _)| selection.contains(c))
            .cloned()
            .collect(),
        _ => Vec::new(),
    };
    cache.ensure_brushed(session, &brushed, pre, post);
    // The spike hovered in Features / Amplitudes / ..., when it is one of
    // the selected clusters' (bold accent).
    let hovered = brushes
        .hover
        .filter(|h| current && selection.contains(&h.cluster))
        .map(|h| (h.cluster, h.local));
    cache.ensure_hovered(session, hovered, pre, post);
    match (brush, &whatif) {
        (Some(b), Some((c, _))) if selection.contains(c) => {
            let slot = selection.iter().position(|x| x == c).unwrap_or(0);
            cache.ensure_whatif(session, b, *c, slot, nc, snippet_len);
        }
        _ => cache.whatif = None,
    }
    let ghost = match brushes.ghost {
        Some(crate::brush::Ghost::Merge(g)) if !selection.contains(&g) => Some(g),
        _ => None,
    };
    cache.ensure_ghost(session, ghost, pre, post);
    let geometry_available = geometry.len() == nc;
    params.max_channels = params.max_channels.clamp(1, nc);
    let peak = cache
        .templates
        .first()
        .and_then(|t| t.first())
        .and_then(|&(tid, _)| session.template_waveform(tid))
        .and_then(|w| template_peak(w, session.template_shape().2))
        .filter(|&ch| ch < nc)
        .or_else(|| cache.peak_channel(0))
        .unwrap_or(0);
    let chans = display_channels(nc, peak, geometry, shanks, params.max_channels);

    let theme = Theme::from_ui(ui);
    let hp = if cache.cfg.hp().is_some() {
        "filtered"
    } else {
        "raw"
    };
    let ms = pre as f32 / sr * 1000.0;
    let head = format!(
        "{} ch · ±{ms:.1} ms{}",
        chans.len(),
        if geometry_available {
            ""
        } else {
            " · no probe geometry"
        }
    );
    // V1: one-row bar. `overlap` (phy `o`) checked (default) superimposes
    // every cluster in the cell; unchecked is phy's side-by-side slots.
    let compact = ui.available_width() < 300.0;
    let (mut overlap, mut pc_shapes) = (params.overlap, params.pc_shapes);
    crate::views::param_bar(
        ui,
        &head,
        |ui| {
            ui.add(
                egui::DragValue::new(&mut params.n_spikes)
                    .range(1..=2000)
                    .speed(1.0)
                    .suffix(" spikes"),
            )
            .on_hover_text("spikes sampled per cluster (phy n_spikes_waveforms)");
            if !compact {
                ui.checkbox(&mut params.overlap, "overlap")
                    .on_hover_text(format!(
                        "{}: superimpose the selected clusters (off = side-by-side slots)",
                        crate::settings::hint(ui.ctx(), "waveforms.overlap")
                    ));
                ui.checkbox(&mut params.pc_shapes, "PC shapes")
                    .on_hover_text(
                        "dashed PC1 / dotted PC2 temporal shape of the focused Features plot's channels",
                    );
            }
        },
        |ui| {
            ui.weak(format!("source: {hp} (follows the trace pane)"));
            if compact {
                ui.checkbox(&mut overlap, "overlap");
                ui.checkbox(&mut pc_shapes, "PC shapes");
            }
            if ui
                .add(
                    egui::DragValue::new(&mut params.max_channels)
                        .range(1..=nc)
                        .suffix(" channels"),
                )
                .on_hover_text("Total channels for all selected clusters; default 4 for a tetrode")
                .changed()
            {
                cache.frozen = None;
            }
            ui.selectable_value(&mut params.mode, WaveformMode::Waveforms, "spikes + mean");
            ui.selectable_value(&mut params.mode, WaveformMode::Mean, "mean");
            if has_templates {
                ui.selectable_value(&mut params.mode, WaveformMode::Templates, "templates");
            }
            ui.checkbox(&mut params.show_labels, "channel labels")
                .on_hover_text(crate::settings::hint(ui.ctx(), "waveforms.labels"));
        },
    );
    if compact {
        params.overlap = overlap;
        params.pc_shapes = pc_shapes;
    }
    ui.add(egui::Label::new(egui::RichText::new(link_hint(&link.set)).small().weak()).truncate())
        .on_hover_text(link_legend(link, params.pc_shapes));
    // V6.2 legend: swatch + cluster id per selection slot.
    ui.horizontal_wrapped(|ui| {
        for (slot, c) in selection.iter().enumerate() {
            let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), Sense::hover());
            ui.painter().rect_filled(r, 2.0, selection_colour(slot));
            ui.label(egui::RichText::new(format!("c{c}")).size(TICK_PX));
        }
    });

    let mut ctx = Ctx {
        session,
        selection,
        cache,
        theme: &theme,
        params,
        link,
        accent: ui.visuals().hyperlink_color,
        mode,
        nc,
        snippet_len,
        sr,
    };
    let fallback;
    let positions = if geometry_available {
        geometry
    } else {
        let mut positions = vec![[0.0, 0.0]; nc];
        for (slot, &ch) in chans.iter().enumerate() {
            positions[ch] = [(slot % 2) as f32 * 32.0, (slot / 2) as f32 * 20.0];
        }
        fallback = positions;
        &fallback
    };
    if let Some(edit) = ctx.probe_layout(ui, positions, &chans) {
        actions.push(ViewAction::Channel(edit));
    }
}

/// Everything the drawing helpers read.
struct Ctx<'a, P: DataProvider> {
    session: &'a Session<P>,
    selection: &'a [ClusterId],
    cache: &'a mut WaveformCache,
    theme: &'a Theme,
    params: &'a WaveformParams,
    link: &'a FeatureLink,
    /// PC shape colour (`hyperlink_color`).
    accent: Color32,
    mode: WaveformMode,
    nc: usize,
    snippet_len: usize,
    sr: f32,
}

impl<P: DataProvider> Ctx<'_, P> {
    /// Signed `(lo, hi)` range for the mode, frozen on first use.
    fn range(&mut self, chans: &[usize]) -> (f32, f32) {
        let (mode, nc, snippet_len, session) = (self.mode, self.nc, self.snippet_len, self.session);
        self.cache.frozen_range(mode, nc, |cache| match mode {
            WaveformMode::Waveforms => signed_range(cache, chans, nc, snippet_len, true),
            WaveformMode::Mean => signed_range(cache, chans, nc, snippet_len, false),
            WaveformMode::Templates => {
                let (_, ns, nch) = session.template_shape();
                let (mut lo, mut hi) = (0.0f32, 0.0f32);
                for &(tid, _) in cache.templates.iter().flatten() {
                    let w = session.template_waveform(tid).unwrap_or(&[]);
                    for row in w.chunks_exact(nch.max(1)).take(ns) {
                        for &ch in chans {
                            let v = row.get(ch).copied().unwrap_or(0.0);
                            lo = lo.min(v);
                            hi = hi.max(v);
                        }
                    }
                }
                pad_range(lo, hi, 0.08)
            }
        })
    }

    /// Returns the channel edit a click on a box asks for.
    fn probe_layout(
        &mut self,
        ui: &mut Ui,
        geometry: &[[f32; 2]],
        chans: &[usize],
    ) -> Option<ChannelEdit> {
        let theme = self.theme;
        let peaks = channel_peak_abs(&self.cache.means, self.nc);
        if chans.iter().all(|&c| peaks[c] <= 0.0) && self.mode != WaveformMode::Templates {
            ui.label("no waveforms in range — try expanding the selection");
            return None;
        }
        let p = self.params;

        // Bounding box of the selected channel positions.
        let (mut x_min, mut x_max) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut y_min, mut y_max) = (f32::INFINITY, f32::NEG_INFINITY);
        for &ch in chans {
            let g = geometry[ch];
            x_min = x_min.min(g[0]);
            x_max = x_max.max(g[0]);
            y_min = y_min.min(g[1]);
            y_max = y_max.max(g[1]);
        }
        let x_span = (x_max - x_min).max(1.0);
        let y_span = (y_max - y_min).max(1.0);

        // Signed amplitude range over the picked channels, so spikes use the
        // whole cell instead of a symmetric ±max.
        let (lo, hi) = self.range(chans);

        let (rect, resp) = alloc_plot(ui, egui::vec2(120.0, 220.0), Sense::click());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, theme.plot_bg);

        // Per-subplot box size: divide the panel into a grid sized to the
        // probe's aspect ratio while keeping each subplot at least 60×40px.
        // 8 px margins; a 30 px strip at the bottom holds the scale bar.
        let panel_w = rect.width() - 16.0;
        let panel_h = rect.height() - 38.0;
        // Sub-rect width/height proportional to the spacing between picked
        // channels — roughly half the smallest x-gap and y-gap.
        let (mut min_dx, mut min_dy) = (x_span, y_span);
        for (i, &a) in chans.iter().enumerate() {
            for &b in &chans[i + 1..] {
                let dx = (geometry[a][0] - geometry[b][0]).abs();
                let dy = (geometry[a][1] - geometry[b][1]).abs();
                if dx > 0.5 && dx < min_dx {
                    min_dx = dx;
                }
                if dy > 0.5 && dy < min_dy {
                    min_dy = dy;
                }
            }
        }
        let sub_w = ((min_dx / x_span) * panel_w).clamp(60.0, panel_w * 0.45) * p.box_scale[0];
        let sub_h = ((min_dy / y_span) * panel_h).clamp(40.0, panel_h * 0.45) * p.box_scale[1];

        let scale_x = (panel_w - sub_w).max(1.0) / x_span * p.probe_scale[0];
        let scale_y = (panel_h - sub_h).max(1.0) / y_span * p.probe_scale[1];
        let centre = Pos2::new(rect.center().x, rect.top() + 8.0 + panel_h * 0.5);
        let (x_mid, y_mid) = ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5);

        let mut boxes: Vec<(usize, egui::Rect)> = Vec::with_capacity(chans.len());
        let present: Vec<usize> = (0..self.selection.len()).collect();
        for &ch in chans {
            let g = geometry[ch];
            // y is flipped (probes have y increasing upward; egui downward).
            let c = Pos2::new(
                centre.x + (g[0] - x_mid) * scale_x,
                centre.y - (g[1] - y_mid) * scale_y,
            );
            let sub = egui::Rect::from_center_size(c, egui::vec2(sub_w, sub_h));
            boxes.push((ch, sub));
            ui.interact(
                sub.intersect(rect),
                ui.id().with(("waveform-channel", ch)),
                Sense::hover(),
            )
            .widget_info(|| {
                egui::WidgetInfo::labeled(
                    egui::WidgetType::Other,
                    true,
                    format!("waveform channel {ch}"),
                )
            });
            painter.rect_filled(sub, 0.0, theme.panel);
            painter.rect_stroke(
                sub,
                0.0,
                Stroke::new(0.5_f32, grid_colour(theme)),
                egui::StrokeKind::Inside,
            );
            let mark = box_mark(self.link, ch as u32);
            self.draw_box(&painter, sub, ch, &present, p.overlap, (lo, hi), 1.5, mark);
            if p.show_labels {
                // Corner chip over a translucent backing, clear of the traces;
                // weak when another channel is focused.
                let col = if mark.dimmed {
                    theme.caption.gamma_multiply(0.5)
                } else {
                    theme.caption
                };
                let g = painter.layout_no_wrap(
                    format!("ch{ch}"),
                    egui::FontId::proportional(TICK_PX),
                    col,
                );
                let chip =
                    egui::Rect::from_min_size(sub.left_top(), g.size() + egui::vec2(6.0, 2.0));
                painter.rect_filled(chip, 2.0, theme.panel.gamma_multiply(0.85));
                painter.galley(chip.min + egui::vec2(3.0, 1.0), g, col);
            }
            self.draw_link_frame(&painter, sub, mark, ui.visuals().selection);
        }

        // One shared scale bar for every tile (same x and y scale).
        let tile_y_scale = sub_h / (hi - lo) * p.amp_scale;
        let amp = nice_ticks(0.0, (hi - lo) * 0.5 / p.amp_scale, 2)
            .last()
            .copied()
            .filter(|&a| a > 0.0)
            .unwrap_or((hi - lo) * 0.25 / p.amp_scale);
        let slot_w = if p.overlap {
            1.0
        } else {
            overlap_slot(0, present.len()).1
        };
        let ms_px = sub_w * slot_w / (self.snippet_len as f32 - 1.0).max(1.0) * self.sr * 0.001;
        scale_bar(
            &painter,
            Pos2::new(rect.left() + 10.0, rect.bottom() - 8.0),
            ms_px,
            amp * tile_y_scale,
            &format!("1 ms / {} amp", tick_label(amp, amp)),
            theme,
        );

        // Shift+click toggles a box in the Features channel set; Ctrl+click /
        // Ctrl+right-click set its first / second channel (phy `select_channel`).
        let (left, right) = (
            resp.clicked_by(egui::PointerButton::Primary),
            resp.clicked_by(egui::PointerButton::Secondary),
        );
        let m = ui.input(|i| i.modifiers);
        let edit: fn(u32) -> ChannelEdit = if m.command && right {
            ChannelEdit::SetB
        } else if m.command && left {
            ChannelEdit::SetA
        } else if m.shift && left {
            ChannelEdit::Toggle
        } else {
            return None;
        };
        let pos = resp.interact_pointer_pos()?;
        let (ch, _) = boxes.iter().find(|(_, r)| r.contains(pos))?;
        Some(edit(*ch as u32))
    }

    /// Outlines and A/B badge for a box's role in the Features link.
    fn draw_link_frame(
        &self,
        painter: &egui::Painter,
        sub: egui::Rect,
        mark: BoxMark,
        sel: egui::style::Selection,
    ) {
        use egui::StrokeKind::Inside;
        if mark.in_set {
            painter.rect_stroke(sub, 0.0, Stroke::new(1.5_f32, sel.bg_fill), Inside);
        }
        if mark.highlighted {
            painter.rect_stroke(sub, 0.0, Stroke::new(2.0_f32, self.theme.text), Inside);
        }
        if let Some(letter) = mark.badge {
            let badge = egui::Rect::from_min_size(
                sub.right_top() + egui::vec2(-15.0, 3.0),
                egui::vec2(12.0, 12.0),
            );
            painter.rect_filled(badge, 2.0, sel.bg_fill);
            painter.text(
                badge.center(),
                egui::Align2::CENTER_CENTER,
                letter,
                egui::FontId::proportional(TICK_PX),
                sel.stroke.color,
            );
        }
    }

    /// One channel box: baseline, 1 ms ticks, then the clusters in `present`
    /// (selection slots) — superimposed if `overlap`, else side by side. Spike clouds come
    /// first, means on top.
    #[allow(clippy::too_many_arguments)]
    fn draw_box(
        &self,
        painter: &egui::Painter,
        sub: egui::Rect,
        ch: usize,
        present: &[usize],
        overlap: bool,
        (lo, hi): (f32, f32),
        mean_width: f32,
        mark: BoxMark,
    ) {
        let theme = self.theme;
        // Cluster curves fade when another channel is the focused plot's.
        let fade = if mark.dimmed { 0.4 } else { 1.0 };
        let painter = painter.with_clip_rect(sub);
        let y_unit = sub.height() / (hi - lo);
        let base = sub.bottom() + lo * y_unit;
        let y_scale = y_unit * self.params.amp_scale;
        painter.line_segment(
            [Pos2::new(sub.left(), base), Pos2::new(sub.right(), base)],
            Stroke::new(0.5_f32, grid_colour(theme)),
        );
        let n = present.len();
        let span = |k: usize| {
            if overlap {
                (0.0, 1.0)
            } else {
                overlap_slot(k, n)
            }
        };
        let map = |k: usize, frac: f32, v: f32| {
            let (a, b) = span(k);
            let w = sub.width();
            Pos2::new(sub.left() + w * (a + (b - a) * frac), base - v * y_scale)
        };

        // 1 ms ticks along the baseline of every slot.
        let pre = self.snippet_len / 2;
        let per_ms = self.sr * 0.001;
        let last = (self.snippet_len as f32 - 1.0).max(1.0);
        for k in 0..n.max(1) {
            let mut ms = -((pre as f32 / per_ms).floor());
            while pre as f32 + ms * per_ms <= last {
                let x = map(k, (pre as f32 + ms * per_ms) / last, 0.0).x;
                painter.line_segment(
                    [Pos2::new(x, sub.bottom()), Pos2::new(x, sub.bottom() - 3.0)],
                    Stroke::new(0.75_f32, grid_colour(theme)),
                );
                ms += 1.0;
            }
        }

        let alpha = if theme.dark { 55 } else { 45 };
        let cache = &*self.cache;
        if self.mode == WaveformMode::Templates {
            let (_, ns, nch) = self.session.template_shape();
            for (k, &ci) in present.iter().enumerate() {
                let c = selection_colour(ci);
                // Smallest share first so the dominant template ends on top.
                for &(tid, share) in cache.templates.get(ci).into_iter().flatten().rev() {
                    let Some(w) = self.session.template_waveform(tid) else {
                        continue;
                    };
                    let a = (255.0 * (0.2 + 0.8 * share) * fade) as u8;
                    let pts: Vec<Pos2> = (0..ns)
                        .map(|s| map(k, s as f32 / (ns as f32 - 1.0).max(1.0), w[s * nch + ch]))
                        .collect();
                    painter.add(egui::Shape::line(
                        pts,
                        Stroke::new(
                            mean_width,
                            Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), a),
                        ),
                    ));
                }
            }
            return;
        }
        if self.mode == WaveformMode::Waveforms {
            for (k, &ci) in present.iter().enumerate() {
                let c = selection_colour(ci);
                let faint = Stroke::new(
                    0.75_f32,
                    Color32::from_rgba_unmultiplied(
                        c.r(),
                        c.g(),
                        c.b(),
                        (alpha as f32 * fade) as u8,
                    ),
                );
                for snip in cache.snippets.get(ci).into_iter().flatten() {
                    let pts: Vec<Pos2> = (0..self.snippet_len)
                        .map(|t| map(k, t as f32 / last, snip[t * self.nc + ch]))
                        .collect();
                    painter.add(egui::Shape::line(pts, faint));
                }
            }
        }
        // Lassoed / k-means spikes from the Features pane, under the means;
        // only on the focused plot's channels when there is one.
        if !mark.dimmed {
            let ink = if theme.dark {
                Color32::from_white_alpha(140)
            } else {
                Color32::from_black_alpha(140)
            };
            let thin = Stroke::new(0.8_f32, ink);
            for snip in self.cache.linked.iter().flat_map(|l| &l.snips) {
                let pts: Vec<Pos2> = (0..self.snippet_len)
                    .map(|t| {
                        let x = sub.left() + sub.width() * t as f32 / last;
                        Pos2::new(x, base - snip[t * self.nc + ch] * y_scale)
                    })
                    .collect();
                painter.add(egui::Shape::line(pts, thin));
            }
        }
        // Brushed spikes (ISI violators; the split's inside part draws as
        // the dashed mean below), thin ink under the means.
        if !self.cache.whatif_on() {
            let ink = if theme.dark {
                Color32::from_white_alpha(200)
            } else {
                Color32::from_black_alpha(200)
            };
            for snip in self.cache.brushed.iter().flat_map(|l| &l.snips) {
                let pts: Vec<Pos2> = (0..self.snippet_len)
                    .map(|t| {
                        let x = sub.left() + sub.width() * t as f32 / last;
                        Pos2::new(x, base - snip[t * self.nc + ch] * y_scale)
                    })
                    .collect();
                painter.add(egui::Shape::line(pts, Stroke::new(0.8_f32, ink)));
            }
        }
        // The spike hovered in another view: one bold accent waveform.
        for snip in self.cache.hovered.iter().flat_map(|l| &l.snips) {
            let pts: Vec<Pos2> = (0..self.snippet_len)
                .map(|t| {
                    let x = sub.left() + sub.width() * t as f32 / last;
                    Pos2::new(x, base - snip[t * self.nc + ch] * y_scale)
                })
                .collect();
            painter.add(egui::Shape::line(pts, Stroke::new(2.0_f32, self.accent)));
        }
        for (k, &ci) in present.iter().enumerate() {
            let Some(mean) = cache.means.get(ci).and_then(|m| m.get(ch)) else {
                continue;
            };
            let pts: Vec<Pos2> = mean
                .iter()
                .enumerate()
                .map(|(t, &v)| map(k, t as f32 / last, v))
                .collect();
            painter.add(egui::Shape::line(
                pts,
                Stroke::new(mean_width, selection_colour(ci).gamma_multiply(fade)),
            ));
            // Split what-if: inside dashed, outside dotted, same colour.
            if let Some((_, slot, [inside, outside])) = &cache.whatif {
                if *slot == ci {
                    let col = selection_colour(ci).gamma_multiply(fade);
                    let line = |m: &Vec<Vec<f32>>| -> Vec<Pos2> {
                        m.get(ch)
                            .map(|m| {
                                m.iter()
                                    .enumerate()
                                    .map(|(t, &v)| map(k, t as f32 / last, v))
                                    .collect()
                            })
                            .unwrap_or_default()
                    };
                    painter.extend(egui::Shape::dashed_line(
                        &line(inside),
                        Stroke::new(mean_width, col),
                        5.0,
                        3.0,
                    ));
                    painter.extend(egui::Shape::dotted_line(
                        &line(outside),
                        col,
                        3.0,
                        mean_width * 0.5,
                    ));
                }
            }
            // Merge ghost: its mean, grey and dashed, in every slot.
            if let Some(gm) = cache.ghost_mean().and_then(|m| m.get(ch)) {
                let pts: Vec<Pos2> = gm
                    .iter()
                    .enumerate()
                    .map(|(t, &v)| map(k, t as f32 / last, v))
                    .collect();
                painter.extend(egui::Shape::dashed_line(
                    &pts,
                    Stroke::new(
                        mean_width,
                        Color32::from_gray(if theme.dark { 150 } else { 120 }),
                    ),
                    5.0,
                    3.0,
                ));
            }
        }
        // Hundreds of overlapping spikes saturate even at `fade`, so a veil
        // of the plot background is what makes a dimmed box read as dimmed.
        if mark.dimmed {
            painter.rect_filled(sub, 0.0, theme.plot_bg.gamma_multiply(0.6));
        }
        // The focused feature plot's PC1 (dashed) / PC2 (dotted) shapes on
        // its own channels, scaled so the larger peak fills ~0.6 of the box.
        if mark.highlighted && self.params.pc_shapes {
            let basis = self.link.basis.iter().find(|(c, _)| *c as usize == ch);
            let Some((_, pcs)) = basis else { return };
            let peak = pcs
                .iter()
                .flatten()
                .fold(0.0f32, |m, &v| m.max(v.abs()))
                .max(f32::MIN_POSITIVE);
            let colour = self.accent;
            for (pc, (dash, gap)) in pcs.iter().zip([(4.0, 3.0), (1.0, 3.0)]) {
                // Stretched over the box whatever the basis length.
                let n = pc.len();
                let pts: Vec<Pos2> = pc
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let frac = i as f32 / (n as f32 - 1.0).max(1.0);
                        Pos2::new(
                            sub.left() + sub.width() * frac,
                            base - v / peak * 0.3 * sub.height(),
                        )
                    })
                    .collect();
                painter.extend(egui::Shape::dashed_line(
                    &pts,
                    Stroke::new(1.2_f32, colour),
                    dash,
                    gap,
                ));
            }
        }
    }
}

/// L-shaped scale bar with its label to the right.
pub(crate) fn scale_bar(
    painter: &egui::Painter,
    origin: Pos2,
    w: f32,
    h: f32,
    label: &str,
    theme: &Theme,
) {
    painter.add(egui::Shape::line(
        vec![
            Pos2::new(origin.x, origin.y - h),
            origin,
            Pos2::new(origin.x + w, origin.y),
        ],
        Stroke::new(1.5_f32, theme.text),
    ));
    painter.text(
        Pos2::new(origin.x + w + 8.0, origin.y),
        egui::Align2::LEFT_BOTTOM,
        label,
        egui::FontId::proportional(TICK_PX),
        theme.caption,
    );
}

/// Signed `(lo, hi)` amplitude range (always spanning 0, 8% padded) over
/// the means (and snippets when `snips`) of `chans`, so spikes use the whole
/// cell.
pub(crate) fn signed_range(
    cache: &WaveformCache,
    chans: &[usize],
    nc: usize,
    snippet_len: usize,
    snips: bool,
) -> (f32, f32) {
    let (mut lo, mut hi) = (0.0f32, 0.0f32);
    for cluster_means in &cache.means {
        for &ch in chans {
            for &v in cluster_means.get(ch).map_or(&[][..], Vec::as_slice) {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    if snips {
        for cluster_snips in &cache.snippets {
            for snip in cluster_snips {
                for cv in snip.chunks_exact(nc).take(snippet_len) {
                    for &ch in chans {
                        lo = lo.min(cv[ch]);
                        hi = hi.max(cv[ch]);
                    }
                }
            }
        }
    }
    pad_range(lo, hi, 0.08)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spread_covers_the_whole_list() {
        let idx: Vec<u32> = (0..150).collect();
        let got: Vec<u32> = spread(&idx, 100).collect();
        assert_eq!(got.len(), 100);
        assert!(got.windows(2).all(|w| w[0] < w[1]), "ordered, distinct");
        assert!(
            got[0] == 0 && got[99] >= 145,
            "reaches the tail: {}",
            got[99]
        );
        let short: Vec<u32> = spread(&idx[..7], 100).collect();
        assert_eq!(short, idx[..7]);
        assert_eq!(spread(&[], 100).count(), 0);
    }

    #[test]
    fn a_link_from_an_older_version_yields_no_spikes() {
        let link = FeatureLink {
            spikes: vec![(ClusterId(1), std::sync::Arc::from([4u32, 5]))],
            version: 3,
            ..Default::default()
        };
        assert_eq!(link_spikes(&link, 3).len(), 1);
        assert!(link_spikes(&link, 4).is_empty(), "after a curation edit");
    }

    #[test]
    fn nearest_channels_stay_on_the_shank() {
        // Two shanks of 20 channels; same y, shank 1 shifted in x so its
        // channels are closer to some of shank 0's than shank 0's own.
        let geometry: Vec<[f32; 2]> = (0..40)
            .map(|c| [if c < 20 { 0.0 } else { 1.0 }, (c % 20) as f32 * 20.0])
            .collect();
        let shanks: Vec<u32> = (0..40).map(|c| (c >= 20) as u32).collect();
        let got = nearest_channels(10, &geometry, &shanks, 12);
        assert_eq!(got.len(), 12, "phylib counts the peak among the n");
        assert!(got.contains(&10) && got.iter().all(|&c| c < 20));
        // Channel 10 plus 11 neighbours on a line; ch4 beats ch16 on the
        // 120-away tie by index.
        assert_eq!(got, (4..=15).collect::<Vec<_>>());
        // Without shank info the other shank's ch30 (1 away) is nearest, then
        // ch9 wins the 20-away tie by index.
        let flat = nearest_channels(10, &geometry, &[], 3);
        assert_eq!(flat, vec![9, 10, 30]);
        // Edge of the shank: fewer neighbours than asked.
        assert_eq!(nearest_channels(0, &geometry[..3], &[], 12), vec![0, 1, 2]);
    }

    #[test]
    fn link_marks_badges_outlines_and_dimming() {
        let none = FeatureLink::default();
        assert_eq!(box_mark(&none, 3), BoxMark::default());
        let link = FeatureLink {
            set: vec![3, 5, 7],
            highlight: vec![3, 5],
            ..Default::default()
        };
        let (a, b, c, other) = (
            box_mark(&link, 3),
            box_mark(&link, 5),
            box_mark(&link, 7),
            box_mark(&link, 1),
        );
        assert_eq!((a.badge, b.badge, c.badge), (Some('A'), Some('B'), None));
        assert!(a.highlighted && b.highlighted && !c.highlighted);
        assert!(c.in_set && c.dimmed && !a.dimmed);
        assert!(!other.in_set && other.dimmed && other.badge.is_none());
        // Without a focused plot nothing is dimmed.
        let unfocused = FeatureLink {
            highlight: vec![],
            ..link
        };
        assert!(!box_mark(&unfocused, 1).dimmed);
    }

    #[test]
    fn hint_lists_the_set_then_the_gestures() {
        assert_eq!(
            link_hint(&[3, 5, 7, 11]),
            "set ch 3 · 5 · 7 · 11 · Shift+click toggles · Ctrl+click A · Ctrl+right-click B"
        );
        assert!(link_hint(&[]).starts_with("Shift+click"));
    }

    #[test]
    fn a_tetrode_shows_all_four_channels_with_or_without_geometry() {
        let geometry = [[0.0, 0.0], [32.0, 0.0], [0.0, 20.0], [32.0, 20.0]];
        let limit = WaveformParams::default().max_channels;
        assert_eq!(limit, 4);
        for peak in 0..4 {
            assert_eq!(
                display_channels(4, peak, &geometry, &[], limit),
                [0, 1, 2, 3]
            );
            assert_eq!(display_channels(4, peak, &[], &[], limit), [0, 1, 2, 3]);
        }
        assert_eq!(display_channels(2, 1, &[], &[], limit), [0, 1]);
        assert!(display_channels(0, 0, &[], &[], limit).is_empty());
    }

    #[test]
    fn the_shared_channel_set_keeps_real_ids_and_the_primary_channel_group() {
        let geometry: Vec<[f32; 2]> = (0..16).map(|ch| [0.0, ch as f32 * 20.0]).collect();
        let groups: Vec<u32> = (0..16).map(|ch| ch / 4).collect();
        let limit = WaveformParams::default().max_channels;
        let channels = display_channels(16, 10, &geometry, &groups, limit);
        assert_eq!(
            channels,
            [8, 9, 10, 11],
            "never replace tetrode IDs with 0..3"
        );
        assert_eq!(display_channels(16, 10, &[], &groups, limit), channels);
    }

    #[test]
    fn larger_channel_views_are_explicit_and_still_respect_group_membership() {
        let geometry: Vec<[f32; 2]> = (0..16).map(|ch| [0.0, ch as f32 * 20.0]).collect();
        assert_eq!(display_channels(16, 8, &geometry, &[], 4).len(), 4);
        assert_eq!(display_channels(16, 8, &geometry, &[], 12).len(), 12);
        assert_eq!(display_channels(16, 8, &geometry, &[], 100).len(), 16);
        assert_eq!(display_channels(16, 8, &geometry, &[], 0), [8]);
        assert_eq!(display_channels(4, 100, &[], &[], 4), [0, 1, 2, 3]);
    }

    #[test]
    fn overlap_slots_tile_the_box_with_gaps() {
        assert_eq!(overlap_slot(0, 1), (0.0, 1.0));
        for n in 2..6 {
            let slots: Vec<_> = (0..n).map(|k| overlap_slot(k, n)).collect();
            assert_eq!(slots[0].0, 0.0);
            assert!((slots[n - 1].1 - 1.0).abs() < 1e-5);
            for w in slots.windows(2) {
                let gap = w[1].0 - w[0].1;
                // gap = slot width / k
                assert!((gap - (w[0].1 - w[0].0) / SLOT_K).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn modes_cycle_and_skip_templates() {
        use WaveformMode::*;
        assert_eq!(next_mode(Waveforms, 1, true), Mean);
        assert_eq!(next_mode(Mean, 1, true), Templates);
        assert_eq!(next_mode(Templates, 1, true), Waveforms);
        assert_eq!(next_mode(Mean, 1, false), Waveforms);
        assert_eq!(next_mode(Waveforms, -1, false), Mean);
    }

    #[test]
    fn median_baseline_zeroes_a_constant_offset() {
        // 2 channels, 5 samples: ch0 offset 3 with one spike, ch1 offset -2.
        let mut s = vec![3.0, -2.0, 3.0, -2.0, -1.0, -2.0, 3.0, -2.0, 3.0, -2.0];
        subtract_median_baseline(&mut s, 2);
        assert_eq!(s, vec![0.0, 0.0, 0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn template_peak_is_argmax_abs_channel() {
        // 2 samples x 3 channels
        assert_eq!(template_peak(&[0.0, 1.0, -0.5, 0.2, -3.0, 0.1], 3), Some(1));
        assert_eq!(template_peak(&[0.0; 6], 3), None);
    }
}

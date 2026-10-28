//! phy WaveformView: spikes, means or templates per channel at probe positions.
//!
//! Channels follow phylib `_find_best_channels` (the primary template's peak
//! channel and its nearest neighbours, 12 in all, on the same shank); modes, overlap and the
//! scaling keys follow phy v2.1's `WaveformView`. Parameters live in
//! [`WaveformParams`] so they persist with the other prefs.

use egui::{Color32, Pos2, Sense, Stroke, Ui};
use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider, SampleIndex};
use sorrel_render::{extract_snippet, snippet_window, TracePreproc};

use crate::layout::ViewAction;
use crate::prefs::{WaveformMode, WaveformParams};
use crate::theme::{selection_colour, Theme};
use crate::views::{
    alloc_plot, axes_data_rect, draw_axes_grid, grid_colour, nice_ticks, pad_range, tick_label,
    TICK_PX,
};

/// phylib's `n_closest_channels`: channels shown besides the peak.
const N_NEAREST: usize = 12;
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
        }
        self.cfg = cfg;
    }

    /// Spikes per cluster for the next [`ensure`](Self::ensure); a change
    /// invalidates the key.
    pub(crate) fn set_n_spikes(&mut self, n: usize) {
        self.n_spikes = n.max(1);
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
            if cluster_snips.is_empty() || nc == 0 {
                means.push(vec![vec![0.0; snippet_len]; nc]);
                continue;
            }
            let mut sum = vec![vec![0.0f64; snippet_len]; nc];
            let count = cluster_snips.len() as f64;
            for snip in cluster_snips {
                for (t, channel_values) in snip.chunks_exact(nc).enumerate().take(snippet_len) {
                    for (ch, channel_sum) in sum.iter_mut().enumerate().take(nc) {
                        channel_sum[t] += channel_values[ch] as f64;
                    }
                }
            }
            let cluster_means: Vec<Vec<f32>> = sum
                .into_iter()
                .map(|row| row.into_iter().map(|v| (v / count) as f32).collect())
                .collect();
            means.push(cluster_means);
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

/// Sorted union of per-cluster channel sets.
pub(crate) fn union_channels(sets: &[Vec<usize>]) -> Vec<usize> {
    let mut all: Vec<usize> = sets.iter().flatten().copied().collect();
    all.sort_unstable();
    all.dedup();
    all
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
fn take_key(i: &mut egui::InputState, key: egui::Key, command: bool, shift: bool) -> bool {
    let mut hit = false;
    i.events.retain(|e| match e {
        egui::Event::Key {
            key: k,
            pressed: true,
            modifiers: m,
            ..
        } if *k == key && m.command == command && m.shift == shift && !m.alt => {
            hit = true;
            false
        }
        _ => true,
    });
    hit
}

/// Pane-local keys (only called while the pane is hovered). Returns true
/// when `R` asked for a reset of the scaling.
fn handle_keys(i: &mut egui::InputState, p: &mut WaveformParams, has_templates: bool) -> bool {
    use egui::Key as K;
    const STEP: f32 = 1.1;
    let mut reset = false;
    if take_key(i, K::W, false, true) {
        p.mode = next_mode(p.mode, -1, has_templates);
    }
    if take_key(i, K::W, false, false) {
        p.mode = next_mode(p.mode, 1, has_templates);
    }
    // phy `toggle_mean_waveforms`: mean-only <-> spikes with the mean on top.
    if take_key(i, K::M, false, false) {
        p.mode = if p.mode == WaveformMode::Mean {
            WaveformMode::Waveforms
        } else {
            WaveformMode::Mean
        };
    }
    if take_key(i, K::O, false, false) {
        p.overlap = !p.overlap;
    }
    if take_key(i, K::L, true, false) {
        p.show_labels = !p.show_labels;
    }
    let scale =
        |v: &mut f32, up: bool| *v = (*v * if up { STEP } else { 1.0 / STEP }).clamp(0.05, 50.0);
    for (key, up) in [(K::ArrowUp, true), (K::ArrowDown, false)] {
        if take_key(i, key, true, false) {
            scale(&mut p.amp_scale, up);
        }
        if take_key(i, key, false, true) {
            scale(&mut p.probe_scale[1], up);
        }
    }
    for (key, up) in [(K::ArrowRight, true), (K::ArrowLeft, false)] {
        if take_key(i, key, true, false) {
            scale(&mut p.box_scale[0], up);
        }
        if take_key(i, key, false, true) {
            scale(&mut p.probe_scale[0], up);
        }
    }
    if take_key(i, K::R, false, false) || take_key(i, K::R, false, true) {
        p.box_scale = [1.0; 2];
        p.probe_scale = [1.0; 2];
        p.amp_scale = 1.0;
        reset = true;
    }
    reset
}

// ---- Drawing -----------------------------------------------------------

/// Multi-cluster waveform view. Shows each selected cluster's best channels
/// (see [`nearest_channels`]) at their probe positions, or a single peak
/// channel when no probe geometry is available. Caches the extracted
/// snippets so cluster-switching stalls happen at most once per change.
/// Ctrl+click on a channel box queues [`ViewAction::SelectChannel`].
#[allow(clippy::too_many_arguments)]
pub fn waveform_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    geometry: &[[f32; 2]],
    shanks: &[u32],
    cache: &mut WaveformCache,
    params: &mut WaveformParams,
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
        if ui.input_mut(|i| handle_keys(i, params, has_templates)) {
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
    let peaks = channel_peak_abs(&cache.means, nc);
    let geometry_available = geometry.len() == nc;

    // Channels per cluster, and their sorted union.
    let per_cluster: Vec<Vec<usize>> = if geometry_available {
        selection
            .iter()
            .enumerate()
            .map(|(ci, _)| {
                let tpl_peak = cache
                    .templates
                    .get(ci)
                    .and_then(|t| t.first())
                    .and_then(|&(tid, _)| session.template_waveform(tid))
                    .and_then(|w| template_peak(w, session.template_shape().2))
                    .filter(|&ch| ch < nc);
                tpl_peak
                    .or_else(|| cache.peak_channel(ci))
                    .map(|p| nearest_channels(p, geometry, shanks, N_NEAREST))
                    .unwrap_or_default()
            })
            .collect()
    } else {
        Vec::new()
    };
    let chans = union_channels(&per_cluster);

    let theme = Theme::from_ui(ui);
    let hp = if cache.cfg.hp().is_some() {
        "filtered"
    } else {
        "raw"
    };
    let ms = pre as f32 / sr * 1000.0;
    let head = if geometry_available {
        format!("{} ch · ±{ms:.1} ms", chans.len())
    } else {
        let peak = peaks
            .iter()
            .enumerate()
            .fold((0, 0.0f32), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
        format!("ch {} · ±{ms:.1} ms · no probe geometry", peak.0)
    };
    // V1: one-row bar. `overlap` (phy `o`) checked (default) superimposes
    // every cluster in the cell; unchecked is phy's side-by-side slots.
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
            ui.checkbox(&mut params.overlap, "overlap")
                .on_hover_text("o: superimpose the selected clusters (off = side-by-side slots)");
        },
        |ui| {
            ui.weak(format!("source: {hp} (follows the trace pane)"));
            ui.selectable_value(&mut params.mode, WaveformMode::Waveforms, "spikes + mean");
            ui.selectable_value(&mut params.mode, WaveformMode::Mean, "mean");
            if has_templates {
                ui.selectable_value(&mut params.mode, WaveformMode::Templates, "templates");
            }
            ui.checkbox(&mut params.show_labels, "channel labels")
                .on_hover_text("ctrl+l");
        },
    );
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
        mode,
        nc,
        snippet_len,
        sr,
    };
    if geometry_available && !chans.is_empty() {
        if let Some((channel, secondary)) = ctx.probe_layout(ui, geometry, &chans, &per_cluster) {
            actions.push(ViewAction::SelectChannel {
                channel: channel as u32,
                secondary,
            });
        }
    } else {
        ctx.single_channel(ui, &peaks);
    }
}

/// Everything the drawing helpers read.
struct Ctx<'a, P: DataProvider> {
    session: &'a Session<P>,
    selection: &'a [ClusterId],
    cache: &'a mut WaveformCache,
    theme: &'a Theme,
    params: &'a WaveformParams,
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

    fn single_channel(mut self, ui: &mut Ui, peaks: &[f32]) {
        let theme = self.theme;
        let (peak_ch, peak_val) =
            peaks.iter().enumerate().fold(
                (0usize, 0.0f32),
                |b, (i, &v)| if v > b.1 { (i, v) } else { b },
            );
        if peak_val <= 0.0 {
            ui.label("no waveforms in range — try expanding the selection");
            return;
        }
        let (lo, hi) = self.range(&[peak_ch]);
        let ms = (self.snippet_len as f32 / 2.0) / self.sr * 1000.0;
        let (rect, _resp) = alloc_plot(ui, egui::vec2(120.0, 160.0), Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, theme.plot_bg);
        let data = axes_data_rect(rect, true);
        let amp = self.params.amp_scale;
        draw_axes_grid(
            &painter,
            rect,
            (-ms, ms),
            (lo / amp, hi / amp),
            "ms",
            "amplitude",
            theme,
        );
        let all: Vec<usize> = (0..self.selection.len()).collect();
        // No geometry: superimpose every cluster on the one channel.
        self.draw_box(&painter, data, peak_ch, &all, true, (lo, hi), 2.0);
    }

    /// Returns the ctrl+clicked channel and whether it was a right click.
    fn probe_layout(
        &mut self,
        ui: &mut Ui,
        geometry: &[[f32; 2]],
        chans: &[usize],
        per_cluster: &[Vec<usize>],
    ) -> Option<(usize, bool)> {
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
        let mut max_slots = 1;
        for &ch in chans {
            let g = geometry[ch];
            // y is flipped (probes have y increasing upward; egui downward).
            let c = Pos2::new(
                centre.x + (g[0] - x_mid) * scale_x,
                centre.y - (g[1] - y_mid) * scale_y,
            );
            let sub = egui::Rect::from_center_size(c, egui::vec2(sub_w, sub_h));
            boxes.push((ch, sub));
            // Clusters whose best channels include this one.
            let present: Vec<usize> = (0..self.selection.len())
                .filter(|&ci| per_cluster[ci].contains(&ch))
                .collect();
            max_slots = max_slots.max(present.len());
            painter.rect_filled(sub, 0.0, theme.panel);
            painter.rect_stroke(
                sub,
                0.0,
                Stroke::new(0.5_f32, grid_colour(theme)),
                egui::StrokeKind::Inside,
            );
            self.draw_box(&painter, sub, ch, &present, p.overlap, (lo, hi), 1.5);
            if p.show_labels {
                // Corner chip over a translucent backing, clear of the traces.
                let g = painter.layout_no_wrap(
                    format!("ch{ch}"),
                    egui::FontId::proportional(TICK_PX),
                    theme.caption,
                );
                let chip =
                    egui::Rect::from_min_size(sub.left_top(), g.size() + egui::vec2(6.0, 2.0));
                painter.rect_filled(chip, 2.0, theme.panel.gamma_multiply(0.85));
                painter.galley(chip.min + egui::vec2(3.0, 1.0), g, theme.caption);
            }
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
            overlap_slot(0, max_slots).1
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

        // Ctrl+click a box: pick it as the FeatureView channel.
        let (left, right) = (
            resp.clicked_by(egui::PointerButton::Primary),
            resp.clicked_by(egui::PointerButton::Secondary),
        );
        if (left || right) && ui.input(|i| i.modifiers.command) {
            let pos = resp.interact_pointer_pos()?;
            return boxes
                .iter()
                .find(|(_, r)| r.contains(pos))
                .map(|&(ch, _)| (ch, right));
        }
        None
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
    ) {
        let theme = self.theme;
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
                    let a = (255.0 * (0.2 + 0.8 * share)) as u8;
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
                    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), alpha),
                );
                for snip in cache.snippets.get(ci).into_iter().flatten() {
                    let pts: Vec<Pos2> = (0..self.snippet_len)
                        .map(|t| map(k, t as f32 / last, snip[t * self.nc + ch]))
                        .collect();
                    painter.add(egui::Shape::line(pts, faint));
                }
            }
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
                Stroke::new(mean_width, selection_colour(ci)),
            ));
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
    fn union_is_sorted_and_deduped() {
        let u = union_channels(&[vec![1, 5, 9], vec![5, 2], vec![]]);
        assert_eq!(u, vec![1, 2, 5, 9]);
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

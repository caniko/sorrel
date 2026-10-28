//! phy's AmplitudeView: spike amplitude over time for the selection, grey
//! background spikes from the other clusters on the first cluster's peak
//! channel, a per-cluster amplitude histogram on the right, and two ways
//! to split: the polygon lasso (Ctrl+click, `k`) and phy v2.1's amplitude
//! threshold (Alt+right-drag, `k` splits the spikes below it).
//!
//! Types (`a` / Shift+`a`): template amplitudes (`amplitudes.npy`), the
//! FeatureView's first PC on its first channel, and raw peak-to-peak on
//! the peak channel. Raw amplitudes come from snippets, so only a sample
//! is read until a split needs every spike.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui};
use rayon::prelude::*;
use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider};
use sorrel_render::{extract_snippet, TracePreproc};

use crate::jobs::{JobCtx, Memo};
use crate::lasso::Lasso;
use crate::theme::{selection_colour, Theme};
use crate::views::{axes_data_rect, draw_axes_grid, waveform_window_samples};
use crate::wgpu_raster::pack_rgba;
use crate::wgpu_scatter::{ScatterCallback, ScatterVertex};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum AmpKind {
    #[default]
    Template,
    Feature,
    Raw,
    /// Projection on the first selected cluster's template
    /// (`template_features.npy`).
    TemplateFeature,
}

impl AmpKind {
    const ALL: [AmpKind; 4] = [
        AmpKind::Template,
        AmpKind::Feature,
        AmpKind::Raw,
        AmpKind::TemplateFeature,
    ];

    fn label(self) -> &'static str {
        match self {
            AmpKind::Template => "template",
            AmpKind::Feature => "feature",
            AmpKind::Raw => "raw",
            AmpKind::TemplateFeature => "template feature",
        }
    }
}

/// phy `n_spikes_amplitudes`; raw amplitudes read a fifth of that (phy).
const N_DISPLAY: usize = 10_000;
const N_RAW: usize = 2_000;
const N_BACKGROUND: usize = 5_000;
const BACKGROUND: u16 = u16::MAX;

#[derive(Copy, Clone, Debug)]
struct AmpPoint {
    local: u32,
    slot: u16,
    t: f32,
    a: f32,
}

#[derive(Clone, Debug, PartialEq)]
struct AmpKey {
    selection: Vec<ClusterId>,
    version: u64,
    kind: AmpKind,
    /// Feature type: `(channel, pc)`.
    feature: Option<(u32, usize)>,
    cfg: TracePreproc,
    shanks: Vec<u32>,
    /// Raw type with every spike, and the peak channel to read it on.
    full: Option<u32>,
    /// phy `toggle_spike_reorder`.
    reorder: bool,
}

struct AmpSet {
    points: Vec<AmpPoint>,
    /// Every selected spike is present (splits need that).
    full: bool,
    /// Raw type: the channel amplitudes were read on.
    channel: Option<u32>,
    range: (f32, f32),
    duration: f32,
    generation: u64,
}

type Vertices = Arc<Vec<ScatterVertex>>;

#[derive(Default)]
pub struct AmplitudeViewState {
    pub kind: AmpKind,
    lasso: Lasso,
    /// Amplitude threshold from Alt+right-drag (phy v2.1).
    threshold: Option<f32>,
    split_requested: bool,
    want_full: bool,
    memo: Memo<AmpKey, Arc<AmpSet>>,
    /// Vertices for `(set generation, dark mode, threshold bits, preview
    /// generation)`, plus the upload generation they were built under.
    vertices: Option<(VertexKey, u64, Vertices)>,
    /// A proposed split shown before it is applied (local indices).
    preview: Option<(ClusterId, std::collections::HashSet<u32>)>,
    preview_gen: u64,
    vertices_gen: u64,
    pz: crate::panzoom::PanZoom,
}

type VertexKey = (u64, bool, Option<u32>, u64);

impl AmplitudeViewState {
    /// Highlight a proposed split (`(cluster, local spike indices)` as in
    /// `CurationCommand::Split`) until it is cleared with `None`.
    pub fn set_split_preview(&mut self, preview: Option<(ClusterId, Vec<u32>)>) {
        self.preview = preview.map(|(c, idx)| (c, idx.into_iter().collect()));
        self.preview_gen += 1;
    }

    /// phy `next_amplitudes_type` / `previous_amplitudes_type` (`a`).
    pub fn cycle(&mut self, delta: i32) -> AmpKind {
        let i = AmpKind::ALL
            .iter()
            .position(|&k| k == self.kind)
            .unwrap_or(0) as i32;
        self.kind = AmpKind::ALL[(i + delta).rem_euclid(AmpKind::ALL.len() as i32) as usize];
        self.cancel_split();
        self.kind
    }

    /// A split is armed: a closed lasso or a threshold.
    pub fn can_split(&self) -> bool {
        self.lasso.is_closed() || self.threshold.is_some()
    }

    pub fn request_split(&mut self) {
        self.split_requested = self.can_split();
    }

    /// Esc: drop the lasso and the threshold. Returns whether there was one.
    pub fn cancel_split(&mut self) -> bool {
        let had = self.lasso.is_active() || self.threshold.is_some();
        self.lasso.clear();
        self.threshold = None;
        self.split_requested = false;
        self.want_full = false;
        had
    }

    pub fn cancel(&mut self) {
        self.memo.cancel();
    }

    pub fn pending(&self) -> bool {
        self.memo.pending()
    }
}

/// What the Amplitudes pane asks the app to do.
pub enum AmpAction {
    Split(Vec<(ClusterId, Vec<u32>)>),
    /// Alt+click: move the trace to this sample.
    Seek(u64),
}

#[allow(clippy::too_many_arguments)]
pub fn amplitude_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    state: &mut AmplitudeViewState,
    jobs: &JobCtx<Session<P>>,
    cfg: TracePreproc,
    shanks: &[u32],
    feature: Option<(u32, usize)>,
    reorder: bool,
) -> Option<AmpAction> {
    let theme = Theme::from_ui(ui);
    if selection.is_empty() {
        ui.label("no selection");
        return None;
    }
    let raw = state.kind == AmpKind::Raw;
    let key = AmpKey {
        selection: selection.to_vec(),
        version: session.version(),
        kind: state.kind,
        feature: (state.kind == AmpKind::Feature)
            .then_some(feature)
            .flatten(),
        cfg: if raw { cfg } else { TracePreproc::default() },
        shanks: if raw { shanks.to_vec() } else { Vec::new() },
        reorder,
        full: if raw && state.want_full {
            state.memo.last().and_then(|s| s.channel)
        } else {
            None
        },
    };
    let job_key = key.clone();
    let current = state
        .memo
        .ensure(key, &jobs.session, jobs.repaint(), move |s, cancel| {
            Arc::new(compute(s, &job_key, cancel))
        })
        .cloned();

    ui.horizontal_wrapped(|ui| {
        for kind in AmpKind::ALL {
            if ui
                .selectable_label(state.kind == kind, kind.label())
                .clicked()
                && state.kind != kind
            {
                state.kind = kind;
                state.cancel_split();
            }
        }
        ui.weak("(a)");
        if current.is_none() {
            ui.spinner();
        }
    });
    let hint = match (state.lasso.is_active(), state.can_split()) {
        (_, true) => "k splits · Esc clears",
        (true, false) => "Ctrl+click to add points",
        _ => "Ctrl+click: lasso · Alt+right-drag: threshold · Alt+click: seek the trace",
    };
    ui.add(egui::Label::new(egui::RichText::new(hint).weak()).truncate());

    let set = current.clone().or_else(|| state.memo.last().cloned())?;
    if set.points.iter().all(|p| p.slot == BACKGROUND) {
        ui.label(match state.kind {
            AmpKind::Template => "amplitudes.npy not loaded for this backend",
            AmpKind::Feature => "feature amplitudes need pc_features.npy",
            AmpKind::Raw => "no raw data around these spikes",
            AmpKind::TemplateFeature => "template_features.npy not loaded",
        });
        return None;
    }

    let avail = ui.available_size_before_wrap();
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(avail.x.max(160.0), avail.y.max(140.0)),
        Sense::click_and_drag(),
    );
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Other, true, "amplitude plot"));
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme.plot_bg);
    // Histogram strip on the right, phy-style.
    let hist_w = (rect.width() * 0.15).clamp(30.0, 90.0);
    let plot = Rect::from_min_max(rect.min, egui::pos2(rect.right() - hist_w, rect.bottom()));
    let data = axes_data_rect(plot, true);
    // 5% padding on both axes so no point sits on an edge.
    let d = set.duration.max(1e-3);
    let x = (-0.05 * d, 1.05 * d);
    let y = set.range;
    let (x, y) = state.pz.apply(x, y);
    draw_axes_grid(
        &painter,
        plot,
        x,
        y,
        "time (s)",
        &format!("amplitude ({})", state.kind.label()),
        &theme,
    );

    let dark = ui.visuals().dark_mode;
    let key: VertexKey = (
        set.generation,
        dark,
        state.threshold.map(f32::to_bits),
        state.preview_gen,
    );
    if !matches!(&state.vertices, Some((k, ..)) if *k == key) {
        let preview = state
            .preview
            .as_ref()
            .and_then(|(c, idx)| Some((selection.iter().position(|s| s == c)? as u16, idx)));
        state.vertices_gen += 1;
        let v = vertices(&set, dark, state.threshold, preview);
        state.vertices = Some((key, state.vertices_gen, Arc::new(v)));
    }
    if let Some((_, upload_gen, v)) = &state.vertices {
        ui.painter()
            .add(eframe::egui_wgpu::Callback::new_paint_callback(
                data,
                ScatterCallback {
                    id: ui.id().with("amplitude-scatter").value(),
                    vertices: Arc::clone(v),
                    generation: *upload_gen,
                    x,
                    y,
                    radius: 1.5,
                    size: data.size(),
                },
            ));
    }
    histograms(
        &painter,
        &set,
        Rect::from_min_max(
            egui::pos2(plot.right(), data.top()),
            egui::pos2(rect.right(), data.bottom()),
        ),
        y,
    );

    let to_screen = |p: [f32; 2]| {
        Pos2::new(
            data.left() + (p[0] - x.0) / (x.1 - x.0) * data.width(),
            data.bottom() - (p[1] - y.0) / (y.1 - y.0) * data.height(),
        )
    };
    let to_data = |p: Pos2| {
        [
            x.0 + (p.x - data.left()) / data.width() * (x.1 - x.0),
            y.0 + (data.bottom() - p.y) / data.height() * (y.1 - y.0),
        ]
    };
    state.pz.interact(&resp, data);
    let mut action = None;
    let (alt, sr) = (
        ui.input(|i| i.modifiers.alt),
        session.provider.sample_rate().max(1.0),
    );
    if !state.lasso.interact(0, &resp, to_data) {
        if alt && resp.dragged_by(egui::PointerButton::Secondary) {
            if let Some(p) = resp.interact_pointer_pos() {
                state.threshold = Some(to_data(p)[1]);
            }
        } else if alt && resp.clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                action = Some(AmpAction::Seek((to_data(p)[0].max(0.0) * sr) as u64));
            }
        }
    }
    state.lasso.paint(&painter, to_screen, theme.text);
    if let Some(th) = state.threshold {
        let sy = to_screen([0.0, th]).y;
        painter.hline(
            data.x_range(),
            sy,
            Stroke::new(1.5_f32, Color32::from_rgb(255, 140, 0)),
        );
    }

    if state.split_requested && state.can_split() {
        match current.filter(|s| s.full) {
            Some(set) => {
                state.split_requested = false;
                state.want_full = false;
                let splits = split_hits(&set, &state.lasso, state.threshold, selection);
                state.lasso.clear();
                state.threshold = None;
                return Some(AmpAction::Split(splits));
            }
            None => state.want_full = true,
        }
    }
    action
}

/// Spikes inside the lasso, or below the threshold (phy v2.1 splits the
/// spikes under the line), as `(cluster, local indices)`.
fn split_hits(
    set: &AmpSet,
    lasso: &Lasso,
    threshold: Option<f32>,
    selection: &[ClusterId],
) -> Vec<(ClusterId, Vec<u32>)> {
    let mut hits: Vec<Vec<u32>> = vec![Vec::new(); selection.len()];
    for p in set.points.iter().filter(|p| p.slot != BACKGROUND) {
        let hit = if lasso.is_closed() {
            lasso.contains([p.t, p.a])
        } else {
            threshold.is_some_and(|th| p.a < th)
        };
        if hit {
            hits[p.slot as usize].push(p.local);
        }
    }
    selection
        .iter()
        .copied()
        .zip(hits)
        .filter(|(_, h)| !h.is_empty())
        .collect()
}

fn vertices(
    set: &AmpSet,
    dark: bool,
    threshold: Option<f32>,
    preview: Option<(u16, &std::collections::HashSet<u32>)>,
) -> Vec<ScatterVertex> {
    let grey = if dark { 150 } else { 110 };
    let previewed =
        |p: &AmpPoint| preview.is_some_and(|(slot, idx)| p.slot == slot && idx.contains(&p.local));
    let mut counts: HashMap<u16, usize> = HashMap::new();
    for p in &set.points {
        *counts.entry(p.slot).or_default() += 1;
    }
    let mut order: Vec<&AmpPoint> = set
        .points
        .iter()
        .filter(|p| {
            p.slot == BACKGROUND
                || previewed(p)
                || (p.local as usize).is_multiple_of((counts[&p.slot] / N_DISPLAY).max(1))
        })
        .collect();
    order.sort_by_key(|p| (p.slot != BACKGROUND, previewed(p)));
    order
        .into_iter()
        .map(|p| ScatterVertex {
            pos: [p.t, p.a],
            color: if p.slot == BACKGROUND {
                pack_rgba(grey, grey, grey, 110)
            } else if previewed(p) {
                if dark {
                    pack_rgba(255, 255, 255, 240)
                } else {
                    pack_rgba(20, 20, 20, 240)
                }
            } else if threshold.is_some_and(|th| p.a < th) {
                pack_rgba(255, 140, 0, 230)
            } else {
                let c = selection_colour(p.slot as usize);
                pack_rgba(c.r(), c.g(), c.b(), 220)
            },
        })
        .collect()
}

/// One horizontal amplitude histogram per selected cluster, overlaid.
fn histograms(painter: &egui::Painter, set: &AmpSet, rect: Rect, y: (f32, f32)) {
    const BINS: usize = 50;
    let mut per: HashMap<u16, [u32; BINS]> = HashMap::new();
    for p in set.points.iter().filter(|p| p.slot != BACKGROUND) {
        let b = ((p.a - y.0) / (y.1 - y.0) * BINS as f32).floor();
        if (0.0..BINS as f32).contains(&b) {
            per.entry(p.slot).or_insert([0; BINS])[b as usize] += 1;
        }
    }
    for (slot, bins) in per {
        let max = *bins.iter().max().unwrap_or(&1).max(&1) as f32;
        let colour = selection_colour(slot as usize);
        let pts: Vec<Pos2> = bins
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                Pos2::new(
                    rect.left() + n as f32 / max * rect.width(),
                    rect.bottom() - (i as f32 + 0.5) / BINS as f32 * rect.height(),
                )
            })
            .collect();
        painter.add(egui::Shape::line(pts, Stroke::new(1.0_f32, colour)));
    }
}

static GENERATION: AtomicU64 = AtomicU64::new(1);

fn compute<P: DataProvider>(s: &Session<P>, key: &AmpKey, cancel: &AtomicBool) -> AmpSet {
    let sr = s.provider.sample_rate().max(1.0);
    let clusters = s.spike_clusters_global();
    let slot_of: HashMap<ClusterId, u16> = key
        .selection
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i as u16))
        .collect();
    let raw = key.kind == AmpKind::Raw;
    let keep_every: Vec<usize> = key
        .selection
        .iter()
        .map(|&c| {
            if raw && key.full.is_none() {
                (s.spike_times(c).len() / N_RAW).max(1)
            } else {
                1
            }
        })
        .collect();
    // Background: clusters whose primary template peaks on the first
    // selected cluster's peak channel (phy: spikes on the same channel).
    let peak = |c: ClusterId| s.primary_template(c).and_then(|t| peak_channel(s, t));
    let bg_peak = key.selection.first().and_then(|&c| peak(c));
    let bg_clusters: Vec<bool> = match (bg_peak, key.full) {
        (Some(ch), None) => (0..s.n_clusters())
            .map(|c| peak(ClusterId(c)) == Some(ch))
            .collect(),
        _ => Vec::new(),
    };
    let bg_total: usize = bg_clusters
        .iter()
        .enumerate()
        .filter(|&(c, &on)| on && !slot_of.contains_key(&ClusterId(c as u32)))
        .map(|(c, _)| s.spike_times(ClusterId(c as u32)).len())
        .sum();
    let bg_step = (bg_total / N_BACKGROUND).max(1);
    let mut counters = vec![0u32; key.selection.len()];
    let mut bg_seen = 0usize;
    // (global, local, slot)
    let mut picked: Vec<(u32, u32, u16)> = Vec::new();
    for (g, c) in clusters.iter().enumerate() {
        match slot_of.get(c) {
            Some(&slot) => {
                let local = counters[slot as usize];
                counters[slot as usize] += 1;
                if (local as usize).is_multiple_of(keep_every[slot as usize]) {
                    picked.push((g as u32, local, slot));
                }
            }
            None if bg_clusters.get(c.idx()).copied().unwrap_or(false) => {
                if bg_seen.is_multiple_of(bg_step) {
                    picked.push((g as u32, 0, BACKGROUND));
                }
                bg_seen += 1;
            }
            None => {}
        }
    }
    let amps_global = s.cluster_index().spike_amplitudes_global();
    let mut channel = None;
    let values: Vec<f32> = match key.kind {
        AmpKind::Template => picked
            .iter()
            .map(|&(g, ..)| amps_global.get(g as usize).copied().unwrap_or(f32::NAN))
            .collect(),
        AmpKind::Feature => match key.feature {
            Some((ch, pc)) => picked
                .iter()
                .map(|&(g, ..)| s.pc_feature(g, ch, pc).unwrap_or(f32::NAN))
                .collect(),
            None => vec![f32::NAN; picked.len()],
        },
        AmpKind::Raw => {
            let (vals, ch) = raw_amplitudes(s, key, &picked, cancel);
            channel = ch;
            vals
        }
        AmpKind::TemplateFeature => {
            let t = key.selection.first().and_then(|&c| s.primary_template(c));
            picked
                .iter()
                .map(|&(g, ..)| {
                    t.and_then(|t| s.cluster_index().template_feature(g, t))
                        .unwrap_or(f32::NAN)
                })
                .collect()
        }
    };
    let points: Vec<AmpPoint> = picked
        .iter()
        .zip(values)
        .filter(|(_, a)| a.is_finite())
        .map(|(&(g, local, slot), a)| AmpPoint {
            local,
            slot,
            t: s.cluster_index().display_time(g, key.reorder).0 as f32 / sr,
            a,
        })
        .collect();
    let (lo, hi) = points
        .iter()
        .filter(|p| p.slot != BACKGROUND)
        .fold((f32::MAX, f32::MIN), |(lo, hi), p| {
            (lo.min(p.a), hi.max(p.a))
        });
    let pad = ((hi - lo) * 0.05).max(1e-3);
    AmpSet {
        full: !raw || key.full.is_some(),
        channel,
        range: if lo <= hi {
            (lo.min(0.0) - pad, hi + pad)
        } else {
            (0.0, 1.0)
        },
        points,
        duration: s.provider.n_samples().0 as f32 / sr,
        generation: GENERATION.fetch_add(1, Ordering::Relaxed),
    }
}

/// Channel with the largest peak-to-peak in `template`.
pub(crate) fn peak_channel<P: DataProvider>(s: &Session<P>, template: u32) -> Option<u32> {
    let w = s.template_waveform(template)?;
    let (_, ns, nc) = s.template_shape();
    (0..nc)
        .map(|ch| {
            let col = (0..ns).filter_map(|i| w.get(i * nc + ch));
            let (lo, hi) = col.fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            (ch as u32, hi - lo)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(ch, _)| ch)
}

/// Peak-to-peak of each picked spike's raw snippet on one channel: the
/// given one, else the channel with the largest mean peak-to-peak over the
/// first cluster's spikes.
fn raw_amplitudes<P: DataProvider>(
    s: &Session<P>,
    key: &AmpKey,
    picked: &[(u32, u32, u16)],
    cancel: &AtomicBool,
) -> (Vec<f32>, Option<u32>) {
    let p = &s.provider;
    let nc = p.n_channels() as usize;
    let half = waveform_window_samples(p.sample_rate());
    let times = s.cluster_index().all_spike_times();
    let ptp = |g: u32| -> Option<Vec<f32>> {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        let snip = extract_snippet(p, times[g as usize], half, half, key.cfg, &key.shanks)?;
        let len = snip.len() / nc.max(1);
        Some(
            (0..nc)
                .map(|ch| {
                    let col = (0..len).map(|i| snip[i * nc + ch]);
                    col.clone().fold(f32::MIN, f32::max) - col.fold(f32::MAX, f32::min)
                })
                .collect(),
        )
    };
    let per: Vec<Option<Vec<f32>>> = picked.par_iter().map(|&(g, ..)| ptp(g)).collect();
    let channel = key.full.or_else(|| {
        let mut sum = vec![0.0f32; nc];
        for v in picked
            .iter()
            .zip(&per)
            .filter(|(x, _)| x.2 == 0)
            .filter_map(|(_, v)| v.as_ref())
        {
            sum.iter_mut().zip(v).for_each(|(a, b)| *a += b);
        }
        (0..nc)
            .max_by(|&a, &b| sum[a].total_cmp(&sum[b]))
            .map(|c| c as u32)
    });
    let vals = per
        .into_iter()
        .map(|v| {
            v.zip(channel)
                .and_then(|(v, ch)| v.get(ch as usize).copied())
                .unwrap_or(f32::NAN)
        })
        .collect();
    (vals, channel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_splits_spikes_below_and_lasso_wins_when_closed() {
        let pt = |local, slot, a| AmpPoint {
            local,
            slot,
            t: 1.0,
            a,
        };
        let set = AmpSet {
            points: vec![
                pt(0, 0, 5.0),
                pt(1, 0, 50.0),
                pt(2, 0, 8.0),
                pt(0, BACKGROUND, 1.0),
            ],
            full: true,
            channel: None,
            range: (0.0, 60.0),
            duration: 2.0,
            generation: 0,
        };
        let c = [ClusterId(4)];
        assert_eq!(
            split_hits(&set, &Lasso::default(), Some(10.0), &c),
            vec![(ClusterId(4), vec![0, 2])]
        );
        let lasso = Lasso {
            cell: 0,
            points: vec![[0.0, 40.0], [2.0, 40.0], [2.0, 60.0], [0.0, 60.0]],
        };
        assert_eq!(
            split_hits(&set, &lasso, Some(10.0), &c),
            vec![(ClusterId(4), vec![1])]
        );
    }

    #[test]
    fn a_cycles_through_types() {
        let mut s = AmplitudeViewState::default();
        assert_eq!(s.cycle(1), AmpKind::Feature);
        assert_eq!(s.cycle(1), AmpKind::Raw);
        assert_eq!(s.cycle(1), AmpKind::TemplateFeature);
        assert_eq!(s.cycle(1), AmpKind::Template);
        assert_eq!(s.cycle(-1), AmpKind::TemplateFeature);
    }
}

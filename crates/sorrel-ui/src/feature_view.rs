//! phy's FeatureView: a 4×4 grid of PC projections on the selection's two
//! best channels with time on the diagonal, grey background spikes drawn
//! underneath, and a polygon lasso for splitting (`k`).
//!
//! Features come from `pc_features.npy` when the sorter wrote it. Without
//! it (SortingAnalyzer, NWB, rez, Kilosort run without PCs) two temporal
//! PCs are fitted on raw peak-channel snippets and every shown spike is
//! projected onto them. Only the displayed subsample is projected until a
//! split needs every spike of the lassoed clusters (phy's `load_all`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use egui::{Pos2, Rect, Sense, Stroke, Ui};
use rayon::prelude::*;
use sorrel_compute::linalg::covariance;
use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider};
use sorrel_render::{extract_snippet, TracePreproc};

use crate::jobs::{JobCtx, Memo};
use crate::lasso::Lasso;
use crate::theme::{selection_colour, Theme};
use crate::views::waveform_window_samples;
use crate::wgpu_raster::pack_rgba;
use crate::wgpu_scatter::{ScatterCallback, ScatterVertex};

/// One axis of a grid cell: time, or PC `pc` on the `ch`-th best channel
/// (phy's `0A`, `1B`, …).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    Time,
    Pc { ch: u8, pc: u8 },
}

const fn pc(ch: u8, pc: u8) -> Dim {
    Dim::Pc { ch, pc }
}

/// phy `FeatureView` default grid, `(x, y)` per cell, row-major.
const GRID: [[(Dim, Dim); 4]; 4] = [
    [
        (Dim::Time, pc(0, 0)),
        (pc(1, 0), pc(0, 0)),
        (pc(0, 1), pc(0, 0)),
        (pc(1, 1), pc(0, 0)),
    ],
    [
        (pc(0, 0), pc(1, 0)),
        (Dim::Time, pc(1, 0)),
        (pc(0, 1), pc(1, 0)),
        (pc(1, 1), pc(1, 0)),
    ],
    [
        (pc(0, 0), pc(0, 1)),
        (pc(1, 0), pc(0, 1)),
        (Dim::Time, pc(0, 1)),
        (pc(1, 1), pc(0, 1)),
    ],
    [
        (pc(0, 0), pc(1, 1)),
        (pc(1, 0), pc(1, 1)),
        (pc(0, 1), pc(1, 1)),
        (Dim::Time, pc(1, 1)),
    ],
];

/// phy `n_spikes_features` / `n_spikes_features_background`.
const N_DISPLAY: usize = 2500;
const N_BACKGROUND: usize = 2500;
/// Spikes per cluster projected from raw data when PCs are computed.
// ponytail: each costs one raw snippet read; raise toward N_DISPLAY if
// that proves cheap on real recordings.
const N_COMPUTED: usize = 500;
/// Snippets the temporal PCs are fitted on.
const N_FIT: usize = 1000;
const BACKGROUND: u16 = u16::MAX;

#[derive(Copy, Clone, Debug)]
struct FeaturePoint {
    /// Position within the cluster's time-sorted bucket (the `Split` index).
    local: u32,
    slot: u16,
    t: f32,
    /// `[0A, 1A, 0B, 1B]`. A selected spike whose template lacks a channel
    /// reads 0 there (phylib `from_sparse`); background spikes read NaN.
    f: [f32; 4],
}

impl FeaturePoint {
    fn value(&self, dim: Dim) -> f32 {
        match dim {
            Dim::Time => self.t,
            Dim::Pc { ch, pc } => self.f[pc as usize * 2 + ch as usize],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct FeatureKey {
    selection: Vec<ClusterId>,
    channels: Option<[u32; 2]>,
    version: u64,
    cfg: TracePreproc,
    shanks: Vec<u32>,
    /// Every spike of the selected clusters, not just the display sample.
    full: bool,
    /// phy `toggle_spike_reorder`: time axes use `spike_reorder.npy`.
    reorder: bool,
}

struct FeatureSet {
    channels: [u32; 2],
    computed: bool,
    full: bool,
    points: Vec<FeaturePoint>,
    /// Robust max |value| per feature column, for symmetric axes.
    bounds: [f32; 4],
    duration: f32,
    generation: u64,
}

type CellVertices = Vec<Arc<Vec<ScatterVertex>>>;

/// Temporal PC basis fitted on raw snippets, reused across selections.
type Basis = Arc<Mutex<Option<(TracePreproc, u32, Arc<[Vec<f32>; 2]>)>>>;

pub struct FeatureViewState {
    /// `None` follows the selection's best channels; phy's `c` fixes them.
    pub fixed_channels: Option<[u32; 2]>,
    /// Feature scaling (Ctrl+wheel); divides the PC axis ranges.
    pub scaling: f32,
    /// Marker radius in points (Alt+wheel).
    pub marker: f32,
    zoom: f32,
    pan: [f32; 2],
    lasso: Lasso,
    split_requested: bool,
    want_full: bool,
    /// Channels of the last drawn set, so `c` can freeze what is on screen.
    shown_channels: Option<[u32; 2]>,
    /// Alt+clicked `(channel, pc)` for the AmplitudeView feature type.
    amplitude_dim: Option<(u32, usize)>,
    memo: Memo<FeatureKey, Arc<FeatureSet>>,
    basis: Basis,
    /// Per-cell vertex buffers for `(set generation, dark mode)`.
    cells: Option<((u64, bool, u64), u64, CellVertices)>,
    /// A proposed split shown before it is applied: the source cluster
    /// and the local indices that would move, drawn highlighted.
    preview: Option<(ClusterId, std::collections::HashSet<u32>)>,
    preview_gen: u64,
    cells_gen: u64,
}

impl Default for FeatureViewState {
    fn default() -> Self {
        Self {
            fixed_channels: None,
            scaling: 1.0,
            marker: 1.5,
            zoom: 1.0,
            pan: [0.0; 2],
            lasso: Lasso::default(),
            split_requested: false,
            want_full: false,
            shown_channels: None,
            amplitude_dim: None,
            memo: Memo::new(),
            basis: Basis::default(),
            cells: None,
            preview: None,
            preview_gen: 0,
            cells_gen: 0,
        }
    }
}

impl FeatureViewState {
    /// Abandon an in-progress lasso (Esc).
    pub fn cancel_lasso(&mut self) {
        self.lasso.clear();
        self.split_requested = false;
        self.want_full = false;
    }

    pub fn lasso_active(&self) -> bool {
        self.lasso.is_active()
    }

    pub fn lasso_closed(&self) -> bool {
        self.lasso.is_closed()
    }

    /// phy `split` (`k`): the next frame splits off the lassoed spikes.
    pub fn request_split(&mut self) {
        self.split_requested = self.lasso.is_closed();
    }

    /// phy `toggle_automatic_channel_selection` (`c`). Returns whether the
    /// channels are now fixed.
    pub fn toggle_channels(&mut self) -> bool {
        self.fixed_channels = match self.fixed_channels {
            Some(_) => None,
            None => self.shown_channels,
        };
        self.fixed_channels.is_some()
    }

    /// phy `on_select_channel` (Ctrl+click on a Waveforms channel): the
    /// left button sets the first feature channel, the right button the
    /// second, swapping when it is already the other one. Fixes channels.
    pub fn select_channel(&mut self, channel: u32, secondary: bool) {
        let mut ch = self
            .fixed_channels
            .or(self.shown_channels)
            .unwrap_or([channel; 2]);
        let (i, j) = if secondary { (1, 0) } else { (0, 1) };
        if ch[j] == channel {
            ch[j] = ch[i];
        }
        ch[i] = channel;
        self.fixed_channels = Some(ch);
    }

    /// The `(channel, pc)` the AmplitudeView's feature type plots: the last
    /// Alt+clicked dimension (phy `select_feature`), else PC1 on the first
    /// channel shown.
    /// Highlight a proposed split (`(cluster, local spike indices)` as in
    /// `CurationCommand::Split`) until it is cleared with `None`.
    pub fn set_split_preview(&mut self, preview: Option<(ClusterId, Vec<u32>)>) {
        self.preview = preview.map(|(c, idx)| (c, idx.into_iter().collect()));
        self.preview_gen += 1;
    }

    pub fn amplitude_feature(&self) -> Option<(u32, usize)> {
        self.amplitude_dim
            .or_else(|| self.shown_channels.map(|c| (c[0], 0)))
    }

    /// Stop the worker before a curation write takes the session lock.
    pub fn cancel(&mut self) {
        self.memo.cancel();
    }

    pub fn pending(&self) -> bool {
        self.memo.pending()
    }
}

/// Draw the grid. Returns `(cluster, local spike indices)` splits when `k`
/// was pressed over a closed lasso and the features for every lassoed
/// spike are in.
#[allow(clippy::too_many_arguments)]
pub fn feature_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    state: &mut FeatureViewState,
    jobs: &JobCtx<Session<P>>,
    cfg: TracePreproc,
    shanks: &[u32],
    reorder: bool,
) -> Option<Vec<(ClusterId, Vec<u32>)>> {
    let theme = Theme::from_ui(ui);
    if selection.is_empty() {
        ui.label("no selection");
        return None;
    }
    let computed = !session.has_pc_features();
    let key = FeatureKey {
        selection: selection.to_vec(),
        channels: state
            .fixed_channels
            .or(state.want_full.then_some(state.shown_channels).flatten()),
        version: session.version(),
        // Precomputed features ignore the trace preprocessing.
        cfg: if computed {
            cfg
        } else {
            TracePreproc::default()
        },
        shanks: if computed {
            shanks.to_vec()
        } else {
            Vec::new()
        },
        full: computed && state.want_full,
        reorder,
    };
    let basis = Arc::clone(&state.basis);
    let job_key = key.clone();
    let current = state
        .memo
        .ensure(key, &jobs.session, jobs.repaint(), move |s, cancel| {
            Arc::new(compute(s, &job_key, &basis, cancel))
        })
        .cloned();
    let set = match current.clone().or_else(|| state.memo.last().cloned()) {
        Some(set) => set,
        None => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(if computed {
                    "fitting PCs on raw snippets…"
                } else {
                    "loading PC features…"
                });
            });
            return None;
        }
    };
    state.shown_channels = Some(set.channels);

    header(ui, state, &set, current.is_none());

    let avail = ui.available_size_before_wrap();
    let side = avail.x.min(avail.y).max(160.0);
    let (grid_rect, _) =
        ui.allocate_exact_size(egui::vec2(avail.x.max(160.0), side), Sense::hover());
    let gap = 2.0;
    let cell_w = (grid_rect.width() - 3.0 * gap) / 4.0;
    let cell_h = (grid_rect.height() - 3.0 * gap) / 4.0;

    let dark = ui.visuals().dark_mode;
    let cells_key = (set.generation, dark, state.preview_gen);
    if !matches!(&state.cells, Some((k, ..)) if *k == cells_key) {
        let preview = state.preview.as_ref().and_then(|(c, idx)| {
            let slot = selection.iter().position(|s| s == c)?;
            Some((slot as u16, idx))
        });
        state.cells_gen += 1;
        state.cells = Some((
            cells_key,
            state.cells_gen,
            cell_vertices(&set, dark, preview),
        ));
    }
    let cells_gen = state.cells_gen;
    let cells = state
        .cells
        .as_ref()
        .map(|(_, _, c)| c.clone())
        .unwrap_or_default();

    let painter = ui.painter_at(grid_rect);
    for (cell, &(dx, dy)) in GRID.iter().flatten().enumerate() {
        let (row, col) = (cell / 4, cell % 4);
        let rect = Rect::from_min_size(
            grid_rect.min + egui::vec2(col as f32 * (cell_w + gap), row as f32 * (cell_h + gap)),
            egui::vec2(cell_w, cell_h),
        );
        painter.rect_filled(rect, 0.0, theme.plot_bg);
        let (x, y) = (
            axis_range(&set, state, dx, 0),
            axis_range(&set, state, dy, 1),
        );
        let to_screen = |p: [f32; 2]| {
            Pos2::new(
                rect.left() + (p[0] - x.0) / (x.1 - x.0) * rect.width(),
                rect.bottom() - (p[1] - y.0) / (y.1 - y.0) * rect.height(),
            )
        };
        let to_data = |p: Pos2| {
            [
                x.0 + (p.x - rect.left()) / rect.width() * (x.1 - x.0),
                y.0 + (rect.bottom() - p.y) / rect.height() * (y.1 - y.0),
            ]
        };
        if let Some(vertices) = cells.get(cell) {
            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    ScatterCallback {
                        id: ui.id().with(("feature-cell", cell)).value(),
                        vertices: Arc::clone(vertices),
                        generation: cells_gen,
                        x,
                        y,
                        radius: state.marker,
                        size: rect.size(),
                    },
                ));
        }
        painter.rect_stroke(
            rect,
            0.0,
            Stroke::new(0.5_f32, theme.grid),
            egui::StrokeKind::Inside,
        );
        painter.text(
            rect.left_top() + egui::vec2(3.0, 2.0),
            egui::Align2::LEFT_TOP,
            format!("{} · {}", dim_label(dx, &set), dim_label(dy, &set)),
            egui::FontId::proportional(9.0),
            theme.caption,
        );
        if state.lasso.is_active() && state.lasso.cell == cell {
            state.lasso.paint(&painter, to_screen, theme.text);
        }

        let resp = ui.interact(
            rect,
            ui.id().with(("feature-hit", cell)),
            Sense::click_and_drag(),
        );
        resp.widget_info(|| {
            egui::WidgetInfo::labeled(
                egui::WidgetType::Other,
                true,
                format!("feature cell {cell}"),
            )
        });
        // phy `select_feature`: Alt+click picks the x (left) or y (right)
        // dimension for the AmplitudeView feature type.
        let alt = ui.input(|i| i.modifiers.alt);
        let picked = match (alt, resp.clicked(), resp.secondary_clicked()) {
            (true, true, _) => Some(dx),
            (true, _, true) => Some(dy),
            _ => None,
        };
        if let Some(Dim::Pc { ch, pc }) = picked {
            state.amplitude_dim = Some((set.channels[ch as usize], pc as usize));
        } else if !state.lasso.interact(cell, &resp, to_data) {
            interact_view(ui, state, &resp);
        }
    }

    if state.split_requested && state.lasso.is_closed() {
        match current.filter(|s| s.full || !s.computed) {
            Some(set) => {
                state.split_requested = false;
                state.want_full = false;
                let splits = lasso_hits(&set, &state.lasso, selection);
                state.lasso.clear();
                return Some(splits);
            }
            // Computed PCs: project every lassoed spike, then split.
            None => state.want_full = true,
        }
    }
    None
}

fn header(ui: &mut Ui, state: &mut FeatureViewState, set: &FeatureSet, busy: bool) {
    ui.horizontal_wrapped(|ui| {
        ui.label(format!(
            "channels {} · {}",
            set.channels[0], set.channels[1]
        ));
        let mut fixed = state.fixed_channels.is_some();
        if ui
            .checkbox(&mut fixed, "fixed (c)")
            .on_hover_text("Keep these channels instead of following the selection's best channels")
            .changed()
        {
            state.toggle_channels();
        }
        if let Some(ch) = state.fixed_channels.as_mut() {
            for c in ch.iter_mut() {
                ui.add(egui::DragValue::new(c).speed(0.2));
            }
        }
        ui.weak(if set.computed {
            "PCs fitted on raw snippets"
        } else {
            "pc_features.npy"
        });
        if busy {
            ui.spinner();
        }
    });
    // One truncated line whatever it says, so the grid never moves under
    // the cursor while a lasso is being drawn.
    let hint = match (state.lasso.is_active(), state.lasso.is_closed()) {
        (_, true) => "k splits the lassoed spikes · Esc clears",
        (true, false) => "Ctrl+click to add points · Ctrl+right-click clears",
        _ => "Ctrl+click: lasso · wheel: zoom · Ctrl+wheel: scale · Alt+wheel: size · double-click: reset",
    };
    ui.add(egui::Label::new(egui::RichText::new(hint).weak()).truncate());
}

/// Pan (drag), zoom (wheel), feature scaling (Ctrl+wheel), marker size
/// (Alt+wheel) and reset (double-click). Pan and zoom are shared by every
/// cell, as in phy.
fn interact_view(ui: &Ui, state: &mut FeatureViewState, resp: &egui::Response) {
    if resp.dragged_by(egui::PointerButton::Primary) {
        let d = resp.drag_delta();
        state.pan[0] -= d.x / resp.rect.width() / state.zoom;
        state.pan[1] += d.y / resp.rect.height() / state.zoom;
    }
    if resp.double_clicked() {
        state.zoom = 1.0;
        state.pan = [0.0; 2];
        state.scaling = 1.0;
    }
    if !resp.hovered() {
        return;
    }
    let (scroll, pinch, alt) =
        ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta(), i.modifiers.alt));
    if pinch != 1.0 {
        // egui reports Ctrl+wheel as a zoom gesture.
        state.scaling = (state.scaling * pinch).clamp(0.05, 50.0);
    } else if scroll != 0.0 {
        let f = (scroll * 0.002).exp();
        if alt {
            state.marker = (state.marker * f).clamp(0.5, 8.0);
        } else {
            state.zoom = (state.zoom * f).clamp(1.0, 100.0);
        }
    }
}

/// Visible data range of `dim` along `axis` (0 = x, 1 = y) after scaling,
/// pan and zoom.
fn axis_range(set: &FeatureSet, state: &FeatureViewState, dim: Dim, axis: usize) -> (f32, f32) {
    let (lo, hi) = match dim {
        Dim::Time => (0.0, set.duration.max(1e-3)),
        Dim::Pc { ch, pc } => {
            let b = set.bounds[pc as usize * 2 + ch as usize] / state.scaling;
            (-b, b)
        }
    };
    let span = hi - lo;
    let centre = 0.5 + state.pan[axis];
    let half = 0.5 / state.zoom;
    (lo + (centre - half) * span, lo + (centre + half) * span)
}

fn dim_label(dim: Dim, set: &FeatureSet) -> String {
    match dim {
        Dim::Time => "time".into(),
        Dim::Pc { ch, pc } => format!("ch{} PC{}", set.channels[ch as usize], pc + 1),
    }
}

/// Per-cell vertex buffers: background first, then each cluster thinned to
/// at most [`N_DISPLAY`] points.
fn cell_vertices(
    set: &FeatureSet,
    dark: bool,
    preview: Option<(u16, &std::collections::HashSet<u32>)>,
) -> Vec<Arc<Vec<ScatterVertex>>> {
    let grey = if dark { 150 } else { 110 };
    let highlight = if dark {
        pack_rgba(255, 255, 255, 240)
    } else {
        pack_rgba(20, 20, 20, 240)
    };
    let previewed = |p: &FeaturePoint| {
        preview.is_some_and(|(slot, idx)| p.slot == slot && idx.contains(&p.local))
    };
    let background = pack_rgba(grey, grey, grey, 110);
    let mut counts: HashMap<u16, usize> = HashMap::new();
    for p in &set.points {
        *counts.entry(p.slot).or_default() += 1;
    }
    let stride = |slot: u16| (counts[&slot] / N_DISPLAY).max(1);
    let mut order: Vec<&FeaturePoint> = set
        .points
        .iter()
        .filter(|p| {
            p.slot == BACKGROUND
                || previewed(p)
                || (p.local as usize).is_multiple_of(stride(p.slot))
        })
        .collect();
    // Background under the clusters, a previewed split on top.
    order.sort_by_key(|p| (p.slot != BACKGROUND, previewed(p)));
    GRID.iter()
        .flatten()
        .map(|&(dx, dy)| {
            Arc::new(
                order
                    .iter()
                    .filter_map(|p| {
                        let (x, y) = (p.value(dx), p.value(dy));
                        (x.is_finite() && y.is_finite()).then(|| ScatterVertex {
                            pos: [x, y],
                            color: if p.slot == BACKGROUND {
                                background
                            } else if previewed(p) {
                                highlight
                            } else {
                                let c = selection_colour(p.slot as usize);
                                pack_rgba(c.r(), c.g(), c.b(), 220)
                            },
                        })
                    })
                    .collect(),
            )
        })
        .collect()
}

fn lasso_hits(
    set: &FeatureSet,
    lasso: &Lasso,
    selection: &[ClusterId],
) -> Vec<(ClusterId, Vec<u32>)> {
    let (dx, dy) = GRID[lasso.cell / 4][lasso.cell % 4];
    let mut hits: Vec<Vec<u32>> = vec![Vec::new(); selection.len()];
    for p in set.points.iter().filter(|p| p.slot != BACKGROUND) {
        let xy = [p.value(dx), p.value(dy)];
        if xy[0].is_finite() && xy[1].is_finite() && lasso.contains(xy) {
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

static GENERATION: AtomicU64 = AtomicU64::new(1);

fn compute<P: DataProvider>(
    s: &Session<P>,
    key: &FeatureKey,
    basis: &Basis,
    cancel: &AtomicBool,
) -> FeatureSet {
    let computed = !s.has_pc_features();
    let sr = s.provider.sample_rate().max(1.0);
    let clusters = s.spike_clusters_global();
    let slot_of: HashMap<ClusterId, u16> = key
        .selection
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i as u16))
        .collect();
    // Raw projection is costly, so computed PCs show a smaller sample.
    let keep_every: Vec<usize> = key
        .selection
        .iter()
        .map(|&c| match (computed, key.full) {
            (true, false) => (s.spike_times(c).len() / N_COMPUTED).max(1),
            _ => 1,
        })
        .collect();
    let bg_step = (clusters.len() / N_BACKGROUND).max(1);
    let mut counters = vec![0u32; key.selection.len()];
    // (global index, local index, slot)
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
            None if !key.full && g % bg_step == 0 => picked.push((g as u32, 0, BACKGROUND)),
            None => {}
        }
    }
    let t_of = |g: u32| s.cluster_index().display_time(g, key.reorder).0 as f32 / sr;

    let (channels, points) = if computed {
        computed_features(s, key, basis, &picked, cancel)
    } else {
        let channels = key.channels.unwrap_or_else(|| {
            let best = s.best_channels(&key.selection);
            let first = best.first().copied().unwrap_or(0);
            [first, best.get(1).copied().unwrap_or(first)]
        });
        let points = picked
            .iter()
            .map(|&(g, local, slot)| FeaturePoint {
                local,
                slot,
                t: t_of(g),
                f: std::array::from_fn(|i| {
                    let missing = if slot == BACKGROUND { f32::NAN } else { 0.0 };
                    s.pc_feature(g, channels[i % 2], i / 2).unwrap_or(missing)
                }),
            })
            .collect();
        (channels, points)
    };
    let points: Vec<FeaturePoint> = points
        .into_iter()
        .filter(|p| p.slot != BACKGROUND || p.f.iter().any(|v| v.is_finite()))
        .collect();
    FeatureSet {
        channels,
        computed,
        full: key.full || !computed,
        bounds: std::array::from_fn(|i| {
            robust_abs_max(
                points
                    .iter()
                    .filter(|p| p.slot != BACKGROUND)
                    .map(|p| p.f[i]),
            )
        }),
        points,
        duration: s.provider.n_samples().0 as f32 / sr,
        generation: GENERATION.fetch_add(1, Ordering::Relaxed),
    }
}

/// 99th percentile of |v| over finite values (1.0 when there are none).
fn robust_abs_max(vals: impl Iterator<Item = f32>) -> f32 {
    let mut v: Vec<f32> = vals.filter(|v| v.is_finite()).map(f32::abs).collect();
    if v.is_empty() {
        return 1.0;
    }
    let k = ((v.len() as f32 * 0.99) as usize).min(v.len() - 1);
    let (_, m, _) = v.select_nth_unstable_by(k, f32::total_cmp);
    m.max(1e-6) * 1.05
}

fn computed_features<P: DataProvider>(
    s: &Session<P>,
    key: &FeatureKey,
    basis: &Basis,
    picked: &[(u32, u32, u16)],
    cancel: &AtomicBool,
) -> ([u32; 2], Vec<FeaturePoint>) {
    let p = &s.provider;
    let sr = p.sample_rate().max(1.0);
    let nc = p.n_channels() as usize;
    let half = waveform_window_samples(sr);
    let times = s.cluster_index().all_spike_times();
    let Some(b) = fitted_basis(s, key, basis, half, cancel) else {
        return ([0, 0], Vec::new());
    };
    let snippet = |g: u32| extract_snippet(p, times[g as usize], half, half, key.cfg, &key.shanks);
    let project = |snip: &[f32], pc: usize, ch: usize| -> f32 {
        b[pc]
            .iter()
            .enumerate()
            .map(|(t, w)| w * snip[t * nc + ch])
            .sum()
    };
    p.prefetch(
        &picked
            .iter()
            .map(|&(g, ..)| {
                let (start, len, _) =
                    sorrel_render::snippet_window(sr, times[g as usize], half, half, key.cfg);
                (start, len)
            })
            .collect::<Vec<_>>(),
    );

    // Automatic channels: the two with the largest mean |PC1| over the
    // first cluster's spikes (the temporal-PC analogue of peak amplitude).
    let channels = key.channels.unwrap_or_else(|| {
        let mut score = vec![0.0f32; nc];
        for g in picked.iter().filter(|x| x.2 == 0).map(|x| x.0) {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            if let Some(snip) = snippet(g) {
                for (ch, sc) in score.iter_mut().enumerate() {
                    *sc += project(&snip, 0, ch).abs();
                }
            }
        }
        let mut order: Vec<usize> = (0..nc).collect();
        order.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
        let first = order.first().copied().unwrap_or(0) as u32;
        [first, order.get(1).map_or(first, |&c| c as u32)]
    });
    let points = picked
        .par_iter()
        .filter_map(|&(g, local, slot)| {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let snip = snippet(g)?;
            Some(FeaturePoint {
                local,
                slot,
                t: s.cluster_index().display_time(g, key.reorder).0 as f32 / sr,
                f: std::array::from_fn(|i| project(&snip, i / 2, channels[i % 2] as usize)),
            })
        })
        .collect();
    (channels, points)
}

/// The fitted basis for `key.cfg`, fitting it on [`N_FIT`] peak-channel
/// snippets spread over the recording when there is none yet.
fn fitted_basis<P: DataProvider>(
    s: &Session<P>,
    key: &FeatureKey,
    basis: &Basis,
    half: u32,
    cancel: &AtomicBool,
) -> Option<Arc<[Vec<f32>; 2]>> {
    let mut guard = basis.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((cfg, h, b)) = guard.as_ref() {
        if *cfg == key.cfg && *h == half {
            return Some(Arc::clone(b));
        }
    }
    let p = &s.provider;
    let nc = p.n_channels() as usize;
    let len = 2 * half as usize + 1;
    let times = s.cluster_index().all_spike_times();
    let step = (times.len() / N_FIT).max(1);
    let rows: Vec<Vec<f64>> = times
        .par_iter()
        .step_by(step)
        .filter_map(|&t| {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let snip = extract_snippet(p, t, half, half, key.cfg, &key.shanks)?;
            let ptp = |ch: usize| {
                let col = (0..len).map(|i| snip[i * nc + ch]);
                col.clone().fold(f32::MIN, f32::max) - col.fold(f32::MAX, f32::min)
            };
            let peak = (0..nc).max_by(|&a, &b| ptp(a).total_cmp(&ptp(b)))?;
            Some((0..len).map(|i| snip[i * nc + peak] as f64).collect())
        })
        .collect();
    if cancel.load(Ordering::Relaxed) || rows.len() < 3 {
        return None;
    }
    let flat: Vec<f64> = rows.concat();
    let top = covariance(&flat, rows.len(), len).top_eigenvectors(2);
    let as_f32 = |v: &Vec<f64>| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let b = Arc::new([as_f32(top.first()?), as_f32(top.get(1)?)]);
    *guard = Some((key.cfg, half, Arc::clone(&b)));
    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_matches_phy_default() {
        // Diagonal is time vs the row's feature; off-diagonal x is the
        // column's feature (phy `time,0A 1A,0A 0B,0A 1B,0A`, …).
        let feats = [pc(0, 0), pc(1, 0), pc(0, 1), pc(1, 1)];
        for (r, row) in GRID.iter().enumerate() {
            for (c, &(x, y)) in row.iter().enumerate() {
                assert_eq!(y, feats[r]);
                assert_eq!(x, if r == c { Dim::Time } else { feats[c] });
            }
        }
    }

    #[test]
    fn lasso_hits_map_to_local_indices_per_cluster() {
        let pt = |local, slot, f0: f32, f1: f32| FeaturePoint {
            local,
            slot,
            t: 0.0,
            f: [f0, f1, 0.0, 0.0],
        };
        let set = FeatureSet {
            channels: [0, 1],
            computed: false,
            full: true,
            points: vec![
                pt(0, 0, 0.5, 0.5),
                pt(1, 0, 5.0, 5.0),
                pt(0, 1, 0.2, 0.8),
                pt(0, BACKGROUND, 0.5, 0.5),
            ],
            bounds: [1.0; 4],
            duration: 1.0,
            generation: 0,
        };
        // Cell 1 plots x = 1A (f[1]) against y = 0A (f[0]).
        let lasso = Lasso {
            cell: 1,
            points: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        };
        let hits = lasso_hits(&set, &lasso, &[ClusterId(7), ClusterId(9)]);
        assert_eq!(hits, vec![(ClusterId(7), vec![0]), (ClusterId(9), vec![0])]);
    }
}

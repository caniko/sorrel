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
/// Grid layout: gap between cells, the label strip above each cell, the
/// pane width one column wants, and the background marker radius (1 px).
const CELL_GAP: f32 = 6.0;
const LABEL_H: f32 = 14.0;
const COL_W: f32 = 140.0;
const BACKGROUND_RADIUS: f32 = 0.5;

/// Columns (2 to 4) and square cell side for a pane `width` wide.
fn grid_geometry(width: f32) -> (usize, f32) {
    let cols = ((width / COL_W) as usize).clamp(2, 4);
    (
        cols,
        ((width - (cols - 1) as f32 * CELL_GAP) / cols as f32).max(40.0),
    )
}

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

/// What a lasso is valid for: the key of the set on screen, without the
/// `full` flag (loading every spike keeps the same polygon) and with the
/// channels resolved.
fn lasso_key(key: &FeatureKey, shown: [u32; 2]) -> FeatureKey {
    FeatureKey {
        full: false,
        channels: Some(key.channels.unwrap_or(shown)),
        ..key.clone()
    }
}

struct FeatureSet {
    channels: [u32; 2],
    computed: bool,
    full: bool,
    points: Vec<FeaturePoint>,
    /// 0.5-99.5 percentile range (padded) per feature column.
    bounds: [(f32, f32); 4],
    duration: f32,
    generation: u64,
}

/// Per cell: `[background, selected]` vertex buffers.
type CellVertices = Vec<[Arc<Vec<ScatterVertex>>; 2]>;

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
    /// The data the lasso was drawn against ([`lasso_key`]); a different
    /// key drops the lasso and any pending split.
    lasso_key: Option<FeatureKey>,
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
    pub km: KmState,
    /// Closed-lasso hit count for `(set generation, lasso vertices)`.
    hits: Option<(u64, Vec<[f32; 2]>, usize)>,
}

/// The k-means auto-lasso: fit on the first selected cluster's features
/// (rayon, off the UI thread), wrap the lasso around one part.
#[derive(Default)]
pub struct KmState {
    /// `None` picks k in 2..=5 by silhouette.
    pub k: Option<usize>,
    /// Refit whenever the features change; the lasso follows unless the
    /// user drew or edited one.
    pub auto_run: bool,
    /// The part the lasso wraps (index into the fit's parts).
    pub part: usize,
    requested: bool,
    /// The fit already turned into a lasso.
    applied: Option<KmKey>,
    /// The lasso came from k-means, so a newer fit may replace it.
    lasso_auto: bool,
    /// The split preview shows a k-means part (not a proposal).
    previewing: bool,
    /// Bring the lasso's cell into view on the next frame.
    scroll_to_lasso: bool,
    memo: Memo<KmKey, Option<Arc<KmFit>>>,
    /// The last finished fit and the key it was made for.
    fit: Option<(KmKey, Option<Arc<KmFit>>)>,
}

/// `(feature set generation, cluster, k choice)`.
type KmKey = (u64, ClusterId, Option<usize>);

/// k-means parts of one cluster's points, largest first, with a lasso
/// around each.
struct KmFit {
    /// Indices into `FeatureSet::points`.
    parts: Vec<Vec<usize>>,
    lassos: Vec<Lasso>,
}

/// Fit on at most this many rows, then label every row.
const KM_FIT_ROWS: usize = 5000;

/// k-means on the 4 PC columns of `slot`'s points, each scaled to `[0, 1]`
/// by the plot bounds. `k = None` tries 2..=5 in parallel and keeps the
/// best silhouette. `None` when the cluster has too few points.
fn fit_kmeans(set: &FeatureSet, slot: u16, k: Option<usize>) -> Option<KmFit> {
    use sorrel_compute::{kmeans, kmeans_assign};
    let idx: Vec<usize> = (0..set.points.len())
        .filter(|&i| set.points[i].slot == slot && set.points[i].f.iter().all(|v| v.is_finite()))
        .collect();
    if idx.len() < 10 {
        return None;
    }
    let data: Vec<f32> = idx
        .iter()
        .flat_map(|&i| {
            let f = set.points[i].f;
            (0..4).map(move |j| {
                let (lo, hi) = set.bounds[j];
                (f[j] - lo) / (hi - lo).max(f32::MIN_POSITIVE)
            })
        })
        .collect();
    let stride = idx.len().div_ceil(KM_FIT_ROWS);
    let sub: Vec<f32> = data
        .as_chunks::<4>()
        .0
        .iter()
        .step_by(stride)
        .flatten()
        .copied()
        .collect();
    let ks: Vec<usize> = k.map_or((2..=5).collect(), |k| vec![k.clamp(2, 8)]);
    let km = ks
        .into_par_iter()
        .map(|k| {
            let km = kmeans(&sub, 4, k, 50);
            (km.silhouette(&sub, 4), km)
        })
        .max_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then(b.1.centroids.len().cmp(&a.1.centroids.len()))
        })?
        .1;
    let mut parts: Vec<Vec<usize>> = vec![Vec::new(); km.centroids.len() / 4];
    for (row, l) in kmeans_assign(&data, 4, &km.centroids)
        .into_iter()
        .enumerate()
    {
        parts[l as usize].push(idx[row]);
    }
    parts.retain(|p| !p.is_empty());
    if parts.len() < 2 {
        return None;
    }
    parts.sort_by_key(|p| std::cmp::Reverse(p.len()));
    let lassos = (0..parts.len())
        .into_par_iter()
        .map(|i| auto_lasso(set, &parts, i))
        .collect();
    Some(KmFit { parts, lassos })
}

/// The lasso around `parts[target]`: in each grid cell, the convex hull of
/// the 95% of the part's points nearest its median; the cell whose hull
/// holds the fewest points of the other parts wins (then the smaller hull).
// ponytail: a polygon in one 2-D view only approximates a 4-D k-means
// part; the split preview shows the exact membership next to it.
fn auto_lasso(set: &FeatureSet, parts: &[Vec<usize>], target: usize) -> Lasso {
    let mut best: Option<(usize, f32, usize, Vec<[f32; 2]>)> = None;
    for (cell, &(dx, dy)) in GRID.iter().flatten().enumerate() {
        let xy = |i: usize| [set.points[i].value(dx), set.points[i].value(dy)];
        let mut pts: Vec<[f32; 2]> = parts[target].iter().map(|&i| xy(i)).collect();
        if pts.len() < 3 {
            continue;
        }
        let median = |a: usize| {
            let mut v: Vec<f32> = pts.iter().map(|p| p[a]).collect();
            let m = v.len() / 2;
            *v.select_nth_unstable_by(m, f32::total_cmp).1
        };
        let (mx, my) = (median(0), median(1));
        // Distances scaled per axis so time and PCs weigh alike.
        let (sx, sy) = pts.iter().fold((0.0f32, 0.0f32), |(a, b), p| {
            (a.max((p[0] - mx).abs()), b.max((p[1] - my).abs()))
        });
        let dist = |p: &[f32; 2]| {
            ((p[0] - mx) / sx.max(f32::MIN_POSITIVE)).powi(2)
                + ((p[1] - my) / sy.max(f32::MIN_POSITIVE)).powi(2)
        };
        pts.sort_by(|a, b| dist(a).total_cmp(&dist(b)));
        pts.truncate((pts.len() * 95).div_ceil(100).max(3));
        let mut hull = convex_hull(pts);
        if hull.len() < 3 {
            continue;
        }
        // Grow it 5% about its centroid: the hull's own vertices sit on the
        // boundary, where the even-odd test may call them outside.
        let n = hull.len() as f32;
        let (cx, cy) = hull
            .iter()
            .fold((0.0, 0.0), |(a, b), p| (a + p[0] / n, b + p[1] / n));
        for p in &mut hull {
            *p = [cx + (p[0] - cx) * 1.05, cy + (p[1] - cy) * 1.05];
        }
        let probe = Lasso {
            cell,
            points: hull.clone(),
        };
        let contamination = parts
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != target)
            .flat_map(|(_, p)| p.iter())
            .filter(|&&i| probe.contains(xy(i)))
            .count();
        // Area in units of each axis' spread, so cells compare.
        let area = polygon_area(&hull) / (sx * sy).max(f32::MIN_POSITIVE);
        if best
            .as_ref()
            .is_none_or(|b| (contamination, area) < (b.2, b.1))
        {
            best = Some((cell, area, contamination, hull));
        }
    }
    best.map_or_else(Lasso::default, |(cell, _, _, points)| Lasso {
        cell,
        points,
    })
}

/// Andrew's monotone chain; counter-clockwise, no repeated last point.
fn convex_hull(mut pts: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    pts.retain(|p| p[0].is_finite() && p[1].is_finite());
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: [f32; 2], a: [f32; 2], b: [f32; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut hull: Vec<[f32; 2]> = Vec::with_capacity(2 * pts.len());
    for pass in [&pts[..], &pts.iter().rev().copied().collect::<Vec<_>>()[..]] {
        let start = hull.len();
        for &p in pass {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull
}

fn polygon_area(poly: &[[f32; 2]]) -> f32 {
    let n = poly.len();
    (0..n)
        .map(|i| {
            let (a, b) = (poly[i], poly[(i + 1) % n]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f32>()
        .abs()
        / 2.0
}

impl Default for FeatureViewState {
    fn default() -> Self {
        Self {
            fixed_channels: None,
            scaling: 1.0,
            marker: 2.5,
            zoom: 1.0,
            pan: [0.0; 2],
            lasso: Lasso::default(),
            lasso_key: None,
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
            km: KmState::default(),
            hits: None,
        }
    }
}

impl FeatureViewState {
    /// Abandon an in-progress lasso (Esc).
    pub fn cancel_lasso(&mut self) {
        self.lasso.clear();
        self.lasso_key = None;
        self.split_requested = false;
        self.want_full = false;
        self.clear_km_preview();
    }

    /// Drop the k-means highlight (not a proposal's) with the lasso it drew.
    fn clear_km_preview(&mut self) {
        if self.km.previewing {
            self.preview = None;
            self.preview_gen += 1;
            self.km.previewing = false;
        }
    }

    pub fn lasso_active(&self) -> bool {
        self.lasso.is_active()
    }

    pub fn lasso_closed(&self) -> bool {
        self.lasso.is_closed()
    }

    /// Drop the lasso and a pending split drawn against other data than `cur`.
    fn drop_stale_lasso(&mut self, cur: &FeatureKey) {
        if self.lasso_key.as_ref().is_some_and(|k| k != cur) {
            self.cancel_lasso();
        }
    }

    /// End of frame: stamp a fresh lasso with `cur` once its set is loaded
    /// (a lasso drawn over a stale set is dropped), and forget the stamp
    /// once the lasso is gone.
    fn stamp_lasso(&mut self, cur: &FeatureKey, loaded: bool) {
        if !self.lasso.is_active() {
            self.lasso_key = None;
        } else if self.lasso_key.is_none() {
            if loaded {
                self.lasso_key = Some(cur.clone());
            } else {
                self.cancel_lasso(); // drawn against a stale set
            }
        }
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

    /// Highlight a proposed split (`(cluster, local spike indices)` as in
    /// `CurationCommand::Split`) until it is cleared with `None`.
    pub fn set_split_preview(&mut self, preview: Option<(ClusterId, Vec<u32>)>) {
        self.preview = preview.map(|(c, idx)| (c, idx.into_iter().collect()));
        self.preview_gen += 1;
        self.km.previewing = false;
    }

    /// Run k-means on the first selected cluster and wrap the lasso around
    /// the smallest part (the k-means button, `:km`). `k = None` picks k by
    /// silhouette.
    pub fn request_kmeans(&mut self, k: Option<Option<usize>>) {
        if let Some(k) = k {
            self.km.k = k;
        }
        self.km.requested = true;
        self.km.applied = None;
    }

    /// The `(channel, pc)` the AmplitudeView's feature type plots: the last
    /// Alt+clicked dimension (phy `select_feature`), else PC1 on the first
    /// channel shown.
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
    let bound_key = key.clone();
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
    let cur = lasso_key(&bound_key, set.channels);
    // A lasso from before a selection/channel change must not hit the new
    // set's points.
    state.drop_stale_lasso(&cur);

    header(ui, state, &set, current.is_none());
    kmeans_controls(ui, state, &set, selection[0], jobs);

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

    egui::ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| {
            let (cols, side) = grid_geometry(ui.available_width());
            let rows = GRID.len() * GRID[0].len() / cols;
            let (grid_rect, _) = ui.allocate_exact_size(
                egui::vec2(
                    cols as f32 * side + (cols - 1) as f32 * CELL_GAP,
                    rows as f32 * (side + LABEL_H) + (rows - 1) as f32 * CELL_GAP,
                ),
                Sense::hover(),
            );
            let painter = ui.painter_at(grid_rect);
            for (cell, &(dx, dy)) in GRID.iter().flatten().enumerate() {
                let (row, col) = (cell / cols, cell % cols);
                let origin = grid_rect.min
                    + egui::vec2(
                        col as f32 * (side + CELL_GAP),
                        row as f32 * (side + LABEL_H + CELL_GAP),
                    );
                let strip = Rect::from_min_size(origin, egui::vec2(side, LABEL_H));
                let rect =
                    Rect::from_min_size(origin + egui::vec2(0.0, LABEL_H), egui::vec2(side, side));
                if state.km.scroll_to_lasso && state.lasso.cell == cell {
                    state.km.scroll_to_lasso = false;
                    ui.scroll_to_rect(strip.union(rect), Some(egui::Align::Center));
                }
                if !ui.is_rect_visible(strip.union(rect)) {
                    continue;
                }
                painter.with_clip_rect(strip).text(
                    strip.left_center() + egui::vec2(2.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    format!("{} × {}", dim_label(dx, &set), dim_label(dy, &set)),
                    egui::FontId::proportional(11.0),
                    theme.caption,
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
                // Background first (1 px, faint), the selection on top.
                if let Some([background, selected]) = cells.get(cell) {
                    for (layer, vertices, radius) in [
                        (
                            "feature-bg",
                            background,
                            if dark { 0.75 } else { BACKGROUND_RADIUS },
                        ),
                        ("feature-cell", selected, state.marker),
                    ] {
                        ui.painter()
                            .add(eframe::egui_wgpu::Callback::new_paint_callback(
                                rect,
                                ScatterCallback {
                                    id: ui.id().with((layer, cell)).value(),
                                    vertices: Arc::clone(vertices),
                                    generation: cells_gen,
                                    x,
                                    y,
                                    radius,
                                    size: rect.size(),
                                },
                            ));
                    }
                }
                painter.rect_stroke(
                    rect,
                    0.0,
                    Stroke::new(0.5_f32, theme.grid),
                    egui::StrokeKind::Inside,
                );
                if state.lasso.is_active() && state.lasso.cell == cell {
                    // The k-means lasso's cell is outlined, so it reads at a
                    // glance which of the 16 views the polygon lives in.
                    if state.km.previewing {
                        painter.rect_stroke(
                            rect,
                            0.0,
                            Stroke::new(2.0_f32, ui.visuals().selection.bg_fill),
                            egui::StrokeKind::Inside,
                        );
                    }
                    state
                        .lasso
                        .paint(&painter, to_screen, ui.visuals().strong_text_color());
                    if state.lasso.is_closed() {
                        let n = lasso_hit_count(state, &set);
                        hit_badge(
                            &painter.with_clip_rect(rect),
                            &state.lasso,
                            to_screen,
                            rect,
                            n,
                            &theme,
                            ui.visuals().strong_text_color(),
                        );
                    }
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
                // phy `select_feature`: Alt+click picks the x (left) or y
                // (right) dimension for the AmplitudeView feature type.
                let alt = ui.input(|i| i.modifiers.alt);
                let picked = match (alt, resp.clicked(), resp.secondary_clicked()) {
                    (true, true, _) => Some(dx),
                    (true, _, true) => Some(dy),
                    _ => None,
                };
                if let Some(Dim::Pc { ch, pc }) = picked {
                    state.amplitude_dim = Some((set.channels[ch as usize], pc as usize));
                } else if state.lasso.interact(cell, &resp, to_data) {
                    // A hand-edited lasso is the user's: auto-run leaves it.
                    state.km.lasso_auto = false;
                    if !state.lasso.is_active() {
                        state.clear_km_preview();
                    }
                } else {
                    interact_view(ui, state, &resp);
                }
            }
        });

    state.stamp_lasso(&cur, current.is_some());
    if state.split_requested && state.lasso.is_closed() {
        match current.filter(|s| s.full || !s.computed) {
            Some(set) => {
                state.split_requested = false;
                state.want_full = false;
                let splits = lasso_hits(&set, &state.lasso, selection);
                state.cancel_lasso();
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
    // One short line whatever it says, so the grid never moves under the
    // cursor while a lasso is being drawn; the full gesture list is on hover.
    let hint = match (state.lasso.is_active(), state.lasso.is_closed()) {
        (_, true) => "K: split · Esc: clear",
        (true, false) => "Ctrl+click: add point · Ctrl+right-click: clear",
        _ => "Ctrl+click: lasso · K: split",
    };
    ui.horizontal(|ui| {
        ui.add(egui::Label::new(egui::RichText::new(hint).weak()).truncate());
        // The old "ⓘ" glyph rendered as an empty checkbox-like box (missing
        // glyph); plain ASCII always draws.
        ui.label("(?)").on_hover_text(
            "Ctrl+click: add lasso point · Ctrl+right-click: clear lasso\n\
             K: split the lassoed spikes · Esc: cancel\n\
             wheel: zoom · drag: pan · double-click: reset\n\
             Ctrl+wheel: feature scale · Alt+wheel: marker size\n\
             Alt+click / Alt+right-click: amplitude feature (x / y)",
        );
    });
}

/// The k-means row: button, k, auto-run and one chip per part. Runs the
/// fit as a job and, when it lands, wraps the lasso around the smallest
/// part unless the user drew their own lasso since.
fn kmeans_controls<P: DataProvider>(
    ui: &mut Ui,
    state: &mut FeatureViewState,
    set: &Arc<FeatureSet>,
    anchor: ClusterId,
    jobs: &JobCtx<Session<P>>,
) {
    let key: KmKey = (set.generation, anchor, state.km.k);
    let mut chosen = None;
    ui.horizontal_wrapped(|ui| {
        if ui
            .button("k-means")
            .on_hover_text(format!(
                "Cluster c{anchor}'s spikes in PC space and lasso the smallest part; K splits it"
            ))
            .clicked()
        {
            state.request_kmeans(None);
        }
        egui::ComboBox::from_id_salt("km-k")
            .width(64.0)
            .selected_text(
                state
                    .km
                    .k
                    .map_or("k: auto".to_string(), |k| format!("k: {k}")),
            )
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut state.km.k, None, "auto")
                    .on_hover_text("best silhouette for k = 2..5");
                for k in 2..=6 {
                    ui.selectable_value(&mut state.km.k, Some(k), k.to_string());
                }
            });
        ui.checkbox(&mut state.km.auto_run, "auto-run")
            .on_hover_text(
                "Refit (on all cores, off the UI thread) whenever the selection or features \
             change; the lasso follows unless you drew one",
            );
        if state.km.memo.pending() {
            ui.spinner();
        }
        match &state.km.fit {
            Some((k, Some(fit))) if *k == key => {
                for (i, part) in fit.parts.iter().enumerate() {
                    let label = format!("part {} · {}", i + 1, part.len());
                    if ui
                        .selectable_label(state.km.part == i && state.km.previewing, label)
                        .clicked()
                    {
                        chosen = Some(i);
                    }
                }
                if state.km.previewing && state.lasso.is_closed() {
                    let (dx, dy) = GRID[state.lasso.cell / 4][state.lasso.cell % 4];
                    ui.weak(format!(
                        "lasso in {} × {}",
                        dim_label(dx, set),
                        dim_label(dy, set)
                    ));
                }
            }
            Some((k, None)) if *k == key => {
                ui.weak("too few spikes to cluster");
            }
            _ => {}
        }
    });

    if state.km.requested || state.km.auto_run {
        let (s, k) = (Arc::clone(set), state.km.k);
        if let Some(fit) = state
            .km
            .memo
            .ensure(key, &jobs.session, jobs.repaint(), move |_, _| {
                fit_kmeans(&s, 0, k).map(Arc::new)
            })
            .cloned()
        {
            state.km.fit = Some((key, fit.clone()));
            if state.km.applied != Some(key) {
                let take = state.km.requested || !state.lasso.is_active() || state.km.lasso_auto;
                if let (true, Some(fit)) = (take, fit) {
                    state.km.part = fit.parts.len() - 1;
                    apply_part(state, set, &fit, anchor);
                }
                state.km.applied = Some(key);
                state.km.requested = false;
            }
        }
    }
    if let (Some(i), Some((_, Some(fit)))) = (chosen, state.km.fit.clone()) {
        state.km.part = i;
        apply_part(state, set, &fit, anchor);
    }
}

/// Lasso the fit's part `state.km.part` and highlight its exact members.
fn apply_part(state: &mut FeatureViewState, set: &FeatureSet, fit: &KmFit, anchor: ClusterId) {
    let part = state.km.part.min(fit.parts.len() - 1);
    state.lasso = fit.lassos[part].clone();
    state.km.lasso_auto = true;
    state.preview = Some((
        anchor,
        fit.parts[part]
            .iter()
            .map(|&i| set.points[i].local)
            .collect(),
    ));
    state.preview_gen += 1;
    state.km.previewing = true;
    state.km.scroll_to_lasso = true;
}

/// Selected spikes inside the closed lasso, cached per set and polygon.
fn lasso_hit_count(state: &mut FeatureViewState, set: &FeatureSet) -> usize {
    if let Some((g, pts, n)) = &state.hits {
        if *g == set.generation && *pts == state.lasso.points {
            return *n;
        }
    }
    let (dx, dy) = GRID[state.lasso.cell / 4][state.lasso.cell % 4];
    let n = set
        .points
        .iter()
        .filter(|p| p.slot != BACKGROUND && state.lasso.contains([p.value(dx), p.value(dy)]))
        .count();
    state.hits = Some((set.generation, state.lasso.points.clone(), n));
    n
}

/// "N spikes · K splits · Esc clears" under the lasso, kept inside `cell`.
fn hit_badge(
    painter: &egui::Painter,
    lasso: &Lasso,
    to_screen: impl Fn([f32; 2]) -> Pos2,
    cell: Rect,
    n: usize,
    theme: &Theme,
    text: egui::Color32,
) {
    let bbox = Rect::from_points(
        &lasso
            .points
            .iter()
            .map(|&p| to_screen(p))
            .collect::<Vec<_>>(),
    );
    let galley = painter.layout_no_wrap(
        format!("{n} spikes · K splits · Esc clears"),
        egui::FontId::proportional(11.0),
        text,
    );
    let size = galley.size() + egui::vec2(8.0, 4.0);
    let min = egui::pos2(bbox.center().x - size.x / 2.0, bbox.bottom() + 4.0)
        .clamp(cell.min, (cell.max - size).max(cell.min));
    let r = Rect::from_min_size(min, size);
    painter.rect_filled(r, 3.0, theme.plot_bg.gamma_multiply(0.85));
    painter.rect_stroke(
        r,
        3.0,
        Stroke::new(0.5_f32, theme.grid),
        egui::StrokeKind::Inside,
    );
    painter.galley(r.min + egui::vec2(4.0, 2.0), galley, text);
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
            let (a, b) = set.bounds[pc as usize * 2 + ch as usize];
            let half = (b - a) / 2.0 / state.scaling;
            (((a + b) / 2.0) - half, ((a + b) / 2.0) + half)
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
) -> CellVertices {
    // Faint background: gray(170)@~0.47 in dark, gray(190)@0.5 in light.
    let background = if dark {
        pack_rgba(170, 170, 170, 120)
    } else {
        pack_rgba(190, 190, 190, 128)
    };
    let highlight = if dark {
        pack_rgba(255, 255, 255, 240)
    } else {
        pack_rgba(20, 20, 20, 240)
    };
    let previewed = |p: &FeaturePoint| {
        preview.is_some_and(|(slot, idx)| p.slot == slot && idx.contains(&p.local))
    };
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
            // `order` is sorted background-first, so a stable partition keeps
            // the previewed split last within the selection.
            let layer = |bg: bool| {
                Arc::new(
                    order
                        .iter()
                        .filter(|p| (p.slot == BACKGROUND) == bg)
                        .filter_map(|p| {
                            let (x, y) = (p.value(dx), p.value(dy));
                            (x.is_finite() && y.is_finite()).then(|| ScatterVertex {
                                pos: [x, y],
                                color: if bg {
                                    background
                                } else if previewed(p) {
                                    highlight
                                } else {
                                    let c = selection_colour(p.slot as usize);
                                    pack_rgba(c.r(), c.g(), c.b(), 255)
                                },
                            })
                        })
                        .collect(),
                )
            };
            [layer(true), layer(false)]
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
            robust_range(
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

/// 0.5-99.5 percentile range of the finite values, padded 5% each side
/// (`(-1, 1)` when there are none).
fn robust_range(vals: impl Iterator<Item = f32>) -> (f32, f32) {
    let mut v: Vec<f32> = vals.filter(|v| v.is_finite()).collect();
    if v.is_empty() {
        return (-1.0, 1.0);
    }
    v.sort_unstable_by(f32::total_cmp);
    let at = |q: f32| v[((v.len() - 1) as f32 * q).round() as usize];
    let (lo, hi) = (at(0.005), at(0.995));
    let pad = ((hi - lo) * 0.05).max(1e-6);
    (lo - pad, hi + pad)
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
    fn lasso_is_dropped_when_the_feature_key_changes() {
        let key = FeatureKey {
            selection: vec![ClusterId(1)],
            channels: None,
            version: 0,
            cfg: TracePreproc::default(),
            shanks: Vec::new(),
            full: false,
            reorder: false,
        };
        let cur = lasso_key(&key, [3, 4]);
        let mut st = FeatureViewState::default();
        let tri = Lasso {
            cell: 0,
            points: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        };
        st.lasso = tri.clone();
        st.stamp_lasso(&cur, false);
        assert!(!st.lasso.is_active(), "drawn over a stale set");
        st.lasso = tri;
        st.stamp_lasso(&cur, true);
        st.request_split();
        // Loading every spike keeps the same polygon.
        let full = FeatureKey {
            full: true,
            ..key.clone()
        };
        st.drop_stale_lasso(&lasso_key(&full, [3, 4]));
        assert!(st.lasso.is_closed() && st.split_requested);
        // A new selection drops lasso and the pending split.
        let other = FeatureKey {
            selection: vec![ClusterId(2)],
            ..key
        };
        st.drop_stale_lasso(&lasso_key(&other, [3, 4]));
        assert!(!st.lasso.is_active() && !st.split_requested);
    }

    #[test]
    fn grid_geometry_clamps_columns_and_stays_square() {
        for (w, cols) in [(100.0, 2), (300.0, 2), (420.0, 3), (700.0, 4), (2000.0, 4)] {
            let (c, side) = grid_geometry(w);
            assert_eq!(c, cols, "width {w}");
            assert!(c as f32 * side + (c - 1) as f32 * CELL_GAP <= w.max(86.0) + 1e-3);
        }
    }

    #[test]
    fn robust_range_ignores_outliers() {
        let mut v: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        v.push(1e6);
        let (lo, hi) = robust_range(v.into_iter());
        assert!(lo < 5.0 && hi > 990.0 && hi < 1100.0, "{lo} {hi}");
        assert_eq!(robust_range(std::iter::empty()), (-1.0, 1.0));
    }

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
    fn convex_hull_drops_interior_points() {
        let pts = vec![
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [0.5, 1.5],
            [2.0, 1.0],
        ];
        let hull = convex_hull(pts);
        assert_eq!(hull, vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]);
        assert_eq!(polygon_area(&hull), 4.0);
    }

    /// 400 spikes around the origin and 100 around 5 in all four PC
    /// columns, plus background that must be ignored.
    fn two_blobs() -> FeatureSet {
        let mut s = 0x5EED_u32;
        let mut noise = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s as f32 / u32::MAX as f32 - 0.5
        };
        let mut points = Vec::new();
        for local in 0..500u32 {
            let c = if local < 400 { 0.0 } else { 5.0 };
            let f = [c + noise(), c + noise(), c + noise(), c + noise()];
            points.push(FeaturePoint {
                local,
                slot: 0,
                t: local as f32 / 500.0,
                f,
            });
        }
        for _ in 0..50 {
            points.push(FeaturePoint {
                local: 0,
                slot: BACKGROUND,
                t: 0.5,
                f: [2.5; 4],
            });
        }
        FeatureSet {
            channels: [0, 1],
            computed: false,
            full: true,
            points,
            bounds: [(-1.0, 6.0); 4],
            duration: 1.0,
            generation: 0,
        }
    }

    #[test]
    fn kmeans_auto_lasso_wraps_the_small_blob() {
        let set = two_blobs();
        let fit = fit_kmeans(&set, 0, None).expect("two blobs cluster");
        assert_eq!(fit.parts.len(), 2, "k = auto picks 2 on two blobs");
        assert_eq!(
            fit.parts.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![400, 100]
        );
        // The smallest part (last) is the lasso the button draws.
        let lasso = &fit.lassos[1];
        assert!(lasso.is_closed());
        let (dx, dy) = GRID[lasso.cell / 4][lasso.cell % 4];
        let inside = |p: &FeaturePoint| lasso.contains([p.value(dx), p.value(dy)]);
        let big = set.points.iter().filter(|p| p.slot == 0 && p.local < 400);
        assert_eq!(
            big.filter(|p| inside(p)).count(),
            0,
            "no spike of the big blob"
        );
        let small = set.points.iter().filter(|p| p.slot == 0 && p.local >= 400);
        // The hull trims the 5% outermost points, so most but not all.
        assert!(small.filter(|p| inside(p)).count() >= 90);
        assert!(fit_kmeans(&set, 0, Some(3)).is_some_and(|f| f.parts.len() == 3));
        assert!(
            fit_kmeans(&set, 7, None).is_none(),
            "no points for that slot"
        );
    }

    /// `cargo test -p sorrel-ui --release kmeans_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn kmeans_timing_on_100k_spikes() {
        let mut set = two_blobs();
        let base = set.points.clone();
        set.points = (0..200)
            .flat_map(|r| {
                base.iter().map(move |p| FeaturePoint {
                    local: p.local + r * 500,
                    ..*p
                })
            })
            .collect();
        let t = std::time::Instant::now();
        let fit = fit_kmeans(&set, 0, None).unwrap();
        eprintln!(
            "k=auto on {} spikes: {:?}, parts {:?}",
            set.points.len(),
            t.elapsed(),
            fit.parts.iter().map(Vec::len).collect::<Vec<_>>()
        );
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
            bounds: [(-1.0, 1.0); 4],
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

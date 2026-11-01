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
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui, Vec2};
use rayon::prelude::*;
use sorrel_compute::linalg::covariance;
use sorrel_data::{Session, SpikeDepths};
use sorrel_io::{ClusterId, DataProvider, SampleIndex};
use sorrel_render::{extract_snippet, TracePreproc};

use crate::brush::{BrushSource, Brushes, Ghost, HoverSpike, SpikeBrush, TimeBrush};
use crate::feature_view3d::{feature_3d, Overlay, PointColour};
use crate::jobs::{JobCtx, Memo};
use crate::lasso::{Lasso, LassoEvent};
use crate::theme::{selection_colour, viridis, Theme};
use crate::views::waveform_window_samples;
use crate::wgpu_raster::pack_rgba;
use crate::wgpu_scatter::{ScatterCallback, ScatterVertex};

/// A change to the feature channel set, from a click in Waveforms or the
/// toolbar chips.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ChannelEdit {
    /// Add the channel to the set, or remove it (never below two or above
    /// [`MAX_SET`]).
    Toggle(u32),
    /// phy `select_channel`: make it A (swapping with B when it is B), and
    /// fix A/B.
    SetA(u32),
    /// Likewise for B.
    SetB(u32),
    /// Back to the automatic set.
    Reset,
}

/// What Features tells Waveforms: the channel set, the channels of the
/// focused view, the spikes a lasso or previewed part picks out and the
/// temporal PC shapes on the highlighted channels. Cheap to clone (Arcs).
#[derive(Clone, Default)]
pub struct FeatureLink {
    /// The channel set, A and B first.
    pub set: Vec<u32>,
    /// Channels of the focused cell (the 3-D scene's triple in 3-D).
    pub highlight: Vec<u32>,
    /// Closed-lasso hits or the previewed part's members per cluster, as
    /// local spike indices, at most 100 per cluster (evenly thinned).
    pub spikes: Vec<(ClusterId, std::sync::Arc<[u32]>)>,
    /// PC1 and PC2 temporal shapes on each highlighted channel.
    pub basis: Vec<(u32, std::sync::Arc<[Vec<f32>; 2]>)>,
}

/// One axis of a grid cell: time, or PC `pc` on the `ch`-th best channel
/// (phy's `0A`, `1B`, …).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Dim {
    Time,
    Pc { ch: u8, pc: u8 },
}

const fn pc(ch: u8, pc: u8) -> Dim {
    Dim::Pc { ch, pc }
}

/// phy `FeatureView` default grid, `(x, y)` per cell, row-major.
pub(crate) const GRID: [[(Dim, Dim); 4]; 4] = [
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
pub(crate) const BACKGROUND: u16 = u16::MAX;
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

/// Channels in the set at most.
// ponytail: 8 channels give 17 dims (time + 2 PCs each) and 680 triples; the
// pair and triple scoring stays fast on 5000 rows, so raise it if it still
// does on real recordings.
pub(crate) const MAX_SET: usize = 8;

/// A selected point; its features are `FeatureSet::row`.
#[derive(Copy, Clone, Debug)]
pub(crate) struct FeaturePoint {
    /// Position within the cluster's time-sorted bucket (the `Split` index).
    pub(crate) local: u32,
    pub(crate) slot: u16,
    pub(crate) t: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FeatureKey {
    pub(crate) selection: Vec<ClusterId>,
    /// The channel set; `None` is the selection's best four.
    pub(crate) channels: Option<Vec<u32>>,
    /// A/B stay the set's first two (`c`, or a click chose them) instead of
    /// following the best-separating channel pair.
    pub(crate) keep_order: bool,
    pub(crate) version: u64,
    pub(crate) cfg: TracePreproc,
    pub(crate) shanks: Vec<u32>,
    /// Every spike of the selected clusters, not just the display sample.
    pub(crate) full: bool,
    /// phy `toggle_spike_reorder`: time axes use `spike_reorder.npy`.
    pub(crate) reorder: bool,
}

/// What a lasso is valid for: the key of the set on screen, without the
/// `full` flag (loading every spike keeps the same polygon) and with the
/// channels resolved to A and B (the other channels do not move a lasso).
fn lasso_key(key: &FeatureKey, shown: &[u32]) -> FeatureKey {
    FeatureKey {
        full: false,
        keep_order: false,
        channels: Some(shown.iter().take(2).copied().collect()),
        ..key.clone()
    }
}

pub(crate) struct FeatureSet {
    /// The channel set, A and B first.
    pub(crate) channels: Vec<u32>,
    pub(crate) computed: bool,
    pub(crate) full: bool,
    pub(crate) points: Vec<FeaturePoint>,
    /// Row `i` is point `i`'s `2 * channels.len()` features, PC-major:
    /// index `pc * n + ch` (phy's `0A 1A 0B 1B` for two channels). A selected
    /// spike whose template lacks a channel reads 0 there (phylib
    /// `from_sparse`); background spikes read NaN.
    pub(crate) feats: Vec<f32>,
    /// 0.5-99.5 percentile range (padded) per feature column.
    pub(crate) bounds: Vec<(f32, f32)>,
    pub(crate) duration: f32,
    pub(crate) generation: u64,
}

impl FeatureSet {
    /// A set over `channels` from `points` and their flat `2·n` features each;
    /// bounds from the selected points.
    pub(crate) fn new(
        channels: Vec<u32>,
        computed: bool,
        full: bool,
        points: Vec<FeaturePoint>,
        feats: Vec<f32>,
        duration: f32,
    ) -> Self {
        let width = 2 * channels.len();
        assert_eq!(feats.len(), points.len() * width);
        let bounds = (0..width)
            .map(|j| {
                robust_range(
                    points
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| p.slot != BACKGROUND)
                        .map(|(i, _)| feats[i * width + j]),
                )
            })
            .collect();
        Self {
            channels,
            computed,
            full,
            points,
            feats,
            bounds,
            duration,
            generation: GENERATION.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// The set with channel `k` of the new order being channel `order[k]` of
    /// this one; a fresh generation.
    fn permuted(&self, order: &[usize]) -> Self {
        let n = self.n();
        let cols: Vec<usize> = (0..2 * n).map(|j| j / n * n + order[j % n]).collect();
        let feats = (0..self.points.len())
            .flat_map(|i| cols.iter().map(move |&c| (i, c)))
            .map(|(i, c)| self.row(i)[c])
            .collect();
        Self {
            channels: order.iter().map(|&o| self.channels[o]).collect(),
            computed: self.computed,
            full: self.full,
            points: self.points.clone(),
            feats,
            bounds: cols.iter().map(|&c| self.bounds[c]).collect(),
            duration: self.duration,
            generation: GENERATION.fetch_add(1, Ordering::Relaxed),
        }
    }

    pub(crate) fn n(&self) -> usize {
        self.channels.len()
    }

    /// Point `i`'s features.
    pub(crate) fn row(&self, i: usize) -> &[f32] {
        let w = 2 * self.n();
        &self.feats[i * w..(i + 1) * w]
    }

    /// Point `i` along `dim`.
    pub(crate) fn value(&self, i: usize, dim: Dim) -> f32 {
        match dim {
            Dim::Time => self.points[i].t,
            Dim::Pc { ch, pc } => self.row(i)[pc as usize * self.n() + ch as usize],
        }
    }

    /// Dimensions: time, then PC 1 on every channel, then PC 2.
    pub(crate) fn n_dims(&self) -> usize {
        1 + 2 * self.n()
    }

    pub(crate) fn dim(&self, idx: usize) -> Dim {
        match idx {
            0 => Dim::Time,
            _ => Dim::Pc {
                ch: ((idx - 1) % self.n()) as u8,
                pc: ((idx - 1) / self.n()) as u8,
            },
        }
    }

    pub(crate) fn dim_idx(&self, dim: Dim) -> usize {
        match dim {
            Dim::Time => 0,
            Dim::Pc { ch, pc } => 1 + pc as usize * self.n() + ch as usize,
        }
    }

    /// Data range of `dim` (time spans the recording).
    pub(crate) fn bound(&self, dim: Dim) -> (f32, f32) {
        match dim {
            Dim::Time => (0.0, self.duration.max(1e-3)),
            Dim::Pc { ch, pc } => self.bounds[pc as usize * self.n() + ch as usize],
        }
    }

    /// `v` along `dim`, `[0, 1]` across its range.
    pub(crate) fn unit(&self, dim: Dim, v: f32) -> f32 {
        let (lo, hi) = self.bound(dim);
        (v - lo) / (hi - lo).max(f32::MIN_POSITIVE)
    }
}

/// Per cell: `[background, selected, brushed]` vertex buffers.
type CellVertices = Vec<[Arc<Vec<ScatterVertex>>; 3]>;

/// How the selected points are coloured (the ⚙ popover's "colour by").
#[derive(Default, Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum ColourBy {
    #[default]
    Cluster,
    /// Recording time, on the viridis ramp.
    Time,
    /// Spike depth (needs `SpikeDepths`), likewise.
    Depth,
}

/// Brushed spikes are drawn in ink, spikes outside a time brush at this alpha.
const OUTSIDE_ALPHA: f32 = 0.2;
/// Pointer reach (px) of a spike for hover and Alt+click seek.
const HOVER_PX: f32 = 6.0;
/// Trace-window rings are skipped for a cluster with more spikes than this.
const WINDOW_RING_MAX: u32 = 200;

/// Everything besides the cluster that colours or hides a point.
struct Paint<'a> {
    dark: bool,
    colour: ColourBy,
    depths: Option<&'a SpikeDepths>,
    selection: &'a [ClusterId],
    brush: Option<&'a SpikeBrush>,
    time: Option<TimeBrush>,
}

impl Paint<'_> {
    /// Whether the spike brush holds this (selected) point.
    fn brushed(&self, p: &FeaturePoint) -> bool {
        self.brush.is_some_and(|b| {
            self.selection
                .get(p.slot as usize)
                .is_some_and(|&c| b.contains(c, p.local))
        })
    }

    /// Changes whenever a vertex buffer built with this paint would.
    fn key(&self) -> u64 {
        let mut h = DefaultHasher::new();
        (self.dark, self.colour).hash(&mut h);
        self.depths
            .map(|d| d as *const SpikeDepths as usize)
            .hash(&mut h);
        self.selection.hash(&mut h);
        hash_brush(self.brush, &mut h);
        self.time
            .map(|t| (t.start.to_bits(), t.end.to_bits(), t.only))
            .hash(&mut h);
        h.finish()
    }

    /// The colour of selected point `p` under the colour mode.
    fn point_colour(&self, p: &FeaturePoint, duration: f32) -> Color32 {
        let cluster = selection_colour(p.slot as usize);
        match self.colour {
            ColourBy::Cluster => cluster,
            ColourBy::Time => ramp_colour(p.t, 0.0, duration),
            ColourBy::Depth => self
                .depths
                .and_then(|d| {
                    let c = self.selection.get(p.slot as usize)?;
                    let v = *d.per_cluster.get(c.idx())?.get(p.local as usize)?;
                    Some(ramp_colour(v, d.min, d.max))
                })
                .unwrap_or(cluster),
        }
    }
}

fn hash_brush(b: Option<&SpikeBrush>, h: &mut impl Hasher) {
    b.map(|b| {
        let (source, version, n, first) = b.key();
        (std::mem::discriminant(&source), version, n, first)
    })
    .hash(h);
}

/// `v` within `[lo, hi]` on the viridis ramp (its middle for an empty range).
fn ramp_colour(v: f32, lo: f32, hi: f32) -> Color32 {
    viridis(if hi > lo { (v - lo) / (hi - lo) } else { 0.5 })
}

/// A spike at `t` seconds under the time brush: its alpha factor, `None` when
/// the brush hides it.
fn time_alpha(time: Option<TimeBrush>, t: f32) -> Option<f32> {
    match time {
        Some(b) if !b.contains(t) => (!b.only).then_some(OUTSIDE_ALPHA),
        _ => Some(1.0),
    }
}

/// The point nearest `at` within `max_px`, from `(id, screen position)`.
// ponytail: O(n) scan over the shown points (a few thousand per cluster);
// a grid index only if hover on a much larger sample ever lags.
fn nearest_spike(pts: impl Iterator<Item = (usize, Pos2)>, at: Pos2, max_px: f32) -> Option<usize> {
    pts.map(|(i, p)| (i, p.distance_sq(at)))
        .filter(|&(_, d)| d <= max_px * max_px)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

/// Local indices of the cluster's spikes inside `[start, start + len)`.
fn window_locals(times: &[SampleIndex], start: u64, len: u32) -> std::ops::Range<u32> {
    let end = start + len as u64;
    times.partition_point(|s| s.0 < start) as u32..times.partition_point(|s| s.0 < end) as u32
}

/// What `brushes.spikes` should be for the lasso / k-means part on screen:
/// its source and a key that changes with its hits.
type BrushWant = Option<(BrushSource, u64)>;

#[derive(Debug, PartialEq, Eq)]
enum BrushAction {
    Keep,
    Set,
    Clear,
}

/// Compare `want` with what was last written: only a change writes.
fn brush_action(last: &mut BrushWant, want: BrushWant) -> BrushAction {
    if *last == want {
        return BrushAction::Keep;
    }
    *last = want;
    if want.is_some() {
        BrushAction::Set
    } else {
        BrushAction::Clear
    }
}

/// Grey vertices per grid cell.
type GhostVerts = Arc<Vec<Arc<Vec<ScatterVertex>>>>;

/// The merge ghost's points per grid cell, grey under the selection.
#[derive(Default)]
struct GhostCells {
    memo: Memo<FeatureKey, Arc<FeatureSet>>,
    /// `((ghost generation, set generation, dark), buffer generation, cells)`.
    built: Option<((u64, u64, bool), u64, GhostVerts)>,
    generation: u64,
}

/// Temporal PC basis fitted on raw snippets, reused across selections.
type Basis = Arc<Mutex<Option<(TracePreproc, u32, Arc<[Vec<f32>; 2]>)>>>;

/// 2-D grid of PC projections, or the rotatable 3-D scene (feature_view3d).
#[derive(Default, Copy, Clone, PartialEq, Eq, Debug)]
pub enum ViewMode {
    #[default]
    Two,
    Three,
}

pub struct FeatureViewState {
    pub mode: ViewMode,
    pub(crate) view3d: crate::feature_view3d::View3dState,
    /// Grid cell the large cell shows (Compact and Expansive layouts).
    focus: usize,
    /// A and B kept as they are (phy's `c`, or a channel click) instead of
    /// following the best-separating pair of the set.
    pub fixed_channels: Option<[u32; 2]>,
    /// The channel set the user chose, A and B first; `None` follows the
    /// selection's best four.
    edited: Option<Vec<u32>>,
    /// `edited` only pins the set `c` froze, so unfixing drops it.
    edited_by_fix: bool,
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
    shown_channels: Option<Vec<u32>>,
    /// Alt+clicked `(channel, pc)` for the AmplitudeView feature type.
    amplitude_dim: Option<(u32, usize)>,
    memo: Memo<FeatureKey, Arc<FeatureSet>>,
    basis: Basis,
    /// Per-cell vertex buffers for `(set generation, dark mode)`.
    cells: Option<((u64, u64, u64), u64, CellVertices)>,
    /// The non-background points those buffers draw (hover, trace rings).
    shown: Arc<Vec<usize>>,
    /// The merge preview's grey points.
    ghost: GhostCells,
    /// How the selected points are coloured.
    pub colour: ColourBy,
    /// What `brushes.spikes` was last set from; it is written only on change.
    brush_written: BrushWant,
    /// This frame's hovered spike, the Alt+clicked one to seek to and the
    /// Shift+drag time brush (`Some(None)` clears it), for `feature_view` to
    /// hand on.
    hover_out: Option<HoverSpike>,
    pub(crate) seek: Option<u64>,
    time_out: Option<Option<TimeBrush>>,
    /// Time where a Shift+drag on a time axis started.
    time_from: Option<f32>,
    /// A proposed split shown before it is applied: the source cluster
    /// and the local indices that would move, drawn highlighted.
    preview: Option<(ClusterId, std::collections::HashSet<u32>)>,
    preview_gen: u64,
    cells_gen: u64,
    pub km: KmState,
    /// Closed-lasso hits for `(set generation, lasso vertices)`.
    hits: Option<Hits>,
    alts: Alts,
    /// What Waveforms draws from Features ([`FeatureViewState::link`]).
    link: FeatureLink,
    /// Key of the spikes in `link`: set, preview, lasso.
    link_key: u64,
    /// One least-squares basis job per highlighted channel.
    basis_jobs: HashMap<u32, BasisMemo>,
}

/// Selected spikes inside a closed lasso.
struct Hits {
    generation: u64,
    polygon: Vec<[f32; 2]>,
    n: usize,
    /// Local indices per cluster, at most [`LINK_SPIKES`] each.
    spikes: Vec<(ClusterId, Arc<[u32]>)>,
    /// Every hit, for the spike brush.
    all: Vec<(ClusterId, Vec<u32>)>,
}

/// Spikes per cluster the link to Waveforms carries.
const LINK_SPIKES: usize = 100;
/// Spikes per cluster of the sample the least-squares basis is fitted on.
const N_BASIS: usize = 500;

/// `(channel, set generation, cfg)`.
type BasisKey = (u32, u64, TracePreproc);
type BasisMemo = Memo<BasisKey, Option<Arc<[Vec<f32>; 2]>>>;

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
    /// `hits[part][cell]`: the part's hull in every grid cell.
    hits: Vec<Vec<Option<CellHit>>>,
}

/// One part's hull in one grid cell, and what else falls inside it.
#[derive(Clone, Debug)]
struct CellHit {
    hull: Vec<[f32; 2]>,
    /// Points of the anchor's other k-means parts.
    parts: usize,
    /// Points of the other selected clusters (not the anchor, not background).
    clusters: usize,
}

impl CellHit {
    fn total(&self) -> usize {
        self.parts + self.clusters
    }
}

/// Fit on at most this many rows, then label every row.
const KM_FIT_ROWS: usize = 5000;

/// k-means on the 4 PC columns of `slot`'s points, each scaled to `[0, 1]`
/// by the plot bounds. `k = None` tries 2..=5 in parallel and keeps the
/// best silhouette. `None` when the cluster has too few points.
fn fit_kmeans(set: &FeatureSet, slot: u16, k: Option<usize>) -> Option<KmFit> {
    use sorrel_compute::{kmeans, kmeans_assign};
    // The four PC columns of A and B (phy's `0A 1A 0B 1B`).
    let n = set.n();
    let cols = [0, 1, n, n + 1];
    let idx: Vec<usize> = (0..set.points.len())
        .filter(|&i| set.points[i].slot == slot && cols.iter().all(|&j| set.row(i)[j].is_finite()))
        .collect();
    if idx.len() < 10 {
        return None;
    }
    let data: Vec<f32> = idx
        .iter()
        .flat_map(|&i| {
            let f = set.row(i);
            cols.map(|j| {
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
    // The other selected clusters' points, for the collision counts.
    let others: Vec<usize> = (0..set.points.len())
        .filter(|&i| set.points[i].slot != slot && set.points[i].slot != BACKGROUND)
        .collect();
    let (lassos, hits) = (0..parts.len())
        .into_par_iter()
        .map(|i| auto_lasso(set, &parts, &others, i))
        .unzip();
    Some(KmFit {
        parts,
        lassos,
        hits,
    })
}

/// The lasso around `parts[target]` and its hull in every grid cell: the
/// convex hull of the 95% of the part's points nearest its median. The cell
/// whose hull holds the fewest points of the other parts and of the
/// `others` clusters wins (then the smaller hull).
// ponytail: a polygon in one 2-D view only approximates a 4-D k-means
// part; the split preview shows the exact membership next to it.
fn auto_lasso(
    set: &FeatureSet,
    parts: &[Vec<usize>],
    others: &[usize],
    target: usize,
) -> (Lasso, Vec<Option<CellHit>>) {
    let mut best: Option<(usize, f32, usize)> = None;
    let mut hits: Vec<Option<CellHit>> = vec![None; GRID.len() * GRID[0].len()];
    for (cell, &(dx, dy)) in GRID.iter().flatten().enumerate() {
        let xy = |i: usize| [set.value(i, dx), set.value(i, dy)];
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
        let probe = Lasso::new(cell, hull.clone());
        let inside =
            |it: &mut dyn Iterator<Item = &usize>| it.filter(|&&i| probe.contains(xy(i))).count();
        let hit = CellHit {
            parts: inside(
                &mut parts
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != target)
                    .flat_map(|(_, p)| p.iter()),
            ),
            clusters: inside(&mut others.iter()),
            hull,
        };
        // Area in units of each axis' spread, so cells compare.
        let area = polygon_area(&hit.hull) / (sx * sy).max(f32::MIN_POSITIVE);
        if best.is_none_or(|b| (hit.total(), area) < (b.2, b.1)) {
            best = Some((cell, area, hit.total()));
        }
        hits[cell] = Some(hit);
    }
    let lasso = best
        .and_then(|(cell, ..)| {
            hits[cell]
                .as_ref()
                .map(|h| Lasso::new(cell, h.hull.clone()))
        })
        .unwrap_or_default();
    (lasso, hits)
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

/// Rows the separation score looks at.
const MAX_ROWS: usize = 5000;
/// Label of a point the score ignores (background, or not in any part).
pub(crate) const NONE: u32 = u32::MAX;

/// What the separation scores tell apart.
#[derive(Copy, Clone)]
pub(crate) enum Labels<'a> {
    /// The selected clusters (two or more).
    Clusters,
    /// The k-means parts (indices into `set.points`).
    Parts(&'a [Vec<usize>]),
    /// Inside vs outside a closed lasso on the `x` × `y` dims.
    Lasso(Dim, Dim, &'a [[f32; 2]]),
    /// One group: spread instead of separation.
    Variance,
}

/// One label per point of `set`, [`NONE`] for those the score ignores.
pub(crate) fn labels_for(set: &FeatureSet, src: &Labels) -> Vec<u32> {
    let mut out = vec![NONE; set.points.len()];
    let selected = |p: &FeaturePoint| p.slot != BACKGROUND;
    match *src {
        Labels::Clusters => {
            for (o, p) in out.iter_mut().zip(&set.points).filter(|(_, p)| selected(p)) {
                *o = p.slot as u32;
            }
        }
        Labels::Parts(parts) => {
            for (j, part) in parts.iter().enumerate() {
                for &i in part {
                    out[i] = j as u32;
                }
            }
        }
        Labels::Lasso(dx, dy, poly) => {
            let lasso = Lasso::new(0, poly.to_vec());
            for (i, (o, p)) in out.iter_mut().zip(&set.points).enumerate() {
                if selected(p) {
                    *o = lasso.contains([set.value(i, dx), set.value(i, dy)]) as u32;
                }
            }
        }
        Labels::Variance => {
            for (o, _) in out.iter_mut().zip(&set.points).filter(|(_, p)| selected(p)) {
                *o = 0;
            }
        }
    }
    out
}

/// The labelled points a score runs on: at most [`MAX_ROWS`], evenly spread.
pub(crate) fn score_rows(labels: &[u32]) -> Vec<usize> {
    let labelled: Vec<usize> = (0..labels.len()).filter(|&i| labels[i] != NONE).collect();
    let stride = labelled.len().div_ceil(MAX_ROWS).max(1);
    labelled.into_iter().step_by(stride).collect()
}

/// Separation of `labels` along `dims` (indices into [`FeatureSet::dim`]),
/// over the point indices `rows`: the mean simplified silhouette of the
/// label groups, or the total variance when fewer than two groups have
/// finite rows. Every dim is scaled to `[0, 1]` by its range.
pub(crate) fn separation(set: &FeatureSet, dims: &[usize], rows: &[usize], labels: &[u32]) -> f32 {
    let d = dims.len();
    let dims: Vec<Dim> = dims.iter().map(|&i| set.dim(i)).collect();
    let (mut data, mut ls) = (Vec::with_capacity(rows.len() * d), Vec::new());
    'rows: for &i in rows {
        let start = data.len();
        for &dim in &dims {
            let v = set.unit(dim, set.value(i, dim));
            if !v.is_finite() {
                data.truncate(start);
                continue 'rows;
            }
            data.push(v);
        }
        ls.push(labels[i]);
    }
    let mut ids = ls.clone();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() < 2 {
        return variance(&data, d);
    }
    let dense: Vec<u32> = ls
        .iter()
        .map(|l| ids.binary_search(l).unwrap_or(0) as u32)
        .collect();
    let mut centroids = vec![0.0f32; ids.len() * d];
    let mut counts = vec![0u32; ids.len()];
    for (row, &l) in data.chunks_exact(d).zip(&dense) {
        counts[l as usize] += 1;
        for (c, v) in centroids[l as usize * d..][..d].iter_mut().zip(row) {
            *c += v;
        }
    }
    for (c, &n) in centroids.chunks_exact_mut(d).zip(&counts) {
        c.iter_mut().for_each(|v| *v /= n.max(1) as f32);
    }
    sorrel_compute::KMeans {
        labels: dense,
        centroids,
        inertia: 0.0,
    }
    .silhouette(&data, d)
}

/// Total variance of the `d`-column rows in `data`.
fn variance(data: &[f32], d: usize) -> f32 {
    let n = (data.len() / d).max(1) as f32;
    (0..d)
        .map(|a| {
            let col = data.iter().skip(a).step_by(d);
            let mean = col.clone().sum::<f32>() / n;
            col.map(|v| (v - mean).powi(2)).sum::<f32>() / n
        })
        .sum()
}

/// Which pair is `x` × `y`, whichever way round.
fn same_pair(a: (Dim, Dim), b: (Dim, Dim)) -> bool {
    a == b || a == (b.1, b.0)
}

/// Alternatives shown under the focused cell.
const N_ALTS: usize = 6;
/// Spikes per cluster in an alternative's mini scatter.
const N_MINI: usize = 400;

/// The label source, its key and the caption describing it.
fn label_source(
    state: &FeatureViewState,
    km_key: KmKey,
    fit: Option<&Arc<KmFit>>,
    selection: &[ClusterId],
) -> (LabelKey, String) {
    let mut h = DefaultHasher::new();
    let (kind, text) = if selection.len() >= 2 {
        (
            0u8,
            match selection {
                [a, b] => format!("silhouette c{} vs c{}", a.0, b.0),
                _ => format!("silhouette of {} clusters", selection.len()),
            },
        )
    } else if fit.is_some() {
        km_key.hash(&mut h);
        (1, "silhouette of the k-means parts".to_owned())
    } else if state.lasso.is_closed() {
        state.lasso.cell.hash(&mut h);
        for v in state.lasso.points.iter().flatten() {
            v.to_bits().hash(&mut h);
        }
        (2, "silhouette inside vs outside the lasso".to_owned())
    } else {
        (3, "spread of the selection".to_owned())
    };
    ((kind, h.finish()), text)
}

/// `(kind, hash)`: [`Labels`] variant and what its data hashes to.
type LabelKey = (u8, u64);

/// `(set generation, label source)`.
type AltKey = (u64, LabelKey);

/// `(dim a, dim b, separation)` of every pair of the set's dims.
type PairScores = Vec<(usize, usize, f32)>;

fn score_pairs(set: &FeatureSet, labels: &[u32], cancel: &AtomicBool) -> PairScores {
    let rows = score_rows(labels);
    let nd = set.n_dims();
    let pairs: Vec<(usize, usize)> = (0..nd)
        .flat_map(|a| (a + 1..nd).map(move |b| (a, b)))
        .collect();
    pairs
        .into_par_iter()
        .map(|(a, b)| {
            let score = if cancel.load(Ordering::Relaxed) {
                f32::NAN
            } else {
                separation(set, &[a, b], &rows, labels)
            };
            (a, b, score)
        })
        .collect()
}

/// An alternative view: a pair of the set's dims and how well it separates.
struct Alt {
    x: Dim,
    y: Dim,
    score: f32,
    verts: [Arc<Vec<ScatterVertex>>; 3],
}

#[derive(Default)]
struct Alts {
    memo: Memo<AltKey, Arc<PairScores>>,
    /// The built minis and what they were built for.
    built: Option<(BuiltKey, Arc<Vec<Alt>>)>,
    generation: u64,
    /// The alternative whose promotion awaits a second click (a lasso would
    /// be dropped).
    armed: Option<String>,
}

/// `(set generation, paint key, preview, scores, focused pair)`.
type BuiltKey = (u64, u64, u64, AltKey, (Dim, Dim));

/// A/B and the grid cell that show `x` × `y` after promoting it: both of the
/// pair's channels become A and B (x's first); a pair on one channel keeps
/// the other of A/B, or A, when it is neither.
fn promotion(channels: &[u32], x: Dim, y: Dim) -> ([u32; 2], usize) {
    let chan = |d: Dim| match d {
        Dim::Pc { ch, .. } => Some(channels[ch as usize]),
        Dim::Time => None,
    };
    let (a, b) = (channels[0], channels[1]);
    let ab = match (chan(x), chan(y)) {
        (Some(u), Some(v)) if u != v => [u, v],
        (Some(u), _) | (_, Some(u)) if u == a || u == b => [a, b],
        (Some(u), _) | (_, Some(u)) => [u, a],
        (None, None) => [a, b],
    };
    let at = |d: Dim| match d {
        Dim::Pc { ch, pc } => Dim::Pc {
            ch: ab
                .iter()
                .position(|&c| c == channels[ch as usize])
                .unwrap_or(0) as u8,
            pc,
        },
        Dim::Time => Dim::Time,
    };
    let (gx, gy) = (at(x), at(y));
    let cell = GRID
        .iter()
        .flatten()
        .position(|&g| g == (gx, gy))
        .or_else(|| GRID.iter().flatten().position(|&g| g == (gy, gx)))
        .unwrap_or(0);
    (ab, cell)
}

impl Default for FeatureViewState {
    fn default() -> Self {
        Self {
            mode: ViewMode::default(),
            view3d: Default::default(),
            focus: 1,
            fixed_channels: None,
            edited: None,
            edited_by_fix: false,
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
            shown: Arc::default(),
            ghost: GhostCells::default(),
            colour: ColourBy::default(),
            brush_written: None,
            hover_out: None,
            seek: None,
            time_out: None,
            time_from: None,
            preview: None,
            preview_gen: 0,
            cells_gen: 0,
            km: KmState::default(),
            hits: None,
            alts: Alts::default(),
            link: FeatureLink::default(),
            link_key: 0,
            basis_jobs: HashMap::new(),
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

    /// phy `toggle_automatic_channel_selection` (`c`): fix A and B, with the
    /// set on screen, or let them follow the best pair again. Returns whether
    /// they are now fixed.
    pub fn toggle_channels(&mut self) -> bool {
        if self.fixed_channels.take().is_some() {
            if std::mem::take(&mut self.edited_by_fix) {
                self.edited = None;
            }
        } else if let Some(shown) = self.shown_channels.clone().filter(|s| s.len() >= 2) {
            self.fixed_channels = Some([shown[0], shown[1]]);
            self.edited_by_fix = self.edited.is_none();
            self.edited = Some(shown);
        }
        self.fixed_channels.is_some()
    }

    /// Change the channel set (Waveforms clicks, the toolbar chips).
    pub fn edit_channels(&mut self, e: ChannelEdit) {
        let mut set = self
            .shown_channels
            .clone()
            .or_else(|| self.edited.clone())
            .unwrap_or_default();
        match e {
            ChannelEdit::Reset => {
                self.edited = None;
                self.edited_by_fix = false;
                self.fixed_channels = None;
                return;
            }
            ChannelEdit::Toggle(ch) => match set.iter().position(|&c| c == ch) {
                Some(i) if set.len() > 2 => {
                    set.remove(i);
                    if self.fixed_channels.is_some_and(|f| f.contains(&ch)) {
                        self.fixed_channels = None;
                    }
                }
                Some(_) => return,
                None if set.len() < MAX_SET => set.push(ch),
                None => return,
            },
            ChannelEdit::SetA(ch) | ChannelEdit::SetB(ch) => {
                // phy `on_select_channel`: swap when it is already the other.
                let mut ab = self
                    .fixed_channels
                    .or_else(|| Some([*set.first()?, *set.get(1)?]))
                    .unwrap_or([ch; 2]);
                let (i, j) = if matches!(e, ChannelEdit::SetB(_)) {
                    (1, 0)
                } else {
                    (0, 1)
                };
                if ab[j] == ch {
                    ab[j] = ab[i];
                }
                ab[i] = ch;
                self.fixed_channels = Some(ab);
                let rest = std::mem::replace(&mut set, ab.to_vec());
                set.dedup();
                // Truncate drops the channels past the cap, never A or B.
                set.extend(rest.into_iter().filter(|c| !ab.contains(c)));
                set.truncate(MAX_SET);
            }
        }
        self.edited = Some(set);
        self.edited_by_fix = false;
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
            .or_else(|| self.shown_channels.as_ref().map(|c| (c[0], 0)))
    }

    /// What Waveforms draws from this view; refreshed every frame the view
    /// is drawn, so this only clones Arcs.
    pub fn link(&self) -> FeatureLink {
        self.link.clone()
    }

    /// Stop the worker before a curation write takes the session lock.
    pub fn cancel(&mut self) {
        self.memo.cancel();
    }

    pub fn pending(&self) -> bool {
        self.memo.pending()
    }
}

/// Pane widths (px): below `COMPACT_W` one large cell and a navigator, from
/// `EXPANSIVE_W` the grid at pane height beside a large cell.
const COMPACT_W: f32 = 480.0;
const EXPANSIVE_W: f32 = 1100.0;
/// Room under the Expansive large cell for the K-MEANS and LASSO panels, and
/// the navigator tile side of the Compact layout.
const PANELS_H: f32 = 240.0;
const NAV_TILE: f32 = 20.0;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Layout {
    Compact,
    Regular,
    Expansive,
}

/// The layout a pane of `size` gets.
fn layout_for(size: egui::Vec2) -> Layout {
    if size.x < COMPACT_W {
        Layout::Compact
    } else if size.x < EXPANSIVE_W {
        Layout::Regular
    } else {
        Layout::Expansive
    }
}

/// Cell side of the Expansive grid: four rows fill `size`'s height, and the
/// grid takes at most half its width.
fn expansive_side(size: egui::Vec2) -> f32 {
    let by_height = (size.y - 3.0 * CELL_GAP) / 4.0 - LABEL_H;
    let by_width = (size.x / 2.0 - 3.0 * CELL_GAP) / 4.0;
    by_height.min(by_width).max(40.0)
}

/// The previewed k-means part with its hull in every grid cell.
struct Coll {
    fit: Arc<KmFit>,
    part: usize,
}

impl Coll {
    fn hit(&self, cell: usize) -> Option<&CellHit> {
        self.fit.hits[self.part][cell].as_ref()
    }

    fn size(&self) -> usize {
        self.fit.parts[self.part].len()
    }

    /// The cell k-means lassoed.
    fn best(&self) -> usize {
        self.fit.lassos[self.part].cell
    }

    /// The cell with the most collisions.
    fn worst(&self) -> usize {
        (0..self.fit.hits[self.part].len())
            .max_by_key(|&c| self.hit(c).map_or(0, CellHit::total))
            .unwrap_or(0)
    }

    /// Names of what else is counted: the anchor's other part(s) and the
    /// other selected clusters (`None` when only one is selected).
    fn others(&self, selection: &[ClusterId]) -> (String, Option<String>) {
        let parts = match self.fit.parts.len() {
            2 => format!("part {}", 2 - self.part),
            _ => "other parts".into(),
        };
        let clusters = match selection.len() {
            0 | 1 => None,
            2 => Some(format!("c{}", selection[1])),
            _ => Some("other clusters".into()),
        };
        (parts, clusters)
    }
}

/// Green at no collisions, amber up to 15% of the part's points, red above.
fn collision_colour(theme: &Theme, total: usize, part: usize) -> Color32 {
    if total == 0 {
        theme.good
    } else if total as f32 <= 0.15 * part as f32 {
        theme.warn
    } else {
        theme.danger
    }
}

/// Matrix tile fill and digit colour: white on red and on dark tiles, the
/// theme's strong text on the light theme's tinted green and amber.
fn tile_colours(theme: &Theme, ui: &Ui, total: usize, part: usize) -> (Color32, Color32) {
    let base = collision_colour(theme, total, part);
    if base == theme.danger {
        (base, Color32::WHITE)
    } else if theme.dark {
        (base.gamma_multiply(0.55), Color32::WHITE)
    } else {
        (
            Color32::WHITE.lerp_to_gamma(base, 0.45),
            ui.visuals().strong_text_color(),
        )
    }
}

/// Hints sit between the caption grey and body text.
fn hint_colour(theme: &Theme) -> Color32 {
    theme.caption.lerp_to_gamma(theme.text, 0.5)
}

fn hint(ui: &mut Ui, text: impl Into<String>) {
    let colour = hint_colour(&Theme::from_ui(ui));
    ui.label(egui::RichText::new(text.into()).color(colour));
}

/// Lasso gestures `(keys, effect)`, the list in lasso.rs's module doc.
const LASSO_HELP: [(&str, &str); 8] = [
    ("Ctrl+drag", "draw freehand (simplified to nodes)"),
    ("Ctrl+click", "add a node"),
    ("Ctrl+right-click", "clear"),
    ("drag node", "move"),
    ("right-click node", "delete (never below three)"),
    ("double-click edge", "insert a node"),
    ("K", "split the lassoed spikes"),
    ("Esc", "clear"),
];

fn help_text() -> String {
    let mut s: String = LASSO_HELP
        .iter()
        .map(|(k, v)| format!("{k}: {v}\n"))
        .collect();
    s.push_str(
        "wheel: zoom · drag: pan · double-click: reset\n\
         Ctrl+wheel: feature scale · Alt+wheel: marker size\n\
         Alt+click / Alt+right-click: amplitude feature (x / y)\n\
         Alt+click a spike: seek the Trace to it\n\
         Shift+drag a time cell: time range · Shift+click clears\n\
         Shift+click a Waveforms channel: add or remove it from the set\n\
         Ctrl+click / Ctrl+right-click a Waveforms channel: set A / B\n\
         Click an alternative: promote it to A/B (twice with a lasso)",
    );
    s
}

/// One short line whatever it says, so nothing moves under the cursor while
/// a lasso is being drawn; the full gesture list is on `(?)`.
fn hint_line(state: &FeatureViewState) -> &'static str {
    match (state.lasso.is_active(), state.lasso.is_closed()) {
        (_, true) => "K: split · Esc: clear",
        (true, false) => "Ctrl+click: add point · Ctrl+right-click: clear",
        _ => "Ctrl+click: lasso · K: split",
    }
}

/// What every layout draws with.
struct Ctx<'a, P: DataProvider> {
    session: &'a Session<P>,
    jobs: &'a JobCtx<Session<P>>,
    selection: &'a [ClusterId],
    key: &'a FeatureKey,
    set: &'a Arc<FeatureSet>,
    km_key: KmKey,
    fit: Option<Arc<KmFit>>,
    /// Some while a k-means part is previewed.
    coll: Option<Coll>,
    theme: Theme,
    dark: bool,
    cells: CellVertices,
    cells_gen: u64,
    /// The focused cell is drawn twice (grid and large cell).
    dup: bool,
    alts: AltsView,
    /// Time brush, the hovered spike's point and the points inside the Trace
    /// window (few enough per cluster to ring), as `FeatureSet::points` ids.
    time: Option<TimeBrush>,
    hover_pt: Option<usize>,
    windowed: Vec<usize>,
    /// The non-background points that are drawn.
    shown: Arc<Vec<usize>>,
    /// The merge preview: cluster, buffer generation, grey vertices per cell.
    ghost: Option<(ClusterId, u64, GhostVerts)>,
}

/// What the alternatives strip draws this frame.
struct AltsView {
    /// The scored views; `None` while the scores are on their way.
    alts: Option<Arc<Vec<Alt>>>,
    generation: u64,
    /// "silhouette c0 vs c2"
    text: String,
}

/// Draw the view. Returns `(cluster, local spike indices)` splits when `k`
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
    brushes: &mut Brushes,
    depths: Option<&SpikeDepths>,
    window: (u64, u32),
) -> Option<Vec<(ClusterId, Vec<u32>)>> {
    let theme = Theme::from_ui(ui);
    if selection.is_empty() {
        ui.label("no selection");
        state.link = FeatureLink::default();
        return None;
    }
    let computed = !session.has_pc_features();
    // Loading every lassoed spike keeps the channels (and order) on screen.
    let pinned = state
        .want_full
        .then(|| state.shown_channels.clone())
        .flatten();
    let key = FeatureKey {
        selection: selection.to_vec(),
        keep_order: pinned.is_some() || state.fixed_channels.is_some(),
        channels: pinned.or_else(|| state.edited.clone()),
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
    state.shown_channels = Some(set.channels.clone());
    let cur = lasso_key(&bound_key, &set.channels);
    // A lasso from before a selection/channel change must not hit the new
    // set's points.
    state.drop_stale_lasso(&cur);

    let km_key: KmKey = (set.generation, selection[0], state.km.k);
    kmeans_update(state, &set, km_key, selection[0], jobs);
    let layout = layout_for(ui.available_size());
    if state.km.scroll_to_lasso && (layout != Layout::Regular || state.mode == ViewMode::Three) {
        // One large cell (or no grid): show the lasso's cell, don't scroll.
        if state.lasso.is_active() {
            state.focus = state.lasso.cell;
        }
        state.km.scroll_to_lasso = false;
    }
    toolbar(ui, state, &set, current.is_none(), layout, depths);

    let fit = km_fit(state, km_key);
    let coll = fit.clone().filter(|_| state.km.previewing).map(|fit| Coll {
        part: state.km.part.min(fit.parts.len() - 1),
        fit,
    });
    let dark = ui.visuals().dark_mode;
    let brush = brushes.spikes.clone();
    let time = brushes.time;
    let paint = Paint {
        dark,
        colour: state.colour,
        depths,
        selection,
        brush: brush.as_ref(),
        time,
    };
    let alts = update_alts(state, &set, &fit, km_key, selection, &paint, jobs);
    let cells_key = (set.generation, paint.key(), state.preview_gen);
    if !matches!(&state.cells, Some((k, ..)) if *k == cells_key) {
        let preview = state.preview.as_ref().and_then(|(c, idx)| {
            let slot = selection.iter().position(|s| s == c)?;
            Some((slot as u16, idx))
        });
        state.cells_gen += 1;
        let (cells, shown) = cell_vertices(&set, &paint, preview);
        state.cells = Some((cells_key, state.cells_gen, cells));
        state.shown = Arc::new(shown);
    }
    let ghost = ghost_cells(
        state,
        &set,
        &bound_key,
        selection,
        dark,
        brushes.ghost,
        jobs,
    );
    let shown = Arc::clone(&state.shown);
    let hover_pt = brushes.hover.and_then(|h| {
        let slot = selection.iter().position(|&c| c == h.cluster)? as u16;
        shown
            .iter()
            .copied()
            .find(|&i| set.points[i].slot == slot && set.points[i].local == h.local)
    });
    let windowed = windowed_points(session, selection, &set, &shown, window);
    let c = Ctx {
        session,
        jobs,
        selection,
        key: &bound_key,
        set: &set,
        km_key,
        fit,
        coll,
        theme,
        dark,
        cells: state
            .cells
            .as_ref()
            .map(|(_, _, c)| c.clone())
            .unwrap_or_default(),
        cells_gen: state.cells_gen,
        dup: layout == Layout::Expansive,
        alts,
        time,
        hover_pt,
        windowed,
        shown,
        ghost,
    };
    match (state.mode, layout) {
        (_, Layout::Expansive) => expansive(ui, &c, state, brushes, depths),
        (ViewMode::Three, _) => {
            kmeans_row(ui, state, &c, layout == Layout::Regular, false);
            show_3d(ui, &c, state, layout == Layout::Compact, brushes, depths);
        }
        (ViewMode::Two, Layout::Compact) => compact(ui, &c, state),
        (ViewMode::Two, Layout::Regular) => regular(ui, &c, state),
    }

    // With the grid beside it (Expansive) the focused 2-D cell wins.
    let from_triple = state.mode == ViewMode::Three && layout != Layout::Expansive;
    sync_brush(state, &set, selection, session.version(), brushes);
    if let Some(h) = state.hover_out.take() {
        brushes.hover_next = Some(h);
    }
    if let Some(t) = state.time_out.take() {
        brushes.time = t;
    }
    refresh_link(
        state,
        &set,
        selection,
        jobs,
        cfg,
        shanks,
        from_triple,
        brushes,
    );
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

/// The merge ghost's grey points, once its features are in (`brushes.ghost`
/// names a cluster that is not selected).
fn ghost_cells<P: DataProvider>(
    state: &mut FeatureViewState,
    set: &FeatureSet,
    key: &FeatureKey,
    selection: &[ClusterId],
    dark: bool,
    ghost: Option<Ghost>,
    jobs: &JobCtx<Session<P>>,
) -> Option<(ClusterId, u64, GhostVerts)> {
    let Some(Ghost::Merge(g)) =
        ghost.filter(|g| !matches!(g, Ghost::Merge(c) if selection.contains(c)))
    else {
        return None;
    };
    let gkey = FeatureKey {
        selection: vec![g],
        channels: Some(set.channels.clone()),
        keep_order: true,
        ..key.clone()
    };
    let job_key = gkey.clone();
    let loaded = state
        .ghost
        .memo
        .ensure(gkey, &jobs.session, jobs.repaint(), move |s, cancel| {
            let basis = Arc::new(Mutex::new(None));
            Arc::new(compute(s, &job_key, &basis, cancel))
        })
        .cloned()?;
    let built_key = (loaded.generation, set.generation, dark);
    if !matches!(&state.ghost.built, Some((k, ..)) if *k == built_key) {
        // Weak grey at half alpha.
        let grey = if dark {
            pack_rgba(190, 190, 190, 128)
        } else {
            pack_rgba(110, 110, 110, 128)
        };
        let order = display_order(&loaded, N_DISPLAY, false, &|_| false);
        let cells = GRID
            .iter()
            .flatten()
            .map(|&(dx, dy)| {
                Arc::new(
                    order
                        .iter()
                        .filter_map(|&i| {
                            let (x, y) = (loaded.value(i, dx), loaded.value(i, dy));
                            (x.is_finite() && y.is_finite()).then_some(ScatterVertex {
                                pos: [x, y],
                                color: grey,
                            })
                        })
                        .collect(),
                )
            })
            .collect();
        state.ghost.generation += 1;
        state.ghost.built = Some((built_key, state.ghost.generation, Arc::new(cells)));
    }
    state
        .ghost
        .built
        .as_ref()
        .map(|(_, gen, cells)| (g, *gen, Arc::clone(cells)))
}

/// Shown points inside the Trace window, for clusters with few enough spikes
/// there to ring.
fn windowed_points<P: DataProvider>(
    session: &Session<P>,
    selection: &[ClusterId],
    set: &FeatureSet,
    shown: &[usize],
    (start, len): (u64, u32),
) -> Vec<usize> {
    let ranges: Vec<std::ops::Range<u32>> = selection
        .iter()
        .map(|&c| window_locals(session.spike_times(c), start, len))
        .map(|r| {
            if r.len() as u32 <= WINDOW_RING_MAX {
                r
            } else {
                0..0
            }
        })
        .collect();
    shown
        .iter()
        .copied()
        .filter(|&i| {
            let p = &set.points[i];
            ranges[p.slot as usize].contains(&p.local)
        })
        .collect()
}

/// What the lasso (or the previewed k-means part) wants `brushes.spikes` to
/// be, and a key for it.
fn want_brush(state: &FeatureViewState, set: &FeatureSet) -> BrushWant {
    let mut h = DefaultHasher::new();
    set.generation.hash(&mut h);
    let source = if state.km.previewing && state.preview.is_some() {
        state.preview_gen.hash(&mut h);
        BrushSource::KMeansPart
    } else if state.lasso.is_closed() {
        state.lasso.cell.hash(&mut h);
        state
            .lasso
            .points
            .iter()
            .flatten()
            .for_each(|v| v.to_bits().hash(&mut h));
        BrushSource::FeatureLasso
    } else {
        return None;
    };
    Some((source, h.finish()))
}

/// Publish the lasso's hits (or the k-means part's members) as the spike
/// brush when they change, and take them back when the lasso goes. Another
/// view's brush is left alone until then: the last writer wins.
fn sync_brush(
    state: &mut FeatureViewState,
    set: &FeatureSet,
    selection: &[ClusterId],
    version: u64,
    brushes: &mut Brushes,
) {
    let want = want_brush(state, set);
    match brush_action(&mut state.brush_written, want) {
        BrushAction::Keep => {}
        BrushAction::Clear => {
            if brushes.spikes.as_ref().is_some_and(|b| {
                matches!(
                    b.source,
                    BrushSource::FeatureLasso | BrushSource::KMeansPart
                )
            }) {
                brushes.spikes = None;
            }
        }
        BrushAction::Set => {
            let Some((source, _)) = want else { return };
            let spikes = match (source, &state.preview) {
                (BrushSource::KMeansPart, Some((c, members))) => {
                    vec![(*c, members.iter().copied().collect())]
                }
                _ => lasso_hits_cached(state, set, selection).all.clone(),
            };
            brushes.spikes = SpikeBrush::new(source, version, spikes);
        }
    }
}

/// The 3-D scene with the k-means part and the lasso drawn into it.
fn show_3d<P: DataProvider>(
    ui: &mut Ui,
    c: &Ctx<P>,
    state: &mut FeatureViewState,
    compact: bool,
    brushes: &mut Brushes,
    depths: Option<&SpikeDepths>,
) {
    let lasso = if state.lasso.is_closed() {
        let (dx, dy) = GRID[state.lasso.cell / 4][state.lasso.cell % 4];
        Some((dx, dy, &state.lasso.points[..]))
    } else {
        None
    };
    let overlay = Overlay {
        focus: GRID[state.focus / 4][state.focus % 4],
        km_part: c.coll.as_ref().map(|k| &k.fit.parts[k.part][..]),
        lasso,
        km_parts: c.fit.as_ref().map(|f| &f.parts[..]),
    };
    feature_3d(
        ui,
        c.session,
        c.selection,
        c.set,
        c.key,
        &mut state.view3d,
        c.jobs,
        overlay,
        compact,
        brushes,
        depths,
        match state.colour {
            ColourBy::Cluster => None,
            ColourBy::Time => Some(PointColour::Time),
            ColourBy::Depth => Some(PointColour::Depth),
        },
    );
}

/// Regular: the grid in 2-4 columns, scrolling.
fn regular<P: DataProvider>(ui: &mut Ui, c: &Ctx<P>, state: &mut FeatureViewState) {
    kmeans_row(ui, state, c, true, true);
    egui::ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| {
            let (cols, side) = grid_geometry(ui.available_width());
            let rows = GRID.len() * GRID[0].len() / cols;
            let width = cols as f32 * side + (cols - 1) as f32 * CELL_GAP;
            // The alternatives strip sits under the focused cell's row.
            let strip_h = strip_height(width, state.alts.armed.is_some()) + CELL_GAP;
            let focus_row = state.focus / cols;
            let (grid_rect, _) = ui.allocate_exact_size(
                egui::vec2(
                    width,
                    rows as f32 * (side + LABEL_H) + (rows - 1) as f32 * CELL_GAP + strip_h,
                ),
                Sense::hover(),
            );
            let row_h = side + LABEL_H + CELL_GAP;
            let strip = Rect::from_min_size(
                grid_rect.min + egui::vec2(0.0, (focus_row + 1) as f32 * row_h),
                egui::vec2(width, strip_h - CELL_GAP),
            );
            ui.scope_builder(egui::UiBuilder::new().max_rect(strip), |ui| {
                alternatives(ui, c, state, width);
            });
            for cell in 0..GRID.len() * GRID[0].len() {
                let (row, col) = (cell / cols, cell % cols);
                let shift = if row > focus_row { strip_h } else { 0.0 };
                let origin = grid_rect.min
                    + egui::vec2(col as f32 * (side + CELL_GAP), row as f32 * row_h + shift);
                if state.km.scroll_to_lasso && state.lasso.cell == cell {
                    state.km.scroll_to_lasso = false;
                    ui.scroll_to_rect(
                        Rect::from_min_size(origin, egui::vec2(side, side + LABEL_H)),
                        Some(egui::Align::Center),
                    );
                }
                draw_cell(ui, c, state, cell, origin, side, false);
            }
        });
}

/// Compact: the k-means row, one large focused cell and the 4×4 collision
/// matrix as navigator.
fn compact<P: DataProvider>(ui: &mut Ui, c: &Ctx<P>, state: &mut FeatureViewState) {
    kmeans_row(ui, state, c, false, false);
    let height = ui.available_height();
    egui::ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| {
            let nav_h = 4.0 * NAV_TILE + 3.0 * 2.0;
            let w = ui.available_width();
            let strip_h = strip_height(w, state.alts.armed.is_some());
            let side = w.min((height - LABEL_H - nav_h - strip_h - 12.0).max(140.0));
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(side, side + LABEL_H), Sense::hover());
            let focus = state.focus;
            draw_cell(ui, c, state, focus, rect.min, side, true);
            ui.add_space(4.0);
            // The navigator first: it is how a narrow pane reaches the other
            // 15 views, so it must not drop below the fold.
            ui.horizontal_top(|ui| {
                let lasso_cell = state.lasso.is_active().then_some(state.lasso.cell);
                let clicked = collision_matrix(
                    ui,
                    &c.theme,
                    c.set,
                    c.coll.as_ref(),
                    state.focus,
                    lasso_cell,
                    NAV_TILE,
                );
                if let Some(cell) = clicked {
                    goto_cell(state, c.coll.as_ref(), cell);
                }
                ui.vertical(|ui| {
                    ui.strong(if c.coll.is_some() {
                        "Navigator + collision"
                    } else {
                        "Navigator"
                    });
                    hint(ui, nav_text(c));
                    ui.horizontal(|ui| {
                        hint(ui, "Ctrl+drag lasso · drag node · K split ·");
                        ui.label("(?)").on_hover_text(help_text());
                    });
                    if c.coll.is_some() {
                        colour_legend(ui, &c.theme);
                    }
                });
            });
            ui.add_space(4.0);
            alternatives(ui, c, state, w);
        });
}

/// "click a tile to focus it", and what the numbers count when a part is
/// previewed.
fn nav_text<P: DataProvider>(c: &Ctx<P>) -> String {
    match &c.coll {
        Some(k) => {
            let (parts, clusters) = k.others(c.selection);
            format!(
                "click a tile to focus it · N = spikes of {parts}{} inside part {}'s hull (of {})",
                clusters.map_or(String::new(), |n| format!(" + {n}")),
                k.part + 1,
                k.size()
            )
        }
        None => "click a tile to focus it".into(),
    }
}

/// Swatches for the tile colours.
fn colour_legend(ui: &mut Ui, theme: &Theme) {
    ui.horizontal(|ui| {
        for (colour, text) in [
            (theme.good, "0"),
            (theme.warn, "≤15% of part"),
            (theme.danger, ">15%"),
        ] {
            let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), Sense::hover());
            ui.painter().rect_filled(r, 2.0, colour);
            hint(ui, text);
        }
    });
}

/// Expansive: the 4×4 grid at pane height on the left (no scroll), the
/// focused cell and its K-MEANS and LASSO panels (or the 3-D scene) on the
/// right.
fn expansive<P: DataProvider>(
    ui: &mut Ui,
    c: &Ctx<P>,
    state: &mut FeatureViewState,
    brushes: &mut Brushes,
    depths: Option<&SpikeDepths>,
) {
    let size = ui.available_size();
    let (area, _) = ui.allocate_exact_size(size, Sense::hover());
    let side = expansive_side(area.size());
    for cell in 0..GRID.len() * GRID[0].len() {
        let origin = area.min
            + egui::vec2(
                (cell % 4) as f32 * (side + CELL_GAP),
                (cell / 4) as f32 * (side + LABEL_H + CELL_GAP),
            );
        draw_cell(ui, c, state, cell, origin, side, false);
    }
    let left = 4.0 * side + 3.0 * CELL_GAP + 2.0 * CELL_GAP;
    let right = Rect::from_min_max(area.min + egui::vec2(left, 0.0), area.max);
    ui.scope_builder(egui::UiBuilder::new().max_rect(right), |ui| {
        if state.mode == ViewMode::Three {
            kmeans_row(ui, state, c, true, false);
            show_3d(ui, c, state, false, brushes, depths);
            return;
        }
        let strip_h = strip_height(right.width(), state.alts.armed.is_some());
        let big = right
            .width()
            .min(right.height() - LABEL_H - PANELS_H - strip_h - CELL_GAP)
            .max(120.0);
        let focus = state.focus;
        draw_cell(ui, c, state, focus, right.min, big, true);
        let strip = Rect::from_min_size(
            right.min + egui::vec2(0.0, LABEL_H + big + CELL_GAP),
            egui::vec2(right.width(), strip_h),
        );
        ui.scope_builder(egui::UiBuilder::new().max_rect(strip), |ui| {
            alternatives(ui, c, state, right.width());
        });
        let panels = Rect::from_min_max(
            right.min + egui::vec2(0.0, LABEL_H + big + strip_h + 2.0 * CELL_GAP),
            right.max,
        );
        ui.scope_builder(egui::UiBuilder::new().max_rect(panels), |ui| {
            ui.separator();
            ui.columns(2, |cols| {
                kmeans_panel(&mut cols[0], c, state);
                lasso_panel(&mut cols[1], c, state);
            });
        });
    });
}

/// Scores the set's dim pairs for the current labels and builds the minis of
/// the best six that are not the focused view.
fn update_alts<P: DataProvider>(
    state: &mut FeatureViewState,
    set: &Arc<FeatureSet>,
    fit: &Option<Arc<KmFit>>,
    km_key: KmKey,
    selection: &[ClusterId],
    paint: &Paint,
    jobs: &JobCtx<Session<P>>,
) -> AltsView {
    let (label_key, text) = label_source(state, km_key, fit.as_ref(), selection);
    let key: AltKey = (set.generation, label_key);
    let (job_set, job_fit) = (Arc::clone(set), fit.clone());
    let (cell, polygon) = (state.lasso.cell, state.lasso.points.clone());
    let scores = state
        .alts
        .memo
        .ensure(key, &jobs.session, jobs.repaint(), move |_, cancel| {
            let (dx, dy) = GRID[cell / 4][cell % 4];
            let src = match (label_key.0, &job_fit) {
                (0, _) => Labels::Clusters,
                (1, Some(fit)) => Labels::Parts(&fit.parts),
                (2, _) => Labels::Lasso(dx, dy, &polygon),
                _ => Labels::Variance,
            };
            let labels = labels_for(&job_set, &src);
            Arc::new(score_pairs(&job_set, &labels, cancel))
        })
        .cloned();
    let focus = GRID[state.focus / 4][state.focus % 4];
    let built_key: BuiltKey = (set.generation, paint.key(), state.preview_gen, key, focus);
    if let (Some(scores), false) = (
        scores,
        matches!(&state.alts.built, Some((k, _)) if *k == built_key),
    ) {
        let preview = state.preview.as_ref().and_then(|(c, idx)| {
            let slot = selection.iter().position(|s| s == c)?;
            Some((slot as u16, idx))
        });
        let previewed = |i: usize| {
            let p = &set.points[i];
            preview.is_some_and(|(slot, idx)| p.slot == slot && idx.contains(&p.local))
        };
        let order = display_order(set, N_MINI, true, &previewed);
        let mut pairs: Vec<(Dim, Dim, f32)> = scores
            .iter()
            .filter(|s| s.2.is_finite())
            .map(|&(a, b, score)| (set.dim(a), set.dim(b), score))
            .filter(|&(x, y, _)| !same_pair((x, y), focus))
            .collect();
        pairs.sort_by(|a, b| b.2.total_cmp(&a.2));
        pairs.truncate(N_ALTS);
        let built: Vec<Alt> = pairs
            .into_iter()
            .map(|(x, y, score)| Alt {
                x,
                y,
                score,
                verts: layer_vertices(set, &order, (x, y), paint, &previewed),
            })
            .collect();
        state.alts.generation += 1;
        state.alts.built = Some((built_key, Arc::new(built)));
    }
    let current = matches!(&state.alts.built, Some((k, _)) if *k == built_key);
    AltsView {
        alts: state
            .alts
            .built
            .as_ref()
            .filter(|_| current)
            .map(|(_, a)| Arc::clone(a)),
        generation: state.alts.generation,
        text,
    }
}

/// Make `x` × `y` the focused view: its channels become A and B (fixed), the
/// grid cell showing it is focused, and the lasso, which belonged to the old
/// channels, is dropped.
fn promote(state: &mut FeatureViewState, set: &FeatureSet, x: Dim, y: Dim) {
    let (ab, cell) = promotion(&set.channels, x, y);
    let mut next = ab.to_vec();
    next.dedup();
    next.extend(set.channels.iter().filter(|c| !ab.contains(c)));
    state.fixed_channels = Some(ab);
    state.edited = Some(next);
    state.edited_by_fix = false;
    state.focus = cell;
    state.cancel_lasso();
    state.alts.armed = None;
}

const ALT_CAPTION_H: f32 = 16.0;
const ALT_LABEL_H: f32 = 26.0;
const TOAST_H: f32 = 24.0;

/// Minis per row and their side for a strip `width` wide.
fn alt_geometry(width: f32) -> (usize, f32) {
    let cols = if width >= 240.0 { N_ALTS } else { N_ALTS / 2 };
    (
        cols,
        ((width - (cols - 1) as f32 * CELL_GAP) / cols as f32).clamp(36.0, 150.0),
    )
}

/// Height of the strip, with the warning row when `armed`.
fn strip_height(width: f32, armed: bool) -> f32 {
    let (cols, side) = alt_geometry(width);
    let rows = N_ALTS.div_ceil(cols) as f32;
    ALT_CAPTION_H
        + rows * (ALT_LABEL_H + side)
        + (rows - 1.0) * CELL_GAP
        + if armed { TOAST_H + 4.0 } else { 0.0 }
}

/// The alternatives strip: the six best-separating views of the set that are
/// not the focused one, as mini scatters. A click promotes one; with a lasso
/// the first click asks for a second.
/// Below this strip width the alternatives caption drops its explanation.
const WIDE_ALT_CAPTION: f32 = 480.0;

fn alternatives<P: DataProvider>(
    ui: &mut Ui,
    c: &Ctx<P>,
    state: &mut FeatureViewState,
    width: f32,
) {
    let (theme, set) = (&c.theme, &**c.set);
    if !state.lasso.is_closed() {
        state.alts.armed = None;
    }
    let (cols, side) = alt_geometry(width);
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(width, strip_height(width, state.alts.armed.is_some())),
        Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    let caption = |text: String| {
        painter.text(
            rect.left_top() + egui::vec2(2.0, 0.0),
            egui::Align2::LEFT_TOP,
            text,
            egui::FontId::proportional(10.0),
            theme.caption,
        );
    };
    let Some(alts) = &c.alts.alts else {
        caption("ALTERNATIVES · scoring the set's views…".into());
        return;
    };
    caption(if width < WIDE_ALT_CAPTION {
        format!("ALTERNATIVES ({}) · click promotes", c.alts.text)
    } else {
        format!(
            "ALTERNATIVES · best-separating pairs over the set ({}) · click promotes to A/B",
            c.alts.text
        )
    });
    let name = |alt: &Alt| format!("{} × {}", dim_label(alt.x, set), dim_label(alt.y, set));
    if state
        .alts
        .armed
        .as_ref()
        .is_some_and(|a| !alts.iter().any(|alt| name(alt) == *a))
    {
        state.alts.armed = None;
    }
    let mut clicked = None;
    for (slot, alt) in alts.iter().enumerate() {
        let origin = rect.min
            + egui::vec2(
                (slot % cols) as f32 * (side + CELL_GAP),
                ALT_CAPTION_H + (slot / cols) as f32 * (ALT_LABEL_H + side + CELL_GAP),
            );
        let label = Rect::from_min_size(origin, egui::vec2(side, ALT_LABEL_H));
        let plot = Rect::from_min_size(origin + egui::vec2(0.0, ALT_LABEL_H), Vec2::splat(side));
        let font = egui::FontId::proportional(10.0);
        let lp = painter.with_clip_rect(label);
        lp.text(
            label.left_top() + egui::vec2(2.0, 1.0),
            egui::Align2::LEFT_TOP,
            format!("{} ×", dim_label(alt.x, set)),
            font.clone(),
            theme.caption,
        );
        lp.text(
            label.left_top() + egui::vec2(2.0, 13.0),
            egui::Align2::LEFT_TOP,
            format!("{} · {:.2}", dim_label(alt.y, set), alt.score),
            font,
            theme.caption,
        );
        painter.rect_filled(plot, 0.0, theme.plot_bg);
        let (x, y) = (set.bound(alt.x), set.bound(alt.y));
        for (layer, vertices, radius) in [
            ("alt-bg", &alt.verts[0], 0.5),
            ("alt-sel", &alt.verts[1], (state.marker * 0.6).max(1.2)),
            ("alt-brush", &alt.verts[2], (state.marker * 0.6).max(1.2)),
        ] {
            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    plot,
                    ScatterCallback {
                        id: ui.id().with((layer, slot)).value(),
                        vertices: Arc::clone(vertices),
                        generation: c.alts.generation,
                        x,
                        y,
                        radius,
                        size: plot.size(),
                    },
                ));
        }
        painter.rect_stroke(
            plot,
            0.0,
            Stroke::new(0.5_f32, theme.grid),
            egui::StrokeKind::Inside,
        );
        let bar = Rect::from_min_size(
            plot.right_bottom() - egui::vec2(4.0 + 0.4 * side * alt.score.clamp(0.0, 1.0), 7.0),
            egui::vec2(0.4 * side * alt.score.clamp(0.0, 1.0), 3.0),
        );
        painter.rect_filled(bar, 1.5, theme.good);
        let resp = ui.interact(plot, ui.id().with(("alt", slot)), Sense::click());
        let access = format!("alternative {}", name(alt));
        resp.widget_info(|| {
            egui::WidgetInfo::labeled(egui::WidgetType::Button, true, access.clone())
        });
        if state.alts.armed.as_deref() == Some(&name(alt)) {
            painter.rect_stroke(
                plot,
                0.0,
                Stroke::new(1.5_f32, ui.visuals().selection.bg_fill),
                egui::StrokeKind::Inside,
            );
        } else if resp.hovered() {
            painter.rect_stroke(
                plot,
                0.0,
                Stroke::new(1.0_f32, ui.visuals().selection.bg_fill),
                egui::StrokeKind::Inside,
            );
        }
        if resp
            .on_hover_text(format!("Promote {} to A/B", name(alt)))
            .clicked()
        {
            clicked = Some(slot);
        }
    }
    if let Some(armed) = &state.alts.armed {
        let row = Rect::from_min_size(
            egui::pos2(rect.left(), rect.bottom() - TOAST_H),
            egui::vec2(width, TOAST_H),
        );
        let (fill, text) = if theme.dark {
            (
                theme.warn.gamma_multiply(0.25),
                ui.visuals().strong_text_color(),
            )
        } else {
            (
                Color32::WHITE.lerp_to_gamma(theme.warn, 0.25),
                ui.visuals().strong_text_color(),
            )
        };
        painter.rect(
            row,
            4.0,
            fill,
            Stroke::new(1.0_f32, theme.warn),
            egui::StrokeKind::Inside,
        );
        painter.text(
            row.left_center() + egui::vec2(8.0, 0.0),
            egui::Align2::LEFT_CENTER,
            format!("promote {armed} drops the lasso \u{2014} click again"),
            egui::FontId::proportional(11.0),
            text,
        );
    }
    if let Some(alt) = clicked.map(|slot| &alts[slot]) {
        if state.lasso.is_closed() && state.alts.armed.as_deref() != Some(&name(alt)) {
            state.alts.armed = Some(name(alt));
        } else {
            promote(state, c.set, alt.x, alt.y);
        }
        ui.ctx().request_repaint();
    }
}

/// Toolbar chips of the channel set: A and B selected; a click on another
/// removes it.
fn set_chips(ui: &mut Ui, state: &mut FeatureViewState, set: &FeatureSet) {
    ui.label(egui::RichText::new("set").color(Theme::from_ui(ui).caption));
    let mut remove = None;
    for (i, &ch) in set.channels.iter().enumerate() {
        let text = match i {
            0 => format!("{ch} A"),
            1 => format!("{ch} B"),
            _ => ch.to_string(),
        };
        let resp = ui.add(egui::Button::new(text).small().selected(i < 2));
        if i < 2 {
            resp.on_hover_text(if i == 0 {
                "A: x-axis channel of the grid"
            } else {
                "B: y-axis channel of the grid"
            });
        } else if resp
            .on_hover_text("Click to remove \u{b7} Shift+click channels in Waveforms to add")
            .clicked()
        {
            remove = Some(ch);
        }
    }
    if let Some(ch) = remove {
        state.edit_channels(ChannelEdit::Toggle(ch));
    }
}

/// Publishes what Waveforms draws from this view: the set, the focused
/// cell's channels, the lasso's (or previewed part's) spikes and the PC
/// shapes on those channels.
#[allow(clippy::too_many_arguments)]
fn refresh_link<P: DataProvider>(
    state: &mut FeatureViewState,
    set: &Arc<FeatureSet>,
    selection: &[ClusterId],
    jobs: &JobCtx<Session<P>>,
    cfg: TracePreproc,
    shanks: &[u32],
    from_triple: bool,
    brushes: &Brushes,
) {
    let (dx, dy) = GRID[state.focus / 4][state.focus % 4];
    // The 3-D scene's triple, else the focused cell's channels.
    let mut highlight: Vec<u32> = if from_triple {
        state.view3d.triple_channels(set)
    } else {
        Vec::new()
    };
    let from_cell = highlight.is_empty();
    for d in [dx, dy].into_iter().filter(|_| from_cell) {
        if let Dim::Pc { ch, .. } = d {
            let c = set.channels[ch as usize];
            if !highlight.contains(&c) {
                highlight.push(c);
            }
        }
    }
    // The part of a proposal or k-means shows its exact members, a lasso the
    // spikes inside it.
    let own_lasso = state.lasso.is_closed() && !state.km.lasso_auto;
    let mut h = DefaultHasher::new();
    (set.generation, state.preview_gen, own_lasso).hash(&mut h);
    hash_brush(brushes.spikes.as_ref(), &mut h);
    if state.lasso.is_closed() {
        state.lasso.cell.hash(&mut h);
        state
            .lasso
            .points
            .iter()
            .flatten()
            .for_each(|v| v.to_bits().hash(&mut h));
    }
    let link_key = h.finish();
    if link_key != state.link_key || state.link.set.is_empty() {
        state.link_key = link_key;
        // Whatever brush exists shows in Waveforms, from any view.
        state.link.spikes = match (&brushes.spikes, &state.preview, own_lasso) {
            (Some(b), ..) => b.spikes.iter().map(|(c, v)| (*c, thin(v))).collect(),
            (None, Some((cluster, members)), false) => {
                let mut v: Vec<u32> = members.iter().copied().collect();
                v.sort_unstable();
                vec![(*cluster, thin(&v))]
            }
            _ if state.lasso.is_closed() => lasso_hits_cached(state, set, selection).spikes.clone(),
            _ => Vec::new(),
        };
    }
    let mut basis = Vec::new();
    if set.computed {
        let fitted = state.basis.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, _, b)) = fitted.as_ref().filter(|(c, ..)| *c == cfg) {
            basis.extend(highlight.iter().map(|&ch| (ch, Arc::clone(b))));
        }
    } else {
        state.basis_jobs.retain(|c, _| highlight.contains(c));
        for &ch in &highlight {
            let job_set = Arc::clone(set);
            let (anchor, shanks) = (selection[0], shanks.to_vec());
            let found = state
                .basis_jobs
                .entry(ch)
                .or_default()
                .ensure(
                    (ch, set.generation, cfg),
                    &jobs.session,
                    jobs.repaint(),
                    move |s, cancel| {
                        let idx = job_set.channels.iter().position(|&c| c == ch)?;
                        estimate_basis(s, &job_set, idx, anchor, cfg, &shanks, cancel).map(Arc::new)
                    },
                )
                .cloned()
                .flatten();
            basis.extend(found.map(|b| (ch, b)));
        }
    }
    state.link.set = set.channels.clone();
    state.link.highlight = highlight;
    state.link.basis = basis;
}

/// Temporal PC shapes on channel `channel_idx` of the set, estimated from
/// the anchor cluster's raw snippets there (precomputed PCs have no basis
/// of their own): `W = (FᵀF)⁻¹FᵀX` with `F` the two PC features and `X` the
/// snippets.
fn estimate_basis<P: DataProvider>(
    s: &Session<P>,
    set: &FeatureSet,
    channel_idx: usize,
    anchor: ClusterId,
    cfg: TracePreproc,
    shanks: &[u32],
    cancel: &AtomicBool,
) -> Option<[Vec<f32>; 2]> {
    let n = set.n();
    let channel = set.channels[channel_idx] as usize;
    let rows: Vec<usize> = (0..set.points.len())
        .filter(|&i| {
            let f = set.row(i);
            set.points[i].slot == 0 && f[channel_idx].is_finite() && f[n + channel_idx].is_finite()
        })
        .collect();
    let rows: Vec<usize> = rows
        .iter()
        .copied()
        .step_by(rows.len().div_ceil(N_BASIS).max(1))
        .collect();
    // Global index of every spike of the anchor, by local index.
    let globals: Vec<u32> = s
        .spike_clusters_global()
        .iter()
        .enumerate()
        .filter(|(_, &c)| c == anchor)
        .map(|(g, _)| g as u32)
        .collect();
    let p = &s.provider;
    let nc = p.n_channels() as usize;
    let half = waveform_window_samples(p.sample_rate().max(1.0));
    let times = s.cluster_index().all_spike_times();
    let (f, x): (Vec<[f32; 2]>, Vec<Vec<f32>>) = rows
        .par_iter()
        .filter_map(|&i| {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let g = *globals.get(set.points[i].local as usize)? as usize;
            let snip = extract_snippet(p, times[g], half, half, cfg, shanks)?;
            let f = set.row(i);
            Some((
                [f[channel_idx], f[n + channel_idx]],
                snip.chunks_exact(nc).map(|row| row[channel]).collect(),
            ))
        })
        .unzip();
    lstsq_basis(&f, &x)
}

/// `W = (FᵀF)⁻¹FᵀX` for `f` (N × 2 features) and `x` (N rows of T samples):
/// the two temporal shapes the features are a projection on. `None` when the
/// features are (nearly) collinear or there are too few rows.
fn lstsq_basis(f: &[[f32; 2]], x: &[Vec<f32>]) -> Option<[Vec<f32>; 2]> {
    let t = x.first()?.len();
    if f.len() < 3 || x.iter().any(|r| r.len() != t) {
        return None;
    }
    let (mut a, mut b, mut c) = (0.0f64, 0.0f64, 0.0f64);
    let (mut u, mut v) = (vec![0.0f64; t], vec![0.0f64; t]);
    for (r, row) in f.iter().zip(x) {
        let (p, q) = (r[0] as f64, r[1] as f64);
        (a, b, c) = (a + p * p, b + p * q, c + q * q);
        for (k, &s) in row.iter().enumerate() {
            u[k] += p * s as f64;
            v[k] += q * s as f64;
        }
    }
    let det = a * c - b * b;
    if det.abs() <= 1e-9 * (a * c).abs().max(f64::MIN_POSITIVE) {
        return None;
    }
    let w0 = (0..t)
        .map(|k| ((c * u[k] - b * v[k]) / det) as f32)
        .collect();
    let w1 = (0..t)
        .map(|k| ((a * v[k] - b * u[k]) / det) as f32)
        .collect();
    Some([w0, w1])
}

fn caption(ui: &mut Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .small()
            .color(Theme::from_ui(ui).caption),
    );
}

/// K-MEANS: the row, then the selected part's 22 px collision matrix with
/// its best and worst view.
fn kmeans_panel<P: DataProvider>(ui: &mut Ui, c: &Ctx<P>, state: &mut FeatureViewState) {
    caption(ui, "K-MEANS");
    kmeans_row(ui, state, c, true, false);
    let Some(k) = &c.coll else { return };
    let (parts, clusters) = k.others(c.selection);
    let text = |cell: usize| {
        let (dx, dy) = GRID[cell / 4][cell % 4];
        let h = k.hit(cell).map_or(0, CellHit::total);
        let n = k.hit(cell).map_or((0, 0), |h| (h.parts, h.clusters));
        format!(
            "{} × {} · {} {parts}{}  ({h})",
            dim_label(dx, c.set),
            dim_label(dy, c.set),
            n.0,
            clusters
                .as_ref()
                .map_or(String::new(), |name| format!(" + {} {name}", n.1)),
        )
    };
    egui::Frame::new()
        .fill(ui.visuals().selection.bg_fill.gamma_multiply(0.35))
        .corner_radius(4.0)
        .inner_margin(8.0)
        .show(ui, |ui| {
            ui.horizontal_top(|ui| {
                let lasso_cell = state.lasso.is_active().then_some(state.lasso.cell);
                if let Some(cell) =
                    collision_matrix(ui, &c.theme, c.set, Some(k), state.focus, lasso_cell, 22.0)
                {
                    goto_cell(state, Some(k), cell);
                }
                ui.vertical(|ui| {
                    ui.label(format!(
                        "part {} · {} spikes{}",
                        k.part + 1,
                        k.size(),
                        if state.lasso.is_closed() {
                            " (lassoed)"
                        } else {
                            ""
                        }
                    ));
                    hint(ui, format!("best  {}", text(k.best())));
                    hint(ui, format!("worst {}", text(k.worst())));
                    hint(ui, "click a tile to move the lasso to that view");
                });
            });
        });
}

/// LASSO: nodes · spikes · cell, and the gesture list.
fn lasso_panel<P: DataProvider>(ui: &mut Ui, c: &Ctx<P>, state: &mut FeatureViewState) {
    caption(ui, "LASSO");
    if state.lasso.is_closed() {
        let n = lasso_hit_count(state, c.set, c.selection);
        let (dx, dy) = GRID[state.lasso.cell / 4][state.lasso.cell % 4];
        ui.strong(format!(
            "{} nodes · {n} spikes · {} × {}",
            state.lasso.points.len(),
            dim_label(dx, c.set),
            dim_label(dy, c.set)
        ));
    } else {
        ui.strong("no lasso");
    }
    egui::Grid::new("lasso-help")
        .min_row_height(14.0)
        .spacing([12.0, 2.0])
        .show(ui, |ui| {
            for (keys, effect) in LASSO_HELP {
                hint(ui, keys);
                hint(ui, effect);
                ui.end_row();
            }
        });
}

/// One grid cell, or the large focused one: `origin` is the top-left of its
/// label strip, `big` tells the two apart. All the plot gestures live here.
fn draw_cell<P: DataProvider>(
    ui: &mut Ui,
    c: &Ctx<P>,
    state: &mut FeatureViewState,
    cell: usize,
    origin: Pos2,
    side: f32,
    big: bool,
) {
    let (theme, set) = (&c.theme, &**c.set);
    let (dx, dy) = GRID[cell / 4][cell % 4];
    let strip = Rect::from_min_size(origin, egui::vec2(side, LABEL_H));
    let rect = Rect::from_min_size(origin + egui::vec2(0.0, LABEL_H), egui::vec2(side, side));
    if !ui.is_rect_visible(strip.union(rect)) {
        return;
    }
    let painter = ui.painter_at(strip.union(rect));
    painter.with_clip_rect(strip).text(
        strip.left_center() + egui::vec2(2.0, 0.0),
        egui::Align2::LEFT_CENTER,
        format!("{} × {}", dim_label(dx, set), dim_label(dy, set)),
        egui::FontId::proportional(11.0),
        theme.caption,
    );
    if let Some((g, ..)) = &c.ghost {
        ui.interact(
            strip,
            ui.id().with(("feature-caption", cell, big)),
            Sense::hover(),
        )
        .on_hover_text(format!("c{} (ghost): the merge candidate, in grey", g.0));
    }
    painter.rect_filled(rect, 0.0, theme.plot_bg);
    let (x, y) = (axis_range(set, state, dx, 0), axis_range(set, state, dy, 1));
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
    // Background first (1 px, faint), the merge ghost (grey, a little
    // smaller), the selection and the brushed spikes (ink) on top.
    if let Some([background, selected, brushed]) = c.cells.get(cell) {
        let ghost = c.ghost.as_ref().map(|(_, gen, cells)| (&cells[cell], *gen));
        let layers = [
            (
                "feature-bg",
                Some((background, c.cells_gen)),
                if c.dark { 0.75 } else { BACKGROUND_RADIUS },
            ),
            ("feature-ghost", ghost, (state.marker * 0.75).max(1.0)),
            ("feature-cell", Some((selected, c.cells_gen)), state.marker),
            (
                "feature-brush",
                Some((brushed, c.cells_gen)),
                state.marker + 0.5,
            ),
        ];
        for (layer, vertices, radius) in layers {
            let Some((vertices, generation)) = vertices else {
                continue;
            };
            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    ScatterCallback {
                        id: ui.id().with((layer, cell, big)).value(),
                        vertices: Arc::clone(vertices),
                        generation,
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
    let strong = ui.visuals().strong_text_color();
    let inner = painter.with_clip_rect(rect);
    // The time brush on a time axis: a band with 1 px edges.
    if let (Some(t), Dim::Time) = (c.time, dx) {
        let (l, r) = (to_screen([t.start, 0.0]).x, to_screen([t.end, 0.0]).x);
        let band = Rect::from_x_y_ranges(l..=r, rect.y_range());
        let sel = ui.visuals().selection;
        inner.rect_filled(band, 0.0, sel.bg_fill.gamma_multiply(0.22));
        for px in [l, r] {
            inner.vline(px, rect.y_range(), Stroke::new(1.0_f32, sel.stroke.color));
        }
    }
    // Spikes inside the Trace window: a thin weak ring.
    let screen_of = |i: usize| {
        let (a, b) = (set.value(i, dx), set.value(i, dy));
        (a.is_finite() && b.is_finite()).then(|| to_screen([a, b]))
    };
    for &i in &c.windowed {
        if let (Some(p), Some(_)) = (screen_of(i), time_alpha(c.time, set.points[i].t)) {
            inner.circle_stroke(p, state.marker + 1.5, Stroke::new(1.0_f32, theme.caption));
        }
    }
    // The spike hovered in any view: the accent ring.
    if let Some(p) = c.hover_pt.and_then(screen_of) {
        inner.circle_stroke(
            p,
            HOVER_PX,
            Stroke::new(1.5_f32, ui.visuals().hyperlink_color),
        );
    }
    if let Some((g, ..)) = &c.ghost {
        painter.text(
            rect.left_top() + egui::vec2(4.0, 4.0),
            egui::Align2::LEFT_TOP,
            format!("c{} (ghost)", g.0),
            egui::FontId::proportional(10.0),
            theme.caption,
        );
    }
    if c.dup && !big && cell == state.focus {
        painter.rect_stroke(
            rect,
            0.0,
            Stroke::new(1.5_f32, ui.visuals().selection.bg_fill),
            egui::StrokeKind::Inside,
        );
    }
    // The part's hull in every other cell, faint and dashed.
    let hit = c.coll.as_ref().and_then(|k| k.hit(cell).map(|h| (k, h)));
    if let Some((_, h)) = hit.filter(|_| !(state.lasso.is_active() && state.lasso.cell == cell)) {
        let hull: Vec<Pos2> = h
            .hull
            .iter()
            .chain(h.hull.first())
            .map(|&p| to_screen(p))
            .collect();
        painter
            .with_clip_rect(rect)
            .extend(egui::Shape::dashed_line(
                &hull,
                Stroke::new(1.0_f32, strong.gamma_multiply(0.5)),
                4.0,
                3.0,
            ));
    }
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
        state.lasso.paint(&painter, to_screen, strong);
        if state.lasso.is_closed() {
            let n = lasso_hit_count(state, set, c.selection);
            hit_badge(
                &painter.with_clip_rect(rect),
                &state.lasso,
                to_screen,
                rect,
                n,
                theme,
                strong,
            );
        }
    }

    let resp = ui.interact(
        rect,
        ui.id().with(("feature-hit", cell, big)),
        Sense::click_and_drag(),
    );
    let large = c.dup && big;
    resp.widget_info(|| {
        egui::WidgetInfo::labeled(
            egui::WidgetType::Other,
            true,
            if large {
                format!("focused feature cell {cell}")
            } else {
                format!("feature cell {cell}")
            },
        )
    });
    // The nearest shown spike of a selected cluster under the pointer.
    // Hovering writes the shared hover; Alt+click on it seeks the Trace.
    let under = resp
        .hover_pos()
        .filter(|_| !resp.dragged())
        .and_then(|at| {
            let pts = c.shown.iter().filter_map(|&i| {
                time_alpha(c.time, set.points[i].t)?;
                Some((i, to_screen([set.value(i, dx), set.value(i, dy)])))
            });
            nearest_spike(
                pts.filter(|(_, p)| p.x.is_finite() && p.y.is_finite()),
                at,
                HOVER_PX,
            )
        })
        .and_then(|i| {
            let p = &set.points[i];
            let cluster = c.selection[p.slot as usize];
            let sample = c.session.spike_times(cluster).get(p.local as usize)?.0;
            Some(HoverSpike {
                cluster,
                local: p.local,
                sample,
            })
        });
    if under.is_some() {
        state.hover_out = under;
    }
    let (alt, shift) = ui.input(|i| (i.modifiers.alt, i.modifiers.shift));
    // Shift+drag along a time axis sets the time brush, Shift+click clears it.
    let time_gesture = shift && !alt && dx == Dim::Time;
    if time_gesture {
        if resp.drag_started() {
            state.time_from = ui
                .input(|i| i.pointer.press_origin())
                .map(|p| to_data(p)[0].clamp(0.0, set.duration));
        }
        if let (true, Some(from), Some(at)) = (resp.dragged(), state.time_from, resp.hover_pos()) {
            let prev_only = c.time.is_some_and(|t| t.only);
            state.time_out = Some(
                TimeBrush::new(from, to_data(at)[0].clamp(0.0, set.duration)).map(|t| TimeBrush {
                    only: prev_only,
                    ..t
                }),
            );
        }
        if resp.clicked() {
            state.time_out = Some(None);
        }
    }
    // phy `select_feature`: Alt+click picks the x (left) or y
    // (right) dimension for the AmplitudeView feature type, unless a spike
    // is under the pointer: then it seeks there.
    let seek = alt && resp.clicked() && under.is_some();
    if let Some(h) = under.filter(|_| seek) {
        state.seek = Some(h.sample);
    }
    let picked = match (alt && !seek, resp.clicked(), resp.secondary_clicked()) {
        (true, true, _) => Some(dx),
        (true, _, true) => Some(dy),
        _ => None,
    };
    if let Some(Dim::Pc { ch, pc }) = picked {
        state.amplitude_dim = Some((set.channels[ch as usize], pc as usize));
    } else if !(seek || time_gesture) {
        // The grid's copy of the focused cell leaves the lasso to the large
        // one, unless it is the one being used: the hover feedback is
        // written by whichever copy runs last.
        let idle =
            c.dup && !big && cell == state.focus && !resp.contains_pointer() && !resp.dragged();
        let ev = if idle {
            LassoEvent::default()
        } else {
            state.lasso.interact(cell, &resp, to_data, to_screen)
        };
        if ev.changed {
            // A hand-edited lasso is the user's: auto-run leaves it.
            state.km.lasso_auto = false;
            if !state.lasso.is_active() {
                state.clear_km_preview();
            } else if state.lasso.cell == cell {
                state.focus = cell;
            }
        }
        if !ev.consumed {
            interact_view(ui, state, &resp);
            if resp.clicked() && !ui.input(|i| i.modifiers.any()) {
                state.focus = cell;
            }
        }
    }

    // Collision badge, top right.
    if let Some((k, h)) = hit {
        let col = collision_colour(theme, h.total(), k.size());
        let galley =
            painter.layout_no_wrap(h.total().to_string(), egui::FontId::proportional(10.0), col);
        let size = (galley.size() + egui::vec2(8.0, 2.0)).max(egui::vec2(18.0, 14.0));
        let pill = Rect::from_min_size(rect.right_top() + egui::vec2(-size.x - 4.0, 4.0), size);
        painter.rect(
            pill,
            size.y / 2.0,
            theme.plot_bg,
            Stroke::new(1.0_f32, col),
            egui::StrokeKind::Inside,
        );
        painter.galley(pill.center() - galley.size() / 2.0, galley, col);
        let (parts, clusters) = k.others(c.selection);
        ui.interact(
            pill,
            ui.id().with(("collision-badge", cell, big)),
            Sense::hover(),
        )
        .on_hover_text(format!(
            "{} of {parts}{} inside part {}'s hull",
            h.parts,
            clusters.map_or(String::new(), |n| format!(", {} of {n}", h.clusters)),
            k.part + 1
        ));
    }
}

/// Focus `cell`; with a previewed part, move the lasso to the part's hull there.
fn goto_cell(state: &mut FeatureViewState, coll: Option<&Coll>, cell: usize) {
    state.focus = cell;
    if let Some(h) = coll.and_then(|k| k.hit(cell)) {
        state.lasso = Lasso::new(cell, h.hull.clone());
        state.km.lasso_auto = true;
        state.km.scroll_to_lasso = true;
    }
}

/// The 4×4 collision matrix: one `tile`-px square per grid cell, coloured by
/// the previewed part's collisions there (neutral without one), the number
/// shown from 18 px. The focused cell is outlined, the lasso's in the
/// selection colour. Returns the clicked cell.
fn collision_matrix(
    ui: &mut Ui,
    theme: &Theme,
    set: &FeatureSet,
    coll: Option<&Coll>,
    focus: usize,
    lasso_cell: Option<usize>,
    tile: f32,
) -> Option<usize> {
    let gap = if tile >= 18.0 { 2.0 } else { 1.0 };
    let (area, _) = ui.allocate_exact_size(Vec2::splat(4.0 * tile + 3.0 * gap), Sense::hover());
    let mut clicked = None;
    for (cell, &(dx, dy)) in GRID.iter().flatten().enumerate() {
        let r = Rect::from_min_size(
            area.min
                + egui::vec2(
                    (cell % 4) as f32 * (tile + gap),
                    (cell / 4) as f32 * (tile + gap),
                ),
            Vec2::splat(tile),
        );
        let resp = ui.interact(
            r,
            ui.make_persistent_id(("collision", cell, tile as u32)),
            Sense::click(),
        );
        let label = format!("collision {} × {}", dim_label(dx, set), dim_label(dy, set));
        resp.widget_info(|| {
            egui::WidgetInfo::labeled(egui::WidgetType::Button, true, label.clone())
        });
        let hit = coll.and_then(|k| k.hit(cell).map(|h| (k, h)));
        let (fill, text) = hit.map_or((theme.grid.gamma_multiply(0.6), theme.text), |(k, h)| {
            tile_colours(theme, ui, h.total(), k.size())
        });
        let painter = ui.painter();
        painter.rect_filled(r, 1.0, fill);
        if let (Some((_, h)), true) = (hit, tile >= 18.0) {
            painter.text(
                r.center(),
                egui::Align2::CENTER_CENTER,
                h.total().to_string(),
                egui::FontId::proportional((tile * 0.45).min(11.0)),
                text,
            );
        }
        if cell == focus {
            painter.rect_stroke(
                r,
                1.0,
                Stroke::new(1.5_f32, ui.visuals().strong_text_color()),
                egui::StrokeKind::Inside,
            );
        } else if lasso_cell == Some(cell) {
            painter.rect_stroke(
                r,
                1.0,
                Stroke::new(1.0_f32, ui.visuals().selection.bg_fill),
                egui::StrokeKind::Inside,
            );
        }
        if resp.clicked() {
            clicked = Some(cell);
        }
        resp.on_hover_text(match hit {
            Some((_, h)) => format!("{label}: {} collisions", h.total()),
            None => label,
        });
    }
    clicked
}

/// Toolbar: `[2D|3D]`, the channels, and the rest inline (Regular and up)
/// or in the ⚙ popover (always for marker size and the source; Compact also
/// holds `fixed (c)`, auto-run and the hint there).
fn toolbar(
    ui: &mut Ui,
    state: &mut FeatureViewState,
    set: &FeatureSet,
    busy: bool,
    layout: Layout,
    depths: Option<&SpikeDepths>,
) {
    ui.horizontal(|ui| {
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            for (mode, text) in [(ViewMode::Two, "2D"), (ViewMode::Three, "3D")] {
                if ui
                    .add(egui::Button::new(text).selected(state.mode == mode))
                    .clicked()
                {
                    state.mode = mode;
                }
            }
        });
        if state.mode == ViewMode::Three && layout != Layout::Compact {
            pin_chip(ui, state, set);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.menu_button("⚙", |ui| {
                ui.set_min_width(180.0);
                if layout == Layout::Compact {
                    fixed_controls(ui, state);
                    ui.checkbox(&mut state.km.auto_run, "auto-run")
                        .on_hover_text(AUTO_RUN_HELP);
                    hint(ui, hint_line(state));
                }
                if state.edited.is_some()
                    && ui
                        .button("automatic channel set")
                        .on_hover_text("Back to the selection's best channels")
                        .clicked()
                {
                    state.edit_channels(ChannelEdit::Reset);
                }
                colour_by(ui, state, set, depths);
                ui.horizontal(|ui| {
                    ui.label("marker size");
                    ui.add(
                        egui::DragValue::new(&mut state.marker)
                            .range(0.5..=8.0)
                            .speed(0.05),
                    );
                });
                hint(
                    ui,
                    if set.computed {
                        "PCs fitted on raw snippets"
                    } else {
                        "pc_features.npy"
                    },
                );
            })
            .response
            .on_hover_text("More settings");
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                if layout != Layout::Compact {
                    set_chips(ui, state, set);
                }
                if busy {
                    ui.spinner();
                }
                if layout != Layout::Compact {
                    fixed_controls(ui, state);
                    // The old "ⓘ" glyph rendered as an empty checkbox-like box
                    // (missing glyph); plain ASCII always draws.
                    ui.label("(?)").on_hover_text(help_text());
                    let colour = hint_colour(&Theme::from_ui(ui));
                    ui.add(
                        egui::Label::new(egui::RichText::new(hint_line(state)).color(colour))
                            .truncate(),
                    );
                }
            });
        });
    });
    if layout == Layout::Compact {
        ui.horizontal_wrapped(|ui| {
            set_chips(ui, state, set);
            if state.mode == ViewMode::Three {
                pin_chip(ui, state, set);
            }
        });
    }
}

/// "colour by: cluster | time | depth" and, for the last two, the ramp.
fn colour_by(
    ui: &mut Ui,
    state: &mut FeatureViewState,
    set: &FeatureSet,
    depths: Option<&SpikeDepths>,
) {
    ui.horizontal(|ui| {
        ui.label("colour by:");
        for (mode, text) in [
            (ColourBy::Cluster, "cluster"),
            (ColourBy::Time, "time"),
            (ColourBy::Depth, "depth"),
        ] {
            let usable = mode != ColourBy::Depth || depths.is_some();
            let resp = ui.add_enabled(
                usable,
                egui::Button::new(text)
                    .small()
                    .selected(state.colour == mode),
            );
            if resp
                .on_disabled_hover_text("No spike depths for this recording")
                .clicked()
            {
                state.colour = mode;
            }
        }
    });
    let (lo, hi, unit) = match (state.colour, depths) {
        (ColourBy::Time, _) => (0.0, set.duration, "s"),
        (ColourBy::Depth, Some(d)) => (d.min, d.max, "\u{b5}m"),
        _ => return,
    };
    let (rect, _) = ui.allocate_exact_size(egui::vec2(160.0, 8.0), Sense::hover());
    let painter = ui.painter();
    for k in 0..5 {
        let w = rect.width() / 5.0;
        painter.rect_filled(
            Rect::from_min_size(
                rect.min + egui::vec2(k as f32 * w, 0.0),
                egui::vec2(w, rect.height()),
            ),
            0.0,
            viridis(k as f32 / 4.0),
        );
    }
    let colour = Theme::from_ui(ui).caption;
    let font = egui::FontId::proportional(9.0);
    painter.text(
        rect.left_bottom() + egui::vec2(0.0, 1.0),
        egui::Align2::LEFT_TOP,
        format!("{lo:.0} {unit}"),
        font.clone(),
        colour,
    );
    painter.text(
        rect.right_bottom() + egui::vec2(0.0, 1.0),
        egui::Align2::RIGHT_TOP,
        format!("{hi:.0} {unit}"),
        font,
        colour,
    );
    ui.add_space(10.0);
}

/// The 3D pin chip for the focused cell's two axes.
fn pin_chip(ui: &mut Ui, state: &mut FeatureViewState, set: &FeatureSet) {
    let focus = GRID[state.focus / 4][state.focus % 4];
    crate::feature_view3d::pin_chip(ui, &mut state.view3d, set, focus);
}

/// phy's `c`: keep A and B as they are instead of following the best pair.
fn fixed_controls(ui: &mut Ui, state: &mut FeatureViewState) {
    let mut fixed = state.fixed_channels.is_some();
    if ui
        .checkbox(&mut fixed, "fixed A/B (c)")
        .on_hover_text("Keep A and B instead of following the best-separating channel pair")
        .changed()
    {
        state.toggle_channels();
    }
    if let Some([mut a, mut b]) = state.fixed_channels {
        let (a0, b0) = (a, b);
        ui.add(egui::DragValue::new(&mut a).speed(0.2));
        ui.add(egui::DragValue::new(&mut b).speed(0.2));
        if a != a0 {
            state.edit_channels(ChannelEdit::SetA(a));
        } else if b != b0 {
            state.edit_channels(ChannelEdit::SetB(b));
        }
    }
}

const AUTO_RUN_HELP: &str = "Refit (on all cores, off the UI thread) whenever the selection or \
features change; the lasso follows unless you drew one";

/// The finished k-means fit for `key`, if there is one.
fn km_fit(state: &FeatureViewState, key: KmKey) -> Option<Arc<KmFit>> {
    match &state.km.fit {
        Some((k, Some(fit))) if *k == key => Some(Arc::clone(fit)),
        _ => None,
    }
}

/// Runs the k-means fit as a job and, when it lands, wraps the lasso around
/// the smallest part unless the user drew their own lasso since.
fn kmeans_update<P: DataProvider>(
    state: &mut FeatureViewState,
    set: &Arc<FeatureSet>,
    key: KmKey,
    anchor: ClusterId,
    jobs: &JobCtx<Session<P>>,
) {
    if !(state.km.requested || state.km.auto_run) {
        return;
    }
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
                // The smallest part a lasso could be drawn around.
                state.km.part = (0..fit.parts.len())
                    .rev()
                    .find(|&i| fit.lassos[i].is_active())
                    .unwrap_or(fit.parts.len() - 1);
                apply_part(state, set, &fit, anchor);
            }
            state.km.applied = Some(key);
            state.km.requested = false;
        }
    }
}

/// The k-means row: button, k, auto-run (`auto_run`) and one chip per part;
/// with `mini` the 5 px collision matrix and the part's best cell.
fn kmeans_row<P: DataProvider>(
    ui: &mut Ui,
    state: &mut FeatureViewState,
    c: &Ctx<P>,
    auto_run: bool,
    mini: bool,
) {
    let anchor = c.selection[0];
    let mut chosen = None;
    let mut goto = None;
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
        if auto_run {
            ui.checkbox(&mut state.km.auto_run, "auto-run")
                .on_hover_text(AUTO_RUN_HELP);
        }
        if state.km.memo.pending() {
            ui.spinner();
        }
        match &state.km.fit {
            Some((k, Some(fit))) if *k == c.km_key => {
                for (i, part) in fit.parts.iter().enumerate() {
                    let label = format!("part {} · {}", i + 1, part.len());
                    if ui
                        .add(
                            egui::Button::new(label)
                                .selected(state.km.part == i && state.km.previewing),
                        )
                        .clicked()
                    {
                        chosen = Some(i);
                    }
                }
            }
            Some((k, None)) if *k == c.km_key => {
                ui.weak("too few spikes to cluster");
            }
            _ => {}
        }
        if let (true, Some(k)) = (mini, &c.coll) {
            let lasso_cell = state.lasso.is_active().then_some(state.lasso.cell);
            goto = collision_matrix(ui, &c.theme, c.set, Some(k), state.focus, lasso_cell, 5.0);
            let best = k.best();
            let (dx, dy) = GRID[best / 4][best % 4];
            hint(
                ui,
                format!(
                    "collisions · best {} × {}: {} of {}",
                    dim_label(dx, c.set),
                    dim_label(dy, c.set),
                    k.hit(best).map_or(0, CellHit::total),
                    k.size()
                ),
            );
        }
    });
    if let (Some(i), Some(fit)) = (chosen, km_fit(state, c.km_key)) {
        state.km.part = i;
        apply_part(state, c.set, &fit, anchor);
        ui.ctx().request_repaint();
    }
    if let Some(cell) = goto {
        goto_cell(state, c.coll.as_ref(), cell);
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

/// Evenly thinned to at most [`LINK_SPIKES`].
fn thin(v: &[u32]) -> Arc<[u32]> {
    if v.len() <= LINK_SPIKES {
        return v.into();
    }
    (0..LINK_SPIKES)
        .map(|k| v[k * v.len() / LINK_SPIKES])
        .collect()
}

/// Selected spikes inside the closed lasso, cached per set and polygon.
fn lasso_hit_count(
    state: &mut FeatureViewState,
    set: &FeatureSet,
    selection: &[ClusterId],
) -> usize {
    lasso_hits_cached(state, set, selection).n
}

fn lasso_hits_cached<'a>(
    state: &'a mut FeatureViewState,
    set: &FeatureSet,
    selection: &[ClusterId],
) -> &'a Hits {
    let fresh = |h: &Hits| h.generation == set.generation && h.polygon == state.lasso.points;
    if !state.hits.as_ref().is_some_and(fresh) {
        let (dx, dy) = GRID[state.lasso.cell / 4][state.lasso.cell % 4];
        let mut per: Vec<Vec<u32>> = vec![Vec::new(); selection.len()];
        for (i, p) in set.points.iter().enumerate() {
            if p.slot != BACKGROUND && state.lasso.contains([set.value(i, dx), set.value(i, dy)]) {
                per[p.slot as usize].push(p.local);
            }
        }
        state.hits = Some(Hits {
            generation: set.generation,
            polygon: state.lasso.points.clone(),
            n: per.iter().map(Vec::len).sum(),
            spikes: selection
                .iter()
                .copied()
                .zip(&per)
                .filter(|(_, v)| !v.is_empty())
                .map(|(c, v)| (c, thin(v)))
                .collect(),
            all: selection
                .iter()
                .copied()
                .zip(per)
                .filter(|(_, v)| !v.is_empty())
                .collect(),
        });
    }
    state.hits.as_ref().expect("just filled")
}

/// "N spikes · M nodes · K splits" under the lasso, kept inside `cell`.
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
        format!("{n} spikes · {} nodes · K splits", lasso.points.len()),
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
        Dim::Time => set.bound(dim),
        Dim::Pc { .. } => {
            let (a, b) = set.bound(dim);
            let half = (b - a) / 2.0 / state.scaling;
            (((a + b) / 2.0) - half, ((a + b) / 2.0) + half)
        }
    };
    let span = hi - lo;
    let centre = 0.5 + state.pan[axis];
    let half = 0.5 / state.zoom;
    (lo + (centre - half) * span, lo + (centre + half) * span)
}

pub(crate) fn dim_label(dim: Dim, set: &FeatureSet) -> String {
    match dim {
        Dim::Time => "time".into(),
        Dim::Pc { ch, pc } => format!("ch{} PC{}", set.channels[ch as usize], pc + 1),
    }
}

/// Point indices to draw, background first, each cluster (and the background
/// with `thin_background`) thinned to at most `cap` points; `kept` points
/// (a previewed split, brushed spikes) are always kept and drawn last.
fn display_order(
    set: &FeatureSet,
    cap: usize,
    thin_background: bool,
    kept: &dyn Fn(usize) -> bool,
) -> Vec<usize> {
    let mut counts: HashMap<u16, usize> = HashMap::new();
    for p in &set.points {
        *counts.entry(p.slot).or_default() += 1;
    }
    let stride = |slot: u16| (counts[&slot] / cap).max(1);
    let mut order: Vec<usize> = (0..set.points.len())
        .filter(|&i| {
            let p = &set.points[i];
            kept(i)
                || if p.slot == BACKGROUND {
                    !thin_background || i.is_multiple_of(stride(BACKGROUND))
                } else {
                    (p.local as usize).is_multiple_of(stride(p.slot))
                }
        })
        .collect();
    // Background under the clusters, a previewed split on top.
    order.sort_by_key(|&i| (set.points[i].slot != BACKGROUND, kept(i)));
    order
}

/// `[background, selected, brushed]` vertices of `order` in the `dx` × `dy`
/// plane: the brushed spikes are drawn again on top in ink, and spikes
/// outside a time brush fade (or drop out).
fn layer_vertices(
    set: &FeatureSet,
    order: &[usize],
    (dx, dy): (Dim, Dim),
    paint: &Paint,
    previewed: &dyn Fn(usize) -> bool,
) -> [Arc<Vec<ScatterVertex>>; 3] {
    let dark = paint.dark;
    // Faint background: gray(170)@~0.47 in dark, gray(190)@0.5 in light.
    let background = if dark {
        pack_rgba(170, 170, 170, 120)
    } else {
        pack_rgba(190, 190, 190, 128)
    };
    // The previewed split and the brushed spikes: brush ink.
    let ink = if dark { [255, 255, 255] } else { [20, 20, 20] };
    // `order` is sorted background-first, so a stable partition keeps the
    // previewed split last within the selection.
    let layer = |which: usize| {
        Arc::new(
            order
                .iter()
                .filter_map(|&i| {
                    let p = &set.points[i];
                    let bg = p.slot == BACKGROUND;
                    let brushed = !bg && paint.brushed(p);
                    if (which == 0) != bg || (which == 2 && !brushed) {
                        return None;
                    }
                    let (x, y) = (set.value(i, dx), set.value(i, dy));
                    if !(x.is_finite() && y.is_finite()) {
                        return None;
                    }
                    let fade = if bg {
                        1.0
                    } else {
                        time_alpha(paint.time, p.t)?
                    };
                    let (rgb, a) = match which {
                        0 => {
                            return Some(ScatterVertex {
                                pos: [x, y],
                                color: background,
                            })
                        }
                        2 => (ink, 240.0),
                        _ if previewed(i) => (ink, 240.0),
                        _ => {
                            let c = paint.point_colour(p, set.duration);
                            ([c.r(), c.g(), c.b()], 255.0)
                        }
                    };
                    Some(ScatterVertex {
                        pos: [x, y],
                        color: pack_rgba(rgb[0], rgb[1], rgb[2], (a * fade) as u8),
                    })
                })
                .collect(),
        )
    };
    [layer(0), layer(1), layer(2)]
}

/// Per-cell vertex buffers: background first, then each cluster thinned to
/// at most [`N_DISPLAY`] points; and the non-background points drawn.
fn cell_vertices(
    set: &FeatureSet,
    paint: &Paint,
    preview: Option<(u16, &std::collections::HashSet<u32>)>,
) -> (CellVertices, Vec<usize>) {
    let previewed = |i: usize| {
        let p = &set.points[i];
        preview.is_some_and(|(slot, idx)| p.slot == slot && idx.contains(&p.local))
    };
    // Brushed spikes survive the thinning, so a brush made elsewhere (ISI
    // violators, an Amplitudes lasso) shows every spike it holds.
    let kept = |i: usize| previewed(i) || paint.brushed(&set.points[i]);
    let order = display_order(set, N_DISPLAY, false, &kept);
    let cells = GRID
        .iter()
        .flatten()
        .map(|&dims| layer_vertices(set, &order, dims, paint, &previewed))
        .collect();
    let shown = order
        .into_iter()
        .filter(|&i| set.points[i].slot != BACKGROUND)
        .collect();
    (cells, shown)
}

fn lasso_hits(
    set: &FeatureSet,
    lasso: &Lasso,
    selection: &[ClusterId],
) -> Vec<(ClusterId, Vec<u32>)> {
    let (dx, dy) = GRID[lasso.cell / 4][lasso.cell % 4];
    let mut hits: Vec<Vec<u32>> = vec![Vec::new(); selection.len()];
    for (i, p) in set
        .points
        .iter()
        .enumerate()
        .filter(|(_, p)| p.slot != BACKGROUND)
    {
        let xy = [set.value(i, dx), set.value(i, dy)];
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

pub(crate) fn compute<P: DataProvider>(
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

    let (channels, points, feats) = if computed {
        computed_features(s, key, basis, &picked, cancel)
    } else {
        let channels = set_channels(key.channels.clone(), || s.best_channels(&key.selection));
        let n = channels.len();
        let points = picked
            .iter()
            .map(|&(g, local, slot)| FeaturePoint {
                local,
                slot,
                t: t_of(g),
            })
            .collect();
        let feats = picked
            .iter()
            .flat_map(|&(g, _, slot)| {
                let missing = if slot == BACKGROUND { f32::NAN } else { 0.0 };
                let channels = &channels;
                (0..2 * n).map(move |i| s.pc_feature(g, channels[i % n], i / n).unwrap_or(missing))
            })
            .collect();
        (channels, points, feats)
    };
    // Background spikes without any feature are dropped.
    let width = 2 * channels.len();
    let (points, feats): (Vec<FeaturePoint>, Vec<f32>) = {
        let keep = |i: usize| {
            points[i].slot != BACKGROUND
                || feats[i * width..(i + 1) * width]
                    .iter()
                    .any(|v| v.is_finite())
        };
        let kept: Vec<usize> = (0..points.len()).filter(|&i| keep(i)).collect();
        (
            kept.iter().map(|&i| points[i]).collect(),
            kept.iter()
                .flat_map(|&i| &feats[i * width..(i + 1) * width])
                .copied()
                .collect(),
        )
    };
    let set = FeatureSet::new(
        channels,
        computed,
        key.full || !computed,
        points,
        feats,
        s.provider.n_samples().0 as f32 / sr,
    );
    if key.keep_order || set.n() < 3 || key.selection.len() < 2 || cancel.load(Ordering::Relaxed) {
        return set;
    }
    // A and B: the channel pair that separates the selected clusters best.
    let labels = labels_for(&set, &Labels::Clusters);
    let rows = score_rows(&labels);
    let n = set.n();
    let pairs: Vec<[usize; 2]> = (0..n)
        .flat_map(|a| (a + 1..n).map(move |b| [a, b]))
        .collect();
    let scores: Vec<f32> = pairs
        .par_iter()
        .map(|&[a, b]| {
            let dims = [a, b].map(|c| [1 + c, 1 + n + c]).concat();
            separation(&set, &dims, &rows, &labels)
        })
        .collect();
    let best = (0..pairs.len())
        .filter(|&k| scores[k].is_finite())
        .max_by(|&x, &y| scores[x].total_cmp(&scores[y]).then(y.cmp(&x)));
    match best {
        Some(k) if pairs[k] != [0, 1] => {
            let [a, b] = pairs[k];
            let order: Vec<usize> = [a, b]
                .into_iter()
                .chain((0..n).filter(|&c| c != a && c != b))
                .collect();
            set.permuted(&order)
        }
        _ => set,
    }
}

/// The channel set to compute: `requested` (or `auto`'s first four when
/// `None`), deduplicated, at most [`MAX_SET`] and at least two.
fn set_channels(requested: Option<Vec<u32>>, auto: impl FnOnce() -> Vec<u32>) -> Vec<u32> {
    let (wanted, cap) = match requested {
        Some(r) => (r, MAX_SET),
        None => (auto(), 4),
    };
    let mut set: Vec<u32> = Vec::new();
    for c in wanted {
        if !set.contains(&c) && set.len() < cap {
            set.push(c);
        }
    }
    if set.is_empty() {
        set.push(0);
    }
    if set.len() < 2 {
        set.push(set[0]);
    }
    set
}

/// 0.5-99.5 percentile range of the finite values, padded 5% each side
/// (`(-1, 1)` when there are none).
pub(crate) fn robust_range(vals: impl Iterator<Item = f32>) -> (f32, f32) {
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

/// `(channels, points, flat features)` of `picked` projected on the fitted
/// temporal PCs.
fn computed_features<P: DataProvider>(
    s: &Session<P>,
    key: &FeatureKey,
    basis: &Basis,
    picked: &[(u32, u32, u16)],
    cancel: &AtomicBool,
) -> (Vec<u32>, Vec<FeaturePoint>, Vec<f32>) {
    let p = &s.provider;
    let sr = p.sample_rate().max(1.0);
    let nc = p.n_channels() as usize;
    let half = waveform_window_samples(sr);
    let times = s.cluster_index().all_spike_times();
    let Some(b) = fitted_basis(s, key, basis, half, cancel) else {
        return (vec![0, 0], Vec::new(), Vec::new());
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

    // Automatic channels: the four with the largest mean |PC1| over the
    // first cluster's spikes (the temporal-PC analogue of peak amplitude).
    let channels = set_channels(key.channels.clone(), || {
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
        order.into_iter().map(|c| c as u32).collect()
    });
    let n = channels.len();
    let rows: Vec<(FeaturePoint, [f32; 2 * MAX_SET])> = picked
        .par_iter()
        .filter_map(|&(g, local, slot)| {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let snip = snippet(g)?;
            let mut f = [0.0; 2 * MAX_SET];
            for (i, v) in f.iter_mut().enumerate().take(2 * n) {
                *v = project(&snip, i / n, channels[i % n] as usize);
            }
            Some((
                FeaturePoint {
                    local,
                    slot,
                    t: s.cluster_index().display_time(g, key.reorder).0 as f32 / sr,
                },
                f,
            ))
        })
        .collect();
    let feats = rows
        .iter()
        .flat_map(|(_, f)| f[..2 * n].iter().copied())
        .collect();
    (channels, rows.into_iter().map(|r| r.0).collect(), feats)
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
            keep_order: false,
            version: 0,
            cfg: TracePreproc::default(),
            shanks: Vec::new(),
            full: false,
            reorder: false,
        };
        let cur = lasso_key(&key, &[3, 4]);
        let mut st = FeatureViewState::default();
        let tri = Lasso::new(0, vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]);
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
        st.drop_stale_lasso(&lasso_key(&full, &[3, 4]));
        assert!(st.lasso.is_closed() && st.split_requested);
        // A new selection drops lasso and the pending split.
        let other = FeatureKey {
            selection: vec![ClusterId(2)],
            ..key
        };
        st.drop_stale_lasso(&lasso_key(&other, &[3, 4]));
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
        blobs(0)
    }

    /// [`two_blobs`] plus `n` points of a second cluster (slot 1) inside the
    /// small blob.
    fn blobs(n: u32) -> FeatureSet {
        let mut s = 0x5EED_u32;
        let mut noise = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s as f32 / u32::MAX as f32 - 0.5
        };
        let (mut points, mut feats) = (Vec::new(), Vec::new());
        let mut push = |local, slot, t, f: [f32; 4]| {
            points.push(FeaturePoint { local, slot, t });
            feats.extend(f);
        };
        for local in 0..500u32 {
            let c = if local < 400 { 0.0 } else { 5.0 };
            let f = [c + noise(), c + noise(), c + noise(), c + noise()];
            push(local, 0, local as f32 / 500.0, f);
        }
        for local in 0..n {
            push(local, 1, 0.9, [5.0; 4]);
        }
        for _ in 0..50 {
            push(0, BACKGROUND, 0.5, [2.5; 4]);
        }
        let mut set = FeatureSet::new(vec![0, 1], false, true, points, feats, 1.0);
        set.bounds = vec![(-1.0, 6.0); 4];
        set
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
        let inside = |i: usize| lasso.contains([set.value(i, dx), set.value(i, dy)]);
        let pts = &set.points;
        let of = |f: fn(u32) -> bool| {
            (0..pts.len()).filter(move |&i| pts[i].slot == 0 && f(pts[i].local))
        };
        assert_eq!(
            of(|l| l < 400).filter(|&i| inside(i)).count(),
            0,
            "no spike of the big blob"
        );
        // The hull trims the 5% outermost points, so most but not all.
        assert!(of(|l| l >= 400).filter(|&i| inside(i)).count() >= 90);
        assert!(fit_kmeans(&set, 0, Some(3)).is_some_and(|f| f.parts.len() == 3));
        assert!(
            fit_kmeans(&set, 7, None).is_none(),
            "no points for that slot"
        );
    }

    #[test]
    fn cell_hits_count_other_parts_and_clusters_inside_each_hull() {
        let set = blobs(30);
        let fit = fit_kmeans(&set, 0, None).expect("two blobs cluster");
        assert_eq!(fit.hits.len(), 2);
        let others: Vec<usize> = (0..set.points.len())
            .filter(|&i| set.points[i].slot == 1)
            .collect();
        for (part, cells) in fit.hits.iter().enumerate() {
            assert_eq!(cells.len(), 16);
            for (cell, hit) in cells.iter().enumerate() {
                let Some(hit) = hit else { continue };
                let (dx, dy) = GRID[cell / 4][cell % 4];
                let probe = Lasso::new(cell, hit.hull.clone());
                let inside = |i: usize| probe.contains([set.value(i, dx), set.value(i, dy)]);
                let old: usize = fit.parts[1 - part].iter().filter(|&&i| inside(i)).count();
                assert_eq!(hit.parts, old, "part {part} cell {cell}");
                let new = others.iter().filter(|&&i| inside(i)).count();
                assert_eq!(hit.clusters, new, "part {part} cell {cell}");
            }
        }
        // Slot 1 sits in the small blob (part 1), in every cell: the hull trims
        // at most the 5% outermost points, and the big blob's never reaches it.
        let best = fit.lassos[1].cell;
        assert!(fit.hits[1][best].as_ref().unwrap().clusters >= 28);
        assert!(fit.hits[0].iter().flatten().all(|h| h.clusters == 0));
        // The best cell of the small part is the one with the fewest.
        let total = |h: &Option<CellHit>| h.as_ref().map_or(usize::MAX, CellHit::total);
        assert_eq!(
            total(&fit.hits[1][best]),
            fit.hits[1].iter().map(total).min().unwrap()
        );
    }

    #[test]
    fn collision_colours_follow_the_15_percent_rule() {
        let theme = Theme::from_visuals(&egui::Visuals::dark());
        assert_eq!(collision_colour(&theme, 0, 100), theme.good);
        assert_eq!(collision_colour(&theme, 15, 100), theme.warn);
        assert_eq!(collision_colour(&theme, 16, 100), theme.danger);
    }

    #[test]
    fn layout_follows_the_pane_width() {
        for (w, layout) in [
            (200.0, Layout::Compact),
            (479.9, Layout::Compact),
            (480.0, Layout::Regular),
            (720.0, Layout::Regular),
            (1099.9, Layout::Regular),
            (1100.0, Layout::Expansive),
            (1920.0, Layout::Expansive),
        ] {
            assert_eq!(layout_for(egui::vec2(w, 700.0)), layout, "width {w}");
        }
    }

    #[test]
    fn expansive_grid_fits_the_height_and_half_the_width() {
        for size in [egui::vec2(1920.0, 1000.0), egui::vec2(1100.0, 900.0)] {
            let side = expansive_side(size);
            let (w, h) = (
                4.0 * side + 3.0 * CELL_GAP,
                4.0 * (side + LABEL_H) + 3.0 * CELL_GAP,
            );
            assert!(w <= size.x / 2.0 + 1e-3 && h <= size.y + 1e-3, "{size:?}");
        }
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
        let rows = [
            (0, 0, 0.5, 0.5),
            (1, 0, 5.0, 5.0),
            (0, 1, 0.2, 0.8),
            (0, BACKGROUND, 0.5, 0.5),
        ];
        let points = rows
            .iter()
            .map(|&(local, slot, ..)| FeaturePoint {
                local,
                slot,
                t: 0.0,
            })
            .collect();
        let feats = rows.iter().flat_map(|r| [r.2, r.3, 0.0, 0.0]).collect();
        let mut set = FeatureSet::new(vec![0, 1], false, true, points, feats, 1.0);
        set.bounds = vec![(-1.0, 1.0); 4];
        // Cell 1 plots x = 1A (f[1]) against y = 0A (f[0]).
        let lasso = Lasso::new(1, vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let hits = lasso_hits(&set, &lasso, &[ClusterId(7), ClusterId(9)]);
        assert_eq!(hits, vec![(ClusterId(7), vec![0]), (ClusterId(9), vec![0])]);
    }

    /// Three channels; clusters 0 and 1 differ only on channels 20 (PC1) and
    /// 30 (PC1), so that pair separates them and nothing else does.
    fn three_channels() -> FeatureSet {
        let mut s = 0xBEEF_u32;
        let mut noise = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s % 1000) as f32 / 1000.0
        };
        let (mut points, mut feats) = (Vec::new(), Vec::new());
        for slot in 0..2u16 {
            for local in 0..120 {
                points.push(FeaturePoint {
                    local,
                    slot,
                    t: noise(),
                });
                let off = slot as f32 * 2.0;
                // Index pc * 3 + ch over channels [10, 20, 30].
                feats.extend([
                    noise(),
                    off + 0.2 * noise(),
                    off + 0.2 * noise(),
                    noise(),
                    noise(),
                    noise(),
                ]);
            }
        }
        FeatureSet::new(vec![10, 20, 30], false, true, points, feats, 1.0)
    }

    #[test]
    fn separation_ranks_the_separating_pair_first() {
        let set = three_channels();
        let labels = labels_for(&set, &Labels::Clusters);
        let rows = score_rows(&labels);
        assert_eq!(rows.len(), 240);
        let dim = |ch, pc| set.dim_idx(Dim::Pc { ch, pc });
        let good = separation(&set, &[dim(1, 0), dim(2, 0)], &rows, &labels);
        let mixed = separation(&set, &[dim(0, 0), dim(1, 0)], &rows, &labels);
        let blind = separation(&set, &[dim(0, 0), dim(0, 1)], &rows, &labels);
        assert!(
            good > 0.7 && good > mixed + 0.2 && good > blind + 0.2,
            "{good} {mixed} {blind}"
        );
        let scores = score_pairs(&set, &labels, &AtomicBool::new(false));
        assert_eq!(scores.len(), set.n_dims() * (set.n_dims() - 1) / 2);
        let best = scores.iter().max_by(|a, b| a.2.total_cmp(&b.2)).unwrap();
        assert_eq!((best.0, best.1), (dim(1, 0), dim(2, 0)));
        // One group has no separation: the spread, not NaN.
        let one = labels_for(&set, &Labels::Variance);
        assert!(separation(&set, &[1, 2], &rows, &one) > 0.0);
    }

    #[test]
    fn permuting_a_set_moves_columns_with_their_channels() {
        let set = three_channels();
        let p = set.permuted(&[2, 0, 1]);
        assert_eq!(p.channels, vec![30, 10, 20]);
        for i in [0, 50, 200] {
            for pc in 0..2u8 {
                for (ch, old) in [(0u8, 2u8), (1, 0), (2, 1)] {
                    assert_eq!(
                        p.value(i, Dim::Pc { ch, pc }),
                        set.value(i, Dim::Pc { ch: old, pc })
                    );
                }
            }
        }
        assert_eq!(
            p.bound(Dim::Pc { ch: 0, pc: 1 }),
            set.bound(Dim::Pc { ch: 2, pc: 1 })
        );
    }

    #[test]
    fn edit_channels_keeps_two_to_max_and_swaps_like_phy() {
        let mut st = FeatureViewState {
            shown_channels: Some(vec![3, 5]),
            ..Default::default()
        };
        st.edit_channels(ChannelEdit::Toggle(5));
        assert_eq!(st.edited, None, "never below two");
        for ch in 7..30 {
            st.edit_channels(ChannelEdit::Toggle(ch));
            st.shown_channels = st.edited.clone();
        }
        assert_eq!(st.edited.as_ref().map(Vec::len), Some(MAX_SET));
        st.edit_channels(ChannelEdit::Toggle(7));
        assert_eq!(st.edited.as_ref().map(Vec::len), Some(MAX_SET - 1));
        // Ctrl+click on the current B makes it A and A goes to B (phy).
        st.edit_channels(ChannelEdit::SetA(5));
        assert_eq!(st.fixed_channels, Some([5, 3]));
        assert_eq!(st.edited.as_ref().unwrap()[..2], [5, 3]);
        st.shown_channels = st.edited.clone();
        st.edit_channels(ChannelEdit::SetB(5));
        assert_eq!(st.fixed_channels, Some([3, 5]), "SetB on A swaps");
        // A channel outside the set joins it, and the cap holds.
        st.shown_channels = st.edited.clone();
        st.edit_channels(ChannelEdit::SetB(99));
        let set = st.edited.clone().unwrap();
        assert_eq!(&set[..2], &[3, 99]);
        assert!(set.len() <= MAX_SET && !set.contains(&5) || set.len() == MAX_SET);
        // Removing a fixed channel unfixes; Reset is automatic again.
        st.shown_channels = st.edited.clone();
        st.edit_channels(ChannelEdit::Toggle(99));
        assert_eq!(st.fixed_channels, None);
        st.edit_channels(ChannelEdit::Reset);
        assert!(st.edited.is_none() && st.fixed_channels.is_none());
    }

    #[test]
    fn promotion_maps_every_pair_to_a_grid_cell() {
        let channels = [10, 11, 12, 13];
        let set = {
            let points = vec![FeaturePoint {
                local: 0,
                slot: 0,
                t: 0.0,
            }];
            FeatureSet::new(channels.to_vec(), false, true, points, vec![0.0; 8], 1.0)
        };
        let id = |d: Dim, ch: &[u32]| match d {
            Dim::Time => None,
            Dim::Pc { ch: c, pc } => Some((ch[c as usize], pc)),
        };
        for a in 0..set.n_dims() {
            for b in a + 1..set.n_dims() {
                let (x, y) = (set.dim(a), set.dim(b));
                let (ab, cell) = promotion(&channels, x, y);
                assert_ne!(ab[0], ab[1]);
                let (gx, gy) = GRID[cell / 4][cell % 4];
                // The cell plots the same two quantities on the new A/B.
                assert_eq!(
                    (id(gx, &ab), id(gy, &ab)),
                    (id(x, &channels), id(y, &channels))
                );
                assert_eq!(gx == Dim::Time, x == Dim::Time);
                // Both channels of the pair are A/B (or stay when only one is used).
                for d in [x, y] {
                    if let Some((c, _)) = id(d, &channels) {
                        assert!(
                            ab.contains(&c)
                                || id(if d == x { y } else { x }, &channels)
                                    .is_some_and(|o| o.0 != c)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn least_squares_recovers_a_known_basis() {
        let t = 30;
        let truth = [
            (0..t).map(|k| (k as f32 * 0.3).sin()).collect::<Vec<f32>>(),
            (0..t)
                .map(|k| (k as f32 * 0.17).cos() - 0.5)
                .collect::<Vec<f32>>(),
        ];
        let mut s = 0x1234_u32;
        let mut noise = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s as f32 / u32::MAX as f32 - 0.5
        };
        let f: Vec<[f32; 2]> = (0..200)
            .map(|_| [noise() * 4.0, noise() * 2.0 + 1.0])
            .collect();
        let x: Vec<Vec<f32>> = f
            .iter()
            .map(|r| {
                (0..t)
                    .map(|k| r[0] * truth[0][k] + r[1] * truth[1][k])
                    .collect()
            })
            .collect();
        let w = lstsq_basis(&f, &x).expect("independent features");
        for (got, want) in w.iter().zip(&truth) {
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() < 1e-3, "{g} vs {w}");
            }
        }
        // Collinear features have no unique basis.
        let same: Vec<[f32; 2]> = f.iter().map(|r| [r[0], 2.0 * r[0]]).collect();
        assert!(lstsq_basis(&same, &x).is_none());
    }

    #[test]
    fn set_channels_dedupes_caps_and_pads() {
        assert_eq!(
            set_channels(None, || vec![4, 4, 9, 1, 2, 3, 8]),
            vec![4, 9, 1, 2]
        );
        assert_eq!(
            set_channels(Some((0..20).collect()), Vec::new).len(),
            MAX_SET
        );
        assert_eq!(set_channels(Some(vec![7]), Vec::new), vec![7, 7]);
        assert_eq!(set_channels(None, Vec::new), vec![0, 0]);
    }

    #[test]
    fn brush_is_written_only_when_its_key_changes() {
        let a = Some((BrushSource::FeatureLasso, 1));
        let mut last = None;
        assert_eq!(brush_action(&mut last, None), BrushAction::Keep);
        assert_eq!(brush_action(&mut last, a), BrushAction::Set);
        assert_eq!(
            brush_action(&mut last, a),
            BrushAction::Keep,
            "same lasso, same brush"
        );
        let b = Some((BrushSource::FeatureLasso, 2));
        assert_eq!(
            brush_action(&mut last, b),
            BrushAction::Set,
            "an edited lasso rewrites"
        );
        let c = Some((BrushSource::KMeansPart, 2));
        assert_eq!(
            brush_action(&mut last, c),
            BrushAction::Set,
            "another source rewrites"
        );
        assert_eq!(brush_action(&mut last, None), BrushAction::Clear);
        assert_eq!(brush_action(&mut last, None), BrushAction::Keep);
    }

    #[test]
    fn nearest_spike_picks_the_closest_within_reach() {
        let pts = || {
            [
                (7, Pos2::new(10.0, 10.0)),
                (8, Pos2::new(13.0, 10.0)),
                (9, Pos2::new(50.0, 50.0)),
            ]
            .into_iter()
        };
        assert_eq!(
            nearest_spike(pts(), Pos2::new(12.0, 10.0), HOVER_PX),
            Some(8)
        );
        assert_eq!(
            nearest_spike(pts(), Pos2::new(48.0, 52.0), HOVER_PX),
            Some(9)
        );
        assert_eq!(nearest_spike(pts(), Pos2::new(30.0, 30.0), HOVER_PX), None);
        // Exactly on the edge still hits; just outside does not.
        assert_eq!(
            nearest_spike(pts(), Pos2::new(56.0, 50.0), HOVER_PX),
            Some(9)
        );
        assert_eq!(nearest_spike(pts(), Pos2::new(56.1, 50.0), HOVER_PX), None);
    }

    #[test]
    fn time_brush_fades_or_hides_spikes_outside() {
        assert_eq!(time_alpha(None, 5.0), Some(1.0));
        let band = TimeBrush::new(2.0, 4.0).unwrap();
        assert_eq!(time_alpha(Some(band), 3.0), Some(1.0));
        assert_eq!(time_alpha(Some(band), 9.0), Some(OUTSIDE_ALPHA));
        let only = TimeBrush { only: true, ..band };
        assert_eq!(time_alpha(Some(only), 3.0), Some(1.0));
        assert_eq!(time_alpha(Some(only), 9.0), None);
    }

    #[test]
    fn colour_ramp_runs_viridis_end_to_end() {
        assert_eq!(ramp_colour(0.0, 0.0, 300.0), viridis(0.0));
        assert_eq!(ramp_colour(300.0, 0.0, 300.0), viridis(1.0));
        assert_eq!(ramp_colour(0.0, 0.0, 300.0), Color32::from_rgb(68, 1, 84));
        assert_eq!(
            ramp_colour(300.0, 0.0, 300.0),
            Color32::from_rgb(253, 231, 37)
        );
        assert_eq!(ramp_colour(-50.0, 0.0, 300.0), viridis(0.0), "clamps below");
        assert_eq!(
            ramp_colour(7.0, 7.0, 7.0),
            viridis(0.5),
            "an empty range is the middle"
        );
    }

    #[test]
    fn trace_window_is_a_range_of_local_indices() {
        let times: Vec<SampleIndex> = [10, 20, 30, 40, 50].map(SampleIndex).to_vec();
        assert_eq!(window_locals(&times, 20, 20), 1..3, "start in, end out");
        assert_eq!(window_locals(&times, 0, 5), 0..0);
        assert_eq!(window_locals(&times, 0, 1000), 0..5);
    }
}

//! Features 3-D: the Features pane's point cloud in three of the channel
//! set's dimensions (time, then PC1 and PC2 on every channel), behind an
//! orbit camera.
//!
//! Inspection only: no selection tools here. The view shows
//! * every axis triple of the set (680 for eight channels), ranked by how
//!   well it separates the selected clusters (or the k-means parts, or just
//!   their spread); the focused 2-D plot's pair is pinned by default, which
//!   leaves only the third axis to choose;
//! * a convex hull per selected cluster: filled for sorter-made (`Auto`)
//!   units, wireframe for Sorrel-made ones;
//! * the pre-split parent of a split child as a dashed grey hull;
//! * brushes: brushed spikes in ink, the hovered spike ringed (and picked
//!   from here), spikes outside the time brush faded; a merge candidate
//!   previewed from Similar and a history node's "before #n" clusters as
//!   dashed grey hulls;
//! * a space family of axes (spike depth) when the backend has depths;
//! * the closed lasso extruded into a prism, and the shown k-means part's hull.
//!
//! Rendering is [`crate::wgpu_scene3d`]; this module only builds scenes.

use std::collections::{BTreeSet, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use eframe::egui_wgpu;
use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Ui, UiBuilder, Vec2};
use rayon::prelude::*;
use sorrel_compute::convex_hull_3d;
use sorrel_data::{Session, SpikeDepths, UnitOrigin};
use sorrel_io::{ClusterId, DataProvider};

use crate::brush::{Brushes, Ghost, HoverSpike};
use crate::feature_view::{
    compute, dim_label, labels_for, score_rows, separation, Dim, FeatureKey, FeatureSet, Labels,
    BACKGROUND,
};
use crate::jobs::{JobCtx, Memo};
use crate::theme::{selection_colour, viridis, Theme};
use crate::wgpu_raster::pack_rgba;
use crate::wgpu_scene3d::{Orbit, Point3, Scene3d, Scene3dCallback, Vert3};

/// Three dimensions of the set (indices into [`FeatureSet::dim`], ascending).
type Triple = [usize; 3];
/// A triple and its separation score (`NaN` when cancelled).
type Scored = (Triple, f32);

/// Triples the sidebar and the compact combo list before "show all".
const TOP: usize = 20;

const SIDEBAR_W: f32 = 240.0;
/// Disc radii in logical pixels.
const POINT_R: f32 = 2.5;
const BACKGROUND_R: f32 = 1.0;
/// Brushed spikes are drawn a little bigger, in their own scene on top.
const INK_R: f32 = 3.5;
/// A hovered spike is picked within this many pixels of the pointer.
const PICK_PX: f32 = 6.0;
/// Id of the pseudo channel carrying the space axes (no real channel has it).
const SPACE_CH: u32 = u32::MAX;

/// How the points are coloured instead of by cluster (Features colour mode).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PointColour {
    Time,
    Depth,
}

/// What `feature_view` knows besides the point set.
pub(crate) struct Overlay<'a> {
    /// Indices into `set.points` of the k-means part shown (exact members), if any.
    pub km_part: Option<&'a [usize]>,
    /// The closed lasso: its cell's (x, y) dims and polygon in data coords.
    pub lasso: Option<(Dim, Dim, &'a [[f32; 2]])>,
    /// All k-means parts, for the separation score when < 2 clusters are selected.
    pub km_parts: Option<&'a [Vec<usize>]>,
    /// The focused 2-D cell's (x, y) dims: what the pin holds.
    pub focus: (Dim, Dim),
}

/// What the triple scores tell apart.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Source {
    Clusters,
    Parts,
    Variance,
}

type ScoreKey = (u64, Source, u64);

/// One selected cluster as the scene draws it.
struct Style {
    colour: Color32,
    /// `Auto` origin: filled hull; otherwise wireframe.
    filled: bool,
}

/// Legend swatch shapes, matching what the scene draws.
#[derive(Copy, Clone)]
enum Swatch {
    Fill(Color32),
    Wire(Color32),
    /// Long-dash outline (the parent ghost).
    Dashed(Color32),
    /// Solid caps over dashed verticals (the lasso prism).
    Prism(Color32),
    /// Dotted outline with a light fill (the k-means part hull).
    Dotted(Color32),
    /// A filled dot (brushed spikes).
    Dot(Color32),
    /// A ring (the hovered spike).
    Ring(Color32),
    /// The viridis ramp (time / depth colouring).
    Ramp,
}

#[derive(Default)]
pub(crate) struct View3dState {
    orbit: Orbit,
    /// Aspect ratio the camera distance was last fitted for (0: never).
    fitted: f32,
    /// The triple the user picked (its dims, so it survives re-ranking); it
    /// applies while it is listed, else the best listed triple does.
    chosen: Option<Triple>,
    /// The triple on screen as of the last frame.
    shown: Option<Triple>,
    /// The pin is off (it is on whenever the view is entered).
    unpinned: bool,
    /// The sidebar lists every triple, not the top [`TOP`].
    show_all: bool,
    /// Frame this view was last drawn in, to tell re-entering from drawing on.
    seen: u64,
    labels: Option<(ScoreKey, Arc<Vec<u32>>)>,
    /// Every triple of the set, best first.
    scores: Memo<ScoreKey, Arc<Vec<Scored>>>,
    /// Pre-split parent hulls, one job per parent.
    ghosts: HashMap<ClusterId, Memo<FeatureKey, Arc<FeatureSet>>>,
    /// The merge candidate being previewed (Similar / Suggest hover).
    merge_ghost: Memo<FeatureKey, Arc<FeatureSet>>,
    /// The set with the space axes, for `(set generation, depths identity)`.
    space: Option<((u64, usize), Arc<FeatureSet>)>,
    /// Channels of the triple on screen as of the last frame.
    shown_channels: Vec<u32>,
    /// The spike under the pointer: what it was computed for, and the point.
    pick: Option<(PickKey, Option<usize>)>,
    scene: Option<Built>,
}

/// The job loading a ghost's features.
fn ghost_job<P: DataProvider>(
    gkey: FeatureKey,
) -> impl FnOnce(&Session<P>, &AtomicBool) -> Arc<FeatureSet> + Send + 'static {
    move |s, cancel| {
        let basis = Arc::new(Mutex::new(None));
        Arc::new(compute(s, &gkey, &basis, cancel))
    }
}

/// What a hover pick depends on; it is recomputed only when one changes.
type PickKey = (Pos2, Orbit, Rect, u64);

struct Built {
    key: u64,
    generation: u64,
    fg: Arc<Scene3d>,
    bg: Arc<Scene3d>,
    /// Brushed spikes, drawn last and a little bigger.
    ink: Arc<Scene3d>,
}

static GENERATION: AtomicU64 = AtomicU64::new(1);

impl View3dState {
    /// Channels of the triple on screen, for Waveforms to highlight.
    pub(crate) fn triple_channels(&self, set: &FeatureSet) -> Vec<u32> {
        let mine = |c: &&u32| set.channels.contains(c);
        let shown = self.shown.is_some();
        if shown {
            self.shown_channels.iter().filter(mine).copied().collect()
        } else {
            Vec::new()
        }
    }
}

/// `set` with a pseudo channel last whose PC1 is spike depth (µm) and whose PC2
/// is unused (NaN: no provider exposes spike x positions yet), so the space
/// axes are ordinary dims of the triples. `selection` maps the points' slots
/// to clusters. Background points have no depth (NaN): they drop out of any
/// triple holding it.
// ponytail: depth only; a spike_positions x would fill the PC2 column and
// `dead_dim` goes away.
fn with_space(set: &FeatureSet, selection: &[ClusterId], depths: &SpikeDepths) -> FeatureSet {
    let n = set.n();
    let mut feats = Vec::with_capacity(set.points.len() * 2 * (n + 1));
    for (i, p) in set.points.iter().enumerate() {
        let depth = selection
            .get(p.slot as usize)
            .and_then(|c| depths.per_cluster.get(c.idx()))
            .and_then(|d| d.get(p.local as usize))
            .copied()
            .unwrap_or(f32::NAN);
        let row = set.row(i);
        feats.extend_from_slice(&row[..n]);
        feats.push(depth);
        feats.extend_from_slice(&row[n..]);
        feats.push(f32::NAN);
    }
    let channels = set.channels.iter().copied().chain([SPACE_CH]).collect();
    FeatureSet::new(
        channels,
        set.computed,
        set.full,
        set.points.clone(),
        feats,
        set.duration,
    )
}

/// The space dims of `set`: `(depth, unused x)`.
fn space_dims(set: &FeatureSet) -> Option<(Dim, Dim)> {
    let ch = (set.channels.last() == Some(&SPACE_CH)).then(|| (set.n() - 1) as u8)?;
    Some((Dim::Pc { ch, pc: 0 }, Dim::Pc { ch, pc: 1 }))
}

/// Axis name: the space axes by name, the rest as in the 2-D plots.
fn label(dim: Dim, set: &FeatureSet) -> String {
    match (dim, space_dims(set)) {
        (d, Some((depth, _))) if d == depth => "depth".into(),
        (d, Some((_, x))) if d == x => "x".into(),
        _ => dim_label(dim, set),
    }
}

/// The channels of `t`'s PC dims, each once (time has none).
fn channels_of(set: &FeatureSet, t: Triple) -> Vec<u32> {
    let mut out = Vec::new();
    for d in t {
        if let Dim::Pc { ch, .. } = set.dim(d) {
            let c = set.channels[ch as usize];
            if c != SPACE_CH && !out.contains(&c) {
                out.push(c);
            }
        }
    }
    out
}

/// Point `i` of `from` in `[0, 1]^3` (each dim scaled by `set`'s bounds, time
/// by the duration); `None` when any coordinate is not finite.
fn unit_pos(set: &FeatureSet, from: &FeatureSet, t: Triple, i: usize) -> Option<[f32; 3]> {
    let mut out = [0.0; 3];
    for (o, &d) in out.iter_mut().zip(&t) {
        let dim = set.dim(d);
        *o = set.unit(dim, from.value(i, dim));
        if !o.is_finite() {
            return None;
        }
    }
    Some(out)
}

/// Scene position in `[-1, 1]^3`.
fn scene_pos(set: &FeatureSet, from: &FeatureSet, t: Triple, i: usize) -> Option<[f32; 3]> {
    unit_pos(set, from, t, i).map(|u| u.map(|v| v * 2.0 - 1.0))
}

/// The triples of `set`: every choice of three of its dims except those with
/// a dim that holds no data (the space family's x).
fn set_triples(set: &FeatureSet) -> Vec<Triple> {
    let dead = space_dims(set).map(|(_, x)| set.dim_idx(x));
    let mut all = triples(set.n_dims());
    all.retain(|t| dead.is_none_or(|d| !t.contains(&d)));
    all
}

/// Every choice of three of `n_dims` dimensions, ascending.
fn triples(n_dims: usize) -> Vec<Triple> {
    let mut out = Vec::new();
    for a in 0..n_dims {
        for b in a + 1..n_dims {
            out.extend((b + 1..n_dims).map(|c| [a, b, c]));
        }
    }
    out
}

/// Every triple of `set` with its score on at most `MAX_ROWS` labelled
/// points, best first (cancelled or unscored last).
fn score_triples(set: &FeatureSet, labels: &[u32], cancel: &AtomicBool) -> Vec<Scored> {
    let rows = score_rows(labels);
    let mut v: Vec<Scored> = set_triples(set)
        .into_par_iter()
        .map(|t| {
            if cancel.load(Ordering::Relaxed) {
                return (t, f32::NAN);
            }
            (t, separation(set, &t, &rows, labels))
        })
        .collect();
    v.sort_by(|a, b| key(b.1).total_cmp(&key(a.1)));
    v
}

fn key(score: f32) -> f32 {
    if score.is_nan() {
        f32::NEG_INFINITY
    } else {
        score
    }
}

/// The triples of `all` that exist in an `n_dims`-dimensional set and, with a
/// `pair`, hold both of its dims.
fn listed(all: &[Scored], n_dims: usize, pair: Option<[usize; 2]>) -> Vec<Scored> {
    all.iter()
        .copied()
        .filter(|(t, _)| t[2] < n_dims && pair.is_none_or(|p| p.iter().all(|d| t.contains(d))))
        .collect()
}

/// The triple shown: the user's pick while it is listed, else the best one.
fn effective(list: &[Scored], chosen: Option<Triple>) -> Triple {
    chosen
        .filter(|c| list.iter().any(|(t, _)| t == c))
        .unwrap_or(list[0].0)
}

fn source_labels(set: &FeatureSet, source: Source, parts: Option<&[Vec<usize>]>) -> Vec<u32> {
    labels_for(
        set,
        &match source {
            Source::Clusters => Labels::Clusters,
            Source::Parts => Labels::Parts(parts.unwrap_or_default()),
            Source::Variance => Labels::Variance,
        },
    )
}

/// Points a hull is built from at most (before the 95% trim).
const HULL_SAMPLE: usize = 2500;

/// The 95% of `pts` nearest their per-axis median, axes scaled by their
/// spread (the 3-D twin of `auto_lasso`'s trimming).
fn trimmed(pts: Vec<[f32; 3]>) -> Vec<[f32; 3]> {
    // `convex_hull_3d` is O(n·h) and runs on the UI thread: an even sample of
    // a large unit (pc_features.npy keeps every spike) bounds it.
    let stride = pts.len().div_ceil(HULL_SAMPLE).max(1);
    let mut pts: Vec<[f32; 3]> = pts.into_iter().step_by(stride).collect();
    if pts.len() < 4 {
        return pts; // no hull anyway; the median of nothing is undefined
    }
    let med: [f32; 3] = std::array::from_fn(|a| {
        let mut v: Vec<f32> = pts.iter().map(|p| p[a]).collect();
        let m = v.len() / 2;
        v.select_nth_unstable_by(m, f32::total_cmp).1.to_owned()
    });
    let scale: [f32; 3] = std::array::from_fn(|a| {
        pts.iter()
            .fold(0.0f32, |m, p| m.max((p[a] - med[a]).abs()))
            .max(f32::MIN_POSITIVE)
    });
    let dist = |p: &[f32; 3]| {
        (0..3)
            .map(|a| ((p[a] - med[a]) / scale[a]).powi(2))
            .sum::<f32>()
    };
    pts.sort_by(|a, b| dist(a).total_cmp(&dist(b)));
    pts.truncate((pts.len() * 95).div_ceil(100).max(4));
    pts
}

/// Brush ink: white in dark, near-black in light.
fn ink_colour(dark: bool) -> Color32 {
    if dark {
        Color32::WHITE
    } else {
        Color32::from_gray(20)
    }
}

fn rgba(c: Color32, alpha: f32) -> u32 {
    let [r, g, b, _] = c.to_array();
    pack_rgba(r, g, b, (alpha.clamp(0.0, 1.0) * 255.0).round() as u8)
}

/// Dash pattern `(on, off)` in scene units; `None` is a solid line.
type Dash = Option<(f32, f32)>;
const DASHED: Dash = Some((0.07, 0.05));
const DOTTED: Dash = Some((0.02, 0.045));

fn seg(out: &mut Vec<Vert3>, a: [f32; 3], b: [f32; 3], color: u32, dash: Dash) {
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let at = |s: f32| -> Vert3 {
        let f = (s / len.max(1e-9)).min(1.0);
        Vert3 {
            pos: [a[0] + d[0] * f, a[1] + d[1] * f, a[2] + d[2] * f],
            color,
        }
    };
    match dash {
        Some((on, off)) if len > on => {
            let mut s = 0.0;
            while s < len {
                out.extend([at(s), at(s + on)]);
                s += on + off;
            }
        }
        _ => out.extend([at(0.0), at(len)]),
    }
}

/// Hull of `pts`: (optionally) its unique edges into `lines` and its faces
/// into `tris`. Returns whether the hull exists (not degenerate).
fn push_hull(
    pts: &[[f32; 3]],
    lines: &mut Vec<Vert3>,
    tris: &mut Vec<Vert3>,
    edge: Option<u32>,
    dash: Dash,
    fill: Option<u32>,
) -> bool {
    let faces = convex_hull_3d(pts);
    if faces.is_empty() {
        return false;
    }
    let mut edges = BTreeSet::new();
    for f in &faces {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            edges.insert((a.min(b), a.max(b)));
        }
        if let Some(color) = fill {
            tris.extend(f.map(|i| Vert3 {
                pos: pts[i as usize],
                color,
            }));
        }
    }
    for (a, b) in edges.into_iter().filter(|_| edge.is_some()) {
        seg(
            lines,
            pts[a as usize],
            pts[b as usize],
            edge.unwrap_or(0),
            dash,
        );
    }
    true
}

/// Where the lasso's x/y dims sit in `triple`, with the leftover axis:
/// `(x axis, y axis, third axis)`. `None` unless both dims are in it.
fn prism_axes(set: &FeatureSet, triple: Triple, dx: Dim, dy: Dim) -> Option<[usize; 3]> {
    let pos = |d: Dim| triple.iter().position(|&t| t == set.dim_idx(d));
    let (x, y) = (pos(dx)?, pos(dy)?);
    (x != y).then(|| [x, y, 3 - x - y])
}

/// The lasso extruded over the whole third axis: both caps and the sides.
fn prism(
    set: &FeatureSet,
    triple: Triple,
    (dx, dy, poly): (Dim, Dim, &[[f32; 2]]),
    color: u32,
    out: &mut Vec<Vert3>,
) -> bool {
    let Some([ax, ay, az]) = prism_axes(set, triple, dx, dy) else {
        return false;
    };
    if poly.len() < 3 {
        return false;
    }
    let at = |p: [f32; 2], z: f32| {
        let mut v = [0.0; 3];
        v[ax] = set.unit(dx, p[0]) * 2.0 - 1.0;
        v[ay] = set.unit(dy, p[1]) * 2.0 - 1.0;
        v[az] = z;
        v
    };
    for (i, &p) in poly.iter().enumerate() {
        let q = poly[(i + 1) % poly.len()];
        seg(out, at(p, -1.0), at(q, -1.0), color, None);
        seg(out, at(p, 1.0), at(q, 1.0), color, None);
        seg(out, at(p, -1.0), at(p, 1.0), color, DASHED);
    }
    true
}

fn cube(out: &mut Vec<Vert3>, color: u32) {
    for a in 0..3 {
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        for (sb, sc) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let mut lo = [0.0; 3];
            lo[a] = -1.0;
            lo[b] = sb;
            lo[c] = sc;
            let mut hi = lo;
            hi[a] = 1.0;
            out.extend([lo, hi].map(|pos| Vert3 { pos, color }));
        }
    }
}

/// Colours the scene needs from the theme.
struct Palette {
    cube: Color32,
    ghost: Color32,
    strong: Color32,
    accent: Color32,
    /// Members of the shown k-means part (as in the 2-D preview).
    part: Color32,
    dark: bool,
}

/// What the brushes ask of the scene.
#[derive(Copy, Clone, Default)]
struct BrushInput<'a> {
    /// The spike brush and the clusters its local indices belong to (by slot).
    spikes: Option<(&'a crate::brush::SpikeBrush, &'a [ClusterId])>,
    time: Option<crate::brush::TimeBrush>,
    colour: Option<PointColour>,
    /// Point indices of each cluster as it was before the history node.
    before: &'a [Vec<usize>],
}

struct SceneInput<'a> {
    set: &'a FeatureSet,
    triple: Triple,
    styles: &'a [Style],
    ghosts: &'a [Arc<FeatureSet>],
    overlay: &'a Overlay<'a>,
    palette: &'a Palette,
    brush: BrushInput<'a>,
}

/// Alpha of a spike outside the time brush.
const FADED: f32 = 0.2;

/// `(foreground, background, ink)` scenes. Background and brushed spikes
/// are scenes of their own because discs share one radius per scene.
// ponytail: the background layer is composited under the foreground, so
// a background disc never occludes a foreground one in depth; at 1 px and
// alpha 0.45 nobody can tell. Give `Point3` its own size to merge them.
fn build_scene(inp: &SceneInput) -> (Scene3d, Scene3d, Scene3d) {
    let SceneInput {
        set,
        triple,
        styles,
        ghosts,
        overlay,
        palette,
        brush,
    } = *inp;
    let depth_dim = space_dims(set).map(|d| d.0);
    let ramp = |p: usize| match brush.colour {
        Some(PointColour::Time) => Some(set.unit(Dim::Time, set.points[p].t)),
        Some(PointColour::Depth) => depth_dim.map(|d| set.unit(d, set.value(p, d))),
        None => None,
    };
    let mut members = vec![false; set.points.len()];
    for &i in overlay.km_part.unwrap_or_default() {
        if let Some(m) = members.get_mut(i) {
            *m = true;
        }
    }
    let (mut points, mut background, mut ink) = (Vec::new(), Vec::new(), Vec::new());
    let mut per_slot: Vec<Vec<[f32; 3]>> = vec![Vec::new(); styles.len()];
    let mut part_pts = Vec::new();
    for (i, p) in set.points.iter().enumerate() {
        let Some(pos) = scene_pos(set, set, triple, i) else {
            continue;
        };
        // Outside the time brush: faded, or gone with "only in range".
        let (shown, fade) = match brush.time {
            Some(t) if !t.contains(p.t) => (!t.only, FADED),
            _ => (true, 1.0),
        };
        if p.slot == BACKGROUND {
            if shown {
                background.push(Point3 {
                    pos,
                    color: rgba(Color32::GRAY, 0.45 * fade),
                });
            }
            continue;
        }
        let Some(style) = styles.get(p.slot as usize) else {
            continue;
        };
        per_slot[p.slot as usize].push(pos);
        if !shown {
            continue;
        }
        let brushed = brush.spikes.is_some_and(|(b, cs)| {
            cs.get(p.slot as usize)
                .is_some_and(|&c| b.contains(c, p.local))
        });
        if brushed {
            // The part's hull is independent of the ink layer.
            if members[i] {
                part_pts.push(pos);
            }
            ink.push(Point3 {
                pos,
                color: rgba(palette.part, 240.0 / 255.0 * fade),
            });
            continue;
        }
        let color = if members[i] {
            part_pts.push(pos);
            rgba(palette.part, fade)
        } else if let Some(t) = ramp(i).filter(|t| t.is_finite()) {
            rgba(viridis(t.clamp(0.0, 1.0)), 0.9 * fade)
        } else {
            rgba(style.colour, 0.9 * fade)
        };
        points.push(Point3 { pos, color });
    }

    let (mut lines, mut tris) = (Vec::new(), Vec::new());
    cube(&mut lines, rgba(palette.cube, 0.9));
    for (style, pts) in styles.iter().zip(per_slot) {
        // Auto: a translucent solid; Sorrel-made: every hull edge (wireframe).
        let (edge, fill) = if style.filled {
            (None, 0.2)
        } else {
            (Some(rgba(style.colour, 0.9)), 0.05)
        };
        push_hull(
            &trimmed(pts),
            &mut lines,
            &mut tris,
            edge,
            None,
            Some(rgba(style.colour, fill)),
        );
    }
    for g in ghosts {
        let pts: Vec<[f32; 3]> = (0..g.points.len())
            .filter(|&i| g.points[i].slot != BACKGROUND)
            .filter_map(|i| scene_pos(set, g, triple, i))
            .collect();
        push_hull(
            &trimmed(pts),
            &mut lines,
            &mut Vec::new(),
            Some(rgba(palette.ghost, 0.85)),
            DASHED,
            None,
        );
    }
    // The history node's clusters as they were: grey dashed hulls over the
    // shown spikes.
    for idx in brush.before {
        let pts: Vec<[f32; 3]> = idx
            .iter()
            .filter_map(|&i| scene_pos(set, set, triple, i))
            .collect();
        push_hull(
            &trimmed(pts),
            &mut lines,
            &mut Vec::new(),
            Some(rgba(palette.ghost, 0.85)),
            DASHED,
            None,
        );
    }
    if let Some(l) = overlay.lasso {
        prism(set, triple, l, rgba(palette.strong, 0.95), &mut lines);
    }
    if part_pts.len() >= 4 {
        let accent = rgba(palette.accent, 0.95);
        let fill = Some(rgba(palette.accent, if palette.dark { 0.10 } else { 0.12 }));
        push_hull(&part_pts, &mut lines, &mut tris, Some(accent), DOTTED, fill);
    }
    let scene = |points: Vec<Point3>, lines: Vec<Vert3>, tris: Vec<Vert3>| Scene3d {
        points: points.into(),
        lines: lines.into(),
        tris: tris.into(),
    };
    (
        scene(points, lines, tris),
        scene(background, Vec::new(), Vec::new()),
        scene(ink, Vec::new(), Vec::new()),
    )
}

fn hash_overlay(h: &mut impl Hasher, overlay: &Overlay, set: &FeatureSet) {
    overlay.km_part.hash(h);
    if let Some((dx, dy, poly)) = overlay.lasso {
        (set.dim_idx(dx), set.dim_idx(dy)).hash(h);
        poly.iter().flatten().for_each(|v| v.to_bits().hash(h));
    }
}

/// What of the brushes the scene shows: the spike brush's identity and the
/// time range.
fn hash_brushes(h: &mut impl Hasher, brushes: &Brushes) {
    // `BrushSource` is not `Hash`; its discriminant stands in.
    brushes
        .spikes
        .as_ref()
        .map(|b| b.key())
        .map(|(src, v, n, ptr)| (std::mem::discriminant(&src), v, n, ptr))
        .hash(h);
    brushes
        .time
        .map(|t| (t.start.to_bits(), t.end.to_bits(), t.only))
        .hash(h);
}

/// Camera distance at which the cube's bounding sphere fits the narrower
/// side of a canvas with this aspect ratio.
fn fit_distance(aspect: f32) -> f32 {
    let half = (std::f32::consts::FRAC_PI_8.tan() * aspect.min(1.0)).atan();
    1.65 / half.sin()
}

fn triple_label(set: &FeatureSet, t: Triple) -> String {
    t.map(|d| label(set.dim(d), set)).join(" · ")
}

/// `3·1 5·1 3·2`: channel and PC number, `t` for time.
fn short_label(set: &FeatureSet, t: Triple) -> String {
    t.map(|d| match set.dim(d) {
        Dim::Time => "t".to_owned(),
        _ if space_dims(set).is_some_and(|s| s.0 == set.dim(d)) => "depth".to_owned(),
        Dim::Pc { ch, pc } => format!("{}·{}", set.channels[ch as usize], pc + 1),
    })
    .join("  ")
}

fn source_label(source: Source, selection: &[ClusterId]) -> String {
    match source {
        Source::Clusters if selection.len() == 2 => {
            format!("silhouette c{} vs c{}", selection[0].0, selection[1].0)
        }
        Source::Clusters => format!("silhouette of {} clusters", selection.len()),
        Source::Parts => "silhouette of k-means parts".into(),
        Source::Variance => "variance".into(),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn feature_3d<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    set: &Arc<FeatureSet>,
    key: &FeatureKey,
    state: &mut View3dState,
    jobs: &JobCtx<Session<P>>,
    overlay: Overlay<'_>,
    compact: bool,
    brushes: &mut Brushes,
    depths: Option<&SpikeDepths>,
    colour: Option<PointColour>,
) {
    let theme = Theme::from_ui(ui);
    let visuals = ui.visuals().clone();
    // Coming back to this view (not drawing on) pins the focused plot again.
    let frame = ui.ctx().cumulative_frame_nr();
    if frame > state.seen + 1 {
        state.unpinned = false;
    }
    state.seen = frame;

    // The set with the space axes (cached per set and depths).
    let orig = set;
    let space_key = (set.generation, depths.map_or(0, |d| d as *const _ as usize));
    if depths.is_none() {
        state.space = None;
    } else if !matches!(&state.space, Some((k, _)) if *k == space_key) {
        let ext = depths.map(|d| Arc::new(with_space(set, selection, d)));
        state.space = ext.map(|e| (space_key, e));
    }
    let set = state.space.as_ref().map_or(orig, |(_, e)| e).clone();
    let set = &set;

    // Separation scores, per label source.
    let parts = overlay
        .km_parts
        .filter(|p| p.iter().filter(|x| !x.is_empty()).count() >= 2);
    let source = if selection.len() >= 2 {
        Source::Clusters
    } else if parts.is_some() {
        Source::Parts
    } else {
        Source::Variance
    };
    let mut h = DefaultHasher::new();
    if source == Source::Parts {
        parts.hash(&mut h);
    }
    let skey: ScoreKey = (set.generation, source, h.finish());
    if !matches!(&state.labels, Some((k, _)) if *k == skey) {
        let labels = source_labels(set, source, parts);
        state.labels = Some((skey, Arc::new(labels)));
    }
    let labels = state.labels.as_ref().map(|(_, l)| Arc::clone(l));
    let job_set = Arc::clone(set);
    let scores = state
        .scores
        .ensure(skey, &jobs.session, jobs.repaint(), move |_, cancel| {
            let labels = labels.unwrap_or_default();
            Arc::new(score_triples(&job_set, &labels, cancel))
        })
        .cloned()
        .or_else(|| state.scores.last().cloned());
    // Triples to choose from: the pin holds the focused plot's pair.
    let pair =
        (!state.unpinned).then(|| [overlay.focus.0, overlay.focus.1].map(|d| set.dim_idx(d)));
    let nd = set.n_dims();
    let mut list = scores
        .as_deref()
        .map_or_else(Vec::new, |s| listed(s, nd, pair));
    if list.is_empty() {
        let unscored: Vec<Scored> = set_triples(set)
            .into_iter()
            .map(|t| (t, f32::NAN))
            .collect();
        list = listed(&unscored, nd, pair);
    }
    let triple = effective(&list, state.chosen);
    state.shown = Some(triple);
    state.shown_channels = channels_of(set, triple);

    // Pre-split parents of the selected split children.
    let mut parents: Vec<(ClusterId, Vec<ClusterId>)> = Vec::new();
    for &c in selection {
        let &[parent] = session.parents(c) else {
            // ponytail: a merged unit has several parents; its pre-merge
            // hulls come from the History hover ("before #n"), not here.
            continue;
        };
        if matches!(session.origin(c), UnitOrigin::Auto) || parents.iter().any(|p| p.0 == parent) {
            continue;
        }
        // Before the split: the children and what the parent kept.
        let siblings = (0..session.n_clusters())
            .map(ClusterId)
            .filter(|&c2| {
                c2 == c
                    || session.parents(c2).contains(&parent)
                    || (c2 == parent && !session.spike_times(c2).is_empty())
            })
            .collect();
        parents.push((parent, siblings));
    }
    state
        .ghosts
        .retain(|p, _| parents.iter().any(|q| q.0 == *p));
    // The features of `clusters` on the set's channels, as a background job.
    let ghost_key = |clusters: Vec<ClusterId>| FeatureKey {
        selection: clusters,
        channels: Some(orig.channels.clone()),
        keep_order: true,
        ..key.clone()
    };
    // (clusters its slots index, what it is, the features)
    let mut ghosts: Vec<(Vec<ClusterId>, ClusterId, Arc<FeatureSet>)> = Vec::new();
    for (parent, siblings) in parents {
        let gkey = ghost_key(siblings.clone());
        let current = state
            .ghosts
            .entry(parent)
            .or_default()
            .ensure(
                gkey.clone(),
                &jobs.session,
                jobs.repaint(),
                ghost_job::<P>(gkey),
            )
            .cloned();
        ghosts.extend(current.map(|g| (siblings, parent, g)));
    }
    // The merge candidate previewed from Similar / Suggest.
    let merge_ghost = match brushes.ghost {
        Some(Ghost::Merge(c)) if !selection.contains(&c) => {
            let gkey = ghost_key(vec![c]);
            state
                .merge_ghost
                .ensure(
                    gkey.clone(),
                    &jobs.session,
                    jobs.repaint(),
                    ghost_job::<P>(gkey),
                )
                .cloned()
                .map(|g| (c, g))
        }
        _ => None,
    };
    // A history node's clusters before it: the shown points of each, by
    // global spike index.
    let history = match brushes.ghost {
        Some(Ghost::History(n)) => session.pre_edit_spikes(n).map(|b| (n, b)),
        _ => None,
    };
    let before: Vec<Vec<usize>> = history.as_ref().map_or_else(Vec::new, |(_, b)| {
        let mut at: HashMap<u32, usize> = HashMap::new();
        for (i, p) in set.points.iter().enumerate() {
            let Some(&c) = selection.get(p.slot as usize) else {
                continue; // background
            };
            if let Some(&g) = session
                .cluster_index()
                .global_indices(c)
                .get(p.local as usize)
            {
                at.insert(g, i);
            }
        }
        b.iter()
            .map(|(_, globals)| globals.iter().filter_map(|g| at.get(g).copied()).collect())
            .collect()
    });

    // Scene, rebuilt only when what it shows changes.
    let dark = theme.dark;
    let styles: Vec<Style> = selection
        .iter()
        .enumerate()
        .map(|(i, &c)| Style {
            colour: selection_colour(i),
            filled: matches!(session.origin(c), UnitOrigin::Auto),
        })
        .collect();
    let colour = colour.filter(|c| *c != PointColour::Depth || space_dims(set).is_some());
    let mut h = DefaultHasher::new();
    (set.generation, triple, dark, selection, colour).hash(&mut h);
    hash_overlay(&mut h, &overlay, set);
    ghosts
        .iter()
        .for_each(|(_, p, g)| (p, g.generation).hash(&mut h));
    merge_ghost
        .iter()
        .for_each(|(c, g)| (c, g.generation).hash(&mut h));
    history.as_ref().map(|(n, _)| n).hash(&mut h);
    hash_brushes(&mut h, brushes);
    let scene_key = h.finish();
    if state.scene.as_ref().is_none_or(|b| b.key != scene_key) {
        let palette = Palette {
            cube: if dark {
                theme.grid
            } else {
                Color32::from_gray(0x8C)
            },
            ghost: visuals.text_color(),
            strong: visuals.strong_text_color(),
            accent: visuals.hyperlink_color,
            part: ink_colour(dark),
            dark,
        };
        // Ghost features get the same space axes as the set.
        let with = |sel: &[ClusterId], g: &Arc<FeatureSet>| match depths {
            Some(d) => Arc::new(with_space(g, sel, d)),
            None => Arc::clone(g),
        };
        let sets: Vec<Arc<FeatureSet>> = ghosts
            .iter()
            .map(|(sel, _, g)| with(sel, g))
            .chain(merge_ghost.iter().map(|(c, g)| with(&[*c], g)))
            .collect();
        let (fg, bg, ink) = build_scene(&SceneInput {
            set,
            triple,
            styles: &styles,
            ghosts: &sets,
            overlay: &overlay,
            palette: &palette,
            brush: BrushInput {
                spikes: brushes.spikes.as_ref().map(|b| (b, selection)),
                time: brushes.time,
                colour,
                before: &before,
            },
        });
        state.scene = Some(Built {
            key: scene_key,
            generation: GENERATION.fetch_add(1, Ordering::Relaxed),
            fg: Arc::new(fg),
            bg: Arc::new(bg),
            ink: Arc::new(ink),
        });
    }

    let prism_missing = overlay
        .lasso
        .filter(|&(dx, dy, _)| prism_axes(set, triple, dx, dy).is_none());
    let mut legend: Vec<(Swatch, String)> = selection
        .iter()
        .zip(&styles)
        .map(|(&c, s)| {
            let text = format!("c{} · {}", c.0, session.origin_label(c));
            (
                if s.filled {
                    Swatch::Fill(s.colour)
                } else {
                    Swatch::Wire(s.colour)
                },
                text,
            )
        })
        .collect();
    if let Some((_, parent, _)) = ghosts.first() {
        legend.push((
            Swatch::Dashed(visuals.text_color()),
            format!("parent before split (c{})", parent.0),
        ));
    }
    if let Some((c, _)) = &merge_ghost {
        legend.push((
            Swatch::Dashed(visuals.text_color()),
            format!("ghost c{}", c.0),
        ));
    }
    if let Some((n, _)) = history.as_ref().filter(|_| !before.is_empty()) {
        legend.push((Swatch::Dashed(visuals.text_color()), format!("before #{n}")));
    }
    if let Some(b) = &brushes.spikes {
        legend.push((
            Swatch::Dot(ink_colour(dark)),
            format!("brush · {} spikes", b.len()),
        ));
    }
    if brushes.hover.is_some() {
        legend.push((
            Swatch::Ring(visuals.hyperlink_color),
            "hovered spike".into(),
        ));
    }
    if let Some(c) = colour {
        legend.push((
            Swatch::Ramp,
            match c {
                PointColour::Time => "points by time",
                PointColour::Depth => "points by depth",
            }
            .into(),
        ));
    }
    if let Some((dx, dy, _)) = overlay.lasso {
        let cell = format!("{} × {}", label(dx, set), label(dy, set));
        legend.push((
            Swatch::Prism(visuals.strong_text_color()),
            format!("lasso prism ({cell})"),
        ));
    }
    if overlay.km_part.is_some() {
        legend.push((
            Swatch::Dotted(visuals.hyperlink_color),
            "k-means part".into(),
        ));
    }

    let mut pick = None;
    let avail = ui.available_rect_before_wrap();
    let canvas = if compact {
        ui.horizontal(|ui| {
            let score = |t: Triple| list.iter().find(|s| s.0 == t).map_or(f32::NAN, |s| s.1);
            let text = |t: Triple, long: bool| {
                let label = match (long, pair) {
                    (true, Some(_)) => third_label(set, t, pair),
                    (true, None) => triple_label(set, t),
                    (false, _) => short_label(set, t),
                };
                if score(t).is_nan() {
                    label
                } else {
                    format!("{label}  ★{:.2}", score(t))
                }
            };
            egui::ComboBox::from_id_salt("feature-3d-axes")
                .selected_text(text(triple, false))
                .width((ui.available_width() - 48.0).max(60.0))
                .show_ui(ui, |ui| {
                    for &(t, _) in list.iter().take(TOP) {
                        if ui.selectable_label(t == triple, text(t, true)).clicked() {
                            pick = Some(t);
                        }
                    }
                });
            ui.menu_button("⚙", |ui| legend_ui(ui, &legend, &theme))
                .response
                .on_hover_text("Legend");
        });
        let top = ui.cursor().top() + 2.0;
        Rect::from_min_max(Pos2::new(avail.left(), top), avail.max)
    } else {
        let side = Rect::from_min_max(
            Pos2::new((avail.right() - SIDEBAR_W).max(avail.left()), avail.top()),
            avail.max,
        );
        ui.scope_builder(UiBuilder::new().max_rect(side), |ui| {
            let sidebar = Sidebar {
                set,
                list: &list,
                pair,
                chosen: triple,
                source,
                selection,
                theme: &theme,
            };
            pick = sidebar.show(ui, &mut state.show_all);
            ui.add_space(14.0);
            legend_ui(ui, &legend, &theme);
        });
        Rect::from_min_max(
            avail.min,
            Pos2::new((side.left() - 8.0).max(avail.left()), avail.bottom()),
        )
    };
    if pick.is_some() {
        state.chosen = pick;
    }
    if canvas.width() < 20.0 || canvas.height() < 20.0 {
        return;
    }

    let resp = ui.allocate_rect(canvas, Sense::click_and_drag());
    resp.widget_info(|| {
        egui::WidgetInfo::labeled(egui::WidgetType::Other, true, "feature 3d canvas")
    });
    state.orbit.interact(&resp);
    // Keep the whole cube in view as the pane resizes, scaling a zoom the
    // user set; a double-click refits from scratch.
    let aspect = canvas.width() / canvas.height();
    if resp.double_clicked() || state.fitted == 0.0 {
        state.orbit.distance = fit_distance(aspect);
    } else {
        state.orbit.distance *= fit_distance(aspect) / fit_distance(state.fitted);
    }
    state.fitted = aspect;
    let painter = ui.painter_at(canvas);
    painter.rect_filled(canvas, 0.0, theme.plot_bg);
    if let Some(built) = &state.scene {
        let ppp = ui.ctx().pixels_per_point();
        let view_proj = state.orbit.view_proj(canvas.width() / canvas.height());
        let size_px = [canvas.width(), canvas.height()].map(|s| (s * ppp).round().max(1.0) as u32);
        let id = ui.id().with("feature-3d").value();
        for (id, scene, radius) in [
            (id ^ 1, &built.bg, BACKGROUND_R),
            (id, &built.fg, POINT_R),
            (id ^ 2, &built.ink, INK_R),
        ] {
            painter.add(egui_wgpu::Callback::new_paint_callback(
                canvas,
                Scene3dCallback {
                    id,
                    scene: Arc::clone(scene),
                    generation: built.generation,
                    view_proj,
                    point_radius_px: radius * ppp,
                    size_px,
                },
            ));
        }
    }
    paint_labels(&painter, canvas, &state.orbit, set, triple, &theme, compact);

    // Hover: the spike under the pointer is written for the other views; the
    // ring is drawn on the spike hovered here, else the one hovered elsewhere.
    let faded_out = |t: f32| brushes.time.is_some_and(|b| b.only && !b.contains(t));
    let hovered = if resp.hovered() && !resp.dragged() {
        resp.hover_pos().and_then(|at| {
            let k: PickKey = (at, state.orbit, canvas, scene_key);
            if !matches!(&state.pick, Some((old, _)) if *old == k) {
                // ponytail: O(n) projection of every shown point, run only
                // when the pointer, camera or scene changed; a screen-space
                // grid would make it O(1) beyond ~50k points.
                let found = pick_nearest(
                    set.points.iter().enumerate().filter_map(|(i, p)| {
                        let at = (p.slot != BACKGROUND && !faded_out(p.t))
                            .then(|| scene_pos(set, set, triple, i))??;
                        Some((i, state.orbit.project(at, canvas)?))
                    }),
                    at,
                );
                state.pick = Some((k, found));
            }
            state.pick.as_ref().and_then(|(_, f)| *f)
        })
    } else {
        None
    };
    if resp.hovered() {
        brushes.hover_next = hovered.and_then(|i| {
            let p = set.points.get(i)?;
            let cluster = *selection.get(p.slot as usize)?;
            let sample = session.spike_times(cluster).get(p.local as usize)?.0;
            Some(HoverSpike {
                cluster,
                local: p.local,
                sample,
            })
        });
    }
    let ring = if resp.hovered() {
        hovered
    } else {
        brushes.hover.and_then(|h| {
            let slot = selection.iter().position(|&c| c == h.cluster)?;
            set.points
                .iter()
                .position(|p| p.slot as usize == slot && p.local == h.local)
        })
    };
    if let Some(at) = ring
        .and_then(|i| scene_pos(set, set, triple, i))
        .and_then(|pos| state.orbit.project(pos, canvas))
    {
        painter.circle_stroke(at, 6.0, Stroke::new(1.5_f32, visuals.hyperlink_color));
    }
    if let Some((dx, dy, _)) = prism_missing {
        let text = format!(
            "lasso in {} × {} — pick a triple with both axes to see its prism",
            label(dx, set),
            label(dy, set)
        );
        let galley = painter.layout(
            text,
            FontId::proportional(11.0),
            theme.caption,
            canvas.width() - 16.0,
        );
        painter.galley(
            canvas.left_top() + Vec2::new(8.0, 6.0),
            galley,
            theme.caption,
        );
    }
}

/// The candidate nearest `at` within [`PICK_PX`], by screen position.
fn pick_nearest(pts: impl Iterator<Item = (usize, Pos2)>, at: Pos2) -> Option<usize> {
    pts.map(|(i, p)| (i, p.distance(at)))
        .filter(|&(_, d)| d <= PICK_PX)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

/// Axis names at the cube's axis ends (clamped inside the canvas), the XYZ
/// gizmo bottom-left, and the gesture hint.
fn paint_labels(
    painter: &egui::Painter,
    canvas: Rect,
    orbit: &Orbit,
    set: &FeatureSet,
    triple: Triple,
    theme: &Theme,
    compact: bool,
) {
    let font = FontId::proportional(11.0);
    let tips = [[1.18, -1.0, -1.0], [-1.0, 1.18, -1.0], [-1.0, -1.0, 1.18]];
    for (tip, &d) in tips.iter().zip(&triple) {
        let Some(p) = orbit.project(*tip, canvas) else {
            continue;
        };
        let text = label(set.dim(d), set);
        let size = painter
            .layout_no_wrap(text.clone(), font.clone(), theme.caption)
            .size();
        let inner = canvas.shrink2(size / 2.0 + Vec2::splat(4.0));
        let at = Pos2::new(
            p.x.clamp(inner.left(), inner.right().max(inner.left())),
            p.y.clamp(inner.top(), inner.bottom().max(inner.top())),
        );
        painter.text(at, Align2::CENTER_CENTER, text, font.clone(), theme.caption);
    }
    let origin = canvas.left_bottom() + Vec2::new(24.0, -24.0);
    if let Some(c) = orbit.project(orbit.target, canvas) {
        let axes = [
            ([1.0, 0.0, 0.0], Color32::from_rgb(230, 80, 80), "x"),
            ([0.0, 1.0, 0.0], Color32::from_rgb(90, 190, 90), "y"),
            ([0.0, 0.0, 1.0], Color32::from_rgb(90, 140, 240), "z"),
        ];
        for (e, colour, name) in axes {
            let t = orbit.target;
            let Some(p) = orbit.project([t[0] + e[0], t[1] + e[1], t[2] + e[2]], canvas) else {
                continue;
            };
            let v = (p - c).normalized() * 14.0;
            painter.line_segment([origin, origin + v], Stroke::new(1.5_f32, colour));
            painter.text(
                origin + v * 1.35,
                Align2::CENTER_CENTER,
                name,
                FontId::proportional(9.0),
                colour,
            );
        }
    }
    if !compact {
        painter.text(
            canvas.left_bottom() + Vec2::new(48.0, -10.0),
            Align2::LEFT_BOTTOM,
            "drag orbit · Shift+drag pan · wheel zoom · double-click reset",
            FontId::proportional(10.0),
            theme.caption,
        );
    }
}

/// `+ <third axis>` for the triples of a pinned pair.
fn third_label(set: &FeatureSet, t: Triple, pair: Option<[usize; 2]>) -> String {
    let third = t.into_iter().find(|d| pair.is_none_or(|p| !p.contains(d)));
    format!("+ {}", label(set.dim(third.unwrap_or(t[2])), set))
}

/// The "pinned · x × y  ×" chip (or the button pinning the focused plot),
/// drawn in the Features toolbar next to the 2D | 3D toggle.
pub(crate) fn pin_chip(ui: &mut Ui, state: &mut View3dState, set: &FeatureSet, focus: (Dim, Dim)) {
    let cell = format!("{} × {}", label(focus.0, set), label(focus.1, set));
    {
        let (text, pinned) = if state.unpinned {
            (format!("pin {cell}"), false)
        } else {
            (format!("pinned · {cell}  ×"), true)
        };
        let mut button = egui::Button::new(text).corner_radius(10.0);
        if pinned {
            button = button.fill(ui.visuals().selection.bg_fill);
        }
        let resp = ui.add(button);
        if pinned {
            resp.widget_info(|| {
                egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "pin focused plot")
            });
        }
        let resp = resp.on_hover_text(if pinned {
            "Only triples holding the focused plot's two axes; × lists them all"
        } else {
            "List only the triples holding the focused plot's two axes"
        });
        if resp.clicked() {
            state.unpinned = pinned;
        }
    }
}

struct Sidebar<'a> {
    set: &'a FeatureSet,
    /// The triples to list, best first.
    list: &'a [Scored],
    /// The pinned pair: the list is the third-axis choice for it.
    pair: Option<[usize; 2]>,
    chosen: Triple,
    source: Source,
    selection: &'a [ClusterId],
    theme: &'a Theme,
}

impl Sidebar<'_> {
    /// The ranked triples (the top [`TOP`], or all in a scroll area when
    /// `show_all`); returns the row clicked this frame.
    fn show(&self, ui: &mut Ui, show_all: &mut bool) -> Option<Triple> {
        let title = if self.pair.is_some() {
            "THIRD AXIS · triples holding the pinned pair".to_owned()
        } else {
            format!(
                "AXES · by separation ({})",
                source_label(self.source, self.selection)
            )
        };
        ui.add(egui::Label::new(egui::RichText::new(title).small().weak()).truncate());
        let top = self
            .list
            .iter()
            .map(|s| s.1)
            .filter(|v| v.is_finite())
            .fold(0.0, f32::max);
        let more = self.list.len() > TOP;
        let mut pick = None;
        if *show_all && more {
            egui::ScrollArea::vertical()
                .max_height((ui.available_height() - 150.0).max(120.0))
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for s in self.list {
                        pick = self.row(ui, *s, top).or(pick);
                    }
                });
        } else {
            for s in self.list.iter().take(TOP) {
                pick = self.row(ui, *s, top).or(pick);
            }
        }
        if more {
            let text = if *show_all {
                format!("show top {TOP}")
            } else {
                format!("show all {}", self.list.len())
            };
            if ui.small_button(text).clicked() {
                *show_all = !*show_all;
            }
        }
        ui.add(
            egui::Label::new(
                egui::RichText::new("chosen triple's channels light up in Waveforms")
                    .small()
                    .weak(),
            )
            .truncate(),
        );
        pick
    }

    /// One triple's row; the triple when clicked.
    fn row(&self, ui: &mut Ui, (t, score): Scored, top: f32) -> Option<Triple> {
        let label = if self.pair.is_some() {
            third_label(self.set, t, self.pair)
        } else {
            triple_label(self.set, t)
        };
        let selected = t == self.chosen;
        let (rect, resp) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), 22.0), Sense::click());
        resp.widget_info(|| {
            egui::WidgetInfo::selected(
                egui::WidgetType::Button,
                true,
                selected,
                format!("axes {}", triple_label(self.set, t)),
            )
        });
        let painter = ui.painter_at(rect);
        if selected {
            painter.rect_filled(rect, 2.0, ui.visuals().selection.bg_fill);
        } else if resp.hovered() {
            painter.rect_filled(rect, 2.0, ui.visuals().widgets.hovered.weak_bg_fill);
        }
        let text_colour = if selected {
            ui.visuals().strong_text_color()
        } else {
            self.theme.caption
        };
        let num = if score.is_nan() {
            String::new()
        } else {
            format!("{score:.2}")
        };
        let num_w = 30.0;
        let bar = Rect::from_min_size(
            Pos2::new(rect.right() - num_w - 52.0, rect.center().y - 2.5),
            Vec2::new(40.0, 5.0),
        );
        painter
            .with_clip_rect(Rect::from_min_max(
                rect.min,
                Pos2::new(bar.left() - 4.0, rect.bottom()),
            ))
            .text(
                rect.left_center() + Vec2::new(6.0, 0.0),
                Align2::LEFT_CENTER,
                label,
                FontId::proportional(11.0),
                text_colour,
            );
        let frac = match self.source {
            Source::Variance => score / top.max(f32::MIN_POSITIVE),
            _ => score,
        };
        painter.rect_filled(bar, 2.5, self.theme.panel);
        if frac.is_finite() && frac > 0.0 {
            let fill = Rect::from_min_size(bar.min, Vec2::new(bar.width() * frac.min(1.0), 5.0));
            painter.rect_filled(fill, 2.5, self.theme.good);
        }
        painter.text(
            Pos2::new(rect.right() - 6.0, rect.center().y),
            Align2::RIGHT_CENTER,
            num,
            FontId::monospace(11.0),
            text_colour,
        );
        resp.clicked().then_some(t)
    }
}

fn legend_ui(ui: &mut Ui, items: &[(Swatch, String)], theme: &Theme) {
    for (swatch, text) in items {
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(Vec2::new(16.0, 12.0), Sense::hover());
            paint_swatch(ui.painter(), rect.shrink2(Vec2::new(1.0, 1.0)), *swatch);
            ui.label(egui::RichText::new(text).small().color(theme.text));
        });
    }
}

fn paint_swatch(p: &egui::Painter, r: Rect, swatch: Swatch) {
    let outline = [
        r.left_top(),
        r.right_top(),
        r.right_bottom(),
        r.left_bottom(),
        r.left_top(),
    ];
    match swatch {
        Swatch::Fill(c) => {
            p.rect_filled(r, 2.0, c.gamma_multiply(0.25));
            p.rect_stroke(r, 2.0, Stroke::new(1.0_f32, c), egui::StrokeKind::Inside);
        }
        Swatch::Wire(c) => {
            p.rect_stroke(r, 2.0, Stroke::new(1.0_f32, c), egui::StrokeKind::Inside);
            p.line_segment([r.left_bottom(), r.right_top()], Stroke::new(1.0_f32, c));
        }
        Swatch::Dashed(c) => {
            let s = Stroke::new(1.0_f32, c.gamma_multiply(0.85));
            p.extend(egui::Shape::dashed_line(&outline, s, 4.0, 3.0));
        }
        Swatch::Prism(c) => {
            let s = Stroke::new(1.0_f32, c);
            p.line_segment([r.left_top(), r.right_top()], s);
            p.line_segment([r.left_bottom(), r.right_bottom()], s);
            for x in [r.left(), r.right()] {
                p.extend(egui::Shape::dashed_line(
                    &[Pos2::new(x, r.top()), Pos2::new(x, r.bottom())],
                    s,
                    2.0,
                    2.0,
                ));
            }
        }
        Swatch::Dot(c) => {
            p.circle_filled(r.center(), 3.5, c);
        }
        Swatch::Ring(c) => {
            p.circle_stroke(r.center(), 5.0, Stroke::new(1.5_f32, c));
        }
        Swatch::Ramp => {
            for k in 0..5 {
                let w = r.width() / 5.0;
                let cell = Rect::from_min_size(
                    r.min + Vec2::new(k as f32 * w, 2.0),
                    Vec2::new(w, r.height() - 4.0),
                );
                p.rect_filled(cell, 0.0, viridis(k as f32 / 4.0));
            }
        }
        Swatch::Dotted(c) => {
            p.rect_filled(r, 2.0, c.gamma_multiply(0.15));
            p.extend(egui::Shape::dashed_line(
                &outline,
                Stroke::new(1.0_f32, c),
                1.0,
                2.0,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_view::FeaturePoint;

    /// Deterministic noise in `[0, 1)`.
    fn noise(s: &mut u32) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        (*s % 10_000) as f32 / 10_000.0
    }

    /// Two blobs over `n` channels that differ in the `sep` feature columns
    /// only (PC-major: column `pc * n + ch`); time and the rest are the same
    /// noise in both.
    fn blobs_n(n: usize, sep: &[usize]) -> FeatureSet {
        let mut s = 0xC0FFEE_u32;
        let (mut points, mut feats) = (Vec::new(), Vec::new());
        for slot in 0..2u16 {
            for _ in 0..80 {
                let off = slot as f32 * 0.8;
                points.push(FeaturePoint {
                    local: 0,
                    slot,
                    t: noise(&mut s) * 10.0,
                });
                for j in 0..2 * n {
                    feats.push(if sep.contains(&j) {
                        off + 0.1 * noise(&mut s)
                    } else {
                        noise(&mut s)
                    });
                }
            }
        }
        let mut set = FeatureSet::new(
            (3..3 + n as u32).collect(),
            false,
            true,
            points,
            feats,
            10.0,
        );
        set.bounds = vec![(0.0, 1.0); 2 * n];
        set
    }

    /// Two channels; the blobs differ in `0A` and `1A` (dims 1 and 2).
    fn blobs() -> FeatureSet {
        blobs_n(2, &[0, 1])
    }

    fn scored(set: &FeatureSet, source: Source) -> Vec<Scored> {
        let labels = source_labels(set, source, None);
        score_triples(set, &labels, &AtomicBool::new(false))
    }

    #[test]
    fn every_triple_of_the_set_is_scored_best_first() {
        // Three channels: time + 6 PCs = 7 dims, C(7, 3) triples.
        let set = blobs_n(3, &[0, 1]);
        assert_eq!(set.n_dims(), 7);
        let scores = scored(&set, Source::Clusters);
        assert_eq!(scores.len(), 35);
        assert_eq!(triples(7).len(), 35);
        let distinct: BTreeSet<_> = scores.iter().map(|s| s.0).collect();
        assert_eq!(distinct.len(), 35);
        assert!(scores.iter().all(|s| s.1.is_finite()), "{scores:?}");
        assert!(scores.windows(2).all(|w| w[0].1 >= w[1].1));
        // Eight channels: 17 dims, 680 triples.
        assert_eq!(triples(17).len(), 680);
        // The separating dims 1 and 2 are in the best triple; one without
        // either scores clearly lower.
        let best = scores[0].0;
        assert!(best.contains(&1) && best.contains(&2), "{best:?}");
        assert!(scores.last().is_some_and(|s| s.1 < scores[0].1 - 0.2));
    }

    #[test]
    fn one_label_falls_back_to_variance() {
        let set = blobs();
        let scores = scored(&set, Source::Variance);
        assert_eq!(scores.len(), 10);
        // Both blobs together spread most along the dims that differ.
        assert!(scores.iter().all(|s| s.1 > 0.0));
        let best = scores[0].0;
        assert!(best.contains(&1) && best.contains(&2), "{best:?}");
    }

    #[test]
    fn pinning_keeps_the_triples_holding_the_pair_and_ranks_the_third_axis() {
        // Pin 0A × 0B (dims 1, 3) while dim 2 (1A) is what separates.
        let set = blobs_n(2, &[0, 1]);
        let all = scored(&set, Source::Clusters);
        let nd = set.n_dims();
        let pinned = listed(&all, nd, Some([1, 3]));
        assert_eq!(pinned.len(), nd - 2);
        assert!(pinned.iter().all(|(t, _)| t.contains(&1) && t.contains(&3)));
        assert_eq!(pinned[0].0, [1, 2, 3], "the separating third axis first");
        assert_eq!(third_label(&set, pinned[0].0, Some([1, 3])), "+ ch4 PC1");
        // Unpinned lists everything; dims the set lacks are dropped.
        assert_eq!(listed(&all, nd, None).len(), 10);
        assert_eq!(listed(&all, 4, None).len(), 4);
    }

    #[test]
    fn the_chosen_triple_survives_a_rerank_and_falls_back_when_its_dims_vanish() {
        let set = blobs();
        let all = scored(&set, Source::Clusters);
        let list = listed(&all, 5, None);
        let worst = list.last().unwrap().0;
        assert_eq!(effective(&list, Some(worst)), worst);
        // Re-ranked by another label source: the same dims are still shown.
        let reranked = listed(&scored(&set, Source::Variance), 5, None);
        assert_eq!(effective(&reranked, Some(worst)), worst);
        // No pick, or one that is not listed (pin moved, set shrank): the best.
        assert_eq!(effective(&list, None), list[0].0);
        let pinned = listed(&all, 5, Some([0, 4]));
        assert_eq!(effective(&pinned, Some([1, 2, 3])), pinned[0].0);
        let small = listed(&all, 4, None);
        assert_eq!(effective(&small, Some([2, 3, 4])), small[0].0);
    }

    #[test]
    fn triple_channels_dedupe_and_leave_out_time() {
        let set = blobs();
        // Time, 0A, 1A: two channels.
        assert_eq!(channels_of(&set, [0, 1, 2]), vec![3, 4]);
        // 0A, 0B (pc 1 on channel 0), time: one channel.
        assert_eq!(channels_of(&set, [0, 1, 3]), vec![3]);
        // 1A, 1B, 0A: channel 4 once, then 3.
        assert_eq!(channels_of(&set, [1, 2, 4]), vec![3, 4]);
        let state = View3dState {
            shown: Some([0, 2, 4]),
            shown_channels: channels_of(&set, [0, 2, 4]),
            ..Default::default()
        };
        assert_eq!(state.triple_channels(&set), vec![4]);
        assert!(View3dState::default().triple_channels(&set).is_empty());
    }

    #[test]
    fn prism_needs_both_lasso_dims_in_the_triple() {
        let set = blobs();
        let (x, y) = (set.dim(1), set.dim(2));
        let poly = [[0.1, 0.1], [0.5, 0.1], [0.3, 0.6]];
        assert_eq!(prism_axes(&set, [1, 2, 3], x, y), Some([0, 1, 2]));
        assert_eq!(prism_axes(&set, [2, 3, 4], x, y), None);
        assert_eq!(prism_axes(&set, [0, 1, 3], x, y), None);
        let mut out = Vec::new();
        assert!(prism(&set, [1, 2, 3], (x, y, &poly), 0, &mut out));
        // Per vertex: two cap edges and one side edge, two vertices each.
        assert!(out.len() >= 3 * 3 * 2);
        out.clear();
        assert!(!prism(&set, [2, 3, 4], (x, y, &poly), 0, &mut out));
        assert!(out.is_empty());
    }

    #[test]
    fn scene_has_hulls_points_and_the_prism_only_when_it_fits() {
        let set = blobs();
        let styles = [
            Style {
                colour: Color32::RED,
                filled: true,
            },
            Style {
                colour: Color32::BLUE,
                filled: false,
            },
        ];
        let poly = [[0.1, 0.1], [0.5, 0.1], [0.3, 0.6]];
        let palette = Palette {
            cube: Color32::GRAY,
            ghost: Color32::GRAY,
            strong: Color32::WHITE,
            accent: Color32::LIGHT_BLUE,
            part: Color32::WHITE,
            dark: true,
        };
        let lines = |triple, lasso| {
            let overlay = Overlay {
                km_part: None,
                lasso,
                km_parts: None,
                focus: (Dim::Time, Dim::Time),
            };
            let (fg, bg, ink) = build_scene(&SceneInput {
                set: &set,
                triple,
                styles: &styles,
                ghosts: &[],
                overlay: &overlay,
                palette: &palette,
                brush: BrushInput::default(),
            });
            assert_eq!(fg.points.len(), 160);
            assert!(ink.points.is_empty());
            assert!(bg.points.is_empty() && !fg.tris.is_empty());
            fg.lines.len()
        };
        let none = lines([1, 2, 3], None);
        let with = lines([1, 2, 3], Some((set.dim(1), set.dim(2), &poly[..])));
        let without = lines([2, 3, 4], Some((set.dim(1), set.dim(2), &poly[..])));
        assert!(with > none);
        assert_eq!(without, lines([2, 3, 4], None));
    }

    /// `blobs_n` with each point's `local` its position in its cluster, and
    /// the clusters' depths 100 µm and 500 µm apart.
    fn with_locals_and_depths(n: usize, sep: &[usize]) -> (FeatureSet, SpikeDepths) {
        let mut set = blobs_n(n, sep);
        for (i, p) in set.points.iter_mut().enumerate() {
            p.local = (i % 80) as u32;
        }
        let per = |base: f32| (0..80).map(|i| base + i as f32 * 0.1).collect();
        let depths = SpikeDepths {
            per_cluster: vec![per(100.0), per(500.0)],
            source: sorrel_data::DepthSource::PcWeighted,
            min: 100.0,
            max: 508.0,
        };
        (set, depths)
    }

    #[test]
    fn space_dims_join_the_triples_and_rank_by_separation() {
        // The PCs do not separate the clusters; their depth does.
        let (set, depths) = with_locals_and_depths(2, &[]);
        let ext = with_space(&set, &[ClusterId(0), ClusterId(1)], &depths);
        let (depth, x) = space_dims(&ext).unwrap();
        assert_eq!(ext.n_dims(), 7);
        assert_eq!(label(depth, &ext), "depth");
        assert_eq!(label(x, &ext), "x");
        // The set's own dims keep their meaning.
        assert_eq!(ext.value(5, Dim::Time), set.value(5, Dim::Time));
        assert_eq!(
            ext.value(5, Dim::Pc { ch: 1, pc: 1 }),
            set.value(5, Dim::Pc { ch: 1, pc: 1 })
        );
        // Triples with the unused x are not offered: C(7, 3) - C(6, 2).
        let (di, xi) = (ext.dim_idx(depth), ext.dim_idx(x));
        let all = scored(&ext, Source::Clusters);
        assert_eq!(all.len(), 35 - 15);
        assert!(all.iter().all(|(t, _)| !t.contains(&xi)));
        assert!(all.iter().all(|s| s.1.is_finite()));
        // Depth is in the best triple and clearly beats a triple without it.
        assert!(all[0].0.contains(&di), "{:?}", all[0]);
        assert!(all.last().is_some_and(|s| s.1 < all[0].1 - 0.2));
        assert_eq!(
            short_label(&ext, [0, di, di + 1]).split("  ").nth(1),
            Some("depth")
        );
        // No pseudo channel is lit up in Waveforms.
        assert!(!channels_of(&ext, all[0].0).contains(&SPACE_CH));
        // Without depths the set is untouched.
        assert!(space_dims(&set).is_none());
        assert_eq!(set_triples(&set).len(), 10);
    }

    #[test]
    fn brushes_ink_fade_and_hide_spikes_in_the_scene() {
        let (set, _) = with_locals_and_depths(2, &[0, 1]);
        let styles = [
            Style {
                colour: Color32::RED,
                filled: true,
            },
            Style {
                colour: Color32::BLUE,
                filled: false,
            },
        ];
        let palette = Palette {
            cube: Color32::GRAY,
            ghost: Color32::GRAY,
            strong: Color32::WHITE,
            accent: Color32::LIGHT_BLUE,
            part: Color32::WHITE,
            dark: true,
        };
        let overlay = Overlay {
            km_part: None,
            lasso: None,
            km_parts: None,
            focus: (Dim::Time, Dim::Time),
        };
        let selection = [ClusterId(0), ClusterId(1)];
        let scene = |brush: BrushInput| {
            build_scene(&SceneInput {
                set: &set,
                triple: [1, 2, 3],
                styles: &styles,
                ghosts: &[],
                overlay: &overlay,
                palette: &palette,
                brush,
            })
        };
        // Three brushed spikes of c0 move to the ink scene.
        let b = crate::brush::SpikeBrush::new(
            crate::brush::BrushSource::FeatureLasso,
            0,
            [(ClusterId(0), vec![0, 1, 2]), (ClusterId(1), vec![5])],
        )
        .unwrap();
        let (fg, _, ink) = scene(BrushInput {
            spikes: Some((&b, &selection)),
            ..Default::default()
        });
        assert_eq!((fg.points.len(), ink.points.len()), (156, 4));
        // Fading keeps every spike; "only" drops those outside the range.
        let inside = set.points.iter().filter(|p| p.t <= 5.0).count();
        assert!((1..160).contains(&inside));
        let mut t = crate::brush::TimeBrush::new(0.0, 5.0).unwrap();
        let faded = scene(BrushInput {
            time: Some(t),
            ..Default::default()
        });
        assert_eq!(faded.0.points.len(), 160);
        let out = set.points.iter().position(|p| p.t > 5.0).unwrap();
        assert_ne!(
            faded.0.points[out].color,
            scene(BrushInput::default()).0.points[out].color
        );
        t.only = true;
        let only = scene(BrushInput {
            time: Some(t),
            ..Default::default()
        });
        assert_eq!(only.0.points.len(), inside);
        // Colouring by time replaces the cluster colour.
        let by_time = scene(BrushInput {
            colour: Some(PointColour::Time),
            ..Default::default()
        });
        assert_ne!(
            by_time.0.points[0].color,
            scene(BrushInput::default()).0.points[0].color
        );
    }

    #[test]
    fn hull_input_is_capped_for_large_units() {
        // 300k points (a big unit with pc_features.npy): the hull sees a
        // bounded, evenly spread sample, and it is built quickly.
        let pts: Vec<[f32; 3]> = (0..300_000)
            .map(|i| {
                let t = i as f32 * 0.618_034;
                [t.sin(), (t * 1.3).cos(), (t * 0.7).sin() * (t * 0.2).cos()]
            })
            .collect();
        let start = std::time::Instant::now();
        let kept = trimmed(pts);
        assert!(kept.len() <= HULL_SAMPLE && kept.len() > HULL_SAMPLE / 2);
        assert!(!convex_hull_3d(&kept).is_empty());
        assert!(start.elapsed().as_secs_f32() < 2.0, "{:?}", start.elapsed());
        assert_eq!(trimmed(vec![[0.0; 3]; 3]).len(), 3, "too few for a hull");
    }

    #[test]
    fn the_part_hull_survives_the_part_being_the_brush() {
        let (set, _) = with_locals_and_depths(2, &[0, 1]);
        let styles = [
            Style {
                colour: Color32::RED,
                filled: true,
            },
            Style {
                colour: Color32::BLUE,
                filled: true,
            },
        ];
        let palette = Palette {
            cube: Color32::GRAY,
            ghost: Color32::GRAY,
            strong: Color32::WHITE,
            accent: Color32::LIGHT_BLUE,
            part: Color32::WHITE,
            dark: true,
        };
        let part: Vec<usize> = (0..set.points.len())
            .filter(|&i| set.points[i].slot == 0)
            .collect();
        let selection = [ClusterId(0), ClusterId(1)];
        let lines = |km_part: Option<&[usize]>, brush: Option<&crate::brush::SpikeBrush>| {
            let overlay = Overlay {
                km_part,
                lasso: None,
                km_parts: None,
                focus: (Dim::Time, Dim::Time),
            };
            build_scene(&SceneInput {
                set: &set,
                triple: [1, 2, 3],
                styles: &styles,
                ghosts: &[],
                overlay: &overlay,
                palette: &palette,
                brush: BrushInput {
                    spikes: brush.map(|b| (b, &selection[..])),
                    ..Default::default()
                },
            })
            .0
            .lines
            .len()
        };
        // The brush is exactly the part's members, as for a k-means preview.
        let members: Vec<u32> = part.iter().map(|&i| set.points[i].local).collect();
        let b = crate::brush::SpikeBrush::new(
            crate::brush::BrushSource::KMeansPart,
            0,
            [(ClusterId(0), members)],
        )
        .unwrap();
        let (plain, with_part) = (lines(None, Some(&b)), lines(Some(&part), Some(&b)));
        assert!(
            with_part > plain,
            "dotted part hull: {with_part} vs {plain}"
        );
        assert_eq!(with_part, lines(Some(&part), None), "same with no brush");
    }

    #[test]
    fn a_new_brush_or_time_range_changes_the_scene_key() {
        let key = |b: &Brushes| {
            let mut h = DefaultHasher::new();
            hash_brushes(&mut h, b);
            h.finish()
        };
        let brush = |locals: Vec<u32>| {
            crate::brush::SpikeBrush::new(
                crate::brush::BrushSource::KMeansPart,
                1,
                [(ClusterId(0), locals)],
            )
        };
        let mut b = Brushes::default();
        let none = key(&b);
        b.spikes = brush(vec![1, 2]);
        let one = key(&b);
        b.spikes = brush(vec![1, 2, 3]);
        let two = key(&b);
        b.time = crate::brush::TimeBrush::new(1.0, 2.0);
        let three = key(&b);
        assert_eq!(BTreeSet::from([none, one, two, three]).len(), 4);
        assert_eq!(three, key(&b), "stable while nothing changes");
    }

    #[test]
    fn the_nearest_candidate_within_reach_is_picked() {
        let at = Pos2::new(10.0, 10.0);
        let pts = [
            (0, Pos2::new(30.0, 10.0)),
            (1, Pos2::new(14.0, 10.0)),
            (2, Pos2::new(11.0, 10.0)),
        ];
        assert_eq!(pick_nearest(pts.into_iter(), at), Some(2));
        assert_eq!(pick_nearest(pts[..1].iter().copied(), at), None);
    }

    #[test]
    fn camera_fits_the_cube() {
        assert!(fit_distance(1.0) > 4.0 && fit_distance(0.5) > fit_distance(1.0));
        assert_eq!(fit_distance(2.0), fit_distance(1.0));
    }
}

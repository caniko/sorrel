//! phy's TemplateFeatureView: with exactly two clusters selected, every
//! spike's projection on the first cluster's template (x) against the
//! second's (y), from `template_features.npy`, with grey background
//! spikes. Ctrl+click lasso, `k` splits.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use egui::{Pos2, Sense, Ui};
use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider};

use crate::jobs::{JobCtx, Memo};
use crate::lasso::Lasso;
use crate::theme::{selection_colour, Theme};
use crate::views::{axes_data_rect, draw_axes_grid};
use crate::wgpu_raster::pack_rgba;
use crate::wgpu_scatter::{ScatterCallback, ScatterVertex};

const N_BACKGROUND: usize = 2500;
const BACKGROUND: u16 = u16::MAX;

/// `(local, slot, x, y)`.
type Point = (u32, u16, f32, f32);

struct TfSet {
    templates: [u32; 2],
    points: Vec<Point>,
    generation: u64,
}

#[derive(Default)]
pub struct TemplateFeatureState {
    lasso: Lasso,
    split_requested: bool,
    pz: crate::panzoom::PanZoom,
    memo: Memo<(Vec<ClusterId>, u64), Option<Arc<TfSet>>>,
    vertices: Option<(u64, bool, Arc<Vec<ScatterVertex>>)>,
}

impl TemplateFeatureState {
    pub fn lasso_closed(&self) -> bool {
        self.lasso.is_closed()
    }
    pub fn request_split(&mut self) {
        self.split_requested = self.lasso.is_closed();
    }
    pub fn cancel_lasso(&mut self) -> bool {
        let had = self.lasso.is_active();
        self.lasso.clear();
        self.split_requested = false;
        had
    }
    pub fn cancel(&mut self) {
        self.memo.cancel();
    }
    pub fn pending(&self) -> bool {
        self.memo.pending()
    }
}

static GENERATION: AtomicU64 = AtomicU64::new(1);

fn compute<P: DataProvider>(s: &Session<P>, selection: &[ClusterId]) -> Option<Arc<TfSet>> {
    let ci = s.cluster_index();
    let templates = [
        s.primary_template(selection[0])?,
        s.primary_template(selection[1])?,
    ];
    let slot_of: HashMap<ClusterId, u16> = selection
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i as u16))
        .collect();
    let clusters = s.spike_clusters_global();
    let bg_step = (clusters.len() / N_BACKGROUND).max(1);
    let mut counters = [0u32; 2];
    let mut points = Vec::new();
    for (g, c) in clusters.iter().enumerate() {
        let slot = slot_of.get(c).copied();
        let local = match slot {
            Some(sl) => {
                counters[sl as usize] += 1;
                counters[sl as usize] - 1
            }
            None if g.is_multiple_of(bg_step) => 0,
            None => continue,
        };
        let f = |t| ci.template_feature(g as u32, t);
        // phylib fills a missing template projection with 0 for the
        // selected clusters; background spikes need both.
        let (x, y) = match slot {
            Some(_) => (
                f(templates[0]).unwrap_or(0.0),
                f(templates[1]).unwrap_or(0.0),
            ),
            None => match (f(templates[0]), f(templates[1])) {
                (Some(x), Some(y)) => (x, y),
                _ => continue,
            },
        };
        points.push((local, slot.unwrap_or(BACKGROUND), x, y));
    }
    Some(Arc::new(TfSet {
        templates,
        points,
        generation: GENERATION.fetch_add(1, Ordering::Relaxed),
    }))
}

pub fn template_feature_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    state: &mut TemplateFeatureState,
    jobs: &JobCtx<Session<P>>,
) -> Option<Vec<(ClusterId, Vec<u32>)>> {
    let theme = Theme::from_ui(ui);
    if !session.cluster_index().has_template_features() {
        ui.label("template_features.npy and template_feature_ind.npy not loaded");
        return None;
    }
    if selection.len() != 2 {
        ui.label("select exactly two clusters (phy TemplateFeatureView)");
        return None;
    }
    let key = (selection.to_vec(), session.version());
    let sel = selection.to_vec();
    let current = state
        .memo
        .ensure(key, &jobs.session, jobs.repaint(), move |s, _| {
            compute(s, &sel)
        })
        .cloned();
    let Some(set) = current.clone().flatten() else {
        if current.is_none() {
            ui.spinner();
        } else {
            ui.label("both clusters need spike templates");
        }
        return None;
    };
    let hint = if state.lasso.is_closed() {
        "k splits · Esc clears"
    } else {
        "Ctrl+click: lasso"
    };
    ui.add(egui::Label::new(egui::RichText::new(hint).weak()).truncate());

    let avail = ui.available_size_before_wrap();
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(avail.x.max(160.0), avail.y.max(140.0)),
        Sense::click_and_drag(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme.plot_bg);
    let range = |f: fn(&Point) -> f32| {
        let (lo, hi) = set
            .points
            .iter()
            .filter(|p| p.1 != BACKGROUND)
            .map(f)
            .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
        let pad = ((hi - lo) * 0.05).max(1e-3);
        if lo <= hi {
            (lo - pad, hi + pad)
        } else {
            (0.0, 1.0)
        }
    };
    let (x, y) = (range(|p| p.2), range(|p| p.3));
    let (x, y) = state.pz.apply(x, y);
    draw_axes_grid(
        &painter,
        rect,
        x,
        y,
        &format!("on t{} (c{})", set.templates[0], selection[0]),
        &format!("on t{} (c{})", set.templates[1], selection[1]),
        &theme,
    );
    let data = axes_data_rect(rect, true);
    state.pz.interact(&resp, data);
    let dark = ui.visuals().dark_mode;
    if !matches!(&state.vertices, Some((g, d, _)) if *g == set.generation && *d == dark) {
        let grey = if dark { 150 } else { 110 };
        let mut order: Vec<&Point> = set.points.iter().collect();
        order.sort_by_key(|p| p.1 != BACKGROUND);
        let v = order
            .into_iter()
            .map(|p| ScatterVertex {
                pos: [p.2, p.3],
                color: if p.1 == BACKGROUND {
                    pack_rgba(grey, grey, grey, 110)
                } else {
                    let c = selection_colour(p.1 as usize);
                    pack_rgba(c.r(), c.g(), c.b(), 220)
                },
            })
            .collect();
        state.vertices = Some((set.generation, dark, Arc::new(v)));
    }
    if let Some((_, _, v)) = &state.vertices {
        ui.painter()
            .add(eframe::egui_wgpu::Callback::new_paint_callback(
                data,
                ScatterCallback {
                    id: ui.id().with("template-feature-scatter").value(),
                    vertices: Arc::clone(v),
                    generation: set.generation * 2 + dark as u64,
                    x,
                    y,
                    radius: 1.5,
                    size: data.size(),
                },
            ));
    }
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
    state.lasso.interact(0, &resp, to_data);
    state.lasso.paint(&painter, to_screen, theme.text);

    if state.split_requested && state.lasso.is_closed() && current.is_some() {
        state.split_requested = false;
        let mut hits: [Vec<u32>; 2] = Default::default();
        for p in set.points.iter().filter(|p| p.1 != BACKGROUND) {
            if state.lasso.contains([p.2, p.3]) {
                hits[p.1 as usize].push(p.0);
            }
        }
        state.lasso.clear();
        return Some(
            selection
                .iter()
                .copied()
                .zip(hits)
                .filter(|(_, h)| !h.is_empty())
                .collect(),
        );
    }
    None
}

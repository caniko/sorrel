//! phy's Spike*Views: one plot per `spike_<name>.npy` attribute, picked in
//! the header. A 1-D attribute is drawn against time, a 2-D one as x vs y;
//! the selection in slot colours over grey background spikes, with the
//! Ctrl+click lasso and `k` to split.

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

struct AttrSet {
    points: Vec<Point>,
    two_d: bool,
    generation: u64,
}

type AttrKey = (Vec<ClusterId>, u64, String, bool);

#[derive(Default)]
pub struct SpikeAttrState {
    /// The attribute shown; the first one when `None`.
    pub name: Option<String>,
    lasso: Lasso,
    split_requested: bool,
    pz: crate::panzoom::PanZoom,
    memo: Memo<AttrKey, Arc<AttrSet>>,
    vertices: Option<(u64, bool, Arc<Vec<ScatterVertex>>)>,
}

impl SpikeAttrState {
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

fn compute<P: DataProvider>(s: &Session<P>, key: &AttrKey) -> AttrSet {
    let (selection, _, name, reorder) = key;
    let ci = s.cluster_index();
    let two_d = ci
        .spike_attribute_names()
        .iter()
        .any(|&(n, k)| n == name && k == 2);
    let sr = s.provider.sample_rate().max(1.0);
    let slot_of: HashMap<ClusterId, u16> = selection
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i as u16))
        .collect();
    let clusters = s.spike_clusters_global();
    let bg_step = (clusters.len() / N_BACKGROUND).max(1);
    let mut counters = vec![0u32; selection.len()];
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
        let Some(v) = ci.spike_attribute(name, g as u32) else {
            continue;
        };
        let (x, y) = if two_d {
            (v[0], v[1])
        } else {
            (ci.display_time(g as u32, *reorder).0 as f32 / sr, v[0])
        };
        if x.is_finite() && y.is_finite() {
            points.push((local, slot.unwrap_or(BACKGROUND), x, y));
        }
    }
    AttrSet {
        points,
        two_d,
        generation: GENERATION.fetch_add(1, Ordering::Relaxed),
    }
}

pub fn spike_attr_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    state: &mut SpikeAttrState,
    jobs: &JobCtx<Session<P>>,
    reorder: bool,
) -> Option<Vec<(ClusterId, Vec<u32>)>> {
    let theme = Theme::from_ui(ui);
    let names: Vec<(String, usize)> = session
        .cluster_index()
        .spike_attribute_names()
        .into_iter()
        .map(|(n, k)| (n.to_string(), k))
        .collect();
    if names.is_empty() {
        ui.label("no spike_<name>.npy attributes in this dataset");
        return None;
    }
    if selection.is_empty() {
        ui.label("no selection");
        return None;
    }
    let name = state
        .name
        .clone()
        .filter(|n| names.iter().any(|(m, _)| m == n))
        .unwrap_or_else(|| names[0].0.clone());
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("spike-attr")
            .selected_text(format!("spike_{name}"))
            .show_ui(ui, |ui| {
                for (n, k) in &names {
                    let label = format!("spike_{n} ({}-D)", k);
                    if ui.selectable_label(*n == name, label).clicked() {
                        state.name = Some(n.clone());
                        state.lasso.clear();
                    }
                }
            });
        ui.weak(if state.lasso.is_closed() {
            "k splits · Esc clears"
        } else {
            "Ctrl+click: lasso"
        });
    });
    let key: AttrKey = (selection.to_vec(), session.version(), name.clone(), reorder);
    let job_key = key.clone();
    let current = state
        .memo
        .ensure(key, &jobs.session, jobs.repaint(), move |s, _| {
            Arc::new(compute(s, &job_key))
        })
        .cloned();
    let Some(set) = current.clone().or_else(|| state.memo.last().cloned()) else {
        ui.spinner();
        return None;
    };

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
    let (xl, yl) = if set.two_d {
        (format!("{name}[0]"), format!("{name}[1]"))
    } else {
        ("time (s)".to_string(), name.clone())
    };
    draw_axes_grid(&painter, rect, x, y, &xl, &yl, &theme);
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
    if let Some((g, d, v)) = &state.vertices {
        ui.painter()
            .add(eframe::egui_wgpu::Callback::new_paint_callback(
                data,
                ScatterCallback {
                    id: ui.id().with("spike-attr-scatter").value(),
                    vertices: Arc::clone(v),
                    generation: g * 2 + *d as u64,
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
        let mut hits: Vec<Vec<u32>> = vec![Vec::new(); selection.len()];
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

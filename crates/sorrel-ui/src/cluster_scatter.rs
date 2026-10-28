//! phy v2.1's ClusterScatterView: one point per cluster, x / y / size
//! chosen from per-cluster columns (defaults amp, depth, log firing
//! rate). Click selects, Shift+click adds or removes, Ctrl+click draws a
//! lasso that selects every cluster inside it.

use egui::{Pos2, Rect, Sense, Stroke, Ui};
use sorrel_data::{Session, SpikeDepths};
use sorrel_io::{ClusterId, DataProvider};

use crate::lasso::Lasso;
use crate::theme::{selection_colour, Theme};
use crate::views::{axes_data_rect, draw_axes_grid};

/// Gap between the pane edge and the plot.
const PLOT_INSET: f32 = 6.0;
/// Margin inside the data rect that points map into, so a selection ring
/// (radius up to 9 + 3, 2 px stroke) at the extreme never clips.
const DATA_INSET: f32 = 12.0;
/// Row above the plot that carries the y-axis title.
const TITLE_H: f32 = 14.0;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Column {
    NSpikes,
    FiringRate,
    Amplitude,
    Depth,
    Channel,
}

impl Column {
    const ALL: [Column; 5] = [
        Column::Amplitude,
        Column::Depth,
        Column::FiringRate,
        Column::NSpikes,
        Column::Channel,
    ];

    /// phy's cluster-view column names.
    pub fn name(self) -> &'static str {
        match self {
            Column::NSpikes => "n_spikes",
            Column::FiringRate => "fr",
            Column::Amplitude => "amp",
            Column::Depth => "depth",
            Column::Channel => "ch",
        }
    }

    /// Axis title with units.
    fn title(self) -> &'static str {
        match self {
            Column::NSpikes => "n_spikes",
            Column::FiringRate => "firing rate (Hz)",
            Column::Amplitude => "amp",
            Column::Depth => "depth (µm)",
            Column::Channel => "channel",
        }
    }

    pub fn parse(s: &str) -> Option<Column> {
        Column::ALL.into_iter().find(|c| c.name() == s)
    }
}

/// `(cluster, [n_spikes, fr, amp, depth, ch])`.
type Row = (ClusterId, [f32; 5]);

pub struct ClusterScatterState {
    pub axes: [Column; 3],
    /// Log scale for x, y and size (phy logs the firing-rate size).
    pub log: [bool; 3],
    lasso: Lasso,
    pz: crate::panzoom::PanZoom,
    /// Per-cluster columns for `(version, depths known)`.
    rows: Option<(u64, bool, Vec<Row>)>,
}

impl Default for ClusterScatterState {
    fn default() -> Self {
        Self {
            axes: [Column::Amplitude, Column::Depth, Column::FiringRate],
            log: [false, false, true],
            lasso: Lasso::default(),
            rows: None,
            pz: Default::default(),
        }
    }
}

impl ClusterScatterState {
    pub fn cancel_lasso(&mut self) -> bool {
        let had = self.lasso.is_active();
        self.lasso.clear();
        had
    }
}

/// Per-cluster `[n_spikes, fr, amp, depth, ch]` for every non-empty
/// cluster. Depth is the mean spike depth when the backend estimates
/// depths, else NaN; ch is the primary template's peak channel.
fn rows<P: DataProvider>(
    s: &Session<P>,
    depths: Option<&SpikeDepths>,
) -> Vec<(ClusterId, [f32; 5])> {
    let duration = (s.provider.n_samples().0 as f32 / s.provider.sample_rate().max(1.0)).max(1e-3);
    let (_, ns, nc) = s.template_shape();
    (0..s.n_clusters())
        .map(ClusterId)
        .filter(|&c| !s.spike_times(c).is_empty())
        .map(|c| {
            let n = s.spike_times(c).len() as f32;
            let mean = |v: &[f32]| {
                if v.is_empty() {
                    f32::NAN
                } else {
                    v.iter().sum::<f32>() / v.len() as f32
                }
            };
            let depth = depths
                .and_then(|d| d.per_cluster.get(c.idx()))
                .map_or(f32::NAN, |d| mean(d));
            let ch = s
                .primary_template(c)
                .and_then(|t| s.template_waveform(t))
                .and_then(|w| {
                    (0..nc).max_by(|&a, &b| {
                        let ptp = |ch: usize| {
                            let col = (0..ns).filter_map(|i| w.get(i * nc + ch));
                            let (lo, hi) = col
                                .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
                            hi - lo
                        };
                        ptp(a).total_cmp(&ptp(b))
                    })
                })
                .map_or(f32::NAN, |ch| ch as f32);
            (c, [n, n / duration, mean(s.spike_amplitudes(c)), depth, ch])
        })
        .collect()
}

/// Returns the new selection when the user clicked or lassoed.
pub fn cluster_scatter_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    depths: Option<&SpikeDepths>,
    state: &mut ClusterScatterState,
    colours: &[Option<egui::Color32>],
    scheme_delta: &mut i32,
) -> Option<Vec<ClusterId>> {
    let theme = Theme::from_ui(ui);
    let key = (session.version(), depths.is_some());
    if !matches!(&state.rows, Some((v, d, _)) if (*v, *d) == key) {
        state.rows = Some((key.0, key.1, rows(session, depths)));
    }
    let n_clusters = state.rows.as_ref().map_or(0, |r| r.2.len());
    let [ax, ay, asize] = &mut state.axes;
    let log = &mut state.log;
    let axis_combo = |ui: &mut Ui, i: usize, what: &str, axis: &mut Column| {
        egui::ComboBox::from_id_salt(("cluster-scatter-axis", i))
            .selected_text(format!("{what}: {}", axis.name()))
            .show_ui(ui, |ui| {
                for c in Column::ALL {
                    ui.selectable_value(axis, c, c.name());
                }
            });
    };
    crate::views::param_bar(
        ui,
        &format!("{n_clusters} clusters"),
        |ui| {
            axis_combo(ui, 0, "x", ax);
            axis_combo(ui, 1, "y", ay);
        },
        |ui| {
            axis_combo(ui, 2, "size", asize);
            for (flag, what) in log.iter_mut().zip(["log x", "log y", "log size"]) {
                ui.checkbox(flag, what);
            }
        },
    );
    let rows = state
        .rows
        .as_ref()
        .map(|r| r.2.as_slice())
        .unwrap_or_default();
    let value = |r: &[f32; 5], axis: usize| {
        let v = r[col_index(state.axes[axis])];
        if state.log[axis] {
            v.max(1e-6).log10()
        } else {
            v
        }
    };
    let pts: Vec<(ClusterId, f32, f32, f32)> = rows
        .iter()
        .map(|(c, r)| (*c, value(r, 0), value(r, 1), value(r, 2)))
        .filter(|p| p.1.is_finite() && p.2.is_finite())
        .collect();
    if pts.is_empty() {
        ui.label(format!(
            "no cluster has both {} and {}",
            state.axes[0].name(),
            state.axes[1].name()
        ));
        return None;
    }
    let range = |f: fn(&(ClusterId, f32, f32, f32)) -> f32| {
        let (lo, hi) = pts
            .iter()
            .map(f)
            .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
        let pad = ((hi - lo) * 0.05).max(1e-3);
        (lo - pad, hi + pad)
    };
    let (x, y) = (range(|p| p.1), range(|p| p.2));
    let (x, y) = state.pz.apply(x, y);
    let (s_lo, s_hi) = pts
        .iter()
        .map(|p| p.3)
        .filter(|v| v.is_finite())
        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));

    let avail = ui.available_size_before_wrap();
    let (outer, resp) = ui.allocate_exact_size(
        egui::vec2(avail.x.max(160.0), avail.y.max(140.0)),
        Sense::click_and_drag(),
    );
    let full = outer.shrink(PLOT_INSET);
    let rect = Rect::from_min_max(Pos2::new(full.left(), full.top() + TITLE_H), full.max);
    resp.widget_info(|| {
        egui::WidgetInfo::labeled(egui::WidgetType::Other, true, "cluster scatter")
    });
    let painter = ui.painter_at(outer);
    painter.rect_filled(rect, 0.0, theme.plot_bg);
    let axis_name = |i: usize| {
        let n = state.axes[i].title();
        if state.log[i] {
            format!("log {n}")
        } else {
            n.to_string()
        }
    };
    let data: Rect = axes_data_rect(rect, true);
    // Widen the axis ranges by the inset so ticks stay true to the mapping.
    let widen = |r: (f32, f32), len: f32| {
        let pad = (r.1 - r.0) * DATA_INSET / (len - 2.0 * DATA_INSET).max(1.0);
        (r.0 - pad, r.1 + pad)
    };
    let (x, y) = (widen(x, data.width()), widen(y, data.height()));
    // The y title sits in its own row above the plot, clear of the spine.
    draw_axes_grid(&painter, rect, x, y, &axis_name(0), "", &theme);
    painter.text(
        full.left_top(),
        egui::Align2::LEFT_TOP,
        axis_name(1),
        egui::FontId::proportional(crate::views::TICK_PX),
        theme.caption,
    );
    state.pz.interact(&resp, data);
    let dp = painter.with_clip_rect(data);
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
    let radius = |s: f32| {
        if s.is_finite() && s_hi > s_lo {
            2.5 + 6.0 * (s - s_lo) / (s_hi - s_lo)
        } else {
            4.0
        }
    };
    // Unselected first, so the selection draws on top.
    let mut order = pts.clone();
    order.sort_by_key(|p| selection.contains(&p.0));
    for &(c, px, py, ps) in &order {
        let at = to_screen([px, py]);
        match selection.iter().position(|&s| s == c) {
            Some(slot) => {
                // Cluster colour inside, detached slot-coloured ring outside.
                let r = radius(ps).max(5.0);
                let fill = colours.get(c.idx()).copied().flatten();
                dp.circle_filled(at, r, fill.unwrap_or(selection_colour(slot)));
                dp.circle_stroke(at, r + 3.0, Stroke::new(2.0_f32, selection_colour(slot)));
            }
            None => {
                let col = colours.get(c.idx()).copied().flatten();
                dp.circle_filled(
                    at,
                    radius(ps),
                    col.unwrap_or(theme.caption.gamma_multiply(0.6)),
                );
            }
        }
    }
    let hovered = resp.hover_pos().and_then(|h| {
        pts.iter()
            .map(|p| (p.0, to_screen([p.1, p.2]).distance(h)))
            .filter(|&(_, d)| d < 10.0)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(c, _)| c)
    });
    if let Some(c) = hovered {
        let row = rows
            .iter()
            .find(|r| r.0 == c)
            .map(|r| r.1)
            .unwrap_or_default();
        resp.clone().on_hover_text(format!(
            "c{c} · n_spikes {} · fr {:.2} Hz · amp {:.1} · depth {:.0} · ch {}",
            row[0], row[1], row[2], row[3], row[4]
        ));
    }

    *scheme_delta += crate::theme::scheme_wheel(ui, &resp);
    state.lasso.paint(&painter, to_screen, theme.text);
    if state.lasso.interact(0, &resp, to_data) {
        if state.lasso.is_closed() {
            let inside: Vec<ClusterId> = pts
                .iter()
                .filter(|p| state.lasso.contains([p.1, p.2]))
                .map(|p| p.0)
                .collect();
            return (!inside.is_empty()).then_some(inside);
        }
        return None;
    }
    if resp.clicked() {
        let c = hovered?;
        let shift = ui.input(|i| i.modifiers.shift);
        let mut next = if shift {
            selection.to_vec()
        } else {
            Vec::new()
        };
        match next.iter().position(|&s| s == c) {
            Some(i) if shift => {
                next.remove(i);
            }
            _ => next.push(c),
        }
        return (!next.is_empty()).then_some(next);
    }
    None
}

fn col_index(c: Column) -> usize {
    match c {
        Column::NSpikes => 0,
        Column::FiringRate => 1,
        Column::Amplitude => 2,
        Column::Depth => 3,
        Column::Channel => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_round_trip_phy_names() {
        for c in Column::ALL {
            assert_eq!(Column::parse(c.name()), Some(c));
        }
        assert_eq!(col_index(Column::Depth), 3);
        assert_eq!(Column::parse("nope"), None);
    }
}

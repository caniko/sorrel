//! phy ISIView: inter-spike-interval histograms, overlaid per cluster.
//!
//! One plot, one slot-coloured step outline per cluster. `n_bins` bins span
//! exactly `[x_min, x_max)` (`n_bins + 1` edges; phy's `linspace` loses one).

use std::sync::Arc;

use egui::{Sense, Stroke, Ui};
use sorrel_compute::{isi_histogram_range, isi_violations};
use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider};

use crate::correlogram_view::{band_fill, wheel_scale};
use crate::prefs::IsiParams;
use crate::theme::{contrast_text, selection_colour, Theme};
use crate::views::{
    axes, axes_data_rect, fill_height, int_ticks, param_bar, AxisOpts, TICK_PX,
};

/// Rows past this are not drawn (phy's grid limit; keeps the pane legible).
const MAX_CLUSTERS: usize = 20;
const MAX_BINS: usize = 500;

type IsiKey = (Vec<ClusterId>, u64, u64, u64, usize);

/// Histograms for one key, parked in egui's temp memory: the compute is a
/// linear scan per cluster, so a key-compared cache is all it needs.
#[derive(Clone)]
struct IsiEntry {
    key: IsiKey,
    hists: Arc<Vec<Vec<u32>>>,
}

/// phy status text: `[x_min ms, x_max ms]`.
pub(crate) fn isi_caption(x_min_ms: f32, x_max_ms: f32) -> String {
    format!("[{x_min_ms:.1} ms, {x_max_ms:.1} ms]")
}

/// Keep the range non-empty and the bin count drawable.
pub(crate) fn clamp_params(p: &mut IsiParams) {
    let fin = |x: f32, lo: f32, hi: f32, d: f32| if x.is_finite() { x.clamp(lo, hi) } else { d };
    p.x_min_ms = fin(p.x_min_ms, 0.0, 9_999.0, 0.0);
    p.x_max_ms = fin(p.x_max_ms, 0.1, 10_000.0, 50.0).max(p.x_min_ms + 0.1);
    p.n_bins = p.n_bins.clamp(1, MAX_BINS);
}

/// Scale a bin count by a wheel factor, always moving at least one step
/// when the factor is not 1 (×1.1 of 5 would otherwise round back to 5).
pub(crate) fn scale_bins(n: usize, f: f32) -> usize {
    let m = (n as f32 * f).round() as usize;
    match (m == n, f.partial_cmp(&1.0)) {
        (true, Some(std::cmp::Ordering::Greater)) => n + 1,
        (true, Some(std::cmp::Ordering::Less)) => n.saturating_sub(1),
        _ => m,
    }
}

/// One row: x max inline, x min and bins behind the gear.
fn header(ui: &mut Ui, p: &mut IsiParams) {
    let caption = format!(
        "ISI histogram · {} · {} bins",
        isi_caption(p.x_min_ms, p.x_max_ms),
        p.n_bins
    );
    let (x_min, x_max, bins) = (&mut p.x_min_ms, &mut p.x_max_ms, &mut p.n_bins);
    param_bar(
        ui,
        &caption,
        |ui| {
            ui.label("x max");
            ui.add(
                egui::DragValue::new(x_max)
                    .speed(0.5)
                    .range(0.1..=10_000.0)
                    .suffix(" ms"),
            );
        },
        |ui| {
            egui::Grid::new("isi_params").show(ui, |ui| {
                ui.label("x min");
                ui.add(
                    egui::DragValue::new(x_min)
                        .speed(0.5)
                        .range(0.0..=9_999.0)
                        .suffix(" ms"),
                );
                ui.end_row();
                ui.label("bins");
                ui.add(egui::DragValue::new(bins).range(1..=MAX_BINS));
                ui.end_row();
            });
        },
    );
}

/// Inter-spike-interval histograms of the selection, overlaid in one plot
/// as slot-coloured step outlines, with a per-cluster violations legend.
pub fn isi_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    p: &mut IsiParams,
) {
    let theme = Theme::from_ui(ui);
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }
    p.x_max_ms *= wheel_scale(ui, true);
    p.n_bins = scale_bins(p.n_bins, wheel_scale(ui, false));
    header(ui, p);
    clamp_params(p);

    let shown = &selection[..selection.len().min(MAX_CLUSTERS)];
    if selection.len() > shown.len() {
        ui.weak(format!(
            "showing the first {MAX_CLUSTERS} of {} selected clusters",
            selection.len()
        ));
    }
    let sr = session.provider.sample_rate().max(1.0);
    let to_samples = |ms: f32| (sr as f64 * ms as f64 / 1e3).round() as u64;
    let (lo, hi) = (to_samples(p.x_min_ms), to_samples(p.x_max_ms));
    let key: IsiKey = (shown.to_vec(), session.version(), lo, hi, p.n_bins);
    let id = egui::Id::new("isi_view_cache");
    let hists = match ui.data(|d| d.get_temp::<IsiEntry>(id)) {
        Some(e) if e.key == key => e.hists,
        _ => {
            let hists = Arc::new(
                shown
                    .iter()
                    .map(|&c| isi_histogram_range(session.spike_times(c), lo, hi, p.n_bins))
                    .collect::<Vec<_>>(),
            );
            ui.data_mut(|d| {
                d.insert_temp(
                    id,
                    IsiEntry {
                        key,
                        hists: Arc::clone(&hists),
                    },
                )
            });
            hists
        }
    };

    // Legend row: one chip per cluster with its <1 ms violations.
    let refractory_samples = (sr * 0.001).round() as u64;
    ui.horizontal_wrapped(|ui| {
        for (slot, &c) in shown.iter().enumerate() {
            let n = session.spike_times(c).len();
            let viol = isi_violations(session.spike_times(c), refractory_samples);
            let pct = if n == 0 { 0.0 } else { viol as f32 / n as f32 * 100.0 };
            let (bg, text) = (selection_colour(slot), format!("c{c} <1 ms {viol}/{n} ({pct:.1}%)"));
            let fg = contrast_text(bg);
            let g = ui
                .painter()
                .layout_no_wrap(text, egui::FontId::proportional(TICK_PX), fg);
            let (r, _) = ui.allocate_exact_size(g.size() + egui::vec2(10.0, 4.0), Sense::hover());
            ui.painter().rect_filled(r, 3.0, bg);
            ui.painter().galley(r.min + egui::vec2(5.0, 2.0), g, fg);
        }
    });

    let (rect, _resp) = ui.allocate_exact_size(
        egui::vec2(ui.available_width().max(120.0), fill_height(ui).max(160.0)),
        Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme.plot_bg);
    let data = axes_data_rect(rect, true);
    let max_count = hists
        .iter()
        .flatten()
        .copied()
        .max()
        .unwrap_or(1)
        .max(1) as f32;
    let y_top = max_count * 1.1;
    let yt = int_ticks(max_count, 3);
    axes(
        &painter,
        rect,
        (p.x_min_ms, p.x_max_ms),
        (0.0, y_top),
        "ms",
        "",
        &theme,
        AxisOpts {
            grid: true,
            y_ticks: Some(&yt),
            ..AxisOpts::default()
        },
    );
    let dp = painter.with_clip_rect(data);
    let x_of = |ms: f32| data.left() + (ms - p.x_min_ms) / (p.x_max_ms - p.x_min_ms) * data.width();

    // Shade [0, 1 ms) over the full height (clipped to the visible range;
    // empty once x_min >= 1), at least 3 px wide so it never reads as a line.
    let band = (p.x_min_ms.max(0.0), 1.0_f32.min(p.x_max_ms));
    if band.1 > band.0 {
        let (x0, x1) = (x_of(band.0), x_of(band.1).max(x_of(band.0) + 3.0));
        dp.rect_filled(
            egui::Rect::from_min_max(egui::pos2(x0, data.top()), egui::pos2(x1, data.bottom())),
            0.0,
            band_fill(&theme),
        );
        dp.line_segment(
            [egui::pos2(x1, data.top()), egui::pos2(x1, data.bottom())],
            Stroke::new(1.0_f32, theme.danger),
        );
        dp.text(
            egui::pos2(x1 + 4.0, data.top() + 2.0),
            egui::Align2::LEFT_TOP,
            "1 ms",
            egui::FontId::proportional(TICK_PX),
            theme.danger,
        );
    }

    // Step outlines, one per cluster, on a shared count axis.
    let bw = data.width() / p.n_bins as f32;
    let y_of = |count: u32| data.bottom() - count as f32 / y_top * data.height();
    for (slot, h) in hists.iter().enumerate() {
        let mut pts = Vec::with_capacity(h.len() * 2 + 2);
        pts.push(egui::pos2(data.left(), data.bottom()));
        for (i, &count) in h.iter().enumerate() {
            let x0 = data.left() + bw * i as f32;
            pts.push(egui::pos2(x0, y_of(count)));
            pts.push(egui::pos2(x0 + bw, y_of(count)));
        }
        pts.push(egui::pos2(data.left() + bw * h.len() as f32, data.bottom()));
        dp.add(egui::Shape::line(
            pts,
            Stroke::new(2.0_f32, selection_colour(slot)),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caption_matches_phy_status() {
        assert_eq!(isi_caption(0.0, 50.0), "[0.0 ms, 50.0 ms]");
    }

    #[test]
    fn wheel_always_moves_the_bin_count() {
        assert_eq!(scale_bins(50, 1.0), 50);
        assert_eq!(scale_bins(50, 1.1), 55);
        assert_eq!(
            scale_bins(5, 1.1),
            6,
            "rounding back to 5 would be a dead wheel"
        );
        assert_eq!(scale_bins(5, 1.0 / 1.1), 4);
        assert_eq!(scale_bins(1, 1.0 / 1.1), 0, "clamp_params restores 1");
    }

    #[test]
    fn clamp_keeps_range_non_empty() {
        let mut p = IsiParams {
            x_min_ms: 60.0,
            x_max_ms: 50.0,
            n_bins: 0,
        };
        clamp_params(&mut p);
        assert!(p.x_max_ms > p.x_min_ms);
        assert_eq!(p.n_bins, 1);
        p.x_min_ms = f32::NAN;
        clamp_params(&mut p);
        assert_eq!(p.x_min_ms, 0.0);
    }
}

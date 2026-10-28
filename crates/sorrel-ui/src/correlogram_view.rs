//! phy CorrelogramView: auto/cross-correlogram grid with phylib binning.
//!
//! Counts come from `sorrel_compute::correlograms` (phylib-exact: odd bin
//! count centred on lag 0, self-pairs excluded, lag-0 symmetrized) on a
//! worker thread, keyed on selection, curation version and parameters.

use std::sync::Arc;

use egui::{Color32, Sense, Stroke, Ui};
use sorrel_compute::{ccg_baseline, correlograms};
use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider, SampleIndex};

use crate::jobs::{JobCtx, Memo};
use crate::prefs::CcgParams;
use crate::theme::{selection_colour, Theme};
use crate::views::{
    axes, axes_data_rect, caption, fill_height, grid_colour, int_ticks, legend, paint_chip,
    refractory_fill, AxisOpts, TICK_PX,
};

/// phy draws at most this many clusters in the grid.
const MAX_CLUSTERS: usize = 20;
/// phylib `n_spikes_correlograms`: spikes fed to the correlogram, in total.
const MAX_SPIKES: usize = 100_000;

/// One finished computation; carries the parameters it was computed for so
/// a stale grid (shown while a new one computes) still draws consistently.
pub struct CcgData {
    clusters: Vec<ClusterId>,
    bin_ms: f32,
    /// `counts[i][j][k]`: spikes of `j` after a spike of `i` at lag bin `k`.
    counts: Vec<Vec<Vec<u32>>>,
    /// Spikes actually used per cluster (after subsampling).
    n_used: Vec<usize>,
    stride: usize,
    duration_s: f64,
}

type CcgKey = (Vec<ClusterId>, u64, f32, f32);

/// Job-backed correlogram memo (selection, curation version, bin, window).
#[derive(Default)]
pub struct CcgCache {
    memo: Memo<CcgKey, Arc<CcgData>>,
}

impl CcgCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// The grid for `key` if ready, else the previous one for the same
    /// clusters (stale but consistent) plus whether a job is running.
    fn ensure<P: DataProvider>(
        &mut self,
        session: &Session<P>,
        clusters: &[ClusterId],
        p: &CcgParams,
        jobs: &JobCtx<Session<P>>,
    ) -> Option<Arc<CcgData>> {
        let key = (clusters.to_vec(), session.version(), p.bin_ms, p.window_ms);
        let (cl, bin_ms, window_ms) = (clusters.to_vec(), p.bin_ms, p.window_ms);
        let cur = self
            .memo
            .ensure(key, &jobs.session, jobs.repaint(), move |s, _cancel| {
                // ponytail: correlograms() has no cancel hook; a window of
                // seconds at 100k spikes can hold the read lock for a while.
                // Upgrade path: a cancel parameter in the pairing loop.
                let sr = s.provider.sample_rate().max(1.0);
                let total: usize = cl.iter().map(|&c| s.spike_times(c).len()).sum();
                let stride = subsample_stride(total, MAX_SPIKES);
                let trains: Vec<Vec<SampleIndex>> = cl
                    .iter()
                    .map(|&c| s.spike_times(c).iter().copied().step_by(stride).collect())
                    .collect();
                let refs: Vec<&[SampleIndex]> = trains.iter().map(Vec::as_slice).collect();
                Arc::new(CcgData {
                    counts: correlograms(&refs, sr, bin_ms as f64 / 1e3, window_ms as f64 / 1e3),
                    n_used: trains.iter().map(Vec::len).collect(),
                    clusters: cl,
                    bin_ms,
                    stride,
                    duration_s: s.provider.n_samples().0 as f64 / sr as f64,
                })
            })
            .cloned();
        cur.or_else(|| self.memo.last().filter(|d| d.clusters == clusters).cloned())
    }

    pub fn pending(&self) -> bool {
        self.memo.pending()
    }
}

/// Every `stride`-th spike keeps at most `max` of `total` (deterministic).
pub(crate) fn subsample_stride(total: usize, max: usize) -> usize {
    total.div_ceil(max).max(1)
}

/// phy status text: `f'{window:.1f} ms ({bin:.1f} ms)'`.
pub(crate) fn ccg_caption(window_ms: f32, bin_ms: f32) -> String {
    format!("{window_ms:.1} ms ({bin_ms:.1} ms)")
}

/// Keep parameters in a range the compute and the plot can take.
pub(crate) fn clamp_params(p: &mut CcgParams) {
    let fin = |x: f32, lo: f32, hi: f32, d: f32| if x.is_finite() { x.clamp(lo, hi) } else { d };
    p.bin_ms = fin(p.bin_ms, 0.1, 1000.0, 1.0);
    p.window_ms = fin(p.window_ms, 1.0, 5000.0, 50.0).max(p.bin_ms);
    // phy `:cr` accepts 0.1 to 100 ms.
    p.refrac_ms = fin(p.refrac_ms, 0.1, 100.0, 2.0);
}

/// Top of the y axis: the cell's own max (per-cell) or the grid's (uniform),
/// never below the baseline, with 15% headroom.
pub(crate) fn y_top(cell_max: u32, global_max: u32, baseline: f64, uniform: bool) -> f32 {
    let m = if uniform { global_max } else { cell_max };
    (m as f32).max(baseline as f32).max(1.0) * 1.15
}

/// Product of ×1.1 steps (wheel up) / ÷1.1 steps (wheel down) from wheel
/// events while the pane is hovered with Ctrl (`ctrl`) or Alt held. 1.0
/// when idle. Read from raw events: egui turns Ctrl+wheel into zoom.
pub(crate) fn wheel_scale(ui: &Ui, ctrl: bool) -> f32 {
    if !ui.rect_contains_pointer(ui.clip_rect()) {
        return 1.0;
    }
    ui.input(|i| {
        i.events
            .iter()
            .filter_map(|e| match e {
                egui::Event::MouseWheel {
                    delta, modifiers, ..
                } if if ctrl {
                    modifiers.command && !modifiers.alt
                } else {
                    modifiers.alt && !modifiers.command
                } =>
                {
                    let d = if delta.y != 0.0 { delta.y } else { delta.x };
                    Some(if d > 0.0 {
                        1.1
                    } else if d < 0.0 {
                        1.0 / 1.1
                    } else {
                        1.0
                    })
                }
                _ => None,
            })
            .product()
    })
}

fn header(ui: &mut Ui, p: &mut CcgParams) {
    ui.horizontal_wrapped(|ui| {
        ui.label("bin");
        ui.add(
            egui::DragValue::new(&mut p.bin_ms)
                .speed(0.1)
                .range(0.1..=1000.0)
                .suffix(" ms"),
        );
        ui.label("window");
        ui.add(
            egui::DragValue::new(&mut p.window_ms)
                .speed(1.0)
                .range(1.0..=5000.0)
                .suffix(" ms"),
        );
        ui.label("refractory");
        ui.add(
            egui::DragValue::new(&mut p.refrac_ms)
                .speed(0.1)
                .range(0.1..=100.0)
                .suffix(" ms"),
        );
        ui.checkbox(&mut p.uniform, "uniform")
            .on_hover_text("One y scale for every cell instead of per-cell (N)");
    });
}

/// Pairwise correlogram grid for the selected clusters.
///
/// The diagonal renders auto-correlograms (one per cluster); off-diagonal
/// cells render cross-correlograms between row-cluster and column-cluster.
/// Counts follow phylib (see module docs), so they reflect the live
/// (post-curation) bucket assignments.
pub fn correlogram_grid_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    cache: &mut CcgCache,
    jobs: &JobCtx<Session<P>>,
    p: &mut CcgParams,
) {
    let theme = Theme::from_ui(ui);
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }

    // Pane-local keys. Plain `N` is unbound in the global keymap (only
    // Ctrl+N is), so no conflict, but skip it while a text field has focus.
    if ui.rect_contains_pointer(ui.clip_rect()) && !ui.ctx().egui_wants_keyboard_input() {
        let n = ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::N));
        p.uniform ^= n;
    }
    p.window_ms *= wheel_scale(ui, true);
    p.bin_ms *= wheel_scale(ui, false);
    header(ui, p);
    clamp_params(p);

    let shown = &selection[..selection.len().min(MAX_CLUSTERS)];
    if selection.len() > shown.len() {
        ui.weak(format!(
            "showing the first {MAX_CLUSTERS} of {} selected clusters",
            selection.len()
        ));
    }
    let Some(data) = cache.ensure(session, shown, p, jobs) else {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.weak("computing correlograms in the background…");
        });
        return;
    };
    if cache.pending() {
        ui.spinner();
    }
    let n = shown.len();
    let nb = data.counts[0][0].len();
    let half = (nb - 1) / 2;
    let bin_ms = data.bin_ms;
    // Bins are centred on 0, so the axis runs half a bin past the last lag.
    let ext = nb as f32 * bin_ms / 2.0;
    let tick_ms = half as f32 * bin_ms;
    caption(
        ui,
        &theme,
        format!(
            "ACG/CCG · {} · {}",
            ccg_caption(p.window_ms, p.bin_ms),
            legend(shown)
        ),
    );
    let global_max = data
        .counts
        .iter()
        .flatten()
        .flatten()
        .copied()
        .max()
        .unwrap_or(0);
    let bin_s = bin_ms as f64 / 1e3;

    let gap = 6.0;
    let gaps = gap * (n as f32 - 1.0);
    let w_total = ui.available_width().max(120.0);
    let cell_w = (w_total - gaps) / n as f32;
    let cell_h = ((fill_height(ui) - gaps) / n as f32).max(60.0);
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(w_total, cell_h * n as f32 + gaps),
        Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme.plot_bg);

    // Small grids drop their axes and keep only the bars.
    let compact = cell_h < 110.0 || cell_w < 110.0;
    let header_h = 22.0;
    let xt = [-tick_ms, 0.0, tick_ms];
    let hover = resp.hover_pos();
    let mut tip = None;
    for (ri, &c_row) in shown.iter().enumerate() {
        for (ci, &c_col) in shown.iter().enumerate() {
            let cell = egui::Rect::from_min_size(
                egui::pos2(
                    rect.left() + ci as f32 * (cell_w + gap),
                    rect.top() + ri as f32 * (cell_h + gap),
                ),
                egui::vec2(cell_w, cell_h),
            );
            painter.rect_filled(cell, 0.0, theme.panel);

            let h = &data.counts[ri][ci];
            let max_count = h.iter().copied().max().unwrap_or(0);
            let baseline = ccg_baseline(data.n_used[ri], data.n_used[ci], bin_s, data.duration_s);
            let y_top = y_top(max_count, global_max, baseline, p.uniform);
            let colour = selection_colour(ri);

            // Header strip: label chip, plus a sparse note when it fits.
            let chip = paint_chip(
                &painter,
                cell.left_top() + egui::vec2(4.0, 3.0),
                &format!("c{c_row}×c{c_col}"),
                colour,
            );
            if max_count < 5 {
                let note = if ri == ci {
                    format!(
                        "n={} · max {max_count}/bin · sparse",
                        session.spike_times(c_row).len()
                    )
                } else {
                    format!("max {max_count}/bin · sparse")
                };
                let g = painter.layout_no_wrap(
                    note,
                    egui::FontId::proportional(TICK_PX),
                    theme.caption,
                );
                if chip.right() + 8.0 + g.size().x <= cell.right() - 4.0 {
                    painter.galley(
                        egui::pos2(cell.right() - 4.0 - g.size().x, cell.top() + 5.0),
                        g,
                        theme.caption,
                    );
                }
            }

            let body = egui::Rect::from_min_max(
                egui::pos2(cell.left(), cell.top() + header_h),
                cell.right_bottom(),
            );
            let data_rect = if compact {
                body.shrink2(egui::vec2(4.0, 2.0))
            } else {
                let yt = int_ticks(y_top / 1.15, 2);
                axes(
                    &painter,
                    body,
                    (-ext, ext),
                    (0.0, y_top),
                    "ms",
                    "",
                    &theme,
                    AxisOpts {
                        grid: true,
                        x_ticks: Some(&xt),
                        y_ticks: Some(&yt),
                        ..AxisOpts::default()
                    },
                );
                axes_data_rect(body, true)
            };
            let dp = painter.with_clip_rect(data_rect);
            let x_of = |ms: f32| data_rect.left() + (ms + ext) / (2.0 * ext) * data_rect.width();
            // Refractory band (±refrac_ms) and the zero-lag line.
            dp.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x_of(-p.refrac_ms), data_rect.top()),
                    egui::pos2(x_of(p.refrac_ms), data_rect.bottom()),
                ),
                0.0,
                refractory_fill(&theme),
            );
            dp.line_segment(
                [
                    egui::pos2(x_of(0.0), data_rect.top()),
                    egui::pos2(x_of(0.0), data_rect.bottom()),
                ],
                Stroke::new(0.5_f32, grid_colour(&theme)),
            );

            let bw = data_rect.width() / nb as f32;
            for (b_idx, &count) in h.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let bx0 = data_rect.left() + bw * b_idx as f32;
                let by0 = data_rect.bottom() - count as f32 / y_top * data_rect.height();
                dp.rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(bx0, by0),
                        egui::pos2(bx0 + bw * 0.85, data_rect.bottom()),
                    ),
                    0.0,
                    colour,
                );
            }
            // Expected count if the trains were independent.
            if baseline > 0.0 {
                let y = data_rect.bottom() - baseline as f32 / y_top * data_rect.height();
                dp.line_segment(
                    [
                        egui::pos2(data_rect.left(), y),
                        egui::pos2(data_rect.right(), y),
                    ],
                    Stroke::new(1.0_f32, Color32::GRAY),
                );
            }
            if let Some(pos) = hover.filter(|q| data_rect.contains(*q)) {
                let k = (((pos.x - data_rect.left()) / bw) as usize).min(nb - 1);
                let lag = (k as f32 - half as f32) * bin_ms;
                tip = Some(format!(
                    "c{c_row}×c{c_col} · lag {lag:+.1} ms · count {}",
                    h[k]
                ));
            }
        }
    }
    let note = if data.stride > 1 {
        format!(" Every {}th spike used (100k cap).", data.stride)
    } else {
        String::new()
    };
    resp.on_hover_text(tip.unwrap_or_else(|| {
        format!(
            "Diagonal: autocorrelograms. Off-diagonal: cross-correlograms. \
             The shaded band is the ±{:.1} ms refractory period; the grey line is the \
             count expected from independent firing.{note}",
            p.refrac_ms
        )
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_caps_total_at_max() {
        assert_eq!(subsample_stride(0, MAX_SPIKES), 1);
        assert_eq!(subsample_stride(100_000, MAX_SPIKES), 1);
        assert_eq!(subsample_stride(100_001, MAX_SPIKES), 2);
        assert_eq!(subsample_stride(1_000_000, MAX_SPIKES), 10);
        let (total, s) = (250_001usize, subsample_stride(250_001, MAX_SPIKES));
        assert!(total.div_ceil(s) <= MAX_SPIKES);
    }

    #[test]
    fn caption_matches_phy_status() {
        assert_eq!(ccg_caption(50.0, 1.0), "50.0 ms (1.0 ms)");
        assert_eq!(ccg_caption(100.04, 0.25), "100.0 ms (0.2 ms)");
    }

    #[test]
    fn normalisation_per_cell_or_uniform() {
        // Per-cell follows the cell; uniform follows the grid max.
        assert_eq!(y_top(10, 40, 0.0, false), 10.0 * 1.15);
        assert_eq!(y_top(10, 40, 0.0, true), 40.0 * 1.15);
        // Empty cells and baselines above the counts still get a scale.
        assert_eq!(y_top(0, 0, 0.0, false), 1.15);
        assert_eq!(y_top(2, 2, 8.0, false), 8.0 * 1.15);
    }

    #[test]
    fn clamp_keeps_params_sane() {
        let mut p = CcgParams {
            bin_ms: f32::NAN,
            window_ms: 0.0,
            refrac_ms: 1e6,
            uniform: false,
        };
        clamp_params(&mut p);
        assert_eq!((p.bin_ms, p.window_ms, p.refrac_ms), (1.0, 1.0, 100.0));
        p.bin_ms = 20.0;
        p.window_ms = 5.0;
        clamp_params(&mut p);
        assert_eq!(p.window_ms, 20.0, "window never below one bin");
    }
}

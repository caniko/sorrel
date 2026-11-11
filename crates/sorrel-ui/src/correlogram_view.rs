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
use crate::theme::{contrast_text, selection_colour, Theme};
use crate::views::{
    axes, blend, fill_height, grid_colour, int_ticks, param_bar, AxisOpts, TICK_PX, X_STRIP,
    Y_GUTTER,
};

/// phy draws at most this many clusters in the grid.
const MAX_CLUSTERS: usize = 20;
/// phylib `n_spikes_correlograms`: spikes fed to the correlogram, in total.
const MAX_SPIKES: usize = 100_000;
/// Time chunks kept when the selection is over [`MAX_SPIKES`] (phy keeps 20).
const N_CHUNKS: u64 = 20;
/// A kept chunk spans at least this many correlogram windows, so pairs lost
/// at chunk edges stay a small fraction of every lag bin, unless the spike
/// budget allows less time than one such chunk.
const MIN_CHUNK_WINDOWS: u64 = 20;

/// One finished computation; carries the parameters it was computed for so
/// a stale grid (shown while a new one computes) still draws consistently.
pub struct CcgData {
    clusters: Vec<ClusterId>,
    bin_ms: f32,
    /// `counts[i][j][k]`: spikes of `j` after a spike of `i` at lag bin `k`.
    counts: Vec<Vec<Vec<u32>>>,
    /// Spikes actually used per cluster (after subsampling).
    n_used: Vec<usize>,
    /// Fraction of the recording the kept chunks cover (1 = every spike).
    coverage: f64,
    /// Seconds the used spikes come from: the baseline's duration.
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
                let n_samples = s.provider.n_samples().0;
                let window = (window_ms as f64 / 1e3 * sr as f64).ceil() as u64;
                let all: Vec<&[SampleIndex]> = cl.iter().map(|&c| s.spike_times(c)).collect();
                let (chunks, trains) = sample_trains(&all, n_samples, MAX_SPIKES, window);
                let refs: Vec<&[SampleIndex]> = trains.iter().map(Vec::as_slice).collect();
                let covered = chunks
                    .as_ref()
                    .map_or(n_samples, |ch| ch.iter().map(|(a, b)| b - a).sum());
                Arc::new(CcgData {
                    counts: correlograms(&refs, sr, bin_ms as f64 / 1e3, window_ms as f64 / 1e3),
                    n_used: trains.iter().map(Vec::len).collect(),
                    clusters: cl,
                    bin_ms,
                    coverage: covered as f64 / n_samples.max(1) as f64,
                    duration_s: covered as f64 / sr as f64,
                })
            })
            .cloned();
        cur.or_else(|| self.memo.last().filter(|d| d.clusters == clusters).cloned())
    }

    pub fn pending(&self) -> bool {
        self.memo.pending()
    }
}

/// `[start, end)` sample ranges to keep when `total` spikes exceed `max`, or
/// `None` to keep every spike.
///
/// Like phy's spike selector for correlograms, this keeps *every* spike inside
/// a few evenly spaced time chunks rather than every n-th spike. Thinning a
/// train by a stride removes short intervals (consecutive kept spikes are n
/// intervals apart), which empties the bins around lag 0 and makes a
/// contaminated or noise cluster look refractory. Chunks keep all short-lag
/// pairs inside them. The chunks are shared by every cluster in the grid, so
/// cross-correlograms compare the same stretches of time.
pub(crate) fn time_chunks(
    n_samples: u64,
    total: usize,
    max: usize,
    window_samples: u64,
) -> Option<Vec<(u64, u64)>> {
    if total <= max || n_samples == 0 {
        return None;
    }
    let keep = (n_samples as f64 * max as f64 / total as f64) as u64;
    let min_len = window_samples.max(1).saturating_mul(MIN_CHUNK_WINDOWS);
    // Fewer, longer chunks when the kept time is short, never below one.
    let k = N_CHUNKS.min(keep / min_len.max(1)).max(1);
    let len = (keep / k).clamp(1, n_samples);
    let stride = n_samples / k;
    Some(
        (0..k)
            .map(|i| {
                // Centre each chunk in its share of the recording.
                let start = (i * stride + stride.saturating_sub(len) / 2).min(n_samples - len);
                (start, start + len)
            })
            .collect(),
    )
}

/// `[start, end)` sample ranges, ascending.
type Chunks = Vec<(u64, u64)>;

/// `trains` restricted to shared [`time_chunks`] holding at most `max` spikes
/// in total. Chunks are sized from the average rate, so when bursts put more
/// than `max` spikes inside them the kept time shrinks by the overshoot.
pub(crate) fn sample_trains(
    trains: &[&[SampleIndex]],
    n_samples: u64,
    max: usize,
    window_samples: u64,
) -> (Option<Chunks>, Vec<Vec<SampleIndex>>) {
    let total: usize = trains.iter().map(|t| t.len()).sum();
    let mut budget = max;
    loop {
        let chunks = time_chunks(n_samples, total, budget, window_samples);
        // Count by binary search; copy only the accepted chunks.
        let n: usize = match chunks.as_deref() {
            None => total,
            Some(ch) => trains
                .iter()
                .flat_map(|t| ch.iter().map(move |&(a, b)| count_in(t, a, b)))
                .sum(),
        };
        if n <= max || budget <= 1 {
            let kept = trains
                .iter()
                .map(|t| in_chunks(t, chunks.as_deref()))
                .collect();
            return (chunks, thin_to_cap(kept, max));
        }
        budget = next_budget(budget, max, n);
    }
}

/// The cap is a hard bound: a burst denser than `max` inside a chunk the
/// budget cannot shrink any further (down to one sample) still holds more
/// spikes than advertised. Keep an evenly spaced `max` of the survivors,
/// spread across the trains in proportion to their size.
fn thin_to_cap(kept: Vec<Vec<SampleIndex>>, max: usize) -> Vec<Vec<SampleIndex>> {
    let total: usize = kept.iter().map(Vec::len).sum();
    if total <= max {
        return kept;
    }
    let mut seen = 0usize;
    kept.into_iter()
        .map(|train| {
            train
                .into_iter()
                .filter(|_| {
                    // Bresenham selection: exactly `max` of `total`, in order.
                    let pick = seen * max / total != (seen + 1) * max / total;
                    seen += 1;
                    pick
                })
                .collect()
        })
        .collect()
}

/// The budget after a pass kept `n > max` spikes: shrunk by the overshoot and
/// by at least a tenth, so a near-cap burst takes a few dozen passes, not one
/// per spike.
fn next_budget(budget: usize, max: usize, n: usize) -> usize {
    ((budget as f64 * max as f64 / n as f64) as usize).min(budget * 9 / 10)
}

/// Spikes of an ascending train in `[a, b)`.
fn count_in(times: &[SampleIndex], a: u64, b: u64) -> usize {
    times.partition_point(|t| t.0 < b) - times.partition_point(|t| t.0 < a)
}

/// The spikes of an ascending train that fall inside `chunks` (all of them
/// when `chunks` is `None`).
pub(crate) fn in_chunks(times: &[SampleIndex], chunks: Option<&[(u64, u64)]>) -> Vec<SampleIndex> {
    let Some(chunks) = chunks else {
        return times.to_vec();
    };
    let mut out = Vec::new();
    for &(a, b) in chunks {
        let lo = times.partition_point(|t| t.0 < a);
        let hi = times.partition_point(|t| t.0 < b);
        out.extend_from_slice(&times[lo..hi]);
    }
    out
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

/// Cells are never smaller than this; past it the grid scrolls.
const MIN_CELL: f32 = 120.0;

/// Refractory band: stronger on the light background.
pub(crate) fn band_fill(theme: &Theme) -> Color32 {
    theme
        .danger
        .gamma_multiply(if theme.dark { 0.30 } else { 0.35 })
}

/// Independent-firing baseline: stronger than the gridlines in both themes.
fn baseline_colour(theme: &Theme) -> Color32 {
    blend(
        theme.plot_bg,
        theme.text,
        if theme.dark { 0.75 } else { 0.9 },
    )
}

/// Rounded label chip with its top-left at `pos`; text contrasts the fill.
fn chip(painter: &egui::Painter, pos: egui::Pos2, text: &str, bg: Color32) -> egui::Rect {
    let fg = contrast_text(bg);
    let g = painter.layout_no_wrap(text.to_owned(), egui::FontId::proportional(TICK_PX), fg);
    let r = egui::Rect::from_min_size(pos, g.size() + egui::vec2(10.0, 4.0));
    painter.rect_filled(r, 3.0, bg);
    painter.galley(r.min + egui::vec2(5.0, 2.0), g, fg);
    r
}

/// One row: window inline, the rest behind the gear.
fn header(ui: &mut Ui, p: &mut CcgParams) {
    let caption = format!("ACG/CCG · {}", ccg_caption(p.window_ms, p.bin_ms));
    let (window, bin, refrac, uniform) = (
        &mut p.window_ms,
        &mut p.bin_ms,
        &mut p.refrac_ms,
        &mut p.uniform,
    );
    param_bar(
        ui,
        &caption,
        |ui| {
            ui.label("window");
            ui.add(
                egui::DragValue::new(window)
                    .speed(1.0)
                    .range(1.0..=5000.0)
                    .suffix(" ms"),
            );
        },
        |ui| {
            egui::Grid::new("ccg_params").show(ui, |ui| {
                ui.label("bin");
                ui.add(
                    egui::DragValue::new(bin)
                        .speed(0.1)
                        .range(0.1..=1000.0)
                        .suffix(" ms"),
                );
                ui.end_row();
                ui.label("refractory");
                ui.add(
                    egui::DragValue::new(refrac)
                        .speed(0.1)
                        .range(0.1..=100.0)
                        .suffix(" ms"),
                );
                ui.end_row();
            });
            ui.checkbox(uniform, "uniform").on_hover_text(format!(
                "One y scale for every cell instead of per-cell ({})",
                crate::settings::hint(ui.ctx(), "ccg.normalize")
            ));
        },
    );
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
    brushes: &crate::brush::Brushes,
) {
    let theme = Theme::from_ui(ui);
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }

    // Pane-local keys. Plain `N` is unbound in the global keymap (only
    // Ctrl+N is), so no conflict, but skip it while a text field has focus.
    if ui.rect_contains_pointer(ui.clip_rect()) && !ui.ctx().egui_wants_keyboard_input() {
        let n = crate::settings::consume(ui.ctx(), crate::shortcuts::Scope::Ccg)
            .contains(&crate::shortcuts::Action::CcgNormalize);
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
    // A merge candidate hovered in Similar / Suggest joins as a grey ghost
    // row and column (last, so it is part of the cache key's cluster list).
    let n_real = shown.len();
    let ghost = match brushes.ghost {
        Some(crate::brush::Ghost::Merge(g)) if n_real < MAX_CLUSTERS && !shown.contains(&g) => {
            Some(g)
        }
        _ => None,
    };
    let with_ghost: Vec<ClusterId> = shown.iter().copied().chain(ghost).collect();
    let shown = &with_ghost[..];
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
    let global_max = data
        .counts
        .iter()
        .flatten()
        .flatten()
        .copied()
        .max()
        .unwrap_or(0);
    let bin_s = bin_ms as f64 / 1e3;

    // Every cell's *data area* is dw x dh; the left column adds the y
    // gutter and the bottom row the x strip, so axes labels cost the other
    // cells nothing. Cells are at least MIN_CELL; past that the grid scrolls.
    let gap = 6.0;
    let gaps = gap * (n as f32 - 1.0);
    let header_h = 22.0;
    let pane_w = ui.available_width().max(MIN_CELL);
    let pane_h = fill_height(ui);
    let dw = ((pane_w - gaps - Y_GUTTER) / n as f32).max(MIN_CELL);
    let dh = ((pane_h - gaps - X_STRIP) / n as f32 - header_h).max(MIN_CELL - header_h);
    let total = egui::vec2(
        dw * n as f32 + Y_GUTTER + gaps,
        (header_h + dh) * n as f32 + X_STRIP + gaps,
    );

    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            grid(
                ui,
                &theme,
                &data,
                shown,
                session,
                p,
                total,
                (dw, dh),
                (gap, header_h),
                ext,
                tick_ms,
                global_max,
                bin_s,
                half,
                n_real,
            );
        });
}

/// Paint the grid into a `total`-sized allocation (see the sizing above).
#[allow(clippy::too_many_arguments)]
fn grid<P: DataProvider>(
    ui: &mut Ui,
    theme: &Theme,
    data: &CcgData,
    shown: &[ClusterId],
    session: &Session<P>,
    p: &CcgParams,
    total: egui::Vec2,
    (dw, dh): (f32, f32),
    (gap, header_h): (f32, f32),
    ext: f32,
    tick_ms: f32,
    global_max: u32,
    bin_s: f64,
    half: usize,
    n_real: usize,
) {
    let n = shown.len();
    let nb = data.counts[0][0].len();
    let bin_ms = data.bin_ms;
    let (rect, resp) = ui.allocate_exact_size(total, Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme.plot_bg);

    let xt = [-tick_ms, 0.0, tick_ms];
    let hover = resp.hover_pos();
    let mut tip = None;
    let ghost_grey = Color32::from_gray(if theme.dark { 130 } else { 150 });
    let name = |c: ClusterId, i: usize| {
        if i >= n_real {
            format!("c{c} (ghost)")
        } else {
            format!("c{c}")
        }
    };
    for (ri, &c_row) in shown.iter().enumerate() {
        for (ci, &c_col) in shown.iter().enumerate() {
            let (left_col, bottom_row) = (ci == 0, ri == n - 1);
            let x0 = if left_col {
                rect.left()
            } else {
                rect.left() + Y_GUTTER + ci as f32 * (dw + gap)
            };
            let cell = egui::Rect::from_min_size(
                egui::pos2(x0, rect.top() + ri as f32 * (header_h + dh + gap)),
                egui::vec2(
                    dw + if left_col { Y_GUTTER } else { 0.0 },
                    header_h + dh + if bottom_row { X_STRIP } else { 0.0 },
                ),
            );
            painter.rect_filled(cell, 0.0, theme.panel);

            let h = &data.counts[ri][ci];
            let max_count = h.iter().copied().max().unwrap_or(0);
            let baseline = ccg_baseline(data.n_used[ri], data.n_used[ci], bin_s, data.duration_s);
            let y_top = y_top(max_count, global_max, baseline, p.uniform);
            let colour = if ri >= n_real {
                ghost_grey
            } else {
                selection_colour(ri)
            };

            // Header strip: label chip, plus a sparse note when it fits.
            let chip = chip(
                &painter,
                cell.left_top() + egui::vec2(4.0, 3.0),
                &format!("{}×{}", name(c_row, ri), name(c_col, ci)),
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

            let data_rect = egui::Rect::from_min_size(
                egui::pos2(
                    cell.left() + if left_col { Y_GUTTER } else { 0.0 },
                    cell.top() + header_h,
                ),
                egui::vec2(dw, dh),
            );
            // `axes` places its gutters around the data area it is given.
            let axes_rect = egui::Rect::from_min_max(
                egui::pos2(
                    data_rect.left() - if left_col { Y_GUTTER } else { 0.0 },
                    data_rect.top(),
                ),
                egui::pos2(data_rect.right(), data_rect.bottom() + X_STRIP),
            );
            let yt = int_ticks(y_top / 1.15, 2);
            // Labels only on the left column (y) and bottom row (lag); an
            // empty y range drops the y axis, so the gridlines are ours.
            axes(
                &painter,
                axes_rect,
                (-ext, ext),
                if left_col { (0.0, y_top) } else { (0.0, 0.0) },
                "ms",
                "",
                theme,
                AxisOpts {
                    grid: true,
                    hide_x: !bottom_row,
                    x_ticks: Some(&xt),
                    y_ticks: Some(&yt),
                    ..AxisOpts::default()
                },
            );
            if !left_col {
                for &v in &yt {
                    let y = data_rect.bottom() - v / y_top * data_rect.height();
                    painter.line_segment(
                        [
                            egui::pos2(data_rect.left(), y),
                            egui::pos2(data_rect.right(), y),
                        ],
                        Stroke::new(0.5_f32, grid_colour(theme)),
                    );
                }
            }
            let dp = painter.with_clip_rect(data_rect);
            let x_of = |ms: f32| data_rect.left() + (ms + ext) / (2.0 * ext) * data_rect.width();
            // Refractory band (±refrac_ms) and the zero-lag line.
            dp.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x_of(-p.refrac_ms), data_rect.top()),
                    egui::pos2(x_of(p.refrac_ms), data_rect.bottom()),
                ),
                0.0,
                band_fill(theme),
            );
            dp.line_segment(
                [
                    egui::pos2(x_of(0.0), data_rect.top()),
                    egui::pos2(x_of(0.0), data_rect.bottom()),
                ],
                Stroke::new(0.5_f32, grid_colour(theme)),
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
                    Stroke::new(1.5_f32, baseline_colour(theme)),
                );
            }
            if let Some(pos) = hover.filter(|q| data_rect.contains(*q)) {
                let k = (((pos.x - data_rect.left()) / bw) as usize).min(nb - 1);
                let lag = (k as f32 - half as f32) * bin_ms;
                tip = Some(format!(
                    "{}×{} · lag {lag:+.1} ms · count {}",
                    name(c_row, ri),
                    name(c_col, ci),
                    h[k]
                ));
            }
        }
    }
    let note = if data.coverage < 1.0 {
        format!(
            " All spikes in {:.0}% of the recording, in time chunks (100k cap).",
            data.coverage * 100.0
        )
    } else {
        String::new()
    };
    resp.on_hover_text(tip.unwrap_or_else(|| {
        format!(
            "Diagonal: autocorrelograms. Off-diagonal: cross-correlograms. \
             The shaded band is the ±{:.1} ms refractory period; the dark line is the \
             count expected from independent firing.{note}",
            p.refrac_ms
        )
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn train(n: u64, step: u64) -> Vec<SampleIndex> {
        (0..n).map(|i| SampleIndex(i * step)).collect()
    }

    #[test]
    fn small_selections_keep_every_spike() {
        assert_eq!(time_chunks(1_000_000, 100_000, MAX_SPIKES, 1600), None);
        let t = train(10, 7);
        assert_eq!(in_chunks(&t, None), t);
    }

    #[test]
    fn chunks_cap_a_uniform_train_near_max() {
        // 1M spikes, one every 300 samples, 50 ms window at 32 kHz.
        let t = train(1_000_000, 300);
        let n = 300 * 1_000_000;
        let ch = time_chunks(n, t.len(), MAX_SPIKES, 1600).unwrap();
        assert_eq!(ch.len() as u64, N_CHUNKS);
        assert!(ch.windows(2).all(|w| w[0].1 <= w[1].0), "chunks overlap");
        let kept = in_chunks(&t, Some(&ch)).len();
        assert!((90_000..=110_000).contains(&kept), "kept {kept}");
    }

    #[test]
    fn short_recordings_use_fewer_longer_chunks() {
        // 100k kept samples fit only 3 chunks of 20 windows (32k samples).
        let ch = time_chunks(1_000_000, 1_000_000, MAX_SPIKES, 1600).unwrap();
        assert_eq!(ch.len(), 3);
        assert!(ch.iter().all(|c| c.1 - c.0 >= 1600 * MIN_CHUNK_WINDOWS));
        assert!(ch.windows(2).all(|w| w[0].1 <= w[1].0) && ch[2].1 <= 1_000_000);
        // Kept time under one minimum chunk: one chunk of just the kept time.
        let ch = time_chunks(20_000, 1_000_000, MAX_SPIKES, 1600).unwrap();
        assert_eq!(ch, vec![(9_000, 11_000)]);
    }

    #[test]
    fn long_windows_on_short_recordings_stay_under_the_cap() {
        // 1M spikes in 2M samples (about a minute) with a 5 s window.
        let t = train(1_000_000, 2);
        let (ch, kept) = sample_trains(&[&t], 2_000_000, MAX_SPIKES, 150_000);
        assert!(ch.is_some());
        assert!(kept[0].len() <= MAX_SPIKES, "kept {}", kept[0].len());
    }

    #[test]
    fn bursts_inside_the_chunks_stay_under_the_cap() {
        // A uniform background plus a 200k-spike burst inside the first chunk.
        let mut t = train(1_000_000, 20);
        t.extend((0..200_000u64).map(|i| SampleIndex(500_000 + i % 1000)));
        t.sort_unstable();
        let n = 20_000_000;
        let naive = time_chunks(n, t.len(), MAX_SPIKES, 1600);
        assert!(in_chunks(&t, naive.as_deref()).len() > MAX_SPIKES);
        let (_, kept) = sample_trains(&[&t], n, MAX_SPIKES, 1600);
        assert!(kept[0].len() <= MAX_SPIKES, "kept {}", kept[0].len());
    }

    #[test]
    fn a_burst_denser_than_the_cap_is_still_capped() {
        // 250k coincident detections at one sample: no chunk set can thin
        // them, so the budget bottoms out at one sample and the cap must
        // still hold.
        let n_samples = 1_000_000;
        let t = vec![SampleIndex(500_000); 250_000];
        let (_, kept) = sample_trains(&[&t], n_samples, MAX_SPIKES, 150_000);
        assert!(
            kept[0].len() <= MAX_SPIKES,
            "kept {} over the hard cap",
            kept[0].len()
        );
    }

    #[test]
    fn a_near_cap_burst_shrinks_the_budget_geometrically() {
        // One spike over the cap must not cost one pass per spike.
        assert!(next_budget(MAX_SPIKES, MAX_SPIKES, MAX_SPIKES + 1) <= MAX_SPIKES * 9 / 10);
        assert_eq!(next_budget(2, MAX_SPIKES, MAX_SPIKES + 1), 1);
        // 20 trains stacked on one burst, 100,001 spikes in total.
        let trains: Vec<Vec<SampleIndex>> = (0..20)
            .map(|i| {
                let n = if i == 0 { 5_001 } else { 5_000 };
                (0..n).map(|j| SampleIndex(52_000_000 + 200 * j)).collect()
            })
            .collect();
        let refs: Vec<&[SampleIndex]> = trains.iter().map(Vec::as_slice).collect();
        let (_, kept) = sample_trains(&refs, 100_000_000, MAX_SPIKES, 150_000);
        let n: usize = kept.iter().map(Vec::len).sum();
        assert!(n <= MAX_SPIKES, "kept {n}");
    }

    /// The regression: a cluster with many near-coincident (duplicate)
    /// detections must keep its lag-0 peak after subsampling. Thinning by a
    /// stride erased it and drew a fake refractory dip.
    #[test]
    fn chunked_acg_keeps_the_zero_lag_peak() {
        // Pairs of spikes 3 samples apart every 400 samples: 1M spikes.
        let t: Vec<SampleIndex> = (0..500_000u64)
            .flat_map(|i| [SampleIndex(i * 400), SampleIndex(i * 400 + 3)])
            .collect();
        let n = 400 * 500_000;
        let (sr, window) = (32_000.0, 0.05);
        let ch = time_chunks(n, t.len(), MAX_SPIKES, (window * sr as f64) as u64);
        let kept = in_chunks(&t, ch.as_deref());
        assert!(kept.len() <= 2 * MAX_SPIKES);
        let acg = &correlograms(&[&kept], sr, 0.001, window)[0][0];
        let half = acg.len() / 2;
        let flank = acg[half + 5] as f64;
        assert!(
            acg[half] as f64 > 5.0 * flank.max(1.0),
            "lag-0 {} vs {flank}",
            acg[half]
        );
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

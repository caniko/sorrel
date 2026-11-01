//! phy ISIView: inter-spike-interval histograms, overlaid per cluster.
//!
//! One plot, one slot-coloured step outline per cluster. `n_bins` bins span
//! exactly `[x_min, x_max)` (`n_bins + 1` edges; phy's `linspace` loses one).

use std::sync::Arc;

use egui::{Sense, Shape, Stroke, Ui};
use sorrel_compute::{isi_histogram_range, isi_violations};
use sorrel_data::{preview_split, Session, SplitPreview};
use sorrel_io::{ClusterId, DataProvider, SampleIndex};

use crate::brush::{BrushSource, Brushes, SpikeBrush};
use crate::correlogram_view::{band_fill, wheel_scale};
use crate::prefs::IsiParams;
use crate::theme::{contrast_text, selection_colour, Theme};
use crate::views::{axes, axes_data_rect, fill_height, int_ticks, param_bar, AxisOpts, TICK_PX};

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

/// The shaded band's width and the violation legend's window.
const REFRACTORY_MS: f32 = 1.0;

/// Local indices of every spike that is part of an ISI below `refractory`
/// samples (both members of each pair, ascending), and the pair count.
pub(crate) fn isi_violators(times: &[SampleIndex], refractory: u64) -> (Vec<u32>, usize) {
    let mut out: Vec<u32> = Vec::new();
    let mut pairs = 0;
    for (i, w) in times.windows(2).enumerate() {
        if w[1].0.saturating_sub(w[0].0) < refractory {
            pairs += 1;
            // Consecutive pairs share a member: push it once.
            if out.last() != Some(&(i as u32)) {
                out.push(i as u32);
            }
            out.push(i as u32 + 1);
        }
    }
    (out, pairs)
}

/// The cluster the split what-if is about: the first selected cluster that
/// the spike brush covers, when the brush is a lasso / k-means part (not ISI
/// violators) and leaves at least one spike outside. Returns the brushed
/// local indices.
pub(crate) fn split_target(
    brushes: &Brushes,
    selection: &[ClusterId],
    session_n: impl Fn(ClusterId) -> usize,
) -> Option<(ClusterId, std::sync::Arc<[u32]>)> {
    let b = brushes.spikes.as_ref()?;
    if b.source == BrushSource::IsiViolators {
        return None;
    }
    selection.iter().find_map(|&c| {
        let (_, idx) = b.spikes.iter().find(|(bc, _)| *bc == c)?;
        (idx.len() < session_n(c)).then(|| (c, std::sync::Arc::clone(idx)))
    })
}

type SplitMemo =
    std::sync::Arc<std::sync::Mutex<crate::jobs::Memo<SplitKey, std::sync::Arc<SplitPreview>>>>;
type SplitKey = ((BrushSource, u64, usize, usize), ClusterId);

/// The what-if for the current brush, computed off the UI thread. Shared by
/// the ISI and Quality panes (one memo in egui's temp memory, keyed by the
/// brush). `None` while it computes or when no brush qualifies.
/// ponytail: not cancelled before a curation write (the Memo is not owned by
/// the app); the job is O(spikes of one cluster) and holds a read lock briefly.
pub(crate) fn split_preview<P: DataProvider>(
    ui: &Ui,
    session: &Session<P>,
    jobs: &crate::jobs::JobCtx<Session<P>>,
    brushes: &Brushes,
    selection: &[ClusterId],
) -> Option<std::sync::Arc<SplitPreview>> {
    let (c, idx) = split_target(brushes, selection, |c| session.spike_times(c).len())?;
    let key = (brushes.spikes.as_ref()?.key(), c);
    let id = egui::Id::new("split_whatif_memo");
    let memo: SplitMemo = ui.data_mut(|d| {
        d.get_temp_mut_or_insert_with(id, SplitMemo::default)
            .clone()
    });
    let mut memo = memo.lock().unwrap_or_else(|p| p.into_inner());
    memo.ensure(key, &jobs.session, jobs.repaint(), move |s, _| {
        std::sync::Arc::new(preview_split(s, c, &idx))
    })
    .cloned()
}

/// Refractory share of a spike train, for captions.
fn refractory_pct(times: &[SampleIndex], refractory: u64) -> f32 {
    if times.is_empty() {
        0.0
    } else {
        isi_violations(times, refractory) as f32 / times.len() as f32 * 100.0
    }
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
    brushes: &mut Brushes,
    jobs: &crate::jobs::JobCtx<Session<P>>,
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
    let refractory_samples = (sr * REFRACTORY_MS / 1e3).round() as u64;
    let mut clicked: Vec<ClusterId> = Vec::new();
    ui.horizontal_wrapped(|ui| {
        for (slot, &c) in shown.iter().enumerate() {
            let n = session.spike_times(c).len();
            let viol = isi_violations(session.spike_times(c), refractory_samples);
            let pct = if n == 0 {
                0.0
            } else {
                viol as f32 / n as f32 * 100.0
            };
            let (bg, text) = (
                selection_colour(slot),
                format!("c{c} <1 ms {viol}/{n} ({pct:.1}%)"),
            );
            let fg = contrast_text(bg);
            let g = ui
                .painter()
                .layout_no_wrap(text, egui::FontId::proportional(TICK_PX), fg);
            let (r, resp) =
                ui.allocate_exact_size(g.size() + egui::vec2(10.0, 4.0), Sense::click());
            ui.painter().rect_filled(r, 3.0, bg);
            ui.painter().galley(r.min + egui::vec2(5.0, 2.0), g, fg);
            resp.widget_info(|| {
                egui::WidgetInfo::labeled(
                    egui::WidgetType::Button,
                    true,
                    format!("isi violators c{c}"),
                )
            });
            if resp
                .on_hover_text("Click to brush this cluster's violating spikes")
                .clicked()
            {
                clicked.push(c);
            }
        }
    });

    let split =
        split_preview(ui, session, jobs, brushes, shown).filter(|sp| shown.contains(&sp.cluster));
    if let Some(sp) = &split {
        // Above the plot: the plot fills what is left of the pane.
        ui.weak(format!(
            "if split: inside {:.1} % · outside {:.1} % refractory  (dashed inside · dotted outside)",
            refractory_pct(&sp.inside_times, refractory_samples),
            refractory_pct(&sp.outside_times, refractory_samples),
        ));
    }
    let (rect, _resp) = ui.allocate_exact_size(
        egui::vec2(ui.available_width().max(120.0), fill_height(ui).max(160.0)),
        Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme.plot_bg);
    let data = axes_data_rect(rect, true);
    let max_count = hists.iter().flatten().copied().max().unwrap_or(1).max(1) as f32;
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
        // The band is a button: click brushes every violating spike of the
        // shown clusters.
        let band_rect =
            egui::Rect::from_min_max(egui::pos2(x0, data.top()), egui::pos2(x1, data.bottom()));
        let resp = ui.interact(band_rect, ui.id().with("isi_band"), Sense::click());
        resp.widget_info(|| {
            egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "isi violators")
        });
        let all: Vec<(ClusterId, Vec<u32>, usize)> = shown
            .iter()
            .map(|&c| {
                let (v, pairs) = isi_violators(session.spike_times(c), refractory_samples);
                (c, v, pairs)
            })
            .collect();
        let picked = brushes
            .spikes
            .as_ref()
            .is_some_and(|b| b.source == BrushSource::IsiViolators);
        if resp.hovered() || picked {
            dp.rect_stroke(
                band_rect,
                0.0,
                Stroke::new(1.5_f32, theme.text),
                egui::StrokeKind::Inside,
            );
        }
        let (n, m) = all
            .iter()
            .fold((0, 0), |a, (_, v, p)| (a.0 + v.len(), a.1 + p));
        let resp = resp.on_hover_text(format!(
            "{n} spikes in {m} violating pairs · click to brush"
        ));
        if resp.clicked() {
            brush_violators(
                brushes,
                session.version(),
                all.into_iter().map(|(c, v, _)| (c, v)),
            );
        }
    }

    if !clicked.is_empty() {
        brush_violators(
            brushes,
            session.version(),
            clicked.iter().map(|&c| {
                (
                    c,
                    isi_violators(session.spike_times(c), refractory_samples).0,
                )
            }),
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

    // Split what-if: inside dashed, outside dotted, in the cluster colour,
    // each part's own counts on the same axis.
    if let Some(sp) = split {
        if let Some(slot) = shown.iter().position(|&c| c == sp.cluster) {
            let colour = selection_colour(slot);
            let step = |h: &[u32]| {
                let mut pts = Vec::with_capacity(h.len() * 2);
                for (i, &count) in h.iter().enumerate() {
                    let x0 = data.left() + bw * i as f32;
                    pts.push(egui::pos2(x0, y_of(count)));
                    pts.push(egui::pos2(x0 + bw, y_of(count)));
                }
                pts
            };
            // Parts' histograms, cached per preview and bin range.
            let hid = egui::Id::new("isi_split_hists");
            let hkey = (Arc::as_ptr(&sp) as usize, lo, hi, p.n_bins);
            let h = match ui
                .data(|d| d.get_temp::<((usize, u64, u64, usize), Arc<(Vec<u32>, Vec<u32>)>)>(hid))
            {
                Some((k, h)) if k == hkey => h,
                _ => {
                    let h = Arc::new((
                        isi_histogram_range(&sp.inside_times, lo, hi, p.n_bins),
                        isi_histogram_range(&sp.outside_times, lo, hi, p.n_bins),
                    ));
                    ui.data_mut(|d| d.insert_temp(hid, (hkey, Arc::clone(&h))));
                    h
                }
            };
            dp.extend(Shape::dashed_line(
                &step(&h.0),
                Stroke::new(2.0_f32, colour),
                6.0,
                4.0,
            ));
            dp.extend(Shape::dotted_line(&step(&h.1), colour, 4.0, 1.2));
        }
    }
}

fn brush_violators(
    brushes: &mut Brushes,
    version: u64,
    per_cluster: impl Iterator<Item = (ClusterId, Vec<u32>)>,
) {
    // No violators (a clean cluster): keep whatever brush is up.
    if let Some(b) = SpikeBrush::new(BrushSource::IsiViolators, version, per_cluster) {
        brushes.spikes = Some(b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn violators_include_both_members_once() {
        let t = |v: &[u64]| v.iter().map(|&x| SampleIndex(x)).collect::<Vec<_>>();
        // Refractory 10: pairs (0,1), (3,4) and (4,5) are < 10 apart; 4 is shared.
        let times = t(&[0, 5, 100, 200, 203, 207, 400]);
        let (v, pairs) = isi_violators(&times, 10);
        assert_eq!((v, pairs), (vec![0, 1, 3, 4, 5], 3));
        assert_eq!(isi_violators(&times, 3), (vec![], 0), "strictly below");
        assert_eq!(isi_violators(&times[..1], 10), (vec![], 0));
        assert_eq!(isi_violations(&times, 10), 3, "same rule as the legend");
    }

    #[test]
    fn split_target_needs_a_part_brush_and_a_remainder() {
        let (a, b) = (ClusterId(1), ClusterId(2));
        let mk = |src, c, idx: Vec<u32>| Brushes {
            spikes: SpikeBrush::new(src, 0, [(c, idx)]),
            ..Default::default()
        };
        let n = |_| 10;
        let br = mk(BrushSource::FeatureLasso, b, vec![1, 2]);
        assert_eq!(split_target(&br, &[a, b], n).map(|t| t.0), Some(b));
        assert!(split_target(&br, &[a], n).is_none(), "cluster not selected");
        let isi = mk(BrushSource::IsiViolators, b, vec![1, 2]);
        assert!(split_target(&isi, &[b], n).is_none());
        let all = mk(BrushSource::KMeansPart, b, (0..10).collect());
        assert!(
            split_target(&all, &[b], n).is_none(),
            "nothing left outside"
        );
    }

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

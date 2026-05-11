use crate::selection::SelectionSet;
use crate::wgpu_trace::TraceCallback;
use egui::{Color32, Rect, Sense, Stroke, Ui, Vec2};
use sorrel_compute::quality_breakdown;
use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider};
use sorrel_render::{
    build_trace_vertices_cfg, build_trace_vertices_gpu, GpuTracePreproc, TracePreproc, TraceVertex,
};

/// Sortable column on the cluster table. The widget mutates the table state
/// on header clicks, so this stays in lockstep with the column rendering.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ClusterColumn {
    Id,
    SpikeCount,
    Amplitude,
    IsiViolations,
    Quality,
    Label,
}

/// Persistent cluster-table UI state — sort + filter live on the app.
#[derive(Clone, Debug)]
pub struct ClusterTableState {
    pub sort_by: ClusterColumn,
    pub sort_descending: bool,
    pub filter_text: String,
    pub hide_empty: bool,
    pub hide_noise: bool,
    /// Cached "quick quality" composite score per cluster — uses only the
    /// cheap metrics (no PC-feature isolation k-NN) so the table sort is
    /// snappy even on 1k-cluster recordings. Indexed by `ClusterId`.
    /// Recomputed when [`Self::quality_key`] no longer matches.
    quality_cache: Vec<f32>,
    /// Snapshot of `(history_len, redo_len, n_clusters)` at the time the
    /// cache was built. Any change invalidates the whole cache.
    quality_key: (usize, usize, u32),
}

impl Default for ClusterTableState {
    fn default() -> Self {
        Self {
            sort_by: ClusterColumn::Id,
            sort_descending: false,
            filter_text: String::new(),
            hide_empty: false,
            hide_noise: false,
            quality_cache: Vec::new(),
            quality_key: (usize::MAX, usize::MAX, u32::MAX),
        }
    }
}

impl ClusterTableState {
    /// Refresh the quality cache if the curation history has moved.
    fn ensure_quality<P: DataProvider>(&mut self, session: &Session<P>) {
        let key = (
            session.history_len(),
            session.redo_len(),
            session.n_clusters(),
        );
        if key == self.quality_key && !self.quality_cache.is_empty() {
            return;
        }
        let sr = session.provider.sample_rate().max(1.0);
        let refractory_samples = (sr * 0.0015).round() as u64;
        let total_duration = session.provider.n_samples();
        // Parallel per-cluster compute. `&Session<P>` is Send + Sync, so
        // rayon can borrow it directly across the worker threads.
        use rayon::prelude::*;
        let out: Vec<f32> = (0..session.n_clusters())
            .into_par_iter()
            .map(ClusterId)
            .map(|c| {
                let q = quality_breakdown(
                    session.spike_times(c),
                    session.spike_amplitudes(c),
                    refractory_samples,
                    total_duration.0,
                    sr,
                    50,
                );
                q.composite()
            })
            .collect();
        self.quality_cache = out;
        self.quality_key = key;
    }

    fn quality_for(&self, c: ClusterId) -> f32 {
        self.quality_cache.get(c.idx()).copied().unwrap_or(f32::NAN)
    }
}

impl ClusterTableState {
    fn toggle_sort(&mut self, col: ClusterColumn) {
        if self.sort_by == col {
            self.sort_descending = !self.sort_descending;
        } else {
            self.sort_by = col;
            self.sort_descending = matches!(
                col,
                ClusterColumn::SpikeCount
                    | ClusterColumn::Amplitude
                    | ClusterColumn::IsiViolations
                    | ClusterColumn::Quality
            );
        }
    }
}

/// Cluster table with click-to-sort headers and a small filter row above.
///
/// Multi-select rules match phy: plain click replaces, Cmd/Ctrl-click
/// toggles, Shift-click range-extends from the anchor.
pub fn cluster_table<P, F>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &mut SelectionSet,
    state: &mut ClusterTableState,
    label_str: F,
) where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    use egui_extras::{Column, TableBuilder};
    use sorrel_compute::{isi_violations, mean_amplitude};

    // Heuristic refractory window: 1 ms — phy's default for ACG view's
    // shaded zone. We translate to samples using the session's sample rate.
    let refractory_samples = (session.provider.sample_rate() * 0.001).round() as u64;

    state.ensure_quality(session);

    // Filter strip above the table.
    ui.horizontal(|ui| {
        ui.label("filter id:");
        ui.add(
            egui::TextEdit::singleline(&mut state.filter_text)
                .desired_width(60.0)
                .hint_text(""),
        );
        ui.checkbox(&mut state.hide_empty, "hide empty");
        ui.checkbox(&mut state.hide_noise, "hide noise");
        let total_visible = "(filter applied)";
        ui.weak(total_visible);
    });

    // Compute the list of visible cluster ids per the filter, then sort.
    let n = session.n_clusters();
    let filter = state.filter_text.trim();
    let mut visible: Vec<ClusterId> = (0..n)
        .map(ClusterId)
        .filter(|&i| {
            if state.hide_empty && session.spike_times(i).is_empty() {
                return false;
            }
            let lbl = session.label(i).unwrap_or_default();
            if state.hide_noise && label_str(&lbl) == "noise" {
                return false;
            }
            if !filter.is_empty() && !i.to_string().contains(filter) {
                return false;
            }
            true
        })
        .collect();

    let cmp_key = |c: ClusterId| -> (i64, f64) {
        // Returns a (rank, tiebreak_id) pair so equal sort keys land in
        // ascending id order.
        let amps = session.spike_amplitudes(c);
        let amp_mean = if amps.is_empty() {
            f64::NEG_INFINITY
        } else {
            mean_amplitude(amps) as f64
        };
        let primary = match state.sort_by {
            ClusterColumn::Id => c.as_f64(),
            ClusterColumn::SpikeCount => session.spike_times(c).len() as f64,
            ClusterColumn::Amplitude => amp_mean,
            ClusterColumn::IsiViolations => {
                isi_violations(session.spike_times(c), refractory_samples) as f64
            }
            ClusterColumn::Quality => {
                let q = state.quality_for(c);
                if q.is_finite() {
                    q as f64
                } else {
                    f64::NEG_INFINITY
                }
            }
            ClusterColumn::Label => {
                label_rank::<P, F>(&label_str, session.label(c).unwrap_or_default()) as f64
            }
        };
        // Encode as (i64, f64) so we can sort with stable secondary on id.
        // We pack the primary into f64 and use id as the tie-breaker
        // (only matters when primary is equal across rows).
        (c.as_i64(), primary)
    };

    visible.sort_by(|&a, &b| {
        let (ai, ak) = cmp_key(a);
        let (bi, bk) = cmp_key(b);
        let primary = ak.partial_cmp(&bk).unwrap_or(std::cmp::Ordering::Equal);
        let primary = if state.sort_descending {
            primary.reverse()
        } else {
            primary
        };
        primary.then(ai.cmp(&bi))
    });

    // Header click helper: returns the response so we can detect clicks
    // and (visually) annotate the active sort column with an arrow.
    let header_label = |ui: &mut Ui, label: &str, col: ClusterColumn, state: &ClusterTableState| {
        let mut text = label.to_string();
        if state.sort_by == col {
            text.push(' ');
            text.push(if state.sort_descending { '▼' } else { '▲' });
        }
        ui.add(egui::Label::new(egui::RichText::new(text).strong()).sense(Sense::click()))
    };

    let mut clicked: Option<ClusterColumn> = None;

    TableBuilder::new(ui)
        .striped(true)
        .resizable(true)
        .column(Column::auto().at_least(48.0))
        .column(Column::auto().at_least(64.0))
        .column(Column::auto().at_least(56.0))
        .column(Column::auto().at_least(56.0))
        .column(Column::auto().at_least(56.0))
        .column(Column::auto().at_least(64.0))
        .header(20.0, |mut h| {
            h.col(|ui| {
                if header_label(ui, "ID", ClusterColumn::Id, state).clicked() {
                    clicked = Some(ClusterColumn::Id);
                }
            });
            h.col(|ui| {
                if header_label(ui, "# spikes", ClusterColumn::SpikeCount, state).clicked() {
                    clicked = Some(ClusterColumn::SpikeCount);
                }
            });
            h.col(|ui| {
                if header_label(ui, "amp", ClusterColumn::Amplitude, state).clicked() {
                    clicked = Some(ClusterColumn::Amplitude);
                }
            });
            h.col(|ui| {
                if header_label(ui, "ISI<1ms", ClusterColumn::IsiViolations, state).clicked() {
                    clicked = Some(ClusterColumn::IsiViolations);
                }
            });
            h.col(|ui| {
                if header_label(ui, "qual", ClusterColumn::Quality, state).clicked() {
                    clicked = Some(ClusterColumn::Quality);
                }
            });
            h.col(|ui| {
                if header_label(ui, "label", ClusterColumn::Label, state).clicked() {
                    clicked = Some(ClusterColumn::Label);
                }
            });
        })
        .body(|body| {
            body.rows(18.0, visible.len(), |mut row| {
                let i = visible[row.index()];
                let is_sel = selection.contains(i);
                row.col(|ui| {
                    let resp = ui.selectable_label(is_sel, format!("{i}"));
                    if resp.clicked() {
                        let mods = ui.ctx().input(|inp| inp.modifiers);
                        if mods.command {
                            selection.toggle(i);
                        } else if mods.shift {
                            selection.extend_to(i);
                        } else {
                            selection.replace(i);
                        }
                    }
                });
                row.col(|ui| {
                    ui.label(format!("{}", session.spike_times(i).len()));
                });
                row.col(|ui| {
                    let amps = session.spike_amplitudes(i);
                    if amps.is_empty() {
                        ui.label("—");
                    } else {
                        ui.label(format!("{:.2}", mean_amplitude(amps)));
                    }
                });
                row.col(|ui| {
                    let v = isi_violations(session.spike_times(i), refractory_samples);
                    ui.label(format!("{v}"));
                });
                row.col(|ui| {
                    let q = state.quality_for(i);
                    if q.is_finite() {
                        // Colour-code: red below 0.4, yellow 0.4–0.7, green 0.7+.
                        let colour = if q < 0.4 {
                            Color32::from_rgb(225, 87, 89)
                        } else if q < 0.7 {
                            Color32::from_rgb(237, 201, 72)
                        } else {
                            Color32::from_rgb(89, 161, 79)
                        };
                        ui.colored_label(colour, format!("{q:.2}"));
                    } else {
                        ui.label("—");
                    }
                });
                row.col(|ui| {
                    let lbl = session.label(i).unwrap_or_default();
                    ui.label(label_str(&lbl));
                });
            });
        });

    if let Some(col) = clicked {
        state.toggle_sort(col);
    }
}

/// Map a label string into a stable sort rank — phy's display order: good,
/// mua, unsorted, noise. Anything we don't recognise sorts to the end.
fn label_rank<P, F>(label_str: &F, label: P::Label) -> i32
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    match label_str(&label) {
        "good" => 0,
        "mua" => 1,
        "unsorted" => 2,
        "noise" => 3,
        _ => 4,
    }
}

/// Trace view. CPU runs LTTB to bound point count; GPU draws one line strip
/// per channel via [`TraceCallback`] so we scale to Neuropixels-grade
/// `n_channels × points_per_channel` without choking the egui painter.
pub fn trace_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    window_start: u64,
    window_len: u32,
    target_points: usize,
    cfg: TracePreproc,
    gpu: Option<&GpuTracePreproc>,
) {
    let avail = ui.available_size();
    let (rect, _resp) = ui.allocate_exact_size(avail, Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_gray(15));

    let n_channels = session.provider.n_channels();
    if n_channels == 0 {
        return;
    }

    let window_start = sorrel_io::SampleIndex(window_start);
    let verts: Vec<TraceVertex> = match gpu {
        Some(g) => {
            build_trace_vertices_gpu(session, window_start, window_len, target_points, cfg, g)
        }
        None => build_trace_vertices_cfg(session, window_start, window_len, target_points, cfg),
    };
    if verts.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "no trace data",
            egui::FontId::proportional(14.0),
            Color32::GRAY,
        );
        return;
    }

    // Each channel's strip is contiguous in the buffer; total len = n_channels * pts.
    let points_per_channel = (verts.len() / n_channels as usize) as u32;
    let t0 = verts.first().map(|v| v.pos[0]).unwrap_or(0.0);
    let t1 = verts.last().map(|v| v.pos[0]).unwrap_or(t0 + 1.0);
    let t_span = (t1 - t0).max(1e-9);

    // Channel separators painted on the egui side so they pick up theming.
    for ch in 1..n_channels {
        let y = rect.top() + rect.height() * ch as f32 / n_channels as f32;
        painter.hline(
            rect.x_range(),
            y,
            Stroke::new(0.5_f32, Color32::from_gray(40)),
        );
    }

    // Amplitude scale: take 80% of a channel's row, normalised by full-scale.
    // Full-scale comes from the provider so non-int16 dtypes don't get squashed.
    let row_clip_height = 2.0 / n_channels as f32;
    let full_scale = session.provider.amplitude_full_scale().max(1.0);
    let amp_scale = (row_clip_height * 0.4) / full_scale;

    let callback = TraceCallback {
        vertices: verts,
        n_channels,
        points_per_channel,
        t0,
        t_span,
        amp_scale,
        color: [0.71, 0.86, 1.0, 1.0], // matches the previous Color32::from_rgb(180, 220, 255)
    };
    ui.painter()
        .add(eframe::egui_wgpu::Callback::new_paint_callback(
            rect, callback,
        ));

    let _ = Vec2::ZERO;
    let _ = Rect::NOTHING;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_sorts_by_id_ascending() {
        let s = ClusterTableState::default();
        assert_eq!(s.sort_by, ClusterColumn::Id);
        assert!(!s.sort_descending);
        assert!(s.filter_text.is_empty());
        assert!(!s.hide_empty);
        assert!(!s.hide_noise);
    }

    #[test]
    fn toggle_sort_on_active_column_flips_direction() {
        let mut s = ClusterTableState::default();
        s.toggle_sort(ClusterColumn::Id); // already active; flip to descending
        assert_eq!(s.sort_by, ClusterColumn::Id);
        assert!(s.sort_descending);
        s.toggle_sort(ClusterColumn::Id); // flip back
        assert!(!s.sort_descending);
    }

    #[test]
    fn toggle_sort_on_numeric_column_starts_descending() {
        let mut s = ClusterTableState::default();
        s.toggle_sort(ClusterColumn::SpikeCount);
        assert_eq!(s.sort_by, ClusterColumn::SpikeCount);
        assert!(
            s.sort_descending,
            "numeric columns should default to descending — most-spikes-first"
        );
        s.toggle_sort(ClusterColumn::Amplitude);
        assert!(s.sort_descending);
        s.toggle_sort(ClusterColumn::IsiViolations);
        assert!(s.sort_descending);
    }

    #[test]
    fn toggle_sort_on_text_column_starts_ascending() {
        let mut s = ClusterTableState::default();
        s.toggle_sort(ClusterColumn::Label);
        assert_eq!(s.sort_by, ClusterColumn::Label);
        assert!(!s.sort_descending);
    }

    #[test]
    fn toggle_sort_swap_resets_direction_to_default_for_new_column() {
        // Start descending on SpikeCount, then switch to Label: should be
        // ascending (the default for text columns).
        let mut s = ClusterTableState::default();
        s.toggle_sort(ClusterColumn::SpikeCount); // desc
        assert!(s.sort_descending);
        s.toggle_sort(ClusterColumn::Label);
        assert_eq!(s.sort_by, ClusterColumn::Label);
        assert!(!s.sort_descending);
    }
}

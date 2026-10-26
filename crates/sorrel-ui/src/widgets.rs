use crate::selection::SelectionSet;
use crate::theme::{selection_colour, Theme};
use crate::views::draw_axes;
use crate::wgpu_trace::TraceCallback;
use egui::{Color32, Sense, Stroke, Ui, Vec2};
use sorrel_compute::quality_breakdown;
use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider, SampleIndex};
use sorrel_render::{
    build_trace_vertices_cfg_shanked, build_trace_vertices_gpu_shanked, GpuTracePreproc,
    TracePreproc, TraceVertex,
};
use std::sync::Arc;

/// Sortable column on the cluster table. The widget mutates the table state
/// on header clicks, so this stays in lockstep with the column rendering.
#[derive(Copy, Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ClusterColumn {
    Id,
    SpikeCount,
    Amplitude,
    IsiViolations,
    Quality,
    Label,
    Tags,
}

/// Persistent cluster-table UI state — sort + filter live on the app.
#[derive(Clone, Debug)]
pub struct ClusterTableState {
    pub sort_by: ClusterColumn,
    pub sort_descending: bool,
    pub filter_text: String,
    pub hide_empty: bool,
    pub hide_noise: bool,
    /// Only clusters carrying a `review:*` tag (the queue `sorrel auto` leaves).
    pub review_only: bool,
    /// Cached "quick quality" composite score per cluster — uses only the
    /// cheap metrics (no PC-feature isolation k-NN) so the table sort is
    /// snappy even on 1k-cluster recordings. Indexed by `ClusterId`.
    /// Recomputed when [`Self::quality_key`] no longer matches.
    quality_cache: Vec<f32>,
    /// Curation version at the time the cache was built. Any change
    /// invalidates the whole cache.
    quality_key: u64,
    /// The rows as last drawn — sorted and filtered. Keyboard navigation
    /// and Shift-click walk this, so "next" means the next row the user
    /// sees, not the next cluster id.
    pub visible: Vec<ClusterId>,
    /// Anchor the table last scrolled to; scroll again only when it
    /// changes so the wheel isn't fought every frame.
    last_scrolled: Option<ClusterId>,
}

impl Default for ClusterTableState {
    fn default() -> Self {
        Self {
            sort_by: ClusterColumn::Id,
            sort_descending: false,
            filter_text: String::new(),
            hide_empty: false,
            hide_noise: false,
            review_only: false,
            quality_cache: Vec::new(),
            quality_key: u64::MAX,
            visible: Vec::new(),
            last_scrolled: None,
        }
    }
}

impl ClusterTableState {
    /// Refresh the quality cache if the curation history has moved.
    fn ensure_quality<P: DataProvider>(&mut self, session: &Session<P>) {
        let key = session.version();
        if key == self.quality_key {
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
                    | ClusterColumn::Tags
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
    let theme = Theme::from_ui(ui);

    // Filter strip above the table; wraps so it never sets the panel width.
    ui.horizontal_wrapped(|ui| {
        let label = ui.label("filter:");
        ui.add(
            egui::TextEdit::singleline(&mut state.filter_text)
                .desired_width(80.0)
                .hint_text("id or tag"),
        )
        .labelled_by(label.id)
        .on_hover_text("Cluster id, or part of a tag (e.g. merged, src:llm, bc:snr)");
        ui.checkbox(&mut state.hide_empty, "hide empty");
        ui.checkbox(&mut state.hide_noise, "hide noise");
        ui.checkbox(&mut state.review_only, "review")
            .on_hover_text("Only clusters `sorrel auto` tagged review:*; any edit clears the tag");
        // `visible` is last frame's list; one frame of lag is invisible.
        ui.weak(format!(
            "{} / {}",
            state.visible.len(),
            session.n_clusters()
        ))
        .on_hover_text("visible / total clusters");
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
            let tags = session.tags(i);
            if state.review_only && !tags.iter().any(|t| t.starts_with("review:")) {
                return false;
            }
            if !filter.is_empty()
                && !i.to_string().contains(filter)
                && !tags.iter().any(|t| t.contains(filter))
            {
                return false;
            }
            true
        })
        .collect();

    let cmp_key = |c: ClusterId| -> (i64, f64) {
        // Returns `(id, primary)`; the id is the tie-breaker so equal sort
        // keys land in ascending id order.
        let primary = match state.sort_by {
            ClusterColumn::Id => c.as_f64(),
            ClusterColumn::SpikeCount => session.spike_times(c).len() as f64,
            ClusterColumn::Amplitude => {
                let amps = session.spike_amplitudes(c);
                if amps.is_empty() {
                    f64::NEG_INFINITY
                } else {
                    mean_amplitude(amps) as f64
                }
            }
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
            // Review tags first, then by how many tags (how much evidence).
            ClusterColumn::Tags => {
                let tags = session.tags(c);
                let review = tags.iter().filter(|t| t.starts_with("review:")).count();
                (review * 1000 + tags.len()) as f64
            }
        };
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
            text.push(if state.sort_descending { 'v' } else { '^' });
        }
        ui.add(egui::Label::new(egui::RichText::new(text).strong()).sense(Sense::click()))
    };

    let mut clicked: Option<ClusterColumn> = None;

    // Keep the keyboard anchor in view: scroll minimally (no alignment) and
    // only when the anchor changed, so clicking a visible row or wheeling
    // away from it never snaps the list.
    let mut scroll_to: Option<usize> = None;
    if let Some(anchor) = selection.anchor() {
        if state.last_scrolled != Some(anchor) {
            scroll_to = visible.iter().position(|&c| c == anchor);
            state.last_scrolled = Some(anchor);
        }
    }

    let headers: [(&str, ClusterColumn, &str); 7] = [
        (
            "ID",
            ClusterColumn::Id,
            "Cluster id. Swatch = colour in the plots.",
        ),
        ("# spikes", ClusterColumn::SpikeCount, "Spike count"),
        ("amp", ClusterColumn::Amplitude, "Mean spike amplitude"),
        (
            "ISI<1ms",
            ClusterColumn::IsiViolations,
            "Inter-spike intervals under 1 ms (refractory violations)",
        ),
        (
            "qual",
            ClusterColumn::Quality,
            "Composite quality 0–1: green ≥ 0.7, yellow ≥ 0.4, red below",
        ),
        (
            "label",
            ClusterColumn::Label,
            "phy label: G/M/N/U keys relabel",
        ),
        (
            "tags",
            ClusterColumn::Tags,
            "Provenance from `sorrel auto` (src:*, merged:*, bc:*, conf:*); review:* in red. \
             Sort puts the review queue first",
        ),
    ];

    let mut table = TableBuilder::new(ui)
        .striped(true)
        .resizable(true)
        .sense(Sense::click())
        // Content-sized columns: fixed minimums summed past 430 px and
        // pushed the side panel to 40% of a laptop screen.
        .columns(Column::auto(), 6)
        .column(Column::remainder().at_least(40.0));
    if let Some(row) = scroll_to {
        table = table.scroll_to_row(row, None);
    }
    table
        .header(20.0, |mut h| {
            for (label, col, tip) in headers {
                h.col(|ui| {
                    if header_label(ui, label, col, state)
                        .on_hover_text(tip)
                        .clicked()
                    {
                        clicked = Some(col);
                    }
                });
            }
        })
        .body(|body| {
            body.rows(18.0, visible.len(), |mut row| {
                let i = visible[row.index()];
                let slot = selection.iter().position(|&c| c == i);
                row.set_selected(slot.is_some());
                row.col(|ui| {
                    ui.horizontal(|ui| {
                        if let Some(slot) = slot {
                            let (r, _) = ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
                            ui.painter().rect_filled(r, 2.0, selection_colour(slot));
                        }
                        ui.label(format!("{i}"));
                    });
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
                        ui.colored_label(theme.quality_colour(q), format!("{q:.2}"));
                    } else {
                        ui.label("—");
                    }
                });
                row.col(|ui| {
                    let lbl = label_str(&session.label(i).unwrap_or_default());
                    ui.colored_label(theme.label_colour(lbl), lbl);
                });
                row.col(|ui| {
                    let tags = session.tags(i);
                    if tags.is_empty() {
                        return;
                    }
                    // Review reasons first and in red; the rest dimmed.
                    let (review, rest): (Vec<&String>, Vec<&String>) =
                        tags.iter().partition(|t| t.starts_with("review:"));
                    ui.horizontal(|ui| {
                        if !review.is_empty() {
                            let text: Vec<&str> =
                                review.iter().map(|t| &t["review:".len()..]).collect();
                            ui.colored_label(theme.danger, text.join(" "));
                        }
                        let src: Vec<&str> =
                            rest.iter().filter_map(|t| t.strip_prefix("src:")).collect();
                        ui.weak(src.join("+"));
                    })
                    .response
                    .on_hover_text(tags.join("\n"));
                });
                // Whole-row click (phy semantics): plain replaces,
                // Cmd/Ctrl toggles, Shift extends in table order.
                if row.response().clicked() {
                    let mods = row.response().ctx.input(|inp| inp.modifiers);
                    if mods.command {
                        selection.toggle(i);
                    } else if mods.shift {
                        selection.extend_to(i, &visible);
                    } else {
                        selection.replace(i);
                    }
                }
            });
        });

    if let Some(col) = clicked {
        state.toggle_sort(col);
    }
    state.visible = visible;
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

/// One selected cluster's spikes to mark on the trace.
pub struct TraceMarker<'a> {
    pub colour: Color32,
    /// Sorted spike times (global samples).
    pub times: &'a [SampleIndex],
    /// Peak channel row to draw the tick on; `None` = full height.
    pub channel: Option<usize>,
}

/// Navigation the user requested with the mouse on the trace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceNav {
    /// Pan in samples (positive = later in the recording).
    pub pan: i64,
    /// Factor to divide the window length by (`> 1` zooms in).
    pub zoom: f32,
    /// Horizontal fraction of the view the zoom should pivot around.
    pub anchor: f32,
}

impl Default for TraceNav {
    fn default() -> Self {
        Self {
            pan: 0,
            zoom: 1.0,
            anchor: 0.5,
        }
    }
}

/// Memo of the downsampled vertices for the current window, so the mmap
/// read plus LTTB run once per change instead of every frame.
#[derive(Default)]
pub struct TraceCache {
    /// `(window_start, window_len, target_points, hp bits, cmr, per-shank)`.
    key: Option<(u64, u32, usize, Option<u32>, bool, bool)>,
    verts: Arc<Vec<TraceVertex>>,
    /// Largest |sample| in `verts`; sets the gain.
    peak: f32,
    generation: u64,
}

/// Trace view. CPU runs LTTB to bound point count; GPU draws one line strip
/// per channel via [`TraceCallback`] so we scale to Neuropixels-grade
/// `n_channels × points_per_channel` without choking the egui painter.
/// Selected clusters' spikes are ticked on their peak channel, the time
/// axis is labelled, and the mouse pans (drag / wheel) and zooms
/// (Ctrl-wheel / pinch).
#[allow(clippy::too_many_arguments)]
pub fn trace_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    window_start: u64,
    window_len: u32,
    target_points: usize,
    cfg: TracePreproc,
    gpu: Option<&GpuTracePreproc>,
    markers: &[TraceMarker<'_>],
    shanks: &[u32],
    cache: &mut TraceCache,
) -> Option<TraceNav> {
    let theme = Theme::from_ui(ui);
    let avail = ui.available_size();
    let (rect, resp) = ui.allocate_exact_size(avail, Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme.plot_bg);

    let n_channels = session.provider.n_channels();
    if n_channels == 0 {
        return None;
    }

    let key = (
        window_start,
        window_len,
        target_points,
        cfg.hp().map(f32::to_bits),
        cfg.has_cmr(),
        !shanks.is_empty(),
    );
    // A remote guest's provider returns an empty window while the bytes
    // are in flight; keep the key unset so the window is rebuilt once they
    // land instead of caching the gap.
    let fetching = session.provider.pending_fetches() > 0;
    if cache.key != Some(key) || fetching {
        let start = SampleIndex(window_start);
        let verts = match gpu {
            Some(g) => build_trace_vertices_gpu_shanked(
                session,
                start,
                window_len,
                target_points,
                cfg,
                g,
                shanks,
            ),
            None => build_trace_vertices_cfg_shanked(
                session,
                start,
                window_len,
                target_points,
                cfg,
                shanks,
            ),
        };
        cache.peak = verts.iter().fold(0.0, |m, v| m.max(v.pos[1].abs()));
        cache.verts = Arc::new(verts);
        cache.generation += 1;
        cache.key = (!fetching).then_some(key);
    }
    if cache.verts.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "no trace data",
            egui::FontId::proportional(14.0),
            theme.caption,
        );
        return None;
    }

    // Each channel's strip is contiguous in the buffer; total len = n_channels * pts.
    let points_per_channel = (cache.verts.len() / n_channels as usize) as u32;
    let t0 = cache.verts.first().map(|v| v.pos[0]).unwrap_or(0.0);
    let t1 = cache.verts.last().map(|v| v.pos[0]).unwrap_or(t0 + 1.0);
    let t_span = (t1 - t0).max(1e-9);

    // Channel separators + labels, painted on the egui side so they pick
    // up theming. Labels thin out when rows get shorter than the text.
    let row_h = rect.height() / n_channels as f32;
    let label_every = (10.0 / row_h).ceil().max(1.0) as u32;
    for ch in 0..n_channels {
        let y = rect.top() + row_h * ch as f32;
        if ch > 0 {
            painter.hline(rect.x_range(), y, Stroke::new(0.5_f32, theme.grid));
        }
        if ch % label_every == 0 {
            painter.text(
                egui::pos2(rect.left() + 3.0, y + row_h * 0.5),
                egui::Align2::LEFT_CENTER,
                format!("{ch}"),
                egui::FontId::monospace(9.0),
                theme.caption,
            );
        }
    }

    // Gain: the window's largest excursion spans one row, so spikes of a
    // few hundred counts stay visible (dtype full-scale flattened them).
    // ponytail: per-window auto-gain rescales while panning; add a gain
    // knob if users want it fixed.
    let row_clip_height = 2.0 / n_channels as f32;
    let amp_scale = row_clip_height * 0.5 / cache.peak.max(f32::EPSILON);

    let callback = TraceCallback {
        vertices: Arc::clone(&cache.verts),
        generation: cache.generation,
        n_channels,
        points_per_channel,
        t0,
        t_span,
        amp_scale,
        color: if theme.dark {
            [0.71, 0.86, 1.0, 1.0]
        } else {
            [0.15, 0.3, 0.55, 1.0]
        },
    };
    ui.painter()
        .add(eframe::egui_wgpu::Callback::new_paint_callback(
            rect, callback,
        ));

    // Spike markers for the selection: a tick on the peak-channel row.
    let w_end = window_start + window_len as u64;
    let mut n_marked = 0usize;
    for m in markers {
        let lo = m.times.partition_point(|t| t.0 < window_start);
        let hi = m.times.partition_point(|t| t.0 < w_end);
        let stroke = Stroke::new(1.0_f32, m.colour.gamma_multiply(0.85));
        for t in &m.times[lo..hi] {
            let x = rect.left() + (t.0 - window_start) as f32 / window_len as f32 * rect.width();
            let (y0, y1) = match m.channel {
                Some(ch) if (ch as u32) < n_channels => {
                    let y = rect.top() + row_h * ch as f32;
                    (y, y + row_h)
                }
                _ => (rect.top(), rect.bottom()),
            };
            painter.line_segment([egui::pos2(x, y0), egui::pos2(x, y1)], stroke);
            n_marked += 1;
        }
    }

    let sr = session.provider.sample_rate().max(1.0);
    draw_axes(
        &painter,
        rect,
        (window_start as f32 / sr, w_end as f32 / sr),
        (0.0, 0.0),
        " s",
        "",
        &theme,
    );
    if !markers.is_empty() {
        painter.text(
            rect.right_top() + egui::vec2(-4.0, 2.0),
            egui::Align2::RIGHT_TOP,
            format!("{n_marked} selected spikes in window"),
            egui::FontId::proportional(10.0),
            theme.caption,
        );
    }

    // Mouse navigation: drag or wheel pans, Ctrl-wheel / pinch zooms
    // around the pointer.
    let mut nav = TraceNav::default();
    if resp.dragged() {
        nav.pan -= (resp.drag_delta().x / rect.width() * window_len as f32) as i64;
    }
    if resp.hovered() {
        let (scroll, zoom, pointer) =
            ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta(), i.pointer.hover_pos()));
        let d = if scroll.x.abs() > scroll.y.abs() {
            scroll.x
        } else {
            scroll.y
        };
        nav.pan -= (d / rect.width() * window_len as f32) as i64;
        nav.zoom = zoom;
        if let Some(p) = pointer {
            nav.anchor = ((p.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
        }
    }
    (nav.pan != 0 || nav.zoom != 1.0).then_some(nav)
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

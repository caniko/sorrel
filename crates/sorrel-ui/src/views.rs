//! Per-cluster summary views for the central panel.
//!
//! Each view function consumes a `&Session<P>` and the active selection,
//! draws into the supplied `egui::Ui`, and stays free of side effects so
//! the caller decides when to repaint.

use egui::{Color32, Pos2, Sense, Stroke, Ui};
use sorrel_compute::{psth, sliding_refractory_contamination, QualityBreakdown};
use sorrel_data::suggest::{rank_merge_candidates_memo, rank_split_candidates_memo, SuggestMemo};
use sorrel_data::{
    cluster_quality, preview_merge, DepthSource, MergeCandidate, Session, SpikeDepths,
    SplitCandidate, SuggestConfig,
};
use sorrel_io::{ClusterId, DataProvider, Events};

use crate::theme::{selection_colour, Theme};

pub use crate::correlogram_view::{correlogram_grid_view, CcgCache};
pub use crate::isi_view::isi_view;

/// "Nice" tick positions covering `[lo, hi]`: steps of 1, 2 or 5 × 10ⁿ,
/// about `target` of them. Returns an empty list for a degenerate range.
pub fn nice_ticks(lo: f32, hi: f32, target: usize) -> Vec<f32> {
    let span = hi - lo;
    if span <= 0.0 || !span.is_finite() || target == 0 {
        return Vec::new();
    }
    let raw = span / target as f32;
    let mag = 10f32.powf(raw.log10().floor());
    let norm = raw / mag;
    let step = if norm < 1.5 {
        1.0
    } else if norm < 3.5 {
        2.0
    } else if norm < 7.5 {
        5.0
    } else {
        10.0
    } * mag;
    let first = (lo / step).ceil();
    let last = (hi / step).floor();
    let mut out = Vec::new();
    let mut k = first;
    while k <= last && out.len() < 64 {
        out.push(k * step);
        k += 1.0;
    }
    out
}

/// Left gutter for y tick labels and bottom strip for x tick labels, so
/// labels never sit on the data.
pub const Y_GUTTER: f32 = 40.0;
pub const X_STRIP: f32 = 22.0;

/// The data area of a plot whose axes [`draw_axes`] draws in `rect`'s
/// gutters; `has_y` is false for plots without a value axis.
pub fn axes_data_rect(rect: egui::Rect, has_y: bool) -> egui::Rect {
    let left = if has_y { Y_GUTTER } else { 0.0 };
    egui::Rect::from_min_max(
        egui::pos2(rect.left() + left, rect.top()),
        egui::pos2(rect.right(), rect.bottom() - X_STRIP),
    )
}

/// Painter text never goes below 11 px.
pub(crate) const TICK_PX: f32 = 11.0;
pub(crate) const CAPTION_PX: f32 = 12.0;
pub(crate) const SLOT_NAMES: [&str; 8] = [
    "blue", "orange", "red", "teal", "green", "yellow", "purple", "pink",
];

pub(crate) fn blend(a: Color32, b: Color32, t: f32) -> Color32 {
    a.lerp_to_gamma(b, t)
}

/// Gridline / reference-line colour, at least 3:1 against `plot_bg` in
/// both themes (a blend toward the text colour, not the near-invisible
/// widget stroke).
pub(crate) fn grid_colour(theme: &Theme) -> Color32 {
    blend(theme.plot_bg, theme.text, 0.6)
}

/// "c1 blue, c2 orange": colour key for the selection.
pub(crate) fn legend(selection: &[ClusterId]) -> String {
    selection
        .iter()
        .enumerate()
        .map(|(i, c)| format!("c{c} {}", SLOT_NAMES[i % SLOT_NAMES.len()]))
        .collect::<Vec<_>>()
        .join(", ")
}

/// One-line plot caption above a plot; truncates, full text on hover.
pub(crate) fn caption(ui: &mut Ui, theme: &Theme, text: String) {
    let r = ui.add(
        egui::Label::new(
            egui::RichText::new(text.clone())
                .size(CAPTION_PX)
                .color(theme.caption),
        )
        .truncate(),
    );
    r.on_hover_text(text);
}

/// Height left in the pane; inside a scroll area, the visible viewport.
pub(crate) fn fill_height(ui: &Ui) -> f32 {
    let a = ui.available_size_before_wrap().y;
    if a.is_finite() && a < 1.0e6 {
        a
    } else {
        (ui.clip_rect().bottom() - ui.cursor().top()).max(0.0)
    }
}

/// Allocate a plot filling the rest of the pane (at least `min`).
pub(crate) fn alloc_plot(
    ui: &mut Ui,
    min: egui::Vec2,
    sense: Sense,
) -> (egui::Rect, egui::Response) {
    let h = fill_height(ui);
    let w = ui.available_width();
    ui.allocate_exact_size(egui::vec2(w.max(min.x), h.max(min.y)), sense)
}

/// Plot-coloured pane with a centred message.
pub(crate) fn empty_plot(ui: &mut Ui, theme: &Theme, msg: &str) {
    let (rect, _) = alloc_plot(ui, egui::vec2(120.0, 120.0), Sense::hover());
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 0.0, theme.plot_bg);
    // Wrapped, so a long message in a narrow pane is not clipped.
    let g = p.layout(
        msg.to_owned(),
        egui::FontId::proportional(13.0),
        theme.caption,
        (rect.width() - 16.0).max(40.0),
    );
    let pos = rect.center() - g.size() / 2.0;
    p.galley(pos, g, theme.caption);
}

/// Pad `[lo, hi]` by `frac` of its span on both sides (degenerate-safe).
pub(crate) fn pad_range(lo: f32, hi: f32, frac: f32) -> (f32, f32) {
    let p = (hi - lo).max(1e-6) * frac;
    (lo - p, hi + p)
}

/// Small slot-colour square.
pub(crate) fn swatch(ui: &mut Ui, colour: Color32) {
    let (r, _) = ui.allocate_exact_size(egui::vec2(9.0, 9.0), Sense::hover());
    ui.painter().rect_filled(r, 2.0, colour);
}

/// Right-align `add` within the cell. The height is one text line: a
/// bare right-to-left layout would take the whole available height and
/// stretch the table row.
pub(crate) fn right_aligned<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    let h = ui
        .text_style_height(&egui::TextStyle::Monospace)
        .max(ui.text_style_height(&egui::TextStyle::Body));
    ui.allocate_ui_with_layout(
        egui::vec2(ui.available_width(), h),
        egui::Layout::right_to_left(egui::Align::Center),
        add,
    )
    .inner
}

/// Right-aligned monospace numeric cell.
pub(crate) fn num_cell(ui: &mut Ui, text: String, colour: Option<Color32>) {
    right_aligned(ui, |ui| {
        let mut t = egui::RichText::new(text).monospace();
        if let Some(c) = colour {
            t = t.color(c);
        }
        ui.label(t);
    });
}

/// One-row parameter bar for plot panes: `primary` controls inline, the
/// rest in a "⚙" popover, `caption` right-aligned and truncated (full
/// text on hover). Keeps pane toolbars on one row at ~300 px.
#[allow(dead_code)] // used by the pane views
pub fn param_bar(
    ui: &mut Ui,
    caption: &str,
    primary: impl FnOnce(&mut Ui),
    secondary: impl FnOnce(&mut Ui),
) {
    ui.horizontal(|ui| {
        primary(ui);
        ui.menu_button("⚙", |ui| {
            ui.set_min_width(180.0);
            secondary(ui);
        })
        .response
        .on_hover_text("More settings");
        if !caption.is_empty() {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add(egui::Label::new(egui::RichText::new(caption).small().weak()).truncate())
                    .on_hover_text(caption);
            });
        }
    });
}

/// Right-aligned strong header cell.
pub(crate) fn num_head(ui: &mut Ui, text: &str) -> egui::Response {
    right_aligned(ui, |ui| ui.strong(text))
}

pub(crate) fn tick_decimals(step: f32) -> usize {
    if step >= 1.0 {
        0
    } else if step >= 0.1 {
        1
    } else if step >= 0.01 {
        2
    } else {
        3
    }
}

/// Tick text for `v`; zero uses the same decimals as its neighbours.
pub(crate) fn tick_label(v: f32, step_hint: f32) -> String {
    let v = if v == 0.0 || v.abs() < step_hint.abs() * 1e-4 {
        0.0
    } else {
        v
    };
    format!("{:.*}", tick_decimals(step_hint), v)
}

pub(crate) fn fmt_tick(v: f32, step: f32, decimals: Option<usize>) -> String {
    match decimals {
        Some(d) => format!("{:.*}", d, if v == 0.0 { 0.0 } else { v }),
        None => tick_label(v, step),
    }
}

pub(crate) fn tick_step(ticks: &[f32], span: f32) -> f32 {
    ticks.windows(2).map(|w| w[1] - w[0]).next().unwrap_or(span)
}

/// Options for [`axes`]; the defaults are what [`draw_axes`] uses.
#[derive(Default, Clone, Copy)]
pub(crate) struct AxisOpts<'a> {
    /// Faint full-length horizontal gridlines at the y ticks.
    pub(crate) grid: bool,
    /// Tick marks only, no x labels or title (shared x axis).
    pub(crate) hide_x: bool,
    pub(crate) x_ticks: Option<&'a [f32]>,
    pub(crate) y_ticks: Option<&'a [f32]>,
    pub(crate) x_decimals: Option<usize>,
}

/// Draw the axes of a plot whose OUTER rect is `rect`: x ticks and labels
/// in the bottom [`X_STRIP`], y ticks right-aligned in the left
/// [`Y_GUTTER`] (when `y.1 > y.0`) of [`axes_data_rect`]. Units are axis
/// titles (x right-aligned at the strip's end, y at the top of the
/// gutter), never appended to a tick. Every label stays inside `rect`.
pub fn draw_axes(
    painter: &egui::Painter,
    rect: egui::Rect,
    x: (f32, f32),
    y: (f32, f32),
    x_unit: &str,
    y_unit: &str,
    theme: &Theme,
) {
    axes(
        painter,
        rect,
        x,
        y,
        x_unit,
        y_unit,
        theme,
        AxisOpts::default(),
    );
}

/// [`draw_axes`] plus faint gridlines across the data area.
pub fn draw_axes_grid(
    painter: &egui::Painter,
    rect: egui::Rect,
    x: (f32, f32),
    y: (f32, f32),
    x_unit: &str,
    y_unit: &str,
    theme: &Theme,
) {
    let opts = AxisOpts {
        grid: true,
        ..AxisOpts::default()
    };
    axes(painter, rect, x, y, x_unit, y_unit, theme, opts);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn axes(
    painter: &egui::Painter,
    rect: egui::Rect,
    x: (f32, f32),
    y: (f32, f32),
    x_unit: &str,
    y_unit: &str,
    theme: &Theme,
    opts: AxisOpts,
) {
    let has_y = y.1 > y.0;
    let data = axes_data_rect(rect, has_y);
    let x_ok = x.1 > x.0;
    if !x_ok || data.width() < 8.0 || data.height() < 8.0 {
        return;
    }
    let font = egui::FontId::proportional(TICK_PX);
    let ink = theme.caption;
    let tick = Stroke::new(1.0_f32, ink);
    let axis = Stroke::new(1.0_f32, grid_colour(theme));
    let faint = Stroke::new(0.5_f32, grid_colour(theme));
    painter.line_segment([data.left_bottom(), data.right_bottom()], axis);
    if has_y {
        painter.line_segment([data.left_top(), data.left_bottom()], axis);
    }

    // x axis.
    let label_top = data.bottom() + 5.0;
    let xt = match opts.x_ticks {
        Some(t) => t.to_vec(),
        None => nice_ticks(x.0, x.1, ((data.width() / 60.0) as usize).clamp(3, 8)),
    };
    let x_step = tick_step(&xt, x.1 - x.0);
    let mut labels = Vec::new();
    let mut right_edge = f32::NEG_INFINITY;
    for &v in &xt {
        let sx = data.left() + (v - x.0) / (x.1 - x.0) * data.width();
        painter.line_segment(
            [
                egui::pos2(sx, data.bottom()),
                egui::pos2(sx, data.bottom() + 3.0),
            ],
            tick,
        );
        if opts.hide_x {
            continue;
        }
        let g = painter.layout_no_wrap(fmt_tick(v, x_step, opts.x_decimals), font.clone(), ink);
        let w = g.size().x;
        let left = (sx - w / 2.0).clamp(rect.left(), (rect.right() - w).max(rect.left()));
        if left < right_edge + 4.0 {
            continue;
        }
        right_edge = left + w;
        labels.push((left, g));
    }
    let unit = x_unit.trim();
    let title = (!unit.is_empty() && !opts.hide_x)
        .then(|| painter.layout_no_wrap(unit.to_owned(), font.clone(), ink));
    let mut title_x = None;
    if let Some(g) = &title {
        let w = g.size().x;
        let right_x = rect.right() - w - 2.0;
        if labels
            .last()
            .is_none_or(|(l, lg)| l + lg.size().x + 6.0 <= right_x)
        {
            title_x = Some(right_x);
        } else if has_y
            && labels
                .first()
                .is_none_or(|(l, _)| *l >= rect.left() + w + 8.0)
        {
            // Ticks win the right end; the title moves under the gutter.
            title_x = Some(rect.left() + 2.0);
        } else {
            // No room anywhere: the title keeps the end, colliding ticks go.
            title_x = Some(right_x);
            labels.retain(|(l, lg)| l + lg.size().x + 6.0 <= right_x);
        }
    }
    for (left, g) in labels {
        painter.galley(egui::pos2(left, label_top), g, ink);
    }
    if let (Some(g), Some(tx)) = (title, title_x) {
        painter.galley(egui::pos2(tx, label_top), g, ink);
    }

    // y axis (views without a value axis pass an empty y range).
    if !has_y {
        return;
    }
    let y_title = (!y_unit.trim().is_empty())
        .then(|| painter.layout_no_wrap(y_unit.trim().to_owned(), font.clone(), ink));
    let title_h = if y_title.is_some() {
        TICK_PX + 4.0
    } else {
        0.0
    };
    let yt = match opts.y_ticks {
        Some(t) => t.to_vec(),
        None => nice_ticks(y.0, y.1, ((data.height() / 40.0) as usize).clamp(2, 6)),
    };
    let y_step = tick_step(&yt, y.1 - y.0);
    for &v in &yt {
        let sy = data.bottom() - (v - y.0) / (y.1 - y.0) * data.height();
        if opts.grid {
            painter.line_segment(
                [egui::pos2(data.left(), sy), egui::pos2(data.right(), sy)],
                faint,
            );
        }
        painter.line_segment(
            [
                egui::pos2(data.left() - 3.0, sy),
                egui::pos2(data.left(), sy),
            ],
            tick,
        );
        let g = painter.layout_no_wrap(fmt_tick(v, y_step, None), font.clone(), ink);
        let h = g.size().y;
        if title_h > 0.0 && sy - h / 2.0 < rect.top() + title_h {
            continue;
        }
        let top = (sy - h / 2.0).clamp(rect.top(), (data.bottom() - h).max(rect.top()));
        let left = (data.left() - 6.0 - g.size().x).max(rect.left());
        painter.galley(egui::pos2(left, top), g, ink);
    }
    if let Some(g) = y_title {
        let left = (data.left() - 6.0 - g.size().x).max(rect.left());
        painter.galley(egui::pos2(left, rect.top()), g, ink);
    }
}

/// Filled step histogram: bars from the baseline plus a solid outline.
pub(crate) fn step_hist(
    painter: &egui::Painter,
    data: egui::Rect,
    vals: &[f32],
    y_top: f32,
    colour: Color32,
    fill: f32,
) {
    if vals.is_empty() || y_top <= 0.0 {
        return;
    }
    let bw = data.width() / vals.len() as f32;
    let y_of = |v: f32| data.bottom() - (v / y_top).clamp(0.0, 1.0) * data.height();
    let mut pts = Vec::with_capacity(vals.len() * 2);
    for (i, &v) in vals.iter().enumerate() {
        let x0 = data.left() + bw * i as f32;
        let y = y_of(v);
        if v > 0.0 {
            painter.rect_filled(
                egui::Rect::from_min_max(egui::pos2(x0, y), egui::pos2(x0 + bw, data.bottom())),
                0.0,
                colour.gamma_multiply(fill),
            );
        }
        pts.push(egui::pos2(x0, y));
        pts.push(egui::pos2(x0 + bw, y));
    }
    painter.add(egui::Shape::line(pts, Stroke::new(1.25_f32, colour)));
}

/// Integer y ticks, about `target` of them, from 0 to `max`.
pub(crate) fn int_ticks(max: f32, target: usize) -> Vec<f32> {
    nice_ticks(0.0, max, target)
        .into_iter()
        .filter(|v| v.fract() == 0.0)
        .collect()
}

/// Short summary block — title, spike counts, label string, palette swatch.
pub fn selection_summary<P, F>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    label_str: F,
) where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    match selection.len() {
        0 => {
            ui.heading("No selection");
        }
        1 => {
            let c = selection[0];
            ui.horizontal(|ui| {
                ui.colored_label(selection_colour(0), "■");
                ui.heading(format!("Cluster {c}"));
            });
            let n_spikes = session.spike_times(c).len();
            ui.label(format!("{n_spikes} spikes"));
            ui.label(format!(
                "label: {}",
                label_str(&session.label(c).unwrap_or_default())
            ));
        }
        n => {
            let theme = Theme::from_ui(ui);
            ui.heading(format!("{n} clusters selected"));
            let sr = session.provider.sample_rate().max(1.0);
            let total_samples = session.provider.n_samples().0;
            let refractory = (sr * 0.0015).round() as u64;
            egui::Grid::new("summary-grid")
                .num_columns(6)
                .spacing([14.0, 6.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("");
                    for h in ["id", "label", "spikes", "amp", "qual"] {
                        ui.weak(h);
                    }
                    ui.end_row();
                    for (slot, &c) in selection.iter().enumerate() {
                        let times = session.spike_times(c);
                        let amps = session.spike_amplitudes(c);
                        swatch(ui, selection_colour(slot));
                        ui.label(egui::RichText::new(format!("c{c}")).strong());
                        let label = label_str(&session.label(c).unwrap_or_default());
                        ui.colored_label(theme.label_colour(label), label);
                        num_cell(ui, times.len().to_string(), None);
                        let amp = if amps.is_empty() {
                            "—".to_owned()
                        } else {
                            format!("{:.2}", sorrel_compute::mean_amplitude(amps))
                        };
                        num_cell(ui, amp, None);
                        let q = sorrel_compute::quality_breakdown(
                            times,
                            amps,
                            refractory,
                            total_samples,
                            sr,
                            50,
                        )
                        .composite();
                        num_cell(ui, format!("{q:.2}"), Some(theme.quality_colour(q)));
                        ui.end_row();
                    }
                });
            let total: usize = selection
                .iter()
                .map(|&c| session.spike_times(c).len())
                .sum();
            ui.weak(format!("{total} spikes (sum)"));
        }
    }
}

// ---- Waveforms (moved to `waveform_view`) ------------------------------

pub(crate) use crate::waveform_view::waveform_window_samples;
pub use crate::waveform_view::{waveform_view, WaveformCache};

// ---- RasterView (global spike scatter) ---------------------------------

use crate::wgpu_raster::{pack_rgba, RasterCallback, RasterVertex};

/// Cached raster vertex buffer. Invalidated by curation history changes
/// (merge/split/relabel — anything that mutates the cluster_index).
///
/// The vertices sit behind an `Arc` and carry a generation counter: the
/// paint callback shares them instead of cloning tens of millions of
/// points per frame, and the GPU upload is skipped while the generation
/// is unchanged.
#[derive(Debug, Default)]
pub struct RasterCache {
    /// Packed `(time_seconds, cluster_row, rgba)` vertices.
    vertices: std::sync::Arc<Vec<RasterVertex>>,
    /// Bumped on every rebuild so the pipeline can skip re-uploads.
    generation: u64,
    /// Map `cluster_id → row_index_in_raster`. Empty when not built.
    rows: Vec<i32>,
    /// Number of populated rows (clusters with at least one spike).
    n_rows: u32,
    /// Earliest spike time in seconds.
    t0: f32,
    /// Latest spike time in seconds.
    t1: f32,
    /// Curation version at last build — any change rebuilds.
    last_key: u64,
    /// Colour-scheme generation the vertices were coloured with.
    colours_gen: u64,
    /// Cluster id of each raster row (gutter labels).
    row_ids: Vec<u32>,
    /// Theme the neutral dot colour was built for.
    dark: bool,
    /// phy's PanZoom over time (x) and rows (y).
    pz: crate::panzoom::PanZoom,
}

impl RasterCache {
    pub fn new() -> Self {
        Self {
            last_key: u64::MAX,
            ..Self::default()
        }
    }

    fn ensure<P: DataProvider>(
        &mut self,
        session: &Session<P>,
        dark: bool,
        colours: &[Option<Color32>],
        colours_gen: u64,
    ) {
        let key = session.version();
        if key == self.last_key && dark == self.dark && colours_gen == self.colours_gen {
            return;
        }
        self.colours_gen = colours_gen;

        let n_clusters = session.n_clusters();
        let inv_sr = 1.0_f32 / session.provider.sample_rate().max(1.0);

        // Assign row indices only to non-empty clusters so the y-axis isn't
        // dominated by holes left behind after merges.
        let mut rows = vec![-1i32; n_clusters as usize];
        let mut n_rows = 0u32;
        let mut row_ids = Vec::new();
        for c in (0..n_clusters).map(ClusterId) {
            if !session.spike_times(c).is_empty() {
                rows[c.idx()] = n_rows as i32;
                row_ids.push(c.0);
                n_rows += 1;
            }
        }

        let total: usize = (0..n_clusters)
            .map(ClusterId)
            .map(|c| session.spike_times(c).len())
            .sum();

        let mut vertices = Vec::with_capacity(total);
        let mut t_min = f32::INFINITY;
        let mut t_max = f32::NEG_INFINITY;
        for c in (0..n_clusters).map(ClusterId) {
            let row = rows[c.idx()];
            if row < 0 {
                continue;
            }
            // Unselected clusters are neutral and faint; the selection is
            // painted in slot colours on top. Darker in light mode.
            // A colour scheme (Shift+wheel, `:cf`) tints them instead.
            let packed = match colours.get(c.idx()).copied().flatten() {
                Some(col) => pack_rgba(col.r(), col.g(), col.b(), 170),
                None if dark => pack_rgba(170, 170, 170, 110),
                None => pack_rgba(60, 60, 60, 150),
            };
            for &t in session.spike_times(c) {
                let ts = t.as_f32() * inv_sr;
                if ts < t_min {
                    t_min = ts;
                }
                if ts > t_max {
                    t_max = ts;
                }
                vertices.push(RasterVertex {
                    pos: [ts, row as f32],
                    color: packed,
                });
            }
        }

        if !t_min.is_finite() {
            t_min = 0.0;
            t_max = 1.0;
        }

        self.vertices = std::sync::Arc::new(vertices);
        self.generation += 1;
        self.rows = rows;
        self.row_ids = row_ids;
        self.dark = dark;
        self.n_rows = n_rows.max(1);
        self.t0 = t_min;
        self.t1 = t_max;
        self.last_key = key;
    }

    pub fn invalidate(&mut self) {
        self.last_key = u64::MAX;
        self.vertices = std::sync::Arc::new(Vec::new());
    }
}

/// Global raster: every spike of every non-empty cluster, time on x, cluster
/// row on y. Selected clusters' rows are highlighted in their selection
/// colour and the current trace window is marked. Click to seek the trace
/// window to that time.
///
/// Returns the time (in samples) the user clicked, if any.
pub fn raster_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    trace_window: (u64, u32),
    cache: &mut RasterCache,
    colours: (&[Option<Color32>], u64),
    scheme_delta: &mut i32,
) -> Option<RasterClick> {
    let theme = Theme::from_ui(ui);
    cache.ensure(session, theme.dark, colours.0, colours.1);

    if cache.vertices.is_empty() {
        ui.label("no spikes");
        return None;
    }

    let t_span = (cache.t1 - cache.t0).max(1e-6);
    let n_vertices = cache.vertices.len() as u32;
    caption(
        ui,
        &theme,
        format!(
            "{n_vertices} spikes · {} clusters · {} selected · {t_span:.1} s",
            cache.n_rows,
            selection.len()
        ),
    );
    // The visible time span and rows (row r_top at the top, r_span shown);
    // PanZoom's y runs bottom-up, rows top-down.
    let n_rows = cache.n_rows as f32;
    let ((v0, v1), (y0, y1)) = cache.pz.apply((cache.t0, cache.t0 + t_span), (0.0, n_rows));
    let (v_span, r_top, r_span) = ((v1 - v0).max(1e-6), n_rows - y1, (y1 - y0).max(1e-3));
    let (rect, resp) = alloc_plot(ui, egui::vec2(120.0, 160.0), Sense::click_and_drag());
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Other, true, "raster plot"));
    let resp = resp.on_hover_text(
        "Click to seek the trace to that time. Drag pans, wheel zooms, double-click resets.",
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme.plot_bg);
    // Row ids are drawn below; an empty y tick list only reserves the gutter.
    axes(
        &painter,
        rect,
        (v0, v1),
        (0.0, 1.0),
        "time (s)",
        "",
        &theme,
        AxisOpts {
            y_ticks: Some(&[]),
            ..AxisOpts::default()
        },
    );
    let data = axes_data_rect(rect, true);
    let row_h = data.height() / r_span;
    let row_y = |row: f32| data.top() + (row - r_top) * row_h;
    let dp = painter.with_clip_rect(data);

    // Selected rows: translucent band under the dots.
    for (slot, &c) in selection.iter().enumerate() {
        let Some(&row) = cache.rows.get(c.idx()) else {
            continue;
        };
        if row < 0 {
            continue;
        }
        let y0 = row_y(row as f32);
        let band = egui::Rect::from_min_max(
            egui::pos2(data.left(), y0),
            egui::pos2(data.right(), y0 + row_h.max(2.0)),
        );
        dp.rect_filled(band, 0.0, selection_colour(slot).gamma_multiply(0.2));
    }

    let callback = RasterCallback {
        vertices: std::sync::Arc::clone(&cache.vertices),
        generation: cache.generation,
        n_vertices,
        t0: v0,
        t_span: v_span,
        rows: (r_top, r_span),
    };
    ui.painter()
        .add(eframe::egui_wgpu::Callback::new_paint_callback(
            data, callback,
        ));

    // Selected clusters in their slot colour, drawn as 2 px ticks over the
    // neutral GPU dots.
    let total_sel: usize = selection
        .iter()
        .map(|&c| session.spike_times(c).len())
        .sum();
    let stride = (total_sel / 20_000).max(1);
    let inv_sr = 1.0 / session.provider.sample_rate().max(1.0);
    let tick_h = (row_h * 0.8).clamp(3.0, 24.0);
    for (slot, &c) in selection.iter().enumerate() {
        let Some(&row) = cache.rows.get(c.idx()) else {
            continue;
        };
        if row < 0 {
            continue;
        }
        let yc = row_y(row as f32 + 0.5);
        let colour = selection_colour(slot);
        for t in session.spike_times(c).iter().step_by(stride) {
            let x = data.left() + (t.as_f32() * inv_sr - v0) / v_span * data.width();
            dp.rect_filled(
                egui::Rect::from_center_size(egui::pos2(x, yc), egui::vec2(2.0, tick_h)),
                0.0,
                colour,
            );
        }
    }

    // Gutter: the cluster id of every Nth row (every row when they fit),
    // plus every selected row in its slot colour.
    let step = [1usize, 5, 10, 20, 50, 100, 200, 500, 1000]
        .into_iter()
        .find(|s| *s as f32 * row_h >= 13.0)
        .unwrap_or(1000);
    for (row, &id) in cache.row_ids.iter().enumerate() {
        let slot = selection.iter().position(|&c| c.0 == id);
        let y = row_y(row as f32 + 0.5);
        if (row % step != 0 && slot.is_none()) || !(data.top()..=data.bottom()).contains(&y) {
            continue;
        }
        painter.text(
            egui::pos2(data.left() - 6.0, y),
            egui::Align2::RIGHT_CENTER,
            id.to_string(),
            egui::FontId::proportional(TICK_PX),
            slot.map_or(theme.caption, selection_colour),
        );
    }

    // Current trace window, so the raster doubles as a scrubber.
    let sr = session.provider.sample_rate().max(1.0);
    let w0 = trace_window.0 as f32 / sr;
    let w1 = (trace_window.0 + trace_window.1 as u64) as f32 / sr;
    let x0 = data.left() + ((w0 - v0) / v_span).clamp(0.0, 1.0) * data.width();
    let x1 = data.left() + ((w1 - v0) / v_span).clamp(0.0, 1.0) * data.width();
    let marker = egui::Rect::from_min_max(
        egui::pos2(x0 - 1.0, data.top()),
        egui::pos2(x1.max(x0 + 2.0), data.bottom()),
    );
    dp.rect_stroke(
        marker,
        0.0,
        Stroke::new(1.0_f32, theme.text.gamma_multiply(0.7)),
        egui::StrokeKind::Inside,
    );

    // Click → seek. Map x in the data area to time, return as samples.
    *scheme_delta += crate::theme::scheme_wheel(ui, &resp);
    cache.pz.interact(&resp, data);
    // phy: Ctrl+click selects the row's cluster, Shift+click adds it.
    let mut seek_to = None;
    if resp.clicked() {
        if let Some(p) = resp.interact_pointer_pos() {
            let m = ui.input(|i| i.modifiers);
            let row = ((p.y - data.top()) / row_h + r_top).floor() as i32;
            let cluster = cache
                .rows
                .iter()
                .position(|&r| r == row)
                .map(|c| ClusterId(c as u32));
            seek_to = match cluster {
                Some(c) if m.command || m.shift => Some(RasterClick::Select(c, m.shift)),
                _ => {
                    let frac = ((p.x - data.left()) / data.width()).clamp(0.0, 1.0);
                    let t_sec = v0 + frac * v_span;
                    Some(RasterClick::Seek(
                        (t_sec * session.provider.sample_rate()).max(0.0) as u64,
                    ))
                }
            };
        }
    }
    seek_to
}

/// What a Raster click asks for.
pub enum RasterClick {
    /// Plain click: move the trace to this sample.
    Seek(u64),
    /// Ctrl+click selects the cluster; Shift+click (`true`) adds it.
    Select(ClusterId, bool),
}

// ---- FiringRateView -----------------------------------------------------

/// Per-cluster firing rate over the recording, plotted as overlaid line
/// graphs colour-coded by cluster. Useful for spotting drift, electrode
/// failure, and post-stimulus modulation — all phy parity.
pub fn firing_rate_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    params: &mut crate::prefs::RateParams,
) {
    let theme = Theme::from_ui(ui);
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }
    let sr = session.provider.sample_rate().max(1.0);
    let total_samples = session.provider.n_samples().0;
    if total_samples == 0 {
        ui.label("recording has zero length");
        return;
    }
    let total_s = total_samples as f32 / sr;

    // phy FiringRateView: `n_bins` over the recording (`:frn`, default
    // 200), x in the chosen unit (v2.1).
    ui.horizontal(|ui| {
        ui.add(
            egui::DragValue::new(&mut params.n_bins)
                .range(2..=5000)
                .prefix("bins "),
        );
        for unit in crate::prefs::TimeUnit::ALL {
            ui.selectable_value(&mut params.unit, unit, unit.label());
        }
    });
    // `n_bins` over the visible span: the `:frmin`/`:frmax` range, then pan/zoom.
    let bins = params.n_bins.clamp(2, 5000);
    let ((x0, x1), _) = params.pz.apply(params.x_range(total_s), (0.0, 1.0));
    let bin_s = (x1 - x0) / bins as f32;
    let unit = params.unit;

    // Bin every selected cluster's spike train into the same axis.
    let mut histograms: Vec<Vec<u32>> = Vec::with_capacity(selection.len());
    let mut max_count = 1u32;
    for &c in selection {
        let times = session.spike_times(c);
        let mut h = vec![0u32; bins];
        for &t in times {
            let u = (t.0 as f64 / sr as f64 - x0 as f64) / bin_s as f64;
            if u >= 0.0 && (u as usize) < bins {
                h[u as usize] += 1;
            }
        }
        if let Some(&m) = h.iter().max() {
            if m > max_count {
                max_count = m;
            }
        }
        histograms.push(h);
    }

    let width = if bin_s >= 1.0 {
        format!("{bin_s:.1} s")
    } else {
        format!("{:.1} ms", bin_s * 1e3)
    };
    caption(
        ui,
        &theme,
        format!(
            "firing rate · {bins} bins × {width} · {}",
            legend(selection)
        ),
    );
    let (rect, resp) = alloc_plot(ui, egui::vec2(160.0, 160.0), Sense::drag());
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Other, true, "rate plot"));
    let resp = resp.on_hover_text("Drag pans, wheel zooms in time, double-click resets.");
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme.plot_bg);
    let data = axes_data_rect(rect, true);
    // Only time pans and zooms; y always fits the visible bins.
    params.pz.interact(&resp, data);

    // 15% headroom above the tallest bin.
    let y_top = max_count as f32 / bin_s * 1.15;
    draw_axes_grid(
        &painter,
        rect,
        (x0 / unit.seconds(), x1 / unit.seconds()),
        (0.0, y_top),
        &format!("time ({})", unit.label()),
        "Hz",
        &theme,
    );
    let dp = painter.with_clip_rect(data);
    let fill = if histograms.len() == 1 { 0.6 } else { 0.2 };
    for (ci, hist) in histograms.iter().enumerate() {
        let rates: Vec<f32> = hist.iter().map(|&c| c as f32 / bin_s).collect();
        step_hist(&dp, data, &rates, y_top, selection_colour(ci), fill);
    }
}

// ---- ProbeView ----------------------------------------------------------

/// 2-D probe layout with channels drawn as small dots. For each selected
/// cluster, the [`WaveformCache`] gives the peak channel and a
/// colour-matching ring is drawn at that position. The cache is refreshed
/// here too, so the rings never lag behind the selection.
pub fn probe_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    channel_positions: &[[f32; 2]],
    waveforms: &mut WaveformCache,
) {
    let theme = Theme::from_ui(ui);
    let pre = waveform_window_samples(session.provider.sample_rate());
    waveforms.ensure(session, selection, pre, pre);
    if channel_positions.is_empty() {
        empty_plot(
            ui,
            &theme,
            "channel_positions.npy not loaded — probe layout unavailable",
        );
        return;
    }

    // Probe extents → screen rect with a small margin.
    let mut x_min = f32::INFINITY;
    let mut x_max = f32::NEG_INFINITY;
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for &[x, y] in channel_positions {
        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }
    let x_span = (x_max - x_min).max(1e-6);
    let y_span = (y_max - y_min).max(1e-6);
    caption(
        ui,
        &theme,
        format!(
            "{} channels · x [{:.0}, {:.0}] · y [{:.0}, {:.0}] µm · 1:1 scale",
            channel_positions.len(),
            x_min,
            x_max,
            y_min,
            y_max,
        ),
    );
    let (rect, _resp) = alloc_plot(ui, egui::vec2(160.0, 220.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme.plot_bg);
    let data = axes_data_rect(rect, true);
    // Room on the sides for "chN" labels.
    let inner = data.shrink2(egui::vec2(40.0, 16.0));

    // One isotropic scale (px per µm), the probe centred in `inner`.
    // y grows upward on probes, so deeper channels sit at the bottom.
    let s = (inner.width() / x_span).min(inner.height() / y_span);
    let ox = inner.center().x - x_span * s / 2.0 - x_min * s;
    let oy = inner.center().y + y_span * s / 2.0 + y_min * s;
    let project = |x: f32, y: f32| egui::pos2(ox + x * s, oy - y * s);
    // The µm range the axes must span for ticks to line up with `project`.
    let x_rng = ((data.left() - ox) / s, (data.right() - ox) / s);
    let y_rng = ((oy - data.bottom()) / s, (oy - data.top()) / s);
    draw_axes_grid(&painter, rect, x_rng, y_rng, "µm", "µm", &theme);
    let painter = painter.with_clip_rect(data.expand2(egui::vec2(Y_GUTTER, 0.0)));

    // Label every channel when the pitch leaves room, else just the peaks.
    let mut ys: Vec<f32> = channel_positions.iter().map(|p| p[1]).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min_dy = ys
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| *d > 1e-3)
        .fold(f32::INFINITY, f32::min);
    let label_all = min_dy * s >= 13.0;
    let x_mid = (x_min + x_max) / 2.0;
    let peaks: Vec<Option<usize>> = (0..selection.len())
        .map(|ci| {
            waveforms
                .peak_channel(ci)
                .filter(|&ch| ch < channel_positions.len())
        })
        .collect();
    let label = |ch: usize, x: f32, colour: Color32| {
        let [px, py] = channel_positions[ch];
        let (side, align) = if px <= x_mid {
            (-8.0, egui::Align2::RIGHT_CENTER)
        } else {
            (8.0, egui::Align2::LEFT_CENTER)
        };
        let _ = x;
        painter.text(
            project(px, py) + egui::vec2(side, 0.0),
            align,
            format!("ch{ch}"),
            egui::FontId::proportional(TICK_PX),
            colour,
        );
    };

    for (ch, &[x, y]) in channel_positions.iter().enumerate() {
        painter.circle_filled(project(x, y), 3.5, theme.caption);
        if label_all {
            label(ch, x, theme.caption);
        }
    }

    // Ring each selected cluster's peak channel in its slot colour.
    for (ci, peak) in peaks.iter().enumerate() {
        let Some(ch) = *peak else { continue };
        let [px, py] = channel_positions[ch];
        let centre = project(px, py);
        let colour = selection_colour(ci);
        // Slightly larger rings per slot so clusters sharing a peak channel
        // don't perfectly occlude each other.
        painter.circle_stroke(centre, 7.0 + ci as f32 * 1.5, Stroke::new(1.6_f32, colour));
        painter.circle_filled(centre, 3.5, colour);
        if !label_all {
            label(ch, px, colour);
        }
    }
}

// ---- ClusterStatistics --------------------------------------------------

/// Histograms of three scalar metrics across *every* cluster: spike count,
/// mean amplitude, ISI<1ms count. Bars are coloured by their relative bin
/// rank so eye-balling outliers is easy.
pub fn cluster_statistics_view<P: DataProvider>(ui: &mut Ui, session: &Session<P>) {
    cluster_statistics_view_for(ui, session, &[]);
}

/// [`cluster_statistics_view`] with the selected clusters' values marked
/// on each histogram as a slot-colour line.
pub fn cluster_statistics_view_for<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
) {
    use sorrel_compute::{isi_violations, mean_amplitude};
    let n = session.n_clusters() as usize;
    if n == 0 {
        ui.label("no clusters");
        return;
    }

    // Collect three series, indexed by cluster id (NaN = no value).
    let refractory_samples = (session.provider.sample_rate() * 0.001).round() as u64;
    let mut spike_counts: Vec<f32> = Vec::with_capacity(n);
    let mut mean_amps: Vec<f32> = Vec::with_capacity(n);
    let mut isi_viols: Vec<f32> = Vec::with_capacity(n);
    for c in (0..n as u32).map(ClusterId) {
        spike_counts.push(session.spike_times(c).len() as f32);
        let amps = session.spike_amplitudes(c);
        mean_amps.push(if amps.is_empty() {
            f32::NAN
        } else {
            mean_amplitude(amps)
        });
        isi_viols.push(isi_violations(session.spike_times(c), refractory_samples) as f32);
    }

    let marks = |vals: &[f32]| -> Vec<(usize, ClusterId, f32)> {
        selection
            .iter()
            .enumerate()
            .filter_map(|(slot, &c)| {
                let v = *vals.get(c.idx())?;
                v.is_finite().then_some((slot, c, v))
            })
            .collect()
    };
    let has_amps = mean_amps.iter().any(|v| v.is_finite());
    let panels = if has_amps { 3.0 } else { 2.0 };
    let gap = 8.0;
    // Split the pane's height evenly.
    let each = ((fill_height(ui) - gap * (panels - 1.0)) / panels).max(96.0);
    histogram_panel(
        ui,
        "spike count",
        &spike_counts,
        30,
        &marks(&spike_counts),
        0,
        each,
    );
    ui.add_space(gap);
    if has_amps {
        histogram_panel(
            ui,
            "mean amplitude",
            &mean_amps,
            30,
            &marks(&mean_amps),
            1,
            each,
        );
        ui.add_space(gap);
    }
    histogram_panel(
        ui,
        "ISI<1ms count",
        &isi_viols,
        30,
        &marks(&isi_viols),
        0,
        each,
    );
}

pub(crate) fn histogram_panel(
    ui: &mut Ui,
    title: &str,
    values: &[f32],
    bins: usize,
    markers: &[(usize, ClusterId, f32)],
    decimals: usize,
    height: f32,
) {
    let theme = Theme::from_ui(ui);
    let values: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if values.is_empty() {
        return;
    }
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let mut max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if (max - min).abs() < 1e-6 {
        // All values identical — pad so the bin range isn't zero-width.
        max = min + 1.0;
    }
    let span = max - min;
    let inv_bin = bins as f32 / span;
    let mut h = vec![0u32; bins];
    for &v in &values {
        let i = (((v - min) * inv_bin) as usize).min(bins - 1);
        h[i] += 1;
    }
    let max_count = h.iter().copied().max().unwrap_or(1).max(1) as f32;

    caption(
        ui,
        &theme,
        format!(
            "{title} · n={} · range {min:.decimals$}–{max:.decimals$}",
            values.len()
        ),
    );
    let (rect, _resp) = ui.allocate_exact_size(
        egui::vec2(ui.available_width().max(120.0), (height - 20.0).max(70.0)),
        Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme.plot_bg);
    let data = axes_data_rect(rect, true);
    let y_top = max_count * 1.08;
    let yt = int_ticks(max_count, 2);
    let xt = [min, (min + max) / 2.0, max];
    axes(
        &painter,
        rect,
        (min, max),
        (0.0, y_top),
        "",
        "",
        &theme,
        AxisOpts {
            grid: true,
            x_ticks: Some(&xt),
            y_ticks: Some(&yt),
            x_decimals: Some(decimals),
            ..AxisOpts::default()
        },
    );
    let dp = painter.with_clip_rect(data.expand2(egui::vec2(2.0, 0.0)));

    // Neutral bars, so the slot-colour markers carry the colour.
    let bar = blend(theme.plot_bg, theme.text, 0.5);
    let bw = data.width() / bins as f32;
    for (i, &count) in h.iter().enumerate() {
        let bx0 = data.left() + bw * i as f32;
        let by0 = data.bottom() - count as f32 / y_top * data.height();
        dp.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(bx0, by0),
                egui::pos2(bx0 + bw * 0.95, data.bottom()),
            ),
            0.0,
            bar,
        );
    }

    // The selected clusters' values.
    for (k, &(slot, c, v)) in markers.iter().enumerate() {
        let x = data.left() + (v - min) / span * data.width();
        let colour = selection_colour(slot);
        dp.line_segment(
            [egui::pos2(x, data.top()), egui::pos2(x, data.bottom())],
            Stroke::new(1.5_f32, colour),
        );
        let (align, dx) = if x > data.center().x {
            (egui::Align2::RIGHT_TOP, -4.0)
        } else {
            (egui::Align2::LEFT_TOP, 4.0)
        };
        dp.text(
            egui::pos2(x + dx, data.top() + 2.0 + 14.0 * k as f32),
            align,
            format!("c{c} · {v:.decimals$}"),
            egui::FontId::proportional(TICK_PX),
            colour,
        );
    }
}

// ---- SuggestView (statistical merge/split candidates) -------------------

/// Action emitted when the curator clicks a row in the suggest panel.
#[derive(Clone, Debug)]
pub enum SuggestAction {
    /// Replace the active selection with these clusters (typically a merge
    /// pair the user is reviewing) — the curator inspects, then triggers
    /// the merge themselves with the existing keybinding.
    Select(Vec<ClusterId>),
    /// Apply the merge directly: reassign all spikes from `sources` into
    /// `target`. The UI dispatches a `CurationCommand::Merge` for this.
    ApplyMerge {
        sources: Vec<ClusterId>,
        target: ClusterId,
    },
    /// Apply every merge in the cache that exceeds `threshold` and whose
    /// preview does not warn, as a single atomic batch. Reversible with
    /// one Undo.
    ApplyHighConfidenceMerges { threshold: f32 },
    /// Apply a GMM-derived split: move the listed local spike indices of
    /// `cluster` to a fresh cluster id. The UI dispatches a
    /// `CurationCommand::Split` for this.
    ApplyGmmSplit {
        cluster: ClusterId,
        spike_idx: Vec<u32>,
    },
}

/// Lightweight summary of a merge preview — only the fields the UI needs.
/// We don't keep the full `MergePreview` because it would force the cache
/// to allocate a `QualityBreakdown` per row.
#[derive(Clone, Copy, Debug, Default)]
pub struct MergePreviewSummary {
    pub composite_delta: f32,
    pub contamination_delta: f32,
    pub warns: bool,
}

/// Ranked merge/split candidates plus predicted merge outcomes, computed
/// on a worker thread against one curation version.
#[derive(Default)]
pub struct SuggestResult {
    merges: Vec<MergeCandidate>,
    splits: Vec<SplitCandidate>,
    /// Indexed parallel to `merges` — predicted post-merge deltas.
    merge_previews: Vec<MergePreviewSummary>,
}

impl SuggestResult {
    fn compute<P: DataProvider>(
        session: &Session<P>,
        config: &SuggestConfig,
        memo: &mut SuggestMemo,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Self {
        use std::sync::atomic::Ordering;
        let merges = rank_merge_candidates_memo(session, config, memo);
        if cancel.load(Ordering::Relaxed) {
            return Self::default();
        }
        let splits = rank_split_candidates_memo(session, config, memo);
        if cancel.load(Ordering::Relaxed) {
            return Self::default();
        }
        use rayon::prelude::*;
        let merge_previews = merges
            .par_iter()
            .map(|m| {
                if cancel.load(Ordering::Relaxed) {
                    return MergePreviewSummary::default();
                }
                let p = preview_merge(session, &[m.a], m.b);
                MergePreviewSummary {
                    composite_delta: p.composite_delta(),
                    contamination_delta: p.contamination_delta(),
                    warns: p.warns(),
                }
            })
            .collect();
        Self {
            merges,
            splits,
            merge_previews,
        }
    }

    fn n_safe(&self, threshold: f32) -> usize {
        self.merges
            .iter()
            .zip(self.merge_previews.iter())
            .filter(|(m, p)| m.score >= threshold && !p.warns)
            .count()
    }
}

/// Job-backed cache of the suggestion lists, keyed on the curation
/// version (any merge/split/relabel/undo recomputes — the suggesters score
/// against the live state) plus a counter the "recompute" button bumps.
#[derive(Default)]
pub struct SuggestCache {
    memo: crate::jobs::Memo<(u64, u64), std::sync::Arc<SuggestResult>>,
    /// Per-pair and per-cluster scores kept across versions; only the worker
    /// thread locks it.
    scores: std::sync::Arc<std::sync::Mutex<SuggestMemo>>,
    /// Probe channel positions (µm); enables the drift-fragment merge source.
    pub positions: Vec<[f32; 2]>,
    force: u64,
    pub config: SuggestConfig,
    /// Current selection, set by the caller each frame so rows involving
    /// it are tinted in the cluster's slot colour.
    pub selection: Vec<ClusterId>,
}

impl SuggestCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// The result for the current version, or `None` while it computes.
    fn ensure<P: DataProvider>(
        &mut self,
        session: &Session<P>,
        jobs: &crate::jobs::JobCtx<Session<P>>,
    ) -> Option<std::sync::Arc<SuggestResult>> {
        let key = (session.version(), self.force);
        let config = self.config;
        let (scores, positions) = (self.scores.clone(), self.positions.clone());
        self.memo
            .ensure(key, &jobs.session, jobs.repaint(), move |s, cancel| {
                let mut memo = scores.lock().unwrap_or_else(|e| e.into_inner());
                if memo.positions != positions {
                    memo.positions = positions;
                }
                std::sync::Arc::new(SuggestResult::compute(s, &config, &mut memo, cancel))
            })
            .cloned()
    }

    /// Drop the result and recompute on the next frame.
    pub fn invalidate(&mut self) {
        self.force = self.force.wrapping_add(1);
        self.memo.invalidate();
    }

    /// Stop the worker so a curation write is not kept waiting.
    pub fn cancel(&mut self) {
        self.memo.cancel();
    }

    pub fn pending(&self) -> bool {
        self.memo.pending()
    }

    /// Pick out the merge pairs whose suggester score is at least
    /// `threshold` and whose preview does not warn (no contamination
    /// regression, no composite drop). Returns `(source, target)` pairs
    /// in ranked order. We deduplicate so a cluster only appears once on
    /// either side: applying overlapping merges in one batch would change
    /// what the second merge meant.
    pub fn high_confidence_merges(&self, threshold: f32) -> Vec<(ClusterId, ClusterId)> {
        use std::collections::HashSet;
        let Some(result) = self.memo.last() else {
            return Vec::new();
        };
        let mut taken: HashSet<ClusterId> = HashSet::new();
        let mut out = Vec::new();
        for (m, p) in result.merges.iter().zip(result.merge_previews.iter()) {
            if m.score < threshold || p.warns {
                continue;
            }
            if taken.contains(&m.a) || taken.contains(&m.b) {
                continue;
            }
            taken.insert(m.a);
            taken.insert(m.b);
            out.push((m.a, m.b));
        }
        out
    }
}

/// Render the suggest panel — two tables (merges, splits) with per-row
/// evidence and click-to-act buttons. Returns the action the curator
/// triggered, if any.
pub fn suggest_view<P, F>(
    ui: &mut Ui,
    session: &Session<P>,
    cache: &mut SuggestCache,
    label_str: F,
    jobs: &crate::jobs::JobCtx<Session<P>>,
) -> Option<SuggestAction>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    let result = cache.ensure(session, jobs);
    let mut action: Option<SuggestAction> = None;
    let sel = cache.selection.clone();

    ui.horizontal(|ui| {
        ui.heading("Statistical suggestions");
        match &result {
            Some(r) => {
                ui.weak(format!(
                    "{} merges · {} splits · refractory {:.1} ms · CCG ±{:.0} ms",
                    r.merges.len(),
                    r.splits.len(),
                    cache.config.refractory_seconds * 1000.0,
                    cache.config.ccg_max_lag_seconds * 1000.0,
                ));
            }
            None => {
                ui.spinner();
                ui.weak("ranking candidates in the background…");
            }
        }
        if ui.small_button("recompute").clicked() {
            cache.invalidate();
        }
        let n_safe = result.as_ref().map_or(0, |r| r.n_safe(0.7));
        let label = format!("apply {n_safe} high-conf merges");
        let response = ui.add_enabled(n_safe > 0, egui::Button::new(label).small());
        if response
            .on_hover_text(
                "Applies every score≥0.70 merge whose predicted Δquality is not negative. \
                 Atomic — one Undo reverts the whole batch.",
            )
            .clicked()
        {
            action = Some(SuggestAction::ApplyHighConfidenceMerges { threshold: 0.7 });
        }
    });
    ui.separator();
    let Some(result) = result else {
        return action;
    };

    egui::CollapsingHeader::new("Merge candidates")
        .default_open(true)
        .show(ui, |ui| {
            if result.merges.is_empty() {
                ui.weak(
                    "No merge candidates above threshold — increase recording length, \
                     or seed amplitudes for stronger evidence.",
                );
            } else {
                let merge_action =
                    render_merge_table(ui, &result.merges, &result.merge_previews, &sel);
                if action.is_none() {
                    action = merge_action;
                }
            }
        });

    ui.add_space(6.0);
    egui::CollapsingHeader::new("Split candidates")
        .default_open(true)
        .show(ui, |ui| {
            if result.splits.is_empty() {
                ui.weak("No split candidates above threshold.");
            } else {
                let split_action =
                    render_split_table(ui, session, &result.splits, &label_str, &sel);
                if action.is_none() {
                    action = split_action;
                }
            }
        });

    action
}

/// Tint the cell when any of `ids` is selected, in that cluster's slot colour.
pub(crate) fn tint_for(ui: &Ui, sel: &[ClusterId], ids: &[ClusterId]) {
    if let Some(slot) = sel.iter().position(|c| ids.contains(c)) {
        let sp = ui.spacing().item_spacing;
        ui.painter().rect_filled(
            ui.max_rect().expand2(sp * 0.5),
            0.0,
            selection_colour(slot).gamma_multiply(0.22),
        );
    }
}

pub(crate) fn render_merge_table(
    ui: &mut Ui,
    merges: &[MergeCandidate],
    previews: &[MergePreviewSummary],
    sel: &[ClusterId],
) -> Option<SuggestAction> {
    let theme = Theme::from_ui(ui);
    use egui_extras::{Column, TableBuilder};
    let mut action: Option<SuggestAction> = None;
    TableBuilder::new(ui)
        .striped(true)
        .resizable(true)
        .column(Column::auto().at_least(56.0))
        .column(Column::auto().at_least(56.0))
        .column(Column::auto().at_least(54.0))
        .column(Column::auto().at_least(54.0))
        .column(Column::auto().at_least(54.0))
        .column(Column::auto().at_least(64.0))
        .column(Column::auto().at_least(64.0))
        .column(Column::auto().at_least(64.0))
        .column(Column::auto().at_least(140.0))
        .header(22.0, |mut h| {
            h.col(|ui| {
                ui.strong("a");
            });
            h.col(|ui| {
                ui.strong("b");
            });
            h.col(|ui| {
                num_head(ui, "score");
            });
            h.col(|ui| {
                num_head(ui, "dip z");
            });
            h.col(|ui| {
                num_head(ui, "KS").on_hover_text("amplitude KS statistic");
            });
            h.col(|ui| {
                num_head(ui, "Δamp").on_hover_text("difference in mean amplitude");
            });
            h.col(|ui| {
                num_head(ui, "Δq")
                    .on_hover_text("predicted change in composite quality after merge");
            });
            h.col(|ui| {
                num_head(ui, "Δcon").on_hover_text("predicted change in refractory contamination");
            });
            h.col(|ui| {
                ui.strong("action");
            });
        })
        .body(|mut body| {
            for (idx, m) in merges.iter().enumerate() {
                let preview = previews.get(idx).copied().unwrap_or_default();
                body.row(20.0, |mut row| {
                    row.col(|ui| {
                        tint_for(ui, sel, &[m.a, m.b]);
                        ui.label(format!("c{}", m.a));
                    });
                    row.col(|ui| {
                        tint_for(ui, sel, &[m.a, m.b]);
                        ui.label(format!("c{}", m.b));
                    });
                    row.col(|ui| {
                        num_cell(ui, format!("{:.2}", m.score), None);
                    });
                    row.col(|ui| {
                        num_cell(ui, format!("{:+.1}", m.ccg_dip_z), None);
                    });
                    row.col(|ui| {
                        num_cell(ui, format!("{:.2}", m.amp_ks), None);
                    });
                    row.col(|ui| {
                        num_cell(ui, format!("{:.2}", m.amp_mean_delta), None);
                    });
                    row.col(|ui| {
                        let txt = format!("{:+.2}", preview.composite_delta);
                        let colour = if preview.composite_delta < -0.05 {
                            theme.danger
                        } else if preview.composite_delta > 0.02 {
                            theme.good
                        } else {
                            theme.caption
                        };
                        num_cell(ui, txt, Some(colour));
                    });
                    row.col(|ui| {
                        let txt = format!("{:+.3}", preview.contamination_delta);
                        let colour = if preview.contamination_delta > 0.05 {
                            theme.danger
                        } else {
                            theme.caption
                        };
                        num_cell(ui, txt, Some(colour));
                    });
                    row.col(|ui| {
                        ui.horizontal(|ui| {
                            if ui.small_button("review").clicked() {
                                action = Some(SuggestAction::Select(vec![m.a, m.b]));
                            }
                            let merge_label = if preview.warns { "merge ⚠" } else { "merge" };
                            let tt = if preview.warns {
                                format!(
                                    "Predicted Δquality {:+.2}, Δcontam {:+.3}. \
                                 The merge looks risky — review before applying.",
                                    preview.composite_delta, preview.contamination_delta,
                                )
                            } else {
                                "Reassigns all spikes from a into b. Reversible via undo.".into()
                            };
                            if ui.small_button(merge_label).on_hover_text(tt).clicked() {
                                action = Some(SuggestAction::ApplyMerge {
                                    sources: vec![m.a],
                                    target: m.b,
                                });
                            }
                        });
                    });
                });
            }
        });
    action
}

pub(crate) fn render_split_table<P, F>(
    ui: &mut Ui,
    session: &Session<P>,
    splits: &[SplitCandidate],
    label_str: &F,
    sel: &[ClusterId],
) -> Option<SuggestAction>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    let theme = Theme::from_ui(ui);
    use egui_extras::{Column, TableBuilder};
    let mut action: Option<SuggestAction> = None;
    TableBuilder::new(ui)
        .striped(true)
        .resizable(true)
        .column(Column::auto().at_least(56.0))
        .column(Column::auto().at_least(54.0))
        .column(Column::auto().at_least(56.0))
        .column(Column::auto().at_least(52.0))
        .column(Column::auto().at_least(64.0))
        .column(Column::auto().at_least(56.0))
        .column(Column::auto().at_least(58.0))
        .column(Column::auto().at_least(56.0))
        .column(Column::auto().at_least(140.0))
        .header(22.0, |mut h| {
            h.col(|ui| {
                ui.strong("cluster");
            });
            h.col(|ui| {
                num_head(ui, "score");
            });
            h.col(|ui| {
                num_head(ui, "BC").on_hover_text("bimodality coefficient");
            });
            h.col(|ui| {
                num_head(ui, "dip").on_hover_text(
                    "Hartigan dip on amplitudes (0–0.25); higher = stronger departure from unimodality",
                );
            });
            h.col(|ui| {
                num_head(ui, "contam");
            });
            h.col(|ui| {
                num_head(ui, "|drift|");
            });
            h.col(|ui| {
                num_head(ui, "ΔBIC").on_hover_text(
                    "BIC(k=1) − BIC(k=2) on amplitudes; >6 = strong evidence for splitting",
                );
            });
            h.col(|ui| {
                ui.strong("label");
            });
            h.col(|ui| {
                ui.strong("action");
            });
        })
        .body(|mut body| {
            for s in splits {
                body.row(20.0, |mut row| {
                    row.col(|ui| {
                        tint_for(ui, sel, &[s.cluster]);
                        ui.label(format!("c{}", s.cluster));
                    });
                    row.col(|ui| {
                        num_cell(ui, format!("{:.2}", s.score), None);
                    });
                    row.col(|ui| {
                        num_cell(ui, format!("{:.2}", s.amp_bimodality), None);
                    });
                    row.col(|ui| {
                        num_cell(ui, format!("{:.3}", s.amp_dip), None);
                    });
                    row.col(|ui| {
                        num_cell(ui, format!("{:.3}", s.contamination), None);
                    });
                    row.col(|ui| {
                        num_cell(ui, format!("{:.2}", s.drift_corr), None);
                    });
                    row.col(|ui| {
                        if s.gmm_bic_delta.is_finite() {
                            let colour = if s.gmm_bic_delta >= 6.0 {
                                theme.good
                            } else {
                                theme.caption
                            };
                            num_cell(ui, format!("{:+.1}", s.gmm_bic_delta), Some(colour));
                        } else {
                            num_cell(ui, "—".to_owned(), None);
                        }
                    });
                    row.col(|ui| {
                        ui.label(label_str(&session.label(s.cluster).unwrap_or_default()));
                    });
                    row.col(|ui| {
                      ui.horizontal(|ui| {
                        if ui
                            .small_button("inspect")
                            .on_hover_text(
                                "Selects this cluster so you can lasso-split it \
                                 in the Features tab.",
                            )
                            .clicked()
                        {
                            action = Some(SuggestAction::Select(vec![s.cluster]));
                        }
                        if !s.bipartition_minor_idx.is_empty()
                            && ui
                                .small_button("split")
                                .on_hover_text(format!(
                                    "Move {} amplitude-outlier spikes to a fresh cluster. \
                                     Reversible via undo.",
                                    s.bipartition_minor_idx.len(),
                                ))
                                .clicked()
                        {
                            action = Some(SuggestAction::ApplyGmmSplit {
                                cluster: s.cluster,
                                spike_idx: s.bipartition_minor_idx.clone(),
                            });
                        }
                      });
                    });
                });
            }
        });
    action
}

// ---- QualityView (per-cluster composite score breakdown) ---------------

/// Job-backed memo of [`cluster_quality`] for the selection, keyed on
/// selection + curation version: the PC-isolation k-NN plus the disk-cache
/// probe it performs are far too heavy for the UI thread.
#[derive(Default)]
pub struct QualityCache {
    memo: crate::jobs::Memo<(Vec<ClusterId>, u64), std::sync::Arc<Vec<QualityBreakdown>>>,
}

impl QualityCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure<P: DataProvider>(
        &mut self,
        session: &Session<P>,
        selection: &[ClusterId],
        jobs: &crate::jobs::JobCtx<Session<P>>,
    ) -> Option<std::sync::Arc<Vec<QualityBreakdown>>> {
        let key = (selection.to_vec(), session.version());
        let clusters = selection.to_vec();
        self.memo
            .ensure(key, &jobs.session, jobs.repaint(), move |s, cancel| {
                use rayon::prelude::*;
                use std::sync::atomic::Ordering;
                let rows = clusters
                    .par_iter()
                    .map(|&c| {
                        if cancel.load(Ordering::Relaxed) {
                            QualityBreakdown::default()
                        } else {
                            cluster_quality(s, c)
                        }
                    })
                    .collect();
                std::sync::Arc::new(rows)
            })
            .cloned()
    }

    pub fn cancel(&mut self) {
        self.memo.cancel();
    }

    pub fn pending(&self) -> bool {
        self.memo.pending()
    }
}

/// Render a quality-score panel for the selected clusters: the composite
/// score plus a breakdown into the six contributing factors. Hovering each
/// score reveals the underlying raw metric.
pub fn quality_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    cache: &mut QualityCache,
    jobs: &crate::jobs::JobCtx<Session<P>>,
) {
    let theme = Theme::from_ui(ui);
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }
    let Some(rows) = cache.ensure(session, selection, jobs) else {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.weak("computing quality metrics in the background…");
        });
        return;
    };
    let has_features = session.has_pc_features();
    // Caption first: below the table it wrapped at the table's width and
    // was clipped by the pane.
    ui.add(
        egui::Label::new(
            egui::RichText::new("Higher is better · hover a score for its metric")
                .size(CAPTION_PX)
                .color(theme.caption),
        )
        .truncate(),
    )
    .on_hover_text(
        "Higher is better. Composite = geometric mean of all available sub-scores — \
         one weak dimension drags it down. Hover any score for the raw metric. \
         The lowest sub-score of each cluster is bold.",
    );
    ui.add_space(4.0);

    use egui_extras::{Column, TableBuilder};
    egui::ScrollArea::horizontal().show(ui, |ui| {
        let mut heads = vec!["composite", "clean", "present", "cover", "SNR", "complete", "stable"];
        if has_features {
            heads.push("isolated");
        }
        // Right-aligned cells make `Column::auto` grow to the pane width, so
        // size each column to its header instead.
        let mut tb = TableBuilder::new(ui)
            .striped(true)
            .resizable(false)
            .column(Column::exact(48.0));
        for name in &heads {
            tb = tb.column(Column::exact((name.len() as f32 * 6.5 + 6.0).max(44.0)));
        }
        tb.header(24.0, |mut h| {
            h.col(|ui| {
                ui.strong("cluster");
            });
            for name in heads {
                h.col(|ui| {
                    num_head(ui, name);
                });
            }
        })
        .body(|mut body| {
            for (slot, &c) in selection.iter().enumerate() {
                let Some(&q) = rows.get(slot) else {
                    continue;
                };
                let composite = q.composite();
                let iso = if has_features {
                    q.isolation
                        .filter(|i| i.has_evidence() && i.score().is_finite())
                } else {
                    None
                };
                let mut subs = vec![
                    (
                        q.contamination_score,
                        format!(
                            "refractory contamination = {:.3}\nISI<refrac = {}",
                            q.raw_contamination, q.raw_isi_violations
                        ),
                    ),
                    (
                        q.presence_score,
                        format!(
                            "presence ratio = {:.2}\npresence CV = {:.2}",
                            q.raw_presence_ratio, q.raw_presence_cv
                        ),
                    ),
                    (
                        q.coverage_score,
                        format!("largest silent gap = {:.2}", q.raw_silent_gap),
                    ),
                    (q.snr_score, format!("amplitude SNR = {:.2}", q.raw_snr)),
                    (
                        q.completeness_score,
                        format!("amplitude cutoff = {:.3}", q.raw_amp_cutoff),
                    ),
                    (
                        q.stability_score,
                        format!("drift correlation = {:+.2}", q.raw_drift_corr),
                    ),
                ];
                if let Some(i) = iso {
                    subs.push((
                        i.score(),
                        format!(
                            "isolation² = {:.1}\nL-ratio = {:.3}\nNN-isolation = {:.2}\nN_in = {}, N_out = {}",
                            i.isolation_distance_sq, i.l_ratio, i.nn_isolation, i.n_in, i.n_out,
                        ),
                    ));
                }
                let min = subs.iter().map(|s| s.0).fold(f32::INFINITY, f32::min);
                body.row(24.0, |mut row| {
                    row.col(|ui| {
                        ui.horizontal(|ui| {
                            swatch(ui, selection_colour(slot));
                            ui.label(format!("c{c}"));
                        });
                    });
                    row.col(|ui| {
                        num_cell(
                            ui,
                            format!("{composite:.2}"),
                            Some(theme.quality_colour(composite)),
                        );
                    });
                    for (s, tip) in &subs {
                        row.col(|ui| score_cell(ui, &theme, *s, *s <= min, tip.clone()));
                    }
                    if has_features && iso.is_none() {
                        row.col(|ui| {
                            right_aligned(ui, |ui| {
                                ui.label("—")
                                    .on_hover_text("not enough spikes / no PC evidence");
                            });
                        });
                    }
                });
            }
        });
    });
}

/// A coloured, right-aligned sub-score with its raw metric on hover.
pub(crate) fn score_cell(ui: &mut Ui, theme: &Theme, s: f32, bold: bool, tip: String) {
    right_aligned(ui, |ui| {
        let mut t = egui::RichText::new(format!("{s:.2}"))
            .monospace()
            .color(theme.quality_colour(s));
        if bold {
            t = t.strong();
        }
        ui.label(t).on_hover_text(tip);
    });
}

// ---- DriftMapView (amp vs time scatter colour-coded by density) ---------

/// Per-spike amplitude (y) vs spike time (x) scatter, with point density
/// reflected in alpha. The single most useful drift diagnostic in phylib —
/// a healthy cluster paints a flat horizontal cloud; a drifting cluster
/// shows a sloped or fragmented one.
///
/// Renders in pure egui — fast enough for tens of thousands of points
/// because we sub-sample at a fixed point budget. Selected clusters are
/// overlaid with the standard palette.
pub fn drift_map_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    depths: Option<&SpikeDepths>,
) {
    let theme = Theme::from_ui(ui);
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }
    // y = spike depth (µm) when the backend can estimate it, else amplitude.
    let ys = |c: ClusterId| -> &[f32] {
        match depths {
            Some(d) => d.per_cluster.get(c.idx()).map(Vec::as_slice).unwrap_or(&[]),
            None => session.spike_amplitudes(c),
        }
    };
    let (y_unit, y_name) = match depths.map(|d| d.source) {
        Some(DepthSource::PcWeighted) => (" µm", "depth (PC-weighted)"),
        Some(DepthSource::TemplatePeak) => (" µm", "depth (template peak)"),
        None => (" amp", "amplitude"),
    };
    let inv_sr = 1.0_f32 / session.provider.sample_rate().max(1.0);

    // Determine global bounds across the selection.
    let mut t_min = f32::INFINITY;
    let mut t_max = f32::NEG_INFINITY;
    let mut a_min = f32::INFINITY;
    let mut a_max = f32::NEG_INFINITY;
    let mut total_spikes = 0usize;
    for &c in selection {
        let times = session.spike_times(c);
        let amps = ys(c);
        if times.is_empty() || amps.is_empty() {
            continue;
        }
        total_spikes += times.len();
        if let (Some(&first), Some(&last)) = (times.first(), times.last()) {
            t_min = t_min.min(first.as_f32() * inv_sr);
            t_max = t_max.max(last.as_f32() * inv_sr);
        }
        for &a in amps {
            if a < a_min {
                a_min = a;
            }
            if a > a_max {
                a_max = a;
            }
        }
    }
    if total_spikes == 0 || !t_min.is_finite() || !a_min.is_finite() {
        empty_plot(
            ui,
            &theme,
            if depths.is_some() {
                "no spike depths for this selection"
            } else {
                "amplitudes not seeded for this backend"
            },
        );
        return;
    }
    // Pad the y range so points don't touch the borders; a degenerate
    // range (all spikes at one depth) gets ±5 so the axis has ticks.
    if a_max - a_min < 1e-3 || (depths.is_some() && a_max - a_min < 1.0) {
        let mid = (a_min + a_max) / 2.0;
        (a_min, a_max) = (mid - 5.0, mid + 5.0);
    } else {
        (a_min, a_max) = pad_range(a_min, a_max, 0.05);
    }
    let t_span = (t_max - t_min).max(1e-6);
    let a_span = (a_max - a_min).max(1e-6);

    caption(
        ui,
        &theme,
        format!("{y_name} over time · {total_spikes} spikes · regression line per cluster"),
    );
    let (rect, _resp) = alloc_plot(ui, egui::vec2(160.0, 180.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme.plot_bg);
    let data = axes_data_rect(rect, true);
    draw_axes_grid(
        &painter,
        rect,
        (t_min, t_min + t_span),
        (a_min, a_max),
        "time (s)",
        y_unit,
        &theme,
    );
    let painter = painter.with_clip_rect(data);
    let mut max_drift = 0.0f32;

    const POINT_BUDGET: usize = 30_000;
    let stride = ((total_spikes as f32 / POINT_BUDGET as f32).ceil() as usize).max(1);

    // Linear regression line per cluster: shows the drift direction.
    for (slot, &c) in selection.iter().enumerate() {
        let times = session.spike_times(c);
        let amps = ys(c);
        if times.len() < 3 || amps.len() < 3 {
            continue;
        }
        let colour = selection_colour(slot);
        let n = times.len().min(amps.len());
        // OLS regression line on (t, a). cheap.
        let nf = n as f64;
        let mean_t: f64 = times[..n]
            .iter()
            .map(|&t| t.as_f64() * inv_sr as f64)
            .sum::<f64>()
            / nf;
        let mean_a: f64 = amps[..n].iter().map(|&a| a as f64).sum::<f64>() / nf;
        let mut num = 0.0_f64;
        let mut denom = 0.0_f64;
        for i in 0..n {
            let dt = times[i].as_f64() * inv_sr as f64 - mean_t;
            let da = amps[i] as f64 - mean_a;
            num += dt * da;
            denom += dt * dt;
        }
        if denom <= 0.0 {
            continue;
        }
        let slope = num / denom;
        let intercept = mean_a - slope * mean_t;
        let predict = |t: f64| (slope * t + intercept) as f32;
        let xy = |t: f32, a: f32| {
            egui::pos2(
                data.left() + (t - t_min) / t_span * data.width(),
                data.bottom() - (a - a_min) / a_span * data.height(),
            )
        };
        let p0 = xy(t_min, predict(t_min as f64));
        let p1 = xy(t_max, predict(t_max as f64));
        max_drift = max_drift.max((slope * (t_max - t_min) as f64).abs() as f32);
        painter.line_segment([p0, p1], Stroke::new(1.6_f32, colour));
    }

    // Dots go over the regression lines.
    for (slot, &c) in selection.iter().enumerate() {
        let times = session.spike_times(c);
        let amps = ys(c);
        if times.len() < 3 || amps.len() < 3 {
            continue;
        }
        let colour = selection_colour(slot);
        let faint = Color32::from_rgba_unmultiplied(colour.r(), colour.g(), colour.b(), 90);
        for i in (0..times.len().min(amps.len())).step_by(stride) {
            let t = times[i].as_f32() * inv_sr;
            let x = data.left() + (t - t_min) / t_span * data.width();
            let y = data.bottom() - (amps[i] - a_min) / a_span * data.height();
            painter.circle_filled(egui::pos2(x, y), 1.6, faint);
        }
    }
    let flat_below = if depths.is_some() { 0.5 } else { a_span * 0.02 };
    if max_drift < flat_below {
        painter.text(
            data.center_top() + egui::vec2(0.0, 8.0),
            egui::Align2::CENTER_TOP,
            "no measurable drift",
            egui::FontId::proportional(CAPTION_PX),
            theme.caption,
        );
    }
}

// ---- ContaminationOverTimeView (sliding refractory contamination) ------

/// Plot refractory contamination across `n_windows` time slices, one line
/// per selected cluster. Spikes in clusters not in the selection are
/// ignored. Useful to spot cases where contamination is concentrated in a
/// short stretch (drift artifact, noise burst) rather than uniform across
/// the recording.
pub fn contamination_over_time_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
) {
    let theme = Theme::from_ui(ui);
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }
    let sr = session.provider.sample_rate().max(1.0);
    let total_samples = session.provider.n_samples().0;
    if total_samples == 0 {
        ui.label("recording has zero length");
        return;
    }
    let total_s = total_samples as f32 / sr;
    let refractory_samples = (sr * 0.0015).round() as u64;
    const N_WINDOWS: usize = 60;
    const MIN_SPIKES_PER_WINDOW: usize = 20;

    // Compute per-cluster series.
    let series: Vec<(ClusterId, Vec<(f32, f32)>)> = selection
        .iter()
        .map(|&c| {
            let times = session.spike_times(c);
            let s = sliding_refractory_contamination(
                times,
                refractory_samples,
                total_samples,
                sr,
                N_WINDOWS,
                MIN_SPIKES_PER_WINDOW,
            );
            (c, s)
        })
        .collect();

    let mut y_max = 0.05_f32;
    for (_, s) in &series {
        for &(_, y) in s {
            if y.is_finite() && y > y_max {
                y_max = y;
            }
        }
    }
    y_max = y_max.clamp(0.05, 2.0);

    let finite: Vec<f32> = series
        .iter()
        .flat_map(|(_, s)| s.iter().map(|p| p.1))
        .filter(|y| y.is_finite())
        .collect();
    if finite.is_empty() {
        empty_plot(
            ui,
            &theme,
            &format!(
                "fewer than {MIN_SPIKES_PER_WINDOW} spikes per window — contamination not estimable"
            ),
        );
        return;
    }
    let all_zero = finite.iter().all(|&y| y <= 0.0);
    caption(
        ui,
        &theme,
        format!(
            "refractory contamination · {N_WINDOWS} windows × {:.1} s · {}",
            total_s / N_WINDOWS as f32,
            legend(selection),
        ),
    );
    let (rect, _resp) = alloc_plot(ui, egui::vec2(160.0, 180.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme.plot_bg);
    let data = axes_data_rect(rect, true);
    // A little room under zero so a flat series stays visible.
    let y_lo = -y_max * 0.06;
    draw_axes_grid(
        &painter,
        rect,
        (0.0, total_s),
        (y_lo, y_max),
        "time (s)",
        "contam",
        &theme,
    );
    let dp = painter.with_clip_rect(data);
    let y_of = |y: f32| data.bottom() - (y - y_lo) / (y_max - y_lo) * data.height();

    // Reference line at the conventional 0.1 contamination threshold.
    let ref_y = 0.1_f32;
    if ref_y < y_max {
        let y = y_of(ref_y);
        dp.extend(egui::Shape::dashed_line(
            &[egui::pos2(data.left(), y), egui::pos2(data.right(), y)],
            Stroke::new(1.0_f32, theme.danger),
            5.0,
            4.0,
        ));
        dp.text(
            egui::pos2(data.right() - 4.0, y - 2.0),
            egui::Align2::RIGHT_BOTTOM,
            "0.1",
            egui::FontId::proportional(TICK_PX),
            theme.danger,
        );
    }

    // Plot each cluster's polyline, with dots so short runs stay visible.
    for (slot, (_, s)) in series.iter().enumerate() {
        let colour = selection_colour(slot);
        let stroke = Stroke::new(1.5_f32, colour);
        let mut prev: Option<egui::Pos2> = None;
        for &(t, y) in s {
            if !y.is_finite() {
                prev = None;
                continue;
            }
            let x = data.left() + (t / total_s).clamp(0.0, 1.0) * data.width();
            let p = egui::pos2(x, y_of(y.min(y_max)));
            if let Some(pp) = prev {
                dp.line_segment([pp, p], stroke);
            }
            dp.circle_filled(p, 2.5, colour);
            prev = Some(p);
        }
    }
    if all_zero {
        dp.text(
            data.center_top() + egui::vec2(0.0, 12.0),
            egui::Align2::CENTER_TOP,
            "no refractory violations in any window",
            egui::FontId::proportional(CAPTION_PX),
            theme.caption,
        );
    }
}

// ---- PSTH (event-aligned histogram + raster) ----------------------------

/// Alignment window and binning for the PSTH pane.
#[derive(Clone, Debug, PartialEq)]
pub struct PsthState {
    pub pre_ms: f32,
    pub post_ms: f32,
    pub bins: usize,
    /// Event label to align to; `None` = every event.
    pub label: Option<u16>,
}

impl Default for PsthState {
    fn default() -> Self {
        Self {
            pre_ms: 200.0,
            post_ms: 500.0,
            bins: 70,
            label: None,
        }
    }
}

/// `(selection, version, pre, post, bins, label)` the cache was built for.
type PsthKey = (Vec<ClusterId>, u64, u64, u64, usize, Option<u16>);

/// Memo of the event-aligned counts per selected cluster plus the
/// per-event relative spike times of the first cluster (for the raster).
#[derive(Default)]
pub struct PsthCache {
    key: Option<PsthKey>,
    counts: Vec<Vec<u32>>,
    raster: Vec<Vec<u64>>,
    n_events: usize,
    /// "tone 12, light 12" — events per label, empty with one label.
    breakdown: String,
}

/// Peri-stimulus time histogram of every selected cluster around the
/// dataset's events, with a per-trial raster of the first selected one.
#[allow(clippy::too_many_arguments)]
pub fn psth_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    events: Option<&Events>,
    state: &mut PsthState,
    cache: &mut PsthCache,
) {
    let theme = Theme::from_ui(ui);
    let Some(events) = events else {
        ui.label("no events.csv / events.npy next to the dataset — PSTH unavailable");
        return;
    };
    if selection.is_empty() {
        ui.label("no selection");
        return;
    }
    let sr = session.provider.sample_rate().max(1.0);

    ui.horizontal_wrapped(|ui| {
        ui.label("event:");
        let current = state
            .label
            .and_then(|l| events.names.get(l as usize))
            .map(String::as_str)
            .unwrap_or("all");
        egui::ComboBox::from_id_salt("psth-label")
            .selected_text(current)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut state.label, None, "all");
                for (i, name) in events.names.iter().enumerate() {
                    ui.selectable_value(&mut state.label, Some(i as u16), name);
                }
            });
        ui.label("pre:");
        ui.add(
            egui::DragValue::new(&mut state.pre_ms)
                .range(0.0..=10_000.0)
                .suffix(" ms"),
        );
        ui.label("post:");
        ui.add(
            egui::DragValue::new(&mut state.post_ms)
                .range(1.0..=10_000.0)
                .suffix(" ms"),
        );
        ui.label("bins:");
        ui.add(egui::DragValue::new(&mut state.bins).range(2..=500));
    });

    let pre = (state.pre_ms / 1000.0 * sr).round() as u64;
    let post = (state.post_ms / 1000.0 * sr).round().max(1.0) as u64;
    let bins = state.bins.max(2);
    let key = (
        selection.to_vec(),
        session.version(),
        pre,
        post,
        bins,
        state.label,
    );
    if cache.key.as_ref() != Some(&key) {
        let ev = events.samples(sr, state.label);
        cache.n_events = ev.len();
        cache.breakdown = if events.names.len() > 1 {
            let mut counts = vec![0usize; events.names.len()];
            for &l in &events.label {
                if state.label.is_none_or(|want| want == l) {
                    if let Some(c) = counts.get_mut(l as usize) {
                        *c += 1;
                    }
                }
            }
            events
                .names
                .iter()
                .zip(counts)
                .filter(|(_, c)| *c > 0)
                .map(|(n, c)| format!("{n} {c}"))
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            String::new()
        };
        let mut results = selection
            .iter()
            .map(|&c| psth(session.spike_times(c), &ev, pre, post, bins));
        let first = results.next();
        let mut counts = Vec::with_capacity(selection.len());
        cache.raster = match first {
            Some((h, per_event)) => {
                counts.push(h);
                per_event
            }
            None => Vec::new(),
        };
        counts.extend(results.map(|(h, _)| h));
        cache.counts = counts;
        cache.key = Some(key);
    }
    if cache.n_events == 0 {
        ui.label("no events with that label");
        return;
    }

    // Histogram as firing rate (Hz) per cluster.
    let span_s = (pre + post) as f32 / sr;
    let bin_s = span_s / bins as f32;
    let rate = |count: u32| count as f32 / (cache.n_events as f32 * bin_s);
    let max_rate = cache
        .counts
        .iter()
        .flatten()
        .map(|&c| rate(c))
        .fold(0.0f32, f32::max)
        .max(1e-3);
    let breakdown = if cache.breakdown.is_empty() {
        String::new()
    } else {
        format!(" ({})", cache.breakdown)
    };
    caption(
        ui,
        &theme,
        format!(
            "{} events{breakdown} · {} ms bins · raster: c{}",
            cache.n_events,
            (bin_s * 1e4).round() / 10.0,
            selection[0]
        ),
    );

    let (rect, _resp) = alloc_plot(ui, egui::vec2(160.0, 220.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme.plot_bg);
    let x_range = (-state.pre_ms, state.post_ms);
    // Histogram on top (its x labels are the raster's), raster below.
    let hist_bottom = rect.top() + (rect.height() - X_STRIP) * 0.58;
    let hist_outer =
        egui::Rect::from_min_max(rect.min, egui::pos2(rect.right(), hist_bottom + X_STRIP));
    let hist = axes_data_rect(hist_outer, true);
    let y_top = max_rate * 1.15;
    axes(
        &painter,
        hist_outer,
        x_range,
        (0.0, y_top),
        "",
        "Hz",
        &theme,
        AxisOpts {
            grid: true,
            hide_x: true,
            ..AxisOpts::default()
        },
    );
    let raster_outer = egui::Rect::from_min_max(
        egui::pos2(rect.left(), hist_bottom + 10.0),
        rect.right_bottom(),
    );
    let raster = axes_data_rect(raster_outer, true);
    axes(
        &painter,
        raster_outer,
        x_range,
        (0.0, 1.0),
        "ms",
        "trial",
        &theme,
        AxisOpts {
            y_ticks: Some(&[]),
            ..AxisOpts::default()
        },
    );

    let hp = painter.with_clip_rect(hist);
    let fill = if cache.counts.len() == 1 { 0.7 } else { 0.25 };
    for (slot, h) in cache.counts.iter().enumerate() {
        let rates: Vec<f32> = h.iter().map(|&c| rate(c)).collect();
        step_hist(&hp, hist, &rates, y_top, selection_colour(slot), fill);
    }

    // Raster of the first cluster: one row per event (subsampled to
    // keep rows ≥ 1 px).
    let rp = painter.with_clip_rect(raster);
    let n_rows = cache
        .raster
        .len()
        .min(raster.height().max(1.0) as usize)
        .max(1);
    let stride = (cache.raster.len() / n_rows).max(1);
    let row_h = raster.height() / n_rows as f32;
    let colour = selection_colour(0);
    for (row, rel_times) in cache.raster.iter().step_by(stride).take(n_rows).enumerate() {
        let y0 = raster.top() + row as f32 * row_h;
        for &rel in rel_times {
            let x = raster.left() + rel as f32 / (pre + post) as f32 * raster.width();
            rp.line_segment(
                [Pos2::new(x, y0), Pos2::new(x, y0 + row_h.max(1.0))],
                Stroke::new(1.0_f32, colour),
            );
        }
    }
    // Event marker at t = 0 across both halves.
    let marker = Stroke::new(1.0_f32, theme.text.gamma_multiply(0.6));
    let frac = pre as f32 / (pre + post) as f32;
    for r in [hist, raster] {
        let x0 = r.left() + frac * r.width();
        painter.line_segment([Pos2::new(x0, r.top()), Pos2::new(x0, r.bottom())], marker);
    }
}

#[cfg(test)]
mod axis_tests {
    use super::{
        axes_data_rect, draw_axes, draw_axes_grid, nice_ticks, tick_label, Theme, X_STRIP, Y_GUTTER,
    };

    #[test]
    fn data_rect_reserves_the_gutters() {
        let r = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(300.0, 200.0));
        let d = axes_data_rect(r, true);
        assert_eq!(d.left(), r.left() + Y_GUTTER);
        assert_eq!(d.bottom(), r.bottom() - X_STRIP);
        assert_eq!((d.top(), d.right()), (r.top(), r.right()));
        assert_eq!(axes_data_rect(r, false).left(), r.left());
    }

    #[test]
    fn zero_tick_uses_its_neighbours_decimals() {
        assert_eq!(tick_label(0.0, 0.5), "0.0");
        assert_eq!(tick_label(-0.0, 0.05), "0.00");
        assert_eq!(tick_label(0.0, 20.0), "0");
        assert_eq!(tick_label(0.5, 0.5), "0.5");
    }

    #[test]
    fn axis_labels_stay_inside_the_outer_rect() {
        let ctx = egui::Context::default();
        let theme = Theme::from_visuals(&egui::Visuals::dark());
        let rect = egui::Rect::from_min_size(egui::pos2(10.0, 10.0), egui::vec2(220.0, 140.0));
        for (x, y, yu) in [
            ((-50.0, 50.0), (0.0, 4.6), "amplitude"),
            ((0.0, 4.0), (0.0, 0.0), ""),
            ((-200.0, 500.0), (-0.03, 0.3), "Hz"),
        ] {
            let out = ctx.run_ui(egui::RawInput::default(), |ui| {
                let p = ui.painter().clone();
                draw_axes(&p, rect, x, y, "ms", yu, &theme);
                draw_axes_grid(&p, rect, x, y, "time (s)", yu, &theme);
            });
            let mut n = 0;
            for s in out.shapes {
                if let egui::Shape::Text(t) = s.shape {
                    n += 1;
                    let r = t.galley.rect.translate(t.pos.to_vec2());
                    assert!(rect.expand(0.5).contains_rect(r), "{r:?} outside {rect:?}");
                }
            }
            assert!(n > 0, "no tick text drawn");
        }
    }

    #[test]
    fn ticks_are_nice_monotonic_and_inside_the_range() {
        for &(lo, hi) in &[(0.0f32, 1.0f32), (0.3, 97.0), (-2.5, 2.5), (1000.0, 1234.0)] {
            let t = nice_ticks(lo, hi, 5);
            assert!(!t.is_empty(), "{lo}..{hi}");
            assert!(t.windows(2).all(|w| w[1] > w[0]));
            assert!(t.iter().all(|&v| v >= lo - 1e-3 && v <= hi + 1e-3), "{t:?}");
            assert!(t.len() <= 12, "{t:?}");
        }
        assert!(nice_ticks(1.0, 1.0, 5).is_empty());
        assert!(nice_ticks(f32::NAN, 1.0, 5).is_empty());
    }
}

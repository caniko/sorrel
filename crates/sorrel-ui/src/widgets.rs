use crate::brush::{BrushSource, Brushes, HoverSpike, SpikeBrush, TimeBrush};
use crate::theme::Theme;
use crate::views::{axes_data_rect, draw_axes};
use crate::wgpu_trace::TraceCallback;
use egui::{Color32, Sense, Stroke, Ui};
use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider, SampleIndex};
use sorrel_render::{
    build_trace_vertices_cfg_shanked, build_trace_vertices_gpu_shanked, GpuTracePreproc,
    TracePreproc, TraceVertex,
};
use std::sync::Arc;

pub use crate::cluster_view::{cluster_table, ClusterColumn, ClusterTableState};

/// One selected cluster's spikes to mark on the trace.
/// phy TraceView display toggles: gain (`Ctrl+Alt+↑/↓`), origin (`Alt+O`),
/// channel labels (`Alt+L`) and whether every cluster's spikes are marked
/// (`Alt+S`) or only the selection's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceDisplay {
    pub gain: f32,
    pub flip: bool,
    pub labels: bool,
    pub all_spikes: bool,
}

impl Default for TraceDisplay {
    fn default() -> Self {
        Self {
            gain: 1.0,
            flip: false,
            labels: true,
            all_spikes: false,
        }
    }
}

pub struct TraceMarker<'a> {
    pub cluster: ClusterId,
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
    /// phy: Ctrl+click picks the spike nearest this sample.
    pub pick: Option<u64>,
}

impl Default for TraceNav {
    fn default() -> Self {
        Self {
            pan: 0,
            zoom: 1.0,
            anchor: 0.5,
            pick: None,
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
    /// The spike a plain click picked, with the session version it was
    /// picked under. Re-written as the hover every frame until the next
    /// click, since hover lives one frame.
    pick: Option<(u64, HoverSpike)>,
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
    display: TraceDisplay,
    shanks: &[u32],
    cache: &mut TraceCache,
    brushes: &mut Brushes,
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
    // The time strip sits below the plot and the channel labels in a 40 px
    // gutter left of it, so neither overprints the traces.
    let data = axes_data_rect(rect, false);
    let plot = egui::Rect::from_min_max(
        egui::pos2(data.left() + 40.0, data.top()),
        data.right_bottom(),
    );
    let row_h = plot.height() / n_channels as f32;
    let label_every = (13.0 / row_h).ceil().max(1.0) as u32;
    // Display row of a channel (rows count from the top).
    let row_of = |ch: u32| {
        if display.flip {
            n_channels - 1 - ch
        } else {
            ch
        }
    };
    for row in 0..n_channels {
        let ch = row_of(row);
        let y = plot.top() + row_h * row as f32;
        if row > 0 {
            painter.hline(plot.x_range(), y, Stroke::new(0.5_f32, theme.grid));
        }
        if display.labels && row % label_every == 0 {
            painter.text(
                egui::pos2(plot.left() - 6.0, y + row_h * 0.5),
                egui::Align2::RIGHT_CENTER,
                format!("{ch}"),
                egui::FontId::monospace(11.0),
                theme.text,
            );
        }
    }

    // Gain: the window's largest excursion spans one row, so spikes of a
    // few hundred counts stay visible (dtype full-scale flattened them).
    // ponytail: per-window auto-gain rescales while panning; add a gain
    // knob if users want it fixed.
    let row_clip_height = 2.0 / n_channels as f32;
    let amp_scale = row_clip_height * 0.5 / cache.peak.max(f32::EPSILON) * display.gain;

    let callback = TraceCallback {
        vertices: Arc::clone(&cache.verts),
        generation: cache.generation,
        n_channels,
        points_per_channel,
        t0,
        t_span,
        flip: display.flip,
        amp_scale,
        // Neutral so the slot-coloured markers (slot 0 is blue) stand out.
        color: if theme.dark {
            [0.745, 0.745, 0.745, 1.0]
        } else {
            [0.235, 0.235, 0.235, 1.0]
        },
    };
    ui.painter()
        .add(eframe::egui_wgpu::Callback::new_paint_callback(
            plot, callback,
        ));

    // Spike markers for the selection: a translucent band on the peak-channel
    // row with a solid tick through it. Brushed spikes get an ink tick on
    // top; every tick is a click target for the sticky pick below.
    let w_end = window_start + window_len as u64;
    let ink_stroke = Stroke::new(2.0_f32, ink(theme.dark));
    // `(spike, x, tick top, tick bottom)`.
    let mut ticks: Vec<(HoverSpike, f32, f32, f32)> = Vec::new();
    for m in markers {
        let lo = m.times.partition_point(|t| t.0 < window_start);
        let hi = m.times.partition_point(|t| t.0 < w_end);
        let stroke = Stroke::new(1.5_f32, m.colour);
        let brushed = brushes.spikes.as_ref().map_or(&[][..], |b| b.of(m.cluster));
        for (local, t) in m.times.iter().enumerate().take(hi).skip(lo) {
            let x = plot.left() + (t.0 - window_start) as f32 / window_len as f32 * plot.width();
            let (y0, y1) = match m.channel {
                Some(ch) if (ch as u32) < n_channels => {
                    let y = plot.top() + row_h * row_of(ch as u32) as f32;
                    (y, y + row_h)
                }
                _ => (plot.top(), plot.bottom()),
            };
            let band = egui::Rect::from_min_max(egui::pos2(x - 2.5, y0), egui::pos2(x + 2.5, y1));
            painter.rect_filled(band, 0.0, m.colour.gamma_multiply(0.3));
            painter.line_segment([egui::pos2(x, y0), egui::pos2(x, y1)], stroke);
            if brushed.binary_search(&(local as u32)).is_ok() {
                painter.line_segment([egui::pos2(x, y0), egui::pos2(x, y1)], ink_stroke);
            }
            let hover = HoverSpike {
                cluster: m.cluster,
                local: local as u32,
                sample: t.0,
            };
            ticks.push((hover, x, y0, y1));
        }
    }
    let sr = session.provider.sample_rate().max(1.0);
    time_overlays(
        ui,
        &painter,
        plot,
        (window_start as f32 / sr, w_end as f32 / sr),
        brushes,
        sr,
    );

    // Outer rect: the plot's left edge and the pane's bottom, so the
    // x ticks land in the time strip under the plot.
    let axes_rect =
        egui::Rect::from_min_max(egui::pos2(plot.left(), rect.top()), rect.right_bottom());
    draw_axes(
        &painter,
        axes_rect,
        (window_start as f32 / sr, w_end as f32 / sr),
        (0.0, 0.0),
        "time (s)",
        "",
        &theme,
    );

    // Mouse navigation: drag or wheel pans, Ctrl-wheel / pinch zooms
    // around the pointer.
    let mut nav = TraceNav::default();
    if resp.dragged() {
        nav.pan -= (resp.drag_delta().x / plot.width() * window_len as f32) as i64;
    }
    if resp.hovered() {
        let (scroll, zoom, pointer) =
            ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta(), i.pointer.hover_pos()));
        let d = if scroll.x.abs() > scroll.y.abs() {
            scroll.x
        } else {
            scroll.y
        };
        nav.pan -= (d / plot.width() * window_len as f32) as i64;
        nav.zoom = zoom;
        if let Some(p) = pointer {
            nav.anchor = ((p.x - plot.left()) / plot.width()).clamp(0.0, 1.0);
        }
    }
    let m = ui.input(|i| i.modifiers);
    if resp.clicked() && m.command {
        if let Some(p) = resp.interact_pointer_pos() {
            let frac = ((p.x - plot.left()) / plot.width()).clamp(0.0, 1.0);
            nav.pick = Some(window_start + (frac * window_len as f32) as u64);
        }
    } else if resp.clicked() && !m.any() {
        // A plain click on a tick picks that spike (within 4 px of the
        // tick), anything else drops the pick.
        let version = session.version();
        cache.pick = resp.interact_pointer_pos().and_then(|p| {
            let on_tick = ticks
                .iter()
                .map(|&(h, x, y0, y1)| (h, egui::pos2(x, p.y.clamp(y0, y1))));
            nearest_within(on_tick, p, 4.0).map(|h| (version, h))
        });
    }
    // Hover lives one frame: re-write the pick until the next click, unless
    // another view is hovering a spike.
    if cache.pick.is_some_and(|(v, h)| {
        v != session.version() || !markers.iter().any(|m| m.cluster == h.cluster)
    }) {
        cache.pick = None;
    }
    if let (Some((_, h)), None) = (cache.pick, brushes.hover_next) {
        brushes.hover_next = Some(h);
    }
    (nav.pan != 0 || nav.zoom != 1.0 || nav.pick.is_some()).then_some(nav)
}

// ---- cross-view brushing helpers (ink, hover, time band) ------------------

/// Brush ink: white in dark, near-black in light, drawn on top of the
/// ordinary points.
pub(crate) fn ink(dark: bool) -> Color32 {
    if dark {
        Color32::from_rgba_unmultiplied(255, 255, 255, 240)
    } else {
        Color32::from_rgba_unmultiplied(20, 20, 20, 240)
    }
}

pub(crate) fn ink_packed(dark: bool) -> u32 {
    let c = ink(dark);
    crate::wgpu_raster::pack_rgba(c.r(), c.g(), c.b(), c.a())
}

/// Accent of the hovered spike (ring and time line).
pub(crate) fn accent(ui: &Ui) -> Color32 {
    ui.visuals().hyperlink_color
}

/// The hovered spike's ring on a point view.
pub(crate) fn ring(painter: &egui::Painter, ui: &Ui, at: egui::Pos2) {
    painter.circle_stroke(at, 6.0, Stroke::new(1.5_f32, accent(ui)));
}

/// The item whose position is nearest `p` and within `r` px, if any.
/// ponytail: callers scan every point (O(n), n <= ~15k); a grid index if a
/// view ever hovers a million.
pub(crate) fn nearest_within<T>(
    items: impl Iterator<Item = (T, egui::Pos2)>,
    p: egui::Pos2,
    r: f32,
) -> Option<T> {
    items
        .map(|(t, q)| (t, q.distance_sq(p)))
        .filter(|(_, d)| *d <= r * r)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(t, _)| t)
}

/// Alpha of a spike at `t` seconds under the time brush: faded to 20 %
/// outside it, or `None` (hidden) when the brush is `only`.
pub(crate) fn time_alpha(tb: Option<TimeBrush>, t: f32, a: u8) -> Option<u8> {
    match tb {
        Some(b) if !b.contains(t) => (!b.only).then_some((a as f32 * 0.2) as u8),
        _ => Some(a),
    }
}

/// Cache identity of a time brush.
pub(crate) fn time_key(tb: Option<TimeBrush>) -> Option<(u32, u32, bool)> {
    tb.map(|b| (b.start.to_bits(), b.end.to_bits(), b.only))
}

fn x_to_t(data: egui::Rect, x: (f32, f32), px: f32) -> f32 {
    x.0 + (px - data.left()) / data.width().max(1.0) * (x.1 - x.0)
}

fn t_to_x(data: egui::Rect, x: (f32, f32), t: f32) -> f32 {
    data.left() + (t - x.0) / (x.1 - x.0).max(1e-9) * data.width()
}

/// The time brush of a Shift+drag from `press_x` to `cur_x` on a time axis
/// spanning `x` seconds over `data`.
pub(crate) fn drag_brush(
    data: egui::Rect,
    x: (f32, f32),
    press_x: f32,
    cur_x: f32,
) -> Option<TimeBrush> {
    let at = |px: f32| x_to_t(data, x, px.clamp(data.left(), data.right()));
    TimeBrush::new(at(press_x), at(cur_x))
}

/// Shift+drag along a time axis sets `brushes.time`; Shift+click clears it
/// (`clear_on_click`, off where Shift+click already means something).
pub(crate) fn time_brush_gesture(
    resp: &egui::Response,
    brushes: &mut Brushes,
    data: egui::Rect,
    x: (f32, f32),
    clear_on_click: bool,
) {
    let (m, press) = resp.ctx.input(|i| (i.modifiers, i.pointer.press_origin()));
    if !m.shift || m.command || m.alt {
        return;
    }
    if clear_on_click && resp.clicked() {
        brushes.time = None;
    } else if resp.dragged_by(egui::PointerButton::Primary) {
        if let (Some(p0), Some(p1)) = (press, resp.interact_pointer_pos()) {
            if let Some(b) = drag_brush(data, x, p0.x, p1.x) {
                let only = brushes.time.is_some_and(|t| t.only);
                brushes.time = Some(TimeBrush { only, ..b });
            }
        }
    }
}

/// The time band and the hovered spike's vertical line on a time axis
/// spanning `x` seconds over `data`.
pub(crate) fn time_overlays(
    ui: &Ui,
    painter: &egui::Painter,
    data: egui::Rect,
    x: (f32, f32),
    brushes: &Brushes,
    sr: f32,
) {
    let inside = |px: f32| (data.left()..=data.right()).contains(&px);
    if let Some(t) = brushes.time {
        let (a, b) = (t_to_x(data, x, t.start), t_to_x(data, x, t.end));
        let band = egui::Rect::from_x_y_ranges(
            a.clamp(data.left(), data.right())..=b.clamp(data.left(), data.right()),
            data.y_range(),
        );
        let sel = ui.visuals().selection;
        painter.rect_filled(band, 0.0, sel.bg_fill.gamma_multiply(0.22));
        for px in [a, b].into_iter().filter(|&px| inside(px)) {
            painter.vline(px, data.y_range(), Stroke::new(1.0_f32, sel.stroke.color));
        }
    }
    if let Some(h) = brushes.hover {
        let px = t_to_x(data, x, h.sample as f32 / sr.max(1.0));
        if inside(px) {
            let c = accent(ui).gamma_multiply(200.0 / 255.0);
            painter.vline(px, data.y_range(), Stroke::new(1.0_f32, c));
        }
    }
}

/// Upload counter for ink layers, so a recreated layer never matches a
/// stale GPU slot.
static INK_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A fresh upload generation for a scatter slot.
pub(crate) fn fresh_generation() -> u64 {
    INK_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// `(set generation, dark, spike brush key, time brush)`: what an ink layer
/// is built from.
pub(crate) type InkKey = (
    u64,
    bool,
    Option<(BrushSource, u64, usize, usize)>,
    Option<(u32, u32, bool)>,
);

pub(crate) fn ink_key(set_generation: u64, dark: bool, b: &Brushes) -> InkKey {
    (
        set_generation,
        dark,
        b.spikes.as_ref().map(SpikeBrush::key),
        time_key(b.time),
    )
}

/// The brushed spikes of a scatter view, drawn as a second, larger scatter
/// on top of it.
#[derive(Default)]
pub(crate) struct InkLayer {
    key: Option<InkKey>,
    generation: u64,
    verts: Arc<Vec<crate::wgpu_scatter::ScatterVertex>>,
}

impl InkLayer {
    /// `build` runs only when `key` changed.
    #[allow(clippy::too_many_arguments)]
    pub fn paint(
        &mut self,
        ui: &Ui,
        data: egui::Rect,
        id: &str,
        key: InkKey,
        x: (f32, f32),
        y: (f32, f32),
        build: impl FnOnce() -> Vec<crate::wgpu_scatter::ScatterVertex>,
    ) {
        if self.key != Some(key) {
            self.verts = Arc::new(build());
            self.generation = fresh_generation();
            self.key = Some(key);
        }
        if self.verts.is_empty() {
            return;
        }
        ui.painter()
            .add(eframe::egui_wgpu::Callback::new_paint_callback(
                data,
                crate::wgpu_scatter::ScatterCallback {
                    id: ui.id().with(id).value(),
                    vertices: Arc::clone(&self.verts),
                    generation: self.generation,
                    x,
                    y,
                    radius: 2.2,
                    size: data.size(),
                },
            ));
    }
}

/// Identity of a lasso / threshold selection over one point set: changes
/// when the set, the polygon or the threshold does.
pub(crate) fn selection_key(
    set_generation: u64,
    lasso: &crate::lasso::Lasso,
    threshold: Option<f32>,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    set_generation.hash(&mut h);
    (lasso.is_closed(), threshold.map(f32::to_bits)).hash(&mut h);
    for p in &lasso.points {
        (p[0].to_bits(), p[1].to_bits()).hash(&mut h);
    }
    h.finish()
}

/// Write a view's lasso / threshold hits as the spike brush, only when
/// `key` changed since the last call. `key == None` (nothing armed) clears
/// the brush when `mine` wrote it; another view's brush stays.
pub(crate) fn sync_spike_brush(
    brushes: &mut Brushes,
    last: &mut Option<u64>,
    key: Option<u64>,
    mine: &[BrushSource],
    make: impl FnOnce() -> Option<SpikeBrush>,
) {
    if *last == key {
        return;
    }
    *last = key;
    match key {
        Some(_) => brushes.spikes = make(),
        None if brushes
            .spikes
            .as_ref()
            .is_some_and(|b| mine.contains(&b.source)) =>
        {
            brushes.spikes = None;
        }
        None => {}
    }
}

#[cfg(test)]
mod brush_tests {
    use super::*;
    use egui::{pos2, Rect};
    use sorrel_io::ClusterId;

    #[test]
    fn spike_brush_is_written_on_change_and_cleared_only_by_its_source() {
        let mut b = Brushes::default();
        let mut last = None;
        let mine = [BrushSource::AmplitudeLasso, BrushSource::AmplitudeThreshold];
        let make = || SpikeBrush::new(BrushSource::AmplitudeLasso, 1, [(ClusterId(2), vec![3, 1])]);
        let mut calls = 0;
        for _ in 0..3 {
            sync_spike_brush(&mut b, &mut last, Some(7), &mine, || {
                calls += 1;
                make()
            });
        }
        assert_eq!(calls, 1, "same key: no rewrite");
        assert_eq!(b.spikes.as_ref().unwrap().of(ClusterId(2)), &[1, 3]);
        // Another view takes over; our unchanged key must not take it back.
        b.spikes = SpikeBrush::new(BrushSource::FeatureLasso, 1, [(ClusterId(2), vec![9])]);
        sync_spike_brush(&mut b, &mut last, Some(7), &mine, make);
        assert_eq!(b.spikes.as_ref().unwrap().source, BrushSource::FeatureLasso);
        sync_spike_brush(&mut b, &mut last, None, &mine, make);
        assert!(b.spikes.is_some(), "not our brush: kept");
        sync_spike_brush(&mut b, &mut last, Some(8), &mine, make);
        sync_spike_brush(&mut b, &mut last, None, &mine, make);
        assert!(b.spikes.is_none(), "our brush: cleared with our lasso");
    }

    #[test]
    fn shift_drag_maps_to_data_seconds_and_clamps_to_the_plot() {
        let data = Rect::from_min_max(pos2(100.0, 0.0), pos2(300.0, 50.0));
        let x = (0.0, 20.0);
        let b = drag_brush(data, x, 200.0, 150.0).unwrap();
        assert_eq!((b.start, b.end, b.only), (5.0, 10.0, false));
        let b = drag_brush(data, x, 250.0, 900.0).unwrap();
        assert_eq!((b.start, b.end), (15.0, 20.0), "pointer clamps to the plot");
        assert!(
            drag_brush(data, x, 200.0, 200.0).is_none(),
            "a click is no range"
        );
    }

    #[test]
    fn nearest_spike_is_the_closest_within_the_radius() {
        let pts = [
            (0, pos2(10.0, 10.0)),
            (1, pos2(14.0, 10.0)),
            (2, pos2(40.0, 40.0)),
        ];
        let at = |p| nearest_within(pts.iter().copied(), p, 6.0);
        assert_eq!(at(pos2(13.0, 10.0)), Some(1));
        assert_eq!(at(pos2(11.0, 11.0)), Some(0));
        assert_eq!(at(pos2(25.0, 25.0)), None, "nothing within 6 px");
    }

    #[test]
    fn time_brush_fades_or_hides_outside() {
        let b = TimeBrush::new(1.0, 2.0).unwrap();
        assert_eq!(time_alpha(Some(b), 1.5, 200), Some(200));
        assert_eq!(time_alpha(Some(b), 3.0, 200), Some(40));
        assert_eq!(
            time_alpha(Some(TimeBrush { only: true, ..b }), 3.0, 200),
            None
        );
        assert_eq!(time_alpha(None, 3.0, 200), Some(200));
    }
}

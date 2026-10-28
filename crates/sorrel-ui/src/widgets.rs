use crate::theme::Theme;
use crate::views::{axes_data_rect, draw_axes};
use crate::wgpu_trace::TraceCallback;
use egui::{Color32, Sense, Stroke, Ui};
use sorrel_data::Session;
use sorrel_io::{DataProvider, SampleIndex};
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
    // row with a solid tick through it.
    let w_end = window_start + window_len as u64;
    for m in markers {
        let lo = m.times.partition_point(|t| t.0 < window_start);
        let hi = m.times.partition_point(|t| t.0 < w_end);
        let stroke = Stroke::new(1.5_f32, m.colour);
        for t in &m.times[lo..hi] {
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
        }
    }

    // Outer rect: the plot's left edge and the pane's bottom, so the
    // x ticks land in the time strip under the plot.
    let sr = session.provider.sample_rate().max(1.0);
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
    if resp.clicked() && ui.input(|i| i.modifiers.command) {
        if let Some(p) = resp.interact_pointer_pos() {
            let frac = ((p.x - plot.left()) / plot.width()).clamp(0.0, 1.0);
            nav.pick = Some(window_start + (frac * window_len as f32) as u64);
        }
    }
    (nav.pan != 0 || nav.zoom != 1.0 || nav.pick.is_some()).then_some(nav)
}

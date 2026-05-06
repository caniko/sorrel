use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui, Vec2};
use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider};
use sorrel_render::{build_trace_vertices, TraceVertex};

/// Cluster table. Generic over `P` so the column extractor inlines the
/// monomorphised `provider.spike_times` for the active backend.
pub fn cluster_table<P, F>(
    ui: &mut Ui,
    session: &Session<P>,
    selected: &mut ClusterId,
    label_str: F,
) where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    use egui_extras::{Column, TableBuilder};
    TableBuilder::new(ui)
        .striped(true)
        .resizable(true)
        .column(Column::auto().at_least(48.0))
        .column(Column::auto().at_least(64.0))
        .column(Column::auto().at_least(72.0))
        .header(20.0, |mut h| {
            h.col(|ui| {
                ui.strong("ID");
            });
            h.col(|ui| {
                ui.strong("# spikes");
            });
            h.col(|ui| {
                ui.strong("label");
            });
        })
        .body(|body| {
            let n = session.provider.n_clusters();
            body.rows(18.0, n as usize, |mut row| {
                let i = row.index() as ClusterId;
                let is_sel = i == *selected;
                row.col(|ui| {
                    if ui.selectable_label(is_sel, format!("{i}")).clicked() {
                        *selected = i;
                    }
                });
                row.col(|ui| {
                    ui.label(format!("{}", session.provider.spike_times(i).len()));
                });
                row.col(|ui| {
                    let lbl = session.label(i).unwrap_or_default();
                    ui.label(label_str(&lbl));
                });
            });
        });
}

/// Trace view. Pulls a window from the monomorphised provider, runs LTTB,
/// and paints with egui's 2D painter.
pub fn trace_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    window_start: u64,
    window_len: u32,
    target_points: usize,
) {
    let avail = ui.available_size();
    let (rect, _resp) = ui.allocate_exact_size(avail, Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, Color32::from_gray(15));

    let nc = session.provider.n_channels().max(1) as f32;
    let row_h = rect.height() / nc;
    let verts: Vec<TraceVertex> =
        build_trace_vertices(session, window_start, window_len, target_points);
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

    // Scale: x covers [t0, t0 + dt], y is centred per-channel row.
    let t0 = verts.first().map(|v| v.pos[0]).unwrap_or(0.0);
    let t1 = verts.last().map(|v| v.pos[0]).unwrap_or(t0 + 1.0);
    let dt = (t1 - t0).max(1e-9);
    let amp_scale = row_h * 0.4 / (i16::MAX as f32);

    let stroke = Stroke::new(1.0_f32, Color32::from_rgb(180, 220, 255));
    let mut prev: Option<(u32, Pos2)> = None;
    for v in &verts {
        let x = rect.left() + (v.pos[0] - t0) / dt * rect.width();
        let cy = rect.top() + row_h * (v.channel as f32 + 0.5);
        let y = cy - v.pos[1] * amp_scale;
        let p = Pos2::new(x, y);
        if let Some((pc, pp)) = prev {
            if pc == v.channel {
                painter.line_segment([pp, p], stroke);
            }
        }
        prev = Some((v.channel, p));
    }

    // Channel separator lines.
    for ch in 1..session.provider.n_channels() {
        let y = rect.top() + row_h * ch as f32;
        painter.hline(rect.x_range(), y, Stroke::new(0.5_f32, Color32::from_gray(40)));
    }
    let _ = Vec2::ZERO;
    let _ = Rect::NOTHING;
}

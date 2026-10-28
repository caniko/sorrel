//! phy's TemplateView: every cluster's template on its peak channel, one
//! small box per cluster in cluster-table order, the selection outlined.
//! Click selects; Ctrl or Shift+click adds or removes.

use egui::{Pos2, Sense, Stroke, Ui};
use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider};

use crate::theme::{selection_colour, Theme};

/// Per-cluster peak-channel template, rebuilt when curation changes.
#[derive(Default)]
pub struct TemplateGridCache {
    version: Option<u64>,
    /// Indexed by cluster id; empty when the cluster has no template.
    traces: Vec<Vec<f32>>,
    /// Max |value| over every trace, for one shared y scale like phy.
    scale: f32,
}

impl TemplateGridCache {
    fn ensure<P: DataProvider>(&mut self, s: &Session<P>) {
        if self.version == Some(s.version()) {
            return;
        }
        let (_, ns, nc) = s.template_shape();
        self.traces = (0..s.n_clusters())
            .map(ClusterId)
            .map(|c| {
                let Some(w) = s.primary_template(c).and_then(|t| s.template_waveform(t)) else {
                    return Vec::new();
                };
                let col = |ch: usize| (0..ns).filter_map(move |i| w.get(i * nc + ch).copied());
                let ptp = |ch: usize| {
                    let (lo, hi) =
                        col(ch).fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
                    hi - lo
                };
                (0..nc)
                    .max_by(|&a, &b| ptp(a).total_cmp(&ptp(b)))
                    .map(|peak| col(peak).collect())
                    .unwrap_or_default()
            })
            .collect();
        self.scale = self
            .traces
            .iter()
            .flatten()
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(f32::MIN_POSITIVE);
        self.version = Some(s.version());
    }
}

/// Returns the new selection after a click.
pub fn template_grid_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selection: &[ClusterId],
    order: &[ClusterId],
    cache: &mut TemplateGridCache,
    colours: &[Option<egui::Color32>],
    scheme_delta: &mut i32,
) -> Option<Vec<ClusterId>> {
    let theme = Theme::from_ui(ui);
    if !session.has_template_waveforms() {
        ui.label("templates.npy not loaded");
        return None;
    }
    cache.ensure(session);
    let mut picked = None;
    egui::ScrollArea::vertical()
        .id_salt("template-grid")
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(3.0, 3.0);
                for &c in order {
                    let Some(trace) = cache.traces.get(c.idx()).filter(|t| t.len() > 1) else {
                        continue;
                    };
                    let (rect, resp) =
                        ui.allocate_exact_size(egui::vec2(64.0, 46.0), Sense::click());
                    resp.widget_info(|| {
                        egui::WidgetInfo::labeled(
                            egui::WidgetType::Button,
                            true,
                            format!("template c{c}"),
                        )
                    });
                    let p = ui.painter_at(rect);
                    p.rect_filled(rect, 2.0, theme.plot_bg);
                    let slot = selection.iter().position(|&s| s == c);
                    let colour = slot.map_or(
                        colours
                            .get(c.idx())
                            .copied()
                            .flatten()
                            .unwrap_or(theme.caption),
                        selection_colour,
                    );
                    *scheme_delta += crate::theme::scheme_wheel(ui, &resp);
                    let n = trace.len() as f32 - 1.0;
                    let pts: Vec<Pos2> = trace
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            Pos2::new(
                                rect.left() + 2.0 + i as f32 / n * (rect.width() - 4.0),
                                rect.center().y + 4.0 - v / cache.scale * (rect.height() * 0.4),
                            )
                        })
                        .collect();
                    p.add(egui::Shape::line(pts, Stroke::new(1.2_f32, colour)));
                    p.text(
                        rect.left_top() + egui::vec2(3.0, 1.0),
                        egui::Align2::LEFT_TOP,
                        format!("{c}"),
                        egui::FontId::proportional(11.0),
                        theme.caption,
                    );
                    if slot.is_some() {
                        p.rect_stroke(
                            rect,
                            2.0,
                            Stroke::new(1.5_f32, colour),
                            egui::StrokeKind::Inside,
                        );
                    }
                    if resp.clicked() {
                        let add = ui.input(|i| i.modifiers.command || i.modifiers.shift);
                        let mut next = if add { selection.to_vec() } else { Vec::new() };
                        match next.iter().position(|&s| s == c) {
                            Some(i) => {
                                next.remove(i);
                            }
                            None => next.push(c),
                        }
                        picked = (!next.is_empty()).then_some(next);
                    }
                }
            });
        });
    picked
}

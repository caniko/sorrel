//! phy's PanZoom for the 2-D plots: drag pans, the wheel zooms around the
//! pointer, double-click resets. Applied to a plot's data ranges, so the
//! views keep drawing in data coordinates.

use egui::Response;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanZoom {
    /// Visible fraction's centre (0.5 = middle) and zoom factor per axis.
    centre: [f32; 2],
    zoom: [f32; 2],
}

impl Default for PanZoom {
    fn default() -> Self {
        Self {
            centre: [0.5; 2],
            zoom: [1.0; 2],
        }
    }
}

impl PanZoom {
    /// The visible part of the full ranges `x`, `y`.
    pub fn apply(&self, x: (f32, f32), y: (f32, f32)) -> ((f32, f32), (f32, f32)) {
        let f = |(lo, hi): (f32, f32), a: usize| {
            let span = hi - lo;
            let half = 0.5 / self.zoom[a];
            (
                lo + (self.centre[a] - half) * span,
                lo + (self.centre[a] + half) * span,
            )
        };
        (f(x, 0), f(y, 1))
    }

    /// Plain drag pans, the plain wheel zooms around the pointer and
    /// double-click resets. Modified gestures (lasso, threshold, colour
    /// scheme) are left to the view.
    pub fn interact(&mut self, resp: &Response, data: egui::Rect) {
        let m = resp.ctx.input(|i| i.modifiers);
        if m.any() {
            return;
        }
        if resp.double_clicked() {
            *self = Self::default();
            return;
        }
        if resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            self.centre[0] -= d.x / data.width() / self.zoom[0];
            self.centre[1] += d.y / data.height() / self.zoom[1];
        }
        if !resp.hovered() {
            return;
        }
        let scroll = resp.ctx.input(|i| i.smooth_scroll_delta.y);
        if scroll == 0.0 {
            return;
        }
        let f = (scroll * 0.002).exp();
        // Keep the point under the pointer fixed.
        if let Some(p) = resp.hover_pos() {
            let u = [
                (p.x - data.left()) / data.width(),
                (data.bottom() - p.y) / data.height(),
            ];
            for a in 0..2 {
                let new_zoom = (self.zoom[a] * f).clamp(1.0, 200.0);
                let at = self.centre[a] + (u[a] - 0.5) / self.zoom[a];
                self.centre[a] = at - (u[a] - 0.5) / new_zoom;
                self.zoom[a] = new_zoom;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_shows_everything_and_zoom_narrows_around_the_centre() {
        let pz = PanZoom::default();
        assert_eq!(
            pz.apply((0.0, 10.0), (-1.0, 1.0)),
            ((0.0, 10.0), (-1.0, 1.0))
        );
        let pz = PanZoom {
            centre: [0.25, 0.5],
            zoom: [2.0, 1.0],
        };
        assert_eq!(pz.apply((0.0, 10.0), (-1.0, 1.0)).0, (0.0, 5.0));
    }
}

//! phy's polygon lasso: Ctrl+click adds a vertex, Ctrl+right-click (or
//! Esc) clears it, and the split action (`k`) takes every spike inside.
//! Vertices are stored in data coordinates so pan/zoom keeps them on the
//! points they were drawn around.

use egui::{Color32, Pos2, Stroke};

/// An in-progress polygon, locked to the plot (`cell`) of its first vertex.
#[derive(Clone, Debug, Default)]
pub struct Lasso {
    pub cell: usize,
    pub points: Vec<[f32; 2]>,
}

impl Lasso {
    /// A closed polygon needs three vertices.
    pub fn is_closed(&self) -> bool {
        self.points.len() >= 3
    }

    pub fn is_active(&self) -> bool {
        !self.points.is_empty()
    }

    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Handle Ctrl+click (add a vertex) and Ctrl+right-click (clear) on the
    /// plot `cell`. Returns true when the lasso changed.
    pub fn interact(
        &mut self,
        cell: usize,
        resp: &egui::Response,
        to_data: impl Fn(Pos2) -> [f32; 2],
    ) -> bool {
        let ctrl = resp.ctx.input(|i| i.modifiers.command);
        if !ctrl {
            return false;
        }
        if resp.secondary_clicked() {
            self.clear();
            return true;
        }
        if resp.clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                if self.points.is_empty() {
                    self.cell = cell;
                }
                if self.cell == cell {
                    self.points.push(to_data(p));
                    return true;
                }
            }
        }
        false
    }

    pub fn contains(&self, p: [f32; 2]) -> bool {
        self.is_closed() && point_in_polygon(p, &self.points)
    }

    /// Outline (closed once it has three vertices) plus vertex dots.
    pub fn paint(
        &self,
        painter: &egui::Painter,
        to_screen: impl Fn([f32; 2]) -> Pos2,
        colour: Color32,
    ) {
        let pts: Vec<Pos2> = self.points.iter().map(|&p| to_screen(p)).collect();
        let stroke = Stroke::new(1.5_f32, colour);
        for w in pts.windows(2) {
            painter.line_segment([w[0], w[1]], stroke);
        }
        if let (true, Some(&first), Some(&last)) = (self.is_closed(), pts.first(), pts.last()) {
            painter.line_segment([last, first], stroke);
        }
        for &p in &pts {
            painter.circle_filled(p, 2.5, colour);
        }
    }
}

/// Even-odd ray cast; points on an edge may land either side.
pub fn point_in_polygon(p: [f32; 2], poly: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let mut j = poly.len().wrapping_sub(1);
    for (i, &a) in poly.iter().enumerate() {
        let b = poly[j];
        if (a[1] > p[1]) != (b[1] > p[1])
            && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concave_polygon_in_and_out() {
        // A "C" shape opening to the right: the notch is outside.
        let c = [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [3.0, 2.0],
            [3.0, 3.0],
            [0.0, 3.0],
        ];
        assert!(point_in_polygon([0.5, 1.5], &c));
        assert!(point_in_polygon([2.0, 0.5], &c));
        assert!(point_in_polygon([2.0, 2.5], &c));
        assert!(!point_in_polygon([2.0, 1.5], &c), "the notch");
        assert!(!point_in_polygon([4.0, 0.5], &c));
        assert!(!point_in_polygon([-1.0, 1.5], &c));
        let open = Lasso {
            cell: 0,
            points: c[..2].to_vec(),
        };
        assert!(!open.contains([0.5, 0.0]), "two vertices enclose nothing");
    }
}

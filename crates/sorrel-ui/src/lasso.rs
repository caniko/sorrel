//! phy's polygon lasso, made editable. Gestures on a plot cell:
//!
//! - Ctrl+drag: freehand stroke; on release it is simplified (Ramer-Douglas-
//!   Peucker, at most [`MAX_NODES`] nodes) and replaces the lasso. Under
//!   three nodes is discarded.
//! - Ctrl+click: add a node (phy). Ctrl+right-click (or Esc): clear.
//! - On a closed lasso, without modifiers: drag a node to move it,
//!   double-click on an edge to insert a node, right-click a node to delete
//!   it (never below three). Hovering shows the node, or a ghost "+" where a
//!   double-click would insert.
//! - `k` splits every spike inside.
//!
//! Nodes are stored in data coordinates so pan/zoom keeps them on the
//! points they were drawn around. [`Lasso::interact`] reports whether it
//! consumed the input, so the caller skips pan/zoom/reset for it.

use egui::{Color32, Pos2, Stroke};

/// Screen-space pick radii (px) and the freehand node budget.
const NODE_HIT: f32 = 6.0;
const EDGE_HIT: f32 = 4.0;
const SIMPLIFY_EPS: f32 = 2.0;
const MAX_NODES: usize = 64;

/// A polygon, locked to the plot (`cell`) it was drawn in.
#[derive(Clone, Debug, Default)]
pub struct Lasso {
    pub cell: usize,
    pub points: Vec<[f32; 2]>,
    /// Node being dragged, and the live Ctrl+drag stroke (screen space).
    drag: Option<usize>,
    stroke: Vec<Pos2>,
    stroke_previous_cell: usize,
    /// A Ctrl-click polygon stays a draft between its mouse clicks.
    polygon_draft: bool,
    /// Latch the gesture's modifier at press, not at a later release/K frame.
    pressed_command: Option<(usize, bool)>,
    /// Escape may run before a captured click's release reaches the pane.
    /// Ignore the old gesture until a genuinely new press begins.
    discard_until_press: bool,
    /// Hover feedback from the last `interact`: node under the pointer, or
    /// the projection of the pointer on a nearby edge (screen space).
    hot: Option<usize>,
    ghost: Option<Pos2>,
}

/// What `interact` did: `changed` the polygon, or `consumed` the input (the
/// caller must then skip its own pan/zoom/double-click handling). `drawn`
/// is set when a Ctrl gesture (stroke release, node click) shaped the
/// lasso, as opposed to an unmodified edit of a closed one (node move,
/// insert, delete); selecting what is inside belongs to `drawn` only.
#[derive(Clone, Copy, Debug, Default)]
pub struct LassoEvent {
    pub changed: bool,
    pub consumed: bool,
    pub drawn: bool,
}

impl Lasso {
    pub fn new(cell: usize, points: Vec<[f32; 2]>) -> Self {
        Self {
            cell,
            points,
            ..Self::default()
        }
    }

    /// A closed polygon needs three vertices.
    pub fn is_closed(&self) -> bool {
        self.points.len() >= 3
    }

    pub fn is_active(&self) -> bool {
        !self.points.is_empty() || self.is_editing()
    }

    /// A freehand stroke or node drag is in progress: the polygon is still
    /// moving, so callers may hold their computed outputs until it settles.
    pub fn is_editing(&self) -> bool {
        self.drag.is_some() || !self.stroke.is_empty()
    }

    pub fn is_drafting(&self) -> bool {
        self.polygon_draft || self.is_editing()
    }

    /// Ctrl release completes a clicked polygon, but never finishes a mouse
    /// gesture before its captured release has been processed.
    pub fn finish_polygon_on_release(&mut self, command: bool, pointer_down: bool) -> bool {
        if self.polygon_draft && !command && !pointer_down && !self.is_editing() {
            self.polygon_draft = false;
            return true;
        }
        false
    }

    /// K is an explicit finish as well as a split; it does not depend on a
    /// preview having computed the polygon's members already.
    pub fn finish_polygon(&mut self) {
        if !self.is_editing() {
            self.polygon_draft = false;
        }
    }

    pub fn clear(&mut self) {
        *self = Self::default();
        self.discard_until_press = true;
    }

    #[cfg(test)]
    pub(crate) fn set_test_editing(&mut self, editing: bool) {
        self.drag = editing.then_some(0);
    }

    #[cfg(test)]
    pub(crate) fn set_test_drafting(&mut self, drafting: bool) {
        self.polygon_draft = drafting;
    }

    /// Delete node `i`; a polygon never drops below three nodes.
    fn remove_node(&mut self, i: usize) -> bool {
        let ok = self.points.len() > 3;
        if ok {
            self.points.remove(i);
        }
        ok
    }

    /// Handle the gestures above on the plot `cell`.
    pub fn interact(
        &mut self,
        cell: usize,
        resp: &egui::Response,
        to_data: impl Fn(Pos2) -> [f32; 2],
        to_screen: impl Fn([f32; 2]) -> Pos2,
    ) -> LassoEvent {
        use egui::PointerButton::Primary;
        let mut ev = LassoEvent::default();
        let (m, press) = resp.ctx.input(|i| (i.modifiers, i.pointer.press_origin()));
        let at = resp.interact_pointer_pos().map(|p| resp.rect.clamp(p));
        let screen = |s: &Self| s.points.iter().map(|&p| to_screen(p)).collect::<Vec<_>>();
        let mine = self.is_closed() && self.cell == cell;
        let (pressed, released) = resp.ctx.input(|i| {
            (
                i.pointer.button_pressed(Primary),
                i.pointer.button_released(Primary),
            )
        });
        if self.discard_until_press {
            if pressed && (resp.is_pointer_button_down_on() || resp.clicked()) {
                self.discard_until_press = false;
            } else {
                ev.consumed = released || resp.ctx.input(|i| i.pointer.primary_down());
                return ev;
            }
        }
        if pressed && (resp.is_pointer_button_down_on() || resp.clicked()) {
            let command = resp.ctx.input(|i| {
                i.events
                    .iter()
                    .find_map(|event| match event {
                        egui::Event::PointerButton {
                            pressed: true,
                            modifiers,
                            pos,
                            ..
                        } if resp.rect.contains(*pos) => Some(modifiers.command),
                        _ => None,
                    })
                    .unwrap_or(m.command)
            });
            self.pressed_command = Some((cell, command));
        }

        // Ctrl+drag: record, draw live, close on release.
        if m.command && resp.drag_started_by(Primary) {
            // egui only blurs text editors on clicks by default, not drags.
            // Drawing a lasso must leave the cluster filter's text focus so
            // the next K is a curation command rather than filter input.
            resp.request_focus();
            self.stroke_previous_cell = self.cell;
            self.cell = cell;
            self.stroke = press.into_iter().collect();
        }
        // A shared lasso is visited by every projection each frame. Only
        // the projection where the gesture began may extend or finish it.
        if self.is_editing() && self.cell != cell {
            return ev;
        }
        // A modifier change must not leave a captured node drag stuck in
        // "editing", which would reject every subsequent K press.
        if released && self.drag.is_some() {
            if let (Some(i), Some(p)) = (self.drag.take(), at) {
                let point = to_data(p);
                ev.changed = self.points[i] != point;
                self.points[i] = point;
            }
            self.pressed_command = None;
            ev.consumed = true;
            resp.ctx.request_repaint();
            return ev;
        }
        if !self.stroke.is_empty() && resp.dragged_by(Primary) {
            self.stroke.extend(at);
            let p = resp
                .ctx
                .layer_painter(resp.layer_id)
                .with_clip_rect(resp.rect);
            let col = resp.ctx.global_style().visuals.strong_text_color();
            p.extend(egui::Shape::dashed_line(
                &self.stroke,
                Stroke::new(1.5_f32, col),
                5.0,
                4.0,
            ));
        }
        if !self.stroke.is_empty() {
            ev.consumed = true;
            if resp.drag_stopped() || released {
                self.stroke.extend(at);
                let stationary = self.stroke.first().is_some_and(|first| {
                    self.stroke
                        .iter()
                        .all(|point| point.distance(*first) <= NODE_HIT)
                });
                let nodes = simplify_stroke(&self.stroke);
                self.stroke.clear();
                self.pressed_command = None;
                if stationary {
                    // Long stationary clicks can be classified as drags by
                    // egui. They are still polygon vertices, not empty loops.
                    if self.points.is_empty() || self.stroke_previous_cell == cell {
                        if let Some(p) = at {
                            self.points.push(to_data(p));
                            self.polygon_draft = m.command;
                            ev.changed = true;
                            ev.drawn = true;
                        }
                    } else {
                        self.cell = self.stroke_previous_cell;
                    }
                } else if nodes.len() >= 3 {
                    *self = Self {
                        cell,
                        points: nodes.iter().map(|&p| to_data(p)).collect(),
                        ..Self::default()
                    };
                    ev.changed = true;
                    ev.drawn = true;
                }
                // The closed polygon (and the hits computed from it) is
                // painted next frame, after the stroke's last frame was drawn.
                resp.ctx.request_repaint();
            }
            return ev;
        }

        // A Ctrl+click release, Ctrl release, and plain K can share one
        // native frame. The frame's final modifiers then omit Ctrl: use
        // the pointer event's modifiers or the last polygon node is lost.
        let click_command = self
            .pressed_command
            .filter(|(owner, _)| *owner == cell)
            .map(|(_, command)| command)
            .unwrap_or_else(|| {
                resp.ctx.input(|i| {
                    i.events
                        .iter()
                        .rev()
                        .find_map(|event| match event {
                            egui::Event::PointerButton {
                                pos,
                                pressed: false,
                                modifiers,
                                ..
                            } if resp.rect.contains(*pos) => Some(modifiers.command),
                            _ => None,
                        })
                        .unwrap_or(m.command)
                })
            });
        if released && self.pressed_command.is_some_and(|(owner, _)| owner == cell) {
            self.pressed_command = None;
        }
        if click_command {
            if resp.secondary_clicked() {
                self.clear();
                return LassoEvent {
                    changed: true,
                    consumed: true,
                    ..LassoEvent::default()
                };
            }
            if let (true, Some(p)) = (resp.clicked(), at) {
                if self.points.is_empty() {
                    self.cell = cell;
                }
                if self.cell == cell {
                    resp.request_focus();
                    self.points.push(to_data(p));
                    self.polygon_draft = m.command;
                    return LassoEvent {
                        changed: true,
                        consumed: true,
                        drawn: true,
                    };
                }
            }
        } else if mine && !m.any() {
            // Edit a closed lasso: move / insert / delete a node.
            let pts = screen(self);
            if resp.drag_started_by(Primary) {
                self.drag = press.and_then(|p| nearest_node(&pts, p, NODE_HIT));
                if self.drag.is_some() {
                    resp.request_focus();
                }
            }
            if let Some(i) = self.drag {
                ev.consumed = true;
                if let (true, Some(p)) = (resp.dragged_by(Primary), at) {
                    self.points[i] = to_data(p);
                    ev.changed = true;
                }
                if resp.drag_stopped() {
                    self.drag = None;
                    // With live tracking off the hits settle after this
                    // frame; ask for the one that draws them.
                    resp.ctx.request_repaint();
                }
            } else if let Some(p) = at {
                let on_node = nearest_node(&pts, p, NODE_HIT);
                if resp.double_clicked() {
                    if let (None, Some((i, q))) = (on_node, nearest_edge(&pts, p, EDGE_HIT)) {
                        self.points.insert(i + 1, to_data(q));
                        ev.changed = true;
                    }
                    ev.consumed = on_node.is_some() || ev.changed;
                } else if let (true, Some(i)) = (resp.secondary_clicked(), on_node) {
                    ev.changed = self.remove_node(i);
                    ev.consumed = true;
                }
            }
        }

        // Hover feedback (the node being dragged stays lit). Only the lasso's
        // own cell may touch it: a grid calls this for every cell per frame.
        if self.cell != cell {
            return ev;
        }
        let was = (self.hot, self.ghost.is_some());
        (self.hot, self.ghost) = (None, None);
        if mine && !m.any() {
            let pts = screen(self);
            self.hot = self.drag.or_else(|| {
                resp.hover_pos()
                    .and_then(|p| nearest_node(&pts, p, NODE_HIT))
            });
            if let (None, Some(p)) = (self.hot, resp.hover_pos()) {
                self.ghost = nearest_edge(&pts, p, EDGE_HIT).map(|(_, q)| q);
            }
        }
        if was != (self.hot, self.ghost.is_some()) {
            // `paint` may already have run this frame (Features grid).
            resp.ctx.request_repaint();
        }
        ev
    }

    pub fn contains(&self, p: [f32; 2]) -> bool {
        self.is_closed() && point_in_polygon(p, &self.points)
    }

    /// Outline (closed once it has three vertices) plus node handles; the
    /// hovered node is larger, and a ghost "+" marks where a double-click
    /// would insert one.
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
        for (i, &p) in pts.iter().enumerate() {
            if self.hot == Some(i) {
                painter.circle(p, 5.0, colour, Stroke::new(1.5_f32, Color32::WHITE));
            } else {
                painter.circle_filled(p, 3.5, colour);
            }
        }
        if let Some(g) = self.ghost {
            painter.circle_stroke(g, 5.0, stroke);
            for d in [egui::vec2(3.0, 0.0), egui::vec2(0.0, 3.0)] {
                painter.line_segment([g - d, g + d], stroke);
            }
        }
    }
}

/// Index of the node within `r` px of `p`, nearest first.
fn nearest_node(pts: &[Pos2], p: Pos2, r: f32) -> Option<usize> {
    pts.iter()
        .enumerate()
        .map(|(i, q)| (i, q.distance(p)))
        .filter(|&(_, d)| d <= r)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

/// The closed polygon's edge `i -> i+1` within `r` px of `p`, with the
/// projection of `p` onto it. Insert the new node at `i + 1`.
fn nearest_edge(pts: &[Pos2], p: Pos2, r: f32) -> Option<(usize, Pos2)> {
    (0..pts.len())
        .map(|i| (i, project(p, pts[i], pts[(i + 1) % pts.len()])))
        .filter(|&(_, q)| q.distance(p) <= r)
        .min_by(|a, b| a.1.distance(p).total_cmp(&b.1.distance(p)))
}

/// Closest point to `p` on the segment `a-b`.
fn project(p: Pos2, a: Pos2, b: Pos2) -> Pos2 {
    let ab = b - a;
    let l2 = ab.length_sq();
    if l2 == 0.0 {
        return a;
    }
    a + ab * ((p - a).dot(ab) / l2).clamp(0.0, 1.0)
}

/// Ramer-Douglas-Peucker on an open polyline (screen space).
fn simplify(pts: &[Pos2], eps: f32) -> Vec<Pos2> {
    let (Some(&a), Some(&b)) = (pts.first(), pts.last()) else {
        return Vec::new();
    };
    let far = pts
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, project(p, a, b).distance(p)))
        .max_by(|x, y| x.1.total_cmp(&y.1));
    match far {
        Some((i, d)) if d > eps && pts.len() > 2 => {
            let mut l = simplify(&pts[..=i], eps);
            l.pop();
            l.extend(simplify(&pts[i..], eps));
            l
        }
        _ if pts.len() > 1 => vec![a, b],
        _ => vec![a],
    }
}

/// A closed freehand stroke to at most [`MAX_NODES`] nodes: the end that
/// lands back on the start is dropped, and ε grows until the budget fits.
fn simplify_stroke(stroke: &[Pos2]) -> Vec<Pos2> {
    let mut eps = SIMPLIFY_EPS;
    loop {
        let mut s = simplify(stroke, eps);
        if s.len() > 1 && s[0].distance(s[s.len() - 1]) < NODE_HIT {
            s.pop();
        }
        if s.len() <= MAX_NODES {
            return s;
        }
        eps *= 1.5;
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

    fn frame(ctx: &egui::Context, lasso: &mut Lasso, events: Vec<egui::Event>) -> [LassoEvent; 2] {
        frame_with_modifiers(ctx, lasso, events, egui::Modifiers::COMMAND)
    }

    fn frame_with_modifiers(
        ctx: &egui::Context,
        lasso: &mut Lasso,
        events: Vec<egui::Event>,
        modifiers: egui::Modifiers,
    ) -> [LassoEvent; 2] {
        let mut result = [LassoEvent::default(); 2];
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                Pos2::ZERO,
                egui::vec2(260.0, 160.0),
            )),
            modifiers,
            events,
            ..Default::default()
        };
        let _ = ctx.run_ui(input, |ui| {
            for (cell, event) in result.iter_mut().enumerate() {
                let rect = egui::Rect::from_min_size(
                    egui::pos2(10.0 + cell as f32 * 120.0, 10.0),
                    egui::Vec2::splat(100.0),
                );
                let response = ui.interact(rect, ui.id().with(cell), egui::Sense::click_and_drag());
                *event = lasso.interact(cell, &response, |p| [p.x, p.y], |p| p.into());
            }
        });
        result
    }

    fn pointer_button(pos: Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::COMMAND,
        }
    }

    #[test]
    fn clicked_polygon_remains_a_draft_between_clicks_and_uses_press_modifiers() {
        let ctx = egui::Context::default();
        let mut lasso = Lasso::default();
        frame(&ctx, &mut lasso, vec![]);
        for pos in [
            egui::pos2(25.0, 25.0),
            egui::pos2(95.0, 25.0),
            egui::pos2(95.0, 95.0),
        ] {
            frame(
                &ctx,
                &mut lasso,
                vec![egui::Event::PointerMoved(pos), pointer_button(pos, true)],
            );
            frame(&ctx, &mut lasso, vec![pointer_button(pos, false)]);
            assert!(lasso.is_drafting());
            assert!(
                !lasso.is_editing(),
                "mouse is up, but polygon construction is not finished"
            );
        }
        assert!(lasso.is_closed());
        assert!(lasso.finish_polygon_on_release(false, false));
        assert!(!lasso.is_drafting());
        let pos = egui::pos2(25.0, 95.0);
        frame(
            &ctx,
            &mut lasso,
            vec![egui::Event::PointerMoved(pos), pointer_button(pos, true)],
        );
        let release = egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        };
        frame_with_modifiers(&ctx, &mut lasso, vec![release], egui::Modifiers::NONE);
        assert_eq!(
            lasso.points.len(),
            4,
            "releasing Ctrl before mouse-up does not lose the last vertex"
        );
    }

    #[test]
    fn a_ctrl_click_uses_its_own_modifiers_not_the_frames_final_keys() {
        let ctx = egui::Context::default();
        let mut lasso = Lasso::new(0, vec![[30.0, 30.0], [90.0, 30.0]]);
        frame(&ctx, &mut lasso, vec![]);
        let pos = egui::pos2(90.0, 90.0);
        frame(
            &ctx,
            &mut lasso,
            vec![egui::Event::PointerMoved(pos), pointer_button(pos, true)],
        );
        let event = frame_with_modifiers(
            &ctx,
            &mut lasso,
            vec![pointer_button(pos, false)],
            egui::Modifiers::NONE,
        );
        assert!(event[0].drawn && event[0].changed);
        assert!(lasso.is_closed());
        assert_eq!(lasso.points[2], [pos.x, pos.y]);
    }

    #[test]
    fn an_unfinished_freehand_stroke_is_active_and_belongs_to_its_projection() {
        let ctx = egui::Context::default();
        let mut lasso = Lasso::default();
        frame(&ctx, &mut lasso, vec![]);
        let start = egui::pos2(150.0, 30.0);
        frame(
            &ctx,
            &mut lasso,
            vec![
                egui::Event::PointerMoved(start),
                pointer_button(start, true),
            ],
        );
        let events = frame(
            &ctx,
            &mut lasso,
            vec![egui::Event::PointerMoved(egui::pos2(200.0, 30.0))],
        );
        assert!(
            lasso.is_active(),
            "Esc must see a stroke before it has vertices"
        );
        assert!(lasso.is_editing());
        assert_eq!(lasso.cell, 1);
        assert!(
            !events[0].consumed,
            "the sibling projection does not own this stroke"
        );
        assert!(events[1].consumed);
    }

    #[test]
    fn clearing_a_freehand_stroke_does_not_recreate_it_when_the_mouse_is_released() {
        let ctx = egui::Context::default();
        let mut lasso = Lasso::default();
        frame(&ctx, &mut lasso, vec![]);
        let start = egui::pos2(30.0, 30.0);
        frame(
            &ctx,
            &mut lasso,
            vec![
                egui::Event::PointerMoved(start),
                pointer_button(start, true),
            ],
        );
        for p in [egui::pos2(90.0, 30.0), egui::pos2(90.0, 90.0)] {
            frame(&ctx, &mut lasso, vec![egui::Event::PointerMoved(p)]);
        }
        assert!(lasso.is_active());
        lasso.clear();
        let end = egui::pos2(30.0, 90.0);
        frame(&ctx, &mut lasso, vec![egui::Event::PointerMoved(end)]);
        frame(&ctx, &mut lasso, vec![pointer_button(end, false)]);
        assert!(!lasso.is_active());
        assert!(!lasso.is_editing());
        assert!(lasso.points.is_empty());
        assert!(lasso.stroke.is_empty());
    }

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
            points: c[..2].to_vec(),
            ..Lasso::default()
        };
        assert!(!open.contains([0.5, 0.0]), "two vertices enclose nothing");
    }

    #[test]
    fn stroke_simplifies_and_nodes_edit() {
        // A jittery 100 px square drawn from its top-left corner, ending
        // 3 px short of where it started.
        let corners = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
            (0.0, 3.0),
        ];
        let mut stroke = Vec::new();
        for w in corners.windows(2) {
            for k in 0..20 {
                let t = k as f32 / 20.0;
                let j = if k % 2 == 0 { 0.6 } else { -0.6 };
                let (x, y) = (
                    w[0].0 + (w[1].0 - w[0].0) * t,
                    w[0].1 + (w[1].1 - w[0].1) * t,
                );
                stroke.push(Pos2::new(x + j, y + j));
            }
        }
        stroke.push(Pos2::new(0.0, 3.0));
        let nodes = simplify_stroke(&stroke);
        assert_eq!(nodes.len(), 4, "corners kept, jitter and the overlap gone");
        for c in &corners[..4] {
            assert!(nearest_node(&nodes, Pos2::new(c.0, c.1), 2.0).is_some());
        }
        // A dense scribble still fits the budget.
        let zig: Vec<Pos2> = (0..500)
            .map(|i| Pos2::new(i as f32, (i % 2) as f32 * 30.0))
            .collect();
        assert!(simplify_stroke(&zig).len() <= MAX_NODES);

        // Insert on the top edge, then delete back down to a triangle.
        let mut l = Lasso {
            points: nodes.iter().map(|p| [p.x, p.y]).collect(),
            ..Lasso::default()
        };
        let hit = Pos2::new(50.0, 2.0);
        assert_eq!(nearest_node(&nodes, hit, NODE_HIT), None);
        let (i, q) = nearest_edge(&nodes, hit, EDGE_HIT).expect("on the top edge");
        assert!((q.x - 50.0).abs() < 1.0 && q.y.abs() < 1.0);
        l.points.insert(i + 1, [q.x, q.y]);
        assert_eq!(l.points.len(), 5);
        assert!(l.remove_node(i + 1) && l.remove_node(0));
        assert!(
            l.is_closed() && !l.remove_node(0),
            "never below three nodes"
        );
        assert!(nearest_edge(&nodes, Pos2::new(50.0, 50.0), EDGE_HIT).is_none());
    }
}

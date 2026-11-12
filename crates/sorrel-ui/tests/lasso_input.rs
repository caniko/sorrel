//! Headless regression tests for the lasso's native pointer gestures.
//!
//! `egui::Context::run_ui` is driven with two adjacent plot cells, a stable
//! screen rectangle and increasing `RawInput::time`, so egui's click/drag
//! classification sees real press durations. Ctrl is reported the way winit
//! reports the Windows primary modifier: `ctrl` and `command` are both set
//! (not `Modifiers::COMMAND`, which leaves `ctrl` false). Nothing here needs
//! a GPU, a filesystem fixture, or the application shell.

use egui::{Event, Key, Modifiers, PointerButton, Pos2, RawInput, Rect, Sense, Vec2};
use sorrel_ui::lasso::{Lasso, LassoEvent};

const CELLS: usize = 2;
const STEP: f64 = 1.0 / 60.0;
const RECT_SIDE: f32 = 100.0;
const FIRST_CELL: f32 = 10.0;
const CELL_PITCH: f32 = 110.0;

fn screen_rect() -> Rect {
    Rect::from_min_size(Pos2::ZERO, Vec2::new(300.0, 140.0))
}

fn plot_rect(cell: usize) -> Rect {
    Rect::from_min_size(
        Pos2::new(FIRST_CELL + cell as f32 * CELL_PITCH, FIRST_CELL),
        Vec2::splat(RECT_SIDE),
    )
}

/// A point at plot-relative coordinates inside `cell`.
fn at(cell: usize, x: f32, y: f32) -> Pos2 {
    let rect = plot_rect(cell);
    Pos2::new(rect.min.x + x, rect.min.y + y)
}

/// The same point in data coordinates (the harness's projection is identity).
fn data_at(cell: usize, x: f32, y: f32) -> [f32; 2] {
    let p = at(cell, x, y);
    [p.x, p.y]
}

/// Windows primary modifier: winit sets both the physical Ctrl and the
/// platform shortcut flag.
fn ctrl() -> Modifiers {
    Modifiers {
        ctrl: true,
        command: true,
        ..Modifiers::default()
    }
}

fn shift() -> Modifiers {
    Modifiers {
        shift: true,
        ..Modifiers::default()
    }
}

fn pointer_button(pos: Pos2, pressed: bool, modifiers: Modifiers) -> Event {
    Event::PointerButton {
        pos,
        button: PointerButton::Primary,
        pressed,
        modifiers,
    }
}

fn key_pressed(key: Key, modifiers: Modifiers) -> Event {
    Event::Key {
        key,
        physical_key: Some(key),
        pressed: true,
        repeat: false,
        modifiers,
    }
}

fn key_k_pressed(modifiers: Modifiers) -> Event {
    key_pressed(Key::K, modifiers)
}

fn closed_lasso(cell: usize) -> Lasso {
    Lasso::new(
        cell,
        vec![
            data_at(cell, 20.0, 20.0),
            data_at(cell, 80.0, 20.0),
            data_at(cell, 80.0, 80.0),
            data_at(cell, 20.0, 80.0),
        ],
    )
}

/// Runs the real `Lasso::interact` for both projections every frame, exactly
/// as a grid of views would, and records what each projection reported.
struct Harness {
    ctx: egui::Context,
    lasso: Lasso,
    time: f64,
}

impl Harness {
    fn new(lasso: Lasso) -> Self {
        let mut harness = Self {
            ctx: egui::Context::default(),
            lasso,
            time: 100.0,
        };
        // Register both widgets and settle the pointer before the gesture.
        harness.frame(Modifiers::NONE, Vec::new());
        harness
    }

    fn frame(&mut self, modifiers: Modifiers, events: Vec<Event>) -> [LassoEvent; CELLS] {
        self.time += STEP;
        let input = RawInput {
            screen_rect: Some(screen_rect()),
            time: Some(self.time),
            modifiers,
            events,
            ..Default::default()
        };
        let mut reported = [LassoEvent::default(); CELLS];
        let ctx = &self.ctx;
        let lasso = &mut self.lasso;
        let _ = ctx.run_ui(input, |ui| {
            for (cell, event) in reported.iter_mut().enumerate() {
                let response = ui.interact(
                    plot_rect(cell),
                    egui::Id::new(("lasso-input-plot", cell)),
                    Sense::click_and_drag(),
                );
                *event = lasso.interact(
                    cell,
                    &response,
                    |p: Pos2| [p.x, p.y],
                    |p: [f32; 2]| Pos2::new(p[0], p[1]),
                );
            }
        });
        reported
    }

    /// A press and release at the same point, with Ctrl held throughout.
    /// Returns the release frame's events (the ones that add a vertex).
    fn ctrl_click(&mut self, cell: usize, pos: Pos2) -> [LassoEvent; CELLS] {
        debug_assert!(plot_rect(cell).contains(pos));
        self.frame(
            ctrl(),
            vec![Event::PointerMoved(pos), pointer_button(pos, true, ctrl())],
        );
        self.frame(ctrl(), vec![pointer_button(pos, false, ctrl())])
    }
}

fn assert_drawn(events: &[LassoEvent; CELLS], cell: usize) {
    assert!(
        events[cell].changed && events[cell].drawn && events[cell].consumed,
        "cell {cell} should report a consumed gesture that changed the polygon: {:?}",
        events[cell]
    );
}

#[test]
fn ctrl_clicked_polygon_stays_a_draft_until_ctrl_is_released() {
    let mut h = Harness::new(Lasso::default());
    let corners = [
        at(0, 20.0, 20.0),
        at(0, 80.0, 20.0),
        at(0, 80.0, 80.0),
        at(0, 20.0, 80.0),
    ];
    for (i, &pos) in corners.iter().enumerate() {
        let events = h.ctrl_click(0, pos);
        assert_eq!(h.lasso.points.len(), i + 1);
        assert_eq!(h.lasso.points[i], [pos.x, pos.y]);
        assert!(
            h.lasso.is_drafting(),
            "vertex {} must leave the clicked polygon open while Ctrl is held",
            i + 1
        );
        assert!(
            !h.lasso.is_editing(),
            "the mouse is up between vertices; this is a draft, not a drag"
        );
        assert_drawn(&events, 0);
        assert_eq!(h.lasso.is_closed(), i + 1 >= 3);
    }

    // Ctrl is still held while the pointer is up: not a finish.
    assert!(!h.lasso.finish_polygon_on_release(true, false));
    assert!(h.lasso.is_drafting());

    // Ctrl release is the finish, and keeps every clicked vertex.
    h.frame(Modifiers::NONE, Vec::new());
    assert!(h.lasso.finish_polygon_on_release(false, false));
    assert!(!h.lasso.is_drafting());
    assert!(!h.lasso.is_editing());
    assert!(h.lasso.is_closed());
    assert_eq!(h.lasso.points.len(), 4);
}

#[test]
fn ctrl_released_before_the_last_mouseup_still_records_the_vertex() {
    let mut h = Harness::new(Lasso::default());
    h.ctrl_click(0, at(0, 20.0, 20.0));
    h.ctrl_click(0, at(0, 80.0, 20.0));
    assert!(h.lasso.is_drafting());

    // The third press latches Ctrl; the release event still carries Ctrl,
    // but the frame's final keyboard state no longer does.
    let last = at(0, 80.0, 80.0);
    h.frame(
        ctrl(),
        vec![
            Event::PointerMoved(last),
            pointer_button(last, true, ctrl()),
        ],
    );
    // Ctrl may go up while the button is still captured; the release frame
    // must be processed before the draft is allowed to finish.
    assert!(!h.lasso.finish_polygon_on_release(false, true));
    assert!(h.lasso.is_drafting());

    // Ctrl is physically up before the mouseup, so even the release event
    // carries no modifiers; only the press-frame latch can save the vertex.
    let events = h.frame(
        Modifiers::NONE,
        vec![pointer_button(last, false, Modifiers::NONE)],
    );

    assert_eq!(h.lasso.points.len(), 3, "the press modifier latches");
    assert_eq!(h.lasso.points[2], [last.x, last.y]);
    assert_drawn(&events, 0);
    assert!(
        !h.lasso.is_drafting(),
        "Ctrl went up before the mouse, so the polygon is finished"
    );
    assert!(!h.lasso.is_editing());
}

#[test]
fn plain_k_finishes_a_ctrl_clicked_draft() {
    let mut h = Harness::new(Lasso::default());
    for pos in [at(0, 20.0, 20.0), at(0, 80.0, 20.0), at(0, 80.0, 80.0)] {
        h.ctrl_click(0, pos);
    }
    assert!(h.lasso.is_drafting());
    assert!(h.lasso.is_closed());

    // The app treats K as an explicit finish as well as a split.
    h.frame(Modifiers::NONE, vec![key_k_pressed(Modifiers::NONE)]);
    h.lasso.finish_polygon();

    assert!(!h.lasso.is_drafting());
    assert!(!h.lasso.is_editing());
    assert!(h.lasso.is_closed());
    assert_eq!(h.lasso.points.len(), 3);
}

#[test]
fn a_modifier_appearing_on_node_release_does_not_leave_the_lasso_editing() {
    for modifier in [ctrl(), shift()] {
        let mut h = Harness::new(closed_lasso(0));
        let node = at(0, 20.0, 20.0);
        h.frame(
            Modifiers::NONE,
            vec![
                Event::PointerMoved(node),
                pointer_button(node, true, Modifiers::NONE),
            ],
        );

        let mid = at(0, 35.0, 35.0);
        let events = h.frame(Modifiers::NONE, vec![Event::PointerMoved(mid)]);
        assert!(h.lasso.is_editing(), "the node drag has started");
        assert!(events[0].changed && events[0].consumed);

        // The modifier appears only on the release frame. The captured drag
        // must still be cleared, or every later K would be rejected as an
        // edit in progress.
        let end = at(0, 45.0, 50.0);
        let events = h.frame(
            modifier,
            vec![
                Event::PointerMoved(end),
                pointer_button(end, false, modifier),
            ],
        );
        assert!(
            !h.lasso.is_editing(),
            "modifier {modifier:?} left the captured node drag stuck"
        );
        assert!(!h.lasso.is_drafting());
        assert!(events[0].changed && events[0].consumed);
        assert_eq!(h.lasso.points.len(), 4, "a node move adds no vertex");
        assert_eq!(h.lasso.points[0], [end.x, end.y]);

        // A plain finish after the release must not be blocked.
        h.lasso.finish_polygon();
        assert!(h.lasso.is_closed());
    }
}

#[test]
fn ctrl_freehand_release_replaces_the_existing_polygon() {
    let old = closed_lasso(0);
    let old_points = old.points.clone();
    let mut h = Harness::new(old);

    let start = at(0, 25.0, 20.0);
    h.frame(
        ctrl(),
        vec![
            Event::PointerMoved(start),
            pointer_button(start, true, ctrl()),
        ],
    );
    for pos in [at(0, 90.0, 20.0), at(0, 90.0, 85.0), at(0, 25.0, 85.0)] {
        h.frame(ctrl(), vec![Event::PointerMoved(pos)]);
    }
    assert!(h.lasso.is_editing(), "the freehand stroke is live");

    // Land next to (not exactly on) the start, as a hand-drawn loop does.
    let end = at(0, 29.0, 24.0);
    let events = h.frame(
        ctrl(),
        vec![Event::PointerMoved(end), pointer_button(end, false, ctrl())],
    );

    assert_drawn(&events, 0);
    assert!(
        !h.lasso.is_editing(),
        "the released stroke is no longer an edit"
    );
    assert!(!h.lasso.is_drafting());
    assert!(h.lasso.is_closed());
    assert_eq!(h.lasso.cell, 0);
    assert_ne!(
        h.lasso.points, old_points,
        "the release replaces the polygon"
    );
    assert!(h.lasso.points.len() >= 3);
    assert!(
        h.lasso.contains([67.5, 62.5]),
        "the replacement follows the drawn outline"
    );
}

#[test]
fn a_long_stationary_ctrl_press_adds_one_vertex_not_an_empty_loop() {
    let mut h = Harness::new(Lasso::default());
    let pos = at(0, 40.0, 40.0);
    h.frame(
        ctrl(),
        vec![Event::PointerMoved(pos), pointer_button(pos, true, ctrl())],
    );

    // Keep the pointer perfectly still for longer than egui's click duration.
    // egui classifies this as a decidedly dragging gesture; the stationary
    // stroke fallback must still turn it into one vertex.
    let mut classified_as_drag = false;
    for _ in 0..90 {
        h.frame(ctrl(), Vec::new());
        classified_as_drag |= h.lasso.is_editing();
    }
    assert!(
        classified_as_drag,
        "the long press must exercise the stationary drag fallback"
    );
    let events = h.frame(ctrl(), vec![pointer_button(pos, false, ctrl())]);

    assert_eq!(h.lasso.points.len(), 1, "one press is one vertex");
    assert_eq!(h.lasso.points[0], [pos.x, pos.y]);
    assert_drawn(&events, 0);
    assert!(!h.lasso.is_editing());
    assert!(
        h.lasso.is_drafting(),
        "Ctrl is still down after the long press"
    );
}

#[test]
fn clear_during_a_clicked_draft_survives_the_pending_mouseup() {
    let mut h = Harness::new(Lasso::default());
    h.ctrl_click(0, at(0, 20.0, 20.0));
    assert_eq!(h.lasso.points.len(), 1);

    let second = at(0, 80.0, 20.0);
    h.frame(
        ctrl(),
        vec![
            Event::PointerMoved(second),
            pointer_button(second, true, ctrl()),
        ],
    );
    // Escape arrives while the second vertex's button is still down. The app
    // dispatches Cancel before drawing the pane, so the lasso is already empty
    // when this frame's Escape event reaches egui.
    h.lasso.clear();
    h.frame(ctrl(), vec![key_pressed(Key::Escape, ctrl())]);
    let events = h.frame(ctrl(), vec![pointer_button(second, false, ctrl())]);

    assert!(
        !h.lasso.is_active(),
        "Escape must not be undone by the pending mouseup \
         (points={:?}, cell={}, editing={})",
        h.lasso.points,
        h.lasso.cell,
        h.lasso.is_editing()
    );
    assert!(h.lasso.points.is_empty());
    assert!(!h.lasso.is_editing());
    assert!(!h.lasso.is_drafting());
    assert!(!events[0].changed);
}

#[test]
fn a_fresh_ctrl_press_after_escape_starts_a_new_polygon() {
    // Cancel during a cell 1 press: cell 0 is visited first on every frame,
    // so the discard latch must survive the release until cell 1 sees it.
    let mut h = Harness::new(Lasso::default());
    h.ctrl_click(1, at(1, 20.0, 20.0));
    let cancelled = at(1, 80.0, 20.0);
    h.frame(
        ctrl(),
        vec![
            Event::PointerMoved(cancelled),
            pointer_button(cancelled, true, ctrl()),
        ],
    );
    h.lasso.clear();
    h.frame(ctrl(), vec![key_pressed(Key::Escape, ctrl())]);
    let events = h.frame(ctrl(), vec![pointer_button(cancelled, false, ctrl())]);

    assert!(!h.lasso.is_active(), "the cancelled click is discarded");
    assert!(h.lasso.points.is_empty());
    assert!(!h.lasso.is_drafting());
    assert!(!events[0].changed && !events[1].changed);

    // The first fresh press is seen by cell 0 while the latch is still set;
    // only the owning response may clear it. Its release must add the first
    // vertex of a new polygon, not be discarded with the old gesture.
    let vertices = [at(1, 20.0, 20.0), at(1, 80.0, 20.0), at(1, 80.0, 80.0)];
    for (i, &pos) in vertices.iter().enumerate() {
        let events = h.ctrl_click(1, pos);
        assert_eq!(h.lasso.cell, 1);
        assert_eq!(h.lasso.points.len(), i + 1);
        assert_eq!(h.lasso.points[i], [pos.x, pos.y]);
        assert!(h.lasso.is_drafting());
        assert_drawn(&events, 1);
    }
    assert!(h.lasso.is_closed());

    h.frame(Modifiers::NONE, Vec::new());
    assert!(h.lasso.finish_polygon_on_release(false, false));
    assert!(!h.lasso.is_drafting());
    assert!(!h.lasso.is_editing());
    assert_eq!(h.lasso.points.len(), 3);
    assert_eq!(h.lasso.cell, 1);
}

#[test]
fn clear_during_a_freehand_stroke_survives_the_pending_mouseup() {
    let mut h = Harness::new(Lasso::default());
    let start = at(0, 20.0, 20.0);
    h.frame(
        ctrl(),
        vec![
            Event::PointerMoved(start),
            pointer_button(start, true, ctrl()),
        ],
    );
    for pos in [at(0, 80.0, 20.0), at(0, 80.0, 80.0)] {
        h.frame(ctrl(), vec![Event::PointerMoved(pos)]);
    }
    assert!(h.lasso.is_editing());

    h.lasso.clear();
    let end = at(0, 20.0, 80.0);
    let events = h.frame(
        ctrl(),
        vec![Event::PointerMoved(end), pointer_button(end, false, ctrl())],
    );

    assert!(!h.lasso.is_active());
    assert!(h.lasso.points.is_empty());
    assert!(!h.lasso.is_editing());
    assert!(!h.lasso.is_drafting());
    assert!(!events[0].changed);
}

#[test]
fn only_the_projection_that_owns_the_gesture_mutates_the_shared_lasso() {
    // A node drag in cell 0: the sibling projection is still visited every
    // frame but must report nothing and leave the point alone.
    let mut h = Harness::new(closed_lasso(0));
    let node = at(0, 20.0, 20.0);
    h.frame(
        Modifiers::NONE,
        vec![
            Event::PointerMoved(node),
            pointer_button(node, true, Modifiers::NONE),
        ],
    );
    let moved = at(0, 35.0, 35.0);
    let events = h.frame(Modifiers::NONE, vec![Event::PointerMoved(moved)]);
    assert!(h.lasso.is_editing());
    assert!(events[0].changed && events[0].consumed);
    assert!(
        !events[1].changed && !events[1].consumed,
        "the sibling does not claim the owner's node drag"
    );
    let release = at(0, 45.0, 50.0);
    h.frame(
        Modifiers::NONE,
        vec![
            Event::PointerMoved(release),
            pointer_button(release, false, Modifiers::NONE),
        ],
    );
    assert_eq!(h.lasso.points[0], [release.x, release.y]);

    // A Ctrl freehand that starts in cell 1 must not move the cell 0
    // polygon, and the sibling's early return must protect the live stroke.
    let h_old_points = closed_lasso(0).points;
    let mut h = Harness::new(closed_lasso(0));
    let start = at(1, 20.0, 20.0);
    h.frame(
        ctrl(),
        vec![
            Event::PointerMoved(start),
            pointer_button(start, true, ctrl()),
        ],
    );
    let stroke = at(1, 80.0, 20.0);
    let events = h.frame(ctrl(), vec![Event::PointerMoved(stroke)]);
    assert_eq!(h.lasso.cell, 1);
    assert!(h.lasso.is_editing());
    assert!(events[1].consumed);
    assert!(!events[1].changed, "a live stroke has no vertices yet");
    assert!(
        !events[0].changed && !events[0].consumed,
        "the sibling cannot consume or extend the stroke"
    );
    assert_eq!(
        h.lasso.points, h_old_points,
        "the cell 0 polygon is untouched by the cell 1 gesture"
    );

    // Even a hover exactly on a cell 0 node, while cell 1 owns the pointer,
    // must not start an edit there or clear the stroke.
    let sibling_node = at(0, 20.0, 20.0);
    let events = h.frame(Modifiers::NONE, vec![Event::PointerMoved(sibling_node)]);
    assert!(h.lasso.is_editing(), "the cell 1 stroke is still live");
    assert!(
        !events[0].changed && !events[0].consumed,
        "a sibling visit is inert while another projection is editing"
    );
    assert_eq!(h.lasso.points, h_old_points);
}

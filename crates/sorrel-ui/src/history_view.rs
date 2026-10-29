//! History pane: the session's action tree, oldest first. Each row is a
//! node with its actor, what it did and when (UTC); a branch is indented
//! under the node it left from, the checked-out node is highlighted, and
//! saves are marked. Clicking a row checks it out.

use egui::Ui;
use sorrel_data::{actor_name, describe_command, Session};
use sorrel_io::DataProvider;

/// One row: the node (`None` = the sort as loaded), its indent and its
/// number of children.
pub(crate) type Row = (Option<u64>, usize, usize);

/// Rows in depth-first order, oldest child first. A node is indented one
/// step further than its parent unless it is the parent's first child, so
/// a linear history stays flat and every branch steps right.
// ponytail: rebuilt every frame, O(nodes); cache on the node count if very
// long sessions make the pane slow.
pub(crate) fn rows<P: DataProvider>(session: &Session<P>) -> Vec<Row> {
    let n = session.n_nodes();
    // children[0] = the root's, children[i + 1] = node i's.
    let mut children: Vec<Vec<u64>> = vec![Vec::new(); n + 1];
    for id in 0..n as u64 {
        let parent = session.node(id).and_then(|(p, ..)| p);
        children[parent.map_or(0, |p| p as usize + 1)].push(id);
    }
    let mut out = Vec::with_capacity(n + 1);
    let mut stack = vec![(None, 0)];
    while let Some((node, indent)) = stack.pop() {
        let kids = &children[node.map_or(0, |n: u64| n as usize + 1)];
        out.push((node, indent, kids.len()));
        for (i, &c) in kids.iter().enumerate().rev() {
            stack.push((Some(c), indent + usize::from(i > 0)));
        }
    }
    out
}

/// `HH:MM:SS` (UTC) of a Unix-millisecond timestamp; empty for 0 (nodes
/// upgraded from a v2 journal have no time).
fn clock(ts_ms: u64) -> String {
    if ts_ms == 0 {
        return String::new();
    }
    let s = ts_ms / 1000 % 86_400;
    format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// Draw the tree; returns the node a click asked to check out.
pub fn history_view<P: DataProvider>(ui: &mut Ui, session: &Session<P>) -> Option<Option<u64>> {
    let theme = crate::theme::Theme::from_ui(ui);
    let head = session.head();
    let saved: Vec<Option<u64>> = session.seals().iter().map(|s| s.0).collect();
    ui.weak("click a row to go back to it · nothing is ever discarded");
    let mut picked = None;
    for (node, indent, branches) in rows(session) {
        ui.horizontal(|ui| {
            ui.add_space(14.0 * indent as f32);
            let (who, what, when) = match node.and_then(|n| session.node(n)) {
                Some((_, actor, ts, cmd)) => (
                    actor_name(actor).to_owned(),
                    describe_command(cmd),
                    clock(ts),
                ),
                None => (String::new(), "the sort as loaded".into(), String::new()),
            };
            // A fixed-width actor slot keeps the descriptions in line.
            ui.add_sized(
                [44.0, 16.0],
                egui::Label::new(egui::RichText::new(who).color(theme.caption).small()).truncate(),
            );
            let text = match branches {
                b if b > 1 => format!("{what}  ({b} branches)"),
                _ => what,
            };
            if ui.selectable_label(node == head, text).clicked() && node != head {
                picked = Some(node);
            }
            if saved.contains(&node) {
                ui.label(egui::RichText::new("saved").color(theme.good).small());
            }
            if !when.is_empty() {
                ui.label(egui::RichText::new(when).color(theme.caption).small());
            }
        });
    }
    picked
}

#[cfg(test)]
mod tests {
    use super::*;
    use sorrel_data::{CurationCommand, Journal, PhyLabelOp};
    use sorrel_io::{ClusterId, SampleIndex, TraceSamples, TraceSlice};

    struct Stub;
    impl DataProvider for Stub {
        type Label = u8;
        fn sample_rate(&self) -> f32 {
            1.0
        }
        fn n_channels(&self) -> u32 {
            1
        }
        fn n_samples(&self) -> SampleIndex {
            SampleIndex(0)
        }
        fn n_clusters(&self) -> u32 {
            2
        }
        fn spike_times(&self, _: ClusterId) -> &[SampleIndex] {
            &[]
        }
        fn trace(&self, start: SampleIndex, _: u32) -> TraceSlice<'_> {
            TraceSlice {
                start,
                n_channels: 1,
                samples: TraceSamples::I16(&[]),
            }
        }
        fn initial_labels(&self) -> Vec<u8> {
            vec![0; 2]
        }
    }
    impl sorrel_data::session::ApplyPhyLabel for Stub {
        fn label_from_op(op: PhyLabelOp) -> u8 {
            op as u8
        }
    }

    /// 0 → 1, then undo to 0 and branch to 2, then undo to the root and
    /// branch to 3: the first child stays in its parent's column, every
    /// later sibling steps right.
    #[test]
    fn branches_indent_and_linear_history_stays_flat() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = Session::new(Stub, Journal::open(&dir.path().join("j")).unwrap());
        let relabel = |c| CurationCommand::Relabel {
            cluster: ClusterId(c),
            op: PhyLabelOp::SetGood,
        };
        s.dispatch(relabel(0)).unwrap();
        s.dispatch(relabel(1)).unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        s.dispatch(relabel(1)).unwrap();
        s.checkout(None).unwrap();
        s.dispatch(relabel(0)).unwrap();
        assert_eq!(
            rows(&s),
            [
                (None, 0, 2),
                (Some(0), 0, 2),
                (Some(1), 0, 0),
                (Some(2), 1, 0),
                (Some(3), 1, 0)
            ]
        );
        assert_eq!(clock(0), "");
        assert_eq!(clock(3_723_000), "01:02:03");
    }
}

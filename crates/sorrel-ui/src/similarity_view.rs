//! phy SimilarityView: clusters ranked by template similarity to the last best cluster.
//!
//! The same table as the ClusterView ([`cluster_table`]) restricted to the
//! ranked list, with a `similarity` column sorted best-first by default.

use crate::brush::{Brushes, Ghost};
use crate::cluster_view::{cluster_table, ClusterRows, ClusterTableState, RowClick, TableCtx};
use crate::jobs::{JobCtx, Memo};
use crate::theme::{selection_colour, Theme};
use egui::Ui;
use sorrel_data::{preview_merge, Session};
use sorrel_io::{ClusterId, DataProvider};
use std::collections::HashMap;

/// What merging one candidate into the target does to the target's quality
/// (`preview_merge` deltas).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MergeTip {
    pub composite: f32,
    pub contamination: f32,
    pub warns: bool,
}

impl MergeTip {
    /// "composite +0.04\ncontamination +0.8 %" (contamination is a fraction).
    pub(crate) fn text(&self) -> String {
        format!(
            "composite {:+.2}\ncontamination {:+.1} %",
            self.composite,
            self.contamination * 100.0
        )
    }
}

/// Merge previews per hovered candidate row, one [`Memo`] each keyed by
/// (session version, target), so a row is computed once and not per frame.
#[derive(Default)]
pub struct MergeTips(HashMap<ClusterId, Memo<(u64, ClusterId), MergeTip>>);

impl MergeTips {
    /// A row for candidate `c` is hovered: ghost it and show what merging it
    /// into `target` would do, in the warn colour when it looks risky.
    pub(crate) fn hover<P: DataProvider>(
        &mut self,
        ui: &Ui,
        jobs: &JobCtx<Session<P>>,
        version: u64,
        target: ClusterId,
        c: ClusterId,
        brushes: &mut Brushes,
    ) {
        if c == target {
            return;
        }
        brushes.ghost_next = Some(Ghost::Merge(c));
        let tip = self
            .0
            .entry(c)
            .or_default()
            .ensure(
                (version, target),
                &jobs.session,
                jobs.repaint(),
                move |s, _| {
                    let p = preview_merge(s, &[c], target);
                    MergeTip {
                        composite: p.composite_delta(),
                        contamination: p.contamination_delta(),
                        warns: p.warns(),
                    }
                },
            )
            .copied();
        let theme = Theme::from_ui(ui);
        let (body, colour) = match tip {
            Some(t) => (t.text(), if t.warns { theme.warn } else { theme.text }),
            None => ("computing…".to_owned(), theme.caption),
        };
        egui::Tooltip::always_open(
            ui.ctx().clone(),
            ui.layer_id(),
            egui::Id::new("merge_tip"),
            egui::PopupAnchor::Pointer,
        )
        .show(|ui| {
            ui.colored_label(colour, format!("merge c{c} into c{target}: {body}"));
        });
    }
}

/// The candidate to ghost for a Suggest merge row `(a, b)`: the one that
/// isn't selected, merged into the first selected cluster. `None` unless
/// exactly one of the pair is selected.
pub(crate) fn ghost_pair(
    sel: &[ClusterId],
    a: ClusterId,
    b: ClusterId,
) -> Option<(ClusterId, ClusterId)> {
    let target = *sel.first()?;
    match (sel.contains(&a), sel.contains(&b)) {
        (true, false) => Some((target, b)),
        (false, true) => Some((target, a)),
        _ => None,
    }
}

/// Similarity list memo and the table state of the Similar pane.
pub struct SimilarityState {
    /// `(best clusters, curation version)` the scores were computed for.
    key: Option<(Vec<ClusterId>, u64)>,
    scores: Vec<(ClusterId, f32)>,
    table: ClusterTableState,
    /// Key the table was last drawn for, so `visible` is trusted only
    /// while it belongs to the current scores.
    drawn: Option<(Vec<ClusterId>, u64)>,
    tips: MergeTips,
}

impl Default for SimilarityState {
    fn default() -> Self {
        Self {
            key: None,
            scores: Vec::new(),
            table: ClusterTableState::similarity(),
            drawn: None,
            tips: MergeTips::default(),
        }
    }
}

impl SimilarityState {
    /// Rank every cluster against the last best one (phy's reference),
    /// recomputing only when the best set or the curation moved.
    pub fn ensure<P: DataProvider>(&mut self, session: &Session<P>, best: &[ClusterId]) {
        let key = (best.to_vec(), session.version());
        if self.key.as_ref() == Some(&key) {
            return;
        }
        self.scores = match best.last() {
            Some(&reference) => session.similar_clusters(reference, best),
            None => Vec::new(),
        };
        self.key = Some(key);
    }

    /// The similar list in display order: the rows as drawn (sorted,
    /// filtered) once the pane has drawn this list, else best-first.
    pub fn order(&self) -> Vec<ClusterId> {
        if self.drawn.is_some() && self.drawn == self.key {
            self.table.visible.clone()
        } else {
            self.scores.iter().map(|&(c, _)| c).collect()
        }
    }

    /// The table state, for the snippet bar (`:f`, `:s`).
    pub fn table_mut(&mut self) -> &mut ClusterTableState {
        &mut self.table
    }
}

/// Everything the Similar pane borrows from the app.
pub struct SimilarityPane<'a> {
    pub state: &'a mut SimilarityState,
    pub rows: &'a ClusterRows,
    /// The best (first-group) selection; its last entry is the reference.
    pub best: &'a [ClusterId],
}

/// Returns `(cluster, add)` when a row was clicked: plain click makes it the
/// similar selection, Cmd/Ctrl-click toggles it in the similar part.
pub fn similarity_view<P: DataProvider>(
    ui: &mut Ui,
    session: &Session<P>,
    selected: &[ClusterId],
    pane: &mut SimilarityPane,
    hidden: &mut Vec<String>,
    jobs: &JobCtx<Session<P>>,
    brushes: &mut Brushes,
) -> Option<(ClusterId, bool)> {
    let theme = Theme::from_ui(ui);
    let Some(&reference) = pane.best.last() else {
        crate::views::empty_plot(ui, &theme, "no selection");
        return None;
    };
    pane.state.ensure(session, pane.best);
    if pane.state.scores.is_empty() {
        let msg = if session.similar_templates().is_empty() && !session.has_template_waveforms() {
            "No template similarity: similar_templates.npy / templates.npy not loaded".to_owned()
        } else {
            format!("No other clusters to compare with cluster {reference}")
        };
        crate::views::empty_plot(ui, &theme, &msg);
        return None;
    }
    let slot = selected.iter().position(|&c| c == reference).unwrap_or(0);
    ui.horizontal(|ui| {
        ui.colored_label(selection_colour(slot), "■");
        ui.heading(format!("Most similar to cluster {reference}"));
    });
    ui.weak("click: compare · Ctrl+click: add").on_hover_text(
        "click a row to compare · Cmd/Ctrl-click to add to the selection · \
             hover: ghosts it in Features, Waveforms and CCG, with the merge preview",
    );
    ui.separator();
    let click = cluster_table(
        ui,
        session,
        &mut pane.state.table,
        TableCtx {
            selected,
            anchor: None,
            rows: pane.rows,
            hidden,
            similar: Some(&pane.state.scores),
            highlight: &[],
        },
    );
    pane.state.drawn = pane.state.key.clone();
    // Hovering a row ghosts that cluster as a merge into the first selected one.
    if let (Some(c), Some(&target)) = (pane.state.table.hovered, selected.first()) {
        pane.state
            .tips
            .hover(ui, jobs, session.version(), target, c, brushes);
    }
    match click? {
        RowClick::Toggle(c) => Some((c, true)),
        // Shift-click has no range meaning here; treat it as a plain pick.
        RowClick::Replace(c) | RowClick::Extend(c) => Some((c, false)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ghost_pair_is_the_unselected_one_into_the_first_selected() {
        let c = ClusterId;
        let sel = [c(3), c(8)];
        assert_eq!(ghost_pair(&sel, c(8), c(5)), Some((c(3), c(5))));
        assert_eq!(ghost_pair(&sel, c(5), c(3)), Some((c(3), c(5))));
        // Both selected, or neither: nothing to preview.
        assert_eq!(ghost_pair(&sel, c(3), c(8)), None);
        assert_eq!(ghost_pair(&sel, c(1), c(2)), None);
        assert_eq!(ghost_pair(&[], c(1), c(2)), None);
    }

    #[test]
    fn tip_text_reads_deltas() {
        let t = MergeTip {
            composite: 0.04,
            contamination: 0.008,
            warns: false,
        };
        assert_eq!(t.text(), "composite +0.04\ncontamination +0.8 %");
    }
}

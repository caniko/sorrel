//! phy SimilarityView: clusters ranked by template similarity to the last best cluster.
//!
//! The same table as the ClusterView ([`cluster_table`]) restricted to the
//! ranked list, with a `similarity` column sorted best-first by default.

use crate::cluster_view::{cluster_table, ClusterRows, ClusterTableState, RowClick, TableCtx};
use crate::theme::{selection_colour, Theme};
use egui::Ui;
use sorrel_data::Session;
use sorrel_io::{ClusterId, DataProvider};

/// Similarity list memo and the table state of the Similar pane.
pub struct SimilarityState {
    /// `(best clusters, curation version)` the scores were computed for.
    key: Option<(Vec<ClusterId>, u64)>,
    scores: Vec<(ClusterId, f32)>,
    table: ClusterTableState,
    /// Key the table was last drawn for, so `visible` is trusted only
    /// while it belongs to the current scores.
    drawn: Option<(Vec<ClusterId>, u64)>,
}

impl Default for SimilarityState {
    fn default() -> Self {
        Self {
            key: None,
            scores: Vec::new(),
            table: ClusterTableState::similarity(),
            drawn: None,
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
    ui.weak("click a row to compare · Cmd/Ctrl-click to add to the selection");
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
        },
    );
    pane.state.drawn = pane.state.key.clone();
    match click? {
        RowClick::Toggle(c) => Some((c, true)),
        // Shift-click has no range meaning here; treat it as a plain pick.
        RowClick::Replace(c) | RowClick::Extend(c) => Some((c, false)),
    }
}

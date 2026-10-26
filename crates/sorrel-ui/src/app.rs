use crate::intent::{from_key_in_context, Intent};
use crate::jobs::{read, write, Job, JobCtx};
use crate::layout::{self, Pane, TraceControls, ViewAction, ViewBehavior};
use crate::prefs::UiPrefs;
use crate::selection::SelectionSet;
use crate::views::{
    CcgCache, FeatureViewState, QualityCache, RasterCache, SuggestAction, SuggestCache,
    WaveformCache,
};
use crate::widgets::{cluster_table, ClusterTableState, TraceCache};
use eframe::egui;
use sorrel_data::session::ApplyPhyLabel;
use sorrel_data::SpikeDepths;
use sorrel_data::{
    export_qc, save_and_reseal_journal, save_to_phy, CurationCommand, PhyLabelOp, Session,
};
use sorrel_gpu::GpuContext;
use sorrel_io::{ClusterId, DataProvider, Events};
use sorrel_render::{GpuTracePreproc, TracePreproc};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, RwLock};
use std::time::Instant;

fn label_name(op: PhyLabelOp) -> &'static str {
    match op {
        PhyLabelOp::SetGood => "Good",
        PhyLabelOp::SetMua => "MUA",
        PhyLabelOp::SetNoise => "Noise",
        PhyLabelOp::SetUnsorted => "Unsorted",
    }
}

/// Short human description of a curation command — for status toasts so
/// undo/redo say *what* they reverted instead of a bare "undone".
fn describe_command(cmd: &CurationCommand) -> String {
    match cmd {
        CurationCommand::Relabel { cluster, op } => {
            format!("relabel cluster {} as {}", cluster.0, label_name(*op))
        }
        CurationCommand::Merge { sources, target } => {
            format!("merge {} cluster(s) into {}", sources.len(), target.0)
        }
        CurationCommand::Split {
            cluster, spike_idx, ..
        } => {
            format!("split cluster {} ({} spikes)", cluster.0, spike_idx.len())
        }
        CurationCommand::Note { cluster, .. } => format!("note on cluster {}", cluster.0),
        CurationCommand::Tag { cluster, .. } => format!("retag cluster {}", cluster.0),
        CurationCommand::Batch { children } => {
            // A multi-select relabel is journaled as one batch so a single
            // Undo reverts it; say so instead of "batch of N ops".
            if let Some(CurationCommand::Relabel { op, .. }) = children.first() {
                let uniform = children
                    .iter()
                    .all(|c| matches!(c, CurationCommand::Relabel { op: o, .. } if o == op));
                if uniform {
                    return format!("relabel {} clusters as {}", children.len(), label_name(*op));
                }
            }
            format!("batch of {} ops", children.len())
        }
        // Undo/Redo records never appear in history — they are applied as
        // moves between stacks rather than pushed.
        CurationCommand::Undo => "undo".into(),
        CurationCommand::Redo => "redo".into(),
    }
}

/// Render a `PathBuf` for the top bar: keep the last two segments so users
/// recognise it (`run42/ks4`) without overflowing the bar on deep paths.
/// Paths that already have ≤2 *named* components (root excluded) are shown
/// verbatim, so `/a/b` stays `/a/b` rather than becoming `…/a/b`.
fn short_path(p: &Path) -> String {
    let named: Vec<&std::ffi::OsStr> = p
        .iter()
        .filter(|s| *s != std::path::Component::RootDir.as_os_str())
        .collect();
    if named.len() <= 2 {
        return p.display().to_string();
    }
    let tail = &named[named.len() - 2..];
    format!(
        "…/{}",
        tail.iter()
            .map(|s| s.to_string_lossy())
            .collect::<Vec<_>>()
            .join("/")
    )
}

fn toast(msg: impl Into<String>) -> Option<(String, Instant)> {
    Some((msg.into(), Instant::now()))
}

/// Per-spike depth estimator a backend can plug in (see
/// [`SorrelApp::set_depth_source`]).
pub type DepthFn<P> = fn(&Session<P>) -> Option<SpikeDepths>;

/// Whole-dataset phy exporter the binary plugs in (see
/// [`SorrelApp::set_phy_export`]); returns the directory it wrote.
pub type PhyExportFn<P> = Arc<dyn Fn(&Session<P>) -> anyhow::Result<PathBuf> + Send + Sync>;

/// Generic eframe app. The whole window is monomorphised against `P`.
///
/// The session sits behind an `Arc<RwLock<_>>`: the UI thread takes short
/// read locks per frame, curation commands take the write lock, and
/// background jobs (Suggest ranking, quality metrics, initial seeding)
/// hold a read or write lock on worker threads — see [`crate::jobs`].
pub struct SorrelApp<P: DataProvider + ApplyPhyLabel> {
    session: Arc<RwLock<Session<P>>>,
    /// `Some` while the initial seed/replay runs on a worker thread; the
    /// session must not be touched until it reports.
    loading: Option<Receiver<anyhow::Result<()>>>,
    load_error: Option<String>,
    label_str: fn(&P::Label) -> &'static str,
    selection: SelectionSet,
    window_start: u64,
    window_len: u32,
    target_points: usize,
    error: Option<String>,
    /// Directory to write `spike_clusters.npy` / `cluster_group.tsv` into
    /// when the user triggers a save. Cmd/Ctrl-S is a no-op when unset.
    save_dir: Option<PathBuf>,
    /// See [`SorrelApp::set_save_dir`].
    reseal_on_save: bool,
    /// Preferences read from the dataset, applied once loading completes.
    pending_prefs: Option<UiPrefs>,
    /// Last status message + the time it was set, so the top bar can fade
    /// it out after a few seconds rather than pinning it.
    status: Option<(String, Instant)>,
    /// Snapshot of `(history_len, redo_len)` at the last successful save. The
    /// title bar shows a "•" dirty marker when current state differs.
    saved_state: (usize, usize),
    /// Optional pre-processing applied to the trace window before LTTB.
    trace_cfg: TracePreproc,
    /// Dockable view layout — every view is a tile the user can drag,
    /// split, close and reopen.
    tree: egui_tiles::Tree<Pane>,
    /// Probe geometry — set by the binary when the backend exposes it.
    /// Empty means the WaveformView falls back to single-channel layout.
    channel_positions: Vec<[f32; 2]>,
    /// Shank id per raw channel; empty when the backend has one shank or
    /// no geometry. Enables the per-shank CMR toggle in the trace pane.
    channel_shanks: Vec<u32>,
    per_shank_cmr: bool,
    waveform_cache: WaveformCache,
    table_state: ClusterTableState,
    feature_state: FeatureViewState,
    raster_cache: RasterCache,
    suggest_cache: SuggestCache,
    ccg_cache: CcgCache,
    quality_cache: QualityCache,
    trace_cache: TraceCache,
    /// Dataset events (`events.csv` / `events.npy` next to the data) for
    /// the PSTH pane.
    events: Option<Arc<Events>>,
    psth_state: crate::views::PsthState,
    psth_cache: crate::views::PsthCache,
    /// Backend-specific spike-depth estimator (see [`Self::set_depth_source`]).
    depth_fn: Option<DepthFn<P>>,
    /// Depths memoised per curation version, computed on a worker.
    depth_memo: crate::jobs::Memo<u64, Arc<Option<SpikeDepths>>>,
    /// Phy exporter (see [`Self::set_phy_export`]); no button when unset.
    phy_export: Option<PhyExportFn<P>>,
    /// The running export. Not cancelled by curation: a command waits for
    /// its read lock instead, so a long copy is never silently dropped.
    phy_export_job: Option<Job<(), anyhow::Result<PathBuf>>>,
    /// GPU compute pipelines for trace CMR + HP filtering. `None` until the
    /// app is installed onto an eframe wgpu render state.
    gpu_preproc: Option<GpuTracePreproc>,
    /// A destructive action awaiting user confirmation. Staging it here
    /// lets the user back out before anything hits the journal.
    pending_confirm: Option<PendingConfirm>,
    /// Set once the user has confirmed quitting with unsaved curation, so
    /// the re-issued close request is not intercepted again.
    close_approved: bool,
    show_help: bool,
    /// Last window title sent to the OS; only re-sent when it changes.
    last_title: String,
}

/// A staged action that needs explicit user approval before it runs.
#[derive(Debug)]
struct PendingConfirm {
    /// Short headline shown in the dialog title.
    title: String,
    /// Human prose explaining what will happen. Multi-line.
    body: String,
    action: ConfirmAction,
    /// Status message to show on success — typically explains how to undo.
    on_success: String,
}

#[derive(Debug)]
enum ConfirmAction {
    /// Dispatch this curation command (bulk merges from the Suggest tab).
    Dispatch(CurationCommand),
    /// Close the window; the dialog offers "Save & quit" as well.
    Quit,
}

impl<P: DataProvider + ApplyPhyLabel> SorrelApp<P> {
    /// App over an already seeded and replayed session.
    pub fn new(session: Session<P>, label_str: fn(&P::Label) -> &'static str) -> Self {
        let mut app = Self::build(session, label_str);
        app.finish_loading();
        app
    }

    /// App over a bare session; `loader` (seeding, journal replay) runs
    /// on a worker thread holding the write lock while the window shows a
    /// loading screen, so a large recording opens without a frozen UI.
    pub fn with_loader(
        session: Session<P>,
        label_str: fn(&P::Label) -> &'static str,
        loader: impl FnOnce(&mut Session<P>) -> anyhow::Result<()> + Send + 'static,
    ) -> Self {
        let mut app = Self::build(session, label_str);
        let (tx, rx) = std::sync::mpsc::channel();
        let shared = Arc::clone(&app.session);
        std::thread::Builder::new()
            .name("sorrel-load".into())
            .spawn(move || {
                let result = loader(&mut write(&shared));
                let _ = tx.send(result);
            })
            .expect("spawn sorrel loader thread");
        app.loading = Some(rx);
        app
    }

    fn build(session: Session<P>, label_str: fn(&P::Label) -> &'static str) -> Self {
        let window_len = (session.provider.sample_rate() as u32).max(1) / 10; // 100 ms
        Self {
            session: Arc::new(RwLock::new(session)),
            loading: None,
            load_error: None,
            saved_state: (0, 0),
            label_str,
            selection: SelectionSet::new(),
            window_start: 0,
            window_len,
            target_points: 1024,
            error: None,
            save_dir: None,
            reseal_on_save: false,
            pending_prefs: None,
            status: None,
            trace_cfg: TracePreproc::Off,
            tree: layout::default_tree(),
            channel_positions: Vec::new(),
            channel_shanks: Vec::new(),
            per_shank_cmr: true,
            waveform_cache: WaveformCache::new(),
            table_state: ClusterTableState::default(),
            feature_state: FeatureViewState::default(),
            raster_cache: RasterCache::new(),
            suggest_cache: SuggestCache::new(),
            ccg_cache: CcgCache::new(),
            quality_cache: QualityCache::new(),
            trace_cache: TraceCache::default(),
            events: None,
            psth_state: Default::default(),
            psth_cache: Default::default(),
            depth_fn: None,
            depth_memo: Default::default(),
            phy_export: None,
            phy_export_job: None,
            gpu_preproc: None,
            pending_confirm: None,
            close_approved: false,
            show_help: false,
            last_title: String::new(),
        }
    }

    /// Called once the session is seeded: initial selection, clean-state
    /// snapshot, and the per-dataset prefs.
    fn finish_loading(&mut self) {
        let (n_clusters, sr, history, redo) = {
            let s = read(&self.session);
            (
                s.n_clusters(),
                s.provider.sample_rate(),
                s.history_len(),
                s.redo_len(),
            )
        };
        self.saved_state = (history, redo);
        if n_clusters > 0 && self.selection.is_empty() {
            self.selection = SelectionSet::single(ClusterId(0));
        }
        if let Some(prefs) = self.pending_prefs.take() {
            self.apply_prefs(prefs, n_clusters, sr);
        }
    }

    fn apply_prefs(&mut self, prefs: UiPrefs, n_clusters: u32, sr: f32) {
        if let Some(tree) = prefs.tree.filter(|t| !t.is_empty()) {
            self.tree = tree;
        }
        if let Some(col) = prefs.sort_by {
            self.table_state.sort_by = col;
            self.table_state.sort_descending = prefs.sort_descending;
        }
        self.table_state.hide_empty = prefs.hide_empty;
        self.table_state.hide_noise = prefs.hide_noise;
        if let Some(ms) = prefs.window_ms {
            self.window_len = ((ms / 1000.0) * sr.max(1.0)).round().max(1.0) as u32;
        }
        let hp = prefs.hp.then_some(TracePreproc::HP_DEFAULT_HZ);
        self.trace_cfg = TracePreproc::from_flags(hp, prefs.cmr);
        if let Some(anchor) = prefs.anchor.filter(|&a| a < n_clusters) {
            self.selection = SelectionSet::single(ClusterId(anchor));
        }
    }

    fn save_prefs(&self) {
        let Some(dir) = self.save_dir.as_deref() else {
            return;
        };
        let sr = if self.loading.is_none() {
            read(&self.session).provider.sample_rate().max(1.0)
        } else {
            return;
        };
        let prefs = UiPrefs {
            tree: Some(self.tree.clone()),
            sort_by: Some(self.table_state.sort_by),
            sort_descending: self.table_state.sort_descending,
            hide_empty: self.table_state.hide_empty,
            hide_noise: self.table_state.hide_noise,
            window_ms: Some(self.window_len as f32 * 1000.0 / sr),
            hp: self.trace_cfg.hp().is_some(),
            cmr: self.trace_cfg.has_cmr(),
            anchor: self.selection.anchor().map(|c| c.0),
        };
        if let Err(e) = prefs.save(dir) {
            log::warn!("could not write UI prefs: {e}");
        }
    }

    /// Hand the app the wgpu device/queue from `eframe`'s render state so
    /// trace CMR + HP filter can run on the GPU. Without this the app
    /// transparently falls back to the CPU implementation.
    pub fn install_gpu_compute(
        &mut self,
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
    ) {
        let ctx = GpuContext::from_existing(device, queue);
        self.gpu_preproc = Some(GpuTracePreproc::new(&ctx));
    }

    /// Configure where Cmd/Ctrl-S writes the curated state. The binary
    /// typically points this at the kilosort root. Also loads the
    /// per-dataset UI prefs stored next to the cache.
    ///
    /// `reseal` must be `true` only for backends that read the saved
    /// `spike_clusters.npy` back on open (phy/Kilosort): the journal is then
    /// truncated at save so the next open does not replay the same edits on
    /// top of already-curated data. Backends that reload the original sort
    /// (SortingAnalyzer, NWB, rez) keep the journal as the source of truth.
    pub fn set_save_dir(&mut self, dir: PathBuf, reseal: bool) {
        let prefs = UiPrefs::load(&dir);
        match Events::detect(&dir) {
            Ok(events) => self.events = events.map(Arc::new),
            Err(e) => log::warn!("ignoring events file in {}: {e}", dir.display()),
        }
        self.save_dir = Some(dir);
        self.reseal_on_save = reseal;
        if self.loading.is_some() {
            self.pending_prefs = Some(prefs);
        } else {
            let (n, sr) = {
                let s = read(&self.session);
                (s.n_clusters(), s.provider.sample_rate())
            };
            self.apply_prefs(prefs, n, sr);
        }
    }

    /// Write the phy artifacts into `dir`, dropping stale derived-cache
    /// entries, and reseal the journal when this backend requires it.
    fn save_to(&mut self, dir: &Path) -> anyhow::Result<()> {
        self.cancel_jobs();
        let mut session = write(&self.session);
        if self.reseal_on_save {
            let journal_path = session.journal_path().to_path_buf();
            // Invalidate before the reseal drops the pending-invalidation list.
            if let Err(err) = session.invalidate_dataset_caches() {
                log::warn!("cache invalidation after save failed: {err}");
            }
            let journal = save_and_reseal_journal(&session, dir, &journal_path, self.label_str)?;
            session.reseal(journal);
        } else {
            save_to_phy(&session, dir, self.label_str)?;
            if let Err(err) = session.invalidate_dataset_caches() {
                log::warn!("cache invalidation after save failed: {err}");
            }
        }
        Ok(())
    }

    /// Supply the backend's per-spike depth estimator (for example
    /// `sorrel_data::spike_depths` on Kilosort data) so the Drift pane
    /// plots depth over time instead of amplitude. It runs on a worker
    /// thread once per curation version.
    pub fn set_depth_source(&mut self, f: DepthFn<P>) {
        self.depth_fn = Some(f);
        self.depth_memo.invalidate();
    }

    /// Events for the PSTH pane, replacing any `events.csv|npy` that
    /// [`Self::set_save_dir`] detected; call it after that.
    pub fn set_events(&mut self, events: Events) {
        self.events = Some(Arc::new(events));
    }

    /// Enable the "Export phy" button; `f` runs on a worker thread.
    pub fn set_phy_export(&mut self, f: PhyExportFn<P>) {
        self.phy_export = Some(f);
    }

    /// Supply probe geometry so the WaveformView can render in probe layout
    /// (top-N channels by amplitude positioned at their `(x, y)` coords).
    /// Length must equal `provider.n_channels()`; mismatched lengths are
    /// ignored and the view falls back to single-channel display.
    /// Supply the shank id of every raw channel (`channel_shanks.npy` on
    /// Kilosort data) so the trace pane can offer per-shank common-median
    /// referencing. Length must equal `provider.n_channels()`.
    pub fn set_channel_shanks(&mut self, shanks: Vec<u32>) {
        self.channel_shanks = shanks;
        self.trace_cache = TraceCache::default();
    }

    pub fn set_channel_positions(&mut self, positions: Vec<[f32; 2]>) {
        self.channel_positions = positions;
    }

    fn dirty(&self) -> bool {
        let s = read(&self.session);
        (s.history_len(), s.redo_len()) != self.saved_state
    }

    /// Stop background jobs so the write lock is free at once.
    fn cancel_jobs(&mut self) {
        self.suggest_cache.cancel();
        self.quality_cache.cancel();
        self.depth_memo.cancel();
    }

    fn jobs_pending(&self) -> bool {
        self.suggest_cache.pending()
            || self.quality_cache.pending()
            || self.depth_memo.pending()
            || self.phy_export_job.is_some()
    }

    /// Dispatch a curation command and do the bookkeeping every mutation
    /// needs: status toast and moving the selection to the result (merge
    /// target / both halves of a split) so the next view shows what just
    /// changed. Returns `true` on success.
    fn commit(&mut self, cmd: CurationCommand) -> bool {
        let follow = match &cmd {
            CurationCommand::Merge { target, .. } => Some(vec![*target]),
            // `ClusterIndex::split` always allocates the new id at the tail.
            CurationCommand::Split { cluster, .. } => {
                Some(vec![*cluster, ClusterId(read(&self.session).n_clusters())])
            }
            _ => None,
        };
        self.commit_with_follow(cmd, follow)
    }

    /// Every view cache is keyed on `Session::version()`, so nothing needs
    /// invalidating by hand here: the next frame rebuilds what changed.
    fn commit_with_follow(&mut self, cmd: CurationCommand, follow: Option<Vec<ClusterId>>) -> bool {
        let desc = describe_command(&cmd);
        self.cancel_jobs();
        // Every GUI edit is a human decision: it clears the touched clusters'
        // `review:*` tags (from `sorrel auto`), so the review queue drains.
        let cmd = read(&self.session).with_human_review(cmd);
        if let Err(e) = write(&self.session).dispatch(cmd) {
            self.error = Some(format!("{desc} failed: {e}"));
            return false;
        }
        if let Some(ids) = follow {
            if let Some((&first, rest)) = ids.split_first() {
                self.selection.replace(first);
                for &c in rest {
                    self.selection.toggle(c);
                }
            }
        }
        self.status = toast(desc);
        true
    }

    /// Turn lasso hits into one journal entry (a batch when several
    /// clusters were lassoed) and select source + new halves afterwards.
    fn apply_lasso_splits(&mut self, splits: Vec<(ClusterId, Vec<u32>)>) {
        let n = read(&self.session).n_clusters();
        let mut cmds = Vec::new();
        let mut follow = Vec::new();
        for (cluster, spike_idx) in splits.into_iter().filter(|(_, s)| !s.is_empty()) {
            follow.push(cluster);
            follow.push(ClusterId(n + cmds.len() as u32));
            cmds.push(CurationCommand::Split {
                cluster,
                spike_idx,
                new_cluster: ClusterId(0), // schema field; ClusterIndex auto-allocates
            });
        }
        let cmd = match cmds.len() {
            0 => return,
            1 => cmds.remove(0),
            _ => match CurationCommand::batch(cmds) {
                Some(b) => b,
                None => return,
            },
        };
        self.commit_with_follow(cmd, Some(follow));
    }

    fn save(&mut self) {
        match self.save_dir.as_deref().map(Path::to_path_buf) {
            Some(dir) => match self.save_to(&dir) {
                Ok(()) => {
                    self.status = toast(format!("saved to {}", dir.display()));
                    self.error = None;
                    let s = read(&self.session);
                    self.saved_state = (s.history_len(), s.redo_len());
                    drop(s);
                    self.save_prefs();
                }
                Err(e) => {
                    self.error = Some(format!("save failed: {e}"));
                }
            },
            None => {
                self.error = Some("save target not configured (no kilosort directory)".into());
            }
        }
    }

    /// Render the confirmation dialog if an action is staged. The modal
    /// blocks pointer input behind it; Enter confirms, Esc cancels.
    fn show_confirm_dialog(&mut self, ctx: &egui::Context) {
        let Some(pending) = self.pending_confirm.as_ref() else {
            return;
        };
        #[derive(Clone, Copy, PartialEq)]
        enum Decision {
            Cancel,
            Confirm,
            SaveThenConfirm,
        }
        let title = pending.title.clone();
        let body = pending.body.clone();
        let is_quit = matches!(pending.action, ConfirmAction::Quit);
        let accent = egui::Color32::from_rgb(80, 130, 60);
        let mut decision: Option<Decision> = None;
        let resp = egui::Modal::new(egui::Id::new("sorrel-confirm")).show(ctx, |ui| {
            ui.set_width(380.0);
            ui.heading(&title);
            ui.label(&body);
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Cancel").on_hover_text("Esc").clicked() {
                    decision = Some(Decision::Cancel);
                }
                if is_quit {
                    if ui.button("Quit without saving").clicked() {
                        decision = Some(Decision::Confirm);
                    }
                    if ui
                        .add(egui::Button::new("Save & quit").fill(accent))
                        .on_hover_text("Enter")
                        .clicked()
                    {
                        decision = Some(Decision::SaveThenConfirm);
                    }
                } else if ui
                    .add(egui::Button::new("Confirm").fill(accent))
                    .on_hover_text("Enter")
                    .clicked()
                {
                    decision = Some(Decision::Confirm);
                }
            });
            if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                decision = Some(if is_quit {
                    Decision::SaveThenConfirm
                } else {
                    Decision::Confirm
                });
            }
        });
        if decision.is_none() && resp.should_close() {
            decision = Some(Decision::Cancel);
        }
        let Some(decision) = decision else {
            return;
        };
        let pending = self.pending_confirm.take().expect("checked above");
        match (decision, pending.action) {
            (Decision::Cancel, _) => {}
            (Decision::SaveThenConfirm, ConfirmAction::Quit) => {
                self.save();
                if self.error.is_none() {
                    self.close_approved = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            (Decision::Confirm, ConfirmAction::Quit) => {
                self.close_approved = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            (_, ConfirmAction::Dispatch(cmd)) => {
                if self.commit(cmd) {
                    self.status = toast(pending.on_success);
                }
            }
        }
    }

    fn show_help_window(&mut self, ctx: &egui::Context) {
        if !self.show_help {
            return;
        }
        let rows: &[(&str, &str)] = &[
            (
                "G / M / N / U",
                "label selection good / MUA / noise / unsorted",
            ),
            (
                "J or Down · K or Up",
                "next / previous cluster (table order)",
            ),
            (
                "H or Left · L or Right",
                "pan the trace window back / forward",
            ),
            (
                "Ctrl/Cmd-G",
                "merge the selection into the first-selected cluster",
            ),
            ("Ctrl/Cmd-Z", "undo"),
            ("Ctrl/Cmd-Shift-Z · Ctrl/Cmd-Y", "redo"),
            ("Ctrl/Cmd-S", "save spike_clusters.npy + cluster_group.tsv"),
            ("Esc", "cancel a dialog or an in-progress lasso"),
            ("? · F1", "toggle this help"),
            ("click", "select one cluster"),
            ("Ctrl/Cmd-click", "add or remove a cluster"),
            ("Shift-click", "select the range of visible rows"),
            ("drag in Features", "lasso spikes, then split them off"),
            ("click in Raster", "seek the trace to that time"),
            (
                "click in Similar",
                "compare with that cluster (Ctrl/Cmd adds it)",
            ),
            (
                "drag / wheel in Trace",
                "pan · Ctrl-wheel zooms around the pointer",
            ),
            (
                "drag a tab",
                "rearrange views · × closes · View menu reopens",
            ),
        ];
        egui::Window::new("Keyboard & mouse")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .open(&mut self.show_help)
            .show(ctx, |ui| {
                egui::Grid::new("sorrel-help")
                    .num_columns(2)
                    .spacing([16.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        for (key, what) in rows {
                            ui.strong(*key);
                            ui.label(*what);
                            ui.end_row();
                        }
                    });
            });
    }

    fn dispatch_intent(&mut self, intent: Intent) {
        let n = read(&self.session).n_clusters();
        match intent {
            Intent::SelectCluster(c) => {
                if c.0 < n {
                    self.selection.replace(c);
                }
            }
            Intent::ToggleCluster(c) => {
                if c.0 < n {
                    self.selection.toggle(c);
                }
            }
            Intent::ExtendCluster(c) => {
                if c.0 < n {
                    self.selection.extend_to(c, &self.table_state.visible);
                }
            }
            Intent::Relabel(op) => {
                // Apply to every cluster in the active selection as one
                // journal entry, so a single Undo reverts the whole relabel.
                let mut cmds: Vec<CurationCommand> = self
                    .selection
                    .iter()
                    .map(|&c| CurationCommand::Relabel { cluster: c, op })
                    .collect();
                let cmd = match cmds.len() {
                    0 => return,
                    1 => cmds.remove(0),
                    _ => match CurationCommand::batch(cmds) {
                        Some(b) => b,
                        None => return,
                    },
                };
                self.commit(cmd);
            }
            Intent::MergeSelection => {
                let sel = self.selection.as_slice();
                if sel.len() < 2 {
                    self.status = toast("select two or more clusters to merge");
                    return;
                }
                let cmd = CurationCommand::Merge {
                    sources: sel[1..].to_vec(),
                    target: sel[0],
                };
                self.commit(cmd);
            }
            Intent::Undo => {
                // Peek the about-to-be-undone command first so the status
                // toast can name it ("undid: relabel cluster 12 → MUA")
                // rather than a generic "undo".
                let desc = read(&self.session)
                    .history_commands()
                    .last()
                    .map(describe_command);
                self.cancel_jobs();
                if let Err(e) = write(&self.session).dispatch(CurationCommand::Undo) {
                    self.error = Some(format!("journal write failed: {e}"));
                } else if let Some(d) = desc {
                    self.status = toast(format!("undid: {d}"));
                }
            }
            Intent::Redo => {
                self.cancel_jobs();
                if let Err(e) = write(&self.session).dispatch(CurationCommand::Redo) {
                    self.error = Some(format!("journal write failed: {e}"));
                } else if let Some(d) = read(&self.session)
                    .history_commands()
                    .last()
                    .map(describe_command)
                {
                    self.status = toast(format!("redid: {d}"));
                }
            }
            Intent::Save => self.save(),
            Intent::NextCluster => {
                self.selection.bump(1, &self.table_state.visible);
            }
            Intent::PrevCluster => {
                self.selection.bump(-1, &self.table_state.visible);
            }
            Intent::PageBack => {
                self.window_start = self.window_start.saturating_sub(self.window_len as u64);
            }
            Intent::PageForward => {
                let max = read(&self.session)
                    .provider
                    .n_samples()
                    .0
                    .saturating_sub(self.window_len as u64);
                self.window_start = (self.window_start + self.window_len as u64).min(max);
            }
            Intent::Cancel => {
                if self.pending_confirm.take().is_some() {
                    // Dialog dismissed.
                } else if self.show_help {
                    self.show_help = false;
                } else if self.feature_state.lasso_active() {
                    self.feature_state.cancel_lasso();
                    self.status = toast("lasso cancelled");
                }
            }
            Intent::ToggleHelp => self.show_help = !self.show_help,
        }
    }

    fn top_bar(&mut self, ui: &mut egui::Ui, dirty: bool) {
        // Everything the bar shows is read up front so no lock is held
        // while buttons dispatch commands.
        let (info, undo, redo) = {
            let s = read(&self.session);
            (
                format!(
                    "{} clusters · {} channels · {:.1} kHz",
                    s.n_clusters(),
                    s.provider.n_channels(),
                    s.provider.sample_rate() / 1000.0
                ),
                s.history_commands().last().map(describe_command),
                s.redo_commands().last().map(describe_command),
            )
        };
        ui.horizontal(|ui| {
            ui.heading(if dirty { "Sorrel •" } else { "Sorrel" })
                .on_hover_text(if dirty {
                    "Unsaved curation — Cmd/Ctrl-S writes phy artefacts."
                } else {
                    "All curation saved."
                });
            ui.separator();
            ui.label(info);
            ui.separator();
            ui.label(format!("{} selected", self.selection.len()));
            ui.separator();

            let undo_tip = undo
                .as_ref()
                .map(|d| format!("Undo {d} (Cmd/Ctrl-Z)"))
                .unwrap_or_else(|| "Nothing to undo".into());
            if ui
                .add_enabled(undo.is_some(), egui::Button::new("Undo"))
                .on_hover_text(undo_tip)
                .on_disabled_hover_text("Nothing to undo")
                .clicked()
            {
                self.dispatch_intent(Intent::Undo);
            }
            let redo_tip = redo
                .as_ref()
                .map(|d| format!("Redo {d} (Cmd/Ctrl-Shift-Z)"))
                .unwrap_or_else(|| "Nothing to redo".into());
            if ui
                .add_enabled(redo.is_some(), egui::Button::new("Redo"))
                .on_hover_text(redo_tip)
                .on_disabled_hover_text("Nothing to redo")
                .clicked()
            {
                self.dispatch_intent(Intent::Redo);
            }
            let n_sel = self.selection.len();
            let target = self
                .selection
                .as_slice()
                .first()
                .map(|c| c.0.to_string())
                .unwrap_or_default();
            if ui
                .add_enabled(n_sel >= 2, egui::Button::new(format!("Merge {n_sel}")))
                .on_hover_text(format!(
                    "Merge the selection into cluster {target} (Cmd/Ctrl-G)"
                ))
                .on_disabled_hover_text("Select two or more clusters to merge (Cmd/Ctrl-G)")
                .clicked()
            {
                self.dispatch_intent(Intent::MergeSelection);
            }

            if let Some(done) = self.phy_export_job.as_ref().and_then(Job::try_take) {
                self.phy_export_job = None;
                match done {
                    Ok(dir) => {
                        self.status = toast(format!("exported phy to {}", dir.display()));
                        self.error = None;
                    }
                    Err(e) => self.error = Some(format!("phy export failed: {e:#}")),
                }
            }
            if let Some(err) = self.error.clone() {
                ui.separator();
                ui.colored_label(ui.visuals().error_fg_color, &err);
                if ui.small_button("×").on_hover_text("dismiss").clicked() {
                    self.error = None;
                }
            } else if let Some((status, _)) = &self.status {
                ui.separator();
                ui.weak(status);
            }
            if self.jobs_pending() {
                ui.separator();
                ui.spinner();
                ui.weak("working…")
                    .on_hover_text("Background metrics are computing; curation stays responsive.");
            }

            // Stable ids for the right-hand widgets: the left side's widget
            // count varies (spinner, error, status), which would otherwise
            // reshuffle auto ids between passes.
            let right = egui::Layout::right_to_left(egui::Align::Center);
            ui.with_layout(right, |ui| {
                ui.push_id("topbar-right", |ui| {
                    if ui
                        .small_button("?")
                        .on_hover_text("Keyboard & mouse help (? / F1)")
                        .clicked()
                    {
                        self.show_help = !self.show_help;
                    }
                    ui.menu_button("View", |ui| {
                        let closed = layout::missing_panes(&self.tree);
                        if closed.is_empty() {
                            ui.weak("all views are open");
                        }
                        for pane in closed {
                            if ui.button(pane.title()).clicked() {
                                layout::add_pane(&mut self.tree, pane, None);
                                ui.close();
                            }
                        }
                        ui.separator();
                        if ui
                            .button("Reset layout")
                            .on_hover_text("Restore the default three-group layout")
                            .clicked()
                        {
                            self.tree = layout::default_tree();
                            ui.close();
                        }
                    })
                    .response
                    .on_hover_text("Drag tabs to rearrange; close with ×; reopen here");
                    if ui
                        .small_button("Export QC")
                        .on_hover_text(
                            "Write cluster_qc.tsv + cluster_qc.json into the save directory \
                         (SpikeInterface-compatible).",
                        )
                        .clicked()
                    {
                        match self.save_dir.as_deref() {
                            Some(dir) => match export_qc(&read(&self.session), dir) {
                                Ok(n) => {
                                    self.status = toast(format!(
                                        "exported QC for {n} clusters to {}",
                                        dir.display()
                                    ));
                                    self.error = None;
                                }
                                Err(e) => {
                                    self.error = Some(format!("QC export failed: {e}"));
                                }
                            },
                            None => {
                                self.error = Some(
                                    "save target not configured (no kilosort directory)".into(),
                                );
                            }
                        }
                    }
                    if let Some(export) = self.phy_export.clone() {
                        if ui
                            .add_enabled(
                                self.phy_export_job.is_none(),
                                egui::Button::new("Export phy").small(),
                            )
                            .on_hover_text(
                                "Write the curated data as a complete phy directory next to \
                                 the dataset (<dataset>_phy); params.py points at the raw file.",
                            )
                            .on_disabled_hover_text("Export running…")
                            .clicked()
                        {
                            let ctx = ui.ctx().clone();
                            self.phy_export_job = Some(Job::spawn(
                                (),
                                Arc::clone(&self.session),
                                move || ctx.request_repaint(),
                                move |s: &Session<P>, _| export(s),
                            ));
                        }
                    }
                    // Show the configured save target so users know where
                    // Cmd/Ctrl-S and Export QC will write.
                    if let Some(dir) = self.save_dir.as_deref() {
                        ui.label(format!("saves to {}", short_path(dir)))
                            .on_hover_text(format!(
                                "Save target: {}\nWrites spike_clusters.npy, cluster_group.tsv \
                             (Cmd/Ctrl-S) and cluster_qc.tsv/json (Export QC).",
                                dir.display()
                            ));
                    } else {
                        ui.colored_label(ui.visuals().warn_fg_color, "no save target")
                            .on_hover_text(
                                "Open sorrel with a kilosort directory to enable saving.",
                            );
                    }
                })
            });
        });
    }

    fn apply_view_action(&mut self, action: ViewAction) {
        match action {
            ViewAction::Seek(sample) => {
                let max = read(&self.session)
                    .provider
                    .n_samples()
                    .0
                    .saturating_sub(self.window_len as u64);
                self.window_start = sample.min(max);
            }
            ViewAction::Split(splits) => self.apply_lasso_splits(splits),
            ViewAction::Suggest(act) => self.apply_suggest_action(act),
            ViewAction::Compare { cluster, add } => {
                if add {
                    self.selection.toggle(cluster);
                } else if let Some(&primary) = self.selection.as_slice().first() {
                    self.selection.replace(primary);
                    if cluster != primary {
                        self.selection.toggle(cluster);
                    }
                }
            }
            ViewAction::AddPane(pane, into) => layout::add_pane(&mut self.tree, pane, Some(into)),
        }
    }

    fn apply_suggest_action(&mut self, act: SuggestAction) {
        match act {
            SuggestAction::Select(clusters) => {
                if let Some(&first) = clusters.first() {
                    self.selection.replace(first);
                    for &c in &clusters[1..] {
                        self.selection.toggle(c);
                    }
                }
            }
            SuggestAction::ApplyMerge { sources, target } => {
                self.commit(CurationCommand::Merge { sources, target });
            }
            SuggestAction::ApplyGmmSplit { cluster, spike_idx } => {
                self.commit(CurationCommand::Split {
                    cluster,
                    spike_idx,
                    new_cluster: ClusterId(0),
                });
            }
            SuggestAction::ApplyHighConfidenceMerges { threshold } => {
                // Bulk merges can collapse dozens of clusters in one click —
                // gate the action behind a confirm dialog.
                let children: Vec<CurationCommand> = self
                    .suggest_cache
                    .high_confidence_merges(threshold)
                    .into_iter()
                    .map(|(a, b)| CurationCommand::Merge {
                        sources: vec![a],
                        target: b,
                    })
                    .collect();
                if children.is_empty() {
                    self.status = toast("no merges met the threshold");
                    return;
                }
                let n = children.len();
                let Some(batch) = CurationCommand::batch(children) else {
                    self.error = Some("internal error: batch contained nested commands".into());
                    return;
                };
                self.pending_confirm = Some(PendingConfirm {
                    title: format!("Apply {n} merges?"),
                    body: format!(
                        "{n} cluster pair(s) at or above similarity threshold {threshold:.2} \
                         will be merged as a single batch. One Undo reverts the whole batch."
                    ),
                    action: ConfirmAction::Dispatch(batch),
                    on_success: format!("applied {n} merges (one Undo reverts all)"),
                });
            }
        }
    }

    /// Loading / load-error screens. Returns `true` when the normal UI
    /// must not run this frame (the loader still owns the session).
    fn show_loading(&mut self, ctx: &egui::Context, root_ui: &mut egui::Ui) -> bool {
        if let Some(rx) = &self.loading {
            match rx.try_recv() {
                Ok(Ok(())) => {
                    self.loading = None;
                    self.finish_loading();
                    return false;
                }
                Ok(Err(e)) => {
                    self.loading = None;
                    self.load_error = Some(e.to_string());
                }
                Err(TryRecvError::Disconnected) => {
                    self.loading = None;
                    self.load_error = Some("loader thread exited without a result".into());
                }
                Err(TryRecvError::Empty) => {
                    let name = self
                        .save_dir
                        .as_deref()
                        .and_then(Path::file_name)
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "recording".into());
                    egui::CentralPanel::default().show_inside(root_ui, |ui| {
                        ui.centered_and_justified(|ui| {
                            ui.vertical_centered(|ui| {
                                ui.add_space(ui.available_height() * 0.4);
                                ui.spinner();
                                ui.heading(format!("Loading {name}…"));
                                ui.weak("seeding amplitudes, features and templates; replaying the journal");
                            });
                        });
                    });
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                    return true;
                }
            }
        }
        if let Some(err) = self.load_error.clone() {
            egui::CentralPanel::default().show_inside(root_ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() * 0.3);
                    ui.heading("Could not open the recording");
                    ui.colored_label(ui.visuals().error_fg_color, err);
                    ui.add_space(8.0);
                    if ui.button("Quit").clicked() {
                        self.close_approved = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
            return true;
        }
        false
    }
}

impl<P: DataProvider + ApplyPhyLabel> eframe::App for SorrelApp<P> {
    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = root_ui.ctx().clone();
        if self.show_loading(&ctx, root_ui) {
            return;
        }

        // Input. Suppress single-letter intents while a text widget has focus
        // so typing in a filter box doesn't relabel selected clusters; global
        // shortcuts (Cmd/Ctrl-S/Z/Shift-Z) still fire. While a confirm
        // dialog is open only Esc gets through — the modal owns the keys.
        let text_input_active = ctx.egui_wants_keyboard_input();
        let input_intents: Vec<Intent> = ctx.input(|i| {
            i.events
                .iter()
                .filter_map(|ev| match ev {
                    egui::Event::Key {
                        key,
                        pressed: true,
                        modifiers,
                        ..
                    } => from_key_in_context(*key, *modifiers, text_input_active),
                    _ => None,
                })
                .collect()
        });
        let dialog_open = self.pending_confirm.is_some();
        for it in input_intents {
            if dialog_open && !matches!(it, Intent::Cancel) {
                continue;
            }
            self.dispatch_intent(it);
        }

        let dirty = self.dirty();

        // Unsaved-changes guard: intercept the window close once and ask.
        if ctx.input(|i| i.viewport().close_requested()) && dirty && !self.close_approved {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.pending_confirm = Some(PendingConfirm {
                title: "Quit with unsaved curation?".into(),
                body: "Edits are journaled, but spike_clusters.npy and cluster_group.tsv \
                       have not been written since the last save."
                    .into(),
                action: ConfirmAction::Quit,
                on_success: String::new(),
            });
        }

        // OS window title: dataset name + dirty marker.
        let dataset = self
            .save_dir
            .as_deref()
            .and_then(Path::file_name)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let title = match (dataset.is_empty(), dirty) {
            (true, false) => "Sorrel".to_string(),
            (true, true) => "Sorrel •".to_string(),
            (false, false) => format!("{dataset} — Sorrel"),
            (false, true) => format!("{dataset} • — Sorrel"),
        };
        if title != self.last_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.last_title = title;
        }

        self.show_confirm_dialog(&ctx);
        self.show_help_window(&ctx);

        // Status auto-fades after 5 s so the top bar doesn't pin a stale
        // "saved" pill from twenty minutes ago.
        const STATUS_TTL_SECS: u64 = 5;
        if let Some((_, when)) = &self.status {
            let elapsed = when.elapsed().as_secs();
            if elapsed >= STATUS_TTL_SECS {
                self.status = None;
            } else {
                // Wake the UI when the toast should fade so users on idle
                // input still see it disappear.
                ctx.request_repaint_after(std::time::Duration::from_secs(
                    STATUS_TTL_SECS - elapsed,
                ));
            }
        }

        egui::Panel::top("top").show_inside(root_ui, |ui| self.top_bar(ui, dirty));

        egui::Panel::left("clusters")
            .resizable(true)
            .default_size(260.0)
            .show_inside(root_ui, |ui| {
                ui.heading("Clusters");
                let session = read(&self.session);
                cluster_table(
                    ui,
                    &session,
                    &mut self.selection,
                    &mut self.table_state,
                    self.label_str,
                );
            });

        egui::CentralPanel::default().show_inside(root_ui, |ui| {
            // Every view is a tile; views only read the session and queue
            // side effects as actions, applied once the read lock is gone.
            let selected: Vec<ClusterId> = self.selection.iter().copied().collect();
            let closed = layout::missing_panes(&self.tree);
            let jobs = JobCtx {
                session: Arc::clone(&self.session),
                egui: ctx.clone(),
            };
            let actions = {
                let session = read(&self.session);
                // Depths are backend-specific and walk every spike, so they
                // come from a worker and are memoised per curation version.
                let depths_arc = self.depth_fn.and_then(|f| {
                    self.depth_memo
                        .ensure(
                            session.version(),
                            &self.session,
                            jobs.repaint(),
                            move |s, _| Arc::new(f(s)),
                        )
                        .cloned()
                });
                let depths: Option<&SpikeDepths> = depths_arc.as_deref().and_then(|d| d.as_ref());
                // Waveforms follow the trace pane's HP/CMR toggles.
                self.waveform_cache.set_preproc(
                    self.trace_cfg,
                    if self.per_shank_cmr {
                        &self.channel_shanks
                    } else {
                        &[]
                    },
                );
                let mut behavior = ViewBehavior {
                    session: &session,
                    selection: &selected,
                    label_str: self.label_str,
                    channel_positions: &self.channel_positions,
                    waveform_cache: &mut self.waveform_cache,
                    feature_state: &mut self.feature_state,
                    raster_cache: &mut self.raster_cache,
                    suggest_cache: &mut self.suggest_cache,
                    ccg_cache: &mut self.ccg_cache,
                    quality_cache: &mut self.quality_cache,
                    jobs: &jobs,
                    events: self.events.as_deref(),
                    depths,
                    psth_state: &mut self.psth_state,
                    psth_cache: &mut self.psth_cache,
                    trace: TraceControls {
                        window_start: &mut self.window_start,
                        window_len: &mut self.window_len,
                        target_points: self.target_points,
                        cfg: &mut self.trace_cfg,
                        gpu: self.gpu_preproc.as_ref(),
                        cache: &mut self.trace_cache,
                        shanks: &self.channel_shanks,
                        per_shank: &mut self.per_shank_cmr,
                    },
                    closed,
                    actions: Vec::new(),
                };
                self.tree.ui(&mut behavior, ui);
                behavior.actions
            };
            for action in actions {
                self.apply_view_action(action);
            }
        });
    }

    fn on_exit(&mut self) {
        self.cancel_jobs();
        self.save_prefs();
        if self.loading.is_none() {
            if let Err(err) = read(&self.session).invalidate_dataset_caches() {
                log::warn!("cache invalidation on exit failed: {err}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Headless tests for the input → state path of `SorrelApp`. These don't
    //! render anything (we don't have a wgpu device in CI) but exercise
    //! `dispatch_intent`, save/dirty bookkeeping, and the toast-message
    //! plumbing — i.e. all the behaviour a user observes that isn't pixels.
    use super::*;
    use sorrel_data::session::ApplyPhyLabel;
    use sorrel_data::{Journal, PhyLabelOp, Session};
    use sorrel_io::{ClusterId, DataProvider, SampleIndex, TraceSamples, TraceSlice};

    /// Minimal in-memory `DataProvider` — just enough metadata for the app's
    /// state machine. Trace, waveform, and template paths are exercised by
    /// `sorrel-data`/`sorrel-render` tests; here we only care about the
    /// session/intent loop.
    struct StubProvider {
        spikes: Vec<Vec<SampleIndex>>,
    }

    impl DataProvider for StubProvider {
        type Label = u8;
        fn sample_rate(&self) -> f32 {
            30_000.0
        }
        fn n_channels(&self) -> u32 {
            1
        }
        fn n_samples(&self) -> SampleIndex {
            SampleIndex(1_000_000)
        }
        fn n_clusters(&self) -> u32 {
            self.spikes.len() as u32
        }
        fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
            self.spikes.get(c.idx()).map(Vec::as_slice).unwrap_or(&[])
        }
        fn trace(&self, _: SampleIndex, _: u32) -> TraceSlice<'_> {
            TraceSlice {
                start: SampleIndex(0),
                n_channels: 0,
                samples: TraceSamples::I16(&[]),
            }
        }
        fn initial_labels(&self) -> Vec<u8> {
            vec![0u8; self.spikes.len()]
        }
    }

    impl ApplyPhyLabel for StubProvider {
        fn label_from_op(op: PhyLabelOp) -> Self::Label {
            match op {
                PhyLabelOp::SetUnsorted => 0,
                PhyLabelOp::SetGood => 1,
                PhyLabelOp::SetMua => 2,
                PhyLabelOp::SetNoise => 3,
            }
        }
    }

    fn label_str(l: &u8) -> &'static str {
        match l {
            0 => "unsorted",
            1 => "good",
            2 => "mua",
            _ => "noise",
        }
    }

    fn stub_session(dir: &Path) -> Session<StubProvider> {
        let provider = StubProvider {
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(30)],
                vec![SampleIndex(40), SampleIndex(60)],
                vec![SampleIndex(80)],
            ],
        };
        let journal = Journal::open_or_create(&dir.join("j.sqlite"), 0xCAFE, 3).unwrap();
        Session::new(provider, journal)
    }

    fn make_app() -> (SorrelApp<StubProvider>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let mut app = SorrelApp::new(stub_session(dir.path()), label_str);
        // The table normally fills this every frame; headless tests use
        // the identity order.
        app.table_state.visible = (0..3).map(ClusterId).collect();
        (app, dir)
    }

    fn label(app: &SorrelApp<StubProvider>, c: u32) -> Option<u8> {
        read(&app.session).label(ClusterId(c))
    }

    fn history_len(app: &SorrelApp<StubProvider>) -> usize {
        read(&app.session).history_len()
    }

    #[test]
    fn relabel_intent_mutates_session_and_marks_dirty() {
        let (mut app, _d) = make_app();
        assert!(!app.dirty(), "fresh session should be clean");

        app.dispatch_intent(Intent::Relabel(PhyLabelOp::SetGood));

        assert_eq!(label(&app, 0), Some(1));
        assert!(
            app.dirty(),
            "post-relabel state should diverge from saved snapshot"
        );
        assert!(app.status.is_some(), "relabel should toast");
    }

    #[test]
    fn multi_relabel_is_one_undo_step() {
        let (mut app, _d) = make_app();
        app.selection.toggle(ClusterId(1));
        app.selection.toggle(ClusterId(2));
        app.dispatch_intent(Intent::Relabel(PhyLabelOp::SetMua));
        assert_eq!(history_len(&app), 1, "batched into one entry");
        assert!(app
            .status
            .as_ref()
            .unwrap()
            .0
            .starts_with("relabel 3 clusters"));
        app.dispatch_intent(Intent::Undo);
        assert_eq!(label(&app, 2), Some(0));
    }

    #[test]
    fn merge_selection_merges_into_first_and_selects_target() {
        let (mut app, _d) = make_app();
        app.dispatch_intent(Intent::MergeSelection);
        assert_eq!(history_len(&app), 0, "single selection: no-op");
        app.selection.replace(ClusterId(2));
        app.selection.toggle(ClusterId(0));
        app.dispatch_intent(Intent::MergeSelection);
        assert_eq!(history_len(&app), 1);
        let s = read(&app.session);
        assert_eq!(s.spike_times(ClusterId(2)).len(), 3);
        assert!(s.spike_times(ClusterId(0)).is_empty());
        drop(s);
        assert_eq!(app.selection.as_slice(), &[ClusterId(2)]);
    }

    #[test]
    fn save_without_save_dir_records_an_error() {
        let (mut app, _d) = make_app();
        app.dispatch_intent(Intent::Save);
        assert!(app.error.is_some(), "save without save_dir should error");
        assert!(app.status.is_none());
    }

    #[test]
    fn save_with_save_dir_clears_dirty_sets_status_and_writes_prefs() {
        let (mut app, dir) = make_app();
        app.set_save_dir(dir.path().to_path_buf(), false);
        app.dispatch_intent(Intent::Relabel(PhyLabelOp::SetMua));
        assert!(app.dirty());

        app.dispatch_intent(Intent::Save);

        assert!(app.error.is_none(), "save should succeed: {:?}", app.error);
        assert!(app.status.is_some(), "save should produce a status toast");
        assert!(
            !app.dirty(),
            "saved_state should snap to current after a successful save"
        );
        let prefs = UiPrefs::load(dir.path());
        assert!(prefs.tree.is_some(), "prefs written next to the dataset");
        assert_eq!(prefs.anchor, Some(0));
    }

    #[test]
    fn prefs_are_applied_when_the_save_dir_is_set() {
        let dir = tempfile::tempdir().unwrap();
        UiPrefs {
            anchor: Some(2),
            hide_noise: true,
            window_ms: Some(50.0),
            ..Default::default()
        }
        .save(dir.path())
        .unwrap();
        let mut app = SorrelApp::new(stub_session(dir.path()), label_str);
        app.set_save_dir(dir.path().to_path_buf(), false);
        assert_eq!(app.selection.as_slice(), &[ClusterId(2)]);
        assert!(app.table_state.hide_noise);
        assert_eq!(app.window_len, 1500, "50 ms at 30 kHz");
    }

    #[test]
    fn loader_runs_off_thread_and_then_unlocks_the_session() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = SorrelApp::with_loader(stub_session(dir.path()), label_str, |s| {
            s.dispatch(CurationCommand::Relabel {
                cluster: ClusterId(1),
                op: PhyLabelOp::SetGood,
            })
        });
        assert!(app.loading.is_some());
        let rx = app.loading.take().unwrap();
        rx.recv_timeout(std::time::Duration::from_secs(5))
            .expect("loader reports")
            .expect("loader succeeds");
        app.finish_loading();
        assert_eq!(label(&app, 1), Some(1));
        assert!(!app.dirty(), "replayed state counts as the saved baseline");
        assert_eq!(app.selection.as_slice(), &[ClusterId(0)]);
    }

    #[test]
    fn undo_status_describes_the_reverted_command() {
        let (mut app, _d) = make_app();
        app.dispatch_intent(Intent::Relabel(PhyLabelOp::SetNoise));
        app.dispatch_intent(Intent::Undo);
        let (msg, _) = app.status.as_ref().expect("undo should produce a status");
        assert!(
            msg.starts_with("undid: relabel cluster 0"),
            "expected descriptive undo, got {msg:?}"
        );
        assert!(msg.contains("Noise"));
    }

    #[test]
    fn navigation_intents_walk_the_table_order() {
        let (mut app, _d) = make_app();
        // Table sorted descending by id, say.
        app.table_state.visible = vec![ClusterId(2), ClusterId(1), ClusterId(0)];
        assert!(app.selection.iter().any(|c| c.0 == 0));
        app.dispatch_intent(Intent::NextCluster);
        // Already on the last visible row: clamps.
        assert_eq!(app.selection.as_slice(), &[ClusterId(0)]);
        app.dispatch_intent(Intent::PrevCluster);
        assert_eq!(app.selection.as_slice(), &[ClusterId(1)]);
        app.dispatch_intent(Intent::PrevCluster);
        app.dispatch_intent(Intent::PrevCluster);
        assert_eq!(app.selection.as_slice(), &[ClusterId(2)]);
    }

    #[test]
    fn pan_intents_advance_window_start_and_clamp_at_end() {
        let (mut app, _d) = make_app();
        let len = app.window_len as u64;
        app.dispatch_intent(Intent::PageForward);
        assert_eq!(app.window_start, len);
        app.dispatch_intent(Intent::PageBack);
        assert_eq!(app.window_start, 0);
        // PageBack at zero should saturate, not underflow.
        app.dispatch_intent(Intent::PageBack);
        assert_eq!(app.window_start, 0);
    }

    #[test]
    fn pending_confirm_does_not_mutate_until_accepted_and_cancel_drops_it() {
        // Stage a batch merge directly (bypassing the Suggest tab UI which
        // requires a real frame). The session must stay clean while the
        // confirm is pending; Esc drops it; accepting dispatches.
        let (mut app, _d) = make_app();
        let baseline = history_len(&app);
        let stage = |app: &mut SorrelApp<StubProvider>| {
            let batch = CurationCommand::batch(vec![CurationCommand::Merge {
                sources: vec![ClusterId(0)],
                target: ClusterId(2),
            }])
            .unwrap();
            app.pending_confirm = Some(PendingConfirm {
                title: "Apply 1 merges?".into(),
                body: "test".into(),
                action: ConfirmAction::Dispatch(batch),
                on_success: "applied 1 merges".into(),
            });
        };
        stage(&mut app);
        assert_eq!(
            history_len(&app),
            baseline,
            "session must not change while a confirm is pending"
        );
        app.dispatch_intent(Intent::Cancel);
        assert!(app.pending_confirm.is_none());
        assert_eq!(history_len(&app), baseline);

        // Simulate the "Confirm" branch of show_confirm_dialog. We can't
        // drive egui without a real Context here, so dispatch the staged
        // command directly the same way the dialog handler does.
        stage(&mut app);
        let pending = app.pending_confirm.take().unwrap();
        let ConfirmAction::Dispatch(cmd) = pending.action else {
            panic!("expected a dispatch action");
        };
        assert!(app.commit(cmd));
        assert!(history_len(&app) > baseline);
    }

    #[test]
    fn short_path_keeps_recognisable_tail() {
        use std::path::PathBuf;
        let p = PathBuf::from("/very/long/nested/recordings/run42/ks4");
        let s = short_path(&p);
        assert!(s.contains("run42"));
        assert!(s.contains("ks4"));
        // Short paths are returned as-is.
        let s2 = short_path(&PathBuf::from("/a/b"));
        assert_eq!(s2, "/a/b");
    }
}

use crate::amplitude_view::AmplitudeViewState;
use crate::cluster_view::{cluster_table, ClusterRows, ClusterTableState, RowInputs, TableCtx};
use crate::feature_view::FeatureViewState;
use crate::intent::{from_key_in_context, Intent, MoveTarget};
use crate::jobs::{read, write, Job, JobCtx};
use crate::layout::{self, Pane, TraceControls, ViewAction, ViewBehavior};
use crate::prefs::UiPrefs;
use crate::selection::SelectionSet;
use crate::similarity_view::{SimilarityPane, SimilarityState};
use crate::views::{
    CcgCache, QualityCache, RasterCache, SuggestAction, SuggestCache, WaveformCache,
};
use crate::widgets::TraceCache;
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
    /// Hover text of the "saved" status chip once a save succeeded.
    last_save: Option<String>,
    /// Optional pre-processing applied to the trace window before LTTB.
    trace_cfg: TracePreproc,
    /// Dockable view layout — every view is a tile the user can drag,
    /// split, close and reopen.
    tree: egui_tiles::Tree<Pane>,
    /// Probe geometry — set by the binary when the backend exposes it.
    /// Empty means the WaveformView falls back to single-channel layout.
    channel_positions: Vec<[f32; 2]>,
    /// Waveform/CCG/ISI parameters and hidden table columns (phy state_attrs).
    pub view_params: crate::prefs::ViewParams,
    /// Shank id per raw channel; empty when the backend has one shank or
    /// no geometry. Enables the per-shank CMR toggle in the trace pane.
    channel_shanks: Vec<u32>,
    per_shank_cmr: bool,
    waveform_cache: WaveformCache,
    table_state: ClusterTableState,
    /// Table cells shared by the Clusters panel and the Similar pane.
    cluster_rows: ClusterRows,
    /// Per-cluster quality metrics (phy `cluster_*.tsv` / `quality_metrics.csv`).
    cluster_metrics: sorrel_io::QualityMetrics,
    similarity: SimilarityState,
    feature_state: FeatureViewState,
    amplitude_state: AmplitudeViewState,
    cluster_scatter: crate::cluster_scatter::ClusterScatterState,
    template_features: crate::template_feature_view::TemplateFeatureState,
    /// Panes frozen on a selection (phy `toggle_auto_update`).
    frozen: std::collections::HashMap<Pane, Vec<ClusterId>>,
    screenshot_pane: Option<Pane>,
    template_grid: crate::template_grid::TemplateGridCache,
    trace_display: crate::widgets::TraceDisplay,
    /// phy `toggle_spike_reorder` (Ctrl+R): time axes use `spike_reorder.npy`.
    spike_reorder: bool,
    trace_image: crate::trace_image::TraceImageCache,
    spike_attrs: crate::spike_attr_view::SpikeAttrState,
    /// phy v2.1 merge mode: the selection from before entry while the
    /// staged clusters (the best part) collect similar candidates.
    merge_mode: Option<SelectionSet>,
    /// phy colour field for unselected clusters, and its colours cached
    /// per `(version, scheme, depths known)` with a generation.
    scheme: crate::theme::ColourScheme,
    scheme_colours: Option<SchemeColours>,
    scheme_gen: u64,
    /// Pane rects waiting for a viewport screenshot.
    pending_screenshots: Vec<(Pane, egui::Rect)>,
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
    /// phy snippet bar text while open (`:` or `l`).
    snippet: Option<String>,
    /// Ctrl+Q: close on the next frame through the unsaved-changes guard.
    quit_requested: bool,
    /// Last window title sent to the OS; only re-sent when it changes.
    last_title: String,
    /// AA-contrast visuals go in on the first frame (no ctx before then).
    visuals_installed: bool,
}

/// Scheme colours keyed by `(version, scheme, depths known)`, with the
/// generation the raster re-colours on.
type SchemeColours = (
    (u64, crate::theme::ColourScheme, bool),
    u64,
    Vec<Option<egui::Color32>>,
);

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
            last_save: None,
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
            view_params: Default::default(),
            channel_shanks: Vec::new(),
            per_shank_cmr: true,
            waveform_cache: WaveformCache::new(),
            table_state: ClusterTableState::default(),
            cluster_rows: ClusterRows::default(),
            cluster_metrics: Default::default(),
            similarity: SimilarityState::default(),
            feature_state: FeatureViewState::default(),
            amplitude_state: AmplitudeViewState::default(),
            cluster_scatter: Default::default(),
            template_features: Default::default(),
            frozen: Default::default(),
            screenshot_pane: None,
            template_grid: Default::default(),
            trace_display: Default::default(),
            spike_reorder: false,
            trace_image: Default::default(),
            spike_attrs: Default::default(),
            merge_mode: None,
            scheme: Default::default(),
            scheme_colours: None,
            scheme_gen: 0,
            pending_screenshots: Vec::new(),
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
            snippet: None,
            quit_requested: false,
            last_title: String::new(),
            visuals_installed: false,
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
        self.view_params = prefs.views;
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
            sort_by: Some(self.table_state.sort_by.clone()),
            sort_descending: self.table_state.sort_descending,
            hide_empty: self.table_state.hide_empty,
            hide_noise: self.table_state.hide_noise,
            window_ms: Some(self.window_len as f32 * 1000.0 / sr),
            hp: self.trace_cfg.hp().is_some(),
            cmr: self.trace_cfg.has_cmr(),
            anchor: self.selection.anchor().map(|c| c.0),
            views: self.view_params.clone(),
        };
        if let Err(e) = prefs.save(dir) {
            log::warn!("could not write UI prefs: {e}");
        }
    }

    /// The shared session (tests and embedders inspect curation state).
    pub fn session(&self) -> &Arc<RwLock<Session<P>>> {
        &self.session
    }

    /// Hand the app the wgpu device/queue from `eframe`'s render state so
    /// trace CMR + HP filter can run on the GPU. Without this the app
    /// transparently falls back to the CPU implementation.
    pub fn install_gpu_compute(
        &mut self,
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
    ) {
        if !sorrel_gpu::supports_compute(&device) {
            log::info!("GPU adapter has no compute support; trace preprocessing stays on the CPU");
            return;
        }
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

    /// Supply the dataset's per-cluster quality metrics (KS `cluster_*.tsv`,
    /// SpikeInterface `quality_metrics.csv`); each becomes a table column.
    pub fn set_cluster_metrics(&mut self, metrics: sorrel_io::QualityMetrics) {
        self.cluster_metrics = metrics;
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
        self.feature_state.cancel();
        self.amplitude_state.cancel();
        self.template_features.cancel();
        self.spike_attrs.cancel();
    }

    /// The similar list the Space wizard steps through: the Similar
    /// table's rows in display order (clusters most similar to the last
    /// best one), with noise and MUA skipped (phy v2.1
    /// `skip_noise_and_mua`, on by default).
    fn similar_order(&mut self) -> Vec<ClusterId> {
        let s = read(&self.session);
        self.similarity.ensure(&s, self.selection.best());
        let labels = s.labels();
        self.similarity
            .order()
            .into_iter()
            .filter(|c| {
                labels
                    .get(c.idx())
                    .is_none_or(|l| !matches!((self.label_str)(l), "noise" | "mua"))
            })
            .collect()
    }

    /// Write `image` to `<data>/.sorrel/screenshots/<pane>-<unix secs>.png`
    /// (phy keeps them in `~/.phy/screenshots`).
    fn save_screenshot(&self, pane: Pane, image: &egui::ColorImage) -> anyhow::Result<PathBuf> {
        let dir = self
            .save_dir
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("no dataset directory"))?
            .join(".sorrel/screenshots");
        std::fs::create_dir_all(&dir)?;
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let name = pane.title().to_lowercase().replace([' ', '·'], "_");
        let path = dir.join(format!("{name}-{secs}.png"));
        let [w, h] = image.size;
        let rgba: Vec<u8> = image
            .pixels
            .iter()
            .flat_map(|c| c.to_srgba_unmultiplied())
            .collect();
        image::RgbaImage::from_raw(w as u32, h as u32, rgba)
            .ok_or_else(|| anyhow::anyhow!("bad image size"))?
            .save(&path)?;
        Ok(path)
    }

    /// phy snippet mode: Enter runs the command, Esc closes the bar.
    fn snippet_bar(&mut self, ui: &mut egui::Ui) {
        let Some(text) = self.snippet.as_mut() else {
            return;
        };
        let resp = ui
            .horizontal(|ui| {
                ui.label(egui::RichText::new("snippet").weak());
                let resp = ui.add(
                    egui::TextEdit::singleline(text)
                        .desired_width(f32::INFINITY)
                        .font(egui::TextStyle::Monospace)
                        .hint_text(":c 3 7 · :g · :l group good · :tg 12.5"),
                );
                resp.request_focus();
                resp
            })
            .inner;
        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            let text = self.snippet.take().unwrap_or_default();
            match crate::snippets::parse(&text) {
                Ok(snippet) => self.run_snippet(snippet),
                Err(e) => self.status = toast(e),
            }
        }
    }

    fn run_snippet(&mut self, snippet: crate::snippets::Snippet) {
        use crate::snippets::Snippet;
        let n = read(&self.session).n_clusters();
        match snippet {
            Snippet::Merge => self.dispatch_intent(Intent::MergeSelection),
            Snippet::Split => self.dispatch_intent(Intent::Split),
            Snippet::Select(ids) => {
                let ids: Vec<ClusterId> = ids.into_iter().filter(|c| c.0 < n).collect();
                let Some((&first, rest)) = ids.split_first() else {
                    self.status = toast("no such cluster");
                    return;
                };
                self.selection.replace(first);
                for &c in rest {
                    self.selection.toggle(c);
                }
            }
            Snippet::Label { name, value } if name == "group" => {
                let op = match value.as_str() {
                    "good" => PhyLabelOp::SetGood,
                    "mua" => PhyLabelOp::SetMua,
                    "noise" => PhyLabelOp::SetNoise,
                    "unsorted" => PhyLabelOp::SetUnsorted,
                    other => {
                        self.status = toast(format!("unknown group {other}"));
                        return;
                    }
                };
                self.dispatch_intent(Intent::Move(MoveTarget::All, op));
            }
            Snippet::Label { name, value } => {
                // phy custom label on every selected cluster, one undo step.
                let cmds: Result<Vec<CurationCommand>, String> = {
                    let s = read(&self.session);
                    self.selection
                        .iter()
                        .map(|&c| s.set_label_command(c, &name, &value))
                        .collect()
                };
                let cmd = match cmds {
                    Ok(mut cmds) if cmds.len() == 1 => cmds.remove(0),
                    Ok(cmds) => match CurationCommand::batch(cmds) {
                        Some(b) => b,
                        None => return,
                    },
                    Err(e) => {
                        self.status = toast(e);
                        return;
                    }
                };
                self.commit(cmd);
            }
            Snippet::Filter(expr) => self.table_state.set_filter(&expr),
            Snippet::Sort(col) => {
                if !self.table_state.sort_by(&col, None) {
                    self.status = toast(format!("unknown column {col}"));
                }
            }
            Snippet::TraceGoto(t) => {
                let s = read(&self.session);
                let sample = (t * s.provider.sample_rate()) as u64;
                let max = s
                    .provider
                    .n_samples()
                    .0
                    .saturating_sub(self.window_len as u64);
                self.window_start = sample.min(max);
            }
            Snippet::KMeans(k) => self.feature_state.request_kmeans(k),
            Snippet::RateBins(n) => self.view_params.rate.n_bins = n,
            Snippet::RateBinSize(secs) => {
                let s = read(&self.session);
                let duration = s.provider.n_samples().0 as f32 / s.provider.sample_rate().max(1.0);
                self.view_params.rate.n_bins =
                    ((duration / secs.max(1e-3)).round() as usize).clamp(2, 5000);
            }
            Snippet::ColourField(name) => match crate::theme::ColourScheme::parse(&name) {
                Some(s) => self.scheme = s,
                None => {
                    self.status = toast(format!(
                        "unknown colour field {name} (none, cluster, group, depth, fr)"
                    ))
                }
            },
            // ponytail: one sequential colormap (viridis); phy also offers
            // diverging and rainbow. Add them if curators ask.
            Snippet::Colormap(name) => {
                self.status = toast(format!("colormap {name}: only viridis is available"));
            }
            Snippet::ScatterAxis(axis, col) => match crate::cluster_scatter::Column::parse(&col) {
                Some(c) => self.cluster_scatter.axes[axis] = c,
                None => {
                    self.status = toast(format!(
                        "unknown column {col} (amp, depth, fr, n_spikes, ch)"
                    ))
                }
            },
            Snippet::TraceWindow(len) => {
                let s = read(&self.session);
                let samples = (len * s.provider.sample_rate()) as u64;
                self.window_len = samples.clamp(64, s.provider.n_samples().0.max(64)) as u32;
            }
        }
    }

    fn jobs_pending(&self) -> bool {
        self.suggest_cache.pending()
            || self.quality_cache.pending()
            || self.feature_state.pending()
            || self.amplitude_state.pending()
            || self.template_features.pending()
            || self.spike_attrs.pending()
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
                    // No toast: the status chip carries the save state, and
                    // a stale edit message must not outlive the save.
                    self.status = None;
                    self.last_save = Some(format!(
                        "Saved to {}{}",
                        dir.display(),
                        if self.reseal_on_save {
                            " · undo history sealed"
                        } else {
                            ""
                        }
                    ));
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
        let danger = ctx.global_style().visuals.error_fg_color;
        let btn = egui::vec2(110.0, 30.0);
        let mut decision: Option<Decision> = None;
        let frame = egui::Frame::popup(&ctx.global_style())
            .inner_margin(20.0)
            .corner_radius(8.0);
        let resp = egui::Modal::new(egui::Id::new("sorrel-confirm"))
            .frame(frame)
            .show(ctx, |ui| {
                ui.set_width(460.0);
                ui.heading(&title);
                ui.add_space(4.0);
                ui.label(&body);
                ui.add_space(14.0);
                // Right-aligned; the primary action is added first (rightmost).
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.spacing_mut().item_spacing.x = 10.0;
                    let primary = egui::Button::new(
                        egui::RichText::new(if is_quit { "Save & quit" } else { "Confirm" })
                            .color(egui::Color32::WHITE),
                    )
                    .fill(accent)
                    .min_size(btn);
                    let resp = ui.add(primary).on_hover_text("Enter");
                    if ui.memory(|m| m.focused().is_none()) {
                        resp.request_focus();
                    }
                    if resp.clicked() {
                        decision = Some(if is_quit {
                            Decision::SaveThenConfirm
                        } else {
                            Decision::Confirm
                        });
                    }
                    if is_quit
                        && ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new("Quit without saving").color(danger),
                                )
                                .stroke(egui::Stroke::new(1.0_f32, danger))
                                .min_size(btn),
                            )
                            .clicked()
                    {
                        decision = Some(Decision::Confirm);
                    }
                    if ui
                        .add(egui::Button::new("Cancel").min_size(btn))
                        .on_hover_text("Esc")
                        .clicked()
                    {
                        decision = Some(Decision::Cancel);
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
        // Keyboard rows come from the keymap itself, so they cannot drift.
        use crate::intent::{help_rows, Section};
        let keys = help_rows();
        let section = |s: Section| -> Vec<(String, &'static str)> {
            keys.iter()
                .filter(|r| r.0 == s)
                .map(|r| (r.1.clone(), r.2))
                .collect()
        };
        let mouse: Vec<(String, &'static str)> = [
            ("click", "select one cluster"),
            ("Ctrl-click", "add or remove a cluster"),
            ("Shift-click", "select the range of visible rows"),
            (
                "Ctrl-click in Features",
                "add a lasso point (Ctrl-right-click clears)",
            ),
            (
                "wheel · Ctrl-wheel · Alt-wheel in Features",
                "zoom · scale · marker size",
            ),
            ("click in Raster", "seek the trace to that time"),
            (
                "click in Similar",
                "compare with that cluster (Ctrl adds it)",
            ),
            (
                "drag / wheel in Trace",
                "pan · Ctrl-wheel zooms around the pointer",
            ),
            (
                "drag a tab",
                "rearrange views · × closes · View menu reopens",
            ),
        ]
        .map(|(k, v)| (k.to_string(), v))
        .to_vec();
        let groups = [
            ("Curation", section(Section::Curation)),
            ("Navigation", section(Section::Navigation)),
            ("Views", section(Section::Views)),
            ("Mouse", mouse),
        ];
        egui::Window::new("Keyboard & mouse")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .open(&mut self.show_help)
            .show(ctx, |ui| {
                ui.set_min_width(640.0);
                // The key list is taller than a small window: scroll it.
                egui::ScrollArea::vertical()
                    // Fit inside the centred window, which egui clamps to the screen.
                    .max_height(ctx.content_rect().height() * 0.6)
                    // A hidden bar made the clipped list look complete.
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
                    .show(ui, |ui| {
                        egui::Grid::new("sorrel-help")
                            .num_columns(2)
                            .spacing([24.0, 2.0])
                            .striped(true)
                            .show(ui, |ui| {
                                for (heading, rows) in &groups {
                                    ui.label(egui::RichText::new(*heading).strong());
                                    ui.end_row();
                                    for (key, what) in rows {
                                        ui.label(egui::RichText::new(key).monospace().strong());
                                        ui.label(*what);
                                        ui.end_row();
                                    }
                                }
                            });
                    });
            });
    }

    fn dispatch_intent(&mut self, intent: Intent) {
        let n = read(&self.session).n_clusters();
        // phy merge mode: picks become merge candidates and labels wait.
        if self.merge_mode.is_some() {
            match intent {
                Intent::SelectCluster(c) | Intent::ToggleCluster(c) | Intent::ExtendCluster(c) => {
                    if c.0 < n && !self.selection.best().contains(&c) {
                        self.selection.toggle_similar(c);
                    }
                    return;
                }
                Intent::Move(..) | Intent::Snippet(":l ") => {
                    self.status = toast("labels are unavailable in merge mode (V leaves it)");
                    return;
                }
                _ => {}
            }
        }
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
            Intent::Move(target, op) => {
                let clusters = match target {
                    MoveTarget::Best => self.selection.best(),
                    MoveTarget::Similar => self.selection.similar(),
                    MoveTarget::All => self.selection.as_slice(),
                }
                .to_vec();
                if clusters.is_empty() {
                    self.status = toast("no similar cluster selected");
                    return;
                }
                // One journal entry, so a single Undo reverts the whole move.
                let mut cmds: Vec<CurationCommand> = clusters
                    .iter()
                    .map(|&c| CurationCommand::Relabel { cluster: c, op })
                    .collect();
                let cmd = match cmds.len() {
                    1 => cmds.remove(0),
                    _ => match CurationCommand::batch(cmds) {
                        Some(b) => b,
                        None => return,
                    },
                };
                if !self.commit(cmd) {
                    return;
                }
                // phy `_after_move`: a moved similar cluster advances the
                // similar list; moving the best advances the cluster table.
                if target == MoveTarget::Similar {
                    let order = self.similar_order();
                    self.selection.step_similar(1, &order);
                } else {
                    self.selection.bump(1, &self.table_state.visible);
                }
            }
            Intent::FirstCluster => self.selection.bump(i32::MIN, &self.table_state.visible),
            Intent::LastCluster => self.selection.bump(i32::MAX, &self.table_state.visible),
            Intent::ResetWizard => self.selection.bump(i32::MIN, &self.table_state.visible),
            Intent::NextSimilar | Intent::PrevSimilar => {
                let order = self.similar_order();
                let delta = if intent == Intent::NextSimilar { 1 } else { -1 };
                self.selection.step_similar(delta, &order);
            }
            Intent::SelectFirstSimilar => {
                let order = self.similar_order();
                self.selection.set_similar(&order[..order.len().min(15)]);
            }
            Intent::UnselectSimilar => self.selection.set_similar(&[]),
            Intent::TraceStart => self.window_start = 0,
            Intent::TraceEnd => {
                self.window_start = read(&self.session)
                    .provider
                    .n_samples()
                    .0
                    .saturating_sub(self.window_len as u64);
            }
            Intent::TraceNarrow => self.window_len = (self.window_len / 2).max(64),
            Intent::TraceWiden => {
                let n = read(&self.session).provider.n_samples().0;
                self.window_len = (self.window_len.saturating_mul(2) as u64).min(n.max(64)) as u32;
            }
            Intent::ToggleSpikeReorder => {
                if !read(&self.session).cluster_index().has_spike_reorder() {
                    self.status = toast("no spike_reorder.npy in this dataset");
                } else {
                    self.spike_reorder = !self.spike_reorder;
                    self.status = toast(if self.spike_reorder {
                        "time axes: reordered (spike_reorder.npy)"
                    } else {
                        "time axes: recorded spike times"
                    });
                }
            }
            Intent::TraceOrigin => self.trace_display.flip = !self.trace_display.flip,
            Intent::TraceAllSpikes => {
                self.trace_display.all_spikes = !self.trace_display.all_spikes;
            }
            Intent::TraceLabels => self.trace_display.labels = !self.trace_display.labels,
            Intent::TraceGain(d) => {
                let g = self.trace_display.gain * 1.25f32.powi(d);
                self.trace_display.gain = g.clamp(0.05, 50.0);
            }
            Intent::NextSpike | Intent::PrevSpike => {
                // The selection's spike nearest after (before) the window
                // centre, centred in the window.
                let s = read(&self.session);
                let centre = self.window_start + self.window_len as u64 / 2;
                let next = intent == Intent::NextSpike;
                let pick = self
                    .selection
                    .iter()
                    .filter_map(|&c| {
                        let t = s.spike_times(c);
                        let i = t.partition_point(|x| x.0 <= centre);
                        if next {
                            t.get(i).map(|x| x.0)
                        } else {
                            let j = t.partition_point(|x| x.0 < centre);
                            j.checked_sub(1).map(|j| t[j].0)
                        }
                    })
                    .reduce(|a, b| if next { a.min(b) } else { a.max(b) });
                if let Some(t) = pick {
                    let max = s
                        .provider
                        .n_samples()
                        .0
                        .saturating_sub(self.window_len as u64);
                    self.window_start = t.saturating_sub(self.window_len as u64 / 2).min(max);
                }
            }
            Intent::Snippet(prefill) => self.snippet = Some(prefill.to_string()),
            Intent::Quit => self.quit_requested = true,
            Intent::ToggleMergeMode => match self.merge_mode.take() {
                // Cancel: back to the selection from before entry.
                Some(before) => self.selection = before,
                None if self.selection.is_empty() => {
                    self.status = toast("select a cluster to stage first");
                }
                None => {
                    // Stage the whole selection, reference (blue) first; the
                    // Similar pane then adds candidates.
                    self.merge_mode = Some(self.selection.clone());
                    let staged = self.selection.as_slice().to_vec();
                    self.selection.replace(staged[0]);
                    for &c in &staged[1..] {
                        self.selection.toggle(c);
                    }
                }
            },
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
                if self.commit(cmd) {
                    self.merge_mode = None;
                }
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
                } else if self.snippet.take().is_some() {
                    // Snippet bar closed.
                } else if self.show_help {
                    self.show_help = false;
                } else if self.spike_attrs.cancel_lasso() {
                    // Spike-data lasso cleared.
                } else if self.template_features.cancel_lasso() {
                    // Template-feature lasso cleared.
                } else if self.cluster_scatter.cancel_lasso() {
                    // Cluster-scatter lasso cleared.
                } else if self.amplitude_state.cancel_split() {
                    self.status = toast("amplitude split cancelled");
                } else if self.feature_state.lasso_active() {
                    self.feature_state.cancel_lasso();
                    self.status = toast("lasso cancelled");
                }
            }
            Intent::ToggleHelp => self.show_help = !self.show_help,
            Intent::SplitInit => {
                let cmds: Vec<CurationCommand> = {
                    let s = read(&self.session);
                    self.selection
                        .iter()
                        .flat_map(|&c| s.split_by_template_commands(c))
                        .collect()
                };
                let nothing = cmds.is_empty();
                match CurationCommand::batch(cmds).filter(|_| !nothing) {
                    Some(cmd) => {
                        self.commit(cmd);
                    }
                    None => self.status = toast("every selected cluster has a single template"),
                }
            }
            // phy asks every view for its lasso; the first armed one splits.
            Intent::Split => {
                if self.feature_state.lasso_closed() {
                    self.feature_state.request_split();
                } else if self.amplitude_state.can_split() {
                    self.amplitude_state.request_split();
                } else if self.template_features.lasso_closed() {
                    self.template_features.request_split();
                } else if self.spike_attrs.lasso_closed() {
                    self.spike_attrs.request_split();
                } else {
                    self.status = toast("draw a lasso first: Ctrl+click in Features or Amplitudes");
                }
            }
            Intent::CycleAmplitudes(delta) => {
                let kind = self.amplitude_state.cycle(delta);
                self.status = toast(format!("amplitudes: {kind:?}").to_lowercase());
            }
            Intent::ToggleFeatureChannels => {
                self.status = toast(if self.feature_state.toggle_channels() {
                    "feature channels fixed"
                } else {
                    "feature channels follow the selection"
                });
            }
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
            ui.heading(egui::RichText::new("Sorrel").strong());
            // Save-state chip: unsaved / saved / failed, with the detail on hover.
            let theme = crate::theme::Theme::from_ui(ui);
            let failed = self
                .error
                .as_deref()
                .filter(|e| e.starts_with("save failed"));
            // Icon kinds are painted: egui's bundled fonts lack ● and ✓.
            #[derive(PartialEq)]
            enum Icon {
                None,
                Dot,
                Check,
                Cross,
            }
            let (icon, text, colour, tip) = if let Some(e) = failed {
                (Icon::Cross, "save failed", theme.danger, e.to_string())
            } else if dirty {
                (
                    Icon::Dot,
                    "unsaved",
                    theme.warn,
                    "Unsaved curation — Cmd/Ctrl-S writes phy artefacts.".to_string(),
                )
            } else if let Some(tip) = &self.last_save {
                (Icon::Check, "saved", theme.good, tip.clone())
            } else {
                (
                    Icon::None,
                    "no changes",
                    theme.caption,
                    "All curation saved.".to_string(),
                )
            };
            egui::Frame::new()
                .fill(colour.gamma_multiply(0.18))
                .corner_radius(10)
                .inner_margin(egui::Margin::symmetric(8, 1))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.x = 5.0;
                    ui.horizontal(|ui| {
                        if icon != Icon::None {
                            let (r, _) =
                                ui.allocate_exact_size(egui::vec2(9.0, 9.0), egui::Sense::hover());
                            let (p, s) = (ui.painter(), egui::Stroke::new(1.6_f32, colour));
                            let c = r.center();
                            match icon {
                                Icon::Dot => {
                                    p.circle_filled(c, 3.5, colour);
                                }
                                Icon::Check => {
                                    p.add(egui::Shape::line(
                                        vec![
                                            egui::pos2(c.x - 4.0, c.y),
                                            egui::pos2(c.x - 1.5, c.y + 3.0),
                                            egui::pos2(c.x + 4.0, c.y - 3.0),
                                        ],
                                        s,
                                    ));
                                }
                                _ => {
                                    p.line_segment(
                                        [c - egui::vec2(3.5, 3.5), c + egui::vec2(3.5, 3.5)],
                                        s,
                                    );
                                    p.line_segment(
                                        [c + egui::vec2(-3.5, 3.5), c + egui::vec2(3.5, -3.5)],
                                        s,
                                    );
                                }
                            }
                        }
                        ui.label(egui::RichText::new(text).color(colour));
                    });
                })
                .response
                .on_hover_text(tip);
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
            let merge_label = if n_sel >= 2 {
                format!("Merge {n_sel} -> c{target}")
            } else {
                "Merge".to_string()
            };
            if ui
                .add_enabled(n_sel >= 2, egui::Button::new(merge_label))
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
            let save_tip = match self.save_dir.as_deref() {
                Some(dir) => format!(
                    "Write spike_clusters.npy + cluster_group.tsv to {} (Cmd/Ctrl-S)",
                    dir.display()
                ),
                None => "Open sorrel with a kilosort directory to enable saving.".to_string(),
            };
            if ui
                .add_enabled(dirty && self.save_dir.is_some(), egui::Button::new("Save"))
                .on_hover_text(&save_tip)
                .on_disabled_hover_text(if self.save_dir.is_some() {
                    "Nothing to save".to_string()
                } else {
                    save_tip.clone()
                })
                .clicked()
            {
                self.dispatch_intent(Intent::Save);
            }

            // Stable ids for the right-hand widgets: the left side's widget
            // count varies (spinner, error, status), which would otherwise
            // reshuffle auto ids between passes.
            let right = egui::Layout::right_to_left(egui::Align::Center);
            ui.with_layout(right, |ui| {
                ui.push_id("topbar-right", |ui| {
                    if ui
                        .button("Help")
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
                        .button("Export QC")
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
                                egui::Button::new("Export phy"),
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
                    // Whatever room is left between the two groups goes to
                    // the transient message, truncated with the full text
                    // on hover.
                    if self.jobs_pending() {
                        ui.weak("working…").on_hover_text(
                            "Background metrics are computing; curation stays responsive.",
                        );
                        ui.spinner();
                    }
                    if let Some(err) = self.error.clone() {
                        if ui.button("×").on_hover_text("dismiss").clicked() {
                            self.error = None;
                        }
                        let text = egui::RichText::new(&err).color(ui.visuals().error_fg_color);
                        ui.add(egui::Label::new(text).truncate())
                            .on_hover_text(&err);
                    } else if let Some((status, _)) = self.status.clone() {
                        let text = egui::RichText::new(&status).weak();
                        ui.add(egui::Label::new(text).truncate())
                            .on_hover_text(&status);
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
            ViewAction::SelectSimilar { cluster, add } => {
                if add {
                    self.selection.toggle_similar(cluster);
                } else {
                    self.selection.set_similar(&[cluster]);
                }
            }
            ViewAction::SelectChannel { channel, secondary } => {
                self.feature_state.select_channel(channel, secondary);
            }
            ViewAction::AddPane(pane, into) => layout::add_pane(&mut self.tree, pane, Some(into)),
            ViewAction::Screenshot(pane, rect) => self.pending_screenshots.push((pane, rect)),
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
        let name = self
            .save_dir
            .as_deref()
            .and_then(Path::file_name)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "the recording".into());
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
                    egui::CentralPanel::default().show_inside(root_ui, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.add_space(ui.available_height() * 0.35);
                            ui.add(egui::Spinner::new().size(28.0));
                            ui.add_space(8.0);
                            ui.label(egui::RichText::new("Sorrel").strong());
                            ui.heading(format!("Loading {name}…"));
                            ui.label(
                                "seeding amplitudes, features and templates; replaying the journal",
                            );
                        });
                    });
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                    return true;
                }
            }
        }
        if let Some(err) = self.load_error.clone() {
            let path = self.save_dir.as_deref().map(|p| p.display().to_string());
            egui::CentralPanel::default().show_inside(root_ui, |ui| {
                ui.vertical_centered(|ui| {
                    let danger = ui.visuals().error_fg_color;
                    ui.add_space(ui.available_height() * 0.35);
                    ui.heading(
                        egui::RichText::new(format!("Could not open {name}"))
                            .strong()
                            .color(danger),
                    );
                    ui.add_space(4.0);
                    if let Some(path) = path {
                        egui::Frame::group(ui.style()).show(ui, |ui| {
                            ui.set_width(480.0);
                            ui.add(egui::Label::new(egui::RichText::new(path).monospace()).wrap());
                        });
                    }
                    ui.scope(|ui| {
                        ui.set_max_width(480.0);
                        ui.add(egui::Label::new(egui::RichText::new(err).color(danger)).wrap());
                    });
                    ui.add_space(8.0);
                    if ui
                        .add(egui::Button::new("Quit").min_size(egui::vec2(160.0, 30.0)))
                        .clicked()
                    {
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
        if !self.visuals_installed {
            crate::theme::install_visuals(&ctx);
            crate::theme::install_style(&ctx);
            self.visuals_installed = true;
        }
        if self.show_loading(&ctx, root_ui) {
            return;
        }

        // Input. Suppress single-letter intents while a text widget has focus
        // so typing in a filter box doesn't relabel selected clusters; global
        // shortcuts (Cmd/Ctrl-S/Z/Shift-Z) still fire. While a confirm
        // dialog is open only Esc gets through — the modal owns the keys.
        // phy `screenshot`: ask for the frame, then crop and save the pane.
        for (pane, rect) in self.pending_screenshots.drain(..) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new((
                pane, rect,
            ))));
        }
        let shots: Vec<(Pane, egui::Rect, std::sync::Arc<egui::ColorImage>)> = ctx.input(|i| {
            i.events
                .iter()
                .filter_map(|ev| match ev {
                    egui::Event::Screenshot {
                        user_data, image, ..
                    } => {
                        let &(pane, rect) = user_data
                            .data
                            .as_ref()?
                            .downcast_ref::<(Pane, egui::Rect)>()?;
                        Some((pane, rect, image.clone()))
                    }
                    _ => None,
                })
                .collect()
        });
        for (pane, rect, image) in shots {
            let ppp = ctx.pixels_per_point();
            self.status = toast(
                match self.save_screenshot(pane, &image.region(&rect, Some(ppp))) {
                    Ok(path) => format!("screenshot saved to {}", path.display()),
                    Err(e) => format!("screenshot failed: {e}"),
                },
            );
        }

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

        if std::mem::take(&mut self.quit_requested) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        let dirty = self.dirty();

        // Unsaved-changes guard: intercept the window close once and ask.
        if ctx.input(|i| i.viewport().close_requested()) && dirty && !self.close_approved {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.pending_confirm = Some(PendingConfirm {
                title: "Quit with unsaved curation?".into(),
                body: format!(
                    "Edits are journaled, but spike_clusters.npy and cluster_group.tsv in {} \
                     have not been written since the last save.",
                    self.save_dir
                        .as_deref()
                        .map_or_else(|| "the save directory".to_string(), short_path)
                ),
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

        // 24 px controls + 6 px margins = a 36 px bar.
        let bar_frame = egui::Frame::side_top_panel(root_ui.style())
            .inner_margin(egui::Margin::symmetric(8, 6));
        egui::Panel::top("top")
            .frame(bar_frame)
            .show_inside(root_ui, |ui| self.top_bar(ui, dirty));
        if self.merge_mode.is_some() {
            let (staged, similar) = (self.selection.best().len(), self.selection.similar().len());
            egui::Panel::bottom("merge-mode")
                .frame(bar_frame)
                .show_inside(root_ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.strong("Merge mode");
                        ui.label(format!(
                            "{staged} staged + {similar} similar · G merges {} · V cancels",
                            staged + similar
                        ));
                        if ui.button("Cancel merge mode").clicked() {
                            self.dispatch_intent(Intent::ToggleMergeMode);
                        }
                    });
                });
        }
        if self.snippet.is_some() {
            egui::Panel::bottom("snippet")
                .frame(bar_frame)
                .show_inside(root_ui, |ui| self.snippet_bar(ui));
        }

        // No side margin: the table runs edge to edge, and the heading and
        // filter strip pad themselves.
        let side_frame = egui::Frame::side_top_panel(root_ui.style()).inner_margin(egui::Margin {
            left: 0,
            right: 0,
            top: 12,
            bottom: 0,
        });
        egui::Panel::left("clusters")
            .frame(side_frame)
            .resizable(true)
            .default_size((root_ui.available_width() * 0.22).max(300.0))
            .size_range(300.0..=640.0)
            .show_inside(root_ui, |ui| {
                let session = read(&self.session);
                ui.horizontal(|ui| {
                    ui.add_space(12.0);
                    ui.heading("Clusters");
                    // `visible` is last frame's list; one frame of lag is invisible.
                    ui.weak(format!(
                        "{} / {}",
                        self.table_state.visible.len(),
                        session.n_clusters()
                    ))
                    .on_hover_text("visible / total clusters");
                });
                self.cluster_rows.ensure(
                    &session,
                    RowInputs {
                        positions: &self.channel_positions,
                        shanks: &self.channel_shanks,
                        metrics: &self.cluster_metrics,
                    },
                    self.label_str,
                );
                let selected: Vec<ClusterId> = self.selection.iter().copied().collect();
                let click = cluster_table(
                    ui,
                    &session,
                    &mut self.table_state,
                    TableCtx {
                        selected: &selected,
                        anchor: self.selection.anchor(),
                        rows: &self.cluster_rows,
                        hidden: &mut self.view_params.hidden_cols,
                        similar: None,
                    },
                );
                if let Some(click) = click {
                    click.apply(&mut self.selection, &self.table_state.visible);
                }
            });

        egui::CentralPanel::default().show_inside(root_ui, |ui| {
            // Every view is a tile; views only read the session and queue
            // side effects as actions, applied once the read lock is gone.
            let selected: Vec<ClusterId> = self.selection.iter().copied().collect();
            let closed = layout::missing_panes(&self.tree);
            let active_tabs = layout::active_tabs(&self.tree);
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
                // Colour-scheme colours for unselected clusters, per curation
                // version and scheme.
                let scheme_key = (session.version(), self.scheme, depths.is_some());
                if self
                    .scheme_colours
                    .as_ref()
                    .is_none_or(|(k, ..)| *k != scheme_key)
                {
                    let n = session.n_clusters();
                    let duration = (session.provider.n_samples().0 as f32
                        / session.provider.sample_rate().max(1.0))
                    .max(1e-3);
                    let ids = (0..n).map(ClusterId);
                    let groups: Vec<&str> = ids
                        .clone()
                        .map(|c| session.label(c).map_or("", |l| (self.label_str)(&l)))
                        .collect();
                    let fr: Vec<f32> = ids
                        .clone()
                        .map(|c| session.spike_times(c).len() as f32 / duration)
                        .collect();
                    let depth: Vec<f32> = ids
                        .map(|c| {
                            let d = depths.and_then(|d| d.per_cluster.get(c.idx()));
                            d.filter(|d| !d.is_empty())
                                .map_or(f32::NAN, |d| d.iter().sum::<f32>() / d.len() as f32)
                        })
                        .collect();
                    let colours = crate::theme::scheme_colours(self.scheme, &groups, &depth, &fr);
                    self.scheme_gen += 1;
                    self.scheme_colours = Some((scheme_key, self.scheme_gen, colours));
                }
                let (_, colours_gen, colours) = self.scheme_colours.as_ref().expect("just built");
                let mut behavior = ViewBehavior {
                    session: &session,
                    selection: &selected,
                    label_str: self.label_str,
                    channel_positions: &self.channel_positions,
                    waveform_cache: &mut self.waveform_cache,
                    feature_state: &mut self.feature_state,
                    amplitude_state: &mut self.amplitude_state,
                    cluster_scatter: &mut self.cluster_scatter,
                    template_features: &mut self.template_features,
                    frozen: &mut self.frozen,
                    screenshot: &mut self.screenshot_pane,
                    table_order: &self.table_state.visible,
                    template_grid: &mut self.template_grid,
                    trace_image: &mut self.trace_image,
                    spike_attrs: &mut self.spike_attrs,
                    colours,
                    colours_gen: *colours_gen,
                    scheme_delta: 0,
                    spike_reorder: self.spike_reorder,
                    raster_cache: &mut self.raster_cache,
                    suggest_cache: &mut self.suggest_cache,
                    ccg_cache: &mut self.ccg_cache,
                    view_params: &mut self.view_params,
                    similar: SimilarityPane {
                        state: &mut self.similarity,
                        rows: &self.cluster_rows,
                        best: self.selection.best(),
                    },
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
                        display: self.trace_display,
                    },
                    closed,
                    active_tabs,
                    actions: Vec::new(),
                };
                self.tree.ui(&mut behavior, ui);
                (behavior.actions, behavior.scheme_delta)
            };
            let (actions, scheme_delta) = actions;
            if scheme_delta != 0 {
                self.scheme = self.scheme.step(scheme_delta);
                self.status = toast(format!("colour scheme: {}", self.scheme.name()));
            }
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
    fn merge_mode_stages_candidates_and_cancel_restores() {
        let (mut app, _d) = make_app();
        app.selection.replace(ClusterId(0));
        app.dispatch_intent(Intent::ToggleMergeMode);
        // Picks become candidates; labels wait.
        app.dispatch_intent(Intent::SelectCluster(ClusterId(2)));
        assert_eq!(app.selection.as_slice(), &[ClusterId(0), ClusterId(2)]);
        app.dispatch_intent(Intent::Move(MoveTarget::All, PhyLabelOp::SetGood));
        assert_eq!(history_len(&app), 0, "no relabel in merge mode");
        // V again cancels back to the pre-entry selection.
        app.dispatch_intent(Intent::ToggleMergeMode);
        assert_eq!(app.selection.as_slice(), &[ClusterId(0)]);
        // G merges staged + candidates and leaves merge mode.
        app.dispatch_intent(Intent::ToggleMergeMode);
        app.dispatch_intent(Intent::ToggleCluster(ClusterId(1)));
        app.dispatch_intent(Intent::MergeSelection);
        assert_eq!(history_len(&app), 1);
        assert!(app.merge_mode.is_none());
    }

    #[test]
    fn alt_pgdn_centres_the_trace_on_the_selections_next_spike() {
        let (mut app, _d) = make_app();
        app.selection.replace(ClusterId(0));
        app.selection.toggle(ClusterId(1));
        (app.window_start, app.window_len) = (0, 10);
        app.dispatch_intent(Intent::NextSpike);
        assert_eq!(app.window_start, 5, "spike 10 centred");
        app.dispatch_intent(Intent::NextSpike);
        assert_eq!(app.window_start, 25, "spike 30, across both clusters");
        app.dispatch_intent(Intent::NextSpike);
        assert_eq!(app.window_start, 35, "spike 40 of cluster 1");
        app.dispatch_intent(Intent::PrevSpike);
        assert_eq!(app.window_start, 25);
    }

    #[test]
    fn relabel_intent_mutates_session_and_marks_dirty() {
        let (mut app, _d) = make_app();
        assert!(!app.dirty(), "fresh session should be clean");

        app.dispatch_intent(Intent::Move(MoveTarget::All, PhyLabelOp::SetGood));

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
        app.dispatch_intent(Intent::Move(MoveTarget::All, PhyLabelOp::SetMua));
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
        app.dispatch_intent(Intent::Move(MoveTarget::All, PhyLabelOp::SetMua));
        assert!(app.dirty());

        app.dispatch_intent(Intent::Save);

        assert!(app.error.is_none(), "save should succeed: {:?}", app.error);
        assert!(
            app.status.is_none(),
            "save clears the toast; the chip says it"
        );
        assert!(app.last_save.is_some(), "save should feed the saved chip");
        assert!(
            !app.dirty(),
            "saved_state should snap to current after a successful save"
        );
        let prefs = UiPrefs::load(dir.path());
        assert!(prefs.tree.is_some(), "prefs written next to the dataset");
        // The move advanced the wizard to the next row (phy `_after_move`).
        assert_eq!(prefs.anchor, Some(1));
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
        app.dispatch_intent(Intent::Move(MoveTarget::All, PhyLabelOp::SetNoise));
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

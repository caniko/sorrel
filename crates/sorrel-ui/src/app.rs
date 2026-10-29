use crate::amplitude_view::AmplitudeViewState;
use crate::cluster_view::{cluster_table, ClusterRows, ClusterTableState, RowInputs, TableCtx};
use crate::feature_view::FeatureViewState;
use crate::intent::{from_key_in_context, Intent, MoveTarget};
use crate::jobs::{read, write, Job, JobCtx};
use crate::layout::{self, Pane, TraceControls, ViewAction, ViewBehavior};
use crate::prefs::UiPrefs;
use crate::selection::SelectionSet;
use crate::shell::CloseState;
use crate::similarity_view::{SimilarityPane, SimilarityState};
use crate::views::{
    CcgCache, QualityCache, RasterCache, SuggestAction, SuggestCache, WaveformCache,
};
use crate::widgets::TraceCache;
use eframe::egui;
use sorrel_data::session::ApplyPhyLabel;
use sorrel_data::SpikeDepths;
use sorrel_data::{
    actor_name, describe_command, export_qc, save_session, Actor, AgentPolicy, CurationCommand,
    HostEvent, PhyLabelOp, Session, SessionHost,
};
use sorrel_gpu::GpuContext;
use sorrel_io::{ClusterId, DataProvider, Events, HasGeometry};
use sorrel_render::{GpuTracePreproc, TracePreproc};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, RwLock};
use std::time::Instant;

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

/// The selection after an edit, computed from the edit's own event.
type Follow = Box<dyn FnOnce(&HostEvent) -> Vec<ClusterId>>;

/// The ids a split allocated: the tail of its event's clusters.
fn new_ids(ev: &HostEvent, k: usize) -> &[ClusterId] {
    &ev.clusters[ev.clusters.len().saturating_sub(k)..]
}

/// Why accepting `p` would no longer do what it proposed, if so: a cluster
/// it names has since been emptied, or a split index is out of range.
fn stale_reason<P: DataProvider>(
    p: &sorrel_data::proposals::Proposal,
    s: &Session<P>,
) -> Option<String> {
    use sorrel_data::proposals::Action;
    let named: Vec<ClusterId> = match &p.action {
        Action::Merge { clusters } => clusters.clone(),
        Action::Relabel { cluster, .. }
        | Action::Split { cluster, .. }
        | Action::Accept { cluster } => vec![*cluster],
    };
    if let Some(c) = named.iter().find(|&&c| s.spike_times(c).is_empty()) {
        return Some(format!("cluster {} is empty now", c.0));
    }
    match &p.action {
        Action::Split { cluster, spike_idx } => {
            // The indices are positions in the membership it was proposed on.
            if p.basis
                .as_ref()
                .is_some_and(|b| *b != sorrel_data::proposals::spike_basis(s, *cluster))
            {
                return Some(format!(
                    "cluster {} changed since it was proposed",
                    cluster.0
                ));
            }
            let n = s.spike_times(*cluster).len();
            spike_idx
                .iter()
                .any(|&i| i as usize >= n)
                .then(|| format!("cluster {} has only {n} spikes now", cluster.0))
        }
        _ => None,
    }
}

/// Select a split's source and the new half it made.
fn follow_split(cluster: ClusterId) -> Follow {
    Box::new(move |ev| [&[cluster][..], new_ids(ev, 1)].concat())
}

/// Per-spike depth estimator a backend can plug in (see
/// [`SorrelApp::set_depth_source`]).
pub type DepthFn<P> = fn(&Session<P>) -> Option<SpikeDepths>;

/// Whole-dataset phy exporter the binary plugs in (see
/// [`SorrelApp::set_phy_export`]); returns the directory it wrote.
pub type PhyExportFn<P> = Arc<dyn Fn(&Session<P>) -> anyhow::Result<PathBuf> + Send + Sync>;

/// What the window keeps while it serves MCP clients: the listener, the
/// `select` requests and the selection it publishes.
struct McpLink {
    _listener: sorrel_data::ListenerHandle,
    select: Receiver<Vec<ClusterId>>,
    selection: sorrel_data::SharedSelection,
}

/// A waker that repaints the window once its context is known.
fn waker(ctx: &Arc<std::sync::OnceLock<egui::Context>>) -> impl Fn() + Send + Sync + 'static {
    let ctx = Arc::clone(ctx);
    move || {
        if let Some(c) = ctx.get() {
            c.request_repaint();
        }
    }
}

/// One line for a proposal: who, what, how sure.
pub(crate) fn describe_proposal(p: &sorrel_data::proposals::Proposal) -> String {
    use sorrel_data::proposals::Action;
    let ids = |c: &[ClusterId]| {
        c.iter()
            .map(|c| c.0.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let what = match &p.action {
        Action::Relabel { cluster, label } => format!("relabel {} as {label}", cluster.0),
        Action::Merge { clusters } => match clusters.split_first() {
            Some((t, rest)) => format!("merge {} into {}", ids(rest), t.0),
            None => "merge".into(),
        },
        Action::Split { cluster, spike_idx } => {
            format!("split {} spikes off {}", spike_idx.len(), cluster.0)
        }
        Action::Accept { cluster } => format!("keep {} as it is", cluster.0),
    };
    let by = p
        .proposer
        .clone()
        .unwrap_or_else(|| p.source.tag().trim_start_matches("src:").to_owned());
    format!("{by}: {what} ({:.0}%)", p.confidence * 100.0)
}

/// How long the clusters another actor touched stay highlighted.
const HIGHLIGHT: std::time::Duration = std::time::Duration::from_secs(3);

/// Generic eframe app. The whole window is monomorphised against `P`.
///
/// The session is owned by a [`SessionHost`] and sits behind its
/// `Arc<RwLock<_>>`: the UI thread takes short read locks per frame,
/// curation commands go through [`SessionHost::apply`] (the write lock),
/// and background jobs (Suggest ranking, quality metrics, initial seeding)
/// hold a read or write lock on worker threads — see [`crate::jobs`].
pub struct SorrelApp<P: DataProvider + ApplyPhyLabel> {
    host: Arc<SessionHost<P>>,
    /// `host.session()`, kept for the per-frame reads.
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
    /// Where this tab is in closing; the [`Shell`](crate::shell::Shell) reads it.
    close_state: CloseState,
    show_help: bool,
    /// phy snippet bar text while open (`:` or `l`).
    snippet: Option<String>,
    /// MCP clients' link to this window (see [`Self::serve_mcp`]).
    mcp: Option<McpLink>,
    /// `serve_mcp` asked while the loader still owned the session: started
    /// once loading succeeds, never if it fails.
    mcp_after_load: Option<fn(&mut Self) -> anyhow::Result<()>>,
    /// The host's events (subscribed at build) and the context their waker
    /// repaints, known from the first frame.
    host_events: Receiver<HostEvent>,
    repaint: Arc<std::sync::OnceLock<egui::Context>>,
    /// Clusters the last edit by another actor touched, and when: tinted
    /// for `highlight_for` ([`HIGHLIGHT`] unless a test pins it).
    highlight: Option<(Vec<ClusterId>, Instant)>,
    highlight_for: std::time::Duration,
    /// How long the status toast stays (5 s unless a test pins it).
    toast_for: std::time::Duration,
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
    /// Close this tab; the dialog offers "Save & quit" as well.
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
        let host = Arc::new(SessionHost::new(session));
        let repaint = Arc::new(std::sync::OnceLock::<egui::Context>::new());
        let host_events = host.subscribe(waker(&repaint));
        Self {
            session: Arc::clone(host.session()),
            host,
            loading: None,
            load_error: None,
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
            close_state: CloseState::Open,
            show_help: false,
            snippet: None,
            mcp: None,
            mcp_after_load: None,
            host_events,
            repaint,
            highlight: None,
            highlight_for: HIGHLIGHT,
            toast_for: std::time::Duration::from_secs(5),
            visuals_installed: false,
        }
    }

    /// Called once the session is seeded: initial selection, clean-state
    /// snapshot, and the per-dataset prefs.
    fn finish_loading(&mut self) {
        let (n_clusters, sr) = {
            let s = read(&self.session);
            (s.n_clusters(), s.provider.sample_rate())
        };
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

    /// The host that owns the session; other frontends (MCP, remote) edit
    /// through it.
    pub fn host(&self) -> &Arc<SessionHost<P>> {
        &self.host
    }

    /// Serve this window's session to MCP clients (`sorrel mcp <dir>`
    /// attaches): a loopback listener next to the save directory, whose
    /// `select` moves this selection. Needs `set_save_dir` first.
    pub fn serve_mcp(&mut self) -> anyhow::Result<()>
    where
        P: HasGeometry + 'static,
        Session<P>: Send + Sync,
    {
        let dir = self
            .save_dir
            .clone()
            .ok_or_else(|| anyhow::anyhow!("MCP needs a save directory"))?;
        // A client must never edit a bare or half-replayed session.
        if self.loading.is_some() {
            self.mcp_after_load = Some(Self::serve_mcp);
            return Ok(());
        }
        let (tx, select) = std::sync::mpsc::channel();
        let selection = sorrel_data::SharedSelection::default();
        let ui = sorrel_data::Ui {
            selection: Arc::clone(&selection),
            select: tx,
            wake: Arc::new(waker(&self.repaint)),
        };
        let listener = sorrel_data::spawn_listener(
            Arc::clone(&self.host),
            &dir,
            self.reseal_on_save,
            self.label_str,
            Some(Arc::new(ui)),
        )?;
        self.mcp = Some(McpLink {
            _listener: listener,
            select,
            selection,
        });
        Ok(())
    }

    /// Show what other actors did since the last frame and apply what MCP
    /// clients asked for: an edit by an agent (or a remote guest) is named
    /// in the toast ("codex: merge …"), its clusters are highlighted and
    /// selected, as a human edit selects its result; `select` replaces the
    /// selection; the selection is published back for `selection`.
    fn poll_host(&mut self) {
        let theirs: Vec<HostEvent> = self
            .host_events
            .try_iter()
            .filter(|e| e.actor != Actor::Human)
            .collect();
        if let Some(ev) = theirs.last() {
            self.status = toast(format!("{}: {}", actor_name(&ev.actor), ev.desc));
            let live: Vec<ClusterId> = {
                let s = read(&self.session);
                theirs
                    .iter()
                    .flat_map(|e| e.clusters.iter().copied())
                    .filter(|&c| !s.spike_times(c).is_empty())
                    .collect()
            };
            if let Some((&first, rest)) = live.split_first() {
                self.selection.replace(first);
                rest.iter().for_each(|&c| self.selection.toggle(c));
            }
            self.highlight = Some((live, Instant::now()));
        }
        let Some(link) = &self.mcp else { return };
        for ids in link.select.try_iter() {
            if let Some((&first, rest)) = ids.split_first() {
                self.selection.replace(first);
                rest.iter().for_each(|&c| self.selection.toggle(c));
            }
        }
        *link.selection.lock().unwrap_or_else(|p| p.into_inner()) =
            self.selection.as_slice().to_vec();
    }

    /// Top-bar agent strip: an activity dot (lit while the last agent edit
    /// is highlighted), the connected MCP clients, the pending proposals,
    /// and the Direct | Propose policy toggle.
    fn agent_indicator(&mut self, ui: &mut egui::Ui) {
        let theme = crate::theme::Theme::from_ui(ui);
        let active = self.highlighted().is_some();
        let agents = self.host.agents();
        let (r, _) = ui.allocate_exact_size(egui::vec2(9.0, 9.0), egui::Sense::hover());
        let dot = if active { theme.good } else { theme.caption };
        ui.painter().circle_filled(r.center(), 3.5, dot);
        let who = if agents.is_empty() {
            "no agent".to_string()
        } else {
            agents.join(", ")
        };
        ui.label(who).on_hover_text(if active {
            "An agent just edited; the touched clusters are highlighted"
        } else {
            "MCP clients connected with `sorrel mcp <DATA_DIR>`"
        });
        let pending = self.host.proposals().len();
        if pending > 0
            && ui
                .button(egui::RichText::new(format!("{pending} to review")).color(theme.warn))
                .on_hover_text("Open the Proposals pane")
                .clicked()
        {
            layout::add_pane(&mut self.tree, Pane::Proposals, None);
        }
        let policy = self.host.policy();
        for (p, name, tip) in [
            (
                AgentPolicy::Direct,
                "Direct",
                "MCP agents edit directly; every edit is undoable",
            ),
            (
                AgentPolicy::ProposeOnly,
                "Propose",
                "MCP agents only propose; you accept or reject each one",
            ),
        ] {
            if ui
                .selectable_label(policy == p, name)
                .on_hover_text(tip)
                .clicked()
            {
                self.host.set_policy(p);
            }
        }
    }

    /// Accept (through the host, as a human decision that keeps the
    /// proposer as the method) or reject proposal `id`, and audit it.
    fn decide_proposal(&mut self, id: u64, accept: bool) {
        use sorrel_data::proposals::Action;
        let Some(p) = self.host.take_proposal(id) else {
            return;
        };
        let what = describe_proposal(&p);
        // Select the result the way the same edit made by hand would.
        let follow: Follow = match &p.action {
            _ if !accept => Box::new(|_| Vec::new()),
            Action::Merge { clusters } => {
                let target: Vec<_> = clusters.first().copied().into_iter().collect();
                Box::new(move |_| target)
            }
            Action::Split { cluster, .. } => follow_split(*cluster),
            _ => Box::new(|_| Vec::new()),
        };
        // Accept and relabel-reject can leave nothing to journal. A proposal
        // that no longer fits the clusters (or whose edit fails) is audited
        // as superseded and its card dropped; the error says why.
        // The staleness check and the commands (whose `Tag` edits replace a
        // cluster's whole tag list) are built under the write lock of the
        // apply itself, so no MCP tag edit can land in between and be lost.
        let build = |s: &Session<P>| {
            if accept {
                if let Some(why) = stale_reason(&p, s) {
                    anyhow::bail!("proposal no longer applies ({why})");
                }
            }
            let cmds = match accept {
                true => p.to_commands(s),
                false => p.reject_commands(s),
            };
            Ok(CurationCommand::batch(cmds).filter(|b| !b.clusters().is_empty()))
        };
        let applied = self.commit_built(&what, build, follow);
        let outcome = match (applied, accept) {
            (false, _) => sorrel_data::proposals::Outcome::Superseded,
            (true, true) => sorrel_data::proposals::Outcome::Accepted,
            (true, false) => sorrel_data::proposals::Outcome::Rejected,
        };
        if let Some(dir) = &self.save_dir {
            if let Err(e) = sorrel_data::proposals::record_outcome(dir, id, outcome) {
                log::warn!("proposal audit log: {e}");
            }
        }
        if !applied {
            return;
        }
        let verb = if accept { "accepted" } else { "rejected" };
        self.status = toast(format!("{verb}: {what}"));
    }

    /// How long the status toast and another actor's highlight stay up.
    /// Snapshot tests pin it long so a slow renderer cannot outlast them.
    pub fn set_feedback_duration(&mut self, d: std::time::Duration) {
        self.highlight_for = d;
        self.toast_for = d;
    }

    /// The highlighted clusters while the highlight lasts.
    fn highlighted(&self) -> Option<&[ClusterId]> {
        self.highlight
            .as_ref()
            .filter(|(_, at)| at.elapsed() < self.highlight_for)
            .map(|(c, _)| c.as_slice())
    }

    /// Hand the app the wgpu device/queue from `eframe`'s render state so
    /// trace CMR + HP filter can run on the GPU. Without this the app
    /// transparently falls back to the CPU implementation.
    pub fn install_gpu_compute(
        &mut self,
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
        adapter: &wgpu::AdapterInfo,
    ) {
        // egui's pipelines exist by now (their failure stays fatal); from here a
        // stray wgpu error is logged instead of killing the session.
        sorrel_gpu::log_uncaptured_errors(&device);
        if !sorrel_gpu::supports_compute(&device) {
            log::info!("GPU adapter has no compute support; trace preprocessing stays on the CPU");
            return;
        }
        let ctx = GpuContext::from_existing(device, queue);
        match GpuTracePreproc::new(&ctx) {
            Ok(p) => self.gpu_preproc = Some(p),
            Err(e) => log::warn!(
                "GPU kernels unavailable on {} ({:?}); trace preprocessing stays on the CPU: {e}",
                adapter.name,
                adapter.backend
            ),
        }
    }

    /// Configure where Cmd/Ctrl-S writes the curated state. The binary
    /// typically points this at the kilosort root. Also loads the
    /// per-dataset UI prefs stored next to the cache.
    ///
    /// `reseal` must be `true` only when `dir` is the phy/Kilosort directory
    /// the session was opened from: the save then captures the root and
    /// seals the journal (`sorrel_data::save_and_seal`). Other targets get a
    /// plain export.
    pub fn set_save_dir(&mut self, dir: PathBuf, reseal: bool) {
        let prefs = UiPrefs::load(&dir);
        match Events::detect(&dir) {
            Ok(events) => self.events = events.map(Arc::new),
            Err(e) => log::warn!("ignoring events file in {}: {e}", dir.display()),
        }
        self.host.restore_proposals(&dir);
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
    /// entries, and seal the journal when this backend requires it.
    fn save_to(&mut self, dir: &Path) -> anyhow::Result<()> {
        self.cancel_jobs();
        let mut session = write(&self.session);
        save_session(&mut session, dir, self.reseal_on_save, self.label_str)?;
        if let Err(err) = session.invalidate_dataset_caches() {
            log::warn!("cache invalidation after save failed: {err}");
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
        self.suggest_cache.positions = positions.clone();
        self.channel_positions = positions;
    }

    /// The title bar shows a "•" dirty marker while the head is not the
    /// saved node.
    fn dirty(&self) -> bool {
        // The loader holds the write lock; never block the UI on it.
        self.loading.is_none() && !read(&self.session).is_saved()
    }

    /// Ask to close this tab. Clean (or already approved): `Closed` at once.
    /// Unsaved curation raises the "Quit with unsaved curation?" dialog and
    /// sets `Asking`; the dialog's answer ends in `Closed` or back to `Open`.
    pub fn request_close(&mut self) {
        if self.close_state != CloseState::Open {
            return;
        }
        if self.load_error.is_some() || !self.dirty() {
            self.close_state = CloseState::Closed;
            return;
        }
        self.close_state = CloseState::Asking;
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

    /// OS window title: dataset name + dirty marker.
    pub fn window_title(&self) -> String {
        let dataset = self
            .save_dir
            .as_deref()
            .and_then(Path::file_name)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        match (dataset.is_empty(), self.dirty()) {
            (true, false) => "Sorrel".to_string(),
            (true, true) => "Sorrel •".to_string(),
            (false, false) => format!("{dataset} — Sorrel"),
            (false, true) => format!("{dataset} • — Sorrel"),
        }
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
                        .hint_text(":c 3 7 · :g · :l group good · :note text · :tg 12.5"),
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
            Snippet::Note(text) => {
                // One note on every selected cluster, one undo step.
                let mut cmds: Vec<CurationCommand> = self
                    .selection
                    .iter()
                    .map(|&cluster| CurationCommand::Note {
                        cluster,
                        text: text.clone(),
                    })
                    .collect();
                match cmds.len() {
                    0 => {}
                    1 => {
                        self.commit(cmds.remove(0));
                    }
                    _ => {
                        if let Some(b) = CurationCommand::batch(cmds) {
                            self.commit(b);
                        }
                    }
                }
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
                let (lo, hi) = self.view_params.rate.x_range(duration);
                self.view_params.rate.n_bins =
                    (((hi - lo) / secs.max(1e-3)).round() as usize).clamp(2, 5000);
            }
            Snippet::RateXMin(t) => {
                self.view_params.rate.x_min = t;
                self.view_params.rate.pz = Default::default();
            }
            Snippet::RateXMax(t) => {
                self.view_params.rate.x_max = t;
                self.view_params.rate.pz = Default::default();
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
        let follow: Follow = match &cmd {
            CurationCommand::Merge { target, .. } => {
                let target = vec![*target];
                Box::new(move |_| target)
            }
            CurationCommand::Split { cluster, .. } => follow_split(*cluster),
            _ => Box::new(|_| Vec::new()),
        };
        self.commit_with_follow(cmd, follow)
    }

    /// Every view cache is keyed on `Session::version()`, so nothing needs
    /// invalidating by hand here: the next frame rebuilds what changed.
    fn commit_with_follow(&mut self, cmd: CurationCommand, follow: Follow) -> bool {
        self.commit_with_follow_if(cmd, follow, |_| Ok(()))
    }

    /// [`Self::commit_with_follow`], but `check` runs under the host's write
    /// lock, so no MCP edit can land between the check and the apply.
    fn commit_with_follow_if(
        &mut self,
        cmd: CurationCommand,
        follow: Follow,
        check: impl FnOnce(&Session<P>) -> anyhow::Result<()>,
    ) -> bool {
        let desc = describe_command(&cmd);
        self.commit_built(
            &desc,
            |s| {
                check(s)?;
                Ok(Some(cmd))
            },
            follow,
        )
    }

    /// Commit the command `build` makes from the session under the host's
    /// write lock (`None` = nothing to journal, which counts as done).
    /// `what` names the edit in the error.
    fn commit_built(
        &mut self,
        what: &str,
        build: impl FnOnce(&Session<P>) -> anyhow::Result<Option<CurationCommand>>,
        follow: Follow,
    ) -> bool {
        self.cancel_jobs();
        // Every GUI edit is a human decision: the host clears the touched
        // clusters' `review:*` tags (from `sorrel auto`), so the review
        // queue drains.
        let ev = match self.host.apply_built(Actor::Human, build) {
            Ok(Some(ev)) => ev,
            Ok(None) => return true,
            Err(e) => {
                self.error = Some(format!("{what} failed: {e}"));
                return false;
            }
        };
        // The ids come from our own event, not from a cluster count read
        // before the apply: an MCP edit may have landed in between.
        if let Some((&first, rest)) = follow(&ev).split_first() {
            self.selection.replace(first);
            for &c in rest {
                self.selection.toggle(c);
            }
        }
        self.status = toast(ev.desc);
        true
    }

    /// Undo or redo; the toast names what moved ("undid: relabel cluster 12
    /// as MUA"), and nothing is said at the end of the walk.
    fn move_head(&mut self, cmd: CurationCommand) {
        self.cancel_jobs();
        let before = read(&self.session).version();
        match self.host.apply(Actor::Human, cmd) {
            Err(e) => self.error = Some(format!("journal write failed: {e}")),
            Ok(ev) if ev.version != before => self.status = toast(ev.desc),
            Ok(_) => {}
        }
    }

    /// Turn lasso hits into one journal entry (a batch when several
    /// clusters were lassoed) and select source + new halves afterwards.
    fn apply_lasso_splits(&mut self, splits: Vec<(ClusterId, Vec<u32>)>) {
        let mut cmds = Vec::new();
        let mut sources = Vec::new();
        for (cluster, spike_idx) in splits.into_iter().filter(|(_, s)| !s.is_empty()) {
            sources.push(cluster);
            cmds.push(CurationCommand::Split {
                cluster,
                spike_idx,
                new_cluster: ClusterId(0), // schema field; ClusterIndex auto-allocates
                by: sorrel_data::command::Method::Manual,
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
        // The new halves are the tail of the event, in command order.
        let follow: Follow = Box::new(move |ev| {
            let new = new_ids(ev, sources.len());
            sources
                .iter()
                .zip(new)
                .flat_map(|(&c, &n)| [c, n])
                .collect()
        });
        self.commit_with_follow(cmd, follow);
    }

    fn save(&mut self) {
        match self.save_dir.as_deref().map(Path::to_path_buf) {
            Some(dir) => match self.save_to(&dir) {
                Ok(()) => {
                    // No toast: the status chip carries the save state, and
                    // a stale edit message must not outlive the save.
                    self.status = None;
                    self.last_save = Some(format!("Saved to {}", dir.display()));
                    self.error = None;
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
                    self.close_state = CloseState::Closed;
                }
            }
            (Decision::Confirm, ConfirmAction::Quit) => self.close_state = CloseState::Closed,
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
            Intent::Quit => self.request_close(),
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
            Intent::Undo => self.move_head(CurationCommand::Undo),
            Intent::Redo => self.move_head(CurationCommand::Redo),
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
            if self.mcp.is_some() {
                ui.separator();
                self.agent_indicator(ui);
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
            ViewAction::Decide { id, accept } => self.decide_proposal(id, accept),
            ViewAction::Checkout(node) => {
                self.cancel_jobs();
                match self.host.checkout(Actor::Human, node) {
                    Ok(ev) => self.status = toast(ev.desc),
                    Err(e) => self.error = Some(format!("checkout failed: {e}")),
                }
            }
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
                    by: sorrel_data::command::Method::Algo("gmm".into()),
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
                    if let Some(start) = self.mcp_after_load.take() {
                        if let Err(e) = start(self) {
                            log::warn!("MCP listener not started: {e:#}");
                        }
                    }
                    return false;
                }
                Ok(Err(e)) => {
                    self.loading = None;
                    self.mcp_after_load = None;
                    self.load_error = Some(e.to_string());
                }
                Err(TryRecvError::Disconnected) => {
                    self.loading = None;
                    self.mcp_after_load = None;
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
                        self.close_state = CloseState::Closed;
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
        let _ = self.repaint.set(ctx.clone());
        self.poll_host();
        if let Some((_, at)) = &self.highlight {
            match self.highlight_for.checked_sub(at.elapsed()) {
                Some(left) => ctx.request_repaint_after(left),
                None => self.highlight = None,
            }
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

        let dirty = self.dirty();

        self.show_confirm_dialog(&ctx);
        // The dialog was dismissed (Cancel/Esc): the tab stays open.
        if self.close_state == CloseState::Asking && self.pending_confirm.is_none() {
            self.close_state = CloseState::Open;
        }
        self.show_help_window(&ctx);

        // Status auto-fades (after 5 s) so the top bar doesn't pin a stale
        // "saved" pill from twenty minutes ago.
        if let Some((_, when)) = &self.status {
            match self.toast_for.checked_sub(when.elapsed()) {
                // Wake the UI when the toast should fade so users on idle
                // input still see it disappear.
                Some(left) => ctx.request_repaint_after(left),
                None => self.status = None,
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
                let highlight = self.highlighted().unwrap_or_default().to_vec();
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
                        highlight: &highlight,
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
            let proposals = self.host.proposals();
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
                    proposals: &proposals,
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

impl<P: DataProvider + ApplyPhyLabel> crate::shell::Tab for SorrelApp<P> {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        eframe::App::ui(self, ui, frame);
    }
    fn label(&self) -> String {
        self.save_dir
            .as_deref()
            .and_then(Path::file_name)
            .map_or_else(|| "Sorrel".into(), |s| s.to_string_lossy().into_owned())
    }
    fn dataset(&self) -> &Path {
        self.save_dir.as_deref().unwrap_or(Path::new(""))
    }
    fn dirty(&self) -> bool {
        Self::dirty(self)
    }
    fn window_title(&self) -> String {
        Self::window_title(self)
    }
    fn request_close(&mut self) {
        Self::request_close(self);
    }
    fn close_state(&self) -> CloseState {
        self.close_state
    }
    fn on_exit(&mut self) {
        eframe::App::on_exit(self);
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

    impl HasGeometry for StubProvider {
        fn channel_positions(&self) -> &[[f32; 2]] {
            &[]
        }
    }

    /// With MCP served, an agent's edit (through the host, as a listener
    /// client would) shows up in the toast, a `select` moves the selection,
    /// and the selection is published back for `selection`.
    #[test]
    fn mcp_edits_and_selects_reach_the_window() {
        let (mut app, dir) = make_app();
        app.set_save_dir(dir.path().to_path_buf(), false);
        app.serve_mcp().unwrap();
        assert!(sorrel_data::serve::read_mcp_json(dir.path()).is_some());
        let agent = Actor::Agent {
            client: "codex".into(),
        };
        app.host
            .apply(
                agent,
                CurationCommand::Relabel {
                    cluster: ClusterId(2),
                    op: PhyLabelOp::SetNoise,
                },
            )
            .unwrap();
        let (port, token) = sorrel_data::serve::read_mcp_json(dir.path()).unwrap();
        let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        use std::io::{BufRead, Write};
        writeln!(c, "{{\"token\":\"{token}\"}}").unwrap();
        writeln!(
            c,
            r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"select","arguments":{{"clusters":[1]}}}}}}"#
        )
        .unwrap();
        let mut r = std::io::BufReader::new(c);
        let mut line = String::new();
        r.read_line(&mut line).unwrap(); // ack
        r.read_line(&mut line).unwrap(); // select reply
        app.poll_host();
        assert_eq!(
            app.status.as_ref().map(|s| s.0.as_str()),
            Some("codex: relabel cluster 2 as Noise")
        );
        assert_eq!(app.highlighted(), Some(&[ClusterId(2)][..]));
        // The agent's edit selected its cluster; the later `select` won.
        assert_eq!(app.selection.as_slice(), &[ClusterId(1)]);
        let published = app.mcp.as_ref().unwrap().selection.lock().unwrap().clone();
        assert_eq!(published, [ClusterId(1)]);
    }

    /// Accepting a proposal applies it as a human decision whose split is
    /// made by the proposing agent, and audits it; rejecting a merge leaves
    /// the not-merge memory.
    #[test]
    fn proposal_cards_accept_and_reject_through_the_host() {
        use sorrel_data::proposals::{self, Action, Outcome, Proposal, Source};
        let (mut app, dir) = make_app();
        app.set_save_dir(dir.path().to_path_buf(), false);
        let queue = |action, app: &SorrelApp<StubProvider>| {
            let mut p = Proposal::new(
                Source::Agent,
                action,
                0.8,
                vec!["x".into()],
                Default::default(),
            );
            p.proposer = Some("codex".into());
            proposals::append(dir.path(), &p).unwrap();
            app.host.propose(p.clone());
            p.id
        };
        let split = queue(
            Action::Split {
                cluster: ClusterId(0),
                spike_idx: vec![1],
            },
            &app,
        );
        let merge = queue(
            Action::Merge {
                clusters: vec![ClusterId(1), ClusterId(2)],
            },
            &app,
        );
        app.apply_view_action(ViewAction::Decide {
            id: split,
            accept: true,
        });
        app.apply_view_action(ViewAction::Decide {
            id: merge,
            accept: false,
        });
        assert!(app.host.proposals().is_empty());
        let s = read(&app.session);
        assert_eq!(s.n_clusters(), 4);
        assert_eq!(s.origin_label(ClusterId(3)), "Sorrel[LLM:codex]");
        assert!(s.tags(ClusterId(1)).iter().any(|t| t == "notmerge:2"));
        assert_eq!(app.selection.as_slice(), &[ClusterId(0), ClusterId(3)]);
        drop(s);
        let log = proposals::read_log(dir.path());
        let outcome = |id| log.iter().find(|p| p.id == id).unwrap().outcome;
        assert_eq!(
            (outcome(split), outcome(merge)),
            (Outcome::Accepted, Outcome::Rejected)
        );
    }

    /// The selection after a split comes from the applied event (the ids
    /// the apply allocated), also when several splits go in one batch.
    #[test]
    fn split_follow_uses_the_ids_the_apply_allocated() {
        let (mut app, _d) = make_app();
        app.apply_lasso_splits(vec![(ClusterId(0), vec![0])]);
        assert_eq!(app.selection.as_slice(), &[ClusterId(0), ClusterId(3)]);
        // Cluster 1 still has both spikes (a split must leave one behind).
        assert!(app.commit(CurationCommand::Split {
            cluster: ClusterId(1),
            spike_idx: vec![0],
            new_cluster: ClusterId(u32::MAX),
            by: sorrel_data::command::Method::Manual,
        }));
        assert_eq!(app.selection.as_slice(), &[ClusterId(1), ClusterId(4)]);
    }

    /// A proposal that no longer applies leaves an audit line and no card.
    #[test]
    fn stale_proposal_is_audited_when_it_fails() {
        use sorrel_data::proposals::{self, Action, Outcome, Proposal, Source};
        let (mut app, dir) = make_app();
        app.set_save_dir(dir.path().to_path_buf(), false);
        let p = Proposal::new(
            Source::Agent,
            // Cluster 2 is merged away before the human gets to it.
            Action::Split {
                cluster: ClusterId(2),
                spike_idx: vec![0],
            },
            0.8,
            vec![],
            Default::default(),
        );
        let id = p.id;
        app.commit(CurationCommand::Merge {
            sources: vec![ClusterId(2)],
            target: ClusterId(0),
        });
        proposals::append(dir.path(), &p).unwrap();
        app.host.propose(p);
        app.apply_view_action(ViewAction::Decide { id, accept: true });
        assert!(app.error.is_some(), "the failure is shown");
        assert!(app.host.proposals().is_empty());
        let log = proposals::read_log(dir.path());
        assert_eq!(log[0].outcome, Outcome::Superseded);
        assert_eq!(read(&app.session).n_clusters(), 3, "nothing applied");
    }

    /// A split's indices refer to the membership it was proposed on: after a
    /// merge into that cluster (indices still in range) accepting is refused.
    #[test]
    fn split_proposal_is_refused_after_membership_changes() {
        use sorrel_data::proposals::{self, Action, Outcome, Proposal, Source};
        let (mut app, dir) = make_app();
        app.set_save_dir(dir.path().to_path_buf(), false);
        let p = Proposal::new(
            Source::Agent,
            Action::Split {
                cluster: ClusterId(0),
                spike_idx: vec![0],
            },
            0.8,
            vec![],
            Default::default(),
        );
        let id = p.id;
        proposals::append(dir.path(), &p).unwrap();
        app.host.propose(p);
        assert!(app.commit(CurationCommand::Merge {
            sources: vec![ClusterId(1)],
            target: ClusterId(0),
        }));
        let n = read(&app.session).n_clusters();
        let version = read(&app.session).version();
        app.apply_view_action(ViewAction::Decide { id, accept: true });
        assert!(app.error.is_some(), "the refusal is shown");
        assert_eq!(read(&app.session).version(), version, "nothing applied");
        assert_eq!(read(&app.session).n_clusters(), n);
        let log = proposals::read_log(dir.path());
        assert_eq!(log[0].outcome, Outcome::Superseded);
    }

    /// Accepting checks the proposal's basis under the write lock of the
    /// apply (no MCP edit can land in between), and a refusal there audits
    /// the proposal as superseded.
    #[test]
    fn proposal_accept_checks_under_the_write_lock() {
        use sorrel_data::proposals::{self, Action, Outcome, Proposal, Source};
        let (mut app, dir) = make_app();
        app.set_save_dir(dir.path().to_path_buf(), false);
        let host = Arc::clone(&app.host);
        let ran = std::cell::Cell::new(false);
        let cmd = CurationCommand::Relabel {
            cluster: ClusterId(1),
            op: PhyLabelOp::SetGood,
        };
        assert!(
            app.commit_with_follow_if(cmd, Box::new(|_| Vec::new()), |_| {
                ran.set(true);
                assert!(host.session().try_read().is_err(), "write lock is held");
                Ok(())
            })
        );
        assert!(ran.get());
        let p = Proposal::new(
            Source::Agent,
            Action::Split {
                cluster: ClusterId(0),
                spike_idx: vec![0],
            },
            0.8,
            vec![],
            Default::default(),
        );
        let id = p.id;
        proposals::append(dir.path(), &p).unwrap();
        app.host.propose(p);
        // The basis stamped at queue time no longer matches by accept time.
        assert!(app.commit(CurationCommand::Merge {
            sources: vec![ClusterId(1)],
            target: ClusterId(0),
        }));
        app.apply_view_action(ViewAction::Decide { id, accept: true });
        assert!(app.error.is_some());
        let log = proposals::read_log(dir.path());
        assert_eq!(log[0].outcome, Outcome::Superseded);
    }

    /// `serve_mcp` while the loader owns the session binds nothing; the
    /// listener starts when loading succeeds and never when it fails.
    #[test]
    fn mcp_listener_waits_for_a_successful_load() {
        for ok in [true, false] {
            let dir = tempfile::tempdir().unwrap();
            let (go, wait) = std::sync::mpsc::channel::<()>();
            let mut app = SorrelApp::with_loader(stub_session(dir.path()), label_str, move |_| {
                wait.recv().ok();
                anyhow::ensure!(ok, "load failed");
                Ok(())
            });
            app.set_save_dir(dir.path().to_path_buf(), false);
            app.serve_mcp().unwrap();
            assert!(app.mcp.is_none());
            assert!(sorrel_data::serve::read_mcp_json(dir.path()).is_none());
            go.send(()).unwrap();
            let ctx = egui::Context::default();
            while app.loading.is_some() {
                let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                    app.show_loading(&ctx, ui);
                });
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            assert_eq!(app.mcp.is_some(), ok);
            assert_eq!(sorrel_data::serve::read_mcp_json(dir.path()).is_some(), ok);
        }
    }

    /// A proposal still pending in the audit log is a card again when the
    /// window opens the dataset; a decided one is not.
    #[test]
    fn pending_proposals_come_back_when_the_dataset_opens() {
        use sorrel_data::proposals::{self, Action, Outcome, Proposal, Source};
        let (app, dir) = make_app();
        let new = |c| {
            let p = Proposal::new(
                Source::Agent,
                Action::Accept {
                    cluster: ClusterId(c),
                },
                0.8,
                vec![],
                Default::default(),
            );
            proposals::append(dir.path(), &p).unwrap();
            p.id
        };
        let (pending, decided) = (new(0), new(1));
        proposals::record_outcome(dir.path(), decided, Outcome::Accepted).unwrap();
        drop(app);
        let (mut app, _d) = make_app();
        app.set_save_dir(dir.path().to_path_buf(), false);
        let ids: Vec<u64> = app.host.proposals().iter().map(|p| p.id).collect();
        assert_eq!(ids, [pending]);
    }

    /// A History click checks out another branch through the host and
    /// names the node it went back to.
    #[test]
    fn history_checkout_crosses_branches() {
        let (mut app, _d) = make_app();
        let relabel = |c, op| CurationCommand::Relabel {
            cluster: ClusterId(c),
            op,
        };
        app.commit(relabel(1, PhyLabelOp::SetGood));
        app.dispatch_intent(Intent::Undo);
        app.commit(relabel(1, PhyLabelOp::SetNoise));
        assert_eq!(label(&app, 1), Some(3));
        app.apply_view_action(ViewAction::Checkout(Some(0)));
        assert_eq!(label(&app, 1), Some(1), "the other branch");
        assert_eq!(
            app.status.as_ref().map(|s| s.0.as_str()),
            Some("back to: relabel cluster 1 as Good")
        );
        app.apply_view_action(ViewAction::Checkout(Some(9)));
        assert!(app.error.is_some());
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
            "the head is the saved node after a successful save"
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
        let mut earlier = stub_session(dir.path());
        earlier
            .dispatch(CurationCommand::Relabel {
                cluster: ClusterId(1),
                op: PhyLabelOp::SetGood,
            })
            .unwrap();
        drop(earlier);
        let mut app =
            SorrelApp::with_loader(stub_session(dir.path()), label_str, |s| s.replay_journal());
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

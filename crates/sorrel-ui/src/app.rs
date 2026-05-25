use crate::intent::{from_key_in_context, Intent};
use crate::selection::SelectionSet;
use crate::views::{
    amplitude_view, cluster_statistics_view, contamination_over_time_view, correlogram_grid_view,
    drift_map_view, feature_view, firing_rate_view, isi_view, probe_view, quality_view,
    raster_view, selection_summary, similarity_view, suggest_view, template_view, waveform_view,
    FeatureViewState, RasterCache, SuggestAction, SuggestCache, WaveformCache,
};
use crate::widgets::{cluster_table, trace_view, ClusterTableState};
use eframe::egui;
use sorrel_data::session::ApplyPhyLabel;
use sorrel_data::{export_qc, save_to_phy, CurationCommand, Session};
use sorrel_gpu::GpuContext;
use sorrel_io::{ClusterId, DataProvider};
use sorrel_render::{GpuTracePreproc, TracePreproc};
use std::path::{Path, PathBuf};

/// Short human description of a curation command — for status toasts so
/// undo/redo say *what* they reverted instead of a bare "undone".
fn describe_command(cmd: &CurationCommand) -> String {
    use sorrel_data::PhyLabelOp;
    match cmd {
        CurationCommand::Relabel { cluster, op } => {
            let label = match op {
                PhyLabelOp::SetGood => "Good",
                PhyLabelOp::SetMua => "MUA",
                PhyLabelOp::SetNoise => "Noise",
                PhyLabelOp::SetUnsorted => "Unsorted",
            };
            format!("relabel cluster {} → {}", cluster.0, label)
        }
        CurationCommand::Merge { sources, target } => {
            format!("merge {} cluster(s) → {}", sources.len(), target.0)
        }
        CurationCommand::Split {
            cluster, spike_idx, ..
        } => {
            format!("split cluster {} ({} spikes)", cluster.0, spike_idx.len())
        }
        CurationCommand::Note { cluster, .. } => format!("note on cluster {}", cluster.0),
        CurationCommand::Batch { children } => format!("batch of {} ops", children.len()),
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

/// Which cluster-summary view fills the central panel. Persisted in the app
/// so switching tabs doesn't churn other state.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum CentralTab {
    Summary,
    Waveforms,
    Templates,
    Amplitudes,
    Isi,
    Ccg,
    Features,
    Raster,
    Rate,
    Probe,
    Similar,
    Stats,
    Quality,
    Suggest,
    DriftMap,
    ContamTime,
}

/// Generic eframe app. The whole window is monomorphised against `P`.
pub struct SorrelApp<P: DataProvider + ApplyPhyLabel> {
    session: Session<P>,
    label_str: fn(&P::Label) -> &'static str,
    selection: SelectionSet,
    window_start: u64,
    window_len: u32,
    target_points: usize,
    error: Option<String>,
    /// Directory to write `spike_clusters.npy` / `cluster_group.tsv` into
    /// when the user triggers a save. Cmd/Ctrl-S is a no-op when unset.
    save_dir: Option<PathBuf>,
    /// Last "save successful" status message + the time it was set, so the
    /// top bar can fade it out after a few seconds rather than pinning it.
    status: Option<(String, std::time::Instant)>,
    /// Snapshot of `(history_len, redo_len)` at the last successful save. The
    /// title bar shows a "•" dirty marker when current state differs.
    saved_state: (usize, usize),
    /// Optional pre-processing applied to the trace window before LTTB.
    trace_cfg: TracePreproc,
    central_tab: CentralTab,
    /// Probe geometry — set by the binary when the backend exposes it.
    /// Empty means the WaveformView falls back to single-channel layout.
    channel_positions: Vec<[f32; 2]>,
    waveform_cache: WaveformCache,
    table_state: ClusterTableState,
    feature_state: FeatureViewState,
    raster_cache: RasterCache,
    suggest_cache: SuggestCache,
    /// GPU compute pipelines for trace CMR + HP filtering. `None` until the
    /// app is installed onto an eframe wgpu render state.
    gpu_preproc: Option<GpuTracePreproc>,
    /// A destructive action awaiting user confirmation. The Suggest tab's
    /// "apply high-confidence merges" can apply dozens of merges at once;
    /// staging it here lets the user back out before anything hits the
    /// journal.
    pending_confirm: Option<PendingConfirm>,
}

/// A staged action that needs explicit user approval before it is
/// dispatched. Currently only the bulk-merge case lands here, but the
/// machinery is general — wire any other "click could destroy work" path
/// through this.
#[derive(Debug)]
struct PendingConfirm {
    /// Short headline shown in the dialog title.
    title: String,
    /// Human prose explaining what will happen. Multi-line.
    body: String,
    /// The command to dispatch on confirm.
    command: CurationCommand,
    /// Status message to show on success — typically explains how to undo.
    on_success: String,
}

impl<P: DataProvider + ApplyPhyLabel> SorrelApp<P> {
    pub fn new(session: Session<P>, label_str: fn(&P::Label) -> &'static str) -> Self {
        let window_len = (session.provider.sample_rate() as u32).max(1) / 10; // 100 ms
        let selection = if session.n_clusters() > 0 {
            SelectionSet::single(ClusterId(0))
        } else {
            SelectionSet::new()
        };
        let saved_state = (session.history_len(), session.redo_len());
        Self {
            session,
            saved_state,
            label_str,
            selection,
            window_start: 0,
            window_len,
            target_points: 1024,
            error: None,
            save_dir: None,
            status: None,
            trace_cfg: TracePreproc::Off,
            central_tab: CentralTab::Summary,
            channel_positions: Vec::new(),
            waveform_cache: WaveformCache::new(),
            table_state: ClusterTableState::default(),
            feature_state: FeatureViewState::default(),
            raster_cache: RasterCache::new(),
            suggest_cache: SuggestCache::new(),
            gpu_preproc: None,
            pending_confirm: None,
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
    /// typically points this at the kilosort root.
    pub fn set_save_dir(&mut self, dir: PathBuf) {
        self.save_dir = Some(dir);
    }

    /// Supply probe geometry so the WaveformView can render in probe layout
    /// (top-N channels by amplitude positioned at their `(x, y)` coords).
    /// Length must equal `provider.n_channels()`; mismatched lengths are
    /// ignored and the view falls back to single-channel display.
    pub fn set_channel_positions(&mut self, positions: Vec<[f32; 2]>) {
        self.channel_positions = positions;
    }

    /// Render the modal-style confirmation dialog if a destructive action is
    /// staged. Confirm dispatches it; Cancel discards. Either choice clears
    /// `pending_confirm`. Suggest-tab caches are invalidated on success
    /// because batch merges change the cluster set.
    fn show_confirm_dialog(&mut self, ctx: &egui::Context) {
        let Some(pending) = self.pending_confirm.as_ref() else {
            return;
        };
        let title = pending.title.clone();
        let body = pending.body.clone();
        let mut decision: Option<bool> = None;
        egui::Window::new(&title)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(body);
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        decision = Some(false);
                    }
                    if ui
                        .add(
                            egui::Button::new("Confirm").fill(egui::Color32::from_rgb(80, 130, 60)),
                        )
                        .clicked()
                    {
                        decision = Some(true);
                    }
                });
            });
        match decision {
            Some(true) => {
                let pending = self.pending_confirm.take().expect("checked above");
                match self.session.dispatch(pending.command) {
                    Ok(()) => {
                        self.status = Some((pending.on_success, std::time::Instant::now()));
                        self.suggest_cache.invalidate();
                        self.waveform_cache = WaveformCache::new();
                        self.raster_cache.invalidate();
                    }
                    Err(e) => {
                        self.error = Some(format!("batch dispatch failed: {e}"));
                    }
                }
            }
            Some(false) => {
                self.pending_confirm = None;
            }
            None => {}
        }
    }

    fn dispatch_intent(&mut self, intent: Intent) {
        let n = self.session.n_clusters();
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
                    self.selection.extend_to(c);
                }
            }
            Intent::Relabel(op) => {
                // Apply to every cluster in the active selection. Each emits
                // its own journal record + undo entry; phy behaves the same.
                let targets: Vec<ClusterId> = self.selection.iter().copied().collect();
                for c in targets {
                    if let Err(e) = self
                        .session
                        .dispatch(CurationCommand::Relabel { cluster: c, op })
                    {
                        self.error = Some(format!("journal write failed: {e}"));
                        break;
                    }
                }
            }
            Intent::Undo => {
                // Peek the about-to-be-undone command first so the status
                // toast can name it ("undid: relabel cluster 12 → MUA")
                // rather than a generic "undo".
                let desc = self.session.history_commands().last().map(describe_command);
                if let Err(e) = self.session.dispatch(CurationCommand::Undo) {
                    self.error = Some(format!("journal write failed: {e}"));
                } else if let Some(d) = desc {
                    self.status = Some((format!("undid: {d}"), std::time::Instant::now()));
                }
            }
            Intent::Redo => {
                if let Err(e) = self.session.dispatch(CurationCommand::Redo) {
                    self.error = Some(format!("journal write failed: {e}"));
                } else if let Some(cmd) = self.session.history_commands().last() {
                    let d = describe_command(cmd);
                    self.status = Some((format!("redid: {d}"), std::time::Instant::now()));
                }
            }
            Intent::Save => match self.save_dir.as_deref() {
                Some(dir) => match save_to_phy(&self.session, dir, self.label_str) {
                    Ok(()) => {
                        if let Err(err) = self.session.invalidate_dataset_caches() {
                            log::warn!("cache invalidation after save failed: {err}");
                        }
                        self.status = Some((
                            format!("saved → {}", dir.display()),
                            std::time::Instant::now(),
                        ));
                        self.error = None;
                        self.saved_state = (self.session.history_len(), self.session.redo_len());
                    }
                    Err(e) => {
                        self.error = Some(format!("save failed: {e}"));
                    }
                },
                None => {
                    self.error = Some("save target not configured (no kilosort directory)".into());
                }
            },
            Intent::NextCluster => {
                self.selection.bump(1, n);
            }
            Intent::PrevCluster => {
                self.selection.bump(-1, n);
            }
            Intent::PageBack => {
                self.window_start = self.window_start.saturating_sub(self.window_len as u64);
            }
            Intent::PageForward => {
                let max = self
                    .session
                    .provider
                    .n_samples()
                    .0
                    .saturating_sub(self.window_len as u64);
                self.window_start = (self.window_start + self.window_len as u64).min(max);
            }
        }
    }
}

impl<P: DataProvider + ApplyPhyLabel> eframe::App for SorrelApp<P> {
    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = root_ui.ctx().clone();
        // Input. Suppress single-letter intents while a text widget has focus
        // so typing in a filter box doesn't relabel selected clusters; global
        // shortcuts (Cmd/Ctrl-S/Z/Shift-Z) still fire.
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
        for it in input_intents {
            self.dispatch_intent(it);
        }

        // Confirmation dialog for staged destructive actions. Rendered first
        // so it sits visually above the panels; any other input this frame
        // is still processed (the dialog is non-modal) but the user has a
        // clear escape hatch.
        self.show_confirm_dialog(&ctx);

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
        let dirty = (self.session.history_len(), self.session.redo_len()) != self.saved_state;

        egui::Panel::top("top").show_inside(root_ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading(if dirty { "Sorrel •" } else { "Sorrel" })
                    .on_hover_text(if dirty {
                        "Unsaved curation — Cmd/Ctrl-S writes phy artefacts."
                    } else {
                        "All curation saved."
                    });
                ui.separator();
                ui.label(format!(
                    "{} clusters · {} channels · {:.1} kHz",
                    self.session.n_clusters(),
                    self.session.provider.n_channels(),
                    self.session.provider.sample_rate() / 1000.0
                ));
                ui.separator();
                ui.label(format!("{} selected", self.selection.len()));
                if let Some(err) = &self.error {
                    ui.separator();
                    ui.colored_label(egui::Color32::RED, err);
                } else if let Some((status, _)) = &self.status {
                    ui.separator();
                    ui.colored_label(egui::Color32::LIGHT_GREEN, status);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("Export QC")
                        .on_hover_text(
                            "Write cluster_qc.tsv + cluster_qc.json into the save directory \
                             (SpikeInterface-compatible).",
                        )
                        .clicked()
                    {
                        match self.save_dir.as_deref() {
                            Some(dir) => match export_qc(&self.session, dir) {
                                Ok(n) => {
                                    self.status = Some((
                                        format!("exported QC for {n} clusters → {}", dir.display()),
                                        std::time::Instant::now(),
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
                    // Show the configured save target so users know where
                    // Cmd/Ctrl-S and Export QC will write.
                    if let Some(dir) = self.save_dir.as_deref() {
                        let label = format!("→ {}", short_path(dir));
                        ui.label(label).on_hover_text(format!(
                            "Save target: {}\nWrites spike_clusters.npy, cluster_group.tsv \
                             (Cmd/Ctrl-S) and cluster_qc.tsv/json (Export QC).",
                            dir.display()
                        ));
                    } else {
                        ui.colored_label(egui::Color32::from_rgb(220, 160, 0), "no save target")
                            .on_hover_text(
                                "Open sorrel with a kilosort directory to enable saving.",
                            );
                    }
                });
            });
        });

        egui::Panel::left("clusters")
            .resizable(true)
            .default_size(260.0)
            .show_inside(root_ui, |ui| {
                ui.heading("Clusters");
                cluster_table(
                    ui,
                    &self.session,
                    &mut self.selection,
                    &mut self.table_state,
                    self.label_str,
                );
            });

        egui::Panel::bottom("trace")
            .resizable(true)
            .default_size(280.0)
            .show_inside(root_ui, |ui| {
                ui.horizontal(|ui| {
                    let sr = self.session.provider.sample_rate().max(1.0);
                    let window_ms = self.window_len as f32 * 1000.0 / sr;
                    ui.label(format!(
                        "t = {:.3} s · window {:.0} ms",
                        self.window_start as f64 / sr as f64,
                        window_ms
                    ))
                    .on_hover_text(format!(
                        "Window start sample: {}\nWindow length samples: {}\n\
                         Sample rate: {:.1} kHz",
                        self.window_start,
                        self.window_len,
                        sr / 1000.0
                    ));
                    ui.separator();
                    // Adjustable window length, 5–500 ms. The default
                    // (sr/10 = 100 ms) was previously baked into `new()`.
                    let mut new_ms = window_ms.round();
                    let resp = ui.add(
                        egui::Slider::new(&mut new_ms, 5.0..=500.0)
                            .text("ms")
                            .integer()
                            .clamping(egui::SliderClamping::Always),
                    );
                    if resp.changed() {
                        self.window_len = ((new_ms / 1000.0) * sr).round().max(1.0) as u32;
                    }
                    ui.separator();
                    let mut hp_on = self.trace_cfg.hp().is_some();
                    let mut cmr_on = self.trace_cfg.has_cmr();
                    let hp_changed = ui
                        .checkbox(&mut hp_on, "HP 300 Hz")
                        .on_hover_text("High-pass filter the trace before display.")
                        .changed();
                    let cmr_changed = ui
                        .checkbox(&mut cmr_on, "CMR")
                        .on_hover_text("Common-median referencing across channels.")
                        .changed();
                    if hp_changed || cmr_changed {
                        let hp = hp_on.then_some(TracePreproc::HP_DEFAULT_HZ);
                        self.trace_cfg = TracePreproc::from_flags(hp, cmr_on);
                    }
                });
                trace_view(
                    ui,
                    &self.session,
                    self.window_start,
                    self.window_len,
                    self.target_points,
                    self.trace_cfg,
                    self.gpu_preproc.as_ref(),
                );
            });

        egui::CentralPanel::default().show_inside(root_ui, |ui| {
            // Tab switcher.
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.central_tab, CentralTab::Summary, "Summary");
                ui.selectable_value(&mut self.central_tab, CentralTab::Waveforms, "Waveforms");
                ui.selectable_value(&mut self.central_tab, CentralTab::Templates, "Templates");
                ui.selectable_value(&mut self.central_tab, CentralTab::Amplitudes, "Amplitudes");
                ui.selectable_value(&mut self.central_tab, CentralTab::Isi, "ISI");
                ui.selectable_value(&mut self.central_tab, CentralTab::Ccg, "CCG");
                ui.selectable_value(&mut self.central_tab, CentralTab::Features, "Features");
                ui.selectable_value(&mut self.central_tab, CentralTab::Raster, "Raster");
                ui.selectable_value(&mut self.central_tab, CentralTab::Rate, "Rate");
                ui.selectable_value(&mut self.central_tab, CentralTab::Probe, "Probe");
                ui.selectable_value(&mut self.central_tab, CentralTab::Similar, "Similar");
                ui.selectable_value(&mut self.central_tab, CentralTab::Stats, "Stats");
                ui.selectable_value(&mut self.central_tab, CentralTab::Quality, "Quality");
                ui.selectable_value(&mut self.central_tab, CentralTab::Suggest, "Suggest");
                ui.selectable_value(&mut self.central_tab, CentralTab::DriftMap, "Drift");
                ui.selectable_value(&mut self.central_tab, CentralTab::ContamTime, "Contam·time");
            });
            ui.separator();

            let selected: Vec<ClusterId> = self.selection.iter().copied().collect();
            match self.central_tab {
                CentralTab::Summary => {
                    selection_summary(ui, &self.session, &selected, self.label_str);
                    ui.separator();
                    ui.label(
                        "Keys: G/M/N/U relabel · J/K next/prev · H/L pan · \
                         Cmd/Ctrl-Z undo · Cmd/Ctrl-Shift-Z redo · Cmd/Ctrl-S save · \
                         click select · Cmd/Ctrl-click toggle · Shift-click extend",
                    );
                }
                CentralTab::Waveforms => {
                    waveform_view(
                        ui,
                        &self.session,
                        &selected,
                        &self.channel_positions,
                        &mut self.waveform_cache,
                    );
                }
                CentralTab::Amplitudes => {
                    amplitude_view(ui, &self.session, &selected);
                }
                CentralTab::Isi => {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        isi_view(ui, &self.session, &selected);
                    });
                }
                CentralTab::Ccg => {
                    correlogram_grid_view(ui, &self.session, &selected);
                }
                CentralTab::Raster => {
                    if let Some(sample) = raster_view(ui, &self.session, &mut self.raster_cache) {
                        // Click → seek the trace window to that time.
                        let max = self
                            .session
                            .provider
                            .n_samples()
                            .0
                            .saturating_sub(self.window_len as u64);
                        self.window_start = sample.min(max);
                    }
                }
                CentralTab::Features => {
                    let splits =
                        feature_view(ui, &self.session, &selected, &mut self.feature_state);
                    if let Some(splits) = splits {
                        for (cluster, spike_idx) in splits {
                            if spike_idx.is_empty() {
                                continue;
                            }
                            let cmd = CurationCommand::Split {
                                cluster,
                                spike_idx,
                                new_cluster: ClusterId(0), // honoured to keep schema; ClusterIndex auto-allocates
                            };
                            if let Err(e) = self.session.dispatch(cmd) {
                                self.error = Some(format!("split failed: {e}"));
                                break;
                            }
                        }
                        // Splits change the cluster set; clear caches that
                        // key on the selection.
                        self.waveform_cache = WaveformCache::new();
                    }
                }
                CentralTab::Rate => {
                    firing_rate_view(ui, &self.session, &selected);
                }
                CentralTab::Probe => {
                    probe_view(
                        ui,
                        &self.session,
                        &selected,
                        &self.channel_positions,
                        &self.waveform_cache,
                    );
                }
                CentralTab::Templates => {
                    template_view(ui, &self.session, &selected);
                }
                CentralTab::Similar => {
                    similarity_view(ui, &self.session, &selected);
                }
                CentralTab::Stats => {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        cluster_statistics_view(ui, &self.session);
                    });
                }
                CentralTab::Quality => {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        quality_view(ui, &self.session, &selected);
                    });
                }
                CentralTab::DriftMap => {
                    drift_map_view(ui, &self.session, &selected);
                }
                CentralTab::ContamTime => {
                    contamination_over_time_view(ui, &self.session, &selected);
                }
                CentralTab::Suggest => {
                    let action = egui::ScrollArea::vertical()
                        .show(ui, |ui| {
                            suggest_view(ui, &self.session, &mut self.suggest_cache, self.label_str)
                        })
                        .inner;
                    if let Some(act) = action {
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
                                let cmd = CurationCommand::Merge { sources, target };
                                if let Err(e) = self.session.dispatch(cmd) {
                                    self.error = Some(format!("merge failed: {e}"));
                                } else {
                                    self.suggest_cache.invalidate();
                                    self.waveform_cache = WaveformCache::new();
                                    self.raster_cache.invalidate();
                                }
                            }
                            SuggestAction::ApplyGmmSplit { cluster, spike_idx } => {
                                let cmd = CurationCommand::Split {
                                    cluster,
                                    spike_idx,
                                    new_cluster: ClusterId(0),
                                };
                                if let Err(e) = self.session.dispatch(cmd) {
                                    self.error = Some(format!("split failed: {e}"));
                                } else {
                                    self.suggest_cache.invalidate();
                                    self.waveform_cache = WaveformCache::new();
                                    self.raster_cache.invalidate();
                                }
                            }
                            SuggestAction::ApplyHighConfidenceMerges { threshold } => {
                                // Bulk merges can collapse dozens of clusters
                                // in one click — gate the action behind a
                                // confirm dialog rather than dispatching here.
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
                                    self.status = Some((
                                        "no merges met the threshold".into(),
                                        std::time::Instant::now(),
                                    ));
                                } else {
                                    let n = children.len();
                                    let Some(batch) = CurationCommand::batch(children) else {
                                        self.error = Some(
                                            "internal error: batch contained nested commands"
                                                .into(),
                                        );
                                        return;
                                    };
                                    self.pending_confirm = Some(PendingConfirm {
                                        title: format!("Apply {n} merges?"),
                                        body: format!(
                                            "{n} cluster pair(s) at or above similarity \
                                             threshold {threshold:.2} will be merged as a \
                                             single batch. One Undo reverts the whole batch."
                                        ),
                                        command: batch,
                                        on_success: format!(
                                            "applied {n} merges (one Undo reverts all)"
                                        ),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    fn on_exit(&mut self) {
        if let Err(err) = self.session.invalidate_dataset_caches() {
            log::warn!("cache invalidation on exit failed: {err}");
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

    fn make_app() -> (SorrelApp<StubProvider>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let provider = StubProvider {
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(30)],
                vec![SampleIndex(40), SampleIndex(60)],
                vec![SampleIndex(80)],
            ],
        };
        let journal = Journal::open_or_create(&dir.path().join("j.sqlite"), 0xCAFE, 3).unwrap();
        let session = Session::new(provider, journal);
        (SorrelApp::new(session, label_str), dir)
    }

    #[test]
    fn relabel_intent_mutates_session_and_marks_dirty() {
        let (mut app, _d) = make_app();
        // Start clean.
        let initial = (app.session.history_len(), app.session.redo_len());
        assert_eq!(initial, app.saved_state, "fresh session should be clean");

        app.dispatch_intent(Intent::Relabel(PhyLabelOp::SetGood));

        assert_eq!(app.session.label(ClusterId(0)), Some(1));
        assert_ne!(
            (app.session.history_len(), app.session.redo_len()),
            app.saved_state,
            "post-relabel state should diverge from saved snapshot"
        );
    }

    #[test]
    fn save_without_save_dir_records_an_error() {
        let (mut app, _d) = make_app();
        app.dispatch_intent(Intent::Save);
        assert!(app.error.is_some(), "save without save_dir should error");
        assert!(app.status.is_none());
    }

    #[test]
    fn save_with_save_dir_clears_dirty_and_sets_status() {
        let (mut app, dir) = make_app();
        app.set_save_dir(dir.path().to_path_buf());
        app.dispatch_intent(Intent::Relabel(PhyLabelOp::SetMua));
        assert_ne!(
            (app.session.history_len(), app.session.redo_len()),
            app.saved_state
        );

        app.dispatch_intent(Intent::Save);

        assert!(app.error.is_none(), "save should succeed: {:?}", app.error);
        assert!(app.status.is_some(), "save should produce a status toast");
        assert_eq!(
            (app.session.history_len(), app.session.redo_len()),
            app.saved_state,
            "saved_state should snap to current after a successful save"
        );
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
    fn navigation_intents_walk_the_selection() {
        let (mut app, _d) = make_app();
        // Selection starts at cluster 0 (made by SorrelApp::new).
        assert!(app.selection.iter().any(|c| c.0 == 0));
        app.dispatch_intent(Intent::NextCluster);
        assert!(app.selection.iter().any(|c| c.0 == 1));
        app.dispatch_intent(Intent::NextCluster);
        assert!(app.selection.iter().any(|c| c.0 == 2));
        // Walks past the last cluster should clamp, not panic.
        app.dispatch_intent(Intent::NextCluster);
        app.dispatch_intent(Intent::PrevCluster);
        // Just assert we still have *some* selection and didn't go negative.
        assert!(!app.selection.is_empty());
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
    fn pending_confirm_does_not_mutate_until_accepted() {
        // Stage a batch merge directly (bypassing the Suggest tab UI which
        // requires a real frame). The session must stay clean while the
        // confirm is pending; accepting it dispatches and bumps history.
        let (mut app, _d) = make_app();
        let baseline = (app.session.history_len(), app.session.redo_len());
        let batch = CurationCommand::batch(vec![CurationCommand::Merge {
            sources: vec![ClusterId(0)],
            target: ClusterId(2),
        }])
        .unwrap();
        app.pending_confirm = Some(PendingConfirm {
            title: "Apply 1 merges?".into(),
            body: "test".into(),
            command: batch,
            on_success: "applied 1 merges".into(),
        });

        // Stash-only — no dispatch yet.
        assert_eq!(
            (app.session.history_len(), app.session.redo_len()),
            baseline,
            "session must not change while a confirm is pending"
        );

        // Simulate the "Confirm" branch of show_confirm_dialog. We can't
        // drive egui without a real Context here, so dispatch the staged
        // command directly the same way the dialog handler does.
        let pending = app.pending_confirm.take().unwrap();
        app.session.dispatch(pending.command).unwrap();
        assert!(app.session.history_len() > baseline.0);
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

use crate::intent::{from_key, Intent};
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
use sorrel_io::{ClusterId, DataProvider};
use sorrel_gpu::GpuContext;
use sorrel_render::{GpuTracePreproc, TraceConfig};
use std::path::PathBuf;

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
    /// Last "save successful" status message.
    status: Option<String>,
    /// Optional pre-processing applied to the trace window before LTTB.
    trace_cfg: TraceConfig,
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
}

impl<P: DataProvider + ApplyPhyLabel> SorrelApp<P> {
    pub fn new(session: Session<P>, label_str: fn(&P::Label) -> &'static str) -> Self {
        let window_len = (session.provider.sample_rate() as u32).max(1) / 10; // 100 ms
        let selection = if session.n_clusters() > 0 {
            SelectionSet::single(0)
        } else {
            SelectionSet::new()
        };
        Self {
            session,
            label_str,
            selection,
            window_start: 0,
            window_len,
            target_points: 1024,
            error: None,
            save_dir: None,
            status: None,
            trace_cfg: TraceConfig::default(),
            central_tab: CentralTab::Summary,
            channel_positions: Vec::new(),
            waveform_cache: WaveformCache::new(),
            table_state: ClusterTableState::default(),
            feature_state: FeatureViewState::default(),
            raster_cache: RasterCache::new(),
            suggest_cache: SuggestCache::new(),
            gpu_preproc: None,
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

    fn dispatch_intent(&mut self, intent: Intent) {
        let n = self.session.n_clusters();
        match intent {
            Intent::SelectCluster(c) => {
                if c < n {
                    self.selection.replace(c);
                }
            }
            Intent::ToggleCluster(c) => {
                if c < n {
                    self.selection.toggle(c);
                }
            }
            Intent::ExtendCluster(c) => {
                if c < n {
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
                if let Err(e) = self.session.dispatch(CurationCommand::Undo) {
                    self.error = Some(format!("journal write failed: {e}"));
                }
            }
            Intent::Redo => {
                if let Err(e) = self.session.dispatch(CurationCommand::Redo) {
                    self.error = Some(format!("journal write failed: {e}"));
                }
            }
            Intent::Save => match self.save_dir.as_deref() {
                Some(dir) => match save_to_phy(&self.session, dir, self.label_str) {
                    Ok(()) => {
                        self.status = Some(format!("saved → {}", dir.display()));
                        self.error = None;
                    }
                    Err(e) => {
                        self.error = Some(format!("save failed: {e}"));
                    }
                },
                None => {
                    self.error =
                        Some("save target not configured (no kilosort directory)".into());
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
                    .saturating_sub(self.window_len as u64);
                self.window_start = (self.window_start + self.window_len as u64).min(max);
            }
        }
    }
}

impl<P: DataProvider + ApplyPhyLabel> eframe::App for SorrelApp<P> {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Input.
        let input_intents: Vec<Intent> = ctx.input(|i| {
            i.events
                .iter()
                .filter_map(|ev| match ev {
                    egui::Event::Key {
                        key,
                        pressed: true,
                        modifiers,
                        ..
                    } => from_key(*key, *modifiers),
                    _ => None,
                })
                .collect()
        });
        for it in input_intents {
            self.dispatch_intent(it);
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Sorrel");
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
                    ui.colored_label(egui::Color32::RED, err);
                } else if let Some(status) = &self.status {
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
                                    self.status = Some(format!(
                                        "exported QC for {n} clusters → {}",
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
                });
            });
        });

        egui::SidePanel::left("clusters")
            .resizable(true)
            .default_width(260.0)
            .show(ctx, |ui| {
                ui.heading("Clusters");
                cluster_table(
                    ui,
                    &self.session,
                    &mut self.selection,
                    &mut self.table_state,
                    self.label_str,
                );
            });

        egui::TopBottomPanel::bottom("trace")
            .resizable(true)
            .default_height(280.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!(
                        "trace @ {} samples, len {}",
                        self.window_start, self.window_len
                    ));
                    ui.separator();
                    let mut hp_on = self.trace_cfg.hp_cutoff_hz.is_some();
                    if ui.checkbox(&mut hp_on, "HP 300 Hz").changed() {
                        self.trace_cfg.hp_cutoff_hz = if hp_on { Some(300.0) } else { None };
                    }
                    ui.checkbox(&mut self.trace_cfg.cmr, "CMR");
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

        egui::CentralPanel::default().show(ctx, |ui| {
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
                            .saturating_sub(self.window_len as u64);
                        self.window_start = sample.min(max);
                    }
                }
                CentralTab::Features => {
                    let splits = feature_view(
                        ui,
                        &self.session,
                        &selected,
                        &mut self.feature_state,
                    );
                    if let Some(splits) = splits {
                        for (cluster, spike_idx) in splits {
                            if spike_idx.is_empty() {
                                continue;
                            }
                            let cmd = CurationCommand::Split {
                                cluster,
                                spike_idx,
                                new_cluster: 0, // honoured to keep schema; ClusterIndex auto-allocates
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
                            suggest_view(
                                ui,
                                &self.session,
                                &mut self.suggest_cache,
                                self.label_str,
                            )
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
                                    new_cluster: 0,
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
                                    self.status = Some("no merges met the threshold".into());
                                } else {
                                    let n = children.len();
                                    let Some(batch) = CurationCommand::batch(children) else {
                                        self.error = Some(
                                            "internal error: batch contained nested commands"
                                                .into(),
                                        );
                                        return;
                                    };
                                    if let Err(e) = self.session.dispatch(batch) {
                                        self.error = Some(format!("batch merge failed: {e}"));
                                    } else {
                                        self.status = Some(format!(
                                            "applied {n} merges (one Undo reverts all)",
                                        ));
                                        self.suggest_cache.invalidate();
                                        self.waveform_cache = WaveformCache::new();
                                        self.raster_cache.invalidate();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }
}

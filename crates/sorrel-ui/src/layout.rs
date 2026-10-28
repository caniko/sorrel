//! Dockable layout. Every view is a [`Pane`] in an [`egui_tiles::Tree`], so
//! the curator can put waveforms, correlograms and features side by side
//! (phy's workflow) instead of flipping through one tab at a time. The tree
//! is plain data: it serialises for the per-dataset prefs file.

use crate::amplitude_view::{amplitude_view, AmpAction, AmplitudeViewState};
use crate::cluster_scatter::{cluster_scatter_view, ClusterScatterState};
use crate::feature_view::{feature_view, FeatureViewState};
use crate::jobs::JobCtx;
use crate::similarity_view::{similarity_view, SimilarityPane};
use crate::spike_attr_view::{spike_attr_view, SpikeAttrState};
use crate::template_feature_view::{template_feature_view, TemplateFeatureState};
use crate::template_grid::{template_grid_view, TemplateGridCache};
use crate::theme::selection_colour;
use crate::trace_image::{trace_image_view, TraceImageCache};
use crate::views::{
    cluster_statistics_view_for, contamination_over_time_view, correlogram_grid_view,
    drift_map_view, firing_rate_view, isi_view, probe_view, psth_view, quality_view, raster_view,
    selection_summary, suggest_view, waveform_view, waveform_window_samples, CcgCache, PsthCache,
    PsthState, QualityCache, RasterCache, RasterClick, SuggestAction, SuggestCache, WaveformCache,
};
use crate::widgets::{trace_view, TraceCache, TraceMarker};
use egui::Ui;
use egui_tiles::{
    Behavior, Container, Linear, LinearDir, SimplificationOptions, Tile, TileId, Tiles, Tree,
    UiResponse,
};
use serde::{Deserialize, Serialize};
use sorrel_data::{Session, SpikeDepths};
use sorrel_io::{ClusterId, DataProvider, Events};
use sorrel_render::{GpuTracePreproc, TracePreproc};

/// One dockable view.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Pane {
    Trace,
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
    Psth,
    ClusterScatter,
    TemplateFeatures,
    TraceImage,
    SpikeData,
}

impl Pane {
    pub const ALL: [Pane; 22] = [
        Pane::Trace,
        Pane::Summary,
        Pane::Waveforms,
        Pane::Templates,
        Pane::Amplitudes,
        Pane::Isi,
        Pane::Ccg,
        Pane::Features,
        Pane::Raster,
        Pane::Rate,
        Pane::Probe,
        Pane::Similar,
        Pane::Stats,
        Pane::Quality,
        Pane::Suggest,
        Pane::DriftMap,
        Pane::ContamTime,
        Pane::Psth,
        Pane::ClusterScatter,
        Pane::TemplateFeatures,
        Pane::TraceImage,
        Pane::SpikeData,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Pane::Trace => "Trace",
            Pane::Summary => "Summary",
            Pane::Waveforms => "Waveforms",
            Pane::Templates => "Templates",
            Pane::Amplitudes => "Amplitudes",
            Pane::Isi => "ISI",
            Pane::Ccg => "CCG",
            Pane::Features => "Features",
            Pane::Raster => "Raster",
            Pane::Rate => "Rate",
            Pane::Probe => "Probe",
            Pane::Similar => "Similar",
            Pane::Stats => "Stats",
            Pane::Quality => "Quality",
            Pane::Suggest => "Suggest",
            Pane::DriftMap => "Drift",
            Pane::ContamTime => "Contam·time",
            Pane::Psth => "PSTH",
            Pane::ClusterScatter => "Scatter",
            Pane::TemplateFeatures => "Tmpl features",
            Pane::TraceImage => "Trace image",
            Pane::SpikeData => "Spike data",
        }
    }
}

const TREE_ID: &str = "sorrel-layout";

/// Three tab groups side by side (shape / comparison / assessment) over the
/// trace with a time-view group (raster / drift / contamination / PSTH)
/// beside it. Every pane is present so nothing is lost versus the old tab
/// strip.
pub fn default_tree() -> Tree<Pane> {
    let mut tiles: Tiles<Pane> = Tiles::default();
    let group = |tiles: &mut Tiles<Pane>, panes: &[Pane]| {
        let ids: Vec<TileId> = panes.iter().map(|&p| tiles.insert_pane(p)).collect();
        tiles.insert_tab_tile(ids)
    };
    let shape = group(
        &mut tiles,
        &[
            Pane::Waveforms,
            Pane::Templates,
            Pane::TemplateFeatures,
            Pane::Probe,
            Pane::Summary,
        ],
    );
    let compare = group(
        &mut tiles,
        &[
            Pane::Ccg,
            Pane::Features,
            Pane::Amplitudes,
            Pane::Isi,
            Pane::Similar,
        ],
    );
    let assess = group(
        &mut tiles,
        &[
            Pane::Quality,
            Pane::Suggest,
            Pane::Stats,
            Pane::Rate,
            Pane::SpikeData,
        ],
    );
    let top = tiles.insert_horizontal_tile(vec![shape, compare, assess]);
    let trace = group(&mut tiles, &[Pane::Trace, Pane::TraceImage]);
    let time = group(
        &mut tiles,
        &[
            Pane::Raster,
            Pane::DriftMap,
            Pane::ContamTime,
            Pane::Psth,
            Pane::ClusterScatter,
        ],
    );
    let bottom = tiles.insert_container(Linear::new_binary(
        LinearDir::Horizontal,
        [trace, time],
        0.7,
    ));
    let root = tiles.insert_container(Linear::new_binary(LinearDir::Vertical, [top, bottom], 0.62));
    Tree::new(TREE_ID, root, tiles)
}

/// Panes the user has closed — candidates for the "View" menu.
pub fn missing_panes(tree: &Tree<Pane>) -> Vec<Pane> {
    Pane::ALL
        .into_iter()
        .filter(|p| tree.tiles.find_pane(p).is_none())
        .collect()
}

/// The active child of every tab group.
pub fn active_tabs(tree: &Tree<Pane>) -> Vec<TileId> {
    tree.tiles
        .tiles()
        .filter_map(|t| match t {
            Tile::Container(Container::Tabs(tabs)) => tabs.active,
            _ => None,
        })
        .collect()
}

fn is_tabs(tiles: &Tiles<Pane>, id: TileId) -> bool {
    matches!(tiles.get(id), Some(Tile::Container(Container::Tabs(_))))
}

/// Depth-first search for the first tab group under `id`.
fn first_tabs(tiles: &Tiles<Pane>, id: TileId) -> Option<TileId> {
    if is_tabs(tiles, id) {
        return Some(id);
    }
    let Some(Tile::Container(c)) = tiles.get(id) else {
        return None;
    };
    c.children_vec()
        .into_iter()
        .find_map(|child| first_tabs(tiles, child))
}

/// Open `pane` as the active tab of `into` (when it is a tab group), else
/// of the first tab group in the tree, else as a new root. Re-opening a
/// pane that is already in the tree just activates it.
pub fn add_pane(tree: &mut Tree<Pane>, pane: Pane, into: Option<TileId>) {
    if let Some(existing) = tree.tiles.find_pane(&pane) {
        tree.make_active(|id, _| id == existing);
        return;
    }
    let id = tree.tiles.insert_pane(pane);
    let group = into
        .filter(|&t| is_tabs(&tree.tiles, t))
        .or_else(|| tree.root().and_then(|root| first_tabs(&tree.tiles, root)));
    match group {
        Some(g) => {
            if let Some(Tile::Container(Container::Tabs(tabs))) = tree.tiles.get_mut(g) {
                tabs.add_child(id);
                tabs.set_active(id);
            }
        }
        None => {
            let children = match tree.root() {
                Some(root) => vec![root, id],
                None => vec![id],
            };
            tree.root = Some(tree.tiles.insert_tab_tile(children));
        }
    }
}

/// Side effects a pane wants the app to apply after the frame — views only
/// read the session, so mutations are queued here.
pub enum ViewAction {
    /// Seek the trace window to this sample (Raster click).
    Seek(u64),
    /// Lasso split from the Features pane.
    Split(Vec<(ClusterId, Vec<u32>)>),
    Suggest(SuggestAction),
    /// Similar-pane row click: make `cluster` the similar selection
    /// (`add == false`) or toggle it in the similar part (`add == true`).
    /// The best clusters are kept either way (phy `_similar_selected`).
    SelectSimilar {
        cluster: ClusterId,
        add: bool,
    },
    /// Waveforms ctrl+click on a channel box: pin it as the FeatureView's
    /// first (`secondary == false`) or second channel (phy `select_channel`).
    SelectChannel {
        channel: u32,
        secondary: bool,
    },
    /// Open a closed pane in the tab group that asked for it.
    AddPane(Pane, TileId),
    /// phy `screenshot`: save this pane's area of the window as a PNG.
    Screenshot(Pane, egui::Rect),
}

/// Trace-pane state the toolbar and mouse navigation edit in place.
pub struct TraceControls<'a> {
    pub window_start: &'a mut u64,
    pub window_len: &'a mut u32,
    pub target_points: usize,
    pub cfg: &'a mut TracePreproc,
    pub gpu: Option<&'a GpuTracePreproc>,
    pub cache: &'a mut TraceCache,
    /// Shank id per raw channel (empty = one shank); enables per-shank CMR.
    pub shanks: &'a [u32],
    pub per_shank: &'a mut bool,
    pub display: crate::widgets::TraceDisplay,
}

/// Borrows of everything the panes draw from, built fresh each frame.
pub struct ViewBehavior<'a, P: DataProvider> {
    pub session: &'a Session<P>,
    pub selection: &'a [ClusterId],
    pub label_str: fn(&P::Label) -> &'static str,
    pub channel_positions: &'a [[f32; 2]],
    pub waveform_cache: &'a mut WaveformCache,
    pub feature_state: &'a mut FeatureViewState,
    pub amplitude_state: &'a mut AmplitudeViewState,
    pub raster_cache: &'a mut RasterCache,
    pub suggest_cache: &'a mut SuggestCache,
    pub ccg_cache: &'a mut CcgCache,
    /// Persisted per-view parameters (`UiPrefs.views`).
    pub view_params: &'a mut crate::prefs::ViewParams,
    /// SimilarityView: ranked list memo, table state, row cache, best clusters.
    pub similar: SimilarityPane<'a>,
    pub quality_cache: &'a mut QualityCache,
    /// Shared session + repaint handle for panes that compute on a worker.
    pub jobs: &'a JobCtx<Session<P>>,
    /// Dataset events (`events.csv` / `events.npy`), if any.
    pub events: Option<&'a Events>,
    /// Per-spike depths when the backend can estimate them.
    pub depths: Option<&'a SpikeDepths>,
    pub cluster_scatter: &'a mut ClusterScatterState,
    pub template_features: &'a mut TemplateFeatureState,
    pub psth_state: &'a mut PsthState,
    pub psth_cache: &'a mut PsthCache,
    pub trace: TraceControls<'a>,
    /// Panes currently closed, for the per-group "+" menu.
    pub closed: Vec<Pane>,
    /// The visible tab of every tab group; only these show a close button.
    pub active_tabs: Vec<TileId>,
    /// phy `toggle_auto_update`: panes showing a frozen selection instead
    /// of the live one.
    pub frozen: &'a mut std::collections::HashMap<Pane, Vec<ClusterId>>,
    /// The cluster table's visible rows, in display order.
    pub table_order: &'a [ClusterId],
    pub template_grid: &'a mut TemplateGridCache,
    pub trace_image: &'a mut TraceImageCache,
    pub spike_attrs: &'a mut SpikeAttrState,
    /// Colour of each unselected cluster under the colour scheme, and a
    /// generation that changes with it.
    pub colours: &'a [Option<egui::Color32>],
    pub colours_gen: u64,
    /// Shift+wheel notches over the coloured panes this frame.
    pub scheme_delta: i32,
    /// phy `toggle_spike_reorder` (Ctrl+R).
    pub spike_reorder: bool,
    /// Pane whose screenshot was asked for from its tab menu.
    pub screenshot: &'a mut Option<Pane>,
    pub actions: Vec<ViewAction>,
}

impl<P: DataProvider> ViewBehavior<'_, P> {
    fn trace_pane(&mut self, ui: &mut Ui) {
        let session = self.session;
        let sr = session.provider.sample_rate().max(1.0);
        let n_samples = session.provider.n_samples().0;
        // Zoom bounds: 1 ms to 5 s.
        let min_len = ((sr * 0.001) as u32).max(1);
        let max_len = ((sr * 5.0) as u32).max(min_len);
        ui.horizontal(|ui| {
            let window_ms = *self.trace.window_len as f32 * 1000.0 / sr;
            ui.label(format!(
                "t = {:.3} / {:.1} s",
                *self.trace.window_start as f64 / sr as f64,
                n_samples as f64 / sr as f64
            ))
            .on_hover_text(format!(
                "Window start sample: {}\nWindow length samples: {}\n\
                 Sample rate: {:.1} kHz\nH/L or Left/Right pan by one window · drag or wheel \
                 pans · Ctrl-wheel zooms",
                *self.trace.window_start,
                *self.trace.window_len,
                sr / 1000.0
            ));
            ui.separator();
            let mut new_ms = window_ms;
            let resp = ui.add(
                egui::Slider::new(&mut new_ms, 1.0..=5000.0)
                    .logarithmic(true)
                    .text("ms")
                    .clamping(egui::SliderClamping::Always),
            );
            if resp.changed() {
                *self.trace.window_len =
                    (((new_ms / 1000.0) * sr).round() as u32).clamp(min_len, max_len);
            }
            ui.separator();
            let mut hp_on = self.trace.cfg.hp().is_some();
            let mut cmr_on = self.trace.cfg.has_cmr();
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
                *self.trace.cfg = TracePreproc::from_flags(hp, cmr_on);
            }
            let n_shanks = {
                let mut ids: Vec<u32> = self.trace.shanks.to_vec();
                ids.sort_unstable();
                ids.dedup();
                ids.len()
            };
            if n_shanks > 1 {
                ui.add_enabled(
                    cmr_on,
                    egui::Checkbox::new(self.trace.per_shank, "per shank"),
                )
                .on_hover_text(format!(
                    "Reference each of the {n_shanks} shanks by its own median (CPU path)."
                ));
            }
            // Remote guest: the raw window streams in; keep repainting so
            // it fills in without the user having to move the mouse.
            if session.provider.pending_fetches() > 0 {
                ui.separator();
                ui.spinner();
                ui.weak("fetching trace…");
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(50));
            }
            if !self.selection.is_empty() {
                let (lo, hi) = (
                    *self.trace.window_start,
                    *self.trace.window_start + *self.trace.window_len as u64,
                );
                let n: usize = self
                    .selection
                    .iter()
                    .map(|&c| {
                        let t = session.spike_times(c);
                        t.partition_point(|s| s.0 < hi) - t.partition_point(|s| s.0 < lo)
                    })
                    .sum();
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let text = format!(
                        "{n} selected spike{} in window",
                        if n == 1 { "" } else { "s" }
                    );
                    ui.add(egui::Label::new(egui::RichText::new(text).weak()).truncate());
                });
            }
        });

        // Spike markers for the selection, on each cluster's peak channel.
        let selected = self.selection;
        let pre = waveform_window_samples(sr);
        self.waveform_cache.ensure(session, selected, pre, pre);
        let markers: Vec<TraceMarker<'_>> = selected
            .iter()
            .enumerate()
            .map(|(slot, &c)| TraceMarker {
                colour: selection_colour(slot),
                times: session.spike_times(c),
                channel: self.waveform_cache.peak_channel(slot),
            })
            .collect();
        // phy `toggle_highlighted_spikes` (Alt+S): every other cluster with a
        // spike in the window too, neutral, on its template's peak channel.
        let mut markers = markers;
        if self.trace.display.all_spikes {
            let times = session.cluster_index().all_spike_times();
            let (lo, hi) = (
                *self.trace.window_start,
                *self.trace.window_start + *self.trace.window_len as u64,
            );
            let (a, b) = (
                times.partition_point(|t| t.0 < lo),
                times.partition_point(|t| t.0 < hi),
            );
            let mut in_window: Vec<ClusterId> = session.spike_clusters_global()[a..b].to_vec();
            in_window.sort_unstable();
            in_window.dedup();
            for c in in_window.into_iter().filter(|c| !selected.contains(c)) {
                markers.push(TraceMarker {
                    colour: ui.visuals().weak_text_color(),
                    times: session.spike_times(c),
                    channel: session
                        .primary_template(c)
                        .and_then(|t| crate::amplitude_view::peak_channel(session, t))
                        .map(|ch| ch as usize),
                });
            }
        }

        let nav = trace_view(
            ui,
            session,
            *self.trace.window_start,
            *self.trace.window_len,
            self.trace.target_points,
            *self.trace.cfg,
            self.trace.gpu,
            &markers,
            self.trace.display,
            if *self.trace.per_shank {
                self.trace.shanks
            } else {
                &[]
            },
            self.trace.cache,
        );
        if let Some(nav) = nav {
            let old_len = *self.trace.window_len;
            let new_len = ((old_len as f32 / nav.zoom).round() as u32).clamp(min_len, max_len);
            // Keep the time under the pointer fixed while zooming.
            let pivot = (nav.anchor * (old_len as f32 - new_len as f32)) as i64;
            let max_start = n_samples.saturating_sub(new_len as u64) as i64;
            let start = *self.trace.window_start as i64 + nav.pan + pivot;
            *self.trace.window_start = start.clamp(0, max_start.max(0)) as u64;
            *self.trace.window_len = new_len;
            if let Some(sample) = nav.pick {
                // ponytail: nearest spike in time within 1 ms on any
                // channel; phy also weighs the clicked channel.
                let times = self.session.cluster_index().all_spike_times();
                let i = times.partition_point(|t| t.0 < sample);
                let tol = (self.session.provider.sample_rate() * 0.001) as u64;
                let nearest = [i.checked_sub(1), Some(i)]
                    .into_iter()
                    .flatten()
                    .filter(|&j| j < times.len())
                    .min_by_key(|&j| times[j].0.abs_diff(sample))
                    .filter(|&j| times[j].0.abs_diff(sample) <= tol);
                if let Some(j) = nearest {
                    let c = self.session.spike_clusters_global()[j];
                    self.actions
                        .push(ViewAction::Suggest(SuggestAction::Select(vec![c])));
                }
            }
        }
    }
}

impl<P: DataProvider> Behavior<Pane> for ViewBehavior<'_, P> {
    fn pane_ui(&mut self, ui: &mut Ui, _tile_id: TileId, pane: &mut Pane) -> UiResponse {
        if *self.screenshot == Some(*pane) {
            *self.screenshot = None;
            self.actions
                .push(ViewAction::Screenshot(*pane, ui.max_rect()));
        }
        let frozen = self.frozen.get(pane).cloned();
        // Inner padding: views draw edge to edge in the rect they are given.
        let padded = ui.max_rect().shrink2(egui::vec2(6.0, 4.0));
        let mut ui = ui.new_child(egui::UiBuilder::new().max_rect(padded));
        let ui = &mut ui;
        let session = self.session;
        let selected: &[ClusterId] = frozen.as_deref().unwrap_or(self.selection);
        let scroll = |ui: &mut Ui, pane: Pane, f: &mut dyn FnMut(&mut Ui)| {
            egui::ScrollArea::vertical()
                .id_salt(pane.title())
                .show(ui, |ui| f(ui));
        };
        match *pane {
            Pane::Trace => self.trace_pane(ui),
            Pane::Summary => {
                selection_summary(ui, session, selected, self.label_str);
                ui.separator();
                ui.weak("Press ? for keyboard and mouse help.");
            }
            Pane::Waveforms => waveform_view(
                ui,
                session,
                selected,
                self.channel_positions,
                self.trace.shanks,
                self.waveform_cache,
                &mut self.view_params.waveform,
                &mut self.actions,
            ),
            Pane::SpikeData => {
                if let Some(splits) = spike_attr_view(
                    ui,
                    session,
                    selected,
                    self.spike_attrs,
                    self.jobs,
                    self.spike_reorder,
                ) {
                    self.actions.push(ViewAction::Split(splits));
                }
            }
            Pane::TraceImage => {
                let shanks = if *self.trace.per_shank {
                    self.trace.shanks
                } else {
                    &[]
                };
                trace_image_view(
                    ui,
                    session,
                    *self.trace.window_start,
                    *self.trace.window_len,
                    *self.trace.cfg,
                    shanks,
                    self.trace.display.flip,
                    self.trace_image,
                );
            }
            Pane::Templates => {
                if let Some(next) = template_grid_view(
                    ui,
                    session,
                    selected,
                    self.table_order,
                    self.template_grid,
                    self.colours,
                    &mut self.scheme_delta,
                ) {
                    self.actions
                        .push(ViewAction::Suggest(SuggestAction::Select(next)));
                }
            }
            Pane::Amplitudes => {
                let shanks = if *self.trace.per_shank {
                    self.trace.shanks
                } else {
                    &[]
                };
                let feature = self.feature_state.amplitude_feature();
                match amplitude_view(
                    ui,
                    session,
                    selected,
                    self.amplitude_state,
                    self.jobs,
                    *self.trace.cfg,
                    shanks,
                    feature,
                    self.spike_reorder,
                ) {
                    Some(AmpAction::Split(splits)) => self.actions.push(ViewAction::Split(splits)),
                    Some(AmpAction::Seek(sample)) => self.actions.push(ViewAction::Seek(sample)),
                    None => {}
                }
            }
            Pane::Isi => {
                let p = &mut self.view_params.isi;
                scroll(ui, *pane, &mut |ui| isi_view(ui, session, selected, p));
            }
            Pane::Ccg => {
                let (cache, p, jobs) = (&mut *self.ccg_cache, &mut self.view_params.ccg, self.jobs);
                scroll(ui, *pane, &mut |ui| {
                    correlogram_grid_view(ui, session, selected, cache, jobs, p)
                });
            }
            Pane::Features => {
                // Computed PCs follow the trace pane's HP/CMR toggles.
                let shanks = if *self.trace.per_shank {
                    self.trace.shanks
                } else {
                    &[]
                };
                let cfg = *self.trace.cfg;
                if let Some(splits) = feature_view(
                    ui,
                    session,
                    selected,
                    self.feature_state,
                    self.jobs,
                    cfg,
                    shanks,
                    self.spike_reorder,
                ) {
                    self.actions.push(ViewAction::Split(splits));
                }
            }
            Pane::Raster => {
                let window = (*self.trace.window_start, *self.trace.window_len);
                match raster_view(
                    ui,
                    session,
                    selected,
                    window,
                    self.raster_cache,
                    (self.colours, self.colours_gen),
                    &mut self.scheme_delta,
                ) {
                    Some(RasterClick::Seek(sample)) => self.actions.push(ViewAction::Seek(sample)),
                    Some(RasterClick::Select(c, add)) => {
                        let mut next = if add { selected.to_vec() } else { Vec::new() };
                        match next.iter().position(|&s| s == c) {
                            Some(i) => {
                                next.remove(i);
                            }
                            None => next.push(c),
                        }
                        if !next.is_empty() {
                            self.actions
                                .push(ViewAction::Suggest(SuggestAction::Select(next)));
                        }
                    }
                    None => {}
                }
            }
            Pane::Rate => firing_rate_view(ui, session, selected, &mut self.view_params.rate),
            Pane::Probe => probe_view(
                ui,
                session,
                selected,
                self.channel_positions,
                self.waveform_cache,
            ),
            Pane::Similar => {
                let hidden = &mut self.view_params.hidden_cols;
                if let Some((cluster, add)) =
                    similarity_view(ui, session, selected, &mut self.similar, hidden)
                {
                    self.actions
                        .push(ViewAction::SelectSimilar { cluster, add });
                }
            }
            Pane::Stats => scroll(ui, *pane, &mut |ui| {
                cluster_statistics_view_for(ui, session, selected)
            }),
            Pane::Quality => {
                let cache = &mut *self.quality_cache;
                let jobs = self.jobs;
                scroll(ui, *pane, &mut |ui| {
                    quality_view(ui, session, selected, cache, jobs)
                });
            }
            Pane::Suggest => {
                self.suggest_cache.selection = selected.to_vec();
                let label_str = self.label_str;
                let jobs = self.jobs;
                let action = egui::ScrollArea::vertical()
                    .id_salt(pane.title())
                    .show(ui, |ui| {
                        suggest_view(ui, session, self.suggest_cache, label_str, jobs)
                    })
                    .inner;
                if let Some(act) = action {
                    self.actions.push(ViewAction::Suggest(act));
                }
            }
            Pane::DriftMap => drift_map_view(ui, session, selected, self.depths),
            Pane::ContamTime => contamination_over_time_view(ui, session, selected),
            Pane::Psth => psth_view(
                ui,
                session,
                selected,
                self.events,
                self.psth_state,
                self.psth_cache,
            ),
            Pane::TemplateFeatures => {
                if let Some(splits) =
                    template_feature_view(ui, session, selected, self.template_features, self.jobs)
                {
                    self.actions.push(ViewAction::Split(splits));
                }
            }
            Pane::ClusterScatter => {
                if let Some(clusters) = cluster_scatter_view(
                    ui,
                    session,
                    selected,
                    self.depths,
                    self.cluster_scatter,
                    self.colours,
                    &mut self.scheme_delta,
                ) {
                    self.actions
                        .push(ViewAction::Suggest(SuggestAction::Select(clusters)));
                }
            }
        }
        UiResponse::None
    }

    fn tab_title_for_pane(&mut self, pane: &Pane) -> egui::WidgetText {
        if self.frozen.contains_key(pane) {
            format!("🔒 {}", pane.title()).into()
        } else {
            pane.title().into()
        }
    }

    /// egui_tiles paints tab titles without widget info, so screen readers
    /// (and AccessKit-driven tests) cannot find or switch panes. Name them.
    fn on_tab_button(
        &mut self,
        tiles: &mut Tiles<Pane>,
        tile_id: TileId,
        response: egui::Response,
    ) -> egui::Response {
        if let Some(Tile::Pane(pane)) = tiles.get(tile_id) {
            let title = pane.title();
            response
                .widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, title));
            // phy view menu: freeze the view on this selection, or save it.
            let pane = *pane;
            response.context_menu(|ui| {
                let frozen = self.frozen.contains_key(&pane);
                if ui
                    .button(if frozen {
                        "Follow the selection"
                    } else {
                        "Freeze on this selection"
                    })
                    .clicked()
                {
                    if frozen {
                        self.frozen.remove(&pane);
                    } else {
                        self.frozen.insert(pane, self.selection.to_vec());
                    }
                    ui.close();
                }
                if ui.button("Save screenshot").clicked() {
                    *self.screenshot = Some(pane);
                    ui.close();
                }
            });
        }
        // egui_tiles never scrolls the strip to the active tab, so a tab
        // opened from the View menu or a keyboard shortcut could sit behind
        // the overflow arrows. Scroll once when a tab becomes active; the
        // user can still scroll the strip freely afterwards.
        let key = egui::Id::new("sorrel-tab-shown").with(tile_id);
        let active = self.active_tabs.contains(&tile_id);
        let shown = response
            .ctx
            .data(|d| d.get_temp::<bool>(key))
            .unwrap_or(false);
        if active && !shown {
            response.scroll_to_me(Some(egui::Align::Center));
        }
        response.ctx.data_mut(|d| d.insert_temp(key, active));
        response
    }

    /// Only the active tab shows its close button; a × on every tab cost
    /// ~20 px each and pushed most titles behind the overflow arrows.
    fn is_tab_closable(&self, _tiles: &Tiles<Pane>, tile_id: TileId) -> bool {
        // egui_tiles detaches the tab group while drawing its bar, so the
        // parent cannot be looked up here; `active_tabs` is taken up front.
        self.active_tabs.contains(&tile_id)
    }

    fn tab_bar_height(&self, _style: &egui::Style) -> f32 {
        30.0
    }

    fn gap_width(&self, _style: &egui::Style) -> f32 {
        6.0
    }

    fn tab_title_spacing(&self, _visuals: &egui::Visuals) -> f32 {
        5.0
    }

    /// 16 px hit target around an 8 px cross.
    fn close_button_outer_size(&self) -> f32 {
        16.0
    }

    fn close_button_inner_margin(&self) -> f32 {
        4.0
    }

    fn simplification_options(&self) -> SimplificationOptions {
        SimplificationOptions {
            // Keep a tab bar on lone panes so they stay named and closable.
            all_panes_must_have_tabs: true,
            ..Default::default()
        }
    }

    fn top_bar_right_ui(
        &mut self,
        _tiles: &Tiles<Pane>,
        ui: &mut Ui,
        tile_id: TileId,
        _tabs: &egui_tiles::Tabs,
        _scroll_offset: &mut f32,
    ) {
        if self.closed.is_empty() {
            return;
        }
        ui.menu_button("+", |ui| {
            for pane in self.closed.clone() {
                if ui.button(pane.title()).clicked() {
                    self.actions.push(ViewAction::AddPane(pane, tile_id));
                    ui.close();
                }
            }
        })
        .response
        .on_hover_text("Open a closed view in this group");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tree_has_every_pane_exactly_once() {
        let tree = default_tree();
        let n_panes = tree
            .tiles
            .tiles()
            .filter(|t| matches!(t, Tile::Pane(_)))
            .count();
        assert_eq!(n_panes, Pane::ALL.len());
        assert!(missing_panes(&tree).is_empty());
    }

    #[test]
    fn closing_then_adding_a_pane_round_trips() {
        let mut tree = default_tree();
        let id = tree.tiles.find_pane(&Pane::Ccg).unwrap();
        tree.remove_recursively(id);
        assert_eq!(missing_panes(&tree), vec![Pane::Ccg]);
        add_pane(&mut tree, Pane::Ccg, None);
        assert!(missing_panes(&tree).is_empty());
        // Adding again is idempotent.
        add_pane(&mut tree, Pane::Ccg, None);
        let n_ccg = tree
            .tiles
            .tiles()
            .filter(|t| matches!(t, Tile::Pane(Pane::Ccg)))
            .count();
        assert_eq!(n_ccg, 1);
    }

    #[test]
    fn empty_tree_gets_a_root_when_a_pane_is_added() {
        let mut tree: Tree<Pane> = Tree::empty(TREE_ID);
        add_pane(&mut tree, Pane::Trace, None);
        assert!(tree.root().is_some());
        assert_eq!(missing_panes(&tree).len(), Pane::ALL.len() - 1);
    }

    #[test]
    fn tree_serialises_round_trip() {
        let tree = default_tree();
        let json = serde_json::to_string(&tree).unwrap();
        let back: Tree<Pane> = serde_json::from_str(&json).unwrap();
        assert!(missing_panes(&back).is_empty());
    }
}

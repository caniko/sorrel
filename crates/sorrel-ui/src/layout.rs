//! Dockable layout. Every view is a [`Pane`] in an [`egui_tiles::Tree`], so
//! the curator can put waveforms, correlograms and features side by side
//! (phy's workflow) instead of flipping through one tab at a time. The tree
//! is plain data: it serialises for the per-dataset prefs file.

use crate::jobs::JobCtx;
use crate::theme::selection_colour;
use crate::views::{
    amplitude_view, cluster_statistics_view, contamination_over_time_view, correlogram_grid_view,
    drift_map_view, feature_view, firing_rate_view, isi_view, probe_view, psth_view, quality_view,
    raster_view, selection_summary, similarity_view, suggest_view, template_view, waveform_view,
    waveform_window_samples, CcgCache, FeatureViewState, PsthCache, PsthState, QualityCache,
    RasterCache, SuggestAction, SuggestCache, WaveformCache,
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
}

impl Pane {
    pub const ALL: [Pane; 18] = [
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
        }
    }
}

const TREE_ID: &str = "sorrel-layout";

/// Three tab groups side by side (shape / comparison / assessment) over a
/// full-width trace. Every pane is present so nothing is lost versus the
/// old tab strip.
pub fn default_tree() -> Tree<Pane> {
    let mut tiles: Tiles<Pane> = Tiles::default();
    let group = |tiles: &mut Tiles<Pane>, panes: &[Pane]| {
        let ids: Vec<TileId> = panes.iter().map(|&p| tiles.insert_pane(p)).collect();
        tiles.insert_tab_tile(ids)
    };
    let shape = group(
        &mut tiles,
        &[Pane::Waveforms, Pane::Templates, Pane::Probe, Pane::Summary],
    );
    let compare = group(
        &mut tiles,
        &[
            Pane::Ccg,
            Pane::Features,
            Pane::Amplitudes,
            Pane::Isi,
            Pane::Rate,
            Pane::Similar,
            Pane::Psth,
        ],
    );
    let assess = group(
        &mut tiles,
        &[
            Pane::Quality,
            Pane::Suggest,
            Pane::Stats,
            Pane::Raster,
            Pane::DriftMap,
            Pane::ContamTime,
        ],
    );
    let top = tiles.insert_horizontal_tile(vec![shape, compare, assess]);
    let trace = tiles.insert_pane(Pane::Trace);
    let root = tiles.insert_container(Linear::new_binary(LinearDir::Vertical, [top, trace], 0.68));
    Tree::new(TREE_ID, root, tiles)
}

/// Panes the user has closed — candidates for the "View" menu.
pub fn missing_panes(tree: &Tree<Pane>) -> Vec<Pane> {
    Pane::ALL
        .into_iter()
        .filter(|p| tree.tiles.find_pane(p).is_none())
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
    /// Similar-pane row click: compare `cluster` with the primary selection
    /// (`add == false`) or add it to the current selection (`add == true`).
    Compare {
        cluster: ClusterId,
        add: bool,
    },
    /// Open a closed pane in the tab group that asked for it.
    AddPane(Pane, TileId),
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
}

/// Borrows of everything the panes draw from, built fresh each frame.
pub struct ViewBehavior<'a, P: DataProvider> {
    pub session: &'a Session<P>,
    pub selection: &'a [ClusterId],
    pub label_str: fn(&P::Label) -> &'static str,
    pub channel_positions: &'a [[f32; 2]],
    pub waveform_cache: &'a mut WaveformCache,
    pub feature_state: &'a mut FeatureViewState,
    pub raster_cache: &'a mut RasterCache,
    pub suggest_cache: &'a mut SuggestCache,
    pub ccg_cache: &'a mut CcgCache,
    pub quality_cache: &'a mut QualityCache,
    /// Shared session + repaint handle for panes that compute on a worker.
    pub jobs: &'a JobCtx<Session<P>>,
    /// Dataset events (`events.csv` / `events.npy`), if any.
    pub events: Option<&'a Events>,
    /// Per-spike depths when the backend can estimate them.
    pub depths: Option<&'a SpikeDepths>,
    pub psth_state: &'a mut PsthState,
    pub psth_cache: &'a mut PsthCache,
    pub trace: TraceControls<'a>,
    /// Panes currently closed, for the per-group "+" menu.
    pub closed: Vec<Pane>,
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
                "t = {:.3} s · window {:.0} ms",
                *self.trace.window_start as f64 / sr as f64,
                window_ms
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

        let nav = trace_view(
            ui,
            session,
            *self.trace.window_start,
            *self.trace.window_len,
            self.trace.target_points,
            *self.trace.cfg,
            self.trace.gpu,
            &markers,
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
        }
    }
}

impl<P: DataProvider> Behavior<Pane> for ViewBehavior<'_, P> {
    fn pane_ui(&mut self, ui: &mut Ui, _tile_id: TileId, pane: &mut Pane) -> UiResponse {
        let session = self.session;
        let selected = self.selection;
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
                self.waveform_cache,
            ),
            Pane::Templates => template_view(ui, session, selected),
            Pane::Amplitudes => amplitude_view(ui, session, selected),
            Pane::Isi => scroll(ui, *pane, &mut |ui| isi_view(ui, session, selected)),
            Pane::Ccg => {
                let cache = &mut *self.ccg_cache;
                scroll(ui, *pane, &mut |ui| {
                    correlogram_grid_view(ui, session, selected, cache)
                });
            }
            Pane::Features => {
                if let Some(splits) = feature_view(ui, session, selected, self.feature_state) {
                    self.actions.push(ViewAction::Split(splits));
                }
            }
            Pane::Raster => {
                let window = (*self.trace.window_start, *self.trace.window_len);
                if let Some(sample) = raster_view(ui, session, selected, window, self.raster_cache)
                {
                    self.actions.push(ViewAction::Seek(sample));
                }
            }
            Pane::Rate => firing_rate_view(ui, session, selected),
            Pane::Probe => probe_view(
                ui,
                session,
                selected,
                self.channel_positions,
                self.waveform_cache,
            ),
            Pane::Similar => {
                if let Some((cluster, add)) = similarity_view(ui, session, selected) {
                    self.actions.push(ViewAction::Compare { cluster, add });
                }
            }
            Pane::Stats => scroll(ui, *pane, &mut |ui| cluster_statistics_view(ui, session)),
            Pane::Quality => {
                let cache = &mut *self.quality_cache;
                let jobs = self.jobs;
                scroll(ui, *pane, &mut |ui| {
                    quality_view(ui, session, selected, cache, jobs)
                });
            }
            Pane::Suggest => {
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
        }
        UiResponse::None
    }

    fn tab_title_for_pane(&mut self, pane: &Pane) -> egui::WidgetText {
        pane.title().into()
    }

    fn is_tab_closable(&self, _tiles: &Tiles<Pane>, _tile_id: TileId) -> bool {
        true
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

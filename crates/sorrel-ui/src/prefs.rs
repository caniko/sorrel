//! Per-dataset UI preferences: the dock layout, table sort/filter, trace
//! settings and the last selected cluster, stored as JSON next to the
//! derived-data cache (`<dataset>/.sorrel/ui.json`) so reopening a
//! recording resumes where curation left off.

use crate::layout::Pane;
use crate::widgets::ClusterColumn;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct UiPrefs {
    pub tree: Option<egui_tiles::Tree<Pane>>,
    pub sort_by: Option<ClusterColumn>,
    pub sort_descending: bool,
    pub hide_empty: bool,
    pub hide_noise: bool,
    pub window_ms: Option<f32>,
    pub hp: bool,
    pub cmr: bool,
    pub anchor: Option<u32>,
    pub views: ViewParams,
}

/// Per-view parameters phy keeps in its `state_attrs` (waveform mode and
/// scaling, correlogram bins, ISI range, hidden table columns).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ViewParams {
    pub waveform: WaveformParams,
    pub ccg: CcgParams,
    pub isi: IsiParams,
    pub rate: RateParams,
    #[serde(default)]
    pub feature: FeatureParams,
    /// Cluster/Similar table columns the user hid (by column name).
    pub hidden_cols: Vec<String>,
}

/// Feature basis shown in the Features pane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureBasis {
    #[default]
    #[serde(rename = "pc_2ch")]
    Pc2Ch,
    #[serde(rename = "pc0_4ch")]
    Pc0_4Ch,
    #[serde(rename = "energy_4ch")]
    Energy4Ch,
    #[serde(rename = "peak_4ch")]
    Peak4Ch,
    #[serde(rename = "valley_4ch")]
    Valley4Ch,
}

impl FeatureBasis {
    pub const ALL: [Self; 5] = [
        Self::Pc2Ch,
        Self::Pc0_4Ch,
        Self::Energy4Ch,
        Self::Peak4Ch,
        Self::Valley4Ch,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Pc2Ch => "PC (2 ch)",
            Self::Pc0_4Ch => "PC0 (4 ch)",
            Self::Energy4Ch => "Energy (4 ch)",
            Self::Peak4Ch => "Peak (4 ch)",
            Self::Valley4Ch => "Valley (4 ch)",
        }
    }

    pub fn is_four_channel(self) -> bool {
        !matches!(self, Self::Pc2Ch)
    }

    pub fn is_raw_scalar(self) -> bool {
        matches!(self, Self::Energy4Ch | Self::Peak4Ch | Self::Valley4Ch)
    }
}

/// How spike clouds are rendered in the Features pane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureDisplayMode {
    #[default]
    Points,
    Density,
    Contours,
}

/// Per-view parameters for the Features pane.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct FeatureParams {
    pub basis: FeatureBasis,
    pub display_mode: FeatureDisplayMode,
    pub same_shank_clusters: bool,
}

/// phy `FiringRateView` (v2.1): bin count over the recording (`:frn`) and
/// the time unit of the x axis.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RateParams {
    pub n_bins: usize,
    pub unit: TimeUnit,
    /// `:frmin` / `:frmax`: the x range in seconds (None = recording edge).
    pub x_min: Option<f32>,
    pub x_max: Option<f32>,
    #[serde(skip)]
    pub pz: crate::panzoom::PanZoom,
}

impl Default for RateParams {
    fn default() -> Self {
        Self {
            n_bins: 200,
            unit: TimeUnit::Seconds,
            x_min: None,
            x_max: None,
            pz: Default::default(),
        }
    }
}

impl RateParams {
    /// The `:frmin`/`:frmax` range within the session, or all of it when
    /// the range is empty.
    pub fn x_range(&self, (start, end): (f32, f32)) -> (f32, f32) {
        let lo = self.x_min.unwrap_or(start).clamp(start, end);
        let hi = self.x_max.unwrap_or(end).clamp(start, end);
        if hi > lo {
            (lo, hi)
        } else {
            (start, end)
        }
    }

    /// As above, but returns seconds relative to the origin so a short
    /// exposure at a large offset retains its width in plot coordinates.
    pub(crate) fn relative_x_range(&self, origin: f64, duration: f32) -> (f32, f32) {
        if origin == 0.0 {
            return self.x_range((0.0, duration));
        }
        let relative = |t: f32| (f64::from(t) - origin) as f32;
        Self {
            x_min: self.x_min.map(relative),
            x_max: self.x_max.map(relative),
            ..self.clone()
        }
        .x_range((0.0, duration))
    }

    pub(crate) fn visible_x_range(&self, origin: f64, duration: f32) -> (f32, f32) {
        self.pz
            .apply(self.relative_x_range(origin, duration), (0.0, 1.0))
            .0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeUnit {
    #[default]
    Seconds,
    Minutes,
    Hours,
}

impl TimeUnit {
    pub const ALL: [TimeUnit; 3] = [TimeUnit::Seconds, TimeUnit::Minutes, TimeUnit::Hours];

    pub fn seconds(self) -> f32 {
        match self {
            TimeUnit::Seconds => 1.0,
            TimeUnit::Minutes => 60.0,
            TimeUnit::Hours => 3600.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TimeUnit::Seconds => "s",
            TimeUnit::Minutes => "min",
            TimeUnit::Hours => "h",
        }
    }
}

/// phy `WaveformView` modes, cycled with `w` / `shift+w`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaveformMode {
    /// Raw spikes with the mean drawn on top (phy shows one or the other).
    #[default]
    Waveforms,
    Mean,
    Templates,
}

/// Which spikes drive the waveform pane.
///
/// `AllSpikes` is the default and what `M` always restores: every spike the
/// selected cluster owns contributes to the mean, not the legacy
/// `n_spikes`-capped sample. `Sampled` is an explicit, clearly-labelled
/// opt-in for a fast preview and is never selected by old preferences.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaveformSource {
    #[default]
    AllSpikes,
    Sampled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WaveformParams {
    pub mode: WaveformMode,
    /// `o` (checkbox "overlap"): true superimposes the selected clusters in
    /// each channel box (default); false is phy's side-by-side slots.
    pub overlap: bool,
    /// `ctrl+l`: channel id labels.
    pub show_labels: bool,
    /// phy `n_spikes_waveforms`, used only by the explicit `Sampled` source.
    /// Old files keep the value, but it no longer caps the default view.
    pub n_spikes: usize,
    /// All spikes (default) or an explicit `n_spikes` preview.
    pub source: WaveformSource,
    /// Total channels in the waveform pane, shared by all selected clusters.
    /// Four is the tetrode default; larger probes can opt in to more.
    pub max_channels: usize,
    /// Box width/height factors (`ctrl+left/right`, `ctrl+wheel`).
    pub box_scale: [f32; 2],
    /// Probe spread factors (`shift+arrows`).
    pub probe_scale: [f32; 2],
    /// Amplitude factor (`ctrl+up/down`).
    pub amp_scale: f32,
    /// Draw the focused feature plot's PC1/PC2 temporal shapes on its
    /// channels (cross-talk with the Features pane).
    pub pc_shapes: bool,
    /// In `WaveformMode::Mean`, draw a translucent ±1 SD ribbon behind each
    /// cluster's mean waveform.
    #[serde(default)]
    pub show_sd: bool,
}

impl Default for WaveformParams {
    fn default() -> Self {
        Self {
            mode: WaveformMode::default(),
            overlap: true,
            show_labels: true,
            n_spikes: 100,
            source: WaveformSource::AllSpikes,
            max_channels: 4,
            box_scale: [1.0, 1.0],
            probe_scale: [1.0, 1.0],
            amp_scale: 1.0,
            pc_shapes: true,
            show_sd: false,
        }
    }
}

/// phy `CorrelogramView` defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CcgParams {
    pub bin_ms: f32,
    pub window_ms: f32,
    /// `n`: one y scale for every cell instead of per-cell.
    pub uniform: bool,
}

impl Default for CcgParams {
    fn default() -> Self {
        Self {
            bin_ms: 1.0,
            window_ms: 50.0,
            uniform: false,
        }
    }
}

fn default_isi_log_min_ms() -> f32 {
    0.2
}

fn default_isi_log_max_ms() -> f32 {
    10_000.0
}

fn default_isi_log_n_bins() -> usize {
    80
}

/// phy `ISIView` defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct IsiParams {
    pub x_min_ms: f32,
    pub x_max_ms: f32,
    pub n_bins: usize,
    #[serde(default)]
    pub log_x: bool,
    #[serde(default = "default_isi_log_min_ms")]
    pub log_min_ms: f32,
    #[serde(default = "default_isi_log_max_ms")]
    pub log_max_ms: f32,
    #[serde(default = "default_isi_log_n_bins")]
    pub log_n_bins: usize,
}

impl Default for IsiParams {
    fn default() -> Self {
        Self {
            x_min_ms: 0.0,
            x_max_ms: 50.0,
            n_bins: 50,
            log_x: false,
            log_min_ms: default_isi_log_min_ms(),
            log_max_ms: default_isi_log_max_ms(),
            log_n_bins: default_isi_log_n_bins(),
        }
    }
}

impl UiPrefs {
    pub fn path(dataset_dir: &Path) -> PathBuf {
        dataset_dir.join(".sorrel").join("ui.json")
    }

    /// Missing or unreadable prefs are not an error: start from defaults.
    pub fn load(dataset_dir: &Path) -> Self {
        let path = Self::path(dataset_dir);
        match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                log::warn!("ignoring malformed {}: {e}", path.display());
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, dataset_dir: &Path) -> std::io::Result<()> {
        let path = Self::path(dataset_dir);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let json = serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, json)?;
        sorrel_io::fsutil::replace_file_io(&tmp, &path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waveform_channel_limit_defaults_to_four_and_survives_old_preferences() {
        assert_eq!(WaveformParams::default().max_channels, 4);
        assert_eq!(
            WaveformParams::default().source,
            WaveformSource::AllSpikes,
            "M and the default view use every spike"
        );
        let old: WaveformParams =
            serde_json::from_str(r#"{"n_spikes":80,"overlap":false}"#).unwrap();
        assert_eq!(old.max_channels, 4);
        assert_eq!(old.n_spikes, 80);
        assert_eq!(
            old.source,
            WaveformSource::AllSpikes,
            "a stored legacy cap must not become the default source"
        );
        assert!(!old.overlap);
        let changed = WaveformParams {
            max_channels: 12,
            source: WaveformSource::Sampled,
            ..Default::default()
        };
        let back: WaveformParams =
            serde_json::from_str(&serde_json::to_string(&changed).unwrap()).unwrap();
        assert_eq!(back.max_channels, 12);
        assert_eq!(back.source, WaveformSource::Sampled);
    }

    #[test]
    fn a_stored_hundred_spike_cap_does_not_become_all_spike_semantics() {
        let old: WaveformParams =
            serde_json::from_str(r#"{"mode":"Waveforms","n_spikes":100}"#).unwrap();
        assert_eq!(old.n_spikes, 100, "the preview setting round-trips");
        assert_eq!(
            old.source,
            WaveformSource::AllSpikes,
            "but M/all spikes ignore it unless the user opts in"
        );
    }

    #[test]
    fn rate_x_range_clamps_and_falls_back_to_the_recording() {
        let mut p = RateParams::default();
        assert_eq!(p.x_range((0.0, 100.0)), (0.0, 100.0));
        (p.x_min, p.x_max) = (Some(10.0), Some(500.0));
        assert_eq!(p.x_range((0.0, 100.0)), (10.0, 100.0));
        p.x_max = Some(5.0);
        assert_eq!(p.x_range((0.0, 100.0)), (0.0, 100.0));
    }

    #[test]
    fn prefs_round_trip_and_tolerate_garbage() {
        let dir = tempfile::tempdir().unwrap();
        let prefs = UiPrefs {
            tree: Some(crate::layout::default_tree()),
            sort_by: Some(ClusterColumn::Quality),
            sort_descending: true,
            hide_noise: true,
            window_ms: Some(250.0),
            hp: true,
            anchor: Some(7),
            views: ViewParams {
                hidden_cols: vec!["sh".into()],
                ..Default::default()
            },
            ..Default::default()
        };
        prefs.save(dir.path()).unwrap();
        let back = UiPrefs::load(dir.path());
        assert_eq!(back.sort_by, Some(ClusterColumn::Quality));
        assert!(back.sort_descending && back.hide_noise && back.hp && !back.cmr);
        assert_eq!(back.window_ms, Some(250.0));
        assert_eq!(back.anchor, Some(7));
        assert_eq!(back.views.hidden_cols, ["sh"]);
        assert_eq!(
            back.views.ccg.window_ms, 50.0,
            "absent fields take phy defaults"
        );
        assert!(crate::layout::missing_panes(back.tree.as_ref().unwrap()).is_empty());

        std::fs::write(UiPrefs::path(dir.path()), b"{ not json").unwrap();
        let fallback = UiPrefs::load(dir.path());
        assert!(fallback.tree.is_none());
        assert!(UiPrefs::load(Path::new("/nonexistent/sorrel"))
            .anchor
            .is_none());
    }

    #[test]
    fn legacy_ui_json_deserializes_with_exact_phy_defaults() {
        let legacy = r#"{
            "views": {
                "waveform": { "mode": "Waveforms", "n_spikes": 100 },
                "isi": { "x_min_ms": 0.0, "x_max_ms": 50.0, "n_bins": 50 }
            }
        }"#;
        let prefs: UiPrefs = serde_json::from_str(legacy).unwrap();
        assert_eq!(prefs.views.feature.basis, FeatureBasis::Pc2Ch);
        assert_eq!(prefs.views.feature.display_mode, FeatureDisplayMode::Points);
        assert!(!prefs.views.feature.same_shank_clusters);
        assert!(!prefs.views.isi.log_x);
        assert_eq!(prefs.views.isi.x_min_ms, 0.0);
        assert_eq!(prefs.views.isi.x_max_ms, 50.0);
        assert_eq!(prefs.views.isi.n_bins, 50);
        assert_eq!(prefs.views.isi.log_min_ms, 0.2);
        assert_eq!(prefs.views.isi.log_max_ms, 10_000.0);
        assert_eq!(prefs.views.isi.log_n_bins, 80);
        assert_eq!(prefs.views.waveform.mode, WaveformMode::Waveforms);
        assert!(!prefs.views.waveform.show_sd);

        // Also an completely empty object produces identical defaults.
        let empty: UiPrefs = serde_json::from_str("{}").unwrap();
        assert_eq!(empty.views.feature, FeatureParams::default());
        assert!(!empty.views.isi.log_x);
        assert_eq!(empty.views.isi.log_min_ms, 0.2);
        assert_eq!(empty.views.isi.log_max_ms, 10_000.0);
        assert_eq!(empty.views.isi.log_n_bins, 80);
        assert!(!empty.views.waveform.show_sd);
    }

    #[test]
    fn new_prefs_fields_and_enum_variants_round_trip() {
        for (basis, expected_json, expected_four_ch, expected_raw_scalar) in [
            (FeatureBasis::Pc2Ch, "\"pc_2ch\"", false, false),
            (FeatureBasis::Pc0_4Ch, "\"pc0_4ch\"", true, false),
            (FeatureBasis::Energy4Ch, "\"energy_4ch\"", true, true),
            (FeatureBasis::Peak4Ch, "\"peak_4ch\"", true, true),
            (FeatureBasis::Valley4Ch, "\"valley_4ch\"", true, true),
        ] {
            let s = serde_json::to_string(&basis).unwrap();
            assert_eq!(s, expected_json);
            assert_eq!(serde_json::from_str::<FeatureBasis>(&s).unwrap(), basis);
            assert_eq!(basis.is_four_channel(), expected_four_ch);
            assert_eq!(basis.is_raw_scalar(), expected_raw_scalar);
            assert!(!basis.label().is_empty());
        }

        for (mode, expected_json) in [
            (FeatureDisplayMode::Points, "\"points\""),
            (FeatureDisplayMode::Density, "\"density\""),
            (FeatureDisplayMode::Contours, "\"contours\""),
        ] {
            let s = serde_json::to_string(&mode).unwrap();
            assert_eq!(s, expected_json);
            assert_eq!(
                serde_json::from_str::<FeatureDisplayMode>(&s).unwrap(),
                mode
            );
        }

        let prefs = UiPrefs {
            views: ViewParams {
                waveform: WaveformParams {
                    show_sd: true,
                    ..Default::default()
                },
                isi: IsiParams {
                    x_min_ms: 1.0,
                    x_max_ms: 100.0,
                    n_bins: 60,
                    log_x: true,
                    log_min_ms: 0.5,
                    log_max_ms: 5_000.0,
                    log_n_bins: 120,
                },
                feature: FeatureParams {
                    basis: FeatureBasis::Energy4Ch,
                    display_mode: FeatureDisplayMode::Contours,
                    same_shank_clusters: true,
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let back: UiPrefs = serde_json::from_str(&serde_json::to_string(&prefs).unwrap()).unwrap();
        assert!(back.views.waveform.show_sd);
        assert!(back.views.isi.log_x);
        assert_eq!(back.views.isi.x_min_ms, 1.0);
        assert_eq!(back.views.isi.x_max_ms, 100.0);
        assert_eq!(back.views.isi.n_bins, 60);
        assert_eq!(back.views.isi.log_min_ms, 0.5);
        assert_eq!(back.views.isi.log_max_ms, 5_000.0);
        assert_eq!(back.views.isi.log_n_bins, 120);
        assert_eq!(back.views.feature, prefs.views.feature);
    }
}

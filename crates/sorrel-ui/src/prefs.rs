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
    /// Cluster/Similar table columns the user hid (by column name).
    pub hidden_cols: Vec<String>,
}

/// phy `FiringRateView` (v2.1): bin count over the recording (`:frn`) and
/// the time unit of the x axis.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RateParams {
    pub n_bins: usize,
    pub unit: TimeUnit,
}

impl Default for RateParams {
    fn default() -> Self {
        Self {
            n_bins: 200,
            unit: TimeUnit::Seconds,
        }
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WaveformParams {
    pub mode: WaveformMode,
    /// `o` (checkbox "overlap"): true superimposes the selected clusters in
    /// each channel box (default); false is phy's side-by-side slots.
    pub overlap: bool,
    /// `ctrl+l`: channel id labels.
    pub show_labels: bool,
    /// phy `n_spikes_waveforms`.
    pub n_spikes: usize,
    /// Box width/height factors (`ctrl+left/right`, `ctrl+wheel`).
    pub box_scale: [f32; 2],
    /// Probe spread factors (`shift+arrows`).
    pub probe_scale: [f32; 2],
    /// Amplitude factor (`ctrl+up/down`).
    pub amp_scale: f32,
}

impl Default for WaveformParams {
    fn default() -> Self {
        Self {
            mode: WaveformMode::default(),
            overlap: true,
            show_labels: true,
            n_spikes: 100,
            box_scale: [1.0, 1.0],
            probe_scale: [1.0, 1.0],
            amp_scale: 1.0,
        }
    }
}

/// phy `CorrelogramView` defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CcgParams {
    pub bin_ms: f32,
    pub window_ms: f32,
    pub refrac_ms: f32,
    /// `n`: one y scale for every cell instead of per-cell.
    pub uniform: bool,
}

impl Default for CcgParams {
    fn default() -> Self {
        Self {
            bin_ms: 1.0,
            window_ms: 50.0,
            refrac_ms: 2.0,
            uniform: false,
        }
    }
}

/// phy `ISIView` defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct IsiParams {
    pub x_min_ms: f32,
    pub x_max_ms: f32,
    pub n_bins: usize,
}

impl Default for IsiParams {
    fn default() -> Self {
        Self {
            x_min_ms: 0.0,
            x_max_ms: 50.0,
            n_bins: 50,
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
        std::fs::rename(tmp, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

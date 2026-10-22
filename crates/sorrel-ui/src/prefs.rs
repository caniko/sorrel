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
            ..Default::default()
        };
        prefs.save(dir.path()).unwrap();
        let back = UiPrefs::load(dir.path());
        assert_eq!(back.sort_by, Some(ClusterColumn::Quality));
        assert!(back.sort_descending && back.hide_noise && back.hp && !back.cmr);
        assert_eq!(back.window_ms, Some(250.0));
        assert_eq!(back.anchor, Some(7));
        assert!(crate::layout::missing_panes(back.tree.as_ref().unwrap()).is_empty());

        std::fs::write(UiPrefs::path(dir.path()), b"{ not json").unwrap();
        let fallback = UiPrefs::load(dir.path());
        assert!(fallback.tree.is_none());
        assert!(UiPrefs::load(Path::new("/nonexistent/sorrel"))
            .anchor
            .is_none());
    }
}

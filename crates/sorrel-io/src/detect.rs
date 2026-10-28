//! Backend detection shared by the binary and the Python bindings.

use anyhow::{bail, Result};
use std::path::Path;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Backend {
    /// phy / Kilosort directory (`spike_times.npy` + `spike_clusters.npy`).
    Kilosort,
    /// SpikeInterface `SortingAnalyzer` saved as `binary_folder`.
    SortingAnalyzer,
    /// `*.nwb` file (needs the `hdf5` feature to open).
    Nwb,
    /// Kilosort `rez.mat` / `rez2.mat` (needs the `hdf5` feature to open).
    Ks4Rez,
}

/// Detect which backend a path looks like. Phy markers take precedence
/// because they're the most specific; HDF5 backends are leaf-file based
/// (`.nwb` / `rez.mat`) so they trigger only when the path is a file.
pub fn detect_backend(root: &Path) -> Result<Backend> {
    if root.is_file() {
        let name = root.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let ext = root
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext == "nwb" {
            return Ok(Backend::Nwb);
        }
        if name == "rez.mat" || name == "rez2.mat" {
            return Ok(Backend::Ks4Rez);
        }
        bail!("unrecognised file {}; expected a directory", root.display());
    }
    if root.join("spike_times.npy").exists() && root.join("spike_clusters.npy").exists() {
        return Ok(Backend::Kilosort);
    }
    if (root.join("recording.json").exists() || root.join("binary.json").exists())
        && root.join("sorting").is_dir()
    {
        return Ok(Backend::SortingAnalyzer);
    }
    bail!(
        "could not detect a supported backend in {} \
         (looked for kilosort outputs and a SortingAnalyzer folder)",
        root.display()
    );
}

/// Lowercase sorter name for `root`, if one is recorded. Verified in SI 0.105.0:
/// `sorters/basesorter.py` writes `sorter_name` to `spikeinterface_log.json`, and
/// `core/core_tools.py` saves the same log as `__sorting_info__` in `sorting/annotations.json`.
pub fn detect_sorter(root: &Path, backend: Backend) -> Option<String> {
    let log_name = |dir: &Path| -> Option<String> {
        let log = std::fs::read(dir.join("spikeinterface_log.json")).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&log).ok()?;
        Some(v.get("sorter_name")?.as_str()?.to_ascii_lowercase())
    };
    if let Some(name) = [Some(root), root.parent()]
        .into_iter()
        .flatten()
        .find_map(log_name)
    {
        return Some(name);
    }
    match backend {
        Backend::Kilosort => {
            if root.join("ops.npy").exists() {
                Some("kilosort4".into())
            } else if root.join("rez.mat").exists() || root.join("cluster_KSLabel.tsv").exists() {
                Some("kilosort".into())
            } else {
                None
            }
        }
        Backend::SortingAnalyzer => {
            let name = std::fs::read(root.join("sorting/annotations.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .and_then(|v| {
                    v["__sorting_info__"]["log"]["sorter_name"]
                        .as_str()
                        .map(str::to_ascii_lowercase)
                });
            Some(name.unwrap_or_else(|| "spikeinterface".into()))
        }
        Backend::Ks4Rez => Some("kilosort".into()),
        Backend::Nwb => Some("nwb".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_by_marker_files_and_extensions() {
        let dir = tempfile::tempdir().unwrap();
        assert!(detect_backend(dir.path()).is_err());
        std::fs::write(dir.path().join("spike_times.npy"), b"").unwrap();
        std::fs::write(dir.path().join("spike_clusters.npy"), b"").unwrap();
        assert_eq!(detect_backend(dir.path()).unwrap(), Backend::Kilosort);

        let sa = tempfile::tempdir().unwrap();
        std::fs::write(sa.path().join("recording.json"), b"{}").unwrap();
        std::fs::create_dir(sa.path().join("sorting")).unwrap();
        assert_eq!(detect_backend(sa.path()).unwrap(), Backend::SortingAnalyzer);

        let nwb = dir.path().join("x.NWB");
        std::fs::write(&nwb, b"").unwrap();
        assert_eq!(detect_backend(&nwb).unwrap(), Backend::Nwb);
        let rez = dir.path().join("rez.mat");
        std::fs::write(&rez, b"").unwrap();
        assert_eq!(detect_backend(&rez).unwrap(), Backend::Ks4Rez);
    }

    #[test]
    fn detects_sorter_name() {
        let top = tempfile::tempdir().unwrap();
        let run = top.path().join("run");
        std::fs::create_dir(&run).unwrap();
        std::fs::write(run.join("ops.npy"), b"").unwrap();
        let log = top.path().join("spikeinterface_log.json");
        std::fs::write(&log, br#"{"sorter_name": "MountainSort5"}"#).unwrap();
        assert_eq!(
            detect_sorter(&run, Backend::Kilosort).as_deref(),
            Some("mountainsort5")
        );
        std::fs::remove_file(&log).unwrap();
        assert_eq!(
            detect_sorter(&run, Backend::Kilosort).as_deref(),
            Some("kilosort4")
        );

        let empty = tempfile::tempdir().unwrap();
        assert_eq!(detect_sorter(empty.path(), Backend::Kilosort), None);
        std::fs::write(empty.path().join("cluster_KSLabel.tsv"), b"").unwrap();
        assert_eq!(
            detect_sorter(empty.path(), Backend::Kilosort).as_deref(),
            Some("kilosort")
        );

        let rez = top.path().join("rez.mat");
        std::fs::write(&rez, b"").unwrap();
        assert_eq!(
            detect_sorter(&rez, Backend::Ks4Rez).as_deref(),
            Some("kilosort")
        );

        let sa = tempfile::tempdir().unwrap();
        std::fs::create_dir(sa.path().join("sorting")).unwrap();
        assert_eq!(
            detect_sorter(sa.path(), Backend::SortingAnalyzer).as_deref(),
            Some("spikeinterface")
        );
        std::fs::write(
            sa.path().join("sorting/annotations.json"),
            br#"{"__sorting_info__": {"log": {"sorter_name": "Kilosort2"}}}"#,
        )
        .unwrap();
        assert_eq!(
            detect_sorter(sa.path(), Backend::SortingAnalyzer).as_deref(),
            Some("kilosort2")
        );
    }
}

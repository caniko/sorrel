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
}

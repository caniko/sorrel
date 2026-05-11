//! Python bindings for sorrel.
//!
//! The Python side hands sorrel a directory to mmap. Two paths exist:
//!
//! * **SortingAnalyzer fast path** — when the user has a SpikeInterface
//!   `SortingAnalyzer` already saved as `binary_folder`, we launch sorrel
//!   directly on its `folder` (zero copy). When the analyzer is in-memory
//!   or zarr, we call `save_as("binary_folder")` into a tempdir; that's
//!   far lighter than a phy export (no PCs, no waveforms recompute).
//! * **Phy escape hatch** — `sorrel.export_phy(...)` still wraps SI's
//!   `export_to_phy` for users who specifically want a phy directory.
//!
//! Direct numpy zero-copy from a live Python kernel is still future work
//! (egui's main-thread requirements on macOS make in-process windowing
//! from a Jupyter kernel brittle).

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider};
use sorrel_io::DataProvider;
use std::path::PathBuf;
use std::process::Command;

/// Validate that `path` is a phy directory we can read. Returns
/// `(n_clusters, n_samples, sample_rate)` on success — useful for the
/// notebook to surface a quick summary before launching the GUI.
#[pyfunction]
fn inspect_phy(path: &str) -> PyResult<(u32, u64, f32)> {
    let provider = KilosortProvider::open(path, KilosortOpenParams::default())
        .map_err(|e| PyRuntimeError::new_err(format!("open phy directory: {e:#}")))?;
    Ok((
        provider.n_clusters(),
        provider.n_samples(),
        provider.sample_rate(),
    ))
}

/// Subprocess-launch the `sorrel` binary on `path`. Returns when the user
/// closes the window (blocking) so the notebook waits.
///
/// `binary` overrides the executable Sorrel finds via PATH; useful when
/// running from a development checkout.
#[pyfunction]
#[pyo3(signature = (path, binary=None, extra_args=None))]
fn launch(path: &str, binary: Option<&str>, extra_args: Option<Vec<String>>) -> PyResult<i32> {
    let exe: PathBuf = binary.map(PathBuf::from).unwrap_or_else(|| "sorrel".into());
    let mut cmd = Command::new(&exe);
    cmd.arg(path);
    if let Some(args) = extra_args {
        cmd.args(args);
    }
    let status = cmd
        .status()
        .map_err(|e| PyRuntimeError::new_err(format!("spawn {}: {e}", exe.display())))?;
    Ok(status.code().unwrap_or(-1))
}

/// Build a Cargo-style version string for the loaded native module.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(inspect_phy, m)?)?;
    m.add_function(wrap_pyfunction!(launch, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

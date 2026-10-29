//! Python bindings for sorrel.
//!
//! Two layers:
//!
//! * **In-process `Session`** — open a phy directory or a SortingAnalyzer
//!   `binary_folder`, read metrics and suggestions, relabel / merge /
//!   split through the same journal the GUI uses, and save. No window,
//!   no subprocess; this is the notebook and pipeline API.
//! * **GUI launcher** — `launch` subprocesses the `sorrel` binary (egui's
//!   main-thread requirements on macOS make in-process windowing from a
//!   Jupyter kernel brittle).
//!
//! Payloads cross the boundary as JSON strings decoded by the Python
//! wrapper: per-cluster scalars are tiny, and it keeps the native module
//! free of numpy at build time.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use sorrel_data::session::ApplyPhyLabel;
use sorrel_data::{
    collect_qc_rows, curation_json, open_phy, open_sorting_analyzer, rank_merge_candidates,
    rank_split_candidates, save_session, seed_phy, CurationCommand, JournalOpts, Method,
    PhyLabelOp, SuggestConfig,
};
use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider, PhyLabel};
use sorrel_io::{detect_backend, Backend, ClusterId, DataProvider, SortingAnalyzerProvider};
use std::path::{Path, PathBuf};
use std::process::Command;

fn err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

fn label_str(l: &PhyLabel) -> &'static str {
    l.as_str()
}

/// The monomorphised sessions a `#[pyclass]` (which cannot be generic) can
/// hold. One arm per backend the Python API supports.
enum AnySession {
    Phy(sorrel_data::Session<KilosortProvider>),
    Sa(sorrel_data::Session<SortingAnalyzerProvider>),
}

macro_rules! with_session {
    ($self:expr, $s:ident => $body:expr) => {
        match &$self.inner {
            AnySession::Phy($s) => $body,
            AnySession::Sa($s) => $body,
        }
    };
    (mut $self:expr, $s:ident => $body:expr) => {
        match &mut $self.inner {
            AnySession::Phy($s) => $body,
            AnySession::Sa($s) => $body,
        }
    };
}

/// A curation session opened in-process: metrics, suggestions and
/// journaled edits without the GUI.
#[pyclass(unsendable)]
struct Session {
    inner: AnySession,
    save_dir: PathBuf,
    /// phy dirs seal the journal on save (see `sorrel_data::save_and_seal`).
    reseal_on_save: bool,
}

fn parse_label(s: &str) -> PyResult<PhyLabelOp> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "good" => PhyLabelOp::SetGood,
        "mua" => PhyLabelOp::SetMua,
        "noise" => PhyLabelOp::SetNoise,
        "unsorted" => PhyLabelOp::SetUnsorted,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown label {other:?}; expected good, mua, noise or unsorted"
            )));
        }
    })
}

fn dispatch<P>(s: &mut sorrel_data::Session<P>, cmd: CurationCommand) -> PyResult<u64>
where
    P: DataProvider + ApplyPhyLabel,
{
    s.dispatch(cmd).map_err(err)?;
    Ok(s.version())
}

#[pymethods]
impl Session {
    /// Open `path` (phy directory or SortingAnalyzer binary_folder), seal
    /// and replay its journal. `journal` overrides the journal path;
    /// `discard_journal` deletes an existing one first.
    #[staticmethod]
    #[pyo3(signature = (path, journal=None, discard_journal=false))]
    fn open(path: &str, journal: Option<&str>, discard_journal: bool) -> PyResult<Self> {
        let root = Path::new(path);
        let opts = JournalOpts {
            path: journal.map(PathBuf::from),
            discard: discard_journal,
        };
        let (inner, reseal_on_save) = match detect_backend(root).map_err(err)? {
            Backend::Kilosort => {
                let mut s = open_phy(root, KilosortOpenParams::default(), &opts).map_err(err)?;
                seed_phy(&mut s).map_err(err)?;
                (AnySession::Phy(s), true)
            }
            Backend::SortingAnalyzer => {
                let mut s = open_sorting_analyzer(root, &opts).map_err(err)?;
                s.replay_journal().map_err(err)?;
                (AnySession::Sa(s), false)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "{other:?} is not supported in-process; use sorrel.launch() for the GUI"
                )));
            }
        };
        Ok(Self {
            inner,
            save_dir: root.to_path_buf(),
            reseal_on_save,
        })
    }

    fn n_clusters(&self) -> u32 {
        with_session!(self, s => s.n_clusters())
    }

    fn sample_rate(&self) -> f32 {
        with_session!(self, s => s.provider.sample_rate())
    }

    /// Monotonic curation-state version (changes on every journaled edit).
    fn version(&self) -> u64 {
        with_session!(self, s => s.version())
    }

    /// Label string per cluster, indexed by cluster id.
    fn labels(&self) -> Vec<&'static str> {
        with_session!(self, s => s.labels().iter().map(label_str).collect())
    }

    /// Spike sample indices of one cluster (time-sorted).
    fn spike_times(&self, cluster: u32) -> Vec<u64> {
        with_session!(self, s => s.spike_times(ClusterId(cluster)).iter().map(|t| t.0).collect())
    }

    /// Origin of `cluster` as text: `Auto[<sorter>]` or `Sorrel[LLM:x]`.
    fn origin(&self, cluster: u32) -> String {
        with_session!(self, s => s.origin_label(ClusterId(cluster)))
    }

    /// Direct parents of `cluster` (the split source or the merged-in units).
    fn parents(&self, cluster: u32) -> Vec<u32> {
        with_session!(self, s => s.parents(ClusterId(cluster)).iter().map(|p| p.0).collect())
    }

    /// Free-text note of `cluster` (empty when none).
    fn note(&self, cluster: u32) -> String {
        with_session!(self, s => s.note(ClusterId(cluster)).to_string())
    }

    /// The sorter named in `Auto[<sorter>]`.
    fn sorter(&self) -> String {
        with_session!(self, s => s.sorter().to_string())
    }

    /// `cluster_qc.tsv` rows as a JSON array of objects.
    fn qc_json(&self) -> String {
        let rows: Vec<serde_json::Value> =
            with_session!(self, s => collect_qc_rows(s).iter().map(|r| r.to_json()).collect());
        serde_json::Value::Array(rows).to_string()
    }

    /// `{"merges": [...], "splits": [...]}` as JSON.
    fn suggest_json(&self) -> String {
        let cfg = SuggestConfig::default();
        with_session!(self, s => serde_json::json!({
            "merges": rank_merge_candidates(s, &cfg),
            "splits": rank_split_candidates(s, &cfg),
        }))
        .to_string()
    }

    /// The SpikeInterface curation document for the current state as JSON.
    fn curation_json(&self) -> String {
        with_session!(self, s => curation_json::curation_document(s, label_str)).to_string()
    }

    fn set_label(&mut self, cluster: u32, label: &str) -> PyResult<u64> {
        let op = parse_label(label)?;
        let cmd = CurationCommand::Relabel {
            cluster: ClusterId(cluster),
            op,
        };
        with_session!(mut self, s => dispatch(s, cmd))
    }

    fn merge(&mut self, sources: Vec<u32>, target: u32) -> PyResult<u64> {
        let cmd = CurationCommand::Merge {
            sources: sources.into_iter().map(ClusterId).collect(),
            target: ClusterId(target),
        };
        with_session!(mut self, s => dispatch(s, cmd))
    }

    /// Move `spike_idx` (positions in the cluster's time-sorted spikes)
    /// into a new cluster made by `by` (`manual`, `algo:<name>`, `llm:<name>`);
    /// returns the new cluster id.
    #[pyo3(signature = (cluster, spike_idx, by="algo:python"))]
    fn split(&mut self, cluster: u32, spike_idx: Vec<u32>, by: &str) -> PyResult<u32> {
        let by = by.parse::<Method>().map_err(PyValueError::new_err)?;
        let cmd = CurationCommand::Split {
            cluster: ClusterId(cluster),
            spike_idx,
            new_cluster: ClusterId(u32::MAX),
            by,
        };
        with_session!(mut self, s => dispatch(s, cmd))?;
        Ok(self.n_clusters() - 1)
    }

    fn undo(&mut self) -> PyResult<u64> {
        with_session!(mut self, s => dispatch(s, CurationCommand::Undo))
    }

    fn redo(&mut self) -> PyResult<u64> {
        with_session!(mut self, s => dispatch(s, CurationCommand::Redo))
    }

    /// Write `spike_clusters.npy`, `cluster_group.tsv` and `curation.json`
    /// into the dataset directory; returns that directory.
    fn save(&mut self) -> PyResult<String> {
        let dir = self.save_dir.clone();
        let seal = self.reseal_on_save;
        with_session!(mut self, s => save_session(s, &dir, seal, label_str).map_err(err)?);
        Ok(dir.display().to_string())
    }

    /// Write the curated session as a complete phy directory at `out_dir`
    /// (`params.py` points at the raw file unless `copy_raw`); returns it.
    #[cfg(feature = "phy-export")]
    #[pyo3(signature = (out_dir, copy_raw = false))]
    fn export_phy(&self, out_dir: PathBuf, copy_raw: bool) -> PyResult<String> {
        with_session!(self, s => sorrel_data::export_phy(s, &out_dir, copy_raw, label_str).map_err(err)?);
        Ok(out_dir.display().to_string())
    }
}

/// Validate that `path` is a phy directory we can read. Returns
/// `(n_clusters, n_samples, sample_rate)` on success — useful for the
/// notebook to surface a quick summary before launching the GUI.
#[pyfunction]
fn inspect_phy(path: &str) -> PyResult<(u32, u64, f32)> {
    let provider = KilosortProvider::open(path, KilosortOpenParams::default())
        .map_err(|e| PyRuntimeError::new_err(format!("open phy directory: {e:#}")))?;
    Ok((
        provider.n_clusters(),
        provider.n_samples().0,
        provider.sample_rate(),
    ))
}

/// Subprocess-launch `sorrel run` on `path`. Blocks until the window closes,
/// so the notebook waits; returns at once when a sorrel window is already
/// running, which opens `path` as a new tab instead.
///
/// `binary` overrides the executable Sorrel finds via PATH; useful when
/// running from a development checkout.
#[pyfunction]
#[pyo3(signature = (path, binary=None, extra_args=None))]
fn launch(path: &str, binary: Option<&str>, extra_args: Option<Vec<String>>) -> PyResult<i32> {
    let exe: PathBuf = binary.map(PathBuf::from).unwrap_or_else(|| "sorrel".into());
    let mut cmd = Command::new(&exe);
    cmd.arg("run").arg(path);
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
    m.add_class::<Session>()?;
    m.add_function(wrap_pyfunction!(inspect_phy, m)?)?;
    m.add_function(wrap_pyfunction!(launch, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

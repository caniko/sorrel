//! Opening a dataset *with its journal correctly sealed*. Shared by the
//! binary and the Python bindings so both apply the same rules:
//!
//! - phy / Kilosort directories are sealed against the bytes of
//!   `spike_clusters.npy`, because that file is read back on open and
//!   rewritten on save (so saves reseal, see `save_and_reseal_journal`);
//! - every other backend reloads its original sort on open, so the seal is
//!   the provider's identity and the journal is never resealed.
//!
//! Sessions come back **unseeded**: run [`seed_phy`] / `replay_journal`
//! (or hand the closure to the GUI loader) before using them.

use crate::journal::{baseline_hash, Journal};
use crate::session::Session;
use anyhow::{Context, Result};
use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider};
use sorrel_io::{detect_sorter, Backend, DataProvider, SortingAnalyzerProvider};
use std::path::{Path, PathBuf};

/// Where the journal lives and whether to throw an existing one away.
#[derive(Debug, Clone, Default)]
pub struct JournalOpts {
    /// Explicit path; `None` = `<dir>/<stem>.journal`.
    pub path: Option<PathBuf>,
    /// Delete the existing journal before opening: the escape hatch for a
    /// journal sealed against data that has since changed.
    pub discard: bool,
}

/// Resolve the journal path: explicit, else `<dir>/<stem>.journal`, falling
/// back to the legacy `<stem>.sqlite` name when only that file exists (the
/// format was never SQLite; the name was historical).
pub fn journal_path(opts: &JournalOpts, dir: &Path, stem: &str) -> PathBuf {
    if let Some(p) = &opts.path {
        return p.clone();
    }
    let current = dir.join(format!("{stem}.journal"));
    let legacy = dir.join(format!("{stem}.sqlite"));
    if !current.exists() && legacy.exists() {
        log::warn!(
            "using legacy journal {}; rename it to {} at your convenience",
            legacy.display(),
            current.display()
        );
        return legacy;
    }
    current
}

/// Open the journal sealed against `baseline`, honouring `opts.discard`. A
/// baseline mismatch means the journal's edits would replay onto data they
/// were not recorded against, so this fails closed and names both ways out.
pub fn open_journal(
    opts: &JournalOpts,
    path: &Path,
    baseline: u64,
    n_clusters: u32,
) -> Result<Journal> {
    if opts.discard && path.exists() {
        std::fs::remove_file(path)
            .with_context(|| format!("discard journal {}", path.display()))?;
        log::warn!("discarded journal {}", path.display());
    }
    Journal::open_or_create(path, baseline, n_clusters).with_context(|| {
        format!(
            "journal {} was sealed against different data (or by an older sorrel without a \
             seal). If that curation is already saved, discard the journal; to keep unsaved \
             edits, restore the spike_clusters.npy the journal was recorded against.",
            path.display()
        )
    })
}

/// Open a phy directory. Unseeded; see [`seed_phy`].
pub fn open_phy(
    root: &Path,
    params: KilosortOpenParams,
    opts: &JournalOpts,
) -> Result<Session<KilosortProvider>> {
    let provider = KilosortProvider::open(root, params)?;
    let clusters_path = root.join("spike_clusters.npy");
    let baseline = baseline_hash(
        &std::fs::read(&clusters_path)
            .with_context(|| format!("read {}", clusters_path.display()))?,
    );
    let journal = open_journal(
        opts,
        &journal_path(opts, root, "sorrel"),
        baseline,
        provider.n_clusters(),
    )?;
    let mut session = Session::new(provider, journal);
    session.fill_detected_sorter(detect_sorter(root, Backend::Kilosort));
    Ok(session)
}

/// Pull the optional per-spike arrays into the index and replay the journal.
pub fn seed_phy(session: &mut Session<KilosortProvider>) -> Result<()> {
    session.seed_amplitudes();
    session.seed_templates();
    session.seed_pc_features();
    session.seed_template_waveforms();
    session.replay_journal()
}

/// Open a SortingAnalyzer `binary_folder`. Unseeded: call `replay_journal`.
pub fn open_sorting_analyzer(
    root: &Path,
    opts: &JournalOpts,
) -> Result<Session<SortingAnalyzerProvider>> {
    let provider = SortingAnalyzerProvider::open(root)?;
    let mut session = open_identity_sealed(provider, &journal_path(opts, root, "sorrel"), opts)?;
    session.fill_detected_sorter(detect_sorter(root, Backend::SortingAnalyzer));
    Ok(session)
}

/// Seal a non-phy provider's journal against its identity bytes. Shared by
/// SortingAnalyzer and the single-file (`.nwb`, `rez.mat`) backends.
pub fn open_identity_sealed<P: DataProvider>(
    provider: P,
    journal_path: &Path,
    opts: &JournalOpts,
) -> Result<Session<P>> {
    let baseline = baseline_hash(&provider.identity_bytes());
    let journal = open_journal(opts, journal_path, baseline, provider.n_clusters())?;
    Ok(Session::new(provider, journal))
}

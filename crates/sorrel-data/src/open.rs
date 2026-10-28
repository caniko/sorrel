//! Opening a dataset *with its journal correctly sealed*. Shared by the
//! binary and the Python bindings so both apply the same rules:
//!
//! - phy / Kilosort directories are sealed against the root's
//!   `spike_clusters.npy` (the sort as it was before sorrel's first save,
//!   see `sorrel_io::RootView`), and the files on disk must be the root or
//!   one of sorrel's saves (`Seal` records): anything else is an external
//!   edit that the next save would overwrite, so the open fails closed;
//! - every other backend reloads its original sort on open, so the seal is
//!   the provider's identity.
//!
//! Sessions come back **unseeded**: run [`seed_phy`] / `replay_journal`
//! (or hand the closure to the GUI loader) before using them.

use crate::journal::{artifact_names, artifacts_hash, baseline_hash, lock_journal, Journal};
use crate::session::Session;
use anyhow::{bail, Context, Result};
use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider, ROOT_DIR, ROOT_MANIFEST};
use sorrel_io::{detect_sorter, Backend, DataProvider, RootView, SortingAnalyzerProvider};
use std::path::{Path, PathBuf};

/// Where the journal lives and whether to throw an existing one away.
#[derive(Debug, Clone, Default)]
pub struct JournalOpts {
    /// Explicit path; `None` = `<dir>/<stem>.journal`.
    pub path: Option<PathBuf>,
    /// Set the existing journal aside before opening (renamed to
    /// `<journal>.<unix seconds>`): the escape hatch for a journal sealed
    /// against data that has since changed.
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
    // Locked first, so a live writer's journal is never set aside under it.
    let lock = lock_journal(path)?;
    open_journal_locked(lock, opts, path, baseline, n_clusters)
}

fn open_journal_locked(
    lock: std::fs::File,
    opts: &JournalOpts,
    path: &Path,
    baseline: u64,
    n_clusters: u32,
) -> Result<Journal> {
    discard_journal(opts, path)?;
    Journal::open_locked(lock, path, baseline, n_clusters).with_context(|| {
        format!(
            "journal {} was sealed against different data (or by an older sorrel without a \
             seal). If that curation is already saved, discard the journal (--discard-journal); to \
             keep unsaved edits, restore the spike_clusters.npy the journal was recorded against.",
            path.display()
        )
    })
}

/// Open a phy directory. Unseeded; see [`seed_phy`].
///
/// A root snapshot without its journal (discarded, deleted, or written by a
/// pre-v3 build without a manifest) is moved aside first, so the files as
/// they are become the new root.
pub fn open_phy(
    root: &Path,
    params: KilosortOpenParams,
    opts: &JournalOpts,
) -> Result<Session<KilosortProvider>> {
    let path = journal_path(opts, root, "sorrel");
    // Held from here (before anything is set aside) until the session drops.
    let lock = lock_journal(&path)?;
    discard_journal(opts, &path)?;
    let snap = root.join(ROOT_DIR);
    if snap.exists() && (!path.exists() || !snap.join(ROOT_MANIFEST).exists()) {
        // ponytail: the snapshot does not record its journal, so an alternate
        // path is refused while the default journal could own it.
        let default = journal_path(&JournalOpts::default(), root, "sorrel");
        if opts.path.is_some()
            && default != path
            && default.exists()
            && snap.join(ROOT_MANIFEST).exists()
        {
            bail!(
                "{}: {ROOT_DIR}/ belongs to {}; an alternate journal would set it aside from \
                 under that journal. Open without --journal, or set that journal aside first \
                 (--discard-journal without --journal).",
                root.display(),
                default.display()
            );
        }
        retire_snapshot(root)?;
    }
    // Rolls an interrupted save forward, then reads the root.
    let provider = KilosortProvider::open(root, params)?;
    let view = RootView::open(root)?;
    let clusters_path = view.path("spike_clusters.npy");
    let baseline = baseline_hash(
        &std::fs::read(&clusters_path)
            .with_context(|| format!("read {}", clusters_path.display()))?,
    );
    let journal = open_journal_locked(lock, opts, &path, baseline, provider.n_clusters())?;
    // Every cluster_*.tsv counts: a phy edit to a custom label file is as
    // real as one to cluster_group.tsv.
    let names = artifact_names(root, &view, &[])?;
    let top_view = RootView::top(root);
    let top = artifacts_hash(&names, |n| top_view.path(n))?;
    let at_root = artifacts_hash(&names, |n| view.path(n))?;
    if top != at_root && !journal.seals()?.iter().any(|s| s.1 == top) {
        bail!(
            "{}: spike_clusters.npy or a cluster_*.tsv changed outside sorrel since its last \
             save (phy?). The next sorrel save would overwrite that work, so this journal is \
             not opened. To curate from the files as they are now, reopen with \
             --discard-journal: the journal and {ROOT_DIR}/ are set aside, not deleted.",
            root.display()
        );
    }
    let mut session = Session::new(provider, journal);
    session.set_top_artifacts(top);
    session.fill_detected_sorter(detect_sorter(root, Backend::Kilosort));
    Ok(session)
}

/// Move `<root>/sorrel_original` to `sorrel_original.<unix seconds>` (never
/// deleted: it may hold the only copy of the pre-rescue arrays).
pub fn retire_snapshot(root: &Path) -> Result<()> {
    set_aside(&root.join(ROOT_DIR))
}

/// Rename `path` to `<path>.<unix seconds>` (plus `-<n>` if taken).
fn set_aside(path: &Path) -> Result<()> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let named = |suffix: String| {
        let mut p = path.as_os_str().to_owned();
        p.push(suffix);
        PathBuf::from(p)
    };
    let to = (0..)
        .map(|i| match i {
            0 => named(format!(".{secs}")),
            i => named(format!(".{secs}-{i}")),
        })
        .find(|p| !p.exists())
        .expect("an unused name");
    std::fs::rename(path, &to)
        .with_context(|| format!("move {} aside to {}", path.display(), to.display()))?;
    log::warn!("moved {} aside to {}", path.display(), to.display());
    Ok(())
}

/// `--discard-journal`: set the journal aside (it holds the whole action
/// tree, so it is renamed, not deleted).
fn discard_journal(opts: &JournalOpts, path: &Path) -> Result<()> {
    if opts.discard && path.exists() {
        set_aside(path)?;
    }
    Ok(())
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

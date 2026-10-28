//! Native, single-file curation journal: the persistent action tree.
//!
//! `Journal` is the *only* place sorrel persists curation state while
//! running. The phy artifacts (`spike_clusters.npy`, `cluster_group.tsv`)
//! are export-on-save: they exist for interchange with phy and downstream
//! tools, never as the running source of truth. State at any node is the
//! root (the provider's original sort) plus a replay of the path to it.
//!
//! # File format (`<root>/sorrel.journal`)
//!
//! ```text
//! magic        | 8 bytes  | b"SORREL\x00\x03"
//! version      | u32 LE   | 3
//! flags        | u32 LE   | reserved (0)
//! baseline     | u64 LE   | xxh3-64 of the root's spike_clusters.npy (or
//!              |          | the provider identity for non-phy backends)
//! n_clusters_0 | u32 LE   | n_clusters at the root (sanity)
//! reserved     | u32 LE   | 0
//! ----- records (repeating) -----
//! payload_len  | u32 LE   | MessagePack payload length in bytes
//! payload      | [u8; len]| MessagePack-encoded [`Record`]
//! ```
//!
//! Records are append-only: a [`Record::Node`] is an edit (a child of the
//! head at the time), a [`Record::Head`] moves the checked-out node (undo,
//! redo, jump) and a [`Record::Seal`] marks a save.
//!
//! Older files (magic `SORREL\x00\x02`, version 1 or 2: a flat list of
//! commands with `Undo`/`Redo` records) are rewritten to v3 on open, see
//! [`fold_legacy`]. The magic changed so a pre-v3 build refuses a v3 file
//! ("bad magic") instead of truncating records it cannot decode. Any change
//! to [`Record`] or the types it carries needs a new version for the same
//! reason: [`Journal::replay_at`] truncates at the first undecodable record.
//!
//! Crash-safety contract: every `append` writes the record in one
//! `write_all` and fsync's the file before returning; a failed write is cut
//! back to the last committed record. On reopen, replay reads records until
//! either EOF or a partial trailing record (whose length doesn't match the
//! remaining bytes) — that one is skipped and the file is truncated to the
//! last fully-committed record.
//!
//! Single writer: [`Journal::open_or_create`] takes an exclusive advisory
//! lock on `<journal>.lock` for the life of the `Journal` (fails with
//! [`JournalBusy`]), and only that owner ever truncates. Anyone else (a
//! remote tail, `journal dump`) reads with [`Journal::read_only_records`],
//! which never modifies the file.

use crate::command::{Actor, CurationCommand};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sorrel_io::RootView;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Magic header. Bumping the trailing byte invalidates older files.
pub const MAGIC: &[u8; 8] = b"SORREL\x00\x03";
/// Magic of v1/v2 journals, upgraded on open.
const LEGACY_MAGIC: &[u8; 8] = b"SORREL\x00\x02";
/// On-disk format version.
pub const FORMAT_VERSION: u32 = 3;
/// Header byte size: magic (8) + version (4) + flags (4) + baseline (8)
/// + n_clusters_0 (4) + reserved (4) = 32 bytes.
pub const HEADER_SIZE: u64 = 32;

/// Hash a `spike_clusters` byte slice (whatever dtype it was on disk) into
/// the 64-bit baseline marker.
pub fn baseline_hash(spike_clusters_bytes: &[u8]) -> u64 {
    use xxhash_rust::xxh3::xxh3_64;
    xxh3_64(spike_clusters_bytes)
}

/// True for the files phy writes on save that sorrel seals:
/// `spike_clusters.npy` and every `cluster_*.tsv` (`cluster_group.tsv`,
/// custom labels, sorrel's sidecars, metrics).
fn is_artifact(name: &str) -> bool {
    name == "spike_clusters.npy" || (name.starts_with("cluster_") && name.ends_with(".tsv"))
}

/// Sorted names [`artifacts_hash`] covers for `root`: the two core files,
/// every top-level `cluster_*.tsv` on disk or captured in the root snapshot
/// (`view`), and the `staged` names of a save about to publish. The same
/// set must be used for the on-disk, root and staged hashes to be comparable.
pub fn artifact_names(root: &Path, view: &RootView, staged: &[String]) -> Result<Vec<String>> {
    let mut names: std::collections::BTreeSet<String> = ["spike_clusters.npy", "cluster_group.tsv"]
        .map(String::from)
        .into();
    for e in std::fs::read_dir(root).with_context(|| format!("list {}", root.display()))? {
        names.extend(e?.file_name().into_string().ok().filter(|n| is_artifact(n)));
    }
    let captured = view.captured().map(|c| c.0);
    names.extend(
        captured
            .chain(staged.iter().map(String::as_str))
            .filter(|n| is_artifact(n))
            .map(String::from),
    );
    Ok(names.into_iter().collect())
}

/// Hash of `names` (see [`artifact_names`]) as `path_of` resolves them. A
/// [`Record::Seal`] carries it, so an open can tell sorrel's own saves from an
/// external edit. Only `spike_clusters.npy` must exist; any other missing
/// file hashes as absent.
pub fn artifacts_hash(names: &[String], path_of: impl Fn(&str) -> PathBuf) -> Result<u64> {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    for n in names {
        let p = path_of(n);
        h.update(&(n.len() as u64).to_le_bytes());
        h.update(n.as_bytes());
        match std::fs::read(&p) {
            Ok(b) => {
                h.update(&(b.len() as u64).to_le_bytes());
                h.update(&b);
            }
            Err(e) if n != "spike_clusters.npy" && e.kind() == std::io::ErrorKind::NotFound => {
                h.update(&u64::MAX.to_le_bytes())
            }
            Err(e) => return Err(e).with_context(|| format!("hash {}", p.display())),
        }
    }
    Ok(h.digest())
}

/// One journal record.
// Records are transient (decoded, then moved into the session's tree), so
// the size gap between `Node` and the rest costs nothing worth a box.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Record {
    /// An edit, child of `parent` (`None` = the root). `id` is the node's
    /// index in record order; `cmd` is always a forward command.
    Node {
        id: u64,
        parent: Option<u64>,
        actor: Actor,
        /// Unix milliseconds (0 for nodes upgraded from a v2 journal).
        ts_ms: u64,
        cmd: CurationCommand,
    },
    /// The checked-out node moved (`None` = the root).
    Head { node: Option<u64> },
    /// A save of the state at `node` whose `spike_clusters.npy` +
    /// `cluster_group.tsv` hash to `artifacts` (see [`artifacts_hash`]).
    /// Appended before the files are published.
    Seal { node: Option<u64>, artifacts: u64 },
}

/// Hash of one encoded record, chained into the node keys that
/// `Session::version` reports.
pub fn record_hash(parent_key: u64, rec: &Record) -> u64 {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    h.update(&parent_key.to_le_bytes());
    h.update(&rmp_serde::to_vec(rec).unwrap_or_default());
    h.digest()
}

/// Single-file append-only journal.
///
/// Append-only journal — single owner, no interior mutability needed.
#[derive(Debug)]
pub struct Journal {
    path: PathBuf,
    file: File,
    /// Byte offset after the last committed record: where a failed append
    /// is cut back to.
    len: u64,
    /// A failed append could not be cut back; the file may hold garbage.
    broken: bool,
    baseline: u64,
    n_clusters_at_baseline: u32,
    /// Fully committed records on disk, kept in memory so it never touches
    /// the file. Monotonic, so it doubles as a sync cursor.
    n_records: u64,
    /// Exclusive advisory lock on `<journal>.lock`; released on drop.
    _lock: File,
    /// Test seam: fail (after writing half) the append after this many more.
    #[cfg(test)]
    pub(crate) fail_after: Option<usize>,
}

/// Another process (or `Journal`) holds the journal's writer lock.
#[derive(Debug, thiserror::Error)]
#[error("journal {} is already open{}", .path.display(), .pid.map_or(String::new(), |p| format!(" by pid {p}")))]
pub struct JournalBusy {
    pub path: PathBuf,
    /// Pid recorded in `<journal>.lock` by the owner, when readable.
    pub pid: Option<u32>,
}

fn lock_path(journal: &Path) -> PathBuf {
    let mut p = journal.as_os_str().to_owned();
    p.push(".lock");
    p.into()
}

/// Take the exclusive writer lock for `journal` (`<journal>.lock`, kept
/// across renames of the journal itself). Fails with [`JournalBusy`].
/// ponytail: the lock file is left behind on drop (deleting it races a
/// concurrent opener); advisory locks, so only sorrel's own opens honour it.
pub(crate) fn lock_journal(journal: &Path) -> Result<File> {
    let lp = lock_path(journal);
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lp)
        .with_context(|| format!("open {}", lp.display()))?;
    let locked = f.try_lock();
    match locked {
        Ok(()) => {
            f.set_len(0)?;
            // Best effort: the pid is only for the busy message.
            let _ = write!(f, "{}", std::process::id()).and_then(|()| f.flush());
            Ok(f)
        }
        Err(std::fs::TryLockError::WouldBlock) => {
            let pid = std::fs::read_to_string(&lp)
                .ok()
                .and_then(|s| s.trim().parse().ok());
            Err(JournalBusy {
                path: journal.to_path_buf(),
                pid,
            }
            .into())
        }
        Err(std::fs::TryLockError::Error(e)) => {
            Err(e).with_context(|| format!("lock {}", lp.display()))
        }
    }
}

/// Best effort (a directory cannot be opened on Windows).
fn sync_dir(path: &Path) {
    if let Some(dir) = path.parent() {
        let _ = File::open(dir).and_then(|d| d.sync_all());
    }
}

impl Journal {
    /// Convenience: open or create with `baseline = 0` and
    /// `n_clusters_at_baseline = 0`. The seal is effectively disabled —
    /// suitable for tests / mock providers, **not** for production use
    /// against a real phy directory.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_or_create(path, 0, 0)
    }

    /// Open an existing journal or create a new one sealed against the
    /// supplied baseline. When the file already exists, the stored baseline
    /// is checked against `expected_baseline`; mismatch returns an error
    /// so the caller can decide whether to discard, fail loudly, or
    /// reset. A v1/v2 file is upgraded to v3 first.
    ///
    /// Takes the exclusive writer lock first, held until the `Journal` is
    /// dropped; a second opener gets a [`JournalBusy`] error. A new journal
    /// is created as tmp + fsync + rename + directory fsync, so a crash
    /// never leaves a partial header under the real name.
    pub fn open_or_create(path: &Path, expected_baseline: u64, n_clusters: u32) -> Result<Self> {
        let lock = lock_journal(path)?;
        Self::open_locked(lock, path, expected_baseline, n_clusters)
    }

    /// [`Self::open_or_create`] with the lock already taken (so a caller can
    /// hold it across moving the old journal aside).
    pub(crate) fn open_locked(
        lock: File,
        path: &Path,
        expected_baseline: u64,
        n_clusters: u32,
    ) -> Result<Self> {
        if path.exists() {
            return Self::open_existing(lock, path, expected_baseline);
        }
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".new");
        let tmp = PathBuf::from(tmp);
        {
            let mut f =
                File::create(&tmp).with_context(|| format!("create journal {}", tmp.display()))?;
            f.write_all(&header(expected_baseline, n_clusters))?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)
            .with_context(|| format!("create journal {}", path.display()))?;
        sync_dir(path);
        Self::reopen(lock, path, expected_baseline, n_clusters, 0)
    }

    fn reopen(
        lock: File,
        path: &Path,
        baseline: u64,
        n_clusters: u32,
        n_records: u64,
    ) -> Result<Self> {
        let file = OpenOptions::new()
            .append(true)
            .open(path)
            .with_context(|| format!("reopen journal {}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            len: file.metadata()?.len(),
            file,
            broken: false,
            baseline,
            n_clusters_at_baseline: n_clusters,
            n_records,
            _lock: lock,
            #[cfg(test)]
            fail_after: None,
        })
    }

    fn open_existing(lock: File, path: &Path, expected_baseline: u64) -> Result<Self> {
        let (legacy, baseline, n_clusters, version) = read_header(path)?;
        if baseline != expected_baseline {
            bail!(
                "{}: spike_clusters.npy has changed since this journal was sealed \
                 (journal baseline 0x{:016x}, current 0x{:016x}). Refusing to \
                 replay stale operations.",
                path.display(),
                baseline,
                expected_baseline
            );
        }
        if legacy {
            upgrade_legacy(path, baseline, n_clusters, version)?;
        }
        // Counting records also truncates any partial trailing record, so
        // the appender below starts at a clean boundary. We hold the lock,
        // so this is the only place a journal is ever truncated.
        let n_records = read_records::<Record>(path, true, false)?.len() as u64;
        Self::reopen(lock, path, baseline, n_clusters, n_records)
    }

    pub fn baseline(&self) -> u64 {
        self.baseline
    }

    pub fn n_clusters_at_baseline(&self) -> u32 {
        self.n_clusters_at_baseline
    }

    /// Number of fully committed records.
    pub fn n_records(&self) -> u64 {
        self.n_records
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one record: prefix and payload in one `write_all`, then
    /// fsync, so the call only returns after the bytes are durable. On any
    /// write or fsync error the file is cut back to the last committed
    /// record, so later appends and readers see a clean journal.
    pub fn append(&mut self, rec: &Record) -> Result<()> {
        ensure!(
            !self.broken,
            "{}: a failed write could not be undone; reopen the journal",
            self.path.display()
        );
        let payload = rmp_serde::to_vec(rec)?;
        let len = u32::try_from(payload.len()).context("journal record exceeds u32::MAX bytes")?;
        let mut buf = Vec::with_capacity(4 + payload.len());
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&payload);
        if let Err(e) = self.write_durably(&buf) {
            if let Err(t) = self.file.set_len(self.len) {
                self.broken = true;
                log::error!(
                    "{}: cannot cut back a failed append: {t}",
                    self.path.display()
                );
            }
            return Err(e);
        }
        self.len += buf.len() as u64;
        self.n_records += 1;
        Ok(())
    }

    fn write_durably(&mut self, buf: &[u8]) -> Result<()> {
        #[cfg(test)]
        if let Some(n) = self.fail_after.as_mut() {
            if *n == 0 {
                self.fail_after = None;
                self.file.write_all(&buf[..buf.len() / 2])?;
                bail!("injected write failure");
            }
            *n -= 1;
        }
        self.file.write_all(buf).context("write journal")?;
        self.file.sync_all().context("fsync journal")
    }

    /// Every fully-committed record in insertion order. Never modifies the
    /// file (the owner's tail is always clean; see [`Self::append`]).
    pub fn replay(&self) -> Result<Vec<Record>> {
        Self::read_only_records(&self.path)
    }

    /// The `artifacts` hash of every [`Record::Seal`], oldest first, with
    /// the node it sealed.
    pub fn seals(&self) -> Result<Vec<(Option<u64>, u64)>> {
        Ok(self
            .replay()?
            .into_iter()
            .filter_map(|r| match r {
                Record::Seal { node, artifacts } => Some((node, artifacts)),
                _ => None,
            })
            .collect())
    }

    /// Records of a journal file read by a non-owner (a remote tail,
    /// `journal dump`, a guest comparing logs): everything up to the last
    /// complete, decodable record, never modifying the file or taking the
    /// lock, so it is safe against a live writer. Refuses a pre-v3 file.
    pub fn read_only_records(path: &Path) -> Result<Vec<Record>> {
        let (legacy, ..) = read_header(path)?;
        ensure!(
            !legacy,
            "{}: pre-v3 journal; open the dataset once to upgrade it",
            path.display()
        );
        read_records(path, false, false)
    }

    /// Alias of [`Self::read_only_records`]; it used to truncate a torn tail,
    /// which only the lock-holding owner may do (see [`Self::open_or_create`]).
    pub fn replay_at(path: &Path) -> Result<Vec<Record>> {
        Self::read_only_records(path)
    }
}

fn header(baseline: u64, n_clusters: u32) -> Vec<u8> {
    let mut h = Vec::with_capacity(HEADER_SIZE as usize);
    h.extend_from_slice(MAGIC);
    h.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    h.extend_from_slice(&0u32.to_le_bytes()); // flags
    h.extend_from_slice(&baseline.to_le_bytes());
    h.extend_from_slice(&n_clusters.to_le_bytes());
    h.extend_from_slice(&0u32.to_le_bytes()); // reserved
    h
}

/// `(legacy, baseline, n_clusters, version)` from a journal header; refuses
/// unknown magic and versions.
fn read_header(path: &Path) -> Result<(bool, u64, u32, u32)> {
    let mut f = File::open(path).with_context(|| format!("open journal {}", path.display()))?;
    let mut header = [0u8; HEADER_SIZE as usize];
    f.read_exact(&mut header)
        .with_context(|| format!("read journal header {}", path.display()))?;
    let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let legacy = match &header[0..8] {
        m if m == MAGIC && version == FORMAT_VERSION => false,
        m if m == LEGACY_MAGIC && (1..=2).contains(&version) => true,
        m if m == MAGIC || m == LEGACY_MAGIC => bail!(
            "{}: unsupported journal version {} (this build reads 1-{})",
            path.display(),
            version,
            FORMAT_VERSION
        ),
        _ => bail!("{}: not a sorrel journal (bad magic)", path.display()),
    };
    let baseline = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let n_clusters = u32::from_le_bytes(header[24..28].try_into().unwrap());
    Ok((legacy, baseline, n_clusters, version))
}

/// Decode every committed record after the header; with `truncate`, cut
/// the file back to the last good record (a partial or undecodable tail).
/// With `strict`, a complete record that does not decode is an error rather
/// than the end of the log (a partial trailing record still just ends it).
fn read_records<T: serde::de::DeserializeOwned>(
    path: &Path,
    truncate: bool,
    strict: bool,
) -> Result<Vec<T>> {
    let mut f = File::open(path).with_context(|| format!("open journal {}", path.display()))?;
    let total = f.metadata()?.len();
    if total < HEADER_SIZE {
        bail!("{}: journal smaller than header", path.display());
    }
    f.seek(SeekFrom::Start(HEADER_SIZE))?;

    let mut out = Vec::new();
    let mut last_good_pos = HEADER_SIZE;
    loop {
        let pos = f.stream_position()?;
        if pos >= total {
            break;
        }
        let mut len_bytes = [0u8; 4];
        if f.read_exact(&mut len_bytes).is_err() {
            // Partial length prefix — discard.
            break;
        }
        let len = u32::from_le_bytes(len_bytes) as u64;
        if pos + 4 + len > total {
            // Partial payload — never durable. Truncate.
            log::warn!(
                "{}: truncating partial record at offset {pos}",
                path.display()
            );
            break;
        }
        let mut payload = vec![0u8; len as usize];
        f.read_exact(&mut payload)?;
        match rmp_serde::from_slice::<T>(&payload) {
            Ok(rec) => {
                out.push(rec);
                last_good_pos = f.stream_position()?;
            }
            Err(e) => {
                ensure!(
                    !strict,
                    "{}: record at offset {pos} does not decode ({e}); refusing to drop it \
                     and everything after it",
                    path.display()
                );
                log::warn!(
                    "{}: skipping malformed record at offset {pos}: {e}",
                    path.display(),
                );
                break;
            }
        }
    }

    // Truncate trailing garbage so subsequent appends start clean.
    if truncate && last_good_pos < total {
        let f = OpenOptions::new()
            .write(true)
            .open(path)
            .with_context(|| format!("open for truncate {}", path.display()))?;
        f.set_len(last_good_pos)?;
        f.sync_all()?;
    }
    Ok(out)
}

/// Rewrite a v1/v2 journal as v3 in place: decoded, [`fold_legacy`]ed,
/// written to `<path>.tmp`, fsynced and renamed over the original. Refuses
/// (touching nothing) when a record does not decode, and keeps the original
/// as `<path>.v<N>.bak`.
/// ponytail: a corrupt length prefix that points past EOF reads as a torn
/// tail and is dropped (the .bak keeps it); decode the rest to catch that.
fn upgrade_legacy(path: &Path, baseline: u64, n_clusters: u32, version: u32) -> Result<()> {
    let records = fold_legacy(read_records(path, false, true)?)
        .with_context(|| format!("upgrade journal {}", path.display()))?;
    let mut bak = path.as_os_str().to_owned();
    bak.push(format!(".v{version}.bak"));
    let bak = PathBuf::from(bak);
    // A hard link keeps the original inode; no window without a journal.
    let _ = std::fs::remove_file(&bak);
    std::fs::hard_link(path, &bak)
        .or_else(|_| std::fs::copy(path, &bak).map(drop))
        .with_context(|| format!("back up {} to {}", path.display(), bak.display()))?;
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    {
        let mut f = BufWriter::new(
            File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?,
        );
        f.write_all(&header(baseline, n_clusters))?;
        for r in &records {
            let payload = rmp_serde::to_vec(r)?;
            f.write_all(&(payload.len() as u32).to_le_bytes())?;
            f.write_all(&payload)?;
        }
        f.into_inner()?.sync_all()?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("replace {}", path.display()))?;
    sync_dir(path);
    log::info!(
        "upgraded journal {} to v3 (original kept as {})",
        path.display(),
        bak.display()
    );
    Ok(())
}

/// Fold a flat v1/v2 command list into v3 records: forward commands become
/// nodes (children of the head), `Undo` moves the head to its parent and
/// `Redo` to the most recently visited child. An edit after an undo starts
/// a branch, so the undone commands the old format discarded are kept.
/// Upgraded nodes are `Human` with `ts_ms = 0`.
pub fn fold_legacy(cmds: Vec<CurationCommand>) -> Result<Vec<Record>> {
    let mut parent: Vec<Option<u64>> = Vec::new();
    // Most recently visited child, per node; slot 0 is the root.
    let mut last_child: Vec<Option<u64>> = vec![None];
    let slot = |n: Option<u64>| n.map_or(0, |n| n as usize + 1);
    let mut head = None;
    let mut out = Vec::with_capacity(cmds.len());
    for cmd in cmds {
        match cmd {
            CurationCommand::Undo => {
                let Some(h) = head else {
                    bail!("Undo with empty history");
                };
                head = parent[h as usize];
                out.push(Record::Head { node: head });
            }
            CurationCommand::Redo => {
                let Some(c) = last_child[slot(head)] else {
                    bail!("Redo with nothing to redo");
                };
                head = Some(c);
                out.push(Record::Head { node: head });
            }
            cmd => {
                let id = parent.len() as u64;
                parent.push(head);
                last_child.push(None);
                last_child[slot(head)] = Some(id);
                out.push(Record::Node {
                    id,
                    parent: head,
                    actor: Actor::Human,
                    ts_ms: 0,
                    cmd,
                });
                head = Some(id);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::PhyLabelOp;
    use sorrel_io::ClusterId;

    fn cmd(c: u32, op: PhyLabelOp) -> CurationCommand {
        CurationCommand::Relabel {
            cluster: ClusterId(c),
            op,
        }
    }

    fn node(id: u64, parent: Option<u64>, c: u32) -> Record {
        Record::Node {
            id,
            parent,
            actor: Actor::Human,
            ts_ms: 1,
            cmd: cmd(c, PhyLabelOp::SetGood),
        }
    }

    /// Write a v1/v2 file by hand: legacy header, then raw command payloads.
    fn write_legacy<T: Serialize>(p: &Path, version: u32, baseline: u64, cmds: &[T]) {
        let mut bytes = header(baseline, 3);
        bytes[..8].copy_from_slice(LEGACY_MAGIC);
        bytes[8..12].copy_from_slice(&version.to_le_bytes());
        for c in cmds {
            let rec = rmp_serde::to_vec(c).unwrap();
            bytes.extend((rec.len() as u32).to_le_bytes());
            bytes.extend(rec);
        }
        std::fs::write(p, bytes).unwrap();
    }

    #[test]
    fn create_then_replay_returns_records_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        let mut j = Journal::open_or_create(&p, 0xDEAD_BEEF, 3).unwrap();
        j.append(&node(0, None, 0)).unwrap();
        j.append(&Record::Head { node: None }).unwrap();
        j.append(&Record::Seal {
            node: None,
            artifacts: 7,
        })
        .unwrap();
        let recs = j.replay().unwrap();
        assert_eq!(recs.len(), 3);
        assert!(matches!(
            recs[0],
            Record::Node {
                id: 0,
                parent: None,
                ..
            }
        ));
        assert!(matches!(recs[1], Record::Head { node: None }));
        assert_eq!(j.seals().unwrap(), vec![(None, 7)]);
        assert_eq!(j.n_records(), 3);
    }

    #[test]
    fn reopen_checks_the_baseline() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        {
            let mut j = Journal::open_or_create(&p, 0xABCD, 5).unwrap();
            j.append(&node(0, None, 0)).unwrap();
        }
        let j = Journal::open_or_create(&p, 0xABCD, 5).unwrap();
        assert_eq!((j.n_records(), j.n_clusters_at_baseline()), (1, 5));
        drop(j);
        let msg = Journal::open_or_create(&p, 0xBBBB, 5)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("changed since this journal was sealed"));
    }

    #[test]
    fn rejects_corrupted_magic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        std::fs::write(&p, b"NOTSORRELxxxxxxxxxxxxxxxxxxxxxxxxxx").unwrap();
        assert!(Journal::open_or_create(&p, 0, 0).is_err());
    }

    #[test]
    fn replay_truncates_partial_trailing_record() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        {
            let mut j = Journal::open_or_create(&p, 0xF00D, 1).unwrap();
            j.append(&node(0, None, 0)).unwrap();
            j.append(&node(1, Some(0), 1)).unwrap();
        }
        let good = std::fs::metadata(&p).unwrap().len();
        {
            let mut f = OpenOptions::new().append(true).open(&p).unwrap();
            f.write_all(&100u32.to_le_bytes()).unwrap(); // claims 100 bytes
            f.write_all(b"abc").unwrap(); // …but only writes 3
        }
        let j = Journal::open_or_create(&p, 0xF00D, 1).unwrap();
        assert_eq!(j.n_records(), 2, "partial record discarded");
        assert_eq!(std::fs::metadata(&p).unwrap().len(), good);
    }

    /// A v2 file with an undo, a redo, and an edit after an undo becomes a
    /// tree: the undone node survives as a branch. The rewritten file has
    /// the v3 magic, so a pre-v3 build (which checks for `SORREL\0\2`)
    /// refuses it instead of truncating.
    #[test]
    fn v2_journal_folds_into_a_tree() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        let a = cmd(0, PhyLabelOp::SetGood);
        let b = cmd(1, PhyLabelOp::SetMua);
        let c = cmd(2, PhyLabelOp::SetNoise);
        use CurationCommand::{Redo, Undo};
        write_legacy(&p, 2, 9, &[a, b, Undo, Redo, Undo, c, Undo, Undo, Redo]);
        let j = Journal::open_or_create(&p, 9, 3).unwrap();
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(&bytes[..8], MAGIC);
        assert_ne!(&bytes[..8], LEGACY_MAGIC);
        let recs = j.replay().unwrap();
        let shape: Vec<String> = recs
            .iter()
            .map(|r| match r {
                Record::Node { id, parent, .. } => format!("n{id}<{parent:?}"),
                Record::Head { node } => format!("h{node:?}"),
                Record::Seal { .. } => "s".into(),
            })
            .collect();
        assert_eq!(
            shape,
            [
                "n0<None",
                "n1<Some(0)",
                "hSome(0)",
                "hSome(1)",
                "hSome(0)",
                "n2<Some(0)",
                "hSome(0)",
                "hNone",
                "hSome(0)",
            ]
        );
        // The upgraded file reopens as v3 without another rewrite.
        drop(j);
        let mtime = std::fs::metadata(&p).unwrap().modified().unwrap();
        Journal::open_or_create(&p, 9, 3).unwrap();
        assert_eq!(std::fs::metadata(&p).unwrap().modified().unwrap(), mtime);
    }

    /// A v1 record (a `Split` without `by`) still decodes through the fold.
    #[test]
    fn v1_journal_upgrades_through_the_fold() {
        #[derive(Serialize)]
        enum Old {
            Split {
                cluster: ClusterId,
                spike_idx: Vec<u32>,
                new_cluster: ClusterId,
            },
        }
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        write_legacy(
            &p,
            1,
            7,
            &[Old::Split {
                cluster: ClusterId(0),
                spike_idx: vec![1],
                new_cluster: ClusterId(3),
            }],
        );
        let j = Journal::open_or_create(&p, 7, 3).unwrap();
        let recs = j.replay().unwrap();
        assert!(matches!(
            &recs[..],
            [Record::Node {
                cmd: CurationCommand::Split {
                    by: crate::command::Method::Manual,
                    ..
                },
                ..
            }]
        ));
    }

    /// A legacy file is never truncated by `replay_at` (which only reads
    /// v3), and a mismatched baseline refuses before any upgrade.
    #[test]
    fn legacy_file_is_left_alone_until_a_matching_open() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        write_legacy(&p, 2, 9, &[cmd(0, PhyLabelOp::SetGood)]);
        let before = std::fs::read(&p).unwrap();
        assert!(Journal::replay_at(&p).is_err());
        assert!(Journal::open_or_create(&p, 8, 3).is_err());
        assert_eq!(std::fs::read(&p).unwrap(), before);
    }

    #[test]
    fn future_version_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        drop(Journal::open_or_create(&p, 7, 3).unwrap());
        let mut f = OpenOptions::new().write(true).open(&p).unwrap();
        f.seek(SeekFrom::Start(8)).unwrap();
        f.write_all(&4u32.to_le_bytes()).unwrap();
        assert!(Journal::open_or_create(&p, 7, 3).is_err());
    }

    #[test]
    fn fold_rejects_impossible_legacy_sequences() {
        assert!(fold_legacy(vec![CurationCommand::Undo]).is_err());
        assert!(fold_legacy(vec![cmd(0, PhyLabelOp::SetGood), CurationCommand::Redo]).is_err());
    }

    #[test]
    fn artifacts_hash_covers_every_cluster_tsv() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let hash = || {
            let names = artifact_names(d, &RootView::top(d), &[]).unwrap();
            artifacts_hash(&names, |n| d.join(n))
        };
        std::fs::write(d.join("spike_clusters.npy"), b"ab").unwrap();
        let missing = hash().unwrap();
        std::fs::write(d.join("cluster_group.tsv"), b"").unwrap();
        let empty = hash().unwrap();
        std::fs::write(d.join("cluster_group.tsv"), b"x").unwrap();
        let x = hash().unwrap();
        assert!(missing != empty && empty != x);
        // A custom label file, created then edited, counts too; other files do not.
        std::fs::write(d.join("cluster_mylabel.tsv"), b"a").unwrap();
        let custom = hash().unwrap();
        std::fs::write(d.join("cluster_mylabel.tsv"), b"b").unwrap();
        assert!(custom != x && hash().unwrap() != custom);
        std::fs::write(d.join("notes.txt"), b"b").unwrap();
        let before = hash().unwrap();
        std::fs::write(d.join("notes.txt"), b"c").unwrap();
        assert_eq!(hash().unwrap(), before);
        std::fs::remove_file(d.join("spike_clusters.npy")).unwrap();
        assert!(hash().is_err());
    }

    #[test]
    fn second_opener_gets_journal_busy_until_the_first_drops() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        let j = Journal::open_or_create(&p, 1, 3).unwrap();
        let err = Journal::open_or_create(&p, 1, 3).unwrap_err();
        let busy = err.downcast_ref::<JournalBusy>().expect("typed error");
        assert_eq!(busy.pid, Some(std::process::id()));
        assert!(err.to_string().contains("already open by pid"));
        drop(j);
        Journal::open_or_create(&p, 1, 3).unwrap();
    }

    /// A non-owner never truncates: a torn tail (here, a record cut short
    /// the way a reader racing a large append sees it) stays as it is.
    #[test]
    fn read_only_records_never_modifies_a_torn_journal() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        let mut j = Journal::open_or_create(&p, 1, 3).unwrap();
        j.append(&node(0, None, 0)).unwrap();
        let mut f = OpenOptions::new().append(true).open(&p).unwrap();
        f.write_all(&100u32.to_le_bytes()).unwrap();
        f.write_all(b"abc").unwrap();
        let torn = std::fs::read(&p).unwrap();
        assert_eq!(Journal::read_only_records(&p).unwrap().len(), 1);
        assert_eq!(Journal::replay_at(&p).unwrap().len(), 1);
        assert_eq!(std::fs::read(&p).unwrap(), torn, "file untouched");
    }

    /// A reader polling a live journal while big records are appended never
    /// shortens the file, and the writer's journal stays fully readable.
    #[test]
    fn concurrent_reader_never_tears_large_appends() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        let mut j = Journal::open_or_create(&p, 1, 3).unwrap();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader = {
            let (p, stop) = (p.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut last = 0;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let n = Journal::read_only_records(&p).unwrap().len();
                    assert!(n >= last, "records vanished: {n} < {last}");
                    last = n;
                }
            })
        };
        let big = |id| Record::Node {
            id,
            parent: id.checked_sub(1),
            actor: Actor::Human,
            ts_ms: 0,
            cmd: CurationCommand::Note {
                cluster: sorrel_io::ClusterId(0),
                text: "x".repeat(2 << 20),
            },
        };
        for id in 0..8 {
            j.append(&big(id)).unwrap();
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        reader.join().unwrap();
        assert_eq!(Journal::read_only_records(&p).unwrap().len(), 8);
    }

    /// A write that dies half way (ENOSPC) is cut back to the last committed
    /// record; later appends and readers see a clean journal.
    #[test]
    fn failed_append_is_cut_back_and_later_appends_work() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        let mut j = Journal::open_or_create(&p, 1, 3).unwrap();
        j.append(&node(0, None, 0)).unwrap();
        let good = std::fs::metadata(&p).unwrap().len();
        j.fail_after = Some(0);
        assert!(j.append(&node(1, Some(0), 1)).is_err());
        assert_eq!(std::fs::metadata(&p).unwrap().len(), good);
        assert_eq!(j.n_records(), 1);
        j.append(&node(1, Some(0), 2)).unwrap();
        assert_eq!(Journal::read_only_records(&p).unwrap().len(), 2);
        drop(j);
        assert_eq!(Journal::open_or_create(&p, 1, 3).unwrap().n_records(), 2);
    }

    /// An undecodable middle record refuses the upgrade and touches nothing.
    #[test]
    fn upgrade_refuses_a_corrupt_middle_record_and_keeps_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        let a = cmd(0, PhyLabelOp::SetGood);
        write_legacy(&p, 2, 9, &[a.clone(), a.clone(), a]);
        // Corrupt the second record's payload.
        let mut bytes = std::fs::read(&p).unwrap();
        let first = u32::from_le_bytes(bytes[32..36].try_into().unwrap()) as usize;
        let second = 32 + 4 + first + 4;
        bytes[second..second + 3].fill(0xc1); // 0xc1 is never valid MessagePack
        std::fs::write(&p, &bytes).unwrap();
        let err = Journal::open_or_create(&p, 9, 3).unwrap_err();
        assert!(format!("{err:#}").contains("does not decode"), "{err:#}");
        assert_eq!(std::fs::read(&p).unwrap(), bytes);
        assert!(!dir.path().join("j.sorrel.v2.bak").exists());
    }

    #[test]
    fn upgrade_keeps_the_original_as_a_backup() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        write_legacy(&p, 2, 9, &[cmd(0, PhyLabelOp::SetGood)]);
        let before = std::fs::read(&p).unwrap();
        drop(Journal::open_or_create(&p, 9, 3).unwrap());
        assert_eq!(
            std::fs::read(dir.path().join("j.sorrel.v2.bak")).unwrap(),
            before
        );
    }

    /// A crashed create leaves only `<journal>.new`; the real name never
    /// holds a partial header, and the next open starts clean.
    #[test]
    fn crashed_create_never_shows_a_partial_header() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        let new = dir.path().join("j.sorrel.new");
        // The header is built under the `.new` name: if that cannot be made,
        // nothing appears under the real one.
        std::fs::create_dir(&new).unwrap();
        assert!(Journal::open_or_create(&p, 1, 3).is_err());
        assert!(!p.exists());
        std::fs::remove_dir(&new).unwrap();
        std::fs::write(&new, &header(1, 3)[..10]).unwrap();
        assert!(!p.exists());
        let j = Journal::open_or_create(&p, 1, 3).unwrap();
        assert_eq!(std::fs::metadata(&p).unwrap().len(), HEADER_SIZE);
        assert_eq!(j.n_records(), 0);
    }
}

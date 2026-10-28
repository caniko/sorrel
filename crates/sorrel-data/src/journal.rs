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
//! Crash-safety contract: every `append` flushes + fsync's the file before
//! returning, so a power loss never produces a half-written record. On
//! reopen, replay reads records until either EOF or a partial trailing
//! record (whose length doesn't match the remaining bytes) — that one is
//! skipped and the file is truncated to the last fully-committed record.

use crate::command::{Actor, CurationCommand};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
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

/// Hash of the two files phy itself writes on save (`spike_clusters.npy`
/// and `cluster_group.tsv`, the latter possibly missing). A [`Record::Seal`]
/// carries it, so an open can tell sorrel's own saves from an external edit.
pub fn artifacts_hash(spike_clusters: &Path, cluster_group: &Path) -> Result<u64> {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    for (i, p) in [spike_clusters, cluster_group].into_iter().enumerate() {
        match std::fs::read(p) {
            Ok(b) => {
                h.update(&(b.len() as u64).to_le_bytes());
                h.update(&b);
            }
            Err(e) if i == 1 && e.kind() == std::io::ErrorKind::NotFound => {
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
    writer: BufWriter<File>,
    baseline: u64,
    n_clusters_at_baseline: u32,
    /// Fully committed records on disk, kept in memory so it never touches
    /// the file. Monotonic, so it doubles as a sync cursor.
    n_records: u64,
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
    pub fn open_or_create(path: &Path, expected_baseline: u64, n_clusters: u32) -> Result<Self> {
        if path.exists() {
            return Self::open_existing(path, expected_baseline);
        }
        let mut f =
            File::create(path).with_context(|| format!("create journal {}", path.display()))?;
        f.write_all(&header(expected_baseline, n_clusters))?;
        f.sync_all()?;
        Self::reopen(path, expected_baseline, n_clusters, 0)
    }

    fn reopen(path: &Path, baseline: u64, n_clusters: u32, n_records: u64) -> Result<Self> {
        let f = OpenOptions::new()
            .append(true)
            .open(path)
            .with_context(|| format!("reopen journal {}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            writer: BufWriter::new(f),
            baseline,
            n_clusters_at_baseline: n_clusters,
            n_records,
        })
    }

    fn open_existing(path: &Path, expected_baseline: u64) -> Result<Self> {
        let (legacy, baseline, n_clusters) = read_header(path)?;
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
            upgrade_legacy(path, baseline, n_clusters)?;
        }
        // Counting records also truncates any partial trailing record, so
        // the appender below starts at a clean boundary.
        let n_records = Self::replay_at(path)?.len() as u64;
        Self::reopen(path, baseline, n_clusters, n_records)
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

    /// Append one record. Buffers, flushes, and fsync's so the call only
    /// returns after the bytes are durable.
    pub fn append(&mut self, rec: &Record) -> Result<()> {
        let payload = rmp_serde::to_vec(rec)?;
        let len = u32::try_from(payload.len()).context("journal record exceeds u32::MAX bytes")?;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&payload)?;
        self.writer.flush()?;
        self.writer.get_ref().sync_all().context("fsync journal")?;
        self.n_records += 1;
        Ok(())
    }

    /// Replay every fully-committed record in insertion order.
    /// Truncates a partial trailing record if one is present (crash
    /// recovery — the partial record was never durable).
    pub fn replay(&self) -> Result<Vec<Record>> {
        Self::replay_at(&self.path)
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

    /// [`Journal::replay`] for a journal file that is not open. Used when
    /// opening (to count committed records) and by headless inspection.
    /// Refuses a file that is not v3 rather than truncating it.
    pub fn replay_at(path: &Path) -> Result<Vec<Record>> {
        let (legacy, _, _) = read_header(path)?;
        ensure!(
            !legacy,
            "{}: pre-v3 journal; open the dataset once to upgrade it",
            path.display()
        );
        read_records(path, true)
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

/// `(legacy, baseline, n_clusters)` from a journal header; refuses unknown
/// magic and versions.
fn read_header(path: &Path) -> Result<(bool, u64, u32)> {
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
    Ok((legacy, baseline, n_clusters))
}

/// Decode every committed record after the header; with `truncate`, cut
/// the file back to the last good record (a partial or undecodable tail).
fn read_records<T: serde::de::DeserializeOwned>(path: &Path, truncate: bool) -> Result<Vec<T>> {
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
/// written to `<path>.tmp`, fsynced and renamed over the original.
fn upgrade_legacy(path: &Path, baseline: u64, n_clusters: u32) -> Result<()> {
    let records = fold_legacy(read_records(path, false)?)
        .with_context(|| format!("upgrade journal {}", path.display()))?;
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
    if let Some(dir) = path.parent() {
        // ponytail: best effort (a directory cannot be opened on Windows).
        let _ = File::open(dir).and_then(|d| d.sync_all());
    }
    log::info!("upgraded journal {} to v3", path.display());
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
    fn artifacts_hash_covers_both_files() {
        let dir = tempfile::tempdir().unwrap();
        let (sc, cg) = (dir.path().join("sc"), dir.path().join("cg"));
        std::fs::write(&sc, b"ab").unwrap();
        let missing = artifacts_hash(&sc, &cg).unwrap();
        std::fs::write(&cg, b"").unwrap();
        let empty = artifacts_hash(&sc, &cg).unwrap();
        std::fs::write(&cg, b"x").unwrap();
        let x = artifacts_hash(&sc, &cg).unwrap();
        assert!(missing != empty && empty != x);
        std::fs::remove_file(&sc).unwrap();
        assert!(artifacts_hash(&sc, &cg).is_err());
    }
}

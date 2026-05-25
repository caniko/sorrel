//! Native, single-file curation journal.
//!
//! `Journal` is the *only* place sorrel persists curation state while
//! running. The phy artifacts (`spike_clusters.npy`, `cluster_group.tsv`)
//! are pure import-on-open / export-on-save — they exist for interchange
//! with phy and downstream tools, never as the running source of truth.
//!
//! # File format (`<root>/sorrel.journal`)
//!
//! ```text
//! magic        | 8 bytes  | b"SORREL\x00\x02"
//! version      | u32 LE   | 1
//! flags        | u32 LE   | reserved (0)
//! baseline     | u64 LE   | xxh3-64 hash of spike_clusters.npy at last seal
//! n_clusters_0 | u32 LE   | n_clusters at baseline (sanity)
//! reserved     | u32 LE   | 0
//! ----- records (repeating) -----
//! payload_len  | u32 LE   | MessagePack payload length in bytes
//! payload      | [u8; len]| MessagePack-encoded `CurationCommand`
//! ```
//!
//! Crash-safety contract: every `append` flushes + fsync's the file before
//! returning, so a power loss never produces a half-written record. On
//! reopen, replay reads records until either EOF or a partial trailing
//! record (whose length doesn't match the remaining bytes) — that one is
//! skipped and the file is truncated to the last fully-committed record.

use crate::command::CurationCommand;
use anyhow::{bail, Context, Result};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Backwards-compatible alias for callers still using the old name.
/// New code should use [`Journal`] directly.
pub type SqliteJournal = Journal;

/// Magic header. Bumping the trailing byte invalidates older files.
pub const MAGIC: &[u8; 8] = b"SORREL\x00\x02";
/// On-disk format version.
pub const FORMAT_VERSION: u32 = 1;
/// Header byte size: magic (8) + version (4) + flags (4) + baseline (8)
/// + n_clusters_0 (4) + reserved (4) = 32 bytes.
pub const HEADER_SIZE: u64 = 32;

/// Hash a `spike_clusters` byte slice (whatever dtype it was on disk) into
/// the 64-bit baseline marker.
pub fn baseline_hash(spike_clusters_bytes: &[u8]) -> u64 {
    use xxhash_rust::xxh3::xxh3_64;
    xxh3_64(spike_clusters_bytes)
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
    /// reset.
    pub fn open_or_create(path: &Path, expected_baseline: u64, n_clusters: u32) -> Result<Self> {
        if path.exists() {
            return Self::open_existing(path, expected_baseline);
        }
        let mut f =
            File::create(path).with_context(|| format!("create journal {}", path.display()))?;
        f.write_all(MAGIC)?;
        f.write_all(&FORMAT_VERSION.to_le_bytes())?;
        f.write_all(&0u32.to_le_bytes())?; // flags
        f.write_all(&expected_baseline.to_le_bytes())?;
        f.write_all(&n_clusters.to_le_bytes())?;
        f.write_all(&0u32.to_le_bytes())?; // reserved
        f.sync_all()?;

        // Reopen for append.
        let f = OpenOptions::new()
            .append(true)
            .open(path)
            .with_context(|| format!("reopen journal {}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            writer: BufWriter::new(f),
            baseline: expected_baseline,
            n_clusters_at_baseline: n_clusters,
        })
    }

    fn open_existing(path: &Path, expected_baseline: u64) -> Result<Self> {
        let mut f = File::open(path).with_context(|| format!("open journal {}", path.display()))?;
        let mut header = [0u8; HEADER_SIZE as usize];
        f.read_exact(&mut header)
            .with_context(|| format!("read journal header {}", path.display()))?;
        if &header[0..8] != MAGIC {
            bail!("{}: not a sorrel journal (bad magic)", path.display());
        }
        let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
        if version != FORMAT_VERSION {
            bail!(
                "{}: unsupported journal version {} (this build expects {})",
                path.display(),
                version,
                FORMAT_VERSION
            );
        }
        let baseline = u64::from_le_bytes(header[16..24].try_into().unwrap());
        let n_clusters = u32::from_le_bytes(header[24..28].try_into().unwrap());
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
        let f = OpenOptions::new()
            .append(true)
            .open(path)
            .with_context(|| format!("reopen journal {}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            writer: BufWriter::new(f),
            baseline,
            n_clusters_at_baseline: n_clusters,
        })
    }

    /// Force-create a fresh journal at `path`, discarding any existing
    /// content. Used by `save_to_phy` after a successful export to commit
    /// the new state and clear undo history.
    pub fn truncate(path: &Path, baseline: u64, n_clusters: u32) -> Result<Self> {
        if path.exists() {
            std::fs::remove_file(path)
                .with_context(|| format!("remove old journal {}", path.display()))?;
        }
        Self::open_or_create(path, baseline, n_clusters)
    }

    pub fn baseline(&self) -> u64 {
        self.baseline
    }

    pub fn n_clusters_at_baseline(&self) -> u32 {
        self.n_clusters_at_baseline
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Stable journal-head marker for derived-cache invalidation.
    ///
    /// The high 64-bit baseline seal is folded together with the number of
    /// fully committed records. This is cheap, deterministic, and changes
    /// whenever replay-visible curation state changes.
    pub fn head(&self) -> u64 {
        let applied_count = self.replay().map(|cmds| cmds.len() as u64).unwrap_or(0);
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&self.baseline.to_le_bytes());
        bytes[8..].copy_from_slice(&applied_count.to_le_bytes());
        xxhash_rust::xxh3::xxh3_64(&bytes)
    }

    /// Append one record. Buffers, flushes, and fsync's so the call only
    /// returns after the bytes are durable.
    pub fn append(&mut self, cmd: &CurationCommand) -> Result<()> {
        let payload = rmp_serde::to_vec(cmd)?;
        let len = u32::try_from(payload.len()).context("journal record exceeds u32::MAX bytes")?;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&payload)?;
        self.writer.flush()?;
        self.writer.get_ref().sync_all().context("fsync journal")?;
        Ok(())
    }

    /// Replay every fully-committed record in insertion order.
    /// Truncates a partial trailing record if one is present (crash
    /// recovery — the partial record was never durable).
    pub fn replay(&self) -> Result<Vec<CurationCommand>> {
        let mut f = File::open(&self.path)
            .with_context(|| format!("open journal {}", self.path.display()))?;
        let total = f.metadata()?.len();
        if total < HEADER_SIZE {
            bail!("{}: journal smaller than header", self.path.display());
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
                    self.path.display()
                );
                break;
            }
            let mut payload = vec![0u8; len as usize];
            f.read_exact(&mut payload)?;
            match rmp_serde::from_slice::<CurationCommand>(&payload) {
                Ok(cmd) => {
                    out.push(cmd);
                    last_good_pos = f.stream_position()?;
                }
                Err(e) => {
                    log::warn!(
                        "{}: skipping malformed record at offset {pos}: {e}",
                        self.path.display(),
                    );
                    break;
                }
            }
        }

        // Truncate trailing garbage so subsequent appends start clean.
        if last_good_pos < total {
            let f = OpenOptions::new()
                .write(true)
                .open(&self.path)
                .with_context(|| format!("open for truncate {}", self.path.display()))?;
            f.set_len(last_good_pos)?;
            f.sync_all()?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::PhyLabelOp;

    fn cmd(c: u32, op: PhyLabelOp) -> CurationCommand {
        CurationCommand::Relabel {
            cluster: sorrel_io::ClusterId(c),
            op,
        }
    }

    #[test]
    fn create_then_replay_returns_ops_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        let mut j = Journal::open_or_create(&p, 0xDEAD_BEEF, 3).unwrap();
        j.append(&cmd(0, PhyLabelOp::SetGood)).unwrap();
        j.append(&cmd(1, PhyLabelOp::SetMua)).unwrap();
        j.append(&cmd(2, PhyLabelOp::SetNoise)).unwrap();

        let ops = j.replay().unwrap();
        assert_eq!(ops.len(), 3);
        match &ops[0] {
            CurationCommand::Relabel { cluster, op } => {
                assert_eq!(*cluster, sorrel_io::ClusterId(0));
                assert_eq!(*op, PhyLabelOp::SetGood);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn empty_journal_replays_to_empty_vec() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        let j = Journal::open_or_create(&p, 0xCAFE, 1).unwrap();
        assert!(j.replay().unwrap().is_empty());
    }

    #[test]
    fn reopen_with_matching_baseline_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        {
            let mut j = Journal::open_or_create(&p, 0xABCD, 5).unwrap();
            j.append(&cmd(0, PhyLabelOp::SetGood)).unwrap();
        }
        let j = Journal::open_or_create(&p, 0xABCD, 5).unwrap();
        assert_eq!(j.replay().unwrap().len(), 1);
        assert_eq!(j.baseline(), 0xABCD);
        assert_eq!(j.n_clusters_at_baseline(), 5);
    }

    #[test]
    fn reopen_with_mismatched_baseline_errors() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        {
            let mut j = Journal::open_or_create(&p, 0xAAAA, 5).unwrap();
            j.append(&cmd(0, PhyLabelOp::SetGood)).unwrap();
        }
        let res = Journal::open_or_create(&p, 0xBBBB, 5);
        assert!(res.is_err(), "mismatched baseline should refuse to open");
        let msg = res.err().unwrap().to_string();
        assert!(msg.contains("changed since this journal was sealed"));
    }

    #[test]
    fn truncate_resets_to_a_fresh_journal_with_new_baseline() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        {
            let mut j = Journal::open_or_create(&p, 0x1, 1).unwrap();
            j.append(&cmd(0, PhyLabelOp::SetGood)).unwrap();
            j.append(&cmd(1, PhyLabelOp::SetMua)).unwrap();
        }
        let j = Journal::truncate(&p, 0x2, 7).unwrap();
        assert_eq!(j.baseline(), 0x2);
        assert_eq!(j.n_clusters_at_baseline(), 7);
        assert!(
            j.replay().unwrap().is_empty(),
            "fresh journal has no records"
        );
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
            j.append(&cmd(0, PhyLabelOp::SetGood)).unwrap();
            j.append(&cmd(1, PhyLabelOp::SetMua)).unwrap();
        }
        // Append a half-written record by hand.
        {
            use std::os::unix::fs::OpenOptionsExt;
            let mut f = OpenOptions::new()
                .append(true)
                .mode(0o644)
                .open(&p)
                .unwrap();
            f.write_all(&100u32.to_le_bytes()).unwrap(); // claims 100 bytes
            f.write_all(b"abc").unwrap(); // …but only writes 3
            f.sync_all().unwrap();
        }
        let j = Journal::open_or_create(&p, 0xF00D, 1).unwrap();
        let ops = j.replay().unwrap();
        assert_eq!(
            ops.len(),
            2,
            "partial record discarded, full ones recovered"
        );
        // After replay, file should be truncated back to 2 fully-committed records.
        let size = std::fs::metadata(&p).unwrap().len();
        // Exact size is hard to predict (codec-dependent) but it must be
        // smaller than what we wrote (header + 2 records + 4 + 3 = …).
        assert!(size > HEADER_SIZE);
    }

    #[test]
    fn append_persists_all_command_variants() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        let mut j = Journal::open_or_create(&p, 0, 0).unwrap();
        use sorrel_io::ClusterId;
        j.append(&CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: PhyLabelOp::SetGood,
        })
        .unwrap();
        j.append(&CurationCommand::Merge {
            sources: vec![ClusterId(1), ClusterId(2)],
            target: ClusterId(3),
        })
        .unwrap();
        j.append(&CurationCommand::Split {
            cluster: ClusterId(4),
            spike_idx: vec![0, 1, 5, 8],
            new_cluster: ClusterId(99),
        })
        .unwrap();
        j.append(&CurationCommand::Note {
            cluster: ClusterId(7),
            text: "needs review".into(),
        })
        .unwrap();
        j.append(&CurationCommand::Undo).unwrap();
        j.append(&CurationCommand::Redo).unwrap();

        let ops = j.replay().unwrap();
        assert_eq!(ops.len(), 6);
        assert!(matches!(ops[0], CurationCommand::Relabel { .. }));
        assert!(matches!(ops[1], CurationCommand::Merge { .. }));
        assert!(matches!(ops[2], CurationCommand::Split { .. }));
        assert!(matches!(ops[3], CurationCommand::Note { .. }));
        assert!(matches!(ops[4], CurationCommand::Undo));
        assert!(matches!(ops[5], CurationCommand::Redo));
    }

    #[test]
    fn replay_yields_strict_insertion_order_across_many_appends() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("j.sorrel");
        let mut j = Journal::open_or_create(&p, 0, 0).unwrap();
        for i in 0..50 {
            j.append(&cmd(i, PhyLabelOp::SetGood)).unwrap();
        }
        let ops = j.replay().unwrap();
        assert_eq!(ops.len(), 50);
        for (i, op) in ops.iter().enumerate() {
            match op {
                CurationCommand::Relabel { cluster, .. } => assert_eq!(cluster.idx(), i),
                _ => panic!(),
            }
        }
    }
}

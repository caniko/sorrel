//! Export the live curation state into a phy-compatible directory.
//!
//! Save semantics: the `.sorrel` journal is the running source of truth.
//! `save_to_phy` is an *export* — it writes:
//! - `spike_clusters.npy` — `uint32` array of length `n_spikes`, the
//!   per-spike cluster assignment.
//! - `cluster_group.tsv` — one row per cluster with a non-default label.
//!
//! After a successful export, the journal is **resealed**: its baseline
//! hash is updated to match the freshly-written `spike_clusters.npy`, and
//! its append history is cleared. This means save = commit, the same
//! mental model as Word / Photoshop / Git: prior undo state is no longer
//! reachable, but the user's data is durable.
//!
//! Both files are written atomically (write-then-rename) so a crash
//! mid-save never leaves the directory half-written.

use crate::journal::{baseline_hash, Journal};
use crate::session::Session;
use anyhow::{Context, Result};
use sorrel_io::npy::write_1d_u32_atomic;
use sorrel_io::DataProvider;
use std::path::Path;

/// Write `spike_clusters.npy` and `cluster_group.tsv` into `root`.
///
/// `label_str` maps each backend label to the string phy expects in the TSV
/// (`"good"`, `"mua"`, `"noise"`, `"unsorted"`). Unsorted entries are
/// elided so a phy directory that has never been curated round-trips with
/// no TSV at all.
///
/// Does **not** touch the journal. Most callers want
/// [`save_and_reseal_journal`] instead, which writes the artifacts and
/// then commits the journal so future opens treat the saved state as
/// canonical.
pub fn save_to_phy<P, F>(session: &Session<P>, root: &Path, label_str: F) -> Result<()>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    let spike_clusters = session.cluster_index().spike_clusters();
    let path = root.join("spike_clusters.npy");
    write_1d_u32_atomic(&path, spike_clusters)
        .with_context(|| format!("write {}", path.display()))?;

    let tsv_path = root.join("cluster_group.tsv");
    let mut text = String::from("cluster_id\tgroup\n");
    for (i, lbl) in session.labels().iter().enumerate() {
        let s = label_str(lbl);
        if !s.is_empty() && s != "unsorted" {
            text.push_str(&format!("{i}\t{s}\n"));
        }
    }
    let tmp = with_tmp_suffix(&tsv_path);
    std::fs::write(&tmp, &text)
        .with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, &tsv_path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), tsv_path.display()))?;
    Ok(())
}

/// Save the curation state to phy artifacts AND reseal the journal at
/// `journal_path`. The previous journal (with all its undo history) is
/// truncated; the new journal's baseline hash matches the freshly-written
/// `spike_clusters.npy`. Subsequent reopens load the new state cleanly
/// with empty undo history.
pub fn save_and_reseal_journal<P, F>(
    session: &Session<P>,
    root: &Path,
    journal_path: &Path,
    label_str: F,
) -> Result<Journal>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    save_to_phy(session, root, label_str)?;

    // Hash the bytes we just wrote (post-rename, post-fsync).
    let bytes = std::fs::read(root.join("spike_clusters.npy"))
        .context("hash freshly-written spike_clusters.npy")?;
    let new_baseline = baseline_hash(&bytes);
    let n_clusters = session.n_clusters();
    Journal::truncate(journal_path, new_baseline, n_clusters)
}

fn with_tmp_suffix(path: &Path) -> std::path::PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".tmp");
    p.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::PhyLabelOp;
    use crate::session::ApplyPhyLabel;
    use crate::{CurationCommand, Session, SqliteJournal};
    use sorrel_io::npy::read_1d_u32;
    use sorrel_io::{ClusterId, DataProvider, SampleIndex, TraceSamples, TraceSlice};

    /// Tiny stand-in provider for the writer tests.
    struct MockProvider {
        spikes: Vec<Vec<SampleIndex>>,
        labels: Vec<MockLabel>,
    }

    #[repr(u8)]
    #[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
    enum MockLabel {
        #[default]
        Unsorted = 0,
        Good = 1,
        Mua = 2,
        Noise = 3,
    }

    impl DataProvider for MockProvider {
        type Label = MockLabel;
        fn sample_rate(&self) -> f32 {
            1000.0
        }
        fn n_channels(&self) -> u32 {
            0
        }
        fn n_samples(&self) -> SampleIndex {
            0
        }
        fn n_clusters(&self) -> u32 {
            self.spikes.len() as u32
        }
        fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
            self.spikes
                .get(c as usize)
                .map(Vec::as_slice)
                .unwrap_or(&[])
        }
        fn trace(&self, _: SampleIndex, _: u32) -> TraceSlice<'_> {
            TraceSlice {
                start: 0,
                n_channels: 0,
                samples: TraceSamples::I16(&[]),
            }
        }
        fn initial_labels(&self) -> Vec<Self::Label> {
            self.labels.clone()
        }
    }

    impl ApplyPhyLabel for MockProvider {
        fn label_from_op(op: PhyLabelOp) -> Self::Label {
            match op {
                PhyLabelOp::SetUnsorted => MockLabel::Unsorted,
                PhyLabelOp::SetGood => MockLabel::Good,
                PhyLabelOp::SetMua => MockLabel::Mua,
                PhyLabelOp::SetNoise => MockLabel::Noise,
            }
        }
    }

    fn label_str(l: &MockLabel) -> &'static str {
        match l {
            MockLabel::Unsorted => "unsorted",
            MockLabel::Good => "good",
            MockLabel::Mua => "mua",
            MockLabel::Noise => "noise",
        }
    }

    fn fresh_session() -> (Session<MockProvider>, tempfile::TempDir) {
        let provider = MockProvider {
            spikes: vec![vec![10, 30, 150], vec![20, 200], vec![100]],
            labels: vec![MockLabel::Unsorted; 3],
        };
        let dir = tempfile::tempdir().unwrap();
        let journal = SqliteJournal::open(&dir.path().join("j.sqlite")).unwrap();
        (Session::new(provider, journal), dir)
    }

    #[test]
    fn save_writes_spike_clusters_matching_current_state() {
        let (mut s, dir) = fresh_session();
        // Merge cluster 0 into 2: every spike in the global arrays that was
        // in 0 should now read as 2 in the file we write.
        s.dispatch(CurationCommand::Merge { sources: vec![0], target: 2 })
            .unwrap();

        save_to_phy(&s, dir.path(), label_str).unwrap();

        let written = read_1d_u32(&dir.path().join("spike_clusters.npy")).unwrap();
        // Time order is [10(c0), 20(c1), 30(c0), 100(c2), 150(c0), 200(c1)].
        // After merging 0 into 2: [2, 1, 2, 2, 2, 1].
        assert_eq!(written, vec![2, 1, 2, 2, 2, 1]);
    }

    #[test]
    fn save_writes_cluster_group_tsv_excluding_unsorted_rows() {
        let (mut s, dir) = fresh_session();
        s.dispatch(CurationCommand::Relabel { cluster: 0, op: PhyLabelOp::SetGood })
            .unwrap();
        s.dispatch(CurationCommand::Relabel { cluster: 2, op: PhyLabelOp::SetNoise })
            .unwrap();

        save_to_phy(&s, dir.path(), label_str).unwrap();

        let tsv = std::fs::read_to_string(dir.path().join("cluster_group.tsv")).unwrap();
        assert_eq!(
            tsv,
            "cluster_id\tgroup\n0\tgood\n2\tnoise\n",
            "unsorted cluster 1 should be elided"
        );
    }

    #[test]
    fn save_is_atomic_no_tmp_files_left_behind() {
        let (s, dir) = fresh_session();
        save_to_phy(&s, dir.path(), label_str).unwrap();

        // Confirm only the expected files exist (no `.tmp` leftovers).
        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| Some(e.ok()?.file_name().to_string_lossy().into_owned()))
            .collect();
        assert!(names.iter().any(|n| n == "spike_clusters.npy"));
        assert!(names.iter().any(|n| n == "cluster_group.tsv"));
        assert!(
            !names.iter().any(|n| n.ends_with(".tmp")),
            "no tmp files should remain after a successful save (got {names:?})"
        );
    }

    #[test]
    fn save_after_split_writes_new_cluster_assignments() {
        let (mut s, dir) = fresh_session();
        // Split spike at local index 1 of cluster 0 -> new cluster id.
        s.dispatch(CurationCommand::Split {
            cluster: 0,
            spike_idx: vec![1],
            new_cluster: 0, // ignored: auto-allocated
        })
        .unwrap();

        save_to_phy(&s, dir.path(), label_str).unwrap();
        let written = read_1d_u32(&dir.path().join("spike_clusters.npy")).unwrap();

        // Time order: [10(c0), 20(c1), 30(c0), 100(c2), 150(c0), 200(c1)].
        // Cluster 0 had local indices [0, 1, 2] for times [10, 30, 150];
        // splitting local idx 1 means t=30 moves to the new cluster.
        // Initial n_clusters was 3, so new id is 3.
        assert_eq!(written, vec![0, 1, 3, 2, 0, 1]);
    }

    #[test]
    fn save_after_undo_writes_pre_change_state() {
        let (mut s, dir) = fresh_session();
        s.dispatch(CurationCommand::Merge { sources: vec![0, 1], target: 2 })
            .unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();

        save_to_phy(&s, dir.path(), label_str).unwrap();
        let written = read_1d_u32(&dir.path().join("spike_clusters.npy")).unwrap();
        // Original time order: [10(c0), 20(c1), 30(c0), 100(c2), 150(c0), 200(c1)].
        assert_eq!(written, vec![0, 1, 0, 2, 0, 1]);
    }

    #[test]
    fn save_round_trip_after_many_relabels() {
        let (mut s, dir) = fresh_session();
        s.dispatch(CurationCommand::Relabel { cluster: 0, op: PhyLabelOp::SetGood })
            .unwrap();
        s.dispatch(CurationCommand::Relabel { cluster: 1, op: PhyLabelOp::SetMua })
            .unwrap();
        s.dispatch(CurationCommand::Relabel { cluster: 2, op: PhyLabelOp::SetNoise })
            .unwrap();

        save_to_phy(&s, dir.path(), label_str).unwrap();
        let tsv = std::fs::read_to_string(dir.path().join("cluster_group.tsv")).unwrap();
        let lines: Vec<&str> = tsv.lines().collect();
        assert_eq!(lines[0], "cluster_id\tgroup");
        // Three rows, in cluster id order.
        assert_eq!(lines[1], "0\tgood");
        assert_eq!(lines[2], "1\tmua");
        assert_eq!(lines[3], "2\tnoise");
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn save_writes_no_tsv_rows_when_no_curation_happened() {
        let (s, dir) = fresh_session();
        save_to_phy(&s, dir.path(), label_str).unwrap();
        let tsv = std::fs::read_to_string(dir.path().join("cluster_group.tsv")).unwrap();
        // Header only — every cluster is "unsorted" by default.
        assert_eq!(tsv, "cluster_id\tgroup\n");
    }

    #[test]
    fn double_save_is_idempotent() {
        let (mut s, dir) = fresh_session();
        s.dispatch(CurationCommand::Relabel { cluster: 1, op: PhyLabelOp::SetGood })
            .unwrap();
        save_to_phy(&s, dir.path(), label_str).unwrap();
        let first = std::fs::read(dir.path().join("spike_clusters.npy")).unwrap();
        save_to_phy(&s, dir.path(), label_str).unwrap();
        let second = std::fs::read(dir.path().join("spike_clusters.npy")).unwrap();
        assert_eq!(first, second);
    }
}


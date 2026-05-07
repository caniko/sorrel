//! End-to-end integration tests: open a synthetic phy directory, run a
//! full curation sequence, save back to disk, reopen, and verify the
//! reopened state matches the saved state.

use sorrel_data::{save_to_phy, CurationCommand, PhyLabelOp, Session, SqliteJournal};
use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider, PhyLabel};
use sorrel_io::{ClusterId, DataProvider, SampleIndex};
use std::io::Write;
use std::path::Path;

fn write_npy_v1(path: &Path, descr: &str, shape_dim: usize, data: &[u8]) {
    let dict = format!(
        "{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape_dim},), }}"
    );
    let prelude_len = 6 + 2 + 2 + dict.len() + 1;
    let pad = (64 - (prelude_len % 64)) % 64;
    let mut header = dict.into_bytes();
    header.extend(std::iter::repeat(b' ').take(pad));
    header.push(b'\n');
    let header_len = header.len() as u16;

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"\x93NUMPY").unwrap();
    f.write_all(&[1u8, 0u8]).unwrap();
    f.write_all(&header_len.to_le_bytes()).unwrap();
    f.write_all(&header).unwrap();
    f.write_all(data).unwrap();
}

fn label_str(l: &PhyLabel) -> &'static str {
    l.as_str()
}

/// Lay down a phy fixture: 6 spikes, 3 clusters, params.py + a tiny .dat.
///
/// Times are written in ascending order to match Kilosort's convention.
/// `save_to_phy` writes `spike_clusters.npy` in the same on-disk row order
/// as `spike_times.npy`, so the original order *must* equal the time-sorted
/// order for round-trip correctness.
fn write_fixture(root: &Path) {
    // Time-sorted: 10, 30, 50, 100, 150, 200.
    // Original cluster assignment (unmerged): [0, 0, 1, 2, 0, 1]
    let times: [u64; 6] = [10, 30, 50, 100, 150, 200];
    let clusters: [u32; 6] = [0, 0, 1, 2, 0, 1];
    let times_bytes: Vec<u8> = times.iter().flat_map(|t| t.to_le_bytes()).collect();
    let clusters_bytes: Vec<u8> =
        clusters.iter().flat_map(|c| c.to_le_bytes()).collect();
    write_npy_v1(&root.join("spike_times.npy"), "<i64", times.len(), &times_bytes);
    write_npy_v1(
        &root.join("spike_clusters.npy"),
        "<u32",
        clusters.len(),
        &clusters_bytes,
    );

    let dat_bytes = vec![0u8; 4 * 8 * 2]; // 4 ch × 8 samples × i16
    std::fs::write(root.join("recording.dat"), &dat_bytes).unwrap();
    std::fs::write(
        root.join("params.py"),
        "n_channels_dat = 4\nsample_rate = 1000.\ndtype = 'int16'\n",
    )
    .unwrap();
}

#[test]
fn merge_save_reopen_round_trips_cluster_assignments() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);

    {
        let provider = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
        let journal =
            SqliteJournal::open(&root.join("sorrel.sqlite")).unwrap();
        let mut session = Session::new(provider, journal);

        // Merge cluster 0 into 2 and label 1 as 'noise'.
        session
            .dispatch(CurationCommand::Merge { sources: vec![ClusterId(0)], target: ClusterId(2),
            })
            .unwrap();
        session
            .dispatch(CurationCommand::Relabel { cluster: ClusterId(1),
                op: PhyLabelOp::SetNoise,
            })
            .unwrap();
        save_to_phy(&session, root, label_str).unwrap();
    }

    // Reopen — should see the merged buckets and the noise label.
    let provider = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    let labels = provider.initial_labels();
    assert_eq!(labels[1], PhyLabel::Noise);
    assert_eq!(labels[0], PhyLabel::Unsorted);
    assert_eq!(labels[2], PhyLabel::Unsorted);

    // Spike-clusters.npy was rewritten — cluster 2 should now hold the
    // pre-merge cluster 0 spikes too.
    let provider_clusters: Vec<&[sorrel_io::SampleIndex]> = (0..provider.n_clusters()).map(sorrel_io::ClusterId).map(|c| provider.spike_times(c))
        .collect();
    assert!(provider_clusters[0].is_empty(), "cluster 0 should be empty after merge");
    assert_eq!(provider_clusters[1], &[sorrel_io::SampleIndex(50), sorrel_io::SampleIndex(200)]);
    let mut c2 = provider_clusters[2].to_vec();
    c2.sort_unstable();
    assert_eq!(c2, vec![sorrel_io::SampleIndex(10), sorrel_io::SampleIndex(30), sorrel_io::SampleIndex(100), sorrel_io::SampleIndex(150)]);
}

#[test]
fn split_save_reopen_introduces_new_cluster_id() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);

    let new_n_clusters;
    {
        let provider = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
        let journal = SqliteJournal::open(&root.join("sorrel.sqlite")).unwrap();
        let mut session = Session::new(provider, journal);
        // Split spike at local idx 1 of cluster 0 (time = 30) off into a
        // freshly allocated cluster.
        session
            .dispatch(CurationCommand::Split { cluster: ClusterId(0), spike_idx: vec![1], new_cluster: ClusterId(0), // ignored
            })
            .unwrap();
        new_n_clusters = session.n_clusters();
        save_to_phy(&session, root, label_str).unwrap();
    }

    let provider = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    assert_eq!(provider.n_clusters(), new_n_clusters);
    // Cluster 0 lost one spike (t=30).
    assert_eq!(provider.spike_times(ClusterId(0)), &[SampleIndex(10), SampleIndex(150)]);
    // Cluster `new_n_clusters - 1` (the newly allocated id) has the moved spike.
    let new_id = ClusterId(new_n_clusters - 1);
    assert_eq!(provider.spike_times(new_id), &[SampleIndex(30)]);
}

#[test]
fn replay_after_save_yields_identical_state() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);

    let journal_path = root.join("sorrel.sqlite");
    {
        let provider = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
        let journal = SqliteJournal::open(&journal_path).unwrap();
        let mut session = Session::new(provider, journal);

        session
            .dispatch(CurationCommand::Relabel { cluster: ClusterId(0),
                op: PhyLabelOp::SetGood,
            })
            .unwrap();
        session
            .dispatch(CurationCommand::Merge { sources: vec![ClusterId(1)], target: ClusterId(2),
            })
            .unwrap();
        // Save spike_clusters.npy / cluster_group.tsv — but the journal also
        // captures the operations.
        save_to_phy(&session, root, label_str).unwrap();
    }

    // New session, reopened from disk. Replay should reapply the journal
    // on top of the freshly-loaded (already-merged) provider state.
    //
    // To make this test crisp, we delete the spike_clusters.npy that was
    // just written and rely *only* on the journal to reconstruct the
    // post-merge bucket layout.
    {
        // Restore the original spike_clusters.npy (clusters [0,0,1,2,0,1]
        // — matches the time-sorted fixture order).
        let clusters: [u32; 6] = [0, 0, 1, 2, 0, 1];
        let bytes: Vec<u8> = clusters.iter().flat_map(|c| c.to_le_bytes()).collect();
        write_npy_v1(
            &root.join("spike_clusters.npy"),
            "<u32",
            clusters.len(),
            &bytes,
        );
    }
    let provider = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    let journal = SqliteJournal::open(&journal_path).unwrap();
    let mut session = Session::new(provider, journal);
    session.replay_journal().unwrap();

    // Cluster 0 was relabeled good.
    assert_eq!(session.label(ClusterId(0)), Some(PhyLabel::Good));
    // Cluster 1 was merged into 2 — so cluster 1 is empty, cluster 2 has both.
    assert!(session.spike_times(ClusterId(1)).is_empty());
    assert_eq!(session.spike_times(ClusterId(2)), &[SampleIndex(50), SampleIndex(100), SampleIndex(200)]);
}

#[test]
fn undo_then_save_emits_pre_merge_state() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);

    let journal_path = root.join("sorrel.sqlite");
    {
        let provider = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
        let journal = SqliteJournal::open(&journal_path).unwrap();
        let mut session = Session::new(provider, journal);

        session
            .dispatch(CurationCommand::Merge { sources: vec![ClusterId(0), ClusterId(1)], target: ClusterId(2) })
            .unwrap();
        session.dispatch(CurationCommand::Undo).unwrap();
        save_to_phy(&session, root, label_str).unwrap();
    }

    let provider = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    // Pre-merge counts: 0 has 3, 1 has 2, 2 has 1.
    assert_eq!(provider.spike_times(ClusterId(0)).len(), 3);
    assert_eq!(provider.spike_times(ClusterId(1)).len(), 2);
    assert_eq!(provider.spike_times(ClusterId(2)).len(), 1);
}

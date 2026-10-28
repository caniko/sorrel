//! End-to-end integration tests: open a synthetic phy directory, run a
//! full curation sequence, save back to disk, reopen, and verify the
//! reopened state matches the saved state.

use sorrel_data::{
    open_phy, save_and_seal, save_to_phy, seed_phy, CurationCommand, Journal, JournalOpts,
    PhyLabelOp, Session,
};
use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider, PhyLabel};
use sorrel_io::{ClusterId, DataProvider, SampleIndex};
use std::io::Write;
use std::path::Path;

fn write_npy_v1(path: &Path, descr: &str, shape_dim: usize, data: &[u8]) {
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape_dim},), }}");
    let prelude_len = 6 + 2 + 2 + dict.len() + 1;
    let pad = (64 - (prelude_len % 64)) % 64;
    let mut header = dict.into_bytes();
    header.resize(header.len() + pad, b' ');
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
/// Times are written in ascending order to match Kilosort's convention;
/// `save_to_phy` keeps `spike_times.npy` row order either way.
fn write_fixture(root: &Path) {
    // Time-sorted: 10, 30, 50, 100, 150, 200.
    // Original cluster assignment (unmerged): [0, 0, 1, 2, 0, 1]
    let times: [u64; 6] = [10, 30, 50, 100, 150, 200];
    let clusters: [u32; 6] = [0, 0, 1, 2, 0, 1];
    let times_bytes: Vec<u8> = times.iter().flat_map(|t| t.to_le_bytes()).collect();
    let clusters_bytes: Vec<u8> = clusters.iter().flat_map(|c| c.to_le_bytes()).collect();
    write_npy_v1(
        &root.join("spike_times.npy"),
        "<i64",
        times.len(),
        &times_bytes,
    );
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
        let journal = Journal::open(&root.join("sorrel.sqlite")).unwrap();
        let mut session = Session::new(provider, journal);

        // Merge cluster 0 into 2 and label 1 as 'noise'.
        session
            .dispatch(CurationCommand::Merge {
                sources: vec![ClusterId(0)],
                target: ClusterId(2),
            })
            .unwrap();
        session
            .dispatch(CurationCommand::Relabel {
                cluster: ClusterId(1),
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
    let provider_clusters: Vec<&[sorrel_io::SampleIndex]> = (0..provider.n_clusters())
        .map(sorrel_io::ClusterId)
        .map(|c| provider.spike_times(c))
        .collect();
    assert!(
        provider_clusters[0].is_empty(),
        "cluster 0 should be empty after merge"
    );
    assert_eq!(
        provider_clusters[1],
        &[sorrel_io::SampleIndex(50), sorrel_io::SampleIndex(200)]
    );
    let mut c2 = provider_clusters[2].to_vec();
    c2.sort_unstable();
    assert_eq!(
        c2,
        vec![
            sorrel_io::SampleIndex(10),
            sorrel_io::SampleIndex(30),
            sorrel_io::SampleIndex(100),
            sorrel_io::SampleIndex(150)
        ]
    );
}

#[test]
fn split_save_reopen_introduces_new_cluster_id() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);

    let new_n_clusters;
    {
        let provider = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
        let journal = Journal::open(&root.join("sorrel.sqlite")).unwrap();
        let mut session = Session::new(provider, journal);
        // Split spike at local idx 1 of cluster 0 (time = 30) off into a
        // freshly allocated cluster.
        session
            .dispatch(CurationCommand::Split {
                cluster: ClusterId(0),
                spike_idx: vec![1],
                new_cluster: ClusterId(0), // ignored
                by: sorrel_data::command::Method::Manual,
            })
            .unwrap();
        new_n_clusters = session.n_clusters();
        save_to_phy(&session, root, label_str).unwrap();
    }

    let provider = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    assert_eq!(provider.n_clusters(), new_n_clusters);
    // Cluster 0 lost one spike (t=30).
    assert_eq!(
        provider.spike_times(ClusterId(0)),
        &[SampleIndex(10), SampleIndex(150)]
    );
    // Cluster `new_n_clusters - 1` (the newly allocated id) has the moved spike.
    let new_id = ClusterId(new_n_clusters - 1);
    assert_eq!(provider.spike_times(new_id), &[SampleIndex(30)]);
}

/// Open and seed the directory the way the binary does.
fn open(root: &Path) -> anyhow::Result<Session<KilosortProvider>> {
    let mut s = open_phy(root, KilosortOpenParams::default(), &JournalOpts::default())?;
    seed_phy(&mut s)?;
    Ok(s)
}

/// The production save path keeps the action tree: reopening replays the
/// journal onto the root snapshot, so the saved state comes back exactly
/// once (a split is not applied twice) and the history is still there.
#[test]
fn reopen_after_save_yields_identical_state_and_keeps_history() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    {
        let mut session = open(root).unwrap();
        session
            .dispatch(CurationCommand::Relabel {
                cluster: ClusterId(0),
                op: PhyLabelOp::SetGood,
            })
            .unwrap();
        session
            .dispatch(CurationCommand::Merge {
                sources: vec![ClusterId(1)],
                target: ClusterId(2),
            })
            .unwrap();
        // Move the spike at t=30 (local index 1 of cluster 0) into a new cluster.
        session
            .dispatch(CurationCommand::Split {
                cluster: ClusterId(0),
                spike_idx: vec![1],
                new_cluster: ClusterId(u32::MAX),
                by: sorrel_data::command::Method::Manual,
            })
            .unwrap();
        save_and_seal(&mut session, root, label_str).unwrap();
        assert!(session.is_saved());
        assert_eq!((session.history_len(), session.redo_len()), (3, 0));
    }

    let mut session = open(root).unwrap();
    assert_eq!(session.history_len(), 3, "the tree survives the save");
    assert!(session.is_saved(), "the files on disk are the head's");
    assert_eq!(
        session.n_clusters(),
        4,
        "one split allocated exactly one cluster"
    );
    assert_eq!(session.label(ClusterId(0)), Some(PhyLabel::Good));
    assert_eq!(
        session.spike_times(ClusterId(0)),
        &[SampleIndex(10), SampleIndex(150)]
    );
    assert_eq!(session.spike_times(ClusterId(3)), &[SampleIndex(30)]);
    assert!(session.spike_times(ClusterId(1)).is_empty());
    assert_eq!(
        session.spike_times(ClusterId(2)),
        &[SampleIndex(50), SampleIndex(100), SampleIndex(200)]
    );
    // Undo past the save reaches the sort as it was.
    for _ in 0..3 {
        session.dispatch(CurationCommand::Undo).unwrap();
    }
    assert!(!session.is_saved());
    assert_eq!(session.n_clusters(), 3);
    assert_eq!(
        session.spike_times(ClusterId(1)),
        &[SampleIndex(50), SampleIndex(200)]
    );
}

/// A saved sorter pin is the default; `set_sorter` (`--sorter`) overrides it.
#[test]
fn set_sorter_overrides_saved_pin() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    {
        let mut session = open(root).unwrap();
        session.set_sorter("kilosort4");
        session
            .dispatch(CurationCommand::Merge {
                sources: vec![ClusterId(1)],
                target: ClusterId(2),
            })
            .unwrap();
        save_and_seal(&mut session, root, label_str).unwrap();
    }

    let provider = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    let journal = Journal::open(&root.join("other.journal")).unwrap();
    let mut session = Session::new(provider, journal);
    assert_eq!(session.sorter(), "kilosort4", "saved pin is the default");
    session.fill_detected_sorter(Some("detected".into()));
    assert_eq!(
        session.sorter(),
        "kilosort4",
        "detection never overrides a pin"
    );
    session.set_sorter("x");
    assert_eq!(session.sorter(), "x");
}

/// Origins, parents, notes and the tuned/split markers survive a save and
/// reopen (replayed from the journal), and land in the annotations sidecar
/// for phy and other readers.
#[test]
fn origin_parents_and_notes_survive_save_and_reopen() {
    use sorrel_data::command::{Method, UnitOrigin};
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    let note = "two\tlines\nand a \\ slash";
    {
        let mut session = open(root).unwrap();
        session.set_sorter("kilosort4");
        let cmds = [
            CurationCommand::Merge {
                sources: vec![ClusterId(1)],
                target: ClusterId(2),
            },
            CurationCommand::Split {
                cluster: ClusterId(0),
                spike_idx: vec![1],
                new_cluster: ClusterId(u32::MAX),
                by: Method::Llm("claude-code".into()),
            },
            CurationCommand::Note {
                cluster: ClusterId(3),
                text: note.into(),
            },
        ];
        for c in cmds {
            session.dispatch(c).unwrap();
        }
        save_and_seal(&mut session, root, label_str).unwrap();
    }

    let session = open(root).unwrap();
    assert_eq!(session.history_len(), 3);
    assert_eq!(session.sorter(), "kilosort4", "Auto row pins the sorter");
    assert_eq!(session.origin_label(ClusterId(0)), "Auto[kilosort4]");
    assert_eq!(session.tags(ClusterId(0)), ["split"]);
    assert_eq!(session.tags(ClusterId(2)), ["tuned"]);
    assert_eq!(session.parents(ClusterId(2)), [ClusterId(1)]);
    let child = ClusterId(3);
    assert_eq!(
        session.origin(child),
        &UnitOrigin::Sorrel(Method::Llm("claude-code".into()))
    );
    assert_eq!(session.origin_label(child), "Sorrel[LLM:claude-code]");
    assert_eq!(session.parents(child), [ClusterId(0)]);
    assert_eq!(session.note(child), note);
    let tsv = std::fs::read_to_string(root.join(sorrel_io::kilosort::ANNOTATIONS_TSV)).unwrap();
    assert!(tsv.starts_with("cluster_id\tsorrel_origin\tsorrel_parents\tsorrel_note\n"));
    assert!(!tsv.contains("\n1\t"), "empty unit 1 gets no row");
}

/// A plain `save_to_phy` into the directory (no seal, no root snapshot)
/// rewrites the root itself, so the next open fails closed.
#[test]
fn open_refuses_journal_after_unsealed_save() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    {
        let mut session = open(root).unwrap();
        session
            .dispatch(CurationCommand::Merge {
                sources: vec![ClusterId(1)],
                target: ClusterId(2),
            })
            .unwrap();
        save_to_phy(&session, root, label_str).unwrap();
    }
    let err = open(root).err().expect("refused");
    assert!(
        format!("{err:#}").contains("changed since this journal was sealed"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn undo_then_save_emits_pre_merge_state() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);

    let journal_path = root.join("sorrel.sqlite");
    {
        let provider = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
        let journal = Journal::open(&journal_path).unwrap();
        let mut session = Session::new(provider, journal);

        session
            .dispatch(CurationCommand::Merge {
                sources: vec![ClusterId(0), ClusterId(1)],
                target: ClusterId(2),
            })
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

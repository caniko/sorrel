//! Tests honouring phy patterns that didn't have a 1:1 Rust analogue earlier:
//!
//! - **GlobalHistory analogue** → cross-session journal resume. phy's
//!   `test_history.py:test_global_history` exercises a multi-stack history
//!   that survives subsystem reload. Sorrel has a single per-session
//!   journal — the equivalent invariant is that closing and reopening the
//!   journal preserves the operation log byte-for-byte.
//!
//! - **Descendants graph** → lineage walk. phy's
//!   `test_clustering.py:test_clustering_descendants_*` track `(old_id →
//!   new_id)` pairs per merge/split. Sorrel records the same information
//!   inside `MergeRecord`/`SplitRecord`; we expose it via the journal log
//!   so a "what-touched-this-cluster" walk is possible after the fact.
//!
//! - **chunk_bounds analogue** → batched waveform extraction. phy's
//!   `test_traces.py:test_get_chunk_bounds` is about streaming-reader
//!   internals; sorrel's analogue is `extract_snippets_single_channel`
//!   correctly handling its full input range without internal chunking
//!   artefacts.
//!
//! - **Save-and-reseal** → the new commit-on-save flow that fixes the
//!   journal-vs-artifacts coupling we identified.

use sorrel_data::session::ApplyPhyLabel;
use sorrel_data::{
    save_and_reseal_journal, save_to_phy, ClusterIndex, CurationCommand, Journal, PhyLabelOp,
    Session,
};
use sorrel_io::{ClusterId, DataProvider, SampleIndex, TraceSamples, TraceSlice};

// ---- Stub provider ------------------------------------------------------

struct Stub {
    spikes: Vec<Vec<SampleIndex>>,
}

impl DataProvider for Stub {
    type Label = u8;
    fn sample_rate(&self) -> f32 {
        1000.0
    }
    fn n_channels(&self) -> u32 {
        1
    }
    fn n_samples(&self) -> SampleIndex {
        SampleIndex(0)
    }
    fn n_clusters(&self) -> u32 {
        self.spikes.len() as u32
    }
    fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
        self.spikes
            .get(c.idx())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
    fn trace(&self, _: SampleIndex, _: u32) -> TraceSlice<'_> {
        TraceSlice {
            start: SampleIndex(0),
            n_channels: 0,
            samples: TraceSamples::I16(&[]),
        }
    }
    fn initial_labels(&self) -> Vec<u8> {
        vec![0u8; self.spikes.len()]
    }
}

impl ApplyPhyLabel for Stub {
    fn label_from_op(op: PhyLabelOp) -> Self::Label {
        match op {
            PhyLabelOp::SetUnsorted => 0,
            PhyLabelOp::SetGood => 1,
            PhyLabelOp::SetMua => 2,
            PhyLabelOp::SetNoise => 3,
        }
    }
}

fn fixture_provider() -> Stub {
    Stub {
        spikes: vec![vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)], vec![SampleIndex(20), SampleIndex(200)], vec![SampleIndex(100)]],
    }
}

// ---- Cross-session journal resume (GlobalHistory analogue) ---------------

/// phy's `test_history.py:test_global_history` invariant: history survives
/// subsystem reload. Sorrel's analogue: closing a session, reopening the
/// journal, replaying produces identical state.
#[test]
fn journal_round_trips_across_session_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let journal_path = dir.path().join("sorrel.journal");

    // Session 1: open, mutate, close.
    let pre_assignment;
    let merged_assignment;
    {
        let provider = fixture_provider();
        let journal = Journal::open_or_create(&journal_path, 0xFEED, 3).unwrap();
        let mut session = Session::new(provider, journal);

        pre_assignment = session.cluster_index().spike_clusters().to_vec();

        session
            .dispatch(CurationCommand::Merge { sources: vec![ClusterId(0)], target: ClusterId(2) })
            .unwrap();
        session
            .dispatch(CurationCommand::Relabel { cluster: ClusterId(1), op: PhyLabelOp::SetMua })
            .unwrap();

        merged_assignment = session.cluster_index().spike_clusters().to_vec();
        assert_ne!(pre_assignment, merged_assignment);
    }

    // Session 2: reopen with the same baseline, replay.
    let provider = fixture_provider();
    let journal = Journal::open_or_create(&journal_path, 0xFEED, 3).unwrap();
    let mut session = Session::new(provider, journal);
    session.replay_journal().unwrap();

    assert_eq!(
        session.cluster_index().spike_clusters(),
        &merged_assignment[..],
        "cross-session replay didn't reach the same state"
    );
    assert_eq!(session.label(ClusterId(1)), Some(2u8)); // Mua
}

/// Same invariant under a longer history: 10 alternating merge/relabel
/// operations should replay byte-identically.
#[test]
fn journal_round_trips_a_long_history() {
    let dir = tempfile::tempdir().unwrap();
    let journal_path = dir.path().join("sorrel.journal");

    let original: Vec<ClusterId>;
    let final_state: Vec<ClusterId>;
    {
        let provider = Stub {
            spikes: (0..10)
                .map(|c| (0..20).map(|i| SampleIndex((c * 100 + i) as u64)).collect())
                .collect(),
        };
        let journal = Journal::open_or_create(&journal_path, 0xCAFE, 10).unwrap();
        let mut session = Session::new(provider, journal);
        original = session.cluster_index().spike_clusters().to_vec();

        for c in (0..9u32).map(ClusterId) {
            session
                .dispatch(CurationCommand::Relabel { cluster: c, op: PhyLabelOp::SetGood })
                .unwrap();
            session
                .dispatch(CurationCommand::Merge { sources: vec![c], target: ClusterId(c.0 + 1) })
                .unwrap();
        }
        final_state = session.cluster_index().spike_clusters().to_vec();
        assert_ne!(original, final_state);
    }

    let provider = Stub {
        spikes: (0..10)
            .map(|c| (0..20).map(|i| SampleIndex((c * 100 + i) as u64)).collect())
            .collect(),
    };
    let journal = Journal::open_or_create(&journal_path, 0xCAFE, 10).unwrap();
    let mut session = Session::new(provider, journal);
    session.replay_journal().unwrap();

    assert_eq!(session.cluster_index().spike_clusters(), &final_state[..]);
}

// ---- Lineage / descendants walk -----------------------------------------

/// phy's `test_clustering_descendants_merge` checks that a merge records
/// `(source_id → target_id)` for every source. Sorrel surfaces the same
/// info via `history_commands()` — walking the journal yields a
/// concrete merge-/split-derived lineage.
#[test]
fn journal_history_exposes_merge_descendants_pairs() {
    let dir = tempfile::tempdir().unwrap();
    let journal_path = dir.path().join("sorrel.journal");
    let provider = fixture_provider();
    let journal = Journal::open_or_create(&journal_path, 0, 3).unwrap();
    let mut session = Session::new(provider, journal);

    session
        .dispatch(CurationCommand::Merge { sources: vec![ClusterId(0), ClusterId(1)], target: ClusterId(2) })
        .unwrap();

    // Walk history, derive descendants pairs the same way phy's
    // `test_clustering_descendants_merge` asserts.
    let mut descendants: Vec<(ClusterId, ClusterId)> = Vec::new();
    for cmd in session.history_commands() {
        if let CurationCommand::Merge { sources, target } = cmd {
            for &s in sources {
                descendants.push((s, *target));
            }
        }
    }
    assert_eq!(descendants, vec![(ClusterId(0), ClusterId(2)), (ClusterId(1), ClusterId(2))]);
}

/// Multi-step lineage walk: trace cluster 2's history through
/// successive merges. phy's descendants graph treats this as a chain.
#[test]
fn journal_history_supports_lineage_chain_walk() {
    let dir = tempfile::tempdir().unwrap();
    let journal_path = dir.path().join("sorrel.journal");
    let provider = Stub {
        spikes: (0..5).map(|c| vec![SampleIndex(c as u64 * 100)]).collect(),
    };
    let journal = Journal::open_or_create(&journal_path, 0, 5).unwrap();
    let mut session = Session::new(provider, journal);

    // 0 → 1 → 2 → 3: chained merges.
    session
        .dispatch(CurationCommand::Merge { sources: vec![ClusterId(0)], target: ClusterId(1) })
        .unwrap();
    session
        .dispatch(CurationCommand::Merge { sources: vec![ClusterId(1)], target: ClusterId(2) })
        .unwrap();
    session
        .dispatch(CurationCommand::Merge { sources: vec![ClusterId(2)], target: ClusterId(3) })
        .unwrap();

    // Trace what 0 became by walking forward through merges.
    let mut current = ClusterId(0);
    for cmd in session.history_commands() {
        if let CurationCommand::Merge { sources, target } = cmd {
            if sources.contains(&current) {
                current = *target;
            }
        }
    }
    assert_eq!(current, ClusterId(3), "cluster 0 was eventually merged into 3");
}

/// Split lineage: a split records `(source → new_cluster)`. The new id
/// can be read back from the post-split bucket count.
#[test]
fn journal_history_records_split_descendants() {
    let dir = tempfile::tempdir().unwrap();
    let journal_path = dir.path().join("sorrel.journal");
    let provider = fixture_provider();
    let journal = Journal::open_or_create(&journal_path, 0, 3).unwrap();
    let mut session = Session::new(provider, journal);

    let n_pre = session.n_clusters();
    session
        .dispatch(CurationCommand::Split { cluster: ClusterId(0), spike_idx: vec![1], new_cluster: ClusterId(0), // ignored — auto-allocated
        })
        .unwrap();
    let n_post = session.n_clusters();
    assert_eq!(n_post, n_pre + 1);

    let split_count = session
        .history_commands()
        .filter(|c| matches!(c, CurationCommand::Split { .. }))
        .count();
    assert_eq!(split_count, 1);
}

// ---- chunk_bounds analogue ----------------------------------------------

/// phy's `test_get_chunk_bounds` is about streaming-reader chunking.
/// Sorrel's closest functional analogue is `extract_snippets_single_channel`
/// processing a long input without missing any in-range spike. This is
/// the property that would catch a bug in any chunked waveform extractor
/// we layer in later.
#[test]
fn waveform_extractor_processes_every_in_range_spike() {
    use sorrel_compute::extract_snippets_single_channel;
    let trace: Vec<f32> = (0..10_000).map(|t| t as f32).collect();
    let spikes: Vec<SampleIndex> = (50..9950).step_by(100).map(|t| SampleIndex(t as u64)).collect();
    let pre = 10u32;
    let post = 10u32;
    let snips = extract_snippets_single_channel(&trace, SampleIndex(0), &spikes, pre, post);
    // Every spike falls inside [pre, len - post - 1] = [10, 9989], so all
    // should produce a snippet.
    assert_eq!(snips.len(), spikes.len());
    for (i, snip) in snips.iter().enumerate() {
        assert_eq!(snip.len(), (pre + post + 1) as usize);
        // Centre sample should equal trace[spike_time].
        assert_eq!(snip[pre as usize], spikes[i].as_f32());
    }
}

/// chunk-bounds correctness under non-uniform spacing: spikes at
/// arbitrary positions, including some near boundaries.
#[test]
fn waveform_extractor_handles_non_uniform_spacing() {
    use sorrel_compute::extract_snippets_single_channel;
    let trace: Vec<f32> = (0..1000).map(|t| t as f32).collect();
    let spikes: [SampleIndex; 7] = [12, 47, 63, 250, 251, 252, 999].map(SampleIndex);
    let pre = 10u32;
    let post = 10u32;
    let snips = extract_snippets_single_channel(&trace, SampleIndex(0), &spikes, pre, post);
    // 999 is dropped (window would extend past trace end).
    assert_eq!(snips.len(), spikes.len() - 1);
    for snip in &snips {
        assert_eq!(snip.len(), (pre + post + 1) as usize);
    }
}

// ---- Save-and-reseal flow (the architectural fix) ------------------------

/// Saving and resealing produces a fresh journal with a new baseline that
/// matches the on-disk artifact. Reopening with the new baseline
/// succeeds; the journal has no records.
#[test]
fn save_and_reseal_writes_artifacts_and_clears_journal_history() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let journal_path = root.join("sorrel.journal");

    let new_journal_baseline;
    {
        let provider = fixture_provider();
        let journal = Journal::open_or_create(&journal_path, 0, 3).unwrap();
        let mut session = Session::new(provider, journal);
        session
            .dispatch(CurationCommand::Merge { sources: vec![ClusterId(0)], target: ClusterId(2) })
            .unwrap();
        session
            .dispatch(CurationCommand::Relabel { cluster: ClusterId(1), op: PhyLabelOp::SetMua })
            .unwrap();

        let new_journal =
            save_and_reseal_journal(&session, &root, &journal_path, label_str_u8).unwrap();
        new_journal_baseline = new_journal.baseline();
    }

    // Reopen with the same baseline.
    let journal = Journal::open_or_create(&journal_path, new_journal_baseline, 3).unwrap();
    assert!(journal.replay().unwrap().is_empty(), "reseal cleared history");
}

/// Once the journal is resealed, attempting to reopen with the OLD
/// baseline fails — exactly the divergence-prevention property we wanted.
#[test]
fn save_and_reseal_invalidates_old_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let journal_path = root.join("sorrel.journal");
    let old_baseline = 0xDEAD_BEEFu64;

    {
        let provider = fixture_provider();
        let journal = Journal::open_or_create(&journal_path, old_baseline, 3).unwrap();
        let mut session = Session::new(provider, journal);
        session
            .dispatch(CurationCommand::Merge { sources: vec![ClusterId(0)], target: ClusterId(2) })
            .unwrap();

        save_and_reseal_journal(&session, &root, &journal_path, label_str_u8).unwrap();
    }

    // Reopen with the *old* baseline should fail.
    let res = Journal::open_or_create(&journal_path, old_baseline, 3);
    assert!(
        res.is_err(),
        "old baseline must not be accepted after reseal",
    );
}

/// External edit of `spike_clusters.npy` between sessions: reopening with
/// the journal sealed against the original baseline fails loudly. This is
/// the journal-vs-artifacts safety net.
#[test]
fn external_edit_to_spike_clusters_invalidates_journal() {
    use sorrel_data::baseline_hash;

    let dir = tempfile::tempdir().unwrap();
    let journal_path = dir.path().join("sorrel.journal");
    let dummy_path = dir.path().join("spike_clusters.npy");

    // Initial baseline = hash of fake "[0, 0, 1, 2, 0, 1]".
    let initial_bytes: Vec<u8> = vec![
        0u32, 0, 1, 2, 0, 1,
    ]
    .iter()
    .flat_map(|v: &u32| v.to_le_bytes())
    .collect();
    std::fs::write(&dummy_path, &initial_bytes).unwrap();
    let initial = baseline_hash(&initial_bytes);

    {
        let provider = fixture_provider();
        let journal = Journal::open_or_create(&journal_path, initial, 3).unwrap();
        let mut session = Session::new(provider, journal);
        session
            .dispatch(CurationCommand::Relabel { cluster: ClusterId(0), op: PhyLabelOp::SetGood })
            .unwrap();
    }

    // External edit: rewrite spike_clusters.npy bytes.
    let altered_bytes: Vec<u8> = vec![1u32, 1, 1, 2, 1, 1]
        .iter()
        .flat_map(|v: &u32| v.to_le_bytes())
        .collect();
    std::fs::write(&dummy_path, &altered_bytes).unwrap();
    let new_baseline = baseline_hash(&altered_bytes);
    assert_ne!(initial, new_baseline);

    // Reopening with the new baseline must refuse — journal is stale.
    let res = Journal::open_or_create(&journal_path, new_baseline, 3);
    assert!(res.is_err());
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("changed since this journal was sealed"),
        "expected divergence error, got: {msg}",
    );
}

/// Saving without reseal (legacy `save_to_phy`) leaves the journal alone.
/// The user's undo history survives. This is what the binary uses today.
#[test]
fn save_to_phy_alone_does_not_touch_journal() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let journal_path = root.join("sorrel.journal");

    let provider = fixture_provider();
    let journal = Journal::open_or_create(&journal_path, 0, 3).unwrap();
    let mut session = Session::new(provider, journal);
    session
        .dispatch(CurationCommand::Merge { sources: vec![ClusterId(0)], target: ClusterId(2) })
        .unwrap();
    save_to_phy(&session, &root, label_str_u8).unwrap();

    // Journal still has the merge record (drop session and reopen).
    drop(session);
    let journal = Journal::open_or_create(&journal_path, 0, 3).unwrap();
    let ops = journal.replay().unwrap();
    assert_eq!(ops.len(), 1);
    assert!(matches!(ops[0], CurationCommand::Merge { .. }));
}

// ---- ClusterIndex spike-count invariant under heavy churn ---------------

/// Long sequence of mixed merge/split — n_spikes should never change.
/// This is the analogue of phy's clustering total-count invariant under
/// many operations.
#[test]
fn cluster_index_n_spikes_invariant_under_long_session() {
    let dir = tempfile::tempdir().unwrap();
    let journal_path = dir.path().join("sorrel.journal");
    let provider = Stub {
        spikes: (0..5)
            .map(|c| (0..10).map(|i| SampleIndex((c * 100 + i) as u64)).collect())
            .collect(),
    };
    let total_spikes = 5 * 10;
    let journal = Journal::open_or_create(&journal_path, 0, 5).unwrap();
    let mut session = Session::new(provider, journal);
    assert_eq!(session.cluster_index().n_spikes(), total_spikes);

    for round in 0..3 {
        for c in (0..4u32).map(ClusterId) {
            session
                .dispatch(CurationCommand::Merge { sources: vec![c], target: ClusterId(c.0 + 1) })
                .unwrap();
        }
        let n = session.cluster_index().n_spikes();
        assert_eq!(n, total_spikes, "n_spikes drift on round {round}: {n}");

        for _ in 0..2 {
            session.dispatch(CurationCommand::Undo).unwrap();
        }
        let n = session.cluster_index().n_spikes();
        assert_eq!(n, total_spikes, "n_spikes drift after undo on round {round}: {n}");
    }
}

fn label_str_u8(l: &u8) -> &'static str {
    match *l {
        0 => "unsorted",
        1 => "good",
        2 => "mua",
        3 => "noise",
        _ => "unsorted",
    }
}

#[allow(dead_code)]
fn _unused_marker(_: &ClusterIndex) {}

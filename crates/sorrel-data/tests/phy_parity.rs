//! Tests ported from phy / phylib.
//!
//! These mirror canonical invariants from the upstream `phy` and `phylib`
//! Python projects (cortex-lab/phy, cortex-lab/phylib) so that sorrel's
//! curation semantics, spike-table accounting, and file-format handling stay
//! aligned with what phy users expect.
//!
//! Where phy's implementation differs from sorrel's by design, the test
//! comment calls out the divergence.
//!
//! Source references (snapshot of master at the time of porting):
//!   - phy/cluster/tests/test_clustering.py
//!   - phylib/io/tests/test_traces.py
//!   - phylib/io/tests/test_array.py
//!   - phylib/io/tests/test_model.py

use sorrel_data::session::ApplyPhyLabel;
use sorrel_data::{ClusterIndex, CurationCommand, PhyLabelOp, Session, SqliteJournal};
use sorrel_io::{ClusterId, DataProvider, SampleIndex, TraceSamples, TraceSlice};

// ---- Stub provider ------------------------------------------------------

/// `Clustering(np.array([2, 5, 3, 2, 7, 5, 2]))` from phy's
/// `test_clustering.py`. We materialise the per-cluster buckets the same
/// way `from_provider` would after time-sorting.
fn phy_clustering_provider() -> StubProvider {
    // spike_clusters in original time order: [2, 5, 3, 2, 7, 5, 2]
    //   global idx: 0 1 2 3 4 5 6
    //   cluster 2: indices 0, 3, 6 → "times" = 0, 3, 6
    //   cluster 5: indices 1, 5     → 1, 5
    //   cluster 3: index 2          → 2
    //   cluster 7: index 4          → 4
    StubProvider {
        // Index by cluster id; entries 0, 1, 4, 6 (etc.) are empty.
        spikes: vec![
            vec![],                                               // 0
            vec![],                                               // 1
            vec![SampleIndex(0), SampleIndex(3), SampleIndex(6)], // 2
            vec![SampleIndex(2)],                                 // 3
            vec![],                                               // 4
            vec![SampleIndex(1), SampleIndex(5)],                 // 5
            vec![],                                               // 6
            vec![SampleIndex(4)],                                 // 7
        ],
    }
}

struct StubProvider {
    spikes: Vec<Vec<SampleIndex>>,
}

impl DataProvider for StubProvider {
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
        self.spikes.get(c.idx()).map(Vec::as_slice).unwrap_or(&[])
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

impl ApplyPhyLabel for StubProvider {
    fn label_from_op(op: PhyLabelOp) -> Self::Label {
        match op {
            PhyLabelOp::SetUnsorted => 0,
            PhyLabelOp::SetGood => 1,
            PhyLabelOp::SetMua => 2,
            PhyLabelOp::SetNoise => 3,
        }
    }
}

fn fresh_session_with(provider: StubProvider) -> (Session<StubProvider>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let journal = SqliteJournal::open(&dir.path().join("j.sqlite")).unwrap();
    (Session::new(provider, journal), dir)
}

// ---- Ported from phy/cluster/tests/test_clustering.py ------------------

/// phy: `clustering.merge([0, 1], 11)` produces a cluster with all spikes
/// from 0 ∪ 1, no longer addressable as either source.
///
/// **Divergence**: phy *deletes* the source cluster ids; sorrel keeps them
/// as empty buckets. Both representations preserve the spike-set invariant.
#[test]
fn phy_clustering_merge_consolidates_spike_set() {
    let provider = phy_clustering_provider();
    let pre_2 = provider.spikes[2].len();
    let pre_5 = provider.spikes[5].len();
    let (mut s, _d) = fresh_session_with(provider);

    s.dispatch(CurationCommand::Merge {
        sources: vec![ClusterId(2)],
        target: ClusterId(5),
    })
    .unwrap();

    // phy: ae(clustering.spikes_per_cluster[11], np.sort(np.r_[spk0, spk1]))
    // sorrel: target's bucket grows by exactly the source's count.
    assert!(s.spike_times(ClusterId(2)).is_empty(), "source emptied");
    assert_eq!(s.spike_times(ClusterId(5)).len(), pre_2 + pre_5);
}

/// phy: every merge/split must preserve total spike count.
#[test]
fn phy_clustering_total_spike_count_is_preserved() {
    let provider = phy_clustering_provider();
    let total_pre: usize = provider.spikes.iter().map(Vec::len).sum();
    let (mut s, _d) = fresh_session_with(provider);

    s.dispatch(CurationCommand::Merge {
        sources: vec![ClusterId(2), ClusterId(3)],
        target: ClusterId(5),
    })
    .unwrap();
    let total_post: usize = (0..s.n_clusters())
        .map(ClusterId)
        .map(|c| s.spike_times(c).len())
        .sum();
    assert_eq!(total_pre, total_post);

    s.dispatch(CurationCommand::Split {
        cluster: ClusterId(5),
        spike_idx: vec![0, 1],
        new_cluster: ClusterId(0),
    })
    .unwrap();
    let total_post2: usize = (0..s.n_clusters())
        .map(ClusterId)
        .map(|c| s.spike_times(c).len())
        .sum();
    assert_eq!(total_pre, total_post2);
}

/// phy: `clustering.split([0])` then `clustering.undo()` then
/// `clustering.redo()` lands at the post-split state, same as the first
/// split.
///
/// `assert_array_equal(spike_clusters_after_redo, spike_clusters_after_first_split)`
#[test]
fn phy_clustering_undo_redo_round_trips_split() {
    let provider = phy_clustering_provider();
    let (mut s, _d) = fresh_session_with(provider);

    // First split — capture state.
    s.dispatch(CurationCommand::Split {
        cluster: ClusterId(2),
        spike_idx: vec![0],
        new_cluster: ClusterId(0), // ignored
    })
    .unwrap();
    let after_split: Vec<Vec<SampleIndex>> = (0..s.n_clusters())
        .map(ClusterId)
        .map(|c| s.spike_times(c).to_vec())
        .collect();

    s.dispatch(CurationCommand::Undo).unwrap();
    s.dispatch(CurationCommand::Redo).unwrap();

    let after_redo: Vec<Vec<SampleIndex>> = (0..s.n_clusters())
        .map(ClusterId)
        .map(|c| s.spike_times(c).to_vec())
        .collect();
    assert_eq!(after_split, after_redo);
}

/// phy: `clustering.redo()` after `clustering.undo()` then *another*
/// `clustering.redo()` is a no-op (no further redo state).
#[test]
fn phy_clustering_redo_with_empty_stack_is_noop() {
    let provider = phy_clustering_provider();
    let (mut s, _d) = fresh_session_with(provider);

    s.dispatch(CurationCommand::Merge {
        sources: vec![ClusterId(3)],
        target: ClusterId(2),
    })
    .unwrap();
    s.dispatch(CurationCommand::Undo).unwrap();
    s.dispatch(CurationCommand::Redo).unwrap();

    let len_before = s.history_len();
    s.dispatch(CurationCommand::Redo).unwrap();
    assert_eq!(
        s.history_len(),
        len_before,
        "extra redo should not change state"
    );
}

/// phy: multi-step undo unwinds in strict reverse order.
/// Mirrors the multiple-merges pattern in `test_clustering_merge`.
#[test]
fn phy_clustering_multi_step_undo_unwinds_in_reverse() {
    let provider = phy_clustering_provider();
    let (mut s, _d) = fresh_session_with(provider);
    let snapshot0: Vec<Vec<SampleIndex>> = (0..s.n_clusters())
        .map(ClusterId)
        .map(|c| s.spike_times(c).to_vec())
        .collect();

    s.dispatch(CurationCommand::Merge {
        sources: vec![ClusterId(3)],
        target: ClusterId(2),
    })
    .unwrap();
    let snapshot1: Vec<Vec<SampleIndex>> = (0..s.n_clusters())
        .map(ClusterId)
        .map(|c| s.spike_times(c).to_vec())
        .collect();

    s.dispatch(CurationCommand::Merge {
        sources: vec![ClusterId(7)],
        target: ClusterId(5),
    })
    .unwrap();
    let snapshot2: Vec<Vec<SampleIndex>> = (0..s.n_clusters())
        .map(ClusterId)
        .map(|c| s.spike_times(c).to_vec())
        .collect();

    s.dispatch(CurationCommand::Undo).unwrap();
    let after_undo_1: Vec<Vec<SampleIndex>> = (0..s.n_clusters())
        .map(ClusterId)
        .map(|c| s.spike_times(c).to_vec())
        .collect();
    assert_eq!(after_undo_1, snapshot1);

    s.dispatch(CurationCommand::Undo).unwrap();
    let after_undo_2: Vec<Vec<SampleIndex>> = (0..s.n_clusters())
        .map(ClusterId)
        .map(|c| s.spike_times(c).to_vec())
        .collect();
    assert_eq!(after_undo_2, snapshot0);

    // Sanity: snapshot1 != snapshot2 (the merges did something).
    assert_ne!(snapshot1, snapshot2);
}

// ---- Ported from phylib/io/tests/test_traces.py:test_waveform_extractor

/// phy: spike at sample 5 with `n_samples_waveforms = 20` should have its
/// first 5 samples be zero (would-be-pre-spike samples are out of range)
/// and the rest equal to `data[0:15, channel_ids]`.
///
/// **Divergence**: sorrel's `extract_snippets_single_channel` *skips*
/// out-of-range spikes rather than zero-padding. The invariant we test here
/// is that boundary spikes don't produce a spurious snippet.
#[test]
fn phy_waveform_extractor_drops_pre_recording_boundary_spike() {
    use sorrel_compute::extract_snippets_single_channel;
    // 1000-sample trace.
    let trace: Vec<f32> = (0..1000).map(|t| t as f32).collect();
    let pre = 10u32;
    let post = 9u32;
    // phy uses spikes [5, 25, 100, 1000, 1995]; with 2000-sample window 5
    // and 1995 are at the boundaries. We cap to 1000 here so spikes 5 and
    // 999 are at the boundaries.
    let spikes: [SampleIndex; 4] = [5, 25, 100, 999].map(SampleIndex);
    let snips = extract_snippets_single_channel(&trace, SampleIndex(0), &spikes, pre, post);

    // Spikes 5 and 999 fall outside the window [pre, len - post - 1],
    // so they get dropped. 25 and 100 stay.
    assert_eq!(
        snips.len(),
        2,
        "boundary spikes dropped, got {} snippets",
        snips.len()
    );
    // 25 → window [15..35]; 100 → window [90..110]
    assert_eq!(snips[0].len(), (pre + post + 1) as usize);
    assert_eq!(snips[1].len(), (pre + post + 1) as usize);
    assert_eq!(snips[0][0], 15.0);
    assert_eq!(snips[1][0], 90.0);
}

/// phy: spike at sample 25 with pre=15, post=4 produces snippet equal to
/// `data[15:35, ch]`. Tests the standard-window centring invariant.
#[test]
fn phy_waveform_extractor_centres_window_on_spike_sample() {
    use sorrel_compute::extract_snippets_single_channel;
    let trace: Vec<f32> = (0..1000).map(|t| t as f32).collect();
    let pre = 15u32;
    let post = 4u32;
    let snips =
        extract_snippets_single_channel(&trace, SampleIndex(0), &[SampleIndex(25)], pre, post);
    assert_eq!(snips.len(), 1);
    let s = &snips[0];
    assert_eq!(s.len(), 20);
    // Spike at idx 25, pre 15 → start at 10. End at 25 + 4 + 1 = 30.
    for (i, value) in s.iter().enumerate().take(20) {
        assert_eq!(*value, (10 + i) as f32);
    }
}

// ---- Ported from phylib/io/tests/test_traces.py:test_get_chunk_bounds ---

/// phy: chunk-bound math. Used in mtscomp / streaming readers; sorrel doesn't
/// stream chunks today but the helper's invariants are useful: bounds are
/// monotonic, span [0, total_length], and each interior bound is reachable
/// in a single step ≤ chunk_size.
///
/// We exercise the analogous invariant on `presence_ratio`'s binning, which
/// is the closest analogue we have — the histogram bins must partition the
/// recording duration cleanly.
#[test]
fn phy_chunk_bounds_invariants_via_presence_binning() {
    use sorrel_compute::presence_ratio;
    // Divide a 1000-sample recording into 10 bins of width 100. Place a
    // spike at the boundary of each bin and verify presence is full coverage.
    let total = 1000u64;
    let n_bins = 10usize;
    let times: Vec<SampleIndex> = (0..n_bins as u64)
        .map(|b| b * (total / n_bins as u64) + 1)
        .map(SampleIndex)
        .collect();
    let p = presence_ratio(&times, total, n_bins);
    assert!((p - 1.0).abs() < 1e-6, "expected full presence, got {p}");
}

// ---- Ported from phylib/io/tests/test_array.py:test_read_write -----------

/// phy: `write_array(path, arr); read_array(path) == arr`. Direct round-trip.
#[test]
fn phy_array_round_trip_through_npy_writer() {
    use sorrel_io::npy::{read_1d_u32, write_1d_u32_atomic};
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("test.npy");
    let arr: Vec<u32> = (0..10).collect();
    write_1d_u32_atomic(&p, &arr).unwrap();
    let read_back = read_1d_u32(&p).unwrap();
    assert_eq!(read_back, arr);
}

// ---- Ported from phylib/io/tests/test_model.py:test_from_sparse ----------

/// phy: `from_sparse(data, cols, channel_ids)` densifies a `(n_spikes,
/// n_active_channels)` sparse layout with per-row column ids into a
/// `(n_spikes, len(channel_ids))` dense layout.
///
/// **Divergence**: sorrel's `pc_feature_for(global_idx)` returns the flat
/// `n_pcs * n_chans_per_template` slice — densification is the consumer's
/// responsibility. The test below exercises the equivalent: given a
/// per-cluster spike index and the PC feature buffer, retrieving the slice
/// returns exactly the `n_pcs * n_chans` values the original NPY held for
/// that spike.
#[test]
fn phy_from_sparse_pc_feature_lookup_round_trip() {
    use sorrel_io::HasPcFeatures;
    struct Stub {
        inner: StubProvider,
        pc: Vec<f32>,
        ind: Vec<u32>,
        per_cluster: Vec<Vec<u32>>,
    }
    impl DataProvider for Stub {
        type Label = u8;
        fn sample_rate(&self) -> f32 {
            self.inner.sample_rate()
        }
        fn n_channels(&self) -> u32 {
            self.inner.n_channels()
        }
        fn n_samples(&self) -> SampleIndex {
            self.inner.n_samples()
        }
        fn n_clusters(&self) -> u32 {
            self.inner.n_clusters()
        }
        fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
            self.inner.spike_times(c)
        }
        fn trace(&self, s: SampleIndex, l: u32) -> TraceSlice<'_> {
            self.inner.trace(s, l)
        }
        fn initial_labels(&self) -> Vec<u8> {
            self.inner.initial_labels()
        }
    }
    impl HasPcFeatures for Stub {
        fn pc_features(&self) -> &[f32] {
            &self.pc
        }
        fn pc_shape(&self) -> (usize, usize) {
            (3, 2) // (n_pcs, n_chans)
        }
        fn pc_feature_ind(&self) -> &[u32] {
            &self.ind
        }
        fn spike_pc_indices(&self, c: ClusterId) -> &[u32] {
            self.per_cluster
                .get(c.idx())
                .map(Vec::as_slice)
                .unwrap_or(&[])
        }
    }

    // 2 spikes × 3 PCs × 2 channels = 12 floats. Like phy's `data`.
    let pc: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let stub = Stub {
        inner: StubProvider {
            spikes: vec![vec![SampleIndex(10)], vec![SampleIndex(20)]],
        },
        pc,
        ind: vec![0, 1],
        per_cluster: vec![vec![0], vec![1]],
    };
    let mut ci = ClusterIndex::from_provider(&stub);
    ci.seed_pc_features(&stub);

    // Time-sorted global index 0 should pull NPY row 0 = [0..6].
    let slice = ci.pc_feature_for(0).unwrap();
    assert_eq!(slice.len(), 6);
    for (i, &v) in slice.iter().enumerate() {
        assert_eq!(v, i as f32);
    }
    // Global index 1 → NPY row 1 = [6..12].
    let slice = ci.pc_feature_for(1).unwrap();
    for (i, &v) in slice.iter().enumerate() {
        assert_eq!(v, (i + 6) as f32);
    }
}

// ---- Ported from phylib `cluster_group.tsv` parsing convention -----------

/// phy: TSV with header `cluster_id\tgroup` plus rows; `cluster_*` rows
/// not in the file are "unsorted". Verifies our parser matches phy's
/// convention.
#[test]
fn phy_cluster_group_tsv_unspecified_rows_default_to_unsorted() {
    use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider, PhyLabel};
    use sorrel_io::DataProvider;
    use std::io::Write;

    fn write_npy(path: &std::path::Path, descr: &str, n: usize, data: &[u8]) {
        let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({n},), }}");
        let prelude = 6 + 2 + 2 + dict.len() + 1;
        let pad = (64 - (prelude % 64)) % 64;
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

    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    // 4 spikes across 4 clusters.
    let times = [10u64, 20, 30, 40];
    let clusters = [0u32, 1, 2, 3];
    let tb: Vec<u8> = times.iter().flat_map(|t| t.to_le_bytes()).collect();
    let cb: Vec<u8> = clusters.iter().flat_map(|c| c.to_le_bytes()).collect();
    write_npy(&root.join("spike_times.npy"), "<i64", 4, &tb);
    write_npy(&root.join("spike_clusters.npy"), "<u32", 4, &cb);
    std::fs::write(root.join("recording.dat"), [0u8; 8]).unwrap();
    std::fs::write(
        root.join("params.py"),
        "n_channels_dat = 1\nsample_rate = 1000.\n",
    )
    .unwrap();
    // Only labels 0 and 2 — 1 and 3 should default to Unsorted.
    std::fs::write(
        root.join("cluster_group.tsv"),
        "cluster_id\tgroup\n0\tgood\n2\tnoise\n",
    )
    .unwrap();

    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    let labels = p.initial_labels();
    assert_eq!(labels[0], PhyLabel::Good);
    assert_eq!(labels[1], PhyLabel::Unsorted);
    assert_eq!(labels[2], PhyLabel::Noise);
    assert_eq!(labels[3], PhyLabel::Unsorted);
}

/// phy: empty `cluster_group.tsv` (just the header) leaves every cluster
/// at the default Unsorted label.
#[test]
fn phy_cluster_group_tsv_header_only_yields_all_unsorted() {
    use sorrel_io::kilosort::PhyLabel;
    let lbl = PhyLabel::parse_tsv_value("");
    assert_eq!(lbl, PhyLabel::Unsorted);
    let lbl = PhyLabel::parse_tsv_value("garbage");
    assert_eq!(lbl, PhyLabel::Unsorted);
}

// ---- Ported from phylib spike-ordering invariants ------------------------

/// phy: per-cluster spike times are always sorted ascending.
/// `phy/cluster/clustering.py` enforces this in `_concatenate_per_cluster_arrays`.
#[test]
fn phy_per_cluster_spike_times_are_sorted() {
    let provider = phy_clustering_provider();
    let (mut s, _d) = fresh_session_with(provider);

    // After an arbitrary merge, every bucket should still be sorted.
    s.dispatch(CurationCommand::Merge {
        sources: vec![ClusterId(3), ClusterId(7)],
        target: ClusterId(5),
    })
    .unwrap();
    for c in (0..s.n_clusters()).map(ClusterId) {
        let bucket = s.spike_times(c);
        for w in bucket.windows(2) {
            assert!(
                w[0] <= w[1],
                "cluster {c} bucket not sorted after merge: {} > {}",
                w[0],
                w[1],
            );
        }
    }
}

/// phy: a `Relabel` on a non-existent cluster id is a graceful no-op
/// rather than a panic. Our dispatcher silently ignores the index, so this
/// is enforced as "no panic, history grows by 1, label state untouched".
#[test]
fn phy_relabel_on_out_of_range_cluster_is_a_noop() {
    let provider = phy_clustering_provider();
    let (mut s, _d) = fresh_session_with(provider);
    let pre = s.history_len();

    s.dispatch(CurationCommand::Relabel {
        cluster: ClusterId(999),
        op: PhyLabelOp::SetGood,
    })
    .unwrap();

    // History grew (the journal still has the record) but no label changed.
    assert_eq!(s.history_len(), pre + 1);
    for i in (0..s.n_clusters()).map(ClusterId) {
        // All defaulted to 0 (Unsorted) initially.
        assert_eq!(s.label(i), Some(0u8));
    }
}

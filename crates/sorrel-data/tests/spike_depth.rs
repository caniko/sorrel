//! `spike_depths` with Kilosort 4 `spike_positions.npy`: y is the depth, x
//! rides along, and both land on the right (cluster, local) spike.

use sorrel_data::session::ApplyPhyLabel;
use sorrel_data::{spike_depths, DepthSource, Journal, PhyLabelOp, Session};
use sorrel_io::{
    ClusterId, DataProvider, HasGeometry, HasPcFeatures, SampleIndex, TraceSamples, TraceSlice,
};

/// 6 spikes, NPY row `g` at time `10 + g` ... but clusters interleave, so the
/// per-cluster rows are not contiguous: cluster 0 has rows 0, 2, 4, cluster 1
/// rows 1, 5 and cluster 2 row 3. Row `g` sits at `[10g, 1000 + g]`.
struct Stub {
    spikes: Vec<Vec<SampleIndex>>,
    rows: Vec<Vec<u32>>,
    xy: Option<Vec<[f32; 2]>>,
}

fn stub(with_positions: bool) -> Stub {
    let t = SampleIndex;
    Stub {
        spikes: vec![
            vec![t(10), t(30), t(150)],
            vec![t(20), t(200)],
            vec![t(100)],
        ],
        rows: vec![vec![0, 2, 4], vec![1, 5], vec![3]],
        xy: with_positions.then(|| {
            (0..6)
                .map(|g| [10.0 * g as f32, 1000.0 + g as f32])
                .collect()
        }),
    }
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
        &self.spikes[c.idx()]
    }
    fn trace(&self, _: SampleIndex, _: u32) -> TraceSlice<'_> {
        TraceSlice {
            start: SampleIndex(0),
            n_channels: 0,
            samples: TraceSamples::I16(&[]),
        }
    }
    fn initial_labels(&self) -> Vec<u8> {
        vec![0; self.spikes.len()]
    }
}
impl ApplyPhyLabel for Stub {
    fn label_from_op(_: PhyLabelOp) -> u8 {
        0
    }
}
impl HasGeometry for Stub {
    fn channel_positions(&self) -> &[[f32; 2]] {
        &[[0.0, 0.0]]
    }
}
impl HasPcFeatures for Stub {
    fn pc_features(&self) -> &[f32] {
        &[]
    }
    fn pc_shape(&self) -> (usize, usize) {
        (0, 0)
    }
    fn pc_feature_ind(&self) -> &[u32] {
        &[]
    }
    fn spike_pc_indices(&self, c: ClusterId) -> &[u32] {
        &self.rows[c.idx()]
    }
    fn spike_positions(&self) -> Option<&[[f32; 2]]> {
        self.xy.as_deref()
    }
}

fn session(with_positions: bool) -> Session<Stub> {
    let dir = tempfile::tempdir().unwrap();
    let journal = Journal::open_or_create(&dir.path().join("j.journal"), 0, 3).unwrap();
    let mut s = Session::new(stub(with_positions), journal);
    s.seed_pc_features();
    s
}

#[test]
fn spike_positions_are_the_best_depth_and_map_to_cluster_and_local() {
    let d = spike_depths(&session(true)).unwrap();
    assert_eq!(d.source, DepthSource::SpikePositions);
    assert_eq!(
        d.per_cluster,
        [vec![1000., 1002., 1004.], vec![1001., 1005.], vec![1003.]]
    );
    assert_eq!(
        d.x.unwrap(),
        [vec![0., 20., 40.], vec![10., 50.], vec![30.]]
    );
    assert_eq!((d.min, d.max), (1000.0, 1005.0));
}

#[test]
fn without_positions_nothing_is_invented() {
    // No PCs, templates or positions: no depth source at all.
    assert!(spike_depths(&session(false)).is_none());
}

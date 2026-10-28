//! `CurationCommand::AddSpikes`: stable insertion, LIFO undo, journal replay
//! and the fail-closed gate, on a tiny phy-like provider.

use sorrel_data::command::SpikeBatch;
use sorrel_data::session::ApplyPhyLabel;
use sorrel_data::{CurationCommand, Journal, PhyLabelOp, Session};
use sorrel_io::{
    ClusterId, DataProvider, HasAmplitudes, HasPcFeatures, HasSpikeTemplates, SampleIndex,
    TraceSamples, TraceSlice,
};
use std::path::Path;
use std::sync::Arc;

/// 6 spikes, global (time) order t = 10,20,30,100,150,200 with clusters
/// 0,1,0,2,0,1. Row `g` carries amplitude `1+g`, template `g%2`, PCs
/// `[10g, 10g+1]`, template features `[g, g+.5]`, attribute `depth = 100+g`.
struct Stub {
    spikes: Vec<Vec<SampleIndex>>,
    amps: Vec<Vec<f32>>,
    templates: Vec<Vec<u32>>,
    rows: Vec<Vec<u32>>,
    phy: bool,
}

fn stub(phy: bool) -> Stub {
    let t = SampleIndex;
    Stub {
        spikes: vec![
            vec![t(10), t(30), t(150)],
            vec![t(20), t(200)],
            vec![t(100)],
        ],
        amps: vec![vec![1., 3., 5.], vec![2., 6.], vec![4.]],
        templates: vec![vec![0, 0, 0], vec![1, 1], vec![1]],
        rows: vec![vec![0, 2, 4], vec![1, 5], vec![3]],
        phy,
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
    fn phy_dir(&self) -> Option<&Path> {
        self.phy.then(|| Path::new("."))
    }
}
impl ApplyPhyLabel for Stub {
    fn label_from_op(_: PhyLabelOp) -> u8 {
        0
    }
}
impl HasAmplitudes for Stub {
    fn spike_amplitudes(&self, c: ClusterId) -> &[f32] {
        &self.amps[c.idx()]
    }
}
impl HasSpikeTemplates for Stub {
    fn spike_templates(&self, c: ClusterId) -> &[u32] {
        &self.templates[c.idx()]
    }
}
impl HasPcFeatures for Stub {
    fn pc_features(&self) -> &[f32] {
        static PC: [f32; 12] = [0., 1., 10., 11., 20., 21., 30., 31., 40., 41., 50., 51.];
        &PC
    }
    fn pc_shape(&self) -> (usize, usize) {
        (1, 2)
    }
    fn pc_feature_ind(&self) -> &[u32] {
        &[0, 1, 1, 2]
    }
    fn spike_pc_indices(&self, c: ClusterId) -> &[u32] {
        &self.rows[c.idx()]
    }
    fn template_features(&self) -> (Arc<[f32]>, usize, &[u32]) {
        let tf: Vec<f32> = (0..6).flat_map(|g| [g as f32, g as f32 + 0.5]).collect();
        (Arc::from(tf), 2, &[0, 1, 0, 1])
    }
    fn spike_attributes(&self) -> Vec<(String, Arc<[f32]>, usize)> {
        vec![(
            "depth".into(),
            Arc::from([100., 101., 102., 103., 104., 105.]),
            1,
        )]
    }
    fn spike_reorder(&self) -> Option<Arc<[u32]>> {
        Some(Arc::from([1, 2, 3, 4, 5, 0]))
    }
}

fn session(phy: bool, journal: &Path) -> Session<Stub> {
    let mut s = Session::new(stub(phy), Journal::open_or_create(journal, 0, 3).unwrap());
    s.seed_amplitudes();
    s.seed_templates();
    s.seed_pc_features();
    s
}

/// Ties at t=20 and t=100 must land after the originals.
fn batch() -> SpikeBatch {
    SpikeBatch {
        times: vec![20, 25, 100],
        clusters: vec![ClusterId(1), ClusterId(0), ClusterId(2)],
        templates: vec![1, 0, 1],
        amplitudes: vec![50., 51., 52.],
        pc_features: vec![900., 901., 910., 911., 920., 921.],
        template_features: vec![7., 7.5, 8., 8.5, 9., 9.5],
        attributes: vec![("depth".into(), 1, vec![900., 901., 902.])],
    }
}

/// Every per-spike array and bucket, floats as bits so NaN compares.
#[derive(Debug, PartialEq)]
struct Layout {
    times: Vec<u64>,
    clusters: Vec<u32>,
    amps: Vec<u32>,
    templates: Vec<u32>,
    shown: Vec<u64>,
    pc: Vec<Option<Vec<u32>>>,
    tf: Vec<Option<Vec<u32>>>,
    depth: Vec<Option<Vec<u32>>>,
    rows: Vec<Option<u32>>,
    buckets: Vec<Bucket>,
}

/// One cluster's (times, amplitude bits, templates, global indices).
type Bucket = (Vec<u64>, Vec<u32>, Vec<u32>, Vec<u32>);

fn layout(s: &Session<Stub>) -> Layout {
    let ci = s.cluster_index();
    let n = ci.n_spikes() as u32;
    let bits = |v: Option<&[f32]>| v.map(|v| v.iter().map(|x| x.to_bits()).collect());
    Layout {
        times: ci.all_spike_times().iter().map(|t| t.0).collect(),
        clusters: ci.spike_clusters().iter().map(|c| c.0).collect(),
        amps: ci
            .spike_amplitudes_global()
            .iter()
            .map(|a| a.to_bits())
            .collect(),
        templates: ci.spike_templates_global().to_vec(),
        shown: (0..n).map(|g| ci.display_time(g, true).0).collect(),
        pc: (0..n).map(|g| bits(ci.pc_feature_for(g))).collect(),
        tf: (0..n).map(|g| bits(ci.template_feature_row(g))).collect(),
        depth: (0..n)
            .map(|g| bits(ci.spike_attribute("depth", g)))
            .collect(),
        rows: ci.spike_pc_indices().iter().map(|r| r.get()).collect(),
        buckets: (0..ci.n_clusters())
            .map(ClusterId)
            .map(|c| {
                (
                    ci.spike_times(c).iter().map(|t| t.0).collect(),
                    ci.spike_amplitudes(c).iter().map(|a| a.to_bits()).collect(),
                    ci.spike_templates(c).to_vec(),
                    ci.global_indices(c).to_vec(),
                )
            })
            .collect(),
    }
}

fn split() -> CurationCommand {
    CurationCommand::Split {
        cluster: ClusterId(0),
        spike_idx: vec![1, 2],
        new_cluster: ClusterId(0),
        by: sorrel_data::command::Method::Manual,
    }
}

fn merge() -> CurationCommand {
    CurationCommand::Merge {
        sources: vec![ClusterId(1)],
        target: ClusterId(2),
    }
}

#[test]
fn new_spikes_land_in_time_order_with_their_features() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = session(true, &dir.path().join("j"));
    s.dispatch(CurationCommand::AddSpikes(batch())).unwrap();
    let times = |c| -> Vec<u64> { s.spike_times(ClusterId(c)).iter().map(|t| t.0).collect() };
    assert_eq!(times(0), [10, 25, 30, 150]);
    assert_eq!(times(1), [20, 20, 200]);
    assert_eq!(times(2), [100, 100]);
    assert_eq!(s.spike_amplitudes(ClusterId(1)), [2., 50., 6.]);
    assert_eq!(s.spike_amplitudes(ClusterId(2)), [4., 52.]);
    assert_eq!(s.spike_templates(ClusterId(0)), [0, 0, 0, 0]);
    // Global order: 10,20,20n,25n,30,100,100n,150,200.
    let ci = s.cluster_index();
    assert_eq!(ci.n_spikes(), 9);
    assert_eq!(ci.pc_feature_for(1), Some(&[10., 11.][..]));
    assert_eq!(ci.pc_feature_for(2), Some(&[900., 901.][..]));
    assert_eq!(ci.pc_feature_for(6), Some(&[920., 921.][..]));
    assert_eq!(ci.template_feature_row(3), Some(&[8., 8.5][..]));
    assert_eq!(ci.spike_attribute("depth", 3), Some(&[901.][..]));
    assert_eq!(ci.spike_attribute("depth", 4), Some(&[102.][..]));
    assert_eq!(ci.n_orig_rows(), 6);
    // spike_reorder: new spikes show at their own time.
    assert_eq!(ci.display_time(2, true).0, 20);
    assert_eq!(s.history_len(), 1);
}

#[test]
fn add_split_merge_then_undo_restores_every_array_bitwise() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = session(true, &dir.path().join("j"));
    let before = layout(&s);
    s.dispatch(CurationCommand::AddSpikes(batch())).unwrap();
    let added = layout(&s);
    assert_ne!(added, before);
    s.dispatch(split()).unwrap();
    s.dispatch(merge()).unwrap();
    s.dispatch(CurationCommand::Undo).unwrap();
    s.dispatch(CurationCommand::Undo).unwrap();
    // Stored global indices of the split/merge stayed valid: undoing them
    // lands exactly on the post-add layout.
    assert_eq!(layout(&s), added);
    s.dispatch(CurationCommand::Undo).unwrap();
    assert_eq!(layout(&s), before);
    // Redo re-applies the add identically.
    s.dispatch(CurationCommand::Redo).unwrap();
    assert_eq!(layout(&s), added);
}

#[test]
fn journal_replay_reproduces_post_add_state() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("j");
    let want = {
        let mut s = session(true, &path);
        s.dispatch(CurationCommand::AddSpikes(batch())).unwrap();
        s.dispatch(split()).unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        layout(&s)
    };
    let mut s = session(true, &path);
    s.replay_journal().unwrap();
    assert_eq!(layout(&s), want);
    assert_eq!((s.history_len(), s.redo_len()), (1, 1));
}

#[test]
fn gate_fails_closed_without_touching_journal_or_state() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = session(false, &dir.path().join("no_phy"));
    assert!(s.dispatch(CurationCommand::AddSpikes(batch())).is_err());
    assert_eq!(s.journal().n_records(), 0);

    let mut s = session(true, &dir.path().join("j"));
    let before = layout(&s);
    type Mutate = Box<dyn Fn(&mut SpikeBatch)>;
    let bad: Vec<Mutate> = vec![
        Box::new(|b| b.amplitudes.pop().map(drop).unwrap_or_default()),
        Box::new(|b| b.pc_features.pop().map(drop).unwrap_or_default()),
        Box::new(|b| b.times = vec![30, 20, 40]),
        Box::new(|b| b.clusters[0] = ClusterId(99)),
        Box::new(|b| b.attributes.clear()),
        Box::new(|b| *b = SpikeBatch::default()),
    ];
    for f in bad {
        let mut b = batch();
        f(&mut b);
        assert!(s.dispatch(CurationCommand::AddSpikes(b)).is_err());
    }
    assert_eq!(s.journal().n_records(), 0);
    assert_eq!(s.history_len(), 0);
    assert_eq!(layout(&s), before);

    // A failing child rolls the whole batch back and never reaches the journal.
    let mut b = batch();
    b.amplitudes.pop();
    let cmd = CurationCommand::batch(vec![split(), CurationCommand::AddSpikes(b)]).unwrap();
    assert!(s.dispatch(cmd).is_err());
    assert_eq!((s.journal().n_records(), layout(&s)), (0, before));

    s.set_record_sink(|_| {});
    assert!(s.dispatch(CurationCommand::AddSpikes(batch())).is_err());
}

#[test]
fn add_spikes_round_trips_through_messagepack() {
    let cmd = CurationCommand::AddSpikes(batch());
    let back: CurationCommand = rmp_serde::from_slice(&rmp_serde::to_vec(&cmd).unwrap()).unwrap();
    assert!(matches!(back, CurationCommand::AddSpikes(b) if b == batch()));
}

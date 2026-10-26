//! `export_phy`: both export paths and the overwrite guard, checked by
//! reopening the result as a phy directory.
#![cfg(feature = "phy-export")]

use sorrel_data::{export_phy, open_phy, seed_phy, CurationCommand, Journal, JournalOpts};
use sorrel_data::{PhyLabelOp, Session};
use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider, PhyLabel};
use sorrel_io::npy::{read_1d_u32, read_npy_f32_flat, write_1d_u32_atomic};
use sorrel_io::{
    ClusterId, DataProvider, HasGeometry, PhyParams, SampleIndex, TraceSamples, TraceSlice,
};
use std::io::Write;
use std::path::Path;

fn label_str(l: &PhyLabel) -> &'static str {
    l.as_str()
}

fn write_npy_v1(path: &Path, descr: &str, n: usize, data: &[u8]) {
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({n},), }}");
    let pad = (64 - ((11 + dict.len()) % 64)) % 64;
    let mut header = dict.into_bytes();
    header.resize(header.len() + pad, b' ');
    header.push(b'\n');
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"\x93NUMPY\x01\x00").unwrap();
    f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
    f.write_all(&header).unwrap();
    f.write_all(data).unwrap();
}

/// A Kilosort directory whose `spike_times.npy` is unsorted and has a
/// cross-cluster tie at t=10, so time order differs from row order.
fn write_ks_fixture(root: &Path) {
    let times: [u64; 6] = [50, 10, 30, 10, 200, 100];
    let clusters: [u32; 6] = [1, 0, 0, 2, 1, 2];
    let amps: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    write_npy_v1(
        &root.join("spike_times.npy"),
        "<u8",
        6,
        &times
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    write_1d_u32_atomic(&root.join("spike_clusters.npy"), &clusters).unwrap();
    write_npy_v1(
        &root.join("amplitudes.npy"),
        "<f4",
        6,
        &amps
            .iter()
            .flat_map(|a| a.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    std::fs::write(
        root.join("cluster_KSLabel.tsv"),
        "cluster_id\tKSLabel\n0\tgood\n",
    )
    .unwrap();
    std::fs::write(root.join("temp_wh.dat"), [0u8; 64]).unwrap();
    std::fs::create_dir(root.join(".phy")).unwrap();
    std::fs::write(root.join("recording.dat"), vec![0u8; 4 * 256 * 2]).unwrap();
    std::fs::write(
        root.join("params.py"),
        "dat_path = 'recording.dat'\nn_channels_dat = 4\nsample_rate = 1000.\n\
         dtype = 'int16'\nhp_filtered = True\n",
    )
    .unwrap();
}

#[test]
fn kilosort_export_copies_the_dir_and_keeps_row_order() {
    let src = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    write_ks_fixture(src.path());
    let source_clusters = std::fs::read(src.path().join("spike_clusters.npy")).unwrap();

    let mut s = open_phy(
        src.path(),
        KilosortOpenParams::default(),
        &JournalOpts::default(),
    )
    .unwrap();
    seed_phy(&mut s).unwrap();
    s.dispatch(CurationCommand::Merge {
        sources: vec![ClusterId(2)],
        target: ClusterId(0),
    })
    .unwrap();
    s.dispatch(CurationCommand::Relabel {
        cluster: ClusterId(1),
        op: PhyLabelOp::SetGood,
    })
    .unwrap();
    export_phy(&s, out.path(), false, label_str).unwrap();

    let o = out.path();
    // Rows [1,0,0,2,1,2] with 2 merged into 0; time order would be [0,0,0,1,0,1].
    assert_eq!(
        read_1d_u32(&o.join("spike_clusters.npy")).unwrap(),
        [1, 0, 0, 0, 1, 0]
    );
    // Cluster 0 is seeded from cluster_KSLabel.tsv; cluster 1 was relabelled.
    assert_eq!(
        std::fs::read_to_string(o.join("cluster_group.tsv")).unwrap(),
        "cluster_id\tgroup\n0\tgood\n1\tgood\n"
    );
    for copied in ["spike_times.npy", "amplitudes.npy", "cluster_KSLabel.tsv"] {
        assert_eq!(
            std::fs::read(o.join(copied)).unwrap(),
            std::fs::read(src.path().join(copied)).unwrap(),
            "{copied} is copied byte for byte"
        );
    }
    for left_behind in [
        "temp_wh.dat",
        "recording.dat",
        "sorrel.journal",
        "curation.json",
        ".phy",
    ] {
        assert!(
            !o.join(left_behind).exists(),
            "{left_behind} must not be exported"
        );
    }
    let params = PhyParams::read(&o.join("params.py")).unwrap();
    let dat = params.dat_path.unwrap();
    assert!(dat.is_absolute());
    assert_eq!(
        dat.canonicalize().unwrap(),
        src.path().join("recording.dat").canonicalize().unwrap()
    );
    assert_eq!(params.hp_filtered, Some(true));
    assert_eq!(
        std::fs::read(src.path().join("spike_clusters.npy")).unwrap(),
        source_clusters,
        "the source is untouched"
    );

    let reopened = KilosortProvider::open(o, KilosortOpenParams::default()).unwrap();
    let times = |c| {
        reopened
            .spike_times(ClusterId(c))
            .iter()
            .map(|t| t.0)
            .collect::<Vec<_>>()
    };
    assert_eq!(times(0), [10, 10, 30, 100]);
    assert_eq!(times(1), [50, 200]);
    assert!(times(2).is_empty());
    assert_eq!(reopened.initial_labels()[1], PhyLabel::Good);
}

#[test]
fn export_refuses_to_overwrite_its_source_directory() {
    let src = tempfile::tempdir().unwrap();
    write_ks_fixture(src.path());
    let before = std::fs::read(src.path().join("spike_times.npy")).unwrap();
    let s = open_phy(
        src.path(),
        KilosortOpenParams::default(),
        &JournalOpts::default(),
    )
    .unwrap();
    assert!(export_phy(&s, src.path(), false, label_str).is_err());
    assert_eq!(
        std::fs::read(src.path().join("spike_times.npy")).unwrap(),
        before
    );
}

/// An in-memory backend with no phy directory and no raw file (like NWB):
/// 3 channels at 10 kHz, cluster 0 firing on channel 1, cluster 1 on 2.
struct Planted {
    spikes: Vec<Vec<SampleIndex>>,
    trace: Vec<i16>,
}

const N_CH: usize = 3;
const N_SAMPLES: usize = 1000;

impl Planted {
    fn new() -> Self {
        let spikes = vec![vec![100, 300, 500], vec![200, 400]];
        let mut trace = vec![0i16; N_SAMPLES * N_CH];
        for (channel, times) in [1, 2].into_iter().zip(&spikes) {
            for &t in times {
                trace[t * N_CH + channel] = -100;
            }
        }
        let spikes = spikes
            .into_iter()
            .map(|ts| ts.into_iter().map(|t| SampleIndex(t as u64)).collect())
            .collect();
        Self { spikes, trace }
    }
}

impl DataProvider for Planted {
    type Label = PhyLabel;
    fn sample_rate(&self) -> f32 {
        10_000.0
    }
    fn n_channels(&self) -> u32 {
        N_CH as u32
    }
    fn n_samples(&self) -> SampleIndex {
        SampleIndex(N_SAMPLES as u64)
    }
    fn n_clusters(&self) -> u32 {
        self.spikes.len() as u32
    }
    fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
        &self.spikes[c.idx()]
    }
    fn trace(&self, start: SampleIndex, len: u32) -> TraceSlice<'_> {
        let s = (start.0 as usize * N_CH).min(self.trace.len());
        let e = (s + len as usize * N_CH).min(self.trace.len());
        TraceSlice {
            start,
            n_channels: N_CH as u32,
            samples: TraceSamples::I16(&self.trace[s..e]),
        }
    }
    fn initial_labels(&self) -> Vec<PhyLabel> {
        vec![PhyLabel::Unsorted; self.spikes.len()]
    }
}

impl HasGeometry for Planted {
    fn channel_positions(&self) -> &[[f32; 2]] {
        &[[0.0, 0.0], [0.0, 20.0], [0.0, 40.0]]
    }
}

#[test]
fn generic_export_computes_templates_and_streams_the_trace() {
    let work = tempfile::tempdir().unwrap();
    let out = work.path().join("phy");
    let journal = Journal::open(&work.path().join("j.journal")).unwrap();
    let s = Session::new(Planted::new(), journal);
    export_phy(&s, &out, false, label_str).unwrap();

    // 1 ms before + 2 ms after at 10 kHz.
    let (templates, shape) = read_npy_f32_flat(&out.join("templates.npy")).unwrap();
    assert_eq!(shape, [2, 30, N_CH]);
    for (cluster, channel) in [(0, 1), (1, 2)] {
        let t = &templates[cluster * 30 * N_CH..(cluster + 1) * 30 * N_CH];
        let trough = (0..t.len()).min_by(|&a, &b| t[a].total_cmp(&t[b])).unwrap();
        assert_eq!(
            (trough / N_CH, trough % N_CH),
            (10, channel),
            "cluster {cluster}"
        );
    }
    assert_eq!(
        read_1d_u32(&out.join("spike_templates.npy")).unwrap(),
        read_1d_u32(&out.join("spike_clusters.npy")).unwrap()
    );
    let dat = std::fs::read(out.join("recording.dat")).unwrap();
    assert_eq!(dat, bytemuck_i16(&s.provider.trace));
    assert_eq!(
        PhyParams::read(&out.join("params.py"))
            .unwrap()
            .dat_path
            .as_deref(),
        Some(Path::new("recording.dat"))
    );

    let reopened = KilosortProvider::open(&out, KilosortOpenParams::default()).unwrap();
    assert_eq!(reopened.n_channels(), N_CH as u32);
    assert_eq!(reopened.n_samples(), SampleIndex(N_SAMPLES as u64));
    assert_eq!(
        reopened.spike_times(ClusterId(1)),
        &s.provider.spikes[1][..]
    );
}

fn bytemuck_i16(v: &[i16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

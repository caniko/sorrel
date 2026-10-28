//! Synthetic phy directory: 4 channels, 2 good units, 10 s at 30 kHz of
//! Gaussian noise plus known templates; 25 of unit 0's 100 spikes are left
//! out of `spike_times.npy` but present in the `.dat`.

use super::*;
use clap::Parser;
use sorrel_data::Journal;
use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider};
use sorrel_io::npy;
use std::path::Path;

const N: usize = 300_000;
const NC: usize = 4;
const SIGMA: f32 = 10.0;

struct Xs(u64);
impl Xs {
    fn u(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
        (-2.0 * self.u().ln()).sqrt() * (std::f32::consts::TAU * self.u()).cos()
    }
}

fn shape(k: usize) -> f32 {
    let x = (k as f32 - PRE as f32) / 3.0;
    -(-x * x / 2.0).exp() + 0.35 * (-(x - 2.5).powi(2) / 2.0).exp()
}

fn write_u32_2d(path: &Path, rows: usize, cols: usize, data: &[u32]) {
    let dict = format!("{{'descr': '<u4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    let mut header = dict.into_bytes();
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend((header.len() as u16).to_le_bytes());
    out.extend(header);
    out.extend(data.iter().flat_map(|v| v.to_le_bytes()));
    std::fs::write(path, out).unwrap();
}

/// Returns the omitted unit-0 spike times.
fn write_fixture(root: &Path) -> Vec<u64> {
    let mut r = Xs(0x1234_5678_9abc_def1);
    // Unit weights per channel and peak amplitude. Both units span every
    // channel (the channel-shift null only breaks spikes that need several
    // channels to line up) with sign patterns whose median is ~0, so the
    // 4-channel median reference leaves the waveform alone.
    let w: [[f32; NC]; 2] = [[1.0, 0.7, -0.7, -1.0], [1.0, -1.0, 0.7, -0.7]];
    let amp = [40.0f32, 25.0];
    let mut raw: Vec<f32> = (0..N * NC).map(|_| SIGMA * r.gauss()).collect();
    // (time, unit, scale), >= 150 samples apart.
    let mut spikes: Vec<(u64, usize, f32)> = vec![];
    for (u, n) in [(0usize, 100usize), (1, 80)] {
        while spikes.iter().filter(|s| s.1 == u).count() < n {
            let t = 200 + (r.u() * (N - 400) as f32) as u64;
            if spikes.iter().all(|s| s.0.abs_diff(t) > 150) {
                spikes.push((t, u, 0.8 + 0.4 * r.u()));
            }
        }
    }
    spikes.sort_by_key(|s| s.0);
    for &(t, u, sc) in &spikes {
        for k in 0..NT {
            for ch in 0..NC {
                raw[(t as usize - PRE + k) * NC + ch] += sc * amp[u] * w[u][ch] * shape(k);
            }
        }
    }
    let bytes: Vec<u8> = raw
        .iter()
        .flat_map(|v| (v.round() as i16).to_le_bytes())
        .collect();
    std::fs::write(root.join("recording.dat"), bytes).unwrap();

    let mut omitted = vec![];
    let mut seen0 = 0;
    let kept: Vec<(u64, usize, f32)> = spikes
        .into_iter()
        .filter(|&(t, u, _)| {
            if u == 0 {
                seen0 += 1;
                if seen0 % 4 == 0 {
                    omitted.push(t);
                    return false;
                }
            }
            true
        })
        .collect();
    let n = kept.len();
    let times: Vec<u64> = kept.iter().map(|s| s.0).collect();
    let cl: Vec<u32> = kept.iter().map(|s| s.1 as u32).collect();
    npy::write_1d_u64_atomic(&root.join("spike_times.npy"), &times).unwrap();
    npy::write_1d_u32_atomic(&root.join("spike_clusters.npy"), &cl).unwrap();
    npy::write_1d_u32_atomic(&root.join("spike_templates.npy"), &cl).unwrap();
    let amps: Vec<f32> = kept.iter().map(|s| s.2 * amp[s.1]).collect();
    npy::write_f32_atomic(&root.join("amplitudes.npy"), &[n], &amps).unwrap();
    // 2 PCs x 4 channels, linear in the spike scale plus a little noise.
    let pcs: Vec<f32> = kept
        .iter()
        .flat_map(|&(_, u, sc)| {
            let mut v = vec![];
            for p in 0..2 {
                for wc in w[u] {
                    v.push(sc * wc * (10.0 - 4.0 * p as f32) + 0.02 * r.gauss());
                }
            }
            v
        })
        .collect();
    npy::write_f32_atomic(&root.join("pc_features.npy"), &[n, 2, NC], &pcs).unwrap();
    write_u32_2d(
        &root.join("pc_feature_ind.npy"),
        2,
        NC,
        &[0, 1, 2, 3, 0, 1, 2, 3],
    );
    std::fs::write(
        root.join("params.py"),
        format!(
            "dat_path = 'recording.dat'\nn_channels_dat = {NC}\ndtype = 'int16'\n\
             offset = 0\nsample_rate = 30000.0\nhp_filtered = False\n"
        ),
    )
    .unwrap();
    std::fs::write(
        root.join("cluster_group.tsv"),
        "cluster_id\tgroup\n0\tgood\n1\tgood\n",
    )
    .unwrap();
    omitted
}

fn open(root: &Path) -> Session<KilosortProvider> {
    let provider = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    let journal =
        Journal::open_or_create(&root.join("sorrel.journal"), 0, provider.n_clusters()).unwrap();
    let mut s = Session::new(provider, journal);
    s.seed_amplitudes();
    s.seed_templates();
    s.seed_pc_features();
    s.seed_template_waveforms();
    s.replay_journal().unwrap();
    s
}

fn args(extra: &[&str]) -> RescueArgs {
    #[derive(Parser)]
    struct W {
        #[command(flatten)]
        a: RescueArgs,
    }
    W::parse_from(std::iter::once("t").chain(extra.iter().copied())).a
}

fn found(out: &Outcome, unit: u32, t: u64) -> bool {
    out.added_times
        .iter()
        .any(|(u, ts)| *u == unit && ts.iter().any(|x| x.abs_diff(t) <= 15))
}

#[test]
fn dry_run_recovers_omitted_spikes_within_the_fdr() {
    let dir = tempfile::tempdir().unwrap();
    let omitted = write_fixture(dir.path());
    let s = open(dir.path());
    let t0 = std::time::Instant::now();
    let out = run(&s, &args(&["--cpu"]), &[]).unwrap();
    eprintln!("cpu rescue: {:?}", t0.elapsed());
    for u in &out.units {
        eprintln!(
            "unit {} added {} theta {:?} fp {} isi {}->{} r2 {} {}",
            u.id, u.added, u.theta, u.false_positives, u.isi_before, u.isi_after, u.min_r2, u.note
        );
    }
    let hit = omitted.iter().filter(|&&t| found(&out, 0, t)).count();
    assert!(
        hit * 5 >= omitted.len() * 4,
        "recovered {hit}/{}",
        omitted.len()
    );
    let fp: f32 = out.units.iter().map(|u| u.false_positives).sum();
    assert!(
        fp <= 0.05 * out.added as f32 + 2.0,
        "fp {fp} of {}",
        out.added
    );
    assert!(out.cmd.is_none());
    // The scale feature is noisy for the low-SNR unit 1; only unit 0 must predict well.
    let r2 = out.units.iter().find(|u| u.id == 0).unwrap().min_r2;
    assert!(r2 > 0.5, "unit 0 min R2 {r2}");
    // The GPU path (CPU reference when no adapter exists) agrees.
    let t0 = std::time::Instant::now();
    let gpu = run(&s, &args(&[]), &[]).unwrap();
    eprintln!("gpu rescue: {:?}", t0.elapsed());
    assert!(
        gpu.added.abs_diff(out.added) <= 2,
        "{} vs {}",
        gpu.added,
        out.added
    );
}

#[test]
fn hybrid_injection_reports_high_recall() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path());
    let s = open(dir.path());
    let out = run(
        &s,
        &args(&["--cpu", "--hybrid-frac", "0.1", "--apply"]),
        &[],
    )
    .unwrap();
    assert!(out.cmd.is_none(), "hybrid runs are dry");
    let (rec, n) = out
        .units
        .iter()
        .find(|u| u.id == 0)
        .unwrap()
        .recall
        .unwrap();
    assert!(n >= 5);
    assert!(rec[0] >= 0.9, "recall {rec:?}");
}

#[test]
fn apply_journals_one_batch_and_adds_the_spikes() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path());
    let mut s = open(dir.path());
    let before = s.cluster_index().n_spikes();
    let out = run(&s, &args(&["--cpu", "--apply"]), &[]).unwrap();
    assert!(out.added > 0);
    s.dispatch(out.cmd.unwrap()).unwrap();
    assert_eq!(s.history_len(), 1);
    assert_eq!(s.cluster_index().n_spikes(), before + out.added);
    assert!(s
        .tags(ClusterId::new(0))
        .iter()
        .any(|t| t == "review:rescue"));
    // A second pass finds (almost) nothing new: the additions are accepted spikes now.
    let again = run(&s, &args(&["--cpu"]), &[]).unwrap();
    assert!(again.added <= 3, "{} new spikes on rerun", again.added);
}

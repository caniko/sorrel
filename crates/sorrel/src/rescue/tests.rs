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
    write_fixture_ext(root, false, false)
}

/// `merged`: a third `mua` cluster 2 of 30 unit-0-shaped spikes spread over
/// three templates (no template reaches `MIN_SPIKES`, so it cannot be
/// trained). `empty0`: unit 0 is relabelled cluster 2, leaving cluster 0
/// empty, and template features are written.
fn write_fixture_ext(root: &Path, merged: bool, empty0: bool) -> Vec<u64> {
    let mut r = Xs(0x1234_5678_9abc_def1);
    // Unit weights per channel and peak amplitude. Both units span every
    // channel (the channel-shift null only breaks spikes that need several
    // channels to line up) with sign patterns whose median is ~0, so the
    // 4-channel median reference leaves the waveform alone.
    let w: [[f32; NC]; 3] = [
        [1.0, 0.7, -0.7, -1.0],
        [1.0, -1.0, 0.7, -0.7],
        [1.0, 0.7, -0.7, -1.0],
    ];
    let amp = [40.0f32, 25.0, 40.0];
    let mut raw: Vec<f32> = (0..N * NC).map(|_| SIGMA * r.gauss()).collect();
    // (time, unit, scale), >= 150 samples apart.
    let mut spikes: Vec<(u64, usize, f32)> = vec![];
    let counts = [(0usize, 100usize), (1, 80), (2, 30)];
    for (u, n) in counts.into_iter().take(if merged { 3 } else { 2 }) {
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
    let tpl: Vec<u32> = kept
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if s.1 == 2 {
                2 + (i % 3) as u32
            } else {
                s.1 as u32
            }
        })
        .collect();
    let cl: Vec<u32> = kept
        .iter()
        .map(|s| if empty0 && s.1 == 0 { 2 } else { s.1 as u32 })
        .collect();
    npy::write_1d_u64_atomic(&root.join("spike_times.npy"), &times).unwrap();
    npy::write_1d_u32_atomic(&root.join("spike_clusters.npy"), &cl).unwrap();
    npy::write_1d_u32_atomic(&root.join("spike_templates.npy"), &tpl).unwrap();
    if empty0 {
        let tf: Vec<f32> = (0..n * 2).map(|i| i as f32).collect();
        npy::write_f32_atomic(&root.join("template_features.npy"), &[n, 2], &tf).unwrap();
        write_u32_2d(
            &root.join("template_feature_ind.npy"),
            3,
            2,
            &[0, 1, 1, 0, 0, 1],
        );
    }
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
    let n_t = if merged { 5 } else { 2 };
    let ind: Vec<u32> = (0..n_t).flat_map(|_| 0..NC as u32).collect();
    write_u32_2d(&root.join("pc_feature_ind.npy"), n_t, NC, &ind);
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
        if merged {
            "cluster_id\tgroup\n0\tgood\n1\tgood\n2\tmua\n"
        } else {
            "cluster_id\tgroup\n0\tgood\n1\tgood\n"
        },
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

fn try_args(extra: &[&str]) -> Result<RescueArgs, clap::Error> {
    #[derive(Parser)]
    struct W {
        #[command(flatten)]
        a: RescueArgs,
    }
    W::try_parse_from(std::iter::once("t").chain(extra.iter().copied())).map(|w| w.a)
}

#[test]
fn q_must_be_a_proper_fdr() {
    for bad in ["0", "1", "1.5", "-0.1", "nan", "inf"] {
        assert!(try_args(&["--q", bad]).is_err(), "--q {bad} accepted");
    }
    assert_eq!(try_args(&["--q", "0.1"]).unwrap().q, 0.1);
}

#[test]
fn untrainable_competitor_is_reported_and_its_spikes_are_not_added() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture_ext(dir.path(), true, false);
    let s = open(dir.path());
    let out = run(&s, &args(&["--cpu"]), &[]).unwrap();
    assert_eq!(out.unmodelled, vec![2]);
    // Cluster 2's spikes look like unit 0's; none may be added to anyone.
    let refr = (0.0015 * s.provider.sample_rate()).ceil() as u64;
    for t2 in s.spike_times(ClusterId::new(2)) {
        for (u, ts) in &out.added_times {
            assert!(
                ts.iter().all(|t| t.abs_diff(t2.0) >= refr),
                "unit {u} gained a spike at cluster 2's {}",
                t2.0
            );
        }
    }
}

#[test]
fn template_feature_width_does_not_depend_on_cluster_0() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture_ext(dir.path(), false, true);
    let s = open(dir.path());
    assert!(s.cluster_index().has_template_features());
    assert!(s.spike_times(ClusterId::new(0)).is_empty());
    assert!(groups_of(&s)
        .iter()
        .any(|(k, w)| matches!(k, Kind::Tf) && *w == 2));
}

#[test]
fn injections_avoid_all_existing_spikes_and_each_other() {
    let gap = 60;
    // A spike every 200 samples leaves [t+gap, t+200-gap) = 80-sample slots.
    // (At a 100-sample period nothing fits, which is what the first draft hit.)
    let existing: Vec<u64> = (0..500).map(|i| 1000 + 200 * i).collect();
    let mut chosen = vec![];
    let mut rng = Rng::new(7);
    let a = place_injections(200, (1000, 100_000), gap, &existing, &mut chosen, &mut rng);
    let b = place_injections(200, (1000, 100_000), gap, &existing, &mut chosen, &mut rng);
    assert!(!a.is_empty() && !b.is_empty());
    let mut ws: Vec<u64> = a.iter().chain(&b).copied().collect();
    for &w in &ws {
        let t = w + PRE as u64;
        assert!(
            existing.iter().all(|&e| e.abs_diff(t) >= gap),
            "{w} near a spike"
        );
    }
    ws.sort_unstable();
    assert!(
        ws.windows(2).all(|p| p[1] - p[0] >= gap),
        "injections overlap"
    );
}

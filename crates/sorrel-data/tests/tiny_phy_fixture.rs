//! `release/fixtures/tiny-phy/`: the phy directory `release/smoke-windows.ps1`
//! feeds to `sorrel qc`, `report` and the GUI on a real Windows runner. It is
//! generated here, deterministically, so the committed bytes can be checked:
//! `SORREL_WRITE_FIXTURE=1 cargo test -p sorrel-data --test tiny_phy_fixture`
//! rewrites them after an intended change.

use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider};
use sorrel_io::{ClusterId, DataProvider};
use std::path::{Path, PathBuf};

const N_CH: usize = 4;
const N_SAMPLES: usize = 45_000; // 1.5 s at 30 kHz
const N_CLUSTERS: usize = 3;
const N_SPIKES: usize = 2000;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn npy(descr: &str, data: &[u8], n: usize) -> Vec<u8> {
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({n},), }}");
    let pad = (64 - (6 + 2 + 2 + dict.len() + 1) % 64) % 64;
    let mut header = dict.into_bytes();
    header.resize(header.len() + pad, b' ');
    header.push(b'\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend((header.len() as u16).to_le_bytes());
    out.extend(header);
    out.extend(data);
    out
}

/// File name -> bytes. Unit `k` is a negative biphasic-ish dip on channel
/// `k` (amplitude 250 + 100 k counts) over uniform noise.
fn files() -> Vec<(&'static str, Vec<u8>)> {
    let mut rng = Rng(0x5eed_5044_1234_abcd);
    let mut spikes: Vec<(u64, u32)> = Vec::with_capacity(N_SPIKES);
    for k in 0..N_CLUSTERS {
        let n = N_SPIKES / N_CLUSTERS + usize::from(k < N_SPIKES % N_CLUSTERS);
        let period = (N_SAMPLES - 120) / n;
        for i in 0..n {
            let jitter = rng.next() as usize % (period / 2);
            spikes.push(((60 + i * period + jitter) as u64, k as u32));
        }
    }
    spikes.sort();

    let mut dat: Vec<i16> = (0..N_SAMPLES * N_CH)
        .map(|_| (rng.next() % 41) as i16 - 20)
        .collect();
    for &(t, k) in &spikes {
        let amp = 250.0 + 100.0 * k as f32;
        for j in 0..24usize {
            let x = (j as f32 - 8.0) / 4.0;
            let v = (-amp * (-x * x).exp() + 0.35 * amp * (-(x - 2.0) * (x - 2.0)).exp()) as i16;
            dat[(t as usize - 8 + j) * N_CH + k as usize] += v;
        }
    }
    let le = |it: &mut dyn Iterator<Item = Vec<u8>>| it.flatten().collect::<Vec<u8>>();
    let times = le(&mut spikes.iter().map(|s| s.0.to_le_bytes().to_vec()));
    let clusters = le(&mut spikes.iter().map(|s| s.1.to_le_bytes().to_vec()));
    vec![
        ("spike_times.npy", npy("<i64", &times, spikes.len())),
        ("spike_clusters.npy", npy("<u4", &clusters, spikes.len())),
        (
            "params.py",
            format!(
                "dat_path = 'recording.dat'\nn_channels_dat = {N_CH}\ndtype = 'int16'\n\
                 offset = 0\nsample_rate = 30000.\nhp_filtered = False\n"
            )
            .into_bytes(),
        ),
        (
            "recording.dat",
            dat.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    ]
}

fn committed_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../release/fixtures/tiny-phy")
}

#[test]
fn committed_fixture_matches_generator_and_opens() {
    let dir = committed_dir();
    if std::env::var_os("SORREL_WRITE_FIXTURE").is_some() {
        std::fs::create_dir_all(&dir).unwrap();
        for (name, bytes) in files() {
            std::fs::write(dir.join(name), bytes).unwrap();
        }
    }
    // The Nix source filter drops release/, so the nix checks have no copy.
    if !dir.exists() {
        eprintln!("skipping: {} not in this source tree", dir.display());
        return;
    }
    for (name, bytes) in files() {
        assert_eq!(
            std::fs::read(dir.join(name)).unwrap(),
            bytes,
            "{name} drifted; rerun with SORREL_WRITE_FIXTURE=1"
        );
    }
    let p = KilosortProvider::open(&dir, KilosortOpenParams::default()).unwrap();
    assert_eq!(p.n_clusters() as usize, N_CLUSTERS);
    assert_eq!(p.spike_times(ClusterId(0)).len(), N_SPIKES / N_CLUSTERS + 1);
}

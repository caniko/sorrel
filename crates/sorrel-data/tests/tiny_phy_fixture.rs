//! `release/fixtures/tiny-phy/`: the phy directory `release/smoke-windows.ps1`
//! feeds to `sorrel qc`, `report` and the GUI on a real Windows runner, and
//! `release/fixtures/rescue-phy/`, the sparse variant `sorrel rescue` needs
//! (see [`RESCUE`]). Both are generated here, deterministically, so the
//! committed bytes can be checked:
//! `SORREL_WRITE_FIXTURE=1 cargo test -p sorrel-data --test tiny_phy_fixture`
//! rewrites them after an intended change.

use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider};
use sorrel_io::{ClusterId, DataProvider};
use std::path::{Path, PathBuf};

const N_CH: usize = 4;
const N_CLUSTERS: usize = 3;

/// One fixture's size. `rescue` adds `amplitudes.npy`, `spike_templates.npy`
/// and `templates.npy`, which `sorrel rescue` reads.
struct Shape {
    dir: &'static str,
    n_samples: usize,
    n_spikes: usize,
    rescue: bool,
}
const TINY: Shape = Shape {
    dir: "tiny-phy",
    n_samples: 45_000, // 1.5 s at 30 kHz
    n_spikes: 2000,
    rescue: false,
};
/// Rescue draws noise windows (261 samples) free of the unit's own spikes
/// on its channel, which `TINY`'s one spike per ~67 samples never leaves.
const RESCUE: Shape = Shape {
    dir: "rescue-phy",
    n_samples: 20_000,
    n_spikes: 120,
    rescue: true,
};

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
    npy_shape(descr, data, &format!("{n},"))
}

/// `shape` is the tuple's inside, e.g. `"3, 82, 4"`.
fn npy_shape(descr: &str, data: &[u8], shape: &str) -> Vec<u8> {
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape}), }}");
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
fn files(sh: &Shape) -> Vec<(&'static str, Vec<u8>)> {
    let (n_samples, n_spikes) = (sh.n_samples, sh.n_spikes);
    let mut rng = Rng(0x5eed_5044_1234_abcd);
    let mut spikes: Vec<(u64, u32)> = Vec::with_capacity(n_spikes);
    for k in 0..N_CLUSTERS {
        let n = n_spikes / N_CLUSTERS + usize::from(k < n_spikes % N_CLUSTERS);
        let period = (n_samples - 120) / n;
        for i in 0..n {
            let jitter = rng.next() as usize % (period / 2);
            spikes.push(((60 + i * period + jitter) as u64, k as u32));
        }
    }
    spikes.sort();

    let mut dat: Vec<i16> = (0..n_samples * N_CH)
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
    let mut out = vec![
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
    ];
    if sh.rescue {
        // Template k = unit k, with unit k's waveform on channel k only.
        const N_T: usize = 82;
        let amp = |k: usize| 250.0 + 100.0 * k as f32;
        let mut tpl = vec![0f32; N_CLUSTERS * N_T * N_CH];
        for k in 0..N_CLUSTERS {
            for j in 0..24usize {
                let x = (j as f32 - 8.0) / 4.0;
                tpl[(k * N_T + 20 + j) * N_CH + k] =
                    -amp(k) * (-x * x).exp() + 0.35 * amp(k) * (-(x - 2.0) * (x - 2.0)).exp();
            }
        }
        let f32s =
            |v: &mut dyn Iterator<Item = f32>| v.flat_map(f32::to_le_bytes).collect::<Vec<u8>>();
        let amps = f32s(&mut spikes.iter().map(|s| amp(s.1 as usize)));
        out.extend([
            ("spike_templates.npy", npy("<u4", &clusters, spikes.len())),
            ("amplitudes.npy", npy("<f4", &amps, spikes.len())),
            (
                "templates.npy",
                npy_shape(
                    "<f4",
                    &f32s(&mut tpl.iter().copied()),
                    &format!("{N_CLUSTERS}, {N_T}, {N_CH}"),
                ),
            ),
        ]);
    }
    out
}

fn committed_dir(sh: &Shape) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../release/fixtures")
        .join(sh.dir)
}

#[test]
fn committed_fixture_matches_generator_and_opens() {
    for sh in [TINY, RESCUE] {
        check(&sh);
    }
}

fn check(sh: &Shape) {
    let dir = committed_dir(sh);
    if std::env::var_os("SORREL_WRITE_FIXTURE").is_some() {
        std::fs::create_dir_all(&dir).unwrap();
        for (name, bytes) in files(sh) {
            std::fs::write(dir.join(name), bytes).unwrap();
        }
    }
    // The Nix source filter drops release/, so the nix checks have no copy.
    if !dir.exists() {
        eprintln!("skipping: {} not in this source tree", dir.display());
        return;
    }
    for (name, bytes) in files(sh) {
        assert_eq!(
            std::fs::read(dir.join(name)).unwrap(),
            bytes,
            "{}/{name} drifted; rerun with SORREL_WRITE_FIXTURE=1",
            sh.dir
        );
    }
    let p = KilosortProvider::open(&dir, KilosortOpenParams::default()).unwrap();
    assert_eq!(p.n_clusters() as usize, N_CLUSTERS);
    assert_eq!(
        p.spike_times(ClusterId(0)).len(),
        sh.n_spikes / N_CLUSTERS + usize::from(!sh.n_spikes.is_multiple_of(N_CLUSTERS))
    );
}

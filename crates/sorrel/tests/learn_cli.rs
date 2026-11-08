//! `sorrel dev bench ... learn-dataset` and `sorrel dev learn cv|train` on the
//! real binary, over a synthetic two-session bench built the way the bench
//! expects: a sealed native map, per-session `input/` + `truth/` directories,
//! no originals touched.
//!
//! The fixture is generated here because the repo's `tiny-phy` has no
//! `spike_templates.npy`/PC features, so it cannot produce the NPY-row map the
//! truth alignment requires (a bench cannot be built from it).
#![cfg(feature = "dev")]

use std::path::Path;
use std::process::{Command, Output};

fn sorrel(user_dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sorrel"))
        .args(args)
        .env("SORREL_USER_DIR", user_dir)
        .env_remove("SORREL_DEV_DB")
        .env_remove("SORREL_DEV_DENY")
        .output()
        .unwrap()
}

fn ok(o: &Output) -> String {
    assert!(
        o.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8_lossy(&o.stdout).into_owned()
}

/// A small phy-style Kilosort output with a complete PC row map.
///
/// NPY row order is deliberately shuffled relative to time order, so the
/// extraction has to align truth through the provider's row map rather than
/// compare index spaces directly.
fn write_fixture(dir: &Path, seed: u64) {
    use sorrel_io::npy::{write_1d_u32_atomic, write_1d_u64_atomic, write_f32_atomic};
    std::fs::create_dir_all(dir).unwrap();
    let n = 400usize;
    let n_chans = 4usize;
    let n_samples = 100_000u64;
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as u32
    };
    // Interleaved clusters, times shuffled away from NPY row order.
    let clusters: Vec<u32> = (0..n).map(|i| ((i * 7 + 3) % 2) as u32).collect();
    let mut times: Vec<u64> = (0..n)
        .map(|i| 1000 + (i as u64 * 137 + seed) % (n_samples - 2000))
        .collect();
    for i in 0..n {
        let j = (i * 37 + 11) % n;
        times.swap(i, j);
    }
    let amps: Vec<f32> = (0..n)
        .map(|i| 0.9 + (i % 11) as f32 * 0.03 + (next() % 100) as f32 * 1e-4)
        .collect();
    // 3 PCs × 4 channels per spike; the two roots sit apart on PC0.
    let mut pc = Vec::with_capacity(n * 3 * n_chans);
    for c in &clusters {
        for pc_i in 0..3 {
            for ch in 0..n_chans {
                let centre = if *c == 0 { 2.0 } else { -2.0 };
                let noise = (next() % 1000) as f32 / 1000.0 - 0.5;
                pc.push(centre * (pc_i == 0) as i32 as f32 + noise * 0.5 + ch as f32 * 1e-3);
            }
        }
    }
    let mut templates = vec![0.0f32; 2 * 4];
    templates[..4].copy_from_slice(&[1.0, 0.5, 0.0, 0.0]);
    templates[4..].copy_from_slice(&[0.2, -1.0, 0.1, 0.0]);
    write_1d_u64_atomic(&dir.join("spike_times.npy"), &times).unwrap();
    write_1d_u32_atomic(&dir.join("spike_clusters.npy"), &clusters).unwrap();
    write_1d_u32_atomic(&dir.join("spike_templates.npy"), &clusters).unwrap();
    write_f32_atomic(&dir.join("amplitudes.npy"), &[n], &amps).unwrap();
    write_f32_atomic(&dir.join("pc_features.npy"), &[n, 3, n_chans], &pc).unwrap();
    write_u32_2d(
        &dir.join("pc_feature_ind.npy"),
        (2, n_chans),
        &[0, 1, 2, 3, 0, 1, 2, 3],
    );
    write_f32_atomic(&dir.join("templates.npy"), &[2, 1, n_chans], &templates).unwrap();
    write_f32_atomic(
        &dir.join("channel_positions.npy"),
        &[n_chans, 2],
        &[0.0, 0.0, 16.0, 0.0, 32.0, 0.0, 48.0, 0.0],
    )
    .unwrap();
    // 4 channels of int16. The provider derives n_samples from the file size.
    std::fs::write(
        dir.join("recording.dat"),
        vec![0u8; n_chans * 2 * n_samples as usize],
    )
    .unwrap();
    std::fs::write(
        dir.join("params.py"),
        format!(
            "dat_path = 'recording.dat'\nn_channels_dat = {n_chans}\ndtype = 'int16'\n\
             offset = 0\nsample_rate = 30000.\nhp_filtered = False\n"
        ),
    )
    .unwrap();
}

/// Minimal 2-D `u32` NPY writer for the one file `sorrel_io::npy` only reads
/// (`pc_feature_ind.npy`): the shared writer API is 1-D or `f32`.
fn write_u32_2d(path: &Path, shape: (usize, usize), data: &[u32]) {
    let mut dict = format!(
        "{{'descr': '<u4', 'fortran_order': False, 'shape': ({}, {}), }}",
        shape.0, shape.1
    );
    // Magic (6) + version (2) + length (2) + dict + newline is a multiple of 64.
    let pad = (64 - (10 + dict.len() + 1) % 64) % 64;
    dict.push_str(&" ".repeat(pad));
    dict.push('\n');
    let mut bytes = Vec::with_capacity(10 + dict.len() + data.len() * 4);
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    bytes.extend_from_slice(dict.as_bytes());
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

fn write_truth(dir: &Path, source: &Path) {
    use sorrel_io::npy::write_1d_u32_atomic;
    std::fs::create_dir_all(dir).unwrap();
    let clusters = sorrel_io::npy::read_1d_u32(&source.join("spike_clusters.npy")).unwrap();
    write_1d_u32_atomic(&dir.join("spike_clusters.npy"), &clusters).unwrap();
    std::fs::write(
        dir.join("cluster_group.tsv"),
        "cluster_id\tgroup\n0\tgood\n1\tnoise\n",
    )
    .unwrap();
}

/// Build `<bench>/{.sealed/map.json,sessions/<sid>/{input,truth}}` for two dev
/// sessions, with the input allowlist the bench's `copy_input` enforces.
fn write_bench(bench: &Path, scratch: &Path) -> Vec<String> {
    let mut sids = Vec::new();
    let mut sessions = serde_json::Map::new();
    for (i, sid) in ["sid-a", "sid-b"].into_iter().enumerate() {
        let source = scratch.join(format!("source-{sid}"));
        write_fixture(&source, 11 + i as u64);
        let home = bench.join("sessions").join(sid);
        let input = home.join("input");
        std::fs::create_dir_all(&input).unwrap();
        let mut keep = Vec::new();
        for e in std::fs::read_dir(&source).unwrap() {
            let e = e.unwrap();
            std::fs::copy(e.path(), input.join(e.file_name())).unwrap();
            keep.push(e.file_name().to_string_lossy().into_owned());
        }
        keep.sort();
        write_truth(&home.join("truth"), &source);
        sessions.insert(
            sid.to_owned(),
            serde_json::json!({
                "path": source,
                "split": "dev",
                "forced_dev": true,
                "pure_manual": true,
                "keep": keep,
                "fingerprint": format!("fingerprint-{sid}"),
            }),
        );
        sids.push(sid.to_owned());
    }
    std::fs::create_dir_all(bench.join(".sealed")).unwrap();
    std::fs::write(
        bench.join(".sealed/map.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "salt": "learn-test-salt",
            "sessions": sessions,
        }))
        .unwrap(),
    )
    .unwrap();
    sids
}

fn read_table(dir: &Path, name: &str) -> sorrel_data::learn::table::Table {
    sorrel_data::learn::table::Table::load(dir, name).unwrap()
}

#[test]
fn learn_dataset_cv_and_train_over_a_synthetic_bench() {
    let tmp = tempfile::tempdir().unwrap();
    let scratch = tmp.path().join("scratch");
    std::fs::create_dir_all(&scratch).unwrap();
    let bench = tmp.path().join("bench");
    let sids = write_bench(&bench, &scratch);
    let out = tmp.path().join("learn");
    let s = |p: &Path| p.to_str().unwrap().to_owned();

    // Two jobs: the parallel path. Results must match the one-job re-run.
    let stdout = ok(&sorrel(
        tmp.path(),
        &[
            "dev",
            "bench",
            "--bench",
            &s(&bench),
            "learn-dataset",
            "--out",
            &s(&out),
            "--jobs",
            "2",
        ],
    ));
    assert!(stdout.contains("2 dev session(s)"), "{stdout}");

    let roots = read_table(&out, "roots");
    assert_eq!(roots.sids, sids);
    assert_eq!(roots.n_rows(), 4, "two roots per session");
    assert!(roots.column("y_verdict").is_some());
    let verdicts: Vec<f32> = (0..roots.n_rows())
        .map(|r| roots.get(r, "y_verdict"))
        .collect();
    let mut sorted = verdicts.clone();
    sorted.sort_by(f32::total_cmp);
    assert_eq!(sorted, vec![0.0, 0.0, 1.0, 1.0], "{verdicts:?}");
    // The whole-root hypothesis matches the human unit exactly.
    assert!(
        (0..roots.n_rows())
            .map(|r| roots.get(r, "y_whole_jaccard"))
            .fold(0.0f32, f32::max)
            >= 0.99
    );

    let candidates = read_table(&out, "candidates");
    assert_eq!(candidates.sids, sids);
    assert!(candidates.n_rows() >= 4, "whole root per root at least");
    assert!(
        (0..candidates.n_rows())
            .map(|r| candidates.get(r, "y_jaccard"))
            .fold(0.0f32, f32::max)
            >= 0.99
    );
    assert!(candidates.codes.contains_key("f_method"));

    let units = read_table(&out, "units");
    assert_eq!(units.n_rows(), 4, "two human units per session, all labels");
    let labels: Vec<f32> = (0..units.n_rows())
        .map(|r| units.get(r, "y_label"))
        .collect();
    assert_eq!(labels.iter().filter(|v| **v == 2.0).count(), 2);
    assert_eq!(labels.iter().filter(|v| **v == 0.0).count(), 2);

    let spikes = read_table(&out, "spikes");
    assert_eq!(spikes.n_rows(), 800, "every spike of both roots");
    assert_eq!(spikes.sids, sids);
    let pairs = read_table(&out, "pairs");
    assert_eq!(pairs.n_rows(), 2, "one similar root pair per session");
    assert!(pairs.column("y_merge").is_some());

    // Every table is row-sorted by (session, ids) whatever the job count.
    for r in 1..roots.n_rows() {
        let (a, b) = (roots.get(r - 1, "sid_idx"), roots.get(r, "sid_idx"));
        assert!(a < b || (a == b && roots.get(r - 1, "root") < roots.get(r, "root")));
    }

    // Resumable: a second run finds every shard and rewrites the same bytes.
    let before = std::fs::read(out.join("roots.f32")).unwrap();
    let again = ok(&sorrel(
        tmp.path(),
        &[
            "dev",
            "bench",
            "--bench",
            &s(&bench),
            "learn-dataset",
            "--out",
            &s(&out),
            "--jobs",
            "1",
        ],
    ));
    assert!(again.contains("already extracted"), "{again}");
    assert_eq!(before, std::fs::read(out.join("roots.f32")).unwrap());

    // And a cold one-job run produces the same bytes as the two-job run.
    std::fs::remove_dir_all(out.join("shards")).unwrap();
    ok(&sorrel(
        tmp.path(),
        &[
            "dev",
            "bench",
            "--bench",
            &s(&bench),
            "learn-dataset",
            "--out",
            &s(&out),
            "--jobs",
            "1",
        ],
    ));
    assert_eq!(before, std::fs::read(out.join("roots.f32")).unwrap());

    // Tiny data: CV plumbing only, not a performance claim.
    let cv = ok(&sorrel(
        tmp.path(),
        &[
            "dev",
            "learn",
            "cv",
            "--data",
            &s(&out),
            "--table",
            "roots",
            "--target",
            "y_verdict",
            "--objective",
            "softmax:4",
            "--folds",
            "2",
        ],
    ));
    let report: serde_json::Value = serde_json::from_str(&cv).unwrap();
    assert_eq!(report["n_rows"], 4);
    assert_eq!(report["per_fold"].as_array().unwrap().len(), 2);
    assert_eq!(report["pooled"]["kind"], "classification");

    // A regression target with a joined root context must also go through.
    let cv = ok(&sorrel(
        tmp.path(),
        &[
            "dev",
            "learn",
            "cv",
            "--data",
            &s(&out),
            "--table",
            "candidates",
            "--target",
            "y_jaccard",
            "--objective",
            "regression",
            "--folds",
            "2",
            "--join-roots",
        ],
    ));
    let report: serde_json::Value = serde_json::from_str(&cv).unwrap();
    assert_eq!(report["pooled"]["kind"], "regression");
    assert_eq!(report["ranking"]["oracle_units"], 2);
    assert!(report["pooled"].get("r2").is_some());

    // Train writes a portable bundle; a second head extends it.
    let bundle = tmp.path().join("curator.sorrel-model");
    let trained = ok(&sorrel(
        tmp.path(),
        &[
            "dev",
            "learn",
            "train",
            "--data",
            &s(&out),
            "--table",
            "roots",
            "--target",
            "y_verdict",
            "--objective",
            "softmax:4",
            "--folds",
            "2",
            "--out",
            &s(&bundle),
            "--head",
            "verdict",
        ],
    ));
    let json: serde_json::Value = serde_json::from_str(&trained).unwrap();
    assert_eq!(json["head"], "verdict");
    assert!(json["metrics"]["cv_accuracy"].as_f64().is_some());
    let loaded = sorrel_data::learn::bundle::CuratorBundle::load(&bundle).unwrap();
    assert!(loaded.heads.contains_key("verdict"));
    assert!(loaded
        .provenance
        .training_sessions
        .contains(&sids[0].clone()));
    assert_eq!(
        loaded.provenance.extractor_version,
        sorrel_data::learn::EXTRACTOR_VERSION
    );

    ok(&sorrel(
        tmp.path(),
        &[
            "dev",
            "learn",
            "train",
            "--data",
            &s(&out),
            "--table",
            "candidates",
            "--target",
            "y_jaccard",
            "--objective",
            "regression",
            "--folds",
            "2",
            "--join-roots",
            "--out",
            &s(&bundle),
            "--head",
            "jaccard",
        ],
    ));
    let loaded = sorrel_data::learn::bundle::CuratorBundle::load(&bundle).unwrap();
    assert_eq!(loaded.heads.len(), 2);
    assert!(loaded.heads.contains_key("verdict"));
    assert!(loaded.heads.contains_key("jaccard"));
    assert_eq!(loaded.provenance.n_rows["verdict"], 4);
}

#[test]
fn learn_cv_refuses_a_table_from_another_extractor_or_a_missing_target() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("ds");
    let mut t = sorrel_data::learn::table::Table::new(
        "some-other-version",
        vec!["sid_idx".into(), "f_a".into(), "y_verdict".into()],
    );
    t.sids = vec!["s".into()];
    t.push_row(&[0.0, 1.0, 1.0]).unwrap();
    t.save(&dir, "roots").unwrap();
    let o = sorrel(
        tmp.path(),
        &[
            "dev",
            "learn",
            "cv",
            "--data",
            dir.to_str().unwrap(),
            "--table",
            "roots",
            "--target",
            "y_verdict",
            "--objective",
            "logistic",
            "--folds",
            "2",
        ],
    );
    assert!(!o.status.success());
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("re-extract"),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );

    let mut t = sorrel_data::learn::table::Table::new(
        sorrel_data::learn::EXTRACTOR_VERSION,
        vec!["sid_idx".into(), "f_a".into()],
    );
    t.sids = vec!["s".into()];
    t.push_row(&[0.0, 1.0]).unwrap();
    t.save(&dir, "roots").unwrap();
    let o = sorrel(
        tmp.path(),
        &[
            "dev",
            "learn",
            "cv",
            "--data",
            dir.to_str().unwrap(),
            "--table",
            "roots",
            "--target",
            "y_verdict",
            "--objective",
            "logistic",
        ],
    );
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("no target column"));
}

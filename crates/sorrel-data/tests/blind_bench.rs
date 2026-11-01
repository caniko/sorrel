//! The double-blind bench pair: `write_blind_input` / `provenance` / `diff_dirs`
//! on a session with a known set of human edits.

use sorrel_data::command::Method;
use sorrel_data::diff::{diff_dirs, SpikeDiffOpts};
use sorrel_data::{
    baseline_hash, open_phy, provenance, save_and_seal, save_to_phy, seed_phy, write_blind_input,
    write_phy_curated_pair, Actor, CurationCommand, JournalOpts, PhyLabelOp, Session,
};
use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider, PhyLabel};
use sorrel_io::npy::read_1d_u32;
use sorrel_io::ClusterId;
use std::io::Write;
use std::path::Path;

fn write_npy(path: &Path, descr: &str, data: &[u8], n: usize) {
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({n},), }}");
    let pad = (64 - (6 + 2 + 2 + dict.len() + 1) % 64) % 64;
    let mut header = dict.into_bytes();
    header.resize(header.len() + pad, b' ');
    header.push(b'\n');
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"\x93NUMPY\x01\x00").unwrap();
    f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
    f.write_all(&header).unwrap();
    f.write_all(data).unwrap();
}

/// 5 root clusters of 20 spikes each, plus the sorter's own metrics file.
fn write_sort(root: &Path) {
    let n = 100;
    let times: Vec<u8> = (0..n as u64).flat_map(|t| (t * 10).to_le_bytes()).collect();
    let clusters: Vec<u8> = (0..n as u32).flat_map(|i| (i / 20).to_le_bytes()).collect();
    write_npy(&root.join("spike_times.npy"), "<u8", &times, n);
    write_npy(&root.join("spike_clusters.npy"), "<u4", &clusters, n);
    std::fs::write(root.join("recording.dat"), vec![0u8; 4 * 8 * 2]).unwrap();
    std::fs::write(
        root.join("params.py"),
        "n_channels_dat = 4\nsample_rate = 1000.\ndtype = 'int16'\n",
    )
    .unwrap();
    std::fs::write(
        root.join("cluster_KSLabel.tsv"),
        "cluster_id\tKSLabel\n0\tunsorted\n",
    )
    .unwrap();
}

fn label_str(l: &PhyLabel) -> &'static str {
    l.as_str()
}

fn open(root: &Path) -> Session<KilosortProvider> {
    let mut s = open_phy(root, KilosortOpenParams::default(), &JournalOpts::default()).unwrap();
    seed_phy(&mut s).unwrap();
    s
}

fn misses(d: &sorrel_data::diff::SpikeDiff) -> Vec<(&'static str, Vec<u32>)> {
    d.misses
        .iter()
        .map(|m| (m.category, m.roots.clone()))
        .collect()
}

#[test]
fn blind_pair_scores_exactly_the_injected_edits() {
    let tmp = tempfile::tempdir().unwrap();
    let (root, input, truth) = (
        tmp.path().join("sort"),
        tmp.path().join("in"),
        tmp.path().join("truth"),
    );
    std::fs::create_dir(&root).unwrap();
    write_sort(&root);
    let baseline = baseline_hash(&std::fs::read(root.join("spike_clusters.npy")).unwrap());

    let mut s = open(&root);
    for cmd in [
        // root 1 into 0: missed_merge [0, 1]
        CurationCommand::Merge {
            sources: vec![ClusterId(1)],
            target: ClusterId(0),
        },
        // half of root 2 out: missed_split [2]
        CurationCommand::Split {
            cluster: ClusterId(2),
            spike_idx: (0..10).collect(),
            new_cluster: ClusterId(u32::MAX),
            by: Method::Manual,
        },
        // label [3], exclusion [4]
        CurationCommand::Relabel {
            cluster: ClusterId(3),
            op: PhyLabelOp::SetMua,
        },
        CurationCommand::Relabel {
            cluster: ClusterId(4),
            op: PhyLabelOp::SetNoise,
        },
    ] {
        s.dispatch(cmd).unwrap();
    }
    save_and_seal(&mut s, &root, label_str).unwrap();
    // Curation leftovers a blind input must not carry.
    for d in ["sorrel_auto", "mcp", ".phy", ".sorrel"] {
        std::fs::create_dir_all(root.join(d)).unwrap();
        std::fs::write(root.join(d).join("x"), "x").unwrap();
    }
    assert!(root.join("sorrel.journal").exists() && root.join("cluster_group.tsv").exists());

    // Blind input: opens as a fresh phy directory sealed to the same baseline.
    let files = write_blind_input(
        &root,
        None,
        &input,
        "spike_clusters.npy",
        s.journal().baseline(),
    )
    .unwrap();
    let mut names: Vec<_> = std::fs::read_dir(&input)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "cluster_KSLabel.tsv",
            "params.py",
            "recording.dat",
            "spike_clusters.npy",
            "spike_times.npy"
        ],
        "{files:?}"
    );
    assert!(std::fs::symlink_metadata(input.join("recording.dat"))
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(
        baseline_hash(&std::fs::read(input.join("spike_clusters.npy")).unwrap()),
        baseline
    );
    assert_eq!(open(&input).journal().baseline(), baseline);
    std::fs::remove_file(input.join("sorrel.journal")).unwrap();
    assert!(
        write_blind_input(&root, None, &input, "spike_clusters.npy", baseline).is_err(),
        "non-empty output refused"
    );

    // Truth = head, and a purely manual one.
    std::fs::create_dir(&truth).unwrap();
    save_to_phy(&s, &truth, label_str).unwrap();
    let head: Vec<u32> = s.spike_clusters_global().iter().map(|c| c.0).collect();
    assert_eq!(
        read_1d_u32(&truth.join("spike_clusters.npy")).unwrap(),
        head
    );
    let p = provenance(&s).unwrap();
    assert!(p.pure_manual && !p.anchored && p.sealed, "{p:?}");
    assert_eq!(
        (p.by_actor["human"], p.by_method["manual"], p.n_applied),
        (4, 4, 4)
    );

    let opts = SpikeDiffOpts::default();
    let same = diff_dirs(&truth, &truth, &input, &opts).unwrap();
    assert!(
        same.misses.is_empty() && same.units.f1 == 1.0 && same.ari == 1.0,
        "{same:?}"
    );
    let d = diff_dirs(&truth, &input, &input, &opts).unwrap();
    assert_eq!(
        misses(&d),
        [
            ("missed_merge", vec![0, 1]),
            ("missed_split", vec![2]),
            ("label", vec![3]),
            ("exclusion", vec![4]),
        ]
    );
    // Disputing the split root drops it from scoring.
    let opts = SpikeDiffOpts {
        disputed: [2].into(),
        ..opts
    };
    assert!(!misses(&diff_dirs(&truth, &input, &input, &opts).unwrap())
        .iter()
        .any(|m| m.1 == [2]));

    // An agent edit anywhere makes the session anchored; off the applied path it stays pure.
    s.dispatch_as(
        Actor::Agent {
            client: "bench".into(),
        },
        CurationCommand::Relabel {
            cluster: ClusterId(3),
            op: PhyLabelOp::SetGood,
        },
    )
    .unwrap();
    let p = provenance(&s).unwrap();
    assert!(
        !p.pure_manual && p.anchored && !p.sealed && p.by_actor["agent"] == 1,
        "{p:?}"
    );
    s.dispatch(CurationCommand::Undo).unwrap();
    let p = provenance(&s).unwrap();
    assert!(p.pure_manual && p.anchored && p.sealed, "{p:?}");
}

/// A sort curated in phy on another machine: `spike_templates.npy` is the
/// sorter's clustering, `spike_clusters.npy` the human's (0+1 merged into 5,
/// root 2 split into 2 and 6), `phy.log` lists the edits and params.py points
/// at a dat_path that only existed there.
#[test]
fn phy_curated_pair_takes_the_sort_from_spike_templates() {
    let tmp = tempfile::tempdir().unwrap();
    let (root, input, truth) = (
        tmp.path().join("sort"),
        tmp.path().join("in"),
        tmp.path().join("truth"),
    );
    std::fs::create_dir(&root).unwrap();
    write_sort(&root);
    std::fs::rename(
        root.join("spike_clusters.npy"),
        root.join("spike_templates.npy"),
    )
    .unwrap();
    let curated: Vec<u32> = (0..100u32)
        .map(|i| match i / 20 {
            0 | 1 => 5,
            2 if i % 2 == 0 => 6,
            c => c,
        })
        .collect();
    let bytes: Vec<u8> = curated.iter().flat_map(|c| c.to_le_bytes()).collect();
    write_npy(&root.join("spike_clusters.npy"), "<u4", &bytes, 100);
    std::fs::write(
        root.join("cluster_group.tsv"),
        "cluster_id\tgroup\n4\tnoise\n",
    )
    .unwrap();
    std::fs::write(root.join("phy.log"), "Merge clusters 0, 1 to 5.\n").unwrap();
    let mut params = std::fs::read_to_string(root.join("params.py")).unwrap();
    params.push_str("dat_path = r'/elsewhere/recording.dat'\n");
    std::fs::write(root.join("params.py"), params).unwrap();

    write_phy_curated_pair(&root, None, &input, &truth).unwrap();
    assert_eq!(
        std::fs::read(input.join("spike_clusters.npy")).unwrap(),
        std::fs::read(root.join("spike_templates.npy")).unwrap()
    );
    for name in ["phy.log", "cluster_group.tsv"] {
        assert!(!input.join(name).exists(), "{name} leaked");
    }
    // The relocated recording opens, and nothing is written beside the source.
    let s = open(&input);
    assert_eq!(s.n_clusters(), 5);
    assert!(!root.join("sorrel.journal").exists());

    let d = diff_dirs(&truth, &input, &input, &SpikeDiffOpts::default()).unwrap();
    assert_eq!(
        misses(&d),
        [
            ("missed_merge", vec![0, 1]),
            ("missed_split", vec![2]),
            ("exclusion", vec![4]),
        ]
    );
}

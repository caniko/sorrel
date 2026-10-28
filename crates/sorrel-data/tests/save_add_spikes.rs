//! Saving a phy directory: after `AddSpikes` every per-spike array is
//! rewritten aligned and time-sorted, the root versions are captured into
//! `sorrel_original/`, unknown per-spike files block the save, a crash at any
//! point of the save reopens cleanly, and an external edit fails closed.

use sorrel_data::command::SpikeBatch;
use sorrel_data::{
    baseline_hash, open_phy, save_and_seal, seed_phy, CurationCommand, JournalOpts, PhyLabelOp,
    Session,
};
use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider, PhyLabel};
use sorrel_io::npy::{read_1d_f32, read_1d_u32, read_npy_f32_flat, read_npy_u32_flat};
use sorrel_io::{ClusterId, DataProvider};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

fn write_npy(path: &Path, descr: &str, shape: &[usize], data: &[u8]) {
    let shape = match shape {
        [n] => format!("({n},)"),
        s => format!(
            "({})",
            s.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
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

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// 6 spikes (t = 10,30,50,100,150,200; clusters 0,0,1,2,0,1) with every
/// per-spike array type. Row `i` has amplitude `1+i`, PCs `[10i, 10i+1]`,
/// template features `[i, i+.5]`, `spike_depth = 100+i`.
fn write_fixture(root: &Path) {
    let times: Vec<u8> = [10u64, 30, 50, 100, 150, 200]
        .iter()
        .flat_map(|t| t.to_le_bytes())
        .collect();
    write_npy(&root.join("spike_times.npy"), "<u8", &[6], &times);
    write_npy(
        &root.join("spike_clusters.npy"),
        "<u4",
        &[6],
        &u32s(&[0, 0, 1, 2, 0, 1]),
    );
    write_npy(
        &root.join("spike_templates.npy"),
        "<u4",
        &[6],
        &u32s(&[0, 0, 1, 1, 0, 1]),
    );
    write_npy(
        &root.join("amplitudes.npy"),
        "<f4",
        &[6],
        &f32s(&[1., 2., 3., 4., 5., 6.]),
    );
    let pc: Vec<f32> = (0..6)
        .flat_map(|i| [10. * i as f32, 10. * i as f32 + 1.])
        .collect();
    write_npy(&root.join("pc_features.npy"), "<f4", &[6, 1, 2], &f32s(&pc));
    write_npy(
        &root.join("pc_feature_ind.npy"),
        "<u4",
        &[2, 2],
        &u32s(&[0, 1, 1, 0]),
    );
    let tf: Vec<f32> = (0..6).flat_map(|i| [i as f32, i as f32 + 0.5]).collect();
    write_npy(
        &root.join("template_features.npy"),
        "<f4",
        &[6, 2],
        &f32s(&tf),
    );
    write_npy(
        &root.join("template_feature_ind.npy"),
        "<u4",
        &[2, 2],
        &u32s(&[0, 1, 1, 0]),
    );
    let depth: Vec<f32> = (0..6).map(|i| 100. + i as f32).collect();
    write_npy(&root.join("spike_depth.npy"), "<f4", &[6], &f32s(&depth));
    write_npy(
        &root.join("spike_reorder.npy"),
        "<u4",
        &[6],
        &u32s(&[1, 2, 3, 4, 5, 0]),
    );
    std::fs::create_dir(root.join(".phy")).unwrap();
    std::fs::write(root.join("recording.dat"), vec![0u8; 4 * 8 * 2]).unwrap();
    std::fs::write(
        root.join("params.py"),
        "n_channels_dat = 4\nsample_rate = 1000.\ndtype = 'int16'\n",
    )
    .unwrap();
}

fn open(root: &Path) -> Session<KilosortProvider> {
    let opts = JournalOpts {
        path: Some(root.join("j.journal")),
        discard: false,
    };
    let mut s = open_phy(root, KilosortOpenParams::default(), &opts).unwrap();
    seed_phy(&mut s).unwrap();
    s
}

/// New spikes at t=40 (cluster 1) and t=100 (cluster 2, ties after the
/// original), landing at global rows 2 and 5 of 8.
fn add_two(s: &mut Session<KilosortProvider>) {
    s.dispatch(CurationCommand::AddSpikes(SpikeBatch {
        times: vec![40, 100],
        clusters: vec![ClusterId(1), ClusterId(2)],
        templates: vec![1, 1],
        amplitudes: vec![9., 8.],
        pc_features: vec![900., 901., 910., 911.],
        template_features: vec![7., 7.5, 8., 8.5],
        attributes: vec![("depth".into(), 1, vec![500., 501.])],
    }))
    .unwrap();
}

fn label_str(l: &PhyLabel) -> &'static str {
    l.as_str()
}

/// Every top-level file's bytes.
fn snapshot(root: &Path) -> BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(root)
        .unwrap()
        .flatten()
        .filter(|e| e.file_type().unwrap().is_file())
        .map(|e| {
            (
                e.file_name().into_string().unwrap(),
                std::fs::read(e.path()).unwrap(),
            )
        })
        .collect()
}

#[test]
fn save_and_reseal_rewrites_every_array_aligned() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    let before = snapshot(root);
    let mut s = open(root);
    add_two(&mut s);
    save_and_seal(&mut s, root, label_str).unwrap();

    let u32f = |n: &str| read_1d_u32(&root.join(n)).unwrap();
    assert_eq!(
        u32f("spike_times.npy"),
        [10, 30, 40, 50, 100, 100, 150, 200]
    );
    assert_eq!(u32f("spike_clusters.npy"), [0, 0, 1, 1, 2, 2, 0, 1]);
    assert_eq!(u32f("spike_templates.npy"), [0, 0, 1, 1, 1, 1, 0, 1]);
    assert_eq!(u32f("spike_sorrel_rescued.npy"), [0, 0, 1, 0, 0, 1, 0, 0]);
    // Old values were rows (1,2,3,4,5,0); they now name the shown spike's new index.
    assert_eq!(u32f("spike_reorder.npy"), [1, 3, 2, 4, 6, 5, 7, 0]);
    let amps = read_1d_f32(&root.join("amplitudes.npy")).unwrap();
    assert_eq!(amps, [1., 2., 9., 3., 4., 8., 5., 6.]);
    let depth = read_1d_f32(&root.join("spike_depth.npy")).unwrap();
    assert_eq!(depth, [100., 101., 500., 102., 103., 501., 104., 105.]);
    let (pc, shape) = read_npy_f32_flat(&root.join("pc_features.npy")).unwrap();
    assert_eq!(shape, [8, 1, 2]);
    assert_eq!(&pc[4..6], [900., 901.]);
    assert_eq!(&pc[10..12], [910., 911.]);
    assert_eq!(&pc[6..8], [20., 21.]); // original row 2 = g 3
    let (tf, shape) = read_npy_f32_flat(&root.join("template_features.npy")).unwrap();
    assert_eq!(shape, [8, 2]);
    assert_eq!(&tf[4..6], [7., 7.5]);
    assert_eq!(&tf[6..8], [2., 2.5]);
    assert!(read_npy_u32_flat(&root.join("pc_feature_ind.npy")).is_ok());

    // The root copies are the untouched originals; the stale phy cache moved aside.
    for n in [
        "spike_times.npy",
        "spike_clusters.npy",
        "amplitudes.npy",
        "pc_features.npy",
    ] {
        assert_eq!(
            &std::fs::read(root.join("sorrel_original").join(n)).unwrap(),
            &before[n]
        );
    }
    assert!(!root
        .join("sorrel_original/spike_sorrel_rescued.npy")
        .exists());
    assert!(!root.join(".phy").exists() && root.join(".phy.sorrel-stale").exists());
    assert!(!root.join("sorrel_rewrite.pending").exists());
    assert!(!root.join("spike_times.npy.tmp").exists());

    // The provider reads the root; the journal replays the rescue onto it.
    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    let times = |c| {
        p.spike_times(ClusterId(c))
            .iter()
            .map(|t| t.0)
            .collect::<Vec<_>>()
    };
    assert_eq!(times(1), [50, 200]);
    assert_eq!(times(2), [100]);
    drop(s);
    let mut s2 = open(root);
    assert!(s2.is_saved());
    assert_eq!(s2.cluster_index().n_spikes(), 8);
    // Undoing the rescue and saving again writes the original arrays back
    // (from the root copies) instead of misaligning them.
    s2.dispatch(CurationCommand::Undo).unwrap();
    save_and_seal(&mut s2, root, label_str).unwrap();
    assert_eq!(u32f("spike_times.npy"), [10, 30, 50, 100, 150, 200]);
    assert_eq!(u32f("spike_clusters.npy"), [0, 0, 1, 2, 0, 1]);
    assert_eq!(u32f("spike_sorrel_rescued.npy"), [0; 6]);
    drop(s2);
    assert_eq!(open(root).cluster_index().n_spikes(), 6);
}

#[test]
fn unknown_per_spike_array_blocks_the_save() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    write_npy(&root.join("foo.npy"), "<f4", &[6], &f32s(&[0.; 6]));
    let before = snapshot(root);
    let mut s = open(root);
    add_two(&mut s);
    let e = save_and_seal(&mut s, root, label_str).unwrap_err();
    assert!(e.to_string().contains("foo.npy"), "{e}");
    let after = snapshot(root);
    // Only the journal (opened by the session) may differ.
    for (k, v) in &before {
        assert_eq!(after.get(k), Some(v), "{k} changed");
    }
    assert!(after
        .keys()
        .all(|k| before.contains_key(k) || k.starts_with("j.journal")));
    assert!(!root.join("sorrel_original").exists());
}

#[test]
fn interrupted_rewrite_rolls_forward_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    write_npy(
        &root.join("spike_clusters.npy.tmp"),
        "<u4",
        &[6],
        &u32s(&[2; 6]),
    );
    // The second pair's tmp was already renamed before the crash.
    std::fs::write(
        root.join("sorrel_rewrite.pending"),
        "spike_clusters.npy.tmp\tspike_clusters.npy\nspike_times.npy.tmp\tspike_times.npy\n",
    )
    .unwrap();
    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    assert!(!root.join("sorrel_rewrite.pending").exists());
    assert!(!root.join("spike_clusters.npy.tmp").exists());
    assert_eq!(
        read_1d_u32(&root.join("spike_clusters.npy")).unwrap(),
        [2; 6]
    );
    assert_eq!(p.spike_times(ClusterId(2)).len(), 6);
}

#[cfg(feature = "phy-export")]
#[test]
fn plain_save_with_added_spikes_errors_and_export_rewrites() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    let before = std::fs::read(root.join("spike_clusters.npy")).unwrap();
    let mut s = open(root);
    add_two(&mut s);
    let e = sorrel_data::save_to_phy(&s, root, label_str).unwrap_err();
    assert!(e.to_string().contains("added spikes"), "{e}");
    assert_eq!(
        std::fs::read(root.join("spike_clusters.npy")).unwrap(),
        before
    );

    let out = tempfile::tempdir().unwrap();
    sorrel_data::export_phy(&s, out.path(), false, label_str).unwrap();
    let o = out.path();
    assert_eq!(read_1d_u32(&o.join("spike_times.npy")).unwrap().len(), 8);
    assert_eq!(
        read_1d_u32(&o.join("spike_clusters.npy")).unwrap(),
        [0, 0, 1, 1, 2, 2, 0, 1]
    );
    assert_eq!(
        read_npy_f32_flat(&o.join("pc_features.npy")).unwrap().1,
        [8, 1, 2]
    );
    assert!(!o.join("spike_times.npy.tmp").exists());
}

/// A `spike_<name>.npy` wider than two columns keeps every column, so the
/// rewrite carries them all instead of failing to encode short rows.
#[test]
fn wide_spike_attribute_survives_the_rewrite() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    let wide: Vec<f32> = (0..6)
        .flat_map(|i| [i as f32, 10. + i as f32, 20.])
        .collect();
    write_npy(&root.join("spike_wide.npy"), "<f4", &[6, 3], &f32s(&wide));
    let mut s = open(root);
    let ci = s.cluster_index();
    assert!(ci.spike_attribute_names().contains(&("wide", 3)));
    assert_eq!(ci.spike_attribute("wide", 1), Some(&[1., 11., 20.][..]));
    s.dispatch(CurationCommand::AddSpikes(SpikeBatch {
        times: vec![40, 100],
        clusters: vec![ClusterId(1), ClusterId(2)],
        templates: vec![1, 1],
        amplitudes: vec![9., 8.],
        pc_features: vec![900., 901., 910., 911.],
        template_features: vec![7., 7.5, 8., 8.5],
        attributes: vec![
            ("depth".into(), 1, vec![500., 501.]),
            ("wide".into(), 3, vec![-1., -2., -3., -4., -5., -6.]),
        ],
    }))
    .unwrap();
    save_and_seal(&mut s, root, label_str).unwrap();
    let (flat, shape) = read_npy_f32_flat(&root.join("spike_wide.npy")).unwrap();
    assert_eq!(shape, [8, 3]);
    assert_eq!(&flat[6..9], [-1., -2., -3.]); // added spike at g = 2
    assert_eq!(&flat[9..12], [2., 12., 20.]); // original row 2 now at g = 3
}

#[test]
fn cleared_custom_label_is_blanked_not_restored() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    let set = |s: &mut Session<KilosortProvider>, v: &str| {
        let cmd = s.set_label_command(ClusterId(1), "quality", v).unwrap();
        s.dispatch(cmd).unwrap();
    };
    let mut s = open(root);
    set(&mut s, "great");
    save_and_seal(&mut s, root, label_str).unwrap();
    drop(s);
    let mut s = open(root);
    assert_eq!(s.custom_label(ClusterId(1), "quality"), Some("great"));

    set(&mut s, "");
    save_and_seal(&mut s, root, label_str).unwrap();
    // The user's file is kept, header only.
    let tsv = std::fs::read_to_string(root.join("cluster_quality.tsv")).unwrap();
    assert_eq!(tsv, "cluster_id\tquality\n");
    drop(s);
    let s = open(root);
    assert_eq!(s.custom_label(ClusterId(1), "quality"), None);
    assert!(s.custom_label_keys().is_empty());
}

/// Where a save was interrupted (see `save_and_seal`'s steps).
#[derive(Clone, Copy, Debug)]
enum Crash {
    /// Tmps staged; nothing captured, nothing journaled.
    BeforeCapture,
    /// Root captured; no seal yet.
    BeforeSeal,
    /// Seal journaled; no manifest yet.
    BeforeManifest,
    /// Manifest published and the first `n` renames done.
    Renamed(usize),
}

/// Run a full add-spikes save, then rebuild the directory as a crash at
/// `at` would have left it.
fn crash(at: Crash) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    let mut s = open(root);
    add_two(&mut s);
    let cmd = s
        .set_label_command(ClusterId(1), "quality", "great")
        .unwrap();
    s.dispatch(cmd).unwrap();
    s.dispatch(CurationCommand::Relabel {
        cluster: ClusterId(2),
        op: PhyLabelOp::SetGood,
    })
    .unwrap();
    let pre = snapshot(root);
    save_and_seal(&mut s, root, label_str).unwrap();
    drop(s);
    let post = snapshot(root);

    let changed: Vec<&String> = post
        .iter()
        .filter(|(k, v)| pre.get(*k) != Some(*v) && k.as_str() != "j.journal")
        .map(|(k, _)| k)
        .collect();
    assert!(changed.iter().any(|k| *k == "spike_clusters.npy"));
    if matches!(at, Crash::BeforeCapture | Crash::BeforeSeal) {
        std::fs::write(root.join("j.journal"), &pre["j.journal"]).unwrap();
    }
    if matches!(at, Crash::BeforeCapture) {
        std::fs::remove_dir_all(root.join("sorrel_original")).unwrap();
    }
    let renamed = match at {
        Crash::Renamed(n) => n,
        _ => 0,
    };
    let mut manifest = String::new();
    for (i, name) in changed.iter().enumerate() {
        match pre.get(*name) {
            Some(old) => std::fs::write(root.join(name), old).unwrap(),
            None => std::fs::remove_file(root.join(name)).unwrap(),
        }
        if i < renamed {
            std::fs::write(root.join(name), &post[*name]).unwrap();
        } else {
            std::fs::write(root.join(format!("{name}.tmp")), &post[*name]).unwrap();
        }
        manifest.push_str(&format!("{name}.tmp\t{name}\n"));
    }
    if matches!(at, Crash::Renamed(_)) {
        std::fs::write(root.join("sorrel_rewrite.pending"), manifest).unwrap();
    }
    dir
}

/// The reopened session is the head with every edit, whatever the files.
fn assert_head_with_everything(s: &Session<KilosortProvider>) {
    assert_eq!(s.cluster_index().n_spikes(), 8);
    assert_eq!(s.custom_label(ClusterId(1), "quality"), Some("great"));
    assert_eq!(s.label(ClusterId(2)), Some(PhyLabel::Good));
    assert_eq!(s.history_len(), 3, "the journal keeps the history");
}

/// After the manifest, the next open finishes the save: the files are the
/// head's and match its seal.
#[test]
fn crash_after_the_manifest_completes_on_open() {
    for n in [0, 3, usize::MAX] {
        let dir = crash(Crash::Renamed(n));
        let root = dir.path();
        let s = open(root); // roll-forward, then the external-edit check
        assert_head_with_everything(&s);
        assert!(s.is_saved(), "renamed {n}");
        assert!(!root.join("sorrel_rewrite.pending").exists());
        assert!(snapshot(root).keys().all(|k| !k.ends_with(".tmp")));
        assert_eq!(read_1d_u32(&root.join("spike_times.npy")).unwrap().len(), 8);
    }
}

/// Before the manifest, the old files stay on top; they are the root (or an
/// older seal), so the directory opens at the head, unsaved, and a second
/// save goes through.
#[test]
fn crash_before_the_manifest_reopens_unsaved_and_saves_again() {
    for at in [
        Crash::BeforeCapture,
        Crash::BeforeSeal,
        Crash::BeforeManifest,
    ] {
        let dir = crash(at);
        let root = dir.path();
        let mut s = open(root);
        assert_head_with_everything(&s);
        assert!(!s.is_saved(), "{at:?}");
        assert_eq!(
            read_1d_u32(&root.join("spike_times.npy")).unwrap().len(),
            6,
            "{at:?}: old files on top"
        );
        save_and_seal(&mut s, root, label_str).unwrap();
        drop(s);
        let s = open(root);
        assert_head_with_everything(&s);
        assert!(s.is_saved(), "{at:?}");
    }
}

/// A phy edit after a sorrel save (a label in `cluster_group.tsv`) would be
/// overwritten by the next save, so the open refuses; `--discard-journal`
/// sets the journal and the snapshot aside and starts from the files.
#[test]
fn external_edit_after_a_save_fails_closed_until_discarded() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    let mut s = open(root);
    s.dispatch(CurationCommand::Relabel {
        cluster: ClusterId(0),
        op: PhyLabelOp::SetGood,
    })
    .unwrap();
    save_and_seal(&mut s, root, label_str).unwrap();
    drop(s);
    std::fs::write(
        root.join("cluster_group.tsv"),
        "cluster_id\tgroup\n0\tgood\n1\tnoise\n",
    )
    .unwrap();

    let opts = |discard| JournalOpts {
        path: Some(root.join("j.journal")),
        discard,
    };
    let e = open_phy(root, KilosortOpenParams::default(), &opts(false))
        .err()
        .expect("refused");
    assert!(e.to_string().contains("--discard-journal"), "{e}");

    let mut s = open_phy(root, KilosortOpenParams::default(), &opts(true)).unwrap();
    seed_phy(&mut s).unwrap();
    assert_eq!(s.history_len(), 0);
    assert_eq!(
        s.label(ClusterId(0)),
        Some(PhyLabel::Good),
        "saved by sorrel"
    );
    assert_eq!(
        s.label(ClusterId(1)),
        Some(PhyLabel::Noise),
        "edited in phy"
    );
    let names: Vec<String> = std::fs::read_dir(root)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().into_string().unwrap())
        .collect();
    let aside = |p: &str| names.iter().any(|n| n.starts_with(p) && n.len() > p.len());
    assert!(
        aside("j.journal.") && aside("sorrel_original."),
        "{names:?}"
    );
    assert!(!root.join("sorrel_original").exists());
}

/// A v2 journal (with an undo) upgrades on open: it replays onto the files
/// it was sealed against, and a manifest-less `sorrel_original/` from a
/// pre-v3 rescue is moved aside rather than read as the root.
#[test]
fn v2_journal_and_legacy_backup_upgrade_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    std::fs::create_dir(root.join("sorrel_original")).unwrap();
    std::fs::write(root.join("sorrel_original/spike_times.npy"), b"old").unwrap();
    let mut bytes = b"SORREL\x00\x02".to_vec();
    bytes.extend(2u32.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    let sc = std::fs::read(root.join("spike_clusters.npy")).unwrap();
    bytes.extend(baseline_hash(&sc).to_le_bytes());
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    for c in [
        CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: PhyLabelOp::SetGood,
        },
        CurationCommand::Relabel {
            cluster: ClusterId(1),
            op: PhyLabelOp::SetMua,
        },
        CurationCommand::Undo,
    ] {
        let rec = rmp_serde::to_vec(&c).unwrap();
        bytes.extend((rec.len() as u32).to_le_bytes());
        bytes.extend(rec);
    }
    std::fs::write(root.join("j.journal"), bytes).unwrap();

    let s = open(root);
    assert_eq!(s.label(ClusterId(0)), Some(PhyLabel::Good));
    assert_eq!(s.label(ClusterId(1)), Some(PhyLabel::Unsorted));
    assert_eq!((s.history_len(), s.redo_len(), s.n_nodes()), (1, 1, 2));
    assert!(!root.join("sorrel_original").exists());
    assert_eq!(
        &std::fs::read(root.join("j.journal")).unwrap()[..8],
        b"SORREL\x00\x03"
    );
}

#[test]
fn crash_before_manifest_keeps_the_old_state_and_journal() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    let mut s = open(root);
    add_two(&mut s);
    // Tmps without a manifest are ignored: the old files and journal reopen.
    std::fs::write(root.join("spike_clusters.npy.tmp"), b"junk").unwrap();
    std::fs::write(root.join("j.journal.tmp"), b"junk").unwrap();
    drop(s);
    let s = open(root);
    assert_eq!(s.cluster_index().n_spikes(), 8);
}

/// A save that fails after the root was captured must leave the snapshot
/// independent of the top-level files: phy rewrites `spike_clusters.npy` in
/// place, which would also change a hard link. Only rescue arrays may link.
#[test]
fn failed_save_then_in_place_rewrite_keeps_the_root_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    let original = std::fs::read(root.join("spike_clusters.npy")).unwrap();
    let mut s = open(root);
    add_two(&mut s);
    // Fails the publish (after capture and seal) by blocking its manifest tmp.
    std::fs::create_dir(root.join("sorrel_rewrite.pending.tmp")).unwrap();
    assert!(save_and_seal(&mut s, root, label_str).is_err());
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .open(root.join("spike_clusters.npy"))
        .unwrap();
    f.write_all(&[0xff; 16]).unwrap();
    drop(f);
    let snap = root.join("sorrel_original/spike_clusters.npy");
    assert_eq!(std::fs::read(snap).unwrap(), original);
}

/// A phy edit to any top-level `cluster_*.tsv` after a sorrel save, one sorrel
/// wrote (a custom label) or one it only found (a metric), fails the open
/// closed; a normal save then reopen is fine.
#[test]
fn external_edit_to_any_cluster_tsv_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_fixture(root);
    std::fs::write(
        root.join("cluster_ContamPct.tsv"),
        "cluster_id\tContamPct\n0\t1.5\n",
    )
    .unwrap();
    let mut s = open(root);
    let cmd = s
        .set_label_command(ClusterId(1), "quality", "great")
        .unwrap();
    s.dispatch(cmd).unwrap();
    save_and_seal(&mut s, root, label_str).unwrap();
    drop(s);
    drop(open(root)); // sorrel's own save opens

    let opts = JournalOpts {
        path: Some(root.join("j.journal")),
        discard: false,
    };
    for name in ["cluster_quality.tsv", "cluster_ContamPct.tsv"] {
        let path = root.join(name);
        let good = std::fs::read(&path).unwrap();
        let mut edited = good.clone();
        edited.extend_from_slice(b"2\t9\n");
        std::fs::write(&path, edited).unwrap();
        let err = open_phy(root, KilosortOpenParams::default(), &opts).err();
        let err = format!("{:#}", err.unwrap_or_else(|| panic!("{name}: opened")));
        assert!(err.contains("changed outside sorrel"), "{name}: {err}");
        std::fs::write(&path, good).unwrap();
        drop(open(root));
    }
}

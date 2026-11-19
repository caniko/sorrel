//! An NWB units table's own ids reach SpikeInterface's `curation.json`.
#![cfg(feature = "hdf5")]

use sorrel_data::curation_json::curation_document;
use sorrel_data::{
    baseline_hash, open_identity_sealed, CurationCommand, Journal, JournalOpts, PhyLabelOp, Session,
};
use sorrel_io::kilosort::PhyLabel;
use sorrel_io::{ClusterId, DataProvider, NwbOpenParams, NwbProvider, SampleIndex};
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../release/fixtures/nwb")
        .join(name)
}

// Reproduce trunk's name/size/mtime identity independently of NwbProvider.
fn trunk_journal(path: &Path, journal: &Path, n_clusters: u32) -> Journal {
    let meta = path.metadata().unwrap();
    let mtime = meta
        .modified()
        .unwrap()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    let mut hash = blake3::Hasher::new();
    hash.update(b"sorrel-provider-identity-v1/nwb");
    hash.update(path.file_name().unwrap().to_str().unwrap().as_bytes());
    hash.update(&meta.len().to_le_bytes());
    hash.update(&mtime.as_secs().to_le_bytes());
    hash.update(&mtime.subsec_nanos().to_le_bytes());
    let baseline = baseline_hash(hash.finalize().as_bytes());
    Journal::open_or_create(journal, baseline, n_clusters).unwrap()
}

#[test]
#[cfg(feature = "phy-export")]
fn nwb_phy_export_refuses_zero_channels_before_writing() {
    let p = NwbProvider::open(fixture("curated_units.nwb")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new(p, Journal::open(&dir.path().join("j.sqlite")).unwrap());
    let out = dir.path().join("phy");
    let err = sorrel_data::export_phy(&s, &out, false, |l| l.as_str()).unwrap_err();
    assert!(err.to_string().contains("--dat"), "{err:#}");
    assert!(!out.exists());
}

#[test]
fn curation_json_uses_nwb_unit_ids() {
    let p = NwbProvider::open(fixture("good.nwb")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::new(p, Journal::open(&dir.path().join("j.sqlite")).unwrap());
    s.dispatch(CurationCommand::Merge {
        sources: vec![ClusterId(0)],
        target: ClusterId(2),
    })
    .unwrap();
    // A merge makes it SpikeInterface's list form: one structural step,
    // then the labels step on the ids that remain.
    let doc = curation_document(&s, |l| l.as_str());
    assert_eq!(doc[0]["unit_ids"], serde_json::json!([5, 7, 9]));
    assert_eq!(doc[0]["merges"][0]["unit_ids"], serde_json::json!([5, 9]));
    assert_eq!(doc[1]["unit_ids"], serde_json::json!([7, 9]));
}

#[test]
fn nwb_trunk_journal_replays_only_with_the_trunk_series() {
    let path = fixture("legacy_series.nwb");
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("trunk.journal");
    let trunk_series = || {
        NwbProvider::open_with(
            &path,
            NwbOpenParams {
                series: Some("Z_second"),
                ..Default::default()
            },
        )
        .unwrap()
    };
    {
        let provider = trunk_series();
        let j = trunk_journal(&path, &journal, provider.n_clusters());
        let mut session = Session::new(provider, j);
        session
            .dispatch(CurationCommand::Merge {
                sources: vec![ClusterId(0)],
                target: ClusterId(2),
            })
            .unwrap();
    }

    let opts = JournalOpts::default();
    let new_default = NwbProvider::open(&path).unwrap();
    assert_ne!(new_default.starting_time(), trunk_series().starting_time());
    let Err(err) = open_identity_sealed(new_default, &journal, &opts) else {
        panic!("trunk journal replayed against the newly visible series");
    };
    assert!(
        err.to_string().contains("sealed against different data"),
        "{err:#}"
    );

    // Z_second still maps spikes with the stored rate and preserves the trunk seal.
    let mut reopened = open_identity_sealed(trunk_series(), &journal, &opts).unwrap();
    reopened.replay_journal().unwrap();
    let doc = curation_document(&reopened, |l| l.as_str());
    assert_eq!(doc[0]["merges"][0]["unit_ids"], serde_json::json!([5, 9]));
}

#[test]
fn nwb_trunk_journal_replays_when_precise_rate_leaves_samples_unchanged() {
    let path = fixture("precise_rate_short.nwb");
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("trunk.journal");
    {
        let provider = NwbProvider::open(&path).unwrap();
        assert_eq!(provider.spike_times(ClusterId(0)), [SampleIndex(100)]);
        let j = trunk_journal(&path, &journal, provider.n_clusters());
        let mut session = Session::new(provider, j);
        session
            .dispatch(CurationCommand::Relabel {
                cluster: ClusterId(0),
                op: PhyLabelOp::SetMua,
            })
            .unwrap();
    }

    // An explicit choice with the same sample mapping also preserves the seal.
    for sample_rate in [None, Some(30000.123)] {
        let provider = NwbProvider::open_with(
            &path,
            NwbOpenParams {
                sample_rate,
                ..Default::default()
            },
        )
        .unwrap();
        let mut reopened =
            open_identity_sealed(provider, &journal, &JournalOpts::default()).unwrap();
        reopened.replay_journal().unwrap();
        assert_eq!(reopened.label(ClusterId(0)), Some(PhyLabel::Mua));
    }
}

#[test]
fn nwb_trunk_journal_replays_only_with_stored_clock_when_mappings_differ() {
    let path = fixture("precise_rate_long.nwb");
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("trunk.journal");
    let open = |sample_rate| {
        NwbProvider::open_with(
            &path,
            NwbOpenParams {
                sample_rate,
                ..Default::default()
            },
        )
    };
    assert!(open(None).is_err());
    let stored = open(Some(f64::from(30000.123_f32))).unwrap();
    assert_eq!(stored.spike_times(ClusterId(0)), [SampleIndex(330_000_001)]);
    drop(trunk_journal(&path, &journal, stored.n_clusters()));
    let opts = JournalOpts::default();
    open_identity_sealed(stored, &journal, &opts).unwrap();

    let precise = open(Some(30000.123)).unwrap();
    let Err(err) = open_identity_sealed(precise, &journal, &opts) else {
        panic!("trunk journal accepted for a different spike mapping");
    };
    assert!(
        err.to_string().contains("sealed against different data"),
        "{err:#}"
    );
}

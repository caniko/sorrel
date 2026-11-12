//! An NWB units table's own ids reach SpikeInterface's `curation.json`.
#![cfg(feature = "hdf5")]

use sorrel_data::curation_json::curation_document;
use sorrel_data::{CurationCommand, Journal, Session};
use sorrel_io::{ClusterId, NwbProvider};
use std::path::PathBuf;

#[test]
fn curation_json_uses_nwb_unit_ids() {
    let p = NwbProvider::open(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../release/fixtures/nwb/good.nwb"),
    )
    .unwrap();
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

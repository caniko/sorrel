//! Reader for SpikeInterface's manual-curation JSON (`curation.json`).
//!
//! Only the label side is consumed here: merges and splits are structural
//! edits that sorrel applies through its own journal, so importing them
//! from JSON is deliberately out of scope. The writer lives in
//! `sorrel-data` because it needs the live session.
//!
//! Both `format_version` `"1"` (`manual_labels[].quality`) and `"2"`
//! (`manual_labels[].labels.quality`) are accepted, matching what
//! `spikeinterface.curation.apply_curation` reads.

use crate::kilosort::PhyLabel;
use std::path::Path;

/// The label key sorrel writes and reads. SpikeInterface's examples use the
/// same name for good/mua/noise.
pub const QUALITY_KEY: &str = "quality";

/// Seed a label vector from `path`. `unit_index_of` maps a JSON `unit_id`
/// onto the dense cluster index (identity for phy directories). Returns
/// `None` when the file is absent or not a curation document, so callers
/// fall through to their next label source.
pub fn read_curation_labels(
    path: &Path,
    n_clusters: usize,
    unit_index_of: impl Fn(u64) -> Option<usize>,
) -> Option<Vec<PhyLabel>> {
    let text = std::fs::read_to_string(path).ok()?;
    let doc: serde_json::Value = serde_json::from_str(&text).ok()?;
    let entries = doc.get("manual_labels")?.as_array()?;
    let mut out = vec![PhyLabel::Unsorted; n_clusters];
    for entry in entries {
        let Some(uid) = entry.get("unit_id").and_then(unit_id_as_u64) else {
            continue;
        };
        // v2 nests label sets under "labels"; v1 keeps them at the top level.
        let quality = entry
            .get("labels")
            .and_then(|l| l.get(QUALITY_KEY))
            .or_else(|| entry.get(QUALITY_KEY));
        let Some(first) = quality.and_then(|q| q.as_array()).and_then(|a| a.first()) else {
            continue;
        };
        if let (Some(idx), Some(s)) = (unit_index_of(uid), first.as_str()) {
            if idx < n_clusters {
                out[idx] = PhyLabel::parse_tsv_value(s);
            }
        }
    }
    Some(out)
}

/// SpikeInterface writes unit ids as ints or strings depending on the sorter.
fn unit_id_as_u64(v: &serde_json::Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_str()?.trim().parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_v2_and_v1_shapes_with_int_or_string_unit_ids() {
        let dir = tempfile::tempdir().unwrap();
        let v2 = dir.path().join("v2.json");
        std::fs::write(
            &v2,
            r#"{"format_version":"2","unit_ids":[0,1,2],
                "manual_labels":[{"unit_id":0,"labels":{"quality":["good"]}},
                                 {"unit_id":"2","labels":{"quality":["noise"]}}],
                "merges":[],"splits":[],"removed":[]}"#,
        )
        .unwrap();
        let labels = read_curation_labels(&v2, 3, |u| Some(u as usize)).unwrap();
        assert_eq!(
            labels,
            vec![PhyLabel::Good, PhyLabel::Unsorted, PhyLabel::Noise]
        );

        let v1 = dir.path().join("v1.json");
        std::fs::write(
            &v1,
            r#"{"format_version":"1","unit_ids":[10,20],
                "manual_labels":[{"unit_id":20,"quality":["mua"]}]}"#,
        )
        .unwrap();
        let sparse = [10u64, 20];
        let labels = read_curation_labels(&v1, 2, |u| sparse.iter().position(|&x| x == u)).unwrap();
        assert_eq!(labels, vec![PhyLabel::Unsorted, PhyLabel::Mua]);

        assert!(
            read_curation_labels(&dir.path().join("missing.json"), 2, |u| Some(u as usize))
                .is_none()
        );
    }
}

//! Reader for SpikeInterface's manual-curation JSON (`curation.json`).
//!
//! [`read_curation_labels`] seeds a session's baseline labels and reads the
//! label side only. [`read_curation`] is the explicit import behind
//! `sorrel apply --curation`: labels, `tags` and merges, which the caller
//! replays through its own journal. The writer lives in `sorrel-data`
//! because it needs the live session.
//!
//! Both `format_version` `"1"` (`manual_labels[].quality`) and `"2"`
//! (`manual_labels[].labels.quality`) are accepted, matching what
//! `spikeinterface.curation.apply_curation` reads. A document with merges or
//! splits on the path to the head is SpikeInterface's list form
//! (`SequentialCuration`): `read_curation_labels` takes the final labels-only
//! step, and `read_curation` refuses the list — replaying a history needs the
//! session's own journal.

use crate::kilosort::PhyLabel;
use std::path::Path;

/// The label key sorrel writes and reads. SpikeInterface's examples use the
/// same name for good/mua/noise.
pub const QUALITY_KEY: &str = "quality";

/// Seed a label vector from `path`. `unit_index_of` maps a JSON `unit_id`
/// onto the dense cluster index (identity for phy directories). Returns
/// `None` when the file is absent or not a curation document, so callers
/// fall through to their next label source.
///
/// A list document's final step carries labels keyed by the final unit ids;
/// units untouched by structure keep their ids, so the dense mapping
/// resolves them. Ids the history created are skipped (the journal that
/// made them also replays their labels).
pub fn read_curation_labels(
    path: &Path,
    n_clusters: usize,
    unit_index_of: impl Fn(&str) -> Option<usize>,
) -> Option<Vec<PhyLabel>> {
    let text = std::fs::read_to_string(path).ok()?;
    let doc: serde_json::Value = serde_json::from_str(&text).ok()?;
    // The list form ends with one labels-only step; earlier steps carry no
    // labels of their own.
    let step = match &doc {
        serde_json::Value::Array(steps) => steps.last()?,
        _ => &doc,
    };
    let entries = step.get("manual_labels")?.as_array()?;
    let mut out = vec![PhyLabel::Unsorted; n_clusters];
    for entry in entries {
        let Some(uid) = entry.get("unit_id").and_then(unit_id_text) else {
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
        if let (Some(idx), Some(s)) = (unit_index_of(&uid), first.as_str()) {
            if idx < n_clusters {
                out[idx] = PhyLabel::parse_tsv_value(s);
            }
        }
    }
    Some(out)
}

/// The non-exclusive label category sorrel uses for provenance/review tags.
pub const TAGS_KEY: &str = "tags";

/// Everything `sorrel apply --curation` imports from a curation document,
/// still keyed by the document's unit ids.
#[derive(Debug, Default, PartialEq)]
pub struct CurationImport {
    /// `(unit_id, quality, tags)` per `manual_labels` entry.
    pub labels: Vec<(String, Option<PhyLabel>, Vec<String>)>,
    /// Unit-id groups to merge (v2 `merges[].unit_ids`, v1 `merge_unit_groups`).
    pub merges: Vec<Vec<String>>,
}

/// Read labels, tags and merges from `path`. Unlike
/// [`read_curation_labels`] this is an explicit import, so a missing or
/// malformed file is an error rather than a fall-through. A list document
/// (a session with merges or splits replayed as steps) is refused: replaying
/// it needs the edits in order, which only sorrel's own journal has.
/// Splits and `removed` units are rejected: the auto pipeline never emits
/// them and silently dropping them would misrepresent the document.
pub fn read_curation(path: &Path) -> anyhow::Result<CurationImport> {
    use anyhow::{bail, Context};
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let doc: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    if doc.is_array() {
        bail!(
            "{}: a sequential curation list (a history with merges or splits) cannot be \
             replayed from JSON; apply it in SpikeInterface or export phy instead",
            path.display()
        );
    }
    let non_empty = |k: &str| {
        doc.get(k)
            .and_then(|v| v.as_array())
            .is_some_and(|a| !a.is_empty())
    };
    if non_empty("splits") || non_empty("removed") || non_empty("removed_units") {
        bail!(
            "{}: splits/removed units are not importable",
            path.display()
        );
    }
    let strings = |v: Option<&serde_json::Value>| -> Vec<String> {
        v.and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut out = CurationImport::default();
    for entry in doc
        .get("manual_labels")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        let uid = entry
            .get("unit_id")
            .and_then(unit_id_text)
            .with_context(|| format!("manual_labels entry without unit_id: {entry}"))?;
        let sets = entry.get("labels").unwrap_or(entry);
        let quality = strings(sets.get(QUALITY_KEY))
            .first()
            .map(|s| PhyLabel::parse_tsv_value(s));
        out.labels.push((uid, quality, strings(sets.get(TAGS_KEY))));
    }
    let merges = doc
        .get("merges")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|m| m.get("unit_ids"))
                .collect::<Vec<_>>()
        })
        .or_else(|| {
            doc.get("merge_unit_groups")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().collect())
        })
        .unwrap_or_default();
    for group in merges {
        let ids: Option<Vec<String>> = group
            .as_array()
            .map(|a| a.iter().map(unit_id_text).collect())
            .unwrap_or(None);
        match ids {
            Some(ids) if ids.len() >= 2 => out.merges.push(ids),
            _ => bail!("{}: malformed merge group {group}", path.display()),
        }
    }
    Ok(out)
}

/// A unit id as text: SpikeInterface writes ints or strings depending on
/// the sorter, and strings need not be numbers (`"unit_a"`). Callers fall
/// back to the number a string reads as (`"7"` vs `7`).
pub fn unit_id_text(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.trim().to_owned()),
        other => other.as_u64().map(|u| u.to_string()),
    }
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
        let labels = read_curation_labels(&v2, 3, |u| u.parse().ok()).unwrap();
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
        let labels =
            read_curation_labels(&v1, 2, |u| sparse.iter().position(|x| x.to_string() == u))
                .unwrap();
        assert_eq!(labels, vec![PhyLabel::Unsorted, PhyLabel::Mua]);

        assert!(
            read_curation_labels(&dir.path().join("missing.json"), 2, |u| u.parse().ok()).is_none()
        );
    }

    #[test]
    fn read_curation_imports_quality_tags_and_merges() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.json");
        std::fs::write(
            &p,
            r#"{"format_version":"2","unit_ids":[0,1,2,3],
                "label_definitions":{"quality":{"label_options":["good","mua","noise"],"exclusive":true},
                                     "tags":{"label_options":["agree","review:merge"],"exclusive":false}},
                "manual_labels":[{"unit_id":0,"labels":{"quality":["good"],"tags":["agree"]}},
                                 {"unit_id":"3","labels":{"tags":["review:merge"]}}],
                "merges":[{"unit_ids":[1,"2"],"new_unit_id":4}],"splits":[],"removed":[]}"#,
        )
        .unwrap();
        let got = read_curation(&p).unwrap();
        assert_eq!(
            got.labels,
            vec![
                ("0".into(), Some(PhyLabel::Good), vec!["agree".to_string()]),
                ("3".into(), None, vec!["review:merge".to_string()]),
            ]
        );
        assert_eq!(got.merges, vec![vec!["1".to_string(), "2".into()]]);

        std::fs::write(&p, r#"{"splits":[{"unit_id":0,"mode":"indices"}]}"#).unwrap();
        assert!(read_curation(&p).is_err());
        assert!(read_curation(&dir.path().join("missing.json")).is_err());
    }

    /// A history with merges/splits is SpikeInterface's list form: labels
    /// live in the final labels-only step and are keyed by the final unit
    /// ids; the explicit import refuses the list rather than dropping the
    /// structure.
    #[test]
    fn reads_list_form_labels_and_refuses_the_import() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.json");
        std::fs::write(
            &p,
            r#"[{"format_version":"2","unit_ids":[0,1],"manual_labels":[],
                 "merges":[{"unit_ids":[1,0],"new_unit_id":0}],"splits":[],"removed":[]},
                {"format_version":"2","unit_ids":[0,2],
                 "manual_labels":[{"unit_id":0,"labels":{"quality":["good"]}},
                                  {"unit_id":2,"labels":{"quality":["noise"]}}],
                 "merges":[],"splits":[],"removed":[]}]"#,
        )
        .unwrap();
        assert_eq!(
            read_curation_labels(&p, 3, |u| u.parse().ok()).unwrap(),
            vec![PhyLabel::Good, PhyLabel::Unsorted, PhyLabel::Noise]
        );
        // Ids the history created are not in the dense mapping: skipped.
        let sparse = [0u64, 9];
        assert_eq!(
            read_curation_labels(&p, 2, |u| sparse.iter().position(|x| x.to_string() == u))
                .unwrap(),
            vec![PhyLabel::Good, PhyLabel::Unsorted]
        );
        let err = read_curation(&p).unwrap_err().to_string();
        assert!(err.contains("sequential curation list"), "{err}");

        std::fs::write(&p, "[]").unwrap();
        assert!(read_curation_labels(&p, 1, |u| u.parse().ok()).is_none());
    }
}

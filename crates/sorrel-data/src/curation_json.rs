//! Writer for SpikeInterface's manual-curation JSON (`format_version` 2).
//!
//! The document describes what is **not yet** in the artifacts the journal
//! is sealed against: labels for every unit, plus the merges and splits on
//! the undo stack. After `save_and_reseal_journal` the stack is empty and
//! the file degrades to labels only, which is exactly right for phy
//! directories where `spike_clusters.npy` already carries the structure.
//! For SortingAnalyzer folders (never resealed) the file is the carrier,
//! and `spikeinterface.curation.apply_curation(analyzer, "curation.json")`
//! reproduces the session.
//!
//! Field names follow `spikeinterface/curation/curation_model.py`:
//! `format_version`, `unit_ids`, `label_definitions`, `manual_labels`,
//! `merges`, `splits`, `removed`. Notes have no slot and are dropped; the
//! unit origin travels as an exclusive `origin` label.
//! `apply_curation` applies merges before splits, so a split recorded
//! after a merge into the same unit indexes the merged, time-sorted spike
//! train — the same order sorrel's buckets keep.

use crate::command::{CurationCommand, Method, UnitOrigin};
use crate::session::{Session, TAG_SPLIT, TAG_TUNED};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use sorrel_io::curation_json::{QUALITY_KEY, TAGS_KEY};
use sorrel_io::DataProvider;
use std::path::Path;

/// Exclusive label holding each unit's origin (`Auto[ks4]`, `Sorrel[Manual]`).
const ORIGIN_KEY: &str = "origin";

/// Write `curation.json` describing `session` to `path` (atomically).
pub fn write_curation_json<P, F>(session: &Session<P>, path: &Path, label_str: F) -> Result<()>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    let doc = curation_document(session, label_str);
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&doc)?)
        .with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

/// The in-memory document; split out so tests and `serve` can inspect it.
pub fn curation_document<P, F>(session: &Session<P>, label_str: F) -> Value
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    build(session, label_str, session.applied_commands())
}

/// The document for a session whose artifacts were just rewritten and whose
/// journal is about to be resealed: labels only, every cluster a unit.
pub fn curation_document_sealed<P, F>(session: &Session<P>, label_str: F) -> Value
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    build(session, label_str, Vec::new())
}

fn build<P, F>(session: &Session<P>, label_str: F, history: Vec<CurationCommand>) -> Value
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    let p = &session.provider;
    let uid = |c| p.unit_id(c);
    // With edits pending, the units are the backend's originals; once
    // resealed every cluster in the artifacts is a unit.
    let n_units = if history.is_empty() {
        session.n_clusters()
    } else {
        p.n_clusters()
    };
    let unit_ids: Vec<u32> = (0..n_units).map(|c| uid(sorrel_io::ClusterId(c))).collect();

    let mut tag_options = std::collections::BTreeSet::new();
    // Every unit has exactly one origin; SpikeInterface needs each used value
    // listed and at least two options.
    let mut origin_options = std::collections::BTreeSet::from([
        UnitOrigin::Auto.label(session.sorter()),
        UnitOrigin::Sorrel(Method::Manual).label(session.sorter()),
    ]);
    let manual_labels: Vec<Value> = (0..n_units)
        .filter_map(|i| {
            let c = sorrel_io::ClusterId(i);
            let mut sets = serde_json::Map::new();
            let origin = session.origin_label(c);
            origin_options.insert(origin.clone());
            sets.insert(ORIGIN_KEY.into(), json!([origin]));
            if let Some(s) = session.label(c).map(|l| label_str(&l)) {
                if !s.is_empty() && s != "unsorted" {
                    sets.insert(QUALITY_KEY.into(), json!([s]));
                }
            }
            let tags = session.tags(c);
            if !tags.is_empty() {
                tag_options.extend(tags.iter().cloned());
                sets.insert(TAGS_KEY.into(), json!(tags));
            }
            (!sets.is_empty()).then(|| json!({ "unit_id": uid(c), "labels": sets }))
        })
        .collect();
    let mut label_definitions = json!({
        QUALITY_KEY: { "label_options": ["good", "mua", "noise"], "exclusive": true },
        ORIGIN_KEY: { "label_options": origin_options, "exclusive": true },
    });
    // SpikeInterface needs every used value listed and at least two options.
    if !tag_options.is_empty() {
        tag_options.insert("src:sorrel".into());
        tag_options.insert("src:human".into());
        label_definitions[TAGS_KEY] = json!({ "label_options": tag_options, "exclusive": false });
    }

    let mut merges = Vec::new();
    let mut splits = Vec::new();
    let mut visit = |cmd: &CurationCommand| match cmd {
        CurationCommand::Merge { sources, target } => {
            let mut ids: Vec<u32> = sources.iter().map(|&s| uid(s)).collect();
            ids.push(uid(*target));
            merges.push(json!({ "unit_ids": ids, "new_unit_id": uid(*target) }));
        }
        CurationCommand::Split {
            cluster,
            spike_idx,
            new_cluster,
            ..
        } => splits.push(json!({
            "unit_id": uid(*cluster),
            "mode": "indices",
            "indices": [spike_idx],
            "new_unit_ids": [uid(*new_cluster)],
        })),
        _ => {}
    };
    for cmd in &history {
        match cmd {
            CurationCommand::Batch { children } => children.iter().for_each(&mut visit),
            other => visit(other),
        }
    }

    json!({
        "format_version": "2",
        "unit_ids": unit_ids,
        "label_definitions": label_definitions,
        "manual_labels": manual_labels,
        "merges": merges,
        "splits": splits,
        "removed": [],
    })
}

/// The commands that replay an imported curation document onto `session`:
/// relabels and retags first (deltas only, so re-importing is a no-op),
/// then one merge per group into its lowest cluster id. Labelling before
/// merging matches `apply_curation`, and the auto pipeline gives every
/// member of a merge group the same label and tags.
///
/// `current` maps a cluster to its present label string (`"good"`, ...).
/// Unknown unit ids, invalid tags and units in two merge groups are errors:
/// a partially applied import would be worse than none.
pub fn import_commands<P: DataProvider>(
    session: &Session<P>,
    doc: &sorrel_io::CurationImport,
    current: impl Fn(sorrel_io::ClusterId) -> &'static str,
) -> Result<Vec<CurationCommand>> {
    use crate::rules::label_op;
    use anyhow::bail;
    use sorrel_io::kilosort::valid_tag;
    use sorrel_io::ClusterId;

    let by_unit: std::collections::HashMap<u64, ClusterId> = (0..session.provider.n_clusters())
        .map(|c| {
            (
                u64::from(session.provider.unit_id(ClusterId(c))),
                ClusterId(c),
            )
        })
        .collect();
    let cluster = |uid: u64| {
        by_unit
            .get(&uid)
            .copied()
            .with_context(|| format!("unit id {uid} is not in this dataset"))
    };

    let mut out = Vec::new();
    for (uid, quality, tags) in &doc.labels {
        let c = cluster(*uid)?;
        if let Some(q) = quality {
            let op = label_op(q.as_str())?;
            if label_op(current(c))? != op {
                out.push(CurationCommand::Relabel { cluster: c, op });
            }
        }
        if let Some(bad) = tags.iter().find(|t| !valid_tag(t)) {
            bail!("unit {uid}: invalid tag {bad:?} (empty, or contains a comma/tab/newline)");
        }
        // Session markers are not review state: ignore them when comparing
        // (re-import stays a no-op), never take them from the file, and keep
        // the unit's own on retag.
        let is_marker = |t: &String| {
            matches!(
                t.as_str(),
                TAG_TUNED | TAG_SPLIT | sorrel_io::kilosort::TAG_EXTERNAL
            )
        };
        let current: Vec<&String> = session.tags(c).iter().filter(|t| !is_marker(t)).collect();
        let wanted: Vec<&String> = tags.iter().filter(|t| !is_marker(t)).collect();
        if current != wanted {
            let mut merged: Vec<String> = wanted.into_iter().cloned().collect();
            merged.extend(session.tags(c).iter().filter(|t| is_marker(t)).cloned());
            out.push(CurationCommand::Tag {
                cluster: c,
                tags: merged,
            });
        }
    }
    let mut seen = std::collections::HashSet::new();
    for group in &doc.merges {
        let mut members = group
            .iter()
            .map(|&u| cluster(u))
            .collect::<Result<Vec<_>>>()?;
        members.sort_unstable();
        members.dedup();
        if let Some(dup) = members.iter().find(|c| !seen.insert(**c)) {
            bail!("cluster {} appears in more than one merge group", dup.0);
        }
        if let Some((&target, sources)) = members.split_first() {
            // Already merged (e.g. imported, then saved): nothing left to move.
            if sources.iter().any(|&s| !session.spike_times(s).is_empty()) {
                out.push(CurationCommand::Merge {
                    sources: sources.to_vec(),
                    target,
                });
            }
        }
    }
    Ok(out)
}

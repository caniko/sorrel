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
//! `merges`, `splits`, `removed`. Notes have no slot and are dropped.
//! `apply_curation` applies merges before splits, so a split recorded
//! after a merge into the same unit indexes the merged, time-sorted spike
//! train — the same order sorrel's buckets keep.

use crate::command::CurationCommand;
use crate::session::Session;
use anyhow::{Context, Result};
use serde_json::{json, Value};
use sorrel_io::curation_json::QUALITY_KEY;
use sorrel_io::DataProvider;
use std::path::Path;

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

    let manual_labels: Vec<Value> = session
        .labels()
        .iter()
        .take(n_units as usize)
        .enumerate()
        .filter_map(|(i, l)| {
            let s = label_str(l);
            (!s.is_empty() && s != "unsorted").then(|| {
                json!({
                    "unit_id": uid(sorrel_io::ClusterId(i as u32)),
                    "labels": { QUALITY_KEY: [s] },
                })
            })
        })
        .collect();

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
        "label_definitions": {
            QUALITY_KEY: { "label_options": ["good", "mua", "noise"], "exclusive": true }
        },
        "manual_labels": manual_labels,
        "merges": merges,
        "splits": splits,
        "removed": [],
    })
}

//! Writer for SpikeInterface's manual-curation JSON (`format_version` 2).
//!
//! The document describes the session relative to the sorter's output:
//! labels for every unit, plus the merges and splits on the path to the
//! head. A phy directory saved onto itself (`save_and_seal`) gets the
//! labels-only form instead, because its `spike_clusters.npy` already
//! carries the structure. For SortingAnalyzer folders the file is the
//! carrier, and `spikeinterface.curation.apply_curation(analyzer,
//! sorrel.read_curation(dir))` reproduces the session
//! (`crates/sorrel-py/python/tests/test_si_curation.py`).
//!
//! Field names follow `spikeinterface/curation/curation_model.py`:
//! `format_version`, `unit_ids`, `label_definitions`, `manual_labels`,
//! `merges`, `splits`, `removed`. Notes have no slot and are dropped; the
//! unit origin travels as an exclusive `origin` label.
//!
//! A labels-only history, and a sealed phy save, are one `Curation` dict.
//! A history with merges or splits is SpikeInterface's list form
//! (`SequentialCuration`): structural steps, then one labels-only step
//! keyed by the final unit ids. Each step's `unit_ids` are the ids it
//! starts from and every merge/split carries explicit new ids, which the
//! list form requires. That expresses what one pass cannot: a unit both
//! merged and split, a second split of a unit, and edits of units sorrel
//! created. Consecutive commands share a step while they touch disjoint
//! units, so a run of merges stays one step.
//!
//! Ids follow `apply_curation`: a merge keeps its target's id, a split's
//! child gets the id sorrel allocated for it (`uid(new_cluster)`) and the
//! spikes left behind get a fresh id, because SpikeInterface never reuses
//! a split unit's id (`next_free`, above every id the history allocates).
//! Sorrel's `Split.spike_idx` indexes the cluster's time-sorted train at
//! that point, which is exactly the train `mode: "indices"` indexes.
//!
//! A command with no SpikeInterface slot (a rescue: `AddSpikes`) does not
//! fail the save; the document names it under `sorrel_unrepresentable`
//! and `sorrel-py`'s `read_curation` raises. SpikeInterface's pydantic
//! models ignore unknown fields.

use crate::command::{describe_command, CurationCommand, Method, UnitOrigin};
use crate::session::{Session, TAG_SPLIT, TAG_TUNED};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use sorrel_io::curation_json::{QUALITY_KEY, TAGS_KEY};
use sorrel_io::{ClusterId, DataProvider};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

/// Exclusive label holding each unit's origin (`Auto[ks4]`, `Sorrel[Manual]`).
const ORIGIN_KEY: &str = "origin";

/// Field listing commands the document cannot express. SpikeInterface
/// ignores unknown keys; `sorrel.read_curation` refuses the document.
const UNREPRESENTABLE_KEY: &str = "sorrel_unrepresentable";

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
    sorrel_io::fsutil::replace_file(&tmp, path)
}

/// The in-memory document; split out so tests and `serve` can inspect it.
pub fn curation_document<P, F>(session: &Session<P>, label_str: F) -> Value
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    build(session, label_str, session.applied_commands())
}

/// The document for a session whose artifacts were just rewritten (a sealed
/// save): labels only, every cluster a unit.
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
    let strings = p.unit_ids_are_strings();
    let uid = |c: ClusterId| -> Value {
        let u = p.unit_id(c);
        if strings {
            json!(u.to_string())
        } else {
            json!(u)
        }
    };
    let sealed = history.is_empty();
    let mut cmds = Vec::new();
    flatten(&history, &mut cmds);
    let structural = cmds.iter().any(|c| {
        matches!(
            c,
            CurationCommand::Merge { .. } | CurationCommand::Split { .. }
        )
    });

    if !structural {
        // Labels only: either a sealed save (every cluster is a unit) or a
        // history that never touched spike membership (the provider's units).
        let n_units = if sealed {
            session.n_clusters()
        } else {
            p.n_clusters()
        };
        let units = (0..n_units).map(|i| (uid(ClusterId(i)), ClusterId(i)));
        let (defs, labels) = labels_document(session, &label_str, units);
        let mut doc = json!({
            "format_version": "2",
            "unit_ids": (0..n_units).map(|i| uid(ClusterId(i))).collect::<Vec<_>>(),
            "label_definitions": defs,
            "manual_labels": labels,
            "merges": [],
            "splits": [],
            "removed": [],
        });
        let bad: Vec<String> = cmds.iter().copied().filter_map(unrepresentable).collect();
        if !bad.is_empty() {
            doc[UNREPRESENTABLE_KEY] = json!(bad);
        }
        return doc;
    }

    structural_document(session, &label_str, &cmds, &uid)
}

/// The list form: packed structural steps, then one labels-only step.
fn structural_document<P, F, U>(
    session: &Session<P>,
    label_str: &F,
    cmds: &[&CurationCommand],
    uid: &U,
) -> Value
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
    U: Fn(ClusterId) -> Value,
{
    let p = &session.provider;
    // Sorrel cluster -> the unit id SpikeInterface currently has for it.
    // Merged-away sources leave the map; a split parent is remapped to the
    // spikes left behind.
    let mut si_of: HashMap<ClusterId, Value> = (0..p.n_clusters())
        .map(|c| (ClusterId(c), uid(ClusterId(c))))
        .collect();
    // The live clusters in SpikeInterface's unit-id order: a merge keeps its
    // target's place, a split replaces the parent with its parts at the end
    // (`_get_ids_after_splitting`).
    let mut order: Vec<ClusterId> = (0..p.n_clusters()).map(ClusterId).collect();
    // Every id the history allocates, so a remainder never reuses an id.
    let mut next_free = session.n_clusters();
    let mut steps: Vec<Step> = Vec::new();
    let mut cur: Option<Step> = None;
    let mut bad: Vec<String> = Vec::new();

    for &cmd in cmds {
        let op = match cmd {
            CurationCommand::Merge { sources, target } => {
                let Some(target_si) = si_of.get(target).cloned() else {
                    bad.push(format!(
                        "{}: an earlier command removed the unit it merges into",
                        describe_command(cmd)
                    ));
                    continue;
                };
                let mut live: Vec<ClusterId> = Vec::new();
                for &s in sources.iter().filter(|s| *s != target) {
                    if !live.contains(&s) && si_of.contains_key(&s) {
                        live.push(s);
                    }
                }
                if live.is_empty() {
                    // Every source is already empty: membership does not move.
                    continue;
                }
                let group: Vec<Value> = live
                    .iter()
                    .map(|s| si_of[s].clone())
                    .chain([target_si.clone()])
                    .collect();
                Op::Merge {
                    group,
                    new: target_si,
                    sources: live,
                }
            }
            CurationCommand::Split {
                cluster,
                spike_idx,
                new_cluster,
                ..
            } => {
                let Some(unit) = si_of.get(cluster).cloned() else {
                    bad.push(format!(
                        "{}: an earlier command removed the unit it splits",
                        describe_command(cmd)
                    ));
                    continue;
                };
                let child = uid(*new_cluster);
                let rest = uid(ClusterId(next_free));
                next_free += 1;
                let used = |id: &Value| si_of.values().any(|v| v == id);
                if used(&child) || used(&rest) {
                    bad.push(format!(
                        "{}: no free SpikeInterface unit id for the new unit",
                        describe_command(cmd)
                    ));
                    continue;
                }
                Op::Split {
                    unit,
                    indices: spike_idx.clone(),
                    new: vec![child.clone(), rest],
                    cluster: *cluster,
                    child: *new_cluster,
                }
            }
            CurationCommand::AddSpikes(_) => {
                bad.push(unrepresentable(cmd).expect("AddSpikes has no curation slot"));
                continue;
            }
            _ => continue,
        };

        // A step applies as one curation: its commands must touch disjoint
        // units, or `apply_curation` would merge/split a unit an earlier
        // command in the same step already consumed or created.
        let touches = op.touches();
        let conflict = cur.as_ref().is_some_and(|s| {
            touches
                .iter()
                .any(|t| s.sealed.iter().any(|done| done == t))
        });
        if cur.is_none() || conflict {
            steps.extend(cur.take());
            cur = Some(Step {
                unit_ids: order.iter().map(|c| si_of[c].clone()).collect(),
                ..Step::default()
            });
        }
        // The step's ids are captured above; record the command, then move
        // the state so later commands and steps see it.
        let step = cur.as_mut().expect("a step was started");
        match &op {
            Op::Merge { group, new, .. } => step
                .merges
                .push(json!({ "unit_ids": group, "new_unit_id": new })),
            Op::Split {
                unit, indices, new, ..
            } => step.splits.push(json!({
                "unit_id": unit,
                "mode": "indices",
                "indices": [indices],
                "new_unit_ids": new,
            })),
        }
        step.sealed.extend(op_seals(&op));
        op.apply(&mut si_of, &mut order);
    }
    steps.extend(cur.take());

    let (defs, labels) = labels_document(
        session,
        label_str,
        order.iter().map(|c| (si_of[c].clone(), *c)),
    );
    let mut docs: Vec<Value> = steps
        .into_iter()
        .map(|s| step_json(s.unit_ids, &s.merges, &s.splits, &defs, Vec::new()))
        .collect();
    let mut last = step_json(
        order.iter().map(|c| si_of[c].clone()).collect(),
        &[],
        &[],
        &defs,
        labels,
    );
    if !bad.is_empty() {
        last[UNREPRESENTABLE_KEY] = json!(bad);
    }
    docs.push(last);
    Value::Array(docs)
}

/// One packed structural step.
#[derive(Default)]
struct Step {
    unit_ids: Vec<Value>,
    merges: Vec<Value>,
    splits: Vec<Value>,
    /// Ids an earlier command in this step consumed or created.
    sealed: Vec<Value>,
}

/// One structural command, resolved against the ids before it applies.
enum Op {
    Merge {
        group: Vec<Value>,
        new: Value,
        sources: Vec<ClusterId>,
    },
    Split {
        unit: Value,
        indices: Vec<u32>,
        new: Vec<Value>,
        cluster: ClusterId,
        child: ClusterId,
    },
}

impl Op {
    /// Every unit id the command names.
    fn touches(&self) -> Vec<Value> {
        match self {
            Self::Merge { group, .. } => group.clone(),
            Self::Split { unit, .. } => vec![unit.clone()],
        }
    }

    /// Move the mapped ids the way `apply_curation` moves the unit list.
    fn apply(self, si_of: &mut HashMap<ClusterId, Value>, order: &mut Vec<ClusterId>) {
        match self {
            Self::Merge { sources, .. } => {
                for s in &sources {
                    si_of.remove(s);
                }
                order.retain(|c| !sources.contains(c));
            }
            Self::Split {
                cluster,
                child,
                new,
                ..
            } => {
                si_of.insert(child, new[0].clone());
                si_of.insert(cluster, new[1].clone());
                order.retain(|c| *c != cluster);
                order.push(child);
                order.push(cluster);
            }
        }
    }
}

/// The ids an op creates or consumes; later commands in the same step may not
/// touch them.
fn op_seals(op: &Op) -> Vec<Value> {
    match op {
        Op::Merge { group, .. } => group.clone(),
        Op::Split { unit, new, .. } => std::iter::once(unit.clone()).chain(new.clone()).collect(),
    }
}

fn step_json(
    unit_ids: Vec<Value>,
    merges: &[Value],
    splits: &[Value],
    defs: &Value,
    manual_labels: Vec<Value>,
) -> Value {
    json!({
        "format_version": "2",
        "unit_ids": unit_ids,
        "label_definitions": defs,
        "manual_labels": manual_labels,
        "merges": merges,
        "splits": splits,
        "removed": [],
    })
}

/// Commands with no SpikeInterface curation slot.
fn unrepresentable(cmd: &CurationCommand) -> Option<String> {
    match cmd {
        CurationCommand::AddSpikes(_) => Some(format!(
            "{}: SpikeInterface curation documents cannot add spikes",
            describe_command(cmd)
        )),
        _ => None,
    }
}

/// Flatten batches in apply order: the document replays them in the order
/// sorrel did.
fn flatten<'a>(history: &'a [CurationCommand], out: &mut Vec<&'a CurationCommand>) {
    for cmd in history {
        match cmd {
            CurationCommand::Batch { children } => flatten(children, out),
            other => out.push(other),
        }
    }
}

/// Label definitions plus one `manual_labels` entry per `(unit id, cluster)`.
fn labels_document<P, F>(
    session: &Session<P>,
    label_str: &F,
    units: impl Iterator<Item = (Value, ClusterId)>,
) -> (Value, Vec<Value>)
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str,
{
    let mut tag_options = BTreeSet::new();
    // Every unit has exactly one origin; SpikeInterface needs each used value
    // listed and at least two options.
    let mut origin_options = BTreeSet::from([
        UnitOrigin::Auto.label(session.sorter()),
        UnitOrigin::Sorrel(Method::Manual).label(session.sorter()),
    ]);
    let manual_labels: Vec<Value> = units
        .filter_map(|(id, c)| {
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
            (!sets.is_empty()).then(|| json!({ "unit_id": id, "labels": sets }))
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
    (label_definitions, manual_labels)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::PhyLabelOp;
    use crate::journal::Journal;
    use crate::session::ApplyPhyLabel;
    use serde_json::json;
    use sorrel_io::{SampleIndex, TraceSamples, TraceSlice};

    /// Four clusters with enough spikes to split twice.
    struct Mock {
        spikes: Vec<Vec<SampleIndex>>,
    }

    #[repr(u8)]
    #[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
    enum MockLabel {
        #[default]
        Unsorted = 0,
        Good = 1,
        Mua = 2,
        Noise = 3,
    }

    impl DataProvider for Mock {
        type Label = MockLabel;
        fn sample_rate(&self) -> f32 {
            1000.0
        }
        fn n_channels(&self) -> u32 {
            0
        }
        fn n_samples(&self) -> SampleIndex {
            SampleIndex(0)
        }
        fn n_clusters(&self) -> u32 {
            self.spikes.len() as u32
        }
        fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
            self.spikes.get(c.idx()).map(Vec::as_slice).unwrap_or(&[])
        }
        fn trace(&self, _: SampleIndex, _: u32) -> TraceSlice<'_> {
            TraceSlice {
                start: SampleIndex(0),
                n_channels: 0,
                samples: TraceSamples::I16(&[]),
            }
        }
        fn initial_labels(&self) -> Vec<Self::Label> {
            vec![MockLabel::Unsorted; self.spikes.len()]
        }
    }

    impl ApplyPhyLabel for Mock {
        fn label_from_op(op: PhyLabelOp) -> Self::Label {
            match op {
                PhyLabelOp::SetUnsorted => MockLabel::Unsorted,
                PhyLabelOp::SetGood => MockLabel::Good,
                PhyLabelOp::SetMua => MockLabel::Mua,
                PhyLabelOp::SetNoise => MockLabel::Noise,
            }
        }
    }

    fn label_str(l: &MockLabel) -> &'static str {
        match l {
            MockLabel::Unsorted => "unsorted",
            MockLabel::Good => "good",
            MockLabel::Mua => "mua",
            MockLabel::Noise => "noise",
        }
    }

    fn fresh_session() -> (Session<Mock>, tempfile::TempDir) {
        let provider = Mock {
            spikes: vec![
                vec![
                    SampleIndex(10),
                    SampleIndex(30),
                    SampleIndex(150),
                    SampleIndex(200),
                ],
                vec![SampleIndex(20)],
                vec![SampleIndex(100)],
                vec![SampleIndex(40)],
            ],
        };
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open(&dir.path().join("s.journal")).unwrap();
        (Session::new(provider, journal), dir)
    }

    fn split(cluster: u32, idx: u32) -> CurationCommand {
        CurationCommand::Split {
            cluster: ClusterId(cluster),
            spike_idx: vec![idx],
            new_cluster: ClusterId(u32::MAX),
            by: Method::Manual,
        }
    }

    fn steps(doc: &Value) -> &Vec<Value> {
        doc.as_array()
            .expect("a history with merges/splits is a list")
    }

    #[test]
    fn merges_only_history_is_one_structural_step() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(0)],
            target: ClusterId(1),
        })
        .unwrap();
        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(3)],
            target: ClusterId(2),
        })
        .unwrap();
        let doc = curation_document(&s, label_str);
        let steps = steps(&doc);
        assert_eq!(steps.len(), 2, "one merge step + one label step");
        assert_eq!(steps[0]["unit_ids"], json!([0, 1, 2, 3]));
        assert_eq!(steps[0]["merges"][0]["unit_ids"], json!([0, 1]));
        assert_eq!(steps[0]["merges"][0]["new_unit_id"], 1);
        assert_eq!(steps[0]["merges"][1]["unit_ids"], json!([3, 2]));
        assert_eq!(steps[0]["merges"][1]["new_unit_id"], 2);
        assert_eq!(steps[0]["splits"], json!([]));
        assert!(steps[0]["label_definitions"].is_object());
        assert_eq!(steps[0]["manual_labels"], json!([]));
        // Sources leave the unit list; targets keep their ids.
        assert_eq!(steps[1]["unit_ids"], json!([1, 2]));
        assert_eq!(steps[1]["manual_labels"][0]["unit_id"], 1);
    }

    #[test]
    fn two_splits_of_one_unit_are_separate_steps_with_fresh_ids() {
        let (mut s, _d) = fresh_session();
        s.dispatch(split(0, 0)).unwrap();
        // The child is 4; label it, so the final step must carry it.
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(4),
            op: PhyLabelOp::SetGood,
        })
        .unwrap();
        s.dispatch(split(0, 0)).unwrap();
        let doc = curation_document(&s, label_str);
        let steps = steps(&doc);
        assert_eq!(steps.len(), 3, "one step per split + the label step");
        // Split 1: child 4, remainder past every id the history allocates (6).
        assert_eq!(steps[0]["unit_ids"], json!([0, 1, 2, 3]));
        assert_eq!(steps[0]["splits"][0]["unit_id"], 0);
        assert_eq!(steps[0]["splits"][0]["indices"], json!([[0]]));
        assert_eq!(steps[0]["splits"][0]["new_unit_ids"], json!([4, 6]));
        // Split 2: the same sorrel cluster, now unit 6 in SpikeInterface.
        assert_eq!(steps[1]["unit_ids"], json!([1, 2, 3, 4, 6]));
        assert_eq!(steps[1]["splits"][0]["unit_id"], 6);
        assert_eq!(steps[1]["splits"][0]["new_unit_ids"], json!([5, 7]));
        // The label travels under the child's final id.
        let entry = steps[2]["manual_labels"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["unit_id"] == 4)
            .expect("child labelled");
        assert_eq!(entry["labels"]["quality"], json!(["good"]));
        assert_eq!(entry["labels"]["origin"], json!(["Sorrel[Manual]"]));
        assert_eq!(steps[2]["unit_ids"], json!([1, 2, 3, 4, 5, 7]));
    }

    #[test]
    fn split_of_a_merge_target_opens_a_new_step() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(0)],
            target: ClusterId(1),
        })
        .unwrap();
        s.dispatch(split(1, 0)).unwrap();
        let doc = curation_document(&s, label_str);
        let steps = steps(&doc);
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0]["merges"][0]["unit_ids"], json!([0, 1]));
        assert_eq!(steps[1]["unit_ids"], json!([1, 2, 3]));
        assert_eq!(steps[1]["splits"][0]["unit_id"], 1);
        assert_eq!(steps[1]["splits"][0]["new_unit_ids"], json!([4, 5]));
        assert_eq!(steps[2]["unit_ids"], json!([2, 3, 4, 5]));
    }

    #[test]
    fn labels_only_history_stays_a_single_dict() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(2),
            op: PhyLabelOp::SetNoise,
        })
        .unwrap();
        s.dispatch(CurationCommand::Tag {
            cluster: ClusterId(2),
            tags: vec!["review:low-conf".into()],
        })
        .unwrap();
        let doc = curation_document(&s, label_str);
        assert!(doc.is_object(), "labels-only stays one Curation dict");
        assert_eq!(doc["unit_ids"], json!([0, 1, 2, 3]));
        assert_eq!(doc["merges"], json!([]));
        assert_eq!(doc["splits"], json!([]));
        assert_eq!(
            doc["manual_labels"][2]["labels"]["quality"],
            json!(["noise"])
        );
        assert_eq!(
            doc["manual_labels"][2]["labels"]["tags"],
            json!(["review:low-conf"])
        );
    }
}

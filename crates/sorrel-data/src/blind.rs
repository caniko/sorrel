//! Double-blind bench inputs (`sorrel blind`): the sorter's own output with
//! every trace of curation removed, and the provenance of the curation that
//! becomes ground truth.

use crate::command::{Actor, CurationCommand, Method};
use crate::journal::baseline_hash;
use crate::session::Session;
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use sorrel_io::npy::read_1d_u32;
use sorrel_io::{DataProvider, PhyParams, RootView};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

/// The sorter's own files a blind input may hold, copied verbatim. The
/// export skill's allowlist minus the human's `cluster_group.tsv`;
/// `spike_clusters.npy` and `params.py` are written separately.
const SORTER_FILES: [&str; 14] = [
    "spike_times.npy",
    "spike_templates.npy",
    "templates.npy",
    "channel_positions.npy",
    "channel_map.npy",
    "amplitudes.npy",
    "template_ind.npy",
    "pc_features.npy",
    "pc_feature_ind.npy",
    "template_features.npy",
    "template_feature_ind.npy",
    "similar_templates.npy",
    "channel_shanks.npy",
    "whitening_mat_inv.npy",
];

/// The sorter's own per-cluster TSVs. They are root state (the human started
/// from them: `cluster_KSLabel.tsv` seeds the labels when there is no
/// `cluster_group.tsv`), but phy rewrites them on save with rows for the
/// clusters the human created, so they are written filtered to the root ids.
const SORTER_TSVS: [&str; 3] = [
    "cluster_KSLabel.tsv",
    "cluster_ContamPct.tsv",
    "cluster_Amplitude.tsv",
];

/// Files of at least this size are hard-linked into a blind input, smaller
/// ones copied: a link shares the inode, so an in-place write through either
/// name would reach the other.
const LINK_MIN: u64 = 64 << 20;

/// Names a blind input never holds: the journals, every other `cluster_*`
/// file (labels, tags, notes, metrics), the curation export and sorrel's own
/// directories (auto sidecar, snapshot, caches, MCP state), phy's log, which
/// lists every merge and split the human made, and editor/atomic-save
/// leftovers. A defence-in-depth post-check: the input is built from an
/// allowlist.
pub fn is_denied(name: &str) -> bool {
    (name.starts_with("cluster_") && !SORTER_TSVS.contains(&name))
        || name.contains(".journal")
        || name.starts_with("curation.json")
        || name.ends_with(".sqlite")
        || name.ends_with(".lock")
        || name.ends_with(".tmp")
        || name.ends_with(".bak")
        || matches!(
            name,
            "phy.log"
                | "sorrel_auto"
                | ".phy"
                | "mcp"
                | ".sorrel"
                | "sorrel_original"
                | "sorrel_rewrite.pending"
        )
}

/// Write the blind input for the phy directory `dir` into `out` (which must
/// be empty or absent) and check it: `spike_clusters.npy` hashes to the
/// journal `baseline`, nothing is denied and nothing is a symlink. Only the
/// sorter's own files are copied (`RootView`, so a saved-over file comes from
/// the snapshot), and the raw recording lands inside `out` under its own name
/// (`params.py` is pointed at it), so the input never reaches back into `dir`.
/// `dat` is the `--dat` override of the open. `clusters_from` names the root
/// file written as `spike_clusters.npy`: itself for a sorrel session,
/// `spike_templates.npy` for a phy-curated sort. Returns the file names written.
pub fn write_blind_input(
    dir: &Path,
    dat: Option<&Path>,
    out: &Path,
    clusters_from: &str,
    baseline: u64,
) -> Result<Vec<String>> {
    create_empty(out)?;
    let view = RootView::open(dir)?;
    let mut written = vec![];

    let clusters = out.join("spike_clusters.npy");
    link_or_copy(&view.path(clusters_from), &clusters)?;
    written.push("spike_clusters.npy".to_owned());
    let got = baseline_hash(
        &std::fs::read(&clusters).with_context(|| format!("read {}", clusters.display()))?,
    );
    ensure!(
        got == baseline,
        "blind spike_clusters.npy hashes to {got:016x}, the journal baseline is {baseline:016x}"
    );
    let root_ids: HashSet<u32> = read_1d_u32(&clusters)?.into_iter().collect();

    for name in SORTER_FILES {
        let src = view.path(name);
        if src.is_file() {
            link_or_copy(&src, &out.join(name))?;
            written.push(name.to_owned());
        }
    }
    for name in SORTER_TSVS {
        let src = view.path(name);
        if src.is_file() {
            let text = std::fs::read_to_string(&src)?;
            let mut lines = text.lines();
            let mut kept = lines.next().map(|h| format!("{h}\n")).unwrap_or_default();
            let mut covered = HashSet::new();
            for l in lines {
                let id = l.split('\t').next().and_then(|c| c.trim().parse().ok());
                if let Some(id) = id.filter(|id| root_ids.contains(id)) {
                    covered.insert(id);
                    kept.push_str(l);
                    kept.push('\n');
                }
            }
            // phy rewrites these on save with rows for the clusters that exist
            // then, so the edited roots lose their rows: a partial file tells the
            // arm which roots the human left alone. Only a file covering every
            // root is the sorter's own.
            if covered == root_ids {
                std::fs::write(out.join(name), kept)?;
                written.push(name.to_owned());
            }
        }
    }

    // The raw recording, resolved like the provider does (relative to `dir`).
    let params = view.path("params.py");
    let params_dat = match params.exists() {
        true => PhyParams::read(&params)?.dat_path,
        false => None,
    };
    let dat_rel = dat
        .map(Path::to_path_buf)
        .or_else(|| params_dat.clone())
        .unwrap_or_else(|| PathBuf::from("recording.dat"));
    let mut dat_abs = dir.join(&dat_rel); // an absolute `dat_rel` replaces `dir`
                                          // A params.py written on another machine: its dat_path is gone, but the
                                          // recording sits beside it.
    if let Some(beside) = dat_rel.file_name().map(|n| dir.join(n)) {
        if !dat_abs.exists() && beside.is_file() {
            dat_abs = beside;
        }
    }
    let mut raw_name = None;
    if dat_abs.is_file() {
        // The recording and its `.ch` (mtscomp) / `.meta` (SpikeGLX) sidecars.
        for p in [
            dat_abs.clone(),
            dat_abs.with_extension("ch"),
            dat_abs.with_extension("meta"),
        ] {
            if let (true, Some(n)) = (p.is_file(), p.file_name()) {
                link_or_copy(&p, &out.join(n))?;
                written.push(n.to_string_lossy().into_owned());
            }
        }
        raw_name = dat_abs
            .file_name()
            .map(|n| n.to_string_lossy().into_owned());
    }
    if params.exists() {
        let mut text = std::fs::read_to_string(&params)?;
        // Point the copy at the raw by its bare name (the old line goes too: it
        // names a path in the source).
        // Rewrite even when the resolved basename is already recording.dat:
        // the source can omit dat_path or use a foreign singleton-list form.
        // Leaving that source line behind leaks a path or defeats intake.
        if let Some(n) = raw_name {
            text = text
                .lines()
                .filter(|l| !l.trim_start().starts_with("dat_path"))
                .map(|l| format!("{l}\n"))
                .collect();
            text.push_str(&format!("dat_path = \"{n}\"\n"));
        }
        std::fs::write(out.join("params.py"), text)?;
        written.push("params.py".to_owned());
    }

    for e in std::fs::read_dir(out)? {
        let e = e?;
        let name = e.file_name().to_string_lossy().into_owned();
        ensure!(
            !is_denied(&name) && !e.file_type()?.is_symlink(),
            "blind input holds {name}, which must never be copied"
        );
    }
    Ok(written)
}

/// The bench pair for a sort curated in phy rather than sorrel. phy never
/// rewrites `spike_templates.npy`, so it is the sorter's clustering (true for
/// an uncurated Kilosort export, where it equals `spike_clusters.npy`); the
/// top-level `spike_clusters.npy` and `cluster_group.tsv` are the human's.
/// Reads `dir` only: no session is opened, so nothing is written beside it.
pub fn write_phy_curated_pair(
    dir: &Path,
    dat: Option<&Path>,
    input: &Path,
    truth: &Path,
) -> Result<Vec<String>> {
    let templates = dir.join("spike_templates.npy");
    let baseline = baseline_hash(
        &std::fs::read(&templates).with_context(|| format!("read {}", templates.display()))?,
    );
    create_empty(truth)?;
    let files = write_blind_input(dir, dat, input, "spike_templates.npy", baseline)?;
    for name in ["spike_clusters.npy", "cluster_group.tsv"] {
        if dir.join(name).is_file() {
            link_or_copy(&dir.join(name), &truth.join(name))?;
        }
    }
    // Sorrel's own leftovers mean the human may have been primed by a machine.
    let sorrel_made = std::fs::read_dir(dir)?.flatten().any(|e| {
        let n = e.file_name().to_string_lossy().into_owned();
        n.contains(".journal") || n == "sorrel_original" || n == "sorrel_auto"
    });
    // Curated = the human's clustering or labels differ from the sorter's.
    let clusters = dir.join("spike_clusters.npy");
    let same_clusters = !clusters.is_file() || read_1d_u32(&clusters)? == read_1d_u32(&templates)?;
    let rows = |p: PathBuf| -> Result<Vec<String>> {
        Ok(std::fs::read_to_string(p)?
            .lines()
            .skip(1)
            .map(str::to_owned)
            .collect())
    };
    let (group, ks) = (
        dir.join("cluster_group.tsv"),
        dir.join("cluster_KSLabel.tsv"),
    );
    let same_labels = !group.is_file() || (ks.is_file() && rows(group)? == rows(ks)?);
    let curated = !(same_clusters && same_labels);
    if !curated {
        eprintln!("warning: {} looks uncurated: spike_clusters equals spike_templates and the labels are the sorter's", dir.display());
    }
    // ponytail: phy keeps no actor history, so purity is assumed unless sorrel's
    // own files are present; an automatic curation done elsewhere can't be told apart.
    std::fs::write(
        truth.join("provenance.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "source": "phy",
            "pure_manual": !sorrel_made,
            "anchored": sorrel_made,
            "curated": curated,
        }))?,
    )?;
    Ok(files)
}

fn create_empty(dir: &Path) -> Result<()> {
    if dir.exists() {
        ensure!(
            std::fs::read_dir(dir)?.next().is_none(),
            "{} is not empty",
            dir.display()
        );
    }
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))
}

pub fn link_or_copy(src: &Path, dst: &Path) -> Result<()> {
    // A link to a symlink would be one (and leads back into the source): link
    // what it names.
    let src = src.canonicalize()?;
    // ponytail: only the big per-spike arrays and the raw are linked; sorrel's own
    // saves are atomic renames, so a link to them is safe from it. A tool that
    // writes them in place would reach the source (copy everything if one ever does).
    if src.metadata()?.len() >= LINK_MIN && std::fs::hard_link(&src, dst).is_ok() {
        return Ok(());
    }
    std::fs::copy(&src, dst).with_context(|| format!("copy {}", src.display()))?;
    Ok(())
}

/// Who made the curation that is the ground truth, as `provenance.json`.
#[derive(Clone, Debug, Serialize)]
pub struct Provenance {
    /// Nodes on the applied path (head back to the root) by actor kind:
    /// `human`, `agent`, `algo`, `remote`.
    pub by_actor: BTreeMap<String, u32>,
    /// Applied edits by method (`Batch` expanded): `Split.by` for a split,
    /// else the actor's (`Human`/`Remote` = manual, `Algo` = algo, `Agent` =
    /// llm). Keys: `manual`, `algo`, `llm`.
    pub by_method: BTreeMap<String, u32>,
    /// Every applied edit is `Actor::Human` with `Method::Manual` (label
    /// edits included). True for an empty path.
    pub pure_manual: bool,
    /// Any algo, LLM or agent node anywhere in the history (applied or not):
    /// the human may have started from, or been primed by, a machine verdict.
    pub anchored: bool,
    /// xxh3-64 of the journal file, hex.
    pub journal_xxh3: String,
    /// Nodes in the whole tree, and on the applied path.
    pub n_nodes: usize,
    pub n_applied: usize,
    /// The checked-out node (`None` = the root).
    pub head: Option<u64>,
    /// A save of exactly the head is journaled.
    pub sealed: bool,
}

fn actor_kind(a: &Actor) -> &'static str {
    match a {
        Actor::Human => "human",
        Actor::Agent { .. } => "agent",
        Actor::Algo { .. } => "algo",
        Actor::Remote { .. } => "remote",
    }
}

fn method_kind(m: &Method) -> &'static str {
    match m {
        Method::Manual => "manual",
        Method::Algo(_) => "algo",
        Method::Llm(_) => "llm",
    }
}

/// Method of every leaf edit in `cmd` (a `Batch` expands to its children).
fn leaf_methods(actor: &Actor, cmd: &CurationCommand, out: &mut Vec<&'static str>) {
    match cmd {
        CurationCommand::Batch { children } => {
            children.iter().for_each(|c| leaf_methods(actor, c, out))
        }
        CurationCommand::Split { by, .. } => out.push(method_kind(by)),
        _ => out.push(match actor {
            Actor::Human | Actor::Remote { .. } => "manual",
            Actor::Algo { .. } => "algo",
            Actor::Agent { .. } => "llm",
        }),
    }
}

pub fn provenance<P: DataProvider>(session: &Session<P>) -> Result<Provenance> {
    let (mut by_actor, mut by_method) = (BTreeMap::new(), BTreeMap::new());
    let (mut pure_manual, mut anchored) = (true, false);
    let mut applied = vec![];
    let mut at = session.head();
    while let Some(id) = at {
        let (parent, ..) = session.node(id).context("head path names a missing node")?;
        applied.push(id);
        at = parent;
    }
    for id in &applied {
        let (_, actor, _, cmd) = session.node(*id).unwrap();
        *by_actor.entry(actor_kind(actor).to_owned()).or_insert(0) += 1;
        pure_manual &= *actor == Actor::Human;
        let mut methods = vec![];
        leaf_methods(actor, cmd, &mut methods);
        for m in methods {
            *by_method.entry(m.to_owned()).or_insert(0) += 1;
            pure_manual &= m == "manual";
        }
    }
    for id in 0..session.n_nodes() as u64 {
        let (_, actor, _, cmd) = session.node(id).unwrap();
        let mut methods = vec![];
        leaf_methods(actor, cmd, &mut methods);
        anchored |= matches!(actor, Actor::Agent { .. } | Actor::Algo { .. })
            || methods.iter().any(|m| *m != "manual");
    }
    let journal = std::fs::read(session.journal_path())
        .with_context(|| format!("read {}", session.journal_path().display()))?;
    Ok(Provenance {
        by_actor,
        by_method,
        pure_manual,
        anchored,
        journal_xxh3: format!("{:016x}", baseline_hash(&journal)),
        n_nodes: session.n_nodes(),
        n_applied: applied.len(),
        head: session.head(),
        sealed: session.seals().iter().any(|(n, _)| *n == session.head()),
    })
}

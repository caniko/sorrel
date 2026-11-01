//! Double-blind bench inputs (`sorrel blind`): the sorter's own output with
//! every trace of curation removed, and the provenance of the curation that
//! becomes ground truth.

use crate::command::{Actor, CurationCommand, Method};
use crate::journal::baseline_hash;
use crate::session::Session;
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use sorrel_io::{DataProvider, PhyParams, RootView};
use std::collections::BTreeMap;
#[cfg(unix)]
use std::os::unix::fs::symlink;
#[cfg(windows)]
use std::os::windows::fs::symlink_file as symlink;
use std::path::{Path, PathBuf};

/// The sorter's own per-cluster TSVs. They are root state (the human started
/// from them: `cluster_KSLabel.tsv` seeds the labels when there is no
/// `cluster_group.tsv`), so the arm gets them too, resolved through `RootView`.
const SORTER_TSVS: [&str; 3] = [
    "cluster_KSLabel.tsv",
    "cluster_ContamPct.tsv",
    "cluster_Amplitude.tsv",
];

/// Names a blind input never holds: the journals, every other `cluster_*`
/// file (labels, tags, notes, metrics), the curation export and sorrel's own
/// directories (auto sidecar, snapshot, caches, MCP state), plus phy's log,
/// which lists every merge and split the human made.
pub fn is_denied(name: &str) -> bool {
    (name.starts_with("cluster_") && !SORTER_TSVS.contains(&name))
        || name.contains(".journal")
        || name.ends_with(".sqlite")
        || name.ends_with(".lock")
        || matches!(
            name,
            "curation.json"
                | "phy.log"
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
/// journal `baseline` and no denied name exists. Files are the root's
/// (`RootView`, so a saved-over file comes from the snapshot), hard-linked
/// where possible and copied otherwise; the raw recording is symlinked.
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

    // The raw recording, resolved like the provider does (relative to `dir`).
    let params = view.path("params.py");
    let mut dat_rel = match (dat, params.exists()) {
        (Some(d), _) => Some(d.to_path_buf()),
        (None, true) => PhyParams::read(&params)?.dat_path,
        _ => None,
    }
    .unwrap_or_else(|| PathBuf::from("recording.dat"));
    let mut dat_abs = dir.join(&dat_rel); // an absolute `dat_rel` replaces `dir`
                                          // A params.py written on another machine: its dat_path is gone, but the
                                          // recording sits beside it. Point the copy at that file.
    let relocated = !dat_abs.exists()
        && dat_rel
            .file_name()
            .is_some_and(|n| dir.join(n).is_file() && n != dat_rel.as_os_str());
    if relocated {
        dat_rel = PathBuf::from(dat_rel.file_name().unwrap());
        dat_abs = dir.join(&dat_rel);
    }
    // ponytail: only a bare file name in `dir` keeps its relative reference;
    // anything else gets an absolute `dat_path` appended to the copied params.py.
    let in_dir = dat_rel.components().count() == 1;
    let mut raw = vec![];
    if in_dir && dat_abs.exists() {
        raw.push(dat_rel.to_string_lossy().into_owned());
        if dat_abs
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("cbin"))
        {
            raw.push(
                dat_abs
                    .with_extension("ch")
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }

    let mut written = vec![];
    for (name, path) in view.files() {
        if is_denied(&name) || !path.is_file() || name == "spike_clusters.npy" {
            continue;
        }
        let (src, dst) = (path.canonicalize()?, out.join(&name));
        if raw.contains(&name) {
            symlink(&src, &dst)?;
        } else if name == "params.py" && (!in_dir || relocated) {
            let target = match in_dir {
                true => dat_rel.clone(),
                false => dat_abs.canonicalize().unwrap_or(dat_abs.clone()),
            };
            let mut text = std::fs::read_to_string(&src)?;
            text.push_str(&format!("\ndat_path = \"{}\"\n", target.display()));
            std::fs::write(&dst, text)?;
        } else {
            link_or_copy(&src, &dst)?;
        }
        written.push(name);
    }
    link_or_copy(&view.path(clusters_from), &out.join("spike_clusters.npy"))?;
    written.push("spike_clusters.npy".into());

    let clusters = out.join("spike_clusters.npy");
    let got = baseline_hash(
        &std::fs::read(&clusters).with_context(|| format!("read {}", clusters.display()))?,
    );
    ensure!(
        got == baseline,
        "blind spike_clusters.npy hashes to {got:016x}, the journal baseline is {baseline:016x}"
    );
    for e in std::fs::read_dir(out)? {
        let name = e?.file_name().to_string_lossy().into_owned();
        if is_denied(&name) {
            bail!("blind input holds {name}, which must never be copied");
        }
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
    let files = write_blind_input(dir, dat, input, "spike_templates.npy", baseline)?;
    create_empty(truth)?;
    for name in ["spike_clusters.npy", "cluster_group.tsv"] {
        if dir.join(name).is_file() {
            link_or_copy(&dir.join(name), &truth.join(name))?;
        }
    }
    // ponytail: phy keeps no actor history, so purity is assumed; a sort that
    // went through an automatic curation before phy can't be told apart.
    std::fs::write(
        truth.join("provenance.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "source": "phy",
            "pure_manual": true,
            "anchored": false,
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

fn link_or_copy(src: &Path, dst: &Path) -> Result<()> {
    if std::fs::hard_link(src, dst).is_err() {
        std::fs::copy(src, dst).with_context(|| format!("copy {}", src.display()))?;
    }
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

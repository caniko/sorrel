//! `sorrel serve`: a JSON-RPC 2.0 server in the shape MCP clients speak
//! (`initialize`, `tools/list`, `tools/call`, `prompts/*`), so an agent or a
//! notebook can inspect metrics, ask for suggestions, see plots and curate.
//!
//! One JSON object per line in, one per line out, hand-rolled on
//! `serde_json`: a handful of synchronous tools do not justify an async
//! runtime. Two transports share [`handle_line`]:
//!
//! - **stdio, headless** ([`serve`], `Mode::Headless`): the agent edits its
//!   own session through the journaled commands, each edit tagged `src:agent`.
//! - **GUI listener** ([`spawn_listener`], `Mode::Attached`): the running GUI
//!   exposes its session on a loopback socket. The agent can read, move the
//!   human's selection and `propose`, but never edits: proposals queue for a
//!   human to accept. `sorrel mcp attach` ([`attach`]) bridges stdio to it.
//!
//! Plots come back as SVG resources (see `report`); PNG needs a GPU
//! surface and is deliberately out of scope.

use crate::command::{CurationCommand, Method};
use crate::proposals::{self, Action, Outcome, Proposal, Source};
use crate::qc_export::{collect_qc_rows, QcRow};
use crate::report::render_cluster_svgs;
use crate::save::save_session;
use crate::session::{ApplyPhyLabel, Session};
use crate::suggest::{
    rank_merge_candidates, rank_split_candidates, MergeCandidate, SplitCandidate, SuggestConfig,
};
use anyhow::Result;
use serde_json::{json, Value};
use sorrel_io::{ClusterId, DataProvider, HasGeometry};
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, RwLock};

/// MCP protocol revision this server answers `initialize` with.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// The curation rubric: what a careful human curator decides. The `triage`
/// prompt serves it, and `sorrel auto` hands it to `sorrel_auto.py` (env
/// `SORREL_RUBRIC`) as the `--llm` system prompt.
pub const RUBRIC: &str = "You review automatically sorted extracellular units (Kilosort 4, \
SpikeInterface metrics) and decide what a careful human curator would decide. \
Two classifiers disagreed or were unsure about the case you get; their outputs \
are included. Judge the evidence, not the classifiers' authority.

Units: label each as
- \"good\": a well-isolated single neuron. Clean biphasic somatic waveform \
(trough then peak, 0.2-1.0 ms trough-to-peak), spatially compact footprint \
decaying over ~50-100 um, empty refractory period (ACG near zero in the first \
1-2 ms, isi_violations_ratio and rp_contamination low), stable firing and \
amplitude across the recording (presence_ratio high), amplitude_cutoff low.
- \"mua\": neural but not isolated. Neural waveform, but refractory violations, \
an amplitude distribution cut off by the detection threshold, or a mixture of \
neurons (unstable amplitude, broad or multi-peaked footprint).
- \"noise\": not neural. Non-physiological shape (too many peaks, very short or \
very long, flat baseline violations), footprint spread uniformly across many \
distant channels or confined to one channel with no decay, line-noise or \
artefact-driven firing, or near-zero amplitude.

Merges: decide whether the listed units are one neuron split by the sorter. \
Merge when templates on the shared channel are nearly identical in shape (an \
amplitude difference alone is fine: drift), units are close (< ~50 um), and the \
cross-correlogram shows a refractory dip at 0 ms like an ACG. Do not merge when \
the CCG has a central peak (synchrony between two neurons) or no dip, or the \
shapes differ.

Amplitudes are in uV when the input says so, otherwise in ADC units. Give a \
confidence in [0, 1] (0.5 = coin flip) and at most three short reasons.";

const TRIAGE_INSTRUCTIONS: &str =
    "You are triaging a spike sort in the Sorrel GUI that a human is \
watching. Walk it like this:
1. Call attention_queue to see which clusters need a decision first.
2. For each cluster you look at, inspect it with metrics, render, split_proposals, \
merge_candidates and features (select moves the human's selection so they see what you discuss; \
selection tells you what they have selected).
3. Call propose for each decision (relabel, merge, split or accept) with a confidence in [0, 1] \
and at most three short reasons.
Never assume a proposal was applied: propose only queues it, and the human accepts or rejects \
each one. Check selection or list_clusters again before relying on the current state.

Judge with this rubric:

";

/// What the listener forwards to the GUI thread.
#[derive(Debug)]
pub enum AgentMsg {
    /// Stage a proposal in the review pane (already logged as pending).
    Propose(Proposal),
    /// Replace the GUI selection with these clusters.
    Select(Vec<ClusterId>),
}

/// The GUI's current selection, for the `selection` tool. The GUI must write
/// it whenever it changes (cheapest: once per frame).
pub type SharedSelection = Arc<Mutex<Vec<ClusterId>>>;

/// Everything attached mode needs besides the session.
pub struct Attached {
    pub save_dir: PathBuf,
    pub msgs: Sender<AgentMsg>,
    pub selection: SharedSelection,
    /// Wake the GUI (a repaint) after queueing a message.
    pub on_change: Arc<dyn Fn() + Send + Sync>,
}

pub enum Mode {
    /// Direct journaled edits (`sorrel serve`).
    Headless,
    /// Read-only plus proposals (the GUI listener).
    Attached(Attached),
}

/// Per-connection state: where to save, how to name labels, and caches keyed
/// on `Session::version()`.
pub struct Env<'a, F> {
    pub save_dir: &'a Path,
    pub reseal_on_save: bool,
    pub label_str: F,
    qc: Option<(u64, Vec<QcRow>)>,
    suggest: Option<(u64, Vec<MergeCandidate>, Vec<SplitCandidate>)>,
    /// MCP `clientInfo.name` from `initialize`; headless splits are named after it.
    client: Option<String>,
}

impl<'a, F> Env<'a, F> {
    pub fn new(save_dir: &'a Path, reseal_on_save: bool, label_str: F) -> Self {
        Self {
            save_dir,
            reseal_on_save,
            label_str,
            qc: None,
            suggest: None,
            client: None,
        }
    }

    fn capture_client(&mut self, line: &str) {
        let Ok(req) = serde_json::from_str::<Value>(line) else {
            return;
        };
        if req["method"] == "initialize" {
            if let Some(name) = req
                .pointer("/params/clientInfo/name")
                .and_then(Value::as_str)
            {
                self.client = Some(name.to_string());
            }
        }
    }
}

fn read_tools() -> Vec<Value> {
    let cluster = json!({"type": "object", "properties": {"cluster": {"type": "integer"}}, "required": ["cluster"]});
    let limit = |d: u32| json!({"type": "object", "properties": {"limit": {"type": "integer", "default": d}}});
    vec![
        json!({"name": "list_clusters", "description": "Cluster ids with spike count, label and tags; optional label filter and tag-prefix filter (\"review:\" = the review queue)",
         "inputSchema": {"type": "object", "properties": {"label": {"type": "string"}, "tag": {"type": "string"}}}}),
        json!({"name": "metrics", "description": "Quality metrics for one cluster (same columns as cluster_qc.tsv)",
         "inputSchema": cluster}),
        json!({"name": "suggest", "description": "Ranked merge and split suggestions for the current curation",
         "inputSchema": {"type": "object", "properties": {}}}),
        json!({"name": "render", "description": "Template waveform, ISI histogram and autocorrelogram of a cluster as SVG",
         "inputSchema": cluster}),
        json!({"name": "evidence", "description": "What `sorrel auto` recorded for a cluster: classifier outputs, Bombcell failures, Claude verdicts and merge records",
         "inputSchema": cluster}),
        json!({"name": "attention_queue", "description": "Clusters ranked by how much they need a decision, with reasons",
         "inputSchema": limit(20)}),
        json!({"name": "split_proposals", "description": "Split candidates for a cluster with the spikes each would move out and per-part stats",
         "inputSchema": cluster}),
        json!({"name": "merge_candidates", "description": "Merge candidates involving a cluster with their evidence (CCG dip, amplitude KS)",
         "inputSchema": cluster}),
        json!({"name": "features", "description": "PC feature values of n evenly spaced spikes of a cluster on its best channels",
         "inputSchema": {"type": "object", "properties": {"cluster": {"type": "integer"}, "n": {"type": "integer", "default": 20}}, "required": ["cluster"]}}),
    ]
}

/// Tool catalogue of the headless server (`tools/list`).
pub fn tool_list() -> Value {
    let mut tools = read_tools();
    tools.extend([
        json!({"name": "relabel", "description": "Set a cluster's label (good | mua | noise | unsorted); journaled and undoable",
         "inputSchema": {"type": "object", "properties": {"cluster": {"type": "integer"}, "label": {"type": "string"}}, "required": ["cluster", "label"]}}),
        json!({"name": "merge", "description": "Merge source clusters into target; journaled and undoable",
         "inputSchema": {"type": "object", "properties": {"sources": {"type": "array", "items": {"type": "integer"}}, "target": {"type": "integer"}}, "required": ["sources", "target"]}}),
        json!({"name": "split", "description": "Move the given spike indices (positions in the cluster's time-sorted spikes) into a new cluster",
         "inputSchema": {"type": "object", "properties": {"cluster": {"type": "integer"}, "spike_idx": {"type": "array", "items": {"type": "integer"}}}, "required": ["cluster", "spike_idx"]}}),
        json!({"name": "tag", "description": "Replace a cluster's tag set (provenance/review markers, e.g. drop review:* once decided); journaled and undoable",
         "inputSchema": {"type": "object", "properties": {"cluster": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["cluster", "tags"]}}),
        json!({"name": "undo", "description": "Undo the last curation command", "inputSchema": {"type": "object", "properties": {}}}),
        json!({"name": "save", "description": "Write spike_clusters.npy, cluster_group.tsv and curation.json", "inputSchema": {"type": "object", "properties": {}}}),
    ]);
    #[cfg(feature = "phy-export")]
    tools.push(json!(
        {"name": "export_phy", "description": "Write the curated session as a complete phy directory at out (params.py references the raw file unless copy_raw)",
         "inputSchema": {"type": "object", "properties": {"out": {"type": "string"}, "copy_raw": {"type": "boolean"}}, "required": ["out"]}}
    ));
    Value::Array(tools)
}

/// Tool catalogue of the GUI listener: reads, `select`, `selection`, `propose`.
pub fn attached_tool_list() -> Value {
    let mut tools = read_tools();
    tools.extend([
        json!({"name": "propose", "description": "Stage a curation proposal for the human to accept or reject; nothing is applied. action: relabel (clusters=[c], label) | merge (clusters, first is the target) | split (clusters=[c], spike_idx) | accept (clusters=[c]). confidence in [0,1]; at most 3 reasons.",
         "inputSchema": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["relabel", "merge", "split", "accept"]},
            "clusters": {"type": "array", "items": {"type": "integer"}},
            "label": {"type": "string"},
            "spike_idx": {"type": "array", "items": {"type": "integer"}},
            "confidence": {"type": "number"},
            "reasons": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
          "required": ["action", "clusters", "confidence", "reasons"]}}),
        json!({"name": "select", "description": "Move the human's GUI selection to these clusters so they see what you are discussing",
         "inputSchema": {"type": "object", "properties": {"clusters": {"type": "array", "items": {"type": "integer"}}}, "required": ["clusters"]}}),
        json!({"name": "selection", "description": "The clusters the human has selected in the GUI right now",
         "inputSchema": {"type": "object", "properties": {}}}),
    ]);
    Value::Array(tools)
}

fn prompts_list() -> Value {
    json!({"prompts": [{"name": "triage",
        "description": "Walk the attention queue, inspect each cluster and propose curation decisions for the human to accept",
        "arguments": []}]})
}

fn prompt_get(name: &str) -> Option<Value> {
    (name == "triage").then(|| {
        json!({"description": "Triage a spike sort", "messages": [{"role": "user",
            "content": {"type": "text", "text": format!("{TRIAGE_INSTRUCTIONS}{RUBRIC}")}}]})
    })
}

/// Serve headless until `input` closes. `reseal_on_save` follows the same
/// rule as the GUI: true only for phy directories.
pub fn serve<P, F>(
    session: &mut Session<P>,
    save_dir: &Path,
    reseal_on_save: bool,
    label_str: F,
    input: impl BufRead,
    mut output: impl Write,
) -> Result<()>
where
    P: DataProvider + ApplyPhyLabel + HasGeometry,
    F: Fn(&P::Label) -> &'static str + Sync,
{
    let mut env = Env::new(save_dir, reseal_on_save, label_str);
    for line in input.lines() {
        if let Some(reply) = handle_line(session, &mut env, &Mode::Headless, &line?) {
            write_json(&mut output, &reply)?;
        }
    }
    Ok(())
}

/// One request line to its reply; `None` for blank lines and notifications.
/// `Headless` edits `session` directly; `Attached` only reads it (and queues
/// proposals), see [`handle_line_attached`] for the read-locked entry point.
pub fn handle_line<P, F>(
    session: &mut Session<P>,
    env: &mut Env<F>,
    mode: &Mode,
    line: &str,
) -> Option<Value>
where
    P: DataProvider + ApplyPhyLabel + HasGeometry,
    F: Fn(&P::Label) -> &'static str + Sync,
{
    match mode {
        Mode::Headless => {
            env.capture_client(line);
            respond(line, false, |name, args| {
                call_tool(session, env, name, args)
            })
        }
        Mode::Attached(a) => handle_line_attached(session, env, a, line),
    }
}

/// [`handle_line`] for `Mode::Attached`, which never needs a write lock.
pub fn handle_line_attached<P, F>(
    session: &Session<P>,
    env: &mut Env<F>,
    attached: &Attached,
    line: &str,
) -> Option<Value>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str + Sync,
{
    respond(line, true, |name, args| {
        call_attached(session, env, attached, name, args)
    })
}

/// The JSON-RPC envelope shared by both modes.
fn respond(
    line: &str,
    attached: bool,
    call: impl FnOnce(&str, &Value) -> Result<Value, ToolError>,
) -> Option<Value> {
    if line.trim().is_empty() {
        return None;
    }
    let req: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return Some(rpc_error(Value::Null, -32700, &e.to_string())),
    };
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
    let params = req.get("params").cloned().unwrap_or(Value::Null);
    if id.is_null() {
        return None; // notification (e.g. notifications/initialized)
    }
    let reply = match method {
        "initialize" => Ok(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}, "prompts": {}},
            "serverInfo": {"name": "sorrel", "version": env!("CARGO_PKG_VERSION")},
        })),
        "ping" => Ok(json!({})),
        "tools/list" => {
            Ok(json!({"tools": if attached { attached_tool_list() } else { tool_list() }}))
        }
        "prompts/list" => Ok(prompts_list()),
        "prompts/get" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            prompt_get(name).ok_or_else(|| (-32602, format!("unknown prompt {name}")))
        }
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            match call(name, &args) {
                Ok(content) => Ok(json!({"content": content, "isError": false})),
                Err(ToolError::Unknown) => Err((-32601, format!("unknown tool {name}"))),
                Err(ToolError::Failed(msg)) => {
                    Ok(json!({"content": [{"type": "text", "text": msg}], "isError": true}))
                }
            }
        }
        other => Err((-32601, format!("unknown method {other}"))),
    };
    Some(match reply {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
        Err((code, message)) => rpc_error(id, code, &message),
    })
}

enum ToolError {
    Unknown,
    Failed(String),
}

impl From<anyhow::Error> for ToolError {
    fn from(e: anyhow::Error) -> Self {
        ToolError::Failed(format!("{e:#}"))
    }
}

fn fail<T>(msg: impl Into<String>) -> Result<T, ToolError> {
    Err(ToolError::Failed(msg.into()))
}

fn text(v: Value) -> Value {
    json!([{"type": "text", "text": v.to_string()}])
}

fn arg_u32(args: &Value, key: &str) -> Result<u32, ToolError> {
    args.get(key)
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| ToolError::Failed(format!("missing or invalid integer argument {key:?}")))
}

fn arg_u32_list(args: &Value, key: &str) -> Result<Vec<u32>, ToolError> {
    args.get(key)
        .and_then(Value::as_array)
        .and_then(|a| {
            a.iter()
                .map(|v| v.as_u64().and_then(|v| u32::try_from(v).ok()))
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| ToolError::Failed(format!("missing or invalid integer list {key:?}")))
}

fn check<P: DataProvider>(session: &Session<P>, c: u32) -> Result<ClusterId, ToolError> {
    if c < session.n_clusters() {
        Ok(ClusterId(c))
    } else {
        fail(format!("cluster {c} out of range"))
    }
}

/// `cmd` as an agent edit: the touched clusters first gain `src:agent`, in
/// the same undoable batch.
fn as_agent<P: DataProvider>(session: &Session<P>, cmd: CurationCommand) -> CurationCommand {
    let mut clusters = match &cmd {
        CurationCommand::Relabel { cluster, .. } | CurationCommand::Split { cluster, .. } => {
            vec![*cluster]
        }
        CurationCommand::Merge { sources, target } => [&[*target][..], sources].concat(),
        _ => Vec::new(),
    };
    clusters.sort_unstable();
    clusters.dedup();
    let mut children: Vec<CurationCommand> = clusters
        .into_iter()
        .filter(|&c| !session.tags(c).iter().any(|t| t == "src:agent"))
        .map(|cluster| {
            let mut tags = session.tags(cluster).to_vec();
            tags.push("src:agent".into());
            CurationCommand::Tag { cluster, tags }
        })
        .collect();
    if children.is_empty() {
        return cmd;
    }
    children.push(cmd);
    CurationCommand::Batch { children }
}

fn call_tool<P, F>(
    session: &mut Session<P>,
    env: &mut Env<F>,
    name: &str,
    args: &Value,
) -> Result<Value, ToolError>
where
    P: DataProvider + ApplyPhyLabel + HasGeometry,
    F: Fn(&P::Label) -> &'static str + Sync,
{
    Ok(match name {
        "tag" => {
            let cluster = check(session, arg_u32(args, "cluster")?)?;
            let mut tags: Vec<String> = args
                .get("tags")
                .and_then(Value::as_array)
                .and_then(|a| a.iter().map(|v| v.as_str().map(str::to_owned)).collect())
                .ok_or_else(|| ToolError::Failed("missing string list \"tags\"".into()))?;
            if let Some(bad) = tags.iter().find(|t| !sorrel_io::kilosort::valid_tag(t)) {
                return fail(format!("invalid tag {bad:?}"));
            }
            if !tags.iter().any(|t| t == "src:agent") {
                tags.push("src:agent".into());
            }
            session.dispatch(CurationCommand::Tag { cluster, tags })?;
            text(json!({"ok": true, "version": session.version()}))
        }
        "relabel" => {
            let cluster = check(session, arg_u32(args, "cluster")?)?;
            let label = args.get("label").and_then(Value::as_str).unwrap_or("");
            let Some(op) = proposals::parse_label(label) else {
                return fail(format!("unknown label {label:?}"));
            };
            session.dispatch(as_agent(session, CurationCommand::Relabel { cluster, op }))?;
            text(json!({"ok": true, "version": session.version()}))
        }
        "merge" => {
            let sources: Vec<ClusterId> = arg_u32_list(args, "sources")?
                .into_iter()
                .map(|c| check(session, c))
                .collect::<Result<_, _>>()?;
            let target = check(session, arg_u32(args, "target")?)?;
            if sources.contains(&target) {
                return fail(format!("target {} is also listed in sources", target.0));
            }
            session.dispatch(as_agent(
                session,
                CurationCommand::Merge { sources, target },
            ))?;
            text(json!({"ok": true, "version": session.version()}))
        }
        "split" => {
            let cluster = check(session, arg_u32(args, "cluster")?)?;
            let spike_idx = arg_u32_list(args, "spike_idx")?;
            session.dispatch(as_agent(
                session,
                CurationCommand::Split {
                    cluster,
                    spike_idx,
                    new_cluster: ClusterId(u32::MAX),
                    by: Method::Llm(env.client.clone().unwrap_or_else(|| "agent".into())),
                },
            ))?;
            let new_cluster = session.n_clusters() - 1;
            text(json!({"ok": true, "new_cluster": new_cluster, "version": session.version()}))
        }
        "undo" => {
            session.dispatch(CurationCommand::Undo)?;
            text(json!({"ok": true, "version": session.version()}))
        }
        "save" => {
            save_session(session, env.save_dir, env.reseal_on_save, &env.label_str)?;
            text(json!({"ok": true, "save_dir": env.save_dir.display().to_string()}))
        }
        #[cfg(feature = "phy-export")]
        "export_phy" => {
            let out = args
                .get("out")
                .and_then(Value::as_str)
                .ok_or_else(|| ToolError::Failed("missing string argument \"out\"".into()))?;
            let copy_raw = args
                .get("copy_raw")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            crate::phy_export::export_phy(session, Path::new(out), copy_raw, &env.label_str)?;
            text(json!({"ok": true, "out": out}))
        }
        _ => return call_read(session, env, name, args),
    })
}

fn call_attached<P, F>(
    session: &Session<P>,
    env: &mut Env<F>,
    a: &Attached,
    name: &str,
    args: &Value,
) -> Result<Value, ToolError>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str + Sync,
{
    Ok(match name {
        "propose" => {
            let p = parse_proposal(session, args)?;
            let (id, queued) = (p.id, p.clone());
            proposals::append(&a.save_dir, &p)
                .map_err(|e| ToolError::Failed(format!("audit log: {e}")))?;
            if a.msgs.send(AgentMsg::Propose(queued)).is_err() {
                let _ = proposals::record_outcome(&a.save_dir, id, Outcome::Superseded);
                return fail("the GUI is no longer listening");
            }
            (a.on_change)();
            text(json!({"ok": true, "id": id, "status": "queued for human review; NOT applied"}))
        }
        "select" => {
            let clusters: Vec<ClusterId> = arg_u32_list(args, "clusters")?
                .into_iter()
                .map(|c| check(session, c))
                .collect::<Result<_, _>>()?;
            if a.msgs.send(AgentMsg::Select(clusters)).is_err() {
                return fail("the GUI is no longer listening");
            }
            (a.on_change)();
            text(json!({"ok": true}))
        }
        "selection" => {
            let sel = a.selection.lock().unwrap_or_else(|p| p.into_inner());
            text(json!({"clusters": sel.iter().map(|c| c.0).collect::<Vec<_>>()}))
        }
        "relabel" | "merge" | "split" | "tag" | "undo" | "save" | "export_phy" => {
            return fail("edits are disabled while the GUI is attached: use `propose`")
        }
        _ => return call_read(session, env, name, args),
    })
}

/// Validate `propose` arguments against the session; nothing is changed.
fn parse_proposal<P: DataProvider>(
    session: &Session<P>,
    args: &Value,
) -> Result<Proposal, ToolError> {
    let live = |c: u32| -> Result<ClusterId, ToolError> {
        let c = check(session, c)?;
        if session.spike_times(c).is_empty() {
            return fail(format!("cluster {} is empty", c.0));
        }
        Ok(c)
    };
    let clusters: Vec<ClusterId> = arg_u32_list(args, "clusters")?
        .into_iter()
        .map(live)
        .collect::<Result<_, _>>()?;
    let mut distinct = clusters.clone();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() != clusters.len() {
        return fail("clusters must be distinct");
    }
    let one = || match clusters[..] {
        [c] => Ok(c),
        _ => fail("this action takes exactly one cluster"),
    };
    let action = match args.get("action").and_then(Value::as_str).unwrap_or("") {
        "relabel" => {
            let label = args.get("label").and_then(Value::as_str).unwrap_or("");
            if proposals::parse_label(label).is_none() {
                return fail(format!("unknown label {label:?}"));
            }
            Action::Relabel {
                cluster: one()?,
                label: label.to_owned(),
            }
        }
        "merge" if clusters.len() >= 2 => Action::Merge { clusters },
        "merge" => return fail("merge needs at least two clusters"),
        "split" => {
            let cluster = one()?;
            let mut spike_idx = arg_u32_list(args, "spike_idx")?;
            spike_idx.sort_unstable();
            spike_idx.dedup();
            let n = session.spike_times(cluster).len();
            if spike_idx.is_empty()
                || spike_idx.len() >= n
                || spike_idx.last().is_some_and(|&i| i as usize >= n)
            {
                return fail(format!(
                    "spike_idx must be a proper non-empty subset of 0..{n}"
                ));
            }
            Action::Split { cluster, spike_idx }
        }
        "accept" => Action::Accept { cluster: one()? },
        other => return fail(format!("unknown action {other:?}")),
    };
    let confidence = args
        .get("confidence")
        .and_then(Value::as_f64)
        .filter(|c| (0.0..=1.0).contains(c))
        .ok_or_else(|| ToolError::Failed("confidence must be a number in [0, 1]".into()))?;
    let reasons: Vec<String> = args
        .get("reasons")
        .and_then(Value::as_array)
        .and_then(|a| a.iter().map(|v| v.as_str().map(str::to_owned)).collect())
        .filter(|r: &Vec<String>| r.len() <= 3)
        .ok_or_else(|| ToolError::Failed("reasons must be at most 3 strings".into()))?;
    Ok(Proposal::new(
        Source::Agent,
        action,
        confidence as f32,
        reasons,
        args.clone(),
    ))
}

/// Suggester output for every pair/cluster (not only the top K), cached per
/// session version. ponytail: the rankers scan the whole session, so the
/// first call after an edit costs a full suggest pass; per-cluster entry
/// points would avoid it.
fn suggestions<'a, P: DataProvider, F>(
    session: &Session<P>,
    env: &'a mut Env<F>,
) -> (&'a [MergeCandidate], &'a [SplitCandidate]) {
    let version = session.version();
    if env.suggest.as_ref().is_none_or(|(v, ..)| *v != version) {
        let cfg = SuggestConfig {
            top_merges: usize::MAX,
            top_splits: usize::MAX,
            ..SuggestConfig::default()
        };
        env.suggest = Some((
            version,
            rank_merge_candidates(session, &cfg),
            rank_split_candidates(session, &cfg),
        ));
    }
    let (_, m, s) = env.suggest.as_ref().expect("just filled");
    (m, s)
}

/// Mean amplitude and count of a part of a cluster (local spike indices).
fn part_stats(amps: &[f32], idx: impl Iterator<Item = usize>) -> Value {
    let v: Vec<f32> = idx.filter_map(|i| amps.get(i).copied()).collect();
    let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
    json!({"n_spikes": v.len(), "mean_amplitude": mean})
}

fn call_read<P, F>(
    session: &Session<P>,
    env: &mut Env<F>,
    name: &str,
    args: &Value,
) -> Result<Value, ToolError>
where
    P: DataProvider,
    F: Fn(&P::Label) -> &'static str + Sync,
{
    let label_str = &env.label_str;
    Ok(match name {
        "list_clusters" => {
            let want = args.get("label").and_then(Value::as_str);
            let tag = args.get("tag").and_then(Value::as_str);
            let rows: Vec<Value> = (0..session.n_clusters())
                .map(ClusterId)
                .filter(|&c| !session.spike_times(c).is_empty())
                .filter(|&c| tag.is_none_or(|t| session.tags(c).iter().any(|x| x.starts_with(t))))
                .filter_map(|c| {
                    let label = label_str(&session.label(c)?);
                    (want.is_none_or(|w| w == label)).then(|| {
                        json!({"cluster": c.0, "n_spikes": session.spike_times(c).len(),
                               "label": label, "tags": session.tags(c),
                               "origin": session.origin_label(c),
                               "parents": session.parents(c).iter().map(|p| p.0).collect::<Vec<_>>(),
                               "note": session.note(c)})
                    })
                })
                .collect();
            text(json!(rows))
        }
        "evidence" => {
            let c = check(session, arg_u32(args, "cluster")?)?;
            let uid = u64::from(session.provider.unit_id(c));
            let path = env.save_dir.join("sorrel_auto").join("evidence.json");
            let doc: Value = std::fs::read_to_string(&path)
                .ok()
                .and_then(|t| serde_json::from_str(&t).ok())
                .ok_or_else(|| {
                    ToolError::Failed(format!("no {} (run `sorrel auto` first)", path.display()))
                })?;
            let has = |v: &Value| {
                v.as_array()
                    .is_some_and(|a| a.iter().any(|m| m.as_u64() == Some(uid)))
            };
            // Merged units list their members; match those before the id key,
            // which a split can reuse for a different unit.
            let unit = doc["units"].as_object().and_then(|units| {
                units
                    .values()
                    .find(|v| has(&v["members"]))
                    .or_else(|| units.get(&uid.to_string()))
                    .cloned()
            });
            let merges: Vec<&Value> = doc["merges"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|m| has(&m["units"]))
                .collect();
            text(json!({"unit": unit, "merges": merges, "tags": session.tags(c)}))
        }
        "metrics" => {
            let c = check(session, arg_u32(args, "cluster")?)?;
            let version = session.version();
            if env.qc.as_ref().is_none_or(|(v, _)| *v != version) {
                env.qc = Some((version, collect_qc_rows(session)));
            }
            let row = env
                .qc
                .as_ref()
                .and_then(|(_, rows)| rows.iter().find(|r| r.cluster_id == c.0))
                .ok_or_else(|| ToolError::Failed(format!("no metrics for cluster {}", c.0)))?;
            text(row.to_json())
        }
        "suggest" => {
            let cfg = SuggestConfig::default();
            text(json!({
                "merges": rank_merge_candidates(session, &cfg),
                "splits": rank_split_candidates(session, &cfg),
            }))
        }
        "render" => {
            let c = check(session, arg_u32(args, "cluster")?)?;
            let svgs = render_cluster_svgs(session, c);
            let res = |name: &str, svg: String| {
                json!({"type": "resource", "resource": {
                    "uri": format!("sorrel://cluster/{}/{name}.svg", c.0),
                    "mimeType": "image/svg+xml", "text": svg}})
            };
            let mut content = Vec::new();
            if let Some(wf) = svgs.waveform {
                content.push(res("template", wf));
            }
            content.push(res("isi", svgs.isi));
            content.push(res("acg", svgs.acg));
            Value::Array(content)
        }
        "attention_queue" => {
            let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(20) as usize;
            let (merges, splits) = suggestions(session, env);
            // Same scorer as `Session::attention`, with the suggestions the
            // env already caches.
            let rows: Vec<Value> = crate::proposers::attention_all(session, merges, splits)
                .into_iter()
                .filter(|r| r.1 > 0.0)
                .take(limit)
                .map(|(c, score, why)| {
                    json!({"cluster": c.0, "attention": score, "reasons": why,
                    "n_spikes": session.spike_times(c).len(), "tags": session.tags(c)})
                })
                .collect();
            text(json!(rows))
        }
        "split_proposals" => {
            let c = check(session, arg_u32(args, "cluster")?)?;
            let amps = session.spike_amplitudes(c);
            let n = session.spike_times(c).len();
            // TODO(lead): the k-means / outlier proposers (p2-stats) join
            // `rank_split_candidates`; their `method` field arrives with them.
            let out: Vec<Value> = suggestions(session, env)
                .1
                .iter()
                .filter(|s| s.cluster == c)
                .map(|s| {
                    let minor: std::collections::HashSet<usize> = s
                        .bipartition_minor_idx
                        .iter()
                        .map(|&i| i as usize)
                        .collect();
                    let mut v = serde_json::to_value(s).unwrap_or(Value::Null);
                    v["parts"] = json!([
                        part_stats(amps, s.bipartition_minor_idx.iter().map(|&i| i as usize)),
                        part_stats(amps, (0..n).filter(|i| !minor.contains(i))),
                    ]);
                    v
                })
                .collect();
            text(json!(out))
        }
        "merge_candidates" => {
            let c = check(session, arg_u32(args, "cluster")?)?;
            let out: Vec<&MergeCandidate> = suggestions(session, env)
                .0
                .iter()
                .filter(|m| m.a == c || m.b == c)
                .collect();
            text(json!(out))
        }
        "features" => {
            let c = check(session, arg_u32(args, "cluster")?)?;
            if !session.has_pc_features() {
                return fail("this dataset has no PC features");
            }
            let n = args.get("n").and_then(Value::as_u64).unwrap_or(20).min(200) as usize;
            let channels: Vec<u32> = session.best_channels(&[c]).into_iter().take(3).collect();
            let n_pcs = session.pc_shape().0.min(3);
            let globals: Vec<u32> = session
                .spike_clusters_global()
                .iter()
                .enumerate()
                .filter(|(_, &k)| k == c)
                .map(|(i, _)| i as u32)
                .collect();
            let times = session.spike_times(c);
            let rows: Vec<Value> = (0..n.min(globals.len()))
                .map(|k| {
                    let local = k * globals.len() / n.min(globals.len());
                    let pcs: Vec<Vec<Option<f32>>> = channels
                        .iter()
                        .map(|&ch| {
                            (0..n_pcs)
                                .map(|pc| session.pc_feature(globals[local], ch, pc))
                                .collect()
                        })
                        .collect();
                    json!({"idx": local, "sample": times.get(local).map(|t| t.0), "pc": pcs})
                })
                .collect();
            text(json!({"cluster": c.0, "channels": channels, "n_pcs": n_pcs, "spikes": rows}))
        }
        _ => return Err(ToolError::Unknown),
    })
}

fn rpc_error(id: Value, code: i32, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

fn write_json(out: &mut impl Write, v: &Value) -> Result<()> {
    serde_json::to_writer(&mut *out, v)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

// --- GUI listener and stdio bridge ------------------------------------

const MCP_JSON: &str = ".sorrel/mcp.json";

/// A fresh 128-bit hex token. std-only: `RandomState` keys come from the OS
/// RNG (once per process), mixed with the clock and a per-call counter via
/// SipHash. ponytail: not a CSPRNG; swap in `getrandom` if the token ever
/// guards more than a loopback socket in a user-only directory.
fn random_token() -> String {
    use std::hash::BuildHasher;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let word = |salt: u8| {
        std::collections::hash_map::RandomState::new().hash_one((salt, nanos, std::process::id()))
    };
    format!("{:016x}{:016x}", word(0), word(1))
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// `<save_dir>/.sorrel/mcp.json` = `{port, token}`, mode 0600, written
/// atomically (tmp + rename) so a client never reads half of it.
fn write_mcp_json(save_dir: &Path, port: u16, token: &str) -> std::io::Result<PathBuf> {
    let path = save_dir.join(MCP_JSON);
    std::fs::create_dir_all(path.parent().expect("has parent"))?;
    let tmp = path.with_extension("json.tmp");
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600);
    let mut f = opts.open(&tmp)?;
    f.write_all(json!({"port": port, "token": token}).to_string().as_bytes())?;
    f.sync_all()?;
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

/// Keeps the listener alive; dropping it removes `mcp.json` and stops
/// accepting (connections already open finish on their own).
pub struct ListenerHandle {
    port: u16,
    token: String,
    mcp_json: PathBuf,
    stop: Arc<std::sync::atomic::AtomicBool>,
}

impl ListenerHandle {
    pub fn port(&self) -> u16 {
        self.port
    }
    pub fn token(&self) -> &str {
        &self.token
    }
}

impl Drop for ListenerHandle {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        let _ = std::fs::remove_file(&self.mcp_json);
        // Wake the blocking accept so its thread sees `stop`.
        let _ = std::net::TcpStream::connect(("127.0.0.1", self.port));
    }
}

/// Expose `session` to MCP clients on `127.0.0.1:<ephemeral>` (Attached
/// mode). Proposals and selection requests go to `msgs`; the GUI should
/// drain it each frame and keep `selection` current. `on_change` wakes the
/// GUI after a message is queued (e.g. `ctx.request_repaint()`).
///
/// Each request holds the session read lock for the length of the tool call
/// (attached mode never writes). ponytail: one thread per connection with no
/// cap; every connection must present the token first, so only local
/// processes that can read the 0600 `mcp.json` get a thread's worth of work.
pub fn spawn_listener<P>(
    session: Arc<RwLock<Session<P>>>,
    save_dir: &Path,
    label_str: fn(&P::Label) -> &'static str,
    msgs: Sender<AgentMsg>,
    selection: SharedSelection,
    on_change: impl Fn() + Send + Sync + 'static,
) -> std::io::Result<ListenerHandle>
where
    P: DataProvider + 'static,
    Session<P>: Send + Sync,
{
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    let token = random_token();
    let mcp_json = write_mcp_json(save_dir, port, &token)?;
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let attached = Arc::new(Attached {
        save_dir: save_dir.to_owned(),
        msgs,
        selection,
        on_change: Arc::new(on_change),
    });
    let handle = ListenerHandle {
        port,
        token: token.clone(),
        mcp_json,
        stop: stop.clone(),
    };
    std::thread::Builder::new()
        .name("sorrel-mcp".into())
        .spawn(move || {
            for stream in listener.incoming() {
                if stop.load(std::sync::atomic::Ordering::SeqCst) {
                    break;
                }
                let Ok(stream) = stream else { continue };
                let (session, attached, token) = (session.clone(), attached.clone(), token.clone());
                std::thread::spawn(move || {
                    if let Err(e) = serve_client(stream, &token, &session, &attached, label_str) {
                        log::debug!("mcp client ended: {e}");
                    }
                });
            }
        })?;
    Ok(handle)
}

fn serve_client<P: DataProvider>(
    stream: std::net::TcpStream,
    token: &str,
    session: &RwLock<Session<P>>,
    attached: &Attached,
    label_str: fn(&P::Label) -> &'static str,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
    let mut reader = std::io::BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    // The first line must be {"token": "..."}; anything else drops the client.
    let mut first = String::new();
    reader.by_ref().take(512).read_line(&mut first)?;
    let sent = serde_json::from_str::<Value>(&first)
        .ok()
        .and_then(|v| v["token"].as_str().map(str::to_owned));
    if !sent.is_some_and(|t| ct_eq(t.as_bytes(), token.as_bytes())) {
        return Ok(());
    }
    writer.set_read_timeout(None)?; // shared socket: clears the reader's timeout too
    writer.write_all(b"{\"ok\":true}\n")?;
    let mut env = Env::new(&attached.save_dir, false, label_str);
    for line in reader.lines() {
        let line = line?;
        let reply = {
            let s = session.read().unwrap_or_else(|p| p.into_inner());
            handle_line_attached(&s, &mut env, attached, &line)
        };
        if let Some(reply) = reply {
            write_json(&mut writer, &reply).map_err(std::io::Error::other)?;
        }
    }
    Ok(())
}

/// `{port, token}` of a running GUI listener in `save_dir`, if any.
pub fn read_mcp_json(save_dir: &Path) -> Option<(u16, String)> {
    let v: Value =
        serde_json::from_str(&std::fs::read_to_string(save_dir.join(MCP_JSON)).ok()?).ok()?;
    Some((
        u16::try_from(v["port"].as_u64()?).ok()?,
        v["token"].as_str()?.to_owned(),
    ))
}

/// `sorrel mcp attach`: pipe `input` to the GUI listener of `save_dir` and
/// its replies to `output`, line by line, until either side closes. Returns
/// `Ok(false)` without touching either stream when no listener answers (no
/// `mcp.json`, refused connection, rejected token): the caller then falls
/// back to a headless `serve`.
pub fn attach(
    save_dir: &Path,
    input: impl Read + Send + 'static,
    mut output: impl Write,
) -> Result<bool> {
    use std::net::{TcpStream, ToSocketAddrs};
    use std::time::Duration;
    let Some((port, token)) = read_mcp_json(save_dir) else {
        return Ok(false);
    };
    let Some(addr) = ("127.0.0.1", port)
        .to_socket_addrs()
        .ok()
        .and_then(|mut a| a.next())
    else {
        return Ok(false);
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_secs(2)) else {
        return Ok(false);
    };
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    write_json(&mut stream, &json!({"token": token}))?;
    let mut reader = std::io::BufReader::new(stream.try_clone()?);
    let mut ack = String::new();
    if reader.read_line(&mut ack).unwrap_or(0) == 0 {
        return Ok(false);
    }
    stream.set_read_timeout(None)?;
    let mut up = stream.try_clone()?;
    std::thread::spawn(move || {
        let mut input = std::io::BufReader::new(input);
        let mut line = String::new();
        while matches!(input.read_line(&mut line), Ok(n) if n > 0) {
            if up
                .write_all(line.as_bytes())
                .and_then(|()| up.flush())
                .is_err()
            {
                return;
            }
            line.clear();
        }
        let _ = up.shutdown(std::net::Shutdown::Write); // stdin closed: let the server finish and hang up
    });
    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        output.write_all(line.as_bytes())?;
        output.flush()?;
        line.clear();
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::Journal;
    use crate::qc_export::tests::MockProvider;
    use sorrel_io::SampleIndex;
    use std::io::Cursor;

    impl HasGeometry for MockProvider {
        fn channel_positions(&self) -> &[[f32; 2]] {
            &[]
        }
    }

    #[test]
    fn list_relabel_merge_and_save_through_the_rpc_loop() {
        let prov = MockProvider {
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)],
                vec![SampleIndex(20), SampleIndex(200)],
                vec![SampleIndex(100)],
            ],
            amps: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0], vec![6.0]],
            n_samples: SampleIndex(1000),
        };
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open(&dir.path().join("j.journal")).unwrap();
        let mut s = Session::new(prov, journal);
        s.seed_amplitudes();
        let label = |l: &u8| match l {
            1 => "good",
            2 => "mua",
            3 => "noise",
            _ => "unsorted",
        };

        let input = concat!(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
            "\n",
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"relabel","arguments":{"cluster":0,"label":"good"}}}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"merge","arguments":{"sources":[1],"target":2}}}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"list_clusters","arguments":{}}}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"metrics","arguments":{"cluster":0}}}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"render","arguments":{"cluster":0}}}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"relabel","arguments":{"cluster":99,"label":"good"}}}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"nope","arguments":{}}}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"save","arguments":{}}}"#,
            "\n",
        );
        let mut out = Vec::new();
        serve(
            &mut s,
            dir.path(),
            false,
            label,
            Cursor::new(input),
            &mut out,
        )
        .unwrap();

        let replies: Vec<Value> = String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(replies.len(), 10, "notification gets no reply");
        assert_eq!(replies[0]["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(
            replies[1]["result"]["tools"].as_array().unwrap().len(),
            15 + usize::from(cfg!(feature = "phy-export"))
        );
        assert_eq!(replies[2]["id"], 3);
        assert_eq!(replies[2]["result"]["isError"], false);
        assert_eq!(s.label(ClusterId(0)), Some(1));
        assert!(s.spike_times(ClusterId(1)).is_empty(), "merged away");

        let listed: Value =
            serde_json::from_str(replies[4]["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(
            listed.as_array().unwrap().len(),
            2,
            "empty cluster 1 hidden"
        );
        assert_eq!(listed[0]["label"], "good");

        let metrics: Value =
            serde_json::from_str(replies[5]["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(metrics["n_spikes"], 3.0);
        assert!(metrics["l_ratio"].is_null(), "NaN becomes null");

        assert_eq!(
            replies[6]["result"]["content"][0]["resource"]["mimeType"],
            "image/svg+xml"
        );
        assert_eq!(
            replies[7]["result"]["isError"], true,
            "out-of-range cluster"
        );
        assert_eq!(replies[8]["error"]["code"], -32601, "unknown tool");
        assert_eq!(replies[9]["result"]["isError"], false);
        assert!(dir.path().join("cluster_group.tsv").exists());
        assert!(dir.path().join("curation.json").exists());
    }

    #[test]
    fn review_queue_tag_and_evidence_tools() {
        let prov = MockProvider {
            spikes: vec![vec![SampleIndex(10)], vec![SampleIndex(20)]],
            amps: vec![vec![1.0], vec![2.0]],
            n_samples: SampleIndex(1000),
        };
        let dir = tempfile::tempdir().unwrap();
        let mut s = Session::new(prov, Journal::open(&dir.path().join("j.journal")).unwrap());
        std::fs::create_dir(dir.path().join("sorrel_auto")).unwrap();
        std::fs::write(
            dir.path().join("sorrel_auto/evidence.json"),
            r#"{"units":{"1":{"bombcell":"noise"},"7":{"members":[0,1],"bombcell":"mua"}},"merges":[{"units":[0,1],"verdict":null}]}"#,
        )
        .unwrap();
        let call = |id: u32, name: &str, args: &str| {
            format!(
                r#"{{"jsonrpc":"2.0","id":{id},"method":"tools/call","params":{{"name":"{name}","arguments":{args}}}}}"#
            ) + "\n"
        };
        let input = [
            call(
                1,
                "tag",
                r#"{"cluster":1,"tags":["review:merge","src:bombcell"]}"#,
            ),
            call(2, "list_clusters", r#"{"tag":"review:"}"#),
            call(3, "tag", r#"{"cluster":1,"tags":["a,b"]}"#),
            call(4, "evidence", r#"{"cluster":1}"#),
        ]
        .concat();
        let mut out = Vec::new();
        serve(
            &mut s,
            dir.path(),
            false,
            |_: &u8| "unsorted",
            Cursor::new(input),
            &mut out,
        )
        .unwrap();
        let replies: Vec<Value> = String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let text = |i: usize| -> Value {
            serde_json::from_str(replies[i]["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap()
        };
        assert_eq!(
            s.tags(ClusterId(1)),
            ["review:merge", "src:bombcell", "src:agent"]
        );
        assert_eq!(
            text(1),
            json!([{"cluster": 1, "n_spikes": 1, "label": "unsorted", "tags": ["review:merge", "src:bombcell", "src:agent"], "origin": "Auto[unknown]", "parents": [], "note": ""}])
        );
        assert_eq!(
            replies[2]["result"]["isError"], true,
            "comma would corrupt the TSV"
        );
        assert_eq!(text(3)["unit"]["bombcell"], "mua");
        assert_eq!(text(3)["merges"].as_array().unwrap().len(), 1);
    }

    fn mock_session(dir: &Path) -> Session<MockProvider> {
        let prov = MockProvider {
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)],
                vec![SampleIndex(20), SampleIndex(200)],
            ],
            amps: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]],
            n_samples: SampleIndex(1000),
        };
        let mut s = Session::new(prov, Journal::open(&dir.join("j.journal")).unwrap());
        s.seed_amplitudes();
        s
    }

    fn label(_: &u8) -> &'static str {
        "unsorted"
    }

    fn rpc(id: u32, method: &str, params: Value) -> String {
        json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string() + "\n"
    }

    fn call(id: u32, name: &str, args: Value) -> String {
        rpc(id, "tools/call", json!({"name": name, "arguments": args}))
    }

    fn payload(reply: &Value) -> Value {
        serde_json::from_str(reply["result"]["content"][0]["text"].as_str().unwrap()).unwrap()
    }

    fn attached(dir: &Path) -> (Attached, std::sync::mpsc::Receiver<AgentMsg>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let a = Attached {
            save_dir: dir.to_owned(),
            msgs: tx,
            selection: Arc::new(Mutex::new(vec![ClusterId(1)])),
            on_change: Arc::new(|| {}),
        };
        (a, rx)
    }

    #[test]
    fn attached_propose_queues_and_changes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = mock_session(dir.path());
        let (a, rx) = attached(dir.path());
        let mode = Mode::Attached(a);
        let mut env = Env::new(dir.path(), false, label);
        let version = s.version();
        let mut run = |line: String| handle_line(&mut s, &mut env, &mode, &line).unwrap();

        let ok = run(call(
            1,
            "propose",
            json!({"action": "merge", "clusters": [0, 1], "confidence": 0.9, "reasons": ["CCG dip"]}),
        ));
        assert_eq!(ok["result"]["isError"], false);
        let Some(AgentMsg::Propose(p)) = rx.try_recv().ok() else {
            panic!("proposal not queued")
        };
        assert_eq!(
            p.action,
            Action::Merge {
                clusters: vec![ClusterId(0), ClusterId(1)]
            }
        );
        assert_eq!(p.source, Source::Agent);
        assert_eq!(
            proposals::read_log(dir.path())[0].id,
            p.id,
            "audited as pending"
        );

        // Invalid arguments and edit tools are refused; nothing is queued.
        for bad in [
            call(
                2,
                "propose",
                json!({"action": "merge", "clusters": [0], "confidence": 0.9, "reasons": []}),
            ),
            call(
                3,
                "propose",
                json!({"action": "relabel", "clusters": [0], "label": "good", "confidence": 2, "reasons": []}),
            ),
            call(
                4,
                "propose",
                json!({"action": "split", "clusters": [0], "spike_idx": [0, 1, 2], "confidence": 0.5, "reasons": []}),
            ),
            call(5, "relabel", json!({"cluster": 0, "label": "good"})),
        ] {
            assert_eq!(run(bad)["result"]["isError"], true);
        }
        assert!(rx.try_recv().is_err());

        let sel = run(call(6, "selection", json!({})));
        assert_eq!(payload(&sel)["clusters"], json!([1]));
        run(call(7, "select", json!({"clusters": [0]})));
        assert!(matches!(rx.try_recv(), Ok(AgentMsg::Select(c)) if c == [ClusterId(0)]));
        let tools = run(rpc(8, "tools/list", json!({})));
        let names: Vec<&str> = tools["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"propose") && !names.contains(&"relabel"));
        let q = run(call(9, "attention_queue", json!({})));
        assert_eq!(q["result"]["isError"], false);
        drop(run);
        assert_eq!(s.version(), version, "session untouched");
        assert!(s.tags(ClusterId(0)).is_empty());
    }

    #[test]
    fn headless_merge_rejects_target_among_sources() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = mock_session(dir.path());
        let mut env = Env::new(dir.path(), false, label);
        let version = s.version();
        let r = handle_line(
            &mut s,
            &mut env,
            &Mode::Headless,
            &call(1, "merge", json!({"sources": [0, 1], "target": 0})),
        )
        .unwrap();
        assert_eq!(r["result"]["isError"], true);
        assert_eq!(s.version(), version);
        assert!(s.parents(ClusterId(0)).is_empty());
    }

    #[test]
    fn headless_relabel_tags_src_agent_and_triage_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = mock_session(dir.path());
        let mut env = Env::new(dir.path(), false, label);
        let mut run = |line: String| handle_line(&mut s, &mut env, &Mode::Headless, &line).unwrap();
        assert_eq!(
            run(call(1, "relabel", json!({"cluster": 0, "label": "good"})))["result"]["isError"],
            false
        );
        assert_eq!(
            run(call(2, "merge", json!({"sources": [1], "target": 0})))["result"]["isError"],
            false
        );

        let init = run(rpc(3, "initialize", json!({})));
        assert!(init["result"]["capabilities"]["prompts"].is_object());
        let list = run(rpc(4, "prompts/list", json!({})));
        assert_eq!(list["result"]["prompts"][0]["name"], "triage");
        let got = run(rpc(5, "prompts/get", json!({"name": "triage"})));
        let text = got["result"]["messages"][0]["content"]["text"]
            .as_str()
            .unwrap();
        assert!(text.contains("attention_queue"));
        assert!(text.contains("Never assume a proposal was applied"));
        assert!(text.ends_with(RUBRIC));
        assert_eq!(
            run(rpc(6, "prompts/get", json!({"name": "nope"})))["error"]["code"],
            -32602
        );
        drop(run);
        assert_eq!(s.label(ClusterId(0)), Some(1));
        assert!(s.tags(ClusterId(0)).iter().any(|t| t == "src:agent"));
        // One undo reverts the relabel's tag and label together.
        s.dispatch(CurationCommand::Undo).unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        assert!(s.tags(ClusterId(0)).is_empty());
    }

    #[test]
    fn headless_split_origin_names_the_mcp_client() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = mock_session(dir.path());
        let mut env = Env::new(dir.path(), false, label);
        let mut run = |line: String| handle_line(&mut s, &mut env, &Mode::Headless, &line).unwrap();
        // No clientInfo yet: the split is attributed to "agent".
        assert_eq!(
            run(call(1, "split", json!({"cluster": 0, "spike_idx": [0]})))["result"]["isError"],
            false
        );
        run(rpc(
            2,
            "initialize",
            json!({"clientInfo": {"name": "codex"}}),
        ));
        assert_eq!(
            run(call(3, "split", json!({"cluster": 0, "spike_idx": [0]})))["result"]["isError"],
            false
        );
        drop(run);
        let n = s.n_clusters();
        assert_eq!(s.origin_label(ClusterId(n - 2)), "Sorrel[LLM:agent]");
        assert_eq!(s.origin_label(ClusterId(n - 1)), "Sorrel[LLM:codex]");
    }

    #[test]
    fn loopback_listener_round_trip_and_bad_token() {
        let dir = tempfile::tempdir().unwrap();
        let s = Arc::new(RwLock::new(mock_session(dir.path())));
        let (tx, rx) = std::sync::mpsc::channel();
        let sel: SharedSelection = Arc::new(Mutex::new(vec![ClusterId(1)]));
        let woke = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let w = woke.clone();
        let handle = spawn_listener(s.clone(), dir.path(), label, tx, sel, move || {
            w.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        })
        .unwrap();
        let (port, token) = read_mcp_json(dir.path()).unwrap();
        assert_eq!((port, token.as_str()), (handle.port(), handle.token()));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.path().join(MCP_JSON))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }

        // A client with the wrong token is dropped without a reply.
        let mut bad = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        bad.write_all(b"{\"token\":\"nope\"}\n").unwrap();
        let mut got = String::new();
        bad.read_to_string(&mut got).unwrap();
        assert!(got.is_empty());

        // The bridge pipes stdio to the same listener.
        let input = rpc(1, "initialize", json!({}))
            + &call(
                2,
                "propose",
                json!({"action": "accept", "clusters": [0], "confidence": 0.7, "reasons": ["clean"]}),
            )
            + &call(3, "relabel", json!({"cluster": 0, "label": "good"}));
        let mut out = Vec::new();
        assert!(attach(dir.path(), Cursor::new(input), &mut out).unwrap());
        let replies: Vec<Value> = String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(replies.len(), 3);
        assert_eq!(replies[1]["result"]["isError"], false);
        assert_eq!(
            replies[2]["result"]["isError"], true,
            "attached edits refused"
        );
        assert!(matches!(rx.try_recv(), Ok(AgentMsg::Propose(_))));
        assert_eq!(woke.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(s.read().unwrap().label(ClusterId(0)), Some(0), "unchanged");

        drop(handle);
        assert!(!dir.path().join(MCP_JSON).exists());
        let mut out = Vec::new();
        assert!(
            !attach(dir.path(), Cursor::new(""), &mut out).unwrap(),
            "no listener: caller falls back"
        );
    }
}

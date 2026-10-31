//! `sorrel serve`: a JSON-RPC 2.0 server in the shape MCP clients speak
//! (`initialize`, `tools/list`, `tools/call`, `prompts/*`), so an agent or a
//! notebook can inspect metrics, ask for suggestions, see plots and curate.
//!
//! One JSON object per line in, one per line out, hand-rolled on
//! `serde_json`: a handful of synchronous tools do not justify an async
//! runtime. One handler, [`handle_line`], serves every client of a
//! [`SessionHost`], over two transports:
//!
//! - **stdio** ([`serve`]): `sorrel mcp <dir>` when no host is running.
//! - **loopback TCP** ([`spawn_listener`]): a running host (`sorrel <dir>`,
//!   `sorrel --headless <dir>`) listens on `127.0.0.1`, behind a token in
//!   `<user_dir>/mcp/<hash>.json` (see `mcp_json_path`; [`list_hosts`] scans
//!   them); `sorrel mcp <dir>` bridges stdio to it ([`attach`]).
//!
//! Edit tools apply as `Actor::Agent` (tagged `src:agent`, journaled and
//! undoable) under [`AgentPolicy::Direct`], and only queue a proposal under
//! [`AgentPolicy::ProposeOnly`]. `propose` always queues. `select` and
//! `selection` need a GUI ([`Ui`]). Every host event caused by someone else
//! reaches the client as a `notifications/message` after its next reply.
//!
//! Plots come back as SVG resources (see `report`); PNG needs a GPU
//! surface and is deliberately out of scope.

use crate::brief::{render_brief, rle, BriefOpts};
use crate::command::{Actor, CurationCommand, Method};
use crate::host::{AgentPolicy, HostEvent, SessionHost};
use crate::proposals::{self, Action, Proposal, Source};
use crate::qc_export::{collect_qc_rows_with, QcRow};
use crate::rules::{apply_rules, eval_report, op_name, RuleSet};
use crate::save::save_session;
use crate::session::{ApplyPhyLabel, Session};
use crate::suggest::{
    rank_merge_candidates, rank_split_candidates, MergeCandidate, SplitCandidate, SuggestConfig,
};
use anyhow::{bail, Result};
use serde_json::{json, Value};
use sorrel_io::{ClusterId, DataProvider, HasGeometry};
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

/// MCP protocol revision this server answers `initialize` with.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// The curation rubric: what a careful human curator decides, stated in the
/// descriptor names of `brief` / `cluster_qc.tsv`. The `review` prompt serves
/// it, and `sorrel auto` hands it to `sorrel_auto.py` (env `SORREL_RUBRIC`)
/// as the `--llm` system prompt, so it must stand alone.
pub const RUBRIC: &str = "You review automatically sorted extracellular units (Kilosort 4, \
SpikeInterface metrics) and decide what a careful human curator would decide. \
Judge the evidence, not any classifier's authority. Evidence is numbers: \
thresholds are relative to the population quantiles in the brief header, so \
call a value high or low against that, not against a textbook.

Units: label each as
- \"good\": a well-isolated single neuron. Biphasic somatic waveform \
(trough_to_peak_ms 0.2-1.0, n_extrema 2-3, half_width_ms in the neural range), \
compact footprint (footprint_decay_um small, footprint_r80_um within a few \
channels), empty refractory period (rp_contamination and isi_violation_ratio \
low; rp_estimate_ms short; acg SAX string near zero in its first symbols), \
stable firing and amplitude (presence_ratio high, amp_t SAX string flat, \
amplitude_cutoff low, drift_slope small).
- \"mua\": neural but not isolated. Neural waveform, but refractory violations \
(rp_contamination high), an amplitude distribution cut off by the detection \
threshold (amplitude_cutoff high), a mixture of neurons (unstable amp_t, broad \
or multi-peaked footprint, low nn_isolation or d_prime), or large \
burst_index with a violated refractory period.
- \"noise\": not neural. Non-physiological shape (n_extrema too high, \
trough_to_peak_ms or half_width_ms far outside the neural range, flat baseline \
violations), footprint spread uniformly over many distant channels \
(footprint_decay_um near zero, large n_active_channels) or confined to one \
channel with no decay, line-noise or artefact-driven firing (periodic or \
bursty, high sync_z with many units), or near-zero amplitude_snr.

Merges: decide whether the listed units are one neuron split by the sorter. \
Merge when templates on the shared channels are nearly identical in shape (an \
amplitude difference alone is fine: drift), units are close (about 50 um or \
less), the presence complement is high (one fades as the other appears: a drift \
fragment), and the merged unit's rp_contamination stays low with a dip_z \
showing a refractory dip at 0 ms like an ACG. Do not merge when the CCG has a \
central peak (sync_z high: two synchronous neurons), no dip, the merged \
rp_contamination or amplitude_cutoff gets worse, or the shapes differ.

Amplitudes are in uV when the input says so, otherwise in ADC units. Give a \
confidence in [0, 1] (0.5 = coin flip) and at most three short reasons: flag \
codes, rule names or at most eight words.";

const REVIEW_INSTRUCTIONS: &str =
    "You are reviewing a spike sort in Sorrel. Review the policy, not every \
unit: your cost should depend on how many units are borderline, not on how many \
there are. Never transcribe spike indices; act on handles (s12.1, m3) and cluster \
ids. Protocol:
1. Call brief (level 1). Read the header: mode, population quantiles, flag \
legend, and the ledger of flagged units.
2. Write a RuleSet (TOML, thresholds from the descriptors and the header \
quantiles) that gives every unit a label, and call rules_eval. Refine the rules \
until margin units and disagreements with the auto classifiers are few.
3. Inspect only margin units, disagreement units and units with a +flag: \
brief {clusters:[..]} for cards, brief level 3 for relations (merge components \
m<k>, split candidates s<cluster>.<k>), eval for a counterfactual of a merge or a \
split. When the precomputed evidence does not settle a hypothesis, test it with \
probe (acg, ccg, compare, window, split by=amp|time|pc|kmeans, eval): one short \
answer per query, and probe split gives a new s handle you can decide on. \
select moves the human's selection so they see what you discuss.
4. Call rules_apply, then decide for the exceptions, the merges (by m handle, or \
by ids after eval) and the splits (by s handle). decide takes \
[target, action, confidence, reasons] items and applies them as one undo step, \
or queues one proposal per item when the human chose propose-only. Reasons are \
codes (flag codes or rule names), at most three; free text only as a last resort, \
at most eight words.
5. Call brief (level 1) again to confirm the sort now looks as intended, then \
stop. Save only if asked.
Check the policy: under propose-only nothing is applied until the human \
accepts it.

Judge with this rubric:

";

/// The GUI's current selection, for the `selection` tool. The GUI must write
/// it whenever it changes (cheapest: once per frame).
pub type SharedSelection = Arc<Mutex<Vec<ClusterId>>>;

/// What a GUI shares with MCP clients besides the host: its selection, and
/// a channel for `select` requests (wake the GUI with a host subscriber).
pub struct Ui {
    pub selection: SharedSelection,
    pub select: Sender<Vec<ClusterId>>,
    /// Wake the GUI after a `select` (e.g. `ctx.request_repaint()`).
    pub wake: Arc<dyn Fn() + Send + Sync>,
}

/// Per-connection state: where to save, how to name labels, the GUI link,
/// host events to forward, and caches keyed on `Session::version()`.
pub struct Env<'a, F> {
    pub save_dir: &'a Path,
    pub reseal_on_save: bool,
    pub label_str: F,
    ui: Option<Arc<Ui>>,
    events: Option<Receiver<HostEvent>>,
    /// Listed among the host's agents once `initialize` named the client.
    seat: Option<crate::host::AgentSeat>,
    qc: Option<(u64, Vec<QcRow>)>,
    suggest: Option<(u64, Vec<MergeCandidate>, Vec<SplitCandidate>)>,
    /// The rule set last given to `rules_eval` / `rules_apply`, as TOML and
    /// parsed; `brief` keeps its thresholds out of the quantiser's way.
    rules: Option<(String, RuleSet)>,
    /// `s12.1` / `m3` handles to suggestions; see [`crate::evaluate::Handles`].
    handles: crate::evaluate::Handles,
    /// `HostEvent::seq` of this connection's own edits, not yet seen on
    /// its event stream. Two clients can share a display name, so the
    /// name alone cannot tell whose edit an event is.
    own: std::collections::HashSet<u64>,
    /// MCP `clientInfo.name` from `initialize`; agent edits are named after it.
    client: Option<String>,
}

impl<'a, F> Env<'a, F> {
    pub fn new(save_dir: &'a Path, reseal_on_save: bool, label_str: F) -> Self {
        Self {
            save_dir,
            reseal_on_save,
            label_str,
            ui: None,
            events: None,
            seat: None,
            qc: None,
            suggest: None,
            rules: None,
            handles: Default::default(),
            client: None,
            own: Default::default(),
        }
    }

    /// Give `select` / `selection` a GUI to act on.
    pub fn with_ui(mut self, ui: Option<Arc<Ui>>) -> Self {
        self.ui = ui;
        self
    }

    /// Forward `host`'s events to this client (see [`Self::notifications`]).
    pub fn subscribe<P: DataProvider + ApplyPhyLabel>(mut self, host: &SessionHost<P>) -> Self {
        self.events = Some(host.subscribe(|| {}));
        self
    }

    /// `notifications/message` for every event since the last call that
    /// another connection caused (this client's own edits are in its replies).
    pub fn notifications(&mut self) -> Vec<Value> {
        let own = &mut self.own;
        self.events
            .iter()
            .flat_map(Receiver::try_iter)
            .filter(|ev| !own.remove(&ev.seq))
            .map(|ev| {
                json!({"jsonrpc": "2.0", "method": "notifications/message", "params": {
                    "level": "info", "logger": "sorrel",
                    "data": {"actor": actor_name(&ev.actor), "change": ev.desc, "node": ev.node,
                             "clusters": ev.clusters.iter().map(|c| c.0).collect::<Vec<_>>(),
                             "version": ev.version}}})
            })
            .collect()
    }

    /// Remember `ev` as this connection's own (it is in the reply already).
    fn mine(&mut self, ev: HostEvent) -> HostEvent {
        if self.events.is_some() && ev.seq != 0 {
            self.own.insert(ev.seq);
        }
        ev
    }

    fn agent(&self) -> Actor {
        Actor::Agent {
            client: self.client.clone().unwrap_or_else(|| "agent".into()),
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

/// Short name of an actor for event text: `you`, the agent's client, the
/// algorithm or the remote peer.
pub fn actor_name(a: &Actor) -> &str {
    match a {
        Actor::Human => "you",
        Actor::Agent { client } => client,
        Actor::Algo { name } => name,
        Actor::Remote { peer } => peer,
    }
}

fn read_tools() -> Vec<Value> {
    let cluster = json!({"type": "object", "properties": {"cluster": {"type": "integer"}}, "required": ["cluster"]});
    let limit = |d: u32| json!({"type": "object", "properties": {"limit": {"type": "integer", "default": d}}});
    vec![
        json!({"name": "list_clusters", "description": "Compact TSV of clusters (id n label tags origin parents note; - = empty); optional label filter and tag-prefix filter (\"review:\" = the review queue). For evidence use brief",
         "inputSchema": {"type": "object", "properties": {"label": {"type": "string"}, "tag": {"type": "string"}}}}),
        json!({"name": "metrics", "description": "Quality metrics of one cluster as JSON (same columns as cluster_qc.tsv). brief level 2 gives the same numbers for many units at once, quantised",
         "inputSchema": cluster}),
        json!({"name": "suggest", "description": "Merge components and split candidates for the current curation, each with a handle (m<k>, s<cluster>.<k>) and a one-line counterfactual; pass the handle as candidate to split/propose/eval",
         "inputSchema": {"type": "object", "properties": {}}}),
        json!({"name": "brief", "description": "Token-lean text review of the whole sort: header (population quantiles, flag legend), ledger of flagged units, and with level 2 or clusters per-unit cards (metrics, ACG/amplitude SAX strings, auto evidence). Numbers are quantised so flag decisions match the raw metrics. Level 3 adds relations: one line per merge component (handle m<k>) and split candidate (handle s<cluster>.<k>) with the counterfactual metrics of the result; act on them by handle (split/propose {candidate}), never by spike index",
         "inputSchema": {"type": "object", "properties": {"level": {"type": "integer", "default": 1}, "clusters": {"type": "array", "items": {"type": "integer"}}, "budget": {"type": "integer"}}}}),
        json!({"name": "rules_eval", "description": "Review a POLICY, not units: run a threshold rule set (TOML) on every unit and get only what needs attention, so review cost is O(rules + margin units), not O(units). Workflow: write rules -> rules_eval -> inspect margin and disagreement units with brief{clusters} / eval -> adjust rules -> rules_apply. Rules: default = \"good\"; [[rule]] metric, op (< <= > >= == !=, prefix | for absolute value), value, label (good|mua|noise|unsorted), optional name (reason code) and all = [{metric, op, value}] ANDed in; first match wins. Output: units decided per rule, label transitions vs current (ids RLE), margin units (a condition within `margin` x IQR of its threshold, closest first), and disagreements with `sorrel auto` Bombcell/UnitRefine labels. Without `rules` uses the last given or <save_dir>/.sorrel/review_rules.toml. Metric names are those of `metrics`; brief quantises around these thresholds",
         "inputSchema": {"type": "object", "properties": {"rules": {"type": "string"}, "margin": {"type": "number", "default": 0.1}}}}),
        json!({"name": "evidence", "description": "What `sorrel auto` recorded for a cluster: classifier outputs, Bombcell failures, Claude verdicts and merge records (brief cards carry a summary)",
         "inputSchema": cluster}),
        json!({"name": "attention_queue", "description": "Clusters ranked by how much they need a decision, with reasons (brief level 1 lists flagged units without a call per unit)",
         "inputSchema": limit(20)}),
        json!({"name": "split_proposals", "description": "Split candidates for a cluster: handle (s<cluster>.<k>, pass as candidate to split/propose), per-part stats and the counterfactual line",
         "inputSchema": cluster}),
        json!({"name": "merge_candidates", "description": "Merge candidates involving a cluster with their evidence (CCG dip, amplitude KS); brief level 3 shows all of them as components (m<k>)",
         "inputSchema": cluster}),
        json!({"name": "probe", "description": "Closed-grammar computations for hypotheses the brief did not anticipate, one compact answer per query. q = one string or a list. acg C [win=50] [bin=1] (refractory estimate, contamination, burst, SAX of log lags, first bin rates); ccg A B [win=50] [bin=1] (dip_z, sync_z, +/- lag asymmetry, SAX); compare A B (template cosine on shared channels, peak-channel distance um, trough-to-peak/half-width deltas, amplitude ratio); window C T0 T1 (seconds: n, rate, rpc, amplitude, presence of the slice vs the whole unit); split C by=amp|time|pc|kmeans [at=X] (creates a NEW split handle s<C>.<k> with its counterfactual line: minor part = spikes above/after X, default the GMM boundary / the biggest amplitude jump / PC1 sign; act on it with split {candidate}, decide or eval); eval merge A B .. | eval s12.1 | eval m3. Read-only except that split registers a handle",
         "inputSchema": {"type": "object", "properties": {"q": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}}, "required": ["q"]}}),
        json!({"name": "eval", "description": "Counterfactual of a merge or split without applying it: per-part and merged contamination, presence and cutoff, plus per-pair CCG dip/synchrony z and amplitude KS. {merge:[ids]} tests any merge; {candidate:\"s12.1\"|\"m3\"} a handle from brief level 3 or suggest",
         "inputSchema": {"type": "object", "properties": {"merge": {"type": "array", "items": {"type": "integer"}}, "candidate": {"type": "string"}}}}),
    ]
}

/// Tool catalogue (`tools/list`).
pub fn tool_list() -> Value {
    let mut tools = read_tools();
    tools.extend([
        json!({"name": "relabel", "description": "Set a cluster's label (good | mua | noise | unsorted); journaled and undoable. Queued as a proposal instead when the human chose propose-only",
         "inputSchema": {"type": "object", "properties": {"cluster": {"type": "integer"}, "label": {"type": "string"}}, "required": ["cluster", "label"]}}),
        json!({"name": "merge", "description": "Merge source clusters into target; journaled and undoable. Queued as a proposal instead when the human chose propose-only",
         "inputSchema": {"type": "object", "properties": {"sources": {"type": "array", "items": {"type": "integer"}}, "target": {"type": "integer"}}, "required": ["sources", "target"]}}),
        json!({"name": "split", "description": "Split a cluster. Preferred: candidate = a split handle (\"s12.1\") from brief level 3, suggest or split_proposals; it is refused as stale once that cluster changed. Alternatively cluster + spike_idx (positions in the cluster's time-sorted spikes) move into a new cluster. Queued as a proposal instead when the human chose propose-only",
         "inputSchema": {"type": "object", "properties": {"candidate": {"type": "string"}, "cluster": {"type": "integer"}, "spike_idx": {"type": "array", "items": {"type": "integer"}}}}}),
        json!({"name": "tag", "description": "Replace a cluster's tag set (provenance/review markers, e.g. drop review:* once decided); journaled and undoable",
         "inputSchema": {"type": "object", "properties": {"cluster": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["cluster", "tags"]}}),
        json!({"name": "rules_apply", "description": "Apply a rule set (TOML, as in rules_eval; default: the last evaluated or <save_dir>/.sorrel/review_rules.toml) as relabels in ONE undoable edit, and remember it in <save_dir>/.sorrel/review_rules.toml. Evaluate with rules_eval first. Queued as one proposal per relabel (reason rule:<name>) instead when the human chose propose-only",
         "inputSchema": {"type": "object", "properties": {"rules": {"type": "string"}}}}),
        json!({"name": "decide", "description": "Many decisions in one call. decisions = [[target, action, confidence, reasons], ...] (or objects {target, action, confidence, reasons, note}). target: cluster id | split handle \"s12.1\" | merge handle \"m3\" | id list (merge; first is the target). action: good|mua|noise|unsorted (relabel) | merge | split (needs an s handle) | accept (drops review:* tags). reasons: at most 3 short codes (flag codes, rule names, or at most 8 words). Everything is validated first: one invalid item applies nothing and the reply lists the item errors. Direct: one undo step. Propose-only: one proposal per item",
         "inputSchema": {"type": "object", "properties": {"decisions": {"type": "array", "items": {}}}, "required": ["decisions"]}}),
        json!({"name": "undo", "description": "Undo the last curation command (it stays in the history as a branch)", "inputSchema": {"type": "object", "properties": {}}}),
        json!({"name": "save", "description": "Write spike_clusters.npy, cluster_group.tsv and curation.json", "inputSchema": {"type": "object", "properties": {}}}),
        json!({"name": "propose", "description": "Stage a curation proposal for the human to accept or reject; nothing is applied. action: relabel (clusters=[c], label) | merge (clusters, first is the target) | split (clusters=[c], spike_idx, or candidate \"s12.1\") | accept (clusters=[c]). A split or merge may instead name a handle from brief level 3 / suggest as candidate (\"s12.1\", \"m3\") with no clusters. confidence in [0,1]; at most 3 reasons.",
         "inputSchema": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["relabel", "merge", "split", "accept"]},
            "clusters": {"type": "array", "items": {"type": "integer"}},
            "label": {"type": "string"},
            "spike_idx": {"type": "array", "items": {"type": "integer"}},
            "candidate": {"type": "string"},
            "confidence": {"type": "number"},
            "reasons": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
          "required": ["action", "confidence", "reasons"]}}),
        json!({"name": "select", "description": "Move the human's GUI selection to these clusters so they see what you are discussing (needs the GUI)",
         "inputSchema": {"type": "object", "properties": {"clusters": {"type": "array", "items": {"type": "integer"}}}, "required": ["clusters"]}}),
        json!({"name": "selection", "description": "The clusters the human has selected in the GUI right now (needs the GUI)",
         "inputSchema": {"type": "object", "properties": {}}}),
    ]);
    #[cfg(feature = "phy-export")]
    tools.push(json!(
        {"name": "export_phy", "description": "Write the curated session as a complete phy directory at out (params.py references the raw file unless copy_raw)",
         "inputSchema": {"type": "object", "properties": {"out": {"type": "string"}, "copy_raw": {"type": "boolean"}}, "required": ["out"]}}
    ));
    Value::Array(tools)
}

fn prompts_list() -> Value {
    json!({"prompts": [{"name": "review",
        "description": "Review a sort by policy: brief, rules, margin units, decide, confirm (alias: triage)",
        "arguments": []}]})
}

fn prompt_get(name: &str) -> Option<Value> {
    matches!(name, "review" | "triage").then(|| {
        json!({"description": "Review a spike sort", "messages": [{"role": "user",
            "content": {"type": "text", "text": format!("{REVIEW_INSTRUCTIONS}{RUBRIC}")}}]})
    })
}

/// Serve `host` over stdio until `input` closes. `reseal_on_save` follows
/// the same rule as the GUI: true only for phy directories.
pub fn serve<P, F>(
    host: &SessionHost<P>,
    save_dir: &Path,
    reseal_on_save: bool,
    label_str: F,
    input: impl BufRead,
    output: impl Write,
) -> Result<()>
where
    P: DataProvider + ApplyPhyLabel + HasGeometry,
    F: Fn(&P::Label) -> &'static str + Sync,
{
    let env = Env::new(save_dir, reseal_on_save, label_str).subscribe(host);
    serve_lines(host, env, input, output).map_err(Into::into)
}

/// Answer each line of `input` on `output`, followed by the notifications
/// that piled up meanwhile.
fn serve_lines<P, F>(
    host: &SessionHost<P>,
    mut env: Env<F>,
    input: impl BufRead,
    mut output: impl Write,
) -> std::io::Result<()>
where
    P: DataProvider + ApplyPhyLabel + HasGeometry,
    F: Fn(&P::Label) -> &'static str + Sync,
{
    for line in input.lines() {
        let reply = handle_line(host, &mut env, &line?);
        for v in reply.into_iter().chain(env.notifications()) {
            write_json(&mut output, &v).map_err(std::io::Error::other)?;
        }
    }
    Ok(())
}

/// One request line to its reply; `None` for blank lines and notifications.
pub fn handle_line<P, F>(host: &SessionHost<P>, env: &mut Env<F>, line: &str) -> Option<Value>
where
    P: DataProvider + ApplyPhyLabel + HasGeometry,
    F: Fn(&P::Label) -> &'static str + Sync,
{
    env.capture_client(line);
    if let (Some(name), None) = (&env.client, &env.seat) {
        env.seat = Some(host.agent_joined(name.clone()));
    }
    respond(line, |name, args| call_tool(host, env, name, args))
}

/// The JSON-RPC envelope.
fn respond(
    line: &str,
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
        "tools/list" => Ok(json!({"tools": tool_list()})),
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

#[derive(Debug)]
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

fn call_tool<P, F>(
    host: &SessionHost<P>,
    env: &mut Env<F>,
    name: &str,
    args: &Value,
) -> Result<Value, ToolError>
where
    P: DataProvider + ApplyPhyLabel + HasGeometry,
    F: Fn(&P::Label) -> &'static str + Sync,
{
    let agent = env.agent();
    let resolved;
    let args = match args.get("candidate").and_then(Value::as_str) {
        Some(id) if matches!(name, "split" | "propose") => {
            resolved = resolve_candidate(&host.read(), &env.handles, name, id, args)?;
            &resolved
        }
        _ => args,
    };
    let ok = |ev: HostEvent| text(json!({"ok": true, "version": ev.version}));
    if host.policy() == AgentPolicy::ProposeOnly {
        match name {
            "relabel" | "merge" | "split" => {
                let args = edit_as_proposal(name, args);
                let p = parse_proposal(&host.read(), &args)?;
                return queue(host, env, p);
            }
            "tag" | "undo" | "save" | "export_phy" => {
                return fail("the human set agents to propose-only: use propose")
            }
            _ => {}
        }
    }
    Ok(match name {
        "tag" => {
            let cluster = check(&host.read(), arg_u32(args, "cluster")?)?;
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
            ok(env.mine(host.apply(agent, CurationCommand::Tag { cluster, tags })?))
        }
        "relabel" => {
            let cluster = check(&host.read(), arg_u32(args, "cluster")?)?;
            let label = args.get("label").and_then(Value::as_str).unwrap_or("");
            let Some(op) = proposals::parse_label(label) else {
                return fail(format!("unknown label {label:?}"));
            };
            ok(env.mine(host.apply(agent, CurationCommand::Relabel { cluster, op })?))
        }
        "merge" => {
            let (sources, target) = {
                let s = host.read();
                let sources: Vec<ClusterId> = arg_u32_list(args, "sources")?
                    .into_iter()
                    .map(|c| check(&s, c))
                    .collect::<Result<_, _>>()?;
                (sources, check(&s, arg_u32(args, "target")?)?)
            };
            if sources.contains(&target) {
                return fail(format!("target {} is also listed in sources", target.0));
            }
            ok(env.mine(host.apply(agent, CurationCommand::Merge { sources, target })?))
        }
        "split" => {
            let (cluster, spike_idx, basis) = {
                let s = host.read();
                let cluster = check(&s, arg_u32(args, "cluster")?)?;
                let spike_idx = split_indices(&s, cluster, args)?;
                // A handle carries the basis it was issued against.
                let basis = args
                    .get("basis")
                    .and_then(Value::as_str)
                    .map_or_else(|| proposals::spike_basis(&s, cluster), str::to_owned);
                (cluster, spike_idx, basis)
            };
            let ev = env.mine(apply_split(host, agent, cluster, spike_idx, &basis)?);
            let new_cluster = ev.clusters.last().map(|c| c.0);
            text(json!({"ok": true, "new_cluster": new_cluster, "version": ev.version}))
        }
        "undo" => ok(env.mine(host.apply(agent, CurationCommand::Undo)?)),
        "rules_apply" => rules_apply(host, env, args)?,
        "decide" => decide(host, env, args)?,
        "save" => {
            save_session(
                &mut host.write(),
                env.save_dir,
                env.reseal_on_save,
                &env.label_str,
            )?;
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
            crate::phy_export::export_phy(&host.read(), Path::new(out), copy_raw, &env.label_str)?;
            text(json!({"ok": true, "out": out}))
        }
        "propose" => {
            let p = parse_proposal(&host.read(), args)?;
            return queue(host, env, p);
        }
        "select" => {
            let ui = env.ui.as_ref().ok_or(no_gui())?;
            let clusters: Vec<ClusterId> = {
                let s = host.read();
                arg_u32_list(args, "clusters")?
                    .into_iter()
                    .map(|c| check(&s, c))
                    .collect::<Result<_, _>>()?
            };
            if ui.select.send(clusters).is_err() {
                return fail("the GUI is no longer listening");
            }
            (ui.wake)();
            text(json!({"ok": true}))
        }
        "selection" => {
            let ui = env.ui.as_ref().ok_or(no_gui())?;
            let sel = ui.selection.lock().unwrap_or_else(|p| p.into_inner());
            text(json!({"clusters": sel.iter().map(|c| c.0).collect::<Vec<_>>()}))
        }
        _ => {
            let session = host.read();
            return call_read(
                &session,
                env,
                name,
                args,
                Some(session.provider.channel_positions()),
            );
        }
    })
}

fn no_gui() -> ToolError {
    ToolError::Failed("no GUI is attached to this session".into())
}

/// Audit `p` as pending and queue it on the host for a human, named after
/// the client that proposed it.
fn queue<P, F>(host: &SessionHost<P>, env: &Env<F>, mut p: Proposal) -> Result<Value, ToolError>
where
    P: DataProvider + ApplyPhyLabel,
{
    p.proposer = Some(actor_name(&env.agent()).to_owned());
    proposals::append(env.save_dir, &p)
        .map_err(|e| ToolError::Failed(format!("audit log: {e}")))?;
    let id = p.id;
    host.propose(p);
    Ok(text(
        json!({"ok": true, "id": id, "status": "queued for human review; NOT applied"}),
    ))
}

/// The `propose` arguments an edit tool's arguments stand for (propose-only
/// policy); confidence defaults to 1 and the reason names the tool.
fn edit_as_proposal(name: &str, args: &Value) -> Value {
    let mut out = json!({"action": name, "reasons": [format!("{name} under propose-only")],
                         "confidence": args.get("confidence").cloned().unwrap_or(json!(1.0))});
    match name {
        "merge" => {
            let mut clusters = vec![args.get("target").cloned().unwrap_or(Value::Null)];
            if let Some(src) = args.get("sources").and_then(Value::as_array) {
                clusters.extend(src.iter().cloned());
            }
            out["clusters"] = json!(clusters);
        }
        _ => {
            out["clusters"] = json!([args.get("cluster").cloned().unwrap_or(Value::Null)]);
            out["label"] = args.get("label").cloned().unwrap_or(Value::Null);
            out["spike_idx"] = args.get("spike_idx").cloned().unwrap_or(Value::Null);
            out["basis"] = args.get("basis").cloned().unwrap_or(Value::Null);
        }
    }
    out
}

/// The sorted, deduplicated `spike_idx` of a split of `cluster`: a proper
/// non-empty subset of its spikes (shared by `split` and `propose`).
fn split_indices<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
    args: &Value,
) -> Result<Vec<u32>, ToolError> {
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
    Ok(spike_idx)
}

/// Split `cluster` by `spike_idx`, positions in the membership `basis`
/// ([`proposals::spike_basis`]) described. The basis is checked under the
/// apply's write lock: an edit since it was read refuses the split instead
/// of moving different spikes.
fn apply_split<P: DataProvider + ApplyPhyLabel>(
    host: &SessionHost<P>,
    agent: Actor,
    cluster: ClusterId,
    spike_idx: Vec<u32>,
    basis: &str,
) -> Result<HostEvent> {
    let by = Method::Llm(actor_name(&agent).to_owned());
    let cmd = CurationCommand::Split {
        cluster,
        spike_idx,
        new_cluster: ClusterId(u32::MAX),
        by,
    };
    host.apply_if(agent, cmd, |s| {
        refuse_if_propose_only(host)?;
        if proposals::spike_basis(s, cluster) != basis {
            bail!(
                "cluster {} changed since its spikes were read; read them again and retry",
                cluster.0
            );
        }
        Ok(())
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
            let spike_idx = split_indices(session, cluster, args)?;
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
    // Stamp a split's basis against the same session its indices were
    // validated on, so it reaches the audit log and survives a restart.
    let basis = match &action {
        Action::Split { cluster, .. } => Some(
            args.get("basis")
                .and_then(Value::as_str)
                .map_or_else(|| proposals::spike_basis(session, *cluster), str::to_owned),
        ),
        _ => None,
    };
    let mut p = Proposal::new(
        Source::Agent,
        action,
        confidence as f32,
        reasons,
        args.clone(),
    );
    p.basis = basis;
    Ok(p)
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

/// Hand the active rule thresholds to `env.handles`, so rendered lines keep
/// every rule decision through quantising.
fn sync_thresholds<F>(env: &mut Env<F>) {
    env.handles.extra = active_rules(env.save_dir, &env.rules)
        .map(|r| r.thresholds())
        .unwrap_or_default();
}

/// Refuse inside an apply: the human may have switched to propose-only since
/// the policy was read.
fn refuse_if_propose_only<P: DataProvider + ApplyPhyLabel>(host: &SessionHost<P>) -> Result<()> {
    if host.policy() == AgentPolicy::ProposeOnly {
        bail!("the human switched to propose-only; nothing applied");
    }
    Ok(())
}

/// Bring the cached suggestions and the handle table up to date.
fn refresh_handles<P: DataProvider, F>(session: &Session<P>, env: &mut Env<F>) {
    sync_thresholds(env);
    let _ = suggestions(session, env);
    if let Some((_, m, s)) = &env.suggest {
        env.handles.refresh(session, m, s);
    }
}

/// `args` of a `split` / `propose` that names a handle, rewritten to the
/// `cluster(s)` + `spike_idx` (+ the handle's `basis`) the handle stands for.
/// Fails when any cluster the handle names changed since it was issued.
fn resolve_candidate<P: DataProvider>(
    session: &Session<P>,
    handles: &crate::evaluate::Handles,
    name: &str,
    id: &str,
    args: &Value,
) -> Result<Value, ToolError> {
    let mut out = args.clone();
    let is_action = |a: &str| name == "split" || args["action"] == a;
    if id.starts_with('s') && is_action("split") {
        let (cluster, minor, basis) = handles
            .resolve_split(session, id)
            .map_err(ToolError::Failed)?;
        out["cluster"] = json!(cluster.0);
        out["clusters"] = json!([cluster.0]);
        out["spike_idx"] = json!(minor);
        out["basis"] = json!(basis);
    } else if id.starts_with('m') && name == "propose" && is_action("merge") {
        let members = handles
            .resolve_merge(session, id)
            .map_err(ToolError::Failed)?;
        out["clusters"] = json!(members.iter().map(|c| c.0).collect::<Vec<_>>());
    } else {
        return fail("candidate: s<cluster>.<k> for split, m<k> for propose action merge");
    }
    Ok(out)
}

/// Mean amplitude and count of a part of a cluster (local spike indices).
fn part_stats(amps: &[f32], idx: impl Iterator<Item = usize>) -> Value {
    let v: Vec<f32> = idx.filter_map(|i| amps.get(i).copied()).collect();
    let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
    json!({"n_spikes": v.len(), "mean_amplitude": mean})
}

/// `<save_dir>/sorrel_auto/evidence.json`, if `sorrel auto` wrote one.
pub(crate) fn load_evidence(save_dir: &Path) -> Option<Value> {
    let text = std::fs::read_to_string(save_dir.join("sorrel_auto").join("evidence.json")).ok()?;
    serde_json::from_str(&text).ok()
}

/// A unit's evidence record and the merge records naming it, by provider
/// unit id.
pub(crate) fn unit_evidence(doc: &Value, uid: u64) -> (Option<Value>, Vec<&Value>) {
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
    let merges = doc["merges"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|m| has(&m["units"]))
        .collect();
    (unit, merges)
}

/// `<save_dir>/.sorrel/review_rules.toml`: the rule set `rules_apply` last used.
fn rules_path(save_dir: &Path) -> PathBuf {
    save_dir.join(".sorrel").join("review_rules.toml")
}

/// The rules in force: the connection's last, else the persisted file.
fn active_rules(save_dir: &Path, slot: &Option<(String, RuleSet)>) -> Option<RuleSet> {
    slot.as_ref().map(|r| r.1.clone()).or_else(|| {
        let text = std::fs::read_to_string(rules_path(save_dir)).ok()?;
        RuleSet::parse(&text).ok()
    })
}

/// The `rules` argument (TOML) parsed and remembered in `slot`; absent, the
/// last one or the persisted file.
fn resolve_rules(
    save_dir: &Path,
    slot: &mut Option<(String, RuleSet)>,
    args: &Value,
) -> Result<(String, RuleSet), ToolError> {
    let bad = |e: anyhow::Error| ToolError::Failed(format!("rules: {e:#}"));
    let given = args.get("rules").and_then(Value::as_str);
    let text = match (given, &*slot) {
        (Some(t), _) => t.to_owned(),
        (None, Some((t, _))) => t.clone(),
        (None, None) => std::fs::read_to_string(rules_path(save_dir)).map_err(|_| {
            ToolError::Failed("no rules: pass `rules` (TOML) or run rules_apply first".into())
        })?,
    };
    let set = RuleSet::parse(&text).map_err(bad)?;
    *slot = Some((text.clone(), set.clone()));
    Ok((text, set))
}

/// QC rows for the session's current version, recomputed only after an edit.
fn cached_qc<'a, P: DataProvider>(
    session: &Session<P>,
    qc: &'a mut Option<(u64, Vec<QcRow>)>,
    positions: Option<&[[f32; 2]]>,
) -> &'a [QcRow] {
    let version = session.version();
    if qc.as_ref().is_none_or(|(v, _)| *v != version) {
        *qc = Some((version, collect_qc_rows_with(session, positions)));
    }
    qc.as_ref().map_or(&[], |(_, rows)| rows)
}

/// Apply `children` as ONE journaled command (one undo step, one event).
/// `version` is the session version the commands were derived from: the
/// check runs under the apply's write lock, so an edit that landed since
/// refuses the whole batch (this also covers every split's membership
/// basis, and the tag lists an `accept` replaces).
fn apply_all<P, F>(
    host: &SessionHost<P>,
    env: &mut Env<F>,
    version: u64,
    mut children: Vec<CurationCommand>,
) -> Result<HostEvent, ToolError>
where
    P: DataProvider + ApplyPhyLabel,
{
    let cmd = match children.len() {
        1 => children.remove(0),
        _ => CurationCommand::batch(children)
            .ok_or_else(|| ToolError::Failed("nested batch".into()))?,
    };
    let ev = host.apply_if(env.agent(), cmd, |s| {
        refuse_if_propose_only(host)?;
        if s.version() != version {
            bail!("the session changed since this was validated; nothing applied, read it again and retry");
        }
        Ok(())
    })?;
    Ok(env.mine(ev))
}

/// `rules_apply`: relabel every unit the rules move, as one batch under
/// Direct (one undo step) and one proposal per relabel under ProposeOnly.
fn rules_apply<P, F>(
    host: &SessionHost<P>,
    env: &mut Env<F>,
    args: &Value,
) -> Result<Value, ToolError>
where
    P: DataProvider + ApplyPhyLabel + HasGeometry,
    F: Fn(&P::Label) -> &'static str + Sync,
{
    let (toml, set) = resolve_rules(env.save_dir, &mut env.rules, args)?;
    let label_str = &env.label_str;
    let current = |s: &Session<P>, c: ClusterId| s.label(c).map_or("unsorted", |l| label_str(&l));
    // (command, deciding rule, present label) per changed unit, read under one lock.
    let (version, plan): (u64, Vec<(CurationCommand, String, &str)>) = {
        let s = host.read();
        let rows = cached_qc(&s, &mut env.qc, Some(s.provider.channel_positions()));
        let cmds = apply_rules(&set, rows, |c| current(&s, c))
            .map_err(|e| ToolError::Failed(format!("{e:#}")))?;
        let plan = cmds
            .into_iter()
            .map(|cmd| {
                let CurationCommand::Relabel { cluster, .. } = cmd else {
                    unreachable!("apply_rules only relabels")
                };
                let by = rows
                    .iter()
                    .find(|r| r.cluster_id == cluster.0)
                    .and_then(|r| set.decide(r))
                    .map_or("default".to_owned(), |i| set.name(i));
                (cmd, by, current(&s, cluster))
            })
            .collect();
        (s.version(), plan)
    };
    let mut moves: std::collections::BTreeMap<(&str, &str), Vec<u32>> = Default::default();
    for (cmd, _, from) in &plan {
        if let CurationCommand::Relabel { cluster, op } = cmd {
            moves
                .entry((*from, op_name(*op)))
                .or_default()
                .push(cluster.0);
        }
    }
    let queued = host.policy() == AgentPolicy::ProposeOnly;
    if queued {
        for (done, (cmd, by, _)) in plan.iter().enumerate() {
            let CurationCommand::Relabel { cluster, op } = cmd else {
                continue;
            };
            let args = json!({"action": "relabel", "clusters": [cluster.0], "label": op_name(*op),
                              "confidence": 1.0, "reasons": [format!("rule:{by}")]});
            if let Err(e) = parse_proposal(&host.read(), &args).and_then(|p| queue(host, env, p)) {
                return fail(format!(
                    "stopped after {done} of {} relabels: {}",
                    plan.len(),
                    err_msg(e)
                ));
            }
        }
    } else if !plan.is_empty() {
        let children = plan.iter().map(|p| p.0.clone()).collect();
        apply_all(host, env, version, children)?;
    }
    let path = rules_path(env.save_dir);
    std::fs::create_dir_all(path.parent().unwrap_or(env.save_dir))
        .and_then(|()| std::fs::write(&path, toml))
        .map_err(|e| ToolError::Failed(format!("write {}: {e}", path.display())))?;
    let transitions: Vec<String> = moves
        .iter()
        .map(|((from, to), ids)| format!("{from}->{to} {}: {}", ids.len(), rle(ids)))
        .collect();
    Ok(text(json!({
        "ok": true, "relabels": plan.len(), "transitions": transitions,
        "undo_steps": usize::from(!queued && !plan.is_empty()),
        "status": if queued { "queued for human review; NOT applied" } else { "applied" },
        "rules": path.display().to_string(),
    })))
}

fn err_msg(e: ToolError) -> String {
    match e {
        ToolError::Failed(m) => m,
        ToolError::Unknown => "unknown tool".into(),
    }
}

/// One `decide` item as a validated proposal. `item` is
/// `[target, action, confidence, reasons, note?]` or the same as an object.
fn decision<P: DataProvider>(
    session: &Session<P>,
    handles: &crate::evaluate::Handles,
    item: &Value,
) -> Result<(Proposal, Option<String>), String> {
    let get = |i: usize, key: &str| match item {
        Value::Array(a) => a.get(i).cloned(),
        o => o.get(key).cloned(),
    };
    let target = get(0, "target").ok_or("missing target")?;
    let action = get(1, "action")
        .and_then(|a| a.as_str().map(str::to_owned))
        .ok_or("missing action")?;
    let note = get(4, "note").and_then(|n| n.as_str().map(str::to_owned));
    let reasons = get(3, "reasons").unwrap_or_else(|| json!([]));
    if let Some(long) = reasons
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .find(|r| r.split_whitespace().count() > 8)
    {
        return Err(format!("reason {long:?} is longer than 8 words"));
    }
    let mut args =
        json!({"confidence": get(2, "confidence").unwrap_or(Value::Null), "reasons": reasons});
    let want = |h: &str, a: &str| match action == a {
        true => Ok(()),
        false => Err(format!("handle {h} takes action {a}, not {action}")),
    };
    match &target {
        Value::String(h) if h.starts_with('s') => {
            want(h, "split")?;
            let (c, minor, basis) = handles.resolve_split(session, h)?;
            args["clusters"] = json!([c.0]);
            args["spike_idx"] = json!(minor);
            args["basis"] = json!(basis);
        }
        Value::String(h) if h.starts_with('m') => {
            want(h, "merge")?;
            let members = handles.resolve_merge(session, h)?;
            args["clusters"] = json!(members.iter().map(|c| c.0).collect::<Vec<_>>());
        }
        Value::Array(ids) => args["clusters"] = json!(ids),
        Value::Number(_) => args["clusters"] = json!([target]),
        Value::String(n) => {
            let n: u32 = n.parse().map_err(|_| format!("bad target {n:?}"))?;
            args["clusters"] = json!([n]);
        }
        _ => return Err("target: cluster id, id list, s<c>.<k> or m<k>".into()),
    }
    match action.as_str() {
        "merge" | "accept" => args["action"] = json!(action),
        "split" if args.get("spike_idx").is_none() => {
            return Err("split needs an s<c>.<k> handle".into())
        }
        "split" => args["action"] = json!("split"),
        label => {
            args["action"] = json!("relabel");
            args["label"] = json!(label);
        }
    }
    let p = parse_proposal(session, &args).map_err(err_msg)?;
    Ok((p, note))
}

/// `decide`: validate every item first, then apply all as one batch (Direct)
/// or queue one proposal each (ProposeOnly).
fn decide<P, F>(host: &SessionHost<P>, env: &mut Env<F>, args: &Value) -> Result<Value, ToolError>
where
    P: DataProvider + ApplyPhyLabel,
{
    let items = args
        .get("decisions")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or_else(|| ToolError::Failed("decisions must be a non-empty list".into()))?;
    let agent = env.agent();
    let by = Method::Llm(actor_name(&agent).to_owned());
    let (version, n_before, props, children, errors) = {
        let s = host.read();
        let (mut props, mut children, mut errors) = (Vec::new(), Vec::new(), Vec::new());
        let mut used: std::collections::HashMap<u32, usize> = Default::default();
        for (i, item) in items.iter().enumerate() {
            let one = decision(&s, &env.handles, item).and_then(|(p, note)| {
                for c in p.clusters() {
                    if let Some(j) = used.insert(c.0, i) {
                        return Err(format!("cluster {} is also in item {j}", c.0));
                    }
                }
                Ok((p, note))
            });
            match one {
                Err(e) => errors.push(format!("{i}: {e}")),
                Ok((p, note)) => {
                    match &p.action {
                        Action::Relabel { cluster, label } => {
                            let op = proposals::parse_label(label).expect("validated");
                            children.push(CurationCommand::Relabel {
                                cluster: *cluster,
                                op,
                            });
                        }
                        Action::Merge { clusters } => {
                            let (&target, rest) = clusters.split_first().expect("validated");
                            children.push(CurationCommand::Merge {
                                sources: rest.to_vec(),
                                target,
                            });
                        }
                        Action::Split { cluster, spike_idx } => {
                            children.push(CurationCommand::Split {
                                cluster: *cluster,
                                spike_idx: spike_idx.clone(),
                                new_cluster: ClusterId(u32::MAX),
                                by: by.clone(),
                            });
                        }
                        Action::Accept { cluster } => {
                            let old = s.tags(*cluster);
                            let mut tags: Vec<String> = old
                                .iter()
                                .filter(|t| !t.starts_with("review:"))
                                .cloned()
                                .collect();
                            if !tags.iter().any(|t| t == "src:agent") {
                                tags.push("src:agent".into());
                            }
                            if tags != old {
                                children.push(CurationCommand::Tag {
                                    cluster: *cluster,
                                    tags,
                                });
                            }
                        }
                    }
                    if let (Some(text), Some(&cluster)) = (note, p.clusters().first()) {
                        children.push(CurationCommand::Note { cluster, text });
                    }
                    props.push(p);
                }
            }
        }
        (s.version(), s.n_clusters(), props, children, errors)
    };
    if !errors.is_empty() {
        return fail(format!(
            "nothing applied; {} of {} invalid\n{}",
            errors.len(),
            items.len(),
            errors.join("\n")
        ));
    }
    if host.policy() == AgentPolicy::ProposeOnly {
        let n = props.len();
        for (done, p) in props.into_iter().enumerate() {
            queue(host, env, p).map_err(|e| {
                ToolError::Failed(format!(
                    "stopped after {done} of {n} proposals: {}",
                    err_msg(e)
                ))
            })?;
        }
        return Ok(plain(format!(
            "ok {n} queued for human review; NOT applied; errors: -"
        )));
    }
    if children.is_empty() {
        return Ok(plain(format!(
            "ok {} applied (0 undo steps); errors: -",
            props.len()
        )));
    }
    let ev = apply_all(host, env, version, children)?;
    let new: Vec<String> = ev
        .clusters
        .iter()
        .filter(|c| c.0 >= n_before)
        .map(|c| c.0.to_string())
        .collect();
    let new = if new.is_empty() {
        String::new()
    } else {
        format!("; new clusters {}", new.join(","))
    };
    Ok(plain(format!(
        "ok {} applied (1 undo step){new}; errors: -",
        props.len()
    )))
}

fn plain(s: String) -> Value {
    json!([{"type": "text", "text": s}])
}

fn call_read<P, F>(
    session: &Session<P>,
    env: &mut Env<F>,
    name: &str,
    args: &Value,
    positions: Option<&[[f32; 2]]>,
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
            let dash = |s: String| {
                if s.is_empty() {
                    "-".into()
                } else {
                    s.replace(['\t', '\n'], " ")
                }
            };
            let ids = |v: Vec<u32>| v.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
            let mut out = String::from("id\tn\tlabel\ttags\torigin\tparents\tnote");
            for c in (0..session.n_clusters()).map(ClusterId) {
                let n = session.spike_times(c).len();
                let Some(label) = session.label(c).map(|l| label_str(&l)) else {
                    continue;
                };
                if n == 0
                    || want.is_some_and(|w| w != label)
                    || tag.is_some_and(|t| !session.tags(c).iter().any(|x| x.starts_with(t)))
                {
                    continue;
                }
                let parents = ids(session.parents(c).iter().map(|p| p.0).collect());
                out += &format!(
                    "\n{}\t{n}\t{label}\t{}\t{}\t{}\t{}",
                    c.0,
                    dash(session.tags(c).join(",")),
                    dash(session.origin_label(c)),
                    dash(parents),
                    dash(session.note(c).to_string()),
                );
            }
            json!([{"type": "text", "text": out}])
        }
        "evidence" => {
            let c = check(session, arg_u32(args, "cluster")?)?;
            let uid = u64::from(session.provider.unit_id(c));
            let doc = load_evidence(env.save_dir).ok_or_else(|| {
                ToolError::Failed(format!(
                    "no {} (run `sorrel auto` first)",
                    env.save_dir
                        .join("sorrel_auto")
                        .join("evidence.json")
                        .display()
                ))
            })?;
            let (unit, merges) = unit_evidence(&doc, uid);
            text(json!({"unit": unit, "merges": merges, "tags": session.tags(c)}))
        }
        "brief" => {
            let opts = BriefOpts {
                level: args
                    .get("level")
                    .and_then(Value::as_u64)
                    .unwrap_or(1)
                    .min(3) as u8,
                clusters: args
                    .get("clusters")
                    .map(|_| arg_u32_list(args, "clusters"))
                    .transpose()?,
                budget_tokens: args
                    .get("budget")
                    .and_then(Value::as_u64)
                    .map(|b| b as usize),
                extra: active_rules(env.save_dir, &env.rules)
                    .map(|r| r.thresholds())
                    .unwrap_or_default(),
            };
            refresh_handles(session, env);
            let version = session.version();
            if env.qc.as_ref().is_none_or(|(v, _)| *v != version) {
                env.qc = Some((version, collect_qc_rows_with(session, positions)));
            }
            let (Some((_, rows)), Some((_, merges, splits))) = (&env.qc, &env.suggest) else {
                return fail("brief caches missing");
            };
            let relations = match opts.level >= 3 {
                true => env
                    .handles
                    .relations(session, merges, opts.clusters.as_deref()),
                false => Vec::new(),
            };
            let brief = render_brief(
                session,
                rows,
                merges,
                splits,
                &relations,
                &env.label_str,
                env.save_dir,
                &opts,
            );
            json!([{"type": "text", "text": brief}])
        }
        "rules_eval" => {
            let (_, set) = resolve_rules(env.save_dir, &mut env.rules, args)?;
            let margin = args
                .get("margin")
                .and_then(Value::as_f64)
                .map_or(0.1, |m| m as f32);
            let rows = cached_qc(session, &mut env.qc, positions);
            let doc = load_evidence(env.save_dir);
            let uid = |id| u64::from(session.provider.unit_id(ClusterId(id)));
            let evidence = |id| doc.as_ref().and_then(|d| unit_evidence(d, uid(id)).0);
            let report = eval_report(
                &set,
                rows,
                margin,
                |id| {
                    session
                        .label(ClusterId(id))
                        .map_or("unsorted", |l| label_str(&l))
                },
                doc.is_some().then_some(&evidence),
            );
            json!([{"type": "text", "text": report}])
        }
        "metrics" => {
            let c = check(session, arg_u32(args, "cluster")?)?;
            let version = session.version();
            if env.qc.as_ref().is_none_or(|(v, _)| *v != version) {
                env.qc = Some((version, collect_qc_rows_with(session, positions)));
            }
            let row = env
                .qc
                .as_ref()
                .and_then(|(_, rows)| rows.iter().find(|r| r.cluster_id == c.0))
                .ok_or_else(|| ToolError::Failed(format!("no metrics for cluster {}", c.0)))?;
            text(row.to_json())
        }
        "suggest" => {
            refresh_handles(session, env);
            let Some((_, merges, _)) = &env.suggest else {
                return fail("suggest cache missing");
            };
            let (comps, splits): (Vec<_>, Vec<_>) = (env.handles.relations(session, merges, None))
                .into_iter()
                .partition(|r| r.handle.starts_with('m'));
            text(json!({
                "components": comps.into_iter().take(20).collect::<Vec<_>>(),
                "splits": splits.into_iter().take(20).collect::<Vec<_>>(),
            }))
        }
        "eval" => {
            sync_thresholds(env);
            let _ = suggestions(session, env);
            let Some((_, merges, _)) = &env.suggest else {
                return fail("suggest cache missing");
            };
            let line = match (
                args.get("candidate").and_then(Value::as_str),
                args.get("merge"),
            ) {
                (Some(id), _) => env
                    .handles
                    .line(session, id, merges)
                    .map_err(ToolError::Failed)?,
                (None, Some(_)) => {
                    let ids = arg_u32_list(args, "merge")?;
                    crate::probe::merge_line(session, merges, &ids, &env.handles.extra)
                        .map_err(ToolError::Failed)?
                }
                _ => return fail("eval takes {merge:[ids]} or {candidate:\"s12.1\"|\"m3\"}"),
            };
            json!([{"type": "text", "text": line}])
        }
        "probe" => {
            use crate::probe::{parse, run, Query, GRAMMAR, MAX_QUERIES};
            let qs: Vec<&str> = match args.get("q") {
                Some(Value::String(s)) => vec![s.as_str()],
                Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect(),
                _ => {
                    return fail(format!(
                        "probe takes {{q: string | [string, ..]}}\n{GRAMMAR}"
                    ))
                }
            };
            if qs.is_empty() || qs.len() > MAX_QUERIES {
                return fail(format!("probe takes 1..={MAX_QUERIES} queries"));
            }
            let parsed: Vec<_> = qs.iter().map(|q| parse(q)).collect();
            sync_thresholds(env);
            if parsed
                .iter()
                .any(|p| matches!(p, Ok(Query::Merge(_) | Query::Handle(_))))
            {
                refresh_handles(session, env);
            }
            let merges: &[MergeCandidate] = env.suggest.as_ref().map_or(&[], |s| &s.1);
            let mut out = String::new();
            let mut bad = false;
            for (q, p) in qs.iter().zip(parsed) {
                bad |= p.is_err();
                let a = p.and_then(|p| run(session, &mut env.handles, merges, positions, &p));
                let a = a.unwrap_or_else(|e| format!("error: {e}"));
                out += &format!("> {q}\n{a}\n");
            }
            if bad {
                out += GRAMMAR;
            }
            plain(out)
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
            refresh_handles(session, env);
            let Some((_, _, splits)) = &env.suggest else {
                return fail("suggest cache missing");
            };
            let out: Vec<Value> = splits
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
                    if let Some((handle, line)) = env.handles.split_for(session, c) {
                        v["handle"] = json!(handle);
                        v["eval"] = json!(line);
                    }
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

#[cfg(test)]
thread_local! {
    /// Per-test-thread `user_dir()` override: tests never touch the real one
    /// or race on the environment.
    static TEST_USER_DIR: std::cell::RefCell<Option<PathBuf>> = const { std::cell::RefCell::new(None) };
}

/// The per-user state directory (mode 0700 on Unix), created on demand.
/// `SORREL_USER_DIR` overrides everything (tests, sandboxes); otherwise
/// `$XDG_RUNTIME_DIR/sorrel`, else `$HOME/.cache/sorrel`; on Windows
/// `%LOCALAPPDATA%\sorrel`. Holds the host registry (`mcp/`) and the GUI
/// instance file (`gui.json`).
pub fn user_dir() -> std::io::Result<PathBuf> {
    #[cfg(test)]
    if let Some(p) = TEST_USER_DIR.with(|d| d.borrow().clone()) {
        return Ok(p);
    }
    let var = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let missing = |k: &str| std::io::Error::other(format!("{k} is not set"));
    let dir = if let Some(d) = var("SORREL_USER_DIR") {
        d
    } else if cfg!(windows) {
        var("LOCALAPPDATA")
            .ok_or_else(|| missing("LOCALAPPDATA"))?
            .join("sorrel")
    } else if let Some(x) = var("XDG_RUNTIME_DIR") {
        x.join("sorrel")
    } else {
        var("HOME")
            .ok_or_else(|| missing("HOME"))?
            .join(".cache")
            .join("sorrel")
    };
    let mut b = std::fs::DirBuilder::new();
    b.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut b, 0o700);
    b.create(&dir)?;
    Ok(dir)
}

/// A fresh 128-bit hex token. std-only: `RandomState` keys come from the OS
/// RNG (once per process), mixed with the clock and a per-call counter via
/// SipHash. ponytail: not a CSPRNG; swap in `getrandom` if the token ever
/// guards more than a loopback socket in a user-only directory.
pub(crate) fn random_token() -> String {
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

fn canonical(save_dir: &Path) -> PathBuf {
    save_dir
        .canonicalize()
        .unwrap_or_else(|_| save_dir.to_owned())
}

/// Where the listener file for `save_dir` lives; `attach` and the listener
/// both go through here: `<user_dir>/mcp/<hash>.json`, named by a hash of the
/// canonical dataset path (lowercased on Windows, where paths are
/// case-insensitive). Off the dataset directory so another account that can
/// read the dataset cannot read the token, and so [`list_hosts`] finds every
/// running host in one place.
fn mcp_json_path(save_dir: &Path) -> std::io::Result<PathBuf> {
    let canon = canonical(save_dir).to_string_lossy().into_owned();
    let canon = if cfg!(windows) {
        canon.to_lowercase()
    } else {
        canon
    };
    let key = blake3::hash(canon.as_bytes());
    Ok(user_dir()?
        .join("mcp")
        .join(format!("{}.json", &key.to_hex()[..32])))
}

/// `{port, token, pid, nonce, dataset, kind}` (see [`mcp_json_path`]).
/// `kind` is `"gui"` or `"headless"`.
fn write_mcp_json(
    save_dir: &Path,
    port: u16,
    token: &str,
    nonce: &str,
    kind: &str,
) -> std::io::Result<PathBuf> {
    let path = mcp_json_path(save_dir)?;
    let dataset = canonical(save_dir).to_string_lossy().into_owned();
    write_private_json(
        &path,
        &json!({"port": port, "token": token, "pid": std::process::id(), "nonce": nonce,
                "dataset": dataset, "kind": kind}),
    )?;
    Ok(path)
}

/// Write `body` to `path` (parent created) atomically (tmp + rename, mode
/// 0600) so a client never reads half of it. Listener files carry `pid` and
/// `nonce` to identify the writer: only it removes the file.
pub(crate) fn write_private_json(path: &Path, body: &Value) -> std::io::Result<()> {
    std::fs::create_dir_all(path.parent().expect("has parent"))?;
    let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
    // `create_new` (O_EXCL) never follows a symlink pre-placed at `tmp`.
    match std::fs::remove_file(&tmp) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e),
        _ => {}
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600);
    let mut f = opts.open(&tmp)?;
    f.write_all(body.to_string().as_bytes())?;
    f.sync_all()?;
    std::fs::rename(&tmp, path)
}

/// Shared tail of a listener handle's `Drop`: stop the accept loop, remove
/// `path` only if this process wrote it (pid + `nonce`), and wake the
/// blocking accept so its thread sees `stop`.
pub(crate) fn shutdown_listener(
    stop: &std::sync::atomic::AtomicBool,
    path: &Path,
    nonce: &str,
    port: u16,
) {
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    // A later host in the same place (or a stale file we no longer own) is
    // left alone.
    let ours = std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .is_some_and(|v| v["pid"] == std::process::id() && v["nonce"].as_str() == Some(nonce));
    if ours {
        let _ = std::fs::remove_file(path);
    }
    let _ = std::net::TcpStream::connect(("127.0.0.1", port));
}

/// Keeps the listener alive; dropping it removes the listener file and stops
/// accepting (connections already open finish on their own).
pub struct ListenerHandle {
    port: u16,
    token: String,
    mcp_json: PathBuf,
    nonce: String,
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
        shutdown_listener(&self.stop, &self.mcp_json, &self.nonce, self.port);
    }
}

/// Expose `host` to MCP clients on `127.0.0.1:<ephemeral>`: the same
/// handler as stdio, with `ui` (when a GUI runs) behind `select` and
/// `selection`. Each connection subscribes to the host's events.
///
/// ponytail: one thread per connection with no cap; every connection must
/// present the token first, so only local processes that can read the 0600
/// listener file get a thread's worth of work.
pub fn spawn_listener<P>(
    host: Arc<SessionHost<P>>,
    save_dir: &Path,
    reseal_on_save: bool,
    label_str: fn(&P::Label) -> &'static str,
    ui: Option<Arc<Ui>>,
) -> std::io::Result<ListenerHandle>
where
    P: DataProvider + ApplyPhyLabel + HasGeometry + 'static,
    Session<P>: Send + Sync,
{
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    let token = random_token();
    let nonce = random_token();
    let kind = if ui.is_some() { "gui" } else { "headless" };
    let mcp_json = write_mcp_json(save_dir, port, &token, &nonce, kind)?;
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let handle = ListenerHandle {
        port,
        token: token.clone(),
        mcp_json,
        nonce,
        stop: stop.clone(),
    };
    let save_dir = save_dir.to_owned();
    // Connections still waiting to send their token. Each one holds a thread
    // for up to the handshake timeout, so cap them: past the cap a new
    // connection is dropped at once instead of costing a thread.
    let handshaking = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    std::thread::Builder::new()
        .name("sorrel-mcp".into())
        .spawn(move || {
            use std::sync::atomic::Ordering::SeqCst;
            for stream in listener.incoming() {
                if stop.load(SeqCst) {
                    break;
                }
                let Ok(stream) = stream else { continue };
                if handshaking.fetch_add(1, SeqCst) >= MAX_HANDSHAKES {
                    handshaking.fetch_sub(1, SeqCst);
                    continue;
                }
                let (host, ui, token, dir, handshaking) = (
                    host.clone(),
                    ui.clone(),
                    token.clone(),
                    save_dir.clone(),
                    Arc::clone(&handshaking),
                );
                std::thread::spawn(move || {
                    let authed = handshake(stream, &token);
                    handshaking.fetch_sub(1, SeqCst);
                    let (reader, writer) = match authed {
                        Ok(Some(rw)) => rw,
                        Ok(None) => return,
                        Err(e) => return log::debug!("mcp handshake failed: {e}"),
                    };
                    // Only an authenticated client subscribes to host events.
                    let env = Env::new(&dir, reseal_on_save, label_str)
                        .with_ui(ui)
                        .subscribe(&host);
                    if let Err(e) = serve_lines(&host, env, reader, writer) {
                        log::debug!("mcp client ended: {e}");
                    }
                });
            }
        })?;
    Ok(handle)
}

/// Unauthenticated connections the listener serves at once.
const MAX_HANDSHAKES: usize = 8;

/// Read the client's `{"token": ...}` line (5 s timeout) and acknowledge it.
/// `None` when the token is wrong or missing: the caller drops the client.
pub(crate) fn handshake(
    stream: std::net::TcpStream,
    token: &str,
) -> std::io::Result<Option<(std::io::BufReader<std::net::TcpStream>, std::net::TcpStream)>> {
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
        return Ok(None);
    }
    writer.set_read_timeout(None)?; // shared socket: clears the reader's timeout too
    writer.write_all(b"{\"ok\":true}\n")?;
    Ok(Some((reader, writer)))
}

/// `{port, token}` from a listener file, if it exists and parses.
pub(crate) fn read_port_token(path: &Path) -> Option<(u16, String)> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    Some((
        u16::try_from(v["port"].as_u64()?).ok()?,
        v["token"].as_str()?.to_owned(),
    ))
}

/// `{port, token}` of a running host's listener in `save_dir`, if any.
pub fn read_mcp_json(save_dir: &Path) -> Option<(u16, String)> {
    read_port_token(&mcp_json_path(save_dir).ok()?)
}

/// A running host found in the registry.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HostRecord {
    pub dataset: PathBuf,
    pub port: u16,
    pub pid: u32,
    /// `"gui"` or `"headless"`.
    pub kind: String,
    /// A loopback connect to `port` succeeded (300 ms): the file may still be
    /// stale if the port was reused.
    pub alive: bool,
}

/// Every host in the registry, sorted by dataset. Unreadable or invalid
/// files are skipped, and nothing is deleted: the files belong to the hosts
/// that wrote them.
pub fn list_hosts() -> Vec<HostRecord> {
    user_dir()
        .map(|d| list_hosts_in(&d.join("mcp")))
        .unwrap_or_default()
}

fn list_hosts_in(dir: &Path) -> Vec<HostRecord> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<HostRecord> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .filter_map(|p| {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()?;
            let port = u16::try_from(v["port"].as_u64()?).ok()?;
            let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
            Some(HostRecord {
                dataset: v["dataset"].as_str()?.into(),
                port,
                pid: u32::try_from(v["pid"].as_u64()?).ok()?,
                kind: v["kind"].as_str()?.to_owned(),
                alive: std::net::TcpStream::connect_timeout(
                    &addr,
                    std::time::Duration::from_millis(300),
                )
                .is_ok(),
            })
        })
        .collect();
    out.sort_by(|a, b| a.dataset.cmp(&b.dataset));
    out
}

/// Connect to a loopback host and complete the token handshake. `Ok(None)`
/// only when the connection is refused (a stale file); a timeout, a wrong
/// token or a bad ack means the host is alive, so those are errors. The
/// returned stream has no read timeout.
pub(crate) fn connect_authed(
    port: u16,
    token: &str,
) -> Result<Option<(std::net::TcpStream, std::io::BufReader<std::net::TcpStream>)>> {
    use std::time::Duration;
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = match std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(2)) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => return Ok(None),
        Err(e) => bail!("sorrel host on port {port} did not accept the connection: {e}"),
    };
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    write_json(&mut stream, &json!({"token": token}))?;
    let mut reader = std::io::BufReader::new(stream.try_clone()?);
    let mut ack = String::new();
    match reader.read_line(&mut ack) {
        Ok(n) if n > 0 && ack.trim() == "{\"ok\":true}" => {}
        Ok(_) => bail!("sorrel host on port {port} rejected the connection (bad token or ack)"),
        Err(e) => bail!("sorrel host on port {port} did not acknowledge: {e}"),
    }
    stream.set_read_timeout(None)?;
    Ok(Some((stream, reader)))
}

/// `sorrel mcp`: pipe `input` to the host listening in `save_dir` and
/// its replies to `output`, line by line, until either side closes. Returns
/// `Ok(false)` without touching either stream only when there is provably no
/// listener (no listener file, or the connection is refused: a stale file):
/// the caller then falls back to a headless `serve`. A host that accepts the
/// connection but times out, rejects the token or sends a bad ack is alive,
/// so that is an error, never a second writer on its journal.
pub fn attach(
    save_dir: &Path,
    input: impl Read + Send + 'static,
    mut output: impl Write,
) -> Result<bool> {
    let Some((port, token)) = read_mcp_json(save_dir) else {
        return Ok(false);
    };
    let Some((stream, mut reader)) = connect_authed(port, &token)? else {
        return Ok(false);
    };
    let mut up = stream;
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

/// One JSON-RPC `request` to the host in `save_dir`, and its first reply.
/// `Ok(None)` when no host listens there (see [`attach`]).
pub fn call_once(save_dir: &Path, request: &Value) -> Result<Option<Value>> {
    let mut out = Vec::new();
    let input = std::io::Cursor::new(format!("{request}\n"));
    if !attach(save_dir, input, &mut out)? {
        return Ok(None);
    }
    let line = out.split(|&b| b == b'\n').next().unwrap_or_default();
    if line.is_empty() {
        bail!(
            "sorrel host in {} closed without a reply",
            save_dir.display()
        );
    }
    Ok(Some(serde_json::from_slice(line)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::Journal;
    use crate::qc_export::tests::MockProvider;
    use sorrel_io::SampleIndex;
    use std::io::Cursor;

    /// Point this test thread's `user_dir()` at a fresh directory.
    fn scoped_user_dir() -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        TEST_USER_DIR.with(|u| *u.borrow_mut() = Some(d.path().to_owned()));
        d
    }

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
        let h = SessionHost::new(s);
        let mut out = Vec::new();
        serve(&h, dir.path(), false, label, Cursor::new(input), &mut out).unwrap();

        let replies: Vec<Value> = String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(replies.len(), 10, "notification gets no reply");
        assert_eq!(replies[0]["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(
            replies[1]["result"]["tools"].as_array().unwrap().len(),
            22 + usize::from(cfg!(feature = "phy-export"))
        );
        assert_eq!(replies[2]["id"], 3);
        assert_eq!(replies[2]["result"]["isError"], false);
        assert_eq!(h.read().label(ClusterId(0)), Some(1));
        assert!(h.read().spike_times(ClusterId(1)).is_empty(), "merged away");

        let listed = replies[4]["result"]["content"][0]["text"].as_str().unwrap();
        let rows: Vec<&str> = listed.lines().collect();
        assert_eq!(
            rows.len(),
            3,
            "header + 2 clusters, empty cluster 1 hidden: {listed}"
        );
        assert_eq!(rows[0], "id\tn\tlabel\ttags\torigin\tparents\tnote");
        assert!(
            rows[1].starts_with("0\t") && rows[1].contains("\tgood\t"),
            "{listed}"
        );

        let metrics: Value =
            serde_json::from_str(replies[5]["result"]["content"][0]["text"].as_str().unwrap())
                .unwrap();
        assert_eq!(metrics["n_spikes"], 3.0);
        assert!(metrics["l_ratio"].is_null(), "NaN becomes null");

        assert_eq!(replies[6]["error"]["code"], -32601, "render was removed");
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
        let s = Session::new(prov, Journal::open(&dir.path().join("j.journal")).unwrap());
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
        let h = SessionHost::new(s);
        let mut out = Vec::new();
        serve(
            &h,
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
            h.read().tags(ClusterId(1)),
            ["review:merge", "src:bombcell", "src:agent"]
        );
        assert_eq!(
            replies[1]["result"]["content"][0]["text"],
            "id\tn\tlabel\ttags\torigin\tparents\tnote\n1\t1\tunsorted\treview:merge,src:bombcell,src:agent\tAuto[unknown]\t-\t-"
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

    fn ui() -> (Arc<Ui>, std::sync::mpsc::Receiver<Vec<ClusterId>>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let ui = Ui {
            selection: Arc::new(Mutex::new(vec![ClusterId(1)])),
            select: tx,
            wake: Arc::new(|| {}),
        };
        (Arc::new(ui), rx)
    }

    #[test]
    fn brief_tool_returns_plain_text_levels() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        let mut env = Env::new(dir.path(), false, label);
        let mut text = |id, args: Value| {
            let r = handle_line(&h, &mut env, &call(id, "brief", args)).unwrap();
            assert_eq!(r["result"]["isError"], false, "{r}");
            r["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .to_string()
        };
        let l1 = text(1, json!({}));
        assert!(l1.starts_with("# sorrel brief v1"), "{l1}");
        assert!(
            l1.contains("RPV: rp_contamination > 0.1") && !l1.contains("\ncards:\n"),
            "{l1}"
        );
        let l2 = text(2, json!({"level": 2, "clusters": [0]}));
        assert!(
            l2.contains("\ncards:\nid\tlab\tflags\ttags\t")
                && l2.contains("\tacg\tamp_t\tev\n")
                && l2.contains("\n0\t"),
            "{l2}"
        );
        let tiny = text(3, json!({"level": 2, "budget": 10}));
        assert!(tiny.contains("omitted:"), "{tiny}");
    }

    #[test]
    fn probe_split_registers_a_handle_that_split_applies() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        let mut env = Env::new(dir.path(), false, label);
        let mut run =
            |id, name: &str, args: Value| handle_line(&h, &mut env, &call(id, name, args)).unwrap();
        // mock rate is 1000 Hz: cluster 0 spikes at 10, 30, 150 samples
        let r = run(
            1,
            "probe",
            json!({"q": ["split 0 by=time at=0.1", "frob", "window 0 0 1", "eval merge 0 1"]}),
        );
        let t = r["result"]["content"][0]["text"].as_str().unwrap();
        assert_eq!(r["result"]["isError"], false, "{t}");
        assert!(
            t.contains("> split 0 by=time at=0.1\ns0.1 time sc -: parts 2/1"),
            "{t}"
        );
        assert!(
            t.contains("error: unknown op \"frob\"")
                && t.ends_with("eval s<C>.<k>   |   eval m<k>"),
            "{t}"
        );
        assert!(
            t.contains("window 0 [0,1]s") && t.contains("merge {0,1}:"),
            "{t}"
        );
        let r = run(2, "split", json!({"candidate": "s0.1"}));
        let new = payload(&r)["new_cluster"].as_u64().unwrap() as u32;
        let s = h.read();
        assert_eq!(s.spike_times(ClusterId(new)), [SampleIndex(150)]);
        assert_eq!(s.spike_times(ClusterId(0)).len(), 2);
        drop(s);
        let r = run(3, "probe", json!({"q": "eval s0.1"}));
        assert!(r["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("stale handle"));
    }

    /// Handles resolve across unrelated edits, are refused once their
    /// cluster changed, and no tool output carries a spike index list.
    #[test]
    fn candidate_handles_resolve_until_their_cluster_changes() {
        let bimodal = |off: usize, n: usize| -> (Vec<SampleIndex>, Vec<f32>) {
            (
                (0..n)
                    .map(|i| SampleIndex((200 * i + off) as u64))
                    .collect(),
                (0..n)
                    .map(|i| (if i % 2 == 0 { 1.0 } else { 5.0 }) + ((i * 7919) % 13) as f32 * 0.01)
                    .collect(),
            )
        };
        let ((t0, a0), (t1, a1), (t2, a2)) = (bimodal(0, 400), bimodal(50, 10), bimodal(100, 400));
        let dir = tempfile::tempdir().unwrap();
        let prov = MockProvider {
            spikes: vec![t0, t1, t2],
            amps: vec![a0, a1, a2],
            n_samples: SampleIndex(100_000),
        };
        let mut s = Session::new(prov, Journal::open(&dir.path().join("j.journal")).unwrap());
        s.seed_amplitudes();
        let h = SessionHost::new(s);
        let mut env = Env::new(dir.path(), false, label);
        let mut run =
            |id, name: &str, args: Value| handle_line(&h, &mut env, &call(id, name, args)).unwrap();
        let msg = |r: &Value| {
            r["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .to_string()
        };
        let l3 = msg(&run(1, "brief", json!({"level": 3})));
        assert!(
            l3.contains("relations:\n") && l3.contains("\ns0.1 ") && l3.contains("\ns2.1 "),
            "{l3}"
        );
        let ok = |r: &Value| r["result"]["isError"] == false;
        // An unrelated edit leaves the handles usable.
        assert!(ok(&run(
            2,
            "relabel",
            json!({"cluster": 1, "label": "good"})
        )));
        assert!(ok(&run(3, "eval", json!({"candidate": "s0.1"}))));
        let r = run(4, "split", json!({"candidate": "s0.1"}));
        assert!(ok(&r), "{r}");
        // Its own cluster changed: refused, with the way out.
        let r = run(5, "split", json!({"candidate": "s0.1"}));
        assert!(!ok(&r) && msg(&r).contains("stale handle"), "{r}");
        assert!(ok(&run(6, "eval", json!({"candidate": "s2.1"}))));
        let r = run(
            7,
            "propose",
            json!({"action": "split", "candidate": "s2.1", "confidence": 0.5, "reasons": []}),
        );
        assert!(ok(&r), "{r}");
        for tool in [
            ("split_proposals", json!({"cluster": 2})),
            ("suggest", json!({})),
        ] {
            let out = msg(&run(8, tool.0, tool.1));
            assert!(!out.contains("bipartition_minor_idx"), "{out}");
        }
        let r = run(9, "eval", json!({"merge": [0, 2]}));
        assert!(msg(&r).starts_with("merge {0,2}: rpc "), "{r}");
        // A refresh after the split never reissues the retired s0.1 for the
        // changed cluster: the old id stays refused, new candidates get new ids.
        let l3 = msg(&run(10, "brief", json!({"level": 3})));
        assert!(!l3.contains("\ns0.1 "), "{l3}");
        let r = run(11, "split", json!({"candidate": "s0.1"}));
        assert!(!ok(&r) && msg(&r).contains("stale handle"), "{r}");
    }

    #[test]
    fn split_proposal_logs_its_basis() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        let mut env = Env::new(dir.path(), false, label);
        let r = handle_line(
            &h,
            &mut env,
            &call(
                1,
                "propose",
                json!({"action": "split", "clusters": [0], "spike_idx": [0], "confidence": 0.5, "reasons": []}),
            ),
        )
        .unwrap();
        assert_eq!(r["result"]["isError"], false, "{r}");
        let want = proposals::spike_basis(&h.read(), ClusterId(0));
        // The audit line (what a restart restores) carries the basis.
        assert_eq!(
            proposals::read_log(dir.path())[0].basis.as_deref(),
            Some(&*want)
        );
    }

    /// A direct split applies positions read from one membership only to that
    /// membership: an edit in between refuses it and applies nothing.
    #[test]
    fn direct_split_refuses_a_cluster_that_changed_since_it_was_read() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        let basis = proposals::spike_basis(&h.read(), ClusterId(0));
        h.apply(
            Actor::Human,
            CurationCommand::Merge {
                sources: vec![ClusterId(1)],
                target: ClusterId(0),
            },
        )
        .unwrap();
        let (n, v) = (h.read().n_clusters(), h.read().version());
        let e = apply_split(&h, Actor::Human, ClusterId(0), vec![0], &basis).unwrap_err();
        assert!(e.to_string().contains("changed since"), "{e}");
        assert_eq!((h.read().n_clusters(), h.read().version()), (n, v));
        let fresh = proposals::spike_basis(&h.read(), ClusterId(0));
        apply_split(&h, Actor::Human, ClusterId(0), vec![0], &fresh).unwrap();
    }

    #[test]
    fn propose_queues_on_the_host_and_changes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        let (ui, rx) = ui();
        let mut env = Env::new(dir.path(), false, label).with_ui(Some(ui));
        let version = h.read().version();
        let mut run = |line: String| handle_line(&h, &mut env, &line).unwrap();

        let ok = run(call(
            1,
            "propose",
            json!({"action": "merge", "clusters": [0, 1], "confidence": 0.9, "reasons": ["CCG dip"]}),
        ));
        assert_eq!(ok["result"]["isError"], false);
        let [p] = &h.proposals()[..] else {
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
            p.proposer.as_deref(),
            Some("agent"),
            "named after the client"
        );
        assert_eq!(
            proposals::read_log(dir.path())[0].id,
            p.id,
            "audited as pending"
        );

        // Invalid arguments are refused; nothing more is queued.
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
        ] {
            assert_eq!(run(bad)["result"]["isError"], true);
        }
        assert_eq!(h.proposals().len(), 1);

        let sel = run(call(6, "selection", json!({})));
        assert_eq!(payload(&sel)["clusters"], json!([1]));
        run(call(7, "select", json!({"clusters": [0]})));
        assert!(matches!(rx.try_recv(), Ok(c) if c == [ClusterId(0)]));
        let tools = run(rpc(8, "tools/list", json!({})));
        let names: Vec<&str> = tools["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"propose") && names.contains(&"relabel"));
        let q = run(call(9, "attention_queue", json!({})));
        assert_eq!(q["result"]["isError"], false);
        drop(run);
        assert_eq!(h.read().version(), version, "session untouched");
        assert!(h.read().tags(ClusterId(0)).is_empty());

        // Without a GUI, select and selection say so.
        let mut env = Env::new(dir.path(), false, label);
        let r = handle_line(&h, &mut env, &call(10, "selection", json!({}))).unwrap();
        assert_eq!(r["result"]["isError"], true);
    }

    /// The human flips to propose-only after validation: a batch or split
    /// that was validated under Direct must refuse and apply nothing.
    #[test]
    fn policy_flip_after_validation_refuses_apply() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        let mut env = Env::new(dir.path(), false, label);
        let version = h.read().version();
        h.set_policy(AgentPolicy::ProposeOnly);
        let relabel = CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: crate::command::PhyLabelOp::SetGood,
        };
        let e = apply_all(&h, &mut env, version, vec![relabel]).unwrap_err();
        assert!(err_msg(e).contains("propose-only"));
        let basis = proposals::spike_basis(&h.read(), ClusterId(0));
        let e = apply_split(&h, env.agent(), ClusterId(0), vec![0], &basis).unwrap_err();
        assert!(format!("{e:#}").contains("propose-only"));
        assert_eq!(h.read().version(), version, "nothing applied");
        h.set_policy(AgentPolicy::Direct);
        let relabel = CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: crate::command::PhyLabelOp::SetGood,
        };
        assert!(apply_all(&h, &mut env, version, vec![relabel]).is_ok());
    }

    /// Under propose-only, edit tools queue proposals (a merge keeps its
    /// target first) and the tools without a proposal form are refused.
    #[test]
    fn propose_only_turns_edits_into_proposals() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        h.set_policy(AgentPolicy::ProposeOnly);
        let mut env = Env::new(dir.path(), false, label);
        let version = h.read().version();
        let mut run = |line: String| handle_line(&h, &mut env, &line).unwrap();
        for (i, (tool, args)) in [
            ("relabel", json!({"cluster": 0, "label": "good"})),
            ("merge", json!({"sources": [1], "target": 0})),
            ("split", json!({"cluster": 0, "spike_idx": [0]})),
        ]
        .into_iter()
        .enumerate()
        {
            let r = run(call(i as u32, tool, args));
            assert_eq!(r["result"]["isError"], false, "{tool}: {r}");
            assert!(payload(&r)["status"]
                .as_str()
                .unwrap()
                .contains("NOT applied"));
        }
        for tool in ["tag", "undo", "save"] {
            let r = run(call(9, tool, json!({"cluster": 0, "tags": []})));
            assert_eq!(r["result"]["isError"], true, "{tool}");
        }
        drop(run);
        let queued: Vec<Action> = h.proposals().into_iter().map(|p| p.action).collect();
        assert_eq!(
            queued,
            [
                Action::Relabel {
                    cluster: ClusterId(0),
                    label: "good".into()
                },
                Action::Merge {
                    clusters: vec![ClusterId(0), ClusterId(1)]
                },
                Action::Split {
                    cluster: ClusterId(0),
                    spike_idx: vec![0]
                },
            ]
        );
        assert_eq!(h.read().version(), version, "nothing applied");
    }

    /// Changes by someone else reach the client as notifications; its own
    /// edits do not.
    #[test]
    fn other_actors_changes_become_notifications() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        let mut env = Env::new(dir.path(), false, label).subscribe(&h);
        handle_line(
            &h,
            &mut env,
            &call(1, "relabel", json!({"cluster": 0, "label": "good"})),
        );
        assert!(env.notifications().is_empty(), "own edit");
        h.apply(
            Actor::Human,
            CurationCommand::Relabel {
                cluster: ClusterId(1),
                op: crate::command::PhyLabelOp::SetNoise,
            },
        )
        .unwrap();
        let n = env.notifications();
        assert_eq!(n.len(), 1);
        assert_eq!(n[0]["method"], "notifications/message");
        assert_eq!(n[0]["params"]["data"]["actor"], "you");
        assert_eq!(n[0]["params"]["data"]["clusters"], json!([1]));
    }

    /// Two connections with the same client name (or both defaulting to
    /// "agent") still see each other's edits, and only their own are quiet.
    #[test]
    fn same_named_connections_notify_each_other() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        let mut a = Env::new(dir.path(), false, label).subscribe(&h);
        let mut b = Env::new(dir.path(), false, label).subscribe(&h);
        handle_line(
            &h,
            &mut a,
            &call(1, "relabel", json!({"cluster": 0, "label": "good"})),
        );
        assert!(a.notifications().is_empty(), "own edit");
        let n = b.notifications();
        assert_eq!(n.len(), 1, "b hears a's edit");
        assert_eq!(n[0]["params"]["data"]["actor"], "agent");
        handle_line(
            &h,
            &mut b,
            &call(2, "relabel", json!({"cluster": 1, "label": "noise"})),
        );
        assert!(b.notifications().is_empty(), "own edit");
        assert_eq!(a.notifications().len(), 1, "a hears b's edit");
    }

    #[test]
    fn headless_merge_rejects_target_among_sources() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        let mut env = Env::new(dir.path(), false, label);
        let version = h.read().version();
        let r = handle_line(
            &h,
            &mut env,
            &call(1, "merge", json!({"sources": [0, 1], "target": 0})),
        )
        .unwrap();
        assert_eq!(r["result"]["isError"], true);
        assert_eq!(h.read().version(), version);
        assert!(h.read().parents(ClusterId(0)).is_empty());
    }

    #[test]
    fn headless_relabel_tags_src_agent_and_triage_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        let mut env = Env::new(dir.path(), false, label);
        let mut run = |line: String| handle_line(&h, &mut env, &line).unwrap();
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
        assert_eq!(list["result"]["prompts"][0]["name"], "review");
        let got = run(rpc(5, "prompts/get", json!({"name": "review"})));
        let text = got["result"]["messages"][0]["content"]["text"]
            .as_str()
            .unwrap();
        assert!(text.contains("rules_eval") && text.contains("decide"));
        assert!(text.ends_with(RUBRIC) && RUBRIC.contains("trough_to_peak_ms"));
        let alias = run(rpc(7, "prompts/get", json!({"name": "triage"})));
        assert_eq!(
            alias["result"]["messages"], got["result"]["messages"],
            "triage aliases review"
        );
        assert_eq!(
            run(rpc(6, "prompts/get", json!({"name": "nope"})))["error"]["code"],
            -32602
        );
        drop(run);
        let mut s = h.write();
        assert_eq!(s.label(ClusterId(0)), Some(1));
        assert!(s.tags(ClusterId(0)).iter().any(|t| t == "src:agent"));
        let node = s.node(s.head().unwrap()).unwrap();
        assert_eq!(
            node.1,
            &Actor::Agent {
                client: "agent".into()
            }
        );
        // One undo reverts the relabel's tag and label together.
        s.dispatch(CurationCommand::Undo).unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        assert!(s.tags(ClusterId(0)).is_empty());
    }

    #[test]
    fn headless_split_origin_names_the_mcp_client() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        let mut env = Env::new(dir.path(), false, label);
        let mut run = |line: String| handle_line(&h, &mut env, &line).unwrap();
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
        let s = h.read();
        let n = s.n_clusters();
        assert_eq!(s.origin_label(ClusterId(n - 2)), "Sorrel[LLM:agent]");
        assert_eq!(s.origin_label(ClusterId(n - 1)), "Sorrel[LLM:codex]");
    }

    /// The listener writes where `attach` reads, and the path is a pure
    /// function of the dataset directory.
    #[test]
    fn token_file_path_is_shared_by_writer_and_reader() {
        let _u = scoped_user_dir();
        let dir = tempfile::tempdir().unwrap();
        let path = mcp_json_path(dir.path()).unwrap();
        assert_eq!(path, mcp_json_path(dir.path()).unwrap());
        assert_eq!(
            write_mcp_json(dir.path(), 1, "t", "n", "gui").unwrap(),
            path
        );
        assert_eq!(read_mcp_json(dir.path()), Some((1, "t".to_owned())));
        let _ = std::fs::remove_file(&path);
    }

    /// Off the dataset directory, so another account that can read the
    /// dataset cannot read the token; the record names the dataset and kind.
    #[test]
    fn token_file_is_outside_the_dataset_and_lists_as_a_host() {
        let user = scoped_user_dir();
        let dir = tempfile::tempdir().unwrap();
        let path = write_mcp_json(dir.path(), 1, "t", "n", "gui").unwrap();
        assert!(path.starts_with(user.path()) && !path.starts_with(dir.path()));
        let hosts = list_hosts();
        assert_eq!(hosts.len(), 1);
        assert_eq!(hosts[0].dataset, dir.path().canonicalize().unwrap());
        assert_eq!((hosts[0].kind.as_str(), hosts[0].port), ("gui", 1));
    }

    /// Live port, dead port and garbage: only the first is alive, the
    /// garbage is skipped, nothing is deleted.
    #[test]
    fn list_hosts_in_reports_liveness_and_skips_garbage() {
        let dir = tempfile::tempdir().unwrap();
        let live = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let dead = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let dead_port = dead.local_addr().unwrap().port();
        drop(dead);
        let rec = |name: &str, dataset: &str, port: u16| {
            let body = json!({"port": port, "token": "t", "pid": 7, "nonce": "n",
                              "dataset": dataset, "kind": "headless"});
            std::fs::write(dir.path().join(name), body.to_string()).unwrap();
        };
        rec("b.json", "/b", live.local_addr().unwrap().port());
        rec("a.json", "/a", dead_port);
        std::fs::write(dir.path().join("c.json"), "not json").unwrap();
        let hosts = list_hosts_in(dir.path());
        let got: Vec<_> = hosts
            .iter()
            .map(|h| (h.dataset.to_str().unwrap(), h.alive))
            .collect();
        assert_eq!(got, [("/a", false), ("/b", true)]);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 3);
    }

    /// A symlink pre-placed at the tmp path is replaced, not written through.
    #[cfg(unix)]
    #[test]
    fn write_does_not_follow_a_preplaced_tmp_symlink() {
        let _u = scoped_user_dir();
        let dir = tempfile::tempdir().unwrap();
        let victim = dir.path().join("victim");
        std::fs::write(&victim, "keep").unwrap();
        let path = mcp_json_path(dir.path()).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
        std::os::unix::fs::symlink(&victim, &tmp).unwrap();
        write_mcp_json(dir.path(), 1, "t", "n", "gui").unwrap();
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "keep");
        assert_eq!(read_mcp_json(dir.path()).unwrap().1, "t");
    }

    #[test]
    fn silent_connections_cannot_exhaust_handshake_workers() {
        let _u = scoped_user_dir();
        let dir = tempfile::tempdir().unwrap();
        let h = Arc::new(SessionHost::new(mock_session(dir.path())));
        let handle = spawn_listener(h, dir.path(), false, label, None).unwrap();
        let connect = || std::net::TcpStream::connect(("127.0.0.1", handle.port())).unwrap();
        // Fill every handshake slot with a client that never sends a token.
        let _silent: Vec<_> = (0..MAX_HANDSHAKES).map(|_| connect()).collect();
        // Give the listener time to accept them all.
        std::thread::sleep(std::time::Duration::from_millis(200));
        // One more is dropped at once (EOF), not parked on a thread for 5 s.
        let mut extra = connect();
        extra
            .set_read_timeout(Some(std::time::Duration::from_secs(2)))
            .unwrap();
        let mut got = Vec::new();
        let started = std::time::Instant::now();
        extra.read_to_end(&mut got).unwrap();
        assert!(got.is_empty());
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }

    #[test]
    fn loopback_listener_round_trip_and_bad_token() {
        let _u = scoped_user_dir();
        let dir = tempfile::tempdir().unwrap();
        let h = Arc::new(SessionHost::new(mock_session(dir.path())));
        let woke = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let w = woke.clone();
        let _gui = h.subscribe(move || {
            w.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });
        let handle = spawn_listener(h.clone(), dir.path(), false, label, None).unwrap();
        let (port, token) = read_mcp_json(dir.path()).unwrap();
        assert_eq!((port, token.as_str()), (handle.port(), handle.token()));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(mcp_json_path(dir.path()).unwrap())
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
        assert_eq!(replies[2]["result"]["isError"], false, "direct edit");
        assert_eq!(h.proposals().len(), 1);
        // The GUI's subscriber woke for the proposal and for the edit.
        assert_eq!(woke.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert_eq!(h.read().label(ClusterId(0)), Some(1), "relabelled");

        // `call_once` returns after one reply (the server hangs up on EOF).
        let req = json!({"jsonrpc": "2.0", "id": 9, "method": "tools/list"});
        assert_eq!(call_once(dir.path(), &req).unwrap().unwrap()["id"], 9);

        drop(handle);
        assert!(!mcp_json_path(dir.path()).unwrap().exists());
        assert!(call_once(dir.path(), &req).unwrap().is_none());
        let mut out = Vec::new();
        assert!(
            !attach(dir.path(), Cursor::new(""), &mut out).unwrap(),
            "no listener: caller falls back"
        );
    }

    /// Direct-mode `split` applies the same proper-subset rule as `propose`.
    #[test]
    fn direct_split_rejects_empty_full_and_out_of_range_indices() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        let mut env = Env::new(dir.path(), false, label);
        let mut run = |idx: Value| {
            let line = call(1, "split", json!({"cluster": 0, "spike_idx": idx}));
            handle_line(&h, &mut env, &line).unwrap()
        };
        for bad in [json!([]), json!([0, 1, 2]), json!([0, 7]), json!([3])] {
            assert_eq!(run(bad.clone())["result"]["isError"], true, "{bad}");
        }
        drop(run);
        assert_eq!(h.read().n_clusters(), 2, "nothing was split");
        assert_eq!(h.read().history_len(), 0);
    }

    /// A refused connection (stale `mcp.json`) falls back; a host that
    /// accepts but never acknowledges is alive, so that is an error.
    #[test]
    fn attach_falls_back_only_when_nobody_listens() {
        let _u = scoped_user_dir();
        let dir = tempfile::tempdir().unwrap();
        let none = || Cursor::new("");
        // No file.
        assert!(!attach(dir.path(), none(), Vec::new()).unwrap());
        // Stale file: the port is closed.
        let dead = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = dead.local_addr().unwrap().port();
        drop(dead);
        write_mcp_json(dir.path(), port, "t", "n", "gui").unwrap();
        assert!(!attach(dir.path(), none(), Vec::new()).unwrap());
        // Live but silent: accepts, never answers.
        let silent = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        write_mcp_json(
            dir.path(),
            silent.local_addr().unwrap().port(),
            "t",
            "n",
            "gui",
        )
        .unwrap();
        let hold = std::thread::spawn(move || silent.accept().map(|(s, _)| s));
        let err = attach(dir.path(), none(), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("did not acknowledge"), "{err}");
        drop(hold.join().unwrap());
    }

    /// A wrong token after connecting is an error too, not a fallback.
    #[test]
    fn attach_with_a_rejected_token_is_an_error() {
        let _u = scoped_user_dir();
        let dir = tempfile::tempdir().unwrap();
        let h = Arc::new(SessionHost::new(mock_session(dir.path())));
        let _handle = spawn_listener(h, dir.path(), false, label, None).unwrap();
        let (port, _) = read_mcp_json(dir.path()).unwrap();
        write_mcp_json(dir.path(), port, "wrong", "n", "gui").unwrap();
        let err = attach(dir.path(), Cursor::new(""), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("rejected"), "{err}");
    }

    /// The handle that wrote `mcp.json` is the only one that removes it.
    #[test]
    fn a_second_handles_drop_keeps_the_first_ones_file() {
        let _u = scoped_user_dir();
        let (a, b, shared) = (
            tempfile::tempdir().unwrap(),
            tempfile::tempdir().unwrap(),
            tempfile::tempdir().unwrap(),
        );
        let host = |d: &tempfile::TempDir| Arc::new(SessionHost::new(mock_session(d.path())));
        let first = spawn_listener(host(&a), shared.path(), false, label, None).unwrap();
        let second = spawn_listener(host(&b), shared.path(), false, label, None).unwrap();
        // The later host owns the file now; the earlier one must not delete it.
        drop(first);
        assert_eq!(read_mcp_json(shared.path()).unwrap().0, second.port());
        drop(second);
        assert!(!mcp_json_path(shared.path()).unwrap().exists());
    }

    /// A second host on a directory whose journal is already open fails
    /// with the busy error and leaves the first host's listener alone.
    #[test]
    fn a_second_host_on_the_same_directory_is_refused() {
        let _u = scoped_user_dir();
        let dir = tempfile::tempdir().unwrap();
        let h = Arc::new(SessionHost::new(mock_session(dir.path())));
        let first = spawn_listener(h, dir.path(), false, label, None).unwrap();
        let err = Journal::open(&dir.path().join("j.journal")).unwrap_err();
        let busy = err.downcast_ref::<crate::journal::JournalBusy>().unwrap();
        assert!(busy.to_string().contains("already open"), "{busy}");
        assert_eq!(read_mcp_json(dir.path()).unwrap().0, first.port());
    }

    const RULES: &str = r#"
        default = "mua"
        [[rule]]
        name = "tiny"
        metric = "n_spikes"
        op = "<"
        value = 3
        label = "noise"
        [[rule]]
        name = "steady"
        metric = "n_spikes"
        op = ">="
        value = 3
        label = "good"
        all = [{ metric = "drift_correlation", op = "|>", value = 5.0 }]
    "#;

    /// rules_eval reports per-rule counts, transitions and a unit sitting
    /// exactly on a threshold; rules_apply relabels and remembers the rules.
    #[test]
    fn rules_eval_then_apply_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        let mut env = Env::new(dir.path(), false, label);
        let r = handle_line(
            &h,
            &mut env,
            &call(1, "rules_eval", json!({"rules": RULES})),
        )
        .unwrap();
        assert_eq!(r["result"]["isError"], false, "{r}");
        let t = r["result"]["content"][0]["text"].as_str().unwrap();
        assert!(t.contains("tiny: n_spikes<3 -> noise: 1"), "{t}");
        assert!(t.contains("default -> mua: 1"), "{t}");
        assert!(
            t.contains("unsorted->noise 1: 1") && t.contains("unsorted->mua 1: 0"),
            "{t}"
        );
        assert!(
            t.contains("0\t<n_spikes\t3\t3\tdefault->mua"),
            "margin unit: {t}"
        );
        let r = handle_line(&h, &mut env, &call(2, "rules_eval", json!({"rules": "[[rule]]\nmetric = \"nope\"\nop = \"<\"\nvalue = 1\nlabel = \"good\"\n"}))).unwrap();
        assert_eq!(r["result"]["isError"], true);

        let r = handle_line(&h, &mut env, &call(3, "rules_apply", json!({}))).unwrap();
        assert_eq!(r["result"]["isError"], false, "{r}");
        assert_eq!(
            payload(&r)["transitions"],
            json!(["unsorted->mua 1: 0", "unsorted->noise 1: 1"])
        );
        assert_eq!(h.read().label(ClusterId(0)), Some(2));
        assert_eq!(h.read().label(ClusterId(1)), Some(3));
        assert!(std::fs::read_to_string(rules_path(dir.path()))
            .unwrap()
            .contains("tiny"));
    }

    #[test]
    fn rules_apply_queues_proposals_under_propose_only() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        h.set_policy(AgentPolicy::ProposeOnly);
        let mut env = Env::new(dir.path(), false, label);
        let version = h.read().version();
        let r = handle_line(
            &h,
            &mut env,
            &call(1, "rules_apply", json!({"rules": RULES})),
        )
        .unwrap();
        assert_eq!(r["result"]["isError"], false, "{r}");
        assert!(payload(&r)["status"]
            .as_str()
            .unwrap()
            .contains("NOT applied"));
        assert_eq!(h.read().version(), version, "nothing applied");
        let reasons: Vec<String> = h.proposals().into_iter().flat_map(|p| p.reasons).collect();
        assert_eq!(reasons, ["rule:default", "rule:tiny"]);
        assert!(rules_path(dir.path()).exists());
    }

    fn texts(r: &Value) -> String {
        r["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string()
    }

    /// A mixed decide batch is one undo step; an invalid item applies nothing.
    #[test]
    fn decide_is_one_undo_step_and_all_or_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        let mut env = Env::new(dir.path(), false, label);
        let before = h.read().version();
        let bad = json!({"decisions": [[0, "good", 0.9, ["x"]], [1, "bogus", 0.9, []], [9, "noise", 1, []]]});
        let r = handle_line(&h, &mut env, &call(1, "decide", bad)).unwrap();
        assert_eq!(r["result"]["isError"], true, "{r}");
        let t = texts(&r);
        assert!(
            t.contains("1: ") && t.contains("2: ") && !t.contains("\n0: "),
            "{t}"
        );
        assert_eq!(h.read().version(), before, "nothing applied");
        let long = json!({"decisions": [[0, "good", 0.9, ["one two three four five six seven eight nine"]]]});
        let r = handle_line(&h, &mut env, &call(2, "decide", long)).unwrap();
        assert_eq!(r["result"]["isError"], true, "{r}");

        let ok = json!({"decisions": [[0, "good", 0.9, ["clean"]], {"target": 1, "action": "noise", "confidence": 0.8, "reasons": ["rpv"], "note": "artefact"}]});
        let r = handle_line(&h, &mut env, &call(3, "decide", ok)).unwrap();
        assert_eq!(r["result"]["isError"], false, "{r}");
        assert!(
            texts(&r).starts_with("ok 2 applied (1 undo step)"),
            "{}",
            texts(&r)
        );
        assert_eq!(h.read().label(ClusterId(0)), Some(1));
        assert_eq!(h.read().label(ClusterId(1)), Some(3));
        assert!(h.read().tags(ClusterId(0)).iter().any(|t| t == "src:agent"));
        handle_line(&h, &mut env, &call(4, "undo", json!({}))).unwrap();
        assert_eq!(h.read().label(ClusterId(0)), Some(0));
        assert_eq!(h.read().label(ClusterId(1)), Some(0));
        assert!(h.read().tags(ClusterId(0)).is_empty());
    }

    #[test]
    fn decide_queues_one_proposal_per_item_under_propose_only() {
        let dir = tempfile::tempdir().unwrap();
        let h = SessionHost::new(mock_session(dir.path()));
        h.set_policy(AgentPolicy::ProposeOnly);
        let mut env = Env::new(dir.path(), false, label);
        let v = h.read().version();
        let args = json!({"decisions": [[0, "good", 0.9, ["clean"]], [[0, 1], "merge", 0.5, []]]});
        let r = handle_line(&h, &mut env, &call(1, "decide", args)).unwrap();
        // cluster 0 appears twice: refused whole.
        assert_eq!(r["result"]["isError"], true, "{r}");
        let args = json!({"decisions": [[0, "good", 0.9, ["clean"]], [1, "accept", 0.5, []]]});
        let r = handle_line(&h, &mut env, &call(2, "decide", args)).unwrap();
        assert!(
            texts(&r).contains("queued") && texts(&r).contains("NOT applied"),
            "{r}"
        );
        assert_eq!(h.proposals().len(), 2);
        assert_eq!(h.read().version(), v);
    }

    /// Split and merge handles resolve inside decide and apply as one batch;
    /// rules_apply is one undo step too.
    #[test]
    fn decide_applies_handles_and_rules_apply_is_one_step() {
        let bimodal = |off: usize, n: usize| -> (Vec<SampleIndex>, Vec<f32>) {
            (
                (0..n)
                    .map(|i| SampleIndex((200 * i + off) as u64))
                    .collect(),
                (0..n)
                    .map(|i| (if i % 2 == 0 { 1.0 } else { 5.0 }) + ((i * 7919) % 13) as f32 * 0.01)
                    .collect(),
            )
        };
        let ((t0, a0), (t1, a1), (t2, a2)) = (bimodal(0, 400), bimodal(50, 10), bimodal(100, 400));
        let dir = tempfile::tempdir().unwrap();
        let prov = MockProvider {
            spikes: vec![t0, t1, t2],
            amps: vec![a0, a1, a2],
            n_samples: SampleIndex(100_000),
        };
        let mut s = Session::new(prov, Journal::open(&dir.path().join("j.journal")).unwrap());
        s.seed_amplitudes();
        let h = SessionHost::new(s);
        let mut env = Env::new(dir.path(), false, label);
        let mut run =
            |id, name: &str, args: Value| handle_line(&h, &mut env, &call(id, name, args)).unwrap();
        run(1, "brief", json!({"level": 3}));
        let r = run(
            2,
            "decide",
            json!({"decisions": [["s0.1", "split", 0.7, ["bimodal"]], [1, "mua", 0.6, []], ["s2.1", "merge", 1, []]]}),
        );
        assert_eq!(
            r["result"]["isError"], true,
            "wrong action for a handle: {r}"
        );
        let n = h.read().n_clusters();
        let r = run(
            3,
            "decide",
            json!({"decisions": [["s0.1", "split", 0.7, ["bimodal"]], [1, "mua", 0.6, []]]}),
        );
        assert_eq!(r["result"]["isError"], false, "{r}");
        assert!(
            texts(&r).contains("(1 undo step); new clusters"),
            "{}",
            texts(&r)
        );
        assert_eq!(h.read().n_clusters(), n + 1);
        assert_eq!(h.read().label(ClusterId(1)), Some(2));
        run(4, "undo", json!({}));
        assert_eq!(
            h.read().n_clusters(),
            n,
            "one undo reverts split and relabel"
        );
        assert_eq!(h.read().label(ClusterId(1)), Some(0));
        // A stale handle after an unrelated cluster edit still resolves; an edit to
        // its own cluster refuses the batch.
        run(5, "relabel", json!({"cluster": 0, "label": "good"}));
        let r = run(
            6,
            "decide",
            json!({"decisions": [["s0.1", "split", 0.7, []]]}),
        );
        assert_eq!(r["result"]["isError"], false, "{r}");
        // rules_apply: one undo step for several relabels.
        let v = h.read().version();
        let rules = "default = \"good\"\n[[rule]]\nname = \"tiny\"\nmetric = \"n_spikes\"\nop = \"<\"\nvalue = 100\nlabel = \"noise\"\n";
        let r = run(7, "rules_apply", json!({"rules": rules}));
        assert_eq!(r["result"]["isError"], false, "{r}");
        assert_eq!(payload(&r)["undo_steps"], 1);
        assert_ne!(h.read().version(), v);
        run(8, "undo", json!({}));
        assert_eq!(h.read().version(), v, "rules_apply reverted by one undo");
    }
}

//! `sorrel serve`: a JSON-RPC 2.0 server over stdio in the shape MCP
//! clients speak (`initialize`, `tools/list`, `tools/call`), so an agent or
//! a notebook can inspect metrics, ask for suggestions, see plots and
//! curate through the same journaled commands the GUI uses.
//!
//! One JSON object per line in, one per line out. Hand-rolled on
//! `serde_json`: a handful of synchronous tools on a single stdin loop do not
//! justify an async runtime.
//!
//! Plots come back as SVG resources (see `report`); PNG needs a GPU
//! surface and is deliberately out of scope.

use crate::command::{CurationCommand, PhyLabelOp};
use crate::qc_export::{collect_qc_rows, QcRow};
use crate::report::render_cluster_svgs;
use crate::save::{save_and_reseal_journal, save_to_phy};
use crate::session::{ApplyPhyLabel, Session};
use crate::suggest::{rank_merge_candidates, rank_split_candidates, SuggestConfig};
use anyhow::Result;
use serde_json::{json, Value};
use sorrel_io::{ClusterId, DataProvider, HasGeometry};
use std::io::{BufRead, Write};
use std::path::Path;

/// MCP protocol revision this server answers `initialize` with.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// Tool catalogue returned by `tools/list`.
pub fn tool_list() -> Value {
    let cluster = json!({"type": "object", "properties": {"cluster": {"type": "integer"}}, "required": ["cluster"]});
    let tools = json!([
        {"name": "list_clusters", "description": "Cluster ids with spike count, label and tags; optional label filter and tag-prefix filter (\"review:\" = the review queue)",
         "inputSchema": {"type": "object", "properties": {"label": {"type": "string"}, "tag": {"type": "string"}}}},
        {"name": "metrics", "description": "Quality metrics for one cluster (same columns as cluster_qc.tsv)",
         "inputSchema": cluster},
        {"name": "suggest", "description": "Ranked merge and split suggestions for the current curation",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "render", "description": "Template waveform, ISI histogram and autocorrelogram of a cluster as SVG",
         "inputSchema": cluster},
        {"name": "relabel", "description": "Set a cluster's label (good | mua | noise | unsorted); journaled and undoable",
         "inputSchema": {"type": "object", "properties": {"cluster": {"type": "integer"}, "label": {"type": "string"}}, "required": ["cluster", "label"]}},
        {"name": "merge", "description": "Merge source clusters into target; journaled and undoable",
         "inputSchema": {"type": "object", "properties": {"sources": {"type": "array", "items": {"type": "integer"}}, "target": {"type": "integer"}}, "required": ["sources", "target"]}},
        {"name": "split", "description": "Move the given spike indices (positions in the cluster's time-sorted spikes) into a new cluster",
         "inputSchema": {"type": "object", "properties": {"cluster": {"type": "integer"}, "spike_idx": {"type": "array", "items": {"type": "integer"}}}, "required": ["cluster", "spike_idx"]}},
        {"name": "tag", "description": "Replace a cluster's tag set (provenance/review markers, e.g. drop review:* once decided); journaled and undoable",
         "inputSchema": {"type": "object", "properties": {"cluster": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["cluster", "tags"]}},
        {"name": "evidence", "description": "What `sorrel auto` recorded for a cluster: classifier outputs, Bombcell failures, Claude verdicts and merge records",
         "inputSchema": cluster},
        {"name": "undo", "description": "Undo the last curation command", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "save", "description": "Write spike_clusters.npy, cluster_group.tsv and curation.json", "inputSchema": {"type": "object", "properties": {}}},
    ]);
    #[cfg(feature = "phy-export")]
    let tools = {
        let mut tools = tools;
        tools.as_array_mut().expect("array").push(json!(
        {"name": "export_phy", "description": "Write the curated session as a complete phy directory at out (params.py references the raw file unless copy_raw)",
         "inputSchema": {"type": "object", "properties": {"out": {"type": "string"}, "copy_raw": {"type": "boolean"}}, "required": ["out"]}}
        ));
        tools
    };
    tools
}

/// Serve until `input` closes. `reseal_on_save` follows the same rule as
/// the GUI: true only for phy directories.
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
    let mut qc: Option<(u64, Vec<QcRow>)> = None;
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                write_json(&mut output, &rpc_error(Value::Null, -32700, &e.to_string()))?;
                continue;
            }
        };
        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(Value::Null);
        if id.is_null() {
            continue; // notification (e.g. notifications/initialized)
        }
        let reply = match method {
            "initialize" => Ok(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "sorrel", "version": env!("CARGO_PKG_VERSION")},
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({"tools": tool_list()})),
            "tools/call" => {
                let name = params.get("name").and_then(Value::as_str).unwrap_or("");
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                match call_tool(
                    session,
                    save_dir,
                    reseal_on_save,
                    &label_str,
                    &mut qc,
                    name,
                    &args,
                ) {
                    Ok(content) => Ok(json!({"content": content, "isError": false})),
                    Err(ToolError::Unknown) => Err((-32601, format!("unknown tool {name}"))),
                    Err(ToolError::Failed(msg)) => {
                        Ok(json!({"content": [{"type": "text", "text": msg}], "isError": true}))
                    }
                }
            }
            other => Err((-32601, format!("unknown method {other}"))),
        };
        let msg = match reply {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Err((code, message)) => rpc_error(id, code, &message),
        };
        write_json(&mut output, &msg)?;
    }
    Ok(())
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

fn call_tool<P, F>(
    session: &mut Session<P>,
    save_dir: &Path,
    reseal_on_save: bool,
    label_str: &F,
    qc: &mut Option<(u64, Vec<QcRow>)>,
    name: &str,
    args: &Value,
) -> Result<Value, ToolError>
where
    P: DataProvider + ApplyPhyLabel + HasGeometry,
    F: Fn(&P::Label) -> &'static str + Sync,
{
    let text = |v: Value| json!([{"type": "text", "text": v.to_string()}]);
    let check = |c: u32| -> Result<ClusterId, ToolError> {
        if c < session.n_clusters() {
            Ok(ClusterId(c))
        } else {
            Err(ToolError::Failed(format!("cluster {c} out of range")))
        }
    };
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
                               "label": label, "tags": session.tags(c)})
                    })
                })
                .collect();
            text(json!(rows))
        }
        "tag" => {
            let cluster = check(arg_u32(args, "cluster")?)?;
            let tags: Vec<String> = args
                .get("tags")
                .and_then(Value::as_array)
                .and_then(|a| a.iter().map(|v| v.as_str().map(str::to_owned)).collect())
                .ok_or_else(|| ToolError::Failed("missing string list \"tags\"".into()))?;
            if let Some(bad) = tags.iter().find(|t| !sorrel_io::kilosort::valid_tag(t)) {
                return Err(ToolError::Failed(format!("invalid tag {bad:?}")));
            }
            session.dispatch(CurationCommand::Tag { cluster, tags })?;
            text(json!({"ok": true, "version": session.version()}))
        }
        "evidence" => {
            let c = check(arg_u32(args, "cluster")?)?;
            let uid = u64::from(session.provider.unit_id(c));
            let path = save_dir.join("sorrel_auto").join("evidence.json");
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
            // Merged units are keyed by their post-merge id and list members.
            let unit = doc["units"].as_object().and_then(|units| {
                units
                    .iter()
                    .find(|(k, v)| k.parse() == Ok(uid) || has(&v["members"]))
                    .map(|(_, v)| v.clone())
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
            let c = check(arg_u32(args, "cluster")?)?;
            let version = session.version();
            if qc.as_ref().is_none_or(|(v, _)| *v != version) {
                *qc = Some((version, collect_qc_rows(session)));
            }
            let row = qc
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
            let c = check(arg_u32(args, "cluster")?)?;
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
        "relabel" => {
            let cluster = check(arg_u32(args, "cluster")?)?;
            let op = match args.get("label").and_then(Value::as_str).unwrap_or("") {
                "good" => PhyLabelOp::SetGood,
                "mua" => PhyLabelOp::SetMua,
                "noise" => PhyLabelOp::SetNoise,
                "unsorted" => PhyLabelOp::SetUnsorted,
                other => return Err(ToolError::Failed(format!("unknown label {other:?}"))),
            };
            session.dispatch(CurationCommand::Relabel { cluster, op })?;
            text(json!({"ok": true, "version": session.version()}))
        }
        "merge" => {
            let sources: Vec<ClusterId> = arg_u32_list(args, "sources")?
                .into_iter()
                .map(check)
                .collect::<Result<_, _>>()?;
            let target = check(arg_u32(args, "target")?)?;
            session.dispatch(CurationCommand::Merge { sources, target })?;
            text(json!({"ok": true, "version": session.version()}))
        }
        "split" => {
            let cluster = check(arg_u32(args, "cluster")?)?;
            let spike_idx = arg_u32_list(args, "spike_idx")?;
            session.dispatch(CurationCommand::Split {
                cluster,
                spike_idx,
                new_cluster: ClusterId(u32::MAX),
            })?;
            let new_cluster = session.n_clusters() - 1;
            text(json!({"ok": true, "new_cluster": new_cluster, "version": session.version()}))
        }
        "undo" => {
            session.dispatch(CurationCommand::Undo)?;
            text(json!({"ok": true, "version": session.version()}))
        }
        "save" => {
            if reseal_on_save {
                let journal_path = session.journal_path().to_path_buf();
                let journal = save_and_reseal_journal(session, save_dir, &journal_path, label_str)?;
                session.reseal(journal);
            } else {
                save_to_phy(session, save_dir, label_str)?;
            }
            text(json!({"ok": true, "save_dir": save_dir.display().to_string()}))
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
            crate::phy_export::export_phy(session, Path::new(out), copy_raw, label_str)?;
            text(json!({"ok": true, "out": out}))
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
            11 + usize::from(cfg!(feature = "phy-export"))
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
            r#"{"units":{"7":{"members":[0,1],"bombcell":"mua"}},"merges":[{"units":[0,1],"verdict":null}]}"#,
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
        assert_eq!(s.tags(ClusterId(1)), ["review:merge", "src:bombcell"]);
        assert_eq!(
            text(1),
            json!([{"cluster": 1, "n_spikes": 1, "label": "unsorted", "tags": ["review:merge", "src:bombcell"]}])
        );
        assert_eq!(
            replies[2]["result"]["isError"], true,
            "comma would corrupt the TSV"
        );
        assert_eq!(text(3)["unit"]["bombcell"], "mua");
        assert_eq!(text(3)["merges"].as_array().unwrap().len(), 1);
    }
}

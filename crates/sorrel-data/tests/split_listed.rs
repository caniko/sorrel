//! A split-off cluster is listed by the MCP `list_clusters` straight away, after
//! `save`, and after a reopen. The new id had no label slot, so `list_clusters`
//! (which skips clusters without a label) dropped it and `cluster_group.tsv`
//! could not carry its label either.

use serde_json::Value;
use sorrel_data::{open_phy, seed_phy, serve::serve, JournalOpts, SessionHost};
use sorrel_io::kilosort::KilosortOpenParams;
use std::io::Cursor;
use std::path::Path;

fn call(id: u32, name: &str, args: &str) -> String {
    format!(
        r#"{{"jsonrpc":"2.0","id":{id},"method":"tools/call","params":{{"name":"{name}","arguments":{args}}}}}"#
    ) + "\n"
}

/// Ids listed by each `list_clusters` reply, in request order.
fn run(root: &Path, requests: &[String]) -> Vec<Vec<u32>> {
    let opts = JournalOpts::default();
    let mut s = open_phy(root, KilosortOpenParams::default(), &opts).unwrap();
    seed_phy(&mut s).unwrap();
    let host = SessionHost::new(s);
    let mut out = Vec::new();
    serve(
        &host,
        root,
        true,
        |l: &sorrel_io::kilosort::PhyLabel| l.as_str(),
        Cursor::new(requests.concat()),
        &mut out,
    )
    .unwrap();
    String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .filter_map(|r| {
            assert_ne!(r["result"]["isError"], true, "{r}");
            let t = r["result"]["content"][0]["text"].as_str()?;
            t.starts_with("id\tn\t").then(|| {
                t.lines()
                    .skip(1)
                    .map(|row| row.split('\t').next().unwrap().parse().unwrap())
                    .collect()
            })
        })
        .collect()
}

#[test]
fn split_child_is_listed_live_after_save_and_after_reopen() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../release/fixtures/tiny-phy");
    // The Nix source filter drops release/, so the nix checks have no copy.
    if !fixture.exists() {
        eprintln!("skipping: {} not in this source tree", fixture.display());
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    for e in std::fs::read_dir(&fixture).unwrap().flatten() {
        std::fs::copy(e.path(), dir.path().join(e.file_name())).unwrap();
    }
    let root = dir.path();

    let lists = run(
        root,
        &[
            call(1, "list_clusters", "{}"),
            call(2, "relabel", r#"{"cluster":0,"label":"good"}"#),
            call(3, "save", "{}"),
            call(
                4,
                "split",
                r#"{"cluster":1,"spike_idx":[0,1,2,3,4,5,6,7,8,9]}"#,
            ),
            call(5, "list_clusters", "{}"),
            call(6, "relabel", r#"{"cluster":3,"label":"mua"}"#),
            call(7, "save", "{}"),
            call(8, "list_clusters", "{}"),
        ],
    );
    assert_eq!(lists[0], [0, 1, 2]);
    assert_eq!(lists[1], [0, 1, 2, 3], "live, before save");
    assert_eq!(lists[2], [0, 1, 2, 3], "live, after save");
    let group = std::fs::read_to_string(root.join("cluster_group.tsv")).unwrap();
    assert!(group.contains("3\tmua"), "{group}");

    let again = run(root, &[call(1, "list_clusters", "{}")]);
    assert_eq!(again[0], [0, 1, 2, 3], "after reopen");
}

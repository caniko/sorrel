//! The agent-facing CLI on the real binary. `SORREL_USER_DIR` points at an
//! empty directory, so no real GUI or host is found and `call` opens the
//! dataset in-process.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn sorrel(user_dir: &Path, args: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sorrel"))
        .args(args)
        .env("SORREL_USER_DIR", user_dir)
        .output()
        .unwrap()
}

fn run(user_dir: &Path, args: &[&str]) -> Output {
    let args: Vec<PathBuf> = args.iter().map(PathBuf::from).collect();
    sorrel(
        user_dir,
        &args.iter().map(PathBuf::as_path).collect::<Vec<_>>(),
    )
}

/// `release/fixtures/tiny-phy` copied out, since `call` writes a journal.
/// `None` when the source has no fixture: the Nix package check filters
/// `release/` out (checks.workspace-tests keeps it).
fn fixture(into: &Path) -> Option<PathBuf> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../release/fixtures/tiny-phy");
    if !src.join("spike_times.npy").exists() {
        eprintln!("skipping: no tiny-phy fixture in this source tree");
        return None;
    }
    let dst = into.join("tiny-phy");
    std::fs::create_dir(&dst).unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), dst.join(e.file_name())).unwrap();
    }
    Some(dst)
}

#[test]
fn call_runs_a_tool_in_process_and_prints_json() {
    let tmp = tempfile::tempdir().unwrap();
    let Some(ds) = fixture(tmp.path()) else {
        return;
    };
    let out = sorrel(tmp.path(), &[Path::new("call"), &ds, Path::new("suggest")]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice::<serde_json::Value>(&out.stdout).expect("stdout is JSON");

    // A text tool prints as text: `sorrel call DIR brief` is the brief CLI.
    let brief = sorrel(tmp.path(), &[Path::new("call"), &ds, Path::new("brief")]);
    assert!(
        String::from_utf8_lossy(&brief.stdout).starts_with("# sorrel brief v1"),
        "{}",
        String::from_utf8_lossy(&brief.stderr)
    );

    let raw = sorrel(
        tmp.path(),
        &[
            Path::new("call"),
            &ds,
            Path::new("list_clusters"),
            Path::new("--raw"),
        ],
    );
    let v: serde_json::Value = serde_json::from_slice(&raw.stdout).unwrap();
    assert_eq!(v["jsonrpc"], "2.0");
}

#[test]
fn call_fails_on_unknown_tool_and_bad_args() {
    let tmp = tempfile::tempdir().unwrap();
    let Some(ds) = fixture(tmp.path()) else {
        return;
    };
    let bad_tool = sorrel(
        tmp.path(),
        &[Path::new("call"), &ds, Path::new("no_such_tool")],
    );
    assert!(!bad_tool.status.success());
    let bad_args = sorrel(
        tmp.path(),
        &[
            Path::new("call"),
            &ds,
            Path::new("list_clusters"),
            Path::new("[1]"),
        ],
    );
    assert!(!bad_args.status.success());
    assert!(String::from_utf8_lossy(&bad_args.stderr).contains("JSON object"));
}

#[test]
fn offline_commands_work() {
    let tmp = tempfile::tempdir().unwrap();
    let tools = run(tmp.path(), &["mcp", "tools", "--json"]);
    assert!(tools.status.success());
    let v: serde_json::Value = serde_json::from_slice(&tools.stdout).unwrap();
    assert!(!v["tools"].as_array().unwrap().is_empty());

    let completions = run(tmp.path(), &["completions", "bash"]);
    assert!(completions.status.success() && !completions.stdout.is_empty());

    let status = run(tmp.path(), &["status", "--json"]);
    let v: serde_json::Value = serde_json::from_slice(&status.stdout).unwrap();
    assert!(v["gui"].is_null() && v["hosts"].as_array().unwrap().is_empty());

    let none = run(tmp.path(), &["mcp"]);
    assert!(!none.status.success());
    assert!(String::from_utf8_lossy(&none.stderr).contains("no running sorrel host"));
}

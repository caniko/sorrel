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

#[test]
fn refractory_flags_change_qc_and_html_report_on_the_real_binary() {
    let tmp = tempfile::tempdir().unwrap();
    let Some(ds) = fixture(tmp.path()) else {
        return;
    };
    let ds = ds.to_str().unwrap();
    for (fixed_ms, expected) in [("1", 0), ("4", 666)] {
        let qc = tmp.path().join(format!("qc-{fixed_ms}"));
        let html = tmp.path().join(format!("report-{fixed_ms}.html"));
        for (cmd, out) in [("qc", &qc), ("report", &html)] {
            let result = run(
                tmp.path(),
                &[
                    cmd,
                    ds,
                    "--out",
                    out.to_str().unwrap(),
                    "--refractory-ms",
                    fixed_ms,
                    "--rp-min-ms",
                    "1.5",
                    "--rp-max-ms",
                    "7.25",
                    "--rp-step-ms",
                    "0.5",
                ],
            );
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
        let rows: serde_json::Value =
            serde_json::from_slice(&std::fs::read(qc.join("cluster_qc.json")).unwrap()).unwrap();
        let row = rows["clusters"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["cluster_id"] == 0)
            .unwrap();
        // The 667-spike fixture unit has no gap <30 samples and every gap
        // is <120 samples, at the declared 30 kHz.
        assert_eq!(row["isi_violations"], expected);
        assert!(row["mean_amplitude"].is_null());
        assert!(row["rp_contamination"].as_f64().unwrap() > 0.0);
        let rate = row["isi_violation_ratio"].as_f64().unwrap();
        let html = std::fs::read_to_string(html).unwrap();
        assert!(
            html.contains(&format!("<dt>ISI viol.</dt><dd>{rate:.0}</dd>")),
            "missing rendered rate {rate}"
        );
    }
}

#[test]
fn invalid_refractory_flags_fail_with_an_actionable_diagnostic() {
    let tmp = tempfile::tempdir().unwrap();
    for (args, diagnostic) in [
        (&["--refractory-ms=0"][..], "fixed_ms"),
        (&["--refractory-ms=NaN"][..], "finite"),
        (
            &["--rp-min-ms=9", "--rp-max-ms=8"][..],
            "sweep_min_ms must be less",
        ),
        (&["--rp-step-ms=0.0001"][..], "1 and 1000 windows"),
    ] {
        let mut command = vec!["qc", "unused-when-validation-fails"];
        command.extend(args);
        let result = run(tmp.path(), &command);
        assert!(!result.status.success());
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            stderr.contains("Invalid refractory settings") && stderr.contains(diagnostic),
            "{stderr}"
        );
    }
}

#[test]
fn qc_cli_distinguishes_confirmed_contamination_from_no_evidence() {
    let tmp = tempfile::tempdir().unwrap();
    let Some(ds) = fixture(tmp.path()) else {
        return;
    };
    let ds = ds.to_str().unwrap();
    let merge = run(
        tmp.path(),
        &["call", ds, "merge", r#"{"sources":[1],"target":2}"#],
    );
    assert!(
        merge.status.success(),
        "{}",
        String::from_utf8_lossy(&merge.stderr)
    );
    for (out, custom) in [
        ("default", &[][..]),
        (
            "unavailable",
            &[
                "--refractory-ms=0.03",
                "--rp-min-ms=0.015",
                "--rp-max-ms=0.035",
                "--rp-step-ms=0.02",
            ][..],
        ),
    ] {
        let qc = tmp.path().join(out);
        let mut args = vec!["qc", ds, "--out", qc.to_str().unwrap()];
        args.extend(custom);
        let result = run(tmp.path(), &args);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let rows: serde_json::Value =
            serde_json::from_slice(&std::fs::read(qc.join("cluster_qc.json")).unwrap()).unwrap();
        let row = |id| {
            rows["clusters"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["cluster_id"] == id)
                .unwrap()
        };
        let clean = row(0);
        let merged = row(2);
        assert!(merged["amplitude_snr"].is_null() && merged["drift_correlation"].is_null());
        if custom.is_empty() {
            assert_eq!(clean["srp_contamination"], 0.005);
            assert_eq!(clean["srp_rp_ms"], 1.0);
            assert!(
                merged["rpv_post_mean"].as_f64().unwrap()
                    > clean["rpv_post_mean"].as_f64().unwrap()
            );
            assert_eq!(merged["srp_contamination"], 0.275);
            assert_eq!(merged["srp_rp_ms"], 1.5);
        } else {
            for name in [
                "srp_contamination",
                "srp_rp_ms",
                "rpv_post_mean",
                "rpv_post_lo95",
                "rpv_post_hi95",
            ] {
                assert!(clean[name].is_null() && merged[name].is_null(), "{name}");
            }
        }
    }
}

#[cfg(feature = "hdf5")]
#[test]
fn nwb_call_preserves_exported_parent_without_a_row() {
    let tmp = tempfile::tempdir().unwrap();
    let src =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../release/fixtures/nwb/curated_units.nwb");
    let nwb = tmp.path().join("units.nwb");
    std::fs::copy(src, &nwb).unwrap();
    let out = sorrel(
        tmp.path(),
        &[Path::new("call"), &nwb, Path::new("list_clusters")],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("\t0,unit:77\t"), "{text}");
}

#[cfg(feature = "zarr")]
#[test]
fn nwb_zarr_call_keeps_the_journal_in_a_temp_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../release/fixtures/nwb/good.nwb.zarr");
    if !fixture.is_dir() {
        return; // Cargo packages omit workspace fixtures.
    }
    // `call` writes a journal next to the input unless explicitly redirected.
    let journal = tmp.path().join("nwb.sorrel.journal");
    let out = sorrel(
        tmp.path(),
        &[
            Path::new("call"),
            &fixture,
            Path::new("list_clusters"),
            Path::new("--journal"),
            &journal,
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(journal.is_file());
}

#[cfg(feature = "hdf5")]
struct Host(std::process::Child);

#[cfg(feature = "hdf5")]
impl Drop for Host {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[cfg(all(feature = "hdf5", unix))]
#[test]
fn nwb_headless_connect_hint_attaches_to_matching_host() {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;

    let tmp = tempfile::tempdir().unwrap();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../release/fixtures/nwb");
    std::fs::copy(
        fixtures.join("units_only.nwb"),
        tmp.path().join("units.nwb"),
    )
    .unwrap();
    std::fs::copy(
        fixtures.join("raw_only.nwb"),
        tmp.path().join("raw's trace.nwb"),
    )
    .unwrap();
    let mut host = Host(
        Command::new(env!("CARGO_BIN_EXE_sorrel"))
            .current_dir(tmp.path())
            .env("SORREL_USER_DIR", tmp.path())
            .args(["run", "--headless"])
            .arg(tmp.path().join("units.nwb"))
            .args([
                "--dat",
                "raw's trace.nwb",
                "--nwb-series",
                "ElectricalSeriesRaw",
                "--journal",
                "my journal",
                "--sample-rate",
                "1000",
                "--sorter=-manual",
                "--discard-journal",
            ])
            .stdout(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let line = BufReader::new(host.0.stdout.take().unwrap())
        .lines()
        .next()
        .unwrap()
        .unwrap();
    let (_, hint) = line.split_once("; connect an agent with `sorrel ").unwrap();
    let args = hint.strip_suffix("`; Ctrl-C stops").unwrap();
    assert!(!args.contains("--discard-journal"));
    // Use the printed shell arguments verbatim from a different directory.
    let out = Command::new("sh")
        .args([
            "-c",
            &format!("exec \"$1\" {args}"),
            "sh",
            env!("CARGO_BIN_EXE_sorrel"),
        ])
        .env("SORREL_USER_DIR", tmp.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{line}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(args.contains("units.nwb"), "{line}");
}

#[cfg(feature = "hdf5")]
#[test]
fn nwb_call_only_reuses_matching_running_session() {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;

    let tmp = tempfile::tempdir().unwrap();
    let src = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../release/fixtures/nwb/multiple_series.nwb");
    std::fs::copy(src, tmp.path().join("units.nwb")).unwrap();
    let mut host = Host(
        Command::new(env!("CARGO_BIN_EXE_sorrel"))
            .current_dir(tmp.path())
            .env("SORREL_USER_DIR", tmp.path())
            .args([
                "run",
                "--headless",
                "units.nwb",
                "--nwb-series",
                "A_first",
                "--journal",
                "first.journal",
            ])
            .stdout(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let stdout = BufReader::new(host.0.stdout.take().unwrap());
    assert!(stdout
        .lines()
        .next()
        .unwrap()
        .unwrap()
        .starts_with("sorrel host on"));
    let call = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_sorrel"))
            .current_dir(tmp.path())
            .env("SORREL_USER_DIR", tmp.path())
            .args(["call", "./units.nwb"])
            .args(extra)
            .arg("list_clusters")
            .output()
            .unwrap()
    };
    let same = call(&[
        "--nwb-series",
        "/acquisition/A_first",
        "--journal",
        "./first.journal",
    ]);
    assert!(
        same.status.success(),
        "{}",
        String::from_utf8_lossy(&same.stderr)
    );
    let discard = call(&[
        "--nwb-series",
        "A_first",
        "--journal",
        "first.journal",
        "--discard-journal",
    ]);
    assert!(!discard.status.success());
    let err = String::from_utf8_lossy(&discard.stderr);
    assert!(
        err.contains("stop the running session or omit the flag"),
        "{err}"
    );
    for different in [
        ["--journal", "first.journal", "--nwb-series", "Z_second"],
        ["--journal", "first.journal", "--nwb-series", "nonexistent"],
        ["--nwb-series", "A_first", "--journal", "other.journal"],
        ["--journal", "first.journal", "--dat", "raw.nwb"],
    ] {
        let out = call(&different);
        assert!(!out.status.success(), "{different:?}");
        assert!(String::from_utf8_lossy(&out.stderr).contains("session conflict"));
    }
}

//! `sorrel blind`, dir-mode `sorrel diff` and `agent-guide --prompt` on the
//! real binary. `SORREL_USER_DIR` points at an empty directory so `call`
//! opens the dataset in-process.

use std::path::Path;
use std::process::{Command, Output};

fn sorrel(user_dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sorrel"))
        .args(args)
        .env("SORREL_USER_DIR", user_dir)
        .env_remove("SORREL_RUBRIC")
        .env_remove("SORREL_REVIEW")
        .output()
        .unwrap()
}

fn ok(o: &Output) -> String {
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[test]
fn blind_then_diff_scores_the_curation_against_the_sort() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../release/fixtures/tiny-phy");
    if !src.join("spike_times.npy").exists() {
        return eprintln!("skipping: no tiny-phy fixture in this source tree");
    }
    let tmp = tempfile::tempdir().unwrap();
    let t = tmp.path();
    let ds = t.join("ds");
    std::fs::create_dir(&ds).unwrap();
    for e in std::fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), ds.join(e.file_name())).unwrap();
    }
    let s = |p: &Path| p.to_str().unwrap().to_owned();
    let (ds_s, input, truth) = (s(&ds), s(&t.join("in")), s(&t.join("truth")));

    ok(&sorrel(
        t,
        &["call", &ds_s, "relabel", r#"{"cluster":0,"label":"good"}"#],
    ));
    ok(&sorrel(
        t,
        &["blind", &ds_s, "--input", &input, "--truth", &truth],
    ));
    assert!(!t.join("in/sorrel.journal").exists() && !t.join("in/cluster_group.tsv").exists());
    let prov: serde_json::Value =
        serde_json::from_slice(&std::fs::read(t.join("truth/provenance.json")).unwrap()).unwrap();
    assert_eq!(prov["n_applied"], 1);

    let diff = sorrel(t, &["diff", &truth, &input, "--root", &input, "--json"]);
    let d: serde_json::Value = serde_json::from_str(&ok(&diff)).unwrap();
    assert_eq!(d["misses"][0]["category"], "label");
    assert!(
        !sorrel(t, &["diff", &truth, &input]).status.success(),
        "dirs need --root"
    );

    // The prompt a bench variant overrides, served exactly as `prompts/get` does.
    let guide = ok(&sorrel(t, &["agent-guide", "--prompt", "review"]));
    assert!(
        guide.starts_with("You are reviewing a spike sort") && guide.contains("trough_to_peak_ms")
    );
}

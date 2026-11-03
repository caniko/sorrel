//! `sorrel dev ...` and `diff --record` on the real binary (feature `dev`), over
//! the tiny phy fixture: a session, a run of an arm, its score, the holdout
//! rules and `distill`.
#![cfg(feature = "dev")]

use std::path::Path;
use std::process::{Command, Output};

fn sorrel(user_dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sorrel"))
        .args(args)
        .env("SORREL_USER_DIR", user_dir)
        .env_remove("SORREL_DEV_DB")
        .env_remove("SORREL_DEV_DENY")
        .output()
        .unwrap()
}

fn ok(o: &Output) -> String {
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[test]
fn a_session_a_run_and_its_score_land_in_the_results_db() {
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
    let (ds_s, input, truth, db) = (
        s(&ds),
        s(&t.join("in")),
        s(&t.join("truth")),
        s(&t.join("r.sqlite")),
    );
    let deny = t.join("deny.txt");
    std::fs::write(&deny, "sealed-sid\nsource_basename\n").unwrap();
    let dev = |args: &[&str]| {
        let mut a = vec!["dev", "--db", &db, "--deny", deny.to_str().unwrap()];
        a.extend_from_slice(args);
        sorrel(t, &a)
    };

    ok(&sorrel(
        t,
        &["call", &ds_s, "relabel", r#"{"cluster":0,"label":"good"}"#],
    ));
    ok(&sorrel(
        t,
        &["blind", &ds_s, "--input", &input, "--truth", &truth],
    ));
    assert!(
        !dev(&["db", "status"]).status.success(),
        "status does not create the DB"
    );
    ok(&dev(&["db", "init"]));

    ok(&dev(&[
        "record-session",
        "--sid",
        "s1",
        "--split",
        "dev",
        "--input",
        &input,
        "--truth",
        &truth,
    ]));
    assert!(
        !t.join("in/sorrel.journal").exists() && !t.join("in/.sorrel").exists(),
        "the input stays untouched"
    );
    ok(&dev(&[
        "record-session",
        "--sid",
        "holdout-1",
        "--split",
        "holdout",
    ]));
    assert!(!dev(&[
        "record-session",
        "--sid",
        "sealed-sid",
        "--split",
        "holdout"
    ])
    .status
    .success());
    assert!(!dev(&[
        "record-session",
        "--sid",
        "h2",
        "--split",
        "holdout",
        "--input",
        &input
    ])
    .status
    .success());

    // An arm's run: a copy of the input the arm curated.
    let run = t.join("runs/r1");
    let data = run.join("data");
    std::fs::create_dir_all(&data).unwrap();
    for e in std::fs::read_dir(&input).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), data.join(e.file_name())).unwrap();
    }
    ok(&sorrel(
        t,
        &[
            "call",
            &s(&data),
            "relabel",
            r#"{"cluster":0,"label":"good"}"#,
        ],
    ));
    std::fs::write(
        run.join("run.json"),
        r#"{"id":"r1","session":"s1","split":"dev","arm":"mcp-review","variant":"v0","repeat":0,"status":"ok","prompt_rev":"p1","cost_usd":0.5,"num_turns":3}"#,
    )
    .unwrap();
    ok(&dev(&["record-run", "--run-dir", &s(&run)]));
    let miss = dev(&["record-run", "--run-dir", &s(&t.join("nope"))]);
    assert!(!miss.status.success());

    let record = |rid: &str| {
        sorrel(
            t,
            &[
                "diff",
                &truth,
                &s(&data),
                "--root",
                &input,
                "--json",
                "--record",
                rid,
                "--db",
                &db,
            ],
        )
    };
    assert!(!record("unknown").status.success(), "a score needs its run");
    // Two label TSVs have no spikes to score: --record must refuse, not succeed silently.
    let (ta, tb) = (t.join("a.tsv"), t.join("b.tsv"));
    std::fs::write(&ta, "cluster_id\tgroup\n0\tgood\n").unwrap();
    std::fs::write(&tb, "cluster_id\tgroup\n0\tnoise\n").unwrap();
    let tsv = sorrel(
        t,
        &["diff", &s(&ta), &s(&tb), "--record", "r1", "--db", &db],
    );
    assert!(!tsv.status.success(), "--record on TSV mode must fail");
    ok(&record("r1"));
    ok(&record("r1")); // idempotent
                       // Another TEST than the run's own output is not that run's score.
    let other = sorrel(
        t,
        &[
            "diff", &input, &truth, "--root", &input, "--json", "--record", "r1", "--db", &db,
        ],
    );
    assert!(!other.status.success(), "a foreign TEST must be refused");
    assert!(
        String::from_utf8_lossy(&other.stderr).contains("not the run's recorded output"),
        "{}",
        String::from_utf8_lossy(&other.stderr)
    );
    // Nor is a score made against another TRUTH than the session's.
    let foreign = sorrel(
        t,
        &[
            "diff",
            &input,
            &s(&data),
            "--root",
            &input,
            "--json",
            "--record",
            "r1",
            "--db",
            &db,
        ],
    );
    assert!(!foreign.status.success(), "a foreign TRUTH must be refused");
    assert!(
        String::from_utf8_lossy(&foreign.stderr).contains("TRUTH is not the session's"),
        "{}",
        String::from_utf8_lossy(&foreign.stderr)
    );

    let q = |sql: &str| ok(&dev(&["db", "query", sql]));
    assert_eq!(q("SELECT COUNT(*) FROM runs"), "COUNT(*)\n1\n");
    assert!(q("SELECT fingerprint, split FROM sessions ORDER BY sid").contains("\tholdout\n"));
    assert!(
        q("SELECT COUNT(*) FROM root_cards WHERE sid = 's1'")
            .lines()
            .nth(1)
            .unwrap()
            .parse::<u32>()
            .unwrap()
            > 0
    );
    assert!(q("SELECT verdict FROM truth_roots WHERE sid = 's1'").contains('\n'));
    assert!(q("SELECT action FROM arm_actions WHERE rid = 'r1'").contains("relabel:good"));
    assert!(
        q("SELECT value FROM metrics WHERE name = 'good_units.recall'")
            .lines()
            .count()
            == 2
    );
    assert!(!dev(&["db", "query", "DELETE FROM runs"]).status.success());

    let out = t.join("learned");
    let rev = ok(&dev(&["distill", "--out", &s(&out)]));
    let tsv = std::fs::read_to_string(out.join("exemplars.tsv")).unwrap();
    assert!(
        rev.contains(
            tsv.lines()
                .nth(2)
                .unwrap()
                .trim_start_matches("# learned_rev=")
        ),
        "{rev}\n{tsv}"
    );
    for f in ["priors.toml", "thresholds.toml", "lineage.md"] {
        assert!(out.join(f).exists(), "{f}");
    }
}

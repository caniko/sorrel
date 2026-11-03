# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Self-check for bench_curation.py with stub `sorrel` and `claude` executables
on PATH (no model, no data): sealed session map and splits, curator-blind
copies, the hardened mcp-review argv, a holdout report that leaks nothing and
is ledgered, and A/B adjudication that disputes only arm-picked units.

    uv run --script scripts/test_bench_curation.py
"""

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

SORREL_STUB = r'''#!{py}
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as f:
    deny = os.environ.get("SORREL_DEV_DENY")
    f.write(json.dumps({{"tool": "sorrel", "argv": a, "env": {{k: os.environ.get(k) for k in ("SORREL_RUBRIC", "SORREL_REVIEW", "SORREL_DEV_DB", "SORREL_DEV_DENY")}},
                         "deny": Path(deny).read_text() if deny else None, "ledger": Path(a[a.index("--json") + 1]).read_text() if a[:2] == ["dev", "ledger"] else None}}) + "\n")
if a[0] == "blind":
    s, i, t = Path(a[1]), Path(a[a.index("--input") + 1]), Path(a[a.index("--truth") + 1])
    i.mkdir(parents=True); t.mkdir(parents=True)
    (i / "spike_clusters.npy").write_text("x"); (i / "params.py").write_text("p")
    (t / "provenance.json").write_text(json.dumps({{"pure_manual": not (s / "IMPURE").exists(), "anchored": False, "curated": not (s / "UNCURATED").exists()}}))
elif a[0] == "agent-guide":
    print("REVIEW GUIDE")
elif a[0] == "auto":
    d = Path(a[1]); (d / "sorrel_auto").mkdir(); (d / "sorrel_auto" / "evidence.json").write_text('{{"prompt_rev": "rev1"}}')
    assert not (d / "sorrel.journal").exists(), "journal leaked into a run copy"
elif a[0] == "call" and a[2] == "brief":
    Path(a[1], "sorrel.journal").write_text("j")  # what a real open leaves behind
    print("# sorrel brief v1\ncards:\nid\tlab\tflags\n3\tgood\t=\n9\tmua\t+A")
elif a[0] == "diff":
    gone = set()
    if "--disputed" in a:
        gone = {{int(x) for x in Path(a[a.index("--disputed") + 1]).read_text().splitlines()[1:]}}
    ms = [m for m in ({{"roots": [3, 4], "category": "missed_merge", "truth": None, "test": None}},
                      {{"roots": [9], "category": "label", "truth": "good", "test": "mua"}}) if not gone & set(m["roots"])]
    print(json.dumps({{"unit": {{"f1": 0.5 + (0.1 if "cand" in a[2] else 0)}}, "misses": ms}}))
'''

CLAUDE_STUB = r'''#!{py}
import json, os, sys
from pathlib import Path
a, prompt = sys.argv[1:], sys.stdin.read()
cfg = json.loads(Path(a[a.index("--mcp-config") + 1]).read_text())
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps({{"tool": "claude", "argv": a, "stdin": prompt, "cfg": cfg, "cwd_files": os.listdir("."), "cwd": os.getcwd(),
                         "env": os.environ.get("SORREL_REVIEW"), "rubric": os.environ.get("SORREL_RUBRIC")}}) + "\n")
print(json.dumps({{"is_error": False, "result": "A" if prompt.startswith("Two curators") else "done", "total_cost_usd": 0.01,
                   "num_turns": 3, "usage": {{"input_tokens": 5}}}}))
'''


def calls(log: Path, tool: str, sub: str | None = None) -> list[dict]:
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    return [r for r in rows if r["tool"] == tool and (sub is None or r["argv"][0] == sub)]


def main() -> None:
    tmp_root = HERE.parent / "target" / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as t:
        t = Path(t)
        stubs = t / "bin"
        stubs.mkdir()
        for name, body in (("sorrel", SORREL_STUB), ("claude", CLAUDE_STUB)):
            (stubs / name).write_text(body.format(py=sys.executable))
            (stubs / name).chmod(0o755)
        log = t / "log.jsonl"
        os.environ |= {"PATH": f"{stubs}:{os.environ['PATH']}", "SORREL_BIN": "sorrel", "STUB_LOG": str(log), "BENCH_SALT": "salt"}
        os.environ.pop("SORREL_DEV_DB", None)
        sys.path.insert(0, str(HERE))
        import bench_curation as bc

        real_claude = bc.claude

        bench = t / "bench"
        go = lambda *a: bc.main(["--bench", str(bench), *map(str, a)])

        # add: opaque ids, sealed map, splits, impure sessions forced to dev
        sessions = []
        for i in range(30):
            s = t / f"real-session-{i}"
            s.mkdir()
            if i % 3 == 0:
                (s / "IMPURE").write_text("")
            sessions.append(s)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            go("add", *sessions)
        m = json.loads((bench / ".sealed" / "map.json").read_text())["sessions"]
        hold = [sid for sid, v in m.items() if v["split"] == "holdout"]
        assert len(m) == 30 and not any("real-session" in p.name for p in (bench / "sessions").iterdir())
        assert {v["split"] for v in m.values()} == {"dev", "holdout"} and hold
        # nothing of a holdout session is visible outside .sealed: not its id in a path, not on stdout
        assert out.getvalue().startswith(f"added 30 session(s), {30 - len(hold)} to dev"), out.getvalue()
        assert not any(sid in out.getvalue() for sid in hold)
        assert not any(sid in str(p.relative_to(bench)) for sid in hold for p in bench.rglob("*") if p.relative_to(bench).parts[0] != ".sealed")
        assert all((bench / ".sealed" / "sessions" / sid / "input").is_dir() for sid in hold)
        # an uncurated phy sort is refused, and leaves nothing behind
        bad = t / "uncurated"
        bad.mkdir()
        (bad / "UNCURATED").write_text("")
        try:
            go("add", bad)
            raise AssertionError("uncurated session must be refused")
        except SystemExit as e:
            assert "uncurated" in str(e)
        assert len(json.loads((bench / ".sealed" / "map.json").read_text())["sessions"]) == 30
        assert all(v["split"] == "dev" for v in m.values() if not v["pure_manual"]) and sum(not v["pure_manual"] for v in m.values()) == 10
        before = len(m)
        go("add", sessions[0])  # idempotent
        assert len(json.loads((bench / ".sealed" / "map.json").read_text())["sessions"]) == before

        # run: both arms, base + cand variants, dev and holdout
        for v in ("base", "cand"):
            (t / v).mkdir()
            (t / v / "review.md").write_text(v)
        os.environ["SORREL_RUBRIC"] = "inherited"  # the variants have no rubric.md: it must not reach an arm
        for split, rep in (("dev", 2), ("holdout", 1)):
            for arm in bc.ARMS:
                for v in ("base", "cand"):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        go("run", "--arm", arm, "--variant", t / v, "--split", split, "--repeats", rep, "--jobs", 4, "--model", "m1", "--max-budget-usd", 2.5)
                    if split == "holdout":
                        assert out.getvalue().strip() == f"{len(hold)} holdout run(s): {len(hold)} ok", out.getvalue()
        del os.environ["SORREL_RUBRIC"]
        assert all(c["env"]["SORREL_RUBRIC"] is None for c in calls(log, "sorrel", "auto")) and all(c["rubric"] is None for c in calls(log, "claude"))
        n_hold = len(hold)
        assert len(list((bench / ".sealed" / "runs").iterdir())) == 4 * n_hold
        assert all(m[json.loads(p.read_text())["session"]]["split"] == "dev" for p in (bench / "runs").glob("*/run.json"))
        mcp = calls(log, "claude")[0]
        a = mcp["argv"]
        assert a[:4] == ["-p", "--output-format", "json", "--strict-mcp-config"], a
        for flag in ("--setting-sources", "--disable-slash-commands", "--no-session-persistence"):
            assert flag in a, flag
        assert a[a.index("--tools") + 1] == "" and a[a.index("--allowedTools") + 1] == "mcp__sorrel__*" and a[a.index("--model") + 1] == "m1"
        assert a[a.index("--max-budget-usd") + 1] == "2.5", a
        assert list(mcp["cfg"]["mcpServers"]) == ["sorrel"] and mcp["cfg"]["mcpServers"]["sorrel"]["args"][0] == "mcp"
        assert mcp["stdin"].startswith("REVIEW GUIDE") and mcp["cwd_files"] == [] and "REVIEW GUIDE" not in " ".join(a)
        assert mcp["env"] and mcp["env"].endswith("review.md")
        # a fresh cwd outside BENCH (no ancestor CLAUDE.md), gone afterwards
        assert not Path(mcp["cwd"]).resolve().is_relative_to(bench) and not Path(mcp["cwd"]).exists(), mcp["cwd"]
        assert any(c["env"]["SORREL_REVIEW"] for c in calls(log, "sorrel", "auto"))
        assert all(c["argv"][-2:] == ["--llm-model", "m1"] or "--save" in c["argv"] for c in calls(log, "sorrel", "auto"))
        rec = json.loads(next((bench / "runs").glob("mcp-review-base-*/run.json")).read_text())
        assert rec["status"] == "ok" and rec["cost_usd"] == 0.01 and rec["usage"] == {"input_tokens": 5} and rec["model"] == "m1" and rec["split"] == "dev"
        assert json.loads(next((bench / "runs").glob("auto-llm-*/run.json")).read_text())["prompt_rev"] == "rev1"

        # score; dev report attaches the brief card; the sealed input never gets a journal
        go("score")
        assert next((bench / "runs").glob("*/score.json")).exists()
        assert len(list((bench / ".sealed" / "runs").glob("*/score.json"))) == len(list((bench / ".sealed" / "runs").glob("*/run.json")))
        go("report", "dev")
        rep = next((bench / "reports").glob("dev-*.md")).read_text()
        assert "### missed_merge" in rep and "9 mua +A" in rep and "cols: `id lab flags`" in rep
        assert not list(bench.glob("**/sessions/*/input/sorrel.journal"))

        # holdout report: aggregates only, ledgered
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            go("report", "holdout", "--baseline", "base", "--candidate", "cand")
        text = out.getvalue()
        assert not any(sid in text for sid in m) and "root" not in text and "session " not in text.replace("paired holdout sessions", ""), text
        assert "unit.f1" in text and "+0.100" in text and "misses.label" in text, text
        assert text.strip().endswith("holdout queries so far: 1")
        assert "n<3" not in text  # many holdout sessions: a CI is reported
        with contextlib.redirect_stdout(out):
            go("report", "holdout", "--baseline", "base", "--candidate", "cand")
        assert len((bench / ".sealed" / "holdout-ledger.jsonl").read_text().splitlines()) == 2

        # adjudicate: A/B randomised, key sealed, unmask disputes only arm-picked units
        go("adjudicate", "--split", "dev")
        cases = bench / "adjudication" / "dev" / "cases.tsv"
        lines = [l.split("\t") for l in cases.read_text().splitlines()]
        assert lines[0][:6] == ["case", "session", "root_clusters", "input_dir", "verdict_A", "verdict_B"] and len(lines) == 1 + 2 * sum(v["split"] == "dev" for v in m.values())
        key = json.loads((bench / ".sealed" / "adjudication-dev.json").read_text())
        assert {k["A"] for k in key.values()} == {"human", "arm"}
        assert "missed_merge" not in cases.read_text() and "root 3,4: merged" in cases.read_text()
        try:
            go("adjudicate", "--split", "dev")
            raise AssertionError("regenerating over a filled sheet must need --force")
        except SystemExit:
            pass
        try:
            go("adjudicate", "--split", "holdout", "--judge", "claude")
            raise AssertionError("LLM judge on holdout must be refused")
        except SystemExit as e:
            assert "human" in str(e)
        arm_case, human_case = (next(c for c, k in key.items() if k["session"] == lines[1][1] and k["ids"] == ids) for ids in ([3, 4], [9]))
        pick = {arm_case: next(l for l, w in key[arm_case].items() if w == "arm" and l in "AB"),
                human_case: next(l for l, w in key[human_case].items() if w == "human" and l in "AB")}
        sid = key[arm_case]["session"]
        rows = [l[:6] + [pick.get(l[0], "")] for l in lines[1:]]
        cases.write_text("\t".join(lines[0][:6] + ["choice"]) + "\n" + "".join("\t".join(r) + "\n" for r in rows))
        go("adjudicate", "--split", "dev", "--unmask")
        disputed = sorted(bench.glob("disputed/*.tsv"))
        assert [p.stem for p in disputed] == [sid], disputed
        assert disputed[0].read_text() == "cluster_id\n3\n4\n", disputed[0].read_text()
        log.write_text("")
        go("score", "--split", "dev")
        assert any("--disputed" in c["argv"] for c in calls(log, "sorrel", "diff"))
        # --judge fills the sheet on dev
        go("adjudicate", "--split", "dev", "--judge", "claude", "--force")
        assert all(l.split("\t")[6] == "A" for l in cases.read_text().splitlines()[1:])

        # a strict judge: only a final line that is exactly A, B or both-wrong counts
        bc.claude = lambda *x, **k: (None, {"result": "I pick A\nB is wrong"})
        assert bc.judge(next(iter(m)), [3], "x", "y", None) == ""
        bc.claude = lambda *x, **k: (None, {"result": "reasoning\nboth-wrong\n"})
        assert bc.judge(next(iter(m)), [3], "x", "y", None) == "both-wrong"

        # holdout adjudication: sheet and disputed units live under .sealed, nothing prints an id
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            go("adjudicate", "--split", "holdout")
        hcases = bench / ".sealed" / "adjudication" / "holdout" / "cases.tsv"
        assert hcases.exists() and not (bench / "adjudication" / "holdout").exists()
        hkey = json.loads((bench / ".sealed" / "adjudication-holdout.json").read_text())
        hrows = [l.split("\t") for l in hcases.read_text().splitlines()]
        pick = next(l for l, w in hkey[hrows[1][0]].items() if w == "arm" and l in "AB")
        hcases.write_text("\n".join("\t".join(r[:6] + [c]) for r, c in zip(hrows, ["choice"] + [pick] + [""] * (len(hrows) - 2))) + "\n")
        with contextlib.redirect_stdout(out):
            go("adjudicate", "--split", "holdout", "--unmask")
        hsid = hkey[hrows[1][0]]["session"]
        assert (bench / ".sealed" / "disputed" / f"{hsid}.tsv").exists() and not (bench / "disputed" / f"{hsid}.tsv").exists()
        log.write_text("")
        go("score", "--split", "holdout")
        assert any("--disputed" in c["argv"] for c in calls(log, "sorrel", "diff"))
        assert not any(sid in out.getvalue() for sid in hold)

        # add --dev: the same sessions that split into a holdout above all stay in dev
        bench3 = t / "bench3"
        with contextlib.redirect_stdout(io.StringIO()):
            bc.main(["--bench", str(bench3), "add", "--dev", *map(str, sessions)])
        m3 = json.loads((bench3 / ".sealed" / "map.json").read_text())["sessions"]
        assert len(m3) == 30 and {v["split"] for v in m3.values()} == {"dev"} and not (bench3 / ".sealed" / "sessions").exists()

        # fewer than 3 paired sessions: no CI
        bench2 = t / "bench2"
        go2 = lambda *a: bc.main(["--bench", str(bench2), *map(str, a)])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            go2("add", *[m[sid]["path"] for sid in hold[:2]])
            for v in ("base", "cand"):
                go2("run", "--arm", "auto-llm", "--variant", t / v, "--split", "holdout", "--repeats", 1)
            go2("score")
            go2("report", "holdout", "--baseline", "base", "--candidate", "cand")
        assert "n<3 sessions: no CI" in out.getvalue() and "95% CI" not in out.getvalue(), out.getvalue()
        bc.claude = real_claude
        # results DB: the phases above ran without one and made no `dev` call
        assert not [c for c in calls(log, "sorrel") if c["argv"][0] == "dev" or "--record" in c["argv"]]
        benchdb, db = t / "benchdb", t / "r.sqlite"
        godb = lambda *a: bc.main(["--bench", str(benchdb), "--db", str(db), *map(str, a)])
        log.write_text("")
        with contextlib.redirect_stdout(io.StringIO()):
            godb("add", *sessions)
            for v in ("base", "cand"):
                for split in ("dev", "holdout"):
                    godb("run", "--arm", "mcp-review", "--variant", t / v, "--split", split, "--repeats", 1)
            godb("score")
            godb("report", "holdout", "--baseline", "base", "--candidate", "cand")
        m4 = json.loads((benchdb / ".sealed" / "map.json").read_text())["sessions"]
        hold4 = [sid for sid, v in m4.items() if v["split"] == "holdout"]
        dev4 = [sid for sid, v in m4.items() if v["split"] == "dev"]
        assert hold4 and dev4
        db_calls = [c for c in calls(log, "sorrel") if c["argv"][0] == "dev" or "--record" in c["argv"]]
        blob = json.dumps([{k: v for k, v in c.items() if k != "deny"} for c in db_calls])  # the deny file lists the sealed ids by design
        assert not any(sid in blob for sid in hold4) and ".sealed/sessions" not in blob and ".sealed/runs" not in blob, "a holdout id reached a DB call"
        assert all(c["env"]["SORREL_DEV_DB"] == str(db.resolve()) for c in db_calls)
        deny = db_calls[-1]["deny"].splitlines()
        assert set(hold4) <= set(deny) and "real-session-0" in deny and not set(dev4) & set(deny), deny
        sess_calls = [c["argv"] for c in db_calls if c["argv"][1] == "record-session"]
        dev_s, hold_s = [a for a in sess_calls if "dev" == a[a.index("--split") + 1]], [a for a in sess_calls if "holdout" == a[a.index("--split") + 1]]
        assert {a[a.index("--sid") + 1] for a in dev_s} == set(dev4) and all("--truth" in a and "--input" in a for a in dev_s)
        assert len(hold_s) == len(hold4) and all("--input" not in a and "--truth" not in a and a[a.index("--sid") + 1].startswith("h") for a in hold_s)
        runs4 = {json.loads(p.read_text())["id"] for p in (benchdb / "runs").glob("*/run.json")}
        rr = [c["argv"] for c in db_calls if c["argv"][1] == "record-run"]
        assert len(rr) == len(runs4) == 2 * len(dev4) and {Path(a[a.index("--run-dir") + 1]).name for a in rr} == runs4 and all("--variant-dir" in a for a in rr)
        rec = [c["argv"] for c in db_calls if "--record" in c["argv"]]
        assert {a[a.index("--record") + 1] for a in rec} == runs4 and all(a[0] == "diff" for a in rec)
        ledger = [c for c in db_calls if c["argv"][1] == "ledger"]
        rows = json.loads(ledger[0]["ledger"])
        assert len(ledger) == 1 and rows and {"query", "baseline", "candidate", "arm", "metric", "delta"} <= set(rows[0]) and len({r["query"] for r in rows}) == 1 and not any(sid in ledger[0]["ledger"] for sid in hold4)
        # a run without a miss category counts 0 of it: one session set for the means and the delta
        fake = [({"arm": "a", "variant": v, "session": sid}, {"unit": {"f1": 0.5}, "misses": [{"category": "label", "roots": [i]} for i in range(n)]})
                for v, sid, n in (("b", "s1", 1), ("b", "s1", 0), ("b", "s2", 0), ("b", "s3", 1), ("c", "s1", 0), ("c", "s2", 2), ("c", "s3", 1))]
        got: list = []
        bc.scored, bc.DB, bc.B, bc.dev = lambda split: iter(fake), Path("db"), t / "bench4", lambda *x: got.append(json.loads(Path(x[2]).read_text()))
        with contextlib.redirect_stdout(io.StringIO()):
            bc.report_holdout(argparse.Namespace(baseline="b", candidate="c"))
        row = next(r for r in got[0] if r["metric"] == "misses.label")
        assert (row["n_sessions"], row["base"], row["cand"], row["delta"]) == (3, 0.5, 1.0, 0.5), row
        # runs of one variant scored under another revision are not averaged into its holdout row
        mixed = [({**r, "sorrel_rev": "x" if i == 0 else "y"}, sc) for i, (r, sc) in enumerate(fake)]
        bc.scored = lambda split: iter(mixed)
        try:
            bc.report_holdout(argparse.Namespace(baseline="b", candidate="c"))
        except SystemExit as e:
            assert "revisions or scoring options" in str(e), e
        else:
            raise AssertionError("mixed revisions were averaged")
        print("ok: add/splits, mcp-review argv, holdout sealing + ledger, results-DB calls, adjudication")


if __name__ == "__main__":
    main()

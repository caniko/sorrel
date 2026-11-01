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
    f.write(json.dumps({{"tool": "sorrel", "argv": a, "env": {{k: os.environ.get(k) for k in ("SORREL_RUBRIC", "SORREL_REVIEW")}}}}) + "\n")
if a[0] == "blind":
    s, i, t = Path(a[1]), Path(a[a.index("--input") + 1]), Path(a[a.index("--truth") + 1])
    i.mkdir(parents=True); t.mkdir(parents=True)
    (i / "spike_clusters.npy").write_text("x"); (i / "params.py").write_text("p")
    (t / "provenance.json").write_text(json.dumps({{"pure_manual": not (s / "IMPURE").exists(), "anchored": False}}))
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
    f.write(json.dumps({{"tool": "claude", "argv": a, "stdin": prompt, "cfg": cfg, "cwd_files": os.listdir("."),
                         "env": os.environ.get("SORREL_REVIEW")}}) + "\n")
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
        sys.path.insert(0, str(HERE))
        import bench_curation as bc

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
        go("add", *sessions)
        m = json.loads((bench / ".sealed" / "map.json").read_text())["sessions"]
        assert len(m) == 30 and not any("real-session" in p.name for p in (bench / "sessions").iterdir())
        assert {v["split"] for v in m.values()} == {"dev", "holdout"}
        assert all(v["split"] == "dev" for v in m.values() if not v["pure_manual"]) and sum(not v["pure_manual"] for v in m.values()) == 10
        before = len(m)
        go("add", sessions[0])  # idempotent
        assert len(json.loads((bench / ".sealed" / "map.json").read_text())["sessions"]) == before

        # run: both arms, base + cand variants, dev and holdout
        for v in ("base", "cand"):
            (t / v).mkdir()
            (t / v / "review.md").write_text(v)
        for split, rep in (("dev", 2), ("holdout", 1)):
            for arm in bc.ARMS:
                for v in ("base", "cand"):
                    go("run", "--arm", arm, "--variant", t / v, "--split", split, "--repeats", rep, "--jobs", 4, "--model", "m1")
        n_hold = sum(v["split"] == "holdout" for v in m.values())
        assert len(list((bench / ".sealed" / "runs").iterdir())) == 4 * n_hold
        assert all(m[json.loads(p.read_text())["session"]]["split"] == "dev" for p in (bench / "runs").glob("*/run.json"))
        mcp = calls(log, "claude")[0]
        a = mcp["argv"]
        assert a[:4] == ["-p", "--output-format", "json", "--strict-mcp-config"], a
        for flag in ("--setting-sources", "--disable-slash-commands", "--no-session-persistence"):
            assert flag in a, flag
        assert a[a.index("--tools") + 1] == "" and a[a.index("--allowedTools") + 1] == "mcp__sorrel__*" and a[a.index("--model") + 1] == "m1"
        assert list(mcp["cfg"]["mcpServers"]) == ["sorrel"] and mcp["cfg"]["mcpServers"]["sorrel"]["args"][0] == "mcp"
        assert mcp["stdin"].startswith("REVIEW GUIDE") and mcp["cwd_files"] == [] and "REVIEW GUIDE" not in " ".join(a)
        assert mcp["env"] and mcp["env"].endswith("review.md")
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
        assert not list((bench / "sessions").glob("*/input/sorrel.journal"))

        # holdout report: aggregates only, ledgered
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            go("report", "holdout", "--baseline", "base", "--candidate", "cand")
        text = out.getvalue()
        assert not any(sid in text for sid in m) and "root" not in text and "session " not in text.replace("paired holdout sessions", ""), text
        assert "unit.f1" in text and "+0.100" in text and "misses.label" in text, text
        assert text.strip().endswith("holdout queries so far: 1")
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
        print("ok: add/splits, mcp-review argv, holdout blind + ledger, adjudication")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "spikeinterface[full]>=0.105",
#   "skops",
#   "huggingface_hub",
#   "anthropic",
# ]
# ///
"""Self-check for sorrel_auto.py on synthetic ground truth (no network
unless --unitrefine): one unit split in two must come back merged, an exact
duplicate must come back `redundant`, and the output must be a valid
SpikeInterface curation document carrying a quality label and a `src:` tag
for every unit.

    uv run --script scripts/test_sorrel_auto.py [--unitrefine] [--keep DIR] [--brief-only]
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import spikeinterface.core as sc
import spikeinterface.curation as scur
from spikeinterface.exporters import export_to_phy


def make_phy(folder: Path) -> tuple[int, int, int, int]:
    rec, gt = sc.generate_ground_truth_recording(durations=[180.0], num_units=8, seed=7)
    trains = {int(u): gt.get_unit_spike_train(u) for u in gt.unit_ids}
    rng = np.random.default_rng(0)
    # Oversplit unit 0 into two halves; duplicate unit 1 exactly.
    a = trains.pop(0)
    half = rng.random(a.size) < 0.5
    trains[100], trains[101] = a[half], a[~half]
    trains[102] = trains[1].copy()
    sorting = sc.NumpySorting.from_unit_dict(trains, rec.sampling_frequency)
    analyzer = sc.create_sorting_analyzer(sorting, rec, sparse=True)
    analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels", "spike_amplitudes"])
    export_to_phy(analyzer, folder, compute_pc_features=False, copy_binary=True, remove_if_exists=True)
    # spike_clusters.npy holds unit indices (sorting order); sorrel and the
    # sidecar both use those, not the `si_unit_id` column.
    ids = [int(u) for u in sorting.unit_ids]
    return ids.index(100), ids.index(101), ids.index(1), ids.index(102)


BRIEF = (
    "# sorrel brief v1\nrec: 180.0 s\nlegend: x\ncards: one row per unit\ndrill: brief{clusters:[..]}\n"
    "ledger\n"
    "cards:\nid\tlab\tflags\tfr\n3\tgood\t=\t4.1\n7\tmua\t+A\t0.2\nomitted: cards 9; raise budget\n"
)


def test_brief_cases() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crates" / "sorrel" / "src"))
    import sorrel_auto as sa

    head, card_head, rows = sa.split_brief(BRIEF)
    assert head.endswith("drill: brief{clusters:[..]}\nledger") and "\ncards:\n" not in head
    assert card_head == "id\tlab\tflags\tfr" and sorted(rows) == [3, 7], rows
    rec = {"bombcell": "mua", "unitrefine": "good", "unitrefine_probability": 0.55, "members": [7]}
    msg = sa.unit_message(card_head, rows, rec, {})
    assert msg == (
        "Decide this unit.\nid\tlab\tflags\tfr\n7\tmua\t+A\t0.2\n"
        'classifiers: {"bombcell":"mua","unitrefine":"good","unitrefine_probability":0.55}'
    ), msg
    merged = sa.unit_message(card_head, rows, {**rec, "members": [3, 7]}, {3: "eval: z=1"})
    assert "merge of 3,7" in merged and "3\tgood" in merged and "7\tmua" in merged and "\neval: z=1\n" in merged
    assert sa.eval_line("merge {3,4}: rpc 0,0→.05\n") == "merge {3,4}: rpc 0,0→.05"
    assert sa.merge_message("eval: m", {"slay", "similarity_correlograms"}).endswith("proposed_by: similarity_correlograms,slay")
    # a unit missing a card (or one member of a merge, or its eval) is left for review, never judged partially
    assert sa.unit_message(card_head, rows, {**rec, "members": [99]}, {}) is None
    assert sa.unit_message(card_head, rows, {**rec, "members": [3, 99]}, {3: "eval: z=1"}) is None
    assert sa.unit_message(card_head, rows, {**rec, "members": [3, 7]}, {3: None}) is None
    assert sa.eval_line("two\nlines") is None and sa.eval_line("") is None
    _, _, ok_rows = sa.split_brief(BRIEF.replace("7\tmua", "x\tbad\n7\tmua"))
    assert sorted(ok_rows) == [3, 7], ok_rows


FAKE = """#!/usr/bin/env python3
import json, os, sys
name, a = os.path.basename(sys.argv[0]), sys.argv[1:]
stdin = "" if name == "opencode" else sys.stdin.read()
rec = {"argv": a, "stdin": stdin, "cwd": os.getcwd(), "listing": os.listdir("."), "perm": os.environ.get("OPENCODE_PERMISSION")}
json.dump(rec, open(os.environ["FAKE_RECORD"], "w"))
fail, V = os.environ["FAKE_MODE"] == "fail", {"label": "mua", "confidence": 1.7, "reasons": ["a", "b", "c", "d"]}
if name == "claude":
    print(json.dumps({"type": "result", "is_error": fail, "result": "Not logged in" if fail else "t", "structured_output": None if fail else V, "modelUsage": {"claude-opus-5-5": {}}}))
elif name == "codex":
    if fail:
        print(json.dumps({"type": "turn.failed", "error": {"message": "codex not logged in"}}))
        sys.exit(1)
    open(a[a.index("-o") + 1], "w").write(json.dumps(V))
    print(json.dumps({"type": "turn.completed"}))
elif fail:
    print(json.dumps({"type": "error", "error": {"type": "x", "message": "opencode no auth"}}))
else:
    print(json.dumps({"type": "text", "part": {"text": "Sure:\\n```json\\n" + json.dumps(V) + "\\n```"}}))
"""


def test_llm_backends() -> None:
    import contextlib
    import io
    import os
    import shutil
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crates" / "sorrel" / "src"))
    import sorrel_auto as sa

    pv = sa.parse_verdict
    ok = {"label": "good", "confidence": 0.9, "reasons": ["r"]}
    assert pv(json.dumps(ok), "unit") == ok
    assert pv("```json\n" + json.dumps(ok) + "\n```", "unit") == ok
    assert pv("Verdict {not json} then " + json.dumps(ok) + " done", "unit") == ok
    assert pv('{"label":"mua","confidence":-3,"reasons":["1","2","3","4"]}', "unit") == {"label": "mua", "confidence": 0.0, "reasons": ["1", "2", "3"]}
    assert pv('{"merge":true,"confidence":2,"reasons":[]}', "merge") == {"merge": True, "confidence": 1.0, "reasons": []}
    for bad, kind in [
        ('{"label":"sua","confidence":1,"reasons":[]}', "unit"),
        ('{"label":"good","confidence":1}', "unit"),
        ('{"label":"good","confidence":"hi","reasons":[]}', "unit"),
        ('{"label":"good","confidence":NaN,"reasons":[]}', "unit"),
        ('{"merge":"yes","confidence":1,"reasons":[]}', "merge"),
        ('{"label":"good","confidence":1,"reasons":[]}', "merge"),
        ("no json here", "unit"),
        (None, "unit"),
    ]:
        assert pv(bad, kind) is None, bad

    with tempfile.TemporaryDirectory() as d:
        bindir, rec = Path(d) / "bin", Path(d) / "rec.json"
        bindir.mkdir()
        for name in ("claude", "codex", "opencode"):
            (bindir / name).write_text(FAKE)
            (bindir / name).chmod(0o755)
        saved = dict(os.environ)
        os.environ.update(PATH=f"{bindir}{os.pathsep}{saved['PATH']}", FAKE_RECORD=str(rec))
        try:
            def ask(backend, model, mode):
                os.environ["FAKE_MODE"] = mode
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    v = sa.ask_cli(backend, model, "SYS", "unit", "MSG")
                r = json.loads(rec.read_text())
                assert Path(r["cwd"]).resolve() != Path.cwd().resolve() and r["listing"] == [], r
                return v, r, err.getvalue()

            for backend, model, want in [
                ("claude", "m1", "claude-cli:claude-opus-5-5"),
                ("claude", None, "claude-cli:claude-opus-5-5"),
                ("codex", "m1", "codex:m1"),
                ("opencode", "m1", "opencode:m1"),
            ]:
                v, r, _ = ask(backend, model, "ok")
                assert v == {"label": "mua", "confidence": 1.0, "reasons": ["a", "b", "c"], "model": want}, (backend, v)
                a = r["argv"]
                flag = "--model" if backend == "claude" else "-m"
                assert (a[a.index(flag) + 1] == model) if model else flag not in a
                if backend == "claude":
                    assert a[:3] == ["-p", "--output-format", "json"] and json.loads(a[a.index("--json-schema") + 1]) == sa.UNIT_SCHEMA
                    assert a[a.index("--tools") + 1] == "" and a[a.index("--system-prompt") + 1] == "SYS"
                    assert a[a.index("--setting-sources") + 1] == "project" and a[a.index("--mcp-config") + 1].endswith("mcp.json")
                    for flag in ("--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands"):
                        assert flag in a, flag
                    assert r["stdin"] == "MSG"
                elif backend == "codex":
                    assert a[:2] == ["exec", "--ephemeral"] and a[-1] == "-" and "--output-schema" in a and "-o" in a and a[a.index("-s") + 1] == "read-only"
                    assert r["stdin"] == "SYS\n\nMSG"
                else:
                    assert a[:3] == ["run", "--format", "json"]
                    assert a[-1].startswith("SYS\n\nMSG") and "JSON object matching this schema" in a[-1]
                    assert r["perm"] == '{"*":"deny"}'
            exe = sa.cli_argv("claude", None, "S", "unit", "M", Path("."))[0][0]
            assert exe == shutil.which("claude") and Path(exe).is_absolute(), exe
            # one raising call becomes None; a job without a message is never asked
            real = sa.ask_cli

            def flaky(backend, model, system, kind, message):
                if message == "boom":
                    raise RuntimeError("x")
                return {"label": "good", "confidence": 1.0, "reasons": [], "model": "m"}

            sa.ask_cli = flaky
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    got = sa.run_llm("claude", None, "SYS", [("unit", "a"), ("unit", "boom"), ("unit", None)])
            finally:
                sa.ask_cli = real
            assert [g and g["label"] for g in got] == ["good", None, None], got
            for backend, msg in [("claude", "Not logged in"), ("codex", "codex not logged in"), ("opencode", "opencode no auth")]:
                v, _, err = ask(backend, None, "fail")
                assert v is None and msg in err, (backend, v, err)
        finally:
            os.environ.clear()
            os.environ.update(saved)


def main() -> None:
    if "--brief-only" in sys.argv:
        test_brief_cases()
        test_llm_backends()
        print("ok: brief cases, llm backends")
        return
    test_brief_cases()
    test_llm_backends()
    keep = sys.argv[sys.argv.index("--keep") + 1] if "--keep" in sys.argv else None
    root = Path(keep or tempfile.mkdtemp(prefix="sorrel-auto-"))
    phy, out = root / "phy", root / "phy" / "sorrel_auto"
    split_a, split_b, orig, dup = make_phy(phy)

    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "crates" / "sorrel" / "src" / "sorrel_auto.py"), "--data-dir", str(phy), "--out", str(out), "--gain-to-uv", "1.0", "--jobs", "1"]
    if "--unitrefine" not in sys.argv:
        cmd.append("--no-unitrefine")
    subprocess.run(cmd, check=True)

    doc = json.loads((out / "curation.json").read_text())
    scur.validate_curation_dict(doc)
    by_unit = {e["unit_id"]: e["labels"] for e in doc["manual_labels"]}
    for u in doc["unit_ids"]:
        assert by_unit[u]["quality"][0] in ("good", "mua", "noise"), (u, by_unit[u])
        assert any(t.startswith("src:") for t in by_unit[u]["tags"]), (u, by_unit[u])
    merged = [set(m["unit_ids"]) for m in doc["merges"]]
    assert any({split_a, split_b} <= g for g in merged), f"oversplit not merged: {doc['merges']}"
    redundant = {u for u, l in by_unit.items() if "redundant" in l["tags"]}
    assert redundant & {orig, dup}, f"duplicate not flagged: {redundant}"
    assert (out / "evidence.json").exists()
    print(f"ok: {len(doc['merges'])} merge(s), redundant {sorted(redundant)}, output in {out}")


if __name__ == "__main__":
    main()

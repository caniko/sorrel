# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Double-blind curation bench: score `sorrel auto --llm` and the MCP review
agent against human curation (see "Double-blind bench" in
docs/src/usage/auto-curation.md).

    bench_curation.py --bench BENCH add SESSION...
    bench_curation.py --bench BENCH run --arm auto-llm|mcp-review --variant DIR --split dev|holdout [--repeats 3] [--model M] [--jobs 2]
    bench_curation.py --bench BENCH score [--split dev|holdout]
    bench_curation.py --bench BENCH report dev
    bench_curation.py --bench BENCH report holdout --baseline NAME --candidate NAME
    bench_curation.py --bench BENCH adjudicate --split S [--judge claude] | --unmask --split S

With `--db FILE` (or $SORREL_DEV_DB) and a `sorrel` built with `--features dev`, every dev result
is also recorded in the results DB (`sorrel dev ...`): `add` records sessions, `run` runs, `score`
the metrics and misses, `report holdout` the aggregate rows. A holdout session reaches the DB only
as an opaque alias row; no holdout run or id ever does. The JSON files stay as a cache.

BENCH is private data outside the repo. `$SORREL_BIN` names the sorrel binary
(default: `sorrel` on PATH); a variant DIR may hold `rubric.md` / `review.md`.
Everything about a holdout session (its copy, runs, scores, adjudication sheet,
disputed units) lives under BENCH/.sealed, and no command prints its ids.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import secrets
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

B = Path()  # the bench root, set by main()
DB = None  # the results DB (--db or $SORREL_DEV_DB), set by main(); None = JSON files only
SORREL = os.environ.get("SORREL_BIN", "sorrel")
HOLDOUT_PCT = 30
LINK_MIN = 64 << 20  # files at least this big are hard-linked into a copy, smaller ones copied
ARMS = ("auto-llm", "mcp-review")
TASK = "Task: review the dataset behind the sorrel MCP server, apply your curation decisions with its tools, then stop."
JUDGE = (
    "Two curators handled the same spike-sorting cluster(s). Using only the evidence cards, decide which verdict is better supported.\n"
    "Reply with exactly one token: A, B or both-wrong.\n"
)


def run(argv, env=None, cwd=None, timeout=600, stdin=None):
    env = {k: v for k, v in {**os.environ, **(env or {})}.items() if v is not None}  # a None value unsets an inherited variable
    return subprocess.run([str(a) for a in argv], env=env, cwd=cwd, input=stdin, capture_output=True, text=True, errors="replace", timeout=timeout)


def sorrel(*args, env=None, timeout=600) -> str:
    r = run([SORREL, *args], env, timeout=timeout)
    if r.returncode:
        raise SystemExit(f"sorrel {args[0]} failed (exit {r.returncode}): {r.stderr.strip()[-400:]}")
    return r.stdout


def dev_env() -> dict:
    """Environment of every results-DB call: the DB, and a deny file of what it must never hold
    (the sealed ids and every source basename), rewritten from the map each time."""
    sessions = load_map()["sessions"]
    deny = B / ".sealed" / "dev-deny.txt"
    deny.parent.mkdir(parents=True, exist_ok=True)
    deny.write_text("".join(f"{x}\n" for x in sorted({sid for sid, v in sessions.items() if v["split"] == "holdout"} | {Path(v["path"]).name for v in sessions.values()})))
    return {"SORREL_DEV_DB": str(DB), "SORREL_DEV_DENY": str(deny)}


def dev(*args) -> str:
    """`sorrel dev ...` on the results DB; a refusal (privacy guard) stops the bench."""
    return sorrel("dev", *args, env=dev_env(), timeout=1800)


def db_session(sid: str) -> None:
    """Record a session. A holdout session is an alias row (salted hash, unlinkable to its sealed id), nothing else."""
    if not DB:
        return
    m = load_map()
    if m["sessions"][sid]["split"] == "dev":
        d = home("dev") / "sessions" / sid
        dev("record-session", "--sid", sid, "--split", "dev", "--input", d / "input", "--truth", d / "truth")
    else:
        dev("record-session", "--sid", "h" + hashlib.sha256((m["salt"] + sid).encode()).hexdigest()[:8], "--split", "holdout")


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(p)


def load_map() -> dict:
    p = B / ".sealed" / "map.json"
    # BENCH_SALT only makes the dev/holdout split reproducible (tests).
    return json.loads(p.read_text()) if p.exists() else {"salt": os.environ.get("BENCH_SALT") or secrets.token_hex(16), "sessions": {}}


def home(split: str) -> Path:
    """Where a split's material lives: holdout under .sealed, never beside dev."""
    return B / ".sealed" if split == "holdout" else B


def sess(sid: str) -> Path:
    return home(load_map()["sessions"][sid]["split"]) / "sessions" / sid


def link_or_copy(src, dst) -> None:
    # ponytail: sorrel's own saves are atomic renames, so a link to a big array is safe
    # from it; a small file is copied because an in-place write would reach the sealed input.
    if os.stat(src).st_size >= LINK_MIN:
        try:
            return os.link(src, dst)
        except OSError:
            pass
    shutil.copy2(src, dst)


def copy_input(sid: str, dst: Path) -> None:
    """Copy of the sealed input (big files hard-linked). Only the files `add` saw are copied,
    so a journal a reviewer's GUI or a stray `sorrel call` left in `input` never
    reaches an arm (curator blind)."""
    src, keep = sess(sid) / "input", set(load_map()["sessions"][sid]["keep"])
    shutil.copytree(src, dst, symlinks=True, copy_function=link_or_copy, ignore=lambda d, ns: [n for n in ns if Path(d) == src and n not in keep])


@contextmanager
def scratch_input(sid: str):
    d = Path(tempfile.mkdtemp(prefix="bench-", dir=B / ".sealed"))
    try:
        copy_input(sid, d / "input")
        yield d / "input"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def cmd_add(a) -> None:
    m = load_map()
    known = {s["path"] for s in m["sessions"].values()}
    n_added = n_dev = n_forced = 0
    for s in a.session:
        path = str(Path(s).resolve())
        if path in known:
            print(f"skip (already added): {s}")
            continue
        sid = secrets.token_hex(4)
        d = B / ".sealed" / "incoming" / sid  # moved out once the split is known: a holdout id never leaves .sealed
        try:
            sorrel("blind", path, "--input", d / "input", "--truth", d / "truth", *(["--phy-curated"] if a.phy_curated else []))
            prov = json.loads((d / "truth" / "provenance.json").read_text())
            if prov.get("curated") is False:
                raise SystemExit(f"refusing {s}: it looks uncurated (spike_clusters equals spike_templates, labels are the sorter's); a bench truth needs a saved human curation")
        except (SystemExit, OSError, ValueError):
            shutil.rmtree(d, ignore_errors=True)
            raise
        pure = prov.get("pure_manual") is True and not prov.get("anchored")
        h = int(hashlib.sha256((m["salt"] + path).encode()).hexdigest()[:8], 16) % 100
        split = "holdout" if pure and not a.dev and h < HOLDOUT_PCT else "dev"
        final = home(split) / "sessions" / sid
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(d, final)
        m["sessions"][sid] = {"path": path, "split": split, "pure_manual": pure, "forced_dev": a.dev, "keep": sorted(p.name for p in (final / "input").iterdir())}
        write_json(B / ".sealed" / "map.json", m)  # per session: a later failure keeps the earlier ones
        db_session(sid)
        n_added, n_dev, n_forced = n_added + 1, n_dev + (split == "dev"), n_forced + (not pure)
    print(f"added {n_added} session(s), {n_dev} to dev ({n_forced} of them not purely manual); the rest are sealed")


def sorrel_rev() -> str | None:
    exe = shutil.which(SORREL)
    if os.environ.get("SORREL_REV") or not exe:
        return os.environ.get("SORREL_REV")
    r = run(["git", "-C", Path(exe).resolve().parent, "rev-parse", "--short=12", "HEAD"])  # ponytail: only a binary built inside its repo
    return r.stdout.strip() or None if r.returncode == 0 else None


def claude(prompt: str, root: Path, servers: dict, allow: str | None, model: str | None, env: dict, timeout: int, budget: float | None = None):
    """One hardened `claude -p` (the sidecar's cli_argv flags): fresh empty cwd, only
    the given MCP servers, no built-in tools, prompt on stdin. -> (proc, json)."""
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "mcp.json", {"mcpServers": servers})
    argv = [shutil.which("claude") or "claude", "-p", "--output-format", "json", "--strict-mcp-config", "--mcp-config", root / "mcp.json",
            "--setting-sources", "project", "--disable-slash-commands", "--no-session-persistence", "--tools", ""]
    argv += ["--allowedTools", allow] if allow else []
    argv += ["--model", model] if model else []
    argv += ["--max-budget-usd", str(budget)] if budget else []
    cwd = tempfile.mkdtemp(prefix="bench-claude-")  # outside BENCH, so no ancestor CLAUDE.md loads
    try:
        r = run(argv, env, cwd=cwd, timeout=timeout, stdin=prompt)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    try:
        return r, json.loads(r.stdout)
    except ValueError:
        return r, {}


def variant_env(v: Path) -> dict:
    """A variant file sets its variable; a missing one unsets it, so a stray inherited override can't leak in."""
    return {k: str((v / f).resolve()) if (v / f).exists() else None for k, f in (("SORREL_RUBRIC", "rubric.md"), ("SORREL_REVIEW", "review.md"))}


def snap(d: Path) -> dict:
    return {str(p.relative_to(d)): (s.st_ino, s.st_size, s.st_mtime_ns) for p in d.rglob("*") if p.is_file() and not p.is_symlink() for s in [p.stat()]}


def run_roots(split: str) -> Path:
    return B / "runs" if split == "dev" else B / ".sealed" / "runs"


def one_run(a, sid: str, k: int) -> dict:
    rid = f"{a.arm}-{a.variant.name}-{sid}-r{k}-{secrets.token_hex(2)}"
    root = run_roots(a.split) / rid
    data, env = root / "data", variant_env(a.variant)
    copy_input(sid, data)
    before, t0 = snap(data), time.monotonic()
    rec = {"id": rid, "session": sid, "split": a.split, "arm": a.arm, "variant": a.variant.name, "repeat": k, "model": a.model, "sorrel_rev": sorrel_rev()}
    try:
        if a.arm == "auto-llm":
            sorrel("auto", data, "--llm", *(["--llm-model", a.model] if a.model else []), "--save", env=env, timeout=a.timeout)
            ev = data / "sorrel_auto" / "evidence.json"
            rec["prompt_rev"] = json.loads(ev.read_text()).get("prompt_rev") if ev.exists() else None
        else:
            prompt = sorrel("agent-guide", "--prompt", "review", env=env) + "\n\n" + TASK + "\n"
            sv = {"command": shutil.which(SORREL) or SORREL, "args": ["mcp", str(data)], **({"env": {k: v for k, v in env.items() if v}} if any(env.values()) else {})}
            r, out = claude(prompt, root, {"sorrel": sv}, "mcp__sorrel__*", a.model, env, a.timeout, a.max_budget_usd)
            rec["prompt_rev"] = hashlib.sha256(prompt.encode()).hexdigest()[:12]
            rec |= {"usage": out.get("usage"), "cost_usd": out.get("total_cost_usd"), "num_turns": out.get("num_turns")}
            (root / "claude.json").write_text(r.stdout)
            if r.returncode or out.get("is_error"):
                raise RuntimeError(f"claude exit {r.returncode}: {out.get('result') or r.stderr.strip()[-300:]}")
            sorrel("call", data, "save", timeout=a.timeout)
        rec["status"] = "ok"
    except subprocess.TimeoutExpired:
        rec["status"] = "timeout"
    except (SystemExit, RuntimeError, OSError, ValueError) as e:  # ValueError: a bad evidence.json
        rec["status"], rec["error"] = "failed", str(e)
    # Hard links share inodes with the sealed input: an in-place write (same inode, new mtime) corrupts it.
    after = snap(data)
    if any(f in after and after[f][0] == v[0] and after[f] != v for f, v in before.items()):
        rec["status"] = "input-mutated"
        print(f"!! {rid if a.split == 'dev' else 'a holdout run'}: sorrel wrote a sealed input file in place; check the run's run.json", file=sys.stderr)
    rec["wall_s"] = round(time.monotonic() - t0, 1)
    write_json(root / "run.json", rec)
    return rec


def cmd_run(a) -> None:
    sids = [s for s, v in load_map()["sessions"].items() if v["split"] == a.split]
    if not sids:
        raise SystemExit(f"no {a.split} sessions")
    jobs = [(s, k) for s in sids for k in range(a.repeats)]
    if DB and a.split == "dev":
        for s in sids:
            db_session(s)  # idempotent; a run needs its session row
    with ThreadPoolExecutor(a.jobs) as ex:
        recs = list(ex.map(lambda j: one_run(a, *j), jobs))
    if DB and a.split == "dev":  # never a holdout run: its id holds the sealed session id
        for r in recs:
            dev("record-run", "--run-dir", run_roots("dev") / r["id"], "--variant-dir", a.variant)
    if a.split == "holdout":  # no run or session ids: they would tell which sessions are sealed
        print(f"{len(recs)} holdout run(s): " + ", ".join(f"{n} {st}" for st in sorted({r["status"] for r in recs}) for n in [sum(r["status"] == st for r in recs)]))
    else:
        for r in recs:
            print(f"{r['id']}  {r['status']}  {r['wall_s']}s")


def runs(split: str):
    """(run.json, run dir) for every run of a split."""
    for p in sorted(run_roots(split).glob("*/run.json")):
        yield json.loads(p.read_text()), p.parent


def cmd_score(a) -> None:
    for split in ("dev", "holdout") if a.split == "all" else (a.split,):
        for rec, root in runs(split):
            if rec["status"] != "ok":
                continue
            s = sess(rec["session"])
            disp = home(split) / "disputed" / f"{rec['session']}.tsv"
            record = bool(DB) and split == "dev"
            argv = ["diff", s / "truth", root / "data", "--root", s / "input", "--json", "--match", a.match] + (["--disputed", disp] if disp.exists() else [])
            r = run([SORREL, *argv, *(["--record", rec["id"], "--db", DB] if record else [])], env=dev_env() if record else None)
            try:
                write_json(root / "score.json", json.loads(r.stdout))
            except ValueError:
                print(f"{rec['id'] if split == 'dev' else 'a holdout run'}: diff failed: {r.stderr.strip()[-200:]}", file=sys.stderr)
            else:
                if record and r.returncode:
                    print(f"{rec['id']}: not recorded in the results DB: {r.stderr.strip()[-200:]}", file=sys.stderr)


def misses(score: dict) -> list:
    return score.get("misses", [])


def miss_ids(m: dict) -> tuple:
    v = m.get("roots", [])
    return tuple(int(x) for x in (v if isinstance(v, list) else [v]))


def miss_cat(m: dict) -> str:
    return str(next((m[k] for k in ("category", "kind", "type") if k in m), "?"))


# (truth, test) wording for the categories whose Miss has no truth/test labels; the
# category name itself is never shown, it would tell the reviewer who is who.
STRUCT = {"missed_merge": ("merged", "separate"), "false_merge": ("separate", "merged"), "missed_split": ("split", "whole"), "false_split": ("whole", "split")}


def verdict(m: dict, side: str) -> str:
    """Neutral text of the human ("truth") or arm ("test") side of a miss."""
    v = m.get(side) or STRUCT.get(miss_cat(m), ("?", "?"))[side == "test"]
    return f"root {','.join(map(str, miss_ids(m)))}: {v}"


def metrics(score: dict) -> dict:
    """Numeric leaves of the diff JSON by dotted name, plus misses.<category> counts."""
    out = {}

    def walk(d, p):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{p}{k}.")
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out[p + k] = v

    walk({k: v for k, v in score.items() if k not in ("misses", "n_spikes", "n_root_clusters", "match_jaccard", "min_spikes")}, "")
    for m in misses(score):
        out[f"misses.{miss_cat(m)}"] = out.get(f"misses.{miss_cat(m)}", 0) + 1
    return out


def scored(split: str):
    for rec, root in runs(split):
        if (root / "score.json").exists():
            yield rec, json.loads((root / "score.json").read_text())


def fmt(xs: list) -> str:
    return f"{statistics.fmean(xs):.3f} ± {statistics.stdev(xs) if len(xs) > 1 else 0:.3f}"  # ponytail: sd over runs mixes repeat and session variance


def brief_rows(sid: str, ids) -> tuple[str, dict]:
    """(card column line, {cluster: card row}) the arm saw, from a scratch copy so no journal lands in `input`."""
    try:
        with scratch_input(sid) as d:
            out = sorrel("call", d, "brief", json.dumps({"level": 2, "clusters": sorted(ids)}))
    except SystemExit as e:
        return f"(brief unavailable: {e})", {}
    head, rows = "", {}
    for line in out.splitlines():
        f = line.split("\t")
        if f[0] == "id":
            head = line
        elif f[0].isdigit() and int(f[0]) in ids:
            rows[int(f[0])] = line
    return head, rows


def report_dev(a) -> None:
    groups: dict = {}  # (arm, variant) -> [(rec, score)]
    for rec, sc in scored("dev"):
        groups.setdefault((rec["arm"], rec["variant"]), []).append((rec, sc))
    if not groups:
        raise SystemExit("no scored dev runs (run, then score)")
    failed = {}
    for rec, _ in runs("dev"):
        if rec["status"] != "ok":
            failed[(rec["arm"], rec["variant"])] = failed.get((rec["arm"], rec["variant"]), 0) + 1
    names = sorted(groups)
    ms = [dict(metrics(sc)) for k in names for _, sc in groups[k]]
    keys = sorted({k for m in ms for k in m})
    md = ["# Dev report", "", f"| metric | {' | '.join('/'.join(n) for n in names)} |", f"|---|{'---|' * len(names)}"]
    md.append("| runs (failed) | " + " | ".join(f"{len(groups[n])} ({failed.get(n, 0)})" for n in names) + " |")
    for k in keys:
        md.append(f"| {k} | " + " | ".join(fmt([metrics(sc)[k] for _, sc in groups[n] if k in metrics(sc)] or [float('nan')]) for n in names) + " |")
    for n in names:
        tally: dict = {}  # (cat, sid, ids) -> runs with that miss
        per_sid: dict = {}
        for rec, sc in groups[n]:
            per_sid[rec["session"]] = per_sid.get(rec["session"], 0) + 1
            for m in misses(sc):
                key = (miss_cat(m), rec["session"], miss_ids(m))
                tally[key] = tally.get(key, 0) + 1
        want: dict = {}
        for _, sid, ids in tally:
            want.setdefault(sid, set()).update(ids)
        cards = {sid: brief_rows(sid, ids) for sid, ids in want.items()}
        md += ["", f"## Misses: {'/'.join(n)}"]
        for cat in sorted({c for c, _, _ in tally}):
            md += ["", f"### {cat}"]
            for (c, sid, ids), cnt in sorted(tally.items()):
                if c != cat:
                    continue
                head, rows = cards[sid]
                card = " || ".join(rows[i].replace("\t", " ") for i in ids if i in rows) or "(no card)"
                md.append(f"- session {sid} root {','.join(map(str, ids))} ({cnt}/{per_sid[sid]} runs)\n  - cols: `{head.replace(chr(9), ' ')}`\n  - card: `{card}`")
    out = B / "reports" / f"dev-{time.strftime('%Y%m%d-%H%M%S')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print(out)


def per_session(split: str, variant: str, arm: str, cats=()) -> dict:
    """{session: {metric: mean over repeats}}; a miss category of `cats` a run lacks counts 0."""
    acc: dict = {}
    for rec, sc in scored(split):
        if rec["variant"] == variant and rec["arm"] == arm:
            for k, v in (dict.fromkeys(cats, 0) | metrics(sc)).items():
                acc.setdefault(rec["session"], {}).setdefault(k, []).append(v)
    return {s: {k: statistics.fmean(v) for k, v in d.items()} for s, d in acc.items()}


def report_holdout(a) -> None:
    arms = sorted({r["arm"] for r, _ in scored("holdout")})
    rng, lines, rows = random.Random(0), [], []
    for arm in arms:
        cats = {f"misses.{miss_cat(m)}" for r, sc in scored("holdout") if r["arm"] == arm and r["variant"] in (a.baseline, a.candidate) for m in misses(sc)}
        base, cand = per_session("holdout", a.baseline, arm, cats), per_session("holdout", a.candidate, arm, cats)
        common = sorted(set(base) & set(cand))
        if not common:
            continue
        lines.append(f"{arm}: {len(common)} paired holdout sessions")
        for k in sorted({k for s in common for k in base[s] if k in cand[s]}):
            ss = [s for s in common if k in base[s] and k in cand[s]]  # the one session set of the means and the deltas
            d = [cand[s][k] - base[s][k] for s in ss]
            boot = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(2000))
            ci = f"95% CI [{boot[50]:+.3f}, {boot[1949]:+.3f}]" if len(d) >= 3 else "n<3 sessions: no CI"
            mb, mc = statistics.fmean([base[s][k] for s in ss]), statistics.fmean([cand[s][k] for s in ss])
            lines.append(f"  {k:36s} base {mb:9.3f}  cand {mc:9.3f}  delta {statistics.fmean(d):+8.3f}  {ci}")
            rows.append({"baseline": a.baseline, "candidate": a.candidate, "arm": arm, "metric": k, "n_sessions": len(d), "base": mb, "cand": mc,
                         "delta": statistics.fmean(d), "ci_lo": boot[50] if len(d) >= 3 else None, "ci_hi": boot[1949] if len(d) >= 3 else None})
    if not lines:
        raise SystemExit(f"no paired scored holdout runs for {a.baseline} vs {a.candidate}")
    led = B / ".sealed" / "holdout-ledger.jsonl"
    led.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    with led.open("a") as f:
        f.write(json.dumps({"ts": ts, "baseline": a.baseline, "candidate": a.candidate, "arms": arms}) + "\n")
    if DB:  # aggregates only: no sid, no run id
        f = B / ".sealed" / f"ledger-{time.strftime('%Y%m%d-%H%M%S')}.json"
        write_json(f, [{"query": f"{ts}-{len(led.read_text().splitlines())}", **r} for r in rows])  # the query id: a repeat of the same comparison appends
        dev("ledger", "--json", f)
    print("\n".join(lines))
    print(f"holdout queries so far: {len(led.read_text().splitlines())}")


def cmd_report(a) -> None:
    if a.which == "dev":
        report_dev(a)
    elif a.baseline and a.candidate:
        report_holdout(a)
    else:
        raise SystemExit("report holdout needs --baseline and --candidate")


def judge(sid: str, ids, va: str, vb: str, model: str | None) -> str:
    head, rows = brief_rows(sid, set(ids))
    prompt = JUDGE + f"\nCluster(s) {list(ids)}\nVerdict A: {va}\nVerdict B: {vb}\n\n{head}\n" + "\n".join(rows.values()) + "\n"
    with tempfile.TemporaryDirectory(dir=B / ".sealed") as d:
        _, out = claude(prompt, Path(d), {}, None, model, {}, 600)
    last = str(out.get("result", "")).strip().splitlines()[-1:]  # strict: the final line is the whole verdict
    return last[0].strip() if last and last[0].strip() in ("A", "B", "both-wrong") else ""  # anything else stays blank for a human


def cmd_adjudicate(a) -> None:
    sealed, d = B / ".sealed" / f"adjudication-{a.split}.json", home(a.split) / "adjudication" / a.split
    if a.unmask:
        key, disputed = json.loads(sealed.read_text()), {}
        with (d / "cases.tsv").open(newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for r in rows:
            k = r["choice"].strip()
            if k in ("A", "B") and key[r["case"]][k] == "arm":
                disputed.setdefault(key[r["case"]]["session"], set()).update(key[r["case"]]["ids"])
        for sid, ids in disputed.items():
            p = home(a.split) / "disputed" / f"{sid}.tsv"
            old = {int(l) for l in p.read_text().splitlines()[1:]} if p.exists() else set()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("cluster_id\n" + "".join(f"{i}\n" for i in sorted(old | set(ids))))  # sorrel diff --disputed format
        print(f"{sum(1 for r in rows if not r['choice'].strip())} case(s) still blank; disputed units written for {len(disputed)} session(s)")
        return
    if a.judge and a.split == "holdout":
        raise SystemExit("refusing: holdout cases are adjudicated by humans only")
    if (d / "cases.tsv").exists() and not a.force:
        raise SystemExit(f"{d / 'cases.tsv'} exists (filled choices would be lost); --force to regenerate")
    cases: dict = {}
    for rec, sc in scored(a.split):
        for m in misses(sc):
            cases.setdefault((rec["session"], miss_ids(m), verdict(m, "truth"), verdict(m, "test")), None)
    rng, order, key, out = random.SystemRandom(), list(cases), {}, []
    rng.shuffle(order)
    for n, (sid, ids, human, arm) in enumerate(order, 1):
        cid, flip = f"c{n:03d}", rng.random() < 0.5
        va, vb = (arm, human) if flip else (human, arm)
        key[cid] = {"session": sid, "ids": list(ids), "A": "arm" if flip else "human", "B": "human" if flip else "arm"}
        choice = judge(sid, ids, va, vb, a.model) if a.judge else ""
        out.append([cid, sid, ",".join(map(str, ids)), sess(sid) / "input", va, vb, choice])
    d.mkdir(parents=True, exist_ok=True)
    write_json(sealed, key)
    with (d / "cases.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["case", "session", "root_clusters", "input_dir", "verdict_A", "verdict_B", "choice"])
        w.writerows(out)
    print(f"{len(out)} case(s) -> {d / 'cases.tsv'}; fill `choice` with A, B or both-wrong, then adjudicate --unmask")


def main(argv=None) -> None:
    global B, DB
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bench", type=Path, required=True)
    p.add_argument("--db", type=Path, help="results DB (default $SORREL_DEV_DB): record dev results with `sorrel dev`")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("add")
    s.add_argument("session", nargs="+")
    s.add_argument("--phy-curated", action="store_true", help="sessions were curated in phy, not sorrel")
    s.add_argument("--dev", action="store_true", help="force into dev: the improver has already seen these truths")
    s = sub.add_parser("run")
    s.add_argument("--arm", choices=ARMS, required=True)
    s.add_argument("--variant", type=Path, required=True)
    s.add_argument("--split", choices=("dev", "holdout"), required=True)
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--model")
    s.add_argument("--jobs", type=int, default=2)
    s.add_argument("--timeout", type=int, default=3600)
    s.add_argument("--max-budget-usd", type=float, default=10.0, help="per mcp-review run (claude --max-budget-usd)")
    s = sub.add_parser("score")
    s.add_argument("--split", choices=("dev", "holdout", "all"), default="all")
    s.add_argument("--match", default="0.8")
    s = sub.add_parser("report")
    s.add_argument("which", choices=("dev", "holdout"))
    s.add_argument("--baseline")
    s.add_argument("--candidate")
    s = sub.add_parser("adjudicate")
    s.add_argument("--split", choices=("dev", "holdout"), required=True)
    s.add_argument("--judge", choices=("claude",))
    s.add_argument("--model")
    s.add_argument("--unmask", action="store_true")
    s.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    B = a.bench.resolve()
    db = a.db or os.environ.get("SORREL_DEV_DB")
    DB = Path(db).resolve() if db else None
    (B / ".sealed").mkdir(parents=True, exist_ok=True)
    {"add": cmd_add, "run": cmd_run, "score": cmd_score, "report": cmd_report, "adjudicate": cmd_adjudicate}[a.cmd](a)


if __name__ == "__main__":
    main()

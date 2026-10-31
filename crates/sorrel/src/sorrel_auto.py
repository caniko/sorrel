# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "spikeinterface[full]>=0.105",
#   "skops",
#   "huggingface_hub",
#   "anthropic",
# ]
# ///
"""SpikeInterface auto-curation sidecar for `sorrel auto`.

Kilosort/phy directory in, two files out (in --out):

- curation.json: SpikeInterface curation format v2. `quality` labels,
  non-exclusive `tags` recording who decided what (`src:*`, `merged:*`,
  `review:*`, ...) and merges, all keyed by the ORIGINAL cluster ids.
  `sorrel apply --curation` replays it as one undoable batch, and
  `spikeinterface.curation.apply_curation` reads it as-is.
- evidence.json: per-unit classifier outputs, Bombcell failures and, with
  --llm, the exact user message the LLM saw (`llm_input`, the shared brief header once as
  `llm_header`) plus its verdict.

Pipeline (current SpikeInterface practice, Oct 2026): remove excess
spikes -> analyzer (cached) -> redundant units -> merge presets
`similarity_correlograms` + `slay` -> Bombcell + UnitRefine labels on the
merged units -> optional LLM tiebreak (claude/codex/opencode CLI or API) on what the classifiers leave
uncertain. Splits are left to sorrel's own `review:split` tags.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HIGH_CONF = 0.85  # UnitRefine probability for `conf:high`
LOW_CONF = 0.6  # below: `conf:low` + `review:low-conf`
LLM_UNSURE = 0.7  # Claude confidence below: `review:llm-unsure`
PRESETS = ("similarity_correlograms", "slay")
EXTENSIONS = [
    "random_spikes",
    "noise_levels",
    "waveforms",
    "templates",
    "spike_amplitudes",
    "unit_locations",
    "spike_locations",  # drift metrics (UnitRefine uses drift_ptp/std/mad)
    "correlograms",
    "template_similarity",
    "principal_components",
    "template_metrics",
    "quality_metrics",
]
EXTENSION_PARAMS = {
    "random_spikes": {"max_spikes_per_unit": 500, "seed": 0},  # reproducible reruns
    "waveforms": {"ms_before": 1.0, "ms_after": 2.0},
    "principal_components": {"n_components": 5, "mode": "by_channel_local"},
    "template_metrics": {"include_multi_channel_metrics": True},
    "correlograms": {"window_ms": 60.0, "bin_ms": 1.0},
}


def log(msg: str) -> None:
    print(f"[sorrel-auto] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Loading


def read_params(data_dir: Path) -> dict:
    """phy's params.py, parsed literal by literal (never exec'd)."""
    out = {}
    path = data_dir / "params.py"
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if not sep:
            continue
        try:
            out[key.strip()] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            continue
    return out


def read_spikeglx_meta(path: Path) -> dict:
    """Sample rate, channel count and AP gain (uV/bit) from a SpikeGLX .meta."""
    meta = dict(
        line.split("=", 1) for line in path.read_text().splitlines() if "=" in line
    )
    out = {"sample_rate": float(meta["imSampRate"]), "n_channels": int(meta["nSavedChans"])}
    # NP1.0-style IMRO tables carry a per-channel AP gain; NP2.0 is fixed at 80.
    gain = 80.0
    imro = meta.get("~imroTbl", "")
    entries = [e for e in imro.strip("()").split(")(")[1:] if e]
    if entries and len(entries[0].split()) >= 6:
        gain = float(entries[0].split()[3])
    if "imAiRangeMax" in meta and "imMaxInt" in meta:
        out["gain_to_uv"] = float(meta["imAiRangeMax"]) / float(meta["imMaxInt"]) / gain * 1e6
    return out


def load(args):
    import spikeinterface.core as sc
    import spikeinterface.curation as scur
    import spikeinterface.preprocessing as spre

    data_dir = Path(args.data_dir)
    params = read_params(data_dir)
    meta = read_spikeglx_meta(Path(args.spikeglx_meta)) if args.spikeglx_meta else {}

    dat = args.dat or params.get("dat_path")
    if isinstance(dat, (list, tuple)):
        dat = dat[0]
    if not dat:
        raise SystemExit("no raw recording: pass --dat or set dat_path in params.py")
    dat = Path(dat)
    if not dat.is_absolute():
        dat = data_dir / dat
    fs = args.sample_rate or meta.get("sample_rate") or params.get("sample_rate")
    n_ch = args.channels or meta.get("n_channels") or params.get("n_channels_dat")
    dtype = args.dtype or params.get("dtype", "int16")
    offset = args.offset if args.offset is not None else params.get("offset", 0)
    if not fs or not n_ch:
        raise SystemExit("sample rate / channel count unknown: pass --sample-rate and --channels")

    rec = sc.read_binary(
        file_paths=[str(dat)],
        sampling_frequency=float(fs),
        dtype=dtype,
        num_channels=int(n_ch),
        file_offset=int(offset),
    )
    gain = args.gain_to_uv or meta.get("gain_to_uv")
    if gain:
        rec.set_channel_gains(float(gain))
        rec.set_channel_offsets(0.0)
    else:
        log(
            "WARNING: no --gain-to-uv; amplitudes stay in ADC units, so Bombcell's "
            "uV amplitude threshold is meaningless for this run"
        )

    # Kilosort sorts a subset of the dat channels (channel_map) at known positions.
    chan_map_path = data_dir / "channel_map.npy"
    if chan_map_path.exists():
        chan_map = np.load(chan_map_path).ravel().astype(int)
        rec = rec.select_channels(rec.channel_ids[chan_map])
    positions = np.load(data_dir / "channel_positions.npy").astype(float)
    gaps = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    geometry_ok = bool(np.min(gaps[gaps > 0], initial=np.inf) >= 5.0)
    if not geometry_ok:
        # Tetrode/index layouts: Bombcell and UnitRefine were built on real
        # probe geometry (Neuropixels), so their spatial metrics mislead here.
        log(
            "WARNING: channels are < 5 um apart; channel_positions look like indices, "
            "so spatial metrics (exp_decay, spread, velocity, drift) and the labels "
            "built on them are unreliable for this recording; every unit gets review:geometry"
        )
    rec.set_dummy_probe_from_locations(positions)
    if not params.get("hp_filtered", False):
        rec = spre.highpass_filter(rec, freq_min=300.0)
    rec = spre.common_reference(rec, operator="median")

    # Exactly the ids sorrel uses: spike_clusters.npy values. (read_kilosort
    # would prefer a `si_unit_id` column when SpikeInterface wrote the folder.)
    times = np.load(data_dir / "spike_times.npy").ravel().astype(np.int64)
    clusters_path = data_dir / "spike_clusters.npy"
    if not clusters_path.exists():
        clusters_path = data_dir / "spike_templates.npy"
    clusters = np.load(clusters_path).ravel().astype(np.int64)
    sorting = sc.NumpySorting.from_samples_and_labels([times], [clusters], float(fs))
    sorting = scur.remove_excess_spikes(sorting, rec)
    return rec, sorting, bool(gain), geometry_ok


def fingerprint(args) -> str:
    """What the cached analyzer depends on: the spike trains and their cluster
    assignment (values, not bytes: a sorrel save rewrites the same ids as
    uint32), the extension parameters and the recording options."""
    h = hashlib.sha256()
    data_dir = Path(args.data_dir)
    clusters = data_dir / "spike_clusters.npy"
    if not clusters.exists():
        clusters = data_dir / "spike_templates.npy"
    for p in (data_dir / "spike_times.npy", clusters):
        h.update(np.load(p).ravel().astype(np.int64).tobytes())
    h.update(json.dumps(EXTENSION_PARAMS, sort_keys=True).encode())
    for key in ("dat", "sample_rate", "channels", "dtype", "offset", "spikeglx_meta", "gain_to_uv"):
        h.update(f"{key}={getattr(args, key)!r};".encode())
    return h.hexdigest()


def build_analyzer(rec, sorting, out: Path, fp: str):
    import spikeinterface.core as sc

    folder = out / "analyzer"
    stamp = out / "analyzer.fingerprint"
    if folder.exists() and stamp.exists() and stamp.read_text() == fp:
        log(f"reusing cached analyzer {folder}")
        analyzer = sc.load_sorting_analyzer(folder)
    else:
        shutil.rmtree(folder, ignore_errors=True)
        stamp.unlink(missing_ok=True)
        analyzer = sc.create_sorting_analyzer(
            sorting, rec, format="binary_folder", folder=folder, sparse=True
        )
    missing = [e for e in EXTENSIONS if not analyzer.has_extension(e)]
    if missing:
        log(f"computing {', '.join(missing)}")
        analyzer.compute(
            missing, extension_params={k: v for k, v in EXTENSION_PARAMS.items() if k in missing}
        )
    stamp.write_text(fp)
    return analyzer


# --------------------------------------------------------------------------
# Metrics and evidence


def metrics_table(analyzer):
    import pandas as pd

    frames = [
        analyzer.get_extension(name).get_data()
        for name in ("quality_metrics", "template_metrics")
        if analyzer.has_extension(name)
    ]
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=analyzer.unit_ids)


def finite(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, 4) if np.isfinite(v) else None


# --------------------------------------------------------------------------
# Brief: the LLM reads the brief's text, not arrays built here. MCP/brief
# cluster ids are ClusterId indices; for Kilosort/phy (the only backend
# `sorrel auto` supports) that is the spike_clusters.npy id, i.e. our unit id.


def sorrel_call(data_dir: str, tool: str, arguments: dict) -> str:
    """One MCP tool (`brief`, `eval`) through `sorrel call`, with the binary and
    open arguments `sorrel auto` passed in the environment. It goes through a
    running host, else opens the dataset itself. Fails loudly: with no brief no
    case can be judged."""
    exe = os.environ.get("SORREL_BIN")
    if not exe:
        raise SystemExit("--llm needs the sorrel binary: run through `sorrel auto` (SORREL_BIN unset)")
    open_args = json.loads(os.environ.get("SORREL_OPEN_ARGS", "[]"))
    cmd = [exe, "call", *open_args, data_dir, tool, json.dumps(arguments)]
    try:
        r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"sorrel call {tool} {arguments} timed out after 600 s")
    if r.returncode:
        raise SystemExit(f"sorrel call {tool} {arguments} failed ({r.returncode}): {r.stderr.strip()}")
    return r.stdout


def split_brief(text: str) -> tuple[str, str, dict[int, str]]:
    """(header, card header line, {unit id: card row}). The header is everything
    before the `cards:` table; rows end at the first line without a tab
    (`omitted: ...`)."""
    header, _, table = text.partition("\ncards:\n")
    lines = table.splitlines()
    rows = {}
    for line in lines[1:]:
        if "\t" not in line:
            break
        try:
            rows[int(line.split("\t", 1)[0])] = line
        except ValueError:
            log(f"brief: skipping malformed card row {line[:60]!r}")
    return header.rstrip("\n"), (lines[0] if lines else ""), rows


def eval_line(text: str) -> str | None:
    """The one-line merge counterfactual the `eval` tool returns, or None (the
    case then stays in review)."""
    line = text.strip()
    if not line or "\n" in line:
        log(f"eval returned no single counterfactual line ({text[-200:]!r}); leaving it for review")
        return None
    return line


def unit_message(card_head: str, rows: dict[int, str], record: dict, evals: dict[int, str | None]) -> str | None:
    """User message for one (possibly merged) unit: its card row(s), the merge
    counterfactual when it is a merge of several units, the classifier outputs.
    None (no LLM call, the unit stays in review) unless every member has a card
    and a merge has its counterfactual: no partial judgement."""
    members = record.get("members", [])
    if not members or any(m not in rows for m in members) or (len(members) > 1 and not evals.get(members[0])):
        log(f"no complete evidence for unit(s) {members}; leaving it for review")
        return None
    cards = [rows[m] for m in members]
    classifiers = {k: v for k, v in record.items() if k != "members"}
    msg = "Decide this unit.\n"
    if len(members) > 1:
        msg = f"Decide this unit, the merge of {','.join(map(str, members))} (cards are its parts).\n"
    msg += card_head + "\n" + "\n".join(cards)
    if len(members) > 1:
        msg += "\n" + evals[members[0]]
    return msg + "\nclassifiers: " + json.dumps(classifiers, separators=(",", ":"))


def merge_message(line: str, presets) -> str:
    return f"Decide this merge.\n{line}\nproposed_by: {','.join(sorted(presets))}"


def bombcell_failures(row, thresholds: dict, section: str) -> list[str]:
    """The metrics of `section` ('noise' / 'mua') a unit fails, for `bc:*` tags.
    Labels themselves come from SpikeInterface; this only explains them, with
    its rules: inclusive bounds (`greater` is >=, `less` is <=), and NaN fails
    every noise threshold but only the listed MUA metrics."""
    from spikeinterface.curation import bombcell_curation

    nan_fails = section == "noise" or None
    listed = getattr(bombcell_curation, "_NAN_MEANS_FAIL_METRICS", ())
    failed = []
    for metric, spec in (thresholds.get(section) or {}).items():
        if not isinstance(spec, dict) or metric not in getattr(row, "index", []):
            continue
        v = float(row[metric])  # unrounded: some bounds are 0.0001 s
        if not np.isfinite(v):
            if nan_fails or metric in listed:
                failed.append(metric)
            continue
        if spec.get("abs"):
            v = abs(v)
        lo, hi = spec.get("greater"), spec.get("less")
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            failed.append(metric)
    return failed


# --------------------------------------------------------------------------
# Claude

RUBRIC = """You review automatically sorted extracellular units (Kilosort 4, \
SpikeInterface metrics) and decide what a careful human curator would decide. \
Two classifiers disagreed or were unsure about the case you get; their outputs \
are included. Judge the evidence, not the classifiers' authority.

Units: label each as
- "good": a well-isolated single neuron. Clean biphasic somatic waveform \
(trough then peak, 0.2-1.0 ms trough-to-peak), spatially compact footprint \
decaying over ~50-100 um, empty refractory period (ACG near zero in the first \
1-2 ms, isi_violations_ratio and rp_contamination low), stable firing and \
amplitude across the recording (presence_ratio high), amplitude_cutoff low.
- "mua": neural but not isolated. Neural waveform, but refractory violations, \
an amplitude distribution cut off by the detection threshold, or a mixture of \
neurons (unstable amplitude, broad or multi-peaked footprint).
- "noise": not neural. Non-physiological shape (too many peaks, very short or \
very long, flat baseline violations), footprint spread uniformly across many \
distant channels or confined to one channel with no decay, line-noise or \
artefact-driven firing, or near-zero amplitude.

Merges: decide whether the listed units are one neuron split by the sorter. \
Merge when templates on the shared channel are nearly identical in shape (an \
amplitude difference alone is fine: drift), units are close (< ~50 um), and the \
cross-correlogram shows a refractory dip at 0 ms like an ACG. Do not merge when \
the CCG has a central peak (synchrony between two neurons) or no dip, or the \
shapes differ.

The input is a sorrel brief: the header above defines the columns, flags \
and SAX strings (compare shape, not level); a unit message is its card row(s) \
and the classifier outputs, a merge message the counterfactual of the merge. \
Amplitudes are in uV when the input says so, otherwise in ADC units. Give a \
confidence in [0, 1] (0.5 = coin flip) and at most three short reasons."""

UNIT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["good", "mua", "noise"]},
        "confidence": {"type": "number"},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["label", "confidence", "reasons"],
    "additionalProperties": False,
}
MERGE_SCHEMA = {
    "type": "object",
    "properties": {
        "merge": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["merge", "confidence", "reasons"],
    "additionalProperties": False,
}


SCHEMAS = {"unit": UNIT_SCHEMA, "merge": MERGE_SCHEMA}
BACKENDS = ("claude", "codex", "opencode", "api")
LOGIN = {
    "claude": "install Claude Code and run `claude`, then /login",
    "codex": "install codex and run `codex login`",
    "opencode": "install opencode and run `opencode auth login`",
}


def parse_verdict(text, kind: str) -> dict | None:
    """The first valid verdict object in `text` (bare, fenced or inside prose),
    normalised; None if there is none. Every backend routes through here, and
    for opencode (no schema enforcement) this is the only check."""
    if not isinstance(text, str):
        return None
    key, dec = ("label" if kind == "unit" else "merge"), json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        v, c, r = obj.get(key), obj.get("confidence"), obj.get("reasons")
        good = v in ("good", "mua", "noise") if kind == "unit" else isinstance(v, bool)
        if good and isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) and isinstance(r, list):
            return {key: v, "confidence": min(max(float(c), 0.0), 1.0), "reasons": [str(x) for x in r][:3]}
    return None


def ask_api(client, model: str, system: str, kind: str, message: str) -> dict | None:
    """One structured verdict from the Anthropic API, or None when Claude
    declines or errors (the caller then falls back to review tags)."""
    import anthropic

    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=4096,
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMAS[kind]}},
            messages=[{"role": "user", "content": message}],
        )
    except anthropic.APIError as e:
        log(f"Claude request failed ({type(e).__name__}: {e}); leaving it for review")
        return None
    if response.stop_reason != "end_turn":
        log(f"Claude stopped with {response.stop_reason}; leaving it for review")
        return None
    verdict = parse_verdict(next((b.text for b in response.content if b.type == "text"), None), kind)
    if verdict:
        verdict["model"] = response.model
    return verdict


def cli_argv(backend: str, model: str | None, system: str, kind: str, message: str, root: Path):
    """(argv, stdin text, extra env, cwd) for one case. `root` is the case's
    scratch dir: the CLI runs in the empty `root/cwd` and reads `root/schema.json`
    / `root/mcp.json` and writes `root/out.json` beside it.
    ponytail: the system prompt is one argv string for claude (Linux caps a
    single arg at 128 KiB, Windows' whole command line at 32 KiB); switch to a
    prompt file if a brief header outgrows it. argv[0] is the resolved path so
    npm `.cmd` shims work on Windows."""
    cwd, m = root / "cwd", (["-m", model] if model else [])
    exe = shutil.which(backend) or backend
    if backend == "claude":
        # Empty MCP set, project-only settings and an empty cwd keep the user's
        # own Claude Code context (~50k tokens) out of every call.
        argv = [exe, "-p", "--output-format", "json", "--json-schema", json.dumps(SCHEMAS[kind]),
                "--system-prompt", system, "--tools", "", "--no-session-persistence", "--strict-mcp-config",
                "--mcp-config", str(root / "mcp.json"), "--setting-sources", "project", "--disable-slash-commands"]
        return argv + (["--model", model] if model else []), message, {}, cwd
    if backend == "codex":  # no system-prompt flag
        argv = [exe, "exec", "--ephemeral", "--skip-git-repo-check", "-s", "read-only", "--output-schema",
                str(root / "schema.json"), "-o", str(root / "out.json"), "-C", str(cwd), "--json", *m, "-"]
        return argv, f"{system}\n\n{message}", {}, cwd
    # opencode: no schema, system or stdin; tools denied through its inline permission config
    prompt = f"{system}\n\n{message}\n\nReply with ONLY a JSON object matching this schema: {json.dumps(SCHEMAS[kind])}"
    return [exe, "run", "--format", "json", *m, prompt], "", {"OPENCODE_PERMISSION": '{"*":"deny"}'}, cwd


def cli_text(backend: str, stdout: str, out_file: Path) -> tuple[str | None, str | None, str | None]:
    """(reply text, model the CLI reported, failure message) from a CLI's output."""
    if backend == "claude":
        try:
            env = json.loads(stdout)
        except json.JSONDecodeError:
            return None, None, "unparseable output"
        if env.get("is_error"):
            return None, None, str(env.get("result"))
        so = env.get("structured_output")
        return (json.dumps(so) if so else env.get("result")), next(iter(env.get("modelUsage") or {}), None), None
    events = []
    for line in stdout.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(e, dict):
            events.append(e)
    why = None
    for e in reversed(events):
        if e.get("type") in ("error", "turn.failed"):
            x = e.get("error")
            why = (x.get("message") if isinstance(x, dict) else x) or e.get("message")
            break
    if backend == "codex":
        return (out_file.read_text() if out_file.exists() else None) or None, None, why
    texts = [(e.get("part") or {}).get("text") or e.get("text") for e in events if e.get("type") == "text"]
    return (texts[-1] if texts else None), None, why


def ask_cli(backend: str, model: str | None, system: str, kind: str, message: str) -> dict | None:
    """One verdict from an agent CLI (one subprocess per case), or None (the
    caller then falls back to review tags).
    ponytail: a process per case, 8 in parallel; a persistent session per
    backend is the upgrade if process start-up dominates."""
    with tempfile.TemporaryDirectory(prefix="sorrel-llm-", ignore_cleanup_errors=True) as d:
        root = Path(d)
        argv, stdin, extra, cwd = cli_argv(backend, model, system, kind, message, root)
        cwd.mkdir()
        (root / "schema.json").write_text(json.dumps(SCHEMAS[kind]))
        (root / "mcp.json").write_text('{"mcpServers":{}}')
        try:
            r = subprocess.run(argv, input=stdin, cwd=cwd, env={**os.environ, **extra}, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
        except (subprocess.TimeoutExpired, OSError) as e:
            log(f"{backend} failed ({type(e).__name__}: {e}); leaving it for review")
            return None
        text, reported, why = cli_text(backend, r.stdout, root / "out.json")
    if r.returncode != 0 or text is None:
        log(f"{backend} failed (exit {r.returncode}: {why or r.stderr.strip()[-300:] or 'no reply'}); leaving it for review")
        return None
    verdict = parse_verdict(text, kind)
    if verdict is None:
        log(f"{backend} reply was not a valid verdict ({text[:120]!r}); leaving it for review")
        return None
    verdict["model"] = f"claude-cli:{reported or model or 'claude'}" if backend == "claude" else f"{backend}:{model or backend}"
    return verdict


def api_credential(env=os.environ) -> bool:
    """Whether the Anthropic SDK will find a credential: an env credential, or
    the active `ant auth login` profile's stored credentials (the SDK's
    documented chain; `ant auth status` must not be scripted against).
    ponytail: workload-identity federation env vars are not checked."""
    if env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    if env.get("ANTHROPIC_CONFIG_DIR"):
        base = Path(env["ANTHROPIC_CONFIG_DIR"])
    elif sys.platform == "win32":
        base = Path(env.get("APPDATA", "")) / "Anthropic"
    else:
        base = Path(env.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "anthropic"
    profile = env.get("ANTHROPIC_PROFILE")
    if not profile:
        active = base / "active_config"
        profile = (active.read_text().strip() if active.is_file() else "") or "default"
    return (base / "credentials" / f"{profile}.json").is_file()


def run_llm(backend: str | None, model: str | None, system: str, jobs: list[tuple[str, str | None]]) -> list[dict | None]:
    """One verdict per (kind, message) job; None for a job without a message or
    whose call raised, so one bad case never loses the others' paid verdicts."""
    if not jobs:
        return []
    if backend == "api":
        import anthropic

        model = model or "claude-opus-5"
        clients = []

        def ask(j):
            # Built on first use, inside `safe`: a credential the SDK cannot
            # resolve leaves the cases in review instead of aborting the run.
            if not clients:
                clients.append(anthropic.Anthropic())
            return ask_api(clients[0], model, system, *j)
    else:
        ask = lambda j: ask_cli(backend, model, system, *j)  # noqa: E731

    def safe(j):
        if j[1] is None:
            return None
        try:
            return ask(j)
        except Exception as e:  # noqa: BLE001
            log(f"{backend} call failed ({type(e).__name__}: {e}); leaving it for review")
            return None

    log(f"asking {backend}{f' ({model})' if model else ''} about {sum(j[1] is not None for j in jobs)} case(s)")
    # ponytail: synchronous calls on a small pool; switch to the Batches API
    # (50% cheaper, but no server-side fallbacks) once api runs reach thousands of cases.
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(safe, jobs))


# --------------------------------------------------------------------------
# Labels


def norm_label(s) -> str | None:
    s = str(s).lower()
    if "noise" in s:
        return "noise"
    if "mua" in s or "multi" in s:
        return "mua"
    if "sua" in s or "good" in s or "single" in s or "neural" in s:
        return "good"
    return None


def resolve_groups(groups_by_preset: dict) -> list[tuple[list, set]]:
    """Union the presets' merge groups; each final group remembers which
    presets proposed (part of) it."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for groups in groups_by_preset.values():
        for g in groups:
            for u in g[1:]:
                parent[find(u)] = find(g[0])
    members = {}
    for u in list(parent):
        members.setdefault(find(u), []).append(u)
    out = []
    for group in members.values():
        group = sorted(group)
        found = {p for p, groups in groups_by_preset.items() if any(len(set(g) & set(group)) >= 2 for g in groups)}
        out.append((group, found))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dat")
    ap.add_argument("--sample-rate", type=float)
    ap.add_argument("--channels", type=int)
    ap.add_argument("--dtype")
    ap.add_argument("--offset", type=int)
    ap.add_argument("--spikeglx-meta")
    ap.add_argument("--gain-to-uv", type=float, help="uV per ADC bit (e.g. 2.34375 for NP1.0 at gain 500)")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--no-unitrefine", action="store_true")
    ap.add_argument("--llm", nargs="?", const="claude", choices=BACKENDS)
    ap.add_argument("--llm-model")
    args = ap.parse_args()
    if args.llm in LOGIN and not shutil.which(args.llm):
        raise SystemExit(f"--llm {args.llm}: `{args.llm}` not found on PATH; {LOGIN[args.llm]}")
    if args.llm == "api" and not api_credential():
        raise SystemExit("--llm=api: no Anthropic credential; set ANTHROPIC_API_KEY or run `ant auth login`")

    import spikeinterface
    import spikeinterface.core as sc
    import spikeinterface.curation as scur
    from packaging.version import Version

    if Version(spikeinterface.__version__) < Version("0.105"):
        raise SystemExit(f"spikeinterface>=0.105 required (found {spikeinterface.__version__})")
    sc.set_global_job_kwargs(n_jobs=args.jobs, chunk_duration="1s", progress_bar=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rec, sorting, in_uv, geometry_ok = load(args)
    analyzer = build_analyzer(rec, sorting, out, fingerprint(args))
    original_ids = [int(u) for u in analyzer.unit_ids]
    labels: dict[int, str] = {}
    tags: dict[int, set] = {u: set() for u in original_ids}
    evidence: dict = {"units": {}, "merges": [], "spikeinterface": spikeinterface.__version__, "amplitudes_in_uv": in_uv, "geometry_ok": geometry_ok}

    # 1. Redundant units (near-duplicate spike trains): labelled noise, never deleted.
    kept = scur.remove_redundant_units(analyzer, remove_strategy="minimum_shift")
    redundant = sorted(set(original_ids) - {int(u) for u in kept.unit_ids})
    for u in redundant:
        labels[u] = "noise"
        tags[u] |= {"redundant", "src:spikeinterface"}
    evidence["redundant"] = redundant
    if redundant:
        analyzer = analyzer.select_units([u for u in analyzer.unit_ids if int(u) not in redundant])
    log(f"{len(redundant)} redundant unit(s)")

    # 2. Merges: each preset separately, so every merge records who found it.
    merges: list[tuple[list, set]] = []
    if not args.no_merge:
        by_preset = {}
        for preset in PRESETS:
            try:
                by_preset[preset] = [[int(u) for u in g] for g in scur.compute_merge_unit_groups(analyzer, preset=preset)]
            except Exception as e:  # a preset missing in this SI build must not sink the run
                log(f"preset {preset} failed: {e}")
                by_preset[preset] = []
        merges = resolve_groups(by_preset)

    llm_merge = [(g, p) for g, p in merges if len(p) < len(PRESETS)] if args.llm else []
    system, evals = RUBRIC, {}
    # Set a previous run's evidence.json aside (renamed, never deleted: its paid
    # verdicts survive a rerun) before anything writes it; the brief's `ev`
    # column would also read it stale.
    old = out / "evidence.json"
    if old.exists():
        old.rename(out / f"evidence.json.{int(time.time())}")
    if args.llm:
        # ponytail: header from the unfiltered level-0 brief (merges are judged
        # before the classifiers pick the unsure units); one `sorrel call`
        # process per merge group, a batched call if runs show hundreds.
        header, _, _ = split_brief(sorrel_call(args.data_dir, "brief", {"level": 0}))
        system = RUBRIC + "\n\n" + header
        evidence["llm_header"] = header
        for g, _ in llm_merge:
            evals[g[0]] = eval_line(sorrel_call(args.data_dir, "eval", {"merge": list(g)}))
    messages = [merge_message(evals[g[0]], p) if evals.get(g[0]) else None for g, p in llm_merge]
    verdicts = run_llm(args.llm, args.llm_model, system, [("merge", m) for m in messages])
    vetoed = []
    for (group, _), verdict, message in zip(llm_merge, verdicts, messages):
        record = {"units": group, "verdict": verdict, "llm_input": message}
        evidence["merges"].append(record)
        if verdict and not verdict["merge"] and verdict["confidence"] >= LLM_UNSURE:
            vetoed.append(group)
            for u in group:
                tags[u] |= {"review:merge", "src:llm"}
    applied = []
    for group, presets in merges:
        if group in vetoed:
            continue
        group_tags = {f"merged:{p}" for p in presets}
        if len(presets) == len(PRESETS):
            group_tags.add("merged:consensus")
        else:
            verdict = next((r["verdict"] for r in evidence["merges"] if r["units"] == group), None)
            if verdict and verdict["confidence"] >= LLM_UNSURE:
                group_tags.add("src:llm")
            else:
                group_tags.add("review:merge")
        for u in group:
            tags[u] |= group_tags
        applied.append(group)
    log(f"{len(applied)} merge(s), {len(vetoed)} vetoed by the LLM")

    # Re-measure merged units: labels must describe the merged neuron.
    members = {u: [u] for u in (int(x) for x in analyzer.unit_ids)}
    if applied:
        next_id = max(original_ids) + 1
        new_ids = list(range(next_id, next_id + len(applied)))
        analyzer = analyzer.merge_units(
            merge_unit_groups=applied, new_unit_ids=new_ids, merging_mode="soft", format="memory"
        )
        for new, group in zip(new_ids, applied):
            members[new] = group
            for u in group:
                members.pop(u, None)

    # 3. Labels on the merged units.
    metrics = metrics_table(analyzer)
    bc = scur.bombcell_label_units(analyzer, split_non_somatic=True)
    thresholds = scur.bombcell_get_default_thresholds()
    ur = None
    if not args.no_unitrefine:
        ur = scur.unitrefine_label_units(
            analyzer,
            noise_neural_classifier="SpikeInterface/UnitRefine_noise_neural_classifier",
            sua_mua_classifier="SpikeInterface/UnitRefine_sua_mua_classifier",
        )

    decided: dict[int, tuple[str, set]] = {}
    llm_units = []
    for unit in analyzer.unit_ids:
        u = int(unit)
        raw_bc = str(bc.loc[unit, "bombcell_label"])
        b = norm_label(raw_bc.replace("non_soma", "")) or "mua"
        t = {"src:bombcell"}
        if "non_soma" in raw_bc:
            t.add("non-somatic")
        row = metrics.loc[unit] if unit in metrics.index else None
        if row is not None and b in ("noise", "mua"):
            t |= {f"bc:{m}" for m in bombcell_failures(row, thresholds, b)}
        record = {"bombcell": raw_bc, "members": members.get(u, [u])}
        label = b
        unsure = False
        if ur is not None:
            r = norm_label(ur.loc[unit, "unitrefine_label"])
            p = finite(ur.loc[unit, "unitrefine_probability"])
            record.update(unitrefine=str(ur.loc[unit, "unitrefine_label"]), unitrefine_probability=p)
            t.add("src:unitrefine")
            if r == b:
                t.add("agree")
            else:
                t.add("disagree")
                label = r or b
                unsure = True
                t.add("review:disagree")
            if p is not None:
                t.add("conf:high" if p >= HIGH_CONF else "conf:low" if p < LOW_CONF else "conf:mid")
                if p < LOW_CONF:
                    unsure = True
                    t.add("review:low-conf")
        decided[u] = (label, t)
        evidence["units"][str(u)] = record
        if unsure and args.llm:
            llm_units.append(u)

    (out / "evidence.json").write_text(json.dumps(evidence, indent=1, default=str))  # merge verdicts survive a crash below
    messages = []
    if llm_units:
        # Merged units are not in the dataset `sorrel call` opens: ask for
        # their members' cards plus the merge counterfactual.
        recs = [evidence["units"][str(u)] for u in llm_units]
        ids = sorted({m for r in recs for m in r["members"]})
        _, card_head, rows = split_brief(
            sorrel_call(args.data_dir, "brief", {"level": 2, "clusters": ids})
        )
        for r in recs:
            m = r["members"]
            if len(m) > 1 and m[0] not in evals:  # None = tried, unusable
                evals[m[0]] = eval_line(sorrel_call(args.data_dir, "eval", {"merge": list(m)}))
        messages = [unit_message(card_head, rows, r, evals) for r in recs]
    verdicts = run_llm(args.llm, args.llm_model, system, [("unit", m) for m in messages])
    for u, verdict, message in zip(llm_units, verdicts, messages):
        label, t = decided[u]
        evidence["units"][str(u)]["llm"] = verdict
        evidence["units"][str(u)]["llm_input"] = message
        if verdict is None:
            continue
        t.add("src:llm")
        label = verdict["label"]
        if verdict["confidence"] >= LLM_UNSURE:
            t -= {"review:disagree", "review:low-conf"}
        else:
            t.add("review:llm-unsure")
        decided[u] = (label, t)

    # 4. Back to original ids: every merge member carries the merged verdict.
    for u, (label, t) in decided.items():
        for m in members.get(u, [u]):
            labels[m] = label
            tags[m] |= t
            if not geometry_ok:
                tags[m].add("review:geometry")

    all_tags = set().union(*tags.values()) | {"src:human", "src:sorrel"}
    doc = {
        "format_version": "2",
        "unit_ids": original_ids,
        "label_definitions": {
            "quality": {"label_options": ["good", "mua", "noise"], "exclusive": True},
            "tags": {"label_options": sorted(all_tags), "exclusive": False},
        },
        "manual_labels": [
            {"unit_id": u, "labels": {"quality": [labels[u]], "tags": sorted(tags[u])}}
            for u in original_ids
            if u in labels
        ],
        "merges": [{"unit_ids": g} for g in applied],
        "splits": [],
        "removed": [],
    }
    scur.validate_curation_dict(doc)
    (out / "curation.json").write_text(json.dumps(doc, indent=1))
    (out / "evidence.json").write_text(json.dumps(evidence, indent=1, default=str))
    review = sum(any(t.startswith("review:") for t in tags[u]) for u in original_ids)
    log(f"wrote {out / 'curation.json'}: {len(doc['manual_labels'])} labels, {len(applied)} merges, {review} for review")


if __name__ == "__main__":
    main()

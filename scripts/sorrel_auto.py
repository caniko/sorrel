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
  --llm, the exact evidence Claude saw plus its verdict.

Pipeline (current SpikeInterface practice, Oct 2026): remove excess
spikes -> analyzer (cached) -> redundant units -> merge presets
`similarity_correlograms` + `slay` -> Bombcell + UnitRefine labels on the
merged units -> optional Claude tiebreak on what the classifiers leave
uncertain. Splits are left to sorrel's own `review:split` tags.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
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


EVIDENCE_METRICS = [
    "num_spikes",
    "firing_rate",
    "presence_ratio",
    "isi_violations_ratio",
    "rp_contamination",
    "sliding_rp_violation",
    "amplitude_median",
    "amplitude_cutoff",
    "amplitude_cv_median",
    "snr",
    "drift_ptp",
    "isolation_distance",
    "l_ratio",
    "d_prime",
    "silhouette",
    "nn_hit_rate",
    "peak_to_trough_duration",
    "peak_to_valley",
    "half_width",
    "num_positive_peaks",
    "num_negative_peaks",
    "repolarization_slope",
    "recovery_slope",
    "exp_decay",
    "spread",
    "velocity_above",
    "velocity_below",
]


def unit_evidence(analyzer, metrics, unit_id) -> dict:
    """Compact, numeric description of one unit for the LLM (no images: every
    signal that decides a label is already a number)."""
    row = metrics.loc[unit_id] if unit_id in metrics.index else {}
    template = analyzer.get_extension("templates").get_unit_template(unit_id, operator="average")
    ptp = np.ptp(template, axis=0)
    peak = int(np.argmax(ptp))
    trace = template[:, peak]
    resampled = np.interp(np.linspace(0, len(trace) - 1, 25), np.arange(len(trace)), trace)
    locs = analyzer.get_channel_locations()
    near = np.argsort(np.linalg.norm(locs - locs[peak], axis=1))[:8]
    fs = analyzer.sampling_frequency

    ccgs, bins = analyzer.get_extension("correlograms").get_data()
    ui = list(analyzer.unit_ids).index(unit_id)
    centers = (bins[:-1] + bins[1:]) / 2
    acg = ccgs[ui, ui][(centers >= 0) & (centers <= 30)]

    times = analyzer.sorting.get_unit_spike_train(unit_id) / fs
    duration = analyzer.get_total_duration()
    edges = np.linspace(0, duration, 11)
    counts = np.histogram(times, edges)[0]
    amps = analyzer.get_extension("spike_amplitudes").get_data(outputs="by_unit")[0][unit_id]
    deciles = np.clip(np.digitize(times, edges) - 1, 0, 9)
    med_amp = [finite(np.median(amps[deciles == d])) if np.any(deciles == d) else None for d in range(10)]

    return {
        "metrics": {k: finite(row[k]) for k in EVIDENCE_METRICS if k in getattr(row, "index", [])},
        "peak_channel_template_25pts": [round(float(v), 1) for v in resampled],
        "template_window_ms": [-1.0, 2.0],
        "footprint_ptp": [
            {
                "dx_um": round(float(locs[c, 0] - locs[peak, 0]), 1),
                "dy_um": round(float(locs[c, 1] - locs[peak, 1]), 1),
                "ptp": round(float(ptp[c]), 1),
            }
            for c in near
        ],
        "acg_counts_0_to_30ms_1ms_bins": [int(v) for v in acg],
        "firing_rate_per_decile_hz": [round(float(c / (duration / 10)), 3) for c in counts],
        "median_amplitude_per_decile": med_amp,
    }


def merge_evidence(analyzer, group, presets) -> dict:
    ids = list(analyzer.unit_ids)
    templates = analyzer.get_extension("templates")
    first = templates.get_unit_template(group[0], operator="average")
    shared = int(np.argmax(np.ptp(first, axis=0)))
    sim = analyzer.get_extension("template_similarity").get_data()
    locs = analyzer.get_extension("unit_locations").get_data()
    ccgs, bins = analyzer.get_extension("correlograms").get_data()
    centers = (bins[:-1] + bins[1:]) / 2
    window = np.abs(centers) <= 30
    a = ids.index(group[0])
    pairs = []
    for other in group[1:]:
        b = ids.index(other)
        pairs.append(
            {
                "units": [int(group[0]), int(other)],
                "template_similarity": finite(sim[a, b]),
                "distance_um": finite(np.linalg.norm(locs[a][:2] - locs[b][:2])),
                "ccg_counts_-30_to_30ms_1ms_bins": [int(v) for v in ccgs[a, b][window]],
            }
        )
    return {
        "proposed_by": sorted(presets),
        "templates_on_shared_channel_25pts": {
            str(u): [
                round(float(v), 1)
                for v in np.interp(
                    np.linspace(0, first.shape[0] - 1, 25),
                    np.arange(first.shape[0]),
                    templates.get_unit_template(u, operator="average")[:, shared],
                )
            ]
            for u in group
        },
        "pairs": pairs,
    }


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


def ask_claude(client, model: str, kind: str, payload: dict) -> dict | None:
    """One structured verdict, or None when Claude declines or errors (the
    caller then falls back to review tags)."""
    import anthropic

    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=4096,
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},
            system=[{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral"}}],
            output_config={
                "effort": "low",
                "format": {
                    "type": "json_schema",
                    "schema": UNIT_SCHEMA if kind == "unit" else MERGE_SCHEMA,
                },
            },
            messages=[
                {
                    "role": "user",
                    "content": f"Decide this {kind}.\n\n" + json.dumps(payload, separators=(",", ":")),
                }
            ],
        )
    except anthropic.APIError as e:
        log(f"Claude request failed ({type(e).__name__}: {e}); leaving it for review")
        return None
    if response.stop_reason != "end_turn":
        log(f"Claude stopped with {response.stop_reason}; leaving it for review")
        return None
    text = next((b.text for b in response.content if b.type == "text"), None)
    try:
        verdict = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    verdict["confidence"] = min(max(float(verdict.get("confidence", 0.0)), 0.0), 1.0)
    verdict["reasons"] = [str(r) for r in verdict.get("reasons", [])][:3]
    verdict["model"] = response.model
    return verdict


def run_claude(model: str, jobs: list[tuple[str, dict]]) -> list[dict | None]:
    if not jobs:
        return []
    import anthropic

    client = anthropic.Anthropic()
    log(f"asking {model} about {len(jobs)} case(s)")
    # ponytail: synchronous calls on a small pool; switch to the Batches API
    # (50% cheaper, but no server-side fallbacks) once runs reach thousands of cases.
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda j: ask_claude(client, model, *j), jobs))


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
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--llm-model", default="claude-opus-5")
    args = ap.parse_args()

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
    verdicts = run_claude(
        args.llm_model, [("merge", merge_evidence(analyzer, g, p)) for g, p in llm_merge]
    )
    vetoed = []
    for (group, _), verdict in zip(llm_merge, verdicts):
        record = {"units": group, "verdict": verdict}
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
    log(f"{len(applied)} merge(s), {len(vetoed)} vetoed by Claude")

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

    verdicts = run_claude(
        args.llm_model,
        [
            (
                "unit",
                {
                    "amplitudes_in_uv": in_uv,
                    "classifiers": {k: v for k, v in evidence["units"][str(u)].items() if k != "members"},
                    **unit_evidence(analyzer, metrics, u),
                },
            )
            for u in llm_units
        ],
    )
    for u, verdict in zip(llm_units, verdicts):
        label, t = decided[u]
        evidence["units"][str(u)]["llm"] = verdict
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

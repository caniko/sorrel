"""Write a curated Sorrel session into an NWB file as its Units table.

Needs PyNWB: ``pip install "sorrel-ephys[nwb]"``. Raw traces, the electrodes
table and waveforms are out of scope; pair this with the recording's own
NeuroConv interface for those (see ``sorrel.neuroconv``).
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from hdmf.common import VectorData, VectorIndex
from pynwb.misc import Units

if TYPE_CHECKING:
    from pynwb import NWBFile

    from . import Session

__all__ = ["add_units_to_nwbfile"]

# `cluster_qc.tsv` columns (QcRow::METRICS in crates/sorrel-data/src/qc_export.rs).
# Amplitudes are the sorter's amplitudes.npy values as loaded: no gain is applied.
METRIC_DESCRIPTIONS = {
    "n_spikes": "Number of spikes assigned to the cluster after curation edits (count); 0 for an empty cluster.",
    "firing_rate": "Mean firing rate in Hz over the whole recording, n_spikes divided by n_samples / sample_rate; 0 when the recording has zero duration.",
    "mean_amplitude": "Mean per-spike amplitude in the sorter's amplitude units (raw ADC counts for int16 data, µV only when the input is calibrated); NaN when the cluster has no amplitudes.",
    "presence_ratio": "Fraction (0-1) of 50 equal-duration bins across the recording that contain at least one spike; 0 for an empty cluster.",
    "isi_violations": "Count of consecutive inter-spike intervals shorter than the 1.5 ms refractory window.",
    "isi_violation_ratio": "ISI violations per second of the cluster's own first-to-last spike span (a rate, not a fraction); 0 with fewer than 2 spikes.",
    "amplitude_cutoff": "Estimated fraction (0-0.5) of spikes missed below the detection threshold, from a 30-bin mirror-tail fit of the amplitude histogram; 0 with fewer than 8 spikes.",
    "amplitude_snr": "Absolute mean divided by the sample SD of per-spike amplitudes (dimensionless); 0 with fewer than 2 spikes or zero SD.",
    "drift_correlation": "Pearson correlation (-1 to 1) between spike time and per-spike amplitude; 0 with fewer than 3 spikes or constant values.",
    "longest_silent_gap": "Longest silent stretch, including before the first and after the last spike, as a fraction (0-1) of the recording duration; 1.0 for an empty cluster.",
    "composite_quality": "Geometric mean (0-1) of the contamination, presence, coverage, SNR, completeness and stability sub-scores, plus the isolation score when available.",
    "isolation_distance_sq": "Squared Mahalanobis distance to the n_in-th nearest background spike in a 3-PC subspace of template-relative channel 0; NaN under 50 cluster spikes, no PC features, or too few background spikes.",
    "l_ratio": "Mean chi-squared survival (df = 3 PCs) of background spikes' Mahalanobis squared distances divided by n_in; lower is better; NaN under 50 cluster spikes, no PC features, or ill-conditioned covariance.",
    "nn_isolation": "Fraction (0-1) of the 8 nearest neighbours, in a background-balanced PC pool, that belong to the cluster; NaN with fewer than 2 cluster spikes or no background.",
    "d_prime": "Standardized separation of cluster and background means along the Fisher LDA axis in PC space (dimensionless, about 4 or more is well isolated); NaN if either class has fewer than 2 spikes.",
    "silhouette": "Mean of (b - a) / max(a, b) over cluster spikes, with a and b the distances to the cluster and background centroids in PC space; range -1 to 1; NaN without both classes.",
    "trough_to_peak_ms": "Milliseconds from the trough (minimum) to the following peak (maximum) of the primary template on its peak channel; NaN without seeded templates.",
    "half_width_ms": "Width of the primary template's trough at half its depth, linearly interpolated, in milliseconds, on its peak channel; NaN without seeded templates.",
    "peak_trough_ratio": "Peak value after the trough divided by the absolute trough depth, for the primary template on its peak channel (dimensionless); NaN without seeded templates.",
    "repolarisation_slope": "(peak - trough) / trough_to_peak_ms for the primary template on its peak channel, in amplitude units per millisecond; 0 when the trace never recovers; NaN without templates.",
    "n_extrema": "Count of interior local extrema in the primary template's peak-channel trace with magnitude at least 20% of the peak-to-peak amplitude; 3 or more suggests a multi-peak waveform.",
    "x_um": "X coordinate (µm) of the template footprint centroid, weighted by per-channel peak-to-peak amplitude squared; NaN without probe geometry or seeded templates.",
    "y_um": "Y coordinate (µm) of the template footprint centroid, weighted by per-channel peak-to-peak amplitude squared; NaN without probe geometry or seeded templates.",
    "footprint_decay_um": "Decay length (µm) from a log-linear fit of per-channel peak-to-peak amplitude versus distance from the peak channel, over channels at 5%+ of the max; NaN without 3 such channels and a negative slope.",
    "footprint_r80_um": "Radius (µm) around the footprint centroid that encloses 80% of the summed squared per-channel peak-to-peak amplitudes; NaN without seeded templates or probe geometry.",
    "n_active_channels": "Count of channels whose peak-to-peak template amplitude is at least 20% of the maximum channel's; NaN without seeded templates or probe geometry.",
    "propagation_ms_per_100um": "Least-squares slope of per-channel trough time in ms versus distance from the peak channel over active channels, in ms per 100 µm; NaN with fewer than 3 active channels.",
    "burst_index": "Mean autocorrelogram count at 2-10 ms lags divided by the mean at 30-50 ms lags, from a 1 ms-bin, +/-50 ms ACG (dimensionless); NaN when the baseline is 0 or the ACG is too short.",
    "rp_estimate_ms": "Refractory window in milliseconds (0.5-10 ms, 20 steps) with the lowest Hill contamination estimate, ties resolving to the longest window; NaN with fewer than 2 spikes.",
    "rp_contamination": "Hill false-positive fraction (0-1) at rp_estimate_ms, the estimated share of spikes that belong to other units; NaN when no refractory window could be estimated.",
    "amplitude_dip": "Hartigan dip statistic (0-0.25) of up to 5000 evenly strided per-spike amplitudes; larger values indicate a second amplitude mode; NaN with fewer than 4 amplitudes.",
    "drift_slope": "Least-squares slope of per-spike amplitude versus time in seconds, in amplitude units per second; NaN with fewer than 2 spikes or when spike and amplitude counts differ.",
}
_INT_METRICS = {"n_spikes", "isi_violations"}


def add_units_to_nwbfile(
    session: Session | str | Path,
    nwbfile: NWBFile,
    *,
    starting_time: float = 0.0,
    description: str | None = None,
) -> Units:
    """Add the curated units of ``session`` to ``nwbfile`` as ``nwbfile.units``.

    One row per cluster that still has spikes, with the backend's unit id
    (as in ``curation.json``) as the row id, and these columns:
    ``spike_times``; ``quality`` (good, mua, noise or unsorted); ``origin``
    (``Auto[<sorter>]`` or ``Sorrel[<method>]``); ``parent_unit_ids``
    (ragged: the unit a split came from, or the units merged into this one;
    a parent can be absent from the table because it no longer has
    spikes); ``tags`` (ragged); and every ``cluster_qc.tsv`` metric that is
    not NaN for every unit. Curation provenance (Sorrel version, sorter,
    journal xxh3, edits by actor and method) goes into the table
    description, and ``("sorrel-ephys", <version>)`` into
    ``nwbfile.was_generated_by``.

    Timing contract: ``spike_times = sample_index / sample_rate +
    starting_time`` in seconds, where ``sample_index`` counts from the
    first sample of the recording the sort was run on, ``sample_rate`` is
    the exact rate in ``params.py`` (phy) or ``recording.json`` (a
    SortingAnalyzer), and ``Units.resolution = 1 / sample_rate``. Pass
    ``starting_time`` when that recording does not start at the NWB
    session start (the offset of its first sample, in seconds, as for an
    ``ElectricalSeries``). Multi-segment SortingAnalyzers are rejected:
    their sample indices restart per segment and the gaps are not known.

    Refuses a file that already has a units table rather than merging into
    rows another tool wrote. Returns the new table.
    """
    from . import Session, __version__

    if not isinstance(session, Session):
        session = Session.open(session)
    if nwbfile.units is not None:
        raise ValueError("nwbfile already has a units table; sorrel will not merge into it")
    rate = _exact_sample_rate(session)

    qc = {row["cluster_id"]: row for row in session.qc()}
    labels = session.labels()
    clusters = [c for c in range(session.n_clusters) if session.spike_times(c)]
    if not clusters:
        raise ValueError("the session has no units with spikes")
    metrics = [
        m for m in METRIC_DESCRIPTIONS if any(qc[c].get(m) is not None for c in clusters)
    ]

    prov = session.provenance()
    counts = lambda d: ", ".join(f"{k}={v}" for k, v in sorted(d.items())) or "none"
    summary = (
        f"Units from {session.sorter()} spike sorting, curated in Sorrel {__version__}. "
        f"Curation provenance: journal_xxh3={prov['journal_xxh3']}, "
        f"head_sealed={str(prov['sealed']).lower()}, "
        f"edits on the applied path={prov['n_applied']} of {prov['n_nodes']} in the history; "
        f"by actor: {counts(prov['by_actor'])}; by method: {counts(prov['by_method'])}; "
        f"pure_manual={str(prov['pure_manual']).lower()}, "
        f"machine_anchored={str(prov['anchored']).lower()}."
    )
    # Every column is built typed up front: hdmf cannot write a ragged
    # column whose rows are all empty (a session without merges or splits
    # has no parents), and add_unit turns appended ints into floats.
    def ragged(name: str, rows: list, dtype: Any, doc: str) -> list:
        data = VectorData(name=name, description=doc,
                          data=np.array([x for r in rows for x in r], dtype=dtype))
        offsets = np.cumsum([len(r) for r in rows], dtype=np.int64)
        return [data, VectorIndex(name=f"{name}_index", data=offsets, target=data)]

    rows = [qc[c] for c in clusters]
    columns = [
        *ragged(
            "spike_times",
            [np.asarray(session.spike_times(c), dtype=np.float64) / rate + starting_time
             for c in clusters],
            np.float64,
            "Spike times in seconds: sample_index / sample_rate + starting_time.",
        ),
        VectorData(name="quality", description="Sorrel curation label: good, mua, noise or unsorted.",
                   data=[labels[c] for c in clusters]),
        VectorData(
            name="origin",
            description="How the unit was made: Auto[<sorter>] if the sorter produced it, "
            "Sorrel[<method>] if a Sorrel split created it (Manual, Algo:<name>, LLM:<name>).",
            data=[session.origin(c) for c in clusters],
        ),
        *ragged(
            "parent_unit_ids",
            [[session.unit_id(p) for p in session.parents(c)] for c in clusters],
            np.int64,
            "Unit ids this unit came from in Sorrel: the unit it was split from, or the "
            "units merged into it. A parent with no spikes left has no row in this table.",
        ),
        *ragged(
            "tags",
            [session.tags(c) for c in clusters],
            str,
            "Sorrel tags: src:* is who last touched the unit, review:* asks for review, "
            "split/tuned mark structural edits.",
        ),
        *(
            VectorData(name=m, description=f"Sorrel QC metric. {METRIC_DESCRIPTIONS[m]}",
                       data=[_metric(m, r.get(m)) for r in rows])
            for m in metrics
        ),
    ]
    units = Units(
        name="units",
        description=f"{description} {summary}" if description else summary,
        resolution=1.0 / rate,
        id=[session.unit_id(c) for c in clusters],
        columns=columns,
    )
    nwbfile.units = units

    entry = ("sorrel-ephys", __version__)
    if nwbfile.was_generated_by is None:
        nwbfile.was_generated_by = [entry]
    elif isinstance(nwbfile.was_generated_by, list) and entry not in nwbfile.was_generated_by:
        nwbfile.was_generated_by.append(entry)
    return units


def _metric(name: str, v: Any) -> Any:
    if name in _INT_METRICS:
        return int(v or 0)
    return math.nan if v is None else float(v)


def _exact_sample_rate(session: Session) -> float:
    """The rate the backend files state, in double precision.

    ``Session.sample_rate`` is single precision; at 30 kHz that is up to
    ~3e-8 relative error, a few samples after an hour, so the timestamps use
    the files' value and only check it against the session's.
    """
    root = session.path
    if root is None:
        raise ValueError("session has no dataset path; open it with Session.open")
    params = root / "params.py"
    if params.is_file():
        rate = _params_sample_rate(params)
    else:
        rate = _analyzer_sample_rate(root)
    if np.float32(rate) != np.float32(session.sample_rate):
        raise ValueError(
            f"{root}: files give sample_rate {rate}, the session uses {session.sample_rate}"
        )
    return rate


def _params_sample_rate(params: Path) -> float:
    for node in ast.parse(params.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "sample_rate" for t in node.targets
        ):
            return float(ast.literal_eval(node.value))
    raise ValueError(f"{params}: no sample_rate")


def _analyzer_sample_rate(root: Path) -> float:
    segments = []
    info = root / "sorting" / "numpysorting_info.json"
    if info.is_file():
        segments.append(json.loads(info.read_text()).get("num_segments", 1))
    attrs = root / "recording_info" / "recording_attributes.json"
    if attrs.is_file():
        segments.append(len(json.loads(attrs.read_text()).get("num_samples", [0])))
    if any(n != 1 for n in segments):
        raise ValueError(
            f"{root}: multi-segment recording ({max(segments)} segments); "
            "sorrel cannot place their spikes on one time axis"
        )
    # The same files, in the same order, the native reader takes the rate from.
    for name in ("recording.json", "binary.json", "recording/recording.json"):
        p = root / name
        if p.is_file():
            v = json.loads(p.read_text())
            return float(v.get("kwargs", v)["sampling_frequency"])
    raise ValueError(f"{root}: no recording.json with sampling_frequency")

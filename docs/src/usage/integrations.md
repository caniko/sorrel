# Integrations

Sorrel speaks the phy2 directory format natively. Most spike-sorting
pipelines either emit phy directly or have a one-line export step that
does. This page lists the integration paths.

## SpikeInterface

[SpikeInterface] (SI) is the dominant Python framework for spike-sorting
workflows. Two ways to bring an SI run into Sorrel:

### 1. Phy export (works today, no extra deps)

```python
import spikeinterface.exporters as siex
siex.export_to_phy(
    sorting_analyzer,        # SortingAnalyzer with templates + waveforms
    output_folder="my_run/phy",
    copy_binary=True,        # so sorrel can mmap recording.dat in place
    compute_pc_features=True # enables the FeatureView
)
```

Then:

```sh
sorrel my_run/phy
```

Sorrel reads `spike_times.npy`, `spike_clusters.npy`, `params.py`,
`channel_positions.npy`, `templates.npy`, `pc_features.npy`,
`similar_templates.npy`, and `cluster_group.tsv` exactly as phy does.

### 2. Native SortingAnalyzer (binary folder)

If the analyzer was saved with `format='binary_folder'` and the sorting
was flattened to `spike_times.npy` + `unit_indices.npy` under
`<root>/sorting/`, point Sorrel directly at the analyzer root:

```sh
sorrel my_run/sorting_analyzer
```

The current implementation reads `recording.json` (sample rate, channel
count, dtype, bin path), the flat sorting arrays, the optional
`probe.json` for geometry, and `extensions/quality_metrics/metrics.csv`
into the cluster table. The Zarr (`zarr_folder`) variant is not
supported; export to `binary_folder` first.

### Round-tripping curation back into SI

Sorrel writes the same two files SI's `read_phy()` reads:

- `spike_clusters.npy` — updated per-spike cluster assignment after
  merges/splits.
- `cluster_group.tsv` — `good`/`mua`/`noise`/`unsorted` labels.

```python
import spikeinterface.extractors as se
sorting = se.read_phy("my_run/phy")  # picks up sorrel's edits
```

## Quality metrics

Sorrel auto-loads sidecar metrics at startup:

- SI's `quality_metrics.csv` (one row per cluster, header columns become
  metric names).
- Phy's `cluster_<metric>.tsv` files (one file per metric).

Metrics surface as columns in the cluster table. NaN means the upstream
tool didn't report a value for that cluster.

## Probe geometry

If the directory contains a `probe.json` or `probegroup.json` in
[probeinterface] format, Sorrel will use it for the WaveformView layout.
Phy's `channel_positions.npy` still wins when both are present (it's
authoritative for the row order in the raw bin).

## Recording-only formats

These are *recording* formats — they don't carry a sorting on their own,
but Sorrel can use them as the raw `.dat` for an existing phy directory:

| Format          | What we read                               | Use as `--dat` |
|-----------------|--------------------------------------------|----------------|
| **SpikeGLX**    | `<run>.<stream>.bin` + `<run>.<stream>.meta` | yes |
| **Open Ephys binary** | `continuous.dat` + `structure.oebin` | yes |
| **Mountainsort MDA**  | `*.mda` (header-prefixed flat array) | yes |

For SpikeGLX or Open Ephys, point `--dat` at the bin file and pass
`--sample-rate`/`--channels` from the `.meta`/`.oebin` (or let Sorrel
auto-detect — see [Backend detection](#backend-detection)).

## NWB / DANDI

```sh
sorrel my_session.nwb            # build with --features hdf5
```

Reads `/units/spike_times` (vlen) bucketed by `/units/id`, picks the first
`ElectricalSeries` under `/acquisition` for the trace, and pulls
`x`/`y`/`rel_x`/`rel_y` from `/general/extracellular_ephys/electrodes` for
probe geometry. Spike times are converted from seconds to samples using
the `ElectricalSeries` rate.

Limitations:

- HDF5 backend is opt-in (`cargo build --features hdf5`); the default
  build does not link libhdf5.
- Trace data is loaded eagerly into RAM (NWB datasets are typically
  chunked + compressed, no mmap).
- Local files only — DANDI streaming is not yet wired up.

## Kilosort 4 `rez.mat`

```sh
sorrel rez.mat --dat path/to/recording.bin   # build with --features hdf5
```

Reads `rez/st3` for sample/template tuples, `rez/ops/fs` and
`rez/ops/Nchan` for recording params, `rez/xc`/`rez/yc` for channel
positions, and `rez/ops/fbinary`/`fproc` for the dat path (overridable
with `--dat`).

If you'd rather not enable the `hdf5` feature, KS4 writes a phy export
under `<output>/phy/` automatically — open that directory instead.

## Python (Jupyter)

The `sorrel-py` crate (build with `maturin develop` from
`crates/sorrel-py/`) exposes:

```python
import sorrel
sorrel.open(recording, sorting)  # SpikeInterface objects
```

Internally this delegates to `export_to_phy` followed by the native
backend. Zero-copy direct-numpy is on the roadmap but not the v1 path.

## Backend detection

`sorrel <DIR>` walks a small detector list:

1. If `spike_times.npy` + `spike_clusters.npy` are present → phy/Kilosort.
2. Else if `recording.json` + `sorting/` are present → SortingAnalyzer.
3. Else fail with a helpful message.

Override with `--backend kilosort|sorting-analyzer`.

[SpikeInterface]: https://spikeinterface.readthedocs.io
[probeinterface]: https://github.com/SpikeInterface/probeinterface

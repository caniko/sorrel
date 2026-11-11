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
sorrel open my_run/phy
```

Sorrel reads `spike_times.npy`, `spike_clusters.npy`, `params.py`,
`channel_positions.npy`, `templates.npy`, `pc_features.npy`,
`similar_templates.npy`, and `cluster_group.tsv` exactly as phy does.

### 2. Native SortingAnalyzer (binary folder)

If the analyzer was saved with `format='binary_folder'` and the sorting
was flattened to `spike_times.npy` + `unit_indices.npy` under
`<root>/sorting/`, point Sorrel directly at the analyzer root:

```sh
sorrel open my_run/sorting_analyzer
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

These are _recording_ formats — they don't carry a sorting on their own,
but Sorrel can use them as the raw `.dat` for an existing phy directory:

| Format                | What we read                                 | Use as `--dat` |
| --------------------- | -------------------------------------------- | -------------- |
| **SpikeGLX**          | `<run>.<stream>.bin` + `<run>.<stream>.meta` | yes            |
| **Open Ephys binary** | `continuous.dat` + `structure.oebin`         | yes            |
| **Mountainsort MDA**  | `*.mda` (header-prefixed flat array)         | yes            |
| **mtscomp (IBL)**     | `<run>.cbin` + `<run>.ch` (zlib chunks)      | yes            |

For SpikeGLX or Open Ephys, point `--dat` at the bin file and pass
`--sample-rate`/`--channels` from the `.meta`/`.oebin` (or let Sorrel
auto-detect — see [Backend detection](#backend-detection)).

## NWB

```sh
sorrel open my_session.nwb            # build with --features hdf5
```

What Sorrel reads:

- **Units.** `/units/spike_times` and `spike_times_index`, one unit per
  row. The row's `/units/id` is the unit id Sorrel reports in
  `curation.json` and MCP evidence, so a table with ids `5, 7, 9` stays
  `5, 7, 9`. Units with no spikes are kept. Spike times (seconds) become
  sample indices as `round((t - starting_time) × rate)`.
- **Trace.** The first group under `/acquisition` whose `neurodata_type`
  is `ElectricalSeries`, or an extension subtype declared in the file's
  cached `/specifications`, with 2-D `data`. A file with none is rejected.
  If there are several, Sorrel reads the first by name and logs the rest.
- **Time base.** `starting_time` and its `rate`. A series with per-sample
  `timestamps` is rejected rather than given an assumed rate.
- **Units of the samples.** int16, uint16, int32 and float32 `data` are
  read in their stored type and converted to microvolts:
  `data × conversion × channel_conversion[ch] + offset`, times 10⁶.
  Other dtypes (float64, int64, …) are rejected. Amplitudes from an NWB
  file are therefore in µV.
- **Channels.** The series' `electrodes` region maps each data column to
  an electrodes-table row. That row's `id` becomes the channel id and its
  `rel_x`/`rel_y` the probe position. `x`/`y`/`z` are brain coordinates
  and are not used as probe geometry; without `rel_x`/`rel_y` the session
  has no channel positions.

Limitations:

- HDF5 backend is opt-in (`cargo build --features hdf5`); the default
  build and the release binaries do not link libhdf5.
- Trace data is loaded eagerly into RAM as float32 (NWB datasets are
  typically chunked and compressed, so there is no mmap).
- One `ElectricalSeries`; multi-segment recordings are not modelled.
- Local files only; DANDI streaming is not wired up. Sorrel does not write
  NWB.

The test fixtures in `release/fixtures/nwb/` are written by PyNWB
(`generate.py` pins the version).

## Kilosort 4 `rez.mat`

```sh
sorrel open rez.mat --dat path/to/recording.bin   # build with --features hdf5
```

Reads `rez/st3` for sample/template tuples, `rez/ops/fs` and
`rez/ops/Nchan` for recording params, `rez/xc`/`rez/yc` for channel
positions, and `rez/ops/fbinary`/`fproc` for the dat path (overridable
with `--dat`).

If you'd rather not enable the `hdf5` feature, KS4 writes a phy export
under `<output>/phy/` automatically — open that directory instead.

## Python (Jupyter)

Install the `sorrel` Python package from the wheel attached to each
GitHub release (see [Installation](../getting-started/installation.md#python));
it bundles the GUI binary. `maturin develop` from `crates/sorrel-py/`
builds it from source. The package has two layers.

**In-process, no window** — the notebook and pipeline API. It opens the
same journal the GUI uses, so edits made here show up in the GUI and
vice versa:

```python
import sorrel
s = sorrel.Session.open("my_run/phy")        # or a SortingAnalyzer binary_folder
s.qc()                                       # list of dicts, cluster_qc.tsv columns
s.suggest()                                  # {"merges": [...], "splits": [...]}
s.set_label(3, "good"); s.merge([4, 5], 6)   # journaled, undoable
new = s.split(0, [1, 7, 9])                  # local spike indices → new cluster id
s.save()                                     # phy files + curation.json
```

**GUI launcher** — subprocesses the `sorrel` binary (egui needs the main
thread, which a Jupyter kernel cannot give it):

```python
sorrel.open(recording, sorting)         # builds a SortingAnalyzer, saves it as binary_folder
result_dir = sorrel.open_analyzer(sorting_analyzer)
```

The directory it returns keeps the curation journal and, once you save
in the GUI, `curation.json`; round-trip into SpikeInterface with
`sc.apply_curation(analyzer, sorrel.read_curation(result_dir))`. The
temporary directory is kept by default so that file survives; pass
`keep_temp=False` to discard it.

## Backend detection

`sorrel open <DIR>` walks a small detector list:

1. If `spike_times.npy` + `spike_clusters.npy` are present → phy/Kilosort.
2. Else if `recording.json` + `sorting/` are present → SortingAnalyzer.
3. Else fail with a helpful message.

Override with `--backend kilosort|sorting-analyzer`.

[SpikeInterface]: https://spikeinterface.readthedocs.io
[probeinterface]: https://github.com/SpikeInterface/probeinterface

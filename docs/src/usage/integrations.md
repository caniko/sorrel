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
count, dtype, bin path), the flat sorting arrays, the unit ids
(`sorting/numpysorting_info.json`, or `sorting/unit_ids.npy` from older
layouts), the optional `probe.json` for geometry, and
`extensions/quality_metrics/metrics.csv` into the cluster table.
`recording.json` may describe a raw binary file (a recording from
`spikeinterface.read_binary`) or a SpikeInterface binary folder
(`rec.save(folder=..., format='binary')`, a `BinaryFolderRecording`): for
the latter Sorrel follows its `folder_path` to that folder's `binary.json`.
Either may sit under a preprocessing chain that keeps every channel and
sample (bandpass, highpass or notch filters, common reference, whitening,
phase shift, scaling, centering, z-scoring, highpass spatial filter).
Sorrel reads the binary source beneath it and does not re-apply the chain:
traces show the source samples through Sorrel's own highpass. A chain that
slices channels or frames, resamples or aggregates fails with an error
naming the class. Zarr is not read: a Zarr analyzer converts with
`analyzer.save_as(format="binary_folder", folder=...)`, and a Zarr
recording needs `recording.save(format="binary", folder=...)` and a new
analyzer on it (`save_as` keeps the Zarr recording).

### Round-tripping curation back into SI

Sorrel writes the same two files SI's `read_phy()` reads:

- `spike_clusters.npy` — updated per-spike cluster assignment after
  merges/splits.
- `cluster_group.tsv` — `good`/`mua`/`noise`/`unsorted` labels.

```python
import spikeinterface.extractors as se
sorting = se.read_phy("my_run/phy")  # picks up sorrel's edits
```

It also writes `curation.json` for SI's `apply_curation`. A session with
merges or splits is SI's list form (`SequentialCuration`): structural
steps, then one labels-only step. That expresses histories a single pass
cannot — a unit both merged and split, a repeated split, edits of units
Sorrel created — and the round trip is one call:

```python
import sorrel
import spikeinterface.curation as sc
curated = sc.apply_curation(analyzer, sorrel.read_curation(result_dir))
```

`read_curation` returns the list as-is; a labels-only session, and a phy
directory saved onto itself, stay one `Curation` dict. Ids are SI's own: a
merge keeps its target's id, a split's child keeps the id Sorrel gave it
and the spikes left behind get a fresh one. A merge into a unit an earlier
merge emptied brings back the target's id; when it has a single source
there is nothing for SI to merge, so that unit keeps the source's SI id. A rescue (`AddSpikes`) has no
SI curation slot, so its document lists the edits under
`sorrel_unrepresentable` and `read_curation` raises instead of silently
dropping spikes.

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

### Writing curated units to NWB

`sorrel.nwb.add_units_to_nwbfile` writes a curated session (phy directory
or SortingAnalyzer folder) into an `NWBFile` as its Units table. It needs
PyNWB: `pip install "sorrel-ephys[nwb]"`.

```python
from datetime import datetime, timezone
from pynwb import NWBFile, NWBHDF5IO
from sorrel.nwb import add_units_to_nwbfile

nwbfile = NWBFile("session", "id-1", datetime(2026, 1, 1, tzinfo=timezone.utc))
add_units_to_nwbfile("my_run/phy", nwbfile, starting_time=0.0)
with NWBHDF5IO("session.nwb", "w") as io:
    io.write(nwbfile)
```

Each cluster that still has spikes gets one row. The row id is the
backend's integer unit id, the one `curation.json` uses, with one
exception: SpikeInterface gives the spikes a split leaves behind a fresh
id, while Sorrel and this table keep the parent's id (likewise a
single-source merge into an emptied unit keeps the source's id in
SpikeInterface). When a
SortingAnalyzer's ids are strings that are not distinct integers
(`"unit_a"`, `"007"` beside `"7"`), rows are numbered densely and a
`unit_name` column holds SpikeInterface's ids. The columns are:

| column            | content                                                                     |
| ----------------- | --------------------------------------------------------------------------- |
| `spike_times`     | seconds, see the timing contract below                                      |
| `quality`         | `good`, `mua`, `noise` or `unsorted`                                        |
| `origin`          | `Auto[<sorter>]`, or `Sorrel[<method>]` for a unit a split created          |
| `parent_unit_ids` | ragged: the unit a split came from, or the units merged into this one       |
| `tags`            | ragged: Sorrel's tags (`src:*`, `review:*`, `split`, `tuned`)               |
| QC metrics        | every `cluster_qc.tsv` column that is not NaN for all units, with its units |

A parent id can be missing from the table when that unit has no spikes
left, e.g. a unit merged into another. That is why the column holds plain
ids and not a `DynamicTableRegion`.

**Timing contract.** `spike_times = sample_index / sample_rate +
starting_time`. `sample_index` counts from the first sample of the
recording the sort ran on. `sample_rate` is the exact rate from
`params.py` or `recording.json`, not Sorrel's single-precision copy.
`Units.resolution` is `1 / sample_rate`. Pass `starting_time` (seconds)
when that recording does not start at the NWB session start. The
recording's `ElectricalSeries` needs the same offset. Multi-segment
SortingAnalyzers are rejected: their sample indices restart per segment.

**Provenance.** The table's `description` records the Sorrel version, the
sorter, the journal's xxh3 hash, whether the saved state is sealed, and the
edits on the applied path counted by actor (human, agent, algo, remote) and
by method (manual, algo, llm). `NWBFile.was_generated_by` gets
`("sorrel-ephys", <version>)`. NWB has no core slot for curation
provenance. The other options were each worse:

- a one-row `DynamicTable` trips NWB Inspector's `check_single_row`;
- JSON in a column trips `check_table_values_for_dict`;
- `LabMetaData` needs an NDX extension;
- the schema says `scratch` data "is not intended for sharing".

The function refuses an `NWBFile` that already has a units table. It does
not write traces, the electrodes table or waveforms. Use the recording's
own NeuroConv interface for those.

### NeuroConv

`sorrel.neuroconv.SorrelSortingInterface` wraps the same function as a
NeuroConv data interface (`pip install "sorrel-ephys[neuroconv]"`), so a
`-to-nwb` converter can put Sorrel's curated units next to its recording
interfaces:

```python
from neuroconv import ConverterPipe
from neuroconv.datainterfaces import SpikeGLXRecordingInterface
from sorrel.neuroconv import SorrelSortingInterface

sorting = SorrelSortingInterface(folder_path="my_run/phy")
sorting.set_aligned_starting_time(0.0)  # offset of the sorted recording
converter = ConverterPipe(data_interfaces={
    "recording": SpikeGLXRecordingInterface(file_path="run_g0_t0.imec0.ap.bin"),
    "sorting": sorting,
})
```

`add_to_nwbfile` takes `units_description`, which goes in front of the
provenance summary. Opening the folder takes the curation journal's
writer lock, so close the Sorrel window on it first.

The interface does not go through NeuroConv's `add_sorting_to_nwbfile`,
for three reasons:

- that writer concatenates the segments of a multi-segment sorting into
  one spike train, where Sorrel refuses to guess;
- it renumbers rows `0..N-1` and moves the sorter's ids into `unit_name`,
  where `parent_unit_ids` and `curation.json` need the ids themselves;
- it would make SpikeInterface a dependency of the NWB path.

The interface does not link units to electrodes.

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

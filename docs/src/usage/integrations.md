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
sorrel open my_session.nwb  # build with --features hdf5
sorrel open my_session.nwb --nwb-series ElectricalSeriesRaw
sorrel open my_session.nwb --nwb-series /processing/ecephys/FilteredEphys/Processed
sorrel open sorted_units.nwb --dat recording.nwb
sorrel open sorted_units.nwb --sample-rate 30000  # if no resolution is recorded
```

What Sorrel reads:

- **Units.** `/units/spike_times` and `spike_times_index`, one unit per
  row. Units with no spikes are kept. `/units/id` supplies the numeric unit
  ID; a unique `unit_name` string column preserves the upstream names in
  `curation.json`. Sorrel reads `good`, `mua`, `noise` and `unsorted` from
  `label`, `quality` or `KSLabel`, in that order. Sorrel's Python-exported
  origin, tags and parent IDs are restored. Parents without a table row are
  displayed as `unit:<id>` instead of local cluster IDs. Columns marked `Sorrel QC metric.` populate the
  cluster table, just like sidecar metrics. Unrecognized columns are ignored.
- **Trace.** Sorrel reads single windows on demand and caches recent blocks
  instead of loading the entire trace. It looks for `ElectricalSeries` (or an
  extension subtype declared in `/specifications`) under `/acquisition` and
  `/processing`, with 2-D `data` (time × channels) or 1-D `data` (one channel).
  Conflicting cached type definitions that affect series selection are refused.
  By default it prefers 2-D series, then takes the first series by path
  (acquisitions before processed series), and logs the available paths when there is
  more than one. Linked series and linked datasets used by the reader are
  refused; open the file or store that holds the data directly. `--nwb-series`
  takes an acquisition name or an absolute path in the NWB file, not a
  filesystem path. Use a separate `--journal` when
  curating against another series or a different paired recording. To reuse
  an open tab or attach `call` or `mcp` to its host, pass the same open options;
  conflicting options are refused rather than silently ignored.
- **Time base.** Spike times on the NWB session clock become sample indices as
  `round((t - starting_time) × rate)`. Sorrel uses the series rate by default.
  If `1 / spike_times.resolution` is within one part per million of that rate,
  it also declares a Units clock. If the two clocks place any spike on
  different samples, pass `--sample-rate <Hz>` to choose one. The value must
  equal a declared clock exactly, including the full value of a stored
  float32 rate widened to float64. Copy one of the rates printed in the
  error; approximate values are refused. In a single file, an unspecified
  or more distant resolution is not a clock.
  With per-sample `timestamps`, the first timestamp gives the start and
  the end-to-end mean interval gives the series rate.
  Timestamps must strictly increase and stay within 0.001 samples of the
  inferred grid, plus four floating-point steps at the largest absolute
  timestamp (converted to samples) to allow for rounding at large clock origins.
  Irregular timestamps are refused.
  NWB trials used as events are shifted by the same start time; CSV/NPY event
  times are already relative to the recording.
- **Sample types.** int8, uint8, int16, uint16, int32, uint32, int64, float32
  and float64 are read in their stored type. Sorrel converts to float32
  microvolts as `(data × conversion × channel_conversion[ch] + offset) × 10⁶`.
  Other dtypes are refused.
- **Channels.** The series' `electrodes` region maps each data column to
  an electrodes-table row. That row's `id` becomes the channel id and its
  `rel_x`/`rel_y` the probe position. `x`/`y`/`z` are brain coordinates
  and are not used as probe geometry; without `rel_x`/`rel_y` the session
  has no channel positions.

**Units without a trace.** Without an ElectricalSeries, Sorrel takes the
rate from `1 / spike_times.resolution`. If that resolution is unspecified
and no raw NWB is paired, supply the rate with `--sample-rate HZ` instead.
Spike times are relative to zero
(`timestamps_reference_time`).
The electrodes table supplies channels if present, but there is no raw trace
or waveform extraction. With `--dat recording.nwb`, the raw file supplies
the trace, channels and time base; `--nwb-series` selects from that file.
If a Units rate is declared, it must be within one part per million of the
raw series rate. The same exact clock choice and spike-index check apply.
Spike indices must fit
inside the raw series. The files must have the same `timestamps_reference_time`;
Sorrel does not infer an offset between files.

The reader needs an NWB with `/units`. HDF5 files require `cargo build
--features hdf5`. One series is read per session; multi-segment stitching is not
supported.

- Timestamps with float64 spacing greater than 0.01 samples are refused; if sampling was regular, rewrite the series with `starting_time` and `rate`.
- With per-sample `timestamps`, each spike must map to a sample inside the series whose recorded timestamp is nearest to the spike time (ties are allowed). Sorrel refuses the file if the inferred grid disagrees.
- The selected clock (`starting_time` with its `rate`, or `timestamps`) and `/units/id`, `/units/spike_times`, and `/units/spike_times_index` must be stored in their respective opened files or stores. For HDF5, every path component must be a hard link; virtual datasets and external raw storage are refused. Open the file holding the data directly, or pass the recording file via `--dat`.

**Zarr.** `sorrel open session.nwb.zarr` reads local NWB-Zarr v2 directories
written by `NWBZarrIO`. Arrays must use C order and a defined fill value. Supported
compressors are Blosc (default LZ4), zlib and gzip; uncompressed chunks are also
supported. Sorrel refuses filters other than variable-length string encodings
and the JSON encoding used for cached NWB specifications. If `.zmetadata` is
present, it must match the individual metadata files. The
`zarr` feature is on by default in `sorrel`. `--dat` can pair an HDF5 file with
a Zarr store in either direction when both features are built. The same exact
clock rules apply: Zarr clock attributes are finite JSON numbers read as
float64, and clock arrays must use little-endian `<f4` or `<f8`. External links and references are not followed;
even same-store `zarr_link` aliases are refused for the clock and spike-mapping
columns, like HDF5 soft links. Pickled columns are not decoded. Journals live
next to the store. Its seal hashes metadata contents and the paths, sizes and
modification times of chunk files, without reading trace chunks at open.
An edit that deliberately preserves chunk size and mtime is not detected.

The test fixtures in `release/fixtures/nwb/` are written by PyNWB and hdmf-zarr
(`generate.py` pins both versions).

### DANDI streaming

```sh
# This units-only example needs an explicit sample grid; it has no recording clock.
sorrel open --sample-rate 1000 'dandi://dandi/000006@0.220126.1855/sub-anm369962/sub-anm369962_ses-20170316.nwb'
sorrel describe --sample-rate 1000 'https://api.dandiarchive.org/api/assets/d97954f2-6d80-4dda-9ecf-89c7c10fed0b/download/'
```

Use `dandi://dandi/<dandiset>[@<version>]/<exact asset path>`. If you omit the
version, Sorrel selects the latest published version, or `draft` if there is
none. Prefer an explicit version for reproducibility. For embargoed data, set
`DANDI_API_KEY` in the Sorrel process that opens the session (including an
already-running GUI if it receives `sorrel open`). The key is used only for
requests to the DANDI API, not to storage. Library consumers should cap
`ureq`, `ureq_proto`, and `rustls` logging at `warn` or turn it off to keep
signed URLs out of logs.

Streaming is read-only. Sorrel loads metadata and sorting data, then fetches
trace windows on demand. Direct HTTPS HDF5 URLs must support range GETs and
provide a strong ETag. For Zarr, use a DANDI locator or DANDI API asset download
URL: a bare storage prefix doesn't provide the required object list. The same
NWBs and series that local readers refuse are refused remotely. The `dandi`
feature is on by default; HDF5 streaming also needs `hdf5`.

Sessions with DANDI inputs always open fresh without reusing a tab or session
host. A session can't combine two distinct draft assets; this also applies to
DANDI asset-ID download URLs without a version.
If a running session owns the journal, stop it or pass a different `--journal`.

The default journal and save directory is local, under the Sorrel user data
directory (`$XDG_DATA_HOME/sorrel` or `~/.local/share/sorrel` on Linux), in
`dandi/<dandiset>/<version>/<path hash>/`. Direct HTTPS URLs are stored under
`dandi/https/<URL hash>/`, without the signed query. Use `--journal` or the
command's `--out` to choose a different location. `export-phy`, `qc`, and
`report` default to this directory for remote sessions. An asset whose ETags
or sizes have changed does not match the old journal seal.

**Limits:** As in any local session, local files are fingerprinted by name,
size, and modification time; a replacement with the same name, size, and
modification time is not detected.

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

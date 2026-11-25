# Backends & Inputs

Sorrel reads several sorting and recording formats. The general rule:

- **Sorting backends** carry both a sorting and (optionally) the raw trace.
- **Recording-only formats** plug into a sorting via `--dat` /
  `--spikeglx-meta` / `--oebin`.

| Kind      | Format                                                   | Notes                                                                                                                                                                                                         |
| --------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sorting   | phy / Kilosort directory                                 | Reads Phy arrays as-is, including Kilosort 4's native `.npy` output.                                                                                                                                          |
| Sorting   | SpikeInterface `SortingAnalyzer` (binary folder)         | Reads `recording.json` (binary, or a channel- and sample-preserving preprocessing chain over binary), the flat sorting under `sorting/`, optional `probe.json`, and `extensions/quality_metrics/metrics.csv`. |
| Sorting   | NWB (local HDF5)                                         | HDF5 is bundled by default. Reads `/units`, with or without a trace. `--nwb-series` selects an acquisition or processed series; `--dat` pairs a separate raw NWB. See [NWB](integrations.md#nwb).             |
| Sorting   | NWB-Zarr (local Zarr v2 directory)                       | Default `zarr` feature. Same units and lazy trace reader as HDF5 NWB; `--dat` can mix storage formats. See [NWB](integrations.md#nwb).                                                                        |
| Sorting   | DANDI NWB / NWB-Zarr (remote)                            | On by default. Read-only range streaming with local journals. See [DANDI streaming](integrations.md#dandi-streaming).                                                                                         |
| Sorting   | Legacy MATLAB `rez.mat` / `rez2.mat`                     | MATLAB v7.3 (HDF5) files from Kilosort 2, 2.5, and 3. See [Legacy MATLAB `rez.mat`](integrations.md#legacy-matlab-rezmat).                                                                                    |
| Spikes    | Neuralynx tetrode `.ntt` file                            | Stored spike snippets on four channels, no continuous recording. See [Neuralynx tetrode files](#neuralynx-tetrode-files).                                                                                     |
| Recording | SpikeGLX (`*.bin` + `*.meta`)                            | Use as `--dat`; pass `--spikeglx-meta` for sample rate / channels / dtype.                                                                                                                                    |
| Recording | Open Ephys binary (`continuous.dat` + `structure.oebin`) | Use as `--dat`; pass `--oebin` to inherit metadata.                                                                                                                                                           |
| Recording | Mountainsort MDA (`*.mda`)                               | Use as `--dat`. Reads the MDA header.                                                                                                                                                                         |

## Backend detection

When `--backend` is not specified, the detector is run on the input path:

1. File path `*.nwb` → NWB (HDF5).
2. File path `rez.mat` / `rez2.mat` → legacy MATLAB reader (HDF5).
3. File path `*.ntt` (case-insensitive) → Neuralynx.
4. Directory containing `spike_times.npy` + `spike_clusters.npy` →
   phy / Kilosort.
5. Directory ending in `.nwb.zarr` or with `nwb_version` in root `.zattrs` → NWB (Zarr).
6. Directory containing `recording.json` (or `binary.json`) and a
   `sorting/` folder → SortingAnalyzer.

If none match, Sorrel exits with a message listing what it looked for.

## Neuralynx tetrode files

Open **one tetrode file at a time** with `sorrel open TT1.ntt`,
`sorrel run TT1.ntt`, or by dropping the file onto the Sorrel window. All
spikes start in cluster `0`, labelled `unsorted`; Sorrel does not import the
file's `CellNumber` assignments. Select cluster `0`, draw a lasso in
Features and press `K` to split the enclosed spikes into a new cluster.
The remainder keeps id `0`, ready for another cut.

Sorrel reads four channels and 32 samples per spike, in µV using
`ADBitVolts`. When `InputInverted` is true it negates the counts during
conversion. Sample zero on the waveform plot is the original alignment
point, usually 7 samples after the start of the snippet, with 24 after it.
Trace preprocessing settings do not change stored waveforms.

- The `.ntt` is not rewritten. The journal is `TT1.ntt.sorrel.journal`;
  `TT1.ntt.sorrel/` holds saved curation arrays, UI preferences, and
  `ntt_source.json`. Reopening replays the journal against the original file.
- Opening (including `sorrel describe`) creates the journal and artifact
  directory. The sidecar records the original header, calibration and
  alignment metadata, and the earliest hardware timestamp `t0_us`. That
  spike sits at session sample `pre`; the session origin is `t0_us` minus
  `pre / sample_rate` seconds (convert `t0_us` from µs first). Event times
  must be relative to that origin. Put `events.csv` in `TT1.ntt.sorrel/`
  or pass `--events`.
- Saved `spike_clusters.npy` rows follow time order. This is file order
  for a chronological file; for an out-of-order file, timestamps are
  sorted stably. If rounded sample indices collide, Sorrel shifts later
  indices forward to keep times unique. A run of collisions can shift a
  spike by more than one sample and affects ISIs and refractory estimates.
  The sidecar records `row_order`, `bumped_times` and `max_shift_samples`.
- There is no continuous trace or probe geometry; Similar has no
  templates to compare, and MCP `probe waveform` needs continuous data.
  Firing rates use the spike span with snippet padding, not the full
  acquisition duration. Features draws up to 2,500 points per cluster,
  but cuts test all projected spikes. With fewer than three spikes,
  there are no PCA points.
- A header-only file cannot be curated. A partial trailing record is
  ignored with a warning. `sorrel export-phy` writes assignment arrays
  with zero templates and no `recording.dat`; it does not convert the
  stored waveforms into phy templates. The `share`, `blind`, `auto`, and
  spike-rescue workflows are not supported for `.ntt`.

## Probe geometry

If a `probe.json` or `probegroup.json` ([probeinterface]) is present in
the directory, Sorrel uses it to lay out the WaveformView and ProbeView.
Phy's `channel_positions.npy` wins when both are present (it's
authoritative for the row order in the raw `.dat`).

## Quality metrics sidecars

Sorrel auto-loads cluster-level metrics at startup from any of:

- `quality_metrics.csv` (one row per cluster, header columns become
  metric names — SI's format).
- `cluster_<metric>.tsv` files (one file per metric — phy's format).
- `/units` columns marked as Sorrel QC metrics by the [Python NWB export](integrations.md#writing-curated-units-to-nwb).

NaNs mean the upstream tool didn't report a value for that cluster.

[probeinterface]: https://github.com/SpikeInterface/probeinterface

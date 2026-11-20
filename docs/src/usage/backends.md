# Backends & Inputs

Sorrel reads several sorting and recording formats. The general rule:

- **Sorting backends** carry both a sorting and (optionally) the raw trace.
- **Recording-only formats** plug into a sorting via `--dat` /
  `--spikeglx-meta` / `--oebin`.

| Kind      | Format                                                   | Notes                                                                                                                                                                                                         |
| --------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sorting   | phy / Kilosort directory                                 | First-class. Reads phy artefacts as-is.                                                                                                                                                                       |
| Sorting   | SpikeInterface `SortingAnalyzer` (binary folder)         | Reads `recording.json` (binary, or a channel- and sample-preserving preprocessing chain over binary), the flat sorting under `sorting/`, optional `probe.json`, and `extensions/quality_metrics/metrics.csv`. |
| Sorting   | NWB (local HDF5)                                         | HDF5 is bundled by default. Reads `/units`, with or without a trace. `--nwb-series` selects an acquisition or processed series; `--dat` pairs a separate raw NWB. See [NWB](integrations.md#nwb).             |
| Sorting   | NWB-Zarr (local Zarr v2 directory)                       | Default `zarr` feature. Same units and lazy trace reader as HDF5 NWB; `--dat` can mix storage formats. See [NWB](integrations.md#nwb).                                                                        |
| Sorting   | DANDI NWB / NWB-Zarr (remote)                            | On by default. Read-only range streaming with local journals. See [DANDI streaming](integrations.md#dandi-streaming).                                                                                         |
| Sorting   | Kilosort 4 `rez.mat`                                     | HDF5 is bundled by default. Reads `rez/st3`, `rez/ops/*`, `rez/xc`, `rez/yc`.                                                                                                                                 |
| Recording | SpikeGLX (`*.bin` + `*.meta`)                            | Use as `--dat`; pass `--spikeglx-meta` for sample rate / channels / dtype.                                                                                                                                    |
| Recording | Open Ephys binary (`continuous.dat` + `structure.oebin`) | Use as `--dat`; pass `--oebin` to inherit metadata.                                                                                                                                                           |
| Recording | Mountainsort MDA (`*.mda`)                               | Use as `--dat`. Reads the MDA header.                                                                                                                                                                         |

## Backend detection

When `--backend` is not specified, the detector is run on the input path:

1. File path `*.nwb` → NWB (HDF5).
2. File path `rez.mat` / `rez2.mat` → KS4 rez (HDF5).
3. Directory containing `spike_times.npy` + `spike_clusters.npy` →
   phy / Kilosort.
4. Directory ending in `.nwb.zarr` or with `nwb_version` in root `.zattrs` → NWB (Zarr).
5. Directory containing `recording.json` (or `binary.json`) and a
   `sorting/` folder → SortingAnalyzer.

If none match, Sorrel exits with a message listing what it looked for.

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

# Backends & Inputs

Sorrel reads several sorting and recording formats. The general rule:

- **Sorting backends** carry both a sorting and (optionally) the raw trace.
- **Recording-only formats** plug into a sorting via `--dat` /
  `--spikeglx-meta` / `--oebin`.

| Kind      | Format                                                   | Notes                                                                                                                           |
| --------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Sorting   | phy / Kilosort directory                                 | First-class. Reads phy artefacts as-is.                                                                                         |
| Sorting   | SpikeInterface `SortingAnalyzer` (binary folder)         | Reads `recording.json`, the flat sorting under `sorting/`, optional `probe.json`, and `extensions/quality_metrics/metrics.csv`. |
| Sorting   | NWB                                                      | Requires `--features hdf5`. Reads `/units/spike_times` + `/general/extracellular_ephys/electrodes`.                             |
| Sorting   | Kilosort 4 `rez.mat`                                     | Requires `--features hdf5`. Reads `rez/st3`, `rez/ops/*`, `rez/xc`, `rez/yc`.                                                   |
| Recording | SpikeGLX (`*.bin` + `*.meta`)                            | Use as `--dat`; pass `--spikeglx-meta` for sample rate / channels / dtype.                                                      |
| Recording | Open Ephys binary (`continuous.dat` + `structure.oebin`) | Use as `--dat`; pass `--oebin` to inherit metadata.                                                                             |
| Recording | Mountainsort MDA (`*.mda`)                               | Use as `--dat`. Reads the MDA header.                                                                                           |

## Backend detection

When `--backend` is not specified, the detector is run on the input path:

1. File path `*.nwb` → NWB (HDF5).
2. File path `rez.mat` / `rez2.mat` → KS4 rez (HDF5).
3. Directory containing `spike_times.npy` + `spike_clusters.npy` →
   phy / Kilosort.
4. Directory containing `recording.json` (or `binary.json`) and a
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

NaNs mean the upstream tool didn't report a value for that cluster.

[probeinterface]: https://github.com/SpikeInterface/probeinterface

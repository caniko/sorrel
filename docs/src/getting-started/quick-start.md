# Quick Start

Point Sorrel at a sorting output and it will auto-detect the backend:

```bash
sorrel open <DATA_DIR>
```

The detector looks for, in order:

1. `spike_times.npy` + `spike_clusters.npy` → **phy / Kilosort**.
2. `recording.json` (or `binary.json`) + `sorting/` → **SpikeInterface
   `SortingAnalyzer`**.
3. A `.nwb` file path → **NWB**.
4. A `rez.mat` / `rez2.mat` file path → **legacy MATLAB v7.3 (HDF5)**.

You can force the choice with `--backend kilosort | sorting-analyzer |
nwb | rez`. The older `ks4-rez` name is still accepted as an alias for `rez`.

## Phy / Kilosort

For Kilosort 4, open the results directory containing its native `.npy`
arrays.

```bash
sorrel open my_run/phy
```

Sample rate, channel count, dtype, header offset, and dat path are read
from `params.py` when present. CLI flags override on a per-field basis:

```bash
sorrel open my_run/phy \
  --dat recording.dat \
  --sample-rate 30000 \
  --channels 384 \
  --dtype int16
```

When the trace metadata isn't in `params.py`, point Sorrel at a SpikeGLX
or Open Ephys metadata file:

```bash
sorrel open my_run/phy --spikeglx-meta run.imec0.ap.meta
sorrel open my_run/phy --oebin Record\ Node\ 101/structure.oebin
```

## SpikeInterface SortingAnalyzer

```bash
sorrel open my_run/sorting_analyzer
```

The analyzer must be saved with `format='binary_folder'` and have a
flattened sorting under `<root>/sorting/`. A Zarr analyzer converts with
`analyzer.save_as(format="binary_folder", folder=...)`.

## NWB and legacy MATLAB `rez.mat`

Both readers are included by default. The `rez.mat` reader is for MATLAB
v7.3 (HDF5) files from Kilosort 2, 2.5, and 3.

```bash
sorrel open my_session.nwb
sorrel open rez.mat --dat path/to/recording.bin
```

## Curation journal

`--journal PATH` overrides the on-disk curation journal. The default is
`<root>/sorrel.journal`. The journal is replayed on every open, so reopening
the same directory restores all prior edits.

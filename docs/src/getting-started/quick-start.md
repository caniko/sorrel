# Quick Start

Point Sorrel at a sorting output and it will auto-detect the backend:

```bash
sorrel <DATA_DIR>
```

The detector looks for, in order:

1. `spike_times.npy` + `spike_clusters.npy` → **phy / Kilosort**.
2. `recording.json` (or `binary.json`) + `sorting/` → **SpikeInterface
   `SortingAnalyzer`**.
3. A `.nwb` file path → **NWB** (requires `--features hdf5`).
4. A `rez.mat` / `rez2.mat` file path → **Kilosort 4 rez** (requires
   `--features hdf5`).

You can force the choice with `--backend kilosort | sorting-analyzer |
nwb | ks4-rez`.

## Phy / Kilosort

```bash
sorrel my_run/phy
```

Sample rate, channel count, dtype, header offset, and dat path are read
from `params.py` when present. CLI flags override on a per-field basis:

```bash
sorrel my_run/phy \
  --dat recording.dat \
  --sample-rate 30000 \
  --channels 384 \
  --dtype int16
```

When the trace metadata isn't in `params.py`, point Sorrel at a SpikeGLX
or Open Ephys metadata file:

```bash
sorrel my_run/phy --spikeglx-meta run.imec0.ap.meta
sorrel my_run/phy --oebin Record\ Node\ 101/structure.oebin
```

## SpikeInterface SortingAnalyzer

```bash
sorrel my_run/sorting_analyzer
```

The analyzer must be saved with `format='binary_folder'` and have a
flattened sorting under `<root>/sorting/`. The Zarr variant is not yet
supported.

## NWB and KS4 rez

```bash
sorrel my_session.nwb                           # --features hdf5
sorrel rez.mat --dat path/to/recording.bin      # --features hdf5
```

## Curation journal

`--journal PATH` overrides the on-disk SQLite log. The default is
`<root>/sorrel.sqlite`. The journal is replayed on every open, so reopening
the same directory restores all prior edits.

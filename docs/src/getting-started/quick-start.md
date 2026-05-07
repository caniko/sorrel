# Quick Start

Run Sorrel against a Kilosort / phy2 output directory:

```bash
sorrel <KILOSORT_DIR> --dat recording.dat --sample-rate 30000 --channels 32
```

`--dat` defaults to `recording.dat` inside the Kilosort directory.

`--sample-rate` defaults to `30000`.

`--channels` defaults to `32`.

`--journal` defaults to `sorrel.sqlite` inside the Kilosort directory.

Sorrel detects the Kilosort backend when the directory contains both `spike_times.npy` and `spike_clusters.npy`.

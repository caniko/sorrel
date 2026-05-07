# Kilosort Inputs

Sorrel V1 expects a Kilosort / phy2 directory with these files:

| File | Purpose |
|------|---------|
| `spike_times.npy` | Spike sample indices. |
| `spike_clusters.npy` | Cluster ID for each spike. |
| `cluster_group.tsv` | Initial phy2 labels. Missing labels default to unsorted. |

The raw trace file is passed with `--dat`. If omitted, Sorrel looks for `recording.dat` in the Kilosort directory.

The `.npy` parser supports the dtypes emitted by Kilosort for spike timing and cluster arrays. Sorrel buckets spike times by cluster once during load, then serves hot-path cluster lookups as slice borrows.

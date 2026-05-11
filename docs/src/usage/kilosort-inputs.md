# Kilosort / phy2

Sorrel reads everything phy reads. The minimum is two arrays plus a
trace; the rest of the phy artefacts are picked up opportunistically and
unlock additional views.

## Required

| File                 | Purpose                    |
| -------------------- | -------------------------- |
| `spike_times.npy`    | Spike sample indices.      |
| `spike_clusters.npy` | Cluster ID for each spike. |

Plus a raw trace — typically `recording.dat`. The path can come from
`params.py:dat_path`, `--dat`, `--oebin`, or default to `recording.dat`
inside the directory.

## Optional, used when present

| File                                                | Powers                                            |
| --------------------------------------------------- | ------------------------------------------------- |
| `params.py`                                         | Sample rate, n channels, dtype, offset, dat path. |
| `cluster_group.tsv`                                 | Initial good / mua / noise / unsorted labels.     |
| `channel_positions.npy`                             | Probe geometry — authoritative row order.         |
| `templates.npy`                                     | Template view, similarity matrix.                 |
| `template_features.npy`, `template_feature_ind.npy` | Template-feature scatter.                         |
| `pc_features.npy`, `pc_feature_ind.npy`             | Feature view (PC space).                          |
| `amplitudes.npy`                                    | Amplitude-vs-time scatter, drift map.             |
| `similar_templates.npy`                             | Similarity view shortcut.                         |
| `cluster_<metric>.tsv`                              | Quality-metric columns in the cluster table.      |
| `quality_metrics.csv`                               | SI-style quality metrics.                         |
| `probe.json` / `probegroup.json`                    | Probe geometry fallback (probeinterface).         |

## Loading model

Spike arrays and the raw `.dat` are memory-mapped. On open, Sorrel
buckets spike times and amplitudes once per cluster; subsequent hot-path
lookups are slice borrows into the bucket. The `.npy` parser supports
the dtypes Kilosort actually emits for these arrays.

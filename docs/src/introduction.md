# Sorrel

Sorrel is a native, high-performance GUI for manual curation of spike-sorted electrophysiology data.

The first supported workflow targets Kilosort / phy2 outputs. Sorrel memory-maps spike metadata and the raw recording, presents a cluster-focused GUI, and journals curation operations to SQLite before applying them in memory.

## Highlights

- Native egui / wgpu interface for spike-sorting curation.
- Generic core built around the `DataProvider` trait.
- Compile-time monomorphisation across data, UI, render, and compute hot paths.
- Kilosort / phy2 input support for `spike_times.npy`, `spike_clusters.npy`, `cluster_group.tsv`, and raw `.dat` traces.
- SQLite curation journal for durable edit history.

Source code is hosted at [codeberg.org/caniko/sorrel](https://codeberg.org/caniko/sorrel).

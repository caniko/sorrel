# Sorrel

Native, high-performance GUI for manual curation of spike-sorted electrophysiology data.

Sorrel is built around **zero-cost abstractions**: the entire core is generic over a `DataProvider` trait, monomorphised at compile time, so hot paths (downsampling, vertex generation, journaling) cross no dynamic-dispatch boundary.

## Workspace

| Crate            | Role                                                   |
|------------------|--------------------------------------------------------|
| `sorrel-io`      | Concrete data backends; `DataProvider` trait bound.    |
| `sorrel-compute` | Generic kernels (LTTB, histograms).                    |
| `sorrel-data`    | Generic `Session<P>`, `CurationCommand`, SQLite journal.|
| `sorrel-render`  | Concrete vertex types + monomorphised buffer builders. |
| `sorrel-ui`      | egui widgets, generic over `P: DataProvider`.          |
| `sorrel`         | Binary; detects backend and instantiates `SorrelApp<P>`.|

## V1 (Kilosort / phy2)

```
sorrel <KILOSORT_DIR> --dat recording.dat --sample-rate 30000 --channels 32
```

- Memory-maps `spike_times.npy`, `spike_clusters.npy`, and the raw `.dat`.
- Loads `cluster_group.tsv` into a `Vec<PhyLabel>` (1 byte per cluster).
- Curation operations are journaled synchronously to SQLite **before** the in-memory state changes.

### Keyboard

| Key            | Action                          |
|----------------|---------------------------------|
| `G` / `M` / `N` / `U` | Set selected cluster Good / MUA / Noise / Unsorted |
| `J` / `K` / arrows    | Next / previous cluster        |
| `H` / `L`             | Pan trace window               |
| `Cmd/Ctrl-Z`          | Undo (V1: stub)                |

## Build

```
cargo build --release -p sorrel
```

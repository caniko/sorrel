# Workspace Layout

Sorrel is a Cargo workspace with one role per crate:

| Crate            | Role                                                                                                                                                                         |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sorrel-io`      | Concrete data backends and the `DataProvider` trait. Phy / Kilosort, SortingAnalyzer, NWB, KS4 rez, plus recording-only readers (SpikeGLX, Open Ephys, MDA, probeinterface). |
| `sorrel-cache`   | Rebuildable derived-artifact cache. Owns content-addressed keys, rkyv archive I/O, atomic writes, and mmap-backed bytechecked reads; deleting it never loses user data.        |
| `sorrel-compute` | CPU kernels: LTTB downsampling, histograms, ISI / CCG, CMR + filter, mean snippet, GMM, drift, quality metrics, isolation distance.                                          |
| `sorrel-gpu`     | wgpu compute pipelines (CMR median, biquad HP filter, mean snippet) sharing the device and queue with the renderer.                                                          |
| `sorrel-data`    | Generic `Session<P>`, `CurationCommand`, SQLite `Journal`, save / replay, QC export, merge/split previews and suggestions.                                                   |
| `sorrel-render`  | Concrete vertex types and monomorphised buffer builders for the trace and raster paths.                                                                                      |
| `sorrel-ui`      | egui widgets and views, generic over `P: DataProvider`.                                                                                                                      |
| `sorrel`         | Binary. Parses CLI, detects backend, instantiates `SorrelApp<P>`.                                                                                                            |
| `sorrel-py`      | Optional Python bridge (PyO3 / maturin) — exposes `sorrel.open(recording, sorting)` for SpikeInterface users.                                                                |

The split is functional, not layered: each crate owns its concerns end
to end, and the binary stitches one fully-monomorphised arm together
per backend it supports.

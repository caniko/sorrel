# Sorrel

Sorrel is a native, high-performance GUI for manual curation of spike-sorted
electrophysiology data. It targets the same workflow as
[phy](https://github.com/cortex-lab/phy) — cluster-by-cluster inspection,
relabelling, merging, and splitting — but is built around a zero-cost,
monomorphised Rust core and a wgpu-backed renderer.

## Highlights

- **Multiple backends.** Reads phy/Kilosort directories, SpikeInterface
  `SortingAnalyzer` (binary folder), Kilosort 4 `rez.mat`, and NWB files.
  Recording-only formats (SpikeGLX, Open Ephys, Mountainsort MDA) are
  supported as raw trace sources.
- **Native egui + wgpu UI.** Trace, raster, ISI, CCG, waveform, template,
  feature, drift, probe, similarity, quality, and suggestion views.
- **GPU compute path.** Optional wgpu kernels for common-median
  referencing, high-pass filtering, and mean-snippet extraction along the
  trace window; transparent CPU fallback.
- **Static dispatch core.** `Session<P>` is generic over a `DataProvider`
  trait — every hot path (downsampling, vertex generation, journaling,
  metrics) is monomorphised per backend.
- **Durable curation.** Every label change, merge, split, and undo/redo is
  written to a SQLite journal *before* the in-memory session moves. The
  journal is replayable and idempotent.
- **Headless QC export.** `--export-qc` runs the full metrics pipeline and
  writes `cluster_qc.tsv` + `cluster_qc.json` without spawning a window.
- **Python bridge.** The `sorrel-py` crate exposes `sorrel.open(recording,
  sorting)` for SpikeInterface users.

Source code is hosted at
[codeberg.org/caniko/sorrel](https://codeberg.org/caniko/sorrel).

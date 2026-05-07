+++
title = "Sorrel"

[extra]
tagline = "Native spike-sorting curation"
subtitle = "A high-performance GUI for manual curation of spike-sorted electrophysiology data — phy / Kilosort, SpikeInterface, NWB, and KS4 rez out of the box, with a monomorphised Rust core, wgpu compute, and a durable SQLite journal."
install = "cargo build --release -p sorrel"

[[extra.features]]
title = "Phy & Kilosort, natively"
description = "Reads spike_times, spike_clusters, params.py, templates, PC features, amplitudes, similar_templates, channel_positions, and quality-metric sidecars exactly as phy does."

[[extra.features]]
title = "Multiple backends"
description = "phy / Kilosort, SpikeInterface SortingAnalyzer (binary folder), Kilosort 4 rez.mat, and NWB. SpikeGLX, Open Ephys, and MDA work as raw trace sources."

[[extra.features]]
title = "Static-dispatch core"
description = "Session, UI, render, and compute paths are generic over the DataProvider trait — every hot path is monomorphised per backend, no Box<dyn Trait>."

[[extra.features]]
title = "GPU trace pipeline"
description = "Optional wgpu kernels for common-median referencing, high-pass filtering, and mean-snippet extraction — sharing the renderer's device and queue, with a transparent CPU fallback."

[[extra.features]]
title = "Durable curation journal"
description = "Every relabel, merge, split, and undo/redo lands in SQLite before the in-memory session moves. Replay on open is idempotent; resealed at save."

[[extra.features]]
title = "Built-in views"
description = "Trace, raster, ISI, CCG, waveform, template, feature, drift, probe, similarity, quality, and merge/split suggestions, all in one window."

[[extra.features]]
title = "Headless QC export"
description = "sorrel --export-qc DIR runs every metric on the loaded session and writes cluster_qc.tsv + cluster_qc.json without spawning a window."

[[extra.features]]
title = "Python bridge"
description = "sorrel.open(recording, sorting) opens a SpikeInterface analyzer in the native GUI via the sorrel-py bridge."
+++

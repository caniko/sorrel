# Saving & QC Export

Sorrel separates two persistence concerns:

1. **The journal** — every action goes here, automatically and
   synchronously, before the in-memory session moves. This is the
   source of truth for "what edits has the user made."
2. **Phy artefacts** — the canonical files other tools read
   (`spike_clusters.npy`, `cluster_group.tsv`). These are written
   on demand.

## Saving phy artefacts

```text
Cmd/Ctrl-S
```

Writes the current curated state into the directory the binary was
opened with (`save_dir`):

- `spike_clusters.npy` — updated per-spike cluster assignment after any
  merges and splits.
- `cluster_group.tsv` — `good` / `mua` / `noise` / `unsorted` labels.

These are exactly the files `spikeinterface.extractors.read_phy()` reads
back, so a round trip into SpikeInterface or phy works without
intermediate steps.

The journal is _resealed_ at save time so a fresh open replays cleanly
from the new baseline rather than from the original sort.

## Headless QC export

```bash
sorrel <DATA_DIR> --export-qc out/qc/
```

Loads the data, runs the full metrics pipeline (firing rate, presence,
refractory contamination, amplitude stats, ISI violations, isolation
distance where PC features are available, composite quality), writes
both:

- `out/qc/cluster_qc.tsv` — one row per cluster, header columns are
  metric names.
- `out/qc/cluster_qc.json` — same data, JSON shape for programmatic
  consumption.

…and exits without spawning a window. The journal is replayed first, so
the export reflects current curation.

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

- `curation.json` — SpikeInterface's manual-curation document
  (`format_version` 2), see below.

The first two are exactly the files `spikeinterface.extractors.read_phy()`
reads back, so a round trip into SpikeInterface or phy works without
intermediate steps.

### The journal and `sorrel_original/`

The journal replays onto the sort as it was loaded, never onto the files
the last save wrote. Saving keeps the whole [history](./history.md).

For phy / Kilosort directories, the first save that replaces a file
copies the original into `sorrel_original/`. The arrays a rescue rewrites
are hard-linked instead. `sorrel_original/ROOT` lists every file captured
this way, including the files that did not exist yet. Sorrel reads those
copies on open, so a reopen rebuilds every node of the history. Leave the
directory in place.

Each save adds a seal to the journal: the hash of the `spike_clusters.npy`
and `cluster_group.tsv` it is about to write. The seal is recorded before
the files are renamed into place, so a crash at any point leaves files that
match the sort as loaded or one of the seals.

Opening checks two things and fails closed when either fails:

- The original `spike_clusters.npy` must match the journal. It does not
  when the sorter was re-run or the file was copied from elsewhere.
- The current `spike_clusters.npy` and `cluster_group.tsv` must be the
  original or one of sorrel's saves. Otherwise someone edited them outside
  sorrel, for example by curating in phy after a sorrel save, and the next
  sorrel save would overwrite that work.

To curate from the files as they are now, reopen with `--discard-journal`.
It renames the journal to `<journal>.<seconds>` and `sorrel_original/` to
`sorrel_original.<seconds>`. Nothing is deleted. The current files become
the new starting point. To keep the journal instead, restore the files it
was recorded against.

SortingAnalyzer, NWB and `rez.mat` sessions reload the original sort on
every open, so they need no copies or checks.

### `curation.json`

Every save also writes SpikeInterface's manual-curation document next to
the phy files. It carries the `good` / `mua` / `noise` labels and, for
SortingAnalyzer, NWB and `rez.mat` sessions, the merges and splits made
since the sort was loaded, so a SortingAnalyzer round trip is one call:

```python
import spikeinterface.curation as sc
curated = sc.apply_curation(analyzer, "my_run/sorting_analyzer/curation.json")
```

For phy directories the saved `spike_clusters.npy` already contains the
merges and splits, so there `curation.json` is labels only and must not
be applied on top of the npy a second time. Sorrel reads the file back
on open when no `cluster_group.tsv` is present (phy dirs) and always for
SortingAnalyzer folders, so labels made in SpikeInterface show up too.

## Exporting a phy directory

Saving updates the curation files in place. Exporting writes a complete
phy template-gui directory somewhere else, from any backend, so phy,
`spikeinterface.extractors.read_phy()` or a colleague without Sorrel can
open the curated result:

```bash
sorrel export-phy <DATA_DIR> --out curated_phy/ [--copy-raw]
phy template-gui curated_phy/params.py
```

The same export is available as the `export_phy` tool of `sorrel mcp`,
the **Export phy** button (it writes `<dataset>_phy` next to the dataset),
and `Session.export_phy(out_dir, copy_raw=False)` in `sorrel-py`. It is the
`phy-export` Cargo feature, on by default.

What goes into the directory depends on the source:

- **phy / Kilosort**: every `*.npy`, `*.tsv` and `*.csv` file is copied
  byte for byte (templates, amplitudes, PC features, whitening, …), with
  `spike_clusters.npy` and `cluster_group.tsv` rewritten from the curation.
  Large non-phy files such as `temp_wh.dat` or `rez.mat` stay behind.
- **SortingAnalyzer, NWB, `rez.mat`**: `spike_times.npy`,
  `spike_clusters.npy`, `spike_templates.npy` (one template per cluster),
  `templates.npy`, `channel_map.npy`, `channel_positions.npy` (and
  `channel_shanks.npy` when known) and `cluster_group.tsv`. Templates are
  mean waveforms of up to 500 spikes per cluster, 1 ms before to 2 ms after
  each spike, from the raw trace with the per-channel mean removed (no
  high-pass filter). There are no amplitudes, PC features or similarity
  matrix, so phy falls back to its defaults for those views.

`params.py` points at the original raw recording by absolute path. With
`--copy-raw`, or for NWB, whose trace is held in memory, the trace is
streamed into `recording.dat` in its native dtype instead, which also
decompresses an mtscomp `.cbin`. `params.py` is written last, so an
interrupted export never looks complete to phy. The export refuses to
write into the source directory. The exported directory has no journal:
opening it in Sorrel starts a fresh curation on top of the exported state.

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

# Changelog

All notable changes to Sorrel are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed (data integrity)

- **Saving no longer corrupts the next session.** `Ctrl-S` on a phy /
  Kilosort directory now reseals the journal against the freshly written
  `spike_clusters.npy`, and the binary seals every journal against its
  data on open, so a reopen loads the saved state once instead of
  replaying every edit (and every split) a second time on top of it.
  Journals sealed against other data fail closed with a message naming
  `--discard-journal`. SortingAnalyzer, NWB and `rez.mat` sessions keep
  their journal as the source of truth and are never resealed.
- The journal file is `sorrel.journal` (it was never SQLite); a legacy
  `sorrel.sqlite` is still picked up with a warning.
- `quality_metrics.csv` accepts tab- and semicolon-separated files;
  `cluster_KSLabel.tsv` seeds labels when there is no `cluster_group.tsv`
  (and is no longer mistaken for a numeric metric); `--export-qc` works
  for every backend; the QC JSON key is `firing_rate` like the TSV.

### Added (CLI and interchange)

- `clap` command line with subcommands: `qc`, `suggest [--json]`,
  `apply --rules FILE [--dry-run] [--save]` (QC-threshold auto-labels as
  one undoable batch), `report` (self-contained HTML with per-cluster
  metrics, template, ISI and ACG as inline SVG), `serve` (JSON-RPC 2.0
  over stdio in MCP shape: metrics, suggestions, SVG plots, journaled
  relabel / merge / split / undo / save), `diff A.tsv B.tsv` (confusion
  table, Cohen's kappa, disagreements), `journal dump|replay`,
  `clear-cache`. `sorrel <DIR> [flags]` still opens the GUI.
- Every save also writes SpikeInterface's `curation.json`
  (`format_version` 2) next to the phy files; labels (and, for
  SortingAnalyzer folders, pending merges and splits) round-trip through
  `spikeinterface.curation.apply_curation`. The file is read back on open
  to seed labels. SortingAnalyzer unit ids are preserved.
- `sorrel-py`: in-process `sorrel.Session` (open, qc, suggest, labels,
  set_label, merge, split, undo, redo, save, curation) sharing the GUI's
  journal; `open_analyzer` keeps its directory by default so the
  curation survives, and `read_curation` loads it for SpikeInterface.
- Library: `Session::version()` (monotonic cache key),
  `Session::reseal`, `applied_commands`, `primary_template`;
  `Journal::replay_at`, `n_records`; `sorrel_data::open` and
  `sorrel_io::detect` shared by the binary and the bindings;
  `DataProvider::unit_id`.

### Added (library, for the science views)

- `sorrel_compute::psth` (peri-event histogram plus per-event relative
  spike times) and `sorrel_io::Events` (`events.csv` `time,label` in
  seconds, or `events.npy`), the data side of an event-aligned view.
- `sorrel_data::spike_depths`: per-spike depth from PC-weighted centre of
  mass with a template-peak fallback, the data side of a real drift map.
- `sorrel_compute::subtract_channel_median_grouped`: common-median
  referencing within each shank, threaded through sorrel-render as
  `build_trace_vertices_{cfg,gpu}_shanked` and a public
  `apply_preproc_cpu`.

### Changed (performance)

- `ClusterIndex::from_provider` merges the per-cluster spike buckets with a
  heap instead of an argmin scan: 1 M spikes at 1000 clusters 1.07 s →
  75 ms, at 100 clusters 116 ms → 23 ms (bench
  `seed_amplitudes_cache`, `SORREL_BENCH_CLUSTERS`).
- `.npy` files are memory-mapped and, when the on-disk type already
  matches, copied with one `memcpy` instead of decoded element by element;
  peak memory per array drops from 2× to 1×. `pc_features.npy` and
  `templates.npy` are shared (`Arc`) between the provider and the cluster
  index instead of copied.
- mtscomp `.cbin` + `.ch` recordings (the IBL archive format) open
  directly: point `params.py:dat_path` (or `--dat`) at the `.cbin` and
  windows are inflated on demand with a small chunk cache.

### Added (UI)

- Dockable layout: every view is an `egui_tiles` tile that can be dragged,
  split, closed and reopened (`View` menu, `+` per tab bar, "Reset layout").
  The layout, table sort/filter, trace settings and last selection persist
  per dataset in `<dataset>/.sorrel/ui.json`.
- Merge the selection with `Ctrl/Cmd-G` or the **Merge N** button; multi-
  cluster relabels are journaled as one batch so one undo reverts them.
- Keyboard help overlay (`?` / `F1`), Undo/Redo buttons that name the
  command, status toasts for every curation step, dismissable errors,
  window title with dataset name and dirty marker, and a modal confirm
  dialog (Enter / Esc) including an unsaved-changes guard on quit.
- Trace pane: channel labels, time axis, spike ticks for the selection on
  each cluster's peak channel, drag/wheel pan and Ctrl-wheel zoom.
- Raster highlights the selected rows and outlines the trace window;
  Similar rows are clickable; plots carry axis ticks; probe rings always
  follow the selection.
- Loading screen while a recording is seeded on a worker thread; Suggest
  ranking and quality metrics run in the background with spinners.
- PSTH pane: event-aligned firing rate and per-trial raster from
  `events.csv` / `events.npy`; the Drift pane plots spike depth over time
  when the backend can estimate it; the Trace pane offers per-shank CMR on
  multi-shank probes.

### Changed (UI)

- `J`/`K` and Shift-click follow the cluster table's visible order instead
  of raw ids; whole rows are clickable and the table scrolls to the
  keyboard selection.
- Clusters are coloured by their position in the selection, consistently
  across all views; plot colours follow the egui light/dark theme.
- `Ctrl/Cmd` + letter no longer triggers the plain-letter label keys.
- Correlograms, quality metrics and trace vertices are memoised per
  selection/curation version instead of being recomputed every frame; the
  raster vertex buffer is uploaded to the GPU once per change instead of
  every frame.

## [0.1.0] - 2026-05-25

### Added

- Initial first-publish release for the Sorrel workspace.
- Native spike-sorting curation GUI and supporting crates for data loading,
  compute kernels, GPU helpers, rendering, and UI integration.
- Rebuildable derived-data cache for expensive local artifacts, including
  cache keys, rkyv archive storage, bounded garbage collection, and dataset
  cache clearing.
- Forgejo release workflows generated by simit for the publishable crates.

### Notes

- All workspace crates are first-publish candidates at `0.1.0`; none are
  expected to exist on crates.io before this release.
- Downstream dry-runs can fail until their local path dependencies have been
  published in order. See `RELEASE.md` for the publish sequence.

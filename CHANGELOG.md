# Changelog

All notable changes to Sorrel are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- A GPU backend that rejects a compute kernel no longer kills Sorrel. Kernel
  creation now runs under wgpu error scopes, so a driver or shader-compiler
  failure (FXC on D3D12 was the case seen) leaves trace preprocessing on the
  CPU and logs one warning naming the adapter and backend, instead of
  panicking at startup. Uncaptured wgpu errors on Sorrel's own devices are
  logged rather than fatal.
- `sorrel gpu-check` reports "kernels unavailable" per adapter and carries
  on with the others instead of panicking. It exits non-zero on a CPU/GPU
  mismatch, or when no compute-capable adapter can run the kernels.

### Changed (release)

- First release on winget (`winget install Caniko.Sorrel`). The release
  workflow now submits the manifests through the GitHub API; 0.2.0's
  submission failed because `wingetcreate` needs .NET, which hosted
  runners lack under wine.
- The workspace crates publish to crates.io from CI in dependency order,
  through simit's trusted publisher, instead of by hand.
- Every change now runs the Windows build on a real `windows-2025` runner:
  `gpu-check` on DX12, headless `qc` and `report`, a GUI launch with a
  screenshot, and a Python wheel install.

## [0.2.0] - 2026-10-26

### Fixed (data integrity)

- Old SpikeInterface phy exports load. Their `amplitudes.npy` is int16 and
  their `channel_positions.npy` int64. Integer arrays read where floats are
  expected are now widened instead of rejected.
- `sorrel auto` changes:
  - **Geometry check.** On recordings whose channel positions look like
    indices (< 5 µm apart, e.g. tetrode sorts), it warns and tags every
    unit `review:geometry`. Bombcell's and UnitRefine's spatial metrics are
    meaningless there. On two labelled tetrode recordings it called 58 of 65
    human-kept units noise.
  - **Reruns are reproducible.** The random spike sample is seeded, and the
    analyzer cache is keyed on the extension parameters.
  - **No full recompute after a save.** The cache key is now the spike-train
    values rather than the file bytes.
  - **`bc:*` reason tags follow SpikeInterface's rules.** Bounds are
    inclusive and NaN metrics fail where Bombcell says they do. Previously
    the tags could name a rule the unit had passed. Labels were unaffected.

- Per-spike arrays stored as `(n, 1)` columns, as written by Kilosort 2 and
  SpikeInterface's `export_to_phy`, now load. They used to fail with
  "expected 1-D array".

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
- `spike_clusters.npy` is saved in `spike_times.npy` row order. It used to be
  written in time order, which misaligned it with `amplitudes.npy`,
  `pc_features.npy` and the other per-spike arrays whenever the source was
  unsorted or had simultaneous spikes in different clusters.

### Fixed (GPU)

- **No more crashes on the OpenGL fallback.** When wgpu picks a GL adapter
  (old Intel iGPUs, NVIDIA without a Vulkan driver, VMs, `WGPU_BACKEND=gl`),
  eframe gets no compute support; trace CMR and high-pass now stay on the
  CPU there instead of panicking on pipeline creation.
- **Long trace windows no longer crash on any GPU.** Windows longer than
  about 2 s at 30 kHz exceeded the 65,535-workgroup and 128 MiB
  storage-buffer limits; the kernels now check device limits and the trace
  falls back to the CPU.
- **OpenGL on NVIDIA no longer loses the device.** wgpu's indirect-call
  validation builds a compute pipeline on every device, which NVIDIA's GL 3.3
  EGL context cannot run. Sorrel issues no indirect calls, so it is off for
  every Sorrel wgpu instance (`WGPU_VALIDATION_INDIRECT_CALL=1` restores it).
- **The Linux tarballs and the AppImage use the host's GPU drivers.** They
  used to ship a Nix-built binary (glibc 2.44, `/nix/store` loader) that
  either did not start on Ubuntu or Fedora or found no GPU. They are now
  built with cargo-zigbuild against glibc 2.31 and load the host's
  Vulkan/GL libraries, and with them its Mesa or NVIDIA driver. The
  AppImage wraps that binary instead of a Nix closure.
- **The Windows release builds again.** `aws-lc-sys` (via libp2p-quic)
  needs winpthreads' `<sched.h>`, which the MinGW compiler did not see.

### Added (automatic curation)

- **`sorrel auto`**: Kilosort output in, curated session out. An embedded
  SpikeInterface (≥ 0.105) pipeline runs:
  - redundant-unit detection
  - the `similarity_correlograms` and `slay` merge presets
  - Bombcell and UnitRefine labels on the merged units
  - with `--llm`, a Claude verdict on the units the classifiers disagree on
    or are unsure about, and on merges only one preset found

  The structured numeric evidence and each verdict are kept in
  `evidence.json`. Everything is applied as one undoable batch.
  See [Automatic curation](docs/src/usage/auto-curation.md).

- **Cluster tags.** Every automatic decision records its source (`src:*`,
  `merged:*`, `bc:*`, `conf:*`), and uncertain ones get `review:*`.
  - Tags are journaled and undoable (`CurationCommand::Tag`).
  - They are saved to `cluster_sorrel_tags.tsv` (a phy column) and to a
    `tags` category in `curation.json`.
  - The cluster table has a tags column and a **review** filter, and its
    filter box matches tags.
  - A GUI edit clears the cluster's `review:*` tags and adds `src:human`.
- `sorrel apply --curation FILE` imports a SpikeInterface `curation.json`
  (labels, tags and merges) as one undoable batch. Importing the same
  document again changes nothing.
- `sorrel report --tag PREFIX` renders only matching clusters
  (`--tag review:` gives the review queue).
- `sorrel serve` adds `tag` and `evidence`, and `list_clusters` takes a tag
  filter.

### Added (CLI and interchange)

- **Python wheels on PyPI as `sorrel-ephys`** (`pip install sorrel-ephys`,
  then `import sorrel`; PyPI's `sorrel` is an unrelated project), also
  attached to every GitHub release: one `cp38-abi3` wheel per
  platform (manylinux_2_34 x86_64 / aarch64, win_amd64, macosx_11_0_arm64)
  with the `sorrel` Python API and the GUI binary, assembled by
  `nix/wheel.nix` from the release cross builds and uploaded by the release
  workflow with PyPI trusted publishing. `sorrel.launch` also finds
  the bundled binary when the kernel's PATH lacks the environment's
  scripts directory.
- **Windows packages on winget (`Caniko.Sorrel`) and Chocolatey
  (`sorrel`).** The release workflow submits both from the release's
  Windows zip on every stable tag. The repository now carries the
  `LICENSE-MIT` and `LICENSE-APACHE` texts its `MIT OR Apache-2.0` license
  names.
- `sorrel gpu-check` lists every GPU adapter wgpu can see (vendor, backend,
  driver) and checks CMR, high-pass and mean-snippet kernels against the CPU
  on each, exiting non-zero on a mismatch. Headless GPU contexts (tests,
  this command) honour `WGPU_BACKEND`, `WGPU_ADAPTER_NAME` and
  `WGPU_POWER_PREF`, like the GUI.
- `clap` command line with subcommands: `qc`, `suggest [--json]`,
  `apply --rules FILE [--dry-run] [--save]` (QC-threshold auto-labels as
  one undoable batch), `report` (self-contained HTML with per-cluster
  metrics, template, ISI and ACG as inline SVG), `serve` (JSON-RPC 2.0
  over stdio in MCP shape: metrics, suggestions, SVG plots, journaled
  relabel / merge / split / undo / save), `diff A.tsv B.tsv` (confusion
  table, Cohen's kappa, disagreements), `journal dump|replay`,
  `clear-cache`. `sorrel <DIR> [flags]` still opens the GUI.
- **Full phy export** (Cargo feature `phy-export`, on by default):
  `sorrel export-phy <DIR> --out OUT [--copy-raw]`, the `export_phy`
  serve tool, an **Export phy** button and `Session.export_phy(out)` in
  `sorrel-py` all write a complete phy template-gui directory from any
  backend. A phy source is copied with its curation applied. SortingAnalyzer,
  NWB and `rez.mat` sessions get mean templates computed from the raw trace.
  `params.py` points at the raw file; `--copy-raw` (or an NWB source, whose
  trace lives in memory) streams it into `recording.dat`.
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

- **Export phy** button next to Export QC: writes the curated session as a
  phy directory beside the dataset (`<dataset>_phy`) on a worker thread.
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
  `events.csv` / `events.npy`, `--events FILE` or an NWB trials table
  (`start_time`, labelled by `trial_type`); the Drift pane plots spike depth over time
  when the backend can estimate it; the Trace pane offers per-shank CMR on
  multi-shank probes.

### Changed (UI)

- Waveforms and mean waveforms follow the Trace pane's HP/CMR toggles
  (per shank when enabled) instead of always showing the raw recording;
  each snippet is read with a filter-settling margin so the HP transient
  never reaches it.
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

### Fixed (UI)

- **Trace view shows spikes.** Gain followed the dtype's full scale (32768
  for int16), so spikes of a few hundred counts drew as flat lines. The
  window's largest excursion now spans one channel row.
- **Views get more room.** The cluster panel no longer opens at ~40% of a
  1280 px window: table columns size to their content and the filter strip
  wraps. Only the active tab shows its close button, so most tab titles fit
  without the overflow arrows.
- Captions that were clipped now fit or wrap: Quality (its wide table
  scrolls sideways), Features axis pickers, CCG and Waveforms.
- **Contrast.** Weak text in dark mode, warnings in light mode and errors in
  both now meet WCAG AA. A unit test pins the colours.
- **Accessibility.** Screen readers are supported (eframe `accesskit`). Pane
  tabs and the cluster filter now have accessible names.
- The load-error screen names the dataset.

### Added (testing)

- `crates/sorrel-ui/tests/ui_states.rs` drives the real app headless with
  egui_kittest. It renders 26 states (startup, selection, help, quit
  confirmation, load error, a 40-unit dataset across eight panes; dark and
  light) through wgpu and compares them with committed snapshots. The dev
  shell pins Mesa's lavapipe (`SORREL_LAVAPIPE_ICD`), so GPU-less CI runners
  render the same pixels as a workstation. Accept intended changes with
  `UPDATE_SNAPSHOTS=1 cargo test -p sorrel-ui --test ui_states`.

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

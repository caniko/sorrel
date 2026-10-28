# Changelog

All notable changes to Sorrel are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- History that never discards: undo moves back, an edit after an undo starts
  a branch, and the whole tree survives saving and reopening. A **History**
  pane lists every edit with who made it and when, marks branches and saves,
  and goes back to any edit on click. See [History](docs/src/usage/history.md).
- One session host for the GUI, MCP agents and remote guests. Every edit is
  journaled with its actor (you, an agent's client name, an algorithm, a
  remote peer).
- `sorrel mcp <DATA_DIR>` connects an agent to the GUI or headless host on
  the directory, or serves the dataset itself. `sorrel --headless <DATA_DIR>`
  runs the host without a window. See [Agents](docs/src/usage/agents.md).
- Agent edits in the GUI: a toast names the agent and the edit, the touched
  clusters are selected and tinted in the table, and the top bar lists the
  connected agents with an activity dot.
- **Direct | Propose** switch for agents. Under Propose, agent edits wait as
  cards in the new **Proposals** pane; Accept applies one as your decision
  (an accepted split keeps the agent as its method), Reject leaves the
  not-merge / not-split memory.
- Agents get `notifications/message` when someone else changes the session.

- Unit origin: every unit is `Auto[<sorter>]` or `Sorrel[Manual]`,
  `Sorrel[Algo:<name>]` or `Sorrel[LLM:<name>]`, with direct parents and a
  free-text note (`:note`). Auto units edited in sorrel are tagged `tuned` or
  `split`; units phy already edited are tagged `external`. Saved in
  `cluster_sorrel_annotations.tsv` and as a curation.json `origin` label.
  `--sorter NAME` overrides the detected sorter name.
- `sorrel rescue`: FDR-controlled matched-filter spike rescue for existing
  units, with hybrid-injection recall checks.
- In-place `AddSpikes`: rescued spikes are journaled as one undoable batch.
- Home Manager module (`homeManagerModules.default`, `programs.sorrel`), and a
  `sorrel.desktop` menu entry in the Nix packages.
- `checks.workspace-tests`: every workspace test under nextest in the Nix
  sandbox, the UI snapshot states on pinned lavapipe (`SORREL_REQUIRE_GPU`
  fails instead of skipping). simit qualifies every x86_64-linux flake check
  as an exact installable (`nix-builds.yaml`).
  Save rewrites every per-spike array, with backups in `sorrel_original/`.

### Changed

- The curation journal is format v3, a tree of edits with their actor and
  time, head moves and save seals. v1 and v2 journals are upgraded on open,
  their undone commands kept as branches; sorrel 0.2.x refuses a v3 journal
  ("bad magic") instead of truncating it. Remote protocol v3.
- Saving no longer clears the undo history. For phy directories, the first
  save that replaces a file keeps its original in `sorrel_original/` (listed
  in `sorrel_original/ROOT`), and reopening replays the journal onto those
  originals.
- Opening a phy directory whose `spike_clusters.npy` or `cluster_group.tsv`
  changed outside sorrel since its last save (e.g. curated in phy) fails
  instead of letting the next save overwrite that work.
  `--discard-journal` now renames the journal and `sorrel_original/` aside
  instead of deleting them.
- After a rescue save, undoing the rescue and saving again writes the
  original per-spike arrays back; the directory no longer needs a reopen.
- `sorrel serve` is now an alias of `sorrel mcp`. MCP agents can edit while
  the GUI is open; `propose`, `select` and `selection` work in the same
  server.
- Suggestions are memoised per cluster. Re-ranking after an edit takes
  137 ms instead of 807 ms on 1000 clusters and 2M spikes (about 5.7×).
- Merge suggestions include drift-fragment pairs: nearby peak channels with
  complementary firing over time.
- Split suggestions ignore amplitude bimodality that bursting explains
  (reason `burst_explained`).
- Per-cluster cache keys no longer include the journal head, so caches
  survive unrelated edits.

### Fixed

- The cluster table's "amp" column is mean spike amplitude × template
  peak-to-peak, as in phy. It used to show the template peak alone, which
  is the same for every cluster on unit-norm Kilosort templates.

### Changed (UI)

- **phy-parity views fit their panes.**
  - CCG, ISI, Waveforms and Scatter keep their main control on one row and
    put the rest in a ⚙ popover, so toolbars no longer wrap or overlap.
  - **Cluster table:**
    - The default columns (ID, ch, fr, n, qual, label) fit the panel, which
      opens at 22% of the window.
    - depth, sh, amp, isi, tags, origin and note are opt-in from the header
      menu.
    - Only the label cell is coloured, and descending sorts show ▼.
  - **Similar:** drops the repeated filter block.
  - **Features:**
    - Selected clusters are drawn over dimmed background spikes.
    - The grid adapts to the pane width, with square cells and readable
      labels.
    - Axes fit the spikes shown.
  - **Waveforms:** overlays the selected clusters by default, with a colour
    legend. "overlap" (`o`) switches to phy's side-by-side layout.
  - **CCG:** cells keep a 120 px minimum and scroll; axes sit on the outer
    edges; chips are readable.
  - **ISI:** overlays every selected cluster in one plot, with a shaded
    refractory band and per-cluster "<1 ms" chips.
  - **Scatter:** fills its pane, rings the selection, and labels axes with
    units.
  - The third selection colour is purple instead of red, so a selected
    cluster no longer looks like an error.
- **Roomier layout.**
  - Text is 14 px, captions are at least 11 px, and controls and table rows are 24 px.
  - The top bar is 36 px and tab bars are 30 px, with 6 px gaps between panes.
  - The cluster panel opens at 340 px.
- **Save state is always visible.** A chip next to the title reads
  "unsaved", "saved" or "save failed". Hovering it shows the save path or
  the full error.
  - A Save button joins Undo, Redo and Merge, and Merge names its target
    ("Merge 2 -> c1").
  - The save message no longer prints over the toolbar.
- **Cluster table.**
  - Numbers are right-aligned monospace.
  - A selected row keeps its red and orange scores readable and shows its
    selection colour in a bar at the left edge.
  - IDs no longer shift when a row is selected.
  - The tags column appears only when some cluster has tags.
  - Clicking a row's text now selects the cluster. It used to start a text
    selection, and the Similar table had the same problem.
- **New default layout.**
  - Three tab groups sit along the top: Shape, Compare and Assess.
  - Below them, a Time group (Raster, Drift, Contam·time, PSTH) sits beside the trace.
  - Saved per-dataset layouts are kept.
- **Plot axes sit outside the data.**
  - Tick labels live in their own margins and units are axis titles, so labels no longer
    collide with data or clip at pane edges.
  - Trace channel numbers have a gutter, and the trace has a time axis.
  - The trace is drawn in neutral grey, so selection markers stand out.
- **Clearer panes.**
  - **CCG:** y-axis, ±50 ms ticks, a visible refractory band and a sparse-data note.
  - **PSTH:** step bars and the per-event breakdown.
  - **Raster:** selected clusters in their selection colour and the rest in neutral grey.
  - **Stats:** marks where the selected cluster falls.
  - **Quality:** every sub-score coloured and the weakest bold.
  - **Drift:** "no measurable drift" instead of repeated ticks.
  - **Contam·time:** says when there is too little data.
  - **Waveforms:** use the cell height and show a scale bar.
  - **Probe:** drawn to scale with channel labels.
- **Dialogs.**
  - **Quit:** the destructive "Quit without saving" is red, and "Save & quit" is the focused default.
  - **Load error:** shows the full path.
  - **Loading:** larger, readable loading screen.
  - **Help:** scrolls instead of clipping.
- **Contrast.**
  - Secondary text, unchecked checkboxes, and plot grid and reference lines now meet
    WCAG AA in both themes (pinned by a unit test).
  - Light-theme window shadows are soft.

## [0.2.1] - 2026-10-28

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

[Unreleased]: https://github.com/caniko/sorrel/compare/0.2.1...HEAD
[0.2.1]: https://github.com/caniko/sorrel/compare/0.2.0...0.2.1

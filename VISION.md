# Sorrel vision

Sorrel is a native, high-performance GUI for manual curation of spike-sorted
electrophysiology data, with a headless CLI for the same operations. It targets
the phy workflow — cluster-by-cluster inspection, relabelling, merging, and
splitting — on a Rust core that is generic over a data provider and a
wgpu-backed renderer.

The `sorrel mcp` server serves MCP over stdio so agents can read a dataset's
metrics, evaluate and apply threshold rules, and make handle-based relabel,
merge, and split edits through the same journaled session host as the GUI.

## Problem

Manual curation is where a spike-sorted dataset becomes usable, and it is
traditionally done in phy over a limited set of backends and file formats. Large
recordings make inspection slow, curation decisions are easy to lose or
misattribute, and there is no reliable way to hand the same session to an agent
or another machine. Sorrel exists to make curation fast, durable, and
programmatically accessible without abandoning the phy workflow.

## Principles

- Durable curation before convenience. Every label change, merge, split, and
  undo/redo is written to an append-only journal before the in-memory session
  moves, and the journal is replayable and idempotent.
- Static dispatch over dynamic. The core is generic over `DataProvider` and
  monomorphised per backend, so downsampling, vertex generation, journaling, and
  metrics cross no dynamic-dispatch boundary on hot paths.
- One owner per session. The GUI, MCP agents, and remote guests all edit through
  the session host, and every edit is journaled with who made it so it can be
  undone and shows up in the History pane.
- Many backends, one model. phy/Kilosort, SpikeInterface `SortingAnalyzer`,
  Kilosort 4 `rez.mat`, and NWB are read through the same `Session<P>`; raw
  trace sources such as SpikeGLX, Open Ephys, and Mountainsort MDA plug in as
  recording-only inputs.
- Derived data is rebuildable. The cache holds derived results under rkyv with
  garbage collection rather than treating them as truth.
- Provenance is part of a score. The bench database keeps a recorded result only
  while the truth files, session input, run actions, and options it was computed
  from are unchanged.
- GPU is optional. wgpu kernels for common-median referencing, high-pass
  filtering, and mean snippets fall back transparently to CPU.

## Scope

- `sorrel-io` backends, `sorrel-compute` kernels, `sorrel-data` session,
  curation commands, journal, QC export, `curation.json`, rules, report, and
  JSON-RPC server.
- `sorrel-cache`, `sorrel-gpu`, `sorrel-render`, `sorrel-ui`, and the monomorphised
  buffer builders.
- The `sorrel` binary: `open` for the GUI and `qc`, `suggest`, `apply`, `report`,
  `mcp`, `diff`, `journal`, and `export-phy` for headless work.
- `sorrel-py` for SpikeInterface users, and `sorrel-remote` for guests.
- Release artifacts for Linux x86_64/aarch64, macOS Apple silicon, and Windows
  x64/ARM64, plus PyPI, Chocolatey, and winget channels.

## Non-goals

The phy-parity reference marks several things deliberately excluded: the plugin
API, custom grid dimensions, and the in-window IPython console. Scripting goes
through `sorrel-py` and the MCP/JSON-RPC surface instead.

## Direction

The changelog and recent commits describe a curation bench loop: a double-blind
improvement loop with sealing rules, a cost guard, and an acceptance rule, plus
a dev-mode results database at schema v4 with provenance-bound scores, core and
cleanup split proposers, and native Windows on ARM64 releases. Those bench
skills and the dev database are the current focus.

## Place in the fleet

Canix consumes Sorrel as a flake input and loads a `productivity/sorrel.nix`
home module in can's profile. Canix comments record that build inputs stay
Sorrel's own so its trunk prebuild can be substituted from the canix cache.

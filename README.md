# Sorrel

<!-- simit:badges:start -->

[![CI](https://img.shields.io/badge/CI-managed-2088ff)](.github/workflows/ci.yaml) [![Nix](https://img.shields.io/badge/Nix-managed-5277c3)](flake.nix) [![docs](https://img.shields.io/badge/docs-enabled-6f42c1)](docs) [![crates.io](https://img.shields.io/badge/crates.io-publishable-f46623)](https://crates.io/crates/sorrel) [![artifacts](https://img.shields.io/badge/artifacts-configured-2ea44f)](.github/workflows/release.yml) [![Chocolatey](https://img.shields.io/badge/Chocolatey-configured-7b3f99)](https://community.chocolatey.org/) [![winget](https://img.shields.io/badge/winget-configured-0078d4)](https://github.com/microsoft/winget-pkgs/tree/master/manifests/c/Caniko/Sorrel)

<!-- simit:badges:end -->

<p align="center"><img src="assets/showcase/hero.webp" alt="Sorrel's curation workspace: cluster table, waveforms, CCGs, quality metrics, trace and raster, with the 3D feature space and probe map floating alongside" width="100%"></p>

Native, high-performance GUI for manual curation of spike-sorted electrophysiology data. More views in the [gallery](https://sorrel.tartanoglu.com/#gallery).

Prebuilt releases cover Linux (x86_64 and aarch64), macOS (Apple silicon) and
Windows (x64 and ARM64).

Sorrel is built around **zero-cost abstractions**: the entire core is generic over a `DataProvider` trait, monomorphised at compile time, so hot paths (downsampling, vertex generation, journaling) cross no dynamic-dispatch boundary.

## Workspace

| Crate            | Role                                                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sorrel-io`      | Concrete data backends; `DataProvider` trait bound.                                                                                               |
| `sorrel-compute` | Generic kernels (LTTB, histograms).                                                                                                               |
| `sorrel-data`    | Generic `Session<P>`, `CurationCommand`, append-only journal, QC export, SI `curation.json`, rules, report, JSON-RPC server.                      |
| `sorrel-cache`   | Rebuildable derived-data cache (rkyv, GC).                                                                                                        |
| `sorrel-gpu`     | wgpu kernels for CMR, high-pass and mean snippets.                                                                                                |
| `sorrel-render`  | Concrete vertex types + monomorphised buffer builders.                                                                                            |
| `sorrel-ui`      | egui widgets, generic over `P: DataProvider`.                                                                                                     |
| `sorrel`         | Binary; `sorrel open <DIR>` opens the GUI, subcommands (`qc`, `suggest`, `apply`, `report`, `mcp`, `diff`, `journal`, `export-phy`) run headless. |
| `sorrel-py`      | PyO3 bindings: in-process `Session` plus the GUI launcher (outside the workspace; `maturin develop`).                                             |

## V1 (Kilosort / phy2)

```
sorrel open <KILOSORT_DIR> --dat recording.dat --sample-rate 30000 --channels 32
```

- Memory-maps `spike_times.npy`, `spike_clusters.npy`, and the raw `.dat`.
- Loads `cluster_group.tsv` into a `Vec<PhyLabel>` (1 byte per cluster).
- Curation operations are journaled synchronously to the journal **before** the in-memory state changes.

Configure shortcuts in **Settings → Keyboard shortcuts**. **Help** shows the
active bindings. Shortcuts and global preferences are saved in
`~/.sorrel/settings.toml`; see [Keyboard & Mouse](docs/src/usage/keyboard-controls.md).

Views are dockable tiles: drag tabs to rearrange, `×` to close, the
**View** menu to reopen. See `docs/src/usage/views.md` and
`docs/src/usage/keyboard-controls.md`.

## Build

```
cargo build --release -p sorrel
```

# Installation

Sorrel is a Rust workspace and a Nix flake. The default build is pure Rust;
the NWB and Kilosort 4 `rez.mat` backends are gated behind an `hdf5`
feature that links against `libhdf5`.

## With Cargo

```bash
cargo build --release -p sorrel
```

The binary is written to `target/release/sorrel`.

To enable the HDF5-backed readers (NWB, KS4 `rez.mat`):

```bash
cargo build --release -p sorrel --features hdf5
```

## With Nix

```bash
nix build .#sorrel        # default build, no HDF5
nix build .#sorrelHdf5    # with HDF5 features
```

`nix build` builds the default `sorrel` package. The flake also exposes:

- `.#website` — the Zola landing page.
- `.#docs` — this mdBook documentation.
- `.#site` — combined static site, with the docs mounted under `/docs/`.

## Development Shell

```bash
nix develop
```

The shell pulls in the Rust toolchain, native GUI dependencies (wgpu /
egui), `mdbook` for the docs, and `zola` for the website.

To run the docs and site locally:

```bash
cd docs    && mdbook serve
cd website && zola serve
```

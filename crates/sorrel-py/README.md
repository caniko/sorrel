# sorrel-py

Python bindings for the [Sorrel](https://github.com/caniko/sorrel)
spike-sorting curation GUI.

## Install

Each GitHub release carries one wheel per platform (Linux x86_64 and
aarch64 with glibc 2.34 or newer, Windows x86_64, macOS arm64), for any
Python from 3.8 on:

```sh
pip install sorrel-ephys
```

The distribution is `sorrel-ephys` (PyPI's `sorrel` is an unrelated
project); the import is still `import sorrel`. The wheel holds the Rust
extension, the Python wrapper and the `sorrel` GUI binary, which pip puts on
the environment's PATH. Each GitHub release also attaches the wheels.

## Build from source

```sh
cd crates/sorrel-py
maturin develop --release
```

This compiles the Rust extension into the active environment; the GUI
launcher then needs a `sorrel` binary on PATH (`cargo install --path
crates/sorrel`). The release wheels are built by `nix/wheel.nix` instead.

## Use

```python
import sorrel
import spikeinterface.extractors as se

recording = se.read_spikeglx("...")
sorting   = se.read_kilosort("...")

# Exports to a tempdir, launches the GUI, blocks until the user exits.
sorrel.open(recording, sorting)
```

To resume curation later:

```python
sorting_curated = se.read_phy("/path/to/the/exported/phy/dir")
```

## Notes

- Requires `spikeinterface[exporters]` for the export pipeline.
- The launcher subprocesses the `sorrel` binary so the GUI runs in its own
  process — required for stable egui windowing from a Jupyter kernel.
- Direct numpy zero-copy bindings are tracked but not yet implemented.

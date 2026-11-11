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
the environment's PATH. Each GitHub release also attaches the wheels. There
is no Windows ARM64 wheel yet; on ARM64 Windows use the native release zip or
winget.

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

## NWB and NeuroConv

```sh
pip install "sorrel-ephys[nwb]"        # sorrel.nwb.add_units_to_nwbfile
pip install "sorrel-ephys[neuroconv]"  # sorrel.neuroconv.SorrelSortingInterface
```

`add_units_to_nwbfile(path_or_session, nwbfile, starting_time=0.0)` writes
the curated units as the NWB Units table: spike times in seconds, labels,
origin, lineage, tags, QC metrics and curation provenance.
`SorrelSortingInterface(folder_path=...)` is the same thing as a NeuroConv
data interface. See the [NWB section of the integrations
guide](https://github.com/caniko/sorrel/blob/trunk/docs/src/usage/integrations.md#writing-curated-units-to-nwb)
for the columns and the timing contract.

## Tests

From the repo root, with a uv-managed Python (on NixOS prefix the `uv`
commands with `nix shell nixpkgs#uv -c`, and run `maturin develop` inside
`nix develop`):

```sh
uv venv --python-preference only-managed --python 3.12 .venv
uv pip install --python .venv/bin/python pip "maturin>=1.4,<2" pytest \
  pynwb "neuroconv==0.10.2" "hdmf-zarr<0.14" nwbinspector spikeinterface scipy
(cd crates/sorrel-py && VIRTUAL_ENV=$PWD/../../.venv PATH=$PWD/../../.venv/bin:$PATH maturin develop --release)
.venv/bin/python -m pytest crates/sorrel-py/python/tests
```

`hdmf-zarr<0.14` is needed because NeuroConv 0.10.2 still uses the zarr 2
API. CI runs the same suite on the release wheel
(`release/smoke-python.sh`, the `python` target smoke).

## Notes

- Requires `spikeinterface[exporters]` for the export pipeline.
- The launcher subprocesses the `sorrel` binary so the GUI runs in its own
  process — required for stable egui windowing from a Jupyter kernel.
- Direct numpy zero-copy bindings are tracked but not yet implemented.

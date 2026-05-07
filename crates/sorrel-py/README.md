# sorrel-py

Python bindings for the [Sorrel](https://codeberg.org/caniko/sorrel)
spike-sorting curation GUI.

## Build

```sh
cd crates/sorrel-py
maturin develop --release
```

This compiles the Rust extension and installs `sorrel` into the active
Python environment alongside its pure-Python wrapper.

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

# CLI Reference

```text
sorrel <DATA_DIR> [OPTIONS]
```

`<DATA_DIR>` is either a directory (phy / Kilosort or SortingAnalyzer)
or a single file (`*.nwb`, `rez.mat`).

## Options

| Flag                | Description                                                                         |
| ------------------- | ----------------------------------------------------------------------------------- |
| `--backend NAME`    | `kilosort` \| `sorting-analyzer` \| `nwb` \| `ks4-rez`. Auto-detected when omitted. |
| `--dat PATH`        | Raw recording. Overrides `params.py:dat_path`.                                      |
| `--sample-rate Hz`  | Overrides `params.py:sample_rate`.                                                  |
| `--channels N`      | Overrides `params.py:n_channels_dat`.                                               |
| `--dtype DTYPE`     | `int16` \| `uint16` \| `int32` \| `float32`. Overrides `params.py:dtype`.           |
| `--offset BYTES`    | Header bytes to skip in the `.dat`.                                                 |
| `--journal PATH`    | SQLite curation log. Default: `<root>/sorrel.sqlite`.                               |
| `--spikeglx-meta P` | Populate sample-rate / channels / dtype from a SpikeGLX `.meta`.                    |
| `--oebin PATH`      | Populate sample-rate / channels / dat path from an Open Ephys `structure.oebin`.    |
| `--export-qc DIR`   | Headless: write `cluster_qc.tsv` + `cluster_qc.json` into `DIR` and exit. No GUI.   |
| `-h`, `--help`      | Print usage.                                                                        |

When `params.py` is present, sample-rate, channels, dtype, offset, and
dat-path all default from it. CLI flags override per-field.

## Examples

```bash
# Auto-detect, defaults from params.py
sorrel my_run/phy

# Force a backend
sorrel my_run --backend sorting-analyzer

# Override sample rate but inherit everything else from params.py
sorrel my_run/phy --sample-rate 25000

# SpikeGLX run without a params.py
sorrel my_run/phy --spikeglx-meta run.imec0.ap.meta

# Run all metrics headlessly and write QC artefacts
sorrel my_run/phy --export-qc out/qc/
```

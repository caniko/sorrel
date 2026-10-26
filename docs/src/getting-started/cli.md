# CLI Reference

```text
sorrel <DATA_DIR> [OPTIONS]            # open the GUI
sorrel <SUBCOMMAND> ...                # headless tools on the same data
```

`<DATA_DIR>` is either a directory (phy / Kilosort or SortingAnalyzer)
or a single file (`*.nwb`, `rez.mat`).

## Subcommands

| Subcommand                                                                              | What it does                                                                                                                                                                                                                     |
| --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `open <DATA_DIR> [OPTIONS]`                                                             | Open the GUI. Same as giving `<DATA_DIR>` with no subcommand.                                                                                                                                                                    |
| `qc <DATA_DIR> --out DIR [OPTIONS]`                                                     | Run every metric on the current curation and write `cluster_qc.tsv` + `cluster_qc.json` into `DIR`.                                                                                                                              |
| `suggest <DATA_DIR> [--json] [OPTIONS]`                                                 | Rank merge and split suggestions; a table by default, `{"merges": [...], "splits": [...]}` with `--json`.                                                                                                                        |
| `apply <DATA_DIR> (--rules FILE \| --curation FILE) [--dry-run] [--save]`               | Label clusters from QC-metric threshold rules, or import a SpikeInterface `curation.json` (labels, `tags`, merges), as one undoable batch; see `docs/src/usage/rules-example.toml`.                                              |
| `auto <DATA_DIR> [--llm] [--gain-to-uv UV] [--no-merge] [--reuse] [--dry-run] [--save]` | Automatic curation with SpikeInterface (merge presets, Bombcell + UnitRefine, optional Claude tiebreak), applied as one undoable batch with provenance and `review:*` tags; see [Automatic curation](../usage/auto-curation.md). |
| `report <DATA_DIR> [--out FILE] [--clusters IDS] [--label L] [--tag PREFIX]`            | Self-contained HTML: one card per cluster with metrics, tags, template, ISI and ACG as inline SVG; `--tag review:` renders the review queue.                                                                                     |
| `share <DATA_DIR> [--port N \| --relay ADDR]`                                           | Host this dataset for a remote curator (see [Remote curation](../usage/remote.md)).                                                                                                                                              |
| `join <CODE> [--direct ADDR \| --relay ADDR] [--into DIR]`                              | Join a shared session as the remote curator.                                                                                                                                                                                     |
| `serve <DATA_DIR> [OPTIONS]`                                                            | JSON-RPC 2.0 over stdio in MCP shape (`tools/list`, `tools/call`): metrics, suggestions, SVG plots, and journaled relabel / merge / split / undo / save, for agents and notebooks.                                               |
| `diff <A.tsv> <B.tsv> [--json]`                                                         | Compare two `cluster_group.tsv` files: confusion table, Cohen's kappa, per-cluster disagreements.                                                                                                                                |
| `journal dump <JOURNAL> [--json]`                                                       | Print every committed journal record, oldest first.                                                                                                                                                                              |
| `journal replay <DATA_DIR> --out DIR`                                                   | Replay the journal and write `spike_clusters.npy`, `cluster_group.tsv`, `curation.json` into `DIR`.                                                                                                                              |
| `export-phy <DATA_DIR> --out DIR [--copy-raw]`                                          | Write the curated data as a complete phy directory into `DIR`, from any backend; see [Exporting a phy directory](../usage/saving.md#exporting-a-phy-directory).                                                                  |
| `clear-cache <DATA_DIR>`                                                                | Remove `<DATA_DIR>/.sorrel/cache/`.                                                                                                                                                                                              |

Every subcommand that takes `<DATA_DIR>` accepts the open options below,
opens the data through the same backend detection, and replays the same
curation journal as the GUI, so headless output always reflects the
current curation.

## Open options

| Flag                | Description                                                                         |
| ------------------- | ----------------------------------------------------------------------------------- |
| `--backend NAME`    | `kilosort` \| `sorting-analyzer` \| `nwb` \| `ks4-rez`. Auto-detected when omitted. |
| `--dat PATH`        | Raw recording. Overrides `params.py:dat_path`.                                      |
| `--sample-rate Hz`  | Overrides `params.py:sample_rate`.                                                  |
| `--channels N`      | Overrides `params.py:n_channels_dat`.                                               |
| `--dtype DTYPE`     | `int16` \| `uint16` \| `int32` \| `float32`. Overrides `params.py:dtype`.           |
| `--offset BYTES`    | Header bytes to skip in the `.dat`.                                                 |
| `--journal PATH`    | Curation journal. Default: `<root>/sorrel.journal`.                                 |
| `--discard-journal` | Delete the existing journal before opening (see [Saving](../usage/saving.md)).      |
| `--spikeglx-meta P` | Populate sample-rate / channels / dtype from a SpikeGLX `.meta`.                    |
| `--oebin PATH`      | Populate sample-rate / channels / dat path from an Open Ephys `structure.oebin`.    |
| `--events FILE`     | PSTH events: `events.csv`, `events.npy`, or an `.nwb` trials table (GUI only).      |
| `-h`, `--help`      | Print usage.                                                                        |

When `params.py` is present, sample-rate, channels, dtype, offset, and
dat-path all default from it. CLI flags override per-field.

`--export-qc DIR` and `--clear-cache` still work as deprecated aliases
for the `qc` and `clear-cache` subcommands.

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
sorrel qc my_run/phy --out out/qc/

# Feed suggestions to a script
sorrel suggest my_run/phy --json | jq '.merges[] | select(.score > 0.7)'

# See what a colleague's journal would do before opening it
sorrel journal dump my_run/phy/sorrel.journal
```

# CLI Reference

```text
sorrel                                 # the GUI, detached, with a launcher
sorrel open <DATA_DIR> [OPTIONS]       # open a dataset as a tab in the GUI
sorrel run [<DATA_DIR>] [OPTIONS]      # the GUI in the foreground, or --headless for agents
sorrel <SUBCOMMAND> ...                # headless tools on the same data
```

`<DATA_DIR>` is either a directory (phy / Kilosort or SortingAnalyzer)
or a single file (`*.nwb`, `rez.mat`).

## Subcommands

| Subcommand                                                                                                                    | What it does                                                                                                                                                                                                                                               |
| ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `open <DATA_DIR> [OPTIONS]`                                                                                                   | Open the dataset as a new tab in the running GUI, or start the GUI detached if none is running. If the dataset is already open, its tab is focused. See [Opening a window](#opening-a-window).                                                             |
| `run [<DATA_DIR>] [OPTIONS]`                                                                                                  | The GUI in the foreground, with logs and Ctrl-C. Hands the dataset to a running window as a tab. `--standalone` always starts a separate window; `--headless` runs the session host with no window.                                                        |
| `status [<DATA_DIR>] [--json]`                                                                                                | The running GUI (pid, tabs), every session host (dataset, kind, pid, port, alive) and the log file of a detached GUI. See [Status](#status).                                                                                                               |
| `call <DATA_DIR> [OPTIONS] <TOOL> [ARGS_JSON]`                                                                                | Run one MCP tool from a shell. See [Call](#call).                                                                                                                                                                                                          |
| `agent-guide`                                                                                                                 | Print a primer for LLM agents, with the live command list, MCP tools and running hosts. See [Agent guide](#agent-guide).                                                                                                                                   |
| `completions <SHELL>`                                                                                                         | Print a completion script for `bash`, `zsh`, `fish`, `elvish` or `powershell`. See [Completions](#completions).                                                                                                                                            |
| `qc <DATA_DIR> --out DIR [OPTIONS]`                                                                                           | Run every metric on the current curation and write `cluster_qc.tsv` + `cluster_qc.json` into `DIR`.                                                                                                                                                        |
| `suggest <DATA_DIR> [--json] [OPTIONS]`                                                                                       | Rank merge and split suggestions; a table by default, `{"merges": [...], "splits": [...]}` with `--json`.                                                                                                                                                  |
| `apply <DATA_DIR> (--rules FILE \| --curation FILE) [--dry-run] [--save]`                                                     | Label clusters from QC-metric threshold rules, or import a SpikeInterface `curation.json` (labels, `tags`, merges), as one undoable batch; see `docs/src/usage/rules-example.toml`.                                                                        |
| `auto <DATA_DIR> [--llm] [--llm-model MODEL] [--gain-to-uv UV] [--no-merge] [--no-unitrefine] [--reuse] [--dry-run] [--save]` | Automatic curation with SpikeInterface (merge presets, Bombcell + UnitRefine, optional Claude tiebreak), applied as one undoable batch with provenance and `review:*` tags; see [Automatic curation](../usage/auto-curation.md).                           |
| `rescue <DATA_DIR> [--apply] [--save] [OPTIONS]`                                                                              | Matched-filter rescue of spikes the sorter missed from existing units (FDR-controlled); a dry run unless `--apply`. See [Spike rescue](../usage/rescue.md).                                                                                                |
| `describe <DATA_DIR> [--json]`                                                                                                | Print a dataset summary like phy's `template-describe`: data file, dtype, sample rate, duration, channels, spikes, clusters per group, templates, PC features, custom labels and the undo history. `--json` prints it as JSON.                             |
| `report <DATA_DIR> [--out FILE] [--clusters IDS] [--label L] [--tag PREFIX]`                                                  | Self-contained HTML: one card per cluster with metrics, tags, template, ISI and ACG as inline SVG; `--tag review:` renders the review queue.                                                                                                               |
| `share <DATA_DIR> [--port N \| --relay ADDR]`                                                                                 | Host this dataset for a remote curator (see [Remote curation](../usage/remote.md)).                                                                                                                                                                        |
| `join <CODE> [--direct ADDR \| --relay ADDR] [--into DIR]`                                                                    | Join a shared session as the remote curator.                                                                                                                                                                                                               |
| `mcp [<DATA_DIR>] [OPTIONS]`                                                                                                  | MCP over stdio for agents and notebooks: metrics, suggestions, SVG plots, journaled relabel / merge / split / undo / save, proposals. Without `<DATA_DIR>` it attaches to the GUI's active tab or the only running host; see [Agents](../usage/agents.md). |
| `mcp tools [--json]`                                                                                                          | List the MCP tools, offline.                                                                                                                                                                                                                               |
| `mcp config [--client claude \| codex \| json] [<DATA_DIR>]`                                                                  | Print how to register sorrel with an agent. See [MCP tools and config](#mcp-tools-and-config).                                                                                                                                                             |
| `diff <A.tsv> <B.tsv> [--json]`                                                                                               | Compare two `cluster_group.tsv` files: confusion table, Cohen's kappa, per-cluster disagreements.                                                                                                                                                          |
| `journal dump <JOURNAL> [--json]`                                                                                             | Print every committed journal record, oldest first.                                                                                                                                                                                                        |
| `journal replay <DATA_DIR> --out DIR`                                                                                         | Replay the journal and write `spike_clusters.npy`, `cluster_group.tsv`, `curation.json` into `DIR`.                                                                                                                                                        |
| `export-phy <DATA_DIR> --out DIR [--copy-raw]`                                                                                | Write the curated data as a complete phy directory into `DIR`, from any backend; see [Exporting a phy directory](../usage/saving.md#exporting-a-phy-directory).                                                                                            |
| `clear-cache <DATA_DIR>`                                                                                                      | Remove `<DATA_DIR>/.sorrel/cache/`.                                                                                                                                                                                                                        |
| `gpu-check`                                                                                                                   | List every GPU adapter and check the compute kernels against the CPU on each; see [GPU drivers](installation.md#gpu-drivers).                                                                                                                              |

Every subcommand that takes `<DATA_DIR>` accepts the open options below,
opens the data through the same backend detection, and replays the same
curation journal as the GUI, so headless output always reflects the
current curation.

The old spellings are removed: `sorrel <DATA_DIR>` is `sorrel open`,
`--headless` is `sorrel run --headless`, `--export-qc DIR` is `sorrel qc
<DATA_DIR> --out DIR`, `--clear-cache` is `sorrel clear-cache`, and `serve`
is `mcp`.

## Opening a window

Bare `sorrel` starts the GUI detached from the terminal, showing a launcher
(Open dataset…, drag-and-drop, recent datasets). If a window is already
running, it is focused. The desktop entry runs this.

`sorrel open` adds a dataset as a tab to the running window, or starts one if
none is running. Each dataset is a tab; Ctrl-O opens another and Ctrl-W closes
the current one.

`sorrel run` is the same GUI in the foreground: the terminal stays attached,
logs go to it, and Ctrl-C stops it. If a window is already running, `run`
hands the dataset over as a tab and returns. `--standalone` always starts a
separate window. `sorrel run --headless <DATA_DIR>` runs the session host and
MCP listener with no window, until Ctrl-C.

## Status

`sorrel status` lists the running GUI (pid, tabs with dataset, dirty and
active), every session host (dataset, kind `gui` or `headless`, pid, port,
alive), and the log file of a detached GUI. `--json` prints the same as JSON.

## Call

`sorrel call <DATA_DIR> [OPTIONS] <TOOL> [ARGS_JSON]` runs one MCP tool through
the running host, or in-process when none is running. It prints the result and
exits non-zero on error. `--raw` prints the JSON-RPC response instead.

```bash
sorrel call ks4/ metrics '{"cluster": 12}'
```

## MCP tools and config

`sorrel mcp tools [--json]` lists the MCP tools without starting a host.

`sorrel mcp config [--client claude|codex|json] [<DATA_DIR>]` prints how to
register sorrel with an agent: a `claude mcp add sorrel -- sorrel mcp` line,
a Codex `[mcp_servers.sorrel]` TOML block, or an `mcpServers` JSON block.

## Agent guide

`sorrel agent-guide` prints a primer for LLM agents, including the live
command list, the MCP tools and the running hosts.

## Completions

`sorrel completions <bash|zsh|fish|elvish|powershell>` prints a completion
script for the shell. For bash:

```bash
sorrel completions bash > ~/.local/share/bash-completion/completions/sorrel
```

## Open options

| Flag                | Description                                                                                                                                                                                    |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--backend NAME`    | `kilosort` \| `sorting-analyzer` \| `nwb` \| `ks4-rez`. Auto-detected when omitted.                                                                                                            |
| `--dat PATH`        | Raw recording. Overrides `params.py:dat_path`.                                                                                                                                                 |
| `--sample-rate Hz`  | Overrides `params.py:sample_rate`.                                                                                                                                                             |
| `--channels N`      | Overrides `params.py:n_channels_dat`.                                                                                                                                                          |
| `--dtype DTYPE`     | `int16` \| `uint16` \| `int32` \| `float32`. Overrides `params.py:dtype`.                                                                                                                      |
| `--offset BYTES`    | Header bytes to skip in the `.dat`.                                                                                                                                                            |
| `--journal PATH`    | Curation journal. Default: `<root>/sorrel.journal`.                                                                                                                                            |
| `--discard-journal` | Set the existing journal aside before opening: renamed to `<journal>.<unix seconds>`, not deleted. On a phy directory, `sorrel_original/` is set aside too (see [Saving](../usage/saving.md)). |
| `--spikeglx-meta P` | Populate sample-rate / channels / dtype from a SpikeGLX `.meta`.                                                                                                                               |
| `--oebin PATH`      | Populate sample-rate / channels / dat path from an Open Ephys `structure.oebin`.                                                                                                               |
| `--events FILE`     | PSTH events: `events.csv`, `events.npy`, or an `.nwb` trials table (GUI only).                                                                                                                 |
| `-h`, `--help`      | Print usage.                                                                                                                                                                                   |

When `params.py` is present, sample-rate, channels, dtype, offset, and
dat-path all default from it. CLI flags override per-field.

## Examples

```bash
# Auto-detect, defaults from params.py
sorrel open my_run/phy

# Force a backend
sorrel open my_run --backend sorting-analyzer

# Override sample rate but inherit everything else from params.py
sorrel open my_run/phy --sample-rate 25000

# SpikeGLX run without a params.py
sorrel open my_run/phy --spikeglx-meta run.imec0.ap.meta

# Run all metrics headlessly and write QC artefacts
sorrel qc my_run/phy --out out/qc/

# Feed suggestions to a script
sorrel suggest my_run/phy --json | jq '.merges[] | select(.score > 0.7)'

# See what a colleague's journal would do before opening it
sorrel journal dump my_run/phy/sorrel.journal
```

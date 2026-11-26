# Coming from phy

Open the directory containing `params.py` rather than the file itself:

```sh
sorrel open my_run/phy
```

Start on a copy if you plan to keep curating the same sort in phy. Sorrel
journals edits as you make them; saving writes back to the opened phy
directory. Sorrel checks for external changes before replaying that history.

This guide describes `trunk`. See [Installation](../getting-started/installation.md)
for differences from 0.4.0. The keys below are defaults; **Primary** means
`Ctrl` on Linux and Windows and `Cmd` on macOS.

| In phy                  | In Sorrel                                                                                                                                                                                                                                                                                     |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Inspect a cluster       | Select it in **Clusters**. Use **Waveforms** for spike shapes, **CCG** for auto- and cross-correlograms, and **ISI** for interspike intervals. In Waveforms, `W` and `Shift+W` cycle modes, and `M` toggles mean-only display while the pointer is over the pane.                             |
| Find merge candidates   | **Similar** ranks candidates against the last cluster selected in **Clusters** when template similarity data is available. Hover a candidate to preview the merge in linked views.                                                                                                            |
| Relabel                 | `Alt+G`, `Alt+M`, `Alt+N`, and `Alt+U` mark the Clusters selection good, MUA, noise, and unsorted. Use `Primary` instead of `Alt` for the Similar selection, or both modifiers to label both selections.                                                                                      |
| Merge                   | Select the target first, add the other cluster(s), then press `G`. The merge keeps the first cluster's ID.                                                                                                                                                                                    |
| Split                   | In **Features**, `Primary`+click adds points to a polygon lasso; `Primary`+drag draws freehand. Press `K` to split off the enclosed spikes.                                                                                                                                                   |
| Undo or revisit an edit | `Primary+Z` undoes. `Primary+Shift+Z` or `Primary+Y` redoes. **History** lets you revisit a different branch, even after saving and reopening.                                                                                                                                                |
| Save or hand off        | `Primary+S` updates `spike_clusters.npy` and `cluster_group.tsv` in the opened phy directory. Use **Export phy** or `sorrel export-phy my_run/phy --out curated_phy` to hand off a separate snapshot without the journal. Add `--copy-raw` if the recipient cannot access the original trace. |

Sorrel does not implement phy's plugin API, custom Features grid dimensions
(`set_grid_dim`), or the in-window IPython console. For scripting, use
[Sorrel's Python API](integrations.md) or [MCP](agents.md).

See the [phy parity ledger](../reference/phy-parity.md) for detailed coverage,
[Keyboard & Mouse](keyboard-controls.md) for more controls, and
[Saving & QC Export](saving.md) for journal and export behavior.

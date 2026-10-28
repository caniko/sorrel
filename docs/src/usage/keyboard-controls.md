# Keyboard & Mouse

Sorrel uses phy's TemplateGUI shortcuts, so phy muscle memory carries
over. Press `h` (or `?`, `F1`, or the `?` button in the top bar) for this
list as an overlay. The overlay is generated from the same keymap table the
app dispatches from. [phy parity](../reference/phy-parity.md) tracks what is
still missing.

"Best" clusters are the ones picked in the cluster table; "similar"
clusters are picked in the Similar pane or stepped through with `Space`.
Views show the best clusters first, then the similar ones.

## Curation

| Key                            | Action                                                                                                |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `G`                            | Merge the selection into the first-selected cluster.                                                  |
| `K`                            | Split off the spikes inside the Features lasso.                                                       |
| `L`                            | Open the snippet bar with `:l ` typed (set a label).                                                  |
| `Alt+G` / `M` / `N` / `U`      | Move the **best** clusters to good / MUA / noise / unsorted.                                          |
| `Ctrl+G` / `M` / `N` / `U`     | Move the **similar** clusters.                                                                        |
| `Ctrl+Alt+G` / `M` / `N` / `U` | Move **all** selected clusters.                                                                       |
| `Ctrl+Z`                       | Undo.                                                                                                 |
| `Ctrl+Shift+Z` / `Ctrl+Y`      | Redo.                                                                                                 |
| `Ctrl+S`                       | Save curated state to disk (see [Saving](./saving.md)).                                               |
| `Shift+Ctrl+K`                 | Split the selection back into its original templates.                                                 |
| `V`                            | Merge mode: stage the selection, add similar candidates (Space, clicks), `G` merges all, `V` cancels. |

A move is journaled as a single entry, so one undo reverts it. Afterwards
the wizard advances as phy's does. Moving the best clusters selects the
next table row. Moving a similar cluster steps to the next similar one.

## Navigation

| Key                         | Action                                                                         |
| --------------------------- | ------------------------------------------------------------------------------ |
| `↓` / `↑`                   | Next / previous best cluster **in table order** (sort + filter respected).     |
| `Home` / `End`              | First / last best cluster.                                                     |
| `Space` / `Shift+Space`     | Next / previous similar cluster; noise and MUA are skipped.                    |
| `Ctrl+Space`                | Select the first 15 similar clusters.                                          |
| `Backspace`                 | Unselect the similar clusters.                                                 |
| `Ctrl+Alt+Space`            | Reset the wizard to the first cluster.                                         |
| `Alt+←` / `Alt+→`           | Pan the trace window back / forward by one window.                             |
| `Alt+Home` / `Alt+End`      | Jump the trace to the start / end of the recording.                            |
| `Alt++` / `Alt+-`           | Narrow / widen the trace window.                                               |
| `Alt+PgDn` / `Alt+PgUp`     | Centre the trace on the selection's next / previous spike.                     |
| `Alt+T`                     | Go to a time (opens `:tg`).                                                    |
| `Alt+O` / `Alt+S` / `Alt+L` | Trace: channel 0 at the bottom / mark every cluster's spikes / channel labels. |
| `Ctrl+Alt+↑` / `Ctrl+Alt+↓` | Trace gain up / down.                                                          |

## Views and snippets

| Key              | Action                                                                       |
| ---------------- | ---------------------------------------------------------------------------- |
| `C`              | Features: fix the current channels, or follow the selection again.           |
| `A` / `Shift+A`  | Amplitudes: next / previous type (template, feature, raw, template feature). |
| `Ctrl+R`         | Time axes use `spike_reorder.npy` (when present).                            |
| `:`              | Snippet bar. Enter runs the snippet, Esc closes it.                          |
| `H` / `?` / `F1` | Toggle the help overlay.                                                     |
| `Ctrl+Q`         | Quit (asks first when there is unsaved curation).                            |
| `Esc`            | Close a dialog, the snippet bar or the help, or clear the lasso.             |

Snippets: `:g` merge, `:k` split, `:c 3 7 10-12` select, `:l name value`
custom label (`:l group good` moves the selection), `:tg 12.5` trace to
12.5 s, `:ts 0.2` 200 ms trace window.

Plain-key shortcuts never fire while a text field has focus. `Ctrl` chords,
`Esc` and `F1` always do.

Undo and redo are first-class records in the curation journal — replaying
the journal reconstructs the same history and redo stacks. The top bar's
Undo/Redo buttons name the command they will revert or re-apply.

After a merge the target cluster is selected; after a split both halves
are. Closing the window with unsaved curation asks whether to save first.

## Cluster table mouse

| Gesture             | Action                                                        |
| ------------------- | ------------------------------------------------------------- |
| Click a row         | Replace selection with this cluster.                          |
| Cmd/Ctrl-click      | Toggle this cluster in the selection.                         |
| Shift-click         | Select the range of **visible rows** from the anchor to here. |
| Click column header | Sort by that column; click again to flip.                     |

Selected rows carry a colour swatch; the same colour marks that cluster in
every view (first selected = blue, second = orange, …).

## Views mouse

| Where    | Gesture                                              | Action                                                                                               |
| -------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Trace    | drag / wheel                                         | Pan in time.                                                                                         |
| Trace    | Ctrl-wheel / pinch                                   | Zoom the window around the pointer (1 ms – 5 s).                                                     |
| Raster   | click                                                | Seek the trace window to that time.                                                                  |
| Features | Ctrl-click                                           | Add a lasso point in that subplot; `Ctrl`-right-click or `Esc` clears; `K` splits the spikes inside. |
| Features | wheel / Ctrl-wheel / Alt-wheel / drag / double-click | Zoom / scale the features / marker size / pan / reset (shared by every subplot).                     |
| Similar  | click a row                                          | Select the primary cluster together with that candidate.                                             |
| Similar  | Cmd/Ctrl-click a row                                 | Add that candidate to the current selection.                                                         |
| Any tab  | drag the tab title                                   | Move the view to another group or split; `×` closes it.                                              |
| Tab bar  | `+`                                                  | Reopen a closed view in that group.                                                                  |

## Journaling guarantee

Every label change, merge, split, and undo/redo is written to the journal
**before** the in-memory session changes. If Sorrel crashes between the
journal write and the next event, the replay on next open recovers the
exact same state.

## Tab menu

Right-click any tab for **Freeze on this selection** (phy's auto-update
toggle: the pane keeps showing the clusters it had, marked 🔒) and **Save
screenshot** (a PNG of the pane in `<dataset>/.sorrel/screenshots/`).
Shift+wheel over Raster, Scatter or Templates cycles the colour scheme for
unselected clusters (also `:cf none|cluster|group|depth|fr`). Drag pans,
the wheel zooms and double-click resets the Amplitudes, Scatter, Tmpl
features and Spike data plots.

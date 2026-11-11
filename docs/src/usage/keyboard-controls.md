# Keyboard & Mouse

Sorrel uses phy's TemplateGUI shortcuts, so phy muscle memory carries
over. Press `h` (or `?`, `F1`, or the `?` button in the top bar) for this
list as an overlay. The overlay is generated from the same keymap table the
app dispatches from. The tables below describe the defaults; **Help** shows
your active bindings. [phy parity](../reference/phy-parity.md) tracks what is
still missing.

Letter keys below are unshifted unless `Shift` is explicitly shown. On
macOS, use Cmd for the primary Ctrl shortcuts, except the physical Ctrl+Tab
dataset-tab navigation noted below.

## Configure shortcuts and settings

Open **Settings** from the launcher, dataset tab bar or top bar. Under
**Keyboard shortcuts**, search an action, remove a binding with its × button,
or choose **Add shortcut** and press the new chord. Multiple alternatives are
supported. **Reset** restores one action; **Restore all shortcut defaults**
restores every action. Conflicting shortcuts in overlapping scopes prevent
saving. Waveforms and CCG shortcuts apply only while their pane is hovered.

**Save settings** applies changes immediately to every tab and pop-out and
writes `~/.sorrel/settings.toml` (`%USERPROFILE%\.sorrel\settings.toml` on
Windows). This file also stores the theme and recent datasets. Set
`SORREL_CONFIG_DIR` to use another settings directory. It is separate from
the host registry/runtime directory (`SORREL_USER_DIR`). Recording-specific
layout, selection and view state remain in `<dataset>/.sorrel/ui.json`.

You can also edit the TOML while Sorrel is closed:

```toml
version = 1

[appearance]
theme = "system" # system, dark or light

[keybindings]
"curation.split" = ["F8"]
"curation.merge" = ["G"]
"app.open_dataset" = ["Primary+O"]
"waveforms.mean" = [] # explicitly unbound
```

Missing action entries inherit their defaults; arrays replace all bindings
for that action. `Primary` means Ctrl on Windows/Linux and Cmd on macOS;
`Ctrl` means the physical Control key. Use named keys such as `Plus`,
`ArrowDown` and `PageUp`. Settings lists the stable action IDs on hover.
Invalid settings cause a visible warning and are never silently overwritten.
Mouse/wheel gestures and native text-editing shortcuts are unchanged.

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

| Key                           | Action                                                                                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `C`                           | Features: fix the current channels, or follow the selection again.                                                                                                        |
| `A` / `Shift+A`               | Amplitudes: next / previous type (template, feature, raw, template feature).                                                                                              |
| `Ctrl+R`                      | Time axes use `spike_reorder.npy` (when present).                                                                                                                         |
| `:`                           | Snippet bar. Enter runs the snippet, Esc closes it.                                                                                                                       |
| `H` / `?` / `F1`              | Toggle the help overlay.                                                                                                                                                  |
| `Ctrl+Q`                      | Quit (asks first when there is unsaved curation).                                                                                                                         |
| `Ctrl+O` / `Ctrl+W`           | Open a dataset / close the active dataset tab.                                                                                                                            |
| `Ctrl+Tab` / `Ctrl+Shift+Tab` | Next / previous dataset tab (physical Ctrl on macOS too).                                                                                                                 |
| `Ctrl+Comma`                  | Open Settings (`Cmd+Comma` on macOS).                                                                                                                                     |
| `Ctrl+Shift+M`                | Maximize the pane under the pointer over the whole view area, or restore.                                                                                                 |
| `F11`                         | Full screen for the window the key is pressed in (main or a popped-out one).                                                                                              |
| `Esc`                         | Close a dialog, the snippet bar or the help, or cancel the lasso and any pending cut without changing the cluster; with nothing else to cancel, restore a maximized pane. |

Snippets: `:g` merge, `:k` split, `:c 3 7 10-12` select, `:l name value`
custom label (`:l group good` moves the selection), `:note text` unit note
(`:note` alone clears it), `:tg 12.5` trace to
12.5 s, `:ts 0.2` 200 ms trace window, `:max` maximize or restore the pane
under the pointer (Features if none yet), `:pop` move that pane to its own
window.

Plain-key shortcuts never fire while a text field has focus. `Ctrl` chords,
`Esc` and `F1` always do.

### Pane-local defaults

These shortcuts apply only while the pointer is over the pane and no text
field has focus. They are configurable in Settings too.

| Pane      | Key                   | Action                                  |
| --------- | --------------------- | --------------------------------------- |
| Waveforms | `W` / `Shift+W`       | Next / previous waveform mode.          |
| Waveforms | `M` / `O` / `Ctrl+L`  | Mean toggle / overlap / channel labels. |
| Waveforms | `Ctrl+↑` / `Ctrl+↓`   | Increase / decrease amplitude scale.    |
| Waveforms | `Ctrl+→` / `Ctrl+←`   | Widen / narrow waveform boxes.          |
| Waveforms | `Shift+↑` / `Shift+↓` | Spread / shrink the probe vertically.   |
| Waveforms | `Shift+→` / `Shift+←` | Spread / shrink the probe horizontally. |
| Waveforms | `R` / `Shift+R`       | Reset scales.                           |
| CCG       | `N`                   | Toggle uniform normalization.           |

Undo moves to the previous edit and redo to the branch you visited last.
An edit after an undo starts a new branch, and the old one stays in the
[History](./history.md) pane. The top bar's Undo/Redo buttons name the
command they will revert or re-apply.

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

| Where                           | Gesture                                                   | Action                                                                                                                                                                                                                                                                                                            |
| ------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Trace                           | drag / wheel                                              | Pan in time.                                                                                                                                                                                                                                                                                                      |
| Trace                           | Ctrl-wheel / pinch                                        | Zoom the window around the pointer (1 ms – 5 s).                                                                                                                                                                                                                                                                  |
| Raster                          | click                                                     | Seek the trace window to that time.                                                                                                                                                                                                                                                                               |
| Raster                          | drag / wheel / double-click                               | Pan / zoom time and rows around the pointer / reset.                                                                                                                                                                                                                                                              |
| Amplitudes, Rate, Drift, Raster | Shift-drag along time                                     | Set the time range (a band; spikes outside fade). Shift-click clears it (not in Raster, where it adds a cluster: use Esc or the chip).                                                                                                                                                                            |
| Amplitudes, Spike data          | hover a point (6 px)                                      | Highlight that spike in every view: ring in the spike views, a time line in Raster, Rate, Drift and Trace.                                                                                                                                                                                                        |
| Trace                           | click a spike tick                                        | Pick that spike (4 px): Features rings it and the pick stays until the next click.                                                                                                                                                                                                                                |
| Rate                            | drag / wheel / double-click                               | Pan / zoom in time (re-binned over what is visible) / reset; `:frmin 10`, `:frmax 600` set the x range in seconds.                                                                                                                                                                                                |
| Templates                       | Ctrl-wheel / pinch / double-click                         | Zoom every template box / reset; the wheel scrolls.                                                                                                                                                                                                                                                               |
| Waveforms                       | Shift-click a channel box                                 | Toggle that channel in the Features channel set.                                                                                                                                                                                                                                                                  |
| Probe                           | Shift-click a site; Ctrl-click / Ctrl-right-click         | Toggle that channel in the Features set; make it A / B.                                                                                                                                                                                                                                                           |
| ISI                             | click the shaded refractory band / a `<1 ms` chip         | Brush the violating spikes (both of each pair < 1 ms) of every shown / that cluster; Esc clears.                                                                                                                                                                                                                  |
| Waveforms                       | Ctrl-click / Ctrl-right-click a channel box               | Make that channel Features channel A / B (phy `select_channel`).                                                                                                                                                                                                                                                  |
| Features                        | Ctrl-click / Ctrl-drag                                    | Ctrl-click adds a lasso node in that subplot; Ctrl-drag draws one freehand (simplified to at most 64 nodes on release, replacing the old lasso); `Ctrl`-right-click or `Esc` clears; `K` splits the spikes inside.                                                                                                |
| Features                        | Drag / double-click / right-click on a lasso              | On a closed lasso: drag a node to move it, double-click an edge to insert a node, right-click a node to delete it (never below three). A plain drag elsewhere moves all projection windows together, not the dots within a window.                                                                                |
| Features                        | **k-means** button / `:km`                                | Lasso the smallest k-means part of the first selected cluster (exact members highlighted, a chip per part to switch); **auto-run** refits as the selection changes. `K` splits.                                                                                                                                   |
| Features                        | Alt+click a spike / Shift+drag a time cell                | Alt+click on a spike (6 px) seeks the Trace to it; elsewhere it picks the amplitude feature. Shift+drag along a time cell sets the time range, Shift+click clears it.                                                                                                                                             |
| Features                        | wheel / Ctrl-wheel / Alt-wheel / drag / double-click      | Zoom each projection's contents about its centre / scale PC features / marker size / move all projection windows together / reset. These gestures do not resize the projection windows; reset returns to the full 4×4 grid.                                                                                       |
| Features 3D                     | drag / Shift-drag (or middle-drag) / wheel / double-click | Orbit / pan / zoom / reset the camera; click a triple in the AXES list (or the combo box) to change the axes; the **pinned** chip holds the focused 2D plot's pair so only its third axis is chosen, **×** on it lists every triple. Inspection only: no lasso here, and drags with Ctrl or Alt held are ignored. |
| Similar                         | click a row                                               | Select the primary cluster together with that candidate.                                                                                                                                                                                                                                                          |
| Similar                         | Cmd/Ctrl-click a row                                      | Add that candidate to the current selection.                                                                                                                                                                                                                                                                      |
| Any tab                         | drag the tab title                                        | Move the view to another group or split; `×` closes it.                                                                                                                                                                                                                                                           |
| Tab bar                         | `+`                                                       | Reopen a closed view in that group.                                                                                                                                                                                                                                                                               |

## Journaling guarantee

Every label change, merge, split, undo/redo and History jump is written to the journal
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

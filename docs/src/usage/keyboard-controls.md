# Keyboard & Mouse

Press `?` (or `F1`, or the `?` button in the top bar) inside Sorrel for
this list as an overlay.

## Labels

| Key | Action                                 |
| --- | -------------------------------------- |
| `G` | Mark active selection as **good**.     |
| `M` | Mark active selection as **MUA**.      |
| `N` | Mark active selection as **noise**.    |
| `U` | Mark active selection as **unsorted**. |

A label key applies to _every_ cluster in the active selection, not just
the focused one, and is journaled as a single entry: one undo reverts the
whole relabel. Label keys never fire while a text field has focus or while
`Ctrl`/`Cmd` is held.

## Navigation

| Key       | Action                                                                |
| --------- | --------------------------------------------------------------------- |
| `J` / `↓` | Select the next cluster **in table order** (sort + filter respected). |
| `K` / `↑` | Select the previous cluster in table order.                           |
| `H` / `←` | Pan the trace window backward by one window.                          |
| `L` / `→` | Pan the trace window forward by one window.                           |

The cluster table scrolls to keep the keyboard selection in view.

## Curation

| Key                               | Action                                                                             |
| --------------------------------- | ---------------------------------------------------------------------------------- |
| `Cmd/Ctrl-G`                      | Merge the selection into the first-selected cluster (also the **Merge N** button). |
| `Cmd/Ctrl-Z`                      | Undo.                                                                              |
| `Cmd/Ctrl-Shift-Z` / `Cmd/Ctrl-Y` | Redo.                                                                              |
| `Cmd/Ctrl-S`                      | Save curated state to disk (see [Saving](./saving.md)).                            |
| `Esc`                             | Cancel the open dialog, close the help overlay, or abandon a lasso.                |

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

| Where    | Gesture              | Action                                                                                                          |
| -------- | -------------------- | --------------------------------------------------------------------------------------------------------------- |
| Trace    | drag / wheel         | Pan in time.                                                                                                    |
| Trace    | Ctrl-wheel / pinch   | Zoom the window around the pointer (1 ms – 5 s).                                                                |
| Raster   | click                | Seek the trace window to that time.                                                                             |
| Features | drag                 | Rectangle lasso; releasing splits the lassoed spikes off. The hit count is shown while dragging; `Esc` cancels. |
| Similar  | click a row          | Select the primary cluster together with that candidate.                                                        |
| Similar  | Cmd/Ctrl-click a row | Add that candidate to the current selection.                                                                    |
| Any tab  | drag the tab title   | Move the view to another group or split; `×` closes it.                                                         |
| Tab bar  | `+`                  | Reopen a closed view in that group.                                                                             |

## Journaling guarantee

Every label change, merge, split, and undo/redo is written to the journal
**before** the in-memory session changes. If Sorrel crashes between the
journal write and the next event, the replay on next open recovers the
exact same state.

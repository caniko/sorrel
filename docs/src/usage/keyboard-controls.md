# Keyboard & Mouse

## Labels

| Key | Action                                 |
| --- | -------------------------------------- |
| `G` | Mark active selection as **good**.     |
| `M` | Mark active selection as **MUA**.      |
| `N` | Mark active selection as **noise**.    |
| `U` | Mark active selection as **unsorted**. |

A label key applies to _every_ cluster in the active selection, not just
the focused one.

## Navigation

| Key       | Action                         |
| --------- | ------------------------------ |
| `J` / `↓` | Select the next cluster.       |
| `K` / `↑` | Select the previous cluster.   |
| `H` / `←` | Pan the trace window backward. |
| `L` / `→` | Pan the trace window forward.  |

## Curation history

| Key                | Action                                                  |
| ------------------ | ------------------------------------------------------- |
| `Cmd/Ctrl-Z`       | Undo.                                                   |
| `Cmd/Ctrl-Shift-Z` | Redo.                                                   |
| `Cmd/Ctrl-S`       | Save curated state to disk (see [Saving](./saving.md)). |

Undo and redo are first-class records in the SQLite journal — replaying
the journal reconstructs the same history and redo stacks.

## Cluster table mouse

| Gesture             | Action                                        |
| ------------------- | --------------------------------------------- |
| Click               | Replace selection with this cluster.          |
| Cmd/Ctrl-click      | Toggle this cluster in the selection.         |
| Shift-click         | Range-extend from the anchor to this cluster. |
| Click column header | Sort by that column; click again to flip.     |

## Journaling guarantee

Every label change, merge, split, and undo/redo is written to SQLite
**before** the in-memory session changes. If Sorrel crashes between the
journal write and the next event, the replay on next open recovers the
exact same state.

# History

Sorrel keeps every edit. The journal is a tree: undo moves back to the
parent, and an edit after an undo starts a new branch next to the old one,
so nothing you undid is lost. The tree survives saving and reopening.

## The History pane

**History** sits at the end of the Assess tab group. **View → History**
reopens it after you close it. It lists the tree oldest first:

- The first row is **the sort as loaded**, before any edit.
- Each edit is a row with who made it (`you`, an agent's name, an
  algorithm or a remote peer), what it did, and the time (UTC).
- A straight run of edits stays in one column. Each later branch is
  indented one step under the edit it left from, and that edit says how
  many branches leave it.
- The current edit is highlighted.
- Saved edits are marked **saved**.

Click a row to go back to it. Sorrel undoes up to the point the two
branches share, then reapplies the edits down to the row you clicked. The
toast names where you went, e.g. `back to: relabel cluster 12 as Good`.
Going back is journaled too, so it survives a reopen.

## Undo and redo

| Key                       | Moves to                       |
| ------------------------- | ------------------------------ |
| `Ctrl+Z`                  | the parent of the current edit |
| `Ctrl+Shift+Z` / `Ctrl+Y` | the branch you visited last    |

An edit after an undo creates a new branch. Redo then follows the new
branch, and the old one is a click away in the History pane.

## Saving and reopening

Saving writes the current edit's state to the phy files and marks it
**saved** in the journal. It does not cut the history. The title-bar chip
reads **unsaved** whenever the current edit is not the saved one, including
after you go back past a save.

On reopen, sorrel replays the journal from the sort as loaded. For phy
directories, the sort as loaded is kept in `sorrel_original/` (see
[Saving](./saving.md#the-journal-and-sorrel_original)), so the saved files
never stand in for it.

## Journal format

The journal is format version 3. Each record is one of:

- an edit, with its parent, actor and time;
- a move of the current edit;
- a save.

Sorrel upgrades a version 1 or 2 journal the first time it opens it.
Undone edits become branches, and the file is rewritten as version 3.
Older sorrel builds refuse a version 3 journal ("not a sorrel journal (bad magic)")
instead of truncating it. Upgrade every sorrel build that opens the same
sort.

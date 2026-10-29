# sorrel for agents

sorrel curates spike-sorted electrophysiology data (phy/Kilosort,
SpikeInterface SortingAnalyzer, NWB, KS4 rez.mat). A dataset has one owner,
its session host: the GUI tab or `sorrel run --headless`. Every edit, by a
human or an agent, is journaled with its actor and is undoable.

## Pick an entry point

- **You have MCP.** Register `sorrel mcp` once; `sorrel mcp config` prints
  the command for your client. Without a DATA_DIR it attaches to the GUI's
  active tab, else the only running host.
- **You only have a shell.** Use `sorrel call <DATA_DIR> <TOOL> '<json>'`. It
  runs one MCP tool through the running host, or opens the dataset in-process
  when none is running. It exits non-zero on a tool error.
- **Batch work.** `describe --json`, `suggest --json`, `qc --out DIR`,
  `apply --rules FILE --dry-run`, `report --tag review:`.

## Review a sort

Read `brief` (level 1), never plots. Write threshold rules over its metric
names, run `rules_eval`, and look only at the margin and disagreement units
(`brief` with `clusters`, level 3 relations, `eval`, `probe`). Then
`rules_apply` and one `decide` for the rest. Act on merges and splits by
handle (`m3`, `s12.1`); never copy spike indices. The MCP `review` prompt
has the whole protocol and the rubric. From a shell:
`sorrel call <DATA_DIR> brief '{"level": 1}'`.

## Find what is running

`sorrel status --json` lists the GUI instance (its tabs: dataset, dirty,
active) and every host (dataset, kind gui|headless, pid, port, alive).
`sorrel open <DATA_DIR>` adds a tab to the running GUI so a human can watch
your edits.

## Rules that protect data

- One writer per journal. If a dataset is open elsewhere, go through that
  host (`mcp`/`call` do this automatically); never pass `--discard-journal`
  to get around a busy journal.
- Edits change only the journal until `save`. `save` writes
  spike_clusters.npy, cluster_group.tsv and curation.json; ask before saving
  someone else's dataset.
- Under **Propose** (set in the GUI), `relabel`/`merge`/`split` queue
  proposals for a human instead of applying. `propose` always queues.
- `review:*` tags mark the queue `sorrel auto` left for a human; clear them
  only when you have settled the question.
- Prefer `--dry-run` and the read tools (`brief`, `rules_eval`, `eval`,
  `probe`) before any edit.

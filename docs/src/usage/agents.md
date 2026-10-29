# Agents & Headless Mode

A sorrel session has one owner, the session host. The GUI, MCP agents and
a remote guest all edit through it. Every edit is journaled with who made
it, so it can be undone and shows up in the [History](./history.md) pane.

```text
sorrel open <DATA_DIR>               # a tab in the running window, or a new window
sorrel run --headless <DATA_DIR>     # host only; stops on Ctrl-C
sorrel mcp [<DATA_DIR>]              # the command an agent launches
```

## Connecting an agent

`sorrel mcp` speaks MCP (JSON-RPC 2.0, one object per line) on stdio. Without
`<DATA_DIR>` it attaches to the GUI's active tab, or else to the only running
host; if there are several, it errors and lists them. With `<DATA_DIR>` it
connects to the host on that directory (the GUI or `run --headless`) when one
is running. Otherwise it opens the dataset itself and serves it until stdin
closes. `sorrel serve` is gone; use `sorrel mcp`.

`sorrel mcp config --client claude` prints the registration line for Claude
Code, `--client codex` a Codex TOML block, and `--client json` an `mcpServers`
entry like this:

```json
{
  "mcpServers": {
    "sorrel": { "command": "sorrel", "args": ["mcp", "/data/run42/ks4"] }
  }
}
```

On Windows, the winget and Chocolatey packages put `sorrel` on `PATH`, so
the same entry works with a Windows path:

```json
{
  "mcpServers": {
    "sorrel": { "command": "sorrel", "args": ["mcp", "C:\\data\\run42\\ks4"] }
  }
}
```

A running host listens on `127.0.0.1` on a random port. It writes the port
and a token to `$XDG_RUNTIME_DIR/sorrel/mcp/<hash>.json` (fallback
`~/.cache/sorrel/mcp/`; `%LOCALAPPDATA%\sorrel\mcp\` on Windows, which only your
account can read), mode 0600 on Unix, and deletes the file when it exits. The file also records the
dataset path and whether the host is a GUI or headless. `SORREL_USER_DIR`
overrides the base directory. A client must send the token first. Only
processes that can read that file can connect.

The client's `clientInfo.name` from `initialize` names the agent. The agent
indicator, the toasts and the History pane use that name, and a split it
makes gets the origin `Sorrel[LLM:<name>]`.

## Tools

| Tool                                                                                                                              | What it does                                                                                                                                                                                      |
| --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `brief`                                                                                                                           | Read the session as compact text. Level 1: header and flagged units. Level 2 (or `clusters`): one card per unit. Level 3: relations (merge components `m<k>`, split candidates `s<cluster>.<k>`). |
| `eval`                                                                                                                            | Counterfactual of a merge (`merge` ids) or a handle (`candidate`), without applying it.                                                                                                           |
| `probe`                                                                                                                           | Closed-grammar computations for hypotheses the brief did not anticipate (`acg`, `ccg`, `compare`, `window`, `split`, `eval`). One compact answer per query; no arbitrary code.                    |
| `rules_eval`                                                                                                                      | Run a threshold rule set (TOML) on every unit. Returns label counts, margin units and disagreements with the auto classifiers.                                                                    |
| `rules_apply`                                                                                                                     | Apply the rule set as relabels in one undo step. Remembered in `.sorrel/review_rules.toml`.                                                                                                       |
| `decide`                                                                                                                          | Many decisions in one call: `[target, action, confidence, reasons]`. One undo step under **Direct**, one proposal per item under **Propose**.                                                     |
| `list_clusters` (TSV), `metrics` (one row, JSON), `suggest`, `evidence`, `attention_queue`, `split_proposals`, `merge_candidates` | Read the session. Each points at the brief for the compact form.                                                                                                                                  |
| `relabel`, `merge`, `split`                                                                                                       | Edit one thing. Journaled and undoable; queued as a proposal instead under **Propose**. `split` takes a handle.                                                                                   |
| `tag`, `undo`, `save`, `export_phy`                                                                                               | Retag a cluster, undo the last edit, save, export a phy directory. Refused under **Propose**.                                                                                                     |
| `propose`                                                                                                                         | Queue one relabel, merge, split or accept for a human, with a confidence and up to three reasons.                                                                                                 |
| `select`, `selection`                                                                                                             | Move the human's selection in the GUI, or read it. These need the GUI.                                                                                                                            |

`render` and `features` are gone. They returned SVG coordinates and raw PC
samples, which a model cannot read. The brief carries the numbers instead:
waveform landmarks, footprint, ACG and amplitude strings. `sorrel report`
still writes the SVGs.

### Handles

`brief` level 3 and `suggest` name every merge component `m<k>` and every
split candidate `s<cluster>.<k>`. Pass the handle as `candidate` to
`split`, `propose` and `eval`, or as the target in `decide`. A handle
records the spikes of the clusters it names and is refused as stale once any
of them changed. Tools never return spike indices.

### `decide`

`decisions` is a list of `[target, action, confidence, reasons]`, or of
objects with the same keys plus an optional `note`.

- `target`: a cluster id, a split handle (`"s12.1"`), a merge handle
  (`"m3"`), or a list of ids for a merge (the first is the target).
- `action`: `good`, `mua`, `noise` or `unsorted` (relabel), `merge`,
  `split` (needs an `s` handle) or `accept` (drops `review:*` tags).
- `reasons`: at most three codes (flag codes or rule names). Free text is
  allowed up to eight words.

Every item is checked first. If one is invalid, nothing is applied and the
reply lists the errors by item. A cluster may appear in only one item. Under
**Direct** the items are one batch, so one `undo` reverts all of them, and
the batch is refused if the session changed since the check. Splits are in
the batch.

### `probe`

`q` is a query string or a list of them; each gets one answer block. Times in
`acg`/`ccg` are milliseconds, in `window`/`split` seconds.

- `acg C [win=50] [bin=1]`: refractory estimate and contamination, burst
  index, a SAX string over log lags and the first six bin rates in Hz.
- `ccg A B [win=50] [bin=1]`: `dip_z`, `sync_z` (always on the standard 1 ms,
  ±50 ms CCG), the +/- lag asymmetry with its z, and a SAX string.
- `compare A B`: template cosine over the union of active channels,
  peak-channel distance in µm, trough-to-peak and half-width deltas,
  amplitude ratio.
- `window C T0 T1`: n, rate, contamination, amplitude and presence of that
  slice against the whole unit.
- `split C by=amp|time|pc|kmeans [at=X]`: registers a new split handle
  `s<C>.<k>` and prints its `eval` line. The minor part is the spikes above
  (`amp`, `pc`) or after (`time`) `at`. Defaults: the GMM boundary, the
  largest amplitude jump on a coarse 20-point scan, PC1 at 0 with the smaller
  side as minor; `kmeans` uses the same proposer as `brief` level 3. Act on the
  handle with `split`, `decide` or `eval`.
- `eval merge A B ..`, `eval s12.1`, `eval m3`: as the `eval` tool.

A parse error in one query lists the grammar and does not stop the others.

## The review protocol

The `review` prompt (`triage` is an alias) walks an agent through this:

1. `brief` level 1: read the mode, the population quantiles and the ledger.
2. Write a rule set for the labels and call `rules_eval`. Refine it until
   few units are on the margin or disagree with the auto classifiers.
3. Look only at margin, disagreement and flagged units: `brief` with
   `clusters`, level 3 for relations, `eval` for a counterfactual, `probe` for a hypothesis neither covers.
4. `rules_apply`, then `decide` for the exceptions, the merges and the
   splits.
5. `brief` level 1 again to confirm.

The cost depends on the number of borderline units, not on the number of
units. The prompt ends with the curation rubric, which names the brief's
descriptors.

## Token budget

- **Levels.** Level 1 is the header and a ledger of flagged units. Level 2
  adds cards. Level 3 adds relations. Ask for `clusters` to get cards for a
  few units, and `budget` (tokens) to cap the reply. What did not fit is
  listed with the handles to fetch it.
- **Mode-delta flags.** The header gives the mode of each metric. A unit
  lists only the flags where it differs from the population, so a clean
  sort is a short reply.
- **Quantiser.** Numbers are rounded to few significant digits and
  log-scaled where useful. Rounding never moves a value across a flag
  threshold (including those of your rule set), so a decision on the
  quantised number is the decision on the raw one.
- **Handles.** A candidate costs a few characters, not a list of spikes.
- **Output.** `decide` takes codes and a batch instead of one `propose` call
  with prose per decision.

Every agent edit adds `src:agent` to the clusters it touches, in the same
undoable step. When someone else changes the session (you in the GUI,
another agent), the agent gets a `notifications/message` after its next
reply. The message names the actor, the change, the clusters and the new
version.

## Direct or Propose

While the GUI serves agents, the top bar shows an agent strip:

- a dot, lit while an agent's last edit is highlighted;
- the names of the connected agents;
- **N to review**, which opens the Proposals pane;
- **Direct | Propose**.

Under **Direct** (the default), `relabel`, `merge` and `split` apply at once.
Under **Propose**, they only queue a proposal. `propose` always queues.

## What you see when an agent edits

- A toast names the agent and the edit, e.g. `codex: merge 1 cluster(s) into 2`.
- The clusters it touched are selected, the same as after your own edit.
- Their rows in the cluster table are tinted for three seconds.
- The edit is a node in the History pane with the agent's name. `Ctrl+Z`
  undoes it like any other edit.

## Proposals

The **Proposals** pane has one card per waiting proposal. A card shows who
proposed it, what it does, the confidence and the reasons:

- Click a cluster chip to select the clusters and look at the evidence in
  the other panes.
- **Accept** applies the proposal as your decision. It adds `src:human`,
  drops `review:*`, and a split keeps the agent as its method
  (`Sorrel[LLM:<name>]`).
- **Reject** applies nothing. A rejected merge tags the clusters
  `notmerge:<other>`, and a rejected split tags the cluster `notsplit`, so
  the suggestion rankers stop offering it.

Every proposal and every decision is appended to
`<DATA_DIR>/.sorrel/proposals.jsonl`.

## Headless mode

`sorrel run --headless <DATA_DIR>` opens the dataset, replays the journal and
serves agents with no window. It prints the port and the command to
connect, then runs until Ctrl-C. Edits land in the same journal the GUI
uses, so opening the GUI afterwards shows them, branches included.

Without a GUI, `select` and `selection` return an error, and agents always
edit directly.

## From a shell

`sorrel call <DATA_DIR> <TOOL> [ARGS_JSON]` runs one MCP tool through the
running host, or in-process when none is running. It prints the result and
exits non-zero on error; `--raw` prints the JSON-RPC response instead:

```bash
sorrel call ks4/ metrics '{"cluster": 12}'
```

`sorrel status --json` lists the running GUI, its tabs and every session host,
for scripts. `sorrel agent-guide` prints a primer for LLM agents, with the live
command list, the MCP tools and the running hosts.

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

| Tool                                                                                                                              | What it does                                                                                    |
| --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `list_clusters`, `metrics`, `suggest`, `render`, `evidence`, `attention_queue`, `split_proposals`, `merge_candidates`, `features` | Read the session: cluster lists, QC metrics, ranked suggestions, SVG plots, PC features.        |
| `relabel`, `merge`, `split`                                                                                                       | Edit. Journaled and undoable; queued as a proposal instead under **Propose** (see below).       |
| `tag`, `undo`, `save`, `export_phy`                                                                                               | Retag a cluster, undo the last edit, save, export a phy directory. Refused under **Propose**.   |
| `propose`                                                                                                                         | Queue a relabel, merge, split or accept for a human, with a confidence and up to three reasons. |
| `select`, `selection`                                                                                                             | Move the human's selection in the GUI, or read it. These need the GUI.                          |

The `triage` prompt walks an agent through the attention queue with the
curation rubric.

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

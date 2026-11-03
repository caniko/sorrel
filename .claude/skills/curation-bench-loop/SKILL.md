---
name: curation-bench-loop
description: Use on the analysis host to run the double-blind improvement loop for sorrel's automated curation (`sorrel auto --llm` and the MCP review agent) against human-curated sorts. Covers adding sessions, running arms, condensing misses, writing one prompt variant, and the dev/holdout acceptance rule.
---

# Curation bench improvement loop

One round: baseline both arms on human-curated sessions, find the largest miss
category, change one thing, rerun, and let a single holdout query decide.
Metrics, the three blinds and the command reference are in the "Double-blind
bench" section of `docs/src/usage/auto-curation.md`; this page only orders the
steps and says who does each.

## Prerequisites

- A `sorrel` binary (`SORREL_BIN` if it is not on `PATH`) and the `claude` CLI.
- uv, for the `auto-llm` arm.
- A private `BENCH` directory outside the repo and a variants directory beside
  it (`$BENCH/../variants`); variants never live in the repo.
- Uploads from `export-curated-sort` unpacked first with
  `export_curated_sort.py unpack PKG OUT_DIR`.

## Agent tiers

<!-- Keep this section's prose in sync with .claude/skills/export-curated-sort/SKILL.md. -->

Roles, not models. `flash` does cheap, mechanical or scouting work; `worker`
does long code or prompt edits and reviews; `lead` is the invoking session, which
decides and runs anything sealed. Name the agents in the skill args, e.g.
`flash=<agent> worker=<agent>`. An agent can be a subagent type, a model alias or
a CLI profile, and is passed through as-is; nothing here assumes a vendor
(`flash=haiku worker=sonnet` in Claude Code is only an example). A role with no
agent named is done by the lead inline.

Delegated agents get only what their task row lists.

## The loop

| #   | Step                                                                                                                    | Tier   | Brief contains                                                                                                   | Returns                                                                               |
| --- | ----------------------------------------------------------------------------------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| 1   | `add` sessions (`--dev` if the lead has seen the truth)                                                                 | flash  | the bench path and the session dirs                                                                              | the counts line                                                                       |
| 2   | `run` the arms, then `score`                                                                                            | flash  | the command lines and the budget                                                                                 | run ids, status, cost, and the path of `report dev`                                   |
| 3   | Condense the dev report into a miss taxonomy                                                                            | flash  | the dev report path only                                                                                         | a table of at most 40 lines: category x count, worst roots, and the card each arm saw |
| 4   | Write ONE variant (`rubric.md` or `review.md`) for the largest miss category, or a code change for a missing capability | worker | the condensed taxonomy, the current prompt (`sorrel agent-guide --prompt review`) and `docs/src/usage/agents.md` | the variant dir and its rationale                                                     |
| 5   | Review a code change                                                                                                    | worker | the diff                                                                                                         | findings                                                                              |
| 6   | Accept or reject on dev, add holdout sessions, run the holdout `report` and adjudication                                | lead   |                                                                                                                  |                                                                                       |

Steps 2 and 3 repeat for the new variant after step 4.

## Sealing

- Flash and worker agents never get `BENCH/.sealed/`, truth directories, the
  source `src/` folder of a session or `phy.log`.
- The holdout is lead-only. Add holdout sessions without inspecting them.
- Any session whose truth the lead has looked at goes in with `add --dev`, so it
  can never reach the holdout.

## Cost

`run --max-budget-usd N` caps each `mcp-review` run (default 10). Ask the user
for a round budget (for example $60) and sum `cost_usd` over the round's
`run.json` files; stop and report when the total passes it, or when any single
run hits its cap.

## Accepting a variant

Rerun dev with the same repeats as the baseline. Accept only if the target
category improves and no other category regresses beyond the spread across
repeats. Then make a single holdout query (`report holdout --baseline ...
--candidate ...`); every holdout query is logged. A paired improvement with no
category regression decides it.

Only an accepted variant becomes a PR, editing `RUBRIC` or `REVIEW_INSTRUCTIONS`
in `crates/sorrel-data/src/serve.rs`. Rejected variants stay in the variants
directory.

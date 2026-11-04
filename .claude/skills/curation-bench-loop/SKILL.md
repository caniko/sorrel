---
name: curation-bench-loop
description: Use on the analysis host to run the double-blind improvement loop for sorrel's automated curation (`sorrel auto --llm` and the MCP review agent) against human-curated sorts. Covers adding sessions, running arms, condensing misses, writing one prompt variant, and the dev/holdout acceptance rule.
---

# Curation bench improvement loop

**The goal is the quality of the human's manual post-sort curation in phy.**
That curation is the training data, and the target is reaching it, not doing
slightly better than the last variant. Most of the distance is capability
sorrel doesn't have yet. On the first dev session the arms recovered 1 of 17
good units because no proposer could isolate the cores the human carved out;
the prompt-only variant made things worse. So each round asks first _what
would sorrel need in order to make the human's decision_: a new proposer,
probe, brief feature, metric, rule or tool. Only after that does it ask how to
word the prompt around it. Expect most accepted rounds to ship a sorrel
feature, with the prompt as the smaller part.

One round:

- baseline both arms on human-curated sessions;
- find the largest gap to the human curation;
- change one thing: a new capability, or a prompt variant that uses one;
- rerun, and let a single holdout query decide.

Metrics, the three blinds and the command reference are in the "Double-blind
bench" section of `docs/src/usage/auto-curation.md`; this page only orders the
steps and says who does each.

## Prerequisites

- A `sorrel` binary (`SORREL_BIN` if it is not on `PATH`) and the `claude` CLI.
- uv, for the `auto-llm` arm.
- For the results DB, a `sorrel` built with `--features dev`; `--db FILE` (or
  `SORREL_DEV_DB`) on the runner turns recording on. Holdout runs are never
  recorded; only the holdout report's aggregates are.
- A private `BENCH` directory outside the repo. Prompt variants are
  publishable (prompt text only) and live in `bench/variants/<name>/`
  (`review.md`, `rubric.md`, `RATIONALE.md`); a variant with no override
  files runs the binary's built-in prompt.
- Uploads from `export-curated-sort` unpacked first with
  `export_curated_sort.py unpack PKG OUT_DIR`.
- A new machine: follow `bench/README.md` (pinned binaries, re-sealing, env).
- Run through the checks in [Pitfalls](#pitfalls-learned-the-hard-way) before
  collecting any data.

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

| #   | Step                                                                                                                                                                                                                                                    | Tier   | Brief contains                                                                                                                                                | Returns                                                                                        |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 1   | `add` sessions (`--dev` for any whose truth the lead has seen). The source folders hold the human truth, so this stays with the lead: a session headed for holdout must never reach a delegated agent                                                   | lead   |                                                                                                                                                               |                                                                                                |
| 2   | `run` the arms on dev, `score`, then `report dev` (a fresh report per variant; a stale one describes the last variant). With `--db`, each dev session, run and score is recorded in the results DB                                                      | flash  | the command lines, the budget, `--split dev` and the DB path                                                                                                  | run ids, status, cost, and the path `report dev` printed                                       |
| 3   | Query the DB (`sorrel dev db query`) and, after new dev data, `sorrel dev distill --out DIR`                                                                                                                                                            | flash  | the DB path, the SQL or the `distill` line and the head of `lineage.md` to report                                                                             | the query output, the distilled `learned_rev` and the head of `lineage.md`                     |
| 4   | Condense the dev report into a miss taxonomy                                                                                                                                                                                                            | flash  | the dev report path only                                                                                                                                      | a table of at most 40 lines: category x count, worst roots, and the card each arm saw          |
| 5   | Close the largest gap with ONE change. First ask what sorrel lacks to make the human's call: a proposer, probe, brief feature, metric, rule or tool; build it. A prompt variant (`review.md`/`rubric.md`) alone only when the capability already exists | worker | the condensed taxonomy, the human split patterns from `truth_roots`, the current prompt (`sorrel agent-guide --prompt review`) and `docs/src/usage/agents.md` | the code change (with tests) and/or the variant dir, and a rationale naming the capability gap |
| 6   | Review a code change, or a fit (the distilled `priors.toml` / `thresholds.toml` against the dev exemplars)                                                                                                                                              | worker | the diff, or the distilled files and `lineage.md`                                                                                                             | findings                                                                                       |
| 7   | Accept or reject on dev. For an accepted candidate: add holdout sessions, `run` baseline and candidate with `--split holdout`, `score --split holdout`, then the holdout `report` (it writes the ledger) and adjudication                               | lead   |                                                                                                                                                               |                                                                                                |

Steps 2 to 4 repeat after step 5, on a binary pinned at the new commit when step 5 changed code.

## Sealing

- The results DB holds derived dev data and holdout aggregates only; flash and
  worker agents may read it. They never get `BENCH/.sealed/`, truth directories, the
  source `src/` folder of a session or `phy.log`.
- The holdout is lead-only. Add holdout sessions without inspecting them.
- Any session whose truth the lead has looked at goes in with `add --dev`, so it
  can never reach the holdout.

## Cost

`run --max-budget-usd N` caps each `mcp-review` run (default 10). Ask the user
for a round budget (for example $60) and sum `cost_usd` over the round's
`run.json` files; stop and report when the total passes it, or when any single
run hits its cap.

## Accepting a change

Rerun dev with the same repeats as the baseline, on a binary pinned at the
change's commit. Accept only if the target gap closes and no other category
regresses beyond the spread across repeats. Then make a single holdout query
(`report holdout --baseline ... --candidate ...`); every holdout query is
logged. A paired improvement with no category regression decides it.

An accepted capability lands as a sorrel PR with its tests. An accepted prompt
variant becomes a PR editing `RUBRIC` or `REVIEW_INSTRUCTIONS` in
`crates/sorrel-data/src/serve.rs`. Rejected variants stay in `bench/variants/`
with their `RATIONALE.md`, and rejected ideas are named in the round's
rationale, so a later round doesn't retry them blind.

Track the distance to the human, not just the delta to the last round: report
good-unit recall and precision against the human curation (1.0 is the target)
in every round summary. When a category stops moving under prompt changes,
that is the signal for a new capability rather than another variant.

## Pitfalls (learned the hard way)

Each item cost a round or leaked an answer once. Run the check before
collecting data; a failed check stops the round.

1. **Kilosort 4 writes `channel_shanks.npy` as float32.** Sorrel before
   2c1991d refuses it, so no KS4 folder opens and the MCP server never starts:
   the agent run "succeeds" with no tools. Check: `sorrel describe <input>`
   prints the cluster count for every new session.
2. **phy rewrites `cluster_KSLabel.tsv`, `cluster_ContamPct.tsv` and
   `cluster_Amplitude.tsv` on save**, with rows only for the clusters that
   exist after curation, so a filtered copy marks which roots the human left
   untouched. `sorrel blind` drops a sorter TSV that doesn't cover every root.
   Check: `ls <input>` shows either all three files or none from phy-saved
   sorts.
3. **The root is `spike_templates.npy`**, which phy never rewrites. Not
   `spike_detection_templates.npy`: Kilosort 4 writes that too, and it differs
   (22 detection templates against 18 final ones on the first dev session).
   Check: `sorrel blind --phy-curated` hashes the input's `spike_clusters.npy`
   against `spike_templates.npy`.
4. **Upload only through `export-curated-sort`.** A raw Drive zip of one
   session was 3.8 GB (Neuralynx `.ncs`, PDFs, plots, `.phy` caches, MATLAB
   exports); what the bench needs packed to 0.8 GB (1.73×; the raw int16 recording compresses 1.81×). Keep `recording.dat` in the
   Kilosort folder (`delete_recording_dat: false`): without it the arms cannot
   check waveforms. Check: one `.sorrel-upload.tar` per recording, and `pack`'s
   summary shows `recording.bin`.
5. **Labs that curate good/noise only need binary scoring.** Scored strictly,
   every `mua` an arm assigns to a human-noise cluster is an exclusion miss.
   `sorrel diff` turns binary on when the truth never uses mua. Check: the
   score's `binary` field matches the lab's practice.
6. **Split shares are measured over kept (non-noise) spikes.** A curator who
   cuts three good cores (5% each) out of a template and calls the rest noise
   did split it. Check: `splits.n_truth` on `sorrel diff TRUTH INPUT --root
INPUT` matches the number of roots with two or more good pieces. A single
   carved core beside a noise remainder is not a split; good-unit recall
   measures that case.
7. **A test that discards a split root has still missed the split.** Labelling
   the whole root noise used to score a vacuous split recall of 1. Check: the
   scorer is at c394a7c or later.
8. **"Cores" and "cleanup" are different human split patterns.** Cores: one to
   three small good units carved out of a noise remainder. Cleanup: the
   majority kept good and a contaminated 10–30% tail dropped as noise. A
   proposer or prompt that knows only one inverts the other: the first core
   proposer turned cleanup roots into a good tail and a noise majority. Check:
   `select verdict, count(*) from truth_roots group by 1` before reading a
   round's misses, and read misses per verdict.
9. **Add every session whose truth the lead has seen with `--dev`.** Looking at
   a truth (a per-root breakdown, `phy.log`, the cluster reports) makes it
   training data. Tuning on a dev session is fine; tuning a constant until one
   dev root comes out right is still fitting, so record such choices in the
   variant's `RATIONALE.md`.
10. **Pin the sorrel and scorer binaries per round**, built from a clean
    `git clone --shared` at a commit, never from the working tree another agent
    is editing. A rebuild mid-round mixes binaries across arms. Score every
    variant of a comparison with the same scorer binary. Check: every run's
    `sorrel_rev` in the DB names a commit, and the round used one scorer.
11. **The first real run on each machine verifies the plumbing.** One
    `mcp-review` run with `--max-budget-usd` set low. Check: `claude.json` shows
    sorrel tool calls (an agent that says "the MCP server isn't connected" cost
    $0.07 and taught nothing), `num_turns` above 3, and a cost under the cap.
12. **Check free space on the bench and temp volumes before runs.** Each run
    copies its input (big files hard-linked, the rest copied), and Claude
    Code's own temp directory filling up silently loses command output. Check:
    `df -h $BENCH "$TMPDIR"` shows room for repeats × arms × the input size.

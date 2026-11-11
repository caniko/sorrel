# Automatic Curation

`sorrel auto` handles the repetitive part of curating a Kilosort sort, using
the classifiers the SpikeInterface project recommends:

```bash
sorrel auto /data/session1 --gain-to-uv 2.34375          # label + merge, journaled
sorrel auto /data/session1 --reuse --save                # re-import, write phy files
sorrel auto /data/session1 --llm                         # an LLM settles the uncertain cases
```

All of its edits go into one batch in the curation journal, so one
**Undo** in the GUI reverts the whole run. Every cluster gets **tags** that
record who decided what. Clusters that still need a person get a
`review:*` tag; the GUI's **review** filter shows only those clusters.

## Pipeline

`sorrel auto` runs `crates/sorrel/src/sorrel_auto.py` (embedded in the binary) using
SpikeInterface ≥ 0.105. The script needs either
[uv](https://docs.astral.sh/uv/) on `PATH`, which installs the pinned
dependencies on first use, or `SORREL_PYTHON` pointing to an interpreter
that already has them. Without either, sorrel falls back to `python3` on
Linux and macOS and to `python` on Windows. The winget and Chocolatey
packages install uv for you.

1. **Load.** It loads the raw recording from `params.py` (or `--dat`,
   `--spikeglx-meta`), applies a 300 Hz high-pass and median CMR, and builds
   the sorting from `spike_times.npy` and `spike_clusters.npy`, so the unit
   ids are the same cluster ids sorrel uses.
2. **Analyzer.** It computes waveforms, templates, PCs, spike locations,
   correlograms, template similarity, quality metrics and template metrics.
   The analyzer is cached in `<DATA_DIR>/sorrel_auto/analyzer` and rebuilt
   only when `spike_clusters.npy` or the recording options change.
3. **Redundant units.** Near-duplicate spike trains found by
   `remove_redundant_units` are labelled `noise` and tagged `redundant`.
   They are not deleted.
4. **Merges.** The `similarity_correlograms` and `slay` presets of
   `compute_merge_unit_groups` run separately. Their groups are combined,
   then merged in the analyzer so the merged units are measured again.
5. **Labels.** Bombcell rules and the two UnitRefine random forests label
   each merged unit. SUA counts as `good`. Bombcell's non-somatic call
   becomes the tag `non-somatic`.
6. **LLM (`--llm`, optional).** An LLM decides the units where the two
   classifiers disagree or UnitRefine is unsure, and the merges that only one
   preset found (see [below](#claude)).
7. **Output.** The script writes `sorrel_auto/curation.json` (SpikeInterface
   curation format v2, keyed by original cluster ids, validated before
   writing) and `sorrel_auto/evidence.json`. Every member of a merge group
   gets the merged unit's label and tags.

Sorrel then imports `curation.json`, the same as running
`sorrel apply --curation`. It also adds `review:split` to clusters where its
own split ranking found a concrete two-way split; SpikeInterface has no
standard automatic split. Importing the same document again changes
nothing.

Bombcell's amplitude threshold is in µV. Pass `--gain-to-uv`
(e.g. `2.34375` for Neuropixels 1.0 at AP gain 500) or `--spikeglx-meta`.
Without one of them, amplitudes stay in ADC units and that one Bombcell
rule is meaningless.

Bombcell and UnitRefine were built on silicon probes with real geometry,
mostly Neuropixels. If any two channels in `channel_positions.npy` are less
than 5 µm apart, as with tetrode sorts that store channel indices instead of
µm, the spatial metrics are meaningless. In that case every unit gets
`review:geometry` and the run is only a starting point. On two such
CINPLA tetrode recordings, `sorrel auto` called 58 of 65 human-kept units noise,
while never keeping a unit the human had marked noise.

## Tags

| Tag                                                       | Meaning                                                                                     |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `src:bombcell`, `src:unitrefine`                          | That classifier contributed to the label.                                                   |
| `src:llm`                                                 | Claude decided the label or the merge.                                                      |
| `src:spikeinterface`                                      | Set by SpikeInterface cleanup (`redundant`).                                                |
| `src:sorrel`                                              | Set by sorrel's own ranking (`review:split`).                                               |
| `src:human`                                               | Someone edited the cluster in the GUI after the run.                                        |
| `agree` / `disagree`                                      | Whether Bombcell and UnitRefine gave the same label.                                        |
| `conf:high` / `conf:mid` / `conf:low`                     | UnitRefine probability ≥ 0.85, in between, or < 0.6.                                        |
| `bc:<metric>`                                             | A Bombcell threshold the unit failed, e.g. `bc:snr`, `bc:sliding_rp_violation`.             |
| `non-somatic`, `redundant`                                | Bombcell's non-somatic waveform call; a near-duplicate spike train.                         |
| `merged:similarity_correlograms`, `merged:slay`           | The merge preset(s) that proposed the merge.                                                |
| `merged:consensus`                                        | Both presets agreed on the merge.                                                           |
| `review:disagree`, `review:low-conf`, `review:llm-unsure` | The label needs a person.                                                                   |
| `review:merge`                                            | The merge (applied or vetoed) needs a person.                                               |
| `review:split`                                            | Sorrel's split ranking has a split proposal for this cluster.                               |
| `review:geometry`                                         | Channel positions look like indices, not µm; every label needs checking.                    |
| `tuned`                                                   | An Auto unit that absorbed a merge or gained rescued spikes. Set by sorrel.                 |
| `split`                                                   | An Auto unit that lost spikes to a split. Set by sorrel; a split child does not inherit it. |
| `external`                                                | Merged, split or created outside sorrel (phy). Set on first open; origin stays Auto.        |

Tags are saved in `cluster_sorrel_tags.tsv`, which phy shows as an ordinary
column, and in the `tags` category of `curation.json`. Any GUI edit to a
tagged cluster (relabel, merge, split) removes its `review:*` tags and adds
`src:human` in the same undoable step.

## Unit origin, lineage and notes

Every unit has exactly one origin:

- `Auto[<sorter>]`: the unit came straight out of the sorter. The sorter name
  is `--sorter NAME` when given, else the one pinned in the saved
  annotations, else detected from the directory (see below).
- `Sorrel[Manual]`, `Sorrel[Algo:<name>]` or `Sorrel[LLM:<client>]`: the unit
  was created inside sorrel.

Only a split creates a unit, so only a split gives a `Sorrel[...]` origin. A
merge keeps the target's id and origin, and the sources disappear. A split
child's origin is the method that made the split:

| Made by                         | Origin                                          |
| ------------------------------- | ----------------------------------------------- |
| Lasso split in the GUI          | `Sorrel[Manual]`                                |
| split_init                      | `Sorrel[Algo:split_init]`                       |
| Suggest, GMM split              | `Sorrel[Algo:gmm]`                              |
| Accepted proposal, agent        | `Sorrel[LLM:<MCP client name>]`                 |
| Accepted proposal, other source | `Sorrel[Algo:<source>]`, e.g. `Algo:kmeans`     |
| `sorrel mcp` split              | `Sorrel[LLM:<MCP client name>]`                 |
| sorrel-py `split(..., by=...)`  | the given method; default `Sorrel[Algo:python]` |
| Tests and fixtures              | `Sorrel[Manual]`                                |

Lineage is stored as direct parents only. A split child's parent is the unit it
was cut from. A merge target's parents are the units merged into it. To see the
full history, follow the parents recursively.

A note is free text on a unit, set with the snippet `:note <text>` on the
selected clusters. `:note` with no text clears it. Notes are journaled and
undo like other edits.

Origins, parents and notes are saved in `cluster_sorrel_annotations.tsv` with
the columns `cluster_id`, `sorrel_origin`, `sorrel_parents` and `sorrel_note`.
It has a row for every non-empty unit, and phy shows the columns. Notes are escaped so each fits one cell.

In `curation.json`, the exclusive `origin` label holds each unit's origin, for
example `Auto[kilosort4]`. Notes are not exported.

The sorter name is detected in this order; the first match wins:

1. `sorter_name` in `spikeinterface_log.json`, in the directory or its parent.
2. `ops.npy`: `kilosort4`.
3. `rez.mat` or `cluster_KSLabel.tsv`: `kilosort`.
4. A SortingAnalyzer folder: the `sorter_name` in `sorting/annotations.json`,
   else `spikeinterface`.
5. A `.nwb` file: `nwb`.
6. Nothing matches: `unknown`.

Override the detected name with `--sorter NAME`, which is a calibration knob.
Precedence: `--sorter`, then the name pinned by a saved `Auto[<sorter>]` row,
then detection. Re-detection never changes a saved name; `--sorter` does, and
the next save pins the new name. A saved `Auto[unknown]` pins nothing.

The journal format is described in [History](./history.md#journal-format).

`origin` is not the same as the `src:*` tags. The origin says where a unit came
from. The `src:*` tags say who labelled or reviewed it, for example `src:llm`
or `src:human`. The two are independent, and the `src:*` tags are unchanged.

## Reviewing

- **GUI:** check **review** above the cluster table. Sort by **tags** to
  list the clusters with the most review tags first. The filter box matches
  tags as well as ids, e.g. `merged`, `src:llm` or `bc:snr`.
- **HTML:** `sorrel report DIR --tag review: --out review.html`.
- **Agents:** `sorrel mcp` adds `list_clusters {"tag": "review:"}`,
  `evidence {"cluster": N}` (the classifier outputs and Claude verdicts) and
  `tag`, so an MCP client can go through the queue with the same evidence.

## Claude

With `--llm`, the uncertain cases go to an LLM. Bare `--llm` uses the
`claude` CLI; `--llm=BACKEND` picks one (the `=` keeps `--llm DIR` unambiguous):

| Backend         | Needs                                     | `--llm-model` default            |
| --------------- | ----------------------------------------- | -------------------------------- |
| `claude` (bare) | Claude Code on PATH, signed in (`/login`) | the CLI's own                    |
| `codex`         | `codex` on PATH, `codex login`            | the CLI's own                    |
| `opencode`      | `opencode` on PATH, `opencode auth login` | the CLI's own (`provider/model`) |
| `api`           | `ANTHROPIC_API_KEY` or `ant auth login`   | `claude-opus-5`                  |

A missing CLI fails the run before any analysis. The CLI backends run one
process per case (8 in parallel) with tools disabled, no session and an empty
working directory: `claude` with no MCP servers, project-only settings and
`--json-schema`; `codex exec` read-only with `--output-schema`; `opencode run`
with every permission denied. `opencode` and `codex` get the system prompt
prepended to the message, and `opencode` has no schema enforcement, so every
reply is validated (label or merge flag, confidence clamped to [0, 1], at most
three reasons). A failed, timed-out (10 minutes) or invalid reply leaves that
case in review.

The LLM is sent **text, not plots**: the same token-lean brief an MCP agent
reads (see [Agents](./agents.md)). `sorrel auto` hands the sidecar its own binary
and open arguments (`SORREL_BIN`, `SORREL_OPEN_ARGS`), so the sidecar runs
`sorrel call DIR brief` and `sorrel call DIR eval` on the dataset exactly as
`sorrel auto` opens it. They run before `sorrel auto` opens the session, so the
journal is not locked; with a host running, `sorrel call` goes through it. If a
call fails, the `--llm` stage fails with its stderr and nothing is applied. A
single case that cannot be judged (a missing card or `eval` line, a failed or
malformed backend reply) is only logged: it gets no verdict and keeps its
`review:*` tags. A merged unit with any member card missing is not asked about
at all. `--llm` cannot be combined with `--reuse`, and with `--discard-journal`
set the journal aside first (`sorrel open --discard-journal`). `--llm=api`
needs `ANTHROPIC_API_KEY` or an active `ant auth login` profile (its stored
credentials, not just the `ant` binary) and fails before the analysis
otherwise. Each run sets an existing `evidence.json` aside as
`evidence.json.<unix seconds>` first, so a rerun cannot overwrite paid
verdicts.

- **System prompt** (cached once for all cases with `--llm=api`; the CLI
  backends re-send it with every case, about 3 KB per case, until a
  persistent-session backend replaces the process per case): the rubric plus the brief
  header, which has the recording, the population quantiles of every metric, the
  flag thresholds and the legend for the cards.
- **One message per unit**: the unit's card row (every metric, the flags it
  trips as a delta against the population, SAX strings for the ACG and for
  amplitude over time) under the card's column line, plus both classifiers'
  outputs. A merged unit gets its members' cards and the merge counterfactual.
- **One message per merge**: the counterfactual line from
  the `eval` tool (`{"merge": [a, b, ...]}`: per-part and merged contamination, presence,
  cutoff, CCG dip and synchrony, amplitude KS) and which presets proposed it.

Amplitudes are in the brief's own units, which its header states (template
units for Kilosort; `--gain-to-uv` scales only the SpikeInterface side). A unit
message is a few hundred bytes where the raw arrays it replaces (25-point
template, 30-bin ACG, 8-channel footprint, deciles; JSON) took 1-2 KB; the
header is paid once with `api` and per case with the CLI backends.

The reply is a JSON verdict (label or merge/no-merge, a confidence and up to
three reasons). Its `model` names the backend: `claude-cli:<model>`,
`codex:<model>`, `opencode:<model>`, or the bare model id for `api`. It is stored in `evidence.json` next to the exact user message
that produced it (`llm_input`, per unit and per merge); the system prompt's
brief header is stored once, as the top-level `llm_header`. A verdict below 0.7
confidence keeps the case in review as `review:llm-unsure`. A confident "do not
merge" vetoes a merge that only one preset found.

## Checking it on your data

Compare `sorrel auto` with your own curation of the same dataset:

```bash
cp DIR/cluster_group.tsv human.tsv                       # a curated copy
sorrel auto DIR --discard-journal --save                 # on a fresh copy of the sort
sorrel diff human.tsv DIR/cluster_group.tsv              # kappa + disagreements
```

Repeat with `--no-unitrefine`, without `--llm` and with `--llm` to see
which stage helps on your recordings. A synthetic self-check lives in
`scripts/test_sorrel_auto.py`
(`uv run --script scripts/test_sorrel_auto.py [--unitrefine]`).

## Double-blind bench

To find out what `sorrel auto --llm` and the MCP `review` agent miss compared
with a person, replay human curations through them and score the difference.

The target is the quality of the human's manual post-sort curation in phy:
those curations are the training data. Sorrel will need new features and ideas
to reach that quality, not only better prompts. On the first dev session both
arms recovered 1 of 17 good units, because nothing in sorrel could propose the
cores the human carved out of under-split templates. Each round of the loop
therefore looks first for the missing capability (a proposer, probe, brief
feature, metric, rule or tool) and only then at the prompt around it.
`scripts/bench_curation.py` (a stdlib uv script) runs the bench; its data lives
in a private `BENCH` directory outside the repo.

**Data.** Sessions curated in sorrel and saved, ideally from scratch: a session
whose history holds a `sorrel auto` or agent batch anchors the human on the
machine's answer, so it is kept in dev and never enters the holdout. About 30%
of the purely manual sessions go to the holdout, chosen by a salted hash. Many
sessions beat many repeats; Neuropixels geometry helps the probe-based checks.
Kilosort sorts curated in phy work too (`add --phy-curated`): the sorter's
clustering comes from `spike_templates.npy`.
Data owners package a phy-curated Kilosort sort with the `export-curated-sort` skill (`.claude/skills/export-curated-sort/`), which keeps only what the bench needs and verifies the compressed raw recording before delivery.

The `curation-bench-loop` skill (`.claude/skills/curation-bench-loop/`) runs the improvement loop below and says which steps a cheap `flash` agent can do, which need a `worker` and which stay with the `lead`. Before collecting data, run the checks in its "Pitfalls (learned the hard way)" section. Each one cost a round or leaked an answer once. To set the bench up on a new machine, including how the sealed holdout survives the move, see `bench/README.md`. Results, prompt variants and lineage live in `bench/` (the results DB through git-lfs).

**Three blinds.**

1. **Curator blind.** `sorrel blind` writes the sorter's own root output (no
   journal, labels, tags, notes, `curation.json` or `.phy/`), checks it against
   the journal's baseline hash, and keeps the human's final state apart as the
   truth. Sessions get opaque ids; the path map and the salt are sealed in
   `BENCH/.sealed/`. Every run works on a fresh hard-linked copy of the input.
2. **Improver blind.** Per-unit misses are only reported on dev. The holdout
   reports aggregates, with a paired bootstrap over sessions, and every holdout
   query is logged.
3. **Blinded A/B adjudication.** Each disagreement goes to a reviewer as two
   verdicts in random order, without saying which is the human's. Units where
   the human loses are `disputed` and drop out of scoring.

```bash
b=~/sorrel-bench; v=variants/base       # a variant dir may hold rubric.md / review.md
bench_curation.py --bench $b add ~/sessions/*                    # blind, seal, split
bench_curation.py --bench $b run --arm mcp-review --variant $v --split dev --repeats 3
bench_curation.py --bench $b run --arm auto-llm   --variant $v --split dev --repeats 3
bench_curation.py --bench $b score                               # sorrel diff per run
bench_curation.py --bench $b report dev                          # misses + the card the arm saw
bench_curation.py --bench $b adjudicate --split dev [--judge claude]
# fill the `choice` column (A, B or both-wrong) of adjudication/dev/cases.tsv, then:
bench_curation.py --bench $b adjudicate --split dev --unmask     # writes disputed/<session>.tsv
bench_curation.py --bench $b report holdout --baseline base --candidate new
```

`SORREL_BIN` picks the sorrel binary. The variant's `rubric.md` and `review.md`
reach both arms as `SORREL_RUBRIC` / `SORREL_REVIEW`, and the prompt revision
is recorded with every run, so a variant needs no rebuild. The `mcp-review` arm
runs `claude -p` with the sidecar's isolation (empty working directory, strict
MCP config holding only `sorrel mcp RUN`, no built-in tools, no session
persistence) and the prompt from `sorrel agent-guide --prompt review`; each
`run.json` records the model, usage, cost and wall time. Reviewers open the
cases' `input_dir` (a copy is safer; stray files there never reach a run). The
LLM judge is dev-only; holdout cases are for humans.

**The improvement loop.**

1. Baseline both arms on every session, 3 repeats each, and score.
2. Adjudicate the disagreements (a human for the holdout, an LLM judge is fine
   on dev), unmask, and score again.
3. `report dev`; pick the largest miss category.
4. The improver changes **one** thing. First it asks what sorrel lacks to make
   the human's decision, and builds that: a new proposer, probe, brief feature,
   metric, rule default or tool. A prompt change (rubric or review
   instructions) on its own is only for gaps the existing capabilities already
   cover. It sees dev reports, the dev truth verdicts and the repo only.
5. Rerun dev. The target category must improve and no other may regress beyond
   the spread across repeats.
6. One holdout query decides: a paired improvement with no category regression.
   The change then lands as a PR: a sorrel feature with its tests, or a prompt
   edit in `serve.rs`. Each round reports the distance to the human curation
   (good-unit recall and precision, where 1.0 means matching the human), not
   just the delta to the previous round.
7. When new sessions arrive, re-split, refresh the holdout and retire the
   ledger.

A smoke test with stub CLIs (`scripts/test_bench_curation.py`) needs neither
data nor a model.

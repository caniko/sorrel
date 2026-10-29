# Automatic Curation

`sorrel auto` handles the repetitive part of curating a Kilosort sort, using
the classifiers the SpikeInterface project recommends:

```bash
sorrel auto /data/session1 --gain-to-uv 2.34375          # label + merge, journaled
sorrel auto /data/session1 --reuse --save                # re-import, write phy files
sorrel auto /data/session1 --llm                         # Claude settles the uncertain cases
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
that already has them.

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
6. **Claude (`--llm`, optional).** Claude decides the units where the two
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
tetrode recordings, `sorrel auto` called 58 of 65 human-kept units noise,
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

With `--llm`, the uncertain cases go to Claude (`--llm-model`, default
`claude-opus-5`; credentials from `ANTHROPIC_API_KEY` or `ant auth login`).

Claude is sent **text, not plots**: the same token-lean brief an MCP agent
reads (see [Agents](./agents.md)). `sorrel auto` hands the sidecar its own binary
and open arguments (`SORREL_BIN`, `SORREL_OPEN_ARGS`), so the sidecar runs
`sorrel call DIR brief` and `sorrel call DIR eval` on the dataset exactly as
`sorrel auto` opens it. They run before `sorrel auto` opens the session, so the
journal is not locked; with a host running, `sorrel call` goes through it. If a
call fails, the `--llm` stage fails with its stderr and nothing is applied.

- **System prompt** (cached once for all cases): the rubric plus the brief
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
template, 30-bin ACG, 8-channel footprint, deciles; JSON) took 1-2 KB, and the
header is paid once, not per case.

The reply is a JSON verdict (label or merge/no-merge, a confidence and up to
three reasons). It is stored in `evidence.json` next to the exact user message
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

# Clustering brainstorm: speed and hard-to-find waveforms

Session 2026-10-28. The ideas, scores and verdicts below are the brainstorm's
proposals, not findings; scores are decision aids. What was built afterwards,
and how it measured, is in the [decision log](#decision-log).

**Focal question.** Which new Sorrel features would make re-clustering much
faster and recover units or spikes that Kilosort 4 leaves behind? The target
cases are low-amplitude units, units buried in MUA or noise, overlapping
spikes, units split by drift, bursting units and sparse units.

**Participants (all AI).**

| ID  | Role                         | Model | Stage       |
| --- | ---------------------------- | ----- | ----------- |
| P01 | Statistics / ML algorithms   | Opus  | independent |
| P02 | GPU / systems                | Opus  | independent |
| P03 | Curator workflow             | Haiku | independent |
| P04 | Cross-field transfer         | Haiku | independent |
| R1  | Adversarial review and rater | Opus  | review      |
| R2  | Blind rater                  | Haiku | review      |
| L1  | Prior-art check (web search) | Haiku | evidence    |
| P05 | Post-evidence ideation       | Haiku | post-check  |

Round 1 produced 44 ideas, grouped into 11 clusters by mechanism. No human
participant generated or rated ideas. That is the main missing perspective.

## Main result

Most recall ideas share one unmet prerequisite, so they should be ordered by
dependency, not by score.

1. **Sorrel cannot yet do residual work correctly.** It never loads
   `whitening_mat*.npy`, `spike_positions.npy` or the KS4 drift shifts. The
   templates live in whitened, drift-corrected space, but the trace mmap is
   raw. Every subtraction or matched-filter idea (C1, C2, C3, C7) is
   therefore wrong until those files are loaded. (Located evidence: grep of
   `crates/`, verified by the facilitator.)
2. **Kilosort 4 already runs template pursuit and resolves most collisions.**
   A native Sorrel detector only adds value below KS4's threshold, and false
   positives concentrate there. So every detector must beat a simple
   baseline first: rerunning KS4 on a channel and time window with a lower
   threshold.
3. **Recall cannot be measured without hybrid injection (C8).** Injection
   comes first, because it is how the gate below gets checked.
4. **Sorrel's unique advantage as a curator** is the human in the loop, the
   journal of decisions, interactive views and the ability to compare
   sorters. The strongest new ideas from the post-evidence round (P05) use
   these, not raw signal processing.

**Gate (applies outside the scores).** A recall feature may only emit
journaled proposals the curator reviews, each with a null or false-positive
estimate. Nothing applies automatically. This rules out "batch-accept
p > 0.95" and "auto-accept the tail" (C9) as originally proposed.

## Proposed sequence (facilitator proposal, not a decision)

| Step | What                                                                                                                                                                                                                                       | Why first                                                                                                    | Smallest check                                                                                      |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| 0    | **Instrumentation:** per-operation timestamps and view dwell time in the journal (R1-new-5)                                                                                                                                                | Speed claims must be measured, not asserted                                                                  | Replay a journal and report the median time per decision                                            |
| 1    | **Mergeable statistics and per-cluster cache keys (P02-02)**                                                                                                                                                                               | Exact, low-risk speed win. Today every cache key includes `journal_head`, so any edit invalidates everything | Fixture assert that CCG(A∪B) = ACG(A) + ACG(B) + CCG(A,B) + CCG(B,A); measure latency after a merge |
| 2    | **Load whitening, spike positions and drift shifts (R1-new-1)**                                                                                                                                                                            | Prerequisite for C1, C2, C3 and C7                                                                           | Whitened mean snippet matches `templates.npy` for a unit                                            |
| 3    | **Virtual hybrid injection (C8)** through a `DataProvider` wrapper; never write to the `.dat`                                                                                                                                              | Measures recall; calibrates C1/C2 thresholds                                                                 | 3 templates at 3 amplitudes are recovered by the mean-snippet shader                                |
| 4    | **Cheap recall and precision proposals:** drift-aware `amplitude_cutoff` per time bin (R1-new-4); ambiguous-assignment proposals from `template_features.npy` (R1-new-3); burst-aware split test (P01-04) with connection tagging (P05-02) | Reuse existing code; show where spikes are lost before building detectors                                    | Count GMM split suggestions that disappear after regressing amplitude on ISI                        |
| 5    | **Drift-fragment merge score (P01-03)**                                                                                                                                                                                                    | Highest-scoring recall cluster; pairwise and reversible                                                      | Curator blind-rates the top 20 pairs against 20 random pairs                                        |
| 6    | **Detectors on the residual (C1, then C2)**, compared against a KS4 window rerun (R1-new-2), after the template-error vs unexplained triage (P05-01)                                                                                       | Only worthwhile if they beat the baseline under injection                                                    | At 0.5–1× threshold, the injected-spike recall of the Sorrel peel exceeds the KS4 rerun             |

## Cluster scores

Two raters (R1, R2) scored each cluster independently from 1 to 5, using
criteria and weights fixed before scoring:

| Criterion   | Weight |
| ----------- | ------ |
| recall      | 0.25   |
| speed       | 0.20   |
| verifiable  | 0.15   |
| feasibility | 0.15   |
| safety      | 0.15   |
| novelty     | 0.10   |

Each cell is the mean of the two raters. The interval spans the widest low and
high either rater gave. The ranking was computed by the skill's
`evaluate_matrix.py` with ±20% one-at-a-time weight sensitivity.

| Cluster                      | Best member                                      | Score [interval] | Rank range | R1 verdict                    | Prior art (L1)                |
| ---------------------------- | ------------------------------------------------ | ---------------- | ---------- | ----------------------------- | ----------------------------- |
| C11 Interactive speed        | P02-02 mergeable statistics                      | 51 [33–66]       | 1–2        | advance P02-02 only           | mixed                         |
| C4 Drift-fragment stitching  | P01-03 shifted template × complementary presence | 51 [25–75]       | 1–2        | advance P01-03 only           | mixed (DARTsort, SpikeSift)   |
| C10 Cross-session priors     | import UnitMatch output                          | 46 [24–71]       | 3–4        | revise: import, don't rebuild | support (UnitMatch)           |
| C9 Learning from the curator | P01-06 in-session active learning                | 45 [23–68]       | 3–4        | revise: drop auto-accept      | mixed (UnitRefine)            |
| C8 Hybrid injection          | virtual `DataProvider` overlay                   | 42 [30–58]       | 5–7        | **advance first** (enabler)   | support (SI hybrid)           |
| C5 Burst-aware split/merge   | P01-04 amplitude~ISI regression                  | 41 [18–61]       | 5–7        | advance (cheap)               | mixed (SLAy)                  |
| C1 Residual matched filter   | P01-01 + second-best-template margin             | 40 [21–68]       | 5–7        | revise                        | mixed (KS4)                   |
| C2 New units from residual   | P03-02 lasso-scoped re-threshold                 | 38 [13–69]       | 8–9        | revise; split off P03-02      | challenge (none located)      |
| C3 Collision recovery        | P02-10 refit only at CCG dips                    | 37 [15–66]       | 8–9        | **park**                      | support (KS4, binary pursuit) |
| C6 MUA density re-clustering | P01-08 recursive dip on computed PCs             | 34 [10–68]       | 10         | advance P01-08; park P01-11   | mixed (ISO-SPLIT)             |
| C7 Sparse units              | P03-07 with held-out templates                   | 31 [3–58]        | 11         | revise; park P04-04           | search incomplete             |

**How to read this.** The ranking is fragile. Every interval overlaps the
others, and 9 of 11 clusters change rank under weight perturbation. The
formula is compensatory, so recall clusters score low because of
false-discovery risk, even though recall is half of the question. Use the
verdicts and the dependency order, not the numbers.

**Where the raters disagree** (gap of one point or more):

| Cluster | Criteria                               | R1 vs R2                                         |
| ------- | -------------------------------------- | ------------------------------------------------ |
| C6      | speed, verifiable, feasibility, safety | R1 higher on each: computed PCs make it testable |
| C3      | recall, novelty                        | R1 lower                                         |
| C4      | safety                                 | R1 higher                                        |
| C10     | feasibility                            | R1 lower                                         |
| C7      | verifiable                             | R1 lower: double-dipping                         |
| C8      | safety                                 | R1 higher                                        |

## Adversarial findings (R1)

- **C1, C2:** recovered spikes may belong to a neighbouring unit, and
  residual hotspots near large units are probably template error.
  Mitigations: require a margin over the second-best template; reject
  candidates with a zero-lag CCG peak against the parent unit; run P05-01's
  refit test before hunting.
- **C4:** KS4 templates are already drift-corrected, so shifting by Δz
  corrects twice. Fragments only appear where nonrigid drift is left over.
- **C5:** an asymmetric 3–15 ms CCG is also the signature of a
  monosynaptic connection. Merging such a pair is a silent error
  (P05-02 addresses this).
- **C7:** building a template from fewer than 100 spikes and scoring those
  same spikes is circular. Build on half A, score half B.
- **C9:** speed gains may just mean the curator stopped looking (automation
  bias). Learning within a session and learning across sessions rest on
  different assumptions.
- **C11:** a 1% reservoir (P02-11) cuts coincident pair counts by about 10⁴,
  which makes ACG and refractory-violation views meaningless. Refractory
  metrics must stay exact.
- **Attractive mainly because of the technology:**

  | Idea   | Method                                     |
  | ------ | ------------------------------------------ |
  | P01-11 | contrastive embedding                      |
  | P04-07 | LSH (a 1000² similarity matrix is trivial) |
  | P04-08 | SPRT (Sorrel does not assign spikes)       |
  | P04-05 | kernel differencing                        |
  | P04-10 | sticky HMM                                 |

  Several P04 items rename existing methods: successive interference
  cancellation is matching pursuit, CFAR is a MAD threshold, and network
  coincidence is a multichannel template.

## Ideas added after the evidence check

| ID       | Idea                                                                                                      | Target             |
| -------- | --------------------------------------------------------------------------------------------------------- | ------------------ |
| R1-new-1 | Load whitening, `spike_positions.npy` and KS4 drift shifts                                                | prerequisite       |
| R1-new-2 | KS4 rerun on a channel/time window at lower threshold, through the `sorrel auto` sidecar, as the baseline | recall (baseline)  |
| R1-new-3 | Reassignment proposals for spikes whose best and second-best templates are close                          | precision          |
| R1-new-4 | `amplitude_cutoff` per time bin: which units lose spikes, and when                                        | recall (diagnosis) |
| R1-new-5 | Decision-time telemetry in the journal                                                                    | measuring speed    |
| P05-01   | Triage residual hotspots into template error (refit with gain/shift) vs unexplained                       | precision          |
| P05-02   | Tag stable-latency, spatially disjoint asymmetric-CCG pairs as putative connections; block merges         | precision          |
| P05-03   | Second sorter through the sidecar; queue only units the sorters disagree on                               | recall             |
| P05-04   | Calibrate merge thresholds on the journal's merge/reject history (selection-biased sample)                | precision, speed   |
| P05-05   | "Stop here" line in the queue from journaled yield per minute; rare-unit flags bypass it                  | speed              |

## Parked

| Idea                              | Reason                                                                      |
| --------------------------------- | --------------------------------------------------------------------------- |
| C3 collision refit                | Measure the zero-lag CCG deficit on 3 KS4 sorts first; close it if under 5% |
| P01-11 contrastive embedding      | Needs a training stack and its own validation                               |
| P04-04 track-before-detect        | No supporting evidence located                                              |
| P04-05, P04-06, P04-10            | Heavy models for a problem a pairwise score covers                          |
| P02-11 approximate views          | Breaks exact refractory metrics                                             |
| P04-07, P04-08                    | No problem to solve at Sorrel's scale                                       |
| P02-12 compute on the remote host | Revisit once a remote GPU workload exists                                   |

## Citations

L1 checked these with web search only and opened no pages. These corrections
apply before anything is cited:

- **Pillow et al. 2013**, PLoS One 8:e62123: authors are Pillow, Shlens,
  Chichilnisky and Simoncelli.
- **Chung et al. 2017**, Neuron 95:1381: the title is "A Fully Automated
  Approach to Spike Sorting". ISO-SPLIT is a separate Magland & Barnett paper.
- **Boussard et al. 2021**, NeurIPS: "Three-dimensional spike localization
  and improved motion correction for Neuropixels recordings".
- **Rossant et al. 2016**, Nat. Neurosci.: hybrid ground truth was not
  confirmed in this paper. It is attributed to Pachitariu et al. 2016
  (KiloSort, NeurIPS).
- **YASS:** Lee et al. 2017, NeurIPS / bioRxiv 151928. The 2020 version is an
  extension.
- **DREDge:** Windolf et al., Nat. Methods 2025. The DOI comes from a search
  summary only.

Verified as cited:

| Work                | Venue                |
| ------------------- | -------------------- |
| Kilosort4           | Nat. Methods 2024    |
| UnitMatch           | Nat. Methods 2024    |
| CEED                | NeurIPS 2023         |
| SpikeForest         | eLife 2020           |
| Shelly 2007         | Nature               |
| Yoon 2015           | Sci. Adv.            |
| Alard & Lupton 1998 | ApJ                  |
| Adams & MacKay 2007 | arXiv                |
| VBx                 | CSL 2022             |
| Fee 1996            | J. Neurosci. Methods |
| Hellerstein 1997    | SIGMOD               |
| Bansal 2004         | Mach. Learn.         |

Not finding prior art in a bounded search does not show that an idea is novel.

## Decision log

- 2026-10-28: the user chose in-place spike addition. Implemented W1–W5 from the plan; KS4 whitening I/O, collision refit, active learning, UnitMatch import and the other pruned items were dropped as low value. New units from the residual are deferred.
- Results:
  - incremental suggestions 5.7× faster;
  - rescue verified end to end on a CINPLA KS2 copy: 661 spikes added, phylib loads the result, arrays aligned;
  - hybrid recall varies by unit (see `usage/rescue.md`).
- Open:
  - choice of null (channel shift vs sign flip);
  - low feature-calibration R²;
  - a labelled Neuropixels KS4 sort for real recall.

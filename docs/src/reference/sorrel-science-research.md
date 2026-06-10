# Sorrel Scientific Correctness Research Dossier

> Evidence gathered 2026-06-10 at commit `d15ccf3` (trunk). Scope: the
> electrophysiology / spike-sorting compute kernels and quality metrics
> in `sorrel-compute` and the `sorrel-data` metric consumers. Every
> algorithm below was read in full and checked against its canonical
> reference; several were brute-forced against oracle implementations.
>
> **Update (same day, post-audit):** the confirmed bugs, the numerical
> concerns, and most fidelity items below have been fixed in the working
> tree — Hill contamination `N²` (with a golden regression test), the
> cutoff↔completeness contract, GMM logsumexp, the presence-bin
> off-by-one, the bimodality-coefficient moment inconsistency,
> `filter.rs`/`drift.rs` scope doc-hardening, the CCG even-bins
> enforcement (#6), NN-isolation in/out balancing (#7, with a
> background-stability golden test), and removal of the misleading dead
> `dip_statistic` alias (#9). See the per-finding **[FIXED]** markers.
>
> **Update 2 (net-new algorithms now implemented):** the stretch metric
> suite has been added and wired in — the real **Hartigan & Hartigan
> (1985) dip** (faithful port of Maechler's `diptest` C reference,
> `distribution::hartigan_dip`, now feeding the split suggester and its
> UI column), the **Llobet et al. (2022) sliding-refractory minimum
> contamination** (`metrics::min_contamination_sliding_refractory`),
> **LDA d-prime** and **simplified silhouette** (added to
> `IsolationMetrics`, the rkyv cache bumped to `ALGO_VERSION = 2`, and the
> QC TSV/JSON export), and **template SNR** (`snippets::template_snr`).
> Each ships with property/golden tests. Workspace green
> (clippy `--deny warnings`, 461 tests). The only intentionally-skipped
> item is renaming `sliding_refractory_contamination` (the windowed-Hill
> UI plot), which already documents itself honestly.

## Goal And Trigger

The user asked to "improve the scientific aspect of the project" with
research-first discipline. Sorrel is a manual-curation GUI for
spike-sorted electrophysiology; its scientific value rests on the
correctness of its quality metrics (contamination, isolation, presence,
amplitude cutoff, drift) and its DSP/statistics kernels (filtering, CMR,
LTTB, correlograms, GMM, covariance/Mahalanobis). This dossier records
an evidence-backed audit of those kernels so a follow-up session can fix
the confirmed defects and close the validation gap without re-deriving
the formulas.

## Current Reality

The numerical *infrastructure* is unusually solid — covariance uses a
two-pass Bessel-corrected estimator, Mahalanobis distance is correct,
LTTB matches Steinarsson 2013 bucket-for-bucket, the median is correct
under brute-force testing, snippet templates accumulate in `f64`, and
the χ² survival function is a faithful Numerical-Recipes incomplete-gamma
implementation. The defects are concentrated in a few **metric formulas**
and one **cross-cutting validation gap**:

- **The headline quality metric is wrong.** `refractory_contamination`
  (`crates/sorrel-compute/src/metrics.rs:149-171`) implements the Hill
  et al. (2011) estimator with the denominator `n · 2 · t_ref / T` —
  **missing a factor of N**. Hill's false-positive fraction is
  `f_p ≈ N_viol · T / (2 · t_ref · N²)`. The code returns a value ~N×
  too large (for N in the thousands it routinely exceeds 1.0), and the
  consumer `quality.rs:111` computes
  `contamination_score = (1 - raw_contamination).clamp(0,1)`, so the
  contamination dimension **saturates to 0 for essentially every real
  cluster**. This is the single most consequential scientific defect.

- **No test pins any metric's numeric value.** There is no
  golden-value or phy/SpikeInterface-parity test for *any* metric
  magnitude. `crates/sorrel-data/tests/synthetic_ground_truth.rs`
  checks only the *directional* behaviour of the merge/split suggesters
  (`merges not empty`, `dip_z > 2.0`, `score > 0.5`); the in-crate unit
  tests assert sanity bounds like `c > 0.0` (`metrics.rs:354-361`).
  Nothing would have caught the Hill scale error, and nothing guards
  the other metrics against future regressions.

- **A consumer contract is internally inconsistent.**
  `quality.rs:119` computes `completeness_score = 1 - raw_cutoff*2`,
  assuming `amplitude_cutoff` maxes at 0.5 — but
  `distribution.rs:204-280` permits its return to climb toward 1.0
  (only a saturating clamp, no 0.5 cap like the Allen routine). When
  cutoff > 0.5, completeness goes negative and clamps to 0.

- **Several functions are named/documented for an algorithm they do not
  implement.** `dip_statistic` returns the bimodality coefficient, not
  Hartigan's dip (`distribution.rs:301`, self-disclosed);
  `sliding_refractory_contamination` is windowed-Hill, not the Llobet
  et al. (2022) minimum-contamination sliding estimator its name evokes
  (`drift.rs:102`); `amplitude_cutoff`'s doc claims equivalence to
  "Hill 2011 §3.2" but it is a coarse mirror-tail approximation
  (`distribution.rs:205`); the `drift.rs` metrics are amplitude-QC
  heuristics, not Kilosort2.5/dredge spatial-motion estimation.

- **No PCA exists in-tree.** PC features are precomputed upstream
  (Kilosort) and only *extracted* in `feature_subspace.rs`; there is no
  eigendecomposition/SVD/whitening anywhere. Any roadmap item assuming
  sorrel computes its own feature subspace is mistaken.

## Evidence Inventory

| Evidence | File:line / command | What it proves |
|---|---|---|
| Hill contamination missing N² | `metrics.rs:166` `denom = n * 2.0 * rp_s / total_s` | Returns `N_viol·T/(2·t_ref·N)`, ~N× the Hill fraction `N_viol·T/(2·t_ref·N²)` |
| Contamination dimension dead | `quality.rs:111` `(1.0 - raw_contamination).clamp(0,1)` | Inflated raw value saturates score to 0 for real clusters |
| Cutoff↔completeness contract mismatch | `quality.rs:119` `1.0 - raw_cutoff*2.0` vs `distribution.rs:279` clamp to 1.0 | `completeness_score` can be driven negative then clamped to 0 |
| No numeric metric tests | `rg 'refractory_contamination\|isolation_distance\|l_ratio\|amplitude_cutoff' crates/**/tests` → none assert values | No golden/parity coverage; scale bugs invisible |
| Suggester tests are directional only | `synthetic_ground_truth.rs:177-252` (`!merges.is_empty()`, `dip_z > 2.0`) | Validates pipeline direction, not metric magnitude |
| Unit tests assert sanity only | `metrics.rs:354-361` (`assert!(c > 0.0)`) | Would not catch an N× scale error |
| presence last-bin off-by-one | `metrics.rs:188-191` (`idx < n_bins` drops `idx==n_bins`); same `drift.rs:81-85` | Boundary spike at `t==total` silently dropped; biases ratio low |
| GMM E-step no logsumexp | `gmm.rs:96-104` linear-domain `pa+pb`; `resp=0.5` on `total==0.0` | Far-outlier underflow fabricates a 0.5 posterior, biasing M-step |
| BC mixes corrected/uncorrected moments | `distribution.rs:116-130` corrected `3(n-1)²/((n-2)(n-3))` with uncorrected `g1`,`g2` (lines 65,73) | Sarle BC internally inconsistent for small n |
| CCG even-bin assumption | `ccg_analysis.rs:54` `half = bins/2` vs `correlograms.rs:277` `ceil(2*window/bin)` | Bin count even only if `2*window % bin == 0`; else zero-lag center off by half a bin |
| CCG right-edge asymmetry | `metrics.rs:46-49` `dt==+max_lag → idx=bins` clamped to `bins-1` | Extreme positive lag folded into last bin; one-bin asymmetry vs `-max_lag` |
| `dip_statistic` ≠ Hartigan dip | `distribution.rs:301-310` returns `bimodality_coefficient` | Name/scale mismatch (Hartigan dip ∈ ~[0,0.25]; BC ∈ [0,1]) |
| `sliding_refractory` ≠ Llobet | `drift.rs:102-143` re-applies (buggy) Hill per window | Not the Llobet 2022 min-contamination estimator |
| NN isolation unbalanced | `metrics_iso.rs:194-258` strides only in-cluster queries, never subsamples background | Deviates from balanced Chung 2017 / SpikeInterface nn_hit_rate |
| Causal display filter | `filter.rs:79-99` single forward-pass DF2T biquad; only caller `sorrel-render/src/buffers.rs:209-226` | Group delay OK for display; must never feed spike-time measurement |
| No PCA in-tree | `rg 'eigen\|jacobi\|svd\|whiten\|principal_component' crates/` → none; `feature_subspace.rs:153,175` extracts precomputed `feats[pc*n_chans+ch]` | PCs come from upstream sorter |
| Covariance correct | `linalg.rs:65-91` `1/(n-1)`, two-pass, symmetrized | Confirmed-correct sample covariance |
| Mahalanobis correct | `linalg.rs:170-186` `(x-μ)ᵀΣ⁻¹(x-μ)`, `.max(0.0)` guard | Confirmed correct |
| LTTB correct | `lttb.rs:39-70` brute-forced vs Steinarsson 2013 oracle, all `(n,threshold)<2000` | Bucket boundaries + triangle area exact; no underflow/div-by-zero |
| Median correct | `cmr.rs:38-53` brute-forced 200k random arrays vs sort oracle | `select_nth_unstable` tie-handling exact for even n |
| L-ratio correct | `metrics_iso.rs:179-181` `Σ χ²-sf(D²,df=d) / N_in` | Matches Schmitzer-Torbert 2005 (initially suspected, confirmed correct) |
| Isolation distance correct | `metrics_iso.rs:173-178` N-th nearest non-cluster Mahalanobis², guarded `n_out≥n_in` | Matches Schmitzer-Torbert 2005 |

## Findings By Severity

### Confirmed bugs

1. **[FIXED] Hill contamination missing N² (`metrics.rs`).** Now
   `f_p = viol * total_s / (2.0 * rp_s * n * n)` accumulated in `f64`,
   clamped to `[0, 1]` (the `f_p > 0.5` "no real root" case saturates to
   1.0). Doc rewritten to state the estimator and the `N²` rationale.
   Guarded by the new golden test
   `refractory_contamination_recovers_known_fraction`, which injects a
   known violation count and asserts the recovered fraction matches the
   analytic Hill value — it fails under the old `÷N` formula.
   **Was highest priority** — the only finding that corrupted a shipped
   composite dimension.
2. **[FIXED] `completeness_score` contract.** `amplitude_cutoff` now
   clamps to `[0, 0.5]` (Allen convention: a hard low-amplitude cutoff
   masks at most half the distribution), so the consumer's
   `1 - cutoff*2` rescale in `quality.rs` is valid and bounded. Both
   docs note the shared 0.5 ceiling.

### Numerical concerns

3. **[FIXED] GMM E-step lacks logsumexp (`gmm.rs`).** E-step and final
   log-likelihood now run in the log domain via `log_normal`:
   responsibility is `1/(1+exp(lb−la))` and the per-point LL is the
   logsumexp of the two log-weighted densities — no more fabricated
   `resp=0.5` on underflow.
4. **[FIXED] Presence-bin off-by-one (`metrics.rs`).** `presence_ratio`
   now clamps the bin index to `n_bins-1` so a spike at the recording
   end is counted, not dropped. (`drift.rs::presence_cv` uses the same
   pattern but its `idx < n_bins` guard is harmless for a CV of counts;
   left as-is.)
5. **[FIXED] Bimodality coefficient mixes corrected/uncorrected moments
   (`distribution.rs`).** Now converts the uncorrected `g1`/`g2` to the
   bias-corrected `G1`/`G2` that the SAS small-sample correction term
   assumes, so the conventions are consistent for small clusters.
6. **[FIXED] CCG even-bins convention.** The one path feeding
   `analyse_refractory_dip` (`suggest.rs`) now rounds `ccg_bins` up to
   even via `(n + 1) & !1`, so zero lag always lands on the
   `half-1`/`half` boundary and the refractory window `[half-r, half+r)`
   stays symmetric; `analyse_refractory_dip` gained a `debug_assert` on
   even bins to catch any future caller. (The extreme-lag right-edge
   clamp in `cross_correlogram` affects only pairs at *exactly*
   `±max_lag` samples — far from the central dip — and was left as-is to
   avoid churning a well-tested path for a negligible-impact edge case.)

### Fidelity / documentation mismatches

7. **[FIXED] NN isolation unbalanced (`metrics_iso.rs`).** Now builds a
   balanced neighbour pool — every in-cluster spike plus a
   deterministically strided background subsample sized to `~n_in` —
   before counting k-NN hits, matching SpikeInterface's balanced
   `nn_hit_rate`. Guarded by `nn_isolation_is_stable_under_background_growth`,
   which asserts the score barely moves when the background grows 10×.
8. **[ADDRESSED] Llobet sliding-refractory minimum now implemented.** The
   genuine Llobet et al. (2022) minimum-contamination-over-`t_ref`
   estimator is added as `metrics::min_contamination_sliding_refractory`
   (sweeps candidate refractory periods, returns the smallest Hill
   fraction). The original `sliding_refractory_contamination` (a
   *time-window* sweep for the UI plot) is retained and unchanged — it
   honestly documents itself as windowed-Hill; only a possible rename
   remains a user decision.
9. **[ADDRESSED] Real Hartigan dip now implemented.** The misleading dead
   `dip_statistic` alias was removed, and a faithful port of Hartigan &
   Hartigan (1985) — Maechler's `diptest` C reference — was added as
   `distribution::hartigan_dip` (`[0, 0.25]`, 0 for unimodal). It now
   corroborates the bimodality coefficient in the split suggester
   (`amp_dip` field, blended via `max`) and shows in the UI split table.
10. **`amplitude_cutoff` doc overclaims Hill-2011 equivalence
    (`distribution.rs:205`).** Soften the doc and/or implement the
    Allen/Hill reflected-Gaussian-tail estimator with the 0.5 cap.
11. **[FIXED] `filter.rs` causal display filter — doc hardening.** Module
    doc now states the single-pass biquad has frequency-dependent group
    delay and must never feed a spike-time/latency/alignment path (those
    need zero-phase filtfilt).
12. **[FIXED] `drift.rs` scope.** Module doc now states these are
    per-cluster amplitude-QC indicators, not a Kilosort/dredge spatial
    motion estimate.

### Confirmed correct (do not "fix")

Covariance (N−1, two-pass, symmetrized), Mahalanobis, χ² survival via
incomplete gamma, LTTB, median tie-handling, snippet `f64` template
accumulation, ACG zero-lag self-exclusion, L-ratio normalization (÷N_in,
df=d), and isolation-distance N-th-neighbor selection were all verified
against their canonical definitions (several by brute force). The
prior dossier's claim that PCA/eigensolvers exist is wrong — none do.

## Work That Should Survive

- The Hill-contamination correction and its `f64` `N²` requirement —
  this is the load-bearing fix.
- The binding constraint that `filter.rs` is **display-only**: any future
  spike-time/latency feature must add a separate zero-phase filtfilt path.
- The fact that PC features are **upstream-provided**, not computed here —
  governs any feature-space metric work.
- The metric→score consumer contracts in `quality.rs` (each `*_score` is
  a `(1 - raw).clamp(0,1)`-style transform of a raw metric); these must
  stay in agreement with each raw metric's documented range.

## Blockers And Missing Artifacts

- **No reference oracle for metric values (validation blocker).** To
  prove a metric is correct (not just self-consistent), a golden dataset
  with known contamination/isolation is needed. Producer: this repo —
  extend `synthetic_ground_truth.rs` with clusters of *known* injected
  contamination fraction and assert the recovered `f_p` matches within
  tolerance; and/or port a handful of SpikeInterface/phy metric outputs
  on a small fixture as parity golden values (mirroring how
  `phy_parity.rs` already pins curation semantics). Validation: those
  tests fail on the current Hill formula and pass after the N² fix.
- No other foundational inputs are missing; all source is present and
  builds/tests green in the Nix dev shell.

## Risks And Constraints

- **Fixing the Hill formula changes user-visible scores.** Clusters that
  currently show `contamination_score = 0` will move to meaningful
  values; any saved thresholds, screenshots, or user expectations
  calibrated against the broken metric will shift. This is a
  behaviour-changing correctness fix, not a silent refactor — note it in
  `CHANGELOG.md` and ideally bump a metrics version if one is exposed.
- **Metric changes ripple into the merge/split suggesters.** The CCG
  binning and contamination fixes feed `analyse_refractory_dip` and the
  suggesters; re-run `synthetic_ground_truth.rs` and confirm the
  directional assertions still hold after each change.
- **`f32` vs `f64` discipline.** The `N²` term and any long count
  accumulations must be `f64`; several metrics currently take
  `len() as f32` early (`metrics.rs:158`).
- **Ordering:** land the validation harness (or at least the
  known-contamination golden test) *alongside or before* the Hill fix so
  the fix is proven, not just plausible.
- **Pre-publish timing.** These are scientific-correctness fixes; landing
  them before the first crates.io publish (see the companion
  `sorrel-improvement-research.md` dossier) avoids shipping a 0.1.0 with a
  known-wrong contamination metric.

## Candidate Next Steps

1. **[DONE] Fix Hill contamination + add a known-contamination golden
   test** — fixed in `metrics.rs` with the golden test
   `refractory_contamination_recovers_known_fraction` (placed in the
   crate unit tests rather than `synthetic_ground_truth.rs`, so it runs
   without a `Session` fixture).
2. **[DONE] Reconcile the cutoff↔completeness contract** — `amplitude_cutoff`
   capped at 0.5 in `distribution.rs`.
3. **[DONE] Add logsumexp to the GMM E-step** — `gmm.rs`.
4. **[DONE] Fix presence-bin off-by-one and the BC moment inconsistency**
   — `metrics.rs` presence clamp, `distribution.rs` bias-corrected G1/G2.
5. **[DONE] Settle the CCG binning convention** — `suggest.rs` rounds
   `ccg_bins` up to even; `analyse_refractory_dip` asserts even bins.
   Suggester tests re-run green.
6. **[DONE] Remove the misleading `dip_statistic` alias** (dead code).
   The `amplitude_cutoff` doc was also corrected (0.5 cap) under step 2.
   `sliding_refractory_contamination` left as-is (honest Hill doc).
7. **[DONE] Balance NN isolation sampling** (`metrics_iso.rs`) — in/out
   balanced pool, with the `nn_isolation_is_stable_under_background_growth`
   golden test.
8. **[DONE] Harden `filter.rs` / `drift.rs` rustdoc** to scope them
   (display-only; QC-only).
9. **[DONE] Fill the missing standard metrics** — the real Hartigan dip
   (`hartigan_dip`), Llobet sliding-refractory minimum
   (`min_contamination_sliding_refractory`), LDA d-prime and simplified
   silhouette (on `IsolationMetrics`, cache `ALGO_VERSION = 2`, QC
   export), and template SNR (`template_snr`). The dip feeds the split
   suggester and its UI column. All ship with tests.

## Open Decisions For The User

- **Rename `sliding_refractory_contamination`?** It is the windowed-Hill
  UI plot (honestly documented), distinct from the new
  `min_contamination_sliding_refractory` (Llobet). The name is still
  slightly suggestive of Llobet; rename for clarity, or leave it since the
  UI relies on it and the doc is honest?
- **Wire d-prime / silhouette into the composite quality score and UI?**
  They are computed, cached, and exported, and already contribute to
  `IsolationMetrics::score()`. A dedicated UI display (beyond the QC
  export) is optional polish.
- **How far to chase phy/SpikeInterface numeric parity?** The metrics are
  algorithmically faithful and property-tested; pinning exact values
  against SpikeInterface fixtures is a larger, optional effort. Where is
  the line for 0.1.0?

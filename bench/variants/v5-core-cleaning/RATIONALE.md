# v5-core-cleaning: distinguish a cleaning boundary from two-neuron separation

ONE change: review protocol only. Keep the built-in rubric, binary
43bc77741014eaff49864959c221f475b3b3fd1e, model, scorer and three repeats.
The scientific code is identical to the unaccepted v4 experiment; no new
capability or threshold is added in this prompt round.

V4 supplies a new core handle, yet its three MCP repeats never apply it and
still recover only 1/17 good units. A dev diagnostic offers a clean core beside
a dirty remainder with negative cross-part dip_z. Such a dip can be evidence
against two separately good neurons without ruling out removal of contaminated
tails from one neuron. This experiment makes that distinction explicit and
requires post-split evidence and undo when a carved part is not truly clean.

Unlike historical v1/v2/v3 core prompts, this protocol does not use the dip
between a clean part and a contaminated remainder as an unconditional split
veto. It does not waive the good-unit rubric, automatically label remainders
noise, name root IDs, use human unit counts, or fabricate thresholds/cuts.
At most eight handles/probes and one extra remainder query per retained core;
the eight-handle budget is shared, not multiplied per unit.

Acceptance: improve good-unit recall and no miss-category regression beyond
baseline repeat spread, first against v4 on the same executable, then ensure
no regression relative to the original v2 baseline. V4 alone was rejected for
no recall gain and worse exclusions. No holdout before dev acceptance of the
combined behavior. Both arms are repeated, although auto uses the unchanged
rubric rather than review.md and has no native split stage.

## Dev result: rejected

All three MCP repeats still recover exactly 1/17 good units. Precision falls
to .142857–.166667 (mean .150794), with 6–7 predicted good units and six
exclusion misses in every repeat (original baseline four; v4 four to five).
All four human multi-unit split patterns remain missed; exact_match=false.
The agent now applies core splits, including root8 in all three repeats, but
this produces false positives rather than additional recovered human units.
The failure is not simply that available handles were ignored.

Auto is unchanged: 0/17, zero predicted good units, seven exclusion misses,
four split misses. MCP reports$0; auto billing unknown. No holdout query.
Fresh private report: dev-1793846580-784ccb16.md. Stop prompt-only tuning here;
the next round needs better boundary/validation evidence, not another wording
of the same core policy. Exact-match remains the reproduction requirement.

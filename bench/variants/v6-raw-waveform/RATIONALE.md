# v6-raw-waveform: current-membership waveform evidence

ONE new capability: a read-only waveform probe that measures actual raw
snippets of a unit's current spikes, rather than inheriting its primary sorter
template. Not a new split algorithm or a fitted label threshold.

Motivation: v5 applies substantially more core splits, yet still recovers only
1/17 human good units; precision falls to .1429–.1667, and exclusions regress
to six/run from the original four. Stop prompt-only tuning. Code inspection
shows brief/QC landmarks and compare use primary template waveforms, which can
be identical for split children irrespective of their actual memberships.

Design declared before diagnostic/candidate results: 128 jittered, temporally
stratified snippets maximum; eight best channels maximum (bounded fallback
when no PC channel ranking exists); -2..+3ms windows; first1ms baseline median
and MAD noise; -0.5..+1.5ms landmarks; early/late mean-waveform cosine. No spike
indices or recording paths in the tool output. Missing raw fails explicitly.
DC subtraction only, with high-pass-input provenance; do not apply filtered
template shape cutoffs to unfiltered/unknown raw. No automatic relabelling or
threshold fit. Determinism, post-split membership sensitivity, bounds and
read-only behavior require tests.

The proposer and rubric remain unchanged relative to the experimental43bc777
binary. Both v4 and v5 were rejected, not accepted baselines. Any candidate
must improve recall without regression against the ORIGINAL v2, not merely
look better than v5. Three repeats per arm, fixed9893 scorer, no holdout before
dev acceptance, exact_match still required for reproduction.

Protocol: v5 verbatim plus one paragraph instructing use of the new raw probe
on the clean candidate and remainder (sixteen waveform queries maximum), with
the filtering/unknown-evidence caveats. This is the smaller API-use part of the
new capability, not a newly fitted rubric or another prompt-only experiment.

Native data-library tests pass248/248 at private source snapshot
6e0857e291250faab3aaf6746b8335329a08f0a2. Windows rebuilt/pinned. Read-only
diagnostics successfully returned128/128 raw windows on five historical dev
roots with explicitly unfiltered-input provenance; no parameters changed.
## Dev result: rejected for curation improvement

All three MCP repeats still recover exactly1/17 human good units, versus
originalv2 also1/17. Precision.125–.142857 (mean.136905), seven to eight
predicted good units, six exclusion misses/run versus originalfour. Allfour
human multi-unit split patterns still missed; exact_match=false in allruns.
The agent DID call the raw waveform probe on split children and remainders.
This is not a tool-integration failure; the new evidence did not fix recovery.
Auto unchanged0/17, no predicted goodunits, seven exclusions/four splitmisses.
MCP reports$0; auto billingunknown. No holdoutquery.

Fresh private report dev-1793850576-575a8adf.md. Probe remains an experimental
measurement capability, not an accepted scientific improvement. Stop repeated
single-pilot tuning; use broader dev results to choose the next boundary or
clustering capability. Do not publish a PR claiming improved human recovery.

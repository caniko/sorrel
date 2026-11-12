# v4-dense-core: interiors of contaminated mixture components

One scientific change: extend the existing core proposer with bounded
high-posterior, Gaussian-density interiors when a complete component cannot
pass its existing gates. No prompt overrides; no thresholds fitted to a root.
The baseline is the built-in review protocol at 59c4f5b, with the same model,
repeats, and fixed scorer. The candidate binary supplies the new proposer.

Development evidence: the three-repeat historical pilot recovers 0/17 good
units with auto-LLM and 1/17 with MCP (precision 0.25–0.3333). Neither recovers
any of four human multi-unit splits. Explicit dev probes cannot propose cores
for some human core roots, and other offers cover too broad a component.
The partial intake dev DB also contains 46 core-pattern roots with 76 human
good units; this is not a full-cohort baseline or holdout evidence.

Design choices (declared before candidate scores): posterior >=0.95; Gaussian
ellipsoid mass 1, .99, .95, .90, .75, .50, .25 in that order; retain the largest
passing interior per component. At most seven additional candidates per
component, in the existing BIC-selected mixture (k<=8). Remote tails with high
posterior still face the density cut. These are exploratory design constants,
not per-recording or per-root fits; the extra searches do not establish a new
calibrated statistical significance claim.

Keep the existing minimum size (100 spikes and 1%), minority limit, d-prime
floor (2), refractory contamination limit (.1), and fewer-than-chance ISI gate.
Compute separation from the actual interior membership. Preserve an already
supported full component instead of trimming it for a more flattering score.
Leave cleanup's two-thirds rule, ranking, prompts and auto split execution
unchanged. The auto arm still cannot apply splits; no auto gain is promised.

Acceptance: three repeats per arm against the immutable baseline; improve
human good-unit recovery with no category regression beyond baseline repeat
spread. No holdout query before dev acceptance. Even a tolerant recovery gain
does not certify exact phy reproduction; all-spike/original-label exact_match
remains the final goal.

## Dev result: rejected as a standalone improvement

Three repeats per arm on the historical dev session, fixed scorer
8ec30ad03ba0a62099b705533a48f9938567b10dcb59ccf847f0727bfb5bbbfd:

- MCP recall remains 1/17 in every repeat. Precision rises from .30556 mean
  (.25–.3333) to .44444 (.3333–.5), but exclusion misses regress from 4/run
  to 4–5/run (mean 4.6667); human multi-unit split recall stays zero.
- Auto remains 0/17 with zero predicted good units. Its precision of 1 is
  vacuous; applied split capability is still absent.
- exact_match=false in all runs. No holdout query. MCP reported $0; all auto
  billing unknown. Candidate mean wall time: MCP 164.14s, auto 299.93s.
- New root-8 core offer is reachable (5,707 spikes), but no review repeat
  applied it. This is not enough to accept a scientific gain.

Fresh report: dev-1793844222-e25eabbb.md (private historical pilot bench).
Code stays experimental/unaccepted while a separate policy experiment tests
whether the reachable capability is being used appropriately. Do not publish
it as an accepted curation improvement or merge a PR on these results.

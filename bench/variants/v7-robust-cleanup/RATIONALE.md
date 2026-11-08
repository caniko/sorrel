# v7 robust cleanup — experimental, not accepted

One new boundary capability: full-covariance contaminated-Gaussian majority
cleanup. The narrow and broad components share a mean and covariance shape;
the broad component has an estimated scalar covariance inflation. This can
cut through a tilted cloud's tails rather than removing whole components of
an axis-aligned mixture. It does NOT solve rare-core or multi-neuron recovery.

Broader dev evidence (first six sessions, three repeats/arm, original arm):
37 human good units: 13 cleanup, 11 cores, 9 kept, 4 other. Auto recovers 7/37;
MCP 8/37 every repeat. Exact match false throughout. Worst session contains
16 human units but only 1 auto /2 MCP recoveries. Several majorities need
substantial boundary trimming, and low parent refractory contamination does
not imply the curator kept every spike. Prior v4/v5/v6 pilot experiments were
all rejected; their dense-core/policy/raw-waveform results are not accepted
baselines. No root-specific cut or learned prior is used here.

Declared before real diagnostic scores: at most 5000 jittered time-stratified
fitting rows, <=12 standardized PC dimensions; full covariance, ridge 1e-6
of mean variance; 80 EM/ECM iterations; fixed inflation starts 4 and 25,
best likelihood wins. Fit clean weight constrained to [.5,1-1e-6], inflation
to [1.01,100]. Require BIC improvement >=10 over a single full Gaussian,
MAP posterior boundary .5, >=100 spikes in each part and majority retained.
These are exploratory model/design constants, not fit to human memberships.
The retained part must have finite rpc<=.1 and no rpc regression; parent rpc
need not exceed .1. No d-prime floor (concentric tails have similar means).
No ISI-guided threshold scan, automatic labels, or altered good-unit rubric.

First run the native DEV-only coverage diagnostic. It constructs boundaries
from input only, then reads truth to measure attainable overlap, distinguishes
refractory-vetoed boundaries, and cannot access holdout. Oracle coverage is a
capability ceiling, NOT model curation performance or acceptance. If it adds
no attainable units over leaving roots intact, reject before further LLM runs.
If promising, isolate the new capability from prior unaccepted experiments,
pin/rebuild the arm, and run matching three repeats/arm on these same six
sessions, original model/scorer/jobs. Require actual recovery gain and no
category regression beyond baseline repeat spread before any holdout query.
Strict all-spike/original-label exact_match remains the reproduction target.

## Dev capability-ceiling result: rejected before LLM trials

Correctly row-aligned native diagnostic at136fbce8bf1484e2ca49dea1f5eefaefb5f28383,
coverage-1793856628-35e13583.json: 37human units, whole-root oracle14/37;
whole+supported-robust still14/37. No additional attainable tolerant matches.
Therefore do not spend curation-model repeats or query holdout for this variant.
This is an oracle diagnostic, not a measured curation recall of14/37.

One earlier diagnostic (coverage-1793856170-e8efae37.json) is explicitly invalid:
Session globals and original NPY rows were misaligned at simultaneous spikes.
It has an adjacent INVALID.md; the corrected command validates the original
row permutation and input-root identity. Existing saved arm outputs were not
affected; no fixed scorer changed and no prior scores were recomputed.

Some actual human cores sit in the broad tail, not the model's kept majority
(e.g. a 14530-spike human unit overlaps a 28625-spike tail at Jaccard.5076).
This motivates a separate bounded hierarchical-tail capability, not tuning
the v7cut until that root passes. All v7model constants remain unchanged.

# v9 per-coordinate mixture boundaries — experimental

ONE input-only boundary capability: fit separate 1-D mixtures on each current
standardized PC coordinate and amplitude, returning cumulative posterior cuts
between adjacent component means. The existing PC probe uses only the first
principal projection at zero; amplitude mixture fitting ignores PC axes. V7/V8
concentric-cloud/tail models added no attainable units. Stop those families.

Motivation from the separate HUMAN-FITTED dev feature audit: eight additional
units beyond whole-root matches have validation Jaccard>=.8 using a single PC
or amplitude cut. This does NOT mean eight automated recoveries. It indicates
some missing memberships are representable by simple coordinates, not more
radial ellipsoids. Fitted human threshold values were not printed/inspected
for this choice and must never be copied into a proposer, prompt or arm edit.

Declared before input-only v9diagnostics: <=5000jittered fitting rows per axis;
BIC picks among k1..4 diagonal (1-D) Gaussian mixtures,60iterations each; need
BIC improvement>=10 over oneGaussian. Order components by mean; between each
adjacent pair find lower-to-upper cumulative posterior=.5 crossing (128point
bracketing,32bisections; nearest mean-midpoint crossing if several). At most
three cuts/axis. No minimum component fraction, no desired count, no truth/ISI
in fitting. Require >=100spikes per part; evaluate rpc afterward, finite<=.1
and no worse than parent for a node to be supported. Neither support nor BIC
is a good label. Heavy tails can mimic mixtures; model review/validation still
needed. Time is NOT fitted, to avoid splitting by firing-rate change alone.

First native `coverage --method axis` on the same verified37-unit/six-dev
diagnostic cohort. Boundaries precede truth reads, originalNPY row mapping
validated. Oracle gain is only capability evidence, not measured performance.
Reject if it adds no reachable units. If promising, isolate the scientific arm
from prior rejected experiments, add bounded handle API, pin/rebuild and test
actual three-repeat/arm curation against original baseline with same model,
fixed scorer and matching scheduling. No holdout before actual dev acceptance.

## Input-only diagnostic result (not curation acceptance)

Windows93f542afde297ec63664a4cf303dc06b75393afb, native report
coverage-1793859518-cc4e620f.json in the verified six-session diagnostic bench:
37human units, whole-root oracle14/37; whole+supported-axis16/37. Two additional
attainable matches, Jaccard.9226 and.8314, both input-fitted PC coordinates and
rpc0. No human fitted threshold was copied. This warrants an actual curation
test but proves neither model recovery, precision, no regression nor exactmatch.

Add one bounded `by=axis` listing API (at most30alternative handles) and the
small API-use paragraph in review.md above the original unchanged rubric.
The scientific arm must start from original59c4f5b and add axis_split +this API
ONLY; do not carry rejected dense-core, robust, density-tree, raw-probe or
feature-oracle behavior into the arm. Runner diagnostics may contain those
modules but are not the arm. Old experimental curation changes stay unaccepted.

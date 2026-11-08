# Dev-only feature representability audit

After four rejected pilot experiments and two unsuccessful boundary-ceiling
diagnostics, do not guess another boundary family without checking what input
coordinates can represent the human's selections.

`sorrel dev bench --bench PRIVATE coverage --dev-limit 6 --method feature-oracle`
is lead-only and dev-only. It reuses the explicit original-row permutation
validation; benchmark inputs, sources, scores, DB and holdout are not edited.

This mode DELIBERATELY uses dev human spike labels. For each human good unit,
each standardized PC coordinate, amplitude and time, it finds one transparent
upper/lower threshold on even session-global spikes. Sorting is bounded to at
most2500examples per class, with original class prevalence restored by weights
in the training Jaccard objective. Feature/direction/threshold selection does
not consult validation labels. The top3training-ranked cuts are evaluated on
odd spikes and on the full dev root, counting foreign-root truth in the union.

This is an approximate representation diagnostic, NOT automated curation,
independent session validation, an exact-reproduction result, or even a proof
that a unit cannot be separated when the sampled search fails. Alternating
spikes remain correlated. Values in the report are human-fitted and must NOT
be copied into per-root production cuts, prompts, or model tool calls.

Use the findings only to choose a general new feature/proposer or a properly
leave-session-out learned boundary model. Then independently benchmark that
input-only mechanism with the original fixed scorer and matched repeats.
Do not query holdout or claim acceptance based on feature-oracle results.

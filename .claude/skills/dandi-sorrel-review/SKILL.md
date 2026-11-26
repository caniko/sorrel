---
name: dandi-sorrel-review
description: Pin a public Dandiset, collect read-only Sorrel and NWB evidence, and pack an offline-verifiable review. Rebuilds the local results table and tracks human-reviewed feedback without sending or publishing anything.
---

# Review a public Dandiset with Sorrel

The [Sorrel × DANDI results portal](https://nexus.beloda.co/sorrel) is separate
from the [Sorrel project website](https://sorrel.beloda.co) and
[Beloda's team and consultation site](https://beloda.co).

Use Python 3.10+ (no packages) and Sorrel built from `origin/trunk` at `a447b68`
or newer (`v0.5.0+`, with the default DANDI streaming features). Record the
actual Sorrel version and revision, kit revision, and any model/prompt
revisions used. Do not use embargoed data, credentials, signed URLs, or
private recordings in this workflow.

## Run

```sh
kit=.claude/skills/dandi-sorrel-review/dandi_review.py
python3 "$kit" init-store "$STORE"
python3 "$kit" pin 000006 --version 0.220126.1855 --store "$STORE" --out "$STAGE/pin.json"
```

Create `$STAGE` before pinning. Prefer published versions (`0.YYMMDD.HHMM`);
`pin` defaults to the latest published version. The default `--nwb-only`
lists only assets whose path ends in `.nwb` or `.nwb.zarr` or whose media type
identifies NWB. `--no-nwb-only` also lists sidecars and other
assets. `n_assets_total` always counts the whole archive, and `n_nwb_assets`
counts NWB assets, not all HDF5 or Zarr assets. Copy both into `asset_selection`; set `scope: "nwb"`
for an NWB-only review (`"archive"` for the full inventory). Record whether
you reviewed all assets in scope or sampled, and why. Keep refused and failed
assets in the result, with reasons.

`pin --store "$STORE"` refreshes `dandisets/<dandiset_id>.json` using the
[Dandiset schema](https://nexus.beloda.co/sorrel/schema/dandiset-1.json).
It retains the complete version response in `dandi_raw_metadata` alongside
extracted fields and the full asset inventory, regardless of `--nwb-only`.
Re-pinning updates the snapshot, keeps local `unique_metadata` and
probe/sorter/anatomy fields, and accumulates `versions_seen`. Use
`--dataset-out "$STAGE/artifacts/dandiset.json"` to also write a snapshot for
the bundle (create `artifacts/` first), or use `--dataset-out` without a store
for a standalone snapshot. Refresh a bundle snapshot after adding local notes.

```sh
python3 "$kit" dataset-meta "$STORE" set 000006 --probe 'Neuropixels' \
  --merge-json '{"task":{"name":"delayed response"}}' --set 'figures=[1,"S2"]'
```

`--merge-json` accepts an inline object or `@file` and recursively merges
objects; arrays and scalars replace prior values. `--set KEY=VALUE` sets a
top-level note, parsing JSON values or keeping plain strings. Repeat `--probe`,
`--sorter`, or `--anatomy` to set a list. Supply `--name` to start a dataset
record that is not already known to the store. An unnamed legacy entry uses
`DANDI:<dandiset_id>` unless you provide a name. The command validates the
record and rebuilds the SQLite projection.

Use `--version draft` or opt in to fallback with `--allow-draft`. Draft
`dandi://…@draft/path` locators resolve again on each open. Prefer the pinned
asset-ID `download_url`: for HDF5, download a local copy and verify its SHA-256
and size before use (and again after capture). Use the asset-ID URL for NWB-Zarr
streaming, which Sorrel resolves via the pinned Zarr ID. For any streamed
draft, run the archive postcheck **after** the last Sorrel open, before
summarizing and packing:

```sh
# HDF5 local verification:
python3 "$kit" verify-pin --manifest "$STAGE/pin.json" --asset-path "$asset_path" --local-file "$local_nwb"
# Or, after streaming:
python3 "$kit" verify-pin --manifest "$STAGE/pin.json" --asset-path "$asset_path"
```

`verify-pin` updates that asset's receipt in the manifest. The API postcheck
compares the path's asset ID, pinned digest, storage IDs and draft
modification token (`draft_modified_source` in the manifest); if the draft
changed, repin and repeat collection.
Every bundled asset needs a verified pin receipt. Draft assets require a
local checksum or API postcheck; `pin` records `published_version` for
published assets. DANDI path locators must match the review's Dandiset,
version, and asset path.

### Collect one asset

For a published version, set `input` to the pinned `locator` from `pin.json`.
This Units-only example uses `--sample-rate 1000` as an **analysis grid
override**, not a native recording rate. Pass it to both Sorrel and
`summarize-asset`, which records `analysis_sample_rate_hz`. Omit it when Sorrel
can use a declared clock. Keep `--sample-rate`, `--dat`, and `--nwb-series`
consistent across all opens. Without verified traces and a positive duration,
`duration_s` and `n_low_presence` are null. `sample_rate_hz` is only recorded
when a native recording rate can be established. Without measured spike
amplitudes, `n_amp_cutoff` and amplitude-based epoch screening are unavailable.

Start each baseline in a fresh per-asset working directory with an explicit
`--journal`; do not reuse a mutated journal or overwrite `clusters.before.tsv`
on reruns. Do not keep the journal open in the GUI during capture. Use
`--discard-journal` only when explicitly resetting a baseline, and never during
after-curation capture.

```sh
asset_path=sub-anm369962/sub-anm369962_ses-20170316.nwb
input="dandi://dandi/000006@0.220126.1855/$asset_path"
workdir="$STAGE/work/$asset_path"
art="$STAGE/artifacts/$asset_path"
journal="$workdir/sorrel.journal"
(
  set -e
  set -C  # refuse to overwrite a baseline
  mkdir -p "$(dirname "$workdir")" "$(dirname "$art")"
  mkdir "$workdir" "$art"
  sorrel describe --json --sample-rate 1000 --journal "$journal" "$input" > "$art/describe.json"
  sorrel call --sample-rate 1000 --journal "$journal" "$input" list_clusters > "$art/clusters.before.tsv"
  sorrel qc --sample-rate 1000 --journal "$journal" "$input" --out "$workdir/qc-before"
  cp "$workdir/qc-before/cluster_qc.json" "$art/cluster_qc.before.json"
  sorrel report --sample-rate 1000 --journal "$journal" "$input" --out "$art/sorrel_report.html"
)
```

Check every exit status. `list_clusters` without `--raw` returns TSV, not JSON.
`qc --out` takes a directory; `report --out` takes a file. For a failure, omit
incomplete outputs and run `summarize-asset` with `--open-status refused` (a
reader refusal) or `--open-status failed` (another failure) and a sanitized
`--refusal-reason`. Do not change the source file to work around a refusal.

For a human review use `sorrel open` with the same input and options. For an
agent review start with `sorrel agent-guide --prompt review`; use `sorrel mcp`
with the same input/options, or `sorrel call … brief '{"level":2}'`. Record
whether a human, agent, or both did the work. Capture the baseline before
edits. To capture a curated state, repeat `list_clusters`, `qc` and optionally
`report` with the same journal after edits; write `clusters.after.tsv` and
`cluster_qc.after.json`. `sorrel call --sample-rate 1000 --journal "$journal"
"$input" save` (on one shell line) reports `save_dir`, where Sorrel writes
`curation.json` and Phy sidecars. For streaming, those files are local; for a
local NWB they are written beside the input. Copy only `curation.json` into
the artifact directory. Keep the journal and raw data out of the bundle.

```sh
python3 "$kit" summarize-asset --manifest "$STAGE/pin.json" --asset-path "$asset_path" \
  --sample-rate 1000 --describe "$art/describe.json" --before-qc "$art/cluster_qc.before.json" \
  --before-clusters "$art/clusters.before.tsv" --unique-metadata '{"session":"20170316"}' \
  --emit-findings --out "$STAGE/asset-summary.json"
```

`--unique-metadata` takes an inline JSON object or `@file` for asset-specific
notes, such as a task condition or custom `/units` columns.

Pass `--after-qc`, `--after-clusters`, `--curation`, and `--nwb-inspector` only
if those snapshots exist. Each paired QC export and TSV must agree exactly on
live cluster IDs and spike counts, including unlabeled units. Structural
counts come from `curation.json`; the TSV alone cannot distinguish a merge
from a removal. Without an after TSV, label-only curations can be replayed
against the baseline. If any structural edits are recorded, keep only their
counts: post-curation unit and relabel counts stay null without the after TSV,
because Sorrel may omit a move into an emptied cluster ID. The after TSV is
authoritative for live labels. With structural edits, only unambiguously
tracked units contribute to relabel counts; reused IDs, structural
descendants and changed lineage are not relabels. If Sorrel marks a v2 export
unrepresentable, retain it as evidence but omit `--curation`; leave structural
counts unknown.
Keep the exact bytes cited by candidates. JSON uses RFC 6901 pointers (array
indexes start at 0); text uses `L<n>` or `L<n>-L<m>` (TSV header is `L1`).
HTML also permits `#<id>` if the report actually contains that ID.
An `ok` asset requires an evidence level. Use `--evidence-level` if no Sorrel
describe, cluster or QC output can establish it. Failed or refused opens
cannot report recording duration, native rate, presence or amplitude-cutoff
counts. `describe --json` rounds display values, and NWB can provide a trace
without filling `data files`. Use `--evidence-level raw` only after verifying
trace access (e.g. a successful `probe` waveform); never infer raw from channel
count alone. Record the author label column when known (`--author-label-source
label`, `quality`, or `KSLabel`).

### Pack and verify

Write `$STAGE/review.json` against
[`review.schema.json`](https://nexus.beloda.co/sorrel/schema/review-1.json)
and write `$STAGE/report.md`.
Use `files: []` in staging; `pack` fills the manifest from `report.md` and
`artifacts/` and checks any supplied hashes or sizes. Use one evidence
subdirectory per asset. Copy the `asset` rows
from `summarize-asset`, assign unique finding IDs across the review, and tie
each improvement to its findings. Keep metrics missing when they were not
measured. The candidates are screening flags, not curation decisions.
Use optional review-level `unique_metadata` for notes specific to the review.
`pack` records an optional `artifacts/dandiset.json` as `dandiset_metadata` and
hashes it like any other artifact. Review notes and snapshots for private
material before packing.

```sh
python3 "$kit" pack "$STAGE" --store "$STORE" --dry-run
python3 "$kit" pack "$STAGE" --store "$STORE"
python3 "$kit" validate "$STORE"
python3 "$kit" project "$STORE"
python3 "$kit" impact "$STORE" --format md
```

`pack` sorts the file manifest by path, validates evidence pointers, and
preflights projection before installing the bundle. `validate STORE` also
preflights the projection. The review ID is the SHA-256 of canonical `review.json`.
Do not edit installed bundles; pack a new one with `supersedes` set to the
prior review ID for that Dandiset. `results.tsv` and `results.sqlite` are
rebuildable from `dandisets.tsv`, `dandisets/*.json`, reviews, and feedback.
Dataset JSON takes precedence for fields it supplies, with legacy metadata
available from the registry and review envelopes. Use one writer per
store. `impact` rebuilds its own in-memory view from validated authoritative
inputs; cached results are never used for the disclosure queue. Rates pair
numerators and denominators only where the relevant count and a positive
good-unit count are available. Missing unit measurements within those assets
can make the measured share a floor; check `n_assets_with_good_rp_*` and
`good_before_with_rp_*` for the exact coverage behind each percentage. Unmeasured
sums are `null` (`N/A` in text or Markdown), not zero. An unlabeled asset has
no good/mua/noise labels on any live unit. Independent latest reviews of the
same asset remain separate observations.

### Offline or older Sorrel

Transfer the pinned manifest and a verified local NWB copy (or intact
NWB-Zarr directory) to the disconnected host. Keep the archive copy
read-only; use a disposable copy for Sorrel, especially for a `save`.
Keep the pinned archive identity in `review.json`, not the local path.
Use the collection steps above with `input` set to the local file instead of
the locator, starting from a fresh working directory.

If a trustworthy manifest cannot be transferred, get the pinned version and
asset identity from a connected operator. Use explicit `--asset-id`, `--path`,
`--size-bytes`, `--encoding`, and `--pin-verification` (using the verified
receipt from the operator) to
summarize a published asset locally. `pin` and `verify-pin` without
`--local-file` need archive access. Contribute the
validated `reviews/<dandiset_id>/<review_id>/` directory, not the local
database, journal, private correspondence, or NWB itself. Include predecessors
when transferring superseding reviews. Inspect exports for private paths,
notes and signed URLs **before** packing. If you redact an artifact, rerun
`summarize-asset` so the evidence hashes match the new bytes.

## Frozen v1 defaults

| Threshold key                 |   Value | Candidate rule                                                                                                    |
| ----------------------------- | ------: | ----------------------------------------------------------------------------------------------------------------- |
| `rp_contamination_lo95_max`   |    0.10 | `rpv_post_lo95` above: lower credible bound exceeds the cutoff (`qc.refractory_contamination`)                    |
| `srp_contamination_upper_max` |    0.10 | `srp_contamination` above: upper confidence limit, screening only (`qc.refractory_screening`)                     |
| `isi_violation_rate_max_hz`   | 0.50 Hz | `isi_violation_ratio` above: a rate, not a proportion; emit a screening finding only if both estimates are absent |
| `presence_ratio_min`          |    0.70 | `presence_ratio` below                                                                                            |
| `amplitude_cutoff_max`        |    0.10 | `amplitude_cutoff` above                                                                                          |
| `epoch_rate_ratio_min`        |    0.25 | `epoch_rate_ratio` below                                                                                          |
| `epoch_amp_ratio_min`         |    0.50 | `epoch_amp_ratio` below                                                                                           |
| `drift_correlation_max`       |    0.50 | Absolute `drift_correlation` above                                                                                |

Store the exact threshold values used; `--thresholds` takes an inline JSON
object or `@file`. The three refractory counts are separate; a screening
flag never confirms contamination. Nulls are not failures or passes.
`n_noise_as_good` counts observed good-to-noise relabels, not screening
flags; it is unmeasured without baseline and after labels.

## Feedback and disclosure

A human must inspect the cited bytes, check the interpretation and authorize
any outreach or public release. Set `human_verified: true` in the review
only for findings that passed that check. Machine-generated `unit_ids` use
`unit_id_namespace: "sorrel_cluster_index"`, not author NWB unit IDs. Verify
the mapping against NWB `/units/id` before marking a finding `human_verified`
for outreach; use `unit_id_namespace: "nwb_unit_id"` if you replace the
identifiers. The kit refuses any notified record quoting an unverified
finding. It never sends a message, opens a PR, or publishes a review. See the
[feedback schema](https://nexus.beloda.co/sorrel/schema/feedback-1.json).

```sh
python3 "$kit" feedback "$STORE" record --dandiset 000006 --status sent \
  --channel email --recipient "$RECIPIENT" --actor "$REVIEWER" --review-id "$REVIEW_ID" \
  --quote-finding "$REVIEW_ID:f01"
python3 "$kit" impact "$STORE" --format json --as-of 2026-02-01T00:00:00Z
```

Record `sent` only after actually notifying the contact. The default
notification window is 30 days (14–90) and the default reminder is at 14
days (at least 7). Use whole-second UTC timestamps or explicit offsets. The
first `sent` starts the clock; `reminded` cannot be recorded before it is
due. For earlier contact, `--status reminded --early-followup` logs the
contact but does not complete the scheduled reminder. Once outreach is
recorded, the review IDs and quoted findings are frozen and the record
cannot return to `drafted`. The first send also freezes the recipient and
channel; both are snapshotted in every send and reminder event. To notify
about new findings or use a different recipient or channel, choose a new
`--feedback-id`; each record gets its own `feedback/<feedback_id>.json` and
disclosure clock. Use an ID without personal information. The result table
shows the most recently notified record linked to that review; the
disclosure queue includes all records. An `acknowledged` or `accepted`
response may carry a dated
extension with a reason. A response or a decline is not ghosting. The
`publish_eligible_ghosted` bucket means the grace period passed without a
recorded response; it is a cue for a human decision, not permission to
publish. Record `republished` only for a confirmed author update, with
version, DOI and corrected-unit count when known. `--as-of` sets the queue
clock against the current log snapshot; it does not undo recorded events.
`init-store` ignores `feedback/` by default: correspondence, recipients and
evidence URLs may be private. Keep the authoritative logs local. Share only
a separately exported, human-approved redacted copy; do not force-add the
private files.

## Agent tiers

Roles, not models. `flash` does cheap, mechanical or scouting work; `worker`
does long code or prompt edits and reviews; `lead` is the invoking session,
which decides and runs anything sealed. Name the agents in the skill args,
e.g. `flash=<agent> worker=<agent>`. A role without a named agent is done by
the lead inline. Delegated agents get only what their task row lists.

| Task                                                                       | Tier   | Gets                                                           | Returns                                       |
| -------------------------------------------------------------------------- | ------ | -------------------------------------------------------------- | --------------------------------------------- |
| Pin public metadata, collect exports, run `summarize-asset`                | flash  | Pinned public locators, read-only commands, artifact directory | Exact artifacts, rows, and any refusal        |
| Review screening flags and draft an evidence-linked report                 | worker | Sanitized artifacts, guide, fixed thresholds                   | Draft findings and suggestions, unverified    |
| Confirm evidence level, human verification, bundle contents and disclosure | lead   | Review draft and authorized source evidence                    | Validated bundle and human-approved next step |

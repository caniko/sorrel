# Spike rescue

`sorrel rescue <DATA_DIR>` finds spikes the sorter missed, for units you
already have. It adds spikes to existing units. It does not discover new
units. It works on Kilosort and phy directories only.

It is a dry run unless you pass `--apply`. `--apply` journals the additions
as one batch, so one **Undo** reverts them. This is the same model as
[Automatic Curation](./auto-curation.md).

## Method

For each target unit:

1. **Templates.** Build a raw-space template for each time block, 60 s by
   default. Each block gets its own template, so drift is absorbed.
2. **Filter.** Compute a matched filter whitened by the noise covariance,
   Σ⁻¹·M, where M is the block template and Σ is the noise covariance.
3. **Scan.** Run the filter on the GPU over the high-passed, CMR'd residual.
   The residual is the signal left after every accepted spike has been
   subtracted.
4. **Accept.** Keep a detection only if it:
   - beats the best competing unit by `--margin`;
   - is more than 1.5 ms from the unit's existing spikes;
   - clears the unit's threshold. The threshold comes from a
     channel-shifted null, set to control the false discovery rate at `--q`.
5. **ISI check.** If the unit's ISI-violation fraction more than doubles,
   drop all of that unit's additions.

## Flags

| Flag                       | Default | Meaning                                                                     |
| -------------------------- | ------- | --------------------------------------------------------------------------- |
| `--units good\|all\|<ids>` | `good`  | Units to rescue: the good units, all units, or a list of cluster ids.       |
| `--q`                      | `0.05`  | Target false discovery rate for each unit's threshold.                      |
| `--margin`                 | `0.2`   | Required margin over the best competing unit.                               |
| `--replicates`             | `3`     | Channel-shifted null replicates.                                            |
| `--block-seconds`          | `60`    | Length of each template time block.                                         |
| `--chunk-seconds`          | `2`     | Length of each GPU processing chunk.                                        |
| `--channels-per-unit`      | `16`    | Channels used by each unit's template and filter.                           |
| `--hybrid-frac F`          | off     | Inject hybrid spikes for a recall check. See [Hybrid check](#hybrid-check). |
| `--json`                   | off     | Print the report as JSON.                                                   |
| `--apply`                  | off     | Journal the additions as one undoable batch.                                |
| `--save`                   | off     | With `--apply`, save the directory. See [Saving](#saving).                  |
| `--seed`                   | 1       | Random seed for the nulls and the hybrid injection.                         |

The usual recording flags also apply, as for `sorrel auto`.

## Dry run

A dry run is the default. It prints one row per target unit:

| Column       | Meaning                                                                        |
| ------------ | ------------------------------------------------------------------------------ |
| `existing`   | Spikes the unit has now.                                                       |
| `added`      | Spikes the rescue would add.                                                   |
| `theta`      | The unit's detection threshold, from the null at `--q`.                        |
| `est_fp`     | False positives among the added spikes, estimated from the null.               |
| `isi_before` | ISI-violation fraction before the rescue.                                      |
| `isi_after`  | ISI-violation fraction after. Step 5 above applies to this column.             |
| `min_r2`     | Lowest feature-calibration R² (see [New spike features](#new-spike-features)). |

## Hybrid check

`--hybrid-frac F` injects F × `n_spikes` copies of each unit's template at
random times into the in-memory signal. The `.dat` file is never written.
The rescue then runs as usual, and the report gives recall per unit and per
amplitude tercile.

A hybrid run is always a dry run. It never applies or saves.

Injected spikes are idealised, so recall on them is likely higher than on
real spikes. It still shows which units the threshold cannot reach. Run it
before you trust a rescue on real data.

## Applying

`--apply` journals one undoable batch:

- an `AddSpikes` record with the added spikes;
- a `review:rescue` tag on each touched unit, shown with the other
  `review:*` tags.

Units that came straight from the sorter (`Auto[...]`) also get `tuned`.
See the tag table in [Automatic Curation](./auto-curation.md#tags).

## Saving

`--save` saves the directory (see [Saving](./saving.md)). Because added
spikes change the row count, the save rewrites every per-spike array in
time order:

- `spike_times.npy`, `spike_clusters.npy`, `spike_templates.npy`,
  `amplitudes.npy`, `pc_features.npy`, `template_features.npy`, every other
  `spike_*.npy`, and `spike_reorder.npy`.
- `spike_sorrel_rescued.npy` is new. It holds 0 or 1 per spike, and marks
  the added rows. phy can show it as a spike attribute.

Safeguards:

- Before the first rewrite, the originals are hard-linked into
  `sorrel_original/`. They are the sort as loaded, which the journal
  replays onto, so the rescue stays undoable after the save.
- phy's `.phy/` cache is renamed to `.phy.sorrel-stale`, so phy does not use
  it with the rewritten arrays.
- A manifest of pending renames is written first, which makes the save
  crash-safe.
- The save refuses to run if the directory holds an unknown per-spike
  `.npy`, because it cannot reorder that file.
- Writing phy files elsewhere without the per-spike arrays (`journal replay`)
  is refused while there are added spikes, because the arrays would no
  longer match.

Undoing the rescue and saving again writes the original arrays back.

## New spike features

Added spikes do not get features computed from their waveforms the way
Kilosort computes them. `pc_features`, `amplitudes`, `template_features` and
the other per-spike attributes are predicted by a ridge regression. The
regression is fitted on each unit's own spikes. `min_r2` in the dry run
reports how well that fit works.

On real tetrode data, R² was low, about 0.1–0.2. Treat the PC features of
rescued spikes as approximate.

## GPU

Rescue runs on the GPU through wgpu. If no GPU is available, it falls back to
a CPU reference and prints a warning. The CPU reference is about 100× slower.

On a 297 s, 32-channel Kilosort 2 tetrode recording with 62 clusters, 12
target units took 3.5 s on the GPU and 9.5 min on the CPU. That is about
160× on this recording.

## Limitations

- The channel-shift null is conservative for large, high-rate units. On the
  test recording, hybrid recall was 60–66% for 2 of 12 units and near 0% for
  most of the others.
- A sign-flipped null recovered 65–90% for 7 units but failed on 2 others.
  Choosing the null is still open.
- There is no joint refit of overlapping spikes.
- Single-channel units cannot be rescued, because the channel-shift null
  cannot break them.
- Rescue does not discover new units.

## Example session

```bash
# 1. Hybrid check: measure recall. Never writes the .dat or the journal.
sorrel rescue /data/session1 --hybrid-frac 0.2 --seed 1

# 2. Dry run on the good units.
sorrel rescue /data/session1

# 3. Journal the additions and save.
sorrel rescue /data/session1 --apply --save
```

Reopen the directory after step 3. Then check the `review:rescue` clusters
with the review filter.

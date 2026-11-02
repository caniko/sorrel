---
name: export-curated-sort
description: Use on the machine that holds the spike-sorting data to package a phy-curated Kilosort 4 sort (exported to phy, then merged/split by hand) for the sorrel double-blind bench upload. Produces one verified .tar with only what the bench needs.
---

# Export a curated sort for the sorrel bench

The bench replays your phy curation against `sorrel auto` and the review agent,
so it needs the sorter's output, your final cluster ids and the raw recording,
and nothing else.

## Run

```bash
uv run .claude/skills/export-curated-sort/export_curated_sort.py pack PHY_DIR OUT_DIR
```

No uv: `pip install numpy zstandard`, then `python .claude/skills/export-curated-sort/export_curated_sort.py pack PHY_DIR OUT_DIR`.

`PHY_DIR` is the phy folder (it holds `params.py`) after you saved the curation
in phy. It is only read. Several recordings go in one call: list every phy folder,
then `OUT_DIR` last; each becomes its own package, named after the folder (widened
with parent folder names when they collide, as `kilosort4` folders do).

Recordings where Kilosort found **no units** (missing or empty `spike_times` /
`spike_templates`) are excluded: there is nothing to compare against, so no package
is written for them and the summary lists them as excluded. Pass all candidate
folders and let the tool drop the empty ones. A folder that fails a check is
reported and skipped, the rest still pack, and the command exits non-zero.

Options: `--name NAME` (single folder only), `--level 19` (zstd level),
`--jobs N` (default: CPUs, capped by free memory), `--dat PATH` (raw file when
`params.py` points somewhere else). Level 19 on a large recording takes a while;
it is the smallest upload, so let it run.

Show the user the printed summary: included files, excluded names, sizes, ratio
and the check result. Upload only `OUT_DIR/NAME.sorrel-upload.tar`; keep the
`.sha256` beside it if the upload tool takes one. If the check line is not
`PASSED` the command exits non-zero and no package is written.

## What goes in

- phy files: `spike_times`, `spike_templates`, `spike_clusters` (your curation),
  `amplitudes`, `templates`, `channel_positions`, `channel_map`, the optional
  features/`similar_templates`/`whitening_mat_inv`, and `cluster_group.tsv`
  (your good/mua/noise labels; the tool warns loudly if it is missing).
- `params.py` rewritten to `dat_path = 'recording.bin'`, `offset = 0` plus the
  five keys needed to read it (the original path holds your username).
- The raw recording, filtered and compressed in chunks (`raw.zst.bin`).
- A SpikeGLX `.meta` next to the raw file, if any, as `recording.meta`. It carries
  the gain calibration but also the probe serial numbers; tell the user. The
  values of `fileName` (a local path) and `userNotes` (free text) are blanked,
  every other key is kept, and the manifest lists the blanked keys in
  `meta_scrubbed`.

## What stays out

`.phy/`, `phy.log` (it lists your edits and local paths), `ops.npy` (absolute
paths), `cluster_info.tsv`, whitened/temp copies, `rez*` and everything not on
the list above. The manifest records excluded **names** only, never contents,
and the source folder's basename, never its full path.

Custom `cluster_<x>.tsv` files may be your own labels and are excluded but
listed as `custom_label_files`: ask the user whether any of them should go in.

## Checks

Before compressing: equal array lengths, spike times inside the recording,
template ids in range, channel count (warning only), and a warning when
`spike_clusters` equals `spike_templates` everywhere (curation not saved).
Merges and splits are counted in the manifest. After compressing, the raw
stream is decoded again and its sha256 compared, and every file in
`phy.tar.zst` is read back and hashed. A mismatch deletes the outputs.

To check a package yourself:

```bash
uv run .claude/skills/export-curated-sort/export_curated_sort.py unpack PKG OUT_DIR
```

rebuilds the phy folder (with `recording.bin`) in a new, empty `OUT_DIR`, verifies
everything and prints the counts. It works in a temp subfolder and moves the files
into `OUT_DIR` only after every check passed, so a failure leaves `OUT_DIR` empty.
`pack` refuses an `OUT_DIR` inside `PHY_DIR`. Self-check of the tool:
`uv run .claude/skills/export-curated-sort/test_export_curated_sort.py`.

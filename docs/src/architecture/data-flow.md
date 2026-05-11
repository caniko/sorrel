# Data Flow

Sorrel's core path is generic over `P: DataProvider`. The binary detects
a backend, constructs the concrete provider, opens the SQLite journal,
and builds a `Session<P>`. From there the UI, render buffers, and
compute kernels are all statically dispatched.

## Open

1. `detect_backend` inspects the path (file vs directory, marker files,
   extension) and picks one of `Kilosort`, `SortingAnalyzer`, `Nwb`, or
   `Ks4Rez`. `--backend` skips the detector.
2. The chosen provider opens its inputs:
   - Kilosort / phy mmaps `spike_times.npy`, `spike_clusters.npy`, and
     the raw `.dat`; reads optional artefacts for templates, PC
     features, amplitudes, similarity, probe geometry, and quality
     metric sidecars.
   - Other backends do the equivalent for their formats.
3. `Session::new` holds the provider and journal. `seed_*` calls bucket
   amplitudes, templates, PC features, and template waveforms once so
   hot-path lookups are O(1) slice borrows.
4. `replay_journal` re-applies every prior `CurationCommand` (including
   undo/redo records) so the open state matches the last running state.

## Edit

`CurationCommand` is a closed enum: `Relabel`, `Merge`, `Split`,
`Note`, `Batch`, `Undo`, `Redo`. The session applies a command in two
strictly-ordered steps:

1. **Append to the journal.** A successful SQLite commit is the
   point-of-no-return.
2. **Mutate in-memory state.** The `ClusterIndex` updates, the
   selection moves, caches that depend on history are invalidated.

If step 1 fails the in-memory session does not move. If the process is
killed between step 1 and step 2, replay reconstructs the missing
transition.

## Save

`Cmd/Ctrl-S` writes `spike_clusters.npy` + `cluster_group.tsv` into the
save directory and reseals the journal so future replays start from the
new baseline rather than from the original sort.

## Render

The trace view assembles a window of the raw trace, optionally runs CMR

- HP filter (CPU or GPU), then LTTB-downsamples per channel into
  `TraceVertex` buffers consumed by the wgpu pipeline. Other views
  (raster, ISI, CCG, features, drift map…) build their geometry directly
  in egui from the per-cluster slice borrows the session exposes.

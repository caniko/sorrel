# Data Flow

Sorrel's core path is generic over `P: DataProvider`. The binary detects
a backend, constructs the concrete provider, opens the curation journal,
and builds a `Session<P>`. From there the UI, render buffers, and
compute kernels are all statically dispatched.

## Open

1. `detect_backend` inspects the path (file vs directory, marker files,
   extension) and picks one of `Kilosort`, `SortingAnalyzer`, `Neuralynx`,
   `Nwb`, or `Ks4Rez`. `--backend` skips the detector.
2. The chosen provider opens its inputs:
   - Kilosort / phy mmaps `spike_times.npy`, `spike_clusters.npy`, and
     the raw `.dat`; reads optional artefacts for templates, PC
     features, amplitudes, similarity, probe geometry, and quality
     metric sidecars.
   - Neuralynx mmaps one `.ntt` and builds a time-sorted record index, a
     strictly increasing sample time per spike and peak-to-peak
     amplitudes in µV. Its content hash seals the journal; the unsorted
     pool, like other backends' initial sorts, is rebuilt before replay.
   - Other backends do the equivalent for their formats.
3. `Session::new` holds the provider and journal. `seed_*` calls bucket
   amplitudes, templates, PC features, and template waveforms once so
   hot-path lookups are O(1) slice borrows.
4. `replay_journal` rebuilds the action tree from the journal and applies
   the path from the sort as loaded to the current edit, so the open state
   matches the last running state.

## Cache

Derived artifacts that have cache support consult
`<dataset>/.sorrel/cache/` first using the provider identity, current
journal head, artifact algorithm version, parameters, and any
per-cluster spike fingerprints. Misses fall through to the existing
compute path and then write an archive for the next open. For `.ntt`,
the cache root is beside the journal, not inside the tetrode's
`<filename>.sorrel/` artifact directory.

The journal head is the invalidation point. Applying a
`CurationCommand` records the previous head and affected cluster
identities in the session's in-memory invalidation log. Sorrel evicts
known pre-edit derived artifacts at save and quit, while
`CacheStore::open` runs bounded GC for orphaned, aged, and over-budget
files. See [Derived Cache](./cache.md) for the full contract.

## Edit

`CurationCommand` is a closed enum: `Relabel`, `Merge`, `Split`,
`Note`, `Tag`, `AddSpikes`, `Batch`, `Undo`, `Redo`. Every frontend (GUI,
MCP, remote) edits through `SessionHost::apply`, which tags the edit with
its actor and announces it to subscribers. A forward command becomes a new
node under the current one; `Undo`/`Redo` and History clicks move the
current node. The session applies an edit in two strictly-ordered steps:

1. **Append to the journal.** A durable (fsynced) journal append is the
   point-of-no-return.
2. **Mutate in-memory state.** The `ClusterIndex` updates, the
   selection moves, caches that depend on history are invalidated.

If step 1 fails the in-memory session does not move. If the process is
killed between step 1 and step 2, replay reconstructs the missing
transition.

## Save

`Cmd/Ctrl-S` writes `spike_clusters.npy` + `cluster_group.tsv` into the
save directory. A phy save also journals a seal; replay starts from the
original sort in `sorrel_original/`. Neuralynx writes its assignments to
`<filename>.sorrel/` in time order and replays from the untouched `.ntt`.

## Render

The trace view assembles a window of the raw trace, optionally runs CMR

- HP filter (CPU or GPU), then LTTB-downsamples per channel into
  `TraceVertex` buffers consumed by the wgpu pipeline. Other views
  (raster, ISI, CCG, features, drift map…) build their geometry directly
  in egui from the per-cluster slice borrows the session exposes.

A provider with no continuous recording can supply stored per-spike
waveforms through `stored_snippet_window()` and `stored_snippet(t, out)`.
Waveforms, computed PCs and raw amplitudes request the provider's exact
window at the spike's immutable sample index; edits only change which
cluster owns that spike. Stored snippets bypass trace CMR and high-pass
filtering.

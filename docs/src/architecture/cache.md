# Derived Cache

Sorrel stores rebuildable derived artifacts under:

```text
<dataset>/.sorrel/cache/
```

The cache is not user data. Deleting this directory must never delete curation
work, journal records, provider inputs, or exported phy files. It only removes
derived archives that Sorrel can recompute from the dataset and journal.

## Contents

The cache stores expensive derived data such as seeded amplitude buckets,
per-cluster PC subspaces, isolation metrics, and correlograms. It does not
store the curation journal, external JSON sidecars, raw traces, spike arrays,
or any file that is required to reopen the dataset.

Each cache entry is an rkyv archive at:

```text
<dataset>/.sorrel/cache/<kind>/<fingerprint>.rkyv
```

`INDEX` is an advisory append-only log with one entry per write:

```text
kind|fingerprint|len|mtime
```

The fingerprint is part of the filename and the index entry. The current file
format does not embed an additional header inside the archive.

## Cache Keys

Every cache key includes:

- Provider identity bytes derived from input content, not the dataset path.
- Journal head, so any replay-visible curation change produces a miss.
- Artifact `algo_version`, bumped whenever the archive shape or computation changes.
- Artifact parameters such as dimensionality, channel index, bin size, window, or k-NN.
- Per-cluster spike fingerprints where applicable.

Per-cluster PC subspaces and isolation metrics include a fingerprint of the
cluster's current spike membership. Correlograms include spike-time
fingerprints for the cluster or ordered cluster pair. Cross-correlogram keys
are directional: `(a, b)` and `(b, a)` are different entries.

## Lifecycle

On read, Sorrel builds the key from the current provider identity, journal
head, parameters, and cluster inputs. A matching rkyv archive is memory-mapped
and bytechecked before use. Corrupt archives are treated as misses and evicted.

On miss, the caller computes the artifact normally and writes the archive
atomically through a temporary file plus rename. The next open can then use the
archive without recomputing.

When a `CurationCommand` is dispatched, the journal append changes the journal
head. The running session records a bounded in-memory invalidation log with the
previous head and affected cluster identities. Sorrel does not evict on every
command; it invalidates opportunistically at save and quit so the current
session keeps warm caches while editing. The journal-head component still
forces newly requested artifacts to miss after the command.

## Garbage Collection

`CacheStore::open` runs a bounded GC pass over the cache root. GC removes:

- Orphan `.rkyv` files that are not referenced by `INDEX`.
- Entries older than `SORREL_CACHE_MAX_AGE_DAYS`, default `30`.
- Least-recently-modified entries when total cache bytes exceed `SORREL_CACHE_MAX_BYTES`, default `4294967296` bytes.

The open-time sweep scans at most 1000 cache files per call to avoid turning a
large cache into a visible startup stall. Larger caches finish over multiple
opens.

## User Operations

Users can clear the cache without losing data:

```bash
sorrel --clear-cache <dataset>
```

This removes `<dataset>/.sorrel/cache/` and exits. Manual deletion of the same
directory is also safe. The command canonicalizes the target and refuses to
remove any path that is not a `.sorrel/cache` directory.

The cache is local rebuildable state. Sorrel does not guarantee cross-machine
cache portability, even if copying a dataset also copies `.sorrel/cache`.

## Boundaries

rkyv is a cache-only codec. It must not leak into:

- **Compute kernels** (`crates/sorrel-compute/`). The kernel crate stays
  dependency-free of rkyv; its types use plain Rust or serde.
- **JSON ingest** (`crates/sorrel-io/src/{probeinterface,sorting_analyzer,open_ephys}.rs`).
  External-format deserialisation continues to use `serde_json`.
- **The curation journal** (`crates/sorrel-data/src/journal.rs`, `command.rs`).
  The journal codec stays `rmp-serde` for schema evolution and forward
  compatibility.

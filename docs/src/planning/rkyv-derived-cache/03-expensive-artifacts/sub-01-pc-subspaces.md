# Sub-layer 03/01 — Cache PC subspaces

> **Recommended Codex model: GPT 5.5 medium**
>
> Moderate complexity, leaf-ish sub-agent role. The compute is
> already isolated in `collect_pc_subspace`; the work is wrapping
> it with a `CacheStore::get`/`put` pair under a per-cluster key
> that includes the cluster's spike-set identity (which changes on
> Merge/Split). The non-trivial call is what to include in the
> *per-cluster* fingerprint beyond the global session fingerprint.

## Working tree

`/data/nvme0/can/Projects/sorrel`. Depends on Phase 02 having
landed (`CacheStore` proven, `identity_bytes` trait method exists,
`journal.head()` accessor exists).

## Goal

`collect_pc_subspace(session, cluster, …)` consults
`CacheStore` first. On hit, returns a `PcSubspace` populated from a
memmapped archive without recomputing. On miss, computes as today
and `put`s. The cache key is per-cluster.

## Why this matters now

`collect_pc_subspace` is one of the hottest selection-driven
computations: it touches every spike in the session, sub-samples a
background pool, and produces a dense `(n_total, d)` buffer. Users
clicking through clusters in the UI re-trigger it constantly.
Memoization tied to journal head is the obvious win.

## Out of scope

- Caching the downstream isolation metric output. That's sub-02.
- Caching across changes in `d_pcs`, `channel_idx`, `max_background`
  — the cache key includes these params, so each combination has its
  own archive. Don't try to share archives across parameter values.

## Plan

1. Add `crates/sorrel-data/src/cache/pc_subspace.rs`:
   ```rust
   #[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
   #[rkyv(check_bytes)]
   pub struct CachedPcSubspace {
       pub features: Vec<f32>,
       pub is_in_cluster: Vec<bool>,
       pub d: u32,
       pub algo_version: u32,
   }

   pub const ALGO_VERSION: u32 = 1;

   pub fn key(
       session_identity: &[u8],
       journal_head: u64,
       cluster: ClusterId,
       d_pcs: usize,
       channel_idx: usize,
       max_background: usize,
   ) -> CacheKey { … }
   ```

2. Wire `collect_pc_subspace` in `feature_subspace.rs`:
   - Compute `key` from session/provider + journal head + params +
     the *cluster's* current spike-count and a hash of its spike
     indices. This is the per-cluster identity component — without
     it, post-merge clusters with the same `ClusterId` (rare, but
     possible after Split) would return stale subspaces.
   - Call `CacheStore::get`. On hit, copy slices out (or, better,
     return a `PcSubspace` whose `Vec<f32>` is a fresh allocation
     populated from the archive — `PcSubspace` is an owned type by
     contract today; don't break that).
   - On miss, recompute and `put`.

3. Per-cluster identity helper in `feature_subspace.rs`:
   ```rust
   fn cluster_spike_fingerprint(session: &Session<P>, cluster: ClusterId) -> [u8; 32];
   ```
   Hash the cluster's spike-index slice. This catches Merge/Split
   even if the journal-head bump didn't (defense in depth).

4. Tests in `crates/sorrel-data/src/cache/pc_subspace.rs`:
   - Roundtrip a synthetic `CachedPcSubspace` and verify byte-equal
     read.
   - Verify that incrementing `algo_version` busts the cache.
   - Verify that changing `d_pcs` produces a different key.

## Acceptance criteria

- [ ] `cargo test -p sorrel-data cache::pc_subspace` passes.
- [ ] Selecting the same cluster twice in the running binary emits
      one `cache miss pc_subspace/<hex>` followed by one `cache hit`.
- [ ] Merging two clusters and re-selecting the merged cluster
      emits a fresh `cache miss` (the spike-fingerprint changed).
- [ ] `collect_pc_subspace` still returns `Option<PcSubspace>` with
      the same semantics as before. No public-API change.

## Files likely touched

- `crates/sorrel-data/src/cache/pc_subspace.rs` (new).
- `crates/sorrel-data/src/cache/mod.rs` — one `pub mod pc_subspace;`.
- `crates/sorrel-data/src/feature_subspace.rs` — cache hook +
  `cluster_spike_fingerprint` helper.
- `crates/sorrel-data/src/lib.rs` — re-export if needed for tests.

## Pitfalls

- **Skipping the per-cluster spike fingerprint.** If you key only on
  journal head + ClusterId, a Split that produces a new cluster with
  a recycled ID could return the *original* cluster's archive.
  Symptom: garbled isolation metrics post-Split. Recovery: include
  `cluster_spike_fingerprint` in the key.
- **Returning `&Archived` straight out of the function.** The
  function signature returns `Option<PcSubspace>` (owned). Don't
  change the signature to leak a lifetime — keep it owned, and
  accept the one-allocation copy on cache hit. The win is avoiding
  the O(n_spikes × d) recompute, not avoiding the allocation.
- **Hashing the entire spike-index slice on every call.** For huge
  clusters this is itself non-trivial. Cache the per-cluster
  fingerprint inside `ClusterIndex` if it shows up in profiles —
  but only after profiling shows it. Don't pre-optimise.

## Reference

- Phase 02 — pattern this sub-layer replicates.
- `crates/sorrel-data/src/feature_subspace.rs` — current
  implementation.
- `crates/sorrel-compute/src/metrics_iso.rs` — downstream consumer
  (sub-02 caches its output).

# Sub-layer 03/03 — Cache correlograms (ACG + CCG)

> **Recommended Codex model: GPT 5.4 medium**
>
> Leaf sub-agent, mostly mechanical: the CCG kernels in
> `sorrel-compute/src/metrics.rs` (`auto_correlogram`,
> `cross_correlogram`) and the refractory analysis in
> `ccg_analysis.rs` already have clean signatures. The work is a
> straightforward `cache::correlograms` module plus consumer-side
> wiring. `5.4 medium` is enough; the design content is light.

## Working tree

`/data/nvme0/can/Projects/sorrel`. Depends on Phase 02. Compatible
with sub-01 and sub-02 landing in parallel — touches different files.

## Goal

ACG (per-cluster) and CCG (per-pair) results are cached. The ACG cache
key is per-cluster; the CCG cache key is per-ordered-pair. Both keyed
on per-cluster spike fingerprints + bin/window params + algo_version.

## Why this matters now

CCGs are recomputed every time the user opens the CCG matrix view or
hovers a pair. For datasets with hundreds of clusters, the pair count
is quadratic. A per-pair cache turns the matrix view from
"recompute everything" to "compute the few pairs whose participating
clusters changed since last open".

## Out of scope

- Caching the refractory-dip analysis output. It's a tiny derived
  number computed from the ACG; not worth a separate cache entry. If
  the ACG hits the cache, the refractory step recomputes from the
  archived ACG bins, which is essentially free.
- Changing the kernels in `sorrel-compute`. The cache wraps them; it
  does not alter their signatures or move them.

## Plan

1. Add `crates/sorrel-data/src/cache/correlograms.rs`:
   ```rust
   #[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
   #[rkyv(check_bytes)]
   pub struct CachedCorrelogram {
       pub bins: Vec<u32>,
       pub bin_size_samples: u32,
       pub window_samples: u32,
       pub algo_version: u32,
   }

   pub const ALGO_VERSION: u32 = 1;

   pub fn acg_key(session_identity: &[u8], journal_head: u64,
                  cluster_fp: [u8; 32],
                  bin_size_samples: u32, window_samples: u32) -> CacheKey;

   pub fn ccg_key(session_identity: &[u8], journal_head: u64,
                  a_fp: [u8; 32], b_fp: [u8; 32],
                  bin_size_samples: u32, window_samples: u32) -> CacheKey;
   ```
   The CCG key must commute: `ccg_key(a, b, …) == ccg_key(b, a, …)`
   *only* if the kernel produces symmetric output. Check
   `cross_correlogram`'s contract — it does *not* (the sign of the
   lag matters), so do NOT sort the fingerprints. Document this in
   the function doc.

2. Add a consumer-side adapter. The CCG matrix view is in the UI
   crate, but the cache hook belongs in a `sorrel-data` helper so
   the UI keeps consuming a thin API:
   ```rust
   // crates/sorrel-data/src/cache/correlograms.rs
   pub fn acg_or_compute<P: DataProvider>(
       session: &Session<P>, cluster: ClusterId,
       bin_size_samples: u32, window_samples: u32,
   ) -> Vec<u32> { … }

   pub fn ccg_or_compute<P: DataProvider>(
       session: &Session<P>, a: ClusterId, b: ClusterId,
       bin_size_samples: u32, window_samples: u32,
   ) -> Vec<u32> { … }
   ```
   Re-export from `crates/sorrel-data/src/lib.rs`. UI sites can be
   updated opportunistically; this sub-layer's acceptance criteria
   do not require the UI flip — only that the helpers exist and pass
   tests.

3. Tests:
   - Roundtrip a `CachedCorrelogram`.
   - `acg_or_compute` called twice → second call returns identical
     bins without invoking the kernel (use a flag-counting fake
     provider).
   - `ccg_or_compute(a, b, …)` and `ccg_or_compute(b, a, …)` produce
     *different* cache entries (lag sign matters).

## Acceptance criteria

- [ ] `cargo test -p sorrel-data cache::correlograms` passes.
- [ ] `acg_or_compute` and `ccg_or_compute` are public and exported
      from `sorrel_data`.
- [ ] Repeated calls in tests produce one miss then one hit per key.
- [ ] `rg "rkyv" crates/sorrel-compute/` returns nothing.

## Files likely touched

- `crates/sorrel-data/src/cache/correlograms.rs` (new).
- `crates/sorrel-data/src/cache/mod.rs` — one `pub mod correlograms;`.
- `crates/sorrel-data/src/lib.rs` — re-export `acg_or_compute`,
  `ccg_or_compute`.

## Pitfalls

- **Symmetric-key shortcut.** Sorting `(a, b)` fingerprints before
  hashing collapses the two directions of a directed CCG into one
  cache entry. Symptom: the CCG matrix shows the same plot on
  reflection. Recovery: never sort; the pair is ordered.
- **Including the raw bin count in the key but not the window/bin
  params.** Two different `(window, bin)` choices can produce the
  same bin count by coincidence. Always include the params
  explicitly.
- **Caching when one of the clusters has 0 spikes.** Just return an
  empty `Vec` without touching the cache. Zero-spike clusters churn
  on merges/splits and flooding the cache with empty entries is
  pointless.

## Reference

- Phase 02 — pattern.
- `crates/sorrel-compute/src/metrics.rs` — `auto_correlogram`,
  `cross_correlogram` (read-only).
- `crates/sorrel-compute/src/ccg_analysis.rs` — `analyse_refractory_dip`
  (downstream consumer of the cached bins).

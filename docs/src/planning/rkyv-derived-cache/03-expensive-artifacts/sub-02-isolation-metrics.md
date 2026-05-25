# Sub-layer 03/02 — Cache isolation metrics

> **Recommended Codex model: GPT 5.5 medium**
>
> Moderate complexity, leaf sub-agent. The output is a small struct
> (`IsolationMetrics`) but the computation is expensive: brute-force
> k-NN over the PC subspace. Caching it is high value because the UI
> displays these per cluster. Routed `medium` because the cache key
> must be composed carefully — it depends on the subspace key from
> sub-01 plus the metric parameters.

## Working tree

`/data/nvme0/can/Projects/sorrel`. Depends on Phase 02. Compatible
with sub-01 landing in parallel — does not edit `feature_subspace.rs`.

## Goal

`compute_isolation` (in `crates/sorrel-data/src/quality_ext.rs`) is
cache-aware. The cached artifact is the small `IsolationMetrics`
struct. Key includes the inputs that determine the subspace plus the
metric's own parameters.

## Why this matters now

`isolation_metrics` runs a brute-force k-NN over the PC subspace —
O(n_in × n_total × d). For large clusters this dominates the cost of
opening the cluster panel. The downstream artifact (an
`IsolationMetrics` struct of a handful of floats) is *small*, so the
cache hit is essentially instant.

## Out of scope

- Caching the PC subspace itself. That's sub-01.
- Caching `cluster_quality` (the cheap part of `quality_ext`).
- Changing the metric kernel in `sorrel-compute/src/metrics_iso.rs`.

## Plan

1. Add `crates/sorrel-data/src/cache/isolation.rs`:
   ```rust
   #[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
   #[rkyv(check_bytes)]
   pub struct CachedIsolationMetrics {
       pub nn_isolation: f32,
       pub nn_hit_rate: f32,
       pub nn_miss_rate: f32,
       // …mirror IsolationMetrics fields exactly…
       pub algo_version: u32,
   }

   pub const ALGO_VERSION: u32 = 1;
   ```
   Mirror every field of `IsolationMetrics` from `sorrel-compute`.
   Do not derive rkyv on `IsolationMetrics` itself — that would pull
   rkyv into `sorrel-compute`, which is forbidden by the plan-set
   constraints. Convert at the cache-module boundary.

2. Wire `compute_isolation` in `quality_ext.rs`:
   - Build the key from the same inputs used by sub-01's PC subspace
     key (session identity, journal head, cluster, `d_pcs`,
     `channel_idx`, `max_background`, per-cluster spike fingerprint)
     plus the isolation-specific params (k, etc.).
   - `get` → on hit, return the converted `IsolationMetrics`.
   - On miss, compute and `put`.

3. Tests in the new module:
   - Roundtrip the cached struct.
   - Bumping `algo_version` busts the cache.
   - Different `k` values produce different keys.

## Acceptance criteria

- [ ] `cargo test -p sorrel-data cache::isolation` passes.
- [ ] Selecting a cluster, then re-selecting it, emits one cache miss
      then one cache hit for `isolation`.
- [ ] `rg "rkyv" crates/sorrel-compute/` returns nothing.
- [ ] `IsolationMetrics` public API in `sorrel-compute` unchanged.

## Files likely touched

- `crates/sorrel-data/src/cache/isolation.rs` (new).
- `crates/sorrel-data/src/cache/mod.rs` — one `pub mod isolation;`.
- `crates/sorrel-data/src/quality_ext.rs` — cache hook in
  `compute_isolation`.

## Pitfalls

- **Deriving rkyv on `IsolationMetrics` in sorrel-compute.** Would
  pull rkyv into the compute crate, violating the layering. Always
  define the cache shape in `sorrel-data/src/cache/isolation.rs` and
  convert at the boundary.
- **Reusing sub-01's key wholesale.** Looks tempting but loses the
  isolation-specific params. Compose the key as `subspace_key ⊕
  metric_params`, not `subspace_key` alone. Symptom: changing `k`
  silently returns the wrong metrics.
- **Caching when `collect_pc_subspace` returns `None`.** Don't put a
  "no subspace available" sentinel into the cache — just return
  `None` and skip the put. Otherwise a future fix that makes
  subspaces available still returns the cached "unavailable" result.

## Reference

- Phase 02 — pattern.
- Sub-01 — owns the subspace cache; this sub-layer's key extends it.
- `crates/sorrel-compute/src/metrics_iso.rs` — kernel (read-only).
- `crates/sorrel-data/src/quality_ext.rs` — current call site.

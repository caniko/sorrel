# Phase 03 — Cache expensive per-cluster artifacts (multi-sub-layer)

> **Recommended Codex model for merge/orchestration: GPT 5.5 medium**
>
> The phase-level merge is straightforward — three sub-layers land
> independently and the only merge is checking that the three new
> `cache::*` modules are wired into `crates/sorrel-data/src/cache/mod.rs`
> without conflict. `5.5 medium` is enough.

## Sub-layers

| # | Slug | Model | Touches | Sub-layer file |
|---|------|-------|---------|----------------|
| 01 | pc-subspaces | 5.5 medium | `crates/sorrel-data/src/feature_subspace.rs`, new `crates/sorrel-data/src/cache/pc_subspace.rs` | [sub-01-pc-subspaces.md](./sub-01-pc-subspaces.md) |
| 02 | isolation-metrics | 5.5 medium | `crates/sorrel-data/src/quality_ext.rs`, new `crates/sorrel-data/src/cache/isolation.rs` | [sub-02-isolation-metrics.md](./sub-02-isolation-metrics.md) |
| 03 | correlograms | 5.4 medium | new `crates/sorrel-data/src/cache/correlograms.rs`, call-site adapter near the CCG view | [sub-03-correlograms.md](./sub-03-correlograms.md) |

Sub-layers route at or below the phase-level tier per
`multi-phase-dispatch` guidance. Sub-03 is the most mechanical (the
CCG kernel signature is already clean) and drops to `5.4 medium`.

## Goal (phase-level)

Three of the most expensive derived-data computations
(`collect_pc_subspace`, `isolation_metrics`,
`auto_correlogram`/`cross_correlogram`) are cache-aware:
recompute-on-miss, memmap-on-hit. Each follows the pattern proven by
Phase 02 — `CacheKey` over inputs + journal head + algo_version,
guard-borrowed zero-copy reads.

## Why this matters now

These three are the real targets of the rkyv layer. Each is recomputed
on every cluster selection or view refresh and costs O(n_spikes × d)
or worse per call. With the journal-head fingerprint, results stay
valid until the user merges or splits — exactly the workflow shape
that benefits from on-disk memoization.

## Out of scope

- Caching snippets / mean snippets. Different lifetime story (raw
  trace window depends on render-time params); revisit after Phase 04
  lands and a clean invalidation story for raw-trace-derived data
  exists.
- Caching quality breakdowns (`QualityBreakdown`). Cheap to compute;
  not worth a cache entry.
- Caching anything in `sorrel-compute` directly. The cache layer
  lives in `sorrel-data`; `sorrel-compute` stays as pure kernels.
- Touching the journal codec. Still off-limits.

## Merge plan

Sub-layers can be developed in three fresh agent sessions in parallel.
They touch disjoint files:
- sub-01: `feature_subspace.rs` + new `cache/pc_subspace.rs`.
- sub-02: `quality_ext.rs` + new `cache/isolation.rs`.
- sub-03: a new call-site adapter + new `cache/correlograms.rs`.

The only shared file is `crates/sorrel-data/src/cache/mod.rs` (each
sub-layer adds one `pub mod …` line). Treat its conflicts as trivial:
the user merging the sub-layer branches resolves by accepting all
three `pub mod` lines, sorted alphabetically.

`crates/sorrel-data/src/lib.rs` may also need three new re-exports;
treat the same way.

## Phase-level acceptance criteria

- [ ] `cargo check --workspace` clean.
- [ ] `cargo test -p sorrel-data` passes, including the per-sub-layer
      roundtrip tests added in each sub-layer.
- [ ] `cache::{pc_subspace, isolation, correlograms}` modules all
      exist and are re-exported from `crates/sorrel-data/src/lib.rs`.
- [ ] Opening a real Kilosort dataset twice with the binary, and
      selecting the same cluster in both runs, emits one `cache miss`
      and one `cache hit` line per artifact kind in the second run's
      debug log.
- [ ] `rg "rkyv" crates/sorrel-compute/ crates/sorrel-io/` returns
      nothing. The kernels and ingest stay pure.
- [ ] No regression in the Phase 02 bench
      (`seed_amplitudes_cache.rs`) — warm-cache ratio still ≥ 4×.

## Reference

- Phase 01 — `CacheStore`, `CacheKey`, `Fingerprint`.
- Phase 02 — first consumer; established pattern these sub-layers
  replicate.
- `crates/sorrel-data/src/feature_subspace.rs` — `collect_pc_subspace`
  call site.
- `crates/sorrel-data/src/quality_ext.rs` — `cluster_quality` /
  `compute_isolation` call sites.
- `crates/sorrel-compute/src/ccg_analysis.rs`,
  `crates/sorrel-compute/src/metrics.rs` — CCG kernels (read-only
  reference; the consumer cache wraps them, doesn't modify them).

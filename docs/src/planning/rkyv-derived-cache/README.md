# Plan: rkyv-backed derived-data cache

> **Recommended Codex model for plan-set orchestration: GPT 5.5 high**
>
> Orchestrating this plan means holding the cache-key invariants in
> mind across four phases that touch disjoint crates but share one
> on-disk contract. Complex × orchestrator coordinates land at
> `5.5 high`. Individual phases route lower.

## Scope

Introduce **rkyv** as a derived-data cache layer for expensive computed
artifacts (PC subspaces, isolation metrics, snippets, correlograms,
amplitude/template-waveform buckets) in a way that:

- Preserves `serde_json` for SpikeInterface / Open Ephys / Kilosort
  JSON ingest. Untouched.
- Preserves `rmp-serde` MessagePack for the long-lived, schema-evolving
  curation journal (`crates/sorrel-data/src/journal.rs`,
  `command.rs`). Untouched.
- Adds a new `sorrel-cache` crate that owns the rkyv archives,
  cache-key fingerprinting, on-disk layout, atomic writes, GC, and
  memmap-backed zero-copy reads.
- Wires consumers in `sorrel-data` (seeded per-cluster buckets) and
  `sorrel-compute` (isolation metrics, CCGs) to populate and read the
  cache. The serde derives on those types are left intact — rkyv is
  *additional*, not a replacement.

## Current state

- `serde` + `serde_json` ingest external JSON formats in
  `crates/sorrel-io/src/{probeinterface,sorting_analyzer,open_ephys}.rs`.
- `rmp-serde` encodes the curation journal in
  `crates/sorrel-data/src/journal.rs` and `command.rs`.
- `baseline_hash` already exists at
  `crates/sorrel-data/src/journal.rs` (FNV-style hash over
  `spike_clusters` bytes). The journal head is currently not surfaced
  as a value the rest of the code can fingerprint against; phase 01
  promotes it.
- `Session::seed_*` calls bucket amplitudes, PC features, template
  waveforms, and spike templates once per open
  (`crates/sorrel-data/src/session.rs`). These are recomputed cold on
  every launch — first target for caching.
- `collect_pc_subspace`, `isolation_metrics`, and CCG kernels in
  `sorrel-compute` are recomputed on every cluster selection / view
  refresh. Second target.

## Phase table

| Phase | File | Model | Touches | Depends on | Can parallel with |
|---|---|---|---|---|---|
| 01 | [01-cache-infrastructure.md](./01-cache-infrastructure.md) | 5.5 high | new `crates/sorrel-cache/`, workspace `Cargo.toml`, `flake.nix`/`nix/deps.nix`, `docs/src/architecture/workspace-layout.md` | — | — |
| 02 | [02-first-consumer.md](./02-first-consumer.md) | 5.5 medium | `crates/sorrel-data/src/session.rs`, new bench under `crates/sorrel-data/benches/` | 01 | — |
| 03 | [03-expensive-artifacts/](./03-expensive-artifacts/README.md) (multi-sub-layer) | 5.5 medium (merge) + per sub-layer | `crates/sorrel-data/src/feature_subspace.rs`, `crates/sorrel-data/src/quality_ext.rs`, `crates/sorrel-compute/src/ccg_analysis.rs` (read-only; consumer sites in `sorrel-data`) | 02 | sub-layers parallelise internally |
| 04 | [04-invalidation-gc-docs.md](./04-invalidation-gc-docs.md) | 5.5 high | `crates/sorrel-data/src/session.rs` (replay hook), `crates/sorrel-cache/src/gc.rs`, `docs/src/architecture/data-flow.md`, new `docs/src/architecture/cache.md` | 02 (minimum); ideally 03 | — |

## Parallelism layer

- **Wave 0** — Phase 01 alone. It defines the `CacheKey`,
  `Fingerprint`, and `CacheStore` types that every later phase imports.
  Nothing else can start cold.
- **Wave 1** — Phase 02 once 01 lands. Single-stream phase; proves the
  end-to-end memmap roundtrip on the cheapest artifact
  (`seed_amplitudes` bucket).
- **Wave 2** — Phase 03 once 02's roundtrip is green. Phase 03 is
  multi-sub-layer; its three sub-layers (PC subspaces, isolation
  metrics, correlograms) touch disjoint files and can fan out to three
  fresh agent sessions.
- **Wave 3** — Phase 04 once Phase 03 has at least sub-01 landed (the
  PC subspace cache is what most exercises invalidation on
  Merge/Split). Phase 04 can technically start after Phase 02, but
  pre-mortems land better with a real per-cluster consumer in tree.

Serialisation points:
- `crates/sorrel-data/src/session.rs` is touched by Phase 02 and
  Phase 04. Phase 04 must rebase on Phase 02.
- `Cargo.toml` (root) is touched by Phase 01 only.
- The three sub-layers of Phase 03 touch three different files each
  and do not collide.

## Whole-set acceptance criteria

- `cargo check --workspace` and `cargo test --workspace` pass after
  every phase.
- `sorrel-cache` crate exists, is consumed by `sorrel-data` only
  (`sorrel-compute` stays free of rkyv; it remains pure kernels).
- On a warm cache, opening a Kilosort dataset re-uses cached
  `seed_amplitudes` output without recomputing — verified by a debug
  log line `cache hit: amplitudes/<fingerprint>` and a wall-clock
  delta visible in the Phase 02 bench.
- Cache invalidation: applying any `CurationCommand` that changes
  cluster identity (`Merge`, `Split`, `Batch` containing either)
  invalidates the per-cluster artifacts touched. Verified by a unit
  test in `sorrel-cache` that wires a fake journal head.
- `serde_json` ingest sites are byte-for-byte unchanged. `rmp-serde`
  journal sites are byte-for-byte unchanged. Grep verification:
  `git diff trunk... -- crates/sorrel-io/src/{probeinterface,sorting_analyzer,open_ephys}.rs crates/sorrel-data/src/{journal,command}.rs`
  reports no functional changes (formatting / import-only OK).
- New `docs/src/architecture/cache.md` exists, is linked from
  `SUMMARY.md`, and documents the cache-key composition + GC policy.

## Global constraints

- **Do not touch JSON ingest.** Phases must not import rkyv or rkyv
  derives into `crates/sorrel-io/src/{probeinterface,sorting_analyzer,open_ephys}.rs`.
- **Do not touch the journal codec.** `journal.rs` continues to use
  `rmp-serde`. The cache imports `baseline_hash` and a new
  journal-head accessor; it does not change how journal records are
  written.
- **Cache is derivable.** Every cached artifact must be reproducible
  from inputs + params + algo version. Treat the cache directory as
  free to delete at any time. No data-loss path can run through it.
- **Cache directory lives under the dataset, not under `$HOME`.**
  Convention: `<dataset_dir>/.sorrel/cache/<kind>/<fingerprint>.rkyv`.
  This keeps the cache co-located with the inputs it depends on and
  makes "delete to invalidate" obvious to users.
- **rkyv is an additive derive.** Types in `sorrel-data` that get an
  `Archive` derive keep their existing `serde::{Serialize, Deserialize}`
  derives if they already had them.
- **Endianness portability.** Use rkyv's portable feature flag so
  cache files written on one host can be read on another.
  Cross-arch cache sharing isn't a target, but in-place endian
  surprises are not a bug we want to debug.

## Reference

- Discussion seed: "why are we using serde? Wouldn't rkyv be better?"
  → conclusion that rkyv is the right tool for derived-data caches,
  not for JSON ingest or the journal.
- `crates/sorrel-data/src/journal.rs:baseline_hash` — existing
  fingerprint primitive the cache extends.
- `docs/src/architecture/data-flow.md` — describes the open / edit /
  save / render pipeline this plan slots into.

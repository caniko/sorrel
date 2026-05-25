# Phase 02 — First consumer: cache `seed_amplitudes` end-to-end

> **Recommended Codex model: GPT 5.5 medium**
>
> Moderate complexity, sub-agent role: the surface is defined by
> Phase 01, the artifact (per-cluster amplitude buckets) is already
> computed in `Session::seed_amplitudes`, and the wiring is mostly
> "check cache, compute on miss, put on success". The non-trivial
> calls are the cache key composition for *this* artifact and the
> benchmark setup that proves we actually saved work. `5.5 medium` is
> the right tier: a `low` model would skip the key composition checks
> or silently regress correctness on the journal-head input.

## Working tree

`/data/nvme0/can/Projects/sorrel`. Same repo. **Depends on Phase 01
having landed** — `sorrel-cache` must exist and expose `CacheStore`,
`CacheKey`, `Fingerprint`. Pull / rebase before starting.

## Goal

`Session::seed_amplitudes` becomes cache-aware: on open, it computes
a `CacheKey` over (provider identity, journal head, algo_version),
calls `CacheStore::get`, returns the memmapped buckets if present,
and computes + `put`s on miss. A benchmark in
`crates/sorrel-data/benches/` measures cold vs warm open and emits a
visible wall-clock delta.

## Why this matters now

This is the proof of life for the entire cache layer. If the
memmap-backed zero-copy read path can't be wired to one real consumer
cleanly, Phase 03's three sub-layers are at risk of all replicating
the same plumbing mistake. Land one consumer first, in the simplest
artifact shape (a `Vec<f32>` per cluster), so the next three reuse a
known-good pattern.

`seed_amplitudes` is chosen because:
- It's already a "compute once on open" call, so the cache fits the
  call site without rearchitecture.
- The output is a flat numeric buffer — rkyv loves these.
- It's not on the per-frame render path, so a wrong cache key
  surfaces as a wrong-numbers bug at open time (loud and fast), not
  as a subtle frame-rate regression.

## Out of scope

- Caching PC subspaces, isolation metrics, snippets, or correlograms.
  Phase 03.
- Invalidation on Merge/Split mid-session. The journal head changes
  on those, so the *next* open will re-fingerprint and miss the
  cache, which is correct. Mid-session invalidation is Phase 04.
- GC of stale entries. Phase 04.
- Touching `seed_template_waveforms`, `seed_pc_features`, or
  `seed_spike_templates` even though they have the same shape.
  Resist the urge — those land cleanly under Phase 03 sub-layers.

## Plan

1. Add `sorrel-cache.workspace = true` to
   `crates/sorrel-data/Cargo.toml`.

2. Define the cached schema in
   `crates/sorrel-data/src/cache/amplitudes.rs` (new module):
   ```rust
   #[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
   #[rkyv(check_bytes)]
   pub struct CachedAmplitudes {
       pub per_cluster: Vec<Vec<f32>>,
       pub algo_version: u32,
   }
   ```
   The `algo_version` field inside the archive is belt-and-braces:
   the key already encodes it, but storing it in the payload makes
   stale archives self-identify when debugging.

3. Add `crates/sorrel-data/src/cache/mod.rs` declaring the submodule
   and re-exporting the artifact types. Wire it from `lib.rs`.

4. Modify `Session::seed_amplitudes` in
   `crates/sorrel-data/src/session.rs`:
   - Build a `Fingerprint` over:
     - `provider.identity_bytes()` (new trait method, see step 5).
     - The current journal head (new accessor, see step 6).
     - `algo_version` (compile-time constant in the `cache::amplitudes`
       module).
   - Call `CacheStore::get`. On `Some`, copy the archived buckets
     into the cluster index and skip the recompute. On `None`,
     recompute as today and `put`.
   - Emit a single `tracing::debug!` line: `cache hit amplitudes/<hex>`
     or `cache miss amplitudes/<hex>`.

5. Extend the `DataProvider` trait in
   `crates/sorrel-io/src/provider.rs` with:
   ```rust
   /// Stable bytes identifying this provider's *input data*, not its
   /// path. Implementations hash file contents (or use mtime+len for
   /// huge raw files) to produce a stable fingerprint that the cache
   /// can key on.
   fn identity_bytes(&self) -> Vec<u8>;
   ```
   Implementations:
   - Kilosort: digest `spike_clusters.npy` + `spike_times.npy` headers
     plus mtime+len of the raw `.dat`.
   - SortingAnalyzer / NWB / Ks4Rez: equivalent.
   Don't go deep on each backend — a one-liner per provider is fine
   for this phase. Phase 03 sub-layers can refine if a consumer
   needs more.

6. Add a journal-head accessor in
   `crates/sorrel-data/src/journal.rs`:
   ```rust
   impl SqliteJournal {
       pub fn head(&self) -> u64 { /* sealed-baseline + applied-row count or a row-hash chain */ }
   }
   ```
   Pick the cheapest stable encoding — a monotonic counter combined
   with the sealed baseline hash is enough, since the cache only
   needs *changes* to journal-head to invalidate.

7. Add a benchmark in
   `crates/sorrel-data/benches/seed_amplitudes_cache.rs`:
   - Fixture: a small synthetic provider (n_spikes ≈ 1e6, n_clusters
     ≈ 100). Put the synthesizer behind a `#[cfg(test)]` helper in
     `session.rs` rather than reading real data — bench fixtures
     belong in-tree.
   - Two scenarios: cold cache (empty `.sorrel/cache/`) and warm
     cache (pre-populated).
   - Print the ratio. Don't gate CI on it — the threshold belongs in
     Phase 04 when GC is wired.

8. Update `docs/src/architecture/data-flow.md`: add a one-paragraph
   note that `seed_*` calls now consult `.sorrel/cache/` first and
   fall through to the existing compute.

## Acceptance criteria

- [ ] `cargo check --workspace` clean.
- [ ] `cargo test -p sorrel-data` passes.
- [ ] `cargo bench -p sorrel-data --bench seed_amplitudes_cache`
      reports a warm-cache time at least 4× faster than cold on the
      synthetic fixture. (Order-of-magnitude target. If it's only
      2×, something is wrong — likely a deserialise-to-owned path
      sneaking in instead of zero-copy memmap reads.)
- [ ] Running the binary on a real Kilosort dataset twice produces
      `cache miss amplitudes/<hex>` then `cache hit amplitudes/<hex>`
      in the second open's debug log.
- [ ] `git diff trunk... -- crates/sorrel-io/src/{probeinterface,sorting_analyzer,open_ephys}.rs crates/sorrel-data/src/{journal,command}.rs`
      shows only the `head()` accessor addition in `journal.rs` and
      the `identity_bytes` trait method propagation in providers —
      no codec changes, no serde→rkyv flips.
- [ ] `rg "rkyv" crates/sorrel-io/` returns nothing (rkyv stays out
      of `sorrel-io`).

## Files likely touched

- `crates/sorrel-data/Cargo.toml` — add `sorrel-cache` dep.
- `crates/sorrel-data/src/cache/mod.rs` (new).
- `crates/sorrel-data/src/cache/amplitudes.rs` (new).
- `crates/sorrel-data/src/session.rs` — cache hook in
  `seed_amplitudes`.
- `crates/sorrel-data/src/lib.rs` — re-export the new `cache` module.
- `crates/sorrel-data/src/journal.rs` — `head()` accessor.
- `crates/sorrel-data/benches/seed_amplitudes_cache.rs` (new).
- `crates/sorrel-io/src/provider.rs` — `identity_bytes()` trait method.
- `crates/sorrel-io/src/{kilosort,sorting_analyzer,nwb,ks4_rez}.rs`
  — implement `identity_bytes()`. Light touch; one helper.
- `docs/src/architecture/data-flow.md` — one-paragraph add.

## Pitfalls

- **Path-based identity.** If `identity_bytes` ends up containing the
  absolute path, every relocation of the dataset busts the cache.
  Symptom: cache hit rate is 0% in any user's workflow that moves
  files. Cause: shortcut implementation. Recovery: hash file bytes
  (or mtime+len for raw `.dat`), not paths.
- **`deserialize` instead of zero-copy.** rkyv's `deserialize` builds
  an owned `Vec<Vec<f32>>` — that defeats the whole point. The
  consumer code should hold a `CacheRead<CachedAmplitudes>` guard
  and read `&archive.per_cluster[i]` slices directly. Symptom: the
  bench shows only 2× warm speedup. Cause: someone wrote
  `let owned = rkyv::deserialize(...)`. Recovery: borrow, don't own.
- **Journal head defined as a row count.** A row count collides
  trivially across sessions that revert to baseline. Combine with
  the sealed-baseline hash so head = `(baseline_hash, applied_count)`.
- **Forgetting algo_version inside the payload.** Without it, you
  can't tell at a glance which cache files were written by which
  algorithm rev. Bench-debugging pain later. Include it.
- **Bench fixture too small.** With n_spikes = 1e4 the cache hit is
  faster than the cold compute by a constant factor that's lost in
  noise. Use 1e6 spikes minimum.

## Reference

- Phase 01 — defines `CacheStore`, `CacheKey`, `Fingerprint`.
- `crates/sorrel-data/src/session.rs:seed_amplitudes` — the call site
  being modified.
- `docs/src/architecture/data-flow.md` — "Open" section is the place
  to add the one-paragraph cache note.

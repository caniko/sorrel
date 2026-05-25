# Phase 04 — Invalidation hooks, GC, and docs

> **Recommended Codex model: GPT 5.5 high**
>
> Complex orchestrator role: this phase touches the cross-cutting
> invariant that "any change to journal head invalidates dependent
> artifacts", wires GC, and writes the user-facing doc that
> establishes the contract for the cache directory. Mediocre work
> here ships subtle wrong-answer bugs (stale cache served after a
> Merge) that are very expensive to find. Worth `high`. Not `max`
> because the design is constrained by the contracts Phases 01–03
> already locked in — this phase enforces them, doesn't reopen them.

## Working tree

`/data/nvme0/can/Projects/sorrel`. Depends on Phase 02 minimum;
ideally Phase 03 sub-01 has landed so invalidation can be tested
against a real per-cluster consumer.

## Goal

The cache invalidates correctly on `CurationCommand` application,
GC reclaims stale entries on open, the UI has a "clear cache" hook,
and `docs/src/architecture/cache.md` exists and documents the cache
contract end-to-end.

## Why this matters now

Phases 01–03 establish the cache surface and three consumers. They
rely on the journal-head fingerprint to invalidate, which is
*correct on next open* but does nothing about:

- The in-memory `CacheStore` returning stale entries to the same
  session that just applied a Merge. (Current behaviour: the next
  `get` rebuilds the key with the new journal head and misses.
  That's fine — but a leftover archive sits on disk until GC.)
- Disk bloat. Each Merge/Split potentially writes a fresh archive
  while leaving the old one behind. Without GC the cache directory
  grows without bound.
- Users who don't know the cache exists and can't find a way to
  clear it when they suspect bad cached data.

## Out of scope

- Adding more consumers. If a fourth artifact is worth caching, it's
  its own phase.
- Changing the cache file format. The layout from Phase 01 stays.
- Cross-machine cache portability. Out of scope; the cache lives
  next to the dataset, so it travels with the dataset by file copy
  if the user wants.

## Plan

1. Add a `Session` hook that tracks journal head deltas:
   - In `crates/sorrel-data/src/session.rs`, after a successful
     `CurationCommand` apply, record the (prev_head, new_head) pair
     and the affected clusters in a small in-memory log.
   - Expose `Session::invalidate_caches(&self, store: &CacheStore)`
     that walks the log and calls `store.evict` on every key that
     could be derived from the prior cluster identities.
   - Call it opportunistically: at save time and at quit time.
     **Do not** call it on every command application — that
     defeats the cache's whole-session warmth.

2. Implement GC in `crates/sorrel-cache/src/gc.rs`:
   - On `CacheStore::open`, scan the cache root, compare each file's
     embedded fingerprint to the INDEX, and evict orphans (files
     not referenced by any in-INDEX entry) and entries older than a
     `max_age` (default 30 days, configurable via env var
     `SORREL_CACHE_MAX_AGE_DAYS`).
   - Cap total cache size at a `max_bytes` (default 4 GiB, env
     `SORREL_CACHE_MAX_BYTES`). On overflow, evict by LRU using
     mtime.
   - Log a single summary line: `cache gc: evicted N files, M MiB`.

3. UI hook (optional in this phase if `sorrel-ui` work is
   prohibitive; ship as a CLI flag at minimum):
   - `sorrel --clear-cache <dataset>` removes
     `<dataset_dir>/.sorrel/cache/` entirely and exits.
   - If the UI is touched, add a menu item under a Debug menu:
     "Clear derived cache" → calls the same path.

4. Write `docs/src/architecture/cache.md`:
   - What the cache stores. What it does not store (journal,
     external JSON).
   - Cache-key composition (provider identity, journal head,
     per-cluster spike fingerprint where applicable, algo_version,
     params).
   - Lifecycle: written lazily on miss, read zero-copy on hit,
     evicted by GC.
   - User-facing operations: where it lives, how to delete it,
     env-var knobs.
   - Invariants: "deleting the cache directory must never lose user
     data" — explicit.

5. Update `docs/src/architecture/data-flow.md`: expand the
   one-paragraph note from Phase 02 into a short "Cache" section
   that references `cache.md` and notes the invalidation point.

6. Update `docs/src/SUMMARY.md`: add the new `cache.md` under
   Architecture.

7. Tests in `crates/sorrel-cache/src/gc.rs`:
   - GC evicts orphan files.
   - GC respects `max_bytes` (LRU eviction).
   - GC respects `max_age` (mtime eviction).
   - `--clear-cache` is exercised by an integration test that
     populates the cache, runs the flag, and asserts the directory
     is empty.

## Acceptance criteria

- [ ] `cargo check --workspace` clean.
- [ ] `cargo test --workspace` passes.
- [ ] Applying a `Merge` followed by a `Save` results in the
      pre-merge per-cluster archives being evicted on the next
      `CacheStore::open` (verified by a unit test, not by visual
      inspection).
- [ ] Running `sorrel --clear-cache <dataset>` empties
      `<dataset_dir>/.sorrel/cache/` and exits with status 0.
- [ ] `docs/src/architecture/cache.md` exists, is linked from
      `SUMMARY.md`, and documents the cache-key composition + GC
      policy.
- [ ] `mdbook build docs/` produces no warnings about broken links.
- [ ] On a real dataset, opening twice with a `--clear-cache`
      between produces the same outputs in both views to the eye
      (sanity smoke; no scripted check).

## Files likely touched

- `crates/sorrel-data/src/session.rs` — invalidation log + hook.
- `crates/sorrel-cache/src/gc.rs` (new).
- `crates/sorrel-cache/src/lib.rs` — re-export GC entry points.
- `crates/sorrel/src/main.rs` (or wherever CLI args live) —
  `--clear-cache` flag.
- `crates/sorrel-ui/...` — optional menu item.
- `docs/src/architecture/cache.md` (new).
- `docs/src/architecture/data-flow.md` — expand the cache note.
- `docs/src/SUMMARY.md` — link the new doc.

## Pitfalls

- **Invalidating too aggressively.** Calling `invalidate_caches` on
  every `CurationCommand` evicts entries that would still be valid
  if the user undoes the command. Symptom: cache hit rate plummets
  during normal use. Recovery: invalidate at save/quit only;
  rely on the journal-head fingerprint to drive next-open misses.
- **GC running on every open synchronously on huge caches.** A
  multi-GB cache with thousands of files can make GC a visible
  open-time stutter. Symptom: open feels slower in the second run.
  Recovery: cap GC's work-per-open (e.g. scan ≤ 1000 entries per
  call); finish the sweep over multiple opens.
- **`--clear-cache` deleting outside the cache directory.** Always
  canonicalize the path and assert it ends in `.sorrel/cache/`
  before any `remove_dir_all` call. Symptom in worst case: user
  data loss. Recovery: defensive path checks; never `remove_dir_all`
  on a path that didn't come from `CacheStore::root`.
- **Docs claiming cross-machine cache sharing.** It is *not* a
  supported contract. The doc must say so explicitly.
- **Invalidation log growing without bound in long sessions.** Cap
  it (last 1024 commands or so) — older entries' targets will be
  collected by GC on next open anyway.

## Reference

- Phase 01 — `CacheStore`, atomic write, INDEX file.
- Phase 02 — journal-head fingerprint introduction.
- Phase 03 — three consumer call sites whose archives the
  invalidation log must reference.
- `crates/sorrel-data/src/session.rs` — current command-apply path
  (the in-memory mutation step after journal append).
- `docs/src/architecture/data-flow.md` — Open / Edit / Save sections.

# Phase 01 — Cache infrastructure: crate, keys, on-disk layout

> **Recommended Codex model: GPT 5.5 high**
>
> This is the foundational phase: it defines the `CacheKey`,
> `Fingerprint`, and `CacheStore` surfaces that every later phase
> imports. The work is mostly mechanical (new crate, derive setup,
> file-format decisions), but the design calls — what the cache key
> *includes*, what GC policy looks like, how memmap reads interact
> with concurrent writes — are non-trivial and hard to revisit later
> without churning every consumer. Routed orchestrator-complex at
> `5.5 high`. A smaller model is likely to skip the
> portability/endianness setting or to bake provider paths into the
> key instead of content fingerprints.

## Working tree

`/data/nvme0/can/Projects/sorrel`. Same repo as the parent project.

## Goal

A new `sorrel-cache` crate exists in the workspace, exposes a typed
`CacheStore`, defines `CacheKey` as content-addressed fingerprint over
(input identity, params, algo version, journal head), provides an
atomic write path and a memmap-backed zero-copy read path, and is
covered by unit tests on a tmp-dir fixture. **No consumer is wired
yet** — that's Phase 02.

## Why this matters now

The discussion that spawned this plan concluded that rkyv is the right
tool for *derived* data only, where the cache can be invalidated and
rebuilt freely. The constraint — "cache is derivable, never load-bearing"
— must be encoded in the API itself, not in convention. A wrong API
shape here propagates into every consumer:

- A cache keyed on file path instead of content hash silently returns
  stale data when an input is overwritten in place.
- A cache that skips the journal head returns post-merge artifacts to
  pre-merge sessions.
- A cache without atomic writes corrupts on Ctrl-C mid-write and then
  silently returns broken archives on the next open.
- A cache without portable rkyv flags works on the dev box and breaks
  on someone else's.

Get the surface right once.

## Out of scope

- Wiring any actual consumer. No `seed_amplitudes` change yet.
- Touching `sorrel-io` JSON parsers. Out of scope, permanently.
- Touching `journal.rs` / `command.rs`. The journal stays on `rmp-serde`.
- Any UI surface for cache stats — Phase 04.
- GC policy implementation beyond a stub function — Phase 04.
- Cross-architecture cache portability *guarantees*. We turn on
  rkyv's portable feature to avoid in-place endian surprises, but we
  don't promise sharing caches across `x86_64` ↔ `aarch64`.

## Plan

1. Add the `sorrel-cache` crate:
   - `crates/sorrel-cache/Cargo.toml` with workspace inheritance.
   - `crates/sorrel-cache/src/lib.rs` re-exporting `CacheKey`,
     `Fingerprint`, `CacheStore`, `CacheError`.
   - Register in root `Cargo.toml` `[workspace] members`.
   - Add `sorrel-cache = { path = "crates/sorrel-cache" }` under
     `[workspace.dependencies]`.

2. Add workspace deps in root `Cargo.toml`:
   ```toml
   rkyv = { version = "0.8", default-features = false, features = [
     "alloc", "bytecheck", "pointer_width_64",
   ] }
   ```
   Use `bytecheck` so deserialise validates before deref — required
   for a cache where files might be partially written or corrupted.
   Keep `memmap2` (already workspace-listed).

3. Update `nix/deps.nix` if it pins crate sources or if any rkyv
   transitive needs allow-listing. Most likely a no-op; verify with
   `nix flake check` (or the project's equivalent).

4. Define `CacheKey` in `crates/sorrel-cache/src/key.rs`:
   ```rust
   pub struct CacheKey {
       pub kind: &'static str,       // "amplitudes", "pc_subspace", …
       pub algo_version: u32,        // bump when the algorithm changes
       pub fingerprint: Fingerprint, // 32-byte BLAKE3 over inputs+params
   }

   pub struct Fingerprint([u8; 32]);

   impl Fingerprint {
       pub fn hex(&self) -> String { … }
       pub fn builder() -> FingerprintBuilder { … }
   }

   pub struct FingerprintBuilder { hasher: blake3::Hasher }
   impl FingerprintBuilder {
       pub fn add_provider_identity(self, bytes: &[u8]) -> Self;
       pub fn add_journal_head(self, head: u64) -> Self;
       pub fn add_params<T: ?Sized + AsBytes>(self, p: &T) -> Self;
       pub fn finish(self) -> Fingerprint;
   }
   ```
   Use `blake3` (workspace-add it; it's a single crate, fast, no
   ecosystem footgun). Don't reuse `baseline_hash` for this — that's
   an FNV-style 64-bit hash for journal sealing, fine for that
   purpose, too narrow for cache keys.

5. Define `CacheStore` in `crates/sorrel-cache/src/store.rs`:
   ```rust
   pub struct CacheStore { root: PathBuf }

   impl CacheStore {
       pub fn open(dataset_dir: &Path) -> Result<Self, CacheError>;
       pub fn get<A>(&self, key: &CacheKey) -> Result<Option<Archived<A>>, CacheError>
       where A: rkyv::Archive, Archived<A>: rkyv::bytecheck::CheckBytes<…>;
       pub fn put<A>(&self, key: &CacheKey, value: &A) -> Result<(), CacheError>
       where A: rkyv::Archive + rkyv::Serialize<…>;
       pub fn evict(&self, key: &CacheKey) -> Result<bool, CacheError>;
   }
   ```
   The `get` return type is the design pivot. Two options:
   - Return a borrowed `&Archived<A>` tied to an owned `Mmap` held by
     a guard struct (zero-copy, what we want).
   - Return an owned `A` via `rkyv::deserialize` (loses the zero-copy
     win but is simpler).

   Pick the guard-borrow form. Define a `CacheRead<A>` newtype that
   owns the `Mmap` and derefs to `&Archived<A>`. Document the lifetime
   contract.

6. On-disk layout:
   ```
   <dataset_dir>/.sorrel/cache/
   ├── INDEX               # one-line-per-entry log: kind|fingerprint|len|mtime
   ├── amplitudes/
   │   └── <hex-fingerprint>.rkyv
   ├── pc_subspace/
   │   └── <hex-fingerprint>.rkyv
   └── …
   ```
   - Atomic writes: write to `<file>.tmp.<pid>.<nonce>`, fsync, rename.
   - On read, validate with `bytecheck` before handing out the
     `Archived<A>` reference. A failed check evicts the file and
     returns `Ok(None)` (cache miss), never an error to the caller.
   - INDEX is advisory; Phase 04 owns its GC use.

7. Tests in `crates/sorrel-cache/src/tests.rs` (tmp-dir fixture):
   - put / get roundtrip for a toy archive struct.
   - corrupted file → get returns `None`, file is evicted.
   - put with the same key twice overwrites atomically.
   - fingerprints differ when any of (provider_identity, journal_head,
     params, algo_version) change.

8. Architecture doc update: add a short paragraph to
   `docs/src/architecture/workspace-layout.md` listing the new crate
   and its single-purpose role.

## Acceptance criteria

- [ ] `cargo check --workspace` clean.
- [ ] `cargo test -p sorrel-cache` passes with at least the four unit
      tests enumerated in step 7.
- [ ] `cargo tree -p sorrel-cache --edges normal` shows `rkyv`,
      `memmap2`, `blake3`, and **no** `serde_json` / `rmp-serde` /
      `serde` (except as transitive of `anyhow` if any).
- [ ] `rg "rkyv" crates/sorrel-io/ crates/sorrel-data/` returns
      nothing. Phase 01 leaves consumers untouched.
- [ ] `crates/sorrel-cache/src/lib.rs` module-doc paragraph explicitly
      states "this cache is derivable; deleting it must never lose
      user data".
- [ ] `docs/src/architecture/workspace-layout.md` mentions
      `sorrel-cache`.

## Files likely touched

- `Cargo.toml` (root) — workspace member + workspace deps + new
  `rkyv`, `blake3` lines.
- `crates/sorrel-cache/Cargo.toml` (new).
- `crates/sorrel-cache/src/lib.rs` (new).
- `crates/sorrel-cache/src/key.rs` (new).
- `crates/sorrel-cache/src/store.rs` (new).
- `crates/sorrel-cache/src/tests.rs` (new).
- `nix/deps.nix` — touch only if `nix flake check` complains.
- `docs/src/architecture/workspace-layout.md` — one-paragraph add.

## Pitfalls

- **Choosing rkyv 0.7 vs 0.8.** 0.8 is the current line; 0.7 has more
  tutorials but is being deprecated. Pick 0.8. Symptom of picking
  0.7: future upgrade work doubled.
- **Forgetting `bytecheck`.** A cache without validation deref's into
  attacker-controlled bytes if a file is corrupted on disk. Always
  validate before exposing `Archived<A>`. Symptom: crashes on the
  first power-loss replay.
- **Returning `&Archived<A>` without a guard struct.** The borrow
  can outlive the `Mmap`. Always wrap the mmap + archive reference in
  a single owned guard type with a single lifetime. Symptom: hard-to-
  read borrowck errors at consumer sites in Phase 02+.
- **Baking absolute paths into the fingerprint.** A dataset moved to a
  new directory should still hit the cache. Fingerprint inputs by
  *content* (e.g. mtime+len of `spike_clusters.npy`, or a real digest
  of the bytes — prefer the digest for the bytes you'd actually read,
  mtime+len for huge memmapped raw files), never by path string.
- **Skipping the algo_version field.** Without it, a future change to
  the algorithm silently returns stale results. Bump policy: any
  observable change to the artifact shape or compute increments
  `algo_version`. Document this in the module doc.
- **Portable feature off.** rkyv defaults are not portable across
  endianness/pointer-width. Turn on `pointer_width_64` (we're 64-bit
  only for sorrel) and document that 32-bit hosts are unsupported. If
  rkyv's native-endian default ever bites us, switch to the
  little-endian-only feature.

## Reference

- rkyv 0.8 docs: <https://rkyv.org> (verify exact crate-features
  spelling against the version landed in `Cargo.lock`).
- `crates/sorrel-data/src/journal.rs:baseline_hash` — existing
  64-bit hash; not reused here but referenced in the module doc.
- `docs/src/architecture/data-flow.md` — open / edit / save flow that
  Phase 02+ slots into.

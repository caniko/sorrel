# Sorrel Improvement Research Dossier

> Evidence gathered 2026-06-10 at commit `d15ccf3` (trunk, in sync with
> `origin/trunk`). Re-verified and extended later the same day at the
> same commit; the working tree now carries an uncommitted simit badge
> block in `README.md`, this dossier plus its `SUMMARY.md` entry, and a
> stray untracked `docs/.cargo/config.toml` (see Current Reality).

## Goal And Trigger

The user asked to "improve the sorrel project" with research-first
discipline. This dossier records the evidence-backed current state and
the improvement workstreams it supports, so future sessions can act
without re-deriving the facts.

## Current Reality

- **Workspace health is strong.** `cargo fmt --check`, `cargo clippy
  --workspace --all-targets`, and `cargo test --workspace` all pass in
  the Nix dev shell (`nix develop -c …`; bare shell has no cargo).
  449 tests pass across 8 crates (164 sorrel-compute, 88+34
  sorrel-data, 62+18 sorrel-io, 47 sorrel-ui, 11 sorrel-render,
  11 sorrel-gpu, 8 sorrel-cache, 5 sorrel bin) with 2 ignored.
  Doc-tests are nearly absent: 6 in total (4 sorrel-compute,
  1 sorrel-data, 1 sorrel-ui; the other 5 crates have zero).
- **The rkyv derived-cache plan set has fully landed** (see Existing
  Plan Status below) but its planning docs are still published in the
  mdBook under `Planning` in `docs/src/SUMMARY.md`.
- **First publish to crates.io has not happened.** The crates.io API
  returns "crate does not exist" for **all 8 workspace crate names**
  (`sorrel`, `-io`, `-cache`, `-compute`, `-gpu`, `-data`, `-render`,
  `-ui`; verified 2026-06-10 — note the API requires a real
  `User-Agent` header or it returns a rate-limit error). `git tag` is
  empty; `CHANGELOG.md` already declares `0.1.0` (2026-05-25) and
  `RELEASE.md` documents the 8-crate publish order.
- **Generated CI has drifted — all 16 simit-managed workflows.**
  `simit init ci --workspace --platform forgejo --check` (simit
  0.16.1) now reports all 8 `ci-sorrel-*.yaml` and all 8
  `publish-crate-*.yaml` files under `.forgejo/workflows/` as
  differing, and prints the canonical regeneration command itself:
  `simit init ci --platform forgejo --runner atlas --workspace`.
  (An earlier check the same day reported 12 files; 16 is the current
  number.) `pages.yaml` is not simit-managed and is not in the drift
  set. The simit-generated README badge block (uncommitted) correctly
  reads `CI: drift`. There is no `simit.toml` anywhere in the repo, so
  generation options are inferred on every check.
- **CI already runs on the self-hosted atlas runner.** 16 of 17
  workflows declare `runs-on: atlas` (container `rust:bookworm`); only
  `pages.yaml` still runs on `codeberg-small` with
  `install-nix-action`.
- **CI on trunk head is green after retries.** Codeberg API shows many
  `failure` runs for `d15ccf3` on 2026-05-25 followed by `success` on
  2026-05-27 — consistent with the "Retry CI/runners" commits
  (`92d528e`, `5661392`) pointing at runner flakiness.
- **Publish workflows are tag-fan-out with no cross-crate ordering.**
  Every `publish-crate-*.yaml` triggers on the same `*.*.*` tag push,
  so one `0.1.0` tag fires all 8 concurrently. Each workflow: verifies
  the tag is GPG-signed by a key in `keys/maintainers.gpg`, checks tag
  == `cargo pkgid` version, runs tests + clippy (all-features and
  no-default-features), `cargo publish --dry-run`, then publishes with
  the `CRATES_IO_API_TOKEN` secret. Publishing is **idempotent**: a
  pre-publish crates.io check skips crates whose version is already
  live, so re-running failed downstream workflows in dependency order
  is safe.
- **Release trust roots are in place locally.** `keys/maintainers.gpg`
  pins exactly one key (`0x4623DEA06FDACFE1`, Can H. Tartanoglu), which
  matches `git config user.signingkey`, and `tag.gpgSign` is `true` —
  a locally created release tag will pass the workflow's
  `git verify-tag` gate.
- **No license files exist anywhere.** All manifests declare
  `license = "MIT OR Apache-2.0"` via `[workspace.package]`, but there
  is no `LICENSE*`/`COPYING*` file at the repo root or in any crate
  directory. No crate has a `README.md` or `readme` field either, so
  published crates.io pages would render with no readme.
- **`deny.toml` exists but nothing runs cargo-deny.** Neither the
  Forgejo workflows nor the flake checks (`nix/checks.nix`,
  `nix/pre-commit.nix`) reference it.
- **README.md is stale.** Its workspace table lists 6 crates, omitting
  `sorrel-cache` and `sorrel-gpu`; its keyboard table still says
  "Undo (V1: stub)" while `crates/sorrel-ui/src/app.rs:293-298`
  implements full undo/redo with per-command status messages, and the
  in-app hint also advertises redo and save.
- **A stray `docs/.cargo/config.toml` appeared (untracked).** It is
  byte-identical to the root `.cargo/config.toml` (rs-harbor-generated,
  nightly-only flags like `-Zthreads`/cranelift) — almost certainly
  dropped by a tool run from inside `docs/`. It serves no purpose
  there and should be deleted, not committed.
- **Two IO backends are deliberate stubs**:
  `crates/sorrel-io/src/nwb.rs:27` and
  `crates/sorrel-io/src/ks4_rez.rs:25` only expose `open_stub`.
- `crates/sorrel-py` (PyO3 bindings) is intentionally excluded from the
  workspace and is the only crate with a `description` field.

## Evidence Inventory

| Evidence | Command / file | What it proves |
|---|---|---|
| Quality gates pass | `nix develop -c cargo fmt --check / clippy --workspace --all-targets / test --workspace` (exit 0, 449 passed) | No lint/test debt blocking other work |
| Doc-test gap | `nix develop -c cargo test --workspace --doc` → 6 doc-tests total, 5 crates with 0 | rustdoc/examples work needed before docs.rs exposure |
| All 8 names unreserved | `curl -A '<real UA>' https://crates.io/api/v1/crates/<name>` → "does not exist" for all 8 | First publish pending; squatting window open on every name |
| Publish metadata gap | `grep -L "^description" crates/*/Cargo.toml` → all 8 workspace crates; `[workspace.package]` has only version/edition/license/repository | `cargo publish` will hard-fail on every crate |
| No license files | `ls LICENSE*` and `ls crates/*/LICENSE*` → nothing | Declared `MIT OR Apache-2.0` has no license texts; legal gap before publish |
| No crate readmes | `ls crates/sorrel-io/`, no `readme` field in manifests | crates.io pages would be empty |
| No release tags | `git tag` → empty | Tag-triggered publish workflows never ran |
| CI drift (16 files) | `nix develop -c simit init ci --workspace --platform forgejo --check` → all 16 `ci-*`/`publish-crate-*` differ; suggests `--runner atlas` regen | Regeneration needed; README badge accurate |
| simit config gap | `rg --files -g 'simit.toml'` → none | CI options inferred, not pinned |
| Workflows on atlas | `grep runs-on .forgejo/workflows/*.yaml` → 16× `atlas`, 1× `codeberg-small` (pages.yaml) | Only Pages still on shared runners |
| Publish workflow anatomy | `.forgejo/workflows/publish-crate-sorrel-io.yaml` (143 lines) | Signed-tag gate, tag==version check, idempotent skip, `CRATES_IO_API_TOKEN` required, no cross-crate ordering |
| Signing key matches trust root | `gpg --show-keys keys/maintainers.gpg` → `0x4623DEA06FDACFE1` == `git config user.signingkey`; `tag.gpgSign=true` | Locally signed tags will pass `git verify-tag` in CI |
| cargo-deny unenforced | `deny.toml` at root; `rg deny .forgejo/workflows/ nix/` → only clippy `--deny warnings` | Supply-chain gate exists on disk but never runs |
| Trunk CI green after retries | Codeberg API `actions/tasks`: failures 2026-05-25, success 2026-05-27 for `d15ccf3` | Runner flakiness, not code failure |
| Cache plan landed | `crates/sorrel-cache/src/{key,store,gc,tests}.rs`, `cache hit` logs in `session.rs:120`, `quality_ext.rs:67`, `feature_subspace.rs:83`, `cache/correlograms.rs:210`, bench `seed_amplitudes_cache.rs`, `docs/src/architecture/cache.md` in `SUMMARY.md` | All four phases shipped (commits `38a8e58`, `38832c3`, `944b50f`, `d15ccf3`) |
| rkyv containment | `rg -l rkyv crates/sorrel-compute/src crates/sorrel-io/src/{probeinterface,sorting_analyzer,open_ephys}.rs` → no matches | Whole-set constraint honoured |
| Invalidation tested | `crates/sorrel-cache/src/tests.rs` (`add_journal_head`), `session.rs:194,231` (`invalidate_caches`) | Acceptance criterion met |
| README staleness | `README.md:39` "Undo (V1: stub)" vs `crates/sorrel-ui/src/app.rs:293-298`; workspace table vs `Cargo.toml` members | Docs drift from implementation |
| Stray cargo config | `diff .cargo/config.toml docs/.cargo/config.toml` → identical; untracked | Accidental artifact; delete |
| Stub backends | `crates/sorrel-io/src/nwb.rs:27`, `ks4_rez.rs:25` | NWB / KS4 `.rez` ingest unimplemented |

## Existing Plan Status

`docs/src/planning/rkyv-derived-cache/` (4 phases, phase 03 with 3
sub-layers), audited against the working tree:

| Phase | Claim | Status | Proof |
|---|---|---|---|
| 01 Cache infrastructure | `sorrel-cache` crate with key/store/GC | done | `crates/sorrel-cache/src/`, workspace member in `Cargo.toml` |
| 02 First consumer | `seed_amplitudes` cached + bench | done | `session.rs:120` cache-hit log; `benches/seed_amplitudes_cache.rs` |
| 03/01 PC subspaces | cached | done | `cache/pc_subspace.rs`, `feature_subspace.rs:83` |
| 03/02 Isolation metrics | cached | done | `cache/isolation.rs`, `quality_ext.rs:67` |
| 03/03 Correlograms | cached | done | `cache/correlograms.rs:210` |
| 04 Invalidation, GC, docs | replay hook, GC, `cache.md` | done | `session.rs:194,231`, `sorrel-cache/src/gc.rs`, `architecture/cache.md` linked in `SUMMARY.md` |
| Whole-set | tests pass; rkyv stays out of `sorrel-compute` and JSON ingest; invalidation unit-tested | done | test run + rg evidence above |

The plan set is complete and ready for formal `plan-and-verify` verify
mode, which on success migrates durable knowledge into stable docs and
retires the planning files.

## Work That Should Survive

- `RELEASE.md` first-publish order (`sorrel-io` → `sorrel-cache` →
  `sorrel-compute` → `sorrel-gpu` → `sorrel-data` → `sorrel-render` →
  `sorrel-ui` → `sorrel`) and the name-reservation check
  (`cargo search <crate> --limit 1`).
- The derived-cache architecture doc (`docs/src/architecture/cache.md`)
  — already in stable docs; planning prose can retire.
- Global constraint from the plan set that remains binding for future
  work: rkyv stays out of `sorrel-compute` and the JSON ingest /
  `rmp-serde` journal codecs.
- The publish-workflow trust model (signed tag verified against
  `keys/maintainers.gpg`, tag==version gate, idempotent re-runs) — any
  CI regeneration must preserve it; confirm after running simit.

## Blockers And Missing Artifacts

- **Crate descriptions (publish blocker).** All 8 workspace
  `Cargo.toml` files lack `description`; `cargo publish` refuses such
  crates. Producer: this repo. Fix: add `description` (and ideally
  `keywords`, `categories`) per crate or via `[workspace.package]`
  inheritance. Validation: `nix develop -c cargo publish --dry-run -p
  sorrel-io` reaches packaging instead of metadata error.
- **License texts (publish-quality blocker).** `license = "MIT OR
  Apache-2.0"` is declared but no `LICENSE-MIT` / `LICENSE-APACHE`
  files exist anywhere, and no crate ships a readme. Not a hard
  `cargo publish` error, but a legal/compliance gap for a public
  release. Producer: this repo (the `rust-crate-legal-readme` skill
  owns this domain). Fix: add dual license texts at the root, point
  each crate at them (`license-file` is not needed when texts are
  included via `include`/symlink convention), and add per-crate
  `readme`. Validation: `cargo package -p sorrel-io --list` shows the
  license files in the bundle.
- **CI drift (release-trust blocker).** All 16 generated workflows,
  including every tag-triggered `publish-crate-*.yaml`, differ from
  simit 0.16.1 output; publishing through drifted generated workflows
  invites surprises. Producer: simit 0.16.1. Fix: `nix develop -c
  simit init ci --platform forgejo --runner atlas --workspace` (the
  exact command simit's check output suggests), review diff, commit —
  the `simit-dependent-fixes` skill owns this domain. Validation:
  the `--check` form reports no differences and the regenerated badge
  block stops saying `drift`.
- **`CRATES_IO_API_TOKEN` secret unverified.** The publish workflows
  hard-fail without it, and repo secrets cannot be read from this
  environment. Producer: the user, in Codeberg repo settings
  (Settings → Actions → Secrets). Validation: a `workflow_dispatch`
  run of one publish workflow reaches the dry-run step, or the first
  real tag publish succeeds for `sorrel-io`.
- No other foundational inputs are missing; nothing here blocks the
  research itself.

## Risks And Constraints

- **Publish sequencing vs tag fan-out:** one `0.1.0` tag fires all 8
  publish workflows concurrently, but downstream crates fail dry-run
  until upstream crates are live (documented in `RELEASE.md`). The
  8-crate order is a hard constraint that CI does not encode. The
  idempotent already-published skip makes the safe procedure: let the
  fan-out run, then re-run failed workflows in dependency order (or
  pre-publish `sorrel-io` → … manually before tagging).
- **Signed-tag gate:** the release tag must be an exact bare semver
  (`0.1.0`, no `v` prefix), must equal the Cargo version, and must be
  GPG-signed by the single key pinned in `keys/maintainers.gpg`.
  Local git config currently satisfies this; a tag pushed from any
  other machine/key will fail every publish workflow.
- **Crate-name squatting:** all 8 names verified unreserved on
  2026-06-10; they stay exposed while publish is delayed. `RELEASE.md`
  mandates stopping on a taken name rather than auto-renaming.
- **Runner flakiness now scoped to Pages:** 16 of 17 workflows already
  run on self-hosted `atlas`; only `pages.yaml` runs
  `install-nix-action` on `codeberg-small`. Moving it to
  `atlas-nix-trusted` is possible but subject to Codeberg ToU
  constraints (see `atlas-runner` skill). The May retry storm predates
  the current runner assignment evidence; treat historical flakiness
  attribution as uncertain.
- **Generated-file discipline:** manual edits to `.forgejo/workflows/`
  will re-introduce drift; there is currently no `simit.toml` at all,
  so every check infers options — pinning `[ci]` options (platform,
  runner) in `simit.toml` would make `--check` deterministic.
- **Supply-chain gate unenforced:** `deny.toml` exists but cargo-deny
  runs nowhere (not CI, not flake checks, not pre-commit). Advisory
  drift accumulates silently until wired in.
- **Docs publish planning prose:** the mdBook publishes the completed
  plan set (and now this dossier via the uncommitted `SUMMARY.md`
  entry); retiring/curating it is a navigation/content change to the
  public site, not just a repo cleanup.

## Candidate Next Steps

1. **Unblock first publish (metadata + legal)** — add
   `description`/`keywords`/`categories` to all 8 crates, add
   `LICENSE-MIT`/`LICENSE-APACHE` texts and per-crate `readme`
   wiring; commit. No dependencies; parallel-safe with (2).
2. **Clear CI drift** — regenerate with `simit init ci --platform
   forgejo --runner atlas --workspace`, add a `simit.toml` pinning the
   `[ci]` options, re-run the badge update, and commit together with
   the pending README badge block. Should land **before** tagging
   `0.1.0` so publish workflows run from generated state. Verify the
   signed-tag/idempotency logic survives regeneration. Parallel-safe
   with (1).
3. **Execute first publish** — confirm `CRATES_IO_API_TOKEN` is set in
   Codeberg secrets; per `RELEASE.md` order: name checks,
   `cargo package -p <crate>` for each, `simit changelog release`,
   create the signed `0.1.0` tag, push it, then re-run failed
   downstream publish workflows in dependency order (idempotent skip
   makes this safe). Depends on (1) and (2).
4. **Verify + retire the rkyv-derived-cache plan set** — run
   `plan-and-verify` verify mode; on success retire
   `docs/src/planning/rkyv-derived-cache/` from `SUMMARY.md`
   (`retire-docs-planning` flow). Independent of (1)–(3).
5. **Refresh README.md** — fix the workspace table (add `sorrel-cache`,
   `sorrel-gpu`), correct the undo row, add redo/save keys. Independent;
   trivial. Also delete the stray untracked `docs/.cargo/config.toml`.
6. **Wire cargo-deny into CI or flake checks** — `deny.toml` already
   exists; add a job/check so it actually gates. Small; independent.
7. **Feature work: real NWB and KS4 `.rez` backends** — currently
   stubs; the `DataProvider` trait boundary is the integration point.
   Larger design work; would warrant its own plan set.
8. **Optional:** move `pages.yaml` to `atlas-nix-trusted`; grow
   rustdoc examples/doc-tests (6 today, 5 crates at zero) before
   docs.rs exposure.

## Open Decisions For The User

- **Publish now or later?** Metadata and drift fixes are mechanical,
  but pushing the `0.1.0` tag publishes 8 crate names to crates.io —
  an outward-facing, irreversible step that should be an explicit call.
  The `CRATES_IO_API_TOKEN` secret must also be provisioned by the
  user before any tag push.
- **Where should the Pages job run** — stay on `codeberg-small` (flaky
  but zero-config) or move to `atlas-nix-trusted` (faster, ToU
  considerations)? CI proper is already on atlas.
- **NWB / KS4 `.rez` priority** — is backend expansion the next major
  feature, or does curation UX take precedence? This determines whether
  step (7) gets a plan set now.

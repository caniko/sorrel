# Curation bench: portable state

The double-blind improvement loop measures sorrel's automated curation against
manual post-sort curation in phy, and steers sorrel towards that quality. The
gap is mostly missing capability, so most rounds ship sorrel features (new
proposers, probes, brief features) as well as prompt variants.

The `bench/` directory holds everything from the loop that is safe to publish:

- `results.sqlite`: every round's derived results (git-lfs). It holds opaque
  session ids, metrics, misses, the root cards the arms saw, dev truth verdicts,
  arm actions, and holdout aggregates only.
- `variants/<name>/`: prompt variants (`review.md`, `rubric.md`) and their
  `RATIONALE.md`. A variant with no override files runs the built-in prompt of
  the binary named in `runs.sorrel_rev`.

The procedure is `.claude/skills/curation-bench-loop/SKILL.md`. Its Pitfalls
section lists the checks to run before collecting data. Metrics and blinding are
in the "Double-blind bench" section of `docs/src/usage/auto-curation.md`.

Never in the repo: recordings, `BENCH/` (blinded inputs, truths and run copies),
`BENCH/.sealed/` (the salt, the id↔path map and the holdout), and anything that
names an animal, a date or a lab path.

## Native CLI

Build with `--features dev` to use `sorrel dev bench` without a Python runner.
Discovery checks names and existence only; intake reads originals without
opening a session host on them, requires a self-contained raw recording, blinds
the input and seals the truth. Duplicate sorter fingerprints cannot straddle
dev and holdout. Add any previously inspected session with `--dev` first.

```sh
sorrel dev bench discover /data/sorts --integer-ids
sorrel dev --db bench/results.sqlite bench --bench /private/BENCH add --phy-curated --dev /data/seen-sort
sorrel dev --db bench/results.sqlite bench --bench /private/BENCH add --phy-curated --discover /data/sorts --integer-ids
sorrel dev --db bench/results.sqlite bench --bench /private/BENCH run --arm mcp-review --variant bench/variants/v2-core-base --split dev --repeats 1 --dev-limit 1 --plumbing-check --backend opencode --model PROVIDER/MODEL --sorrel-rev COMMIT --max-budget-usd 2
sorrel dev --db bench/results.sqlite bench --bench /private/BENCH score --split dev
sorrel dev bench --bench /private/BENCH report dev
```

Remove `--dev-limit` and `--plumbing-check` and use three repeats for an actual comparison; run the
`auto-llm` arm too. `--sorrel` and `--scorer` select immutable binaries; the
scorer must stay fixed across variants. Repeating the same command resumes it:
successful runs with identical configuration and repeat number are skipped.
`--jobs 2` runs two independent sessions concurrently (default 1, maximum 8);
repeats of a session stay sequential. Use the same job count for comparative
rounds and check memory before increasing it. Any failure stops new work;
already active runs finish and are accounted for before the command returns
an error. Per-run timeouts apply to every arm. `--max-budget-usd` caps the
`mcp-review` arm (Claude receives the flag; OpenCode is stopped once its
reported cost reaches it) and defaults to 10 USD. The `auto-llm` arm runs
`sorrel auto`, whose LLM calls take no cap, so a capped run of that arm is
refused; omit the flag to run it uncapped.

`sorrel dev bench --bench /private/BENCH coverage --dev-limit 6` is a lead-only,
dev-only boundary diagnostic. It fits the experimental robust majority-cleanup
boundary on input alone, then measures oracle overlap with existing dev truth.
It opens only disposable allowlisted input copies, never original sorts or
holdout, and writes a separate coverage report (not a scored arm run). Oracle
reachability is a capability ceiling, not automated curation performance or
scientific acceptance. No threshold or boundary is optimized using truth.
`--method density` instead measures the bounded hierarchical-tail proposer
(at most four levels /15 fitted nodes per root), with the same row-alignment
and dev-only safeguards. Neither diagnostic is an LLM curation-arm score.
`--method feature-oracle` is different: it deliberately fits human training
labels to transparent PC/amplitude/time cuts and evaluates the fixed cuts on
alternate spikes. Feature/threshold selection uses training data only, with
bounded class-weighted sampling. These correlated same-session validation
spikes do NOT establish leave-session-out accuracy. This mode diagnoses the
input representation; its fitted thresholds must never enter an arm directly.
Each run records the binary hash and
commit, and saves revision-pinned input cards. Failed, capped, disconnected or
input-mutating runs stop collection and are not scored. Sidecar billing that
is not exposed is reported as unknown, never as free.

The usual Jaccard-0.8 recovery metrics measure progress, not identity. `sorrel
diff` also reports `exact_match`: all spike memberships and original labels,
including noise, must agree up to a bijective cluster-id rename, with no
disputed roots excluded. Only that check certifies an exact reproduction.

Holdout commands are lead-only. They keep runs and scores under `.sealed/`;
`report holdout --baseline NAME --candidate NAME` logs each attempted query and
exports paired aggregates only. Keep the adjudication requirement from the
skill: a score without human holdout adjudication is not a final acceptance.
The legacy Python runner below remains available for older benches; new
reusable utilities belong in the sorrel CLI.

## Bootstrap on a new machine

1. **Clone with LFS.** Run `git lfs install` once per user, then clone. Check
   that `git lfs ls-files` lists `bench/results.sqlite` and that the file starts
   with `SQLite format 3`, not with a `version https://git-lfs` pointer.
   - A repo-local `core.hooksPath` (the dev shell's git-hooks.nix sets
     `.git/hooks`) resolves to a file in a linked worktree, so no LFS hook runs
     there. Push from such a worktree with `git lfs push origin <branch>` before
     `git push`.
2. **Pin the binaries for this round** from a clean shared clone, never from a
   working tree someone is editing:
   ```sh
   git clone --shared <repo> /tmp/sorrel-pin && cd /tmp/sorrel-pin && git checkout <commit>
   nix develop . --no-update-lock-file --command \
     cargo build --release -p sorrel --features dev
   cp target/release/sorrel $PINS/sorrel-<commit>
   ```
   Use one binary for the arms (`SORREL_BIN`) and one scorer for every variant
   you compare. Set `SORREL_REV=<commit>` so runs record it.
3. **Pick private directories outside the repo**, on a volume with room for
   repeats × arms × input size: `BENCH=<private>/BENCH` and `uploads=<private>/uploads`.
4. **Unpack the uploads** from the `export-curated-sort` skill (each is
   `NAME.sorrel-upload.tar` plus a `.sha256`):
   ```sh
   uv run .claude/skills/export-curated-sort/export_curated_sort.py unpack NAME.sorrel-upload.tar $uploads/NAME
   ```
5. **Re-seal** (see below), then add:
   ```sh
   export SORREL_DEV_DB=$PWD/bench/results.sqlite
   scripts/bench_curation.py --bench $BENCH add --phy-curated --dev $uploads/<seen sessions>
   scripts/bench_curation.py --bench $BENCH add --phy-curated $uploads/<new sessions>
   ```
6. **First run per machine: check the plumbing** (Pitfall 11). One
   `run --arm mcp-review --repeats 1 --max-budget-usd 2`, then check
   `claude.json` for sorrel tool calls.

**Environment.**

| Variable                         | Read by              | Meaning                                                                             |
| -------------------------------- | -------------------- | ----------------------------------------------------------------------------------- |
| `SORREL_BIN`                     | runner               | the pinned sorrel binary (default `sorrel` on PATH)                                 |
| `SORREL_REV`                     | runner               | commit to record when the binary isn't inside its repo                              |
| `SORREL_DEV_DB`                  | runner, `sorrel dev` | results DB; also `--db`                                                             |
| `SORREL_DEV_DENY`                | `sorrel dev`         | file of strings the DB must refuse; the runner writes `BENCH/.sealed/dev-deny.txt`  |
| `SORREL_RUBRIC`, `SORREL_REVIEW` | sorrel               | prompt overrides (a file path or the text); the runner sets them from a variant dir |
| `BENCH_SALT`                     | runner               | tests only: a reproducible split                                                    |

## How the sealed holdout survives a move

**Decision: re-seal on the new machine. The sealed map and salt never travel,
and never touch the repo.**

On the new machine, `add` creates a fresh salt and assigns about 30% of the
purely manual sessions to the holdout by salted hash. Every session the lead has
already seen, meaning every former dev session plus anything inspected by hand,
is added with `--dev`.

**Why re-sealing rather than a private transfer:**

- **It cannot leak.** A new salt can move a former holdout session into dev,
  which spends some holdout but reveals nothing that was sealed. It can never
  move a seen session into the holdout, because `--dev` forces those. A
  transferred `.sealed/` carries real source paths and the salt, and a copy that
  goes astray (a backup, a sync folder, a mistaken commit) is a leak.
- **The history that matters is already public.** The holdout query count and
  its aggregate results are in `holdout_ledger` in the DB, so nothing about past
  acceptance decisions is lost. A new machine's holdout starts its own ledger
  rows.
- **It was free at the first move.** The only session so far was dev (seen by
  the lead), so there was no holdout to lose.

**Transfer privately instead only if** a holdout large enough to be worth
keeping exists and the target machine is trusted. In that case, copy
`BENCH/.sealed/` and `BENCH/sessions/` over an encrypted channel (for example
`rsync` over SSH), never through the repo, a shared drive or chat, and delete the
source copy.

## Lessons that apply on any machine

**Build and environment.**

- Run cargo, tests and `treefmt` inside `nix develop . --no-update-lock-file`.
  A plain shell may have `cargo` without `rustc`, and `python3`/`treefmt` exist
  only inside the dev shell. A plain-PATH `python3 - <<EOF` can silently do
  nothing.
- The `auto-llm` arm runs a SpikeInterface sidecar through uv. Without a system
  uv, use `nix shell nixpkgs#uv nixpkgs#python313 nixpkgs#stdenv.cc.cc.lib` with
  `LD_LIBRARY_PATH` set to that libstdc++ and `UV_PYTHON_DOWNLOADS=never`.
- Scratch volumes fill up. Claude Code's temp directory filling up loses
  command output, so check `df` before a round.

**LLM backends and cost.**

- Run `claude -p` with `--strict-mcp-config`, an explicit `--mcp-config`,
  `--setting-sources project`, `--no-session-persistence`, built-in tools off
  and an empty cwd outside any repo. Otherwise each call carries the user's
  whole Claude Code context: about $0.42 against $0.019 per `auto --llm` case.
  The runner does all of this.
- One `mcp-review` run on an 18-cluster tetrode session cost about $0.25–0.32
  (Opus 5.5, 9–14 turns, 40–90 s). Keep `--max-budget-usd` on every run.
- OpenCode needs `--llm-model provider/model`, because it has no default model.

**Data.**

- CINPLA tetrode sorts (Kilosort 2, phy-curated) have fake channel positions
  (indices) and labels of noise/unsorted only. They are useful for merges and
  splits, not for footprint features or `good` recall.
- SpikeInterface's `export_to_phy` stores unit indices in `spike_clusters.npy`
  and the original ids in `si_unit_id`; read the sorting from the npy files.
- Check polarity once per lab: sorrel computes waveforms from `recording.dat`,
  and Kilosort 4 can run with `invert_sign`. Compare a mean raw snippet with
  the template before trusting waveform features.

**Journals and copies.**

- A journal has one writer (`<journal>.lock`). A second opener gets
  `JournalBusy`, so never point two runs at one directory. The runner copies
  the input per run, and `sorrel dev record-session` opens a scratch copy.
- `sorrel call DIR …` without a running host writes a journal into `DIR`, so
  only call it on a copy, never on `BENCH/sessions/*/input`.

**CI.**

- Hosted runners can queue for hours. Pending is not passing.
- simit's qualification workflows refuse reruns; dispatch a fresh
  `workflow_dispatch` run instead.
- Push CI runs on trunk only, because cancelled push runs on branches blocked
  merges.

**Agents.**

- Give scouts explicit deep paths: a scout once missed phy folders nested inside
  `main.exdir/…/sorter_output`. Long code tasks can hit a subagent turn limit,
  so resume the same agent rather than starting a new one.
- Keep delegated agents to their task row in the skill. They never get
  `.sealed/`, truth directories or `phy.log`.

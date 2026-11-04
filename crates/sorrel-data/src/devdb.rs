//! Dev-mode results DB (feature `dev`): derived data of the double-blind
//! curation bench in one SQLite file that is versioned with git-lfs in the
//! public repo, so it holds opaque ids and numbers only.
//!
//! * Every `record_*` is an idempotent upsert keyed by the table's primary key.
//! * **Privacy guard.** Every text value is refused when it holds `/` or `\`
//!   (so also every absolute path) or any string of the caller's deny set
//!   (sealed sids, source basenames: [`DevDb::with_deny`]). The one exception
//!   is `variants.rubric` / `variants.review`: prompt text legitimately holds
//!   `/` ("good/mua"), so it skips the separator rule but not the deny set.
//!   Free text from a model (`arm_actions.reasons`) has its separators
//!   replaced by spaces instead of being refused.
//! * **Holdout.** A holdout session is a `sessions` row with only sid and
//!   split: cards, truth and runs are refused for it. Holdout results enter
//!   only as aggregate [`LedgerRow`]s, which carry no sid.
//! * Reads: [`DevDb::exemplars`] (dev root cards joined to their truth
//!   verdict, leave-one-session-out by fingerprint), [`DevDb::query`]
//!   (read-only SQL as TSV) and [`DevDb::distill`] (plain-text artifacts).

use crate::command::{CurationCommand, PhyLabelOp};
use crate::diff::{RootTruth, SpikeDiff};
use crate::journal::Record;
use anyhow::{bail, ensure, Context, Result};
use rusqlite::{params, types::ValueRef, Connection, OpenFlags, OptionalExtension};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const SCHEMA_V1: &str = "
CREATE TABLE sessions(
    sid TEXT PRIMARY KEY,
    split TEXT NOT NULL CHECK (split IN ('dev', 'holdout')),
    fingerprint TEXT, sorter TEXT, n_spikes INTEGER, n_roots INTEGER,
    n_channels INTEGER, binary INTEGER, added TEXT);
CREATE TABLE variants(
    name TEXT PRIMARY KEY, prompt_rev TEXT, rubric TEXT, review TEXT,
    sorrel_rev TEXT, learned_rev TEXT);
CREATE TABLE runs(
    rid TEXT PRIMARY KEY, sid TEXT NOT NULL, variant TEXT NOT NULL,
    arm TEXT NOT NULL, model TEXT, repeat INTEGER, status TEXT NOT NULL,
    cost_usd REAL, turns INTEGER, wall_s REAL, usage TEXT,
    prompt_rev TEXT, sorrel_rev TEXT, learned_rev TEXT);
CREATE TABLE metrics(
    rid TEXT NOT NULL, name TEXT NOT NULL, value REAL NOT NULL,
    PRIMARY KEY (rid, name)) WITHOUT ROWID;
CREATE TABLE misses(
    rid TEXT NOT NULL, category TEXT NOT NULL, roots TEXT NOT NULL,
    truth TEXT, test TEXT,
    PRIMARY KEY (rid, category, roots)) WITHOUT ROWID;
CREATE TABLE root_cards(
    sid TEXT NOT NULL, root INTEGER NOT NULL, field TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (sid, root, field)) WITHOUT ROWID;
CREATE TABLE truth_roots(
    sid TEXT NOT NULL, root INTEGER NOT NULL, n_spikes INTEGER NOT NULL,
    verdict TEXT NOT NULL, n_good INTEGER NOT NULL, good_share REAL NOT NULL,
    core_shares TEXT NOT NULL, core_sizes TEXT NOT NULL,
    PRIMARY KEY (sid, root)) WITHOUT ROWID;
CREATE TABLE arm_actions(
    rid TEXT NOT NULL, seq INTEGER NOT NULL, target TEXT NOT NULL,
    action TEXT NOT NULL, confidence REAL, reasons TEXT,
    PRIMARY KEY (rid, seq)) WITHOUT ROWID;
CREATE TABLE holdout_ledger(
    baseline TEXT NOT NULL, candidate TEXT NOT NULL, arm TEXT NOT NULL,
    metric TEXT NOT NULL, n_sessions INTEGER, base REAL, cand REAL,
    delta REAL, ci_lo REAL, ci_hi REAL,
    PRIMARY KEY (baseline, candidate, arm, metric)) WITHOUT ROWID;
";

/// v2: a repeated comparison appends instead of replacing; `query` names the
/// holdout query (the runner's ledger timestamp), old rows get `v1`.
const SCHEMA_V2: &str = "
ALTER TABLE holdout_ledger RENAME TO holdout_ledger_v1;
CREATE TABLE holdout_ledger(
    query TEXT NOT NULL, baseline TEXT NOT NULL, candidate TEXT NOT NULL,
    arm TEXT NOT NULL, metric TEXT NOT NULL, n_sessions INTEGER, base REAL,
    cand REAL, delta REAL, ci_lo REAL, ci_hi REAL,
    PRIMARY KEY (query, baseline, candidate, arm, metric)) WITHOUT ROWID;
INSERT INTO holdout_ledger SELECT 'v1', * FROM holdout_ledger_v1;
DROP TABLE holdout_ledger_v1;
";

/// v3: score provenance. `sessions.truth_hash` and `runs.state_hash` hash the
/// inputs a score depends on, `runs.basis` those the stored score was made from
/// (see [`drop_stale_scores`]). Old rows get NULL, which counts as changed.
const SCHEMA_V3: &str = "
ALTER TABLE sessions ADD COLUMN truth_hash TEXT;
ALTER TABLE runs ADD COLUMN state_hash TEXT;
ALTER TABLE runs ADD COLUMN basis TEXT;
";

/// v4: a score names the run's curated output and the options it was made with.
/// `runs.output_hash` is the [`truth_hash`] of the run's own curation (NULL is
/// unknown: no score until `record-run` supplies it), `runs.score_opts` the
/// comparison settings of the stored score.
const SCHEMA_V4: &str = "
ALTER TABLE runs ADD COLUMN output_hash TEXT;
ALTER TABLE runs ADD COLUMN score_opts TEXT;
";

/// The verdicts [`crate::diff::root_truths`] assigns, the only ones `truth_roots` holds.
const VERDICTS: [&str; 5] = ["noise", "kept", "cores", "cleanup", "other"];

/// `MIGRATIONS[i]` takes the schema from version `i` to `i + 1`.
const MIGRATIONS: &[&str] = &[SCHEMA_V1, SCHEMA_V2, SCHEMA_V3, SCHEMA_V4];

/// Brief card columns that are not numeric metrics, in exemplar-file order.
const CARD_HEAD: [&str; 3] = ["lab", "flags", "tags"];
const CARD_TAIL: [&str; 3] = ["acg", "amp_t", "ev"];
/// Metrics shown in `lineage.md`.
const LINEAGE_METRICS: [&str; 5] = [
    "good_units.recall",
    "good_units.precision",
    "units.recall",
    "labels.kappa",
    "ari",
];

pub struct DevDb {
    conn: Connection,
    path: PathBuf,
    deny: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct SessionRow {
    pub sid: String,
    /// `dev` or `holdout`; a holdout row keeps only these two.
    pub split: String,
    /// `baseline_hash` of the blind input's `spike_clusters.npy` (required for dev).
    pub fingerprint: Option<String>,
    pub sorter: Option<String>,
    pub n_spikes: Option<i64>,
    pub n_roots: Option<i64>,
    pub n_channels: Option<i64>,
    pub binary: Option<bool>,
    /// [`truth_hash`] of the truth the scores are made against.
    pub truth_hash: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct VariantRow {
    pub name: String,
    pub prompt_rev: Option<String>,
    /// Prompt texts: exempt from the separator rule (see the module doc).
    pub rubric: Option<String>,
    pub review: Option<String>,
    pub sorrel_rev: Option<String>,
    pub learned_rev: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct RunRow {
    pub rid: String,
    pub sid: String,
    pub variant: String,
    pub arm: String,
    pub model: Option<String>,
    pub repeat: i64,
    pub status: String,
    pub cost_usd: Option<f64>,
    pub turns: Option<i64>,
    pub wall_s: Option<f64>,
    /// Token usage as JSON text.
    pub usage: Option<String>,
    pub prompt_rev: Option<String>,
    pub sorrel_rev: Option<String>,
    pub learned_rev: Option<String>,
    /// [`truth_hash`] of the run's curated output (what `diff` scores as TEST);
    /// `None` is unknown.
    pub output_hash: Option<String>,
}

/// One edit of the arm's applied path, in order.
#[derive(Clone, Debug, PartialEq)]
pub struct ArmAction {
    pub target: String,
    pub action: String,
    pub confidence: Option<f64>,
    pub reasons: Option<String>,
}

/// One aggregate of a holdout comparison; no sid, no run id.
#[derive(Clone, Debug, Deserialize)]
pub struct LedgerRow {
    /// Which holdout query this comparison came from (the runner's ledger
    /// timestamp): the same id re-records in place, a new one appends.
    pub query: String,
    pub baseline: String,
    pub candidate: String,
    pub arm: String,
    pub metric: String,
    pub n_sessions: i64,
    pub base: f64,
    pub cand: f64,
    pub delta: f64,
    pub ci_lo: Option<f64>,
    pub ci_hi: Option<f64>,
}

/// A dev root cluster: its blind-input card and what the human did with it.
#[derive(Clone, Debug)]
pub struct Exemplar {
    pub sid: String,
    pub fingerprint: String,
    pub root: u32,
    /// Card columns by name (`lab`, `flags`, `tags`, metrics, `acg`, `amp_t`, `ev`); `-` is missing.
    pub card: BTreeMap<String, String>,
    /// `noise`, `kept`, `cores`, `cleanup` or `other` (see [`RootTruth`]).
    pub verdict: String,
    pub n_good: u32,
    /// Good spikes over the root's spikes.
    pub good_share: f64,
    /// Comma-joined, as in `truth_roots`.
    pub core_shares: String,
    pub core_sizes: String,
}

impl Exemplar {
    /// What the human did, in words: `cleanup: kept 71% good, dropped 29% noise`.
    pub fn text(&self) -> String {
        let (good, rest) = (
            (self.good_share * 100.0).round(),
            (100.0 - self.good_share * 100.0).round(),
        );
        match self.verdict.as_str() {
            "noise" => "noise: dropped all as noise".into(),
            "kept" => format!("kept: one good unit holds {good}% of the root"),
            "cores" => format!(
                "cores: {} good unit(s) carved out, {good}% of the root, {rest}% dropped as noise",
                self.n_good
            ),
            "cleanup" => format!("cleanup: kept {good}% good, dropped {rest}% noise"),
            _ => format!("other: {good}% good in {} unit(s)", self.n_good),
        }
    }

    /// A card value as a number (`None` for `-` or text).
    pub fn num(&self, field: &str) -> Option<f64> {
        self.card.get(field)?.parse().ok()
    }
}

/// Blake3 hex of a JSON value: the provenance hashes.
fn digest(v: Value) -> String {
    blake3::hash(v.to_string().as_bytes()).to_hex().to_string()
}

/// Hash of a curation the scorer reads (a session's truth, a run's output):
/// the bytes of `dir`'s `spike_clusters.npy` and `cluster_group.tsv`, the only
/// files it reads (a mua to unsorted edit changes it, though not the per-root summary).
pub fn truth_hash(dir: &Path) -> Result<String> {
    let read = |f: &str| std::fs::read(dir.join(f)).with_context(|| format!("read {f}"));
    let hex = |b: &[u8]| blake3::hash(b).to_hex().to_string();
    let tsv = dir.join("cluster_group.tsv").exists();
    Ok(digest(json!([
        hex(&read("spike_clusters.npy")?),
        tsv.then(|| read("cluster_group.tsv"))
            .transpose()?
            .map(|b| hex(&b)),
    ])))
}

/// What the score of run `rid` depends on, hashed: its session's input and
/// truth, the run's own recorded state (row, actions and output) and the
/// options of its stored score.
fn basis(conn: &Connection, rid: &str) -> Result<String> {
    let parts: [Option<String>; 4] = conn.query_row(
        "SELECT s.fingerprint, s.truth_hash, r.state_hash, r.score_opts FROM runs r JOIN sessions s USING (sid) WHERE r.rid = ?1",
        [rid],
        |r| Ok([r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?]),
    )?;
    Ok(digest(json!(parts)))
}

/// A score holds only for the exact inputs it was computed from: drop the
/// scores of `sid`'s runs whose stored `basis` is not what their inputs hash to
/// now. Every record that changes an input ends here, so none compares fields.
fn drop_stale_scores(conn: &Connection, sid: &str) -> Result<()> {
    let runs: Vec<(String, Option<String>)> = conn
        .prepare("SELECT rid, basis FROM runs WHERE sid = ?1")?
        .query_map([sid], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (rid, stored) in runs {
        if stored.as_deref() != Some(basis(conn, &rid)?.as_str()) {
            for q in [
                "DELETE FROM metrics WHERE rid = ?1",
                "DELETE FROM misses WHERE rid = ?1",
                "UPDATE runs SET basis = NULL, score_opts = NULL WHERE rid = ?1",
            ] {
                conn.execute(q, [&rid])?;
            }
        }
    }
    Ok(())
}

impl DevDb {
    /// Open `path`, creating the file and applying any pending migration.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(d) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(d).with_context(|| format!("create {}", d.display()))?;
        }
        let mut conn =
            Connection::open(path).with_context(|| format!("open {}", path.display()))?;
        conn.execute_batch("CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL)")?;
        let have: i64 = conn.query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |r| r.get(0),
        )?;
        ensure!(
            have as usize <= MIGRATIONS.len(),
            "{}: schema v{have} is newer than this build (v{})",
            path.display(),
            MIGRATIONS.len()
        );
        for (i, sql) in MIGRATIONS.iter().enumerate().skip(have as usize) {
            let tx = conn.transaction()?;
            tx.execute_batch(sql)?;
            tx.execute("INSERT INTO schema_version VALUES (?1)", [i as i64 + 1])?;
            tx.commit()?;
        }
        Ok(Self {
            conn,
            path: path.to_owned(),
            deny: Vec::new(),
        })
    }

    /// Refuse any value containing one of `deny` (sealed sids, source basenames).
    /// ponytail: substring match, entries under 3 bytes are ignored (they would
    /// match everything); a salted-hash set would keep the list itself private.
    pub fn with_deny(mut self, deny: impl IntoIterator<Item = String>) -> Self {
        self.deny = deny
            .into_iter()
            .map(|d| d.trim().to_owned())
            .filter(|d| d.len() >= 3)
            .collect();
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn deny_check(&self, what: &str, v: &str) -> Result<()> {
        // The value is never echoed: it may be exactly what the guard protects.
        ensure!(
            !self.deny.iter().any(|d| v.contains(d.as_str())),
            "privacy guard: {what} holds a denied string"
        );
        Ok(())
    }

    fn guard<'a>(&self, what: &str, v: &'a str) -> Result<&'a str> {
        ensure!(
            !v.contains(['/', '\\']),
            "privacy guard: {what} holds a path separator"
        );
        self.deny_check(what, v)?;
        Ok(v)
    }

    fn guard_opt<'a>(&self, what: &str, v: Option<&'a str>) -> Result<Option<&'a str>> {
        v.map(|v| self.guard(what, v)).transpose()
    }

    fn split_of(&self, sid: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT split FROM sessions WHERE sid = ?1", [sid], |r| {
                r.get(0)
            })
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                e => Err(e),
            })?)
    }

    /// The session must be a recorded dev one (cards, truth and runs are dev only).
    fn require_dev(&self, what: &str, sid: &str) -> Result<()> {
        match self.split_of(sid)?.as_deref() {
            Some("dev") => Ok(()),
            Some(_) => bail!("{what}: holdout sessions keep no {what}"),
            None => bail!("{what}: unknown session; record-session first"),
        }
    }

    pub fn record_session(&mut self, s: &SessionRow) -> Result<()> {
        let sid = self.guard("sid", &s.sid)?;
        ensure!(
            matches!(s.split.as_str(), "dev" | "holdout"),
            "split must be dev or holdout"
        );
        // Whatever the caller knows about a holdout session stays out.
        let s = if s.split == "holdout" {
            SessionRow {
                sid: s.sid.clone(),
                split: s.split.clone(),
                ..Default::default()
            }
        } else {
            ensure!(
                s.fingerprint.is_some(),
                "a dev session needs its fingerprint"
            );
            s.clone()
        };
        let (fp, sorter, truth_hash) = (
            self.guard_opt("fingerprint", s.fingerprint.as_deref())?,
            self.guard_opt("sorter", s.sorter.as_deref())?,
            self.guard_opt("truth hash", s.truth_hash.as_deref())?,
        );
        let added = (s.split == "dev").then_some("now");
        let tx = self.conn.transaction()?;
        let (old, old_truth): (Option<Option<String>>, Option<String>) = tx
            .query_row(
                "SELECT fingerprint, truth_hash FROM sessions WHERE sid = ?1",
                [sid],
                |r| Ok((Some(r.get(0)?), r.get(1)?)),
            )
            .optional()?
            .unwrap_or((None, None));
        tx.execute(
            "INSERT INTO sessions VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, CASE WHEN ?9 IS NULL THEN NULL ELSE date('now') END, ?10)
             ON CONFLICT(sid) DO UPDATE SET split = excluded.split, fingerprint = excluded.fingerprint,
                 sorter = excluded.sorter, n_spikes = excluded.n_spikes, n_roots = excluded.n_roots,
                 n_channels = excluded.n_channels, added = CASE WHEN excluded.added IS NULL THEN NULL ELSE COALESCE(added, excluded.added) END,
                 binary = COALESCE(excluded.binary, binary), truth_hash = COALESCE(excluded.truth_hash, truth_hash)",
            params![sid, s.split, fp, sorter, s.n_spikes, s.n_roots, s.n_channels, s.binary, added, truth_hash],
        )?;
        // A session re-recorded as holdout, or on a changed input, loses everything
        // it had: truth, cards and the runs scored against the old input describe
        // another clustering. Re-record the truth after a changed fingerprint.
        // On the same input a changed truth only stales the scores (below);
        // a re-record that supplies no truth keeps the stored one. A new truth
        // drops the old summary in this transaction, so an interrupted re-record
        // leaves no truth (skipped by the exemplar join), never stale truth.
        if truth_hash.is_some() && old_truth.as_deref() != truth_hash {
            tx.execute("DELETE FROM truth_roots WHERE sid = ?1", [sid])?;
        }
        if s.split == "holdout" || old.is_some_and(|o| o.as_deref() != fp) {
            tx.execute(
                "UPDATE sessions SET truth_hash = ?2, binary = ?3 WHERE sid = ?1",
                params![sid, truth_hash, s.binary],
            )?;
            for q in [
                "DELETE FROM metrics WHERE rid IN (SELECT rid FROM runs WHERE sid = ?1)",
                "DELETE FROM misses WHERE rid IN (SELECT rid FROM runs WHERE sid = ?1)",
                "DELETE FROM arm_actions WHERE rid IN (SELECT rid FROM runs WHERE sid = ?1)",
                "DELETE FROM runs WHERE sid = ?1",
                "DELETE FROM root_cards WHERE sid = ?1",
                "DELETE FROM truth_roots WHERE sid = ?1",
            ] {
                tx.execute(q, [sid])?;
            }
        }
        drop_stale_scores(&tx, sid)?;
        tx.commit()?;
        Ok(())
    }

    /// Cards of a dev session from a level-2 brief (see [`parse_cards`]).
    pub fn record_cards(&mut self, sid: &str, brief: &str) -> Result<usize> {
        self.require_dev("cards", sid)?;
        let cards = parse_cards(brief);
        for (_, fields) in &cards {
            for (f, v) in fields {
                self.guard("card field", f)?;
                self.guard("card value", v)?;
            }
        }
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM root_cards WHERE sid = ?1", [sid])?;
        for (root, fields) in &cards {
            for (f, v) in fields {
                tx.execute(
                    "INSERT INTO root_cards VALUES (?1, ?2, ?3, ?4)",
                    params![sid, root, f, v],
                )?;
            }
        }
        tx.commit()?;
        Ok(cards.len())
    }

    pub fn record_truth(&mut self, sid: &str, roots: &[RootTruth]) -> Result<()> {
        self.require_dev("truth", sid)?;
        // Verdicts come from `root_truths`; anything else is refused, never stored.
        if let Some(r) = roots.iter().find(|r| !VERDICTS.contains(&r.verdict)) {
            bail!("root {}: unknown truth verdict", r.root);
        }
        let tx = self.conn.transaction()?;
        let rows: Vec<_> = roots
            .iter()
            .map(|r| {
                let join = |f: &dyn Fn(&(f64, u32)) -> String| {
                    r.cores.iter().map(f).collect::<Vec<_>>().join(",")
                };
                (
                    r.root,
                    r.n_spikes,
                    r.verdict.to_owned(),
                    r.n_good,
                    r.good_share,
                    join(&|c| format!("{:.4}", c.0)),
                    join(&|c| c.1.to_string()),
                )
            })
            .collect();
        // The summary is derived data; whether the scores still hold is the
        // truth hash `record_session` stored.
        tx.execute("DELETE FROM truth_roots WHERE sid = ?1", [sid])?;
        for (root, n_spikes, verdict, n_good, share, shares, sizes) in &rows {
            tx.execute(
                "INSERT INTO truth_roots VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![sid, root, n_spikes, verdict, n_good, share, shares, sizes],
            )?;
        }
        drop_stale_scores(&tx, sid)?;
        tx.commit()?;
        Ok(())
    }

    pub fn record_variant(&mut self, v: &VariantRow) -> Result<()> {
        let name = self.guard("variant name", &v.name)?;
        let (prompt_rev, sorrel_rev, learned_rev) = (
            self.guard_opt("prompt_rev", v.prompt_rev.as_deref())?,
            self.guard_opt("sorrel_rev", v.sorrel_rev.as_deref())?,
            self.guard_opt("learned_rev", v.learned_rev.as_deref())?,
        );
        for t in [&v.rubric, &v.review].into_iter().flatten() {
            self.deny_check("variant prompt", t)?;
        }
        // A later row only fills what an earlier one lacked or changed.
        self.conn.execute(
            "INSERT INTO variants VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(name) DO UPDATE SET prompt_rev = COALESCE(excluded.prompt_rev, prompt_rev),
                 rubric = COALESCE(excluded.rubric, rubric), review = COALESCE(excluded.review, review),
                 sorrel_rev = COALESCE(excluded.sorrel_rev, sorrel_rev),
                 learned_rev = COALESCE(excluded.learned_rev, learned_rev)",
            params![name, prompt_rev, v.rubric, v.review, sorrel_rev, learned_rev],
        )?;
        Ok(())
    }

    /// A run of a dev session and the arm's actions (replacing earlier ones).
    pub fn record_run(&mut self, r: &RunRow, actions: &[ArmAction]) -> Result<()> {
        self.require_dev("runs", &r.sid)?;
        let (rid, sid) = (self.guard("rid", &r.rid)?, self.guard("sid", &r.sid)?);
        let (variant, arm, status) = (
            self.guard("variant", &r.variant)?,
            self.guard("arm", &r.arm)?,
            self.guard("status", &r.status)?,
        );
        let (model, usage) = (
            self.guard_opt("model", r.model.as_deref())?,
            self.guard_opt("usage", r.usage.as_deref())?,
        );
        let (prompt_rev, sorrel_rev, learned_rev) = (
            self.guard_opt("prompt_rev", r.prompt_rev.as_deref())?,
            self.guard_opt("sorrel_rev", r.sorrel_rev.as_deref())?,
            self.guard_opt("learned_rev", r.learned_rev.as_deref())?,
        );
        let output_hash = self.guard_opt("output hash", r.output_hash.as_deref())?;
        let mut rows = Vec::new();
        for (i, a) in actions.iter().enumerate() {
            // A model's reasons are free text: slashes become spaces, the deny set still refuses.
            let reasons = a.reasons.as_deref().map(|t| t.replace(['/', '\\'], " "));
            self.deny_check("reasons", reasons.as_deref().unwrap_or(""))?;
            rows.push((
                i as i64,
                self.guard("target", &a.target)?,
                self.guard("action", &a.action)?,
                a.confidence,
                reasons,
            ));
        }
        let state = digest(json!([
            rid,
            sid,
            variant,
            arm,
            model,
            r.repeat,
            status,
            r.cost_usd,
            r.turns,
            r.wall_s,
            usage,
            prompt_rev,
            sorrel_rev,
            learned_rev,
            output_hash,
            rows
        ]));
        let tx = self.conn.transaction()?;
        // The score's basis and options survive the replace; a changed state makes them stale.
        tx.execute(
            "INSERT OR REPLACE INTO runs VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, (SELECT basis FROM runs WHERE rid = ?1), ?16, (SELECT score_opts FROM runs WHERE rid = ?1))",
            params![rid, sid, variant, arm, model, r.repeat, status, r.cost_usd, r.turns, r.wall_s, usage, prompt_rev, sorrel_rev, learned_rev, state, output_hash],
        )?;
        tx.execute("DELETE FROM arm_actions WHERE rid = ?1", [rid])?;
        for (seq, target, action, confidence, reasons) in rows {
            tx.execute(
                "INSERT INTO arm_actions VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![rid, seq, target, action, confidence, reasons],
            )?;
        }
        drop_stale_scores(&tx, sid)?;
        tx.commit()?;
        Ok(())
    }

    /// The score of a recorded run: every numeric leaf of the diff by dotted
    /// name (`good_units.recall`), `misses.<category>` counts, and the misses.
    /// `test_hash` and `truth` are the [`truth_hash`] of the TEST and TRUTH
    /// curations that were scored, `fingerprint` that of the `--root` input:
    /// TEST must be the run's own output, the others the session's. `d` echoes
    /// the scoring options but the disputed set, so that comes in too; together
    /// they are stored with the score.
    pub fn record_diff(
        &mut self,
        rid: &str,
        d: &SpikeDiff,
        test_hash: &str,
        truth: &str,
        fingerprint: &str,
        disputed: &HashSet<u32>,
    ) -> Result<()> {
        self.guard("rid", rid)?;
        let stored: Option<[Option<String>; 3]> = self
            .conn
            .query_row(
                "SELECT r.output_hash, s.truth_hash, s.fingerprint FROM runs r JOIN sessions s USING (sid) WHERE r.rid = ?1",
                [rid],
                |r| Ok([r.get(0)?, r.get(1)?, r.get(2)?]),
            )
            .optional()?;
        let [output, s_truth, s_fp] = stored.context("unknown run; record-run first")?;
        ensure!(
            output.is_some(),
            "run has no recorded output hash; re-run record-run"
        );
        ensure!(
            output.as_deref() == Some(test_hash),
            "TEST is not the run's recorded output (its curation hashes to {test_hash}, the run's to {}); score the run's own data dir",
            output.unwrap_or_default()
        );
        ensure!(
            s_truth.is_some(),
            "session has no recorded truth; re-run record-session --truth"
        );
        ensure!(
            s_truth.as_deref() == Some(truth),
            "TRUTH is not the session's recorded truth (it hashes to {truth}, the session's to {}); score against the session's own truth",
            s_truth.unwrap_or_default()
        );
        ensure!(
            s_fp.as_deref() == Some(fingerprint),
            "--root is not the session's recorded input (its spike_clusters.npy hashes to {fingerprint}, the session's to {}); pass the session's own input",
            s_fp.unwrap_or_default()
        );
        let mut disputed: Vec<_> = disputed.iter().collect();
        disputed.sort();
        let opts = json!({
            "match_jaccard": d.match_jaccard,
            "min_spikes": d.min_spikes,
            "split_share": d.split_share,
            "binary": d.binary,
            "disputed": disputed,
        })
        .to_string();
        let mut metrics = BTreeMap::new();
        // Settings and sizes of the comparison, not scores.
        let skip = [
            "misses",
            "n_spikes",
            "n_root_clusters",
            "match_jaccard",
            "min_spikes",
            "split_share",
        ];
        flatten(&serde_json::to_value(d)?, "", &skip, &mut metrics);
        for m in &d.misses {
            *metrics
                .entry(format!("misses.{}", m.category))
                .or_insert(0.0) += 1.0;
        }
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM metrics WHERE rid = ?1", [rid])?;
        tx.execute("DELETE FROM misses WHERE rid = ?1", [rid])?;
        for (name, v) in &metrics {
            tx.execute(
                "INSERT INTO metrics VALUES (?1, ?2, ?3)",
                params![rid, name, v],
            )?;
        }
        for m in &d.misses {
            let roots = m
                .roots
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",");
            tx.execute(
                "INSERT OR REPLACE INTO misses VALUES (?1, ?2, ?3, ?4, ?5)",
                params![rid, m.category, roots, m.truth, m.test],
            )?;
        }
        tx.execute(
            "UPDATE runs SET score_opts = ?2 WHERE rid = ?1",
            params![rid, opts],
        )?;
        let basis = basis(&tx, rid)?;
        tx.execute(
            "UPDATE runs SET basis = ?2 WHERE rid = ?1",
            params![rid, basis],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn record_ledger(&mut self, rows: &[LedgerRow]) -> Result<()> {
        for r in rows {
            for (w, v) in [
                ("query", &r.query),
                ("baseline", &r.baseline),
                ("candidate", &r.candidate),
                ("arm", &r.arm),
                ("metric", &r.metric),
            ] {
                self.guard(w, v)?;
            }
        }
        let tx = self.conn.transaction()?;
        for r in rows {
            tx.execute(
                "INSERT OR REPLACE INTO holdout_ledger VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![r.query, r.baseline, r.candidate, r.arm, r.metric, r.n_sessions, r.base, r.cand, r.delta, r.ci_lo, r.ci_hi],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Dev root clusters with their card and truth verdict, ordered by
    /// (fingerprint, root). Sessions whose fingerprint is `exclude_fingerprint`
    /// are left out (leave-one-session-out for the dataset being reviewed).
    pub fn exemplars(&self, exclude_fingerprint: Option<&str>) -> Result<Vec<Exemplar>> {
        let mut st = self.conn.prepare(
            "SELECT s.sid, s.fingerprint, t.root, t.verdict, t.n_good, t.core_shares, t.core_sizes, c.field, c.value, t.good_share
             FROM truth_roots t JOIN sessions s ON s.sid = t.sid AND s.split = 'dev'
             JOIN root_cards c ON c.sid = t.sid AND c.root = t.root
             WHERE s.fingerprint IS NOT NULL AND s.fingerprint IS NOT ?1
             ORDER BY s.fingerprint, s.sid, t.root, c.field",
        )?;
        let mut out: Vec<Exemplar> = Vec::new();
        let mut rows = st.query([exclude_fingerprint])?;
        while let Some(r) = rows.next()? {
            let (sid, root): (String, u32) = (r.get(0)?, r.get(2)?);
            if out.last().is_none_or(|e| (&e.sid, e.root) != (&sid, root)) {
                out.push(Exemplar {
                    fingerprint: r.get(1)?,
                    verdict: r.get(3)?,
                    n_good: r.get(4)?,
                    core_shares: r.get(5)?,
                    core_sizes: r.get(6)?,
                    good_share: r.get(9)?,
                    card: BTreeMap::new(),
                    sid,
                    root,
                });
            }
            out.last_mut().unwrap().card.insert(r.get(7)?, r.get(8)?);
        }
        Ok(out)
    }

    /// Run `sql` on a separate read-only connection (open flag and
    /// `PRAGMA query_only`); rows as TSV with a header line.
    pub fn query(&self, sql: &str) -> Result<String> {
        let ro = Connection::open_with_flags(
            &self.path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        ro.execute_batch("PRAGMA query_only = ON")?;
        let mut st = ro.prepare(sql)?;
        let n = st.column_count();
        let mut out = st.column_names().collect::<Vec<_>>().join("\t") + "\n";
        let mut rows = st.query([])?;
        while let Some(r) = rows.next()? {
            let cells: Vec<String> = (0..n)
                .map(|i| {
                    Ok(match r.get_ref(i)? {
                        ValueRef::Null => String::new(),
                        ValueRef::Integer(v) => v.to_string(),
                        ValueRef::Real(v) => v.to_string(),
                        ValueRef::Text(t) => String::from_utf8_lossy(t).replace(['\t', '\n'], " "),
                        ValueRef::Blob(b) => format!("<{} bytes>", b.len()),
                    })
                })
                .collect::<rusqlite::Result<_>>()?;
            out += &cells.join("\t");
            out.push('\n');
        }
        Ok(out)
    }

    /// Schema version, size and row counts.
    pub fn status(&self) -> Result<String> {
        let one = |sql: &str| self.conn.query_row(sql, [], |r| r.get::<_, i64>(0));
        let mut s = format!(
            "{}\nschema v{}, {} bytes\n",
            self.path.display(),
            one("SELECT MAX(version) FROM schema_version")?,
            std::fs::metadata(&self.path)?.len()
        );
        for t in [
            "sessions",
            "variants",
            "runs",
            "metrics",
            "misses",
            "root_cards",
            "truth_roots",
            "arm_actions",
            "holdout_ledger",
        ] {
            let _ = writeln!(s, "{t}\t{}", one(&format!("SELECT COUNT(*) FROM {t}"))?);
        }
        let _ = writeln!(
            s,
            "dev sessions\t{}",
            one("SELECT COUNT(*) FROM sessions WHERE split = 'dev'")?
        );
        let _ = writeln!(
            s,
            "holdout sessions\t{}",
            one("SELECT COUNT(*) FROM sessions WHERE split = 'holdout'")?
        );
        Ok(s)
    }

    /// Write `exemplars.tsv`, `priors.toml`, `thresholds.toml` and `lineage.md`
    /// into `out` from the dev sessions only; returns `learned_rev`, a hash of
    /// the three consumed files' content. Each file's header names the DB's
    /// sha, the dev sids used and that rev (`# ` lines, `<!-- -->` in the md).
    pub fn distill(&self, out: &Path) -> Result<String> {
        let ex = self.exemplars(None)?;
        let sids: BTreeSet<&str> = ex.iter().map(|e| e.sid.as_str()).collect();
        let (tsv, priors, thresholds) = (exemplars_tsv(&ex), priors_toml(&ex), thresholds_toml());
        let rev = blake3::hash(
            [&tsv, &priors, &thresholds]
                .map(String::as_str)
                .join("\0")
                .as_bytes(),
        )
        .to_hex()[..12]
            .to_owned();
        let sha = blake3::hash(&std::fs::read(&self.path)?).to_hex()[..12].to_owned();
        let head = format!(
            "# db_sha={sha}\n# dev_sids={}\n# learned_rev={rev}\n",
            sids.iter().copied().collect::<Vec<_>>().join(",")
        );
        std::fs::create_dir_all(out)?;
        for (name, body) in [
            ("exemplars.tsv", &tsv),
            ("priors.toml", &priors),
            ("thresholds.toml", &thresholds),
        ] {
            std::fs::write(out.join(name), format!("{head}{body}"))?;
        }
        let md_head = format!(
            "<!-- db_sha={sha} dev_sids={} learned_rev={rev} -->\n",
            sids.iter().copied().collect::<Vec<_>>().join(",")
        );
        std::fs::write(out.join("lineage.md"), md_head + &self.lineage()?)?;
        Ok(rev)
    }

    /// One row per (variant, arm, revisions, score options): mean dev metrics over its scored runs and
    /// the holdout ledger rows that name it as the candidate.
    fn lineage(&self) -> Result<String> {
        let mut md = String::from("# Prompt lineage\n\nDev metrics are means over scored runs; holdout is the ledger delta of good_units.recall against its baseline.\n\n");
        md += &format!("| variant | arm | runs | prompt_rev | sorrel_rev | learned_rev | score_opts | {} | cost_usd | holdout |\n", LINEAGE_METRICS.join(" | "));
        md += &format!(
            "|---|---|---|---|---|---|---|{}---|---|\n",
            "---|".repeat(LINEAGE_METRICS.len())
        );
        let mut st = self.conn.prepare(
            "SELECT variant, arm, COUNT(*), prompt_rev, sorrel_rev, learned_rev, AVG(cost_usd), score_opts
             FROM runs GROUP BY variant, arm, prompt_rev, sorrel_rev, learned_rev, score_opts
             ORDER BY MIN(rowid), variant, arm",
        )?;
        let cell = |v: Option<f64>| v.map_or("-".into(), |v| format!("{v:.3}"));
        let txt = |v: Option<String>| v.unwrap_or_else(|| "-".into());
        for row in st.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
                [3, 4, 5].map(|i| r.get::<_, Option<String>>(i).unwrap_or(None)),
                r.get::<_, Option<f64>>(6)?,
                r.get::<_, Option<String>>(7)?,
            ))
        })? {
            let (variant, arm, n, revs, cost, opts) = row?;
            let mean = |name: &str| -> Result<Option<f64>> {
                Ok(self.conn.query_row(
                    "SELECT AVG(m.value) FROM metrics m JOIN runs r USING (rid) WHERE r.variant = ?1 AND r.arm = ?2 AND m.name = ?3
                     AND r.prompt_rev IS ?4 AND r.sorrel_rev IS ?5 AND r.learned_rev IS ?6 AND r.score_opts IS ?7",
                    params![variant, arm, name, revs[0], revs[1], revs[2], opts],
                    |r| r.get(0),
                )?)
            };
            let means = LINEAGE_METRICS
                .iter()
                .map(|m| mean(m).map(cell))
                .collect::<Result<Vec<_>>>()?;
            let holdout: Vec<String> = self
                .conn
                .prepare("SELECT baseline, delta, n_sessions FROM holdout_ledger WHERE candidate = ?1 AND arm = ?2 AND metric = 'good_units.recall'")?
                .query_map(params![variant, arm], |r| Ok(format!("{:+.3} vs {} (n={})", r.get::<_, f64>(1)?, r.get::<_, String>(0)?, r.get::<_, i64>(2)?)))?
                .collect::<rusqlite::Result<_>>()?;
            // The ledger records no configuration: a holdout result belongs to a
            // row only when the variant/arm was run under exactly one.
            let n_cfg: i64 = self.conn.query_row(
                "SELECT COUNT(*) FROM (SELECT DISTINCT prompt_rev, sorrel_rev, learned_rev, score_opts FROM runs WHERE variant = ?1 AND arm = ?2)",
                params![variant, arm],
                |r| r.get(0),
            )?;
            let holdout = match (holdout.is_empty(), n_cfg) {
                (true, _) => "-".into(),
                (false, 1) => holdout.join("; "),
                (false, n) => format!("see ledger ({n} configurations)"),
            };
            let [p, s, l] = revs.map(txt);
            let opts = txt(opts);
            md += &format!(
                "| {variant} | {arm} | {n} | {p} | {s} | {l} | {opts} | {} | {} | {holdout} |\n",
                means.join(" | "),
                cell(cost)
            );
        }
        Ok(md)
    }
}

/// Numeric leaves of `v` by dotted name; arrays, bools and keys in `skip` (top level) are left out.
fn flatten(v: &Value, prefix: &str, skip: &[&str], out: &mut BTreeMap<String, f64>) {
    match v {
        Value::Object(o) => {
            for (k, x) in o
                .iter()
                .filter(|(k, _)| !prefix.is_empty() || !skip.contains(&k.as_str()))
            {
                flatten(x, &format!("{prefix}{k}."), skip, out);
            }
        }
        Value::Number(n) => {
            out.insert(
                prefix.trim_end_matches('.').to_owned(),
                n.as_f64().unwrap_or(f64::NAN),
            );
        }
        _ => {}
    }
}

/// The rows of the `cards:` table of a level-2 brief as (root, (column, value)).
/// The first column is the root id; the table ends at the first line that does
/// not start with a digit (`relations:`, `omitted:`, ...).
pub fn parse_cards(brief: &str) -> Vec<(u32, Vec<(String, String)>)> {
    let mut lines = brief.lines().skip_while(|l| *l != "cards:").skip(1);
    let Some(head) = lines.next() else {
        return Vec::new();
    };
    let cols: Vec<&str> = head.split('\t').collect();
    lines
        .take_while(|l| l.starts_with(|c: char| c.is_ascii_digit()))
        .filter_map(|l| {
            let mut f = l.split('\t');
            let root = f.next()?.parse().ok()?;
            Some((
                root,
                cols[1..]
                    .iter()
                    .zip(f)
                    .map(|(c, v)| (c.to_string(), v.to_owned()))
                    .collect(),
            ))
        })
        .collect()
}

/// Journal records to the applied path's actions (head back to the root,
/// oldest first). A note is not an action: its text becomes the `reasons` of
/// the actions on that cluster; `evidence` (auto's evidence.json) adds the
/// LLM's confidence and reasons for the units it decided.
pub fn arm_actions(records: &[Record], evidence: Option<&Value>) -> Vec<ArmAction> {
    let mut head = None;
    let mut nodes: HashMap<u64, (&Option<u64>, &CurationCommand)> = HashMap::new();
    for r in records {
        match r {
            Record::Node {
                id, parent, cmd, ..
            } => {
                nodes.insert(*id, (parent, cmd));
                head = Some(*id);
            }
            Record::Head { node } => head = *node,
            Record::Seal { .. } => {}
        }
    }
    let mut path = Vec::new();
    while let Some((parent, cmd)) = head.and_then(|h| nodes.get(&h)) {
        path.push(*cmd);
        head = **parent;
    }
    let mut flat = Vec::new();
    for cmd in path.into_iter().rev() {
        match cmd {
            CurationCommand::Batch { children } => flat.extend(children),
            c => flat.push(c),
        }
    }
    let mut notes: HashMap<u32, String> = HashMap::new();
    for c in &flat {
        if let CurationCommand::Note { cluster, text } = c {
            notes.insert(cluster.0, text.clone());
        }
    }
    let llm = |c: u32| {
        evidence
            .and_then(|e| e["units"].get(c.to_string()))
            .map(|u| &u["llm"])
            .filter(|l| l.is_object())
    };
    flat.iter()
        .filter_map(|c| {
            let (cluster, target, action) = match c {
                CurationCommand::Relabel { cluster, op } => {
                    let label = match op {
                        PhyLabelOp::SetUnsorted => "unsorted",
                        PhyLabelOp::SetGood => "good",
                        PhyLabelOp::SetMua => "mua",
                        PhyLabelOp::SetNoise => "noise",
                    };
                    (cluster.0, cluster.0.to_string(), format!("relabel:{label}"))
                }
                CurationCommand::Merge { sources, target } => (
                    target.0,
                    sources
                        .iter()
                        .map(|s| s.0.to_string())
                        .collect::<Vec<_>>()
                        .join("+"),
                    "merge".into(),
                ),
                CurationCommand::Split { cluster, by, .. } => {
                    (cluster.0, cluster.0.to_string(), format!("split:{by}"))
                }
                CurationCommand::Tag { cluster, tags } => (
                    cluster.0,
                    cluster.0.to_string(),
                    format!("tag:{}", tags.join(",")),
                ),
                CurationCommand::AddSpikes(b) => (
                    b.clusters.first().map_or(0, |c| c.0),
                    "-".into(),
                    "add_spikes".into(),
                ),
                CurationCommand::Note { .. }
                | CurationCommand::Batch { .. }
                | CurationCommand::Undo
                | CurationCommand::Redo => return None,
            };
            let l = llm(cluster);
            let reasons = l
                .and_then(|l| l["reasons"].as_array())
                .map(|r| {
                    r.iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join("; ")
                })
                .or_else(|| notes.get(&cluster).cloned());
            Some(ArmAction {
                target,
                action,
                confidence: l.and_then(|l| l["confidence"].as_f64()),
                reasons,
            })
        })
        .collect()
}

fn exemplars_tsv(ex: &[Exemplar]) -> String {
    let skip: BTreeSet<&str> = CARD_HEAD.iter().chain(&CARD_TAIL).copied().collect();
    let metrics: BTreeSet<&str> = ex
        .iter()
        .flat_map(|e| e.card.keys().map(String::as_str))
        .filter(|k| !skip.contains(k))
        .collect();
    let cols: Vec<&str> = CARD_HEAD
        .iter()
        .copied()
        .chain(metrics)
        .chain(CARD_TAIL)
        .collect();
    let mut s = format!(
        "fingerprint\troot\tverdict\tn_good\tgood_share\tcore_shares\tcore_sizes\ttext\t{}\n",
        cols.join("\t")
    );
    for e in ex {
        let _ = write!(
            s,
            "{}\t{}\t{}\t{}\t{:.4}\t{}\t{}\t{}",
            e.fingerprint,
            e.root,
            e.verdict,
            e.n_good,
            e.good_share,
            e.core_shares,
            e.core_sizes,
            e.text()
        );
        for c in &cols {
            let _ = write!(s, "\t{}", e.card.get(*c).map_or("-", String::as_str));
        }
        s.push('\n');
    }
    s
}

/// Mean/sd standardisation and three ridge logistic fits on the dev exemplars:
/// P(cores), P(cleanup) and P(noise) of a root (the verdicts of [`RootTruth`]).
fn priors_toml(ex: &[Exemplar]) -> String {
    let skip: BTreeSet<&str> = CARD_HEAD.iter().chain(&CARD_TAIL).copied().collect();
    let fields: BTreeSet<&str> = ex
        .iter()
        .flat_map(|e| e.card.keys().map(String::as_str))
        .filter(|k| !skip.contains(k))
        .collect();
    // (name, mean, sd) of every metric measured on at least two roots with some spread.
    let cols: Vec<(&str, f64, f64)> = fields
        .into_iter()
        .filter_map(|f| {
            let v: Vec<f64> = ex
                .iter()
                .filter_map(|e| e.num(f))
                .filter(|x| x.is_finite())
                .collect();
            let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
            let sd =
                (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt();
            (v.len() >= 2 && sd > 1e-9).then_some((f, mean, sd))
        })
        .collect();
    // A root missing a metric sits at the mean (0 after standardising).
    let x: Vec<Vec<f64>> = ex
        .iter()
        .map(|e| {
            cols.iter()
                .map(|&(f, m, sd)| {
                    e.num(f)
                        .filter(|v| v.is_finite())
                        .map_or(0.0, |v| (v - m) / sd)
                })
                .collect()
        })
        .collect();
    let mut s = String::from("[features.mean]\n");
    for (f, m, _) in &cols {
        let _ = writeln!(s, "{f} = {m:e}");
    }
    s += "\n[features.sd]\n";
    for (f, _, sd) in &cols {
        let _ = writeln!(s, "{f} = {sd:e}");
    }
    for name in ["cores", "cleanup", "noise"] {
        let y: Vec<bool> = ex.iter().map(|e| e.verdict == name).collect();
        let (bias, w) = fit_logistic(&x, &y, 0.1);
        let _ = write!(
            s,
            "\n[{name}]\nn = {}\nn_pos = {}\nbias = {bias:e}\n\n[{name}.coef]\n",
            y.len(),
            y.iter().filter(|&&v| v).count()
        );
        for ((f, ..), w) in cols.iter().zip(w) {
            let _ = writeln!(s, "{f} = {w:e}");
        }
    }
    s
}

/// The constants in force when nothing has been fitted: the brief's flag rules.
fn thresholds_toml() -> String {
    let mut s = String::from("source = \"defaults\"\n");
    for f in crate::brief::FLAGS {
        let _ = write!(
            s,
            "\n[flags.{}]\nmetric = {:?}\nop = {:?}\nvalue = {}\n",
            f.code, f.metric, f.op, f.value
        );
    }
    s
}

/// L2-regularised logistic regression (full-batch gradient descent, bias
/// unpenalised) on standardised features: `(bias, coefficients)`.
/// ponytail: fixed 500 steps of 0.5; a line search or L-BFGS if the features ever get ill-conditioned.
fn fit_logistic(x: &[Vec<f64>], y: &[bool], l2: f64) -> (f64, Vec<f64>) {
    let d = x.first().map_or(0, Vec::len);
    let inv = 1.0 / x.len().max(1) as f64;
    let (mut b, mut w) = (0.0, vec![0.0; d]);
    for _ in 0..500 {
        let (mut gb, mut gw) = (0.0, vec![0.0; d]);
        for (xi, &yi) in x.iter().zip(y) {
            let z = b + xi.iter().zip(&w).map(|(a, c)| a * c).sum::<f64>();
            let e = 1.0 / (1.0 + (-z).exp()) - f64::from(u8::from(yi));
            gb += e;
            gw.iter_mut().zip(xi).for_each(|(g, a)| *g += e * a);
        }
        b -= 0.5 * gb * inv;
        w.iter_mut()
            .zip(&gw)
            .for_each(|(wj, gj)| *wj -= 0.5 * (gj * inv + l2 * *wj));
    }
    (b, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::{Actor, Method};
    use crate::diff::{diff_spikes, Curation, SpikeDiffOpts};
    use sorrel_io::kilosort::PhyLabel;
    use sorrel_io::ClusterId;

    fn db(dir: &Path) -> DevDb {
        DevDb::open(&dir.join("r.sqlite"))
            .unwrap()
            .with_deny(["deadbeef".into(), "sess_ks4".into()])
    }

    fn dev(sid: &str, fp: &str) -> SessionRow {
        SessionRow {
            sid: sid.into(),
            split: "dev".into(),
            fingerprint: Some(fp.into()),
            n_roots: Some(3),
            truth_hash: Some("t1".into()),
            ..Default::default()
        }
    }

    const BRIEF: &str = "# sorrel brief v1\ncards:\nid\tlab\tflags\ttags\tn_spikes\tfiring_rate\tacg\tamp_t\tev\n\
        0\tunsorted\t=\t-\t100\t1.5\tabc\tcde\t-\n1\tunsorted\t+RPV\t-\t200\t3\tabc\tcde\t-\n2\tunsorted\t=\t-\t300\t6\tabc\tcde\t-\nomitted: none\n";

    fn truth(root: u32, verdict: &'static str, n_good: u32) -> RootTruth {
        let cores = if verdict == "noise" {
            vec![]
        } else {
            vec![(0.25, 25)]
        };
        RootTruth {
            root,
            n_spikes: 100,
            verdict,
            n_good,
            good_share: if verdict == "noise" { 0.0 } else { 0.71 },
            cores,
        }
    }

    fn fill(db: &mut DevDb, sid: &str, fp: &str) {
        db.record_session(&dev(sid, fp)).unwrap();
        db.record_cards(sid, BRIEF).unwrap();
        db.record_truth(
            sid,
            &[
                truth(0, "noise", 0),
                truth(1, "cores", 1),
                truth(2, "cleanup", 1),
            ],
        )
        .unwrap();
    }

    #[test]
    fn parses_cards_until_the_next_section() {
        let c = parse_cards(BRIEF);
        assert_eq!(c.len(), 3);
        assert_eq!(c[1].0, 1);
        assert!(
            c[1].1.contains(&("flags".into(), "+RPV".into()))
                && c[1].1.contains(&("firing_rate".into(), "3".into()))
        );
    }

    #[test]
    fn records_are_idempotent_upserts() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        for _ in 0..2 {
            fill(&mut d, "s1", "fp1");
            d.record_variant(&VariantRow {
                name: "v0".into(),
                prompt_rev: Some("p".into()),
                rubric: Some("good/mua rule".into()),
                ..Default::default()
            })
            .unwrap();
            d.record_variant(&VariantRow {
                name: "v0".into(),
                sorrel_rev: Some("r".into()),
                ..Default::default()
            })
            .unwrap();
            let run = RunRow {
                rid: "r1".into(),
                sid: "s1".into(),
                variant: "v0".into(),
                arm: "mcp-review".into(),
                status: "ok".into(),
                ..Default::default()
            };
            let act = ArmAction {
                target: "1".into(),
                action: "relabel:good".into(),
                confidence: Some(0.9),
                reasons: Some("isi/acg clean".into()),
            };
            d.record_run(&run, &[act]).unwrap();
            d.record_ledger(&[LedgerRow {
                query: "q1".into(),
                baseline: "v0".into(),
                candidate: "v1".into(),
                arm: "mcp-review".into(),
                metric: "m".into(),
                n_sessions: 3,
                base: 0.1,
                cand: 0.2,
                delta: 0.1,
                ci_lo: None,
                ci_hi: None,
            }])
            .unwrap();
        }
        let n = |t: &str| {
            d.query(&format!("SELECT COUNT(*) FROM {t}"))
                .unwrap()
                .lines()
                .nth(1)
                .unwrap()
                .to_owned()
        };
        assert_eq!(
            [
                n("sessions"),
                n("root_cards"),
                n("truth_roots"),
                n("variants"),
                n("runs"),
                n("arm_actions"),
                n("holdout_ledger")
            ],
            ["1", "24", "3", "1", "1", "1", "1"]
        );
        assert_eq!(
            d.query("SELECT sorrel_rev, rubric FROM variants").unwrap(),
            "sorrel_rev\trubric\nr\tgood/mua rule\n"
        );
        assert_eq!(
            d.query("SELECT reasons FROM arm_actions").unwrap(),
            "reasons\nisi acg clean\n"
        );
    }

    #[test]
    fn privacy_guard_refuses_paths_basenames_and_sealed_sids() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        for bad in ["/data/x", "a\\b", "ok/ok", "sess_ks4", "run-deadbeef-r0"] {
            let e = d
                .record_session(&SessionRow {
                    sid: bad.into(),
                    split: "holdout".into(),
                    ..Default::default()
                })
                .unwrap_err()
                .to_string();
            assert!(e.starts_with("privacy guard") && !e.contains(bad), "{e}");
        }
        d.record_session(&dev("s1", "fp1")).unwrap();
        let run = |variant: &str| RunRow {
            rid: "r1".into(),
            sid: "s1".into(),
            variant: variant.into(),
            arm: "a".into(),
            status: "ok".into(),
            ..Default::default()
        };
        assert!(
            d.record_run(&run("v/x"), &[]).is_err() && d.record_run(&run("sess_ks4"), &[]).is_err()
        );
        let act = ArmAction {
            target: "1".into(),
            action: "tag:x".into(),
            confidence: None,
            reasons: Some("seen in deadbeef".into()),
        };
        assert!(d.record_run(&run("v0"), &[act]).is_err());
        // prompt text may hold a slash, but not a denied string
        let prompt = |t: &str| VariantRow {
            name: "v".into(),
            review: Some(t.into()),
            ..Default::default()
        };
        d.record_variant(&prompt("good/mua")).unwrap();
        assert!(d.record_variant(&prompt("see sess_ks4")).is_err());
        // provenance hashes are guarded too: a path or a denied name never lands
        for bad in ["/data/x", "sess_ks4"] {
            assert!(d
                .record_session(&SessionRow {
                    truth_hash: Some(bad.into()),
                    ..dev("s1", "fp1")
                })
                .is_err());
            assert!(d
                .record_run(
                    &RunRow {
                        output_hash: Some(bad.into()),
                        ..run("v0")
                    },
                    &[]
                )
                .is_err());
        }
        assert_eq!(
            d.query("SELECT COUNT(*) FROM runs").unwrap(),
            "COUNT(*)\n0\n"
        );
    }

    #[test]
    fn query_is_read_only() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        d.record_session(&dev("s1", "fp1")).unwrap();
        for sql in [
            "DELETE FROM sessions",
            "PRAGMA query_only = OFF",
            "INSERT INTO variants(name) VALUES ('x')",
            "DROP TABLE sessions",
        ] {
            let _ = d.query(sql);
            assert_eq!(
                d.query("SELECT COUNT(*) FROM sessions").unwrap(),
                "COUNT(*)\n1\n",
                "{sql}"
            );
        }
        assert!(d.query("DELETE FROM sessions").is_err());
        assert!(d.query("INSERT INTO variants(name) VALUES ('x')").is_err());
    }

    #[test]
    fn a_holdout_session_keeps_no_cards_truth_or_runs() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        d.record_session(&SessionRow {
            sid: "holdout-1".into(),
            split: "holdout".into(),
            fingerprint: Some("fp9".into()),
            n_spikes: Some(5),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            d.query("SELECT fingerprint, n_spikes, added FROM sessions")
                .unwrap(),
            "fingerprint\tn_spikes\tadded\n\t\t\n"
        );
        assert!(d.record_cards("holdout-1", BRIEF).is_err());
        assert!(d
            .record_truth("holdout-1", &[truth(0, "noise", 0)])
            .is_err());
        let run = RunRow {
            rid: "r".into(),
            sid: "holdout-1".into(),
            variant: "v".into(),
            arm: "a".into(),
            status: "ok".into(),
            ..Default::default()
        };
        assert!(d.record_run(&run, &[]).is_err());
        assert!(d.record_cards("nope", BRIEF).is_err());
        // a dev session re-recorded as holdout loses what it had
        fill(&mut d, "s1", "fp1");
        d.record_session(&SessionRow {
            sid: "s1".into(),
            split: "holdout".into(),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(d.query("SELECT (SELECT COUNT(*) FROM root_cards) + (SELECT COUNT(*) FROM truth_roots) AS n").unwrap(), "n\n0\n");
    }

    /// A run of a dev session with an action, a metric and a miss.
    fn scored_run(d: &mut DevDb, sid: &str, rid: &str, sorrel_rev: Option<&str>, recall: f64) {
        record_plain_run(d, sid, rid, sorrel_rev);
        d.conn
            .execute(
                "INSERT INTO metrics VALUES (?1, 'good_units.recall', ?2)",
                params![rid, recall],
            )
            .unwrap();
        d.conn
            .execute(
                "INSERT INTO misses VALUES (?1, 'missed_merge', '1,2', NULL, NULL)",
                [rid],
            )
            .unwrap();
        d.conn
            .execute(
                "UPDATE runs SET basis = ?2 WHERE rid = ?1",
                params![rid, basis(&d.conn, rid).unwrap()],
            )
            .unwrap();
    }

    /// A run of a dev session with an action, not scored.
    fn record_plain_run(d: &mut DevDb, sid: &str, rid: &str, sorrel_rev: Option<&str>) {
        record_run_doing(d, sid, rid, sorrel_rev, "relabel:good");
    }

    /// As [`record_plain_run`], the one action being `action`.
    fn record_run_doing(
        d: &mut DevDb,
        sid: &str,
        rid: &str,
        sorrel_rev: Option<&str>,
        action: &str,
    ) {
        let run = RunRow {
            rid: rid.into(),
            sid: sid.into(),
            variant: "v0".into(),
            arm: "a".into(),
            status: "ok".into(),
            sorrel_rev: sorrel_rev.map(Into::into),
            ..Default::default()
        };
        let act = ArmAction {
            target: "1".into(),
            action: action.into(),
            confidence: None,
            reasons: None,
        };
        d.record_run(&run, &[act]).unwrap();
    }

    /// Rows of `tables` (all session-derived ones by default).
    fn rows_in(d: &DevDb, tables: &[&str]) -> i64 {
        tables
            .iter()
            .map(|t| {
                d.conn
                    .query_row(&format!("SELECT COUNT(*) FROM {t}"), [], |r| {
                        r.get::<_, i64>(0)
                    })
                    .unwrap()
            })
            .sum()
    }

    const RUN_DATA: [&str; 4] = ["runs", "metrics", "misses", "arm_actions"];

    #[test]
    fn re_recording_as_holdout_purges_the_runs_too() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        scored_run(&mut d, "s1", "r1", None, 0.5);
        assert_eq!(rows_in(&d, &RUN_DATA), 4);
        d.record_session(&SessionRow {
            sid: "s1".into(),
            split: "holdout".into(),
            ..Default::default()
        })
        .unwrap();
        let all = [&RUN_DATA[..], &["root_cards", "truth_roots"]].concat();
        assert_eq!(rows_in(&d, &all), 0);
        assert_eq!(rows_in(&d, &["sessions"]), 1);
    }

    #[test]
    fn a_changed_input_drops_the_stale_truth_and_runs_but_the_same_one_keeps_them() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        scored_run(&mut d, "s1", "r1", None, 0.5);
        // The same input again, without truth: nothing changes.
        d.record_session(&dev("s1", "fp1")).unwrap();
        d.record_cards("s1", BRIEF).unwrap();
        assert_eq!(rows_in(&d, &["truth_roots"]), 3);
        assert_eq!(rows_in(&d, &RUN_DATA), 4);
        // A new input without truth: nothing scored against the old one remains.
        d.record_session(&dev("s1", "fp2")).unwrap();
        d.record_cards("s1", BRIEF).unwrap();
        assert_eq!(rows_in(&d, &["truth_roots"]), 0);
        assert_eq!(rows_in(&d, &RUN_DATA), 0);
        assert!(d.exemplars(None).unwrap().is_empty());
    }

    #[test]
    fn a_dev_session_re_recorded_as_holdout_keeps_no_date() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        assert_eq!(
            d.query("SELECT added IS NULL AS a FROM sessions").unwrap(),
            "a\n0\n"
        );
        d.record_session(&SessionRow {
            sid: "s1".into(),
            split: "holdout".into(),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            d.query("SELECT split, fingerprint IS NULL AS f, added IS NULL AS a FROM sessions")
                .unwrap(),
            "split\tf\ta\nholdout\t1\t1\n"
        );
    }

    #[test]
    fn a_changed_run_drops_its_scores_but_an_identical_re_record_keeps_them() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        scored_run(&mut d, "s1", "r1", Some("rev1"), 0.5);
        scored_run(&mut d, "s1", "r2", Some("rev1"), 0.5);
        // the same run again (runner, backfill): scores stay
        record_plain_run(&mut d, "s1", "r1", Some("rev1"));
        assert_eq!(rows_in(&d, &["metrics", "misses"]), 4);
        // r1 changed: its scores go, r2's stay, the run is refreshed
        record_plain_run(&mut d, "s1", "r1", Some("rev2"));
        assert_eq!(rows_in(&d, &["metrics", "misses"]), 2);
        assert_eq!(
            d.query("SELECT COUNT(*) AS n FROM metrics WHERE rid = 'r1'")
                .unwrap(),
            "n\n0\n"
        );
        assert_eq!(rows_in(&d, &["runs", "arm_actions"]), 4);
    }

    #[test]
    fn a_truth_edit_with_the_same_summary_drops_the_scores_and_an_identical_one_keeps_all() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        scored_run(&mut d, "s1", "r1", None, 0.5);
        fill(&mut d, "s1", "fp1"); // the same truth again
        assert_eq!(rows_in(&d, &RUN_DATA), 4);
        // mua turned unsorted: other truth bytes, the same per-root summary
        d.record_session(&SessionRow {
            truth_hash: Some("t2".into()),
            ..dev("s1", "fp1")
        })
        .unwrap();
        let again = [
            truth(0, "noise", 0),
            truth(1, "cores", 1),
            truth(2, "cleanup", 1),
        ];
        d.record_truth("s1", &again).unwrap();
        assert_eq!(rows_in(&d, &["metrics", "misses"]), 0);
        assert_eq!(rows_in(&d, &["runs", "arm_actions", "truth_roots"]), 5);
        // scored again against t2, an identical re-record keeps it
        scored_run(&mut d, "s1", "r1", None, 0.5);
        d.record_session(&SessionRow {
            truth_hash: Some("t2".into()),
            ..dev("s1", "fp1")
        })
        .unwrap();
        d.record_truth("s1", &again).unwrap();
        assert_eq!(rows_in(&d, &["metrics", "misses"]), 2);
    }

    #[test]
    fn an_interrupted_truth_re_record_leaves_no_stale_verdicts() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        assert_eq!(rows_in(&d, &["truth_roots"]), 3);
        // the same truth again keeps its summary
        d.record_session(&dev("s1", "fp1")).unwrap();
        assert_eq!(rows_in(&d, &["truth_roots"]), 3);
        // a new truth whose record_truth never ran: no summary rather than the old one
        d.record_session(&SessionRow {
            truth_hash: Some("t2".into()),
            ..dev("s1", "fp1")
        })
        .unwrap();
        assert_eq!(rows_in(&d, &["truth_roots"]), 0);
    }

    #[test]
    fn a_re_record_without_truth_keeps_the_stored_truth_and_its_scores() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        scored_run(&mut d, "s1", "r1", None, 0.5);
        // same input, no truth supplied: the stored truth still stands
        d.record_session(&SessionRow {
            truth_hash: None,
            ..dev("s1", "fp1")
        })
        .unwrap();
        assert_eq!(
            d.query("SELECT truth_hash FROM sessions").unwrap(),
            "truth_hash\nt1\n"
        );
        assert_eq!(rows_in(&d, &["metrics", "misses"]), 2);
        // a changed input with no truth: truth, hash and scores all go
        d.record_session(&SessionRow {
            truth_hash: None,
            ..dev("s1", "fp2")
        })
        .unwrap();
        assert_eq!(
            d.query("SELECT truth_hash IS NULL AS t FROM sessions")
                .unwrap(),
            "t\n1\n"
        );
        assert_eq!(rows_in(&d, &["metrics", "misses", "truth_roots"]), 0);
    }

    #[test]
    fn a_changed_action_with_the_same_run_row_drops_the_scores() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        scored_run(&mut d, "s1", "r1", None, 0.5);
        scored_run(&mut d, "s1", "r2", None, 0.5);
        record_run_doing(&mut d, "s1", "r1", None, "relabel:good");
        assert_eq!(rows_in(&d, &["metrics", "misses"]), 4);
        record_run_doing(&mut d, "s1", "r1", None, "relabel:noise");
        assert_eq!(rows_in(&d, &["metrics", "misses"]), 2);
        assert_eq!(
            d.query("SELECT COUNT(*) AS n FROM metrics WHERE rid = 'r1'")
                .unwrap(),
            "n\n0\n"
        );
        assert_eq!(rows_in(&d, &["runs", "arm_actions"]), 4);
    }

    #[test]
    fn lineage_keeps_revisions_apart_with_their_own_means() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        scored_run(&mut d, "s1", "r1", Some("rev-a"), 0.2);
        scored_run(&mut d, "s1", "r2", Some("rev-b"), 0.8);
        scored_run(&mut d, "s1", "r3", None, 0.4);
        scored_run(&mut d, "s1", "r4", None, 0.6);
        let md = d.lineage().unwrap();
        let rows: Vec<Vec<&str>> = md
            .lines()
            .filter(|l| l.starts_with("| v0 "))
            .map(|l| l.split(" | ").collect())
            .collect();
        // (runs, sorrel_rev, good_units.recall mean); NULL revisions are one group.
        let mut got: Vec<_> = rows.iter().map(|c| (c[2], c[4], c[7])).collect();
        got.sort();
        assert_eq!(
            got,
            [
                ("1", "rev-a", "0.200"),
                ("1", "rev-b", "0.800"),
                ("2", "-", "0.500")
            ],
            "{md}"
        );
    }

    #[test]
    fn lineage_keeps_scoring_options_apart_with_their_own_means() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        scored_run(&mut d, "s1", "r1", None, 0.2);
        scored_run(&mut d, "s1", "r2", None, 0.8);
        // after both records: a later record-run drops scores whose basis moved
        for (rid, opts) in [("r1", "{\"a\":1}"), ("r2", "{\"a\":2}")] {
            d.conn
                .execute(
                    "UPDATE runs SET score_opts = ?2 WHERE rid = ?1",
                    params![rid, opts],
                )
                .unwrap();
        }
        let md = d.lineage().unwrap();
        let mut got: Vec<(&str, &str)> = md
            .lines()
            .filter(|l| l.starts_with("| v0 "))
            .map(|l| l.split(" | ").collect::<Vec<_>>())
            .map(|c| (c[6], c[7]))
            .collect();
        got.sort();
        assert_eq!(
            got,
            [("{\"a\":1}", "0.200"), ("{\"a\":2}", "0.800")],
            "{md}"
        );
    }

    #[test]
    fn a_holdout_result_is_shown_only_on_a_single_configuration() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        scored_run(&mut d, "s1", "r1", None, 0.2);
        let row = LedgerRow {
            candidate: "v0".into(),
            metric: "good_units.recall".into(),
            ..ledger("q1", 0.1)
        };
        d.record_ledger(&[row]).unwrap();
        let hold = |d: &DevDb| -> Vec<String> {
            d.lineage()
                .unwrap()
                .lines()
                .filter(|l| l.starts_with("| v0 "))
                .map(|l| {
                    l.trim_end_matches(" |")
                        .rsplit(" | ")
                        .next()
                        .unwrap()
                        .to_owned()
                })
                .collect()
        };
        assert_eq!(hold(&d), ["+0.100 vs v0 (n=3)"]);
        // a second configuration of the same variant/arm: the ledger can't say which it ran
        scored_run(&mut d, "s1", "r2", Some("rev2"), 0.8);
        assert_eq!(hold(&d), ["see ledger (2 configurations)"; 2]);
    }

    #[test]
    fn an_unknown_truth_verdict_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        assert!(d.record_truth("s1", &[truth(0, "/data/leak", 1)]).is_err());
        assert!(d.record_truth("s1", &[truth(0, "cleanup", 1)]).is_ok());
    }

    fn ledger(query: &str, delta: f64) -> LedgerRow {
        LedgerRow {
            query: query.into(),
            baseline: "v0".into(),
            candidate: "v1".into(),
            arm: "a".into(),
            metric: "m".into(),
            n_sessions: 3,
            base: 0.1,
            cand: 0.1 + delta,
            delta,
            ci_lo: None,
            ci_hi: None,
        }
    }

    #[test]
    fn the_ledger_appends_per_query_and_is_idempotent_within_one() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        d.record_ledger(&[ledger("q1", 0.1)]).unwrap();
        d.record_ledger(&[ledger("q1", 0.1)]).unwrap();
        assert_eq!(rows_in(&d, &["holdout_ledger"]), 1);
        d.record_ledger(&[ledger("q2", 0.3)]).unwrap();
        assert_eq!(
            d.query("SELECT query, delta FROM holdout_ledger ORDER BY query")
                .unwrap(),
            "query\tdelta\nq1\t0.1\nq2\t0.3\n"
        );
    }

    #[test]
    fn a_v1_db_migrates_with_its_ledger_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("r.sqlite");
        {
            let c = Connection::open(&path).unwrap();
            c.execute_batch(SCHEMA_V1).unwrap();
            c.execute_batch(
                "CREATE TABLE schema_version(version INTEGER NOT NULL); INSERT INTO schema_version VALUES (1);
                 INSERT INTO holdout_ledger VALUES ('v0', 'v1', 'a', 'm', 3, 0.1, 0.2, 0.1, NULL, NULL);",
            )
            .unwrap();
        }
        let mut d = DevDb::open(&path).unwrap();
        assert_eq!(
            d.query("SELECT MAX(version) FROM schema_version").unwrap(),
            "MAX(version)\n4\n"
        );
        assert_eq!(
            d.query("SELECT query, baseline, delta FROM holdout_ledger")
                .unwrap(),
            "query\tbaseline\tdelta\nv1\tv0\t0.1\n"
        );
        d.record_ledger(&[ledger("q1", 0.2)]).unwrap();
        assert_eq!(rows_in(&d, &["holdout_ledger"]), 2);
    }

    #[test]
    fn a_v2_db_migrates_keeping_its_rows_and_scores_until_a_record_changes_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("r.sqlite");
        {
            let c = Connection::open(&path).unwrap();
            c.execute_batch(SCHEMA_V1).unwrap();
            c.execute_batch(SCHEMA_V2).unwrap();
            c.execute_batch(
                "CREATE TABLE schema_version(version INTEGER NOT NULL); INSERT INTO schema_version VALUES (1), (2);
                 INSERT INTO sessions VALUES ('s1', 'dev', 'fp1', NULL, NULL, NULL, NULL, NULL, '2026-01-01');
                 INSERT INTO runs VALUES ('r1', 's1', 'v0', 'a', NULL, 0, 'ok', NULL, NULL, NULL, NULL, NULL, NULL, NULL);
                 INSERT INTO metrics VALUES ('r1', 'good_units.recall', 0.5);",
            )
            .unwrap();
        }
        let mut d = DevDb::open(&path).unwrap();
        assert_eq!(
            d.query("SELECT MAX(version) FROM schema_version").unwrap(),
            "MAX(version)\n4\n"
        );
        assert_eq!(rows_in(&d, &["sessions", "runs", "metrics"]), 3);
        assert_eq!(
            d.query("SELECT s.truth_hash IS NULL AS t, r.state_hash IS NULL AS h, r.basis IS NULL AS b FROM sessions s, runs r")
                .unwrap(),
            "t\th\tb\n1\t1\t1\n"
        );
        // a NULL hash is not a known input: the first record of the run re-derives it
        record_plain_run(&mut d, "s1", "r1", None);
        assert_eq!(rows_in(&d, &["runs", "metrics"]), 1);
    }

    #[test]
    fn a_v3_db_migrates_keeping_its_rows_with_an_unknown_output() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("r.sqlite");
        {
            let c = Connection::open(&path).unwrap();
            for s in [SCHEMA_V1, SCHEMA_V2, SCHEMA_V3] {
                c.execute_batch(s).unwrap();
            }
            c.execute_batch(
                "CREATE TABLE schema_version(version INTEGER NOT NULL); INSERT INTO schema_version VALUES (1), (2), (3);
                 INSERT INTO sessions VALUES ('s1', 'dev', 'fp1', NULL, NULL, NULL, NULL, NULL, '2026-01-01', 't1');
                 INSERT INTO runs VALUES ('r1', 's1', 'v0', 'a', NULL, 0, 'ok', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'h', 'b');
                 INSERT INTO metrics VALUES ('r1', 'good_units.recall', 0.5);",
            )
            .unwrap();
        }
        let mut d = DevDb::open(&path).unwrap();
        assert_eq!(
            d.query("SELECT MAX(version) FROM schema_version").unwrap(),
            "MAX(version)\n4\n"
        );
        assert_eq!(rows_in(&d, &["sessions", "runs", "metrics"]), 3);
        assert_eq!(
            d.query("SELECT output_hash IS NULL AS o, score_opts IS NULL AS s, basis FROM runs")
                .unwrap(),
            "o\ts\tbasis\n1\t1\tb\n"
        );
        // unknown output: no score until record-run supplies it
        assert!(d
            .record_diff(
                "r1",
                &score(&SpikeDiffOpts::default()),
                "o1",
                "t1",
                "fp1",
                &HashSet::new()
            )
            .is_err());
    }

    /// The diff of a two-cluster sort against itself under `opts`.
    fn score(opts: &SpikeDiffOpts) -> SpikeDiff {
        let root = [0, 0, 1, 1];
        let labels = [(0, PhyLabel::Good), (1, PhyLabel::Good)];
        let c = Curation {
            clusters: &root,
            labels: &labels,
        };
        diff_spikes(&root, &c, &c, opts).unwrap()
    }

    /// A scored-able run of `s1` whose curated output hashes to `output`.
    fn run_with_output(d: &mut DevDb, output: Option<&str>) {
        let run = RunRow {
            rid: "r1".into(),
            sid: "s1".into(),
            variant: "v0".into(),
            arm: "a".into(),
            status: "ok".into(),
            output_hash: output.map(Into::into),
            ..Default::default()
        };
        d.record_run(&run, &[]).unwrap();
    }

    #[test]
    fn a_score_is_only_recorded_for_the_runs_own_output() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        let (diff, none) = (score(&SpikeDiffOpts::default()), HashSet::new());
        run_with_output(&mut d, None);
        let err = d
            .record_diff("r1", &diff, "o1", "t1", "fp1", &none)
            .unwrap_err();
        assert!(err.to_string().contains("re-run record-run"), "{err}");
        run_with_output(&mut d, Some("o1"));
        let err = d
            .record_diff("r1", &diff, "o2", "t1", "fp1", &none)
            .unwrap_err();
        assert!(
            err.to_string().contains("not the run's recorded output"),
            "{err}"
        );
        assert_eq!(rows_in(&d, &["metrics"]), 0);
        d.record_diff("r1", &diff, "o1", "t1", "fp1", &none)
            .unwrap();
        assert!(rows_in(&d, &["metrics"]) > 0);
        // the run's output changed: the score goes with it
        run_with_output(&mut d, Some("o2"));
        assert_eq!(rows_in(&d, &["metrics"]), 0);
    }

    #[test]
    fn a_score_is_only_recorded_against_the_sessions_truth_and_input() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        run_with_output(&mut d, Some("o1"));
        let (diff, none) = (score(&SpikeDiffOpts::default()), HashSet::new());
        let err = d
            .record_diff("r1", &diff, "o1", "t2", "fp1", &none)
            .unwrap_err();
        assert!(
            err.to_string().contains("TRUTH is not the session's"),
            "{err}"
        );
        let err = d
            .record_diff("r1", &diff, "o1", "t1", "fp2", &none)
            .unwrap_err();
        assert!(
            err.to_string().contains("--root is not the session's"),
            "{err}"
        );
        assert_eq!(rows_in(&d, &["metrics"]), 0);
        d.record_diff("r1", &diff, "o1", "t1", "fp1", &none)
            .unwrap();
        assert!(rows_in(&d, &["metrics"]) > 0);
        // a session whose truth was never recorded cannot vouch for any
        d.conn
            .execute("UPDATE sessions SET truth_hash = NULL", [])
            .unwrap();
        let err = d
            .record_diff("r1", &diff, "o1", "t1", "fp1", &none)
            .unwrap_err();
        assert!(err.to_string().contains("no recorded truth"), "{err}");
    }

    #[test]
    fn the_scoring_options_are_part_of_the_basis() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        run_with_output(&mut d, Some("o1"));
        let mut basis_of = |opts: &SpikeDiffOpts, disputed: &[u32]| {
            let disputed = disputed.iter().copied().collect();
            d.record_diff("r1", &score(opts), "o1", "t1", "fp1", &disputed)
                .unwrap();
            d.query("SELECT basis FROM runs").unwrap()
        };
        let base = basis_of(&SpikeDiffOpts::default(), &[]);
        assert_eq!(base, basis_of(&SpikeDiffOpts::default(), &[]));
        let other = |f: fn(&mut SpikeDiffOpts)| {
            let mut o = SpikeDiffOpts::default();
            f(&mut o);
            o
        };
        let changed = [
            basis_of(&other(|o| o.match_jaccard = 0.5), &[]),
            basis_of(&other(|o| o.min_spikes = 1), &[]),
            basis_of(&other(|o| o.split_share = 0.3), &[]),
            basis_of(&other(|o| o.binary = Some(false)), &[]),
            basis_of(&SpikeDiffOpts::default(), &[7]),
        ];
        for b in &changed {
            assert_ne!(b, &base);
        }
        assert_eq!(changed.iter().collect::<BTreeSet<_>>().len(), changed.len());
    }

    #[test]
    fn exemplars_exclude_the_open_fingerprint_and_holdout() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        fill(&mut d, "s2", "fp2");
        d.record_session(&SessionRow {
            sid: "holdout-1".into(),
            split: "holdout".into(),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(d.exemplars(None).unwrap().len(), 6);
        let ex = d.exemplars(Some("fp1")).unwrap();
        assert!(ex.len() == 3 && ex.iter().all(|e| e.fingerprint == "fp2"));
        assert_eq!(
            (
                ex[1].verdict.as_str(),
                ex[1].num("firing_rate"),
                ex[1].card["flags"].as_str()
            ),
            ("cores", Some(3.0), "+RPV")
        );
    }

    #[test]
    fn distill_uses_dev_sessions_only_and_writes_parseable_files() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = db(dir.path());
        fill(&mut d, "s1", "fp1");
        fill(&mut d, "s2", "fp2");
        d.record_session(&SessionRow {
            sid: "holdout-1".into(),
            split: "holdout".into(),
            ..Default::default()
        })
        .unwrap();
        let out = dir.path().join("learned");
        let rev = d.distill(&out).unwrap();
        let read = |f: &str| std::fs::read_to_string(out.join(f)).unwrap();
        for f in ["exemplars.tsv", "priors.toml", "thresholds.toml"] {
            let t = read(f);
            assert!(
                t.starts_with("# db_sha=")
                    && t.contains("# dev_sids=s1,s2\n")
                    && t.contains(&format!("# learned_rev={rev}\n")),
                "{f}"
            );
        }
        assert!(
            !read("exemplars.tsv").contains("holdout")
                && read("exemplars.tsv")
                    .lines()
                    .filter(|l| !l.starts_with('#'))
                    .count()
                    == 7
        );
        let priors: toml::Value = read("priors.toml").parse().unwrap();
        assert_eq!(priors["cores"]["n"].as_integer(), Some(6));
        assert!(
            priors["features"]["mean"]["firing_rate"]
                .as_float()
                .unwrap()
                > 0.0
        );
        assert!(priors["cleanup"]["coef"]["firing_rate"]
            .as_float()
            .is_some());
        let thr: toml::Value = read("thresholds.toml").parse().unwrap();
        assert_eq!(thr["source"].as_str(), Some("defaults"));
        assert!(thr["flags"].as_table().is_some_and(|t| !t.is_empty()));
        assert!(read("lineage.md").starts_with("<!-- db_sha="));
        // same DB, same rev
        assert_eq!(d.distill(&out).unwrap(), rev);
    }

    #[test]
    fn logistic_regression_separates_a_threshold() {
        let x: Vec<Vec<f64>> = (-10..=10)
            .filter(|&i| i != 0)
            .map(|i| vec![f64::from(i) / 5.0, 0.0])
            .collect();
        let y: Vec<bool> = x.iter().map(|r| r[0] > 0.0).collect();
        let (b, w) = fit_logistic(&x, &y, 0.1);
        let p = |v: f64| 1.0 / (1.0 + (-(b + w[0] * v)).exp());
        assert!(p(1.5) > 0.8 && p(-1.5) < 0.2 && w[1].abs() < 1e-9);
    }

    #[test]
    fn actions_follow_the_applied_path_with_notes_and_llm_verdicts() {
        let node = |id, parent, cmd| Record::Node {
            id,
            parent,
            actor: Actor::Human,
            ts_ms: 0,
            cmd,
        };
        let c = ClusterId;
        let recs = vec![
            node(
                0,
                None,
                CurationCommand::Relabel {
                    cluster: c(3),
                    op: PhyLabelOp::SetGood,
                },
            ),
            node(
                1,
                Some(0),
                CurationCommand::Note {
                    cluster: c(3),
                    text: "clean ACG".into(),
                },
            ),
            node(
                2,
                Some(1),
                CurationCommand::Relabel {
                    cluster: c(4),
                    op: PhyLabelOp::SetNoise,
                },
            ),
            node(
                3,
                Some(1),
                CurationCommand::Split {
                    cluster: c(5),
                    spike_idx: vec![1],
                    new_cluster: c(9),
                    by: Method::Algo("core".into()),
                },
            ),
            Record::Head { node: Some(2) },
        ];
        let ev = serde_json::json!({"units": {"4": {"llm": {"label": "noise", "confidence": 0.8, "reasons": ["rpv", "low snr"]}}}});
        let a = arm_actions(&recs, Some(&ev));
        let got: Vec<_> = a
            .iter()
            .map(|a| {
                (
                    a.target.as_str(),
                    a.action.as_str(),
                    a.confidence,
                    a.reasons.as_deref(),
                )
            })
            .collect();
        assert_eq!(
            got,
            [
                ("3", "relabel:good", None, Some("clean ACG")),
                ("4", "relabel:noise", Some(0.8), Some("rpv; low snr"))
            ]
        );
    }
}

//! `sorrel dev ...` (feature `dev`): the double-blind bench's results DB
//! (`sorrel_data::devdb`, see "Dev mode" in the CLI docs). The DB is
//! `--db`, else `$SORREL_DEV_DB`, else `bench/results.sqlite` of the enclosing
//! repo. Strings the DB must never hold (sealed sids, source basenames) come
//! from the file named by `--deny`, else `$SORREL_DEV_DENY`, one per line.

use super::{phy_label_str, seeded, OpenArgs, OpenCtx, Seed, SessionVisitor};
use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Subcommand};
use serde_json::{json, Value};
use sorrel_data::devdb::{
    arm_actions, truth_hash, DevDb, LedgerRow, RunRow, SessionRow, VariantRow,
};
use sorrel_data::diff::{root_truths_dirs, SpikeDiff, SpikeDiffOpts};
use sorrel_data::session::ApplyPhyLabel;
use sorrel_data::{baseline_hash, Journal, Session};
use sorrel_io::kilosort::PhyLabel;
use sorrel_io::{DataProvider, HasGeometry};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct DevArgs {
    /// Results DB (default: $SORREL_DEV_DB, else bench/results.sqlite of the enclosing repo)
    #[arg(long, global = true, value_name = "FILE")]
    db: Option<PathBuf>,
    /// File of strings the DB must refuse, one per line (default: $SORREL_DEV_DENY)
    #[arg(long, global = true, value_name = "FILE")]
    deny: Option<PathBuf>,
    #[command(subcommand)]
    cmd: DevCmd,
}

#[derive(Subcommand, Debug)]
enum DevCmd {
    /// Create, inspect or query the DB
    Db {
        #[command(subcommand)]
        action: DbCmd,
    },
    /// Record a bench session: its fingerprint and shape, and for a dev session
    /// the root cards of its input and (with --truth) the human verdict per root
    RecordSession {
        /// Opaque session id (a holdout one must not be the sealed id)
        #[arg(long)]
        sid: String,
        #[arg(long, value_parser = ["dev", "holdout"])]
        split: String,
        /// The blind input (dev only)
        #[arg(long, value_name = "DIR")]
        input: Option<PathBuf>,
        /// The human curation (dev only)
        #[arg(long, value_name = "DIR")]
        truth: Option<PathBuf>,
    },
    /// Record a dev run: run.json, evidence.json, claude.json usage and the
    /// journal's actions
    RecordRun {
        #[arg(long, value_name = "DIR")]
        run_dir: PathBuf,
        /// The variant's directory: rubric.md / review.md are stored with it
        #[arg(long, value_name = "DIR")]
        variant_dir: Option<PathBuf>,
    },
    /// Record holdout aggregates (a JSON array of ledger rows, no sids)
    Ledger {
        #[arg(long, value_name = "FILE")]
        json: PathBuf,
    },
    /// Write exemplars.tsv, priors.toml, thresholds.toml and lineage.md from the dev sessions
    Distill {
        #[arg(long, value_name = "DIR")]
        out: PathBuf,
    },
}

#[derive(Subcommand, Debug)]
enum DbCmd {
    /// Create the file and apply migrations
    Init,
    /// Schema version, size and row counts
    Status,
    /// Run read-only SQL; rows as TSV
    Query { sql: String },
}

/// `--db`, else `$SORREL_DEV_DB`, else `bench/results.sqlite` in the nearest
/// ancestor of the cwd that has one, else in the nearest git checkout.
pub fn db_path(flag: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = flag
        .map(Path::to_owned)
        .or_else(|| std::env::var_os("SORREL_DEV_DB").map(Into::into))
    {
        return Ok(p);
    }
    let cwd = std::env::current_dir()?;
    let mut repo = None;
    for d in cwd.ancestors() {
        let p = d.join("bench/results.sqlite");
        if p.exists() {
            return Ok(p);
        }
        if repo.is_none() && d.join(".git").exists() {
            repo = Some(p);
        }
    }
    repo.context("no results DB: pass --db or set SORREL_DEV_DB")
}

fn open_db(db: &Option<PathBuf>, deny: &Option<PathBuf>) -> Result<DevDb> {
    let deny = match deny
        .clone()
        .or_else(|| std::env::var_os("SORREL_DEV_DENY").map(Into::into))
    {
        Some(p) => std::fs::read_to_string(&p)
            .with_context(|| format!("read deny file {}", p.display()))?,
        None => String::new(),
    };
    Ok(DevDb::open(&db_path(db.as_deref())?)?.with_deny(deny.lines().map(str::to_owned)))
}

/// The session fingerprint of an input dir: `baseline_hash` of its `spike_clusters.npy`, in hex.
fn input_fingerprint(input: &Path) -> Result<String> {
    let path = sorrel_io::RootView::open(input)?.path("spike_clusters.npy");
    let hash =
        baseline_hash(&std::fs::read(&path).with_context(|| format!("read {}", path.display()))?);
    Ok(format!("{hash:016x}"))
}

/// `sorrel diff --record RID`: write the score of a recorded run, made against
/// the curations in `test` and `truth` and the input `root`, with `disputed` dropped.
pub fn record_diff(
    db: Option<&Path>,
    rid: &str,
    d: &SpikeDiff,
    (test, truth, root): (&Path, &Path, &Path),
    disputed: &HashSet<u32>,
) -> Result<()> {
    open_db(&db.map(Path::to_owned), &None)?.record_diff(
        rid,
        d,
        &truth_hash(test)?,
        &truth_hash(truth)?,
        &input_fingerprint(root)?,
        disputed,
    )
}

pub fn run(a: DevArgs) -> Result<()> {
    // Every command but `init` and the recorders refuses to create a DB.
    let existing = || -> Result<DevDb> {
        let path = db_path(a.db.as_deref())?;
        ensure!(
            path.exists(),
            "no DB at {}; run `sorrel dev db init`",
            path.display()
        );
        open_db(&a.db, &a.deny)
    };
    match &a.cmd {
        DevCmd::Db {
            action: DbCmd::Init,
        } => print!("{}", open_db(&a.db, &a.deny)?.status()?),
        DevCmd::Db {
            action: DbCmd::Status,
        } => print!("{}", existing()?.status()?),
        DevCmd::Db {
            action: DbCmd::Query { sql },
        } => print!("{}", existing()?.query(sql)?),
        DevCmd::RecordSession {
            sid,
            split,
            input,
            truth,
        } => record_session(
            &mut open_db(&a.db, &a.deny)?,
            sid,
            split,
            input.as_deref(),
            truth.as_deref(),
        )?,
        DevCmd::RecordRun {
            run_dir,
            variant_dir,
        } => {
            let rid = record_run(
                &mut open_db(&a.db, &a.deny)?,
                run_dir,
                variant_dir.as_deref(),
            )?;
            println!("recorded run {rid}");
        }
        DevCmd::Ledger { json } => {
            let text = std::fs::read_to_string(json)
                .with_context(|| format!("read {}", json.display()))?;
            let rows: Vec<LedgerRow> = serde_json::from_str(&text)
                .context("ledger JSON is not an array of ledger rows")?;
            open_db(&a.db, &a.deny)?.record_ledger(&rows)?;
            println!("recorded {} ledger row(s)", rows.len());
        }
        DevCmd::Distill { out } => println!(
            "learned_rev {} -> {}",
            existing()?.distill(out)?,
            out.display()
        ),
    }
    Ok(())
}

fn record_session(
    db: &mut DevDb,
    sid: &str,
    split: &str,
    input: Option<&Path>,
    truth: Option<&Path>,
) -> Result<()> {
    if split == "holdout" {
        ensure!(
            input.is_none() && truth.is_none(),
            "a holdout session takes no --input or --truth"
        );
        return db.record_session(&SessionRow {
            sid: sid.into(),
            split: split.into(),
            ..Default::default()
        });
    }
    let input = input
        .context("a dev session needs --input")?
        .canonicalize()?;
    let clusters_path = sorrel_io::RootView::open(&input)?.path("spike_clusters.npy");
    let fingerprint = input_fingerprint(&input)?;
    let clusters = sorrel_io::npy::read_1d_u32(&clusters_path)?;
    let roots: Vec<u32> = clusters
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    let request = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "brief", "arguments": {"level": 2, "clusters": roots}}})
    .to_string();
    let (mut reply, mut info) = (None, (String::new(), 0));
    {
        // The session opens in a copy beside the input, so no journal or cache lands in it.
        let scratch = Scratch::copy(&input)?;
        let open = OpenArgs {
            root: Some(scratch.0.clone()),
            ..Default::default()
        };
        super::with_session(
            &open,
            Probe {
                request: &request,
                reply: &mut reply,
                info: &mut info,
            },
        )?;
    }
    let result = &reply.context("no reply")?["result"];
    ensure!(
        result["isError"] != true,
        "brief failed: {}",
        result["content"][0]["text"]
    );
    let brief = result["content"][0]["text"]
        .as_str()
        .context("brief returned no text")?;

    let truth_hash = truth.map(truth_hash).transpose()?;
    let truth = truth
        .map(|t| root_truths_dirs(t, &input, &SpikeDiffOpts::default()))
        .transpose()?;
    db.record_session(&SessionRow {
        sid: sid.into(),
        split: split.into(),
        fingerprint: Some(fingerprint),
        sorter: Some(info.0),
        n_spikes: Some(clusters.len() as i64),
        n_roots: Some(roots.len() as i64),
        n_channels: Some(i64::from(info.1)),
        binary: truth.as_ref().map(|t| t.0),
        truth_hash,
    })?;
    let n = db.record_cards(sid, brief)?;
    if let Some((_, rows)) = &truth {
        db.record_truth(sid, rows)?;
    }
    println!(
        "recorded {split} session {sid}: {n} cards, {} truth roots",
        truth.map_or(0, |t| t.1.len())
    );
    Ok(())
}

fn read_json(p: &Path) -> Option<Value> {
    serde_json::from_slice(&std::fs::read(p).ok()?).ok()
}

fn record_run(db: &mut DevDb, dir: &Path, variant_dir: Option<&Path>) -> Result<String> {
    let run = read_json(&dir.join("run.json"))
        .with_context(|| format!("read {}/run.json", dir.display()))?;
    ensure!(run["split"] == "dev", "only dev runs are recorded");
    let claude = read_json(&dir.join("claude.json")).unwrap_or(Value::Null);
    let evidence = read_json(&dir.join("data/sorrel_auto/evidence.json"));
    let text = |k: &str| run[k].as_str().map(str::to_owned);
    let rid = text("id").context("run.json has no id")?;
    let llm_model = evidence
        .iter()
        .flat_map(|e| e["units"].as_object().into_iter().flatten())
        .find_map(|(_, u)| u["llm"]["model"].as_str());
    let model = text("model")
        .or_else(|| {
            claude["modelUsage"]
                .as_object()
                .and_then(|m| m.keys().next().cloned())
        })
        .or_else(|| llm_model.map(str::to_owned));
    let usage = [&run["usage"], &claude["usage"]]
        .into_iter()
        .find(|u| u.is_object())
        .map(Value::to_string);
    let learned_rev = text("learned_rev").or_else(|| {
        evidence
            .as_ref()
            .and_then(|e| e["learned_rev"].as_str().map(str::to_owned))
    });
    let journal = dir.join("data/sorrel.journal");
    let records = if journal.exists() {
        Journal::read_only_records(&journal)?
    } else {
        Vec::new()
    };

    let read_variant = |f: &str| variant_dir.and_then(|d| std::fs::read_to_string(d.join(f)).ok());
    let variant = text("variant").context("run.json has no variant")?;
    db.record_variant(&VariantRow {
        name: variant.clone(),
        prompt_rev: text("prompt_rev"),
        rubric: read_variant("rubric.md"),
        review: read_variant("review.md"),
        sorrel_rev: text("sorrel_rev"),
        learned_rev: learned_rev.clone(),
    })?;
    db.record_run(
        &RunRow {
            rid: rid.clone(),
            sid: text("session").context("run.json has no session")?,
            variant,
            arm: text("arm").context("run.json has no arm")?,
            model,
            repeat: run["repeat"].as_i64().unwrap_or(0),
            status: text("status").unwrap_or_else(|| "unknown".into()),
            cost_usd: run["cost_usd"]
                .as_f64()
                .or_else(|| claude["total_cost_usd"].as_f64()),
            turns: run["num_turns"]
                .as_i64()
                .or_else(|| claude["num_turns"].as_i64()),
            wall_s: run["wall_s"].as_f64(),
            usage,
            prompt_rev: text("prompt_rev"),
            sorrel_rev: text("sorrel_rev"),
            learned_rev,
            // The curated output `score` diffs as TEST; none (no data dir) is unknown.
            output_hash: truth_hash(&dir.join("data")).ok(),
        },
        &arm_actions(&records, evidence.as_ref()),
    )?;
    Ok(rid)
}

/// A copy of a blind input next to it (same filesystem, so big arrays are
/// hard-linked), removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn copy(input: &Path) -> Result<Self> {
        let dir = input.with_file_name(format!(".dev-scratch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let s = Self(dir);
        copy_tree(input, &s.0)?;
        Ok(s)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Files and directories of `src` except sorrel's own journal and cache.
fn copy_tree(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let name = e.file_name();
        let (from, to) = (e.path(), dst.join(&name));
        if name == ".sorrel" || name.to_string_lossy().starts_with("sorrel.journal") {
            continue;
        }
        match from.is_dir() {
            true => copy_tree(&from, &to)?,
            false => sorrel_data::blind::link_or_copy(&from, &to)?,
        }
    }
    Ok(())
}

/// One tool call on the dataset, plus what the open session knows about it.
struct Probe<'a> {
    request: &'a str,
    reply: &'a mut Option<Value>,
    /// (sorter, channels)
    info: &'a mut (String, u32),
}

impl SessionVisitor for Probe<'_> {
    fn visit<P>(self, session: Session<P>, seed: Seed<P>, ctx: OpenCtx<P>) -> Result<()>
    where
        P: DataProvider<Label = PhyLabel> + ApplyPhyLabel + HasGeometry,
    {
        *self.info = (session.sorter().to_owned(), session.provider.n_channels());
        let host = sorrel_data::SessionHost::new(seeded(session, seed)?);
        let mut env = sorrel_data::Env::new(&ctx.save_dir, ctx.reseal_on_save, phy_label_str);
        *self.reply = sorrel_data::handle_line(&host, &mut env, self.request);
        if self.reply.is_none() {
            bail!("the brief call got no reply");
        }
        Ok(())
    }
}

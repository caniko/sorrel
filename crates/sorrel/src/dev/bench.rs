//! Native benchmark orchestration. Private paths, truths and holdout runs never
//! enter the results database. No external Python runner is used.

mod coverage;
mod learn;
mod process;
mod reporting;
mod runner;

use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sorrel_data::learn::features::{ExtractData, Truth};
use sorrel_data::learn::shard::{InputFile, InputManifest};
use sorrel_io::kilosort::read_cluster_group_tsv;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Private bench directory, outside any git checkout
    #[arg(long, global = true)]
    bench: Option<PathBuf>,
    /// Commit-pinned sorrel binary for the arms (default: this executable)
    #[arg(long, global = true)]
    sorrel: Option<PathBuf>,
    /// Commit-pinned scorer, unchanged across a comparison
    #[arg(long, global = true)]
    scorer: Option<PathBuf>,
    #[command(subcommand)]
    cmd: BenchCmd,
}

/// Arguments of `learn-dataset`, filled from the clap variant below.
#[derive(Debug)]
pub struct LearnDatasetArgs {
    /// Output directory (tables and `shards/`; must not be inside a session)
    pub out: PathBuf,
    /// Extract at most this many dev sessions (deterministic first-N cohort)
    pub dev_limit: Option<usize>,
    /// Concurrent sessions (1-8)
    pub jobs: usize,
    /// Upper bound on `spikes` rows per root
    pub max_spikes_per_root: usize,
    /// Extraction seed (recorded; the extractor itself is deterministic)
    pub seed: u64,
    /// Recompute only truth targets from stored features; no proposers, no
    /// feature work, and the raw recording is not opened
    pub retarget: bool,
}

#[derive(Subcommand, Debug)]
enum BenchCmd {
    /// Find phy folders by names/existence only; never read human labels
    Discover {
        root: PathBuf,
        /// Only integer-ID and integer-IDtodo immediate subdirectories
        #[arg(long)]
        integer_ids: bool,
        #[arg(long)]
        json: bool,
    },
    /// Blind and seal reviewed sorts; originals are never opened for editing
    Add {
        #[arg(required_unless_present = "discover")]
        sessions: Vec<PathBuf>,
        /// Discover all phy folders under these roots and ingest them
        #[arg(long)]
        discover: Vec<PathBuf>,
        #[arg(long)]
        integer_ids: bool,
        /// Truth has already been seen; it must never reach holdout
        #[arg(long)]
        dev: bool,
        #[arg(long)]
        phy_curated: bool,
        /// A second human curation of a recording already in dev: ingest it
        /// as an extra dev session instead of skipping it as a duplicate
        #[arg(long, requires = "dev")]
        recuration: bool,
    },
    /// Run one arm on fresh copies; OpenCode gets only the sorrel MCP tools
    Run {
        #[arg(long, value_enum)]
        arm: Arm,
        #[arg(long)]
        variant: PathBuf,
        #[arg(long, value_enum)]
        split: Split,
        #[arg(long, default_value_t = 3)]
        repeats: u32,
        /// Concurrent sessions (1-8); repeats of one session remain sequential
        #[arg(long, default_value_t = 1)]
        jobs: usize,
        #[arg(long, value_enum, default_value = "opencode")]
        backend: process::Backend,
        #[arg(long)]
        model: Option<String>,
        /// Limit a plumbing check to this many dev sessions (never applies to holdout)
        #[arg(long)]
        dev_limit: Option<usize>,
        /// First-machine MCP check: require at least four model turns (dev only)
        #[arg(long)]
        plumbing_check: bool,
        /// Commit of the pinned arm binary (required for reproducibility)
        #[arg(long)]
        sorrel_rev: String,
        /// Per-run cost cap for the mcp-review arm (default 10 USD). The
        /// auto-llm arm cannot enforce a cap and refuses one; omit it there
        #[arg(long)]
        max_budget_usd: Option<f64>,
        #[arg(long, default_value_t = 3600)]
        timeout: u64,
    },
    /// Score every successful run with one fixed scorer
    Score {
        #[arg(long, value_enum)]
        split: Split,
        #[arg(long = "match", default_value_t = 0.8)]
        match_jaccard: f64,
    },
    /// Dev misses and cards, or a single ledgered aggregate holdout query
    Report {
        #[arg(value_enum)]
        split: Split,
        #[arg(long)]
        baseline: Option<String>,
        #[arg(long)]
        candidate: Option<String>,
        /// Match the deterministic first-N dev cohort used by run; never holdout
        #[arg(long)]
        dev_limit: Option<usize>,
    },
    /// Inspect aggregate status and cost without revealing holdout identities
    Status,
    /// Lead-only DEV diagnostic: oracle overlap of input-only robust boundaries
    Coverage {
        #[arg(long, default_value_t = 6)]
        dev_limit: usize,
        #[arg(long, value_enum, default_value = "robust")]
        method: CoverageMethod,
    },
    /// DEV only: extract the learned curator's feature/target tables
    LearnDataset {
        /// Output directory (tables and `shards/`; must not be inside a session)
        #[arg(long, value_name = "DIR")]
        out: PathBuf,
        /// Extract at most this many dev sessions (deterministic first-N cohort)
        #[arg(long)]
        dev_limit: Option<usize>,
        /// Concurrent sessions (1-8)
        #[arg(long, default_value_t = 1)]
        jobs: usize,
        /// Upper bound on `spikes` rows per root
        #[arg(long, default_value_t = 1500)]
        max_spikes_per_root: usize,
        /// Extraction seed (recorded; the extractor itself is deterministic)
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Recompute only truth targets from stored features (no proposers)
        #[arg(long)]
        retarget: bool,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum Split {
    Dev,
    Holdout,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum CoverageMethod {
    Robust,
    Density,
    FeatureOracle,
    Axis,
}
impl CoverageMethod {
    fn name(self) -> &'static str {
        match self {
            Self::Robust => "robust",
            Self::Density => "density",
            Self::FeatureOracle => "feature-oracle",
            Self::Axis => "axis",
        }
    }
}

impl Split {
    fn name(self) -> &'static str {
        match self {
            Self::Dev => "dev",
            Self::Holdout => "holdout",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum Arm {
    AutoLlm,
    McpReview,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::AutoLlm => "auto-llm",
            Self::McpReview => "mcp-review",
        }
    }
}

#[derive(Debug, Serialize)]
struct Candidate {
    path: PathBuf,
    has_labels: bool,
    custom_labels: Vec<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct SessionMap {
    salt: String,
    sessions: BTreeMap<String, SessionEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SessionEntry {
    path: PathBuf,
    split: Split,
    forced_dev: bool,
    pure_manual: bool,
    keep: Vec<String>,
    /// Input-only hash: duplicate recordings cannot straddle dev and holdout.
    #[serde(default)]
    fingerprint: String,
    /// The dev session whose recording this is a second curation of; group
    /// both together in cross-validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    recuration_of: Option<String>,
}

struct Bench {
    root: PathBuf,
    sorrel: PathBuf,
    scorer: PathBuf,
    db: Option<PathBuf>,
    map: SessionMap,
    // One writer per bench, including across Windows processes.
    _lock: fs::File,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn opaque() -> String {
    rand::random::<[u8; 16]>()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn holdout_alias(salt: &str, sid: &str) -> String {
    format!(
        "h{}",
        &blake3::hash(format!("{salt}{sid}").as_bytes()).to_hex()[..8]
    )
}

fn read_json(p: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(p)?)?)
}

fn write_json(p: &Path, value: &impl Serialize) -> Result<()> {
    let parent = p.parent().context("output has no parent")?;
    fs::create_dir_all(parent)?;
    let mut tmp = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut tmp, value)?;
    tmp.as_file().sync_all()?;
    tmp.persist(p).map_err(|e| e.error)?;
    Ok(())
}

fn valid_id(name: &str) -> bool {
    let n = name.strip_suffix("todo").unwrap_or(name);
    !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())
}

fn is_link(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}

/// Canonical path of `path`, resolving existing ancestors even when the leaf
/// does not exist yet. Nothing is created.
pub(super) fn canonical_target(path: &Path) -> Result<PathBuf> {
    let path = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut missing = Vec::new();
    let mut current = path.as_path();
    loop {
        if current.exists() {
            let mut resolved = current.canonicalize()?;
            for part in missing.iter().rev() {
                resolved.push(part);
            }
            return Ok(normalize(resolved));
        }
        let name = current
            .file_name()
            .with_context(|| format!("{} has no existing ancestor", path.display()))?;
        missing.push(name.to_owned());
        current = current
            .parent()
            .with_context(|| format!("{} has no existing ancestor", path.display()))?;
    }
}

/// Lexically resolve `.` and `..` in an already-canonical prefix. Missing tail
/// components cannot be symlinks, so removing `..` there cannot change which
/// filesystem directory a later write reaches.
fn normalize(path: PathBuf) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn discover(root: &Path, integer_ids: bool) -> Result<Vec<Candidate>> {
    fn visit(dir: &Path, out: &mut Vec<Candidate>) -> Result<()> {
        if dir.join("params.py").is_file() && dir.join("spike_templates.npy").is_file() {
            let known = [
                "cluster_group.tsv",
                "cluster_info.tsv",
                "cluster_KSLabel.tsv",
                "cluster_Amplitude.tsv",
                "cluster_ContamPct.tsv",
            ];
            let mut custom_labels = Vec::new();
            for e in fs::read_dir(dir)? {
                let name = e?.file_name().to_string_lossy().into_owned();
                if name.starts_with("cluster_")
                    && name.ends_with(".tsv")
                    && !known.contains(&name.as_str())
                {
                    custom_labels.push(name);
                }
            }
            custom_labels.sort();
            out.push(Candidate {
                path: dir.to_owned(),
                has_labels: dir.join("cluster_group.tsv").is_file(),
                custom_labels,
            });
        }
        for entry in fs::read_dir(dir).with_context(|| format!("enumerate {}", dir.display()))? {
            let e = entry?;
            let m = fs::symlink_metadata(e.path())?;
            if m.is_dir() && !is_link(&m) && e.file_name() != ".sealed" && e.file_name() != ".phy" {
                visit(&e.path(), out)?;
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    ensure!(root.is_dir(), "discovery root is not a directory");
    if integer_ids {
        for entry in fs::read_dir(root)? {
            let e = entry?;
            let m = fs::symlink_metadata(e.path())?;
            if m.is_dir() && !is_link(&m) && valid_id(&e.file_name().to_string_lossy()) {
                visit(&e.path(), &mut out)?;
            }
        }
    } else {
        visit(root, &mut out)?;
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

impl Bench {
    fn open(a: &BenchArgs, db: Option<&Path>) -> Result<Self> {
        let root = a.bench.as_ref().context("pass --bench PRIVATE_DIR")?;
        fs::create_dir_all(root)?;
        let root = root.canonicalize()?;
        ensure!(
            !root.ancestors().any(|p| p.join(".git").exists()),
            "BENCH must be outside a git checkout"
        );
        fs::create_dir_all(root.join(".sealed"))?;
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(root.join(".sealed/bench.lock"))?;
        lock.try_lock()
            .context("bench is busy (another CLI process owns it)")?;
        let map_path = root.join(".sealed/map.json");
        let map = if map_path.exists() {
            serde_json::from_slice(&fs::read(map_path)?)?
        } else {
            SessionMap {
                salt: opaque(),
                ..Default::default()
            }
        };
        let sorrel = a
            .sorrel
            .clone()
            .map(Ok)
            .unwrap_or_else(std::env::current_exe)?;
        let scorer = a.scorer.clone().unwrap_or_else(|| sorrel.clone());
        let db = db
            .map(Path::to_owned)
            .or_else(|| std::env::var_os("SORREL_DEV_DB").map(Into::into));
        Ok(Self {
            root,
            sorrel,
            scorer,
            db,
            map,
            _lock: lock,
        })
    }

    fn home(&self, split: Split) -> PathBuf {
        if split == Split::Holdout {
            self.root.join(".sealed")
        } else {
            self.root.clone()
        }
    }
    fn session(&self, sid: &str) -> PathBuf {
        self.home(self.map.sessions[sid].split)
            .join("sessions")
            .join(sid)
    }
    fn save_map(&self) -> Result<()> {
        write_json(&self.root.join(".sealed/map.json"), &self.map)
    }

    fn command(&self, scorer: bool) -> Command {
        Command::new(if scorer { &self.scorer } else { &self.sorrel })
    }

    fn db_command(&self) -> Result<Command> {
        let deny = self.root.join(".sealed/dev-deny.txt");
        let mut names = BTreeSet::new();
        for (sid, s) in &self.map.sessions {
            if s.split == Split::Holdout {
                names.insert(sid.clone());
            }
            if let Some(n) = s.path.file_name() {
                names.insert(n.to_string_lossy().into_owned());
            }
        }
        let text = names
            .into_iter()
            .map(|n| format!("{n}\n"))
            .collect::<String>();
        // run_arm prepares this once before spawning workers. Never truncate a
        // guard file while a concurrent DB subprocess could be reading it.
        if fs::read_to_string(&deny).ok().as_deref() != Some(text.as_str()) {
            fs::write(&deny, text)?;
        }
        // DB recording belongs to the runner, not to an older arm binary.
        let mut c = Command::new(std::env::current_exe()?);
        c.env("SORREL_DEV_DB", self.db.as_ref().context("no results DB")?)
            .env("SORREL_DEV_DENY", deny)
            .arg("dev");
        Ok(c)
    }

    fn record_session(&self, sid: &str) -> Result<()> {
        if self.db.is_none() {
            return Ok(());
        }
        let s = &self.map.sessions[sid];
        let mut c = self.db_command()?;
        c.args(["record-session", "--split", s.split.name(), "--sid"]);
        if s.split == Split::Dev {
            c.arg(sid)
                .arg("--input")
                .arg(self.session(sid).join("input"))
                .arg("--truth")
                .arg(self.session(sid).join("truth"));
        } else {
            c.arg(holdout_alias(&self.map.salt, sid));
        }
        process::checked(&mut c, Duration::from_secs(1800))?;
        Ok(())
    }

    fn copy_input(&self, sid: &str, to: &Path) -> Result<()> {
        fs::create_dir(to)?;
        let input = self.session(sid).join("input");
        for name in &self.map.sessions[sid].keep {
            ensure!(
                Path::new(name).components().count() == 1 && !sorrel_data::blind::is_denied(name),
                "invalid blind allowlist"
            );
            let from = input.join(name);
            ensure!(
                from.is_file() && !is_link(&fs::symlink_metadata(&from)?),
                "blind file is missing or a link"
            );
            sorrel_data::blind::link_or_copy(&from, &to.join(name))?;
        }
        Ok(())
    }

    /// The dev session a re-curation with this input `fingerprint` and
    /// `truth` belongs to, or `None` when a session already holds this exact
    /// truth. A recording with any holdout session is refused: its second
    /// curation would put the sealed truth's recording in dev.
    fn recuration_target(&self, fingerprint: &str, truth: &Path) -> Result<Option<String>> {
        let same: Vec<_> = self
            .map
            .sessions
            .iter()
            .filter(|(_, s)| s.fingerprint == fingerprint)
            .collect();
        ensure!(
            !same.is_empty(),
            "--recuration: this recording is not in the bench yet; add it without --recuration"
        );
        ensure!(
            same.iter().all(|(_, s)| s.split == Split::Dev),
            "--recuration: this recording is in holdout; promote it with `add --dev` first"
        );
        // Either truth file may be absent (phy saved no labels or no splits).
        let read = |dir: &Path| {
            ["spike_clusters.npy", "cluster_group.tsv"].map(|name| fs::read(dir.join(name)).ok())
        };
        let incoming = read(truth);
        for (sid, _) in &same {
            if read(&self.session(sid).join("truth")) == incoming {
                return Ok(None);
            }
        }
        let original = same
            .iter()
            .find(|(_, s)| s.recuration_of.is_none())
            .unwrap_or(&same[0]);
        Ok(Some(original.0.clone()))
    }

    fn add(
        &mut self,
        paths: Vec<PathBuf>,
        force_dev: bool,
        phy: bool,
        recuration: bool,
    ) -> Result<()> {
        ensure!(phy, "native intake currently requires --phy-curated");
        let mut added = 0;
        let mut skipped = 0;
        for path in paths {
            let path = path.canonicalize()?;
            // A re-curation may reuse a known path; the fingerprint/truth checks decide.
            if let Some((sid, entry)) = self
                .map
                .sessions
                .iter()
                .find(|(_, s)| !recuration && s.path == path)
            {
                let sid = sid.clone();
                if force_dev && entry.split == Split::Holdout {
                    self.promote(&sid)?;
                } else {
                    skipped += 1;
                }
                continue;
            }
            let incoming = tempfile::Builder::new()
                .prefix("incoming-")
                .tempdir_in(self.root.join(".sealed"))?;
            let input = incoming.path().join("input");
            let truth = incoming.path().join("truth");
            // No session host is opened on a source: this API reads originals only.
            let keep = sorrel_data::blind::write_phy_curated_pair(&path, None, &input, &truth)?;
            let provenance = read_json(&truth.join("provenance.json"))?;
            if provenance["curated"] == false {
                skipped += 1;
                continue;
            }
            let params = sorrel_io::PhyParams::read(&input.join("params.py"))?;
            let raw = params
                .dat_path
                .unwrap_or_else(|| PathBuf::from("recording.dat"));
            ensure!(
                raw.components().count() == 1 && input.join(&raw).is_file(),
                "blind input has no self-contained raw recording; intake stopped"
            );
            let mut check = self.command(false);
            check.arg("describe").arg(&input);
            process::checked(&mut check, Duration::from_secs(1800))
                .context("new-session describe preflight failed")?;
            let pure = provenance["pure_manual"] == true && provenance["anchored"] != true;
            let fingerprint = input_hash(&input)?;
            let mut recuration_of = None;
            if recuration {
                match self.recuration_target(&fingerprint, &truth)? {
                    Some(sid) => recuration_of = Some(sid),
                    None => {
                        skipped += 1;
                        continue;
                    }
                }
            } else if let Some((sid, entry)) = self
                .map
                .sessions
                .iter()
                .find(|(_, s)| s.fingerprint == fingerprint)
            {
                let sid = sid.clone();
                if force_dev && entry.split == Split::Holdout {
                    self.promote(&sid)?;
                }
                skipped += 1;
                continue;
            }
            let hash = blake3::hash(format!("{}{}", self.map.salt, path.display()).as_bytes());
            let split = if pure
                && !force_dev
                && u32::from_le_bytes(hash.as_bytes()[..4].try_into()?) % 100 < 30
            {
                Split::Holdout
            } else {
                Split::Dev
            };
            let sid = opaque();
            let to = self.home(split).join("sessions").join(&sid);
            fs::create_dir_all(to.parent().context("session parent")?)?;
            fs::rename(incoming.path(), &to)?;
            self.map.sessions.insert(
                sid.clone(),
                SessionEntry {
                    path,
                    split,
                    forced_dev: force_dev,
                    pure_manual: pure,
                    keep,
                    fingerprint,
                    recuration_of,
                },
            );
            self.save_map()?;
            self.record_session(&sid)?;
            added += 1;
        }
        println!(
            "added {added}; skipped {skipped}; dev {}; holdout sealed",
            self.map
                .sessions
                .values()
                .filter(|s| s.split == Split::Dev)
                .count()
        );
        Ok(())
    }

    fn promote(&mut self, sid: &str) -> Result<()> {
        let old = self.session(sid);
        let to = self.root.join("sessions").join(sid);
        fs::create_dir_all(to.parent().context("session parent")?)?;
        fs::rename(old, to)?;
        let old_disputed = self
            .home(Split::Holdout)
            .join("disputed")
            .join(format!("{sid}.tsv"));
        if old_disputed.is_file() {
            fs::create_dir_all(self.root.join("disputed"))?;
            fs::rename(
                old_disputed,
                self.root.join("disputed").join(format!("{sid}.tsv")),
            )?;
        }
        for run in self.run_dirs(Split::Holdout)? {
            if read_json(&run.join("run.json"))?["session"] == sid {
                fs::remove_dir_all(run)?;
            }
        }
        let entry = self.map.sessions.get_mut(sid).context("unknown session")?;
        entry.split = Split::Dev;
        entry.forced_dev = true;
        self.save_map()?;
        self.record_session(sid)
    }

    fn run_dirs(&self, split: Split) -> Result<Vec<PathBuf>> {
        let dir = self.home(split).join("runs");
        if !dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut dirs = Vec::new();
        for e in fs::read_dir(dir)? {
            let p = e?.path();
            if p.join("run.json").is_file() {
                dirs.push(p);
            }
        }
        dirs.sort();
        Ok(dirs)
    }
}

fn input_hash(input: &Path) -> Result<String> {
    let mut h = blake3::Hasher::new();
    let mut buf = vec![0; 1 << 20];
    for name in ["spike_times.npy", "spike_templates.npy", "templates.npy"] {
        h.update(name.as_bytes());
        let mut f = fs::File::open(input.join(name))?;
        h.update(&f.metadata()?.len().to_le_bytes());
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
    }
    Ok(h.finalize().to_hex().to_string())
}

/// Read the truth in `truth_dir`, align it through the stored row map and hash
/// it. Shared by `learn-dataset` and `learn predict`; the caller decides
/// whether a truth directory is present at all.
pub(super) fn load_truth(truth_dir: &Path, data: &ExtractData) -> Result<(Truth, String, f64)> {
    let started = std::time::Instant::now();
    let truth_rows = sorrel_io::npy::read_1d_u32(&truth_dir.join("spike_clusters.npy"))?;
    ensure!(
        truth_rows.len() == data.root_of_spike.len(),
        "{}: truth has {} spikes, session {}",
        truth_dir.display(),
        truth_rows.len(),
        data.root_of_spike.len()
    );
    // The row map and session roots were validated against the input roots at
    // extraction time; rebuild the original-order roots from them so a
    // retarget never has to open the input.
    let mut input_roots = vec![u32::MAX; truth_rows.len()];
    for (global, row) in data.row_map.iter().enumerate() {
        let row = row.context("truth needs a complete original NPY-row map")? as usize;
        ensure!(
            row < input_roots.len() && input_roots[row] == u32::MAX,
            "{}: stored NPY-row map is not a permutation",
            truth_dir.display()
        );
        input_roots[row] = data.session_roots[global];
    }
    let cluster = coverage::aligned_truth(
        &data.row_map,
        &truth_rows,
        &input_roots,
        &data.session_roots,
    )?;
    let labels = read_cluster_group_tsv(&truth_dir.join("cluster_group.tsv"))
        .context("dev truth labels missing or unreadable")?;
    let hash = sorrel_data::devdb::truth_hash(truth_dir)?;
    Ok((
        Truth { cluster, labels },
        hash,
        started.elapsed().as_secs_f64(),
    ))
}

/// Files at or below this size are hashed in full for a feature-key manifest.
const MANIFEST_FULL_MAX: u64 = 8 << 20;
/// Large files contribute their size plus a digest of this many bytes at each
/// end. See [`InputManifest`] for why.
const MANIFEST_EDGE: u64 = 1 << 20;

/// Content identity of the files a feature extraction will consume, for the
/// feature key.
///
/// `input_hash` above stays exactly as it was: it is the legacy key that
/// feature shards from earlier builds recorded, and a new build must still be
/// able to recognise (and warn about) those shards. The manifest is the
/// superset that also covers amplitudes, PC features, `params.py`, probe
/// geometry, templates, similarity and the recording length. `names` is the
/// bench's blind allowlist, exactly what `copy_input` places in the temporary
/// input the extractor opens (the sealed session also holds sorrel journals,
/// which are not consumed).
fn input_manifest(input: &Path, names: &[String]) -> Result<InputManifest> {
    let mut files = Vec::with_capacity(names.len());
    for name in names {
        ensure!(
            Path::new(name).components().count() == 1,
            "input allowlist entry {name:?} is not a plain file name"
        );
        let path = input.join(name);
        let meta = fs::symlink_metadata(&path)
            .with_context(|| format!("session input file {}", path.display()))?;
        ensure!(
            meta.file_type().is_file() && !is_link(&meta),
            "session input {} is not a regular file; the manifest cannot identify it",
            path.display()
        );
        files.push(input_file(path, name.clone(), meta.len())?);
    }
    files.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(InputManifest { files })
}

fn input_file(path: PathBuf, name: String, size: u64) -> Result<InputFile> {
    let mut h = blake3::Hasher::new();
    let sampled = size > MANIFEST_FULL_MAX;
    if sampled {
        // Size plus both edges, never mtime: a bench copy legitimately gets a
        // fresh mtime while keeping the same bytes.
        h.update(&size.to_le_bytes());
        let mut f = fs::File::open(&path)?;
        let mut buf = vec![0; MANIFEST_EDGE as usize];
        f.read_exact(&mut buf)?;
        h.update(&buf);
        f.seek(SeekFrom::Start(size - MANIFEST_EDGE))?;
        f.read_exact(&mut buf)?;
        h.update(&buf);
    } else {
        let mut f = fs::File::open(&path)?;
        let mut buf = vec![0; 1 << 20];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
    }
    Ok(InputFile {
        name,
        size,
        sampled,
        digest: h.finalize().to_hex().to_string(),
    })
}

pub(super) fn run(a: &BenchArgs, db: Option<&Path>) -> Result<()> {
    if let BenchCmd::Discover {
        root,
        integer_ids,
        json,
    } = &a.cmd
    {
        let candidates = discover(root, *integer_ids)?;
        if *json {
            println!("{}", serde_json::to_string_pretty(&candidates)?);
        } else {
            println!("path\thas_labels\tcustom_labels");
            for c in &candidates {
                println!(
                    "{}\t{}\t{}",
                    c.path.display(),
                    c.has_labels,
                    c.custom_labels.join(",")
                );
            }
            println!("{} candidate(s)", candidates.len());
        }
        return Ok(());
    }
    let mut b = Bench::open(a, db)?;
    match &a.cmd {
        BenchCmd::Discover { .. } => unreachable!(),
        BenchCmd::Add {
            sessions,
            discover: roots,
            integer_ids,
            dev,
            phy_curated,
            recuration,
        } => {
            let mut paths = sessions.clone();
            for root in roots {
                paths.extend(discover(root, *integer_ids)?.into_iter().map(|c| c.path));
            }
            b.add(paths, *dev, *phy_curated, *recuration)?;
        }
        BenchCmd::Status => {
            for split in [Split::Dev, Split::Holdout] {
                let mut count = BTreeMap::new();
                let mut cost = 0.0;
                let mut unknown_cost = 0;
                for p in b.run_dirs(split)? {
                    let r = read_json(&p.join("run.json"))?;
                    *count
                        .entry(r["status"].as_str().unwrap_or("unknown").to_owned())
                        .or_insert(0) += 1;
                    if let Some(value) = r["cost_usd"].as_f64() {
                        cost += value;
                    } else {
                        unknown_cost += 1;
                    }
                }
                println!("{}: {} sessions, runs {}, recorded cost ${cost:.4}, unknown cost on {unknown_cost} runs", split.name(), b.map.sessions.values().filter(|s| s.split == split).count(), json!(count));
            }
        }
        BenchCmd::Run { .. } => b.run_arm(&a.cmd)?,
        BenchCmd::Coverage { dev_limit, method } => b.coverage(*dev_limit, *method)?,
        BenchCmd::LearnDataset {
            out,
            dev_limit,
            jobs,
            max_spikes_per_root,
            seed,
            retarget,
        } => b.learn_dataset(&LearnDatasetArgs {
            out: out.clone(),
            dev_limit: *dev_limit,
            jobs: *jobs,
            max_spikes_per_root: *max_spikes_per_root,
            seed: *seed,
            retarget: *retarget,
        })?,
        BenchCmd::Score {
            split,
            match_jaccard,
        } => b.score(*split, *match_jaccard)?,
        BenchCmd::Report {
            split,
            baseline,
            candidate,
            dev_limit,
        } => b.report(
            *split,
            baseline.as_deref(),
            candidate.as_deref(),
            *dev_limit,
        )?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_does_not_read_truth_and_honors_ids() {
        let t = tempfile::tempdir().unwrap();
        for name in ["12", "34todo", "KS4", "12todo-extra"] {
            let p = t.path().join(name).join("deep/sorter_output");
            fs::create_dir_all(&p).unwrap();
            // Invalid content proves discovery only checks existence.
            for f in ["params.py", "spike_templates.npy", "cluster_group.tsv"] {
                fs::write(p.join(f), "unreadable as data").unwrap();
            }
        }
        let found = discover(t.path(), true).unwrap();
        assert_eq!(found.len(), 2);
        assert!(found.iter().all(|c| c.has_labels));
        assert_eq!(discover(t.path(), false).unwrap().len(), 4);
    }

    #[test]
    fn bench_lock_and_atomic_map() {
        let t = tempfile::tempdir().unwrap();
        let args = BenchArgs {
            bench: Some(t.path().to_owned()),
            sorrel: None,
            scorer: None,
            cmd: BenchCmd::Status,
        };
        let b = Bench::open(&args, None).unwrap();
        b.save_map().unwrap();
        assert!(Bench::open(&args, None).is_err());
        assert!(read_json(&t.path().join(".sealed/map.json")).unwrap()["sessions"].is_object());
        drop(b);
        assert!(Bench::open(&args, None).is_ok());
    }

    #[test]
    fn recuration_joins_its_dev_recording_unless_the_truth_is_already_there() {
        let t = tempfile::tempdir().unwrap();
        let args = BenchArgs {
            bench: Some(t.path().to_owned()),
            sorrel: None,
            scorer: None,
            cmd: BenchCmd::Status,
        };
        let mut b = Bench::open(&args, None).unwrap();
        let truth = |dir: &Path, clusters: &[u8]| {
            fs::create_dir_all(dir).unwrap();
            fs::write(dir.join("spike_clusters.npy"), clusters).unwrap();
            fs::write(dir.join("cluster_group.tsv"), b"cluster_id\tgroup\n").unwrap();
        };
        let entry = |split, fingerprint: &str, recuration_of: Option<&str>| SessionEntry {
            path: PathBuf::new(),
            split,
            forced_dev: false,
            pure_manual: true,
            keep: Vec::new(),
            fingerprint: fingerprint.into(),
            recuration_of: recuration_of.map(Into::into),
        };
        b.map
            .sessions
            .insert("orig".into(), entry(Split::Dev, "rec", None));
        truth(&b.session("orig").join("truth"), b"phy");
        let incoming = t.path().join("incoming");

        truth(&incoming, b"phy");
        assert_eq!(b.recuration_target("rec", &incoming).unwrap(), None);
        truth(&incoming, b"sorrel");
        assert_eq!(
            b.recuration_target("rec", &incoming).unwrap().as_deref(),
            Some("orig")
        );
        b.map
            .sessions
            .insert("again".into(), entry(Split::Dev, "rec", Some("orig")));
        truth(&b.session("again").join("truth"), b"sorrel");
        assert_eq!(b.recuration_target("rec", &incoming).unwrap(), None);

        assert!(b.recuration_target("unknown", &incoming).is_err());
        b.map
            .sessions
            .insert("sealed".into(), entry(Split::Holdout, "held", None));
        assert!(b.recuration_target("held", &incoming).is_err());
    }

    #[test]
    fn holdout_alias_is_db_compatible_and_unlinkable() {
        let alias = holdout_alias("salt-a", "sealed-session-id");
        assert_eq!(alias.len(), 9);
        assert!(alias.starts_with('h'));
        assert!(alias[1..].bytes().all(|b| b.is_ascii_hexdigit()));
        assert_ne!(alias, holdout_alias("salt-b", "sealed-session-id"));
        assert!(!alias.contains("sealed-session-id"));
    }

    #[test]
    fn fingerprint_ignores_human_curation() {
        let t = tempfile::tempdir().unwrap();
        for file in ["spike_times.npy", "spike_templates.npy", "templates.npy"] {
            fs::write(t.path().join(file), file.as_bytes()).unwrap();
        }
        let before = input_hash(t.path()).unwrap();
        fs::write(t.path().join("spike_clusters.npy"), b"human split").unwrap();
        fs::write(t.path().join("cluster_group.tsv"), b"human labels").unwrap();
        assert_eq!(before, input_hash(t.path()).unwrap());
        fs::write(t.path().join("spike_times.npy"), b"another recording").unwrap();
        assert_ne!(before, input_hash(t.path()).unwrap());
    }

    #[test]
    fn input_manifest_covers_files_beyond_the_legacy_fingerprint() {
        let t = tempfile::tempdir().unwrap();
        let files = [
            "spike_times.npy",
            "spike_templates.npy",
            "templates.npy",
            "amplitudes.npy",
            "pc_features.npy",
            "params.py",
            "channel_positions.npy",
            "similar_templates.npy",
        ];
        for file in files {
            fs::write(t.path().join(file), file.as_bytes()).unwrap();
        }
        let names: Vec<String> = files.iter().map(|name| (*name).to_owned()).collect();
        let before = input_manifest(t.path(), &names).unwrap();
        assert_eq!(before.files.len(), 8);
        assert!(before.files.iter().all(|f| !f.sampled));
        // Files the blind allowlist does not copy are not consumed and must
        // not enter the key (sealed inputs also hold sorrel's own journals).
        fs::write(t.path().join("sorrel.journal"), b"not an input").unwrap();
        assert_eq!(before, input_manifest(t.path(), &names).unwrap());
        let legacy_before = input_hash(t.path()).unwrap();
        // An amplitude edit leaves the legacy three-NPY key untouched but
        // must change the manifest that the new feature key records.
        fs::write(t.path().join("amplitudes.npy"), b"new amplitudes").unwrap();
        assert_eq!(legacy_before, input_hash(t.path()).unwrap());
        assert_ne!(before, input_manifest(t.path(), &names).unwrap());
        fs::write(t.path().join("amplitudes.npy"), b"amplitudes.npy").unwrap();
        assert_eq!(before, input_manifest(t.path(), &names).unwrap());
        fs::write(t.path().join("params.py"), b"sample_rate = 1").unwrap();
        assert_ne!(before, input_manifest(t.path(), &names).unwrap());
    }

    #[test]
    fn large_inputs_use_size_and_both_edge_digests() {
        let t = tempfile::tempdir().unwrap();
        let path = t.path().join("recording.dat");
        let names = vec!["recording.dat".to_owned()];
        let mut bytes = vec![0u8; (MANIFEST_FULL_MAX + 1024) as usize];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = (i % 251) as u8;
        }
        fs::write(&path, &bytes).unwrap();
        let before = input_manifest(t.path(), &names).unwrap();
        assert!(before.files[0].sampled);

        bytes[0] ^= 0xFF;
        fs::write(&path, &bytes).unwrap();
        assert_ne!(
            before,
            input_manifest(t.path(), &names).unwrap(),
            "first edge"
        );
        bytes[0] ^= 0xFF;

        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        fs::write(&path, &bytes).unwrap();
        assert_ne!(
            before,
            input_manifest(t.path(), &names).unwrap(),
            "last edge"
        );
        bytes[last] ^= 0xFF;

        // The documented heuristic: a middle-only edit and an unchanged
        // length are not detected. This keeps resumes cheap.
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0xFF;
        fs::write(&path, &bytes).unwrap();
        assert_eq!(before, input_manifest(t.path(), &names).unwrap());

        fs::write(&path, &bytes[..bytes.len() - 8]).unwrap();
        assert_ne!(
            before,
            input_manifest(t.path(), &names).unwrap(),
            "truncation"
        );
    }

    #[test]
    fn even_a_refused_holdout_query_is_ledgered() {
        let t = tempfile::tempdir().unwrap();
        let args = BenchArgs {
            bench: Some(t.path().to_owned()),
            sorrel: None,
            scorer: None,
            cmd: BenchCmd::Status,
        };
        let b = Bench::open(&args, None).unwrap();
        assert!(b
            .report(Split::Holdout, Some("baseline"), Some("candidate"), None)
            .is_err());
        let ledger = fs::read_to_string(t.path().join(".sealed/holdout-ledger.jsonl")).unwrap();
        assert_eq!(ledger.lines().count(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&ledger).unwrap()["baseline"],
            "baseline"
        );
    }
}

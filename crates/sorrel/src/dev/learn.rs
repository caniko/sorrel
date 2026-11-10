//! `sorrel dev learn cv|train|predict`: grouped cross-validation, model
//! fitting on an extracted `learn-dataset` directory, and scoring of one phy
//! folder with a stored head.
//!
//! Only the plumbing lives here. Feature selection, fold assignment, metrics
//! and model training are in `sorrel_data::learn`, so the same code can be
//! called from a test, an MCP tool or a future GUI without a CLI in between.

use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Subcommand};
use serde::Serialize;
use serde_json::json;
use sorrel_data::learn::bundle::{CuratorBundle, FeatureSchema, Head, Provenance};
use sorrel_data::learn::cv::{
    cross_validate, fit_full, objective_name, parse_objective, read_groups, row_groups, CvOptions,
    CvReport, FoldMetrics, Problem,
};
use sorrel_data::learn::features::{self, ExtractOpts};
use sorrel_data::learn::gbdt::Objective;
use sorrel_data::learn::metrics::{classification, root_ranking, RootRanking};
use sorrel_data::learn::table::Table;
use sorrel_data::learn::{EXTRACTOR_VERSION, TARGET_VERSION};
use sorrel_data::{Journal, Session};
use sorrel_io::kilosort::KilosortProvider;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Subcommand, Debug)]
pub enum LearnCmd {
    /// Grouped K-fold cross-validation of one table and target
    Cv(CvArgs),
    /// Fit on every row and write/update a `.sorrel-model` bundle head
    Train(TrainArgs),
    /// Score one phy folder's roots with a stored head; the folder is not written
    Predict(PredictArgs),
    /// Curate one phy folder with a verdict head and a spike head, into `--out`
    Curate(CurateArgs),
}

#[derive(Args, Debug)]
pub struct CvArgs {
    /// Directory produced by `dev bench learn-dataset`
    #[arg(long, value_name = "DIR")]
    data: PathBuf,
    /// Table name: roots, candidates, pairs, units or spikes
    #[arg(long, value_name = "NAME")]
    table: String,
    /// Target column, e.g. y_verdict (5 documented classes: use softmax:5)
    #[arg(long, value_name = "COLUMN")]
    target: String,
    /// logistic, softmax:K or regression
    #[arg(long, value_name = "OBJECTIVE", value_parser = objective_parser)]
    objective: String,
    /// Merge one integer class into another: OLD=NEW (repeatable, one step)
    #[arg(long = "class-map", value_name = "OLD=NEW")]
    class_map: Vec<String>,
    /// JSON `{"groups": {sid: group}}`; default: one group per sid
    #[arg(long, value_name = "FILE")]
    groups: Option<PathBuf>,
    #[arg(long, default_value_t = 5)]
    folds: usize,
    /// Drop every feature column starting with this prefix (repeatable)
    #[arg(long = "exclude-prefix", value_name = "PREFIX")]
    exclude_prefix: Vec<String>,
    /// Append the roots table's f_* columns (from `--data`) as root_*
    #[arg(long)]
    join_roots: bool,
    /// Keep only rows whose root has one of these true `y_verdict` names
    /// (comma-separated, e.g. kept,cleanup,cores); needs the roots table
    #[arg(long = "root-verdict", value_name = "NAMES", value_delimiter = ',')]
    root_verdict: Vec<String>,
    /// Keep every Nth usable row: a uniform subsample of a large table
    #[arg(long = "row-stride", value_name = "N", default_value_t = 1)]
    row_stride: usize,
}

#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Directory produced by `dev bench learn-dataset`
    #[arg(long, value_name = "DIR")]
    data: PathBuf,
    #[arg(long, value_name = "NAME")]
    table: String,
    #[arg(long, value_name = "COLUMN")]
    target: String,
    #[arg(long, value_name = "OBJECTIVE", value_parser = objective_parser)]
    objective: String,
    /// Merge one integer class into another: OLD=NEW (repeatable, one step)
    #[arg(long = "class-map", value_name = "OLD=NEW")]
    class_map: Vec<String>,
    #[arg(long, value_name = "FILE")]
    groups: Option<PathBuf>,
    /// Folds used for the CV metrics stored in the head
    #[arg(long, default_value_t = 5)]
    folds: usize,
    #[arg(long = "exclude-prefix", value_name = "PREFIX")]
    exclude_prefix: Vec<String>,
    #[arg(long)]
    join_roots: bool,
    /// Keep only rows whose root has one of these true `y_verdict` names
    /// (comma-separated, e.g. kept,cleanup,cores); needs the roots table
    #[arg(long = "root-verdict", value_name = "NAMES", value_delimiter = ',')]
    root_verdict: Vec<String>,
    /// Keep every Nth usable row: a uniform subsample of a large table
    #[arg(long = "row-stride", value_name = "N", default_value_t = 1)]
    row_stride: usize,
    /// Destination `.sorrel-model` (created, or extended with `--head`)
    #[arg(long, value_name = "FILE")]
    out: PathBuf,
    /// Head key in the bundle; an existing key is replaced
    #[arg(long, value_name = "NAME")]
    head: String,
}

#[derive(Args, Debug)]
pub struct PredictArgs {
    /// phy/Kilosort folder (with params.py); its input and curation are only read
    #[arg(value_name = "PHY_DIR")]
    phy_dir: PathBuf,
    /// `.sorrel-model` bundle holding the head
    #[arg(long, value_name = "FILE")]
    model: PathBuf,
    /// Head key; default: the bundle's only head
    #[arg(long, value_name = "KEY")]
    head: Option<String>,
    /// Output directory (predictions.tsv and report.json; must be outside PHY_DIR)
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// Upper bound on `spikes` rows per root used by extraction
    #[arg(long, default_value_t = 1500)]
    max_spikes_per_root: usize,
    /// Extraction seed (recorded; the extractor itself is deterministic)
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

#[derive(Args, Debug)]
pub struct CurateArgs {
    /// phy/Kilosort folder (with params.py); its input and curation are only read
    #[arg(value_name = "PHY_DIR")]
    phy_dir: PathBuf,
    /// `.sorrel-model` bundle holding both heads
    #[arg(long, value_name = "FILE")]
    model: PathBuf,
    /// Head that labels each root (softmax with a `noise` class)
    #[arg(
        long = "verdict-head",
        value_name = "KEY",
        default_value = "roots_verdict"
    )]
    verdict_head: String,
    /// Logistic head that scores each spike of a kept root (1 = keep)
    #[arg(long = "spike-head", value_name = "KEY", default_value = "spikes_keep")]
    spike_head: String,
    /// A spike stays in its unit when the spike head gives at least this
    #[arg(long, value_name = "P", default_value_t = 0.5)]
    threshold: f64,
    /// Logistic head on `candidates` (1 = the membership is a human good
    /// unit). A kept root with two or more disjoint candidates at
    /// `--split-threshold` becomes those units; without it roots never split
    #[arg(long = "split-head", value_name = "KEY")]
    split_head: Option<String>,
    #[arg(long = "split-threshold", value_name = "P", default_value_t = 0.5)]
    split_threshold: f64,
    /// A unit with fewer spikes than this after cleanup is noise
    #[arg(long = "min-spikes", value_name = "N", default_value_t = 300)]
    min_spikes: usize,
    /// Exception to `--min-spikes`: a unit that was silent for this fraction
    /// of the recording and then fired at least 100 spikes is kept
    #[arg(long = "late-onset", value_name = "FRACTION", default_value_t = 0.5)]
    late_onset: f64,
    /// A unit whose ACG centre bar (lags within 0.5 ms, per ms) is above
    /// this fraction of its 20-50 ms baseline is noise; the centre must be
    /// empty or barely there
    #[arg(long = "max-centre", value_name = "RATIO", default_value_t = 0.1)]
    max_centre: f64,
    /// Passes over the folder. Each pass after the first reopens it with the
    /// previous pass's clusters and curates what that pass set aside (dropped
    /// spikes, split remainders), which is where a root's second unit is.
    /// Experimental: the heads were not trained on leftovers, and on day13
    /// a second pass called nearly every leftover pile good
    #[arg(long, value_name = "N", default_value_t = 1, hide = true)]
    rounds: usize,
    /// Output directory (spike_clusters.npy, cluster_group.tsv, roots.tsv,
    /// report.json; must be outside PHY_DIR)
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// Size of the per-root reference subsample; use the value the spike
    /// head's training data was extracted with
    #[arg(long, default_value_t = 1500)]
    max_spikes_per_root: usize,
}

fn objective_parser(text: &str) -> std::result::Result<String, String> {
    parse_objective(text).map_err(|e| e.to_string())?;
    Ok(text.to_owned())
}

/// Where one selected feature comes from.
enum Source {
    /// A column of the main table.
    Base(usize),
    /// A column of the roots table, joined by `(sid_idx, root)`.
    Root(usize),
}

/// Joined root features and the `(sid_idx, root)` row index into them.
type RootJoin = (BTreeMap<(u32, u32), usize>, Table);

/// Everything selected from one table, kept only for rows with a usable target.
#[derive(Debug)]
struct Selection {
    names: Vec<String>,
    /// Class-index -> name for classification targets (empty for regression).
    classes: Vec<String>,
    x: Vec<f32>,
    y: Vec<f32>,
    groups: Vec<String>,
    sids: Vec<String>,
    roots: Vec<u32>,
    hit: Vec<bool>,
    /// Truth unit id per row (`u32::MAX` when the table has none).
    unit: Vec<u32>,
    /// Session index per row.
    sid_idx: Vec<u32>,
}

/// Load `data/<table>`, select features (`f_*` plus optional joined root
/// features) and the target, and drop rows without a usable label.
///
/// Only targets that are truly undefined stay non-finite and are dropped here:
/// `y_verdict`, `y_n_good` and `y_good_share` for roots below the bench's
/// ten-spike floor (the dev DB's `root_truths` omits them), and
/// `y_best_unit`, which is an id rather than a target and is NaN when a
/// candidate touches no good unit. Every overlap target is 0, never NaN, when
/// the intersection is empty.
fn select(data: &Path, a: TableArgs<'_>) -> Result<Selection> {
    let table = Table::load(data, a.table)?;
    ensure!(
        table.extractor_version == EXTRACTOR_VERSION,
        "table was extracted with {:?}, this binary expects {EXTRACTOR_VERSION}; re-extract",
        table.extractor_version
    );
    if table.columns.iter().any(|c| c.starts_with("y_")) {
        ensure!(
            table.target_version.as_deref() == Some(TARGET_VERSION),
            "table carries targets from {:?}, this binary expects {TARGET_VERSION}; re-extract",
            table.target_version
        );
    }
    ensure!(
        table.column(a.target).is_some(),
        "table {:?} has no target column {:?}",
        a.table,
        a.target
    );
    let mut names: Vec<String> = table
        .columns
        .iter()
        .filter(|c| c.starts_with("f_") && !a.exclude.iter().any(|p| c.starts_with(p)))
        .cloned()
        .collect();
    if names.is_empty() {
        bail!("no feature columns selected; relax --exclude-prefix");
    }

    let mut sources: Vec<Source> = names
        .iter()
        .map(|name| Source::Base(table.column(name).expect("selected feature exists")))
        .collect();

    ensure!(a.row_stride > 0, "--row-stride must be positive");
    // The roots table, keyed by (sid_idx, root): optional joined features
    // and the optional --root-verdict row filter.
    let joined: Option<RootJoin> = if a.join_roots || !a.root_verdict.is_empty() {
        ensure!(
            table.column("root").is_some(),
            "table {:?} has no root column to join the roots table on",
            a.table
        );
        let roots = Table::load(data, "roots")
            .context("--join-roots and --root-verdict need the roots table")?;
        ensure!(
            roots.extractor_version == EXTRACTOR_VERSION,
            "roots table has a different extractor version"
        );
        if roots.columns.iter().any(|c| c.starts_with("y_")) {
            ensure!(
                roots.target_version.as_deref() == Some(TARGET_VERSION),
                "roots table has targets from {:?}, expected {TARGET_VERSION}",
                roots.target_version
            );
        }
        let root_sid: Vec<Option<usize>> = roots
            .sids
            .iter()
            .map(|s| table.sids.iter().position(|x| x == s))
            .collect();
        let mut index = BTreeMap::new();
        for row in 0..roots.n_rows() {
            let sid = roots.get(row, "sid_idx");
            if !sid.is_finite() {
                continue;
            }
            let Some(Some(sid)) = root_sid.get(sid as usize) else {
                continue;
            };
            let root = roots.get(row, "root");
            if root.is_finite() {
                index.insert((*sid as u32, root as u32), row);
            }
        }
        let mut any = !a.join_roots;
        let mut column = 0;
        while column < roots.columns.len() {
            let name = &roots.columns[column];
            if a.join_roots
                && name.starts_with("f_")
                && !a.exclude.iter().any(|p| name.starts_with(p))
            {
                // The roots table is a different table: build its column list
                // first, then push the paired sources in the same order.
                names.push(format!("root_{name}"));
                sources.push(Source::Root(column));
                any = true;
            }
            column += 1;
        }
        ensure!(any, "roots table has no f_* columns to join");
        Some((index, roots))
    } else {
        None
    };

    // --root-verdict: the y_verdict codes a row's root may have.
    let verdicts: Option<(usize, BTreeSet<i64>)> = match &joined {
        Some((_, roots)) if !a.root_verdict.is_empty() => {
            let column = roots
                .column("y_verdict")
                .context("--root-verdict: the roots table has no y_verdict")?;
            let codes = roots
                .codes
                .get("y_verdict")
                .context("--root-verdict: the roots table documents no y_verdict codes")?;
            let wanted = a
                .root_verdict
                .iter()
                .map(|name| {
                    codes.get(name).copied().with_context(|| {
                        format!(
                            "--root-verdict {name:?} is not a y_verdict name ({})",
                            codes.keys().cloned().collect::<Vec<_>>().join(", ")
                        )
                    })
                })
                .collect::<Result<_>>()?;
            Some((column, wanted))
        }
        _ => None,
    };

    let n_features = names.len();
    let y_column = table.column(a.target).expect("checked above");
    let objective = parse_objective(a.objective).expect("clap validated the objective");
    let class_map = parse_class_map(a.class_map)?;
    ensure!(
        !matches!(objective, sorrel_data::learn::gbdt::Objective::SquaredError)
            || class_map.is_empty(),
        "--class-map only applies to logistic or softmax objectives"
    );
    let n_classes = match objective {
        sorrel_data::learn::gbdt::Objective::Softmax { n_classes } => n_classes,
        sorrel_data::learn::gbdt::Objective::Logistic => 2,
        sorrel_data::learn::gbdt::Objective::SquaredError => 0,
    };
    // The header documents the target's class codes. Checking the requested
    // K against them catches the silent-drop bug before any row is skipped:
    // e.g. `softmax:4` on the five-class `y_verdict`, whose `other` is 4.
    if !matches!(objective, sorrel_data::learn::gbdt::Objective::SquaredError) {
        check_documented_classes(&table, a.target, n_classes, &class_map)?;
    }
    // Names travel with the head so predict can label its probability columns.
    let classes = match objective {
        Objective::SquaredError => Vec::new(),
        _ => class_names(&table, a.target, n_classes, &class_map),
    };
    let has_hit = table.column("y_hit");
    let has_unit = table.column("y_best_unit");
    let has_root = table.column("root");

    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut sids = Vec::new();
    let mut roots = Vec::new();
    let mut hit = Vec::new();
    let mut unit = Vec::new();
    let mut sid_indices = Vec::new();
    let mut usable = 0usize;
    let mut outside: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
    let mut fractional: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for row in 0..table.n_rows() {
        let raw = table.value(row, y_column);
        if !raw.is_finite() {
            continue;
        }
        let label = match objective {
            sorrel_data::learn::gbdt::Objective::SquaredError => raw as f64,
            _ => {
                let c = raw as i64;
                if (raw - c as f32).abs() > 1e-4 {
                    *fractional.entry(format!("{raw}")).or_default() += 1;
                    continue;
                }
                let mapped = class_map.get(&c).copied().unwrap_or(c);
                if mapped < 0 || (mapped as usize) >= n_classes {
                    *outside.entry(c).or_default() += 1;
                    continue;
                }
                mapped as f64
            }
        };
        let sid = table.get(row, "sid_idx");
        if !sid.is_finite() || (sid as usize) >= table.sids.len() {
            continue;
        }
        let sid_idx = sid as usize;
        let root = has_root.map_or(0u32, |_| {
            let v = table.get(row, "root");
            if v.is_finite() {
                v as u32
            } else {
                u32::MAX
            }
        });
        if let (Some((column, wanted)), Some((index, roots_table))) = (&verdicts, &joined) {
            let verdict = index
                .get(&(sid_idx as u32, root))
                .map_or(f32::NAN, |&r| roots_table.value(r, *column));
            if !verdict.is_finite() || !wanted.contains(&(verdict as i64)) {
                continue;
            }
        }
        usable += 1;
        if !(usable - 1).is_multiple_of(a.row_stride) {
            continue;
        }
        for source in &sources {
            let v = match source {
                Source::Base(column) => table.value(row, *column),
                Source::Root(column) => match &joined {
                    Some((index, roots_table)) if root != u32::MAX => index
                        .get(&(sid_idx as u32, root))
                        .map_or(f32::NAN, |&r| roots_table.value(r, *column)),
                    _ => f32::NAN,
                },
            };
            x.push(v);
        }
        y.push(label as f32);
        sids.push(table.sids[sid_idx].clone());
        roots.push(root);
        hit.push(has_hit.is_some_and(|_| table.get(row, "y_hit") >= 0.5));
        sid_indices.push(sid_idx as u32);
        unit.push(has_unit.map_or(u32::MAX, |_| {
            let v = table.get(row, "y_best_unit");
            if v.is_finite() {
                v as u32
            } else {
                u32::MAX
            }
        }));
    }
    if !fractional.is_empty() {
        let detail = fractional
            .iter()
            .map(|(value, count)| format!("{value} ({count} row(s))"))
            .collect::<Vec<_>>()
            .join(", ");
        bail!(
            "{:?} of table {:?} has non-integer value(s) {detail}; classification targets must \
             be whole class codes",
            a.target,
            a.table
        );
    }
    if !outside.is_empty() {
        let total: usize = outside.values().sum();
        let detail = outside
            .iter()
            .map(|(value, count)| format!("{value} ({count} row(s))"))
            .collect::<Vec<_>>()
            .join(", ");
        let documented = table
            .codes
            .get(a.target)
            .and_then(|codes| codes.values().max())
            .map(|max| max + 1);
        let hint = match objective {
            sorrel_data::learn::gbdt::Objective::Logistic => documented.map_or_else(
                || {
                    "logistic needs labels 0 or 1; use regression, another objective, or \
                     --class-map OLD=NEW to map the extra classes"
                        .to_owned()
                },
                |classes| {
                    format!(
                        "{} has {classes} classes (codes in the table header); logistic needs \
                         binary 0/1 labels; use softmax:{classes} or --class-map OLD=NEW",
                        a.target
                    )
                },
            ),
            _ => documented.map_or_else(
                || {
                    format!(
                        "use a wider objective (softmax:{}) or --class-map OLD=NEW",
                        outside
                            .keys()
                            .next_back()
                            .map_or(2, |max| (*max + 1).max(2))
                    )
                },
                |classes| {
                    format!(
                        "{} has {classes} classes (codes in the table header); use softmax:{classes}",
                        a.target
                    )
                },
            ),
        };
        bail!(
            "{total} row(s) of table {:?} have {} = {detail}, outside {} (classes 0..{}); \
             {hint}",
            a.table,
            a.target,
            objective_name(objective),
            n_classes.saturating_sub(1)
        );
    }
    ensure!(
        !y.is_empty(),
        "no rows of {:?} have a finite {:?}",
        a.table,
        a.target
    );
    let by_sid = a.groups.map(read_groups).transpose()?.unwrap_or_default();
    let groups = row_groups(&sids, &by_sid);
    ensure!(
        x.len() == y.len() * n_features,
        "feature matrix shape drifted during selection"
    );
    Ok(Selection {
        names,
        classes,
        x,
        y,
        groups,
        sids,
        roots,
        hit,
        unit,
        sid_idx: sid_indices,
    })
}

/// Parse repeatable `--class-map OLD=NEW`. Application is one step, never
/// transitive, so a target that is also a source is refused as ambiguous.
fn parse_class_map(items: &[String]) -> Result<BTreeMap<i64, i64>> {
    let mut out = BTreeMap::new();
    for item in items {
        let (from, to) = item
            .split_once('=')
            .with_context(|| format!("--class-map expects OLD=NEW, not {item:?}"))?;
        let from: i64 = from
            .trim()
            .parse()
            .with_context(|| format!("--class-map {item:?}: OLD is not an integer"))?;
        let to: i64 = to
            .trim()
            .parse()
            .with_context(|| format!("--class-map {item:?}: NEW is not an integer"))?;
        ensure!(
            from >= 0 && to >= 0,
            "--class-map {item:?}: class codes must be non-negative"
        );
        ensure!(from != to, "--class-map {item:?} maps a class to itself");
        ensure!(
            out.insert(from, to).is_none(),
            "--class-map repeats class {from}"
        );
    }
    for (&from, &to) in &out {
        ensure!(
            !out.contains_key(&to),
            "--class-map is one step ({} -> {to} and {to} -> ... cannot chain); \
             map both classes to the same final class instead",
            from
        );
    }
    Ok(out)
}

/// Check the requested class count against the codes documented for `target`
/// in the table header. Empty code maps skip the check (the row scan still
/// refuses values outside `[0, K)`).
fn check_documented_classes(
    table: &Table,
    target: &str,
    n_classes: usize,
    class_map: &BTreeMap<i64, i64>,
) -> Result<()> {
    let Some(codes) = table.codes.get(target) else {
        return Ok(());
    };
    let mut documented = BTreeSet::new();
    for (code, name) in codes.iter().map(|(name, &code)| (code, name)) {
        ensure!(
            code >= 0,
            "{target} documents {name:?} = {code}; class codes must be non-negative"
        );
        let mapped = class_map.get(&code).copied().unwrap_or(code);
        ensure!(
            mapped >= 0,
            "--class-map maps {target} = {code} to the negative class {mapped}"
        );
        documented.insert(mapped);
    }
    let expected = documented
        .iter()
        .next_back()
        .map_or(0, |&max| max as usize + 1);
    if expected != n_classes {
        let objective = if n_classes == 2 && class_map.is_empty() {
            format!(
                "logistic needs binary 0/1 labels; use softmax:{expected} or --class-map OLD=NEW"
            )
        } else if class_map.is_empty() {
            format!("use softmax:{expected}")
        } else {
            format!(
                "--class-map leaves {expected} classes; use softmax:{expected} (or adjust the map)"
            )
        };
        bail!("{target} has {expected} classes (codes in the table header); {objective}");
    }
    Ok(())
}

/// Class-index -> name, from the target's documented codes after `--class-map`.
/// Codes with no documented name (or several names mapping to one class) fall
/// back to `class_<k>`, so a stored head always names every probability column.
fn class_names(
    table: &Table,
    target: &str,
    n_classes: usize,
    class_map: &BTreeMap<i64, i64>,
) -> Vec<String> {
    let mut names = vec![String::new(); n_classes];
    if let Some(codes) = table.codes.get(target) {
        for (name, &code) in codes {
            let mapped = class_map.get(&code).copied().unwrap_or(code);
            if mapped >= 0 && (mapped as usize) < n_classes {
                let slot = &mut names[mapped as usize];
                if slot.is_empty() {
                    *slot = name.clone();
                }
            }
        }
    }
    for (k, name) in names.iter_mut().enumerate() {
        if name.is_empty() {
            *name = format!("class_{k}");
        }
    }
    names
}

/// Arguments shared by `cv` and `train`.
struct TableArgs<'a> {
    table: &'a str,
    target: &'a str,
    objective: &'a str,
    class_map: &'a [String],
    exclude: &'a [String],
    join_roots: bool,
    root_verdict: &'a [String],
    row_stride: usize,
    groups: Option<&'a Path>,
}

#[derive(Serialize)]
struct CvCliReport {
    #[serde(flatten)]
    cv: CvReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    ranking: Option<RootRanking>,
}

pub fn run(cmd: &LearnCmd) -> Result<()> {
    match cmd {
        LearnCmd::Cv(a) => cv(a),
        LearnCmd::Train(a) => train(a),
        LearnCmd::Predict(a) => predict(a),
        LearnCmd::Curate(a) => curate(a),
    }
}

fn cv(a: &CvArgs) -> Result<()> {
    let objective = parse_objective(&a.objective)?;
    let sel = select(
        &a.data,
        TableArgs {
            table: &a.table,
            target: &a.target,
            objective: &a.objective,
            class_map: &a.class_map,
            exclude: &a.exclude_prefix,
            join_roots: a.join_roots,
            root_verdict: &a.root_verdict,
            row_stride: a.row_stride,
            groups: a.groups.as_deref(),
        },
    )?;
    let folds = a
        .folds
        .min(sel.groups.iter().collect::<BTreeSet<_>>().len());
    ensure!(
        folds >= 2,
        "{} group(s) cannot make a cross-validation; pass a groups file or more sessions",
        folds
    );
    let problem = Problem {
        x: &sel.x,
        n_features: sel.names.len(),
        columns: &sel.names,
        y: &sel.y,
        groups: &sel.groups,
    };
    let output = cross_validate(
        &problem,
        &CvOptions {
            folds,
            seed: 0,
            params: None,
        },
        objective,
    )?;
    let ranking = ranking(&a.table, &a.target, objective, &sel, &output.predictions);
    let report = CvCliReport {
        cv: output.report,
        ranking,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn train(a: &TrainArgs) -> Result<()> {
    let objective = parse_objective(&a.objective)?;
    let sel = select(
        &a.data,
        TableArgs {
            table: &a.table,
            target: &a.target,
            objective: &a.objective,
            class_map: &a.class_map,
            exclude: &a.exclude_prefix,
            join_roots: a.join_roots,
            root_verdict: &a.root_verdict,
            row_stride: a.row_stride,
            groups: a.groups.as_deref(),
        },
    )?;
    let problem = Problem {
        x: &sel.x,
        n_features: sel.names.len(),
        columns: &sel.names,
        y: &sel.y,
        groups: &sel.groups,
    };

    // CV metrics first, so a head is never written without its held-out score.
    let distinct = sel.groups.iter().collect::<BTreeSet<_>>().len();
    ensure!(
        distinct >= 2,
        "training needs at least two groups for the stored CV metrics"
    );
    let folds = a.folds.clamp(2, distinct);
    let output = cross_validate(
        &problem,
        &CvOptions {
            folds,
            seed: 0,
            params: None,
        },
        objective,
    )?;
    let model = fit_full(&problem, 0, objective)?;

    let schema = FeatureSchema::new(sel.names.clone(), EXTRACTOR_VERSION);
    // Key by content hash, not by table/target: two heads trained on the same
    // columns share one schema, and a retrained head with different columns
    // adds a new entry instead of overwriting a schema another head uses.
    let schema_key = schema.key();
    let mut head = Head::new(schema_key.clone(), model).with_classes(sel.classes.clone());
    for (key, value) in head_metrics(&output.report) {
        head = head.with_metric(key, value);
    }
    head = head.with_metric("cv_n_rows", output.report.n_rows as f64);
    head = head.with_metric("cv_folds", output.report.folds as f64);

    let mut bundle = if a.out.is_file() {
        let b = CuratorBundle::load(&a.out)?;
        ensure!(
            b.provenance.extractor_version == EXTRACTOR_VERSION,
            "bundle was built with extractor {:?}, this binary uses {EXTRACTOR_VERSION}",
            b.provenance.extractor_version
        );
        b
    } else {
        CuratorBundle::new(Provenance::now(
            env!("CARGO_PKG_VERSION"),
            sorrel_rev(),
            EXTRACTOR_VERSION,
        ))
    };
    let mut sessions: BTreeSet<String> = bundle
        .provenance
        .training_sessions
        .iter()
        .cloned()
        .collect();
    sessions.extend(sel.sids.iter().cloned());
    bundle.provenance.training_sessions = sessions.into_iter().collect();
    bundle.provenance.sorrel_version = env!("CARGO_PKG_VERSION").to_owned();
    bundle.provenance.sorrel_rev = sorrel_rev();
    bundle.provenance.notes =
        format!(
        "learn-dataset {EXTRACTOR_VERSION}; last head {}/{}, objective {}, {} rows, {} features{}",
        a.table,
        a.target,
        objective_name(objective),
        sel.y.len(),
        sel.names.len(),
        if a.join_roots { ", joined root features" } else { "" }
    );
    bundle
        .provenance
        .n_rows
        .insert(a.head.clone(), sel.y.len() as u64);
    bundle.add_schema_checked(schema_key.clone(), schema)?;
    bundle.add_head(a.head.clone(), head);
    bundle.validate()?;
    bundle.save(&a.out)?;

    let ranking = ranking(&a.table, &a.target, objective, &sel, &output.predictions);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "out": a.out.display().to_string(),
            "head": a.head,
            "schema": schema_key,
            "n_rows": sel.y.len(),
            "n_features": sel.names.len(),
            "folds": folds,
            "metrics": bundle.heads[&a.head].metrics,
            "ranking": ranking,
        }))?
    );
    Ok(())
}

#[derive(Serialize)]
struct ConfusionReport {
    classes: Vec<String>,
    matrix: Vec<Vec<usize>>,
}

#[derive(Serialize)]
struct ClassScore {
    class: String,
    support: usize,
    predicted: usize,
    precision: f64,
    recall: f64,
    f1: f64,
}

#[derive(Serialize)]
struct PredictReport {
    model: String,
    head: String,
    n_roots: usize,
    truth: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_truth: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    accuracy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    confusion: Option<ConfusionReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    per_class: Option<Vec<ClassScore>>,
}

/// The head's classes in class-index order: stored names when the bundle has
/// them, else positional names so every probability column still has a header.
fn head_class_names(key: &str, head: &Head) -> Result<Vec<String>> {
    match head.model.objective {
        Objective::Softmax { n_classes } => {
            if head.classes.is_empty() {
                Ok((0..n_classes).map(|k| format!("class_{k}")).collect())
            } else {
                ensure!(
                    head.classes.len() == n_classes,
                    "head {key:?} names {} classes but its softmax has {n_classes}",
                    head.classes.len()
                );
                Ok(head.classes.clone())
            }
        }
        Objective::Logistic => Ok(if head.classes.is_empty() {
            vec!["neg".to_owned(), "pos".to_owned()]
        } else {
            head.classes.clone()
        }),
        Objective::SquaredError => {
            bail!("head {key:?} is a regression head; predict needs a classification head")
        }
    }
}

fn load_bundle(path: &Path) -> Result<CuratorBundle> {
    let bundle = CuratorBundle::load(path)?;
    ensure!(
        bundle.provenance.extractor_version == EXTRACTOR_VERSION,
        "bundle was built with extractor {:?}, this binary uses {EXTRACTOR_VERSION}; retrain",
        bundle.provenance.extractor_version
    );
    Ok(bundle)
}

fn head_schema<'a>(bundle: &'a CuratorBundle, key: &str) -> Result<(&'a Head, &'a FeatureSchema)> {
    let head = bundle.heads.get(key).with_context(|| {
        format!(
            "bundle has no head {key:?}; heads: {}",
            bundle.heads.keys().cloned().collect::<Vec<_>>().join(", ")
        )
    })?;
    let schema = bundle
        .feature_schemas
        .get(&head.schema)
        .with_context(|| format!("head {key:?} references missing schema {:?}", head.schema))?;
    Ok((head, schema))
}

/// A phy folder blinded under `--out`, extracted, and labelled when it has a
/// curation. `temp` owns the blinded copy the session reads.
struct Blinded {
    session: Session<KilosortProvider>,
    data: features::ExtractData,
    /// The folder's own curation in session spike order, when it has one.
    truth: Option<features::Truth>,
    temp: tempfile::TempDir,
}

/// Blind PHY_DIR into a temporary directory under `out` and extract exactly as
/// `learn-dataset` does. PHY_DIR is never opened for writing and no temporary
/// file is created in it.
fn extract_blinded(phy_dir: &Path, out_dir: &Path, opts: &ExtractOpts) -> Result<Blinded> {
    ensure!(
        opts.max_spikes_per_root > 0,
        "--max-spikes-per-root must be positive"
    );
    ensure!(
        phy_dir.is_dir(),
        "PHY_DIR {} is not a directory",
        phy_dir.display()
    );
    ensure!(
        phy_dir.join("params.py").is_file(),
        "PHY_DIR {} has no params.py; pass a phy/Kilosort output folder",
        phy_dir.display()
    );
    // Resolve both sides through junctions/symlinks before anything is written:
    // --out must never land inside the folder the command promises not to touch.
    let phy = super::bench::canonical_target(phy_dir)?;
    let out = super::bench::canonical_target(out_dir)?;
    ensure!(
        !out.starts_with(&phy),
        "--out {} is inside PHY_DIR {}; choose a directory outside it",
        out.display(),
        phy.display()
    );

    // The blinded copy lives under --out, not the system temp dir: blinding
    // hard-links on one volume, so --out on another volume gives a real copy.
    fs::create_dir_all(out_dir).with_context(|| format!("create --out {}", out_dir.display()))?;
    let temp = tempfile::Builder::new()
        .prefix("predict-")
        .tempdir_in(out_dir)?;
    let input = temp.path().join("input");
    let truth_dir = temp.path().join("truth");
    // Reads PHY_DIR only; the session below opens the blinded copy.
    sorrel_data::blind::write_phy_curated_pair(&phy, None, &input, &truth_dir)?;
    let params = sorrel_io::PhyParams::read(&input.join("params.py"))?;
    let raw = params
        .dat_path
        .unwrap_or_else(|| PathBuf::from("recording.dat"));
    ensure!(
        raw.components().count() == 1 && input.join(&raw).is_file(),
        "blind input has no self-contained raw recording; not a phy/Kilosort folder"
    );

    let (session, mut data) = open_extract(&input, &temp.path().join("predict.journal"), opts)?;

    // Truth last, exactly as learn-dataset: only after every input-only feature.
    // An uncurated Kilosort folder has spike_clusters.npy but no labels.
    let truth = if truth_dir.join("spike_clusters.npy").is_file()
        && truth_dir.join("cluster_group.tsv").is_file()
    {
        let (truth, _, _) = super::bench::load_truth(&truth_dir, &data)?;
        features::label(&mut data, &truth)?;
        Some(truth)
    } else {
        None
    };
    Ok(Blinded {
        session,
        data,
        truth,
        temp,
    })
}

/// Open the phy folder `input` with a fresh journal and extract its tables,
/// with its current clusters as the roots.
fn open_extract(
    input: &Path,
    journal: &Path,
    opts: &ExtractOpts,
) -> Result<(Session<KilosortProvider>, features::ExtractData)> {
    let provider = KilosortProvider::open(input, Default::default())?;
    let mut session = Session::new(provider, Journal::open(journal)?);
    session.seed_pc_features();
    session.seed_templates();
    session.seed_template_waveforms();
    session.seed_amplitudes();
    let data = features::extract(&session, "predict", opts)?;
    Ok((session, data))
}

/// Row-major feature matrix of `table` in `schema` order. A `root_f_*` name
/// the table lacks is read from `roots` at the row's root; on the roots table
/// itself it is the same column.
fn schema_matrix(
    table: &Table,
    roots: &Table,
    schema: &FeatureSchema,
    key: &str,
) -> Result<Vec<f32>> {
    enum Source {
        Own(usize),
        Root(usize),
    }
    let mut sources = Vec::with_capacity(schema.names.len());
    let mut missing = Vec::new();
    for name in &schema.names {
        let joined = name.strip_prefix("root_").and_then(|n| roots.column(n));
        match (table.column(name), joined) {
            (Some(column), _) => sources.push(Source::Own(column)),
            (None, Some(column)) => sources.push(Source::Root(column)),
            (None, None) => missing.push(name.clone()),
        }
    }
    ensure!(
        missing.is_empty(),
        "the scored table has no column(s) {} required by head {key:?} schema {}; \
         was the head trained on this table?",
        missing.join(", "),
        schema.hash
    );
    let root_row: BTreeMap<u32, usize> = (0..roots.n_rows())
        .map(|row| (roots.get(row, "root") as u32, row))
        .collect();
    let root_column = table.column("root");
    let mut x = Vec::with_capacity(table.n_rows() * sources.len());
    for row in 0..table.n_rows() {
        let joined = root_column.and_then(|c| root_row.get(&(table.value(row, c) as u32)));
        for source in &sources {
            x.push(match source {
                Source::Own(column) => table.value(row, *column),
                Source::Root(column) => joined.map_or(f32::NAN, |&r| roots.value(r, *column)),
            });
        }
    }
    Ok(x)
}

/// `sorrel dev learn predict`: blind PHY_DIR, extract exactly as
/// `learn-dataset`, score the roots table with a stored head, and write the
/// predictions plus an optional truth report. PHY_DIR is never opened for
/// writing and no temporary file is created in it.
fn predict(a: &PredictArgs) -> Result<()> {
    let bundle = load_bundle(&a.model)?;
    let key = match a.head.as_deref() {
        Some(key) => key,
        None => match bundle.heads.len() {
            1 => bundle.heads.keys().next().expect("one head"),
            0 => bail!("bundle {} has no heads", a.model.display()),
            n => bail!(
                "bundle {} has {n} heads ({}); pass --head KEY",
                a.model.display(),
                bundle.heads.keys().cloned().collect::<Vec<_>>().join(", ")
            ),
        },
    };
    let (head, schema) = head_schema(&bundle, key)?;
    let classes = head_class_names(key, head)?;
    let opts = ExtractOpts {
        max_spikes_per_root: a.max_spikes_per_root,
        seed: a.seed,
    };
    let Blinded {
        data,
        truth,
        temp: _temp,
        ..
    } = extract_blinded(&a.phy_dir, &a.out, &opts)?;
    let truth_present = truth.is_some();
    let roots = &data.tables.roots;
    let n_roots = roots.n_rows();

    let matrix = schema_matrix(roots, roots, schema, key)?;
    let n_features = schema.names.len();
    let mut predicted = Vec::with_capacity(n_roots);
    let mut probabilities = Vec::with_capacity(n_roots);
    for row in 0..n_roots {
        let x = &matrix[row * n_features..(row + 1) * n_features];
        let mut p = head.predict_proba(&bundle, x, &schema.hash)?;
        if matches!(head.model.objective, Objective::Logistic) && p.len() == 1 {
            p = vec![1.0 - p[0], p[0]];
        }
        ensure!(
            p.len() == classes.len(),
            "head {key:?} returned {} probabilities for {} classes",
            p.len(),
            classes.len()
        );
        let mut best = 0;
        for (k, value) in p.iter().enumerate() {
            if *value > p[best] {
                best = k;
            }
        }
        predicted.push(best);
        probabilities.push(p);
    }

    let truth_classes: Option<Vec<Option<usize>>> = truth_present.then(|| {
        (0..n_roots)
            .map(|row| {
                let value = roots.get(row, "y_verdict");
                let class = value as usize;
                if value.is_finite()
                    && value >= 0.0
                    && class < classes.len()
                    && (value - value.round()).abs() < 1e-4
                {
                    Some(class)
                } else {
                    None
                }
            })
            .collect()
    });

    let mut metrics = None;
    let mut n_truth = 0usize;
    let mut correct = 0usize;
    if let Some(truth) = &truth_classes {
        let pairs: Vec<(usize, usize)> = predicted
            .iter()
            .copied()
            .zip(truth.iter().copied())
            .filter_map(|(p, t)| t.map(|t| (t, p)))
            .collect();
        n_truth = pairs.len();
        if n_truth > 0 {
            let y: Vec<usize> = pairs.iter().map(|&(t, _)| t).collect();
            let p: Vec<usize> = pairs.iter().map(|&(_, p)| p).collect();
            let m = classification(&y, &p, classes.len())?;
            correct = m.confusion.iter().enumerate().map(|(k, row)| row[k]).sum();
            metrics = Some(m);
        }
    }

    let mut tsv = String::new();
    let _ = write!(tsv, "cluster_id\tn_spikes\tpredicted");
    for class in &classes {
        let _ = write!(tsv, "\tp_{class}");
    }
    tsv.push_str("\ttruth\n");
    for row in 0..n_roots {
        let _ = write!(
            tsv,
            "{}\t{}\t{}",
            roots.get(row, "root") as u32,
            roots.get(row, "f_n_spikes") as usize,
            classes[predicted[row]]
        );
        for p in &probabilities[row] {
            let _ = write!(tsv, "\t{p:.6}");
        }
        let truth = truth_classes
            .as_ref()
            .and_then(|t| t[row])
            .map(|c| classes[c].as_str())
            .unwrap_or("");
        let _ = writeln!(tsv, "\t{truth}");
    }

    let predictions_path = a.out.join("predictions.tsv");
    let report_path = a.out.join("report.json");
    fs::write(&predictions_path, tsv)
        .with_context(|| format!("write {}", predictions_path.display()))?;
    let report = PredictReport {
        model: a.model.display().to_string(),
        head: key.to_owned(),
        n_roots,
        truth: truth_present,
        n_truth: truth_present.then_some(n_truth),
        accuracy: metrics.as_ref().map(|m| m.accuracy),
        confusion: metrics.as_ref().map(|m| ConfusionReport {
            classes: classes.clone(),
            matrix: m.confusion.clone(),
        }),
        per_class: metrics.as_ref().map(|m| {
            m.classes
                .iter()
                .enumerate()
                .map(|(k, class)| ClassScore {
                    class: classes[k].clone(),
                    support: class.support,
                    predicted: class.predicted,
                    precision: class.precision,
                    recall: class.recall,
                    f1: class.f1,
                })
                .collect()
        }),
    };
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", report_path.display()))?;

    println!("predict: {n_roots} root(s) from {}", a.phy_dir.display());
    println!(
        "head {key:?}: {} class(es), {} feature(s), model {}",
        classes.len(),
        schema.names.len(),
        a.model.display()
    );
    if truth_present {
        println!("truth: {n_truth}/{n_roots} root(s) labelled");
        if let Some(m) = &metrics {
            println!("accuracy {:.3} ({correct}/{n_truth})", m.accuracy);
        }
    } else {
        println!("truth: none");
    }
    println!(
        "wrote {} and {}",
        predictions_path.display(),
        report_path.display()
    );
    Ok(())
}

#[derive(Serialize)]
struct CurateReport {
    model: String,
    verdict_head: String,
    spike_head: String,
    threshold: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    split_head: Option<String>,
    split_threshold: f64,
    min_spikes: usize,
    late_onset: f64,
    max_centre: f64,
    rounds: usize,
    n_roots: usize,
    n_units_centre_bar: usize,
    n_roots_split: usize,
    n_units_under_floor: usize,
    n_good_units: usize,
    n_spikes: usize,
    n_spikes_dropped: usize,
    /// The bench score against PHY_DIR's own curation, when it has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<sorrel_data::diff::SpikeDiff>,
}

/// The units a root splits into: candidate memberships scored at least
/// `threshold`, taken best first, each keeping only the spikes no better
/// candidate took and accepted while that is at least 90% of it. Fewer than
/// two such units is no split (empty).
fn pick_split(mut scored: Vec<(f64, &[u32])>, threshold: f64) -> Vec<Vec<u32>> {
    scored.retain(|(p, _)| *p >= threshold);
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.len().cmp(&a.1.len())));
    let mut taken: BTreeSet<u32> = BTreeSet::new();
    let mut units = Vec::new();
    for (_, members) in scored {
        let fresh: Vec<u32> = members
            .iter()
            .copied()
            .filter(|g| !taken.contains(g))
            .collect();
        if fresh.len() * 10 >= members.len() * 9 {
            taken.extend(&fresh);
            units.push(fresh);
        }
    }
    if units.len() < 2 {
        units.clear();
    }
    units
}

/// The spike-count floor: at least `min_spikes`, or a late-onset unit (at
/// least `MIN_ROOT_SPIKES`, 95% of them after `late_onset` of the recording).
fn passes_floor(mut times: Vec<u64>, total: u64, min_spikes: usize, late_onset: f64) -> bool {
    if times.len() >= min_spikes {
        return true;
    }
    if times.len() < features::MIN_ROOT_SPIKES {
        return false;
    }
    times.sort_unstable();
    times[times.len() / 20] as f64 >= late_onset * total as f64
}

/// The ACG centre bar of a spike train against its baseline: spike pairs
/// within 0.5 ms per ms of lag, over pairs 20-50 ms apart per ms (both
/// signs count alike). 0 when the baseline is empty.
fn centre_ratio(mut times: Vec<u64>, sr: f32) -> f64 {
    times.sort_unstable();
    let ms = |m: f64| (m * sr as f64 / 1e3).round() as u64;
    let (centre, lo, hi) = (ms(0.5), ms(20.0), ms(50.0));
    let (mut near, mut far) = (0u64, 0u64);
    for (i, &t) in times.iter().enumerate() {
        for &u in &times[i + 1..] {
            let d = u - t;
            if d > hi {
                break;
            }
            if d <= centre {
                near += 1;
            } else if d > lo {
                far += 1;
            }
        }
    }
    if far == 0 {
        return 0.0;
    }
    (near as f64 / 0.5) / (far as f64 / 30.0)
}

/// `sorrel dev learn curate`: the verdict head decides which roots are units.
/// A kept root either splits into the split head's candidate units, or stays
/// one unit that the spike head cleans up. Units under the spike floor, or
/// with an ACG centre bar above `--max-centre`, become noise. The result is written as phy files under `--out` (never into
/// PHY_DIR), and scored against PHY_DIR's own curation when it has one.
///
/// ponytail: no merges yet, and split units get no spike cleanup. Later
/// rounds score clusters the heads were not trained on (leftovers, not
/// sorter roots).
fn curate(a: &CurateArgs) -> Result<()> {
    ensure!(
        (0.0..=1.0).contains(&a.threshold),
        "--threshold must be within 0..=1"
    );
    ensure!(
        (0.0..=1.0).contains(&a.late_onset),
        "--late-onset must be within 0..=1"
    );
    ensure!(a.rounds > 0, "--rounds must be positive");
    let bundle = load_bundle(&a.model)?;
    let (verdict, verdict_schema) = head_schema(&bundle, &a.verdict_head)?;
    let (spike, spike_schema) = head_schema(&bundle, &a.spike_head)?;
    let classes = head_class_names(&a.verdict_head, verdict)?;
    let noise = classes.iter().position(|c| c == "noise").with_context(|| {
        format!(
            "verdict head {:?} has no `noise` class ({})",
            a.verdict_head,
            classes.join(", ")
        )
    })?;
    ensure!(
        matches!(spike.model.objective, Objective::Logistic),
        "spike head {:?} must be a logistic keep/drop head",
        a.spike_head
    );
    let split = a
        .split_head
        .as_deref()
        .map(|key| {
            let (head, schema) = head_schema(&bundle, key)?;
            ensure!(
                matches!(head.model.objective, Objective::Logistic),
                "split head {key:?} must be a logistic head"
            );
            Ok((key, head, schema))
        })
        .transpose()?;
    let opts = ExtractOpts {
        max_spikes_per_root: a.max_spikes_per_root,
        seed: 0,
    };
    let Blinded {
        session,
        data,
        truth,
        temp,
    } = extract_blinded(&a.phy_dir, &a.out, &opts)?;
    let n_spikes = data.root_of_spike.len();
    let n_roots = data.tables.roots.n_rows();

    // Spikes are indexed in the first session's order throughout. A later
    // round reopens the folder with this round's clusters as its roots, and
    // its spikes are mapped back through their original NPY rows.
    let input_roots = data.root_of_spike.clone();
    let mut first_of_row = vec![u32::MAX; n_spikes];
    for (g, row) in data.row_map.iter().enumerate() {
        let row = row.context("the session has no original NPY-row map")? as usize;
        ensure!(
            row < n_spikes && first_of_row[row] == u32::MAX,
            "the session's NPY-row map is not a permutation"
        );
        first_of_row[row] = g as u32;
    }
    let by_row = |cluster: &[u32]| -> Vec<u32> {
        let mut out = vec![0; n_spikes];
        for (row, &g) in first_of_row.iter().enumerate() {
            out[row] = cluster[g as usize];
        }
        out
    };

    // Every spike starts in its root; dropped spikes move to a new noise unit.
    let mut cluster = input_roots.clone();
    let mut next_id = input_roots.iter().max().map_or(0, |&r| r + 1);
    let mut labels: BTreeMap<u32, &str> = BTreeMap::new();
    let mut rows = String::from(
        "cluster_id\tn_spikes\tverdict\tgroup\tn_dropped\tdropped_to\tsplit_into\tnote\tround\n",
    );
    let mut cand_rows =
        String::from("round\troot\tcand_idx\tmethod\tn_spikes\tp_hit\ty_jaccard\ty_best_unit\n");
    let (mut n_good, mut n_dropped_total) = (0usize, 0usize);
    let (mut n_split, mut n_floored, mut n_centre) = (0usize, 0usize, 0usize);
    // The clusters this round looks at; `None` is every root.
    let mut todo: Option<BTreeSet<u32>> = None;
    let mut pass = (session, data);
    for round in 1..=a.rounds {
        if round > 1 {
            let input = temp.path().join("input");
            sorrel_io::npy::write_1d_u32_atomic(
                &input.join("spike_clusters.npy"),
                &by_row(&cluster),
            )?;
            drop(pass);
            pass = open_extract(
                &input,
                &temp.path().join(format!("round{round}.journal")),
                &opts,
            )?;
        }
        let (session, data) = &pass;
        ensure!(
            data.root_of_spike.len() == n_spikes,
            "round {round} sees {} spikes, not {n_spikes}",
            data.root_of_spike.len()
        );
        // First-session index of each of this session's spikes.
        let first: Vec<u32> = data
            .row_map
            .iter()
            .map(|row| {
                let row = row.context("the session has no original NPY-row map")? as usize;
                first_of_row
                    .get(row)
                    .copied()
                    .context("the session's NPY-row map is out of range")
            })
            .collect::<Result<_>>()?;
        let roots = &data.tables.roots;
        let mut globals: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (g, &r) in data.root_of_spike.iter().enumerate() {
            globals.entry(r).or_default().push(g as u32);
        }
        // Candidate rows per root with the split head's score of each.
        let cands = &data.tables.candidates;
        let mut cand_scores: BTreeMap<u32, Vec<(f64, usize)>> = BTreeMap::new();
        if let Some((key, head, schema)) = split {
            if cands.n_rows() > 0 {
                let x = schema_matrix(cands, roots, schema, key)?;
                let p = head
                    .model
                    .predict_proba_batch(&x, schema.names.len())
                    .context("score the candidates")?;
                for (row, &p) in p.iter().enumerate() {
                    let root = cands.get(row, "root") as u32;
                    if todo.as_ref().is_some_and(|t| !t.contains(&root)) {
                        continue;
                    }
                    // Truth overlap is only labelled in the first round.
                    let _ = writeln!(
                        cand_rows,
                        "{round}\t{root}\t{}\t{}\t{}\t{p:.3}\t{:.3}\t{}",
                        cands.get(row, "cand_idx"),
                        cands.get(row, "f_method"),
                        data.candidate_members[row].len(),
                        cands.get(row, "y_jaccard"),
                        cands.get(row, "y_best_unit"),
                    );
                    // The whole root is the no-split case, not a split unit.
                    if cands.get(row, "f_method") != 0.0 {
                        cand_scores.entry(root).or_default().push((p, row));
                    }
                }
            }
        }
        let floor = |unit: &[u32]| {
            passes_floor(
                unit.iter()
                    .map(|&g| data.time_of_spike[g as usize].0)
                    .collect(),
                data.total_samples,
                a.min_spikes,
                a.late_onset,
            )
        };
        let clean_centre = |unit: &[u32]| {
            centre_ratio(
                unit.iter()
                    .map(|&g| data.time_of_spike[g as usize].0)
                    .collect(),
                data.sr,
            ) <= a.max_centre
        };
        let verdict_x = schema_matrix(roots, roots, verdict_schema, &a.verdict_head)?;
        let n_verdict = verdict_schema.names.len();
        // Leftovers of this round that are big enough to hold a unit.
        let mut leftovers: BTreeSet<u32> = BTreeSet::new();
        for row in 0..roots.n_rows() {
            let root = roots.get(row, "root") as u32;
            if todo.as_ref().is_some_and(|t| !t.contains(&root)) {
                continue;
            }
            let p = verdict.predict_proba(
                &bundle,
                &verdict_x[row * n_verdict..(row + 1) * n_verdict],
                &verdict_schema.hash,
            )?;
            ensure!(
                p.len() == classes.len(),
                "head {:?} returned {} probabilities for {} classes",
                a.verdict_head,
                p.len(),
                classes.len()
            );
            let mut best = 0;
            for (k, value) in p.iter().enumerate() {
                if *value > p[best] {
                    best = k;
                }
            }
            let members = globals.get(&root).map_or(&[][..], Vec::as_slice);
            let units = if best == noise {
                Vec::new()
            } else {
                let scored = cand_scores.get(&root).map_or(Vec::new(), |rows| {
                    rows.iter()
                        .map(|&(p, row)| (p, data.candidate_members[row].as_slice()))
                        .collect()
                });
                let mut units = pick_split(scored, a.split_threshold);
                let before = units.len();
                units.retain(|unit| floor(unit));
                n_floored += before - units.len();
                let before = units.len();
                units.retain(|unit| clean_centre(unit));
                n_centre += before - units.len();
                if units.len() < 2 {
                    units.clear();
                }
                units
            };
            if !units.is_empty() {
                // The root id keeps the remainder as noise; each unit is new.
                n_split += 1;
                let mut ids = Vec::new();
                let mut moved = 0;
                for unit in &units {
                    for &g in unit {
                        cluster[first[g as usize] as usize] = next_id;
                    }
                    moved += unit.len();
                    labels.insert(next_id, "good");
                    ids.push(next_id.to_string());
                    next_id += 1;
                }
                n_good += units.len();
                n_dropped_total += members.len() - moved;
                labels.insert(root, "noise");
                if members.len() - moved >= a.min_spikes {
                    leftovers.insert(root);
                }
                let _ = writeln!(
                    rows,
                    "{root}\t{}\t{}\tnoise\t{}\t\t{}\tsplit\t{round}",
                    members.len(),
                    classes[best],
                    members.len() - moved,
                    ids.join(","),
                );
                continue;
            }
            let mut dropped: Vec<u32> = Vec::new();
            if best != noise {
                // Root-local order equals global order within a root, so spike
                // row `i` is `members[i]`. A root too small for spike rows is
                // kept whole.
                let table = features::root_spike_features(session, data, root, &opts)?;
                ensure!(
                    table.n_rows() == 0 || table.n_rows() == members.len(),
                    "root {root} has {} spike rows for {} spikes",
                    table.n_rows(),
                    members.len()
                );
                let x = schema_matrix(&table, roots, spike_schema, &a.spike_head)?;
                if table.n_rows() > 0 {
                    let p_keep = spike
                        .model
                        .predict_proba_batch(&x, spike_schema.names.len())
                        .with_context(|| format!("score the spikes of root {root}"))?;
                    dropped = members
                        .iter()
                        .zip(&p_keep)
                        .filter(|(_, &p)| p < a.threshold)
                        .map(|(&g, _)| g)
                        .collect();
                }
            }
            let mut note = "";
            let group = if best == noise || dropped.len() == members.len() {
                "noise"
            } else {
                let gone: BTreeSet<u32> = dropped.iter().copied().collect();
                let kept: Vec<u32> = members
                    .iter()
                    .copied()
                    .filter(|g| !gone.contains(g))
                    .collect();
                if !floor(&kept) {
                    n_floored += 1;
                    note = "under the spike floor";
                    "noise"
                } else if !clean_centre(&kept) {
                    n_centre += 1;
                    note = "ACG centre bar";
                    "noise"
                } else {
                    n_good += 1;
                    "good"
                }
            };
            labels.insert(root, group);
            n_dropped_total += dropped.len();
            let dropped_to = if dropped.is_empty() || group == "noise" {
                String::new()
            } else {
                for &g in &dropped {
                    cluster[first[g as usize] as usize] = next_id;
                }
                labels.insert(next_id, "noise");
                if dropped.len() >= a.min_spikes {
                    leftovers.insert(next_id);
                }
                next_id += 1;
                (next_id - 1).to_string()
            };
            let _ = writeln!(
                rows,
                "{root}\t{}\t{}\t{group}\t{}\t{dropped_to}\t\t{note}\t{round}",
                members.len(),
                classes[best],
                dropped.len(),
            );
        }
        if leftovers.is_empty() {
            break;
        }
        todo = Some(leftovers);
    }
    drop(pass);

    let labels: Vec<(u32, &str)> = labels.into_iter().collect();
    let score = truth
        .as_ref()
        .map(|truth| {
            let test_labels: Vec<(u32, sorrel_io::kilosort::PhyLabel)> = labels
                .iter()
                .map(|&(id, group)| (id, sorrel_io::kilosort::PhyLabel::parse_tsv_value(group)))
                .collect();
            sorrel_data::diff::diff_spikes(
                &input_roots,
                &sorrel_data::diff::Curation {
                    clusters: &truth.cluster,
                    labels: &truth.labels,
                },
                &sorrel_data::diff::Curation {
                    clusters: &cluster,
                    labels: &test_labels,
                },
                &Default::default(),
            )
        })
        .transpose()?;

    // phy files are in original NPY row order, not session order.
    sorrel_io::npy::write_1d_u32_atomic(&a.out.join("spike_clusters.npy"), &by_row(&cluster))?;
    let mut tsv = String::from("cluster_id\tgroup\n");
    for (id, group) in &labels {
        let _ = writeln!(tsv, "{id}\t{group}");
    }
    let mut files = vec![("cluster_group.tsv", &tsv), ("roots.tsv", &rows)];
    if split.is_some() {
        files.push(("candidates.tsv", &cand_rows));
    }
    for (name, text) in files {
        let path = a.out.join(name);
        fs::write(&path, text).with_context(|| format!("write {}", path.display()))?;
    }
    let report = CurateReport {
        model: a.model.display().to_string(),
        verdict_head: a.verdict_head.clone(),
        spike_head: a.spike_head.clone(),
        threshold: a.threshold,
        split_head: a.split_head.clone(),
        split_threshold: a.split_threshold,
        min_spikes: a.min_spikes,
        late_onset: a.late_onset,
        max_centre: a.max_centre,
        rounds: a.rounds,
        n_units_centre_bar: n_centre,
        n_roots,
        n_roots_split: n_split,
        n_units_under_floor: n_floored,
        n_good_units: n_good,
        n_spikes,
        n_spikes_dropped: n_dropped_total,
        score,
    };
    let report_path = a.out.join("report.json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", report_path.display()))?;

    println!(
        "curate: {n_roots} root(s), {n_split} split, {n_good} good unit(s), {n_floored} under the spike floor, {n_centre} with an ACG centre bar, {n_dropped_total}/{n_spikes} spike(s) dropped",
    );
    match &report.score {
        Some(score) => print!("{}", score.render()),
        None => println!("truth: none"),
    }
    println!("wrote {}", a.out.display());
    Ok(())
}

fn sorrel_rev() -> Option<String> {
    std::env::var("SORREL_REV")
        .ok()
        .or_else(|| option_env!("SORREL_GIT_REV").map(str::to_owned))
}

/// Pooled CV numbers under a `cv_` prefix; per-class F1 under `cv_f1_c{k}`.
fn head_metrics(report: &CvReport) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    match &report.pooled {
        FoldMetrics::Classification(m) => {
            out.push(("cv_accuracy".into(), m.accuracy));
            out.push(("cv_balanced_accuracy".into(), m.balanced_accuracy));
            out.push(("cv_macro_f1".into(), m.macro_f1));
            for (k, class) in m.classes.iter().enumerate() {
                if class.support > 0 {
                    out.push((format!("cv_f1_c{k}"), class.f1));
                    out.push((format!("cv_recall_c{k}"), class.recall));
                }
            }
        }
        FoldMetrics::Regression(m) => {
            out.push(("cv_r2".into(), m.r2));
            out.push(("cv_spearman".into(), m.spearman));
            out.push(("cv_rmse".into(), m.rmse));
        }
    }
    out.retain(|(_, v)| v.is_finite());
    out
}

fn ranking(
    table: &str,
    target: &str,
    objective: sorrel_data::learn::gbdt::Objective,
    sel: &Selection,
    predictions: &[f64],
) -> Option<RootRanking> {
    if table != "candidates" || target != "y_jaccard" {
        return None;
    }
    if !matches!(objective, sorrel_data::learn::gbdt::Objective::SquaredError) {
        return None;
    }
    // The candidate table carries no `root` if the caller dropped it; the
    // selection then failed earlier, so this is safe.
    let keys: Vec<(u32, u32)> = sel
        .sid_idx
        .iter()
        .zip(&sel.roots)
        .map(|(&sid, &root)| (sid, root))
        .collect();
    let units: Vec<(u32, u32)> = sel
        .sid_idx
        .iter()
        .zip(&sel.unit)
        .map(|(&sid, &u)| (sid, if u == u32::MAX { 0 } else { u }))
        .collect();
    root_ranking(&keys, predictions, &sel.hit, &units).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_split_needs_two_disjoint_candidates_over_the_threshold() {
        let (a, b, both, low): (&[u32], &[u32], &[u32], &[u32]) =
            (&[0, 1, 2], &[3, 4, 5], &[0, 1, 2, 3, 4, 5], &[6, 7]);
        // `both` overlaps the better picks entirely; `low` is under the bar.
        let units = pick_split(vec![(0.7, both), (0.9, a), (0.8, b), (0.2, low)], 0.5);
        assert_eq!(units, vec![a.to_vec(), b.to_vec()]);
        assert!(pick_split(vec![(0.9, a), (0.2, b)], 0.5).is_empty());
    }

    #[test]
    fn the_centre_ratio_sees_only_pairs_inside_half_a_millisecond() {
        // sr 1 kHz: one sample is 1 ms. Spikes 30 ms apart fill the baseline.
        let clean: Vec<u64> = (0..100).map(|i| i * 30).collect();
        assert_eq!(centre_ratio(clean.clone(), 1000.0), 0.0);
        let mut dirty = clean;
        dirty.extend((0..100).map(|i| i * 30)); // same-sample doubles
        assert!(centre_ratio(dirty, 1000.0) > 1.0);
        assert_eq!(centre_ratio(vec![0, 1000], 1000.0), 0.0);
    }

    #[test]
    fn the_spike_floor_spares_only_a_late_onset_unit() {
        let late: Vec<u64> = (0..150).map(|i| 800 + i).collect();
        let early: Vec<u64> = (0..150).collect();
        assert!(passes_floor(late.clone(), 1000, 300, 0.5));
        assert!(!passes_floor(early, 1000, 300, 0.5));
        assert!(!passes_floor(late[..15].to_vec(), 1000, 300, 0.5));
        assert!(passes_floor((0..300).collect(), 1000, 300, 0.5));
    }

    #[test]
    fn lesioned_table_selection_drops_rows_and_joins_roots() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = Table::new(
            EXTRACTOR_VERSION,
            [
                "sid_idx",
                "root",
                "f_a",
                "f_b",
                "y_jaccard",
                "y_hit",
                "y_best_unit",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        );
        t.sids = vec!["s0".into()];
        t.target_version = Some(TARGET_VERSION.into());
        t.push_row(&[0.0, 1.0, 2.0, f32::NAN, 0.9, 1.0, 7.0])
            .unwrap();
        t.push_row(&[0.0, 1.0, 3.0, 0.0, f32::NAN, 0.0, f32::NAN])
            .unwrap();
        t.save(dir.path(), "candidates").unwrap();
        let mut r = Table::new(
            EXTRACTOR_VERSION,
            ["sid_idx", "root", "f_pct"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        );
        r.sids = vec!["s0".into()];
        r.push_row(&[0.0, 1.0, 0.5]).unwrap();
        r.save(dir.path(), "roots").unwrap();
        let sel = select(
            dir.path(),
            TableArgs {
                table: "candidates",
                target: "y_jaccard",
                objective: "regression",
                class_map: &[],
                exclude: &[],
                join_roots: true,
                root_verdict: &[],
                row_stride: 1,
                groups: None,
            },
        )
        .unwrap();
        assert_eq!(sel.y.len(), 1);
        assert_eq!(sel.names, vec!["f_a", "f_b", "root_f_pct"]);
        assert_eq!(sel.x[0], 2.0);
        assert!(sel.x[1].is_nan());
        assert_eq!(sel.x[2], 0.5);
        assert_eq!(sel.roots, vec![1]);
        assert!(sel.hit[0]);
    }

    /// A roots table with an optional documented encoding of `y_verdict`.
    fn verdict_table(dir: &Path, documented: bool, values: &[f32]) {
        let mut t = Table::new(
            EXTRACTOR_VERSION,
            ["sid_idx", "root", "f_a", "y_verdict"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        );
        t.sids = vec!["s0".into()];
        t.target_version = Some(TARGET_VERSION.into());
        if documented {
            t.codes.insert(
                "y_verdict".into(),
                [
                    ("noise", 0),
                    ("kept", 1),
                    ("cleanup", 2),
                    ("cores", 3),
                    ("other", 4),
                ]
                .into_iter()
                .map(|(name, code)| (name.to_owned(), code))
                .collect(),
            );
        }
        for (row, &value) in values.iter().enumerate() {
            t.push_row(&[0.0, row as f32, 1.0, value]).unwrap();
        }
        t.save(dir, "roots").unwrap();
    }

    fn select_verdict(dir: &Path, objective: &str, class_map: &[String]) -> Result<Selection> {
        select(
            dir,
            TableArgs {
                table: "roots",
                target: "y_verdict",
                objective,
                class_map,
                exclude: &[],
                join_roots: false,
                root_verdict: &[],
                row_stride: 1,
                groups: None,
            },
        )
    }

    #[test]
    fn documented_class_count_is_checked_before_rows_are_dropped() {
        let dir = tempfile::tempdir().unwrap();
        verdict_table(dir.path(), true, &[0.0, 1.0, 4.0, 4.0]);
        let err = select_verdict(dir.path(), "softmax:4", &[])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("y_verdict has 5 classes (codes in the table header); use softmax:5"),
            "{err}"
        );
        // Without a documented encoding the row scan itself must refuse the
        // out-of-range value, naming it, its count and the fix.
        verdict_table(dir.path(), false, &[0.0, 1.0, 4.0, 4.0]);
        let err = select_verdict(dir.path(), "softmax:4", &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("y_verdict = 4 (2 row(s))"), "{err}");
        assert!(err.contains("outside softmax:4"), "{err}");
        assert!(err.contains("softmax:5"), "{err}");
    }

    #[test]
    fn class_map_merges_classes_explicitly_and_only_once() {
        let dir = tempfile::tempdir().unwrap();
        verdict_table(dir.path(), true, &[0.0, 3.0, 4.0, 4.0]);
        let map = vec!["4=3".to_owned()];
        let sel = select_verdict(dir.path(), "softmax:4", &map).unwrap();
        assert_eq!(sel.y, vec![0.0, 3.0, 3.0, 3.0]);
        // The documented five-class encoding conflicts with softmax:5 once
        // the map merges `other` into `cores`.
        let err = select_verdict(dir.path(), "softmax:5", &map)
            .unwrap_err()
            .to_string();
        assert!(err.contains("--class-map leaves 4 classes"), "{err}");
        // Chains are ambiguous: apply one explicit step instead.
        let chain = vec!["4=3".to_owned(), "3=2".to_owned()];
        let err = parse_class_map(&chain).unwrap_err().to_string();
        assert!(err.contains("cannot chain"), "{err}");
    }

    #[test]
    fn logistic_rejects_anything_that_is_not_binary() {
        let dir = tempfile::tempdir().unwrap();
        verdict_table(dir.path(), true, &[0.0, 1.0]);
        let err = select_verdict(dir.path(), "logistic", &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("logistic needs binary 0/1"), "{err}");
        verdict_table(dir.path(), false, &[0.0, 2.0]);
        let err = select_verdict(dir.path(), "logistic", &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("logistic needs labels 0 or 1"), "{err}");
        assert!(err.contains("2 (1 row(s))"), "{err}");
    }

    #[test]
    fn a_fractional_class_is_an_error_not_a_silent_drop() {
        let dir = tempfile::tempdir().unwrap();
        verdict_table(dir.path(), false, &[0.0, 1.5]);
        let err = select_verdict(dir.path(), "softmax:2", &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("non-integer"), "{err}");
        assert!(err.contains("1.5"), "{err}");
    }

    #[test]
    fn schema_keys_are_content_hashes_so_two_heads_share_one_schema() {
        let a = FeatureSchema::new(vec!["f_a".into(), "f_b".into()], EXTRACTOR_VERSION);
        let same = FeatureSchema::new(vec!["f_a".into(), "f_b".into()], EXTRACTOR_VERSION);
        let different = FeatureSchema::new(vec!["f_a".into(), "f_c".into()], EXTRACTOR_VERSION);
        assert_eq!(a.key(), same.key());
        assert_ne!(a.key(), different.key());
        assert_eq!(a.key(), format!("schema:{}", a.hash));

        let mut bundle = CuratorBundle::new(Provenance::new(0, "0.4.0", None, EXTRACTOR_VERSION));
        bundle.add_schema_checked(a.key(), a.clone()).unwrap();
        bundle.add_schema_checked(same.key(), same.clone()).unwrap();
        bundle
            .add_schema_checked(different.key(), different.clone())
            .unwrap();
        assert_eq!(bundle.feature_schemas.len(), 2);
    }
}

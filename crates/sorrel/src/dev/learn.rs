//! `sorrel dev learn cv|train`: grouped cross-validation and model fitting on
//! an extracted `learn-dataset` directory.
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
use sorrel_data::learn::metrics::{root_ranking, RootRanking};
use sorrel_data::learn::table::Table;
use sorrel_data::learn::{EXTRACTOR_VERSION, TARGET_VERSION};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Subcommand, Debug)]
pub enum LearnCmd {
    /// Grouped K-fold cross-validation of one table and target
    Cv(CvArgs),
    /// Fit on every row and write/update a `.sorrel-model` bundle head
    Train(TrainArgs),
}

#[derive(Args, Debug)]
pub struct CvArgs {
    /// Directory produced by `dev bench learn-dataset`
    #[arg(long, value_name = "DIR")]
    data: PathBuf,
    /// Table name: roots, candidates, pairs, units or spikes
    #[arg(long, value_name = "NAME")]
    table: String,
    /// Target column, e.g. y_verdict
    #[arg(long, value_name = "COLUMN")]
    target: String,
    /// logistic, softmax:K or regression
    #[arg(long, value_name = "OBJECTIVE", value_parser = objective_parser)]
    objective: String,
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
    #[arg(long, value_name = "FILE")]
    groups: Option<PathBuf>,
    /// Folds used for the CV metrics stored in the head
    #[arg(long, default_value_t = 5)]
    folds: usize,
    #[arg(long = "exclude-prefix", value_name = "PREFIX")]
    exclude_prefix: Vec<String>,
    #[arg(long)]
    join_roots: bool,
    /// Destination `.sorrel-model` (created, or extended with `--head`)
    #[arg(long, value_name = "FILE")]
    out: PathBuf,
    /// Head key in the bundle; an existing key is replaced
    #[arg(long, value_name = "NAME")]
    head: String,
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
struct Selection {
    names: Vec<String>,
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

    // Optional joined root features, keyed by (sid_idx, root).
    let joined: Option<RootJoin> = if a.join_roots {
        ensure!(
            table.column("root").is_some(),
            "table {:?} has no root column to join the roots table on",
            a.table
        );
        let roots = Table::load(data, "roots").context("--join-roots needs the roots table")?;
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
        let mut any = false;
        let mut column = 0;
        while column < roots.columns.len() {
            let name = &roots.columns[column];
            if name.starts_with("f_") && !a.exclude.iter().any(|p| name.starts_with(p)) {
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

    let n_features = names.len();
    let y_column = table.column(a.target).expect("checked above");
    let objective = parse_objective(a.objective).expect("clap validated the objective");
    let n_classes = match objective {
        sorrel_data::learn::gbdt::Objective::Softmax { n_classes } => n_classes,
        sorrel_data::learn::gbdt::Objective::Logistic => 2,
        sorrel_data::learn::gbdt::Objective::SquaredError => 1,
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
    for row in 0..table.n_rows() {
        let raw = table.value(row, y_column);
        if !raw.is_finite() {
            continue;
        }
        let label = match objective {
            sorrel_data::learn::gbdt::Objective::SquaredError => raw as f64,
            _ => {
                let c = raw as i64;
                if (raw - c as f32).abs() > 1e-4 || c < 0 || (c as usize) >= n_classes {
                    continue;
                }
                c as f64
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

/// Arguments shared by `cv` and `train`.
struct TableArgs<'a> {
    table: &'a str,
    target: &'a str,
    objective: &'a str,
    exclude: &'a [String],
    join_roots: bool,
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
            exclude: &a.exclude_prefix,
            join_roots: a.join_roots,
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
            exclude: &a.exclude_prefix,
            join_roots: a.join_roots,
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
    let mut head = Head::new(schema_key.clone(), model);
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
                exclude: &[],
                join_roots: true,
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

//! Grouped K-fold cross-validation and full-data fitting for the learned
//! curator, on top of [`super::gbdt`].
//!
//! Folds are assigned to whole groups, never rows: recordings (or animals, when
//! a groups file is supplied) must not straddle the train/test boundary, or a
//! model can memorise a recording's quirks instead of learning curation.
//! Assignment is deterministic and greedy-balanced: groups sorted by size
//! (largest first, ties by name) each go to the currently lightest fold.
//!
//! Classification folds weight rows by inverse class frequency so a rare
//! "cores" verdict is not drowned by "noise", and report per-class
//! precision/recall/F1 plus a confusion matrix. Regression reports R² and
//! Spearman. Feature importances are mean split gains across folds, so a
//! feature that only one fold used does not rank like one all folds used.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};

use super::gbdt::{self, Model, Objective, Params};
use super::metrics::{
    balanced_weights, classification, regression, ClassificationMetrics, RegressionMetrics,
};

/// Rows fed to a cross-validation or fit.
pub struct Problem<'a> {
    /// Row-major features (`n_rows × n_features`).
    pub x: &'a [f32],
    /// Number of feature columns.
    pub n_features: usize,
    /// Feature names, one per column.
    pub columns: &'a [String],
    /// One label per row (class index, 0/1, or a regression value).
    pub y: &'a [f32],
    /// Group name per row (a session id, or an animal id).
    pub groups: &'a [String],
}

/// Hyper-parameters of one fit. Defaults are [`Params::default`] with the seed
/// from the caller.
#[derive(Clone, Debug)]
pub struct CvOptions {
    /// Number of folds; the number of distinct groups must be at least this.
    pub folds: usize,
    /// Model seed.
    pub seed: u64,
    /// Override the library defaults (e.g. smaller trees for tiny tables).
    pub params: Option<Params>,
}

impl Default for CvOptions {
    fn default() -> Self {
        Self {
            folds: 5,
            seed: 0,
            params: None,
        }
    }
}

/// Metrics of one fold, tagged by objective family in JSON.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FoldMetrics {
    /// Logistic or softmax held-out scoring.
    Classification(ClassificationMetrics),
    /// Squared-error held-out scoring.
    Regression(RegressionMetrics),
}

/// One feature's mean gain across folds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Column name.
    pub name: String,
    /// Mean split gain over the folds that trained a model.
    pub gain: f64,
}

/// Everything a cross-validation run reports.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CvReport {
    /// Objective as written on the CLI (`logistic`, `softmax:4`, `regression`).
    pub objective: String,
    /// Number of scored rows.
    pub n_rows: usize,
    /// Number of selected feature columns.
    pub n_features: usize,
    /// Number of folds requested.
    pub folds: usize,
    /// `(fold, rows, groups)` for each fold.
    pub fold_sizes: Vec<(usize, usize, usize)>,
    /// Per-fold metrics, in fold order.
    pub per_fold: Vec<FoldMetrics>,
    /// Metrics over all out-of-fold predictions together.
    pub pooled: FoldMetrics,
    /// Features with non-zero mean gain, highest first, at most 20.
    pub importances: Vec<FeatureImportance>,
}

/// A report plus the out-of-fold predictions that produced it (in row order),
/// so callers can compute per-root ranking metrics or write a calibration
/// table.
pub struct CvOutput {
    /// Metrics and importances.
    pub report: CvReport,
    /// One out-of-fold prediction per row: class index for classification,
    /// value for regression.
    pub predictions: Vec<f64>,
}

/// `"logistic"`, `"softmax:K"`, or `"regression"`.
pub fn parse_objective(text: &str) -> Result<Objective> {
    if text.eq_ignore_ascii_case("logistic") {
        return Ok(Objective::Logistic);
    }
    if text.eq_ignore_ascii_case("regression")
        || text.eq_ignore_ascii_case("squared-error")
        || text.eq_ignore_ascii_case("squared_error")
    {
        return Ok(Objective::SquaredError);
    }
    if let Some(k) = text.strip_prefix("softmax:") {
        let k: usize = k
            .parse()
            .with_context(|| format!("softmax class count in {text:?} is not a number"))?;
        ensure!(k >= 2, "softmax needs at least two classes");
        return Ok(Objective::Softmax { n_classes: k });
    }
    bail!("--objective must be logistic, softmax:K or regression, not {text:?}")
}

/// Objective in the CLI's spelling, for reports and provenance.
pub fn objective_name(objective: Objective) -> String {
    match objective {
        Objective::Logistic => "logistic".into(),
        Objective::Softmax { n_classes } => format!("softmax:{n_classes}"),
        Objective::SquaredError => "regression".into(),
    }
}

/// Assign every row to a fold, keeping groups whole and balancing row counts.
///
/// The returned vector has one fold index in `0..folds` per row. The mapping
/// is a pure function of the group labels, so two runs (and two machines)
/// produce the same split.
pub fn assign_folds(groups: &[String], folds: usize) -> Result<Vec<usize>> {
    ensure!(folds >= 2, "--folds must be at least 2");
    let mut sizes: BTreeMap<&str, usize> = BTreeMap::new();
    for g in groups {
        *sizes.entry(g.as_str()).or_default() += 1;
    }
    ensure!(
        sizes.len() >= folds,
        "cannot make {folds} folds from {} distinct group(s); pass a groups file or use fewer folds",
        sizes.len()
    );
    let mut order: Vec<(&str, usize)> = sizes.into_iter().collect();
    order.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    let mut loads = vec![0usize; folds];
    let mut where_: BTreeMap<&str, usize> = BTreeMap::new();
    for (group, size) in order {
        // Ties go to the lowest fold index, which keeps the split stable.
        let fold = (0..folds)
            .min_by_key(|&f| (loads[f], f))
            .unwrap_or_default();
        loads[fold] += size;
        where_.insert(group, fold);
    }
    Ok(groups.iter().map(|g| where_[g.as_str()]).collect())
}

/// Read the optional `{"groups": {sid: group}}` JSON file.
pub fn read_groups(path: &Path) -> Result<BTreeMap<String, String>> {
    #[derive(Deserialize)]
    struct File {
        groups: BTreeMap<String, String>,
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read groups file {}", path.display()))?;
    let file: File = serde_json::from_str(&text)
        .with_context(|| format!("parse groups file {}", path.display()))?;
    // An empty map would read as "no --groups" and skip the animal checks.
    ensure!(
        !file.groups.is_empty(),
        "groups file {} maps no sessions",
        path.display()
    );
    Ok(file.groups)
}

/// Group name per row: the recording id, unless `by_sid` maps the session to a
/// coarser group (an animal). Re-curations of one recording share its id, so
/// they cannot land in different folds. A requested animal mapping must cover
/// every session: falling back to the sid would turn an unmapped re-curation
/// into its own group again.
pub fn row_groups(
    sids: &[String],
    recordings: &BTreeMap<String, String>,
    by_sid: &BTreeMap<String, String>,
) -> Result<Vec<String>> {
    let missing: BTreeSet<&str> = sids
        .iter()
        .filter(|sid| !recordings.contains_key(*sid))
        .map(String::as_str)
        .collect();
    ensure!(
        missing.is_empty(),
        "the table has no recording id for session(s) {}; regenerate it with \
         `sorrel dev bench learn-dataset` so re-curations of one recording share a fold",
        missing.into_iter().collect::<Vec<_>>().join(", ")
    );
    if by_sid.is_empty() {
        return Ok(sids.iter().map(|sid| recordings[sid].clone()).collect());
    }
    let mut unmapped: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for sid in sids {
        if !by_sid.contains_key(sid) {
            let recording = recordings[sid].as_str();
            unmapped.entry(recording).or_default().insert(sid);
        }
    }
    ensure!(
        unmapped.is_empty(),
        "the groups file maps no animal for recording(s) {}; map every session so one recording cannot span folds",
        unmapped
            .iter()
            .map(|(rec, missing)| format!(
                "{rec:?} ({})",
                missing.iter().copied().collect::<Vec<_>>().join(", ")
            ))
            .collect::<Vec<_>>()
            .join("; ")
    );
    // One recording must also agree with itself: two curations mapped to
    // different animals would still be split by the animal grouping.
    let mut group_of: BTreeMap<&str, &str> = BTreeMap::new();
    for sid in sids {
        let recording = recordings[sid].as_str();
        let group = by_sid[sid].as_str();
        if let Some(&other) = group_of.get(recording) {
            ensure!(
                other == group,
                "recording {recording:?} has sessions in two animals ({other:?} and {group:?}); \
                 one recording must stay in one group"
            );
        } else {
            group_of.insert(recording, group);
        }
    }
    Ok(sids.iter().map(|s| by_sid[s].clone()).collect())
}

/// Train on all but one fold at a time and score the held-out rows.
pub fn cross_validate(
    problem: &Problem<'_>,
    options: &CvOptions,
    objective: Objective,
) -> Result<CvOutput> {
    let n = problem.y.len();
    ensure!(n > 0, "no rows to cross-validate");
    ensure!(
        problem.x.len() == n * problem.n_features,
        "feature matrix has {} values for {n} rows × {} features",
        problem.x.len(),
        problem.n_features
    );
    ensure!(problem.groups.len() == n, "one group per row is required");
    ensure!(
        problem.columns.len() == problem.n_features,
        "{} column names for {} features",
        problem.columns.len(),
        problem.n_features
    );
    validate_labels(problem.y, objective)?;
    let fold_of = assign_folds(problem.groups, options.folds)?;
    let base_params = options.params.clone().unwrap_or(Params {
        seed: options.seed,
        ..Params::default()
    });
    let mut predictions = vec![f64::NAN; n];
    let mut per_fold = Vec::with_capacity(options.folds);
    let mut fold_sizes = Vec::with_capacity(options.folds);
    let mut importance_sum = vec![0.0f64; problem.n_features];
    let mut trained = 0usize;
    for fold in 0..options.folds {
        let train_rows: Vec<usize> = (0..n).filter(|&i| fold_of[i] != fold).collect();
        let test_rows: Vec<usize> = (0..n).filter(|&i| fold_of[i] == fold).collect();
        ensure!(
            !train_rows.is_empty() && !test_rows.is_empty(),
            "fold {fold} is empty; use more groups or fewer folds"
        );
        // Everything a fold fits must see training rows only. Class weights
        // below, the missing-class dummy rows and the trainer's binning are
        // all built from `train_rows`; so is the leaf floor.
        let mut params = base_params.clone();
        params.min_samples_leaf = leaf_floor(params.min_samples_leaf, train_rows.len());
        let mut distinct: Vec<&str> = test_rows
            .iter()
            .map(|&i| problem.groups[i].as_str())
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        fold_sizes.push((fold, test_rows.len(), distinct.len()));
        let (mut x_train, mut y_train) = gather(problem, &train_rows);
        let mut weights = match objective {
            Objective::SquaredError => None,
            Objective::Logistic => {
                let labels: Vec<usize> = y_train.iter().map(|&v| v as usize).collect();
                Some(balanced_weights(&labels, 2))
            }
            Objective::Softmax { n_classes } => {
                let labels: Vec<usize> = y_train.iter().map(|&v| v as usize).collect();
                Some(balanced_weights(&labels, n_classes))
            }
        };
        // A fold can miss a class even when the full table has it (and a tiny
        // first-N cohort can miss it entirely). The trainer needs every class
        // to exist; a zero-weight all-NaN row keeps the label space complete
        // without teaching the model anything about the missing class's shape.
        if let Some(w) = weights.as_mut() {
            add_missing_classes(&mut x_train, &mut y_train, w, problem.n_features, objective);
        }
        let model = gbdt::train(
            &x_train,
            problem.n_features,
            &y_train,
            weights.as_deref(),
            objective,
            &params,
        )
        .with_context(|| format!("train fold {fold}"))?;
        let x_test = gather_x(problem, &test_rows);
        let predicted = model
            .predict_batch(&x_test, problem.n_features)
            .with_context(|| format!("score fold {fold}"))?;
        for (&row, &p) in test_rows.iter().zip(&predicted) {
            predictions[row] = p;
        }
        per_fold.push(score_fold(
            objective,
            &test_rows.iter().map(|&i| problem.y[i]).collect::<Vec<_>>(),
            &predicted,
        )?);
        for (f, g) in model.feature_importance().iter().enumerate() {
            importance_sum[f] += g;
        }
        trained += 1;
    }
    let pooled = score_fold(objective, problem.y, &predictions)?;
    let mut importances: Vec<FeatureImportance> = importance_sum
        .iter()
        .zip(problem.columns)
        .map(|(&gain, name)| FeatureImportance {
            name: name.clone(),
            gain: gain / trained.max(1) as f64,
        })
        .filter(|f| f.gain > 0.0)
        .collect();
    importances.sort_by(|a, b| b.gain.total_cmp(&a.gain).then_with(|| a.name.cmp(&b.name)));
    importances.truncate(20);
    Ok(CvOutput {
        report: CvReport {
            objective: objective_name(objective),
            n_rows: n,
            n_features: problem.n_features,
            folds: options.folds,
            fold_sizes,
            per_fold,
            pooled,
            importances,
        },
        predictions,
    })
}

/// Fit one model on every row. Used by `learn train`; no fold is held out.
pub fn fit_full(problem: &Problem<'_>, seed: u64, objective: Objective) -> Result<Model> {
    let n = problem.y.len();
    ensure!(n > 0, "no rows to train on");
    ensure!(
        problem.x.len() == n * problem.n_features,
        "feature matrix has {} values for {n} rows × {} features",
        problem.x.len(),
        problem.n_features
    );
    validate_labels(problem.y, objective)?;
    let weights = match objective {
        Objective::SquaredError => None,
        Objective::Logistic => {
            let labels: Vec<usize> = problem.y.iter().map(|&v| v as usize).collect();
            Some(balanced_weights(&labels, 2))
        }
        Objective::Softmax { n_classes } => {
            let labels: Vec<usize> = problem.y.iter().map(|&v| v as usize).collect();
            Some(balanced_weights(&labels, n_classes))
        }
    };
    let (mut x, mut y) = (problem.x.to_vec(), problem.y.to_vec());
    let mut weights = weights;
    if let Some(w) = weights.as_mut() {
        add_missing_classes(&mut x, &mut y, w, problem.n_features, objective);
    }
    let mut params = Params {
        seed,
        ..Params::default()
    };
    params.min_samples_leaf = leaf_floor(params.min_samples_leaf, n);
    gbdt::train(
        &x,
        problem.n_features,
        &y,
        weights.as_deref(),
        objective,
        &params,
    )
}

/// A leaf floor of 20 rows is right for thousands of roots and fatal for a
/// 40-row smoke table, where nothing can split and every fold predicts one
/// class. Lower it to `n/20` (never below 2) for the rows the model actually
/// trains on. Callers must pass *training* row counts: validation rows must
/// never influence a fitted parameter.
fn leaf_floor(default: usize, n_train: usize) -> usize {
    default.min((n_train / 20).max(2))
}

/// Append one zero-weight, all-NaN row per class absent from `y`.
///
/// The trainer rejects a softmax (or logistic) problem whose label space is
/// incomplete, and a held-out fold legitimately lacks rare classes. A weight
/// of zero keeps such a row out of every histogram and leaf while still
/// telling the objective that the class exists.
fn add_missing_classes(
    x: &mut Vec<f32>,
    y: &mut Vec<f32>,
    weights: &mut Vec<f32>,
    n_features: usize,
    objective: Objective,
) {
    let n_classes = match objective {
        Objective::Logistic => 2,
        Objective::Softmax { n_classes } => n_classes,
        Objective::SquaredError => return,
    };
    debug_assert_eq!(weights.len(), y.len());
    for class in 0..n_classes {
        if y.iter().any(|&v| v as usize == class) {
            continue;
        }
        x.extend(std::iter::repeat_n(f32::NAN, n_features));
        y.push(class as f32);
        weights.push(0.0);
    }
}

fn gather(problem: &Problem<'_>, rows: &[usize]) -> (Vec<f32>, Vec<f32>) {
    (
        gather_x(problem, rows),
        rows.iter().map(|&i| problem.y[i]).collect(),
    )
}

fn gather_x(problem: &Problem<'_>, rows: &[usize]) -> Vec<f32> {
    let d = problem.n_features;
    let mut out = Vec::with_capacity(rows.len() * d);
    for &i in rows {
        out.extend_from_slice(&problem.x[i * d..(i + 1) * d]);
    }
    out
}

fn validate_labels(y: &[f32], objective: Objective) -> Result<()> {
    match objective {
        Objective::Logistic => ensure!(
            y.iter().all(|&v| v == 0.0 || v == 1.0),
            "logistic labels must be 0 or 1"
        ),
        Objective::Softmax { n_classes } => ensure!(
            y.iter()
                .all(|&v| v.is_finite() && v >= 0.0 && (v as usize) < n_classes),
            "softmax labels must be class indices in 0..{n_classes}"
        ),
        Objective::SquaredError => ensure!(
            y.iter().all(|v| v.is_finite()),
            "regression labels must be finite"
        ),
    }
    Ok(())
}

fn score_fold(objective: Objective, y: &[f32], predictions: &[f64]) -> Result<FoldMetrics> {
    Ok(match objective {
        Objective::SquaredError => FoldMetrics::Regression(regression(
            &y.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            predictions,
        )?),
        Objective::Logistic | Objective::Softmax { .. } => {
            let classes = match objective {
                Objective::Softmax { n_classes } => n_classes,
                _ => 2,
            };
            let truth: Vec<usize> = y.iter().map(|&v| v as usize).collect();
            let predicted: Vec<usize> = predictions
                .iter()
                .map(|&p| {
                    if p.is_finite() && p >= 0.0 {
                        p as usize
                    } else {
                        classes
                    }
                })
                .collect();
            FoldMetrics::Classification(classification(&truth, &predicted, classes)?)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(
        n_per_group: usize,
        groups: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<String>, Vec<String>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut g = Vec::new();
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as f32 / (1u64 << 31) as f32
        };
        for group in 0..groups {
            for i in 0..n_per_group {
                let class = (i * 3 / n_per_group).min(2);
                let base = class as f32 * 6.0;
                x.push(base + next() * 2.0);
                x.push(base * 0.5 + next() * 2.0);
                x.push(next());
                y.push(class as f32);
                g.push(format!("g{group}"));
            }
        }
        let columns = vec!["f_a".into(), "f_b".into(), "f_c".into()];
        (x, y, g, columns)
    }

    #[test]
    fn folds_are_whole_group_and_balanced() {
        let groups: Vec<String> = (0..7).flat_map(|g| vec![format!("g{g}"); g + 1]).collect();
        let folds = assign_folds(&groups, 3).unwrap();
        assert_eq!(folds.len(), groups.len());
        for g in 0..7 {
            let name = format!("g{g}");
            let fs: Vec<usize> = groups
                .iter()
                .zip(&folds)
                .filter(|(s, _)| **s == name)
                .map(|(_, &f)| f)
                .collect();
            assert!(fs.iter().all(|f| *f == fs[0]), "group {g} spans folds");
        }
        let loads: Vec<usize> = (0..3)
            .map(|f| folds.iter().filter(|&&x| x == f).count())
            .collect();
        assert_eq!(loads.iter().sum::<usize>(), groups.len());
        assert!(loads.iter().max().unwrap() - loads.iter().min().unwrap() <= 4);
        assert!(assign_folds(&groups, 8).is_err());
    }

    #[test]
    fn re_curations_of_one_recording_share_a_fold_and_an_animal_needs_a_full_map() {
        // Two curations of recording r1 and two of r2: four sessions, two
        // recordings. Grouping by session would scatter r1 across folds.
        let sids: Vec<String> = ["a1", "a2", "b1", "b2"].map(str::to_owned).to_vec();
        let recordings: BTreeMap<String, String> =
            [("a1", "r1"), ("a2", "r1"), ("b1", "r2"), ("b2", "r2")]
                .map(|(sid, rec)| (sid.to_owned(), rec.to_owned()))
                .into();
        let groups = row_groups(&sids, &recordings, &BTreeMap::new()).unwrap();
        assert_eq!(groups, ["r1", "r1", "r2", "r2"]);
        // A table without recording ids (stale) is refused, not grouped by sid.
        let mut stale = recordings.clone();
        stale.remove("a2");
        let error = row_groups(&sids, &stale, &BTreeMap::new())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("a2") && error.contains("learn-dataset"),
            "{error}"
        );
        let folds = assign_folds(&groups, 2).unwrap();
        assert_eq!(folds[0], folds[1], "a re-curation changed folds");
        assert_eq!(folds[2], folds[3], "a re-curation changed folds");
        assert_ne!(folds[0], folds[2]);

        // Animal grouping: a partial map is refused, naming the recording and
        // its unmapped session, instead of falling back to a session group.
        let partial: BTreeMap<String, String> = [("a1".to_owned(), "animal-7".to_owned())].into();
        let error = row_groups(&sids, &recordings, &partial)
            .unwrap_err()
            .to_string();
        assert!(error.contains("r1") && error.contains("a2"), "{error}");

        // A recording mapped to two animals is refused too.
        let split: BTreeMap<String, String> = [
            ("a1".to_owned(), "animal-7".to_owned()),
            ("a2".to_owned(), "animal-8".to_owned()),
            ("b1".to_owned(), "animal-8".to_owned()),
            ("b2".to_owned(), "animal-8".to_owned()),
        ]
        .into();
        assert!(row_groups(&sids, &recordings, &split).is_err());

        let complete: BTreeMap<String, String> = [
            ("a1".to_owned(), "animal-7".to_owned()),
            ("a2".to_owned(), "animal-7".to_owned()),
            ("b1".to_owned(), "animal-8".to_owned()),
            ("b2".to_owned(), "animal-8".to_owned()),
        ]
        .into();
        assert_eq!(
            row_groups(&sids, &recordings, &complete).unwrap(),
            ["animal-7", "animal-7", "animal-8", "animal-8"]
        );
    }

    #[test]
    fn softmax_cv_learns_a_separable_problem() {
        let (x, y, groups, columns) = synthetic(60, 8);
        let problem = Problem {
            x: &x,
            n_features: 3,
            columns: &columns,
            y: &y,
            groups: &groups,
        };
        let out = cross_validate(
            &problem,
            &CvOptions {
                folds: 4,
                seed: 7,
                params: None,
            },
            Objective::Softmax { n_classes: 3 },
        )
        .unwrap();
        assert_eq!(out.report.per_fold.len(), 4);
        assert_eq!(out.report.n_rows, 480);
        let FoldMetrics::Classification(pooled) = &out.report.pooled else {
            panic!("softmax pooled metrics are classification");
        };
        assert!(
            pooled.accuracy > 0.8,
            "separability should survive CV, got {}",
            pooled.accuracy
        );
        assert!(!out.report.importances.is_empty());
        assert!(out.report.importances[0].name.starts_with("f_"));
        assert_eq!(out.predictions.len(), 480);
    }

    #[test]
    fn regression_cv_reports_r2_and_spearman() {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut groups = Vec::new();
        for g in 0..6 {
            for i in 0..40 {
                let a = (i as f32 / 40.0) - 0.5;
                let b = ((g * 7 + i) % 11) as f32 / 11.0;
                x.push(a);
                x.push(b);
                y.push(0.5 + 0.4 * a);
                groups.push(format!("g{g}"));
            }
        }
        let columns = vec!["f_a".into(), "f_b".into()];
        let problem = Problem {
            x: &x,
            n_features: 2,
            columns: &columns,
            y: &y,
            groups: &groups,
        };
        let out = cross_validate(
            &problem,
            &CvOptions {
                folds: 3,
                seed: 1,
                params: None,
            },
            Objective::SquaredError,
        )
        .unwrap();
        let FoldMetrics::Regression(pooled) = &out.report.pooled else {
            panic!("regression pooled metrics");
        };
        assert!(pooled.r2 > 0.3, "r2 {}", pooled.r2);
        assert!(pooled.spearman > 0.4, "spearman {}", pooled.spearman);
    }

    #[test]
    fn objective_parsing_is_explicit() {
        assert_eq!(parse_objective("logistic").unwrap(), Objective::Logistic);
        assert_eq!(
            parse_objective("softmax:4").unwrap(),
            Objective::Softmax { n_classes: 4 }
        );
        assert_eq!(
            parse_objective("regression").unwrap(),
            Objective::SquaredError
        );
        assert!(parse_objective("softmax:1").is_err());
        assert!(parse_objective("bce").is_err());
        assert_eq!(
            objective_name(Objective::Softmax { n_classes: 3 }),
            "softmax:3"
        );
    }

    #[test]
    fn leaf_floor_uses_the_training_row_count() {
        assert_eq!(leaf_floor(20, 2000), 20);
        assert_eq!(leaf_floor(20, 40), 2);
        assert_eq!(leaf_floor(1000, 40), 2);
        assert_eq!(leaf_floor(1000, 200), 10);

        // 207 rows in five folds: the global floor would be 10, the training
        // floor 7. A 9-row leaf at x=10 is therefore only reachable when the
        // floor is computed from training rows, as it must be.
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut groups = Vec::new();
        for g in 0..6 {
            for _ in 0..23 {
                x.push(0.0);
                y.push(0.0);
                groups.push(format!("g{g}"));
            }
        }
        for _ in 0..60 {
            x.push(10.0);
            y.push(1.0);
            groups.push("a".to_string());
        }
        for _ in 0..9 {
            x.push(10.0);
            y.push(1.0);
            groups.push("b".to_string());
        }
        let columns = vec!["f_a".to_string()];
        let problem = Problem {
            x: &x,
            n_features: 1,
            columns: &columns,
            y: &y,
            groups: &groups,
        };
        let params = Params {
            n_rounds: 1,
            learning_rate: 1.0,
            max_depth: 1,
            max_leaves: 2,
            l2: 0.0,
            min_gain: 0.0,
            ..Params::default()
        };
        let out = cross_validate(
            &problem,
            &CvOptions {
                folds: 5,
                seed: 0,
                params: Some(params),
            },
            Objective::SquaredError,
        )
        .unwrap();
        let fold_of = assign_folds(&groups, 5).unwrap();
        let fold_a = fold_of[groups.iter().position(|g| g == "a").unwrap()];
        let special: Vec<usize> = (0..groups.len()).filter(|&i| groups[i] == "a").collect();
        assert!(special.iter().all(|&i| fold_of[i] == fold_a));
        for &i in &special {
            assert!(
                out.predictions[i] > 0.5,
                "row {i} predicted {} with the wrong leaf floor",
                out.predictions[i]
            );
        }
    }
}

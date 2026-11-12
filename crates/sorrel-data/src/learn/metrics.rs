//! Held-out metrics for learned-curator heads: classification (accuracy,
//! balanced accuracy, per-class precision/recall/F1, confusion matrix) and
//! regression (R², Spearman rank correlation).
//!
//! Everything here is plain arithmetic over label/prediction vectors so the
//! CLI can report fold-by-fold and pooled numbers without an ML dependency,
//! and the unit tests can pin down tie handling (Spearman uses average ranks).

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One class's side of a confusion matrix.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassReport {
    /// Rows whose true label is this class.
    pub support: usize,
    /// Rows predicted as this class.
    pub predicted: usize,
    /// `tp / predicted`; `0` when nothing was predicted.
    pub precision: f64,
    /// `tp / support`; `NaN` when the class has no rows (undefined, not zero).
    pub recall: f64,
    /// Harmonic mean of precision and recall; `0` when the class was never
    /// predicted, `NaN` when it has no rows.
    pub f1: f64,
}

/// Classification metrics over one set of rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    /// Rows scored.
    pub n: usize,
    /// `correct / n`; `NaN` for no rows.
    pub accuracy: f64,
    /// Mean recall over classes that have support in `y`; `NaN` when none do.
    pub balanced_accuracy: f64,
    /// One entry per class, in class-index order.
    pub classes: Vec<ClassReport>,
    /// `confusion[true][predicted]`.
    pub confusion: Vec<Vec<usize>>,
    /// Macro F1 over classes with support.
    pub macro_f1: f64,
}

/// Score `y` against `pred` for `n_classes` classes (0..n_classes).
///
/// Predictions outside `0..n_classes` are counted as errors and are not added
/// to any confusion cell: a model cannot produce one, but a mixed table can.
pub fn classification(
    y: &[usize],
    pred: &[usize],
    n_classes: usize,
) -> Result<ClassificationMetrics> {
    ensure!(
        y.len() == pred.len(),
        "labels and predictions differ in length"
    );
    ensure!(n_classes > 0, "n_classes must be positive");
    let mut confusion = vec![vec![0usize; n_classes]; n_classes];
    let mut correct = 0usize;
    for (&t, &p) in y.iter().zip(pred) {
        ensure!(t < n_classes, "true label {t} outside 0..{n_classes}");
        if p < n_classes {
            confusion[t][p] += 1;
            if t == p {
                correct += 1;
            }
        }
    }
    let n = y.len();
    let mut classes = Vec::with_capacity(n_classes);
    for k in 0..n_classes {
        let support: usize = confusion[k].iter().sum();
        let predicted: usize = confusion.iter().map(|row| row[k]).sum();
        let tp = confusion[k][k];
        let precision = if predicted == 0 {
            0.0
        } else {
            tp as f64 / predicted as f64
        };
        let recall = if support == 0 {
            f64::NAN
        } else {
            tp as f64 / support as f64
        };
        let f1 = if support == 0 {
            f64::NAN
        } else if predicted == 0 || precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        classes.push(ClassReport {
            support,
            predicted,
            precision,
            recall,
            f1,
        });
    }
    let with_support: Vec<&ClassReport> = classes.iter().filter(|c| c.support > 0).collect();
    let balanced_accuracy = if with_support.is_empty() {
        f64::NAN
    } else {
        with_support.iter().map(|c| c.recall).sum::<f64>() / with_support.len() as f64
    };
    let macro_f1 = if with_support.is_empty() {
        f64::NAN
    } else {
        with_support.iter().map(|c| c.f1).sum::<f64>() / with_support.len() as f64
    };
    Ok(ClassificationMetrics {
        n,
        accuracy: if n == 0 {
            f64::NAN
        } else {
            correct as f64 / n as f64
        },
        balanced_accuracy,
        classes,
        confusion,
        macro_f1,
    })
}

/// Regression metrics over one set of rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegressionMetrics {
    /// Rows with a finite prediction and label.
    pub n: usize,
    /// Coefficient of determination; `NaN` when fewer than two rows or the
    /// labels are constant (the denominator is zero).
    pub r2: f64,
    /// Spearman rank correlation with average ranks; `NaN` when undefined.
    pub spearman: f64,
    /// Mean prediction error (`pred - y`).
    pub bias: f64,
    /// Root mean squared error.
    pub rmse: f64,
}

/// Score regression predictions, dropping rows with a non-finite value.
pub fn regression(y: &[f64], pred: &[f64]) -> Result<RegressionMetrics> {
    ensure!(
        y.len() == pred.len(),
        "labels and predictions differ in length"
    );
    let pairs: Vec<(f64, f64)> = y
        .iter()
        .copied()
        .zip(pred.iter().copied())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .collect();
    let n = pairs.len();
    if n == 0 {
        return Ok(RegressionMetrics {
            n,
            r2: f64::NAN,
            spearman: f64::NAN,
            bias: f64::NAN,
            rmse: f64::NAN,
        });
    }
    let mean_y = pairs.iter().map(|p| p.0).sum::<f64>() / n as f64;
    let ss_res: f64 = pairs.iter().map(|(a, b)| (a - b).powi(2)).sum();
    let ss_tot: f64 = pairs.iter().map(|(a, _)| (a - mean_y).powi(2)).sum();
    let (bias, rmse) = (
        pairs.iter().map(|(a, b)| b - a).sum::<f64>() / n as f64,
        (ss_res / n as f64).sqrt(),
    );
    Ok(RegressionMetrics {
        n,
        r2: if n < 2 || ss_tot <= 0.0 {
            f64::NAN
        } else {
            1.0 - ss_res / ss_tot
        },
        spearman: spearman(&pairs),
        bias,
        rmse,
    })
}

/// Spearman correlation of paired values: Pearson correlation of their
/// average ranks (ties share the mean rank). `NaN` when fewer than two pairs
/// or either side is constant.
fn spearman(pairs: &[(f64, f64)]) -> f64 {
    if pairs.len() < 2 {
        return f64::NAN;
    }
    let rank = |values: &[f64]| -> Vec<f64> {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let mut ranks = vec![0.0; values.len()];
        let mut i = 0;
        while i < order.len() {
            let mut j = i + 1;
            while j < order.len() && values[order[j]] == values[order[i]] {
                j += 1;
            }
            let mean = (i + j - 1) as f64 / 2.0 + 1.0;
            for &k in &order[i..j] {
                ranks[k] = mean;
            }
            i = j;
        }
        ranks
    };
    let a: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let b: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let (ra, rb) = (rank(&a), rank(&b));
    let n = ra.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let cov: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let (va, vb): (f64, f64) = (
        ra.iter().map(|x| (x - ma).powi(2)).sum(),
        rb.iter().map(|y| (y - mb).powi(2)).sum(),
    );
    if va <= 0.0 || vb <= 0.0 {
        return f64::NAN;
    }
    cov / (va.sqrt() * vb.sqrt())
}

/// One root's best score, whether that best is a hit, and the human units its
/// hit candidates reach.
type RootAccum = (f64, bool, Vec<(u32, u32)>);

/// Per-root ranking quality for the `candidates` table: of the roots that have
/// at least one hit candidate (the oracle ceiling), the fraction whose
/// highest-predicted candidate is a hit.
///
/// `root_keys[i]` is `(session, root)` and `units_of_row[i]` is
/// `(session, unit)`: root ids and human unit ids both repeat across
/// sessions, so the pair is the only safe key. Returns the roots with at least
/// one hit, the number whose highest-scored candidate is a hit, the fraction,
/// and the number of distinct human units reachable by any hit candidate.
pub fn root_ranking(
    root_keys: &[(u32, u32)],
    scores: &[f64],
    hits: &[bool],
    units_of_row: &[(u32, u32)],
) -> Result<RootRanking> {
    ensure!(
        root_keys.len() == scores.len()
            && root_keys.len() == hits.len()
            && root_keys.len() == units_of_row.len(),
        "ranking inputs differ in length"
    );
    let mut by_root: BTreeMap<(u32, u32), RootAccum> = BTreeMap::new();
    for (i, &key) in root_keys.iter().enumerate() {
        let entry = by_root
            .entry(key)
            .or_insert((f64::NEG_INFINITY, false, Vec::new()));
        // Deterministic ties: first row wins (tables are already row-sorted).
        if scores[i].is_finite() && scores[i] > entry.0 {
            entry.0 = scores[i];
            entry.1 = hits[i];
        }
        if hits[i] {
            entry.2.push(units_of_row[i]);
        }
    }
    let roots_with_hit = by_root.values().filter(|e| !e.2.is_empty()).count();
    let top1_hits = by_root.values().filter(|e| !e.2.is_empty() && e.1).count();
    let mut units: Vec<(u32, u32)> = by_root.values().flat_map(|e| e.2.iter().copied()).collect();
    units.sort_unstable();
    units.dedup();
    Ok(RootRanking {
        n_roots: by_root.len(),
        roots_with_hit,
        top1_hits,
        top1_fraction: if roots_with_hit == 0 {
            f64::NAN
        } else {
            top1_hits as f64 / roots_with_hit as f64
        },
        oracle_units: units.len(),
    })
}

/// Output of [`root_ranking`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RootRanking {
    /// Roots that have at least one candidate row.
    pub n_roots: usize,
    /// Roots with at least one hit candidate.
    pub roots_with_hit: usize,
    /// Roots whose highest-scored candidate is a hit.
    pub top1_hits: usize,
    /// `top1_hits / roots_with_hit`; `NaN` when no root has a hit.
    pub top1_fraction: f64,
    /// Distinct human units reachable by any hit candidate.
    pub oracle_units: usize,
}

/// Inverse-frequency sample weights, mean 1, for class indices `y`.
///
/// Rows of absent classes are left alone; each present class's rows share
/// `n / (n_classes_present × count_c)`.
pub fn balanced_weights(y: &[usize], n_classes: usize) -> Vec<f32> {
    let mut counts = vec![0usize; n_classes];
    for &c in y {
        if c < n_classes {
            counts[c] += 1;
        }
    }
    let present = counts.iter().filter(|&&c| c > 0).count().max(1);
    let n = y.len() as f64;
    y.iter()
        .map(|&c| {
            if c < n_classes && counts[c] > 0 {
                (n / (present as f64 * counts[c] as f64)) as f32
            } else {
                1.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_scores_and_confusion() {
        let y = [0, 0, 1, 1, 2];
        let p = [0, 1, 1, 1, 2];
        let m = classification(&y, &p, 3).unwrap();
        assert_eq!(m.n, 5);
        assert_eq!(
            m.confusion,
            vec![vec![1, 1, 0], vec![0, 2, 0], vec![0, 0, 1]]
        );
        assert!((m.accuracy - 0.8).abs() < 1e-12);
        assert!((m.classes[0].precision - 1.0).abs() < 1e-12);
        assert!((m.classes[0].recall - 0.5).abs() < 1e-12);
        assert!((m.balanced_accuracy - (0.5 + 1.0 + 1.0) / 3.0).abs() < 1e-12);
        assert!((m.macro_f1 - (2.0 / 3.0 + 0.8 + 1.0) / 3.0).abs() < 1e-12);
        // An absent class has undefined recall, not zero.
        let m = classification(&[0, 1], &[0, 1], 3).unwrap();
        assert!(m.classes[2].recall.is_nan());
        assert_eq!(m.classes[2].support, 0);
        assert!(classification(&[0], &[0], 0).is_err());
    }

    #[test]
    fn regression_r2_and_spearman_with_ties() {
        let y = [1.0, 2.0, 3.0, 4.0];
        let p = [1.5, 2.0, 3.5, 3.0];
        let m = regression(&y, &p).unwrap();
        assert_eq!(m.n, 4);
        assert!(m.r2 < 1.0 && m.r2 > 0.0);
        assert!((m.bias - 0.0).abs() < 1e-12);
        assert!(m.spearman > 0.7 && m.spearman < 1.0);
        // Perfect monotone but non-linear predictions: Spearman 1, R² < 1.
        let m = regression(&[1.0, 2.0, 3.0], &[1.0, 4.0, 9.0]).unwrap();
        assert!((m.spearman - 1.0).abs() < 1e-12);
        assert!(m.r2 < 0.0);
        // Ties share the average rank, so a constant side is undefined.
        let m = regression(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).unwrap();
        assert!(m.spearman.is_nan());
        assert!(m.r2.is_nan());
        // Non-finite rows drop out.
        assert_eq!(regression(&[1.0, f64::NAN], &[1.0, 2.0]).unwrap().n, 1);
    }

    #[test]
    fn root_ranking_counts_only_roots_with_a_hit() {
        let roots = [(0, 0), (0, 0), (0, 0), (0, 1), (0, 1), (0, 2)];
        let scores = [0.9, 0.2, 0.1, 0.5, 0.1, 0.3];
        let hits = [true, false, false, false, false, false];
        let units = [(0, 7), (0, 7), (0, 8), (0, 9), (1, 42), (0, 50)];
        let r = root_ranking(&roots, &scores, &hits, &units).unwrap();
        assert_eq!(r.n_roots, 3);
        assert_eq!(r.roots_with_hit, 1);
        assert_eq!(r.top1_hits, 1);
        assert!((r.top1_fraction - 1.0).abs() < 1e-12);
        assert_eq!(r.oracle_units, 1);
        // Root 1's best candidate is not a hit: the ceiling is 1/2. Unit 7 in
        // session 1 is not the same unit as unit 7 in session 0.
        let r = root_ranking(
            &roots,
            &scores,
            &[false, true, false, true, false, false],
            &[(0, 7), (0, 7), (0, 8), (1, 7), (1, 42), (0, 50)],
        )
        .unwrap();
        assert_eq!(r.roots_with_hit, 2);
        assert_eq!(r.top1_hits, 1);
        assert_eq!(r.oracle_units, 2);
        // The same root id in two sessions is two roots.
        let roots2 = [(0, 0), (1, 0)];
        let r = root_ranking(&roots2, &[0.1, 0.2], &[true, true], &[(0, 7), (1, 7)]).unwrap();
        assert_eq!(r.n_roots, 2);
        assert_eq!(r.oracle_units, 2);
    }

    #[test]
    fn balanced_weights_even_out_classes() {
        let w = balanced_weights(&[0, 0, 0, 1], 2);
        assert!((w[0] - 2.0 / 3.0).abs() < 1e-6);
        assert!((w[3] - 2.0).abs() < 1e-6);
        let mean: f32 = w.iter().sum::<f32>() / w.len() as f32;
        assert!((mean - 1.0).abs() < 1e-6);
    }
}

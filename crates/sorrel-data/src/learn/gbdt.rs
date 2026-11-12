//! Native histogram gradient-boosted decision trees, trained and applied
//! without an ML runtime.
//!
//! The learned curator trains small supervised models on a human's past
//! manual curations and must be able to score spikes on any machine that runs
//! sorrel. Pulling in a Python runtime or a native ML stack for that would be
//! far more trouble than a few hundred lines of trees, so this module
//! implements the parts of LightGBM / sklearn `HistGradientBoosting` that the
//! curator actually needs:
//!
//! * quantile binning to at most [`Params::max_bins`] bins, with NaN as its own
//!   bin and a learned default direction at every split;
//! * logistic, softmax and squared-error objectives with per-row weights;
//! * best-first (leaf-wise) tree growth bounded by `max_leaves` and
//!   `max_depth`, with `l2` leaf regularisation;
//! * row and column subsampling, early stopping and gain importance.
//!
//! Determinism is a hard requirement: the same data and seed must produce the
//! same bytes on every run and at every rayon thread count. Rows keep their
//! insertion order inside every node, histograms are accumulated in f64 in
//! that fixed order, and the only parallel loops partition work by feature —
//! each histogram cell has exactly one writer — so scheduling cannot change
//! an addition. Inputs are validated up front and every public entry point
//! returns `anyhow` errors rather than panicking on bad data.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use anyhow::{bail, ensure, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Rows sampled to derive bin edges. Bounded so binning cost is
/// O(features × 200k log 200k), independent of the full training size.
const MAX_BIN_ROWS: usize = 200_000;
/// Gradient, hessian and weight mass stored per histogram cell.
const HIST_STATS: usize = 3;
/// Work below this many (row × feature) visits is accumulated serially;
/// rayon's overhead is larger than the loop for tiny nodes.
const PAR_MIN_PAIRS: usize = 200_000;
/// Probability floor for log loss and base scores.
const PROB_EPS: f64 = 1e-6;
/// Improvement below this relative amount does not reset early-stop patience.
const EARLY_STOP_EPS: f64 = 1e-12;

/// Prediction target for a fitted model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// Binary classification. Labels are exactly `0.0` or `1.0`.
    Logistic,
    /// Multiclass classification. Labels are class indices in
    /// `0..n_classes`, and one tree is grown per class per round.
    Softmax { n_classes: usize },
    /// Regression, e.g. predicting a Jaccard score in `[0, 1]`.
    SquaredError,
}

impl Objective {
    /// Number of values produced by [`Model::predict_raw`].
    pub fn n_outputs(self) -> usize {
        match self {
            Objective::Softmax { n_classes } => n_classes,
            Objective::Logistic | Objective::SquaredError => 1,
        }
    }
}

/// Held-out rows used to stop boosting before overfitting.
///
/// The rows are stored by value because training needs the exact same feature
/// encoding as the training matrix, and callers routinely build them in a
/// different order. `patience` counts rounds without a loss improvement
/// better than 1e-12 relative; the returned model is truncated to the best
/// round.
#[derive(Clone, Debug)]
pub struct EarlyStopping {
    /// Row-major validation features (same `n_features` as training).
    pub x: Vec<f32>,
    /// One label per validation row, encoded like the training labels.
    pub y: Vec<f32>,
    /// Optional per-row validation weights.
    pub weights: Option<Vec<f32>>,
    /// Rounds to wait for an improvement before stopping.
    pub patience: usize,
}

/// Training hyper-parameters.
///
/// Defaults are deliberately small enough to train on a laptop and shipped
/// inside the curator so a bundle's provenance can record exactly what built
/// it. `l2` is added to every hessian, so it also shrinks leaf values (not
/// just split gains) and acts as the main guard against one-spike leaves when
/// no explicit `min_samples_leaf` constraint bites.
#[derive(Clone, Debug)]
pub struct Params {
    /// Boosting rounds. For softmax this means one tree per class per round.
    pub n_rounds: usize,
    /// Shrinkage applied to each tree's leaf values.
    pub learning_rate: f64,
    /// Maximum depth of any tree (root is depth 0).
    pub max_depth: usize,
    /// Maximum leaves per tree; growth is best-first across leaves.
    pub max_leaves: usize,
    /// Minimum weighted sample mass in each child of a split.
    pub min_samples_leaf: usize,
    /// L2 penalty added to every hessian.
    pub l2: f64,
    /// Reject splits whose gain does not exceed this value.
    pub min_gain: f64,
    /// Fraction of rows used for each round (deterministic, without
    /// replacement).
    pub row_subsample: f64,
    /// Fraction of features considered per tree (deterministic).
    pub col_subsample: f64,
    /// Maximum non-missing bins per feature; 255 keeps bins in a `u8` with
    /// room for the NaN sentinel. Must be at most 255.
    pub max_bins: usize,
    /// Seed for subsampling and binning.
    pub seed: u64,
    /// Optional validation-based early stopping.
    pub early_stopping: Option<EarlyStopping>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            n_rounds: 300,
            learning_rate: 0.05,
            max_depth: 6,
            max_leaves: 31,
            min_samples_leaf: 20,
            l2: 1.0,
            min_gain: 0.0,
            row_subsample: 1.0,
            col_subsample: 1.0,
            max_bins: 255,
            seed: 0,
            early_stopping: None,
        }
    }
}

/// One fitted regression tree, stored as parallel arrays.
///
/// Nodes are addressed by index; node `0` is the root. `split_feature < 0`
/// marks a leaf, in which case [`Tree::value`] is its additive prediction.
/// Internal nodes split on `x[feature] <= threshold`, and a NaN feature value
/// follows `default_left`. The raw thresholds (not bin indices) are stored so
/// a model can be scored without the training-time binning state, including
/// from a different language.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Tree {
    /// Split feature per node; `-1` for a leaf.
    pub split_feature: Vec<i32>,
    /// Raw threshold per node (`x <= threshold` goes left).
    pub threshold: Vec<f32>,
    /// Whether NaN follows the left branch.
    pub default_left: Vec<bool>,
    /// Left child node index per node.
    pub left: Vec<u32>,
    /// Right child node index per node.
    pub right: Vec<u32>,
    /// Leaf value per node (zero on internal nodes). Values already include
    /// the training learning-rate shrinkage, so prediction is a plain sum.
    pub value: Vec<f64>,
}

impl Tree {
    fn push_leaf(&mut self, value: f64) -> u32 {
        let node = self.value.len() as u32;
        self.split_feature.push(-1);
        self.threshold.push(0.0);
        self.default_left.push(false);
        self.left.push(0);
        self.right.push(0);
        self.value.push(value);
        node
    }

    fn set_internal(
        &mut self,
        node: u32,
        feature: usize,
        threshold: f32,
        default_left: bool,
        left: u32,
        right: u32,
    ) {
        let n = node as usize;
        self.split_feature[n] = feature as i32;
        self.threshold[n] = threshold;
        self.default_left[n] = default_left;
        self.left[n] = left;
        self.right[n] = right;
        self.value[n] = 0.0;
    }

    /// Score one row against raw (not binned) features. A row shorter than
    /// the model's feature count is treated as NaN for the absent tail, and a
    /// malformed tree yields NaN instead of looping forever.
    fn predict(&self, row: &[f32]) -> f64 {
        if self.split_feature.len() != self.value.len() {
            return f64::NAN;
        }
        let mut node = 0usize;
        for _ in 0..=self.value.len() {
            if node >= self.value.len() {
                return f64::NAN;
            }
            let feature = self.split_feature[node];
            if feature < 0 {
                return self.value[node];
            }
            let x = row.get(feature as usize).copied().unwrap_or(f32::NAN);
            let go_left = if x.is_nan() {
                self.default_left[node]
            } else {
                x <= self.threshold[node]
            };
            node = if go_left {
                self.left[node] as usize
            } else {
                self.right[node] as usize
            };
        }
        f64::NAN
    }

    fn validate(&self, n_features: usize) -> Result<()> {
        let nodes = self.value.len();
        ensure!(nodes > 0, "tree has no nodes");
        for (name, len) in [
            ("split_feature", self.split_feature.len()),
            ("threshold", self.threshold.len()),
            ("default_left", self.default_left.len()),
            ("left", self.left.len()),
            ("right", self.right.len()),
            ("value", self.value.len()),
        ] {
            ensure!(
                len == nodes,
                "tree {name} has {len} nodes, expected {nodes}"
            );
        }
        let mut seen = vec![false; nodes];
        seen[0] = true;
        let mut stack = vec![0usize];
        while let Some(node) = stack.pop() {
            ensure!(
                self.value[node].is_finite(),
                "tree node {node} has a non-finite value"
            );
            let feature = self.split_feature[node];
            if feature < 0 {
                continue;
            }
            ensure!(
                (feature as usize) < n_features,
                "tree node {node} splits on feature {feature}, model has {n_features}"
            );
            ensure!(
                self.threshold[node].is_finite(),
                "tree node {node} has a non-finite threshold"
            );
            for child in [self.left[node], self.right[node]] {
                let child = child as usize;
                ensure!(child < nodes, "tree node {node} points past the tree");
                ensure!(
                    !seen[child],
                    "tree node {child} is reachable twice (cycle or shared child)"
                );
                seen[child] = true;
                stack.push(child);
            }
        }
        ensure!(
            seen.iter().all(|&v| v),
            "tree has nodes unreachable from the root"
        );
        Ok(())
    }
}

/// A fitted model: base score, one or more trees per round, and gain
/// importance, all portable plain data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    /// Number of features each row must provide.
    pub n_features: usize,
    /// Objective the trees were fitted for.
    pub objective: Objective,
    /// Initial margin per output (one score, or one per class).
    pub base_score: Vec<f64>,
    /// Trees added per boosting round (1, or `n_classes` for softmax).
    pub trees_per_round: usize,
    /// Round-major trees; round `r`, output `k` lives at `r * trees_per_round + k`.
    pub trees: Vec<Tree>,
    /// Total split gain attributed to each input feature.
    pub importance: Vec<f64>,
}

impl Model {
    /// Number of raw outputs (1 for logistic/regression, `n_classes` for softmax).
    pub fn n_outputs(&self) -> usize {
        self.objective.n_outputs()
    }

    /// Number of boosting rounds that survived early stopping.
    pub fn n_rounds(&self) -> usize {
        self.trees.len() / self.trees_per_round.max(1)
    }

    /// Total gain per input feature, indexed by feature.
    pub fn feature_importance(&self) -> &[f64] {
        &self.importance
    }

    /// Structural validation, used when reading a model from a bundle.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.n_features > 0, "model has no input features");
        if let Objective::Softmax { n_classes } = self.objective {
            ensure!(
                n_classes >= 2,
                "softmax model has {n_classes} classes; at least two are required"
            );
        }
        let n_outputs = self.n_outputs();
        ensure!(n_outputs > 0, "model objective has no outputs");
        ensure!(self.trees_per_round > 0, "model has no trees per round");
        ensure!(
            self.trees_per_round == n_outputs,
            "model has {} trees per round for {n_outputs} outputs",
            self.trees_per_round
        );
        ensure!(
            self.base_score.len() == n_outputs,
            "model has {} base scores for {n_outputs} outputs",
            self.base_score.len()
        );
        ensure!(
            self.base_score.iter().all(|v| v.is_finite()),
            "model has a non-finite base score"
        );
        ensure!(
            self.importance.len() == self.n_features,
            "model importance has {} entries for {} features",
            self.importance.len(),
            self.n_features
        );
        ensure!(
            self.importance.iter().all(|v| v.is_finite()),
            "model has a non-finite feature importance"
        );
        ensure!(
            self.trees.len().is_multiple_of(self.trees_per_round),
            "model has {} trees, not a multiple of {} per round",
            self.trees.len(),
            self.trees_per_round
        );
        for (i, tree) in self.trees.iter().enumerate() {
            tree.validate(self.n_features)
                .map_err(|e| anyhow::anyhow!("tree {i}: {e}"))?;
        }
        Ok(())
    }

    fn predict_raw_into(&self, row: &[f32], out: &mut [f64]) {
        out.fill(0.0);
        let copied = self.base_score.len().min(out.len());
        out[..copied].copy_from_slice(&self.base_score[..copied]);
        let n_out = self.n_outputs();
        if n_out == 0 {
            return;
        }
        for (i, tree) in self.trees.iter().enumerate() {
            out[i % n_out] += tree.predict(row);
        }
    }

    /// Raw margins for one row: one value for logistic/regression, one per
    /// class for softmax.
    pub fn predict_raw(&self, row: &[f32]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_outputs().max(1)];
        self.predict_raw_into(row, &mut out);
        out
    }

    /// Probabilities for one row. Logistic returns one positive-class
    /// probability, softmax returns one probability per class, and squared
    /// error returns the point estimate as a single "probability" (there is
    /// no probabilistic interpretation for regression).
    pub fn predict_proba(&self, row: &[f32]) -> Vec<f64> {
        let raw = self.predict_raw(row);
        match self.objective {
            Objective::Logistic => vec![sigmoid(raw[0])],
            Objective::Softmax { n_classes } => softmax(&raw, n_classes),
            Objective::SquaredError => vec![raw[0]],
        }
    }

    /// The model's decision: 0/1 for logistic, the winning class index for
    /// softmax, and the predicted value for regression.
    pub fn predict(&self, row: &[f32]) -> f64 {
        match self.objective {
            Objective::Logistic => f64::from(self.predict_proba(row)[0] >= 0.5),
            Objective::Softmax { n_classes } => {
                let raw = self.predict_raw(row);
                argmax(&raw, n_classes) as f64
            }
            Objective::SquaredError => self.predict_raw(row)[0],
        }
    }

    fn check_batch(&self, x: &[f32], n_features: usize) -> Result<usize> {
        ensure!(n_features > 0, "n_features must be positive");
        ensure!(
            n_features == self.n_features,
            "batch has {n_features} features, model expects {}",
            self.n_features
        );
        ensure!(
            x.len().is_multiple_of(n_features),
            "batch length {} is not a multiple of {n_features} features",
            x.len()
        );
        Ok(x.len() / n_features)
    }

    /// Raw margins for a row-major batch, returned row-major as
    /// `n_rows × n_outputs`.
    pub fn predict_raw_batch(&self, x: &[f32], n_features: usize) -> Result<Vec<f64>> {
        let n = self.check_batch(x, n_features)?;
        let n_out = self.n_outputs().max(1);
        let mut out = vec![0.0; n * n_out];
        out.par_chunks_mut(n_out).enumerate().for_each(|(i, row)| {
            self.predict_raw_into(&x[i * n_features..(i + 1) * n_features], row);
        });
        Ok(out)
    }

    /// Probabilities for a row-major batch, returned row-major as
    /// `n_rows × n_proba` (see [`Model::predict_proba`]).
    pub fn predict_proba_batch(&self, x: &[f32], n_features: usize) -> Result<Vec<f64>> {
        let n = self.check_batch(x, n_features)?;
        let width = self.proba_width();
        let mut out = vec![0.0; n * width];
        out.par_chunks_mut(width).enumerate().for_each(|(i, row)| {
            let p = self.predict_proba(&x[i * n_features..(i + 1) * n_features]);
            row.copy_from_slice(&p);
        });
        Ok(out)
    }

    /// Decisions for a row-major batch (see [`Model::predict`]).
    pub fn predict_batch(&self, x: &[f32], n_features: usize) -> Result<Vec<f64>> {
        let n = self.check_batch(x, n_features)?;
        let mut out = vec![0.0; n];
        out.par_iter_mut().enumerate().for_each(|(i, v)| {
            *v = self.predict(&x[i * n_features..(i + 1) * n_features]);
        });
        Ok(out)
    }

    fn proba_width(&self) -> usize {
        match self.objective {
            Objective::Softmax { n_classes } => n_classes,
            Objective::Logistic | Objective::SquaredError => 1,
        }
    }
}

/// Fit a model. `x` is row-major with `n_features` columns and `y` holds one
/// label per row; `weights` defaults to one per row.
pub fn train(
    x: &[f32],
    n_features: usize,
    y: &[f32],
    weights: Option<&[f32]>,
    objective: Objective,
    params: &Params,
) -> Result<Model> {
    validate_params(params)?;
    ensure!(n_features > 0, "n_features must be positive");
    ensure!(!x.is_empty(), "training matrix is empty");
    ensure!(
        x.len().is_multiple_of(n_features),
        "training matrix length {} is not a multiple of {n_features} features",
        x.len()
    );
    let n = x.len() / n_features;
    ensure!(
        n <= u32::MAX as usize,
        "training set has too many rows ({n})"
    );
    ensure!(
        y.len() == n,
        "training labels have {} rows, features have {n}",
        y.len()
    );
    validate_labels(y, objective, "training")?;
    let w = make_weights(weights, n, "training")?;
    validate_features(x, n_features)?;

    let val_weights: Option<Vec<f64>> = match params.early_stopping.as_ref() {
        None => None,
        Some(es) => {
            ensure!(
                es.patience >= 1,
                "early-stopping patience must be at least 1"
            );
            ensure!(!es.x.is_empty(), "early-stopping validation set is empty");
            ensure!(
                es.x.len().is_multiple_of(n_features),
                "early-stopping features length {} is not a multiple of {n_features}",
                es.x.len()
            );
            let nv = es.x.len() / n_features;
            ensure!(
                es.y.len() == nv,
                "early-stopping labels have {} rows, features have {nv}",
                es.y.len()
            );
            validate_labels(&es.y, objective, "early-stopping")?;
            validate_features(&es.x, n_features)?;
            Some(make_weights(es.weights.as_deref(), nv, "early-stopping")?)
        }
    };
    let val_rows = params
        .early_stopping
        .as_ref()
        .map(|es| es.x.len() / n_features);

    let binned = BinData::fit(x, n, n_features, params.max_bins, params.seed);
    let base_score = base_scores(y, &w, objective)?;
    let n_out = objective.n_outputs();

    let mut raw = vec![0.0; n * n_out];
    if n_out == 1 {
        raw.fill(base_score[0]);
    } else {
        for row in raw.chunks_mut(n_out) {
            row.copy_from_slice(&base_score);
        }
    }
    let mut val_raw = val_rows.map(|nv| {
        let mut v = vec![0.0; nv * n_out];
        if n_out == 1 {
            v.fill(base_score[0]);
        } else {
            for row in v.chunks_mut(n_out) {
                row.copy_from_slice(&base_score);
            }
        }
        v
    });

    let mut trees = Vec::new();
    let mut round_importance: Vec<Vec<f64>> = Vec::new();
    // Early stopping judges round 0 too: the base-score-only model is the
    // candidate every round has to beat, and a zero-tree model is allowed.
    let mut best: Option<(f64, usize)> = match (
        params.early_stopping.as_ref(),
        val_raw.as_deref(),
        val_weights.as_deref(),
    ) {
        (Some(es), Some(val), Some(vw)) => Some((validation_loss(objective, val, &es.y, vw), 0)),
        _ => None,
    };
    let mut since_best = 0usize;
    let mut probs = if matches!(objective, Objective::Softmax { .. }) {
        vec![0.0; n * n_out]
    } else {
        Vec::new()
    };

    for round in 0..params.n_rounds {
        let bag = sample_rows(n, params.row_subsample, params.seed, round);
        let mut round_gain = vec![0.0; n_features];
        match objective {
            Objective::Logistic => {
                let (grad, hess) = logistic_gradients(&raw, y, &w);
                let features =
                    feature_subset(n_features, params.col_subsample, params.seed, round, 0);
                let grown = binned.grow_tree(&grad, &hess, &w, &bag, &features, params);
                apply_grown(&mut raw, 0, n_out, &grown, x, n_features);
                if let (Some(es), Some(val)) = (params.early_stopping.as_ref(), val_raw.as_mut()) {
                    add_tree_raw(val, 0, n_out, &grown.tree, &es.x, n_features);
                }
                add_gain(&mut round_gain, &grown.importance);
                trees.push(grown.tree);
            }
            Objective::Softmax { n_classes } => {
                softmax_probs(&raw, n_classes, &mut probs);
                for k in 0..n_classes {
                    let (grad, hess) = softmax_gradients(&probs, y, &w, k, n_classes);
                    let features =
                        feature_subset(n_features, params.col_subsample, params.seed, round, k);
                    let grown = binned.grow_tree(&grad, &hess, &w, &bag, &features, params);
                    apply_grown(&mut raw, k, n_out, &grown, x, n_features);
                    if let (Some(es), Some(val)) =
                        (params.early_stopping.as_ref(), val_raw.as_mut())
                    {
                        add_tree_raw(val, k, n_out, &grown.tree, &es.x, n_features);
                    }
                    add_gain(&mut round_gain, &grown.importance);
                    trees.push(grown.tree);
                }
            }
            Objective::SquaredError => {
                let (grad, hess) = squared_gradients(&raw, y, &w);
                let features =
                    feature_subset(n_features, params.col_subsample, params.seed, round, 0);
                let grown = binned.grow_tree(&grad, &hess, &w, &bag, &features, params);
                apply_grown(&mut raw, 0, n_out, &grown, x, n_features);
                if let (Some(es), Some(val)) = (params.early_stopping.as_ref(), val_raw.as_mut()) {
                    add_tree_raw(val, 0, n_out, &grown.tree, &es.x, n_features);
                }
                add_gain(&mut round_gain, &grown.importance);
                trees.push(grown.tree);
            }
        }
        round_importance.push(round_gain);

        if let (Some(es), Some(val), Some(vw)) = (
            params.early_stopping.as_ref(),
            val_raw.as_deref(),
            val_weights.as_deref(),
        ) {
            let loss = validation_loss(objective, val, &es.y, vw);
            let improved = match best {
                None => true,
                Some((best_loss, _)) => {
                    loss < best_loss - EARLY_STOP_EPS * best_loss.abs().max(1.0)
                }
            };
            if improved {
                best = Some((loss, round + 1));
                since_best = 0;
            } else {
                since_best += 1;
            }
            if since_best >= es.patience {
                break;
            }
        }
    }

    let kept_rounds = match (&val_rows, best) {
        (Some(_), Some((_, r))) => r,
        (Some(_), None) => 0,
        (None, _) => params.n_rounds,
    };
    trees.truncate(kept_rounds * n_out);
    let mut importance = vec![0.0; n_features];
    for round in round_importance.iter().take(kept_rounds) {
        for (total, gain) in importance.iter_mut().zip(round) {
            *total += *gain;
        }
    }

    let model = Model {
        n_features,
        objective,
        base_score,
        trees_per_round: n_out,
        trees,
        importance,
    };
    model.validate()?;
    Ok(model)
}

fn add_gain(total: &mut [f64], round: &[f64]) {
    for (t, g) in total.iter_mut().zip(round) {
        *t += *g;
    }
}

fn add_tree_raw(
    raw: &mut [f64],
    output: usize,
    n_out: usize,
    tree: &Tree,
    x: &[f32],
    n_features: usize,
) {
    raw.par_chunks_mut(n_out).enumerate().for_each(|(i, row)| {
        let base = i * n_features;
        row[output] += tree.predict(&x[base..base + n_features]);
    });
}

fn apply_grown(
    raw: &mut [f64],
    output: usize,
    n_out: usize,
    grown: &GrownTree,
    x: &[f32],
    n_features: usize,
) {
    raw.par_chunks_mut(n_out).enumerate().for_each(|(i, row)| {
        let added = match grown.leaf_of[i] {
            u32::MAX => grown.tree.predict(&x[i * n_features..(i + 1) * n_features]),
            node => grown.tree.value[node as usize],
        };
        row[output] += added;
    });
}

fn validate_params(params: &Params) -> Result<()> {
    ensure!(
        params.n_rounds <= 1_000_000,
        "n_rounds is unreasonably large"
    );
    ensure!(
        params.learning_rate.is_finite() && params.learning_rate > 0.0,
        "learning_rate must be finite and positive"
    );
    ensure!(params.max_depth >= 1, "max_depth must be at least 1");
    ensure!(params.max_depth <= 64, "max_depth must be at most 64");
    ensure!(params.max_leaves >= 1, "max_leaves must be at least 1");
    ensure!(
        params.max_leaves <= (1 << 20),
        "max_leaves is unreasonably large"
    );
    ensure!(
        params.min_samples_leaf >= 1,
        "min_samples_leaf must be at least 1"
    );
    ensure!(
        params.l2.is_finite() && params.l2 >= 0.0,
        "l2 must be finite and non-negative"
    );
    ensure!(params.min_gain.is_finite(), "min_gain must be finite");
    ensure!(
        params.row_subsample.is_finite()
            && params.row_subsample > 0.0
            && params.row_subsample <= 1.0,
        "row_subsample must be in (0, 1]"
    );
    ensure!(
        params.col_subsample.is_finite()
            && params.col_subsample > 0.0
            && params.col_subsample <= 1.0,
        "col_subsample must be in (0, 1]"
    );
    ensure!(
        (2..=255).contains(&params.max_bins),
        "max_bins must be in 2..=255"
    );
    Ok(())
}

fn validate_features(x: &[f32], n_features: usize) -> Result<()> {
    if let Some(pos) = x.iter().position(|v| v.is_infinite()) {
        bail!(
            "feature [{}, {}] is infinite; use NaN for missing values",
            pos / n_features,
            pos % n_features
        );
    }
    Ok(())
}

fn validate_labels(y: &[f32], objective: Objective, what: &str) -> Result<()> {
    match objective {
        Objective::Logistic => {
            let (mut neg, mut pos) = (0usize, 0usize);
            for (i, &v) in y.iter().enumerate() {
                ensure!(
                    v == 0.0 || v == 1.0,
                    "{what} label at row {i} is {v}; logistic labels must be 0 or 1"
                );
                if v == 0.0 {
                    neg += 1;
                } else {
                    pos += 1;
                }
            }
            ensure!(
                neg > 0 && pos > 0,
                "{what} labels contain only one class; logistic needs both 0 and 1"
            );
        }
        Objective::Softmax { n_classes } => {
            ensure!(n_classes >= 2, "softmax needs at least two classes");
            let mut counts = vec![0usize; n_classes];
            for (i, &v) in y.iter().enumerate() {
                ensure!(
                    v.is_finite() && v.fract() == 0.0 && v >= 0.0 && (v as usize) < n_classes,
                    "{what} label at row {i} is {v}; softmax labels must be whole numbers in 0..{n_classes}"
                );
                counts[v as usize] += 1;
            }
            ensure!(
                counts.iter().all(|&c| c > 0),
                "{what} labels are missing a class; softmax needs every class 0..{n_classes}"
            );
        }
        Objective::SquaredError => {
            for (i, &v) in y.iter().enumerate() {
                ensure!(v.is_finite(), "{what} label at row {i} is not finite");
            }
        }
    }
    Ok(())
}

fn make_weights(weights: Option<&[f32]>, n: usize, what: &str) -> Result<Vec<f64>> {
    match weights {
        None => Ok(vec![1.0; n]),
        Some(ws) => {
            ensure!(
                ws.len() == n,
                "{what} weights have {} rows, expected {n}",
                ws.len()
            );
            let mut out = Vec::with_capacity(n);
            let mut sum = 0.0;
            for (i, &w) in ws.iter().enumerate() {
                ensure!(w.is_finite(), "{what} weight at row {i} is not finite");
                ensure!(w >= 0.0, "{what} weight at row {i} is negative");
                out.push(w as f64);
                sum += w as f64;
            }
            ensure!(sum > 0.0, "{what} weights sum to zero");
            Ok(out)
        }
    }
}

fn base_scores(y: &[f32], w: &[f64], objective: Objective) -> Result<Vec<f64>> {
    let total: f64 = w.iter().sum();
    ensure!(total > 0.0, "sample weights sum to zero");
    match objective {
        Objective::Logistic => {
            let pos: f64 = w.iter().zip(y).map(|(&w, &y)| w * y as f64).sum();
            let p = (pos / total).clamp(PROB_EPS, 1.0 - PROB_EPS);
            Ok(vec![(p / (1.0 - p)).ln()])
        }
        Objective::Softmax { n_classes } => {
            let mut sums = vec![0.0; n_classes];
            for (&w, &y) in w.iter().zip(y) {
                sums[y as usize] += w;
            }
            Ok(sums
                .into_iter()
                .map(|s| (s / total).clamp(PROB_EPS, 1.0 - PROB_EPS).ln())
                .collect())
        }
        Objective::SquaredError => {
            let mean: f64 = w.iter().zip(y).map(|(&w, &y)| w * y as f64).sum();
            Ok(vec![mean / total])
        }
    }
}

fn logistic_gradients(raw: &[f64], y: &[f32], w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut grad = vec![0.0; raw.len()];
    let mut hess = vec![0.0; raw.len()];
    grad.par_iter_mut()
        .zip(hess.par_iter_mut())
        .enumerate()
        .for_each(|(i, (g, h))| {
            let p = sigmoid(raw[i]);
            *g = w[i] * (y[i] as f64 - p);
            *h = w[i] * p * (1.0 - p);
        });
    (grad, hess)
}

fn squared_gradients(raw: &[f64], y: &[f32], w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut grad = vec![0.0; raw.len()];
    let mut hess = vec![0.0; raw.len()];
    grad.par_iter_mut()
        .zip(hess.par_iter_mut())
        .enumerate()
        .for_each(|(i, (g, h))| {
            *g = w[i] * (y[i] as f64 - raw[i]);
            *h = w[i];
        });
    (grad, hess)
}

fn softmax_probs(raw: &[f64], n_classes: usize, out: &mut [f64]) {
    out.par_chunks_mut(n_classes)
        .enumerate()
        .for_each(|(i, probs)| {
            let z = &raw[i * n_classes..(i + 1) * n_classes];
            let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for (p, &v) in probs.iter_mut().zip(z) {
                *p = (v - max).exp();
                sum += *p;
            }
            for p in probs.iter_mut() {
                *p /= sum;
            }
        });
}

fn softmax_gradients(
    probs: &[f64],
    y: &[f32],
    w: &[f64],
    k: usize,
    n_classes: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut grad = vec![0.0; w.len()];
    let mut hess = vec![0.0; w.len()];
    grad.par_iter_mut()
        .zip(hess.par_iter_mut())
        .enumerate()
        .for_each(|(i, (g, h))| {
            let p = probs[i * n_classes + k];
            let target = f64::from((y[i] as usize) == k);
            *g = w[i] * (target - p);
            *h = w[i] * p * (1.0 - p);
        });
    (grad, hess)
}

fn validation_loss(objective: Objective, raw: &[f64], y: &[f32], weights: &[f64]) -> f64 {
    let mut loss = 0.0;
    let mut total = 0.0;
    match objective {
        Objective::Logistic => {
            for (i, &label) in y.iter().enumerate() {
                let p = sigmoid(raw[i]).clamp(1e-15, 1.0 - 1e-15);
                let target = label as f64;
                loss -= weights[i] * (target * p.ln() + (1.0 - target) * (1.0 - p).ln());
                total += weights[i];
            }
        }
        Objective::Softmax { n_classes } => {
            for (i, &label) in y.iter().enumerate() {
                let row = &raw[i * n_classes..(i + 1) * n_classes];
                let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = row.iter().map(|&v| (v - max).exp()).sum();
                let log_p = row[label as usize] - max - sum.ln();
                loss -= weights[i] * log_p;
                total += weights[i];
            }
        }
        Objective::SquaredError => {
            for (i, &label) in y.iter().enumerate() {
                let err = raw[i] - label as f64;
                loss += weights[i] * err * err;
                total += weights[i];
            }
        }
    }
    if total > 0.0 {
        loss / total
    } else {
        0.0
    }
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn softmax(raw: &[f64], n_classes: usize) -> Vec<f64> {
    let max = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out: Vec<f64> = raw.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = out.iter().sum();
    for p in &mut out {
        *p /= sum;
    }
    out.truncate(n_classes);
    out
}

fn argmax(raw: &[f64], n_classes: usize) -> usize {
    let mut best = 0usize;
    for k in 1..n_classes.min(raw.len()) {
        if raw[k] > raw[best] {
            best = k;
        }
    }
    best
}

/// Splitmix64; used for sampling so results do not depend on a platform RNG.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.0)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Select `k` distinct indices in `0..n` by a seeded partial Fisher-Yates
/// shuffle. Deterministic and independent of thread scheduling.
fn sample_indices(n: usize, k: usize, seed: u64) -> Vec<u32> {
    let k = k.min(n);
    let mut idx: Vec<u32> = (0..n as u32).collect();
    let mut rng = SplitMix64::new(seed);
    for i in 0..k {
        let j = i + rng.below(n - i);
        idx.swap(i, j);
    }
    idx.truncate(k);
    idx
}

fn sample_rows(n: usize, frac: f64, seed: u64, round: usize) -> Vec<u32> {
    if frac >= 1.0 {
        return (0..n as u32).collect();
    }
    let k = ((n as f64 * frac).round() as usize).clamp(1, n);
    sample_indices(n, k, mix64(seed ^ mix64(round as u64)))
}

fn feature_subset(d: usize, frac: f64, seed: u64, round: usize, output: usize) -> Vec<usize> {
    if frac >= 1.0 {
        return (0..d).collect();
    }
    let k = ((d as f64 * frac).round() as usize).clamp(1, d);
    let tree_seed = mix64(seed ^ mix64(round as u64) ^ mix64(0xA5A5_5A5A ^ output as u64));
    let mut subset: Vec<usize> = sample_indices(d, k, tree_seed)
        .into_iter()
        .map(|f| f as usize)
        .collect();
    subset.sort_unstable();
    subset
}

/// Binned training matrix, column-major so each feature is a contiguous
/// `u8` column. `missing_bin` is the NaN sentinel; `edges[f]` are the raw
/// thresholds that separate the non-missing bins.
struct BinData {
    n: usize,
    d: usize,
    bins: usize,
    missing_bin: usize,
    data: Vec<u8>,
    edges: Vec<Vec<f32>>,
}

impl BinData {
    fn fit(x: &[f32], n: usize, d: usize, max_bins: usize, seed: u64) -> Self {
        let sample = if n <= MAX_BIN_ROWS {
            (0..n as u32).collect()
        } else {
            sample_indices(n, MAX_BIN_ROWS, mix64(seed ^ 0xB1F_5EED))
        };
        let mut edges = Vec::with_capacity(d);
        let mut values = Vec::with_capacity(sample.len());
        for feature in 0..d {
            values.clear();
            for &row in &sample {
                let mut v = x[row as usize * d + feature];
                if v.is_nan() {
                    continue;
                }
                // Normalise signed zero so `x <= edges[b]` and the bin rule
                // agree on values at the boundary.
                if v == 0.0 {
                    v = 0.0;
                }
                values.push(v);
            }
            values.sort_unstable_by(f32::total_cmp);
            let mut feature_edges = Vec::new();
            for j in 1..max_bins {
                let idx = (j * values.len()) / max_bins;
                if idx >= values.len() {
                    continue;
                }
                let v = values[idx];
                if feature_edges.last().is_none_or(|&last| v > last) {
                    feature_edges.push(v);
                }
            }
            edges.push(feature_edges);
        }

        let missing_bin = max_bins;
        let missing = missing_bin as u8;
        let mut row_major = vec![missing; n * d];
        row_major
            .par_chunks_mut(d)
            .enumerate()
            .for_each(|(i, row)| {
                for (feature, cell) in row.iter_mut().enumerate() {
                    let mut v = x[i * d + feature];
                    if v == 0.0 {
                        v = 0.0;
                    }
                    *cell = if v.is_nan() {
                        missing
                    } else {
                        edges[feature].partition_point(|&e| e < v) as u8
                    };
                }
            });
        let mut data = vec![0u8; n * d];
        data.par_chunks_mut(n).enumerate().for_each(|(f, col)| {
            for (i, cell) in col.iter_mut().enumerate() {
                *cell = row_major[i * d + f];
            }
        });
        Self {
            n,
            d,
            bins: max_bins + 1,
            missing_bin,
            data,
            edges,
        }
    }

    fn accumulate_feature(
        &self,
        feature: usize,
        cell: &mut [f64],
        rows: &[u32],
        grad: &[f64],
        hess: &[f64],
        w: &[f64],
    ) {
        let start = feature * self.n;
        let col = &self.data[start..start + self.n];
        for &row in rows {
            let i = row as usize;
            let slot = col[i] as usize * HIST_STATS;
            cell[slot] += grad[i];
            cell[slot + 1] += hess[i];
            cell[slot + 2] += w[i];
        }
    }

    fn build_hist(
        &self,
        rows: &[u32],
        grad: &[f64],
        hess: &[f64],
        w: &[f64],
        mask: &[bool],
    ) -> Vec<f64> {
        let stride = self.bins * HIST_STATS;
        let mut hist = vec![0.0; self.d * stride];
        if rows.len().saturating_mul(self.d) >= PAR_MIN_PAIRS {
            hist.par_chunks_mut(stride)
                .enumerate()
                .for_each(|(feature, cell)| {
                    if mask[feature] {
                        self.accumulate_feature(feature, cell, rows, grad, hess, w);
                    }
                });
        } else {
            hist.chunks_mut(stride)
                .enumerate()
                .for_each(|(feature, cell)| {
                    if mask[feature] {
                        self.accumulate_feature(feature, cell, rows, grad, hess, w);
                    }
                });
        }
        hist
    }

    fn find_split(
        &self,
        hist: &[f64],
        features: &[usize],
        totals: Totals,
        params: &Params,
    ) -> Option<Split> {
        let stride = self.bins * HIST_STATS;
        let min_mass = params.min_samples_leaf as f64;
        let parent_den = totals.h + params.l2;
        let parent_score = if parent_den > 0.0 {
            totals.g * totals.g / parent_den
        } else {
            0.0
        };
        let mut best: Option<Split> = None;
        for &feature in features {
            if self.edges[feature].is_empty() {
                continue;
            }
            let base = feature * stride;
            let missing_slot = base + self.missing_bin * HIST_STATS;
            let (missing_g, missing_h, missing_w) = (
                hist[missing_slot],
                hist[missing_slot + 1],
                hist[missing_slot + 2],
            );
            // A missing value is its own group: try sending it left, then
            // right, and let the gain pick the direction. The raw threshold is
            // the edge that ends the left group, so bins and raw features
            // agree at prediction time.
            for missing_left in [true, false] {
                let (mut gl, mut hl, mut wl) = if missing_left {
                    (missing_g, missing_h, missing_w)
                } else {
                    (0.0, 0.0, 0.0)
                };
                let (mut gr, mut hr, mut wr) = (totals.g - gl, totals.h - hl, totals.w - wl);
                for (b, &edge) in self.edges[feature].iter().enumerate() {
                    let slot = base + b * HIST_STATS;
                    gl += hist[slot];
                    hl += hist[slot + 1];
                    wl += hist[slot + 2];
                    gr -= hist[slot];
                    hr -= hist[slot + 1];
                    wr -= hist[slot + 2];
                    if wl < min_mass || wr < min_mass {
                        continue;
                    }
                    let left_den = hl + params.l2;
                    let right_den = hr + params.l2;
                    if left_den <= 0.0 || right_den <= 0.0 {
                        continue;
                    }
                    let gain = 0.5 * (gl * gl / left_den + gr * gr / right_den - parent_score);
                    if gain > params.min_gain && best.is_none_or(|s| gain > s.gain) {
                        best = Some(Split {
                            feature,
                            bin: b as u8,
                            threshold: edge,
                            default_left: missing_left,
                            gain,
                        });
                    }
                }
            }
        }
        best
    }

    fn partition(&self, rows: &[u32], split: &Split) -> (Vec<u32>, Vec<u32>) {
        let start = split.feature * self.n;
        let col = &self.data[start..start + self.n];
        let mut left = Vec::with_capacity(rows.len() / 2);
        let mut right = Vec::with_capacity(rows.len() / 2);
        for &row in rows {
            let bin = col[row as usize] as usize;
            let go_left = if bin == self.missing_bin {
                split.default_left
            } else {
                bin <= split.bin as usize
            };
            if go_left {
                left.push(row);
            } else {
                right.push(row);
            }
        }
        (left, right)
    }

    fn leaf_value(totals: Totals, params: &Params) -> f64 {
        let den = totals.h + params.l2;
        if den > 0.0 {
            params.learning_rate * totals.g / den
        } else {
            0.0
        }
    }

    fn grow_tree(
        &self,
        grad: &[f64],
        hess: &[f64],
        w: &[f64],
        rows: &[u32],
        features: &[usize],
        params: &Params,
    ) -> GrownTree {
        let mut mask = vec![false; self.d];
        for &f in features {
            mask[f] = true;
        }
        let mut importance = vec![0.0; self.d];
        let mut tree = Tree::default();
        let totals = Totals::from_rows(rows, grad, hess, w);
        let root_value = Self::leaf_value(totals, params);
        tree.push_leaf(root_value);

        let root_hist = if params.max_leaves > 1 {
            self.build_hist(rows, grad, hess, w, &mask)
        } else {
            Vec::new()
        };
        let root_split = if params.max_leaves > 1 {
            self.find_split(&root_hist, features, totals, params)
        } else {
            None
        };
        let mut leaves = vec![LeafState {
            rows: rows.to_vec(),
            depth: 0,
            node: 0,
            hist: root_hist,
            split: root_split,
        }];
        let mut heap = BinaryHeap::new();
        if let Some(split) = root_split {
            heap.push(Candidate {
                gain: split.gain,
                seq: 0,
                leaf: 0,
            });
        }
        let mut leaf_count = 1usize;
        let mut next_seq: u64 = 1;

        loop {
            if leaf_count >= params.max_leaves {
                break;
            }
            let Some(candidate) = heap.pop() else {
                break;
            };
            let (parent_rows, parent_hist, depth, node, split) = {
                let leaf = &mut leaves[candidate.leaf];
                let Some(split) = leaf.split.take() else {
                    continue;
                };
                (
                    std::mem::take(&mut leaf.rows),
                    std::mem::take(&mut leaf.hist),
                    leaf.depth,
                    leaf.node,
                    split,
                )
            };
            let (left_rows, right_rows) = self.partition(&parent_rows, &split);
            if left_rows.is_empty() || right_rows.is_empty() {
                // Zero-weight rows can still be present; a degenerate split
                // would otherwise create an unreachable leaf.
                continue;
            }
            let left_totals = Totals::from_rows(&left_rows, grad, hess, w);
            let right_totals = Totals::from_rows(&right_rows, grad, hess, w);
            let left_node = tree.push_leaf(Self::leaf_value(left_totals, params));
            let right_node = tree.push_leaf(Self::leaf_value(right_totals, params));
            tree.set_internal(
                node,
                split.feature,
                split.threshold,
                split.default_left,
                left_node,
                right_node,
            );
            importance[split.feature] += split.gain;
            leaf_count += 1;

            // Rebuild the smaller child from its rows and recover the larger
            // child's histogram by subtraction. This keeps tree growth at
            // roughly one histogram pass per level instead of one per node.
            let small_is_left = left_rows.len() <= right_rows.len();
            let small_rows = if small_is_left {
                &left_rows
            } else {
                &right_rows
            };
            let small_hist = self.build_hist(small_rows, grad, hess, w, &mask);
            let mut large_hist = parent_hist;
            for (p, s) in large_hist.iter_mut().zip(&small_hist) {
                *p -= *s;
            }
            let (left_hist, right_hist) = if small_is_left {
                (small_hist, large_hist)
            } else {
                (large_hist, small_hist)
            };

            let child_depth = depth + 1;
            let mut push_child = |child_rows: Vec<u32>,
                                  child_hist: Vec<f64>,
                                  child_totals: Totals,
                                  child_node: u32| {
                let child_split = if child_depth < params.max_depth {
                    self.find_split(&child_hist, features, child_totals, params)
                } else {
                    None
                };
                let idx = leaves.len();
                leaves.push(LeafState {
                    rows: child_rows,
                    depth: child_depth,
                    node: child_node,
                    hist: child_hist,
                    split: child_split,
                });
                if let Some(s) = child_split {
                    heap.push(Candidate {
                        gain: s.gain,
                        seq: next_seq,
                        leaf: idx,
                    });
                    next_seq += 1;
                }
            };
            push_child(left_rows, left_hist, left_totals, left_node);
            push_child(right_rows, right_hist, right_totals, right_node);
        }

        let mut leaf_of = vec![u32::MAX; self.n];
        for leaf in &leaves {
            if tree.split_feature[leaf.node as usize] < 0 {
                for &row in &leaf.rows {
                    leaf_of[row as usize] = leaf.node;
                }
            }
        }
        GrownTree {
            tree,
            leaf_of,
            importance,
        }
    }
}

impl std::fmt::Debug for BinData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinData")
            .field("n", &self.n)
            .field("d", &self.d)
            .field("bins", &self.bins)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Totals {
    g: f64,
    h: f64,
    w: f64,
}

impl Totals {
    fn from_rows(rows: &[u32], grad: &[f64], hess: &[f64], w: &[f64]) -> Self {
        let mut totals = Self::default();
        for &row in rows {
            let i = row as usize;
            totals.g += grad[i];
            totals.h += hess[i];
            totals.w += w[i];
        }
        totals
    }
}

#[derive(Clone, Copy, Debug)]
struct Split {
    feature: usize,
    bin: u8,
    threshold: f32,
    default_left: bool,
    gain: f64,
}

struct LeafState {
    rows: Vec<u32>,
    depth: usize,
    node: u32,
    hist: Vec<f64>,
    split: Option<Split>,
}

struct GrownTree {
    tree: Tree,
    /// Final leaf node per training row, or `u32::MAX` when the row was not
    /// in this round's bag. Lets training update predictions in O(rows)
    /// instead of re-traversing every tree, without affecting the model.
    leaf_of: Vec<u32>,
    importance: Vec<f64>,
}

struct Candidate {
    gain: f64,
    seq: u64,
    leaf: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain
            .total_cmp(&other.gain)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::test_support::Rng;

    fn split_xy(
        x: Vec<f32>,
        y: Vec<f32>,
        d: usize,
        train: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            x[..train * d].to_vec(),
            y[..train].to_vec(),
            x[train * d..].to_vec(),
            y[train..].to_vec(),
        )
    }

    fn accuracy(model: &Model, x: &[f32], y: &[f32], d: usize, classes: usize) -> f64 {
        let mut hits = 0usize;
        for (row, &label) in x.chunks_exact(d).zip(y) {
            let raw = model.predict_raw(row);
            let pred = if classes == 2 {
                usize::from(sigmoid(raw[0]) >= 0.5)
            } else {
                argmax(&raw, classes)
            };
            hits += usize::from(pred == label as usize);
        }
        hits as f64 / y.len() as f64
    }

    fn xor_rows(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = Rng::new(seed);
        let mut x = Vec::with_capacity(n * 2);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let sx = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
            let sy = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
            x.push((sx + 0.35 * rng.normal()) as f32);
            x.push((sy + 0.35 * rng.normal()) as f32);
            y.push(if sx * sy > 0.0 { 1.0 } else { 0.0 });
        }
        (x, y)
    }

    #[test]
    fn xor_checkerboard_is_learned() {
        let (x, y) = xor_rows(6000, 11);
        let (xt, yt, xv, yv) = split_xy(x, y, 2, 5000);
        let params = Params::default();
        let model = train(&xt, 2, &yt, None, Objective::Logistic, &params).unwrap();
        let acc = accuracy(&model, &xv, &yv, 2, 2);
        assert!(acc > 0.95, "XOR accuracy {acc:.3} <= 0.95");
        assert_eq!(model.feature_importance().len(), 2);
        assert!(model.feature_importance().iter().all(|&g| g >= 0.0));
    }

    #[test]
    fn softmax_separates_three_blobs() {
        let mut rng = Rng::new(23);
        let centers = [(-2.5f32, 0.0f32), (2.5, 0.0), (0.0, 3.0)];
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for (class, &(cx, cy)) in centers.iter().enumerate() {
            for _ in 0..800 {
                x.push(cx + 0.7 * rng.normal() as f32);
                x.push(cy + 0.7 * rng.normal() as f32);
                y.push(class as f32);
            }
        }
        let (xt, yt, xv, yv) = split_xy(x, y, 2, 2000);
        let model = train(
            &xt,
            2,
            &yt,
            None,
            Objective::Softmax { n_classes: 3 },
            &Params::default(),
        )
        .unwrap();
        let acc = accuracy(&model, &xv, &yv, 2, 3);
        assert!(acc > 0.95, "blob accuracy {acc:.3} <= 0.95");
        for row in xv.as_chunks::<2>().0.iter().take(50) {
            let p = model.predict_proba(row);
            assert_eq!(p.len(), 3);
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(p.iter().all(|&v| (0.0..=1.0).contains(&v)));
        }
        let proba = model.predict_proba_batch(&xv, 2).unwrap();
        assert_eq!(proba.len(), yv.len() * 3);
        for row in proba.as_chunks::<3>().0 {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn squared_error_fits_smooth_function() {
        let mut rng = Rng::new(31);
        let n = 5000;
        let mut x = Vec::with_capacity(n * 3);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let a = rng.next_f64() * 6.0 - 3.0;
            let b = rng.next_f64() * 6.0 - 3.0;
            x.push(a as f32);
            x.push(b as f32);
            x.push(rng.next_f64() as f32);
            y.push((a.sin() + b * b + 0.05 * rng.normal()) as f32);
        }
        let (xt, yt, xv, yv) = split_xy(x, y, 3, 4500);
        let params = Params {
            n_rounds: 400,
            learning_rate: 0.08,
            ..Params::default()
        };
        let model = train(&xt, 3, &yt, None, Objective::SquaredError, &params).unwrap();
        let mean = yv.iter().map(|&v| v as f64).sum::<f64>() / yv.len() as f64;
        let mut sse = 0.0;
        let mut sst = 0.0;
        for (row, &label) in xv.as_chunks::<3>().0.iter().zip(&yv) {
            let pred = model.predict(row);
            sse += (pred - label as f64).powi(2);
            sst += (label as f64 - mean).powi(2);
        }
        let r2 = 1.0 - sse / sst;
        assert!(r2 > 0.9, "R^2 {r2:.3} <= 0.9");
    }

    #[test]
    fn nan_is_learned_as_a_default_direction() {
        let mut rng = Rng::new(41);
        let n = 3000;
        let mut x = Vec::with_capacity(n * 2);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let label = rng.next_f64() < 0.5;
            x.push(0.5 * rng.normal() as f32);
            x.push(if label { f32::NAN } else { rng.normal() as f32 });
            y.push(f32::from(label));
        }
        let (xt, yt, xv, yv) = split_xy(x, y, 2, 2400);
        let model = train(&xt, 2, &yt, None, Objective::Logistic, &Params::default()).unwrap();
        let acc = accuracy(&model, &xv, &yv, 2, 2);
        assert!(
            acc > 0.95,
            "NaN default-direction accuracy {acc:.3} <= 0.95"
        );
    }

    #[test]
    fn duplicated_rows_match_doubling_weights() {
        let mut rng = Rng::new(53);
        let n = 800;
        let d = 3;
        let mut x = Vec::with_capacity(n * d);
        let mut y = Vec::with_capacity(n);
        let mut w = Vec::with_capacity(n);
        for _ in 0..n {
            for _ in 0..d {
                // A small discrete grid keeps bin edges identical when rows
                // are duplicated, so the comparison isolates the weights.
                x.push((rng.below(6) as f32) - 2.5);
            }
            y.push(f32::from(rng.next_f64() < 0.5));
            w.push(if rng.next_f64() < 0.3 { 2.0 } else { 1.0 });
        }
        let mut xd = Vec::with_capacity(x.len() * 2);
        let mut yd = Vec::with_capacity(n * 2);
        let mut wd = Vec::with_capacity(n * 2);
        for i in 0..n {
            let copies = if w[i] == 2.0 { 2 } else { 1 };
            for _ in 0..copies {
                xd.extend_from_slice(&x[i * d..(i + 1) * d]);
                yd.push(y[i]);
                wd.push(1.0);
            }
        }
        let params = Params {
            n_rounds: 80,
            ..Params::default()
        };
        let weighted = train(&x, d, &y, Some(&w), Objective::Logistic, &params).unwrap();
        let duplicated = train(&xd, d, &yd, Some(&wd), Objective::Logistic, &params).unwrap();
        let mut rng = Rng::new(59);
        let mut max_diff = 0.0f64;
        for _ in 0..300 {
            let row: Vec<f32> = (0..d).map(|_| (rng.below(6) as f32) - 2.5).collect();
            let a = weighted.predict_proba(&row)[0];
            let b = duplicated.predict_proba(&row)[0];
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 1e-3,
            "duplicate/weight predictions differ by {max_diff}"
        );
    }

    #[test]
    fn same_seed_is_bit_identical_across_thread_counts() {
        let mut rng = Rng::new(67);
        let n = 20_000;
        let d = 12;
        let mut x = Vec::with_capacity(n * d);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let mut z = 0.0;
            for _ in 0..d {
                z += 0.15 * rng.normal();
                x.push(rng.normal() as f32);
            }
            y.push(f32::from(0.35 * z + 0.5 * rng.normal() > 0.0));
        }
        let params = Params {
            n_rounds: 25,
            learning_rate: 0.1,
            col_subsample: 0.7,
            row_subsample: 0.8,
            seed: 907,
            ..Params::default()
        };
        let pool1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let pool4 = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let a = pool1.install(|| train(&x, d, &y, None, Objective::Logistic, &params).unwrap());
        let b = pool4.install(|| train(&x, d, &y, None, Objective::Logistic, &params).unwrap());
        assert_eq!(
            rmp_serde::to_vec_named(&a).unwrap(),
            rmp_serde::to_vec_named(&b).unwrap(),
            "thread count changed the fitted model"
        );
        assert_eq!(a.importance, b.importance);
    }

    #[test]
    fn early_stopping_restores_best_round() {
        let (x, y) = xor_rows(3000, 71);
        let (xt, yt, xv, yv) = split_xy(x, y, 2, 2400);
        let params = Params {
            n_rounds: 2000,
            learning_rate: 0.2,
            early_stopping: Some(EarlyStopping {
                x: xv,
                y: yv,
                weights: None,
                patience: 10,
            }),
            ..Params::default()
        };
        let model = train(&xt, 2, &yt, None, Objective::Logistic, &params).unwrap();
        assert!(
            model.n_rounds() < 2000,
            "early stopping did not fire ({} rounds)",
            model.n_rounds()
        );
        assert!(model.n_rounds() >= 1);
    }

    #[test]
    fn early_stopping_can_keep_the_base_score_only_model() {
        // Training is balanced, so the base score is p=0.5. Validation labels
        // are the opposite of the fitted direction on both sides of the split,
        // so every round makes the validation loss worse: the base-score-only
        // model must be kept and a zero-tree model is allowed.
        let mut x = Vec::with_capacity(200);
        let mut y = Vec::with_capacity(200);
        for i in 0..200 {
            x.push(if i < 100 { 0.0f32 } else { 1.0 });
            y.push(if i < 100 { 0.0f32 } else { 1.0 });
        }
        let mut xv = Vec::with_capacity(200);
        let mut yv = Vec::with_capacity(200);
        for i in 0..200 {
            xv.push(if i < 100 { 0.0f32 } else { 1.0 });
            yv.push(if i < 100 { 1.0f32 } else { 0.0 });
        }
        let params = Params {
            n_rounds: 10,
            learning_rate: 0.5,
            min_samples_leaf: 5,
            l2: 0.0,
            early_stopping: Some(EarlyStopping {
                x: xv,
                y: yv,
                weights: None,
                patience: 2,
            }),
            ..Params::default()
        };
        let model = train(&x, 1, &y, None, Objective::Logistic, &params).unwrap();
        assert!(model.trees.is_empty(), "kept {} trees", model.trees.len());
        assert_eq!(model.n_rounds(), 0);
        assert!(model.importance.iter().all(|&g| g == 0.0));
        // Prediction still works from the base score alone.
        let p = model.predict_proba(&[1.0])[0];
        assert!((p - 0.5).abs() < 1e-9, "{p}");
    }

    #[test]
    fn model_validation_rejects_inconsistent_or_non_finite_models() {
        let valid = || Model {
            n_features: 2,
            objective: Objective::Logistic,
            base_score: vec![0.0],
            trees_per_round: 1,
            trees: Vec::new(),
            importance: vec![0.0, 0.0],
        };
        assert!(valid().validate().is_ok());

        let mut m = valid();
        m.objective = Objective::Softmax { n_classes: 0 };
        let err = m.validate().unwrap_err().to_string();
        assert!(err.contains("at least two"), "{err}");

        let mut m = valid();
        m.objective = Objective::Softmax { n_classes: 1 };
        assert!(m.validate().is_err());

        let mut m = valid();
        m.trees_per_round = 2;
        let err = m.validate().unwrap_err().to_string();
        assert!(err.contains("trees per round"), "{err}");

        let mut m = valid();
        m.base_score = vec![f64::NAN];
        let err = m.validate().unwrap_err().to_string();
        assert!(err.contains("base score"), "{err}");

        let mut m = valid();
        m.importance[1] = f64::INFINITY;
        let err = m.validate().unwrap_err().to_string();
        assert!(err.contains("importance"), "{err}");
    }

    #[test]
    fn invalid_inputs_are_errors_not_panics() {
        let x = [0.0f32, 1.0, 1.0, 0.0];
        let y = [0.0f32, 2.0];
        assert!(train(&x, 2, &y, None, Objective::Logistic, &Params::default()).is_err());
        assert!(train(
            &x[..3],
            2,
            &y[..1],
            None,
            Objective::Logistic,
            &Params::default()
        )
        .is_err());
        assert!(train(
            &x,
            2,
            &[0.0, 1.0],
            Some(&[1.0, f32::NAN]),
            Objective::Logistic,
            &Params::default()
        )
        .is_err());
        assert!(train(
            &[0.0, f32::INFINITY, 1.0, 0.0],
            2,
            &[0.0, 1.0],
            None,
            Objective::Logistic,
            &Params::default()
        )
        .is_err());
        assert!(train(
            &x,
            2,
            &[0.0, 0.0],
            None,
            Objective::Logistic,
            &Params::default()
        )
        .is_err());
        assert!(train(
            &x,
            2,
            &[0.0, 1.0],
            None,
            Objective::Softmax { n_classes: 3 },
            &Params::default()
        )
        .is_err());
        let empty: [f32; 0] = [];
        assert!(train(
            &empty,
            2,
            &[],
            None,
            Objective::SquaredError,
            &Params::default()
        )
        .is_err());
        let params = Params {
            max_bins: 256,
            ..Params::default()
        };
        assert!(train(&x, 2, &[0.0, 1.0], None, Objective::Logistic, &params).is_err());
    }

    #[test]
    fn batch_prediction_matches_single_rows() {
        let (x, y) = xor_rows(500, 73);
        let model = train(
            &x,
            2,
            &y,
            None,
            Objective::Logistic,
            &Params {
                n_rounds: 20,
                ..Params::default()
            },
        )
        .unwrap();
        let raw = model.predict_raw_batch(&x, 2).unwrap();
        assert_eq!(raw.len(), x.len() / 2);
        for (i, row) in x.as_chunks::<2>().0.iter().enumerate() {
            assert_eq!(raw[i], model.predict_raw(row)[0]);
        }
        let proba = model.predict_proba_batch(&x, 2).unwrap();
        assert_eq!(proba.len(), x.len() / 2);
        for (i, row) in x.as_chunks::<2>().0.iter().enumerate() {
            assert!((proba[i] - model.predict_proba(row)[0]).abs() < 1e-12);
        }
        let pred = model.predict_batch(&x, 2).unwrap();
        for (i, row) in x.as_chunks::<2>().0.iter().enumerate() {
            assert_eq!(pred[i], model.predict(row));
        }
        assert!(model.predict_batch(&x, 3).is_err());
    }

    /// Not part of the normal suite: `cargo test --release -p sorrel-data
    /// learn::gbdt::tests::benchmark -- --ignored --nocapture`.
    #[test]
    #[ignore = "slow 1,000,000 x 100 training benchmark"]
    fn benchmark_million_by_hundred() {
        use std::time::Instant;

        let n = 1_000_000usize;
        let d = 100usize;
        let mut rng = Rng::new(1_234_567);
        let mut x = Vec::with_capacity(n * d);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            if i % 100_000 == 0 {
                eprintln!("generating rows {i}/{n}");
            }
            let mut score = 0.0;
            for f in 0..d {
                let v = rng.normal();
                if f < 4 {
                    score += v * (f as f64 + 1.0) * 0.35;
                }
                x.push(v as f32);
            }
            score += 0.5 * rng.normal();
            y.push(f32::from(score > 0.0));
        }
        let t = Instant::now();
        let model = train(&x, d, &y, None, Objective::Logistic, &Params::default()).unwrap();
        let elapsed = t.elapsed();
        let raw_bytes = rmp_serde::to_vec_named(&model).unwrap().len();
        let names: Vec<String> = (0..d).map(|f| format!("f{f}")).collect();
        let mut bundle =
            crate::learn::bundle::CuratorBundle::new(crate::learn::bundle::Provenance::new(
                0,
                env!("CARGO_PKG_VERSION"),
                None,
                "bench-extractor-v1",
            ));
        bundle.add_schema(
            "bench",
            crate::learn::bundle::FeatureSchema::new(names, "bench-extractor-v1"),
        );
        bundle.add_head(
            "bench",
            crate::learn::bundle::Head::new("bench", model.clone()),
        );
        let bundle_bytes = bundle.to_bytes().unwrap().len();
        eprintln!(
            "trained {} rounds, {} trees, {} nodes in {:.1}s ({} rows x {} features)",
            model.n_rounds(),
            model.trees.len(),
            model.trees.iter().map(|t| t.value.len()).sum::<usize>(),
            elapsed.as_secs_f64(),
            n,
            d
        );
        eprintln!("model MessagePack size: {raw_bytes} bytes");
        eprintln!("compressed .sorrel-model size: {bundle_bytes} bytes");
    }
}

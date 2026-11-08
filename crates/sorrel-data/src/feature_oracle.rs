//! DEV-only representability diagnostic. Human truth fits a transparent axis
//! cut on alternating training spikes; the fixed cut is evaluated separately.
//! Never call this from a proposer, MCP server, or production classifier.

const MAX_PER_CLASS: usize = 2500;

#[derive(Clone, Debug)]
pub struct AxisAudit {
    pub at: f32,
    pub above: bool,
    pub train_jaccard: f64,
    pub validation_jaccard: f64,
    pub full_jaccard: f64,
    pub sampled_training: usize,
    pub predicted_validation: usize,
    pub hit_validation: usize,
}

/// Original global parity determines train/validation, not local root parity.
/// Global truth totals include a unit's spikes assigned to other roots.
pub fn audit_axis(
    values: &[f32],
    target: &[bool],
    training: &[bool],
    truth_totals: [usize; 3],
) -> Option<AxisAudit> {
    if values.len() != target.len() || target.len() != training.len() || truth_totals[0] == 0 {
        return None;
    }
    let mut classes = [Vec::new(), Vec::new()];
    for i in 0..values.len() {
        if training[i] && values[i].is_finite() {
            classes[usize::from(target[i])].push(i);
        }
    }
    if classes[1].is_empty() {
        return None;
    }
    // Bound sorting while preserving actual training class prevalence in the
    // objective: balancing the sample must not turn a .1% unit into a 50% one.
    let mut sample = Vec::new();
    for (label, indices) in classes.iter().enumerate() {
        let m = indices.len().min(MAX_PER_CLASS);
        if m == 0 {
            continue;
        }
        let weight = indices.len() as f64 / m as f64;
        for j in 0..m {
            let i = indices[j * indices.len() / m];
            sample.push((values[i], label == 1, weight));
        }
    }
    sample.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total_tp: f64 = sample.iter().filter(|p| p.1).map(|p| p.2).sum();
    let total_fp: f64 = sample.iter().filter(|p| !p.1).map(|p| p.2).sum();
    let (mut tp, mut fp, mut cursor) = (0.0, 0.0, 0);
    let mut best: Option<(f64, f32, bool)> = None;
    while cursor < sample.len() {
        let value = sample[cursor].0;
        while cursor < sample.len() && sample[cursor].0 == value {
            let (_, label, w) = sample[cursor];
            if label {
                tp += w;
            } else {
                fp += w;
            }
            cursor += 1;
        }
        for (hits, false_pos, above) in [(tp, fp, false), (total_tp - tp, total_fp - fp, true)] {
            let denominator = truth_totals[0] as f64 + false_pos.max(0.0);
            let score = hits.max(0.0) / denominator;
            if best.is_none_or(|b| score > b.0) {
                best = Some((score, value, above));
            }
        }
    }
    let (score, at, above) = best?;
    let (mut val_hit, mut val_pred, mut full_hit, mut full_pred) = (0, 0, 0, 0);
    for i in 0..values.len() {
        let selected = values[i].is_finite()
            && if above {
                values[i] > at
            } else {
                values[i] <= at
            };
        if selected {
            full_pred += 1;
            full_hit += usize::from(target[i]);
            if !training[i] {
                val_pred += 1;
                val_hit += usize::from(target[i]);
            }
        }
    }
    let ratio = |hit: usize, pred: usize, truth: usize| {
        let union = pred + truth - hit;
        if union > 0 {
            hit as f64 / union as f64
        } else {
            0.0
        }
    };
    Some(AxisAudit {
        at,
        above,
        train_jaccard: score,
        validation_jaccard: ratio(val_hit, val_pred, truth_totals[1]),
        full_jaccard: ratio(full_hit, full_pred, truth_totals[2]),
        sampled_training: sample.len(),
        predicted_validation: val_pred,
        hit_validation: val_hit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fits_training_only_and_does_not_choose_from_validation_labels() {
        let x = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let train = [true, false, true, false, true, false, true, false];
        let labels = [false, false, false, false, true, true, true, true];
        let a = audit_axis(&x, &labels, &train, [2, 2, 4]).unwrap();
        assert!(a.above);
        assert_eq!(a.at, 1.0);
        assert_eq!(a.validation_jaccard, 1.0);
        let opposite = [false, true, false, true, true, false, true, false];
        let b = audit_axis(&x, &opposite, &train, [2, 2, 4]).unwrap();
        assert_eq!(
            (a.at, a.above, a.train_jaccard),
            (b.at, b.above, b.train_jaccard)
        );
        assert_eq!(b.validation_jaccard, 0.0);
    }
    #[test]
    fn preserves_rare_class_weights_and_global_truth_denominator() {
        let x: Vec<_> = (0..20_000)
            .map(|i| if i % 1000 == 0 { 10.0 } else { 0.0 })
            .collect();
        let labels: Vec<_> = (0..20_000).map(|i| i % 1000 == 0).collect();
        let result = audit_axis(&x, &labels, &vec![true; 20_000], [40, 0, 40]).unwrap();
        assert!(result.sampled_training <= 5000);
        assert!((result.train_jaccard - 0.5).abs() < 1e-10);
        assert_eq!(
            result.full_jaccard, 0.5,
            "other-root truth spikes still count"
        );
    }
    #[test]
    fn missing_values_and_empty_truth_never_certify_agreement() {
        assert!(audit_axis(&[f32::NAN], &[true], &[true], [1, 0, 1]).is_none());
        assert!(audit_axis(&[1.0], &[true], &[true], [0, 0, 0]).is_none());
        assert!(audit_axis(&[1.0], &[], &[], [1, 1, 2]).is_none());
    }
}

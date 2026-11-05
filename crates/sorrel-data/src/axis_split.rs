//! Input-only 1-D mixture boundaries across individual PC coordinates and
//! amplitude. Does not use feature-oracle thresholds or any human labels.

use crate::proposers::cluster_feature_space;
use crate::session::Session;
use sorrel_compute::{fit_diag_gmm, sliding_refractory_argmin, DiagGmm};
use sorrel_io::{ClusterId, DataProvider};

const MAX_FIT: usize = 5000;
const MAX_K: usize = 4;
const BIC_GAIN: f64 = 10.0;
const MIN_PART: usize = 100;

/// Stable lower-group posterior minus .5. Components are ordered by mean,
/// not by their arbitrary fitted IDs.
fn odds(model: &DiagGmm, lower: &[usize], x: f64) -> f64 {
    let logs: Vec<_> = (0..model.k)
        .map(|j| {
            model.weights[j].ln()
                - 0.5 * ((x - model.means[j]).powi(2) / model.vars[j] + model.vars[j].ln())
        })
        .collect();
    let m = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = logs.iter().map(|v| (v - m).exp()).collect();
    lower.iter().map(|&j| weights[j]).sum::<f64>() / weights.iter().sum::<f64>() - 0.5
}

/// Adjacent-mean cumulative-posterior crossings of the BIC-selected mixture.
/// No scan for maximal refractory gain or overlap with an external label.
pub fn axis_boundaries(values: &[f32]) -> Vec<(f32, f64)> {
    let finite: Vec<_> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.len() < 2 * MIN_PART {
        return vec![];
    }
    let m = finite.len().min(MAX_FIT);
    let mut seed = 0x5e7a_4381_04bd_19f3_u64;
    let sub: Vec<_> = (0..m)
        .map(|j| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let (lo, hi) = (j * finite.len() / m, (j + 1) * finite.len() / m);
            finite[lo + seed as usize % (hi - lo)]
        })
        .collect();
    let Some(single) = fit_diag_gmm(&sub, 1, 1, 60) else {
        return vec![];
    };
    let Some(model) = (1..=MAX_K)
        .filter_map(|k| fit_diag_gmm(&sub, 1, k, 60))
        .min_by(|a, b| a.bic().total_cmp(&b.bic()))
    else {
        return vec![];
    };
    let gain = single.bic() - model.bic();
    if model.k < 2 || !gain.is_finite() || gain < BIC_GAIN {
        return vec![];
    }
    let mut order: Vec<_> = (0..model.k).collect();
    order.sort_by(|&a, &b| model.means[a].total_cmp(&model.means[b]));
    let mut cuts = Vec::new();
    for split in 1..model.k {
        let (left, right) = (model.means[order[split - 1]], model.means[order[split]]);
        if right - left < 1e-6 {
            continue;
        }
        let lower = &order[..split];
        // Find a lower->upper posterior transition between adjacent means.
        // Unequal variances need not imply a unique crossing; use the one
        // nearest the mean midpoint, without consulting spike timings.
        let mut brackets = Vec::new();
        let mut previous = (left, odds(&model, lower, left));
        for step in 1..=128 {
            let x = left + (right - left) * step as f64 / 128.0;
            let current = odds(&model, lower, x);
            if previous.1 >= 0.0 && current <= 0.0 {
                brackets.push((previous.0, x));
            }
            previous = (x, current);
        }
        let midpoint = (left + right) * 0.5;
        let Some((mut lo, mut hi)) = brackets.into_iter().min_by(|a, b| {
            (((a.0 + a.1) * 0.5 - midpoint).abs())
                .total_cmp(&(((b.0 + b.1) * 0.5 - midpoint).abs()))
        }) else {
            continue;
        };
        for _ in 0..32 {
            let x = (lo + hi) * 0.5;
            if odds(&model, lower, x) > 0.0 {
                lo = x;
            } else {
                hi = x;
            }
        }
        let cut = ((lo + hi) * 0.5) as f32;
        if cut.is_finite() && !cuts.iter().any(|(x, _)| *x == cut) {
            cuts.push((cut, gain));
        }
    }
    cuts
}

#[derive(Clone, Debug)]
pub struct AxisCandidate {
    pub members: Vec<u32>,
    pub feature: String,
    pub at: f32,
    pub above: bool,
    pub bic_gain: f64,
    pub contamination: f32,
    pub supported: bool,
}

pub fn axis_candidates<P: DataProvider>(
    session: &Session<P>,
    c: ClusterId,
    globals: &[u32],
) -> Vec<AxisCandidate> {
    if globals.len() < 2 * MIN_PART {
        return vec![];
    }
    let mut columns = Vec::new();
    if let Some((feats, d)) = cluster_feature_space(session, c, globals) {
        for j in 0..d {
            columns.push((
                format!("pc_whitened_{j}"),
                feats.chunks_exact(d).map(|r| r[j]).collect::<Vec<_>>(),
            ));
        }
    }
    let amps = session.spike_amplitudes(c);
    if amps.len() == globals.len() {
        columns.push(("amplitude".into(), amps.to_vec()));
    }
    let times = session.spike_times(c);
    let sr = session.provider.sample_rate();
    if !sr.is_finite() || sr <= 0.0 {
        return vec![];
    }
    let duration = session.provider.n_samples().0;
    let contamination = |t: &[_]| {
        let (c, rp) = sliding_refractory_argmin(
            t,
            (sr * 0.0005) as u64,
            (sr * 0.01) as u64,
            20,
            duration,
            sr,
        );
        if rp > 0 {
            c
        } else {
            f32::NAN
        }
    };
    let whole = contamination(times);
    let mut candidates = Vec::new();
    for (feature, values) in columns {
        for (at, bic_gain) in axis_boundaries(&values) {
            let (mut high, mut low) = (Vec::new(), Vec::new());
            for (i, &v) in values.iter().enumerate() {
                if v.is_finite() && v > at {
                    high.push(i as u32);
                } else {
                    low.push(i as u32);
                }
            }
            if high.len() < MIN_PART || low.len() < MIN_PART {
                continue;
            }
            for (above, members) in [(true, high), (false, low)] {
                let t: Vec<_> = members.iter().map(|&i| times[i as usize]).collect();
                let rpc = contamination(&t);
                candidates.push(AxisCandidate {
                    members,
                    feature: feature.clone(),
                    at,
                    above,
                    bic_gain,
                    contamination: rpc,
                    supported: rpc.is_finite() && whole.is_finite() && rpc <= 0.1 && rpc <= whole,
                });
            }
        }
    }
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    fn gaussian(s: &mut u32) -> f32 {
        let mut u = || {
            *s ^= *s << 13;
            *s ^= *s >> 17;
            *s ^= *s << 5;
            (*s as f64 / u32::MAX as f64).clamp(1e-9, 1.0)
        };
        ((-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos()) as f32
    }
    #[test]
    fn mixture_cuts_do_not_depend_on_human_thresholds_and_preserve_rare_modes() {
        let mut seed = 2374;
        let x: Vec<_> = (0..40_000)
            .map(|i| {
                if i % 100 == 0 {
                    12.0 + 0.2 * gaussian(&mut seed)
                } else {
                    gaussian(&mut seed)
                }
            })
            .collect();
        let cuts = axis_boundaries(&x);
        assert!(cuts.len() <= MAX_K - 1);
        assert!(
            cuts.iter().any(|(at, _)| {
                let predicted = x.iter().filter(|&&v| v > *at).count();
                let hit = x
                    .iter()
                    .enumerate()
                    .filter(|(i, v)| i % 100 == 0 && **v > *at)
                    .count();
                hit as f64 / (400 + predicted - hit) as f64 > 0.8
            }),
            "cuts {cuts:?}"
        );
        assert_eq!(cuts, axis_boundaries(&x));
    }
    #[test]
    fn no_structure_on_a_gaussian_or_invalid_data() {
        assert!(axis_boundaries(&[f32::NAN; 400]).is_empty());
        assert!(axis_boundaries(&[1.0; 400]).is_empty());
        let mut seed = 3427;
        let x: Vec<_> = (0..5000).map(|_| gaussian(&mut seed)).collect();
        assert!(axis_boundaries(&x).is_empty());
    }
}

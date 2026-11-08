//! Experimental covariance-aware majority cleanup. The statistical boundary
//! is fitted without timing/truth; the refractory checks only veto an offer.

use crate::proposers::cluster_feature_space;
use crate::session::Session;
use sorrel_compute::contaminated_gaussian::fit_contaminated_gaussian;
use sorrel_compute::sliding_refractory_argmin;
use sorrel_io::{ClusterId, DataProvider};

const MAX_FIT: usize = 5000;
const MIN_PART: usize = 100;
const MIN_BIC_GAIN: f64 = 10.0;

#[derive(Clone, Debug)]
pub struct RobustCleanup {
    /// Local ascending indices of broad contamination, not the kept majority.
    pub minor_idx: Vec<u32>,
    pub bic_gain: f64,
    pub clean_weight: f64,
    pub inflation: f64,
    pub whole_contamination: f32,
    pub kept_contamination: f32,
    pub supported: bool,
}

/// Fit one posterior boundary, never an ISI-optimized scan of nested cuts.
/// At least 100 spikes in each part; the retained part must be a majority.
pub fn robust_boundary_proposal(feats: &[f32], d: usize) -> Option<RobustCleanup> {
    if d == 0 || !feats.len().is_multiple_of(d) || feats.len() / d > u32::MAX as usize {
        return None;
    }
    let n = feats.len() / d;
    if n < 2 * MIN_PART {
        return None;
    }
    // Jittered strata avoid periodic-stride aliasing and keep rows time-ordered.
    let m = n.min(MAX_FIT);
    let mut seed = 0x283e_ab91_715c_43d7_u64;
    let mut sub = Vec::with_capacity(m * d);
    for j in 0..m {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let (lo, hi) = (j * n / m, (j + 1) * n / m);
        let i = lo + (seed as usize % (hi - lo));
        sub.extend_from_slice(&feats[i * d..(i + 1) * d]);
    }
    let model = fit_contaminated_gaussian(&sub, d)?;
    if model.bic_gain < MIN_BIC_GAIN {
        return None;
    }
    let minor_idx: Vec<_> = model
        .clean_posteriors(feats)?
        .into_iter()
        .enumerate()
        .filter(|(_, p)| *p < 0.5)
        .map(|(i, _)| i as u32)
        .collect();
    if minor_idx.len() < MIN_PART || minor_idx.len() * 2 >= n || n - minor_idx.len() < MIN_PART {
        return None;
    }
    Some(RobustCleanup {
        minor_idx,
        bic_gain: model.bic_gain,
        clean_weight: model.clean_weight,
        inflation: model.inflation,
        whole_contamination: f32::NAN,
        kept_contamination: f32::NAN,
        supported: false,
    })
}

/// Input-only boundary plus explicit refractory veto. Diagnostic users can
/// inspect unsupported boundaries, but the agent may act only on supported
/// offers. A clean whole is allowed: geometric contamination need not create
/// measurable refractory violations, especially in low-rate units.
pub fn robust_cleanup<P: DataProvider>(
    session: &Session<P>,
    c: ClusterId,
    globals: &[u32],
) -> Option<RobustCleanup> {
    if globals.len() < 2 * MIN_PART {
        return None;
    }
    let (feats, d) = cluster_feature_space(session, c, globals)?;
    let mut proposal = robust_boundary_proposal(&feats, d)?;
    let times = session.spike_times(c);
    let sr = session.provider.sample_rate();
    if !sr.is_finite() || sr <= 0.0 {
        return None;
    }
    let duration = session.provider.n_samples().0;
    let measure = |t: &[_]| {
        let (cont, rp) = sliding_refractory_argmin(
            t,
            (sr * 0.0005) as u64,
            (sr * 0.010) as u64,
            20,
            duration,
            sr,
        );
        if rp > 0 {
            cont
        } else {
            f32::NAN
        }
    };
    let kept: Vec<_> = times
        .iter()
        .enumerate()
        .filter(|(i, _)| proposal.minor_idx.binary_search(&(*i as u32)).is_err())
        .map(|(_, t)| *t)
        .collect();
    proposal.whole_contamination = measure(times);
    proposal.kept_contamination = measure(&kept);
    proposal.supported = supported(proposal.whole_contamination, proposal.kept_contamination);
    Some(proposal)
}

fn supported(whole: f32, kept: f32) -> bool {
    whole.is_finite() && kept.is_finite() && kept <= 0.1 && kept <= whole
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_sizes_shape_and_refractory_veto() {
        assert!(robust_boundary_proposal(&[0.0; 100], 2).is_none());
        assert!(robust_boundary_proposal(&[0.0; 401], 2).is_none());
        assert!(robust_boundary_proposal(&[f32::INFINITY; 400], 2).is_none());
        assert!(supported(0.0, 0.0));
        assert!(supported(0.8, 0.05));
        assert!(!supported(0.01, 0.02));
        assert!(!supported(0.8, 0.11));
        assert!(!supported(f32::NAN, 0.0));
    }
}

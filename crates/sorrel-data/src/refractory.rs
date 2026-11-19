use sorrel_compute::{sliding_refractory_argmin, RefractoryParams};
use sorrel_io::SampleIndex;

/// Versioned model features (including QC, evaluations and proposer inputs)
/// must never follow user settings. Keep the legacy sample rounding and
/// inclusive 20-window grid exactly as the extractor used them.
#[derive(Clone, Copy)]
pub(crate) enum RefractoryWindows<'a> {
    Session(&'a RefractoryParams),
    Learn,
}

impl RefractoryWindows<'_> {
    pub(crate) const LEARN_FIXED_MS: f32 = 1.5;

    pub(crate) fn fixed_samples(self, sr: f32) -> u64 {
        match self {
            Self::Session(params) => params.fixed_samples(sr),
            Self::Learn => (sr * (Self::LEARN_FIXED_MS / 1000.0)).round() as u64,
        }
    }

    /// Refractory half-width of the 1 ms-bin CCG dip score.
    pub(crate) fn ccg_dip_bins(self) -> usize {
        match self {
            Self::Session(params) => ccg_refractory_bins(params.fixed_ms, 1.0),
            Self::Learn => 2,
        }
    }
}

/// CCG bins covering a refractory half-width of `fixed_ms`, shared by merge
/// ranking and eval/probe so x.5 ratios round alike: the epsilon absorbs f32
/// roundoff in a bin width such as `2 * 0.05 s / 100`.
pub(crate) fn ccg_refractory_bins(fixed_ms: f32, bin_ms: f64) -> usize {
    ((f64::from(fixed_ms) / bin_ms.max(1e-9) + 1e-6).round() as usize).max(1)
}

#[cfg(test)]
#[test]
fn ccg_bins_round_alike() {
    let suggest_bin_ms = 2000.0 * f64::from(0.05_f32) / 100.0;
    for (ms, bins) in [(1.5, 2), (2.5, 3), (0.2, 1)] {
        assert_eq!(ccg_refractory_bins(ms, 1.0), bins);
        assert_eq!(ccg_refractory_bins(ms, suggest_bin_ms), bins);
    }
}

impl RefractoryWindows<'_> {
    pub(crate) fn sweep(self, times: &[SampleIndex], total: u64, sr: f32) -> (f32, u64) {
        match self {
            Self::Session(params) => params.sweep(times, total, sr),
            Self::Learn => sliding_refractory_argmin(
                times,
                (sr * 0.0005) as u64,
                (sr * 0.010) as u64,
                20,
                total,
                sr,
            ),
        }
    }
}

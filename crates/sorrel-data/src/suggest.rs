//! Curation suggestion engine.
//!
//! Walks the live `Session` state and proposes ranked merge/split candidates
//! the curator can review with one click. The scoring stays inside this
//! module so the UI is just a thin renderer; tests can poke at the scores
//! directly without spinning up egui.
//!
//! The signals come from `sorrel-compute` (everything quantitative lives
//! there); the heuristics here are about combining those signals into a
//! single `score: f32` that ranks well against curator intuition. We err on
//! the side of *too few* false positives — surfacing too many junk pairs
//! makes the panel useless.

use crate::session::Session;
use sorrel_compute::{
    amplitude_drift_correlation, amplitude_snr, analyse_refractory_dip, bimodality_coefficient,
    cross_correlogram, gmm_split_proposal, ks_two_sample, mean_amplitude, percentile,
    refractory_contamination, refractory_dip_score,
};
use sorrel_io::{ClusterId, DataProvider};

/// One suggested merge between two clusters, with the underlying evidence
/// retained so the UI can show *why* this pair scored where it did.
#[derive(Clone, Copy, Debug)]
pub struct MergeCandidate {
    pub a: ClusterId,
    pub b: ClusterId,
    /// Combined score in `[0, 1]`. Higher = stronger evidence the two
    /// clusters are the same neuron.
    pub score: f32,
    /// Refractory-dip z-score from the cross-correlogram. Large positive
    /// means the centre of the CCG is suppressed below baseline.
    pub ccg_dip_z: f32,
    /// Kolmogorov–Smirnov statistic between the two amplitude
    /// distributions. Small means the amplitudes are consistent with the
    /// same distribution.
    pub amp_ks: f32,
    /// Difference in mean amplitude as a fraction of the larger value.
    pub amp_mean_delta: f32,
    pub n_a: usize,
    pub n_b: usize,
}

/// One cluster the suggester believes should be split. May carry a
/// concrete bipartition proposal (`bipartition_minor_idx`) when the
/// amplitude distribution fits a 2-component GMM well — that lets the UI
/// offer a one-click apply.
#[derive(Clone, Debug)]
pub struct SplitCandidate {
    pub cluster: ClusterId,
    /// Combined score in `[0, 1]`. Higher = stronger evidence of structure
    /// inside the cluster's amplitude/feature distributions.
    pub score: f32,
    /// Sarle's bimodality coefficient on the amplitude distribution.
    /// Above 0.555 = evidence of bimodality; we map this onto `[0, 1]` for
    /// the score.
    pub amp_bimodality: f32,
    /// Refractory contamination — high values often co-occur with
    /// over-merged clusters that should be split apart.
    pub contamination: f32,
    /// Pearson correlation of amplitude with time. Strong drift sometimes
    /// means the cluster captures two separate units across a movement
    /// event.
    pub drift_corr: f32,
    pub n_spikes: usize,
    /// `BIC(k=1) - BIC(k=2)` for the GMM fit on this cluster's
    /// amplitudes. Positive = mixture model wins. NaN if the GMM was
    /// unable to fit (small cluster, no amplitudes, etc.).
    pub gmm_bic_delta: f32,
    /// Local spike indices the GMM would move out as a fresh cluster.
    /// Empty when no proposal could be made — the curator falls back to
    /// the FeatureView lasso path in that case.
    pub bipartition_minor_idx: Vec<u32>,
}

/// Suggester configuration. Defaults are tuned for ~30 kHz Neuropixels-style
/// data — callers can override per recording if needed.
#[derive(Clone, Copy, Debug)]
pub struct SuggestConfig {
    /// Refractory window in seconds — the centre of the CCG that should be
    /// empty for the same-unit hypothesis.
    pub refractory_seconds: f32,
    /// Half-width of the CCG analysis window in seconds.
    pub ccg_max_lag_seconds: f32,
    /// Number of CCG bins (must be even — the histogram is centred).
    pub ccg_bins: usize,
    /// Minimum spikes per cluster for either suggester.
    pub min_spikes: usize,
    /// Maximum number of merge pairs returned (top-K by score).
    pub top_merges: usize,
    /// Maximum number of split candidates returned.
    pub top_splits: usize,
    /// Skip merge pairs whose mean-amplitude differ by more than this
    /// fraction (it's almost never the same unit if the amplitudes are
    /// fundamentally different).
    pub max_amp_mean_delta: f32,
    /// Minimum score to include a candidate at all.
    pub min_score: f32,
}

impl Default for SuggestConfig {
    fn default() -> Self {
        Self {
            refractory_seconds: 0.0015, // 1.5 ms — slightly conservative
            ccg_max_lag_seconds: 0.050,
            ccg_bins: 100,
            min_spikes: 30,
            top_merges: 20,
            top_splits: 20,
            max_amp_mean_delta: 0.5,
            min_score: 0.15,
        }
    }
}

/// Rank merge candidates over the entire session. Compares every cluster
/// pair `(a < b)` whose mean amplitudes are within `max_amp_mean_delta`,
/// scores by a weighted combination of refractory-dip strength, amplitude
/// distribution similarity, and mean-amp closeness, and returns the top-K.
pub fn rank_merge_candidates<P: DataProvider>(
    session: &Session<P>,
    cfg: &SuggestConfig,
) -> Vec<MergeCandidate> {
    let n = session.n_clusters();
    if n < 2 {
        return Vec::new();
    }
    let sr = session.provider.sample_rate().max(1.0);
    let max_lag_samples = (sr * cfg.ccg_max_lag_seconds).round() as u64;
    if max_lag_samples == 0 {
        return Vec::new();
    }
    let bins = cfg.ccg_bins.max(2);
    let bin_width = (2.0 * cfg.ccg_max_lag_seconds) / bins as f32;
    let refractory_bins = (cfg.refractory_seconds / bin_width.max(1e-9)).round() as usize;
    let refractory_bins = refractory_bins.max(1);
    let shoulder_bins = (refractory_bins * 4).max(4);

    // Pre-compute per-cluster summaries so the inner loop is cheap.
    let mut summaries: Vec<Option<ClusterSummary>> = Vec::with_capacity(n as usize);
    for c in (0..n).map(ClusterId) {
        let times = session.spike_times(c);
        let amps = session.spike_amplitudes(c);
        if times.len() < cfg.min_spikes {
            summaries.push(None);
            continue;
        }
        let mean_amp = if amps.is_empty() {
            f32::NAN
        } else {
            mean_amplitude(amps)
        };
        summaries.push(Some(ClusterSummary {
            mean_amp,
            n: times.len(),
        }));
    }

    // Build the (a, b) pairs that survive the cheap amplitude pre-filter,
    // then score each pair in parallel. Rayon spreads the C² work across
    // all cores; `&Session<P>` is Send + Sync (journal is a plain
    // BufWriter<File>) so the closure can borrow it directly.
    use rayon::prelude::*;

    let mut pairs: Vec<(ClusterId, ClusterId, ClusterSummary, ClusterSummary, f32)> = Vec::new();
    for a in 0..n {
        let Some(sa) = summaries[a as usize] else {
            continue;
        };
        for b in (a + 1)..n {
            let Some(sb) = summaries[b as usize] else {
                continue;
            };
            let a = ClusterId(a);
            let b = ClusterId(b);
            let amp_mean_delta = if sa.mean_amp.is_finite() && sb.mean_amp.is_finite() {
                let lo = sa.mean_amp.abs().min(sb.mean_amp.abs()).max(1e-6);
                ((sa.mean_amp - sb.mean_amp).abs() / lo).min(1.0)
            } else {
                0.0
            };
            if amp_mean_delta > cfg.max_amp_mean_delta {
                continue;
            }
            pairs.push((a, b, sa, sb, amp_mean_delta));
        }
    }

    let mut out: Vec<MergeCandidate> = pairs
        .par_iter()
        .filter_map(|&(a, b, sa, sb, amp_mean_delta)| {
            let times_a = session.spike_times(a);
            let times_b = session.spike_times(b);
            let h = cross_correlogram(times_a, times_b, max_lag_samples, bins);
            let analysis = analyse_refractory_dip(&h, refractory_bins, shoulder_bins);
            let dip_score = refractory_dip_score(&analysis);

            let amps_a = session.spike_amplitudes(a);
            let amps_b = session.spike_amplitudes(b);
            let amp_ks = if amps_a.is_empty() || amps_b.is_empty() {
                0.0
            } else {
                ks_two_sample(amps_a, amps_b)
            };
            let amp_sim = (1.0 - amp_ks).clamp(0.0, 1.0);
            let amp_mean_sim = (1.0 - amp_mean_delta).clamp(0.0, 1.0);

            // Weighted combination — refractory dip is the strongest single
            // signal (0.6); amplitude evidence rounds it out (0.3 + 0.1).
            let score = (0.6 * dip_score + 0.3 * amp_sim + 0.1 * amp_mean_sim).clamp(0.0, 1.0);
            if score < cfg.min_score {
                return None;
            }
            Some(MergeCandidate {
                a,
                b,
                score,
                ccg_dip_z: analysis.z,
                amp_ks,
                amp_mean_delta,
                n_a: sa.n,
                n_b: sb.n,
            })
        })
        .collect();

    out.sort_by(|x, y| {
        y.score
            .partial_cmp(&x.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(cfg.top_merges);
    out
}

/// Rank split candidates over the entire session. A cluster scores high if
/// its amplitude distribution is bimodal (Sarle BC > 0.555) AND/OR it shows
/// significant refractory contamination AND/OR a strong amplitude drift.
pub fn rank_split_candidates<P: DataProvider>(
    session: &Session<P>,
    cfg: &SuggestConfig,
) -> Vec<SplitCandidate> {
    let n = session.n_clusters();
    if n == 0 {
        return Vec::new();
    }
    let sr = session.provider.sample_rate().max(1.0);
    let refractory_samples = (sr * cfg.refractory_seconds).round() as u64;
    let total_duration = session.provider.n_samples();

    use rayon::prelude::*;
    let mut out: Vec<SplitCandidate> = (0..n)
        .into_par_iter()
        .map(ClusterId)
        .filter_map(|c| {
            let times = session.spike_times(c);
            let amps = session.spike_amplitudes(c);
            if times.len() < cfg.min_spikes {
                return None;
            }
            let bc = if amps.len() >= 8 {
                bimodality_coefficient(amps)
            } else {
                0.0
            };
            let bc_score = ((bc - 0.4) / 0.4).clamp(0.0, 1.0);
            let contamination =
                refractory_contamination(times, refractory_samples, total_duration.0, sr);
            let cont_score = (contamination * 2.0).clamp(0.0, 1.0);
            let drift = amplitude_drift_correlation(times, amps).abs();
            let drift_score = ((drift - 0.3) / 0.5).clamp(0.0, 1.0);
            let snr = amplitude_snr(amps);
            let snr_gate = (snr / (snr + 0.5)).clamp(0.0, 1.0);

            let (gmm_delta, bipartition) = if bc >= 0.5 && amps.len() >= 50 {
                match gmm_split_proposal(amps) {
                    Some(p) if p.bic_delta >= 6.0 => (p.bic_delta, p.minor_spike_idx),
                    Some(p) => (p.bic_delta, Vec::new()),
                    None => (f32::NAN, Vec::new()),
                }
            } else {
                (f32::NAN, Vec::new())
            };
            let gmm_bonus = if bipartition.is_empty() {
                0.0
            } else {
                (gmm_delta.min(40.0) / 40.0 * 0.15).clamp(0.0, 0.15)
            };

            let raw = ((0.6 * bc_score + 0.25 * cont_score + 0.15 * drift_score) * snr_gate
                + gmm_bonus)
                .clamp(0.0, 1.0);
            if raw < cfg.min_score {
                return None;
            }
            Some(SplitCandidate {
                cluster: c,
                score: raw,
                amp_bimodality: bc,
                contamination,
                drift_corr: drift,
                n_spikes: times.len(),
                gmm_bic_delta: gmm_delta,
                bipartition_minor_idx: bipartition,
            })
        })
        .collect();

    out.sort_by(|x, y| {
        y.score
            .partial_cmp(&x.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(cfg.top_splits);
    out
}

/// Per-cluster summary cached during merge ranking.
#[derive(Clone, Copy, Debug)]
struct ClusterSummary {
    mean_amp: f32,
    n: usize,
}

/// Helper: a robust amplitude-distribution overlap metric. The middle 80%
/// of one cluster's amplitudes should overlap heavily with the other's if
/// they're the same unit.
pub fn amplitude_iqr_overlap(a: &[f32], b: &[f32]) -> f32 {
    if a.len() < 4 || b.len() < 4 {
        return 0.0;
    }
    let a_lo = percentile(a.to_vec(), 0.10);
    let a_hi = percentile(a.to_vec(), 0.90);
    let b_lo = percentile(b.to_vec(), 0.10);
    let b_hi = percentile(b.to_vec(), 0.90);
    let lo = a_lo.max(b_lo);
    let hi = a_hi.min(b_hi);
    if hi <= lo {
        return 0.0;
    }
    let union_lo = a_lo.min(b_lo);
    let union_hi = a_hi.max(b_hi);
    let union_span = (union_hi - union_lo).max(1e-9);
    ((hi - lo) / union_span).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::SqliteJournal;
    use sorrel_io::{
        ClusterId, DataProvider, HasAmplitudes, SampleIndex, TraceSamples, TraceSlice,
    };

    /// In-memory provider backed by per-cluster spike times + amplitudes.
    struct MockProvider {
        spikes: Vec<Vec<SampleIndex>>,
        amps: Vec<Vec<f32>>,
        n_samples: SampleIndex,
    }

    impl DataProvider for MockProvider {
        type Label = u8;
        fn sample_rate(&self) -> f32 {
            1000.0
        }
        fn n_channels(&self) -> u32 {
            1
        }
        fn n_samples(&self) -> SampleIndex {
            self.n_samples
        }
        fn n_clusters(&self) -> u32 {
            self.spikes.len() as u32
        }
        fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
            self.spikes.get(c.idx()).map(Vec::as_slice).unwrap_or(&[])
        }
        fn trace(&self, _: SampleIndex, _: u32) -> TraceSlice<'_> {
            TraceSlice {
                start: SampleIndex(0),
                n_channels: 1,
                samples: TraceSamples::I16(&[]),
            }
        }
        fn initial_labels(&self) -> Vec<u8> {
            vec![0; self.spikes.len()]
        }
    }

    impl HasAmplitudes for MockProvider {
        fn spike_amplitudes(&self, c: ClusterId) -> &[f32] {
            self.amps.get(c.idx()).map(Vec::as_slice).unwrap_or(&[])
        }
    }

    fn fresh_session(prov: MockProvider) -> (Session<MockProvider>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let journal = SqliteJournal::open(&dir.path().join("j.sqlite")).unwrap();
        let mut s = Session::new(prov, journal);
        s.seed_amplitudes();
        (s, dir)
    }

    #[test]
    fn merge_suggester_picks_pair_with_refractory_dip() {
        // Two clusters that are clearly the same neuron: identical amp dist,
        // and their cross-correlogram has a deep refractory dip because
        // their spikes never co-fire within 1.5 ms.
        let mut a_times = Vec::new();
        let mut b_times = Vec::new();
        for i in 0..500 {
            // alternate clusters at 5 ms intervals; never within 1.5 ms.
            a_times.push(SampleIndex((i as u64) * 10));
            b_times.push(SampleIndex((i as u64) * 10 + 5));
        }
        let a_amps: Vec<f32> = (0..500).map(|_| 5.0).collect();
        let b_amps: Vec<f32> = (0..500).map(|_| 5.0).collect();
        // A control cluster that has nothing to do with the others.
        let c_times: Vec<SampleIndex> = (0..500).map(|i| SampleIndex((i as u64) * 7 + 3)).collect();
        let c_amps: Vec<f32> = (0..500).map(|_| 5.0).collect();

        let prov = MockProvider {
            spikes: vec![a_times, b_times, c_times],
            amps: vec![a_amps, b_amps, c_amps],
            n_samples: SampleIndex(10_000),
        };
        let (sess, _d) = fresh_session(prov);
        let cfg = SuggestConfig::default();
        let merges = rank_merge_candidates(&sess, &cfg);
        assert!(!merges.is_empty());
        // The 0-1 pair should rank first.
        assert_eq!(
            (merges[0].a, merges[0].b),
            (ClusterId(0), ClusterId(1)),
            "expected (0,1) at top, got {:?}",
            (merges[0].a, merges[0].b),
        );
    }

    #[test]
    fn split_suggester_flags_bimodal_amplitudes() {
        // Cluster 0: bimodal amplitudes (clearly two populations).
        let mut amps_bi: Vec<f32> = Vec::new();
        amps_bi.resize(500, 2.0);
        amps_bi.resize(1000, 8.0);
        let times_bi: Vec<SampleIndex> = (0..1000).map(|i| SampleIndex((i as u64) * 10)).collect();

        // Cluster 1: unimodal amplitudes (Gaussian-ish around 5).
        let amps_uni: Vec<f32> = (0..1000)
            .map(|i| 5.0 + ((i as f32 * 0.7).sin()) * 0.5)
            .collect();
        let times_uni: Vec<SampleIndex> = (0..1000).map(|i| SampleIndex((i as u64) * 10)).collect();

        let prov = MockProvider {
            spikes: vec![times_bi, times_uni],
            amps: vec![amps_bi, amps_uni],
            n_samples: SampleIndex(10_000),
        };
        let (sess, _d) = fresh_session(prov);
        let cfg = SuggestConfig::default();
        let splits = rank_split_candidates(&sess, &cfg);
        assert!(!splits.is_empty());
        assert_eq!(
            splits[0].cluster,
            ClusterId(0),
            "expected bimodal cluster 0 to top"
        );
    }

    #[test]
    fn iqr_overlap_one_for_identical_distributions() {
        let v: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let o = amplitude_iqr_overlap(&v, &v);
        assert!(o > 0.79, "expected ≈0.8 for identical 10-90 IQR, got {o}");
    }

    #[test]
    fn iqr_overlap_zero_for_disjoint_distributions() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (200..300).map(|i| i as f32).collect();
        assert_eq!(amplitude_iqr_overlap(&a, &b), 0.0);
    }
}

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
    cross_correlogram, gmm_split_proposal, hartigan_dip, ks_two_sample, mean_amplitude, percentile,
    refractory_contamination, refractory_dip_score,
};
use sorrel_io::{ClusterId, DataProvider, SampleIndex};
use std::collections::HashMap;

/// One suggested merge between two clusters, with the underlying evidence
/// retained so the UI can show *why* this pair scored where it did.
#[derive(Clone, Copy, Debug, serde::Serialize)]
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
#[derive(Clone, Debug, serde::Serialize)]
pub struct SplitCandidate {
    pub cluster: ClusterId,
    /// Combined score in `[0, 1]`. Higher = stronger evidence of structure
    /// inside the cluster's amplitude/feature distributions.
    pub score: f32,
    /// Sarle's bimodality coefficient on the amplitude distribution.
    /// Above 0.555 = evidence of bimodality; we map this onto `[0, 1]` for
    /// the score.
    pub amp_bimodality: f32,
    /// Hartigan dip statistic on the amplitude distribution — a rigorous
    /// departure-from-unimodality measure (`[0, 0.25]`) that corroborates
    /// the bimodality coefficient and catches multimodal shapes the
    /// moment-based BC misses.
    pub amp_dip: f32,
    /// Refractory contamination — high values often co-occur with
    /// over-merged clusters that should be split apart.
    pub contamination: f32,
    /// Pearson correlation of amplitude with time. Strong drift sometimes
    /// means the cluster captures two separate units across a movement
    /// event.
    pub drift_corr: f32,
    pub n_spikes: usize,
    /// Which proposer produced `bipartition_minor_idx`: `"kmeans"` (PC-feature
    /// k-means) or `"gmm"` (amplitude GMM; also the label when there is no
    /// proposal). k-means is preferred when both pass.
    pub method: &'static str,
    /// `BIC(k=1) - BIC(k=2)` for the GMM fit on this cluster's
    /// amplitudes. Positive = mixture model wins. NaN if the GMM was
    /// unable to fit (small cluster, no amplitudes, etc.).
    pub gmm_bic_delta: f32,
    /// Local spike indices the GMM would move out as a fresh cluster.
    /// Empty when no proposal could be made — the curator falls back to
    /// the FeatureView lasso path in that case.
    pub bipartition_minor_idx: Vec<u32>,
    /// Why the usual evidence was discounted. `"burst_explained"`: the
    /// amplitude bimodality vanishes once amplitude is regressed on
    /// ln(preceding ISI), i.e. a bursting cell rather than two units.
    pub reasons: Vec<&'static str>,
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

/// Peak channels closer than this (µm) make a pair a drift-fragment candidate.
const DRIFT_MAX_UM: f32 = 40.0;
/// Minimum [`presence_complement`] for a pair to enter via the drift source.
const DRIFT_MIN_COMPLEMENT: f32 = 0.3;
/// Weight of the presence-complement term in the merge score.
const DRIFT_WEIGHT: f32 = 0.25;
/// Time bins used for the firing-presence comparison.
const PRESENCE_BINS: usize = 20;

#[derive(Clone, Copy)]
struct PairScore {
    dip_score: f32,
    z: f32,
    amp_ks: f32,
}

/// Scores kept across calls so an edit only rescoring what it touched.
///
/// Entries are keyed by a blake3 of the cluster's spike times and
/// amplitudes (split entries also hash the cluster's global spike indices,
/// which fix its PC features), never by cluster id, so a merge or relabel
/// leaves every other cluster's entries valid. Gating (similar set,
/// `notmerge`/`notsplit` tags, amplitude pre-filter, `min_score`, top-K) is
/// recomputed per call. Entries not used by the latest call are dropped.
#[derive(Default)]
pub struct SuggestMemo {
    /// Channel positions (µm) for the drift-fragment merge source; empty
    /// turns that source off (it needs a `HasGeometry` provider).
    pub positions: Vec<[f32; 2]>,
    gen: u64,
    pair_sig: [u64; 4],
    pairs: HashMap<([u8; 32], [u8; 32]), (PairScore, u64)>,
    split_sig: [u64; 4],
    splits: HashMap<[u8; 32], (SplitCandidate, u64)>,
}

fn feed<T: Copy, const N: usize>(h: &mut blake3::Hasher, s: &[T], f: impl Fn(T) -> [u8; N]) {
    h.update(&(s.len() as u64).to_le_bytes());
    let mut buf = Vec::with_capacity(4096 * N);
    for chunk in s.chunks(4096) {
        buf.clear();
        buf.extend(chunk.iter().flat_map(|&v| f(v)));
        h.update(&buf);
    }
}

/// Content key of one cluster: everything the pair scorer reads from it.
fn content_key(times: &[SampleIndex], amps: &[f32]) -> blake3::Hasher {
    let mut h = blake3::Hasher::new();
    feed(&mut h, times, |t| t.0.to_le_bytes());
    feed(&mut h, amps, f32::to_le_bytes);
    h
}

/// Firing counts of `times` in [`PRESENCE_BINS`] equal slices of the recording.
fn presence_counts(times: &[SampleIndex], total: u64) -> Vec<f32> {
    let width = (total as f64 / PRESENCE_BINS as f64).max(1.0);
    let mut c = vec![0.0f32; PRESENCE_BINS];
    for t in times {
        c[((t.0 as f64 / width) as usize).min(PRESENCE_BINS - 1)] += 1.0;
    }
    c
}

/// How complementary two presence histograms are, in `[0, 1]`: negative
/// Pearson correlation of the bin counts, scaled by the fraction of
/// non-co-occupied bins (a bin is occupied above 10% of the cluster's peak
/// bin). A unit split in two by a drift event fires in one fragment or the
/// other, so this is near 1; independent stationary units score near 0.
fn presence_complement(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        cov += (x - ma) * (y - mb);
        va += (x - ma) * (x - ma);
        vb += (y - mb) * (y - mb);
    }
    if va <= 0.0 || vb <= 0.0 {
        return 0.0;
    }
    let occ = |v: &[f32]| {
        let cut = v.iter().copied().fold(0.0, f32::max) * 0.1;
        v.iter().map(|&x| x > cut).collect::<Vec<_>>()
    };
    let (oa, ob) = (occ(a), occ(b));
    let both = oa.iter().zip(&ob).filter(|(x, y)| **x && **y).count();
    let denom = oa
        .iter()
        .filter(|x| **x)
        .count()
        .min(ob.iter().filter(|x| **x).count());
    let co = both as f32 / denom.max(1) as f32;
    (-cov / (va * vb).sqrt()).clamp(0.0, 1.0) * (1.0 - co)
}

/// Channel with the largest peak-to-peak in the cluster's primary template.
fn peak_channel<P: DataProvider>(session: &Session<P>, c: ClusterId) -> Option<usize> {
    let w = session.template_waveform(session.primary_template(c)?)?;
    let (_, ns, nc) = session.template_shape();
    let ptp = |ch: usize| {
        let (lo, hi) = (0..ns)
            .filter_map(|i| w.get(i * nc + ch))
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        hi - lo
    };
    (0..nc).max_by(|&a, &b| ptp(a).total_cmp(&ptp(b)))
}

/// Rank merge candidates over the entire session. Compares every cluster
/// pair `(a < b)` whose mean amplitudes are within `max_amp_mean_delta`,
/// scores by a weighted combination of refractory-dip strength, amplitude
/// distribution similarity, and mean-amp closeness, and returns the top-K.
pub fn rank_merge_candidates<P: DataProvider>(
    session: &Session<P>,
    cfg: &SuggestConfig,
) -> Vec<MergeCandidate> {
    rank_merge_candidates_memo(session, cfg, &mut SuggestMemo::default())
}

/// [`rank_merge_candidates`] reusing pair scores from `memo`; the result is
/// identical to a fresh-memo run.
///
/// A second candidate source finds drift fragments: pairs whose peak
/// channels lie within [`DRIFT_MAX_UM`] and whose firing presence is
/// complementary ([`presence_complement`] at least [`DRIFT_MIN_COMPLEMENT`];
/// needs `memo.positions` and template waveforms). Their score adds
/// `0.25 * presence_complement` to the usual `0.6 dip + 0.3 amp + 0.1 mean`
/// (clamped to 1): the CCG of two non-overlapping fragments carries little
/// dip evidence, so the term lifts them over `min_score`.
pub fn rank_merge_candidates_memo<P: DataProvider>(
    session: &Session<P>,
    cfg: &SuggestConfig,
    memo: &mut SuggestMemo,
) -> Vec<MergeCandidate> {
    use rayon::prelude::*;
    let n = session.n_clusters();
    if n < 2 {
        return Vec::new();
    }
    let sr = session.provider.sample_rate().max(1.0);
    let max_lag_samples = (sr * cfg.ccg_max_lag_seconds).round() as u64;
    if max_lag_samples == 0 {
        return Vec::new();
    }
    // `analyse_refractory_dip` assumes an even bin count so zero lag falls on
    // the boundary between bins `half-1` and `half` and the refractory window
    // `[half-r, half+r)` is symmetric around it. Round up to even so a
    // user-configured odd `ccg_bins` can't silently mis-center the dip.
    let bins = (cfg.ccg_bins.max(2) + 1) & !1;
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
    let keys: Vec<Option<[u8; 32]>> = (0..n)
        .into_par_iter()
        .map(|c| {
            summaries[c as usize].map(|_| {
                let c = ClusterId(c);
                *content_key(session.spike_times(c), session.spike_amplitudes(c))
                    .finalize()
                    .as_bytes()
            })
        })
        .collect();

    // Drift source inputs: peak-channel position and presence histogram per
    // cluster (both `None` without positions / templates).
    let total = session.provider.n_samples().0;
    let drift: Vec<Option<([f32; 2], Vec<f32>)>> = (0..n)
        .map(|c| {
            summaries[c as usize].as_ref()?;
            let c = ClusterId(c);
            let pos = *memo.positions.get(peak_channel(session, c)?)?;
            Some((pos, presence_counts(session.spike_times(c), total)))
        })
        .collect();

    // Build the (a, b) pairs that survive the cheap pre-filter, then score
    // each pair in parallel. Rayon spreads the work across all cores;
    // `&Session<P>` is Send + Sync (journal is a plain BufWriter<File>) so
    // the closure can borrow it directly.

    // With templates loaded, only each cluster's top-10 template-similar
    // clusters (score >= 0.5) are candidates (plus drift fragments);
    // otherwise every pair goes through the amplitude filter below.
    let similar: Option<std::collections::HashSet<(u32, u32)>> =
        (session.has_template_waveforms() || !session.similar_templates().is_empty()).then(|| {
            let mut set = std::collections::HashSet::new();
            for a in (0..n).filter(|&a| summaries[a as usize].is_some()) {
                for (b, _) in session
                    .similar_clusters(ClusterId(a), &[])
                    .into_iter()
                    .filter(|&(_, s)| s >= 0.5)
                    .take(10)
                {
                    set.insert((a.min(b.0), a.max(b.0)));
                }
            }
            set
        });
    // Rejection memory: `notmerge:<other>` on either cluster hides the pair.
    let rejected = |a: ClusterId, b: ClusterId| {
        let has = |x: ClusterId, y: ClusterId| {
            session
                .tags(x)
                .iter()
                .any(|t| *t == format!("notmerge:{}", y.0))
        };
        has(a, b) || has(b, a)
    };

    // (a, b, n_a, n_b, amp_mean_delta, presence_complement)
    let mut pairs: Vec<(ClusterId, ClusterId, usize, usize, f32, f32)> = Vec::new();
    for a in 0..n {
        let Some(sa) = summaries[a as usize] else {
            continue;
        };
        for b in (a + 1)..n {
            let Some(sb) = summaries[b as usize] else {
                continue;
            };
            let complement = match (&drift[a as usize], &drift[b as usize]) {
                (Some((pa, ca)), Some((pb, cb)))
                    if (pa[0] - pb[0]).hypot(pa[1] - pb[1]) <= DRIFT_MAX_UM =>
                {
                    presence_complement(ca, cb)
                }
                _ => 0.0,
            };
            let complement = if complement >= DRIFT_MIN_COMPLEMENT {
                complement
            } else {
                0.0
            };
            if complement == 0.0 && similar.as_ref().is_some_and(|s| !s.contains(&(a, b))) {
                continue;
            }
            let a = ClusterId(a);
            let b = ClusterId(b);
            if rejected(a, b) {
                continue;
            }
            let amp_mean_delta = if sa.mean_amp.is_finite() && sb.mean_amp.is_finite() {
                let hi = sa.mean_amp.abs().max(sb.mean_amp.abs()).max(1e-6);
                ((sa.mean_amp - sb.mean_amp).abs() / hi).min(1.0)
            } else {
                0.0
            };
            if amp_mean_delta > cfg.max_amp_mean_delta {
                continue;
            }
            pairs.push((a, b, sa.n, sb.n, amp_mean_delta, complement));
        }
    }

    // Resolve pair scores from the memo; score only the misses.
    let sig = [
        cfg.refractory_seconds.to_bits() as u64,
        cfg.ccg_max_lag_seconds.to_bits() as u64,
        bins as u64,
        sr.to_bits() as u64,
    ];
    if memo.pair_sig != sig {
        memo.pairs.clear();
        memo.pair_sig = sig;
    }
    memo.gen += 1;
    let gen = memo.gen;
    let pair_key = |a: ClusterId, b: ClusterId| (keys[a.idx()].unwrap(), keys[b.idx()].unwrap());
    let mut scores: Vec<Option<PairScore>> = vec![None; pairs.len()];
    let mut miss: Vec<usize> = Vec::new();
    for (i, p) in pairs.iter().enumerate() {
        match memo.pairs.get_mut(&pair_key(p.0, p.1)) {
            Some((s, g)) => {
                *g = gen;
                scores[i] = Some(*s);
            }
            None => miss.push(i),
        }
    }
    let computed: Vec<PairScore> = miss
        .par_iter()
        .map(|&i| {
            let (a, b, ..) = pairs[i];
            let h = cross_correlogram(
                session.spike_times(a),
                session.spike_times(b),
                max_lag_samples,
                bins,
            );
            let analysis = analyse_refractory_dip(&h, refractory_bins, shoulder_bins);
            let amps_a = session.spike_amplitudes(a);
            let amps_b = session.spike_amplitudes(b);
            PairScore {
                dip_score: refractory_dip_score(&analysis),
                z: analysis.z,
                amp_ks: if amps_a.is_empty() || amps_b.is_empty() {
                    0.0
                } else {
                    ks_two_sample(amps_a, amps_b)
                },
            }
        })
        .collect();
    for (&i, s) in miss.iter().zip(computed) {
        memo.pairs
            .insert(pair_key(pairs[i].0, pairs[i].1), (s, gen));
        scores[i] = Some(s);
    }
    memo.pairs.retain(|_, v| v.1 == gen);

    let mut out: Vec<MergeCandidate> = pairs
        .iter()
        .zip(scores)
        .filter_map(|(&(a, b, n_a, n_b, amp_mean_delta, complement), s)| {
            let s = s?;
            // Missing amplitudes are absent evidence, not a perfect match:
            // the amplitude terms contribute nothing.
            let has_amps =
                !session.spike_amplitudes(a).is_empty() && !session.spike_amplitudes(b).is_empty();
            let (amp_sim, amp_mean_sim) = if has_amps {
                (
                    (1.0 - s.amp_ks).clamp(0.0, 1.0),
                    (1.0 - amp_mean_delta).clamp(0.0, 1.0),
                )
            } else {
                (0.0, 0.0)
            };

            // Weighted combination — refractory dip is the strongest single
            // signal (0.6); amplitude evidence rounds it out (0.3 + 0.1).
            let score = (0.6 * s.dip_score
                + 0.3 * amp_sim
                + 0.1 * amp_mean_sim
                + DRIFT_WEIGHT * complement)
                .clamp(0.0, 1.0);
            if score < cfg.min_score {
                return None;
            }
            Some(MergeCandidate {
                a,
                b,
                score,
                ccg_dip_z: s.z,
                amp_ks: s.amp_ks,
                amp_mean_delta,
                n_a,
                n_b,
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

/// Amplitude residuals after removing the linear dependence on
/// `ln(preceding ISI in ms)` (the first spike has no ISI and is skipped),
/// shifted back to the mean amplitude, with the fitted slope. `None` for
/// fewer than 9 spikes or no ISI variance. Bursting cells lose amplitude
/// on short ISIs, which can make a single unit look bimodal.
fn isi_residuals(times: &[SampleIndex], amps: &[f32], sr: f32) -> Option<(f64, Vec<f32>)> {
    if times.len() != amps.len() || amps.len() < 9 {
        return None;
    }
    let x: Vec<f64> = times
        .windows(2)
        .map(|w| ((w[1].0 - w[0].0) as f64 / sr as f64 * 1e3).max(1e-3).ln())
        .collect();
    let y = &amps[1..];
    let nf = x.len() as f64;
    let mx = x.iter().sum::<f64>() / nf;
    let my = y.iter().map(|&v| v as f64).sum::<f64>() / nf;
    let sxx: f64 = x.iter().map(|v| (v - mx) * (v - mx)).sum();
    if sxx < 1e-9 {
        return None;
    }
    let sxy: f64 = x
        .iter()
        .zip(y)
        .map(|(v, &w)| (v - mx) * (w as f64 - my))
        .sum();
    let b = sxy / sxx;
    let res = x
        .iter()
        .zip(y)
        .map(|(v, &w)| (w as f64 - b * (v - mx)) as f32)
        .collect();
    Some((b, res))
}

/// Score one cluster's split evidence (before the `min_score` gate).
fn score_split<P: DataProvider>(
    session: &Session<P>,
    c: ClusterId,
    globals: Option<&[Vec<u32>]>,
    refractory_samples: u64,
) -> SplitCandidate {
    let sr = session.provider.sample_rate().max(1.0);
    let total_duration = session.provider.n_samples();
    let times = session.spike_times(c);
    let amps = session.spike_amplitudes(c);
    let bc = if amps.len() >= 8 {
        bimodality_coefficient(amps)
    } else {
        0.0
    };
    // Hartigan dip corroborates the moment-based BC. Blend via `max`
    // so the dip can only strengthen a detection the BC already
    // supports (or catch a multimodal shape BC underweights), never
    // weaken one.
    let dip = if amps.len() >= 8 {
        hartigan_dip(amps)
    } else {
        0.0
    };
    // dip ≈ 0.02 is noise-level; ≳ 0.06 is a clear gap.
    let score_of = |bc: f32, dip: f32| {
        let dip_score = ((dip - 0.02) / 0.04).clamp(0.0, 1.0);
        ((bc - 0.4) / 0.4).clamp(0.0, 1.0).max(dip_score)
    };
    let mut bc_score = score_of(bc, dip);
    // Burst-aware: when the amplitudes look bimodal but the part
    // unexplained by ISI (with a physiological positive slope) does not,
    // score the residuals instead and drop the amplitude GMM proposal.
    // ponytail: linear in ln(ISI) only; no recovery-time-constant fit.
    let mut reasons = Vec::new();
    if bc_score >= 0.5 {
        if let Some((slope, res)) = isi_residuals(times, amps, sr) {
            let res_score = score_of(bimodality_coefficient(&res), hartigan_dip(&res));
            if slope > 0.0 && res_score < 0.4 {
                bc_score = res_score;
                reasons.push("burst_explained");
            }
        }
    }
    let contamination = refractory_contamination(times, refractory_samples, total_duration.0, sr);
    let cont_score = (contamination * 2.0).clamp(0.0, 1.0);
    let drift = amplitude_drift_correlation(times, amps).abs();
    let drift_score = ((drift - 0.3) / 0.5).clamp(0.0, 1.0);
    let snr = amplitude_snr(amps);
    let snr_gate = (snr / (snr + 0.5)).clamp(0.0, 1.0);

    let (gmm_delta, bipartition) = if bc >= 0.5 && amps.len() >= 50 && reasons.is_empty() {
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

    let mut raw = ((0.6 * bc_score + 0.25 * cont_score + 0.15 * drift_score) * snr_gate
        + gmm_bonus)
        .clamp(0.0, 1.0);
    // PC-feature k-means is the stronger proposer: when it passes
    // its validation it replaces the GMM bipartition and lifts the
    // score to [0.5, 1] by silhouette.
    // ponytail: hand-set mapping; calibrate on accepted splits.
    let mut method = "gmm";
    let mut bipartition = bipartition;
    if let Some(km) = globals.and_then(|g| crate::proposers::kmeans_split(session, c, &g[c.idx()]))
    {
        raw = raw.max(0.5 + (km.silhouette - 0.25).clamp(0.0, 0.5));
        bipartition = km.minor_idx;
        method = "kmeans";
    }
    SplitCandidate {
        cluster: c,
        score: raw,
        amp_bimodality: bc,
        amp_dip: dip,
        contamination,
        drift_corr: drift,
        n_spikes: times.len(),
        method,
        gmm_bic_delta: gmm_delta,
        bipartition_minor_idx: bipartition,
        reasons,
    }
}

/// Rank split candidates over the entire session. A cluster scores high if
/// its amplitude distribution is bimodal (Sarle BC > 0.555) AND/OR it shows
/// significant refractory contamination AND/OR a strong amplitude drift.
pub fn rank_split_candidates<P: DataProvider>(
    session: &Session<P>,
    cfg: &SuggestConfig,
) -> Vec<SplitCandidate> {
    rank_split_candidates_memo(session, cfg, &mut SuggestMemo::default())
}

/// [`rank_split_candidates`] reusing per-cluster scores from `memo`; the
/// result is identical to a fresh-memo run. The scorer reads nothing
/// outside the cluster's times, amplitudes and global spike indices (PC
/// features, templates and channels are fixed per global index), so that is
/// exactly what the memo key hashes.
pub fn rank_split_candidates_memo<P: DataProvider>(
    session: &Session<P>,
    cfg: &SuggestConfig,
    memo: &mut SuggestMemo,
) -> Vec<SplitCandidate> {
    use rayon::prelude::*;
    let n = session.n_clusters();
    if n == 0 {
        return Vec::new();
    }
    let sr = session.provider.sample_rate().max(1.0);
    let refractory_samples = (sr * cfg.refractory_seconds).round() as u64;
    let globals = session
        .has_pc_features()
        .then(|| crate::proposers::all_cluster_globals(session));
    let sig = [
        cfg.refractory_seconds.to_bits() as u64,
        sr.to_bits() as u64,
        session.provider.n_samples().0,
        globals.is_some() as u64,
    ];
    if memo.split_sig != sig {
        memo.splits.clear();
        memo.split_sig = sig;
    }
    memo.gen += 1;
    let gen = memo.gen;

    let keyed: Vec<(ClusterId, [u8; 32])> = (0..n)
        .into_par_iter()
        .map(ClusterId)
        .filter(|&c| {
            // Rejection memory: a `notsplit` cluster is never proposed again.
            session.spike_times(c).len() >= cfg.min_spikes
                && !session.tags(c).iter().any(|t| t == "notsplit")
        })
        .map(|c| {
            let mut h = content_key(session.spike_times(c), session.spike_amplitudes(c));
            if let Some(g) = &globals {
                feed(&mut h, &g[c.idx()], u32::to_le_bytes);
            }
            (c, *h.finalize().as_bytes())
        })
        .collect();
    let miss: Vec<&(ClusterId, [u8; 32])> = keyed
        .iter()
        .filter(|(_, k)| match memo.splits.get_mut(k) {
            Some(e) => {
                e.1 = gen;
                false
            }
            None => true,
        })
        .collect();
    let computed: Vec<SplitCandidate> = miss
        .par_iter()
        .map(|&&(c, _)| score_split(session, c, globals.as_deref(), refractory_samples))
        .collect();
    for (&&(_, k), s) in miss.iter().zip(computed) {
        memo.splits.insert(k, (s, gen));
    }
    memo.splits.retain(|_, v| v.1 == gen);

    let mut out: Vec<SplitCandidate> = keyed
        .iter()
        .filter_map(|&(c, k)| {
            let mut s = memo.splits[&k].0.clone();
            s.cluster = c;
            (s.score >= cfg.min_score).then_some(s)
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
    use crate::journal::Journal;
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
        let journal = Journal::open(&dir.path().join("j.sqlite")).unwrap();
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
    fn mean_amp_delta_is_a_fraction_of_the_larger_mean() {
        // Means 10 and 18: 8/18 = 0.44 of the larger, under the default 0.5
        // (dividing by the smaller, 0.8, used to discard the pair).
        let a_times: Vec<SampleIndex> = (0..500).map(|i| SampleIndex((i as u64) * 10)).collect();
        let b_times: Vec<SampleIndex> =
            (0..500).map(|i| SampleIndex((i as u64) * 10 + 5)).collect();
        let prov = MockProvider {
            spikes: vec![a_times, b_times],
            amps: vec![vec![10.0; 500], vec![18.0; 500]],
            n_samples: SampleIndex(10_000),
        };
        let (sess, _d) = fresh_session(prov);
        let merges = rank_merge_candidates(&sess, &SuggestConfig::default());
        let m = merges
            .iter()
            .find(|m| (m.a, m.b) == (ClusterId(0), ClusterId(1)))
            .expect("pair kept");
        assert!((m.amp_mean_delta - 8.0 / 18.0).abs() < 1e-4);
    }

    #[test]
    fn split_suggester_flags_bimodal_amplitudes() {
        // Cluster 0: bimodal amplitudes (clearly two populations).
        let mut amps_bi: Vec<f32> = Vec::new();
        amps_bi.resize(500, 2.0);
        amps_bi.resize(1000, 8.0);
        let times_bi: Vec<SampleIndex> = (0..1000).map(|i| SampleIndex((i as u64) * 10)).collect();

        // Cluster 1: genuinely unimodal amplitudes — a deterministic Gaussian
        // cloud around 5.0. (The earlier `sin()` sweep was U-shaped/arcsine,
        // i.e. genuinely *non*-unimodal, which the Hartigan dip correctly
        // flags — so it was the wrong "unimodal" control.)
        let amps_uni: Vec<f32> = {
            let mut s: u32 = 0x9E37_79B9;
            let mut out = Vec::with_capacity(1000);
            for _ in 0..1000 {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let u1 = (s as f64 / u32::MAX as f64).clamp(1e-9, 1.0);
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let u2 = (s as f64 / u32::MAX as f64).clamp(1e-9, 1.0);
                let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                out.push((5.0 + g * 0.5) as f32);
            }
            out
        };
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
    fn merge_suggester_ignores_unrelated_pair_without_amplitudes() {
        // No amplitude data and no refractory dip: nothing to go on.
        let a: Vec<SampleIndex> = (0..500).map(|i| SampleIndex(i * 10)).collect();
        let b: Vec<SampleIndex> = (0..500).map(|i| SampleIndex(i * 7 + 3)).collect();
        let prov = MockProvider {
            spikes: vec![a, b],
            amps: vec![vec![], vec![]],
            n_samples: SampleIndex(10_000),
        };
        let (sess, _d) = fresh_session(prov);
        let merges = rank_merge_candidates(&sess, &SuggestConfig::default());
        assert!(merges.is_empty(), "{merges:?}");
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

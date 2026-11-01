//! Preview-before-commit metric computation.
//!
//! Curators want to know what a candidate merge will *look like* before
//! they commit it: contamination going up is a red flag, isolation
//! collapsing is a red flag, drift becoming worse is a red flag. This
//! module simulates the post-merge state in a borrow-only way (no
//! mutation, no journal write) and returns a [`QualityBreakdown`] for the
//! hypothetical merged cluster, plus a [`MergeDelta`] that subtracts the
//! pre-merge state of the *target* (the typical comparison the UI shows).
//!
//! Implementation note: the [`ClusterIndex`] keeps per-cluster amplitude /
//! template buckets in time-sorted order. To preview a merge we synthesise
//! the union spike train + amplitude vector by walking each source's
//! bucket once with cursor-merging (same algorithm `from_provider` uses).
//! That's O(sum_of_spike_counts) and stays inside this crate — no extra
//! allocations on the session, no journal interaction.

use crate::feature_subspace::{DEFAULT_CHANNEL_IDX, DEFAULT_D, DEFAULT_MAX_BACKGROUND};
use crate::session::Session;
use sorrel_compute::{isolation_metrics, quality_breakdown, IsolationMetrics, QualityBreakdown};
use sorrel_io::{ClusterId, DataProvider, SampleIndex};

const REFRACTORY_SECONDS: f32 = 0.0015;
const PRESENCE_BINS: usize = 50;

/// Preview the result of merging `sources` into `target`. Returns the
/// hypothetical [`QualityBreakdown`] of the merged cluster *plus* the
/// pre-merge breakdown of `target`, so the UI can show a delta.
pub fn preview_merge<P: DataProvider>(
    session: &Session<P>,
    sources: &[ClusterId],
    target: ClusterId,
) -> MergePreview {
    let sr = session.provider.sample_rate().max(1.0);
    let refractory_samples = (sr * REFRACTORY_SECONDS).round() as u64;
    let total_duration = session.provider.n_samples();

    // Pre-merge target breakdown (session-level so isolation is included).
    let pre = crate::quality_ext::cluster_quality(session, target);

    // Synthesise the merged spike train + amplitude buffer.
    let mut all: Vec<ClusterId> = Vec::with_capacity(sources.len() + 1);
    all.push(target);
    for &s in sources {
        if s != target {
            all.push(s);
        }
    }
    let (times, amps) = merge_spike_data(session, &all);

    // Cheap baseline metrics from compute.
    let mut post = quality_breakdown(
        &times,
        &amps,
        refractory_samples,
        total_duration.0,
        sr,
        PRESENCE_BINS,
    );
    // Isolation requires the live PC features and a "is in cluster" mask
    // that treats every spike currently in any of `all` as in-cluster.
    post.isolation = preview_isolation(session, |_, c| all.contains(&c));

    MergePreview {
        sources: sources.to_vec(),
        target,
        pre,
        post,
    }
}

/// What splitting `cluster` by a spike set would leave behind: the quality
/// of the part that moves out (`inside`, the future new cluster), the part
/// that stays (`outside`) and the cluster as it is now (`parent`), plus both
/// parts' spike trains for ISI histograms.
#[derive(Clone, Debug)]
pub struct SplitPreview {
    pub cluster: ClusterId,
    pub inside: QualityBreakdown,
    pub outside: QualityBreakdown,
    pub parent: QualityBreakdown,
    pub inside_times: Vec<SampleIndex>,
    pub outside_times: Vec<SampleIndex>,
}

/// Preview `CurationCommand::Split { cluster, spike_idx: inside_local }`
/// without applying it: same borrow-only approach as [`preview_merge`].
/// `inside_local` indexes the cluster's time-sorted bucket; out-of-range
/// indices are ignored (as the real split does).
pub fn preview_split<P: DataProvider>(
    session: &Session<P>,
    cluster: ClusterId,
    inside_local: &[u32],
) -> SplitPreview {
    let sr = session.provider.sample_rate().max(1.0);
    let refractory_samples = (sr * REFRACTORY_SECONDS).round() as u64;
    let total = session.provider.n_samples().0;
    let times = session.spike_times(cluster);
    let amps = session.spike_amplitudes(cluster);
    let has_amps = amps.len() == times.len();

    let mut is_in = vec![false; times.len()];
    for &i in inside_local {
        if let Some(f) = is_in.get_mut(i as usize) {
            *f = true;
        }
    }
    let pick = |want: bool| -> (Vec<SampleIndex>, Vec<f32>) {
        let keep = (0..times.len()).filter(|&i| is_in[i] == want);
        let t = keep.clone().map(|i| times[i]).collect();
        let a = if has_amps {
            keep.map(|i| amps[i]).collect()
        } else {
            Vec::new()
        };
        (t, a)
    };
    // Global spike index -> 0 elsewhere, 1 inside, 2 outside (the bucket is
    // the cluster's spikes in global order, the same walk `split` does).
    let mut part = vec![0u8; session.spike_clusters_global().len()];
    let mut local = 0usize;
    for (g, &c) in session.spike_clusters_global().iter().enumerate() {
        if c == cluster {
            part[g] = if is_in.get(local).copied().unwrap_or(false) {
                1
            } else {
                2
            };
            local += 1;
        }
    }
    let quality = |(t, a): &(Vec<SampleIndex>, Vec<f32>), parts: &[u8]| {
        let mut q = quality_breakdown(t, a, refractory_samples, total, sr, PRESENCE_BINS);
        q.isolation = preview_isolation(session, |g, _| parts.contains(&part[g]));
        q
    };
    let (inside, outside) = (pick(true), pick(false));
    SplitPreview {
        cluster,
        inside: quality(&inside, &[1]),
        outside: quality(&outside, &[2]),
        parent: crate::quality_ext::cluster_quality(session, cluster),
        inside_times: inside.0,
        outside_times: outside.0,
    }
}

/// Side-by-side breakdown of one merge candidate.
#[derive(Clone, Debug)]
pub struct MergePreview {
    pub sources: Vec<ClusterId>,
    pub target: ClusterId,
    pub pre: QualityBreakdown,
    pub post: QualityBreakdown,
}

impl MergePreview {
    /// Composite delta = post - pre. Positive means the merge improves the
    /// composite score. Curators pay most attention to this single number.
    pub fn composite_delta(&self) -> f32 {
        self.post.composite() - self.pre.composite()
    }

    /// Refractory contamination delta = post - pre. Positive = worse.
    pub fn contamination_delta(&self) -> f32 {
        self.post.raw_contamination - self.pre.raw_contamination
    }

    /// True when the merge meaningfully degrades the composite score
    /// (drop > 0.05). Used by the UI to colour the "merge →" button.
    pub fn warns(&self) -> bool {
        self.composite_delta() < -0.05 || self.contamination_delta() > 0.05
    }
}

/// Synthesise the merged cluster's spike-times + amplitudes vectors by
/// cursor-merging each source's already-sorted bucket. Same algorithm the
/// real merge uses, but writing to fresh buffers instead of mutating.
fn merge_spike_data<P: DataProvider>(
    session: &Session<P>,
    clusters: &[ClusterId],
) -> (Vec<SampleIndex>, Vec<f32>) {
    let total: usize = clusters.iter().map(|&c| session.spike_times(c).len()).sum();
    let mut times = Vec::with_capacity(total);
    let mut amps = Vec::with_capacity(total);
    let mut cursors = vec![0usize; clusters.len()];
    for _ in 0..total {
        let mut best: Option<(usize, SampleIndex)> = None;
        for (i, &c) in clusters.iter().enumerate() {
            let bucket = session.spike_times(c);
            if cursors[i] < bucket.len() {
                let t = bucket[cursors[i]];
                if match best {
                    Some((_, bt)) => t < bt,
                    None => true,
                } {
                    best = Some((i, t));
                }
            }
        }
        let Some((i, t)) = best else { break };
        let cluster = clusters[i];
        times.push(t);
        let cluster_amps = session.spike_amplitudes(cluster);
        if let Some(&a) = cluster_amps.get(cursors[i]) {
            amps.push(a);
        }
        cursors[i] += 1;
    }
    if amps.len() != times.len() {
        // Backend doesn't expose amplitudes — emit empty so downstream
        // metrics treat amplitude evidence as absent.
        amps.clear();
    }
    (times, amps)
}

/// Compute isolation for a *hypothetical cluster*: `is_member(global spike
/// index, its current cluster)` says which spikes are in-cluster (a merge
/// passes the union of its clusters, a split one part of a cluster).
fn preview_isolation<P: DataProvider>(
    session: &Session<P>,
    is_member: impl Fn(usize, ClusterId) -> bool,
) -> Option<IsolationMetrics> {
    let (n_pcs, n_chans) = session.pc_shape();
    if n_pcs == 0 || n_chans == 0 {
        return None;
    }
    let d = DEFAULT_D.min(n_pcs);
    let channel_idx = DEFAULT_CHANNEL_IDX;
    let stride = n_pcs * n_chans;

    let spike_clusters = session.spike_clusters_global();
    if spike_clusters.is_empty() {
        return None;
    }
    let mut n_in = 0usize;
    let mut n_out = 0usize;
    for (g, &c) in spike_clusters.iter().enumerate() {
        if is_member(g, c) {
            n_in += 1;
        } else {
            n_out += 1;
        }
    }
    if n_in == 0 || n_out == 0 {
        return None;
    }
    let bg_keep = n_out.min(DEFAULT_MAX_BACKGROUND);
    let bg_stride = (n_out / bg_keep).max(1);

    let mut features: Vec<f32> = Vec::with_capacity((n_in + bg_keep) * d);
    let mut is_in: Vec<bool> = Vec::with_capacity(n_in + bg_keep);

    for (g, &c) in spike_clusters.iter().enumerate() {
        if !is_member(g, c) {
            continue;
        }
        let Some(feats) = session.pc_feature_for(g as u32) else {
            continue;
        };
        if feats.len() < stride {
            continue;
        }
        for pc in 0..d {
            features.push(feats[pc * n_chans + channel_idx]);
        }
        is_in.push(true);
    }
    let mut bg_seen = 0usize;
    for (g, &c) in spike_clusters.iter().enumerate() {
        if is_member(g, c) {
            continue;
        }
        let take = bg_seen.checked_rem(bg_stride) == Some(0);
        bg_seen += 1;
        if !take {
            continue;
        }
        let Some(feats) = session.pc_feature_for(g as u32) else {
            continue;
        };
        if feats.len() < stride {
            continue;
        }
        for pc in 0..d {
            features.push(feats[pc * n_chans + channel_idx]);
        }
        is_in.push(false);
    }
    if features.is_empty() {
        return None;
    }
    Some(isolation_metrics(&features, &is_in, d, 8))
}

/// Optional pre-vs-post delta on each sub-score. Useful for the "evidence"
/// sub-row in the suggest panel.
pub struct MergeDelta {
    pub composite: f32,
    pub contamination: f32,
    pub presence: f32,
    pub stability: f32,
    pub completeness: f32,
}

impl MergePreview {
    pub fn delta(&self) -> MergeDelta {
        MergeDelta {
            composite: self.composite_delta(),
            contamination: self.contamination_delta(),
            presence: self.post.presence_score - self.pre.presence_score,
            stability: self.post.stability_score - self.pre.stability_score,
            completeness: self.post.completeness_score - self.pre.completeness_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::Journal;
    use sorrel_io::{
        ClusterId, DataProvider, HasAmplitudes, SampleIndex, TraceSamples, TraceSlice,
    };

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
    impl crate::session::ApplyPhyLabel for MockProvider {
        fn label_from_op(_: crate::command::PhyLabelOp) -> u8 {
            0
        }
    }
    impl HasAmplitudes for MockProvider {
        fn spike_amplitudes(&self, c: ClusterId) -> &[f32] {
            self.amps.get(c.idx()).map(Vec::as_slice).unwrap_or(&[])
        }
    }

    fn make_session(prov: MockProvider) -> (Session<MockProvider>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open(&dir.path().join("j.sqlite")).unwrap();
        let mut s = Session::new(prov, journal);
        s.seed_amplitudes();
        (s, dir)
    }

    #[test]
    fn preview_does_not_mutate_session() {
        let prov = MockProvider {
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)],
                vec![SampleIndex(20), SampleIndex(200)],
                vec![SampleIndex(100)],
            ],
            amps: vec![vec![1.0, 2.0, 3.0], vec![1.0, 1.0], vec![5.0]],
            n_samples: SampleIndex(500),
        };
        let (s, _d) = make_session(prov);
        let pre_target = s.spike_times(ClusterId(2)).to_vec();
        let pre_n = s.n_clusters();
        let _ = preview_merge(&s, &[ClusterId(0), ClusterId(1)], ClusterId(2));
        assert_eq!(s.spike_times(ClusterId(2)), &pre_target[..]);
        assert_eq!(s.n_clusters(), pre_n);
    }

    #[test]
    fn preview_merge_produces_union_spike_count() {
        let prov = MockProvider {
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)],
                vec![SampleIndex(20), SampleIndex(200)],
                vec![SampleIndex(100)],
            ],
            amps: vec![vec![1.0, 2.0, 3.0], vec![1.0, 1.0], vec![5.0]],
            n_samples: SampleIndex(500),
        };
        let (s, _d) = make_session(prov);
        let p = preview_merge(&s, &[ClusterId(0), ClusterId(1)], ClusterId(2));
        // Union has 6 spikes — post.n_spikes should reflect that.
        assert_eq!(p.post.n_spikes, 6);
    }

    #[test]
    fn preview_merge_contamination_delta_detects_worse() {
        // Target cluster is clean; sources contaminate it.
        let prov = MockProvider {
            // Target = 0: well-spaced (clean).
            spikes: vec![
                vec![0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
                    .into_iter()
                    .map(SampleIndex)
                    .collect(),
                // Source = 1: spikes packed close to target's → many violations.
                vec![1, 1001, 2001, 3001, 4001, 5001, 6001, 7001, 8001, 9001]
                    .into_iter()
                    .map(SampleIndex)
                    .collect(),
            ],
            amps: vec![vec![5.0; 10], vec![5.0; 10]],
            n_samples: SampleIndex(10_000),
        };
        let (s, _d) = make_session(prov);
        let p = preview_merge(&s, &[ClusterId(1)], ClusterId(0));
        assert!(
            p.contamination_delta() > 0.0,
            "expected merge to worsen contamination"
        );
        assert!(p.warns());
    }

    #[test]
    fn preview_isolation_none_without_pc_features() {
        let prov = MockProvider {
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(30)],
                vec![SampleIndex(20), SampleIndex(40)],
            ],
            amps: vec![vec![1.0, 2.0], vec![1.0, 1.0]],
            n_samples: SampleIndex(500),
        };
        let (s, _d) = make_session(prov);
        let p = preview_merge(&s, &[ClusterId(1)], ClusterId(0));
        assert!(p.post.isolation.is_none());
    }
    #[test]
    fn preview_split_matches_an_applied_split() {
        use crate::command::CurationCommand;
        // 60 spikes at 1 kHz-ish spacing with a few close pairs and a
        // two-level amplitude pattern; split out every third spike.
        let make = || {
            let times: Vec<SampleIndex> = (0..60u64)
                .map(|i| SampleIndex(i * 150 + if i % 7 == 0 { 1 } else { 0 }))
                .collect();
            let amps: Vec<f32> = (0..60).map(|i| 4.0 + (i % 5) as f32 * 0.5).collect();
            MockProvider {
                spikes: vec![times, vec![SampleIndex(77), SampleIndex(5000)]],
                amps: vec![amps, vec![3.0, 3.0]],
                n_samples: SampleIndex(10_000),
            }
        };
        let inside: Vec<u32> = (0..60).step_by(3).collect();
        let (s, _d) = make_session(make());
        let before = s.spike_times(ClusterId(0)).to_vec();
        let p = preview_split(&s, ClusterId(0), &inside);
        assert_eq!(s.spike_times(ClusterId(0)), &before[..], "borrow-only");
        assert_eq!((p.inside_times.len(), p.outside_times.len()), (20, 40));

        let (mut real, _d2) = make_session(make());
        real.dispatch(CurationCommand::Split {
            cluster: ClusterId(0),
            spike_idx: inside,
            new_cluster: ClusterId(2),
            by: crate::command::Method::legacy(),
        })
        .unwrap();
        let same = |a: &QualityBreakdown, b: &QualityBreakdown| {
            assert_eq!(a.n_spikes, b.n_spikes);
            assert_eq!(a.raw_isi_violations, b.raw_isi_violations);
            assert!((a.composite() - b.composite()).abs() < 1e-6);
            assert!((a.raw_contamination - b.raw_contamination).abs() < 1e-6);
        };
        same(
            &p.inside,
            &crate::quality_ext::cluster_quality(&real, ClusterId(2)),
        );
        same(
            &p.outside,
            &crate::quality_ext::cluster_quality(&real, ClusterId(0)),
        );
        same(
            &p.parent,
            &crate::quality_ext::cluster_quality(&s, ClusterId(0)),
        );
        assert_eq!(p.inside_times, real.spike_times(ClusterId(2)));
    }
}

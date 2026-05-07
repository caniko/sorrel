//! Mutable curation surface that sits on top of an immutable `DataProvider`.
//!
//! The provider gives us the original spike→cluster assignment baked into
//! the on-disk arrays. Curation (merge/split) mutates that assignment,
//! so we keep the mutation in `ClusterIndex` and rebuild the per-cluster
//! buckets lazily — never touching the underlying mmap.
//!
//! Per-spike side data (amplitudes, spike_templates) is *immutable* — phy
//! treats those as ground truth that survives every curation decision.
//! Only the cluster id of each spike changes.

use sorrel_io::{
    ClusterId, DataProvider, HasAmplitudes, HasPcFeatures, HasSpikeTemplates,
    HasTemplateWaveforms, OptionalRow, SampleIndex,
};

/// Mutable index over the spike→cluster assignment.
///
/// Per-spike state (`spike_times`, `spike_clusters`, `spike_amplitudes`,
/// `spike_templates`) is stored as flat global arrays in time-sorted order.
/// `*_per_cluster` are derived caches that get rebuilt for individual
/// clusters whenever their membership changes.
///
/// # Examples
///
/// Merge round-trip — undo restores the pre-merge buckets exactly:
///
/// ```
/// use sorrel_data::ClusterIndex;
/// use sorrel_io::{ClusterId, DataProvider, SampleIndex, TraceSamples, TraceSlice};
///
/// struct Stub { spikes: Vec<Vec<SampleIndex>> }
/// impl DataProvider for Stub {
///     type Label = u8;
///     fn sample_rate(&self) -> f32 { 1000.0 }
///     fn n_channels(&self) -> u32 { 1 }
///     fn n_samples(&self) -> SampleIndex { SampleIndex(0) }
///     fn n_clusters(&self) -> u32 { self.spikes.len() as u32 }
///     fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
///         self.spikes.get(c.idx()).map(Vec::as_slice).unwrap_or(&[])
///     }
///     fn trace(&self, _: SampleIndex, _: u32) -> TraceSlice<'_> {
///         TraceSlice { start: SampleIndex(0), n_channels: 0, samples: TraceSamples::I16(&[]) }
///     }
///     fn initial_labels(&self) -> Vec<u8> { vec![0; self.spikes.len()] }
/// }
///
/// let mut ci = ClusterIndex::from_provider(&Stub {
///     spikes: vec![vec![SampleIndex(10), SampleIndex(30)], vec![SampleIndex(20)], vec![SampleIndex(100)]],
/// });
/// let pre = ci.spike_times(ClusterId(0)).to_vec();
///
/// let rec = ci.merge(&[ClusterId(0), ClusterId(1)], ClusterId(2));
/// assert!(ci.spike_times(ClusterId(0)).is_empty());
/// assert_eq!(
///     ci.spike_times(ClusterId(2)),
///     &[SampleIndex(10), SampleIndex(20), SampleIndex(30), SampleIndex(100)]
/// );
///
/// ci.unmerge(rec);
/// assert_eq!(ci.spike_times(ClusterId(0)), &pre[..]);
/// ```
#[derive(Debug)]
pub struct ClusterIndex {
    /// Global per-spike sample indices, sorted ascending. Length = n_spikes.
    spike_times: Vec<SampleIndex>,
    /// Mutable per-spike cluster assignment. Length = n_spikes.
    spike_clusters: Vec<ClusterId>,
    /// Per-spike amplitudes; empty when the backend can't provide them.
    spike_amplitudes: Vec<f32>,
    /// Per-spike template id; empty when not provided.
    spike_templates: Vec<u32>,
    /// Per-spike index into the *original* `pc_features.npy` row
    /// (`spike_times.npy` order). Length equals n_spikes when seeded; empty
    /// otherwise. Indexed by global time-sorted spike index, NOT original.
    /// Each entry is an [`OptionalRow`] — `OptionalRow::NONE` means the
    /// provider didn't supply a row for that spike.
    spike_pc_indices: Vec<OptionalRow>,
    /// Flat `(n_spikes_orig, n_pcs, n_channels_per_template)` row-major
    /// PC feature buffer. Empty when not seeded. Indexed via
    /// `spike_pc_indices[g] * stride`.
    pc_features_flat: Vec<f32>,
    /// `(n_pcs, n_channels_per_template)` shape of the PC feature buffer.
    pc_shape: (usize, usize),

    /// Flat `(n_templates, n_samples_per_template, n_channels)` template
    /// waveforms, copied from the provider when seeded. Empty otherwise.
    template_waveforms: Vec<f32>,
    template_shape: (usize, usize, usize),
    /// `(n_templates × n_templates)` similarity matrix, flat row-major.
    similar_templates: Vec<f32>,

    // Cached per-cluster buckets. Indexed by ClusterId; `0..n_clusters`.
    times_per_cluster: Vec<Vec<SampleIndex>>,
    amps_per_cluster: Vec<Vec<f32>>,
    templates_per_cluster: Vec<Vec<u32>>,
}

impl ClusterIndex {
    /// Build an index from any `DataProvider`. Pulls spike times only —
    /// amplitudes and templates remain empty unless added via the more
    /// specific builders below.
    pub fn from_provider<P: DataProvider>(provider: &P) -> Self {
        let n_clusters = provider.n_clusters() as usize;
        let mut times_per_cluster: Vec<Vec<SampleIndex>> = Vec::with_capacity(n_clusters);
        let mut total_spikes = 0usize;
        for c in (0..n_clusters as u32).map(ClusterId) {
            let bucket: Vec<SampleIndex> = provider.spike_times(c).to_vec();
            total_spikes += bucket.len();
            times_per_cluster.push(bucket);
        }

        // Materialise the global per-spike arrays in time order. Since each
        // bucket is already sorted ascending, we merge by repeated argmin.
        let mut spike_times = Vec::with_capacity(total_spikes);
        let mut spike_clusters = Vec::with_capacity(total_spikes);
        // Per-cluster cursors into times_per_cluster.
        let mut cursors = vec![0usize; n_clusters];
        for _ in 0..total_spikes {
            // Find the cluster whose next spike has the smallest time.
            let mut best: Option<(ClusterId, SampleIndex)> = None;
            for c in 0..n_clusters {
                if cursors[c] < times_per_cluster[c].len() {
                    let t = times_per_cluster[c][cursors[c]];
                    if best.is_none_or(|(_, bt)| t < bt) {
                        best = Some((ClusterId(c as u32), t));
                    }
                }
            }
            let (c, t) = best.expect("total_spikes counts the loop iterations exactly");
            spike_times.push(t);
            spike_clusters.push(c);
            cursors[c.idx()] += 1;
        }

        Self {
            spike_times,
            spike_clusters,
            spike_amplitudes: Vec::new(),
            spike_templates: Vec::new(),
            spike_pc_indices: Vec::new(),
            pc_features_flat: Vec::new(),
            pc_shape: (0, 0),
            template_waveforms: Vec::new(),
            template_shape: (0, 0, 0),
            similar_templates: Vec::new(),
            times_per_cluster,
            amps_per_cluster: vec![Vec::new(); n_clusters],
            templates_per_cluster: vec![Vec::new(); n_clusters],
        }
    }

    /// Pull per-spike amplitudes from a backend that has them. Discards any
    /// previously seeded amplitudes.
    pub fn seed_amplitudes<P: HasAmplitudes>(&mut self, provider: &P) {
        let n_clusters = provider.n_clusters() as usize;
        let amps_per_cluster: Vec<Vec<f32>> = (0..n_clusters as u32)
            .map(ClusterId)
            .map(|c| provider.spike_amplitudes(c).to_vec())
            .collect();
        // If the provider gave a different number of amplitudes than spikes
        // for a cluster, fall back to NaN so the consumer can notice without
        // a panic.
        self.spike_amplitudes = rebucket_to_global(
            &self.spike_clusters,
            &amps_per_cluster,
            |v| v,
            f32::NAN,
        );
        self.amps_per_cluster = amps_per_cluster;
    }

    /// Pull PC features and the per-spike NPY-row mapping from a backend
    /// that has them. The flat features buffer is copied verbatim from the
    /// provider; per-spike NPY-row indices are reordered to match the
    /// global time-sorted spike layout we use everywhere else.
    pub fn seed_pc_features<P: HasPcFeatures>(&mut self, provider: &P) {
        let n_clusters = provider.n_clusters() as usize;
        let pc_indices_per_cluster: Vec<Vec<u32>> = (0..n_clusters as u32)
            .map(ClusterId)
            .map(|c| provider.spike_pc_indices(c).to_vec())
            .collect();
        self.spike_pc_indices = rebucket_to_global(
            &self.spike_clusters,
            &pc_indices_per_cluster,
            OptionalRow::new,
            OptionalRow::NONE,
        );
        self.pc_features_flat = provider.pc_features().to_vec();
        self.pc_shape = provider.pc_shape();
    }

    /// Pull per-spike template ids from a backend that has them. Same shape
    /// as [`Self::seed_amplitudes`].
    pub fn seed_templates<P: HasSpikeTemplates>(&mut self, provider: &P) {
        let n_clusters = provider.n_clusters() as usize;
        let templates_per_cluster: Vec<Vec<u32>> = (0..n_clusters as u32)
            .map(ClusterId)
            .map(|c| provider.spike_templates(c).to_vec())
            .collect();
        self.spike_templates = rebucket_to_global(
            &self.spike_clusters,
            &templates_per_cluster,
            |v| v,
            u32::MAX,
        );
        self.templates_per_cluster = templates_per_cluster;
    }

    /// Total number of clusters currently addressable. May grow when split
    /// allocates a new cluster id.
    #[inline]
    pub fn n_clusters(&self) -> u32 {
        self.times_per_cluster.len() as u32
    }

    /// Number of spikes globally (sum across all clusters).
    #[inline]
    pub fn n_spikes(&self) -> usize {
        self.spike_times.len()
    }

    /// Read-only access to the global per-spike cluster assignment. Used by
    /// the save-to-phy writer.
    #[inline]
    pub fn spike_clusters(&self) -> &[ClusterId] {
        &self.spike_clusters
    }

    #[inline]
    pub fn spike_times(&self, cluster: ClusterId) -> &[SampleIndex] {
        self.times_per_cluster
            .get(cluster.idx())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    #[inline]
    pub fn spike_amplitudes(&self, cluster: ClusterId) -> &[f32] {
        self.amps_per_cluster
            .get(cluster.idx())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    #[inline]
    pub fn spike_templates(&self, cluster: ClusterId) -> &[u32] {
        self.templates_per_cluster
            .get(cluster.idx())
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Flat `(n_spikes_orig, n_pcs, n_channels_per_template)` PC features.
    /// Empty when [`Self::seed_pc_features`] was never called.
    #[inline]
    pub fn pc_features_flat(&self) -> &[f32] {
        &self.pc_features_flat
    }

    /// `(n_pcs, n_channels_per_template)` for the PC feature buffer.
    /// Returns `(0, 0)` when features aren't seeded.
    #[inline]
    pub fn pc_shape(&self) -> (usize, usize) {
        self.pc_shape
    }

    /// For each global (time-sorted) spike index, the row in `pc_features_flat`
    /// that holds its features. Length 0 when features aren't seeded. Use
    /// [`OptionalRow::get`] on entries to handle the "missing" sentinel.
    #[inline]
    pub fn spike_pc_indices(&self) -> &[OptionalRow] {
        &self.spike_pc_indices
    }

    /// PC feature slice (length = `n_pcs * n_channels_per_template`) for the
    /// spike at `global_idx`. `None` when features aren't seeded or the
    /// per-spike mapping points outside the buffer.
    pub fn pc_feature_for(&self, global_idx: u32) -> Option<&[f32]> {
        let (n_pcs, n_chans) = self.pc_shape;
        let stride = n_pcs * n_chans;
        if stride == 0 {
            return None;
        }
        let orig = self.spike_pc_indices.get(global_idx as usize)?.get()? as usize;
        let start = orig * stride;
        let end = start + stride;
        self.pc_features_flat.get(start..end)
    }

    /// Pull template waveforms + similarity matrix from a backend that has
    /// them. Templates are immutable across curation, so we just copy them
    /// in once.
    pub fn seed_template_waveforms<P: HasTemplateWaveforms>(&mut self, provider: &P) {
        self.template_waveforms = provider.template_waveforms().to_vec();
        self.template_shape = provider.template_shape();
        self.similar_templates = provider.similar_templates().to_vec();
    }

    /// Flat `(n_templates, n_samples, n_channels)` template buffer. Empty
    /// when not seeded.
    #[inline]
    pub fn template_waveforms(&self) -> &[f32] {
        &self.template_waveforms
    }

    #[inline]
    pub fn template_shape(&self) -> (usize, usize, usize) {
        self.template_shape
    }

    /// Single template's waveform — `&[n_samples * n_channels]`. Returns
    /// `None` if templates aren't seeded or `template_id` is out of range.
    pub fn template_waveform(&self, template_id: u32) -> Option<&[f32]> {
        let (n_templates, n_samples, n_channels) = self.template_shape;
        if n_templates == 0 || (template_id as usize) >= n_templates {
            return None;
        }
        let stride = n_samples * n_channels;
        let start = (template_id as usize) * stride;
        self.template_waveforms.get(start..start + stride)
    }

    /// `n_templates × n_templates` similarity matrix, flat row-major.
    /// Empty slice when not seeded.
    #[inline]
    pub fn similar_templates(&self) -> &[f32] {
        &self.similar_templates
    }

    /// Allocate a fresh `ClusterId` (used by split). Grows the internal
    /// bucket vectors by one.
    pub fn allocate_cluster(&mut self) -> ClusterId {
        let id = ClusterId(self.times_per_cluster.len() as u32);
        self.times_per_cluster.push(Vec::new());
        self.amps_per_cluster.push(Vec::new());
        self.templates_per_cluster.push(Vec::new());
        id
    }

    /// Reassign every spike currently in `sources` to `target`, returning a
    /// record sufficient for [`Self::unmerge`].
    ///
    /// Sources equal to `target` are skipped (no-op). Out-of-range cluster
    /// ids are silently ignored.
    pub fn merge(&mut self, sources: &[ClusterId], target: ClusterId) -> MergeRecord {
        // Capture pre-merge state for the touched clusters.
        let mut snapshots = Vec::with_capacity(sources.len() + 1);
        let mut affected = Vec::with_capacity(sources.len());
        if target.idx() < self.times_per_cluster.len() {
            snapshots.push(self.snapshot(target));
        }
        for &s in sources {
            if s == target {
                continue;
            }
            if s.idx() >= self.times_per_cluster.len() {
                continue;
            }
            snapshots.push(self.snapshot(s));
            affected.push(s);
        }

        // Mutate the global assignment.
        let needs_amps = !self.spike_amplitudes.is_empty();
        let needs_templates = !self.spike_templates.is_empty();
        let _ = (needs_amps, needs_templates); // assignments are by-cluster only

        for c in &mut self.spike_clusters {
            if affected.contains(c) {
                *c = target;
            }
        }
        // Rebuild buckets for target + each affected source.
        self.rebuild_bucket(target);
        for &s in &affected {
            self.rebuild_bucket(s);
        }

        MergeRecord { snapshots }
    }

    /// Reverse a previously recorded merge: restore the snapshots verbatim.
    pub fn unmerge(&mut self, record: MergeRecord) {
        for snap in record.snapshots {
            self.restore_snapshot(snap);
        }
    }

    /// Move the spikes at `local_idx` (indices into the source cluster's
    /// own bucket) to a freshly-allocated cluster. Returns a record that
    /// undoes the split.
    ///
    /// Indices out of range are skipped. If `local_idx` is empty the result
    /// still allocates a new (empty) cluster — phy does the same.
    pub fn split(&mut self, source: ClusterId, local_idx: &[u32]) -> SplitRecord {
        let new_cluster = self.allocate_cluster();
        // Capture pre-split state for source AND the just-allocated bucket
        // (which is empty, but record it for symmetry on undo).
        let pre_source = self.snapshot(source);
        let pre_new = self.snapshot(new_cluster);

        // Translate local indices to global spike indices via the source's
        // bucket. (Local i -> global = position of the i-th spike of the
        // source in the global arrays. Easiest path: rebuild a small mapping
        // by walking the global arrays once.)
        let global_idx: Vec<u32> = self.global_indices_for(source);

        // Reassign the chosen spikes.
        for &li in local_idx {
            if let Some(&gi) = global_idx.get(li as usize) {
                self.spike_clusters[gi as usize] = new_cluster;
            }
        }

        self.rebuild_bucket(source);
        self.rebuild_bucket(new_cluster);

        SplitRecord {
            source,
            new_cluster,
            pre_source,
            pre_new,
        }
    }

    /// Reverse a previously recorded split: restore source & new-cluster
    /// snapshots, then truncate `times_per_cluster` etc. back if the split
    /// allocated the cluster id at the tail (the common case).
    pub fn unsplit(&mut self, record: SplitRecord) {
        self.restore_snapshot(record.pre_source);
        self.restore_snapshot(record.pre_new);
        // If `new_cluster` is the last id and is now empty, free the slot
        // so n_clusters returns to its pre-split value.
        let last = self.times_per_cluster.len() as u32;
        if record.new_cluster.0 + 1 == last
            && self.times_per_cluster[record.new_cluster.idx()].is_empty()
        {
            self.times_per_cluster.pop();
            self.amps_per_cluster.pop();
            self.templates_per_cluster.pop();
        }
    }

    fn snapshot(&self, c: ClusterId) -> ClusterSnapshot {
        ClusterSnapshot {
            cluster: c,
            global_indices: self.global_indices_for(c),
            times: self.times_per_cluster[c.idx()].clone(),
            amps: self
                .amps_per_cluster
                .get(c.idx())
                .cloned()
                .unwrap_or_default(),
            templates: self
                .templates_per_cluster
                .get(c.idx())
                .cloned()
                .unwrap_or_default(),
        }
    }

    fn restore_snapshot(&mut self, snap: ClusterSnapshot) {
        // Re-mark every previously-cached global spike as belonging to this
        // cluster. The corresponding bucket on the *current* state will be
        // rebuilt by `rebuild_bucket` at the end.
        for &gi in &snap.global_indices {
            self.spike_clusters[gi as usize] = snap.cluster;
        }
        let c = snap.cluster.idx();
        if c >= self.times_per_cluster.len() {
            // Snapshot referenced an id beyond current bounds — grow.
            self.times_per_cluster.resize_with(c + 1, Vec::new);
            self.amps_per_cluster.resize_with(c + 1, Vec::new);
            self.templates_per_cluster.resize_with(c + 1, Vec::new);
        }
        self.times_per_cluster[c] = snap.times;
        self.amps_per_cluster[c] = snap.amps;
        self.templates_per_cluster[c] = snap.templates;
    }

    /// Sorted list of global spike indices currently belonging to `cluster`.
    fn global_indices_for(&self, cluster: ClusterId) -> Vec<u32> {
        self.spike_clusters
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| (c == cluster).then_some(i as u32))
            .collect()
    }

    /// Rebuild the cached bucket for `cluster` from the global per-spike
    /// arrays. The result is sorted by time (the global arrays are already
    /// in time order).
    fn rebuild_bucket(&mut self, cluster: ClusterId) {
        let c = cluster.idx();
        if c >= self.times_per_cluster.len() {
            return;
        }
        let has_amps = !self.spike_amplitudes.is_empty();
        let has_templates = !self.spike_templates.is_empty();

        let mut times = Vec::new();
        let mut amps: Vec<f32> = Vec::new();
        let mut templates: Vec<u32> = Vec::new();
        for (i, &cl) in self.spike_clusters.iter().enumerate() {
            if cl == cluster {
                times.push(self.spike_times[i]);
                if has_amps {
                    amps.push(self.spike_amplitudes[i]);
                }
                if has_templates {
                    templates.push(self.spike_templates[i]);
                }
            }
        }
        self.times_per_cluster[c] = times;
        self.amps_per_cluster[c] = amps;
        self.templates_per_cluster[c] = templates;
    }
}

/// Walk the global time-sorted `spike_clusters` array and emit one item per
/// spike, drawn from the per-cluster bucket via a per-cluster cursor. This
/// is the merge step that turns provider-side per-cluster vectors into the
/// flat global array.
///
/// `wrap` runs on each successfully-popped value (typically `identity` or
/// `OptionalRow::new`); `missing` is used when a cluster's bucket is shorter
/// than its spike count (a robustness path against malformed providers).
fn rebucket_to_global<S, T, W>(
    spike_clusters: &[ClusterId],
    per_cluster: &[Vec<S>],
    mut wrap: W,
    missing: T,
) -> Vec<T>
where
    S: Copy,
    T: Copy,
    W: FnMut(S) -> T,
{
    let mut cursors = vec![0usize; per_cluster.len()];
    let mut out = Vec::with_capacity(spike_clusters.len());
    for &c in spike_clusters {
        let i = cursors[c.idx()];
        let v = per_cluster[c.idx()]
            .get(i)
            .copied()
            .map_or(missing, &mut wrap);
        out.push(v);
        cursors[c.idx()] += 1;
    }
    out
}

/// Snapshot of a single cluster's complete state, used by both merge and
/// split inverses to roll back without losing spike alignment.
#[derive(Clone, Debug)]
struct ClusterSnapshot {
    cluster: ClusterId,
    /// Global spike indices that were assigned to `cluster` at snapshot time.
    /// Preserved here so that on restore we can re-assign exactly those.
    global_indices: Vec<u32>,
    times: Vec<SampleIndex>,
    amps: Vec<f32>,
    templates: Vec<u32>,
}

/// Reversible record of a merge.
#[derive(Clone, Debug)]
pub struct MergeRecord {
    snapshots: Vec<ClusterSnapshot>,
}

/// Reversible record of a split.
#[derive(Clone, Debug)]
pub struct SplitRecord {
    pub source: ClusterId,
    pub new_cluster: ClusterId,
    pre_source: ClusterSnapshot,
    pre_new: ClusterSnapshot,
}

#[cfg(test)]
mod tests {
    use super::*;
    use sorrel_io::{ClusterId, DataProvider, SampleIndex, TraceSamples, TraceSlice};

    /// Tiny provider used to seed a `ClusterIndex` deterministically.
    struct StubProvider {
        spikes: Vec<Vec<SampleIndex>>,
    }

    impl DataProvider for StubProvider {
        type Label = u8;
        fn sample_rate(&self) -> f32 {
            1000.0
        }
        fn n_channels(&self) -> u32 {
            1
        }
        fn n_samples(&self) -> SampleIndex {
            SampleIndex(0)
        }
        fn n_clusters(&self) -> u32 {
            self.spikes.len() as u32
        }
        fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
            self.spikes
                .get(c.idx())
                .map(Vec::as_slice)
                .unwrap_or(&[])
        }
        fn trace(&self, _: SampleIndex, _: u32) -> TraceSlice<'_> {
            TraceSlice {
                start: SampleIndex(0),
                n_channels: 0,
                samples: TraceSamples::I16(&[]),
            }
        }
        fn initial_labels(&self) -> Vec<u8> {
            vec![0; self.spikes.len()]
        }
    }

    fn cidx() -> ClusterIndex {
        // 3 clusters, deliberately interleaved in time so the merge cursor
        // gets exercised.
        ClusterIndex::from_provider(&StubProvider {
            spikes: vec![vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)], vec![SampleIndex(20), SampleIndex(200)], vec![SampleIndex(100)]],
        })
    }

    #[test]
    fn from_provider_builds_buckets_and_global_arrays() {
        let ci = cidx();
        assert_eq!(ci.n_clusters(), 3);
        assert_eq!(ci.n_spikes(), 6);
        assert_eq!(ci.spike_times(ClusterId(0)), &[SampleIndex(10), SampleIndex(30), SampleIndex(150)]);
        assert_eq!(ci.spike_times(ClusterId(1)), &[SampleIndex(20), SampleIndex(200)]);
        assert_eq!(ci.spike_times(ClusterId(2)), &[SampleIndex(100)]);
        // Global cluster ids in time order:
        // t=10 c=0, t=20 c=1, t=30 c=0, t=100 c=2, t=150 c=0, t=200 c=1.
        assert_eq!(ci.spike_clusters(), &[ClusterId(0), ClusterId(1), ClusterId(0), ClusterId(2), ClusterId(0), ClusterId(1)]);
    }

    #[test]
    fn merge_reassigns_spikes_to_target_and_keeps_buckets_time_sorted() {
        let mut ci = cidx();
        let _rec = ci.merge(&[ClusterId(0), ClusterId(1)], ClusterId(2));
        assert!(ci.spike_times(ClusterId(0)).is_empty());
        assert!(ci.spike_times(ClusterId(1)).is_empty());
        // Target now contains every spike, in ascending time order.
        assert_eq!(ci.spike_times(ClusterId(2)), &[SampleIndex(10), SampleIndex(20), SampleIndex(30), SampleIndex(100), SampleIndex(150), SampleIndex(200)]);
        assert_eq!(ci.spike_clusters(), &[ClusterId(2), ClusterId(2), ClusterId(2), ClusterId(2), ClusterId(2), ClusterId(2)]);
    }

    #[test]
    fn unmerge_restores_buckets_exactly() {
        let mut ci = cidx();
        let pre_buckets: Vec<Vec<SampleIndex>> = (0..ci.n_clusters())
            .map(ClusterId)
            .map(|c| ci.spike_times(c).to_vec())
            .collect();
        let pre_assign = ci.spike_clusters().to_vec();
        let rec = ci.merge(&[ClusterId(0), ClusterId(1)], ClusterId(2));
        ci.unmerge(rec);
        for c in (0..ci.n_clusters()).map(ClusterId) {
            assert_eq!(ci.spike_times(c), &pre_buckets[c.idx()][..]);
        }
        assert_eq!(ci.spike_clusters(), &pre_assign[..]);
    }

    #[test]
    fn split_moves_chosen_spikes_to_a_fresh_cluster() {
        let mut ci = cidx();
        // Cluster 0 has [10,30,150]; move the middle one.
        let rec = ci.split(ClusterId(0), &[1]);
        assert_eq!(rec.source, ClusterId(0));
        assert_eq!(rec.new_cluster, ClusterId(3));
        assert_eq!(ci.n_clusters(), 4);
        assert_eq!(ci.spike_times(ClusterId(0)), &[SampleIndex(10), SampleIndex(150)]);
        assert_eq!(ci.spike_times(ClusterId(3)), &[SampleIndex(30)]);
    }

    #[test]
    fn unsplit_restores_state_and_frees_trailing_cluster_id() {
        let mut ci = cidx();
        let n_before = ci.n_clusters();
        let pre_source = ci.spike_times(ClusterId(0)).to_vec();
        let pre_assign = ci.spike_clusters().to_vec();

        let rec = ci.split(ClusterId(0), &[0, 2]);
        ci.unsplit(rec);

        assert_eq!(ci.n_clusters(), n_before);
        assert_eq!(ci.spike_times(ClusterId(0)), &pre_source[..]);
        assert_eq!(ci.spike_clusters(), &pre_assign[..]);
    }

    #[test]
    fn split_with_out_of_range_local_idx_is_a_noop_for_those_indices() {
        let mut ci = cidx();
        // Cluster 2 has 1 spike; idx 99 is bogus.
        let rec = ci.split(ClusterId(2), &[99]);
        assert_eq!(ci.spike_times(ClusterId(2)), &[SampleIndex(100)]);
        assert!(ci.spike_times(rec.new_cluster).is_empty());
    }

    #[test]
    fn merge_with_target_in_sources_is_a_noop_for_target() {
        let mut ci = cidx();
        // sources include target=2; only 0 and 1 should be merged in.
        let _rec = ci.merge(&[ClusterId(0), ClusterId(1), ClusterId(2)], ClusterId(2));
        assert_eq!(ci.spike_times(ClusterId(2)), &[SampleIndex(10), SampleIndex(20), SampleIndex(30), SampleIndex(100), SampleIndex(150), SampleIndex(200)]);
    }

    #[test]
    fn seed_amplitudes_round_trips_per_cluster_and_globally() {
        struct WithAmps {
            inner: StubProvider,
            amps: Vec<Vec<f32>>,
        }
        impl DataProvider for WithAmps {
            type Label = u8;
            fn sample_rate(&self) -> f32 {
                self.inner.sample_rate()
            }
            fn n_channels(&self) -> u32 {
                self.inner.n_channels()
            }
            fn n_samples(&self) -> SampleIndex {
                self.inner.n_samples()
            }
            fn n_clusters(&self) -> u32 {
                self.inner.n_clusters()
            }
            fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
                self.inner.spike_times(c)
            }
            fn trace(&self, s: SampleIndex, l: u32) -> TraceSlice<'_> {
                self.inner.trace(s, l)
            }
            fn initial_labels(&self) -> Vec<u8> {
                self.inner.initial_labels()
            }
        }
        impl HasAmplitudes for WithAmps {
            fn spike_amplitudes(&self, c: ClusterId) -> &[f32] {
                self.amps
                    .get(c.idx())
                    .map(Vec::as_slice)
                    .unwrap_or(&[])
            }
        }
        let prov = WithAmps {
            inner: StubProvider {
                spikes: vec![vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)], vec![SampleIndex(20), SampleIndex(200)], vec![SampleIndex(100)]],
            },
            amps: vec![vec![1.0, 3.0, 6.0], vec![2.0, 5.0], vec![4.0]],
        };
        let mut ci = ClusterIndex::from_provider(&prov);
        ci.seed_amplitudes(&prov);
        assert_eq!(ci.spike_amplitudes(ClusterId(0)), &[1.0, 3.0, 6.0]);
        assert_eq!(ci.spike_amplitudes(ClusterId(1)), &[2.0, 5.0]);
        assert_eq!(ci.spike_amplitudes(ClusterId(2)), &[4.0]);

        // After merging cluster 0 into 2, cluster 2's amplitudes should
        // contain {4.0, 1.0, 3.0, 6.0} in time order: t=10 a=1, t=30 a=3,
        // t=100 a=4, t=150 a=6.
        let _ = ci.merge(&[ClusterId(0)], ClusterId(2));
        assert_eq!(ci.spike_amplitudes(ClusterId(2)), &[1.0, 3.0, 4.0, 6.0]);
    }

    /// Sequence: split → merge → undo merge → undo split. The state must
    /// match initial after both undos.
    #[test]
    fn split_then_merge_then_undo_both_returns_to_initial() {
        let mut ci = cidx();
        let pre_buckets: Vec<Vec<SampleIndex>> = (0..ci.n_clusters())
            .map(ClusterId)
            .map(|c| ci.spike_times(c).to_vec())
            .collect();
        let pre_assign = ci.spike_clusters().to_vec();

        let split_rec = ci.split(ClusterId(0), &[1]); // splits one spike off cluster 0
        let new_id = split_rec.new_cluster;
        let merge_rec = ci.merge(&[new_id], ClusterId(2)); // merge that new cluster into 2

        // Undo in reverse order.
        ci.unmerge(merge_rec);
        ci.unsplit(split_rec);

        for c in (0..ci.n_clusters()).map(ClusterId) {
            assert_eq!(
                ci.spike_times(c),
                &pre_buckets[c.idx()][..],
                "bucket {c} not restored",
            );
        }
        assert_eq!(ci.spike_clusters(), &pre_assign[..]);
    }

    /// Two consecutive splits produce two distinct new cluster ids; undoing
    /// in reverse frees both.
    #[test]
    fn two_splits_allocate_distinct_ids_and_free_in_reverse() {
        let mut ci = cidx();
        let n0 = ci.n_clusters();
        let r1 = ci.split(ClusterId(0), &[0]);
        let r2 = ci.split(ClusterId(0), &[0]);
        assert_ne!(r1.new_cluster, r2.new_cluster);
        assert_eq!(ci.n_clusters(), n0 + 2);
        ci.unsplit(r2);
        ci.unsplit(r1);
        assert_eq!(ci.n_clusters(), n0);
    }

    /// Merging into a target whose bucket already had spikes and then
    /// undoing must restore the *target's* prior contents exactly.
    #[test]
    fn merge_preserves_pre_target_spikes_through_undo() {
        let mut ci = cidx();
        let pre_target = ci.spike_times(ClusterId(2)).to_vec();
        let rec = ci.merge(&[ClusterId(0), ClusterId(1)], ClusterId(2));
        // Target should now have everything; not equal to pre.
        assert_ne!(ci.spike_times(ClusterId(2)), &pre_target[..]);
        ci.unmerge(rec);
        assert_eq!(ci.spike_times(ClusterId(2)), &pre_target[..]);
    }

    /// Splitting *all* of a cluster's spikes leaves the source bucket empty.
    #[test]
    fn split_all_leaves_source_empty() {
        let mut ci = cidx();
        let n0 = ci.spike_times(ClusterId(0)).len();
        let local_idx: Vec<u32> = (0..n0 as u32).collect();
        let rec = ci.split(ClusterId(0), &local_idx);
        assert!(ci.spike_times(ClusterId(0)).is_empty());
        assert_eq!(ci.spike_times(rec.new_cluster).len(), n0);
    }

    /// Splitting an empty index list creates a new (empty) cluster but
    /// changes nothing else.
    #[test]
    fn split_empty_idx_only_allocates_a_cluster() {
        let mut ci = cidx();
        let pre = ci.spike_times(ClusterId(0)).to_vec();
        let rec = ci.split(ClusterId(0), &[]);
        assert_eq!(ci.spike_times(ClusterId(0)), &pre[..]);
        assert!(ci.spike_times(rec.new_cluster).is_empty());
    }

    /// `n_spikes` is preserved across any sequence of merges and splits.
    #[test]
    fn total_spike_count_is_preserved_under_curation() {
        let mut ci = cidx();
        let total = ci.n_spikes();
        let _ = ci.merge(&[ClusterId(0)], ClusterId(1));
        assert_eq!(ci.n_spikes(), total);
        let _ = ci.split(ClusterId(1), &[0, 2]);
        assert_eq!(ci.n_spikes(), total);
        let _ = ci.merge(&[ClusterId(1), ClusterId(2)], ClusterId(0));
        assert_eq!(ci.n_spikes(), total);
    }

    /// Per-cluster bucket sums equal n_spikes globally.
    #[test]
    fn bucket_sum_equals_n_spikes() {
        let ci = cidx();
        let sum: usize = (0..ci.n_clusters())
            .map(ClusterId)
            .map(|c| ci.spike_times(c).len())
            .sum();
        assert_eq!(sum, ci.n_spikes());
    }

    /// Each bucket's times are sorted ascending, before AND after curation.
    #[test]
    fn buckets_are_always_sorted_ascending() {
        let mut ci = cidx();
        for c in (0..ci.n_clusters()).map(ClusterId) {
            let bucket = ci.spike_times(c);
            for w in bucket.windows(2) {
                assert!(w[0] <= w[1], "bucket {c} not sorted");
            }
        }
        let _ = ci.merge(&[ClusterId(0), ClusterId(1)], ClusterId(2));
        let bucket = ci.spike_times(ClusterId(2));
        for w in bucket.windows(2) {
            assert!(w[0] <= w[1], "merged bucket not sorted");
        }
    }

    #[test]
    fn merge_with_self_only_is_a_noop() {
        let mut ci = cidx();
        let pre = ci.spike_clusters().to_vec();
        let _ = ci.merge(&[ClusterId(2)], ClusterId(2));
        assert_eq!(ci.spike_clusters(), &pre[..]);
    }

    #[test]
    fn merge_into_empty_target_takes_all_spikes() {
        let mut ci = cidx();
        // Free up cluster 2 to use as a target after pulling its single spike.
        let _ = ci.merge(&[ClusterId(2)], ClusterId(0));
        // Now 2 is empty. Merge 0 into 2.
        let _ = ci.merge(&[ClusterId(0)], ClusterId(2));
        assert_eq!(ci.spike_times(ClusterId(2)), &[SampleIndex(10), SampleIndex(30), SampleIndex(100), SampleIndex(150)]);
        assert!(ci.spike_times(ClusterId(0)).is_empty());
    }

    #[test]
    fn merge_with_out_of_range_source_is_ignored() {
        let mut ci = cidx();
        let pre_target = ci.spike_times(ClusterId(2)).to_vec();
        let _ = ci.merge(&[ClusterId(99)], ClusterId(2));
        assert_eq!(ci.spike_times(ClusterId(2)), &pre_target[..]);
    }

    #[test]
    fn split_with_duplicate_indices_moves_each_only_once() {
        let mut ci = cidx();
        // Cluster 0 has 3 spikes; ask to move local index 1 twice.
        let rec = ci.split(ClusterId(0), &[1, 1]);
        assert_eq!(ci.spike_times(ClusterId(0)).len(), 2);
        assert_eq!(ci.spike_times(rec.new_cluster).len(), 1);
    }

    #[test]
    fn deep_merge_split_undo_chain_is_consistent() {
        let mut ci = cidx();
        let pre = ci.spike_clusters().to_vec();
        let pre_n = ci.n_clusters();

        // merge → split → merge → unsplit → unmerge → unmerge
        let r1 = ci.merge(&[ClusterId(0)], ClusterId(2));
        let r2 = ci.split(ClusterId(2), &[0, 2]);
        let r3 = ci.merge(&[ClusterId(1)], r2.new_cluster);

        // Reverse all in LIFO order.
        // Need pristine snapshots before each subsequent inverse since the
        // index state changes; record types are owned, so we already have them.
        ci.unmerge(r3);
        ci.unsplit(r2);
        ci.unmerge(r1);

        assert_eq!(ci.n_clusters(), pre_n);
        assert_eq!(ci.spike_clusters(), &pre[..]);
    }

    #[test]
    fn merge_total_count_in_target_equals_sum_of_sources() {
        let mut ci = cidx();
        let s0 = ci.spike_times(ClusterId(0)).len();
        let s1 = ci.spike_times(ClusterId(1)).len();
        let s2 = ci.spike_times(ClusterId(2)).len();
        let _ = ci.merge(&[ClusterId(0), ClusterId(1)], ClusterId(2));
        assert_eq!(ci.spike_times(ClusterId(2)).len(), s0 + s1 + s2);
    }

    #[test]
    fn split_then_split_again_on_new_cluster() {
        let mut ci = cidx();
        let r1 = ci.split(ClusterId(0), &[0, 1]); // moves first two spikes off cluster 0
        let n_in_new = ci.spike_times(r1.new_cluster).len();
        assert_eq!(n_in_new, 2);
        let r2 = ci.split(r1.new_cluster, &[0]);
        assert_eq!(ci.spike_times(r1.new_cluster).len(), n_in_new - 1);
        assert_eq!(ci.spike_times(r2.new_cluster).len(), 1);
    }

    #[test]
    fn allocate_cluster_is_monotonic() {
        let mut ci = cidx();
        let n = ci.n_clusters();
        let a = ci.allocate_cluster();
        let b = ci.allocate_cluster();
        assert_eq!(a, ClusterId(n));
        assert_eq!(b, ClusterId(n + 1));
        assert_eq!(ci.n_clusters(), n + 2);
        assert!(ci.spike_times(a).is_empty());
        assert!(ci.spike_times(b).is_empty());
    }

    #[test]
    fn merge_then_unmerge_idempotent_under_repetition() {
        let mut ci = cidx();
        let pre = ci.spike_clusters().to_vec();
        for _ in 0..5 {
            let rec = ci.merge(&[ClusterId(0), ClusterId(1)], ClusterId(2));
            ci.unmerge(rec);
            assert_eq!(ci.spike_clusters(), &pre[..]);
        }
    }

    /// PC features + their per-spike row indices round-trip via seed.
    #[test]
    fn seed_pc_features_aligns_with_time_sorted_spikes() {
        use sorrel_io::HasPcFeatures;

        struct WithPc {
            inner: StubProvider,
            pc: Vec<f32>,
            n_pcs: usize,
            n_chans: usize,
            ind: Vec<u32>,
            indices_per_cluster: Vec<Vec<u32>>,
        }
        impl DataProvider for WithPc {
            type Label = u8;
            fn sample_rate(&self) -> f32 {
                self.inner.sample_rate()
            }
            fn n_channels(&self) -> u32 {
                self.inner.n_channels()
            }
            fn n_samples(&self) -> SampleIndex {
                self.inner.n_samples()
            }
            fn n_clusters(&self) -> u32 {
                self.inner.n_clusters()
            }
            fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
                self.inner.spike_times(c)
            }
            fn trace(&self, s: SampleIndex, l: u32) -> sorrel_io::TraceSlice<'_> {
                self.inner.trace(s, l)
            }
            fn initial_labels(&self) -> Vec<u8> {
                self.inner.initial_labels()
            }
        }
        impl HasPcFeatures for WithPc {
            fn pc_features(&self) -> &[f32] {
                &self.pc
            }
            fn pc_shape(&self) -> (usize, usize) {
                (self.n_pcs, self.n_chans)
            }
            fn pc_feature_ind(&self) -> &[u32] {
                &self.ind
            }
            fn spike_pc_indices(&self, c: ClusterId) -> &[u32] {
                self.indices_per_cluster
                    .get(c.idx())
                    .map(Vec::as_slice)
                    .unwrap_or(&[])
            }
        }

        // 6 spikes total: cluster 0 has {0, 2, 5}, cluster 1 has {1, 4}, c2 {3}.
        // (Original NPY row indices.)
        let prov = WithPc {
            inner: StubProvider {
                spikes: vec![vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)], vec![SampleIndex(20), SampleIndex(200)], vec![SampleIndex(100)]],
            },
            pc: (0..6 * 3 * 2).map(|i| i as f32).collect(), // 6 spikes × 3 PCs × 2 chans
            n_pcs: 3,
            n_chans: 2,
            ind: vec![0, 1, 2, 3, 4, 5], // identity remap
            indices_per_cluster: vec![vec![0, 2, 5], vec![1, 4], vec![3]],
        };

        let mut ci = ClusterIndex::from_provider(&prov);
        ci.seed_pc_features(&prov);
        assert_eq!(ci.pc_shape(), (3, 2));
        assert_eq!(ci.pc_features_flat().len(), 6 * 3 * 2);

        // Spot-check one PC feature for a known global index.
        let feat = ci.pc_feature_for(0).unwrap();
        assert_eq!(feat.len(), 3 * 2);
        // Global idx 0 is the time-first spike (t=10). After seeding,
        // pc_feature_for should look up the original NPY row that the
        // provider reported for that cluster's first spike — original idx 0.
        // So feat == pc[0..6].
        for (i, &v) in feat.iter().enumerate() {
            assert!((v - i as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn seed_template_waveforms_round_trips() {
        use sorrel_io::HasTemplateWaveforms;

        struct WithTpl {
            inner: StubProvider,
            templates: Vec<f32>,
            shape: (usize, usize, usize),
            sim: Vec<f32>,
        }
        impl DataProvider for WithTpl {
            type Label = u8;
            fn sample_rate(&self) -> f32 {
                self.inner.sample_rate()
            }
            fn n_channels(&self) -> u32 {
                self.inner.n_channels()
            }
            fn n_samples(&self) -> SampleIndex {
                self.inner.n_samples()
            }
            fn n_clusters(&self) -> u32 {
                self.inner.n_clusters()
            }
            fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
                self.inner.spike_times(c)
            }
            fn trace(&self, s: SampleIndex, l: u32) -> sorrel_io::TraceSlice<'_> {
                self.inner.trace(s, l)
            }
            fn initial_labels(&self) -> Vec<u8> {
                self.inner.initial_labels()
            }
        }
        impl HasTemplateWaveforms for WithTpl {
            fn template_waveforms(&self) -> &[f32] {
                &self.templates
            }
            fn template_shape(&self) -> (usize, usize, usize) {
                self.shape
            }
            fn similar_templates(&self) -> &[f32] {
                &self.sim
            }
        }

        let prov = WithTpl {
            inner: StubProvider {
                spikes: vec![vec![SampleIndex(10)], vec![SampleIndex(20)]],
            },
            // 2 templates × 3 samples × 2 channels = 12 floats.
            templates: (0..12).map(|i| i as f32).collect(),
            shape: (2, 3, 2),
            sim: vec![1.0, 0.5, 0.5, 1.0],
        };

        let mut ci = ClusterIndex::from_provider(&prov);
        ci.seed_template_waveforms(&prov);
        assert_eq!(ci.template_shape(), (2, 3, 2));
        assert_eq!(ci.template_waveforms().len(), 12);
        assert_eq!(ci.similar_templates(), &[1.0, 0.5, 0.5, 1.0]);

        let t0 = ci.template_waveform(0).unwrap();
        assert_eq!(t0.len(), 6);
        assert_eq!(t0[0], 0.0);
        assert_eq!(t0[5], 5.0);
        let t1 = ci.template_waveform(1).unwrap();
        assert_eq!(t1[0], 6.0);
        assert!(ci.template_waveform(99).is_none());
    }

    /// Stress: 1000 clusters × 1000 spikes each. Constructing the index,
    /// merging half into one bucket, and undoing should all complete in
    /// reasonable time. `#[ignore]` so it doesn't run by default.
    #[test]
    #[ignore]
    fn stress_thousand_clusters_thousand_spikes_each() {
        let n_clusters = 1000usize;
        let n_per = 1000usize;
        let spikes: Vec<Vec<SampleIndex>> = (0..n_clusters)
            .map(|c| {
                (0..n_per)
                    .map(|i| SampleIndex((c * n_per + i) as u64))
                    .collect()
            })
            .collect();
        let prov = StubProvider { spikes };
        let mut ci = ClusterIndex::from_provider(&prov);
        assert_eq!(ci.n_spikes(), n_clusters * n_per);

        let half: Vec<ClusterId> = (0..n_clusters as u32 / 2).map(ClusterId).collect();
        let rec = ci.merge(&half, ClusterId(n_clusters as u32 - 1));
        let target_count = ci.spike_times(ClusterId(n_clusters as u32 - 1)).len();
        assert_eq!(target_count, (half.len() + 1) * n_per);

        ci.unmerge(rec);
        assert_eq!(ci.n_spikes(), n_clusters * n_per);
    }
}

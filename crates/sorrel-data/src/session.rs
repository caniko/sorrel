use crate::cluster_index::{ClusterIndex, MergeRecord, SplitRecord};
use crate::command::{CurationCommand, PhyLabelOp};
use crate::journal::SqliteJournal;
use anyhow::{bail, Result};
use sorrel_io::{
    ClusterId, DataProvider, HasAmplitudes, HasPcFeatures, HasSpikeTemplates,
    HasTemplateWaveforms, SampleIndex,
};

/// Internal inverse-operation record. Lives only in memory, never journaled —
/// the journal stores only the user-visible `CurationCommand`. The inverse
/// captures the *prior* state of any cell that the forward op overwrote.
#[derive(Debug)]
enum InverseOp<L: Copy> {
    /// No-op (note in V1; emitted for unrecognised stub variants too).
    None,
    Relabel {
        cluster: ClusterId,
        prev: L,
    },
    Merge {
        /// Pre-merge label for each source cluster, captured in source order.
        labels: Vec<(ClusterId, L)>,
        /// Reversible record of the spike-table rewrite.
        record: MergeRecord,
    },
    Split {
        record: SplitRecord,
    },
    /// Inverse of a [`CurationCommand::Batch`] — stores per-child inverses
    /// in the order they were *applied*, so undo runs them in reverse and
    /// every step rolls back exactly the state it overwrote. Children may
    /// not themselves be `Batch` (enforced by [`CurationCommand::batch`]).
    Batch {
        inverses: Vec<InverseOp<L>>,
    },
}

#[derive(Debug)]
struct HistoryEntry<L: Copy> {
    cmd: CurationCommand,
    inverse: InverseOp<L>,
}

/// The core, generic curation session. `Session<P>` is monomorphised once per
/// backend type so calls into `provider` are inlined and devirtualised.
///
/// `provider` is the immutable raw-data layer; `cluster_index` is the mutable
/// curation surface (spike→cluster reassignment lives here, not in the
/// provider). All view code reads `Session::spike_times` / `spike_amplitudes`
/// / `spike_templates` so it sees the live, post-curation state.
pub struct Session<P: DataProvider> {
    pub provider: P,
    /// Indexed by `ClusterId`. `P::Label` is `Copy + 'static`, so this is a
    /// flat, cache-friendly buffer (1 byte per cluster for `PhyLabel`).
    labels: Vec<P::Label>,
    cluster_index: ClusterIndex,
    journal: SqliteJournal,
    history: Vec<HistoryEntry<P::Label>>,
    redo: Vec<HistoryEntry<P::Label>>,
}

impl<P: DataProvider> Session<P> {
    pub fn new(provider: P, journal: SqliteJournal) -> Self {
        let labels = provider.initial_labels();
        let cluster_index = ClusterIndex::from_provider(&provider);
        Self {
            provider,
            labels,
            cluster_index,
            journal,
            history: Vec::new(),
            redo: Vec::new(),
        }
    }

    /// Pull per-spike amplitudes from the provider into the cluster index.
    /// Backends opt in via [`HasAmplitudes`].
    pub fn seed_amplitudes(&mut self)
    where
        P: HasAmplitudes,
    {
        self.cluster_index.seed_amplitudes(&self.provider);
    }

    /// Pull per-spike template ids from the provider.
    pub fn seed_templates(&mut self)
    where
        P: HasSpikeTemplates,
    {
        self.cluster_index.seed_templates(&self.provider);
    }

    /// Pull PC features (and the per-spike NPY-row mapping) from the provider.
    pub fn seed_pc_features(&mut self)
    where
        P: HasPcFeatures,
    {
        self.cluster_index.seed_pc_features(&self.provider);
    }

    /// Pull template waveforms + similarity matrix from the provider.
    pub fn seed_template_waveforms(&mut self)
    where
        P: HasTemplateWaveforms,
    {
        self.cluster_index.seed_template_waveforms(&self.provider);
    }

    /// True when template waveforms have been seeded.
    #[inline]
    pub fn has_template_waveforms(&self) -> bool {
        self.cluster_index.template_shape() != (0, 0, 0)
    }

    /// Single template's waveform (`n_samples × n_channels` flat).
    pub fn template_waveform(&self, template_id: u32) -> Option<&[f32]> {
        self.cluster_index.template_waveform(template_id)
    }

    /// `(n_templates, n_samples, n_channels)` shape.
    pub fn template_shape(&self) -> (usize, usize, usize) {
        self.cluster_index.template_shape()
    }

    /// Pairwise template similarity matrix (`n_templates × n_templates`),
    /// flat row-major. Empty slice when not seeded.
    pub fn similar_templates(&self) -> &[f32] {
        self.cluster_index.similar_templates()
    }

    /// True when PC features have been seeded.
    #[inline]
    pub fn has_pc_features(&self) -> bool {
        self.cluster_index.pc_shape() != (0, 0)
    }

    /// Read-only access to the global per-spike cluster assignment. Useful
    /// for views that need to walk every spike (the FeatureView lasso).
    #[inline]
    pub fn spike_clusters_global(&self) -> &[ClusterId] {
        self.cluster_index.spike_clusters()
    }

    /// PC feature slice for the spike at `global_idx`.
    pub fn pc_feature_for(&self, global_idx: u32) -> Option<&[f32]> {
        self.cluster_index.pc_feature_for(global_idx)
    }

    /// `(n_pcs, n_channels_per_template)` shape of the PC feature buffer.
    pub fn pc_shape(&self) -> (usize, usize) {
        self.cluster_index.pc_shape()
    }

    #[inline]
    pub fn labels(&self) -> &[P::Label] {
        &self.labels
    }

    #[inline]
    pub fn label(&self, cluster: ClusterId) -> Option<P::Label> {
        self.labels.get(cluster as usize).copied()
    }

    /// Number of clusters currently addressable. Grows when a split allocates
    /// a fresh id; never shrinks (use the `is_empty` helper to filter).
    #[inline]
    pub fn n_clusters(&self) -> u32 {
        self.cluster_index.n_clusters()
    }

    #[inline]
    pub fn spike_times(&self, cluster: ClusterId) -> &[SampleIndex] {
        self.cluster_index.spike_times(cluster)
    }

    #[inline]
    pub fn spike_amplitudes(&self, cluster: ClusterId) -> &[f32] {
        self.cluster_index.spike_amplitudes(cluster)
    }

    #[inline]
    pub fn spike_templates(&self, cluster: ClusterId) -> &[u32] {
        self.cluster_index.spike_templates(cluster)
    }

    /// Read-only access to the underlying `ClusterIndex` (used by the
    /// save-to-phy writer).
    #[inline]
    pub fn cluster_index(&self) -> &ClusterIndex {
        &self.cluster_index
    }

    /// Number of forward operations on the undo stack.
    #[inline]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Number of operations that can be re-applied via [`CurationCommand::Redo`].
    #[inline]
    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    /// Iterator over the user-visible forward commands currently on the
    /// undo stack, oldest first.
    pub fn history_commands(&self) -> impl Iterator<Item = &CurationCommand> {
        self.history.iter().map(|e| &e.cmd)
    }

    /// Replay an existing journal on top of an already-loaded session,
    /// reconstructing both the in-memory state and the undo/redo stacks.
    /// Bad records (e.g. an `Undo` with an empty history) make replay fail —
    /// the journal should never contain such sequences.
    pub fn replay_journal(&mut self) -> Result<()>
    where
        P: ApplyPhyLabel,
    {
        let ops = self.journal.replay()?;
        for op in ops {
            self.apply_record(op)?;
        }
        Ok(())
    }

    /// Durably journal then apply. The fsync happens *before* the in-memory
    /// state changes, satisfying the crash-safety contract.
    ///
    /// Forward ops clear the redo stack; `Undo`/`Redo` records traverse it.
    pub fn dispatch(&mut self, cmd: CurationCommand) -> Result<()>
    where
        P: ApplyPhyLabel,
    {
        // For Undo/Redo we must verify the stack isn't empty *before* writing
        // to the journal, otherwise a no-op would still create a permanent
        // (and mis-leading) record.
        match &cmd {
            CurationCommand::Undo if self.history.is_empty() => return Ok(()),
            CurationCommand::Redo if self.redo.is_empty() => return Ok(()),
            _ => {}
        }
        self.journal.append(&cmd)?;
        self.apply_record(cmd)
    }

    /// Apply a single journal record (used both by `dispatch` after the
    /// journal append and by `replay_journal`). Maintains the undo/redo
    /// stacks; never touches the journal itself.
    fn apply_record(&mut self, cmd: CurationCommand) -> Result<()>
    where
        P: ApplyPhyLabel,
    {
        match cmd {
            CurationCommand::Undo => {
                let Some(entry) = self.history.pop() else {
                    bail!("journal contains Undo with empty history");
                };
                self.apply_inverse(entry.inverse);
                self.redo.push(HistoryEntry {
                    cmd: entry.cmd,
                    inverse: InverseOp::None,
                });
            }
            CurationCommand::Redo => {
                let Some(entry) = self.redo.pop() else {
                    bail!("journal contains Redo with empty redo stack");
                };
                let inverse = self.apply_forward(&entry.cmd);
                self.history.push(HistoryEntry {
                    cmd: entry.cmd,
                    inverse,
                });
            }
            forward => {
                let inverse = self.apply_forward(&forward);
                self.history.push(HistoryEntry {
                    cmd: forward,
                    inverse,
                });
                self.redo.clear();
            }
        }
        Ok(())
    }

    /// Apply the forward op AND return its inverse. Capturing the inverse
    /// inline (rather than as a separate pass) means we read state once
    /// before the mutation and once after — which matters for merge/split
    /// where the cluster_index records the affected snapshots.
    fn apply_forward(&mut self, cmd: &CurationCommand) -> InverseOp<P::Label>
    where
        P: ApplyPhyLabel,
    {
        match cmd {
            CurationCommand::Relabel { cluster, op } => {
                let prev = self.label(*cluster).unwrap_or_default();
                if let Some(slot) = self.labels.get_mut(*cluster as usize) {
                    *slot = P::label_from_op(*op);
                }
                InverseOp::Relabel {
                    cluster: *cluster,
                    prev,
                }
            }
            CurationCommand::Merge { sources, target } => {
                // Capture pre-merge labels so undo can restore them.
                let labels: Vec<(ClusterId, P::Label)> = sources
                    .iter()
                    .map(|&s| (s, self.label(s).unwrap_or_default()))
                    .collect();
                let record = self.cluster_index.merge(sources, *target);
                InverseOp::Merge { labels, record }
            }
            CurationCommand::Split {
                cluster,
                spike_idx,
                new_cluster: _,
            } => {
                // We auto-allocate a fresh cluster id rather than honouring
                // the requested `new_cluster` field — the on-disk schema
                // captures user intent ("split this cluster"), but the
                // physical id is determined by the index that performs the
                // mutation, which guarantees no collision after replay.
                let record = self.cluster_index.split(*cluster, spike_idx);
                InverseOp::Split { record }
            }
            CurationCommand::Note { .. } => InverseOp::None,
            CurationCommand::Batch { children } => {
                let mut inverses = Vec::with_capacity(children.len());
                for child in children {
                    debug_assert!(
                        !matches!(
                            child,
                            CurationCommand::Undo
                                | CurationCommand::Redo
                                | CurationCommand::Batch { .. }
                        ),
                        "Batch children must be forward, non-nested"
                    );
                    inverses.push(self.apply_forward(child));
                }
                InverseOp::Batch { inverses }
            }
            CurationCommand::Undo | CurationCommand::Redo => {
                debug_assert!(false, "Undo/Redo handled by caller, not apply_forward");
                InverseOp::None
            }
        }
    }

    fn apply_inverse(&mut self, inv: InverseOp<P::Label>) {
        match inv {
            InverseOp::None => {}
            InverseOp::Relabel { cluster, prev } => {
                if let Some(slot) = self.labels.get_mut(cluster as usize) {
                    *slot = prev;
                }
            }
            InverseOp::Merge { labels, record } => {
                self.cluster_index.unmerge(record);
                for (c, l) in labels {
                    if let Some(slot) = self.labels.get_mut(c as usize) {
                        *slot = l;
                    }
                }
            }
            InverseOp::Split { record } => {
                self.cluster_index.unsplit(record);
            }
            InverseOp::Batch { inverses } => {
                // Reverse order: undo the last-applied child first so each
                // step rolls back exactly the state it had overwritten.
                for inv in inverses.into_iter().rev() {
                    self.apply_inverse(inv);
                }
            }
        }
    }
}

/// Compile-time assertion: `Session<P>` is `Send + Sync` whenever the
/// underlying provider is. The rayon-parallel suggester / quality loops
/// borrow `&Session` across worker threads, so this needs to stay true.
/// Regressions (e.g. adding a non-Send field to Session) will surface as
/// a compile error here instead of a runtime failure in the UI.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Session<sorrel_io::KilosortProvider>>();
};

/// Backends that map `PhyLabelOp` -> their associated `Label`.
pub trait ApplyPhyLabel: DataProvider {
    fn label_from_op(op: PhyLabelOp) -> Self::Label;
}

impl ApplyPhyLabel for sorrel_io::KilosortProvider {
    #[inline]
    fn label_from_op(op: PhyLabelOp) -> Self::Label {
        use sorrel_io::kilosort::PhyLabel;
        match op {
            PhyLabelOp::SetUnsorted => PhyLabel::Unsorted,
            PhyLabelOp::SetGood => PhyLabel::Good,
            PhyLabelOp::SetMua => PhyLabel::Mua,
            PhyLabelOp::SetNoise => PhyLabel::Noise,
        }
    }
}

impl ApplyPhyLabel for sorrel_io::SortingAnalyzerProvider {
    #[inline]
    fn label_from_op(op: PhyLabelOp) -> Self::Label {
        use sorrel_io::kilosort::PhyLabel;
        match op {
            PhyLabelOp::SetUnsorted => PhyLabel::Unsorted,
            PhyLabelOp::SetGood => PhyLabel::Good,
            PhyLabelOp::SetMua => PhyLabel::Mua,
            PhyLabelOp::SetNoise => PhyLabel::Noise,
        }
    }
}

#[cfg(feature = "hdf5")]
impl ApplyPhyLabel for sorrel_io::NwbProvider {
    #[inline]
    fn label_from_op(op: PhyLabelOp) -> Self::Label {
        use sorrel_io::kilosort::PhyLabel;
        match op {
            PhyLabelOp::SetUnsorted => PhyLabel::Unsorted,
            PhyLabelOp::SetGood => PhyLabel::Good,
            PhyLabelOp::SetMua => PhyLabel::Mua,
            PhyLabelOp::SetNoise => PhyLabel::Noise,
        }
    }
}

#[cfg(feature = "hdf5")]
impl ApplyPhyLabel for sorrel_io::Ks4RezProvider {
    #[inline]
    fn label_from_op(op: PhyLabelOp) -> Self::Label {
        use sorrel_io::kilosort::PhyLabel;
        match op {
            PhyLabelOp::SetUnsorted => PhyLabel::Unsorted,
            PhyLabelOp::SetGood => PhyLabel::Good,
            PhyLabelOp::SetMua => PhyLabel::Mua,
            PhyLabelOp::SetNoise => PhyLabel::Noise,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sorrel_io::{ClusterId, DataProvider, SampleIndex, TraceSamples, TraceSlice};

    /// Tiny in-memory provider for testing the generic session machinery.
    struct MockProvider {
        spikes: Vec<Vec<SampleIndex>>,
        labels: Vec<MockLabel>,
        trace: Vec<i16>,
        n_channels: u32,
    }

    #[repr(u8)]
    #[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
    enum MockLabel {
        #[default]
        Unsorted = 0,
        Good = 1,
        Mua = 2,
        Noise = 3,
    }

    impl DataProvider for MockProvider {
        type Label = MockLabel;

        fn sample_rate(&self) -> f32 { 1000.0 }
        fn n_channels(&self) -> u32 { self.n_channels }
        fn n_samples(&self) -> SampleIndex {
            (self.trace.len() / self.n_channels.max(1) as usize) as u64
        }
        fn n_clusters(&self) -> u32 { self.spikes.len() as u32 }
        fn spike_times(&self, cluster: ClusterId) -> &[SampleIndex] {
            self.spikes.get(cluster as usize).map(Vec::as_slice).unwrap_or(&[])
        }
        fn trace(&self, start: SampleIndex, len: u32) -> TraceSlice<'_> {
            let nc = self.n_channels as usize;
            let s = (start as usize * nc).min(self.trace.len());
            let e = (s + len as usize * nc).min(self.trace.len());
            TraceSlice {
                start,
                n_channels: self.n_channels,
                samples: TraceSamples::I16(&self.trace[s..e]),
            }
        }
        fn initial_labels(&self) -> Vec<Self::Label> { self.labels.clone() }
    }

    impl ApplyPhyLabel for MockProvider {
        fn label_from_op(op: PhyLabelOp) -> Self::Label {
            match op {
                PhyLabelOp::SetUnsorted => MockLabel::Unsorted,
                PhyLabelOp::SetGood => MockLabel::Good,
                PhyLabelOp::SetMua => MockLabel::Mua,
                PhyLabelOp::SetNoise => MockLabel::Noise,
            }
        }
    }

    fn fresh_session() -> (Session<MockProvider>, tempfile::TempDir) {
        let provider = MockProvider {
            spikes: vec![vec![10, 30, 150], vec![20, 200], vec![100]],
            labels: vec![MockLabel::Unsorted; 3],
            trace: vec![0i16; 4 * 16],
            n_channels: 4,
        };
        let dir = tempfile::tempdir().unwrap();
        let journal = SqliteJournal::open(&dir.path().join("j.sqlite")).unwrap();
        let session = Session::new(provider, journal);
        (session, dir)
    }

    #[test]
    fn new_session_seeds_labels_and_buckets_from_provider() {
        let (s, _d) = fresh_session();
        assert_eq!(s.labels().len(), 3);
        assert_eq!(s.label(0), Some(MockLabel::Unsorted));
        assert_eq!(s.label(99), None);
        assert_eq!(s.history_len(), 0);
        assert_eq!(s.redo_len(), 0);
        // Bucketed access flows through the ClusterIndex now.
        assert_eq!(s.spike_times(0), &[10, 30, 150]);
        assert_eq!(s.spike_times(2), &[100]);
    }

    #[test]
    fn dispatch_relabel_updates_label_and_history() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Relabel { cluster: 1, op: PhyLabelOp::SetGood }).unwrap();
        assert_eq!(s.label(1), Some(MockLabel::Good));
        assert_eq!(s.history_len(), 1);
    }

    #[test]
    fn dispatch_merge_reassigns_spikes_to_target() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Merge { sources: vec![0, 1], target: 2 }).unwrap();
        assert!(s.spike_times(0).is_empty());
        assert!(s.spike_times(1).is_empty());
        assert_eq!(s.spike_times(2), &[10, 20, 30, 100, 150, 200]);
        // Labels are NOT changed by merge anymore — phy keeps source labels
        // alongside the empty bucket and lets the user filter "no spikes".
        assert_eq!(s.label(0), Some(MockLabel::Unsorted));
    }

    #[test]
    fn undo_merge_restores_buckets_and_source_labels() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Relabel { cluster: 0, op: PhyLabelOp::SetGood }).unwrap();
        s.dispatch(CurationCommand::Merge { sources: vec![0], target: 2 }).unwrap();
        assert!(s.spike_times(0).is_empty());

        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.spike_times(0), &[10, 30, 150]);
        assert_eq!(s.label(0), Some(MockLabel::Good));
    }

    #[test]
    fn batch_applies_children_in_order_and_undoes_atomically() {
        let (mut s, _d) = fresh_session();
        let batch = CurationCommand::batch(vec![
            CurationCommand::Relabel { cluster: 0, op: PhyLabelOp::SetGood },
            CurationCommand::Relabel { cluster: 1, op: PhyLabelOp::SetMua },
            CurationCommand::Merge { sources: vec![0], target: 1 },
        ])
        .unwrap();
        s.dispatch(batch).unwrap();
        // All three children took effect.
        assert_eq!(s.label(0), Some(MockLabel::Good));
        assert_eq!(s.label(1), Some(MockLabel::Mua));
        assert!(s.spike_times(0).is_empty());
        assert_eq!(s.history_len(), 1, "batch counts as one history entry");

        // One undo reverts the entire batch.
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.label(0), Some(MockLabel::Unsorted));
        assert_eq!(s.label(1), Some(MockLabel::Unsorted));
        assert_eq!(s.spike_times(0), &[10, 30, 150]);
        assert_eq!(s.history_len(), 0);
        assert_eq!(s.redo_len(), 1);

        // Redo re-applies the entire batch.
        s.dispatch(CurationCommand::Redo).unwrap();
        assert_eq!(s.label(0), Some(MockLabel::Good));
        assert_eq!(s.label(1), Some(MockLabel::Mua));
        assert!(s.spike_times(0).is_empty());
    }

    #[test]
    fn empty_batch_is_a_noop_with_one_history_entry() {
        let (mut s, _d) = fresh_session();
        let batch = CurationCommand::batch(vec![]).unwrap();
        s.dispatch(batch).unwrap();
        assert_eq!(s.history_len(), 1);
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.history_len(), 0);
    }

    #[test]
    fn dispatch_split_moves_spikes_to_a_fresh_cluster() {
        let (mut s, _d) = fresh_session();
        let n_before = s.n_clusters();
        s.dispatch(CurationCommand::Split {
            cluster: 0,
            spike_idx: vec![1],
            new_cluster: 99, // ignored: id auto-allocated
        })
        .unwrap();
        assert_eq!(s.n_clusters(), n_before + 1);
        assert_eq!(s.spike_times(0), &[10, 150]);
        assert_eq!(s.spike_times(n_before), &[30]);
    }

    #[test]
    fn undo_split_restores_state_and_frees_cluster_id() {
        let (mut s, _d) = fresh_session();
        let n_before = s.n_clusters();
        let pre = s.spike_times(0).to_vec();
        s.dispatch(CurationCommand::Split {
            cluster: 0,
            spike_idx: vec![0, 2],
            new_cluster: 0,
        })
        .unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.n_clusters(), n_before);
        assert_eq!(s.spike_times(0), &pre[..]);
    }

    #[test]
    fn dispatch_note_is_a_noop() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Note { cluster: 0, text: "x".into() }).unwrap();
        assert_eq!(s.label(0), Some(MockLabel::Unsorted));
        assert_eq!(s.history_len(), 1);
    }

    #[test]
    fn undo_restores_previous_label() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Relabel { cluster: 1, op: PhyLabelOp::SetGood }).unwrap();
        assert_eq!(s.label(1), Some(MockLabel::Good));

        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.label(1), Some(MockLabel::Unsorted));
        assert_eq!(s.history_len(), 0);
        assert_eq!(s.redo_len(), 1);
    }

    #[test]
    fn redo_reapplies_after_undo() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Relabel { cluster: 1, op: PhyLabelOp::SetMua }).unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.label(1), Some(MockLabel::Unsorted));
        s.dispatch(CurationCommand::Redo).unwrap();
        assert_eq!(s.label(1), Some(MockLabel::Mua));
        assert_eq!(s.redo_len(), 0);
        assert_eq!(s.history_len(), 1);
    }

    #[test]
    fn redo_merge_reapplies_spike_reassignment() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Merge { sources: vec![0, 1], target: 2 }).unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.spike_times(0), &[10, 30, 150]);

        s.dispatch(CurationCommand::Redo).unwrap();
        assert!(s.spike_times(0).is_empty());
        assert_eq!(s.spike_times(2), &[10, 20, 30, 100, 150, 200]);
    }

    #[test]
    fn forward_op_after_undo_clears_redo() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Relabel { cluster: 0, op: PhyLabelOp::SetGood }).unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.redo_len(), 1);

        s.dispatch(CurationCommand::Relabel { cluster: 0, op: PhyLabelOp::SetMua }).unwrap();
        assert_eq!(s.redo_len(), 0);
        assert_eq!(s.label(0), Some(MockLabel::Mua));
    }

    #[test]
    fn undo_with_empty_history_is_a_noop() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.history_len(), 0);
        assert_eq!(s.redo_len(), 0);
    }

    #[test]
    fn redo_with_empty_redo_stack_is_a_noop() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Relabel { cluster: 0, op: PhyLabelOp::SetGood }).unwrap();
        s.dispatch(CurationCommand::Redo).unwrap();
        assert_eq!(s.label(0), Some(MockLabel::Good));
        assert_eq!(s.redo_len(), 0);
    }

    #[test]
    fn multi_step_undo_walks_the_stack_in_reverse() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Relabel { cluster: 0, op: PhyLabelOp::SetGood }).unwrap();
        s.dispatch(CurationCommand::Relabel { cluster: 0, op: PhyLabelOp::SetMua }).unwrap();
        s.dispatch(CurationCommand::Relabel { cluster: 0, op: PhyLabelOp::SetNoise }).unwrap();
        assert_eq!(s.label(0), Some(MockLabel::Noise));

        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.label(0), Some(MockLabel::Mua));
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.label(0), Some(MockLabel::Good));
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.label(0), Some(MockLabel::Unsorted));
        assert_eq!(s.history_len(), 0);
    }

    #[test]
    fn replay_journal_reconstructs_state_and_redo_stack() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.sqlite");
        {
            let mut j = SqliteJournal::open(&path).unwrap();
            j.append(&CurationCommand::Relabel { cluster: 0, op: PhyLabelOp::SetMua }).unwrap();
            j.append(&CurationCommand::Relabel { cluster: 1, op: PhyLabelOp::SetNoise }).unwrap();
            j.append(&CurationCommand::Undo).unwrap();
        }

        let provider = MockProvider {
            spikes: vec![vec![1], vec![2], vec![3]],
            labels: vec![MockLabel::Unsorted; 3],
            trace: vec![],
            n_channels: 1,
        };
        let journal = SqliteJournal::open(&path).unwrap();
        let mut session = Session::new(provider, journal);
        session.replay_journal().unwrap();

        assert_eq!(session.label(0), Some(MockLabel::Mua));
        assert_eq!(session.label(1), Some(MockLabel::Unsorted));
        assert_eq!(session.label(2), Some(MockLabel::Unsorted));
        assert_eq!(session.history_len(), 1);
        assert_eq!(session.redo_len(), 1);
    }

    #[test]
    fn replay_journal_with_orphan_undo_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.sqlite");
        {
            let mut j = SqliteJournal::open(&path).unwrap();
            j.append(&CurationCommand::Undo).unwrap();
        }

        let provider = MockProvider {
            spikes: vec![],
            labels: vec![],
            trace: vec![],
            n_channels: 1,
        };
        let journal = SqliteJournal::open(&path).unwrap();
        let mut session = Session::new(provider, journal);
        assert!(session.replay_journal().is_err());
    }

    #[test]
    fn replay_journal_reconstructs_merged_buckets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.sqlite");
        {
            let mut j = SqliteJournal::open(&path).unwrap();
            j.append(&CurationCommand::Merge { sources: vec![0, 1], target: 2 }).unwrap();
        }

        let provider = MockProvider {
            spikes: vec![vec![10, 30, 150], vec![20, 200], vec![100]],
            labels: vec![MockLabel::Unsorted; 3],
            trace: vec![],
            n_channels: 1,
        };
        let journal = SqliteJournal::open(&path).unwrap();
        let mut session = Session::new(provider, journal);
        session.replay_journal().unwrap();

        assert_eq!(session.spike_times(2), &[10, 20, 30, 100, 150, 200]);
        assert!(session.spike_times(0).is_empty());
    }
}

use crate::cache::amplitudes::{
    CachedAmplitudes, ALGO_VERSION as AMPLITUDES_ALGO_VERSION, KIND as AMPLITUDES_KIND,
};
use crate::cache::{isolation, pc_subspace};
use crate::cluster_index::{AddRecord, ClusterIndex, MergeRecord, SplitRecord};
use crate::command::{CurationCommand, Method, PhyLabelOp, UnitOrigin};
use crate::feature_subspace::{DEFAULT_CHANNEL_IDX, DEFAULT_D, DEFAULT_MAX_BACKGROUND};
use crate::journal::Journal;
use crate::quality_ext::DEFAULT_K_NN;
use anyhow::{bail, Result};
use sorrel_cache::{CacheKey, CacheStore, Fingerprint};
use sorrel_io::{
    ClusterId, DataProvider, HasAmplitudes, HasPcFeatures, HasSpikeTemplates, HasTemplateWaveforms,
    SampleIndex,
};
use std::collections::VecDeque;
use std::path::Path;

const MAX_CACHE_INVALIDATIONS: usize = 1024;

/// Tags the session sets on an Auto unit that absorbed a merge / lost spikes
/// to a split. A split child does not inherit them.
pub(crate) const TAG_TUNED: &str = "tuned";
pub(crate) const TAG_SPLIT: &str = "split";

static AUTO: UnitOrigin = UnitOrigin::Auto;

/// Set `v[i]` (grown on demand with defaults), returning the previous value.
fn set_slot<T: Default>(v: &mut Vec<T>, c: ClusterId, val: T) -> T {
    if v.len() <= c.idx() {
        v.resize_with(c.idx() + 1, T::default);
    }
    std::mem::replace(&mut v[c.idx()], val)
}

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
        /// Target and its tag set before the sources' tags were unioned in.
        target_tags: (ClusterId, Vec<String>),
        /// The target's parents before the sources were appended.
        target_parents: Vec<ClusterId>,
        /// The target's note before the sources' notes were appended.
        target_note: String,
        /// Reversible record of the spike-table rewrite.
        record: MergeRecord,
    },
    Split {
        record: SplitRecord,
        /// The split parent's tags before it gained `split`.
        parent_tags: Vec<String>,
    },
    Note {
        cluster: ClusterId,
        prev: String,
    },
    /// Inverse of a [`CurationCommand::Batch`] — stores per-child inverses
    /// in the order they were *applied*, so undo runs them in reverse and
    /// every step rolls back exactly the state it overwrote. Children may
    /// not themselves be `Batch` (enforced by [`CurationCommand::batch`]).
    Batch {
        inverses: Vec<InverseOp<L>>,
    },
    Tag {
        cluster: ClusterId,
        prev: Vec<String>,
    },
    AddSpikes {
        record: AddRecord,
        /// Tags of the Auto clusters that gained `tuned`, before the add.
        prev_tags: Vec<(ClusterId, Vec<String>)>,
    },
}

#[derive(Debug)]
struct HistoryEntry<L: Copy> {
    cmd: CurationCommand,
    inverse: InverseOp<L>,
}

#[derive(Clone, Debug)]
struct CacheClusterIdentity {
    cluster: ClusterId,
    spike_count: usize,
    cluster_spike_fingerprint: [u8; 32],
}

#[derive(Clone, Debug)]
struct CacheInvalidation {
    prev_head: u64,
    new_head: u64,
    /// Total spike count before the edit (part of the pc-subspace key).
    n_spikes: u64,
    clusters: Vec<CacheClusterIdentity>,
}

/// The core, generic curation session. `Session<P>` is monomorphised once per
/// backend type so calls into `provider` are inlined and devirtualised.
///
/// `provider` is the immutable raw-data layer; `cluster_index` is the mutable
/// curation surface (spike→cluster reassignment lives here, not in the
/// provider). All view code reads `Session::spike_times` / `spike_amplitudes`
/// / `spike_templates` so it sees the live, post-curation state.
/// Observer installed by [`Session::set_record_sink`].
pub type RecordSink = Box<dyn FnMut(&CurationCommand) + Send + Sync>;

pub struct Session<P: DataProvider> {
    pub provider: P,
    /// Indexed by `ClusterId`. `P::Label` is `Copy + 'static`, so this is a
    /// flat, cache-friendly buffer (1 byte per cluster for `PhyLabel`).
    labels: Vec<P::Label>,
    /// Provenance/review tags per cluster, indexed by `ClusterId`; grown on
    /// demand, so a missing slot means no tags.
    tags: Vec<Vec<String>>,
    /// Origin, direct parents and note per cluster, indexed like `tags`; a
    /// missing slot means `Auto`, no parents, no note.
    origins: Vec<UnitOrigin>,
    parents: Vec<Vec<ClusterId>>,
    notes: Vec<String>,
    /// Sorter name rendered in `Auto[<sorter>]`; `"unknown"` until set.
    sorter: String,
    cluster_index: ClusterIndex,
    journal: Journal,
    history: Vec<HistoryEntry<P::Label>>,
    redo: Vec<HistoryEntry<P::Label>>,
    cache_invalidations: VecDeque<CacheInvalidation>,
    /// Observer called once per journaled record, right after the append is
    /// durable and before the in-memory state moves. Used to stream records
    /// to a remote host; `None` for ordinary local sessions.
    record_sink: Option<RecordSink>,
}

impl<P: DataProvider> Session<P> {
    pub fn new(provider: P, journal: Journal) -> Self {
        let labels = provider.initial_labels();
        let tags = provider.initial_tags();
        let mut origins = Vec::new();
        let mut parents = Vec::new();
        let mut notes = Vec::new();
        for (i, a) in provider.initial_annotations().into_iter().enumerate() {
            let c = ClusterId(i as u32);
            // An unparseable origin degrades to Auto rather than refusing to open.
            let (origin, _) = UnitOrigin::parse(&a.origin).unwrap_or((UnitOrigin::Auto, None));
            set_slot(&mut origins, c, origin);
            set_slot(&mut parents, c, a.parents);
            set_slot(&mut notes, c, a.note);
        }
        let sorter = provider.sorter().unwrap_or_else(|| "unknown".into());
        let cluster_index = ClusterIndex::from_provider(&provider);
        Self {
            provider,
            labels,
            tags,
            origins,
            parents,
            notes,
            sorter,
            cluster_index,
            journal,
            history: Vec::new(),
            redo: Vec::new(),
            cache_invalidations: VecDeque::new(),
            record_sink: None,
        }
    }

    /// Install the per-record observer (see [`Session::dispatch`]). Replay does
    /// not call it: only records this process appends are forwarded.
    pub fn set_record_sink(&mut self, sink: impl FnMut(&CurationCommand) + Send + Sync + 'static) {
        self.record_sink = Some(Box::new(sink));
    }

    /// Pull per-spike amplitudes from the provider into the cluster index.
    /// Backends opt in via [`HasAmplitudes`].
    pub fn seed_amplitudes(&mut self)
    where
        P: HasAmplitudes,
    {
        let key = self.amplitudes_cache_key();
        let Some(dataset_dir) = self.journal.path().parent() else {
            self.cluster_index.seed_amplitudes(&self.provider);
            return;
        };

        match CacheStore::open(dataset_dir) {
            Ok(store) => {
                match store.get::<CachedAmplitudes>(&key) {
                    Ok(Some(cached)) => {
                        log::debug!("cache hit {}/{}", AMPLITUDES_KIND, key.fingerprint.hex());
                        let buckets = cached
                            .per_cluster
                            .iter()
                            .map(|bucket| {
                                bucket
                                    .as_slice()
                                    .iter()
                                    .map(|amp| amp.to_native())
                                    .collect()
                            })
                            .collect();
                        self.cluster_index.seed_amplitudes_from_buckets(buckets);
                        return;
                    }
                    Ok(None) => {
                        log::debug!("cache miss {}/{}", AMPLITUDES_KIND, key.fingerprint.hex());
                    }
                    Err(err) => {
                        log::debug!(
                            "cache miss {}/{} after read error: {err}",
                            AMPLITUDES_KIND,
                            key.fingerprint.hex()
                        );
                    }
                }

                self.cluster_index.seed_amplitudes(&self.provider);
                let value = CachedAmplitudes {
                    per_cluster: (0..self.cluster_index.n_clusters())
                        .map(ClusterId)
                        .map(|cluster| self.cluster_index.spike_amplitudes(cluster).to_vec())
                        .collect(),
                    algo_version: AMPLITUDES_ALGO_VERSION,
                };
                if let Err(err) = store.put(&key, &value) {
                    log::debug!(
                        "failed to write cache {}/{}: {err}",
                        AMPLITUDES_KIND,
                        key.fingerprint.hex()
                    );
                }
            }
            Err(err) => {
                log::debug!(
                    "cache miss {}/{} after open error: {err}",
                    AMPLITUDES_KIND,
                    key.fingerprint.hex()
                );
                self.cluster_index.seed_amplitudes(&self.provider);
            }
        }
    }

    fn amplitudes_cache_key(&self) -> CacheKey {
        let fingerprint = Fingerprint::builder()
            .add_provider_identity(&self.provider.identity_bytes())
            .add_journal_head(self.journal.head())
            .add_algo_version(AMPLITUDES_ALGO_VERSION)
            .finish();
        CacheKey::new(AMPLITUDES_KIND, AMPLITUDES_ALGO_VERSION, fingerprint)
    }

    pub(crate) fn cache_dataset_dir(&self) -> Option<&Path> {
        self.journal.path().parent()
    }

    /// Monotonic version of the replay-visible curation state: changes on
    /// every journaled command (including undo/redo) and on reseal. Use it
    /// as the cache key for anything derived from curation state; the
    /// `(history_len, redo_len)` pair is *not* safe because undo followed by
    /// a different forward op lands on the same pair with different state.
    #[inline]
    pub fn version(&self) -> u64 {
        self.journal.head()
    }

    /// Where the curation journal lives on disk.
    pub fn journal_path(&self) -> &Path {
        self.journal.path()
    }

    /// The sealed journal: baseline, record count, and path.
    pub fn journal(&self) -> &Journal {
        &self.journal
    }

    /// Swap in a freshly sealed journal (see `save_and_reseal_journal`).
    /// The saved artifacts are now canonical, so the undo/redo stacks and
    /// pending cache invalidations are dropped with the old journal.
    pub fn reseal(&mut self, journal: Journal) {
        self.journal = journal;
        self.history.clear();
        self.redo.clear();
        self.cache_invalidations.clear();
    }

    /// Evict cache entries known to have been derived from pre-edit journal
    /// heads. This is intentionally called at save/quit boundaries, not after
    /// each command, so a running session keeps its warm in-memory state.
    pub fn invalidate_caches(
        &self,
        store: &CacheStore,
    ) -> std::result::Result<(), sorrel_cache::CacheError> {
        let provider_identity = self.provider.identity_bytes();
        for entry in &self.cache_invalidations {
            if entry.prev_head == entry.new_head {
                continue;
            }
            store.evict(&amplitudes_cache_key_for(
                &provider_identity,
                entry.prev_head,
            ))?;

            for cluster in &entry.clusters {
                if cluster.spike_count == 0 {
                    continue;
                }
                let subspace_key = pc_subspace::key(pc_subspace::KeyParts {
                    session_identity: &provider_identity,
                    n_spikes_total: entry.n_spikes,
                    cluster: cluster.cluster,
                    cluster_spike_count: cluster.spike_count,
                    cluster_spike_fingerprint: cluster.cluster_spike_fingerprint,
                    d_pcs: DEFAULT_D,
                    channel_idx: DEFAULT_CHANNEL_IDX,
                    max_background: DEFAULT_MAX_BACKGROUND,
                });
                store.evict(&subspace_key)?;
                store.evict(&isolation::key(&subspace_key, DEFAULT_K_NN))?;
            }
        }
        Ok(())
    }

    /// Open the dataset-local cache and apply pending invalidations. Missing
    /// cache directories are not errors: the cache is derivable.
    pub fn invalidate_dataset_caches(&self) -> std::result::Result<(), sorrel_cache::CacheError> {
        let Some(dataset_dir) = self.cache_dataset_dir() else {
            return Ok(());
        };
        let store = CacheStore::open(dataset_dir)?;
        self.invalidate_caches(&store)
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

    /// PC `pc` of the spike at `global_idx` on real channel `channel`;
    /// `None` when the spike's template has no feature on that channel.
    pub fn pc_feature(&self, global_idx: u32, channel: u32, pc: usize) -> Option<f32> {
        self.cluster_index.pc_feature(global_idx, channel, pc)
    }

    /// FeatureView channels for `selection` (phy `FeatureView._get_channel_ids`):
    /// the first cluster's PC channels ranked by template amplitude, with the
    /// channels every other selected cluster also has moved to the front.
    pub fn best_channels(&self, selection: &[ClusterId]) -> Vec<u32> {
        let Some((&first, rest)) = selection.split_first() else {
            return Vec::new();
        };
        let ranked = |c: ClusterId| -> Vec<u32> {
            let Some(t) = self.primary_template(c) else {
                return Vec::new();
            };
            let mut chans = self.cluster_index.pc_channels(t).to_vec();
            if let Some(w) = self.template_waveform(t) {
                let (_, n_samples, n_chans) = self.template_shape();
                let ptp = |ch: u32| {
                    let col = (0..n_samples).filter_map(|s| w.get(s * n_chans + ch as usize));
                    let (lo, hi) =
                        col.fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
                    (hi - lo).max(0.0)
                };
                chans.sort_by(|&a, &b| ptp(b).total_cmp(&ptp(a)));
            }
            chans
        };
        let mut best = ranked(first);
        let others: Vec<Vec<u32>> = rest.iter().map(|&c| ranked(c)).collect();
        // Stable sort: shared channels first, amplitude order kept within each part.
        best.sort_by_key(|ch| !others.iter().all(|o| o.contains(ch)));
        best
    }

    /// phy's cluster similarity (SimilarityView and the `space` wizard):
    /// for every non-empty cluster other than `reference` and `exclude`, the
    /// max of `similar_templates` over all template pairs of the two
    /// clusters, highest first (ties by id). A merged cluster owns the union
    /// of its sources' templates. Without the matrix it falls back to the
    /// cosine of the flattened template waveforms under the same max rule,
    /// and is empty when neither is loaded.
    pub fn similar_clusters(
        &self,
        reference: ClusterId,
        exclude: &[ClusterId],
    ) -> Vec<(ClusterId, f32)> {
        use rayon::prelude::*;
        let templates_of = |c: ClusterId| {
            let mut t = self.spike_templates(c).to_vec();
            t.sort_unstable();
            t.dedup();
            t
        };
        let mine = templates_of(reference);
        let (n_t, _, _) = self.template_shape();
        let sim = self.similar_templates();
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (norm(a) * norm(b)).max(f32::MIN_POSITIVE)
        };
        let have_matrix = n_t > 0 && sim.len() == n_t * n_t;
        let pair = |a: u32, b: u32| -> Option<f32> {
            if have_matrix {
                let (a, b) = (a as usize, b as usize);
                (a < n_t && b < n_t).then(|| sim[a * n_t + b])
            } else {
                // ponytail: recomputes both norms per pair; cache per-template
                // norms if clusters ever span hundreds of templates.
                Some(cosine(
                    self.template_waveform(a)?,
                    self.template_waveform(b)?,
                ))
            }
        };
        let score = |c: ClusterId| -> Option<f32> {
            let theirs = templates_of(c);
            mine.iter()
                .flat_map(|&a| theirs.iter().map(move |&b| (a, b)))
                .filter_map(|(a, b)| pair(a, b))
                .reduce(f32::max)
        };
        let mut out: Vec<(ClusterId, f32)> = (0..self.n_clusters())
            .into_par_iter()
            .map(ClusterId)
            .filter(|&c| c != reference && !exclude.contains(&c) && !self.spike_times(c).is_empty())
            .filter_map(|c| score(c).filter(|v| v.is_finite()).map(|v| (c, v)))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    #[inline]
    pub fn labels(&self) -> &[P::Label] {
        &self.labels
    }

    #[inline]
    pub fn label(&self, cluster: ClusterId) -> Option<P::Label> {
        self.labels.get(cluster.idx()).copied()
    }

    /// Provenance/review tags of `cluster` (empty when it has none).
    pub fn tags(&self, cluster: ClusterId) -> &[String] {
        self.tags.get(cluster.idx()).map_or(&[], Vec::as_slice)
    }

    /// Where `cluster` came from (`Auto` for anything the sorter produced).
    pub fn origin(&self, cluster: ClusterId) -> &UnitOrigin {
        self.origins.get(cluster.idx()).unwrap_or(&AUTO)
    }

    /// [`Session::origin`] as text: `Auto[<sorter>]` or `Sorrel[Algo:gmm]`.
    pub fn origin_label(&self, cluster: ClusterId) -> String {
        self.origin(cluster).label(&self.sorter)
    }

    /// Direct parents of `cluster` (the unit it was split from, or the units
    /// merged into it); walk transitively for the full lineage.
    pub fn parents(&self, cluster: ClusterId) -> &[ClusterId] {
        self.parents.get(cluster.idx()).map_or(&[], Vec::as_slice)
    }

    /// Free-text note of `cluster` (empty when none).
    pub fn note(&self, cluster: ClusterId) -> &str {
        self.notes.get(cluster.idx()).map_or("", String::as_str)
    }

    /// The sorter named in `Auto[<sorter>]`; `"unknown"` if nothing set it.
    pub fn sorter(&self) -> &str {
        &self.sorter
    }

    /// Set the sorter name explicitly (`--sorter`); overrides a saved pin.
    pub fn set_sorter(&mut self, name: impl Into<String>) {
        self.sorter = name.into();
    }

    /// Fill in a detected sorter name (`unknown` when `None`), unless the
    /// saved annotations pinned one: detection never overrides a pin.
    pub fn fill_detected_sorter(&mut self, detected: Option<String>) {
        if self.provider.sorter().is_none() {
            self.sorter = detected.unwrap_or_else(|| "unknown".into());
        }
    }

    /// `tags` plus `tuned` when `cluster` is an Auto unit (it was edited but
    /// kept its identity), sorted and deduplicated. Callers that change an
    /// Auto unit's spikes in place use it for the new tag set.
    fn tune_tags(&self, cluster: ClusterId, mut tags: Vec<String>) -> Vec<String> {
        if *self.origin(cluster) == UnitOrigin::Auto {
            tags.push(TAG_TUNED.to_string());
        }
        tags.sort_unstable();
        tags.dedup();
        tags
    }

    /// phy `split_init` for one cluster: the `Split`s that give each of its
    /// original templates (beyond the most common one, which stays) a
    /// cluster of its own. Indices are local to the bucket as it shrinks,
    /// so the commands apply in order, e.g. as one batch.
    pub fn split_by_template_commands(&self, cluster: ClusterId) -> Vec<CurationCommand> {
        let mut rest = self.spike_templates(cluster).to_vec();
        let Some(keep) = self.primary_template(cluster) else {
            return Vec::new();
        };
        let mut others: Vec<u32> = rest.iter().copied().filter(|&t| t != keep).collect();
        others.sort_unstable();
        others.dedup();
        others
            .into_iter()
            .map(|t| {
                let spike_idx: Vec<u32> = (0..rest.len() as u32)
                    .filter(|&i| rest[i as usize] == t)
                    .collect();
                rest.retain(|&x| x != t);
                CurationCommand::Split {
                    cluster,
                    spike_idx,
                    new_cluster: ClusterId(0),
                    by: Method::Algo("split_init".into()),
                }
            })
            .collect()
    }

    /// phy custom label `key` of `cluster` (`:l key value`), stored as a
    /// `key=value` tag so it journals, undoes and follows merges and
    /// splits like any tag. After a merge the target's value wins.
    pub fn custom_label(&self, cluster: ClusterId, key: &str) -> Option<&str> {
        self.tags(cluster)
            .iter()
            .find_map(|t| t.strip_prefix(key)?.strip_prefix('='))
    }

    /// Every custom label key in use, sorted.
    pub fn custom_label_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .tags
            .iter()
            .flatten()
            .filter_map(|t| t.split_once('=').map(|(k, _)| k.to_string()))
            .collect();
        keys.sort();
        keys.dedup();
        keys
    }

    /// The tag edit that sets custom label `key` to `value` on `cluster`
    /// (an empty value clears it). Keys are phy column names: letters,
    /// digits and `_`, and not one of phy's own label files.
    pub fn set_label_command(
        &self,
        cluster: ClusterId,
        key: &str,
        value: &str,
    ) -> Result<CurationCommand, String> {
        if key.is_empty()
            || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            || sorrel_io::kilosort::RESERVED_COLUMNS.contains(&key)
        {
            return Err(format!("{key:?} cannot be a label name"));
        }
        let prefix = format!("{key}=");
        let mut tags: Vec<String> = self
            .tags(cluster)
            .iter()
            .filter(|t| !t.starts_with(&prefix))
            .cloned()
            .collect();
        if !value.is_empty() {
            let tag = format!("{prefix}{value}");
            if !sorrel_io::kilosort::valid_tag(&tag) {
                return Err("label values cannot contain commas, tabs or newlines".into());
            }
            tags.push(tag);
        }
        Ok(CurationCommand::Tag { cluster, tags })
    }

    /// `cmd` as a human decision: preceded by retags that drop the touched
    /// clusters' `review:*` tags and add `src:human`, all in one undoable
    /// batch. Untagged clusters are left alone (sessions that never ran
    /// `sorrel auto` stay tag-free). Retags go first so a merge unions, and
    /// a split child inherits, the cleaned set.
    pub fn with_human_review(&self, cmd: CurationCommand) -> CurationCommand {
        let children: &[CurationCommand] = match &cmd {
            CurationCommand::Batch { children } => children,
            CurationCommand::Undo | CurationCommand::Redo => return cmd,
            other => std::slice::from_ref(other),
        };
        let mut clusters = Vec::new();
        for c in children {
            match c {
                CurationCommand::Relabel { cluster, .. }
                | CurationCommand::Split { cluster, .. } => clusters.push(*cluster),
                CurationCommand::Merge { sources, target } => {
                    clusters.push(*target);
                    clusters.extend_from_slice(sources);
                }
                _ => {}
            }
        }
        clusters.sort_unstable();
        clusters.dedup();
        let mut out: Vec<CurationCommand> = clusters
            .into_iter()
            .filter_map(|c| {
                let old = self.tags(c);
                if old.is_empty() {
                    return None;
                }
                let mut tags: Vec<String> = old
                    .iter()
                    .filter(|t| !t.starts_with("review:"))
                    .cloned()
                    .collect();
                if !tags.iter().any(|t| t == "src:human") {
                    tags.push("src:human".into());
                }
                (tags != old).then_some(CurationCommand::Tag { cluster: c, tags })
            })
            .collect();
        if out.is_empty() {
            return cmd;
        }
        match cmd {
            CurationCommand::Batch { children } => out.extend(children),
            other => out.push(other),
        }
        CurationCommand::Batch { children: out }
    }

    /// Replace `cluster`'s tags, returning the previous set.
    fn swap_tags(&mut self, cluster: ClusterId, tags: Vec<String>) -> Vec<String> {
        if self.tags.len() <= cluster.idx() {
            if tags.is_empty() {
                return Vec::new();
            }
            self.tags.resize_with(cluster.idx() + 1, Vec::new);
        }
        std::mem::replace(&mut self.tags[cluster.idx()], tags)
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

    /// The template most of this cluster's spikes were assigned to, or
    /// `None` when spike templates are not seeded. After merges a cluster
    /// spans several templates; the mode is what phy shows as "its"
    /// waveform.
    pub fn primary_template(&self, cluster: ClusterId) -> Option<u32> {
        let templates = self.spike_templates(cluster);
        if templates.is_empty() {
            return None;
        }
        let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for &t in templates {
            *counts.entry(t).or_insert(0) += 1;
        }
        // Ties resolve to the lowest id so the answer is deterministic.
        counts
            .into_iter()
            .max_by_key(|&(t, c)| (c, std::cmp::Reverse(t)))
            .map(|(t, _)| t)
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

    /// Iterator over the commands on the redo stack, next-to-redo last.
    pub fn redo_commands(&self) -> impl Iterator<Item = &CurationCommand> {
        self.redo.iter().map(|e| &e.cmd)
    }

    /// The forward commands on the undo stack, oldest first, with every
    /// `Split.new_cluster` replaced by the id the index actually allocated
    /// (the journal stores a placeholder; see `apply_forward`). This is the
    /// list interchange writers need, since downstream tools must know
    /// which unit a split produced.
    pub fn applied_commands(&self) -> Vec<CurationCommand> {
        fn resolve<L: Copy>(cmd: &CurationCommand, inv: &InverseOp<L>) -> CurationCommand {
            match (cmd, inv) {
                (
                    CurationCommand::Split {
                        cluster,
                        spike_idx,
                        by,
                        ..
                    },
                    InverseOp::Split { record, .. },
                ) => CurationCommand::Split {
                    cluster: *cluster,
                    spike_idx: spike_idx.clone(),
                    new_cluster: record.new_cluster,
                    by: by.clone(),
                },
                (CurationCommand::Batch { children }, InverseOp::Batch { inverses }) => {
                    CurationCommand::Batch {
                        children: children
                            .iter()
                            .zip(inverses)
                            .map(|(c, i)| resolve(c, i))
                            .collect(),
                    }
                }
                _ => cmd.clone(),
            }
        }
        self.history
            .iter()
            .map(|e| resolve(&e.cmd, &e.inverse))
            .collect()
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
        // Fail closed before anything reaches the journal.
        match &cmd {
            CurationCommand::Redo => {
                if let Some(e) = self.redo.last() {
                    self.check_gate(&e.cmd)?;
                }
            }
            forward => self.check_gate(forward)?,
        }
        let prev_head = self.journal.head();
        let affected = self.cache_identities_for_command(&cmd);
        let n_before = self.cluster_index.n_spikes() as u64;
        self.journal.append(&cmd)?;
        if let Some(sink) = &mut self.record_sink {
            sink(&cmd);
        }
        let new_head = self.journal.head();
        self.apply_record(cmd)?;
        self.record_cache_invalidation(prev_head, new_head, n_before, affected);
        Ok(())
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
                let inverse = match self.apply_forward(&entry.cmd) {
                    Ok(inv) => inv,
                    Err(e) => {
                        self.redo.push(entry);
                        return Err(e);
                    }
                };
                self.history.push(HistoryEntry {
                    cmd: entry.cmd,
                    inverse,
                });
            }
            forward => {
                let inverse = self.apply_forward(&forward)?;
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
    fn apply_forward(&mut self, cmd: &CurationCommand) -> Result<InverseOp<P::Label>>
    where
        P: ApplyPhyLabel,
    {
        match cmd {
            CurationCommand::AddSpikes(batch) => {
                self.check_gate(cmd)?;
                let record = self.cluster_index.add_spikes(batch)?;
                // Auto units that gained spikes are tuned; undo restores their tags.
                let mut touched = batch.clusters.clone();
                touched.sort_unstable();
                touched.dedup();
                let mut prev_tags = Vec::new();
                for c in touched {
                    if *self.origin(c) == UnitOrigin::Auto {
                        let tuned = self.tune_tags(c, self.tags(c).to_vec());
                        prev_tags.push((c, self.swap_tags(c, tuned)));
                    }
                }
                Ok(InverseOp::AddSpikes { record, prev_tags })
            }
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
                    match self.apply_forward(child) {
                        Ok(inv) => inverses.push(inv),
                        Err(e) => {
                            // Roll back the children already applied.
                            for inv in inverses.into_iter().rev() {
                                self.apply_inverse(inv);
                            }
                            return Err(e);
                        }
                    }
                }
                Ok(InverseOp::Batch { inverses })
            }
            other => Ok(self.apply_simple(other)),
        }
    }

    /// Gate for commands that need more than the in-memory state: AddSpikes
    /// is only valid on a phy-directory provider with no remote record sink
    /// (a sink would have to carry the payload, and replay must be local).
    fn check_gate(&self, cmd: &CurationCommand) -> Result<()> {
        match cmd {
            CurationCommand::AddSpikes(batch) => {
                if self.provider.phy_dir().is_none() {
                    bail!("AddSpikes needs a phy/Kilosort directory provider");
                }
                if self.record_sink.is_some() {
                    bail!("AddSpikes is not supported on a session with a remote record sink");
                }
                self.cluster_index.check_add_spikes(batch)
            }
            CurationCommand::Batch { children } => {
                children.iter().try_for_each(|c| self.check_gate(c))
            }
            _ => Ok(()),
        }
    }

    /// The infallible forward ops (everything but `Batch` and `AddSpikes`).
    fn apply_simple(&mut self, cmd: &CurationCommand) -> InverseOp<P::Label>
    where
        P: ApplyPhyLabel,
    {
        match cmd {
            CurationCommand::Relabel { cluster, op } => {
                let prev = self.label(*cluster).unwrap_or_default();
                if let Some(slot) = self.labels.get_mut(cluster.idx()) {
                    *slot = P::label_from_op(*op);
                }
                InverseOp::Relabel {
                    cluster: *cluster,
                    prev,
                }
            }
            CurationCommand::Merge { sources, target } => {
                // A target listed among its sources is not its own parent
                // (the index already skips the self-move).
                let sources: Vec<ClusterId> =
                    sources.iter().copied().filter(|s| s != target).collect();
                let sources = &sources[..];
                // Capture pre-merge labels so undo can restore them.
                let labels: Vec<(ClusterId, P::Label)> = sources
                    .iter()
                    .map(|&s| (s, self.label(s).unwrap_or_default()))
                    .collect();
                // The merged unit carries every source's provenance.
                let mut union = self.tags(*target).to_vec();
                for &s in sources {
                    union.extend_from_slice(self.tags(s));
                }
                union.sort_unstable();
                union.dedup();
                let union = self.tune_tags(*target, union);
                let target_tags = (*target, self.swap_tags(*target, union));
                // Direct lineage: the sources join the target's parents.
                let mut parents = self.parents(*target).to_vec();
                for s in sources {
                    if !parents.contains(s) {
                        parents.push(*s);
                    }
                }
                let target_parents = set_slot(&mut self.parents, *target, parents);
                // Source notes survive as `#<id>: <note>` lines on the target.
                // A line the target already has is skipped, so replay stays clean.
                let mut note = self.note(*target).to_string();
                for &s in sources {
                    let text = self.note(s);
                    if text.is_empty() {
                        continue;
                    }
                    let line = format!("#{}: {text}", s.0);
                    if !format!("\n{note}\n").contains(&format!("\n{line}\n")) {
                        if !note.is_empty() {
                            note.push('\n');
                        }
                        note.push_str(&line);
                    }
                }
                let target_note = set_slot(&mut self.notes, *target, note);
                let record = self.cluster_index.merge(sources, *target);
                InverseOp::Merge {
                    labels,
                    target_tags,
                    target_parents,
                    target_note,
                    record,
                }
            }
            CurationCommand::Split {
                cluster,
                spike_idx,
                new_cluster: _,
                by,
            } => {
                // We auto-allocate a fresh cluster id rather than honouring
                // the requested `new_cluster` field — the on-disk schema
                // captures user intent ("split this cluster"), but the
                // physical id is determined by the index that performs the
                // mutation, which guarantees no collision after replay.
                let record = self.cluster_index.split(*cluster, spike_idx);
                // The child inherits the parent's provenance but not its
                // tuned/split markers; it is a sorrel unit with one parent.
                let parent_tags = self.tags(*cluster).to_vec();
                let inherited = parent_tags
                    .iter()
                    .filter(|t| !matches!(t.as_str(), TAG_TUNED | TAG_SPLIT))
                    .cloned()
                    .collect();
                let child = record.new_cluster;
                self.swap_tags(child, inherited);
                set_slot(&mut self.origins, child, UnitOrigin::Sorrel(by.clone()));
                set_slot(&mut self.parents, child, vec![*cluster]);
                set_slot(&mut self.notes, child, String::new());
                if *self.origin(*cluster) == UnitOrigin::Auto {
                    let mut marked = parent_tags.clone();
                    marked.push(TAG_SPLIT.to_string());
                    marked.sort_unstable();
                    marked.dedup();
                    self.swap_tags(*cluster, marked);
                }
                InverseOp::Split {
                    record,
                    parent_tags,
                }
            }
            CurationCommand::Note { cluster, text } => InverseOp::Note {
                cluster: *cluster,
                prev: set_slot(&mut self.notes, *cluster, text.clone()),
            },
            CurationCommand::Tag { cluster, tags } => InverseOp::Tag {
                cluster: *cluster,
                prev: self.swap_tags(*cluster, tags.clone()),
            },
            CurationCommand::Batch { .. } | CurationCommand::AddSpikes(_) => {
                unreachable!("handled by apply_forward")
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
                if let Some(slot) = self.labels.get_mut(cluster.idx()) {
                    *slot = prev;
                }
            }
            InverseOp::Merge {
                labels,
                target_tags,
                target_parents,
                target_note,
                record,
            } => {
                self.swap_tags(target_tags.0, target_tags.1);
                set_slot(&mut self.parents, target_tags.0, target_parents);
                set_slot(&mut self.notes, target_tags.0, target_note);
                self.cluster_index.unmerge(record);
                for (c, l) in labels {
                    if let Some(slot) = self.labels.get_mut(c.idx()) {
                        *slot = l;
                    }
                }
            }
            InverseOp::Split {
                record,
                parent_tags,
            } => {
                let child = record.new_cluster;
                self.swap_tags(child, Vec::new());
                set_slot(&mut self.origins, child, UnitOrigin::Auto);
                set_slot(&mut self.parents, child, Vec::new());
                set_slot(&mut self.notes, child, String::new());
                self.swap_tags(record.source, parent_tags);
                self.cluster_index.unsplit(record);
            }
            InverseOp::Note { cluster, prev } => {
                set_slot(&mut self.notes, cluster, prev);
            }
            InverseOp::Tag { cluster, prev } => {
                self.swap_tags(cluster, prev);
            }
            InverseOp::AddSpikes { record, prev_tags } => {
                for (c, tags) in prev_tags {
                    self.swap_tags(c, tags);
                }
                self.cluster_index.remove_added(record)
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

    fn record_cache_invalidation(
        &mut self,
        prev_head: u64,
        new_head: u64,
        n_spikes: u64,
        clusters: Vec<CacheClusterIdentity>,
    ) {
        if prev_head == new_head {
            return;
        }
        if self.cache_invalidations.len() == MAX_CACHE_INVALIDATIONS {
            self.cache_invalidations.pop_front();
        }
        self.cache_invalidations.push_back(CacheInvalidation {
            prev_head,
            new_head,
            n_spikes,
            clusters,
        });
    }

    fn cache_identities_for_command(&self, cmd: &CurationCommand) -> Vec<CacheClusterIdentity> {
        let clusters = match cmd {
            c if touches_all(c) => (0..self.n_clusters()).map(ClusterId).collect(),
            forward => affected_clusters(forward),
        };
        clusters
            .into_iter()
            .filter(|cluster| cluster.0 < self.n_clusters())
            .map(|cluster| self.cache_identity(cluster))
            .collect()
    }

    fn cache_identity(&self, cluster: ClusterId) -> CacheClusterIdentity {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"sorrel-pc-subspace-cluster-spikes-v1");
        // The cluster's own bucket of global indices, ascending.
        let indices = self.cluster_index.global_indices(cluster);
        for &global_idx in indices {
            hasher.update(&(global_idx as u64).to_le_bytes());
        }
        let count = indices.len();
        CacheClusterIdentity {
            cluster,
            spike_count: count,
            cluster_spike_fingerprint: *hasher.finalize().as_bytes(),
        }
    }
}

/// Review-queue ordering (see [`crate::proposers::attention_with`]).
impl<P: DataProvider> Session<P> {
    /// `(score, reasons)` of how much `cluster` needs a human. Ranks every
    /// merge/split to find this cluster's share, so use
    /// [`Self::attention_all`] (or `proposers::attention_all` with
    /// suggestions you already hold) when scoring more than one.
    pub fn attention(&self, cluster: ClusterId) -> (f32, Vec<&'static str>) {
        let (merges, splits) = self.all_suggestions();
        let globals = crate::proposers::cluster_globals(self, cluster);
        crate::proposers::attention_with(self, cluster, &globals, &merges, &splits)
    }

    /// Attention of every non-empty cluster, highest first; suggestions are
    /// computed once.
    pub fn attention_all(&self) -> Vec<(ClusterId, f32, Vec<&'static str>)> {
        let (merges, splits) = self.all_suggestions();
        crate::proposers::attention_all(self, &merges, &splits)
    }

    /// Untruncated merge and split suggestions (default thresholds).
    fn all_suggestions(
        &self,
    ) -> (
        Vec<crate::suggest::MergeCandidate>,
        Vec<crate::suggest::SplitCandidate>,
    ) {
        let cfg = crate::suggest::SuggestConfig {
            top_merges: usize::MAX,
            top_splits: usize::MAX,
            ..Default::default()
        };
        (
            crate::suggest::rank_merge_candidates(self, &cfg),
            crate::suggest::rank_split_candidates(self, &cfg),
        )
    }
}

/// Commands whose cache effect is not confined to named clusters.
fn touches_all(cmd: &CurationCommand) -> bool {
    match cmd {
        CurationCommand::Undo | CurationCommand::Redo | CurationCommand::AddSpikes(_) => true,
        CurationCommand::Batch { children } => children.iter().any(touches_all),
        _ => false,
    }
}

fn affected_clusters(cmd: &CurationCommand) -> Vec<ClusterId> {
    let mut out = Vec::new();
    collect_affected_clusters(cmd, &mut out);
    out.sort_unstable();
    out.dedup();
    out
}

fn collect_affected_clusters(cmd: &CurationCommand, out: &mut Vec<ClusterId>) {
    match cmd {
        CurationCommand::Relabel { cluster, .. } | CurationCommand::Note { cluster, .. } => {
            out.push(*cluster);
        }
        CurationCommand::Merge { sources, target } => {
            out.push(*target);
            out.extend(sources.iter().copied());
        }
        CurationCommand::Split { cluster, .. } => out.push(*cluster),
        CurationCommand::Batch { children } => {
            for child in children {
                collect_affected_clusters(child, out);
            }
        }
        // Tags never touch spikes, so no cache depends on them.
        CurationCommand::Tag { .. }
        | CurationCommand::Undo
        | CurationCommand::Redo
        | CurationCommand::AddSpikes(_) => {}
    }
}

fn amplitudes_cache_key_for(provider_identity: &[u8], journal_head: u64) -> CacheKey {
    let fingerprint = Fingerprint::builder()
        .add_provider_identity(provider_identity)
        .add_journal_head(journal_head)
        .add_algo_version(AMPLITUDES_ALGO_VERSION)
        .finish();
    CacheKey::new(AMPLITUDES_KIND, AMPLITUDES_ALGO_VERSION, fingerprint)
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

        fn sample_rate(&self) -> f32 {
            1000.0
        }
        fn n_channels(&self) -> u32 {
            self.n_channels
        }
        fn n_samples(&self) -> SampleIndex {
            SampleIndex((self.trace.len() / self.n_channels.max(1) as usize) as u64)
        }
        fn n_clusters(&self) -> u32 {
            self.spikes.len() as u32
        }
        fn spike_times(&self, cluster: ClusterId) -> &[SampleIndex] {
            self.spikes
                .get(cluster.idx())
                .map(Vec::as_slice)
                .unwrap_or(&[])
        }
        fn trace(&self, start: SampleIndex, len: u32) -> TraceSlice<'_> {
            let nc = self.n_channels as usize;
            let s = (start.idx() * nc).min(self.trace.len());
            let e = (s + len as usize * nc).min(self.trace.len());
            TraceSlice {
                start,
                n_channels: self.n_channels,
                samples: TraceSamples::I16(&self.trace[s..e]),
            }
        }
        fn initial_labels(&self) -> Vec<Self::Label> {
            self.labels.clone()
        }
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
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)],
                vec![SampleIndex(20), SampleIndex(200)],
                vec![SampleIndex(100)],
            ],
            labels: vec![MockLabel::Unsorted; 3],
            trace: vec![0i16; 4 * 16],
            n_channels: 4,
        };
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::open(&dir.path().join("j.sqlite")).unwrap();
        let session = Session::new(provider, journal);
        (session, dir)
    }

    #[test]
    fn set_label_command_rejects_reserved_sidecar_names() {
        let (s, _d) = fresh_session();
        for k in ["sorrel_annotations", "sorrel_origin", "group", "id"] {
            assert!(s.set_label_command(ClusterId(0), k, "x").is_err(), "{k}");
        }
        assert!(s.set_label_command(ClusterId(0), "my_label", "x").is_ok());
    }

    #[test]
    fn new_session_seeds_labels_and_buckets_from_provider() {
        let (s, _d) = fresh_session();
        assert_eq!(s.labels().len(), 3);
        assert_eq!(s.label(ClusterId(0)), Some(MockLabel::Unsorted));
        assert_eq!(s.label(ClusterId(99)), None);
        assert_eq!(s.history_len(), 0);
        assert_eq!(s.redo_len(), 0);
        // Bucketed access flows through the ClusterIndex now.
        assert_eq!(
            s.spike_times(ClusterId(0)),
            &[SampleIndex(10), SampleIndex(30), SampleIndex(150)]
        );
        assert_eq!(s.spike_times(ClusterId(2)), &[SampleIndex(100)]);
    }

    #[test]
    fn dispatch_relabel_updates_label_and_history() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(1),
            op: PhyLabelOp::SetGood,
        })
        .unwrap();
        assert_eq!(s.label(ClusterId(1)), Some(MockLabel::Good));
        assert_eq!(s.history_len(), 1);
    }

    #[test]
    fn dispatch_merge_reassigns_spikes_to_target() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(0), ClusterId(1)],
            target: ClusterId(2),
        })
        .unwrap();
        assert!(s.spike_times(ClusterId(0)).is_empty());
        assert!(s.spike_times(ClusterId(1)).is_empty());
        assert_eq!(
            s.spike_times(ClusterId(2)),
            &[
                SampleIndex(10),
                SampleIndex(20),
                SampleIndex(30),
                SampleIndex(100),
                SampleIndex(150),
                SampleIndex(200)
            ]
        );
        // Labels are NOT changed by merge anymore — phy keeps source labels
        // alongside the empty bucket and lets the user filter "no spikes".
        assert_eq!(s.label(ClusterId(0)), Some(MockLabel::Unsorted));
    }

    #[test]
    fn undo_merge_restores_buckets_and_source_labels() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: PhyLabelOp::SetGood,
        })
        .unwrap();
        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(0)],
            target: ClusterId(2),
        })
        .unwrap();
        assert!(s.spike_times(ClusterId(0)).is_empty());

        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(
            s.spike_times(ClusterId(0)),
            &[SampleIndex(10), SampleIndex(30), SampleIndex(150)]
        );
        assert_eq!(s.label(ClusterId(0)), Some(MockLabel::Good));
    }

    #[test]
    fn batch_applies_children_in_order_and_undoes_atomically() {
        let (mut s, _d) = fresh_session();
        let batch = CurationCommand::batch(vec![
            CurationCommand::Relabel {
                cluster: ClusterId(0),
                op: PhyLabelOp::SetGood,
            },
            CurationCommand::Relabel {
                cluster: ClusterId(1),
                op: PhyLabelOp::SetMua,
            },
            CurationCommand::Merge {
                sources: vec![ClusterId(0)],
                target: ClusterId(1),
            },
        ])
        .unwrap();
        s.dispatch(batch).unwrap();
        // All three children took effect.
        assert_eq!(s.label(ClusterId(0)), Some(MockLabel::Good));
        assert_eq!(s.label(ClusterId(1)), Some(MockLabel::Mua));
        assert!(s.spike_times(ClusterId(0)).is_empty());
        assert_eq!(s.history_len(), 1, "batch counts as one history entry");

        // One undo reverts the entire batch.
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.label(ClusterId(0)), Some(MockLabel::Unsorted));
        assert_eq!(s.label(ClusterId(1)), Some(MockLabel::Unsorted));
        assert_eq!(
            s.spike_times(ClusterId(0)),
            &[SampleIndex(10), SampleIndex(30), SampleIndex(150)]
        );
        assert_eq!(s.history_len(), 0);
        assert_eq!(s.redo_len(), 1);

        // Redo re-applies the entire batch.
        s.dispatch(CurationCommand::Redo).unwrap();
        assert_eq!(s.label(ClusterId(0)), Some(MockLabel::Good));
        assert_eq!(s.label(ClusterId(1)), Some(MockLabel::Mua));
        assert!(s.spike_times(ClusterId(0)).is_empty());
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
            cluster: ClusterId(0),
            spike_idx: vec![1],
            new_cluster: ClusterId(99), // ignored: id auto-allocated
            by: Method::Manual,
        })
        .unwrap();
        assert_eq!(s.n_clusters(), n_before + 1);
        assert_eq!(
            s.spike_times(ClusterId(0)),
            &[SampleIndex(10), SampleIndex(150)]
        );
        assert_eq!(s.spike_times(ClusterId(n_before)), &[SampleIndex(30)]);
    }

    #[test]
    fn undo_split_restores_state_and_frees_cluster_id() {
        let (mut s, _d) = fresh_session();
        let n_before = s.n_clusters();
        let pre = s.spike_times(ClusterId(0)).to_vec();
        s.dispatch(CurationCommand::Split {
            cluster: ClusterId(0),
            spike_idx: vec![0, 2],
            new_cluster: ClusterId(0),
            by: Method::Manual,
        })
        .unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.n_clusters(), n_before);
        assert_eq!(s.spike_times(ClusterId(0)), &pre[..]);
    }

    #[test]
    fn dispatch_note_is_a_noop() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Note {
            cluster: ClusterId(0),
            text: "x".into(),
        })
        .unwrap();
        assert_eq!(s.label(ClusterId(0)), Some(MockLabel::Unsorted));
        assert_eq!(s.history_len(), 1);
    }

    #[test]
    fn merge_appends_source_notes_and_undo_redo_restores_them() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Note {
            cluster: ClusterId(1),
            text: "one".into(),
        })
        .unwrap();
        s.dispatch(CurationCommand::Note {
            cluster: ClusterId(2),
            text: "two".into(),
        })
        .unwrap();
        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(1)],
            target: ClusterId(2),
        })
        .unwrap();
        assert_eq!(s.note(ClusterId(2)), "two\n#1: one");
        assert_eq!(s.note(ClusterId(1)), "one", "source keeps its own note");
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.note(ClusterId(2)), "two");
        s.dispatch(CurationCommand::Redo).unwrap();
        assert_eq!(s.note(ClusterId(2)), "two\n#1: one", "redo appends once");
    }

    #[test]
    fn merge_with_target_among_sources_is_not_its_own_parent() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(2), ClusterId(1)],
            target: ClusterId(2),
        })
        .unwrap();
        assert_eq!(s.parents(ClusterId(2)), [ClusterId(1)]);
    }

    #[test]
    fn origin_parents_notes_and_markers_follow_undo_redo_and_replay() {
        let (mut s, d) = fresh_session();
        let tags = |s: &Session<MockProvider>, c: u32| s.tags(ClusterId(c)).to_vec();
        let state = |s: &Session<MockProvider>| {
            (0..5)
                .map(|c| {
                    let c = ClusterId(c);
                    (
                        s.origin_label(c),
                        s.parents(c).to_vec(),
                        s.note(c).to_string(),
                        s.tags(c).to_vec(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let pristine = state(&s);
        assert_eq!(s.sorter(), "unknown");
        assert_eq!(s.origin_label(ClusterId(0)), "Auto[unknown]");

        s.dispatch(CurationCommand::Note {
            cluster: ClusterId(0),
            text: "hello".into(),
        })
        .unwrap();
        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(1)],
            target: ClusterId(2),
        })
        .unwrap();
        s.dispatch(CurationCommand::Split {
            cluster: ClusterId(2),
            spike_idx: vec![0],
            new_cluster: ClusterId(0),
            by: Method::Algo("gmm".into()),
        })
        .unwrap();
        let child = ClusterId(3);
        assert_eq!(s.note(ClusterId(0)), "hello");
        assert_eq!(s.parents(ClusterId(2)), [ClusterId(1)]);
        assert_eq!(s.origin(ClusterId(2)), &UnitOrigin::Auto);
        assert_eq!(tags(&s, 2), ["split", "tuned"]);
        assert_eq!(s.origin_label(child), "Sorrel[Algo:gmm]");
        assert_eq!(s.parents(child), [ClusterId(2)]);
        assert!(s.tags(child).is_empty(), "child drops tuned/split");
        let done = state(&s);

        for _ in 0..3 {
            s.dispatch(CurationCommand::Undo).unwrap();
        }
        assert_eq!(state(&s), pristine);
        for _ in 0..3 {
            s.dispatch(CurationCommand::Redo).unwrap();
        }
        assert_eq!(state(&s), done);

        // Replaying the journal on a fresh session reproduces the state.
        let provider = MockProvider {
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)],
                vec![SampleIndex(20), SampleIndex(200)],
                vec![SampleIndex(100)],
            ],
            labels: vec![MockLabel::Unsorted; 3],
            trace: vec![],
            n_channels: 1,
        };
        let mut r = Session::new(provider, Journal::open(&d.path().join("j.sqlite")).unwrap());
        r.replay_journal().unwrap();
        assert_eq!(state(&r), done);
    }

    #[test]
    fn tags_follow_merge_split_undo_and_replay() {
        let (mut s, d) = fresh_session();
        let tag = |c: u32, t: &[&str]| CurationCommand::Tag {
            cluster: ClusterId(c),
            tags: t.iter().map(|s| s.to_string()).collect(),
        };
        s.dispatch(tag(0, &["src:bombcell", "agree"])).unwrap();
        s.dispatch(tag(2, &["review:merge"])).unwrap();
        assert_eq!(s.tags(ClusterId(0)), ["src:bombcell", "agree"]);
        assert!(s.tags(ClusterId(1)).is_empty());

        // Merge unions source tags into the target; undo restores them.
        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(0)],
            target: ClusterId(2),
        })
        .unwrap();
        assert_eq!(
            s.tags(ClusterId(2)),
            ["agree", "review:merge", "src:bombcell", "tuned"]
        );
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.tags(ClusterId(2)), ["review:merge"]);

        // A split child inherits the parent's tags and loses them on undo.
        let child = ClusterId(s.n_clusters());
        s.dispatch(CurationCommand::Split {
            cluster: ClusterId(0),
            spike_idx: vec![1],
            new_cluster: child,
            by: Method::Manual,
        })
        .unwrap();
        assert_eq!(s.tags(child), ["src:bombcell", "agree"]);
        s.dispatch(CurationCommand::Undo).unwrap();
        assert!(s.tags(child).is_empty());

        // A human decision clears review tags in the same undoable batch,
        // and leaves untagged clusters untagged.
        let human = s.with_human_review(CurationCommand::Relabel {
            cluster: ClusterId(2),
            op: PhyLabelOp::SetGood,
        });
        s.dispatch(human).unwrap();
        assert_eq!(s.tags(ClusterId(2)), ["src:human"]);
        let plain = s.with_human_review(CurationCommand::Relabel {
            cluster: ClusterId(1),
            op: PhyLabelOp::SetGood,
        });
        assert!(matches!(plain, CurationCommand::Relabel { .. }));
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.tags(ClusterId(2)), ["review:merge"]);

        // Retag undo restores the previous set; the journal replays it.
        s.dispatch(tag(0, &["src:human"])).unwrap();
        s.dispatch(tag(0, &[])).unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.tags(ClusterId(0)), ["src:human"]);

        let path = s.journal_path().to_path_buf();
        let provider = MockProvider {
            spikes: (0..3)
                .map(|c| s.provider.spike_times(ClusterId(c)).to_vec())
                .collect(),
            labels: vec![MockLabel::Unsorted; 3],
            trace: vec![],
            n_channels: 1,
        };
        drop(s);
        let mut r = Session::new(provider, Journal::open(&path).unwrap());
        r.replay_journal().unwrap();
        assert_eq!(r.tags(ClusterId(0)), ["src:human"]);
        assert_eq!(r.tags(ClusterId(2)), ["review:merge"]);
        drop(d);
    }

    #[test]
    fn undo_restores_previous_label() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(1),
            op: PhyLabelOp::SetGood,
        })
        .unwrap();
        assert_eq!(s.label(ClusterId(1)), Some(MockLabel::Good));

        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.label(ClusterId(1)), Some(MockLabel::Unsorted));
        assert_eq!(s.history_len(), 0);
        assert_eq!(s.redo_len(), 1);
    }

    #[test]
    fn redo_reapplies_after_undo() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(1),
            op: PhyLabelOp::SetMua,
        })
        .unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.label(ClusterId(1)), Some(MockLabel::Unsorted));
        s.dispatch(CurationCommand::Redo).unwrap();
        assert_eq!(s.label(ClusterId(1)), Some(MockLabel::Mua));
        assert_eq!(s.redo_len(), 0);
        assert_eq!(s.history_len(), 1);
    }

    #[test]
    fn redo_merge_reapplies_spike_reassignment() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(0), ClusterId(1)],
            target: ClusterId(2),
        })
        .unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(
            s.spike_times(ClusterId(0)),
            &[SampleIndex(10), SampleIndex(30), SampleIndex(150)]
        );

        s.dispatch(CurationCommand::Redo).unwrap();
        assert!(s.spike_times(ClusterId(0)).is_empty());
        assert_eq!(
            s.spike_times(ClusterId(2)),
            &[
                SampleIndex(10),
                SampleIndex(20),
                SampleIndex(30),
                SampleIndex(100),
                SampleIndex(150),
                SampleIndex(200)
            ]
        );
    }

    #[test]
    fn forward_op_after_undo_clears_redo() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: PhyLabelOp::SetGood,
        })
        .unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.redo_len(), 1);

        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: PhyLabelOp::SetMua,
        })
        .unwrap();
        assert_eq!(s.redo_len(), 0);
        assert_eq!(s.label(ClusterId(0)), Some(MockLabel::Mua));
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
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: PhyLabelOp::SetGood,
        })
        .unwrap();
        s.dispatch(CurationCommand::Redo).unwrap();
        assert_eq!(s.label(ClusterId(0)), Some(MockLabel::Good));
        assert_eq!(s.redo_len(), 0);
    }

    #[test]
    fn multi_step_undo_walks_the_stack_in_reverse() {
        let (mut s, _d) = fresh_session();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: PhyLabelOp::SetGood,
        })
        .unwrap();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: PhyLabelOp::SetMua,
        })
        .unwrap();
        s.dispatch(CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: PhyLabelOp::SetNoise,
        })
        .unwrap();
        assert_eq!(s.label(ClusterId(0)), Some(MockLabel::Noise));

        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.label(ClusterId(0)), Some(MockLabel::Mua));
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.label(ClusterId(0)), Some(MockLabel::Good));
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.label(ClusterId(0)), Some(MockLabel::Unsorted));
        assert_eq!(s.history_len(), 0);
    }

    #[test]
    fn replay_journal_reconstructs_state_and_redo_stack() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.sqlite");
        {
            let mut j = Journal::open(&path).unwrap();
            j.append(&CurationCommand::Relabel {
                cluster: ClusterId(0),
                op: PhyLabelOp::SetMua,
            })
            .unwrap();
            j.append(&CurationCommand::Relabel {
                cluster: ClusterId(1),
                op: PhyLabelOp::SetNoise,
            })
            .unwrap();
            j.append(&CurationCommand::Undo).unwrap();
        }

        let provider = MockProvider {
            spikes: vec![
                vec![SampleIndex(1)],
                vec![SampleIndex(2)],
                vec![SampleIndex(3)],
            ],
            labels: vec![MockLabel::Unsorted; 3],
            trace: vec![],
            n_channels: 1,
        };
        let journal = Journal::open(&path).unwrap();
        let mut session = Session::new(provider, journal);
        session.replay_journal().unwrap();

        assert_eq!(session.label(ClusterId(0)), Some(MockLabel::Mua));
        assert_eq!(session.label(ClusterId(1)), Some(MockLabel::Unsorted));
        assert_eq!(session.label(ClusterId(2)), Some(MockLabel::Unsorted));
        assert_eq!(session.history_len(), 1);
        assert_eq!(session.redo_len(), 1);
    }

    #[test]
    fn replay_journal_with_orphan_undo_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.sqlite");
        {
            let mut j = Journal::open(&path).unwrap();
            j.append(&CurationCommand::Undo).unwrap();
        }

        let provider = MockProvider {
            spikes: vec![],
            labels: vec![],
            trace: vec![],
            n_channels: 1,
        };
        let journal = Journal::open(&path).unwrap();
        let mut session = Session::new(provider, journal);
        assert!(session.replay_journal().is_err());
    }

    #[test]
    fn replay_journal_reconstructs_merged_buckets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.sqlite");
        {
            let mut j = Journal::open(&path).unwrap();
            j.append(&CurationCommand::Merge {
                sources: vec![ClusterId(0), ClusterId(1)],
                target: ClusterId(2),
            })
            .unwrap();
        }

        let provider = MockProvider {
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)],
                vec![SampleIndex(20), SampleIndex(200)],
                vec![SampleIndex(100)],
            ],
            labels: vec![MockLabel::Unsorted; 3],
            trace: vec![],
            n_channels: 1,
        };
        let journal = Journal::open(&path).unwrap();
        let mut session = Session::new(provider, journal);
        session.replay_journal().unwrap();

        assert_eq!(
            session.spike_times(ClusterId(2)),
            &[
                SampleIndex(10),
                SampleIndex(20),
                SampleIndex(30),
                SampleIndex(100),
                SampleIndex(150),
                SampleIndex(200)
            ]
        );
        assert!(session.spike_times(ClusterId(0)).is_empty());
    }

    #[test]
    fn merge_then_cache_invalidation_evicts_pre_merge_cluster_archive() {
        let (mut s, dir) = fresh_session();
        let store = CacheStore::open(dir.path()).unwrap();
        let old_identity = s.cache_identity(ClusterId(0));
        let old_key = pc_subspace::key(pc_subspace::KeyParts {
            session_identity: &s.provider.identity_bytes(),
            n_spikes_total: s.cluster_index.n_spikes() as u64,
            cluster: old_identity.cluster,
            cluster_spike_count: old_identity.spike_count,
            cluster_spike_fingerprint: old_identity.cluster_spike_fingerprint,
            d_pcs: DEFAULT_D,
            channel_idx: DEFAULT_CHANNEL_IDX,
            max_background: DEFAULT_MAX_BACKGROUND,
        });
        store
            .put(
                &old_key,
                &pc_subspace::CachedPcSubspace {
                    features: vec![1.0, 2.0, 3.0],
                    is_in_cluster: vec![true],
                    d: DEFAULT_D as u32,
                    algo_version: pc_subspace::ALGO_VERSION,
                },
            )
            .unwrap();
        let old_path = store
            .root()
            .join(old_key.kind)
            .join(format!("{}.rkyv", old_key.fingerprint.hex()));
        assert!(old_path.exists());

        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(0)],
            target: ClusterId(2),
        })
        .unwrap();
        s.invalidate_caches(&store).unwrap();
        drop(store);

        let reopened = CacheStore::open(dir.path()).unwrap();
        assert!(
            reopened
                .get::<pc_subspace::CachedPcSubspace>(&old_key)
                .unwrap()
                .is_none(),
            "pre-merge archive should not survive save/next-open invalidation"
        );
        assert!(!old_path.exists());
    }

    /// The bucket-based identity must hash exactly what the old full scan
    /// did, or existing cache entries would silently stop matching.
    #[test]
    fn cache_identity_matches_the_full_scan_after_merge_and_split() {
        let reference = |s: &Session<MockProvider>, cluster: ClusterId| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"sorrel-pc-subspace-cluster-spikes-v1");
            let mut count = 0usize;
            for (g, &c) in s.spike_clusters_global().iter().enumerate() {
                if c == cluster {
                    count += 1;
                    hasher.update(&(g as u64).to_le_bytes());
                }
            }
            (count, *hasher.finalize().as_bytes())
        };
        let (mut s, _d) = fresh_session();
        let check = |s: &Session<MockProvider>| {
            for c in (0..s.n_clusters()).map(ClusterId) {
                let id = s.cache_identity(c);
                assert_eq!(
                    (id.spike_count, id.cluster_spike_fingerprint),
                    reference(s, c)
                );
            }
        };
        check(&s);
        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(0)],
            target: ClusterId(2),
        })
        .unwrap();
        check(&s);
        s.dispatch(CurationCommand::Split {
            cluster: ClusterId(2),
            spike_idx: vec![1, 3],
            new_cluster: ClusterId(0),
            by: crate::command::Method::Manual,
        })
        .unwrap();
        check(&s);
        s.dispatch(CurationCommand::Undo).unwrap();
        check(&s);
    }
}

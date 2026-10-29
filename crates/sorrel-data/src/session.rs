use crate::cache::amplitudes::{
    CachedAmplitudes, ALGO_VERSION as AMPLITUDES_ALGO_VERSION, KIND as AMPLITUDES_KIND,
};
use crate::cache::{isolation, pc_subspace};
use crate::cluster_index::{AddRecord, ClusterIndex, MergeRecord, SplitRecord};
use crate::command::{Actor, CurationCommand, Method, PhyLabelOp, UnitOrigin};
use crate::feature_subspace::{DEFAULT_CHANNEL_IDX, DEFAULT_D, DEFAULT_MAX_BACKGROUND};
use crate::journal::{record_hash, Journal, Record};
use crate::quality_ext::DEFAULT_K_NN;
#[cfg(test)]
use anyhow::ensure;
use anyhow::{bail, Result};
use sorrel_cache::{CacheKey, CacheStore, Fingerprint};
use sorrel_io::{
    ClusterId, DataProvider, HasAmplitudes, HasPcFeatures, HasSpikeTemplates, HasTemplateWaveforms,
    SampleIndex,
};
use std::collections::{HashMap, VecDeque};
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

/// `last_child` slot of a node (`None` = the root).
fn slot(node: Option<u64>) -> usize {
    node.map_or(0, |n| n as usize + 1)
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

/// One edit in the action tree. Node ids are indices into `Session::nodes`.
#[derive(Debug)]
struct Node<L: Copy> {
    parent: Option<u64>,
    actor: Actor,
    ts_ms: u64,
    cmd: CurationCommand,
    /// Chain hash of the records on the path to this node: the session
    /// version while it is the head.
    key: u64,
    /// Set exactly while the node is on the path to the head (captured when
    /// it is applied, consumed when it is reverted).
    inverse: Option<InverseOp<L>>,
}

/// Running state of [`Session::check_ids`] over a batch: the next id a
/// split allocates, and cluster sizes as the earlier children left them.
struct IdCheck {
    limit: usize,
    sizes: HashMap<ClusterId, usize>,
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
    /// The action tree; state at a node = provider (the root) + the path.
    nodes: Vec<Node<P::Label>>,
    head: Option<u64>,
    /// Most recently visited child per node (slot 0 = the root, node `n` at
    /// `n + 1`): where redo goes. Along the head's path it is the next node.
    last_child: Vec<Option<u64>>,
    /// `(node, artifacts hash)` of every save, oldest first.
    seals: Vec<(Option<u64>, u64)>,
    /// Node whose state the saved files hold.
    saved: Option<u64>,
    /// Artifacts hash of the files on disk at open (phy directories), used
    /// by replay to find which seal they are; `None` = clean at open.
    top_artifacts: Option<u64>,
    cache_invalidations: VecDeque<CacheInvalidation>,
    /// Observer called once per journaled record, right after the append is
    /// durable and before the in-memory state moves. Used to stream records
    /// to a remote host; `None` for ordinary local sessions.
    record_sink: Option<RecordSink>,
    /// Test seam: make every forward apply fail.
    #[cfg(test)]
    fail_apply: bool,
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
            nodes: Vec::new(),
            head: None,
            last_child: vec![None],
            seals: Vec::new(),
            saved: None,
            top_artifacts: None,
            cache_invalidations: VecDeque::new(),
            record_sink: None,
            #[cfg(test)]
            fail_apply: false,
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

    /// The buckets are the provider's (the root's), which no edit changes,
    /// so the key is the root and edits never evict it.
    fn amplitudes_cache_key(&self) -> CacheKey {
        let fingerprint = Fingerprint::builder()
            .add_provider_identity(&self.provider.identity_bytes())
            .add_journal_head(self.journal.baseline())
            .add_algo_version(AMPLITUDES_ALGO_VERSION)
            .finish();
        CacheKey::new(AMPLITUDES_KIND, AMPLITUDES_ALGO_VERSION, fingerprint)
    }

    pub(crate) fn cache_dataset_dir(&self) -> Option<&Path> {
        self.journal.path().parent()
    }

    /// Identity of the replay-visible curation state: the chain hash of the
    /// path to the head (the baseline at the root). Every edit and head move
    /// changes it, and returning to a node returns its version, so it keys
    /// anything derived from curation state, across reopens too.
    #[inline]
    pub fn version(&self) -> u64 {
        self.key(self.head)
    }

    fn key(&self, node: Option<u64>) -> u64 {
        node.map_or(self.journal.baseline(), |n| self.nodes[n as usize].key)
    }

    /// Where the curation journal lives on disk.
    pub fn journal_path(&self) -> &Path {
        self.journal.path()
    }

    /// The sealed journal: baseline, record count, and path.
    pub fn journal(&self) -> &Journal {
        &self.journal
    }

    /// The checked-out node (`None` = the root).
    pub fn head(&self) -> Option<u64> {
        self.head
    }

    /// Number of nodes in the action tree.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// `(parent, actor, ts_ms, cmd)` of `node`; `None` for an unknown id.
    pub fn node(&self, node: u64) -> Option<(Option<u64>, &Actor, u64, &CurationCommand)> {
        let n = self.nodes.get(node as usize)?;
        Some((n.parent, &n.actor, n.ts_ms, &n.cmd))
    }

    /// Journal a save of the head whose files hash to `artifacts` (see
    /// `journal::artifacts_hash`). Appended *before* the files are published,
    /// so whatever an interrupted save leaves on disk matches a seal; call
    /// [`Self::mark_saved`] once they are.
    pub fn seal(&mut self, artifacts: u64) -> Result<()> {
        self.journal.append(&Record::Seal {
            node: self.head,
            artifacts,
        })?;
        self.seals.push((self.head, artifacts));
        Ok(())
    }

    /// `(node, artifacts hash)` of every save, oldest first.
    pub fn seals(&self) -> &[(Option<u64>, u64)] {
        &self.seals
    }

    /// The saved files now hold the head's state.
    pub fn mark_saved(&mut self) {
        self.saved = self.head;
    }

    /// True when the saved files hold the head's state.
    pub fn is_saved(&self) -> bool {
        self.saved == self.head
    }

    /// Tell replay which files are on disk (their artifacts hash), so it
    /// can mark the matching seal as saved rather than the head.
    pub fn set_top_artifacts(&mut self, artifacts: u64) {
        self.top_artifacts = Some(artifacts);
    }

    /// The artifacts hash of the files as this session last saw them (at
    /// open, then after each of its own saves); `None` when not tracked.
    pub fn top_artifacts(&self) -> Option<u64> {
        self.top_artifacts
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

    /// `cmd` as an agent edit: the touched clusters first gain `src:agent`,
    /// in the same undoable batch.
    pub fn with_agent_source(&self, cmd: CurationCommand) -> CurationCommand {
        let mut clusters = match &cmd {
            CurationCommand::Relabel { cluster, .. } | CurationCommand::Split { cluster, .. } => {
                vec![*cluster]
            }
            CurationCommand::Merge { sources, target } => [&[*target][..], sources].concat(),
            _ => Vec::new(),
        };
        clusters.sort_unstable();
        clusters.dedup();
        let mut children: Vec<CurationCommand> = clusters
            .into_iter()
            .filter(|&c| !self.tags(c).iter().any(|t| t == "src:agent"))
            .map(|cluster| {
                let mut tags = self.tags(cluster).to_vec();
                tags.push("src:agent".into());
                CurationCommand::Tag { cluster, tags }
            })
            .collect();
        if children.is_empty() {
            return cmd;
        }
        children.push(cmd);
        CurationCommand::Batch { children }
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

    /// Number of edits on the path to the head (the undo depth).
    #[inline]
    pub fn history_len(&self) -> usize {
        self.path(self.head).len()
    }

    /// Number of edits [`CurationCommand::Redo`] can walk down from the
    /// head (following the most recently visited child).
    #[inline]
    pub fn redo_len(&self) -> usize {
        self.redo_chain().len()
    }

    /// The commands on the path to the head, oldest first.
    pub fn history_commands(&self) -> impl Iterator<Item = &CurationCommand> {
        self.path(self.head)
            .into_iter()
            .map(|n| &self.nodes[n as usize].cmd)
    }

    /// The commands redo would apply, next-to-redo last.
    pub fn redo_commands(&self) -> impl Iterator<Item = &CurationCommand> {
        self.redo_chain()
            .into_iter()
            .rev()
            .map(|n| &self.nodes[n as usize].cmd)
    }

    /// The commands on the path to the head, oldest first, with every
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
        self.path(self.head)
            .into_iter()
            .map(|n| &self.nodes[n as usize])
            .map(|n| match &n.inverse {
                Some(inv) => resolve(&n.cmd, inv),
                None => n.cmd.clone(),
            })
            .collect()
    }

    /// Node ids from the root down to `node` (empty for the root).
    fn path(&self, mut node: Option<u64>) -> Vec<u64> {
        let mut out = Vec::new();
        while let Some(n) = node {
            out.push(n);
            node = self.nodes[n as usize].parent;
        }
        out.reverse();
        out
    }

    fn redo_chain(&self) -> Vec<u64> {
        let mut out = Vec::new();
        let mut at = self.head;
        while let Some(c) = self.last_child[slot(at)] {
            out.push(c);
            at = Some(c);
        }
        out
    }

    /// Rebuild the action tree from the journal and check out its head:
    /// the records only build the tree, then the path from the root to the
    /// head is applied once. Bad records (a node whose parent is not an
    /// earlier node, a move to an unknown node) make replay fail.
    ///
    /// A node's `parent` is explicit, so a node that is not a child of the
    /// current head (its failed predecessor's head fixup never reached the
    /// journal) simply moves the head to its parent. If the final node
    /// cannot be applied either (a crash before that fixup), it is dropped
    /// from the redo path and the head fixup is journaled now.
    pub fn replay_journal(&mut self) -> Result<()>
    where
        P: ApplyPhyLabel,
    {
        let mut target = self.head;
        // `(id, redo target of its parent before it)` while the last record is a node.
        let mut last_node = None;
        for rec in self.journal.replay()? {
            last_node = None;
            match rec {
                Record::Node { id, parent, .. } => {
                    if id != self.nodes.len() as u64 || parent.is_some_and(|p| p >= id) {
                        bail!("journal node {id} has parent {parent:?} outside the tree");
                    }
                    last_node = Some((id, self.last_child[slot(parent)]));
                    self.push_node(rec);
                    target = Some(id);
                }
                Record::Head { node } => {
                    if node.is_some_and(|n| n >= self.nodes.len() as u64) {
                        bail!("journal moves the head to unknown node {node:?}");
                    }
                    target = node;
                    self.visit(target);
                }
                Record::Seal { node, artifacts } => self.seals.push((node, artifacts)),
            }
        }
        if let Err(e) = self.move_to(target) {
            let Some((id, redo)) = last_node else {
                return Err(e);
            };
            log::warn!("journal node {id} does not apply ({e:#}); dropping it from the head");
            let parent = self.nodes[id as usize].parent;
            self.last_child[slot(parent)] = redo;
            self.move_to(parent)?;
            self.journal.append(&Record::Head { node: self.head })?;
        }
        // The saved state: the newest seal of the files on disk, else the
        // root; without that hash (non-phy backends) the session opens clean.
        self.saved = match self.top_artifacts {
            Some(top) => self
                .seals
                .iter()
                .rev()
                .find(|s| s.1 == top)
                .and_then(|s| s.0),
            None => self.head,
        };
        Ok(())
    }

    /// Add a node record to the tree (not applied) and make it its parent's
    /// most recent child.
    fn push_node(&mut self, rec: Record) {
        let Record::Node { id, parent, .. } = rec else {
            unreachable!("push_node takes nodes")
        };
        let key = record_hash(self.key(parent), &rec);
        let Record::Node {
            actor, ts_ms, cmd, ..
        } = rec
        else {
            unreachable!()
        };
        self.nodes.push(Node {
            parent,
            actor,
            ts_ms,
            cmd,
            key,
            inverse: None,
        });
        self.last_child.push(None);
        self.last_child[slot(parent)] = Some(id);
    }

    /// Make every node on the path to `node` its parent's most recent child.
    fn visit(&mut self, node: Option<u64>) {
        let path = self.path(node);
        let mut parent = None;
        for n in path {
            self.last_child[slot(parent)] = Some(n);
            parent = Some(n);
        }
    }

    /// Revert to the lowest common ancestor of the head and `target`, then
    /// apply forward down to `target`. Never touches the journal. On a
    /// failed apply the head stays at the last node applied.
    fn move_to(&mut self, target: Option<u64>) -> Result<()>
    where
        P: ApplyPhyLabel,
    {
        let from = self.path(self.head);
        let to = self.path(target);
        let common = from.iter().zip(&to).take_while(|(a, b)| a == b).count();
        for &n in from[common..].iter().rev() {
            let node = &mut self.nodes[n as usize];
            let inv = node
                .inverse
                .take()
                .expect("nodes on the head path hold inverses");
            self.head = node.parent;
            self.apply_inverse(inv);
        }
        for &n in &to[common..] {
            // Taken out for the apply, so a large payload is never cloned.
            let cmd = std::mem::replace(&mut self.nodes[n as usize].cmd, CurationCommand::Undo);
            let applied = self.apply_forward(&cmd);
            let node = &mut self.nodes[n as usize];
            node.cmd = cmd;
            node.inverse = Some(applied?);
            self.head = Some(n);
        }
        self.visit(target);
        Ok(())
    }

    /// Check out `target` (`None` = the root) as a journaled head move.
    /// Refused on a remote guest: only `Undo`/`Redo` are forwarded, so a jump
    /// would leave the host's tree behind.
    pub fn checkout(&mut self, target: Option<u64>) -> Result<()>
    where
        P: ApplyPhyLabel,
    {
        if self.record_sink.is_some() {
            bail!("history jumps aren't available on a remote guest yet");
        }
        self.move_head(target, None)
    }

    /// [`Self::checkout`]; `forwarded` (the `Undo`/`Redo` that asked for the
    /// move) goes to the record sink.
    fn move_head(&mut self, target: Option<u64>, forwarded: Option<&CurationCommand>) -> Result<()>
    where
        P: ApplyPhyLabel,
    {
        if target.is_some_and(|n| n >= self.nodes.len() as u64) {
            bail!("no node {target:?} in the action tree");
        }
        if target == self.head {
            return Ok(());
        }
        let prev = self.version();
        let n_before = self.cluster_index.n_spikes() as u64;
        let affected = (0..self.n_clusters())
            .map(|c| self.cache_identity(ClusterId(c)))
            .collect();
        self.journal.append(&Record::Head { node: target })?;
        if let (Some(sink), Some(cmd)) = (&mut self.record_sink, forwarded) {
            sink(cmd);
        }
        let moved = self.move_to(target);
        let moved = moved.map_err(|e| self.fix_journal_head(e));
        self.record_cache_invalidation(prev, self.version(), n_before, affected);
        moved
    }

    /// After a failed apply, journal where the head really is, so replay does
    /// not end on the failed node. Returns `e`; if the fixup cannot be written
    /// either, that is added to it (replay copes: see [`Self::replay_journal`]).
    fn fix_journal_head(&mut self, e: anyhow::Error) -> anyhow::Error {
        match self.journal.append(&Record::Head { node: self.head }) {
            Ok(()) => e,
            Err(j) => anyhow::anyhow!("{e:#}; and the journal head could not be corrected: {j:#}"),
        }
    }

    /// Durably journal then apply, as a human edit (see [`Self::dispatch_as`]).
    pub fn dispatch(&mut self, cmd: CurationCommand) -> Result<()>
    where
        P: ApplyPhyLabel,
    {
        self.dispatch_as(Actor::Human, cmd)
    }

    /// Durably journal then apply. The fsync happens *before* the in-memory
    /// state changes, satisfying the crash-safety contract.
    ///
    /// A forward command becomes a new child of the head (a new branch when
    /// the head already has children). `Undo` checks out the parent and
    /// `Redo` the most recently visited child; either is a no-op (nothing
    /// journaled) at the end of its walk.
    pub fn dispatch_as(&mut self, actor: Actor, cmd: CurationCommand) -> Result<()>
    where
        P: ApplyPhyLabel,
    {
        match cmd {
            CurationCommand::Undo => {
                let Some(h) = self.head else { return Ok(()) };
                let parent = self.nodes[h as usize].parent;
                return self.move_head(parent, Some(&cmd));
            }
            CurationCommand::Redo => {
                let Some(c) = self.last_child[slot(self.head)] else {
                    return Ok(());
                };
                return self.move_head(Some(c), Some(&cmd));
            }
            _ => {}
        }
        // Fail closed before anything reaches the journal.
        self.check_gate(&cmd)?;
        let prev = self.version();
        let affected = self.cache_identities_for_command(&cmd);
        let n_before = self.cluster_index.n_spikes() as u64;
        let id = self.nodes.len() as u64;
        let rec = Record::Node {
            id,
            parent: self.head,
            actor,
            ts_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_millis() as u64),
            cmd,
        };
        self.journal.append(&rec)?;
        if let (Some(sink), Record::Node { cmd, .. }) = (&mut self.record_sink, &rec) {
            sink(cmd);
        }
        let redo_target = self.last_child[slot(self.head)];
        self.push_node(rec);
        let cmd = std::mem::replace(&mut self.nodes[id as usize].cmd, CurationCommand::Undo);
        let applied = self.apply_forward(&cmd);
        self.nodes[id as usize].cmd = cmd;
        match applied {
            Ok(inv) => self.nodes[id as usize].inverse = Some(inv),
            Err(e) => {
                // The node stays in the tree, off the head's path, and is
                // never where Redo goes.
                self.last_child[slot(self.head)] = redo_target;
                return Err(self.fix_journal_head(e));
            }
        }
        self.head = Some(id);
        self.record_cache_invalidation(prev, self.version(), n_before, affected);
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
        #[cfg(test)]
        ensure!(!self.fail_apply, "injected apply failure");
        match cmd {
            CurationCommand::AddSpikes(batch) => {
                self.check_add_spikes_gate(cmd)?;
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

    /// Fail closed before a live command reaches the journal: cluster ids and
    /// split indices must be valid ([`Self::check_ids`]) and AddSpikes must
    /// pass its provider gate.
    fn check_gate(&self, cmd: &CurationCommand) -> Result<()> {
        self.check_ids(cmd)?;
        self.check_add_spikes_gate(cmd)
    }

    /// AddSpikes is only valid on a phy-directory provider with no remote
    /// record sink (a sink would have to carry the payload).
    fn check_add_spikes_gate(&self, cmd: &CurationCommand) -> Result<()> {
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
            CurationCommand::Batch { children } => children
                .iter()
                .try_for_each(|c| self.check_add_spikes_gate(c)),
            _ => Ok(()),
        }
    }

    /// Refuse cluster ids at or past `limit` and splits that are not a proper
    /// subset of their cluster's spikes, for every caller (GUI, MCP, guests,
    /// imports) before anything is journaled. Replay does not run this, so old
    /// journals with such no-op records still open.
    fn check_ids(&self, cmd: &CurationCommand) -> Result<()> {
        let mut st = IdCheck {
            limit: self.n_clusters() as usize,
            sizes: HashMap::new(),
        };
        self.check_ids_in(cmd, &mut st)
    }

    /// [`Self::check_ids`] walking a batch in order: `st` tracks the ids
    /// earlier splits allocated and every cluster's size after the earlier
    /// children, so a split of a batch-allocated child (or a second split of
    /// one cluster) is checked against its real membership.
    fn check_ids_in(&self, cmd: &CurationCommand, st: &mut IdCheck) -> Result<()> {
        let id = |c: &ClusterId, st: &IdCheck| {
            if c.idx() < st.limit {
                Ok(())
            } else {
                Err(anyhow::anyhow!("cluster {} does not exist", c.0))
            }
        };
        let size = |c: ClusterId, st: &IdCheck| {
            st.sizes.get(&c).copied().unwrap_or_else(|| {
                if c.0 < self.n_clusters() {
                    self.spike_times(c).len()
                } else {
                    0
                }
            })
        };
        match cmd {
            CurationCommand::Relabel { cluster, .. }
            | CurationCommand::Note { cluster, .. }
            | CurationCommand::Tag { cluster, .. } => id(cluster, st),
            CurationCommand::Merge { sources, target } => {
                sources.iter().chain([target]).try_for_each(|c| id(c, st))?;
                let moved: usize = sources
                    .iter()
                    .filter(|s| *s != target)
                    .map(|&s| size(s, st))
                    .sum();
                let grown = size(*target, st) + moved;
                for &s in sources.iter().filter(|s| *s != target) {
                    st.sizes.insert(s, 0);
                }
                st.sizes.insert(*target, grown);
                Ok(())
            }
            CurationCommand::Split {
                cluster, spike_idx, ..
            } => {
                id(cluster, st)?;
                let n = size(*cluster, st);
                let mut idx = spike_idx.clone();
                idx.sort_unstable();
                idx.dedup();
                if idx.is_empty() || idx.len() >= n || idx.last().is_some_and(|&i| i as usize >= n)
                {
                    bail!(
                        "a split must move some but not all of cluster {}'s {n} spikes",
                        cluster.0
                    );
                }
                // The index allocates the child at the tail.
                st.sizes.insert(*cluster, n - idx.len());
                st.sizes.insert(ClusterId(st.limit as u32), idx.len());
                st.limit += 1;
                Ok(())
            }
            CurationCommand::Batch { children } => {
                children.iter().try_for_each(|c| self.check_ids_in(c, st))
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

    /// Bad ids and improper splits are refused before the journal for every
    /// caller; a batch may still address the id its own split allocates.
    #[test]
    fn dispatch_refuses_unknown_clusters_and_improper_splits() {
        let (mut s, _d) = fresh_session();
        let split = |c: u32, idx: Vec<u32>| CurationCommand::Split {
            cluster: ClusterId(c),
            spike_idx: idx,
            new_cluster: ClusterId(0),
            by: Method::Manual,
        };
        let relabel = |c: u32| CurationCommand::Relabel {
            cluster: ClusterId(c),
            op: PhyLabelOp::SetGood,
        };
        let before = s.version();
        for bad in [
            split(9, vec![0]),
            split(0, vec![]),
            split(0, vec![0, 1, 2]),
            split(0, vec![7]),
            relabel(9),
            CurationCommand::Merge {
                sources: vec![ClusterId(9)],
                target: ClusterId(0),
            },
            // Id 3 only exists after the split that comes later in the batch.
            CurationCommand::Batch {
                children: vec![relabel(3), split(0, vec![0])],
            },
            // The batch-allocated child 3 holds one spike: moving it all.
            CurationCommand::Batch {
                children: vec![split(0, vec![0]), split(3, vec![0])],
            },
            // Cluster 0 keeps two spikes after the first split: moving both.
            CurationCommand::Batch {
                children: vec![split(0, vec![0]), split(0, vec![0, 1])],
            },
        ] {
            assert!(s.dispatch(bad.clone()).is_err(), "{bad:?} was accepted");
        }
        assert_eq!(s.version(), before, "nothing reached the journal");
        // Cluster 3 is the id the batch's own split allocates.
        s.dispatch(CurationCommand::Batch {
            children: vec![split(0, vec![0]), relabel(3)],
        })
        .unwrap();
        assert_eq!(s.spike_times(ClusterId(3)).len(), 1);
        // A proper split of a batch-allocated child is fine: child 3 takes two
        // of cluster 0's three spikes, then gives one of them to child 4.
        let (mut s, _d2) = fresh_session();
        s.dispatch(CurationCommand::Batch {
            children: vec![split(0, vec![0, 1]), split(3, vec![0])],
        })
        .unwrap();
        assert_eq!(s.spike_times(ClusterId(3)).len(), 1);
        assert_eq!(s.spike_times(ClusterId(4)).len(), 1);
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
        drop(s); // one writer per journal
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
    fn forward_op_after_undo_starts_a_leaf() {
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
        assert_eq!(s.redo_len(), 0, "the new node has no children");
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

    fn three_unit_provider() -> MockProvider {
        MockProvider {
            spikes: vec![
                vec![SampleIndex(10), SampleIndex(30), SampleIndex(150)],
                vec![SampleIndex(20), SampleIndex(200)],
                vec![SampleIndex(100)],
            ],
            labels: vec![MockLabel::Unsorted; 3],
            trace: vec![],
            n_channels: 1,
        }
    }

    fn reopen(dir: &Path) -> Session<MockProvider> {
        let journal = Journal::open(&dir.join("j.sqlite")).unwrap();
        let mut s = Session::new(three_unit_provider(), journal);
        s.replay_journal().unwrap();
        s
    }

    fn relabel(c: u32, op: PhyLabelOp) -> CurationCommand {
        CurationCommand::Relabel {
            cluster: ClusterId(c),
            op,
        }
    }

    /// What a node's state looks like from outside: labels, bucket sizes.
    fn shape(s: &Session<MockProvider>) -> (Vec<MockLabel>, Vec<usize>) {
        let sizes = (0..s.n_clusters())
            .map(|c| s.spike_times(ClusterId(c)).len())
            .collect();
        (s.labels().to_vec(), sizes)
    }

    #[test]
    fn replay_journal_reconstructs_state_and_redo_stack() {
        let (mut s, dir) = fresh_session();
        s.dispatch(relabel(0, PhyLabelOp::SetMua)).unwrap();
        s.dispatch(relabel(1, PhyLabelOp::SetNoise)).unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        drop(s);

        let session = reopen(dir.path());
        assert_eq!(session.label(ClusterId(0)), Some(MockLabel::Mua));
        assert_eq!(session.label(ClusterId(1)), Some(MockLabel::Unsorted));
        assert_eq!(session.label(ClusterId(2)), Some(MockLabel::Unsorted));
        assert_eq!(session.history_len(), 1);
        assert_eq!(session.redo_len(), 1);
    }

    #[test]
    fn replay_journal_rejects_impossible_records() {
        for bad in [
            Record::Head { node: Some(5) },
            Record::Node {
                id: 0,
                parent: Some(3),
                actor: Actor::Human,
                ts_ms: 0,
                cmd: relabel(0, PhyLabelOp::SetGood),
            },
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("j.sqlite");
            Journal::open(&path).unwrap().append(&bad).unwrap();
            let journal = Journal::open(&path).unwrap();
            let mut session = Session::new(three_unit_provider(), journal);
            assert!(session.replay_journal().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn replay_journal_reconstructs_merged_buckets() {
        let dir = tempfile::tempdir().unwrap();
        Journal::open(&dir.path().join("j.sqlite"))
            .unwrap()
            .append(&Record::Node {
                id: 0,
                parent: None,
                actor: Actor::Human,
                ts_ms: 0,
                cmd: CurationCommand::Merge {
                    sources: vec![ClusterId(0), ClusterId(1)],
                    target: ClusterId(2),
                },
            })
            .unwrap();
        let session = reopen(dir.path());
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

    /// Undo never discards: an edit after an undo is a sibling branch, and
    /// checking out either branch rebuilds its state, including split ids
    /// allocated on both.
    #[test]
    fn edit_after_undo_branches_and_checkout_crosses_branches() {
        let (mut s, _d) = fresh_session();
        let root = shape(&s);
        s.dispatch(CurationCommand::Merge {
            sources: vec![ClusterId(0)],
            target: ClusterId(1),
        })
        .unwrap();
        s.dispatch(CurationCommand::Split {
            cluster: ClusterId(1),
            spike_idx: vec![0],
            new_cluster: ClusterId(u32::MAX),
            by: Method::Manual,
        })
        .unwrap();
        let (a, a_shape, a_version) = (s.head(), shape(&s), s.version());
        s.dispatch(CurationCommand::Undo).unwrap();
        s.dispatch(CurationCommand::Split {
            cluster: ClusterId(1),
            spike_idx: vec![0, 1],
            new_cluster: ClusterId(u32::MAX),
            by: Method::Manual,
        })
        .unwrap();
        let (b, b_shape) = (s.head(), shape(&s));
        assert_eq!(s.n_nodes(), 3, "the undone split was kept");
        let parent = |n: Option<u64>| s.node(n.unwrap()).unwrap().0;
        assert_eq!(parent(a), parent(b), "siblings");
        assert_ne!(a_shape, b_shape);
        assert_ne!(s.version(), a_version);

        s.checkout(a).unwrap();
        assert_eq!((shape(&s), s.version()), (a_shape.clone(), a_version));
        s.checkout(None).unwrap();
        assert_eq!(shape(&s), root);
        // Redo follows the most recently visited branch: a.
        s.dispatch(CurationCommand::Redo).unwrap();
        s.dispatch(CurationCommand::Redo).unwrap();
        assert_eq!(s.head(), a);
        s.checkout(b).unwrap();
        assert_eq!(shape(&s), b_shape);
        assert!(s.checkout(Some(9)).is_err());
    }

    /// A remote guest forwards only Undo/Redo, so a node-addressed jump is
    /// refused (it would diverge from the host) and changes nothing.
    #[test]
    fn checkout_is_refused_on_a_remote_guest() {
        let (mut s, _dir) = fresh_session();
        s.dispatch(relabel(0, PhyLabelOp::SetGood)).unwrap();
        s.set_record_sink(|_| {});
        let (head, n_records) = (s.head(), s.journal().n_records());
        let e = s.checkout(None).unwrap_err();
        assert!(e.to_string().contains("remote guest"), "{e}");
        assert_eq!((s.head(), s.journal().n_records()), (head, n_records));
        s.dispatch(CurationCommand::Undo).unwrap(); // still forwarded
        assert_eq!(s.head(), None);
    }

    /// The tree, the head and the redo branch come back on reopen, with the
    /// same state and version.
    #[test]
    fn tree_and_head_survive_reopen() {
        let (mut s, dir) = fresh_session();
        s.dispatch(relabel(0, PhyLabelOp::SetGood)).unwrap();
        s.dispatch(relabel(1, PhyLabelOp::SetMua)).unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        s.dispatch(relabel(2, PhyLabelOp::SetNoise)).unwrap();
        s.checkout(Some(1)).unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        let want = (
            shape(&s),
            s.head(),
            s.version(),
            s.history_len(),
            s.redo_commands().count(),
        );
        drop(s);
        let mut r = reopen(dir.path());
        let got = (
            shape(&r),
            r.head(),
            r.version(),
            r.history_len(),
            r.redo_commands().count(),
        );
        assert_eq!(got, want);
        assert_eq!(r.n_nodes(), 3);
        // Redo goes back down the branch visited last (node 1), not node 2.
        r.dispatch(CurationCommand::Redo).unwrap();
        assert_eq!(r.head(), Some(1));
        assert_eq!(r.label(ClusterId(1)), Some(MockLabel::Mua));
        assert_eq!(r.label(ClusterId(2)), Some(MockLabel::Unsorted));
    }

    #[test]
    fn version_changes_on_every_head_move() {
        let (mut s, _d) = fresh_session();
        let v0 = s.version();
        s.dispatch(relabel(0, PhyLabelOp::SetGood)).unwrap();
        let v1 = s.version();
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.version(), v0, "back at the root");
        s.dispatch(relabel(0, PhyLabelOp::SetGood)).unwrap();
        let v2 = s.version();
        assert!(v1 != v0 && v2 != v1 && v2 != v0, "same edit, new node");
    }

    #[test]
    fn saved_tracks_the_head() {
        let (mut s, _d) = fresh_session();
        assert!(s.is_saved());
        s.dispatch(relabel(0, PhyLabelOp::SetGood)).unwrap();
        assert!(!s.is_saved());
        s.mark_saved();
        s.dispatch(CurationCommand::Undo).unwrap();
        assert!(!s.is_saved(), "undo after a save is unsaved");
        s.dispatch(CurationCommand::Redo).unwrap();
        assert!(s.is_saved());
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

    /// A failed apply is never the redo target, and when the journal cannot
    /// take the head fixup either, the error says so and later edits still
    /// replay (the journal's tail names the failed node as head).
    #[test]
    fn failed_apply_is_not_a_redo_target_and_propagates_the_fixup_error() {
        let (mut s, dir) = fresh_session();
        s.dispatch(relabel(0, PhyLabelOp::SetMua)).unwrap();
        s.dispatch(CurationCommand::Undo).unwrap();
        assert_eq!(s.redo_len(), 1);
        s.dispatch(CurationCommand::Redo).unwrap();
        // Fails, and its fixup is journaled: Redo must not enter it.
        s.fail_apply = true;
        assert!(s.dispatch(relabel(1, PhyLabelOp::SetNoise)).is_err());
        s.fail_apply = false;
        assert_eq!((s.history_len(), s.redo_len()), (1, 0));
        s.dispatch(CurationCommand::Redo).unwrap();
        assert_eq!(s.label(ClusterId(1)), Some(MockLabel::Unsorted));

        // Same failure with a journal that rejects the fixup append.
        s.fail_apply = true;
        s.journal.fail_after = Some(1);
        let err = s.dispatch(relabel(1, PhyLabelOp::SetNoise)).unwrap_err();
        assert!(format!("{err:#}").contains("journal head could not be corrected"));
        s.fail_apply = false;
        // The next edit's parent is the old head, not the failed node.
        s.dispatch(relabel(2, PhyLabelOp::SetGood)).unwrap();
        drop(s);
        let s = reopen(dir.path());
        assert_eq!(s.label(ClusterId(2)), Some(MockLabel::Good));
        assert_eq!(s.label(ClusterId(1)), Some(MockLabel::Unsorted));
        assert_eq!(s.label(ClusterId(0)), Some(MockLabel::Mua));
    }

    /// A crash between a failed node and its head fixup leaves the failed
    /// node as the journal's head: reopening drops it (and journals the
    /// fixup) instead of failing, and Redo does not walk into it.
    #[test]
    fn reopen_after_a_crash_before_the_head_fixup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.sqlite");
        let node = |id, parent, cmd| Record::Node {
            id,
            parent,
            actor: Actor::Human,
            ts_ms: 0,
            cmd,
        };
        {
            let mut j = Journal::open(&path).unwrap();
            j.append(&node(0, None, relabel(0, PhyLabelOp::SetMua)))
                .unwrap();
            // AddSpikes cannot apply on a provider without a phy directory.
            let bad = CurationCommand::AddSpikes(Default::default());
            j.append(&node(1, Some(0), bad)).unwrap();
        }
        let mut s = reopen(dir.path());
        assert_eq!(s.label(ClusterId(0)), Some(MockLabel::Mua));
        assert_eq!((s.history_len(), s.redo_len()), (1, 0));
        s.dispatch(CurationCommand::Redo).unwrap(); // nothing to redo
                                                    // The fixup is durable: a second reopen needs no repair.
        drop(s);
        let j = Journal::open(&path).unwrap();
        assert!(matches!(
            j.replay().unwrap().last(),
            Some(Record::Head { node: Some(0) })
        ));
        drop(j);
        assert_eq!(reopen(dir.path()).history_len(), 1);
    }
}

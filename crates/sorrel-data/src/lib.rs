//! sorrel-data: the monomorphised core. `Session<P>` is generic over a
//! `DataProvider`, so the entire state machine specialises per backend.

pub mod cache;
pub mod cluster_index;
pub mod command;
pub mod curation_json;
pub mod diff;
pub mod feature_subspace;
pub mod journal;
pub mod open;
pub mod preview;
pub mod qc_export;
pub mod quality_ext;
pub mod report;
pub mod rules;
pub mod save;
pub mod serve;
pub mod session;
pub mod spike_depth;
pub mod suggest;

pub use cache::correlograms::{acg_or_compute, ccg_or_compute};
pub use cluster_index::{ClusterIndex, MergeRecord, SplitRecord};
pub use command::{CurationCommand, PhyLabelOp};
pub use curation_json::write_curation_json;
pub use diff::{diff_labels, LabelDiff};
pub use feature_subspace::{
    collect_pc_subspace, PcSubspace, DEFAULT_CHANNEL_IDX, DEFAULT_D, DEFAULT_MAX_BACKGROUND,
};
pub use journal::{baseline_hash, Journal};
pub use open::{
    journal_path, open_identity_sealed, open_journal, open_phy, open_sorting_analyzer, seed_phy,
    JournalOpts,
};
pub use preview::{preview_merge, MergeDelta, MergePreview};
pub use qc_export::{collect_qc_rows, export_qc, write_qc_json, write_qc_tsv, QcRow};
pub use quality_ext::{cluster_quality, compute_isolation};
pub use report::{render_cluster_svgs, render_report, write_report, ClusterSvgs, ReportFilter};
pub use rules::{apply_rules, Rule, RuleSet};
pub use save::{save_and_reseal_journal, save_to_phy};
pub use serve::{serve, tool_list, PROTOCOL_VERSION};
pub use session::Session;
pub use spike_depth::{pc_weighted_depth, spike_depths, DepthSource, SpikeDepths};
pub use suggest::{
    amplitude_iqr_overlap, rank_merge_candidates, rank_split_candidates, MergeCandidate,
    SplitCandidate, SuggestConfig,
};

//! sorrel-data: the monomorphised core. `Session<P>` is generic over a
//! `DataProvider`, so the entire state machine specialises per backend.

pub mod brief;
pub mod cache;
pub mod cluster_index;
pub mod command;
pub mod curation_json;
pub mod diff;
pub mod evaluate;
pub mod feature_subspace;
pub mod host;
pub mod instance;
pub mod journal;
pub mod open;
#[cfg(feature = "phy-export")]
pub mod phy_export;
pub mod preview;
pub mod probe;
pub mod proposals;
pub mod proposers;
pub mod qc_export;
pub mod quality_ext;
pub mod report;
pub mod rules;
pub mod save;
pub mod serve;
pub mod session;
pub mod spike_depth;
pub mod suggest;

pub use brief::{quantise, render_brief, BriefOpts, Flag, FLAGS};
pub use cache::correlograms::{acg_or_compute, ccg_or_compute};
pub use cluster_index::{ClusterIndex, MergeRecord, SplitRecord};
pub use command::{describe_command, Actor, CurationCommand, Method, PhyLabelOp, UnitOrigin};
pub use curation_json::write_curation_json;
pub use diff::{diff_labels, LabelDiff};
pub use evaluate::{eval_merge, eval_split, Handles, MergeEval, Relation, SplitEval};
pub use feature_subspace::{
    collect_pc_subspace, PcSubspace, DEFAULT_CHANNEL_IDX, DEFAULT_D, DEFAULT_MAX_BACKGROUND,
};
pub use host::{AgentPolicy, AgentSeat, HostEvent, SessionHost};
pub use instance::{send_instance, serve_instance, InstanceHandle, InstanceRequest, Pending};
pub use journal::{artifacts_hash, baseline_hash, Journal, Record};
pub use open::{
    journal_path, open_identity_sealed, open_journal, open_phy, open_sorting_analyzer,
    retire_snapshot, seed_phy, JournalOpts,
};
#[cfg(feature = "phy-export")]
pub use phy_export::export_phy;
pub use preview::{preview_merge, MergeDelta, MergePreview};
pub use proposers::{
    attention_all, attention_with, kmeans_split, kmeans_split_proposal, outlier_proposal,
    outlier_split, KmeansSplit,
};
pub use qc_export::{
    collect_qc_rows, collect_qc_rows_with, export_qc, write_qc_json, write_qc_tsv, QcRow,
};
pub use quality_ext::{cluster_quality, compute_isolation};
pub use report::{render_cluster_svgs, render_report, write_report, ClusterSvgs, ReportFilter};
pub use rules::{apply_rules, Rule, RuleSet};
pub use save::{save_and_seal, save_session, save_to_phy};
pub use serve::{
    actor_name, attach, call_once, handle_line, list_hosts, read_mcp_json, serve, spawn_listener,
    tool_list, user_dir, Env, HostRecord, ListenerHandle, SharedSelection, Ui, PROTOCOL_VERSION,
    RUBRIC,
};
pub use session::Session;
pub use spike_depth::{pc_weighted_depth, spike_depths, DepthSource, SpikeDepths};
pub use suggest::{
    amplitude_iqr_overlap, rank_merge_candidates, rank_split_candidates, MergeCandidate,
    SplitCandidate, SuggestConfig,
};

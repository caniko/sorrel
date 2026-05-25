//! sorrel-data: the monomorphised core. `Session<P>` is generic over a
//! `DataProvider`, so the entire state machine specialises per backend.

pub mod cache;
pub mod cluster_index;
pub mod command;
pub mod feature_subspace;
pub mod journal;
pub mod preview;
pub mod qc_export;
pub mod quality_ext;
pub mod save;
pub mod session;
pub mod suggest;

pub use cache::correlograms::{acg_or_compute, ccg_or_compute};
pub use cluster_index::{ClusterIndex, MergeRecord, SplitRecord};
pub use command::{CurationCommand, PhyLabelOp};
pub use feature_subspace::{
    collect_pc_subspace, PcSubspace, DEFAULT_CHANNEL_IDX, DEFAULT_D, DEFAULT_MAX_BACKGROUND,
};
pub use journal::{baseline_hash, Journal, SqliteJournal};
pub use preview::{preview_merge, MergeDelta, MergePreview};
pub use qc_export::{collect_qc_rows, export_qc, write_qc_json, write_qc_tsv, QcRow};
pub use quality_ext::{cluster_quality, compute_isolation};
pub use save::{save_and_reseal_journal, save_to_phy};
pub use session::Session;
pub use suggest::{
    amplitude_iqr_overlap, rank_merge_candidates, rank_split_candidates, MergeCandidate,
    SplitCandidate, SuggestConfig,
};

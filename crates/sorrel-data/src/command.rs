use serde::{Deserialize, Serialize};
use sorrel_io::ClusterId;

/// The phy2 label transitions, modelled as a closed enum so the undo stack
/// stores a fixed-size tag and the journal serialises a couple of bytes.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PhyLabelOp {
    SetUnsorted = 0,
    SetGood = 1,
    SetMua = 2,
    SetNoise = 3,
}

/// Strongly-typed curation operations. Concrete enum -> no `Box<dyn Trait>`,
/// no per-op heap allocation, perfectly packed history.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CurationCommand {
    Relabel {
        cluster: ClusterId,
        op: PhyLabelOp,
    },
    Merge {
        sources: Vec<ClusterId>,
        target: ClusterId,
    },
    Split {
        cluster: ClusterId,
        /// Spike indices (into the cluster's local spike list) that move out.
        spike_idx: Vec<u32>,
        new_cluster: ClusterId,
    },
    Note {
        cluster: ClusterId,
        text: String,
    },
}

impl CurationCommand {
    pub fn primary_cluster(&self) -> ClusterId {
        match self {
            Self::Relabel { cluster, .. } => *cluster,
            Self::Merge { target, .. } => *target,
            Self::Split { cluster, .. } => *cluster,
            Self::Note { cluster, .. } => *cluster,
        }
    }
}

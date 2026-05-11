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
///
/// `Undo`/`Redo` are first-class journal records: the on-disk log captures
/// every user action, including reversions. Replay reconstructs the same
/// in-memory history/redo stacks the live session would hold.
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
    /// Group of forward operations applied as a single undoable unit. The
    /// session executes each child in order; one `Undo` reverts the entire
    /// batch. Children must themselves be forward ops (no nested
    /// `Undo`/`Redo`/`Batch`) — that's enforced at construction by
    /// [`Self::batch`] and asserted at apply time.
    Batch {
        children: Vec<CurationCommand>,
    },
    Undo,
    Redo,
}

impl CurationCommand {
    /// Forward operations have a "primary" cluster used for cursor focus.
    /// `Undo`/`Redo` don't: callers must avoid this method on those variants.
    pub fn primary_cluster(&self) -> Option<ClusterId> {
        match self {
            Self::Relabel { cluster, .. } => Some(*cluster),
            Self::Merge { target, .. } => Some(*target),
            Self::Split { cluster, .. } => Some(*cluster),
            Self::Note { cluster, .. } => Some(*cluster),
            Self::Batch { children } => children.first().and_then(|c| c.primary_cluster()),
            Self::Undo | Self::Redo => None,
        }
    }

    /// True for the curation actions that move state forward (i.e. anything
    /// that gets pushed onto the undo stack).
    pub fn is_forward(&self) -> bool {
        !matches!(self, Self::Undo | Self::Redo)
    }

    /// Wrap a list of forward operations into a single atomic batch.
    /// Returns `None` if any child is itself an `Undo`, `Redo`, or `Batch`
    /// — those would make the apply/undo logic ambiguous.
    pub fn batch(children: Vec<CurationCommand>) -> Option<Self> {
        for c in &children {
            if matches!(c, Self::Undo | Self::Redo | Self::Batch { .. }) {
                return None;
            }
        }
        Some(Self::Batch { children })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_cluster_matches_variant_field() {
        assert_eq!(
            CurationCommand::Relabel {
                cluster: ClusterId(7),
                op: PhyLabelOp::SetGood
            }
            .primary_cluster(),
            Some(ClusterId(7))
        );
        assert_eq!(
            CurationCommand::Merge {
                sources: vec![ClusterId(1), ClusterId(2)],
                target: ClusterId(9)
            }
            .primary_cluster(),
            Some(ClusterId(9))
        );
        assert_eq!(
            CurationCommand::Split {
                cluster: ClusterId(4),
                spike_idx: vec![],
                new_cluster: ClusterId(5)
            }
            .primary_cluster(),
            Some(ClusterId(4))
        );
        assert_eq!(
            CurationCommand::Note {
                cluster: ClusterId(3),
                text: "hi".into()
            }
            .primary_cluster(),
            Some(ClusterId(3))
        );
        assert_eq!(CurationCommand::Undo.primary_cluster(), None);
        assert_eq!(CurationCommand::Redo.primary_cluster(), None);
    }

    #[test]
    fn is_forward_distinguishes_curation_from_revert_variants() {
        assert!(CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: PhyLabelOp::SetGood
        }
        .is_forward());
        assert!(CurationCommand::Merge {
            sources: vec![],
            target: ClusterId(0)
        }
        .is_forward());
        assert!(!CurationCommand::Undo.is_forward());
        assert!(!CurationCommand::Redo.is_forward());
    }

    #[test]
    fn round_trips_through_messagepack() {
        let original = CurationCommand::Relabel {
            cluster: ClusterId(42),
            op: PhyLabelOp::SetMua,
        };
        let bytes = rmp_serde::to_vec(&original).unwrap();
        let decoded: CurationCommand = rmp_serde::from_slice(&bytes).unwrap();
        match decoded {
            CurationCommand::Relabel { cluster, op } => {
                assert_eq!(cluster, ClusterId(42));
                assert_eq!(op, PhyLabelOp::SetMua);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn batch_constructor_rejects_nested_or_revert_children() {
        assert!(CurationCommand::batch(vec![CurationCommand::Undo]).is_none());
        assert!(CurationCommand::batch(vec![CurationCommand::Redo]).is_none());
        assert!(
            CurationCommand::batch(vec![CurationCommand::Batch { children: vec![] }]).is_none()
        );

        let ok = CurationCommand::batch(vec![
            CurationCommand::Relabel {
                cluster: ClusterId(1),
                op: PhyLabelOp::SetGood,
            },
            CurationCommand::Merge {
                sources: vec![ClusterId(2)],
                target: ClusterId(3),
            },
        ]);
        assert!(ok.is_some());
    }

    #[test]
    fn batch_round_trips_through_messagepack() {
        let original = CurationCommand::batch(vec![
            CurationCommand::Relabel {
                cluster: ClusterId(1),
                op: PhyLabelOp::SetGood,
            },
            CurationCommand::Merge {
                sources: vec![ClusterId(2), ClusterId(3)],
                target: ClusterId(4),
            },
        ])
        .unwrap();
        let bytes = rmp_serde::to_vec(&original).unwrap();
        let decoded: CurationCommand = rmp_serde::from_slice(&bytes).unwrap();
        match decoded {
            CurationCommand::Batch { children } => {
                assert_eq!(children.len(), 2);
                assert!(matches!(children[0], CurationCommand::Relabel { .. }));
                assert!(matches!(children[1], CurationCommand::Merge { .. }));
            }
            _ => panic!("wrong variant"),
        }
    }
}

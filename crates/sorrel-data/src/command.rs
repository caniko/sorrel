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

/// How a unit created inside sorrel was made. Closed: exactly three.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Method {
    Manual,
    Algo(String),
    Llm(String),
}

impl Method {
    /// Method assumed for journal records written before `Split.by` existed.
    // ponytail: legacy splits made by `sorrel serve` also decode as Manual;
    // that path is days old, so no per-record migration.
    pub fn legacy() -> Self {
        Self::Manual
    }
}

impl std::fmt::Display for Method {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Manual => f.write_str("Manual"),
            Self::Algo(n) => write!(f, "Algo:{n}"),
            Self::Llm(n) => write!(f, "LLM:{n}"),
        }
    }
}

impl std::str::FromStr for Method {
    type Err = String;
    /// `manual`, `algo:<name>`, `llm:<name>`; keywords are case-insensitive,
    /// the name keeps its case.
    fn from_str(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("manual") {
            return Ok(Self::Manual);
        }
        let (kind, name) = s.split_once(':').ok_or_else(|| {
            format!("unknown method {s:?}; want manual, algo:<name> or llm:<name>")
        })?;
        let name = name.trim();
        if name.is_empty() {
            return Err(format!("method {s:?} needs a name after ':'"));
        }
        match kind.trim().to_ascii_lowercase().as_str() {
            "algo" => Ok(Self::Algo(name.to_string())),
            "llm" => Ok(Self::Llm(name.to_string())),
            _ => Err(format!("unknown method kind {kind:?}; want algo or llm")),
        }
    }
}

/// Who made an edit in the action tree. Journaled with every node; closed,
/// so a new kind needs a journal version bump.
#[derive(Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Actor {
    #[default]
    Human,
    Agent {
        client: String,
    },
    Algo {
        name: String,
    },
    Remote {
        peer: String,
    },
}

/// Where a unit came from: straight out of the sorter, or created in sorrel.
/// Exclusive; there is no "unknown" state.
#[derive(Clone, Debug, Default, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum UnitOrigin {
    #[default]
    Auto,
    Sorrel(Method),
}

impl UnitOrigin {
    /// `Auto[<sorter>]` or `Sorrel[Manual|Algo:x|LLM:x]`.
    pub fn label(&self, sorter: &str) -> String {
        match self {
            Self::Auto => format!("Auto[{sorter}]"),
            Self::Sorrel(m) => format!("Sorrel[{m}]"),
        }
    }

    /// The one parser for origin text (TSV, sorrel-py `by=`, MCP). Accepts
    /// `Auto`, `Auto[<sorter>]`, `Sorrel[<method>]` and a bare `<method>`,
    /// case-insensitive on keywords. The second value is the sorter name an
    /// `Auto[x]` pins.
    pub fn parse(s: &str) -> Result<(Self, Option<String>), String> {
        let s = s.trim();
        let lower = s.to_ascii_lowercase();
        let inner = |prefix: &str| {
            s[prefix.len()..]
                .strip_prefix('[')
                .and_then(|r| r.strip_suffix(']'))
                .map(str::trim)
        };
        if lower == "auto" {
            Ok((Self::Auto, None))
        } else if lower.starts_with("auto[") {
            let sorter = inner("auto").filter(|n| !n.is_empty());
            let sorter = sorter.ok_or_else(|| format!("bad origin {s:?}"))?;
            Ok((Self::Auto, Some(sorter.to_string())))
        } else if lower.starts_with("sorrel[") {
            let m = inner("sorrel").ok_or_else(|| format!("bad origin {s:?}"))?;
            Ok((Self::Sorrel(m.parse()?), None))
        } else {
            Ok((Self::Sorrel(s.parse()?), None))
        }
    }
}

impl std::fmt::Display for UnitOrigin {
    /// Without a sorter name: `Auto` or `Sorrel[..]`. Use [`UnitOrigin::label`]
    /// to render `Auto[<sorter>]`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("Auto"),
            Self::Sorrel(_) => f.write_str(&self.label("")),
        }
    }
}

impl std::str::FromStr for UnitOrigin {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        Self::parse(s).map(|(o, _)| o)
    }
}

/// Strongly-typed curation operations. Concrete enum -> no `Box<dyn Trait>`,
/// no per-op heap allocation, perfectly packed history.
///
/// `Undo`/`Redo` move the head of the action tree (parent / most recently
/// visited child); the journal records them as head moves, never as nodes.
/// Pre-v3 journals stored them as records, which the v3 upgrade folds.
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
        /// Who made the split; the new unit's origin is `Sorrel(by)`. Trailing
        /// with a serde default: rmp-serde writes structs as positional arrays
        /// and a missing trailing element takes the default, so v1 records
        /// still decode.
        #[serde(default = "Method::legacy")]
        by: Method,
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
    /// Replace the cluster's tag set (provenance and review markers such as
    /// `src:unitrefine` or `review:merge`). Appended last so older journals
    /// keep decoding (rmp-serde encodes variant names, not indices).
    Tag {
        cluster: ClusterId,
        tags: Vec<String>,
    },
    /// Insert detected spikes into existing clusters (rescue). Everything is
    /// precomputed so replay never re-detects. Appended after `Tag`.
    AddSpikes(SpikeBatch),
}

/// Payload of [`CurationCommand::AddSpikes`]: `n = times.len()` new spikes.
/// `times` is non-decreasing; a new spike sorts after originals with an
/// equal time. Per-spike vectors are parallel to `times`; the feature
/// vectors are row-major and must be empty exactly when the session has not
/// loaded that feature (otherwise the apply fails closed).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SpikeBatch {
    pub times: Vec<u64>,
    pub clusters: Vec<ClusterId>,
    pub templates: Vec<u32>,
    pub amplitudes: Vec<f32>,
    /// `n * (n_pcs * n_channels_per_template)` or empty.
    pub pc_features: Vec<f32>,
    /// `n * k` (template features) or empty.
    pub template_features: Vec<f32>,
    /// `(name, columns, n * columns values)`, one per loaded `spike_<name>`
    /// attribute, in the session's order.
    pub attributes: Vec<(String, u32, Vec<f32>)>,
}

impl CurationCommand {
    /// Forward operations have a "primary" cluster used for cursor focus.
    /// `Undo`/`Redo` don't: callers must avoid this method on those variants.
    pub fn primary_cluster(&self) -> Option<ClusterId> {
        match self {
            Self::Relabel { cluster, .. } => Some(*cluster),
            Self::Merge { target, .. } => Some(*target),
            Self::Split { cluster, .. } => Some(*cluster),
            Self::Note { cluster, .. } | Self::Tag { cluster, .. } => Some(*cluster),
            Self::AddSpikes(b) => b.clusters.first().copied(),
            Self::Batch { children } => children.first().and_then(|c| c.primary_cluster()),
            Self::Undo | Self::Redo => None,
        }
    }

    /// Every cluster the command names (for a split, the source only: the
    /// new id is allocated at apply time), sorted and deduplicated.
    pub fn clusters(&self) -> Vec<ClusterId> {
        let mut out = match self {
            Self::Relabel { cluster, .. }
            | Self::Split { cluster, .. }
            | Self::Note { cluster, .. }
            | Self::Tag { cluster, .. } => vec![*cluster],
            Self::Merge { sources, target } => [&[*target][..], sources].concat(),
            Self::AddSpikes(b) => b.clusters.clone(),
            Self::Batch { children } => children.iter().flat_map(Self::clusters).collect(),
            Self::Undo | Self::Redo => Vec::new(),
        };
        out.sort_unstable();
        out.dedup();
        out
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

fn label_name(op: PhyLabelOp) -> &'static str {
    match op {
        PhyLabelOp::SetGood => "Good",
        PhyLabelOp::SetMua => "MUA",
        PhyLabelOp::SetNoise => "Noise",
        PhyLabelOp::SetUnsorted => "Unsorted",
    }
}

/// Short human description of a curation command, shared by GUI toasts,
/// MCP replies and host events, so undo/redo say *what* they reverted
/// instead of a bare "undone".
pub fn describe_command(cmd: &CurationCommand) -> String {
    match cmd {
        CurationCommand::Relabel { cluster, op } => {
            format!("relabel cluster {} as {}", cluster.0, label_name(*op))
        }
        CurationCommand::Merge { sources, target } => {
            format!("merge {} cluster(s) into {}", sources.len(), target.0)
        }
        CurationCommand::Split {
            cluster, spike_idx, ..
        } => {
            format!("split cluster {} ({} spikes)", cluster.0, spike_idx.len())
        }
        CurationCommand::Note { cluster, text } if text.is_empty() => {
            format!("clear note on cluster {}", cluster.0)
        }
        CurationCommand::Note { cluster, text } => {
            let brief: String = text.chars().take(24).collect();
            let dots = if text.chars().count() > 24 { "…" } else { "" };
            format!("note on cluster {}: {brief}{dots}", cluster.0)
        }
        CurationCommand::Tag { cluster, .. } => format!("retag cluster {}", cluster.0),
        CurationCommand::AddSpikes(b) => {
            let mut clusters = b.clusters.clone();
            clusters.sort_unstable();
            clusters.dedup();
            format!(
                "add {} spikes to {} clusters",
                b.times.len(),
                clusters.len()
            )
        }
        CurationCommand::Batch { children } => {
            // A multi-select relabel is journaled as one batch so a single
            // Undo reverts it; say so instead of "batch of N ops".
            if let Some(CurationCommand::Relabel { op, .. }) = children.first() {
                let uniform = children
                    .iter()
                    .all(|c| matches!(c, CurationCommand::Relabel { op: o, .. } if o == op));
                if uniform {
                    return format!("relabel {} clusters as {}", children.len(), label_name(*op));
                }
            }
            format!("batch of {} ops", children.len())
        }
        // Undo/Redo are head moves, never nodes.
        CurationCommand::Undo => "undo".into(),
        CurationCommand::Redo => "redo".into(),
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
                new_cluster: ClusterId(5),
                by: Method::Manual
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

    /// A pre-`by` Split record (positional array of three) still decodes, with
    /// `by` defaulting to Manual; a new one round-trips its method.
    #[test]
    fn legacy_split_record_decodes_with_default_method() {
        #[derive(Serialize)]
        enum Old {
            Split {
                cluster: ClusterId,
                spike_idx: Vec<u32>,
                new_cluster: ClusterId,
            },
        }
        let bytes = rmp_serde::to_vec(&Old::Split {
            cluster: ClusterId(1),
            spike_idx: vec![2, 3],
            new_cluster: ClusterId(9),
        })
        .unwrap();
        match rmp_serde::from_slice::<CurationCommand>(&bytes).unwrap() {
            CurationCommand::Split {
                cluster,
                spike_idx,
                new_cluster,
                by,
            } => {
                assert_eq!((cluster, new_cluster), (ClusterId(1), ClusterId(9)));
                assert_eq!(spike_idx, vec![2, 3]);
                assert_eq!(by, Method::Manual);
            }
            other => panic!("wrong variant {other:?}"),
        }
        let new = CurationCommand::Split {
            cluster: ClusterId(1),
            spike_idx: vec![],
            new_cluster: ClusterId(2),
            by: Method::Llm("x".into()),
        };
        let back: CurationCommand =
            rmp_serde::from_slice(&rmp_serde::to_vec(&new).unwrap()).unwrap();
        assert!(matches!(back, CurationCommand::Split { by: Method::Llm(n), .. } if n == "x"));
    }

    #[test]
    fn origin_text_parses_and_renders() {
        let algo = Method::Algo("gmm".into());
        assert_eq!(
            UnitOrigin::Sorrel(algo.clone()).label("ks4"),
            "Sorrel[Algo:gmm]"
        );
        assert_eq!(UnitOrigin::Auto.label("ks4"), "Auto[ks4]");
        assert_eq!(
            UnitOrigin::Sorrel(Method::Llm("x".into())).to_string(),
            "Sorrel[LLM:x]"
        );
        assert_eq!("MANUAL".parse(), Ok(Method::Manual));
        assert_eq!("Algo:python".parse(), Ok(Method::Algo("python".into())));
        assert_eq!(
            "llm:claude-code".parse(),
            Ok(Method::Llm("claude-code".into()))
        );
        assert_eq!(
            UnitOrigin::parse("Auto[kilosort4]"),
            Ok((UnitOrigin::Auto, Some("kilosort4".into())))
        );
        assert_eq!("sorrel[algo:gmm]".parse(), Ok(UnitOrigin::Sorrel(algo)));
        assert_eq!(
            "Sorrel[Manual]".parse(),
            Ok(UnitOrigin::Sorrel(Method::Manual))
        );
        assert_eq!("manual".parse(), Ok(UnitOrigin::Sorrel(Method::Manual)));
        for bad in [
            "",
            "algo",
            "algo:",
            "bogus:x",
            "Auto[]",
            "Sorrel[Manual",
            "Sorrel[]",
        ] {
            assert!(UnitOrigin::parse(bad).is_err(), "{bad:?}");
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

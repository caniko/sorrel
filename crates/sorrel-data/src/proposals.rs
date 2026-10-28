//! Staged curation proposals: what an agent (over MCP `propose`) or a
//! statistical proposer suggests, kept apart from the journal until a human
//! accepts. Every proposal and every decision lands in an append-only JSONL
//! audit log, `<save_dir>/.sorrel/proposals.jsonl`.

use crate::command::{CurationCommand, Method, PhyLabelOp};
use crate::session::Session;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sorrel_io::{ClusterId, DataProvider};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Source {
    Agent,
    Kmeans,
    Gmm,
    Outlier,
    MergeRank,
}

impl Source {
    /// The provenance tag a decided proposal leaves on its clusters.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Agent => "src:agent",
            Self::Kmeans => "src:kmeans",
            Self::Gmm => "src:gmm",
            Self::Outlier => "src:outlier",
            Self::MergeRank => "src:merge-rank",
        }
    }
}

/// `Relabel.label` is `good | mua | noise | unsorted`; `Merge` folds the
/// rest into the first cluster (the GUI's "merge selection" rule).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "lowercase")]
pub enum Action {
    Relabel {
        cluster: ClusterId,
        label: String,
    },
    Merge {
        clusters: Vec<ClusterId>,
    },
    Split {
        cluster: ClusterId,
        spike_idx: Vec<u32>,
    },
    /// Keep the cluster as it is and clear it from the review queue.
    Accept {
        cluster: ClusterId,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Pending,
    Accepted,
    Rejected,
    Superseded,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    pub id: u64,
    pub source: Source,
    #[serde(flatten)]
    pub action: Action,
    pub confidence: f32,
    pub reasons: Vec<String>,
    /// Unix seconds.
    pub created: u64,
    /// The raw tool arguments the agent sent (`Null` for built-in proposers).
    pub args: Value,
    pub outcome: Outcome,
}

/// Process-unique, increasing id: microseconds since the epoch, bumped past
/// the previous id when two land in the same microsecond (about 1.8e15, so
/// still exact in a JSON double).
fn new_id() -> u64 {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_micros() as u64);
    // compare_exchange loop: fetch_update is deprecated and its rename
    // (try_update) is newer than the MSRV.
    let mut prev = LAST.load(Ordering::Relaxed);
    loop {
        let next = (prev + 1).max(now);
        match LAST.compare_exchange_weak(prev, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return next,
            Err(seen) => prev = seen,
        }
    }
}

impl Proposal {
    pub fn new(
        source: Source,
        action: Action,
        confidence: f32,
        reasons: Vec<String>,
        args: Value,
    ) -> Self {
        let id = new_id();
        Self {
            id,
            source,
            action,
            confidence,
            reasons,
            created: id / 1_000_000,
            args,
            outcome: Outcome::Pending,
        }
    }

    /// Clusters the proposal touches, target first.
    pub fn clusters(&self) -> Vec<ClusterId> {
        match &self.action {
            Action::Relabel { cluster, .. }
            | Action::Split { cluster, .. }
            | Action::Accept { cluster } => vec![*cluster],
            Action::Merge { clusters } => clusters.clone(),
        }
    }

    /// What accepting dispatches (wrap it in `CurationCommand::batch` and
    /// `Session::with_human_review`). Retags go first so a merge unions, and
    /// a split child inherits, the set: `src:<source>` and `src:human` are
    /// added and `review:*` dropped, so the record shows a human accepted a
    /// machine proposal even on clusters that had no tags before.
    pub fn to_commands<P: DataProvider>(&self, session: &Session<P>) -> Vec<CurationCommand> {
        let mut out = self.retag(session, |tags| {
            tags.retain(|t| !t.starts_with("review:"));
            for add in [self.source.tag(), "src:human"] {
                add_tag(tags, add);
            }
        });
        match &self.action {
            Action::Relabel { cluster, label } => {
                if let Some(op) = parse_label(label) {
                    out.push(CurationCommand::Relabel {
                        cluster: *cluster,
                        op,
                    });
                }
            }
            Action::Merge { clusters } => {
                if let Some((&target, rest)) = clusters.split_first() {
                    out.push(CurationCommand::Merge {
                        sources: rest.to_vec(),
                        target,
                    });
                }
            }
            Action::Split { cluster, spike_idx } => out.push(CurationCommand::Split {
                cluster: *cluster,
                spike_idx: spike_idx.clone(),
                new_cluster: ClusterId(u32::MAX),
                by: match self.source {
                    Source::Agent => Method::Llm("agent".into()),
                    s => Method::Algo(s.tag().trim_start_matches("src:").into()),
                },
            }),
            Action::Accept { .. } => {}
        }
        out
    }

    /// What rejecting dispatches: the rejection memory the rankers honour.
    /// A merge tags every involved cluster `notmerge:<other>`, a split tags
    /// the cluster `notsplit`; relabels and accepts leave nothing behind.
    pub fn reject_commands<P: DataProvider>(&self, session: &Session<P>) -> Vec<CurationCommand> {
        match &self.action {
            Action::Merge { clusters } => {
                let all = clusters.clone();
                clusters
                    .iter()
                    .filter_map(|&c| {
                        let mut tags = session.tags(c).to_vec();
                        for o in all.iter().filter(|&&o| o != c) {
                            add_tag(&mut tags, &format!("notmerge:{}", o.0));
                        }
                        (tags != session.tags(c))
                            .then_some(CurationCommand::Tag { cluster: c, tags })
                    })
                    .collect()
            }
            Action::Split { cluster, .. } => {
                let mut tags = session.tags(*cluster).to_vec();
                add_tag(&mut tags, "notsplit");
                if tags == session.tags(*cluster) {
                    return Vec::new();
                }
                vec![CurationCommand::Tag {
                    cluster: *cluster,
                    tags,
                }]
            }
            _ => Vec::new(),
        }
    }

    /// One `Tag` per touched cluster whose tag set `edit` changes.
    fn retag<P: DataProvider>(
        &self,
        session: &Session<P>,
        edit: impl Fn(&mut Vec<String>),
    ) -> Vec<CurationCommand> {
        let mut clusters = self.clusters();
        clusters.sort_unstable();
        clusters.dedup();
        clusters
            .into_iter()
            .filter_map(|c| {
                let mut tags = session.tags(c).to_vec();
                edit(&mut tags);
                (tags != session.tags(c)).then_some(CurationCommand::Tag { cluster: c, tags })
            })
            .collect()
    }
}

fn add_tag(tags: &mut Vec<String>, tag: &str) {
    if !tags.iter().any(|t| t == tag) {
        tags.push(tag.to_owned());
    }
}

/// `good | mua | noise | unsorted` as a command op.
pub fn parse_label(s: &str) -> Option<PhyLabelOp> {
    Some(match s {
        "good" => PhyLabelOp::SetGood,
        "mua" => PhyLabelOp::SetMua,
        "noise" => PhyLabelOp::SetNoise,
        "unsorted" => PhyLabelOp::SetUnsorted,
        _ => return None,
    })
}

/// One line of the audit log.
#[derive(Serialize, Deserialize)]
#[serde(tag = "entry", rename_all = "lowercase")]
enum Entry {
    Proposal(Proposal),
    Outcome { id: u64, outcome: Outcome, at: u64 },
}

pub fn log_path(save_dir: &Path) -> PathBuf {
    save_dir.join(".sorrel").join("proposals.jsonl")
}

fn append_entry(save_dir: &Path, entry: &Entry) -> std::io::Result<()> {
    let path = log_path(save_dir);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut line = serde_json::to_vec(entry)?;
    line.push(b'\n');
    // One write call per line: O_APPEND keeps concurrent lines whole.
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(&line)
}

/// Log a newly staged proposal.
pub fn append(save_dir: &Path, p: &Proposal) -> std::io::Result<()> {
    append_entry(save_dir, &Entry::Proposal(p.clone()))
}

/// Log a decision. The proposal's own line is never rewritten; `read_log`
/// folds outcomes back onto it.
pub fn record_outcome(save_dir: &Path, id: u64, outcome: Outcome) -> std::io::Result<()> {
    let at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    append_entry(save_dir, &Entry::Outcome { id, outcome, at })
}

/// Every logged proposal with its latest outcome; unreadable lines are skipped.
pub fn read_log(save_dir: &Path) -> Vec<Proposal> {
    let text = std::fs::read_to_string(log_path(save_dir)).unwrap_or_default();
    let mut out: Vec<Proposal> = Vec::new();
    for entry in text.lines().filter_map(|l| serde_json::from_str(l).ok()) {
        match entry {
            Entry::Proposal(p) => out.push(p),
            Entry::Outcome { id, outcome, .. } => {
                if let Some(p) = out.iter_mut().find(|p| p.id == id) {
                    p.outcome = outcome;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::Journal;
    use crate::qc_export::tests::MockProvider;
    use sorrel_io::SampleIndex;

    fn session(dir: &Path) -> Session<MockProvider> {
        let prov = MockProvider {
            spikes: vec![vec![SampleIndex(10)], vec![SampleIndex(20)]],
            amps: vec![vec![1.0], vec![2.0]],
            n_samples: SampleIndex(1000),
        };
        Session::new(prov, Journal::open(&dir.join("j.journal")).unwrap())
    }

    #[test]
    fn jsonl_round_trip_and_commands() {
        let dir = tempfile::tempdir().unwrap();
        let s = session(dir.path());
        let p = Proposal::new(
            Source::Agent,
            Action::Merge {
                clusters: vec![ClusterId(1), ClusterId(0)],
            },
            0.8,
            vec!["CCG dip".into()],
            serde_json::json!({"action": "merge"}),
        );
        let q = Proposal::new(
            Source::Kmeans,
            Action::Relabel {
                cluster: ClusterId(0),
                label: "good".into(),
            },
            0.6,
            vec![],
            Value::Null,
        );
        assert!(q.id > p.id, "ids increase");
        append(dir.path(), &p).unwrap();
        append(dir.path(), &q).unwrap();
        record_outcome(dir.path(), p.id, Outcome::Accepted).unwrap();
        let log = read_log(dir.path());
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].outcome, Outcome::Accepted);
        assert_eq!(log[0].action, p.action);
        assert_eq!(log[1], q, "untouched proposal round-trips exactly");

        // Accept: retags (src:agent + src:human) then merge into the first.
        let cmds = p.to_commands(&s);
        assert!(matches!(
            &cmds[..],
            [
                CurationCommand::Tag { cluster: ClusterId(0), tags: t0 },
                CurationCommand::Tag { cluster: ClusterId(1), .. },
                CurationCommand::Merge { target: ClusterId(1), .. },
            ] if t0 == &["src:agent", "src:human"]
        ));
        // Reject: notmerge memory on both sides.
        let rej = p.reject_commands(&s);
        assert!(matches!(
            &rej[..],
            [
                CurationCommand::Tag { cluster: ClusterId(1), tags: a },
                CurationCommand::Tag { cluster: ClusterId(0), tags: b },
            ] if a == &["notmerge:0"] && b == &["notmerge:1"]
        ));
        let split = Proposal::new(
            Source::Gmm,
            Action::Split {
                cluster: ClusterId(0),
                spike_idx: vec![0],
            },
            0.5,
            vec![],
            Value::Null,
        );
        assert!(matches!(
            &split.reject_commands(&s)[..],
            [CurationCommand::Tag { tags, .. }] if tags == &["notsplit"]
        ));
    }
}

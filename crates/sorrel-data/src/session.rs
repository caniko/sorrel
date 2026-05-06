use crate::command::{CurationCommand, PhyLabelOp};
use crate::journal::SqliteJournal;
use anyhow::Result;
use sorrel_io::{ClusterId, DataProvider};

/// The core, generic curation session. `Session<P>` is monomorphised once per
/// backend type so calls into `provider` are inlined and devirtualised.
pub struct Session<P: DataProvider> {
    pub provider: P,
    /// Indexed by `ClusterId`. `P::Label` is `Copy + 'static`, so this is a
    /// flat, cache-friendly buffer (1 byte per cluster for `PhyLabel`).
    labels: Vec<P::Label>,
    journal: SqliteJournal,
    history: Vec<CurationCommand>,
    redo: Vec<CurationCommand>,
}

impl<P: DataProvider> Session<P> {
    pub fn new(provider: P, journal: SqliteJournal) -> Self {
        let labels = provider.initial_labels();
        Self {
            provider,
            labels,
            journal,
            history: Vec::new(),
            redo: Vec::new(),
        }
    }

    #[inline]
    pub fn labels(&self) -> &[P::Label] {
        &self.labels
    }

    #[inline]
    pub fn label(&self, cluster: ClusterId) -> Option<P::Label> {
        self.labels.get(cluster as usize).copied()
    }

    pub fn history(&self) -> &[CurationCommand] {
        &self.history
    }

    /// Replay an existing journal on top of an already-loaded session. The
    /// caller is responsible for applying side effects via `apply`.
    pub fn replay_journal(&mut self) -> Result<()>
    where
        P: ApplyPhyLabel,
    {
        let ops = self.journal.replay()?;
        for op in ops {
            self.apply_in_memory(&op);
            self.history.push(op);
        }
        Ok(())
    }

    /// Durably journal then apply. The fsync happens *before* the in-memory
    /// state changes, satisfying the crash-safety contract.
    pub fn dispatch(&mut self, cmd: CurationCommand) -> Result<()>
    where
        P: ApplyPhyLabel,
    {
        self.journal.append(&cmd)?;
        self.apply_in_memory(&cmd);
        self.history.push(cmd);
        self.redo.clear();
        Ok(())
    }

    fn apply_in_memory(&mut self, cmd: &CurationCommand)
    where
        P: ApplyPhyLabel,
    {
        match cmd {
            CurationCommand::Relabel { cluster, op } => {
                if let Some(slot) = self.labels.get_mut(*cluster as usize) {
                    *slot = P::label_from_op(*op);
                }
            }
            CurationCommand::Merge { sources, target } => {
                // V1: just relabel sources to Noise so they fall out of the
                // active list; full spike-table merging arrives later.
                for s in sources {
                    if let Some(slot) = self.labels.get_mut(*s as usize) {
                        *slot = P::label_from_op(PhyLabelOp::SetNoise);
                    }
                }
                let _ = target;
            }
            CurationCommand::Split { .. } | CurationCommand::Note { .. } => {
                // V1 stubs.
            }
        }
    }
}

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

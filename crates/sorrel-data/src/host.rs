//! `SessionHost`: the one owner of a curation session. The GUI, the MCP
//! server and the remote host all hold an `Arc<SessionHost>` and edit
//! through [`SessionHost::apply`] / [`SessionHost::checkout`], so every edit
//! is journaled with its [`Actor`], tagged the same way, and announced to
//! every subscriber as a [`HostEvent`]. The host also holds the agents'
//! [`AgentPolicy`] and the queue of proposals waiting for a human.

use crate::command::{describe_command, Actor, CurationCommand};
use crate::proposals::Proposal;
use crate::session::{ApplyPhyLabel, Session};
use anyhow::Result;
use sorrel_io::{ClusterId, DataProvider};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// One applied edit or head move, as subscribers see it.
#[derive(Clone, Debug, PartialEq)]
pub struct HostEvent {
    /// The head after the change.
    pub node: Option<u64>,
    pub actor: Actor,
    /// [`describe_command`] text; `undid: …` / `redid: …` / `checkout …` for
    /// head moves.
    pub desc: String,
    /// Clusters the change touched (for a head move, those of the nodes it
    /// crossed), including the ids a split allocated.
    pub clusters: Vec<ClusterId>,
    /// `Session::version()` after the change.
    pub version: u64,
}

type Waker = Box<dyn Fn() + Send + Sync>;

/// What an agent's edit tools do: apply directly (each edit visible and
/// undoable), or only queue a proposal for a human to accept.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AgentPolicy {
    #[default]
    Direct,
    ProposeOnly,
}

/// Owner of a session behind a lock, plus the event subscribers.
pub struct SessionHost<P: DataProvider> {
    session: Arc<RwLock<Session<P>>>,
    subscribers: Mutex<Vec<(Sender<HostEvent>, Waker)>>,
    policy: Mutex<AgentPolicy>,
    proposals: Mutex<Vec<Proposal>>,
    agents: Arc<Mutex<Vec<(u64, String)>>>,
}

/// A connected agent, listed by [`SessionHost::agents`] until dropped.
pub struct AgentSeat {
    agents: Arc<Mutex<Vec<(u64, String)>>>,
    id: u64,
}

impl Drop for AgentSeat {
    fn drop(&mut self) {
        lock(&self.agents).retain(|a| a.0 != self.id);
    }
}

impl<P: DataProvider + ApplyPhyLabel> SessionHost<P> {
    pub fn new(session: Session<P>) -> Self {
        Self {
            session: Arc::new(RwLock::new(session)),
            subscribers: Mutex::default(),
            policy: Mutex::default(),
            proposals: Mutex::default(),
            agents: Arc::default(),
        }
    }

    /// The shared session, for readers (view caches, background jobs) and
    /// for the non-edit writes (loading, saving). Edits go through
    /// [`Self::apply`].
    pub fn session(&self) -> &Arc<RwLock<Session<P>>> {
        &self.session
    }

    /// Read lock that survives a poisoned lock.
    pub fn read(&self) -> RwLockReadGuard<'_, Session<P>> {
        self.session.read().unwrap_or_else(|p| p.into_inner())
    }

    /// Write lock that survives a poisoned lock.
    pub fn write(&self) -> RwLockWriteGuard<'_, Session<P>> {
        self.session.write().unwrap_or_else(|p| p.into_inner())
    }

    /// Journal and apply `cmd` as `actor`, then announce it. The touched
    /// clusters' provenance tags come from the actor here, and only here:
    /// a human edit drops `review:*` and adds `src:human`, an agent edit
    /// adds `src:agent`. `Undo`/`Redo` move the head.
    pub fn apply(&self, actor: Actor, cmd: CurationCommand) -> Result<HostEvent> {
        self.apply_if(actor, cmd, |_| Ok(()))
    }

    /// [`Self::apply`], but `check` sees the session first, under the same
    /// write lock as the apply: no other edit can land between the check
    /// and the edit. An `Err` from `check` applies nothing.
    pub fn apply_if(
        &self,
        actor: Actor,
        cmd: CurationCommand,
        check: impl FnOnce(&Session<P>) -> Result<()>,
    ) -> Result<HostEvent> {
        let mut s = self.write();
        check(&s)?;
        let (desc, mut clusters) = match &cmd {
            CurationCommand::Undo => crossed(["undid", "undo"], s.history_commands().last()),
            CurationCommand::Redo => crossed(["redid", "redo"], s.redo_commands().last()),
            c => (describe_command(c), c.clusters()),
        };
        let cmd = match &actor {
            Actor::Human => s.with_human_review(cmd),
            Actor::Agent { .. } => s.with_agent_source(cmd),
            Actor::Algo { .. } | Actor::Remote { .. } => cmd,
        };
        let (n_before, before) = (s.n_clusters(), s.version());
        s.dispatch_as(actor.clone(), cmd)?;
        // Splits allocate ids at the tail; an undone split frees them again.
        clusters.extend((n_before..s.n_clusters()).map(ClusterId));
        let ev = HostEvent {
            node: s.head(),
            actor,
            desc,
            clusters,
            version: s.version(),
        };
        // Published under the write lock, so events leave in version order.
        // An undo at the root (or redo at a leaf) changed nothing: no event.
        if ev.version != before {
            self.publish(&ev);
        }
        Ok(ev)
    }

    /// Check out `node` (`None` = the root) as `actor`, then announce it.
    /// The event lists the clusters of the edits the head move crossed.
    pub fn checkout(&self, actor: Actor, node: Option<u64>) -> Result<HostEvent> {
        let mut s = self.write();
        let (n_before, old) = (s.n_clusters(), s.head());
        s.checkout(node)?;
        // Nodes on the way up from the old head and on the way down to the
        // new one, below their common ancestor.
        let ancestry = |mut at: Option<u64>| {
            let mut up = Vec::new();
            while let Some(n) = at {
                up.push(n);
                at = s.node(n).and_then(|(parent, ..)| parent);
            }
            up
        };
        let (from, to) = (ancestry(old), ancestry(node));
        let common = from
            .iter()
            .rev()
            .zip(to.iter().rev())
            .take_while(|(a, b)| a == b)
            .count();
        let mut clusters: Vec<ClusterId> = from[..from.len() - common]
            .iter()
            .chain(&to[..to.len() - common])
            .filter_map(|&n| s.node(n))
            .flat_map(|(.., cmd)| cmd.clusters())
            .collect();
        // Splits allocate ids at the tail (and an undone one frees them).
        let (lo, hi) = (n_before.min(s.n_clusters()), n_before.max(s.n_clusters()));
        clusters.extend((lo..hi).map(ClusterId));
        clusters.sort_unstable();
        clusters.dedup();
        let ev = HostEvent {
            node,
            actor,
            desc: match node.and_then(|n| s.node(n)) {
                Some((.., cmd)) => format!("back to: {}", describe_command(cmd)),
                None => "back to the sort as loaded".into(),
            },
            clusters,
            version: s.version(),
        };
        self.publish(&ev);
        Ok(ev)
    }

    /// Receive every event from now on; `waker` runs after each one is
    /// queued (a GUI repaint, a notification flush). Dropping the receiver
    /// unsubscribes at the next event.
    ///
    /// Events are published while the session's write lock is held, so a
    /// waker must be cheap and must never touch the host or its session
    /// (that would deadlock): request a repaint, set a flag, wake a thread.
    pub fn subscribe(&self, waker: impl Fn() + Send + Sync + 'static) -> Receiver<HostEvent> {
        let (tx, rx) = channel();
        lock(&self.subscribers).push((tx, Box::new(waker)));
        rx
    }

    /// List `name` among the connected agents while the seat lives.
    /// ponytail: a leaving agent wakes nobody; the list is current at the
    /// next repaint.
    pub fn agent_joined(&self, name: String) -> AgentSeat {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        lock(&self.agents).push((id, name));
        self.wake();
        AgentSeat {
            agents: Arc::clone(&self.agents),
            id,
        }
    }

    /// Names of the connected agents, sorted and deduplicated.
    pub fn agents(&self) -> Vec<String> {
        let mut names: Vec<String> = lock(&self.agents).iter().map(|a| a.1.clone()).collect();
        names.sort();
        names.dedup();
        names
    }

    fn wake(&self) {
        lock(&self.subscribers).iter().for_each(|(_, wake)| wake());
    }

    pub fn policy(&self) -> AgentPolicy {
        *lock(&self.policy)
    }

    pub fn set_policy(&self, policy: AgentPolicy) {
        *lock(&self.policy) = policy;
    }

    /// Queue `p` for a human and wake the subscribers (no event: nothing
    /// changed yet). The caller writes the audit log.
    pub fn propose(&self, p: Proposal) {
        lock(&self.proposals).push(p);
        self.wake();
    }

    /// The proposals waiting for a decision, oldest first.
    pub fn proposals(&self) -> Vec<Proposal> {
        lock(&self.proposals).clone()
    }

    /// Remove proposal `id` from the queue (to accept or reject it).
    pub fn take_proposal(&self, id: u64) -> Option<Proposal> {
        let mut q = lock(&self.proposals);
        let i = q.iter().position(|p| p.id == id)?;
        Some(q.remove(i))
    }

    fn publish(&self, ev: &HostEvent) {
        lock(&self.subscribers).retain(|(tx, wake)| {
            let alive = tx.send(ev.clone()).is_ok();
            if alive {
                wake();
            }
            alive
        });
    }
}

/// Mutex lock that survives poisoning (a panicking frontend thread must not
/// take the host down).
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Description and clusters of a head move over `cmd` (`None` at the end of
/// the walk); `[done, todo]` name the move.
fn crossed([done, todo]: [&str; 2], cmd: Option<&CurationCommand>) -> (String, Vec<ClusterId>) {
    match cmd {
        Some(c) => (format!("{done}: {}", describe_command(c)), c.clusters()),
        None => (format!("nothing to {todo}"), Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::PhyLabelOp;
    use crate::journal::Journal;
    use sorrel_io::{SampleIndex, TraceSamples, TraceSlice};

    struct Stub(Vec<Vec<SampleIndex>>);

    impl DataProvider for Stub {
        type Label = u8;
        fn sample_rate(&self) -> f32 {
            1000.0
        }
        fn n_channels(&self) -> u32 {
            1
        }
        fn n_samples(&self) -> SampleIndex {
            SampleIndex(0)
        }
        fn n_clusters(&self) -> u32 {
            self.0.len() as u32
        }
        fn spike_times(&self, c: ClusterId) -> &[SampleIndex] {
            self.0.get(c.idx()).map(Vec::as_slice).unwrap_or(&[])
        }
        fn trace(&self, start: SampleIndex, _: u32) -> TraceSlice<'_> {
            TraceSlice {
                start,
                n_channels: 1,
                samples: TraceSamples::I16(&[]),
            }
        }
        fn initial_labels(&self) -> Vec<u8> {
            vec![0; self.0.len()]
        }
    }

    impl ApplyPhyLabel for Stub {
        fn label_from_op(op: PhyLabelOp) -> u8 {
            op as u8
        }
    }

    fn host(dir: &std::path::Path) -> Arc<SessionHost<Stub>> {
        let spikes = vec![vec![SampleIndex(1), SampleIndex(2)], vec![SampleIndex(3)]];
        let journal = Journal::open(&dir.join("j.journal")).unwrap();
        Arc::new(SessionHost::new(Session::new(Stub(spikes), journal)))
    }

    /// An agent edit lands while another thread holds a read lock (it waits
    /// for it), is journaled with its actor and tagged `src:agent`, and every
    /// subscriber hears about it, including a split's new id.
    #[test]
    fn agent_edit_waits_for_readers_and_reaches_subscribers() {
        let dir = tempfile::tempdir().unwrap();
        let h = host(dir.path());
        let woken = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let w = Arc::clone(&woken);
        let rx = h.subscribe(move || {
            w.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });

        let reader = h.read();
        let agent = Actor::Agent {
            client: "codex".into(),
        };
        let edit = {
            let (h, agent) = (Arc::clone(&h), agent.clone());
            std::thread::spawn(move || {
                h.apply(
                    agent,
                    CurationCommand::Split {
                        cluster: ClusterId(0),
                        spike_idx: vec![1],
                        new_cluster: ClusterId(u32::MAX),
                        by: crate::command::Method::Llm("codex".into()),
                    },
                )
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(reader.n_clusters(), 2, "the reader sees a stable session");
        drop(reader);
        let ev = edit.join().unwrap().unwrap();

        assert_eq!(rx.recv().unwrap(), ev);
        assert_eq!(woken.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(ev.actor, agent);
        assert_eq!(ev.clusters, [ClusterId(0), ClusterId(2)]);
        assert!(ev.desc.starts_with("split cluster 0"), "{}", ev.desc);
        let s = h.read();
        assert_eq!(s.node(ev.node.unwrap()).unwrap().1, &agent);
        assert!(s.tags(ClusterId(0)).iter().any(|t| t == "src:agent"));
        assert_eq!(ev.version, s.version());
    }

    #[test]
    fn undo_and_checkout_are_announced_and_dead_subscribers_drop() {
        let dir = tempfile::tempdir().unwrap();
        let h = host(dir.path());
        let rx = h.subscribe(|| {});
        drop(h.subscribe(|| {}));
        let relabel = CurationCommand::Relabel {
            cluster: ClusterId(1),
            op: PhyLabelOp::SetGood,
        };
        let first = h.apply(Actor::Human, relabel).unwrap();
        let undo = h.apply(Actor::Human, CurationCommand::Undo).unwrap();
        assert_eq!(undo.desc, "undid: relabel cluster 1 as Good");
        assert_eq!(
            (undo.node, undo.clusters.as_slice()),
            (None, &[ClusterId(1)][..])
        );
        let back = h.checkout(Actor::Human, first.node).unwrap();
        assert_eq!(back.version, first.version);
        assert_eq!(rx.try_iter().count(), 3);
        assert_eq!(h.subscribers.lock().unwrap().len(), 1);
        let seat = h.agent_joined("codex".into());
        let other = h.agent_joined("codex".into());
        assert_eq!(h.agents(), ["codex"]);
        drop((seat, other));
        assert!(h.agents().is_empty());
        let end = h.apply(Actor::Human, CurationCommand::Redo).unwrap();
        assert_eq!(end.desc, "nothing to redo");
        assert_eq!(rx.try_iter().count(), 0, "a no-op is not announced");
    }

    fn relabel(c: u32, op: PhyLabelOp) -> CurationCommand {
        CurationCommand::Relabel {
            cluster: ClusterId(c),
            op,
        }
    }

    /// Four threads editing at once: subscribers see the heads in
    /// journal order (`version` is a hash, not a counter), because events are published under the write lock.
    #[test]
    fn concurrent_applies_publish_events_in_journal_order() {
        let dir = tempfile::tempdir().unwrap();
        let h = host(dir.path());
        let rx = h.subscribe(|| {});
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let h = Arc::clone(&h);
                std::thread::spawn(move || {
                    for i in 0..100 {
                        let op = [PhyLabelOp::SetGood, PhyLabelOp::SetNoise][i % 2];
                        h.apply(Actor::Human, relabel(t % 2, op)).unwrap();
                    }
                })
            })
            .collect();
        threads.into_iter().for_each(|t| t.join().unwrap());
        // Edits extend one branch, so the head after each is the next node id.
        let heads: Vec<u64> = rx.try_iter().map(|e| e.node.unwrap()).collect();
        assert_eq!(heads, (0..400).collect::<Vec<_>>());
    }

    /// Events are published while the write lock is held (which is what
    /// keeps them in order): a waker finds the session locked.
    #[test]
    fn events_are_published_under_the_write_lock() {
        let dir = tempfile::tempdir().unwrap();
        let h = host(dir.path());
        let lock = Arc::clone(h.session());
        let held = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&held);
        let _rx = h.subscribe(move || {
            seen.lock().unwrap().push(lock.try_read().is_err());
        });
        h.apply(Actor::Human, relabel(0, PhyLabelOp::SetGood))
            .unwrap();
        h.checkout(Actor::Human, None).unwrap();
        assert_eq!(*held.lock().unwrap(), [true, true]);
    }

    /// A head move reports the clusters of the edits it crossed, including
    /// the id a crossed split allocated.
    #[test]
    fn checkout_event_lists_the_touched_clusters() {
        let dir = tempfile::tempdir().unwrap();
        let h = host(dir.path());
        let a = h
            .apply(Actor::Human, relabel(1, PhyLabelOp::SetGood))
            .unwrap();
        let split = CurationCommand::Split {
            cluster: ClusterId(0),
            spike_idx: vec![1],
            new_cluster: ClusterId(u32::MAX),
            by: crate::command::Method::Manual,
        };
        h.apply(Actor::Human, split).unwrap();
        let back = h.checkout(Actor::Human, a.node).unwrap();
        assert_eq!(back.clusters, [ClusterId(0), ClusterId(2)], "the split");
        let root = h.checkout(Actor::Human, None).unwrap();
        assert_eq!(root.clusters, [ClusterId(1)], "the relabel");
        let fwd = h.checkout(Actor::Human, a.node).unwrap();
        assert_eq!(fwd.clusters, [ClusterId(1)]);
    }

    /// The check runs under the apply's write lock: an edit that lands
    /// between a guest's cursor read and its apply makes the check fail.
    #[test]
    fn apply_if_checks_under_the_write_lock() {
        let dir = tempfile::tempdir().unwrap();
        let h = host(dir.path());
        let seen = h.read().version();
        h.apply(Actor::Human, relabel(0, PhyLabelOp::SetGood))
            .unwrap(); // host edit lands
        let refused = h.apply_if(Actor::Human, relabel(1, PhyLabelOp::SetNoise), |s| {
            anyhow::ensure!(s.version() == seen, "stale");
            Ok(())
        });
        assert_eq!(refused.unwrap_err().to_string(), "stale");
        assert_eq!(h.read().label(ClusterId(1)), Some(0), "nothing applied");
        h.apply_if(Actor::Human, relabel(1, PhyLabelOp::SetNoise), |_| Ok(()))
            .unwrap();
    }
}

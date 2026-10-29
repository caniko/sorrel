//! `SessionHost`: the one owner of a curation session. The GUI, the MCP
//! server and the remote host all hold an `Arc<SessionHost>` and edit
//! through [`SessionHost::apply`] / [`SessionHost::checkout`], so every edit
//! is journaled with its [`Actor`], tagged the same way, and announced to
//! every subscriber as a [`HostEvent`]. The host also holds the agents'
//! [`AgentPolicy`] and the queue of proposals waiting for a human.

use crate::command::{describe_command, Actor, CurationCommand};
use crate::proposals::{spike_basis, Action, Outcome, Proposal};
use crate::session::{ApplyPhyLabel, Session};
use anyhow::Result;
use sorrel_io::{ClusterId, DataProvider};
use std::sync::atomic::{AtomicU64, Ordering};
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
    /// Publication order, unique per host (versions repeat across undo/redo);
    /// 0 when nothing was published. Lets a connection recognise its own
    /// events without relying on the (non-unique) actor name.
    pub seq: u64,
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
    seq: AtomicU64,
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
            seq: AtomicU64::new(1),
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
        self.apply_if_then(actor, cmd, check, |_| ())
            .map(|(ev, ())| ev)
    }

    /// [`Self::apply_if`], and `then` reads the session after the edit while
    /// the write lock is still held (e.g. the remote cursor the edit produced).
    pub fn apply_if_then<R>(
        &self,
        actor: Actor,
        cmd: CurationCommand,
        check: impl FnOnce(&Session<P>) -> Result<()>,
        then: impl FnOnce(&Session<P>) -> R,
    ) -> Result<(HostEvent, R)> {
        self.apply_locked(
            actor,
            |s| {
                check(s)?;
                Ok(Some(cmd))
            },
            then,
        )?
        .ok_or_else(|| anyhow::anyhow!("no command to apply"))
    }

    /// [`Self::apply`], but `build` makes the command from the session under
    /// the same write lock as the apply, so an edit that changes the state
    /// the command was derived from (e.g. a tag list it replaces) cannot
    /// land in between. `Ok(None)` applies nothing; an `Err` applies nothing.
    pub fn apply_built(
        &self,
        actor: Actor,
        build: impl FnOnce(&Session<P>) -> Result<Option<CurationCommand>>,
    ) -> Result<Option<HostEvent>> {
        Ok(self.apply_locked(actor, build, |_| ())?.map(|(ev, ())| ev))
    }

    fn apply_locked<R>(
        &self,
        actor: Actor,
        build: impl FnOnce(&Session<P>) -> Result<Option<CurationCommand>>,
        then: impl FnOnce(&Session<P>) -> R,
    ) -> Result<Option<(HostEvent, R)>> {
        let mut s = self.write();
        let Some(cmd) = build(&s)? else {
            return Ok(None);
        };
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
        // Splits allocate ids at the tail; an undone split frees them again,
        // so report the ids between the before and after counts either way.
        let (lo, hi) = (n_before.min(s.n_clusters()), n_before.max(s.n_clusters()));
        clusters.extend((lo..hi).map(ClusterId));
        let mut ev = HostEvent {
            node: s.head(),
            actor,
            desc,
            clusters,
            version: s.version(),
            seq: 0,
        };
        // Published under the write lock, so events leave in version order.
        // An undo at the root (or redo at a leaf) changed nothing: no event.
        if ev.version != before {
            ev.seq = self.seq.fetch_add(1, Ordering::Relaxed);
            self.publish(&ev);
        }
        let r = then(&s);
        Ok(Some((ev, r)))
    }

    /// Check out `node` (`None` = the root) as `actor`, then announce it.
    /// The event lists the clusters of the edits the head move crossed.
    pub fn checkout(&self, actor: Actor, node: Option<u64>) -> Result<HostEvent> {
        let mut s = self.write();
        let (n_before, old) = (s.n_clusters(), s.head());
        // Ids the old branch's splits allocated, before it is unwound.
        let left = s.applied_commands();
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
        // Sibling branches can allocate the same id for different spikes,
        // leaving the count (so the range above) unchanged: name the child
        // of every crossed split.
        let (entered, left) = (
            s.applied_commands(),
            &left[left.len() - (from.len() - common)..],
        );
        clusters.extend(
            entered[entered.len() - (to.len() - common)..]
                .iter()
                .chain(left)
                .flat_map(split_children),
        );
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
            seq: self.seq.fetch_add(1, Ordering::Relaxed),
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
    /// changed yet). The caller writes the audit log. A split records the
    /// cluster's spikes now, so a later membership change refuses it.
    pub fn propose(&self, mut p: Proposal) {
        if let (Action::Split { cluster, .. }, None) = (&p.action, &p.basis) {
            p.basis = Some(spike_basis(&self.read(), *cluster));
        }
        lock(&self.proposals).push(p);
        self.wake();
    }

    /// Requeue the proposals `save_dir`'s audit log still has as `Pending`
    /// (a previous run exited before a human decided), keeping a split's
    /// recorded basis so a changed membership still refuses it. Ids already
    /// queued are skipped, so calling this twice is harmless.
    pub fn restore_proposals(&self, save_dir: &std::path::Path) {
        let mut q = lock(&self.proposals);
        for p in crate::proposals::read_log(save_dir) {
            if p.outcome == Outcome::Pending && !q.iter().any(|k| k.id == p.id) {
                q.push(p);
            }
        }
        drop(q);
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

/// The ids a resolved command's splits allocated (see
/// [`Session::applied_commands`]).
fn split_children(cmd: &CurationCommand) -> Vec<ClusterId> {
    match cmd {
        CurationCommand::Split { new_cluster, .. } => vec![*new_cluster],
        CurationCommand::Batch { children } => children.iter().flat_map(split_children).collect(),
        _ => Vec::new(),
    }
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

    /// Moving between sibling split branches that allocate the same id for
    /// different spikes (the cluster count does not change) still names it.
    #[test]
    fn same_size_checkout_between_split_branches_names_the_child() {
        let dir = tempfile::tempdir().unwrap();
        let h = host(dir.path());
        let split = |i| CurationCommand::Split {
            cluster: ClusterId(0),
            spike_idx: vec![i],
            new_cluster: ClusterId(u32::MAX),
            by: crate::command::Method::Manual,
        };
        let a = h.apply(Actor::Human, split(1)).unwrap();
        h.checkout(Actor::Human, None).unwrap();
        h.apply(Actor::Human, split(0)).unwrap();
        let ev = h.checkout(Actor::Human, a.node).unwrap();
        assert_eq!(ev.clusters, [ClusterId(0), ClusterId(2)]);
    }

    /// Undoing a split names the child id it frees, as the split named it.
    #[test]
    fn undo_of_a_split_reports_the_freed_id() {
        let dir = tempfile::tempdir().unwrap();
        let h = host(dir.path());
        let split = CurationCommand::Split {
            cluster: ClusterId(0),
            spike_idx: vec![1],
            new_cluster: ClusterId(u32::MAX),
            by: crate::command::Method::Manual,
        };
        let done = h.apply(Actor::Human, split).unwrap();
        assert!(done.clusters.contains(&ClusterId(2)));
        let undone = h.apply(Actor::Human, CurationCommand::Undo).unwrap();
        assert!(undone.clusters.contains(&ClusterId(2)), "{undone:?}");
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

    /// Tag commands carry a replacement list, so they must be built from the
    /// state the apply sees: a tag edit that lands between reading the
    /// session and applying survives accept and reject when `apply_built`
    /// builds under the lock, and is overwritten when built from a stale read.
    #[test]
    fn apply_built_keeps_a_tag_edit_that_landed_first() {
        use crate::proposals::{Action, Proposal, Source};
        let tag = |c: u32, t: &str| CurationCommand::Tag {
            cluster: ClusterId(c),
            tags: vec![t.into()],
        };
        let p = Proposal::new(
            Source::Agent,
            Action::Merge {
                clusters: vec![ClusterId(0), ClusterId(1)],
            },
            0.5,
            vec![],
            Default::default(),
        );
        for accept in [true, false] {
            let build = |s: &Session<Stub>| {
                let cmds = if accept {
                    p.to_commands(s)
                } else {
                    p.reject_commands(s)
                };
                CurationCommand::batch(cmds)
            };
            // Old flow: commands from a read, a tag edit lands, then apply.
            let dir = tempfile::tempdir().unwrap();
            let h = host(dir.path());
            let stale = build(&h.read());
            h.apply(
                Actor::Agent {
                    client: "mcp".into(),
                },
                tag(0, "mine"),
            )
            .unwrap();
            h.apply(Actor::Human, stale.unwrap()).unwrap();
            assert!(
                !h.read().tags(ClusterId(0)).contains(&"mine".to_owned()),
                "a stale build overwrites the tag"
            );
            // New flow: the builder runs under the write lock, after the edit.
            let dir = tempfile::tempdir().unwrap();
            let h = host(dir.path());
            h.apply(
                Actor::Agent {
                    client: "mcp".into(),
                },
                tag(0, "mine"),
            )
            .unwrap();
            h.apply_built(Actor::Human, |s| {
                assert!(h.session().try_read().is_err(), "write lock is held");
                Ok(build(s))
            })
            .unwrap();
            assert!(
                h.read().tags(ClusterId(0)).contains(&"mine".to_owned()),
                "accept={accept}: the tag edit survives"
            );
        }
    }

    /// A proposal still `Pending` in the audit log when the host closed
    /// comes back as a card (split basis intact) on the next open; a decided
    /// one does not, and restoring twice does not duplicate.
    #[test]
    fn pending_proposals_survive_a_restart() {
        use crate::proposals::{self, Source};
        let dir = tempfile::tempdir().unwrap();
        let new = |action| Proposal::new(Source::Agent, action, 0.5, vec![], Default::default());
        let mut split = new(Action::Split {
            cluster: ClusterId(0),
            spike_idx: vec![0],
        });
        let accept = new(Action::Accept {
            cluster: ClusterId(1),
        });
        let (split_id, accept_id) = (split.id, accept.id);
        let first = host(dir.path());
        // The proposer stamps the basis when it parses, before the audit line.
        split.basis = Some(spike_basis(&first.read(), ClusterId(0)));
        for p in [split, accept] {
            proposals::append(dir.path(), &p).unwrap();
            first.propose(p);
        }
        proposals::record_outcome(dir.path(), accept_id, Outcome::Rejected).unwrap();
        let basis = first.proposals()[0].basis.clone();
        assert!(basis.is_some());
        drop(first);

        let second = host(dir.path());
        assert!(second.proposals().is_empty(), "a new host starts empty");
        second.restore_proposals(dir.path());
        second.restore_proposals(dir.path());
        let back = second.proposals();
        assert_eq!(back.len(), 1, "only the undecided one, once");
        assert_eq!((back[0].id, back[0].basis.clone()), (split_id, basis));
    }
}

//! `SessionHost`: the one owner of a curation session. The GUI, the MCP
//! server and the remote host all hold an `Arc<SessionHost>` and edit
//! through [`SessionHost::apply`] / [`SessionHost::checkout`], so every edit
//! is journaled with its [`Actor`], tagged the same way, and announced to
//! every subscriber as a [`HostEvent`].

use crate::command::{describe_command, Actor, CurationCommand};
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

/// Owner of a session behind a lock, plus the event subscribers.
pub struct SessionHost<P: DataProvider> {
    session: Arc<RwLock<Session<P>>>,
    subscribers: Mutex<Vec<(Sender<HostEvent>, Waker)>>,
}

impl<P: DataProvider + ApplyPhyLabel> SessionHost<P> {
    pub fn new(session: Session<P>) -> Self {
        Self {
            session: Arc::new(RwLock::new(session)),
            subscribers: Mutex::default(),
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
        let mut s = self.write();
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
        drop(s);
        // An undo at the root (or redo at a leaf) changed nothing: no event.
        if ev.version != before {
            self.publish(&ev);
        }
        Ok(ev)
    }

    /// Check out `node` (`None` = the root) as `actor`, then announce it.
    pub fn checkout(&self, actor: Actor, node: Option<u64>) -> Result<HostEvent> {
        let mut s = self.write();
        s.checkout(node)?;
        let ev = HostEvent {
            node,
            actor,
            desc: match node {
                Some(n) => format!("checkout node {n}"),
                None => "checkout the sort as loaded".into(),
            },
            clusters: Vec::new(),
            version: s.version(),
        };
        drop(s);
        self.publish(&ev);
        Ok(ev)
    }

    /// Receive every event from now on; `waker` runs after each one is
    /// queued (a GUI repaint, a notification flush). Dropping the receiver
    /// unsubscribes at the next event.
    pub fn subscribe(&self, waker: impl Fn() + Send + Sync + 'static) -> Receiver<HostEvent> {
        let (tx, rx) = channel();
        self.subscribers
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((tx, Box::new(waker)));
        rx
    }

    fn publish(&self, ev: &HostEvent) {
        let mut subs = self.subscribers.lock().unwrap_or_else(|p| p.into_inner());
        subs.retain(|(tx, wake)| {
            let alive = tx.send(ev.clone()).is_ok();
            if alive {
                wake();
            }
            alive
        });
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
        let end = h.apply(Actor::Human, CurationCommand::Redo).unwrap();
        assert_eq!(end.desc, "nothing to redo");
        assert_eq!(rx.try_iter().count(), 0, "a no-op is not announced");
    }
}

//! Guest-side actor that owns the connection to the host: forwards journal
//! records in order and fetches trace ranges for the mirror.
//!
//! One actor, one queue: a trace fetch can delay a record by at most one
//! request. Local `tell(..).try_send()` on its unbounded mailbox is what the
//! egui thread calls, so the UI never blocks on the network.

use std::sync::Arc;

use piying::actor::RemoteActorRef;
use piying::message::{Context, Message};
use sorrel_data::CurationCommand;
use tokio::sync::watch;

use crate::host::HostActor;
use crate::mirror::Mirror;
use crate::protocol::{Cursor, FetchTrace, HostError, Record};

#[derive(piying::Actor)]
pub struct LinkActor {
    host: RemoteActorRef<HostActor>,
    mirror: Option<Arc<Mirror>>,
    cursor: Cursor,
    /// Set once forwarding is impossible (host diverged); records stay in
    /// the guest journal and a later `join` catches up.
    stopped: bool,
    error: watch::Sender<Option<String>>,
}

impl LinkActor {
    pub fn new(
        host: RemoteActorRef<HostActor>,
        cursor: Cursor,
        error: watch::Sender<Option<String>>,
    ) -> Self {
        Self {
            host,
            mirror: None,
            cursor,
            stopped: false,
            error,
        }
    }
}

pub struct SetMirror(pub Arc<Mirror>);
pub struct Fetch {
    pub offset: u64,
    pub len: u64,
}
pub struct Forward(pub CurationCommand);
/// Reply: the cursor the host is at after everything queued so far.
pub struct Flush;

impl Message<SetMirror> for LinkActor {
    type Reply = ();
    async fn handle(&mut self, m: SetMirror, _: &mut Context<Self, Self::Reply>) {
        self.mirror = Some(m.0);
    }
}

impl Message<Fetch> for LinkActor {
    type Reply = ();
    async fn handle(&mut self, m: Fetch, _: &mut Context<Self, Self::Reply>) {
        let Some(mirror) = self.mirror.clone() else {
            return;
        };
        let req = FetchTrace {
            offset: m.offset,
            len: m.len,
        };
        match self.host.ask(&req).send().await {
            Ok(bytes) => mirror.complete(m.offset, &bytes.0),
            Err(e) => {
                log::warn!("trace fetch {}+{} failed: {e}", m.offset, m.len);
                mirror.fail(m.offset..m.offset + m.len);
            }
        }
    }
}

impl Message<Forward> for LinkActor {
    type Reply = ();
    async fn handle(&mut self, m: Forward, _: &mut Context<Self, Self::Reply>) {
        if self.stopped {
            return;
        }
        let rec = Record {
            cursor: self.cursor,
            cmd: m.0,
        };
        match self.host.ask(&rec).send().await {
            Ok(next) => {
                self.cursor = next;
                let _ = self.error.send(None);
            }
            Err(e) => {
                let diverged = matches!(
                    &e,
                    piying::error::RemoteSendError::HandlerError(HostError::Diverged { .. })
                );
                let msg = format!("record not forwarded: {e}");
                log::error!("{msg}");
                let _ = self.error.send(Some(msg));
                if diverged {
                    self.stopped = true;
                }
            }
        }
    }
}

impl Message<Flush> for LinkActor {
    type Reply = Cursor;
    async fn handle(&mut self, _: Flush, _: &mut Context<Self, Self::Reply>) -> Cursor {
        self.cursor
    }
}

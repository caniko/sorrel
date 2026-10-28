//! Host side: a piying actor that owns the session and answers the guest.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use piying::actor::{ActorId, Spawn};
use piying::message::{Context as ActorContext, Message};
use sorrel_data::session::ApplyPhyLabel;
use sorrel_data::{CurationCommand, Journal, Record as JournalRecord, Session};
use sorrel_io::{DataProvider, TraceDtype};

use crate::pairing::{Pairing, PairingCode};
use crate::protocol::*;
use crate::swarm::{Mode, Role};

/// What the actor needs from a session, independent of the provider type so
/// the actor itself stays non-generic (piying's derive macros want that).
pub trait Served: Send + 'static {
    fn cursor(&self) -> Cursor;
    fn n_clusters_at_baseline(&self) -> u32;
    fn identity_bytes(&self) -> Vec<u8>;
    fn journal_path(&self) -> &Path;
    fn apply(&mut self, cmd: CurationCommand) -> Result<()>;
}

pub struct SessionServed<P: DataProvider>(pub Session<P>);

impl<P: DataProvider + ApplyPhyLabel> Served for SessionServed<P> {
    fn cursor(&self) -> Cursor {
        let j = self.0.journal();
        Cursor {
            baseline: j.baseline(),
            n_records: j.n_records(),
        }
    }
    fn n_clusters_at_baseline(&self) -> u32 {
        self.0.journal().n_clusters_at_baseline()
    }
    fn identity_bytes(&self) -> Vec<u8> {
        self.0.provider.identity_bytes()
    }
    fn journal_path(&self) -> &Path {
        self.0.journal_path()
    }
    fn apply(&mut self, cmd: CurationCommand) -> Result<()> {
        self.0.dispatch(cmd)
    }
}

/// The raw recording as the host sees it.
#[derive(Clone, Debug)]
pub struct DatSource {
    pub path: PathBuf,
    pub offset: u64,
    pub dtype: TraceDtype,
    pub n_channels: u32,
    pub sample_rate: f32,
}

#[derive(piying::Actor, piying::RemoteActor)]
#[remote_actor(id = "sorrel/host/1")]
pub struct HostActor {
    served: Box<dyn Served>,
    dat: DatSource,
    dat_file: File,
    payload_bytes: u64,
    files: Vec<FileEntry>,
    by_path: HashMap<String, PathBuf>,
}

impl HostActor {
    pub fn new(served: Box<dyn Served>, root: PathBuf, dat: DatSource) -> Result<Self> {
        let dat_file =
            File::open(&dat.path).with_context(|| format!("open {}", dat.path.display()))?;
        let total = dat_file.metadata()?.len();
        if dat.offset > total {
            bail!("dat offset {} exceeds file size {total}", dat.offset);
        }
        let exclude = [canon(&dat.path), canon(served.journal_path())];
        let mut files = Vec::new();
        let mut by_path = HashMap::new();
        walk(&root, &root, &exclude, &mut files, &mut by_path)?;
        files.sort_by(|a, b| a.path.cmp(&b.path));
        log::info!(
            "sharing {} files ({} MiB) + {} MiB raw trace on demand",
            files.len(),
            files.iter().map(|f| f.len).sum::<u64>() >> 20,
            (total - dat.offset) >> 20
        );
        Ok(Self {
            served,
            payload_bytes: total - dat.offset,
            dat,
            dat_file,
            files,
            by_path,
        })
    }
}

fn canon(p: &Path) -> PathBuf {
    p.canonicalize().unwrap_or_else(|_| p.to_path_buf())
}

fn walk(
    root: &Path,
    dir: &Path,
    exclude: &[PathBuf],
    out: &mut Vec<FileEntry>,
    by_path: &mut HashMap<String, PathBuf>,
) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with(".sorrel") {
            continue; // derived cache, remote mirror
        }
        let ty = entry.file_type()?;
        if ty.is_dir() {
            walk(root, &path, exclude, out, by_path)?;
            continue;
        }
        if !ty.is_file() || exclude.contains(&canon(&path)) {
            continue;
        }
        let rel = path
            .strip_prefix(root)?
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        let mut hasher = blake3::Hasher::new();
        hasher.update_reader(File::open(&path)?)?;
        out.push(FileEntry {
            len: entry.metadata()?.len(),
            blake3: *hasher.finalize().as_bytes(),
            path: rel.clone(),
        });
        by_path.insert(rel, path);
    }
    Ok(())
}

pub(crate) fn read_at(f: &File, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.read_exact_at(&mut buf, offset)?;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            let n = f.seek_read(&mut buf[done..], offset + done as u64)?;
            if n == 0 {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            done += n;
        }
    }
    Ok(buf)
}

fn failed(e: impl std::fmt::Display) -> HostError {
    HostError::Failed(e.to_string())
}

#[piying::remote_message("sorrel/hello/1")]
impl Message<Hello> for HostActor {
    type Reply = Result<HostInfo, HostError>;

    async fn handle(&mut self, msg: Hello, _: &mut ActorContext<Self, Self::Reply>) -> Self::Reply {
        if msg.proto != PROTO_VERSION {
            return Err(HostError::Version {
                host: PROTO_VERSION,
            });
        }
        Ok(HostInfo {
            proto: PROTO_VERSION,
            files: self.files.clone(),
            dat: DatInfo {
                payload_bytes: self.payload_bytes,
                dtype: dtype_tag(self.dat.dtype),
                n_channels: self.dat.n_channels,
                sample_rate: self.dat.sample_rate,
            },
            cursor: self.served.cursor(),
            n_clusters_at_baseline: self.served.n_clusters_at_baseline(),
            identity_bytes: self.served.identity_bytes(),
        })
    }
}

#[piying::remote_message("sorrel/fetch-file/1")]
impl Message<FetchFile> for HostActor {
    type Reply = Result<Bytes, HostError>;

    async fn handle(
        &mut self,
        msg: FetchFile,
        _: &mut ActorContext<Self, Self::Reply>,
    ) -> Self::Reply {
        let Some(path) = self.by_path.get(&msg.path) else {
            return Err(HostError::Rejected(format!("unknown file {}", msg.path)));
        };
        if msg.len > CHUNK {
            return Err(HostError::Rejected(format!("chunk {} > {CHUNK}", msg.len)));
        }
        let f = File::open(path).map_err(failed)?;
        let total = f.metadata().map_err(failed)?.len();
        let end = msg.offset.saturating_add(msg.len).min(total);
        if msg.offset >= end {
            return Ok(Bytes(Vec::new()));
        }
        read_at(&f, msg.offset, (end - msg.offset) as usize)
            .map(Bytes)
            .map_err(failed)
    }
}

#[piying::remote_message("sorrel/journal-tail/2")]
impl Message<JournalTail> for HostActor {
    type Reply = Result<Vec<JournalRecord>, HostError>;

    async fn handle(
        &mut self,
        msg: JournalTail,
        _: &mut ActorContext<Self, Self::Reply>,
    ) -> Self::Reply {
        let all = Journal::replay_at(self.served.journal_path()).map_err(failed)?;
        let from = (msg.from as usize).min(all.len());
        Ok(all[from..].to_vec())
    }
}

#[piying::remote_message("sorrel/fetch-trace/1")]
impl Message<FetchTrace> for HostActor {
    type Reply = Result<Bytes, HostError>;

    async fn handle(
        &mut self,
        msg: FetchTrace,
        _: &mut ActorContext<Self, Self::Reply>,
    ) -> Self::Reply {
        if msg.len > CHUNK {
            return Err(HostError::Rejected(format!("chunk {} > {CHUNK}", msg.len)));
        }
        let end = msg.offset.saturating_add(msg.len).min(self.payload_bytes);
        if msg.offset >= end {
            return Err(HostError::Rejected(format!(
                "trace range {}..{} outside payload {}",
                msg.offset,
                msg.offset + msg.len,
                self.payload_bytes
            )));
        }
        read_at(
            &self.dat_file,
            self.dat.offset + msg.offset,
            (end - msg.offset) as usize,
        )
        .map(Bytes)
        .map_err(failed)
    }
}

#[piying::remote_message("sorrel/record/1")]
impl Message<Record> for HostActor {
    type Reply = Result<Cursor, HostError>;

    async fn handle(
        &mut self,
        msg: Record,
        _: &mut ActorContext<Self, Self::Reply>,
    ) -> Self::Reply {
        let here = self.served.cursor();
        if msg.cursor != here {
            return Err(HostError::Diverged { host: here });
        }
        self.served.apply(msg.cmd).map_err(failed)?;
        Ok(self.served.cursor())
    }
}

/// A running host: the swarm plus the actor the guest talks to.
pub struct Serving {
    pub net: crate::swarm::Net,
    pub actor: piying::actor::ActorRef<HostActor>,
}

/// Start the swarm and register the host actor; returns immediately.
pub async fn serve(
    served: Box<dyn Served>,
    root: PathBuf,
    dat: DatSource,
    mode: &Mode,
    pairing: &Pairing,
) -> Result<Serving> {
    let net = crate::swarm::start(
        pairing.host.clone(),
        mode,
        Role::Host {
            guest: pairing.guest_peer(),
        },
    )
    .await?;
    let actor = HostActor::new(served, root, dat)?;
    let actor_ref = HostActor::spawn_with_mailbox(actor, piying::mailbox::unbounded());
    piying::remote::register_actor_local_sync(&actor_ref, ActorId::new_with_peer_id(0, net.peer));
    Ok(Serving {
        net,
        actor: actor_ref,
    })
}

/// Headless host: print the pairing code, serve until Ctrl-C.
pub fn share(served: Box<dyn Served>, root: PathBuf, dat: DatSource, mode: Mode) -> Result<()> {
    let code = PairingCode::generate()?;
    let pairing = Pairing::derive(&code)?;
    println!("pairing code:  {code}");
    crate::rt().block_on(async move {
        let Serving { mut net, .. } = serve(served, root, dat, &mode, &pairing).await?;

        match &mode {
            Mode::Direct { .. } => {
                // Give the listeners a moment to report per-interface addresses.
                tokio::time::sleep(Duration::from_millis(300)).await;
                let addrs = net.listen_addrs.borrow().clone();
                println!("on the guest machine run one of:");
                for a in addrs
                    .iter()
                    .filter(|a| !a.to_string().contains("/127.0.0.1/"))
                {
                    println!("  sorrel join \"{code}\" --direct {a}");
                }
                println!("  sorrel join \"{code}\"          # same LAN, found via mDNS");
            }
            Mode::Relay(relay) => {
                println!("on the guest machine run:");
                println!("  sorrel join \"{code}\" --relay {relay}");
            }
        }
        println!("serving; Ctrl-C to stop");

        let ctrl_c = tokio::signal::ctrl_c();
        tokio::pin!(ctrl_c);
        loop {
            tokio::select! {
                _ = &mut ctrl_c => break,
                r = net.connected.changed() => {
                    if r.is_err() { break; }
                    let up = *net.connected.borrow();
                    println!("{}", if up { "guest connected" } else { "guest disconnected" });
                }
            }
        }
        Ok(())
    })
}

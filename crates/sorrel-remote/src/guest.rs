//! Guest side: download the phy artefacts, mirror the recording on demand,
//! open a normal `Session<KilosortProvider>`, and stream its records home.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use piying::actor::{RemoteActorRef, Spawn};
use sorrel_data::{baseline_hash, Journal, Session};
use sorrel_io::kilosort::KilosortOpenParams;
use sorrel_io::{HasGeometry, KilosortProvider};
use tokio::sync::watch;

use crate::host::HostActor;
use crate::link::{Flush, Forward, LinkActor, SetMirror};
use crate::mirror::Mirror;
use crate::pairing::{Pairing, PairingCode};
use crate::protocol::*;
use crate::swarm::{Mode, Role};

pub const JOURNAL_NAME: &str = "sorrel.journal";
const MIRROR_REL: &str = ".sorrel/remote/recording.dat";

pub struct Joined {
    pub session: Session<KilosortProvider>,
    /// Direct handle to the host actor (tests and diagnostics).
    pub host: RemoteActorRef<HostActor>,
    pub root: PathBuf,
    pub positions: Vec<[f32; 2]>,
    /// `true` while connected to the host.
    pub connected: watch::Receiver<bool>,
    /// Last forwarding or fetch error, `None` once things work again.
    pub error: watch::Receiver<Option<String>>,
}

/// Connect, sync `into/`, and build the session. Re-running with the same
/// `into` resumes: matching files are skipped and the guest's own journal is
/// compared against the host cursor.
pub fn join(code: &PairingCode, mode: &Mode, into: &Path) -> Result<Joined> {
    let pairing = Pairing::derive(code)?;
    let host_peer = pairing.host_peer();
    std::fs::create_dir_all(into)?;
    crate::rt().block_on(async move {
        let mut net =
            crate::swarm::start(pairing.guest.clone(), mode, Role::Guest { host: host_peer })
                .await?;
        println!("looking for host {host_peer} …");
        net.wait_connected(Duration::from_secs(60)).await?;
        let host =
            RemoteActorRef::<HostActor>::for_peer_with_sender(host_peer, net.swarm_tx.clone());

        let info: HostInfo = host
            .ask(&Hello {
                proto: PROTO_VERSION,
            })
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("hello: {e}"))?;
        if info.proto != PROTO_VERSION {
            bail!(
                "host speaks protocol v{}, this build v{PROTO_VERSION}",
                info.proto
            );
        }
        let dtype = info.dat.dtype().context("host sent unknown trace dtype")?;

        for f in &info.files {
            sync_file(&host, into, f).await?;
        }

        // Journal: ours must be a prefix-or-extension of the host's.
        let journal_path = into.join(JOURNAL_NAME);
        let mut journal = Journal::open_or_create(
            &journal_path,
            info.cursor.baseline,
            info.n_clusters_at_baseline,
        )?;
        let local_n = journal.n_records();
        let host_n = info.cursor.n_records;
        let mut cursor = info.cursor;
        if local_n < host_n {
            let tail: Vec<_> = host
                .ask(&JournalTail { from: local_n })
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("journal tail: {e}"))?;
            for rec in &tail {
                journal.append(rec)?;
            }
            println!("journal: pulled {} records from host", tail.len());
        } else if local_n > host_n {
            // We curated offline (or the host was down): push our tail first.
            let ours = Journal::replay_at(&journal_path)?;
            for rec in &ours[host_n as usize..] {
                // ponytail: only offline edits are pushed; an offline undo,
                // redo or save needs the node-id protocol (PROTO v4).
                let sorrel_data::Record::Node { cmd, .. } = rec else {
                    bail!("cannot push an offline undo, redo or save to the host; re-join fresh");
                };
                cursor = host
                    .ask(&Record {
                        cursor,
                        cmd: cmd.clone(),
                    })
                    .send()
                    .await
                    .map_err(|e| anyhow::anyhow!("catch-up record rejected: {e}"))?;
            }
            println!("journal: pushed {} records to host", local_n - host_n);
        }
        let local_baseline = baseline_hash(&std::fs::read(into.join("spike_clusters.npy"))?);
        if local_baseline != info.cursor.baseline {
            bail!(
                "spike_clusters.npy hash differs from the host journal baseline; download corrupt?"
            );
        }

        let (err_tx, error) = watch::channel(None);
        let link = LinkActor::spawn_with_mailbox(
            LinkActor::new(host.clone(), cursor, err_tx),
            piying::mailbox::unbounded(),
        );
        let mirror = Mirror::create(&into.join(MIRROR_REL), info.dat.payload_bytes, link.clone())?;
        link.tell(SetMirror(mirror.clone())).send().await?;

        let provider = KilosortProvider::open(
            into,
            KilosortOpenParams {
                sample_rate: Some(info.dat.sample_rate),
                n_channels: Some(info.dat.n_channels),
                dtype: Some(dtype),
                offset: Some(0),
                dat_path: Some(into.join(MIRROR_REL)),
                trace_fetch: Some(mirror.clone()),
                identity_override: Some(info.identity_bytes.clone()),
            },
        )?;
        let mut session = Session::new(provider, journal);
        session.seed_amplitudes();
        session.seed_templates();
        session.seed_pc_features();
        session.seed_template_waveforms();
        session.replay_journal()?;
        let sink_link = link.clone();
        session.set_record_sink(move |cmd| {
            if sink_link.tell(Forward(cmd.clone())).try_send().is_err() {
                log::error!("link actor gone; record stays local only");
            }
        });
        // Make sure the catch-up above is acknowledged before the UI starts.
        let _ = link.ask(Flush).send().await;

        let positions = session.provider.channel_positions().to_vec();
        Ok(Joined {
            session,
            host,
            root: into.to_path_buf(),
            positions,
            connected: net.connected.clone(),
            error,
        })
    })
}

async fn sync_file(host: &RemoteActorRef<HostActor>, into: &Path, f: &FileEntry) -> Result<()> {
    if f.path.contains("..") || f.path.starts_with('/') {
        bail!("host sent unsafe path {}", f.path);
    }
    let dest = into.join(&f.path);
    if let Ok(meta) = dest.metadata() {
        if meta.len() == f.len && blake3_of(&dest)? == f.blake3 {
            return Ok(());
        }
    }
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = dest.with_extension("part");
    let mut out = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp)?;
    let mut hasher = blake3::Hasher::new();
    let mut offset = 0u64;
    while offset < f.len {
        let len = CHUNK.min(f.len - offset);
        let chunk: Bytes = host
            .ask(&FetchFile {
                path: f.path.clone(),
                offset,
                len,
            })
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("fetch {}: {e}", f.path))?;
        if chunk.0.is_empty() {
            bail!("host returned no data for {} at {offset}", f.path);
        }
        hasher.update(&chunk.0);
        out.write_all(&chunk.0)?;
        offset += chunk.0.len() as u64;
    }
    out.sync_all()?;
    if *hasher.finalize().as_bytes() != f.blake3 {
        bail!("{}: hash mismatch after download", f.path);
    }
    std::fs::rename(&tmp, &dest)?;
    println!("synced {} ({} KiB)", f.path, f.len >> 10);
    Ok(())
}

fn blake3_of(path: &Path) -> Result<[u8; 32]> {
    let mut h = blake3::Hasher::new();
    h.update_reader(File::open(path)?)?;
    Ok(*h.finalize().as_bytes())
}

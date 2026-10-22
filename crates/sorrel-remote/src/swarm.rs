//! One libp2p swarm per process: piying request-response on top of QUIC/TCP
//! (direct) or a circuit relay reached over WebSocket/TLS (proxy mode).
//!
//! piying's `bootstrap_on` hides the swarm and drops its events, so we build
//! and drive it ourselves: that is what lets us compose the relay client,
//! mDNS, and the peer allow-list, and watch connection state.

use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use futures::StreamExt;
use libp2p::{
    allow_block_list::{self, AllowedPeers},
    identity::Keypair,
    mdns,
    multiaddr::Protocol,
    noise, relay,
    swarm::{behaviour::toggle::Toggle, NetworkBehaviour, SwarmEvent},
    tcp, yamux, Multiaddr, PeerId, SwarmBuilder,
};
use tokio::sync::watch;

/// How the two machines reach each other.
#[derive(Clone, Debug)]
pub enum Mode {
    /// Host listens on `port` (QUIC + TCP). Guest dials `dial`, or finds the
    /// host via mDNS on the same LAN when `dial` is `None`.
    Direct { port: u16, dial: Option<Multiaddr> },
    /// Both sides dial this relay (`/dns4/host/tcp/443/wss/p2p/<RelayId>`);
    /// the host takes a reservation, the guest dials through the circuit.
    Relay(Multiaddr),
}

#[derive(Clone, Copy, Debug)]
pub enum Role {
    Host { guest: PeerId },
    Guest { host: PeerId },
}

/// Handle to the running swarm task.
pub struct Net {
    pub peer: PeerId,
    /// The one peer we talk to (guest for a host, host for a guest).
    pub remote: PeerId,
    /// `true` while at least one connection to `remote` is open.
    pub connected: watch::Receiver<bool>,
    /// Addresses we listen on, as libp2p reports them (per interface).
    pub listen_addrs: watch::Receiver<Vec<Multiaddr>>,
    /// Sender for `RemoteActorRef::for_peer_with_sender`; avoids relying on
    /// piying's process-global swarm (there may be two swarms in a test).
    pub swarm_tx: piying::remote::SwarmSender,
}

// allow_block_list has an uninhabited event type; the derive's match arm for
// it is unreachable by construction, hence the module-level allow.
#[allow(unreachable_code)]
mod behaviour {
    use super::*;

    #[derive(NetworkBehaviour)]
    pub(super) struct Behaviour {
        pub(super) allow: allow_block_list::Behaviour<AllowedPeers>,
        pub(super) relay: relay::client::Behaviour,
        pub(super) mdns: Toggle<mdns::tokio::Behaviour>,
        pub(super) piying: piying::remote::Behaviour,
    }
}
use behaviour::{Behaviour, BehaviourEvent};

/// `/p2p/<id>` component of a multiaddr.
pub fn peer_of(addr: &Multiaddr) -> Option<PeerId> {
    addr.iter().find_map(|p| match p {
        Protocol::P2p(id) => Some(id),
        _ => None,
    })
}

fn without_p2p(addr: &Multiaddr) -> Multiaddr {
    addr.iter()
        .filter(|p| !matches!(p, Protocol::P2p(_)))
        .collect()
}

fn messaging_config() -> piying::remote::messaging::Config {
    // A Split carries one u32 per moved spike; trace chunks are 1 MiB.
    piying::remote::messaging::Config::default()
        .with_request_timeout(Duration::from_secs(120))
        .with_request_size_maximum(64 << 20)
        .with_response_size_maximum(16 << 20)
}

pub async fn start(key: Keypair, mode: &Mode, role: Role) -> Result<Net> {
    let local = key.public().to_peer_id();
    let remote = match role {
        Role::Host { guest } => guest,
        Role::Guest { host } => host,
    };
    let relay_peer = match mode {
        Mode::Relay(a) => Some(peer_of(a).context("relay address must end in /p2p/<RelayId>")?),
        Mode::Direct { .. } => None,
    };
    let mut allow = allow_block_list::Behaviour::<AllowedPeers>::default();
    allow.allow_peer(remote);
    if let Some(r) = relay_peer {
        allow.allow_peer(r);
    }
    // mDNS is best-effort: sandboxes without multicast still work via --direct.
    let mdns = match mode {
        Mode::Direct { .. } => match mdns::tokio::Behaviour::new(mdns::Config::default(), local) {
            Ok(m) => Some(m),
            Err(e) => {
                log::warn!("mDNS unavailable ({e}); pass --direct on the guest");
                None
            }
        },
        Mode::Relay(_) => None,
    };

    let mut swarm = SwarmBuilder::with_existing_identity(key)
        .with_tokio()
        .with_tcp(
            tcp::Config::default().nodelay(true),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_quic()
        .with_dns()?
        .with_websocket(noise::Config::new, yamux::Config::default)
        .await?
        .with_relay_client(noise::Config::new, yamux::Config::default)?
        .with_behaviour(|_key, relay| Behaviour {
            allow,
            relay,
            mdns: Toggle::from(mdns),
            piying: piying::remote::Behaviour::new(local, messaging_config()),
        })?
        // libp2p's default is 10 s, which would redial between bursts.
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(30 * 60)))
        .build();

    // Harmless if another swarm in this process already set it; we route
    // through `swarm_tx` explicitly anyway.
    let _ = swarm.behaviour().piying.try_init_global();
    let swarm_tx = swarm.behaviour().piying.swarm_sender();

    match (mode, role) {
        (Mode::Direct { port, .. }, Role::Host { .. }) => {
            swarm.listen_on(format!("/ip4/0.0.0.0/udp/{port}/quic-v1").parse()?)?;
            swarm.listen_on(format!("/ip4/0.0.0.0/tcp/{port}").parse()?)?;
        }
        (Mode::Direct { dial, .. }, Role::Guest { host }) => {
            if let Some(addr) = dial {
                let addr = without_p2p(addr);
                swarm.add_peer_address(host, addr.clone());
                swarm.dial(addr.with(Protocol::P2p(host)))?;
            }
            // else: wait for mDNS to announce `host` (handled in the loop).
        }
        (Mode::Relay(relay_addr), Role::Host { .. }) => {
            swarm.listen_on(relay_addr.clone().with(Protocol::P2pCircuit))?;
        }
        (Mode::Relay(relay_addr), Role::Guest { host }) => {
            let via = relay_addr
                .clone()
                .with(Protocol::P2pCircuit)
                .with(Protocol::P2p(host));
            swarm.add_peer_address(host, via.clone());
            swarm.dial(via)?;
        }
    }

    let (conn_tx, connected) = watch::channel(false);
    let (addr_tx, listen_addrs) = watch::channel(Vec::<Multiaddr>::new());
    tokio::spawn(async move {
        let mut addrs = Vec::new();
        loop {
            match swarm.select_next_some().await {
                SwarmEvent::NewListenAddr { address, .. } => {
                    log::info!("listening on {address}");
                    addrs.push(address);
                    let _ = addr_tx.send(addrs.clone());
                }
                SwarmEvent::ConnectionEstablished { peer_id, .. } if peer_id == remote => {
                    log::info!("connected to {peer_id}");
                    let _ = conn_tx.send(true);
                }
                SwarmEvent::ConnectionClosed {
                    peer_id,
                    num_established: 0,
                    cause,
                    ..
                } if peer_id == remote => {
                    log::warn!("disconnected from {peer_id}: {cause:?}");
                    let _ = conn_tx.send(false);
                }
                SwarmEvent::Behaviour(BehaviourEvent::Mdns(mdns::Event::Discovered(list))) => {
                    for (peer, addr) in list {
                        if peer == remote {
                            log::info!("mDNS found {peer} at {addr}");
                            swarm.add_peer_address(peer, addr.clone());
                            let _ = swarm.dial(addr.with(Protocol::P2p(peer)));
                        }
                    }
                }
                SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                    log::warn!("dial {peer_id:?} failed: {error}");
                }
                SwarmEvent::Behaviour(BehaviourEvent::Relay(ev)) => log::debug!("relay: {ev:?}"),
                _ => {}
            }
        }
    });

    Ok(Net {
        peer: local,
        remote,
        connected,
        listen_addrs,
        swarm_tx,
    })
}

impl Net {
    /// Wait until `remote` is connected, or fail after `timeout`.
    pub async fn wait_connected(&mut self, timeout: Duration) -> Result<()> {
        tokio::time::timeout(timeout, self.connected.wait_for(|c| *c))
            .await
            .map_err(|_| anyhow!("no connection to {} within {timeout:?}", self.remote))?
            .map_err(|_| anyhow!("swarm task ended"))?;
        Ok(())
    }
}

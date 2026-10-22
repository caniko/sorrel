//! Circuit-relay-v2 server used by proxy mode. Listens on plain WebSocket;
//! TLS is terminated in front (caddy, nginx, Cloudflare), so guests dial
//! `/dns4/<host>/tcp/443/wss/p2p/<RelayId>`. Byte and duration limits are
//! lifted: this relay exists to carry traces.

use std::time::Duration;

use anyhow::{Context, Result};
use futures::StreamExt;
use libp2p::{
    identity::Keypair,
    noise, relay,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux, Multiaddr, PeerId, SwarmBuilder,
};
use tokio::sync::watch;

#[derive(NetworkBehaviour)]
struct Behaviour {
    relay: relay::Behaviour,
}

pub struct RelayHandle {
    pub peer: PeerId,
    /// Listen addresses including the trailing `/p2p/<RelayId>`.
    pub listen_addrs: watch::Receiver<Vec<Multiaddr>>,
}

fn config() -> relay::Config {
    relay::Config {
        max_reservations: 1024,
        max_reservations_per_peer: 8,
        reservation_duration: Duration::from_secs(3600),
        reservation_rate_limiters: Vec::new(),
        max_circuits: 1024,
        max_circuits_per_peer: 64,
        // The relay implementation treats 0 bytes as "no limit".
        max_circuit_duration: Duration::from_secs(24 * 3600),
        max_circuit_bytes: 0,
        circuit_src_rate_limiters: Vec::new(),
    }
}

/// Start serving on `listen` (e.g. `/ip4/0.0.0.0/tcp/4001/ws`) and return
/// once the socket is bound. The swarm runs on the current tokio runtime.
/// `external` are the addresses clients reach this relay at (e.g.
/// `/dns4/relay.example.org/tcp/443/wss`); reservations must carry at least
/// one, so when the list is empty the listen addresses are used.
pub async fn start(
    key: Keypair,
    listen: Multiaddr,
    external: Vec<Multiaddr>,
) -> Result<RelayHandle> {
    let peer = key.public().to_peer_id();
    let mut swarm = SwarmBuilder::with_existing_identity(key)
        .with_tokio()
        .with_tcp(
            tcp::Config::default().nodelay(true),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_websocket(noise::Config::new, yamux::Config::default)
        .await?
        .with_behaviour(|_| Behaviour {
            relay: relay::Behaviour::new(peer, config()),
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(30 * 60)))
        .build();
    // The server only offers the hop protocol once it believes it is publicly
    // reachable; behind a TLS-terminating proxy it never learns that, so say so.
    swarm
        .behaviour_mut()
        .relay
        .set_status(Some(relay::Status::Enable));
    for a in &external {
        swarm.add_external_address(a.clone());
    }
    let advertise_listen = external.is_empty();
    swarm
        .listen_on(listen.clone())
        .with_context(|| format!("listen on {listen}"))?;

    let (addr_tx, listen_addrs) = watch::channel(Vec::new());
    tokio::spawn(async move {
        let mut addrs = Vec::new();
        loop {
            match swarm.select_next_some().await {
                SwarmEvent::NewListenAddr { address, .. } => {
                    if advertise_listen {
                        swarm.add_external_address(address.clone());
                    }
                    let full = address.with(libp2p::multiaddr::Protocol::P2p(peer));
                    log::info!("relay listening on {full}");
                    addrs.push(full);
                    let _ = addr_tx.send(addrs.clone());
                }
                SwarmEvent::Behaviour(BehaviourEvent::Relay(ev)) => log::info!("{ev:?}"),
                SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                    log::info!("peer {peer_id} connected")
                }
                SwarmEvent::ConnectionClosed { peer_id, .. } => log::info!("peer {peer_id} left"),
                _ => {}
            }
        }
    });
    let mut rx = listen_addrs.clone();
    rx.wait_for(|a| !a.is_empty()).await.ok();
    Ok(RelayHandle { peer, listen_addrs })
}

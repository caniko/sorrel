//! Remote work protocol for Sorrel.
//!
//! A *host* machine that owns a Kilosort/phy dataset shares it with a *guest*
//! machine. The guest downloads the small phy artefacts, fetches raw-trace
//! byte ranges on demand into a sparse local mirror, and streams every
//! curation command back so the host's journal follows live. Transport is
//! piying's remote actors over libp2p (QUIC/TCP directly, or a circuit relay
//! reached over WebSocket/TLS for networks that only allow outbound 443).
//!
//! The journal is fast-forward only: exactly one writer (the guest) exists
//! while a share is active, and each record carries the `(baseline,
//! n_records)` cursor it was applied against.

pub mod guest;
pub mod host;
pub mod link;
pub mod mirror;
pub mod pairing;
pub mod protocol;
pub mod relay;
pub mod swarm;

pub use guest::{join, Joined};
pub use host::{serve, share, DatSource, Served, Serving, SessionServed};
pub use libp2p::Multiaddr;
pub use pairing::{Pairing, PairingCode};
pub use swarm::Mode;

use std::sync::OnceLock;

/// One tokio runtime per process; actors and the swarm live on it while the
/// egui thread stays synchronous.
pub fn rt() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("sorrel-remote")
            .build()
            .expect("tokio runtime")
    })
}

/// Block until Ctrl-C (SIGINT), so a headless host can return and drop its
/// listener instead of dying with a stale `mcp.json`.
pub fn wait_for_ctrl_c() {
    rt().block_on(async {
        let _ = tokio::signal::ctrl_c().await;
    });
}

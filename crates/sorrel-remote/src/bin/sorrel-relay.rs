//! `sorrel-relay [--listen MULTIADDR] [--identity FILE]`
//!
//! See `sorrel_remote::relay`. The identity file keeps the relay's PeerId
//! stable across restarts; it is created on first run readable by the current user only (mode 0600,
//! or an ACL on Windows).

use std::path::PathBuf;

use anyhow::{Context, Result};
use libp2p::{identity::Keypair, Multiaddr};

fn usage() -> ! {
    eprintln!(
        "usage: sorrel-relay [--listen MULTIADDR] [--identity FILE] [--external MULTIADDR]..."
    );
    eprintln!("  defaults: --listen /ip4/0.0.0.0/tcp/4001/ws  --identity ./relay.key");
    eprintln!(
        "  --external: public address clients dial, e.g. /dns4/relay.example.org/tcp/443/wss"
    );
    std::process::exit(2)
}

fn load_or_create_key(path: &PathBuf) -> Result<Keypair> {
    if path.exists() {
        let bytes = std::fs::read(path)?;
        return Keypair::from_protobuf_encoding(&bytes).context("decode relay identity");
    }
    let key = Keypair::generate_ed25519();
    let bytes = key.to_protobuf_encoding()?;
    sorrel_io::fsutil::write_private_new(path, &bytes)?;
    Ok(key)
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let mut listen: Multiaddr = "/ip4/0.0.0.0/tcp/4001/ws".parse()?;
    let mut identity = PathBuf::from("relay.key");
    let mut external: Vec<Multiaddr> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--listen" => listen = args.next().unwrap_or_else(|| usage()).parse()?,
            "--identity" => identity = args.next().unwrap_or_else(|| usage()).into(),
            "--external" => external.push(args.next().unwrap_or_else(|| usage()).parse()?),
            _ => usage(),
        }
    }
    let key = load_or_create_key(&identity)?;
    let relay = sorrel_remote::relay::start(key, listen, external).await?;
    println!("relay peer id: {}", relay.peer);
    println!(
        "guests and hosts use: <public wss multiaddr>/p2p/{}",
        relay.peer
    );
    for a in relay.listen_addrs.borrow().iter() {
        println!("listening on {a}");
    }
    tokio::signal::ctrl_c().await?;
    Ok(())
}

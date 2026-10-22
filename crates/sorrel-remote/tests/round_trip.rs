//! Host and guest in one process, over loopback QUIC and through an
//! in-process circuit relay. Fails if files, trace bytes, or journal records
//! stop matching, or if a wrong pairing code gets a connection.

use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use libp2p::{identity::Keypair, Multiaddr};
use sorrel_data::{baseline_hash, CurationCommand, Journal, PhyLabelOp, Session};
use sorrel_io::kilosort::{KilosortOpenParams, KilosortProvider};
use sorrel_io::{ClusterId, DataProvider, SampleIndex, TraceSamples};
use sorrel_remote::protocol::{Cursor, HostError, Record};
use sorrel_remote::swarm::{Mode, Role};
use sorrel_remote::{join, rt, serve, DatSource, Pairing, PairingCode, Served, SessionServed};

const N_CH: usize = 4;
const N_SAMPLES: usize = 40_000; // 320 KiB of int16: five 64 KiB blocks

fn write_npy_v1(path: &Path, descr: &str, shape_dim: usize, data: &[u8]) {
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape_dim},), }}");
    let prelude_len = 6 + 2 + 2 + dict.len() + 1;
    let pad = (64 - (prelude_len % 64)) % 64;
    let mut header = dict.into_bytes();
    header.resize(header.len() + pad, b' ');
    header.push(b'\n');
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"\x93NUMPY").unwrap();
    f.write_all(&[1u8, 0u8]).unwrap();
    f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
    f.write_all(&header).unwrap();
    f.write_all(data).unwrap();
}

/// Sample value = (sample index + channel) as i16, so any byte range is checkable.
fn dat_bytes() -> Vec<u8> {
    (0..N_SAMPLES)
        .flat_map(|t| (0..N_CH).map(move |c| ((t + c) as i16).to_le_bytes()))
        .flatten()
        .collect()
}

fn write_fixture(root: &Path) {
    let times: [u64; 6] = [10, 30, 50, 100, 150, 200];
    let clusters: [u32; 6] = [0, 0, 1, 2, 0, 1];
    let tb: Vec<u8> = times.iter().flat_map(|t| t.to_le_bytes()).collect();
    let cb: Vec<u8> = clusters.iter().flat_map(|c| c.to_le_bytes()).collect();
    write_npy_v1(&root.join("spike_times.npy"), "<i64", times.len(), &tb);
    write_npy_v1(
        &root.join("spike_clusters.npy"),
        "<u32",
        clusters.len(),
        &cb,
    );
    std::fs::write(root.join("recording.dat"), dat_bytes()).unwrap();
    std::fs::write(
        root.join("params.py"),
        "n_channels_dat = 4\nsample_rate = 1000.\ndtype = 'int16'\n",
    )
    .unwrap();
    std::fs::create_dir_all(root.join(".sorrel/cache")).unwrap();
    std::fs::write(root.join(".sorrel/cache/INDEX"), b"not shared").unwrap();
}

fn open_host(root: &Path) -> (Box<dyn Served>, DatSource, u64) {
    let p = KilosortProvider::open(root, KilosortOpenParams::default()).unwrap();
    let baseline = baseline_hash(&std::fs::read(root.join("spike_clusters.npy")).unwrap());
    let j =
        Journal::open_or_create(&root.join("sorrel.journal"), baseline, p.n_clusters()).unwrap();
    let dat = DatSource {
        path: p.dat_path().to_path_buf(),
        offset: p.dat_offset(),
        dtype: p.dtype(),
        n_channels: p.n_channels(),
        sample_rate: p.sample_rate(),
    };
    let mut s = Session::new(p, j);
    s.replay_journal().unwrap();
    (Box::new(SessionServed(s)), dat, baseline)
}

fn wait_until(what: &str, mut f: impl FnMut() -> bool) {
    let start = Instant::now();
    while !f() {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "timed out waiting for {what}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn loopback_quic(addrs: &[Multiaddr]) -> Option<Multiaddr> {
    addrs
        .iter()
        .find(|a| {
            let s = a.to_string();
            s.contains("/127.0.0.1/") && s.contains("/quic-v1")
        })
        .cloned()
}

fn host_listen_addr(serving: &sorrel_remote::Serving) -> Multiaddr {
    rt().block_on(async {
        let mut rx = serving.net.listen_addrs.clone();
        rx.wait_for(|a| loopback_quic(a).is_some()).await.unwrap();
        let addr = loopback_quic(&rx.borrow()).unwrap();
        addr
    })
}

fn exercise(mut joined: sorrel_remote::Joined, host_root: &Path, guest_root: &Path, baseline: u64) {
    for name in ["spike_times.npy", "spike_clusters.npy", "params.py"] {
        assert_eq!(
            std::fs::read(host_root.join(name)).unwrap(),
            std::fs::read(guest_root.join(name)).unwrap(),
            "{name} differs"
        );
    }
    assert!(
        !guest_root.join("recording.dat").exists(),
        "raw recording must not be copied"
    );
    // The guest builds its own cache; the host's marker must not have travelled.
    if let Ok(idx) = std::fs::read(guest_root.join(".sorrel/cache/INDEX")) {
        assert_ne!(idx, b"not shared", "cache must not be shared");
    }

    // Trace: first look is empty and schedules a fetch; then bytes match.
    let p = &joined.session.provider;
    let first = p.trace(SampleIndex(1_000), 50);
    assert!(first.samples.is_empty());
    assert!(p.pending_fetches() > 0);
    wait_until("trace fetch", || p.pending_fetches() == 0);
    let slice = p.trace(SampleIndex(1_000), 50);
    let expected: Vec<i16> = (1_000..1_050)
        .flat_map(|t| (0..N_CH).map(move |c| (t + c) as i16))
        .collect();
    match slice.samples {
        TraceSamples::I16(s) => assert_eq!(s, &expected[..]),
        other => panic!("unexpected dtype {other:?}"),
    }
    // Prefetch far away, then read without a stall.
    p.prefetch(&[(SampleIndex(30_000), 100)]);
    wait_until("prefetch", || p.pending_fetches() == 0);
    assert_eq!(p.trace(SampleIndex(30_000), 100).samples.len(), 100 * N_CH);

    // Records flow into the host journal in order.
    for c in 0..3u32 {
        joined
            .session
            .dispatch(CurationCommand::Relabel {
                cluster: ClusterId(c),
                op: PhyLabelOp::SetGood,
            })
            .unwrap();
    }
    let host_journal = host_root.join("sorrel.journal");
    wait_until("3 records on host", || {
        Journal::replay_at(&host_journal)
            .map(|r| r.len())
            .unwrap_or(0)
            == 3
    });
    assert_eq!(
        Journal::replay_at(&host_journal).unwrap().len(),
        Journal::replay_at(&guest_root.join("sorrel.journal"))
            .unwrap()
            .len()
    );

    // A stale cursor is refused, not applied.
    let stale = Record {
        cursor: Cursor {
            baseline,
            n_records: 0,
        },
        cmd: CurationCommand::Relabel {
            cluster: ClusterId(0),
            op: PhyLabelOp::SetNoise,
        },
    };
    let r = rt().block_on(joined.host.ask(&stale).send());
    assert!(
        matches!(
            r,
            Err(piying::error::RemoteSendError::HandlerError(
                HostError::Diverged { .. }
            ))
        ),
        "expected Diverged, got {r:?}"
    );
    assert_eq!(Journal::replay_at(&host_journal).unwrap().len(), 3);
}

fn dirs() -> (tempfile::TempDir, tempfile::TempDir) {
    let _ = env_logger::builder().is_test(true).try_init();
    let host = tempfile::tempdir().unwrap();
    write_fixture(host.path());
    (host, tempfile::tempdir().unwrap())
}

#[test]
fn direct_mode_round_trip() {
    let (host_dir, guest_dir) = dirs();
    let code = PairingCode::parse("abandon ability able about above absent").unwrap();
    let pairing = Pairing::derive(&code).unwrap();
    let (served, dat, baseline) = open_host(host_dir.path());
    let mode = Mode::Direct {
        port: 0,
        dial: None,
    };
    let serving = rt()
        .block_on(serve(
            served,
            host_dir.path().to_path_buf(),
            dat,
            &mode,
            &pairing,
        ))
        .unwrap();
    let addr = host_listen_addr(&serving);
    let joined = join(
        &code,
        &Mode::Direct {
            port: 0,
            dial: Some(addr),
        },
        guest_dir.path(),
    )
    .unwrap();
    exercise(joined, host_dir.path(), guest_dir.path(), baseline);
}

#[test]
fn wrong_code_is_refused() {
    let (host_dir, _guest_dir) = dirs();
    let code = PairingCode::parse("zoo zone zero youth young yellow").unwrap();
    let pairing = Pairing::derive(&code).unwrap();
    let (served, dat, _) = open_host(host_dir.path());
    let mode = Mode::Direct {
        port: 0,
        dial: None,
    };
    let serving = rt()
        .block_on(serve(
            served,
            host_dir.path().to_path_buf(),
            dat,
            &mode,
            &pairing,
        ))
        .unwrap();
    let addr = host_listen_addr(&serving);
    let wrong =
        Pairing::derive(&PairingCode::parse("wrong wrist write wrap world work").unwrap()).unwrap();
    let res = rt().block_on(async {
        let mut net = sorrel_remote::swarm::start(
            wrong.guest.clone(),
            &Mode::Direct {
                port: 0,
                dial: Some(addr),
            },
            Role::Guest {
                host: wrong.host_peer(),
            },
        )
        .await
        .unwrap();
        net.wait_connected(Duration::from_secs(3)).await
    });
    assert!(res.is_err(), "a guest with the wrong code must not connect");
}

#[test]
fn relay_mode_round_trip() {
    let (host_dir, guest_dir) = dirs();
    let relay = rt()
        .block_on(sorrel_remote::relay::start(
            Keypair::generate_ed25519(),
            "/ip4/127.0.0.1/tcp/0/ws".parse().unwrap(),
            Vec::new(),
        ))
        .unwrap();
    let relay_addr: Multiaddr = relay.listen_addrs.borrow()[0].clone();
    let code = PairingCode::parse("rely remain remember remind remove render").unwrap();
    let pairing = Pairing::derive(&code).unwrap();
    let (served, dat, baseline) = open_host(host_dir.path());
    let mode = Mode::Relay(relay_addr);
    let _serving = rt()
        .block_on(serve(
            served,
            host_dir.path().to_path_buf(),
            dat,
            &mode,
            &pairing,
        ))
        .unwrap();
    let joined = join(&code, &mode, guest_dir.path()).unwrap();
    exercise(joined, host_dir.path(), guest_dir.path(), baseline);
}

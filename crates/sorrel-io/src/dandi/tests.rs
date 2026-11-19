use super::*;
use crate::{DataProvider, NwbOpenParams, NwbProvider, SampleIndex};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, Write},
    mem::MaybeUninit,
    net::{TcpListener, TcpStream},
    sync::atomic::{AtomicBool, Ordering},
    thread::{self, JoinHandle},
};

struct Server {
    base: Url,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

type Headers = BTreeMap<String, String>;

impl Server {
    fn new(handler: impl Fn(&Url, &Headers, TcpStream) + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = Url::parse(&format!("http://{}/api/", listener.local_addr().unwrap())).unwrap();
        let origin = base.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let thread = thread::spawn(move || {
            for stream in listener.incoming() {
                if stopping.load(Ordering::Relaxed) {
                    break;
                }
                let mut reader = BufReader::new(stream.unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let mut parts = line.split_whitespace();
                assert_eq!(parts.next(), Some("GET"));
                let url = origin.join(parts.next().unwrap()).unwrap();
                let mut headers = Headers::new();
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line.trim().is_empty() {
                        break;
                    }
                    let (name, value) = line.split_once(':').unwrap();
                    headers.insert(name.to_ascii_lowercase(), value.trim().into());
                }
                assert_eq!(headers.get("accept-encoding").unwrap(), "identity");
                handler(&url, &headers, reader.into_inner());
            }
        });
        Self {
            base,
            stop,
            thread: Some(thread),
        }
    }

    fn api(&self, timeout: Duration) -> Api {
        Api {
            base: self.base.clone(),
            agent: http_range::agent(timeout),
            key: Some("private-key".into()),
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = TcpStream::connect(("127.0.0.1", self.base.port().unwrap()));
        self.thread.take().unwrap().join().unwrap();
    }
}

fn reply(mut stream: TcpStream, status: u16, headers: &str, body: &[u8]) {
    let _ = write!(
        stream,
        "HTTP/1.1 {status} test\r\nConnection: close\r\n{headers}\r\n"
    );
    let _ = stream.write_all(body);
}

fn json_reply(stream: TcpStream, value: Value) {
    let body = value.to_string();
    reply(
        stream,
        200,
        &format!("Content-Length: {}\r\n", body.len()),
        body.as_bytes(),
    );
}

#[derive(Clone, Copy, Debug)]
enum Fault {
    None,
    Ignored,
    Short,
    Wrong,
    Timeout,
    Forbidden,
    Changed,
    Total,
}

fn serve_range(
    stream: TcpStream,
    headers: &Headers,
    data: &[u8],
    fault: Fault,
    etag: &str,
) -> usize {
    assert!(!headers.contains_key("authorization"));
    let (start, stop) = headers["range"]
        .strip_prefix("bytes=")
        .unwrap()
        .split_once('-')
        .unwrap();
    let (start, stop): (usize, usize) = (start.parse().unwrap(), stop.parse().unwrap());
    if let Some(tag) = headers.get("if-match") {
        assert_eq!(tag, etag);
    }
    let body = &data[start..=stop];
    match fault {
        Fault::Timeout => {
            thread::sleep(Duration::from_millis(120));
            return 0;
        }
        Fault::Forbidden => {
            reply(stream, 403, "Content-Length: 0\r\n", &[]);
            return 0;
        }
        Fault::Ignored => {
            reply(stream, 200, "Content-Length: 999999999\r\n", &[]);
            return 0;
        }
        _ => {}
    }
    let total = data.len() + usize::from(matches!(fault, Fault::Total));
    let header_stop = stop + usize::from(matches!(fault, Fault::Wrong));
    let etag = if matches!(fault, Fault::Changed) {
        "\"changed\""
    } else {
        etag
    };
    let sent = if matches!(fault, Fault::Short) {
        &body[..body.len() - 1]
    } else {
        body
    };
    reply(stream, 206, &format!("Content-Range: bytes {start}-{header_stop}/{total}\r\nContent-Length: {}\r\nETag: {etag}\r\n", body.len()), sent);
    sent.len()
}

fn range_case(fault: Fault, expected: &str) {
    let server = Server::new(move |_, headers, stream| {
        assert_eq!(headers["if-match"], "\"initial\"");
        serve_range(stream, headers, &[7; 16], fault, "\"initial\"");
    });
    let object = http_range::Object::new(
        0,
        http_range::Cache::new(),
        Source {
            label: "example NWB".into(),
            agent: server
                .api(Duration::from_millis(if matches!(fault, Fault::Timeout) {
                    30
                } else {
                    1000
                }))
                .agent,
            url: Some(server.base.join("/storage?secret=signature").unwrap()),
            refresh: None,
        },
        16,
        "\"initial\"".into(),
    );
    let object = Arc::new(object);
    let err = format!(
        "{:#}",
        object
            .read_uninit(4, &mut [MaybeUninit::uninit(); 5], false)
            .unwrap_err()
    );
    assert!(
        err.contains(expected)
            && err.contains("example NWB: bytes 0-15")
            && !err.contains("signature"),
        "{err}"
    );
}

#[test]
fn ignored_range() {
    range_case(Fault::Ignored, "expected HTTP 206");
}
#[test]
fn short_body() {
    range_case(Fault::Short, "range body");
}
#[test]
fn wrong_content_range() {
    range_case(Fault::Wrong, "unexpected Content-Range");
}
#[test]
fn changed_total() {
    range_case(Fault::Total, "total changed");
}
#[test]
fn timeout() {
    range_case(Fault::Timeout, "timed out");
}
#[test]
fn changed_etag() {
    range_case(Fault::Changed, "ETag changed");
}
#[test]
fn forbidden_without_refresh() {
    range_case(Fault::Forbidden, "got 403");
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../release/fixtures/nwb")
        .join(name)
}

fn objects(path: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, path: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for entry in fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let key = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .replace('\\', "/");
                out.insert(key, fs::read(&path).unwrap());
            }
        }
    }
    let mut out = BTreeMap::new();
    if path.is_dir() {
        walk(path, path, &mut out);
    } else {
        out.insert("blob".into(), fs::read(path).unwrap());
    }
    out
}

fn remote_fixture(
    name: &'static str,
    draft: bool,
) -> (Server, Arc<std::sync::atomic::AtomicUsize>) {
    use std::sync::atomic::AtomicUsize;
    let data = objects(&fixture(name));
    let blob = data.get("blob");
    let asset = json!({"asset_id":"asset-id", "path":format!("test/{name}"), "size":blob.map_or(0, Vec::len), "blob":blob.map(|_| "blob-id"), "zarr":blob.is_none().then_some("zarr-id")});
    let version = if draft { "draft" } else { "0.42.17" };
    let downloads = AtomicUsize::new(0);
    let bytes = Arc::new(AtomicUsize::new(0));
    let count = bytes.clone();
    let server = Server::new(move |url, headers, stream| {
        let query: BTreeMap<_, _> = url.query_pairs().collect();
        if url.path().starts_with("/api/") {
            assert_eq!(headers["authorization"], "Token private-key");
        }
        match url.path() {
            "/api/dandisets/000123/" => json_reply(
                stream,
                json!({"most_recent_published_version": (!draft).then_some(json!({"version": version})), "draft_version": {"version":"draft"}}),
            ),
            "/api/assets/asset-id/info/" => json_reply(stream, asset.clone()),
            "/api/assets/asset-id/download/" => {
                let n = downloads.fetch_add(1, Ordering::Relaxed);
                reply(
                    stream,
                    302,
                    &format!("Location: /storage?key=blob&signature={n}\r\nContent-Length: 0\r\n"),
                    &[],
                );
            }
            "/api/zarr/zarr-id/files/" if query.contains_key("download") => {
                assert_eq!(query["download"], "true");
                let n = downloads.fetch_add(1, Ordering::Relaxed);
                let mut storage = url.join("/storage").unwrap();
                storage
                    .query_pairs_mut()
                    .append_pair("key", &query["prefix"])
                    .append_pair("signature", &n.to_string());
                reply(
                    stream,
                    302,
                    &format!(
                        "Location: {}\r\nContent-Length: 0\r\n",
                        &storage[Position::BeforePath..]
                    ),
                    &[],
                );
            }
            "/api/zarr/zarr-id/files/" => {
                let second = query.contains_key("after");
                let results: Vec<_> = data
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| (*idx >= 7) == second)
                    .map(|(_, (key, value))| json!({"Key":key,"ETag":"initial","Size":value.len()}))
                    .collect();
                json_reply(
                    stream,
                    json!({"next":(!second).then_some("?after=7"), "results":results}),
                );
            }
            "/storage" => {
                let fault = if query["signature"] == "0" {
                    Fault::Forbidden
                } else {
                    Fault::None
                };
                count.fetch_add(
                    serve_range(
                        stream,
                        headers,
                        &data[query["key"].as_ref()],
                        fault,
                        "\"initial\"",
                    ),
                    Ordering::Relaxed,
                );
            }
            path => {
                assert_eq!(
                    path,
                    format!("/api/dandisets/000123/versions/{version}/assets/")
                );
                let requested = format!("test/{name}");
                assert_eq!(query["path"], requested);
                if query.contains_key("page") {
                    json_reply(stream, json!({"next":null, "results":[asset]}));
                } else {
                    let mut prefix_match = asset.clone();
                    prefix_match["path"] = json!(format!("{requested}.backup"));
                    let mut next = url.clone();
                    next.query_pairs_mut().append_pair("page", "2");
                    json_reply(
                        stream,
                        json!({"next":next.as_str(), "results":[prefix_match]}),
                    );
                }
            }
        }
    });
    (server, bytes)
}

fn parity(name: &'static str, draft: bool) -> Result<()> {
    use crate::{ClusterId, HasGeometry};
    use std::sync::atomic::Ordering::Relaxed;
    let (server, bytes) = remote_fixture(name, draft);
    let local = NwbProvider::open(fixture(name));
    let remote = NwbProvider::open_resolved(
        server
            .api(Duration::from_secs(2))
            .resolve(&format!("dandi://dandi/000123/test/{name}"))?,
        NwbOpenParams::default(),
    );
    if let Err(local) = local {
        let remote = remote
            .err()
            .expect("remote accepted a refused local fixture");
        assert_eq!(format!("{remote:#}"), format!("{local:#}"));
        return Ok(());
    }
    let (local, remote) = (local?, remote?);
    assert_eq!(local.n_clusters(), remote.n_clusters());
    assert_eq!(local.sample_rate(), remote.sample_rate());
    assert_eq!(local.n_samples(), remote.n_samples());
    assert_eq!(local.channel_positions(), remote.channel_positions());
    assert_eq!(local.channel_shanks(), remote.channel_shanks());
    assert_eq!(local.initial_labels(), remote.initial_labels());
    for k in 0..local.n_clusters() {
        assert_eq!(
            local.unit_id_json(ClusterId(k)),
            remote.unit_id_json(ClusterId(k))
        );
        assert_eq!(
            local.spike_times(ClusterId(k)),
            remote.spike_times(ClusterId(k))
        );
    }
    if local.n_samples().0 > 64 {
        let start = SampleIndex(9);
        let expected = local.read_trace(start, 32)?.samples;
        assert!(matches!(expected, crate::TraceSamples::OwnedF32(_)));
        let crate::TraceSamples::OwnedF32(expected) = expected else {
            unreachable!()
        };
        let crate::TraceSamples::OwnedF32(actual) = remote.read_trace(start, 32)?.samples else {
            unreachable!()
        };
        assert_eq!(actual, expected);
        let before = bytes.load(Relaxed);
        remote.read_trace(start, 32)?;
        assert_eq!(before, bytes.load(Relaxed));
    }
    Ok(())
}

#[test]
fn zarr_parity() -> Result<()> {
    for name in [
        "good.nwb.zarr",
        "float32.nwb.zarr",
        "curated_units.nwb.zarr",
        "irregular_timestamps.nwb.zarr",
        "multiple_series.nwb.zarr",
    ] {
        parity(name, true)?;
    }
    Ok(())
}

#[test]
#[cfg(feature = "hdf5")]
fn blob_parity() -> Result<()> {
    for name in [
        "good.nwb",
        "float32.nwb",
        "curated_units.nwb",
        "irregular_timestamps.nwb",
        "multiple_series.nwb",
    ] {
        parity(name, false)?;
    }
    Ok(())
}

#[test]
#[cfg(feature = "hdf5")]
#[ignore = "requires SORREL_DANDI_NETWORK=1 and public DANDI/S3 access"]
fn public_blob() -> Result<()> {
    if std::env::var("SORREL_DANDI_NETWORK").as_deref() != Ok("1") {
        return Ok(());
    }
    let (provider, _) = NwbProvider::open_remote(
        "dandi://dandi/000006@0.220126.1855/sub-anm369962/sub-anm369962_ses-20170316.nwb",
        // The public fixture has Units but no recording clock. Use an
        // explicit test grid, not an inferred acquisition sampling rate.
        NwbOpenParams {
            sample_rate: Some(1000.0),
            ..Default::default()
        },
    )?;
    assert!(provider.n_clusters() > 0);
    eprintln!(
        "public DANDI NWB: {} units, {} channels, {} Hz",
        provider.n_clusters(),
        provider.n_channels(),
        provider.sample_rate()
    );
    Ok(())
}

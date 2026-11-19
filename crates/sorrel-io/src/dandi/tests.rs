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
            url: Some((
                server.base.join("/storage?secret=signature").unwrap(),
                false,
            )),
            refresh: None,
            identity: None,
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
    range_case(Fault::Forbidden, "got HTTP 403");
}

#[test]
#[cfg(feature = "hdf5")]
fn blob_refresh_pins_effective_object_and_seal() -> Result<()> {
    use std::sync::atomic::AtomicUsize;
    let refresh = Arc::new(AtomicUsize::new(0));
    let current = refresh.clone();
    let data = vec![7; 3 * 256 * 1024];
    let server = Server::new(move |url, headers, stream| {
        let signature = url.query_pairs().find(|(k, _)| k == "X-Amz-Signature");
        let expired =
            signature.is_some_and(|(_, v)| v != current.load(Ordering::Relaxed).to_string());
        serve_range(
            stream,
            headers,
            &data,
            if expired {
                Fault::Forbidden
            } else {
                Fault::None
            },
            "\"initial\"",
        );
    });
    let api = server.api(Duration::from_secs(2));
    let refreshed_url = |revision| server.base.join(&format!("/object?versionId={revision}"));
    let base = refreshed_url("one")?;
    let current = refresh.clone();
    let resolved = Api::blob(
        Source {
            label: "asset".into(),
            agent: api.agent.clone(),
            url: None,
            identity: None,
            refresh: Some(Box::new(move || {
                let n = current.load(Ordering::Relaxed);
                let revision = if n < 2 { "one" } else { "two" };
                Ok(base.join(&format!("?X-Amz-Signature={n}&versionId={revision}"))?)
            })),
        },
        None,
        "asset-id".into(),
        Some("blob-id".into()),
        PathBuf::new(),
    )?;
    let Store::Blob(obj) = resolved.store else {
        unreachable!()
    };
    let read = |offset| obj.read_uninit(offset, &mut [MaybeUninit::uninit(); 1], false);
    read(0)?;
    refresh.store(1, Ordering::Relaxed);
    read(256 * 1024)?; // Same object, new signature.
    refresh.store(2, Ordering::Relaxed);
    assert!(format!("{:#}", read(2 * 256 * 1024).unwrap_err()).contains("object identity changed"));
    read(0)?; // Existing cache entries still belong to the pinned revision.
    let first = api.resolve(refreshed_url("one")?.as_str())?;
    assert_ne!(
        first.identity,
        api.resolve(refreshed_url("two")?.as_str())?.identity
    );
    Ok(())
}

#[test]
#[cfg(feature = "hdf5")]
fn direct_query_pins_resource_across_redirects() -> Result<()> {
    use std::sync::atomic::AtomicBool;
    const BLOCK: usize = 256 * 1024;
    let redirected = Arc::new(AtomicBool::new(false));
    let flag = redirected.clone();
    let a = vec![1; 2 * BLOCK];
    let b = vec![2; 2 * BLOCK];
    let server = Server::new(move |url, headers, stream| {
        let other = url.query() == Some("Signature=B");
        if !other && flag.load(Ordering::Relaxed) {
            reply(
                stream,
                302,
                "Location: /data.nwb?Signature=B\r\nContent-Length: 0\r\n",
                &[],
            );
        } else {
            serve_range(
                stream,
                headers,
                if other { &b } else { &a },
                Fault::None,
                "\"initial\"",
            );
        }
    });
    let api = server.api(Duration::from_secs(2));
    let first = api.resolve(server.base.join("/data.nwb?Signature=A")?.as_str())?;
    let second = api.resolve(server.base.join("/data.nwb?Signature=B")?.as_str())?;
    assert_ne!(first.identity, second.identity);
    let Store::Blob(object) = first.store else {
        unreachable!()
    };
    let read = |offset| object.read_uninit(offset, &mut [MaybeUninit::uninit(); 1], false);
    read(0)?;
    redirected.store(true, Ordering::Relaxed);
    let error = format!("{:#}", read(BLOCK as u64).unwrap_err());
    assert!(error.contains("object identity changed"), "{error}");
    assert!(!error.contains("Signature="), "{error}");
    Ok(())
}

#[test]
fn distant_seeks_coalesce_and_share_cached_reads() -> Result<()> {
    use std::sync::atomic::AtomicUsize;
    const SIZE: usize = 12 * 1024 * 1024;
    let gets = Arc::new(AtomicUsize::new(0));
    let bytes = Arc::new(AtomicUsize::new(0));
    let (gets_on_server, bytes_on_server) = (gets.clone(), bytes.clone());
    let data = vec![19; SIZE];
    let server = Server::new(move |_, headers, stream| {
        gets_on_server.fetch_add(1, Ordering::Relaxed);
        bytes_on_server.fetch_add(
            serve_range(stream, headers, &data, Fault::None, "\"initial\""),
            Ordering::Relaxed,
        );
    });
    let object = Arc::new(http_range::Object::new(
        0,
        http_range::Cache::new(),
        Source {
            label: "large asset".into(),
            agent: server.api(Duration::from_secs(2)).agent,
            url: Some((server.base.join("/storage").unwrap(), false)),
            refresh: None,
            identity: None,
        },
        SIZE as u64,
        "\"initial\"".into(),
    ));
    let read = |offset| -> Result<()> {
        let mut out = vec![MaybeUninit::uninit(); 768 * 1024];
        object.read_uninit(offset, &mut out, false)?;
        // All bytes were initialized by the successful read.
        assert!(out.into_iter().all(|b| unsafe { b.assume_init() } == 19));
        Ok(())
    };
    read(0)?;
    thread::scope(|scope| -> Result<()> {
        let other = scope.spawn(|| read(10 * 1024 * 1024));
        read(10 * 1024 * 1024)?;
        other.join().unwrap()
    })?;
    let before = gets.load(Ordering::Relaxed);
    assert_eq!(before, 2);
    read(0)?;
    assert_eq!(gets.load(Ordering::Relaxed), before);
    assert!(bytes.load(Ordering::Relaxed) < SIZE / 4);
    Ok(())
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../release/fixtures/nwb")
        .join(name)
}

#[test]
#[cfg(feature = "hdf5")]
fn paired_locators_share_implicit_release_and_identical_store() -> Result<()> {
    use std::sync::atomic::AtomicUsize;
    let versions = Arc::new(AtomicUsize::new(0));
    let listings = Arc::new(AtomicUsize::new(0));
    let (lookups, assets) = (versions.clone(), listings.clone());
    let server = Server::new(move |url, headers, stream| match url.path() {
        "/api/dandisets/000123/" => {
            let version = if lookups.fetch_add(1, Ordering::Relaxed) == 0 {
                "0.42.1"
            } else {
                "0.42.2"
            };
            json_reply(
                stream,
                json!({"most_recent_published_version": {"version": version}, "draft_version": {"version": "draft"}}),
            );
        }
        "/api/dandisets/000123/versions/0.42.1/assets/"
        | "/api/dandisets/000123/versions/0.42.2/assets/" => {
            assets.fetch_add(1, Ordering::Relaxed);
            let path = url
                .query_pairs()
                .find(|(name, _)| name == "path")
                .unwrap()
                .1;
            json_reply(
                stream,
                json!({"next": null, "results": [{"asset_id": "asset", "blob": "blob", "size": 8, "path": path}]}),
            );
        }
        "/api/assets/asset/download/" => {
            reply(stream, 302, "Location: /blob\r\nContent-Length: 0\r\n", &[])
        }
        "/blob" => {
            serve_range(stream, headers, b"12345678", Fault::None, "\"one\"");
        }
        _ => panic!("unexpected URL: {url}"),
    });
    let api = server.api(Duration::from_secs(2));
    let root = Some("dandi://dandi/000123/sorting.nwb");
    let dat = Some("dandi://dandi/000123/recording.nwb");
    for (asset, path) in api
        .resolve_pair(root, dat)?
        .into_iter()
        .zip(["sorting.nwb", "recording.nwb"])
    {
        let asset = asset.unwrap();
        assert_eq!(
            asset.state_path(),
            Path::new("000123/0.42.1").join(blake3::hash(path.as_bytes()).to_hex().as_str())
        );
    }
    assert_eq!(versions.load(Ordering::Relaxed), 1);
    assert_eq!(listings.load(Ordering::Relaxed), 2);

    let [a, b] = api.resolve_pair(root, root)?.map(Option::unwrap);
    assert_eq!(a.state_path().parent(), Some(Path::new("000123/0.42.2")));
    let (Store::Blob(a), Store::Blob(b)) = (a.store, b.store) else {
        panic!("expected pinned blobs");
    };
    assert!(Arc::ptr_eq(&a, &b));
    assert_eq!(versions.load(Ordering::Relaxed), 2);
    assert_eq!(listings.load(Ordering::Relaxed), 3);
    Ok(())
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
    remote_fixture_without(name, draft, None)
}

fn remote_fixture_without(
    name: &'static str,
    draft: bool,
    omitted: Option<&'static str>,
) -> (Server, Arc<std::sync::atomic::AtomicUsize>) {
    use std::sync::atomic::AtomicUsize;
    let data = objects(&fixture(name));
    let blob = data.get("blob");
    let asset = json!({"asset_id":"asset-id", "path":format!("test/{name}"), "size":blob.map_or(0, Vec::len), "blob":blob.map(|_| "blob-id"), "zarr":blob.is_none().then_some("zarr-id")});
    // Reference zarr_checksum 0.4.7, with the mock ETag "initial" on every file.
    let checksum = match name {
        "good.nwb.zarr" => "e30a3acf1096e11521a59d4e87ba1692-103--42756",
        "curated_units.nwb.zarr" => "d4aabfd7d190242c4710b2a4cdf1ba99-65--11884",
        "float32.nwb.zarr" => "da97f19336aba618bfd758992f7b173e-90--16048",
        "irregular_timestamps.nwb.zarr" => "683159f77981e56c45db8835c7be677f-93--18364",
        "multiple_series.nwb.zarr" => "c9e6511e852d0a0bdda3186910cab7e9-116--20707",
        "units_only.nwb.zarr" => "7ea57689edf3a39bde84a2d3b26b6ff7-91--15980",
        _ => "",
    };
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
                    &format!("Location: /storage?key=blob&Signature={n}\r\nContent-Length: 0\r\n"),
                    &[],
                );
            }
            "/api/zarr/zarr-id/" => json_reply(stream, json!({"checksum": checksum})),
            "/api/zarr/zarr-id/files/" if query.contains_key("download") => {
                assert_eq!(query["download"], "true");
                let n = downloads.fetch_add(1, Ordering::Relaxed);
                let mut storage = url.join("/storage").unwrap();
                storage
                    .query_pairs_mut()
                    .append_pair("key", &query["prefix"])
                    .append_pair("Signature", &n.to_string());
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
                    .filter(|(idx, (key, _))| {
                        (*idx >= 7) == second && Some(key.as_str()) != omitted
                    })
                    .map(|(_, (key, value))| json!({"Key":key,"ETag":"initial","Size":value.len()}))
                    .collect();
                json_reply(
                    stream,
                    json!({"next":(!second).then_some("?after=7"), "results":results}),
                );
            }
            "/storage" => {
                let fault = if query["Signature"] == "0" {
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
    let remote = NwbProvider::open_pinned(
        Path::new("remote NWB"),
        NwbOpenParams::default(),
        Some(
            server
                .api(Duration::from_secs(2))
                .resolve(&format!("dandi://dandi/000123/test/{name}"))?,
        ),
        None,
    );
    if let Err(local) = local {
        let remote = remote
            .err()
            .expect("remote accepted a refused local fixture");
        assert_eq!(format!("{remote:#}"), format!("{local:#}"));
        return Ok(());
    }
    let (local, remote) = (local?, remote?);
    if name == "units_only.nwb.zarr" {
        let trials = remote.read_remote_trials()?;
        assert!(!trials.is_empty());
        assert_eq!(trials, crate::Events::read(&fixture(name))?);
    }
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
fn incomplete_zarr_listing_is_refused() {
    let (server, _) = remote_fixture_without(
        "good.nwb.zarr",
        true,
        Some("acquisition/ElectricalSeriesRaw/data/0.0"),
    );
    let error = server
        .api(Duration::from_secs(2))
        .resolve("dandi://dandi/000123@draft/test/good.nwb.zarr")
        .err()
        .expect("accepted a listing missing a real chunk");
    assert!(format!("{error:#}").contains("Zarr listing checksum mismatch"));
}

#[test]
fn zarr_parity() -> Result<()> {
    for name in [
        "good.nwb.zarr",
        "float32.nwb.zarr",
        "curated_units.nwb.zarr",
        "irregular_timestamps.nwb.zarr",
        "multiple_series.nwb.zarr",
        "units_only.nwb.zarr",
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
fn pinned_remote_dat_supplies_samples_without_resolving_again() -> Result<()> {
    let units = fixture("units_only.nwb");
    let read = |provider: NwbProvider| {
        Ok::<_, anyhow::Error>(
            provider
                .read_trace(SampleIndex(0), 1)?
                .samples
                .to_f32_centred(),
        )
    };
    for name in ["good.nwb", "uint16.nwb"] {
        let (server, _) = remote_fixture(name, false);
        let raw = server
            .api(Duration::from_secs(2))
            .resolve(&format!("dandi://dandi/000123/test/{name}"))?;
        let expected = NwbProvider::open_with(
            &units,
            NwbOpenParams {
                raw_path: Some(&fixture(name)),
                ..Default::default()
            },
        )?;
        // This locator is unusable. Opening must use only the supplied store.
        let pinned = NwbProvider::open_pinned(
            &units,
            NwbOpenParams {
                raw_path: Some(Path::new("dandi://unresolvable")),
                ..Default::default()
            },
            None,
            Some(raw),
        )?;
        assert_eq!(read(pinned)?, read(expected)?);
    }
    Ok(())
}

#[test]
fn direct_dandi_https_asset_endpoint_resolves_zarr() -> Result<()> {
    let (server, _) = remote_fixture("good.nwb.zarr", false);
    let api = server.api(Duration::from_secs(2));
    let direct = api.resolve(server.base.join("assets/asset-id/download/")?.as_str())?;
    let by_path = api.resolve("dandi://dandi/000123/test/good.nwb.zarr")?;
    assert_eq!(direct.identity, by_path.identity);
    let provider = NwbProvider::open_pinned(
        Path::new("remote NWB"),
        NwbOpenParams::default(),
        Some(direct),
        None,
    )?;
    assert!(provider.n_clusters() > 0);
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

#[test]
#[cfg(feature = "hdf5")]
#[ignore = "requires SORREL_DANDI_NETWORK=1 and public DANDI/S3 access"]
fn public_zarr() -> Result<()> {
    use zarrs::storage::{ReadableStorageTraits, StoreKey};

    if std::env::var("SORREL_DANDI_NETWORK").as_deref() != Ok("1") {
        return Ok(());
    }
    // This small intracellular electrophysiology benchmark lacks Units and
    // also has non-JSON consolidated metadata. Verify that we can resolve and
    // range-read the real archive before the shared NWB reader refuses it.
    let resolved = resolve(concat!(
        "dandi://dandi/000719@draft/icephys_DS_11_21_24/",
        "sub-1214579789_ses-1214621812_icephys_DirectoryStore.nwb.zarr"
    ))?;
    let Store::Zarr(store) = &resolved.store else {
        bail!("expected a Zarr asset");
    };
    let attrs = store
        .get(&StoreKey::new(".zattrs")?)?
        .context("missing NWB-Zarr root attributes")?;
    let attrs: Value = serde_json::from_slice(&attrs)?;
    assert_eq!(attrs["neurodata_type"], "NWBFile");
    let error = NwbProvider::open_pinned(
        Path::new("remote NWB"),
        NwbOpenParams::default(),
        Some(resolved),
        None,
    )
    .err()
    .context("unexpectedly opened the unsupported benchmark")?;
    let message = format!("{error:#}");
    assert!(
        message.contains("parse .zmetadata") || message.contains("/units"),
        "{message}"
    );
    Ok(())
}

//! Identity-pinned, read-only HTTP range I/O. URLs and HTTP-library errors
//! never escape this module's request boundary (they can contain signatures).
use anyhow::{anyhow, bail, ensure, Context, Result};
use lru::LruCache;
use std::io::Read;
use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use ureq::http::{Response, StatusCode};
use url::{Position, Url};

const BLOCK: u64 = 256 * 1024;
const RUN: u64 = 8 * BLOCK;

pub(crate) fn agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .max_redirects(0)
        .http_status_as_error(false)
        .timeout_global(Some(timeout))
        .build()
        .into()
}

pub(crate) fn request(
    agent: &ureq::Agent,
    url: &Url,
    headers: &[(&str, &str)],
) -> Result<Response<ureq::Body>> {
    for attempt in 0..3 {
        match request_once(agent, url, headers) {
            Ok(res) if attempt < 2 && transient_status(res.status()) => {}
            Ok(res) => return Ok(res),
            Err(ureq::Error::Timeout(_) | ureq::Error::Io(_)) if attempt < 2 => {}
            Err(ureq::Error::Timeout(_)) => bail!("HTTP GET timed out"),
            Err(_) => bail!("HTTP GET failed"),
        }
        backoff(attempt);
    }
    unreachable!()
}

fn request_once(
    agent: &ureq::Agent,
    url: &Url,
    headers: &[(&str, &str)],
) -> std::result::Result<Response<ureq::Body>, ureq::Error> {
    let mut req = agent
        .get(url.as_str())
        .header("Accept-Encoding", "identity");
    for &(key, value) in headers {
        req = req.header(key, value);
    }
    req.call()
}

fn transient_status(status: StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 429 | 500 | 502 | 503 | 504)
}

fn backoff(attempt: u32) {
    std::thread::sleep(Duration::from_millis(50 << attempt));
}

pub(crate) fn validate_url(url: &Url) -> Result<()> {
    let https = url.scheme() == "https";
    #[cfg(test)]
    let https = https || (url.scheme() == "http" && url.host_str() == Some("127.0.0.1"));
    ensure!(
        https
            && url.host().is_some()
            && url.username().is_empty()
            && url.password().is_none()
            && url.fragment().is_none(),
        "expected an HTTPS URL without credentials or fragment"
    );
    Ok(())
}

pub(crate) fn redirect(url: &Url, res: &Response<ureq::Body>) -> Result<Url> {
    ensure!(
        matches!(res.status().as_u16(), 301 | 302 | 303 | 307 | 308),
        "expected a download redirect, got HTTP {}",
        res.status().as_u16()
    );
    let location = res
        .headers()
        .get("location")
        .and_then(|h| h.to_str().ok())
        .context("missing redirect Location")?;
    let next = url
        .join(location)
        .map_err(|_| anyhow!("invalid redirect Location"))?;
    validate_url(&next)?;
    Ok(next)
}

pub(crate) fn strong_etag(value: &str) -> Result<&str> {
    ensure!(
        value.len() >= 2
            && value.starts_with('"')
            && value.ends_with('"')
            && value[1..value.len() - 1]
                .bytes()
                .all(|b| b == 0x21 || (0x23..=0x7e).contains(&b)),
        "a strong ETag is required"
    );
    Ok(value)
}

fn object_identity(url: &Url, from_dandi_api: bool) -> String {
    let dandi_storage = url.host_str() == Some("dandiarchive.s3.amazonaws.com")
        && url.port_or_known_default() == Some(443);
    #[cfg(test)]
    let dandi_storage = dandi_storage || url.host_str() == Some("127.0.0.1");
    if !(from_dandi_api && dandi_storage) {
        return url.as_str().to_owned();
    }
    let mut identity = url[..Position::AfterPath].to_owned();
    if let Some(query) = url.query() {
        let mut separator = '?';
        for field in query.split('&') {
            let key = field.split_once('=').map_or(field, |(key, _)| key);
            if key.starts_with("X-Amz-")
                || matches!(
                    key,
                    "Signature"
                        | "Expires"
                        | "AWSAccessKeyId"
                        | "response-cache-control"
                        | "response-content-disposition"
                        | "response-content-encoding"
                        | "response-content-language"
                        | "response-content-type"
                        | "response-expires"
                )
            {
                continue;
            }
            // Preserve raw bytes, ordering, and duplicates for unknown parameters.
            identity.push(separator);
            identity.push_str(field);
            separator = '&';
        }
    }
    identity
}

pub(crate) type Refresh = Box<dyn Fn() -> Result<Url> + Send + Sync>;

pub(crate) struct Source {
    pub label: String,
    pub agent: ureq::Agent,
    // The flag marks URLs issued directly by the DANDI API, not by storage redirects.
    pub url: Option<(Url, bool)>,
    pub refresh: Option<Refresh>,
    pub identity: Option<String>,
}

impl Source {
    fn fetch(
        &mut self,
        first: u64,
        last: u64,
        total: Option<u64>,
        etag: Option<&str>,
    ) -> Result<(Vec<u8>, u64, String)> {
        self.fetch_inner(first, last, total, etag)
            .with_context(|| format!("{}: bytes {first}-{last}", self.label))
    }

    fn fetch_inner(
        &mut self,
        first: u64,
        last: u64,
        total: Option<u64>,
        etag: Option<&str>,
    ) -> Result<(Vec<u8>, u64, String)> {
        let range = format!("bytes={first}-{last}");
        let mut refreshed = false;
        let mut candidate = self.url.clone();
        let mut transitions = 0;
        let mut retries = 0;
        let mut partial_pin: Option<(String, u64, String)> = None;
        while transitions <= 6 {
            let (url, from_dandi_api) = match candidate.take() {
                Some(candidate) => candidate,
                None => (
                    self.refresh.as_ref().context("missing download URL")?()?,
                    true,
                ),
            };
            let mut headers = vec![("Range", range.as_str())];
            if let Some(tag) = etag.or_else(|| partial_pin.as_ref().map(|p| p.2.as_str())) {
                headers.push(("If-Match", tag));
            }
            let mut res = match request_once(&self.agent, &url, &headers) {
                Ok(res) if retries == 2 || !transient_status(res.status()) => res,
                Ok(_) | Err(ureq::Error::Timeout(_) | ureq::Error::Io(_)) if retries < 2 => {
                    candidate = Some((url, from_dandi_api));
                    backoff(retries);
                    retries += 1;
                    continue;
                }
                Err(ureq::Error::Timeout(_)) => bail!("HTTP GET timed out"),
                Err(_) => bail!("HTTP GET failed"),
                Ok(res) => res,
            };
            if res.status().is_redirection() {
                candidate = Some((redirect(&url, &res)?, false));
                transitions += 1;
                continue;
            }
            if res.status() == StatusCode::FORBIDDEN && self.refresh.is_some() && !refreshed {
                refreshed = true;
                transitions += 1;
                continue;
            }
            ensure!(
                res.status() == StatusCode::PARTIAL_CONTENT,
                "expected HTTP 206, got HTTP {}",
                res.status().as_u16()
            );
            let identity = object_identity(&url, from_dandi_api);
            ensure!(
                self.identity
                    .as_ref()
                    .is_none_or(|pinned| *pinned == identity),
                "download object identity changed"
            );
            let h = res.headers();
            ensure!(
                h.get("content-encoding").is_none_or(|v| v == "identity"),
                "unexpected Content-Encoding"
            );
            let actual_tag = strong_etag(
                h.get("etag")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or_default(),
            )?;
            ensure!(
                etag.is_none_or(|expected| actual_tag == expected),
                "ETag changed during read"
            );
            let actual_tag = actual_tag.to_owned();
            let content_range = h
                .get("content-range")
                .and_then(|v| v.to_str().ok())
                .context("missing Content-Range")?;
            let actual_total: u64 = content_range
                .rsplit_once('/')
                .and_then(|(_, len)| len.parse().ok())
                .context("invalid Content-Range total")?;
            ensure!(
                total.is_none_or(|n| n == actual_total),
                "Content-Range total changed"
            );
            ensure!(
                partial_pin.as_ref().is_none_or(|(key, size, tag)| {
                    key == &identity && *size == actual_total && tag == &actual_tag
                }),
                "range identity changed after an incomplete response"
            );
            ensure!(
                last < actual_total
                    && content_range == format!("bytes {first}-{last}/{actual_total}"),
                "unexpected Content-Range"
            );
            let count = last - first + 1;
            ensure!(
                h.get("content-length").is_none_or(|v| v
                    .to_str()
                    .ok()
                    .and_then(|s| s.parse().ok())
                    == Some(count)),
                "wrong Content-Length"
            );
            let mut data = Vec::with_capacity(count as usize);
            let read = res
                .body_mut()
                .as_reader()
                .take(count + 1)
                .read_to_end(&mut data);
            // Never append partial bytes. Retry the same range against the
            // identity validated from this response, even on the initial probe.
            if retries < 2
                && (data.len() as u64) <= count
                && (read.is_err() || (data.len() as u64) < count)
            {
                partial_pin = Some((identity, actual_total, actual_tag));
                candidate = Some((url, from_dandi_api));
                backoff(retries);
                retries += 1;
                continue;
            }
            read.map_err(|_| anyhow!("could not read complete range body"))?;
            ensure!(data.len() as u64 == count, "wrong range body length");
            self.url = Some((url, from_dandi_api));
            self.identity = Some(identity);
            return Ok((data, actual_total, actual_tag));
        }
        bail!("too many download redirects")
    }
}

// The ID is unique to an Object within this cache. Its source identity can
// only be set once, and every fill must match that identity, ETag and size.
type Blocks = LruCache<(usize, u64), Vec<u8>>;

#[derive(Clone)]
pub(crate) struct Cache(Arc<Mutex<Blocks>>);

impl Cache {
    pub fn new() -> Self {
        // Every entry is at most BLOCK bytes, including short tail blocks.
        Self(Arc::new(Mutex::new(LruCache::new(
            NonZeroUsize::new(256).unwrap(),
        ))))
    }
}

#[derive(Default)]
struct Sequential {
    last: Option<(u64, u64)>,
    generation: u64,
    runs: u8,
    working: bool,
}

pub(crate) struct Object {
    pub size: u64,
    pub etag: String,
    pub label: String,
    id: usize,
    cache: Cache,
    // ponytail: serialize misses for one object to dedupe overlap; independent
    // objects run in parallel. Use per-run locks if concurrent HDF5 I/O matters.
    source: Mutex<Source>,
    sequential: Mutex<Sequential>,
}

impl Object {
    #[cfg(feature = "hdf5")]
    pub fn probe(mut source: Source, expected_size: Option<u64>) -> Result<(Arc<Self>, String)> {
        let (_, size, etag) = source.fetch(0, 7, expected_size, None)?;
        let identity = source.identity.as_ref().unwrap().clone();
        Ok((
            Arc::new(Self::new(0, Cache::new(), source, size, etag)),
            identity,
        ))
    }

    pub fn new(id: usize, cache: Cache, source: Source, size: u64, etag: String) -> Self {
        Self {
            size,
            etag,
            label: source.label.clone(),
            id,
            cache,
            source: Mutex::new(source),
            sequential: Mutex::default(),
        }
    }

    fn fill(&self, source: &mut Source, start: u64, end: u64) -> Result<()> {
        let (data, _, _) = source.fetch(start, end - 1, Some(self.size), Some(&self.etag))?;
        let mut cache = self.cache.0.lock().unwrap();
        for (i, block) in data.chunks(BLOCK as usize).enumerate() {
            cache.put((self.id, start + i as u64 * BLOCK), block.to_vec());
        }
        Ok(())
    }

    pub fn read_uninit(
        self: &Arc<Self>,
        offset: u64,
        buf: &mut [MaybeUninit<u8>],
        readahead: bool,
    ) -> Result<()> {
        let end = offset
            .checked_add(buf.len() as u64)
            .context("range offset overflow")?;
        ensure!(
            end <= self.size,
            "{}: bytes {offset}..{end} exceed EOF",
            self.label
        );
        if buf.is_empty() {
            return Ok(());
        }
        let mut sequential = false;
        if readahead {
            let mut seq = self.sequential.lock().unwrap();
            sequential = seq
                .last
                .is_some_and(|(a, b)| offset >= a && offset <= b && end > b);
            if !sequential && seq.last != Some((offset, end)) {
                seq.generation += 1;
                seq.runs = 0;
            }
            seq.last = Some((offset, end));
        }
        let mut pos = offset;
        let mut hit = true;
        while pos < end {
            let start = pos / BLOCK * BLOCK;
            let mut cache = self.cache.0.lock().unwrap();
            if let Some(block) = cache.get(&(self.id, start)) {
                let stop = end.min(start + block.len() as u64);
                let dest = &mut buf[(pos - offset) as usize..(stop - offset) as usize];
                let src = &block[(pos - start) as usize..(stop - start) as usize];
                // MaybeUninit<u8> has the layout of u8; the slices have the
                // same length and the HDF5/output buffer cannot alias the cache.
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dest.as_mut_ptr().cast(), src.len())
                };
                pos = stop;
            } else {
                drop(cache);
                let mut source = self.source.lock().unwrap();
                let cache = self.cache.0.lock().unwrap();
                if cache.contains(&(self.id, start)) {
                    continue;
                }
                let limit = end
                    .div_ceil(BLOCK)
                    .saturating_mul(BLOCK)
                    .min(start.saturating_add(RUN))
                    .min(self.size);
                let stop = (start.saturating_add(BLOCK)..limit)
                    .step_by(BLOCK as usize)
                    .find(|&p| cache.contains(&(self.id, p)))
                    .unwrap_or(limit);
                drop(cache);
                self.fill(&mut source, start, stop)?;
                hit = false;
            }
        }
        if sequential && hit {
            self.readahead(end);
        }
        Ok(())
    }

    fn readahead(self: &Arc<Self>, end: u64) {
        let start = end.div_ceil(BLOCK).saturating_mul(BLOCK);
        if start >= self.size {
            return;
        }
        let mut seq = self.sequential.lock().unwrap();
        if seq.working || seq.runs >= 2 || self.cache.0.lock().unwrap().contains(&(self.id, start))
        {
            return;
        }
        seq.working = true;
        seq.runs += 1;
        let generation = seq.generation;
        let weak = Arc::downgrade(self);
        if std::thread::Builder::new()
            .name("sorrel-range".into())
            .spawn(move || {
                if let Some(this) = weak.upgrade() {
                    // Do not queue behind demand I/O. A seek invalidates queued
                    // work; an HTTP request already sent may finish within timeout.
                    if let Ok(mut source) = this.source.try_lock() {
                        let current = this.sequential.lock().unwrap().generation == generation;
                        if current && !this.cache.0.lock().unwrap().contains(&(this.id, start)) {
                            let _ = this.fill(
                                &mut source,
                                start,
                                start.saturating_add(BLOCK).min(this.size),
                            );
                        }
                    }
                    this.sequential.lock().unwrap().working = false;
                }
            })
            .is_err()
        {
            seq.working = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    fn serve(replies: &[&'static str]) -> (Source, std::thread::JoinHandle<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = Url::parse(&format!("http://{}", listener.local_addr().unwrap())).unwrap();
        let replies = replies.to_vec();
        let server = std::thread::spawn(move || {
            replies
                .into_iter()
                .map(|reply| {
                    let (mut stream, _) = listener.accept().unwrap();
                    stream
                        .set_read_timeout(Some(Duration::from_secs(2)))
                        .unwrap();
                    let mut reader = BufReader::new(&stream);
                    let mut request = String::new();
                    loop {
                        let offset = request.len();
                        if reader.read_line(&mut request).unwrap() == 0
                            || &request[offset..] == "\r\n"
                        {
                            break;
                        }
                    }
                    stream.write_all(reply.as_bytes()).unwrap();
                    request.to_ascii_lowercase()
                })
                .collect()
        });
        let source = Source {
            label: "fixture".into(),
            agent: agent(Duration::from_secs(2)),
            url: Some((url, false)),
            refresh: None,
            identity: None,
        };
        (source, server)
    }

    const TRUNCATED: &str = "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-3/8\r\nContent-Length: 4\r\nETag: \"v1\"\r\nConnection: close\r\n\r\nab";
    const COMPLETE: &str = "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-3/8\r\nContent-Length: 4\r\nETag: \"v1\"\r\nConnection: close\r\n\r\nabcd";

    #[test]
    fn range_retries_status_and_truncated_body_with_one_budget() {
        let (mut source, server) = serve(&[
            "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            TRUNCATED,
            COMPLETE,
        ]);
        assert_eq!(source.fetch(0, 3, None, None).unwrap().0, b"abcd");
        let requests = server.join().unwrap();
        assert!(requests.iter().all(|req| req.contains("range: bytes=0-3")));
        assert!(requests[2].contains("if-match: \"v1\""));
    }

    #[test]
    fn range_stops_after_three_truncations_without_committing_identity() {
        let (mut source, server) = serve(&[TRUNCATED; 3]);
        assert!(source.fetch(0, 3, None, None).is_err());
        assert!(source.identity.is_none());
        assert_eq!(server.join().unwrap().len(), 3);
    }

    #[test]
    fn range_does_not_retry_pin_mismatch() {
        let (mut source, server) = serve(&[COMPLETE]);
        let error = source.fetch(0, 3, Some(8), Some("\"v2\"")).unwrap_err();
        assert!(format!("{error:#}").contains("ETag changed"));
        assert_eq!(server.join().unwrap().len(), 1);
    }
}

//! Resolve DANDI asset locators without persisting signed storage URLs.
#[cfg(feature = "hdf5")]
use crate::http_range::Object;
use crate::http_range::{self, Source};
use anyhow::{anyhow, bail, ensure, Context, Result};
use percent_encoding::percent_decode_str;
use serde::{de::DeserializeOwned, Deserialize};
use std::{
    collections::HashSet,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use ureq::http::Response;
use url::{Position, Url};

#[path = "dandi/zarr.rs"]
mod zarr;
pub(crate) use zarr::ZarrStore;

const BASE: &str = "https://api.dandiarchive.org/api/";

/// Whether this dataset argument names a remote NWB source, not a local path.
pub fn is_remote(path: &Path) -> bool {
    path.to_str()
        .is_some_and(|s| s.starts_with("dandi://") || s.starts_with("https://"))
}

pub(crate) enum Store {
    #[cfg(feature = "hdf5")]
    Blob(Arc<Object>),
    Zarr(Arc<ZarrStore>),
}

pub(crate) struct Resolved {
    pub store: Store,
    pub identity: Vec<u8>,
    pub state: PathBuf,
}

pub(crate) fn resolve(input: &str) -> Result<Resolved> {
    Api::production().resolve(input)
}

/// Find the local state subdirectory without opening or listing the asset.
pub fn state_path(input: &str) -> Result<PathBuf> {
    if let Some(locator) = Api::production().dandiset_locator(input)? {
        return Ok(locator.state_path());
    }
    Ok(https_state_path(&https_url(input)?))
}

#[derive(Clone)]
pub(super) struct Api {
    base: Url,
    key: Option<String>,
    agent: ureq::Agent,
}

#[derive(Deserialize)]
struct Page<T> {
    next: Option<String>,
    results: Vec<T>,
}

#[derive(Deserialize)]
struct Dandiset {
    most_recent_published_version: Option<Version>,
    draft_version: Version,
}

#[derive(Deserialize)]
struct Version {
    version: String,
}

#[derive(Deserialize)]
struct Asset {
    asset_id: String,
    path: String,
    size: u64,
    blob: Option<String>,
    zarr: Option<String>,
}

fn component(part: &str) -> Result<&str> {
    ensure!(
        !part.is_empty()
            && part
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.')
            && part != "."
            && part != ".."
            && !part.ends_with('.'),
        "invalid DANDI identifier"
    );
    Ok(part)
}

pub(super) fn hash_field(hash: &mut blake3::Hasher, field: &str) {
    hash.update(&(field.len() as u64).to_le_bytes());
    hash.update(field.as_bytes());
}

struct Locator<'a> {
    id: &'a str,
    version: String,
    path: std::borrow::Cow<'a, str>,
}

impl Locator<'_> {
    fn state_path(&self) -> PathBuf {
        PathBuf::from(self.id)
            .join(&self.version)
            .join(blake3::hash(self.path.as_bytes()).to_hex().as_str())
    }
}

fn https_url(input: &str) -> Result<Url> {
    let url = Url::parse(input).map_err(|_| anyhow!("invalid remote NWB URL"))?;
    http_range::validate_url(&url)?;
    Ok(url)
}

fn https_state_path(url: &Url) -> PathBuf {
    PathBuf::from("https").join(
        blake3::hash(url[..Position::AfterPath].as_bytes())
            .to_hex()
            .as_str(),
    )
}

impl Api {
    fn production() -> Self {
        Self {
            base: Url::parse(BASE).expect("valid DANDI API base"),
            key: std::env::var("DANDI_API_KEY")
                .ok()
                .filter(|s| !s.is_empty()),
            agent: http_range::agent(Duration::from_secs(15)),
        }
    }

    fn dandiset_locator<'a>(&self, input: &'a str) -> Result<Option<Locator<'a>>> {
        let Some(rest) = input.strip_prefix("dandi://dandi/") else {
            return Ok(None);
        };
        let (dataset, path) = rest.split_once('/').context("missing DANDI asset path")?;
        let (id, version) = dataset
            .split_once('@')
            .map_or((dataset, None), |(id, v)| (id, Some(v)));
        ensure!(
            id.len() == 6 && id.bytes().all(|b| b.is_ascii_digit()),
            "expected a six-digit Dandiset ID"
        );
        let path = percent_decode_str(path)
            .decode_utf8()
            .context("asset path is not UTF-8")?;
        ensure!(
            path.split('/').all(|p| !matches!(p, "" | "." | ".."))
                && !path.chars().any(char::is_control),
            "invalid DANDI asset path"
        );
        let version = match version {
            Some(version) => component(version)?.to_owned(),
            None => {
                let dataset: Dandiset = self.json(&self.endpoint(&format!("dandisets/{id}/")))?;
                dataset
                    .most_recent_published_version
                    .unwrap_or(dataset.draft_version)
                    .version
            }
        };
        component(&version)?;
        Ok(Some(Locator { id, version, path }))
    }

    fn resolve(&self, input: &str) -> Result<Resolved> {
        if let Some(locator) = self.dandiset_locator(input)? {
            let Locator { id, version, path } = &locator;
            let mut list = self.endpoint(&format!("dandisets/{id}/versions/{version}/assets/"));
            list.query_pairs_mut().append_pair("path", path);
            let mut exact = None;
            self.pages(list, |asset: Asset| {
                if asset.path == path.as_ref() {
                    ensure!(
                        exact.replace(asset).is_none(),
                        "multiple DANDI assets have the exact path"
                    );
                }
                Ok(())
            })
            .with_context(|| format!("list Dandiset {id}@{version}"))?;
            let asset =
                exact.with_context(|| format!("Dandiset {id}@{version} has no asset at {path}"))?;
            return self.asset(asset, locator.state_path(), None);
        }

        let url = https_url(input)?;
        let canonical = url[..Position::AfterPath].to_owned();
        let state = https_state_path(&url);
        // DANDI API download URLs can describe either kind of asset, including
        // Zarr assets whose download endpoint cannot return a single object.
        if url.origin() == self.base.origin() {
            let segments: Vec<_> = url.path().trim_end_matches('/').split('/').collect();
            let asset_id = match segments.as_slice() {
                ["", "api", "assets", id, "download"] => Some(*id),
                ["", "api", "dandisets", _, "versions", _, "assets", id, "download"] => Some(*id),
                _ => None,
            };
            if let Some(id) = asset_id {
                component(id)?;
                let prefix = url
                    .path()
                    .trim_end_matches('/')
                    .strip_suffix("download")
                    .unwrap();
                let info = url
                    .join(&format!("{prefix}info/"))
                    .map_err(|_| anyhow!("invalid DANDI asset endpoint"))?;
                let asset: Asset = self.json(&info)?;
                ensure!(asset.asset_id == id, "DANDI returned a different asset ID");
                return self.asset(asset, state, Some(url));
            }
        }
        ensure!(!url.path().trim_end_matches('/').ends_with(".zarr"),
            "use a dandi:// address or a DANDI API asset download URL for NWB-Zarr (requires an object list)");
        Self::blob(
            Source {
                label: canonical.clone(),
                agent: self.agent.clone(),
                url: Some(url),
                refresh: None,
            },
            None,
            canonical,
            state,
        )
    }

    fn asset(&self, asset: Asset, state: PathBuf, download: Option<Url>) -> Result<Resolved> {
        let id = component(&asset.asset_id)?;
        let label = format!("DANDI asset {id}");
        match (asset.blob, asset.zarr) {
            (Some(_), None) => Self::blob(
                self.source(
                    label,
                    download.unwrap_or_else(|| self.endpoint(&format!("assets/{id}/download/"))),
                ),
                Some(asset.size),
                asset.asset_id,
                state,
            ),
            (None, Some(zarr)) => {
                let (store, identity) = ZarrStore::open(self, id, component(&zarr)?)
                    .with_context(|| format!("list objects for {label}"))?;
                Ok(Resolved {
                    store: Store::Zarr(Arc::new(store)),
                    identity,
                    state,
                })
            }
            _ => bail!("{label}: expected either a blob or a Zarr store"),
        }
    }

    fn blob(source: Source, size: Option<u64>, id: String, state: PathBuf) -> Result<Resolved> {
        #[cfg(not(feature = "hdf5"))]
        {
            let _ = (source, size, id, state);
            bail!("streaming HDF5 NWB assets requires building with --features hdf5");
        }
        #[cfg(feature = "hdf5")]
        {
            let object = Object::probe(source, size)?;
            let mut hash =
                blake3::Hasher::new_derive_key("sorrel-provider-identity-v1/remote-nwb-blob");
            hash_field(&mut hash, &id);
            hash_field(&mut hash, &object.etag);
            hash.update(&object.size.to_le_bytes());
            Ok(Resolved {
                store: Store::Blob(object),
                identity: hash.finalize().as_bytes().to_vec(),
                state,
            })
        }
    }

    fn endpoint(&self, path: &str) -> Url {
        self.base.join(path).expect("validated DANDI endpoint path")
    }

    fn get(&self, url: &Url) -> Result<Response<ureq::Body>> {
        http_range::validate_url(url)?;
        ensure!(
            url.origin() == self.base.origin() && url.path().starts_with(self.base.path()),
            "DANDI API link left the API origin or prefix"
        );
        let token = self.key.as_ref().map(|key| format!("Token {key}"));
        let headers = token.as_deref().map(|token| [("Authorization", token)]);
        http_range::request(&self.agent, url, headers.as_ref().map_or(&[], |v| &v[..]))
            .with_context(|| format!("DANDI API GET {}", url.path()))
    }

    fn json<T: DeserializeOwned>(&self, url: &Url) -> Result<T> {
        let mut response = self.get(url)?;
        ensure!(
            response.status().is_success(),
            "DANDI API {}: HTTP {}",
            url.path(),
            response.status().as_u16()
        );
        const LIMIT: u64 = 16 * 1024 * 1024;
        let mut body = Vec::new();
        response
            .body_mut()
            .as_reader()
            .take(LIMIT + 1)
            .read_to_end(&mut body)
            .map_err(|_| anyhow!("failed to read DANDI API response"))?;
        ensure!(
            body.len() as u64 <= LIMIT,
            "DANDI API response exceeds size limit"
        );
        serde_json::from_slice(&body).map_err(|err| {
            anyhow!(
                "DANDI API {}: invalid JSON at line {} column {}",
                url.path(),
                err.line(),
                err.column()
            )
        })
    }

    fn pages<T: DeserializeOwned>(
        &self,
        mut url: Url,
        mut visit: impl FnMut(T) -> Result<()>,
    ) -> Result<()> {
        let mut seen = HashSet::new();
        let endpoint = url.path().to_owned();
        loop {
            ensure!(
                url.path() == endpoint && seen.insert(url.clone()),
                "invalid DANDI pagination link or cycle"
            );
            let page: Page<T> = self.json(&url)?;
            for item in page.results {
                visit(item)?;
            }
            let Some(next) = page.next else {
                return Ok(());
            };
            url = url
                .join(&next)
                .map_err(|_| anyhow!("invalid DANDI pagination URL"))?;
        }
    }

    fn source(&self, label: String, endpoint: Url) -> Source {
        let api = self.clone();
        Source {
            label,
            agent: self.agent.clone(),
            url: None,
            refresh: Some(Box::new(move || {
                let response = api.get(&endpoint)?;
                http_range::redirect(&endpoint, &response)
            })),
        }
    }
}

#[cfg(test)]
mod tests;

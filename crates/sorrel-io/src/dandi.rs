//! Resolve DANDI asset locators without persisting signed storage URLs.
#[cfg(feature = "hdf5")]
use crate::http_range::Object;
use crate::http_range::{self, Source};
use anyhow::{anyhow, bail, ensure, Context, Result};
use percent_encoding::percent_decode_str;
use serde::{de::DeserializeOwned, Deserialize};
use std::{
    collections::{HashMap, HashSet},
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
const LOCATOR_PREFIX: &str = "dandi://dandi/";

/// Whether this dataset argument names a remote NWB source, not a local path.
pub fn is_remote(path: &Path) -> bool {
    path.to_str()
        .is_some_and(|s| s.starts_with("dandi://") || s.starts_with("https://"))
}

#[derive(Clone)]
pub(crate) enum Store {
    #[cfg(feature = "hdf5")]
    Blob(Arc<Object>),
    Zarr(Arc<ZarrStore>),
}

/// A verified NWB asset revision and the store pinned to it.
#[derive(Clone)]
pub struct Resolved {
    pub(crate) store: Store,
    pub(crate) identity: Vec<u8>,
    pub(crate) state: PathBuf,
}

impl Resolved {
    /// Relative local state directory for the resolved locator.
    pub fn state_path(&self) -> &Path {
        &self.state
    }
}

/// Resolve and verify an asset, without opening its NWB contents.
pub fn resolve(input: &str) -> Result<Resolved> {
    Api::production().resolve(input)
}

/// Resolve remote sorting and recording locators in one request. Local inputs are `None`.
/// Versionless locators share one version lookup per Dandiset; equivalent locators share a store.
/// Two distinct draft assets cannot be pinned to a common snapshot.
pub fn resolve_pair(root: Option<&str>, dat: Option<&str>) -> Result<[Option<Resolved>; 2]> {
    if root.is_none() && dat.is_none() {
        return Ok([None, None]);
    }
    Api::production().resolve_pair(root, dat)
}

/// Find the local state subdirectory without opening or listing the asset.
pub fn state_path(input: &str) -> Result<PathBuf> {
    if let Some(locator) = Api::production().dandiset_locator(input, &mut HashMap::new())? {
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
    metadata: Option<AssetMetadata>,
}

#[derive(Deserialize)]
struct AssetMetadata {
    digest: Option<AssetDigest>,
}

#[derive(Deserialize)]
struct AssetDigest {
    #[serde(rename = "dandi:dandi-zarr-checksum")]
    zarr_checksum: Option<String>,
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

#[derive(PartialEq, Eq)]
struct Locator<'a> {
    id: &'a str,
    version: String,
    path: std::borrow::Cow<'a, str>,
}

fn decode_asset_path(encoded: &str) -> Result<std::borrow::Cow<'_, str>> {
    let bytes = encoded.as_bytes();
    ensure!(
        bytes.iter().enumerate().all(|(i, &b)| b != b'%'
            || bytes
                .get(i + 1..i + 3)
                .is_some_and(|digits| digits.iter().all(u8::is_ascii_hexdigit))),
        "invalid percent escape in DANDI asset path"
    );
    let path = percent_decode_str(encoded)
        .decode_utf8()
        .context("DANDI asset path is not UTF-8")?;
    ensure!(
        path.split('/').count() == encoded.split('/').count()
            && path.split('/').all(|p| !matches!(p, "" | "." | ".."))
            && !path.contains('\\')
            && !path.chars().any(char::is_control),
        "invalid or ambiguous DANDI asset path"
    );
    Ok(path)
}

struct DownloadAsset<'a> {
    location: Option<(&'a str, &'a str)>,
    asset_id: &'a str,
}

fn download_asset(path: &str) -> Option<DownloadAsset<'_>> {
    let segments: Vec<_> = path.split('/').collect();
    match segments.as_slice() {
        ["api", "assets", asset_id, "download"] => Some(DownloadAsset {
            location: None,
            asset_id,
        }),
        ["api", "dandisets", id, "versions", version, "assets", asset_id, "download"] => {
            Some(DownloadAsset {
                location: Some((id, version)),
                asset_id,
            })
        }
        _ => None,
    }
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

    fn dandiset_locator<'a>(
        &self,
        input: &'a str,
        versions: &mut HashMap<&'a str, String>,
    ) -> Result<Option<Locator<'a>>> {
        let Some(rest) = input.strip_prefix(LOCATOR_PREFIX) else {
            return Ok(None);
        };
        let (dataset, encoded_path) = rest.split_once('/').context("missing DANDI asset path")?;
        let (id, version) = dataset
            .split_once('@')
            .map_or((dataset, None), |(id, v)| (id, Some(v)));
        ensure!(
            id.len() == 6 && id.bytes().all(|b| b.is_ascii_digit()),
            "expected a six-digit Dandiset ID"
        );
        let path = decode_asset_path(encoded_path)?;
        let version = if let Some(version) = version {
            component(version)?.to_owned()
        } else if let Some(version) = versions.get(id) {
            version.clone()
        } else {
            let dataset: Dandiset = self.json(&self.endpoint(&format!("dandisets/{id}/")))?;
            let version = dataset
                .most_recent_published_version
                .unwrap_or(dataset.draft_version)
                .version;
            component(&version)?;
            versions.insert(id, version.clone());
            version
        };
        Ok(Some(Locator { id, version, path }))
    }

    fn resolve(&self, input: &str) -> Result<Resolved> {
        self.resolve_input(input, self.dandiset_locator(input, &mut HashMap::new())?)
    }

    fn resolve_pair<'a>(
        &self,
        root: Option<&'a str>,
        dat: Option<&'a str>,
    ) -> Result<[Option<Resolved>; 2]> {
        let mut versions = HashMap::new();
        let mut locator = |input: Option<&'a str>| {
            input
                .map(|input| self.dandiset_locator(input, &mut versions))
                .transpose()
                .map(Option::flatten)
        };
        let sorting_locator = locator(root).context("resolve remote dataset")?;
        let recording_locator = locator(dat).context("resolve remote --dat")?;
        let same =
            root == dat || (sorting_locator.is_some() && sorting_locator == recording_locator);
        if !same {
            let [sorting_draft, recording_draft] = [
                (root, sorting_locator.as_ref()),
                (dat, recording_locator.as_ref()),
            ]
            .map(|(input, locator)| self.draft_input(input, locator));
            if let (Some(a), Some(b)) = (sorting_draft?, recording_draft?) {
                const DISTINCT_DRAFTS: &str = "cannot combine two different DANDI draft assets in one session: drafts have no immutable snapshot; use a published version or a local file";
                let resolved = match (a, b) {
                    (DraftInput::Path(_), DraftInput::Path(_)) => bail!(DISTINCT_DRAFTS),
                    (DraftInput::Download(a, ad, ai), DraftInput::Download(_, bd, bi)) => {
                        ensure!(ad == bd && ai == bi, DISTINCT_DRAFTS);
                        self.resolve_input(a.as_str(), None)?
                    }
                    (DraftInput::Path(locator), DraftInput::Download(url, id, asset_id))
                    | (DraftInput::Download(url, id, asset_id), DraftInput::Path(locator)) => {
                        ensure!(locator.id == id, DISTINCT_DRAFTS);
                        let asset = self.find_asset(locator)?;
                        ensure!(asset.asset_id == asset_id, DISTINCT_DRAFTS);
                        let state = if sorting_locator.is_some() {
                            locator.state_path()
                        } else {
                            https_state_path(&url)
                        };
                        self.asset(asset, state, None)?
                    }
                };
                return Ok([Some(resolved.clone()), Some(resolved)]);
            }
        }
        let sorting = root
            .map(|input| self.resolve_input(input, sorting_locator))
            .transpose()
            .context("resolve remote dataset")?;
        let recording = if same {
            sorting.clone()
        } else {
            dat.map(|input| self.resolve_input(input, recording_locator))
                .transpose()
                .context("resolve remote --dat")?
        };
        Ok([sorting, recording])
    }

    fn draft_input<'a>(
        &self,
        input: Option<&str>,
        locator: Option<&'a Locator<'a>>,
    ) -> Result<Option<DraftInput<'a>>> {
        if let Some(locator) = locator {
            return Ok((locator.version == "draft").then_some(DraftInput::Path(locator)));
        }
        let Some(input) = input else { return Ok(None) };
        let url = https_url(input)?;
        if url.origin() != self.base.origin() {
            return Ok(None);
        }
        let path = decode_asset_path(url.path().trim_matches('/'))?;
        Ok(match download_asset(&path) {
            Some(DownloadAsset {
                location: Some((id, "draft")),
                asset_id,
            }) => Some(DraftInput::Download(
                url,
                component(id)?.to_owned(),
                component(asset_id)?.to_owned(),
            )),
            _ => None,
        })
    }

    fn find_asset(&self, locator: &Locator<'_>) -> Result<Asset> {
        let Locator { id, version, path } = locator;
        let mut list = self.endpoint(&format!("dandisets/{id}/versions/{version}/assets/"));
        list.query_pairs_mut()
            .append_pair("path", path)
            .append_pair("metadata", "true");
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
        exact.with_context(|| format!("Dandiset {id}@{version} has no asset at {path}"))
    }

    fn resolve_input(&self, input: &str, locator: Option<Locator<'_>>) -> Result<Resolved> {
        if let Some(locator) = locator {
            return self.asset(self.find_asset(&locator)?, locator.state_path(), None);
        }

        let url = https_url(input)?;
        let canonical = url[..Position::AfterPath].to_owned();
        let state = https_state_path(&url);
        // DANDI API download URLs can describe either kind of asset, including
        // Zarr assets whose download endpoint cannot return a single object.
        if url.origin() == self.base.origin() {
            let path = decode_asset_path(url.path().trim_matches('/'))?;
            if let Some(DownloadAsset {
                location,
                asset_id: id,
            }) = download_asset(&path)
            {
                component(id)?;
                if let Some((dandiset, version)) = location {
                    component(dandiset)?;
                    component(version)?;
                }
                let prefix = path.strip_suffix("download").unwrap();
                let info = url
                    .join(&format!("/{prefix}info/"))
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
                url: Some((url, false)),
                refresh: None,
                identity: None,
            },
            None,
            canonical,
            None,
            state,
        )
    }

    fn asset(&self, asset: Asset, state: PathBuf, download: Option<Url>) -> Result<Resolved> {
        let id = component(&asset.asset_id)?;
        let label = format!("DANDI asset {id}");
        match (asset.blob, asset.zarr) {
            (Some(blob), None) => Self::blob(
                self.source(
                    label,
                    download.unwrap_or_else(|| self.endpoint(&format!("assets/{id}/download/"))),
                ),
                Some(asset.size),
                asset.asset_id,
                Some(blob),
                state,
            ),
            (None, Some(zarr)) => {
                let zarr = component(&zarr)?;
                let digest = asset
                    .metadata
                    .and_then(|m| m.digest)
                    .and_then(|d| d.zarr_checksum);
                let (store, identity) = ZarrStore::open(self, id, zarr, digest.as_deref())
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

    fn blob(
        source: Source,
        size: Option<u64>,
        id: String,
        blob: Option<String>,
        state: PathBuf,
    ) -> Result<Resolved> {
        #[cfg(not(feature = "hdf5"))]
        {
            let _ = (source, size, id, blob, state);
            bail!("streaming HDF5 NWB assets requires building with --features hdf5");
        }
        #[cfg(feature = "hdf5")]
        {
            let (object, effective_identity) = Object::probe(source, size)?;
            let mut hash =
                blake3::Hasher::new_derive_key("sorrel-provider-identity-v2/remote-nwb-blob");
            hash_field(&mut hash, &id);
            hash_field(&mut hash, blob.as_deref().unwrap_or_default());
            hash_field(&mut hash, &effective_identity);
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
        Self::decode_json(url, self.get(url)?)
    }

    fn decode_json<T: DeserializeOwned>(
        url: &Url,
        mut response: Response<ureq::Body>,
    ) -> Result<T> {
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
            identity: None,
            refresh: Some(Box::new(move || {
                let response = api.get(&endpoint)?;
                http_range::redirect(&endpoint, &response)
            })),
        }
    }
}

enum DraftInput<'a> {
    Path(&'a Locator<'a>),
    Download(Url, String, String),
}

#[cfg(test)]
mod tests;

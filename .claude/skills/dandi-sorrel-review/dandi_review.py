#!/usr/bin/env python3
"""Collect local, content-addressed Sorrel reviews of public DANDI assets."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import csv
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
from html import unescape
from html.parser import HTMLParser
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import statistics
import sys
import tempfile
from typing import Any, Callable, Iterable
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


HERE = Path(__file__).resolve().parent
DEFAULT_THRESHOLDS = {
    "rp_contamination_lo95_max": 0.10,
    "srp_contamination_upper_max": 0.10,
    "isi_violation_rate_max_hz": 0.50,
    "presence_ratio_min": 0.70,
    "amplitude_cutoff_max": 0.10,
    "epoch_rate_ratio_min": 0.25,
    "epoch_amp_ratio_min": 0.50,
    "drift_correlation_max": 0.50,
}
API_BASE = "https://api.dandiarchive.org/api"
REGISTRY_FIELDS = ("dandiset_id", "name", "doi", "license")
DATASET_JSON_FIELDS = ("contributors", "measurement_techniques", "anatomy", "keywords", "unique_metadata")
LOCAL_DATASET_FIELDS = ("probe", "sorter", "anatomy")
ASSET_FIELDS = ("asset_id", "path", "size_bytes", "encoding", "is_nwb", "sha256", "zarr_checksum", "locator",
                "media_type", "download_url", "blob_id", "zarr_id", "pin_verification")
ARTIFACT_KINDS = {
    "report.md": "report_md", "cluster_qc.before.json": "cluster_qc_before",
    "cluster_qc.after.json": "cluster_qc_after", "clusters.before.tsv": "clusters_before",
    "clusters.after.tsv": "clusters_after", "describe.json": "describe_json",
    "curation.json": "curation", "sorrel_report.html": "report_html",
    "nwb_inspector.json": "nwb_inspector", "dandiset.json": "dandiset_metadata",
}
LABELS = ("good", "mua", "noise")
RESPONSE = {"acknowledged", "accepted"}
PUBLISHED = {"republished", "no_response_published"}
TERMINAL = PUBLISHED | {"declined"}
OUTREACH_ORDER = ("drafted", "sent", "reminded", "acknowledged", "accepted")
SUBJECT_FIELDS = ("review_ids", "quoted_finding_ids")
DESTINATION_FIELDS = ("recipient", "channel")
BUCKETS = ("publish_eligible_ghosted", "reminder_due", "in_grace_period", "extended", "resolved", "published")
STAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt](?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]"
    r"(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
)
DISCLOSURE_CLOCK = "SELECT CAST(strftime('%s', 'now') AS INTEGER)"
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
SECOND = timedelta(seconds=1)
Doc = dict[str, Any]
UnitId = int | str
ClusterRow = tuple[str, int, int, str | None]  # label, TSV line, spike count, parents


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def timestamp(value: str) -> datetime:
    require(isinstance(value, str) and STAMP.fullmatch(value), "expected a whole-second date-time with UTC offset")
    return datetime.fromisoformat(value.upper().replace("Z", "+00:00")).astimezone(timezone.utc)


def utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def epoch(value: datetime) -> int:
    return (value - EPOCH) // SECOND


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def fields(doc: Doc, names: Iterable[str]) -> Doc:
    return {key: doc[key] for key in names if key in doc}


def pairs(items: list[tuple[str, Any]]) -> Doc:
    result: Doc = {}
    for key, val in items:
        require(key not in result, f"duplicate JSON property: {key}")
        result[key] = val
    return result


def loads(raw: str | bytes, *, qc: bool = False) -> Any:
    def real(token: str) -> float:
        value = float(token)
        require(qc or math.isfinite(value), f"non-finite JSON number: {token}")
        return value

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=real, parse_float=real)


def read_json(path: Path, *, qc: bool = False) -> Any:
    try:
        return loads(path.read_text(encoding="utf-8"), qc=qc)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def json_object(value: str, option: str) -> Doc:
    doc = read_json(Path(value[1:])) if value.startswith("@") else loads(value)
    require(isinstance(doc, dict), f"{option}: expected object")
    return doc


def canonical_json_bytes(doc: Any) -> bytes:
    return (json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def emit(doc: Any, out: Path | None) -> None:
    content = canonical_json_bytes(doc)
    if out is None:
        print(content.decode("utf-8"), end="")
    else:
        atomic_write(out, content)


def atomic_write(path: Path, data: bytes) -> None:
    with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as scratch:
        temp = Path(scratch) / path.name
        temp.write_bytes(data)
        temp.replace(path)


def digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


@lru_cache(maxsize=3)
def schema(name: str) -> Doc:
    return read_json(HERE / f"{name}.schema.json")


class SchemaMismatch(ValueError):
    pass


def check_schema(value: Any, rule: Doc, root: Doc, at: str = "$") -> None:
    """Validate the keyword subset used in the bundled schemas, not arbitrary schemas."""
    def check(condition: Any, message: str) -> None:
        if not condition:
            raise SchemaMismatch(f"{at}: {message}")

    if "$ref" in rule:
        check_schema(value, root["$defs"][rule["$ref"].removeprefix("#/$defs/")], root, at)
    if "type" in rule:
        kinds = rule["type"] if isinstance(rule["type"], list) else [rule["type"]]
        number = finite(value) is not None
        tests = {"object": isinstance(value, dict), "array": isinstance(value, list),
                 "string": isinstance(value, str), "null": value is None, "boolean": type(value) is bool,
                 "number": number, "integer": type(value) is int or (number and value.is_integer())}
        check(any(tests[kind] for kind in kinds), f"expected {' or '.join(kinds)}")
    if "const" in rule:
        const = rule["const"]
        check(value == const and (type(value) is bool) == (type(const) is bool), f"expected {const!r}")
    if "enum" in rule:
        check(value in rule["enum"], f"expected one of {', '.join(map(str, rule['enum']))}")
    if isinstance(value, dict):
        for name in rule.get("required", []):
            check(name in value, f"missing required property {name!r}")
        props = rule.get("properties", {})
        if rule.get("additionalProperties") is False:
            check(value.keys() <= props.keys(), f"unexpected properties: {', '.join(sorted(value.keys() - props.keys()))}")
        for name in (name for name in props if name in value):
            check_schema(value[name], props[name], root, f"{at}.{name}")
    if isinstance(value, str):
        if rule.get("format") == "date-time":
            try:
                timestamp(value)
            except (ValueError, OverflowError) as exc:
                raise SchemaMismatch(f"{at}: invalid date-time; expected a whole-second UTC timestamp or offset") from exc
        if "pattern" in rule:
            check(re.fullmatch(rule["pattern"], value), f"expected pattern {rule['pattern']}")
        if "minLength" in rule:
            check(len(value) >= rule["minLength"], f"must have at least {rule['minLength']} character(s)")
        if "maxLength" in rule:
            check(len(value) <= rule["maxLength"], f"must have at most {rule['maxLength']} character(s)")
    if type(value) in (int, float):
        if "minimum" in rule:
            check(value >= rule["minimum"], f"must be >= {rule['minimum']}")
        if "maximum" in rule:
            check(value <= rule["maximum"], f"must be <= {rule['maximum']}")
    if isinstance(value, list):
        if "minItems" in rule:
            check(len(value) >= rule["minItems"], f"must have at least {rule['minItems']} item(s)")
        if "items" in rule:
            for i, item in enumerate(value):
                check_schema(item, rule["items"], root, f"{at}[{i}]")
    if "if" in rule:
        try:
            check_schema(value, rule["if"], root, at)
        except SchemaMismatch:
            pass
        else:
            check_schema(value, rule["then"], root, at)
    for subrule in rule.get("allOf", []):
        check_schema(value, subrule, root, at)
    if "anyOf" in rule:
        errors = []
        for subrule in rule["anyOf"]:
            try:
                check_schema(value, subrule, root, at)
                break
            except SchemaMismatch as exc:
                errors.append(str(exc))
        else:
            check(False, "need one of: " + "; ".join(errors))


def check(value: Any, name: str, definition: str | None = None) -> None:
    root = schema(name)
    check_schema(value, root if definition is None else root["$defs"][definition], root,
                 "$" if definition is None else definition)


def unique(values: Iterable[Any], what: str) -> set[Any]:
    seen: set[Any] = set()
    for val in values:
        require(val not in seen, f"duplicate {what}: {val}")
        seen.add(val)
    return seen


def relative_path(value: str) -> None:
    require(isinstance(value, str) and not re.search(r"[\x00-\x1f\x7f\\]", value)
            and all(part not in {"", ".", ".."} for part in value.split("/")),
            f"not a safe relative path: {value!r}")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def api_json(url: str) -> Doc:
    try:
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "sorrel-dandi-review/1"})
        with build_opener(NoRedirect()).open(request, timeout=30) as response:
            result = loads(response.read())
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise ValueError(f"DANDI API HTTP {exc.code}: public access only; an embargo is not supported") from exc
        raise
    require(isinstance(result, dict), "expected a DANDI API JSON object")
    return result


def public_dataset(url: str) -> Doc:
    document = api_json(url)
    status = str(document.get("embargo_status", "unknown")).upper()
    require(status == "OPEN", f"Dandiset is {status}; this kit only reviews public data")
    return document


def dandi_endpoint(api_base: str, ident: str) -> str:
    require(re.fullmatch(r"[0-9]{6}", ident), "Dandiset id must have six digits")
    base = urlsplit(api_base)
    require((base.scheme == "https" or (base.scheme == "http" and base.hostname in {"127.0.0.1", "::1", "localhost"}))
            and base.hostname and not any((base.username, base.password, base.query, base.fragment)),
            "--api-base must be an HTTPS URL (loopback HTTP is allowed for tests)")
    return f"{api_base.rstrip('/')}/dandisets/{ident}/"


def asset_download_url(asset_id: str) -> str:
    require(isinstance(asset_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", asset_id), "invalid DANDI asset ID")
    return f"{API_BASE}/assets/{asset_id}/download/"


def is_nwb_asset(asset: Doc) -> bool:
    mime = asset.get("media_type", "").split(";", 1)[0].lower()
    return asset["path"].lower().endswith((".nwb", ".nwb.zarr")) or "nwb" in mime


def pinned_assets(dataset_url: str, ident: str, version: str, *, path: str | None = None) -> list[Doc]:
    assets = []
    seen = set()
    listing = f"{dataset_url}versions/{version}/assets/"
    endpoint = urlsplit(listing)
    url = listing + "?metadata=true"
    if path is not None:
        url += "&path=" + quote(path, safe="")
    total = None
    while url:
        current = urlsplit(url)
        require(current.scheme == endpoint.scheme and current.netloc == endpoint.netloc
                and current.path == endpoint.path and not current.fragment, "unsafe DANDI pagination URL")
        require(url not in seen, "cycle in DANDI pagination")
        seen.add(url)
        page = api_json(url)
        if total is None:
            total = page.get("count")
        require(isinstance(page.get("results"), list), "DANDI asset page needs a results array")
        for item in page["results"]:
            require(isinstance(item, dict), "DANDI asset must be an object")
            meta = item.get("metadata") or {}
            require(isinstance(meta, dict), "DANDI asset metadata must be an object")
            digests = meta.get("digest") or item.get("digest") or {}
            require(isinstance(digests, dict), "DANDI asset digest must be an object")
            name = item["path"]
            relative_path(name)
            mime = (meta.get("encodingFormat") or item.get("encodingFormat")
                    or meta.get("content_type") or item.get("content_type"))
            kind = (mime or "").split(";", 1)[0].strip().lower()
            encoding = ("zarr" if name.lower().endswith(".zarr") or kind == "application/x-zarr"
                        else "hdf5" if name.lower().endswith((".nwb", ".h5", ".hdf5"))
                        or kind in {"application/x-hdf", "application/x-hdf5", "application/x-nwb"}
                        else "other")
            asset = {"asset_id": item["asset_id"], "path": name, "size_bytes": item["size"],
                     "encoding": encoding, "locator": f"dandi://dandi/{ident}@{version}/{quote(name, safe='/')}",
                     "download_url": asset_download_url(item["asset_id"])}
            if mime:
                asset["media_type"] = mime
            asset["is_nwb"] = is_nwb_asset(asset)
            for key in ("blob", "zarr"):
                if item.get(key) is not None:
                    asset[f"{key}_id"] = item[key]
            for key, dandi_key in (("sha256", "dandi:sha2-256"), ("zarr_checksum", "dandi:dandi-zarr-checksum")):
                if digests.get(dandi_key) is not None:
                    asset[key] = digests[dandi_key]
            check(asset | {"open_status": "failed"}, "review", "asset")  # Inventory has not opened it.
            asset["pin_verification"] = "unverified" if version == "draft" else "published_version"
            assets.append(asset)
        next_page = page.get("next")
        url = urljoin(url, next_page) if next_page else ""
    if total is not None:
        require(len(assets) == total, f"DANDI pagination returned {len(assets)} of {total} assets")
    unique((a["asset_id"] for a in assets), "asset_id")
    unique((a["path"] for a in assets), "asset path")
    return sorted(assets, key=lambda a: (a["path"], a["asset_id"]))


def draft_modified(dataset: Doc) -> str:
    value = (dataset.get("draft_version") or {}).get("modified")
    require(isinstance(value, str) and value, "public DANDI draft has no modification timestamp")
    return value


def named_terms(values: Any) -> list[str]:
    return list(dict.fromkeys(
        name for value in (values if isinstance(values, list) else [])
        if isinstance(name := (value.get("name") or value.get("identifier") if isinstance(value, dict)
                               else value), str) and name))


def dandi_anatomy(raw: Doc) -> list[str]:
    return named_terms([item for item in (raw.get("about") or [])
                        if isinstance(item, dict) and item.get("schemaKey") == "Anatomy"])


def dataset_metadata(raw: Doc, dataset: Doc, ident: str, version: str, assets: list[Doc], n_nwb: int) -> Doc:
    doc = {"schema_version": "1", "dandiset_id": ident, "name": raw.get("name"),
           "updated_at": utc(now_utc()), "dandi_raw_metadata": raw, "unique_metadata": {},
           "pinned_version": version, "versions_seen": [version], "assets": assets,
           "assets_summary": {"n_assets_total": len(assets), "n_nwb_assets": n_nwb,
                              "total_size_bytes": sum(a["size_bytes"] for a in assets),
                              "encodings": dict(Counter(a["encoding"] for a in assets))}}
    contributors = []
    contact = None
    for entry in raw.get("contributor") or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"]:
            continue
        row = {"name": entry["name"]}
        if isinstance(entry.get("email"), str) and entry["email"]:
            row["email"] = entry["email"]
        orcid = entry.get("identifier")
        if isinstance(orcid, str) and re.fullmatch(r"(?:(?:https?://orcid\.org/)|(?:orcid:))?"
                                                   r"[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]", orcid):
            row["orcid"] = orcid
        for key, value in (("roles", entry.get("roleName")), ("affiliations", entry.get("affiliation"))):
            if terms := named_terms(value):
                row[key] = terms
        if not contact and "dcite:ContactPerson" in row.get("roles", []):
            contact = row.get("email") or row["name"]
        contributors.append(row)
    summary = raw.get("assetsSummary")
    summary = summary if isinstance(summary, dict) else {}
    resources = [item for item in (raw.get("relatedResource") or []) if isinstance(item, dict)]
    related_dois = []
    for resource in resources:
        for key in ("identifier", "url"):
            value = resource.get(key)
            if isinstance(value, str):
                match = re.fullmatch(r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.[0-9]{4,}/\S+)",
                                     value.strip(), re.IGNORECASE)
                if match:
                    related_dois.append(match[1])
    license_value = raw.get("license")
    for key, value in (
        *((key, raw.get(key)) for key in ("doi", "url", "description", "citation")),
        ("license", ", ".join(named_terms(license_value)) if isinstance(license_value, list) else license_value),
        ("contact", contact or dataset.get("contact_person")),
        ("embargo_status", dataset.get("embargo_status")),
        ("dandi_created", raw.get("dateCreated") or dataset.get("created")),
        ("dandi_modified", raw.get("dateModified") or dataset.get("modified")),
    ):
        if isinstance(value, str):
            doc[key] = value
    for key, values in (
        ("contributors", contributors), ("species", named_terms(summary.get("species"))),
        ("modality", named_terms(summary.get("approach"))),
        ("measurement_techniques", named_terms(summary.get("measurementTechnique"))),
        ("anatomy", dandi_anatomy(raw)),
        ("keywords", named_terms(raw.get("keywords"))),
        ("related_dois", list(dict.fromkeys(related_dois))), ("related_resources", resources),
    ):
        if values:
            doc[key] = values
    return doc


def pin(args: argparse.Namespace) -> None:
    ident = args.dandiset_id
    dataset_url = dandi_endpoint(args.api_base, ident)
    dataset = public_dataset(dataset_url)
    published = dataset.get("most_recent_published_version") or {}
    version = args.version or published.get("version") or ("draft" if args.allow_draft else None)
    require(version is not None, "no published version; use --allow-draft or --version draft to pin the draft")
    is_draft = version == "draft"
    source = (dataset.get("draft_version") or {}) if is_draft else published if version == published.get("version") else {}
    dandiset = {k: source[k] for k in ("name", "doi") if source.get(k) is not None}
    dandiset.update(id=ident, version=version)
    snapshot = {}
    if is_draft:
        modified = draft_modified(dataset)
        # Retain the exact API token for the race check, not for clock math.
        snapshot["draft_modified_source"] = modified
        dandiset["draft_modified"] = utc(timestamp(re.sub(r"\.[0-9]+(?=[Zz+-])", "", modified)))
    check(dandiset, "review", "dandiset")
    raw = api_json(f"{dataset_url}versions/{version}/") if args.store or args.dataset_out else None
    assets = pinned_assets(dataset_url, ident, version)
    nwb = [asset for asset in assets if asset["is_nwb"]]
    if is_draft:
        require(draft_modified(public_dataset(dataset_url)) == modified, "draft changed during pin; retry")
        print("warning: draft path locators are mutable; use asset-ID download_url and verify-pin before packing",
              file=sys.stderr)
    if raw is not None:
        doc = dataset_metadata(raw, dataset, ident, version, assets, len(nwb))
        store, snapshot_out = args.store, args.dataset_out
        path = store / "dandisets" / f"{ident}.json" if store is not None else None
        require(args.out is None or all(args.out.resolve() != p.resolve() for p in (path, snapshot_out) if p),
                "--out must not overwrite a dataset metadata output")
        if path is not None:
            registry, reviews, feedback = load_store(store)
            previous = read_dataset(path) if path.exists() else {}
            doc.update(fields(previous, ("unique_metadata", "probe", "sorter")))
            if "anatomy" in previous and previous["anatomy"] != dandi_anatomy(previous["dandi_raw_metadata"]):
                doc["anatomy"] = previous["anatomy"]
            seen = {version, *previous.get("versions_seen", [])}
            if "pinned_version" in previous:
                seen.add(previous["pinned_version"])
            doc["versions_seen"] = sorted(seen)
            if snapshot_out is not None:
                if snapshot_out.resolve() == path.resolve():
                    snapshot_out = None
                else:
                    require(not snapshot_out.resolve().is_relative_to(path.parent.resolve()),
                            "--dataset-out must not overwrite another store Dandiset")
            save_dataset(store, doc, registry, reviews, feedback)
        else:
            check(doc, "dandiset")
        if snapshot_out is not None:
            emit(doc, snapshot_out)
    emit({"dandiset": dandiset, "n_assets_total": len(assets), "n_nwb_assets": len(nwb),
          "assets": nwb if args.nwb_only else assets, **snapshot}, args.out)


def manifest_asset(manifest: Doc, path: str) -> Doc:
    matches = [item for item in manifest["assets"] if item["path"] == path]
    require(len(matches) == 1, f"expected one manifest asset at {path!r}")
    return matches[0]


def verify_pin(args: argparse.Namespace) -> None:
    manifest = read_json(args.manifest)
    dandiset = manifest["dandiset"]
    check(dandiset, "review", "dandiset")
    require(dandiset["version"] == "draft", "verify-pin is only needed for a draft")
    asset = manifest_asset(manifest, args.asset_path)
    metadata = dict(asset, open_status="failed")
    if metadata.get("pin_verification") == "unverified":
        del metadata["pin_verification"]
    check(metadata, "review", "asset")
    if args.local_file is not None:
        regular_file(args.local_file)
        require(asset["encoding"] != "zarr" and asset.get("sha256"), "local verification requires a pinned file SHA-256")
        require(digest_file(args.local_file) == (asset["sha256"], asset["size_bytes"]),
                "local file does not match pinned SHA-256 and size")
        asset["pin_verification"] = "local_checksum"
    else:
        ident = dandiset["id"]
        url = dandi_endpoint(args.api_base, ident)
        modified = manifest["draft_modified_source"]
        require(draft_modified(public_dataset(url)) == modified, "draft changed since pin; repin and repeat the run")
        current = manifest_asset({"assets": pinned_assets(url, ident, "draft", path=asset["path"])}, asset["path"])
        digest = "zarr_checksum" if asset["encoding"] == "zarr" else "sha256"
        require(asset.get(digest), f"draft postcheck needs a pinned {digest}")
        if asset["encoding"] == "zarr":
            require(asset.get("zarr_id"), "draft postcheck needs a pinned zarr_id")
        for key in ("asset_id", digest, "blob_id", "zarr_id", "encoding", "size_bytes"):
            require(asset.get(key) == current.get(key), f"draft asset {key} changed; repin and repeat the run")
        require(draft_modified(public_dataset(url)) == modified, "draft changed during postcheck; repin and repeat the run")
        asset["pin_verification"] = "api_postcheck"
    emit(manifest, args.out or args.manifest)


def asset_from_args(args: argparse.Namespace) -> Doc:
    explicit = {k: value for k in ASSET_FIELDS if (value := getattr(args, k, None)) is not None}
    if args.manifest is not None:
        require(args.asset_path and not explicit, "--manifest needs --asset-path and no explicit asset fields")
        manifest = read_json(args.manifest)
        explicit = fields(manifest_asset(manifest, args.asset_path), ASSET_FIELDS)
    else:
        require(not args.asset_path, "--asset-path needs --manifest")
    asset = explicit | {"open_status": args.open_status}
    if args.refusal_reason is not None:
        asset["refusal_reason"] = args.refusal_reason
    if args.author_label_source is not None:
        asset["author_label_source"] = args.author_label_source
    relative_path(asset["path"])
    return asset


def parse_describe(doc: Doc) -> Doc:
    require(isinstance(doc, dict), "expected a describe JSON object")

    def quantity(text: str, scales: dict[str, int]) -> float | None:
        parts = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)", text)
        if (not parts or any(unit not in scales for _, unit in parts)
                or re.sub(r"[0-9]+(?:\.[0-9]+)?\s*[a-zA-Z]+\s*", "", text.strip())):
            return None
        return finite(sum(float(number) * scales[unit] for number, unit in parts))

    channels = doc.get("# channels", "")
    raw = doc.get("data files", "").strip()
    templates = re.match(r"\s*([0-9]+)", doc.get("# templates", ""))
    level = ("raw" if raw not in {"", "-", "—"} else
             "waveforms" if templates and int(templates[1]) > 0 else "spike_times")
    return {
        "sample_rate_hz": quantity(doc.get("sample rate", ""), {"Hz": 1, "kHz": 1000}),
        "duration_s": (None if doc.get("n_samples") in (0, "0") else
                       quantity(doc.get("duration", ""), {"s": 1, "m": 60, "h": 3600})),
        "n_channels": int(channels) if re.fullmatch(r"[0-9]+", channels) else None,
        "evidence_level": level,
    }


def finite(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def qc_rows(path: Path | None) -> dict[int, tuple[int, Doc]]:
    if path is None:
        return {}
    rows = read_json(path, qc=True)
    require(isinstance(rows, list), f"{path}: expected an array of QC rows")
    result = {}
    seen = set()
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"{path}: QC row {index} must be an object")
        unit = row["cluster_id"]
        require(type(unit) is int and unit >= 0 and unit not in seen, f"{path}: invalid or duplicate cluster_id {unit}")
        seen.add(unit)
        count = row.get("n_spikes")
        require(type(count) is int and count >= 0, f"{path}: invalid n_spikes for cluster_id {unit}")
        if count > 0:
            result[unit] = (index, row)
    return result


def clusters(path: Path) -> dict[int, ClusterRow]:
    result = {}
    seen = set()
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        require(reader.fieldnames and {"id", "label", "n"} <= set(reader.fieldnames), f"{path}: expected id/label/n TSV")
        for row in reader:
            unit = int(row["id"])
            count = int(row["n"])
            require(unit >= 0 and count >= 0 and unit not in seen and row["label"] is not None,
                    f"{path}: invalid or duplicate cluster row {reader.line_num}")
            seen.add(unit)
            if not count:  # list_clusters itself omits empty slots.
                continue
            label = row["label"].strip().lower()
            result[unit] = (label if label in LABELS else "unsorted", reader.line_num, count, row.get("parents"))
    return result


def label_counts(labels: Iterable[str | None], suffix: str) -> Doc:
    values = list(labels)
    counts = {f"{label}_{suffix}": values.count(label) for label in LABELS}
    if suffix == "before":
        counts["unlabeled_before"] = values.count("unsorted")
    return counts


def curation_delta(path: Path, before: dict[int, ClusterRow] | None, live_before: set[int] | None
                   ) -> tuple[Doc, dict[int, tuple[str | None, str | None]]]:
    doc = read_json(path)
    steps = doc if isinstance(doc, list) else [doc]
    require(steps, f"{path}: expected one or more v2 curation steps")
    first_ids = steps[0]["unit_ids"]
    # SpikeInterface lists empty provider slots, too. Track live spikes and
    # original Sorrel cluster lineage separately from step membership.
    active = {uid: (before[i][0] if before is not None and i in before else None, None, i)
              for i, uid in enumerate(first_ids)}
    live = None if live_before is None else {uid for i, uid in enumerate(first_ids) if i in live_before}
    require(live is None or len(live) == len(live_before), f"{path}: curation does not enumerate all baseline units")
    summary = dict(n_merges=0, n_splits=0, n_removed=0)
    for index, step in enumerate(steps):
        require(step["format_version"] == "2", f"{path}: only SpikeInterface curation v2 is supported")
        require(not step.get("sorrel_unrepresentable"), f"{path}: contains Sorrel edits not representable in curation v2")
        listed = unique(step["unit_ids"], "curation unit_id")
        require(listed == set(active), f"{path}: step {index + 1} unit_ids do not match the preceding step")
        for kind in ("merges", "splits", "removed"):
            summary[f"n_{kind}"] += len(step.get(kind, []))
        for uid in step.get("removed", []):
            active.pop(uid)
            if live is not None:
                live.discard(uid)
        for merge in step.get("merges", []):
            target = merge["new_unit_id"]
            group = merge["unit_ids"]
            require(target not in active or target in group, f"{path}: merge target is not in its source group")
            moved_live = live is not None and any(uid in live for uid in group)
            for uid in group:
                active.pop(uid)
                if live is not None:
                    live.discard(uid)
            active[target] = (None, None, None)
            if live is not None and moved_live:
                live.add(target)
        for split in step.get("splits", []):
            parent = split["unit_id"]
            child, remainder = split["new_unit_ids"]
            require(live is None or parent in live, f"{path}: split of non-live unit {parent!r}")
            active.pop(parent)
            require(child != remainder and child not in active and remainder not in active,
                    f"{path}: split reuses a listed ID")
            if live is not None:
                live.remove(parent)
                live.update((child, remainder))
            active[child] = (None, None, None)
            active[remainder] = (None, None, None)
        prefix = f"/{index}" if isinstance(doc, list) else ""
        for i, label in enumerate(step.get("manual_labels", [])):
            uid = label["unit_id"]
            require(uid in active, f"{path}: label references absent unit {uid!r}")
            quality = label["labels"].get("quality")
            if quality is not None:
                require(isinstance(quality, list) and len(quality) <= 1 and all(x in LABELS or x == "unsorted" for x in quality),
                        f"{path}: invalid exclusive quality label")
                active[uid] = (quality[0] if quality else "unsorted", f"{prefix}/manual_labels/{i}/labels/quality",
                               active[uid][2])
            elif "origin" in label["labels"] and step.get("label_definitions", {}).get("quality", {}).get("exclusive"):
                # Sorrel's complete labels step omits quality for unsorted units.
                active[uid] = ("unsorted", f"{prefix}/manual_labels/{i}/labels", active[uid][2])
    if live is None:
        return summary, {}
    if not any(summary.values()):
        summary["units_after"] = len(live)
        if all(active[uid][0] is not None for uid in live):
            summary.update(label_counts((active[uid][0] for uid in live), "after"))
    return summary, {origin: (label, pointer) for uid in live for label, pointer, origin in [active[uid]]
                     if origin is not None and before is not None and origin in before}


def summarize_asset(args: argparse.Namespace) -> None:
    asset = asset_from_args(args)
    if args.unique_metadata is not None:
        asset["unique_metadata"] = json_object(args.unique_metadata, "--unique-metadata")
    thresholds = DEFAULT_THRESHOLDS.copy()
    if args.thresholds:
        overrides = (read_json(Path(args.thresholds[1:])) if args.thresholds.startswith("@")
                     else loads(args.thresholds))
        require(isinstance(overrides, dict), "--thresholds must be a JSON object")
        thresholds.update(overrides)
    check(thresholds, "review", "thresholds")
    findings: list[Doc] = []
    hashes: dict[Path, str] = {}

    def citation(path: Path, pointer: str) -> Doc:
        if path not in hashes:
            hashes[path] = digest_file(path)[0]
        return {"artifact_sha256": hashes[path], "pointer": pointer}

    def finding(code: str, message: str, severity: str = "major", **extra: Any) -> None:
        if args.emit_findings:
            if extra.get("unit_ids"):
                extra["unit_id_namespace"] = "sorrel_cluster_index"
            findings.append({"id": f"f{len(findings) + 1:02}", "code": code, "severity": severity,
                             "scope": "asset", "statement_md": message, "human_verified": False,
                             "asset_id": asset["asset_id"], **extra})

    if args.describe:
        asset.update(parse_describe(read_json(args.describe)))
    if args.evidence_level:
        asset["evidence_level"] = args.evidence_level
    if asset["open_status"] == "ok" and any((args.before_qc, args.after_qc, args.before_clusters, args.after_clusters)):
        asset.setdefault("evidence_level", "spike_times")
    recording_evidence = asset["open_status"] == "ok" and asset.get("evidence_level")
    if args.sample_rate is not None:
        require(finite(args.sample_rate) is not None and args.sample_rate > 0, "--sample-rate must be positive")
    asset["analysis_sample_rate_hz"] = args.sample_rate
    if recording_evidence != "raw" or (asset.get("duration_s") or 0) <= 0:
        asset["duration_s"] = None
    if not asset.get("duration_s") or args.sample_rate is not None or not asset.get("sample_rate_hz"):
        asset["sample_rate_hz"] = None
    if asset["open_status"] == "refused":
        finding("data.sorrel_refused", "Sorrel refused to open this asset; consult the recorded refusal.")

    before = clusters(args.before_clusters) if args.before_clusters else None
    after = clusters(args.after_clusters) if args.after_clusters else None
    snapshots = {}
    for suffix, labels, qc in (("before", before, args.before_qc), ("after", after, args.after_qc)):
        rows = qc_rows(qc)
        snapshots[suffix] = rows
        if qc and labels is not None:
            missing, extra = sorted(labels.keys() - rows.keys()), sorted(rows.keys() - labels.keys())
            require(not (missing or extra), f"{suffix} QC/TSV live IDs differ; "
                    f"missing QC IDs: {missing}; extra QC IDs: {extra}")
            for unit, (_, _, count, _) in labels.items():
                observed = rows[unit][1]["n_spikes"]
                require(observed == count, f"{suffix} QC/TSV spike-count mismatch for cluster {unit}: "
                        f"QC n_spikes={observed}, TSV n={count}")
        if labels is not None:
            asset.update({f"units_{suffix}": len(labels)}, **label_counts((v[0] for v in labels.values()), suffix))
        elif qc:
            asset[f"units_{suffix}"] = len(rows)
        if qc:
            quality = [value for _, row in rows.values() if (value := finite(row.get("composite_quality"))) is not None]
            asset[f"median_quality_{suffix}"] = statistics.median(quality) if quality else None

    rows = snapshots["before"]
    good = {u for u, row in (before or {}).items() if row[0] == "good"}
    rules = (
        ("refractory_contamination", "n_rp_contaminated", ("rpv_post_lo95",),
         ("rp_contamination_lo95_max",), "above the refractory contamination cutoff at the 95% credible lower bound"),
        ("refractory_screening", "n_rp_screened", ("srp_contamination",),
         ("srp_contamination_upper_max",), "flagged by the SRP upper confidence limit (screening only)"),
        ("refractory_screening", "n_isi_rate_flagged", ("isi_violation_ratio",),
         ("isi_violation_rate_max_hz",), "flagged by the ISI violation rate in Hz (screening only)"),
        ("low_presence", "n_low_presence", ("presence_ratio",),
         ("presence_ratio_min",), "below the presence ratio cutoff"),
        ("epoch_instability", "n_unstable", ("epoch_rate_ratio", "epoch_amp_ratio"),
         ("epoch_rate_ratio_min", "epoch_amp_ratio_min"), "below an epoch stability cutoff"),
        ("amplitude_cutoff", "n_amp_cutoff", ("amplitude_cutoff",),
         ("amplitude_cutoff_max",), "above the amplitude cutoff"),
        ("drift", None, ("drift_correlation",), ("drift_correlation_max",), "above the absolute drift correlation cutoff"),
    )
    for code, counter, metrics, cutoffs, message in rules:
        observed: set[int] = set()
        hits: dict[int, list[tuple[int, str]]] = {}
        for unit, (index, row) in sorted(rows.items()):
            for metric, cutoff in zip(metrics, cutoffs):
                if metric == "presence_ratio" and asset["duration_s"] is None:
                    continue
                if metric in {"amplitude_cutoff", "epoch_amp_ratio"} and (
                        recording_evidence not in {"raw", "waveforms"} or not finite(row.get("mean_amplitude"))):
                    continue
                value = finite(row.get(metric))
                if value is None:
                    continue
                observed.add(unit)
                threshold = thresholds[cutoff]
                if code == "drift":
                    value = abs(value)
                if (value < threshold if cutoff.endswith("_min") else value > threshold):
                    hits.setdefault(unit, []).append((index, metric))
        if counter and args.before_qc:
            asset[counter] = len(hits) if observed else None
            if counter in {"n_rp_contaminated", "n_rp_screened"} and before is not None:
                asset[f"n_good_{counter[2:]}"] = (
                    len(good & hits.keys()) if observed and (not good or observed & good) else None)
        if counter == "n_isi_rate_flagged":
            hits = {u: pointers for u, pointers in hits.items() if all(
                finite(rows[u][1].get(m)) is None for m in ("rpv_post_lo95", "srp_contamination"))}
        if hits and args.emit_findings:
            evidence = [citation(args.before_qc, f"/{index}/{metric}")
                        for pointers in hits.values() for index, metric in pointers]
            extra = {"metric": metrics[0]} if len(metrics) == 1 else {}
            if len(cutoffs) == 1:
                extra["threshold"] = thresholds[cutoffs[0]]
            finding(f"qc.{code}", f"{len(hits)} units are {message}.",
                    scope="unit", severity="info" if code == "refractory_screening" else "major",
                    unit_ids=list(hits), n_units=len(hits), evidence=evidence, **extra)

    mapped_after: dict[int, tuple[str | None, str | None]] = {}
    structural = False
    if args.curation:
        live_before = set(before) if before is not None else set(rows) if args.before_qc else None
        delta, mapped_after = curation_delta(args.curation, before, live_before)
        structural = any(delta[key] for key in ("n_merges", "n_splits", "n_removed"))
        # Native TSV is authoritative for the live Sorrel cluster namespace.
        asset.update({k: v for k, v in delta.items() if k not in asset})
        if structural and after is None:
            asset.update(dict.fromkeys(("units_after", "good_after", "mua_after", "noise_after",
                                        "n_relabelled", "n_noise_as_good")))
    if after is not None and before is not None:
        # A rekey can be absent from curation.json, but it changes the
        # Sorrel slot's parents. Don't count a structural descendant as a relabel.
        mapped_after = {
            u: (label, f"L{line}") for u, (label, line, count, parents) in after.items()
            if u in before and (args.curation and not structural or
                                (not args.curation or u in mapped_after)
                                and parents is not None and (count, parents) == before[u][2:])
        }
    if before is not None and (after is not None or args.curation and not structural):
        changed = sorted(u for u, (label, _) in mapped_after.items()
                         if label is not None and u in before and label != before[u][0])
        noise = [u for u in changed if u in good and mapped_after[u][0] == "noise"]
        measured = bool(mapped_after) or (args.curation and not structural)
        asset.update(n_relabelled=len(changed) if measured else None,
                     n_noise_as_good=len(noise) if measured else None)
        if noise and args.emit_findings:
            source = args.after_clusters or args.curation
            evidence = [citation(args.before_clusters, f"L{before[u][1]}") for u in noise]
            evidence.extend(citation(source, pointer) for u in noise if (pointer := mapped_after[u][1]))
            finding("qc.noise_as_good", f"{len(noise)} author-good units were relabeled noise in this review.",
                     scope="unit", n_units=len(noise), unit_ids=noise, evidence=evidence)
    if before and len(before) == asset.get("unlabeled_before") and args.emit_findings:
        finding("meta.no_quality_labels", "No live units have good, mua or noise labels in the baseline TSV.",
                 severity="info", evidence=[citation(args.before_clusters, "L1")])
    if args.nwb_inspector:
        messages = read_json(args.nwb_inspector)["messages"]
        require(isinstance(messages, list), "NWB Inspector messages must be an array")
        for severity, key, code in (
            ("CRITICAL", "nwbi_critical", "nwb.inspector_critical"),
            ("BEST_PRACTICE_VIOLATION", "nwbi_violation", "nwb.dandi_profile_violation"),
            ("BEST_PRACTICE_SUGGESTION", "nwbi_suggestion", None),
        ):
            indices = [i for i, msg in enumerate(messages)
                       if severity in (msg.get("severity"), msg.get("importance"))]
            asset[key] = len(indices)
            if indices and code and args.emit_findings:
                finding(code, f"NWB Inspector reported {len(indices)} {severity} messages; check the report and profile.",
                        severity="critical" if severity == "CRITICAL" else "major",
                        evidence=[citation(args.nwb_inspector, f"/messages/{i}") for i in indices])
    check(asset, "review", "asset")
    check_asset_counts(asset)
    emit({"asset": asset, "candidate_findings": findings} if args.emit_findings else asset, args.out)


def regular_file(path: Path) -> None:
    require(path.is_file() and not path.is_symlink(), f"expected regular file (no symlink): {path}")


def directory(path: Path) -> None:
    require(path.is_dir() and not path.is_symlink(), f"expected directory (no symlink): {path}")


def tree_files(root: Path) -> Iterable[Path]:
    directory(root)
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.is_symlink():
            yield from tree_files(child)
        else:
            regular_file(child)
            yield child


def artifact_path(path: str) -> None:
    relative_path(path)
    require(path == "report.md" or (path.startswith("artifacts/") and ":" not in path),
             f"artifact must be report.md or under artifacts/: {path!r}")


def check_asset_counts(asset: Doc) -> None:
    def known(*keys: str) -> bool:
        return all(asset.get(key) is not None for key in keys)

    def at_most(key: str, limit: str) -> None:
        require(not known(key, limit) or asset[key] <= asset[limit],
                f"{asset['asset_id']}: {key} cannot exceed {limit}")

    for state in ("before", "after"):
        total = f"units_{state}"
        labels = [f"{label}_{state}" for label in LABELS]
        if state == "before":
            labels.append("unlabeled_before")
        if known(total):
            labelled = sum(asset.get(key) or 0 for key in labels)
            require(labelled <= asset[total] and (
                state != "before" or not known(*labels) or labelled == asset[total]),
                f"{asset['asset_id']}: label counts inconsistent with {total}")
    for metric in ("contaminated", "screened"):
        all_units, good_units = f"n_rp_{metric}", f"n_good_rp_{metric}"
        at_most(good_units, "good_before")
        at_most(good_units, all_units)
        at_most(good_units, "units_before")
        at_most(all_units, "units_before")
    for metric in ("n_low_presence", "n_unstable", "n_amp_cutoff", "n_isi_rate_flagged"):
        at_most(metric, "units_before")
    at_most("n_noise_as_good", "good_before")
    level = asset.get("evidence_level")
    require(asset["open_status"] != "ok" or level is not None,
            f"{asset['asset_id']}: evidence_level is required when open_status is ok")
    if asset["open_status"] != "ok" or level is None:
        for key in ("duration_s", "sample_rate_hz", "n_low_presence", "n_amp_cutoff"):
            require(not known(key), f"{asset['asset_id']}: {key} requires an open asset with evidence")
    for key in ("duration_s", "sample_rate_hz", "analysis_sample_rate_hz"):
        require(not known(key) or asset[key] > 0, f"{asset['asset_id']}: {key} must be positive if measured")
    require(not known("n_low_presence",) or known("duration_s"),
            f"{asset['asset_id']}: n_low_presence needs a measured duration_s")
    if asset.get("evidence_level") in {"spike_times", "waveforms"}:
        require(not known("duration_s") and not known("sample_rate_hz"),
                f"{asset['asset_id']}: duration_s and sample_rate_hz need a native recording")
    if asset.get("evidence_level") == "spike_times":
        require(not known("n_amp_cutoff"), f"{asset['asset_id']}: n_amp_cutoff needs amplitude data")


def check_review(doc: Doc) -> None:
    check(doc, "review")
    asset_ids = unique((a["asset_id"] for a in doc["assets"]), "asset_id")
    unique((a["path"] for a in doc["assets"]), "asset path")
    ident, version = doc["dandiset"]["id"], doc["dandiset"]["version"]
    nwb_actual = 0
    for asset in doc["assets"]:
        relative_path(asset["path"])
        check_asset_counts(asset)
        nwb = is_nwb_asset(asset)
        require(asset.get("is_nwb", nwb) == nwb, f"{asset['asset_id']}: is_nwb disagrees with path and media_type")
        nwb_actual += nwb
        locator = asset.get("locator", "")
        prefix = "dandi://dandi/"
        if locator.startswith(prefix):
            pin, slash, path = locator[len(prefix):].partition("/")
            require(slash and pin == f"{ident}@{version}" and "?" not in path and "#" not in path
                    and unquote(path, errors="strict") == asset["path"],
                    f"{asset['asset_id']}: locator must match the pinned Dandiset ID, version, and asset path")
        if "download_url" in asset:
            require(asset["download_url"] == asset_download_url(asset["asset_id"]),
                    f"{asset['asset_id']}: download_url must be the pinned DANDI asset-ID URL")
        if version == "draft":
            receipt = asset["pin_verification"]
            require(receipt in {"local_checksum", "api_postcheck"},
                    f"{asset['asset_id']}: draft asset identity unverified; run verify-pin")
            if asset["encoding"] == "zarr":
                require(receipt == "api_postcheck" and asset.get("zarr_id") and asset.get("zarr_checksum"),
                        f"{asset['asset_id']}: verifying a draft Zarr requires its Zarr ID and checksum")
            else:
                require(asset.get("sha256"), f"{asset['asset_id']}: verifying a draft file requires its SHA-256")
    selection = doc["asset_selection"]
    archive_total = selection["n_assets_total"]
    nwb_total = selection.get("n_nwb_assets")
    require(nwb_total is None or nwb_actual <= nwb_total <= archive_total,
            "n_nwb_assets must cover the reviewed NWB assets and not exceed n_assets_total")
    nwb_only = selection.get("scope") == "nwb"
    require(not nwb_only or (nwb_total is not None and nwb_actual == len(asset_ids)),
            "NWB selection requires n_nwb_assets and only NWB assets")
    selected_total = nwb_total if nwb_only else archive_total
    require(len(asset_ids) <= selected_total, "assets exceed the selected scope total")
    if selection["policy"] == "all":
        require(len(asset_ids) == selected_total, "policy all requires all selected assets")
    finding_ids = unique((f["id"] for f in doc["findings"]), "finding id")
    unique((i["id"] for i in doc["improvements"]), "improvement id")
    ordered_paths = [f["path"] for f in doc["files"]]
    paths = unique(ordered_paths, "artifact path")
    require(ordered_paths == sorted(ordered_paths), "files must be sorted by artifact path")
    hashes = {f["sha256"] for f in doc["files"]}
    for path in paths:
        artifact_path(path)
    require("report.md" in paths, "missing listed report.md")
    for row in [*doc["findings"], *doc["improvements"]]:
        if row.get("asset_id") is not None:
            require(row["asset_id"] in asset_ids, f"{row['id']}: unknown asset_id {row['asset_id']}")
    for finding in doc["findings"]:
        metric = finding.get("metric")
        code = finding["code"]
        if metric is not None and code == "qc.refractory_contamination":
            require(metric == "rpv_post_lo95", f"{finding['id']}: contamination requires rpv_post_lo95")
        elif metric is not None and code == "qc.refractory_screening":
            require(metric in {"srp_contamination", "isi_violation_ratio"},
                    f"{finding['id']}: screening needs the SRP upper limit or ISI violation rate")
        for evidence in finding.get("evidence", []):
            require(evidence["artifact_sha256"] in hashes,
                    f"{finding['id']}: evidence hash is not in the bundle")
    for improvement in doc["improvements"]:
        if "deliverable_sha256" in improvement:
            require(improvement["deliverable_sha256"] in hashes,
                    f"{improvement['id']}: deliverable_sha256 is not in the bundle")
        for finding_id in improvement["addresses"]:
            require(finding_id in finding_ids, f"{improvement['id']}: unknown finding {finding_id}")


def json_pointer(value: Any, pointer: str) -> tuple[bool, Any]:
    if not pointer.startswith("/") or re.search(r"~(?![01])", pointer):
        return False, None
    for token in pointer[1:].split("/"):
        key = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif (isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", key)
              and len(key) <= len(str(len(value))) and int(key) < len(value)):
            value = value[int(key)]
        else:
            return False, None
    return True, value


def json_pointer_exists(value: Any, pointer: str) -> bool:
    return json_pointer(value, pointer)[0]


class HTMLIds(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not any(name == "id" for name, _ in attrs):
            return
        # Tokenize attribute values so id-like text inside another attribute doesn't count.
        for name, double, single in re.findall(
                r"""([^\s"'<>/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|[^\s>]+))?""",
                self.get_starttag_text()[1 + len(tag):]):
            if name.lower() == "id" and (double or single):
                self.ids.add(unescape(double or single))


def check_evidence(doc: Doc, paths: dict[str, Path]) -> None:
    json_kinds = {"cluster_qc_before", "cluster_qc_after", "describe_json", "curation", "nwb_inspector"}
    artifacts: dict[tuple[str, str], Any] = {}
    by_hash: dict[str, list[Doc]] = {}
    for row in doc["files"]:
        by_hash.setdefault(row["sha256"], []).append(row)
    for finding in doc["findings"]:
        for evidence in finding.get("evidence", []):
            sha, pointer = evidence["artifact_sha256"], evidence["pointer"]
            valid = []
            for row in by_hash[sha]:
                name, kind = row["path"], row["kind"]
                path = paths[name]
                suffix = path.suffix.lower()
                html = kind == "report_html" or suffix == ".html"
                if not html and (kind in json_kinds or suffix == ".json"):
                    mode = "qc" if kind in {"cluster_qc_before", "cluster_qc_after"} else "json"
                    key = name, mode
                    if key not in artifacts:
                        artifacts[key] = read_json(path, qc=mode == "qc")
                    valid.append(json_pointer_exists(artifacts[key], pointer))
                else:
                    key = name, "text"
                    if key not in artifacts:
                        try:
                            text = path.read_text(encoding="utf-8")
                        except UnicodeDecodeError:
                            artifacts[key] = None
                        else:
                            parser = HTMLIds() if html else None
                            if parser is not None:
                                parser.feed(text)
                                parser.close()
                            artifacts[key] = len(text.splitlines()), parser.ids if parser is not None else set()
                    metadata = artifacts[key]
                    if metadata is None:
                        continue
                    limit, anchors = metadata
                    if html and pointer.startswith("#"):
                        valid.append(pointer[1:] in anchors)
                        continue
                    lines = re.fullmatch(r"L([1-9][0-9]*)(?:-L([1-9][0-9]*))?", pointer)
                    valid.append(bool(lines and all(len(n) <= len(str(limit)) for n in lines.groups() if n)
                                      and int(lines[1]) <= int(lines[2] or lines[1]) <= limit))
            require(any(valid),
                    f"{finding['id']}: evidence pointer {pointer!r} does not resolve in artifact {sha}")


def bundle_doc(bundle: Path, review_id: str, dandiset_id: str | None = None) -> Doc:
    directory(bundle)
    review_path = bundle / "review.json"
    regular_file(review_path)
    raw = review_path.read_bytes()
    doc = loads(raw.decode("utf-8"))
    require(raw == canonical_json_bytes(doc), f"{review_path}: expected canonical review.json bytes")
    check_review(doc)
    actual = hashlib.sha256(raw).hexdigest()
    require(actual == review_id, f"{bundle}: review_id does not match review.json bytes (expected {actual})")
    if dandiset_id is not None:
        require(doc["dandiset"]["id"] == dandiset_id, f"{bundle}: wrong Dandiset directory")
    listed = {f["path"]: f for f in doc["files"]}
    actual_files = {p.relative_to(bundle).as_posix(): p for p in tree_files(bundle) if p != review_path}
    extra = actual_files.keys() - listed.keys()
    missing = listed.keys() - actual_files.keys()
    require(not extra, f"{bundle}: unlisted extra files: {', '.join(sorted(extra))}")
    require(not missing, f"{bundle}: missing files: {', '.join(sorted(missing))}")
    for name, path in actual_files.items():
        sha, size = digest_file(path)
        require((sha, size) == (listed[name]["sha256"], listed[name]["size_bytes"]),
                 f"{path}: artifact hash or size mismatch")
    check_evidence(doc, actual_files)
    return doc


def check_supersedes(reviews: dict[str, Doc]) -> None:
    checked: set[str] = set()
    for review_id, doc in reviews.items():
        chain: set[str] = set()
        node: str | None = review_id
        while node is not None and node not in checked:
            require(node in reviews and doc["dandiset"]["id"] == reviews[node]["dandiset"]["id"],
                    f"{review_id}: supersedes {node} is missing or belongs to a different Dandiset")
            require(node not in chain, f"{review_id}: supersedes cycle")
            chain.add(node)
            node = reviews[node].get("supersedes")
        checked.update(chain)
        if node is not None:
            require(doc["dandiset"]["id"] == reviews[node]["dandiset"]["id"],
                    f"{review_id}: supersedes {node} belongs to a different Dandiset")


def feedback_days(doc: Doc, key: str) -> int:
    return int(doc.get(key, schema("feedback")["properties"][key]["default"]))


def scheduled_reminder_epoch(doc: Doc, first_sent: datetime) -> int:
    due = epoch(first_sent) + feedback_days(doc, "reminder_after_days") * 86400
    require(INT64_MIN <= due <= INT64_MAX, "scheduled reminder epoch does not fit signed 64-bit SQLite INTEGER")
    return due


def check_feedback(doc: Doc, reviews: dict[str, Doc]) -> None:
    check(doc, "feedback")
    events = doc["events"]
    notified = any(event["status"] != "drafted" for event in events)
    review_ids = unique(doc["review_ids"], "feedback review_id")
    quoted = unique(((q["review_id"], q["finding_id"]) for q in doc["quoted_finding_ids"]), "quoted finding")
    for review_id in review_ids:
        require(review_id in reviews and reviews[review_id]["dandiset"]["id"] == doc["dandiset_id"],
                f"feedback review_id {review_id} is missing or belongs to a different Dandiset")
    for review_id, finding_id in sorted(quoted):
        require(review_id in review_ids, f"quoted review {review_id} must be in review_ids")
        row = next((f for f in reviews[review_id]["findings"] if f["id"] == finding_id), None)
        require(row is not None, f"quoted finding {review_id}:{finding_id} does not exist")
        require(not notified or row["human_verified"],
                f"quoted finding {review_id}:{finding_id} needs human_verified=true")

    dated = [(timestamp(e["at"]), e["status"]) for e in events]
    dates = [at for at, _ in dated]
    require(dates == sorted(dates), "feedback event dates must be nondecreasing")
    require(events[-1]["status"] == doc["status"], "feedback status must match the last event")
    first_sent = next((at for (at, status), event in zip(dated, events)
                       if status == "sent" and not event.get("early_followup")), None)
    reminder = next((at for at, status in dated if status == "reminded"), None)
    published = next((at for at, status in dated if status in PUBLISHED), None)
    eligible = first_sent + timedelta(days=feedback_days(doc, "disclosure_days")) if first_sent else None
    due = scheduled_reminder_epoch(doc, first_sent) if first_sent else None
    for key, expected in (("first_notified_at", first_sent), ("reminder_sent_at", reminder),
                          ("publish_eligible_at", eligible), ("published_at", published)):
        require((timestamp(doc[key]) if key in doc else None) == expected,
                f"{key} does not match the feedback event log or disclosure timer")
    if "extension_until" in doc:
        require(doc["status"] in RESPONSE and doc["extension_reason"].strip(),
                "extension needs an active response and a reason")
    subject = fields(doc, SUBJECT_FIELDS)
    previous = "drafted"
    for (at, status), event in zip(dated, events):
        if previous in TERMINAL:
            require(status == previous, f"cannot change outreach after {previous}")
        elif status in OUTREACH_ORDER:
            require(OUTREACH_ORDER.index(status) >= OUTREACH_ORDER.index(previous),
                    f"cannot return to {status} after {previous}")
        if (status in {"sent", "reminded"} or (status != "drafted" and previous == "drafted")
                or "subject" in event):
            require(event.get("subject") == subject, "feedback subject differs from its notified event snapshot")
        if status in {"sent", "reminded"}:
            for key in DESTINATION_FIELDS:
                require(event[key] == doc.get(key), f"feedback {key} differs from its notified event snapshot")
        if event.get("early_followup"):
            require(previous == "sent" and due is not None and epoch(at) < due,
                    "early follow-up requires a prior send and must precede the scheduled reminder")
        if status == "reminded":
            require(due is not None and epoch(at) >= due, "scheduled reminder requires reaching reminder_due")
        if status == "no_response_published":
            require(eligible is not None and at >= eligible and previous not in RESPONSE,
                    "no-response publication requires elapsed grace and no response")
        previous = status


def read_registry(store: Path) -> dict[str, Doc]:
    path = store / "dandisets.tsv"
    if not path.exists() and not path.is_symlink():
        return {}
    regular_file(path)
    with path.open(encoding="utf-8", newline="") as stream:
        table = csv.DictReader(stream, delimiter="\t")
        require(tuple(table.fieldnames or ()) == REGISTRY_FIELDS, f"{path}: unexpected TSV header")
        result = {}
        for row in table:
            ident = row["dandiset_id"]
            require(ident is not None and re.fullmatch(r"[0-9]{6}", ident) and ident not in result,
                    f"{path}: invalid or duplicate Dandiset id {ident!r}")
            require(None not in row and None not in row.values(), f"{path}: malformed TSV row {table.line_num}")
            result[ident] = row
    return result


def read_dataset(path: Path) -> Doc:
    directory(path.parent)
    regular_file(path)
    doc = read_json(path)
    try:
        check(doc, "dandiset")
        require(path.stem == doc["dandiset_id"], "filename does not match dandiset_id")
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return doc


def dataset_row(doc: Doc) -> Doc:
    return (fields(doc, (*REGISTRY_FIELDS, "description", "citation"))
            | json_columns(fields(doc, DATASET_JSON_FIELDS) | {"dandi_metadata": doc["dandi_raw_metadata"]},
                           *DATASET_JSON_FIELDS, "dandi_metadata"))


def load_store(store: Path) -> tuple[dict[str, Doc], dict[str, Doc], dict[str, Doc]]:
    # ponytail: one writer per local store; take a store lock before supporting concurrent writers.
    if store.exists() or store.is_symlink():
        directory(store)
    registry = read_registry(store)
    top = store / "dandisets"
    if top.exists() or top.is_symlink():
        directory(top)
        for path in sorted(top.glob("*.json")):
            doc = read_dataset(path)
            registry[path.stem] = registry.get(path.stem, {}) | dataset_row(doc)
    reviews: dict[str, Doc] = {}
    top = store / "reviews"
    if top.exists() or top.is_symlink():
        directory(top)
        for parent in sorted(top.iterdir()):
            directory(parent)
            require(re.fullmatch(r"[0-9]{6}", parent.name), f"{parent}: expected six-digit Dandiset directory")
            for bundle in sorted(parent.iterdir()):
                require(bundle.name not in reviews, f"duplicate review_id {bundle.name}")
                reviews[bundle.name] = bundle_doc(bundle, bundle.name, parent.name)
    check_supersedes(reviews)
    feedback = {}
    top = store / "feedback"
    if top.exists() or top.is_symlink():
        directory(top)
        for path in sorted(top.glob("*.json")):
            regular_file(path)
            doc = read_json(path)
            check_feedback(doc, reviews)
            require(path.stem == doc["feedback_id"], f"{path}: does not match feedback_id")
            feedback[path.stem] = doc
    return registry, reviews, feedback


def validate(args: argparse.Namespace) -> None:
    target = args.target.absolute()
    if target.is_dir() and not (target / "review.json").exists():
        require((target / "reviews").is_dir() and (target / "dandisets.tsv").is_file(),
                f"{target}: expected review bundle or store")
        preflight_projection(*load_store(target))
    elif target.parent.name == "feedback" and target.suffix == ".json":
        _, _, feedback = load_store(target.parent.parent)
        require(target.stem in feedback, f"no feedback at {target}")
    elif target.parent.name == "dandisets" and target.suffix == ".json":
        read_dataset(target)
    else:
        if target.is_dir():
            bundle = target
        else:
            require(target.name == "review.json", "expected a bundle, review.json, feedback JSON, or store")
            bundle = target.parent
        doc = bundle_doc(bundle, bundle.name)
        parent = bundle.parent
        if parent.parent.name == "reviews":
            _, reviews, _ = load_store(parent.parent.parent)
            require(bundle.name in reviews, f"{target}: not in the store")
        elif "supersedes" in doc:
            # Detached bundles can be validated together in a common directory.
            reviews = {bundle.name: doc}
            prev = doc.get("supersedes")
            while prev and prev not in reviews:
                require((parent / prev).exists(), f"{bundle}: missing supersedes target {prev}")
                reviews[prev] = bundle_doc(parent / prev, prev, doc["dandiset"]["id"])
                prev = reviews[prev].get("supersedes")
            check_supersedes(reviews)
    print(f"valid {target}")


def tabular_bytes(header: Iterable[str], rows: Iterable[Iterable[Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def write_registry(store: Path, registry: dict[str, Doc]) -> None:
    atomic_write(store / "dandisets.tsv", tabular_bytes(
        REGISTRY_FIELDS, ([row.get(key) for key in REGISTRY_FIELDS] for _, row in sorted(registry.items()))))


def scaffold(store: Path) -> None:
    store.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory(store)
    for name in ("reviews", "feedback", "dandisets"):
        path = store / name
        path.mkdir(mode=0o700 if name == "feedback" else 0o755, exist_ok=True)
        directory(path)
    files = {
        "dandisets.tsv": tabular_bytes(REGISTRY_FIELDS, ()),
        ".gitignore": (
            b"# Private correspondence, recipients and evidence URLs stay local.\n"
            b"# Share separately reviewed and redacted exports explicitly.\n"
            b"/feedback/\n/results.sqlite\n/results.sqlite-*\n/.projection-*/\n/.install-*/\n"
        ),
        "README.md": (
            b"# Sorrel DANDI review store\n\n"
            b"`reviews/` contains immutable bundles. `feedback/` is a mutable, possibly private\n"
            b"outreach log. Check permissions and redact private material before sharing.\n"
            b"`dandisets.tsv` is the registry. `dandisets/<dandiset_id>.json` stores pinned\n"
            b"public metadata and local inspection notes. Review notes before sharing.\n"
            b"`results.tsv` and `results.sqlite` are derived.\n"
            b"Run `dandi_review.py project STORE` to rebuild both. Do not edit installed\n"
            b"bundles; pack a new review with `supersedes` instead. Use one writer per store.\n"
        ),
    }
    for name, content in files.items():
        path = store / name
        if path.exists() or path.is_symlink():
            regular_file(path)
        else:
            with path.open("xb") as stream:
                stream.write(content)


def save_dataset(store: Path, doc: Doc, registry: dict[str, Doc],
                 reviews: dict[str, Doc], feedback: dict[str, Doc]) -> None:
    check(doc, "dandiset")
    ident = doc["dandiset_id"]
    registry = registry | {ident: registry.get(ident, {}) | dataset_row(doc)}
    preflight_projection(registry, reviews, feedback)
    scaffold(store)
    emit(doc, store / "dandisets" / f"{ident}.json")
    rebuild(store, registry, reviews, feedback)


def merge_metadata(old: Doc, new: Doc) -> Doc:
    return old | {key: merge_metadata(old[key], value)
                  if isinstance(value, dict) and isinstance(old.get(key), dict) else value
                  for key, value in new.items()}


def set_dataset_metadata(args: argparse.Namespace) -> None:
    store, ident = args.store_dir, args.dandiset_id
    require(re.fullmatch(r"[0-9]{6}", ident), "Dandiset id must have six digits")
    directory(store)
    registry, reviews, feedback = load_store(store)
    path = store / "dandisets" / f"{ident}.json"
    if path.exists():
        doc = read_dataset(path)
    else:
        row = registry.get(ident, {})
        envelopes = [review["dandiset"] for review in reviews.values() if review["dandiset"]["id"] == ident]
        require(args.name is not None or row or envelopes, f"Dandiset {ident} is not in the store; supply --name")
        name = (row.get("name") or next((entry["name"] for entry in envelopes if entry.get("name")), None)
                or f"DANDI:{ident}")
        doc = {
            "schema_version": "1", "dandiset_id": ident, "name": name,
            "dandi_raw_metadata": {}, "unique_metadata": {},
        }
    if args.name is not None:
        doc["name"] = args.name
    for key in LOCAL_DATASET_FIELDS:
        if (value := getattr(args, key)) is not None:
            doc[key] = value
    notes = doc["unique_metadata"]
    for value in args.merge_json:
        notes = merge_metadata(notes, json_object(value, "--merge-json"))
    for assignment in args.set:
        key, eq, value = assignment.partition("=")
        require(key and eq, "--set requires KEY=VALUE")
        try:
            notes[key] = loads(value)
        except json.JSONDecodeError:
            notes[key] = value
    doc.update(unique_metadata=notes, updated_at=utc(now_utc()))
    save_dataset(store, doc, registry, reviews, feedback)
    print(path)


def pack(args: argparse.Namespace) -> None:
    stage, store = args.staging_dir, args.store
    directory(stage)
    require(not store.resolve().is_relative_to(stage.resolve()), "store must not be inside the staging directory")
    for name in ("reviews", "feedback", "dandisets"):
        require(not stage.resolve().is_relative_to((store / name).resolve()),
                 "staging must not be inside the store's reviews/, feedback/, or dandisets/")
    review_path = stage / "review.json"
    regular_file(review_path)
    doc = read_json(review_path)
    require(isinstance(doc, dict) and isinstance(doc.get("files", []), list), "expected review.json with a files array")
    hints = {}
    for row in doc.get("files", []):
        require(isinstance(row, dict) and {"path", "kind"} <= row.keys()
                and row.keys() <= {"path", "kind", "sha256", "size_bytes"}, "invalid review.json file kind hint")
        artifact_path(row["path"])
        require(row["path"] not in hints, f"duplicate artifact hint: {row['path']}")
        hints[row["path"]] = row
    report = stage / "report.md"
    regular_file(report)
    artifacts = stage / "artifacts"
    inputs = [report]
    if artifacts.exists() or artifacts.is_symlink():
        inputs.extend(tree_files(artifacts))
    files = []
    sources = sorted((path.relative_to(stage).as_posix(), path) for path in inputs)
    require(hints.keys() <= {name for name, _ in sources}, "review.json hints reference absent artifacts")
    for name, path in sources:
        artifact_path(name)
        sha, size = digest_file(path)
        hint = hints.get(name, {})
        for key, actual in (("sha256", sha), ("size_bytes", size)):
            if key in hint:
                require(type(hint[key]) is type(actual) and hint[key] == actual,
                        f"{path}: supplied {key} does not match the artifact")
        files.append({"path": name, "kind": hint.get("kind", ARTIFACT_KINDS.get(path.name, "other")),
                      "sha256": sha, "size_bytes": size})
    doc["files"] = files
    check_review(doc)
    check_evidence(doc, dict(sources))
    canonical = canonical_json_bytes(doc)
    review_id = hashlib.sha256(canonical).hexdigest()
    ident = doc["dandiset"]["id"]
    destination = store / "reviews" / ident / review_id
    registry, reviews, feedback = load_store(store)
    reviews[review_id] = doc
    check_supersedes(reviews)
    preflight_projection(registry, reviews, feedback)
    if not args.dry_run:
        scaffold(store)
        if not destination.exists():
            destination.parent.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".install-", dir=store) as scratch:
                bundle = Path(scratch) / review_id
                bundle.mkdir()
                for name, path in sources:
                    target = bundle / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
                (bundle / "review.json").write_bytes(canonical)
                bundle_doc(bundle, review_id, ident)
                bundle.rename(destination)
        tsv = read_registry(store)
        if ident not in tsv:
            tsv[ident] = {"dandiset_id": ident, **fields(doc["dandiset"], REGISTRY_FIELDS[1:])}
            write_registry(store, tsv)
        rebuild(store, registry, reviews, feedback)
    print(f"{review_id}\t{destination}")


def insert(db: sqlite3.Connection, table: str, row: Doc) -> None:
    # Column identifiers are code-owned or taken from the bundled closed schemas.
    names = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    # Schema-bounded integer columns fit int64. Large integers in REAL fields
    # are legal JSON numbers but must be bound as floats by sqlite3.
    values = (float(v) if type(v) is int and not INT64_MIN <= v <= INT64_MAX else v for v in row.values())
    db.execute(f"INSERT INTO {table} ({names}) VALUES ({marks})", tuple(values))


def json_columns(row: Doc, *keys: str) -> Doc:
    return {f"{key}_json" if key in keys else key:
            json.dumps(val, ensure_ascii=False, allow_nan=False, sort_keys=True) if key in keys else val
            for key, val in row.items()}


def project_rows(db: sqlite3.Connection, registry: dict[str, Doc],
                 reviews: dict[str, Doc], feedback: dict[str, Doc]) -> None:
    registry = dict(registry)
    for doc in reviews.values():
        ident = doc["dandiset"]["id"]
        envelope = {"dandiset_id": ident, **fields(doc["dandiset"], REGISTRY_FIELDS[1:])}
        if ident not in registry:
            registry[ident] = envelope
        elif "dandi_metadata_json" in registry[ident]:
            registry[ident] = registry[ident] | {key: value for key, value in envelope.items()
                                                 if not registry[ident].get(key) and value}
    for _, row in sorted(registry.items()):
        insert(db, "dandisets", row)
    seen: set[tuple[str, str]] = set()
    for review_id, doc in sorted(reviews.items()):
        dandiset = doc["dandiset"]
        ident = dandiset["id"]
        selection = doc["asset_selection"]
        insert(db, "reviews", {
            "review_id": review_id,
            "dandiset_id": ident,
            **fields(dandiset, ("version", "draft_modified")),
            **fields(doc, ("mode", "skill_rev", "prompt_rev", "learned_rev", "supersedes")),
            "created_at": utc(timestamp(doc["created_at"])),
            "contributor": doc["contributor"]["name"],
            "sorrel_version": doc["sorrel"]["version"],
            "sorrel_rev": doc["sorrel"]["rev"],
            "selection_policy": selection["policy"],
            "selection_scope": selection.get("scope", "archive"),
            **fields(selection, ("n_assets_total", "n_nwb_assets")),
            **doc.get("timing", {}),
            **json_columns(fields(doc, ("unique_metadata",)), "unique_metadata"),
            "review_json": canonical_json_bytes(doc).decode("utf-8"),
        })
        for ordinal, model in enumerate(doc["models"]):
            insert(db, "review_models", {"review_id": review_id, "ordinal": ordinal, **json_columns(model, "usage")})
        for a in doc["assets"]:
            key = ident, a["asset_id"]
            if key not in seen:
                insert(db, "assets", {"dandiset_id": ident, "asset_id": a["asset_id"]})
                seen.add(key)
            insert(db, "review_assets", {"review_id": review_id, "dandiset_id": ident, "is_nwb": is_nwb_asset(a),
                                         **json_columns(a, "unique_metadata")})
        for table, field, lists in (("findings", "findings", ("unit_ids", "evidence")),
                                    ("improvements", "improvements", ("addresses",)),
                                    ("review_artifacts", "files", ())):
            for row in doc[field]:
                insert(db, table, {"review_id": review_id, **json_columns(row, *lists)})
    for feedback_id, doc in sorted(feedback.items()):
        columns = {k: v for k, v in doc.items() if k not in {"schema_version", "events"}}
        for key, epoch_key in (("first_notified_at", "first_notified_epoch"), ("reminder_sent_at", "reminder_sent_epoch"),
                               ("publish_eligible_at", "publish_eligible_epoch"), ("extension_until", "extension_until_epoch"),
                               ("published_at", "published_epoch")):
            if key in columns:
                at = timestamp(columns[key])
                columns[key] = utc(at)
                columns[epoch_key] = epoch(at)
                if key == "first_notified_at":
                    columns["reminder_due_epoch"] = scheduled_reminder_epoch(doc, at)
        insert(db, "feedback", json_columns(columns, "review_ids", "quoted_finding_ids"))
        for review_id in doc["review_ids"]:
            insert(db, "feedback_reviews", {"feedback_id": feedback_id, "review_id": review_id})
        for ordinal, event in enumerate(doc["events"]):
            insert(db, "feedback_events", {"feedback_id": feedback_id, "ordinal": ordinal,
                                            **json_columns(event, "subject"), "at": utc(timestamp(event["at"]))})


def populate(db: sqlite3.Connection, registry: dict[str, Doc], reviews: dict[str, Doc],
             feedback: dict[str, Doc]) -> None:
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript((HERE / "schema.sql").read_text(encoding="utf-8"))
    with db:
        project_rows(db, registry, reviews, feedback)


def preflight_projection(registry: dict[str, Doc], reviews: dict[str, Doc], feedback: dict[str, Doc]) -> None:
    with contextlib.closing(sqlite3.connect(":memory:")) as db:
        populate(db, registry, reviews, feedback)
        check_projected_views(db)


def check_projected_views(db: sqlite3.Connection) -> None:
    # Force evaluation: individually valid INTEGER and REAL values may overflow in a sum.
    for sql in ("SELECT * FROM v_scientific_impact", MODEL_COVERAGE, "SELECT * FROM v_disclosure_queue",
                "SELECT * FROM v_result_table"):
        for row in db.execute(sql):
            require(all(type(value) is not float or math.isfinite(value) for value in row),
                    "non-finite projection value")


def rebuild(store: Path, registry: dict[str, Doc], reviews: dict[str, Doc], feedback: dict[str, Doc]) -> None:
    with tempfile.TemporaryDirectory(prefix=".projection-", dir=store) as scratch:
        scratch_path = Path(scratch)
        db_path = scratch_path / "results.sqlite"
        tsv_path = scratch_path / "results.tsv"
        db_path.touch(mode=0o600)
        with contextlib.closing(sqlite3.connect(db_path)) as db:
            populate(db, registry, reviews, feedback)
            check_projected_views(db)
            rows = db.execute("SELECT * FROM v_result_table ORDER BY dandiset_id, version, review_id, path, asset_id")
            with tsv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(col[0] for col in rows.description)
                writer.writerows(rows)
        db_path.replace(store / db_path.name)
        tsv_path.replace(store / tsv_path.name)


def init_store(args: argparse.Namespace) -> None:
    store = args.store_dir
    scaffold(store)
    rebuild(store, *load_store(store))
    print(store)


def project(args: argparse.Namespace) -> None:
    store = args.store_dir
    directory(store)
    rebuild(store, *load_store(store))
    print(store / "results.sqlite")


def record_feedback(args: argparse.Namespace) -> None:
    store, ident = args.store_dir, args.dandiset
    require(re.fullmatch(r"[0-9]{6}", ident), "--dandiset must have six digits")
    directory(store)
    registry, reviews, feedback = load_store(store)
    feedback_id = args.feedback_id if args.feedback_id is not None else ident
    check_schema(feedback_id, schema("feedback")["properties"]["feedback_id"], schema("feedback"), "feedback_id")
    before = feedback.get(feedback_id, {})
    require(not before or before["dandiset_id"] == ident, f"feedback_id {feedback_id} belongs to a different Dandiset")
    frozen = bool(before and before["status"] != "drafted")
    doc = dict(before) or {
        "schema_version": "1", "feedback_id": feedback_id, "dandiset_id": ident,
        "review_ids": [], "quoted_finding_ids": [], "events": [],
    }
    for key in ("channel", "recipient", "url", "disclosure_days", "reminder_after_days",
                "resulting_version", "resulting_doi", "units_corrected", "response_summary"):
        value = getattr(args, key)
        if value is not None:
            if frozen and key in {"disclosure_days", "reminder_after_days"}:
                require(value == feedback_days(before, key), f"{key} is frozen after first notification")
            if "first_notified_at" in before and key in DESTINATION_FIELDS:
                require(value == before.get(key), f"{key} is frozen after notification; create a new --feedback-id")
            doc[key] = value
    for key in ("disclosure_days", "reminder_after_days"):
        doc.setdefault(key, feedback_days(doc, key))
        check_schema(doc[key], schema("feedback")["properties"][key], schema("feedback"), key)
    if args.review_id is not None:
        doc["review_ids"] = args.review_id
    if args.quote_finding is not None:
        quotes = []
        for item in args.quote_finding:
            review_id, separator, finding_id = item.partition(":")
            require(separator, "--quote-finding expects REVIEW_ID:FINDING_ID")
            quotes.append({"review_id": review_id, "finding_id": finding_id})
        doc["quoted_finding_ids"] = quotes
    subject = fields(doc, SUBJECT_FIELDS)
    require(not frozen or subject == fields(before, SUBJECT_FIELDS),
            "feedback subject is frozen after notification; create a new --feedback-id")
    status = args.status
    at = timestamp(args.at) if args.at is not None else now_utc()
    if args.early_followup:
        require(status == "reminded" and before.get("status") == "sent",
                "--early-followup requires --status reminded on an already sent record")
        status = "sent"
    doc["status"] = status
    event: Doc = {"at": utc(at), "status": status, "actor": args.actor}
    if status in {"sent", "reminded"}:
        event.update({key: doc.get(key) for key in DESTINATION_FIELDS})
    if status in {"sent", "reminded"} or (status != "drafted" and not frozen):
        event["subject"] = subject
    if args.early_followup:
        event["early_followup"] = True
    if args.evidence_url is not None:
        event["evidence_url"] = args.evidence_url
    if args.note is not None:
        event["note"] = args.note
    if args.extension_until is not None or args.extension_reason is not None:
        require(status in RESPONSE and args.extension_until and args.extension_reason and args.extension_reason.strip(),
                "extension requires an acknowledged or accepted response, an end date, and a reason")
        end = timestamp(args.extension_until)
        require(end > at, "extension must end after the response")
        doc.update(extension_until=utc(end), extension_reason=args.extension_reason)
        event["note"] = " ".join(filter(None, [event.get("note"),
                                             f"Extension until {utc(end)}: {args.extension_reason}"]))
    if status not in RESPONSE:
        doc.pop("extension_until", None)
        doc.pop("extension_reason", None)
    doc["events"] = [*doc["events"], event]
    if status == "sent" and "first_notified_at" not in doc:
        doc["first_notified_at"] = utc(at)
        doc["publish_eligible_at"] = utc(at + timedelta(days=doc["disclosure_days"]))
    if status == "reminded":
        doc.setdefault("reminder_sent_at", utc(at))
    if status in PUBLISHED:
        doc.setdefault("published_at", utc(at))
    check_feedback(doc, reviews)
    feedback[feedback_id] = doc
    preflight_projection(registry, reviews, feedback)
    directory(store / "feedback")
    path = store / "feedback" / f"{feedback_id}.json"
    atomic_write(path, canonical_json_bytes(doc))
    rebuild(store, registry, reviews, feedback)
    print(path)


MODEL_COVERAGE = """
WITH models AS (
    SELECT m.*, r.dandiset_id,
        (SELECT COUNT(*) FROM review_assets a WHERE a.review_id = m.review_id) AS n_assets
    FROM review_models m JOIN v_latest_reviews r ON r.review_id = m.review_id
), per_review AS (
    SELECT provider, model_id, role, review_id, dandiset_id, n_assets, COUNT(*) AS model_rows,
        SUM(cost_usd) AS cost_usd, SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens
    FROM models GROUP BY provider, model_id, role, review_id, dandiset_id, n_assets
)
SELECT provider, model_id, role, COUNT(*) AS n_reviews, COUNT(DISTINCT dandiset_id) AS n_dandisets,
    SUM(n_assets) AS n_assets, SUM(model_rows) AS model_rows, SUM(cost_usd) AS cost_usd,
    SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens
FROM per_review GROUP BY provider, model_id, role ORDER BY provider, model_id, role
"""


def impact(args: argparse.Namespace) -> None:
    store = args.store_dir
    directory(store)
    as_of = timestamp(args.as_of) if args.as_of is not None else now_utc()
    # ponytail: revalidate and project on every run; add content-addressed
    # caching only if evidence validation becomes a demonstrated bottleneck.
    with contextlib.closing(sqlite3.connect(":memory:")) as db:
        populate(db, *load_store(store))
        db.row_factory = sqlite3.Row
        summary = dict(db.execute("SELECT * FROM v_scientific_impact").fetchone())
        models = [dict(row) for row in db.execute(MODEL_COVERAGE)]
        # Bind the same clock used by the SQL view to this snapshot's as-of time.
        ddl = db.execute("SELECT sql FROM sqlite_master WHERE name = 'v_disclosure_queue'").fetchone()[0]
        _, sep, sql = ddl.partition(" AS\n")
        require(sep and DISCLOSURE_CLOCK in sql, "disclosure view is missing its clock")
        queue = db.execute(sql.replace(DISCLOSURE_CLOCK, "SELECT ?", 1) + " ORDER BY dandiset_id, feedback_id",
                           (epoch(as_of),))
        buckets: dict[str, list[Doc]] = {bucket: [] for bucket in BUCKETS}
        for row in queue:
            buckets[row["bucket"]].append(fields(dict(row), (
                "dandiset_id", "feedback_id", "status", "first_notified_at", "reminder_sent_at",
                "publish_eligible_at", "extension_until", "published_at",
            )))
    if args.format == "json":
        emit({"as_of": utc(as_of), "scientific_impact": summary, "models": models, "disclosure_queue": buckets}, None)
        return

    md = args.format == "md"

    def table(title: str, header: tuple[str, ...], rows: Iterable[Iterable[Any]]) -> None:
        print(f"{'## ' if md else ''}{title}")
        if md:
            print("| " + " | ".join(header) + " |")
            print("| " + " | ".join("---" for _ in header) + " |")
        else:
            print("\t".join(header))
        for row in rows:
            values = [str(value).replace("\n", " ") if value is not None else "N/A" for value in row]
            if md:
                print("| " + " | ".join(v.replace("|", r"\|") for v in values) + " |")
            else:
                print("\t".join(values))
        print()

    table("Scientific impact", ("Metric", "Value"), summary.items())
    model_fields = ("provider", "model_id", "role", "n_reviews", "n_dandisets", "n_assets",
                    "cost_usd", "input_tokens", "output_tokens")
    table("Model cost and coverage", model_fields, ([row[k] for k in model_fields] for row in models))
    table(f"Disclosure queue ({utc(as_of)})", ("Bucket", "Feedback records"),
          ((bucket, ", ".join(f"{row['dandiset_id']}/{row['feedback_id']} ({row['status']})" for row in entries) or "—")
           for bucket, entries in buckets.items()))


AUDIT_TEXT_RISKS = {
    "SUSPICIOUS_EPOCH_SPAN": r"1000(?:\.0+)?\s*[x×]|epochs?.{0,60}millisecond",
    "TRACE_SPIKE_SPAN_MISMATCH": (
        r"(?:trace|recording).{0,90}(?:span|duration).{0,55}(?:mismatch|shorter|longer|exceeds)"
        r"|spikes?.{0,45}(?:beyond|outside).{0,55}(?:trace|recording)"
        r"|(?:trace|duration).{0,100}vs\.?.{0,45}(?:sorted )?spike (?:extent|span)"
        r"|mismatch.{0,100}trace span.{0,100}(?:spike|sorted)"
    ),
    "CCG_ONLY_MERGE_SCREENING": r"merge-screen|ccg[- ]only|ccgs? alone|merge\?.{0,100}(?:ccg|waveform)",
    "RP_EPOCH_CONTEXT_CONFOUND": r"\bburst(?:s|ing|y)?\b|state transitions?|(?:cortical|opto\w*) stimulat|optogenetic_series",
    "UNMAPPED_AUTHOR_LABELS": (
        r"\bgood2\b|\bexcellent\b|numeric(?:al)? .{0,20}\bquality\b|\bquality\b.{0,30}numeric"
        r"|\b(?:cell_type|cell_class|unit_type|ACGquality|UnitStability|unit_quality_post_sorting)\b"
    ),
    "NONSTANDARD_CLOCK": (
        r"nominal.{0,45}(?:grid|clock)|(?:analysis|quantization)[- ]grid|"
        r"timestamps_reference_time.{0,25}mismatch|non[- ]standard clock"
    ),
    "WAVEFORM_PROBE_TIMEOUT": (
        r"waveform.{0,110}(?:timed? ?out|timeout|hit.{0,45}cap)|"
        r"(?:timed? ?out|timeout).{0,45}waveform"
    ),
    "INSPECTOR_TIMEOUT": r"(?:nwb.?inspector|nwbi).{0,160}(?:timed? ?out|timeout)",
}
PRIVATE_TEXT = re.compile(
    r"(?<!free of )\b(?:private|non[- ]public|unredacted|identifiable)\s+(?:human\s+)?"
    r"(?:subject|patient|participant|identifier|id)\b"
    r"|\b(?:mrn|medical record (?:number|id)|social security (?:number|id)|date of birth|patient name)\s*[:=#]"
    r"|\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I
)
AUDIT_VERDICTS = ("outreach_ready", "needs_human_review", "blocked")


def audit_text(text: str) -> set[str]:
    return {code for code, pattern in AUDIT_TEXT_RISKS.items() if re.search(pattern, text, re.I)}


def audit_result(blockers: set[str], warnings: set[str], hold: bool = False) -> Doc:
    return {"verdict": "blocked" if blockers - {"HUMAN_VERIFICATION_REQUIRED"}
            else "needs_human_review" if blockers or hold else "outreach_ready",
            "blockers": sorted(blockers), "warnings": sorted(warnings)}


def audit_notes(value: Any) -> str:
    # Do not reproduce source prose, paths, subject IDs or raw values in the audit.
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def audit_timeout(value: Any) -> bool:
    return bool(re.search(r'time[ -]?out|timed out|"(?:rc|exit_code)": 124\b', audit_notes(value), re.I))


def audit_asset(bundle: Path, asset: Doc, inventory: dict[str, Doc], listed: set[str], human: bool) -> Doc:
    blockers: set[str] = set()
    notes = asset.get("unique_metadata", {})
    text = audit_notes(notes)
    warnings = audit_text(text)
    path = PurePosixPath(asset["path"])
    archived = inventory.get(asset["path"])
    if archived is None:
        blockers.add("PIN_ASSET_NOT_IN_INVENTORY")
    elif any(asset.get(k) != archived.get(k)
             for k in ("asset_id", "size_bytes", "encoding", "blob_id", "sha256", "zarr_id", "zarr_checksum")):
        blockers.add("PIN_ASSET_MISMATCH")
    required = ("zarr_id", "zarr_checksum") if asset["encoding"] == "zarr" else ("blob_id", "sha256")
    if not all(asset.get(key) for key in required):
        blockers.add("PIN_STORAGE_IDENTITY_INCOMPLETE")
    if asset["open_status"] != "ok":
        warnings.add("ASSET_REFUSED" if asset["open_status"] == "refused" else "ASSET_OPEN_FAILED")
    if PRIVATE_TEXT.search(text) or PRIVATE_TEXT.search(asset.get("refusal_reason", "")):
        blockers.add("POSSIBLE_PRIVATE_IDENTIFIER")

    span = notes.get("session_span") or {}
    raw = notes.get("raw_access") or notes.get("raw_access_79eeaad")
    raw_note = audit_notes(raw)
    trace = (asset.get("evidence_level") == "raw" or
             isinstance(span, dict) and span.get("kind") == "measured" and span.get("source") == "trace")
    waveform = bool(re.search(r"windows\s+[1-9]\d*/\d+", raw_note)
                    or isinstance(raw, dict) and raw.get("verified") is True)
    if audit_timeout(raw):
        warnings.add("WAVEFORM_PROBE_TIMEOUT")
        waveform = False
    if (asset["open_status"] == "ok" and not asset.get("sample_rate_hz")
            and not any(notes.get(k) for k in (
                "selected_clock", "clock", "declared_units_resolution_s", "analysis_grid",
                "units_spike_times_resolution_s", "series_clock_metadata",
            ))):
        warnings.add("CLOCK_PROVENANCE_MISSING")
    if asset.get("analysis_sample_rate_hz"):
        warnings.add("NONSTANDARD_CLOCK")
    if any(audit_timeout(notes.get(k)) for k in ("nwbinspector", "nwb_inspector")):
        warnings.add("INSPECTOR_TIMEOUT")

    stem = path.name.removesuffix(".nwb.zarr").removesuffix(".nwb")
    dirs = {PurePosixPath("artifacts") / path, PurePosixPath("artifacts") / stem,
            PurePosixPath("artifacts") / path.parent / stem}
    artifact_files = {p.name: bundle / p for name in sorted(listed) if (p := PurePosixPath(name)).parent in dirs}
    if human and "nwb_meta.json" in artifact_files:
        metadata = read_json(artifact_files["nwb_meta.json"])
        subject = metadata.get("subject") or {}
        subject_id = subject.get("subject_id") or metadata.get("/general/subject/subject_id") or notes.get("subject")
        if isinstance(subject_id, dict):
            subject_id = subject_id.get("value")
        if isinstance(subject_id, (str, int)):
            public_ids = {p.removeprefix("sub-") for p in path.parts[:-1] if p.startswith("sub-")}
            if str(subject_id).removeprefix("sub-") not in public_ids:
                blockers.add("SUBJECT_IDENTIFIER_NOT_IN_PUBLIC_ASSET_PATH")
        elif asset["open_status"] == "ok":
            warnings.add("PUBLIC_SUBJECT_ID_NOT_RECONCILED")
    provenance = [notes.get("provenance") or {}]
    for name in ("describe.json", "cluster_qc.provenance.json", "cluster_qc.before.provenance.json"):
        if name in artifact_files:
            evidence = read_json(artifact_files[name])
            provenance.append(evidence.get("provenance") or {})
            span = evidence.get("session_span") or {}
            trace |= span.get("kind") == "measured" and span.get("source") == "trace"
    for source in provenance:
        if not isinstance(source, dict):
            continue
        trace |= source.get("mode") == "trace"
        for item in source.get("source_diagnostics", []):
            if item.get("code") == "non_increasing_source_spikes":
                blockers.add("NON_INCREASING_SOURCE_SPIKES")
                if item.get("equal_pairs", 0) > 0:
                    blockers.add("EQUAL_CONSECUTIVE_SOURCE_SPIKES")
            elif item.get("code") == "identical_cross_unit_spike_vectors":
                blockers.add("IDENTICAL_CROSS_UNIT_SPIKE_VECTORS")
            elif item.get("code") == "suspicious_declared_span":
                warnings.add("TRACE_SPIKE_SPAN_MISMATCH" if item.get("source") == "trace" else "SUSPICIOUS_EPOCH_SPAN")
        if any(v.get("unmapped") or not v.get("maps_quality")
               for v in source.get("author_label_columns", {}).values()):
            warnings.add("UNMAPPED_AUTHOR_LABELS")
    spike_extent = finite(notes.get("spike_extent_s"))
    epoch_span = finite(notes.get("epoch_span_s"))
    trace_span = finite(notes.get("trace_span_s")) or finite(asset.get("duration_s"))
    if spike_extent and spike_extent > 0:
        if epoch_span and epoch_span > 100 * spike_extent:
            warnings.add("SUSPICIOUS_EPOCH_SPAN")
        if trace_span and (trace_span > 2 * spike_extent or spike_extent > 1.01 * trace_span):
            warnings.add("TRACE_SPIKE_SPAN_MISMATCH")
    elif trace:
        warnings.add("TRACE_SPIKE_SPAN_NOT_RECONCILED")
    if trace and not waveform:
        warnings.add("TRACE_BACKED_NOT_WAVEFORM_VERIFIED")
    mapped_good = asset.get("good_before")
    if asset["open_status"] == "ok" and "clusters.before.tsv" in artifact_files:
        mapped_good = sum(label == "good" for label, *_ in clusters(artifact_files["clusters.before.tsv"]).values())
        if asset.get("good_before") != mapped_good:
            blockers.add("MAPPED_GOOD_DENOMINATOR_MISMATCH")
    elif mapped_good:
        warnings.add("MAPPED_GOOD_DENOMINATOR_UNCHECKED")
    false_critical = real_critical = 0
    if "nwb_inspector.json" in artifact_files:
        report = read_json(artifact_files["nwb_inspector.json"])
        messages = report.get("messages", []) if isinstance(report, dict) else report
        if isinstance(report, dict) and audit_timeout(report.get("status")):
            warnings.add("INSPECTOR_TIMEOUT")
        for row in messages:
            check_name, msg = row.get("check_function_name"), row.get("message", "")
            if check_name == "check_ascending_spike_times":
                blockers.add("NON_INCREASING_SOURCE_SPIKES")
                if re.search(r"equal|identical|duplicate", msg, re.I):
                    blockers.add("EQUAL_CONSECUTIVE_SOURCE_SPIKES")
            if check_name == "check_image_series_external_file_valid":
                match = re.search(r"external file ['\"]([^'\"]+)['\"]", msg, re.I)
                if match and (path.parent / match[1]).as_posix() in inventory:
                    warnings.add("INSPECTOR_EXTERNAL_FILE_PRESENT")
                    false_critical += row.get("importance") == "CRITICAL"
                    continue
            if check_name == "check_file_extension" and "remfile" in msg.lower():
                warnings.add("INSPECTOR_STREAMING_FILENAME_ARTIFACT")
            real_critical += row.get("importance") == "CRITICAL"
    return {"asset_id": asset["asset_id"], "open_status": asset["open_status"],
            "units": asset.get("units_before") or 0, "mapped_good": mapped_good,
            "trace_backed": trace, "waveform_probe_succeeded": waveform,
            "inspector_external_file_false_criticals": false_critical,
            "inspector_other_criticals": real_critical,
            **audit_result(blockers, warnings, bool(warnings))}


def audit_metric_claim(finding: Doc, evidence: list[Doc], by_digest: dict[str, str],
                       read_evidence: Callable[[str], Any]) -> str | None:
    metric = finding.get("metric")
    threshold = finite(finding.get("threshold"))
    if threshold is None or not metric:
        return None
    values = []
    for ref in evidence:
        name = by_digest[ref["artifact_sha256"]]
        pointer = ref.get("pointer", "")
        if name.endswith(".json") and pointer.rsplit("/", 1)[-1] == metric:
            _, value = json_pointer(read_evidence(name), pointer)
            values.append(finite(value))
    if not values:
        return "METRIC_THRESHOLD_EVIDENCE_UNMATCHED"
    below = finding["code"] in {"qc.low_presence", "qc.epoch_instability"}
    if any(v is None or not (v < threshold if below else v > threshold) for v in values):
        return "METRIC_THRESHOLD_EVIDENCE_CONTRADICTS_CLAIM"
    return None


def audit_bundle(bundle: Path, doc: Doc, snapshot: Doc,
                 overlap: dict[str, dict[str, list[Doc]]], active: bool) -> Doc:
    ident, version = doc["dandiset"]["id"], doc["dandiset"]["version"]
    inventory = {a["path"]: a for a in snapshot.get("assets", [])}
    selection = doc["asset_selection"]
    nwb_total = sum(is_nwb_asset(a) for a in inventory.values())
    listed = {f["path"] for f in doc["files"]}
    report = (bundle / "report.md").read_text(encoding="utf-8")
    text = "\n".join((report, selection.get("rationale", ""), audit_notes(doc.get("unique_metadata", {}))))
    warnings = audit_text(text)
    blockers: set[str] = set()
    if not snapshot:
        blockers.add("PIN_INVENTORY_MISSING")
    elif (snapshot.get("pinned_version"), snapshot.get("dandiset_id")) != (version, ident):
        blockers.add("PIN_VERSION_OR_DANDISET_MISMATCH")
        inventory = {}  # Never use a different version to exonerate an Inspector error.
    if (len(inventory) != len(snapshot.get("assets", [])) or len(inventory) != selection["n_assets_total"]
            or selection.get("n_nwb_assets", nwb_total) != nwb_total):
        blockers.add("SELECTION_DENOMINATOR_MISMATCH")
    excluded = (nwb_total if selection.get("scope") == "nwb" else len(inventory)) - len(doc["assets"])
    if excluded > 0:
        warnings.add("ASSETS_EXCLUDED")
        if not selection.get("rationale", "").strip():
            blockers.add("EXCLUSION_RATIONALE_MISSING")
    if PRIVATE_TEXT.search(text):
        blockers.add("POSSIBLE_PRIVATE_IDENTIFIER")
    if version == "draft":
        warnings.add("DRAFT_PIN_RECHECK_BEFORE_OUTREACH")
    shared = {path for path in inventory if overlap.get(path, {}).keys() - {ident}}
    selected_shared = sum(a["path"] in shared for a in doc["assets"])
    if shared:
        warnings.add("CROSS_ARCHIVE_SAME_PATH")
    same_content = 0
    for path in shared:
        asset = inventory[path]
        key = "zarr_checksum" if asset["encoding"] == "zarr" else "sha256"
        if asset.get(key) and any(asset[key] == other.get(key)
                                  for ds, peers in overlap[path].items() if ds != ident
                                  for other in peers):
            same_content += 1
    if same_content:
        warnings.add("CROSS_ARCHIVE_SAME_PATH_AND_DIGEST")
    if selected_shared:
        warnings.add("CROSS_ARCHIVE_SELECTED_PATH_REVIEW")

    human = any("homo sapiens" in s.lower() for s in snapshot.get("species", []))
    if human:
        warnings.add("HUMAN_DATA_PRIVACY_REVIEW")
    assets = [audit_asset(bundle, a, inventory, listed, human) for a in doc["assets"]]
    by_id = {a["asset_id"]: a for a in assets}
    by_digest = {f["sha256"]: f["path"] for f in doc["files"]}

    @lru_cache(maxsize=8)
    def read_evidence(name: str) -> Any:
        return read_json(bundle / name, qc=True)

    findings = []
    for f in sorted(doc["findings"], key=lambda f: f["id"]):
        claim = f["statement_md"]
        code = f["code"]
        risks = audit_text(claim)
        b = set(blockers)  # A finding cannot clear a broken review-wide pin or denominator.
        if f.get("human_verified") is not True:
            b.add("HUMAN_VERIFICATION_REQUIRED")
            if re.search(r"(?<!not )\b(?:rp confirmed|confirmed refractory|confirmed contamination)\b", claim, re.I):
                risks.add("UNVERIFIED_CONFIRMATION_WORDING")
        if PRIVATE_TEXT.search(claim):
            b.add("POSSIBLE_PRIVATE_IDENTIFIER")
        if code == "curation.duplicate_unit":
            b.add("IDENTICAL_CROSS_UNIT_SPIKE_VECTORS")
        evidence = f.get("evidence") or []
        if not evidence:
            risks.add("FINDING_EVIDENCE_MISSING")
        elif claim_issue := audit_metric_claim(f, evidence, by_digest, read_evidence):
            if claim_issue == "METRIC_THRESHOLD_EVIDENCE_CONTRADICTS_CLAIM":
                b.add(claim_issue)
            else:
                risks.add(claim_issue)
        asset = by_id.get(f.get("asset_id"))
        if asset:
            b.update(asset["blockers"])
            if (code.startswith("nwb.") and asset["inspector_external_file_false_criticals"]
                    and re.search(r"external|avi|critical", claim, re.I)):
                b.add("INSPECTOR_STREAMING_ARTIFACT_CLAIM")
        elif f["scope"] == "dandiset":
            b.update(code for row in assets for code in row["blockers"])
        if re.search(r"\b(?:merge|waveforms? verified)\b", claim, re.I) and not any(
                "waveform" in (by_digest[e["artifact_sha256"]] + e.get("pointer", "")).lower()
                for e in evidence):
            risks.add("WAVEFORM_SUPPORT_MISSING")
        if code.startswith(("qc.refractory", "qc.epoch")):
            risks.update((warnings | set(asset["warnings"] if asset else ())) & {
                "SUSPICIOUS_EPOCH_SPAN", "TRACE_SPIKE_SPAN_MISMATCH",
                "TRACE_SPIKE_SPAN_NOT_RECONCILED", "CLOCK_PROVENANCE_MISSING",
                "NONSTANDARD_CLOCK", "RP_EPOCH_CONTEXT_CONFOUND",
            })
        findings.append({"finding_id": f["id"], "code": code, **audit_result(b, risks, bool(risks))})
    if any(f.get("human_verified") is not True for f in doc["findings"]) and re.search(
            r"\b(?:rp confirmed|confirmed refractory|confirmed rp)\b", text, re.I):
        warnings.add("UNVERIFIED_CONFIRMATION_WORDING")
    if not findings:
        warnings.add("NO_FINDINGS_TO_SEND")
    for row in [*assets, *findings]:
        warnings.update(row["warnings"])
        blockers.update(row["blockers"])
    hold = bool(warnings - {"ASSETS_EXCLUDED", "CROSS_ARCHIVE_SAME_PATH",
                            "TRACE_BACKED_NOT_WAVEFORM_VERIFIED", "TRACE_SPIKE_SPAN_NOT_RECONCILED",
                            "INSPECTOR_STREAMING_FILENAME_ARTIFACT"})
    hold |= any(f["verdict"] == "needs_human_review" for f in findings)
    return {"version": version, "active": active,
            "archive_assets": len(inventory), "archive_nwb_assets": nwb_total,
            "excluded_in_scope": max(excluded, 0), "shared_archive_paths": len(shared),
            "shared_archive_paths_with_matching_digest": same_content,
            "shared_selected_paths": selected_shared, "assets": assets, "findings": findings,
            **audit_result(blockers, warnings, hold)}


def audit_review(args: argparse.Namespace) -> None:
    root = args.review_dir
    directory(root)
    outputs = [p.resolve() for p in (args.json_out, args.md_out) if p]
    require(len(set(outputs)) == len(outputs), "use different JSON and Markdown outputs")
    require(not any(p.is_relative_to(root.resolve()) for p in outputs), "audit outputs must be outside the input tree")
    single = (root / "review.json").exists()
    if single:
        regular_file(root / "review.json")
        paths = {hashlib.sha256((root / "review.json").read_bytes()).hexdigest(): root}
        docs = {rid: bundle_doc(root, rid) for rid in paths}
    else:
        _, docs, _ = load_store(root)
        paths = {rid: root / "reviews" / d["dandiset"]["id"] / rid for rid, d in docs.items()}
    require(docs, "no packed reviews found")
    snapshots = {}
    overlap: dict[str, dict[str, list[Doc]]] = {}
    for rid, doc in docs.items():
        pin = paths[rid] / "artifacts/dandiset.json"
        snapshot = read_json(pin) if any(f["path"] == "artifacts/dandiset.json" for f in doc["files"]) else {}
        snapshots[rid] = snapshot
        if ((snapshot.get("dandiset_id"), snapshot.get("pinned_version"))
                == (doc["dandiset"]["id"], doc["dandiset"]["version"])):
            for asset in snapshot["assets"]:
                overlap.setdefault(asset["path"], {}).setdefault(snapshot["dandiset_id"], []).append(asset)
    superseded = {d.get("supersedes") for d in docs.values()}
    grouped: dict[str, list[Doc]] = {}
    for rid, doc in sorted(docs.items()):
        row = audit_bundle(paths[rid], doc, snapshots[rid], overlap, rid not in superseded)
        row["review_id"] = rid
        grouped.setdefault(doc["dandiset"]["id"], []).append(row)
    datasets = []
    for ident, reviews in sorted(grouped.items()):
        active = [row for row in reviews if row["active"]]
        blockers = {code for row in active for code in row["blockers"]}
        warnings = {code for row in active for code in row["warnings"]}
        if len(active) > 1:
            warnings.add("MULTIPLE_ACTIVE_REVIEWS")
        hold = len(active) > 1 or any(row["verdict"] == "needs_human_review" for row in active)
        datasets.append({"dandiset_id": ident, "reviews": reviews, **audit_result(blockers, warnings, hold)})
    active = [r for d in datasets for r in d["reviews"] if r["active"]]
    active_assets = [a for r in active for a in r["assets"]]
    active_findings = [f for r in active for f in r["findings"]]
    def counts(rows: list[Doc]) -> Doc:
        return {v: sum(r["verdict"] == v for r in rows) for v in AUDIT_VERDICTS}

    summary = {
        "dandisets": counts(datasets), "findings": counts(active_findings),
        "review_bundles": len(docs), "active_review_bundles": len(active),
        "assets": dict(sorted(Counter(a["open_status"] for a in active_assets).items())),
        "units": sum(a["units"] for a in active_assets),
    }
    result = {"schema_version": 1, "summary": summary, "dandisets": datasets}
    if args.batch_summary:
        active_ids = {r["review_id"] for r in active}
        matched: set[str] = set()
        batches = []
        for path in sorted(args.batch_summary):
            hashes = set(re.findall(r"\b[0-9a-f]{64}\b", path.read_text(encoding="utf-8")))
            batch = active_ids & hashes
            matched.update(batch)
            batches.append({"file": path.name, "active_review_ids_matched": len(batch)})
        result["batch_summary_crosscheck"] = {
            "files": batches, "matched_active_reviews": len(matched),
            "unmatched_active_review_ids": sorted(active_ids - matched),
        }
    if args.md_out:
        lines = ["# DANDI pre-outreach audit", "",
                 "Offline screening of pinned review bundles, not a new source inspection or permission to contact labs.",
                 "Counts exclude superseded reviews. No finding prose or subject identifiers are included.", ""]
        for label in ("dandisets", "findings", "assets"):
            lines.append(f"- {label.capitalize()}: " + ", ".join(f"{k}: {v}" for k, v in summary[label].items()))
        lines += [f"- Review bundles: {len(active)} active of {len(docs)}; units: {summary['units']}", ""]
        if args.batch_summary:
            batch_counts = ", ".join("{}: {}".format(b["file"], b["active_review_ids_matched"]) for b in batches)
            lines += [f"- Batch summaries cite {len(matched)}/{len(active)} active review IDs "
                      f"({batch_counts}).", ""]
        for d in datasets:
            lines += [f"## DANDI:{d['dandiset_id']} — {d['verdict']}", ""]
            for row in d["reviews"]:
                state = row["verdict"] if row["active"] else "superseded"
                lines.append(f"- `{row['review_id']}` ({row['version']}): {state}; "
                             f"codes: {', '.join(row['blockers'] + row['warnings']) or 'none'}")
                for f in row["findings"]:
                    lines.append(f"  - `{f['finding_id']}` `{f['code']}`: {f['verdict']}; "
                                 f"{', '.join(f['blockers'] + f['warnings']) or 'no flags'}")
            lines.append("")
        atomic_write(args.md_out, ("\n".join(lines) + "\n").encode())
    if args.json_out or not args.md_out:
        emit(result, args.json_out)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    for name, handler in (("init-store", init_store), ("project", project), ("impact", impact)):
        cmd = sub.add_parser(name)
        cmd.add_argument("store_dir", type=Path)
        if name == "impact":
            cmd.add_argument("--format", choices=("text", "md", "json"), default="text")
            cmd.add_argument("--as-of")
        cmd.set_defaults(handler=handler)

    cmd = sub.add_parser("audit-review", help="screen packed DANDI reviews before outreach")
    cmd.add_argument("review_dir", type=Path, help="packed review directory or complete review store")
    cmd.add_argument("--json-out", type=Path)
    cmd.add_argument("--md-out", type=Path)
    cmd.add_argument("--batch-summary", type=Path, action="append", help="cross-check active review IDs cited in a batch summary")
    cmd.set_defaults(handler=audit_review)

    cmd = sub.add_parser("pin")
    cmd.add_argument("dandiset_id")
    cmd.add_argument("--version")
    cmd.add_argument("--allow-draft", action="store_true")
    cmd.add_argument("--nwb-only", action=argparse.BooleanOptionalAction, default=True,
                     help="list only NWB candidates (default); --no-nwb-only lists all assets")
    cmd.add_argument("--api-base", default=API_BASE)
    cmd.add_argument("--out", type=Path)
    cmd.add_argument("--store", type=Path)
    cmd.add_argument("--dataset-out", type=Path, help="write an optional Dandiset metadata snapshot")
    cmd.set_defaults(handler=pin)

    dataset = sub.add_parser("dataset-meta")
    dataset.add_argument("store_dir", type=Path)
    cmd = dataset.add_subparsers(dest="dataset_command", required=True).add_parser("set")
    cmd.add_argument("dandiset_id")
    cmd.add_argument("--name")
    cmd.add_argument("--merge-json", action="append", default=[], metavar="JSON|@FILE")
    cmd.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    for key in LOCAL_DATASET_FIELDS:
        cmd.add_argument(f"--{key}", action="append")
    cmd.set_defaults(handler=set_dataset_metadata)

    cmd = sub.add_parser("verify-pin")
    cmd.add_argument("--manifest", type=Path, required=True)
    cmd.add_argument("--asset-path", required=True)
    cmd.add_argument("--local-file", type=Path)
    cmd.add_argument("--api-base", default=API_BASE)
    cmd.add_argument("--out", type=Path, help="write verified manifest here instead of updating it in place")
    cmd.set_defaults(handler=verify_pin)

    cmd = sub.add_parser("summarize-asset")
    for key in ("manifest", "describe", "before-qc", "after-qc", "before-clusters", "after-clusters",
                "curation", "nwb-inspector", "out"):
        cmd.add_argument(f"--{key}", type=Path)
    for key in ("asset-path", "refusal-reason", "author-label-source", "thresholds", "unique-metadata",
                *(k.replace("_", "-") for k in ASSET_FIELDS if k not in {"size_bytes", "encoding", "is_nwb"})):
        cmd.add_argument(f"--{key}")
    cmd.add_argument("--size-bytes", type=int)
    cmd.add_argument("--sample-rate", type=float, help="explicit Sorrel analysis grid, in Hz")
    cmd.add_argument("--encoding", choices=("hdf5", "zarr", "other"))
    cmd.add_argument("--open-status", choices=("ok", "refused", "failed"), default="ok")
    cmd.add_argument("--evidence-level", choices=("spike_times", "waveforms", "raw"))
    cmd.add_argument("--emit-findings", action="store_true")
    cmd.set_defaults(handler=summarize_asset)

    cmd = sub.add_parser("pack")
    cmd.add_argument("staging_dir", type=Path)
    cmd.add_argument("--store", type=Path, required=True)
    cmd.add_argument("--dry-run", action="store_true")
    cmd.set_defaults(handler=pack)

    cmd = sub.add_parser("validate")
    cmd.add_argument("target", type=Path)
    cmd.set_defaults(handler=validate)

    feedback = sub.add_parser("feedback")
    feedback.add_argument("store_dir", type=Path)
    cmd = feedback.add_subparsers(dest="feedback_command", required=True).add_parser("record")
    cmd.add_argument("--dandiset", required=True)
    cmd.add_argument("--feedback-id", help="outreach record ID; defaults to the Dandiset ID")
    cmd.add_argument("--status", choices=schema("feedback")["$defs"]["status"]["enum"], required=True)
    cmd.add_argument("--channel", choices=schema("feedback")["$defs"]["channel"]["enum"])
    cmd.add_argument("--actor", required=True)
    cmd.add_argument("--review-id", action="append")
    cmd.add_argument("--quote-finding", action="append")
    cmd.add_argument("--early-followup", action="store_true",
                     help="record an early follow-up without completing the scheduled reminder")
    for key in ("recipient", "url", "at", "extension-until", "extension-reason", "resulting-version",
                "resulting-doi", "response-summary", "evidence-url", "note"):
        cmd.add_argument(f"--{key}")
    for key in ("disclosure-days", "reminder-after-days", "units-corrected"):
        cmd.add_argument(f"--{key}", type=int)
    cmd.set_defaults(handler=record_feedback)
    return cli


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    try:
        args.handler(args)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, AttributeError, sqlite3.Error) as exc:
        cli.exit(2, f"{cli.prog}: error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

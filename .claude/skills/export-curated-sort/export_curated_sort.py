# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "zstandard"]
# ///
"""Package a phy-curated Kilosort sort for the sorrel double-blind bench.

    export_curated_sort.py pack PHY_DIR OUT_DIR [--name NAME] [--level 19] [--jobs N] [--dat PATH]
    export_curated_sort.py unpack PKG OUT_DIR

`pack` copies only the allowlisted phy files (plus a scrubbed params.py and the
SpikeGLX .meta), compresses the raw recording with a chunked filter+zstd codec,
verifies both by decoding them again, and writes one uncompressed
`NAME.sorrel-upload.tar` plus its `.sha256`. PHY_DIR is only ever read.
`unpack` rebuilds a phy directory (recording.bin included) and re-runs the checks.
"""

import argparse
import ast
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zstandard

VERSION = "1"
REQUIRED = ["params.py", "spike_times.npy", "spike_templates.npy", "spike_clusters.npy",
            "templates.npy", "channel_positions.npy", "channel_map.npy", "amplitudes.npy"]
OPTIONAL = ["cluster_group.tsv", "template_ind.npy", "pc_features.npy", "pc_feature_ind.npy",
            "template_features.npy", "template_feature_ind.npy", "similar_templates.npy",
            "channel_shanks.npy", "whitening_mat_inv.npy", "cluster_KSLabel.tsv",
            "cluster_ContamPct.tsv", "cluster_Amplitude.tsv"]
SCRUB_META = (b"fileName", b"userNotes")  # SpikeGLX .meta keys that hold local paths and free text
KEEP_PARAMS = ["n_channels_dat", "dtype", "offset", "sample_rate", "hp_filtered"]
# Hidden knob for the self-check: chunk size in bytes.
CHUNK = int(os.environ.get("SORREL_EXPORT_CHUNK_BYTES", 64 << 20))
SAMPLE_SECONDS = 10
COUNTS = ["n_spikes", "n_templates", "n_clusters", "n_merged_clusters", "n_split_templates"]


def die(msg):
    sys.exit(f"error: {msg}")


def warn(msg, sink=None):
    print(f"WARNING: {msg}", file=sys.stderr)
    if sink is not None:
        sink.append(msg)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def pmap(fn, items, workers):
    """Ordered process-pool map with a bounded in-flight window, so finished
    results never pile up in memory when the consumer (disk, sha256) is slower."""
    with ProcessPoolExecutor(workers) as ex:
        pend = deque()
        for it in items:
            pend.append(ex.submit(fn, it))
            # Window = workers, not 2x: a queued result is a whole decoded chunk in this process,
            # which the per-worker memory budget (n_workers) does not count.
            if len(pend) >= workers:
                yield pend.popleft().result()
        while pend:
            yield pend.popleft().result()


def avail_mem():
    try:
        if sys.platform.startswith("linux"):
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError, KeyError):
        return None  # Windows, or a libc without the sysconf name: no cap


def n_workers(jobs, level, chunk):
    if jobs:
        return max(1, jobs)
    n = os.cpu_count() or 1
    mem = avail_mem()
    if mem is not None:
        # ponytail: budget = 4 chunk copies (memmap page cache, filtered copy, shuffled copy,
        # frame) + zstd's own context estimate; a rough bound, not a measurement. Pass --jobs
        # to override when the host knows better.
        ctx = zstandard.ZstdCompressionParameters.from_level(level).estimated_compression_context_size()
        n = min(n, max(1, mem // (chunk * 4 + ctx)))
    return n


# ---- params.py -------------------------------------------------------------

def read_params(path):
    """params.py is a file of `name = literal` lines; parse it without exec."""
    out = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass  # non-literal expression (os.path.join(...)): not one of ours anyway
    return out


def scrubbed_params(p):
    lines = ["dat_path = 'recording.bin'", "offset = 0"]
    lines += [f"{k} = {p[k]!r}" for k in KEEP_PARAMS if k in p and k != "offset"]
    return ("\n".join(lines) + "\n").encode()


def scrubbed_meta(path):
    """The SpikeGLX .meta with the values of SCRUB_META blanked (every other key kept).
    Returns (bytes, names of the keys that were blanked)."""
    out, hit = [], []
    for line in Path(path).read_bytes().splitlines(keepends=True):
        eol = line[len(line.rstrip(b"\r\n")):]
        key, sep, val = line.rstrip(b"\r\n").partition(b"=")
        if sep and key in SCRUB_META and val:
            line, hit = key + sep + eol, hit + [key.decode()]
        out.append(line)
    return b"".join(out), hit


def find_dat(p, phy, override):
    if override:
        d = Path(override)
        d.is_file() or die(f"--dat {override} is not a file")
        return d
    dp = p.get("dat_path")
    if isinstance(dp, (list, tuple)):
        die("multi-file dat_path is not supported; concatenate first and pass --dat")
    if not isinstance(dp, str):
        die("params.py has no dat_path; pass --dat")
    d = Path(dp)
    for cand in (d, phy / d, phy / re.split(r"[\\/]", dp)[-1]):  # last: foreign-machine path, same file name
        if cand.is_file():
            return cand
    die(f"raw recording not found (params dat_path {dp!r}, nor next to the phy dir); pass --dat PATH")


# ---- consistency checks ----------------------------------------------------

def check(d, n_samples, sr):
    """Cross-file sanity checks and merge/split counts on a phy directory."""
    d = Path(d)
    ld = lambda n: np.load(d / n, mmap_mode="r", allow_pickle=False)
    st, sc, sp, amp = (ld(n).reshape(-1) for n in
                       ("spike_times.npy", "spike_clusters.npy", "spike_templates.npy", "amplitudes.npy"))
    if not len(st) == len(sc) == len(sp) == len(amp):
        die(f"length mismatch: spike_times {len(st)}, spike_clusters {len(sc)}, "
            f"spike_templates {len(sp)}, amplitudes {len(amp)}")
    n = len(st)
    n or die("no spikes")
    warns = []
    # Kilosort pads batches, so a few spikes land just past the last sample (seen in real
    # sorts: 5 spikes, <= 14 samples). ponytail: tolerate up to 0.1 s past the end, error beyond.
    if int(st.max()) >= n_samples + int(0.1 * sr):
        die(f"max spike time {int(st.max())} >= recording length {n_samples} samples")
    if int(st.max()) >= n_samples:
        warn(f"{int((st >= n_samples).sum())} spike(s) lie up to {int(st.max()) - n_samples + 1} samples past the "
             "end of the recording (Kilosort batch padding)", warns)
    tpl = ld("templates.npy")
    if tpl.shape[0] <= int(sp.max()):
        die(f"spike_templates reference template {int(sp.max())} but templates.npy has {tpl.shape[0]}")
    rows = ld("channel_positions.npy").shape[0]
    ti = ld("template_ind.npy") if (d / "template_ind.npy").exists() else None
    if rows != tpl.shape[2] and not (ti is not None and (rows == ti.shape[1] or int(ti.max()) < rows)):
        warn(f"channel_positions has {rows} rows, templates have {tpl.shape[2]} channels", warns)
    # ponytail: unique (cluster, template) pairs are built in 16M-spike slices; memory is
    # bounded by the number of distinct pairs, which is tiny next to the spike count.
    M = int(sp.max()) + 1
    pairs, same = np.empty(0, np.int64), True
    for i in range(0, n, 1 << 24):
        c, t = sc[i:i + (1 << 24)].astype(np.int64), sp[i:i + (1 << 24)].astype(np.int64)
        same = same and bool(np.array_equal(c, t))
        pairs = np.union1d(pairs, c * M + t)
    if same:
        warn("spike_clusters == spike_templates everywhere: no merges/splits, was the curation saved in phy?", warns)
    cl, tp = pairs // M, pairs % M
    return {"n_spikes": n, "n_templates": int(tpl.shape[0]), "n_clusters": int(np.unique(cl).size),
            "n_merged_clusters": int((np.unique(cl, return_counts=True)[1] >= 2).sum()),
            "n_split_templates": int((np.unique(tp, return_counts=True)[1] >= 2).sum()), "warnings": warns}


# ---- raw codec -------------------------------------------------------------

def fwd(a, f):
    """Filter an (n, C) array to bytes. Integer arithmetic wraps by dtype."""
    if f.startswith("diff"):
        a = np.concatenate([a[:1], a[1:] - a[:-1]])
    if f.endswith("shuffle"):  # one plane per byte of the element
        return np.ascontiguousarray(a).reshape(-1).view(np.uint8).reshape(-1, a.itemsize).T.tobytes()
    return a.tobytes()


def inv(b, f, n, C, dt):
    if f.endswith("shuffle"):
        b = np.frombuffer(b, np.uint8).reshape(dt.itemsize, -1).T.tobytes()
    a = np.frombuffer(b, dt).reshape(n, C)
    return np.cumsum(a, axis=0, dtype=dt) if f.startswith("diff") else a


def enc_job(job):
    dat, base, dtype, C, start, n, f, level = job
    dt = np.dtype(dtype)
    a = np.memmap(dat, dt, "r", base + start * C * dt.itemsize, (n, C))
    return zstandard.ZstdCompressor(level=level, write_checksum=True).compress(fwd(np.asarray(a), f))


def dec_job(job):
    zst, lo, hi, f, n, C, dtype = job
    dt = np.dtype(dtype)
    with open(zst, "rb") as fh:
        fh.seek(lo)
        buf = fh.read(hi - lo)
    raw = zstandard.ZstdDecompressor().decompress(buf, max_output_size=n * C * dt.itemsize)
    return inv(raw, f, n, C, dt).tobytes()


def decode_stream(zst, info, workers, sink=None):
    """Decode every chunk (in parallel), return sha256 of the stream; write to sink if given."""
    h, b = hashlib.sha256(), info["chunk_bounds"]
    jobs = [(str(zst), info["chunk_offsets"][i], info["chunk_offsets"][i + 1], info["filter"],
             b[i + 1] - b[i], info["n_channels"], info["dtype"]) for i in range(len(b) - 1)]
    for blob in pmap(dec_job, jobs, workers):
        h.update(blob)
        sink and sink.write(blob)
    return h.hexdigest()


def pick_filter(dat, base, dtype, C, n_samples, sr, level, workers):
    """Compress a sample with every candidate filter. Returns (candidate sizes, sample rows, chunk rows)."""
    dt = np.dtype(dtype)
    rows = max(1, CHUNK // (C * dt.itemsize))
    filters = ["diff+shuffle", "diff", "shuffle", "none"] if dt.kind in "iu" else ["shuffle", "none"]
    # ponytail: the sample is ~10 s but capped at one chunk so it fits the per-worker memory
    # budget; a recording whose first seconds are unrepresentative can mis-pick. Raise the cap
    # or sample several windows if that ever shows up in the candidate sizes.
    m = min(n_samples, rows, max(1, int(sr * SAMPLE_SECONDS)))
    s0 = (n_samples - m) // 2
    cand = dict(zip(filters, (len(x) for x in pmap(
        enc_job, [(str(dat), base, dtype, C, s0, m, f, level) for f in filters], min(workers, len(filters))))))
    return cand, m, rows


def compress_chunks(dat, base, dtype, C, n_samples, level, workers, filt, rows, out):
    h, dt = hashlib.sha256(), np.dtype(dtype)
    bounds = list(range(0, n_samples, rows)) + [n_samples]
    offs = [0]
    jobs = [(str(dat), base, dtype, C, bounds[i], bounds[i + 1] - bounds[i], filt, level)
            for i in range(len(bounds) - 1)]
    with open(out, "wb") as fo:
        for i, blob in enumerate(pmap(enc_job, jobs, workers)):
            fo.write(blob)
            offs.append(offs[-1] + len(blob))
            a = np.memmap(dat, dt, "r", base + bounds[i] * C * dt.itemsize, (bounds[i + 1] - bounds[i], C))
            h.update(a)
            del a
    return bounds, offs, h.hexdigest()


# ---- phy.tar.zst -----------------------------------------------------------

def write_phy_tar(dest, items, level, threads):
    prm = zstandard.ZstdCompressionParameters.from_level(
        level, threads=threads, enable_ldm=True, window_log=27, write_checksum=True)
    with open(dest, "wb") as fo, zstandard.ZstdCompressor(compression_params=prm).stream_writer(fo) as zw:
        with tarfile.open(fileobj=zw, mode="w|") as tf:
            for name, src in items:
                if isinstance(src, bytes):
                    ti = tarfile.TarInfo(name)
                    ti.size, ti.mtime = len(src), int(time.time())
                    tf.addfile(ti, io.BytesIO(src))
                else:
                    ti = tf.gettarinfo(str(src), name)
                    ti.uid = ti.gid = 0
                    ti.uname = ti.gname = ""
                    with open(src, "rb") as f:
                        tf.addfile(ti, f)


def read_phy_tar(path, expect, out_dir=None):
    """Stream the tar back, check each member's sha256 against `expect`; optionally extract."""
    seen = set()
    with open(path, "rb") as fh, tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(fh), mode="r|") as tf:
        for m in tf:
            if not m.isreg() or m.name != Path(m.name).name or m.name in ("", ".", ".."):
                die(f"unsafe or unexpected member in phy.tar.zst: {m.name!r}")
            m.name in expect or die(f"unexpected member {m.name!r} in phy.tar.zst")
            h, f = hashlib.sha256(), tf.extractfile(m)
            fo = open(Path(out_dir) / m.name, "wb") if out_dir else None
            while b := f.read(1 << 22):
                h.update(b)
                fo and fo.write(b)
            fo and fo.close()
            if h.hexdigest() != expect[m.name]:
                die(f"sha256 mismatch for {m.name}")
            seen.add(m.name)
    seen == set(expect) or die(f"phy.tar.zst is missing {sorted(set(expect) - seen)}")


# ---- pack / unpack ---------------------------------------------------------

def need_space(path, nbytes):
    free = shutil.disk_usage(path).free
    if free < nbytes:
        die(f"{path} has {free >> 20} MiB free, need about {nbytes >> 20} MiB")


def pack(a):
    t0 = time.time()
    phy, out = Path(a.phy_dir).resolve(), Path(a.out_dir)
    phy.is_dir() or die(f"{a.phy_dir} is not a directory")
    out.resolve().is_relative_to(phy) and die(f"OUT_DIR {out} is inside PHY_DIR; pick a folder outside it")
    miss = [n for n in REQUIRED if not (phy / n).is_file()]
    miss and die(f"missing required files: {', '.join(miss)}")
    name = a.name or phy.name
    re.fullmatch(r"[\w.-]+", name) or die(f"--name {name!r} must be letters, digits, '.', '_' or '-'")
    1 <= a.level <= 22 or die("--level must be 1..22")
    warns = []
    (phy / "cluster_group.tsv").is_file() or warn(
        "cluster_group.tsv is missing: it holds the human's good/mua/noise labels", warns)

    p = read_params(phy / "params.py")
    absent = [k for k in ("n_channels_dat", "dtype", "sample_rate") if k not in p]
    absent and die(f"params.py lacks {', '.join(absent)}")
    C, dtype, sr, base = int(p["n_channels_dat"]), str(p["dtype"]), float(p["sample_rate"]), int(p.get("offset", 0))
    dt = np.dtype(dtype)
    dt.kind in "iuf" or die(f"unsupported dtype {dtype}")
    dat = find_dat(p, phy, a.dat)
    meta = dat.with_suffix(".meta")
    body = dat.stat().st_size - base
    body > 0 and body % (C * dt.itemsize) == 0 or die(
        f"recording region ({body} bytes after offset {base}) is not a whole number of "
        f"{C}-channel {dtype} samples")
    n_samples = body // (C * dt.itemsize)

    chk = check(phy, n_samples, sr)
    warns += chk["warnings"]

    items = [(n, phy / n) for n in REQUIRED[1:] + OPTIONAL if (phy / n).is_file()]
    items.insert(0, ("params.py", scrubbed_params(p)))
    scrubbed = []
    if meta.is_file():
        blob, scrubbed = scrubbed_meta(meta)
        items.append(("recording.meta", blob))
    files = {}
    for n, s in items:
        files[n] = {"size": len(s) if isinstance(s, bytes) else s.stat().st_size,
                    "sha256": hashlib.sha256(s).hexdigest() if isinstance(s, bytes) else sha256_file(s)}
    allow = set(REQUIRED + OPTIONAL) | {dat.name, meta.name}
    excluded = sorted(e.name + ("/" if e.is_dir() else "") for e in phy.iterdir() if e.name not in allow)
    custom = [e for e in excluded if re.fullmatch(r"cluster_.*\.tsv", e) and e != "cluster_info.tsv"]
    small = sum(f["size"] for f in files.values())

    workers = n_workers(a.jobs, a.level, CHUNK)
    t1 = time.time()
    cand, m, rows = pick_filter(dat, base, dtype, C, n_samples, sr, a.level, workers)
    filt = min(cand, key=cand.get)
    ratio_est = min(1.0, 1.25 * cand[filt] / (m * C * dt.itemsize))
    # ponytail: peak = parts + outer tar = 2x the packaged size; streaming the parts straight
    # into the outer tar would halve it but needs the member sizes up front.
    out.mkdir(parents=True, exist_ok=True)
    need_space(out, int(2 * (ratio_est * body + small) + (64 << 20)))

    final = out / f"{name}.sorrel-upload.tar"
    work, part = out / f".{name}.tmp-{os.getpid()}", out / f".{name}.sorrel-upload.tar.part"
    work.mkdir()
    try:
        bounds, offs, sha = compress_chunks(dat, base, dtype, C, n_samples, a.level, workers, filt, rows,
                                            work / "raw.zst.bin")
        info = {"n_channels": C, "dtype": dtype, "sample_rate": sr, "n_samples": n_samples, "filter": filt,
                "level": a.level, "chunk_bounds": bounds, "chunk_offsets": offs, "sha256": sha}
        if decode_stream(work / "raw.zst.bin", info, workers) != sha:
            die("raw verification FAILED: decoded stream does not match the source; nothing delivered")
        (work / "raw.json").write_text(json.dumps(info))
        raw_secs = time.time() - t1
        write_phy_tar(work / "phy.tar.zst", items, a.level, workers)
        read_phy_tar(work / "phy.tar.zst", {n: f["sha256"] for n, f in files.items()})
        rz, tz = offs[-1], (work / "phy.tar.zst").stat().st_size
        man = {"tool": "export_curated_sort", "version": VERSION,
               "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "source_dir": phy.name, "files": files, "meta_scrubbed": scrubbed, "excluded": excluded, "custom_label_files": custom,
               "checks": {**chk, "warnings": warns, "raw_verified": True, "phy_tar_verified": True},
               "raw": {"filter": filt, "candidates": cand, "level": a.level, "workers": workers,
                       "seconds": round(raw_secs, 1), "n_chunks": len(bounds) - 1, "sha256": sha,
                       "original_bytes": body, "compressed_bytes": rz, "ratio": round(body / rz, 3)},
               "phy_tar": {"original_bytes": small, "compressed_bytes": tz, "ratio": round(small / tz, 3)},
               "original_bytes": body + small, "packaged_bytes": rz + tz,
               "ratio": round((body + small) / (rz + tz), 3)}
        (work / "MANIFEST.json").write_text(json.dumps(man, indent=1))
        with tarfile.open(part, "w") as tf:  # plain tar: the members are already compressed
            for n in ("MANIFEST.json", "phy.tar.zst", "raw.json", "raw.zst.bin"):
                ti = tf.gettarinfo(str(work / n), n)
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                with open(work / n, "rb") as f:
                    tf.addfile(ti, f)
        digest = sha256_file(part)
        (out / f".{name}.sha256.part").write_text(f"{digest}  {final.name}\n")
        os.replace(part, final)
        os.replace(out / f".{name}.sha256.part", out / (final.name + ".sha256"))
    except BaseException:
        part.unlink(missing_ok=True)
        (out / f".{name}.sha256.part").unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"packed {phy.name} -> {final}  ({final.stat().st_size / 1e6:.1f} MB, sha256 {digest[:12]}...)")
    print("included:")
    for n, f in files.items():
        print(f"  {n:28s} {f['size']:>14,d} B")
    print(f"  recording.bin                {body:>14,d} B  ({n_samples} samples x {C} ch {dtype})")
    print(f"excluded: {', '.join(excluded) or '-'}")
    print(f"recording.meta values blanked: {', '.join(scrubbed) or '-'}")
    print(f"custom label files (ask the user): {', '.join(custom) or '-'}")
    print(f"raw: filter {filt}, candidates {cand}, {body / rz:.2f}x, {workers} workers, {raw_secs:.0f} s")
    print(f"phy.tar.zst: {small:,d} -> {tz:,d} B ({small / tz:.2f}x)")
    print(f"total: {man['original_bytes']:,d} -> {man['packaged_bytes']:,d} B ({man['ratio']:.2f}x)")
    print("counts: " + ", ".join(f"{k}={chk[k]}" for k in COUNTS))
    print("check: PASSED (raw decoded + sha256 equal, phy.tar.zst member sha256s equal)")
    for w in warns:
        print(f"WARNING: {w}")


def unpack(a):
    pkg, out = Path(a.pkg), Path(a.out_dir)
    pkg.is_file() or die(f"{pkg} not found")
    side = pkg.with_name(pkg.name + ".sha256")
    if side.is_file():
        want = side.read_text().split()[0]
        sha256_file(pkg) == want or die("package sha256 does not match the .sha256 file")
    else:
        print("note: no .sha256 beside the package, outer hash not checked", file=sys.stderr)
    out.mkdir(parents=True, exist_ok=True)
    any(out.iterdir()) and die(f"{out} is not empty")
    # Everything is built and verified in a temp subdir and moved into OUT_DIR last, so a
    # failure leaves OUT_DIR empty and a retry works.
    tmp = out / ".unpack-tmp"
    try:
        tmp.mkdir()
        with tarfile.open(pkg) as tf:
            ms = tf.getmembers()
            names = {m.name for m in ms}
            names == {"MANIFEST.json", "phy.tar.zst", "raw.json", "raw.zst.bin"} and all(m.isreg() for m in ms) \
                or die(f"unexpected package members {sorted(names)}")
            tf.extractall(tmp, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
        man, info = (json.loads((tmp / n).read_text()) for n in ("MANIFEST.json", "raw.json"))
        need_space(out, man["original_bytes"] + (64 << 20))
        stage = tmp / "phy"
        stage.mkdir()
        read_phy_tar(tmp / "phy.tar.zst", {n: f["sha256"] for n, f in man["files"].items()}, stage)
        workers = n_workers(a.jobs, 1, CHUNK)
        with open(stage / "recording.bin", "wb") as fo:
            got = decode_stream(tmp / "raw.zst.bin", info, workers, fo)
        got == info["sha256"] == man["raw"]["sha256"] or die("recording sha256 mismatch: package is corrupt")
        chk = check(stage, info["n_samples"], info["sample_rate"])
        bad = [k for k in COUNTS if chk[k] != man["checks"][k]]
        bad and die(f"counts differ from the manifest: {bad}")
        for f in stage.iterdir():
            os.replace(f, out / f.name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"unpacked to {out}: recording.bin verified (sha256 {got[:12]}...)")
    print("counts: " + ", ".join(f"{k}={chk[k]}" for k in COUNTS))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("phy_dir")
    p.add_argument("out_dir")
    p.add_argument("--name")
    p.add_argument("--level", type=int, default=19)
    p.add_argument("--jobs", type=int)
    p.add_argument("--dat")
    u = sub.add_parser("unpack")
    u.add_argument("pkg")
    u.add_argument("out_dir")
    u.add_argument("--jobs", type=int)
    a = ap.parse_args()
    try:
        {"pack": pack, "unpack": unpack}[a.cmd](a)
    except (OSError, ValueError, zstandard.ZstdError, tarfile.TarError, KeyError) as e:
        die(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

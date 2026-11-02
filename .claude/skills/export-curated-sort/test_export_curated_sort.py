# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "zstandard"]
# ///
"""Self-check for export_curated_sort.py on a synthetic Kilosort-4-style phy dir.

    uv run --script .claude/skills/export-curated-sort/test_export_curated_sort.py
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

TOOL = Path(__file__).with_name("export_curated_sort.py")
ENV = {**os.environ, "SORREL_EXPORT_CHUNK_BYTES": str(256 << 10)}  # several chunks
ALLOWED = {"spike_times.npy", "spike_templates.npy", "spike_clusters.npy", "templates.npy",
           "channel_positions.npy", "channel_map.npy", "amplitudes.npy", "cluster_group.tsv",
           "pc_features.npy", "pc_feature_ind.npy", "similar_templates.npy", "whitening_mat_inv.npy",
           "cluster_KSLabel.tsv"}
DECOYS = {"phy.log", "ops.npy", "cluster_info.tsv", "cluster_notes.tsv", ".phy"}


def run(*args, ok=True):
    r = subprocess.run([sys.executable, str(TOOL), *map(str, args)], env=ENV, capture_output=True, text=True)
    assert (r.returncode == 0) == ok, (args, r.returncode, r.stdout, r.stderr)
    return r


def snapshot(d):
    return sorted((str(p.relative_to(d)), p.stat().st_size, p.stat().st_mtime_ns) for p in d.rglob("*"))


def make_phy(root, dtype, dat_in_phy, header=0):
    """Return (phy_dir, dat_path, data_bytes). 3 templates; t0+t1 merged into cluster 0,
    t2 split into clusters 2 and 3."""
    rng = np.random.default_rng(1)
    phy, n, C = root / "phy", 5 * 30000, 8
    phy.mkdir()
    walk = np.cumsum(rng.integers(-3, 4, (n, C)), axis=0)
    walk[rng.integers(0, n, 400), rng.integers(0, C, 400)] += 300  # spikes
    data = walk.astype(dtype)
    dat = (phy if dat_in_phy else root) / "rec.dat"
    dat.write_bytes(b"\x07" * header + data.tobytes())
    N = 3000
    st = np.sort(rng.integers(0, n, N)).astype(np.uint64)
    sp = rng.integers(0, 3, N).astype(np.uint32)
    sc = np.where(sp == 1, 0, sp).astype(np.uint32)
    sc[(sp == 2) & (rng.random(N) < 0.5)] = 3
    for name, arr in {
        "spike_times.npy": st, "spike_templates.npy": sp, "spike_clusters.npy": sc,
        "templates.npy": rng.normal(size=(3, 61, C)).astype(np.float32),
        "channel_positions.npy": rng.normal(size=(C, 2)), "channel_map.npy": np.arange(C, dtype=np.int32),
        "amplitudes.npy": rng.random(N), "pc_features.npy": rng.normal(size=(N, 3, 4)).astype(np.float32),
        "pc_feature_ind.npy": np.zeros((3, 4), np.uint32), "similar_templates.npy": np.eye(3, dtype=np.float32),
        "whitening_mat_inv.npy": np.eye(C), "ops.npy": np.array([1]),
    }.items():
        np.save(phy / name, arr)
    (phy / "params.py").write_text(
        f"dat_path = '/home/someuser/data/rec.dat'\nn_channels_dat = {C}\ndtype = '{np.dtype(dtype).name}'\n"
        f"offset = {header}\nsample_rate = 30000.0\nhp_filtered = False\ndir_path = '/home/someuser'\n")
    (phy / "cluster_group.tsv").write_text("cluster_id\tgroup\n0\tgood\n2\tmua\n")
    (phy / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n0\tgood\n")
    (phy / "cluster_info.tsv").write_text("id\n0\n")
    (phy / "cluster_notes.tsv").write_text("cluster_id\tnotes\n0\tsomeuser says nice\n")
    (phy / "phy.log").write_text("merge /home/someuser\n")
    (phy / ".phy").mkdir()
    (phy / ".phy" / "state.json").write_text("{}")
    (dat.with_suffix(".meta")).write_text("imDatPrb_sn=1234\nimSampRate=30000\nfileName=C:\\Users\\someuser\\rec.bin\r\nuserNotes=someuser note\n"
        "~imroTbl=(0,384)(0 0 0 500 250 1)\nimAiRangeMax=0.6\n")
    return phy, dat, data.tobytes()


def roundtrip(root, dtype, dat_in_phy, header, filters):
    base = root / f"{np.dtype(dtype).name}-{header}"
    base.mkdir()
    phy, dat, raw = make_phy(base, dtype, dat_in_phy, header)
    before, out = snapshot(base), root / f"out-{base.name}"
    if not dat_in_phy:  # bin elsewhere and params names a foreign path: needs --dat
        run("pack", phy, out, "--level", 3, ok=False)
    run("pack", phy, out, "--level", 3, "--jobs", 2, *([] if dat_in_phy else ["--dat", dat]))
    assert snapshot(base) == before, "source modified"
    assert sorted(p.name for p in out.iterdir()) == ["phy.sorrel-upload.tar", "phy.sorrel-upload.tar.sha256"]
    pkg = out / "phy.sorrel-upload.tar"
    man = json.loads(tarfile.open(pkg).extractfile("MANIFEST.json").read())
    assert man["raw"]["n_chunks"] > 1 and set(man["raw"]["candidates"]) == set(filters), man["raw"]
    assert man["checks"]["n_merged_clusters"] == 1 and man["checks"]["n_split_templates"] == 1, man["checks"]
    assert man["checks"]["n_clusters"] == 3 and man["checks"]["n_templates"] == 3
    assert man["custom_label_files"] == ["cluster_notes.tsv"], man["custom_label_files"]
    assert DECOYS <= {e.rstrip("/") for e in man["excluded"]}, man["excluded"]
    text = json.dumps(man)
    assert str(base) not in text and "someuser" not in text and man["source_dir"] == "phy"
    un = base / "un"
    run("unpack", pkg, un)
    assert sorted(p.name for p in un.iterdir()) == sorted(
        ALLOWED | {"params.py", "recording.bin", "recording.meta"}), sorted(p.name for p in un.iterdir())
    assert (un / "recording.bin").read_bytes() == raw and not DECOYS & {p.name for p in un.iterdir()}
    for n in ALLOWED:
        assert (un / n).read_bytes() == (phy / n).read_bytes(), n
    # The meta is scrubbed (path and notes blanked), every other key, the gain ones included, survives.
    want = dat.with_suffix(".meta").read_bytes().replace(b"C:\\Users\\someuser\\rec.bin", b"").replace(b"someuser note", b"")
    assert (un / "recording.meta").read_bytes() == want and b"someuser" not in want
    assert b"~imroTbl=(0,384)(0 0 0 500 250 1)" in want and b"imAiRangeMax=0.6" in want
    assert man["meta_scrubbed"] == ["fileName", "userNotes"], man["meta_scrubbed"]
    params = (un / "params.py").read_text()
    assert "/" not in params and "someuser" not in params and "dat_path = 'recording.bin'" in params
    return base, pkg, man


def main():
    tmp = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    tmp.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="export-curated-", dir=tmp))
    try:
        # int16 random walk, bin beside params.py (found by file name): diff must win.
        base, pkg, man = roundtrip(root, np.int16, True, 0, ["diff+shuffle", "diff", "shuffle", "none"])
        assert man["raw"]["filter"].startswith("diff") and man["raw"]["ratio"] > 1.5, man["raw"]

        # Corrupt one byte of raw.zst.bin inside a copy (no .sha256 beside it): unpack must fail.
        bad = base / "bad" / pkg.name
        bad.parent.mkdir()
        shutil.copy(pkg, bad)
        with tarfile.open(bad) as tf:
            m = tf.getmember("raw.zst.bin")
        with open(bad, "r+b") as f:
            f.seek(m.offset_data + m.size // 2)
            b = f.read(1)
            f.seek(-1, 1)
            f.write(bytes([b[0] ^ 0xFF]))
        r = run("unpack", bad, base / "bad-un", ok=False)
        assert "error" in r.stderr and not any((base / "bad-un").iterdir()), r.stderr  # nothing left behind
        shutil.copy(pkg, bad)  # a retry into the same OUT_DIR with a good package works
        run("unpack", bad, base / "bad-un")
        assert (base / "bad-un" / "recording.bin").is_file() and not (base / "bad-un" / ".unpack-tmp").exists()
        with open(bad, "r+b") as f:
            f.seek(m.offset_data + m.size // 2)
            f.write(bytes([b[0] ^ 0xFF]))
        # With the original .sha256 beside it the outer hash catches it first.
        shutil.copy(str(pkg) + ".sha256", str(bad) + ".sha256")
        assert "sha256" in run("unpack", bad, base / "bad-un2", ok=False).stderr

        # float32 with a 64-byte header and the bin elsewhere (--dat): shuffle/none only.
        _, _, man2 = roundtrip(root, np.float32, False, 64, ["shuffle", "none"])

        # OUT_DIR inside PHY_DIR is refused.
        phy = root / "int16-0" / "phy"
        assert "inside PHY_DIR" in run("pack", phy, phy / "out", ok=False).stderr and not (phy / "out").exists()

        # Sorts where Kilosort found no units are excluded, alone or in a batch, and leave no package:
        # empty spike arrays, and no spike arrays at all (both seen from Kilosort 4).
        empty = []
        for i, drop in enumerate((False, True)):
            e = root / f"rec{i}" / "kilosort4"
            shutil.copytree(phy, e)
            for f in ("spike_times.npy", "spike_templates.npy", "spike_clusters.npy", "amplitudes.npy"):
                (e / f).unlink() if drop else np.save(e / f, np.empty(0, np.uint32))
            empty.append(e)
        alone = run("pack", empty[0], root / "out-empty")
        assert "excluded" in alone.stdout and not (root / "out-empty").exists(), alone.stdout
        good = root / "rec2" / "kilosort4"
        shutil.copytree(phy, good)
        batch = run("pack", *empty, good, root / "out-batch", "--level", 3)
        assert "1 packaged, 2 excluded" in batch.stdout, batch.stdout
        assert sorted(p.name for p in (root / "out-batch").iterdir()) == [
            "rec2_kilosort4.sorrel-upload.tar", "rec2_kilosort4.sorrel-upload.tar.sha256"]
        # A broken folder in a batch fails on its own; the rest still packs, and the exit is non-zero.
        (good / "amplitudes.npy").rename(good / "amp.bak")
        r = run("pack", empty[0], good, root / "out-batch2", ok=False)
        assert "0 packaged, 1 excluded" in r.stdout and "1 failed" in r.stdout, r.stdout

        # A missing required file is an error.
        (phy / "amplitudes.npy").rename(phy / "amp.bak")
        assert "amplitudes.npy" in run("pack", phy, root / "out-missing", ok=False).stderr
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("ok")


if __name__ == "__main__":
    main()

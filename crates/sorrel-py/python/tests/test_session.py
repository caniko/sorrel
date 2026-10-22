"""In-process Session round trip on a synthetic phy directory.

Run after `maturin develop` from crates/sorrel-py:  pytest python/tests
No numpy needed: the fixture writes .npy headers by hand.
"""
import struct
from pathlib import Path

import pytest

import sorrel


def _write_npy(path: Path, descr: str, values, fmt: str) -> None:
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': ({len(values)},), }}".encode()
    pad = 16 - ((10 + len(header) + 1) % 16)
    header += b" " * pad + b"\n"
    path.write_bytes(
        b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
        + struct.pack("<" + fmt * len(values), *values)
    )


@pytest.fixture
def phy_dir(tmp_path: Path) -> Path:
    # 6 spikes, 3 clusters: times sorted, clusters [0,0,1,2,0,1]
    _write_npy(tmp_path / "spike_times.npy", "<u8", [10, 30, 50, 100, 150, 200], "Q")
    _write_npy(tmp_path / "spike_clusters.npy", "<u4", [0, 0, 1, 2, 0, 1], "I")
    (tmp_path / "recording.dat").write_bytes(b"\0" * (2 * 2 * 300))
    (tmp_path / "params.py").write_text("n_channels_dat = 2\nsample_rate = 1000.\ndtype = 'int16'\n")
    return tmp_path


def test_edit_save_reopen(phy_dir: Path) -> None:
    s = sorrel.Session.open(phy_dir)
    assert s.n_clusters == 3
    assert s.labels() == ["unsorted"] * 3

    v0 = s.version
    s.set_label(0, "good")
    assert s.version != v0
    s.merge([1], 2)
    new = s.split(0, [1])          # spike t=30 → new cluster
    assert new == 3
    assert s.spike_times(3) == [30]

    rows = s.qc()
    assert {r["cluster_id"] for r in rows} == {0, 1, 2, 3}
    assert rows[0]["n_spikes"] == 2
    assert rows[0]["l_ratio"] is None          # NaN → None

    doc = s.curation()
    assert doc["format_version"] == "2"
    assert doc["merges"][0]["unit_ids"] == [1, 2]
    assert doc["splits"][0]["new_unit_ids"] == [3]

    out = s.save()
    assert (Path(out) / "cluster_group.tsv").read_text() == "cluster_id\tgroup\n0\tgood\n"
    assert sorrel.read_curation(out)["merges"] == []   # resealed → labels only

    # Reopen: the saved state loads once, with an empty undo stack.
    s2 = sorrel.Session.open(phy_dir)
    assert s2.n_clusters == 4
    assert s2.labels()[0] == "good"
    assert s2.spike_times(2) == [50, 100, 200]
    with pytest.raises(ValueError):
        s2.set_label(0, "excellent")

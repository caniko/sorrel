"""Curated tiny-phy session -> NWB Units table, directly and through NeuroConv.

Expected values come from the fixture's .npy files and the edits below,
never from sorrel. Run with the test dependencies (see crates/sorrel-py/README.md).
"""
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from nwbinspector import Importance, inspect_nwbfile, load_config
from pynwb import NWBHDF5IO, NWBFile, validate
from pynwb.file import Subject

import sorrel
from sorrel.neuroconv import SorrelSortingInterface
from sorrel.nwb import add_units_to_nwbfile

FIXTURE = Path(__file__).resolve().parents[4] / "release" / "fixtures" / "tiny-phy"
RATE = 30000.1234  # not a float32: catches single-precision timestamps
START = 12.5
MOVED = [0, 2, 4, 6]  # positions in unit 0's time-sorted train, split off
SUBJECT = dict(subject_id="m1", species="Mus musculus", sex="U", age="P90D")


@pytest.fixture
def curated(tmp_path: Path) -> Path:
    d = tmp_path / "phy"
    shutil.copytree(FIXTURE, d)
    params = d / "params.py"
    params.write_text(params.read_text().replace("sample_rate = 30000.", f"sample_rate = {RATE}"))
    s = sorrel.Session.open(d)
    s.set_label(0, "good")
    s.set_label(1, "mua")
    s.merge([1], 2)
    s.set_label(2, "noise")
    assert s.split(0, MOVED, by="manual") == 3
    s.save()
    return d


def expected_trains() -> dict:
    times = np.load(FIXTURE / "spike_times.npy").ravel()
    clusters = np.load(FIXTURE / "spike_clusters.npy").ravel()
    t0 = np.sort(times[clusters == 0])
    return {
        0: np.delete(t0, MOVED),
        2: np.sort(times[(clusters == 1) | (clusters == 2)]),
        3: t0[MOVED],
    }


def write_direct(phy: Path, out: Path) -> None:
    nwb = NWBFile("tiny", "tiny-direct", datetime(2026, 1, 1, tzinfo=timezone.utc),
                  subject=Subject(**SUBJECT))
    add_units_to_nwbfile(phy, nwb, starting_time=START)
    with NWBHDF5IO(out, "w") as io:
        io.write(nwb)


def write_neuroconv(phy: Path, out: Path) -> None:
    interface = SorrelSortingInterface(folder_path=phy)
    interface.set_aligned_starting_time(START)
    metadata = interface.get_metadata()
    metadata["NWBFile"].update(session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                               session_description="tiny")
    metadata["Subject"] = SUBJECT
    interface.run_conversion(nwbfile_path=out, metadata=metadata, overwrite=True)


@pytest.mark.parametrize("write", [write_direct, write_neuroconv])
def test_units_round_trip(curated: Path, tmp_path: Path, write) -> None:
    out = tmp_path / "tiny.nwb"
    write(curated, out)

    with NWBHDF5IO(out, "r") as io:
        units = io.read().units
        df = units.to_dataframe()
        resolution = units.resolution
        description = units.description
        generated = [tuple(x) for x in io.read().was_generated_by]

    assert resolution == 1.0 / RATE
    want = expected_trains()
    assert sorted(df.index) == sorted(want)
    for uid, samples in want.items():
        np.testing.assert_allclose(df.loc[uid, "spike_times"], samples / RATE + START,
                                   rtol=0, atol=1e-12)
    # A split-off unit starts unsorted: it is a new unit nobody has judged.
    assert df["quality"].to_dict() == {0: "good", 2: "noise", 3: "unsorted"}
    assert df.loc[3, "origin"] == "Sorrel[Manual]"
    assert df.loc[0, "origin"].startswith("Auto[")
    assert {u: list(p) for u, p in df["parent_unit_ids"].items()} == {0: [], 2: [1], 3: [0]}
    assert df.loc[2, "parent_unit_ids"].dtype.kind == "i"
    assert df.loc[2, "n_spikes"] == len(want[2])
    assert "by actor: human=" in description and "journal_xxh3=" in description
    assert ("sorrel-ephys", sorrel.__version__) in generated

    errors = validate(path=out)
    assert not errors, errors
    messages = list(inspect_nwbfile(nwbfile_path=out, config=load_config("dandi")))
    serious = [m for m in messages
               if m.importance in (Importance.CRITICAL, Importance.BEST_PRACTICE_VIOLATION)]
    assert not serious, serious
    for m in messages:  # suggestions are allowed; show them with -s
        print(m.importance.name, m.check_function_name, m.location, m.message)


def test_label_only_session(tmp_path: Path) -> None:
    # No merge or split: every lineage row is empty, which hdmf cannot type.
    d = tmp_path / "phy"
    shutil.copytree(FIXTURE, d)
    s = sorrel.Session.open(d)
    s.set_label(0, "good")
    s.save()
    del s
    out = tmp_path / "labels.nwb"
    write_direct(d, out)
    with NWBHDF5IO(out, "r") as io:
        df = io.read().units.to_dataframe()
    assert sorted(df.index) == [0, 1, 2]
    assert df.loc[0, "quality"] == "good"
    assert all(len(p) == 0 for p in df["parent_unit_ids"])


def test_refuses_existing_units(curated: Path) -> None:
    nwb = NWBFile("x", "x", datetime(2026, 1, 1, tzinfo=timezone.utc))
    nwb.add_unit(spike_times=[0.1])
    with pytest.raises(ValueError, match="already has a units table"):
        add_units_to_nwbfile(curated, nwb)


def test_rejects_multi_segment(tmp_path: Path) -> None:
    import spikeinterface.core as si
    import spikeinterface.extractors as se

    warnings.simplefilter("ignore")
    rec, sort = si.generate_ground_truth_recording(durations=[1.0, 1.0], num_units=3, seed=0)
    files = [tmp_path / "seg0.raw", tmp_path / "seg1.raw"]
    si.write_binary_recording(rec, file_paths=files, dtype="int16")
    raw = se.read_binary(files, sampling_frequency=rec.sampling_frequency,
                         num_channels=rec.get_num_channels(), dtype="int16",
                         channel_ids=rec.channel_ids)
    raw.set_probe(rec.get_probe(), in_place=True)
    si.create_sorting_analyzer(sort, raw, format="binary_folder", folder=tmp_path / "sa",
                               sparse=False)
    nwb = NWBFile("x", "x", datetime(2026, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="multi-segment"):
        add_units_to_nwbfile(tmp_path / "sa", nwb)

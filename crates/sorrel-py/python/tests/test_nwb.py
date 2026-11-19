"""Curated tiny-phy session -> NWB Units table, directly and through NeuroConv.

Expected values come from the fixture's .npy files and the edits below,
never from sorrel. Run with the test dependencies (see crates/sorrel-py/README.md).
"""
import json
import os
import shutil
import subprocess
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest
from nwbinspector import Importance, inspect_nwbfile, load_config
from pynwb import NWBHDF5IO, NWBFile, validate
from pynwb.ecephys import ElectricalSeries
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


def whitematter_interface(curated: Path):
    # A real recorder reader (SpikeInterface's WhiteMatter: an 8-byte header,
    # then int16 samples) over the session's own raw file.
    from neuroconv.datainterfaces import WhiteMatterRecordingInterface

    bin_path = curated / "recording.bin"
    bin_path.write_bytes(bytes(8) + (curated / "recording.dat").read_bytes())
    params = curated / "params.py"
    params.write_text(params.read_text()
                      .replace("'recording.dat'", "'recording.bin'").replace("offset = 0", "offset = 8"))
    return WhiteMatterRecordingInterface(file_path=bin_path, sampling_frequency=RATE, num_channels=4)


@pytest.mark.parametrize("recorder", ["mock", "whitematter"])
def test_converter_pipe_with_recording(curated: Path, tmp_path: Path, recorder: str) -> None:
    # The usual NeuroConv shape: a recording interface next to the sorting.
    from neuroconv import ConverterPipe
    from neuroconv.tools.testing.mock_interfaces import MockRecordingInterface

    if recorder == "mock":
        rec = MockRecordingInterface(num_channels=4, sampling_frequency=RATE, durations=(1.0,))
    else:
        rec = whitematter_interface(curated)
    # NeuroConv writes "unknown" otherwise, which is not a CCF term.
    rec.recording_extractor.set_property("brain_area", ["CA1"] * 4)
    sorting = SorrelSortingInterface(folder_path=curated)
    sorting.set_aligned_starting_time(START)
    pipe = ConverterPipe(data_interfaces=dict(Recording=rec, Sorting=sorting))
    metadata = pipe.get_metadata()
    metadata["NWBFile"].update(session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                               session_description="tiny")
    metadata["Subject"] = SUBJECT
    out = tmp_path / "pipe.nwb"
    pipe.run_conversion(nwbfile_path=out, metadata=metadata, overwrite=True)

    with NWBHDF5IO(out, "r") as io:
        nwb = io.read()
        assert "ElectricalSeries" in nwb.acquisition
        assert sorted(nwb.units.id[:]) == sorted(expected_trains())
        if recorder == "whitematter":
            raw = np.fromfile(curated / "recording.dat", dtype=np.int16).reshape(-1, 4)
            np.testing.assert_array_equal(nwb.acquisition["ElectricalSeries"].data[:], raw)
    assert not validate(path=out)
    serious = [m for m in inspect_nwbfile(nwbfile_path=out, config=load_config("dandi"))
               if m.importance in (Importance.CRITICAL, Importance.BEST_PRACTICE_VIOLATION)]
    assert not serious, serious


def test_keeps_callers_was_generated_by(curated: Path) -> None:
    # pynwb accepts a tuple here and the attribute is write-once.
    nwb = NWBFile("x", "x", datetime(2026, 1, 1, tzinfo=timezone.utc),
                  was_generated_by=(("acq", "1.0"),))
    add_units_to_nwbfile(curated, nwb)
    assert [tuple(e) for e in nwb.was_generated_by] == [("acq", "1.0"),
                                                        ("sorrel-ephys", sorrel.__version__)]


@pytest.fixture
def native_cli():
    binary = os.environ.get("SORREL_NWB_BIN")
    if binary is None:
        pytest.skip("set SORREL_NWB_BIN to the sorrel binary built with --features hdf5")
    return binary


@pytest.mark.parametrize("append", [False, True], ids=["new", "append"])
def test_native_units_match_python(curated: Path, tmp_path: Path, native_cli, append: bool):
    native = tmp_path / "native.nwb"
    python = tmp_path / "python.nwb"
    write_direct(curated, python)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    if append:
        nwb = NWBFile("existing", "existing", start, subject=Subject(**SUBJECT),
                      was_generated_by=[["acquisition", "1.0"]])
        with NWBHDF5IO(native, "w") as io:
            io.write(nwb)
        args = ["--into", native]
    else:
        args = ["--out", native, "--session-start-time", start.isoformat(),
                "--identifier", "native", "--session-description", "tiny"]
    cmd = [native_cli, "export-nwb", curated, *args, "--starting-time", str(START)]
    subprocess.run(cmd, check=True)
    assert not validate(path=native)
    with h5py.File(native, "r") as f, h5py.File(python, "r") as py:
        assert set(f["units/spike_times"].attrs) == set(py["units/spike_times"].attrs)
        for name in ("id", "spike_times", "spike_times_index"):
            assert isinstance(f["units"].get(name, getlink=True), h5py.HardLink)
            dataset = f[f"units/{name}"]
            assert not dataset.is_virtual and not dataset.external
    with NWBHDF5IO(native, "r") as actual, NWBHDF5IO(python, "r") as expected:
        dest, src = actual.read(), expected.read()
        got, want = dest.units, src.units
        assert got.colnames == want.colnames
        assert got.resolution == want.resolution == 1 / RATE
        assert got.description == want.description
        np.testing.assert_array_equal(got.id[:], want.id[:])
        for name in want.colnames:
            for row in range(len(want)):
                np.testing.assert_array_equal(got[name][row], want[name][row])
        assert [tuple(x) for x in dest.was_generated_by] == (
            ([("acquisition", "1.0")] if append else []) + [tuple(x) for x in src.was_generated_by]
        )
        if not append:
            assert dest.identifier == "native"
            assert dest.session_start_time == start
            assert len(dest.electrodes) == 4
            assert all(g.device.name == "probe" for g in dest.electrodes["group"][:])
            assert not dest.acquisition

    messages = list(inspect_nwbfile(nwbfile_path=native, config=load_config("dandi")))
    for m in messages:
        print(m.importance.name, m.check_function_name, m.location, m.message)
    # Standalone sort exports deliberately don't invent subject metadata.
    assert not [m for m in messages if m.importance in (Importance.CRITICAL,
                                                       Importance.BEST_PRACTICE_VIOLATION)
                and (append or m.check_function_name != "check_subject_exists")]
    before = native.read_bytes()
    result = subprocess.run(cmd, text=True, capture_output=True)
    assert result.returncode != 0
    assert ("already has /units" if append else "already exists") in result.stderr
    assert native.read_bytes() == before


def test_native_export_from_nwb(tmp_path: Path, native_cli):
    source = tmp_path / "input.nwb"
    shutil.copyfile(FIXTURE.parent / "nwb" / "good.nwb", source)
    before = source.read_bytes()
    out = tmp_path / "sort.nwb"
    subprocess.run([native_cli, "export-nwb", source, "--out", out], check=True)
    assert source.read_bytes() == before
    assert not validate(path=out)
    with NWBHDF5IO(source, "r") as original, NWBHDF5IO(out, "r") as exported:
        src, dest = original.read(), exported.read()
        assert dest.identifier == src.identifier
        assert dest.session_description == src.session_description
        assert dest.session_start_time == src.session_start_time
        assert dest.timestamps_reference_time == src.timestamps_reference_time
        assert dest.units.resolution == 0.001
        np.testing.assert_array_equal(dest.units.id[:], [5, 9])
        np.testing.assert_allclose(dest.units["spike_times"][0], src.units["spike_times"][0])
        np.testing.assert_allclose(dest.units["spike_times"][1], src.units["spike_times"][2])
        np.testing.assert_array_equal(dest.electrodes.id[:], [13, 10, 15, 11])


def test_native_export_units_only_with_explicit_rate(tmp_path: Path, native_cli):
    source = tmp_path / "input.nwb"
    shutil.copyfile(FIXTURE.parent / "nwb" / "curated_units.nwb", source)
    with h5py.File(source, "r+") as f:
        del f["units/spike_times"].attrs["resolution"]
    out = tmp_path / "sort.nwb"
    rate = 1234.5
    subprocess.run([native_cli, "export-nwb", source, "--out", out,
                    "--sample-rate", str(rate)], check=True)
    assert not validate(path=out)
    with NWBHDF5IO(out, "r") as io:
        nwb = io.read()
        assert nwb.electrodes is None
        assert nwb.units.resolution == 1 / rate
        assert nwb.units["spike_times"][0][0] == round(0.55 * rate) / rate


@pytest.mark.parametrize("compatible", [True, False], ids=["float32", "mismatch"])
def test_native_append_preserves_declared_clock(tmp_path: Path, native_cli, compatible):
    source = tmp_path / "input.nwb"
    dest = tmp_path / "recording.nwb"
    shutil.copyfile(FIXTURE.parent / "nwb" / "precise_rate_long.nwb", source)
    shutil.copyfile(source, dest)
    with h5py.File(source, "r") as f:
        rate = 1 / float(f["units/spike_times"].attrs["resolution"])
    with h5py.File(dest, "r+") as f:
        del f["units"]
        start = f["acquisition/ElectricalSeriesRaw/starting_time"]
        assert start.attrs["rate"].dtype == np.dtype("float32")
        if not compatible:
            start.attrs.modify("rate", np.float32(30001))
    before = dest.read_bytes()
    result = subprocess.run([native_cli, "export-nwb", source, "--into", dest,
                             "--sample-rate", str(rate)], text=True, capture_output=True)
    if not compatible:
        assert result.returncode != 0 and "NWB rate mismatch" in result.stderr
        assert dest.read_bytes() == before
        return

    assert result.returncode == 0, result.stderr
    assert not validate(path=dest)
    with h5py.File(dest, "r") as f:
        spikes = f["units/spike_times"]
        assert spikes.dtype == spikes.attrs["resolution"].dtype == np.dtype("float64")
        assert spikes[0] == 330_000_000 / rate
        declared = 1 / float(spikes.attrs["resolution"])
    reopened = tmp_path / "reopened.nwb"
    cmd = [native_cli, "export-nwb", dest, "--out", reopened]
    result = subprocess.run(cmd, text=True, capture_output=True)
    assert result.returncode != 0 and "Ambiguous NWB sample rate" in result.stderr
    subprocess.run([*cmd, "--sample-rate", str(declared)], check=True)
    assert not validate(path=reopened)
    with h5py.File(reopened, "r") as f:
        assert f["units/spike_times"][0] == 330_000_000 / rate


def test_native_export_destinations(native_cli):
    help_text = subprocess.check_output([native_cli, "export-nwb", "--help"], text=True)
    assert "--out" in help_text and "--into" in help_text
    assert "--in-place" not in help_text


def test_native_append_checks_timestamp_reference(tmp_path: Path, native_cli):
    source = tmp_path / "source.nwb"
    shutil.copyfile(FIXTURE.parent / "nwb" / "good.nwb", source)
    with NWBHDF5IO(source, "r") as io:
        original = io.read()
        ref = original.timestamps_reference_time
        first_spike = original.units["spike_times"][0][0]
    out = tmp_path / "recording.nwb"
    dest_ref = ref - timedelta(seconds=10)
    with NWBHDF5IO(out, "w") as io:
        io.write(NWBFile("x", "x", ref, timestamps_reference_time=dest_ref))
    before = out.read_bytes()
    cmd = [native_cli, "export-nwb", source, "--into", out]
    result = subprocess.run(cmd, text=True, capture_output=True)
    assert result.returncode != 0
    assert "timestamps_reference_time" in result.stderr and "--starting-time" in result.stderr
    assert ref.date().isoformat() in result.stderr and dest_ref.date().isoformat() in result.stderr
    assert out.read_bytes() == before
    subprocess.run([*cmd, "--starting-time", "10.5"], check=True)
    assert not validate(path=out)
    with NWBHDF5IO(out, "r") as io:
        assert io.read().units["spike_times"][0][0] == pytest.approx(first_spike + 10)


@pytest.mark.parametrize("start", [0.25, -0.25])
def test_native_export_uses_selected_series_time_base(tmp_path: Path, native_cli, start):
    source, out = (tmp_path / name for name in ("source.nwb", "sort.nwb"))
    nwb = NWBFile("x", "x", datetime(2026, 1, 1, tzinfo=timezone.utc))
    device = nwb.create_device("probe")
    group = nwb.create_electrode_group("g", "test", "CA1", device)
    for i in range(2):
        nwb.add_electrode(id=i, group=group, location="CA1", rel_x=0.0, rel_y=float(i))
    for i in range(2):
        electrodes = nwb.create_electrode_table_region([0, 1], "test")
        nwb.add_acquisition(ElectricalSeries(name=f"series{i}", data=np.zeros((300, 2), np.int16),
                                             electrodes=electrodes, starting_time=start + i / 2,
                                              rate=RATE))
    first_spike = start + 25 / RATE
    nwb.add_unit(id=0, spike_times=[first_spike])
    with NWBHDF5IO(source, "w") as io:
        io.write(nwb)
    subprocess.run([native_cli, "export-nwb", source, "--out", out], check=True)
    assert not validate(path=out)
    with NWBHDF5IO(out, "r") as io:
        units = io.read().units
        assert units.resolution == 1 / RATE
        assert units["spike_times"][0][0] == first_spike
    if start < 0:
        reopened = tmp_path / "reopened.nwb"
        cmd = [native_cli, "export-nwb", out, "--out", reopened]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode != 0 and "--dat" in result.stderr
        subprocess.run([*cmd, "--dat", source], check=True)
        assert not validate(path=reopened)
        with NWBHDF5IO(reopened, "r") as io:
            assert io.read().units["spike_times"][0][0] == first_spike


def test_sorting_analyzer_sample_rate_parity(tmp_path: Path):
    rate = 30000.014396912702  # serde_json without float_roundtrip parses one ULP high
    sorting = tmp_path / "sorting"
    sorting.mkdir()
    np.save(sorting / "spikes.npy", np.array([(100, 0, 0), (200, 1, 0)], dtype=[
        ("sample_index", "i8"), ("unit_index", "i8"), ("segment_index", "i8")]))
    (sorting / "numpysorting_info.json").write_text(json.dumps(dict(unit_ids=["007", "unit_b"],
                                                                  num_segments=1)))
    (tmp_path / "data.raw").write_bytes(bytes(600))
    (tmp_path / "recording.json").write_text(json.dumps(dict(kwargs=dict(
        sampling_frequency=rate, num_channels=1, dtype="int16", file_paths=["data.raw"]))))
    nwb = NWBFile("x", "x", datetime(2026, 1, 1, tzinfo=timezone.utc))
    units = add_units_to_nwbfile(tmp_path, nwb)
    assert list(units["unit_name"][:]) == ["007", "unit_b"]
    assert units.resolution == 1 / rate


@pytest.fixture
def empty_nwb(tmp_path: Path):
    out = tmp_path / "recording.nwb"
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with NWBHDF5IO(out, "w") as io:
        io.write(NWBFile("x", "x", now, file_create_date=[now, now + timedelta(days=1)]))
    return out


def test_native_sample_rate_override_is_exact(curated: Path, empty_nwb: Path, native_cli):
    rate = 31000.1234
    subprocess.run([native_cli, "export-nwb", curated, "--into", empty_nwb,
                    "--sample-rate", str(rate)], check=True)
    assert not validate(path=empty_nwb)
    with NWBHDF5IO(empty_nwb, "r") as io:
        assert io.read().units.resolution == 1 / rate


def test_native_append_records_modification_date(curated: Path, empty_nwb: Path, native_cli):
    with NWBHDF5IO(empty_nwb, "r") as io:
        before = list(io.read().file_create_date)
    earliest = datetime.now(timezone.utc) - timedelta(seconds=1)
    subprocess.run([native_cli, "export-nwb", curated, "--into", empty_nwb], check=True)
    assert not validate(path=empty_nwb)
    with NWBHDF5IO(empty_nwb, "r") as io:
        dates = io.read().file_create_date
        assert dates[:-1] == before
        assert earliest <= dates[-1] <= datetime.now(timezone.utc)


def test_native_append_accepts_fixed_length_root_attributes(curated: Path, empty_nwb: Path,
                                                           native_cli):
    with h5py.File(empty_nwb, "r+") as f:
        for name in ("nwb_version", "neurodata_type"):
            value = np.bytes_(f.attrs[name])
            del f.attrs[name]
            f.attrs[name] = value
    subprocess.run([native_cli, "export-nwb", curated, "--into", empty_nwb], check=True)
    assert not validate(path=empty_nwb)
    with NWBHDF5IO(empty_nwb, "r") as io:
        assert sorted(io.read().units.id[:]) == sorted(expected_trains())


def test_native_export_rejects_journal_aliases(tmp_path: Path, empty_nwb: Path, native_cli):
    source = tmp_path / "source.nwb"
    shutil.copyfile(FIXTURE.parent / "nwb" / "good.nwb", source)
    out = tmp_path / "sort.nwb"
    hardlink = tmp_path / "hardlink.journal"
    hardlink.hardlink_to(empty_nwb)
    symlink = tmp_path / "symlink.journal"
    symlink.symlink_to(source)
    dangling = tmp_path / "dangling.journal"
    dangling.symlink_to(out)
    before = source.read_bytes(), empty_nwb.read_bytes()
    for target, journal in [(["--out", out], source), (["--out", out], symlink),
                            (["--into", empty_nwb], hardlink), (["--out", out], out),
                            (["--out", out], dangling)]:
        result = subprocess.run([native_cli, "export-nwb", source, *target,
                                 "--journal", journal, "--discard-journal"],
                                text=True, capture_output=True)
        assert result.returncode != 0 and "--journal" in result.stderr and "aliases" in result.stderr
        assert (source.read_bytes(), empty_nwb.read_bytes()) == before
        assert not out.exists()
        assert not journal.with_name(journal.name + ".lock").exists()


def test_native_append_refuses_readonly_file(curated: Path, empty_nwb: Path, native_cli):
    empty_nwb.chmod(0o444)
    if os.access(empty_nwb, os.W_OK):
        pytest.skip("requires a user without write access to a read-only file")
    before = empty_nwb.read_bytes()
    result = subprocess.run([native_cli, "export-nwb", curated, "--into", empty_nwb],
                            text=True, capture_output=True)
    assert result.returncode != 0 and "for writing" in result.stderr
    assert empty_nwb.read_bytes() == before


@pytest.mark.parametrize("encoding,length,label", [("ascii", 32, "acquisition"),
                                                   ("utf-8", 48, "acquisición"),
                                                   ("ascii", None, "acquisition"),
                                                   ("utf-8", None, "acquisición")])
def test_native_append_reads_stored_provenance_strings(curated: Path, empty_nwb: Path, native_cli,
                                                       encoding, length, label):
    with h5py.File(empty_nwb, "r+") as f:
        f["general"].create_dataset("was_generated_by", data=[[label.encode(encoding), b"1.0"]],
                                    dtype=h5py.string_dtype(encoding, length))
    subprocess.run([native_cli, "export-nwb", curated, "--into", empty_nwb], check=True)
    assert not validate(path=empty_nwb)
    with NWBHDF5IO(empty_nwb, "r") as io:
        assert [tuple(row) for row in io.read().was_generated_by] == [
            (label, "1.0"), ("sorrel-ephys", sorrel.__version__)]

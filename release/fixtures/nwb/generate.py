#!/usr/bin/env -S uv run --python-preference only-managed
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "pynwb==4.2.0",
#     "hdmf==6.2.0",
#     "h5py==3.16.0",
#     "numpy==2.5.4",
# ]
# ///
"""Tiny deterministic NWB fixtures for sorrel's reader tests.

Pinned to PyNWB 4.2.0 (hdmf 6.2.0, h5py 3.16.0, numpy 2.5.4).
Regenerate exactly with:

    nix shell nixpkgs#uv -c uv run --python-preference only-managed release/fixtures/nwb/generate.py

Files are written next to this script. Contents are deterministic for a fixed
toolchain (identifiers, session_start_time and file_create_date are fixed).
Byte-identity is not guaranteed: PyNWB assigns a random object_id UUID to every
container, so re-running changes object_ids but not the h5py-visible content.
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import VectorData, VectorIndex
from pynwb import NWBFile, NWBHDF5IO, get_class, load_namespaces
from pynwb.base import TimeSeries
from pynwb.ecephys import ElectricalSeries, FilteredEphys
from pynwb.misc import Units
from pynwb.spec import NWBGroupSpec, NWBNamespaceBuilder

OUT = Path(__file__).resolve().parent
SESSION_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
FILE_CREATE = datetime(2026, 1, 2, tzinfo=timezone.utc)
N_SAMPLES = 300
N_CHANNELS = 4
CHANNEL_REGION = [3, 0, 5, 1]  # permuted subset of the 6 electrodes, in this order
REL_X = [0.0, 0.0, 16.0, 16.0, 32.0, 32.0]
REL_Y = [0.0, 20.0, 0.0, 20.0, 0.0, 20.0]
UNITS = [(5, [0.6, 0.7, 0.75]), (7, []), (9, [0.55, 0.79])]


def make_file(identifier, *, units=True, electrodes=True, resolution=None):
    """NWBFile with optional electrodes and units."""
    nwbfile = NWBFile(
        session_description="sorrel NWB reader fixture",
        identifier=identifier,
        session_start_time=SESSION_START,
        file_create_date=[FILE_CREATE],
        session_id="sorrel-fixture",
    )
    if units:
        if resolution is not None:
            nwbfile.units = Units(name="units", resolution=resolution)
        for unit_id, spike_times in UNITS:
            nwbfile.add_unit(id=unit_id, spike_times=spike_times)
    if not electrodes:
        return nwbfile
    device = nwbfile.create_device(name="Device")
    group = nwbfile.create_electrode_group(
        name="ElectrodeGroup",
        description="CA1 electrodes",
        device=device,
        location="CA1",
    )
    for i in range(6):
        nwbfile.add_electrode(
            id=10 + i,
            x=float(100 + i),
            y=float(200 + i),
            z=float(300 + i),
            imp=float("nan"),
            location="CA1",
            filtering="none",
            group=group,
            rel_x=REL_X[i],
            rel_y=REL_Y[i],
        )
    return nwbfile


def make_series(nwbfile, data, name="ElectricalSeriesRaw", **kwargs):
    region = nwbfile.create_electrode_table_region(CHANNEL_REGION, "permuted subset")
    return ElectricalSeries(name=name, data=data, electrodes=region, **kwargs)


def add_series(nwbfile, data, **kwargs):
    nwbfile.add_acquisition(make_series(nwbfile, data, **kwargs))


def good_data():
    t = np.arange(N_SAMPLES)[:, None]
    c = np.arange(N_CHANNELS)[None, :]
    return ((t % 50) * (c + 1) - 100).astype(np.int16)


def write(nwbfile, name, cache_spec=False):
    path = OUT / name
    with NWBHDF5IO(path, "w") as io:
        io.write(nwbfile, cache_spec=cache_spec)
    print(f"wrote {path.name}: {path.stat().st_size} bytes")


def write_good():
    nwbfile = make_file("sorrel-fixture-good")
    add_series(
        nwbfile,
        H5DataIO(good_data(), chunks=(32, N_CHANNELS), compression="gzip"),
        rate=1000.0,
        starting_time=0.5,
        conversion=0.195e-6,
        offset=-2e-6,
        channel_conversion=[1.0, 2.0, 1.0, 0.5],
    )
    write(nwbfile, "good.nwb")


def write_uint16():
    nwbfile = make_file("sorrel-fixture-uint16")
    data = 40000 + np.arange(N_SAMPLES)[:, None] + np.arange(N_CHANNELS)[None, :]
    add_series(
        nwbfile,
        data.astype(np.uint16),
        rate=1000.0,
        starting_time=0.5,
        conversion=1e-6,
        offset=0.0,
    )
    write(nwbfile, "uint16.nwb")


def write_dtype(dtype, value):
    nwbfile = make_file(f"sorrel-fixture-{dtype}")
    add_series(
        nwbfile,
        np.full((1, N_CHANNELS), value, dtype=dtype),
        rate=1000.0,
        starting_time=0.5,
        conversion=1e44 if dtype == "float64" else 1e-6,
    )
    write(nwbfile, f"{dtype}.nwb")


def write_timestamps(irregular=False):
    name = "irregular_timestamps" if irregular else "timestamps"
    nwbfile = make_file(f"sorrel-fixture-{name}")
    timestamps = 0.5 + np.arange(N_SAMPLES) / 1000.0
    if irregular:
        timestamps[197] += 3e-4
    add_series(
        nwbfile,
        good_data(),
        timestamps=timestamps,
        conversion=0.195e-6,
        offset=-2e-6,
        channel_conversion=[1.0, 2.0, 1.0, 0.5],
    )
    write(nwbfile, f"{name}.nwb")


def write_multiple_series():
    nwbfile = make_file("sorrel-fixture-multiple-series", resolution=0.001)
    for name, value in [("Z_second", 2), ("A_first", 1)]:
        add_series(nwbfile, np.full((N_SAMPLES, N_CHANNELS), value, dtype=np.int16),
                   name=name, rate=1000.0, starting_time=0.5, conversion=1e-6)
    processed = make_series(
        nwbfile, np.full((N_SAMPLES, N_CHANNELS), 3, dtype=np.int16),
        name="Processed", rate=1000.0, starting_time=0.5, conversion=1e-6,
    )
    module = nwbfile.create_processing_module("ecephys", "Processed ephys")
    filtered = FilteredEphys()
    module.add(filtered)
    filtered.add_electrical_series(processed)
    write(nwbfile, "multiple_series.nwb")


def write_legacy_series():
    nwbfile = make_file("sorrel-fixture-legacy-series")
    add_series(nwbfile, good_data(), name="A_first", rate=1000.0, starting_time=0.52)
    add_series(nwbfile, good_data(), name="Z_second", rate=np.float32(1000), starting_time=0.5)
    name = "legacy_series.nwb"
    write(nwbfile, name)
    with h5py.File(OUT / name, "r+") as f:
        attrs = f["acquisition/A_first"].attrs
        del attrs["neurodata_type"]
        attrs["neurodata_type"] = np.bytes_("ElectricalSeries")


def write_units_and_raw():
    units = make_file("sorrel-fixture-units-only", resolution=0.001)
    units.add_trial(start_time=0.6, stop_time=0.7)
    write(units, "units_only.nwb")
    raw = make_file("sorrel-fixture-raw-only", units=False)
    add_series(raw, good_data(), rate=1000.0, starting_time=0.5, conversion=1e-6)
    write(raw, "raw_only.nwb")


def write_curated_units():
    """Use the same Units column layout as sorrel.nwb.add_units_to_nwbfile."""
    nwbfile = make_file("sorrel-fixture-curated", units=False, electrodes=False)

    def ragged(name, rows, dtype, doc):
        data = VectorData(name=name, description=doc,
                          data=np.array([x for row in rows for x in row], dtype=dtype))
        index = VectorIndex(name=f"{name}_index", target=data,
                            data=np.cumsum([len(row) for row in rows], dtype=np.int64))
        return [data, index]

    nwbfile.units = Units(name="units", id=[9, 5], resolution=1.0 / 1000.0, columns=[
        *ragged("spike_times", [[0.55, 0.79], [0.6, 0.7, 0.75]], np.float64,
                "Spike times in seconds: sample_index / sample_rate + starting_time."),
        VectorData(name="quality", description="Sorrel curation label: good, mua, noise or unsorted.",
                   data=["mua", "good"]),
        VectorData(name="origin", description="How the unit was made", data=["Auto[kilosort4]", "Sorrel[Manual]"]),
        *ragged("parent_unit_ids", [[], [9, 77]], np.int64, "Unit ids this unit came from in Sorrel"),
        *ragged("tags", [[], ["split", "review:waveform"]], str, "Sorrel tags"),
        VectorData(name="unit_name", description="The sorter's unit id (SpikeInterface unit_ids) when it is not an integer.",
                   data=["007", "unit_a"]),
        VectorData(name="n_spikes", description="Sorrel QC metric. Number of spikes assigned to the cluster after curation edits (count); 0 for an empty cluster.",
                   data=[2, 3]),
        VectorData(name="l_ratio", description="Sorrel QC metric. Mean chi-squared survival (df = 3 PCs) of background spikes' Mahalanobis squared distances divided by n_in; lower is better; NaN with fewer than 50 spikes or no PC features.",
                   data=[float("nan"), 0.25]),
    ])
    write(nwbfile, "curated_units.nwb")


def write_reader_edge_cases():
    one = make_file("one-channel")
    one.add_acquisition(ElectricalSeries(
        name="Raw", data=np.arange(N_SAMPLES, dtype=np.int16),
        electrodes=one.create_electrode_table_region([0], "only channel"),
        rate=1000.0, starting_time=0.5, conversion=1e-6,
    ))
    write(one, "one_channel.nwb")

    exact_rate = 30000.123
    units = make_file("precise-units", units=False, electrodes=False)
    units.units = Units(name="units", resolution=1.0 / exact_rate)
    # The two rates round this timestamp to different samples.
    units.add_unit(id=0, spike_times=[0.5 + 30.49999998 / exact_rate])
    write(units, "precise_units.nwb")
    raw = make_file("rounded-raw", units=False)
    add_series(raw, good_data(), rate=np.float32(exact_rate), starting_time=0.5)
    write(raw, "rounded_raw.nwb")

    shifted = make_file("shifted-raw", units=False)
    shifted.fields["timestamps_reference_time"] = SESSION_START - timedelta(milliseconds=50)
    add_series(shifted, good_data(), rate=1000.0, starting_time=0.5)
    write(shifted, "shifted_raw.nwb")


def write_single_file_precise_rates():
    rate = 30000.123
    for length, sample in [("short", 100), ("long", 330_000_000)]:
        name = f"precise_rate_{length}"
        nwbfile = make_file(name, units=False)
        nwbfile.units = Units(name="units", resolution=1 / rate)
        nwbfile.add_unit(id=0, spike_times=[sample / rate])
        # Chunked data with no allocated samples, even for the long recording.
        data = H5DataIO(shape=(sample + 2, N_CHANNELS), dtype=np.int16, chunks=(64, N_CHANNELS))
        add_series(nwbfile, data, rate=np.float32(rate), starting_time=0.0)
        write(nwbfile, name + ".nwb")


def write_foreign_quality():
    nwbfile = make_file("sorrel-fixture-quality", units=False, electrodes=False)
    for name in ["quality", "KSLabel", "unrelated_number"]:
        nwbfile.add_unit_column(name=name, description=f"Foreign {name}")
    for (idx, times), quality, ks in zip(UNITS, [" GOOD ", "ambiguous", "unknown"], ["mua", "NoIsE", "???"]):
        nwbfile.add_unit(id=idx, spike_times=times, quality=quality, KSLabel=ks, unrelated_number=3.5)
    write(nwbfile, "foreign_quality.nwb")


def write_not_electrical():
    nwbfile = make_file("sorrel-fixture-not-electrical")
    data = (
        np.arange(N_SAMPLES)[:, None] + np.arange(N_CHANNELS)[None, :]
    ).astype(np.float32)
    nwbfile.add_acquisition(
        TimeSeries(name="Behavior", data=data, rate=1000.0, unit="m")
    )
    write(nwbfile, "not_electrical.nwb")


def write_subtype():
    ns_name = "ndx-sorrel-test"
    spec = NWBGroupSpec(
        doc="ElectricalSeries subtype used to test reader subtype handling.",
        neurodata_type_def="TestSeries",
        neurodata_type_inc="ElectricalSeries",
    )
    ns_builder = NWBNamespaceBuilder(
        doc="sorrel NWB reader test fixtures",
        name=ns_name,
        version="0.1.0",
        author="sorrel",
    )
    ns_builder.add_spec(f"{ns_name}.extensions.yaml", spec)
    with tempfile.TemporaryDirectory() as tmpdir:
        ns_builder.export(f"{ns_name}.namespace.yaml", outdir=tmpdir)
        load_namespaces(os.path.join(tmpdir, f"{ns_name}.namespace.yaml"))
        test_series = get_class("TestSeries", ns_name)
        nwbfile = make_file("sorrel-fixture-subtype")
        region = nwbfile.create_electrode_table_region(CHANNEL_REGION, "permuted subset")
        nwbfile.add_acquisition(
            test_series(
                name="TestSeriesRaw",
                data=good_data(),
                electrodes=region,
                rate=1000.0,
                starting_time=0.5,
                conversion=1e-6,
                offset=0.0,
            )
        )
        write(nwbfile, "subtype.nwb", cache_spec=True)


def main():
    write_good()
    write_uint16()
    for dtype, value in [
        ("int8", -120), ("uint8", 250), ("int32", -2_000_000_000),
        ("uint32", 4_000_000_000), ("int64", -2**48),
        ("float32", 0.125), ("float64", 1e-50), ("uint64", 2**63),
    ]:
        write_dtype(dtype, value)
    write_timestamps()
    write_timestamps(irregular=True)
    write_multiple_series()
    write_legacy_series()
    write_units_and_raw()
    write_curated_units()
    write_reader_edge_cases()
    write_single_file_precise_rates()
    write_foreign_quality()
    write_not_electrical()
    write_subtype()


if __name__ == "__main__":
    main()

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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pynwb import NWBFile, NWBHDF5IO, get_class, load_namespaces
from pynwb.base import TimeSeries
from pynwb.ecephys import ElectricalSeries
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


def make_file(identifier):
    """NWBFile with the shared Device, ElectrodeGroup, electrodes and units."""
    nwbfile = NWBFile(
        session_description="sorrel NWB reader fixture",
        identifier=identifier,
        session_start_time=SESSION_START,
        file_create_date=[FILE_CREATE],
        session_id="sorrel-fixture",
    )
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
    for unit_id, spike_times in UNITS:
        nwbfile.add_unit(id=unit_id, spike_times=spike_times)
    return nwbfile


def add_series(nwbfile, data, name="ElectricalSeriesRaw", **kwargs):
    region = nwbfile.create_electrode_table_region(CHANNEL_REGION, "permuted subset")
    nwbfile.add_acquisition(
        ElectricalSeries(name=name, data=data, electrodes=region, **kwargs)
    )


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
        good_data(),
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


def write_float64():
    nwbfile = make_file("sorrel-fixture-float64")
    data = np.arange(N_SAMPLES)[:, None] + np.arange(N_CHANNELS)[None, :]
    add_series(
        nwbfile,
        data.astype(np.float64),
        rate=1000.0,
        starting_time=0.5,
        conversion=1e-6,
        offset=0.0,
    )
    write(nwbfile, "float64.nwb")


def write_timestamps():
    nwbfile = make_file("sorrel-fixture-timestamps")
    add_series(
        nwbfile,
        good_data(),
        timestamps=0.5 + np.arange(N_SAMPLES) / 1000.0,
        conversion=0.195e-6,
        offset=-2e-6,
        channel_conversion=[1.0, 2.0, 1.0, 0.5],
    )
    write(nwbfile, "timestamps.nwb")


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
    write_float64()
    write_timestamps()
    write_not_electrical()
    write_subtype()


if __name__ == "__main__":
    main()

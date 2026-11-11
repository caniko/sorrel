"""An analyzer whose recording is a SpikeInterface binary folder.

`rec.save(folder=..., format='binary')` returns a BinaryFolderRecording, so
the analyzer's `recording.json` only names that folder and the extractor's
own kwargs (rate, dtype, bin path) live in the folder's `binary.json`.
Sorrel has to follow that link to open it. Preprocessing chains that keep
every channel and every sample are followed the same way; slicing ones are
refused by name.
"""
from datetime import datetime, timezone
from pathlib import Path
import warnings

import numpy as np
import pytest
import spikeinterface.core as si
import spikeinterface.preprocessing as pre
from pynwb import NWBFile

import sorrel
from sorrel.nwb import add_units_to_nwbfile


def test_open_analyzer_on_binary_folder_recording(tmp_path: Path) -> None:
    warnings.simplefilter("ignore")
    rec, sort = si.generate_ground_truth_recording(durations=[3.0], num_units=5, seed=0)
    saved = rec.save(folder=tmp_path / "rec", format="binary")
    si.create_sorting_analyzer(sort, saved, format="binary_folder",
                               folder=tmp_path / "sa", sparse=False)

    s = sorrel.Session.open(tmp_path / "sa")
    assert s.sample_rate == rec.sampling_frequency
    assert s.n_clusters == 5
    for c in range(s.n_clusters):
        np.testing.assert_array_equal(
            s.spike_times(c), sort.get_unit_spike_train(str(s.unit_id(c)))
        )

    # export_phy re-reads every trace through the provider's mmap: the copied
    # raw must hold exactly one frame per sample the analyzer recorded.
    phy = tmp_path / "phy"
    s.export_phy(phy, copy_raw=True)
    assert (phy / "recording.dat").stat().st_size == (
        rec.get_num_samples(segment_index=0) * rec.get_num_channels()
        * np.dtype(saved.get_dtype()).itemsize
    )

    nwb = NWBFile("x", "x", datetime(2026, 1, 1, tzinfo=timezone.utc))
    units = add_units_to_nwbfile(s, nwb)
    assert units.resolution == 1.0 / rec.sampling_frequency


def test_open_analyzer_on_preserving_preprocessing_chain(tmp_path: Path) -> None:
    warnings.simplefilter("ignore")
    rec, sort = si.generate_ground_truth_recording(durations=[3.0], num_units=5, seed=0)
    saved = rec.save(folder=tmp_path / "rec", format="binary")
    chained = pre.bandpass_filter(pre.common_reference(saved))
    si.create_sorting_analyzer(sort, chained, format="binary_folder",
                               folder=tmp_path / "sa", sparse=False)

    s = sorrel.Session.open(tmp_path / "sa")
    assert s.sample_rate == rec.sampling_frequency
    assert s.n_clusters == 5
    for c in range(s.n_clusters):
        np.testing.assert_array_equal(
            s.spike_times(c), sort.get_unit_spike_train(str(s.unit_id(c)))
        )

    nwb = NWBFile("x", "x", datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert add_units_to_nwbfile(s, nwb).resolution == 1.0 / rec.sampling_frequency


def test_open_analyzer_on_channel_slice_names_the_class(tmp_path: Path) -> None:
    warnings.simplefilter("ignore")
    rec, sort = si.generate_ground_truth_recording(durations=[3.0], num_units=5, seed=0)
    saved = rec.save(folder=tmp_path / "rec", format="binary")
    sliced = saved.select_channels(saved.channel_ids[:2])
    # The generator tags units with a channel id; keep the analyzer's channel
    # validation out of the way.
    sort.delete_property("main_channel_id")
    si.create_sorting_analyzer(sort, sliced, format="binary_folder",
                               folder=tmp_path / "sa", sparse=False)

    with pytest.raises(RuntimeError, match="ChannelSliceRecording"):
        sorrel.Session.open(tmp_path / "sa")

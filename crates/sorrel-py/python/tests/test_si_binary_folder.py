"""An analyzer whose recording is a SpikeInterface binary folder.

`rec.save(folder=..., format='binary')` returns a BinaryFolderRecording, so
the analyzer's `recording.json` only names that folder and the extractor's
own kwargs (rate, dtype, bin path) live in the folder's `binary.json`.
Sorrel has to follow that link to open it.
"""
from pathlib import Path
import warnings

import numpy as np
import spikeinterface.core as si

import sorrel


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

"""The round trip docs/src/usage/saving.md promises: edit a SortingAnalyzer
in sorrel.Session, save, and spikeinterface.curation.apply_curation
reproduces the units. Expected trains come from the original sorting."""
import warnings
from pathlib import Path

import numpy as np
import spikeinterface.core as si
import spikeinterface.curation as sc
import spikeinterface.extractors as se

import sorrel

MOVED = [0, 1, 2, 5]


def test_apply_curation_reproduces_session(tmp_path: Path) -> None:
    warnings.simplefilter("ignore")
    # SI 0.105's generator names units "0".."4" (strings), as many extractors do.
    rec, sort = si.generate_ground_truth_recording(durations=[3.0], num_units=5, seed=0)
    # Sorrel maps the trace from a raw binary named in recording.json.
    si.write_binary_recording(rec, file_paths=[tmp_path / "rec.raw"], dtype="int16")
    raw = se.read_binary(tmp_path / "rec.raw", sampling_frequency=rec.sampling_frequency,
                         num_channels=rec.get_num_channels(), dtype="int16",
                         channel_ids=rec.channel_ids)
    raw.set_probe(rec.get_probe(), in_place=True)
    analyzer = si.create_sorting_analyzer(sort, raw, format="binary_folder",
                                          folder=tmp_path / "sa", sparse=False)
    train = {str(u): sort.get_unit_spike_train(u) for u in sort.unit_ids}
    # Sorrel merges into the target id; SI gives both halves of a split new
    # ids, so the left-behind spikes of "3" get the next free one.
    want = {
        "0": train["0"],
        "2": np.sort(np.concatenate([train["1"], train["2"]])),
        "4": train["4"],
        "5": np.sort(train["3"])[MOVED],
        "6": np.delete(np.sort(train["3"]), MOVED),
    }

    s = sorrel.Session.open(tmp_path / "sa")
    assert [s.unit_id(c) for c in range(s.n_clusters)] == [0, 1, 2, 3, 4]
    s.set_label(0, "good")
    s.merge([1], 2)
    new = s.split(3, MOVED)
    np.testing.assert_array_equal(s.spike_times(new), want["5"])
    s.save()
    del s

    curated = sc.apply_curation(analyzer, sorrel.read_curation(tmp_path / "sa")).sorting
    assert sorted(curated.unit_ids) == sorted(want)
    for uid, spikes in want.items():
        np.testing.assert_array_equal(np.sort(curated.get_unit_spike_train(uid)), spikes)
    quality = dict(zip(curated.unit_ids, curated.get_property("quality")))
    assert quality["0"] == "good"

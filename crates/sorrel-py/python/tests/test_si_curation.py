"""The round trip docs/src/usage/saving.md promises: edit a SortingAnalyzer
in sorrel.Session, save, and spikeinterface.curation.apply_curation
reproduces the units. Expected trains come from the original sorting."""
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import spikeinterface.core as si
import spikeinterface.curation as sc
import spikeinterface.extractors as se

import sorrel

MOVED = [0, 1, 2, 5]


def make_analyzer(tmp_path: Path):
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
    return analyzer, sort


def test_apply_curation_reproduces_session(tmp_path: Path) -> None:
    analyzer, sort = make_analyzer(tmp_path)
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
    s.set_label(1, "noise")  # merged away: the merged unit keeps "good"
    s.set_label(2, "good")
    s.merge([1], 2)
    new = s.split(3, MOVED)
    np.testing.assert_array_equal(s.spike_times(new), want["5"])
    s.save()
    del s

    doc = sorrel.read_curation(tmp_path / "sa")
    assert isinstance(doc, list), "a history with merges/splits is SI's list form"
    curated = sc.apply_curation(analyzer, doc).sorting
    assert sorted(curated.unit_ids) == sorted(want)
    for uid, spikes in want.items():
        np.testing.assert_array_equal(np.sort(curated.get_unit_spike_train(uid)), spikes)
    quality = dict(zip(curated.unit_ids, curated.get_property("quality")))
    assert quality["0"] == "good"
    assert quality["2"] == "good"


def test_apply_curation_reproduces_split_labels_and_repeated_edits(tmp_path: Path) -> None:
    """A history one Curation cannot express: a label on a split child,
    merging a split child, splitting a merged unit and splitting one unit
    twice. Expected unit ids and trains come from the original sorting."""
    analyzer, sort = make_analyzer(tmp_path)
    t = {str(u): np.sort(sort.get_unit_spike_train(u)) for u in sort.unit_ids}

    s = sorrel.Session.open(tmp_path / "sa")
    a = s.split(0, [0])          # labelled child
    s.set_label(a, "good")
    b = s.split(2, [0])          # child merged into unit 1
    s.merge([b], 1)
    s.merge([4], 3)              # unit 3 absorbs unit 4, then gets split
    s.split(3, [0])
    s.split(1, [0])              # first split of (the remainder of) unit 1
    s.split(1, [0])              # second split of the same sorrel cluster
    s.save()
    del s

    u1 = np.sort(np.concatenate([t["1"], t["2"][[0]]]))
    u3 = np.sort(np.concatenate([t["3"], t["4"]]))
    # Child ids follow sorrel's allocated cluster ids ("5".."9"); the spikes
    # left behind get fresh ids above every id the history allocates
    # ("10".."14"), in split order.
    want = {
        "5": t["0"][[0]],
        "10": np.delete(t["0"], [0]),
        "11": np.delete(t["2"], [0]),
        "7": u3[[0]],
        "12": u3[1:],
        "8": u1[0:1],
        "9": u1[1:2],
        "14": u1[2:],
    }

    curated = sc.apply_curation(analyzer, sorrel.read_curation(tmp_path / "sa")).sorting
    assert sorted(curated.unit_ids) == sorted(want)
    for uid, spikes in want.items():
        np.testing.assert_array_equal(np.sort(curated.get_unit_spike_train(uid)), spikes)
    quality = dict(zip(curated.unit_ids, curated.get_property("quality")))
    assert quality["5"] == "good"
    assert [q for u, q in quality.items() if u != "5"] == [""] * (len(want) - 1)


def test_reopen_undo_relabel_restores_presession_label(tmp_path: Path) -> None:
    # sorrel rewrites curation.json with the head's labels, and replaying the
    # journal onto those would make undo restore the edited label.
    make_analyzer(tmp_path)
    (tmp_path / "sa" / "curation.json").write_text(json.dumps({
        "format_version": "2",
        "unit_ids": ["0", "1", "2", "3", "4"],
        "label_definitions": {"quality": {"label_options": ["good", "mua", "noise"],
                                          "exclusive": True}},
        "manual_labels": [{"unit_id": "2", "labels": {"quality": ["mua"]}}],
    }))
    s = sorrel.Session.open(tmp_path / "sa")
    assert s.labels()[2] == "mua"
    s.set_label(2, "good")
    s.save()
    del s
    s = sorrel.Session.open(tmp_path / "sa")
    assert s.labels()[2] == "good"
    s.undo()
    assert s.labels()[2] == "mua"


def test_reopen_undo_merge_restores_source_label(tmp_path: Path) -> None:
    # The merged-away source's label is not in the final labels step (the
    # unit is gone), so it can only come from the label source the journal
    # replays onto. Unit 1's "noise" comes from an earlier curation.json.
    make_analyzer(tmp_path)
    (tmp_path / "sa" / "curation.json").write_text(json.dumps({
        "format_version": "2",
        "unit_ids": ["0", "1", "2", "3", "4"],
        "label_definitions": {"quality": {"label_options": ["good", "mua", "noise"],
                                          "exclusive": True}},
        "manual_labels": [{"unit_id": "1", "labels": {"quality": ["noise"]}}],
    }))
    s = sorrel.Session.open(tmp_path / "sa")
    assert s.labels()[1] == "noise"
    s.set_label(2, "good")
    s.merge([1], 2)
    s.save()
    del s
    s = sorrel.Session.open(tmp_path / "sa")
    assert s.labels()[2] == "good"
    s.undo()
    assert s.labels()[1] == "noise"
    assert s.labels()[2] == "good"


def test_read_curation_rejects_unrepresentable_edits(tmp_path: Path) -> None:
    # A rescue (AddSpikes) has no SI curation slot; the document records it
    # and the reader refuses instead of silently dropping spikes.
    p = tmp_path / "curation.json"
    p.write_text(json.dumps({
        "format_version": "2",
        "unit_ids": ["0"],
        "sorrel_unrepresentable": ["add 3 spikes to 1 clusters: SI cannot add spikes"],
    }))
    with pytest.raises(ValueError, match="add 3 spikes"):
        sorrel.read_curation(p)

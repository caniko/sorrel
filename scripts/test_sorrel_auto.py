# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "spikeinterface[full]>=0.105",
#   "skops",
#   "huggingface_hub",
#   "anthropic",
# ]
# ///
"""Self-check for sorrel_auto.py on synthetic ground truth (no network
unless --unitrefine): one unit split in two must come back merged, an exact
duplicate must come back `redundant`, and the output must be a valid
SpikeInterface curation document carrying a quality label and a `src:` tag
for every unit.

    uv run --script scripts/test_sorrel_auto.py [--unitrefine] [--keep DIR] [--brief-only]
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import spikeinterface.core as sc
import spikeinterface.curation as scur
from spikeinterface.exporters import export_to_phy


def make_phy(folder: Path) -> tuple[int, int, int, int]:
    rec, gt = sc.generate_ground_truth_recording(durations=[180.0], num_units=8, seed=7)
    trains = {int(u): gt.get_unit_spike_train(u) for u in gt.unit_ids}
    rng = np.random.default_rng(0)
    # Oversplit unit 0 into two halves; duplicate unit 1 exactly.
    a = trains.pop(0)
    half = rng.random(a.size) < 0.5
    trains[100], trains[101] = a[half], a[~half]
    trains[102] = trains[1].copy()
    sorting = sc.NumpySorting.from_unit_dict(trains, rec.sampling_frequency)
    analyzer = sc.create_sorting_analyzer(sorting, rec, sparse=True)
    analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels", "spike_amplitudes"])
    export_to_phy(analyzer, folder, compute_pc_features=False, copy_binary=True, remove_if_exists=True)
    # spike_clusters.npy holds unit indices (sorting order); sorrel and the
    # sidecar both use those, not the `si_unit_id` column.
    ids = [int(u) for u in sorting.unit_ids]
    return ids.index(100), ids.index(101), ids.index(1), ids.index(102)


BRIEF = (
    "# sorrel brief v1\nrec: 180.0 s\nlegend: x\ncards: one row per unit\ndrill: brief{clusters:[..]}\n"
    "ledger\n"
    "cards:\nid\tlab\tflags\tfr\n3\tgood\t=\t4.1\n7\tmua\t+A\t0.2\nomitted: cards 9; raise budget\n"
)


def test_brief_cases() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crates" / "sorrel" / "src"))
    import sorrel_auto as sa

    head, card_head, rows = sa.split_brief(BRIEF)
    assert head.endswith("drill: brief{clusters:[..]}\nledger") and "\ncards:\n" not in head
    assert card_head == "id\tlab\tflags\tfr" and sorted(rows) == [3, 7], rows
    rec = {"bombcell": "mua", "unitrefine": "good", "unitrefine_probability": 0.55, "members": [7]}
    msg = sa.unit_message(card_head, rows, rec, {})
    assert msg == (
        "Decide this unit.\nid\tlab\tflags\tfr\n7\tmua\t+A\t0.2\n"
        'classifiers: {"bombcell":"mua","unitrefine":"good","unitrefine_probability":0.55}'
    ), msg
    merged = sa.unit_message(card_head, rows, {**rec, "members": [3, 7]}, {3: "eval: z=1"})
    assert "merge of 3,7" in merged and "3\tgood" in merged and "7\tmua" in merged and "\neval: z=1\n" in merged
    assert sa.eval_line("merge {3,4}: rpc 0,0→.05\n") == "merge {3,4}: rpc 0,0→.05"
    assert sa.merge_message("eval: m", {"slay", "similarity_correlograms"}).endswith("proposed_by: similarity_correlograms,slay")
    try:
        sa.unit_message(card_head, rows, {**rec, "members": [99]}, {})
    except SystemExit:
        pass
    else:
        raise AssertionError("a unit without a card must fail loudly")


def main() -> None:
    if "--brief-only" in sys.argv:
        test_brief_cases()
        print("ok: brief cases")
        return
    test_brief_cases()
    keep = sys.argv[sys.argv.index("--keep") + 1] if "--keep" in sys.argv else None
    root = Path(keep or tempfile.mkdtemp(prefix="sorrel-auto-"))
    phy, out = root / "phy", root / "phy" / "sorrel_auto"
    split_a, split_b, orig, dup = make_phy(phy)

    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "crates" / "sorrel" / "src" / "sorrel_auto.py"), "--data-dir", str(phy), "--out", str(out), "--gain-to-uv", "1.0", "--jobs", "1"]
    if "--unitrefine" not in sys.argv:
        cmd.append("--no-unitrefine")
    subprocess.run(cmd, check=True)

    doc = json.loads((out / "curation.json").read_text())
    scur.validate_curation_dict(doc)
    by_unit = {e["unit_id"]: e["labels"] for e in doc["manual_labels"]}
    for u in doc["unit_ids"]:
        assert by_unit[u]["quality"][0] in ("good", "mua", "noise"), (u, by_unit[u])
        assert any(t.startswith("src:") for t in by_unit[u]["tags"]), (u, by_unit[u])
    merged = [set(m["unit_ids"]) for m in doc["merges"]]
    assert any({split_a, split_b} <= g for g in merged), f"oversplit not merged: {doc['merges']}"
    redundant = {u for u, l in by_unit.items() if "redundant" in l["tags"]}
    assert redundant & {orig, dup}, f"duplicate not flagged: {redundant}"
    assert (out / "evidence.json").exists()
    print(f"ok: {len(doc['merges'])} merge(s), redundant {sorted(redundant)}, output in {out}")


if __name__ == "__main__":
    main()

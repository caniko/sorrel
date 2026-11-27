"""Offline CLI and published-format fixtures for the DANDI review kit."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dandi_review as kit


QC_FIELDS = """cluster_id n_spikes firing_rate_hz mean_amplitude presence_ratio
isi_violations isi_violation_ratio amplitude_cutoff amplitude_snr drift_correlation
longest_silent_gap composite_quality isolation_distance_sq l_ratio nn_isolation d_prime
silhouette trough_to_peak_ms half_width_ms peak_trough_ratio repolarisation_slope
n_extrema x_um y_um footprint_decay_um footprint_r80_um n_active_channels
propagation_ms_per_100um burst_index rp_estimate_ms rp_contamination amplitude_dip
drift_slope srp_contamination srp_rp_ms rpv_post_mean rpv_post_lo95 rpv_post_hi95
n_epochs epoch_amp_ratio epoch_rate_ratio""".split()
ASSET = {"asset_id": "asset-1", "path": "sub-a/sub-a.nwb", "size_bytes": 123,
         "encoding": "hdf5", "open_status": "ok", "evidence_level": "spike_times",
         "pin_verification": "published_version"}
CLUSTER_HEADER = "id\tn\tlabel\ttags\torigin\tparents\tnote\n"


def curation(ids, labels=(), merges=(), splits=(), removed=()):
    return {
        "format_version": "2", "unit_ids": ids,
        "label_definitions": {"quality": {"label_options": ["good", "mua", "noise"], "exclusive": True}},
        "manual_labels": [{"unit_id": unit, "labels": {"quality": [label]}} for unit, label in labels],
        "merges": list(merges), "splits": list(splits), "removed": list(removed),
    }


class ReviewKitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = self.root / "store"
        self.stage = self.root / "stage"
        self.stage.mkdir()
        (self.stage / "artifacts").mkdir()
        (self.stage / "report.md").write_text("# Source review\n", encoding="utf-8")

    def cli(self, *args, ok=True, error=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                status = kit.main([str(arg) for arg in args])
            except SystemExit as exc:
                status = exc.code
        self.assertEqual(status == 0, ok, err.getvalue())
        if error:
            self.assertIn(error, err.getvalue())
        return out.getvalue()

    def dump(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def clusters(self, name, rows):
        path = self.root / name
        path.write_text(CLUSTER_HEADER + "".join(
            f"{i}\t{n}\t{label}\t-\tAuto[ks4]\t-\t{note}\n" for i, label, note, *counts in rows
            for n in [counts[0] if counts else 100]
        ), encoding="utf-8")
        return path

    def review(self, verified=True):
        return {
            "schema_version": "1", "dandiset": {"id": "000006", "version": "0.220126.1855"},
            "created_at": "2026-01-01T00:00:00Z", "mode": "human",
            "contributor": {"name": "Reviewer"}, "sorrel": {"version": "0.5.0", "rev": "a447b68"},
            "skill_rev": "test", "thresholds": kit.DEFAULT_THRESHOLDS,
            "asset_selection": {"policy": "all", "n_assets_total": 1},
            "models": [], "assets": [dict(ASSET, units_before=3, good_before=2,
                                         unlabeled_before=1, n_good_rp_contaminated=1)],
            "findings": [{"id": "f01", "code": "qc.refractory_contamination", "severity": "major",
                          "scope": "asset", "asset_id": ASSET["asset_id"],
                          "statement_md": "Review refractory violations.", "human_verified": verified}],
            "improvements": [], "files": [],
        }

    def pack(self, review=None):
        self.dump("stage/review.json", self.review() if review is None else review)
        result = self.cli("pack", self.stage, "--store", self.store)
        review_id = result.split()[0]
        return self.store / "reviews" / "000006" / review_id

    def audit_fixture(self, verified=True, note="", **finding):
        doc = self.review(verified)
        asset = dict(doc["assets"][0], blob_id="blob-1", sha256="a" * 64,
                     evidence_level="raw", duration_s=10, sample_rate_hz=20_000,
                     unique_metadata={"spike_extent_s": 9, "raw_access": {"result": "windows 8/8"}})
        doc["assets"] = [asset]
        doc["findings"][0].update(finding)
        artifact = self.stage / "artifacts"
        asset_dir = artifact / asset["path"]
        asset_dir.mkdir(parents=True, exist_ok=True)
        (asset_dir / "clusters.before.tsv").write_text(
            CLUSTER_HEADER + "".join(f"{i}\t100\t{label}\t-\t-\t-\t\n"
                                     for i, label in enumerate(("good", "good", "good2"))))
        (artifact / "dandiset.json").write_text(json.dumps({
            "dandiset_id": "000006", "pinned_version": doc["dandiset"]["version"],
            "assets": [asset],
        }))
        evidence = artifact / "evidence.json"
        evidence.write_text('{"rpv_post_lo95": 0.2}')
        doc["findings"][0]["evidence"] = [{
            "artifact_sha256": kit.digest_file(evidence)[0], "pointer": "/rpv_post_lo95",
        }]
        (self.stage / "report.md").write_text(note)
        return doc

    def test_audit_ready_and_unverified_and_deterministic(self):
        bundle = self.pack(self.audit_fixture())
        first = self.cli("audit-review", bundle)
        self.assertEqual(json.loads(first)["dandisets"][0]["verdict"], "outreach_ready")
        self.assertEqual(first, self.cli("audit-review", self.store))
        bundle = self.pack(self.audit_fixture(False, "RP confirmed"))
        result = json.loads(self.cli("audit-review", bundle))["dandisets"][0]
        self.assertEqual(result["verdict"], "needs_human_review")
        findings = result["reviews"][0]["findings"]
        self.assertIn("HUMAN_VERIFICATION_REQUIRED", findings[0]["blockers"])
        self.assertIn("UNVERIFIED_CONFIRMATION_WORDING", result["reviews"][0]["warnings"])

    def test_audit_reconciles_pins_and_redacts_private_text(self):
        doc = self.audit_fixture(statement_md="Private subject ID: private-patient-42")
        doc["assets"][0]["sha256"] = "b" * 64
        bundle = self.pack(doc)
        j, m = self.root / "audit.json", self.root / "audit.md"
        self.cli("audit-review", bundle, "--json-out", j, "--md-out", m)
        result = json.loads(j.read_text())["dandisets"][0]
        self.assertEqual(result["verdict"], "blocked")
        self.assertIn("PIN_ASSET_MISMATCH", result["reviews"][0]["assets"][0]["blockers"])
        self.assertIn("POSSIBLE_PRIVATE_IDENTIFIER", result["reviews"][0]["findings"][0]["blockers"])
        self.assertNotIn("private-patient-42", j.read_text() + m.read_text())

    def test_audit_corpus_risk_signals(self):
        doc = self.audit_fixture(False, """
        Epochs in milliseconds, a 1000× ratio versus spike extent.
        Trace span mismatch. Waveform probe hit the 900 s cap.
        CCG-only merge suggestions without waveform support.
        CA1 bursting and cortical stimulation confound epoch metrics.
        NWB Inspector timed out. Raw labels good2, excellent and numeric quality;
        the cell_type column is not a quality label. Nominal analysis grid.
        """)
        a = self.stage / "artifacts" / doc["assets"][0]["path"]
        (a / "nwb_inspector.json").write_text(json.dumps({"messages": [
            {"check_function_name": "check_ascending_spike_times", "message": "Equal consecutive spike times"}
        ]}))
        doc["findings"][0]["code"] = "curation.duplicate_unit"
        bundle = self.pack(doc)
        result = json.loads(self.cli("audit-review", bundle))["dandisets"][0]["reviews"][0]
        self.assertLessEqual({
            "SUSPICIOUS_EPOCH_SPAN", "TRACE_SPIKE_SPAN_MISMATCH", "WAVEFORM_PROBE_TIMEOUT",
            "CCG_ONLY_MERGE_SCREENING", "RP_EPOCH_CONTEXT_CONFOUND", "INSPECTOR_TIMEOUT",
            "UNMAPPED_AUTHOR_LABELS", "NONSTANDARD_CLOCK",
        }, set(result["warnings"]))
        self.assertIn("NON_INCREASING_SOURCE_SPIKES", result["assets"][0]["blockers"])
        self.assertIn("EQUAL_CONSECUTIVE_SOURCE_SPIKES", result["assets"][0]["blockers"])
        self.assertIn("IDENTICAL_CROSS_UNIT_SPIKE_VECTORS", result["findings"][0]["blockers"])

    def test_audit_does_not_treat_pinned_external_avi_or_good2_as_a_source_claim(self):
        doc = self.audit_fixture(
            statement_md="Inspector reported missing external AVI; two author good units.",
            code="nwb.inspector_critical",
        )
        a = self.stage / "artifacts" / doc["assets"][0]["path"]
        (a / "nwb_inspector.json").write_text(json.dumps({"messages": [{
            "check_function_name": "check_image_series_external_file_valid",
            "message": "The external file 'sidecar.avi' does not exist.", "importance": "CRITICAL",
        }]}))
        snapshot = self.stage / "artifacts" / "dandiset.json"
        pin = json.loads(snapshot.read_text())
        pin["assets"].append(dict(ASSET, path="sub-a/sidecar.avi", asset_id="sidecar"))
        snapshot.write_text(json.dumps(pin))
        doc["asset_selection"] = {"policy": "all", "scope": "nwb", "n_assets_total": 2, "n_nwb_assets": 1}
        result = json.loads(self.cli("audit-review", self.pack(doc)))["dandisets"][0]["reviews"][0]
        self.assertEqual(result["assets"][0]["mapped_good"], 2)
        self.assertEqual(result["assets"][0]["inspector_external_file_false_criticals"], 1)
        self.assertIn("INSPECTOR_STREAMING_ARTIFACT_CLAIM", result["findings"][0]["blockers"])

    def test_audit_superseded_reviews_do_not_count_and_crosscheck_reports(self):
        first = self.pack(self.audit_fixture(False))
        second = self.pack(dict(self.audit_fixture(), supersedes=first.name))
        batch = self.root / "batch.md"
        batch.write_text(f"`{first.name}` is superseded by `{second.name}`")
        result = json.loads(self.cli("audit-review", self.store, "--batch-summary", batch))
        self.assertEqual(result["summary"]["active_review_bundles"], 1)
        self.assertEqual(result["summary"]["findings"]["outreach_ready"], 1)
        self.assertEqual(result["batch_summary_crosscheck"]["matched_active_reviews"], 1)

    def test_audit_checks_metric_direction_against_pointed_value(self):
        doc = self.audit_fixture(metric="rpv_post_lo95", threshold=0.3)
        result = json.loads(self.cli("audit-review", self.pack(doc)))
        finding = result["dandisets"][0]["reviews"][0]["findings"][0]
        self.assertEqual(finding["verdict"], "blocked")
        self.assertIn("METRIC_THRESHOLD_EVIDENCE_CONTRADICTS_CLAIM", finding["blockers"])

    def test_audit_cross_archive_same_path_is_not_an_asset_identity(self):
        self.pack(self.audit_fixture())
        doc = self.audit_fixture()
        doc["dandiset"]["id"] = "000007"
        doc["assets"][0]["asset_id"] = "asset-7"
        doc["findings"][0]["asset_id"] = "asset-7"
        path = self.stage / "artifacts" / "dandiset.json"
        pin = json.loads(path.read_text())
        pin.update(dandiset_id="000007", assets=doc["assets"])
        path.write_text(json.dumps(pin))
        self.dump("stage/review.json", doc)
        self.cli("pack", self.stage, "--store", self.store)
        result = json.loads(self.cli("audit-review", self.store))
        for dataset in result["dandisets"]:
            review = dataset["reviews"][0]
            self.assertIn("CROSS_ARCHIVE_SAME_PATH_AND_DIGEST", review["warnings"])
            self.assertEqual(review["shared_selected_paths"], 1)
            self.assertNotIn("PIN_ASSET_MISMATCH", review["blockers"])

    def summary(self, *args):
        return json.loads(self.cli("summarize-asset", "--asset-id", ASSET["asset_id"],
                                   "--path", ASSET["path"], "--size-bytes", "123",
                                    "--encoding", "hdf5", *args))

    def rewrite_bundle(self, source, edit, parent=None):
        doc = json.loads((source / "review.json").read_text())
        edit(doc)
        raw = kit.canonical_json_bytes(doc)
        target = (parent or self.root) / hashlib.sha256(raw).hexdigest()
        shutil.copytree(source, target)
        (target / "review.json").write_bytes(raw)
        return target

    def query(self, sql):
        with contextlib.closing(sqlite3.connect(self.store / "results.sqlite")) as db:
            return db.execute(sql).fetchall()

    def test_init_store_idempotent(self):
        self.cli("init-store", self.store)
        expected = {"reviews", "feedback", "dandisets", "dandisets.tsv", "results.tsv", "results.sqlite",
                    ".gitignore", "README.md"}
        self.assertEqual({p.name for p in self.store.iterdir()}, expected)
        self.assertIn("dandisets/<dandiset_id>.json", (self.store / "README.md").read_text())
        (self.store / "README.md").write_text("my notes")
        self.cli("init-store", self.store)
        self.assertEqual((self.store / "README.md").read_text(), "my notes")
        self.assertIn("review_id", (self.store / "results.tsv").read_text().splitlines()[0])
        self.assertEqual(self.query("SELECT count(*) FROM v_result_table"), [(0,)])
        self.assertIn("/feedback/\n", (self.store / ".gitignore").read_text())

    def test_pin_dataset_metadata_preserves_raw_payload_and_local_notes(self):
        self.cli("init-store", self.store)
        base = "http://127.0.0.1:1234/api"
        endpoint = f"{base}/dandisets/000006/"
        versions = ["0.220126.1855", "0.220127.1855"]
        raw = {
            "name": "Motor cortex", "description": "Task recordings", "citation": "Author et al.",
            "license": ["spdx:CC-BY-4.0"], "doi": "10.48324/dandi.000006/0.220126.1855",
            "contributor": [{"name": "Author", "identifier": "0000-0002-6670-7362",
                             "roleName": ["dcite:ContactPerson"], "email": "author@example.org",
                             "affiliation": [{"name": "Institution"}]}],
            "assetsSummary": {"species": [{"name": "House mouse"}], "approach": [{"name": "behavioral"}],
                              "measurementTechnique": [{"name": "spike sorting technique"}]},
            "about": [{"name": "motor cortex", "schemaKey": "Anatomy"}],
            "relatedResource": [{"identifier": "doi:10.1038/example", "unknown": True}],
            "futureField": {"nested": [True, None, 17]},
        }
        pages = {endpoint: {"embargo_status": "OPEN", "created": "2020-03-17T13:30:38.667000Z"}}
        for version in versions:
            version_url = f"{endpoint}versions/{version}/"
            pages[version_url] = dict(raw, version=version)
            pages[f"{version_url}assets/?metadata=true"] = {
                "results": [{"asset_id": "a", "path": "a.nwb", "size": 123},
                            {"asset_id": "b", "path": "notes.txt", "size": 2}], "count": 2}
        path = self.store / "dandisets/000006.json"
        snapshot = self.root / "dandiset.json"
        with patch.object(kit, "api_json", side_effect=pages.__getitem__):
            manifest = json.loads(self.cli("pin", "000006", "--api-base", base, "--version", versions[0],
                                           "--store", self.store))
            self.assertEqual(len(manifest["assets"]), 1)
            self.cli("dataset-meta", self.store, "set", "000006", "--probe", "NP1", "--set", "shanks=2")
            self.cli("pin", "000006", "--api-base", base, "--version", versions[1],
                     "--store", self.store, "--dataset-out", snapshot)
        doc = json.loads(path.read_text())
        self.assertEqual(path.read_bytes(), snapshot.read_bytes())
        self.assertEqual(path.read_bytes(), kit.canonical_json_bytes(doc))
        self.assertEqual(doc["dandi_raw_metadata"], dict(raw, version=versions[1]))
        self.assertEqual(doc["contributors"], [{"name": "Author", "email": "author@example.org",
                                                "orcid": "0000-0002-6670-7362", "roles": ["dcite:ContactPerson"],
                                                "affiliations": ["Institution"]}])
        self.assertEqual(doc["species"], ["House mouse"])
        self.assertEqual(doc["related_dois"], ["10.1038/example"])
        self.assertEqual(doc["modality"], ["behavioral"])
        self.assertEqual(doc["contact"], "author@example.org")
        self.assertEqual(doc["related_resources"], raw["relatedResource"])
        self.assertEqual(doc["anatomy"], ["motor cortex"])
        self.assertEqual(doc["measurement_techniques"], ["spike sorting technique"])
        self.assertEqual(doc["unique_metadata"], {"shanks": 2})
        self.assertEqual(doc["probe"], ["NP1"])
        self.assertEqual(doc["versions_seen"], versions)
        self.assertEqual(doc["assets_summary"]["total_size_bytes"], 125)
        self.assertEqual(len(doc["assets"]), 2)
        self.cli("project", self.store)
        self.assertEqual(json.loads(self.query("SELECT dandi_metadata_json FROM dandisets")[0][0]),
                         doc["dandi_raw_metadata"])
        self.assertEqual(self.query("SELECT description, citation, license FROM dandisets"),
                         [("Task recordings", "Author et al.", "spdx:CC-BY-4.0")])

    def test_dataset_meta_merge_validation_and_projection(self):
        self.cli("init-store", self.store)
        opts = ("dataset-meta", self.store, "set", "000006")
        self.cli(*opts, "--name", "Motor cortex", "--merge-json", '{"notes":{"old":1},"flag":true}')
        self.cli(*opts, "--merge-json", "@" + str(self.dump("updates.json", {"notes": {"new": 2}})),
                 "--set", "flag=false", "--set", "task=lick", "--sorter", "KS4", "--anatomy", "cortex")
        path = self.store / "dandisets/000006.json"
        doc = json.loads(path.read_text())
        expected = {"notes": {"old": 1, "new": 2}, "flag": False, "task": "lick"}
        self.assertEqual(doc["unique_metadata"], expected)
        self.assertEqual(doc["sorter"], ["KS4"])
        self.assertEqual(doc["anatomy"], ["cortex"])
        self.assertEqual(json.loads(self.query("SELECT unique_metadata_json FROM dandisets")[0][0]), expected)
        for malformed in ("{", "[]", "null", '{"x":NaN}'):
            self.cli(*opts, "--merge-json", malformed, ok=False)
            self.assertEqual(json.loads(path.read_text()), doc)
        self.cli(*opts, "--set", "missing-equals", ok=False, error="KEY=VALUE")
        self.cli("validate", self.store)
        self.cli("impact", self.store, "--format", "json")

    def test_dataset_meta_merges_legacy_registry_and_review_envelope(self):
        self.cli("init-store", self.store)
        path = self.store / "dandisets.tsv"
        path.write_text("dandiset_id\tname\tdoi\tlicense\n000006\tRegistry name\t\tCC-BY-4.0\n")
        self.cli("dataset-meta", self.store, "set", "000006", "--set", "local=true")
        self.pack(self.review() | {"dandiset": dict(self.review()["dandiset"], doi="10.1234/example",
                                                   name="Envelope name")})
        self.cli("project", self.store)
        name, doi, license_, unique, raw = self.query(
            "SELECT name, doi, license, unique_metadata_json, dandi_metadata_json FROM dandisets")[0]
        self.assertEqual((name, doi, license_), ("Registry name", "10.1234/example", "CC-BY-4.0"))
        self.assertEqual((json.loads(unique), json.loads(raw)), ({"local": True}, {}))
        self.assertEqual(path.read_text(), "dandiset_id\tname\tdoi\tlicense\n000006\tRegistry name\t\tCC-BY-4.0\n")
        with path.open("a") as stream:
            stream.write("000007\t\t\t\n")
        self.cli("dataset-meta", self.store, "set", "000007", "--set", "local=true")
        self.assertEqual(self.query("SELECT name FROM dandisets WHERE dandiset_id='000007'"), [("DANDI:000007",)])

    def test_dandiset_schema_and_store_reject_bad_dataset_metadata(self):
        self.cli("init-store", self.store)
        self.cli("dataset-meta", self.store, "set", "000006", "--name", "Motor cortex")
        schema = kit.schema("dandiset")
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["$id"], "https://nexus.beloda.co/sorrel/schema/dandiset-1.json")
        path = self.store / "dandisets/000006.json"
        doc = json.loads(path.read_text())
        for changes, error in [
            ({"dandiset_id": "000007"}, "does not match dandiset_id"),
            ({"typo": "no"}, "unexpected properties"),
            ({"unique_metadata": []}, "expected object"),
            ({"dandi_raw_metadata": None}, "expected object"),
            ({"updated_at": "2026-01-01T00:00:00.001Z"}, "date-time"),
            ({"updated_at": "2026-01-01T01:00:00+01:00"}, "expected pattern"),
            ({"assets": [{"asset_id": "a", "path": "a.nwb", "encoding": "hdf5",
                          "size_bytes": 2**63}]}, "must be <="),
        ]:
            with self.subTest(changes=changes):
                path.write_text(json.dumps(doc | changes))
                self.cli("validate", self.store, ok=False, error=error)
        path.unlink()
        path.symlink_to(self.dump("linked.json", doc))
        self.cli("validate", self.store, ok=False, error="no symlink")
        self.cli("impact", self.store, "--format", "json", ok=False, error="no symlink")

    def test_review_and_asset_unique_metadata_round_trip(self):
        self.cli("init-store", self.store)
        self.cli("dataset-meta", self.store, "set", "000006", "--name", "Motor cortex")
        shutil.copyfile(self.store / "dandisets/000006.json", self.stage / "artifacts/dandiset.json")
        notes = {"session": "20170316", "units_columns": ["quality", "amplitude"]}
        asset = self.summary("--unique-metadata", "@" + str(self.dump("asset-meta.json", notes)),
                             "--evidence-level", "spike_times", "--emit-findings")["asset"]
        review = self.review()
        review["assets"] = [asset | {"pin_verification": "published_version"}]
        review["findings"] = []
        review["unique_metadata"] = {"paper_figures": [1, "S2"]}
        bundle = self.pack(review)
        manifest = json.loads((bundle / "review.json").read_text())
        self.assertIn("dandiset_metadata", [row["kind"] for row in manifest["files"]])
        self.cli("validate", bundle)
        self.cli("validate", self.store)
        self.cli("project", self.store)
        self.assertEqual(json.loads(self.query("SELECT unique_metadata_json FROM review_assets")[0][0]), notes)
        self.assertEqual(json.loads(self.query("SELECT unique_metadata_json FROM reviews")[0][0]),
                         review["unique_metadata"])
        self.cli("summarize-asset", "--asset-id", "a", "--path", "a.nwb", "--size-bytes", 1,
                 "--encoding", "hdf5", "--unique-metadata", "[]", ok=False, error="expected object")
        for owner in (review, review["assets"][0]):
            with patch.dict(owner, unique_metadata=None):
                self.dump("stage/review.json", review)
                self.cli("pack", self.stage, "--store", self.store, ok=False, error="expected object")

    def test_units_only_grid_does_not_invent_recording_metrics(self):
        qc = self.dump("qc.json", [{"cluster_id": 0, "n_spikes": 100, "presence_ratio": 0,
                                  "mean_amplitude": 0, "amplitude_cutoff": .9, "epoch_amp_ratio": 0}])
        for doc, flags in [
            ({"sample rate": "1 kHz", "duration": "0.0 s"}, []),
            ({"sample rate": "1 kHz", "duration": "30 s"}, ["--evidence-level", "spike_times"]),
            ({"sample rate": "1 kHz", "duration": "30 s", "n_samples": "0", "data files": "trace.dat"}, []),
        ]:
            result = self.summary("--describe", self.dump("describe.json", doc), "--sample-rate", "1000",
                                  "--before-qc", qc, "--emit-findings", *flags)
            self.assertEqual(result["asset"]["analysis_sample_rate_hz"], 1000)
            for key in ("sample_rate_hz", "duration_s", "n_low_presence", "n_amp_cutoff", "n_unstable"):
                self.assertIsNone(result["asset"][key])
            self.assertEqual(result["candidate_findings"], [])
        raw = self.summary("--describe", self.dump("describe.json", {
            "sample rate": "30 kHz", "duration": "20 s", "data files": "trace.dat"}), "--before-qc", qc)
        self.assertEqual((raw["sample_rate_hz"], raw["duration_s"], raw["n_low_presence"]), (30000, 20, 1))
        self.assertIsNone(raw["n_amp_cutoff"])

    def test_qc_and_tsv_must_agree_on_live_ids_and_counts_before_and_after(self):
        tsv = self.clusters("clusters.tsv", [(2, "unsorted", "-"), (3, "good", "-")])
        rows = [{"cluster_id": 2, "n_spikes": 100}, {"cluster_id": 3, "n_spikes": 100},
                {"cluster_id": 7, "n_spikes": 0}]
        for suffix in ("before", "after"):
            for data, error in ((rows, None), (rows[1:], "missing QC IDs: [2]"),
                                (rows + [{"cluster_id": 8, "n_spikes": 1}], "extra QC IDs: [8]"),
                                ([dict(rows[0], n_spikes=99), *rows[1:]], "spike-count mismatch")):
                with self.subTest(suffix=suffix, error=error):
                    self.cli("summarize-asset", "--asset-id", "a", "--path", "a.nwb", "--size-bytes", 1,
                             "--encoding", "hdf5", f"--{suffix}-qc", self.dump("qc.json", data),
                             f"--{suffix}-clusters", tsv, ok=error is None, error=error)

    def test_split_remainder_and_revived_merge_target_match_live_tsv(self):
        before = self.clusters("before.tsv", [(0, "good", "-"), (1, "mua", "-"), (2, "noise", "-")])
        after = self.clusters("after.tsv", [(0, "noise", "-"), (3, "mua", "-"), (4, "good", "-")])
        doc = self.dump("curation.json", [
            curation([10, 20, 30, 40], splits=[{"unit_id": 10, "new_unit_ids": [50, 60]}]),
            curation([20, 30, 40, 50, 60], merges=[{"unit_ids": [20, 30, 40], "new_unit_id": 40}]),
            curation([40, 50, 60], [(60, "noise"), (50, "good"), (40, "mua")]),
        ])
        options = ("--before-clusters", before, "--curation", doc, "--emit-findings")
        only = self.summary(*options)
        with_tsv = self.summary(*options, "--after-clusters", after)
        for key in ("units_after", "good_after", "mua_after", "noise_after", "n_relabelled", "n_noise_as_good"):
            self.assertIsNone(only["asset"][key])
        self.assertEqual((only["asset"]["n_merges"], only["asset"]["n_splits"]), (1, 1))
        self.assertEqual(only["candidate_findings"], [])
        self.assertEqual((with_tsv["asset"]["units_after"], with_tsv["asset"]["good_after"]), (3, 1))
        self.assertFalse(with_tsv["candidate_findings"])

        doc = self.dump("rekeyed.json", [
            curation([10, 20, 30], merges=[{"unit_ids": [10, 20], "new_unit_id": 10}]),
            curation([10, 30], merges=[{"unit_ids": [10, 30], "new_unit_id": 20}]),
            curation([20], [(20, "good")]),
        ])
        after = self.clusters("rekeyed.tsv", [(1, "good", "-", 300)])
        options = ("--before-clusters", before, "--curation", doc)
        replayed = self.summary(*options)
        self.assertIsNone(replayed["units_after"])
        with_tsv = self.summary(*options, "--after-clusters", after)
        self.assertEqual((with_tsv["units_after"], with_tsv["good_after"]), (1, 1))
        self.assertIsNone(with_tsv["n_relabelled"])

        no_baseline = self.summary("--curation", doc, "--evidence-level", "spike_times")
        self.assertEqual(no_baseline["n_merges"], 2)
        self.assertIsNone(no_baseline["units_after"])
        qc = self.dump("before.json", [{"cluster_id": i, "n_spikes": 100} for i in range(3)])
        qc_baseline = self.summary("--before-qc", qc, "--curation", doc)
        self.assertIsNone(qc_baseline["units_after"])
        self.assertIsNone(qc_baseline["n_relabelled"])

    def test_baseline_docs_use_a_fresh_journal_and_noclobber(self):
        guide = (kit.HERE / "SKILL.md").read_text()
        self.assertIn('journal="$workdir/sorrel.journal"', guide)
        self.assertIn("set -C", guide)
        self.assertRegex(guide, r"(?s)--discard-journal.*never.*after-curation")
        for name in ("review", "feedback", "dandiset"):
            url = f"https://nexus.beloda.co/sorrel/schema/{name}-1.json"
            self.assertEqual(kit.schema(name)["$id"], url)
            self.assertIn(url, guide)
        self.assertIn("https://sorrel.beloda.co)", guide)
        self.assertIn("https://beloda.co)", guide)

    def test_summarize_real_qc_describe_clusters_and_inspector(self):
        self.assertEqual(len(QC_FIELDS), 41)
        data = [dict.fromkeys(QC_FIELDS) | row for row in (
            dict(cluster_id=0, n_spikes=100, srp_contamination=.14, rpv_post_lo95=.02, presence_ratio=.6,
                 drift_correlation=-.7, mean_amplitude=20, epoch_amp_ratio=.4, composite_quality=.3),
            dict(cluster_id=1, n_spikes=100, rpv_post_lo95=.11, mean_amplitude=20, amplitude_cutoff=.2, composite_quality=.7),
            dict(cluster_id=2, n_spikes=100, srp_contamination=.1, rp_contamination=.9, isi_violation_ratio=.9),
            dict(cluster_id=3, n_spikes=100, srp_contamination=float("nan"), rp_contamination=.2, isi_violation_ratio=.6),
            dict(cluster_id=4, n_spikes=0, rp_contamination=.8),
        )]
        qc = self.dump("before.json", data)
        after = self.dump("after.json", [dict(row, composite_quality=.6 if row["cluster_id"] == 0 else None)
                                        for row in data])
        labels = self.clusters("before.tsv", [(0, "good", 'a "quoted" note'), (1, "mua", "-"),
                                              (2, "unsorted", "-"), (3, "noise", "-")])
        labels_after = self.clusters("after.tsv", [(0, "noise", "-"), (1, "good", "-"),
                                                  (2, "unsorted", "-"), (3, "noise", "-")])
        describe = self.dump("describe.json", {
            "data files": "—", "dtype": "—", "sample rate": "30.0 kHz", "duration": "1200.5 s",
            "# channels": "384", "# clusters": "4 (ids up to 5)", "# spikes": "400",
            "groups": "good 1, mua 1, noise 1, unsorted 1", "# templates": "8 × 61 samples",
            "pc features": "— (FeatureView fits PCs on raw snippets)", "labels": "—", "undo history": "0",
        })
        inspector = self.dump("nwb_inspector.json", {"messages": [
            {"severity": "CRITICAL"}, {"severity": "BEST_PRACTICE_VIOLATION"},
            {"severity": "BEST_PRACTICE_SUGGESTION"}, {"severity": "ERROR"},
        ]})
        result = self.summary("--describe", describe, "--before-qc", qc, "--after-qc", after,
                              "--before-clusters", labels, "--after-clusters", labels_after,
                              "--nwb-inspector", inspector, "--evidence-level", "raw", "--emit-findings")
        asset = result["asset"]
        expected = dict(sample_rate_hz=30000, duration_s=1200.5, n_channels=384, evidence_level="raw",
                        units_before=4, good_before=1, mua_before=1, noise_before=1, unlabeled_before=1,
                        n_rp_contaminated=1, n_good_rp_contaminated=0, n_rp_screened=1, n_good_rp_screened=1,
                        n_isi_rate_flagged=2, n_low_presence=1, n_unstable=1,
                        n_amp_cutoff=1, n_noise_as_good=1, units_after=4, good_after=1, noise_after=2,
                        n_relabelled=2, median_quality_before=.5, median_quality_after=.6,
                        nwbi_critical=1, nwbi_violation=1, nwbi_suggestion=1)
        self.assertEqual({k: asset[k] for k in expected}, expected)
        candidates = {row["code"]: row for row in result["candidate_findings"]}
        self.assertEqual(candidates["qc.drift"]["unit_ids"], [0])
        self.assertEqual(candidates["qc.refractory_contamination"]["unit_ids"], [1])
        screened = {f["metric"]: f for f in result["candidate_findings"] if f["code"] == "qc.refractory_screening"}
        self.assertEqual(screened["isi_violation_ratio"]["unit_ids"], [3])
        self.assertIn({"pointer": "/0/srp_contamination", "artifact_sha256": hashlib.sha256(qc.read_bytes()).hexdigest()},
                      screened["srp_contamination"]["evidence"])
        self.assertTrue(all(not row["human_verified"] for row in candidates.values()))

    def test_summarize_legacy_single_curation_and_refused(self):
        qc = self.dump("legacy.json", [dict.fromkeys(QC_FIELDS[:33]) | dict(
            cluster_id=0, n_spikes=100, isi_violation_ratio=.25)])
        labels = self.clusters("before.tsv", [(0, "good", "-")])
        params = ["--before-qc", qc, "--before-clusters", labels]
        asset = self.summary(*params)
        self.assertIsNone(asset["n_good_rp_contaminated"])
        self.assertIsNone(asset["n_rp_screened"])
        self.assertIsNone(asset.get("n_noise_as_good"))
        self.assertEqual(asset["n_isi_rate_flagged"], 0)
        self.assertIsNone(asset["n_unstable"])
        doc = self.dump("curation.json", curation(["nonlocal-unit-42"], [("nonlocal-unit-42", "noise")]))
        desc = self.dump("old-describe.json", {"sample rate": "30000 Hz", "duration": "20m 0.5s",
                                              "# channels": "384", "data files": "recording.dat"})
        asset = self.summary(*params, "--curation", doc, "--describe", desc)
        self.assertEqual((asset["duration_s"], asset["evidence_level"]), (1200.5, "raw"))
        self.assertEqual((asset["units_after"], asset["noise_after"], asset["n_relabelled"],
                          asset["n_noise_as_good"], asset["n_merges"]), (1, 1, 1, 1, 0))
        result = self.summary("--open-status", "refused", "--refusal-reason", "No Units",
                              "--emit-findings")
        self.assertEqual(result["asset"]["open_status"], "refused")
        self.assertIsNone(result["asset"].get("units_before"))
        self.assertEqual(result["candidate_findings"][0]["code"], "data.sorrel_refused")

    def test_summarize_multistep_curation(self):
        labels = self.clusters("before.tsv", [(0, "good", "-"), (1, "mua", "-"), (2, "noise", "-"),
                                              (3, "good", "-")])
        doc = self.dump("curation.json", [
            curation([40, 42, 90, 99], merges=[{"unit_ids": [42, 40], "new_unit_id": 40}]),
            curation([40, 90, 99], splits=[{"unit_id": 90, "mode": "indices", "indices": [[0]],
                                            "new_unit_ids": [120, 121]}], removed=[99]),
            curation([40, 120, 121], [(40, "noise"), (120, "good"), (121, "mua")]),
        ])
        asset = self.summary("--before-clusters", labels, "--curation", doc)
        self.assertEqual([asset[k] for k in ("n_merges", "n_splits", "n_removed", "n_relabelled",
                                            "n_noise_as_good", "units_after", "good_after", "mua_after", "noise_after")],
                         [1, 1, 1, None, None, None, None, None, None])

    def test_structural_curation_counts_only_relabels_of_tracked_unchanged_units(self):
        before = self.clusters("before.tsv", [(i, "good", "-") for i in range(5)])
        after = self.clusters("after.tsv", [(0, "noise", "-", 200), (1, "noise", "-"),
                                            (2, "noise", "-"), (4, "noise", "-")])
        # Two unrecorded rekeys: 2 -> 1, then 3 -> 2. Slot 2 now has
        # the same spike count but different lineage.
        after.write_text(after.read_text().replace("2\t100\tnoise\t-\tAuto[ks4]\t-",
                                                  "2\t100\tnoise\t-\tAuto[ks4]\t3"))
        doc = self.dump("curation.json", curation([10, 20, 30, 40, 50],
                        merges=[{"unit_ids": [10, 20], "new_unit_id": 10}]))
        result = self.summary("--before-clusters", before, "--after-clusters", after, "--curation", doc,
                              "--emit-findings")
        self.assertEqual((result["asset"]["noise_after"], result["asset"]["n_relabelled"],
                          result["asset"]["n_noise_as_good"]), (4, 1, 1))
        self.assertEqual(result["candidate_findings"][0]["unit_ids"], [4])

    def test_unknown_measurements_and_threshold_override(self):
        labels = self.clusters("before.tsv", [(0, "good", "-", 90), (1, "unsorted", "-", 30)])
        qc = self.dump("qc.json", [dict(cluster_id=0, n_spikes=90),
                                   dict(cluster_id=1, n_spikes=30, rpv_post_lo95=.12)])
        override = self.dump("thresholds.json", {"rp_contamination_lo95_max": .15})
        path = self.root / "summary.json"
        result = self.cli("summarize-asset", "--asset-id", ASSET["asset_id"], "--path", ASSET["path"],
                          "--size-bytes", "123", "--encoding", "hdf5", "--before-clusters", labels,
                          "--before-qc", qc, "--thresholds", "@" + str(override), "--out", path)
        self.assertEqual(result, "")
        asset = json.loads(path.read_text())
        self.assertEqual(asset["n_rp_contaminated"], 0)
        self.assertIsNone(asset["n_good_rp_contaminated"])
        self.assertIsNone(asset["median_quality_before"])
        self.assertEqual(self.summary("--before-qc", qc, "--thresholds", '{"rp_contamination_lo95_max":0.11}')
                         ["n_rp_contaminated"], 1)

    def test_pack_idempotent_dry_run_and_tampering(self):
        path = self.stage / "artifacts" / "cluster_qc.before.json"
        path.write_text("[]\n")
        self.dump("stage/review.json", self.review())
        self.cli("pack", self.stage, "--store", self.store, "--dry-run")
        self.assertFalse(self.store.exists())
        bundle = self.pack()
        self.assertEqual(bundle, self.pack())
        self.cli("validate", bundle)
        review = bundle / "review.json"
        canonical = review.read_bytes()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), bundle.name)
        self.assertEqual([f["kind"] for f in json.loads(canonical)["files"]], ["cluster_qc_before", "report_md"])
        report = bundle / "report.md"
        original = report.read_bytes()
        report.write_bytes(original + b"tampered")
        self.cli("validate", bundle, ok=False, error="report.md")
        report.write_bytes(original)
        review.write_bytes(canonical + b"\n")
        self.cli("validate", review, ok=False, error="canonical")
        review.write_bytes(canonical)
        (bundle / "extra").touch()
        self.cli("validate", bundle, ok=False, error="extra")
        self.cli("pack", self.stage, "--store", self.store, ok=False, error="extra")

    def test_validate_schema_and_references(self):
        for updates, error in [
            ({"findings": [dict(self.review()["findings"][0], code="qc.made_up")]}, "code"),
            ({"dandiset": {"id": "000006", "version": "draft"}}, "draft_modified"),
            ({"mode": "agent"}, "models"),
            ({"supersedes": "f" * 64}, "supersedes"),
            ({"created_at": "2026-02-30T00:00:00Z"}, "created_at"),
            ({"unrecognized": 8}, "unrecognized"),
        ]:
            with self.subTest(error=error):
                self.dump("stage/review.json", self.review() | updates)
                self.cli("pack", self.stage, "--store", self.store, ok=False, error=error)
        self.assertFalse(self.store.exists())

    def test_pack_checks_evidence_and_does_not_follow_symlinks(self):
        report = self.stage / "report.md"
        review = self.review()
        review["findings"][0]["evidence"] = [{"artifact_sha256": "f" * 64, "pointer": "L1"}]
        self.dump("stage/review.json", review)
        self.cli("pack", self.stage, "--store", self.store, ok=False, error="evidence")
        review["findings"][0]["evidence"][0]["artifact_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
        self.dump("stage/review.json", review)
        link = self.stage / "artifacts" / "linked.md"
        link.symlink_to(report)
        self.cli("pack", self.stage, "--store", self.store, ok=False, error="symlink")
        link.unlink()
        bundle = self.pack(review)
        (bundle / "artifacts").symlink_to(self.stage / "artifacts", target_is_directory=True)
        self.cli("validate", bundle, ok=False, error="symlink")

    def test_pack_and_validate_resolve_evidence_and_deliverables(self):
        review = self.review()
        artifact = self.stage / "artifacts" / "data.json"
        artifact.write_text('[{"a/b~": 5}]\n')
        sha = kit.digest_file(artifact)[0]
        report_sha, size = kit.digest_file(self.stage / "report.md")
        review["improvements"] = [{"id": "i01", "kind": "relabel", "addresses": ["f01"],
                                   "description_md": "Fix", "deliverable_sha256": report_sha}]
        evidence = review["findings"][0]["evidence"] = [
            {"artifact_sha256": sha, "pointer": "/0/a~1b~0"},
            {"artifact_sha256": report_sha, "pointer": "L1-L1"},
        ]
        review["files"] = [{"kind": "report_md", "path": "report.md", "sha256": report_sha, "size_bytes": size}]
        self.cli("validate", self.pack(review))
        for key, bad in (("sha256", "f" * 64), ("size_bytes", size + 1)):
            review["files"][0][key] = bad
            self.dump("stage/review.json", review)
            self.cli("pack", self.stage, "--store", self.store, ok=False, error=f"supplied {key}")
            review["files"][0].update(sha256=report_sha, size_bytes=size)
        review["improvements"][0]["deliverable_sha256"] = "f" * 64
        self.dump("stage/review.json", review)
        self.cli("pack", self.stage, "--store", self.store, ok=False, error="deliverable_sha256")
        review["improvements"][0]["deliverable_sha256"] = report_sha
        for ordinal, bad in ((0, "/1/a~1b~0"), (0, "/00/a~1b~0"), (0, "/0/a~2b"), (1, "L1-L2"), (1, "L0")):
            with self.subTest(pointer=bad):
                original = evidence[ordinal]["pointer"]
                evidence[ordinal]["pointer"] = bad
                self.dump("stage/review.json", review)
                self.cli("pack", self.stage, "--store", self.store, "--dry-run", ok=False, error="evidence pointer")
                evidence[ordinal]["pointer"] = original
        # An imported bundle can have a valid canonical review ID yet an invalid pointer.
        source = self.pack(review)
        review = json.loads((source / "review.json").read_text())
        review["findings"][0]["evidence"][0]["pointer"] = "/99"
        canonical = (json.dumps(review, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
        detached = self.root / hashlib.sha256(canonical).hexdigest()
        source.rename(detached)
        (detached / "review.json").write_bytes(canonical)
        self.cli("validate", detached, ok=False, error="evidence pointer")

    def test_validate_rejects_noncanonical_review_even_at_correct_byte_hash(self):
        source = self.pack()
        raw = json.dumps(json.loads((source / "review.json").read_text())).encode()
        target = self.root / hashlib.sha256(raw).hexdigest()
        source.rename(target)
        (target / "review.json").write_bytes(raw)
        self.cli("validate", target, ok=False, error="canonical")

    def test_validate_asset_pin_receipt_locator_and_evidence(self):
        locator = "dandi://dandi/000006@0.220126.1855/" + ASSET["path"]
        bundle = self.pack(self.review() | {"assets": [dict(ASSET, locator=locator)]})
        for updates, missing, error in [
            ({"locator": locator.replace("@0.220126.1855", "@draft")}, None, "locator"),
            ({"locator": locator.replace("@0.220126.1855", "")}, None, "locator"),
            ({"locator": locator.replace("000006", "000007")}, None, "locator"),
            ({"locator": locator + ".wrong"}, None, "locator"),
            ({"pin_verification": "unverified"}, None, "pin_verification"),
            ({}, "pin_verification", "pin_verification"),
            ({}, "evidence_level", "evidence_level"),
            ({"evidence_level": None}, None, "evidence_level"),
            ({"open_status": "failed", "duration_s": 10}, "evidence_level", "duration_s"),
            ({"open_status": "refused", "sample_rate_hz": 10}, None, "sample_rate_hz"),
            ({"open_status": "refused", "n_low_presence": 0}, "evidence_level", "n_low_presence"),
            ({"open_status": "failed", "n_amp_cutoff": 0}, "evidence_level", "n_amp_cutoff"),
        ]:
            with self.subTest(updates=updates, missing=missing):
                def edit(doc):
                    doc["assets"][0].update(updates)
                    doc["assets"][0].pop(missing, None)
                self.cli("validate", self.rewrite_bundle(bundle, edit), ok=False, error=error)

    def test_validate_requires_sorted_file_paths(self):
        (self.stage / "artifacts" / "data.txt").write_text("evidence\n")
        bundle = self.pack()
        tampered = self.rewrite_bundle(bundle, lambda doc: doc["files"].reverse())
        self.cli("validate", tampered, ok=False, error="sorted")

    def test_html_and_other_evidence_pointers_must_resolve(self):
        html = self.stage / "artifacts" / "sorrel_report.html"
        html.write_text('<section id="cluster-0" data-example="id=\'fake-attribute\'">Unit</section>\n'
                        "<div id='cluster-1'></div><!-- <div id=\"fake\"></div> -->\n")
        other = self.stage / "artifacts" / "data.unknown"
        other.write_text("one\ntwo\n")
        review = self.review()
        evidence = review["findings"][0]["evidence"] = [
            {"artifact_sha256": kit.digest_file(html)[0], "pointer": "#cluster-0"},
            {"artifact_sha256": kit.digest_file(other)[0], "pointer": "L1-L2"},
        ]
        bundle = self.pack(review)
        self.cli("validate", bundle)
        for good in ("#cluster-1", "L1-L2"):
            self.cli("validate", self.rewrite_bundle(bundle, lambda doc: doc["findings"][0]["evidence"][0].update(pointer=good)))
        for i, bad in ((0, "#missing-anchor"), (0, "#fake"), (0, "#fake-attribute"), (0, "L3"),
                       (1, "text"), (1, "L3")):
            with self.subTest(pointer=bad):
                with patch.dict(evidence[i], pointer=bad):
                    self.dump("stage/review.json", review)
                    self.cli("pack", self.stage, "--store", self.store, ok=False, error="evidence pointer")
                tampered = self.rewrite_bundle(bundle, lambda doc: doc["findings"][0]["evidence"][i].update(pointer=bad))
                self.cli("validate", tampered, ok=False, error="evidence pointer")

    def test_validate_count_relationships_and_int64_bounds(self):
        base = dict(ASSET, units_before=3, good_before=1, mua_before=1, noise_before=0, unlabeled_before=1,
                    units_after=3, good_after=1, mua_after=1, noise_after=1, n_rp_contaminated=2,
                    n_good_rp_contaminated=1, n_rp_screened=1, n_good_rp_screened=1)
        for updates, error in [
            ({"noise_before": 1}, "units_before"), ({"noise_after": 2}, "units_after"),
            ({"n_good_rp_contaminated": 2}, "good_before"), ({"n_rp_contaminated": 0}, "n_rp_contaminated"),
            ({"n_rp_contaminated": 4}, "units_before"), ({"n_good_rp_screened": 2}, "good_before"),
            ({"n_rp_screened": 0}, "n_rp_screened"), ({"n_rp_screened": 4}, "units_before"),
            ({"n_low_presence": 4}, "n_low_presence"), ({"n_unstable": 4}, "n_unstable"),
            ({"n_amp_cutoff": 4}, "n_amp_cutoff"), ({"n_isi_rate_flagged": 4}, "n_isi_rate_flagged"),
            ({"n_noise_as_good": 2}, "n_noise_as_good"),
            ({"size_bytes": 2**63}, "9223372036854775807"),
        ]:
            with self.subTest(updates=updates):
                self.dump("stage/review.json", self.review() | {"assets": [base | updates]})
                self.cli("pack", self.stage, "--store", self.store, ok=False, error=error)
        self.assertFalse(self.store.exists())
        self.pack(self.review() | {"assets": [base | {"size_bytes": 2**63 - 1}]})

    def test_pack_preflights_projection_before_install(self):
        max_units = dict(ASSET, units_before=2**63 - 1)
        original = self.pack(self.review() | {"assets": [max_units]})
        self.dump("stage/review.json", self.review() | {"assets": [dict(ASSET, units_before=1)]})
        for options in (("--dry-run",), ()):
            self.cli("pack", self.stage, "--store", self.store, *options, ok=False, error="integer overflow")
        self.assertEqual(list(original.parent.iterdir()), [original])
        self.assertEqual(self.query("SELECT units_before FROM v_scientific_impact"), [(2**63 - 1,)])
        self.rewrite_bundle(original, lambda doc: doc["assets"][0].update(units_before=1), original.parent)
        self.cli("validate", self.store, ok=False, error="integer overflow")

    def test_finding_scope_namespace_and_per_review_feedback_join(self):
        review = self.review()
        candidate = review["findings"][0]
        for changes in (
            {"scope": "dandiset"}, {"asset_id": None}, {"asset_id": "not-here"},
            {"scope": "unit"}, {"unit_ids": [0]}, {"scope": "unit", "n_units": 0},
        ):
            with self.subTest(changes=changes):
                self.dump("stage/review.json", review | {"findings": [candidate | changes]})
                self.cli("pack", self.stage, "--store", self.store, ok=False)
        dataset = {k: v for k, v in candidate.items() if k != "asset_id"} | {"scope": "dandiset", "id": "f02"}
        unit = candidate | {"scope": "unit", "unit_ids": [0], "unit_id_namespace": "sorrel_cluster_index"}
        first = self.pack(review | {"findings": [unit, dataset]})
        second = self.pack(review | {"findings": [dataset | {"asset_id": None}]})
        self.cli("feedback", self.store, "record", "--dandiset", "000006", "--actor", "Reviewer",
                 "--status", "sent", "--channel", "email", "--review-id", first.name)
        self.assertEqual(dict((rid, (asset_n, dataset_n, status)) for rid, asset_n, dataset_n, status in
                              self.query("SELECT review_id, n_findings, dandiset_finding_count, feedback_status "
                                         "FROM v_result_table")),
                         {first.name: (1, 1, "sent"), second.name: (0, 1, None)})

    def test_impact_pairwise_denominators_and_null_coverage(self):
        self.cli("init-store", self.store)
        nulls = json.loads(self.cli("impact", self.store, "--format", "json"))["scientific_impact"]
        for metric in ("n_merges", "units_before", "n_good_rp_contaminated", "n_good_rp_screened",
                       "units_corrected", "total_cost_usd"):
            self.assertIsNone(nulls[metric])
        for coverage in ("n_assets_with_units", "n_assets_with_rp_post", "n_assets_with_rp_screen",
                         "n_assets_with_curation", "n_feedback_with_units_corrected", "n_reviews_with_cost"):
            self.assertEqual(nulls[coverage], 0)
        self.assertIn("| units_before | N/A |", self.cli("impact", self.store, "--format", "md"))
        values = ({"good_before": 5, "n_good_rp_contaminated": 1}, {"n_good_rp_contaminated": 3},
                  {"good_before": 7, "n_good_rp_screened": 2}, {"n_good_rp_screened": 4},
                  {"good_before": 100}, {"good_before": 0, "n_good_rp_contaminated": 0})
        self.pack(self.review() | {"asset_selection": {"policy": "all", "n_assets_total": len(values)},
                                   "assets": [dict(ASSET, asset_id=f"asset-{i}", path=f"{i}.nwb") | item
                                              for i, item in enumerate(values, 1)]})
        self.assertEqual(self.query("SELECT n_good_rp_contaminated, pct_good_rp_contaminated, "
                                    "n_good_rp_screened, pct_good_rp_screened FROM v_scientific_impact"),
                         [(4, 20.0, 6, 200/7)])
        coverage = {"n_assets_with_good_rp_contaminated": 1, "good_before_with_rp_contaminated": 5,
                    "n_assets_with_good_rp_screened": 1, "good_before_with_rp_screened": 7}
        impact = json.loads(self.cli("impact", self.store, "--format", "json"))["scientific_impact"]
        for key, count in coverage.items():
            self.assertEqual(impact[key], count)
            self.assertIn(f"{key}\t{count}", self.cli("impact", self.store, "--format", "text"))
            self.assertIn(f"| {key} | {count} |", self.cli("impact", self.store, "--format", "md"))

    def test_project_supersession_and_kpis(self):
        first = self.pack()
        model = {"role": "curation", "provider": "example", "model_id": "cheap", "cost_usd": 2}
        review = self.review() | {"supersedes": first.name, "mode": "hybrid", "models": [model],
                                  "asset_selection": {"policy": "all", "n_assets_total": 2},
                                  "assets": [self.review()["assets"][0],
                                             dict(ASSET, asset_id="asset-2", path="sub-b.nwb", open_status="refused")]}
        second = self.pack(review)
        self.cli("validate", self.store)
        self.cli("project", self.store)
        self.assertEqual(self.query("SELECT review_id FROM v_latest_reviews"), [(second.name,)])
        self.assertEqual(self.query("SELECT review_id, asset_id FROM v_result_table ORDER BY asset_id"),
                         [(second.name, "asset-1"), (second.name, "asset-2")])
        self.assertEqual(self.query("SELECT assets_reviewed, assets_ok, assets_refused, units_before, "
                                    "pct_good_rp_contaminated, total_cost_usd FROM v_scientific_impact"),
                         [(2, 1, 1, 3, 50.0, 2.0)])
        with (self.store / "results.tsv").open() as stream:
            self.assertEqual([row["asset_id"] for row in csv.DictReader(stream, delimiter="\t")],
                             ["asset-1", "asset-2"])
        self.assertTrue({"dandisets", "assets", "reviews", "review_models", "review_assets", "findings",
                         "improvements", "feedback", "feedback_events", "review_artifacts"}.issubset(
            {r[0] for r in self.query("SELECT name FROM sqlite_master WHERE type='table'")}))
        impact = json.loads(self.cli("impact", self.store, "--format", "json"))
        self.assertEqual([(m["n_reviews"], m["n_assets"], m["cost_usd"]) for m in impact["models"]], [(1, 2, 2.0)])

    def test_project_keeps_last_projection_if_source_invalid(self):
        existing = self.pack()
        review = self.review() | {"dandiset": {"id": "000007", "version": "0.220126.1855"},
                                  "supersedes": existing.name}
        self.dump("stage/review.json", review)
        self.cli("pack", self.stage, "--store", self.store, ok=False, error="different Dandiset")
        current_tsv = (self.store / "results.tsv").read_bytes()
        with (existing / "report.md").open("a") as stream:
            stream.write("changed\n")
        self.cli("project", self.store, ok=False, error="hash or size mismatch")
        self.assertEqual(self.query("SELECT review_id FROM v_latest_reviews"), [(existing.name,)])
        self.assertEqual((self.store / "results.tsv").read_bytes(), current_tsv)

    def test_feedback_and_disclosure_clock(self):
        review_id = self.pack().name
        cmd = ("feedback", self.store, "record", "--dandiset", "000006", "--actor", "Reviewer")
        self.cli(*cmd, "--status", "sent", "--channel", "email", "--review-id", review_id,
                 "--quote-finding", f"{review_id}:f01", "--at", "2026-01-01T00:00:00Z")

        def impact(at, bucket):
            result = json.loads(self.cli("impact", self.store, "--format", "json", "--as-of", at))
            self.assertEqual([row["dandiset_id"] for row in result["disclosure_queue"][bucket]], ["000006"])
            return result

        impact("2026-01-01T00:00:00Z", "in_grace_period")
        impact("2026-01-16T00:00:00Z", "reminder_due")
        impact("2026-02-01T00:00:00Z", "publish_eligible_ghosted")
        self.assertEqual(self.query("SELECT bucket FROM v_disclosure_queue"), [("publish_eligible_ghosted",)])
        self.cli(*cmd, "--status", "sent", "--extension-until", "2026-02-15T00:00:00Z",
                 "--extension-reason", "Maintenance", ok=False, error="extension")
        self.cli(*cmd, "--status", "acknowledged", "--at", "2026-01-21T00:00:00Z",
                 "--extension-until", "2026-02-15T00:00:00Z", "--extension-reason", "Maintenance")
        impact("2026-02-01T00:00:00Z", "extended")
        self.cli(*cmd, "--status", "republished", "--at", "2026-02-10T00:00:00Z",
                 "--units-corrected", "2", "--resulting-version", "0.260210.1200")
        final = impact("2026-02-10T00:00:00Z", "published")
        self.assertEqual(final["scientific_impact"]["units_corrected"], 2)
        self.assertEqual(final["scientific_impact"]["n_feedback_republished"], 1)
        self.assertIn("Scientific impact", self.cli("impact", self.store, "--format", "md"))
        self.assertIn("Scientific impact", self.cli("impact", self.store))
        self.assertEqual(self.query("SELECT count(*) FROM feedback_events"), [(3,)])
        self.cli("validate", self.store / "feedback" / "000006.json")

    def test_no_response_publication_grace_and_reminder(self):
        review_id = self.pack().name
        cmd = ("feedback", self.store, "record", "--dandiset", "000006", "--actor", "Reviewer")
        opts = ("--status", "sent", "--channel", "email", "--review-id", review_id,
                "--disclosure-days", "20", "--reminder-after-days", "7")
        self.cli(*cmd, *opts, "--at", "2026-01-01T05:30:00.1+05:30", ok=False, error="whole-second")
        self.cli(*cmd, *opts, "--at", "2026-01-01T05:30:00+05:30")
        t = "2026-01-07T23:59:59Z"
        self.cli(*cmd, "--status", "reminded", "--at", t, ok=False, error="scheduled reminder")
        self.cli(*cmd, "--status", "reminded", "--at", t, "--early-followup", "--note", "Early follow-up")
        feedback_path = self.store / "feedback" / "000006.json"
        self.assertNotIn("reminder_sent_at", json.loads(feedback_path.read_text()))
        self.assertEqual(self.query("SELECT reminder_due_epoch - first_notified_epoch FROM feedback"), [(7 * 86400,)])
        for at, bucket in ((t, "in_grace_period"), ("2026-01-08T00:00:00Z", "reminder_due")):
            row = json.loads(self.cli("impact", self.store, "--as-of", at, "--format", "json"))
            self.assertEqual(len(row["disclosure_queue"][bucket]), 1)
        self.cli(*cmd, "--status", "reminded", "--at", "2026-01-08T00:00:00Z")
        self.cli(*cmd, "--status", "no_response_published", "--at", "2026-01-20T23:59:59Z",
                 ok=False, error="elapsed grace")
        self.cli(*cmd, "--status", "no_response_published", "--at", "2026-01-21T00:00:00Z")
        self.cli(*cmd, "--status", "sent", "--at", "2026-01-22T00:00:00Z", ok=False, error="after")
        record = json.loads(feedback_path.read_text())
        self.assertEqual([e["status"] for e in record["events"]], ["sent", "sent", "reminded", "no_response_published"])
        self.assertTrue(record["events"][1]["early_followup"])
        self.assertEqual(record["published_at"], "2026-01-21T00:00:00Z")
        self.assertEqual([(e["channel"], e["recipient"]) for e in record["events"][:3]], [("email", None)] * 3)
        with patch.dict(record["events"][2], channel="other"):
            feedback_path.write_text(json.dumps(record))
            self.cli("validate", feedback_path, ok=False, error="channel")
        record["events"][1]["at"] = "2026-01-07T23:59:59.9Z"
        feedback_path.write_text(json.dumps(record))
        self.cli("validate", feedback_path, ok=False, error="whole-second")

    def test_notified_subject_frozen_and_new_outreach_has_its_own_clock(self):
        rid = self.pack().name
        other = self.pack(self.review() | {"skill_rev": "another review"}).name
        unverified = self.pack(self.review(verified=False)).name
        cmd = ("feedback", self.store, "record", "--dandiset", "000006", "--actor", "Reviewer")
        self.cli(*cmd, "--status", "sent", "--channel", "email", "--recipient", "A", "--review-id", rid,
                 "--quote-finding", f"{rid}:f01", "--at", "2026-01-01T00:00:00Z")
        self.cli(*cmd, "--status", "sent", "--recipient", "B", ok=False, error="recipient")
        self.cli(*cmd, "--status", "sent", "--channel", "other", ok=False, error="channel")
        self.cli(*cmd, "--status", "drafted", ok=False, error="after")
        self.cli(*cmd, "--status", "sent", "--review-id", other, ok=False, error="subject")
        self.cli(*cmd, "--status", "sent", "--quote-finding", f"{other}:f01", ok=False, error="subject")
        self.cli(*cmd, "--status", "sent", "--feedback-id", "another", "--channel", "email",
                 "--review-id", other, "--at", "2026-01-20T00:00:00Z")
        path = self.store / "feedback" / "000006.json"
        original = json.loads(path.read_text())
        self.assertEqual((original["events"][0]["recipient"], original["events"][0]["channel"]), ("A", "email"))
        self.assertEqual(json.loads((path.parent / "another.json").read_text())["first_notified_at"],
                         "2026-01-20T00:00:00Z")
        self.cli(*cmd, "--status", "acknowledged", "--units-corrected", 2**63,
                 ok=False, error="9223372036854775807")
        ack = {"at": "2026-01-21T00:00:00Z", "status": "acknowledged", "actor": "Importer"}
        draft = dict(ack, status="drafted")
        subject = {"review_ids": [unverified], "quoted_finding_ids": [{"review_id": unverified, "finding_id": "f01"}]}
        variants = [
            (dict(recipient="B"), "recipient"),
            (dict(channel="other"), "channel"),
            (dict(first_notified_at="2026-01-02T00:00:00Z"), "first_notified_at"),
            (dict(status="drafted", events=[*original["events"], draft]), "after"),
            (dict(status="acknowledged", events=[*original["events"], ack], quoted_finding_ids=[]), "subject"),
            (subject | dict(status="acknowledged", events=[dict(original["events"][0], subject=subject), ack]),
             "human_verified"),
        ]
        for changes, error in variants:
            with self.subTest(error=error):
                path.write_text(json.dumps(original | changes))
                self.cli("validate", path, ok=False, error=error)

    def test_impact_always_reads_imported_authoritative_feedback(self):
        rid = self.pack().name
        self.cli("feedback", self.store, "record", "--dandiset", "000006", "--actor", "Reviewer",
                 "--status", "sent", "--channel", "email", "--review-id", rid, "--at", "2026-01-01T00:00:00Z")
        opts = ("impact", self.store, "--as-of", "2026-03-01T00:00:00Z", "--format", "json")
        self.assertEqual(len(json.loads(self.cli(*opts))["disclosure_queue"]["publish_eligible_ghosted"]), 1)
        path = self.store / "feedback" / "000006.json"
        doc = json.loads(path.read_text())
        doc["events"].append({"at": "2026-02-20T09:00:00Z", "status": "acknowledged", "actor": "Importer"})
        path.write_text(json.dumps(dict(doc, status="acknowledged")))
        queue = json.loads(self.cli(*opts))["disclosure_queue"]
        self.assertEqual([row["status"] for row in queue["resolved"]], ["acknowledged"])
        self.assertEqual(queue["publish_eligible_ghosted"], [])
        self.assertEqual(self.query("SELECT status FROM feedback"), [("sent",)])
        path.write_text("{}")
        self.cli(*opts, ok=False, error="missing required")

    def test_feedback_human_verification_gate(self):
        review_id = self.pack(self.review(verified=False)).name
        cmd = ("feedback", self.store, "record", "--dandiset", "000006", "--actor", "Reviewer")
        self.cli(*cmd, "--status", "drafted", "--channel", "email", "--review-id", review_id,
                 "--quote-finding", f"{review_id}:f01")
        self.cli(*cmd, "--status", "sent", ok=False, error="human_verified")
        feedback = self.store / "feedback" / "000006.json"
        self.assertEqual(json.loads(feedback.read_text())["status"], "drafted")
        doc = json.loads(feedback.read_text())
        doc["quoted_finding_ids"][0]["finding_id"] = "f77"
        feedback.write_text(json.dumps(doc))
        self.cli("validate", feedback, ok=False, error="f77")

    def test_pin_pagination_encoding_draft_and_embargo(self):
        base = "https://mock.example/api"
        item = {"version": "0.220126.1855", "name": "Example"}
        dataset = {"embargo_status": "OPEN", "most_recent_published_version": item,
                   "draft_version": {"version": "draft", "modified": "2026-01-01T00:00:00Z"}}
        listing = f"{base}/dandisets/000006/versions/{item['version']}/assets/?metadata=true"
        next_page = listing + "&page=2"
        pages = {
            base + "/dandisets/000006/": dataset,
            listing: {"results": [{"asset_id": "asset-1", "path": "sub-a a.nwb", "size": 1024,
                                   "metadata": {"digest": {"dandi:sha2-256": "a" * 64}}}], "next": next_page},
            next_page: {"results": [{"asset_id": "asset-2", "path": "sub-b.nwb.zarr", "size": 2048,
                                     "metadata": {"encodingFormat": "application/x-zarr",
                                                  "digest": {"dandi:dandi-zarr-checksum": "sum"}}},
                                    {"asset_id": "video", "path": "video.mp4", "size": 6,
                                     "metadata": {"encodingFormat": "video/mp4"}},
                                    {"asset_id": "ome", "path": "volume.ome.zarr", "size": 6},
                                    {"asset_id": "aux", "path": "aux.h5", "size": 6},
                                    {"asset_id": "mime-nwb", "path": "typed.bin", "size": 6,
                                     "metadata": {"encodingFormat": "application/x-nwb"}}], "next": None},
            f"{base}/dandisets/000006/versions/draft/assets/?metadata=true": {"results": [], "next": None},
        }
        out = self.root / "pin.json"
        with patch.object(kit, "api_json", side_effect=pages.__getitem__) as fetch:
            self.cli("pin", "000006", "--api-base", base, "--out", out)
            manifest = json.loads(out.read_text())
            self.assertEqual(fetch.call_count, 3)
            self.assertEqual((manifest["n_assets_total"], manifest["n_nwb_assets"]), (6, 3))
            self.assertEqual([(a["asset_id"], a["encoding"], a["is_nwb"]) for a in manifest["assets"]],
                             [("asset-1", "hdf5", True), ("asset-2", "zarr", True), ("mime-nwb", "hdf5", True)])
            all_assets = json.loads(self.cli("pin", "000006", "--api-base", base, "--no-nwb-only"))
            sidecars = [a for a in all_assets["assets"] if not a["is_nwb"]]
            self.assertEqual([(a["asset_id"], a["encoding"]) for a in sidecars],
                             [("aux", "hdf5"), ("video", "other"), ("ome", "zarr")])
            self.assertEqual(manifest["assets"][0]["sha256"], "a" * 64)
            self.assertTrue(manifest["assets"][0]["locator"].endswith("/sub-a%20a.nwb"))
            self.assertEqual(manifest["assets"][1]["zarr_checksum"], "sum")
            asset = json.loads(self.cli("summarize-asset", "--manifest", out, "--asset-path", "sub-b.nwb.zarr",
                                        "--evidence-level", "spike_times"))
            self.assertEqual(asset["asset_id"], "asset-2")
            review = self.review() | {"findings": [], "asset_selection": {
                "policy": "all", "scope": "nwb", "n_assets_total": 6, "n_nwb_assets": 3}}
            review["assets"] = [dict(a, open_status="ok", evidence_level="spike_times") for a in manifest["assets"]]
            self.pack(review)
            self.assertEqual(self.query("SELECT DISTINCT selection_scope, n_assets_total, n_nwb_assets FROM reviews"),
                             [("nwb", 6, 3)])
            review["asset_selection"]["scope"] = "archive"
            self.dump("stage/review.json", review)
            self.cli("pack", self.stage, "--store", self.store, ok=False, error="policy all")
            review["assets"].extend(dict(a, open_status="refused") for a in sidecars)
            self.pack(review)
            self.assertEqual(self.query("SELECT DISTINCT encoding, media_type FROM review_assets WHERE asset_id='video'"),
                             [("other", "video/mp4")])
            dataset["most_recent_published_version"] = None
            self.cli("pin", "000006", "--api-base", base, ok=False, error="--allow-draft")
            draft = json.loads(self.cli("pin", "000006", "--api-base", base, "--allow-draft"))
            self.assertEqual(draft["dandiset"]["draft_modified"], dataset["draft_version"]["modified"])
            self.cli("pin", "000006", "--api-base", base, "--version", "draft")
            dataset["embargo_status"] = "EMBARGOED"
            fetch.reset_mock()
            self.cli("pin", "000006", "--api-base", base, ok=False, error="EMBARGOED")
            self.assertEqual(fetch.call_count, 1)

    def test_draft_requires_local_checksum_or_unchanged_api_postcheck(self):
        base = "https://mock.example/api"
        local = self.root / "local.nwb"
        local.write_bytes(b"original")
        sha = hashlib.sha256(local.read_bytes()).hexdigest()
        item = {"asset_id": "asset-1", "path": ASSET["path"], "size": 8, "blob": "blob-1",
                "metadata": {"digest": {"dandi:sha2-256": sha}}}
        zarr = {"asset_id": "asset-2", "path": "sub-z.nwb.zarr", "size": 9, "zarr": "zarr-2",
                "metadata": {"digest": {"dandi:dandi-zarr-checksum": "sum"}}}
        current = f"{base}/dandisets/000006/"
        dataset = {"embargo_status": "OPEN", "draft_version": {"modified": "2026-01-01T00:00:00.123Z"}}
        pages = {current: dataset, current + "versions/draft/assets/?metadata=true": {"results": [item, zarr]}}
        for entry in (item, zarr):
            pages[current + "versions/draft/assets/?metadata=true&path=" + kit.quote(entry["path"], safe="")] = {
                "results": [entry]}
        path = self.root / "pin.json"
        with patch.object(kit, "api_json", side_effect=pages.__getitem__):
            self.cli("pin", "000006", "--version", "draft", "--api-base", base, "--out", path)
            pin = json.loads(path.read_text())
            asset = pin["assets"][0]
            self.assertEqual(asset["download_url"], f"{kit.API_BASE}/assets/asset-1/download/")
            self.assertEqual((asset["blob_id"], asset["pin_verification"]), ("blob-1", "unverified"))
            self.assertEqual(pin["assets"][1]["zarr_id"], "zarr-2")
            review = self.review()
            review["dandiset"] = pin["dandiset"]
            review["assets"] = [dict(asset, open_status="ok", evidence_level="spike_times")]
            self.dump("stage/review.json", review)
            self.cli("pack", self.stage, "--store", self.store, ok=False, error="pin_verification")
            opts = ("verify-pin", "--manifest", path, "--asset-path", ASSET["path"])
            local.write_bytes(b"modified")
            self.cli(*opts, "--local-file", local, ok=False, error="SHA-256")
            local.write_bytes(b"original")
            self.cli(*opts, "--local-file", local)
            checked = json.loads(self.cli("summarize-asset", "--manifest", path, "--asset-path", ASSET["path"],
                                          "--evidence-level", "spike_times"))
            self.assertEqual(checked["pin_verification"], "local_checksum")
            review["assets"] = [checked]
            self.cli("validate", self.pack(review))
            zarr_opts = ("verify-pin", "--manifest", path, "--asset-path", zarr["path"], "--api-base", base)
            zarr["metadata"]["digest"]["dandi:dandi-zarr-checksum"] = "changed"
            self.cli(*zarr_opts, ok=False, error="zarr_checksum")
            zarr["metadata"]["digest"]["dandi:dandi-zarr-checksum"] = "sum"
            item["asset_id"] = "replaced"
            self.cli(*opts, "--api-base", base, ok=False, error="asset_id")
            self.cli(*zarr_opts)
            self.assertEqual(json.loads(path.read_text())["assets"][1]["pin_verification"], "api_postcheck")

    def test_pin_rejects_untrusted_pagination(self):
        base = "https://mock.example/api"
        replies = [
            {"embargo_status": "OPEN", "most_recent_published_version": {"version": "0.220126.1855"}},
            {"results": [], "next": "https://elsewhere.example/api/dandisets/000006/versions/0.220126.1855/assets/"},
        ]
        with patch.object(kit, "api_json", side_effect=replies) as fetch:
            self.cli("pin", "000006", "--api-base", base, ok=False, error="unsafe DANDI pagination")
            self.assertEqual(fetch.call_count, 2)

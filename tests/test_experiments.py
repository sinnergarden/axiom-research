"""Record lineage, concurrency and readonly behavior independent of executors."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from axiom_research import (ArtifactRef, ExperimentReader, ExperimentStore,
                            ExperimentRecordError, RevisionConflict)


def digest(char):
    return "sha256:" + char * 64


def ref(kind, char, uri="fixture:synthetic"):
    return ArtifactRef(artifact_type=kind, artifact_id=digest(char),
        artifact_contract_version="synthetic_v1", content_digest=digest(char), uri=uri)


SIGNAL = ref("SignalRun", "a")
FEATURE = ref("FeatureBuild", "b")
SNAPSHOT = ref("DataSnapshot", "c")
BACKTEST = {"run_id": digest("d"), "content_digest": digest("e"),
            "signal_ref": SIGNAL.artifact_id, "committed_sequence": 20,
            "uri": "fixture:synthetic-backtest"}
EVALUATION = {"evaluation_ref": digest("f"), "evaluation_content_digest": digest("1"),
              "input_run_ref": {k: BACKTEST[k] for k in
                               ("run_id", "content_digest", "committed_sequence")},
              "uri": "fixture:synthetic-evaluation"}


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "records" / "index.json"
        self.store = ExperimentStore(self.path)
        self.reader = ExperimentReader(self.path)
        self.question = self.store.create_question(question_id="momentum", title="ETF baseline",
            description="Synthetic registration, not a return claim", hypothesis="Explicit hypothesis")
        self.version = self.version_record()

    def version_record(self, **changes):
        args = dict(question_id="momentum", label="baseline", explanation="Fixed rule",
                    parameters={"window": 20, "filter": None}, input_refs=[SNAPSHOT],
                    explicit_changes=["Initial fixed baseline"])
        args.update(changes)
        return self.store.create_version(**args)

    def run_record(self, **changes):
        args = dict(question_id="momentum", version_ref=self.version["version_ref"],
                    status="COMPLETE", output_refs=[FEATURE, SIGNAL], backtest_ref=BACKTEST,
                    evaluation_ref=EVALUATION)
        args.update(changes)
        return self.store.record_run(**args)

    def test_registration_is_idempotent_and_refs_bind_version_and_owner_identity(self):
        first = self.run_record()
        before = self.path.read_bytes(), self.path.stat().st_mtime_ns
        self.assertEqual(first, self.run_record())
        self.assertEqual(before, (self.path.read_bytes(), self.path.stat().st_mtime_ns))
        relocated = deepcopy(BACKTEST)
        relocated["uri"] = "fixture:relocated"
        self.assertEqual(first, self.run_record(backtest_ref=relocated))
        child = self.version_record(parent_version_ref=self.version["version_ref"],
                                    label="explanation change", explicit_changes=["Clarify evidence"])
        another = self.run_record(version_ref=child["version_ref"])
        self.assertNotEqual(first["run_record_ref"], another["run_record_ref"])
        self.assertEqual(another["backtest_ref"], BACKTEST)
        self.assertEqual(another["evaluation_ref"], EVALUATION)
        self.assertEqual(self.reader.detail("momentum")["questions"][0]["question"], self.question)

    def test_organization_filters_and_history_preserve_business_state(self):
        self.run_record()
        failed = self.run_record(status="FAILED", output_refs=[], backtest_ref=None,
                          evaluation_ref=None, reason="Synthetic missing input", outcome=None)
        state = self.store.update_organization("momentum", expected_revision=0,
                  groups=["ETF"], tags=["baseline", "fixed"], favorite=True, shelved=False)
        self.assertEqual(state["revision"], 1)
        self.assertEqual(len(self.reader.index(group="ETF", tags=["fixed"], favorite=True,
                                             status="FAILED")["questions"]), 1)
        self.assertEqual(self.reader.index(tags=["missing"])["questions"], [])
        self.assertEqual(self.reader.index(status="FAILED")["questions"][0]["runs"], [failed])
        self.assertEqual(self.reader.index(shelved=True)["questions"], [])
        self.store.update_organization("momentum", expected_revision=1,
            groups=["ETF"], tags=["baseline"], favorite=False, shelved=True)
        saved = json.loads(self.path.read_text())
        self.assertEqual([r["revision"] for r in saved["organizations"]["momentum"]], [0, 1, 2])
        before = self.path.read_bytes()
        with self.assertRaises(RevisionConflict):
            self.store.update_organization("momentum", expected_revision=1,
                groups=[], tags=[], favorite=True, shelved=False)
        self.assertEqual(before, self.path.read_bytes())

    def test_concurrent_cas_accepts_one_writer_without_lost_update(self):
        def update(tag):
            try:
                ExperimentStore(self.path).update_organization("momentum", expected_revision=0,
                    groups=[], tags=[tag], favorite=False, shelved=False)
                return "saved"
            except RevisionConflict:
                return "conflict"
        with ThreadPoolExecutor(max_workers=2) as workers:
            outcomes = list(workers.map(update, ["first", "second"]))
        self.assertCountEqual(outcomes, ["saved", "conflict"])
        self.assertEqual(self.reader.detail("momentum")["questions"][0]["organization"]["revision"], 1)

    def test_readonly_projection_does_not_scan_or_execute_and_leaves_hash_mtime(self):
        self.run_record()
        child = self.version_record(parent_version_ref=self.version["version_ref"],
            parameters={"window": 20, "filter": None, "new": None}, label="clarified",
            explicit_changes=["Add explicit absent-setting marker"])
        before = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                  for p in self.root.rglob("*") if p.is_file()}
        with patch.object(Path, "rglob", side_effect=AssertionError("Artifact scan")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("Readonly mkdir")), \
             patch("axiom_research.rotation.build_rotation_experiment",
                   side_effect=AssertionError("Feature execution")), \
             patch("axiom_engine.runtime.run_backtest", side_effect=AssertionError("Backtest")):
            self.reader.index()
            self.reader.detail("momentum")
            difference = self.reader.compare_versions(self.version["version_ref"], child["version_ref"])
        after = {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(difference["parameter_changes"], [
            {"path": "/new", "before_present": False, "after_present": True,
             "before": None, "after": None}])
        self.assertEqual(difference["explicit_changes"], ["Add explicit absent-setting marker"])

    def test_missing_reader_does_not_initialize_and_unrun_version_is_visible(self):
        missing = self.root / "absent" / "index.json"
        reader = ExperimentReader(missing)
        self.assertFalse(missing.parent.exists())
        with self.assertRaises(FileNotFoundError):
            reader.index()
        self.assertFalse(missing.parent.exists())
        self.assertEqual(len(self.reader.index(version_ref=self.version["version_ref"])["questions"]), 1)
        self.assertEqual(self.reader.index(status="COMPLETE")["questions"], [])

    def test_cold_public_reader_import_has_no_data_or_core_runtime_import(self):
        code = """
import builtins,sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('axiom_data','axiom_engine'):
        raise AssertionError('Readonly import crossed runtime boundary: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from axiom_research import ExperimentReader
print(ExperimentReader(sys.argv[1]).index()['contract_version'])
"""
        completed = subprocess.run([sys.executable, "-c", code, str(self.path)],
                                   text=True, capture_output=True, check=True)
        self.assertEqual(completed.stdout.strip(), "experiment_projection_v1")

    def test_public_synthetic_index_matches_saved_projection_shape(self):
        examples = Path(__file__).parents[1] / "examples"
        result = ExperimentReader(examples / "experiment_index.synthetic.json").index()
        self.assertEqual(result, json.loads((examples / "experiment_projection.synthetic.json").read_text()))

    def test_atomic_registration_rejects_invalid_run_or_parent_without_partial_records(self):
        question = dict(question_id="batch", title="Batch", description="", hypothesis="")
        version = dict(label="baseline", explanation="Fixed",
                       parameters={}, input_refs=[SNAPSHOT])
        run = dict(status="COMPLETE", output_refs=[SIGNAL])
        target = self.root / "batch" / "index.json"
        store = ExperimentStore(target)
        for wrong_version, wrong_run in (
            (version, {"status": "FAILED", "output_refs": []}),
            ({**version, "parent_version_ref": digest("9")}, run),
        ):
            with self.assertRaises(ExperimentRecordError):
                store.register_saved_experiment(question=question, version=wrong_version, run=wrong_run)
            self.assertFalse(target.exists())
        first = store.register_saved_experiment(question=question, version=version, run=run)
        before = target.read_bytes(), target.stat().st_mtime_ns
        self.assertEqual(first, store.register_saved_experiment(question=question, version=version, run=run))
        self.assertEqual(before, (target.read_bytes(), target.stat().st_mtime_ns))
        self.assertEqual(ExperimentReader(target).index()["store_revision"], 1)
        with self.assertRaises(ExperimentRecordError):
            store.register_saved_experiment(question={**question, "question_id": "new"},
                version={**version, "parent_version_ref": first["version"]["version_ref"]}, run=run)
        self.assertEqual(before, (target.read_bytes(), target.stat().st_mtime_ns))

    def test_cross_question_parent_run_and_comparison_are_rejected(self):
        self.store.create_question(question_id="other", title="Other", description="", hypothesis="")
        before = self.path.read_bytes()
        for action in (
            lambda: self.version_record(question_id="other",
                parent_version_ref=self.version["version_ref"]),
            lambda: self.run_record(question_id="other"),
            lambda: self.run_record(question_id="unknown"),
        ):
            with self.assertRaises(ExperimentRecordError):
                action()
            self.assertEqual(before, self.path.read_bytes())
        other = self.version_record(question_id="other")
        with self.assertRaises(ExperimentRecordError):
            self.reader.compare_versions(self.version["version_ref"], other["version_ref"])

    def test_wrong_signal_evaluation_digest_or_watermark_cannot_cross_bind(self):
        before = self.path.read_bytes()
        wrong_signal = deepcopy(BACKTEST)
        wrong_signal["signal_ref"] = digest("2")
        wrong_digest = deepcopy(EVALUATION)
        wrong_digest["input_run_ref"]["content_digest"] = digest("3")
        wrong_sequence = deepcopy(EVALUATION)
        wrong_sequence["input_run_ref"]["committed_sequence"] += 1
        invalid_sequence = deepcopy(EVALUATION)
        invalid_sequence["input_run_ref"]["committed_sequence"] = True
        for changes in ({"backtest_ref": wrong_signal}, {"evaluation_ref": wrong_digest},
                        {"evaluation_ref": wrong_sequence}, {"evaluation_ref": invalid_sequence},
                        {"backtest_ref": None}):
            with self.assertRaises(ExperimentRecordError):
                self.run_record(**changes)
            self.assertEqual(before, self.path.read_bytes())

    def test_invalid_values_and_metadata_tampering_are_rejected(self):
        for action in (
            lambda: self.run_record(status="FAILED", reason=None),
            lambda: self.run_record(backtest_ref=[]),
            lambda: self.run_record(status="COMPLETE", output_refs=[], backtest_ref=None, evaluation_ref=None),
            lambda: self.version_record(parameters={"bad": float("nan")}),
            lambda: self.version_record(parameters={1: "not a JSON object key"}),
            lambda: self.version_record(input_refs=[SNAPSHOT, SNAPSHOT]),
            lambda: self.store.create_question(question_id="momentum", title="Rewritten",
                description="", hypothesis=""),
            lambda: self.store.update_organization("momentum", expected_revision=True,
                groups=[], tags=[], favorite=True, shelved=False),
        ):
            with self.assertRaises(ExperimentRecordError):
                action()
        original = self.path.read_text()
        saved = json.loads(original)
        saved["questions"]["momentum"]["hypothesis"] = "Tampered"
        self.path.write_text(json.dumps(saved))
        with self.assertRaises(ExperimentRecordError):
            self.reader.index()
        self.path.write_text('{"contract_version":"experiment_index_v1","contract_version":"duplicate"}')
        with self.assertRaises(ExperimentRecordError):
            self.reader.index()


if __name__ == "__main__":
    unittest.main()

"""Saved projections: source windows, receipt attribution and isolated loading."""
from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from axiom_research.stock_artifacts import _read, digest, file_digest, write_json
from axiom_research.stock_stage_report import export_stock_stage_report, load_stock_stage_report, _identity
from test_stock_ml_review import saved_fixture, seal


def fixture(path, *, mutate=None):
    saved_fixture(path)
    values = {n: _read(path / n) for n in ("features.json", "labels.json", "dataset.json", "model.json",
              "predictions.json", "signal-evidence.json", "experiment.json")}
    config = {"fit_cutoff": "2023-12-28T12:30:00Z", "feature_sessions": ["2023-12-20", "2023-12-21", "2023-12-28", "2023-12-29"],
              "cutoff_by_session": {s: s + "T20:30:00+08:00" for s in ["2023-12-20", "2023-12-21", "2023-12-28", "2023-12-29"]}}
    values["experiment.json"]["definition"] = {"config": config, "input_reuse": {"experiment_ref": "sha256:" + "a" * 64,
                                            "feature_ref": values["features.json"]["feature_ref"]}}
    values["dataset.json"].update(training_keys=[["A", "2023-12-20"], ["A", "2023-12-21"], ["B", "2023-12-21"]],
                                  training_row_count=3, excluded={"IMMATURE": 2})
    values["predictions.json"].update(score_semantics="normalized_prediction", rows=[{"security_id": "A", "session": "2023-12-29", "valid": True, "score": 0.1,
        "knowledge_cutoff": "2023-12-29T20:30:00+08:00", "available_at": "2023-12-29T20:00:00+08:00"},
        {"security_id": "B", "session": "2023-12-29", "valid": False, "score": None},
        {"security_id": "A", "session": "2024-01-02", "valid": True, "score": 0.2},
        {"security_id": "B", "session": "2024-01-02", "valid": True, "score": 0.3}])
    values["signal-evidence.json"].update(evaluation_cutoff="2024-01-10T20:30:00+08:00",
        score_semantics="normalized_prediction", label_semantics="raw_return", minimum_pairs=20, rank_ties="average",
        series=[{"session": "2023-12-29", "ic": 0.2, "rank_ic": None, "valid_pair_count": 1,
                 "excluded_pair_count": 1, "prediction_valid_count": 1},
                {"session": "2024-01-02", "ic": 0.6, "rank_ic": 0.8, "valid_pair_count": 2,
                 "excluded_pair_count": 0, "prediction_valid_count": 2}])
    if mutate:
        mutate(values)
    for name, key in (("dataset.json", "dataset_ref"), ("model.json", "model_ref"),
                      ("predictions.json", "signal_run_ref"), ("signal-evidence.json", "evidence_ref")):
        if name == "model.json": values[name]["dataset_ref"] = values["dataset.json"]["dataset_ref"]
        if name == "predictions.json": values[name]["model_ref"] = values["model.json"]["model_ref"]
        if name == "signal-evidence.json": values[name]["signal_ref"] = values["predictions.json"]["signal_run_ref"]
        values[name].pop(key); values[name] = seal(values[name], key)
    experiment = values["experiment.json"]
    for name, key in (("dataset.json", "dataset_ref"), ("model.json", "model_ref"),
                      ("predictions.json", "signal_run_ref"), ("signal-evidence.json", "evidence_ref")):
        experiment[key] = values[name][key]
    experiment.pop("experiment_ref"); values["experiment.json"] = seal(experiment, "experiment_ref")
    for name, value in values.items(): write_json(path / name, value)
    manifest = _read(path / "manifest.json")
    manifest["files"] = {n: file_digest(path / n) for n in manifest["files"]}
    manifest["experiment_ref"] = values["experiment.json"]["experiment_ref"]
    write_json(path / "manifest.json", manifest)
    return values["experiment.json"]


class StockStageReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.saved = self.root / "saved"; self.saved.mkdir()
        self.experiment = fixture(self.saved)
        self.output = self.root / "stage.json"

    def export(self, receipts=()):
        return export_stock_stage_report(self.saved, timing_receipts=receipts, destination=self.output)

    def test_actual_window_aware_declared_window_and_separate_null_means(self):
        result = self.export()
        self.assertEqual(result["training"]["declared"]["last_feature_session"], "2023-12-28")
        self.assertEqual(result["training"]["actual"], {"first_feature_session": "2023-12-20",
            "last_feature_session": "2023-12-21", "session_count": 2, "training_row_count": 3})
        summary = result["signal_summary"]
        self.assertEqual(summary["ic"], {"session_count": 2, "mean": 0.4})
        self.assertEqual(summary["rank_ic"], {"session_count": 1, "mean": 0.8})
        self.assertEqual((summary["prediction_row_count"], summary["prediction_valid_row_count"],
                          summary["valid_pair_count"], summary["excluded_pair_count"]), (4, 3, 3, 1))
        self.assertTrue(all(r["status"] == "NOT_PROVIDED" and r["seconds"] is None for r in result["measurements"]))

    def test_empty_and_all_null_statistics(self):
        for series in ([], [{"session": "2023-12-29", "ic": None, "rank_ic": None,
                            "prediction_valid_count": 1, "valid_pair_count": 1, "excluded_pair_count": 1}]):
            def mutate(v):
                v["signal-evidence.json"].update(series=series)
                v["predictions.json"]["rows"] = v["predictions.json"]["rows"][:2] if series else []
            fixture(self.saved, mutate=mutate)
            output = self.root / ("empty" + str(len(series)) + ".json")
            result = export_stock_stage_report(self.saved, destination=output)
            for name in ("ic", "rank_ic"): self.assertEqual(result["signal_summary"][name], {"session_count": 0, "mean": None})

    def test_receipts_current_reused_zero_and_inherited_feature_only(self):
        current, old = self.root / "current.json", self.root / "old.json"
        write_json(current, {"experiment": self.experiment, "metrics": {"feature_cache_hit": True,
            "feature_seconds": 0, "qlib_seconds": 0, "train_seconds": 0.05, "build_seconds": 12, "training_rows": 3},
            "cache_reuse": {"seconds": 3.48}})
        write_json(old, {"experiment_ref": self.experiment["definition"]["input_reuse"]["experiment_ref"],
                        "metrics": {"feature_seconds": 1020, "qlib_seconds": 5, "train_seconds": 0.9, "total_seconds": 1046}})
        result = self.export([current, old]); rows = result["measurements"]
        skipped = [r for r in rows if r["status"] == "REUSED_NOT_EXECUTED"]
        self.assertEqual({r["stage"] for r in skipped}, {"feature", "qlib"})
        self.assertTrue(all(r["seconds"] is None and r["reported_seconds"] == 0 for r in skipped))
        inherited = [r for r in rows if r["inherited_feature_only"]]
        self.assertEqual({r["stage"] for r in inherited}, {"feature", "qlib"})
        self.assertTrue(all(r["mode"] == "cold_build" and r["receipt_file_digest"] == file_digest(old) for r in inherited))
        self.assertEqual(next(r for r in rows if r["stage"] == "cache_load")["seconds"], 3.48)
        self.assertEqual(next(r for r in rows if r["stage"] == "total")["seconds"], None)

    def test_wrong_receipt_refs_are_rejected_without_following_uri(self):
        receipt = self.root / "wrong.json"
        write_json(receipt, {"experiment_ref": "sha256:" + "f" * 64, "experiment_path": "/never/read", "metrics": {}})
        with self.assertRaisesRegex(ValueError, "input_reuse"): self.export([receipt])
        wrong = deepcopy(self.experiment); wrong["model_ref"] = "sha256:" + "f" * 64
        write_json(receipt, {"experiment": wrong, "metrics": {}})
        with self.assertRaisesRegex(ValueError, "current experiment"): self.export([receipt])

    def test_flat_current_receipt_respects_current_input_reuse_mode(self):
        receipt = self.root / "flat.json"
        write_json(receipt, {"experiment_ref": self.experiment["experiment_ref"], "metrics": {"train_seconds": 0.05}})
        result = self.export([receipt])
        self.assertEqual(next(r for r in result["measurements"] if r["stage"] == "train")["mode"], "saved_input_build")
        self.output.unlink()
        self.experiment = fixture(self.saved, mutate=lambda v: v["experiment.json"]["definition"].update(input_reuse=None))
        write_json(receipt, {"experiment_ref": self.experiment["experiment_ref"], "metrics": {"train_seconds": 0.05}})
        result = self.export([receipt])
        self.assertEqual(next(r for r in result["measurements"] if r["stage"] == "train")["mode"], "cold_build")

    def test_cache_hit_build_receipts_and_ambiguous_skips_are_rejected(self):
        receipt = self.root / "cache.json"
        sources = [self.experiment["experiment_ref"], self.experiment["definition"]["input_reuse"]["experiment_ref"]]
        for source in sources:
            write_json(receipt, {"experiment_ref": source, "metrics": {"cache_hit": True,
                "feature_seconds": 0, "qlib_seconds": 0, "train_seconds": 0, "predict_seconds": 0,
                "train_calls": 0, "predict_calls": 0}})
            with self.assertRaisesRegex(ValueError, "cache-hit receipt"): self.export([receipt])
        for metrics in [{"feature_cache_hit": True, "feature_seconds": 0, "feature_core_calls": 1},
                        {"feature_seconds": 0, "feature_core_calls": 0}, {"train_seconds": 0, "train_calls": 0}]:
            write_json(receipt, {"experiment": self.experiment, "metrics": metrics})
            with self.assertRaises(ValueError): self.export([receipt])
        self.experiment = fixture(self.saved, mutate=lambda v: v["experiment.json"]["definition"].update(input_reuse=None))
        write_json(receipt, {"experiment_ref": self.experiment["experiment_ref"], "metrics": {"cache_hit": True,
            "feature_seconds": 0, "qlib_seconds": 0, "train_seconds": 0, "predict_seconds": 0,
            "train_calls": 0, "predict_calls": 0}})
        with self.assertRaisesRegex(ValueError, "cache-hit receipt"): self.export([receipt])

    def test_real_zero_with_execution_and_feature_only_reuse_remain_legal(self):
        receipt = self.root / "zero.json"
        write_json(receipt, {"experiment": self.experiment, "metrics": {"cache_hit": False,
            "feature_cache_hit": True, "feature_core_calls": 0, "feature_seconds": 0, "qlib_seconds": 0,
            "train_seconds": 0, "train_calls": 1, "predict_seconds": 0}, "cache_reuse": {"seconds": 3}})
        result = self.export([receipt])
        self.assertEqual(next(r for r in result["measurements"] if r["stage"] == "train")["status"], "MEASURED")
        self.assertEqual(next(r for r in result["measurements"] if r["stage"] == "predict")["status"], "MEASURED")
        self.assertEqual(next(r for r in result["measurements"] if r["stage"] == "feature")["status"], "REUSED_NOT_EXECUTED")
        self.assertEqual(next(r for r in result["measurements"] if r["stage"] == "cache_load")["seconds"], 3)
        self.output.unlink()
        self.experiment = fixture(self.saved, mutate=lambda v: v["experiment.json"]["definition"].update(input_reuse=None))
        write_json(receipt, {"experiment_ref": self.experiment["experiment_ref"], "metrics": {"cache_hit": False,
            "feature_seconds": 0, "feature_core_calls": 1, "qlib_seconds": 0, "qlib_calls": 1,
            "train_seconds": 0, "train_calls": 1}})
        result = self.export([receipt])
        for stage in ("feature", "qlib", "train"):
            row = next(r for r in result["measurements"] if r["stage"] == stage)
            self.assertEqual((row["mode"], row["status"], row["seconds"]), ("cold_build", "MEASURED", 0))

    def test_receipt_parse_and_hash_share_one_byte_snapshot(self):
        receipt = self.root / "changing.json"
        value = {"experiment": self.experiment, "metrics": {"train_seconds": 1}}
        write_json(receipt, value); original_bytes = receipt.read_bytes()
        original_read = Path.read_bytes
        reads = []
        def changing_read(path):
            result = original_read(path)
            if path == receipt:
                reads.append(path)
                write_json(receipt, {"experiment": self.experiment, "metrics": {"train_seconds": 999}})
            return result
        with patch.object(Path, "read_bytes", changing_read):
            result = self.export([receipt])
        row = next(r for r in result["measurements"] if r["stage"] == "train")
        self.assertEqual(len(reads), 1)
        self.assertEqual(row["seconds"], 1)
        self.assertEqual(row["receipt_file_digest"], "sha256:" + sha256(original_bytes).hexdigest())
        self.assertNotEqual(row["receipt_file_digest"], file_digest(receipt))

    def test_receipt_snapshot_rejects_duplicate_keys_and_nonfinite_json(self):
        receipt = self.root / "invalid.json"
        for raw in ('{"metrics":{},"metrics":{}}', '{"metrics":{"train_seconds":NaN}}'):
            receipt.write_text(raw)
            with self.assertRaises(ValueError): self.export([receipt])

    def test_evidence_session_counts_semantics_and_canonical_dates_are_required(self):
        cases = [lambda v: v["signal-evidence.json"]["series"][0].update(prediction_valid_count=2),
                 lambda v: v["signal-evidence.json"]["series"][0].update(valid_pair_count=2, excluded_pair_count=0),
                 lambda v: v["signal-evidence.json"]["series"][0].update(excluded_pair_count=0),
                 lambda v: v["signal-evidence.json"].update(series=v["signal-evidence.json"]["series"][:1]),
                 lambda v: v["signal-evidence.json"].update(score_semantics="different"),
                 lambda v: v["predictions.json"]["rows"][0].update(session="20231229")]
        for mutate in cases:
            fixture(self.saved, mutate=mutate)
            with self.assertRaises(ValueError): self.export()

    def test_source_semantic_errors_are_rejected_even_after_resealing(self):
        cases = [lambda v: v["dataset.json"]["training_keys"].append(["A", "2023-12-20"]),
                 lambda v: v["dataset.json"]["excluded"].update(IMMATURE=-1),
                 lambda v: v["experiment.json"]["definition"]["config"].update(fit_cutoff="2023-12-28T12:30:00"),
                 lambda v: v["predictions.json"]["rows"].append(deepcopy(v["predictions.json"]["rows"][0])),
                 lambda v: v["signal-evidence.json"]["series"][0].update(excluded_pair_count=-1),
                 lambda v: v["signal-evidence.json"]["series"].append(deepcopy(v["signal-evidence.json"]["series"][0]))]
        for mutate in cases:
            fixture(self.saved, mutate=mutate)
            with self.assertRaises(ValueError): self.export()

    def test_nonfinite_ic_is_rejected(self):
        from axiom_research.stock_stage_report import _signal_summary
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            _signal_summary({"score_semantics": "score", "rows": [{"security_id": "A", "session": "2023-12-29", "valid": False}]},
                {"evaluation_cutoff": "2024-01-01T00:00:00Z", "series": [{"session": "2023-12-29", "ic": float("inf"),
                "rank_ic": None, "prediction_valid_count": 0, "valid_pair_count": 0, "excluded_pair_count": 1}],
                "score_semantics": "score", "label_semantics": "return", "minimum_pairs": 20, "rank_ties": "average"})

    def test_integrity_stored_old_implementation_and_reference_closure(self):
        result = self.export(); self.assertEqual(load_stock_stage_report(self.output), result)
        changed = deepcopy(result); changed["training"]["actual"]["training_row_count"] += 1
        write_json(self.output, changed)
        with self.assertRaisesRegex(ValueError, "content digest"): load_stock_stage_report(self.output)
        changed = deepcopy(result); changed["implementation_ref"] = "sha256:" + "c" * 64
        changed["stage_report_ref"] = _identity(changed)
        changed["content_digest"] = digest({k:v for k,v in changed.items() if k != "content_digest"})
        write_json(self.output, changed); self.assertEqual(load_stock_stage_report(self.output), changed)
        changed["measurements"][0]["feature_ref"] = "sha256:" + "f" * 64
        changed["content_digest"] = digest({k:v for k,v in changed.items() if k != "content_digest"})
        write_json(self.output, changed)
        with self.assertRaisesRegex(ValueError, "feature mismatch"): load_stock_stage_report(self.output)

    def test_loader_does_not_follow_inputs_receipts_or_recompute_summary(self):
        result = self.export()
        for path in self.saved.iterdir(): path.unlink()
        self.saved.rmdir()
        with patch("axiom_research.stock_stage_report.load_stock_ml_experiment", side_effect=AssertionError("upstream load")), \
                patch("axiom_research.stock_stage_report._signal_summary", side_effect=AssertionError("statistics")):
            self.assertEqual(load_stock_stage_report(self.output), result)

    def test_identity_and_declared_source_closure_are_checked_when_content_is_resealed(self):
        result = self.export()
        wrong = deepcopy(result); wrong["input_refs"]["signal_run_ref"] = "sha256:" + "f" * 64
        wrong["content_digest"] = digest({k:v for k,v in wrong.items() if k != "content_digest"})
        write_json(self.output, wrong)
        with self.assertRaisesRegex(ValueError, "identity mismatch"): load_stock_stage_report(self.output)
        receipt = self.root / "source.json"
        write_json(receipt, {"experiment_ref": self.experiment["definition"]["input_reuse"]["experiment_ref"],
                            "metrics": {"feature_seconds": 100}})
        self.output.unlink(); result = self.export([receipt])
        result["source_input_reuse"]["experiment_ref"] = "sha256:" + "f" * 64
        result["stage_report_ref"] = _identity(result)
        result["content_digest"] = digest({k:v for k,v in result.items() if k != "content_digest"})
        write_json(self.output, result)
        with self.assertRaisesRegex(ValueError, "declared source input mismatch"): load_stock_stage_report(self.output)

    def test_preserves_saved_files_identical_reuse_and_refuses_overwrite(self):
        before = {p.name: (file_digest(p), p.stat().st_mtime_ns) for p in self.saved.iterdir()}
        result = self.export(); report_before = (file_digest(self.output), self.output.stat().st_mtime_ns)
        self.assertEqual(self.export(), result)
        self.assertEqual(report_before, (file_digest(self.output), self.output.stat().st_mtime_ns))
        self.assertEqual(before, {p.name: (file_digest(p), p.stat().st_mtime_ns) for p in self.saved.iterdir()})
        with self.assertRaisesRegex(ValueError, "outside saved experiment"):
            export_stock_stage_report(self.saved, destination=self.saved / "new-report.json")
        fixture(self.saved, mutate=lambda v: v["dataset.json"]["excluded"].update(IMMATURE=3))
        with self.assertRaises(FileExistsError): self.export()
        self.assertEqual(report_before, (file_digest(self.output), self.output.stat().st_mtime_ns))

    def test_atomic_publish_failure_and_complete_visibility(self):
        with patch("axiom_research.stock_stage_report.os.fsync", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"): self.export()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".stage.json.*.tmp")), [])
        original_link = os.link
        def publish(source, destination):
            self.assertFalse(Path(destination).exists())
            complete = load_stock_stage_report(source)
            self.assertEqual(complete["training"]["actual"]["training_row_count"], 3)
            return original_link(source, destination)
        with patch("axiom_research.stock_stage_report.os.link", publish):
            result = self.export()
        self.assertEqual(load_stock_stage_report(self.output), result)
        self.assertEqual(list(self.root.glob(".stage.json.*.tmp")), [])

    def test_loader_validates_stored_version_and_measurement_shape(self):
        result = self.export()
        for change in [lambda r: r.update(report_version="axiom.stock_stage_report/2"),
                       lambda r: r["measurements"][0].update(seconds=-1),
                       lambda r: r["measurements"][0].update(status="REUSED_NOT_EXECUTED", stage="train"),
                       lambda r: r["measurements"][0].update(status="MEASURED")]:
            wrong = deepcopy(result); change(wrong)
            wrong["stage_report_ref"] = _identity(wrong)
            wrong["content_digest"] = digest({k:v for k,v in wrong.items() if k != "content_digest"})
            write_json(self.output, wrong)
            with self.assertRaises(ValueError): load_stock_stage_report(self.output)

    def test_publication_race_reuses_only_identical_complete_winner(self):
        original_link = os.link
        for same in (True, False):
            winner_stat = []
            def publish(source, destination):
                winner = _read(source)
                if not same:
                    winner["training"]["excluded"]["IMMATURE"] += 1
                    winner["content_digest"] = digest({k:v for k,v in winner.items() if k != "content_digest"})
                write_json(destination, winner)
                winner_stat.append((file_digest(destination), Path(destination).stat().st_mtime_ns))
                return original_link(source, destination)
            with patch("axiom_research.stock_stage_report.os.link", publish):
                if same: self.export()
                else:
                    with self.assertRaises(FileExistsError): self.export()
            self.assertEqual(winner_stat[0], (file_digest(self.output), self.output.stat().st_mtime_ns))
            self.assertEqual(list(self.root.glob(".stage.json.*.tmp")), [])
            self.output.unlink()

    def test_fresh_process_export_and_loader_block_upstream_and_training(self):
        code = '''import builtins,sys
original=builtins.__import__
def guard(name,*a,**k):
 if name.startswith(('axiom_data','axiom_engine','lightgbm','qlib','numpy','pandas','axiom_research.stock_ml')): raise AssertionError(name)
 return original(name,*a,**k)
builtins.__import__=guard
from axiom_research import export_stock_stage_report,load_stock_stage_report
r=export_stock_stage_report(sys.argv[1],destination=sys.argv[2])
assert load_stock_stage_report(sys.argv[2])==r
assert not any(n.startswith(('axiom_data','axiom_engine','qlib','lightgbm','axiom_research.stock_ml')) for n in sys.modules)
'''
        subprocess.run([sys.executable, "-c", code, str(self.saved), str(self.output)], check=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)


if __name__ == "__main__": unittest.main()

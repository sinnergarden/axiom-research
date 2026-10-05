"""Small independent regressions for projection, saved linkage and pure loading."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from axiom_research.stock_artifacts import digest, file_digest, write_json, load_stock_ml_experiment
from axiom_research.stock_ml import _project_qlib, _validate_config


def seal(value, key):
    return {**value, key: digest(value)}


def saved_fixture(path, *, wrong_dataset=False):
    """Nine sealed files; no model training or upstream execution."""
    columns, fit = ["MOM010"], "2023-12-28T12:30:00Z"
    (path / "booster.txt").write_text("offline saved fixture")
    training = seal({"contract_version": "stock_label_build_v1", "rows": []}, "label_ref")
    evaluation = seal({"contract_version": "stock_label_build_v1", "rows": []}, "label_ref")
    features = seal({"contract_version": "stock_feature_build_v1", "ordered_features": columns,
                     "input_evidence_ref": digest([])}, "feature_ref")
    labels = seal({"contract_version": "stock_label_bundle_v1", "training": training,
                   "evaluation": evaluation}, "label_ref")
    dataset = seal({"contract_version": "stock_training_dataset_v1", "feature_ref": features["feature_ref"],
                    "label_ref": training["label_ref"], "ordered_features": columns,
                    "fit_cutoff": fit}, "dataset_ref")
    model = seal({"contract_version": "stock_model_release_v1", "dataset_ref":
        "sha256:" + "f" * 64 if wrong_dataset else dataset["dataset_ref"],
        "feature_ref": features["feature_ref"], "ordered_features": columns,
        "fit_cutoff": fit, "booster_digest": file_digest(path / "booster.txt")}, "model_ref")
    predictions = seal({"contract_version": "stock_prediction_run_v1", "model_ref": model["model_ref"],
                        "feature_ref": features["feature_ref"]}, "signal_run_ref")
    evidence = seal({"contract_version": "stock_signal_evidence_v1", "signal_ref": predictions["signal_run_ref"],
                     "label_ref": evaluation["label_ref"]}, "evidence_ref")
    values = {"features.json": features, "labels.json": labels, "dataset.json": dataset,
              "model.json": model, "predictions.json": predictions, "signal-evidence.json": evidence,
              "feature-inputs.json": []}
    refs = {key: values[name][key] for name, key in (("features.json", "feature_ref"),
        ("labels.json", "label_ref"), ("dataset.json", "dataset_ref"), ("model.json", "model_ref"),
        ("predictions.json", "signal_run_ref"), ("signal-evidence.json", "evidence_ref"))}
    experiment = seal({"contract_version": "stock_ml_experiment_v1", "definition": {}, **refs}, "experiment_ref")
    values["experiment.json"] = experiment
    for name, value in values.items():
        write_json(path / name, value)
    write_json(path / "manifest.json", {"contract_version": "stock_ml_manifest_v1",
        "experiment_ref": experiment["experiment_ref"],
        "files": {name: file_digest(path / name) for name in (*values, "booster.txt")}})
    return experiment


class StockMLReviewTests(unittest.TestCase):
    def test_projection_accepts_only_exact_float32_rounding(self):
        import numpy as np
        import pandas as pd
        from axiom_data import DataBatch
        raw = 100.000001
        frame = pd.DataFrame([{"security_id": "A", "session": "2024-01-02", "close": raw}])
        batch = DataBatch(frame, {"close": {"by_key": []}}, {"query": {"fields": ["close"]}})
        rounded = float(np.float32(raw))
        result = _project_qlib(batch, {("A", "2024-01-02"): {"close": rounded}}, "fixed-view")
        self.assertEqual(float(result.frame.iloc[0]["close"]), rounded)
        self.assertEqual(float(batch.frame.iloc[0]["close"]), raw)
        with self.assertRaisesRegex(ValueError, "revision or value mismatch"):
            _project_qlib(batch, {("A", "2024-01-02"): {"close": float(np.float32(100.00005))}}, "fixed-view")

    def test_calendar_hole_rejected_before_core(self):
        calendar = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        config = dict(snapshot="s-fixed", universe_id="csi300", symbols=["A"], calendar=calendar,
            read_sessions=calendar, feature_sessions=calendar, prediction_sessions=[calendar[-1]],
            fit_cutoff="2024-01-04T20:00:00Z", pit_policy="market_pit_safe_v1",
            cutoff_by_session={s: s + "T20:30:00Z" for s in calendar},
            evaluation_cutoff="2024-01-08T20:30:00Z", feature_selection=[], scope_ref="fixed", calendar_ref="fixed")
        _validate_config(config)
        broken = deepcopy(config)
        for key in ("read_sessions", "feature_sessions"):
            broken[key] = [s for s in broken[key] if s != calendar[1]]
        broken["cutoff_by_session"].pop(calendar[1])
        with self.assertRaisesRegex(ValueError, "every actual exchange session"):
            _validate_config(broken)

    def test_sealed_wrong_model_dataset_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            experiment = saved_fixture(path)
            self.assertEqual(load_stock_ml_experiment(path).identity, experiment["experiment_ref"])
            saved_fixture(path, wrong_dataset=True)
            with self.assertRaisesRegex(ValueError, "stage linkage"):
                load_stock_ml_experiment(path)

    def test_loader_parses_once_but_verifies_again_on_every_invocation(self):
        import axiom_research.stock_artifacts as artifacts
        expected_names = {"manifest.json", "experiment.json", "features.json", "labels.json",
                          "dataset.json", "model.json", "predictions.json", "signal-evidence.json"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            saved_fixture(path)
            with patch.object(artifacts, "_read", wraps=artifacts._read) as read:
                artifacts.load_stock_ml_experiment(path)
                self.assertEqual(len(read.call_args_list), len(expected_names))
                self.assertEqual({call.args[0].name for call in read.call_args_list}, expected_names)
                read.reset_mock()
                artifacts.load_stock_ml_experiment(path)
                self.assertEqual(len(read.call_args_list), len(expected_names))
            saved_fixture(path, wrong_dataset=True)
            with self.assertRaisesRegex(ValueError, "stage linkage"):
                artifacts.load_stock_ml_experiment(path)
            saved_fixture(path)
            with (path / "features.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "file mismatch: features.json"):
                artifacts.load_stock_ml_experiment(path)

    def test_loader_import_and_execution_have_no_upstream_dependencies(self):
        import builtins
        import axiom_research.stock_artifacts as artifacts
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.startswith(("axiom_data", "axiom_engine", "lightgbm", "qlib", "numpy", "pandas")):
                raise AssertionError("saved loader imported upstream runtime: " + name)
            return original_import(name, *args, **kwargs)

        name = "isolated_stock_artifact_review"
        spec = importlib.util.spec_from_file_location(name, artifacts.__file__)
        module = importlib.util.module_from_spec(spec)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {name: module}):
            path = Path(directory)
            expected = saved_fixture(path)
            with patch("builtins.__import__", guarded):
                spec.loader.exec_module(module)
                loaded = module.load_stock_ml_experiment(path)
                self.assertEqual(loaded.identity, expected["experiment_ref"])
                self.assertEqual(module.load_stock_model(path)["ordered_features"], ["MOM010"])

    def test_label_source_context_is_frozen_after_build(self):
        from test_stock_labels import batch, build
        value = batch()
        result = build(value)
        expected = deepcopy(result)
        value.wire["context"]["limitations"].append("later caller mutation")
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

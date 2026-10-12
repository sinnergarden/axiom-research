"""Real saved Research admission plus isolated trusted Engine owner projections."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import (ExperimentReader, ExperimentStore, ExperimentRecordError,
    attach_stock_account_from_basis, begin_stock_result_registration, publish_stock_result)
from axiom_research.result_discovery import read_json
from axiom_research.result_registration import _save_stock_sequential_receipt
from axiom_research.stock_artifacts import digest, write_json
from axiom_engine.runtime.stock_stream_contracts import logical_ref
from test_result_metadata_budget import saved_output_fixture


class RegisteredBasisAccountTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.engine = self.root / "engine"; self.engine.mkdir()
        self.roots = dict(research=self.root, engine=self.engine)
        self.index = self.root / "index.json"
        cfg, result = saved_output_fixture(self.root, padding_bytes=32)
        self.receipt = _save_stock_sequential_receipt(result, destination=self.root)
        self.request = self.root / "request.json"
        options = dict(request_path=self.request, index_path=self.index, roots=self.roots)
        begin_stock_result_registration(cfg, **options, required_stages=["training", "signal_evaluation"])
        signal_ref = result["prediction_bindings"][0]["signal_run_ref"]
        evaluation = dict(contract_version="static_signal_evaluation_v1",
            evaluation_ref=digest("old Signal evaluation"), content_digest=digest("old evaluation content"),
            input_signal_refs=[signal_ref])
        self.old_evaluation = self.root / "signal-evaluation.json"
        write_json(self.old_evaluation, evaluation)
        outcome = self.root / "evaluation-outcome.json"
        write_json(outcome, dict(statistics={"all": dict(path=str(self.old_evaluation))}))
        owner = SimpleNamespace(identity=evaluation["evaluation_ref"], to_dict=lambda: deepcopy(evaluation))
        with patch("axiom_research.stock_signal_evaluation.load_stock_signal_evaluation", return_value=owner):
            head = publish_stock_result(**options, experiment_result=self.receipt, signal_evaluation=outcome)
        self.basis = {k: deepcopy(head[k]) for k in ("result_id", "revision", "manifest_ref")}
        self.basis["content_digest"] = head["manifest_ref"]["content_digest"]
        self.previous = self.manifest()
        frames = read_json(self.receipt.parent / "prediction-bindings.json")
        inputs = dict(contract_version="stock_prediction_input_refs_v2", frames=frames)
        inputs["prediction_ref"] = logical_ref(inputs, "prediction_ref")
        self.run_ref = dict(run_id=digest("new saved account"), content_digest=digest("new saved account content"),
                            committed_sequence=7)
        wire = dict(contract_version="static_backtest_run_v1", **self.run_ref, status="COMPLETE", stopped=None,
            signal_ref=inputs["prediction_ref"], request_manifest=dict(portfolio_policy={"strategy": "new"},
                                                                      prediction_input=inputs))
        self.run_path = self.engine / "run.json"; write_json(self.run_path, wire)
        self.leaf = self.engine / "part.json"; write_json(self.leaf, {"trusted owner leaf": "saved"})
        report = dict(contract_version="static_backtest_evaluation_v1", evaluation_ref=digest("new evaluation"),
                      content_digest=digest("new evaluation content"), input_run_ref=self.run_ref)
        self.report_path = self.engine / "evaluation.json"; write_json(self.report_path, report)

    def manifest(self):
        return ExperimentReader(self.index).result(self.basis["result_id"], roots=self.roots)

    def owners(self, stack):
        def load(path, *, artifact_reader, limits):
            read_json(artifact_reader({"manifest_uri": str(self.leaf)}))
            wire = read_json(path)
            ref = {k: wire[k] for k in ("run_id", "content_digest", "committed_sequence")}
            return SimpleNamespace(input_run_ref=ref, verify=lambda: SimpleNamespace(wire=wire))
        run = stack.enter_context(patch("axiom_engine.runtime.load_stock_backtest_projection", side_effect=load))
        evaluation = stack.enter_context(patch("axiom_engine.runtime.load_backtest_evaluation",
            side_effect=lambda path: SimpleNamespace(to_dict=lambda: read_json(path))))
        return run, evaluation

    def forbidden_history(self, stack):
        spies = []
        for loader in ("load_stock_fold_projection", "_load_stock_prediction_registration_projection",
                       "load_stock_sequential_result", "load_stock_prediction_bindings"):
            spies.append(stack.enter_context(patch("axiom_research.stock_result_projection." + loader,
                side_effect=AssertionError("historical output audit"))))
        spies.append(stack.enter_context(patch("axiom_research.stock_signal_evaluation.load_stock_signal_evaluation",
            side_effect=AssertionError("historical Signal evaluation audit"))))
        for name in ("run_backtest", "evaluate_backtest"):
            stack.enter_context(patch("axiom_engine.runtime." + name, side_effect=AssertionError("account replay")))
        stack.enter_context(patch("axiom_research.stock_training.fit_predict_stock_model",
                                  side_effect=AssertionError("training replay")))
        return spies

    def attach(self):
        return attach_stock_account_from_basis(self.basis, index_path=self.index, roots=self.roots,
            run_path=self.run_path, evaluation_paths=[self.report_path])

    def test_new_account_inherits_history_and_exact_retry_keeps_index_and_manifest(self):
        with ExitStack() as stack:
            run, evaluation = self.owners(stack)
            history = self.forbidden_history(stack)
            head = self.attach()
            actual = self.manifest()
            self.assertEqual((head["status"], head["revision"]), ("COMPLETE", self.basis["revision"] + 1))
            self.assertEqual(len(self.previous["accounts"]), 0)
            self.assertEqual(len(actual["accounts"]), 1)
            self.assertEqual(actual["accounts"][0]["backtest_ref"]["run_id"], self.run_ref["run_id"])
            self.assertEqual(len(actual["accounts"][0]["evaluation_refs"]), 1)
            for key in ("folds", "signals", "configuration_ref"):
                self.assertEqual(actual[key], self.previous[key])
            self.assertTrue(all(a in actual["artifacts"] for a in self.previous["artifacts"]))
            self.assertEqual((run.call_count, evaluation.call_count), (1, 1))
            target = self.root / head["manifest_ref"]["relative_path"]
            before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in (self.index, target)]
            self.assertEqual(self.attach(), head)
            self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in (self.index, target)])
            self.assertTrue(all(loader.call_count == 0 for loader in history))

    def test_wrong_prediction_or_new_evaluation_binding_refuses_without_commit(self):
        before = self.index.read_bytes()
        original = read_json(self.run_path)
        changed = deepcopy(original)
        inputs = changed["request_manifest"]["prediction_input"]
        inputs["frames"][0]["model_ref"] = digest("different admitted model")
        inputs["prediction_ref"] = logical_ref(inputs, "prediction_ref")
        changed["signal_ref"] = inputs["prediction_ref"]
        with ExitStack() as stack:
            self.owners(stack); self.forbidden_history(stack)
            with patch("axiom_engine.runtime.load_stock_backtest_projection",
                       side_effect=ValueError("owner refuses invalid saved account")):
                with self.assertRaisesRegex(ValueError, "invalid saved account"): self.attach()
            invalid = deepcopy(original); invalid["stopped"] = {"reason": "inconsistent saved status"}
            write_json(self.run_path, invalid)
            with self.assertRaisesRegex(ExperimentRecordError, "status/stopped mismatch"): self.attach()
            write_json(self.run_path, changed)
            with self.assertRaisesRegex(ExperimentRecordError, "basis original prediction"):
                self.attach()
            write_json(self.run_path, original)
            report = read_json(self.report_path)
            report["input_run_ref"]["content_digest"] = digest("another account")
            write_json(self.report_path, report)
            with self.assertRaisesRegex(ExperimentRecordError, "evaluation run binding"):
                self.attach()
        self.assertEqual(self.index.read_bytes(), before)

    def test_changed_small_basis_documents_are_not_borrowed_from_registration(self):
        for path in (self.request, self.receipt.parent / "prediction-bindings.json"):
            original = path.read_bytes()
            value = read_json(path)
            if type(value) is dict:
                value["configuration"]["dataset"]["scope"]["universe"] = ["B"]
            else:
                value[0]["feature_ref"] = digest("changed frame content")
            write_json(path, value)
            before = self.index.read_bytes()
            with ExitStack() as stack:
                run, _ = self.owners(stack); self.forbidden_history(stack)
                with self.assertRaises(ExperimentRecordError): self.attach()
                self.assertEqual(run.call_count, 0)
            self.assertEqual(self.index.read_bytes(), before)
            path.write_bytes(original)

    def test_basis_head_cas_and_nonidentical_retry_refuse(self):
        mutate = ExperimentStore._mutate
        def conflict(store, operation):
            def concurrent(state):
                state["result_heads"][self.basis["result_id"]]["reason"] = "concurrent change"
                return operation(state)
            return mutate(store, concurrent)
        before = self.index.read_bytes()
        with ExitStack() as stack:
            self.owners(stack); self.forbidden_history(stack)
            with patch.object(ExperimentStore, "_mutate", conflict):
                with self.assertRaisesRegex(ExperimentRecordError, "basis/head conflict"): self.attach()
            self.assertEqual(self.index.read_bytes(), before)
            self.attach()
            before = self.index.read_bytes()
            wire = read_json(self.run_path)
            wire["request_manifest"]["portfolio_policy"]["strategy"] = "different request"
            write_json(self.run_path, wire)
            with self.assertRaisesRegex(ExperimentRecordError, "basis conflict"): self.attach()
            self.assertEqual(self.index.read_bytes(), before)

    def test_new_nested_leaf_change_and_atomic_index_failure_preserve_old_head(self):
        original = read_json(self.report_path)
        def changed_leaf(path):
            write_json(self.leaf, {"trusted owner leaf": "changed after validation"})
            return SimpleNamespace(to_dict=lambda: original)
        before = self.index.read_bytes()
        with ExitStack() as stack:
            self.owners(stack); self.forbidden_history(stack)
            with patch("axiom_engine.runtime.load_backtest_evaluation", side_effect=changed_leaf):
                with self.assertRaisesRegex(ExperimentRecordError, "changed during projection session"): self.attach()
            self.assertEqual(self.index.read_bytes(), before)
            replace = os.replace
            def refused(source, target):
                if Path(target) == self.index: raise OSError("synthetic atomic index refusal")
                return replace(source, target)
            with patch("axiom_research.experiments.os.replace", side_effect=refused):
                with self.assertRaisesRegex(OSError, "atomic index refusal"): self.attach()
            self.assertEqual(self.index.read_bytes(), before)
            self.assertEqual(self.manifest(), self.previous)
            self.assertEqual(self.attach()["revision"], self.basis["revision"] + 1)

    def test_fast_path_inherits_admission_but_full_audit_still_reads_actual_history(self):
        # A historical leaf's loss is not hidden by an owner-audit cache or stat inventory.
        (self.root / "fold" / "booster.txt").unlink()
        with ExitStack() as stack:
            self.owners(stack); history = self.forbidden_history(stack)
            self.attach()
            self.assertTrue(all(loader.call_count == 0 for loader in history))
        before = self.index.read_bytes()
        with ExitStack() as stack:
            self.owners(stack)
            with self.assertRaises(FileNotFoundError):
                ExperimentStore(self.index).publish_result(self.manifest(), roots=self.roots)
        self.assertEqual(self.index.read_bytes(), before)

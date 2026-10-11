from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import (ExperimentStore, ExperimentReader, ExperimentRecordError,
    begin_stock_result_registration, publish_stock_result)
from axiom_research.stock_artifacts import digest
from axiom_research.result_registration import _seal


def configuration():
    spec = digest({"contract_version":"stock_effective_configuration_v1","spec_refs":{}})
    dataset = {"fold_specs":[{}]}
    dataset_ref = digest(dataset)
    return {"configuration_ref":digest({"contract_version":"stock_sequential_effective_configuration_v1",
        "specification_ref":spec,"dataset_ref":dataset_ref}),"dataset":dataset,"dataset_ref":dataset_ref,
        "specification":{"configuration_ref":spec,"spec_refs":{}}}


class ResultDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.roots = {"research":self.root}
        self.index = self.root/"index.json"
        self.request = self.root/"attempt.json"
        self.options = dict(index_path=self.index,roots=self.roots,request_path=self.request,
                            required_stages=["training","signal_evaluation"])
        self.frozen = begin_stock_result_registration(configuration(), **self.options)
        self.store = ExperimentStore(self.index)
        self.reader = ExperimentReader(self.index)

    def tearDown(self):
        self.temp.cleanup()

    def first(self):
        return publish_stock_result(self.request,index_path=self.index,roots=self.roots)

    def revision(self, previous, **changes):
        value = deepcopy(previous); value.pop("content_digest")
        value.update({"revision":value["revision"]+1, **changes})
        return _seal(value)

    def test_attempt_is_frozen_before_execution_and_retry_reuses_it(self):
        before = self.request.read_bytes(), self.request.stat().st_mtime_ns
        self.assertEqual(begin_stock_result_registration(configuration(),**self.options),self.frozen)
        self.assertEqual(before,(self.request.read_bytes(),self.request.stat().st_mtime_ns))
        self.assertEqual(self.frozen["registration_mode"],"pre_execution")
        self.assertEqual(self.reader.index()["contract_version"],"experiment_projection_v1")
        head = self.first()
        self.assertEqual(head["status"],"RUNNING")
        self.assertEqual(self.reader.index()["contract_version"],"experiment_projection_v2")
        self.assertEqual(self.reader.index()["result_heads"],[head])

    def test_atomic_idempotent_publish_keeps_bytes_mtime_and_old_records(self):
        old = self.reader.index()["questions"]
        first = self.first()
        before = self.index.read_bytes(),self.index.stat().st_mtime_ns
        self.assertEqual(self.first(),first)
        self.assertEqual(before,(self.index.read_bytes(),self.index.stat().st_mtime_ns))
        self.assertEqual(self.reader.index()["questions"],old)
        self.assertEqual(self.reader.index()["questions"][0]["saved_backtest_count"],0)

    def test_failed_revision_keeps_identity_and_immutable_history(self):
        old = self.first(); previous = self.reader.result(old["result_id"],roots=self.roots)
        new = self.revision(previous,status="FAILED",reason="Synthetic failure")
        head = self.store.publish_result(new,roots=self.roots)
        self.assertEqual((head["result_id"],head["revision"]),(old["result_id"],1))
        self.assertEqual(len(self.reader.index()["result_heads"]),1)
        self.assertEqual(self.reader.index(status="FAILED")["result_heads"],[head])
        self.assertEqual(json.loads((self.root/old["manifest_ref"]["relative_path"]).read_text()),previous)

    def test_conflict_skipped_revision_and_changed_artifact_leave_old_head(self):
        head = self.first(); previous = self.reader.result(head["result_id"],roots=self.roots)
        before = self.index.read_bytes()
        conflict = deepcopy(previous);conflict.update(status="BLOCKED",reason="different same revision")
        conflict.pop("content_digest"); conflict=_seal(conflict)
        for bad in (conflict,self.revision(previous,revision=4,status="BLOCKED",reason="gap")):
            with self.assertRaises(ExperimentRecordError):
                self.store.publish_result(bad,roots=self.roots)
            self.assertEqual(self.index.read_bytes(),before)
        self.request.write_text("{}")
        with self.assertRaises((KeyError,ExperimentRecordError)):
            self.store.publish_result(self.revision(previous,status="FAILED",reason="retry"),roots=self.roots)
        self.assertEqual(self.index.read_bytes(),before)

    def test_roots_refuse_absolute_traversal_network_and_escaping_symlink(self):
        head = self.first(); previous = self.reader.result(head["result_id"],roots=self.roots)
        for relative in ("/tmp/x","../x","a//b","https://example.com/x","a\\b","."):
            bad = self.revision(previous)
            bad["artifacts"][0]["relative_path"]=relative;bad.pop("content_digest");bad=_seal(bad)
            with self.assertRaises(ExperimentRecordError): self.store.publish_result(bad,roots=self.roots)
        outside = self.root.parent/(self.root.name+"-outside.json")
        outside.write_bytes(self.request.read_bytes())
        try:
            (self.root/"escape.json").symlink_to(outside)
            bad=self.revision(previous);bad["artifacts"][0]["relative_path"]="escape.json"
            bad.pop("content_digest");bad=_seal(bad)
            with self.assertRaises(ExperimentRecordError): self.store.publish_result(bad,roots=self.roots)
        finally: outside.unlink()

    def test_complete_cannot_be_inferred_from_nonempty_files(self):
        head=self.first();previous=self.reader.result(head["result_id"],roots=self.roots)
        with self.assertRaises(ExperimentRecordError):
            self.store.publish_result(self.revision(previous,status="COMPLETE",reason=None),roots=self.roots)

    def test_reader_does_not_open_owner_original_or_import_runtime(self):
        head=self.first();request=self.request
        original=Path.open
        def checked(path,*args,**kwargs):
            if path==request: raise AssertionError("owner original opened")
            return original(path,*args,**kwargs)
        with patch.object(Path,"open",checked), patch(
            "axiom_research.stock_result_projection.verify_result_artifacts",
            side_effect=AssertionError("owner validation executed")):
            self.assertEqual(self.reader.result(head["result_id"],roots=self.roots)["result_id"],head["result_id"])
            self.assertEqual(len(self.reader.index()["result_heads"]),1)

    def test_concurrent_revision_conflict_cannot_drop_updates(self):
        head=self.first();previous=self.reader.result(head["result_id"],roots=self.roots)
        def writer(reason):
            try: return self.store.publish_result(self.revision(previous,status="BLOCKED",reason=reason),roots=self.roots)
            except ExperimentRecordError:return None
        with ThreadPoolExecutor(2) as pool: results=list(pool.map(writer,["first","second"]))
        self.assertEqual(sum(r is not None for r in results),1)
        self.assertEqual(self.reader.index()["result_heads"][0]["revision"],1)

    def test_concurrent_request_retry_preserves_attempt(self):
        options={**self.options,"request_path":self.root/"concurrent.json"}
        with ThreadPoolExecutor(2) as pool:
            results=list(pool.map(lambda _:begin_stock_result_registration(configuration(),**options),range(2)))
        self.assertEqual(results[0],results[1])

    def test_bad_evaluation_binding_is_rejected_before_owner_loading(self):
        head=self.first();previous=self.reader.result(head["result_id"],roots=self.roots)
        bad=self.revision(previous)
        bad["accounts"]=[dict(strategy_ref=digest("policy"),signal_run_ref=digest("frames"),
            status="PARTIAL",reason="missing",backtest_ref=None,evaluation_refs=[dict(
                artifact_key="configuration",evaluation_ref=digest("eval"),content_digest=digest("eval"),
                input_run_ref={"run_id":digest("other"),"content_digest":digest("out"),"committed_sequence":0})])]
        bad.pop("content_digest");bad=_seal(bad)
        with self.assertRaises(ExperimentRecordError): self.store.publish_result(bad,roots=self.roots)

    def test_configured_notebook_freezes_and_registers_failure_before_owner_execution(self):
        from axiom_research.stock_experiment_entry import build_configured_stock_sequential_experiment
        cfg=configuration()
        cfg["dataset"].update(preparation_options={},scope={},model_feature_selection=None,signal_contexts=None)
        cfg["dataset_ref"]=digest(cfg["dataset"])
        cfg["configuration_ref"]=digest({"contract_version":"stock_sequential_effective_configuration_v1",
            "specification_ref":cfg["specification"]["configuration_ref"],"dataset_ref":cfg["dataset_ref"]})
        context={**self.options};context.pop("required_stages");context["request_path"]=self.root/"notebook.json"
        def forbidden(**kwargs):
            self.assertTrue(context["request_path"].exists())
            self.assertEqual(self.reader.index()["result_heads"][0]["status"],"RUNNING")
            raise RuntimeError("Synthetic stop before training")
        with patch("axiom_research.stock_experiment_config.load_stock_sequential_configuration",return_value=cfg), \
             patch("axiom_research.stock_experiment_entry._build_stock_sequential_experiment",
                   side_effect=lambda *args,**kwargs:forbidden(**kwargs)):
            with self.assertRaisesRegex(RuntimeError,"Synthetic stop"):
                build_configured_stock_sequential_experiment(None,configuration_path=self.root/"config.yaml",
                    feature_inputs=None,column_source=None,destination=self.root/"output"/"training",
                    registration=context)
        self.assertEqual(self.reader.index()["result_heads"][0]["status"],"FAILED")

    def test_signal_evaluation_refuses_nested_external_and_symlink_inputs(self):
        from axiom_research import ArtifactRef,to_dict,load_stock_signal_evaluation
        from axiom_research.stock_artifacts import write_json,file_digest
        from axiom_research.stock_signal_evaluation import SPEC
        from axiom_research.stock_signal_evaluation_projection import _group_key
        outside=self.root.parent/(self.root.name+"-input.json")
        outside.write_text("{}")
        folder=self.root/"signal";folder.mkdir()
        try:
            (self.root/"input-link.json").symlink_to(outside)
            for location in (outside,self.root/"input-link.json"):
                reference=to_dict(ArtifactRef(artifact_type="StockSignalEvaluationInputs",artifact_id=digest("input"),
                    artifact_contract_version="stock_signal_evaluation_inputs_v4",
                    content_digest=file_digest(outside),uri=str(location)))
                group=dict(contract_version="stock_signal_evaluation_group_v1",input_ref=reference,
                    admission_receipt_ref=digest("admit"),scope=dict(sessions=["2023-01-03"],
                        calendar=["2023-01-03"],universe=["stock"],evaluation_cutoff="2023-12-31T00:00:00Z"),
                    signal_order=["model"],spec_ref=digest(SPEC),spec=SPEC,sample_mask_ref=digest("mask"),
                    common_input_ref=digest("common"),native_input_ref=digest("native"),
                    implementation_ref=digest({}),implementation_sources={})
                group["evaluation_key"]=_group_key(group)
                write_json(folder/"group.json",group)
                write_json(folder/"manifest.json",dict(contract_version="stock_signal_evaluation_manifest_v2",
                    evaluation_key=group["evaluation_key"],files={"group.json":file_digest(folder/"group.json")}))
                with self.assertRaisesRegex(ValueError,"escapes research_root"):
                    load_stock_signal_evaluation(folder,research_root=self.root)
        finally:outside.unlink()


if __name__=="__main__": unittest.main()

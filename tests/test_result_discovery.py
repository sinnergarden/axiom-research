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

    def saved_output_fixture(self):
        """Trusted owner projections isolate registration's cross-owner checks."""
        from axiom_research import (LabelSpec, MaturitySpec, TrainingSpec, ResourceSpec,
            SessionRange, semantic_identity)
        from axiom_research.stock_target_spec import resolve_stock_label_spec
        from axiom_research.stock_training import LGBM_PARAMETERS, resolve_stock_training_spec
        from axiom_research.stock_artifacts import write_json, file_digest
        label = LabelSpec(contract_version='2',name='label',key=('security_id','session'),
            horizon_sessions=5,feature_session='f',formula='close(f+h) / open(f+1) - 1',
            return_start_rule='next_session_open',return_end_rule='horizon_session_close',
            return_start_offset_sessions=1,return_end_offset_sessions=5,
            price_basis='common_anchor_adjusted_v1',benchmark_semantics='absolute_return',
            corporate_action_semantics='factor_ratio_no_separate_cashflow',normalization_policy='none',
            maturity=MaturitySpec(lag_sessions=5,rule='all_outcome_dependencies_strictly_before_fit_cutoff',
                calendar_policy='actual_exchange_sessions',availability_rule='max_endpoint_price_factor_anchor_usable_from'),
            missing_delisting_policy='invalid_null_preserve_grid')
        model = TrainingSpec(name='model',dataset_name='dataset',backend='lightgbm',backend_version='4.6.0',
            objective='regression',parameters={**LGBM_PARAMETERS,'n_estimators':100},seed=42,
            preprocessing='none',fit_scope='train_only',fit_range=SessionRange(start='2020-01-01',end='2026-09-30'),
            selection_protocol='fixed_parameters_no_validation_no_early_stopping',
            resources=ResourceSpec(threads=1,concurrent_folds=1,memory_limit_mb=512))
        refs = {k:semantic_identity(v) for k,v in dict(label=label,model=model).items()}
        specs = dict(label=label,model=model)
        target = resolve_stock_label_spec(label)
        parameters, rounds, training = resolve_stock_training_spec(model)
        scope = dict(calendar=['2023-01-03'],sessions=['2023-01-03'],universe=['stock'],
                     evaluation_cutoff='2024-01-01T00:00:00Z')
        dataset = dict(fold_specs=[{'window':0},{'window':1}],preparation_options={'column_block':32},
                       model_feature_selection=None,scope=scope,signal_contexts=None)
        specref = digest(dict(contract_version='stock_effective_configuration_v1',spec_refs=refs))
        cfg = dict(dataset=dataset,dataset_ref=digest(dataset),specification=dict(configuration_ref=specref,
                   spec_refs=refs,specs=specs))
        cfg['configuration_ref'] = digest(dict(contract_version='stock_sequential_effective_configuration_v1',
            specification_ref=specref,dataset_ref=cfg['dataset_ref']))
        request = self.root/'output-request.json'
        frozen = begin_stock_result_registration(cfg,**{**self.options,'request_path':request,
                                                        'required_stages':['training']})
        first = publish_stock_result(request,index_path=self.index,roots=self.roots)
        previous = self.reader.result(first['result_id'],roots=self.roots)
        inputs = self.root/'inputs.json'; write_json(inputs,dict(input_id=digest('input'),scope=scope))
        view = dict(spec={k:scope[k] for k in ('calendar','universe')})
        batch = dict(contract_version='stock_ml_batch_inputs_v5',batch_ref=digest('batch'),
            content_digest=digest('batch-content'),definition=dict(fold_specs=dataset['fold_specs'],
                preparation_options={**dataset['preparation_options'],'control_layout':'fold_controls_v2'},
                target_spec=target,feature_view=view),
            folds=[dict(fold_spec=spec,input_manifest={'input':i}) for i,spec in enumerate(dataset['fold_specs'])])
        folds, predictions = [], []
        for i, spec in enumerate(dataset['fold_specs']):
            fold = dict(contract_version='stock_ml_fold_v4',fold_ref=digest(['fold',i]),
                content_digest=digest(['fold-content',i]),signal_run_ref=digest(['signal',i]),
                model_ref=digest(['model',i]),feature_ref=digest(['feature',i]),definition=dict(
                    fold_spec=spec,fold_spec_ref=digest(spec),input_manifest={'input':i},target_spec=target,
                    parameters=parameters,num_boost_round=rounds,**training))
            folds.append(fold)
            predictions.append(dict(contract_version='stock_prediction_frame_v2',fold_spec_ref=digest(spec),
                **{k:fold[k] for k in ('signal_run_ref','model_ref','feature_ref')}))
        bindings = dict(contract_version='stock_prediction_input_refs_v2',prediction_ref=digest('frames'),frames=[])
        sequential = dict(contract_version='stock_sequential_result_projection_v1',result_ref=digest('sequential'),
            configuration_ref=cfg['configuration_ref'],dataset_configuration_ref=cfg['dataset_ref'],spec_refs=refs,
            batch_ref=batch['batch_ref'],batch_content_digest=batch['content_digest'],
            folds=[{k:f[k] for k in ('fold_ref','signal_run_ref')} for f in folds],prediction_input=bindings,
            signal_evaluation_input=dict(uri=str(inputs),artifact_id=digest('input'),content_digest=file_digest(inputs)))
        outputs = {'sequential':('StockSequentialResult',sequential,'result_ref',sequential['result_ref']),
            'batch':('StockMLBatchManifest',batch,'batch_ref',batch['content_digest']),
            'bindings':('StockPredictionBindings',bindings,'prediction_ref',digest('bindings-content'))}
        for i in range(2):
            outputs['fold-'+str(i)] = ('StockMLFoldProjection',folds[i],'fold_ref',folds[i]['content_digest'])
            outputs['prediction-'+str(i)] = ('StockPredictionRun',predictions[i],'signal_run_ref',digest(predictions[i]))
        manifest = self.revision(previous,status='COMPLETE',reason=None)
        for key,(kind,wire,identity,content) in outputs.items():
            write_json(self.root/(key+'.json'),wire)
            manifest['artifacts'].append(dict(artifact_key=key,owner='Research',artifact_type=kind,
                artifact_id=wire[identity],artifact_contract_version=wire['contract_version'],content_digest=content,
                root_id='research',relative_path=key+'.json'))
        manifest['folds'] = [dict(fold_ref=f['fold_ref'],status='COMPLETE',reason=None,
            artifact_keys=['fold-'+str(i),'batch','bindings']) for i,f in enumerate(folds)]
        manifest['signals'] = [dict(signal_run_ref=f['signal_run_ref'],status='COMPLETE',reason=None,
            prediction_artifact_key='prediction-'+str(i),lineage_artifact_keys=['fold-'+str(i),'bindings'],
            evaluation_artifact_keys=[]) for i,f in enumerate(folds)]
        manifest.pop('content_digest');manifest = _seal(manifest)
        return frozen, sequential, batch, folds, outputs, manifest

    def test_saved_definitions_must_match_frozen_configuration_not_self_reported_ref(self):
        from axiom_research.stock_result_projection import _verify_configuration_outputs
        frozen,seq,batch,folds,_,_ = self.saved_output_fixture()
        indexed = {f['fold_ref']:f for f in folds}
        check = lambda b,f: _verify_configuration_outputs(frozen,seq,b,f,self.root)
        check(batch,indexed)
        # Each output still claims the same top-level configuration. The actual
        # owner definitions differ, as a legitimate alternate experiment would.
        for field, replacement in [('fold_specs',[{'window':7},{'window':8}]),
            ('preparation_options',{'column_block':64}),('target_spec',{'different_label':True}),
            ('model_feature_selection',{'selection':[{'id':'other','semantic_version':'1'}]})]:
            bad = deepcopy(batch);bad['definition'][field]=replacement
            with self.subTest(field=field),self.assertRaises(ExperimentRecordError):check(bad,indexed)
        for field,replacement in [('fold_spec',{'window':9}),('training_spec_ref',digest('other model')),
            ('parameters',{'learning_rate':0.8}),('model_feature_selection',{'selection':[]})]:
            bad = deepcopy(indexed);bad[folds[0]['fold_ref']]['definition'][field]=replacement
            with self.subTest(field=field),self.assertRaises(ExperimentRecordError):check(batch,bad)

    def test_false_configuration_claim_is_rejected_before_publishing(self):
        from contextlib import ExitStack
        from axiom_research.stock_result_projection import StockSavedProjection
        frozen,seq,batch,_,outputs,manifest = self.saved_output_fixture()
        # The owner has admitted a different saved preparation, while the small
        # result document still claims the original frozen configuration.
        batch['definition']['preparation_options']['column_block'] = 64
        self.assertEqual(seq['configuration_ref'],frozen['configuration']['configuration_ref'])
        def saved(path,**kwargs):
            _,wire,identity,content = outputs[Path(path).stem]
            return StockSavedProjection(wire,wire[identity],content)
        before = self.index.read_bytes(),self.index.stat().st_mtime_ns
        with ExitStack() as stack:
            for loader in ('load_stock_sequential_result','load_stock_ml_batch_manifest',
                           'load_stock_prediction_bindings','load_stock_fold_projection','load_stock_prediction_projection'):
                stack.enter_context(patch('axiom_research.stock_result_projection.'+loader,side_effect=saved))
            with self.assertRaisesRegex(ExperimentRecordError,'frozen Dataset'):
                self.store.publish_result(manifest,roots=self.roots)
        self.assertEqual((self.index.read_bytes(),self.index.stat().st_mtime_ns),before)

    def test_swapped_fold_and_signal_lineage_are_rejected_without_changing_head(self):
        from contextlib import ExitStack
        from axiom_research.stock_result_projection import StockSavedProjection
        _,_,_,_,outputs,manifest = self.saved_output_fixture()
        def saved(path,**kwargs):
            _,wire,identity,content = outputs[Path(path).stem]
            return StockSavedProjection(wire,wire[identity],content)
        with ExitStack() as stack:
            for loader in ('load_stock_sequential_result','load_stock_ml_batch_manifest',
                           'load_stock_prediction_bindings','load_stock_fold_projection','load_stock_prediction_projection'):
                stack.enter_context(patch('axiom_research.stock_result_projection.'+loader,side_effect=saved))
            before = self.index.read_bytes(),self.index.stat().st_mtime_ns
            for table,field in [('folds','artifact_keys'),('signals','lineage_artifact_keys')]:
                bad = deepcopy(manifest)
                bad[table][0][field],bad[table][1][field] = bad[table][1][field],bad[table][0][field]
                bad.pop('content_digest');bad = _seal(bad)
                with self.subTest(table=table),self.assertRaisesRegex(ExperimentRecordError,'lineage identity'):
                    self.store.publish_result(bad,roots=self.roots)
                self.assertEqual((self.index.read_bytes(),self.index.stat().st_mtime_ns),before)
            self.assertEqual(self.store.publish_result(manifest,roots=self.roots)['status'],'COMPLETE')

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

    def test_existing_derived_builder_returns_outputs_with_explicit_discovery_limitation(self):
        from axiom_research import ArtifactRef
        from axiom_research.stock_experiment_entry import build_configured_stock_sequential_experiment
        cfg=configuration()
        cfg['dataset'].update(preparation_options={},scope={},model_feature_selection=None,signal_contexts=None)
        cfg['dataset_ref']=digest(cfg['dataset'])
        cfg['configuration_ref']=digest(dict(contract_version='stock_sequential_effective_configuration_v1',
            specification_ref=cfg['specification']['configuration_ref'],dataset_ref=cfg['dataset_ref']))
        cfg['configuration_files']=[str(self.root/'config.yaml')]
        context={**self.options};context.pop('required_stages');context['request_path']=self.root/'derived-request.json'
        reference=ArtifactRef(artifact_type='StockSignalEvaluationInputs',artifact_id=digest('input'),
            artifact_contract_version='stock_signal_evaluation_inputs_v4',content_digest=digest('input'),
            uri=str(self.root/'saved-input.json'))
        output=dict(batch_manifest={'saved':'owner'},derived_signals=[{'signal_run_ref':digest('derived')}],
            prediction_bindings=[],raw_signal_evaluation_input=reference,signal_evaluation_input=reference)
        with patch('axiom_research.stock_experiment_config.load_stock_sequential_configuration',return_value=cfg), \
             patch('axiom_research.stock_experiment_entry._build_stock_sequential_experiment',return_value=output), \
             patch('axiom_research.stock_result_projection.load_stock_sequential_result',
                   side_effect=AssertionError('Derived passed to raw-only discovery')):
            result=build_configured_stock_sequential_experiment(None,configuration_path=self.root/'config.yaml',
                feature_inputs=None,column_source=None,destination=self.root/'derived'/'training',registration=context)
        self.assertIs(result,output)
        self.assertTrue((self.root/'derived'/'experiment-result.json').exists())
        head=self.reader.index()['result_heads'][0]
        self.assertEqual(head['status'],'BLOCKED')
        self.assertIn('Derived',head['reason'])

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

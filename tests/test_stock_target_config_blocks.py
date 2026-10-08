"""Small contract, native proof and shared lifetime checks; no real source run."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import gc
import weakref

from axiom_research import (LabelSpec, MaturitySpec, semantic_identity, validate,
    load_stock_experiment_specs, resolve_stock_label_spec, load_stock_feature_view)
from axiom_research.api import to_dict
from axiom_research.stock_compact_store import (feature_training_blocks, set_feature_window,
    clear_feature_window, _view_data)
from axiom_research.stock_fold_inputs import validate_spec
from test_stock_compact_v3 import CompactV3Tests


def training():
    from axiom_research import TrainingSpec,ResourceSpec,SessionRange
    from axiom_research.stock_training import LGBM_PARAMETERS
    return TrainingSpec(name='explicit-stock-model',dataset_name='stock-dataset',backend='lightgbm',
        backend_version='4.6.0',objective='regression',parameters={**LGBM_PARAMETERS,'n_estimators':100},seed=42,
        preprocessing='none',fit_scope='train_only',fit_range=SessionRange(start='2020-01-01',end='2026-09-30'),
        selection_protocol='fixed_parameters_no_validation_no_early_stopping',
        resources=ResourceSpec(threads=1,concurrent_folds=1,memory_limit_mb=512))


def label(h=5):
    return LabelSpec(contract_version='2',name='stock-open-close',key=('security_id','session'),
        horizon_sessions=h,feature_session='f',formula='close(f+h) / open(f+1) - 1',
        return_start_rule='next_session_open',return_end_rule='horizon_session_close',
        return_start_offset_sessions=1,return_end_offset_sessions=h,
        price_basis='common_anchor_adjusted_v1',benchmark_semantics='absolute_return',
        corporate_action_semantics='factor_ratio_no_separate_cashflow',
        normalization_policy='none',
        maturity=MaturitySpec(lag_sessions=h,rule='all_outcome_dependencies_strictly_before_fit_cutoff',
            calendar_policy='actual_exchange_sessions',availability_rule='max_endpoint_price_factor_anchor_usable_from'),
        missing_delisting_policy='invalid_null_preserve_grid')


class TargetConfigTests(unittest.TestCase):
    def test_concrete_dataset_yaml_rejects_clock_window_scope_and_budget_changes(self):
        import yaml
        from axiom_research import load_stock_sequential_configuration
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,path=CompactV3Tests().fixture(root)
            (root/'specs.yaml').write_text(yaml.safe_dump({'contract_version':'stock_experiment_config_v1',
                'specs':{'label':to_dict(label(3)),'model':to_dict(training())}}))
            fold=deepcopy(f.folds()[0]);fold['contract_version']='stock_ml_fold_spec_v3'
            fold['training_window']['length']=32
            dataset={'contract_version':'stock_dataset_schedule_v1','fold_specs':[fold],
                'scope':{'calendar':f.calendar,'universe':f.universe,
                    'sessions':sorted(fold['inference_cutoff_by_session']),'evaluation_cutoff':fold['evaluation_cutoff']},
                'preparation_options':{'row_block_sessions':10,'column_block':32,'maximum_resident_bytes':64*1024**2,
                    'normalization_backend':'core_cs_batch_v1'},'model_feature_selection':None,'signal_contexts':None}
            (root/'dataset.yaml').write_text(yaml.safe_dump(dataset))
            config=root/'experiment.yaml';config.write_text(yaml.safe_dump({'contract_version':'stock_sequential_configuration_v1',
                'specification_files':['specs.yaml'],'dataset_file':'dataset.yaml'}))
            saved=load_stock_sequential_configuration(config)
            self.assertEqual(saved['dataset']['fold_specs'][0]['training_window']['length'],32)
            (root/'dataset.yaml').write_text('# same declared schedule\n'+yaml.safe_dump(dataset,sort_keys=False))
            self.assertEqual(load_stock_sequential_configuration(config)['configuration_ref'],saved['configuration_ref'])
            cases=[]
            bad=deepcopy(dataset);bad['fold_specs'][0]['training_window']['length']=True;cases.append(bad)
            bad=deepcopy(dataset);bad['fold_specs'][0]['contract_version']='stock_ml_fold_spec_v1';cases.append(bad)
            bad=deepcopy(dataset);bad['scope']['sessions']=bad['scope']['sessions'][:-1];cases.append(bad)
            bad=deepcopy(dataset);bad['preparation_options']['maximum_resident_bytes']=True;cases.append(bad)
            bad=deepcopy(dataset);bad['fold_specs'][0]['simulated_model_available_at']=fold['fit_session']+'T22:00:00+08:00';cases.append(bad)
            for bad in cases:
                (root/'dataset.yaml').write_text(yaml.safe_dump(bad))
                with self.assertRaises(ValueError):load_stock_sequential_configuration(config)

    def test_experiment_references_and_explicit_overrides_resolve_effective_specs(self):
        import yaml
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);label_path=root/'label.yaml';model_path=root/'model.yaml';experiment=root/'experiment.yaml'
            label_path.write_text(yaml.safe_dump({'contract_version':'stock_experiment_config_v1',
                'specs':{'label':to_dict(label(3))}}))
            model_path.write_text(yaml.safe_dump({'contract_version':'stock_experiment_config_v1',
                'specs':{'model':to_dict(training())}}))
            wire={'contract_version':'stock_experiment_v1','files':['label.yaml','model.yaml'],
                'overrides':{'model':{'parameters':{'learning_rate':.03,'n_estimators':12}}}}
            experiment.write_text(yaml.safe_dump(wire))
            config=load_stock_experiment_specs([experiment])
            expected=replace(training(),parameters={**training().parameters,'learning_rate':.03,'n_estimators':12})
            self.assertEqual(config['spec_refs']['model'],semantic_identity(expected))
            self.assertEqual(config['specs']['model'],expected)
            self.assertEqual(config['explicit_overrides'],wire['overrides'])
            for change in ({'model':{'resources':{'threadz':1}}}, {'absent':{}},
                {'model':{'contract_version':'2'}}):
                wire['overrides']=change;experiment.write_text(yaml.safe_dump(wire))
                with self.assertRaises(ValueError):load_stock_experiment_specs([experiment])
            wire['files']=['experiment.yaml'];wire['overrides']={};experiment.write_text(yaml.safe_dump(wire))
            with self.assertRaises(ValueError):load_stock_experiment_specs([experiment])
    def test_typed_model_resolves_supplied_parameters_and_annotations_are_not_inputs(self):
        from axiom_research import resolve_stock_training_spec
        model=training();parameters,rounds,binding=resolve_stock_training_spec(model)
        self.assertEqual(rounds,100);self.assertEqual(parameters['seed'],42)
        self.assertEqual(resolve_stock_training_spec(binding['training_spec']),(parameters,rounds,binding))
        changed=replace(model,metadata={'note':'annotation'},
            fit_range=replace(model.fit_range,metadata={'note':'annotation'}))
        self.assertEqual(resolve_stock_training_spec(changed),(parameters,rounds,binding))
        changed=replace(model,parameters={**model.parameters,'learning_rate':.03,'n_estimators':12})
        p,n,b=resolve_stock_training_spec(changed)
        self.assertEqual((p['learning_rate'],n),(.03,12));self.assertNotEqual(b,binding)
        for bad in (replace(model,preprocessing='fit_new_scaler'),
            replace(model,parameters={'n_estimators':100}),
            replace(model,parameters={**model.parameters,'num_leaves':True}),
            replace(model,parameters={**model.parameters,'learning_rate':10**1000})):
            with self.assertRaises(ValueError):resolve_stock_training_spec(bad)
    def test_explicit_endpoint_horizon_preserves_old_distance_definition(self):
        for h in (1,3,5,20):
            resolved=resolve_stock_label_spec(label(h))
            self.assertEqual(resolved['horizon_sessions'],h)
            self.assertEqual(resolve_stock_label_spec(resolved['label_spec']),resolved)
        with self.assertRaises(ValueError): validate(replace(label(),contract_version='1'))
        old=replace(label(),contract_version='1',return_end_offset_sessions=6,
                    maturity=replace(label().maturity,lag_sessions=6))
        self.assertEqual(validate(old),old)
        with self.assertRaises(ValueError): resolve_stock_label_spec(old)
        for h in (True,0,-1):
            with self.assertRaises(ValueError): resolve_stock_label_spec(label(h))
        with self.assertRaisesRegex(ValueError,'strictly_before_fit_cutoff only'):
            resolve_stock_label_spec(replace(label(),maturity=replace(label().maturity,
                rule='all_outcome_dependencies_at_or_before_fit_cutoff')))

    def test_yaml_comments_paths_annotations_do_not_change_effective_specs(self):
        import yaml
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);a=root/'a.yaml';b=root/'b.yaml'
            wire=to_dict(label())
            payload={'contract_version':'stock_experiment_config_v1','specs':{'label':wire}}
            a.write_text(yaml.safe_dump(payload))
            payload['specs']['label']['metadata']={'comment':'transport annotation'}
            b.write_text('# A different filename and comment\n'+yaml.safe_dump(payload))
            x=load_stock_experiment_specs([a]);y=load_stock_experiment_specs([b])
            self.assertEqual(x['configuration_ref'],y['configuration_ref'])
            self.assertEqual(x['spec_refs']['label'],semantic_identity(label()))
            with self.assertRaises(ValueError):load_stock_experiment_specs([a,b])

    def test_yaml_rejects_unknown_duplicate_alias_and_unsafe_tag(self):
        import yaml
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'bad.yaml'
            cases=('contract_version: a\ncontract_version: b\nspecs: {}',
                'contract_version: stock_experiment_config_v1\nspecs: &s {}\nextra: *s',
                '!!python/object/apply:os.system ["echo forbidden"]')
            for text in cases:
                path.write_text(text)
                with self.assertRaises((ValueError,yaml.YAMLError)):load_stock_experiment_specs([path])
            wire=to_dict(label());wire['surprise']=1
            path.write_text(yaml.safe_dump({'contract_version':'stock_experiment_config_v1','specs':{'label':wire}}))
            with self.assertRaises(ValueError):load_stock_experiment_specs([path])


class TrainingBlockTests(unittest.TestCase):
    def test_consuming_proof_failure_clears_detached_graph_before_refunding_lease(self):
        from unittest.mock import patch
        from axiom_research.stock_compact_store import OwnedStore
        from axiom_research.stock_matrix_storage import write_part
        from axiom_research.stock_fold_inputs import seal
        from axiom_research.stock_artifacts import digest
        from axiom_research.stock_training_blocks import training_block_binding,validate_training_block_binding
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,path=CompactV3Tests().fixture(root)
            with load_stock_feature_view(path,residency='sequential') as feature,OwnedStore() as store:
                offsets=list(range(20));set_feature_window(feature,offsets)
                store.shared_bytes=_view_data(feature)['store'].resident_bytes
                header=seal({'buffers':{'validity':{'buffer_digest':digest('mask')}}},'target_ref')
                normalized={**write_part(root,header,'target_ref'),'buffers':header['buffers']}
                selector={'selector_ref':digest('selector'),'keys_digest':digest('keys')};cohort=digest('cohort')
                def fail(*args,**kwargs):
                    try:raise RuntimeError('synthetic downstream proof read failure')
                    finally:args=kwargs=None
                binding=training_block_binding(feature,offsets,normalized=normalized,cohort_ref=cohort,
                    selector=selector,store=store,destination=root/'proofs')
                calls=(lambda:training_block_binding(feature,offsets,normalized=normalized,cohort_ref=cohort,
                    selector=selector,store=store,destination=root/'proofs'),
                    lambda:validate_training_block_binding(binding,feature,offsets,normalized=normalized,
                        cohort_ref=cohort,selector=selector,store=store))
                checked=set()
                for call in calls:
                    try:
                        with patch.object(store,'read_json',side_effect=fail):call()
                    except RuntimeError as error:
                        traceback=error.__traceback__
                        while traceback:
                            name=traceback.tb_frame.f_code.co_name
                            if name in ('training_block_binding','_training_block_binding',
                                'validate_training_block_binding','_validate_training_block_binding'):
                                checked.add(name)
                                self.assertIsNone(traceback.tb_frame.f_locals['blocks'])
                                self.assertIsNone(traceback.tb_frame.f_locals['feature'])
                                self.assertIsNone(traceback.tb_frame.f_locals['store'])
                            traceback=traceback.tb_next
                    else:self.fail('downstream proof read failure expected')
                    self.assertEqual(store.lease_bytes,0)
                self.assertEqual(checked,{'training_block_binding','_training_block_binding',
                    'validate_training_block_binding','_validate_training_block_binding'})

    def test_proof_work_uses_combined_owner_budget_and_cleans_failure_aliases(self):
        from axiom_research.stock_compact_store import OwnedStore
        from axiom_research.stock_matrix_storage import write_buffer
        from axiom_research.stock_training_blocks import _proof_blocks
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,path=CompactV3Tests().fixture(root)
            with load_stock_feature_view(path,residency='sequential') as feature,OwnedStore() as main:
                offsets=list(range(20));set_feature_window(feature,offsets)
                fd=_view_data(feature);fstore=fd['store']
                previous=(fstore.limits,fstore.shared_bytes,fstore.shared_source_bytes)
                main.buffer(write_buffer(root,[1.0]*1024,dtype='float64_le',shape=[1024]))
                main.shared_bytes=fstore.resident_bytes
                main.shared_source_bytes=fstore.metrics['source_bytes']
                main.limits={**main.limits,'maximum_matrix_bytes':main.shared_bytes+main.resident_bytes+2048}
                references=[weakref.ref(a) for block in fd['blocks'] if block['parts'] is not None
                    for _,arrays in block['parts'] for a in arrays.values()]
                failure=None
                try:
                    with _proof_blocks(feature,offsets,store=main):self.fail('combined budget must reject before proof copy')
                except ValueError as error:failure=error
                self.assertIsNotNone(failure)
                self.assertEqual(main.lease_bytes,0)
                self.assertEqual((fstore.limits,fstore.shared_bytes,fstore.shared_source_bytes),previous)
                self.assertEqual(main.shared_bytes,fstore.resident_bytes)
                checked=False;traceback=failure.__traceback__
                while traceback:
                    if traceback.tb_frame.f_code.co_name=='feature_training_blocks':
                        checked=True
                        for name in ('arrays','a','parts','rows','block','cached','payload','handle'):
                            self.assertIsNone(traceback.tb_frame.f_locals[name])
                    traceback=traceback.tb_next
                self.assertTrue(checked)
                clear_feature_window(feature);gc.collect()
                self.assertTrue(all(ref() is None for ref in references))

    def test_v5_empty_checkpoint_binds_target_without_source_or_numerical_execution(self):
        from axiom_research.stock_compact_labels import _prepare_compact_incrementally
        from axiom_research.stock_artifacts import _read
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,path=CompactV3Tests().fixture(root)
            with _prepare_compact_incrementally(None,feature_inputs=path,fold_specs=f.folds()[:1],
                destination=root/'prepared',label_spec=label(3),column_source=object(),
                preparation_options={'row_block_sessions':5,'column_block':32,
                    'maximum_resident_bytes':64*1024**2,'normalization_backend':'core_cs_batch_v1'}) as owner:
                self.assertEqual(owner.definition['version'],'axiom.stock_ml_batch_inputs/5')
                self.assertEqual(owner.state.batch['contract_version'],'stock_ml_batch_inputs_v5')
                view=_read(owner.state.batch['prepared_view']['path'])
                self.assertEqual(view['contract_version'],'stock_ml_prepared_view_v4')
                self.assertEqual(view['definition']['target_spec']['horizon_sessions'],3)
                self.assertEqual(owner.stats['data_read_calls'],0)
                self.assertEqual(owner.stats['core_calls'],0)
                self.assertEqual(owner.ready,[])

    def test_configured_fold_window_and_clocks_do_not_reinterpret_old_specs(self):
        with tempfile.TemporaryDirectory() as temporary:
            f,path=CompactV3Tests().fixture(Path(temporary))
            old=f.folds()[0];training,_=validate_spec(old,f.calendar)
            self.assertEqual(len(training),65)
            configured=deepcopy(old);configured['contract_version']='stock_ml_fold_spec_v3'
            configured['training_window']['length']=12
            configured['fit_cutoff']=configured['fit_session']+'T20:50:00+08:00'
            configured['simulated_model_available_at']=configured['fit_session']+'T21:00:00+08:00'
            configured['inference_cutoff_by_session']={d:d+'T21:15:00+08:00'
                for d in configured['inference_cutoff_by_session']}
            self.assertEqual(len(validate_spec(configured,f.calendar)[0]),12)
            configured['training_window']['length']=True
            with self.assertRaises(ValueError):validate_spec(configured,f.calendar)

    def test_one_native_proof_per_window_and_release_accounts_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            f,path=CompactV3Tests().fixture(Path(temporary))
            with load_stock_feature_view(path,residency='sequential') as feature:
                offsets=list(range(20));set_feature_window(feature,offsets)
                before=_view_data(feature)['store'].resident_bytes
                first=feature_training_blocks(feature,offsets)
                store=_view_data(feature)['store'];hashes=store.metrics['file_hash_calls']
                builds=store.metrics['training_block_proof_builds'];charged=store.resident_bytes
                self.assertGreater(charged,before)
                second=feature_training_blocks(feature,offsets[2:])
                self.assertEqual(second,first)
                self.assertEqual(store.metrics['file_hash_calls'],hashes)
                self.assertEqual(store.metrics['training_block_proof_builds'],builds)
                self.assertEqual(store.resident_bytes,charged)
                clear_feature_window(feature)
                self.assertTrue(all('training_block_proofs' not in block for block in _view_data(feature)['blocks']))
                self.assertLess(store.resident_bytes,before)

    def test_selected_column_order_and_source_tamper_are_guarded(self):
        with tempfile.TemporaryDirectory() as temporary:
            f,path=CompactV3Tests().fixture(Path(temporary))
            with load_stock_feature_view(path,residency='sequential') as feature:
                offsets=list(range(20));selection=f.selection[:2]
                set_feature_window(feature,offsets,model_feature_selection=selection)
                body=feature_training_blocks(feature,offsets,model_feature_selection=selection)[0]
                self.assertEqual(body['ordered_features'],[s['id'] for s in selection])
                self.assertEqual(list(body['columns']),body['ordered_features'])
                part=_view_data(feature)['blocks'][0]['descriptors'][0]
                source=Path(part['buffers']['values']['path'])
                original=source.read_bytes();source.write_bytes(original[:-1]+bytes([original[-1]^1]))
                with self.assertRaises(ValueError):feature_training_blocks(feature,offsets,model_feature_selection=selection)


if __name__=='__main__':unittest.main()

"""Compact four-fit closure with synthetic Data and the fixed actual Core."""
from copy import deepcopy
from pathlib import Path
import builtins
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from axiom_research import (prepare_stock_ml_batch_inputs,load_stock_ml_batch_inputs,
    build_stock_ml_fold_from_saved_inputs,load_stock_ml_fold,load_stock_model)
from axiom_research.stock_artifacts import digest,file_digest,_read,write_json
from test_stock_matrix_prepare import PrepareFeatureFixture,PublicDataFixture,Query
from test_stock_folds import backend


def prepared(root, legacy=False):
    fixture=PrepareFeatureFixture(root)
    with patch('axiom_research.feature_catalog.load_feature_catalog',return_value=fixture.catalog):
        saved=fixture.build() if legacy else fixture.matrix()
    data=PublicDataFixture(fixture.spec)
    module=types.ModuleType('axiom_data'); module.QuerySpec=Query; module.adjust_prices=data.adjust
    with patch.dict(sys.modules,{'axiom_data':module}), \
         patch('axiom_research.stock_ml._implementation',return_value=fixture.implementation), \
         patch('axiom_research.stock_ml._environment',return_value=fixture.environment):
        manifest=prepare_stock_ml_batch_inputs(data,feature_inputs=saved,fold_specs=fixture.folds(),
            destination=root/'prepared',preparation_options={'row_block_sessions':32,'column_block':32,
                'maximum_resident_bytes':64*1024**2,'normalization_backend':'core_cs_batch_v1'})
    if hasattr(saved,'close'): saved.close()
    return fixture,manifest


class CompactFoldTests(unittest.TestCase):
    def build(self, fixture, manifest, root, batch=None):
        fold=manifest['folds'][0]; metrics={}
        with patch('axiom_research.feature_catalog.load_feature_catalog',return_value=fixture.catalog), \
             patch('axiom_research.stock_ml._implementation',return_value=fixture.implementation), \
             patch('axiom_research.stock_ml._environment',return_value=fixture.environment), \
             patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=backend):
            run=build_stock_ml_fold_from_saved_inputs(fold['input_manifest'],fold_spec=fold['fold_spec'],
                destination=root/'folds',metrics=metrics,batch=batch)
        return run,metrics

    def test_build_readonly_load_hit_and_one_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture,manifest=prepared(root)
            with load_stock_ml_batch_inputs(manifest) as batch:
                admitted=batch.metrics; run,metrics=self.build(fixture,manifest,root,batch)
                self.assertEqual((metrics['train_calls'],metrics['predict_calls'],metrics['core_calls']), (1,1,0))
                self.assertEqual(run.to_dict()['contract_version'],'stock_ml_fold_v3')
                self.assertEqual(_read(run.path/'manifest.json')['contract_version'],'stock_ml_fold_manifest_v2')
                dataset=_read(run.path/'dataset.json'); self.assertEqual(dataset['contract_version'],'stock_fold_dataset_v3')
                self.assertNotIn('training_keys',dataset); self.assertGreaterEqual(dataset['training_row_count'],40)
                before={p.name:file_digest(p) for p in run.path.iterdir()}
                cached,hits=self.build(fixture,manifest,root,batch)
                self.assertTrue(cached.reused); self.assertEqual(cached.identity,run.identity)
                self.assertEqual((hits['train_calls'],hits['predict_calls'],hits['core_calls']), (0,0,0))
                loaded=load_stock_ml_fold(run.path,batch=batch); self.assertEqual(loaded.predictions(),run.predictions())
                self.assertEqual({p.name:file_digest(p) for p in run.path.iterdir()},before)
                self.assertEqual(batch.metrics['file_hash_calls'],admitted['file_hash_calls'])
                self.assertEqual(batch.metrics['json_decode_calls'],admitted['json_decode_calls'])
                self.assertEqual(batch.metrics['common_key_index_builds'],1)
            real=builtins.__import__
            def blocked(name,*args,**kwargs):
                if name.startswith(('axiom_data','axiom_engine','qlib','lightgbm','pandas')):
                    raise AssertionError('readonly imported runtime '+name)
                return real(name,*args,**kwargs)
            with patch('builtins.__import__',side_effect=blocked):
                standalone=load_stock_ml_fold(run.path)
                self.assertEqual(standalone.identity,run.identity)
                self.assertEqual(load_stock_model(run.path)['contract_version'],'stock_model_release_v2')

    def test_projection_lease_budget_and_changed_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture,manifest=prepared(root); fold=manifest['folds'][0]
            batch=load_stock_ml_batch_inputs(manifest)
            with batch._matrix_project(fold['input_manifest'],fold['fold_spec']) as projection:
                self.assertEqual(projection.X.shape[0],len(projection.training_keys))
                self.assertEqual(projection.X.shape[0],len(projection.y))
                self.assertEqual(projection.P.shape[0],len(projection.candidate_keys))
                self.assertEqual(projection.X.dtype.name,'float64')
                self.assertEqual(len(projection.features['rows']),9)
                with self.assertRaisesRegex(ValueError,'borrow'): batch.close()
            desc=_read(manifest['prepared_view']['path'])['partitions'][0]['buffers']['values']
            source=Path(desc['path']); raw=source.read_bytes(); source.write_bytes(bytes([raw[0]^1])+raw[1:])
            with self.assertRaisesRegex(ValueError,'changed'): batch._check_sources()
            with self.assertRaisesRegex(ValueError,'digest mismatch'): load_stock_ml_batch_inputs(manifest)

    def test_original_v1_feature_index_is_retained_as_ancestry(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture,manifest=prepared(root,legacy=True)
            source=manifest['definition']['feature_inputs']; index=_read(source['path'])
            self.assertEqual(index['contract_version'],'stock_feature_inputs_v1')
            self.assertEqual(source['feature_inputs_ref'],index['feature_inputs_ref'])
            before={p:file_digest(p) for parent in index['feature_parents'] for p in
                (parent['features']['path'],parent['input_evidence']['path'])}
            with load_stock_ml_batch_inputs(manifest) as batch:
                run,_=self.build(fixture,manifest,root,batch)
                self.assertEqual(run.predictions()['contract_version'],'stock_prediction_run_v2')
            self.assertEqual({p:file_digest(p) for p in before},before)

    def test_failed_staged_closure_does_not_publish_complete_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture=PrepareFeatureFixture(root); saved=fixture.matrix()
            data=PublicDataFixture(fixture.spec); module=types.ModuleType('axiom_data')
            module.QuerySpec=Query; module.adjust_prices=data.adjust
            with patch.dict(sys.modules,{'axiom_data':module}), \
                 patch('axiom_research.stock_ml._implementation',return_value=fixture.implementation), \
                 patch('axiom_research.stock_ml._environment',return_value=fixture.environment), \
                 patch('axiom_research.stock_matrix_reader._validate_staged_matrix_batch',side_effect=ValueError('injected invalid closure')):
                with self.assertRaisesRegex(ValueError,'invalid closure'):
                    prepare_stock_ml_batch_inputs(data,feature_inputs=saved,fold_specs=fixture.folds(),
                        destination=root/'prepared',preparation_options={'row_block_sessions':32,'column_block':32,
                            'maximum_resident_bytes':64*1024**2,'normalization_backend':'core_cs_batch_v1'})
            self.assertEqual(list((root/'prepared').iterdir()),[])
            self.assertEqual(data.cache_clears,1); saved.close()

    def test_standalone_backend_cannot_publish_after_parent_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture,manifest=prepared(root); fold=manifest['folds'][0]
            desc=_read(manifest['prepared_view']['path'])['partitions'][0]['buffers']['values']
            def corrupting_backend(*args,**kwargs):
                source=Path(desc['path']); raw=source.read_bytes()
                source.write_bytes(bytes([raw[0]^1])+raw[1:])
                return backend(*args,**kwargs)
            with patch('axiom_research.feature_catalog.load_feature_catalog',return_value=fixture.catalog), \
                 patch('axiom_research.stock_ml._implementation',return_value=fixture.implementation), \
                 patch('axiom_research.stock_ml._environment',return_value=fixture.environment), \
                 patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=corrupting_backend):
                with self.assertRaisesRegex(ValueError,'changed'):
                    build_stock_ml_fold_from_saved_inputs(fold['input_manifest'],fold_spec=fold['fold_spec'],
                        destination=root/'folds')
            self.assertEqual(list((root/'folds').iterdir()),[])

    def test_transplanted_selector_and_saved_prediction_clock_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture,manifest=prepared(root); fold=manifest['folds'][0]
            with load_stock_ml_batch_inputs(manifest) as batch:
                inputs=deepcopy(fold['input_manifest'])
                inputs['selectors']['training']=manifest['folds'][1]['input_manifest']['selectors']['training']
                inputs['input_ref']=digest({k:v for k,v in inputs.items() if k!='input_ref'})
                with self.assertRaises(ValueError): batch._matrix_project(inputs,fold['fold_spec'])
                run,_=self.build(fixture,manifest,root,batch)
            prediction=run.predictions(); prediction['rows'][0]['feature_available_at']=None
            prediction['signal_run_ref']=digest({k:v for k,v in prediction.items() if k!='signal_run_ref'})
            write_json(run.path/'predictions.json',prediction)
            saved=run.to_dict(); saved['signal_run_ref']=prediction['signal_run_ref']
            refs={k:saved[k] for k in ('feature_ref','label_ref','dataset_ref','model_ref','signal_run_ref','evidence_ref')}
            saved['fold_ref']=digest({'definition_ref':saved['definition_ref'],**refs})
            saved['content_digest']=digest({k:v for k,v in saved.items() if k!='content_digest'}); write_json(run.path/'fold.json',saved)
            saved_manifest=_read(run.path/'manifest.json'); saved_manifest['fold_ref']=saved['fold_ref']
            for name in ('predictions.json','fold.json'): saved_manifest['files'][name]=file_digest(run.path/name)
            write_json(run.path/'manifest.json',saved_manifest)
            with self.assertRaisesRegex(ValueError,'source/clock'): load_stock_ml_fold(run.path)


if __name__=='__main__': unittest.main()

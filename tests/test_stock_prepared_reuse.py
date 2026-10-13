"""Small two-fold prepared reuse and live proof descriptor mutation checks."""
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from axiom_research import (build_configured_stock_sequential_experiment, load_stock_feature_view,
    load_stock_ml_batch_inputs, load_stock_sequential_configuration, load_stock_model, predict_stock_model)
from axiom_research.stock_artifacts import _read, file_digest, write_json
from axiom_research.stock_compact_store import OwnedStore, _view_data, clear_feature_window, feature_training_blocks
from axiom_research.stock_training_blocks import _training_block_descriptor
from test_stock_shared_execution import real_configuration
import test_stock_compact_v3 as fixtures


def variant(root, original, *, rate=None, subset=None):
    root.mkdir()
    for name in ('specs.yaml','dataset.yaml','experiment.yaml'):
        value=yaml.safe_load((original.parent/name).read_text())
        if name=='specs.yaml':
            model=value['specs']['model']['parameters']; model['n_estimators']=2
            if rate is not None:model['learning_rate']=rate
        if name=='dataset.yaml' and subset is not None:value['model_feature_selection']=subset
        (root/name).write_text(yaml.safe_dump(value,sort_keys=False))
    return root/'experiment.yaml'


class PreparedReuseTests(unittest.TestCase):
    def test_descriptor_reuse_keeps_bytes_marks_and_window_charge(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); _,path=fixtures.CompactV3Tests().fixture(root)
            with load_stock_feature_view(path,residency='sequential') as feature,OwnedStore() as store:
                fstore=_view_data(feature)['store']; calls=[]
                from axiom_research.stock_training_blocks import write_part
                def observed(*args,**kwargs):
                    calls.append(args[1]['feature_block_ref']); return write_part(*args,**kwargs)
                with patch('axiom_research.stock_training_blocks.write_part',side_effect=observed):
                    with fstore.operation(),store.operation():
                        body=feature_training_blocks(feature,[0])[0]
                        first=_training_block_descriptor(feature,body,store,root/'proofs')
                    before=file_digest(first['path']),Path(first['path']).stat().st_mtime_ns
                    store.release_payloads()
                    with fstore.operation(),store.operation():
                        again=_training_block_descriptor(feature,body,store,root/'proofs')
                    self.assertEqual(first,again); self.assertEqual(len(calls),1)
                    self.assertEqual(before,(file_digest(first['path']),Path(first['path']).stat().st_mtime_ns))
                    self.assertEqual(store.json[first['path']],body)
                    with self.assertRaisesRegex(ValueError,'compact source changed'),fstore.operation(),store.operation():
                        _training_block_descriptor(feature,body,store,root/'proofs')
                        payload=Path(first['path']).read_bytes()
                        Path(first['path']).write_bytes(bytes([payload[0]^1])+payload[1:])
                        store.validate_boundary()
                    self.assertEqual(len(calls),1)
                clear_feature_window(feature)
                self.assertTrue(all('training_block_proof_descriptors' not in b for b in _view_data(feature)['blocks']))

    def test_changed_model_consumes_complete_batch_and_subset_reprepares_its_cohort(self):
        from axiom_research.stock_training import fit_predict_stock_model
        phases={}; phase='first'; proofs=Counter()
        from axiom_research.stock_training_blocks import write_part
        def observed_proof(*args,**kwargs):
            proofs[phase]+=1; return write_part(*args,**kwargs)
        def observed_backend(X,y,P,**kwargs):
            result=fit_predict_stock_model(X,y,P,**kwargs)
            phases.setdefault(phase,[]).append(dict(X=X.copy(),y=y.copy(),P=P.copy(),
                ordered_features=list(kwargs['ordered_features']),parameters=deepcopy(kwargs['parameters']),
                num_boost_round=kwargs['num_boost_round'],predictions=result[1].copy(),metrics=dict(kwargs['metrics'])))
            return result
        evidence=os.environ.get('AXIOM_PREPARED_REUSE_EVIDENCE_ROOT')
        directory=nullcontext(evidence) if evidence else tempfile.TemporaryDirectory()
        with directory as temp:
            root=Path(temp); root.mkdir(parents=True,exist_ok=True)
            data,snapshot,fixture,feature_path,original=real_configuration(root)
            available={item['id']:item for item in fixture.catalog.select(fixture.selection)}
            fixture.catalog.select=lambda selection:[deepcopy(available[item['id']]) for item in selection]
            first_cfg=variant(root/'first-config',original,rate=.1)
            second_cfg=variant(root/'second-config',original,rate=.2)
            subset_cfg=variant(root/'subset-config',original,rate=.2,subset=fixture.spec['feature_selection'][:2])
            def registration(name):
                return dict(request_path=root/name/'.request.json',index_path=root/'index.json',roots={'research':root})
            first_stats={}; reuse_stats={}; subset_stats={}; control_stats={}; progress=[]
            with patch('axiom_research.feature_catalog.load_feature_catalog',return_value=fixture.catalog), \
                 patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=observed_backend), \
                 patch('axiom_research.stock_training_blocks.write_part',side_effect=observed_proof), \
                 load_stock_feature_view(feature_path,residency='sequential') as feature, \
                 data.open_column_source(snapshot=snapshot,limits={'cache_bytes':16*1024**2,'max_working_bytes':64*1024**2}) as source:
                first=build_configured_stock_sequential_experiment(data,configuration_path=first_cfg,feature_inputs=feature,
                    column_source=source,destination=root/'first-run/training',prepared_destination=root/'shared-prepared',
                    metrics=first_stats,registration=registration('first-run'),progress=progress.append)
                self.assertEqual([e['completed'] for e in progress if e['stage']=='sequential_fold'],[1,2])
                self.assertEqual(proofs['first'],1)
                self.assertGreater(first_stats['shared_execution']['feature']['training_block_descriptor_reuses'],0)
                pinned={p:(file_digest(p),p.stat().st_mtime_ns) for p in (root/'shared-prepared').rglob('*') if p.is_file()}
                with load_stock_ml_batch_inputs(first['batch_manifest'],feature_inputs=feature,residency='sequential') as prepared:
                    phase='reuse'
                    feature_result=root/'feature-result.json';write_json(feature_result,{'path':str(feature_path)})
                    output=root/'second-run';output.mkdir()
                    batch_path=Path(first['batch_manifest']['prepared_view']['path']).parent/'batch.json'
                    options=dict(experiment_file=str(second_cfg),feature_result=str(feature_result),prepared_batch=str(batch_path))
                    execute=runpy.run_path(str(Path(__file__).parents[1]/'examples/a01_one_year.py'))['execute']
                    with patch.object(source,'select',side_effect=AssertionError('reused model selected Data')), \
                         patch('axiom_data.Data',side_effect=AssertionError('reused model constructed Data')):
                        execute(SimpleNamespace(stage='train'),options,load_stock_sequential_configuration(second_cfg),
                            output,registration('second-run'))
                    from axiom_research.result_registration import _registered_sequential_path
                    second=_read(_registered_sequential_path(**registration('second-run')))
                    reuse_stats=second['metrics']
                    self.assertEqual(second['batch_manifest'],first['batch_manifest'])
                    self.assertEqual(proofs['reuse'],0)
                    self.assertEqual(reuse_stats['raw_operator_calls'],0); self.assertEqual(reuse_stats['core_calls'],0)
                    for saved,a,b in zip(second['folds'],phases['first'],phases['reuse']):
                        for key in ('X','y','P'):np.testing.assert_array_equal(a[key],b[key])
                        self.assertNotEqual(a['parameters'],b['parameters'])
                        model=load_stock_model(saved['path'],batch=prepared)
                        self.assertEqual(model['contract_version'],'stock_model_release_v3')
                        independent=predict_stock_model(saved['path'],[{'values':r.tolist()} for r in b['P']],
                            ordered_features=model['ordered_features'],feature_selection=model['feature_selection'],batch=prepared)
                        np.testing.assert_array_equal(b['predictions'],independent)
                    # A cold public call follows the same complete fold admission.
                    cold_model=load_stock_model(second['folds'][0]['path'])
                    self.assertEqual(cold_model,load_stock_model(second['folds'][0]['path'],batch=prepared))
                    self.assertEqual(reuse_stats['sequential']['train_calls'],2)
                    self.assertEqual(reuse_stats['sequential']['predict_calls'],2)
                    self.assertAlmostEqual(sum(reuse_stats['sequential']['stages'].values()),
                                           reuse_stats['sequential']['total_seconds'],places=6)
                    with self.assertRaisesRegex(ValueError,'selection/cohort differs'):
                        build_configured_stock_sequential_experiment(None,configuration_path=subset_cfg,feature_inputs=feature,
                            column_source=None,destination=root/'invalid-subset/training',prepared_batch=prepared,
                            registration=registration('invalid-subset'))
                    self.assertEqual(len(phases['reuse']),2)
                    phase='subset'
                    with patch.object(source,'select',side_effect=AssertionError('subset failed to reuse Raw')):
                        subset=build_configured_stock_sequential_experiment(data,configuration_path=subset_cfg,feature_inputs=feature,
                            column_source=source,destination=root/'subset-run/training',prepared_destination=root/'shared-prepared',
                            reuse_raw_from_batch=prepared,metrics=subset_stats,registration=registration('subset-run'))
                    self.assertEqual(subset_stats['raw_operator_calls'],0)
                    self.assertGreater(subset_stats['core_calls'],0)
                    phase='control'
                    control=build_configured_stock_sequential_experiment(data,configuration_path=subset_cfg,feature_inputs=feature,
                        column_source=source,destination=root/'control-run/training',prepared_destination=root/'control-prepared',
                        metrics=control_stats,registration=registration('control-run'))
                    for a,b in zip(phases['subset'],phases['control']):
                        for key in ('X','y','P','predictions'):np.testing.assert_array_equal(a[key],b[key])
                    for old,new in zip(first['batch_manifest']['folds'],subset['batch_manifest']['folds']):
                        a=_read(old['input_manifest']['fold_control']['path']);b=_read(new['input_manifest']['fold_control']['path'])
                        self.assertEqual(a['raw_parts'],b['raw_parts']); self.assertEqual(a['evaluation_parts'],b['evaluation_parts'])
                        self.assertNotEqual(a['cohort_ref'],b['cohort_ref'])
                    self.assertEqual(pinned,{p:(file_digest(p),p.stat().st_mtime_ns) for p in pinned})
            if evidence:
                write_json(root/'acceptance.json',dict(status='PASS_SMALL_PREPARED_REUSE',synthetic=True,
                    fold_count=2,proof_write_calls=dict(proofs),reuse_preparation_metrics=reuse_stats,
                    subset_preparation_metrics=subset_stats,first_preparation_metrics=first_stats,
                    fit_predict_calls={k:len(v) for k,v in phases.items()},
                    independent_public_v3_predict_calls=2,normal_CLI_complete_batch_reuse=True,
                    same_model_inputs_exact=True,reused_predictions_match_saved_model_exact=True,
                    subset_matches_fresh_preparation_exact=True,old_prepared_hash_mtime_unchanged=True,
                    real_Data_reads=0,supplier_calls=0,account_calls=0))

    def test_raw_feature_stage_is_rejected_before_v3_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            data,snapshot,fixture,path,config=real_configuration(root,feature_stage='feature')
            with patch('axiom_research.feature_catalog.load_feature_catalog',return_value=fixture.catalog), \
                 patch('axiom_research.stock_training.fit_predict_stock_model') as fit, \
                 data.open_column_source(snapshot=snapshot,limits={'cache_bytes':16*1024**2,'max_working_bytes':64*1024**2}) as source:
                with self.assertRaisesRegex(ValueError,'raw training is unsupported'):
                    build_configured_stock_sequential_experiment(data,configuration_path=config,feature_inputs=path,
                        column_source=source,destination=root/'raw-run/training')
                fit.assert_not_called()
            self.assertFalse(list((root/'raw-run').rglob('booster.txt')))


if __name__=='__main__':unittest.main()

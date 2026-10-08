"""Bounded source/identity acceptance; artificial Data and the original fake model."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import numpy as np

from axiom_research import load_stock_ml_fold,load_stock_ml_batch_inputs,load_stock_feature_view
from axiom_research.stock_artifacts import _read,write_json,file_digest
from axiom_research.stock_batch import _data
from axiom_research.stock_compact_labels import (_prepare_compact_incrementally,
    _load_compact_checkpoint_owner,_prepare_legacy_compact_batch)
from axiom_research.stock_signal_evaluation_build import _freeze_build_oos_inputs
from axiom_research.stock_fold_inputs import seal
import test_stock_compact_v3 as features
import test_stock_sequential_windows as models
from test_stock_matrix_prepare import PublicDataFixture,Query

OPTIONS={'row_block_sessions':5,'column_block':32,'maximum_resident_bytes':64*1024**2,
    'normalization_backend':'core_cs_batch_v1'}


class Data(PublicDataFixture):
    def adjust(self,*args,**kwargs):
        value=super().adjust(*args,**kwargs)
        value.wire['context']['derivation']['factor_domain']='adjustment_factors'
        return value


@contextmanager
def data_module(data):
    module=types.ModuleType('axiom_data');module.QuerySpec=Query;module.adjust_prices=data.adjust
    with patch.dict(sys.modules,{'axiom_data':module}):yield


def controls(f):
    return {'spec':f.spec,'calendar':f.calendar,'universe':f.universe,'columns':f.columns,
        'folds':f.folds()[:2],'implementation':f.implementation,'environment':f.environment,
        'catalog_row':f.catalog.select(f.selection)[0]}


def restore_fixture(control):
    f=SimpleNamespace(**control)
    f.catalog=SimpleNamespace(identity=f.spec['catalog_ref'],
        select=lambda selection:[{**f.catalog_row,'id':c} for c in f.columns])
    return f


def scope(control):
    return {'calendar':control['calendar'],'universe':control['universe'],
        'sessions':sorted({d for f in control['folds'] for d in f['inference_cutoff_by_session']}),
        'evaluation_cutoff':control['folds'][0]['evaluation_cutoff']}


def snapshots(runs,report,manifest):
    return {'stages':[{p.name:file_digest(p) for p in run.path.iterdir()} for run in runs],
        'refs':[run.identity for run in runs],'manifest':manifest,
        'report_root':_read(report.uri),'report_dates':{
            d:file_digest(Path(report.uri).parent/(d+'.json')) for d in scope({'calendar':[],
                'universe':[],'folds':[f['fold_spec'] for f in manifest['folds']]})['sessions']}}


def execute(root,control,*,stop=None):
    f=restore_fixture(control);data=Data(f.spec);fit_calls=[];preparation={};runs=[]
    def fit(*args,**kwargs):
        fit_calls.append(True);return models.backend(*args,**kwargs)
    with data_module(data),_prepare_compact_incrementally(data,feature_inputs=root/'feature-table',
        fold_specs=control['folds'],destination=root/'prepared',preparation_options=OPTIONS,metrics=preparation) as owner:
        target=owner.target
        with _freeze_build_oos_inputs(owner.batch,scope=scope(control),destination=root/'oos',signal_name='model') as writer:
            while (item:=owner.next_fold()) is not None:
                runs.append(models.SequentialWindowTests().build(f,item,owner.batch,root/'folds',fit=fit))
                if stop is not None and len(runs)==stop:
                    assert not (target/'batch.json').exists()
                    try:owner.batch.to_dict()
                    except ValueError:pass
                    else:raise AssertionError('partial owner exposed COMPLETE')
                    try:writer.finish()
                    except ValueError:pass
                    else:raise AssertionError('partial OOS published')
                    break
            else:
                manifest=owner.finish();report=writer.finish()
                result=snapshots(runs,report,manifest)
        if stop is not None:
            result={'target':str(target),'first_path':str(runs[0].path),'first_ref':runs[0].identity,
                'stage_hashes':{p.name:file_digest(p) for p in runs[0].path.iterdir()}}
    result['fit_calls']=len(fit_calls);result['prepare_metrics']=preparation
    result['Data_calls']=len(data.queries)
    return result


class IncrementalCheckpointTests(unittest.TestCase):
    def fixture(self,root):
        f,path=features.CompactV3Tests().fixture(root)
        control=controls(f);write_json(root/'fixture.json',control)
        return f,control

    def test_continuous_vs_stop_first_then_new_process_exact_and_old_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,control=self.fixture(root)
            golden=execute(root,control)
            self.assertEqual(golden['fit_calls'],2)
            for name in ('prepared','folds','oos'):shutil.rmtree(root/name)
            prefix=execute(root,control,stop=1)
            self.assertEqual(prefix['fit_calls'],1)
            self.assertLess(prefix['Data_calls'],golden['Data_calls'])
            with _load_compact_checkpoint_owner(prefix['target']) as owner, \
                 patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=AssertionError('readonly fit')), \
                 patch('axiom_engine.core.execute_cs_zscore_batch',side_effect=AssertionError('readonly normalization')):
                self.assertEqual(load_stock_ml_fold(prefix['first_path'],batch=owner).identity,prefix['first_ref'])
            child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'resume',str(root)],
                env=os.environ.copy(),capture_output=True,text=True,timeout=30)
            self.assertEqual(child.returncode,0,child.stdout+child.stderr)
            resumed=_read(root/'resumed.json')
            for key in ('stages','refs','manifest','report_root','report_dates'):self.assertEqual(resumed[key],golden[key],key)
            self.assertEqual(resumed['fit_calls'],1)
            self.assertEqual(resumed['prepare_metrics']['label_core_calls'],1)
            self.assertEqual(resumed['Data_calls'],golden['Data_calls']-prefix['Data_calls'])
            self.assertEqual({p.name:file_digest(p) for p in Path(prefix['first_path']).iterdir()},prefix['stage_hashes'])
            with load_stock_ml_batch_inputs(resumed['manifest'],residency='sequential') as batch:
                self.assertEqual(batch.to_dict(),resumed['manifest'])
                for index,p in enumerate(sorted((root/'folds').iterdir())):
                    self.assertIn(load_stock_ml_fold(p,batch=batch).identity,resumed['refs'])

    def test_original_eager_math_matches_shared_incremental_primitive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,control=self.fixture(root);data=Data(f.spec)
            with data_module(data):
                original=_prepare_legacy_compact_batch(data,feature_inputs=root/'feature-table',fold_specs=control['folds'],
                    destination=root/'prepared',preparation_options=OPTIONS)
            with load_stock_ml_batch_inputs(original,residency='sequential') as batch:
                runs=[models.SequentialWindowTests().build(f,item,batch,root/'folds') for item in original['folds']]
                original_stages=[{p.name:file_digest(p) for p in run.path.iterdir()} for run in runs]
                original_refs=[run.identity for run in runs]
            shutil.rmtree(root/'prepared')
            shutil.rmtree(root/'folds')
            with data_module(data),_prepare_compact_incrementally(data,feature_inputs=root/'feature-table',
                fold_specs=control['folds'],destination=root/'prepared',preparation_options=OPTIONS) as owner:
                runs=[]
                while (item:=owner.next_fold()) is not None:
                    runs.append(models.SequentialWindowTests().build(f,item,owner.batch,root/'folds'))
                self.assertEqual(owner.finish(),original)
                self.assertEqual([{p.name:file_digest(p) for p in run.path.iterdir()} for run in runs],original_stages)
                self.assertEqual([run.identity for run in runs],original_refs)

    def test_corrupt_checkpoint_or_source_rejects_without_data_or_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,control=self.fixture(root);prefix=execute(root,control,stop=1)
            path=Path(prefix['target'])/'checkpoint.json';original=path.read_bytes()
            changed=_read(path);changed['folds']=[]
            write_json(path,changed)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                with _load_compact_checkpoint_owner(prefix['target']):pass
            path.write_bytes(original)
            changed=_read(path)
            self.assertIsNotNone(changed['raw_outputs'][1]['evaluation_parts'][0])
            changed['raw_outputs'][1]['evaluation_parts'][0]=None
            write_json(path,seal({k:v for k,v in changed.items() if k!='checkpoint_ref'},'checkpoint_ref'))
            with self.assertRaisesRegex(ValueError,'unfinished checkpoint price domain'):
                with _load_compact_checkpoint_owner(prefix['target']):pass
            path.write_bytes(original)
            desc=_read(_read(path)['folds'][0]['input_manifest']['fold_control']['path'])['normalized']
            target=_read(desc['path']);bad=Path(target['buffers']['values']['path'])
            bad.write_bytes(b'\x01'*bad.stat().st_size)
            with self.assertRaisesRegex(ValueError,'digest mismatch'):
                with _load_compact_checkpoint_owner(prefix['target']):pass
            self.assertFalse((Path(prefix['target'])/'batch.json').exists())

    def test_raw_reuse_prepares_whole_touched_final_domain_and_resumes_without_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,control=self.fixture(root);data=Data(f.spec)
            options={**OPTIONS,'row_block_sessions':64}
            with data_module(data):
                source_manifest=_prepare_legacy_compact_batch(data,feature_inputs=root/'feature-table',fold_specs=control['folds'],
                    destination=root/'source',preparation_options=options)
            with load_stock_feature_view(root/'feature-table',residency='sequential') as feature, \
                 load_stock_ml_batch_inputs(source_manifest,feature_inputs=feature,residency='sequential') as source, \
                 patch('axiom_research.stock_compact_labels._price_view',side_effect=AssertionError('Raw reuse Data')), \
                 patch('axiom_research.stock_compact_labels._raw',side_effect=AssertionError('Raw reuse operator')):
                args=dict(feature_inputs=feature,fold_specs=control['folds'],destination=root/'reuse',
                    preparation_options=options,model_feature_selection=f.selection[:1],reuse_raw_from_batch=source)
                with _prepare_compact_incrementally(None,**args) as owner:
                    first=owner.next_fold();target=owner.target
                    checkpoint=_read(owner.checkpoint)
                    self.assertTrue(all(d is not None for out in checkpoint['raw_outputs'] for d in out['evaluation_parts']))
                    self.assertTrue(all(d is None for d in checkpoint['raw_outputs'][1]['raw_parts']))
                    self.assertFalse((target/'batch.json').exists())
                with _load_compact_checkpoint_owner(target,feature_inputs=feature) as saved:
                    self.assertEqual(_data(saved)['manifest']['folds'],[first])
                metrics={}
                with _prepare_compact_incrementally(None,**args,metrics=metrics) as owner:
                    self.assertEqual(owner.next_fold(),first)
                    self.assertIsNotNone(owner.next_fold());self.assertIsNone(owner.next_fold())
                    complete=owner.finish()
                self.assertEqual((metrics['data_read_calls'],metrics['raw_operator_calls'],metrics['label_core_calls']),(0,0,1))
                with load_stock_ml_batch_inputs(complete,feature_inputs=feature,residency='sequential') as saved:
                    self.assertEqual(saved.to_dict(),complete)

    def test_failed_backend_leaves_no_complete_model_or_temporary_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,control=self.fixture(root);data=Data(f.spec)
            with data_module(data),_prepare_compact_incrementally(data,feature_inputs=root/'feature-table',
                fold_specs=control['folds'],destination=root/'prepared',preparation_options=OPTIONS) as owner:
                item=owner.next_fold();state=_data(owner.batch)['matrix_state']
                def fail(*args,**kwargs):raise RuntimeError('bounded failed backend')
                with self.assertRaisesRegex(RuntimeError,'failed backend'):
                    models.SequentialWindowTests().build(f,item,owner.batch,root/'folds',fit=fail)
                self.assertEqual((state.active,state.store.borrowers,state.store.lease_bytes),(0,0,0))
                self.assertFalse((owner.target/'batch.json').exists())
                self.assertFalse((root/'folds').exists())
                self.assertEqual(list(owner.target.glob('.checkpoint-*')),[])

    def test_final_control_and_source_gates_run_before_publication(self):
        for gate in ('budget','source'):
            with self.subTest(gate=gate),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);f,control=self.fixture(root);data=Data(f.spec)
                with data_module(data),_prepare_compact_incrementally(data,feature_inputs=root/'feature-table',
                    fold_specs=control['folds'],destination=root/'prepared',preparation_options=OPTIONS) as owner:
                    while owner.next_fold() is not None:pass
                    target=owner.target;checkpoint=owner.checkpoint.read_bytes();state=owner.state
                    size=state._controls_size;check=state.check;final=[False]
                    def final_size(manifest):
                        if 'batch_ref' in manifest:
                            final[0]=True
                            if gate=='budget':raise ValueError('final control byte budget exceeded')
                        return size(manifest)
                    def final_check():
                        if final[0] and gate=='source':raise ValueError('final source changed')
                        return check()
                    with patch.object(state,'_controls_size',new=final_size),patch.object(state,'check',new=final_check):
                        with self.assertRaisesRegex(ValueError,'final (control|source)'):owner.finish()
                    self.assertTrue(state.incomplete)
                    self.assertFalse((target/'batch.json').exists())
                    self.assertEqual(owner.checkpoint.read_bytes(),checkpoint)
                    self.assertEqual(list(target.glob('.checkpoint-*')),[])
                    with self.assertRaisesRegex(ValueError,'no complete batch identity'):owner.batch.identity
                with _load_compact_checkpoint_owner(target) as saved:
                    self.assertEqual(len(_data(saved)['manifest']['folds']),2)

    def test_checkpoint_parent_budget_rejects_before_raw_admission(self):
        for filename in ('checkpoint.json','view.json','definition.json'):
            with self.subTest(filename=filename),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);f,control=self.fixture(root);data=Data(f.spec)
                options={**OPTIONS,'maximum_parent_bytes':512*1024}
                with data_module(data),_prepare_compact_incrementally(data,feature_inputs=root/'feature-table',
                    fold_specs=control['folds'],destination=root/'prepared',preparation_options=options) as owner:
                    owner.next_fold();target=owner.target
                path=target/filename;path.write_bytes(path.read_bytes()+b'\n'*(513*1024))
                with patch('axiom_research.stock_compact_labels._normalized',side_effect=AssertionError('resume Core')), \
                     patch('axiom_research.stock_compact_batch.read_target',side_effect=AssertionError('resume Raw')):
                    with self.assertRaisesRegex(ValueError,'(parent|plan) byte budget exceeded'):
                        with _load_compact_checkpoint_owner(target):pass
                self.assertFalse((target/'batch.json').exists())

    def test_resume_failure_traceback_drops_new_plan_and_admission_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,control=self.fixture(root);data=Data(f.spec)
            with data_module(data),_prepare_compact_incrementally(data,feature_inputs=root/'feature-table',
                fold_specs=control['folds'],destination=root/'prepared',preparation_options=OPTIONS) as owner:
                item=owner.next_fold();target=owner.target
            normalized=_read(item['input_manifest']['fold_control']['path'])['normalized']
            values=Path(_read(normalized['path'])['buffers']['values']['path'])
            values.write_bytes(b'\x01'*values.stat().st_size)
            failure=None
            try:
                with _load_compact_checkpoint_owner(target):pass
            except ValueError as error:failure=error
            self.assertIsNotNone(failure)
            expected={
                '__init__':('fold_specs','preparation_options','_saved_definition','training','inference','jobs','chunks',
                    'evaluation','calendar_positions','group','days','fold','feature_inputs','data'),
                '_load_checkpoint_state':('state','store','feature_inputs','publication_store','binding','fold','definition','folds'),
                '_load_compact_checkpoint_owner':('definition','saved','source','selection','options','admission','owner'),
                '_prepare_compact_incrementally':('data','owner')}
            checked=set();trace=failure.__traceback__
            while trace is not None:
                frame=trace.tb_frame;name=frame.f_code.co_name
                if name in expected and 'stock_compact_' in frame.f_code.co_filename:
                    for key in expected[name]:self.assertIsNone(frame.f_locals.get(key),(name,key))
                    if name=='__init__':
                        closed=frame.f_locals['self'];self.assertTrue(closed.closed)
                        self.assertIsNone(closed.plans);self.assertIsNone(closed.state)
                    if name=='_prepare_compact_incrementally':self.assertEqual(frame.f_locals['kwargs'],{})
                    checked.add(name)
                trace=trace.tb_next
            self.assertEqual(checked,set(expected))


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='resume':
        root=Path(sys.argv[2]);write_json(root/'resumed.json',execute(root,_read(root/'fixture.json')))
    else:unittest.main()

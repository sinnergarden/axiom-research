"""Small shared-owner lifecycle acceptance, including real public Data/fit.

All facts are synthetic; the user's data, suppliers and accounts are untouched.
Run with the public Data fixture and Engine source on PYTHONPATH, as for the
existing public Data column-contract tests.
"""
from collections import Counter,defaultdict
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from axiom_research.stock_artifacts import _read,digest,file_digest,write_json
from axiom_research.stock_compact_store import OwnedStore,_view_data,load_stock_feature_view
from axiom_research.stock_compact_batch import ConcatRows,TargetRows,target_eligibility_reasons
from axiom_research.stock_matrix_storage import instant_us,write_buffer
from test_stock_compact_v3 import CompactV3Tests
from test_stock_matrix_prepare import PrepareFeatureFixture
from test_stock_target_config_blocks import label,training


def real_configuration(root):
    import yaml
    from column_source_fixture import prepared
    from axiom_research import SessionRange,to_dict
    prototype=root/'prototype';prototype.mkdir();axes=PrepareFeatureFixture(prototype)
    def facts(domains,rows):
        for domain in ('market_daily','adjustment_factors'):
            groups=defaultdict(list)
            for i,day in enumerate(axes.calendar):
                for j,security in enumerate(axes.universe):
                    values=({'open':10.+i*.04+j*.3,'close':11.+i*.03+j*j*.17}
                        if domain=='market_daily' else {'factor':1.+i*.001+j*.01})
                    groups[day[:7]].append(dict(security_id=security,session=day,revision_id='r1',revision_sequence=1,
                        raw_batch_id='synthetic-shared-owner',first_observed_at=day+'T08:00:00.123456Z',
                        source_available_at=None,evidence_ref=None,**values))
            domains[domain]['partitions']=[]
            for month,records in groups.items():
                uri=domain+'/'+month+'.parquet';rows[uri]=records
                domains[domain]['partitions'].append(dict(partition=month,uri=uri,rows=len(records),file_sha256='0'*64))
        rows['public_evidence/history.parquet'][0].update(target_revision='r1',
            target_key=json.dumps({'security_id':'A','session':axes.calendar[0]},sort_keys=True,separators=(',',':')),
            public_at=axes.calendar[0]+'T08:15:00.123456Z')
    data,snapshot=prepared(root/'data',modify=facts)
    class JoinedFeature(PrepareFeatureFixture):
        def __init__(self,target):
            super().__init__(target);self.snapshot=snapshot;self.spec['snapshot']=snapshot
            scope=_read(target/'scope.json');scope['request']['snapshot']=snapshot
            scope['request_ref']=digest(scope['request']);scope['result']['snapshot_id']=snapshot
            scope['result_ref']=digest(scope['result']);scope.pop('scope_bundle_ref');scope['scope_bundle_ref']=digest(scope)
            write_json(target/'scope.json',scope)
            self.spec['scope'].update(file_digest=file_digest(target/'scope.json'),scope_bundle_ref=scope['scope_bundle_ref'])
    fixture,feature=CompactV3Tests().fixture(root,feature_class=JoinedFeature)
    folds=deepcopy(fixture.folds()[:2])
    for fold in folds:fold['contract_version']='stock_ml_fold_spec_v3'
    model=replace(training(),backend_version=importlib.metadata.version('lightgbm'),
        fit_range=SessionRange(start=fixture.calendar[0],end=fixture.calendar[-1]))
    scope=dict(calendar=fixture.calendar,universe=fixture.universe,
        sessions=sorted({day for fold in folds for day in fold['inference_cutoff_by_session']}),
        evaluation_cutoff=folds[0]['evaluation_cutoff'])
    documents={'specs.yaml':dict(contract_version='stock_experiment_config_v1',specs=dict(label=to_dict(label(5)),model=to_dict(model))),
        'dataset.yaml':dict(contract_version='stock_dataset_schedule_v1',fold_specs=folds,scope=scope,
            preparation_options=dict(row_block_sessions=10,column_block=32,maximum_resident_bytes=64*1024**2,
                normalization_backend='core_cs_batch_v1'),model_feature_selection=None,signal_contexts=None),
        'experiment.yaml':dict(contract_version='stock_sequential_configuration_v1',specification_files=['specs.yaml'],dataset_file='dataset.yaml')}
    for name,value in documents.items():(root/name).write_text(yaml.safe_dump(value,sort_keys=False))
    return data,snapshot,fixture,feature,root/'experiment.yaml'


class SharedExecutionTests(unittest.TestCase):
    def test_real_public_two_fold_train_save_oos_and_exact_resume(self):
        from axiom_research import build_configured_stock_sequential_experiment,load_stock_ml_fold
        from axiom_data.column_source import ColumnSelection
        from axiom_research.stock_training import fit_predict_stock_model
        admissions=Counter();backend_metrics=[];prepare_metrics={}
        row_dicts=Counter();getitem=TargetRows.__getitem__
        def observed_row(rows,index):
            if rows.value['contract_version']=='stock_compact_raw_v2':
                row_dicts[rows.value['definition'].get('cutoff')]+=1
            return getitem(rows,index)
        original=OwnedStore.read
        def observed(store,descriptor,**kwargs):
            value=original(store,descriptor,**kwargs)
            if value is not None:admissions[descriptor['path']]+=1
            return value
        def backend(*args,**kwargs):
            result=fit_predict_stock_model(*args,**kwargs);backend_metrics.append(dict(kwargs['metrics']));return result
        evidence_root=os.environ.get('AXIOM_SHARED_SYNTHETIC_ACCEPTANCE_ROOT')
        directory=nullcontext(evidence_root) if evidence_root else tempfile.TemporaryDirectory()
        with directory as temp:
            root=Path(temp);data,snapshot,fixture,feature,config=real_configuration(root)
            with patch('axiom_research.feature_catalog.load_feature_catalog',return_value=fixture.catalog), \
                patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=backend), \
                patch.object(ColumnSelection,'to_batch',side_effect=AssertionError('legacy Data conversion')), \
                patch.object(OwnedStore,'read',observed), \
                patch.object(TargetRows,'__getitem__',observed_row), \
                data.open_column_source(snapshot=snapshot,limits={'cache_bytes':16*1024**2,'max_working_bytes':64*1024**2}) as source:
                started=time.monotonic()
                result=build_configured_stock_sequential_experiment(data,configuration_path=config,feature_inputs=feature,
                    column_source=source,destination=root/'experiment',metrics=prepare_metrics)
                first_seconds=time.monotonic()-started
                self.assertEqual(len(backend_metrics),2)
                self.assertTrue(all(m['train_calls']==m['predict_calls']==1 for m in backend_metrics))
                self.assertEqual(len(result['folds']),2)
                self.assertEqual(prepare_metrics['columnar_normalized_windows'],2)
                self.assertGreater(prepare_metrics['normalized_daily_reuses'],0)
                shared=prepare_metrics['shared_execution']
                self.assertGreater(shared['targets']['target_array_borrows'],0)
                self.assertGreater(shared['targets']['fold_window_borrows'],0)
                self.assertGreater(shared['feature']['training_block_proof_borrows'],0)
                self.assertEqual(shared['feature']['training_block_proof_builds'],1)
                self.assertEqual(shared['feature']['feature_eligibility_column_gathers'],2)
                self.assertEqual(shared['targets']['produced_fold_window_borrows'],2)
                self.assertTrue(all(count==1 for path,count in admissions.items() if Path(path).is_relative_to(feature)))
                for fold in result['batch_manifest']['definition']['fold_specs']:
                    self.assertEqual(row_dicts[fold['fit_cutoff']],0)
                pins={str(p):file_digest(p) for fold in result['folds'] for p in Path(fold['path']).iterdir()}
                before=dict(source.statistics)
                with patch.object(source,'select',side_effect=AssertionError('resume selected Data')), \
                    patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=AssertionError('resume trained')):
                    reused=build_configured_stock_sequential_experiment(data,configuration_path=config,feature_inputs=feature,
                        column_source=source,destination=root/'experiment')
                    for fold in reused['folds']:load_stock_ml_fold(fold['path'])
                self.assertEqual(source.statistics,before)
                self.assertEqual(reused['folds'],result['folds'])
                self.assertEqual(pins,{p:file_digest(p) for p in pins})
                if evidence_root:
                    write_json(root/'acceptance.json',dict(status='PASS_SMALL_SYNTHETIC_SHARED_EXECUTION',synthetic=True,
                        real_Data_root_reads=0,supplier_calls=0,feature_executor_calls=0,account_calls=0,
                        fold_count=2,first_fit_calls=2,first_predict_calls=2,resume_Data_select_calls=0,
                        resume_fit_calls=0,resume_predict_calls=0,feature_file_admissions_once=True,
                        training_raw_row_dictionary_projections=0,first_wall_seconds=first_seconds,
                        Data_statistics=dict(source.statistics),Research_metrics=prepare_metrics,backend_metrics=backend_metrics,
                        folds=result['folds'],batch_ref=result['batch_manifest']['batch_ref'],
                        frozen_oos_ref=__import__('axiom_research').to_dict(result['raw_signal_evaluation_input']),
                        saved_fold_file_digests=pins,unchanged_on_resume=True,
                        limitations=['Small synthetic correctness/count acceptance only; no long-history performance PASS.',
                            'OwnedStore admissions count logical hashing/decoding, not all writer byte-verification IO.']))

    def test_maturity_equality_keeps_scalar_not_member_precedence(self):
        from axiom_research.stock_target_spec import resolve_stock_label_spec,eligible_target_reason
        from axiom_research.stock_label_contracts import _instant
        from axiom_research.stock_compact_store import iter_feature_eligibility
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def changed(day,rows):
                for row in rows:row['member']=True
                rows[0]['member']=False
            fixture,path=CompactV3Tests().fixture(root,transform=changed)
            day=fixture.days[0];cutoff=fixture.days[2]+'T20:30:00+08:00';width=len(fixture.universe)
            calendar=fixture.calendar;pos=calendar.index(day)
            immutable=lambda a:np.frombuffer(a.tobytes(),dtype=a.dtype)
            arrays={name:immutable(np.full(width,value,dtype=dtype)) for name,value,dtype in (
                ('values',.1,'<f8'),('validity',1,'u1'),('availability',instant_us(cutoff),'<i8'),
                ('availability_validity',1,'u1'),('reason_codes',0,'<i4'),('start_session',pos+1,'<i4'),
                ('end_session',pos+2,'<i4'),('source_codes',0,'<i4'))}
            raw=ConcatRows([TargetRows(dict(contract_version='stock_compact_raw_v2',row_count=width,
                definition=dict(calendar=calendar,sessions=[day],universe=fixture.universe),
                reason_dictionary=[None],source_dictionary=[[digest('source')]]),arrays)])
            common={**fixture.spec,'target_spec':resolve_stock_label_spec(label(2))}
            with load_stock_feature_view(path,residency='sequential') as feature:
                actual=target_eligibility_reasons(raw,feature,list(range(width)),cutoff,common)
                expected=[eligible_target_reason(row,facts,len(fixture.columns),_instant(cutoff),common)
                    for row,facts in zip(raw,iter_feature_eligibility(feature,list(range(width))))]
            self.assertEqual(actual,expected);self.assertEqual(actual,['NOT_MEMBER','LABEL_NOT_MATURE','LABEL_NOT_MATURE'])

    def test_nested_lease_checks_changed_file_and_revokes_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);descriptor=write_buffer(root,[1.],dtype='float64_le',shape=[1])
            with OwnedStore() as store:
                store.buffer(descriptor)
                with self.assertRaisesRegex(ValueError,'source changed'):
                    with store.operation():
                        for _ in range(20):
                            with store.operation():store.buffer(descriptor)
                        path=Path(descriptor['path']);path.write_bytes(path.read_bytes()[:-1]+b'!')
                        store.validate_boundary()
                with self.assertRaises(ValueError):
                    with store.operation():pass

    def test_cold_target_admission_rejects_resealed_stale_cohort(self):
        from axiom_research.stock_compact_batch import read_target
        from axiom_research.stock_fold_inputs import seal
        from axiom_research.stock_label_contracts import NORMALIZATION_SPEC
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);day='2024-01-02';cutoff=day+'T20:30:00+08:00'
            cohort=dict(contract_version='stock_compact_cohort_v1',sessions=[day],universe=['A'],cutoff=cutoff,
                raw_refs=[digest('raw')],feature_view_ref=digest('feature'),eligibility_reasons=[None],eligible_keys=[['A',day]])
            definition=dict(calendar=[day],sessions=[day],universe=['A'],cutoff=cutoff,raw_refs=cohort['raw_refs'],
                cohort_ref=digest(cohort),normalization_spec=NORMALIZATION_SPEC)
            stale=deepcopy(cohort);stale['eligible_keys']=[]
            buffers={name:write_buffer(root,values,dtype=dtype,shape=[1]) for name,values,dtype in (
                ('values',[1.],'float64_le'),('validity',[True],'bool_u8'),('availability',[instant_us(cutoff)],'int64_le'),
                ('availability_validity',[True],'bool_u8'),('reason_codes',[0],'int32_le'))}
            # Both byte digest and top content ref are valid. The nested cohort
            # still differs from the logical identity and must be rejected.
            value=seal(dict(contract_version='stock_compact_normalized_v1',definition=definition,definition_ref=digest(definition),
                row_count=1,reason_dictionary=[None],source_dictionary=[],buffers=buffers,core_ref=digest('core'),cohort=stale),'target_ref')
            write_json(root/'target.json',value)
            descriptor=dict(path=str(root/'target.json'),file_digest=file_digest(root/'target.json'),target_ref=value['target_ref'])
            with OwnedStore() as store,self.assertRaisesRegex(ValueError,'cohort differs'):
                read_target(store,descriptor)

    def test_feature_mutation_during_build_blocks_publication_and_cleans_owner(self):
        import sys
        from types import ModuleType
        from axiom_research import build_stock_ml_fold_from_saved_inputs
        from axiom_research.stock_compact_labels import _prepare_compact_incrementally
        from test_stock_column_asof import ColumnSource,Query
        from test_stock_sequential_windows import backend
        module=ModuleType('axiom_data');module.QuerySpec=Query
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fixture,path=CompactV3Tests().fixture(root);source=ColumnSource(fixture.spec)
            model=replace(training(),fit_range=__import__('axiom_research').SessionRange(
                start=fixture.calendar[0],end=fixture.calendar[-1]))
            fixture.environment={'synthetic':'fixed environment','packages':{'lightgbm':'4.6.0'}}
            def changed(*args,**kwargs):
                result=backend(*args,**kwargs)
                index=path/'index.json';index.write_bytes(index.read_bytes()+b' ')
                return result
            with patch.dict(sys.modules,{'axiom_data':module}), \
                patch('axiom_research.feature_catalog.load_feature_catalog',return_value=fixture.catalog), \
                patch('axiom_research.stock_ml._environment',return_value=fixture.environment), \
                patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=changed):
                with _prepare_compact_incrementally(object(),feature_inputs=path,fold_specs=fixture.folds()[:1],
                    destination=root/'prepared',label_spec=label(5),column_source=source,
                    preparation_options=dict(row_block_sessions=10,column_block=32,maximum_resident_bytes=64*1024**2,
                        normalization_backend='core_cs_batch_v1')) as owner:
                    state=owner.state;feature_store=_view_data(owner.feature,check=False)['store']
                    with self.assertRaisesRegex(ValueError,'source changed'):
                        with owner.operation():
                            item=owner.next_fold()
                            record=state._parts(item['input_manifest'],item['fold_spec'])
                            value,owned=state.targets[record['normalized']['target_ref']]
                            clone=TargetRows(value,owned.arrays,owned.raw_rows)
                            with self.assertRaisesRegex(ValueError,'not this owner'):
                                state._borrow_produced_fold(item,record,clone)
                            clone=owned=value=record=None
                            build_stock_ml_fold_from_saved_inputs(item['input_manifest'],fold_spec=item['fold_spec'],
                                destination=root/'folds',batch=owner.batch,training_spec=model)
                    self.assertEqual(state.active,0);self.assertEqual(state.store.lease_bytes,0)
                    self.assertTrue(feature_store._invalid)
                    self.assertFalse(list((root/'folds').glob('*/fold.json')))
            self.assertEqual(state.store.resident_bytes,0);self.assertEqual(feature_store.resident_bytes,0)


if __name__=='__main__':unittest.main()

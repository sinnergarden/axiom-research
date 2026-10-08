"""Artificial public column ABI; never reads the user's Data root."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
import sys
import tempfile
import unittest
import gc
import weakref
import numpy as np

from axiom_research import load_stock_ml_batch_inputs,load_stock_ml_fold
from axiom_research.stock_artifacts import digest,_read
from axiom_research.stock_column_inputs import query_binding,ColumnPriceDomain
from axiom_research.stock_compact_labels import _prepare_compact_incrementally
from axiom_research.stock_batch import _data
from axiom_research.stock_matrix_storage import instant_us
from test_stock_compact_v3 import CompactV3Tests
from test_stock_matrix_prepare import Query
from test_stock_sequential_windows import SequentialWindowTests
from test_stock_target_config_blocks import label,training


def readonly(a):return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


class Borrow:
    def __init__(self,a):self.a=a
    def to_numpy(self,*,dtype=None):return readonly(np.asarray(self.a,dtype=dtype))


class Selection:
    contract_version='data_column_selection_v1'
    def __init__(self,source,query,*,adjusted=False,anchor=None):
        self.query=query;self.snapshot_ref=source.spec['snapshot'];self.query_binding=query_binding(query)
        self.axes={'sessions':query.sessions,'security':query.symbols,'fields':query.fields};self.closed=False
        shape=len(query.sessions),len(query.symbols);width=shape[1]
        positions=np.asarray([source.spec['calendar'].index(d) for d in query.sessions])
        grid=positions[:,None]*width+np.arange(width)[None,:]
        self.source_block_refs=(digest('synthetic-price-block'),digest('synthetic-factor-block'))
        index=np.stack((np.zeros(shape,dtype='<i8'),grid),axis=-1).astype('<i8')
        if adjusted:
            factor=index.copy();factor[...,0]=1;anchor_factor=factor.copy()
            anchor_factor[...,1]=source.spec['calendar'].index(anchor)*width+np.arange(width)[None,:]
            self.selected_version_indices={k:Borrow(a) for k,a in
                [('price',index),('factor',factor),('anchor_factor',anchor_factor)]}
        else:self.selected_version_indices={'native':Borrow(index)}
        self.revision_binding={'indices':{'index:'+k:digest(v.a.tolist()) for k,v in self.selected_version_indices.items()},
            'source_block_refs':list(self.source_block_refs)}
        self.source_binding={'snapshot_ref':self.snapshot_ref,'domain':query.domain,
            'source_block_refs':list(self.source_block_refs),'contract_ref':digest(query.domain)}
        self.evidence_binding={'source_block_refs':[digest('synthetic-evidence')],
            'selected_versions':deepcopy(self.revision_binding)}
        self.derivation=({'recipe_version':'common_anchor_price_v1','formula':'price_t * factor_t / factor_anchor',
            'anchor_session':anchor,'decision_session':anchor,'factor_field':'factor',
            'decision_cutoff':self.query_binding['cutoff_by_session'][anchor],
            'price_selection_ref':digest('price selection'),'factor_selection_ref':digest('factor selection')}
            if adjusted else None)
        self.selection_ref=digest({'query':self.query_binding,'revision':self.revision_binding})
        clocks=np.asarray([[np.datetime64(d+'T08:15:00.123456','ns') for _ in query.symbols] for d in query.sessions])
        if adjusted:clocks=np.maximum(clocks,np.datetime64(anchor+'T08:15:00.123456','ns'))
        self.columns={}
        for field in query.fields:
            values=np.ones(shape) if field=='factor' else np.broadcast_to(
                (10.0 if field=='open' else 11.0)+positions[:,None]*.01+
                (np.zeros((1,width)) if field=='open' else np.arange(width)[None,:]**2*.4),shape)
            self.columns[field]=SimpleNamespace(metadata={'price_basis':query.price_basis,'basis':query.price_basis,
                'dtype':'float64','unit':'CNY/share','recipe_version':'common_anchor_price_v1'},values=Borrow(values),
                validity=Borrow(np.ones(shape,dtype='?')),available_at=Borrow(clocks),missing_reason=None)
    def close(self):self.closed=True


class ColumnSource:
    def __init__(self,spec):self.spec=spec;self.select_calls=0;self.adjust_calls=0;self.selections=[]
    def select(self,*,query,previous=None):
        if previous is not None:assert not previous.closed
        self.select_calls+=1;v=Selection(self,query);self.selections.append(v);return v
    def adjust(self,prices,factors,*,fields,anchor_session,factor_field,decision_session,previous=None):
        if previous is not None:assert not previous.closed
        self.adjust_calls+=1
        v=Selection(self,replace(prices.query,fields=fields,price_basis='common_anchor_adjusted_v1',
            adjustment_anchor=anchor_session),adjusted=True,anchor=anchor_session)
        self.selections.append(v);return v


def forward_golden(start,end,*,endpoint_validity):
    """Test oracle for the unchanged divide-then-subtract order, not production."""
    out=np.zeros(start.shape,dtype='<f8');flags=endpoint_validity.copy()
    with np.errstate(all='ignore'):
        out[flags]=end[flags]/start[flags]
        out[flags]=out[flags]-1.0
    flags &= np.isfinite(out);out[~flags]=0.0
    return {'values':readonly(out),'validity':readonly(flags)}


class ColumnAsOfTests(unittest.TestCase):
    def test_joint_ingress_rejects_before_json_decode_and_selects_raw_once(self):
        from axiom_research import stock_signal_evaluation_derived as joint
        from axiom_research import stock_signal_evaluation_projection as projection
        from axiom_research.stock_derived_signal import load_stock_derived_signal
        freeze=joint.save_stock_derived_signal_evaluation_inputs;checked=[]
        def verify(raw,**kwargs):
            with patch('axiom_research.stock_compact_store.json.loads',
                side_effect=AssertionError('budget must reject before decoding')):
                with self.assertRaisesRegex(ValueError,'byte budget'):
                    freeze(raw,**{**kwargs,'destination':Path(kwargs['destination']).parent/'tiny-joint',
                        'maximum_resident_bytes':1})
                with self.assertRaisesRegex(ValueError,'byte budget'):
                    load_stock_derived_signal(next(iter(kwargs['derived_inputs'].values()))[0],
                        limits={'maximum_matrix_bytes':1,'maximum_parent_bytes':1})
            saved=freeze(raw,**kwargs)
            with patch.object(projection,'_select_inputs',side_effect=AssertionError('Raw mask must not be built twice')), \
                patch.object(joint,'_select_inputs',wraps=joint._select_inputs) as selector:
                projection._load_inputs(saved,kwargs['scope'])
                self.assertEqual(selector.call_count,1)
            low=replace(saved,metadata={'maximum_resident_bytes':1})
            with patch('axiom_research.stock_compact_store.json.loads',side_effect=AssertionError('joint root decoded')):
                with self.assertRaisesRegex(ValueError,'byte budget'):projection._load_inputs(low,kwargs['scope'])
            checked.append(saved.artifact_id);return saved
        with patch.object(joint,'save_stock_derived_signal_evaluation_inputs',side_effect=verify):
            self.test_public_sequence_entry_and_actual_raw_evaluation()
        self.assertEqual(len(checked),1)

    def test_joint_sample_allocation_checks_shared_budget_before_selector(self):
        from axiom_research import stock_signal_evaluation_derived as joint
        from axiom_research.stock_signal_evaluation_projection import _load_inputs
        from axiom_research.stock_compact_store import OwnedStore,_size
        freeze=joint.save_stock_derived_signal_evaluation_inputs;checked=[]
        def verify(raw,**kwargs):
            saved=freeze(raw,**kwargs)
            _,root,selected,admission=_load_inputs(saved,kwargs['scope'],include_admission=True)
            with OwnedStore(joint._limits(root['maximum_resident_bytes'])) as store:
                store.shared_bytes=_size([root,selected,admission])
                # A caller has already retained the admitted inputs and leaves
                # only 64 KiB for two sample tables, masks and sorting.
                store.limits={**store.limits,'maximum_matrix_bytes':store.shared_bytes+65536}
                with patch.object(joint,'_select_inputs',side_effect=AssertionError('unbudgeted sample allocation')):
                    with self.assertRaisesRegex(ValueError,'byte budget'):
                        joint._select_joint_inputs(admission,kwargs['scope'],store)
            checked.append(saved.artifact_id);return saved
        with patch.object(joint,'save_stock_derived_signal_evaluation_inputs',side_effect=verify):
            self.test_public_sequence_entry_and_actual_raw_evaluation()
        self.assertEqual(len(checked),1)

    def test_joint_raw_shard_swap_after_staged_validation_blocks_publication(self):
        from axiom_research import stock_signal_evaluation_derived as joint
        freeze=joint.save_stock_derived_signal_evaluation_inputs;load=joint._load_joint_inputs;checked=[]
        def verify(raw,**kwargs):
            manifest=_read(raw.uri);descriptor=manifest['shards'][kwargs['scope']['sessions'][0]]
            shard=Path(raw.uri).parent/descriptor['file'];original=shard.read_bytes();mutated=[]
            def replace_shard(ref,scope,**options):
                result=load(ref,scope,**options)
                if options.get('_validate_only') and not mutated:
                    shard.write_bytes(original+b' ');mutated.append(True)
                return result
            race=Path(kwargs['destination']).parent/'joint-publication-race'
            try:
                with patch.object(joint,'_load_joint_inputs',side_effect=replace_shard), \
                    patch.object(joint.os,'rename',side_effect=AssertionError('changed Raw shard reached publication')):
                    with self.assertRaisesRegex(ValueError,'frozen evaluation file changed'):
                        freeze(raw,**{**kwargs,'destination':race})
                self.assertEqual(mutated,[True]);self.assertFalse(list(race.glob('*/manifest.json')))
            finally:shard.write_bytes(original)
            checked.append(True);return freeze(raw,**kwargs)
        with patch.object(joint,'save_stock_derived_signal_evaluation_inputs',side_effect=verify):
            self.test_public_sequence_entry_and_actual_raw_evaluation()
        self.assertEqual(checked,[True])

    def test_joint_year_budget_counts_real_saved_report_payloads_before_selector(self):
        from axiom_research import stock_signal_evaluation_derived as joint
        from axiom_research.stock_signal_evaluation_projection import _load_inputs,evaluate_stock_signal_input_periods
        from axiom_research.stock_compact_store import _size
        freeze=joint.save_stock_derived_signal_evaluation_inputs;checked=[]
        def verify(raw,**kwargs):
            saved=freeze(raw,**kwargs)
            _,root,selected,admission=_load_inputs(saved,kwargs['scope'],include_admission=True)
            reports=evaluate_stock_signal_input_periods(saved,scope=kwargs['scope'],
                destination=Path(kwargs['destination']).parent/'payload-budget-reports')
            owner_report=next(iter(reports['all'].values()))
            # Keep a genuine public saved-report object, with a large retained
            # descriptive payload. Counting its dataclass shell cannot see it.
            body=deepcopy(owner_report._verified)
            body['limitations'].append('synthetic retained report text '*131072)
            object.__setattr__(owner_report,'_verified',body)
            maximum=_size([root,selected,admission])+1024**2
            with patch.object(joint,'_select_inputs',side_effect=AssertionError('report payload escaped year budget')), \
                patch.object(type(owner_report),'to_dict',side_effect=AssertionError('budget copied the saved report')):
                with self.assertRaisesRegex(ValueError,'accounting workspace budget'):
                    joint._select_joint_period(admission,kwargs['scope'],maximum=maximum,
                        retained_graph=[root,admission,selected],saved_results=reports)
            checked.append(saved.artifact_id);return saved
        with patch.object(joint,'save_stock_derived_signal_evaluation_inputs',side_effect=verify):
            self.test_public_sequence_entry_and_actual_raw_evaluation()
        self.assertEqual(len(checked),1)

    def test_public_second_fold_core_failure_releases_warm_normalization_vectors(self):
        from axiom_research import open_stock_ml_batch_preparation
        module=ModuleType('axiom_data');module.QuerySpec=Query
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,path=CompactV3Tests().fixture(root);source=ColumnSource(f.spec)
            references=[];failure=None
            def fail(carrier,**kwargs):
                try:raise RuntimeError('synthetic second-fold normalization failure')
                finally:carrier=None
            try:
                with patch.dict(sys.modules,{'axiom_data':module}),open_stock_ml_batch_preparation(object(),feature_inputs=path,
                    fold_specs=f.folds()[:2],destination=root/'prepared',label_spec=label(3),column_source=source,
                    preparation_options={'row_block_sessions':10,'column_block':32,'maximum_resident_bytes':64*1024**2,
                        'normalization_backend':'core_cs_batch_v1'}) as owner:
                    owner.next_fold()
                    references=[weakref.ref(entry[name]) for entry in owner.column_targets.normalized.values()
                        for name in ('values','validity')]
                    self.assertTrue(references)
                    with patch('axiom_engine.core.execute_cs_zscore_batch',side_effect=fail):owner.next_fold()
            except RuntimeError as error:failure=error
            self.assertIsNotNone(failure)
            checked=False;traceback=failure.__traceback__
            while traceback:
                if traceback.tb_frame.f_code.co_name=='_normalized':
                    checked=True
                    for name in ('daily','cached','vals','flags','result','feature','metrics','rows'):
                        self.assertIsNone(traceback.tb_frame.f_locals[name])
                traceback=traceback.tb_next
            self.assertTrue(checked)
            gc.collect();self.assertTrue(all(ref() is None for ref in references))
            self.assertTrue(all(selection.closed for selection in source.selections))

    def test_public_v5_excludes_outcome_exactly_at_fit_without_rejecting_fold(self):
        from axiom_research import open_stock_ml_batch_preparation
        from axiom_research.stock_compact_store import OwnedStore
        from axiom_research.stock_compact_batch import read_target
        from axiom_research.stock_matrix_storage import instant_us
        module=ModuleType('axiom_data');module.QuerySpec=Query
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,path=CompactV3Tests().fixture(root);fold=f.folds()[0]
            fit_cutoff=fold['fit_cutoff'];equality_day=f.calendar[f.calendar.index(fold['fit_session'])-3]
            class EqualitySource(ColumnSource):
                def adjust(self,*args,**kwargs):
                    selected=super().adjust(*args,**kwargs)
                    if selected.query_binding['cutoff_by_session'][selected.query.sessions[0]]==query_binding(
                        replace(selected.query,cutoff_by_session={d:fit_cutoff for d in selected.query.sessions}))['cutoff_by_session'][selected.query.sessions[0]]:
                        row=selected.query.sessions.index(fold['fit_session'])
                        for name in ('open','close'):
                            clocks=selected.columns[name].available_at.a.copy()
                            clocks[row,0]=np.datetime64(instant_us(fit_cutoff),'us').astype('datetime64[ns]')
                            selected.columns[name].available_at=Borrow(clocks)
                    return selected
            source=EqualitySource(f.spec)
            with patch.dict(sys.modules,{'axiom_data':module}),open_stock_ml_batch_preparation(object(),feature_inputs=path,
                fold_specs=[fold],destination=root/'prepared',label_spec=label(3),column_source=source,
                preparation_options={'row_block_sessions':10,'column_block':32,'maximum_resident_bytes':64*1024**2,
                    'normalization_backend':'core_cs_batch_v1'}) as owner:
                item=owner.next_fold();self.assertIsNotNone(item)
                control=_read(item['input_manifest']['fold_control']['path'])
                with OwnedStore() as store:
                    raw=[]
                    for part in control['raw_parts']:
                        _,rows=read_target(store,part);raw.extend(rows)
                    normalized,rows=read_target(store,control['normalized'],raw_rows=raw)
                    index=next(i for i,row in enumerate(raw) if row['feature_session']==equality_day and
                        row['security_id']==f.universe[0])
                    self.assertEqual(instant_us(raw[index]['label_available_at']),instant_us(fit_cutoff))
                    self.assertEqual(normalized['cohort']['eligibility_reasons'][index],'LABEL_NOT_MATURE')
                    self.assertFalse(rows[index]['valid'])
                    self.assertGreater(control['training_binding']['training_row_count'],40)
                self.assertEqual(owner.finish()['status'],'COMPLETE')

    def test_public_sequence_entry_and_actual_raw_evaluation(self):
        import yaml
        from axiom_research import (build_configured_stock_sequential_experiment,evaluate_stock_sequential_signals,
            SessionRange,to_dict,load_stock_signal_evaluation,SignalPlanSpec,SignalInput,SignalNode)
        from axiom_research.stock_label_contracts import NORMALIZATION_SPEC
        from test_stock_sequential_windows import backend
        module=ModuleType('axiom_data');module.QuerySpec=Query
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,path=CompactV3Tests().fixture(root);source=ColumnSource(f.spec)
            model=replace(training(),fit_range=SessionRange(start=f.calendar[0],end=f.calendar[-1]))
            plan=SignalPlanSpec(name='configured-original-raw-zscore',key=('security_id','session'),
                inputs=(SignalInput(alias='raw',label=label(3),source_stage='raw_prediction',
                    score_semantics='forward_3_session_cs_zscore_prediction'),),
                nodes=(SignalNode(name='z',op='daily_zscore',inputs=('raw',),input_stages=('raw_prediction',),
                    output_stage='daily_zscore',weights=(),reference_universe='saved-fixture-members',
                    missing_policy='skip',parameters=NORMALIZATION_SPEC['params']),),output='z',
                join_policy='inner_on_security_session',score_semantics='fixture-zscore',
                available_time_semantics='max_original_dependencies')
            config=root/'model.yaml';config.write_text(yaml.safe_dump({'contract_version':'stock_experiment_config_v1',
                'specs':{'label':to_dict(label(3)),'model':to_dict(model),'signal':to_dict(plan)}}))
            f.environment={'synthetic':'fixed environment','packages':{'lightgbm':'4.6.0'}}
            folds=deepcopy(f.folds()[:1]);fold=folds[0];fold['contract_version']='stock_ml_fold_spec_v3'
            fold['training_window']['length']=32
            fold['fit_cutoff']=fold['fit_session']+'T20:50:00+08:00'
            fold['simulated_model_available_at']=fold['fit_session']+'T21:00:00+08:00'
            fold['inference_cutoff_by_session']={day:day+'T21:15:00+08:00' for day in fold['inference_cutoff_by_session']}
            scope={'calendar':f.calendar,'universe':f.universe,
                'sessions':sorted(folds[0]['inference_cutoff_by_session']),'evaluation_cutoff':folds[0]['evaluation_cutoff']}
            context={'calendar_ref':digest({'contract_version':'stock_label_calendar_v1','sessions':f.calendar}),
                'reference_universe':'saved-fixture-members','reference_universe_ref':digest('frozen-fixture-members'),
                'reference_members':{day:[{'security_id':row['security_id'],'member':row['member'],
                    'available_at':row['knowledge_cutoff'],'source_refs':row['source_refs']}
                    for row in f.day(day)[0]] for day in scope['sessions']},
                'cutoff_by_session':deepcopy(fold['inference_cutoff_by_session']),'clock_basis':'declared_simulation'}
            dataset=root/'dataset.yaml';dataset.write_text(yaml.safe_dump({'contract_version':'stock_dataset_schedule_v1',
                'fold_specs':folds,'scope':scope,'preparation_options':{'row_block_sessions':10,'column_block':32,
                    'maximum_resident_bytes':64*1024**2,'normalization_backend':'core_cs_batch_v1'},
                'model_feature_selection':None,'signal_contexts':{digest(fold):context}}))
            configured=root/'experiment.yaml';configured.write_text(yaml.safe_dump({
                'contract_version':'stock_sequential_configuration_v1','specification_files':['model.yaml'],
                'dataset_file':'dataset.yaml'}))
            with patch.dict(sys.modules,{'axiom_data':module}), \
                patch('axiom_research.feature_catalog.load_feature_catalog',return_value=f.catalog), \
                patch('axiom_research.stock_ml._environment',return_value=f.environment), \
                patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=backend):
                experiment=build_configured_stock_sequential_experiment(object(),configuration_path=configured,
                    feature_inputs=path,column_source=source,destination=root/'experiment')
            self.assertEqual(experiment['batch_manifest']['status'],'COMPLETE')
            self.assertEqual(experiment['prediction_bindings'][0]['kind'],'derived')
            self.assertEqual(experiment['prediction_bindings'][0]['parent_inputs']['raw']['signal_run_ref'],
                experiment['folds'][0]['signal_run_ref'])
            self.assertTrue(all(v.closed for v in source.selections))
            reports=evaluate_stock_sequential_signals(experiment,scope=scope,destination=root/'evaluation')
            self.assertEqual(set(reports),{'all','by_year'})
            self.assertEqual(set(reports['all']),{'model','derived'})
            self.assertEqual(reports['all']['model'].to_dict()['contract_version'],'stock_signal_evidence_v7')
            with patch('axiom_engine.core.evaluate_signal_statistics',side_effect=AssertionError('readonly statistics')):
                self.assertEqual(load_stock_signal_evaluation(reports['all']['model'].path).identity,
                    reports['all']['model'].identity)
                self.assertTrue(evaluate_stock_sequential_signals(experiment,scope=scope,destination=root/'evaluation')
                    ['all']['model'].reused)

    def test_core_failure_drops_producer_native_aliases_and_cache_charge(self):
        from axiom_research.stock_column_inputs import ColumnMathReuse
        from axiom_research.stock_compact_store import OwnedStore,limits
        with OwnedStore() as store,OwnedStore() as feature_store:
            metrics={'_store':store,'_feature_store':feature_store,'_limits':limits(),
                '_caller_bytes':0,'_caller_source_bytes':0,'_retained_raw_bytes':0,'maximum_working_bytes':0}
            reuse=ColumnMathReuse(metrics)
            def failing_core(a,b,**kwargs):raise RuntimeError('synthetic kernel failure')
            a=readonly(np.ones((2,2)));valid=readonly(np.ones(a.shape,dtype='?'))
            try:
                reuse.forward_block([('training','a',3),('training','b',3)],a,a,valid,
                    anchor='x',dependency_refs=[digest('a'),digest('b')],operator=failing_core)
            except RuntimeError as exc:
                tb=exc.__traceback__
                while tb:
                    if tb.tb_frame.f_code.co_name=='forward_block':
                        for name in ('a','b','mask','result','values','flags','vals','flags_day','opening','closing','valid'):
                            self.assertIsNone(tb.tb_frame.f_locals[name])
                    tb=tb.tb_next
            else:self.fail('expected synthetic kernel failure')
            self.assertEqual(reuse.raw,{})
            reuse.close();self.assertEqual(metrics['_column_math_bytes'],0)

    def test_fixed_data_candidate_public_arrays_match_old_forward_order(self):
        # Only the Data owner's checked-in tiny Parquet fixture is copied to
        # this temporary directory. The user's root is never opened.
        from column_source_fixture import prepared,LIMITS,EARLY
        from axiom_data import QuerySpec,adjust_prices
        from axiom_data.column_source import ColumnSelection
        from axiom_research.stock_target_spec import resolve_stock_label_spec
        from axiom_research.stock_column_inputs import validate_column_raw_binding
        from axiom_research.labels import _forward_rows
        with tempfile.TemporaryDirectory() as temporary:
            data,snapshot=prepared(Path(temporary))
            q=QuerySpec(domain='market_daily',fields=('open','close'),symbols=('A','B','C'),
                sessions=('2024-01-03',),pit_policy='operational_pit_v1',
                cutoff_by_session={'2024-01-03':EARLY},purpose='label_outcomes',price_basis='unadjusted')
            fq=replace(q,domain='adjustment_factors',fields=('factor',))
            legacy=adjust_prices(data.read(snapshot=snapshot,query=q),data.read(snapshot=snapshot,query=fq),
                fields=('open','close'),anchor_session='2024-01-03',decision_session='2024-01-03',factor_field='factor').to_json()
            spec={'snapshot':snapshot,'universe':['A','B','C'],'calendar':['2024-01-02','2024-01-03','2024-01-05'],
                'feature_sessions':['2024-01-02'],'pit_policy':'operational_pit_v1',
                'target_spec':resolve_stock_label_spec(label(1))}
            with data.open_column_source(snapshot=snapshot,limits=LIMITS) as source, \
                patch.object(ColumnSelection,'to_batch',side_effect=AssertionError('Research legacy materialization')):
                prices=source.select(query=q);factors=source.select(query=fq)
                adjusted=source.adjust(prices,factors,fields=('open','close'),anchor_session='2024-01-03',
                    decision_session='2024-01-03',factor_field='factor')
                with ColumnPriceDomain(adjusted,spec=spec,query=replace(q,price_basis='common_anchor_adjusted_v1',
                    adjustment_anchor='2024-01-03'),anchor='2024-01-03',
                    input_queries={'price':prices.query_binding,'factor':factors.query_binding}) as domain:
                    from axiom_engine.core import execute_forward_returns
                    actual=list(domain.rows(['2024-01-02'],horizon=1,forward_operator=execute_forward_returns))
                    expected=list(_forward_rows(legacy['records'],legacy['field_meta'],legacy['context'],
                        calendar=spec['calendar'],features=spec['feature_sessions'],horizon_sessions=1,source_ref=domain.source_ref))
                    self.assertEqual(actual,expected)
                    definition={**{k:spec[k] for k in ('snapshot','universe','calendar')},'sessions':spec['feature_sessions'],
                        'price_view':domain.source,'cutoff':EARLY,'formula':'close(f+1) / open(f+1) - 1',
                        'horizon_sessions':1,'start_session_offset':1,'end_session_offset':1,
                        'label_definition_ref':spec['target_spec']['label_definition_ref'],
                        'price_basis':'common_anchor_adjusted_v1','missing_policy':'invalid_null_preserve_grid'}
                    validate_column_raw_binding(definition,spec,EARLY)

    def test_raw_cache_one_batch_changed_day_and_anchor_recompute(self):
        from axiom_research.stock_column_inputs import ColumnMathReuse
        from axiom_research.stock_compact_store import OwnedStore,limits
        with OwnedStore() as store,OwnedStore() as feature_store:
            metrics={'_store':store,'_feature_store':feature_store,'_limits':limits(),
                '_caller_bytes':0,'_caller_source_bytes':0,'_retained_raw_bytes':0,'maximum_working_bytes':0}
            reuse=ColumnMathReuse(metrics);calls=[]
            def core(a,b,**kwargs):
                from axiom_engine.core import execute_forward_returns
                calls.append(a.shape);return execute_forward_returns(a,b,**kwargs)
            a=readonly(np.array([[8.,8.],[8.,8.],[8.,8.]]));b=readonly(np.array([[7.,7.],[7.,7.],[7.,7.]]))
            flags=readonly(np.ones(a.shape,dtype='?'));keys=[('training',d,3) for d in ('a','b','c')]
            deps=[digest(d) for d in ('a','b','c')]
            first=reuse.forward_block(keys,a,b,flags,anchor='x',dependency_refs=deps,operator=core)
            self.assertEqual(calls,[(6,)])
            self.assertTrue(all(np.array_equal(r['values'],np.array([-.125,-.125])) for r in first))
            second=reuse.forward_block(keys,a,b,flags,anchor='x',dependency_refs=deps,operator=core)
            self.assertEqual(calls,[(6,)])
            self.assertTrue(all(x is y for x,y in zip(first,second)))
            deps[1]=digest('changed physical version')
            reuse.forward_block(keys,a,b,flags,anchor='x',dependency_refs=deps,operator=core)
            self.assertEqual(calls,[(6,),(2,)])
            reuse.forward_block(keys,a,b,flags,anchor='y',dependency_refs=deps,operator=core)
            self.assertEqual(calls,[(6,),(2,),(6,)])
            reuse.close();self.assertEqual(metrics['_column_math_bytes'],0)

    def test_two_fold_v5_save_build_and_readonly_closure(self):
        module=ModuleType('axiom_data');module.QuerySpec=Query
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);f,path=CompactV3Tests().fixture(root);source=ColumnSource(f.spec)
            model_config=training()
            from axiom_research import SessionRange
            model_config=replace(model_config,fit_range=SessionRange(start=f.calendar[0],end=f.calendar[-1]),
                parameters={**model_config.parameters,'learning_rate':.03,'n_estimators':12})
            f.environment={'synthetic':'fixed environment','packages':{'lightgbm':'4.6.0'}}
            with patch.dict(sys.modules,{'axiom_data':module}):
                with _prepare_compact_incrementally(object(),feature_inputs=path,fold_specs=f.folds()[:2],
                    destination=root/'prepared',label_spec=label(3),column_source=source,
                    preparation_options={'row_block_sessions':10,'column_block':32,
                        'maximum_resident_bytes':64*1024**2,'normalization_backend':'core_cs_batch_v1'}) as owner:
                    from axiom_research.stock_signal_evaluation_build import _freeze_build_oos_inputs
                    scope={'calendar':f.calendar,'universe':f.universe,'sessions':sorted({d for fold in f.folds()[:2]
                        for d in fold['inference_cutoff_by_session']}),'evaluation_cutoff':f.folds()[0]['evaluation_cutoff']}
                    runs=[]
                    with _freeze_build_oos_inputs(owner.batch,scope=scope,destination=root/'oos',signal_name='model') as writer:
                        while (item:=owner.next_fold()) is not None:
                            from axiom_research import build_stock_ml_fold_from_saved_inputs
                            from test_stock_sequential_windows import backend
                            with patch('axiom_research.feature_catalog.load_feature_catalog',return_value=f.catalog), \
                                patch('axiom_research.stock_ml._environment',return_value=f.environment), \
                                patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=backend):
                                runs.append(build_stock_ml_fold_from_saved_inputs(item['input_manifest'],fold_spec=item['fold_spec'],
                                    destination=root/'folds',batch=owner.batch,training_spec=model_config))
                        manifest=owner.finish();report=writer.finish()
                    self.assertEqual(_read(report.uri)['contract_version'],'stock_signal_evaluation_inputs_v4')
                    self.assertEqual(manifest['contract_version'],'stock_ml_batch_inputs_v5')
                    self.assertEqual(owner.stats['feature_core_calls'],0)
                    self.assertGreater(owner.stats.get('normalized_daily_reuses',0),0)
                    self.assertGreater(owner.stats['forward_core_calls'],0)
                    derived=[]
                    for run in runs:
                        self.assertEqual(_read(run.path/'dataset.json')['contract_version'],'stock_fold_dataset_v4')
                        prediction=_read(run.path/'predictions.json')
                        self.assertEqual(run.model()['parameters']['learning_rate'],.03)
                        self.assertEqual(run.model()['num_boost_round'],12)
                        self.assertEqual(prediction['label_spec']['horizon_sessions'],3)
                        self.assertEqual(prediction['signal_stage'],'raw_prediction')
                        from axiom_engine.core import StockPredictionFrame,validate_stock_predictions
                        wire,indexed=validate_stock_predictions(StockPredictionFrame.from_dict(prediction))
                        self.assertEqual(wire,prediction)
                        self.assertEqual(len(indexed),len(prediction['rows']))
                        self.assertEqual(load_stock_ml_fold(run.path,batch=owner.batch).identity,run.identity)
                        from axiom_research import (SignalPlanSpec,SignalInput,SignalNode,
                            build_stock_derived_signal,semantic_identity)
                        from axiom_research.stock_label_contracts import NORMALIZATION_SPEC
                        plan=SignalPlanSpec(name='one-real-raw-parent',key=('security_id','session'),
                            inputs=(SignalInput(alias='raw',label=label(3),source_stage='raw_prediction',
                                score_semantics=prediction['score_semantics']),),
                            nodes=(SignalNode(name='z',op='daily_zscore',inputs=('raw',),
                                input_stages=('raw_prediction',),output_stage='daily_zscore',weights=(),
                                reference_universe='saved-fixture-members',missing_policy=NORMALIZATION_SPEC['params']['missing'],
                                parameters=NORMALIZATION_SPEC['params']),
                                SignalNode(name='final',op='weighted_combine',inputs=('z',),input_stages=('daily_zscore',),
                                output_stage='final',weights=(1.0,),reference_universe='saved-fixture-members',
                                missing_policy='propagate',parameters={})),
                            output='final',join_policy='inner_on_security_session',score_semantics='fixture-zscore',
                            available_time_semantics='max_original_dependencies')
                        days=sorted({r['session'] for r in prediction['rows']})
                        context={'calendar_ref':prediction['label_spec']['calendar_ref'],
                            'reference_universe':'saved-fixture-members','reference_universe_ref':digest('frozen-fixture-members'),
                            'reference_members':{d:[{'security_id':r['security_id'],'member':r['member'],
                                'available_at':r['feature_knowledge_cutoff'],'source_refs':[prediction['feature_ref']]}
                                for r in prediction['rows'] if r['session']==d] for d in days},
                            'cutoff_by_session':{d:next(r['knowledge_cutoff'] for r in prediction['rows'] if r['session']==d)
                                for d in days},'clock_basis':'declared_simulation'}
                        saved=build_stock_derived_signal(plan,prediction_inputs={'raw':run.path},context=context,
                            destination=root/'derived',batch=owner.batch)
                        wire=saved.to_dict();self.assertEqual(wire['signal_plan_ref'],semantic_identity(plan))
                        self.assertEqual(wire['parent_signal_refs'],{'raw':prediction['signal_run_ref']})
                        self.assertEqual(wire['signal_stage'],'final')
                        self.assertEqual(saved.engine_input_binding()['parent_inputs']['raw']['model_ref'],run.model()['model_ref'])
                        derived.append(saved)
                        with patch('axiom_engine.core.execute_signal_plan',side_effect=AssertionError('signal HIT Core')):
                            self.assertTrue(build_stock_derived_signal(plan,prediction_inputs={'raw':run.path},context=context,
                                destination=root/'derived',batch=owner.batch).reused)
            self.assertTrue(all(v.closed for v in source.selections))
            with patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=AssertionError('readonly fit')), \
                patch('axiom_engine.core.execute_cs_zscore_batch',side_effect=AssertionError('readonly Core')):
                with load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                    for run in runs:self.assertEqual(load_stock_ml_fold(run.path,batch=batch).identity,run.identity)
                from axiom_research.stock_signal_evaluation_projection import _load_inputs
                _load_inputs(report,scope)
                from axiom_research import (save_stock_derived_signal_evaluation_inputs,
                    evaluate_stock_signal_input_periods,load_stock_signal_evaluation)
                joint=save_stock_derived_signal_evaluation_inputs(report,
                    derived_inputs={'combined':[str(saved.path) for saved in derived]},scope=scope,
                    destination=root/'joint',maximum_resident_bytes=64*1024**2)
                with self.assertRaises(ValueError):save_stock_derived_signal_evaluation_inputs(report,
                    derived_inputs={'combined':[str(saved.path) for saved in derived]},scope=scope,
                    destination=root/'too-small-joint',maximum_resident_bytes=1)
                self.assertFalse(list((root/'too-small-joint').glob('*/manifest.json')))
                reports=evaluate_stock_signal_input_periods(joint,scope=scope,destination=root/'joint-evaluations')
                self.assertEqual(set(reports['all']),{'model','combined'})
                for name,saved in reports['all'].items():
                    body=saved.to_dict();self.assertEqual(body['contract_version'],'stock_signal_evidence_v7')
                    self.assertEqual(body['label_ref'],reports['all']['model'].to_dict()['label_ref'])
                    self.assertEqual(body['input_signal_refs'],[item.identity for item in derived] if name=='combined' else
                        [_read(run.path/'predictions.json')['signal_run_ref'] for run in runs])
                    self.assertEqual(body['signal_lineage'][0]['signal_stage'],'final' if name=='combined' else 'raw_prediction')
                with patch('axiom_engine.core.evaluate_signal_statistics',side_effect=AssertionError('joint HIT statistics')), \
                    patch('axiom_engine.core.execute_signal_plan',side_effect=AssertionError('joint readonly signal math')):
                    for saved in reports['all'].values():self.assertEqual(load_stock_signal_evaluation(saved.path).identity,saved.identity)
                    self.assertTrue(evaluate_stock_signal_input_periods(joint,scope=scope,destination=root/'joint-evaluations')
                        ['all']['combined'].reused)
                    original_signal=derived[0].path/'signal.json';original_bytes=original_signal.read_bytes()
                    original_signal.write_bytes(original_bytes[:-1]+bytes([original_bytes[-1]^1]))
                    self.assertEqual(load_stock_signal_evaluation(reports['all']['combined'].path).identity,
                        reports['all']['combined'].identity)
                    from axiom_research.stock_signal_evaluation_projection import _audit_input
                    with self.assertRaises(ValueError):_audit_input(joint)
                    original_signal.write_bytes(original_bytes)
                frozen=_read(report.uri)
                frozen_paths={row['path'] for row in frozen['admission_receipt']['source_records']}
                proof_paths={descriptor['path'] for item in manifest['folds'] for descriptor in
                    _read(item['input_manifest']['fold_control']['path'])['training_binding']['feature_blocks']}
                self.assertTrue(proof_paths)
                self.assertTrue(proof_paths<=frozen_paths)
                from axiom_research import load_stock_derived_signal
                with patch('axiom_engine.core.execute_signal_plan',side_effect=AssertionError('readonly Derived Core')):
                    for saved in derived:self.assertEqual(load_stock_derived_signal(saved.path).identity,saved.identity)
                proof=Path(sorted(proof_paths)[0]);original=proof.read_bytes()
                proof.write_bytes(original[:-1]+bytes([original[-1]^1]))
                # Normal saved reads use the frozen copies. An explicit owner
                # audit must also detect changed original training proof bytes.
                from axiom_research.stock_signal_evaluation_projection import _audit_input
                with self.assertRaises(ValueError):_audit_input(report)


if __name__=='__main__':unittest.main()

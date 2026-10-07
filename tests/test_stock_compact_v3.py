"""Small vertical v3 acceptance; no real data, supplier or account execution."""
from copy import deepcopy
from pathlib import Path
import builtins
import json
import math
import os
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from axiom_research import (load_stock_feature_view,prepare_stock_ml_batch_inputs,
    load_stock_ml_batch_inputs,build_stock_ml_fold_from_saved_inputs,load_stock_ml_fold,audit_stock_ml_batch_inputs)
from axiom_research.stock_artifacts import _read,digest,file_digest,write_json
from axiom_research.stock_compact_store import StockFeatureView,_view_data,OwnedStore,limits
from axiom_research.stock_compact_batch import read_target
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_matrix_storage import write_buffer,write_part,instant_us
from test_stock_matrix_prepare import PrepareFeatureFixture,PublicDataFixture,Query
from test_stock_folds import backend


class CompactV3Tests(unittest.TestCase):
    def fixture(self,root,transform=None,feature_class=PrepareFeatureFixture,width=6):
        """Write a tiny explicit table, not a Feature numerical executor."""
        f=feature_class(root); path=root/'feature-table'; path.mkdir()
        if width!=6:
            f.columns=['SYN'+str(i).zfill(3) for i in range(width)]
            f.selection=[{'id':column,'semantic_version':'1.0.0'} for column in f.columns]
            f.spec.update(ordered_features=f.columns,feature_selection=f.selection)
            original_width=f.day
            def wider(day):
                values,proof=original_width(day)
                for row in values:
                    row['values']=[float(i+1) for i in range(width)]; row['validity']=[True]*width
                    row['availability']=[day+'T20:00:00.123456+08:00']*width; row['reasons']=[[] for _ in range(width)]
                return values,proof
            f.day=wider
        if transform is not None:
            original=f.day
            def changed(day):
                values,proof=original(day); transform(day,values); return values,proof
            f.day=changed
        rows=[]; parents={}
        for day in f.days:
            dr,proof=f.day(day); rows.extend(dr)
            parents[day]={'feature_ref':digest(proof),'qlib_view_ref':digest(['saved-qlib',day])}
        schema=[{'name':c,'dtype':'float64','unit':'dimensionless','stage':'cross_sectional','missing':'preserve'} for c in f.columns]
        row_index=write_part(path,seal({'sessions':f.days,'security_ids':f.universe,
            'row_count':len(rows),'order':'session_security'},'row_index_ref'),'row_index_ref')
        metadata=write_part(path,seal({'rows':rows,'row_references':parents},'metadata_ref'),'metadata_ref')
        shape=[len(rows),len(f.columns)]
        buffers={k:write_buffer(path,values,dtype=dtype,shape=shape) for k,dtype,values in (
            ('values','float64_le',[v if flag else 0.0 for r in rows for v,flag in zip(r['values'],r['validity'])]),
            ('value_validity','bool_u8',[v for r in rows for v in r['validity']]),
            ('available_at_utc_us','int64_le',[instant_us(a) if a is not None else 0 for r in rows for a in r['availability']]),
            ('available_at_validity','bool_u8',[a is not None for r in rows for a in r['availability']]))}
        part=seal({'table':'features','row_index_ref':row_index['row_index_ref'],'schema_digest':digest(schema),
            'row_offset':0,'row_count':len(rows),'columns':f.columns,'buffers':buffers,'metadata':metadata},'partition_ref')
        write_json(path/'index.json',{'contract_version':'stock_feature_inputs_v3','status':'COMPLETE',
            'definition':{'spec':f.spec},'schema':schema,'row_index':row_index,'partitions':[part],
            'feature_inputs_ref':digest(['synthetic-owner-history',f.spec])})
        return f,path

    def prepare(self,f,view,root,metrics=None,folds=None,data=None,block=32):
        data=data or PublicDataFixture(f.spec); module=types.ModuleType('axiom_data')
        module.QuerySpec=Query; module.adjust_prices=data.adjust
        with patch.dict(sys.modules,{'axiom_data':module}),patch('axiom_research.stock_ml._environment',return_value=f.environment):
            result=prepare_stock_ml_batch_inputs(data,feature_inputs=view,fold_specs=folds or f.folds(),
                destination=root/'prepared',preparation_options={'row_block_sessions':block,'column_block':32,
                    'maximum_resident_bytes':64*1024**2,'normalization_backend':'core_cs_batch_v1'},metrics=metrics)
        return result,data

    def test_shared_batch_to_saved_prediction_and_hit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=self.fixture(root); stats={}
            with load_stock_feature_view(path) as view:
                initial=view.metrics['file_hash_calls']; manifest,data=self.prepare(f,view,root,stats)
                self.assertEqual(manifest['contract_version'],'stock_ml_batch_inputs_v3')
                self.assertEqual(stats['core_calls'],4); self.assertEqual(stats['data_read_calls'],len(data.queries))
                self.assertEqual(stats['legacy_ancestor_reads'],0); self.assertEqual(view.metrics['file_hash_calls'],initial)
                fold=manifest['folds'][0]
                with load_stock_ml_batch_inputs(manifest,feature_inputs=view) as batch:
                    counts=batch.metrics['file_hash_calls']; metrics={}
                    with patch('axiom_research.feature_catalog.load_feature_catalog',return_value=f.catalog), \
                         patch('axiom_research.stock_ml._implementation',return_value=f.implementation), \
                         patch('axiom_research.stock_ml._environment',return_value=f.environment), \
                         patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=backend):
                        run=build_stock_ml_fold_from_saved_inputs(fold['input_manifest'],fold_spec=fold['fold_spec'],
                            destination=root/'folds',batch=batch,metrics=metrics)
                        cached=build_stock_ml_fold_from_saved_inputs(fold['input_manifest'],fold_spec=fold['fold_spec'],
                            destination=root/'folds',batch=batch)
                    self.assertTrue(cached.reused); self.assertEqual(run.identity,cached.identity)
                    self.assertEqual((metrics['train_calls'],metrics['predict_calls'],metrics['core_calls']),(1,1,0))
                    self.assertEqual(batch.metrics['file_hash_calls'],counts)
                    self.assertEqual(load_stock_ml_fold(run.path,batch=batch).predictions(),run.predictions())
                    self.assertEqual(len(run.predictions()['rows']),9)
                    from axiom_engine.core import StockPredictionFrame,validate_stock_predictions
                    wire,indexed=validate_stock_predictions(StockPredictionFrame.from_dict(run.predictions()))
                    self.assertEqual(len(indexed),9); self.assertEqual(wire['signal_run_ref'],run.to_dict()['signal_run_ref'])
                    with batch._matrix_project(fold['input_manifest'],fold['fold_spec']):
                        with self.assertRaisesRegex(ValueError,'borrow'): batch.close()
                real=builtins.__import__
                def blocked(name,*args,**kwargs):
                    if name.startswith(('axiom_data','axiom_engine','qlib','lightgbm','pandas')):
                        raise AssertionError('ordinary loader imported '+name)
                    return real(name,*args,**kwargs)
                with patch('builtins.__import__',side_effect=blocked):
                    self.assertEqual(load_stock_ml_fold(run.path).identity,run.identity)
                hit={}; again,data2=self.prepare(f,view,root,hit)
                self.assertEqual(again,manifest); self.assertTrue(hit['cache_hit']); self.assertEqual(data2.queries,[])
                self.assertEqual((hit['raw_operator_calls'],hit['core_calls']),(0,0))

    def test_saved_consumer_owns_verified_documents_and_opens_outputs_once(self):
        from axiom_research.stock_fold_artifacts import StockMLFold
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=self.fixture(root)
            with load_stock_feature_view(path) as view:
                manifest,_=self.prepare(f,view,root,folds=f.folds()[:1]); fold=manifest['folds'][0]
                with load_stock_ml_batch_inputs(manifest,feature_inputs=view) as batch:
                    with patch('axiom_research.feature_catalog.load_feature_catalog',return_value=f.catalog), \
                         patch('axiom_research.stock_ml._implementation',return_value=f.implementation), \
                         patch('axiom_research.stock_ml._environment',return_value=f.environment), \
                         patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=backend):
                        run=build_stock_ml_fold_from_saved_inputs(fold['input_manifest'],fold_spec=fold['fold_spec'],
                            destination=root/'folds',batch=batch)
                    expected=run.predictions(); real=Path.open; opened=[]; saved_root=run.path.resolve()
                    def watch(path,*args,**kwargs):
                        if path.parent.resolve()==saved_root: opened.append(path.name)
                        return real(path,*args,**kwargs)
                    with patch.object(Path,'open',watch): loaded=load_stock_ml_fold(run.path,batch=batch)
                    self.assertEqual(len(opened),9); self.assertEqual(len(set(opened)),9)
                    with patch.object(Path,'open',side_effect=AssertionError('consumer reread')):
                        self.assertEqual(loaded.predictions(),expected)
                        self.assertEqual(loaded.identity,run.identity)
                        self.assertEqual(loaded.model(),run.model()); self.assertEqual(loaded.evidence(),run.evidence())
                    copied=loaded.predictions(); copied['rows'][0]['score']=123.0
                    self.assertEqual(loaded.predictions(),expected)
                    write_json(run.path/'predictions.json',copied)
                    self.assertEqual(run.predictions(),expected); self.assertEqual(loaded.predictions(),expected)
                    # Equal path dataclasses are not equal owner admissions.
                    self.assertEqual(StockMLFold(run.path).predictions(),copied)
                    with self.assertRaisesRegex(ValueError,'digest mismatch'):
                        load_stock_ml_fold(run.path,batch=batch)

    def test_sealed_handle_and_wrong_cutoff_borrow(self):
        fake=object.__new__(StockFeatureView)
        with self.assertRaisesRegex(ValueError,'owner-loaded'): _view_data(fake)
        with self.assertRaises(AttributeError): fake.validated=True
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=self.fixture(root)
            with load_stock_feature_view(path) as view:
                copied=view.to_dict(); copied['spec']['cutoff_by_session'].clear()
                self.assertTrue(view.to_dict()['spec']['cutoff_by_session'])
                with self.assertRaises(AttributeError): view.payload=copied
                manifest,_=self.prepare(f,view,root)
                changed=deepcopy(manifest); changed['definition']['feature_view']['spec']['cutoff_by_session'].clear()
                changed['definition_ref']=digest(changed['definition']); changed.pop('content_digest'); changed.pop('batch_ref')
                changed['batch_ref']=digest(changed); changed['content_digest']=digest(changed)
                with self.assertRaisesRegex(ValueError,'identity/cutoff'): load_stock_ml_batch_inputs(changed,feature_inputs=view)

    def test_used_bytes_survive_mutation_and_lifecycle_rejects_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=self.fixture(root)
            view=load_stock_feature_view(path); fd=_view_data(view); part=fd['blocks'][0]['parts'][0]
            array=part[1]['values']; original=array.tobytes(); source=Path(part[0]['buffers']['values']['path'])
            payload=source.read_bytes(); source.write_bytes(bytes([payload[0]^1])+payload[1:])
            self.assertEqual(array.tobytes(),original)
            with self.assertRaisesRegex(ValueError,'changed'): view.to_dict()
            with self.assertRaisesRegex(ValueError,'digest mismatch'): load_stock_feature_view(path)
            view.close()

    def test_feature_view_ends_legacy_ancestor_chain(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f=PrepareFeatureFixture(root); old=f.matrix(); path=old.path; old.close()
            real=Path.open; opened=[]
            def watch(path,*args,**kwargs):
                opened.append(str(path)); return real(path,*args,**kwargs)
            with patch.object(Path,'open',watch),patch('axiom_research.labels._batch_source_refs',side_effect=AssertionError('old hash')):
                with load_stock_feature_view(path) as view:
                    self.assertEqual(view.metrics['legacy_ancestor_reads'],0)
                    self.assertEqual(view.metrics['legacy_native_hash_calls'],0)
                    self.assertEqual(view.metrics['file_hash_calls'],len(_view_data(view)['store'].marks))
            self.assertFalse(any('/qlib/' in p or p.endswith('scope.json') for p in opened))

    def test_bit_exact_legacy_oracle_cohort_maturity_and_chunking(self):
        from axiom_research.labels import build_forward_labels
        from axiom_research.stock_label_normalization import normalize_forward_labels
        from test_stock_matrix_prepare import Batch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=self.fixture(root); fold=f.folds()[0]
            with load_stock_feature_view(path) as view:
                stats={}; manifest,data=self.prepare(f,view,root,stats,folds=[fold])
                with load_stock_ml_batch_inputs(manifest,feature_inputs=view) as batch:
                    from axiom_research.stock_batch import _data
                    state=_data(batch)['matrix_state']; record=state.view['fold_targets'][0]
                    normalized=list(state.targets[record['normalized']['target_ref']][1])
                    raws=[]
                    for wire,desc in zip(data.adjusted,record['raw_parts']):
                        actual=list(state.targets[desc['target_ref']][1]); days=state.targets[desc['target_ref']][0]['definition']['sessions']
                        old=build_forward_labels(Batch(wire),calendar=f.calendar,feature_sessions=days)
                        for a,b in zip(actual,old['rows']):
                            for key in ('return','valid','invalid_reason','start_session','end_session','label_available_at'):
                                self.assertEqual(a[key].hex() if type(a[key]) is float else a[key],
                                                 b[key].hex() if type(b[key]) is float else b[key])
                        raws.extend(old['rows'])
                    raw=seal({'contract_version':'stock_label_build_v1','label_spec':{'horizon_sessions':5,
                        'normalization':'none','price_basis':'common_anchor_adjusted_v1'},
                        'calendar_ref':digest(f.calendar),'rows':raws},'label_ref')
                    days=sorted({r['feature_session'] for r in raws})
                    features=seal({'contract_version':'stock_feature_build_v1','ordered_features':f.columns,
                        'rows':[r for d in days for r in f.day(d)[0]]},'feature_ref')
                    oracle=normalize_forward_labels(raw,features=features,cutoff=fold['fit_cutoff'])
                    for a,b in zip(normalized,oracle['rows']):
                        self.assertEqual((a['valid'],a['invalid_reason'],a['raw_return']),
                                         (b['valid'],b['invalid_reason'],b['raw_return']))
                        self.assertEqual(a['return'].hex() if a['valid'] else None,
                                         b['normalized_target'].hex() if b['valid'] else None)
                        self.assertEqual(a['label_available_at'],b['normalized_available_at'])
                    self.assertEqual(len(record['training_offsets']),61*2)
                    first=normalized[0]; index=f.calendar.index(first['feature_session'])
                    hand=(11.0+(index+5)*.01)/(10.0+(index+1)*.01)-1.0
                    self.assertEqual(first['raw_return'].hex(),hand.hex())
                    expected_keys=state.targets[record['normalized']['target_ref']][0]['cohort']['eligible_keys']
                second,_=self.prepare(f,view,root,folds=[fold],block=17)
                with load_stock_ml_batch_inputs(second,feature_inputs=view) as batch:
                    state=_data(batch)['matrix_state']; record=state.view['fold_targets'][0]
                    rows=list(state.targets[record['normalized']['target_ref']][1])
                    self.assertEqual([r['return'].hex() if r['valid'] else None for r in rows],
                                     [r['return'].hex() if r['valid'] else None for r in normalized])
                    self.assertEqual(state.targets[record['normalized']['target_ref']][0]['cohort']['eligible_keys'],expected_keys)

    def test_explicit_audit_and_cache_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=self.fixture(root); fold=f.folds()[0]
            with load_stock_feature_view(path) as view:
                manifest,_=self.prepare(f,view,root,folds=[fold])
                data=PublicDataFixture(f.spec); module=types.ModuleType('axiom_data'); module.QuerySpec=Query; module.adjust_prices=data.adjust
                with patch.dict(sys.modules,{'axiom_data':module}),patch('axiom_research.stock_ml._environment',return_value=f.environment):
                    report=audit_stock_ml_batch_inputs(manifest,data=data,feature_inputs=view)
                self.assertEqual(report['status'],'PASS'); self.assertEqual(report['replay_calls']['core_calls'],1)
                self.assertGreater(len(data.queries),0)
                # Different batch identity, same Raw/normalized cache: no
                # arithmetic replay, but as-known Data selection is required.
                before=manifest['definition']['preparation_options']
                changed={**before,'column_block':64}; stats={}
                data=PublicDataFixture(f.spec); module.adjust_prices=data.adjust
                with patch.dict(sys.modules,{'axiom_data':module}),patch('axiom_research.stock_ml._environment',return_value=f.environment):
                    again=prepare_stock_ml_batch_inputs(data,feature_inputs=view,fold_specs=[fold],destination=root/'prepared',
                        preparation_options=changed,metrics=stats)
                self.assertNotEqual(again['batch_ref'],manifest['batch_ref'])
                self.assertEqual((stats['raw_operator_calls'],stats['core_calls']),(0,0))
                self.assertEqual(stats['raw_cache_hits'],4); self.assertEqual(stats['normalized_cache_hits'],1)
                self.assertEqual(stats['data_read_calls'],8)

    def test_budget_rejects_without_rereading_or_losing_pending_handle(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=self.fixture(root)
            with load_stock_feature_view(path) as view:
                manifest,_=self.prepare(f,view,root,folds=f.folds()[:1]); fd=_view_data(view)
                pending=fd['prepared'][manifest['batch_ref']]; counts=(view.metrics['file_hash_calls'],pending.store.metrics['file_hash_calls'])
                with patch.object(OwnedStore,'read',side_effect=AssertionError('budget re-read')):
                    with self.assertRaisesRegex(ValueError,'accounting'):
                        load_stock_ml_batch_inputs(manifest,feature_inputs=view,limits={'maximum_source_bytes':1})
                self.assertIs(fd['prepared'][manifest['batch_ref']],pending)
                with load_stock_ml_batch_inputs(manifest,feature_inputs=view) as batch:
                    self.assertEqual((view.metrics['file_hash_calls'],pending.store.metrics['file_hash_calls']),counts)
                    from axiom_research.stock_batch import _data
                    state=_data(batch)['matrix_state']; state.store.limits['maximum_matrix_bytes']=state.store.shared_bytes+state.store.resident_bytes+1
                    with self.assertRaisesRegex(ValueError,'byte budget'): batch._matrix_project(manifest['folds'][0]['input_manifest'],manifest['folds'][0]['fold_spec'])
                    self.assertEqual(state.store.borrowers,0); self.assertEqual(state.store.lease_bytes,0)

    def test_raw_cache_independent_of_feature_values_and_normalized_cohort_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); left=root/'left'; right=root/'right'; left.mkdir(); right.mkdir()
            f,path=self.fixture(left); fold=f.folds()[0]
            def missing(day,rows):
                if day==f.days[0]:
                    rows[0]['values'][-1]=None; rows[0]['validity'][-1]=False; rows[0]['reasons'][-1]=['synthetic_missing']
            changed,newpath=self.fixture(right,transform=missing)
            with load_stock_feature_view(path) as view:
                before,_=self.prepare(f,view,root,folds=[fold])
                old=_read(before['prepared_view']['path'])['fold_targets'][0]
            with load_stock_feature_view(newpath) as view:
                stats={}; after,_=self.prepare(changed,view,root,stats,folds=[fold])
                new=_read(after['prepared_view']['path'])['fold_targets'][0]
                self.assertEqual(new['raw_parts'],old['raw_parts']); self.assertEqual(new['evaluation'],old['evaluation'])
                self.assertEqual(stats['raw_operator_calls'],0); self.assertEqual(stats['raw_cache_hits'],4)
                self.assertEqual(stats['core_calls'],1); self.assertNotEqual(new['cohort_ref'],old['cohort_ref'])
                with load_stock_ml_batch_inputs(after,feature_inputs=view) as batch:
                    from axiom_research.stock_batch import _data
                    state=_data(batch)['matrix_state']; rows=list(state.targets[new['normalized']['target_ref']][1])
                    self.assertEqual([r['invalid_reason'] for r in rows[:3]],
                        ['FEATURE_MISSING','NORMALIZATION_UNDEFINED','NOT_MEMBER'])

    def test_fold_as_known_vintage_and_six_clock_oracle(self):
        from test_stock_matrix_prepare import Batch
        class RevisedData(PublicDataFixture):
            def read(self,*,snapshot,query):
                result=super().read(snapshot=snapshot,query=query); cutoff=next(iter(query.cutoff_by_session.values()))
                if query.domain=='market_daily':
                    for row in result.wire['records']:
                        if row['security_id']=='A': row['close']=12.0 if cutoff<self.revision_cutoff else 18.0
                return result
            def adjust(self,*args,**kwargs):
                result=super().adjust(*args,**kwargs); wire=result.wire
                # The selected Data-adjusted opening includes the declared
                # common anchor effect. Research never cancels that factor.
                for row in wire['records']:
                    if row['security_id']=='A': row['open']=8.0
                # One of the six clocks arrives latest, at exact microsecond.
                latest=kwargs['anchor_session']+'T09:01:02.123456Z'
                for field in ('open','close'):
                    for meta in wire['field_meta'][field]['by_key']:
                        if meta['security_id']=='A': meta['anchor_factor_provenance']['usable_from']=latest
                self.adjusted[-1]=deepcopy(wire); return Batch(wire)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=self.fixture(root); folds=f.folds()[:2]; data=RevisedData(f.spec)
            data.revision_cutoff=folds[1]['fit_cutoff']
            with load_stock_feature_view(path) as view:
                manifest,_=self.prepare(f,view,root,folds=folds,data=data)
                with load_stock_ml_batch_inputs(manifest,feature_inputs=view) as batch:
                    from axiom_research.stock_batch import _data
                    state=_data(batch)['matrix_state']; first,second=state.view['fold_targets']
                    day=f.days[5]; results=[]
                    for rec in (first,second):
                        rows=[r for d in rec['raw_parts'] for r in state.targets[d['target_ref']][1]]
                        row=next(r for r in rows if r['security_id']=='A' and r['feature_session']==day)
                        results.append(row)
                        anchor=rec['fold_spec']['fit_session']
                        self.assertEqual(row['label_available_at'],anchor+'T09:01:02.123456Z')
                    self.assertEqual([r['return'].hex() for r in results],[(12.0/8.0-1.0).hex(),(18.0/8.0-1.0).hex()])
                    self.assertNotEqual(first['raw_parts'][0]['target_ref'],second['raw_parts'][0]['target_ref'])
                    self.assertEqual(len({next(iter(q['cutoff_by_session'].values())) for q in data.queries}),3)

    def test_unmodified_public_data_reader_adjustment_and_revision(self):
        from axiom_data import Data
        from test_stock_matrix_prepare import PublicMemoryFeatureFixture,PublicMemoryStore
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=self.fixture(root,feature_class=PublicMemoryFeatureFixture)
            data=Data(root/'absent-fact-root',cache_bytes=0); data.store=PublicMemoryStore(f)
            source_before=digest(data.store.partitions); folds=f.folds()[:2]; stats={}
            with load_stock_feature_view(path) as view,patch('axiom_research.stock_ml._environment',return_value=f.environment):
                manifest=prepare_stock_ml_batch_inputs(data,feature_inputs=view,fold_specs=folds,destination=root/'prepared',
                    preparation_options={'row_block_sessions':32,'column_block':32,'maximum_resident_bytes':64*1024**2,
                        'normalization_backend':'core_cs_batch_v1'},metrics=stats)
                with load_stock_ml_batch_inputs(manifest,feature_inputs=view) as batch:
                    from axiom_research.stock_batch import _data
                    state=_data(batch)['matrix_state']; actual=[]
                    for rec in state.view['fold_targets']:
                        row=next(r for d in rec['raw_parts'] for r in state.targets[d['target_ref']][1]
                                 if r['security_id']=='B' and r['feature_session']==f.revised_feature_session)
                        actual.append(row['return'])
                    feature_pos=f.calendar.index(f.revised_feature_session); start,end=feature_pos+1,feature_pos+5
                    expected=[]
                    for rec in state.view['fold_targets']:
                        anchor=f.calendar.index(rec['fold_spec']['fit_session'])
                        anchor_factor=1.0+(anchor%7)*.15+.05
                        op=(10.0+.5+start*.05)*(1.0+(start%7)*.15+.05)/anchor_factor
                        close=12.0+.8+end*.05; factor=1.0+(end%7)*.15+.05
                        if rec is state.view['fold_targets'][1]: close+=3.25; factor*=1.125
                        expected.append((close*factor/anchor_factor)/op-1.0)
                    self.assertEqual([v.hex() for v in actual],[v.hex() for v in expected])
                    self.assertNotEqual(actual[0],actual[1]); self.assertEqual(stats['core_calls'],2)
            self.assertEqual(digest(data.store.partitions),source_before); self.assertFalse((root/'absent-fact-root').exists())

    def test_ordinary_load_is_not_a_math_audit(self):
        from axiom_research.stock_compact_labels import _publish
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=self.fixture(root); folds=f.folds()[:1]
            with load_stock_feature_view(path) as view:
                original,_=self.prepare(f,view,root,folds=folds)
                with load_stock_ml_batch_inputs(original,feature_inputs=view) as batch:
                    from axiom_research.stock_batch import _data
                    state=_data(batch)['matrix_state']; rec=state.view['fold_targets'][0]
                    value,rows=state.targets[rec['evaluation']['target_ref']]; changed_rows=list(rows)
                    changed_rows[0]['return']=22.0
                    desc=_publish(root/'producer-bug-target',value['definition'],changed_rows,budgets=limits())
                changed_view=_read(original['prepared_view']['path']); changed_view['fold_targets'][0]['evaluation']=desc
                changed_view.pop('prepared_view_ref'); changed_view=seal(changed_view,'prepared_view_ref')
                vd=root/'producer-bug-view.json'; write_json(vd,changed_view)
                descriptor={'path':str(vd),'file_digest':file_digest(vd),'prepared_view_ref':changed_view['prepared_view_ref']}
                altered=deepcopy(original); altered['prepared_view']=descriptor
                for fold in altered['folds']:
                    fold['input_manifest']['prepared_view']=descriptor
                    fold['input_manifest'].pop('input_ref'); fold['input_manifest']=seal(fold['input_manifest'],'input_ref')
                altered.pop('batch_ref'); altered.pop('content_digest'); altered['batch_ref']=digest(altered); altered=seal(altered,'content_digest')
                real=builtins.__import__
                def blocked(name,*a,**k):
                    if name.startswith(('axiom_data','axiom_engine')): raise AssertionError('ordinary loader executed '+name)
                    return real(name,*a,**k)
                with patch('builtins.__import__',side_effect=blocked),load_stock_ml_batch_inputs(altered,feature_inputs=view): pass
                data=PublicDataFixture(f.spec); module=types.ModuleType('axiom_data'); module.QuerySpec=Query; module.adjust_prices=data.adjust
                with patch.dict(sys.modules,{'axiom_data':module}),patch('axiom_research.stock_ml._environment',return_value=f.environment):
                    with self.assertRaisesRegex(ValueError,'audit saved target differs: return'):
                        audit_stock_ml_batch_inputs(altered,data=data,feature_inputs=view)

    def test_backend_source_mutation_prevents_complete_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=self.fixture(root)
            view=load_stock_feature_view(path); manifest,_=self.prepare(f,view,root,folds=f.folds()[:1])
            fd=_view_data(view); target=Path(fd['definition']['partitions'][0]['buffers']['values']['path'])
            original=target.read_bytes()
            def corrupt(*args,**kwargs):
                result=backend(*args,**kwargs); target.write_bytes(bytes([original[0]^1])+original[1:]); return result
            with load_stock_ml_batch_inputs(manifest,feature_inputs=view) as batch:
                fold=manifest['folds'][0]
                with patch('axiom_research.feature_catalog.load_feature_catalog',return_value=f.catalog), \
                     patch('axiom_research.stock_ml._implementation',return_value=f.implementation), \
                     patch('axiom_research.stock_ml._environment',return_value=f.environment), \
                     patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=corrupt):
                    with self.assertRaisesRegex(ValueError,'changed'):
                        build_stock_ml_fold_from_saved_inputs(fold['input_manifest'],fold_spec=fold['fold_spec'],
                            destination=root/'folds',batch=batch)
            self.assertFalse(list((root/'folds').glob('*/fold.json'))); view.close()

    def test_wide_typed_features_use_same_path_and_all_columns_select_cohort(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            first_day=[None]
            def bad(day,rows):
                if first_day[0] is None: first_day[0]=day
                if day==first_day[0]:
                    rows[0]['values'][39]=None; rows[0]['validity'][39]=False; rows[0]['reasons'][39]=['synthetic_missing']
            f,path=self.fixture(root,width=40,transform=bad)
            with load_stock_feature_view(path) as view:
                manifest,_=self.prepare(f,view,root,folds=f.folds()[:1])
                with load_stock_ml_batch_inputs(manifest,feature_inputs=view) as batch:
                    fold=manifest['folds'][0]
                    with batch._matrix_project(fold['input_manifest'],fold['fold_spec']) as projection:
                        self.assertEqual(projection.X.shape,(120,40)); self.assertEqual(projection.P.shape,(6,40))
                        self.assertEqual(projection.X[0].tolist(),[float(i+1) for i in range(40)])
                        self.assertEqual(projection.excluded['FEATURE_MISSING'],1)
                        self.assertEqual(projection.excluded['NORMALIZATION_UNDEFINED'],1)

    def test_partial_publication_and_target_schema_clock_ref_rejected(self):
        from axiom_research.stock_compact_labels import _publish
        from test_stock_labels import batch,build,CALENDAR,CUTOFF
        raw=build(batch())
        definition={'sessions':[CALENDAR[0]],'universe':['A','B'],'calendar':list(CALENDAR),'cutoff':CUTOFF}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); target=root/'target'
            with patch('axiom_research.stock_compact_batch.read_target',side_effect=ValueError('injected schema failure')):
                with self.assertRaisesRegex(ValueError,'schema failure'):
                    _publish(target,definition,raw['rows'],budgets=limits())
            self.assertFalse(target.exists())
            desc=_publish(target,definition,raw['rows'],budgets=limits()); original=_read(desc['path'])
            for role in ('dtype','shape','ref','clock'):
                value=deepcopy(original)
                if role=='dtype': value['buffers']['values']['dtype']='int64_le'
                elif role=='shape': value['buffers']['values']['shape']=[1]
                elif role=='ref': value['definition_ref']=digest('wrong-definition')
                else:
                    value['buffers']['availability_validity']=write_buffer(target,[False,False],dtype='bool_u8',shape=[2])
                value.pop('target_ref'); value=seal(value,'target_ref'); path=target/(role+'.json'); write_json(path,value)
                with OwnedStore() as store,self.assertRaises(ValueError):
                    read_target(store,{'path':str(path),'file_digest':file_digest(path),'target_ref':value['target_ref']})


class OwnedBytesTests(unittest.TestCase):
    def test_owned_bytes_are_hashed_once_not_reopened_for_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            desc=write_buffer(temp,[1.0,math.nextafter(1.0,2.0)],dtype='float64_le',shape=[2])
            store=OwnedStore(); real=Path.open; reads=[]
            def watch(path,*args,**kwargs): reads.append(str(path)); return real(path,*args,**kwargs)
            with patch.object(Path,'open',watch):
                a=store.buffer(desc); b=store.buffer(desc)
            self.assertEqual(reads,[desc['path']]); self.assertEqual(store.metrics['file_hash_calls'],1)
            self.assertEqual(a.tobytes(),struct.pack('<2d',1.0,math.nextafter(1.0,2.0)))
            with self.assertRaises(ValueError): a.flags.writeable=True
            self.assertEqual(b.tobytes(),a.tobytes()); store.close()

    def test_replace_and_change_during_same_fd_read_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            desc=write_buffer(temp,[1.0],dtype='float64_le',shape=[1]); store=OwnedStore()
            real=Path.open; path=Path(desc['path'])
            class MutatingStream:
                def __init__(self,stream): self.stream=stream
                def __enter__(self): return self
                def __exit__(self,*args): self.stream.close()
                def fileno(self): return self.stream.fileno()
                def read(self,n):
                    data=self.stream.read(n)
                    with real(path,'wb') as writer: writer.write(data+b'x')
                    return data
            with patch.object(Path,'open',lambda p,*a,**k:MutatingStream(real(p,*a,**k))):
                with self.assertRaisesRegex(ValueError,'during read'): store.buffer(desc)
            self.assertEqual(store.metrics['file_hash_calls'],0); store.close()

    def test_ref_dtype_shape_bool_and_json_schema_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            desc=write_buffer(temp,[1.0],dtype='float64_le',shape=[1])
            for changed in ({**desc,'file_digest':digest('wrong')},{**desc,'shape':[2]},
                            {**desc,'dtype':'float32'},{**desc,'shape':[True]}):
                store=OwnedStore()
                with self.assertRaises(ValueError): store.buffer(changed)
                store.close()
            for value in (b'{"x":1,"x":2}',b'{"x":NaN}'):
                path=Path(temp)/'bad.json'; path.write_bytes(value); store=OwnedStore()
                with self.assertRaises(ValueError): store.read_json({'path':str(path),'file_digest':file_digest(path)})
                store.close()


if __name__=='__main__': unittest.main()

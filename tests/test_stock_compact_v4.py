"""Bounded synthetic v4 controls, queries and same-owner audit regressions."""
from copy import deepcopy
from datetime import date,timedelta
from pathlib import Path
import sys,tempfile,types,unittest,weakref,json
from unittest.mock import patch
import numpy as np

from axiom_research import (load_stock_feature_view,prepare_stock_ml_batch_inputs,
    load_stock_ml_batch_inputs,audit_stock_ml_batch_inputs,load_stock_ml_fold)
from axiom_research.stock_artifacts import _read,digest,file_digest,write_json
from axiom_research.stock_fold_inputs import seal,validate_spec
from axiom_research.stock_compact_store import OwnedStore,_view_data
from axiom_research.stock_compact_controls import grid_ranges,price_domain_ranges,validate_price_part
from axiom_research.stock_matrix_storage import write_part
from axiom_research.stock_batch import _data
import test_stock_compact_v3 as fixtures
import test_stock_sequential_windows as window_fixtures
from test_stock_matrix_prepare import PublicDataFixture,Query


class CompactV4Tests(unittest.TestCase):
    def prepare(self,root,*,block=5):
        f,path=fixtures.CompactV3Tests().fixture(root); data=PublicDataFixture(f.spec)
        module=types.ModuleType('axiom_data'); module.QuerySpec=Query; module.adjust_prices=data.adjust
        with patch.dict(sys.modules,{'axiom_data':module}):
            manifest=prepare_stock_ml_batch_inputs(data,feature_inputs=path,fold_specs=f.folds()[:2],
                destination=root/'prepared',preparation_options={'row_block_sessions':block,'column_block':32,
                    'maximum_resident_bytes':64*1024**2,'normalization_backend':'core_cs_batch_v1'})
        return f,path,manifest,data

    def test_adjacent_folds_share_two_bounded_evaluation_domains_and_full_fit_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            f,path,manifest,data=self.prepare(Path(temp)); self.assertEqual(manifest['contract_version'],'stock_ml_batch_inputs_v4')
            markets=[q for q in data.queries if q['domain']=='market_daily']
            self.assertEqual(len(data.queries),8)
            evaluation=[q for q in markets if set(q['cutoff_by_session'].values())=={f.folds()[0]['evaluation_cutoff']}]
            self.assertEqual(len(evaluation),2)
            for query in evaluation: self.assertLessEqual(len(query['sessions']),11)
            print('V4_DOMAIN_METRICS '+json.dumps({'folds':2,'evaluation_block_sessions':5,'training_domains':2,
                'evaluation_domains':2,'data_read_calls':len(data.queries),'maximum_evaluation_query_sessions':max(len(q['sessions']) for q in evaluation)}))
            for fold in f.folds()[:2]:
                queries=[q for q in markets if set(q['cutoff_by_session'].values())=={fold['fit_cutoff']}]
                self.assertEqual(len(queries),1)
                training,_=validate_spec(fold,f.calendar); anchor=fold['fit_session']; expected={anchor}
                for day in training:
                    pos=f.calendar.index(day)
                    expected.update(f.calendar[pos+n] for n in (1,5) if pos+n<len(f.calendar) and f.calendar[pos+n]<=anchor)
                self.assertEqual(queries[0]['sessions'],sorted(expected))
            common=_read(manifest['prepared_view']['path']); self.assertNotIn('fold_targets',common)
            with load_stock_ml_batch_inputs(manifest) as eager,load_stock_ml_batch_inputs(manifest,residency='sequential') as sequential:
                for fold in manifest['folds']:
                    control=_read(fold['input_manifest']['fold_control']['path'])
                    self.assertNotIn('training_offsets',control)
                    with eager._matrix_project(fold['input_manifest'],fold['fold_spec']) as a,sequential._matrix_project(fold['input_manifest'],fold['fold_spec']) as b:
                        for key in ('X','y','P'):
                            x,y=getattr(a,key),getattr(b,key); self.assertEqual((x.shape,x.dtype,x.tobytes()),(y.shape,y.dtype,y.tobytes()))
                        training,inference=validate_spec(fold['fold_spec'],f.calendar)
                        mature=[day for day in training if f.calendar[f.calendar.index(day)+5]<=fold['fold_spec']['fit_session']]
                        self.assertEqual(b.training_keys,[[s,d] for d in mature for s in ('A','B')])
                        self.assertEqual([[r['security_id'],r['feature_session']] for r in b.evaluation['rows']],
                            [[s,d] for d in inference for s in f.universe])
                        np.testing.assert_allclose(b.y,np.tile([-1.,1.],len(mature)),rtol=0,atol=2e-14)
                state=_data(sequential)['matrix_state']; state.verify_all()
                self.assertEqual(state.targets,{})
                self.assertFalse(any(value.get('contract_version')=='stock_ml_fold_control_v1' for value in state.store.json.values()))

    def test_260_controls_do_not_expand_or_read_declared_150000_row_selectors(self):
        """Header-only synthetic descriptors deliberately cannot admit leaves."""
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=fixtures.CompactV3Tests().fixture(root); index=_read(path/'index.json')
            calendar=[]; day=date(2010,1,1)
            while len(calendar)<2200:
                if day.weekday()<5: calendar.append(day.isoformat())
                day+=timedelta(days=1)
            universe=['S'+str(i).zfill(3) for i in range(300)]; spec=index['definition']['spec']
            spec.update(calendar=calendar,read_sessions=calendar,feature_sessions=calendar,universe=universe,
                cutoff_by_session={d:d+'T20:30:00+08:00' for d in calendar})
            row_index=seal({'sessions':calendar,'security_ids':universe,'row_count':len(calendar)*300,'order':'session_security'},'row_index_ref')
            index['row_index']=write_part(path,row_index,'row_index_ref')
            part=index['partitions'][0]; part['row_index_ref']=row_index['row_index_ref']; part['row_count']=len(calendar)*300
            for descriptor in part['buffers'].values(): descriptor['shape']=[part['row_count'],6]
            index['partitions']=[seal({k:v for k,v in part.items() if k!='partition_ref'},'partition_ref')]; write_json(path/'index.json',index)
            with load_stock_feature_view(path,residency='sequential') as feature:
                common={k:deepcopy(spec[k]) for k in ('scope','snapshot','pit_policy','calendar','universe','catalog_ref','feature_selection','ordered_features')}
                view=seal({'contract_version':'stock_ml_prepared_view_v3','definition':common,'feature_view':feature.to_dict()},'prepared_view_ref')
                descriptor=write_part(root,view,'prepared_view_ref'); folds=[]
                first=next(i for i,d in enumerate(calendar) if d>='2012-01-04')
                for n in range(260):
                    pos=first+n*5; fit=calendar[pos]; inference=calendar[pos:pos+3]
                    fold={'contract_version':'stock_ml_fold_spec_v2','training_window':{'unit':'calendar_years','length':2,
                        'end':'previous_fit_session','start':'fit_date_minus_years_inclusive','leap_day':'clamp_feb_28'},
                        'fit_session':fit,'fit_cutoff':fit+'T20:30:00+08:00','simulated_model_available_at':fit+'T20:45:00+08:00',
                        'oos_trade_sessions':calendar[pos+1:pos+4],'inference_cutoff_by_session':{d:d+'T21:00:00+08:00' for d in inference},
                        'evaluation_cutoff':calendar[-1]+'T20:30:00+08:00'}
                    training,_=validate_spec(fold,calendar)
                    def selected(kind,days,count):
                        return seal({'contract_version':'stock_fold_selector_v1','kind':kind,'row_index_ref':row_index['row_index_ref'],
                            'ranges':grid_ranges(days,calendar,300),'target_refs':[digest('unadmitted target')],
                            'selected_count':count,'keys_digest':digest([[s,d] for d in days for s in universe]) if kind=='complete_grid' else digest('unadmitted keys')},'selector_ref')
                    train=selected('normalized_valid_rows',training,150000); infer=selected('complete_grid',inference,900)
                    inputs=seal({'contract_version':'stock_ml_saved_inputs_v4','prepared_view':descriptor,'fold_spec_ref':digest(fold),
                        'fold_control':{'path':str(root/f'not-admitted-{n}.json'),'file_digest':digest(n),'fold_control_ref':digest(['control',n])},
                        'selectors':{'training':train,'training_labels':train,'inference':infer,'evaluation_labels':infer,'validation':None},
                        'core_result_refs':[digest('unadmitted Core')]},'input_ref')
                    folds.append({'fold_spec':fold,'input_manifest':inputs})
                definition={'version':'axiom.stock_ml_batch_inputs/4','feature_view':feature.to_dict(),'fold_specs':[f['fold_spec'] for f in folds],
                    'price_domain_plan':'fit_window_evaluation_calendar_blocks_v1','preparation_options':{'row_block_sessions':32}}
                manifest={'contract_version':'stock_ml_batch_inputs_v4','definition':definition,'definition_ref':digest(definition),
                    'prepared_view':descriptor,'folds':folds,'status':'COMPLETE'}
                manifest['batch_ref']=digest(manifest); manifest=seal(manifest,'content_digest')
                with patch('axiom_research.stock_compact_batch.expand_ranges',side_effect=AssertionError('selector expansion')), \
                     patch('axiom_research.stock_compact_batch.read_target',side_effect=AssertionError('target admission')), \
                     patch.object(OwnedStore,'buffer',side_effect=AssertionError('leaf admission')):
                    with load_stock_ml_batch_inputs(manifest,feature_inputs=feature,residency='sequential') as batch:
                        state=_data(batch)['matrix_state']; self.assertEqual(len(state.records),260)
                        self.assertEqual(state.targets,{})
                        self.assertLess(batch.metrics['resident_bytes'],32*1024**2)
                        print('V4_CONTROL_METRICS '+json.dumps({'folds':260,'declared_selected_rows_per_fold':150000,
                            'resident_bytes':batch.metrics['resident_bytes'],'target_admissions':0,'selector_expansions':0}))
                        self.assertTrue(all(set(record)=={'path','file_digest','fold_control_ref'} for _,record in state.records.values()))

    def test_audit_activates_releases_same_owner_and_accounts_both_windows(self):
        from axiom_research import stock_compact_audit as audit
        from axiom_research import stock_compact_batch as owner
        with tempfile.TemporaryDirectory() as temp:
            f,path,manifest,_=self.prepare(Path(temp)); data=PublicDataFixture(f.spec)
            module=types.ModuleType('axiom_data'); module.QuerySpec=Query; module.adjust_prices=data.adjust
            actual=audit._compare_fold; read=owner.read_target; refs=[]; comparisons=[]
            def tracked(*args,**kwargs):
                value,rows=read(*args,**kwargs); refs.append(weakref.ref(rows)); return value,rows
            def compare(saved,replay,left,right):
                comparisons.append(left['fold_spec']['fit_session'])
                combined=saved.store.resident_bytes+replay.store.resident_bytes+_view_data(saved.feature)['store'].resident_bytes
                self.assertEqual(replay.store.shared_bytes+replay.store.resident_bytes,combined)
                self.assertGreaterEqual(replay._fixed_shared_source_bytes,saved.store.metrics['source_bytes'])
                self.assertEqual(saved._active_record,replay._active_record)
                return actual(saved,replay,left,right)
            with patch.dict(sys.modules,{'axiom_data':module}),patch.object(audit,'_compare_fold',new=compare),patch.object(owner,'read_target',new=tracked):
                report=audit_stock_ml_batch_inputs(manifest,data=data)
            self.assertEqual(report['status'],'PASS'); self.assertEqual(len(comparisons),2)
            self.assertTrue(all(ref() is None for ref in refs))

    def test_v4_standalone_saved_hit_keeps_training_matrix_and_backend_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path,manifest,_=self.prepare(root); fold=manifest['folds'][0]; helper=window_fixtures.SequentialWindowTests()
            with load_stock_ml_batch_inputs(manifest,residency='sequential') as batch:
                run=helper.build(f,fold,batch,root/'folds')
                with patch('axiom_research.stock_compact_batch.training_matrix',side_effect=AssertionError('HIT X allocation')):
                    metrics={}; hit=helper.build(f,fold,batch,root/'folds',fit=AssertionError('HIT backend'),metrics=metrics)
                self.assertEqual((metrics['train_calls'],metrics['predict_calls']),(0,0)); self.assertTrue(hit.reused)
            with patch('axiom_research.stock_compact_batch.training_matrix',side_effect=AssertionError('standalone X allocation')):
                self.assertEqual(load_stock_ml_fold(run.path).identity,run.identity)

    def test_fixed_domain_validator_rejects_cross_block_and_wrong_full_query(self):
        with tempfile.TemporaryDirectory() as temp:
            f,path,manifest,_=self.prepare(Path(temp)); folds=manifest['folds']; common=_read(manifest['prepared_view']['path'])['definition']
            domains=price_domain_ranges(folds,f.calendar,5); control=_read(folds[0]['input_manifest']['fold_control']['path'])
            definition=_read(control['evaluation_parts'][0]['path'])['definition']
            changed=deepcopy(definition); start=f.calendar.index(changed['sessions'][0]); changed['sessions']=[f.calendar[start],f.calendar[start+5]]
            with self.assertRaisesRegex(ValueError,'crosses fixed'): validate_price_part(changed,common,domains,5,evaluation=True)
            changed=deepcopy(definition); query=changed['price_view']['context']['query']; removed=next(d for d in query['sessions'] if d!=query['adjustment_anchor'])
            query['sessions'].remove(removed); query['cutoff_by_session'].pop(removed)
            with self.assertRaisesRegex(ValueError,'domain|native'): validate_price_part(changed,common,domains,5,evaluation=True)


if __name__=='__main__': unittest.main()

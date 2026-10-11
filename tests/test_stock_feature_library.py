"""One synthetic saved-source case: original owner admission, raw reuse, Core CS."""
from pathlib import Path
import tempfile
import unittest
from datetime import date, timedelta
import numpy as np

from axiom_engine.core import compile_feature_columns, execute_feature_columns
from axiom_engine.feature_library import FeatureLibrary, feature_definitions
from axiom_research import load_a01_feature_catalog, SavedColumnFeatureSource, CatalogFeatureLibrary
from axiom_research.stock_artifacts import digest, file_digest, write_json
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_feature_columns import REASONS, feature_column_plan
from axiom_research.stock_matrix_storage import write_buffer, write_part, instant_us


def saved_source(root, *, day_count=6, generation=0, value_shift=0, snapshot='synthetic_only'):
    """Synthetic selected vendor-shaped cells; never used as real data evidence."""
    days=[(date(2020,1,1)+timedelta(days=i)).isoformat() for i in range(day_count)]
    securities=['A','B','C']; shape=(day_count,3,6)
    values=np.empty(shape,dtype='<f8')
    for t in range(day_count):
        for j in range(3):
            opening=10*(j+1)+t+value_shift
            values[t,j]=[opening,opening+3+j+t,opening-2,opening+1+j+t,1000*(j+1)+t,2**t]
    available=np.array([[instant_us(day+'T12:00:00Z')]*3 for day in days],dtype='<i8')
    arrays={'values':values,'original_values':values.copy(),'validity':np.ones(shape,dtype=bool),
        'available':np.broadcast_to(available[:,:,None],shape).copy(),'available_validity':np.ones(shape,dtype=bool),
        'reasons':np.zeros(shape,dtype='u1'),'member':np.ones(shape[:2],dtype=bool),
        'member_available':available.copy(),'member_available_validity':np.ones(shape[:2],dtype=bool)}
    buffers={}
    for name,array in arrays.items():
        dtype='bool_u8' if array.dtype.kind=='b' else 'uint8' if array.dtype==np.dtype('u1') else 'int64_le' if array.dtype.kind=='i' else 'float64_le'
        buffers[name]=write_buffer(root,array.astype('u1',copy=False).reshape(-1) if dtype=='bool_u8' else array.reshape(-1),
            dtype=dtype,shape=list(array.shape))
    profiles={}; evidence=[]
    for domain,fields in [('market_daily',['open','high','low','close','amount_cny']),
                          ('adjustment_factors',['factor']),('universe_membership',['is_member'])]:
        metadata={}
        for field in fields:
            metadata[field]={'dtype':'bool' if field=='is_member' else 'float64',
                'unit':None if field=='is_member' else 'CNY' if field=='amount_cny' else 'dimensionless' if field=='factor' else 'CNY/share',
                'by_key':[{'security_id':s,'session':day,'revision_id':f'{field}:{s}:{day}',
                    'revision_sequence':0,'raw_batch_id':'synthetic','first_observed_at':day+'T12:00:00Z',
                    'usable_from':day+'T12:00:00Z','availability_basis':'declared_vendor_assumption',
                    'evidence_ref':None,'missing_reason':None} for day in days for s in securities]}
        child=seal({'contract_version':'reader_column_evidence_v1','context':{'domain':domain},
            'field_meta':metadata,'fields':fields,'sessions':days},'evidence_ref')
        evidence.append(write_part(root,child,'evidence_ref'))
        profiles[domain]={'contract':{'contract_id':'synthetic_'+domain,'logical_key':['security_id','session'],
            'fields':{f:{'dtype':m['dtype'],'unit':m['unit']} for f,m in metadata.items()}},
            'source_profile':{'id':'synthetic_'+domain}}
    catalog=load_a01_feature_catalog(); selection=[{'id':'ALP001','semantic_version':'1.0.0'}]
    spec={'read_sessions':days,'feature_sessions':days[2:],'universe':securities,'calendar':days,
        'cutoff_by_session':{day:day+'T13:00:00Z' for day in days},'snapshot':snapshot,
        'pit_policy':'best_effort_vendor_v1','catalog_ref':catalog.identity,'feature_selection':selection}
    impl={'fixture':digest(['explicit synthetic source'])}
    manifest=seal({'contract_version':'research_feature_column_source_v1','spec':spec,'qlib_view':{'universe_name':'csi300'},
        'buffers':buffers,'chunks':[{'sessions':days,'evidence':evidence}],'reason_codes':list(REASONS),
        'source_profiles':profiles,'selection_invariance':'declared_vendor_release_visible_at_each_historical_row_cutoff',
        'implementation_sources':impl,'implementation_ref':digest(impl),
        'environment':{'synthetic':True,'generation':generation}},'source_ref')
    path=Path(root)/'source.json'; write_json(path,manifest)
    return {'path':str(path),'file_digest':file_digest(path),'source_ref':manifest['source_ref']}


def tiles(view):
    cap=view.describe()
    return [{"session":day,"column":column,"partition":receipt["partition"]}
            for day in cap["sessions"]
            for column,receipt in view.column_receipts(day).items()]


class StockFeatureLibraryTests(unittest.TestCase):
    def test_second_experiment_adds_other_source_column_and_rejects_wrong_training_source(self):
        import json
        from axiom_research.stock_feature_library_adapter import (
            build_feature_library_training_view,load_feature_library_training_view)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); library=FeatureLibrary(root/'features',maximum_working_bytes=8*1024**2)
            catalog=load_a01_feature_catalog(); selection=[{'id':n,'semantic_version':'1.0.0'} for n in ('ALP001','ALP003')]
            original=saved_source(root/'source-0',generation=0)
            current=saved_source(root/'source-1',generation=1)
            wrong=saved_source(root/'source-wrong',generation=2,value_shift=10,snapshot='different_snapshot')
            with SavedColumnFeatureSource(original,maximum_working_bytes=8*1024**2) as source:
                day=source.source.manifest['spec']['feature_sessions'][0]
                old,_=CatalogFeatureLibrary(library,source,catalog).ensure(selection=selection[:1],
                    output_sessions=[day],normalized=True)
                with library.open_feature_view(old) as view: old_part=tiles(view)[0]['partition']['partition_ref']
            with SavedColumnFeatureSource(current,maximum_working_bytes=8*1024**2) as source:
                descriptor,stats=CatalogFeatureLibrary(library,source,catalog).ensure(selection=selection,
                    output_sessions=[day],normalized=True)
                self.assertEqual((stats['raw']['core_calls'],stats['cross_sectional']['core_calls']),(1,1))
                self.assertEqual((stats['raw']['computed_output_cells'],stats['cross_sectional']['computed_output_cells']),(3,3))
                with library.open_feature_view(descriptor) as view:
                    self.assertEqual(tiles(view)[0]['partition']['partition_ref'],old_part)
                    actual=view.project(); expected_plan=source.plan(catalog,selection,normalized=True)
                    inputs=source.prepare(expected_plan,day,['A','B','C'],{s['id']:digest(['direct']) for s in selection})
                    expected=execute_feature_columns(compile_feature_columns(expected_plan),inputs=inputs,
                        output_sessions=[day],max_working_bytes=8*1024**2)
                    for key in ('values','validity','available_at_utc_us'):
                        np.testing.assert_array_equal(actual[key],expected[key][0])
                    new_tile=tiles(view)[1]['partition']
                    part=view._store.read_json({k:new_tile[k] for k in ('path','file_digest','partition_ref')},'partition_ref')
                    self.assertEqual([s['view_ref'] for s in part['column_source_bindings']['ALP003']],
                                     [current['source_ref']]*len(part['column_source_bindings']['ALP003']))
                    projection=build_feature_library_training_view(view,source,selection=selection,
                        destination=root/'training',maximum_working_bytes=8*1024**2)
                    with load_feature_library_training_view(projection,source=source,
                            limits={'maximum_matrix_bytes':8*1024**2}) as training:
                        self.assertEqual(training.to_dict()['historical_feature_inputs_ref'],view.identity)
                    with SavedColumnFeatureSource(wrong,maximum_working_bytes=8*1024**2) as mismatch:
                        with self.assertRaisesRegex(ValueError,'local dependency or reference mismatch'):
                            build_feature_library_training_view(view,mismatch,selection=selection,
                                destination=root/'wrong-training',maximum_working_bytes=8*1024**2)
                # A resealed transport cannot relabel the saved view to a new source.
                forged=json.loads(Path(projection['path']).read_text()); forged.pop('training_adapter_ref')
                forged['definition']['source']=wrong
                forged['definition']['spec']['snapshot']='different_snapshot'
                forged=write_part(root/'forged',seal(forged,'training_adapter_ref'),'training_adapter_ref')
                with self.assertRaisesRegex(ValueError,'local dependency or reference mismatch'):
                    load_feature_library_training_view(forged,limits={'maximum_matrix_bytes':8*1024**2})

    def test_ten_source_generations_only_bind_actual_single_day_cs_gap(self):
        from axiom_engine.core import FeaturePlan
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); library=FeatureLibrary(root/'features',maximum_working_bytes=8*1024**2)
            catalog=load_a01_feature_catalog(); selection=[{'id':n,'semantic_version':'1.0.0'} for n in ('ALP001','ALP003')]
            descriptors=[]; days=[]
            for generation in range(10):
                descriptor=saved_source(root/f'source-{generation}',day_count=12,generation=generation)
                descriptors.append(descriptor)
                with SavedColumnFeatureSource(descriptor,maximum_working_bytes=8*1024**2) as source:
                    day=source.source.manifest['spec']['feature_sessions'][generation]; days.append(day)
                    _,stats=CatalogFeatureLibrary(library,source,catalog).ensure(selection=selection,output_sessions=[day])
                    self.assertEqual(stats['raw']['core_calls'],1)
            with SavedColumnFeatureSource(descriptors[-1],maximum_working_bytes=8*1024**2) as current:
                facade=CatalogFeatureLibrary(library,current,catalog)
                facade.ensure(selection=selection,output_sessions=days[:-1],normalized=True)
                descriptor,stats=facade.ensure(selection=selection,output_sessions=days,normalized=True)
                self.assertEqual((stats['raw']['core_calls'],stats['cross_sectional']['core_calls']),(0,1))
                self.assertLessEqual(stats['cross_sectional']['maximum_execution_source_ids'],7)
                hashes=current.metrics['physical_field_day_hashes']
                facade.ensure(selection=selection,output_sessions=days,normalized=True)
                self.assertEqual(current.metrics['physical_field_day_hashes'],hashes)
                with library.open_feature_view(descriptor) as view:
                    for day,original_descriptor in zip(days,descriptors):
                        with SavedColumnFeatureSource(original_descriptor,maximum_working_bytes=8*1024**2) as original:
                            plan=original.plan(catalog,selection,normalized=True); p=plan.to_dict()
                            inputs=original.prepare(plan,day,['A','B','C'],{s['id']:digest(['direct']) for s in selection})
                            old_ref=original.source.source_bindings()[-1]['id']
                            reference=current.source.source_bindings()[-1]
                            p['sources']=[b for b in p['sources'] if b['id']!=old_ref]+[reference]
                            inputs['source_bindings']=p['sources']
                            inputs['source_sets']=[sorted(reference['id'] if s==old_ref else s for s in group)
                                                   for group in inputs['source_sets']]
                            expected=execute_feature_columns(compile_feature_columns(FeaturePlan.from_dict(p)),
                                inputs=inputs,output_sessions=[day],max_working_bytes=8*1024**2)
                        actual=view.project([(s,day) for s in ('A','B','C')])
                        for key in ('values','validity','available_at_utc_us'):
                            np.testing.assert_array_equal(actual[key],expected[key][0])
                        for i in range(3):
                            for j in range(2):
                                self.assertEqual(actual['reason_dictionary'][int(actual['reason_codes'][i,j])],
                                    expected['reason_dictionary'][int(expected['reason_codes'][0,i,j])])
                                self.assertEqual(actual['source_sets'][int(actual['source_codes'][i,j])],
                                    expected['source_sets'][int(expected['source_codes'][0,i,j])])

    def test_layers_match_original_dag_and_reference_change_reuses_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); descriptor=saved_source(root); catalog=load_a01_feature_catalog()
            library=FeatureLibrary(root/'features',maximum_working_bytes=8*1024**2)
            selection=[{'id':name,'semantic_version':'1.0.0'} for name in ('ALP001','ALP003')]
            with SavedColumnFeatureSource(descriptor,maximum_working_bytes=8*1024**2) as source:
                facade=CatalogFeatureLibrary(library,source,catalog); days=source.source.manifest['spec']['feature_sessions']
                ref,stats=facade.ensure(selection=selection,output_sessions=days,normalized=True,sample_universe=['A'])
                self.assertEqual((stats['raw']['core_calls'],stats['cross_sectional']['core_calls']),(4,4))
                original=feature_column_plan(source.source,catalog,selection=selection,normalized=True)
                definitions=feature_definitions(original,input_semantics=source.input_semantics())
                with library.open_feature_view(ref) as view:
                    from axiom_research.stock_feature_library_adapter import (
                        build_feature_library_training_view,load_feature_library_training_view)
                    from axiom_research.stock_compact_store import iter_feature_rows
                    projection=build_feature_library_training_view(view,source,selection=selection,
                        destination=root/'training-view',maximum_working_bytes=8*1024**2)
                    with load_feature_library_training_view(projection,limits={'maximum_matrix_bytes':8*1024**2}) as training:
                        rows=list(iter_feature_rows(training,list(range(len(days)))))
                        direct=view.project()
                        np.testing.assert_array_equal([r['values'] for r in rows],direct['values'])
                        self.assertEqual(training.to_dict()['historical_feature_inputs_ref'],view.identity)
                    # Virtual cached-raw input retains the same full-DAG semantic identity.
                    self.assertEqual({n:d['semantic_ref'] for n,d in view.describe()['definitions'].items()},
                                     {n:d['semantic_ref'] for n,d in definitions.items()})
                    for day in days:
                        inputs=source.prepare(original,day,['A','B','C'],{n:digest(['direct']) for n in definitions})
                        expected=execute_feature_columns(compile_feature_columns(original),inputs=inputs,
                            output_sessions=[day],max_working_bytes=8*1024**2)
                        actual=view.project([(s,day) for s in ('A','B','C')])
                        np.testing.assert_array_equal(actual['values'],expected['values'][0])
                        np.testing.assert_array_equal(actual['validity'],expected['validity'][0])
                        np.testing.assert_array_equal(actual['available_at_utc_us'],expected['available_at_utc_us'][0])
                _,stats=facade.ensure(selection=selection[:1],output_sessions=days,normalized=True,sample_universe=['B'])
                self.assertEqual((stats['raw']['core_calls'],stats['cross_sectional']['core_calls']),(0,0))
            with SavedColumnFeatureSource(descriptor,reference_security_ids=['A','B'],maximum_working_bytes=8*1024**2) as source:
                _,stats=CatalogFeatureLibrary(library,source,catalog).ensure(selection=selection,
                    output_sessions=days,normalized=True,sample_universe=['B'])
                self.assertEqual((stats['raw']['core_calls'],stats['cross_sectional']['core_calls']),(0,4))


if __name__=='__main__': unittest.main()

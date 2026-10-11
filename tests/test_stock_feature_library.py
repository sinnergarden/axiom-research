"""One synthetic saved-source case: original owner admission, raw reuse, Core CS."""
from pathlib import Path
import tempfile
import unittest
import numpy as np

from axiom_engine.core import compile_feature_columns, execute_feature_columns
from axiom_engine.feature_library import FeatureLibrary, feature_definitions
from axiom_research import load_a01_feature_catalog, SavedColumnFeatureSource, CatalogFeatureLibrary
from axiom_research.stock_artifacts import digest, file_digest, write_json
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_feature_columns import REASONS, feature_column_plan
from axiom_research.stock_matrix_storage import write_buffer, write_part, instant_us


def saved_source(root):
    """Synthetic selected vendor-shaped cells; never used as real data evidence."""
    days=[f'2020-01-0{i}' for i in range(1,7)]; securities=['A','B','C']; shape=(6,3,6)
    values=np.empty(shape,dtype='<f8')
    for t in range(6):
        for j in range(3):
            opening=10*(j+1)+t
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
        'cutoff_by_session':{day:day+'T13:00:00Z' for day in days},'snapshot':'synthetic_only',
        'pit_policy':'best_effort_vendor_v1','catalog_ref':catalog.identity,'feature_selection':selection}
    impl={'fixture':digest(['explicit synthetic source'])}
    manifest=seal({'contract_version':'research_feature_column_source_v1','spec':spec,'qlib_view':{'universe_name':'csi300'},
        'buffers':buffers,'chunks':[{'sessions':days,'evidence':evidence}],'reason_codes':list(REASONS),
        'source_profiles':profiles,'selection_invariance':'declared_vendor_release_visible_at_each_historical_row_cutoff',
        'implementation_sources':impl,'implementation_ref':digest(impl),'environment':{'synthetic':True}},'source_ref')
    path=Path(root)/'source.json'; write_json(path,manifest)
    return {'path':str(path),'file_digest':file_digest(path),'source_ref':manifest['source_ref']}


class StockFeatureLibraryTests(unittest.TestCase):
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

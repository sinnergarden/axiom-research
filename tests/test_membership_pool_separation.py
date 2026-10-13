"""Single focused synthetic check for physical facts versus PIT reference proof."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

from axiom_engine.core import FeaturePlan, compile_feature_columns, execute_feature_columns
from axiom_engine.feature_library import FeatureLibrary
from axiom_research import SavedColumnFeatureSource, CatalogFeatureLibrary, load_a01_feature_catalog
from axiom_research.stock_feature_columns import _membership_masks
from axiom_research.stock_matrix_feature_producer import prepare_matrix_qlib
from test_conformance import setup, node, cs
from test_feature_columns import packed
from test_stock_feature_library import saved_source
from test_local_reader import (MemoryStore, membership_manifest, membership_row,
                               membership_query, SnapshotQueryReader, utc)


class MembershipPoolSeparation(unittest.TestCase):
    def test_physical_unknown_complete_incomplete_and_subset_reuse(self):
        # Data selects complete state at the requested clock; raw positives alone
        # cannot manufacture a complete pool after that state's interval ends.
        reader=SnapshotQueryReader(MemoryStore(membership_manifest(),{
            'history':[membership_row('again','2024-01-05',None,utc(5))]}),'s1')
        selected=reader.read(membership_query(cutoff=utc(6),sessions=('2024-01-02','2024-01-05')))
        pools=selected.field_meta['is_member']['reference_pools']
        self.assertTrue(pools['2024-01-02']['complete'])
        self.assertEqual(pools['2024-01-02']['member_count'],1)
        self.assertFalse(pools['2024-01-05']['complete'])
        invisible=reader.read(membership_query(cutoff=utc(1)))
        self.assertFalse(invisible.field_meta['is_member']['reference_pools']['2024-01-02']['complete'])

        # Known full pool A/B plus unknown outside C is complete only because
        # Data selected an explicit complete state declaring those two members.
        days=['2020-01-01','2020-01-02']
        series=pd.Series([True,True,None,True,True,None])
        ordered=[{'missing_reason':None if v is not None else 'source_missing'} for v in series]
        masks=_membership_masks(series,ordered,{'reference_pools':{
            days[0]:{'complete':True,'member_count':2},
            days[1]:{'complete':False,'member_count':None}}},days,(2,3))
        values,known,complete,reasons=masks
        self.assertEqual(complete[:,0].tolist(),[True,False])
        self.assertEqual(known[:,2].tolist(),[False,False])
        self.assertTrue((reasons[:,2]>0).all())
        self.assertFalse(_membership_masks(series,ordered,{},days,(2,3))[2].any())
        p,f,c=setup({'A':[10,11],'B':[20,21],'C':[100,101]},
            [node('raw','identity',['x']),cs('rank'),cs('z','cs_zscore',excluded='zero_if_undefined')],
            ['raw','rank','z'],members={'A':[True,True],'B':[True,True],'C':[False,False]})
        compiled=compile_feature_columns(FeaturePlan.from_dict(p)); inputs=packed(p,f,c)
        def run(inputs):
            return execute_feature_columns(compiled,inputs=inputs,output_sessions=c['sessions'],max_working_bytes=2**20)
        baseline=run(copy.deepcopy(inputs))
        all_known=copy.deepcopy(inputs)
        all_known.update(reference_member_known=np.ones((2,3),bool),reference_pool_complete=np.ones((2,3),bool))
        unchanged=run(all_known)
        for key in ('values','validity','reason_codes','available_at_utc_us','source_codes'):
            np.testing.assert_array_equal(baseline[key],unchanged[key])
        inputs.update(reference_member_known=known,reference_pool_complete=complete)
        result=run(inputs)
        self.assertTrue(result['validity'][:,:,0].all())
        np.testing.assert_array_equal(result['values'][:,:,0],[[10,20,100],[11,21,101]])
        np.testing.assert_array_equal(result['values'][0,:2,1:],baseline['values'][0,:2,1:])
        self.assertFalse(result['validity'][0,2,1:].any())
        self.assertFalse(result['validity'][1,:,1:].any())
        self.assertEqual(result['reason_dictionary'][result['reason_codes'][1,0,1]],'REFERENCE_MISSING')

        # Physical Qlib export omits the optional interval export, while keeping
        # the member QuerySpec for the independent Data read.
        data=Mock(); config={'symbols':['A','B','C'],'read_sessions':days,
            'snapshot':'s','pit_policy':'best_effort_vendor_v1','cutoff_by_session':{d:d+'T23:00:00Z' for d in days},
            'universe_id':'IDX','scope_ref':'scope'}
        with tempfile.TemporaryDirectory() as temp, patch('axiom_research.qlib_adapter.QlibView'):
            prepared=prepare_matrix_qlib(data,config=config,destination=temp,physical_only=True)
            self.assertIsNone(data.export_qlib.call_args.kwargs['universe_query'])
            self.assertEqual(prepared['member_query'].symbols,('A','B','C'))

        # Persist/admit the masks through the normal library and reuse a subset.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def mutate(a):
                a['member'][:,2]=False
                a['member_known']=np.ones(a['member'].shape,bool); a['member_known'][:,2]=False
                a['member_pool_complete']=np.ones(a['member'].shape,bool); a['member_pool_complete'][3,:]=False
                a['member_reasons']=np.zeros(a['member'].shape,dtype='u1'); a['member_reasons'][:,2]=1
            desc=saved_source(root/'source',mutate_arrays=mutate)
            with SavedColumnFeatureSource(desc,maximum_working_bytes=8*1024**2) as source:
                lib=CatalogFeatureLibrary(FeatureLibrary(root/'library',maximum_working_bytes=8*1024**2),source,load_a01_feature_catalog())
                selection=[{'id':n,'semantic_version':'1.0.0'} for n in ('ALP001','ALP003')]
                output=source.source.manifest['spec']['feature_sessions']
                lib.ensure(selection=selection,output_sessions=output,normalized=True)
                _,stats=lib.ensure(selection=selection[:1],output_sessions=output[-1:],normalized=True)
                self.assertEqual(stats['raw']['core_calls'],0)
                self.assertEqual(stats['cross_sectional']['core_calls'],0)

if __name__=='__main__': unittest.main()

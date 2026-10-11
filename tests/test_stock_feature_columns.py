import unittest

import numpy as np
import pandas as pd

from axiom_data import DataBatch,adjust_prices
from axiom_research.stock_artifacts import digest
from axiom_research.stock_feature_columns import FeatureColumnSource,REASONS
from axiom_research.stock_matrix_storage import instant_us


class FeatureColumnSourceTests(unittest.TestCase):
    def source(self):
        days=['2020-01-02','2020-01-03','2020-01-06']; securities=['A','B']
        shape=(3,2,6); values=np.zeros(shape,dtype='<f8')
        for t in range(3):
            values[t,:,:4]=np.asarray([[11.123456+t,13+t,10+t,12+t],[20+t,23+t,19+t,22+t]],dtype=np.float32).astype(np.float64)
            values[t,:,4]=10000+t; values[t,:,5]=[1<<t,2]
        valid=np.ones(shape,dtype=bool); valid[0,1,0]=False; values[0,1,0]=0
        values[2,1,5]=-1
        available=np.empty(shape,dtype='<i8')
        for t,day in enumerate(days): available[t]=instant_us(day+'T12:00:00.000001Z')
        known=np.ones(shape,dtype=bool); known[0,1,0]=False; known[0,:,4]=False
        reason=np.zeros(shape,dtype='u1'); reason[0,1,0]=1
        arrays={'values':values,'original_values':values.copy(),'validity':valid,
            'available':available,'available_validity':known,'reasons':reason,
            'member':np.ones(shape[:2],dtype=bool),'member_available':available[:,:,0].copy(),
            'member_available_validity':np.ones(shape[:2],dtype=bool)}
        spec={'read_sessions':days,'feature_sessions':[days[-1]],'universe':securities,
            'cutoff_by_session':{d:d+'T12:30:00.900000Z' for d in days},
            'snapshot':'s_fixture','pit_policy':'best_effort_vendor_v1'}
        manifest={'spec':spec,'source_ref':digest({'fixture':'columns'})}
        return FeatureColumnSource(manifest,arrays)

    def test_public_adjustment_values_and_missing_precedence(self):
        source=self.source(); spec=source.manifest['spec']; a=source.arrays
        fields=('open','high','low','close'); rows=[]; factors=[]; price_meta={f:[] for f in fields}; factor_meta=[]
        for t,day in enumerate(spec['read_sessions']):
            for s,security in enumerate(spec['universe']):
                row={'security_id':security,'session':day}
                for j,f in enumerate(fields):
                    row[f]=float(a['values'][t,s,j]) if a['validity'][t,s,j] else None
                    price_meta[f].append({'security_id':security,'session':day,
                        'usable_from':day+'T12:00:00.000001Z' if a['available_validity'][t,s,j] else None,
                        'missing_reason':REASONS[a['reasons'][t,s,j]]})
                rows.append(row); factors.append({'security_id':security,'session':day,'factor':float(a['values'][t,s,5])})
                factor_meta.append({'security_id':security,'session':day,'usable_from':day+'T12:00:00.000001Z','missing_reason':None})
        cutoff=spec['cutoff_by_session'][spec['feature_sessions'][0]]
        query={'symbols':spec['universe'],'sessions':spec['read_sessions'],'cutoff_by_session':{d:cutoff for d in spec['read_sessions']},
            'pit_policy':spec['pit_policy'],'purpose':'decision_facts','price_basis':'unadjusted','fields':list(fields)}
        context={'contract_version':'data_batch_v1','snapshot_id':spec['snapshot'],'query':query,
            'reader_version':'local_reader_v5'}
        prices=DataBatch(pd.DataFrame(rows),{f:{'dtype':'float64','unit':'CNY/share','by_key':price_meta[f]} for f in fields},
            {**context,'domain':'market_daily'})
        factor=DataBatch(pd.DataFrame(factors),{'factor':{'dtype':'float64','unit':'dimensionless','by_key':factor_meta}},
            {**context,'domain':'adjustment_factors','query':{**query,'fields':['factor']}})
        adjusted=adjust_prices(prices,factor,fields=fields,factor_field='factor',anchor_session=spec['feature_sessions'][0])
        actual=source.select(spec['feature_sessions'][0],history_sessions=3)
        for j,f in enumerate(fields):
            expected=adjusted.frame[f].to_numpy(dtype=np.float64).reshape(3,2)
            np.testing.assert_array_equal(actual['validity'][f],np.isfinite(expected))
            np.testing.assert_array_equal(actual['values'][f][actual['validity'][f]],expected[np.isfinite(expected)])
            by_key={(r['security_id'],r['session']):r for r in adjusted.field_meta[f]['by_key']}
            for t,day in enumerate(spec['read_sessions']):
                for s,security in enumerate(spec['universe']):
                    self.assertEqual(REASONS[actual['missing_reasons'][f][t,s]],by_key[security,day]['missing_reason'])
        self.assertEqual(actual['missing_reasons']['open'][0,1],1)
        self.assertEqual(actual['missing_reasons']['close'][0,1],7)

    def test_leaf_clock_nulls_and_core_second_precision(self):
        source=self.source(); day=source.manifest['spec']['feature_sessions'][0]
        view=source.select(day,history_sessions=3)
        # Missing one price clock still retains the factor/anchor maximum;
        # absence-only amount cells use this output cutoff, not their row cutoff.
        self.assertEqual(int(view['availability']['open'][0,1]),instant_us(day+'T12:00:01Z'))
        self.assertEqual(int(view['availability']['amount_cny'][0,0]),instant_us(day+'T12:30:00Z'))
        self.assertEqual(float(view['values']['amount_cny'][0,0]),10000.0)
        self.assertTrue(source.arrays['values'].flags.writeable is False)

    def test_adapter_to_public_core_columns(self):
        from axiom_engine.core import ABI,SEMANTICS,FeaturePlan,compile_feature_columns,execute_feature_columns
        source=self.source(); spec=source.manifest['spec']; day=spec['feature_sessions'][0]
        ref=digest(['synthetic adapter to public Core'])
        schema=[{'name':f,'dtype':'float64','unit':'CNY' if f=='amount_cny' else 'CNY/share',
            'stage':'fact','missing':'preserve'} for f in ('open','high','low','close','amount_cny')]
        output={'name':'out','dtype':'float64','unit':'CNY/share','stage':'base','missing':'preserve'}
        plan=FeaturePlan.from_dict({'abi':ABI,'semantics':SEMANTICS,'recipe_ref':ref,'calendar_ref':ref,
            'reference_ref':ref,'reference_members':{d:{s:None for s in spec['universe']} for d in spec['read_sessions']},
            'input_schema':schema,'event_schema':{},'sources':sorted(source.source_bindings(),key=lambda b:b['id']),
            'observation_domain':'sessions','history_policy':'partial',
            'nodes':[{'name':'out','op':'identity','version':'1','inputs':['close'],'params':{},'column':output}],
            'outputs':[{'node':'out','column':output}],'obligations':[]})
        inputs,selection=source.core_inputs(day,calendar_ref=ref,reference_ref=ref,schema=schema,history_sessions=3)
        result=execute_feature_columns(compile_feature_columns(plan),inputs=inputs,output_sessions=[day],max_working_bytes=1024**2)
        self.assertEqual(result['values'].shape,(1,2,1))
        self.assertEqual(result['validity'][0,:,0].tolist(),[True,False])
        self.assertEqual(float(result['values'][0,0,0]),14.0)
        self.assertEqual(result['reason_dictionary'][result['reason_codes'][0,1,0]],'invalid_anchor_factor')
        self.assertEqual(result['input_refs'],{'source_ref':source.identity,'selection_ref':digest(selection)})


if __name__=='__main__': unittest.main()

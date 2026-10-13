"""One bounded interruption/recovery check; no supplier or model execution."""
import copy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from axiom_data import DataBatch, QuerySpec
from axiom_research.stock_feature_columns import build_feature_column_source
from axiom_research.stock_matrix_feature_producer import FIELDS


class FeatureInputCheckpointTests(unittest.TestCase):
    def test_interruption_resume_equivalence_and_no_repeat(self):
        days=['2020-01-02','2020-01-03','2020-01-06']; symbols=['A','B']
        spec={'pit_policy':'best_effort_vendor_v1','snapshot':'fixture','read_sessions':days,
              'feature_sessions':days,'universe':symbols,
              'cutoff_by_session':{d:d+'T20:30:00+00:00' for d in days}}
        price=QuerySpec('market_daily',FIELDS[:-1],tuple(symbols),tuple(days),spec['pit_policy'],spec['cutoff_by_session'])
        factor=QuerySpec('adjustment_factors',('factor',),tuple(symbols),tuple(days),spec['pit_policy'],spec['cutoff_by_session'])
        member=QuerySpec('universe_membership',('is_member',),tuple(symbols),tuple(days),spec['pit_policy'],spec['cutoff_by_session'],universe_id='IDX')
        def value(day,symbol,field): return 1.0 if field=='factor' else float(10+days.index(day)*3+symbols.index(symbol))
        class Data:
            calls=[]
            def describe_snapshot(self,**kw):
                return {'domains':{n:{'contract':{'contract_id':'local.tushare_dated_membership.v1'},
                    'source_profile':{'id':'tushare.index_weight.dated_membership.v1'}} for n in kw['domains']}}
            def read(self,*,query,**kw):
                self.calls.append((query.domain,query.sessions))
                keys=[(d,s) for d in query.sessions for s in query.symbols]
                rows=[{'session':d,'security_id':s,**{f:True if f=='is_member' else value(d,s,f) for f in query.fields}} for d,s in keys]
                meta={f:{'dtype':'bool' if f=='is_member' else 'float64','unit':None,
                    'by_key':[{'session':d,'security_id':s,'usable_from':d+'T12:00:00+00:00',
                               'availability_basis':'declared_vendor_assumption','missing_reason':None} for d,s in keys]} for f in query.fields}
                if 'is_member' in meta: meta['is_member']['reference_pools']={d:{'complete':True,'member_count':2} for d in query.sessions}
                return DataBatch(pd.DataFrame(rows),meta,{'domain':query.domain})
            members=read
        class View:
            def read(self,*,fields,symbols,start,end):
                selected=[d for d in days if start<=d<=end]
                index=pd.MultiIndex.from_tuples([(s,d) for d in selected for s in symbols],names=['instrument','datetime'])
                return pd.DataFrame({ '$'+f:[value(d,s,f) for d in selected for s in symbols] for f in fields},index=index)
        prepared={'view':View(),'view_reference':{'instrument_map':{'A':'A','B':'B'},'snapshot':'fixture'},
                  'price_query':price,'factor_query':factor,'member_query':member,'seconds':0.0}
        data=Data()
        with tempfile.TemporaryDirectory() as temp, \
             patch('axiom_research.stock_feature_columns._source_config',return_value={}), \
             patch('axiom_research.stock_feature_columns.prepare_matrix_qlib',return_value=prepared), \
             patch('axiom_research.stock_ml._implementation',return_value={'fixture':'stable'}), \
             patch('axiom_research.stock_ml._environment',return_value={'fixture':True}), \
             patch('axiom_data.reader._release_time',side_effect=lambda profile,day:datetime.fromisoformat(day+'T12:00:00+00:00')):
            root=Path(temp); target=root/'resumed'
            baseline,_=build_feature_column_source(data,spec=spec,destination=root/'baseline',block_sessions=1)
            data.calls.clear()
            def interrupt(p): raise RuntimeError('fixture interruption after durable block')
            with self.assertRaisesRegex(RuntimeError,'fixture interruption'):
                build_feature_column_source(data,spec=spec,destination=target,block_sessions=1,checkpoint=True,progress=interrupt)
            self.assertEqual(len(data.calls),3)
            receipt=json.loads((target/'.feature-input-checkpoint.json').read_text())
            self.assertEqual(receipt['completed_dates'],1)
            changed=copy.deepcopy(spec); changed['snapshot']='other'
            with self.assertRaisesRegex(ValueError,'binding/layout mismatch'):
                build_feature_column_source(data,spec=changed,destination=target,block_sessions=1,checkpoint=True)
            path=Path(receipt['files']['values']); original=path.read_bytes()
            path.write_bytes(original[:-1])
            with self.assertRaisesRegex(ValueError,'missing/truncated'):
                build_feature_column_source(data,spec=spec,destination=target,block_sessions=1,checkpoint=True)
            path.write_bytes(original)
            data.calls.clear()
            resumed,_=build_feature_column_source(data,spec=spec,destination=target,block_sessions=1,checkpoint=True)
            self.assertEqual(len(data.calls),6)
            self.assertTrue(all(days[0] not in query_days for _,query_days in data.calls))
            for name in baseline.arrays: np.testing.assert_array_equal(baseline.arrays[name],resumed.arrays[name])
            def content(chunks):
                return [(c['sessions'],[(e['file_digest'],e['evidence_ref']) for e in c['evidence']]) for c in chunks]
            self.assertEqual(content(baseline.manifest['chunks']),content(resumed.manifest['chunks']))
            before=(target/'source.json').read_bytes(); data.calls.clear()
            again,stats=build_feature_column_source(data,spec=spec,destination=target,block_sessions=1,checkpoint=True)
            self.assertEqual(data.calls,[]); self.assertEqual(stats['data_read_calls'],0)
            self.assertEqual((target/'source.json').read_bytes(),before)
            self.assertEqual(again.identity,resumed.identity)
            # One tiny child also proves the supervisor records its actual PID
            # and nonzero exit, without executing the history runner.
            supervisor=Path(__file__).resolve().parents[1]/'tools/supervise_qsys_history.py'
            run=subprocess.run([sys.executable,str(supervisor),'--state-dir',str(root/'supervisor'),
                '--progress',str(root/'absent-progress'),'--',sys.executable,'-c',
                'import os,sys; print("fixture_child_pid="+str(os.getpid()),flush=True); sys.exit(3)'],capture_output=True,text=True)
            self.assertEqual(run.returncode,3,run.stderr)
            state=json.loads((root/'supervisor/current.json').read_text())
            self.assertEqual(state['returncode'],3); self.assertEqual(state['status'],'finished')
            self.assertIn('fixture_child_pid='+str(state['child_pid']),(root/'supervisor/worker.log').read_text())

if __name__=='__main__': unittest.main()

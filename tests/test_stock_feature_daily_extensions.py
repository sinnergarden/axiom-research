"""Synthetic typed extensions: units, shared anchors, saved admission and reuse."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest

import numpy as np
import pandas as pd

from axiom_research.stock_feature_columns import extend_feature_column_source, REASONS
from axiom_research.stock_feature_library import SavedColumnFeatureSource, CatalogFeatureLibrary
from axiom_research.stock_artifacts import file_digest, digest
from axiom_research import load_a01_feature_catalog
from axiom_engine.feature_library import FeatureLibrary
from test_stock_feature_library import saved_source


class DailyExtensionTests(unittest.TestCase):
    def extend(self, source, root, *, unit='shares', late=False, zero=False, advance=False, change_old=False):
        spec=source.manifest['spec']; profiles=deepcopy(source.manifest['source_profiles'])
        profiles['market_daily']['contract']['fields']['volume_shares']={'dtype':'int64','unit':unit}
        calls=[]
        def read(*,snapshot,query):
            calls.append(query)
            rows=[{'security_id':s,'session':d,'volume_shares':0 if zero and s=='A' else 100}
                  for d in query.sessions for s in query.symbols]
            meta=[{'security_id':r['security_id'],'session':r['session'],'revision_id':'volume:'+r['session'],
                   'missing_reason':None,'usable_from':r['session']+('T14:00:00Z' if late else 'T12:00:00Z'),
                   'availability_basis':'declared_vendor_assumption'} for r in rows]
            return SimpleNamespace(frame=pd.DataFrame(rows),field_meta={'volume_shares':
                {'dtype':'int64','unit':unit,'by_key':meta}},context={'domain':'market_daily'})
        candidate=deepcopy(profiles)
        if change_old: candidate['adjustment_factors']['source_profile']['changed']=True
        def describe_snapshot(*,snapshot,domains,include_domain_refs=False):
            manifest=candidate if snapshot=='candidate' else profiles
            return {'snapshot_id':snapshot,'domains':{name:{**deepcopy(manifest[name]),
                **({'domain_ref':digest(manifest[name])} if include_domain_refs else {})} for name in domains}}
        data=SimpleNamespace(describe_snapshot=describe_snapshot,read=read)

        with patch('axiom_data.reader._release_time',side_effect=lambda p,d:pd.Timestamp(d+'T12:00:00Z')):
            result,metrics=extend_feature_column_source(data,source=source,
                daily_fields={'volume_shares':{'domain':'market_daily','dtype':'int64','unit':'shares'}},
                destination=root,block_sessions=2,snapshot='candidate' if advance else None)
        return result,metrics,calls

    def test_saved_append_reuses_old_buffers_and_cached_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); desc=saved_source(root/'base')
            library=FeatureLibrary(root/'features',maximum_working_bytes=8*1024**2)
            catalog=load_a01_feature_catalog(); selection=[{'id':'ALP001','semantic_version':'1.0.0'}]
            with SavedColumnFeatureSource(desc,maximum_working_bytes=8*1024**2) as old:
                days=old.source.manifest['spec']['feature_sessions']
                _,first=CatalogFeatureLibrary(library,old,catalog).ensure(selection=selection,output_sessions=days)
                expanded,metrics,calls=self.extend(old.source,root/'extended',advance=True)
                self.assertEqual(expanded.manifest['spec']['snapshot'],'candidate')
                self.assertEqual(metrics['existing_field_reads'],0)
                self.assertTrue(all(q.fields==('volume_shares',) for q in calls))
                for name,buffer in old.source.manifest['buffers'].items():
                    self.assertEqual(expanded.manifest['buffers'][name],buffer)
                path=root/'extended'/'source.json'
                new_desc={'path':str(path),'file_digest':file_digest(path),'source_ref':expanded.identity}
                with SavedColumnFeatureSource(new_desc,maximum_working_bytes=8*1024**2) as new:
                    _,second=CatalogFeatureLibrary(library,new,catalog).ensure(selection=selection,output_sessions=days)
                    self.assertEqual(second['raw']['core_calls'],0)
                    self.assertEqual(second['raw']['computed_output_cells'],0)
                    self.assertEqual(old.input_semantics()['close'],new.input_semantics()['close'])
                    view=new.source.select(days[-1],history_sessions=3,fields=['vwap','volume_shares'])
                    a=new.source.arrays; factor=a['values'][-3:,:,5]/a['values'][-1:,:,5]
                    expected=a['original_values'][-3:,:,4]/100*factor
                    np.testing.assert_array_equal(view['values']['vwap'],expected)
                    self.assertTrue(view['validity']['vwap'].all())
                    schema=[c for c in new.source.input_schema() if c['name'] in ('vwap','volume_shares')]
                    inputs,_=new.source.core_inputs(days[-1],calendar_ref='x',reference_ref='x',schema=schema,history_sessions=3)
                    self.assertEqual(len(inputs['source_sets'][int(inputs['source_codes']['vwap'][0,0])]),3)
                    from axiom_engine.core import ABI, SEMANTICS, FeaturePlan, compile_feature_columns, execute_feature_columns
                    from axiom_research.stock_artifacts import digest
                    ref=digest(['synthetic extended core'])
                    output={'name':'out','dtype':'float64','unit':'CNY/share','stage':'base','missing':'preserve'}
                    plan=FeaturePlan.from_dict({'abi':ABI,'semantics':SEMANTICS,'recipe_ref':ref,
                        'calendar_ref':ref,'reference_ref':ref,'input_schema':schema,'event_schema':{},
                        'sources':sorted(new.source.source_bindings(),key=lambda b:b['id']),
                        'observation_domain':'sessions','history_policy':'partial','reference_members':{d:{s:None for s in ['A','B','C']} for d in new.source.manifest['spec']['read_sessions']},
                        'nodes':[{'name':'out','op':'identity','version':'1','inputs':['vwap'],'params':{},'column':output}],
                        'outputs':[{'node':'out','column':output}],'obligations':[]})
                    inputs,_=new.source.core_inputs(days[-1],calendar_ref=ref,reference_ref=ref,schema=schema,history_sessions=3)
                    result=execute_feature_columns(compile_feature_columns(plan),inputs=inputs,
                        output_sessions=[days[-1]],max_working_bytes=8*1024**2)
                    np.testing.assert_array_equal(result['values'][0,:,0],expected[-1])


    def test_unit_and_late_clock_rejection_and_zero_volume_reason(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with SavedColumnFeatureSource(saved_source(root/'base'),maximum_working_bytes=8*1024**2) as old:
                with self.assertRaisesRegex(ValueError,'changes reused source domains'):
                    self.extend(old.source,root/'changed',advance=True,change_old=True)
                with self.assertRaisesRegex(ValueError,'dtype/unit mismatch'):
                    self.extend(old.source,root/'wrong',unit='lots')
                with self.assertRaisesRegex(ValueError,'selection invariance'):
                    self.extend(old.source,root/'late',late=True)
                source,_,_=self.extend(old.source,root/'zero',zero=True)
                view=source.select(source.manifest['spec']['feature_sessions'][-1],history_sessions=2,fields=['vwap'])
                self.assertFalse(view['validity']['vwap'][:,0].any())
                self.assertEqual(REASONS[int(view['missing_reasons']['vwap'][0,0])],'invalid_price')


if __name__=='__main__': unittest.main()

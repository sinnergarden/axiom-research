"""Synthetic revision boundaries through actual Data and the saved Feature path."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_data import Data, EventQuery
from axiom_research import extend_financial_feature_source
from axiom_research.stock_feature_library import SavedColumnFeatureSource, CatalogFeatureLibrary
from axiom_research.qsys_catalog import load_qsys_feature_catalog, select_qsys_features
from axiom_research.stock_artifacts import file_digest
from axiom_engine.feature_library import FeatureLibrary, FeatureRequest
from test_stock_feature_library import saved_source


def events(root, *, missing=False):
    data=Data(root); fields={n:{'dtype':'string'} for n in ('security_id','endpoint','report_type','report_period','announcement_date','revenue__status')}
    fields['revenue']={'dtype':'float64','unit':'CNY','basis':'cumulative_ytd','status_field':'revenue__status'}
    contract={'contract_id':'synthetic.financial.v1','logical_key':['security_id','endpoint','report_type','report_period'],'fields':fields}
    profile={'id':'synthetic.financial.v1','revision_order':'source_sequence_only','availability':
        {'date_field':'announcement_date','date_rule':'same_day_release','timezone':'Asia/Shanghai','session_release_time':'09:30:00'}}
    rows=[]
    for security in ('A','B','C'):
        for year in (2018,2019):
            for quarter,suffix in enumerate(('03-31','06-30','09-30','12-31'),1):
                if missing and year==2018 and quarter==2: continue
                period=f'{year}-{suffix}';value=(10 if year==2018 else 20)*quarter
                rows.append(dict(security_id=security,endpoint='income',report_type='1',report_period=period,
                    announcement_date='2020-01-01',revenue=value,revenue__status='value',revision_id=f'{security}:{period}:r1',
                    revision_sequence=1,first_observed_at=datetime(2020,1,1,tzinfo=timezone.utc),raw_batch_id='synthetic'))
        for i,(day,value,status) in enumerate((('2020-01-03',120,'value'),('2020-01-05',None,'retracted')),2):
            rows.append(dict(security_id=security,endpoint='income',report_type='1',report_period='2019-12-31',
                announcement_date=day,revenue=value,revenue__status=status,revision_id=f'{security}:2019FY:r{i}',
                revision_sequence=i,first_observed_at=datetime(2020,1,1,tzinfo=timezone.utc),raw_batch_id='synthetic'))
    part=data.store.write_partition('financial_events','synthetic-reports',rows,contract)
    snapshot=data.store.publish_snapshot({'financial_events':{'contract':contract,'source_profile':profile,
        'partitions':[part],'raw_batch_ids':[],'coverage':{},'build_context':{'synthetic':True}}},
        parent_snapshot=None,build_context={'synthetic':True},promote=False)['snapshot_id']
    return data,snapshot


class FinancialFeatureColumnsTests(unittest.TestCase):
    def build(self, root, *, missing=False, covered=('A','B','C')):
        data,snapshot=events(root/'events',missing=missing)
        base=saved_source(root/'base')
        old=SavedColumnFeatureSource(base,maximum_working_bytes=16*1024**2)
        query=EventQuery('financial_events',('revenue',),covered,'2018-01-01','2019-12-31',
            '2020-01-06T13:00:00Z','best_effort_vendor_v1','report_period',{'endpoint':'income','report_type':'1'})
        with patch.object(data.store,'read_partition',wraps=data.store.read_partition) as reads:
            expanded,metrics=extend_financial_feature_source(data,source=old.source,query=query,field='revenue',
                outputs={'ttm_revenue':{'derivation':'ttm','quarters_back':0},
                         'ttm_revenue_prior_year':{'derivation':'ttm','quarters_back':4}},unit='CNY',
                destination=root/'expanded',event_snapshot=snapshot)
            self.assertEqual(reads.call_count,1)  # shared across six cutoffs and both outputs
        self.assertEqual(expanded.manifest['spec']['snapshot'],'synthetic_only')
        self.assertEqual(expanded.manifest['financial_materializations']['ttm_revenue']['event_snapshot'],snapshot)
        for name,desc in old.source.manifest['buffers'].items():self.assertEqual(expanded.manifest['buffers'][name],desc)
        old.close()
        path=root/'expanded'/'source.json'
        desc={'path':str(path),'file_digest':file_digest(path),'source_ref':expanded.identity}
        return SavedColumnFeatureSource(desc,maximum_working_bytes=16*1024**2),metrics

    def test_revision_same_day_withdraw_and_formal_library_subset_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,metrics=self.build(root)
            with source:
                catalog=load_qsys_feature_catalog(); selection=select_qsys_features(['ttm_revenue_yoy'])
                library=FeatureLibrary(root/'features',maximum_working_bytes=16*1024**2)
                days=source.source.manifest['spec']['feature_sessions']
                descriptor,first=CatalogFeatureLibrary(library,source,catalog).ensure(selection=selection,output_sessions=days)
                with library.open_feature_view(descriptor) as view:
                    out=view.project([('A',d) for d in days])
                    self.assertEqual(out['validity'][:,0].tolist(),[True,True,True,False])
                    self.assertEqual(out['values'][:,0].tolist(),[1.,2.,2.,0.])
                _,repeat=CatalogFeatureLibrary(library,source,catalog).ensure(selection=selection,output_sessions=days[1:3],sample_universe=['A'])
                self.assertEqual(repeat['raw']['core_calls'],0)
                self.assertEqual(repeat['raw']['computed_output_cells'],0)
                import json
                evidence=source.source.manifest['daily_extensions'][-1]['chunks'][0]['evidence'][0]
                child=json.loads(Path(evidence['path']).read_text())
                rows=child['field_meta']['ttm_revenue']['by_key']
                last=next(r for r in rows if r['security_id']=='A' and r['session']==days[-1])
                self.assertEqual(last['financial_missing_reason'],'retracted')
                self.assertTrue(any(c['provenance'] and c['provenance']['revision_id']=='A:2019FY:r3' for c in last['components']))
                self.assertEqual(last['report_period'],'2019-12-31')
                self.assertEqual(metrics['event_cutoff_selections'],6)

    def test_component_cutoff_and_declaration_cannot_be_forged(self):
        from axiom_research.financial_feature_columns import validate_financial_materializations
        import json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,_=self.build(root)
            with source:
                manifest=source.source.manifest
                desc=manifest['daily_extensions'][-1]['chunks'][0]['evidence'][0]
                child=json.loads(Path(desc['path']).read_text())
                row=next(r for r in child['field_meta']['ttm_revenue']['by_key'] if r['components'])
                row['components'][0]['provenance']['usable_from']='2021-01-01T00:00:00Z'
                with self.assertRaisesRegex(ValueError,'future financial'):
                    validate_financial_materializations(manifest,child)
                bad=deepcopy(manifest);bad['financial_materializations']={}
                with self.assertRaisesRegex(ValueError,'every derived field'):
                    validate_financial_materializations(bad)

    def test_missing_quarter_and_uncovered_security_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,_=self.build(root,missing=True,covered=('A',))
            with source:
                day=source.source.manifest['spec']['feature_sessions'][0]
                with self.assertRaisesRegex(ValueError,'coverage|covered'):
                    source.source.select(day,history_sessions=1,fields=['ttm_revenue'],securities=['B'])
                catalog=load_qsys_feature_catalog();selection=select_qsys_features(['ttm_revenue_yoy'])
                plan=source.plan(catalog,selection,normalized=False)
                from axiom_engine.feature_library.identity import select_security_plan
                plan=select_security_plan(plan,['A'])
                request=FeatureRequest(plan,(day,),('A',),source.input_semantics())
                lib=FeatureLibrary(root/'features',maximum_working_bytes=16*1024**2)
                desc,_=lib.ensure_features(request,source)
                with lib.open_feature_view(desc) as view:self.assertFalse(view.project()['validity'][0,0])


if __name__=='__main__':unittest.main()

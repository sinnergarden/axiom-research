"""Scoped joint-input/reuse failures; Core executes, Data owns revision choice."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_data import QuerySpec, EventQuery
from axiom_research import ArtifactRef, ViewRef, validate
from axiom_research.data_adapter import AdapterError
from axiom_research.joint_build import build_joint_features
from test_data_adapter import Batch, batch, SESSIONS

IMPLEMENTATION = ArtifactRef(artifact_type='ImplementationPackage', artifact_id='synthetic-review-v1',
    artifact_contract_version='1', content_digest='sha256:'+'b'*64, uri='fixture:reviewed-package')
CUTOFFS = {s:s+'T23:00:00Z' for s in SESSIONS}
MARKET = QuerySpec('daily_bars', ('close',), ('A',), tuple(SESSIONS), 'market_pit_safe_v1', CUTOFFS)
MEMBERS = replace(MARKET, domain='universe_membership', fields=('is_member',), universe_id='U')
FINANCE = EventQuery('financial_events', ('total_revenue',), ('A',), '2024-12-31','2024-12-31',
    CUTOFFS[SESSIONS[-1]], 'market_pit_safe_v1','report_period', {'endpoint':'income','report_type':'1'})


class Store:
    def load_snapshot(self, snapshot):
        return {'snapshot_id':snapshot,'domains': {
            'daily_bars': {'contract':{'fields':{'close':{'dtype':'float64','unit':'CNY/share'}}}},
            'financial_events': {'contract':{'logical_key':['security_id','endpoint','report_type','report_period'], 'fields':{'total_revenue':{'dtype':'float64','unit':'CNY'}}}}}}


class DataFixture:
    store = Store()
    def __init__(self, *, retracted=False, absent=False):
        self.calls=[];self.retracted=retracted;self.absent=absent

    def read(self, *, snapshot, query):
        self.calls.append('read')
        b = batch('close',[10,None,20]);b.wire['context']['snapshot_id']=snapshot
        b.wire['context']['query']['cutoff_by_session']=dict(query.cutoff_by_session)
        return b

    def members(self, *, snapshot, query):
        self.calls.append('members')
        b = batch('is_member',[True]*3,membership=True);b.wire['context']['snapshot_id']=snapshot
        b.wire['context']['query']['cutoff_by_session']=dict(query.cutoff_by_session)
        return b

    def events(self, *, snapshot, query):
        self.calls.append(('events',query.cutoff))
        corrected=str(query.cutoff)[:10]==SESSIONS[-1]
        keys=dict(security_id='A',endpoint='income',report_type='1',report_period='2024-12-31')
        value=None if self.retracted and corrected else 120 if corrected else 100
        clock=(SESSIONS[-1] if corrected else SESSIONS[0])+'T12:00:00Z'
        record={**keys,'total_revenue':value}
        meta={**keys,'usable_from':clock,'first_observed_at':clock,'revision_id':'r2' if corrected else 'r1',
            'raw_batch_id':'raw2' if corrected else 'raw1','status':'retracted' if value is None else 'value',
            'missing_reason':'retracted' if value is None else None,'availability_basis':'synthetic', 'evidence_ref':'fixture'}
        absent=self.absent and str(query.cutoff)[:10]==SESSIONS[0]
        return Batch({'records':[] if absent else [record],
            'field_meta':{'total_revenue':{'dtype':'float64','unit':'CNY','by_key':[] if absent else [meta]}},
            'context':{'contract_version':'data_batch_v1','snapshot_id':snapshot,'domain':'financial_events',
                'reader_version':'local_reader_v5','logical_key':list(keys),
                'query':{'fields':['total_revenue'],'symbols':['A'],'start':query.start,'end':query.end,
                    'cutoff':query.cutoff,'time_field':'report_period','filters':dict(query.filters),
                    'pit_policy':query.pit_policy,'purpose':query.purpose}}})


def build(data,destination,**changes):
    params=dict(snapshot='s1',market_queries={'daily':MARKET},membership_query=MEMBERS,financial_queries={'income':FINANCE},
        destination=destination,implementation_package=IMPLEMENTATION)
    params.update(changes)
    return build_joint_features(data,**params)


class JointBuildTests(unittest.TestCase):
    def test_cutoff_revisions_core_asof_provenance_and_cross_process_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            data=DataFixture()
            first=build(data,directory)
            self.assertFalse(first.reused)
            self.assertEqual(first.read()['income__total_revenue'].tolist(),[100,100,120])
            self.assertTrue(first.read()['daily__close__return'].isna().all())
            evidence=first.evidence()
            self.assertEqual([call[1] for call in data.calls if isinstance(call,tuple)],list(CUTOFFS.values()))
            revisions={m['provenance']['revision_id'] for m in evidence['source_evidence'].values()}
            self.assertTrue({'r1','r2'}<=revisions)
            self.assertEqual(evidence['frames'][0]['schema'][-1]['unit'],'CNY')
            validate(first.identity)
            class NoReads(DataFixture):
                def read(self,**kwargs):raise AssertionError('reused build must not reread Data')
                members=read
                events=read
            with patch('axiom_research.joint_build.execute_feature_plan',side_effect=AssertionError('no Core rerun')):
                second=build(NoReads(),directory)
                self.assertTrue(second.reused)
                self.assertEqual(second.read().to_json(),first.read().to_json())
            different=build(DataFixture(),directory,lag_sessions=2)
            self.assertNotEqual(different.path,first.path)
            expected=first.read().to_json()
            moved=Path(directory)/'relocated';first.path.rename(moved)
            from axiom_research.joint_build import FeatureBuild, load_feature_build
            self.assertEqual(load_feature_build(moved).read().to_json(),expected)

    def test_absent_and_retracted_are_not_filled(self):
        with tempfile.TemporaryDirectory() as directory:
            result=build(DataFixture(absent=True,retracted=True),directory)
            values=result.read()['income__total_revenue']
            self.assertTrue(values.isna().iloc[0]);self.assertEqual(values.iloc[1],100)
            self.assertTrue(values.isna().iloc[2])
            self.assertIn('income:financial_events:'+SESSIONS[0],result.evidence()['query_contexts'])
            frames=result.evidence()['frames']
            self.assertIn('NO_VISIBLE_EVENT',frames[0]['rows'][0]['reasons'][-1])
            self.assertIn('retracted',frames[-1]['rows'][0]['reasons'][-1])

    def test_multi_period_selection_keeps_new_report_retraction_and_ignores_old_correction(self):
        class MultiPeriod(DataFixture):
            def events(self, *, snapshot, query):
                wire = deepcopy(super().events(snapshot=snapshot,query=query).wire)
                recent = wire['records'][0]
                recent_meta = wire['field_meta']['total_revenue']['by_key'][0]
                old = {**recent,'report_period':'2024-09-30','total_revenue':999}
                old_meta = {**recent_meta,'report_period':'2024-09-30',
                    'revision_id':'old-report-late-correction','usable_from':'2025-01-02T12:00:00Z',
                    'missing_reason':None,'status':'value'}
                if str(query.cutoff)[:10]==SESSIONS[0]:
                    wire['records']=[];wire['field_meta']['total_revenue']['by_key']=[]
                    old['total_revenue']=50
                elif str(query.cutoff)[:10]==SESSIONS[1]:
                    recent_meta['usable_from']=SESSIONS[1]+'T12:00:00Z'
                # Unsorted input proves report-period selection, not row order.
                wire['records'].append(old)
                wire['field_meta']['total_revenue']['by_key'].append(old_meta)
                return Batch(wire)
        with tempfile.TemporaryDirectory() as directory:
            result=build(MultiPeriod(retracted=True),directory,
                financial_queries={'income':replace(FINANCE,start='2020-01-01')})
            values=result.read()['income__total_revenue']
            self.assertEqual(values.iloc[:2].tolist(),[50,100])
            self.assertTrue(values.isna().iloc[2])
            financial=[e for e in result.evidence()['source_evidence'].values() if e['field']=='total_revenue']
            self.assertEqual(sorted({e['provenance']['report_period'] for e in financial}),
                ['2024-09-30','2024-12-31'])
            self.assertTrue(all(result.evidence()['query_contexts'][e['query_context_ref']]['query']['start']=='2020-01-01' for e in financial))

    def test_integrity_and_partial_publication_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('pyarrow.parquet.write_table',side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError,'disk full'):build(DataFixture(),directory)
            self.assertEqual(list(Path(directory).iterdir()),[])
            result=build(DataFixture(),directory)
            (result.path/'panel.parquet').write_bytes(b'corrupt')
            with self.assertRaisesRegex(AdapterError,'integrity'):build(DataFixture(),directory)

    def test_multiple_fields_and_domains_merge_by_key(self):
        class MultiStore(Store):
            def load_snapshot(self,snapshot):
                result=super().load_snapshot(snapshot)
                result['domains']['daily_bars']['contract']['fields']['volume']={'dtype':'float64','unit':'shares'}
                result['domains']['valuation']={'contract':{'fields':{'pb':{'dtype':'float64','unit':'dimensionless'}}}}
                result['domains']['financial_events']['contract']['fields']['parent_net_income']={'dtype':'float64','unit':'CNY'}
                result['domains']['balance_sheet_events']={'contract':{'logical_key':['security_id','endpoint','report_type','report_period'],
                    'fields':{'total_assets':{'dtype':'float64','unit':'CNY'}}}}
                return result
        class MultiData(DataFixture):
            store=MultiStore()
            def read(self,*,snapshot,query):
                wire=deepcopy(super().read(snapshot=snapshot,query=query).wire)
                wire['context']['domain']=query.domain;wire['context']['query']['fields']=list(query.fields)
                values={'close':[10,None,20],'volume':[100,110,121],'pb':[1,2,3]}
                base=deepcopy(wire['field_meta']['close'])
                wire['field_meta']={field:{**deepcopy(base),'unit':self.store.load_snapshot(snapshot)['domains'][query.domain]['contract']['fields'][field]['unit']}
                    for field in query.fields}
                wire['records']=[{'security_id':'A','session':session,**{f:values[f][i] for f in query.fields}}
                    for i,session in enumerate(SESSIONS)]
                for field,spec in wire['field_meta'].items():
                    for meta in spec['by_key']:
                        value=values[field][SESSIONS.index(meta['session'])]
                        meta['usable_from']=meta['session']+'T10:00:00Z';meta['first_observed_at']=meta['usable_from']
                        meta['missing_reason']='source_missing' if value is None else None
                wire['records'].reverse()
                return Batch(wire)
            def events(self,*,snapshot,query):
                wire=deepcopy(super().events(snapshot=snapshot,query=query).wire)
                wire['context']['domain']=query.domain;wire['context']['query']['fields']=list(query.fields)
                wire['context']['query']['filters']=dict(query.filters)
                base=wire['field_meta']['total_revenue']
                for row in wire['records']:
                    row['endpoint']=query.filters['endpoint']
                    row['parent_net_income']=row['total_revenue']/10
                    row['total_assets']=row['total_revenue']*10
                wire['field_meta']={field:deepcopy(base) for field in query.fields}
                for spec in wire['field_meta'].values():
                    for meta in spec['by_key']:meta['endpoint']=query.filters['endpoint']
                return Batch(wire)
        with tempfile.TemporaryDirectory() as directory:
            result=build(MultiData(),directory,
                market_queries={'daily':replace(MARKET,fields=('close','volume')),
                    'valuation':replace(MARKET,domain='valuation',fields=('pb',))},
                financial_queries={'income':replace(FINANCE,fields=('total_revenue','parent_net_income')),
                    'balance':replace(FINANCE,domain='balance_sheet_events',fields=('total_assets',),
                        filters={'endpoint':'balancesheet','report_type':'1'})})
            panel=result.read()
            self.assertEqual(panel['session'].tolist(),SESSIONS)
            self.assertEqual(panel['daily__volume'].tolist(),[100,110,121])
            self.assertAlmostEqual(panel['daily__volume__return'].iloc[1],0.1)
            self.assertEqual(panel['valuation__pb'].tolist(),[1,2,3])
            self.assertEqual(panel['income__parent_net_income'].tolist(),[10,10,12])
            self.assertEqual(panel['balance__total_assets'].tolist(),[1000,1000,1200])
            self.assertEqual(len(panel.columns),11)

    def test_scope_and_event_view_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AdapterError,'ordered report range'):
                build(DataFixture(),directory,financial_queries={'income':replace(FINANCE,start='2025-01-01')})
            with self.assertRaisesRegex(AdapterError,'decision_facts'):
                build(DataFixture(),directory,financial_queries={'income':replace(FINANCE,purpose='label_outcomes')})
            data=DataFixture()
            event=data.events(snapshot='s1',query=FINANCE)
            view=ViewRef.from_batch(event)
            self.assertEqual(view.query['time_field'],'report_period')
            validate(view.as_artifact_ref())


if __name__=='__main__':unittest.main()

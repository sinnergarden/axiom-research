"""Small numerical Core checks and exact per-load Raw reuse; no provider I/O."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_research.stock_artifacts import digest, file_digest, write_json, _read
from axiom_research.stock_fold_inputs import (project_saved_fold, read_parent, validate_raw, seal)
from axiom_research.stock_label_normalization import normalize_forward_labels
from axiom_research.feature_catalog import load_feature_catalog
from axiom_research.stock_ml import _iter_stock_feature_days, _prepare_stock_features
from test_stock_folds import fixture


def decision_fixture():
    import pandas as pd
    from axiom_data import DataBatch, QuerySpec
    days = [(date(2024,1,1)+timedelta(days=i)).isoformat() for i in range(23)]
    symbols = ['A','B','EX']; catalog = load_feature_catalog()
    selection = [v.to_dict() for v in catalog.default_selection]
    config = dict(snapshot='s-fixed',pit_policy='best_effort_vendor_v1',symbols=symbols,
        universe_id='synthetic',calendar=days,read_sessions=days,feature_sessions=days[20:],
        cutoff_by_session={d:d+'T20:30:00+08:00' for d in days},feature_selection=selection)
    query = QuerySpec('market_daily',('open','high','low','close','amount_cny'),tuple(symbols),
        tuple(days),config['pit_policy'],config['cutoff_by_session'])
    from dataclasses import replace
    factor = replace(query,domain='adjustment_factors',fields=('factor',))
    member = replace(query,domain='universe_membership',fields=('is_member',),universe_id='synthetic')
    native = {}
    for s,initial,slope,amount,step in [('A',100,1,1000,100),('B',60,2,700,30),('EX',40,3,500,20)]:
        for i,d in enumerate(days):
            close = float(initial+slope*i)
            native[s,d] = dict(open=close-slope,high=close+5,low=close-2,close=close,
                              amount_cny=float(amount+step*i),factor=1.)
    class Data:
        def batch(self,q):
            records = []; meta = {}
            for f in q.fields:
                meta[f] = dict(dtype='bool' if f=='is_member' else 'float64',
                    unit=None if f in ('is_member','factor') else 'CNY' if f=='amount_cny' else 'CNY/share',by_key=[])
            for d in q.sessions:
                for s in q.symbols:
                    row = dict(security_id=s,session=d)
                    for f in q.fields:
                        row[f] = s!='EX' if f=='is_member' else native[s,d][f]
                        meta[f]['by_key'].append(dict(security_id=s,session=d,usable_from=d+'T20:00:00+08:00',
                            first_observed_at=d+'T20:00:00+08:00',revision_id='r1',availability_basis='synthetic',
                            evidence_ref='fixture',missing_reason=None))
                    records.append(row)
            wire_query = dict(fields=list(q.fields),symbols=list(q.symbols),sessions=list(q.sessions),
                pit_policy=q.pit_policy,cutoff_by_session=dict(q.cutoff_by_session),purpose=q.purpose,
                policy_by_session=q.policy_by_session,price_basis=q.price_basis,
                adjustment_anchor=q.adjustment_anchor,universe_id=q.universe_id)
            return DataBatch(pd.DataFrame(records),meta,dict(contract_version='data_batch_v1',
                snapshot_id='s-fixed',domain=q.domain,reader_version='synthetic',query=wire_query,limitations=[]))
        def read(self,*,snapshot,query): return self.batch(query)
        def members(self,*,snapshot,query): return self.batch(query)
    prepared = dict(values=native,view_reference={'view_id':digest('synthetic-qlib')},
        price_query=query,factor_query=factor,member_query=member,seconds=0)
    return Data(),config,catalog,catalog.select(selection),prepared


class PreparationReuseTests(unittest.TestCase):
    def test_actual_core_proof_matches_new_index_source_closure(self):
        from axiom_research.stock_feature_inputs import _parents,DEFAULT_LIMITS
        data,config,catalog,chosen,prepared=decision_fixture()
        with patch('axiom_research.stock_ml._prepare_stock_qlib',return_value=prepared):
            feature,proof,_=_prepare_stock_features(data,config=config,destination='unused',catalog=catalog,chosen=chosen,progress=None)
        spec={'calendar':config['calendar'],'universe':config['symbols'],'snapshot':config['snapshot'],
            'pit_policy':config['pit_policy'],'ordered_features':feature['ordered_features'],
            'catalog_ref':catalog.identity,'feature_selection':config['feature_selection'],
            'read_sessions':config['read_sessions'],'feature_sessions':config['feature_sessions'],
            'cutoff_by_session':config['cutoff_by_session']}
        view={**feature['qlib_view'],'snapshot_id':config['snapshot'],'queries':[
            {'pit_policy':config['pit_policy'],'symbols':config['symbols']}],
            'universe_query':{'pit_policy':config['pit_policy'],'symbols':config['symbols']}}
        feature=seal({**{k:v for k,v in feature.items() if k!='feature_ref'},'qlib_view':view},'feature_ref')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);write_json(root/'feature.json',feature);write_json(root/'proof.json',proof)
            descriptor={'sessions':config['feature_sessions'],
                'features':{'path':str(root/'feature.json'),'file_digest':file_digest(root/'feature.json'),'feature_ref':feature['feature_ref']},
                'input_evidence':{'path':str(root/'proof.json'),'file_digest':file_digest(root/'proof.json'),'input_evidence_ref':feature['input_evidence_ref']}}
            covered,_=_parents([descriptor],spec,view,config['universe_id'],DEFAULT_LIMITS,complete=True)
        self.assertEqual(covered,config['feature_sessions'])

    def test_complete_day_wire_count_and_old_collector_identity(self):
        from axiom_data import DataBatch
        data,config,catalog,chosen,prepared = decision_fixture()
        original = DataBatch.to_json; counts = []
        def counted(batch):
            counts.append(batch.context['domain']); return original(batch)
        with patch.object(DataBatch,'to_json',counted):
            days = list(_iter_stock_feature_days(data,config=config,catalog=catalog,chosen=chosen,qlib_inputs=prepared))
        self.assertEqual(len(counts),6*len(config['feature_sessions']))
        # Independent financial signs for this two-member cross section; excluded
        # EX stays null. Core retains original member/clock and source identities.
        for rows,proof in days:
            a = next(r for r in rows if r['security_id']=='A')
            for got,want in zip(a['values'],[-1.,-1.,-1.,-1.,-1.,1.]): self.assertAlmostEqual(got,want,places=11)
            self.assertEqual(next(r for r in rows if r['security_id']=='EX')['values'],[None]*6)
            self.assertEqual(a['source_refs'],[proof['core_frame_ref'],digest(proof['core_plan'])])
        with patch('axiom_research.stock_ml._prepare_stock_qlib',return_value=prepared):
            old,proof,_ = _prepare_stock_features(data,config=config,destination='unused',catalog=catalog,chosen=chosen,progress=None)
        self.assertEqual(old['rows'],[row for rows,p in days for row in rows])
        self.assertEqual(proof,[p for rows,p in days])
        self.assertEqual(old['input_evidence_ref'],digest(proof))
        self.assertEqual(old['feature_ref'],digest({k:v for k,v in old.items() if k!='feature_ref'}))

    def test_shared_raw_admitted_once_and_normalized_parents_still_independent(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); manifest,spec=fixture(root)
            baseline=project_saved_fold(manifest,spec)
            raw=_read(root/'raw.json'); feature=_read(root/'features.json')
            original=manifest['training_labels'][0]; dates=original['sessions']; parents=[]
            for i,part in enumerate((dates[:31],dates[31:])):
                projected={k:v for k,v in raw.items() if k not in ('rows','label_ref')}
                projected.update(rows=[r for r in raw['rows'] if r['feature_session'] in part],
                    parent_label_ref=raw['label_ref'],feature_parent_ref=feature['feature_ref'],date_projection=part)
                norm=normalize_forward_labels(seal(projected,'label_ref'),features=feature,cutoff=spec['fit_cutoff'])
                path=root/f'normalized-{i}.json';write_json(path,norm)
                parents.append({**original,'sessions':part,'normalized':{'path':str(path),
                    'file_digest':file_digest(path),'label_ref':norm['label_ref']}})
            manifest['training_labels']=parents; reads=[]
            class CountedRows(list):
                scans=0
                def __iter__(self):
                    self.scans+=1
                    return super().__iter__()
            admitted=[]
            def reader(desc,key):
                reads.append(desc['path']); value=read_parent(desc,key)
                if desc['path']==str(root/'raw.json'):
                    value['rows']=CountedRows(value['rows']);admitted.append(value['rows'])
                return value
            with patch('axiom_research.stock_fold_inputs.validate_raw',wraps=validate_raw) as admit:
                values=project_saved_fold(manifest,spec,reader=reader)
            self.assertEqual(reads.count(str(root/'raw.json')),1)
            # Two original validator passes plus one admission bucket pass;
            # normalized parent count adds no whole-Raw row scan.
            self.assertEqual(admitted[0].scans,3)
            self.assertEqual([call.args[2] for call in admit.call_args_list],[spec['fit_cutoff'],spec['evaluation_cutoff']])
            self.assertEqual(values[2],baseline[2]);self.assertEqual(values[3],baseline[3])
            self.assertEqual(values[4],baseline[4])
            self.assertEqual(set(reads),{str(root/'raw.json'),str(root/'evaluation.json'),
                str(root/'normalized-0.json'),str(root/'normalized-1.json')})
            # Another call admits again, and a changed source cannot inherit the
            # first call's verification. Different fit cutoff also fails closed.
            changed=deepcopy(spec);changed['fit_cutoff']=changed['fit_session']+'T20:29:00+08:00'
            with self.assertRaises(ValueError): project_saved_fold(manifest,changed)
            (root/'raw.json').write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'file digest'): project_saved_fold(manifest,spec)


if __name__=='__main__': unittest.main()

"""Wire equality against public Data serialization and per-call clock checks."""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
import unittest
from unittest.mock import patch

import pandas as pd
from axiom_data import DataBatch, QueryError
import axiom_research.data_adapter as adapter
from axiom_research.stock_ml import _adjust_feature
from test_data_adapter import batch, REF, SESSIONS


def prices_factors():
    symbols = ['B','A']; dates = ['2024-01-02','2024-01-03','2024-01-04']
    fields = ['open','high','low','close','amount_cny']
    rows=[]; metas={name:[] for name in fields}; factor_rows=[]; factor_meta=[]
    for i,day in enumerate(dates):
        for j,symbol in enumerate(symbols):
            rows.append(dict(security_id=symbol,session=day,open=10+i+j,high=12+i+j,
                low=9+i+j,close=None if (i,j)==(0,1) else 11+i+j,
                amount_cny=None if (i,j)==(1,0) else 2**53+1+i+j))
            factor_rows.append(dict(security_id=symbol,session=day,factor=1.0+i/10))
            common=dict(security_id=symbol,session=day,usable_from=day+'T20:00:00+08:00',
                revision_id='r1',availability_basis='synthetic',first_observed_at=day+'T20:00:00+08:00')
            for name in fields:
                metas[name].append({**common,'missing_reason':None if rows[-1][name] is not None else 'source_missing'})
            factor_meta.append({**common,'missing_reason':None})
    query=dict(fields=fields,symbols=symbols,sessions=dates,pit_policy='best_effort_vendor_v1',
        cutoff_by_session={d:dates[-1]+'T20:30:00+08:00' for d in dates},
        purpose='decision_facts',price_basis='unadjusted',adjustment_anchor=None,universe_id=None,policy_by_session=None)
    context=dict(contract_version='data_batch_v1',snapshot_id='s_fixture',domain='market_daily',
        reader_version='fixture_v1',query=query,limitations=[],
        diagnostic_date=datetime(2024,1,4,12,30,tzinfo=timezone.utc))
    frame=pd.DataFrame.from_records(list(reversed(rows)))
    frame['amount_cny']=pd.array([r['amount_cny'] for r in reversed(rows)],dtype='Int64')
    metadata={name:dict(dtype='int64' if name=='amount_cny' else 'float64',unit='CNY',by_key=list(reversed(metas[name]))) for name in fields}
    price=DataBatch(frame,metadata,context)
    factor=DataBatch(pd.DataFrame.from_records(factor_rows),{'factor':dict(dtype='float64',unit='dimensionless',by_key=factor_meta)},
        {**context,'domain':'adjustment_factors','query':{**query,'fields':['factor']}})
    return price,factor,dates[-1]


class FeatureWireTests(unittest.TestCase):
    def test_join_wire_matches_public_serializer_types_missingness_and_order(self):
        price,factor,day=prices_factors();before=deepcopy((price.to_json(),factor.to_json()))
        actual,wire=_adjust_feature(price,factor,day,_with_wire=True)
        oracle=actual.to_json()
        encode=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
        self.assertEqual(encode(wire),encode(oracle))
        self.assertEqual(wire,_adjust_feature(price,factor,day).to_json())
        self.assertEqual((price.to_json(),factor.to_json()),before)
        amounts=[r['amount_cny'] for r in wire['records']]
        self.assertIn(2**53+1,amounts);self.assertIn(None,amounts)
        self.assertTrue(all(type(v) is int for v in amounts if v is not None))
        self.assertEqual([r['security_id'] for r in wire['records'][:2]],['B','A'])

    def test_wire_mutation_does_not_change_returned_batch_or_original_inputs(self):
        price,factor,day=prices_factors();original=deepcopy((price.to_json(),factor.to_json()))
        actual,wire=_adjust_feature(price,factor,day,_with_wire=True)
        before=actual.to_json()
        wire['context']['research_projection']['native_fields'].append('close')
        wire['context']['query']['cutoff_by_session'][day]='2024-01-05T20:30:00+08:00'
        wire['field_meta']['open']['by_key'][0]['price_provenance']['usable_from']='changed'
        wire['field_meta']['amount_cny']['by_key'][0]['missing_reason']='changed'
        wire['records'][0]['amount_cny']=0
        self.assertEqual(actual.to_json(),before)
        self.assertEqual((price.to_json(),factor.to_json()),original)

    def test_two_output_clocks_and_anchors_get_independent_wires(self):
        price,factor,day=prices_factors()
        _,first=_adjust_feature(price,factor,day,_with_wire=True)
        later=deepcopy(price.context['query']['cutoff_by_session'])
        later={d:'2024-01-05T20:30:00+08:00' for d in later}
        p=DataBatch(price.frame,price.field_meta,{**price.context,'query':{**price.context['query'],'cutoff_by_session':later}})
        f=DataBatch(factor.frame,factor.field_meta,{**factor.context,'query':{**factor.context['query'],'cutoff_by_session':later}})
        _,second=_adjust_feature(p,f,day,_with_wire=True)
        self.assertNotEqual(first['context']['research_projection'],second['context']['research_projection'])
        self.assertEqual(second,_adjust_feature(p,f,day).to_json())
        with self.assertRaises(QueryError):
            _adjust_feature(price,factor,'2024-01-05',_with_wire=True)

    def test_same_public_adjustment_errors_remain_before_wire_reuse(self):
        price,factor,day=prices_factors()
        changed=deepcopy(factor.field_meta);changed['factor']['by_key'][-1]['usable_from']='2024-01-04T21:00:00+08:00'
        future=DataBatch(factor.frame,changed,factor.context)
        for use_wire in (False,True):
            with self.assertRaisesRegex(QueryError,'later than the decision cutoff'):
                _adjust_feature(price,future,day,_with_wire=use_wire)


class AvailabilityClockTests(unittest.TestCase):
    def test_repeated_strings_parse_once_but_every_cutoff_is_checked(self):
        clock=adapter._AvailabilityClock();meta={'usable_from':'2024-01-02T10:00:00.123456Z'}
        with patch.object(adapter,'_utc',wraps=adapter._utc) as parse:
            self.assertEqual(adapter._availability(meta,'2024-01-02T10:00:01Z',clock=clock),'2024-01-02T10:00:01Z')
            self.assertEqual(adapter._availability(meta,'2024-01-02T11:00:00Z',clock=clock),'2024-01-02T10:00:01Z')
            with self.assertRaisesRegex(adapter.AdapterError,'exceeds'):
                adapter._availability(meta,'2024-01-02T10:00:00Z',clock=clock)
            self.assertEqual(parse.call_count,1)

    def test_cache_is_bounded_and_nonstring_inputs_remain_validated(self):
        clock=adapter._AvailabilityClock()
        base=datetime(2024,1,1,tzinfo=timezone.utc)
        for i in range(160):
            v=(base+timedelta(seconds=i)).isoformat()
            self.assertEqual(clock(v),adapter._utc(v,availability=True))
        self.assertEqual(len(clock.values),128)
        aware=base+timedelta(microseconds=1)
        self.assertEqual(clock(aware),adapter._utc(aware,availability=True))
        with self.assertRaises(adapter.AdapterError): clock(datetime(2024,1,1))
        with self.assertRaises(adapter.AdapterError): clock('bad-clock')

    def test_adapter_scopes_clock_reuse_to_one_call_and_keeps_outputs(self):
        price=batch('close',[10.,11.,12.]);member=batch('is_member',[True]*3,membership=True)
        with patch.object(adapter,'_utc',wraps=adapter._utc) as parse:
            first=adapter.adapt_decision_batch(price,reference=member,recipe_ref=REF,source_granularity='batch_field')
            self.assertEqual(sum(c.kwargs.get('availability') is True for c in parse.call_args_list),3)
            parse.reset_mock()
            second=adapter.adapt_decision_batch(price,reference=member,recipe_ref=REF,source_granularity='batch_field')
            self.assertEqual(sum(c.kwargs.get('availability') is True for c in parse.call_args_list),3)
        self.assertEqual(first.facts.to_dict(),second.facts.to_dict())
        self.assertEqual([r['availability'][0] for r in first.facts.to_dict()['rows']],
                         [day+'T10:00:00Z' for day in SESSIONS])
        self.assertEqual(first.context.to_dict(),second.context.to_dict())


if __name__=='__main__': unittest.main()

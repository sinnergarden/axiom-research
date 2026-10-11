"""Small synthetic checks of the fixed public Data adjustment allocation path."""
from copy import deepcopy
from datetime import date, timedelta
import unittest
from unittest.mock import patch

import pandas as pd

from axiom_data import DataBatch, adjust_prices
from axiom_research.stock_artifacts import digest
from axiom_research.stock_feature_memory import adjustment_peak_bytes
from axiom_research.stock_matrix_feature_producer import _owned_bytes
from axiom_research.stock_ml import _join_feature_wire, _feature_projection


OHLC = ('open', 'high', 'low', 'close')


def inputs(symbol_count=3, day_count=5, shared_proof=None):
    symbols = ['S'+str(i) for i in range(symbol_count)]
    sessions = [(date(2026, 1, 2)+timedelta(days=i)).isoformat() for i in range(day_count)]
    cutoff = sessions[-1]+'T20:30:00+08:00'
    price_rows, factor_rows = [], []
    metas = {f: [] for f in (*OHLC, 'amount_cny', 'factor')}
    for i, session in enumerate(sessions):
        for j, symbol in enumerate(symbols):
            key = {'security_id': symbol, 'session': session}
            opening = 10.0+i+j
            price_rows.append({**key, 'open': opening, 'high': opening+1,
                               'low': opening-1, 'close': opening+.5, 'amount_cny': 1000.0+i})
            factor_rows.append({**key, 'factor': 1.0+i*.25})
            for field in metas:
                metas[field].append({**key, 'revision_id': field+'-'+symbol+'-'+session,
                    'revision_sequence': 1, 'raw_batch_id': 'synthetic',
                    'usable_from': session+'T17:00:00+08:00', 'first_observed_at': session+'T09:00:00Z',
                    'availability_basis': 'operational', 'evidence_ref': None, 'missing_reason': None})

    def context(domain, fields):
        result = {'contract_version': 'data_batch_v1', 'reader_version': 'synthetic',
            'snapshot_id': 'synthetic-fixed', 'domain': domain,
            'contract_id': domain+'_v1', 'source_profile_id': 'synthetic', 'limitations': [],
            'query': {'domain': domain, 'symbols': symbols, 'sessions': sessions,
                'fields': list(fields), 'cutoff_by_session': {s: cutoff for s in sessions},
                'pit_policy': 'operational_pit_v1', 'purpose': 'feature',
                'price_basis': 'unadjusted', 'adjustment_anchor': None}}
        if shared_proof is not None:
            result['coverage'] = shared_proof
        return result

    price_meta = {f: {'dtype': 'float64', 'unit': 'CNY' if f == 'amount_cny' else 'CNY/share',
                     'by_key': metas[f]} for f in (*OHLC, 'amount_cny')}
    factor_meta = {'factor': {'dtype': 'float64', 'unit': 'dimensionless', 'by_key': metas['factor']}}
    prices = DataBatch(pd.DataFrame(price_rows), price_meta, context('market_daily', (*OHLC, 'amount_cny')))
    factors = DataBatch(pd.DataFrame(factor_rows), factor_meta, context('adjustment_factors', ('factor',)))
    native = {'records': deepcopy(price_rows), 'field_meta': prices.field_meta, 'context': prices.context}
    return prices, factors, native, sessions[-1]


class StockFeatureMemoryTests(unittest.TestCase):
    def test_public_adjust_serialize_join_graphs_fit_phase_charges(self):
        for securities, days in ((1, 1), (3, 5), (8, 10)):
            with self.subTest(securities=securities, days=days):
                prices, factors, native, anchor = inputs(securities, days)
                before = digest(native)
                charge = adjustment_peak_bytes(prices, factors, native)
                roots = [prices.field_meta, prices.context, factors.field_meta, factors.context, native]
                adjusted = adjust_prices(prices, factors, fields=OHLC, factor_field='factor',
                                         anchor_session=anchor, decision_session=anchor)
                wire = adjusted.to_json()
                joined = _join_feature_wire(wire, native, _feature_projection(wire, native))
                for stage, outputs in (
                    ('adjustment_bytes', [adjusted.field_meta, adjusted.context]),
                    ('serialization_bytes', [adjusted.field_meta, adjusted.context, wire]),
                    ('join_bytes', [wire, joined])):
                    seen = set()
                    _owned_bytes(roots, seen)
                    actual_additional = _owned_bytes(outputs, seen)
                    self.assertLessEqual(actual_additional, charge['stages'][stage])
                self.assertEqual(charge['peak_bytes'], charge['owned_bytes']+charge['temporary_bytes'])
                self.assertGreaterEqual(charge['charges']['input_frame_bytes'],
                    sum(int(b.frame.memory_usage(deep=True).sum()) for b in (prices, factors)))
                # Public adjustment remains the same arithmetic and source lineage.
                first = wire['records'][0]
                self.assertEqual(first['open'], 10.0/(1.0+(days-1)*.25))
                self.assertEqual(wire['field_meta']['open']['by_key'][0]['price_provenance'],
                                 prices.field_meta['open']['by_key'][0])
                self.assertEqual(joined['records'][0]['amount_cny'], 1000.0)
                self.assertEqual(digest(native), before)

    def test_shared_source_proof_is_charged_once_and_scalar_size_not_multiplied(self):
        proof = {'proof': 'public-synthetic-proof-'*10000, 'parts': ['fixed']}
        prices, factors, native, _ = inputs(shared_proof=proof)
        with patch('axiom_data.adjust_prices', side_effect=AssertionError('measurement executed Data')):
            charge = adjustment_peak_bytes(prices, factors, native)
        roots = [prices.field_meta, prices.context, factors.field_meta, factors.context, native]
        self.assertEqual(charge['charges']['shared_input_graph_bytes'], _owned_bytes(roots))
        self.assertLess(charge['charges']['shared_input_graph_bytes'], sum(_owned_bytes(r) for r in roots))
        plain = adjustment_peak_bytes(*inputs()[:3])
        # Long immutable evidence strings increase existing ownership, while
        # detached adjustment/serialization copies add their containers only.
        self.assertGreater(charge['owned_bytes']-plain['owned_bytes'], len(proof['proof']))
        self.assertLess(charge['stages']['adjustment_bytes']-plain['stages']['adjustment_bytes'], 4096)

    def test_incomplete_or_nested_raw_provenance_is_not_this_stage_shape(self):
        prices, factors, native, _ = inputs()
        prices.field_meta['open']['by_key'].pop()
        with self.assertRaisesRegex(ValueError, 'complete Reader provenance'):
            adjustment_peak_bytes(prices, factors, native)
        prices, factors, native, _ = inputs()
        factors.field_meta['factor']['by_key'][0]['nested'] = {'source': 'unexpected'}
        with self.assertRaisesRegex(ValueError, 'scalar Reader provenance'):
            adjustment_peak_bytes(prices, factors, native)


if __name__ == '__main__':
    unittest.main()

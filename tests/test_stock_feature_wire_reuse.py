"""Small same-call wire/leaf/identity checks; synthetic inputs only."""
from copy import deepcopy
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from axiom_data import DataBatch
from axiom_engine.core import FeatureFrame, FeaturePlan
from axiom_research import data_adapter as adapter
from axiom_research.stock_artifacts import digest
from axiom_research.stock_matrix_feature_producer import FIELDS, _NativeWindow, iter_matrix_feature_days
from axiom_research.stock_ml import _adjust_feature, _adjust_feature_wire, _feature_projection, _join_feature_wire
from test_data_adapter import REF, batch as adapter_batch
from test_stock_matrix_feature_batch import arguments, prepared_inputs
from test_stock_matrix_feature_producer import MemoryFeatureInputs


LIMIT = 32*1024**2
DAYS = ['2024-01-02', '2024-01-03']


def inputs():
    keys = [('A', day) for day in DAYS]
    records = [{'security_id': key[0], 'session': key[1], 'open': -0.0,
        'high': 100.000001+i, 'low': None if i else 0.125,
        'close': 99.1234567+i, 'amount_cny': 16777217+i} for i, key in enumerate(keys)]
    frame = pd.DataFrame(records)
    frame['amount_cny'] = pd.array([row['amount_cny'] for row in records], dtype='Int64')
    query = {'symbols': ['A'], 'sessions': DAYS, 'fields': list(FIELDS[:-1]),
        'pit_policy': 'operational_pit_v1', 'purpose': 'decision_facts',
        'price_basis': 'unadjusted', 'adjustment_anchor': None,
        'cutoff_by_session': {day: DAYS[-1]+'T20:00:00Z' for day in DAYS}}
    context = {'contract_version': 'data_batch_v1', 'snapshot_id': 's-wire-fixture',
        'reader_version': 'fixture/1', 'domain': 'market_daily', 'query': query}
    def meta(field, values):
        return {'dtype': 'int64' if field=='amount_cny' else 'float64',
            'unit': 'CNY' if field=='amount_cny' else 'dimensionless' if field=='factor' else 'CNY/share',
            'by_key': [{'security_id': key[0], 'session': key[1], 'revision_id': 'r-'+key[1],
                'usable_from': key[1]+'T08:00:00.123456Z', 'first_observed_at': key[1]+'T08:00:00Z',
                'availability_basis': 'source_receipt', 'missing_reason': 'source_missing' if value is None else None}
                for key, value in zip(keys, values)]}
    prices = DataBatch(frame, {field: meta(field, [row[field] for row in records]) for field in FIELDS[:-1]}, context)
    factors = DataBatch(pd.DataFrame([{'security_id': key[0], 'session': key[1], 'factor': value}
        for key, value in zip(keys, [1.1, 2.375])]), {'factor': meta('factor', [1.1, 2.375])},
        {**context, 'domain': 'adjustment_factors', 'query': {**query, 'fields': ['factor']}})
    values = [[float(np.float32(row[field])) if row[field] is not None else np.nan
               for field in FIELDS[:-1]]+[float(np.float32(factors.frame.iloc[i]['factor']))]
              for i, row in enumerate(records)]
    values[0][FIELDS.index('close')] = 0.0  # Actual Reader fallback, not frozen native value.
    native = pd.DataFrame(values[::-1], columns=['$'+field for field in FIELDS],
        index=pd.MultiIndex.from_tuples([('AA', pd.Timestamp(day)) for day in DAYS[::-1]]), dtype=np.float32)
    return prices, factors, _NativeWindow(native, sessions=DAYS, symbols=['A'], instrument_map={'A': 'AA'})


def project(window, value, *, wire=False):
    stats = {key: 0 for key in ('native_value_reuses', 'reader_projection_fallback_cells',
                               'reader_projection_fallback_null_cells')}
    return window.project(value, view_ref='fixed-native-view', stats=stats,
                          maximum_resident_bytes=LIMIT, _with_wire=wire)


class FeatureWireReuseTests(unittest.TestCase):
    def test_projected_wire_is_exact_current_frame_with_distinct_raw_reference(self):
        source, _, window = inputs(); before = source.to_json()
        projected, wire = project(window, source, wire=True)
        self.assertEqual(wire, projected.to_json())
        self.assertEqual(wire['context']['numeric_projection']['reader_batch_ref'], digest(before))
        self.assertNotEqual(digest(wire), digest(before))
        self.assertEqual(wire['records'][0]['close'], float(np.float32(before['records'][0]['close'])))
        self.assertEqual(type(wire['records'][0]['amount_cny']), int)
        self.assertEqual(wire['records'][0]['amount_cny'], 16777216)
        self.assertIsNone(wire['records'][1]['low'])
        self.assertEqual(struct.pack('>d', wire['records'][0]['open']), struct.pack('>d', -0.0))
        self.assertEqual(source.to_json(), before)

    def test_original_mutation_cannot_change_private_frame_metadata_or_wire(self):
        source, _, window = inputs(); projected, wire = project(window, source, wire=True)
        expected = deepcopy(wire)
        source.frame.loc[0, 'close'] = 1e9
        source.field_meta['close']['by_key'][0]['usable_from'] = '2099-01-01T00:00:00Z'
        source.context['query']['cutoff_by_session'][DAYS[0]] = '2099-01-01T00:00:00Z'
        self.assertEqual(wire, expected)
        self.assertEqual(projected.to_json(), expected)

    def test_internal_adjustment_preserves_binary64_sources_and_avoids_joined_frame(self):
        source, source_factors, window = inputs()
        prices, wire = project(window, source, wire=True); factors = project(window, source_factors)
        public_batch, expected = _adjust_feature(prices, factors, DAYS[-1], _with_wire=True)
        original = DataBatch.to_json; serializations = []
        def serialize(value):
            self.assertIsNot(value, prices); self.assertIsNot(value, factors)
            serializations.append(value)
            return original(value)
        before = deepcopy(wire)
        with patch.object(DataBatch, 'to_json', new=serialize), \
             patch.object(pd.DataFrame, 'merge', side_effect=AssertionError('joined DataFrame constructed')):
            actual = _adjust_feature_wire(prices, factors, DAYS[-1], wire)
        self.assertEqual(len(serializations), 1)
        self.assertEqual(actual, expected); self.assertEqual(digest(actual), digest(expected))
        self.assertEqual(actual, public_batch.to_json()); self.assertEqual(wire, before)
        self.assertEqual(actual['context']['research_projection']['native_ref'], digest(wire))
        for left, right in zip(actual['records'], expected['records']):
            for field in FIELDS[:-1]:
                if left[field] is not None:
                    self.assertEqual(struct.pack('>d', float(left[field])), struct.pack('>d', float(right[field])))

    def test_wire_join_rejects_duplicate_missing_keys_and_amount_before_return(self):
        source, source_factors, window = inputs()
        prices, native = project(window, source, wire=True); factors = project(window, source_factors)
        _, joined = _adjust_feature(prices, factors, DAYS[-1], _with_wire=True)
        adjusted = deepcopy(joined)
        adjusted['records'] = [{k: v for k, v in row.items() if k!='amount_cny'} for row in adjusted['records']]
        cases = ('native_duplicate', 'adjusted_duplicate', 'missing_key', 'missing_amount', 'missing_metadata')
        for kind in cases:
            a, n = deepcopy(adjusted), deepcopy(native)
            if kind=='native_duplicate': n['records'].append(deepcopy(n['records'][0]))
            elif kind=='adjusted_duplicate': a['records'].append(deepcopy(a['records'][0]))
            elif kind=='missing_key': n['records'].pop()
            elif kind=='missing_amount': n['records'][0].pop('amount_cny')
            else: n['field_meta'].pop('amount_cny')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                _join_feature_wire(a, n, _feature_projection(a, n))

    def test_public_adjustment_error_precedes_internal_join_validation(self):
        source, source_factors, window = inputs()
        prices, _ = project(window, source, wire=True); factors = project(window, source_factors)
        factors.context['snapshot_id'] = 'different-snapshot'
        errors = []
        for call in (lambda: _adjust_feature(prices, factors, DAYS[-1], _with_wire=True),
                     lambda: _adjust_feature_wire(prices, factors, DAYS[-1], {})):
            try: call()
            except Exception as exc: errors.append((type(exc), str(exc)))
        self.assertEqual(len(errors), 2); self.assertEqual(errors[0], errors[1])
        self.assertIn('Snapshot IDs must match', errors[0][1])

    def test_public_joined_wire_mutation_does_not_change_returned_batch(self):
        source, source_factors, window = inputs()
        prices, _ = project(window, source, wire=True); factors = project(window, source_factors)
        result, wire = _adjust_feature(prices, factors, DAYS[-1], _with_wire=True)
        before = result.to_json()
        wire['context']['research_projection']['native_fields'].append('close')
        wire['context']['query']['fields'].append('factor')
        wire['field_meta']['amount_cny']['by_key'][0]['revision_id'] = 'caller mutation'
        self.assertEqual(result.to_json(), before)

    def test_adapter_preserves_exception_order_and_checks_each_cell_again(self):
        cases = [('unit', 'unit unknown'), ('member', 'unknown membership'),
                 ('reason', 'present fact has missing reason'), ('derived', 'invalid derived provenance'),
                 ('clock', 'exceeds its decision cutoff')]
        for mode in ('cell', 'batch_field'):
            for kind, reason in cases:
                price = adapter_batch('close', [10.0, 11.0, 12.0]).wire
                member = adapter_batch('is_member', [True]*3, membership=True).wire
                meta = price['field_meta']['close']['by_key'][-1]
                meta['usable_from'] = '2099-01-01T00:00:00Z'
                if kind in ('unit', 'member'): member['records'][0]['is_member'] = None
                if kind=='unit': price['field_meta']['close']['unit'] = None
                if kind in ('reason', 'derived'): meta['factor_provenance'] = 'bad'
                if kind=='reason': meta['missing_reason'] = 'source_missing'
                with self.subTest(mode=mode, kind=kind), self.assertRaisesRegex(ValueError, reason):
                    adapter._adapt_decision_wires(price, member, recipe_ref=REF, source_granularity=mode)

    def test_one_day_full_frame_proof_and_call_counts_match_fixed_baseline(self):
        calls = {'to_json': 0, 'leaves_roots': 0, 'leaves_all': 0, 'identities': 0}; depth = 0
        original_json, original_leaves = DataBatch.to_json, adapter._leaves
        frame_identity, plan_identity = FeatureFrame.identity.fget, FeaturePlan.identity.fget
        def serialize(value):
            calls['to_json'] += 1
            return original_json(value)
        def leaves(meta):
            nonlocal depth
            calls['leaves_all'] += 1
            if depth==0: calls['leaves_roots'] += 1
            depth += 1
            try: return original_leaves(meta)
            finally: depth -= 1
        def identity(getter, value):
            if sys._getframe(2).f_code.co_name=='_iter_core_feature_batch': calls['identities'] += 1
            return getter(value)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture = MemoryFeatureInputs(root); prepared = prepared_inputs(fixture, root)
            config = {**fixture.config, 'feature_sessions': fixture.outputs[:1]}
            with patch.object(DataBatch, 'to_json', new=serialize), patch.object(adapter, '_leaves', new=leaves), \
                 patch.object(FeatureFrame, 'identity', property(lambda value: identity(frame_identity, value))), \
                 patch.object(FeaturePlan, 'identity', property(lambda value: identity(plan_identity, value))), \
                 patch('axiom_research.qlib_adapter.QlibView.read', lambda view, **kwargs: fixture.native_read(view, **kwargs)):
                output = list(iter_matrix_feature_days(fixture.data,
                    **{**arguments(fixture, prepared, reuse_budget_bytes=0), 'config': config}))
            self.assertEqual(digest(output), 'sha256:210ee8dfa6107dfcbdedeca4a2924471411206d815bd400ce119271b7095cad9')
            self.assertEqual(calls, {'to_json': 4, 'leaves_roots': 378, 'leaves_all': 1134, 'identities': 2})

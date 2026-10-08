"""Actual tiny Data selections across the Research wire and saved Raw boundary."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from axiom_data import QuerySpec
from column_source_fixture import prepared, LIMITS
from axiom_research.stock_artifacts import digest
from axiom_research.stock_column_inputs import (
    ColumnPriceDomain, query_binding, validate_column_raw_binding)
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_target_spec import resolve_stock_label_spec
from test_stock_target_config_blocks import label


def plain(value):
    if isinstance(value, Mapping): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


def source_spec(snapshot):
    return {'snapshot': snapshot, 'universe': ['A', 'B', 'C'],
        'calendar': ['2024-01-02', '2024-01-03', '2024-01-05'],
        'feature_sessions': ['2024-01-02'], 'pit_policy': 'operational_pit_v1',
        'target_spec': resolve_stock_label_spec(label(1))}


def query(cutoff):
    return QuerySpec(domain='market_daily', fields=('open', 'close'),
        symbols=('A', 'B', 'C'), sessions=('2024-01-03',),
        pit_policy='operational_pit_v1', cutoff_by_session={'2024-01-03': cutoff},
        purpose='label_outcomes', price_basis='unadjusted')


def raw_definition(spec, source, cutoff):
    return {**{k: spec[k] for k in ('snapshot', 'universe', 'calendar')},
        'sessions': spec['feature_sessions'], 'price_view': source, 'cutoff': cutoff,
        'formula': 'close(f+1) / open(f+1) - 1', 'horizon_sessions': 1,
        'start_session_offset': 1, 'end_session_offset': 1,
        'label_definition_ref': spec['target_spec']['label_definition_ref'],
        'price_basis': 'common_anchor_adjusted_v1',
        'missing_policy': 'invalid_null_preserve_grid'}


class PublicDataColumnContractTests(unittest.TestCase):
    def test_timezone_precision_and_optional_fields_match_actual_Data(self):
        clocks = ['2024-01-03T20:00:00+08:00', '2024-01-03T12:00:00Z',
            '2024-01-03T12:00:00.000000+00:00',
            '2024-01-03T20:00:00.123456+08:00',
            datetime.fromisoformat('2024-01-03T17:30:00.123456+05:30')]
        with tempfile.TemporaryDirectory() as temp:
            data, snapshot = prepared(Path(temp))
            spec = source_spec(snapshot)
            with data.open_column_source(snapshot=snapshot, limits=LIMITS) as owner:
                for clock in clocks:
                    with self.subTest(clock=clock):
                        q = query(clock); fq = replace(q, domain='adjustment_factors', fields=('factor',))
                        price = owner.select(query=q); factor = owner.select(query=fq)
                        adjusted = owner.adjust(price, factor, fields=q.fields,
                            anchor_session='2024-01-03', decision_session='2024-01-03')
                        aq = replace(q, price_basis='common_anchor_adjusted_v1', adjustment_anchor='2024-01-03')
                        for selection, expected in ((price, q), (factor, fq), (adjusted, aq)):
                            self.assertEqual(plain(selection.query_binding), query_binding(expected))
                            self.assertIsNone(selection.query_binding['universe_id'])
                            self.assertIsNone(selection.query_binding['policy_by_session'])
                        with ColumnPriceDomain(adjusted, spec=spec, query=aq,
                            anchor='2024-01-03', input_queries={
                                'price': price.query_binding, 'factor': factor.query_binding}) as domain:
                            validate_column_raw_binding(raw_definition(spec, domain.source, clock), spec, clock)
                            self.assertEqual(domain.source['query_binding'], plain(adjusted.query_binding))
                            self.assertEqual(domain.source['revision_binding'], plain(adjusted.revision_binding))
                            self.assertEqual(domain.source['evidence_binding'], plain(adjusted.evidence_binding))
                            self.assertEqual(domain.source['source_binding'], plain(adjusted.source_binding))
                            self.assertEqual(domain.source['adjustment_binding'], plain(adjusted.derivation))
                            for name in q.fields:
                                self.assertEqual(domain.source['column_metadata'][name], plain(adjusted.columns[name].metadata))
                        adjusted.close(); factor.close(); price.close()
            for bad in ('2024-01-03T20:00:00', datetime(2024, 1, 3, 20)):
                with self.subTest(naive_clock=bad), self.assertRaises(ValueError): query_binding(query(bad))
            # These optional values are not silently dropped or coalesced.
            self.assertNotEqual(query_binding(query(clocks[0])), query_binding(replace(query(clocks[0]), policy_by_session={})))

    def test_live_query_fields_and_saved_proof_mutations_still_reject(self):
        cutoff = '2024-01-03T20:00:00.123456+08:00'
        with tempfile.TemporaryDirectory() as temp:
            data, snapshot = prepared(Path(temp)); spec = source_spec(snapshot)
            with data.open_column_source(snapshot=snapshot, limits=LIMITS) as owner:
                q = query(cutoff); fq = replace(q, domain='adjustment_factors', fields=('factor',))
                price = owner.select(query=q); factor = owner.select(query=fq)
                adjusted = owner.adjust(price, factor, fields=q.fields,
                    anchor_session='2024-01-03', decision_session='2024-01-03')
                aq = replace(q, price_basis='common_anchor_adjusted_v1', adjustment_anchor='2024-01-03')
                mutations = {'domain': 'adjustment_factors', 'fields': ['close', 'open'],
                    'symbols': ['B', 'A', 'C'], 'sessions': ['2024-01-02'],
                    'pit_policy': 'best_effort_vendor_v1', 'purpose': 'decision_facts',
                    'price_basis': 'unadjusted', 'adjustment_anchor': '2024-01-02',
                    'universe_id': 'changed-universe', 'policy_by_session': {},
                    'cutoff_by_session': {'2024-01-03': '2024-01-03T20:00:00.123457+08:00'},
                    'unexpected_field': None}
                for field, value in mutations.items():
                    with self.subTest(live_query_field=field):
                        altered = plain(adjusted.query_binding); altered[field] = value
                        proxy = SimpleNamespace(contract_version=adjusted.contract_version,
                            snapshot_ref=adjusted.snapshot_ref, axes=adjusted.axes, query_binding=altered)
                        with self.assertRaisesRegex(ValueError, 'exact requested QuerySpec'):
                            ColumnPriceDomain(proxy, spec=spec, query=aq, anchor='2024-01-03',
                                input_queries={'price': price.query_binding, 'factor': factor.query_binding})
                with ColumnPriceDomain(adjusted, spec=spec, query=aq, anchor='2024-01-03',
                    input_queries={'price': price.query_binding, 'factor': factor.query_binding}) as domain:
                    original = domain.source
                    cases = [
                        ('input_queries', 'price', 'universe_id', 'changed-universe'),
                        ('input_queries', 'price', 'policy_by_session', {}),
                        ('input_queries', 'factor', 'fields', ['open']),
                        ('input_queries', 'factor', 'price_basis', 'common_anchor_adjusted_v1'),
                        ('query_binding', 'cutoff_by_session', '2024-01-03', '2024-01-03T20:00:00.123457+08:00'),
                        ('adjustment_binding', 'decision_cutoff', '2024-01-03T20:00:00.123457+08:00'),
                        ('adjustment_binding', 'anchor_session', '2024-01-02'),
                        ('adjustment_binding', 'price_selection_ref', 'invalid'),
                        ('source_binding', 'snapshot_ref', 'changed-snapshot'),
                        ('source_binding', 'domain', 'adjustment_factors'),
                        ('source_binding', 'source_block_refs', [digest('changed-block')]),
                        ('revision_binding', 'indices', 'index:price', 'invalid'),
                        ('revision_binding', 'source_block_refs', [digest('changed-block')]),
                        ('evidence_binding', 'selected_versions', {}),
                        ('evidence_binding', 'source_block_refs', ['invalid']),
                        ('column_metadata', 'open', 'unit', 'changed-unit'),
                        ('column_metadata', 'close', 'basis', 'unadjusted'),
                        ('column_metadata', 'close', 'recipe_version', 'changed-recipe'),
                    ]
                    for *keys, value in cases:
                        with self.subTest(saved_binding_path=keys):
                            source = deepcopy(original); leaf = source
                            for key in keys[:-1]: leaf = leaf[key]
                            leaf[keys[-1]] = value
                            source.pop('price_view_ref'); source = seal(source, 'price_view_ref')
                            with self.assertRaises(ValueError):
                                validate_column_raw_binding(raw_definition(spec, source, cutoff), spec, cutoff)
                    source = deepcopy(original)
                    for wire in (source['query_binding'], *source['input_queries'].values()): wire['unexpected_field'] = None
                    source.pop('price_view_ref'); source = seal(source, 'price_view_ref')
                    with self.assertRaisesRegex(ValueError, 'exact column QuerySpec'):
                        validate_column_raw_binding(raw_definition(spec, source, cutoff), spec, cutoff)
                adjusted.close(); factor.close(); price.close()


if __name__ == '__main__': unittest.main()

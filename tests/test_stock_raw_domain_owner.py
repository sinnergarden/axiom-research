"""Small synthetic domain ownership; no supplier, Feature or model execution."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from axiom_research import labels
from axiom_research.stock_artifacts import digest
from axiom_research import stock_compact_labels as owner
from axiom_research.stock_compact_batch import read_target
from axiom_research.stock_compact_store import OwnedStore, limits
from test_stock_labels import CALENDAR, CUTOFF
from test_stock_matrix_prepare import Batch, PublicDataFixture, Query


class RawDomainOwnerTests(unittest.TestCase):
    @contextmanager
    def environment(self, transform=None):
        spec = {'snapshot': 's_raw_domain_fixture', 'pit_policy': 'market_pit_safe_v1',
                'calendar': list(CALENDAR), 'universe': ['A', 'B']}
        class Data(PublicDataFixture):
            def adjust(self, *args, **kwargs):
                value = super().adjust(*args, **kwargs)
                if transform is not None: transform(value.wire)
                self.adjusted[-1] = deepcopy(value.wire)
                return value
        data = Data(spec)
        module = ModuleType('axiom_data'); module.QuerySpec = Query; module.adjust_prices = data.adjust
        budgets = limits({'maximum_matrix_bytes': 4 * 1024**2})
        with OwnedStore(budgets) as store, patch.dict(sys.modules, {'axiom_data': module}):
            stats = {'_limits': budgets, '_store': store,
                     '_feature_store': SimpleNamespace(resident_bytes=0, metrics={'source_bytes': 0}),
                     '_feature_bytes': 0, '_caller_bytes': 0, '_caller_source_bytes': 0,
                     '_retained_raw_bytes': 0, '_price_view_bytes': 0, 'maximum_working_bytes': 0,
                     'data_read_calls': 0, 'price_domains': [], 'raw_operator_calls': 0, 'raw_cache_hits': 0}
            yield spec, data, stats

    def expected(self, wire, days, source_ref):
        return list(labels._forward_rows(wire['records'], wire['field_meta'], wire['context'],
            calendar=CALENDAR, features=days, horizon_sessions=5, source_ref=source_ref))

    def test_multiple_children_compile_one_domain_and_keep_exact_rows_and_source(self):
        with self.environment() as (spec, data, stats), tempfile.TemporaryDirectory() as temp:
            with owner._price_view(data, spec, CUTOFF, list(CALENDAR), stats) as domain:
                wire = data.adjusted[-1]; source = owner._source_view(**{
                    'records': wire['records'], 'meta': wire['field_meta'], 'context': wire['context']})
                self.assertEqual(domain.source, source)
                expected = self.expected(wire, list(CALENDAR), source['price_view_ref'])
                chunks = [list(CALENDAR[:2]), list(CALENDAR[2:4]), list(CALENDAR[4:])]
                implementation = owner._raw_implementation()  # Pin actual source before counting wrappers.
                with patch.object(owner, '_raw_implementation', return_value=implementation), \
                     patch.object(labels, '_compile_forward_index', wraps=labels._compile_forward_index) as compile_index, \
                     patch.object(labels, '_keyed', wraps=labels._keyed) as keyed, \
                     patch.object(labels, '_query_context', wraps=labels._query_context) as context:
                    actual = []
                    for days in chunks:
                        descriptor, rows = owner._raw(spec, CUTOFF, days, Path(temp), stats, domain)
                        actual.extend(rows)
                        saved, loaded = read_target(stats['_store'], descriptor)
                        self.assertEqual(saved['definition']['price_view'], source)
                        self.assertEqual(list(loaded), rows)
                        stats['_store'].release_payloads()
                    self.assertEqual(compile_index.call_count, 1)
                    self.assertEqual(keyed.call_count, 3)
                    self.assertEqual(context.call_count, 0)  # Once at domain creation, already compiled.
                self.assertEqual(actual, expected)
                self.assertEqual(stats['price_domain_index_builds'], 1)
                self.assertEqual(stats['price_domain_indexed_rows'], 3 * len(wire['records']))
                self.assertGreater(stats['price_domain_compile_peak_bytes'], 0)
                self.assertGreater(stats['_price_view_bytes'], 0)
                print('RAW_DOMAIN_METRICS ' + str({k: stats[k] for k in (
                    'data_read_calls', 'price_domain_index_builds', 'price_domain_indexed_rows',
                    'raw_operator_calls', 'price_domain_compile_peak_bytes', 'maximum_working_bytes')}))
            self.assertEqual(stats['_price_view_bytes'], 0)
            self.assertEqual(stats['price_domain_release_calls'], 1)

    def test_exact_Raw_HIT_does_not_compile_or_run_the_operator(self):
        with self.environment() as (spec, data, stats), tempfile.TemporaryDirectory() as temp:
            days = list(CALENDAR[:2])
            with owner._price_view(data, spec, CUTOFF, days, stats) as first:
                original, rows = owner._raw(spec, CUTOFF, days, Path(temp), stats, first)
            builds = stats['price_domain_index_builds']
            implementation = owner._raw_implementation()
            with owner._price_view(data, spec, CUTOFF, days, stats) as second, \
                 patch.object(owner, '_raw_implementation', return_value=implementation), \
                 patch.object(labels, '_compile_forward_index', side_effect=AssertionError('HIT compiled')):
                cached, again = owner._raw(spec, CUTOFF, days, Path(temp), stats, second)
                self.assertEqual((original, rows), (cached, again))
            self.assertEqual(stats['price_domain_index_builds'], builds)
            self.assertEqual(stats['raw_operator_calls'], 1)
            self.assertEqual(stats['raw_cache_hits'], 1)

    def test_native_and_returned_control_mutation_do_not_change_owned_sources(self):
        with self.environment() as (spec, data, stats):
            with owner._price_view(data, spec, CUTOFF, list(CALENDAR), stats) as domain:
                before = domain.source; wire = deepcopy(data.adjusted[-1])
                expected = self.expected(wire, list(CALENDAR), before['price_view_ref'])
                data.adjusted[-1]['records'][0]['open'] = 9999.
                data.adjusted[-1]['field_meta']['close']['by_key'][0]['factor_provenance']['usable_from'] = None
                exposed = domain.source; exposed['context']['query']['symbols'].clear()
                self.assertEqual(domain.source, before)
                actual = list(labels._forward_rows(None, None, None, calendar=CALENDAR,
                    features=list(CALENDAR), horizon_sessions=5, source_ref=domain.source_ref, _domain=domain))
                actual[0]['source_refs'].append(digest('caller change'))
                repeated = list(labels._forward_rows(None, None, None, calendar=CALENDAR,
                    features=list(CALENDAR), horizon_sessions=5, source_ref=domain.source_ref, _domain=domain))
                self.assertEqual(repeated, expected)

    def test_cutoff_snapshot_source_selector_and_process_cannot_cross_domains(self):
        with self.environment() as (spec, data, stats), tempfile.TemporaryDirectory() as temp:
            days = [CALENDAR[0]]
            with owner._price_view(data, spec, CUTOFF, days, stats) as domain:
                for changed_spec, cutoff, requested in (
                    ({**spec, 'snapshot': 's_other'}, CUTOFF, days),
                    (spec, '2024-03-11T18:00:00Z', days),
                    (spec, CUTOFF, [CALENDAR[1]]),
                ):
                    with self.assertRaisesRegex(ValueError, 'request/cutoff'):
                        owner._raw(changed_spec, cutoff, requested, Path(temp), stats, domain)
                with self.assertRaisesRegex(ValueError, 'selector/source'):
                    list(labels._forward_rows(None, None, None, calendar=CALENDAR, features=days,
                        horizon_sessions=5, source_ref=digest('different revision'), _domain=domain))
                pid = owner.os.getpid()
                with patch.object(owner.os, 'getpid', return_value=pid + 1):
                    with self.assertRaisesRegex(ValueError, 'another process'): domain.source
                self.assertNotIn('price_domain_index_builds', stats)

    def test_borrowed_generator_blocks_close_and_explicit_close_releases_it(self):
        with self.environment() as (spec, data, stats):
            domain = owner._price_view(data, spec, CUTOFF, list(CALENDAR), stats)
            iterator = labels._forward_rows(None, None, None, calendar=CALENDAR,
                features=list(CALENDAR), horizon_sessions=5, source_ref=domain.source_ref, _domain=domain)
            next(iterator)
            with self.assertRaisesRegex(ValueError, 'still borrowed'): domain.close()
            self.assertGreater(stats['_price_view_bytes'], 0)
            iterator.close(); domain.close()
            self.assertEqual(stats['_price_view_bytes'], 0)
            with self.assertRaisesRegex(ValueError, 'closed'): domain.source

    def test_index_budget_is_reserved_before_any_full_key_build(self):
        with self.environment() as (spec, data, stats):
            domain = owner._price_view(data, spec, CUTOFF, list(CALENDAR), stats)
            before = stats['_price_view_bytes']; cap = stats['_limits']['maximum_matrix_bytes']
            stats['_limits']['maximum_matrix_bytes'] = before + 1
            with patch.object(labels, '_compile_forward_index', side_effect=AssertionError('built before budget')):
                with self.assertRaisesRegex(ValueError, 'working byte budget'):
                    list(labels._forward_rows(None, None, None, calendar=CALENDAR,
                        features=list(CALENDAR), horizon_sessions=5, source_ref=domain.source_ref, _domain=domain))
            self.assertEqual(stats['_price_view_bytes'], before)
            self.assertNotIn('price_domain_index_builds', stats)
            stats['_limits']['maximum_matrix_bytes'] = cap
            domain.close(); self.assertEqual(stats['_price_view_bytes'], 0)

    def test_missing_late_and_invalid_endpoint_results_match_uncompiled_path(self):
        def transformed(wire):
            row = next(r for r in wire['records'] if r['security_id'] == 'A' and r['session'] == CALENDAR[1])
            row['open'] = None
            meta = next(m for m in wire['field_meta']['close']['by_key']
                        if m['security_id'] == 'B' and m['session'] == CALENDAR[5])
            meta['factor_provenance']['usable_from'] = '2024-03-11T00:00:00Z'
        with self.environment(transformed) as (spec, data, stats):
            with owner._price_view(data, spec, CUTOFF, list(CALENDAR), stats) as domain:
                expected = self.expected(data.adjusted[-1], list(CALENDAR), domain.source_ref)
                actual = []
                for days in ([CALENDAR[0]], list(CALENDAR[1:])):
                    actual.extend(labels._forward_rows(None, None, None, calendar=CALENDAR,
                        features=days, horizon_sessions=5, source_ref=domain.source_ref, _domain=domain))
                self.assertEqual(actual, expected)
                self.assertEqual(actual[0]['invalid_reason'], 'missing_start_open')
                self.assertEqual(actual[1]['invalid_reason'], 'unavailable_end_close:provenance_exceeds_query_cutoff')
                self.assertTrue(all(r['return'] is None for r in actual if not r['valid']))

    def test_failed_actual_index_charge_never_leaves_an_unaccounted_cached_index(self):
        with self.environment() as (spec, data, stats):
            with owner._price_view(data, spec, CUTOFF, list(CALENDAR), stats) as domain:
                before = stats['_price_view_bytes']
                with patch.object(owner._RawPriceDomain, '_charge', side_effect=ValueError('actual owner budget')):
                    with self.assertRaisesRegex(ValueError, 'actual owner budget'):
                        list(labels._forward_rows(None, None, None, calendar=CALENDAR,
                            features=list(CALENDAR), horizon_sessions=5, source_ref=domain.source_ref, _domain=domain))
                self.assertIsNone(owner._RAW_DOMAINS[domain]['compiled'])
                self.assertEqual(stats['_price_view_bytes'], before)
                self.assertEqual(stats['price_domain_index_build_attempts'], 1)
                self.assertNotIn('price_domain_index_builds', stats)
                rows = list(labels._forward_rows(None, None, None, calendar=CALENDAR,
                    features=list(CALENDAR), horizon_sessions=5, source_ref=domain.source_ref, _domain=domain))
                self.assertEqual(rows, self.expected(data.adjusted[-1], list(CALENDAR), domain.source_ref))
                self.assertEqual(stats['price_domain_index_build_attempts'], 2)
                self.assertEqual(stats['price_domain_index_builds'], 1)

    def test_saved_target_validation_charges_the_domain_and_child_and_restores_on_failure(self):
        with self.environment() as (spec, data, stats), tempfile.TemporaryDirectory() as temp:
            with owner._price_view(data, spec, CUTOFF, list(CALENDAR), stats) as domain:
                publish = owner._publish
                def measured(*args, **kwargs):
                    self.assertGreater(stats['_retained_raw_bytes'], 0)
                    self.assertEqual(stats['_store'].shared_bytes,
                        stats['_price_view_bytes'] + stats['_retained_raw_bytes'])
                    return publish(*args, **kwargs)
                with patch.object(owner, '_publish', side_effect=measured):
                    owner._raw(spec, CUTOFF, [CALENDAR[0]], Path(temp), stats, domain)
                self.assertEqual(stats['_retained_raw_bytes'], 0)
                self.assertEqual(stats['_store'].shared_bytes, stats['_price_view_bytes'])
                with patch.object(owner, '_publish', side_effect=ValueError('synthetic publication failure')):
                    with self.assertRaisesRegex(ValueError, 'synthetic publication failure'):
                        owner._raw(spec, CUTOFF, [CALENDAR[1]], Path(temp), stats, domain)
                self.assertEqual(stats['_retained_raw_bytes'], 0)
                self.assertEqual(stats['_store'].shared_bytes, stats['_price_view_bytes'])
            self.assertEqual(stats['_store'].shared_bytes, 0)


if __name__ == '__main__': unittest.main()

"""Small public Data/Core goldens for the opt-in bounded Feature producer.

Only Qlib's native read/activation boundary is substituted. The actual public
export format, Reader, membership, adjustment, catalog and Core mathematics
are exercised; this is not a supplier/real-input/RSS acceptance.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from axiom_data import Data
from axiom_engine.core import ExecutionContext, execute_feature_plan, execute_feature_plan_batch
from axiom_research.data_adapter import _adapt_decision_wires
from axiom_research.feature_catalog import load_feature_catalog, build_feature_plan
from axiom_research.stock_artifacts import digest
from axiom_research.stock_ml import _adjust_feature, _instant, _project_qlib
from axiom_research.stock_matrix_feature_producer import (
    FIELDS, _NativeWindow, _view_signature, classify_shared_feature_views,
    prepare_matrix_qlib, iter_matrix_feature_days)


class MemoryFeatureInputs:
    """Test-only Store protocol with immutable facts and a real Data facade."""
    def __init__(self, root):
        self.catalog = load_feature_catalog()
        self.selection = [entry.to_dict() for entry in self.catalog.default_selection]
        self.chosen = self.catalog.select(self.selection)
        self.history = max(entry['lookback'] for entry in self.chosen)
        start = date(2026, 1, 2)
        self.calendar = [(start+timedelta(days=2*i)).isoformat() for i in range(self.history+4)]
        self.outputs = self.calendar[self.history-1:]
        self.symbols = ['A', 'B', 'C']
        self.snapshot = 's-memory-feature-fixed'
        self.universe_id = 'memory_members_v1'
        self.config = {'snapshot': self.snapshot, 'pit_policy': 'operational_pit_v1',
            'symbols': self.symbols, 'universe_id': self.universe_id, 'calendar': self.calendar,
            'read_sessions': self.calendar, 'feature_sessions': self.outputs,
            'cutoff_by_session': {day: day+'T20:30:00+08:00' for day in self.calendar},
            'feature_selection': self.selection, 'scope_ref': digest('memory feature scope')}
        self.partitions = {name: {} for name in ('market_daily', 'adjustment_factors',
                                                'universe_membership', 'trading_calendar')}
        for i, day in enumerate(self.calendar):
            calendar = {'session': day, 'is_open': True, 'revision_id': 'calendar-'+day,
                'revision_sequence': 1, 'first_observed_at': day+'T00:00:00Z'}
            self.partitions['trading_calendar'].setdefault(day[:7], []).append(calendar)
            for j, symbol in enumerate(self.symbols):
                base = {'security_id': symbol, 'session': day,
                    'revision_sequence': 1, 'first_observed_at': day+'T08:15:00.123456Z',
                    'source_available_at': None, 'evidence_ref': None}
                opening = 10.0+j*.4+i*.07
                row = {**base, 'revision_id': 'price-'+symbol+'-'+day, 'raw_batch_id': 'price-'+day,
                    'open': opening, 'high': opening+1.0+j*.08, 'low': opening-.8-j*.03,
                    'close': opening+.3+j*j*.06+(i % 4)*.025, 'amount_cny': 1000.0+i*(j+1)*11.3}
                self.partitions['market_daily'].setdefault(day[:7], []).append(row)
                factor = {**base, 'revision_id': 'factor-'+symbol+'-'+day, 'raw_batch_id': 'factor-'+day,
                          'factor': 1.0+(i % 7)*.15+j*.05}
                self.partitions['adjustment_factors'].setdefault(day[:7], []).append(factor)
        self.revised_day = self.calendar[self.history-2]
        self.late_clock = (_instant(self.config['cutoff_by_session'][self.outputs[0]])+
                           timedelta(microseconds=1)).isoformat().replace('+00:00', 'Z')
        self._revision('market_daily', 'B', self.revised_day, {'close': 18.125}, 'late-price')
        self._revision('adjustment_factors', 'B', self.revised_day, {'factor': 2.375}, 'late-factor')
        self.null_day = self.calendar[self.history-3]
        self._revision('market_daily', 'C', self.null_day, {'close': None}, 'late-null')
        self.restored_day = self.calendar[self.history-4]
        restored = self._row('market_daily', 'A', self.restored_day)['close']
        self._row('market_daily', 'A', self.restored_day)['close'] = None
        self._revision('market_daily', 'A', self.restored_day, {'close': restored}, 'late-restored')
        stop = self.outputs[1]
        membership = []
        for symbol in self.symbols:
            membership.append({'security_id': symbol, 'universe_id': self.universe_id,
                'membership_id': symbol+'-member', 'effective_from': self.calendar[0],
                'effective_to': stop if symbol == 'C' else None, 'revision_id': symbol+'-member',
                'revision_sequence': 1, 'raw_batch_id': 'memory-members',
                'first_observed_at': self.calendar[0]+'T08:15:00.123456Z',
                'source_available_at': None, 'evidence_ref': None})
        self.partitions['universe_membership']['history'] = membership
        self.manifest = {'snapshot_id': self.snapshot, 'domains': {}}
        fields = {field: {'dtype': 'float64', 'unit': 'CNY' if field == 'amount_cny' else 'CNY/share'}
                  for field in FIELDS[:-1]}
        for domain, schema, keys in [('market_daily', fields, ['security_id', 'session']),
                ('adjustment_factors', {'factor': {'dtype': 'float64', 'unit': 'dimensionless'}},
                 ['security_id', 'session']),
                ('universe_membership', {'universe_id': {'dtype': 'string'},
                    'effective_from': {'dtype': 'date'}, 'effective_to': {'dtype': 'date'}}, ['membership_id']),
                ('trading_calendar', {'is_open': {'dtype': 'bool'}}, ['session'])]:
            self.manifest['domains'][domain] = {'contract': {'contract_id': domain+'_memory_v1',
                'logical_key': keys, 'fields': schema}, 'source_profile': {'id': domain+'_memory',
                    'availability': {'timezone': 'Asia/Shanghai', 'session_release_time': '17:00:00'}},
                'partitions': [{'partition': month, 'domain': domain}
                               for month in sorted(self.partitions[domain])], 'coverage': {}}
        self.manifest['domains']['universe_membership']['coverage']['complete_states'] = [
            {'universe_id': self.universe_id, 'complete': True, 'effective_from': self.calendar[0],
             'effective_to': stop, 'members': self.symbols, 'raw_batch_id': 'memory-state-initial',
             'first_observed_at': self.calendar[0]+'T08:15:00.123456Z'},
            {'universe_id': self.universe_id, 'complete': True, 'effective_from': stop,
             'effective_to': None, 'members': ['A', 'B'], 'raw_batch_id': 'memory-state-later',
             'first_observed_at': stop+'T08:15:00.123456Z'}]
        self.fact_root = root/'fact-root-must-not-be-created'
        self.data = Data(self.fact_root, cache_bytes=0)
        self.data.store = self
        self.native_calls = []

    def _row(self, domain, symbol, day):
        return next(row for row in self.partitions[domain][day[:7]]
                    if row['security_id'] == symbol and row['session'] == day and row['revision_sequence'] == 1)

    def _revision(self, domain, symbol, day, changes, ref):
        row = {**deepcopy(self._row(domain, symbol, day)), **changes,
            'revision_sequence': 2, 'revision_id': ref, 'raw_batch_id': ref,
            'first_observed_at': self.late_clock}
        self.partitions[domain][day[:7]].append(row)

    def load_snapshot(self, snapshot_id):
        if snapshot_id != self.snapshot: raise AssertionError('snapshot changed')
        return deepcopy(self.manifest)

    def verify_partition(self, part):
        pass

    def read_partition(self, part, *, columns=None, symbols=None, sessions=None):
        import pyarrow as pa
        rows = self.partitions[part['domain']][part['partition']]
        if symbols is not None: rows = [row for row in rows if row.get('security_id') in symbols]
        if sessions is not None and part['domain'] != 'universe_membership':
            rows = [row for row in rows if row['session'] in sessions]
        return pa.Table.from_pylist(rows if columns is None else
                                   [{column: row.get(column) for column in columns} for row in rows])

    def native_read(self, view, *, fields, symbols, start, end):
        """Read the actual exporter bytes; substitute only Qlib native API."""
        self.native_calls.append({'start': start, 'end': end, 'fields': fields, 'symbols': symbols})
        calendar = view.manifest['calendar']
        positions = range(calendar.index(start), calendar.index(end)+1)
        rows, keys = [], []
        for symbol in symbols:
            instrument = view.manifest['instrument_map'][symbol]
            arrays = [np.memmap(view.path/'features'/instrument.lower()/(field+'.day.bin'),
                               mode='r', dtype='<f4') for field in fields]
            for position in positions:
                rows.append([float(values[position+1]) for values in arrays])
                keys.append((instrument, pd.Timestamp(calendar[position])))
            for values in arrays: values._mmap.close()
        # Deliberately reverse physical order: only stable keys can align it.
        return pd.DataFrame(rows[::-1], columns=['$'+field for field in fields],
                            index=pd.MultiIndex.from_tuples(keys[::-1]), dtype=np.float32)


class MatrixFeatureProducerTests(unittest.TestCase):
    def _prepared(self, fixture, root):
        with patch('axiom_research.qlib_adapter.QlibView.activate', lambda view: view), \
             patch('axiom_research.qlib_adapter.QlibView.read', lambda view, **kwargs: fixture.native_read(view, **kwargs)):
            return prepare_matrix_qlib(fixture.data, config=fixture.config, destination=root/'export')

    def _original_view(self, fixture, prepared, day, *, all_outputs=False):
        """Original Data/projection/adapter/catalog/Core golden for one view.

        The temporary mapping contains this actual Reader selection, rather
        than claiming those later revisions exist in the frozen Qlib file.
        Only codec/Feature arithmetic is compared; its Frame refs differ.
        """
        if all_outputs: history = tuple(fixture.calendar)
        else:
            i = fixture.calendar.index(day)
            history = tuple(fixture.calendar[i-fixture.history+1:i+1])
        cutoffs = {session: fixture.config['cutoff_by_session'][day] for session in history}
        price = fixture.data.read(snapshot=fixture.snapshot,
            query=replace(prepared['price_query'], sessions=history, cutoff_by_session=cutoffs))
        factor = fixture.data.read(snapshot=fixture.snapshot,
            query=replace(prepared['factor_query'], sessions=history, cutoff_by_session=cutoffs))
        values = {}
        for batch in (price, factor):
            for row in batch.to_json()['records']:
                key = row['security_id'], row['session']
                values.setdefault(key, {}).update({field: None if row[field] is None else float(np.float32(row[field]))
                    for field in batch.context['query']['fields']})
        price = _project_qlib(price, values, 'actual_reader_rounding_oracle')
        factor = _project_qlib(factor, values, 'actual_reader_rounding_oracle')
        _, adjusted = _adjust_feature(price, factor, day, _with_wire=True)
        member = fixture.data.members(snapshot=fixture.snapshot,
            query=replace(prepared['member_query'], sessions=history, cutoff_by_session=cutoffs))
        output_days = fixture.outputs if all_outputs else [day]
        adapted = _adapt_decision_wires(adjusted, member.to_json(),
            recipe_ref=fixture.catalog.recipe_ref(fixture.selection, normalized=True),
            output_keys=tuple((symbol, session) for session in output_days for symbol in fixture.symbols),
            source_granularity='batch_field')
        plan = build_feature_plan(adapted.plan, fixture.selection, catalog=fixture.catalog, normalized=True)
        return plan, adapted, member

    def test_prepare_exports_original_view_without_full_native_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture = MemoryFeatureInputs(root)
            before = digest({'manifest': fixture.manifest, 'facts': fixture.partitions})
            prepared = self._prepared(fixture, root)
            self.assertEqual(fixture.native_calls, [])
            self.assertNotIn('values', prepared)
            self.assertEqual(set(prepared), {'view', 'view_reference', 'price_query', 'factor_query',
                                            'member_query', 'qlib_path', 'seconds'})
            self.assertTrue((prepared['qlib_path']/'axiom-qlib.json').is_file())
            self.assertEqual(prepared['view_reference']['snapshot_id'], fixture.snapshot)
            self.assertEqual(before, digest({'manifest': fixture.manifest, 'facts': fixture.partitions}))
            self.assertFalse(fixture.fact_root.exists())

    def test_bounded_public_reader_revision_null_cohort_and_core_goldens(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture = MemoryFeatureInputs(root); prepared = self._prepared(fixture, root)
            immutable = digest({'manifest': fixture.manifest, 'facts': fixture.partitions})
            stats, executed, queries = {}, {}, []
            original_read = fixture.data.read
            def read(*, snapshot, query):
                queries.append(query)
                return original_read(snapshot=snapshot, query=query)
            def execute(requests, *, reuse_budget_bytes):
                result = execute_feature_plan_batch(requests, reuse_budget_bytes=reuse_budget_bytes)
                for (plan, facts, context), frame in zip(requests,result['frames']):
                    day = frame.to_dict()['rows'][0]['session']
                    executed[day] = frame.identity, plan.identity, facts.identity, context.identity
                return result
            with patch('axiom_research.qlib_adapter.QlibView.read', lambda view, **kwargs: fixture.native_read(view, **kwargs)), \
                 patch.object(fixture.data, 'read', side_effect=read), \
                 patch('axiom_engine.core.execute_feature_plan_batch', side_effect=execute) as core, \
                 patch('axiom_research.stock_training.fit_predict_stock_model', side_effect=AssertionError('trained')):
                actual = list(iter_matrix_feature_days(fixture.data, config=fixture.config, catalog=fixture.catalog,
                    chosen=fixture.chosen, qlib_inputs=prepared, history_sessions=fixture.history,
                    output_block_sessions=2, maximum_resident_bytes=64*1024**2, stats=stats))
                self.assertEqual(core.call_count, 3)
            self.assertEqual(len(actual), 5)
            self.assertEqual(stats['native_window_reads'], 3)
            self.assertLess(stats['native_window_peak_sessions'], len(fixture.calendar))
            self.assertEqual(stats['core_single_output_groups'], 5)
            self.assertEqual(stats['core_daily_frames'],5)
            self.assertEqual(stats['actual_core_multi_view_batches'],2)
            self.assertEqual(stats['actual_core_multi_output_groups'], 0)
            self.assertEqual(stats['data_read_calls'], 15)
            self.assertGreater(stats['reader_projection_fallback_cells'], 0)
            self.assertGreater(stats['reader_projection_fallback_null_cells'], 0)
            self.assertGreater(stats['native_value_reuses'], 0)
            self.assertEqual(stats['classifier_status_counts'], {'REQUIRES_VIEW_FALLBACK': 2})
            self.assertGreater(stats['classifier_reason_counts']['FACT_SOURCE_CONFLICT'], 0)
            self.assertLess(stats['working_graph_peak_bytes'], 64*1024**2)
            self.assertEqual(stats['combined_working_graph_peak_bytes'], stats['working_graph_peak_bytes'])
            self.assertEqual(stats['yield_live_bytes'], 0)
            for (rows, proof), day in zip(actual, fixture.outputs):
                plan, adapted, membership = self._original_view(fixture, prepared, day)
                expected = execute_feature_plan(plan, adapted.facts, adapted.context).to_dict()['rows']
                by_symbol = {row['security_id']: row for row in expected}
                member = {row['security_id']: row['is_member'] for row in membership.to_json()['records'] if row['session'] == day}
                self.assertEqual(len(rows), 3)
                self.assertEqual(proof['session'], day)
                self.assertEqual(proof['core_frame_ref'], executed[day][0])
                self.assertEqual(digest(proof['core_plan']), executed[day][1])
                self.assertEqual((proof['fact_ref'], proof['context_ref']), executed[day][2:])
                self.assertEqual(set(proof['cutoffs'].values()), {fixture.config['cutoff_by_session'][day]})
                for row in rows:
                    oracle = by_symbol[row['security_id']]
                    for field, original in [('values', 'values'), ('availability', 'availability'),
                                             ('validity', 'valid'), ('reasons', 'reasons')]:
                        self.assertEqual(row[field], oracle[original], (day, row['security_id'], field))
                    self.assertEqual(row['member'], member[row['security_id']])
                    self.assertEqual(row['source_refs'], list(executed[day][:2]))
                self.assertEqual(rows[2]['member'], day == fixture.outputs[0])
                if day != fixture.outputs[0]:
                    self.assertTrue(all(value is None for value in rows[2]['values']))
                oracle_sources = {source['field']: source for source in adapted.source_evidence.values()}
                for source in proof['source_evidence'].values():
                    q = source['query_context']['query']
                    self.assertEqual(q['sessions'], proof['sessions'])
                    self.assertEqual(set(q['cutoff_by_session'].values()), {fixture.config['cutoff_by_session'][day]})
                    self.assertEqual(source['provenance_by_key_ref'], digest(oracle_sources[source['field']]['provenance_by_key']))
                    if source['field'] != 'is_member':
                        self.assertEqual(q['adjustment_anchor'], day)
                        self.assertEqual(source['batch_ref'], proof['adjusted_input_ref'])
            # The old strict projection still diagnoses this ordinary revision;
            # the new path handled it while retaining actual current provenance.
            day = fixture.outputs[1]
            i = fixture.calendar.index(day); history = tuple(fixture.calendar[i-fixture.history+1:i+1])
            price = fixture.data.read(snapshot=fixture.snapshot, query=replace(prepared['price_query'],
                sessions=history, cutoff_by_session={session: fixture.config['cutoff_by_session'][day] for session in history}))
            fields = price.to_json()['field_meta']['close']['by_key']
            revision = next(meta for meta in fields if (meta['security_id'], meta['session']) == ('B', fixture.revised_day))
            self.assertEqual(revision['revision_id'], 'late-price')
            self.assertEqual(_instant(revision['usable_from']), _instant(fixture.late_clock))
            for query, (_, proof) in zip(queries[::3], actual):
                self.assertEqual(query.sessions, tuple(proof['sessions']))
                self.assertEqual(query.symbols, tuple(fixture.symbols))
                self.assertEqual(query.purpose, 'decision_facts')
                self.assertEqual(query.pit_policy, 'operational_pit_v1')
            self.assertEqual(immutable, digest({'manifest': fixture.manifest, 'facts': fixture.partitions}))
            self.assertFalse(fixture.fact_root.exists())

    def test_existing_core_compatible_panel_is_one_true_multi_output_call(self):
        """Core capability proof, not original per-day PIT/anchor admission."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture = MemoryFeatureInputs(root); prepared = self._prepared(fixture, root)
            plan, adapted, _ = self._original_view(fixture, prepared, fixture.outputs[-1], all_outputs=True)
            with patch('axiom_engine.core.execute_feature_plan', wraps=execute_feature_plan) as core:
                from axiom_engine.core import execute_feature_plan as public_execute
                grouped = public_execute(plan, adapted.facts, adapted.context)
                self.assertEqual(core.call_count, 1)
            indexed = {(row['security_id'], row['session']): row for row in grouped.to_dict()['rows']}
            previous = None
            for day in fixture.outputs:
                wire = adapted.context.to_dict()
                wire['output_keys'] = [[symbol, day] for symbol in fixture.symbols]
                context = ExecutionContext.from_dict(wire)
                single = execute_feature_plan(plan, adapted.facts, context)
                self.assertNotEqual(single.identity, grouped.identity)
                for row in single.to_dict()['rows']:
                    self.assertEqual(row, indexed[row['security_id'], row['session']])
                signature = _view_signature(plan, adapted.facts, context)
                if previous is not None:
                    self.assertEqual(classify_shared_feature_views(previous, signature),
                                     {'status': 'SHARED_PANEL_COMPATIBLE', 'reasons': []})
                previous = signature

    def test_classifier_rejects_same_values_with_different_clock_source_or_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture = MemoryFeatureInputs(root); prepared = self._prepared(fixture, root)
            plan, adapted, _ = self._original_view(fixture, prepared, fixture.outputs[0])
            signature = _view_signature(plan, adapted.facts, adapted.context)
            key = next(iter(signature['facts']))
            for field, reason in [('availability', 'FACT_CLOCK_CONFLICT'), ('sources', 'FACT_SOURCE_CONFLICT'),
                                  ('reasons', 'FACT_MISSING_REASON_CONFLICT')]:
                with self.subTest(field=field):
                    changed = deepcopy(signature); changed['facts'][key][field] = digest('changed '+field)
                    classified = classify_shared_feature_views(signature, changed)
                    self.assertEqual(classified['status'], 'REQUIRES_VIEW_FALLBACK')
                    self.assertIn(reason, classified['reasons'])
            changed = deepcopy(signature); changed['reference'][key] = digest('different member')
            self.assertIn('REFERENCE_CONFLICT', classify_shared_feature_views(signature, changed)['reasons'])

    def test_identity_factors_still_require_original_anchor_clocks_and_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture = MemoryFeatureInputs(root)
            # Eliminate all revisions and numeric adjustment differences, so
            # this counterexample is caused by original anchor provenance.
            for domain in ('market_daily', 'adjustment_factors'):
                for month, rows in fixture.partitions[domain].items():
                    fixture.partitions[domain][month] = [row for row in rows if row['revision_sequence'] == 1]
                    for row in fixture.partitions[domain][month]:
                        if domain == 'adjustment_factors': row['factor'] = 1.0
                        elif row['close'] is None: row['close'] = 12.125
            prepared = self._prepared(fixture, root)
            config = deepcopy(fixture.config); config['feature_sessions'] = fixture.outputs[:2]
            key = 'A', fixture.revised_day
            inputs = []
            def capture(requests, *, reuse_budget_bytes):
                for plan,facts,context in requests:
                    fields = [column['name'] for column in facts.to_dict()['schema']]
                    row = next(row for row in facts.to_dict()['rows']
                               if (row['security_id'], row['session']) == key)
                    i = fields.index('close')
                    inputs.append((row['values'][i], row['availability'][i], row['sources'][i]))
                return execute_feature_plan_batch(requests,reuse_budget_bytes=reuse_budget_bytes)
            stats = {}
            with patch('axiom_research.qlib_adapter.QlibView.read', lambda view, **kwargs: fixture.native_read(view, **kwargs)), \
                 patch('axiom_engine.core.execute_feature_plan_batch', side_effect=capture):
                list(iter_matrix_feature_days(fixture.data, config=config, catalog=fixture.catalog,
                    chosen=fixture.chosen, qlib_inputs=prepared, history_sessions=fixture.history,
                    output_block_sessions=2, maximum_resident_bytes=64*1024**2, stats=stats))
            self.assertEqual(inputs[0][0], inputs[1][0])
            self.assertNotEqual(inputs[0][1], inputs[1][1])
            self.assertNotEqual(inputs[0][2], inputs[1][2])
            self.assertEqual(stats['classifier_status_counts'], {'REQUIRES_VIEW_FALLBACK': 1})
            self.assertEqual(stats['classifier_reason_counts']['FACT_CLOCK_CONFLICT'], 1)
            self.assertEqual(stats['classifier_reason_counts']['FACT_SOURCE_CONFLICT'], 1)
            self.assertEqual(stats['actual_core_multi_output_groups'], 0)

    def test_large_reader_source_context_is_rejected_before_json_duplication(self):
        from axiom_data import DataBatch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture = MemoryFeatureInputs(root); prepared = self._prepared(fixture, root)
            day = fixture.outputs[0]; history = tuple(fixture.calendar[:fixture.history])
            query = replace(prepared['price_query'], sessions=history,
                            cutoff_by_session={session: fixture.config['cutoff_by_session'][day] for session in history})
            original = fixture.data.read(snapshot=fixture.snapshot, query=query)
            large = DataBatch(original.frame, original.field_meta,
                              {**original.context, 'coverage': {'retained_source_proof': 'x'*300000}})
            native = fixture.native_read(prepared['view'], fields=FIELDS, symbols=tuple(fixture.symbols),
                                        start=history[0], end=history[-1])
            window = _NativeWindow(native, sessions=history, symbols=fixture.symbols,
                                   instrument_map=prepared['view_reference']['instrument_map'])
            with patch('axiom_data.protocols.DataBatch.to_json', side_effect=AssertionError('oversized source serialized')):
                with self.assertRaisesRegex(ValueError, 'Reader projection working set exceeds resident budget'):
                    window.project(large, view_ref=prepared['view_reference']['view_id'], stats={},
                                   maximum_resident_bytes=128*1024)

    def test_dynamic_caller_growth_rejects_combined_budget_before_new_json_or_core(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture = MemoryFeatureInputs(root); prepared = self._prepared(fixture, root)
            budget = 64*1024**2
            retained = [0]
            stats = {}
            with patch('axiom_research.qlib_adapter.QlibView.read', lambda view, **kwargs: fixture.native_read(view, **kwargs)), \
                 patch('axiom_engine.core.execute_feature_plan_batch', wraps=execute_feature_plan_batch) as core:
                iterator = iter_matrix_feature_days(fixture.data, config=fixture.config, catalog=fixture.catalog,
                    chosen=fixture.chosen, qlib_inputs=prepared, history_sessions=fixture.history,
                    output_block_sessions=2, maximum_resident_bytes=budget, stats=stats,
                    caller_retained_bytes=lambda: retained[0])
                first = next(iterator)
                self.assertEqual(first[1]['session'], fixture.outputs[0])
                self.assertEqual(core.call_count, 1)
                self.assertGreater(stats['yield_live_bytes'], 0)
                # Caller alone fits, as does the producer alone. Their sum
                # must be rejected on resumption before duplicating the next
                # Reader wire or executing another Core cross-section.
                retained[0] = budget
                with patch('axiom_data.protocols.DataBatch.to_json', side_effect=AssertionError('combined budget serialized')):
                    with self.assertRaisesRegex(ValueError, 'combined producer/caller resident budget exceeded'):
                        next(iterator)
                self.assertEqual(core.call_count, 1)
                self.assertGreater(stats['combined_working_graph_peak_bytes'], budget)
                self.assertLess(stats['working_graph_peak_bytes'], budget)
                self.assertEqual(stats['yield_live_bytes'], 0)

    def test_caller_getter_rejects_negative_bool_and_non_callable_before_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture = MemoryFeatureInputs(root); prepared = self._prepared(fixture, root)
            for getter in (lambda: -1, lambda: True, lambda: False, 0):
                with self.subTest(getter=getter), \
                     patch.object(fixture.data, 'read', side_effect=AssertionError('invalid caller read Data')), \
                     patch('axiom_research.qlib_adapter.QlibView.read', side_effect=AssertionError('invalid caller read native')):
                    with self.assertRaisesRegex(ValueError, 'caller_retained_bytes'):
                        next(iter_matrix_feature_days(fixture.data, config=fixture.config, catalog=fixture.catalog,
                            chosen=fixture.chosen, qlib_inputs=prepared, history_sessions=fixture.history,
                            output_block_sessions=2, maximum_resident_bytes=64*1024**2,
                            caller_retained_bytes=getter))

    def test_budget_and_duplicate_missing_native_keys_fail_before_core(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture = MemoryFeatureInputs(root); prepared = self._prepared(fixture, root)
            with patch('axiom_research.qlib_adapter.QlibView.read', lambda view, **kwargs: fixture.native_read(view, **kwargs)), \
                 patch('axiom_engine.core.execute_feature_plan_batch', side_effect=AssertionError('Core on invalid window')):
                with self.assertRaisesRegex(ValueError, 'one native Feature window exceeds resident budget'):
                    list(iter_matrix_feature_days(fixture.data, config=fixture.config, catalog=fixture.catalog,
                        chosen=fixture.chosen, qlib_inputs=prepared, history_sessions=fixture.history,
                        output_block_sessions=2, maximum_resident_bytes=1))
                self.assertEqual(fixture.native_calls, [])
                native = fixture.native_read(prepared['view'], fields=FIELDS, symbols=tuple(fixture.symbols),
                                            start=fixture.calendar[0], end=fixture.outputs[0])
                for invalid in (native.iloc[:-1], pd.concat([native, native.iloc[:1]])):
                    with self.assertRaisesRegex(ValueError, 'complete unique Qlib native window grid'):
                        _NativeWindow(invalid, sessions=fixture.calendar[:fixture.history], symbols=fixture.symbols,
                                      instrument_map=prepared['view_reference']['instrument_map'])


if __name__ == '__main__':
    unittest.main()

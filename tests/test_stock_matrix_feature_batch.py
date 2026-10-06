"""Small public Reader/export/Core parity; no real data root or ML execution."""
from copy import deepcopy
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import weakref
from unittest.mock import Mock, patch

from axiom_engine.core import execute_feature_plan, execute_feature_plan_batch
from axiom_research.stock_artifacts import digest
from axiom_research.stock_matrix_feature_producer import (
    iter_matrix_feature_days, prepare_matrix_qlib)
from test_stock_matrix_feature_producer import MemoryFeatureInputs


LIMIT = 32*1024**2
_ACCEPTANCE = []
_BUDGET_ACCEPTANCE = []


def prepared_inputs(fixture, root):
    with patch('axiom_research.qlib_adapter.QlibView.activate', lambda view: view):
        return prepare_matrix_qlib(fixture.data, config=fixture.config, destination=root/'export')


def arguments(fixture, prepared, **options):
    return dict(config=fixture.config, catalog=fixture.catalog, chosen=fixture.chosen,
        qlib_inputs=prepared, history_sessions=fixture.history,
        output_block_sessions=5, maximum_resident_bytes=LIMIT, **options)


class MatrixFeatureBatchTests(unittest.TestCase):
    def test_sealed_private_view_charges_cover_shared_graphs_and_release_without_rescanning(self):
        from axiom_research import stock_matrix_feature_producer as producer
        stats = {}
        shared = {'source': ['source-ref']*64}
        pending = []
        for index in range(16):
            evidence = {'session': str(index), 'source_evidence': shared,
                        'records': [{'value': value, 'sources': shared} for value in range(32)]}
            pending.append(producer._seal_pending_view(
                (shared, shared, shared), evidence, {'A': True}, 1024, 512, stats=stats))
            self.assertGreaterEqual(producer._pending_owned_bytes(pending), producer._owned_bytes(pending))
        self.assertEqual(stats['pending_view_size_measurements'], 16)
        before = producer._pending_owned_bytes(pending)
        released = pending[0][5]
        pending[0] = None
        self.assertEqual(producer._pending_owned_bytes(pending), before-released)
        self.assertGreaterEqual(producer._pending_owned_bytes(pending), producer._owned_bytes(pending))
        with patch.object(producer, '_owned_bytes', side_effect=AssertionError('sealed graph rescanned')):
            for unused in range(100):
                self.assertGreater(producer._pending_core_bytes(pending), before-released)
            pending.clear()
            self.assertEqual(producer._pending_owned_bytes(pending), sys.getsizeof(pending))

    def test_zero_and_positive_reuse_execute_actual_five_view_batches_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = MemoryFeatureInputs(root)
            prepared = prepared_inputs(fixture, root)
            original_facts = digest({'manifest': fixture.manifest, 'facts': fixture.partitions})
            variant_rows = []
            for budget in (0, LIMIT//8):
                stats, captured, invocations = {}, {}, []

                def batch(requests, *, reuse_budget_bytes):
                    self.assertIs(type(requests), tuple)
                    self.assertEqual(reuse_budget_bytes, budget)
                    tick = time.perf_counter_ns()
                    result = execute_feature_plan_batch(requests, reuse_budget_bytes=reuse_budget_bytes)
                    invocations.append({'views': len(requests), 'wall_ns': time.perf_counter_ns()-tick,
                                        'stats': deepcopy(result['stats'])})
                    self.assertEqual(set(result), {'frames', 'stats'})
                    self.assertIs(type(result['frames']), tuple)
                    for request, frame in zip(requests, result['frames']):
                        day = frame.to_dict()['rows'][0]['session']
                        captured[day] = (request, frame)
                    return result

                with patch('axiom_research.qlib_adapter.QlibView.read',
                           lambda view, **kwargs: fixture.native_read(view, **kwargs)), \
                     patch('axiom_engine.core.execute_feature_plan_batch', side_effect=batch) as calls, \
                     patch('axiom_engine.core.execute_feature_plan', side_effect=AssertionError('legacy public Core called')), \
                     patch('axiom_research.stock_training.fit_predict_stock_model', side_effect=AssertionError('ML called')):
                    pipeline_begin = time.perf_counter_ns()
                    actual = list(iter_matrix_feature_days(fixture.data,
                        **arguments(fixture, prepared, reuse_budget_bytes=budget, stats=stats)))
                    pipeline_wall_ns = time.perf_counter_ns()-pipeline_begin
                self.assertEqual(calls.call_count, 1)
                self.assertEqual([entry['views'] for entry in invocations], [5])
                self.assertEqual((stats['core_calls'], stats['feature_core_calls'], stats['core_batch_calls']), (1, 1, 1))
                self.assertEqual((stats['core_batch_views'], stats['core_daily_frames'], stats['core_single_output_groups']), (5, 5, 5))
                self.assertEqual(stats['actual_core_multi_view_batches'], 1)
                self.assertEqual(stats['actual_core_multi_output_groups'], 0)
                self.assertEqual(stats['data_read_calls'], 15)
                self.assertEqual(stats['native_window_reads'], 1)
                self.assertEqual(stats['core_reuse_budget_bytes'], budget)
                self.assertEqual(stats['core_batch_total_ns'], sum(entry['stats']['total_ns'] for entry in invocations))
                self.assertEqual(stats['core_batch_retained_reuse_bytes'], 0)
                self.assertGreater(stats['core_batch_total_ns'], 0)
                self.assertGreater(stats['feature_pipeline_total_ns'], stats['core_batch_total_ns'])
                self.assertGreater(stats['native_window_read_ns'], 0)
                self.assertGreater(stats['native_projection_ns'], 0)
                self.assertGreater(stats['reader_projection_fallback_cells'], 0)
                self.assertGreater(stats['reader_projection_fallback_null_cells'], 0)
                self.assertGreater(stats['native_value_reuses'], 0)
                self.assertLessEqual(stats['combined_working_graph_peak_bytes'], LIMIT)
                self.assertGreater(stats['pending_view_peak_bytes'], 0)
                self.assertEqual(stats['pending_view_size_measurements'], 5)
                self.assertEqual(stats['core_frame_size_measurements'], 5)
                self.assertGreater(stats['pending_view_size_measurement_ns'], 0)
                self.assertGreater(stats['core_frame_size_measurement_ns'], 0)
                self.assertEqual(stats['yield_live_bytes'], 0)
                if budget == 0:
                    self.assertEqual(stats['core_batch_key_attempts'], 0)
                    self.assertEqual(stats['core_batch_keys_built'], 0)
                    self.assertEqual(stats['core_batch_peak_reuse_bytes'], 0)
                # This small original 3-member/short-window fixture deliberately
                # stays below Core's 64-value helper threshold. Record zeros;
                # a single public invocation is not evidence of arithmetic gain.
                self.assertEqual(sum(value['reused'] for value in stats['core_batch_helpers'].values()), 0)
                self.assertEqual(len(actual), 5)
                for (rows, proof), day in zip(actual, fixture.outputs):
                    request, frame = captured[day]
                    plan, facts, context = request
                    original = execute_feature_plan(*request)
                    self.assertEqual(frame.to_dict(), original.to_dict())
                    self.assertEqual(frame.identity, original.identity)
                    wire = frame.to_dict()
                    self.assertEqual(proof['session'], day)
                    self.assertEqual(proof['core_frame_ref'], frame.identity)
                    self.assertEqual(proof['core_plan'], plan.to_dict())
                    self.assertEqual(proof['fact_ref'], facts.identity)
                    self.assertEqual(proof['context_ref'], context.identity)
                    position = fixture.calendar.index(day)
                    self.assertEqual(proof['sessions'], fixture.calendar[position-fixture.history+1:position+1])
                    self.assertEqual(set(proof['cutoffs'].values()), {fixture.config['cutoff_by_session'][day]})
                    self.assertEqual(context.to_dict()['output_keys'], [[symbol, day] for symbol in fixture.symbols])
                    expected = {row['security_id']: row for row in wire['rows']}
                    for row in rows:
                        oracle = expected[row['security_id']]
                        for actual_key, wire_key in [('values', 'values'), ('validity', 'valid'),
                                                     ('availability', 'availability'), ('reasons', 'reasons')]:
                            self.assertEqual(row[actual_key], oracle[wire_key])
                        self.assertEqual(row['source_refs'], [frame.identity, plan.identity])
                        self.assertEqual(row['knowledge_cutoff'], fixture.config['cutoff_by_session'][day])
                        self.assertEqual(row['member'], row['security_id'] != 'C' or day == fixture.outputs[0])
                    for source in proof['source_evidence'].values():
                        query = source['query_context']['query']
                        self.assertEqual(query['sessions'], proof['sessions'])
                        self.assertEqual(query['symbols'], fixture.symbols)
                        self.assertEqual(set(query['cutoff_by_session'].values()), {fixture.config['cutoff_by_session'][day]})
                        self.assertNotIn('provenance_by_key', source)
                        if source['field'] != 'is_member':
                            self.assertEqual(query['adjustment_anchor'], day)
                            self.assertEqual(source['batch_ref'], proof['adjusted_input_ref'])
                variant_rows.append(actual)
                _ACCEPTANCE.append({'reuse_budget_bytes': budget, 'producer_stats': deepcopy(stats),
                    'pipeline_wall_ns': pipeline_wall_ns,
                    'core_batch_total_ns': stats['core_batch_total_ns'],
                    'core_invocations': invocations, 'frame_refs': [captured[day][1].identity for day in fixture.outputs],
                    'full_frame_wire_and_identity_exact': True, 'daily_rows_and_proof_exact': True})
            self.assertEqual(variant_rows[0], variant_rows[1])
            self.assertEqual(original_facts, digest({'manifest': fixture.manifest, 'facts': fixture.partitions}))
            self.assertFalse(fixture.fact_root.exists())

    def test_default_reuse_and_native_windows_keep_daily_frames_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = MemoryFeatureInputs(root)
            prepared = prepared_inputs(fixture, root)
            stats = {}
            kwargs = arguments(fixture, prepared, stats=stats)
            kwargs['output_block_sessions'] = 2
            with patch('axiom_research.qlib_adapter.QlibView.read',
                       lambda view, **kwargs: fixture.native_read(view, **kwargs)), \
                 patch('axiom_engine.core.execute_feature_plan_batch', wraps=execute_feature_plan_batch) as calls:
                actual = list(iter_matrix_feature_days(fixture.data, **kwargs))
            self.assertEqual([len(call.args[0]) for call in calls.call_args_list], [2, 2, 1])
            self.assertEqual(stats['native_window_reads'], 3)
            self.assertEqual(stats['core_calls'], 3)
            self.assertEqual(stats['core_daily_frames'], 5)
            self.assertEqual(stats['actual_core_multi_view_batches'], 2)
            self.assertEqual(stats['core_reuse_budget_bytes'], LIMIT//8)
            self.assertEqual(len({proof['core_frame_ref'] for _, proof in actual}), 5)
            self.assertLess(stats['native_window_peak_sessions'], len(fixture.calendar))

    def test_invalid_reuse_budgets_fail_before_native_or_data_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = MemoryFeatureInputs(root)
            prepared = prepared_inputs(fixture, root)
            for budget in (True, False, -1, 1.0, '0', LIMIT+1):
                with self.subTest(budget=budget), \
                     patch.object(fixture.data, 'read', side_effect=AssertionError('invalid budget read Data')), \
                     patch('axiom_research.qlib_adapter.QlibView.read', side_effect=AssertionError('invalid budget read native')):
                    with self.assertRaisesRegex(ValueError, 'reuse'):
                        next(iter_matrix_feature_days(fixture.data,
                            **arguments(fixture, prepared, reuse_budget_bytes=budget)))

    def test_tighter_budget_flushes_batches_and_source_frames_are_released_before_core(self):
        from axiom_research.stock_ml import _adjust_feature
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = MemoryFeatureInputs(root)
            prepared = prepared_inputs(fixture, root)
            source_frames = []
            original_read, original_members = fixture.data.read, fixture.data.members

            def read(**kwargs):
                value = original_read(**kwargs)
                source_frames.append(weakref.ref(value.frame))
                return value

            def members(**kwargs):
                value = original_members(**kwargs)
                source_frames.append(weakref.ref(value.frame))
                return value

            def adjust(prices, factors, *args, **kwargs):
                source_frames.extend([weakref.ref(prices.frame), weakref.ref(factors.frame)])
                result = _adjust_feature(prices, factors, *args, **kwargs)
                source_frames.append(weakref.ref(result[0].frame))
                return result

            def batch(requests, **kwargs):
                self.assertTrue(all(reference() is None for reference in source_frames))
                return execute_feature_plan_batch(requests, **kwargs)

            stats = {}
            kwargs = arguments(fixture, prepared, stats=stats, reuse_budget_bytes=0)
            kwargs['maximum_resident_bytes'] = 12*1024**2
            pressure = [True]
            kwargs['caller_retained_bytes'] = lambda: 8*1024**2 if pressure[0] and stats.get('pending_view_peak_bytes', 0) else 0
            kwargs['progress'] = lambda update: pressure.__setitem__(0, False)
            with patch.object(fixture.data, 'read', side_effect=read), \
                 patch.object(fixture.data, 'members', side_effect=members), \
                 patch('axiom_research.stock_ml._adjust_feature', new=adjust), \
                 patch('axiom_research.qlib_adapter.QlibView.read',
                       lambda view, **kwargs: fixture.native_read(view, **kwargs)), \
                 patch('axiom_engine.core.execute_feature_plan_batch', side_effect=batch) as calls:
                actual = list(iter_matrix_feature_days(fixture.data, **kwargs))
            self.assertEqual([proof['session'] for _, proof in actual], fixture.outputs)
            self.assertEqual(sum(len(call.args[0]) for call in calls.call_args_list), 5)
            self.assertTrue(all(len(call.args[0]) < 5 for call in calls.call_args_list))
            self.assertGreater(stats['batch_budget_flushes'], 0)
            self.assertGreater(stats['core_batch_calls'], 1)
            self.assertLessEqual(stats['combined_working_graph_peak_bytes'], kwargs['maximum_resident_bytes'])

    def test_single_view_too_large_is_rejected_before_any_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = MemoryFeatureInputs(root)
            prepared = prepared_inputs(fixture, root)
            kwargs = arguments(fixture, prepared, reuse_budget_bytes=0)
            kwargs['maximum_resident_bytes'] = 1
            with patch.object(fixture.data, 'read', side_effect=AssertionError('oversized view read Data')), \
                 patch('axiom_research.qlib_adapter.QlibView.read', side_effect=AssertionError('oversized view read native')):
                with self.assertRaisesRegex(ValueError, 'one native Feature window exceeds resident budget'):
                    next(iter_matrix_feature_days(fixture.data, **kwargs))

    def test_pending_frames_and_caller_growth_are_charged_during_yield(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = MemoryFeatureInputs(root)
            prepared = prepared_inputs(fixture, root)
            stats, caller = {}, [0]
            with patch('axiom_research.qlib_adapter.QlibView.read',
                       lambda view, **kwargs: fixture.native_read(view, **kwargs)), \
                 patch('axiom_engine.core.execute_feature_plan_batch', wraps=execute_feature_plan_batch) as calls:
                iterator = iter_matrix_feature_days(fixture.data,
                    **arguments(fixture, prepared, stats=stats, caller_retained_bytes=lambda: caller[0]))
                first = next(iterator)
                self.assertEqual(first[1]['session'], fixture.outputs[0])
                self.assertEqual(len(calls.call_args.args[0]), 5)
                self.assertEqual(stats['core_daily_frames'], 5)
                self.assertGreater(stats['yield_live_bytes'], 0)
                caller[0] = LIMIT
                with self.assertRaisesRegex(ValueError, 'combined producer/caller resident budget exceeded'):
                    next(iterator)
                self.assertEqual(calls.call_count, 1)
                self.assertEqual(stats['yield_live_bytes'], 0)
                self.assertGreater(stats['combined_working_graph_peak_bytes'], LIMIT)

    def test_large_source_metadata_is_rejected_before_its_wire_copy(self):
        from axiom_data import DataBatch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = MemoryFeatureInputs(root)
            prepared = prepared_inputs(fixture, root)
            original_read = fixture.data.read
            large_ids = set()

            def read(**kwargs):
                value = original_read(**kwargs)
                result = DataBatch(value.frame, value.field_meta,
                    {**value.context, 'large_source_proof': 'x'*(20*1024**2)})
                large_ids.add(id(result))
                return result

            original_json = DataBatch.to_json

            def to_json(value):
                if id(value) in large_ids:
                    raise AssertionError('oversized metadata serialized')
                return original_json(value)

            with patch.object(fixture.data, 'read', side_effect=read), \
                 patch('axiom_data.protocols.DataBatch.to_json', new=to_json), \
                 patch('axiom_research.qlib_adapter.QlibView.read',
                       lambda view, **kwargs: fixture.native_read(view, **kwargs)), \
                 patch('axiom_engine.core.execute_feature_plan_batch', side_effect=AssertionError('oversized metadata called Core')):
                with self.assertRaisesRegex(ValueError, 'Reader projection working set exceeds resident budget'):
                    next(iter_matrix_feature_days(fixture.data,
                        **arguments(fixture, prepared, reuse_budget_bytes=0)))

    def test_metadata_alias_growth_at_next_data_call_invalidates_local_size_charge(self):
        from axiom_data import DataBatch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = MemoryFeatureInputs(root)
            prepared = prepared_inputs(fixture, root)
            original_read, original_json = fixture.data.read, DataBatch.to_json
            prices, forbidden = [], set()
            def read(**kwargs):
                value = original_read(**kwargs)
                if kwargs['query'].domain == 'market_daily':
                    prices.append(value)
                else:
                    prices[-1].field_meta['alias_growth'] = 'x'*LIMIT
                    forbidden.add(id(value))
                return value
            def to_json(value):
                if id(value) in forbidden:
                    raise AssertionError('factor serialized before rechecking old price metadata')
                return original_json(value)
            with patch.object(fixture.data,'read',side_effect=read), \
                 patch.object(DataBatch,'to_json',new=to_json), \
                 patch('axiom_research.qlib_adapter.QlibView.read',
                       lambda view, **kwargs: fixture.native_read(view, **kwargs)), \
                 patch('axiom_engine.core.execute_feature_plan_batch',side_effect=AssertionError('Core called')):
                with self.assertRaisesRegex(ValueError,'Reader projection working set exceeds resident budget'):
                    next(iter_matrix_feature_days(fixture.data,
                        **arguments(fixture,prepared,reuse_budget_bytes=0)))


class CoreBudgetScopeTargetTests(unittest.TestCase):
    """Dimension admission and original Core scope witnesses, within 32 MiB.

    The 314-security gate allocates only symbols and a one-security plan header.
    It stops before native/Data reads; it does not admit real Case C inputs.
    """

    def test_314_union_dimension_gate_reaches_read_boundary_without_full_data(self):
        from axiom_research.feature_catalog import load_feature_catalog
        from axiom_research.stock_matrix_feature_producer import _core_budget_plan, _core_preflight_bounds, _owned_bytes
        catalog = load_feature_catalog()
        selection = [item.to_dict() for item in catalog.default_selection]
        chosen = catalog.select(selection)
        plan = _core_budget_plan(catalog, selection)
        workspace, output, dimensions = _core_preflight_bounds(plan, 21, 314)
        self.assertEqual(len(plan['nodes']), 33)
        self.assertEqual(dimensions['input_cells'], 32970)
        self.assertEqual(dimensions['node_cells'], 35482)
        historical = {'_zero_price', '_positive_close_flag', '_null_price', '_positive_close'}
        self.assertEqual({name for name, count in dimensions['nodes'].items() if count == 21*314}, historical)
        self.assertTrue(all(count == 314 for name, count in dimensions['nodes'].items() if name not in historical))
        self.assertLess(_owned_bytes(plan), 1024**2)
        symbols = ['SEC'+str(index).zfill(3) for index in range(314)]
        sessions = [(date(2026, 1, 1)+timedelta(days=index)).isoformat() for index in range(21)]
        config = {'symbols': symbols, 'read_sessions': sessions, 'feature_sessions': sessions[-1:],
            'cutoff_by_session': {day: day+'T20:30:00+08:00' for day in sessions}, 'feature_selection': selection}
        native = Mock()
        native.read.side_effect = RuntimeError('dimension gate reached native boundary')
        prepared = {'view': native, 'view_reference': {'instrument_map': {symbol: symbol for symbol in symbols}}}
        data, stats = Mock(), {}
        with self.assertRaisesRegex(RuntimeError, 'dimension gate reached native boundary'):
            next(iter_matrix_feature_days(data, config=config, catalog=catalog, chosen=chosen,
                qlib_inputs=prepared, history_sessions=21, output_block_sessions=64,
                maximum_resident_bytes=512*1024**2, stats=stats))
        native.read.assert_called_once()
        data.read.assert_not_called()
        data.members.assert_not_called()
        self.assertEqual(stats['core_preflight_input_cells'], 32970)
        self.assertEqual(stats['core_preflight_node_cells'], 35482)
        self.assertLess(stats['combined_working_graph_peak_bytes'], 512*1024**2)
        _BUDGET_ACCEPTANCE.append({'kind': 'DIMENSION_ONLY_BEFORE_NATIVE_READ', 'case_c_admitted': False,
            'symbol_count': 314, 'history_sessions': 21, 'selected_features': 6,
            'old_coarse_core_bytes': 21*314*(6+4*6)*3072,
            'input_cells': dimensions['input_cells'], 'derived_cells': dimensions['node_cells'],
            'workspace_reserve_bytes': workspace, 'output_reserve_bytes': output,
            'combined_preflight_charge_bytes': stats['combined_working_graph_peak_bytes'],
            'configured_budget_bytes': 512*1024**2, 'actual_template_owned_bytes': _owned_bytes(plan),
            'native_read_completed': False, 'data_read_calls': 0})

    def test_actual_daily_scope_counts_match_fixed_core_dependency_planner(self):
        from axiom_engine.core import validate_plan
        from axiom_engine.core.execution import _inputs, _reference_index, _required_cells
        from axiom_research.stock_matrix_feature_producer import _core_scope_counts, _core_view_bounds, _owned_bytes
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = MemoryFeatureInputs(root)
            prepared = prepared_inputs(fixture, root)
            config = deepcopy(fixture.config)
            config['feature_sessions'] = [fixture.outputs[0], fixture.outputs[-1]]
            captured, stats = [], {}

            def batch(requests, **kwargs):
                for request in requests:
                    plan, facts, context = request
                    p = validate_plan(plan, execution=True)
                    c, keys, outputs, rows, by_security, refs, events = _inputs(p, facts, context)
                    needed = _required_cells(p, outputs, by_security, refs, _reference_index(refs))
                    counts = _core_scope_counts(p, len(c['sessions']), len(fixture.symbols))
                    self.assertEqual(counts['nodes'], {name: len(scope) for name, scope in needed.items()})
                    self.assertEqual(counts['input_cells'], 315)
                    self.assertEqual(counts['node_cells'], 339)
                    workspace, output = _core_view_bounds(p, facts.to_dict(), c)
                    self.assertGreater(workspace, 3*_owned_bytes([p, facts.to_dict(), c]))
                    self.assertGreater(output, 0)
                    captured.append((request, counts, workspace, output))
                return execute_feature_plan_batch(requests, **kwargs)

            with patch('axiom_research.qlib_adapter.QlibView.read',
                       lambda view, **kwargs: fixture.native_read(view, **kwargs)), \
                 patch('axiom_engine.core.execute_feature_plan_batch', new=batch):
                actual = list(iter_matrix_feature_days(fixture.data, config=config, catalog=fixture.catalog,
                    chosen=fixture.chosen, qlib_inputs=prepared, history_sessions=fixture.history,
                    output_block_sessions=2, maximum_resident_bytes=LIMIT, reuse_budget_bytes=0, stats=stats))
            self.assertEqual(len(captured), 2)
            self.assertTrue(actual[0][0][2]['member'])
            self.assertFalse(actual[1][0][2]['member'])
            for (rows, proof), (request, counts, workspace, output) in zip(actual, captured):
                original = execute_feature_plan(*request)
                self.assertEqual(proof['core_frame_ref'], original.identity)
                self.assertEqual([row['values'] for row in rows], [row['values'] for row in original.to_dict()['rows']])
            self.assertLessEqual(stats['combined_working_graph_peak_bytes'], LIMIT)
            _BUDGET_ACCEPTANCE.append({'kind': 'SYNTHETIC_PUBLIC_READER_CORE_SCOPE_WITNESS',
                'case_c_admitted': False, 'symbol_count': 3, 'history_sessions': 21,
                'input_cells': 315, 'derived_cells': 339, 'original_core_scope_exact': True,
                'changed_membership_full_cohort_preserved': True,
                'workspace_reserve_bytes': max(item[2] for item in captured),
                'output_reserve_bytes': max(item[3] for item in captured),
                'combined_peak_charge_bytes': stats['combined_working_graph_peak_bytes']})

    def test_unsupported_memory_operator_is_rejected_instead_of_omitted(self):
        from axiom_research.feature_catalog import load_feature_catalog
        from axiom_research.stock_matrix_feature_producer import _core_budget_plan, _core_scope_counts
        catalog = load_feature_catalog()
        plan = _core_budget_plan(catalog, [item.to_dict() for item in catalog.default_selection])
        for operation in ('asof', 'unadmitted_op'):
            with self.subTest(operation=operation):
                changed = deepcopy(plan)
                changed['nodes'][0]['op'] = operation
                with self.assertRaisesRegex(ValueError, 'unsupported Core memory scope operator'):
                    _core_scope_counts(changed, 21, 314)

    def test_tight_combined_budget_flushes_before_read_and_rechecks_after_yield(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = MemoryFeatureInputs(root)
            prepared = prepared_inputs(fixture, root)
            stats, pressure = {}, [True]
            caller = lambda: 8*1024**2 if pressure[0] and stats.get('pending_view_peak_bytes', 0) else 0
            def progress(update):
                pressure[0] = False
            with patch('axiom_research.qlib_adapter.QlibView.read',
                       lambda view, **kwargs: fixture.native_read(view, **kwargs)), \
                 patch('axiom_engine.core.execute_feature_plan_batch', wraps=execute_feature_plan_batch) as calls:
                kwargs = arguments(fixture, prepared, stats=stats, reuse_budget_bytes=0,
                                   caller_retained_bytes=caller, progress=progress)
                kwargs['maximum_resident_bytes'] = 12*1024**2
                actual = list(iter_matrix_feature_days(fixture.data, **kwargs))
            self.assertGreater(stats['batch_budget_flushes'], 0)
            self.assertGreater(calls.call_count, 1)
            self.assertEqual(sum(len(call.args[0]) for call in calls.call_args_list), 5)
            self.assertEqual([proof['session'] for _, proof in actual], fixture.outputs)
            self.assertLessEqual(stats['combined_working_graph_peak_bytes'], 12*1024**2)
            self.assertEqual(stats['yield_live_bytes'], 0)
            _BUDGET_ACCEPTANCE.append({'kind': 'SYNTHETIC_CALLER_PRESSURE_FLUSH', 'case_c_admitted': False,
                'configured_budget_bytes': 12*1024**2, 'caller_pressure_bytes': 8*1024**2,
                'batch_view_counts': [len(call.args[0]) for call in calls.call_args_list],
                'budget_flushes': stats['batch_budget_flushes'],
                'combined_peak_charge_bytes': stats['combined_working_graph_peak_bytes']})


if __name__ == '__main__':
    outcome = unittest.main(exit=False)
    if outcome.result.wasSuccessful() and _ACCEPTANCE:
        destination = Path(__file__).resolve().parents[1]/'.artifacts'/'feature-batch'/'acceptance.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({'contract_version': 'research_feature_batch_synthetic_acceptance_v1',
            'core_source': '60df1fc952f8f0d50a2da5a88c1cf2305ee9c809',
            'data_source': 'e321a665', 'fixture': 'MemoryFeatureInputs:five_daily_views',
            'maximum_owned_graph_bytes': LIMIT, 'acceptance_kind': 'SYNTHETIC_PUBLIC_READER_EXPORT_CORE',
            'real_input_execution': False, 'rss_measured': False,
            'performance_claim': 'None; short reductions have no numeric helper reuse.',
            'variants': _ACCEPTANCE}, sort_keys=True, indent=2)+'\n')
        print('Saved synthetic acceptance:', destination)
    if outcome.result.wasSuccessful() and _BUDGET_ACCEPTANCE:
        destination = Path(__file__).resolve().parents[1]/'.artifacts'/'feature-batch'/'budget-target.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({'contract_version': 'research_feature_core_budget_scope_target_v1',
            'core_source': '60df1fc952f8f0d50a2da5a88c1cf2305ee9c809', 'data_source': 'e321a665',
            'producer_source_sha256': sha256((Path(__file__).resolve().parents[1]/'src'/'axiom_research'/
                'stock_matrix_feature_producer.py').read_bytes()).hexdigest(),
            'target_test_source_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
            'case_c_admitted': False, 'real_input_execution': False, 'rss_measured': False,
            'bound_kind': 'conservative owned graphs/workspaces; not RSS',
            'targets': _BUDGET_ACCEPTANCE}, sort_keys=True, indent=2)+'\n')
        print('Saved synthetic budget target:', destination)
    if not outcome.result.wasSuccessful():
        sys.exit(1)

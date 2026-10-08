"""One small real owner lease -> frozen consumer acceptance; synthetic sources.

Preparation uses artificial prices and a fixed fake backend. Inference admission
and evaluation prohibit Data, fitting, training matrices and all-fold scans.
"""
from contextlib import ExitStack, contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
import builtins
import json
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import numpy as np  # Load native extensions before the temporary Data module patch.

from axiom_research import (prepare_stock_ml_batch_inputs, load_stock_ml_batch_inputs,
    save_stock_signal_evaluation_inputs, evaluate_stock_signal_input_periods,
    load_stock_signal_evaluation, audit_stock_signal_evaluation)
from axiom_research.stock_artifacts import _read, digest, file_digest, write_json
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_batch import StockMLBatchInputs, _data
from axiom_research.stock_compact_store import OwnedStore
from axiom_research.stock_signal_evaluation_lease import _FoldLease
from axiom_research.stock_matrix_folds import admit_stock_signal_evaluation_fold
from axiom_research.stock_signal_evaluation import SPEC
from axiom_research import stock_signal_evaluation_projection as frozen
from axiom_engine.core import evaluate_signal_statistics
import test_stock_compact_v3 as storage_fixtures
import test_stock_matrix_prepare as matrix_fixtures
import test_stock_sequential_windows as window_fixtures


class CrossYearFeature(matrix_fixtures.PrepareFeatureFixture):
    def __init__(self, root):
        super().__init__(root)
        first = self.history-1+65
        start = date(2023, 12, 27)-timedelta(days=2*first)
        self.calendar = [(start+timedelta(days=2*i)).isoformat() for i in range(len(self.calendar))]
        self.days = self.calendar[self.history-1:]
        self.universe = [f'S{i:03}' for i in range(24)]
        self.spec.update(calendar=self.calendar, universe=self.universe, read_sessions=self.calendar,
            feature_sessions=self.days, cutoff_by_session={d: d+'T20:30:00+08:00' for d in self.calendar})
        scope = _read(self.spec['scope']['path'])
        scope['result'].update(read_sessions=self.calendar, read_symbols=self.universe)
        scope['result_ref'] = digest(scope['result'])
        scope = seal({k: v for k, v in scope.items() if k != 'scope_bundle_ref'}, 'scope_bundle_ref')
        write_json(self.spec['scope']['path'], scope)
        self.spec['scope'].update(file_digest=file_digest(self.spec['scope']['path']), scope_bundle_ref=scope['scope_bundle_ref'])

    def day(self, day):
        rows, proof = super().day(day)
        proof['core_plan']['reference_members'][day] = self.universe[:-1]
        source_refs = [proof['core_frame_ref'], digest(proof['core_plan'])]
        for i, row in enumerate(rows):
            row.update(member=i < 23, values=[float(i//3+j) for j in range(6)], source_refs=source_refs)
            if day == self.folds()[0]['fit_session'] and i == 2:
                row.update(values=[None]*6, validity=[False]*6, availability=[None]*6,
                    reasons=[['SYNTHETIC_INVALID']]*6)
        return rows, proof


@contextmanager
def readonly_execution(*, allow_statistics):
    with ExitStack() as stack:
        original_import = builtins.__import__
        def imports(name, *args, **kwargs):
            blocked = ('axiom_data', 'qlib', 'lightgbm', 'pandas', 'sklearn')
            if not allow_statistics: blocked += ('axiom_engine',)
            if name.startswith(blocked): raise AssertionError('readonly execution imported '+name)
            return original_import(name, *args, **kwargs)
        stack.enter_context(patch('builtins.__import__', side_effect=imports))
        for target in ('axiom_research.stock_compact_batch.training_matrix',
                'axiom_research.stock_compact_batch.CompactState.verify_all',
                'axiom_research.stock_training.fit_predict_stock_model',
                'test_stock_matrix_prepare.PublicDataFixture.read',
                'test_stock_matrix_prepare.PublicDataFixture.adjust'):
            stack.enter_context(patch(target, side_effect=AssertionError('readonly execution used '+target)))
        yield


class RealOwnerFrozenVerticalTests(unittest.TestCase):
    def test_exact_masks_years_hit_and_mutation_with_real_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            print('VERTICAL_SYNTHETIC_SETUP_STARTED', flush=True)
            feature, table = storage_fixtures.CompactV3Tests().fixture(root, feature_class=CrossYearFeature)
            data = matrix_fixtures.PublicDataFixture(feature.spec)
            module = types.ModuleType('axiom_data'); module.QuerySpec = matrix_fixtures.Query
            module.adjust_prices = data.adjust
            with patch.dict(sys.modules, {'axiom_data': module}):
                manifest = prepare_stock_ml_batch_inputs(data, feature_inputs=table, fold_specs=feature.folds()[:2],
                    destination=root/'prepared', preparation_options={'row_block_sessions': 5, 'column_block': 32,
                        'maximum_resident_bytes': 64*1024**2, 'normalization_backend': 'core_cs_batch_v1'})
            fake_backend_calls = []
            def backend(X, y, P, **kwargs):
                fake_backend_calls.append(1)
                class Model:
                    def save_model(self, path): Path(path).write_text('synthetic saved booster; no real fitting')
                kwargs['metrics'].update(train_calls=1, predict_calls=1)
                return Model(), [float(row[0]) for row in P]
            with load_stock_ml_batch_inputs(manifest, residency='sequential') as batch:
                runs = [window_fixtures.SequentialWindowTests().build(feature, fold, batch, root/'folds', fit=backend)
                    for fold in manifest['folds']]
            descriptors = [{'path': str(run.path/'predictions.json'), 'file_digest': file_digest(run.path/'predictions.json'),
                'signal_run_ref': _read(run.path/'predictions.json')['signal_run_ref']} for run in runs]
            signals = {'full': descriptors, 'partial': descriptors[:1]}
            predictions = {name: {(r['security_id'], r['session']): r for desc in inputs
                for r in _read(desc['path'])['rows']} for name, inputs in signals.items()}
            days = sorted({r['session'] for r in predictions['full'].values()})
            scope = {'calendar': feature.calendar, 'sessions': days, 'universe': feature.universe,
                'evaluation_cutoff': manifest['folds'][0]['fold_spec']['evaluation_cutoff']}
            self.assertEqual([day[:4] for day in (days[0], days[-1])], ['2023', '2024'])
            leased_rows, borrowed = [], []
            lease_checks, batch_checks, probes = [], [], []
            original_lease_check = _FoldLease._check_sources
            original_store_check = OwnedStore.check
            original_batch_check = StockMLBatchInputs._check_sources
            def store_check(store):
                if probes: probes[-1].append(store)
                return original_store_check(store)
            def batch_check(batch):
                started = time.perf_counter()
                try: return original_batch_check(batch)
                finally: batch_checks.append(time.perf_counter()-started)
            def lease_check(lease):
                stores = []; before = len(batch_checks)
                expected = [lease._ingress, lease._state.store]
                feature_store = lease._state.store.check_hook.__self__
                watched = [*expected, feature_store]
                checks_before = [(store.metrics['lifecycle_check_calls'], store.metrics['source_stat_calls'])
                    for store in watched]
                probes.append(stores)
                try: original_lease_check(lease)
                finally: probes.pop()
                self.assertEqual(stores, expected)
                self.assertEqual(len(batch_checks)-before, 1)
                # Feature's callback was bound when the batch was loaded;
                # retained metrics include it even when method spies do not.
                for store, (checks, stats) in zip(watched, checks_before):
                    self.assertEqual(store.metrics['lifecycle_check_calls']-checks, 1)
                    self.assertEqual(store.metrics['source_stat_calls']-stats, len(store.marks))
                lease_checks.append({'direct_store_calls': len(stores), 'feature_hook_checks': 1,
                    'stat_calls': sum(len(store.marks) for store in watched),
                    'saved_duplicate_stat_calls': sum(len(store.marks) for store in watched[1:])})
            @contextmanager
            def observed(descriptor, *, batch):
                with admit_stock_signal_evaluation_fold(descriptor, batch=batch) as lease:
                    self.assertFalse(any(hasattr(lease, name) for name in ('X', 'y', 'P', 'training_keys', 'booster')))
                    targets = lease.evaluation_targets
                    original_days = sorted(lease.documents['fold.json']['definition']['fold_spec']['inference_cutoff_by_session'])
                    self.assertEqual([day for target in targets for day in target['header']['definition']['sessions']], original_days)
                    count = sum(len(target['rows']) for target in targets)
                    self.assertEqual(count, len(original_days)*24); leased_rows.append(count)
                    borrowed.extend(target['rows'] for target in targets)
                    yield lease
            print('VERTICAL_REAL_OWNER_FREEZE_STARTED', flush=True)
            with readonly_execution(allow_statistics=False), load_stock_ml_batch_inputs(manifest, residency='sequential') as batch:
                state = _data(batch)['matrix_state']
                with patch('axiom_research.stock_matrix_folds.admit_stock_signal_evaluation_fold', observed), \
                        patch.object(_FoldLease, '_check_sources', lease_check), \
                        patch.object(OwnedStore, 'check', store_check), \
                        patch.object(StockMLBatchInputs, '_check_sources', batch_check):
                    ref = save_stock_signal_evaluation_inputs(signals, scope=scope, destination=root/'inputs', batch=batch)
                self.assertEqual((state.active, state.store.lease_bytes), (0, 0))
            self.assertEqual(leased_rows, [72, 72, 72])
            self.assertEqual(len(lease_checks), 2*len(leased_rows))
            self.assertTrue(all(rows._rows is None for rows in borrowed))
            root_wire = _read(ref.uri)
            self.assertIn('stock_signal_evaluation_lease.py', root_wire['admission_receipt']['validation_sources'])
            reads = []; original_read = frozen._read_checked
            def read(path, *args, **kwargs):
                reads.append(Path(path)); return original_read(path, *args, **kwargs)
            with readonly_execution(allow_statistics=True), patch.object(frozen, '_read_checked', side_effect=read):
                result = evaluate_stock_signal_input_periods(ref, scope=scope, destination=root/'reports')
            self.assertTrue(all(reads.count(path) == 1 for path in [Path(ref.uri),
                *(Path(ref.uri).parent/(day+'.json') for day in days)]))
            self.assertEqual(list(result['by_year']), ['2023', '2024'])
            signal_keys = {name: digest({'input_signal_refs': [desc['signal_run_ref'] for desc in inputs]})
                for name, inputs in signals.items()}
            expected_keys = {}
            for label, reports in [('all', result['all']), *result['by_year'].items()]:
                chosen_days = days if label == 'all' else [day for day in days if day[:4] == label]
                for native in (False, True):
                    pairs = []
                    for name in signals:
                        kept = []
                        for day in chosen_days:
                            for security in feature.universe:
                                key = security, day; row = predictions[name].get(key)
                                eligible = row is not None and row['valid'] and row['member']
                                if not native:
                                    eligible = eligible and all(key in rows and rows[key]['valid'] and rows[key]['member']
                                        for rows in predictions.values())
                                if not eligible: continue
                                i = feature.universe.index(security); pos = feature.calendar.index(day)
                                raw_return = (11.0+i*i*.4+(pos+5)*.01)/(10.0+(pos+1)*.01)-1
                                pairs.append({'signal_key': signal_keys[name], 'session': day, 'security_id': security,
                                    'score': row['score'], 'outcome': raw_return}); kept.append([security, day])
                        if label == 'all' and native: expected_keys[name] = kept
                    expected = evaluate_signal_statistics({'contract_version': 'signal_statistics_input_v1',
                        'sessions': chosen_days, 'signal_keys': list(signal_keys.values()), 'pairs': pairs}, spec=SPEC).to_dict()
                    for name, saved in reports.items():
                        report = saved.to_dict(); stats = report['coverage']['native'] if native else report
                        self.assertEqual(stats['series'], [row for row in expected['series'] if row['signal_key'] == signal_keys[name]])
                        self.assertEqual(stats['summary'], next(row for row in expected['summary'] if row['signal_key'] == signal_keys[name]))
                        self.assertEqual((stats['statistics_input_ref'], stats['statistics_ref']), (expected['input_ref'], expected['statistics_ref']))
                        self.assertEqual(datetime.fromisoformat(report['scope']['evaluation_cutoff'].replace('Z', '+00:00')),
                            datetime.fromisoformat(scope['evaluation_cutoff'].replace('Z', '+00:00')))
                        self.assertEqual(report['spec'], SPEC)
            admitted = frozen._load_inputs(ref, scope)[2]
            self.assertEqual(admitted['mask']['native_keys'], expected_keys)
            self.assertEqual(admitted['mask']['common_keys'], expected_keys['partial'])
            self.assertEqual((len(expected_keys['full']), len(expected_keys['partial'])), (137, 68))
            self.assertEqual(admitted['native_coverage']['partial']['excluded_counts'],
                {'NOT_MEMBER': 6, 'SIGNAL_INVALID': 1, 'SIGNAL_MISSING': 69})
            with readonly_execution(allow_statistics=True), \
                 patch('axiom_engine.core.evaluate_signal_statistics', side_effect=AssertionError('HIT recomputed Core')):
                hit = evaluate_stock_signal_input_periods(ref, scope=scope, destination=root/'reports')
            self.assertTrue(all(saved.reused for group in [hit['all'], *hit['by_year'].values()] for saved in group.values()))
            report = result['all']['full']; self.assertEqual(load_stock_signal_evaluation(report.path).to_dict(), report.to_dict())
            with readonly_execution(allow_statistics=False): audit_stock_signal_evaluation(report.path)
            # Same real hook; corrupt the current original buffer after owner
            # admission, before the consumer begins copying its detached rows.
            buffer = None; original_bytes = None
            @contextmanager
            def mutate(descriptor, *, batch):
                nonlocal buffer, original_bytes
                with admit_stock_signal_evaluation_fold(descriptor, batch=batch) as lease:
                    buffer = Path(lease.evaluation_targets[0]['header']['buffers']['values']['path'])
                    original_bytes = buffer.read_bytes(); buffer.write_bytes(b'X'*len(original_bytes))
                    yield lease
            with readonly_execution(allow_statistics=False), load_stock_ml_batch_inputs(manifest, residency='sequential') as batch:
                state = _data(batch)['matrix_state']
                with patch('axiom_research.stock_matrix_folds.admit_stock_signal_evaluation_fold', mutate):
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        save_stock_signal_evaluation_inputs(signals, scope=scope, destination=root/'failed', batch=batch)
                self.assertEqual((state.active, state.store.lease_bytes), (0, 0))
            self.assertFalse((root/'failed').exists())
            self.assertEqual(load_stock_signal_evaluation(report.path).to_dict(), report.to_dict())
            with readonly_execution(allow_statistics=False):
                with self.assertRaises(ValueError): audit_stock_signal_evaluation(report.path)
            buffer.write_bytes(original_bytes)
            with readonly_execution(allow_statistics=False): audit_stock_signal_evaluation(report.path)
            print(json.dumps({'real_owner_vertical': {'owner_source': 'cf14e76908e68f10fb680821fd283673167b5da1',
                'folds': 2, 'signals': 2, 'securities': 24, 'sessions': len(days), 'years': ['2023', '2024'],
                'leased_target_rows': leased_rows, 'native_pair_counts': [137, 68], 'common_pair_count': 68,
                'freeze_lease_checks': lease_checks, 'freeze_batch_stat_check_calls': len(batch_checks),
                'freeze_batch_stat_check_seconds': sum(batch_checks),
                'synthetic_backend_setup_calls': len(fake_backend_calls), 'synthetic_price_setup_queries': len(data.queries),
                'real_data_reads': 0, 'real_fit_calls': 0, 'account_calls': 0}}))

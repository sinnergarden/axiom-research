"""Small four-fold saved consumers over the fixed actual matrix owner and Core.

Only source Data and the fixture model backend are synthetic. Frozen admission,
fold loading, leases, provenance, masks and statistics use their real interfaces.
"""
from copy import deepcopy
from pathlib import Path
import builtins
import gc
import os
import subprocess
import sys
import tempfile
import types
import unittest
import weakref
from unittest.mock import patch

from axiom_engine.core import evaluate_signal_statistics
from axiom_research import (prepare_stock_ml_batch_inputs, load_stock_ml_batch_inputs,
    build_stock_ml_fold_from_saved_inputs, save_stock_signal_evaluation_inputs,
    evaluate_stock_signal_inputs, load_stock_signal_evaluation, audit_stock_signal_evaluation)
from axiom_research.labels import build_forward_labels
from axiom_research.stock_artifacts import _read, digest, file_digest, write_json
from axiom_research.stock_batch import _data
from axiom_research import stock_signal_evaluation_projection as frozen
from axiom_research.stock_fold_artifacts import load_stock_ml_fold
from axiom_research.stock_signal_evaluation_projection import _load_inputs
from test_stock_matrix_prepare import PrepareFeatureFixture, PublicDataFixture, Query, Batch
from test_stock_folds import backend


class ExactData(PublicDataFixture):
    def adjust(self, *args, **kwargs):
        adjusted = super().adjust(*args, **kwargs)
        wire = adjusted.to_json()
        wire['context']['derivation']['factor_domain'] = 'adjustment_factors'
        return Batch(wire)


def saved_four_folds(root):
    fixture = PrepareFeatureFixture(root)
    with patch('axiom_research.feature_catalog.load_feature_catalog', return_value=fixture.catalog):
        saved = fixture.matrix()
    data = ExactData(fixture.spec)
    module = types.ModuleType('axiom_data'); module.QuerySpec = Query; module.adjust_prices = data.adjust
    with patch.dict(sys.modules, {'axiom_data': module}), \
         patch('axiom_research.stock_ml._implementation', return_value=fixture.implementation), \
         patch('axiom_research.stock_ml._environment', return_value=fixture.environment):
        manifest = prepare_stock_ml_batch_inputs(data, feature_inputs=saved, fold_specs=fixture.folds(),
            destination=root/'prepared', preparation_options={'row_block_sessions': 32, 'column_block': 32,
                'maximum_resident_bytes': 64*1024**2, 'normalization_backend': 'core_cs_batch_v1'})
    saved.close()
    return fixture, manifest, data


def raw_override(root, fixture, data, scope, horizon=5):
    calendar = scope['calendar']; start = calendar.index(scope['sessions'][0])+1
    end = calendar.index(scope['sessions'][-1])+horizon
    sessions = tuple(calendar[start:end+1]); anchor = sessions[-1]
    common = dict(symbols=tuple(scope['universe']), sessions=sessions,
        pit_policy=fixture.pit, cutoff_by_session={day: scope['evaluation_cutoff'] for day in sessions},
        purpose='label_outcomes')
    prices = data.read(snapshot=fixture.snapshot, query=Query(domain='market_daily', fields=('open', 'close'), **common))
    factors = data.read(snapshot=fixture.snapshot, query=Query(domain='adjustment_factors', fields=('factor',), **common))
    adjusted = data.adjust(prices, factors, fields=('open', 'close'), anchor_session=anchor,
        decision_session=anchor, factor_field='factor')
    raw = build_forward_labels(adjusted, calendar=calendar, feature_sessions=scope['sessions'], horizon_sessions=horizon)
    path = root/f'raw-t{horizon}.json'; write_json(path, raw)
    return {'path': str(path), 'file_digest': file_digest(path), 'label_ref': raw['label_ref']}


class MatrixFrozenRealOwnerTests(unittest.TestCase):
    def test_four_saved_folds_oos_admission_override_core_hit_and_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture, manifest, data = saved_four_folds(root)
            with load_stock_ml_batch_inputs(manifest) as batch:
                signals, original_labels, members = [], {}, {}
                for fold in manifest['folds']:
                    with patch('axiom_research.feature_catalog.load_feature_catalog', return_value=fixture.catalog), \
                         patch('axiom_research.stock_ml._implementation', return_value=fixture.implementation), \
                         patch('axiom_research.stock_ml._environment', return_value=fixture.environment), \
                         patch('axiom_research.stock_training.fit_predict_stock_model', side_effect=backend):
                        run = build_stock_ml_fold_from_saved_inputs(fold['input_manifest'], fold_spec=fold['fold_spec'],
                            destination=root/'folds', batch=batch)
                    prediction = run.predictions()
                    signals.append({'path': str(run.path/'predictions.json'),
                        'file_digest': file_digest(run.path/'predictions.json'), 'signal_run_ref': prediction['signal_run_ref']})
                    with batch._project_evaluation(fold['input_manifest'], fold['fold_spec']) as view:
                        for row in view.evaluation['rows']:
                            original_labels[row['security_id'], row['feature_session']] = deepcopy(row)
                        for row in view.features['rows']:
                            members[row['security_id'], row['session']] = row['member']
                sessions = sorted({key[1] for key in original_labels})
                scope = {'sessions': sessions, 'universe': fixture.universe, 'calendar': fixture.calendar,
                    'evaluation_cutoff': manifest['folds'][0]['fold_spec']['evaluation_cutoff']}
                shared = raw_override(root, fixture, data, scope)
                independent = raw_override(root, fixture, data, scope, horizon=6)
                inputs = {'a': signals, 'b': signals[:3]}
                admitted = batch.metrics
                source_table = batch._evaluation_source_records()
                self.assertIs(source_table, batch._evaluation_source_records())
                self.assertIsInstance(source_table, tuple)
                state = _data(batch)['matrix_state']
                class WeakDict(dict): pass
                class WeakList(list): pass
                tracked, checked = [], []
                checked_read = frozen._read_checked
                def read(path, *args, **kwargs):
                    wire, ref = checked_read(path, *args, **kwargs)
                    if Path(path).name == 'feature-slice.json':
                        wire = WeakDict(wire); tracked.append(weakref.ref(wire)); wide_rows = []
                        for original in wire['rows']:
                            row = WeakDict(original); tracked.append(weakref.ref(row))
                            for field in ('values', 'validity', 'availability', 'reasons'):
                                row[field] = WeakList(row[field]); tracked.append(weakref.ref(row[field]))
                            wide_rows.append(row)
                        wire['rows'] = wide_rows
                    return wire, ref
                def load(path, *, batch):
                    if tracked:
                        gc.collect()
                        self.assertTrue(all(ref() is None for ref in tracked), 'previous actual fold retained wide features')
                        checked.append(str(path))
                    return load_stock_ml_fold(path, batch=batch)
                with patch.object(type(batch), '_matrix_project', side_effect=AssertionError('training projection')), \
                     patch('axiom_research.stock_matrix_reader._stream_ref', side_effect=AssertionError('training ref walk')), \
                     patch('axiom_research.stock_training.fit_predict_stock_model', side_effect=AssertionError('model executed')), \
                     patch('axiom_engine.core.evaluate_signal_statistics', side_effect=AssertionError('freeze statistics')):
                    with patch.object(frozen, '_read_checked', new=read), \
                         patch('axiom_research.stock_fold_artifacts.load_stock_ml_fold', new=load):
                        default = save_stock_signal_evaluation_inputs(inputs, batch=batch, scope=scope, destination=root/'default')
                    gc.collect(); self.assertTrue(all(ref() is None for ref in tracked))
                    self.assertEqual(len(checked), 6)
                    override = save_stock_signal_evaluation_inputs(inputs, batch=batch, raw_label_input=shared,
                        scope=scope, destination=root/'override')
                    t6 = save_stock_signal_evaluation_inputs(inputs, batch=batch, raw_label_input=independent,
                        scope=scope, destination=root/'t6')
                self.assertEqual(state.store.borrowers, 0); self.assertEqual(state.store.lease_bytes, 0)
                for counter in ('file_hash_calls', 'json_decode_calls', 'common_key_index_builds'):
                    self.assertEqual(batch.metrics[counter], admitted[counter], counter)
                _, frozen_root, selected = _load_inputs(default, scope)
                for pair in selected['native_input']['pairs']:
                    key = pair['security_id'], pair['session']
                    self.assertTrue(members[key]); self.assertTrue(original_labels[key]['valid'])
                    self.assertEqual(pair['outcome'], original_labels[key]['return'])
                self.assertGreater(len(selected['mask']['native_keys']['a']), len(selected['mask']['common_keys']))
                self.assertEqual(selected['mask']['native_keys']['b'], selected['mask']['common_keys'])
                self.assertEqual(frozen_root['raw_metadata']['mode'], 'fold_targets')
                self.assertEqual(len(frozen_root['signal_refs']['a']), 4)
                with patch('axiom_engine.core.evaluate_signal_statistics', wraps=evaluate_signal_statistics) as core:
                    reports = evaluate_stock_signal_inputs(default, scope=scope, destination=root/'reports')
                    overrides = evaluate_stock_signal_inputs(override, scope=scope, destination=root/'reports-override')
                    self.assertEqual(core.call_count, 4)
                    self.assertEqual(core.call_args_list[0].args, core.call_args_list[2].args)
                    self.assertEqual(core.call_args_list[1].args, core.call_args_list[3].args)
                for name in inputs:
                    self.assertEqual(reports[name].to_dict()['series'], overrides[name].to_dict()['series'])
                    self.assertEqual(reports[name].to_dict()['contract_version'], 'stock_signal_evidence_v4')
                self.assertEqual(_read(t6.uri)['raw_metadata']['label_spec']['horizon_sessions'], 6)
                self.assertEqual(_read(t6.uri)['signal_metadata']['a'][0]['model']['target_semantics'],
                    _read(default.uri)['signal_metadata']['a'][0]['model']['target_semantics'])
            real = builtins.__import__
            def blocked(name, *args, **kwargs):
                if name.startswith(('axiom_data', 'axiom_engine', 'qlib', 'lightgbm', 'pandas')):
                    raise AssertionError('ordinary HIT imported runtime '+name)
                return real(name, *args, **kwargs)
            with patch('builtins.__import__', side_effect=blocked), \
                 patch('axiom_research.stock_batch.load_stock_ml_batch_inputs', side_effect=AssertionError('ancestor load')):
                hit = evaluate_stock_signal_inputs(default, scope=scope, destination=root/'reports')
                self.assertTrue(all(report.reused for report in hit.values()))
                self.assertEqual(load_stock_signal_evaluation(hit['a'].path).to_dict(), reports['a'].to_dict())
            script = '''import builtins,sys
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0] in {'axiom_engine','axiom_data','qlib','lightgbm','numpy','pandas','pyarrow'}:
        raise AssertionError('runtime import '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guard
from axiom_research import load_stock_signal_evaluation
print(load_stock_signal_evaluation(sys.argv[1]).identity)
'''
            fresh = subprocess.run([sys.executable, '-B', '-c', script, str(reports['a'].path)],
                env=os.environ, capture_output=True, text=True, check=True, timeout=15)
            self.assertEqual(fresh.stdout.strip(), reports['a'].identity)
            audit_stock_signal_evaluation(reports['a'].path)
            (Path(signals[0]['path']).parent/'booster.txt').write_text('ancestor drift')
            self.assertEqual(load_stock_signal_evaluation(reports['a'].path).identity, reports['a'].identity)
            with self.assertRaises(ValueError):
                audit_stock_signal_evaluation(reports['a'].path)

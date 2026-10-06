"""Synthetic batch acceptance: exact keyed matrices, closure and read reuse."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import builtins
import gc
import os
import subprocess
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch

from axiom_research import (build_stock_ml_fold_from_saved_inputs, load_stock_ml_fold,
                            load_stock_ml_batch_inputs, predict_stock_model)
from axiom_research.stock_artifacts import digest, file_digest, write_json, _read
from axiom_research.stock_fold_inputs import seal, project_saved_fold, read_parent
from axiom_research.stock_label_normalization import normalize_forward_labels
from test_stock_folds import fixture, backend


def batch_fixture(root, *, two_year=False, reverse=False):
    calendar = ([(date(2021, 1, 1)+timedelta(days=10*i+(i//7))).isoformat() for i in range(100)]
                if two_year else None)
    items = []
    for i, fit in enumerate((80, 84) if two_year else (65, 69)):
        folder = root/str(i); folder.mkdir()
        items.append(fixture(folder, fit_index=fit, calendar=calendar, two_year=two_year))
    parents = [_read(Path(m['feature_parents'][0]['features']['path'])) for m, _ in items]
    rows = {(r['security_id'], r['session']): r for p in parents for r in p['rows']}
    dates = sorted({k[1] for k in rows}); universe = items[0][0]['universe']
    for (s, d), r in rows.items():
        r['values'] = [float(dates.index(d)*10+universe.index(s)+1)]
    proof = [{'session': d, 'core_plan': {'synthetic_session': d}, 'core_frame_ref': digest(['frame', d])} for d in dates]
    feature = {k: v for k, v in parents[0].items() if k != 'feature_ref'}
    feature.update(rows=[rows[k] for k in sorted(rows, key=lambda k: (k[1], k[0]))], input_evidence_ref=digest(proof))
    if reverse:
        feature['rows'].reverse()
    feature = seal(feature, 'feature_ref')
    write_json(root/'shared-feature.json', feature); write_json(root/'shared-proof.json', proof)
    shared = {'features': {'path': str(root/'shared-feature.json'), 'file_digest': file_digest(root/'shared-feature.json'),
        'feature_ref': feature['feature_ref']}, 'input_evidence': {'path': str(root/'shared-proof.json'),
        'file_digest': file_digest(root/'shared-proof.json'), 'input_evidence_ref': digest(proof)}, 'sessions': dates}
    evaluations = [_read(m['evaluation_labels']['path']) for m, _ in items]
    eval_rows = {(r['security_id'], r['feature_session']): r for e in evaluations for r in e['rows']}
    evaluation = {k: v for k, v in evaluations[0].items() if k != 'label_ref'}
    evaluation['rows'] = list(eval_rows.values()); evaluation = seal(evaluation, 'label_ref')
    write_json(root/'shared-evaluation.json', evaluation)
    for m, s in items:
        m['scope'] = deepcopy(items[0][0]['scope']); m['feature_parents'] = [deepcopy(shared)]
        m['evaluation_labels'] = {'path': str(root/'shared-evaluation.json'),
            'file_digest': file_digest(root/'shared-evaluation.json'), 'label_ref': evaluation['label_ref']}
        desc = m['training_labels'][0]; raw = _read(desc['raw']['path'])
        if reverse:
            raw['rows'].reverse(); raw = seal({k: v for k, v in raw.items() if k != 'label_ref'}, 'label_ref')
            write_json(desc['raw']['path'], raw)
            desc['raw'].update(file_digest=file_digest(desc['raw']['path']), label_ref=raw['label_ref'])
        projected = {k: v for k, v in raw.items() if k not in ('rows', 'label_ref')}
        projected.update(rows=raw['rows'], parent_label_ref=raw['label_ref'],
                         feature_parent_ref=feature['feature_ref'], date_projection=desc['sessions'])
        norm = normalize_forward_labels(seal(projected, 'label_ref'), features=feature, cutoff=s['fit_cutoff'])
        if reverse:
            norm['rows'].reverse()
            norm = seal({k: v for k, v in norm.items() if k != 'label_ref'}, 'label_ref')
        write_json(desc['normalized']['path'], norm)
        desc['normalized'].update(file_digest=file_digest(desc['normalized']['path']), label_ref=norm['label_ref'])
    return {'contract_version': 'stock_ml_batch_inputs_v1',
            'folds': [{'input_manifest': m, 'fold_spec': s} for m, s in items]}


class StockBatchTests(unittest.TestCase):
    def build(self, item, destination, batch=None, capture=None):
        def fit(X, y, P, **kw):
            if capture is not None: capture.append((X.copy(), y.copy(), P.copy()))
            return backend(X, y, P, **kw)
        with patch('axiom_research.stock_training.fit_predict_stock_model', side_effect=fit), \
             patch('axiom_research.stock_folds._environment', return_value={'synthetic': True}), \
             patch('axiom_research.stock_folds._implementation', return_value={'synthetic': digest('code')}):
            stats = {}; run = build_stock_ml_fold_from_saved_inputs(item['input_manifest'], fold_spec=item['fold_spec'],
                       destination=destination, batch=batch, metrics=stats)
        return run, stats

    def test_unique_inputs_once_same_artifact_and_exact_hit_without_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); definition = batch_fixture(root)
            with patch('axiom_research.stock_batch.read_parent', wraps=read_parent) as reader, \
                 patch('axiom_research.stock_fold_inputs._read', wraps=_read) as common_reads:
                batch = load_stock_ml_batch_inputs(definition)
                calls = reader.call_count
                self.assertEqual(calls, 5)
                runs = [self.build(item, root/'batch', batch=batch)[0] for item in definition['folds']]
                self.assertEqual(reader.call_count, calls)
                paths = [str(call.args[0]) for call in common_reads.call_args_list]
                self.assertEqual(paths.count(str(root/'shared-proof.json')), 1)
                self.assertEqual(paths.count(str(root/'shared-feature.json')), 1)
            self.assertEqual(batch.metrics['shared_matrix_builds'], 1)
            self.assertEqual(batch.metrics['proof_reads'], 1)
            self.assertEqual(batch.metrics['fold_projection_calls'], 2)
            for item, run in zip(definition['folds'], runs):
                fresh, _ = self.build(item, root/'fresh')
                self.assertEqual(fresh.to_dict(), run.to_dict())
                self.assertEqual(fresh.predictions(), run.predictions())
                self.assertEqual(load_stock_ml_fold(run.path).identity, run.identity)
                hit, metrics = self.build(item, root/'batch', batch=batch)
                self.assertTrue(hit.reused)
                self.assertEqual((metrics['train_calls'], metrics['predict_calls'], metrics['core_calls']), (0, 0, 0))
            batch.close()
            with self.assertRaisesRegex(ValueError, 'closed'): batch.metrics

    def test_reordered_feature_and_label_rows_keep_joint_keyed_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a = root/'a'; b = root/'b'; a.mkdir(); b.mkdir()
            ordered = batch_fixture(a); reversed_ = batch_fixture(b, reverse=True)
            captures = []
            for definition in (ordered, reversed_):
                with load_stock_ml_batch_inputs(definition) as batch:
                    item = definition['folds'][0]
                    self.build(item, root/'out', batch=batch, capture=captures)
                    training = project_saved_fold(item['input_manifest'], item['fold_spec'])[2]
                    self.assertEqual(captures[-1][0].tolist(), [r['values'] for r in training])
                    self.assertEqual(captures[-1][1].tolist(), [r['label'] for r in training])
            for x, y in zip(captures[0], captures[1]): self.assertEqual(x.tolist(), y.tolist())

    def test_v2_native_booster_reloads_predictions_and_reuses_same_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); definition = batch_fixture(root, two_year=True)
            item = definition['folds'][0]
            with load_stock_ml_batch_inputs(definition) as batch, \
                 patch('axiom_research.stock_folds._environment', return_value={'synthetic': True}), \
                 patch('axiom_research.stock_folds._implementation', return_value={'synthetic': digest('code')}):
                run = build_stock_ml_fold_from_saved_inputs(item['input_manifest'], fold_spec=item['fold_spec'],
                                                           destination=root/'folds', batch=batch)
                feature = _read(run.path/'feature-slice.json')
                rows = [r for r in feature['rows'] if r['member'] and all(r['validity'])]
                independent = predict_stock_model(run.path, rows, ordered_features=item['input_manifest']['ordered_features'],
                                    feature_selection=item['input_manifest']['feature_selection'])
                self.assertEqual(independent, [r['score'] for r in run.predictions()['rows'] if r['valid']])
                with patch('axiom_research.stock_training.fit_predict_stock_model', side_effect=AssertionError('HIT trained')):
                    stats = {}; hit = build_stock_ml_fold_from_saved_inputs(item['input_manifest'], fold_spec=item['fold_spec'],
                                             destination=root/'folds', batch=batch, metrics=stats)
                self.assertEqual(hit.identity, run.identity)
                self.assertEqual((stats['train_calls'], stats['predict_calls'], stats['core_calls']), (0, 0, 0))

    def test_v2_compact_closure_and_stdlib_fresh_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); definition = batch_fixture(root, two_year=True)
            with load_stock_ml_batch_inputs(definition) as batch:
                run, _ = self.build(definition['folds'][0], root/'folds', batch=batch)
            feature = _read(run.path/'feature-slice.json'); label = _read(run.path/'label-slice.json')
            dataset = _read(run.path/'dataset.json')
            self.assertEqual(run.to_dict()['contract_version'], 'stock_ml_fold_v2')
            self.assertEqual(dataset['contract_version'], 'stock_fold_dataset_v2')
            self.assertEqual(len(feature['rows']), 6)
            self.assertNotIn('rows', label)
            self.assertEqual(label['selection_count'], len(feature['training_sessions'])*3)
            self.assertGreater(dataset['training_row_count'], 100)
            self.assertEqual(run.predictions()['contract_version'], 'stock_prediction_run_v2')
            from axiom_engine.core import StockPredictionFrame, validate_stock_predictions
            wire, indexed = validate_stock_predictions(StockPredictionFrame.from_dict(run.predictions()))
            self.assertEqual(wire, run.predictions())
            self.assertEqual(len(indexed), 6)
            original = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.startswith(('axiom_data', 'axiom_engine', 'numpy', 'pandas', 'qlib', 'lightgbm')):
                    raise AssertionError('fresh loader imported runtime '+name)
                return original(name, *args, **kwargs)
            with patch('builtins.__import__', guarded):
                self.assertEqual(load_stock_ml_fold(run.path).identity, run.identity)
            # Guard before the first import in a fresh process, so import cache
            # cannot conceal an optional runtime dependency in the public loader.
            script = '''import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('axiom_data','axiom_engine','numpy','pandas','qlib','lightgbm')):
        raise AssertionError('runtime imported '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from axiom_research import load_stock_ml_fold
print(load_stock_ml_fold(sys.argv[1]).identity)
'''
            loaded = subprocess.run([sys.executable, '-c', script, str(run.path)],
                                    env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                                    capture_output=True, text=True, timeout=15, check=True)
            self.assertEqual(loaded.stdout.strip(), run.identity)
            # Even resealed compact metadata cannot change the actual training
            # selection without the fresh immutable-parent reconstruction.
            label['selection_count'] -= 1
            label = seal({k: v for k, v in label.items() if k != 'label_ref'}, 'label_ref')
            write_json(run.path/'label-slice.json', label)
            fold = run.to_dict(); fold['label_ref'] = label['label_ref']
            from axiom_research.stock_fold_artifacts import OUTPUTS
            refs = {k: fold[k] for k in OUTPUTS.values()}
            fold['fold_ref'] = digest({'definition_ref': fold['definition_ref'], **refs})
            fold = seal({k: v for k, v in fold.items() if k != 'content_digest'}, 'content_digest')
            write_json(run.path/'fold.json', fold)
            manifest = _read(run.path/'manifest.json'); manifest['fold_ref'] = fold['fold_ref']
            for name in manifest['files']: manifest['files'][name] = file_digest(run.path/name)
            write_json(run.path/'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'slice/immutable parent mismatch'): load_stock_ml_fold(run.path)

    def test_limits_definition_changes_and_source_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); definition = batch_fixture(root)
            for limits in ({'maximum_source_bytes': True, 'maximum_matrix_bytes': 100},
                           {'maximum_source_bytes': 1, 'maximum_matrix_bytes': 100000},
                           {'maximum_source_bytes': 10000000, 'maximum_matrix_bytes': 1}):
                with self.assertRaises(ValueError): load_stock_ml_batch_inputs(definition, limits=limits)
            bad = deepcopy(definition); bad['folds'][1]['input_manifest']['snapshot'] = 'other'
            with self.assertRaisesRegex(ValueError, 'common source'): load_stock_ml_batch_inputs(bad)
            bad = deepcopy(definition); bad['folds'].reverse()
            with self.assertRaisesRegex(ValueError, 'overlap or are unordered'): load_stock_ml_batch_inputs(bad)
            batch = load_stock_ml_batch_inputs(definition)
            bad = deepcopy(definition['folds'][0]); bad['fold_spec']['evaluation_cutoff'] = '2030-01-01T00:00:00Z'
            with self.assertRaisesRegex(ValueError, 'outside saved batch'): self.build(bad, root/'out', batch=batch)
            with self.assertRaisesRegex(ValueError, 'validated saved batch'):
                self.build(definition['folds'][0], root/'out', batch=True)
            with (root/'shared-proof.json').open('a') as stream: stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'source changed'): self.build(definition['folds'][0], root/'out', batch=batch)
            batch.close()

    def test_one_day_outcome_shift_and_duplicate_raw_key_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); definition = batch_fixture(root)
            item = definition['folds'][0]; desc = item['input_manifest']['training_labels'][0]['raw']
            original = _read(desc['path'])
            for duplicate in (False, True):
                raw = deepcopy(original)
                if duplicate: raw['rows'].append(deepcopy(raw['rows'][0]))
                else: raw['rows'][0]['end_session'] = item['input_manifest']['calendar'][6]
                raw = seal({k: v for k, v in raw.items() if k != 'label_ref'}, 'label_ref'); write_json(desc['path'], raw)
                desc.update(file_digest=file_digest(desc['path']), label_ref=raw['label_ref'])
                # Full per-fit Label admission now happens before the batch
                # returns, while standalone fold loading keeps its disk path.
                with self.assertRaisesRegex(ValueError, 'endpoint conflict|duplicate'):
                    load_stock_ml_batch_inputs(definition)

    def test_complete_label_objects_released_and_only_selection_inputs_retained(self):
        class Parent(dict):
            pass
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); definition = batch_fixture(root)
            live = []
            evaluation_path = definition['folds'][0]['input_manifest']['evaluation_labels']['path']
            def tracked(desc, key):
                parent = Parent(read_parent(desc, key))
                live.append((desc['path'], weakref.ref(parent)))
                return parent
            with patch('axiom_research.stock_batch.read_parent', side_effect=tracked):
                batch = load_stock_ml_batch_inputs(definition)
            from axiom_research.stock_batch import _DATA
            value = _DATA[batch]
            self.assertNotIn('read_parent', value)
            self.assertNotIn('cache', value)
            self.assertEqual(len(value['labels']), len(definition['folds']))
            fields = {'valid', 'invalid_reason', 'normalized_target', 'raw_return',
                      'label_available_at', 'normalized_available_at', 'normalized_parent_ref'}
            for admission in value['labels'].values():
                self.assertEqual(set(admission), {'input_manifest_ref', 'fold_spec_ref',
                    'rows', 'parents', 'section_refs', 'raw_refs', 'evaluation'})
                self.assertTrue(admission['rows'])
                self.assertTrue(any(not row['valid'] for row in admission['rows'].values()))
                self.assertTrue(all(set(row) == fields for row in admission['rows'].values()))
            gc.collect()
            self.assertTrue(all(ref() is None for path, ref in live if path != evaluation_path))
            self.assertEqual(sum(ref() is not None for path, ref in live), 1)
            del admission, value
            batch.close(); gc.collect()
            self.assertTrue(all(ref() is None for path, ref in live))

    def test_six_value_projection_exact_with_zero_parent_reads_and_no_mutation_leak(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); definition = batch_fixture(root, two_year=True, reverse=True)
            originals = [project_saved_fold(f['input_manifest'], f['fold_spec']) for f in definition['folds']]
            with load_stock_ml_batch_inputs(definition) as batch, \
                 patch('axiom_research.stock_batch.read_parent', side_effect=AssertionError('projection read parent')), \
                 patch('axiom_research.stock_batch._admit_saved_fold_labels', side_effect=AssertionError('projection admitted again')):
                for f, expected in zip(definition['folds'], originals):
                    projected = batch._project(f['input_manifest'], f['fold_spec'])
                    self.assertEqual(projected, expected)
                    projected[0]['rows'][0]['values'][0] = -999
                    projected[1]['parents'][0]['sessions'].clear()
                    projected[1]['section_refs'].clear()
                    projected[2][0]['values'][0] = -999
                    projected[3]['injected'] = 1
                    projected[4].clear()
                    projected[5]['rows'][0]['return'] = -999
                    self.assertEqual(batch._project(f['input_manifest'], f['fold_spec']), expected)

    def test_missing_immature_rows_remain_distinct_from_explicit_invalid_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); definition = batch_fixture(root)
            item = definition['folds'][0]
            manifest, spec = item['input_manifest'], item['fold_spec']
            original = project_saved_fold(manifest, spec)
            shortened = deepcopy(item)
            desc = shortened['input_manifest']['training_labels'][0]
            raw = _read(desc['raw']['path'])
            fit = manifest['calendar'].index(spec['fit_session'])
            mature = [d for d in desc['sessions'] if manifest['calendar'].index(d) + 5 <= fit]
            projected = {k: v for k, v in raw.items() if k not in ('rows', 'label_ref')}
            feature = _read(manifest['feature_parents'][0]['features']['path'])
            projected.update(rows=[r for r in raw['rows'] if r['feature_session'] in mature],
                             parent_label_ref=raw['label_ref'], feature_parent_ref=feature['feature_ref'],
                             date_projection=mature)
            norm = normalize_forward_labels(seal(projected, 'label_ref'), features=feature, cutoff=spec['fit_cutoff'])
            path = root/'mature-only.json'; write_json(path, norm)
            desc['sessions'] = mature
            desc['normalized'] = {'path': str(path), 'file_digest': file_digest(path), 'label_ref': norm['label_ref']}
            expected = project_saved_fold(shortened['input_manifest'], spec)
            self.assertEqual(expected[2], original[2])
            self.assertEqual(expected[3], original[3])
            self.assertGreater(expected[3]['LABEL_NOT_MATURE'], 0)
            missing = [r for r in expected[1]['rows'] if r['invalid_reason'] == 'LABEL_NOT_MATURE']
            explicit = [r for r in original[1]['rows'] if r['invalid_reason'] == 'LABEL_NOT_MATURE']
            self.assertTrue(all(r['normalized_parent_ref'] is None for r in missing))
            self.assertTrue(all(r['normalized_parent_ref'] is not None for r in explicit))
            one = {'contract_version': definition['contract_version'], 'folds': [shortened]}
            with load_stock_ml_batch_inputs(one) as batch:
                self.assertEqual(batch._project(shortened['input_manifest'], spec), expected)

    def test_label_file_change_after_compact_admission_still_rejects_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); definition = batch_fixture(root)
            with load_stock_ml_batch_inputs(definition) as batch:
                item = definition['folds'][0]
                desc = item['input_manifest']['training_labels'][0]['normalized']
                with Path(desc['path']).open('a') as stream:
                    stream.write(' ')
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    batch._project(item['input_manifest'], item['fold_spec'])


if __name__ == '__main__': unittest.main()

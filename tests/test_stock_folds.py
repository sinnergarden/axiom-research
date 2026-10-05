"""Small synthetic saved parents; no supplier/Feature/account or real input read."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import builtins
import json
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import (build_stock_ml_fold_from_saved_inputs, load_stock_ml_fold,
                            load_stock_model, predict_stock_model)
from axiom_research.feature_catalog import load_feature_catalog
from axiom_research.stock_artifacts import digest, file_digest, write_json, _read
from axiom_research.stock_fold_inputs import seal, project_saved_fold, validate_spec
from axiom_research.stock_label_normalization import normalize_forward_labels


def fixture(root, *, fit_index=65, all_null_prediction=False, projected=True,
            label_start=None, labels_include_immature=True, calendar=None, two_year=False):
    # Explicit artificial sessions with irregular holes, not a market calendar inference.
    calendar = calendar or [(date(2023, 9, 1)+timedelta(days=2*i+(i//7))).isoformat() for i in range(80)]
    fit = calendar[fit_index]
    start = fit_index-65
    if two_year:
        f = date.fromisoformat(fit)
        try: boundary = f.replace(year=f.year-2)
        except ValueError: boundary = f.replace(year=f.year-2, day=28)
        start = next(i for i, d in enumerate(calendar) if d >= boundary.isoformat())
    train_dates = calendar[start:fit_index]
    pred_dates = [calendar[fit_index], calendar[fit_index+2]]
    fit_cutoff = fit+'T20:30:00+08:00'; universe = ['A', 'B', 'C']
    catalog = load_feature_catalog(); selection = [catalog.default_selection[0].to_dict()]
    label_dates = calendar[start if label_start is None else label_start:
                           fit_index if labels_include_immature else fit_index-4]
    columns = [selection[0]['id']]; dates = sorted(set(train_dates+pred_dates+label_dates))
    proof = [{'session': d, 'core_plan': {'synthetic_session': d}, 'core_frame_ref': digest(['frame', d])} for d in dates]
    features = seal({'contract_version': 'stock_feature_build_v1', 'catalog_ref': catalog.identity,
        'selection': selection, 'ordered_features': columns, 'qlib_view': {'snapshot_id': 's-fixed', 'view_id': digest('view'),
            'queries': [{'pit_policy': 'best_effort_vendor_v1', 'symbols': universe}],
            'universe_query': {'pit_policy': 'best_effort_vendor_v1', 'symbols': universe}},
        'input_evidence_ref': digest(proof), 'rows': [{'security_id': s, 'session': d, 'values': [float(i+1)],
            'validity': [True], 'availability': [None if all_null_prediction and d in pred_dates and s == 'A'
                else d+'T20:00:00+08:00'], 'reasons': [[]], 'member': s != 'C',
            'knowledge_cutoff': d+'T20:30:00+08:00', 'source_refs': [digest(['frame', d]), digest({'synthetic_session': d})]}
            for d in dates for i, s in enumerate(universe)]}, 'feature_ref')
    def raw(dates, cutoff):
        rows = []
        for s in universe:
            for d in dates:
                i = calendar.index(d); end = calendar[i+5]
                valid = end <= cutoff[:10]
                rows.append({'security_id': s, 'feature_session': d, 'start_session': calendar[i+1], 'end_session': end,
                    'return': (universe.index(s)+1)/100 if valid else None, 'valid': valid,
                    'invalid_reason': None if valid else 'missing_end_close',
                    'label_available_at': end+'T20:00:00+08:00' if valid else None, 'source_refs': [digest('raw')]})
        return seal({'contract_version': 'stock_label_build_v1', 'calendar_ref': digest({
            'contract_version': 'stock_label_calendar_v1', 'sessions': calendar}),
            'label_spec': {'label_id': 'forward_5_session_open_close_v1', 'horizon_sessions': 5, 'normalization': 'none'},
            'source_evidence': {'context': {'snapshot_id': 's-fixed', 'query': {'pit_policy': 'best_effort_vendor_v1',
                'purpose': 'label_outcomes', 'symbols': universe, 'cutoff_by_session': {d: cutoff for d in calendar}}}},
            'rows': rows}, 'label_ref')
    training_raw = raw(label_dates, fit_cutoff)
    normalized_raw = training_raw
    if projected:
        normalized_raw = {k: v for k, v in training_raw.items() if k not in ('rows', 'label_ref')}
        normalized_raw.update(rows=training_raw['rows'], parent_label_ref=training_raw['label_ref'],
            feature_parent_ref=features['feature_ref'], date_projection=label_dates)
        normalized_raw = seal(normalized_raw, 'label_ref')
    normalized = normalize_forward_labels(normalized_raw, features=features, cutoff=fit_cutoff)
    eval_cutoff = calendar[-1]+'T20:30:00+08:00'; evaluation = raw(pred_dates, eval_cutoff)
    result = {'read_sessions': calendar, 'read_symbols': universe, 'snapshot_id': 's-fixed', 'pit_policy': 'best_effort_vendor_v1'}
    scope = seal({'contract_version': 'private_frozen_scope_v1', 'request': {}, 'request_ref': digest({}),
        'result': result, 'result_ref': digest(result), 'source_proof': {'synthetic': True},
        'source_proof_ref': digest({'synthetic': True})}, 'scope_bundle_ref')
    values = {'features.json': features, 'proof.json': proof, 'raw.json': training_raw,
              'normalized.json': normalized, 'evaluation.json': evaluation, 'scope.json': scope}
    for name, value in values.items(): write_json(root/name, value)
    def desc(name, ref): return {'path': str(root/name), 'file_digest': file_digest(root/name), ref: values[name][ref]}
    manifest = {'contract_version': 'stock_ml_saved_inputs_v1', 'scope': desc('scope.json', 'scope_bundle_ref'),
        'snapshot': 's-fixed', 'pit_policy': 'best_effort_vendor_v1', 'calendar': calendar, 'universe': universe,
        'catalog_ref': catalog.identity, 'feature_selection': selection, 'ordered_features': columns,
        'feature_parents': [{'features': desc('features.json', 'feature_ref'), 'sessions': dates,
            'input_evidence': {'path': str(root/'proof.json'), 'file_digest': file_digest(root/'proof.json'),
                               'input_evidence_ref': digest(proof)}}],
        'training_labels': [{'raw': desc('raw.json', 'label_ref'), 'normalized': desc('normalized.json', 'label_ref'),
            'sessions': label_dates, 'raw_projection': projected}], 'evaluation_labels': desc('evaluation.json', 'label_ref')}
    spec = {'contract_version': 'stock_ml_fold_spec_v1', 'training_window': {
        'unit': 'feature_sessions', 'length': 65, 'end': 'previous_fit_session'},
        'fit_session': fit, 'fit_cutoff': fit_cutoff, 'simulated_model_available_at': fit+'T20:45:00+08:00',
        'oos_trade_sessions': [calendar[calendar.index(d)+1] for d in pred_dates],
        'inference_cutoff_by_session': {d: d+'T21:00:00+08:00' for d in pred_dates}, 'evaluation_cutoff': eval_cutoff}
    if two_year:
        spec.update(contract_version='stock_ml_fold_spec_v2', training_window={
            'unit': 'calendar_years', 'length': 2, 'end': 'previous_fit_session',
            'start': 'fit_date_minus_years_inclusive', 'leap_day': 'clamp_feb_28'})
    return manifest, spec


def backend(X, y, P, *, ordered_features, parameters, num_boost_round, metrics):
    class Model:
        def save_model(self, path): Path(path).write_text('synthetic saved booster')
    metrics.update(train_calls=1, predict_calls=1)
    return Model(), [float(row[0]) for row in P]


class SavedFoldTests(unittest.TestCase):
    def build(self, manifest, spec, destination):
        with patch('axiom_research.stock_training.fit_predict_stock_model', side_effect=backend), \
             patch('axiom_research.stock_folds._environment', return_value={'synthetic': True}), \
             patch('axiom_research.stock_folds._implementation', return_value={'synthetic': digest('code')}):
            metrics = {}; run = build_stock_ml_fold_from_saved_inputs(manifest, fold_spec=spec, destination=destination, metrics=metrics)
        return run, metrics

    def test_sliding_window_uses_actual_calendar_and_per_fit_maturity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a = root/'a'; b = root/'b'; a.mkdir(); b.mkdir()
            m1, s1 = fixture(a); m2, s2 = fixture(b, fit_index=69)
            f1, _, t1, e1, _, _ = project_saved_fold(m1, s1); f2, _, t2, e2, _, _ = project_saved_fold(m2, s2)
            self.assertEqual(f1['training_sessions'], m1['calendar'][:65])
            self.assertEqual(f2['training_sessions'], m2['calendar'][4:69])
            self.assertEqual((len(t1), len(t2)), (122, 122))
            self.assertEqual((e1['LABEL_NOT_MATURE'], e2['LABEL_NOT_MATURE']), (12, 12))
            self.assertEqual(max(r['session'] for r in t2), m2['calendar'][64])

    def test_full_normalized_parent_outside_window_and_unprovided_immature_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); m, s = fixture(root, fit_index=69, label_start=0, labels_include_immature=False)
            feature, labels, training, excluded, _, _ = project_saved_fold(m, s)
            self.assertEqual(feature['training_sessions'], m['calendar'][4:69])
            self.assertEqual(len(labels['rows']), 195); self.assertEqual(len(training), 122)
            self.assertEqual(excluded['LABEL_NOT_MATURE'], 12)
            self.assertEqual(labels['parents'][0]['sessions'], m['calendar'][:65])
            self.assertFalse(any(r['session'] in m['calendar'][:4] for r in feature['rows']))

    def test_build_load_exact_hit_and_old_input_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); m, s = fixture(root); before = {p.name: file_digest(p) for p in root.glob('*.json')}
            run, metrics = self.build(m, s, root/'folds')
            self.assertEqual(metrics['data_read_calls'], 0); self.assertEqual(metrics['feature_core_calls'], 0)
            self.assertEqual(metrics['train_calls'], 1); self.assertEqual(metrics['predict_calls'], 1)
            self.assertEqual(load_stock_model(run.path)['contract_version'], 'stock_model_release_v2')
            self.assertEqual({p.name: file_digest(p) for p in root.glob('*.json')}, before)
            with patch('axiom_research.stock_training.fit_predict_stock_model', side_effect=AssertionError('HIT trained')), \
                 patch('axiom_research.stock_folds._environment', return_value={'synthetic': True}), \
                 patch('axiom_research.stock_folds._implementation', return_value={'synthetic': digest('code')}):
                stats = {}; cached = build_stock_ml_fold_from_saved_inputs(m, fold_spec=s, destination=root/'folds', metrics=stats)
            self.assertTrue(cached.reused); self.assertEqual(cached.identity, run.identity)
            self.assertEqual((stats['train_calls'], stats['predict_calls'], stats['core_calls']), (0, 0, 0))
            (run.path/'booster.txt').write_text('damaged')
            with self.assertRaisesRegex(ValueError, 'file mismatch'): self.build(m, s, root/'folds')

    def test_original_feature_clocks_and_null_availability_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); m, s = fixture(root, all_null_prediction=True, projected=False)
            run, _ = self.build(m, s, root/'folds'); rows = run.predictions()['rows']
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertEqual(row['feature_knowledge_cutoff'], row['session']+'T20:30:00+08:00')
                self.assertEqual(row['knowledge_cutoff'], row['available_at'])
                self.assertEqual(row['knowledge_cutoff'], row['session']+'T21:00:00+08:00')
                if row['security_id'] == 'A':
                    self.assertIsNone(row['feature_available_at']); self.assertIsNone(row['score']); self.assertFalse(row['valid'])
            self.assertEqual(run.to_dict()['engine_admission']['runtime'], 'UNSUPPORTED_V2')

    def test_v2_native_booster_independent_prediction_matches_saved_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); m, s = fixture(root)
            with patch('axiom_research.stock_folds._environment', return_value={'synthetic': True}), \
                 patch('axiom_research.stock_folds._implementation', return_value={'synthetic': digest('code')}):
                run = build_stock_ml_fold_from_saved_inputs(m, fold_spec=s, destination=root/'folds')
            saved_features = _read(run.path/'feature-slice.json')
            rows = [r for r in saved_features['rows'] if r['session'] in saved_features['prediction_sessions'] and r['member']]
            independent = predict_stock_model(run.path, rows, ordered_features=m['ordered_features'],
                                                feature_selection=m['feature_selection'])
            self.assertEqual(independent, [r['score'] for r in run.predictions()['rows'] if r['valid']])

    def test_parent_corruption_and_future_normalization_are_not_cache_hits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); m, s = fixture(root); run, _ = self.build(m, s, root/'folds')
            with (root/'proof.json').open('a') as stream: stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'input evidence mismatch'): load_stock_ml_fold(run.path)
            m, s = fixture(root); norm = _read(root/'normalized.json'); norm['cutoff'] = s['simulated_model_available_at']
            norm = seal({k: v for k, v in norm.items() if k != 'label_ref'}, 'label_ref'); write_json(root/'normalized.json', norm)
            m['training_labels'][0]['normalized'].update(file_digest=file_digest(root/'normalized.json'), label_ref=norm['label_ref'])
            with self.assertRaises(ValueError): project_saved_fold(m, s)

    def test_original_json_format_preserves_evidence_ref_and_saved_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); m, s = fixture(root); proof = _read(root/'proof.json')
            expected = project_saved_fold(m, s)
            for name, text in (('spaced', json.dumps(proof, sort_keys=True)+'\n'),
                               ('indented', json.dumps(proof, indent=2)+'\n'),
                               ('no_final_lf', json.dumps(proof))):
                with self.subTest(format=name):
                    (root/'proof.json').write_text(text)
                    m['feature_parents'][0]['input_evidence']['file_digest'] = file_digest(root/'proof.json')
                    actual = project_saved_fold(m, s)
                    self.assertEqual(actual[2:], expected[2:])
                    run, _ = self.build(m, s, root/name)
                    self.assertEqual(load_stock_ml_fold(run.path).identity, run.identity)
                    self.assertEqual((root/'proof.json').read_text(), text)

    def test_formatted_proof_rehash_cannot_replace_logical_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); m, s = fixture(root); proof = _read(root/'proof.json')
            proof[0]['core_frame_ref'] = digest('changed evidence')
            (root/'proof.json').write_text(json.dumps(proof, indent=2)+'\n')
            # Even updating the byte descriptor cannot change the Feature's ref.
            m['feature_parents'][0]['input_evidence']['file_digest'] = file_digest(root/'proof.json')
            with self.assertRaisesRegex(ValueError, 'input evidence mismatch'):
                project_saved_fold(m, s)

    def test_bool_window_missing_previous_session_and_model_clock_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            m, s = fixture(Path(temp))
            for change in ('bool', 'before_model', 'same_fit'):
                bad = deepcopy(s)
                if change == 'bool': bad['training_window']['length'] = True
                elif change == 'before_model': bad['simulated_model_available_at'] = bad['fit_cutoff']
                else: bad['inference_cutoff_by_session'][m['calendar'][65]] = bad['fit_cutoff']
                with self.assertRaises(ValueError): validate_spec(bad, m['calendar'])

    def test_resealed_core_raw_inputs_and_invalid_reason_cannot_change_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); m, s = fixture(root); original = _read(root/'normalized.json')
            for kind in ('facts', 'reason', 'plan', 'context'):
                with self.subTest(kind=kind):
                    value = deepcopy(original)
                    if kind == 'facts':
                        value['core_facts'][0]['rows'][0]['values'][0] = 99.0
                        for row in value['core_frames'][0]['rows'][:2]:
                            row['values'][0] = 1.0 if row['security_id'] == 'A' else -1.0
                        for row in value['rows']:
                            if row['feature_session'] == m['calendar'][0] and row['security_id'] in ('A', 'B'):
                                row['normalized_target'] = 1.0 if row['security_id'] == 'A' else -1.0
                    elif kind == 'reason':
                        next(r for r in value['rows'] if r['security_id'] == 'C')['invalid_reason'] = None
                    elif kind == 'plan': value['core_plan'][0]['nodes'][0]['params']['ddof'] = 1
                    else: value['core_context'][0]['cutoffs'][m['calendar'][0]] = s['simulated_model_available_at']
                    for section, plan, facts, context, frame in zip(value['sections'], value['core_plan'],
                            value['core_facts'], value['core_context'], value['core_frames']):
                        section.update(core_plan_ref=digest(plan), fact_ref=digest(facts), core_context_ref=digest(context))
                        frame.update(plan_identity=section['core_plan_ref'], fact_identity=section['fact_ref'],
                                     context_identity=section['core_context_ref'])
                        section['frame_ref'] = digest(frame)
                    value['frame_ref'] = digest([{'feature_session': x['feature_session'], 'frame_ref': x['frame_ref']}
                                                  for x in value['sections']])
                    for row in value['rows']:
                        section = next(x for x in value['sections'] if x['feature_session'] == row['feature_session'])
                        old_section = next(x for x in original['sections'] if x['feature_session'] == row['feature_session'])
                        row['source_refs'] = sorted(set(section['frame_ref'] if x == old_section['frame_ref'] else x
                                                       for x in row['source_refs']))
                    value = seal({k: v for k, v in value.items() if k != 'label_ref'}, 'label_ref')
                    write_json(root/'normalized.json', value)
                    m['training_labels'][0]['normalized'].update(file_digest=file_digest(root/'normalized.json'), label_ref=value['label_ref'])
                    with self.assertRaisesRegex(ValueError, 'Core/raw input|saved row/Core/raw'):
                        project_saved_fold(m, s)

    def test_loader_has_no_runtime_imports_or_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); m, s = fixture(root); run, _ = self.build(m, s, root/'folds')
            original = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.startswith(('axiom_data', 'axiom_engine', 'lightgbm', 'qlib', 'numpy', 'pandas')):
                    raise AssertionError('loader imported runtime '+name)
                return original(name, *args, **kwargs)
            with patch('builtins.__import__', guarded):
                self.assertEqual(load_stock_ml_fold(run.path).identity, run.identity)
                self.assertEqual(load_stock_model(run.path)['model_ref'], run.model()['model_ref'])

    def test_resealed_prediction_clock_forgery_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); m, s = fixture(root); run, _ = self.build(m, s, root/'folds')
            pred = run.predictions(); pred['rows'][0]['available_at'] = s['fit_cutoff']
            pred = seal({k: v for k, v in pred.items() if k != 'signal_run_ref'}, 'signal_run_ref')
            write_json(run.path/'predictions.json', pred)
            fold = run.to_dict(); fold['signal_run_ref'] = pred['signal_run_ref']
            evidence = run.evidence(); evidence['signal_ref'] = pred['signal_run_ref']
            evidence = seal({k: v for k, v in evidence.items() if k != 'evidence_ref'}, 'evidence_ref')
            write_json(run.path/'signal-evidence.json', evidence); fold['evidence_ref'] = evidence['evidence_ref']
            from axiom_research.stock_fold_artifacts import OUTPUTS
            refs = {k: fold[k] for k in OUTPUTS.values()}; fold['fold_ref'] = digest({'definition_ref': fold['definition_ref'], **refs})
            fold = seal({k: v for k, v in fold.items() if k != 'content_digest'}, 'content_digest'); write_json(run.path/'fold.json', fold)
            wire = _read(run.path/'manifest.json'); wire['fold_ref'] = fold['fold_ref']
            for name in wire['files']: wire['files'][name] = file_digest(run.path/name)
            write_json(run.path/'manifest.json', wire)
            with self.assertRaisesRegex(ValueError, 'inference clock'): load_stock_ml_fold(run.path)


if __name__ == '__main__': unittest.main()

"""Small frozen API edge cases; real owner integration has a separate fixture."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import gc
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import weakref
from unittest.mock import patch

from axiom_research import (save_stock_signal_evaluation_inputs, evaluate_stock_signal_inputs,
    load_stock_signal_evaluation, audit_stock_signal_evaluation, evaluate_stock_signals)
from axiom_research.stock_artifacts import digest, file_digest, _read, write_json, _verify_ref
from axiom_research.stock_batch import StockMLBatchInputs, _TOKEN
from axiom_research.stock_fold_inputs import file_fingerprint
from axiom_research import stock_signal_evaluation_projection as frozen
from test_stock_signal_evaluation import fixture, seal, raw_variant


class MockOwner:
    """Patch the owner boundary only for focused frozen API edge cases."""
    def __init__(self, root, *, two_signals=False):
        self.root = root; self.views = {}; self.active = 0; self.calls = 0
        a, label, self.scope = fixture(root/'a')
        descriptors = [('a', a)]
        if two_signals:
            b, _, _ = fixture(root/'b', reverse=True,
                invalid=[(security, day) for security in self.scope['universe'][:2] for day in self.scope['sessions']])
            descriptors.append(('b', b))
        self.raw = _read(Path(label['path']))['evaluation']; self.label = label
        self.common = {'snapshot': 'synthetic_snapshot_v1', 'pit_policy': 'synthetic_pit_v1',
            'calendar': self.scope['calendar'], 'universe': self.scope['universe']}
        view_wire = seal({'definition': self.common}, 'prepared_view_ref')
        write_json(root/'prepared.json', view_wire)
        prepared = {'path': str(root/'prepared.json'), 'file_digest': file_digest(root/'prepared.json'),
            'prepared_view_ref': view_wire['prepared_view_ref']}
        self.inputs = {}; folds = []
        for name, descriptor in descriptors:
            path = Path(descriptor['path']).parent
            old_features = _read(path/'features.json'); old_model = _read(path/'model.json')
            spec = {'training_tag': 'training_'+name, 'fit_cutoff': old_model['fit_cutoff'],
                'simulated_model_available_at': '2024-01-01T20:00:00Z',
                'evaluation_cutoff': self.scope['evaluation_cutoff']}
            selector = {'path': str(root/('selector-'+name+'.json')),
                'file_digest': digest(['selector-file', name]), 'selector_ref': digest(['selector', name])}
            inputs = seal({'contract_version': 'stock_ml_saved_inputs_v2', 'prepared_view': prepared,
                'fold_spec_ref': digest(spec), 'selectors': {'evaluation_labels': selector},
                'core_result_refs': []}, 'input_ref')
            features = seal({'contract_version': 'stock_feature_slice_v3',
                'input_manifest_ref': digest(inputs), 'prepared_view_ref': view_wire['prepared_view_ref'],
                'selectors': inputs['selectors'], 'universe': self.scope['universe'],
                'ordered_features': old_features['ordered_features'], 'catalog_ref': old_features['catalog_ref'],
                'selection': old_features['selection'], 'training_sessions': [],
                'prediction_sessions': self.scope['sessions'],
                'parents_by_session': {day: {'feature_ref': digest(['original-day', day]),
                    'qlib_view_ref': digest('qlib')} for day in self.scope['sessions']},
                'rows': old_features['rows']}, 'feature_ref')
            model = seal({**{k: v for k, v in old_model.items() if k != 'model_ref'},
                'contract_version': 'stock_model_release_v2', 'feature_ref': features['feature_ref'],
                'simulated_available_at': spec['simulated_model_available_at'],
                'clock_basis': 'declared_simulation'}, 'model_ref')
            signal = _read(path/'predictions.json')
            for row in signal['rows']:
                original = next(r for r in features['rows'] if (r['security_id'], r['session']) ==
                                (row['security_id'], row['session']))
                row.update(available_at=row['knowledge_cutoff'],
                    feature_knowledge_cutoff=original['knowledge_cutoff'],
                    feature_available_at=max(original['availability']),
                    simulated_model_available_at=model['simulated_available_at'],
                    source_refs=[features['feature_ref'], model['model_ref'], digest('catalog'), digest('qlib')])
            signal = seal({**{k: v for k, v in signal.items() if k != 'signal_run_ref'},
                'contract_version': 'stock_prediction_run_v2', 'feature_ref': features['feature_ref'],
                'model_ref': model['model_ref'], 'fold_spec_ref': digest(spec),
                'clock_basis': 'declared_simulation'}, 'signal_run_ref')
            evaluation = seal({'contract_version': 'stock_matrix_evaluation_target_slice_v1',
                'prepared_view_ref': view_wire['prepared_view_ref'], 'fold_spec_ref': digest(spec),
                'selector': selector, 'cutoff': self.scope['evaluation_cutoff'],
                'rows': deepcopy(self.raw['rows'])}, 'label_ref')
            header = {'raw_input': deepcopy(label), 'contract_version': self.raw['contract_version'],
                'label_ref': self.raw['label_ref'], 'label_spec': deepcopy(self.raw['label_spec']),
                'calendar_ref': self.raw['calendar_ref'], 'source_ref': self.raw['source_ref'],
                'source_context': deepcopy(self.raw['source_evidence']['context']), 'sessions': self.scope['sessions']}
            contents, leaves = {}, []
            for row in self.raw['rows']:
                endpoint = {'security_id': row['security_id'], 'feature_session': row['feature_session']}
                ref = digest(endpoint); contents[ref] = endpoint
                leaves.append({**endpoint, 'raw_label_ref': header['label_ref'],
                    'source_ref': header['source_ref'], 'query_ref': digest(header['source_context']['query']),
                    'adjustment_anchor': header['label_spec']['adjustment_anchor'],
                    'start_open_ref': ref, 'end_close_ref': ref})
            value = SimpleNamespace(common=self.common, features=features, evaluation=evaluation,
                fold_binding={'training_raw_provenance': []},
                raw_provenance=[{'raw_build': header['raw_input'], **{key: header[key] for key in
                    ('contract_version', 'label_ref', 'label_spec', 'calendar_ref', 'source_ref')},
                    'source_evidence': self.raw['source_evidence']}],
                label_leaf_bindings={'rows': leaves, 'contents': contents})
            self.views[digest(inputs), digest(spec)] = value
            fold = seal({'contract_version': 'stock_ml_fold_v3', 'status': 'COMPLETE',
                'definition': {'input_manifest': inputs, 'fold_spec': spec},
                'signal_run_ref': signal['signal_run_ref'], 'model_ref': model['model_ref'],
                'feature_ref': features['feature_ref']}, 'content_digest')
            for filename, wire in [('feature-slice.json', features), ('model.json', model),
                                   ('predictions.json', signal), ('fold.json', fold)]:
                write_json(path/filename, wire)
            filenames = ['feature-slice.json', 'model.json', 'predictions.json', 'fold.json', 'booster.txt']
            write_json(path/'manifest.json', {'contract_version': 'stock_ml_fold_manifest_v2',
                'files': {filename: file_digest(path/filename) for filename in filenames}})
            self.inputs[name] = {'path': str(path/'predictions.json'),
                'file_digest': file_digest(path/'predictions.json'), 'signal_run_ref': signal['signal_run_ref']}
            folds.append({'input_manifest': inputs, 'fold_spec': spec})
        manifest = {'contract_version': 'stock_ml_batch_inputs_v2', 'definition': {'mock': True},
            'definition_ref': digest({'mock': True}), 'prepared_view': prepared, 'folds': folds, 'status': 'COMPLETE'}
        manifest['batch_ref'] = digest(manifest); manifest = seal(manifest, 'content_digest')
        self.table = tuple((str(path), file_digest(path))
                           for path in sorted(root.rglob('*')) if path.is_file())
        self.marks = {Path(path): file_fingerprint(path) for path, _ in self.table}
        self.manifest = manifest

    def check(self):
        if any(file_fingerprint(path) != mark for path, mark in self.marks.items()):
            raise ValueError('saved matrix source changed')

    def batch(self):
        state = SimpleNamespace(store=SimpleNamespace(check=self.check), close=lambda: None)
        return StockMLBatchInputs(_TOKEN, {'identity': self.manifest['batch_ref'], 'manifest': self.manifest,
            'matrix_state': state, 'metrics': {}, 'closed': False})

    @contextmanager
    def project(self, batch, inputs, spec):
        self.check(); self.active += 1; self.calls += 1
        view = self.views[digest(inputs), digest(spec)]
        view.source_record_indices = tuple(range(len(self.table)))
        try:
            yield view
            self.check()
        finally:
            self.active -= 1

    def load_fold(self, path, *, batch):
        self.check(); manifest = _read(Path(path)/'manifest.json')
        for name, expected in manifest['files'].items():
            if file_digest(Path(path)/name) != expected: raise ValueError('saved fold file mismatch')
        for name, key in [('fold.json', 'content_digest'), ('model.json', 'model_ref'),
                          ('feature-slice.json', 'feature_ref'), ('predictions.json', 'signal_run_ref')]:
            _verify_ref(_read(Path(path)/name), key)

    @contextmanager
    def patch_owner(self):
        owner = self
        def project(batch, inputs, spec): return owner.project(batch, inputs, spec)
        def raw(batch, descriptor):
            from axiom_research.stock_matrix_reader import _admit_raw_label
            return _admit_raw_label(descriptor)
        with patch.object(StockMLBatchInputs, '_project_evaluation', new=project, create=True), \
             patch.object(StockMLBatchInputs, '_evaluation_source_records', return_value=self.table), \
             patch.object(StockMLBatchInputs, '_admit_raw_label', new=raw), \
             patch('axiom_research.stock_fold_artifacts.load_stock_ml_fold', side_effect=self.load_fold), \
             patch('axiom_research.stock_batch.load_stock_ml_batch_inputs', side_effect=lambda manifest: self.batch()):
            yield


class MatrixPublicFrozenMockTests(unittest.TestCase):
    def test_default_override_same_core_input_v4_and_zero_compute_hit(self):
        from axiom_engine.core import evaluate_signal_statistics
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); owner = MockOwner(root/'source', two_signals=True)
            with owner.patch_owner(), owner.batch() as batch:
                with patch('axiom_engine.core.evaluate_signal_statistics', side_effect=AssertionError('freeze computed')):
                    default = save_stock_signal_evaluation_inputs(owner.inputs, batch=batch,
                        scope=owner.scope, destination=root/'default')
                    override = save_stock_signal_evaluation_inputs(owner.inputs, batch=batch,
                        raw_label_input=owner.label, scope=owner.scope, destination=root/'override')
                self.assertEqual(default.artifact_contract_version, 'stock_signal_evaluation_inputs_v2')
                self.assertEqual(owner.active, 0)
                self.assertEqual(owner.calls, 4)
                with patch('axiom_engine.core.evaluate_signal_statistics', wraps=evaluate_signal_statistics) as core:
                    a = evaluate_stock_signal_inputs(default, scope=owner.scope, destination=root/'reports-a')
                    b = evaluate_stock_signal_inputs(override, scope=owner.scope, destination=root/'reports-b')
                    self.assertEqual(core.call_args_list[0].args, core.call_args_list[2].args)
                    self.assertEqual(core.call_args_list[1].args, core.call_args_list[3].args)
                for name in owner.inputs:
                    self.assertEqual(a[name].to_dict()['series'], b[name].to_dict()['series'])
                    self.assertEqual(b[name].to_dict()['contract_version'], 'stock_signal_evidence_v4')
                    self.assertEqual(b[name].to_dict()['label_ref'], owner.label['label_ref'])
                    self.assertEqual(load_stock_signal_evaluation(b[name].path).to_dict(), b[name].to_dict())
                with patch('axiom_engine.core.evaluate_signal_statistics', side_effect=AssertionError('HIT computed')):
                    hit = evaluate_stock_signal_inputs(override, scope=owner.scope, destination=root/'reports-b')
                self.assertTrue(all(report.reused for report in hit.values()))
                audit_stock_signal_evaluation(hit['a'].path)

    def test_common_override_horizon_independent_and_matches_old_math(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); owner = MockOwner(root/'source')
            def horizon(raw):
                raw['label_spec'].update(label_id='forward_6_session_open_close_v1', horizon_sessions=6, end_session_offset=6)
                for row in raw['rows']:
                    pos = owner.scope['calendar'].index(row['feature_session'])
                    row.update(end_session=owner.scope['calendar'][pos+6],
                               label_available_at=owner.scope['calendar'][pos+6]+'T20:00:00Z')
            target = raw_variant(root, owner.label, horizon)
            legacy_signal, _, _ = fixture(root/'legacy')
            legacy = evaluate_stock_signals({'a': legacy_signal}, raw_label_input=target, scope=owner.scope)['a']
            with owner.patch_owner(), owner.batch() as batch:
                ref = save_stock_signal_evaluation_inputs(owner.inputs, raw_label_input=target,
                    batch=batch, scope=owner.scope, destination=root/'frozen')
            report = evaluate_stock_signal_inputs(ref, scope=owner.scope, destination=root/'reports')['a'].to_dict()
            self.assertEqual(report['label_spec']['horizon_sessions'], 6)
            self.assertEqual([{k: v for k, v in row.items() if k != 'signal_key'} for row in report['series']],
                             [{k: v for k, v in row.items() if k != 'signal_key'} for row in legacy['series']])
            self.assertEqual({k: v for k, v in report['summary'].items() if k != 'signal_key'},
                             {k: v for k, v in legacy['summary'].items() if k != 'signal_key'})

    def test_shared_override_buffer_read_once_and_normalized_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); owner = MockOwner(root/'source', two_signals=True)
            with owner.patch_owner(), owner.batch() as batch:
                admitted = batch._admit_raw_label
                with patch.object(StockMLBatchInputs, '_admit_raw_label', side_effect=admitted) as raw:
                    save_stock_signal_evaluation_inputs(owner.inputs, raw_label_input=owner.label,
                        batch=batch, scope=owner.scope, destination=root/'frozen')
                self.assertEqual(raw.call_count, 1)
                self.assertEqual(raw.call_args.args, (owner.label,))
            target = raw_variant(root, owner.label, lambda raw: raw['label_spec'].update(normalization='cs_zscore'))
            with owner.patch_owner(), owner.batch() as batch, self.assertRaisesRegex(ValueError, 'endpoint semantics'):
                save_stock_signal_evaluation_inputs(owner.inputs, raw_label_input=target,
                    batch=batch, scope=owner.scope, destination=root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_fold_mutation_after_saved_loader_does_not_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); owner = MockOwner(root/'source')
            original = owner.load_fold
            def changed(path, *, batch):
                original(path, batch=batch)
                (Path(path)/'booster.txt').write_text('changed after saved admission')
            with owner.patch_owner(), owner.batch() as batch, \
                 patch('axiom_research.stock_fold_artifacts.load_stock_ml_fold', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    save_stock_signal_evaluation_inputs(owner.inputs, batch=batch,
                        scope=owner.scope, destination=root/'frozen')
            self.assertFalse((root/'frozen').exists()); self.assertEqual(owner.active, 0)

    def test_resealed_invalid_label_and_receipt_byte_binding_are_rejected(self):
        from axiom_research.stock_signal_evaluation_matrix import _label_day_ref
        for mutation in ('invalid_nonnull', 'raw_byte_pin'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); owner = MockOwner(root/'source')
                with owner.patch_owner(), owner.batch() as batch:
                    ref = save_stock_signal_evaluation_inputs(owner.inputs, batch=batch,
                        scope=owner.scope, destination=root/'frozen')
                wire = _read(ref.uri)
                if mutation == 'invalid_nonnull':
                    day = owner.scope['sessions'][0]; path = Path(ref.uri).parent/(day+'.json')
                    shard = _read(path)
                    shard['rows'][0]['label'].update(valid=False, invalid_reason='original-null')
                    write_json(path, shard); wire['shards'][day]['file_digest'] = file_digest(path)
                    wire['raw_metadata']['label_shard_refs'][day] = _label_day_ref(shard)
                    from axiom_research.stock_signal_evaluation_matrix import _label_projection_ref
                    wire['raw_metadata']['label_ref'] = _label_projection_ref(wire['raw_metadata'])
                else:
                    header = next(iter(wire['raw_metadata']['sources'].values()))
                    header['raw_input']['file_digest'] = digest('different original bytes')
                wire['input_id'] = frozen._root_id(wire); write_json(ref.uri, wire)
                altered = replace(ref, artifact_id=wire['input_id'], content_digest=file_digest(ref.uri))
                with self.assertRaises(ValueError):
                    evaluate_stock_signal_inputs(altered, scope=owner.scope, destination=root/'reports')
                self.assertFalse((root/'reports').exists())

    def test_closed_batch_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); owner = MockOwner(root/'source'); batch = owner.batch(); batch.close()
            with owner.patch_owner(), self.assertRaisesRegex(ValueError, 'closed'):
                save_stock_signal_evaluation_inputs(owner.inputs, batch=batch,
                    scope=owner.scope, destination=root/'frozen')
            self.assertFalse((root/'frozen').exists())

    def test_narrow_features_preserve_aware_max_and_original_shard(self):
        from axiom_research.stock_signal_evaluation_matrix import _compact_features
        key = 'S00', '2024-01-01'
        for availability, expected in (
                ([None, '2024-01-01T23:00:00+14:00', '2024-01-01T10:30:00.123456Z'],
                 '2024-01-01T10:30:00.123456Z'),
                ([None, None], None)):
            with self.subTest(availability=availability):
                original = {'member': True, 'knowledge_cutoff': '2024-01-01T20:00:00Z',
                    'availability': availability, 'values': [1.0]*len(availability),
                    'validity': [True]*len(availability), 'reasons': [[]]*len(availability),
                    'source_refs': [digest('original feature')]}
                members, predictions = _compact_features({key: original})
                self.assertEqual(members[key], {'member': True})
                self.assertEqual(set(predictions[key]), {'knowledge_cutoff', 'feature_available_at', 'source_refs'})
                self.assertEqual(predictions[key]['feature_available_at'], expected)
                base = {'labels': {key: {'security_id': key[0], 'feature_session': key[1]}},
                    'projected': {'a': {'rows': {key: {'score': 1.0}},
                        'members': {key: original}, 'prediction_features': {key: original}}}}
                before = frozen._shard(base, {'universe': [key[0]]}, key[1])
                base['projected']['a'].update(members=members, prediction_features=predictions)
                self.assertEqual(before, frozen._shard(base, {'universe': [key[0]]}, key[1]))
                original['source_refs'].append(digest('later mutation'))
                self.assertEqual(predictions[key]['source_refs'], [digest('original feature')])

    def test_wide_rows_and_lease_released_before_next_fold_admission(self):
        class WeakList(list): pass
        class WeakDict(dict): pass
        class Borrowed: pass
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); owner = MockOwner(root/'source', two_signals=True)
            tracked, checked = [], []
            def track(wire):
                wire = WeakDict(wire); tracked.append(weakref.ref(wire))
                rows = []
                for original in wire['rows']:
                    row = WeakDict(original); tracked.append(weakref.ref(row))
                    for field in ('values', 'validity', 'availability', 'reasons'):
                        if field not in row: continue
                        row[field] = WeakList(row[field]); tracked.append(weakref.ref(row[field]))
                    rows.append(row)
                wire['rows'] = rows
                return wire
            original_read = frozen._read_checked
            def read(path, *args, **kwargs):
                wire, ref = original_read(path, *args, **kwargs)
                return (track(wire), ref) if Path(path).name == 'feature-slice.json' else (wire, ref)
            @contextmanager
            def project(batch, inputs, spec):
                view = Borrowed(); view.__dict__.update(vars(owner.views[digest(inputs), digest(spec)]))
                view.features = track(deepcopy(view.features))
                view.source_record_indices = tuple(range(len(owner.table)))
                tracked.append(weakref.ref(view))
                try: yield view
                finally: view.__dict__.clear()
            def load(path, *, batch):
                if tracked:
                    gc.collect()
                    self.assertTrue(all(ref() is None for ref in tracked), 'previous fold retained a wide graph or lease')
                    checked.append(str(path))
                return owner.load_fold(path, batch=batch)
            with owner.patch_owner(), owner.batch() as batch, \
                 patch.object(StockMLBatchInputs, '_project_evaluation', new=project), \
                 patch.object(frozen, '_read_checked', new=read), \
                 patch('axiom_research.stock_fold_artifacts.load_stock_ml_fold', new=load):
                save_stock_signal_evaluation_inputs(owner.inputs, batch=batch,
                    scope=owner.scope, destination=root/'frozen')
            gc.collect()
            self.assertEqual(len(checked), 1)
            self.assertTrue(all(ref() is None for ref in tracked))

    def test_narrow_admission_exact_shards_refs_masks_and_reports(self):
        from axiom_research import stock_signal_evaluation_matrix as matrix
        from axiom_engine.core import evaluate_signal_statistics
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); owner = MockOwner(root/'source', two_signals=True)
            captured = []; original_shard = frozen._shard
            def shard(*args, **kwargs):
                wire = original_shard(*args, **kwargs); captured.append(deepcopy(wire)); return wire
            def wide(indexed): return dict(indexed), dict(indexed)
            with owner.patch_owner(), owner.batch() as batch, patch.object(frozen, '_shard', new=shard):
                with patch.object(matrix, '_compact_features', new=wide):
                    before = save_stock_signal_evaluation_inputs(owner.inputs, batch=batch,
                        scope=owner.scope, destination=root/'frozen')
                old_shards = captured[:]; captured.clear()
                after = save_stock_signal_evaluation_inputs(owner.inputs, batch=batch,
                    scope=owner.scope, destination=root/'frozen')
                self.assertEqual(captured, old_shards); self.assertEqual(after, before)
            old_selected = frozen._load_inputs(before, owner.scope)[2]
            new_selected = frozen._load_inputs(after, owner.scope)[2]
            self.assertEqual(new_selected, old_selected)
            with patch('axiom_engine.core.evaluate_signal_statistics', wraps=evaluate_signal_statistics) as core:
                old_reports = evaluate_stock_signal_inputs(before, scope=owner.scope, destination=root/'old-reports')
                new_reports = evaluate_stock_signal_inputs(after, scope=owner.scope, destination=root/'new-reports')
                self.assertEqual([call.args for call in core.call_args_list[:2]],
                    [call.args for call in core.call_args_list[2:]])
            self.assertEqual({name: report.to_dict() for name, report in new_reports.items()},
                {name: report.to_dict() for name, report in old_reports.items()})


if __name__ == '__main__':
    unittest.main()

"""Small saved-output fixtures exercise the actual admission and publish chain."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest.mock import patch

from axiom_research import (ExperimentReader, ExperimentRecordError, ExperimentStore,
    begin_stock_result_registration, publish_stock_result)
from axiom_research.result_registration import _save_stock_sequential_receipt
from axiom_research.stock_artifacts import digest, write_json
from axiom_research import stock_result_projection as projection
from axiom_research.stock_matrix_folds import _predictions
from test_result_metadata_budget import saved_output_fixture


def multi_fold_fixture(root, count):
    results = []
    for minute in range(count):
        part = root/str(minute); part.mkdir()
        cfg, result = saved_output_fixture(part, padding_bytes=32, cutoff_minute=minute)
        results.append(result)
    cfg = deepcopy(cfg)
    specs = [result['batch_manifest']['definition']['fold_specs'][0] for result in results]
    cfg['dataset']['fold_specs'] = specs
    cfg['dataset_ref'] = digest(cfg['dataset'])
    cfg['configuration_ref'] = digest(dict(contract_version='stock_sequential_effective_configuration_v1',
        specification_ref=cfg['specification']['configuration_ref'], dataset_ref=cfg['dataset_ref']))
    result = deepcopy(results[0]); batch = result['batch_manifest']
    batch['definition']['fold_specs'] = specs
    batch['definition_ref'] = digest(batch['definition'])
    batch['folds'] = [item['batch_manifest']['folds'][0] for item in results]
    batch['batch_ref'] = digest({key: value for key, value in batch.items()
                                if key not in ('batch_ref', 'content_digest')})
    batch['content_digest'] = digest({key: value for key, value in batch.items() if key != 'content_digest'})
    result.update(configuration_ref=cfg['configuration_ref'], dataset_configuration_ref=cfg['dataset_ref'],
        folds=[item['folds'][0] for item in results],
        prediction_bindings=[item['prediction_bindings'][0] for item in results])
    return cfg, result


class ResultProjectionReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        cfg, self.result = saved_output_fixture(self.root, padding_bytes=32)
        self.receipt = _save_stock_sequential_receipt(self.result, destination=self.root)
        self.folder = Path(self.result['folds'][0]['path'])
        self.context = dict(request_path=self.root/'request.json', index_path=self.root/'index.json',
                            roots={'research': self.root})
        begin_stock_result_registration(cfg, **self.context, required_stages=['training'])

    def test_publish_reads_hashes_parses_and_validates_each_fold_once(self):
        reads = Counter()
        original = Path.open
        def counted(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get('mode', 'r')
            if 'r' in mode:
                reads[path.resolve()] += 1
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', counted), patch(
            'axiom_research.stock_matrix_folds._predictions', wraps=_predictions) as validate:
            head = publish_stock_result(**self.context, experiment_result=self.receipt)
        self.assertEqual(head['status'], 'COMPLETE')
        self.assertEqual(validate.call_count, 1)
        for path in [*self.folder.iterdir(), *self.receipt.parent.iterdir(), self.context['request_path']]:
            self.assertEqual(reads[path.resolve()], 1, path.name)
        reader = ExperimentReader(self.context['index_path'])
        manifest = reader.result(head['result_id'], roots=self.context['roots'])
        # The normal stored-manifest publish entry starts cold and deduplicates too.
        with patch('axiom_research.stock_matrix_folds._predictions', wraps=_predictions) as validate:
            self.assertEqual(ExperimentStore(self.context['index_path']).publish_result(
                manifest, roots=self.context['roots']), head)
        self.assertEqual(validate.call_count, 1)

    def test_root_binding_and_batch_checks_are_not_borrowed_from_path_hit(self):
        binding = self.result['prediction_bindings'][0]
        item = self.result['batch_manifest']['folds'][0]
        with projection._projection_session(), patch(
            'axiom_research.stock_matrix_folds._predictions', wraps=_predictions) as validate:
            original = projection.load_stock_fold_projection(self.folder, research_root=self.root,
                batch_item=item, binding=binding)
            bad = deepcopy(binding); bad['model_ref'] = digest('other model')
            with self.assertRaisesRegex(ExperimentRecordError, 'frame/fold mismatch'):
                projection.load_stock_fold_projection(self.folder, research_root=self.root, binding=bad)
            bad = deepcopy(binding); bad['prediction_artifact']['content_digest'] = digest('other bytes')
            with self.assertRaisesRegex(ExperimentRecordError, 'frame output artifact mismatch'):
                projection.load_stock_fold_projection(self.folder, research_root=self.root, binding=bad)
            with self.assertRaisesRegex(ExperimentRecordError, 'fold/batch mismatch'):
                projection.load_stock_fold_projection(self.folder, research_root=self.root,
                    batch_item={**item, 'fold_spec': {'other': True}})
            # A separately trusted enclosing root gets its own admission.
            projection.load_stock_fold_projection(self.folder, research_root=self.folder)
            self.assertEqual(validate.call_count, 2)
            other = self.root/'other-root'; other.mkdir()
            with self.assertRaisesRegex(ExperimentRecordError, 'escapes Research root'):
                projection.load_stock_fold_projection(self.folder, research_root=other)
            original.wire['definition']['fold_spec'] = {'caller mutation': True}
            self.assertEqual(projection.load_stock_fold_projection(self.folder,
                research_root=self.root).wire['definition']['fold_spec'], item['fold_spec'])
            prediction = projection.load_stock_prediction_projection(self.folder/'predictions.json',
                research_root=self.root)
            prediction.wire['rows'][0]['score'] = 999
            self.assertEqual(projection.load_stock_prediction_projection(self.folder/'predictions.json',
                research_root=self.root).wire['rows'][0]['score'], .25)

    def test_cold_loads_do_not_trust_saved_flags_and_mutation_invalidates_session(self):
        with patch('axiom_research.stock_matrix_folds._predictions', wraps=_predictions) as validate:
            projection.load_stock_sequential_result(self.receipt, research_root=self.root)
            projection.load_stock_sequential_result(self.receipt, research_root=self.root)
        self.assertEqual(validate.call_count, 2)
        target = self.folder/'predictions.json'
        payload = target.read_bytes()
        with self.assertRaisesRegex(ExperimentRecordError, 'changed during projection session'):
            with projection._projection_session():
                projection.load_stock_fold_projection(self.folder, research_root=self.root)
                target.write_bytes(payload.replace(b'0.25', b'0.26'))
                projection.load_stock_prediction_projection(target, research_root=self.root)
        # Even adding an alleged trust flag cannot bypass the cold byte pins.
        target.write_bytes(payload[:-2] + b',"verified":true}\n')
        with self.assertRaisesRegex(ExperimentRecordError, 'hash mismatch'):
            projection.load_stock_sequential_result(self.receipt, research_root=self.root)

    def test_nested_mutation_before_commit_keeps_old_head_and_index_bytes(self):
        head = publish_stock_result(**self.context, experiment_result=self.receipt)
        before = self.context['index_path'].read_bytes(), self.context['index_path'].stat().st_mtime_ns
        original = ExperimentStore._mutate
        target = self.folder/'feature-slice.json'  # Not a typed manifest ref.
        def mutate(store, operation):
            target.write_bytes(target.read_bytes() + b' ')
            return original(store, operation)
        with patch.object(ExperimentStore, '_mutate', mutate), self.assertRaisesRegex(
            ExperimentRecordError, 'changed during projection session'):
            publish_stock_result(**self.context, failure='new owner stage failed')
        self.assertEqual((self.context['index_path'].read_bytes(),
                          self.context['index_path'].stat().st_mtime_ns), before)
        self.assertEqual(ExperimentReader(self.context['index_path']).index()['result_heads'], [head])
        self.assertEqual(len(list((self.root/'results'/head['result_id'][7:]).glob('*.json'))), 1)

    def test_nested_symlink_retarget_cannot_borrow_an_admitted_snapshot(self):
        target = self.folder/'model.json'
        original = self.folder/'original-model.json'; target.rename(original)
        replacement = self.folder/'replacement-model.json'; replacement.write_bytes(original.read_bytes())
        target.symlink_to(original)
        with self.assertRaisesRegex(ExperimentRecordError, 'locator changed during projection session'):
            with projection._projection_session():
                projection.load_stock_fold_projection(self.folder, research_root=self.root)
                target.unlink(); target.symlink_to(replacement)
                projection.load_stock_fold_projection(self.folder, research_root=self.root)

    def test_post_commit_change_does_not_report_a_successful_commit_as_failure(self):
        original = ExperimentStore._mutate
        target = self.folder/'feature-slice.json'
        def mutate_after_commit(store, operation):
            head = original(store, operation)
            target.write_bytes(target.read_bytes() + b' ')
            return head
        with patch.object(ExperimentStore, '_mutate', mutate_after_commit):
            head = publish_stock_result(**self.context, experiment_result=self.receipt)
        self.assertEqual(ExperimentReader(self.context['index_path']).index()['result_heads'], [head])
        with self.assertRaisesRegex(ExperimentRecordError, 'hash mismatch'):
            projection.load_stock_fold_projection(self.folder, research_root=self.root)

    def test_derived_bindings_keep_their_owner_parent_validation(self):
        from test_derived_result_registration import saved_combination, plan
        from axiom_research.stock_derived_signal import load_stock_derived_signal
        derived = saved_combination(self.root, 'derived', plan())
        binding = load_stock_derived_signal(derived, research_root=self.root).engine_input_binding()
        path = self.root/'derived-bindings.json'; write_json(path, [binding])
        with projection._projection_session():
            projection.load_stock_prediction_bindings(path, research_root=self.root)
            target = derived/'a'/'model.json'; target.write_bytes(target.read_bytes() + b' ')
            with self.assertRaises(ValueError):
                projection.load_stock_prediction_bindings(path, research_root=self.root)

    def test_multi_fold_registration_releases_rows_before_the_next_fold(self):
        root = self.root/'multiple'; root.mkdir()
        cfg, result = multi_fold_fixture(root, 4)
        receipt = _save_stock_sequential_receipt(result, destination=root)
        context = dict(request_path=root/'request.json', index_path=root/'index.json', roots={'research':root})
        begin_stock_result_registration(cfg, **context, required_stages=['training'])
        row_arrays = []; rows = []; alive_at_ingress = []; alive_after_fold = []
        class TrackedRows(list):
            pass
        class TrackedRow(dict):
            pass
        checked = projection._checked; fold_outputs = projection._fold_outputs
        def track(path, *args, **kwargs):
            value = checked(path, *args, **kwargs)
            if Path(path).name == 'predictions.json':
                value['rows'] = TrackedRows(TrackedRow(row) for row in value['rows'])
                row_arrays.append(weakref.ref(value['rows']))
                rows.extend(weakref.ref(row) for row in value['rows'])
                alive_at_ingress.append(sum(ref() is not None for ref in row_arrays))
            return value
        def after_fold(*args, **kwargs):
            saved = fold_outputs(*args, **kwargs)
            alive_after_fold.append((sum(ref() is not None for ref in row_arrays),
                                     sum(ref() is not None for ref in rows)))
            return saved
        with patch.object(projection, '_checked', track), patch.object(projection, '_fold_outputs', after_fold), \
             patch.object(projection, 'load_stock_prediction_projection',
                          side_effect=AssertionError('registration requested complete rows')):
            head = publish_stock_result(**context, experiment_result=receipt)
        self.assertEqual(head['status'], 'COMPLETE')
        self.assertEqual(alive_at_ingress, [1]*4)
        self.assertEqual(alive_after_fold, [(0, 0)]*4)
        self.assertTrue(all(ref() is None for ref in [*row_arrays, *rows]))

    def test_public_prediction_read_returns_all_rows_with_only_one_extra_leaf_read(self):
        calls = Counter(); checked = projection._checked
        def count(path, *args, **kwargs):
            calls[Path(path).name] += 1
            return checked(path, *args, **kwargs)
        with patch.object(projection, '_checked', count), patch(
            'axiom_research.stock_matrix_folds._predictions', wraps=_predictions) as validate:
            saved = projection.load_stock_prediction_projection(self.folder/'predictions.json', research_root=self.root)
        self.assertEqual(validate.call_count, 1)
        self.assertEqual(calls['predictions.json'], 2)
        self.assertEqual(saved.wire['rows'][0]['score'], .25)
        for name, count in calls.items():
            if name != 'predictions.json':
                self.assertEqual(count, 1, name)


if __name__ == '__main__':
    unittest.main()

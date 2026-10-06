"""Small synthetic frozen evaluation fixtures; no provider or real fitting."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import (save_stock_signal_evaluation_inputs, evaluate_stock_signal_inputs,
    audit_stock_signal_evaluation, evaluate_stock_signals, load_stock_signal_evaluation,
    save_stock_signal_evaluation, semantic_identity)
from axiom_research.stock_artifacts import _read, write_json, file_digest
from axiom_research import stock_signal_evaluation_projection as projection
from test_stock_signal_evaluation import fixture, mutate_signal, raw_variant


class FrozenSignalEvaluationTests(unittest.TestCase):
    def freeze(self, root, inputs, label, scope):
        return save_stock_signal_evaluation_inputs(inputs, raw_label_input=label, scope=scope,
                                                  destination=root/'inputs')

    def compare(self, legacy, frozen):
        for field in ('input_signal_refs', 'label_ref', 'label_spec', 'scope', 'spec', 'spec_ref',
                      'sample_mask_ref', 'statistics_input_ref', 'statistics_ref', 'series', 'summary', 'status'):
            self.assertEqual(legacy[field], frozen[field], field)
        coverage = {k: v for k, v in legacy['coverage'].items()
                    if k not in ('common_statistics', 'native_statistics')}
        self.assertEqual(coverage, {k: v for k, v in frozen['coverage'].items()
                                   if k not in ('common_statistics_ref', 'native_statistics_ref')})

    def test_exact_core_inputs_common_native_and_zero_compute_hit(self):
        from axiom_engine.core import evaluate_signal_statistics
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a, label, scope = fixture(root/'a', ties=True)
            invalid = [(s, d) for s in scope['universe'][:5] for d in scope['sessions']]
            b, _, _ = fixture(root/'b', reverse=True, invalid=invalid)
            inputs = {'z': a, 'a': b}
            with patch('axiom_engine.core.evaluate_signal_statistics', wraps=evaluate_signal_statistics) as stats:
                original = evaluate_stock_signals(inputs, raw_label_input=label, scope=scope)
                core_inputs = [deepcopy(c.args[0]) for c in stats.call_args_list]
            with patch('axiom_engine.core.evaluate_signal_statistics', side_effect=AssertionError('freeze computed statistics')):
                ref = self.freeze(root, inputs, label, scope)
            with patch('axiom_research.stock_signal_evaluation_projection._admit_inputs',
                    side_effect=AssertionError('daily load read ancestors')):
                with patch('axiom_engine.core.evaluate_signal_statistics', wraps=evaluate_signal_statistics) as stats:
                    reports = evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'reports')
                    self.assertEqual([c.args[0] for c in stats.call_args_list], core_inputs)
            self.assertEqual(list(reports), ['z', 'a'])
            for name in reports:
                self.compare(original[name], reports[name].to_dict())
                self.assertFalse(reports[name].reused)
            reads = []
            original_read = projection._read_checked
            def read(path, *args, **kwargs):
                reads.append(str(path)); return original_read(path, *args, **kwargs)
            with patch('axiom_engine.core.evaluate_signal_statistics', side_effect=AssertionError('HIT recomputed')), \
                 patch.object(projection, '_read_checked', side_effect=read):
                hit = evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'reports')
            self.assertTrue(all(saved.reused for saved in hit.values()))
            self.assertEqual(reads.count(ref.uri), 1)
            for day in scope['sessions']:
                self.assertEqual(reads.count(str(Path(ref.uri).parent/(day+'.json'))), 1)
            audit_stock_signal_evaluation(hit['z'].path)

    def test_numerical_boundary_cases_exact_same_core_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, options in [('constant', {'constant': True}), ('ties', {'ties': True}),
                                  ('normal', {}), ('extremes', {})]:
                with self.subTest(name=name):
                    p = root/name; a, label, scope = fixture(p/'source', **options)
                    if name == 'extremes':
                        mutate_signal(a, lambda wire: [r.update(score=(-1e308 if i % 2 else 1e308))
                                                      for i, r in enumerate(wire['rows'])])
                    for index, days in enumerate([scope['sessions'], scope['sessions'][:1]]):
                        selected = {**scope, 'sessions': days}
                        original = evaluate_stock_signals({'signal': a}, raw_label_input=label, scope=selected)
                        ref = self.freeze(p/str(index), {'signal': a}, label, scope)
                        report = evaluate_stock_signal_inputs(ref, scope=selected, destination=p/str(index)/'reports')
                        self.compare(original['signal'], report['signal'].to_dict())

    def test_scope_revision_visibility_and_missing_predictions_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a, label, scope = fixture(root/'source', dates=None)
            inputs = {'signal': a}; ref = self.freeze(root, inputs, label, scope)
            selected = {**scope, 'sessions': scope['sessions'][:2], 'universe': scope['universe'][:21]}
            old = evaluate_stock_signals(inputs, raw_label_input=label, scope=selected)['signal']
            saved = evaluate_stock_signal_inputs(ref, scope=selected, destination=root/'reports')['signal']
            self.compare(old, saved.to_dict())
            early = {**scope, 'evaluation_cutoff': scope['calendar'][-1]+'T22:59:59Z'}
            with self.assertRaisesRegex(ValueError, 'revision/Signal visibility'):
                evaluate_stock_signal_inputs(ref, scope=early, destination=root/'reports')
            for invalid in [{**scope, 'universe': scope['universe']+['Z']},
                            {**scope, 'calendar': scope['calendar'][:-1]}]:
                with self.assertRaises(ValueError):
                    evaluate_stock_signal_inputs(ref, scope=invalid, destination=root/'reports')
            # A later evaluation clock does not fill a saved missing prediction.
            partial, _, _ = fixture(root/'partial', dates=scope['sessions'][:2])
            ref2 = self.freeze(root/'partial-input', {'signal': partial}, label, scope)
            old = evaluate_stock_signals({'signal': partial}, raw_label_input=label, scope=scope)['signal']
            saved = evaluate_stock_signal_inputs(ref2, scope=scope, destination=root/'reports')['signal']
            self.compare(old, saved.to_dict())
            self.assertEqual(saved.to_dict()['coverage']['excluded_counts']['MEMBERSHIP_UNKNOWN'], 48)
            known = {'full': a, 'partial': partial}
            ref3 = self.freeze(root/'known-member-input', known, label, scope)
            old = evaluate_stock_signals(known, raw_label_input=label, scope=scope)['partial']
            saved = evaluate_stock_signal_inputs(ref3, scope=scope, destination=root/'reports')['partial']
            self.compare(old, saved.to_dict())
            self.assertEqual(saved.to_dict()['coverage']['excluded_counts']['SIGNAL_MISSING'], 48)

    def test_source_move_changes_audit_and_old_v2_keeps_full_closure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a, label, scope = fixture(root/'source')
            old = evaluate_stock_signals({'signal': a}, raw_label_input=label, scope=scope)['signal']
            legacy = save_stock_signal_evaluation(old, destination=root/'legacy')
            ref = self.freeze(root, {'signal': a}, label, scope)
            saved = evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'reports')['signal']
            shutil.rmtree(root/'source')
            self.assertEqual(load_stock_signal_evaluation(saved.path).to_dict(), saved.to_dict())
            self.assertTrue(evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'reports')['signal'].reused)
            with self.assertRaises(FileNotFoundError):
                audit_stock_signal_evaluation(saved.path)
            with self.assertRaises(FileNotFoundError):
                load_stock_signal_evaluation(legacy.path)

    def test_content_mutation_missing_shard_and_same_byte_consumption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a, label, scope = fixture(root/'source')
            ref = self.freeze(root, {'signal': a}, label, scope)
            shard = Path(ref.uri).parent/(scope['sessions'][0]+'.json')
            original = shard.read_bytes()
            shard.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'reports')
            shard.write_bytes(original)
            real_json = projection.json.loads
            def mutate_after_read(payload, *args, **kwargs):
                if b'"stock_signal_evaluation_date_v1"' in payload:
                    shard.write_bytes(original+b' ')
                return real_json(payload, *args, **kwargs)
            with patch.object(projection.json, 'loads', side_effect=mutate_after_read):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'reports')
            shard.unlink()
            with self.assertRaises(FileNotFoundError):
                evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'reports')
            self.assertFalse((root/'reports').exists())

    def test_handle_consumes_verified_snapshot_and_new_load_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a, label, scope = fixture(root/'source')
            ref = self.freeze(root, {'signal': a}, label, scope)
            saved = evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'reports')['signal']
            loaded = load_stock_signal_evaluation(saved.path); original = loaded.to_dict()
            changed = loaded.to_dict(); changed['series'].clear()
            saved.path.write_text('{}')
            self.assertEqual(loaded.to_dict(), original)
            with self.assertRaises(ValueError):
                load_stock_signal_evaluation(saved.path)

    def test_input_exact_hit_and_locator_is_not_artifact_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a, label, scope = fixture(root/'source')
            ref = self.freeze(root, {'signal': a}, label, scope)
            self.assertEqual(ref, self.freeze(root, {'signal': a}, label, scope))
            moved = root/'moved'; shutil.copytree(Path(ref.uri).parent, moved)
            relocated = replace(ref, uri=str(moved/'manifest.json'), metadata={'title': 'moved'})
            self.assertEqual(semantic_identity(ref), semantic_identity(relocated))
            original = evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'reports')['signal']
            self.assertTrue(evaluate_stock_signal_inputs(relocated, scope=scope, destination=root/'reports')['signal'].reused)
            # Locator relocation changes the saved locator, but computation ID stays fixed.
            other = evaluate_stock_signal_inputs(relocated, scope=scope, destination=root/'other')['signal']
            self.assertEqual(original.identity, other.identity)

    def test_group_missing_output_fails_without_core_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a, label, scope = fixture(root/'source')
            ref = self.freeze(root, {'signal': a}, label, scope)
            with patch.object(projection.os, 'rename', side_effect=OSError('synthetic publication failure')):
                with self.assertRaisesRegex(OSError, 'publication failure'):
                    evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'failed')
            self.assertEqual(list((root/'failed').iterdir()), [])
            saved = evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'reports')['signal']
            (saved.path.parent/'native-statistics.json').unlink()
            with patch('axiom_engine.core.evaluate_signal_statistics', side_effect=AssertionError('incomplete HIT computed')):
                with self.assertRaises(FileNotFoundError):
                    evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'reports')

    def test_publication_race_verifies_complete_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a, label, scope = fixture(root/'source')
            ref = self.freeze(root, {'signal': a}, label, scope)
            rename = os.rename
            def race(source, target):
                rename(source, target); raise OSError('winner already published')
            with patch.object(projection.os, 'rename', side_effect=race):
                saved = evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'reports')['signal']
            self.assertTrue(saved.reused)
            self.assertEqual(load_stock_signal_evaluation(saved.path).identity, saved.identity)

    def test_source_mutation_during_freezing_stops_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a, label, scope = fixture(root/'source')
            original_admit = projection._admit_inputs
            def changed(*args, **kwargs):
                admitted = original_admit(*args, **kwargs)
                (root/'source'/'booster.txt').write_text('changed after admission')
                return admitted
            with patch.object(projection, '_admit_inputs', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    self.freeze(root, {'signal': a}, label, scope)
            self.assertFalse((root/'inputs').exists())

    def test_source_and_raw_label_revisions_miss_without_rewriting_old_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a, label, scope = fixture(root/'source')
            ref = self.freeze(root, {'signal': a}, label, scope)
            old = evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'reports')['signal']
            old_bytes = Path(ref.uri).read_bytes()
            mutate_signal(a, lambda wire: wire['rows'][0].update(score=50.0))
            revised = self.freeze(root, {'signal': a}, label, scope)
            new = evaluate_stock_signal_inputs(revised, scope=scope, destination=root/'reports')['signal']
            self.assertFalse(new.reused); self.assertNotEqual(old.identity, new.identity)
            self.assertEqual(Path(ref.uri).read_bytes(), old_bytes)
            self.assertEqual(load_stock_signal_evaluation(old.path).to_dict(), old.to_dict())
            with self.assertRaisesRegex(ValueError, 'audit original source mismatch'):
                audit_stock_signal_evaluation(old.path)
            raw = raw_variant(root, label, lambda wire: wire['rows'][0].update({'return': 0.123}))
            changed = self.freeze(root, {'signal': a}, raw, scope)
            latest = evaluate_stock_signal_inputs(changed, scope=scope, destination=root/'reports')['signal']
            self.assertNotEqual(new.identity, latest.identity); self.assertFalse(latest.reused)

    def test_freeze_rejects_original_bad_clocks_and_duplicate_signal_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a, label, scope = fixture(root/'source')
            with self.assertRaisesRegex(ValueError, 'duplicate comparison Signal identity'):
                self.freeze(root, {'one': a, 'two': a}, label, scope)
            mutate_signal(a, lambda wire: wire['rows'][0].update(available_at='2099-01-01T00:00:00Z'))
            with self.assertRaisesRegex(ValueError, 'clock conflict'):
                self.freeze(root, {'signal': a}, label, scope)
            self.assertFalse((root/'inputs').exists())

    def test_fresh_stdlib_loader_no_runtime_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a, label, scope = fixture(root/'source')
            ref = self.freeze(root, {'signal': a}, label, scope)
            saved = evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'reports')['signal']
            script = '''import builtins,sys
original=builtins.__import__
def guard(name,*a,**kw):
    if name.split('.')[0] in {'axiom_engine','axiom_data','qlib','lightgbm','numpy','pandas','pyarrow'}:
        raise AssertionError('runtime import '+name)
    return original(name,*a,**kw)
builtins.__import__=guard
from axiom_research import load_stock_signal_evaluation
print(load_stock_signal_evaluation(sys.argv[1]).identity)
'''
            run = subprocess.run([sys.executable, '-c', script, str(saved.path)], env=os.environ,
                capture_output=True, text=True, check=True, timeout=15)
            self.assertEqual(run.stdout.strip(), saved.identity)

    def test_original_and_compact_v2_fold_projection_exact(self):
        import test_stock_signal_evaluation as original_tests
        original_evaluate = original_tests.evaluate_stock_signal
        def compare_fold(signal, *, raw_label_input, scope):
            legacy = original_evaluate(signal, raw_label_input=raw_label_input, scope=scope)
            root = Path(signal['path']).parent.parent/'projection-tests'
            ref = self.freeze(root, {'signal': signal}, raw_label_input, scope)
            saved = evaluate_stock_signal_inputs(ref, scope=scope, destination=root/'reports')['signal']
            self.compare(legacy, saved.to_dict()); audit_stock_signal_evaluation(saved.path)
            return legacy
        owner = original_tests.SavedSignalEvaluationTests()
        with patch.object(original_tests, 'evaluate_stock_signal', side_effect=compare_fold):
            owner._saved_fold_source_closure(compact=False)
            owner._saved_fold_source_closure(compact=True)


if __name__ == '__main__':
    unittest.main()

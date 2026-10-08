"""Independent outcome production; artificial Data columns and real Core."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import build_stock_evaluation_label_inputs, load_stock_evaluation_label_inputs
from axiom_research.stock_artifacts import file_digest, write_json
from axiom_research.stock_compact_store import OwnedStore
from axiom_research.stock_evaluation_labels import _parts, _read_manifest
from test_stock_column_asof import ColumnSource
from test_stock_compact_v3 import CompactV3Tests
from test_stock_target_config_blocks import label


class EvaluationLabelTests(unittest.TestCase):
    def inputs(self, root):
        fixture, _ = CompactV3Tests().fixture(root)
        spec = fixture.spec
        scope = {'calendar': spec['calendar'], 'universe': spec['universe'],
                 'sessions': spec['feature_sessions'][-8:-5],
                 'evaluation_cutoff': spec['calendar'][-1]+'T20:30:00+08:00'}
        return spec, scope, ColumnSource(spec)

    def build(self, root, spec, scope, source, horizon=5, limits=None):
        return build_stock_evaluation_label_inputs(object(), snapshot=spec['snapshot'],
            pit_policy=spec['pit_policy'], label_spec=label(horizon), scope=scope,
            column_source=source, destination=root/'evaluation-labels', limits=limits)

    def test_saved_raw_matches_existing_core_and_warm_has_no_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); spec, scope, source = self.inputs(root)
            with patch('axiom_research.stock_compact_labels._normalized', side_effect=AssertionError('no normalization')):
                descriptor = self.build(root, spec, scope, source)
            self.assertEqual((source.select_calls, source.adjust_calls), (2, 1))
            manifest = load_stock_evaluation_label_inputs(descriptor)
            self.assertEqual(manifest['definition']['scope']['sessions'], scope['sessions'])
            with OwnedStore() as store:
                saved = _read_manifest(descriptor, store)
                rows = [row for _, _, part, _ in _parts(saved, store) for row in part]
            self.assertEqual(len(rows), len(scope['sessions'])*len(scope['universe']))
            for row in rows:
                start = spec['calendar'].index(row['start_session'])
                end = spec['calendar'].index(row['end_session'])
                i = spec['universe'].index(row['security_id'])
                self.assertAlmostEqual(row['return'], (11+end*.01+i*i*.4)/(10+start*.01)-1)
                self.assertTrue(row['valid'])
            with patch.object(source, 'select', side_effect=AssertionError('warm selected Data')):
                self.assertEqual(self.build(root, spec, scope, source), descriptor)
            self.assertTrue(all(s.closed for s in source.selections))

    def test_endpoints_after_evaluation_range_remain_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); spec, scope, source = self.inputs(root)
            descriptor = self.build(root, spec, scope, source, horizon=3)
            with OwnedStore() as store:
                manifest = _read_manifest(descriptor, store)
                rows = [r for _, _, part, _ in _parts(manifest, store) for r in part]
            self.assertTrue(all(r['valid'] and r['end_session'] > scope['sessions'][-1] for r in rows))

    def test_calendar_tail_and_late_cutoff_keep_invalid_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); spec, scope, source = self.inputs(root)
            scope['sessions'] = spec['calendar'][-2:]
            descriptor = self.build(root, spec, scope, source)
            with OwnedStore() as store:
                manifest = _read_manifest(descriptor, store)
                rows = [r for _, _, part, _ in _parts(manifest, store) for r in part]
            self.assertEqual(len(rows), 2*len(scope['universe']))
            self.assertTrue(all(not r['valid'] and r['return'] is None and r['invalid_reason'] for r in rows))
            scope['sessions'] = spec['feature_sessions'][-8:-5]
            scope['evaluation_cutoff'] = scope['sessions'][-1]+'T20:30:00+08:00'
            descriptor = self.build(root, spec, scope, source)
            with OwnedStore() as store:
                manifest = _read_manifest(descriptor, store)
                rows = [r for _, _, part, _ in _parts(manifest, store) for r in part]
            self.assertTrue(all(not r['valid'] for r in rows))

    def test_corrupt_cached_buffer_rejects_without_source_or_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); spec, scope, source = self.inputs(root)
            descriptor = self.build(root, spec, scope, source)
            manifest = load_stock_evaluation_label_inputs(descriptor)
            from axiom_research.stock_artifacts import _read
            header = _read(manifest['raw_parts'][0]['path'])
            path = Path(header['buffers']['values']['path'])
            path.write_bytes(b'X'*path.stat().st_size)
            with patch.object(source, 'select', side_effect=AssertionError('corruption triggered rebuild')):
                with self.assertRaises(ValueError): self.build(root, spec, scope, source)
                with self.assertRaises(ValueError): load_stock_evaluation_label_inputs(descriptor)

    def test_scope_manifest_tamper_and_tiny_budget_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); spec, scope, source = self.inputs(root)
            with self.assertRaisesRegex(ValueError, 'byte budget'):
                self.build(root, spec, scope, source, limits={'maximum_matrix_bytes': 1})
            self.assertEqual(source.select_calls, 0)
            descriptor = self.build(root, spec, scope, source)
            manifest = load_stock_evaluation_label_inputs(descriptor)
            manifest['definition']['scope']['sessions'] = manifest['definition']['scope']['sessions'][:-1]
            path = Path(descriptor['path']); write_json(path, manifest)
            bad = {**descriptor, 'file_digest': file_digest(path)}
            with self.assertRaises(ValueError): load_stock_evaluation_label_inputs(bad)


if __name__ == '__main__':
    unittest.main()

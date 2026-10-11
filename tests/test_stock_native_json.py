"""Small native-hash goldens; no Data, numeric runtime or coverage DOM reader."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_research.stock_native_json import (
    make_native_carrier, native_digest, validate_carrier_shape, verify_carrier_native,
)


WORKSPACE = 2*1024*1024
ABSENT = object()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def old_digest(value, exclude=None):
    if exclude is not None:
        value = {key: child for key, child in value.items() if key != exclude}
    return 'sha256:'+sha256(canonical(value)).hexdigest()


def seal(value, ref_key):
    value[ref_key] = old_digest(value, ref_key)
    return value


def raw(coverage=ABSENT):
    context = {'query': {'sessions': ['2026-01-02'], 'fields': ['close']}, 'reader_version': 'reader/1'}
    if coverage is not ABSENT:
        context['coverage'] = coverage
    return seal({'contract_version': 'stock_label_build_v1', 'label_spec': {'horizon': 1},
                 'source_evidence': {'context': context, 'records_ref': old_digest([])},
                 'rows': [{'score': -0.0, 'text': '甲\n"\\𝄞'}]}, 'label_ref')


def label(coverage):
    selected = {'context': {'domain': 'market_daily', 'coverage': coverage},
                'records': [{'close': 12.5, 'session': '2026-01-02'}], 'field_meta': {}}
    return seal({'contract_version': 'stock_matrix_label_metadata_v1',
                 'contents': {old_digest(selected): selected, old_digest({'query': 1}): {'query': 1}},
                 'generation': 'raw'}, 'metadata_ref')


def feature(coverage):
    source_id = old_digest({'source': 1})
    versions = {source_id: {'field': 'close', 'batch_ref': old_digest({'batch': 1}),
                           'query_context': {'domain': 'market_daily', 'coverage': coverage}}}
    return seal({'contract_version': 'stock_matrix_feature_metadata_v1',
                 'input_evidence': [{'core_plan': {'sources': [{'id': source_id}]},
                                     'source_evidence': versions}],
                 'contents': {old_digest(versions): deepcopy(versions)}, 'rows': []}, 'metadata_ref')


def reseal_carrier(carrier):
    carrier['carrier_ref'] = old_digest(carrier, 'carrier_ref')
    return carrier


class NativeCarrierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='axiom-native-json-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def write(self, value, ref_key, **kwargs):
        arguments = dict(maximum_source_bytes=4*1024*1024, maximum_parent_bytes=64*1024,
                         maximum_workspace_bytes=WORKSPACE)
        arguments.update(kwargs)
        desc = make_native_carrier(self.root, value, ref_key, **arguments)
        carrier = None if desc is None else json.loads(Path(desc['path']).read_bytes())
        return desc, carrier

    def admit_and_verify(self, carrier, ref_key, **kwargs):
        # A public Store must do shared syntax admission first. Here bytes are
        # trusted test writer outputs; shape and identity replay remain real.
        bindings = validate_carrier_shape(carrier, ref_key, maximum_workspace_bytes=WORKSPACE)
        verify_carrier_native(carrier, ref_key, coverage_bindings=bindings,
                              maximum_workspace_bytes=WORKSPACE, **kwargs)
        return bindings

    def test_native_digest_exact_original_scalars_strings_and_arrays(self):
        values = [None, True, False, 0, -12345678901234567890, -0.0, 1e-20,
                  {'𝄞': [1, {'甲': 'a\b\f\n\r\t\u0000"\\/'}], 'a': [], 'z': 1.5},
                  '甲𝄞'*2000]
        for value in values:
            with self.subTest(type=type(value)):
                self.assertEqual(native_digest(value, maximum_workspace_bytes=WORKSPACE), old_digest(value))

    def test_budgeted_ordinary_subtrees_use_exact_stdlib_bytes(self):
        metrics={}
        value={'mixed':[True,1,-0.0,0.0,None,'甲𝄞\n"\\'],
               'rows':[{'k':i,'v':i/7} for i in range(40)]}
        self.assertEqual(native_digest(value,maximum_workspace_bytes=WORKSPACE,metrics=metrics),old_digest(value))
        self.assertEqual(metrics['ordinary_encode_calls'],1)
        self.assertEqual(metrics['ordinary_encode_bytes'],len(canonical(value)))
        self.assertEqual(metrics['native_hash_bytes'],len(canonical(value)))
        self.assertLessEqual(metrics['peak_combined_workspace_bytes'],WORKSPACE)

    def test_ordinary_encoder_never_swallows_nested_coverage_binding(self):
        coverage={'proof':[True,1,-0.0,'甲']}
        path=self.root/'nested.bin';path.write_bytes(canonical(coverage))
        ref='sha256:'+sha256(path.read_bytes()).hexdigest()
        desc={'path':str(path),'file_digest':ref,'buffer_digest':ref,'dtype':'uint8','shape':[path.stat().st_size]}
        context={'query':'fixed'}
        skeleton={'ordinary':[True,1,-0.0],'nested':[{'context':context}]}
        original={'ordinary':[True,1,-0.0],'nested':[{'context':{**context,'coverage':coverage}}]}
        metrics={}
        self.assertEqual(native_digest(skeleton,coverage_bindings=((context,desc),),
            maximum_workspace_bytes=WORKSPACE,metrics=metrics),old_digest(original))
        self.assertEqual(metrics['coverage_replay_calls'],1)
        self.assertGreater(metrics['ordinary_encode_calls'],0)
        path.write_bytes(b'x'*path.stat().st_size)
        with self.assertRaisesRegex(ValueError,'coverage replay digest mismatch'):
            native_digest(skeleton,coverage_bindings=((context,desc),),maximum_workspace_bytes=WORKSPACE)

    def test_ordinary_encoding_does_not_borrow_mutable_caller_graph(self):
        import axiom_research.stock_native_json as native
        value={'nested':[{'v':-0.0,'flag':False}]};expected=old_digest(value)
        original_dumps=json.dumps
        def mutate_original_before_encoding(snapshot,**kwargs):
            self.assertIsNot(snapshot,value)
            self.assertIsNot(snapshot['nested'],value['nested'])
            value['nested'][0].update(v=0.0,flag=0)
            return original_dumps(snapshot,**kwargs)
        with patch.object(native.json,'dumps',side_effect=mutate_original_before_encoding):
            self.assertEqual(native_digest(value,maximum_workspace_bytes=WORKSPACE),expected)
        self.assertNotEqual(native_digest(value,maximum_workspace_bytes=WORKSPACE),expected)

    def test_three_holders_preserve_native_and_children_refs(self):
        coverage = {'observed': [{'security': '甲', 'value': -0.0}, {'value': None}], 'complete': True}
        for value, ref_key in ((raw(coverage), 'label_ref'), (label(coverage), 'metadata_ref'),
                               (feature(coverage), 'metadata_ref')):
            with self.subTest(contract=value['contract_version']):
                unchanged = deepcopy(value)
                metrics = {}
                desc, carrier = self.write(value, ref_key, metrics=metrics)
                self.assertEqual(desc[ref_key], value[ref_key])
                self.assertEqual(carrier['native_ref'], value[ref_key])
                self.assertEqual(carrier['carrier_ref'], old_digest(carrier, 'carrier_ref'))
                self.assertEqual(desc['file_digest'], 'sha256:'+sha256(Path(desc['path']).read_bytes()).hexdigest())
                self.assertTrue(Path(desc['path']).read_bytes().endswith(b'\n'))
                bindings = self.admit_and_verify(carrier, ref_key)
                self.assertEqual(native_digest(carrier['skeleton'], coverage_bindings=bindings,
                    exclude_ref_key=ref_key, maximum_workspace_bytes=WORKSPACE), value[ref_key])
                self.assertGreater(metrics['coverage_replay_bytes'], 0)
                self.assertGreater(metrics['native_hash_calls_by_ref'][value[ref_key]], 1)
                self.assertTrue(all('coverage' not in context for context, _ in bindings))
                self.assertEqual(value, unchanged)

    def test_present_null_four_bytes_and_absent_inline(self):
        absent = raw()
        self.assertEqual(self.write(absent, 'label_ref'), (None, None))
        value = raw(None)
        _, carrier = self.write(value, 'label_ref')
        desc = carrier['coverage_slots'][0]['bytes']
        self.assertEqual(desc['shape'], [4])
        self.assertEqual(Path(desc['path']).read_bytes(), b'null')
        self.admit_and_verify(carrier, 'label_ref')
        self.assertIn('coverage', value['source_evidence']['context'])
        self.assertIsNone(value['source_evidence']['context']['coverage'])

    def test_distinct_slots_share_cas_but_native_replays_are_counted(self):
        metrics = {}
        _, carrier = self.write(feature({'a': [1, 2, 3]}), 'metadata_ref', metrics=metrics)
        slots = carrier['coverage_slots']
        self.assertEqual(len(slots), 2)
        self.assertEqual(slots[0]['bytes'], slots[1]['bytes'])
        size = slots[0]['bytes']['shape'][0]
        self.assertEqual(metrics['created_coverage_source_bytes'], size)
        self.assertGreaterEqual(metrics['coverage_replay_calls'], 3)
        self.assertGreaterEqual(metrics['coverage_replay_bytes'], 3*size)
        self.assertEqual(len(list((self.root/'buffers').glob('*.bin'))), 1)

    def test_stage_mapper_resolver_and_reseal_existing_binding(self):
        stage, target = self.root/'stage', self.root/'target'
        mapper_calls = []
        def mapper(desc):
            mapper_calls.append(desc['path'])
            return {**desc, 'path': str(target/Path(desc['path']).relative_to(stage))}
        def resolver(path):
            return stage/Path(path).relative_to(target)
        args = dict(maximum_source_bytes=4*1024*1024, maximum_parent_bytes=64*1024,
                    maximum_workspace_bytes=WORKSPACE, descriptor_mapper=mapper, path_resolver=resolver)
        value = label({'a': [1, None]})
        first = make_native_carrier(stage, value, 'metadata_ref', **args)
        original = json.loads(Path(first['path']).read_bytes())
        self.assertEqual(len(mapper_calls), 1)
        self.assertTrue(first['path'].startswith(str(stage)))
        self.assertTrue(original['coverage_slots'][0]['bytes']['path'].startswith(str(target)))
        bindings = validate_carrier_shape(original, 'metadata_ref', maximum_workspace_bytes=WORKSPACE)
        new_value = dict(original['skeleton'])
        new_value['generation'] = 'normalized'
        new_value['metadata_ref'] = native_digest(new_value, exclude_ref_key='metadata_ref',
            coverage_bindings=bindings, maximum_workspace_bytes=WORKSPACE, path_resolver=resolver)
        second = make_native_carrier(stage, new_value, 'metadata_ref', coverage_bindings=bindings, **args)
        new_carrier = json.loads(Path(second['path']).read_bytes())
        self.assertEqual(len(mapper_calls), 1, 'admitted final descriptors must not be remapped')
        self.assertEqual(original['coverage_slots'][0]['bytes'], new_carrier['coverage_slots'][0]['bytes'])
        self.assertNotEqual(original['native_ref'], new_carrier['native_ref'])
        stage.rename(target)
        self.admit_and_verify(new_carrier, 'metadata_ref')
        self.assertEqual(Path(target/Path(first['path']).relative_to(stage)).read_bytes(),
                         canonical(original)+b'\n')

    def test_only_identity_bound_objects_receive_omitted_coverage(self):
        _, carrier = self.write(raw({'a': 1}), 'label_ref')
        bindings = self.admit_and_verify(carrier, 'label_ref')
        equal_but_distinct = deepcopy(carrier['skeleton'])
        self.assertNotEqual(native_digest(equal_but_distinct, coverage_bindings=bindings,
            exclude_ref_key='label_ref', maximum_workspace_bytes=WORKSPACE), carrier['native_ref'])
        with self.assertRaisesRegex(ValueError, 'bindings differ'):
            verify_carrier_native(carrier, 'label_ref', coverage_bindings=[(deepcopy(bindings[0][0]), bindings[0][1])],
                                  maximum_workspace_bytes=WORKSPACE)

    def test_shape_does_not_open_blobs_but_replay_rejects_tamper(self):
        _, carrier = self.write(raw({'a': 1}), 'label_ref')
        desc = carrier['coverage_slots'][0]['bytes']
        Path(desc['path']).write_bytes(b'{"a":2}')
        with patch.object(Path, 'open', side_effect=AssertionError('shape opened a blob')):
            bindings = validate_carrier_shape(carrier, 'label_ref', maximum_workspace_bytes=WORKSPACE)
        with self.assertRaisesRegex(ValueError, 'replay digest'):
            verify_carrier_native(carrier, 'label_ref', coverage_bindings=bindings,
                                  maximum_workspace_bytes=WORKSPACE)

    def test_replay_rejects_mutation_during_read(self):
        _, carrier = self.write(raw({'a': 1}), 'label_ref')
        bindings = validate_carrier_shape(carrier, 'label_ref', maximum_workspace_bytes=WORKSPACE)
        blob = Path(bindings[0][1]['path'])
        original_open = Path.open
        class MutatingReader:
            def __init__(self, stream): self.stream, self.changed = stream, False
            def __enter__(self): return self
            def __exit__(self, *args): self.stream.close()
            def fileno(self): return self.stream.fileno()
            def read(self, size):
                data = self.stream.read(size)
                if data and not self.changed:
                    self.changed = True
                    with original_open(blob, 'wb') as target: target.write(b'{"a":2}')
                return data
        def opening(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return MutatingReader(stream) if path == blob and args == ('rb',) else stream
        with patch.object(Path, 'open', opening):
            with self.assertRaisesRegex(ValueError, 'changed during replay'):
                verify_carrier_native(carrier, 'label_ref', coverage_bindings=bindings,
                                      maximum_workspace_bytes=WORKSPACE)

    def test_illegal_slot_paths_duplicates_unsorted_and_missing_intermediates(self):
        _, carrier = self.write(feature({'a': 1}), 'metadata_ref')
        mutations = []
        wrong = deepcopy(carrier); wrong['coverage_slots'][0]['path'] = ['contents', 'query', 'coverage']; mutations.append(wrong)
        wrong = deepcopy(carrier); wrong['coverage_slots'][0]['path'] = ['input_evidence', True, 'coverage']; mutations.append(wrong)
        wrong = deepcopy(carrier); wrong['coverage_slots'][0]['path'] = ['input_evidence', 9, 'coverage']; mutations.append(wrong)
        wrong = deepcopy(carrier); wrong['coverage_slots'].append(deepcopy(wrong['coverage_slots'][0])); mutations.append(wrong)
        wrong = deepcopy(carrier); wrong['coverage_slots'].reverse(); mutations.append(wrong)
        wrong = deepcopy(carrier); del wrong['skeleton']['input_evidence'][0]['source_evidence']; mutations.append(wrong)
        wrong = deepcopy(carrier)
        path = wrong['coverage_slots'][0]['path']; current = wrong['skeleton']
        for step in path[:-1]: current = current[step]
        current['coverage'] = None; mutations.append(wrong)
        for wrong in mutations:
            reseal_carrier(wrong)
            with self.subTest(path=wrong['coverage_slots'][0]['path']):
                with self.assertRaises(ValueError):
                    validate_carrier_shape(wrong, 'metadata_ref', maximum_workspace_bytes=WORKSPACE)

    def test_self_and_carrier_ref_tampering(self):
        _, carrier = self.write(raw(None), 'label_ref')
        wrong = deepcopy(carrier); wrong['native_ref'] = old_digest('wrong'); reseal_carrier(wrong)
        with self.assertRaisesRegex(ValueError, 'self reference mismatch'):
            validate_carrier_shape(wrong, 'label_ref', maximum_workspace_bytes=WORKSPACE)
        wrong = deepcopy(carrier); wrong['carrier_ref'] = old_digest('wrong')
        with self.assertRaisesRegex(ValueError, 'carrier reference mismatch'):
            validate_carrier_shape(wrong, 'label_ref', maximum_workspace_bytes=WORKSPACE)
        wrong = deepcopy(carrier); wrong['skeleton']['rows'][0]['score'] = 7; reseal_carrier(wrong)
        with self.assertRaisesRegex(ValueError, 'native reference mismatch'):
            self.admit_and_verify(wrong, 'label_ref')

    def test_original_child_hash_is_verified_independently(self):
        for value in (label({'a': 1}), feature({'a': 1})):
            selected_ref = next(ref for ref, child in value['contents'].items() if child != {'query': 1})
            selected = value['contents'].pop(selected_ref)
            value['contents'][old_digest('incorrect child reference')] = selected
            seal(value, 'metadata_ref')
            with self.subTest(contract=value['contract_version']):
                with self.assertRaisesRegex(ValueError, 'original child reference mismatch'):
                    self.write(value, 'metadata_ref')

    def test_exact_carrier_and_buffer_shapes(self):
        _, carrier = self.write(raw(None), 'label_ref')
        mutations = []
        wrong = deepcopy(carrier); wrong['extra'] = 1; mutations.append(wrong)
        wrong = deepcopy(carrier); wrong['coverage_slots'] = []; mutations.append(wrong)
        for key, child in (('dtype', 'float64_le'), ('shape', [True]), ('shape', [0]),
                           ('shape', [4, 1]), ('buffer_digest', old_digest('wrong')),
                           ('path', '')):
            wrong = deepcopy(carrier); wrong['coverage_slots'][0]['bytes'][key] = child; mutations.append(wrong)
        wrong = deepcopy(carrier); wrong['coverage_slots'][0]['bytes']['extra'] = 1; mutations.append(wrong)
        for wrong in mutations:
            reseal_carrier(wrong)
            with self.assertRaises(ValueError):
                validate_carrier_shape(wrong, 'label_ref', maximum_workspace_bytes=WORKSPACE)

    def test_caller_growth_rejects_before_encoding_and_does_not_publish_partial_file(self):
        calls = []
        def retained():
            calls.append(1)
            return 0 if len(calls) < 5 else WORKSPACE
        with self.assertRaisesRegex(ValueError, 'workspace budget'):
            self.write(raw({'a': 1}), 'label_ref', caller_retained_bytes=retained)
        self.assertEqual(list(self.root.rglob('*.bin')), [])
        self.assertEqual(list(self.root.rglob('*.json')), [])

    def test_conflicting_bindings_are_rejected(self):
        _, carrier = self.write(raw({'a': 1}), 'label_ref')
        bindings = self.admit_and_verify(carrier, 'label_ref')
        context, desc = bindings[0]
        changed = {**desc, 'path': desc['path']+'.other'}
        with self.assertRaisesRegex(ValueError, 'conflicting coverage object binding'):
            native_digest(context, coverage_bindings=[(context, desc), (context, changed)],
                          maximum_workspace_bytes=WORKSPACE)

    def test_feature_source_ids_cannot_acquire_unbound_slots(self):
        value = feature({'a': 1})
        value['input_evidence'][0]['core_plan']['sources'] = []
        seal(value, 'metadata_ref')
        with self.assertRaisesRegex(ValueError, 'source IDs differ'):
            self.write(value, 'metadata_ref')

    def test_rejects_nonfinite_unknown_exclusion_and_invalid_budgets(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                native_digest({'x': value}, maximum_workspace_bytes=WORKSPACE)
        with self.assertRaisesRegex(ValueError, 'self-reference exclusion'):
            native_digest({'x': 1}, exclude_ref_key='x', maximum_workspace_bytes=WORKSPACE)
        for retained in (-1, True, 1.5):
            with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                native_digest(None, maximum_workspace_bytes=WORKSPACE, caller_retained_bytes=lambda: retained)
        with self.assertRaisesRegex(ValueError, 'workspace budget'):
            self.write(raw({'a': 1}), 'label_ref', maximum_workspace_bytes=128)
        with self.assertRaisesRegex(ValueError, 'source byte budget'):
            self.write(raw({'a': 1}), 'label_ref', maximum_source_bytes=4)
        with self.assertRaisesRegex(ValueError, 'parent byte budget'):
            self.write(raw(None), 'label_ref', maximum_parent_bytes=64)

    def test_large_leaf_streams_with_smaller_workspace_and_counts_actual_bytes(self):
        # One MiB, comfortably below the task's total 32 MiB microbenchmark cap.
        coverage = {'text': 'x'*(1024*1024)}
        value = raw(coverage)
        metrics = {}
        _, carrier = self.write(value, 'label_ref', maximum_workspace_bytes=256*1024, metrics=metrics)
        self.admit_and_verify(carrier, 'label_ref')
        desc = carrier['coverage_slots'][0]['bytes']
        self.assertGreater(desc['shape'][0], 1024*1024)
        self.assertLessEqual(metrics['peak_combined_workspace_bytes'], 256*1024)
        self.assertEqual(metrics['created_coverage_source_bytes'], len(canonical(coverage)))
        self.assertEqual(metrics['coverage_encode_bytes'], len(canonical(coverage)))
        self.assertEqual(metrics['coverage_replay_bytes'], len(canonical(coverage)))
        self.assertEqual(metrics['native_hash_calls_by_ref'][value['label_ref']], 2)

    def test_large_ref_counter_map_is_scanned_once_per_budget(self):
        class CountingRefMap(dict):
            def __init__(self, *args):
                super().__init__(*args)
                self.items_calls = 0
            def items(self):
                self.items_calls += 1
                return super().items()
        refs = CountingRefMap({old_digest({'past': i}): 1 for i in range(1500)})
        metrics = {'native_hash_calls': 7, 'native_hash_bytes': 11,
                   'native_hash_calls_by_ref': refs}
        getter_calls = 0
        def retained():
            nonlocal getter_calls
            getter_calls += 1
            return 0
        value = {'rows': [{'v': i, 'text': '甲'} for i in range(2000)]}
        expected = old_digest(value)
        result = native_digest(value, maximum_workspace_bytes=WORKSPACE,
                               caller_retained_bytes=retained, metrics=metrics)
        self.assertEqual(result, expected)
        self.assertEqual(refs.items_calls, 1, 'token/span guards must use cached counter bytes')
        self.assertGreater(getter_calls, 2000, 'dynamic caller checks must remain active')
        self.assertEqual(metrics['native_hash_calls'], 8)
        self.assertEqual(metrics['native_hash_bytes'], 11+len(canonical(value)))
        self.assertEqual(refs[expected], 1)
        self.assertEqual(len(refs), 1501)
        # A new operation may scan once again, then O(1) update the old ref.
        native_digest(value, maximum_workspace_bytes=WORKSPACE, metrics=metrics)
        self.assertEqual(refs.items_calls, 2)
        self.assertEqual(refs[expected], 2)

    def test_counter_growth_precharged_and_large_counter_admission_rejected(self):
        metrics = {'native_hash_calls': 2**30-1, 'native_hash_bytes': 2**30-1}
        expected = old_digest([1, 2, 3])
        self.assertEqual(native_digest([1, 2, 3], maximum_workspace_bytes=WORKSPACE,
                                       metrics=metrics), expected)
        self.assertEqual(metrics['native_hash_calls'], 2**30)
        self.assertEqual(metrics['native_hash_bytes'], 2**30-1+len(canonical([1, 2, 3])))
        refs = {old_digest(i): 1 for i in range(1000)}
        with self.assertRaisesRegex(ValueError, 'workspace budget'):
            native_digest(None, maximum_workspace_bytes=16*1024,
                          metrics={'native_hash_calls_by_ref': refs})


if __name__ == '__main__':
    unittest.main()

"""Raw-only carrier storage admission; no business PIT or numerical executor.

Every fixture is small and synthetic. Deliberately invalid coverage is sealed
at every identity layer so rejection cannot be attributed to an old digest.
"""
from copy import deepcopy
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

from axiom_research.stock_matrix_reader import VerifiedMatrixStore, _admit_raw_label
from axiom_research.stock_native_json import make_native_carrier


ABSENT = object()
WORKSPACE = 2 * 1024 * 1024
SOURCE_LIMIT = 8 * 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def byte_ref(value):
    return 'sha256:' + sha256(value).hexdigest()


def sealed(value, key):
    result = {k: v for k, v in value.items() if k != key}
    result[key] = byte_ref(canonical(result))
    return result


def raw_build(coverage=ABSENT, *, query='one'):
    context = {'query': {'name': query, 'sessions': ['2026-01-02'],
                         'fields': ['close']}, 'reader_version': 'test-reader/1'}
    if coverage is not ABSENT:
        context['coverage'] = coverage
    return sealed({'contract_version': 'stock_label_build_v1',
        'label_spec': {'horizon': 1}, 'calendar_ref': byte_ref(b'calendar'),
        'source_ref': byte_ref(b'source'),
        'source_evidence': {'context': context, 'records_ref': byte_ref(b'[]')},
        # Storage admission preserves these bytes; consumer row/PIT checks
        # are deliberately outside this module's scope.
        'rows': [{'security_id': 'TEST', 'feature_session': '2026-01-02',
                  'valid': False, 'return': None, 'sample': -0.0,
                  'note': '甲𝄞\n"\\'}]}, 'label_ref')


class NativeCarrierReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='axiom-native-reader-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.sequence = 0

    def store(self, **overrides):
        arguments = dict(maximum_source_bytes=SOURCE_LIMIT,
                         maximum_matrix_bytes=WORKSPACE,
                         maximum_parent_bytes=64 * 1024)
        arguments.update(overrides)
        store = VerifiedMatrixStore(**arguments)
        self.addCleanup(lambda: None if store.closed else store.close())
        return store

    def save(self, value, key='label_ref', *, content_ref=None):
        self.sequence += 1
        path = self.root / ('parent-%s.json' % self.sequence)
        data = canonical(value) + b'\n'
        path.write_bytes(data)
        return {'path': str(path), 'file_digest': byte_ref(data),
                key: value[key] if content_ref is None else content_ref}

    def carried(self, value):
        desc = make_native_carrier(self.root, value, 'label_ref',
            maximum_source_bytes=SOURCE_LIMIT, maximum_parent_bytes=64 * 1024,
            maximum_workspace_bytes=WORKSPACE)
        self.assertIsNotNone(desc)
        return desc, json.loads(Path(desc['path']).read_bytes())

    def save_carrier(self, carrier):
        carrier = sealed(carrier, 'carrier_ref')
        return self.save(carrier, content_ref=carrier['native_ref']), carrier

    def malformed_carrier(self, payload):
        """Independently reseal physical and native refs around arbitrary bytes."""
        self.sequence += 1
        path = self.root / ('coverage-invalid-%s.bin' % self.sequence)
        path.write_bytes(payload)
        marker = '__raw_coverage_placeholder__'
        original = raw_build(marker)
        body = {k: v for k, v in original.items() if k != 'label_ref'}
        encoded = canonical(body)
        token = canonical(marker)
        self.assertEqual(encoded.count(token), 1)
        restored = encoded.replace(token, payload)
        native_ref = byte_ref(restored)
        skeleton = deepcopy(original)
        del skeleton['source_evidence']['context']['coverage']
        skeleton['label_ref'] = native_ref
        blob_ref = byte_ref(payload)
        carrier = {'contract_version': 'stock_native_json_carrier_v1',
            'native_ref': native_ref, 'skeleton': skeleton,
            'coverage_slots': [{'path': ['source_evidence', 'context', 'coverage'],
                'bytes': {'path': str(path), 'file_digest': blob_ref,
                          'buffer_digest': blob_ref, 'dtype': 'uint8',
                          'shape': [len(payload)]}}]}
        desc, carrier = self.save_carrier(carrier)
        self.assertEqual(desc['label_ref'], byte_ref(restored))
        self.assertEqual(carrier['carrier_ref'], byte_ref(canonical(
            {k: v for k, v in carrier.items() if k != 'carrier_ref'})))
        self.assertEqual(desc['file_digest'], byte_ref(Path(desc['path']).read_bytes()))
        return desc, carrier

    def test_inline_and_carried_use_the_same_original_descriptor_reference(self):
        original = raw_build({'complete': True, 'rows': [None, -0.0, '甲𝄞']})
        inline = self.save(original)
        carried, carrier = self.carried(original)
        store = self.store()
        self.assertEqual(store.read_json(inline, 'label_ref'), original)
        skeleton = store.read_json(carried, 'label_ref')
        self.assertEqual(inline['label_ref'], carried['label_ref'])
        self.assertEqual(skeleton['rows'], original['rows'])
        self.assertEqual(skeleton['source_evidence']['context']['query'],
                         original['source_evidence']['context']['query'])
        self.assertNotIn('coverage', skeleton['source_evidence']['context'])
        self.assertEqual(store.native_digest(skeleton, exclude_ref_key='label_ref'),
                         original['label_ref'])
        self.assertEqual(store.context_equal(skeleton['source_evidence']['context'],
                                             original['source_evidence']['context']), True)
        self.assertIsNone(store.storage_binding(inline))
        self.assertEqual(store.storage_binding(carried),
            {'representation': 'stock_native_json_carrier_v1', 'carrier': carried})
        self.assertEqual(store.source_paths(carried),
                         (carried['path'], carrier['coverage_slots'][0]['bytes']['path']))
        self.assertIsNone(store._np)

    def test_two_query_holders_share_one_syntax_admission_but_replay_native_hashes(self):
        coverage = {'complete': True, 'observed': [1, 2, None]}
        first, c1 = self.carried(raw_build(coverage, query='first'))
        second, c2 = self.carried(raw_build(deepcopy(coverage), query='second'))
        self.assertNotEqual(first['label_ref'], second['label_ref'])
        self.assertEqual(c1['coverage_slots'][0]['bytes'], c2['coverage_slots'][0]['bytes'])
        store = self.store()
        store.read_json(first, 'label_ref')
        store.read_json(second, 'label_ref')
        before = deepcopy(store.metrics)
        store.read_json(first, 'label_ref')
        store.read_json(second, 'label_ref')
        self.assertEqual(store.metrics, before)
        self.assertEqual(before['coverage_validation_calls'], 1)
        self.assertEqual(before['json_decode_calls'], 2)
        self.assertEqual(before['file_hash_calls'], 3)
        size = c1['coverage_slots'][0]['bytes']['shape'][0]
        self.assertEqual(before['coverage_validation_bytes'], size)
        self.assertEqual(before['coverage_replay_calls'], 2)
        self.assertEqual(before['coverage_replay_bytes'], 2 * size)
        self.assertEqual(before['native_hash_calls_by_ref'],
                         {first['label_ref']: 1, second['label_ref']: 1})
        self.assertEqual(set(store.source_paths(first)) & set(store.source_paths(second)),
                         {c1['coverage_slots'][0]['bytes']['path']})

    def test_same_blob_bytes_at_a_new_path_require_independent_validation(self):
        first, carrier = self.carried(raw_build({'a': 1}))
        blob = carrier['coverage_slots'][0]['bytes']
        original_path = blob['path']
        copy_path = self.root / 'same-bytes-new-path.bin'
        copy_path.write_bytes(Path(blob['path']).read_bytes())
        carrier['coverage_slots'][0]['bytes']['path'] = str(copy_path)
        second, _ = self.save_carrier(carrier)
        store = self.store()
        store.read_json(first, 'label_ref')
        store.read_json(second, 'label_ref')
        self.assertEqual(store.metrics['coverage_validation_calls'], 2)
        self.assertEqual(set(store._coverage_verified), {original_path, str(copy_path)})

    def test_invalid_resealed_json_is_rejected_by_syntax_without_success_cache(self):
        payloads = (b'{"a":}', b'{"a":1,"a":2}', b'NaN', b'Infinity', b'1e400',
                    b'{"a": 1}', b'{"b":1,"a":2}', b'{"a":1}\n',
                    b'1.0e+0', b'"\\u7532"', b'\xef\xbb\xbfnull', b'"\xc3"')
        for payload in payloads:
            with self.subTest(payload=payload):
                desc, carrier = self.malformed_carrier(payload)
                store = self.store()
                with self.assertRaises(ValueError) as caught:
                    store.read_json(desc, 'label_ref')
                self.assertRegex(str(caught.exception),
                                 '(?i)json|whitespace|nonfinite|noncanonical')
                self.assertNotIn('reference mismatch', str(caught.exception))
                self.assertNotIn('digest mismatch', str(caught.exception))
                blob = carrier['coverage_slots'][0]['bytes']['path']
                self.assertNotIn(blob, store._coverage_verified)
                self.assertNotIn(desc['path'], store._native_verified)
                self.assertEqual(store.metrics.get('coverage_validation_calls', 0), 0)

    def test_present_null_and_absent_coverage_remain_distinct(self):
        absent = raw_build()
        self.assertIsNone(make_native_carrier(self.root, absent, 'label_ref',
            maximum_source_bytes=SOURCE_LIMIT, maximum_parent_bytes=64 * 1024,
            maximum_workspace_bytes=WORKSPACE))
        inline = self.save(absent)
        carried, carrier = self.carried(raw_build(None))
        blob = carrier['coverage_slots'][0]['bytes']
        self.assertEqual(blob['shape'], [4])
        self.assertEqual(Path(blob['path']).read_bytes(), b'null')
        store = self.store()
        missing = store.read_json(inline, 'label_ref')['source_evidence']['context']
        present = store.read_json(carried, 'label_ref')['source_evidence']['context']
        self.assertEqual(store.coverage_identity(missing), (False, None))
        self.assertEqual(store.coverage_identity(present), (True, byte_ref(b'null')))
        self.assertFalse(store.context_equal(missing, present))

    def test_illegal_slots_resealed_wrapper_cannot_be_admitted(self):
        _, original = self.carried(raw_build({'a': 1}))
        cases = []
        value = deepcopy(original); value['coverage_slots'] = []; cases.append(value)
        value = deepcopy(original); value['coverage_slots'].append(deepcopy(value['coverage_slots'][0])); cases.append(value)
        value = deepcopy(original); value['coverage_slots'][0]['path'] = ['rows', 0, 'coverage']; cases.append(value)
        value = deepcopy(original); value['coverage_slots'][0]['path'] = ['source_evidence', True, 'coverage']; cases.append(value)
        value = deepcopy(original); del value['skeleton']['source_evidence']['context']; cases.append(value)
        value = deepcopy(original); value['skeleton']['source_evidence']['context']['coverage'] = None; cases.append(value)
        value = deepcopy(original); value['coverage_slots'][0]['bytes']['shape'] = [True]; cases.append(value)
        for value in cases:
            with self.subTest(value=value['coverage_slots']):
                desc, _ = self.save_carrier(value)
                store = self.store()
                with self.assertRaises(ValueError): store.read_json(desc, 'label_ref')
                self.assertFalse(store._coverage_verified)
                self.assertNotIn(desc['path'], store._native_verified)

    def test_resealed_wrapper_does_not_replace_original_native_identity(self):
        _, carrier = self.carried(raw_build({'a': 1}))
        carrier['skeleton']['rows'][0]['note'] = 'changed'
        desc, _ = self.save_carrier(carrier)
        store = self.store()
        with self.assertRaisesRegex(ValueError, 'restored native reference mismatch'):
            store.read_json(desc, 'label_ref')
        self.assertNotIn(desc['path'], store._native_verified)

    def test_blob_tamper_and_missing_file_are_not_producer_trusted(self):
        desc, carrier = self.carried(raw_build({'a': 1}))
        path = Path(carrier['coverage_slots'][0]['bytes']['path'])
        path.write_bytes(b'{"a":2}')
        store = self.store()
        with self.assertRaisesRegex(ValueError, 'coverage identity mismatch'):
            store.read_json(desc, 'label_ref')
        self.assertFalse(store._coverage_verified)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.store().read_json(desc, 'label_ref')

    def test_cache_checks_changed_blob_fingerprint_on_read(self):
        desc, carrier = self.carried(raw_build({'a': 1}))
        store = self.store()
        store.read_json(desc, 'label_ref')
        Path(carrier['coverage_slots'][0]['bytes']['path']).write_bytes(b'{"a":2}')
        with self.assertRaisesRegex(ValueError, 'source changed|coverage.*changed'):
            store.read_json(desc, 'label_ref')

    def test_new_store_validates_again_instead_of_trusting_previous_store(self):
        desc, carrier = self.carried(raw_build({'a': 1}))
        first = self.store()
        first.read_json(desc, 'label_ref')
        first.close()
        second = self.store()
        second.read_json(desc, 'label_ref')
        self.assertEqual(second.metrics['coverage_validation_calls'], 1)
        self.assertEqual(second.metrics['coverage_validation_bytes'],
                         carrier['coverage_slots'][0]['bytes']['shape'][0])

    def test_parent_source_and_workspace_budgets_fail_before_success(self):
        desc, carrier = self.carried(raw_build({'values': list(range(100))}))
        parent_size = Path(desc['path']).stat().st_size
        cases = ({'maximum_parent_bytes': parent_size - 1},
                 {'maximum_source_bytes': parent_size},
                 {'maximum_matrix_bytes': 1024})
        for limits in cases:
            with self.subTest(limits=limits):
                store = self.store(**limits)
                with self.assertRaisesRegex(ValueError, '(?i)budget'):
                    store.read_json(desc, 'label_ref')
                self.assertFalse(store._coverage_verified)
                self.assertNotIn(desc['path'], store._native_verified)

    def test_large_leaf_is_streamed_and_not_decoded_as_a_parent(self):
        original = raw_build({'values': list(range(12000))})
        desc, carrier = self.carried(original)
        blob = carrier['coverage_slots'][0]['bytes']
        self.assertGreater(blob['shape'][0], 4096)
        self.assertLess(Path(desc['path']).stat().st_size, 4096)
        store = self.store(maximum_parent_bytes=4096, maximum_matrix_bytes=256 * 1024)
        from axiom_research import stock_matrix_reader as reader
        with patch.object(reader, '_read', wraps=reader._read) as read:
            value = store.read_json(desc, 'label_ref')
        self.assertEqual([str(call.args[0]) for call in read.call_args_list], [desc['path']])
        self.assertNotIn('coverage', value['source_evidence']['context'])
        self.assertEqual(store.metrics['coverage_validation_bytes'], blob['shape'][0])
        self.assertIsNone(store._np)

    def test_source_paths_and_fingerprints_include_every_physical_blob(self):
        desc, carrier = self.carried(raw_build({'a': [1, 2]}))
        store = self.store()
        with _admit_raw_label(desc, store=store) as lease:
            expected = {desc['path'], carrier['coverage_slots'][0]['bytes']['path']}
            self.assertEqual(set(store.source_paths(desc)), expected)
            self.assertEqual(set(dict(lease.source_records)), expected)
            self.assertEqual(set(lease.source_fingerprints), expected)
            for path, reference in lease.source_records:
                self.assertEqual(reference, byte_ref(Path(path).read_bytes()))
                self.assertEqual(lease.source_fingerprints[path], store.fingerprints[path])
            self.assertEqual(lease.storage_binding,
                {'representation': 'stock_native_json_carrier_v1', 'carrier': desc})

    def test_external_override_does_not_need_batch_membership(self):
        inline = self.save(raw_build(query='unrelated-initial'))
        override, _ = self.carried(raw_build({'a': 1}, query='external-override'))
        store = self.store()
        store.read_json(inline, 'label_ref')
        self.assertNotIn(override['path'], store.descriptors)
        with _admit_raw_label(override, store=store) as lease:
            self.assertEqual(lease.raw['source_evidence']['context']['query']['name'],
                             'external-override')
            self.assertEqual(lease.raw['label_ref'], override['label_ref'])
        self.assertFalse(store.closed)
        self.assertEqual(store.borrowers, 0)
        self.assertEqual(store.lease_bytes, 0)

    def test_returned_raw_and_storage_binding_do_not_pollute_next_lease(self):
        original = raw_build({'a': 1})
        desc, _ = self.carried(original)
        store = self.store()
        with _admit_raw_label(desc, store=store) as first:
            first.raw['rows'][0]['note'] = 'caller mutation'
            first.raw['source_evidence']['context']['query']['name'] = 'caller query'
            first.storage_binding['carrier']['label_ref'] = byte_ref(b'caller ref')
            first.source_fingerprints.clear()
            with _admit_raw_label(desc, store=store) as second:
                self.assertEqual(second.raw['rows'], original['rows'])
                self.assertEqual(second.raw['source_evidence']['context']['query'],
                                 original['source_evidence']['context']['query'])
                self.assertEqual(second.storage_binding['carrier'], desc)
                self.assertEqual(set(second.source_fingerprints),
                                 set(store.source_paths(desc)))
            self.assertEqual(store.storage_binding(desc)['carrier'], desc)
            self.assertEqual(store.metrics['coverage_validation_calls'], 1)

    def test_inline_bundle_selects_exact_original_raw_child(self):
        training, evaluation = raw_build(query='training'), raw_build(query='evaluation')
        bundle = sealed({'contract_version': 'stock_label_bundle_v1',
                         'training': training, 'evaluation': evaluation}, 'label_ref')
        for child in (training, evaluation):
            with self.subTest(query=child['source_evidence']['context']['query']):
                desc = self.save(bundle, content_ref=child['label_ref'])
                with _admit_raw_label(desc) as lease:
                    self.assertEqual(lease.raw, child)
                    self.assertIsNone(lease.storage_binding)
                    self.assertEqual(lease.source_records,
                                     ((desc['path'], desc['file_digest']),))
                    self.assertEqual(set(lease.source_fingerprints), {desc['path']})

    def test_bundle_missing_ambiguous_or_bad_self_identity_is_rejected(self):
        child = raw_build()
        bundle = sealed({'contract_version': 'stock_label_bundle_v1',
                         'training': child, 'evaluation': deepcopy(child)}, 'label_ref')
        ambiguous = self.save(bundle, content_ref=child['label_ref'])
        missing = self.save(bundle, content_ref=byte_ref(b'unknown-child'))
        bad = deepcopy(bundle); bad['training']['rows'].append({'wrong': 1})
        invalid = self.save(bad, content_ref=child['label_ref'])
        for desc, message in ((ambiguous, 'missing/ambiguous'), (missing, 'missing/ambiguous'),
                              (invalid, 'saved label_ref mismatch')):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message): _admit_raw_label(desc)

    def test_two_active_leases_close_and_gc_return_borrower_and_byte_counts(self):
        desc, _ = self.carried(raw_build({'a': 1}))
        store = self.store()
        first = _admit_raw_label(desc, store=store)
        second = _admit_raw_label(desc, store=store)
        self.assertEqual(store.borrowers, 2)
        self.assertEqual(store.lease_bytes, first._lease_bytes + second._lease_bytes)
        self.assertEqual(store.metrics['coverage_validation_calls'], 1)
        with self.assertRaisesRegex(ValueError, 'borrowed'): store.close()
        first.close(); first.close()
        self.assertIsNone(first.raw)
        self.assertIsNone(first.storage_binding)
        self.assertIsNone(first.source_records)
        self.assertIsNone(first.source_fingerprints)
        self.assertEqual(store.borrowers, 1)
        self.assertEqual(store.lease_bytes, second._lease_bytes)
        reference = weakref.ref(second)
        del second
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(store.borrowers, 0)
        self.assertEqual(store.lease_bytes, 0)
        self.assertFalse(store.closed)

    def test_fresh_store_auto_closes_on_explicit_release_and_gc(self):
        desc, _ = self.carried(raw_build({'a': 1}))
        lease = _admit_raw_label(desc)
        store = lease._store
        self.assertFalse(store.closed)
        self.assertIsNone(store._np)
        lease.close()
        self.assertTrue(store.closed)
        self.assertEqual(store.borrowers, 0)
        self.assertEqual(store.lease_bytes, 0)
        lease = _admit_raw_label(desc)
        store = lease._store
        reference = weakref.ref(lease)
        del lease
        gc.collect()
        self.assertIsNone(reference())
        self.assertTrue(store.closed)
        self.assertEqual(store.lease_bytes, 0)

    def test_failed_owned_admission_closes_store_and_shared_failure_leaks_no_lease(self):
        desc = self.save(raw_build())
        from axiom_research import stock_matrix_reader as reader
        created = []
        def factory(**kwargs):
            store = VerifiedMatrixStore(**kwargs); created.append(store); return store
        with patch.object(reader, 'VerifiedMatrixStore', side_effect=factory):
            with self.assertRaisesRegex(ValueError, 'budget'):
                _admit_raw_label(desc, limits={'maximum_source_bytes': SOURCE_LIMIT,
                                             'maximum_matrix_bytes': 32})
        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].closed)
        store = self.store(maximum_matrix_bytes=32)
        with self.assertRaisesRegex(ValueError, 'budget'): _admit_raw_label(desc, store=store)
        self.assertEqual(store.borrowers, 0)
        self.assertEqual(store.lease_bytes, 0)
        self.assertFalse(store.closed)

    def test_closed_store_and_mixed_existing_limits_are_rejected(self):
        desc = self.save(raw_build())
        store = self.store()
        with self.assertRaisesRegex(ValueError, 'owns Raw admission limits'):
            _admit_raw_label(desc, store=store, limits={})
        store.close()
        with self.assertRaisesRegex(ValueError, 'closed'): _admit_raw_label(desc, store=store)

    def test_raw_only_fresh_process_forbids_optional_runtime_imports_and_writes(self):
        inline = self.save(raw_build())
        carried, _ = self.carried(raw_build({'a': [1, None]}))
        before = {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.root.rglob('*') if path.is_file()}
        program = '''
import gc, importlib.abc, json, sys
blocked = {'numpy', 'pandas', 'qlib', 'axiom_data', 'axiom_engine', 'lightgbm'}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Forbidden Raw-only import: ' + fullname)
sys.meta_path.insert(0, Guard())
from axiom_research.stock_matrix_reader import _admit_raw_label
for descriptor in json.loads(sys.argv[1]):
    lease = _admit_raw_label(descriptor)
    store = lease._store
    assert store._np is None and store.borrowers == 1
    assert lease.raw['label_ref'] == descriptor['label_ref']
    lease.close()
    assert store.closed and store.lease_bytes == 0 and store.borrowers == 0
assert not any(name.split('.')[0] in blocked for name in sys.modules)
print('RAW_ONLY_STDLIB_PASS')
'''
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        result = subprocess.run([sys.executable, '-B', '-c', program,
                                 json.dumps([inline, carried])], env=environment,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('RAW_ONLY_STDLIB_PASS', result.stdout)
        after = {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
                 for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(after, before)


if __name__ == '__main__':
    unittest.main()

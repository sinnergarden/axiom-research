"""Legacy parent workspace admission, using only bounded synthetic files."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_research.stock_artifacts import _read, digest, file_digest
from axiom_research.stock_canonical_json import validate_canonical_chunks
from axiom_research.stock_matrix_reader import VerifiedMatrixStore
from axiom_research.stock_parent_json import preflight_parent_json


class ParentJsonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='axiom-parent-json-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def saved(self, payload, name='parent.json'):
        path = self.root/name
        path.write_bytes(payload)
        return path

    def store(self, **kwargs):
        store = VerifiedMatrixStore(**kwargs)
        self.addCleanup(store.close)
        return store

    def preflight(self, payload, maximum=512*1024**2, **kwargs):
        return preflight_parent_json((payload[start:start+4096] for start in range(0,len(payload),4096)),
            physical_size=len(payload), maximum_workspace_bytes=maximum, **kwargs)

    def test_small_public_label_builder_matches_old_read_and_native_identity(self):
        from test_stock_labels import build
        value = build()
        payload = (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode()
        path = self.saved(payload)
        legacy = _read(path)
        self.assertEqual(value, legacy)
        descriptor = {'path': str(path), 'file_digest': file_digest(path), 'label_ref': value['label_ref']}
        store = self.store()
        loaded = store.read_json(descriptor, 'label_ref')
        self.assertEqual(loaded, legacy)
        self.assertEqual(digest({k:v for k,v in loaded.items() if k!='label_ref'}), value['label_ref'])
        self.assertEqual(store.metrics['json_decode_calls'], 1)
        self.assertIs(store._decode_once(path), loaded)
        self.assertEqual(store.metrics['json_decode_calls'], 1)

    def test_over_four_mib_public_raw_inline_string_admitted_under_unchanged_default(self):
        from test_stock_labels import batch, build
        source = batch()
        source.wire['context']['coverage'] = {'legacy_note': 'x'*(5*1024**2)}
        value = build(source)
        payload = (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode()
        self.assertGreater(len(payload), 4*1024**2)
        self.assertGreater(128*len(payload), 512*1024**2, 'old global multiplier would reject')
        path = self.saved(payload)
        proof = self.preflight(payload)
        self.assertLess(proof['decode_workspace_bytes'], 512*1024**2)
        store = self.store()
        loaded = store.read_json({'path': str(path), 'file_digest': file_digest(path),
                                 'label_ref': value['label_ref']}, 'label_ref')
        self.assertEqual(loaded, _read(path))
        self.assertEqual(loaded['label_ref'], value['label_ref'])
        self.assertEqual(store.maximum_matrix_bytes, 512*1024**2)

    def test_over_four_mib_nested_array_matches_original_whole_decoder(self):
        value = {'items': [[{'text': '甲'+('x'*(5*1024**2))}], [1, None, True, -0.0]]}
        payload = (json.dumps(value, ensure_ascii=False, indent=1)+'\r\n').encode()
        path = self.saved(payload)
        proof = self.preflight(payload)
        self.assertGreater(proof['decoded_graph_upper_bytes'], 5*1024**2)
        self.assertGreater(proof['postcheck_workspace_upper_bytes'], 0)
        store = self.store()
        self.assertEqual(store._decode_once(path), _read(path))

    def test_legacy_key_order_whitespace_escapes_and_numbers_preserved(self):
        payload = b' \r\n {"z":-0.0,"a":[1e+01,"\\u0061",true,null],"x":"\\ud83d\\ude00"}\t\n'
        path = self.saved(payload)
        proof = self.preflight(payload)
        self.assertGreater(proof['pairs_hook_upper_bytes'], 0)
        self.assertGreater(proof['key_memo_upper_bytes'], 0)
        self.assertEqual(self.store()._decode_once(path), _read(path))
        with self.assertRaises(ValueError):
            validate_canonical_chunks([payload], maximum_workspace_bytes=1024*1024)

    def test_invalid_json_and_duplicate_keys_never_enter_decoded_cache(self):
        for index, payload in enumerate((b'{"a":1,"a":2}', b'[1,]', b'01', b'true false',
            b'{"a":NaN}', b'{"a":Infinity}', b'"unterminated', b'"\xff"', b'{"a":1}\x00')):
            with self.subTest(payload=payload):
                path = self.saved(payload, str(index)+'.json')
                store = self.store()
                with self.assertRaises((ValueError, UnicodeError)):
                    store._decode_once(path)
                self.assertNotIn(str(path), store._decoded)
                self.assertNotIn(str(path), store.json)
                self.assertEqual(store.metrics['json_decode_calls'], 1 if index==0 else 0)

    def test_dense_numeric_graph_rejected_before_whole_read(self):
        payload = ('['+','.join(str(1000+i) for i in range(5000))+']\n').encode()
        path = self.saved(payload)
        store = self.store(maximum_matrix_bytes=128*1024)
        with patch('axiom_research.stock_matrix_reader._read', side_effect=AssertionError('unadmitted whole read')):
            with self.assertRaisesRegex(ValueError, 'workspace budget'):
                store._decode_once(path)
        self.assertEqual(store._decoded, {})

    def test_oversized_scalar_rejected_before_scalar_stdlib_decode(self):
        payload = b'"'+b'x'*4096+b'"'
        with patch('axiom_research.stock_parent_json.json.loads', side_effect=AssertionError('unadmitted scalar decode')):
            with self.assertRaisesRegex(ValueError, 'workspace budget'):
                self.preflight(payload, maximum=48*1024)

    def test_dynamic_and_invalid_caller_budgets_rejected_without_cache(self):
        path = self.saved(b'{"a":[1,2,3]}\n')
        for retained in (-1, True, 1.5):
            store = self.store(_caller_retained_bytes=lambda: retained)
            with self.assertRaisesRegex(ValueError, 'retained byte|nonnegative int'):
                store._decode_once(path)
            self.assertEqual(store._decoded, {})
        calls = 0
        def retained():
            nonlocal calls
            calls += 1
            return 0 if calls < 8 else 64*1024
        store = self.store(maximum_matrix_bytes=64*1024, _caller_retained_bytes=retained)
        with patch('axiom_research.stock_matrix_reader._read', side_effect=AssertionError('unadmitted whole read')):
            with self.assertRaisesRegex(ValueError, 'workspace budget'):
                store._decode_once(path)
        self.assertGreaterEqual(calls, 8)
        self.assertEqual(store._decoded, {})

    def test_actual_output_graph_postguard_before_cache(self):
        path = self.saved(b'{"a":1}\n')
        store = self.store(maximum_matrix_bytes=128*1024)
        with patch('axiom_research.stock_matrix_reader._read', return_value={'oversized': 'x'*(256*1024)}):
            with self.assertRaisesRegex(ValueError, 'resident .*budget'):
                store._decode_once(path)
        self.assertEqual(store._decoded, {})
        self.assertEqual(store.metrics['json_decode_calls'], 1)

    def test_parent_size_and_stored_fingerprint_guard(self):
        path = self.saved(b'{"a":1}\n')
        store = self.store(maximum_parent_bytes=4)
        with patch('axiom_research.stock_matrix_reader._read', side_effect=AssertionError('oversize read')):
            with self.assertRaisesRegex(ValueError, 'parent byte budget'):
                store._decode_once(path)
        self.assertEqual(store._decoded, {})
        store = self.store()
        store._hash_once(path)
        path.write_bytes(b'{"a":2}\n')
        with self.assertRaisesRegex(ValueError, 'changed before decode'):
            store._decode_once(path)
        self.assertEqual(store._decoded, {})

    def test_mutation_during_preflight_and_original_read_never_caches(self):
        path = self.saved(b'{"a":[1,2,3]}\n')
        calls = 0
        def retained():
            nonlocal calls
            calls += 1
            if calls == 8:
                path.write_bytes(b'{"a":[3,2,1]}\n')
            return 0
        store = self.store(_caller_retained_bytes=retained)
        with self.assertRaisesRegex(ValueError, 'changed'):
            store._decode_once(path)
        self.assertEqual(store._decoded, {})
        store = self.store()
        def changed_read(target, **kwargs):
            result = _read(target, **kwargs)
            path.write_bytes(b'{"a":[1,2,3]}\n')
            return result
        with patch('axiom_research.stock_matrix_reader._read', changed_read):
            with self.assertRaisesRegex(ValueError, 'changed during decode'):
                store._decode_once(path)
        self.assertEqual(store._decoded, {})

    def test_existing_store_graph_accounting_workspace_is_charged_before_decode(self):
        existing=self.saved(json.dumps({str(i):i for i in range(400)}).encode(),'existing.json')
        store=self.store()
        store._decode_once(existing)
        from axiom_research.stock_matrix_reader import _resident_size
        store.maximum_matrix_bytes=_resident_size(store._decoded)+64*1024
        path=self.saved(b'{"a":1}\n','new.json')
        with patch('axiom_research.stock_matrix_reader._read',side_effect=AssertionError('unbudgeted graph walk')):
            with self.assertRaisesRegex(ValueError,'accounting workspace budget'):
                store._decode_once(path)
        self.assertNotIn(str(path),store._decoded)

    def test_replacement_path_cannot_change_the_preflighted_decode_stream(self):
        path=self.saved(b'{"a":1}\n')
        replacement=self.saved(b'{"a":"'+b'x'*(256*1024)+b'"}\n','replacement.json')
        store=self.store(maximum_matrix_bytes=128*1024)
        seen=[]
        def replacing_read(target,**kwargs):
            original=path.with_name('original.json')
            path.rename(original); replacement.rename(path)
            try:
                seen.append(_read(target,**kwargs))
                return seen[-1]
            finally:
                path.rename(replacement); original.rename(path)
        with patch('axiom_research.stock_matrix_reader._read',replacing_read):
            with self.assertRaisesRegex(ValueError,'changed during decode'):
                store._decode_once(path)
        self.assertEqual(seen,[{'a':1}])
        self.assertEqual(store._decoded,{})

    def test_growth_after_preflight_is_capped_before_whole_decode(self):
        path=self.saved(b'{"a":1}\n')
        store=self.store(maximum_matrix_bytes=128*1024)
        def growing_read(target,**kwargs):
            with path.open('ab') as stream: stream.write(b' '*(256*1024))
            return _read(target,**kwargs)
        with patch('axiom_research.stock_matrix_reader._read',growing_read):
            with self.assertRaisesRegex(ValueError,'byte length changed'):
                store._decode_once(path)
        self.assertEqual(store._decoded,{})

    def test_same_length_inplace_rewrite_is_rejected_before_dom_allocation(self):
        payload=b'"'+b'x'*8192+b'"'
        replacement=b'['+b'{},'*2730+b'{}]'
        self.assertEqual(len(payload),len(replacement))
        path=self.saved(payload)
        store=self.store()
        def rewritten_read(target,**kwargs):
            path.write_bytes(replacement)
            with patch('axiom_research.stock_artifacts.json.loads',side_effect=AssertionError('unadmitted DOM allocation')):
                return _read(target,**kwargs)
        with patch('axiom_research.stock_matrix_reader._read',rewritten_read):
            with self.assertRaisesRegex(ValueError,'stream bytes changed'):
                store._decode_once(path)
        self.assertEqual(store._decoded,{})


if __name__ == '__main__':
    unittest.main()

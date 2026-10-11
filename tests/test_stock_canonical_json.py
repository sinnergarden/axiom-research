import hashlib
import json
import math
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

from axiom_research.stock_canonical_json import validate_canonical_chunks


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


class CanonicalChunksTests(unittest.TestCase):
    def validate(self, chunks, maximum=8 * 1024 * 1024, retained=None):
        return validate_canonical_chunks(
            chunks, maximum_workspace_bytes=maximum,
            caller_retained_bytes=retained,
        )

    def assert_valid(self, raw, chunks=None):
        result = self.validate([raw] if chunks is None else chunks)
        self.assertEqual(set(result), {
            "digest", "size", "peak_workspace_bytes", "token_count",
        })
        self.assertEqual(result["digest"], "sha256:" + hashlib.sha256(raw).hexdigest())
        self.assertEqual(result["size"], len(raw))
        self.assertGreater(result["peak_workspace_bytes"], 0)
        self.assertLessEqual(result["peak_workspace_bytes"], 8 * 1024 * 1024)
        return result

    def test_native_scalar_and_container_goldens(self):
        values = [
            None, True, False, 0, -1, 10 ** 1000, 0.0, -0.0,
            5e-324, sys.float_info.max, 1e20, 1e-7, "",
            "ASCII/é雪🙂\"\\\b\f\n\r\t\x00", [], {},
            {"a": [1, True, None], "b": {"c": "雪"}},
            {"a": {}, "aa": [], "é": {"雪": False}, "🙂": -0.0},
        ]
        for value in values:
            with self.subTest(value_type=type(value).__name__):
                self.assert_valid(canonical(value))
        self.assertEqual(self.assert_valid(canonical(values[-2]))["token_count"], 7)
        self.assertEqual(self.assert_valid(b"{}")["token_count"], 0)
        self.assertEqual(self.assert_valid(b"[]")["token_count"], 0)
        self.assertEqual(self.assert_valid(b"null")["token_count"], 1)

    def test_all_two_chunk_cuts_and_single_byte_chunks(self):
        raw = canonical({
            "a": [None, True, False, -0.0, 1e-7, "雪🙂\"\\\n\x00"],
            "b": {"x": [], "y": {"é": 123456789}},
        })
        for cut in range(len(raw) + 1):
            with self.subTest(cut=cut):
                self.assert_valid(raw, [raw[:cut], b"", raw[cut:]])
        self.assert_valid(raw, [raw[i:i + 1] for i in range(len(raw))])

    def test_random_canonical_trees_and_chunk_boundaries(self):
        rng = random.Random(4106)

        def tree(depth):
            scalar = rng.choice([None, False, True, -0.0, 1e-9, "雪\\\n🙂", 17])
            if not depth:
                return scalar
            kind = rng.randrange(3)
            if kind == 0:
                return scalar
            if kind == 1:
                return [tree(depth - 1) for _ in range(rng.randrange(5))]
            return {f"k{i}": tree(depth - 1) for i in range(rng.randrange(5))}

        for _ in range(100):
            raw = canonical(tree(4))
            chunks = []
            cursor = 0
            while cursor < len(raw):
                width = rng.randrange(1, 12)
                chunks.append(raw[cursor:cursor + width])
                cursor += width
            self.assert_valid(raw, chunks)

    def test_small_mutations_match_native_roundtrip_oracle(self):
        rng = random.Random(6006)
        raw = canonical({"a": [1, -0.0, True, None, "雪\\\"\n"], "b": {"x": []}})
        alphabet = b'{}[],:"\\ \n\x00tfenul0123456789-+.eE\xff'
        for _ in range(400):
            position = rng.randrange(len(raw) + 1)
            change = bytes([rng.choice(alphabet)])
            operation = rng.randrange(3)
            if operation == 0:
                candidate = raw[:position] + change + raw[position:]
            elif operation == 1:
                candidate = raw[:position] + change + raw[position + 1:]
            else:
                candidate = raw[:position] + raw[position + 1:]
            try:
                expected = canonical(json.loads(candidate)) == candidate
            except (ValueError, UnicodeError, OverflowError):
                expected = False
            chunks = [candidate[i:i + 3] for i in range(0, len(candidate), 3)]
            with self.subTest(candidate=candidate):
                if expected:
                    self.assert_valid(candidate, chunks)
                else:
                    with self.assertRaises(ValueError):
                        self.validate(chunks)

    def test_noncanonical_valid_json_is_rejected(self):
        invalid = [
            b' {"a":1}', b'{"a": 1}', b'{"a":1}\n', b'null\t',
            b'{"b":1,"a":2}', b'{"a":1,"a":2}',
            b'{"a":1,"\\u0061":2}', b'1.00', b'-0', b'1E+20',
            b'1e15', b'1e-400', b'"\\u0061"', b'"\\/"',
            b'"\\u96ea"', b'"\\ud83d\\ude42"', b'"\\u000a"',
            b'"\\u001F"', b'\xef\xbb\xbf{}',
        ]
        for raw in invalid:
            for cut in (0, len(raw) // 2, len(raw)):
                with self.subTest(raw=raw, cut=cut):
                    with self.assertRaises(ValueError):
                        self.validate([raw[:cut], raw[cut:]])

    def test_syntax_eof_nonfinite_and_utf8_rejections(self):
        invalid = [
            b'', b' ', b'{', b'[', b'"', b'"a\\', b'"\\u12"',
            b'"\\q"', b'"a\x00b"', b'"a\\\nb"', b'"\xff"',
            b'"\xc0\xaf"', b'"\xed\xa0\x80"', b'"\xf4\x90\x80\x80"',
            b'"\\ud800"', b'"\\udc00"', b'\xff',
            b'01', b'+1', b'1.', b'.1', b'1e', b'1e+', b'truefalse',
            b'NaN', b'Infinity', b'-Infinity', b'1e400', b'-1e400',
            b'{1:2}', b'{"a"1}', b'{"a":}', b'{"a":1,}',
            b'{"a":1 "b":2}', b'[1,]', b'[,1]', b'[1:2]',
            b'{]', b'[}', b'[]{}', b'truefalse', b'null"x"',
        ]
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    self.validate([raw[i:i + 1] for i in range(len(raw))])
        self.assert_valid(b'"\xf4\x8f\xbf\xbf"')

    def test_large_token_budget_precedes_stdlib_decode(self):
        raw = canonical("x" * 16384)
        with patch("axiom_research.stock_canonical_json.json.loads") as decode:
            with self.assertRaisesRegex(ValueError, "budget"):
                self.validate([raw], maximum=128 * 1024)
            decode.assert_not_called()
        self.assert_valid(raw)

    def test_input_and_stack_and_previous_key_are_budgeted(self):
        with patch("axiom_research.stock_canonical_json.json.loads") as decode:
            with self.assertRaisesRegex(ValueError, "budget"):
                self.validate([b'"' + b'x' * 65536 + b'"'], maximum=32768)
            decode.assert_not_called()
        deep = b'[' * 2000 + b'0' + b']' * 2000
        self.assert_valid(deep)
        with self.assertRaisesRegex(ValueError, "budget"):
            self.validate([deep], maximum=32768)
        # Many live previous keys, although every individual key is small.
        key = canonical("k" * 1024)
        nested = (b'{' + key + b':') * 100 + b'0' + b'}' * 100
        self.assert_valid(nested)
        with self.assertRaisesRegex(ValueError, "budget"):
            self.validate([nested[i:i + 1024] for i in range(0, len(nested), 1024)],
                          maximum=100000)

    def test_combined_dynamic_caller_budget(self):
        self.assert_valid(b'"abc"')
        retained = [1000]
        first = self.validate([b'"abc"'], retained=lambda: retained[0])
        maximum = first["peak_workspace_bytes"] + retained[0]
        self.validate([b'"abc"'], maximum=maximum, retained=lambda: retained[0])
        with self.assertRaises(ValueError):
            self.validate([b'"abc"'], maximum=maximum - 1,
                          retained=lambda: retained[0])

        def chunks():
            yield b'{"a":'
            retained[0] = 1000000
            yield b'1}'

        with self.assertRaisesRegex(ValueError, "budget"):
            self.validate(chunks(), maximum=100000, retained=lambda: retained[0])

    def test_span_scanning_does_not_call_getter_per_byte_or_escape(self):
        for value in ("x" * 8192, "\\\"\n" * 2048):
            calls = [0]

            def retained():
                calls[0] += 1
                return 0

            self.validate([canonical(value)], retained=retained)
            self.assertLess(calls[0], 20)

    def test_strict_api_types_and_no_persistent_failure_state(self):
        for maximum in (True, False, 0, -1, 1.0, None):
            with self.subTest(maximum=maximum):
                with self.assertRaises(ValueError):
                    self.validate([b'0'], maximum=maximum)
        for retained in (0, "", lambda: True, lambda: -1, lambda: 0.0, lambda: None):
            with self.subTest(retained=retained):
                with self.assertRaises(ValueError):
                    self.validate([b'0'], retained=retained)
        for chunk in (bytearray(b'0'), memoryview(b'0'), "0", 0, None):
            with self.subTest(chunk_type=type(chunk).__name__):
                with self.assertRaises(ValueError):
                    self.validate([chunk])
        with self.assertRaises(ValueError):
            self.validate([b'{"b":0,"a":1}'])
        self.assert_valid(b'{"a":1,"b":0}')

    def test_current_python_integer_limit_is_not_changed(self):
        previous = sys.get_int_max_str_digits()
        if previous:
            with self.assertRaises(ValueError):
                self.validate([b'1' * (previous + 1)])
        self.assertEqual(sys.get_int_max_str_digits(), previous)
        self.assertTrue(math.copysign(1, json.loads('-0.0')) < 0)

    def test_leaf_import_does_not_require_business_or_numeric_packages(self):
        program = '''
import importlib.abc
import sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'axiom_data', 'axiom_engine', 'numpy', 'pandas'}:
            raise AssertionError('forbidden import ' + fullname)
sys.meta_path.insert(0, Reject())
from axiom_research.stock_canonical_json import validate_canonical_chunks
assert validate_canonical_chunks([b'{"a":1}'], maximum_workspace_bytes=100000)['size'] == 7
'''
        result = subprocess.run([sys.executable, "-B", "-c", program],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

"""Exact existing hashes/outputs against independently serialized legacy wire."""
from copy import deepcopy
from hashlib import sha256
import json
import math
import unittest
from unittest.mock import patch

from axiom_research import labels
from axiom_research.data_adapter import AdapterError
from test_stock_labels import CALENDAR, batch, build


def legacy_ref(value):
    return "sha256:" + sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def legacy_refs(records, field_meta, context):
    return {"source_ref": legacy_ref({"records": records, "field_meta": field_meta, "context": context}),
            "records_ref": legacy_ref(records), "field_meta_ref": legacy_ref(field_meta)}


class LabelEncodingReuseTests(unittest.TestCase):
    def test_exact_existing_refs_with_unicode_order_float_and_null(self):
        records = [{"session": "2024-01-02", "security_id": "中证", "price": -0.0,
                    "nullable": None, "large": 1e300, "tiny": 1e-300}]
        metadata = {"z": {"note": "引号\"换行\n", "boolean": True}, "a": [0, 1.25]}
        context = {"nested": {"z": 2, "a": 1}, "empty": []}
        self.assertEqual(labels._batch_source_refs(records, metadata, context),
                         legacy_refs(records, metadata, context))
        self.assertEqual(labels._batch_source_refs([], {}, {}), legacy_refs([], {}, {}))

    def test_one_encoding_per_subtree_and_no_retained_payload(self):
        values = ([{"value": 1.25}], {"field": None}, {"purpose": "label_outcomes"})
        with patch.object(labels, "_canonical_bytes", wraps=labels._canonical_bytes) as encode:
            refs = labels._batch_source_refs(*values)
        self.assertEqual(encode.call_count, 3)
        self.assertEqual([call.args[0] for call in encode.call_args_list],
                         [values[2], values[1], values[0]])
        self.assertTrue(all(type(v) is str and v.startswith("sha256:") for v in refs.values()))

    def test_complete_builder_same_legacy_bytes_and_checks(self):
        for features in ((CALENDAR[0],), (CALENDAR[0], CALENDAR[4], CALENDAR[-1])):
            for mutation in ("original", "missing", "future_clock", "unicode_provenance"):
                with self.subTest(features=features, mutation=mutation):
                    value = batch()
                    if mutation == "missing":
                        value.wire["records"][2]["open"] = None
                    elif mutation == "future_clock":
                        value.wire["field_meta"]["close"]["by_key"][10]["factor_provenance"]["usable_from"] = "2024-04-01T00:00:00Z"
                    elif mutation == "unicode_provenance":
                        value.wire["context"]["limitations"].append("冻结版本：α")
                    before = deepcopy(value.wire)
                    with patch.object(labels, "_batch_source_refs", side_effect=legacy_refs):
                        old = build(value, features=features)
                    new = build(value, features=features)
                    self.assertEqual(new, old)
                    self.assertEqual(legacy_ref(new), legacy_ref(old))
                    self.assertEqual(value.wire, before)

    def test_strict_json_errors_preserved_in_every_subtree(self):
        for bad in (math.nan, math.inf, object()):
            for position in range(3):
                values = [[], {}, {}]
                values[position] = [bad] if position == 0 else {"bad": bad}
                with self.subTest(position=position, bad=type(bad).__name__):
                    with self.assertRaisesRegex(AdapterError, "strict JSON"):
                        labels._batch_source_refs(*values)

    def test_changed_source_version_is_not_reused_across_calls(self):
        value = batch()
        first = build(value)
        value.wire["context"]["limitations"].append("new revision")
        second = build(value)
        self.assertNotEqual(first["source_ref"], second["source_ref"])
        self.assertNotEqual(first["label_ref"], second["label_ref"])

    def test_surrogate_and_circular_json_keep_error_behavior(self):
        circular = {}
        circular["self"] = circular
        for invalid in ({"text": "\ud800"}, circular):
            with self.subTest(circular=invalid is circular):
                with self.assertRaisesRegex(AdapterError, "strict JSON"):
                    labels._content_ref({"records": [], "field_meta": {}, "context": invalid})
                with self.assertRaisesRegex(AdapterError, "strict JSON"):
                    labels._batch_source_refs([], {}, invalid)

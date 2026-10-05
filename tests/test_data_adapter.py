"""Independent Data-to-Core examples with keyed provenance and null history."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from axiom_research.data_adapter import AdapterError, adapt_decision_batch
import axiom_research.data_adapter as adapter
from axiom_research import ViewRef, validate

REF = "sha256:" + "a" * 64
SESSIONS = ["2025-01-02", "2025-01-03", "2025-01-06"]


class Batch:
    def __init__(self, wire):
        self.wire = wire

    def to_json(self):
        return self.wire


def batch(field, values, *, membership=False, purpose="decision_facts"):
    records, metadata = [], []
    for session, value in zip(SESSIONS, values):
        records.append(dict(security_id="A", session=session, **{field: value}))
        metadata.append(dict(security_id="A", session=session,
                             usable_from=session + "T10:00:00+00:00" if value is not None else None,
                             first_observed_at=session + "T10:00:00+00:00" if value is not None else None,
                             revision_id="r1" if value is not None else None,
                             availability_basis="synthetic", evidence_ref="fixture",
                             missing_reason=None if value is not None else "source_missing"))
    return Batch({"records": records,
                  "field_meta": {field: {"dtype": "bool" if membership else "float64",
                                          "unit": None if membership else "CNY/share",
                                          "by_key": list(reversed(metadata))}},
                  "context": {"contract_version": "data_batch_v1", "snapshot_id": "s1",
                              "domain": "universe_membership" if membership else "daily_bars",
                              "reader_version": "local_reader_v1",
                              "query": {"fields": [field], "symbols": ["A"], "sessions": SESSIONS,
                                        "pit_policy": "market_pit_safe_v1", "policy_by_session": None,
                                        "cutoff_by_session": {s: s + "T23:00:00+00:00" for s in SESSIONS},
                                        "purpose": purpose, "price_basis": "unadjusted",
                                        "adjustment_anchor": None,
                                        "universe_id": "U" if membership else None},
                              "limitations": []}})


class AdapterTests(unittest.TestCase):
    def test_one_serialized_snapshot_preserves_full_batch_and_view_refs(self):
        class CountedBatch(Batch):
            reads = 0

            def to_json(self):
                self.reads += 1
                return deepcopy(self.wire)

        for mode in ("cell", "batch_field"):
            with self.subTest(mode=mode):
                price = CountedBatch(batch("close", [10.0, None, 12.0]).wire)
                reference = CountedBatch(batch("is_member", [True] * 3, membership=True).wire)
                price.wire["context"].update(derivation={"recipe_version": "fixture"},
                                             event_reader_version="fixture_event_reader")
                before = deepcopy((price.wire, reference.wire))
                expected_view = ViewRef.from_batch(Batch(before[0]))
                adapted = adapt_decision_batch(price, reference=reference, recipe_ref=REF,
                                               source_granularity=mode)
                self.assertEqual((price.reads, reference.reads), (1, 1))
                self.assertEqual(adapted.view_ref.to_dict(), expected_view.to_dict())
                self.assertEqual(adapted.view_ref.digest, expected_view.digest)
                self.assertEqual((price.wire, reference.wire), before)
                if mode == "batch_field":
                    self.assertEqual({v["batch_ref"] for v in adapted.source_evidence.values()},
                                     {adapter._digest(wire) for wire in before})

    def test_compact_source_hash_reuse_keeps_every_cell_provenance(self):
        price = batch("close", [10.0, None, 12.0])
        reference = batch("is_member", [True] * 3, membership=True)
        for mode, expected_hashes in (("batch_field", 2), ("cell", 6)):
            with self.subTest(mode=mode), patch.object(adapter, "_digest", wraps=adapter._digest) as hash_value:
                adapted = adapt_decision_batch(price, reference=reference, recipe_ref=REF,
                                               source_granularity=mode)
                shape = ({"field", "batch_ref", "qualification", "basis"} if mode == "batch_field"
                         else {"field", "key", "meta", "reference"})
                source_calls = [call for call in hash_value.call_args_list if set(call.args[0]) == shape]
                self.assertEqual(len(source_calls), expected_hashes)
            for field, value in (("close", price), ("is_member", reference)):
                observed = [meta for item in adapted.source_evidence.values() if item["field"] == field
                            for meta in (item["provenance_by_key"] if mode == "batch_field"
                                         else [item["provenance"]])]
                self.assertEqual(observed, sorted(value.wire["field_meta"][field]["by_key"],
                                                 key=lambda meta: (meta["security_id"], meta["session"])))

    def test_compact_grouping_recomputes_qualification_and_does_not_survive_call(self):
        price = batch("close", [10.0, 11.0, 12.0])
        reference = batch("is_member", [True] * 3, membership=True)
        by_session = {meta["session"]: meta for meta in price.wire["field_meta"]["close"]["by_key"]}
        for meta in by_session.values():
            meta["availability_basis"] = "source_receipt"
        by_session[SESSIONS[1]]["evidence_ref"] = None
        by_session[SESSIONS[2]].update(availability_basis="declared_vendor_assumption",
                                     evidence_ref=None, first_observed_at=None)
        with patch.object(adapter, "_digest", wraps=adapter._digest) as hash_value:
            first = adapt_decision_batch(price, reference=reference, recipe_ref=REF,
                                         source_granularity="batch_field")
            source_calls = [call.args[0] for call in hash_value.call_args_list
                            if set(call.args[0]) == {"field", "batch_ref", "qualification", "basis"}]
            self.assertEqual(len(source_calls), 4)
            hash_value.reset_mock()
            by_session[SESSIONS[2]]["revision_id"] = "r2"
            second = adapt_decision_batch(price, reference=reference, recipe_ref=REF,
                                          source_granularity="batch_field")
            self.assertEqual(sum(set(call.args[0]) == {"field", "batch_ref", "qualification", "basis"}
                                 for call in hash_value.call_args_list), 4)
        self.assertEqual({(source["qualification"], source["availability_basis"])
                          for source in first.facts.to_dict()["sources"]
                          if source["availability_basis"] != "synthetic"},
                         {("verified", "source_receipt"), ("observed", "source_receipt"),
                          ("best_effort", "declared_vendor_assumption")})
        self.assertNotEqual(first.facts.identity, second.facts.identity)
        self.assertEqual(first.view_ref.digest, second.view_ref.digest)

    def test_cached_group_never_skips_later_cell_validation(self):
        cases = ("future_clock", "unknown_member", "nonfinite", "inexact_integer",
                 "present_missing_reason", "invalid_nested_provenance")
        for mode in ("cell", "batch_field"):
            for case in cases:
                with self.subTest(mode=mode, case=case):
                    price = batch("close", [10.0, 11.0, 12.0])
                    reference = batch("is_member", [True] * 3, membership=True)
                    last_meta = next(meta for meta in price.wire["field_meta"]["close"]["by_key"]
                                     if meta["session"] == SESSIONS[-1])
                    if case == "future_clock":
                        last_meta["usable_from"] = SESSIONS[-1] + "T23:00:00.000001Z"
                    elif case == "unknown_member":
                        reference.wire["records"][-1]["is_member"] = None
                    elif case in ("nonfinite", "inexact_integer"):
                        price.wire["records"][-1]["close"] = float("inf") if case == "nonfinite" else 2**53 + 1
                    elif case == "present_missing_reason":
                        last_meta["missing_reason"] = "source_missing"
                    else:
                        last_meta["factor_provenance"] = "invalid"
                    with self.assertRaises(ValueError):
                        adapt_decision_batch(price, reference=reference, recipe_ref=REF,
                                             source_granularity=mode)

    def test_real_observation_precision_and_vendor_assumption_are_preserved(self):
        price = batch("close", [10.0, 11.0, 12.0])
        reference = batch("is_member", [True] * 3, membership=True)
        meta = price.wire["field_meta"]["close"]["by_key"][0]
        meta.update(usable_from=meta["session"] + "T12:00:00.123456Z",
                    first_observed_at="2026-09-28T12:00:00.123456Z",
                    availability_basis="declared_vendor_assumption")
        adapted = adapt_decision_batch(price, reference=reference, recipe_ref=REF)
        record = next(row for row in adapted.facts.to_dict()["rows"] if row["session"] == meta["session"])
        self.assertEqual(record["availability"][0], meta["session"] + "T12:00:01Z")
        self.assertTrue(any(v["provenance"].get("usable_from") == meta["usable_from"]
                            for v in adapted.source_evidence.values()))
        sources = adapted.facts.to_dict()["sources"]
        self.assertTrue(any(source["qualification"] == "best_effort" for source in sources))

    def test_core_executes_identity_lag_return_with_metadata(self):
        price = batch("close", [10.0, None, 20.0])
        membership = batch("is_member", [True, True, True], membership=True)
        adapted = adapt_decision_batch(price, reference=membership, recipe_ref=REF)
        result = adapted.execute().to_dict()
        self.assertEqual([r["values"] for r in result["rows"]],
                         [[10.0, None, None], [None, 10.0, None], [20.0, None, None]])
        self.assertEqual(result["schema"][0]["unit"], "CNY/share")
        self.assertEqual(result["schema"][-1]["unit"], "dimensionless")
        self.assertEqual(adapted.context.to_dict()["cutoffs"][SESSIONS[0]],
                         "2025-01-02T23:00:00Z")
        self.assertTrue(any(v["provenance"]["revision_id"] == "r1"
                            for v in adapted.source_evidence.values()))
        legacy_ref = adapted.view_ref.as_artifact_ref()
        validate(legacy_ref)
        self.assertEqual(legacy_ref.content_digest, adapted.view_ref.digest)
        self.assertEqual(legacy_ref.metadata["logical_view"]["snapshot_id"], "s1")
        frozen_digest = adapted.view_ref.digest
        price.wire["context"]["query"]["fields"].append("later")
        self.assertEqual(adapted.view_ref.digest, frozen_digest)

    def test_rejects_wrong_purpose_and_unmatched_membership(self):
        reference = batch("is_member", [True] * 3, membership=True)
        with self.assertRaisesRegex(AdapterError, "decision_facts"):
            adapt_decision_batch(batch("close", [1, 2, 3], purpose="label_outcomes"),
                                 reference=reference, recipe_ref=REF)
        reference.wire["context"]["snapshot_id"] = "s2"
        with self.assertRaisesRegex(AdapterError, "mismatch"):
            adapt_decision_batch(batch("close", [1, 2, 3]), reference=reference, recipe_ref=REF)

    def test_adjusted_provenance_flattens_price_factor_and_anchor(self):
        price = batch("close", [10.0, 11.0, 12.0])
        price.wire["context"]["query"]["price_basis"] = "common_anchor_adjusted_v1"
        price.wire["context"]["query"]["adjustment_anchor"] = SESSIONS[-1]
        price.wire["context"]["derivation"] = {"recipe_version": "common_anchor_price_v1"}
        meta = price.wire["field_meta"]["close"]["by_key"][-1]
        original = dict(meta)
        meta.clear()
        meta.update(security_id="A", session=SESSIONS[0], missing_reason=None,
                    price_provenance=original,
                    factor_provenance={**original, "usable_from": "2025-01-02T11:00:00Z"},
                    anchor_factor_provenance={**original, "usable_from": "2025-01-02T12:00:00Z"})
        adapted = adapt_decision_batch(price,
            reference=batch("is_member", [True] * 3, membership=True), recipe_ref=REF)
        fact = adapted.facts.to_dict()["rows"][0]
        self.assertEqual(fact["availability"][0], "2025-01-02T12:00:00Z")
        self.assertEqual(adapted.view_ref.to_dict()["derivation"]["recipe_version"],
                         "common_anchor_price_v1")


if __name__ == "__main__":
    unittest.main()

"""Independent Data-to-Core examples with keyed provenance and null history."""
import unittest

from axiom_research.data_adapter import AdapterError, adapt_decision_batch
from axiom_research import validate

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

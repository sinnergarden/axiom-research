"""Outcome-only section membership, maturity and existing-Core execution."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from axiom_engine.core import ExecutionContext, FeatureFrame, FeaturePlan
from axiom_research.data_adapter import AdapterError
from axiom_research.stock_artifacts import digest
from axiom_research.stock_label_normalization import (
    NORMALIZATION_SPEC, normalize_forward_labels,
)
import axiom_research.stock_label_normalization as normalization


CUTOFF = "2024-02-23T18:00:00Z"
SESSIONS = ("2024-02-07", "2024-02-19")  # Exchange holiday, deliberately irregular.


def seal(build, ref):
    build.pop(ref, None)
    build[ref] = digest(build)
    return build


def inputs(securities=("A", "B"), sessions=SESSIONS):
    source = digest({"fixture": "frozen_outcomes"})
    raw = {"contract_version": "stock_label_build_v1",
           "label_spec": {"horizon_sessions": 5, "normalization": "none",
                          "price_basis": "common_anchor_adjusted_v1"},
           "source_ref": source, "calendar_ref": digest({"actual_sessions": list(sessions)}),
           "rows": []}
    features = {"contract_version": "stock_feature_build_v1", "ordered_features": ["MOM010", "VOL010"],
                "rows": []}
    for session in sessions:
        for index, security in enumerate(securities):
            raw["rows"].append({"security_id": security, "feature_session": session,
                "start_session": "2024-02-20", "end_session": "2024-02-22",
                "return": float(index * 2), "label_available_at": "2024-02-22T08:00:00Z",
                "valid": True, "invalid_reason": None, "source_refs": [source]})
            features["rows"].append({"security_id": security, "session": session,
                "values": [0.1, 0.2], "validity": [True, True], "member": True,
                "knowledge_cutoff": session + "T18:00:00Z"})
    return seal(raw, "label_ref"), seal(features, "feature_ref")


def row(build, security, session=SESSIONS[0]):
    return next(item for item in build["rows"] if item["security_id"] == security and
                item.get("feature_session", item.get("session")) == session)


class LabelNormalizationTests(unittest.TestCase):
    def test_existing_raw_builder_on_irregular_calendar_including_uncovered_tail(self):
        from test_stock_labels import batch, build, CALENDAR, CUTOFF as outcome_cutoff
        value = batch()
        for item in value.wire["records"]:
            if item["security_id"] == "B":
                item["close"] = 25.0
        sessions = (CALENDAR[0], CALENDAR[1], CALENDAR[-1])
        raw = build(value, features=sessions)
        _, features = inputs(sessions=sessions)
        before = deepcopy(raw)
        result = normalize_forward_labels(raw, features=features, cutoff=outcome_cutoff)
        for session in sessions[:2]:
            self.assertEqual(row(result, "A", session)["raw_return"], 0.5)
            self.assertEqual(row(result, "B", session)["raw_return"], 1.5)
            self.assertEqual(row(result, "A", session)["normalized_target"], -1.0)
            self.assertEqual(row(result, "B", session)["normalized_target"], 1.0)
        self.assertEqual(result["sections"][-1]["eligible_keys"], [])
        self.assertEqual(row(result, "A", sessions[-1])["invalid_reason"], "calendar_endpoint_uncovered")
        self.assertEqual(raw, before)

    def test_actual_sections_population_scale_and_core_execution_evidence(self):
        raw, features = inputs()
        with patch.object(normalization, "execute_feature_plan",
                          wraps=normalization.execute_feature_plan) as execute:
            result = normalize_forward_labels(raw, features=features, cutoff=CUTOFF)
        self.assertEqual(execute.call_count, len(SESSIONS))
        self.assertEqual(len(result["core_frames"]), len(SESSIONS))
        self.assertEqual([section["feature_session"] for section in result["sections"]], list(SESSIONS))
        for session in SESSIONS:
            self.assertEqual(row(result, "A", session)["normalized_target"], -1.0)
            self.assertEqual(row(result, "B", session)["normalized_target"], 1.0)
        self.assertEqual(result["normalization_spec"], NORMALIZATION_SPEC)
        for section, p, c, f in zip(result["sections"], result["core_plan"],
                                     result["core_context"], result["core_frames"]):
            self.assertEqual(FeaturePlan.from_dict(p).identity, section["core_plan_ref"])
            self.assertEqual(ExecutionContext.from_dict(c).identity, section["core_context_ref"])
            self.assertEqual(FeatureFrame.from_dict(f).identity, section["frame_ref"])
            self.assertEqual(f["fact_identity"], section["fact_ref"])
            self.assertEqual(p["nodes"][0]["op"], "cs_zscore")
            self.assertEqual(p["nodes"][0]["params"], NORMALIZATION_SPEC["params"])
            self.assertEqual(c["sessions"], [section["feature_session"]])
            self.assertEqual(c["cutoffs"], {section["feature_session"]: CUTOFF})
        self.assertEqual(result["frame_ref"], digest([
            {"feature_session": section["feature_session"], "frame_ref": section["frame_ref"]}
            for section in result["sections"]]))

    def test_frame_identity_is_resolved_once_per_immutable_section(self):
        raw, features = inputs(("A", "B", "C"))
        original = normalization.execute_feature_plan
        resolved = []

        class CountedFrame:
            def __init__(self, frame):
                self.frame, self.identity_reads = frame, 0

            @property
            def identity(self):
                self.identity_reads += 1
                return self.frame.identity

            def to_dict(self):
                return self.frame.to_dict()

        def execute(*args):
            frame = CountedFrame(original(*args))
            resolved.append(frame)
            return frame

        with patch.object(normalization, "execute_feature_plan", side_effect=execute):
            result = normalize_forward_labels(raw, features=features, cutoff=CUTOFF)
        self.assertEqual([frame.identity_reads for frame in resolved], [1] * len(SESSIONS))
        self.assertEqual(result, normalize_forward_labels(raw, features=features, cutoff=CUTOFF))

    def test_membership_feature_missing_invalid_and_immature_do_not_change_section(self):
        raw, features = inputs(("A", "B", "C", "D", "E", "F"), sessions=(SESSIONS[0],))
        row(features, "C")["member"] = False
        row(features, "D")["values"][0] = None
        row(features, "E")["validity"][1] = False
        row(raw, "F")["label_available_at"] = "2024-02-24T00:00:00Z"
        seal(raw, "label_ref")
        seal(features, "feature_ref")
        result = normalize_forward_labels(raw, features=features, cutoff=CUTOFF)
        self.assertEqual(result["sections"][0]["eligible_keys"], [["A", SESSIONS[0]], ["B", SESSIONS[0]]])
        self.assertEqual([row(result, s)["normalized_target"] for s in ("A", "B")], [-1.0, 1.0])
        for security, reason in (("C", "NOT_MEMBER"), ("D", "FEATURE_MISSING"),
                                 ("E", "FEATURE_INVALID"), ("F", "LABEL_NOT_MATURE")):
            self.assertEqual(row(result, security)["invalid_reason"], reason)
            self.assertIsNone(row(result, security)["normalized_target"])
            self.assertFalse(row(result, security)["valid"])
        self.assertEqual(len(result["rows"]), 6)

    def test_constant_section_remains_missing_without_zero_fill(self):
        raw, features = inputs()
        for item in raw["rows"]:
            item["return"] = 0.25
        seal(raw, "label_ref")
        result = normalize_forward_labels(raw, features=features, cutoff=CUTOFF)
        self.assertTrue(all(item["normalized_target"] is None and not item["valid"] and
                            item["invalid_reason"] == "NORMALIZATION_UNDEFINED" for item in result["rows"]))
        self.assertTrue(all(len(section["eligible_keys"]) == 2 for section in result["sections"]))

    def test_raw_missing_and_empty_section_preserve_endpoint_clock_and_reason(self):
        raw, features = inputs(sessions=(SESSIONS[0],))
        for item in raw["rows"]:
            item.update({"return": None, "valid": False, "invalid_reason": "missing_end_close",
                         "label_available_at": None, "end_session": None})
        seal(raw, "label_ref")
        result = normalize_forward_labels(raw, features=features, cutoff=CUTOFF)
        self.assertEqual(result["sections"][0]["eligible_keys"], [])
        for item in result["rows"]:
            self.assertIsNone(item["raw_return"])
            self.assertIsNone(item["end_session"])
            self.assertIsNone(item["label_available_at"])
            self.assertEqual(item["invalid_reason"], "missing_end_close")

    def test_precise_maturity_and_future_endpoint_guard(self):
        raw, features = inputs(("A", "B", "C", "D"), sessions=(SESSIONS[0],))
        cutoff = "2024-02-23T18:00:00.123456Z"
        row(raw, "A")["label_available_at"] = cutoff
        row(raw, "C")["label_available_at"] = "2024-02-23T18:00:00.123457Z"
        row(raw, "D")["end_session"] = "2024-02-24"
        seal(raw, "label_ref")
        result = normalize_forward_labels(raw, features=features, cutoff=cutoff)
        self.assertEqual(result["sections"][0]["eligible_keys"], [["A", SESSIONS[0]], ["B", SESSIONS[0]]])
        self.assertTrue(row(result, "A")["valid"])
        self.assertEqual(row(result, "A")["label_available_at"], cutoff)
        self.assertEqual(row(result, "A")["normalized_available_at"], "2024-02-23T18:00:01Z")
        self.assertEqual(result["core_context"][0]["cutoffs"][SESSIONS[0]], "2024-02-23T18:00:01Z")
        self.assertEqual(row(result, "C")["invalid_reason"], "LABEL_NOT_MATURE")
        self.assertEqual(row(result, "D")["invalid_reason"], "LABEL_NOT_MATURE")

    def test_missing_raw_clock_and_future_feature_context_are_excluded(self):
        raw, features = inputs(("A", "B", "C", "D"), sessions=(SESSIONS[0],))
        row(raw, "C")["label_available_at"] = None
        row(features, "D")["knowledge_cutoff"] = "2024-02-24T18:00:00Z"
        seal(raw, "label_ref")
        seal(features, "feature_ref")
        result = normalize_forward_labels(raw, features=features, cutoff=CUTOFF)
        self.assertEqual(row(result, "C")["invalid_reason"], "LABEL_CLOCK_OR_ENDPOINT_UNKNOWN")
        self.assertEqual(row(result, "D")["invalid_reason"], "FEATURE_NOT_AVAILABLE")
        self.assertTrue(row(result, "A")["valid"])

    def test_missing_feature_key_is_preserved_and_extra_feature_dates_are_allowed(self):
        raw, features = inputs()
        features["rows"] = [r for r in features["rows"] if (r["security_id"], r["session"]) != ("B", SESSIONS[0])]
        extra = deepcopy(features["rows"][0])
        extra.update(session="2024-02-26", knowledge_cutoff="2024-02-26T18:00:00Z")
        features["rows"].append(extra)
        seal(features, "feature_ref")
        result = normalize_forward_labels(raw, features=features, cutoff=CUTOFF)
        self.assertEqual(row(result, "B")["invalid_reason"], "FEATURE_MISSING")
        self.assertTrue(row(result, "A", SESSIONS[1])["valid"])
        self.assertEqual(len(result["rows"]), len(raw["rows"]))

    def test_same_inputs_sealed_identity_and_no_mutation(self):
        raw, features = inputs()
        before = deepcopy((raw, features))
        first = normalize_forward_labels(raw, features=features, cutoff=CUTOFF)
        second = normalize_forward_labels(raw, features=features, cutoff=CUTOFF)
        self.assertEqual(first, second)
        self.assertEqual((raw, features), before)
        self.assertEqual(first["label_ref"], digest({k: v for k, v in first.items() if k != "label_ref"}))
        self.assertEqual(first["raw_label_ref"], raw["label_ref"])
        self.assertEqual(first["feature_ref"], features["feature_ref"])
        self.assertEqual(raw["label_spec"]["normalization"], "none")
        for output in first["rows"]:
            section = next(s for s in first["sections"] if s["feature_session"] == output["feature_session"])
            self.assertTrue({raw["label_ref"], features["feature_ref"], section["section_ref"],
                             section["frame_ref"]} <= set(output["source_refs"]))
        changed = normalize_forward_labels(raw, features=features, cutoff="2024-02-24T18:00:00Z")
        self.assertNotEqual(first["label_ref"], changed["label_ref"])
        first["normalization_spec"]["params"]["ddof"] = 1
        self.assertEqual(NORMALIZATION_SPEC["params"]["ddof"], 0)

    def test_tampered_input_and_incomplete_raw_grid_are_rejected(self):
        raw, features = inputs()
        row(raw, "A")["return"] = 0.75
        with self.assertRaisesRegex(AdapterError, "integrity mismatch"):
            normalize_forward_labels(raw, features=features, cutoff=CUTOFF)
        seal(raw, "label_ref")
        raw["rows"].pop()
        seal(raw, "label_ref")
        with self.assertRaisesRegex(AdapterError, "complete raw label"):
            normalize_forward_labels(raw, features=features, cutoff=CUTOFF)


if __name__ == "__main__":
    unittest.main()

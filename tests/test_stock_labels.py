"""Independent endpoint, actual-calendar and label visibility contracts."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
import unittest

from axiom_research.data_adapter import AdapterError
from axiom_research.labels import build_forward_labels, mature_training_rows


CALENDAR = ("2024-02-07", "2024-02-08", "2024-02-19", "2024-02-20",
            "2024-02-21", "2024-02-22", "2024-02-23")
CUTOFF = "2024-03-10T18:00:00Z"


class Batch:
    def __init__(self, wire):
        self.wire = wire

    def to_json(self):
        return self.wire


def native_batch(domain, fields, sessions=CALENDAR, symbols=("A", "B")):
    records, meta = [], {field: {"dtype": "float64", "unit": "CNY/share" if
        domain == "market_daily" else "dimensionless", "by_key": []} for field in fields}
    for session in sessions:
        for symbol in symbols:
            row = dict(security_id=symbol, session=session)
            for field in fields:
                row[field] = 10.0 if field == "open" else 15.0 if field == "close" else 1.0
                meta[field]["by_key"].append(dict(security_id=symbol, session=session,
                    usable_from=session + "T08:15:00Z", missing_reason=None,
                    revision_id=f"{domain}:{symbol}:{session}", availability_basis="synthetic"))
            records.append(row)
    return {"records": records, "field_meta": meta,
        "context": {"contract_version": "data_batch_v1", "snapshot_id": "s_frozen_stock_fixture",
            "reader_version": "local_reader_v5", "domain": domain, "contract_id": domain + "_v1",
            "source_profile_id": "synthetic", "limitations": ["synthetic input"], "query": {
                "domain": domain, "fields": list(fields), "symbols": list(symbols),
                "sessions": list(sessions), "cutoff_by_session": {s: CUTOFF for s in sessions},
                "pit_policy": "market_pit_safe_v1", "policy_by_session": None,
                "purpose": "label_outcomes", "price_basis": "unadjusted",
                "adjustment_anchor": None, "universe_id": None}}}


def batch():
    prices = native_batch("market_daily", ("open", "close"))
    factors = native_batch("adjustment_factors", ("factor",))
    wire = deepcopy(prices)
    wire["context"]["query"].update(price_basis="common_anchor_adjusted_v1",
                                      adjustment_anchor=CALENDAR[-1])
    wire["context"]["derivation"] = dict(recipe_version="common_anchor_price_v1",
        formula="price_t * factor_t / factor_anchor", factor_field="factor",
        anchor_session=CALENDAR[-1], decision_session=CALENDAR[-1], decision_cutoff=CUTOFF,
        price_query=deepcopy(prices["context"]["query"]),
        factor_query=deepcopy(factors["context"]["query"]))
    factor_meta = {(m["security_id"], m["session"]): m
                   for m in factors["field_meta"]["factor"]["by_key"]}
    for field in ("open", "close"):
        wire["field_meta"][field]["recipe_version"] = "common_anchor_price_v1"
        wire["field_meta"][field]["by_key"] = [dict(
            security_id=m["security_id"], session=m["session"], missing_reason=None,
            price_provenance=deepcopy(m),
            factor_provenance=deepcopy(factor_meta[m["security_id"], m["session"]]),
            anchor_factor_provenance=deepcopy(factor_meta[m["security_id"], CALENDAR[-1]]))
            for m in wire["field_meta"][field]["by_key"]]
    return Batch(wire)


def endpoint_meta(value, field="close", symbol="A", session=CALENDAR[5]):
    return next(m for m in value.wire["field_meta"][field]["by_key"]
                if (m["security_id"], m["session"]) == (symbol, session))


def build(value=None, features=(CALENDAR[0],), horizon=5):
    return build_forward_labels(value or batch(), calendar=CALENDAR,
        feature_sessions=features, horizon_sessions=horizon)


class StockLabelTests(unittest.TestCase):
    def test_irregular_exchange_calendar_and_open_to_fifth_close(self):
        result = build()
        self.assertEqual(len(result["rows"]), 2)
        for row in result["rows"]:
            self.assertEqual((row["start_session"], row["end_session"]),
                             ("2024-02-08", "2024-02-22"))
            self.assertEqual(row["return"], 0.5)
            self.assertTrue(row["valid"])
            # The shared anchor was observed after the end-session price.
            self.assertEqual(row["label_available_at"], "2024-02-23T08:15:00Z")

    def test_same_input_has_same_digests_and_does_not_mutate_batch(self):
        value = batch()
        before = deepcopy(value.wire)
        first, second = build(value), build(value)
        self.assertEqual(first, second)
        self.assertEqual(before, value.wire)
        self.assertEqual(first["source_ref"], first["rows"][0]["source_refs"][0])
        json.dumps(first, allow_nan=False)
        value.wire["context"]["limitations"].append("different frozen provenance")
        changed = build(value)
        self.assertNotEqual(changed["source_ref"], first["source_ref"])
        self.assertNotEqual(changed["label_ref"], first["label_ref"])

    def test_incomplete_tail_preserves_whole_security_feature_grid(self):
        result = build(features=(CALENDAR[0], CALENDAR[4], CALENDAR[-1]))
        self.assertEqual(len(result["rows"]), 6)
        invalid = [row for row in result["rows"] if not row["valid"]]
        self.assertEqual(len(invalid), 4)
        self.assertTrue(all(row["return"] is None and row["label_available_at"] is None and
                            row["invalid_reason"] == "calendar_endpoint_uncovered" for row in invalid))
        self.assertIsNone(invalid[-1]["start_session"])
        self.assertIsNone(invalid[-1]["end_session"])

    def test_late_nested_clock_and_subsecond_fit_boundary(self):
        for name in ("price_provenance", "factor_provenance", "anchor_factor_provenance"):
            with self.subTest(name=name):
                value = batch()
                endpoint_meta(value)[name]["usable_from"] = "2024-03-01T12:00:00.123456Z"
                result = build(value)
                first = result["rows"][0]
                self.assertEqual(first["label_available_at"], "2024-03-01T12:00:00.123456Z")
                before = mature_training_rows(result, "2024-03-01T12:00:00.123455Z")
                self.assertEqual([r["security_id"] for r in before], ["B"])
                on = mature_training_rows(result, "2024-03-01T20:00:00.123456+08:00")
                self.assertEqual([r["security_id"] for r in on], ["A", "B"])

    def test_missing_clock_never_uses_query_cutoff_or_outer_clock(self):
        for name in ("price_provenance", "factor_provenance", "anchor_factor_provenance"):
            with self.subTest(name=name):
                value = batch()
                meta = endpoint_meta(value)
                meta["usable_from"] = "2024-02-22T08:15:00Z"
                meta[name].pop("usable_from")
                row = build(value)["rows"][0]
                self.assertFalse(row["valid"])
                self.assertIsNone(row["return"])
                self.assertIsNone(row["label_available_at"])
                self.assertEqual(row["invalid_reason"], "unknown_end_close_availability")

    def test_invalid_unknown_or_unavailable_endpoint(self):
        for clock, reason in ((None, "unknown_end_close_availability"),
                              ("2024-02-22T08:15:00", "unknown_end_close_availability"),
                              ("2024-03-11T00:00:00Z",
                               "unavailable_end_close:provenance_exceeds_query_cutoff")):
            with self.subTest(clock=clock):
                value = batch()
                endpoint_meta(value)["factor_provenance"]["usable_from"] = clock
                self.assertEqual(build(value)["rows"][0]["invalid_reason"], reason)
        value = batch()
        endpoint_meta(value)["factor_provenance"]["missing_reason"] = "not_visible_at_cutoff"
        self.assertEqual(build(value)["rows"][0]["invalid_reason"],
                         "unavailable_end_close:not_visible_at_cutoff")

    def test_missing_and_nonpositive_values_are_null_not_zero_returns(self):
        for field, session in (("open", CALENDAR[1]), ("close", CALENDAR[5])):
            for missing in (None, 0.0, -1.0):
                with self.subTest(field=field, missing=missing):
                    value = batch()
                    row = next(r for r in value.wire["records"]
                               if (r["security_id"], r["session"]) == ("A", session))
                    row[field] = missing
                    result = build(value)
                    self.assertFalse(result["rows"][0]["valid"])
                    self.assertIsNone(result["rows"][0]["return"])
                    self.assertEqual(len(mature_training_rows(result, CUTOFF)), 1)
        value = batch()
        value.wire["records"] = [r for r in value.wire["records"] if
            (r["security_id"], r["session"]) != ("A", CALENDAR[5])]
        self.assertEqual(build(value)["rows"][0]["invalid_reason"], "missing_end_close")

    def test_maturity_checks_endpoint_date_as_well_as_known_clock(self):
        value = batch()
        for field in ("open", "close"):
            for meta in value.wire["field_meta"][field]["by_key"]:
                for name in ("price_provenance", "factor_provenance", "anchor_factor_provenance"):
                    meta[name]["usable_from"] = "2024-02-07T00:00:00Z"
        result = build(value)
        self.assertEqual(mature_training_rows(result, "2024-02-21T23:59:59Z"), [])
        self.assertEqual(len(mature_training_rows(result, "2024-02-22T00:00:00Z")), 2)
        with self.assertRaisesRegex(AdapterError, "timezone-aware"):
            mature_training_rows(result, "2024-02-23")
        changed = deepcopy(result)
        changed["rows"][0]["return"] = 9.0
        with self.assertRaisesRegex(AdapterError, "integrity"):
            mature_training_rows(changed, CUTOFF)

    def test_fixed_outcome_context_and_real_derivation_required(self):
        mutations = [lambda w: w["context"]["query"].update(purpose="decision_facts"),
                     lambda w: w["context"].update(snapshot_id="latest"),
                     lambda w: w["context"].pop("derivation"),
                     lambda w: w["context"]["derivation"]["price_query"].update(purpose="decision_facts"),
                     lambda w: w["context"]["query"].update(price_basis="unadjusted")]
        for mutate in mutations:
            value = batch()
            mutate(value.wire)
            with self.assertRaises(AdapterError):
                build(value)
        with self.assertRaisesRegex(AdapterError, "ordered and unique"):
            build_forward_labels(batch(), calendar=tuple(reversed(CALENDAR)), feature_sessions=(CALENDAR[0],))
        with self.assertRaisesRegex(AdapterError, "positive integer"):
            build(horizon=True)

    def test_180_uses_supplied_sessions_instead_of_weekday_approximation(self):
        start = datetime(2023, 1, 1)
        sessions = tuple((start + timedelta(days=i * 2)).date().isoformat() for i in range(182))
        value = batch()
        value.wire = native_batch("market_daily", ("open", "close"), sessions=sessions)
        # A fully missing but structurally pinned outcome remains invalid.
        value.wire["context"]["query"].update(price_basis="common_anchor_adjusted_v1",
                                              adjustment_anchor=sessions[-1])
        derived = deepcopy(batch().wire["context"]["derivation"])
        derived.update(anchor_session=sessions[-1], decision_session=sessions[-1])
        for name, fields, domain in (("price_query", ("open", "close"), "market_daily"),
                                     ("factor_query", ("factor",), "adjustment_factors")):
            derived[name] = native_batch(domain, fields, sessions=sessions)["context"]["query"]
        value.wire["context"]["derivation"] = derived
        result = build_forward_labels(value, calendar=sessions, feature_sessions=(sessions[0],),
                                      horizon_sessions=180)
        self.assertEqual(result["rows"][0]["start_session"], sessions[1])
        self.assertEqual(result["rows"][0]["end_session"], sessions[180])
        self.assertEqual(result["label_spec"]["horizon_sessions"], 180)

    def test_real_data_adjust_prices_output_is_consumable(self):
        import pandas as pd
        from axiom_data import adjust_prices
        from axiom_data.protocols import DataBatch

        def data(value):
            return DataBatch(pd.DataFrame.from_records(value["records"]),
                             value["field_meta"], value["context"])

        prices = native_batch("market_daily", ("open", "close"))
        factors = native_batch("adjustment_factors", ("factor",))
        for row in factors["records"]:
            if row["session"] >= CALENDAR[5]:
                row["factor"] = 2.0
        adjusted = adjust_prices(data(prices), data(factors), fields=("open", "close"),
                                 anchor_session=CALENDAR[-1], factor_field="factor")
        result = build_forward_labels(adjusted, calendar=CALENDAR, feature_sessions=(CALENDAR[0],))
        self.assertEqual(result["rows"][0]["return"], 2.0)
        self.assertTrue(result["rows"][0]["valid"])


if __name__ == "__main__":
    unittest.main()

"""Hand-computed synthetic checks for the A01 recipes executed by Core."""
from datetime import date, timedelta
import math
import statistics
import unittest

from axiom_engine.core import (ABI, SEMANTICS, ExecutionContext, FactBatch,
    FeaturePlan, execute_feature_plan, required_history)
from axiom_research.a01_catalog import (A01_FEATURE_COUNT,
    A01_ALPHA158_FEATURE_COUNT, load_a01_feature_catalog)
from axiom_research.feature_catalog import build_feature_plan, load_feature_catalog


REF = "sha256:" + "a" * 64


def fixture(*, changes=None, amount_only=False, late_close=False):
    sessions = [(date(2022, 1, 1) + timedelta(days=i)).isoformat() for i in range(252)]
    names = ["amount_cny"] if amount_only else ["open", "high", "low", "close", "amount_cny"]
    columns = [dict(name=f, dtype="float64", unit="CNY" if f == "amount_cny" else "CNY/share",
                    stage="fact", missing="preserve") for f in names]
    sources = [dict(id="synthetic", data_ref=REF, view_ref=REF,
        revision_policy="frozen-synthetic-v1", qualification="synthetic",
        availability_basis="synthetic-visible-common-anchor")]
    rows, keys, reference = [], [], []
    for security, initial, slope in (("A", 100, 1), ("B", 50, 2), ("EX", 70, 1.5)):
        for i, session in enumerate(sessions):
            close = initial + slope * i
            values = dict(open=close-.6, high=close+2, low=close-3,
                          close=close, amount_cny=1000.+10*i)
            if changes and (security, i) in changes:
                values.update(changes[security, i])
            keys.append([security, session])
            rows.append(dict(security_id=security, session=session,
                values=[values[f] for f in names],
                availability=[session + ("T23:00:00Z" if late_close and security == "A"
                    and i == 251 and f == "close" else "T12:00:00Z") for f in names],
                sources=[["synthetic"] for _ in names],
                missing_reasons=["SOURCE_MISSING" if values[f] is None else None for f in names]))
            reference.append(dict(security_id=security, session=session,
                member=security != "EX", industry=None,
                available_at=session+"T00:00:00Z", source="synthetic"))
    base = FeaturePlan.from_dict(dict(abi=ABI, semantics=SEMANTICS, recipe_ref=REF,
        calendar_ref=REF, reference_ref=REF,
        reference_members={s: {"A": None, "B": None} for s in sessions},
        input_schema=columns, event_schema={}, sources=sources,
        observation_domain="sessions", history_policy="partial", nodes=[], outputs=[], obligations=[]))
    facts = FactBatch.from_dict(dict(abi=ABI, calendar_ref=REF, schema=columns,
        sources=sources, rows=rows, event_schema={}, events=[]))
    context = ExecutionContext.from_dict(dict(abi=ABI, calendar_ref=REF, reference_ref=REF,
        sessions=sessions, cutoffs={s:s+"T20:00:00Z" for s in sessions}, history_keys=keys,
        output_keys=[key for key in keys if key[1] in (sessions[0], sessions[-1])], reference=reference))
    return base, facts, context, sessions


def execute(names=None, *, normalized=False, **kwargs):
    catalog = load_a01_feature_catalog()
    definitions = catalog.definitions
    selected = catalog.default_selection if names is None else [
        {"id": next(f["id"] for f in definitions if f["name"] == name), "semantic_version":"1.0.0"}
        for name in names]
    base, facts, context, sessions = fixture(**kwargs)
    plan = build_feature_plan(base, selected, catalog=catalog, normalized=normalized)
    output = execute_feature_plan(plan, facts, context).to_dict()
    out_names = [f["name"] for f in catalog.select(selected)]
    values = {(row["security_id"], row["session"]):dict(zip(out_names, row["values"]))
              for row in output["rows"]}
    return plan, output, values, sessions


class A01CatalogTests(unittest.TestCase):
    def test_all_recipes_compile_in_both_stages_with_declared_history(self):
        catalog = load_a01_feature_catalog()
        definitions = catalog.definitions
        self.assertEqual(len(definitions), A01_FEATURE_COUNT)
        self.assertEqual(sum(f["group"] == "ALP" for f in definitions), A01_ALPHA158_FEATURE_COUNT)
        self.assertEqual(len({f["formula"] for f in definitions}), 88)
        by_name = {f["name"]: f for f in definitions}
        self.assertEqual(by_name["KMID"]["id"], "ALP001")
        self.assertEqual(by_name["ROC5"]["id"], "ALP014")  # excluded VWAP0 keeps ALP013 reserved
        for excluded in ("VWAP0", "BETA5", "QTLU5", "IMAX5", "CORR5", "VMA5"):
            self.assertNotIn(excluded, by_name)
        self.assertEqual(len(load_feature_catalog().definitions), 6)
        base, _, _, _ = fixture()
        for normalized in (False, True):
            plan = build_feature_plan(base, catalog.default_selection, catalog=catalog, normalized=normalized)
            history = required_history(plan)
            self.assertEqual(history, {f["id"]: f["lookback"]-1 for f in definitions})
            self.assertEqual(max(history.values()), 251)
            self.assertEqual(len(plan.to_dict()["outputs"]), 88)
            if normalized:
                self.assertTrue(all(o["column"]["stage"] == "cross_sectional" for o in plan.to_dict()["outputs"]))

    def test_representative_original_and_extension_formulas_match_hand_calculation(self):
        _, output, values, sessions = execute()
        actual = values["A", sessions[-1]]
        expected = {
            "KMID": .6/350.4, "KLEN":5/350.4, "KMID2":.6/(5+1e-12),
            "KUP":2/350.4, "KLOW":2.4/350.4, "KSFT":1/350.4,
            "OPEN0":350.4/351, "ROC5":346/351, "MA5":349/351,
            "STD5":math.sqrt(2.5)/351, "MAX5":353/351, "MIN5":344/351,
            "RANK5":1., "RSV5":7/(9+1e-12), "CNTP5":1., "CNTN5":0.,
            "CNTD5":1., "SUMP5":5/(5+1e-12), "SUMN5":0., "SUMD5":5/(5+1e-12),
            "RET120":351/231-1, "UP120":1., "SMOOTH120":120/(120+1e-12),
            "PRICEPOS252":254/(256+1e-12), "LOG_AMOUNT":math.log1p(3510),
            "AMOUNT_REL20":3510/3415, "AMOUNT_REL60":3510/3215,
            "AMOUNT_Z20":95/(statistics.pstdev(range(3320,3520,10))+1e-12),
            "AMOUNT_CV20":statistics.stdev(range(3320,3520,10))/3415,
            "ILLIQ20":statistics.mean(1/(99+i)/(1000+10*i) for i in range(232,252)),
            "VOLADJRET120":(351/231-1)/(statistics.stdev(1/(99+i) for i in range(132,252))+1e-12),
        }
        for name, expected_value in expected.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(actual[name], expected_value, places=10)
        units = {c["name"]: c["unit"] for c in output["schema"]}
        illiq_id = next(f["id"] for f in load_a01_feature_catalog().definitions if f["name"] == "ILLIQ20")
        self.assertEqual(units[illiq_id], "1/CNY")
        self.assertIsNone(values["A", sessions[0]]["MA5"])
        self.assertIsNotNone(values["A", sessions[0]]["KMID"])

    def test_missing_zero_and_cutoff_policies_do_not_fill_or_peek(self):
        names = ["ROC5", "MA5", "SUMP5", "LOG_AMOUNT", "ILLIQ20"]
        _, _, values, sessions = execute(names, changes={("A",248):{"close":None}})
        self.assertIsNotNone(values["A",sessions[-1]]["ROC5"])
        self.assertIsNone(values["A",sessions[-1]]["MA5"])
        self.assertIsNone(values["A",sessions[-1]]["SUMP5"])
        _, _, values, sessions = execute(names, changes={("A",251):{"amount_cny":0.}})
        self.assertEqual(values["A",sessions[-1]]["LOG_AMOUNT"], 0.)
        self.assertIsNone(values["A",sessions[-1]]["ILLIQ20"])
        for kwargs in ({"changes":{("A",251):{"close":0.}}}, {"late_close":True}):
            _, _, values, sessions = execute(names, **kwargs)
            self.assertIsNone(values["A",sessions[-1]]["ROC5"])
            self.assertIsNotNone(values["A",sessions[-1]]["LOG_AMOUNT"])

    def test_normalization_uses_reference_members_and_preserves_constant_columns(self):
        _, _, values, sessions = execute(["ROC5", "CNTP5"], normalized=True)
        self.assertAlmostEqual(values["A",sessions[-1]]["ROC5"], 1.)
        self.assertAlmostEqual(values["B",sessions[-1]]["ROC5"], -1.)
        self.assertIsNone(values["A",sessions[-1]]["CNTP5"])
        self.assertEqual(values["EX",sessions[-1]], {"ROC5":None,"CNTP5":None})

    def test_amount_subset_prunes_price_dependencies_and_retains_requested_order(self):
        plan, output, values, sessions = execute(["AMOUNT_REL20", "LOG_AMOUNT"], amount_only=True)
        self.assertEqual([o["column"]["name"] for o in plan.to_dict()["outputs"]], ["LQD002", "LQD001"])
        self.assertAlmostEqual(values["A",sessions[-1]]["AMOUNT_REL20"], 3510/3415)
        self.assertTrue(all(c["unit"] == "dimensionless" for c in output["schema"]))


if __name__ == "__main__":
    unittest.main()

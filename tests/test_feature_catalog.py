"""Independent synthetic values and admission checks for catalog-driven Core plans."""
from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from axiom_engine.core import (ABI, SEMANTICS, ExecutionContext, FactBatch,
                               FeaturePlan, execute_feature_plan, required_history)
from axiom_research.feature_catalog import (FeatureCatalog, FeatureCatalogError,
    FeatureSelection, build_feature_plan, load_feature_catalog, render_feature_catalog)

REF = "sha256:" + "a" * 64
IDS = ("MOM010", "MOM020", "MOM030", "VOL010", "PRC010", "LIQ010")
ROOT = Path(__file__).resolve().parents[1]


def fixture(*, constant_price_ratio=False, missing_close=None, zero_close=None,
            missing_amount=None, late_close=False, only_amount=False):
    sessions = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(22)]
    fields = ["amount_cny"] if only_amount else ["open", "high", "low", "close", "amount_cny"]
    columns = [dict(name=f, dtype="float64", unit="CNY" if f == "amount_cny" else "CNY/share",
                    stage="fact", missing="preserve") for f in fields]
    sources = [dict(id="synthetic", data_ref=REF, view_ref=REF,
                    revision_policy="frozen-synthetic-v1", qualification="synthetic",
                    availability_basis="synthetic-visible-common-anchor")]
    rows, reference, keys = [], [], []
    for security, initial, slope, amount, amount_slope in (
            ("A", 100, 1, 1000, 100), ("B", 60, 2, 700, 30), ("EX", 40, 3, 500, 20)):
        for i, session in enumerate(sessions):
            close = float(initial + slope * i)
            values = dict(open=close / 2 if constant_price_ratio else close - slope,
                          high=close + 5, low=close - 2, close=close,
                          amount_cny=float(amount + amount_slope * i))
            if (security, i) == missing_close:
                values["close"] = None
            if (security, i) == zero_close:
                values["close"] = 0.0
            if (security, i) == missing_amount:
                values["amount_cny"] = None
            keys.append([security, session])
            rows.append(dict(security_id=security, session=session,
                values=[values[f] for f in fields],
                availability=[session + ("T23:00:00Z" if late_close and security == "A"
                    and i == 21 and f == "close" else "T12:00:00Z") for f in fields],
                sources=[["synthetic"] for _ in fields],
                missing_reasons=["SOURCE_MISSING" if values[f] is None else None for f in fields]))
            reference.append(dict(security_id=security, session=session, member=security != "EX",
                                  industry=None, available_at=session+"T00:00:00Z", source="synthetic"))
    base = FeaturePlan.from_dict(dict(abi=ABI, semantics=SEMANTICS, recipe_ref=REF,
        calendar_ref=REF, reference_ref=REF,
        reference_members={s: {"A": None, "B": None} for s in sessions},
        input_schema=columns, event_schema={}, sources=sources,
        observation_domain="sessions", history_policy="partial", nodes=[], outputs=[], obligations=[]))
    facts = FactBatch.from_dict(dict(abi=ABI, calendar_ref=REF, schema=columns,
                                    sources=sources, rows=rows, event_schema={}, events=[]))
    context = ExecutionContext.from_dict(dict(abi=ABI, calendar_ref=REF, reference_ref=REF,
        sessions=sessions, cutoffs={s: s+"T20:00:00Z" for s in sessions},
        history_keys=keys, output_keys=keys, reference=reference))
    return base, facts, context, sessions


def execute(*, normalized=False, selection=None, **kwargs):
    catalog = load_feature_catalog()
    base, facts, context, sessions = fixture(**kwargs)
    plan = build_feature_plan(base, catalog.default_selection if selection is None else selection,
                              catalog=catalog, normalized=normalized)
    result = execute_feature_plan(plan, facts, context).to_dict()
    values = {(r["security_id"], r["session"]): r["values"] for r in result["rows"]}
    return plan, result, values, sessions


class FeatureCatalogTests(unittest.TestCase):
    def test_model_selection_requires_exact_ids_and_versions_in_model_order(self):
        catalog = load_feature_catalog()
        self.assertEqual(tuple(f["id"] for f in catalog.definitions), IDS)
        chosen = [{"id": "MOM030", "semantic_version": "1.0.0"},
                  {"id": "MOM010", "semantic_version": "1.0.0"}]
        self.assertEqual([f["id"] for f in catalog.select(chosen)], ["MOM030", "MOM010"])
        for invalid in ([{"id": "MOM010"}], [{"id": "MOM010", "semantic_version": "latest"}],
                        [{"id": "MOM011", "semantic_version": "1.0.0"}],
                        [chosen[0], chosen[0]], [], ["MOM010"]):
            with self.assertRaises(FeatureCatalogError):
                catalog.select(invalid)

    def test_selection_and_catalog_semantics_change_recipe_identity(self):
        catalog = load_feature_catalog()
        selection = catalog.default_selection
        identity = catalog.recipe_ref(selection)
        self.assertNotEqual(identity, catalog.recipe_ref(selection[::-1]))
        self.assertNotEqual(identity, catalog.recipe_ref(selection, normalized=True))
        changed = catalog.to_dict()
        changed["features"][0]["formula"] += "; revised recipe"
        self.assertNotEqual(identity, FeatureCatalog.from_dict(changed).recipe_ref(selection))
        duplicate = catalog.to_dict()
        duplicate["features"].append(deepcopy(duplicate["features"][0]))
        with self.assertRaises(FeatureCatalogError):
            FeatureCatalog.from_dict(duplicate)
        with self.assertRaises(FeatureCatalogError):
            FeatureCatalog('{"schema_version":"a","schema_version":"b"}')

    def test_raw_values_match_independent_finance_formulas(self):
        plan, frame, values, sessions = execute()
        self.assertEqual([c["name"] for c in frame["schema"]], list(IDS))
        self.assertTrue(all(c["stage"] == "base" for c in frame["schema"]))
        expected = [121/120-1, 121/116-1, 121/101-1, 7/121, 121/120-1,
                    3100 / ((2700+2800+2900+3000+3100)/5)]
        for actual, want in zip(values["A", sessions[-1]], expected):
            self.assertAlmostEqual(actual, want, places=12)
        self.assertIsNone(values["A", sessions[0]][0])
        self.assertIsNone(values["A", sessions[19]][2])
        self.assertIsNone(values["A", sessions[3]][5])
        self.assertEqual(required_history(plan),
                         dict(zip(IDS, (1, 5, 20, 0, 0, 4))))

    def test_missing_and_nonpositive_interior_momentum_window_stays_missing(self):
        for kwargs in ({"missing_close": ("A", 10)}, {"zero_close": ("A", 10)}):
            _, _, values, sessions = execute(**kwargs)
            self.assertIsNone(values["A", sessions[-1]][2])
            self.assertIsNotNone(values["A", sessions[-1]][0])
        _, _, values, sessions = execute(missing_amount=("A", 19))
        self.assertIsNone(values["A", sessions[-1]][5])

    def test_cs_zscore_reference_set_and_constant_columns_preserve_missing(self):
        plan, frame, values, sessions = execute(normalized=True, constant_price_ratio=True)
        self.assertTrue(all(c["stage"] == "cross_sectional" for c in frame["schema"]))
        self.assertEqual([c["name"] for c in frame["schema"]], list(IDS))
        for symbol in ("A", "B", "EX"):
            self.assertIsNone(values[symbol, sessions[-1]][4])  # constant PRC010 is missing, never zero
        self.assertEqual(values["EX", sessions[-1]], [None] * 6)
        self.assertAlmostEqual(values["A", sessions[-1]][0], -1.0)
        self.assertAlmostEqual(values["B", sessions[-1]][0], 1.0)
        for node in plan.to_dict()["nodes"]:
            if node["op"] == "cs_zscore":
                self.assertEqual(node["params"]["constant"], "missing")
                self.assertEqual(node["params"]["excluded"], "missing")
                self.assertEqual(node["params"]["ddof"], 0)

    def test_subset_requires_only_its_real_dependencies_and_preserves_order(self):
        chosen = [FeatureSelection("LIQ010", "1.0.0")]
        _, frame, values, sessions = execute(selection=chosen, only_amount=True)
        self.assertEqual([c["name"] for c in frame["schema"]], ["LIQ010"])
        self.assertAlmostEqual(values["A", sessions[-1]][0], 3100/2900)
        _, frame, _, _ = execute(selection=[
            FeatureSelection("PRC010", "1.0.0"), FeatureSelection("MOM020", "1.0.0")])
        self.assertEqual([c["name"] for c in frame["schema"]], ["PRC010", "MOM020"])

    def test_unavailable_cells_are_never_read_through_cutoff(self):
        _, _, values, sessions = execute(late_close=True)
        self.assertEqual(values["A", sessions[-1]][:5], [None] * 5)
        self.assertIsNotNone(values["A", sessions[-1]][5])

    def test_catalog_lookback_and_core_policy_are_checked_at_compilation(self):
        catalog = load_feature_catalog()
        base, _, _, _ = fixture()
        changed = catalog.to_dict()
        changed["features"][0]["lookback"] = 99
        with self.assertRaises(FeatureCatalogError):
            build_feature_plan(base, catalog.default_selection,
                               catalog=FeatureCatalog.from_dict(changed))
        changed = catalog.to_dict()
        changed["features"][0]["normalization"]["ddof"] = 2
        with self.assertRaises(ValueError):
            build_feature_plan(base, catalog.default_selection, normalized=True,
                               catalog=FeatureCatalog.from_dict(changed))

    def test_markdown_is_derived_and_cli_writes_only_explicit_destination(self):
        catalog = load_feature_catalog()
        text = render_feature_catalog(catalog)
        self.assertIn(catalog.identity, text)
        for definition in catalog.definitions:
            self.assertIn(definition["id"], text)
            self.assertIn(definition["formula"], text)
        tool = ROOT / "tools" / "render_feature_catalog.py"
        stdout = subprocess.run([sys.executable, str(tool)], check=True,
                                capture_output=True, text=True).stdout
        self.assertEqual(stdout, text)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "catalog.md"
            result = subprocess.run([sys.executable, str(tool), "--output", str(destination)],
                                    check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout, "")
            self.assertEqual(destination.read_text(), text)


if __name__ == "__main__":
    unittest.main()


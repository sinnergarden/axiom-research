"""Versioned stock feature definitions compiled into the existing Core plan ABI.

The JSON catalog is the definition source. This module validates and selects
recipes; Core remains the sole numerical executor. Data must supply OHLC using
one visible common adjustment anchor and unadjusted amount_cny in CNY.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from axiom_engine.core import FeaturePlan, required_history, validate_plan


class FeatureCatalogError(ValueError):
    """An invalid catalog or unresolved model feature selection."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FeatureCatalogError(message)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FeatureCatalogError("catalog must contain finite JSON values") from exc


def _exact(value: Any, names: str, label: str) -> None:
    _require(type(value) is dict and set(value) == set(names.split()),
             f"{label}: expected exact fields {names}")


def _unique_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


@dataclass(frozen=True)
class FeatureSelection:
    id: str
    semantic_version: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "semantic_version": self.semantic_version}


@dataclass(frozen=True)
class FeatureCatalog:
    """Immutable JSON catalog; callers receive fresh copies of definitions."""

    payload: str

    def __post_init__(self) -> None:
        try:
            value = json.loads(self.payload, object_pairs_hook=_unique_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise FeatureCatalogError("invalid catalog JSON") from exc
        _validate_catalog(value)
        object.__setattr__(self, "payload", _canonical(value))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureCatalog":
        return cls(_canonical(value))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.payload)

    @property
    def identity(self) -> str:
        return "sha256:" + hashlib.sha256(self.payload.encode("utf-8")).hexdigest()

    @property
    def definitions(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.to_dict()["features"])

    @property
    def default_selection(self) -> tuple[FeatureSelection, ...]:
        return tuple(FeatureSelection(f["id"], f["semantic_version"])
                     for f in self.definitions)

    def select(self, selections: Sequence[FeatureSelection | Mapping[str, str]]
               ) -> tuple[dict[str, Any], ...]:
        _require(isinstance(selections, (list, tuple)) and bool(selections),
                 "explicit nonempty id/version feature selection required")
        available = {(f["id"], f["semantic_version"]): f for f in self.definitions}
        result, seen = [], set()
        for item in selections:
            entry = item.to_dict() if isinstance(item, FeatureSelection) else item
            _exact(entry, "id semantic_version", "feature selection")
            pair = (entry["id"], entry["semantic_version"])
            _require(all(type(v) is str for v in pair), "feature id/version must be strings")
            _require(pair in available, f"unknown feature id/version: {pair}")
            _require(pair[0] not in seen, "duplicate feature id in model selection: " + pair[0])
            seen.add(pair[0])
            result.append(available[pair])
        return tuple(result)

    def recipe_ref(self, selections: Sequence[FeatureSelection | Mapping[str, str]],
                   *, normalized: bool = False) -> str:
        chosen = self.select(selections)
        _require(type(normalized) is bool, "normalized must be bool")
        recipe = {"catalog": self.identity, "selection": [
            {"id": f["id"], "semantic_version": f["semantic_version"]} for f in chosen],
            "normalized": normalized}
        return "sha256:" + hashlib.sha256(_canonical(recipe).encode("utf-8")).hexdigest()


def _validate_catalog(value: Any) -> None:
    _exact(value, "schema_version catalog_id groups input_contract shared_nodes features", "catalog")
    _require(value["schema_version"] == "stock_feature_catalog_v1", "unsupported catalog schema")
    _require(type(value["catalog_id"]) is str and bool(value["catalog_id"]), "catalog id required")
    _require(type(value["groups"]) is list and bool(value["groups"]), "feature groups required")
    groups = set()
    for group in value["groups"]:
        _exact(group, "id name numbering_policy", "group")
        _require(type(group["id"]) is str and re.fullmatch(r"[A-Z]{3}", group["id"]) is not None,
                 "group id must be three uppercase letters")
        _require(group["id"] not in groups, "duplicate group id")
        _require(type(group["name"]) is str and bool(group["name"]), "group name required")
        _require(group["numbering_policy"] == "sparse_reserved_never_renumber",
                 "preserve reserved sparse feature numbers")
        groups.add(group["id"])
    _exact(value["input_contract"], "price_basis adjustment_anchor amount_unit", "input contract")
    _require(value["input_contract"] == {
        "price_basis": "common_anchor_adjusted_v1",
        "adjustment_anchor": "feature_session_visible_at_cutoff", "amount_unit": "CNY"},
        "stock input basis/anchor/units must be explicit")
    _require(type(value["shared_nodes"]) is list, "shared Core nodes must be an array")
    _require(type(value["features"]) is list and bool(value["features"]), "features required")
    pairs, names, node_names = set(), set(), set()
    for feature in value["features"]:
        _exact(feature, "group id name formula dependencies lookback normalization missing_policy semantic_version core_nodes",
               "feature")
        _require(feature["group"] in groups and type(feature["id"]) is str and
                 re.fullmatch(feature["group"] + r"[0-9]{3}", feature["id"]) is not None,
                 "feature id must retain its group and reserved number")
        version = feature["semantic_version"]
        _require(type(version) is str and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is not None,
                 "explicit semantic version required")
        pair = (feature["id"], version)
        _require(pair not in pairs, "duplicate feature id/version")
        _require(type(feature["name"]) is str and bool(feature["name"]) and
                 (feature["name"], version) not in names, "unique feature name/version required")
        pairs.add(pair); names.add((feature["name"], version))
        _require(type(feature["formula"]) is str and bool(feature["formula"]), "formula required")
        dependencies = feature["dependencies"]
        _require(type(dependencies) is list and bool(dependencies) and
                 all(type(d) is str and re.fullmatch(r"[a-z_]+\.[a-z_]+", d) for d in dependencies) and
                 len(dependencies) == len(set(dependencies)), "unique qualified Data dependencies required")
        _require(type(feature["lookback"]) is int and feature["lookback"] >= 1,
                 "lookback is inclusive history sessions")
        _exact(feature["normalization"],
               "op reference group unknown_group missing ddof epsilon constant clip excluded",
               "normalization")
        _require(feature["normalization"]["op"] == "cs_zscore" and
                 feature["normalization"]["reference"] == "reference_members",
                 "normalization must use Core with frozen reference_members")
        _require(type(feature["missing_policy"]) is str and bool(feature["missing_policy"]),
                 "explicit missing policy required")
        _require(type(feature["core_nodes"]) is list and bool(feature["core_nodes"]),
                 "Core recipe required")
        _require(feature["core_nodes"][-1]["name"] == feature["id"],
                 "last raw node must be the stable feature id")
    for node in value["shared_nodes"] + [n for f in value["features"] for n in f["core_nodes"]]:
        _exact(node, "name op version inputs params column", "Core node")
        _require(type(node["name"]) is str and node["name"] not in node_names,
                 "duplicate Core recipe node")
        node_names.add(node["name"])


def load_feature_catalog(path: str | Path | None = None) -> FeatureCatalog:
    """Read the installed catalog or one explicitly supplied catalog path."""
    source = files("axiom_research").joinpath("catalogs", "stock_ml_v1.json") if path is None else Path(path)
    return FeatureCatalog(source.read_text(encoding="utf-8"))


def build_feature_plan(base_plan: FeaturePlan,
                       selections: Sequence[FeatureSelection | Mapping[str, str]], *,
                       catalog: FeatureCatalog | None = None,
                       normalized: bool = False) -> FeaturePlan:
    """Compile selected catalog recipes; preserve adapter source/context bindings.

    Output columns keep the selected IDs in model order. Raw and normalized
    modes have different recipe identities and stages; neither fills missing
    values. The upstream Data adjustment is a caller-owned provenance boundary.
    """
    _require(type(base_plan) is FeaturePlan, "base_plan must be a Core FeaturePlan")
    catalog = catalog if catalog is not None else load_feature_catalog()
    selected = catalog.select(selections)
    _require(type(normalized) is bool, "normalized must be bool")
    p = base_plan.to_dict()
    all_nodes = catalog.to_dict()["shared_nodes"] + [n for f in selected for n in f["core_nodes"]]
    by_name = {n["name"]: n for n in all_nodes}
    needed = set()

    def include(name: str) -> None:
        if name in by_name and name not in needed:
            needed.add(name)
            for parent in by_name[name]["inputs"]:
                include(parent)

    for feature in selected:
        include(feature["id"])
    p["nodes"] = [n for n in all_nodes if n["name"] in needed]
    p["outputs"] = []
    for feature in selected:
        name = feature["id"]
        node = by_name[name]
        if normalized:
            norm = feature["normalization"]
            name = name + "_cs_zscore"
            node = dict(name=name, op=norm["op"], version="1", inputs=[feature["id"]],
                        params={k: v for k, v in norm.items() if k not in ("op", "reference")},
                        column=dict(name=name, dtype="float64", unit="dimensionless",
                                    stage="cross_sectional", missing="preserve"))
            p["nodes"].append(node)
        column = {**node["column"], "name": feature["id"]}
        p["outputs"].append({"node": name, "column": column})
    p["recipe_ref"] = catalog.recipe_ref(selections, normalized=normalized)
    plan = FeaturePlan.from_dict(p)
    validate_plan(plan, execution=True)
    history = required_history(plan)
    _require(all(history[f["id"]] + 1 == f["lookback"] for f in selected),
             "catalog lookback differs from Core recipe dependency closure")
    return plan


def render_feature_catalog(catalog: FeatureCatalog | None = None) -> str:
    """Derive human documentation from the sole machine catalog."""
    catalog = catalog if catalog is not None else load_feature_catalog()
    value = catalog.to_dict()

    def cell(item: Any) -> str:
        text = _canonical(item) if isinstance(item, (list, dict)) else str(item)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [f"# {value['catalog_id']}", "", f"Catalog identity: {catalog.identity}", "",
             "Lookback counts the current feature session. Inputs use a visible common anchor; amount remains CNY.",
             "", "| Group | ID | Name | Version | Formula | Dependencies | Lookback | Normalization | Missing policy |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for feature in catalog.definitions:
        lines.append("| " + " | ".join(cell(feature[k]) for k in (
            "group", "id", "name", "semantic_version", "formula", "dependencies",
            "lookback", "normalization", "missing_policy")) + " |")
    return "\n".join(lines) + "\n"


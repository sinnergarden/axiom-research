"""Installed QSYS recipes and the complete source-to-recipe migration register.

The register is an inventory, not an executor. Blocked source rows cannot be
selected and never enter FeatureCatalog. ``normalized=True`` in the existing
compiler means Axiom reference-pool CS zscore, not QSYS absnorm expressions.
"""
from __future__ import annotations

from importlib.resources import files
import json
from typing import Sequence

from .feature_catalog import FeatureCatalog, FeatureCatalogError, FeatureSelection
from .qsys_input_contracts import DAILY_FIELD_CONTRACTS, FINANCIAL_FIELD_CONTRACTS
from copy import deepcopy


QSYS_SOURCE_REVISION = "852bcb73125f8781b7a80618b8188928ef4d38f2"


def load_qsys_feature_catalog() -> FeatureCatalog:
    """Load executable recipes; this does not assert historical input coverage."""
    return FeatureCatalog(files("axiom_research").joinpath(
        "catalogs", "qsys_stock_v2.json").read_text(encoding="utf-8"))


def load_qsys_feature_registry() -> dict:
    """Return a fresh, validated mapping of all 388 forensic source rows."""
    value = json.loads(files("axiom_research").joinpath(
        "catalogs", "qsys_registry_v2.json").read_text(encoding="utf-8"))
    if (value["schema_version"] != "qsys_registry_v1" or
            value["source_revision"] != QSYS_SOURCE_REVISION or
            {r["inventory_row"] for r in value["rows"]} != set(range(1, 389)) or
            len(value["rows"]) != 388):
        raise FeatureCatalogError("invalid QSYS migration register identity/coverage")
    catalog = load_qsys_feature_catalog()
    available = {(f["id"], f["semantic_version"]) for f in catalog.definitions}
    for row in value["rows"]:
        if row["status"] in ("executable", "alias"):
            selection = row["selection"]
            if set(selection) != {"id", "semantic_version"} or (selection["id"], selection["semantic_version"]) not in available:
                raise FeatureCatalogError("registry selection does not resolve to installed recipe")
            if "blocker" in row:
                raise FeatureCatalogError("executable row cannot carry a blocker")
        elif row["status"] == "blocked":
            if "selection" in row or not all(row["blocker"].get(k) for k in ("code", "reason")):
                raise FeatureCatalogError("blocked row needs reason and cannot select a recipe")
        else:
            raise FeatureCatalogError("unknown QSYS migration status")
    return value


def select_qsys_features(names: Sequence[str]) -> tuple[FeatureSelection, ...]:
    """Resolve exact public names or stable IDs, retaining requested order.

    Certified aliases resolve to one existing identity. Selecting an alias and
    its target twice is rejected. Blocked names fail with their specific reason.
    No fuzzy matching, pe/pe_ttm substitution, or null recipes are permitted.
    """
    if not isinstance(names, (tuple, list)) or not names or any(type(n) is not str for n in names):
        raise FeatureCatalogError("explicit nonempty QSYS names required")
    catalog = load_qsys_feature_catalog()
    lookup = {f[k]: FeatureSelection(f["id"], f["semantic_version"])
              for f in catalog.definitions for k in ("id", "name")}
    blocked = {}
    for row in load_qsys_feature_registry()["rows"]:
        if row["status"] == "blocked":
            blocked[row["name"]] = row["blocker"]
        else:
            lookup[row["name"]] = FeatureSelection(**row["selection"])
    result = []
    for name in names:
        if name in blocked:
            b = blocked[name]
            raise FeatureCatalogError(f"QSYS feature {name!r} blocked [{b['code']}]: {b['reason']}")
        if name not in lookup:
            raise FeatureCatalogError("unknown QSYS feature name/id: " + name)
        result.append(lookup[name])
    catalog.select(result)
    return tuple(result)


def qsys_input_requirements(names: Sequence[str] | None = None) -> dict[str, dict]:
    """Return all extra input requirements, explicitly partitioned by bridge.

    ``daily_fields`` goes to ``extend_feature_column_source``. Each bundle in
    ``financial_fields`` describes one EventQuery and the ``outputs`` mapping
    for ``extend_financial_feature_source``. The caller must bind report-range,
    symbols and frozen source scope; this function performs no preparation.
    """
    catalog = load_qsys_feature_catalog()
    selected = catalog.select(catalog.default_selection if names is None else select_qsys_features(names))
    dependencies = {d for feature in selected for d in feature['dependencies']}
    nodes = {n['name']: n for n in catalog.to_dict()['shared_nodes']}
    nodes.update({n['name']: n for feature in selected for n in feature['core_nodes']})
    leaves, visited = set(), set()
    def include(name):
        if name in visited:
            return
        visited.add(name)
        if name not in nodes:
            leaves.add(name)
        else:
            for parent in nodes[name]['inputs']:
                include(parent)
    for feature in selected:
        include(feature['id'])
    financial = {}
    bound_event_dependencies = set()
    for key, contract in FINANCIAL_FIELD_CONTRACTS.items():
        outputs = {leaf: dict(spec) for leaf, spec in contract['outputs'].items() if leaf in leaves}
        if outputs:
            financial[key] = {**deepcopy(contract), 'outputs': outputs}
            bound_event_dependencies.add(contract['domain'] + '.' + contract['field'])
    if 'market_daily.vwap' in dependencies:
        dependencies.add('market_daily.volume_shares')
    daily = {}
    for dependency in sorted(dependencies):
        domain, field = dependency.split('.')
        if dependency in bound_event_dependencies:
            continue
        if domain == 'adjustment_factors' or domain == 'market_daily' and field != 'volume_shares':
            continue
        declaration = DAILY_FIELD_CONTRACTS.get(field)
        if declaration is None or declaration['domain'] != domain:
            raise FeatureCatalogError('dependency requires a separately bound input bridge: ' + dependency)
        daily[field] = dict(declaration)
    return {'daily_fields': daily, 'financial_fields': financial}


def qsys_daily_fields(names: Sequence[str] | None = None) -> dict[str, dict[str, str]]:
    """Return additional daily columns only for an entirely daily selection.

    Existing OHLC/amount buffers are reused; the source derives VWAP from volume,
    amount and the common anchor. With a financial selection (including the new
    default all), use ``qsys_input_requirements`` instead: dropping financial
    inputs silently would make the preparation request incomplete.
    """
    requirements = qsys_input_requirements(names)
    if requirements['financial_fields']:
        raise FeatureCatalogError('selection requires financial materialization; use qsys_input_requirements() for complete daily_fields and financial_fields')
    return requirements['daily_fields']


def qsys_financial_fields(names: Sequence[str] | None = None) -> dict[str, dict]:
    """Return explicit event-to-typed-leaf bridge bundles; no fake daily domain."""
    return qsys_input_requirements(names)['financial_fields']

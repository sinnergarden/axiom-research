"""Read the A01 Alpha158 OHLC subset and price/amount extensions.

The JSON catalog owns every definition. This module neither imports Qlib nor
executes numerical formulas; the existing FeatureCatalog compiler and Engine
Core execute the selected recipes. Alpha IDs preserve Qlib 0.9.7's original
158-column positions, including gaps for unsupported columns.
"""
from importlib.resources import files

from .feature_catalog import FeatureCatalog


A01_CATALOG_ID = "a01_price_v1"
A01_ALPHA158_FEATURE_COUNT = 77
A01_EXTENSION_FEATURE_COUNT = 11
A01_FEATURE_COUNT = A01_ALPHA158_FEATURE_COUNT + A01_EXTENSION_FEATURE_COUNT
A01_EXCLUDED_ALPHA158_FAMILIES = (
    "VWAP", "BETA", "RSQR", "RESI", "QTLU", "QTLD", "IMAX", "IMIN",
    "IMXD", "CORR", "CORD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN",
    "VSUMD",
)


def load_a01_feature_catalog() -> FeatureCatalog:
    """Load immutable definitions without changing the legacy default catalog.

    Prices use the visible common adjustment anchor. Amount remains unadjusted
    CNY and is never presented as Qlib volume. Rolling recipes require complete
    windows and propagate missing inputs; Qlib's partial-window results are not
    claimed as compatible. Final CS normalization remains caller-selected via
    the existing ``build_feature_plan(..., normalized=True)`` interface.
    """
    source = files("axiom_research").joinpath("catalogs", "a01_price_v1.json")
    return FeatureCatalog(source.read_text(encoding="utf-8"))

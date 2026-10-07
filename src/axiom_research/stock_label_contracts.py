"""Shared outcome eligibility definitions; no Data/Core/ML runtime imports."""
from datetime import date, datetime, timezone, timedelta
from copy import deepcopy
from dataclasses import dataclass
import math
TARGET_SEMANTICS = 'forward_5_session_cs_zscore_prediction'
from typing import Any


def _session(value):
    if type(value) is not str or date.fromisoformat(value).isoformat() != value:
        raise ValueError('session must be YYYY-MM-DD')
    return value


def _instant(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('aware timestamp required')
    return value.astimezone(timezone.utc)


NORMALIZATION_SPEC = {
    "normalization_id": "forward_label_cs_zscore_v1",
    "semantic_version": "1",
    "purpose": "label_outcomes",
    "context_mode": "offline_label_cutoff",
    "executor": "axiom_engine.core.execute_feature_plan",
    "operator": "cs_zscore",
    "operator_version": "1",
    "params": {"group": "session", "unknown_group": "reject", "missing": "skip",
               "ddof": 0, "epsilon": 1e-12, "constant": "missing", "clip": None,
               "excluded": "missing"},
    "eligibility": "original_member_all_features_finite_valid_and_raw_label_mature",
    "maturity": "label_available_at_lte_cutoff_and_end_session_lte_cutoff_UTC_date",
    "grid_policy": "preserve_all_raw_label_keys",
    "core_partition": "one_feature_session_per_execution",
    "core_clock_projection": "exact_eligibility_first_then_ceil_cutoff_and_availability_to_seconds",
    "label_available_at": "original_raw_label_clock",
    "normalized_available_at": "Core_output_dependency_clock_at_offline_cutoff",
}

# The stock Target profile is narrower than the neutral Core carrier. Keeping
# these saved schemas in the readonly contract module avoids a build-runtime
# import when admitting historical matrix files.
RAW_TARGET_SCHEMA = [{'name':'raw_return','dtype':'float64','unit':'dimensionless',
                      'stage':'fact','missing':'preserve'}]
NORMALIZED_TARGET_SCHEMA = [{'name':'normalized_target','dtype':'float64','unit':'dimensionless',
                             'stage':'cross_sectional','missing':'preserve'}]


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class _FeatureEligibility:
    """Facts projected only from an admitted typed Feature row."""
    member: bool
    complete_finite: bool
    validity_all: bool
    knowledge_cutoff: str
    maximum_available_at_utc_us: int | None


def _eligible_reason(raw: dict, feature: dict | _FeatureEligibility | None, width: int, cutoff: datetime) -> str | None:
    if raw.get("valid") is not True:
        return raw.get("invalid_reason") or "RAW_LABEL_INVALID"
    if not _finite(raw.get("return")):
        return "RAW_LABEL_INVALID"
    try:
        end = _session(raw.get("end_session"))
        available = _instant(raw.get("label_available_at"))
    except (ValueError, TypeError):
        return "LABEL_CLOCK_OR_ENDPOINT_UNKNOWN"
    if end > cutoff.date().isoformat() or available > cutoff:
        return "LABEL_NOT_MATURE"
    if feature is None:
        return "FEATURE_MISSING"
    typed = type(feature) is _FeatureEligibility
    member = feature.member if typed else feature.get('member')
    if member is not True:
        return "NOT_MEMBER" if member is False else "MEMBERSHIP_UNKNOWN"
    if typed:
        complete, valid = feature.complete_finite, feature.validity_all
        knowledge = feature.knowledge_cutoff
    else:
        values, validity = feature.get("values"), feature.get("validity")
        complete = (isinstance(values,list) and isinstance(validity,list) and
                    len(values)==width and len(validity)==width and all(_finite(v) for v in values))
        valid = isinstance(validity,list) and all(v is True for v in validity)
        knowledge = feature.get('knowledge_cutoff')
    if not complete: return "FEATURE_MISSING"
    if not valid:
        return "FEATURE_INVALID"
    try:
        if _instant(knowledge) > cutoff:
            return "FEATURE_NOT_AVAILABLE"
    except (ValueError, TypeError):
        return "FEATURE_CLOCK_UNKNOWN"
    return None


def core_clock(value):
    """The existing label stage's ceil-to-Core-second clock projection."""
    instant = _instant(value)
    if instant.microsecond:
        instant += timedelta(seconds=1)
    return instant.replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _normalization_sources(raw_ref, feature_ref, session, cutoff, eligible):
    """The existing section binding, shared by dict and typed projections."""
    from .stock_artifacts import digest
    section_definition = {"contract_version":"stock_label_section_v1", "feature_session":session,
        "eligible_keys":eligible, "raw_label_ref":raw_ref, "feature_ref":feature_ref,
        "cutoff":_instant(cutoff).isoformat().replace('+00:00','Z'), "normalization_spec":deepcopy(NORMALIZATION_SPEC)}
    section_ref = digest(section_definition)
    return [
        {"id":"raw_labels", "data_ref":raw_ref, "view_ref":raw_ref,
         "revision_policy":"frozen_saved_label_build", "qualification":"observed",
         "availability_basis":"exact_raw_label_available_at"},
        {"id":"offline_eligibility", "data_ref":feature_ref, "view_ref":section_ref,
         "revision_policy":"frozen_feature_membership_and_explicit_outcome_cutoff",
         "qualification":"observed", "availability_basis":"derived_offline_cutoff_selection"}]


def normalization_section_inputs(raw_build, *, feature_ref, feature_rows, session,
                                 securities, width, cutoff, abi='axiom.feature/1',
                                 semantics='axiom.operators/1', raw_index=None):
    """Project the existing section inputs, without executing/normalizing values."""
    from .stock_artifacts import digest
    raw_ref = raw_build['label_ref']; indexed = feature_rows
    if raw_index is None:
        raw_index = {(r['security_id'], r['feature_session']): r for r in raw_build['rows']}
    instant = _instant(cutoff)
    core_cutoff = core_clock(cutoff); spec = deepcopy(NORMALIZATION_SPEC)
    recipe_ref = digest({'normalization_spec': spec, 'raw_label_spec': raw_build['label_spec']})
    input_column = {'name': 'raw_return', 'dtype': 'float64', 'unit': 'dimensionless', 'stage': 'fact', 'missing': 'preserve'}
    output_column = {'name': 'normalized_target', 'dtype': 'float64', 'unit': 'dimensionless', 'stage': 'cross_sectional', 'missing': 'preserve'}
    keys = [(security, session) for security in securities]; columns = range(width)
    ABI, SEMANTICS = abi, semantics
    reasons = {key: _eligible_reason(raw_index[key], indexed.get(key), len(columns), instant)
               for key in keys}
    eligible = [list(key) for key in keys if reasons[key] is None]
    sources = _normalization_sources(raw_ref,feature_ref,session,cutoff,eligible)
    section_ref = sources[1]['view_ref']
    plan = ({"abi": ABI, "semantics": SEMANTICS, "recipe_ref": recipe_ref,
        "calendar_ref": raw_build["calendar_ref"], "reference_ref": section_ref,
        "reference_members": {session: {key[0]: None for key in keys if reasons[key] is None}},
        "input_schema": [input_column], "event_schema": {}, "sources": sources,
        "observation_domain": "sessions", "history_policy": "partial",
        "nodes": [{"name": "normalized_target", "op": "cs_zscore", "version": "1",
                   "inputs": ["raw_return"], "params": deepcopy(spec["params"]),
                   "column": output_column}],
        "outputs": [{"node": "normalized_target", "column": output_column}], "obligations": []})
    facts = ({"abi": ABI, "calendar_ref": raw_build["calendar_ref"],
        "schema": [input_column], "sources": sources, "event_schema": {}, "events": [],
        "rows": [{"security_id": key[0], "session": session,
                  "values": [float(raw_index[key]["return"]) if reasons[key] is None else None],
                  "availability": [core_clock(raw_index[key]["label_available_at"])
                                   if reasons[key] is None else core_cutoff],
                  "sources": [["raw_labels"] if reasons[key] is None else ["offline_eligibility"]],
                  "missing_reasons": [reasons[key]]} for key in keys]})
    context = ({"abi": ABI, "calendar_ref": raw_build["calendar_ref"],
        "reference_ref": section_ref, "sessions": [session], "cutoffs": {session: core_cutoff},
        "history_keys": [list(key) for key in keys], "output_keys": [list(key) for key in keys],
        "reference": [{"security_id": key[0], "session": session, "member": reasons[key] is None,
                       "industry": None, "available_at": core_cutoff, "source": "offline_eligibility"}
                      for key in keys]})
    return {'plan': plan, 'facts': facts, 'context': context, 'reasons': reasons,
            'eligible_keys': eligible, 'section_ref': section_ref}

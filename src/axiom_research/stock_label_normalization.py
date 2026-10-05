"""Offline outcome normalization through the existing Feature Core.

This consumes saved label/feature builds. It neither reads Data nor admits
outcomes to the decision-fact adapter. Each actual feature session is a separate
Core cross-section; there is no fitted scaler or second numerical executor.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import math
from typing import Any

from axiom_engine.core import (ABI, SEMANTICS, ExecutionContext, FactBatch,
                               FeaturePlan, execute_feature_plan)

from .data_adapter import AdapterError, _require, _utc
from .labels import _instant, _session
from .stock_artifacts import digest


CONTRACT_VERSION = "stock_normalized_label_build_v1"
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


def _sealed(build: Any, version: str, ref_name: str) -> str:
    _require(isinstance(build, dict) and build.get("contract_version") == version,
             "unsupported " + version)
    ref = build.get(ref_name)
    _require(ref == digest({key: value for key, value in build.items() if key != ref_name}),
             ref_name + " integrity mismatch")
    return ref


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _eligible_reason(raw: dict, feature: dict | None, width: int, cutoff: datetime) -> str | None:
    if raw.get("valid") is not True:
        return raw.get("invalid_reason") or "RAW_LABEL_INVALID"
    if not _finite(raw.get("return")):
        return "RAW_LABEL_INVALID"
    try:
        end = _session(raw.get("end_session"))
        available = _instant(raw.get("label_available_at"))
    except AdapterError:
        return "LABEL_CLOCK_OR_ENDPOINT_UNKNOWN"
    if end > cutoff.date().isoformat() or available > cutoff:
        return "LABEL_NOT_MATURE"
    if feature is None:
        return "FEATURE_MISSING"
    if feature.get("member") is not True:
        return "NOT_MEMBER" if feature.get("member") is False else "MEMBERSHIP_UNKNOWN"
    values, validity = feature.get("values"), feature.get("validity")
    if (not isinstance(values, list) or not isinstance(validity, list) or
            len(values) != width or len(validity) != width or
            not all(_finite(value) for value in values)):
        return "FEATURE_MISSING"
    if not all(value is True for value in validity):
        return "FEATURE_INVALID"
    try:
        if _instant(feature.get("knowledge_cutoff")) > cutoff:
            return "FEATURE_NOT_AVAILABLE"
    except AdapterError:
        return "FEATURE_CLOCK_UNKNOWN"
    return None


def normalize_forward_labels(raw_build: dict, *, features: dict,
                             cutoff: str | datetime) -> dict:
    """Normalize mature outcomes within the original feature-date universe.

    Feature validity applies to every saved selected feature. Training and OOS
    callers must supply their own explicit offline cutoff. Exact raw clocks are
    checked before projecting the execution context to Core's second precision.
    ``label_available_at`` remains the raw clock; ``normalized_available_at``
    describes the whole section's Core dependency clock.
    """
    raw_ref = _sealed(raw_build, "stock_label_build_v1", "label_ref")
    feature_ref = _sealed(features, "stock_feature_build_v1", "feature_ref")
    instant = _instant(cutoff)
    cutoff_text = instant.isoformat().replace("+00:00", "Z")
    core_cutoff = _utc(cutoff_text, availability=True)
    columns = features.get("ordered_features")
    _require(isinstance(columns, list) and bool(columns) and
             all(isinstance(name, str) and name for name in columns) and
             len(columns) == len(set(columns)), "ordered feature columns required")
    feature_rows = features.get("rows")
    raw_rows = raw_build.get("rows")
    _require(isinstance(feature_rows, list) and isinstance(raw_rows, list) and bool(raw_rows),
             "saved label and feature rows required")
    indexed = {}
    for row in feature_rows:
        _require(isinstance(row, dict), "malformed feature row")
        key = row.get("security_id"), _session(row.get("session"))
        _require(isinstance(key[0], str) and bool(key[0]) and key not in indexed,
                 "duplicate or missing feature key")
        indexed[key] = row
    raw_index = {}
    for row in raw_rows:
        _require(isinstance(row, dict), "malformed raw label row")
        key = row.get("security_id"), _session(row.get("feature_session"))
        _require(isinstance(key[0], str) and bool(key[0]) and key not in raw_index,
                 "duplicate or missing raw label key")
        _require(isinstance(row.get("source_refs"), list), "raw label source refs required")
        raw_index[key] = row
    securities = sorted({key[0] for key in raw_index})
    sessions = sorted({key[1] for key in raw_index})
    _require(set(raw_index) == {(security, session) for security in securities for session in sessions},
             "complete raw label security/session grid required")
    spec = deepcopy(NORMALIZATION_SPEC)
    recipe_ref = digest({"normalization_spec": spec, "raw_label_spec": raw_build["label_spec"]})
    input_column = {"name": "raw_return", "dtype": "float64", "unit": "dimensionless",
                    "stage": "fact", "missing": "preserve"}
    output_column = {"name": "normalized_target", "dtype": "float64", "unit": "dimensionless",
                     "stage": "cross_sectional", "missing": "preserve"}
    rows, sections, plans, facts_list, contexts, frames = [], [], [], [], [], []
    for session in sessions:
        keys = [(security, session) for security in securities]
        reasons = {key: _eligible_reason(raw_index[key], indexed.get(key), len(columns), instant)
                   for key in keys}
        eligible = [list(key) for key in keys if reasons[key] is None]
        section_definition = {"contract_version": "stock_label_section_v1", "feature_session": session,
                              "eligible_keys": eligible, "raw_label_ref": raw_ref,
                              "feature_ref": feature_ref, "cutoff": cutoff_text,
                              "normalization_spec": spec}
        section_ref = digest(section_definition)
        sources = [
            {"id": "raw_labels", "data_ref": raw_ref, "view_ref": raw_ref,
             "revision_policy": "frozen_saved_label_build", "qualification": "observed",
             "availability_basis": "exact_raw_label_available_at"},
            {"id": "offline_eligibility", "data_ref": feature_ref, "view_ref": section_ref,
             "revision_policy": "frozen_feature_membership_and_explicit_outcome_cutoff",
             "qualification": "observed", "availability_basis": "derived_offline_cutoff_selection"}]
        plan = FeaturePlan.from_dict({"abi": ABI, "semantics": SEMANTICS, "recipe_ref": recipe_ref,
            "calendar_ref": raw_build["calendar_ref"], "reference_ref": section_ref,
            "reference_members": {session: {key[0]: None for key in keys if reasons[key] is None}},
            "input_schema": [input_column], "event_schema": {}, "sources": sources,
            "observation_domain": "sessions", "history_policy": "partial",
            "nodes": [{"name": "normalized_target", "op": "cs_zscore", "version": "1",
                       "inputs": ["raw_return"], "params": deepcopy(spec["params"]),
                       "column": output_column}],
            "outputs": [{"node": "normalized_target", "column": output_column}], "obligations": []})
        facts = FactBatch.from_dict({"abi": ABI, "calendar_ref": raw_build["calendar_ref"],
            "schema": [input_column], "sources": sources, "event_schema": {}, "events": [],
            "rows": [{"security_id": key[0], "session": session,
                      "values": [float(raw_index[key]["return"]) if reasons[key] is None else None],
                      "availability": [_utc(raw_index[key]["label_available_at"], availability=True)
                                       if reasons[key] is None else core_cutoff],
                      "sources": [["raw_labels"] if reasons[key] is None else ["offline_eligibility"]],
                      "missing_reasons": [reasons[key]]} for key in keys]})
        context = ExecutionContext.from_dict({"abi": ABI, "calendar_ref": raw_build["calendar_ref"],
            "reference_ref": section_ref, "sessions": [session], "cutoffs": {session: core_cutoff},
            "history_keys": [list(key) for key in keys], "output_keys": [list(key) for key in keys],
            "reference": [{"security_id": key[0], "session": session, "member": reasons[key] is None,
                           "industry": None, "available_at": core_cutoff, "source": "offline_eligibility"}
                          for key in keys]})
        frame = execute_feature_plan(plan, facts, context)
        frame_ref = frame.identity
        wire = frame.to_dict()
        projected = {(row["security_id"], row["session"]): row for row in wire["rows"]}
        sections.append({"feature_session": session, "eligible_keys": eligible, "section_ref": section_ref,
                         "core_plan_ref": plan.identity, "core_context_ref": context.identity,
                         "fact_ref": facts.identity, "frame_ref": frame_ref})
        plans.append(plan.to_dict())
        facts_list.append(facts.to_dict())
        contexts.append(context.to_dict())
        frames.append(wire)
        for key in keys:
            raw, computed = raw_index[key], projected[key]
            target = computed["values"][0]
            valid = reasons[key] is None and computed["valid"][0] and _finite(target)
            refs = deepcopy(raw["source_refs"])
            refs.extend([raw_ref, feature_ref, section_ref, frame_ref])
            rows.append({"security_id": key[0], "feature_session": session,
                "start_session": raw.get("start_session"), "end_session": raw.get("end_session"),
                "raw_return": raw.get("return"), "normalized_target": float(target) if valid else None,
                "label_available_at": raw.get("label_available_at"),
                "normalized_available_at": computed["availability"][0] if valid else None,
                "valid": bool(valid), "invalid_reason": None if valid else
                         reasons[key] or "NORMALIZATION_UNDEFINED",
                "source_refs": sorted(set(refs))})
    result = {"contract_version": CONTRACT_VERSION, "raw_label_ref": raw_ref, "feature_ref": feature_ref,
              "normalization_spec": spec, "cutoff": cutoff_text, "sections": sections,
              "core_plan": plans, "core_facts": facts_list, "core_context": contexts, "core_frames": frames,
              "frame_ref": digest([{ "feature_session": section["feature_session"],
                                     "frame_ref": section["frame_ref"]} for section in sections]), "rows": rows}
    result["label_ref"] = digest(result)
    return result

"""Offline outcome normalization through the existing Feature Core.

This consumes saved label/feature builds. It neither reads Data nor admits
outcomes to the decision-fact adapter. Each actual feature session is a separate
Core cross-section; there is no fitted scaler or second numerical executor.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from axiom_engine.core import (ABI, SEMANTICS, ExecutionContext, FactBatch,
                               FeaturePlan, execute_feature_plan)

from .data_adapter import _require
from .labels import _instant, _session
from .stock_artifacts import digest


CONTRACT_VERSION = "stock_normalized_label_build_v1"
from .stock_label_contracts import NORMALIZATION_SPEC, _finite, normalization_section_inputs


def _sealed(build: Any, version: str, ref_name: str) -> str:
    _require(isinstance(build, dict) and build.get("contract_version") == version,
             "unsupported " + version)
    ref = build.get(ref_name)
    _require(ref == digest({key: value for key, value in build.items() if key != ref_name}),
             ref_name + " integrity mismatch")
    return ref




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
    rows, sections, plans, facts_list, contexts, frames = [], [], [], [], [], []
    for session in sessions:
        keys = [(security, session) for security in securities]
        parts = normalization_section_inputs(raw_build, feature_ref=feature_ref, feature_rows=indexed,
            session=session, securities=securities, width=len(columns), cutoff=cutoff_text,
            abi=ABI, semantics=SEMANTICS, raw_index=raw_index)
        reasons, eligible, section_ref = parts['reasons'], parts['eligible_keys'], parts['section_ref']
        plan = FeaturePlan.from_dict(parts['plan'])
        facts = FactBatch.from_dict(parts['facts'])
        context = ExecutionContext.from_dict(parts['context'])
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

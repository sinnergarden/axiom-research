"""DataBatch -> Core ABI at the Research ownership boundary.

This maps already selected daily facts; it does not select revisions, compute a
feature, materialize a View, or admit event/label/replay data as decision facts.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from hashlib import sha256
import json
import math
from typing import Any, Mapping

from axiom_engine.core import (ABI, SEMANTICS, ExecutionContext, FactBatch,
                               FeaturePlan, FeatureFrame, execute_feature_plan)

from .view_ref import ViewRef


class AdapterError(ValueError):
    """The Data response cannot be mapped without losing required semantics."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise AdapterError(reason)


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False, default=str)
    return "sha256:" + sha256(payload.encode()).hexdigest()


def _utc(value: Any, *, availability: bool = False) -> str:
    try:
        instant = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise AdapterError("invalid provenance/cutoff timestamp") from exc
    _require(instant.tzinfo is not None and instant.utcoffset() is not None,
             "timestamp lacks timezone")
    instant = instant.astimezone(timezone.utc)
    # Core's current ABI has second precision. Availability rounds up and
    # cutoff rounds down, so this adapter never admits information early.
    # Original subsecond timestamps remain in source_evidence.
    if availability and instant.microsecond:
        instant += timedelta(seconds=1)
    instant = instant.replace(microsecond=0)
    return instant.strftime("%Y-%m-%dT%H:%M:%SZ")


def _versioned(batch: Any, purpose: str) -> tuple[dict, dict, dict]:
    _require(hasattr(batch, "to_json"), "expected DataBatch")
    wire = batch.to_json()
    _require(isinstance(wire, dict) and set(wire) == {"records", "field_meta", "context"},
             "unknown DataBatch shape")
    records, metadata, context = wire["records"], wire["field_meta"], wire["context"]
    _require(isinstance(records, list) and isinstance(metadata, dict) and isinstance(context, dict),
             "malformed DataBatch")
    _require(context.get("contract_version") == "data_batch_v1" and
             isinstance(context.get("reader_version"), str) and context["reader_version"] and
             isinstance(context.get("snapshot_id"), str) and
             context["snapshot_id"] not in ("", "current", "latest"),
             "unsupported or unpinned DataBatch")
    query = context.get("query")
    _require(isinstance(query, dict) and query.get("purpose") == purpose,
             f"expected {purpose} DataBatch")
    return records, metadata, context


def _metadata(field_meta: Mapping[str, Any], field: str,
              keys: set[tuple[str, str]]) -> tuple[dict, dict]:
    spec = field_meta.get(field)
    _require(isinstance(spec, dict) and isinstance(spec.get("by_key"), list),
             f"missing field provenance: {field}")
    by_key = {}
    for item in spec["by_key"]:
        _require(isinstance(item, dict), f"invalid provenance: {field}")
        key = item.get("security_id"), item.get("session")
        _require(key in keys and key not in by_key, f"duplicate/unexpected provenance: {field}")
        by_key[key] = item
    _require(set(by_key) == keys, f"incomplete provenance: {field}")
    return spec, by_key


def _row_index(records: list, fields: tuple[str, ...]) -> dict[tuple[str, str], dict]:
    rows = {}
    for row in records:
        _require(isinstance(row, dict) and all(f in row for f in fields),
                 "record lacks requested fields")
        key = row.get("security_id"), row.get("session")
        _require(all(isinstance(v, str) and v for v in key) and key not in rows,
                 "missing or duplicate security/session key")
        rows[key] = row
    return rows


def _leaves(meta: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """An adjusted cell has price, factor and anchor provenance, not one clock."""
    nested = [meta[k] for k in ("price_provenance", "factor_provenance",
                                 "anchor_factor_provenance") if k in meta]
    if nested:
        _require(all(isinstance(item, dict) for item in nested), "invalid derived provenance")
        return [leaf for item in nested for leaf in _leaves(item)]
    return [meta]


def _availability(meta: Mapping[str, Any], cutoff: str) -> str:
    times = [_utc(leaf["usable_from"], availability=True) for leaf in _leaves(meta)
             if leaf.get("usable_from") is not None]
    available = max(times) if times else cutoff  # absence established by the query at cutoff
    _require(available <= cutoff, "cell provenance exceeds its decision cutoff")
    return available


def _value(value: Any, dtype: str) -> Any:
    if value is None:
        return None
    if dtype == "float64":
        _require(type(value) in (int, float) and not isinstance(value, bool) and
                 math.isfinite(value), "nonfinite/non-numeric fact")
        _require(type(value) is not int or abs(value) <= 2**53,
                 "integer cannot be represented exactly in Core float64")
        return float(value)
    if dtype == "bool":
        _require(type(value) is bool, "invalid Boolean fact")
    elif dtype in ("date", "string"):
        _require(type(value) is str and bool(value), "invalid text/date fact")
    return value


@dataclass(frozen=True)
class AdaptedFacts:
    facts: FactBatch
    context: ExecutionContext
    plan: FeaturePlan
    source_evidence: Mapping[str, Mapping[str, Any]]
    view_ref: ViewRef

    def execute(self) -> FeatureFrame:
        """Run the existing Core executor against the frozen adapter output."""
        return execute_feature_plan(self.plan, self.facts, self.context)


def adapt_fixed_universe_batch(batch: Any, *, universe: tuple[str, ...],
                               recipe_ref: str,
                               output_keys: tuple[tuple[str, str], ...] | None = None,
                               lag_sessions: int = 1) -> AdaptedFacts:
    """Map a Research-declared fixed universe to the existing Core reference ABI.

    This declaration is not Data membership or historical index evidence. Its
    explicit origin and digest remain in source_evidence. Missing prices still
    remain missing; membership never implies listing, tradeability or coverage.
    """
    _, _, ctx = _versioned(batch, "decision_facts")
    q = ctx["query"]
    _require(isinstance(universe, tuple) and universe and
             len(set(universe)) == len(universe) and list(universe) == q["symbols"],
             "fixed universe must exactly match query symbols/order")
    declaration = {"schema_version": "research_fixed_universe_v1", "members": list(universe)}
    declaration_ref = _digest(declaration)
    records, metadata = [], []
    for security in universe:
        for session in q["sessions"]:
            records.append(dict(security_id=security, session=session, is_member=True))
            metadata.append(dict(security_id=security, session=session,
                usable_from=q["cutoff_by_session"][session], missing_reason=None,
                availability_basis="research_declared_fixed_universe",
                declaration_ref=declaration_ref))
    wire = dict(records=records,
        field_meta={"is_member": {"dtype": "bool", "unit": None, "by_key": metadata}},
        context={"contract_version": "data_batch_v1", "snapshot_id": ctx["snapshot_id"],
            "domain": "universe_membership", "reader_version": "research_fixed_universe/1",
            "origin": "Research declaration; not a Data.members read",
            "declaration": declaration,
            "query": {**q, "fields": ["is_member"], "price_basis": "unadjusted",
                      "adjustment_anchor": None, "universe_id": declaration_ref}})

    class Reference:
        def to_json(self):
            return wire

    return adapt_decision_batch(batch, reference=Reference(), recipe_ref=recipe_ref,
                               output_keys=output_keys, lag_sessions=lag_sessions)


def adapt_decision_batch(batch: Any, *, reference: Any, recipe_ref: str,
                         output_keys: tuple[tuple[str, str], ...] | None = None,
                         lag_sessions: int = 1) -> AdaptedFacts:
    """Map matching daily Data reads to a Core identity and return plan.

    `reference` must be a Data.members result for the same snapshot, sessions,
    symbols, PIT policy and cutoffs. Missing membership fails closed. Input rows
    and metadata are keyed independently so DataFrame order never binds cells.
    Caller pins both reads before this function; it performs no I/O or writes.
    """
    _require(isinstance(recipe_ref, str) and recipe_ref.startswith("sha256:") and
             len(recipe_ref) == 71, "recipe_ref must be an immutable digest")
    _require(type(lag_sessions) is int and lag_sessions >= 1, "positive lag required")
    records, field_meta, ctx = _versioned(batch, "decision_facts")
    ref_records, ref_meta, ref_ctx = _versioned(reference, "decision_facts")
    q, rq = ctx["query"], ref_ctx["query"]
    _require(ref_ctx.get("domain") == "universe_membership" and
             rq.get("fields") == ["is_member"], "explicit membership batch required")
    _require(ctx["snapshot_id"] == ref_ctx["snapshot_id"] and
             all(q.get(k) == rq.get(k) for k in
                 ("symbols", "sessions", "pit_policy", "policy_by_session", "cutoff_by_session")),
             "reference/fact read context mismatch")
    fields = q.get("fields")
    _require(isinstance(fields, list) and fields and len(set(fields)) == len(fields),
             "explicit unique fact fields required")
    _require(q.get("price_basis") in ("unadjusted", "common_anchor_adjusted_v1"),
             "unsupported price basis")
    _require(q.get("price_basis") != "common_anchor_adjusted_v1" or
             isinstance(ctx.get("derivation"), dict), "adjusted facts lack derivation")
    sessions, symbols = q.get("sessions"), q.get("symbols")
    _require(isinstance(sessions, list) and sessions == sorted(set(sessions)) and
             isinstance(symbols, list) and symbols and len(symbols) == len(set(symbols)),
             "explicit ordered sessions/symbols required")
    cutoffs = q.get("cutoff_by_session")
    _require(isinstance(cutoffs, dict) and set(cutoffs) == set(sessions),
             "per-session cutoffs required")
    cutoffs = {s: _utc(cutoffs[s]) for s in sessions}
    expected = {(symbol, session) for symbol in symbols for session in sessions}
    rows = _row_index(records, tuple(fields))
    refs = _row_index(ref_records, ("is_member",))
    _require(set(rows) == set(refs) == expected, "incomplete daily/reference keys")
    specs = {name: _metadata(field_meta, name, expected) for name in fields}
    _, membership_meta = _metadata(ref_meta, "is_member", expected)
    schema = []
    for field in fields:
        definition = specs[field][0]
        original_type = str(definition.get("dtype", "")).lower()
        dtype = ("float64" if original_type in
                 {"float", "float32", "float64", "double", "int", "integer", "int32", "int64"}
                 else "bool" if original_type in {"bool", "boolean"}
                 else "string" if original_type in {"str", "string", "utf8"}
                 else "date" if original_type == "date" else None)
        _require(dtype is not None, f"Core cannot represent dtype for {field}")
        unit = definition.get("unit")
        _require(isinstance(unit, str) and bool(unit) or dtype == "bool" and unit is None,
                 f"unit unknown for {field}")
        schema.append(dict(name=field, dtype=dtype, unit=unit or "dimensionless",
                           stage="fact", missing="preserve"))
    data_ref = _digest({"snapshot_id": ctx["snapshot_id"], "domain": ctx["domain"]})
    view_ref = _digest({"snapshot_id": ctx["snapshot_id"], "query": q,
                        "reader_version": ctx["reader_version"],
                        "derivation": ctx.get("derivation")})
    reference_ref = _digest({"snapshot_id": ref_ctx["snapshot_id"], "query": rq,
                             "reader_version": ref_ctx["reader_version"]})
    calendar_ref = _digest({"sessions": sessions, "snapshot_id": ctx["snapshot_id"]})
    sources, evidence = {}, {}

    def bind(field: str, key: tuple[str, str], meta: dict, *, ref: bool = False) -> str:
        source_id = _digest({"field": field, "key": key, "meta": meta, "reference": ref})
        if source_id not in sources:
            leaves = _leaves(meta)
            basis = "+".join(sorted({str(m.get("availability_basis") or "missing") for m in leaves}))
            qualification = ("synthetic" if "synthetic" in basis else
                             "best_effort" if "assumption" in basis else
                             "verified" if all(m.get("evidence_ref") for m in leaves) else
                             "observed" if all(m.get("first_observed_at") for m in leaves) else
                             "best_effort")
            sources[source_id] = dict(id=source_id, data_ref=data_ref if not ref else reference_ref,
                                      view_ref=view_ref if not ref else reference_ref,
                                      revision_policy=str(q["pit_policy"] if not ref else rq["pit_policy"]),
                                      qualification=qualification, availability_basis=basis)
            evidence[source_id] = {"field": field, "security_id": key[0], "session": key[1],
                                   "provenance": meta, "query_context": ref_ctx if ref else ctx}
        return source_id

    fact_rows, reference_rows, members = [], [], {session: {} for session in sessions}
    for key in sorted(expected):
        symbol, session = key
        member = refs[key]["is_member"]
        _require(type(member) is bool, "unknown membership cannot form Core mask")
        member_provenance = membership_meta[key]
        source = bind("is_member", key, member_provenance, ref=True)
        reference_rows.append(dict(security_id=symbol, session=session, member=member,
                                   industry=None, available_at=_availability(member_provenance, cutoffs[session]),
                                   source=source))
        if member:
            members[session][symbol] = None
        values, availability, ids, reasons = [], [], [], []
        for field, col in zip(fields, schema):
            value = _value(rows[key][field], col["dtype"])
            meta = specs[field][1][key]
            reason = meta.get("missing_reason")
            _require(value is None or reason is None, "present fact has missing reason")
            values.append(value)
            availability.append(_availability(meta, cutoffs[session]))
            ids.append([bind(field, key, meta)])
            reasons.append(str(reason or "MISSING") if value is None else None)
        fact_rows.append(dict(security_id=symbol, session=session, values=values,
                              availability=availability, sources=ids, missing_reasons=reasons))
    keys = sorted(expected)
    outputs = list(output_keys) if output_keys is not None else keys
    _require(bool(outputs) and len(outputs) == len(set(outputs)) and set(outputs) <= expected,
             "invalid output keys")
    numeric = next((col for col in schema if col["dtype"] == "float64"), None)
    _require(numeric is not None, "return example requires a numeric input")
    base = numeric["name"]
    def col(name: str, unit: str) -> dict:
        return dict(name=name, dtype="float64", unit=unit, stage="base", missing="preserve")
    nodes = [dict(name="value", op="identity", version="1", inputs=[base], params={},
                  column=col("value", numeric["unit"])),
             dict(name="lag", op="shift", version="1", inputs=[base],
                  params={"periods": lag_sessions}, column=col("lag", numeric["unit"])),
             dict(name="return", op="pct_change", version="1", inputs=[base],
                  params={"periods": lag_sessions, "fill_method": "none", "zero": "missing"},
                  column=col("return", "dimensionless"))]
    plan = FeaturePlan.from_dict(dict(abi=ABI, semantics=SEMANTICS, recipe_ref=recipe_ref,
        calendar_ref=calendar_ref, reference_ref=reference_ref, reference_members=members,
        input_schema=schema, event_schema={}, sources=sorted(sources.values(), key=lambda s: s["id"]),
        observation_domain="sessions", history_policy="partial", nodes=nodes,
        outputs=[{"node": n["name"], "column": n["column"]} for n in nodes], obligations=[]))
    facts = FactBatch.from_dict(dict(abi=ABI, calendar_ref=calendar_ref, schema=schema,
        sources=sorted(sources.values(), key=lambda s: s["id"]), rows=fact_rows,
        event_schema={}, events=[]))
    context = ExecutionContext.from_dict(dict(abi=ABI, calendar_ref=calendar_ref,
        reference_ref=reference_ref, sessions=sessions, cutoffs=cutoffs,
        history_keys=[list(k) for k in keys], output_keys=[list(k) for k in outputs],
        reference=reference_rows))
    return AdaptedFacts(facts, context, plan, evidence, ViewRef.from_batch(batch))

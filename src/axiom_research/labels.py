"""Outcome-only forward labels over Data's already adjusted price batches.

This module selects endpoints on an explicit exchange calendar. It neither
adjusts prices nor supplies decision facts, trading execution or a fitted model.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math
from typing import Any, Mapping

from .data_adapter import AdapterError, _require, _versioned


CONTRACT_VERSION = "stock_label_build_v1"
PRICE_BASIS = "common_anchor_adjusted_v1"


def _canonical_bytes(value: Any) -> bytes:
    """Encode strict canonical JSON; never stringify an unknown scalar."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AdapterError("label evidence must be strict JSON") from exc


def _content_ref(value: Any) -> str:
    """Hash strict canonical JSON; never stringify an unknown scalar."""
    return "sha256:" + sha256(_canonical_bytes(value)).hexdigest()


def _batch_source_refs(records: list, field_meta: dict, context: dict) -> dict:
    """Resolve the three existing refs with one encoding of each subtree.

    The fixed DataBatch outer keys are already admitted by _versioned. Feed
    their exact canonical JSON bytes in sorted order, releasing each buffer
    before encoding the next one. Nothing is reused across calls or cutoffs.
    """
    whole = sha256()
    whole.update(b'{"context":')
    context_bytes = _canonical_bytes(context)
    whole.update(context_bytes)
    del context_bytes
    whole.update(b',"field_meta":')
    metadata_bytes = _canonical_bytes(field_meta)
    metadata_ref = "sha256:" + sha256(metadata_bytes).hexdigest()
    whole.update(metadata_bytes)
    del metadata_bytes
    whole.update(b',"records":')
    record_bytes = _canonical_bytes(records)
    records_ref = "sha256:" + sha256(record_bytes).hexdigest()
    whole.update(record_bytes)
    del record_bytes
    whole.update(b'}')
    return {"source_ref": "sha256:" + whole.hexdigest(),
            "records_ref": records_ref, "field_meta_ref": metadata_ref}


def _session(value: Any) -> str:
    _require(isinstance(value, str), "session must be YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise AdapterError("session must be YYYY-MM-DD") from exc
    _require(parsed.isoformat() == value, "session must be YYYY-MM-DD")
    return value


def _instant(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise AdapterError("label timestamp must be timezone-aware") from exc
    _require(isinstance(value, datetime) and value.tzinfo is not None and
             value.utcoffset() is not None, "label timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _sessions(values: Any, label: str) -> tuple[str, ...]:
    _require(isinstance(values, (list, tuple)) and bool(values), label + " required")
    values = tuple(_session(value) for value in values)
    _require(values == tuple(sorted(set(values))), label + " must be ordered and unique")
    return values


def _query_context(ctx: dict, calendar: tuple[str, ...]) -> tuple[dict, tuple[str, ...], str, datetime]:
    q, derived = ctx["query"], ctx.get("derivation")
    _require(ctx.get("domain") == "market_daily" and
             isinstance(q.get("fields"), list) and len(q["fields"]) == 2 and
             set(q["fields"]) == {"open", "close"} and
             q.get("price_basis") == PRICE_BASIS,
             "label outcomes require Data-adjusted market_daily open/close")
    _require(isinstance(q.get("pit_policy"), str) and bool(q["pit_policy"]),
             "label query PIT policy required")
    symbols = q.get("symbols")
    _require(isinstance(symbols, list) and bool(symbols) and
             all(isinstance(s, str) and bool(s) for s in symbols) and
             len(symbols) == len(set(symbols)), "unique label securities required")
    sessions = _sessions(q.get("sessions"), "query sessions")
    _require(set(sessions) <= set(calendar), "query sessions must belong to the supplied calendar")
    cutoffs = q.get("cutoff_by_session")
    _require(isinstance(cutoffs, dict) and set(cutoffs) == set(sessions),
             "complete label query cutoffs required")
    instants = {_instant(value) for value in cutoffs.values()}
    _require(len(instants) == 1, "common-anchor labels require one outcome query cutoff")
    cutoff = next(iter(instants))
    anchor = q.get("adjustment_anchor")
    _require(anchor in calendar and isinstance(derived, dict) and
             derived.get("recipe_version") == "common_anchor_price_v1" and
             derived.get("formula") == "price_t * factor_t / factor_anchor" and
             derived.get("anchor_session") == anchor and
             derived.get("decision_session") == sessions[-1] and anchor <= sessions[-1],
             "Data common-anchor derivation required")
    _require(_instant(derived.get("decision_cutoff")) == cutoff,
             "derivation cutoff mismatch")
    factor_field = derived.get("factor_field")
    _require(isinstance(factor_field, str) and bool(factor_field), "factor field required")
    for name in ("price_query", "factor_query"):
        native = derived.get(name)
        expected_sessions = set(sessions) if name == "price_query" else set(sessions) | {anchor}
        _require(isinstance(native, dict) and native.get("purpose") == "label_outcomes" and
                 native.get("price_basis") == "unadjusted" and
                 native.get("adjustment_anchor") is None and
                 isinstance(native.get("symbols"), list) and
                 len(native["symbols"]) == len(symbols) and set(native["symbols"]) == set(symbols) and
                 set(native.get("sessions") or ()) == expected_sessions and
                 native.get("pit_policy") == q.get("pit_policy"),
                 "label derivation must retain matching outcome-only native queries")
        _require(all((native.get("policy_by_session") or {}).get(session) ==
                     (q.get("policy_by_session") or {}).get(session) for session in sessions),
                 "native per-session outcome policy mismatch")
        native_cutoffs = native.get("cutoff_by_session")
        _require(isinstance(native_cutoffs, dict) and set(native_cutoffs) == expected_sessions and
                 all(_instant(value) == cutoff for value in native_cutoffs.values()),
                 "native outcome cutoff mismatch")
        _require({"open", "close"} <= set(native.get("fields") or ()) if name == "price_query"
                 else factor_field in (native.get("fields") or ()), "native derivation fields mismatch")
    return q, tuple(symbols), anchor, cutoff


def _keyed(items: Any, expected: set[tuple[str, str]], label: str) -> dict:
    try:
        _require(isinstance(items, list), "missing " + label)
        indexed = {}
        for item in items:
            _require(isinstance(item, dict), "malformed " + label)
            key = item.get("security_id"), item.get("session")
            _require(key in expected and key not in indexed, "duplicate/unexpected " + label + " key")
            indexed[key] = item
        return indexed
    finally:
        items = expected = item = key = indexed = None


def _endpoint(rows: dict, metadata: dict, key: tuple[str, str], field: str,
              anchor: str, cutoff: datetime) -> tuple[float | None, list[datetime], str | None]:
    label = "start_open" if field == "open" else "end_close"
    row, meta = rows.get(key), metadata.get(key)
    if row is None or field not in row or row[field] is None:
        return None, [], "missing_" + label
    value = row[field]
    if type(value) not in (int, float) or not math.isfinite(value):
        return None, [], "invalid_" + label
    if value <= 0:
        return None, [], "nonpositive_" + label
    if meta is None:
        return None, [], "missing_" + label + "_provenance"
    if meta.get("missing_reason"):
        return None, [], "unavailable_" + label + ":" + str(meta["missing_reason"])
    clocks = []
    for name in ("price_provenance", "factor_provenance", "anchor_factor_provenance"):
        leaf = meta.get(name)
        expected_session = anchor if name == "anchor_factor_provenance" else key[1]
        if not isinstance(leaf, Mapping) or leaf.get("usable_from") is None:
            return None, clocks, "unknown_" + label + "_availability"
        if (leaf.get("security_id"), leaf.get("session")) != (key[0], expected_session):
            return None, clocks, "invalid_" + label + "_provenance_key"
        if leaf.get("missing_reason"):
            return None, clocks, "unavailable_" + label + ":" + str(leaf["missing_reason"])
        try:
            clock = _instant(leaf["usable_from"])
        except AdapterError:
            return None, clocks, "unknown_" + label + "_availability"
        if clock > cutoff:
            return None, clocks, "unavailable_" + label + ":provenance_exceeds_query_cutoff"
        clocks.append(clock)
    return float(value), clocks, None


def _compile_forward_index(records, field_meta, ctx, calendar, *, parsed_query=None):
    """Compile one already-selected domain; never select another revision."""
    keys = indexed = metadata = None
    try:
        q, securities, anchor, cutoff = (parsed_query if parsed_query is not None else
                                         _query_context(ctx, calendar))
        keys = {(security, session) for security in securities for session in q["sessions"]}
        indexed = _keyed(records, keys, "label record")
        metadata = {}
        for field in ("open", "close"):
            definition = field_meta.get(field) or {}
            _require(isinstance(definition, dict), "malformed " + field + " metadata")
            metadata[field] = _keyed(definition.get("by_key", []), keys, field + " provenance")
        return dict(securities=securities, anchor=anchor, cutoff=cutoff, indexed=indexed,
                    metadata=metadata, positions={session: i for i, session in enumerate(calendar)})
    except BaseException:
        if indexed is not None: indexed.clear()
        if metadata is not None:
            for value in metadata.values(): value.clear()
            metadata.clear()
        raise
    finally:
        records = field_meta = ctx = keys = indexed = metadata = definition = None


def _forward_rows(records, field_meta, ctx, *, calendar, features, horizon_sessions, source_ref,
                  _domain=None):
    """The single Raw operator for dynamic and compact cached outcomes.

    Storage identities are supplied by the owner; arithmetic, null precedence,
    exchange endpoints and the six native clocks never depend on the codec.
    """
    _require(type(horizon_sessions) is int and horizon_sessions > 0,
             "horizon_sessions must be a positive integer")
    _require(set(features) <= set(calendar), "feature sessions must belong to the supplied calendar")
    data = indexed = metadata = positions = None
    borrowed = False
    try:
        if _domain is None:
            data = _compile_forward_index(records, field_meta, ctx, calendar)
        else:
            from .stock_compact_labels import _RawPriceDomain
            _require(type(_domain) is _RawPriceDomain, "owner-loaded Raw price domain required")
            data = _domain._borrow(calendar, features, source_ref, horizon_sessions)
            borrowed = True
        securities, anchor, cutoff = (data[k] for k in ("securities", "anchor", "cutoff"))
        indexed, metadata, positions = (data[k] for k in ("indexed", "metadata", "positions"))
        for feature in features:
            position = positions[feature]
            start = calendar[position + 1] if position + 1 < len(calendar) else None
            end = calendar[position + horizon_sessions] if position + horizon_sessions < len(calendar) else None
            for security in securities:
                value, available, reason = None, None, "calendar_endpoint_uncovered"
                if start is not None and end is not None:
                    opening, start_clocks, start_reason = _endpoint(indexed, metadata["open"],
                        (security, start), "open", anchor, cutoff)
                    closing, end_clocks, end_reason = _endpoint(indexed, metadata["close"],
                        (security, end), "close", anchor, cutoff)
                    reason = start_reason or end_reason
                    if reason is None:
                        value = closing / opening - 1.0
                        if not math.isfinite(value):
                            value, reason = None, "nonfinite_return"
                        else:
                            available = max(start_clocks + end_clocks).isoformat().replace("+00:00", "Z")
                yield dict(security_id=security, feature_session=feature,
                    start_session=start, end_session=end, **{"return": value},
                    label_available_at=available, valid=reason is None,
                    invalid_reason=reason, source_refs=[source_ref])
    finally:
        if borrowed: _domain._release()
        records = field_meta = ctx = data = indexed = metadata = positions = _domain = None


def build_forward_labels(batch: Any, *, calendar: tuple[str, ...] | list[str],
                         feature_sessions: tuple[str, ...] | list[str],
                         horizon_sessions: int = 5) -> dict:
    """Legacy logical projection of the same operator; not the v3 writer."""
    calendar = _sessions(calendar, "calendar")
    features = _sessions(feature_sessions, "feature sessions")
    records, field_meta, ctx = _versioned(batch, "label_outcomes")
    _, _, anchor, _ = _query_context(ctx, calendar)
    source_refs = _batch_source_refs(records, field_meta, ctx)
    source_ref = source_refs["source_ref"]
    calendar_ref = _content_ref({"contract_version": "stock_label_calendar_v1", "sessions": list(calendar)})
    rows = list(_forward_rows(records, field_meta, ctx, calendar=calendar, features=features,
                             horizon_sessions=horizon_sessions, source_ref=source_ref))
    result = dict(contract_version=CONTRACT_VERSION, label_spec={
        "label_id": f"forward_{horizon_sessions}_session_open_close_v1", "semantic_version": "1",
        "horizon_sessions": horizon_sessions, "start_session_offset": 1,
        "end_session_offset": horizon_sessions, "start_price": "open", "end_price": "close",
        "price_basis": PRICE_BASIS, "adjustment_anchor": anchor,
        "formula": "close(f+h) / open(f+1) - 1", "normalization": "none", "costs": "none",
        "availability": "max_endpoint_price_factor_anchor_usable_from",
        "missing_policy": "invalid_null_preserve_grid"},
        source_ref=source_ref, calendar_ref=calendar_ref,
        source_evidence={"context": deepcopy(ctx), "records_ref": source_refs["records_ref"],
                         "field_meta_ref": source_refs["field_meta_ref"],
                         "recovery": "fixed_snapshot_reader_queries_and_declared_Data_adjust_prices"},
        rows=rows)
    result["label_ref"] = _content_ref(result)
    return result


def mature_training_rows(label_build: dict, fit_cutoff: str | datetime) -> list[dict]:
    """Select valid outcomes actually available at the timezone-aware fit cutoff.

    No maturity is inferred from a label name or horizon. The saved endpoint
    must also be on/before the cutoff's UTC date; future-session outcomes cannot
    enter training even if an erroneous provenance clock claims early release.
    """
    _require(isinstance(label_build, dict) and label_build.get("contract_version") == CONTRACT_VERSION,
             "unsupported label build")
    _require(label_build.get("label_ref") == _content_ref(
        {key: value for key, value in label_build.items() if key != "label_ref"}),
        "label build integrity mismatch")
    cutoff = _instant(fit_cutoff)
    rows = label_build.get("rows")
    _require(isinstance(rows, list), "label rows required")
    return [deepcopy(row) for row in rows if row.get("valid") is True and
            _session(row.get("end_session")) <= cutoff.date().isoformat() and
            _instant(row.get("label_available_at")) <= cutoff]

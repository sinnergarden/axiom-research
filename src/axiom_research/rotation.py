"""A bounded ETF experiment: existing Data adjustment -> Core -> saved signals.

No target selection, execution, labels, training or account simulation lives here.
The R0 model-based SignalIdentity remains unchanged; this deterministic baseline
has a small, explicitly versioned feature-signal identity of its own.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
import errno
import json
from pathlib import Path
import tempfile
from typing import Any

from axiom_engine.core import ABI, FeaturePlan, execute_feature_plan
from . import contracts as c
from .api import from_dict, semantic_identity, to_dict, validate
from .data_adapter import adapt_fixed_universe_batch, _require, _digest, _utc
from .joint_build import FeatureBuild, load_feature_build, _daily_query, _ref, _file_digest
from .view_ref import ViewRef


POLICY = {
    "rebalance": "first_exchange_session_of_iso_week",
    "signal_session": "strict_previous_exchange_session",
    "positive_filter": "score > 0", "top_k": 1,
    "tie_break": "security_id_ascending",
    "no_positive": "cash", "execution_price": "unadjusted_open",
    "commission_rate": 0.0003, "minimum_commission": 0.0,
    "tax_rate": 0.0, "slippage_rate": 0.0,
}


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def _check_saved(path, version, identity, files):
    manifest = json.loads((path / "manifest.json").read_text())
    saved_identity = manifest.get("identity")
    identity_matches = (semantic_identity(from_dict(saved_identity)) == semantic_identity(from_dict(identity))
                        if version == "research_feature_build_v1" else saved_identity == identity)
    _require(manifest.get("schema_version") == version and identity_matches, "saved artifact identity mismatch")
    _require(set(manifest["files"]) == set(files), "unexpected artifact files")
    for name, digest in manifest["files"].items():
        _require(_file_digest(path / name) == digest, "saved artifact integrity failure: " + name)
    return manifest


def _publish(target, version, identity, write, files):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rotation-", dir=target.parent) as staging:
        stage = Path(staging) / "complete"
        stage.mkdir()
        write(stage)
        _write_json(stage / "manifest.json", dict(schema_version=version, identity=identity,
            files={name: _file_digest(stage / name) for name in files}))
        try:
            stage.rename(target)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            _check_saved(target, version, identity, files)
            return True
    return False


def _feature_identity(data, snapshot, prices, factors, implementation):
    from axiom_data.reader import READER_VERSION
    from axiom_data.derived import PRICE_ADJUSTMENT_VERSION
    validate(implementation, require_resolved=True)
    _require(snapshot not in ("", "current", "latest"), "pin a concrete Snapshot")
    p, f = _daily_query(prices), _daily_query(factors)
    _require(prices.domain == "market_daily" and prices.fields == ("close",) and
             factors.domain == "adjustment_factors" and factors.fields == ("factor",),
             "ETF native close/factor queries required")
    _require(prices.purpose == factors.purpose == "decision_facts" and
             prices.price_basis == factors.price_basis == "unadjusted" and
             prices.adjustment_anchor is factors.adjustment_anchor is None and
             prices.universe_id is factors.universe_id is None and
             prices.policy_by_session is factors.policy_by_session is None,
             "native decision facts with one PIT policy required")
    _require(all(p[k] == f[k] for k in ("symbols", "sessions", "pit_policy", "cutoff_by_session")),
             "price/factor scope/cutoff mismatch")
    _require(prices.sessions and tuple(prices.sessions) == tuple(sorted(set(prices.sessions))) and
             prices.symbols and len(set(prices.symbols)) == len(prices.symbols),
             "ordered unique sessions and unique symbols required")
    _require(set(prices.cutoff_by_session) == set(prices.sessions), "complete signal cutoffs required")
    for session in prices.sessions:
        _require(date.fromisoformat(session).isoformat() == session and
                 _utc(prices.cutoff_by_session[session])[:10] == session,
                 "each signal cutoff must be on its feature session (UTC)")
    manifest = data.store.load_snapshot(snapshot)  # metadata only on reuse
    _require(manifest.get("snapshot_id") == snapshot, "Snapshot identity mismatch")
    unit = manifest["domains"][prices.domain]["contract"]["fields"]["close"]["unit"]
    factor_spec = manifest["domains"][factors.domain]["contract"]["fields"]["factor"]
    _require(unit == "CNY/fund unit" and factor_spec["unit"] == "dimensionless", "ETF units mismatch")
    _require("trading_calendar" in manifest["domains"], "Snapshot calendar required")
    declaration = _ref("FixedResearchUniverse", {
        "schema_version": "research_fixed_universe_v1", "members": list(prices.symbols)})
    views = tuple(ViewRef(snapshot, q.domain, _daily_query(q), READER_VERSION,
        {"window_query": "last_up_to_21_exchange_sessions_at_each_feature_cutoff",
         "adjustment_version": PRICE_ADJUSTMENT_VERSION, "anchor": "feature_session"}).as_artifact_ref()
        for q in (prices, factors))
    scope = c.SessionRange(start=prices.sessions[0], end=prices.sessions[-1])
    columns = (c.Column(name="adjusted_close", dtype="float64", unit=unit, stage="base"),
               c.Column(name="momentum_20d", dtype="float64", unit="dimensionless", stage="base"),
               c.Column(name="momentum_rank", dtype="float64", unit="dimensionless", stage="cross_sectional"))
    formulas = ("Data.adjust_prices(close_t * factor_t / factor_feature_session)",
                "Core pct_change(20); require 21 finite positive adjusted closes; no filling",
                "Core cs_rank(momentum_20d, reference=fixed universe, missing=skip, ties=average)")
    definitions = tuple(c.FeatureDefinition(name=col.name, business_definition=col.name,
        inputs=("market_daily.close", "adjustment_factors.factor"), formula=formula,
        implementation_ref=implementation, window_sessions=21, lag_sessions=0,
        missing_policy="preserve; incomplete/nonpositive window invalid", outlier_policy="none",
        reference_universe_policy="Research-declared fixed universe; not historical index membership")
        for col, formula in zip(columns, formulas))
    requirements = c.DataRequirements(requirements=(c.Requirement(name="etf_rotation_inputs",
        required_by=tuple(col.name for col in columns), input_semantics="fixed ETF Snapshot and same-cutoff price/factor window",
        output_semantics="Data common-anchor adjusted close and Core 20-session momentum",
        edge_cases=("missing factor", "interior gap", "zero price", "insufficient history", "calendar gap"),
        reference_fixture="tests/test_rotation.py", status="DECISION", evidence=views),),
        key=("security_id", "session"), scope=scope, lookback_sessions=21,
        label_extension_sessions=0, pit_policy=prices.pit_policy, public_view_binding=views[0])
    release = c.FeatureRelease(name="etf_adjusted_momentum_20d_v1",
        business_definition="Fixed daily ETF momentum baseline; no fitted model or return claim",
        plan=c.FeaturePlanSpec(key=("security_id", "session"), features=definitions,
            ordered_output_schema=columns, data_requirements=requirements, history_policy="partial",
            pit_policy=prices.pit_policy, cutoff_policy="each feature window uses one explicit feature-session cutoff",
            execution_abi=ABI))
    return validate(c.FeatureBuildIdentity(data_refs=(_ref("DataSnapshot", manifest),),
        view_refs=views, feature_release=release, scope=scope, lookback_sessions=21,
        pit_policy=prices.pit_policy, cutoff_policy=release.plan.cutoff_policy,
        reference_universe=declaration, implementation_package=implementation))


def _calendar_check(data, snapshot, query):
    """Use Data's public state diagnostics to reject a compressed exchange calendar.

    Status unknown is allowed only after Data established the open calendar.
    It is not evidence of open-time tradeability; Runtime owns that question.
    """
    start, end = map(date.fromisoformat, (query.sessions[0], query.sessions[-1]))
    days = tuple((start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1))
    probe = replace(query, sessions=days, cutoff_by_session={s: query.cutoff_by_session.get(
        s, s + "T20:30:00+08:00") for s in days})
    batch = data.states(snapshot=snapshot, query=probe)
    wire = batch.to_json()
    metadata = {(m["security_id"], m["session"]): m for m in wire["field_meta"]["market_state"]["by_key"]}
    expected = {(security, day) for security in query.symbols for day in days}
    _require(len(wire["records"]) == len(expected) and
             {(r["security_id"], r["session"]) for r in wire["records"]} == expected and
             len(metadata) == len(expected), "incomplete calendar diagnostics")
    observed = {security: [] for security in query.symbols}
    for row in wire["records"]:
        reason = metadata[row["security_id"], row["session"]].get("missing_reason") or ""
        _require(not reason.startswith(("calendar_", "identity_")), "unproven calendar/identity: " + row["session"])
        state = row["market_state"]
        if state != "calendar_closed":
            _require(state in ("not_listed", "delisted", "normal_trading", "suspended", "source_gap") or
                     state == "unknown_status" and reason.startswith(("status_", "market_", "partial_session_")),
                     "unknown calendar state")
            observed[row["security_id"]].append(row["session"])
    _require(all(tuple(sorted(sessions)) == query.sessions for sessions in observed.values()),
             "supplied sessions omit/include exchange calendar dates")
    return wire


def _momentum_plan(adapted):
    p = adapted.plan.to_dict()
    unit = p["input_schema"][0]["unit"]
    def node(name, op, inputs, params, dtype="float64", stage="base", output_unit="dimensionless"):
        return dict(name=name, op=op, version="1", inputs=inputs, params=params,
                    column=dict(name=name, dtype=dtype, unit=output_unit, stage=stage, missing="preserve"))
    p["nodes"] = [
        node("zero_price", "constant", [], {"value": 0.0}, output_unit=unit),
        node("positive_price", "gt", ["close", "zero_price"], {"missing": "preserve"}, dtype="bool"),
        node("null_price", "constant", [], {"value": None}, output_unit=unit),
        node("adjusted_close", "where", ["positive_price", "close", "null_price"], {"missing": "preserve"}, output_unit=unit),
        node("window_min", "rolling", ["adjusted_close"], {"window": 21, "min_periods": 21,
             "inclusive_current": True, "reduction": "min", "missing": "propagate", "ddof": 0, "ties": "average"}, output_unit=unit),
        node("window_missing", "is_missing", ["window_min"], {}, dtype="bool"),
        node("window_valid", "not", ["window_missing"], {"missing": "preserve"}, dtype="bool"),
        node("endpoint_return", "pct_change", ["adjusted_close"], {"periods": 20, "fill_method": "none", "zero": "missing"}),
        node("null_score", "constant", [], {"value": None}),
        node("momentum_20d", "where", ["window_valid", "endpoint_return", "null_score"], {"missing": "preserve"}),
        node("momentum_rank", "cs_rank", ["momentum_20d"], {"group": "session", "unknown_group": "reject",
             "missing": "skip", "ties": "average", "excluded": "missing"}, stage="cross_sectional"),
    ]
    names = ("adjusted_close", "momentum_20d", "momentum_rank")
    p["outputs"] = [{"node": n["name"], "column": n["column"]} for n in p["nodes"] if n["name"] in names]
    return FeaturePlan.from_dict(p)


def _compact_evidence(frames, plans, evidence, contexts, calendar):
    """Save repeated provenance, source sets and Core bindings once by digest."""
    provenance, source_sets, binding_groups, bindings = {}, {}, {}, {}
    def intern(table, value):
        ref = _digest(value)
        table.setdefault(ref, value)
        return ref
    def provenance_ref(value):
        normalized = {k: {"provenance_ref": provenance_ref(v)} if k.endswith("_provenance") else v
                      for k, v in value.items()}
        return intern(provenance, normalized)
    compact_sources = {}
    for source, item in evidence.items():
        compact_sources[source] = {**{k: v for k, v in item.items() if k != "provenance"},
                                   "provenance_ref": provenance_ref(item["provenance"])}
    compact_frames = []
    for frame in frames:
        ids = []
        for binding in frame["source_bindings"]:
            source = binding["id"]
            bindings[source] = intern(binding_groups, {k: v for k, v in binding.items() if k != "id"})
            ids.append(source)
        rows = [{**{k: v for k, v in row.items() if k != "sources"},
                 "source_set_refs": [intern(source_sets, sources) for sources in row["sources"]]}
                for row in frame["rows"]]
        compact_frames.append({**{k: v for k, v in frame.items() if k not in ("source_bindings", "rows")},
                               "source_bindings_ref": intern(source_sets, ids), "rows": rows})
    compact_plans = [{**{k: v for k, v in plan.items() if k != "sources"},
                      "source_bindings_ref": frame["source_bindings_ref"]}
                     for plan, frame in zip(plans, compact_frames)]
    return dict(schema_version="rotation_feature_evidence_v1", frames=compact_frames,
        core_plans=compact_plans, source_evidence=compact_sources, query_contexts=contexts,
        provenance=provenance, source_sets=source_sets, source_bindings=bindings,
        binding_groups=binding_groups, calendar_diagnostics=calendar,
        limitations=sorted({x for ctx in contexts.values() for x in ctx.get("limitations", [])}))


def build_rotation_features(data: Any, *, snapshot: str, price_query: Any,
        factor_query: Any, destination: str | Path, implementation_package: c.ArtifactRef) -> FeatureBuild:
    """Persist/reuse the fixed 20D ETF FeatureBuild using only Data and Core."""
    identity = _feature_identity(data, snapshot, price_query, factor_query, implementation_package)
    target = Path(destination) / semantic_identity(identity)[7:]
    if target.exists():
        cached = load_feature_build(target)
        _require(semantic_identity(cached.identity) == semantic_identity(identity), "saved feature identity mismatch")
        return cached
    from axiom_data import adjust_prices
    import pyarrow as pa
    import pyarrow.parquet as pq
    calendar = _calendar_check(data, snapshot, price_query)
    frames, records, evidence, contexts, plans = [], [], {}, {}, []
    recipe = semantic_identity(identity.feature_release)
    for i, session in enumerate(price_query.sessions):
        history = price_query.sessions[max(0, i - 20):i + 1]
        cutoff = price_query.cutoff_by_session[session]
        cutoffs = {s: cutoff for s in history}
        price = data.read(snapshot=snapshot, query=replace(price_query, sessions=history, cutoff_by_session=cutoffs))
        factor = data.read(snapshot=snapshot, query=replace(factor_query, sessions=history, cutoff_by_session=cutoffs))
        adjusted = adjust_prices(price, factor, fields=("close",), factor_field="factor",
                                 anchor_session=session, decision_session=session)
        adapted = adapt_fixed_universe_batch(adjusted, universe=price_query.symbols, recipe_ref=recipe,
            output_keys=tuple((s, session) for s in price_query.symbols), lag_sessions=20)
        plan = _momentum_plan(adapted)
        frame = execute_feature_plan(plan, adapted.facts, adapted.context).to_dict()
        _require([(x["name"], x["unit"]) for x in frame["schema"]] ==
                 [(x.name, x.unit) for x in identity.feature_release.plan.ordered_output_schema], "feature schema mismatch")
        frames.append(frame)
        plans.append(plan.to_dict())
        for source, item in adapted.source_evidence.items():
            context_ref = _digest(item["query_context"])
            contexts[context_ref] = item["query_context"]
            evidence[source] = {**{k: v for k, v in item.items() if k != "query_context"}, "query_context_ref": context_ref}
        for row in frame["rows"]:
            records.append({"security_id": row["security_id"], "session": row["session"],
                            **dict(zip((c["name"] for c in frame["schema"]), row["values"]))})
    schema = pa.schema([("security_id", pa.string()), ("session", pa.string())] +
        [(col.name, pa.float64()) for col in identity.feature_release.plan.ordered_output_schema])
    def write(stage):
        pq.write_table(pa.Table.from_pylist(records, schema=schema), stage / "panel.parquet")
        _write_json(stage / "evidence.json", _compact_evidence(frames, plans, evidence, contexts, calendar))
    reused = _publish(target, "research_feature_build_v1", to_dict(identity), write,
                      ("panel.parquet", "evidence.json"))
    return FeatureBuild(identity, target, reused)


def _feature_ref(build):
    return _ref("FeatureBuild", {"identity": semantic_identity(build.identity),
        "files": json.loads((build.path / "manifest.json").read_text())["files"]}, uri=str(build.path))


@dataclass(frozen=True)
class RotationExperiment:
    path: Path
    reused: bool
    feature_build: FeatureBuild
    signal_path: Path
    feature_reused: bool
    signal_reused: bool

    def feature_frames(self):
        """Expand saved Core frames from shared proof tables, without execution."""
        evidence = self.feature_build.evidence()
        return [{**{k: v for k, v in frame.items() if k not in ("source_bindings_ref", "rows")},
            "source_bindings": [{"id": source, **evidence["binding_groups"][evidence["source_bindings"][source]]}
                                for source in evidence["source_sets"][frame["source_bindings_ref"]]],
            "rows": [{**{k: v for k, v in row.items() if k != "source_set_refs"},
                      "sources": [evidence["source_sets"][ref] for ref in row["source_set_refs"]]}
                     for row in frame["rows"]]} for frame in evidence["frames"]]

    def manifest(self):
        manifest = json.loads((self.path / "manifest.json").read_text())
        _check_saved(self.path, "rotation_experiment_v1", manifest["identity"], ("experiment.json",))
        _require(self.path.name == _digest(manifest["identity"])[7:], "experiment path identity mismatch")
        return json.loads((self.path / "experiment.json").read_text())

    def signal_frame(self):
        """Return the neutral signal_frame_v1 accepted by Engine SignalFrame."""
        experiment = self.manifest()
        manifest = json.loads((self.signal_path / "manifest.json").read_text())
        _require(_digest(manifest["identity"]) == experiment["signal_run_ref"], "signal identity mismatch")
        _check_saved(self.signal_path, "momentum_signal_run_v1", manifest["identity"], ("signal-frame.json",))
        _require(semantic_identity(_feature_ref(load_feature_build(self.feature_build.path))) ==
                 semantic_identity(from_dict(experiment["feature_build"])),
                 "feature dependency mismatch")
        return json.loads((self.signal_path / "signal-frame.json").read_text())


def load_rotation_experiment(path: str | Path) -> RotationExperiment:
    """Read a relocatable saved experiment with no Data root/runtime required."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    _check_saved(path, "rotation_experiment_v1", manifest["identity"], ("experiment.json",))
    experiment = json.loads((path / "experiment.json").read_text())
    build = load_feature_build(path.parent.parent / "features" / experiment["feature_identity"][7:])
    signal_path = path.parent.parent / "signals" / experiment["signal_run_ref"][7:]
    result = RotationExperiment(path, True, build, signal_path, True, True)
    result.signal_frame()
    return result


def build_rotation_experiment(data: Any, *, snapshot: str, price_query: Any, factor_query: Any,
        evaluation_range: c.SessionRange, destination: str | Path,
        implementation_package: c.ArtifactRef) -> RotationExperiment:
    """Build fixed features/signals and freeze the offline Engine handoff policy.

    evaluation_range changes reuse the full feature and signal artifacts. The
    fixed baseline policy is recorded for Engine, never evaluated here.
    """
    validate(evaluation_range, require_resolved=True)
    sessions = price_query.sessions
    _require(evaluation_range.start in sessions and evaluation_range.end in sessions and
             sessions.index(evaluation_range.start) >= 21,
             "evaluation requires an explicit prior session plus 21-observation warmup")
    destination = Path(destination)
    build = build_rotation_features(data, snapshot=snapshot, price_query=price_query,
        factor_query=factor_query, destination=destination / "features", implementation_package=implementation_package)
    feature_ref = _feature_ref(build)
    signal_identity = dict(schema_version="momentum_signal_identity_v1", feature_build=to_dict(feature_ref),
        implementation_package=to_dict(build.identity.implementation_package), signal_stage="final",
        score_semantics="momentum_20d", score_column="momentum_20d")
    # URI and display metadata do not change the binding or inhibit relocation.
    signal_identity["feature_build"]["uri"] = "relative:features/" + semantic_identity(build.identity)[7:]
    signal_identity["implementation_package"]["uri"] = "immutable:implementation"
    signal_identity["implementation_package"]["metadata"] = {}
    signal_ref = _digest(signal_identity)
    signal_path = destination / "signals" / signal_ref[7:]
    signal_reused = signal_path.exists()
    if signal_reused:
        _check_saved(signal_path, "momentum_signal_run_v1", signal_identity, ("signal-frame.json",))
    else:
        rows = []
        evidence = build.evidence()
        for frame in evidence["frames"]:
            for row in frame["rows"]:
                score, valid = row["values"][1], row["valid"][1]
                rows.append(dict(security_id=row["security_id"], session=row["session"],
                    knowledge_cutoff=row["cutoff"], available_at=row["availability"][1] or row["cutoff"],
                    score=score, valid=valid, invalid_reason=None if valid else
                        ";".join(row["reasons"][1]) or "INCOMPLETE_OR_NONPOSITIVE_WINDOW",
                    source_refs=sorted({feature_ref.content_digest,
                                        *(ref.content_digest for ref in build.identity.view_refs)})))
        frame = dict(contract_version="signal_frame_v1", signal_run_ref=signal_ref, signal_stage="final",
                     score_semantics="momentum_20d", universe=list(price_query.symbols),
                     rows=sorted(rows, key=lambda r: (r["security_id"], r["session"])))
        signal_reused = _publish(signal_path, "momentum_signal_run_v1", signal_identity,
            lambda stage: _write_json(stage / "signal-frame.json", frame), ("signal-frame.json",))
    experiment_identity = dict(schema_version="rotation_experiment_identity_v1", signal_run_ref=signal_ref,
        feature_identity=semantic_identity(build.identity), evaluation_range=to_dict(evaluation_range), policy=POLICY)
    target = destination / "experiments" / _digest(experiment_identity)[7:]
    reused = target.exists()
    if reused:
        _check_saved(target, "rotation_experiment_v1", experiment_identity, ("experiment.json",))
    else:
        portable_ref = to_dict(feature_ref)
        portable_ref["uri"] = "relative:features/" + semantic_identity(build.identity)[7:]
        experiment = {**experiment_identity, "feature_build": portable_ref,
            "data_snapshot": snapshot, "universe": list(price_query.symbols),
            "calendar_sessions": list(sessions), "signal_price_basis": "common_anchor_adjusted_v1",
            "baseline_status": "this-round explicit default; historical baseline equivalence unproven",
            "scope": "offline feature/signal acceptance; no profitability or strict historical PIT claim"}
        reused = _publish(target, "rotation_experiment_v1", experiment_identity,
            lambda stage: _write_json(stage / "experiment.json", experiment), ("experiment.json",))
    result = RotationExperiment(target, reused, build, signal_path, build.reused, signal_reused)
    result.signal_frame()
    return result

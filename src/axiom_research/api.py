"""Strict JSON serialization, semantic validation and content identity only."""
from __future__ import annotations

import hashlib
import json
import math
import re
import types
from dataclasses import fields, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from . import contracts as c


class ContractError(ValueError):
    pass


TYPES = {name: cls for name, cls in vars(c).items()
         if isinstance(cls, type) and issubclass(cls, c.Contract) and cls is not c.Contract}
KEY = ("security_id", "session")
BUILD_TYPES = (c.FeatureBuildIdentity, c.LabelBuildIdentity, c.DatasetIdentity,
               c.ModelIdentity, c.ModelReleaseManifest, c.SignalIdentity,
               c.SignalRunManifest)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is list:
        return [_json_value(v) for v in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: _json_value(v) for k, v in value.items()}
    raise ContractError("Expected finite JSON value with string object keys")


def _encode(value: Any, semantic: bool = False) -> Any:
    if isinstance(value, c.Contract):
        return {"contract_type": type(value).__name__, **{
            f.name: _encode(getattr(value, f.name), semantic)
            for f in fields(value)
            if not (semantic and (f.name == "metadata" or
                                   isinstance(value, c.ArtifactRef) and f.name == "uri"))}}
    if isinstance(value, (tuple, list)):
        return [_encode(v, semantic) for v in value]
    if isinstance(value, dict):
        if "contract_type" in value:
            return _encode(_decode(value, Any), semantic)
        return {k: _encode(v, semantic) for k, v in value.items()}
    return _json_value(value)


def _decode(value: Any, expected: Any) -> Any:
    if expected is Any:
        if type(value) is dict:
            _require(all(type(k) is str for k in value), "Expected string object keys")
            if "contract_type" in value:
                tag = value["contract_type"]
                _require(type(tag) is str and tag in TYPES, "Unknown contract type")
                return _decode(value, TYPES[tag])
            return {k: _decode(v, Any) for k, v in value.items()}
        if type(value) is list:
            return [_decode(v, Any) for v in value]
        return _json_value(value)
    origin, args = get_origin(expected), get_args(expected)
    if origin in (types.UnionType, Union):
        for option in args:
            try:
                return _decode(value, option)
            except ContractError:
                pass
        raise ContractError(f"Value does not match {expected}")
    if origin is Literal:
        _require(any(type(value) is type(x) and value == x for x in args),
                 f"Expected one of {args}")
        return value
    if origin is tuple:
        _require(type(value) is list, "Expected JSON array")
        return tuple(_decode(v, args[0]) for v in value)
    if origin is dict:
        _require(type(value) is dict and all(type(k) is str for k in value),
                 "Expected JSON object")
        return {k: _decode(v, args[1]) for k, v in value.items()}
    if isinstance(expected, type) and issubclass(expected, c.Contract):
        _require(type(value) is dict, f"Expected {expected.__name__} object")
        names = {f.name for f in fields(expected)}
        _require(set(value) == names | {"contract_type"},
                 f"{expected.__name__}: missing/extra fields: "
                 f"{set(value) ^ (names | {'contract_type'})}")
        _require(value["contract_type"] == expected.__name__, "Wrong contract type")
        hints = get_type_hints(expected)
        return expected(**{k: _decode(value[k], hints[k]) for k in names})
    if expected is float:
        _require(type(value) in (float, int) and math.isfinite(value), "Expected finite number")
        return float(value)
    _require(type(value) is expected, f"Expected {expected}, got {type(value)}")
    if expected is str:
        _require(bool(value.strip()), "Empty semantic string")
    return value


def _walk(value: Any):
    if type(value) is dict and "contract_type" in value:
        value = _decode(value, Any)
    yield value
    if is_dataclass(value):
        for f in fields(value):
            if f.name != "metadata":
                yield from _walk(getattr(value, f.name))
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from _walk(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _walk(v)


def unresolved(contract: c.Contract | dict | list) -> tuple[c.Unknown, ...]:
    """Return preserved blockers, deduplicated by ID; never manufacture values."""
    found = {}
    for v in _walk(contract):
        if isinstance(v, c.Requirement) and v.status == "UNKNOWN":
            v = c.Unknown(unknown_id="REQUIREMENT:" + v.name,
                          reason=v.input_semantics, required_evidence=v.output_semantics)
        if isinstance(v, c.Unknown):
            _require(v.unknown_id not in found or found[v.unknown_id] == v,
                     f"Conflicting UNKNOWN definitions: {v.unknown_id}")
            found[v.unknown_id] = v
    return tuple(found.values())


def _schema_names(schema: tuple[c.Column, ...]) -> tuple[str, ...]:
    names = tuple(x.name for x in schema)
    _require(bool(names) and len(names) == len(set(names)), "Empty/duplicate Feature columns")
    return names


def _same(left: c.Contract, right: c.Contract) -> bool:
    return _encode(left, True) == _encode(right, True)


def _inside(inner: c.SessionRange, outer: c.SessionRange) -> bool:
    return outer.start <= inner.start <= inner.end <= outer.end


def _semantics(obj: c.Contract) -> None:
    if hasattr(obj, "key"):
        _require(obj.key == KEY, "Explicit key must be (security_id, session)")
    for f in fields(obj):
        value = getattr(obj, f.name)
        if f.name in ("lag_sessions", "window_sessions", "lookback_sessions"):
            _require(isinstance(value, c.Unknown) or value >= 0, f"Invalid {f.name}")
        if f.name in ("horizon_sessions", "retrain_step_sessions", "train_window_sessions",
                      "threads", "concurrent_folds", "memory_limit_mb", "label_extension_sessions"):
            _require(isinstance(value, c.Unknown) or value > 0, f"Invalid {f.name}")
    if isinstance(obj, c.ArtifactRef):
        _require(bool(re.fullmatch(r"sha256:[0-9a-f]{64}", obj.content_digest)), "Invalid digest")
        _require(not re.search(r"(^|[/\\:@.])(?:latest|current)(?=$|[/\\:@.])",
                               obj.artifact_id + "/" + obj.uri, re.I), "Mutable reference")
        _require("/" not in obj.artifact_id and "\\" not in obj.artifact_id,
                 "A file path is not an artifact ID")
    if isinstance(obj, c.SessionRange):
        try:
            _require(date.fromisoformat(obj.start).isoformat() == obj.start and
                     date.fromisoformat(obj.end).isoformat() == obj.end, "Use ISO dates")
        except ValueError as exc:
            raise ContractError("Invalid session date") from exc
        _require(obj.start <= obj.end, "Reversed session range")
    if isinstance(obj, c.Requirement):
        _require(bool(obj.required_by) and bool(obj.edge_cases), "Incomplete requirement")
        if obj.status == "FACT":
            _require(bool(obj.evidence), "FACT requirement needs evidence")
    if isinstance(obj, (c.DataRequirements, c.CoreCapabilityRequirements)):
        names = [r.name for r in obj.requirements]
        _require(bool(names) and len(names) == len(set(names)), "Duplicate/empty requirements")
    if isinstance(obj, c.PITPolicy) and obj.exact_date_matching == "source_mode_dependent":
        _require(isinstance(obj.original_materialization, c.Unknown),
                 "Source-mode-dependent date matching requires unresolved original materialization")
    if isinstance(obj, c.FeaturePlanSpec):
        names = tuple(f.name for f in obj.features)
        _require(names == _schema_names(obj.ordered_output_schema), "Feature order/schema mismatch")
        _require(all(f.inputs for f in obj.features), "Feature inputs required")
    if isinstance(obj, c.LabelSpec):
        if not isinstance(obj.price_basis, c.Unknown) and not isinstance(obj.corporate_action_semantics, c.Unknown):
            expected_action = {"close": "none", "close_times_factor": "supplier_cumulative_factor"}
            _require(obj.corporate_action_semantics == expected_action[obj.price_basis],
                     "Label price basis/corporate action mismatch")
    if isinstance(obj, c.SplitSpec):
        bounds = [obj.train, obj.validation, obj.oos]
        for i, left in enumerate(bounds):
            for right in bounds[i + 1:]:
                if isinstance(left, c.SessionRange) and isinstance(right, c.SessionRange):
                    _require(left.end < right.start, "Dataset splits overlap or are out of order")
    if isinstance(obj, c.FoldSpec):
        if isinstance(obj.allowed_prediction_interval, c.SessionRange) and isinstance(obj.split.oos, c.SessionRange):
            _require(_inside(obj.allowed_prediction_interval, obj.split.oos), "Fold prediction outside OOS")
    if isinstance(obj, c.DatasetSpec):
        _schema_names(obj.ordered_model_schema)
        expected = obj.feature_release.plan.ordered_output_schema
        _require(_encode(obj.ordered_model_schema, True) == _encode(expected, True),
                 "Dataset model schema differs from Feature schema")
        if isinstance(obj.rolling_folds, tuple):
            _require(bool(obj.rolling_folds), "Empty rolling folds")
            ids = [x.fold_id for x in obj.rolling_folds]
            _require(len(ids) == len(set(ids)), "Duplicate fold IDs")
            intervals = [x.allowed_prediction_interval for x in obj.rolling_folds
                         if isinstance(x.allowed_prediction_interval, c.SessionRange)]
            for a, b in zip(intervals, intervals[1:]):
                _require(a.end < b.start, "Overlapping/unordered fold predictions")
            if isinstance(obj.split.oos, c.SessionRange):
                for fold in obj.rolling_folds:
                    if isinstance(fold.split.oos, c.SessionRange):
                        _require(_inside(fold.split.oos, obj.split.oos), "Fold OOS outside Dataset OOS")
    if isinstance(obj, c.SignalPlanSpec):
        stages = {x.alias: x.source_stage for x in obj.inputs}
        _require(bool(stages) and len(stages) == len(obj.inputs), "Duplicate/empty signal inputs")
        for node in obj.nodes:
            _require(node.name not in stages, "Duplicate signal node")
            _require(bool(node.inputs) and len(node.inputs) == len(node.input_stages),
                     "Signal input stage must be explicit")
            _require(len(node.inputs) == len(set(node.inputs)), "Duplicate node inputs")
            for name, stage in zip(node.inputs, node.input_stages):
                _require(name in stages and stages[name] == stage, "Missing input or wrong signal stage")
            if node.op == "daily_zscore":
                _require(len(node.inputs) == 1 and node.input_stages == ("raw_prediction",) and
                         node.output_stage == "daily_zscore" and not node.weights,
                         "Invalid daily zscore stages/weights")
            else:
                _require(len(node.inputs) == len(node.weights) and node.output_stage == "final",
                         "Invalid blend weights/output stage")
                _require(all(w >= 0 for w in node.weights) and
                         math.isclose(sum(node.weights), 1.0, abs_tol=1e-12), "Blend weights must sum to one")
            stages[node.name] = node.output_stage
        _require(obj.output in stages and obj.output not in {x.alias for x in obj.inputs},
                 "Signal output must name a declared node")
    if isinstance(obj, c.TrainingSpec):
        if type(obj.seed) is int:
            _require(obj.seed >= 0, "Invalid seed")
        for key in ("seed", "objective"):
            if key in obj.parameters:
                _require(obj.parameters[key] == getattr(obj, key), "Conflicting training parameter: " + key)
        if "n_estimators" in obj.parameters:
            _require(type(obj.parameters["n_estimators"]) is int and obj.parameters["n_estimators"] > 0,
                     "Invalid boosting rounds")
    if isinstance(obj, c.StrategyRecipeDraft):
        _require(bool(obj.datasets) and len(obj.datasets) == len(obj.training), "Dataset/training count mismatch")
        dataset_names = [ds.name for ds in obj.datasets]
        training_names = [ts.dataset_name for ts in obj.training]
        _require(len(set(dataset_names)) == len(dataset_names) and
                 len(set(training_names)) == len(training_names) and
                 set(dataset_names) == set(training_names), "Explicit dataset/training mapping required")
        labels = []
        for ds in obj.datasets:
            _require(_same(ds.feature_release, obj.feature_release), "Recipe Feature release mismatch")
            labels.append(_encode(ds.label, True))
        _require(len({json.dumps(x, sort_keys=True) for x in labels}) == len(labels), "Duplicate recipe labels")
        _require(all(_encode(i.label, True) in labels for i in obj.signal_plan.inputs),
                 "Signal input label absent from recipe")
        _require(all(_encode(e.label, True) in labels for e in obj.evaluation),
                 "Evaluation label absent from recipe")
    if isinstance(obj, c.FeatureBuildIdentity):
        _require(bool(obj.data_refs) and bool(obj.view_refs), "FeatureBuild requires Data/View refs")
        _require(obj.pit_policy == obj.feature_release.plan.pit_policy and
                 obj.cutoff_policy == obj.feature_release.plan.cutoff_policy,
                 "FeatureBuild policy differs from release")
        windows = [f.window_sessions + f.lag_sessions for f in obj.feature_release.plan.features
                   if type(f.window_sessions) is int and type(f.lag_sessions) is int]
        _require(obj.lookback_sessions >= max(windows, default=0), "Insufficient lookback")
    if isinstance(obj, c.LabelBuildIdentity):
        _require(bool(obj.data_refs), "LabelBuild requires Data refs")
        _require(obj.benchmark_semantics == obj.label.benchmark_semantics and
                 obj.corporate_action_semantics == obj.label.corporate_action_semantics,
                 "LabelBuild semantics differ from LabelSpec")
    if isinstance(obj, c.DatasetIdentity):
        _require(obj.feature_build.artifact_type == "FeatureBuild" and
                 obj.label_build.artifact_type == "LabelBuild", "Wrong dataset input artifact types")
    if isinstance(obj, c.ModelIdentity):
        _require(obj.dataset.artifact_type == "TrainingDataset", "Model requires TrainingDataset")
        _require(obj.training_spec.dataset_name == obj.dataset_spec.name, "Training recipe/Dataset mismatch")
        folds = obj.dataset_spec.rolling_folds
        _require(isinstance(folds, tuple) and any(_same(obj.fold, x) for x in folds),
                 "Model fold absent from DatasetSpec")
        if isinstance(obj.training_spec.fit_range, c.SessionRange) and isinstance(obj.fold.allowed_prediction_interval, c.SessionRange):
            _require(obj.training_spec.fit_range.end < obj.fold.allowed_prediction_interval.start,
                     "Model fit range overlaps allowed predictions")
        if obj.training_spec.fit_scope == "train_only":
            _require(isinstance(obj.fold.split.train, c.SessionRange) and
                     isinstance(obj.training_spec.fit_range, c.SessionRange) and
                     _inside(obj.training_spec.fit_range, obj.fold.split.train),
                     "train_only fit range must be contained in fold training partition")
    if isinstance(obj, c.ModelReleaseManifest):
        _schema_names(obj.ordered_schema)
        _require(_encode(obj.ordered_schema, True) ==
                 _encode(obj.identity.dataset_spec.ordered_model_schema, True), "Model/Feature schema mismatch")
        _require(obj.model_bytes.artifact_type == "ModelBytes" and
                 obj.fitted_preprocessing_state.artifact_type == "TransformState", "Wrong model payload types")
    if isinstance(obj, c.SignalIdentity):
        inputs = {x.alias: x for x in obj.plan.inputs}
        aliases = [b.alias for b in obj.bindings]
        _require(set(aliases) == set(inputs) and len(set(aliases)) == len(aliases), "Signal binding mismatch")
        for b in obj.bindings:
            _require(_same(b.model.identity.dataset_spec.label, inputs[b.alias].label), "Model label/input mismatch")
            interval = b.model.identity.fold.allowed_prediction_interval
            _require(isinstance(interval, c.SessionRange) and _inside(obj.allowed_dates, interval),
                     "Signal dates outside allowed model interval")
            _require(b.feature_build.artifact_type == "FeatureBuild", "Signal requires FeatureBuild")
    if isinstance(obj, c.SignalRunManifest):
        nodes = {n.name: n for n in obj.identity.plan.nodes}
        _require(obj.output_stage == nodes[obj.identity.plan.output].output_stage, "SignalRun stage mismatch")
        _require(set(obj.time_columns) == {"knowledge_cutoff", "simulated_available_at"}, "Signal time columns required")
        _require(set(obj.validity_columns) == {"valid", "invalid_reason"}, "Signal validity columns required")
        _require(obj.output_bytes.artifact_type == "SignalFrame", "Wrong signal payload type")


def validate(contract: c.Contract, *, require_resolved: bool = False) -> c.Contract:
    _require(type(contract).__name__ in TYPES, "Unknown contract type")
    # The same strict field/type boundary applies to constructed and loaded objects.
    contract = _decode(_encode(contract), type(contract))
    for obj in _walk(contract):
        if isinstance(obj, c.Contract):
            _semantics(obj)
    missing = unresolved(contract)
    if require_resolved or any(isinstance(x, BUILD_TYPES) for x in _walk(contract)):
        _require(not missing, "Unresolved correctness: " + ", ".join(x.unknown_id for x in missing))
    return contract


def to_dict(contract: c.Contract) -> dict:
    return _encode(validate(contract))


def from_dict(value: dict) -> c.Contract:
    _require(type(value) is dict and value.get("contract_type") in TYPES, "Unknown contract type")
    return validate(_decode(value, TYPES[value["contract_type"]]))


def dumps(contract: c.Contract) -> str:
    return json.dumps(to_dict(contract), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def loads(text: str) -> c.Contract:
    try:
        value = json.loads(text, object_pairs_hook=_unique_object,
                           parse_constant=lambda x: (_ for _ in ()).throw(ContractError(f"Nonfinite JSON: {x}")))
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    return from_dict(value)


def load(path: str | Path) -> c.Contract:
    return loads(Path(path).read_text(encoding="utf-8"))


def save(contract: c.Contract, path: str | Path) -> None:
    """Create a new manifest exclusively; never overwrite an existing artifact."""
    payload = dumps(contract) + "\n"
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(payload)


def semantic_identity(contract: c.Contract) -> str:
    """Definition/build-request identity. Does not claim output bytes exist."""
    contract = validate(contract)
    payload = json.dumps(_encode(contract, True), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def content_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def contract_schema(contract_type: type[c.Contract]) -> dict:
    """JSON Schema for transport shape. validate() also enforces cross-field rules."""
    definitions = {}

    def shape(typ):
        origin, args = get_origin(typ), get_args(typ)
        if typ is Any:
            return {}
        if origin in (types.UnionType, Union):
            return {"anyOf": [shape(x) for x in args]}
        if origin is Literal:
            return {"enum": list(args)}
        if origin is tuple:
            return {"type": "array", "items": shape(args[0])}
        if origin is dict:
            return {"type": "object", "additionalProperties": shape(args[1])}
        if isinstance(typ, type) and issubclass(typ, c.Contract):
            name = typ.__name__
            if name not in definitions:
                definitions[name] = {}
                props = {"contract_type": {"const": name},
                         **{k: shape(v) for k, v in get_type_hints(typ).items()}}
                definitions[name] = {"type": "object", "properties": props,
                                     "required": list(props), "additionalProperties": False}
            return {"$ref": f"#/$defs/{name}"}
        return {"type": {str: "string", int: "integer", float: "number", bool: "boolean"}[typ]}

    root = shape(contract_type)
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **root, "$defs": definitions}

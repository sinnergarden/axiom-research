"""Read saved sequential outputs and bindings without training-input expansion."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json

from .stock_artifacts import digest, digest_stream, file_digest
from .result_discovery import read_json, mark
from .experiments import _require, _pairs


@dataclass(frozen=True)
class StockSavedProjection:
    wire: dict
    identity: str
    content_digest: str

    def to_dict(self):
        return deepcopy(self.wire)


def _sealed(value, field):
    _require(value[field] == digest_stream({k: v for k, v in value.items() if k != field}),
             "saved " + field + " mismatch")


def _local(path, root):
    resolved = Path(path).resolve(strict=True)
    _require(resolved.is_relative_to(Path(root).resolve(strict=True)), "nested output escapes Research root")
    return resolved


def _checked(path, expected=None):
    path = Path(path)
    before = mark(path)
    _require(before[2] <= 256 * 1024 * 1024, "saved output exceeds projection read budget")
    payload = path.read_bytes()
    from hashlib import sha256
    _require(mark(path) == before and (expected is None or
             "sha256:" + sha256(payload).hexdigest() == expected), "saved output changed or hash mismatch")
    return json.loads(payload, object_pairs_hook=_pairs, parse_constant=lambda _: (
        _ for _ in ()).throw(ValueError("nonfinite saved output")))


def load_stock_ml_batch_manifest(path):
    """Verify just the saved batch document; never expand its prepared-view parents."""
    value = read_json(path)
    _require(value["contract_version"] in ("stock_ml_batch_inputs_v3",
        "stock_ml_batch_inputs_v4", "stock_ml_batch_inputs_v5") and
        value["status"] == "COMPLETE", "unsupported saved batch")
    _sealed(value, "content_digest")
    _require(value["definition_ref"] == digest(value["definition"]) and
        value["batch_ref"] == digest({k: v for k, v in value.items()
                                     if k not in ("content_digest", "batch_ref")}), "batch identity mismatch")
    return StockSavedProjection(value, value["batch_ref"], value["content_digest"])


def load_stock_prediction_bindings(path):
    """Verify original frame descriptors and the existing Engine logical input identity."""
    value = read_json(path)
    _require(type(value) is list and bool(value), "original prediction bindings required")
    from axiom_engine.runtime.stock_stream_contracts import validate_raw_binding, logical_ref
    seen = set()
    for frame in value:
        _require(frame.get("kind") == "raw", "this projection admits original raw folds")
        validate_raw_binding(frame, modern=True)
        _require(frame["signal_run_ref"] not in seen, "duplicate prediction frame")
        seen.add(frame["signal_run_ref"])
    logical = {"contract_version": "stock_prediction_input_refs_v2", "frames": value}
    reference = logical_ref(logical, "prediction_ref")
    logical["prediction_ref"] = reference
    return StockSavedProjection(logical, reference, digest(value))


def load_stock_fold_projection(path, *, research_root=None, batch_item=None, binding=None):
    """Verify saved fold outputs, not their large training/Feature ancestors.

    This output integrity/clock projection complements the full fold loader; it
    does not re-establish training-input or native-price numerical validity.
    """
    path = Path(path)
    base = path.parent if path.is_file() else path
    root = research_root or base
    base = _local(base, root)
    manifest = read_json(_local(base / "manifest.json", root))
    from .stock_matrix_folds import OUTPUTS, _predictions
    expected = set(OUTPUTS) | {"fold.json", "booster.txt"}
    _require(manifest["contract_version"] == "stock_ml_fold_manifest_v2" and
             set(manifest["files"]) == expected, "unsupported fold output manifest")
    documents = {}
    marks = {}
    for name, expected_digest in manifest["files"].items():
        target = _local(base / name, root); marks[target] = mark(target)
        if name == "booster.txt":
            _require(file_digest(target) == expected_digest, "booster file mismatch")
        else:
            documents[name] = _checked(target, expected_digest)
    fold = documents["fold.json"]; _sealed(fold, "content_digest")
    definition = fold["definition"]
    _require(fold["contract_version"] in ("stock_ml_fold_v3", "stock_ml_fold_v4") and
             fold["status"] == "COMPLETE" and definition["input_manifest_ref"] ==
             digest(definition["input_manifest"]) and definition["fold_spec_ref"] ==
             digest(definition["fold_spec"]) and fold["definition_ref"] == digest(definition),
             "fold definition mismatch")
    refs = {}
    for name, field in OUTPUTS.items():
        value = documents[name]; _sealed(value, field); refs[field] = value[field]
        _require(fold[field] == value[field], "fold output identity mismatch")
    _require(fold["fold_ref"] == manifest["fold_ref"] ==
             digest({"definition_ref": fold["definition_ref"], **refs}), "fold identity mismatch")
    model = documents["model.json"]; features = documents["feature-slice.json"]
    predictions = documents["predictions.json"]
    _require(model["booster_digest"] == manifest["files"]["booster.txt"] and
             model["dataset_ref"] == fold["dataset_ref"] and model["feature_ref"] == fold["feature_ref"],
             "model/output linkage mismatch")
    if "training_spec" in definition:
        from .stock_training import resolve_stock_training_spec
        parameters, rounds, training = resolve_stock_training_spec(definition["training_spec"])
        _require(definition["training_spec_ref"] == training["training_spec_ref"] and
                 definition["parameters"] == parameters and definition["num_boost_round"] == rounds,
                 "saved TrainingSpec/parameters mismatch")
    _require(model["parameters"] == definition["parameters"] and
             model["num_boost_round"] == definition["num_boost_round"],
             "saved model/training definition mismatch")
    if "target_spec" in definition:
        from .stock_target_spec import resolve_stock_label_spec
        _require(definition["target_spec"] == resolve_stock_label_spec(
            definition["target_spec"]["label_spec"]), "saved LabelSpec definition mismatch")
    selection = definition.get("model_feature_selection")
    if selection is not None:
        _require(model["feature_selection"] == selection["selection"] and
                 model["ordered_features"] == selection["ordered_features"],
                 "saved model/Feature selection mismatch")
    common = {"universe": features["universe"], "catalog_ref": features["catalog_ref"]}
    if "target_spec" in definition:
        common["target_spec"] = definition["target_spec"]
    _predictions(predictions, features, common, definition["fold_spec"], model)
    if batch_item is not None:
        _require(batch_item["fold_spec"] == definition["fold_spec"] and
                 batch_item["input_manifest"] == definition["input_manifest"], "fold/batch mismatch")
    if binding is not None:
        _require(all(binding[k] == fold[k] for k in
            ("fold_ref", "model_ref", "feature_ref", "signal_run_ref")) and
            binding["fold_spec_ref"] == definition["fold_spec_ref"], "original frame/fold mismatch")
        for key, name in (("model_metadata_artifact", "model.json"),
                          ("prediction_artifact", "predictions.json")):
            ref = binding[key]
            _require(_local(ref["manifest_uri"], root) == base / name and
                     ref["content_digest"] == digest_stream(documents[name]) and
                     ref["artifact_id"] == fold["model_ref" if name == "model.json" else "signal_run_ref"],
                     "frame output artifact mismatch")
        ref = binding["fold_spec_artifact"]
        _require(ref["manifest_uri"] == str(base / "fold.json") + "#definition/fold_spec" and
                 ref["artifact_id"] == definition["fold_spec_ref"] ==
                 ref["content_digest"], "frame fold-spec mismatch")
    _require(all(mark(p) == m for p, m in marks.items()), "fold changed during projection")
    return StockSavedProjection(fold, fold["fold_ref"], fold["content_digest"])


def load_stock_sequential_result(path, *, research_root=None):
    """Public lightweight A01 saved-result projection bound to batch and frames."""
    path = Path(path).resolve(strict=True); root = research_root or path.parent
    _local(path, root)
    value = read_json(path)
    batch = load_stock_ml_batch_manifest(_local(path.parent / "batch.json", root))
    bindings = load_stock_prediction_bindings(_local(path.parent / "prediction-bindings.json", root))
    _require(value["batch_manifest"] == batch.wire and
             value["prediction_bindings"] == bindings.wire["frames"] and
             len(value["folds"]) == len(batch.wire["folds"]) == len(bindings.wire["frames"]),
             "sequential result/batch/frame mismatch")
    _require(not value["derived_signals"], "Derived sequential projection is not admitted by this profile")
    for saved, item, binding in zip(value["folds"], batch.wire["folds"], bindings.wire["frames"]):
        fold = load_stock_fold_projection(_local(saved["path"], root), research_root=root,
                                          batch_item=item, binding=binding)
        _require(saved["fold_ref"] == fold.identity and saved["signal_run_ref"] ==
                 fold.wire["signal_run_ref"] and saved["fold_spec_ref"] ==
                 fold.wire["definition"]["fold_spec_ref"], "saved sequential fold mismatch")
    identity = digest(value)
    projection = {"contract_version": "stock_sequential_result_projection_v1",
        "result_ref": identity, "configuration_ref": value["configuration_ref"],
        "dataset_configuration_ref": value.get("dataset_configuration_ref"),
        "spec_refs": value["spec_refs"], "batch_ref": batch.identity,
        "batch_content_digest": batch.content_digest, "folds": value["folds"],
        "prediction_input": bindings.wire, "signal_evaluation_input": value["signal_evaluation_input"],
        "limitations": ["Saved output integrity and linkage; training/native-price ancestors not expanded.",
                        "No Data, feature calculation, fit, predict, evaluation or account execution."]}
    return StockSavedProjection(projection, identity, identity)


def load_stock_result_configuration(path):
    value = read_json(path); _sealed(value, "content_digest")
    _require(value["contract_version"] == "stock_result_configuration_v1", "unsupported result configuration")
    definition = value["configuration"]
    from .api import from_dict, semantic_identity
    _require(set(definition["specs"]) == set(definition["spec_refs"]) and
             all(semantic_identity(from_dict(spec)) == definition["spec_refs"][name]
                 for name,spec in definition["specs"].items()), "saved configuration specs mismatch")
    _require(definition["specification_ref"] == digest({
        "contract_version": "stock_effective_configuration_v1", "spec_refs": definition["spec_refs"]}),
        "saved configuration specification mismatch")
    dataset = definition["dataset"]
    expected_dataset = digest(dataset) if definition["compiler_ref"] is None else digest({
        "dataset": dataset, "compiler_ref": definition["compiler_ref"]})
    _require(definition["dataset_ref"] == expected_dataset and
        definition["configuration_ref"] == digest({
            "contract_version": "stock_sequential_effective_configuration_v1" if
                definition["compiler_ref"] is None else "stock_sequential_effective_configuration_v2",
            "specification_ref": definition["specification_ref"], "dataset_ref": expected_dataset}),
        "saved configuration identity mismatch")
    _require(value["result_id"] == digest({"question_id": value["question_id"],
        "version_ref": value["version_ref"], "attempt_id": value["attempt_id"]}), "stable result identity mismatch")
    _require(type(value["required_stages"]) is list and
             set(value["required_stages"]) <= {"training", "signal_evaluation", "accounts", "account_evaluation"} and
             "training" in value["required_stages"] and
             type(value["required_strategy_refs"]) is list and
             len(set(value["required_strategy_refs"])) == len(value["required_strategy_refs"]) and
             ("accounts" not in value["required_stages"] or bool(value["required_strategy_refs"])),
             "invalid required owner stages")
    return StockSavedProjection(value, definition["configuration_ref"], value["content_digest"])


def load_stock_prediction_projection(path, *, research_root=None):
    """Original saved prediction after output-only fold integrity/clock validation."""
    load_stock_fold_projection(Path(path).parent, research_root=research_root)
    value = _checked(path)
    return StockSavedProjection(value, value["signal_run_ref"], digest_stream(value))


def _verify_configuration_outputs(configuration, sequential, batch, folds, root):
    """Compare actual saved definitions, not just a result's configuration claim."""
    from .stock_target_spec import resolve_stock_label_spec
    from .stock_training import resolve_stock_training_spec
    from .stock_compact_store import _model_feature_binding
    from .stock_signal_evaluation_inputs import _scope
    cfg = configuration["configuration"]; dataset = cfg["dataset"]
    definition = batch["definition"]
    options = deepcopy(dataset["preparation_options"])
    # The existing column preparation owner saves this declared default even
    # when it was omitted from the YAML transport.
    options.setdefault("control_layout", "fold_controls_v2")
    _require(sequential["configuration_ref"] == cfg["configuration_ref"] and
             sequential["dataset_configuration_ref"] == cfg["dataset_ref"] and
             sequential["spec_refs"] == cfg["spec_refs"], "result configuration mismatch")
    _require(definition["fold_specs"] == dataset["fold_specs"] and
             [f["fold_spec"] for f in batch["folds"]] == dataset["fold_specs"] and
             definition["preparation_options"] == options,
             "saved batch differs from frozen Dataset")
    target = resolve_stock_label_spec(cfg["specs"]["label"])
    parameters, rounds, training = resolve_stock_training_spec(cfg["specs"]["model"])
    _require(definition["target_spec"] == target and
             target["label_definition_ref"] == cfg["spec_refs"]["label"] and
             training["training_spec_ref"] == cfg["spec_refs"]["model"],
             "saved batch differs from frozen Label/Training definitions")
    view = definition["feature_view"]
    selection = _model_feature_binding({"definition": view}, dataset["model_feature_selection"])
    _require(definition.get("model_feature_selection") == selection and
             all(view["spec"][k] == dataset["scope"][k] for k in ("calendar", "universe")),
             "saved Feature scope/selection differs from frozen Dataset")
    input_ref = sequential["signal_evaluation_input"]
    input_path = _local(input_ref["uri"], root)
    inputs = _checked(input_path, input_ref["content_digest"])
    _require(inputs["input_id"] == input_ref["artifact_id"] and
             inputs["scope"] == _scope(dataset["scope"]), "saved Signal input scope mismatch")
    for saved, item, spec in zip(sequential["folds"], batch["folds"], dataset["fold_specs"]):
        _require(saved["fold_ref"] in folds, "saved fold artifact missing")
        fold = folds[saved["fold_ref"]]; actual = fold["definition"]
        _require(actual["fold_spec"] == spec and actual["input_manifest"] == item["input_manifest"] and
                 actual["target_spec"] == target and actual.get("model_feature_selection") == selection and
                 actual["training_spec"] == training["training_spec"] and
                 actual["training_spec_ref"] == training["training_spec_ref"] and
                 actual["parameters"] == parameters and actual["num_boost_round"] == rounds,
                 "saved fold differs from frozen Dataset/Label/Training/Feature selection")


def _verify_manifest_lineage(manifest, artifacts, verified, sequential, batch, folds):
    """Keep each navigation edge on its actual owner identity and original frame."""
    frames = sequential["prediction_input"]
    def check(keys, fold):
        found = set()
        for key in keys:
            kind = artifacts[key]["artifact_type"]; value = verified[key]
            if kind == "StockMLFoldProjection":
                _require(value["fold_ref"] == fold["fold_ref"], "fold lineage identity mismatch")
            elif kind == "StockMLBatchManifest":
                _require(value == batch, "fold lineage batch mismatch")
            elif kind == "StockPredictionBindings":
                _require(value == frames, "fold lineage original frames mismatch")
            elif kind == "StockSequentialResult":
                _require(value == sequential, "fold lineage sequential result mismatch")
            else:
                raise ValueError("unsupported fold lineage artifact type")
            found.add(kind)
        _require("StockMLFoldProjection" in found, "fold lineage artifact missing")
    for declaration in manifest["folds"]:
        check(declaration["artifact_keys"], folds[declaration["fold_ref"]])
    by_signal = {folds[f["fold_ref"]]["signal_run_ref"]: folds[f["fold_ref"]]
                 for f in sequential["folds"]}
    for signal in manifest["signals"]:
        _require(signal["signal_run_ref"] in by_signal, "Signal original fold missing")
        fold = by_signal[signal["signal_run_ref"]]
        check(signal["lineage_artifact_keys"], fold)
        key = signal["prediction_artifact_key"]
        _require(key is not None and artifacts[key]["artifact_type"] == "StockPredictionRun" and
                 all(verified[key][k] == fold[k] for k in ("signal_run_ref", "model_ref", "feature_ref")) and
                 verified[key]["fold_spec_ref"] == fold["definition"]["fold_spec_ref"],
                 "Signal prediction/fold lineage mismatch")


def _saved_account_status(wire):
    state = wire["status"]; stopped = wire["stopped"]
    _require(state in ("COMPLETE", "BLOCKED") and
             (state == "COMPLETE") == (stopped is None), "saved account status/stopped mismatch")
    if stopped is None:
        return state, None
    _require(type(stopped) is dict and type(stopped.get("reason")) is str and
             bool(stopped["reason"]), "saved account stopped reason required")
    return state, stopped["reason"]


def verify_result_artifacts(manifest, artifacts, paths, roots):
    """Fixed owner loaders; no loader names or code are read from manifests."""
    verified = {}
    for key, ref in artifacts.items():
        path = paths[key]; kind = ref["artifact_type"]
        if ref["owner"] == "Research":
            if kind == "StockResultConfiguration": saved = load_stock_result_configuration(path)
            elif kind == "StockSequentialResult": saved = load_stock_sequential_result(path, research_root=roots["research"])
            elif kind == "StockMLBatchManifest": saved = load_stock_ml_batch_manifest(path)
            elif kind == "StockPredictionBindings": saved = load_stock_prediction_bindings(path)
            elif kind == "StockMLFoldProjection": saved = load_stock_fold_projection(path, research_root=roots["research"])
            elif kind == "StockPredictionRun":
                saved = load_stock_prediction_projection(path, research_root=roots["research"])
            elif kind == "StockSignalEvaluation":
                from .stock_signal_evaluation import load_stock_signal_evaluation
                obj = load_stock_signal_evaluation(path, research_root=roots["research"]); value = obj.to_dict()
                saved = StockSavedProjection(value, obj.identity, value["content_digest"])
            else: raise ValueError("unsupported Research result artifact type")
            _require(saved.wire["contract_version"] == ref["artifact_contract_version"] and
                     saved.identity == ref["artifact_id"] and saved.content_digest == ref["content_digest"],
                     "Research artifact identity mismatch")
            verified[key] = saved.wire
        else:
            from axiom_engine.runtime import load_stock_backtest_projection, load_backtest_evaluation
            if kind == "BacktestRun":
                from .result_discovery import resolve
                def reader(nested):
                    candidate = Path(nested["manifest_uri"]).resolve(strict=True)
                    root = Path(roots["engine"]).resolve(strict=True)
                    _require(candidate.is_relative_to(root), "Engine nested artifact escapes root")
                    return candidate
                from .result_registration import STOCK_LIMITS
                run = load_stock_backtest_projection(path, artifact_reader=reader, limits=STOCK_LIMITS)
                value = run.verify().wire
                _require(value["run_id"] == ref["artifact_id"], "Engine run identity mismatch")
            elif kind == "BacktestEvaluation":
                value = load_backtest_evaluation(path).to_dict()
                _require(value["evaluation_ref"] == ref["artifact_id"], "Engine evaluation identity mismatch")
            else: raise ValueError("unsupported Engine result artifact type")
            _require(value["contract_version"] == ref["artifact_contract_version"] and
                     value["content_digest"] == ref["content_digest"], "Engine artifact contract/digest mismatch")
            verified[key] = value
    configuration = verified[manifest["configuration_ref"]]
    _require(all(configuration[k] == manifest[k] for k in ("result_id", "question_id", "version_ref")),
             "result/configuration identity mismatch")
    sequential = [v for k, v in verified.items() if artifacts[k]["artifact_type"] == "StockSequentialResult"]
    folds = {v["fold_ref"]: v for k, v in verified.items()
             if artifacts[k]["artifact_type"] == "StockMLFoldProjection"}
    _require(len(sequential) <= 1 and (sequential or not (manifest["folds"] or manifest["signals"])),
             "one saved sequential result required for fold/Signal declarations")
    for value in sequential:
        batches = [v for k, v in verified.items() if artifacts[k]["artifact_type"] == "StockMLBatchManifest" and
                   v["batch_ref"] == value["batch_ref"] and v["content_digest"] == value["batch_content_digest"]]
        _require(bool(batches), "original sequential batch artifact missing")
        _verify_configuration_outputs(configuration, value, batches[0], folds, roots["research"])
        _require({f["fold_ref"] for f in value["folds"]} == {f["fold_ref"] for f in manifest["folds"]} and
                 {f["signal_run_ref"] for f in value["folds"]} ==
                 {s["signal_run_ref"] for s in manifest["signals"]}, "result fold/Signal declaration mismatch")
        _verify_manifest_lineage(manifest, artifacts, verified, value, batches[0], folds)
    if manifest["status"] == "COMPLETE":
        _require(len(sequential) == 1 and
                 len(manifest["folds"]) == len(configuration["configuration"]["dataset"]["fold_specs"]),
                 "declared training stage is incomplete")
        if "signal_evaluation" in configuration["required_stages"]:
            _require(all(s["evaluation_artifact_keys"] for s in manifest["signals"]),
                     "required Signal evaluation missing")
        if "accounts" in configuration["required_stages"]:
            _require(set(configuration["required_strategy_refs"]) <=
                     {a["strategy_ref"] for a in manifest["accounts"]},
                     "required strategy accounts missing")
        if "account_evaluation" in configuration["required_stages"]:
            _require(manifest["accounts"] and all(a["evaluation_refs"] for a in manifest["accounts"]),
                     "required account evaluations missing")
    frame_sets = [v["prediction_input"] for v in sequential]
    for account in manifest["accounts"]:
        backtest = account["backtest_ref"]
        if backtest is None: continue
        run = verified[backtest["artifact_key"]]
        _require((account["status"], account["reason"]) == _saved_account_status(run),
                 "account declaration differs from saved Engine status/reason")
        _require(run["committed_sequence"] == backtest["committed_sequence"] and
                 run["signal_ref"] == account["signal_run_ref"], "account run/signal binding mismatch")
        _require(digest(run["request_manifest"]["portfolio_policy"]) == account["strategy_ref"],
                 "strategy policy reference mismatch")
        _require(any(run["request_manifest"]["prediction_input"] == frames for frames in frame_sets),
                 "account must consume the original saved prediction frames")
        for evaluation in account["evaluation_refs"]:
            report = verified[evaluation["artifact_key"]]
            _require(report["input_run_ref"] == evaluation["input_run_ref"], "evaluation run binding mismatch")
    for signal in manifest["signals"]:
        for key in signal["evaluation_artifact_keys"]:
            _require(signal["signal_run_ref"] in verified[key]["input_signal_refs"],
                     "Signal evaluation belongs to different inputs")

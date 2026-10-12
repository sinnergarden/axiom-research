"""Producer registration after owner saves; no execution or directory discovery."""
from copy import deepcopy
from pathlib import Path
import uuid

from .experiments import ExperimentStore, ExperimentReader, _require, _now
from .stock_artifacts import digest, write_json
from .stock_result_projection import (load_stock_result_configuration,
    load_stock_sequential_result, load_stock_fold_projection, load_stock_prediction_bindings,
    _saved_account_status)

STOCK_LIMITS = dict(max_folds=10, max_prediction_rows=20000, max_market_rows=25000,
    max_input_bytes=64*1024*1024, max_read_bytes=64*1024, max_block_bytes=64*1024*1024,
    max_result_part_bytes=8*1024*1024, max_result_buffer_bytes=64*1024*1024,
    max_result_bytes=512*1024*1024)


def _seal(value):
    value = deepcopy(value); value["content_digest"] = digest(value)
    return value


def _immutable(path, value):
    import os
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation retains a previously frozen attempt across retries.
    payload = __import__("json").dumps(value, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False) + "\n"
    try:
        with path.open("x") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    except FileExistsError:
        _require(path.read_text() == payload, "conflicting frozen result request")


def _save_stock_sequential_receipt(result, *, destination, metrics=None):
    """Save a complete immutable sibling group before publishing its index ref."""
    import os
    import shutil
    import tempfile
    from .api import to_dict
    from .result_discovery import read_json
    saved = deepcopy(result)
    for name in ("raw_signal_evaluation_input", "signal_evaluation_input"):
        if type(saved[name]) is not dict:
            saved[name] = to_dict(saved[name])
    if metrics is not None:
        saved["metrics"] = deepcopy(metrics)
    files = {"experiment-result.json": saved, "batch.json": saved["batch_manifest"],
             "prediction-bindings.json": saved["prediction_bindings"]}
    parent = Path(destination).resolve() / "saved-sequential"
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / digest(saved)[7:]
    def check():
        _require(target.is_dir() and not target.is_symlink() and
                 {p.name for p in target.iterdir()} == set(files) and
                 all(not (target/name).is_symlink() and read_json(target/name) == value
                     for name, value in files.items()), "immutable sequential receipt conflict")
    if target.exists():
        check()
    else:
        stage = Path(tempfile.mkdtemp(prefix=".receipt-", dir=parent))
        try:
            for name, value in files.items():
                _immutable(stage/name, value)
            try:
                os.rename(stage, target)
            except OSError:
                if not target.exists():
                    raise
            check()
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    return target / "experiment-result.json"


def _registered_sequential_path(request_path, *, index_path, roots, exists=True):
    frozen = load_stock_result_configuration(request_path).wire
    reader = ExperimentReader(index_path)
    if not any(h["result_id"] == frozen["result_id"] for h in reader.index().get("result_heads", [])):
        return None
    manifest = reader.result(frozen["result_id"], roots=roots)
    refs = [a for a in manifest["artifacts"] if a["artifact_type"] == "StockSequentialResult"]
    if not refs:
        return None
    _require(len(refs) == 1, "one registered sequential receipt required")
    from .result_discovery import resolve
    ref = refs[0]
    return resolve(ref["root_id"], ref["relative_path"], roots, exists=exists)


def _record_stock_result_failure(request_path, *, index_path, roots, failure):
    """A failed retry must keep an already published saved result consumable."""
    if _registered_sequential_path(request_path, index_path=index_path, roots=roots, exists=False) is None:
        return publish_stock_result(request_path, index_path=index_path, roots=roots, failure=failure)
    return None


def begin_stock_result_registration(configuration, **options):
    """Lock request creation so concurrent retries share the same frozen attempt."""
    import fcntl
    path = Path(options["request_path"])
    root = Path(options["roots"]["research"]).resolve(strict=True)
    _require(path.resolve().is_relative_to(root), "result request escapes configured root")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name+".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _begin_stock_result_registration(configuration, **options)


def _begin_stock_result_registration(configuration, *, index_path, roots, request_path,
        question=None, version=None, required_stages=("training", "signal_evaluation"),
        strategy_refs=(), imported=False):
    """Freeze one attempt before execution; historical imports are explicitly marked."""
    from .result_discovery import read_json
    request_path = Path(request_path)
    cfgref = configuration["configuration_ref"]
    if request_path.exists():
        saved = load_stock_result_configuration(request_path).wire
        _require(saved["configuration"]["configuration_ref"] == cfgref and
                 saved["required_stages"] == list(required_stages) and
                 saved["required_strategy_refs"] == list(strategy_refs),
                 "retry request configuration changed")
        return saved
    writer = ExperimentStore(index_path)
    from .api import to_dict
    specs={name:to_dict(spec) for name,spec in configuration["specification"].get("specs",{}).items()}
    question = question or dict(question_id="stock-" + cfgref[7:23],
        title="Stock sequential experiment", description="Configuration-driven saved experiment",
        hypothesis="")
    q = writer.create_question(**question)
    version = version or dict(label="fixed configuration", explanation="Frozen owner configuration",
        parameters={"configuration_ref": cfgref, "dataset_ref": configuration["dataset_ref"],
                    "spec_refs": configuration["specification"]["spec_refs"],"specs":specs,
                    "preparation_options":configuration["dataset"].get("preparation_options"),
                    "model_feature_selection":configuration["dataset"].get("model_feature_selection")},
        input_refs=[], explicit_changes=[])
    v = writer.create_version(question_id=q["question_id"], **version)
    _require(set(required_stages) <= {"training", "signal_evaluation", "accounts", "account_evaluation"} and
             ("training" in required_stages or "signal" in specs) and
             ("accounts" in required_stages or not strategy_refs), "unsupported required stages")
    attempt = uuid.uuid4().hex
    frozen = _seal({"contract_version": "stock_result_configuration_v1",
        "question_id": q["question_id"], "version_ref": v["version_ref"], "attempt_id": attempt,
        "result_id": digest({"question_id": q["question_id"], "version_ref": v["version_ref"],
                             "attempt_id": attempt}),
        "created_at": _now(), "registration_mode": "existing_saved_import" if imported else "pre_execution",
        "required_stages": list(required_stages), "required_strategy_refs": list(strategy_refs),
        "configuration": {"configuration_ref": cfgref,
            "specification_ref": configuration["specification"]["configuration_ref"],
            "spec_refs": configuration["specification"]["spec_refs"],
            "specs":specs,
            "dataset_ref": configuration["dataset_ref"], "dataset": configuration["dataset"],
            "compiler_ref": (configuration.get("periodic_compilation") or
                             configuration.get("weekly_compilation") or {}).get("compiler_ref")}})
    _immutable(request_path, frozen)
    load_stock_result_configuration(request_path)
    return frozen


def publish_stock_result(request_path, *, index_path, roots, experiment_result=None,
        derived_signals=(), signal_evaluation=None, engine_handoff=None, failure=None):
    """Append a manifest revision referencing verified outputs in place.

    No fixed producer directory is scanned. All supplied paths are owner save
    receipts, or the existing bound result's immutable prior typed references.
    """
    from .result_discovery import read_json
    from .stock_result_projection import load_stock_ml_batch_manifest
    frozen = load_stock_result_configuration(request_path).wire
    derived_signals = list(derived_signals)
    store = ExperimentStore(index_path); reader = ExperimentReader(index_path)
    previous = None
    if Path(index_path).exists():
        head = next((h for h in reader.index().get("result_heads", [])
                     if h["result_id"] == frozen["result_id"]), None)
        if head is not None:
            previous = reader.result(frozen["result_id"], roots=roots)
    manifest = {k: frozen[k] for k in ("result_id", "question_id", "version_ref")}
    manifest.update(contract_version="experiment_result_manifest_v1",
        revision=0 if previous is None else previous["revision"] + 1,
        created_at=_now(), status="RUNNING", reason="Declared owner stages not yet saved",
        configuration_ref="configuration", artifacts=[], folds=[], signals=[], accounts=[])
    if previous is not None:
        for key in ("artifacts", "folds", "signals", "accounts"):
            manifest[key] = deepcopy(previous[key])
    def artifact(key, owner, kind, identity, version, content_digest, path):
        root_id = owner.lower(); root = Path(roots[root_id]).resolve(strict=True)
        resolved = Path(path).resolve(strict=True)
        _require(resolved.is_relative_to(root), "saved artifact outside configured root")
        value = dict(artifact_key=key, owner=owner, artifact_type=kind, artifact_id=identity,
            artifact_contract_version=version, content_digest=content_digest, root_id=root_id,
            relative_path=resolved.relative_to(root).as_posix())
        manifest["artifacts"] = [a for a in manifest["artifacts"] if a["artifact_key"] != key] + [value]
    artifact("configuration", "Research", "StockResultConfiguration",
        frozen["configuration"]["configuration_ref"], frozen["contract_version"],
        frozen["content_digest"], request_path)
    if experiment_result is not None:
        saved = load_stock_sequential_result(experiment_result, research_root=roots["research"])
        result = saved.to_dict()
        artifact("sequential", "Research", "StockSequentialResult", saved.identity,
            "stock_sequential_result_projection_v1", saved.content_digest, experiment_result)
        base = Path(experiment_result).parent
        batch = load_stock_ml_batch_manifest(base / "batch.json")
        artifact("batch", "Research", "StockMLBatchManifest", batch.identity,
            batch.wire["contract_version"], batch.content_digest, base / "batch.json")
        frames = load_stock_prediction_bindings(base / "prediction-bindings.json", research_root=roots["research"])
        artifact("bindings", "Research", "StockPredictionBindings", frames.identity,
            "stock_prediction_input_refs_v2", frames.content_digest, base / "prediction-bindings.json")
        manifest["folds"], manifest["signals"] = [], []
        for item, frame in zip(result["folds"], frames.wire["frames"]):
            key = "fold-" + item["fold_ref"][7:]
            fold = load_stock_fold_projection(item["path"], research_root=roots["research"])
            artifact(key, "Research", "StockMLFoldProjection", fold.identity,
                fold.wire["contract_version"], fold.content_digest, Path(item["path"]) / "fold.json")
            manifest["folds"].append(dict(fold_ref=fold.identity, status="COMPLETE", reason=None,
                                          artifact_keys=[key, "batch", "bindings"]))
            if frame.get("kind") == "derived":
                parents = [p for p in frame["parent_inputs"].values() if p["fold_ref"] == item["fold_ref"]]
                _require(len(parents) == 1, "sequential Derived original fold missing")
                frame = parents[0]
            prediction = frame["prediction_artifact"]; signal_key = "signal-" + frame["signal_run_ref"][7:]
            artifact(signal_key, "Research", "StockPredictionRun", frame["signal_run_ref"],
                prediction["contract_version"], prediction["content_digest"], prediction["manifest_uri"])
            old = next((s for s in (previous or {}).get("signals", [])
                        if s["signal_run_ref"] == frame["signal_run_ref"]), None)
            manifest["signals"].append(dict(signal_run_ref=frame["signal_run_ref"], status="COMPLETE",
                reason=None, prediction_artifact_key=signal_key, lineage_artifact_keys=[key, "bindings"],
                evaluation_artifact_keys=[] if old is None else old["evaluation_artifact_keys"]))
        derived_signals.extend(item["path"] for item in result.get("derived_signals", []))
    for path in derived_signals:
        from .stock_derived_signal import load_stock_derived_signal
        saved = load_stock_derived_signal(path, research_root=roots["research"])
        wire = saved.to_dict(); key = "signal-" + saved.identity[7:]
        artifact(key, "Research", "StockDerivedSignal", saved.identity,
                 wire["contract_version"], saved.content_digest, saved.path / "signal.json")
        old = next((s for s in manifest["signals"] if s["signal_run_ref"] == saved.identity), None)
        if old is None:
            old = next((s for s in (previous or {}).get("signals", []) if s["signal_run_ref"] == saved.identity), None)
        parent_folds = {p["fold_ref"] for p in saved.engine_input_binding()["parent_inputs"].values()}
        lineage = ["configuration"] + ["fold-" + f["fold_ref"][7:] for f in manifest["folds"]
                                      if f["fold_ref"] in parent_folds]
        signal = dict(signal_run_ref=saved.identity, status="COMPLETE", reason=None,
            prediction_artifact_key=key, lineage_artifact_keys=lineage,
            evaluation_artifact_keys=[] if old is None else old["evaluation_artifact_keys"])
        if any(s["signal_run_ref"] == saved.identity for s in manifest["signals"]):
            manifest["signals"] = [signal if s["signal_run_ref"] == saved.identity else s for s in manifest["signals"]]
        else:
            manifest["signals"].append(signal)
    if signal_evaluation is not None:
        from .stock_signal_evaluation import load_stock_signal_evaluation
        result = read_json(signal_evaluation)
        items=list(result["statistics"].values())
        for group in result.get("by_year",{}).values():
            items.extend(group.values())
        for item in items:
            saved = load_stock_signal_evaluation(item["path"], research_root=roots["research"]); wire = saved.to_dict()
            key = "signal-evaluation-" + saved.identity[7:]
            artifact(key, "Research", "StockSignalEvaluation", saved.identity,
                wire["contract_version"], wire["content_digest"], item["path"])
            for signal in manifest["signals"]:
                if signal["signal_run_ref"] in wire["input_signal_refs"] and key not in signal["evaluation_artifact_keys"]:
                    signal["evaluation_artifact_keys"].append(key)
    if engine_handoff is not None:
        handoff = engine_handoff if type(engine_handoff) is dict else read_json(engine_handoff)
        for item in handoff["accounts"]:
            from axiom_engine.runtime import load_stock_backtest_projection
            def nested(ref):
                p = Path(ref["manifest_uri"]).resolve(strict=True)
                _require(p.is_relative_to(Path(roots["engine"]).resolve(strict=True)), "Engine nested root escape")
                return p
            run = load_stock_backtest_projection(item["run_manifest"], artifact_reader=nested, limits=STOCK_LIMITS)
            wire = run.verify().wire; strategy_ref = digest(wire["request_manifest"]["portfolio_policy"])
            _require(run.input_run_ref == item["run_ref"], "handoff run mismatch")
            key = "account-" + wire["run_id"][7:]
            artifact(key, "Engine", "BacktestRun", wire["run_id"], wire["contract_version"],
                     wire["content_digest"], item["run_manifest"])
            old = next((a for a in manifest["accounts"] if a["strategy_ref"] == strategy_ref and
                        a["signal_run_ref"] == wire["signal_ref"]), None)
            state, reason = _saved_account_status(wire)
            account = dict(strategy_ref=strategy_ref, signal_run_ref=wire["signal_ref"], status=state,
                reason=reason, backtest_ref={"artifact_key": key, **run.input_run_ref},
                evaluation_refs=[] if old is None else deepcopy(old["evaluation_refs"]))
            del run
            for report in item["evaluations"].values():
                from axiom_engine.runtime import load_backtest_evaluation
                evaluation = load_backtest_evaluation(report["path"]).to_dict()
                _require(evaluation["input_run_ref"] == item["run_ref"], "handoff evaluation mismatch")
                ekey = "account-evaluation-" + evaluation["evaluation_ref"][7:]
                artifact(ekey, "Engine", "BacktestEvaluation", evaluation["evaluation_ref"],
                    evaluation["contract_version"], evaluation["content_digest"], report["path"])
                if not any(e["evaluation_ref"] == evaluation["evaluation_ref"] for e in account["evaluation_refs"]):
                    account["evaluation_refs"].append(dict(artifact_key=ekey,
                        evaluation_ref=evaluation["evaluation_ref"], content_digest=evaluation["content_digest"],
                        input_run_ref=evaluation["input_run_ref"]))
            manifest["accounts"] = [a for a in manifest["accounts"] if
                (a["strategy_ref"], a["signal_run_ref"]) != (strategy_ref, wire["signal_ref"])] + [account]
    complete = bool(manifest["signals"]) and ("training" not in frozen["required_stages"] or
        bool(manifest["folds"])) and all(
        row["status"] == "COMPLETE" for table in ("folds", "signals", "accounts") for row in manifest[table])
    if "signal_evaluation" in frozen["required_stages"]:
        complete &= all(s["evaluation_artifact_keys"] for s in manifest["signals"])
    if "accounts" in frozen["required_stages"]:
        complete &= set(frozen["required_strategy_refs"]) <= {a["strategy_ref"] for a in manifest["accounts"]}
    if "account_evaluation" in frozen["required_stages"]:
        complete &= bool(manifest["accounts"]) and all(a["evaluation_refs"] for a in manifest["accounts"])
    if complete:
        manifest.update(status="COMPLETE", reason=None)
    elif manifest["folds"] or manifest["signals"]:
        manifest.update(status="PARTIAL", reason="Required owner stages are not all saved")
    blocked = next((a for a in manifest["accounts"] if a["status"] == "BLOCKED"), None)
    if blocked is not None:
        manifest.update(status="BLOCKED", reason="Saved Engine account is BLOCKED: " + blocked["reason"])
    if failure is not None:
        manifest.update(status="FAILED", reason=str(failure))
    manifest["artifacts"].sort(key=lambda a: a["artifact_key"])
    manifest["accounts"].sort(key=lambda a: (a["strategy_ref"], a["signal_run_ref"]))
    for signal in manifest["signals"]:
        signal["evaluation_artifact_keys"].sort()
    for account in manifest["accounts"]:
        account["evaluation_refs"].sort(key=lambda e: e["evaluation_ref"])
    # Repeating an unchanged save keeps both the head and immutable revision.
    if previous is not None:
        semantic = lambda m: {k:v for k,v in m.items() if k not in ("revision","created_at","content_digest")}
        if semantic(previous) == semantic(manifest):
            return store.publish_result(previous, roots=roots)
    return store.publish_result(_seal(manifest), roots=roots)


def attach_stock_account(request_path, *, index_path, roots, run_path, evaluation_paths=()):
    """Pipeline save hook over existing owner loaders, not a manual registry."""
    from axiom_engine.runtime import load_stock_backtest_projection, load_backtest_evaluation
    def reader(ref):
        path = Path(ref["manifest_uri"]).resolve(strict=True)
        _require(path.is_relative_to(Path(roots["engine"]).resolve(strict=True)), "Engine nested root escape")
        return path
    run = load_stock_backtest_projection(run_path, artifact_reader=reader, limits=STOCK_LIMITS)
    run_ref = run.input_run_ref
    reports = {}
    for path in evaluation_paths:
        report = load_backtest_evaluation(path).to_dict()
        _require(report["input_run_ref"] == run_ref, "account evaluation mismatch")
        reports[report["evaluation_ref"]] = {"path": str(path)}
    del run
    return publish_stock_result(request_path, index_path=index_path, roots=roots,
        engine_handoff={"accounts": [{"run_manifest": str(run_path), "run_ref": run_ref,
                                     "evaluations": reports}]})

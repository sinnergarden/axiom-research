"""Small saved-result manifests over the sole ExperimentStore index."""
from copy import deepcopy
from pathlib import Path
import os
import tempfile

from .experiments import (_require, _fields, _hash, _text, _integer, _timestamp,
                          _digest, _pairs, ExperimentRecordError)

STATUSES = {"RUNNING", "PARTIAL", "COMPLETE", "FAILED", "BLOCKED"}
HEAD_FIELDS = "result_id question_id version_ref revision manifest_ref status reason updated_at".split()
ARTIFACT_FIELDS = ("artifact_key owner artifact_type artifact_id artifact_contract_version "
                   "content_digest root_id relative_path").split()
MANIFEST_FIELDS = ("contract_version result_id revision question_id version_ref created_at "
                   "status reason configuration_ref artifacts folds signals accounts content_digest").split()


def status(value):
    _require(value["status"] in STATUSES, "invalid result status")
    if value["status"] != "COMPLETE":
        _text(value["reason"], "incomplete result reason")
    elif value["reason"] is not None:
        _text(value["reason"], "result reason")


def locator(value):
    _fields(value, ("root_id", "relative_path", "content_digest"), "manifest locator")
    _hash(value["content_digest"], "manifest content_digest")
    relative_path(value["root_id"], value["relative_path"])


def relative_path(root_id, relative):
    _require(root_id in ("research", "engine"), "unknown trusted root_id")
    _require(type(relative) is str and relative and "\\" not in relative and ":" not in relative
             and not relative.startswith("/") and
             all(p not in ("", ".", "..") for p in relative.split("/")),
             "invalid root-relative POSIX path")


def resolve(root_id, relative, roots, *, exists=True):
    relative_path(root_id, relative)
    _require(root_id in roots, "missing trusted root")
    root = Path(roots[root_id]).resolve(strict=True)
    _require(root.is_dir(), "trusted root must be a directory")
    target = (root / relative).resolve(strict=exists)
    _require(target.is_relative_to(root), "artifact escapes trusted root")
    if exists:
        _require(target.is_file(), "artifact must be a file")
    return target


def read_json(path, maximum=2 * 1024 * 1024):
    import json
    path = Path(path)
    _require(path.stat().st_size <= maximum, "saved metadata exceeds byte budget")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    _require(len(data) <= maximum, "saved metadata grew beyond byte budget")
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=lambda _: (
        _ for _ in ()).throw(ExperimentRecordError("nonfinite saved metadata")))


def validate_head(head):
    _fields(head, HEAD_FIELDS, "result head")
    for key in ("result_id", "version_ref"):
        _hash(head[key], key)
    _text(head["question_id"], "question_id")
    _integer(head["revision"], "revision")
    _timestamp(head["updated_at"])
    status(head)
    locator(head["manifest_ref"])
    _require(head["manifest_ref"]["root_id"] == "research", "manifest belongs to Research")


def validate_manifest(value):
    _fields(value, MANIFEST_FIELDS, "result manifest")
    _require(value["contract_version"] == "experiment_result_manifest_v1", "unknown result manifest")
    for key in ("result_id", "version_ref", "content_digest"):
        _hash(value[key], key)
    _require(value["content_digest"] == _digest({k: v for k, v in value.items()
                                                if k != "content_digest"}), "manifest digest mismatch")
    _integer(value["revision"], "revision")
    _text(value["question_id"], "question_id")
    _timestamp(value["created_at"])
    status(value)
    _require(type(value["artifacts"]) is list, "typed artifacts must be an array")
    artifacts = {}
    for artifact in value["artifacts"]:
        _fields(artifact, ARTIFACT_FIELDS, "typed artifact")
        _text(artifact["artifact_key"], "artifact_key")
        _require(artifact["artifact_key"] not in artifacts, "duplicate artifact_key")
        _require(artifact["owner"] in ("Research", "Engine") and
                 artifact["root_id"] == artifact["owner"].lower(), "artifact owner/root mismatch")
        for key in ("artifact_id", "content_digest"):
            _hash(artifact[key], key)
        for key in ("artifact_type", "artifact_contract_version"):
            _text(artifact[key], key)
        relative_path(artifact["root_id"], artifact["relative_path"])
        artifacts[artifact["artifact_key"]] = artifact
    def key(name):
        _require(name in artifacts, "unresolved artifact key")
        return artifacts[name]
    _require(key(value["configuration_ref"])["owner"] == "Research", "Research configuration required")
    for table in ("folds", "signals", "accounts"):
        _require(type(value[table]) is list, table + " must be an array")
    seen_folds, seen_signals, seen_accounts = set(), set(), set()
    for fold in value["folds"]:
        _fields(fold, ("fold_ref", "status", "reason", "artifact_keys"), "fold")
        _hash(fold["fold_ref"], "fold_ref")
        _require(fold["fold_ref"] not in seen_folds, "duplicate fold")
        seen_folds.add(fold["fold_ref"]); status(fold)
        _require(type(fold["artifact_keys"]) is list and
                 len(set(fold["artifact_keys"])) == len(fold["artifact_keys"]), "duplicate fold artifact")
        for name in fold["artifact_keys"]:
            key(name)
    for signal in value["signals"]:
        _fields(signal, ("signal_run_ref", "status", "reason", "prediction_artifact_key",
                         "lineage_artifact_keys", "evaluation_artifact_keys"), "signal")
        _hash(signal["signal_run_ref"], "signal_run_ref")
        _require(signal["signal_run_ref"] not in seen_signals, "duplicate signal")
        seen_signals.add(signal["signal_run_ref"]); status(signal)
        if signal["prediction_artifact_key"] is not None:
            _require(key(signal["prediction_artifact_key"])["artifact_id"] == signal["signal_run_ref"],
                     "Signal prediction identity mismatch")
        else:
            _require(signal["status"] != "COMPLETE", "complete Signal requires prediction")
        for group in ("lineage_artifact_keys", "evaluation_artifact_keys"):
            _require(type(signal[group]) is list and len(set(signal[group])) == len(signal[group]),
                     "duplicate Signal artifact")
            for name in signal[group]:
                key(name)
    for account in value["accounts"]:
        _fields(account, ("strategy_ref", "signal_run_ref", "status", "reason", "backtest_ref",
                          "evaluation_refs"), "strategy account")
        _hash(account["strategy_ref"], "strategy_ref")
        _hash(account["signal_run_ref"], "account Signal input reference")
        identity = account["strategy_ref"], account["signal_run_ref"]
        _require(identity not in seen_accounts, "duplicate strategy account")
        seen_accounts.add(identity); status(account)
        backtest = account["backtest_ref"]
        if backtest is not None:
            _fields(backtest, ("artifact_key", "run_id", "content_digest", "committed_sequence"), "backtest_ref")
            ref = key(backtest["artifact_key"])
            _require(ref["owner"] == "Engine" and ref["artifact_type"] == "BacktestRun" and
                     ref["artifact_id"] == backtest["run_id"] and
                     ref["content_digest"] == backtest["content_digest"], "Backtest artifact mismatch")
            _integer(backtest["committed_sequence"], "committed_sequence")
        else:
            _require(account["status"] != "COMPLETE", "complete account requires a saved run")
        _require(type(account["evaluation_refs"]) is list, "evaluations must be an array")
        seen = set()
        for evaluation in account["evaluation_refs"]:
            _fields(evaluation, ("artifact_key", "evaluation_ref", "content_digest", "input_run_ref"),
                    "account evaluation")
            ref = key(evaluation["artifact_key"])
            _require(evaluation["evaluation_ref"] not in seen, "duplicate saved evaluation")
            seen.add(evaluation["evaluation_ref"])
            _require(backtest is not None and evaluation["input_run_ref"] ==
                     {k: backtest[k] for k in ("run_id", "content_digest", "committed_sequence")},
                     "evaluation belongs to a different account")
            _require(ref["owner"] == "Engine" and ref["artifact_type"] == "BacktestEvaluation" and
                     ref["artifact_id"] == evaluation["evaluation_ref"] and
                     ref["content_digest"] == evaluation["content_digest"], "Evaluation artifact mismatch")
    if value["status"] == "COMPLETE":
        _require(value["signals"] and
                 all(item["status"] == "COMPLETE" for table in ("folds", "signals", "accounts")
                     for item in value[table]), "COMPLETE requires all declared outputs")
    return artifacts


def mark(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def publish(store, manifest, roots):
    """Owner validation precedes the original index lock and atomic replacement."""
    manifest = deepcopy(manifest)
    artifacts = validate_manifest(manifest)
    from .stock_result_projection import verify_result_artifacts
    paths = {key: resolve(ref["root_id"], ref["relative_path"], roots)
             for key, ref in artifacts.items()}
    marks = {path: mark(path) for path in paths.values()}
    verify_result_artifacts(manifest, artifacts, paths, roots)
    relative = "results/" + manifest["result_id"][7:] + "/" + (
        str(manifest["revision"]) + "-" + manifest["content_digest"][7:] + ".json")
    path = resolve("research", relative, roots, exists=False)
    head = {k: manifest[k] for k in ("result_id", "question_id", "version_ref", "revision", "status", "reason")}
    head.update(updated_at=manifest["created_at"], manifest_ref={
        "root_id": "research", "relative_path": relative, "content_digest": manifest["content_digest"]})
    def operation(state):
        _require(manifest["question_id"] in state["questions"] and
                 manifest["version_ref"] in state["versions"] and
                 state["versions"][manifest["version_ref"]]["question_id"] == manifest["question_id"],
                 "result must belong to an existing question/version")
        for source, expected in marks.items():
            _require(mark(source) == expected, "artifact changed during publication")
        heads = state.get("result_heads", {})
        old = heads.get(manifest["result_id"])
        if old is not None:
            _require((old["question_id"], old["version_ref"]) ==
                     (head["question_id"], head["version_ref"]), "stable result changed ownership")
            if old["revision"] == head["revision"]:
                _require(old == head, "conflicting result revision")
                _require(read_json(path) == manifest, "saved manifest changed")
                return old, False
            _require(head["revision"] == old["revision"] + 1, "stale or skipped result revision")
        else:
            _require(head["revision"] == 0, "first result revision must be zero")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            _require(read_json(path) == manifest, "immutable manifest conflict")
        else:
            from .stock_artifacts import write_json
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                write_json(temporary, manifest)
                with temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        state["contract_version"] = "experiment_index_v2"
        state.setdefault("result_heads", {})[manifest["result_id"]] = head
        return head, True
    return store._mutate(operation)


def result(reader, result_id, roots):
    from .experiments import _read
    _hash(result_id, "result_id")
    state = _read(reader.path)
    head = state.get("result_heads", {}).get(result_id)
    _require(head is not None, "unknown result_id")
    manifest = read_json(resolve(head["manifest_ref"]["root_id"],
                                 head["manifest_ref"]["relative_path"], roots))
    artifacts = validate_manifest(manifest)
    _require(all(head[k] == manifest[k] for k in
                 ("result_id", "revision", "question_id", "version_ref", "status", "reason")) and
             head["manifest_ref"]["content_digest"] == manifest["content_digest"] and
             head["updated_at"] == manifest["created_at"], "head/manifest mismatch")
    for ref in artifacts.values():
        # A frozen manifest remains readable after one owner leaf is lost.
        # Resolve existing symlinks for containment; existence and owner hash
        # checks belong to each artifact loader, allowing UI item isolation.
        resolve(ref["root_id"], ref["relative_path"], roots, exists=False)
    return manifest

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


def read_json(path, maximum=64 * 1024 * 1024):
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
    from .stock_result_projection import _projection_session
    # The final check belongs inside the index transaction. A change after a
    # successful commit must not turn that committed publication into failure.
    with _projection_session(check_on_exit=False) as session:
        return _publish(store, manifest, roots, session)


def _publish(store, manifest, roots, session):
    manifest = deepcopy(manifest)
    artifacts = validate_manifest(manifest)
    from .stock_result_projection import verify_result_artifacts
    paths = {key: resolve(ref["root_id"], ref["relative_path"], roots)
             for key, ref in artifacts.items()}
    marks = {path: mark(path) for path in paths.values()}
    verify_result_artifacts(manifest, artifacts, paths, roots)
    return _commit_result(store, manifest, roots, session, artifacts, paths, marks)


def _commit_result(store, manifest, roots, session, artifacts, paths, marks, *, compare=None):
    """The common locked, atomic commit after the selected admission contract."""
    relative = "results/" + manifest["result_id"][7:] + "/" + (
        str(manifest["revision"]) + "-" + manifest["content_digest"][7:] + ".json")
    path = resolve("research", relative, roots, exists=False)
    head = {k: manifest[k] for k in ("result_id", "question_id", "version_ref", "revision", "status", "reason")}
    head.update(updated_at=manifest["created_at"], manifest_ref={
        "root_id": "research", "relative_path": relative, "content_digest": manifest["content_digest"]})
    def operation(state):
        if compare is not None:
            compare(state)
        _require(manifest["question_id"] in state["questions"] and
                 manifest["version_ref"] in state["versions"] and
                 state["versions"][manifest["version_ref"]]["question_id"] == manifest["question_id"],
                 "result must belong to an existing question/version")
        # Include every nested owner leaf read while assembling this publish,
        # not merely the small typed navigation documents in the manifest.
        session.check()
        for key, ref in artifacts.items():
            _require(resolve(ref["root_id"], ref["relative_path"], roots) == paths[key],
                     "artifact locator changed during publication")
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


def _registered_account_basis(reader, basis, roots):
    """Read a pinned registered revision; only an immediate exact retry may be ahead."""
    from .experiments import _read
    from .stock_result_projection import _metadata, _local
    _fields(basis, ("result_id", "revision", "manifest_ref", "content_digest"), "account basis")
    _hash(basis["result_id"], "basis result_id")
    _integer(basis["revision"], "basis revision")
    _hash(basis["content_digest"], "basis content_digest")
    locator(basis["manifest_ref"])
    _require(basis["manifest_ref"]["root_id"] == "research" and
             basis["manifest_ref"]["content_digest"] == basis["content_digest"],
             "basis manifest identity mismatch")
    path = resolve("research", basis["manifest_ref"]["relative_path"], roots)
    manifest = _metadata(_local(path, roots["research"]))
    validate_manifest(manifest)
    _require(all(manifest[k] == basis[k] for k in ("result_id", "revision", "content_digest")),
             "basis manifest identity mismatch")
    fixed_head = {k: manifest[k] for k in
                  ("result_id", "question_id", "version_ref", "revision", "status", "reason")}
    fixed_head.update(updated_at=manifest["created_at"], manifest_ref=deepcopy(basis["manifest_ref"]))
    current = _read(reader.path).get("result_heads", {}).get(basis["result_id"])
    _require(current == fixed_head or (current is not None and
             current["revision"] == basis["revision"] + 1 and
             (current["question_id"], current["version_ref"]) ==
             (fixed_head["question_id"], fixed_head["version_ref"])), "registered basis conflict")
    return manifest, fixed_head, current


def _publish_registered_account(store, basis, roots, *, run_path, evaluation_paths):
    """Inherit registered Research identities and admit one new saved Engine account."""
    from .experiments import ExperimentReader, _now
    from .stock_result_projection import _projection_session, _project_registered_account
    with _projection_session(check_on_exit=False) as session:
        previous, fixed_head, expected_head = _registered_account_basis(
            ExperimentReader(store.path), basis, roots)
        account, additions, configuration = _project_registered_account(
            previous, roots, run_path=run_path, evaluation_paths=evaluation_paths)
        _require(not any((a["strategy_ref"], a["signal_run_ref"]) ==
                         (account["strategy_ref"], account["signal_run_ref"])
                         for a in previous["accounts"]), "basis already contains this strategy account")
        manifest = deepcopy(previous)
        inherited_keys = {a["artifact_key"] for a in previous["artifacts"]}
        _require(not inherited_keys.intersection(a["artifact_key"] for a in additions),
                 "new account artifact collides with basis")
        manifest["artifacts"].extend(additions)
        manifest["artifacts"].sort(key=lambda a: a["artifact_key"])
        manifest["accounts"].append(account)
        manifest["accounts"].sort(key=lambda a: (a["strategy_ref"], a["signal_run_ref"]))
        from .result_registration import _stock_result_status
        _stock_result_status(manifest, configuration)
        manifest.update(revision=previous["revision"] + 1, created_at=_now())
        manifest.pop("content_digest")
        from .stock_artifacts import digest
        manifest["content_digest"] = digest(manifest)
        if expected_head != fixed_head:
            current = ExperimentReader(store.path).result(basis["result_id"], roots=roots)
            semantic = lambda m: {k: v for k, v in m.items()
                                  if k not in ("created_at", "content_digest")}
            _require(semantic(current) == semantic(manifest), "registered basis conflict")
            manifest = current
        validate_manifest(manifest)
        # Only new owner outputs and the small basis/binding documents were read.
        # Their nested leaves are retained by the session until the locked commit.
        new_artifacts = {a["artifact_key"]: a for a in additions}
        paths = {k: resolve(a["root_id"], a["relative_path"], roots) for k, a in new_artifacts.items()}
        marks = {p: mark(p) for p in paths.values()}
        def compare(state):
            _require(state.get("result_heads", {}).get(basis["result_id"]) == expected_head,
                     "registered basis/head conflict")
            # Re-read the fixed small manifest inside the lock as well as checking
            # the observed locators/metadata of every newly validated owner leaf.
            pinned = read_json(resolve("research", basis["manifest_ref"]["relative_path"], roots))
            _require(pinned == previous, "registered basis changed during publication")
        return _commit_result(store, manifest, roots, session, new_artifacts, paths, marks, compare=compare)


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

"""Small Research-owned records and read-only projections of saved references.

This module never opens artifact directories or executes Data/Core/Engine.
Writers supply owner-verified references; consumers use each owner's loader
when opening the actual artifact. Organizational history is separate from
immutable questions, versions and run records.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import os
import tempfile

from .api import ContractError, content_digest, from_dict, semantic_identity, to_dict
from .contracts import ArtifactRef


class ExperimentRecordError(ContractError):
    """Invalid metadata, identity or cross-record association."""


class RevisionConflict(ExperimentRecordError):
    """Organizational state changed since the caller's last read."""


def _require(condition, message):
    if not condition:
        raise ExperimentRecordError(message)


def _text(value, name, *, empty=False):
    _require(type(value) is str and (empty or bool(value.strip())), f"{name}: expected text")
    return value


def _integer(value, name):
    _require(type(value) is int and value >= 0, f"{name}: expected nonnegative integer")


def _hash(value, name):
    _require(type(value) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", value),
             f"{name}: expected sha256 digest")


def _json(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        _require(float("-inf") < value < float("inf"), "Nonfinite JSON value")
        return value
    if type(value) is list:
        return [_json(v) for v in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: _json(v) for k, v in value.items()}
    raise ExperimentRecordError("Expected finite JSON values and string object keys")


def _digest(value):
    return content_digest(json.dumps(_json(value), ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode())


def _fields(value, expected, name):
    _require(type(value) is dict and set(value) == set(expected), f"{name}: wrong fields")


def _refs(values):
    _require(type(values) in (list, tuple), "Artifact refs must be an array")
    result = []
    identities = set()
    for value in values:
        try:
            ref = value if isinstance(value, ArtifactRef) else from_dict(value)
            _require(isinstance(ref, ArtifactRef), "Expected ArtifactRef")
            _hash(ref.content_digest, "ArtifactRef.content_digest")
            identity = semantic_identity(ref)
            _require(identity not in identities, "Duplicate ArtifactRef")
            identities.add(identity)
            result.append(to_dict(ref))
        except ContractError as exc:
            raise ExperimentRecordError(str(exc)) from exc
    return result


def _semantic(record):
    value = deepcopy(record)
    value.pop("created_at", None)
    identity_field = ("run_record_ref" if "output_refs" in value else
                      "version_ref" if "input_refs" in value else "question_ref")
    value.pop(identity_field, None)
    for key in ("input_refs", "output_refs"):
        if key in value:
            value[key] = [semantic_identity(from_dict(ref)) for ref in value[key]]
    for key in ("backtest_ref", "evaluation_ref"):
        if value.get(key) is not None:
            _require(type(value[key]) is dict, f"{key}: expected object/null")
            value[key].pop("uri", None)
    return value


def _now():
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value):
    _text(value, "created_at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExperimentRecordError("Invalid created_at") from exc
    _require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
             "created_at must be UTC")


def _organization(groups, tags, favorite, shelved, revision):
    result = {"revision": revision, "groups": _texts(groups), "tags": _texts(tags),
              "favorite": favorite, "shelved": shelved}
    _validate_organization(result)
    return result


def _texts(values):
    _require(type(values) in (list, tuple), "Expected text array")
    return sorted({_text(v, "label") for v in values})


def _validate_organization(value):
    _fields(value, ("revision", "groups", "tags", "favorite", "shelved"), "organization")
    _integer(value["revision"], "organization.revision")
    for key in ("groups", "tags"):
        _require(type(value[key]) is list and value[key] == _texts(value[key]),
                 f"organization.{key}: expected sorted unique labels")
    for key in ("favorite", "shelved"):
        _require(type(value[key]) is bool, f"organization.{key}: expected bool")


def _validate_backtest(value, output_refs):
    if value is None:
        return
    _fields(value, ("run_id", "content_digest", "signal_ref", "committed_sequence", "uri"),
            "backtest_ref")
    for key in ("run_id", "content_digest", "signal_ref"):
        _hash(value[key], f"backtest_ref.{key}")
    _integer(value["committed_sequence"], "backtest_ref.committed_sequence")
    _text(value["uri"], "backtest_ref.uri")
    _require(any(ref["artifact_type"] == "SignalRun" and
                 ref["artifact_id"] == value["signal_ref"] for ref in output_refs),
             "Backtest signal_ref must match an output SignalRun")


def _validate_evaluation(value, backtest):
    if value is None:
        return
    _require(backtest is not None, "Evaluation requires an associated BacktestRun")
    _fields(value, ("evaluation_ref", "evaluation_content_digest", "input_run_ref", "uri"),
            "evaluation_ref")
    for key in ("evaluation_ref", "evaluation_content_digest"):
        _hash(value[key], key)
    _text(value["uri"], "evaluation_ref.uri")
    _fields(value["input_run_ref"], ("run_id", "content_digest", "committed_sequence"),
            "evaluation_ref.input_run_ref")
    for key in ("run_id", "content_digest"):
        _hash(value["input_run_ref"][key], f"input_run_ref.{key}")
    _integer(value["input_run_ref"]["committed_sequence"], "input_run_ref.committed_sequence")
    _require(value["input_run_ref"] == {key: backtest[key] for key in
             ("run_id", "content_digest", "committed_sequence")},
             "Evaluation input_run_ref does not match the BacktestRun")


def _validate_record(record, kind):
    common = ("question_id", "created_at")
    specific = {
        "question_ref": ("question_ref", "title", "description", "hypothesis"),
        "version_ref": ("version_ref", "parent_version_ref", "label", "explanation",
                        "parameters", "input_refs", "explicit_changes"),
        "run_record_ref": ("run_record_ref", "version_ref", "status", "reason", "outcome",
                           "output_refs", "backtest_ref", "evaluation_ref"),
    }
    _fields(record, common + specific[kind], kind)
    _text(record["question_id"], "question_id")
    _timestamp(record["created_at"])
    _hash(record[kind], kind)
    _require(record[kind] == _digest(_semantic(record)), f"{kind}: identity mismatch")
    if kind == "question_ref":
        _text(record["title"], "title")
        _text(record["description"], "description", empty=True)
        _text(record["hypothesis"], "hypothesis", empty=True)
    elif kind == "version_ref":
        if record["parent_version_ref"] is not None:
            _hash(record["parent_version_ref"], "parent_version_ref")
        for key in ("label", "explanation"):
            _text(record[key], key)
        _require(type(record["parameters"]) is dict, "parameters must be an object")
        _json(record["parameters"])
        _require(record["input_refs"] == _refs(record["input_refs"]), "Invalid input_refs")
        _require(type(record["explicit_changes"]) is list, "explicit_changes must be an array")
        for change in record["explicit_changes"]:
            _text(change, "explicit_change")
    else:
        _hash(record["version_ref"], "version_ref")
        _require(record["status"] in ("COMPLETE", "FAILED", "BLOCKED"), "Invalid run status")
        if record["status"] != "COMPLETE" or record["reason"] is not None:
            _text(record["reason"], "reason")
        if record["outcome"] is not None:
            _text(record["outcome"], "outcome")
        _require(record["output_refs"] == _refs(record["output_refs"]), "Invalid output_refs")
        _require(record["status"] != "COMPLETE" or bool(record["output_refs"]),
                 "Complete run requires an output reference")
        _validate_backtest(record["backtest_ref"], record["output_refs"])
        _validate_evaluation(record["evaluation_ref"], record["backtest_ref"])


def _empty():
    return {"contract_version": "experiment_index_v1", "store_revision": 0,
            "questions": {}, "versions": {}, "runs": {}, "organizations": {}}


def _new_record(value, kind):
    value["created_at"] = _now()
    value[kind] = _digest(_semantic(value))
    _validate_record(value, kind)
    return value


def _validate(state):
    _fields(state, (*_empty(), "content_digest"), "experiment index")
    _require(state["contract_version"] == "experiment_index_v1", "Unknown index contract")
    _integer(state["store_revision"], "store_revision")
    _hash(state["content_digest"], "index.content_digest")
    _require(state["content_digest"] == _digest({k: v for k, v in state.items()
             if k != "content_digest"}), "Index content_digest mismatch")
    for table, kind in (("questions", "question_ref"), ("versions", "version_ref"),
                        ("runs", "run_record_ref")):
        _require(type(state[table]) is dict, f"{table}: expected object")
        for key, value in state[table].items():
            _validate_record(value, kind)
            _require(key == value["question_id" if table == "questions" else kind],
                     f"{table}: key mismatch")
            _require(value["question_id"] in state["questions"], "Unknown question")
    _require(type(state["organizations"]) is dict and
             set(state["organizations"]) == set(state["questions"]),
             "Organization/question keys mismatch")
    for history in state["organizations"].values():
        _require(type(history) is list and bool(history), "Missing organizational history")
        for revision, value in enumerate(history):
            _validate_organization(value)
            _require(value["revision"] == revision, "Organization revision gap")
    for version in state["versions"].values():
        parent = version["parent_version_ref"]
        if parent is not None:
            _require(parent in state["versions"] and
                     state["versions"][parent]["question_id"] == version["question_id"],
                     "Version parent must belong to the same question")
        seen = {version["version_ref"]}
        while parent is not None:
            _require(parent not in seen, "Cyclic version parents")
            seen.add(parent)
            parent = state["versions"][parent]["parent_version_ref"]
    for run in state["runs"].values():
        _require(run["version_ref"] in state["versions"] and
                 state["versions"][run["version_ref"]]["question_id"] == run["question_id"],
                 "Run version must belong to the same question")
    return state


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def _read(path):
    try:
        state = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               ExperimentRecordError("Nonfinite JSON constant")))
    except (ValueError, TypeError) as exc:
        raise ExperimentRecordError(str(exc)) from exc
    return _validate(state)


class ExperimentStore:
    """The explicit writer; each operation locks and atomically replaces one JSON."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _mutate(self, operation):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_name(self.path.name + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = _read(self.path) if self.path.exists() else _empty()
            result, changed = operation(state)
            if changed:
                state["store_revision"] += 1
                state["content_digest"] = _digest({k: v for k, v in state.items()
                                                  if k != "content_digest"})
                _validate(state)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                            dir=self.path.parent, prefix="." + self.path.name + ".",
                            delete=False) as stream:
                        temporary = Path(stream.name)
                        json.dump(state, stream, ensure_ascii=False, sort_keys=True,
                                  indent=2, allow_nan=False)
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, self.path)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            return deepcopy(result)

    def create_question(self, *, question_id, title, description, hypothesis):
        record = _new_record(dict(question_id=question_id, title=title, description=description,
                                  hypothesis=hypothesis), "question_ref")
        def operation(state):
            old = state["questions"].get(question_id)
            if old is not None:
                _require(old["question_ref"] == record["question_ref"],
                         "question_id already names different immutable content")
                return old, False
            state["questions"][question_id] = record
            state["organizations"][question_id] = [_organization([], [], False, False, 0)]
            return record, True
        return self._mutate(operation)

    def _insert(self, record, table, kind):
        record = _new_record(record, kind)
        def operation(state):
            _require(record["question_id"] in state["questions"], "Unknown question")
            old = state[table].get(record[kind])
            if old is not None:
                return old, False
            state[table][record[kind]] = record
            return record, True
        return self._mutate(operation)

    def create_version(self, *, question_id, label, explanation, parameters, input_refs,
                       parent_version_ref=None, explicit_changes=()):
        _require(type(explicit_changes) in (list, tuple), "explicit_changes must be an array")
        return self._insert(dict(question_id=question_id, parent_version_ref=parent_version_ref,
            label=label, explanation=explanation, parameters=_json(parameters),
            input_refs=_refs(input_refs), explicit_changes=list(explicit_changes)),
            "versions", "version_ref")

    def record_run(self, *, question_id, version_ref, status, output_refs, reason=None,
                   outcome=None, backtest_ref=None, evaluation_ref=None):
        return self._insert(dict(question_id=question_id, version_ref=version_ref, status=status,
            reason=reason, outcome=outcome, output_refs=_refs(output_refs),
            backtest_ref=_json(backtest_ref), evaluation_ref=_json(evaluation_ref)),
            "runs", "run_record_ref")

    def register_saved_experiment(self, *, question, version, run):
        """Atomically register all three records using the single-record schemas.

        question supplies create_question arguments; version supplies create_version
        arguments except question_id; run supplies record_run arguments except
        question_id/version_ref. Defaults are the same as their individual APIs.
        """
        _fields(question, ("question_id", "title", "description", "hypothesis"), "question")
        _require(type(version) is dict and type(run) is dict, "Expected version/run objects")
        version = {"parent_version_ref": None, "explicit_changes": [], **version}
        run = {"reason": None, "outcome": None, "backtest_ref": None, "evaluation_ref": None, **run}
        _fields(version, ("parent_version_ref", "label", "explanation", "parameters",
                          "input_refs", "explicit_changes"), "version")
        _fields(run, ("status", "reason", "outcome", "output_refs", "backtest_ref",
                      "evaluation_ref"), "run")
        _require(type(version["explicit_changes"]) in (list, tuple),
                 "explicit_changes must be an array")
        question = _new_record(deepcopy(question), "question_ref")
        version = _new_record({**version, "question_id": question["question_id"],
            "parameters": _json(version["parameters"]), "input_refs": _refs(version["input_refs"]),
            "explicit_changes": list(version["explicit_changes"])}, "version_ref")
        run = _new_record({**run, "question_id": question["question_id"],
            "version_ref": version["version_ref"], "output_refs": _refs(run["output_refs"]),
            "backtest_ref": _json(run["backtest_ref"]),
            "evaluation_ref": _json(run["evaluation_ref"])}, "run_record_ref")
        def operation(state):
            changed = False
            qid = question["question_id"]
            old = state["questions"].get(qid)
            if old is not None:
                _require(old["question_ref"] == question["question_ref"],
                         "question_id already names different immutable content")
            else:
                state["questions"][qid] = question
                state["organizations"][qid] = [_organization([], [], False, False, 0)]
                changed = True
            for record, table, kind in ((version, "versions", "version_ref"),
                                       (run, "runs", "run_record_ref")):
                if record[kind] not in state[table]:
                    state[table][record[kind]] = record
                    changed = True
            return {"question": state["questions"][qid],
                    "version": state["versions"][version["version_ref"]],
                    "run": state["runs"][run["run_record_ref"]]}, changed
        return self._mutate(operation)

    def update_organization(self, question_id, *, expected_revision, groups, tags,
                            favorite, shelved):
        _integer(expected_revision, "expected_revision")
        record = _organization(groups, tags, favorite, shelved, expected_revision + 1)
        def operation(state):
            _require(question_id in state["organizations"], "Unknown question")
            history = state["organizations"][question_id]
            if history[-1]["revision"] != expected_revision:
                raise RevisionConflict("Stale organization revision")
            history.append(record)
            return record, True
        return self._mutate(operation)


class ExperimentReader:
    """Only reads the index metadata; construction has no filesystem effects."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def index(self, *, group=None, tags=(), status=None, favorite=None, shelved=None,
              question_id=None, version_ref=None):
        for name, value in (("group", group), ("question_id", question_id)):
            if value is not None:
                _text(value, name)
        if version_ref is not None:
            _hash(version_ref, "version_ref")
        _require(status is None or status in ("COMPLETE", "FAILED", "BLOCKED"),
                 "Invalid status filter")
        for name, value in (("favorite", favorite), ("shelved", shelved)):
            _require(value is None or type(value) is bool, f"{name}: expected bool/null")
        tags = set(_texts(tags))
        state = _read(self.path)
        entries = []
        for question in state["questions"].values():
            qid = question["question_id"]
            organization = state["organizations"][qid][-1]
            if (question_id is not None and qid != question_id or
                group is not None and group not in organization["groups"] or
                not tags.issubset(organization["tags"]) or
                favorite is not None and favorite != organization["favorite"] or
                shelved is not None and shelved != organization["shelved"]):
                continue
            versions = [v for v in state["versions"].values() if v["question_id"] == qid and
                        (version_ref is None or v["version_ref"] == version_ref)]
            if version_ref is not None and not versions:
                continue
            runs = [r for r in state["runs"].values() if r["question_id"] == qid and
                    (version_ref is None or r["version_ref"] == version_ref) and
                    (status is None or r["status"] == status)]
            if status is not None and not runs:
                continue
            entries.append(dict(question=question, organization=organization,
                versions=sorted(versions, key=lambda v: (v["created_at"], v["version_ref"])),
                runs=sorted(runs, key=lambda r: (r["created_at"], r["run_record_ref"]))))
        entries.sort(key=lambda e: (e["question"]["created_at"], e["question"]["question_id"]),
                     reverse=True)
        return {"contract_version": "experiment_projection_v1",
                "store_revision": state["store_revision"],
                "content_digest": state["content_digest"], "questions": entries}

    def detail(self, question_id):
        result = self.index(question_id=question_id)
        _require(bool(result["questions"]), "Unknown question")
        return result

    def compare_versions(self, left_version_ref, right_version_ref):
        state = _read(self.path)
        _require(left_version_ref in state["versions"] and right_version_ref in state["versions"],
                 "Unknown version")
        left, right = (state["versions"][ref] for ref in (left_version_ref, right_version_ref))
        _require(left["question_id"] == right["question_id"], "Cannot compare unrelated questions")
        return {"contract_version": "experiment_version_diff_v1",
            "store_revision": state["store_revision"], "content_digest": state["content_digest"],
            "question_id": left["question_id"], "left_version_ref": left_version_ref,
            "right_version_ref": right_version_ref,
            "left_explanation": left["explanation"], "right_explanation": right["explanation"],
            "explicit_changes": deepcopy(right["explicit_changes"]),
            "parameter_changes": _diff(left["parameters"], right["parameters"]),
            "input_changes": _diff(_reference_values(left["input_refs"]),
                                   _reference_values(right["input_refs"]))}


def _reference_values(refs):
    return [{k: v for k, v in ref.items() if k not in ("uri", "metadata")} for ref in refs]


def _diff(left, right, path=""):
    """Presence flags distinguish a missing field from an explicit JSON null."""
    if type(left) is dict and type(right) is dict:
        changes = []
        for key in sorted(left.keys() | right.keys()):
            child = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key in left and key in right:
                changes.extend(_diff(left[key], right[key], child))
            else:
                changes.append({"path": child, "before_present": key in left,
                    "after_present": key in right, "before": left.get(key), "after": right.get(key)})
        return changes
    if type(left) is list and type(right) is list and len(left) == len(right):
        return [change for i, (before, after) in enumerate(zip(left, right))
                for change in _diff(before, after, path + "/" + str(i))]
    if type(left) is type(right) and left == right:
        return []
    return [{"path": path, "before_present": True, "after_present": True,
             "before": left, "after": right}]

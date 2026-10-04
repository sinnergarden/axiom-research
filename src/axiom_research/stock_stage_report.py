"""Saved-only stock stage projections, independent of build runtimes."""
from __future__ import annotations
from datetime import date, datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .stock_artifacts import _read, digest, file_digest, load_stock_ml_experiment

REPORT_VERSION = "axiom.stock_stage_report/1"
_INPUT_KEYS = ("experiment_ref", "feature_ref", "dataset_ref", "model_ref", "signal_run_ref", "evidence_ref")
_STAGES = {"feature": "feature_seconds", "qlib": "qlib_seconds", "label_dataset": "label_dataset_seconds",
           "train": "train_seconds", "predict": "predict_seconds", "build": "build_seconds", "total": "total_seconds"}
_EXECUTION_COUNTS = {"feature": "feature_core_calls", "qlib": "qlib_calls",
                     "train": "train_calls", "predict": "predict_calls"}


def _instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("stage report requires aware timestamps")
    return result


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("stage report requires nonnegative integer counts")
    return value


def _seconds(value):
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
        raise ValueError("stage report requires finite nonnegative seconds")
    return value


def _training(config, dataset):
    fit = _instant(config["fit_cutoff"])
    if _instant(dataset["fit_cutoff"]) != fit:
        raise ValueError("stage report fit cutoff mismatch")
    declared = sorted(s for s in config["feature_sessions"] if _instant(config["cutoff_by_session"][s]) <= fit)
    if len(set(config["feature_sessions"])) != len(config["feature_sessions"]):
        raise ValueError("duplicate declared feature session")
    keys = dataset["training_keys"]
    if any(type(k) is not list or len(k) != 2 or any(type(x) is not str or not x for x in k) for k in keys):
        raise ValueError("invalid saved training keys")
    if len({tuple(k) for k in keys}) != len(keys):
        raise ValueError("duplicate saved training key")
    sessions = sorted({k[1] for k in keys})
    for session in declared + sessions:
        if date.fromisoformat(session).isoformat() != session:
            raise ValueError("invalid training session")
    if not set(sessions).issubset(declared):
        raise ValueError("actual training key outside declared cutoff window")
    if _count(dataset["training_row_count"]) != len(keys):
        raise ValueError("saved training row count mismatch")
    excluded = {k: _count(v) for k, v in dataset["excluded"].items()}
    def window(values):
        return {"first_feature_session": values[0] if values else None,
                "last_feature_session": values[-1] if values else None, "session_count": len(values)}
    return {"fit_cutoff": config["fit_cutoff"], "declared": window(declared),
            "actual": {**window(sessions), "training_row_count": len(keys)}, "excluded": excluded,
            "actual_source": "dataset.training_keys",
            "declared_rule": "feature-session knowledge_cutoff <= fit_cutoff (aware instant)"}


def _signal_summary(predictions, evidence):
    _instant(evidence["evaluation_cutoff"])
    series = evidence["series"]
    if len({r["session"] for r in series}) != len(series):
        raise ValueError("duplicate evidence session")
    def statistic(field):
        values = [r[field] for r in series if r[field] is not None]
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
            raise ValueError("nonfinite saved signal statistic")
        return {"session_count": len(values), "mean": math.fsum(values) / len(values) if values else None}
    rows = predictions["rows"]
    if len({(r["security_id"], r["session"]) for r in rows}) != len(rows):
        raise ValueError("duplicate saved prediction key")
    groups = {}
    for row in rows:
        if date.fromisoformat(row["session"]).isoformat() != row["session"]:
            raise ValueError("noncanonical prediction session")
        if type(row["valid"]) is not bool:
            raise ValueError("invalid saved prediction validity")
        groups.setdefault(row["session"], []).append(row)
        for field in ("knowledge_cutoff", "available_at"):
            if row.get(field) is not None:
                _instant(row[field])
        score = row.get("score")
        if score is not None and (type(score) not in (int, float) or not math.isfinite(score)):
            raise ValueError("nonfinite saved prediction score")
    if evidence["score_semantics"] != predictions["score_semantics"]:
        raise ValueError("saved signal score semantics mismatch")
    if {r["session"] for r in series} != set(groups):
        raise ValueError("evidence and prediction session set mismatch")
    for row in series:
        if date.fromisoformat(row["session"]).isoformat() != row["session"]:
            raise ValueError("noncanonical evidence session")
        actual_valid = sum(r["valid"] for r in groups[row["session"]])
        valid = _count(row["valid_pair_count"])
        excluded = _count(row["excluded_pair_count"])
        if (_count(row["prediction_valid_count"]) != actual_valid or valid > actual_valid or
                valid + excluded != len(groups[row["session"]])):
            raise ValueError("saved signal per-session count mismatch")
    return {"evaluation_cutoff": evidence["evaluation_cutoff"], "score_semantics": evidence["score_semantics"],
            "label_semantics": evidence["label_semantics"], "minimum_pairs": _count(evidence["minimum_pairs"]),
            "rank_ties": evidence["rank_ties"], "weighting": "equal_valid_session",
            "missing_policy": "exclude_null_no_fill", "ic": statistic("ic"), "rank_ic": statistic("rank_ic"),
            "evidence_session_count": len(series), "prediction_row_count": len(rows),
            "prediction_valid_row_count": sum(r["valid"] for r in rows),
            "prediction_invalid_row_count": sum(not r["valid"] for r in rows),
            "valid_pair_count": sum(_count(r["valid_pair_count"]) for r in series),
            "excluded_pair_count": sum(_count(r["excluded_pair_count"]) for r in series)}


def _receipt_snapshot(path):
    """Parse and hash exactly one byte snapshot, retaining strict JSON rules."""
    data = Path(path).read_bytes()
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    receipt = json.loads(data.decode("utf-8"), object_pairs_hook=unique,
                         parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if type(receipt) is not dict:
        raise ValueError("timing receipt requires a JSON object")
    return receipt, "sha256:" + sha256(data).hexdigest()


def _measurements(paths, experiment):
    inputs = {k: experiment[k] for k in _INPUT_KEYS}
    reuse = experiment["definition"].get("input_reuse") or {}
    measurements, receipts = [], []
    for path in paths:
        receipt, receipt_digest = _receipt_snapshot(path)
        if receipt.get("created_at") is not None:
            _instant(receipt["created_at"])
        nested = receipt.get("experiment")
        if nested is not None:
            if type(nested) is not dict or any(nested.get(k) != inputs[k] for k in _INPUT_KEYS):
                raise ValueError("timing receipt current experiment reference mismatch")
            source, feature, inherited = nested["experiment_ref"], nested["feature_ref"], False
            mode = "saved_input_build" if reuse else "cold_build"
        else:
            source = receipt.get("experiment_ref")
            inherited = source != inputs["experiment_ref"]
            if not source or (inherited and (source != reuse.get("experiment_ref") or reuse.get("feature_ref") != inputs["feature_ref"])):
                raise ValueError("timing receipt input_reuse reference mismatch")
            if receipt.get("feature_ref", inputs["feature_ref"]) != inputs["feature_ref"]:
                raise ValueError("timing receipt feature reference mismatch")
            feature = inputs["feature_ref"]
            mode = "saved_input_build" if reuse and not inherited else "cold_build"
        receipts.append({"file_digest": receipt_digest, "source_experiment_ref": source, "feature_ref": feature,
                         "binding": "declared_input_reuse" if inherited else "current_experiment"})
        metrics = receipt.get("metrics", {})
        if type(metrics) is not dict:
            raise ValueError("invalid owner timing metrics")
        if metrics.get("cache_hit") is True:
            raise ValueError("cache-hit receipt cannot supply build measurements; use separate cache_reuse metrics")
        size = {k: _count(v) for k, v in metrics.items() if k.endswith("_rows") or k in (
            "training_rows", "valid_predictions", "artifact_bytes", "peak_rss_bytes")}
        peak = receipt.get("peak_rss_bytes")
        if peak is not None:
            _count(peak)
        observation = {"scope": "receipt_observation_not_stage_allocation", "metrics": size, "peak_rss_bytes": peak}
        for stage, field in _STAGES.items():
            if inherited and stage not in ("feature", "qlib"):
                continue
            seconds = _seconds(metrics.get(field))
            skipped = stage in ("feature", "qlib") and metrics.get("feature_cache_hit") is True
            if skipped and seconds not in (None, 0):
                raise ValueError("reused stage has nonzero execution seconds")
            count_field = _EXECUTION_COUNTS.get(stage)
            if count_field in metrics:
                calls = _count(metrics[count_field])
                if skipped and calls:
                    raise ValueError("reused stage contradicts execution count")
                if not skipped and seconds == 0 and calls == 0:
                    raise ValueError("ambiguous zero-duration unexecuted stage without cache flags")
            measurements.append({"stage": stage, "mode": mode,
                "status": "REUSED_NOT_EXECUTED" if skipped else "NOT_PROVIDED" if seconds is None else "MEASURED",
                "seconds": None if skipped else seconds, "reported_seconds": seconds, "metric_path": "metrics." + field,
                "receipt_file_digest": receipt_digest, "source_experiment_ref": source, "feature_ref": feature,
                "inherited_feature_only": inherited, "observation": observation})
        if not inherited:
            cache = receipt.get("cache_reuse", {})
            if type(cache) is not dict:
                raise ValueError("invalid cache timing receipt")
            seconds = _seconds(cache.get("seconds"))
            measurements.append({"stage": "cache_load", "mode": "cache_reuse",
                "status": "MEASURED" if seconds is not None else "NOT_PROVIDED", "seconds": seconds,
                "reported_seconds": seconds, "metric_path": "cache_reuse.seconds", "receipt_file_digest": receipt_digest,
                "source_experiment_ref": source, "feature_ref": feature, "inherited_feature_only": False,
                "observation": {"scope": "cache_receipt", "metrics": cache}})
    current = {r["stage"] for r in measurements if not r["inherited_feature_only"]}
    for stage in (*_STAGES, "cache_load"):
        if stage not in current:
            measurements.append({"stage": stage, "mode": None, "status": "NOT_PROVIDED", "seconds": None,
                "reported_seconds": None, "metric_path": None, "receipt_file_digest": None,
                "source_experiment_ref": inputs["experiment_ref"], "feature_ref": inputs["feature_ref"],
                "inherited_feature_only": False, "observation": None})
    return measurements, receipts


def _identity(value):
    return digest({k: value[k] for k in ("contract_version", "report_version", "implementation_ref", "input_refs", "source_input_reuse", "timing_receipts")})


def _verify_closure(value):
    refs = value["input_refs"]
    if set(refs) != set(_INPUT_KEYS):
        raise ValueError("invalid stage report input references")
    for ref in [*refs.values(), value["implementation_ref"], *(r["file_digest"] for r in value["timing_receipts"])]:
        if type(ref) is not str or not re.fullmatch(r"sha256:[0-9a-f]{64}", ref):
            raise ValueError("invalid stage report digest reference")
    receipts = {r["file_digest"]: r for r in value["timing_receipts"]}
    reuse = value["source_input_reuse"] or {}
    for receipt in receipts.values():
        if receipt["binding"] not in ("declared_input_reuse", "current_experiment"):
            raise ValueError("invalid timing receipt binding")
        if receipt["feature_ref"] != refs["feature_ref"]:
            raise ValueError("stage report receipt feature mismatch")
        if receipt["binding"] == "declared_input_reuse" and (receipt["source_experiment_ref"] != reuse.get("experiment_ref") or
                receipt["feature_ref"] != reuse.get("feature_ref")):
            raise ValueError("stage report declared source input mismatch")
        if receipt["binding"] == "current_experiment" and receipt["source_experiment_ref"] != refs["experiment_ref"]:
            raise ValueError("stage report current receipt mismatch")
    for row in value["measurements"]:
        seconds = _seconds(row["seconds"])
        _seconds(row["reported_seconds"])
        if row["status"] not in ("MEASURED", "NOT_PROVIDED", "REUSED_NOT_EXECUTED"):
            raise ValueError("invalid stage measurement status")
        if row["mode"] not in (None, "cold_build", "saved_input_build", "cache_reuse", "readonly_load"):
            raise ValueError("invalid stage measurement mode")
        if row["status"] != "NOT_PROVIDED" and row["mode"] is None:
            raise ValueError("executed/reused stage requires an execution mode")
        if ((row["status"] == "MEASURED" and seconds is None) or
                (row["status"] != "MEASURED" and seconds is not None)):
            raise ValueError("stage measurement status/seconds mismatch")
        if ((row["status"] == "MEASURED" and seconds != row["reported_seconds"]) or
                (row["status"] == "NOT_PROVIDED" and row["reported_seconds"] is not None)):
            raise ValueError("stage measurement reported seconds mismatch")
        if row["status"] == "REUSED_NOT_EXECUTED" and (row["stage"] not in ("feature", "qlib") or
                row["reported_seconds"] not in (None, 0)):
            raise ValueError("invalid reused stage measurement")
        if row["feature_ref"] != refs["feature_ref"]:
            raise ValueError("stage report measurement feature mismatch")
        if row["receipt_file_digest"] is None:
            if row["status"] != "NOT_PROVIDED" or row["seconds"] is not None or row["source_experiment_ref"] != refs["experiment_ref"]:
                raise ValueError("invalid unprovided stage measurement")
            continue
        receipt = receipts.get(row["receipt_file_digest"])
        if receipt is None or receipt["feature_ref"] != row["feature_ref"] or receipt["source_experiment_ref"] != row["source_experiment_ref"]:
            raise ValueError("stage report receipt reference mismatch")
        inherited = receipt["binding"] == "declared_input_reuse"
        if inherited != row["inherited_feature_only"] or (inherited and row["stage"] not in ("feature", "qlib")):
            raise ValueError("invalid inherited measurement scope")
        if not inherited and receipt["source_experiment_ref"] != refs["experiment_ref"]:
            raise ValueError("stage report current experiment mismatch")


def export_stock_stage_report(experiment_path, *, timing_receipts=(), destination):
    """Project verified saved input to a JSON file outside the experiment directory."""
    experiment_path, destination = Path(experiment_path).resolve(), Path(destination).resolve()
    if destination == experiment_path or experiment_path in destination.parents:
        raise ValueError("stage report destination must be outside saved experiment directory")
    run = load_stock_ml_experiment(experiment_path)
    experiment = run.to_dict()
    predictions, evidence = run.predictions(), run.evidence()
    measurements, receipts = _measurements(timing_receipts, experiment)
    result = {"contract_version": "stock_stage_report_v1", "report_version": REPORT_VERSION,
        "implementation_ref": digest({"stock_stage_report.py": file_digest(__file__)}),
        "input_refs": {k: experiment[k] for k in _INPUT_KEYS},
        "source_input_reuse": experiment["definition"].get("input_reuse"),
        "training": _training(experiment["definition"]["config"], _read(run.path / "dataset.json")),
        "signal_summary": _signal_summary(predictions, evidence),
        "timing_receipts": receipts, "measurements": measurements,
        "limitations": ["Saved-only projection; original Data facts and clocks are not requeried.",
            "IC/RankIC describe forward labels, not account returns; no ICIR or significance inference.",
            "Inherited Feature timings do not describe current-model cold training or totals.",
            "Build and total remain separate native observations; stage durations are not summed.",
            "Longer-range throughput is not guaranteed; scale, cache, I/O and observation count differ.",
            *predictions.get("limitations", []), *evidence.get("limitations", [])]}
    result["stage_report_ref"] = _identity(result)
    result["content_digest"] = digest(result)
    _verify_closure(result)
    if destination.exists():
        if load_stock_stage_report(destination) != result:
            raise FileExistsError("different stage report already exists")
        return result
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                prefix="." + destination.name + ".", suffix=".tmp", delete=False) as f:
            staged = Path(f.name)
            f.write(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(staged, destination)
        except FileExistsError:
            if load_stock_stage_report(destination) != result:
                raise FileExistsError("different stage report already exists") from None
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    return result


def load_stock_stage_report(path):
    """Verify stored hashes/reference closure without following inputs or recomputing."""
    value = _read(path)
    if (value.get("contract_version"), value.get("report_version")) != ("stock_stage_report_v1", "axiom.stock_stage_report/1"):
        raise ValueError("unsupported stock stage report contract/version")
    if value.get("content_digest") != digest({k: v for k, v in value.items() if k != "content_digest"}):
        raise ValueError("stock stage report content digest mismatch")
    if value.get("stage_report_ref") != _identity(value):
        raise ValueError("stock stage report identity mismatch")
    _verify_closure(value)
    return value

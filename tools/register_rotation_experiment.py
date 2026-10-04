"""Register an existing frozen ETF experiment; never build features or backtest."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from axiom_research import (ArtifactRef, ExperimentStore, ExperimentReader,
    content_digest, from_dict, load_rotation_experiment, semantic_identity)


def register(args):
    experiment = load_rotation_experiment(args.experiment)
    saved = experiment.manifest()
    frame = experiment.signal_frame()
    identity = experiment.feature_build.identity
    inputs = [*identity.data_refs, *identity.view_refs,
              identity.reference_universe, identity.implementation_package]
    outputs = [
        replace(from_dict(saved["feature_build"]), uri=str(experiment.feature_build.path.resolve())),
        ArtifactRef(artifact_type="SignalRun", artifact_id=frame["signal_run_ref"],
            artifact_contract_version="signal_frame_v1",
            content_digest=content_digest((experiment.signal_path / "signal-frame.json").read_bytes()),
            uri=str((experiment.signal_path / "signal-frame.json").resolve())),
        ArtifactRef(artifact_type="RotationExperiment", artifact_id="sha256:" + experiment.path.name,
            artifact_contract_version="rotation_experiment_v1",
            content_digest=content_digest((experiment.path / "experiment.json").read_bytes()),
            uri=str(experiment.path.resolve())),
    ]
    backtest = evaluation = None
    if args.backtest is not None:
        from axiom_engine.runtime import load_backtest_run
        run = load_backtest_run(args.backtest).to_dict()
        backtest = {key: run[key] for key in
                    ("run_id", "content_digest", "signal_ref", "committed_sequence")}
        backtest["uri"] = str(args.backtest.resolve())
    if args.evaluation is not None:
        from axiom_engine.runtime import load_backtest_evaluation
        report = load_backtest_evaluation(args.evaluation).to_dict()
        evaluation = {"evaluation_ref": report["evaluation_ref"],
            "evaluation_content_digest": report["content_digest"],
            "input_run_ref": report["input_run_ref"], "uri": str(args.evaluation.resolve())}
    writer = ExperimentStore(args.index)
    writer.register_saved_experiment(
        question=dict(question_id=args.question_id, title=args.title,
            description=args.description, hypothesis=args.hypothesis),
        version=dict(label=args.version_label, explanation=args.explanation,
            parent_version_ref=args.parent_version_ref,
            parameters={"data_snapshot": saved["data_snapshot"], "policy": saved["policy"],
            "evaluation_range": saved["evaluation_range"], "universe": saved["universe"],
            "signal_price_basis": saved["signal_price_basis"],
            "feature_release_ref": semantic_identity(identity.feature_release),
            "lookback_sessions": identity.lookback_sessions, "pit_policy": identity.pit_policy},
            input_refs=inputs, explicit_changes=args.change),
        run=dict(status=args.status, reason=args.reason, outcome=args.outcome,
            output_refs=outputs, backtest_ref=backtest, evaluation_ref=evaluation))
    return ExperimentReader(args.index).detail(args.question_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("index", "experiment"):
        parser.add_argument("--" + field, required=True, type=Path)
    for field in ("question-id", "title", "description", "hypothesis",
                  "version-label", "explanation"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--parent-version-ref")
    parser.add_argument("--change", action="append", default=[])
    parser.add_argument("--status", choices=("COMPLETE", "FAILED", "BLOCKED"), default="COMPLETE")
    parser.add_argument("--reason")
    parser.add_argument("--outcome")
    parser.add_argument("--backtest", type=Path)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    print(json.dumps(register(args), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

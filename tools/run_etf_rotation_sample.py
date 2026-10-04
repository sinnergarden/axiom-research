"""Offline acceptance of the existing pinned seven-ETF sample; no acquisition."""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

from axiom_data import Data, QuerySpec
from axiom_research import SessionRange, semantic_identity
from axiom_research.joint_build import _ref
from axiom_research.rotation import POLICY, build_rotation_experiment


def digest_file(path):
    return sha256(path.read_bytes()).hexdigest()


def freeze_runtime():
    """Bind actually imported Data/Research/Core code without installing packages."""
    import axiom_data
    import axiom_research
    import axiom_engine.core
    roots = {"Data": Path(axiom_data.__file__).parent,
             "Research": Path(axiom_research.__file__).parent,
             "Core": Path(axiom_engine.core.__file__).parent}
    descriptor = {"schema_version": "research_runtime_source_v1", "python": sys.version,
                  "files": {name: {str(p.relative_to(root)): digest_file(p)
                                   for p in sorted(root.rglob("*.py"))} for name, root in roots.items()}}
    return _ref("ImplementationPackage", descriptor), descriptor


def inventory(root):
    return {str(p.relative_to(root)): (digest_file(p), p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path(__file__).resolve().parents[1] / "examples/etf_rotation_sample.json")
    args = parser.parse_args()
    root, destination = args.data_root.resolve(), args.destination.resolve()
    if destination == root or root in destination.parents:
        parser.error("Research destination must be outside the Data root")
    spec = json.loads(args.spec.read_text())
    assert spec["schema_version"] == "research_etf_sample_v1"
    assert spec["price_basis"] == "common_anchor_adjusted_v1"
    assert all(spec[key] == POLICY[key] for key in ("positive_filter", "rebalance", "tie_break")), "unsupported baseline policy"
    package, descriptor = freeze_runtime()
    before = inventory(root)
    cutoffs = {s: s + "T" + spec["signal_cutoff_local_time"] for s in spec["sessions"]}
    prices = QuerySpec("market_daily", ("close",), tuple(spec["symbols"]), tuple(spec["sessions"]),
                       spec["pit_policy"], cutoffs)
    factors = replace(prices, domain="adjustment_factors", fields=("factor",))
    parameters = dict(snapshot=spec["snapshot_id"], price_query=prices, factor_query=factors,
        evaluation_range=SessionRange(start=spec["evaluation_start"], end=spec["evaluation_end"]),
        implementation_package=package)
    started = time.perf_counter()
    first = build_rotation_experiment(Data(root), destination=destination / "primary", **parameters)
    build_seconds = time.perf_counter() - started
    original = first.signal_frame()
    artifacts_before = inventory(destination / "primary")
    class NoFacts(Data):
        def read(self, **kwargs):
            raise AssertionError("cache must not read facts")
        states = read
        members = read
        events = read
    started = time.perf_counter()
    with patch("axiom_research.rotation.execute_feature_plan", side_effect=AssertionError("cache must not execute Core")):
        cached = build_rotation_experiment(NoFacts(root), destination=destination / "primary", **parameters)
    reuse_seconds = time.perf_counter() - started
    assert cached.reused and cached.feature_reused and cached.signal_reused
    assert cached.signal_frame() == original and inventory(destination / "primary") == artifacts_before
    reproduced = build_rotation_experiment(Data(root), destination=destination / "reproduction", **parameters)
    assert reproduced.signal_frame() == original
    assert inventory(root) == before, "source Data root was changed"
    probe = subprocess.run([sys.executable, "-c",
        "import json,sys; from axiom_research import load_rotation_experiment; "
        "print(json.dumps(load_rotation_experiment(sys.argv[1]).signal_frame(),sort_keys=True))", str(first.path)],
        check=True, text=True, capture_output=True)
    assert json.loads(probe.stdout) == original
    eligible = [r for r in original["rows"] if spec["evaluation_start"] <= r["session"] <= spec["evaluation_end"]]
    assert all(r["valid"] for r in eligible), "real sample has invalid scores; inspect evidence"
    report = dict(schema_version="research_etf_acceptance_v1", snapshot_id=spec["snapshot_id"],
        implementation_ref=package.content_digest, feature_identity=semantic_identity(first.feature_build.identity),
        signal_run_ref=original["signal_run_ref"], experiment_path=str(first.path),
        signal_frame_path=str(first.signal_path / "signal-frame.json"),
        input_sessions=len(prices.sessions), universe=len(prices.symbols), signal_rows=len(original["rows"]),
        valid_rows=sum(r["valid"] for r in original["rows"]), invalid_warmup_rows=sum(not r["valid"] for r in original["rows"]),
        evaluation_rows=len(eligible), build_seconds=build_seconds, cache_seconds=reuse_seconds,
        bytes=sum(p.stat().st_size for p in (destination / "primary").rglob("*") if p.is_file()),
        identical_input_reproduction=True, cache_without_fact_reads_or_core=True,
        cache_mtime_preserved=True, fresh_process_read=True, source_tree_unchanged=True,
        limitations=first.feature_build.evidence()["limitations"],
        scope="offline feature/signal acceptance only; Engine owns portfolio/account/backtest")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "implementation-source.json").write_text(json.dumps(descriptor, sort_keys=True, indent=2) + "\n")
    (destination / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

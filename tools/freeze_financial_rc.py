"""Freeze inspected definitions and evidence; never import or run legacy code.

Run from axiom-research with PYTHONPATH=src after reviewing this tool.
Writes exclusively within this repository, refuses differing existing outputs.
"""
from pathlib import Path
import json
import subprocess

import axiom_research as r

ROOT = Path(__file__).resolve().parents[1]
LEGACY = Path("/home/liuming/.openclaw/workspace/SysQ")
ARCHIVE = ROOT.parent / "data/forensic/sysq-c969c74-20260904"
sources = {}


def u(name, reason, proof):
    return r.Unknown(unknown_id=name, reason=reason, required_evidence=proof)


BLOCKERS = {
    "DATA_CLOSURE": u("DATA_CLOSURE", "Original consumed Data/View/units/history closure not established",
                      "Fixed public Data/View refs covering actual fields, keys, lookback and label ends"),
    "PIT_BASELINE": u("PIT_BASELINE", "Saved models used current_constituents_snapshot; no historical PIT qualification",
                      "Revision-specific financial/holder evidence and historical membership/industry admission"),
    "LABEL_LINEAGE": u("LABEL_LINEAGE", "Original label bytes and producer/price/calendar lineage need complete validation",
                       "Original label producer, dense session mapping, price factor/action and delisting coverage"),
    "OOS_LINEAGE": u("OOS_LINEAGE", "Pinned serving metadata has no allowed historical OOS interval",
                     "Explicit model/fold/sample-key lineage and matured fit/validation/allowed dates"),
    "CORE_ABI": u("CORE_ABI", "E0 compatible frozen Feature/Inference/Signal package has not been implemented",
                  "Independent E0 conformance and frozen package/ABI refs"),
}


def source(relative):
    relative = str(relative)
    path = (ROOT.parent if relative.startswith("design/") else LEGACY) / relative
    digest = r.content_digest(path.read_bytes())
    result = r.ArtifactRef(artifact_type="LegacyEvidence", artifact_id=digest[7:],
                           artifact_contract_version="legacy-bytes-v1", content_digest=digest,
                           uri="sysq-forensic://" + digest[7:] + "/" + relative)
    sources[relative] = {"path": str(path), "ref": r.to_dict(result)}
    return result


def write_checked(path, text):
    payload = (text + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing output differs; preserve it and choose a reviewed new version: {path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)


def feature_pit_policy(feature):
    """Preserve inspected per-source rules without selecting an original source mode."""
    policy = feature.get("source_publication_policy", BLOCKERS["DATA_CLOSURE"])
    dependency = update = exact = BLOCKERS["DATA_CLOSURE"]
    if isinstance(policy, str) and policy.startswith("Income quarterly facts"):
        dependency = "Per-feature maximum availability of current and all cumulative-decomposition, rolling-quarter and year-over-year dependencies; reject values without dependency availability"
        update = "Order events by security, dependency availability and report end; at equal availability retain latest report end; emit only strictly newer report periods per feature availability stream"
        exact = "source_mode_dependent"
    elif isinstance(policy, str) and policy.startswith("Legacy shareholder loader"):
        dependency = "Legacy announcement date on each security's announcement-ordered rows; previous-row dependencies follow that order; revision-specific visibility requires source artifacts"
        update = "Previous values computed in announcement order, then backward as-of joined; duplicate/revision identity requires source artifacts"
        exact = "allow_on_source_date"
    return r.PITPolicy(source_publication_policy=policy,
                       availability_dependency=dependency,
                       report_period_update_semantics=update,
                       exact_date_matching=exact,
                       original_materialization=BLOCKERS["PIT_BASELINE"])


def build():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=LEGACY, check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != "c969c74a66b148fa66396c3896e48760004683d4":
        raise ValueError("Legacy source HEAD differs from reviewed evidence")
    dirty = subprocess.run(["git", "diff", "--binary"], cwd=LEGACY, check=True,
                           capture_output=True).stdout
    if dirty != (ARCHIVE/"source_state/dirty_worktree.patch").read_bytes():
        raise ValueError("Legacy working patch differs from frozen source evidence")
    fmap = json.loads((ROOT / "reports/feature-map.json").read_text())
    if isinstance(fmap, dict):
        fmap = fmap["features"]
    config = ARCHIVE / "payload/configs/features/v3a_plus_liquidity_financial_rc.yaml"
    ordered = [line[2:].strip() for line in config.read_text().splitlines() if line.startswith("- ")]
    if len(ordered) != 96 or [f["name"] for f in fmap] != ordered:
        raise ValueError("Feature map does not exactly match frozen 96-column order")
    feature_evidence = source("configs/features/v3a_plus_liquidity_financial_rc.yaml")
    if r.content_digest(config.read_bytes()) != feature_evidence.content_digest:
        raise ValueError("Live/frozen feature config differ")
    strategy_evidence = source("configs/strategies/financial_rc.yaml")
    for p in ("qsys/feature/builder.py", "qsys/feature/transforms.py", "qsys/data/adapter.py",
              "qsys/label/compute.py", "qsys/model/financial_rc_trainer.py",
              "qsys/research/generators/lightgbm_single_label.py", "qsys/research/generators/utils.py",
              "qsys/research/rolling_window.py", "qsys/research/matrix_job.py",
              "qsys/signal/alpha_v1/training.py", "qsys/signal/alpha_v1/labels.py",
              "qsys/signal/model_blend_inference.py",
              "tests/model/test_financial_rc_trainer.py", "tests/signal/test_model_blend_inference.py",
              "tests/research/test_lightgbm_single_label.py", "tests/label/test_label_compute_pit_continuity.py"):
        source(p)
    for h in (60, 180):
        exp = f"financial_rc_{h}d_rolling_5y_to_202607_v3"
        for suffix in ("", "_pit"):
            source(f"configs/research/60d/{exp}{suffix}.yaml")
        source(f"data/research/experiments/{exp}/signal_research_manifest.json")
        run = f"rolling__{exp}__v3a_growth_financial_{h}d__fwd_ret_{h}d_raw__daily_zscore__2021-01-01_2026-07-31"
        source(f"data/research/signals/fwd_ret_{h}d_raw__daily_zscore/{run}/manifest.json")
    r3 = "financial_rc_180d_rolling_5y_to_202607_v3_pit_csi1800_terminal_r3"
    source(f"configs/research/60d/{r3}.yaml")
    source("reports/research/csi1800_s180_r3_certification_report_20260830.md")
    run = f"rolling__{r3}__v3a_growth_financial_180d_pit_csi1800_terminal_r3__fwd_ret_180d_raw_pit_csi1800__daily_zscore__2021-01-01_2026-07-31"
    source(f"data/research/signals/fwd_ret_180d_raw_pit_csi1800__daily_zscore/{run}/manifest.json")

    requirement_rows = [
        ("security_calendar", ["all features", "LabelSpec", "DatasetSpec"],
         "Permanent security_id mapping, actual exchange sessions; legacy instrument/trade_date alias mapping",
         "Dense continuous keyed history and next-session addressing; no weekday fallback",
         ["code reuse", "holiday", "membership exit and reentry", "suspension gap"]),
        ("market_valuation", ["relative_strength", "liquidity", "LabelSpec"],
         "Daily OHLC, volume/amount, pe/pb, total/circ_mv, turnover, factor with explicit units",
         "Immutable public facts; close/factor adjustment semantics and coverage through label end",
         ["zero denominator", "delisting", "factor revision", "future anchor"]),
        ("financial", ["fundamental_context", "growth_confirmation_v0"],
         "roe, margins, debt/assets, operating_cf, net_income, revenue, total_assets; quarterly income/cost/profit",
         "Revision-bound as-of facts and stable single-quarter/TTM with dependency availability",
         ["restatement", "missing quarter", "same-day publication", "currency/percent scaling"]),
        ("margin", ["margin_*"], "margin_balance, buy, repay, amount and float market value; research source lag0",
         "Fixed source and effective session; source lag separate from inference snapshot lag1",
         ["ineligible vs missing", "balance units", "publication delay"]),
        ("shareholder", ["holder_*", "top10_*", "avg_shares_per_holder_chg_qoq"],
         "Complete holder-number and top10 report groups/revisions with announcement/observed times",
         "As-of projections, predecessor announcement rows and explicit stale/missing reasons",
         ["incomplete top10", "same-date revisions", "ann_date not revision proof"]),
        ("membership_industry", ["rank", "industry zscore", "signal daily zscore"],
         "Explicit reference membership mask, industry and current-snapshot versus historical PIT status",
         "Separate continuous read union, feature reference CS and eligible scored CS",
         ["no mask means all legacy rows", "single member", "unknown industry", "exit/reentry"]),
    ]
    requirements = tuple(r.Requirement(name=name, required_by=tuple(by), input_semantics=inp,
                                       output_semantics=out, edge_cases=tuple(edges),
                                       reference_fixture="financial_rc.json#feature_release.plan",
                                       status="DECISION", evidence=(feature_evidence,))
                         for name, by, inp, out, edges in requirement_rows)
    dr = r.DataRequirements(requirements=requirements, key=("security_id", "session"),
                            scope=BLOCKERS["DATA_CLOSURE"], lookback_sessions=756,
                            label_extension_sessions=181,
                            pit_policy="legacy_current_snapshot_not_historical_pit",
                            public_view_binding=BLOCKERS["DATA_CLOSURE"])
    defs, columns = [], []
    for f in fmap:
        path = f["source_file"]
        impl = source(path)
        def value(key):
            if f[key] is not None:
                return f[key]
            return u("FEATURE:" + f["name"] + ":" + key,
                     "Exact " + key + " not proven for " + f["name"],
                     "Validate source dependency closure and actual sample calendar")
        defs.append(r.FeatureDefinition(name=f["name"], business_definition=f["business_definition"],
                    inputs=tuple(f["inputs"]), formula=value("formula"), implementation_ref=impl,
                    window_sessions=value("window_sessions"), lag_sessions=value("lag_sessions"),
                    missing_policy=value("missing_policy"), outlier_policy=value("outlier_policy"),
                    reference_universe_policy=value("reference_universe_policy"),
                    pit_policy=feature_pit_policy(f),
                    metadata={"source_lines": f["source_lines"], "classification": "FACT: inspected code definition"}))
        policy = f["reference_universe_policy"]
        if policy.startswith("Single-security time series;"):
            stage = "base"
        elif policy.startswith("Only rows selected by the explicit _pit_member mask"):
            stage = "cross_sectional"
        else:
            raise ValueError("Unreviewed Feature stage for " + f["name"])
        columns.append(r.Column(name=f["name"], dtype="float32", unit=BLOCKERS["DATA_CLOSURE"], stage=stage))
    plan = r.FeaturePlanSpec(key=("security_id", "session"), features=tuple(defs),
            ordered_output_schema=tuple(columns), data_requirements=dr,
            history_policy="Continuous per-security history; 756 prior daily observations plus quarterly/announcement predecessors; legacy1461-calendar-day padding is not density proof",
            pit_policy="legacy_current_snapshot_not_historical_pit",
            cutoff_policy="legacy_current_snapshot: feature f; canonical decision next_session(f) at18:00 and candidates following session; source publication qualification unresolved",
            execution_abi=BLOCKERS["CORE_ABI"])
    feature = r.FeatureRelease(name="v3a_plus_liquidity_financial_rc.c969c74.definition",
            business_definition="Financial/value/growth, relative strength, margin, shareholder and liquidity research recipe; inspected-source definition, original model replay not certified",
            plan=plan)
    datasets, training, labels, anchors, evaluations = [], [], [], [], []
    for h, aid in ((60, "e75f67aee783d789"), (180, "b918966fd9fab47b")):
        mpath = f"data/research/models/{h}d_v3a_growth_financial/{aid}/meta.json"
        meta = json.loads((LEGACY/mpath).read_text())
        metrics = meta["metrics"]
        if meta["ordered_features"] != ordered:
            raise ValueError("Model ordered schema differs from frozen feature list")
        evidence = source(mpath)
        for filename in ("model.txt", "center.json", "scale.json"):
            actual = source(str(Path(mpath).with_name(filename)))
            if actual.content_digest != "sha256:" + meta["artifact_sha256"][filename]:
                raise ValueError("Pinned model/scaler bytes differ from metadata")
        source(f"configs/labels/fwd_ret_{h}d_raw.yaml")
        label = r.LabelSpec(name=f"financial_rc_{h}d_next_session_close.c969c74.definition",
                 key=("security_id", "session"), horizon_sessions=h, feature_session="f (Axiom session key); legacy LabelStore addressed at next_session(f)",
                 return_start_rule="next_session_close", return_end_rule="horizon_after_start",
                 price_basis="close_times_factor", benchmark_semantics="absolute_return",
                 corporate_action_semantics="supplier_cumulative_factor",
                 normalization_policy="none; raw denotes no label normalization; stored float32",
                 maturity=r.MaturitySpec(rule="outcomes_available_strictly_before_cutoff",
                     calendar_policy="actual immutable exchange session calendar required; original fallback usage unresolved",
                     availability_rule=BLOCKERS["LABEL_LINEAGE"]),
                 missing_delisting_policy="legacy drops NaN label values after shift; no explicit delisting payoff; original missing/exit coverage unresolved",
                 metadata={"legacy_formula_evidence": f"A[f+{h+1}]/A[f+1]-1; A=close*factor; LabelStore itself stores A[t+{h}]/A[t]-1"})
        split = r.SplitSpec(train=u(f"EVAL_TRAIN_{h}", "Metadata has evaluation_train_end=" + metrics["evaluation_train_end"] + " but no evaluation_train_start",
                                  "Original purged evaluation sample-key manifest with exact bounds"),
                 validation=r.SessionRange(start=metrics["validation_start"], end=metrics["validation_end"]),
                 oos=BLOCKERS["OOS_LINEAGE"], purge=f"H+1={h+1} intervening sessions; evaluation_train_end index=validation_start_index-H-2",
                 maturity_cutoff=meta["as_of_date"] + " (saved serving metadata as-of); target must already be realized",
                 fit_scope="train_only", sample_alignment="keyed_security_session")
        ds = r.DatasetSpec(name=f"financial_rc_{h}d_pinned_evidence", key=("security_id", "session"),
                feature_release=feature, label=label,
                sample_universe="CSI800 current_constituents_snapshot; historical rows not PIT",
                sample_filter="keyed next-session label join; retain nonmissing label and legacy shareholder/feature gates; exact original row closure requires evidence",
                split=split, rolling_folds=(r.FoldSpec(fold_id="legacy-serving-" + aid, split=split,
                                                      allowed_prediction_interval=BLOCKERS["OOS_LINEAGE"]),),
                rolling_rule="Pinned serving pair: one matured fit per model, trailing40-session evaluation then full matured refit; rolling20-session study is a separate variant",
                retrain_step_sessions=u("SERVING_RETRAIN", "Pinned serving config does not establish periodic20-session retrain schedule", "Explicit approved retrain schedule; do not import rolling experiment step"),
                train_window_sessions=504, ordered_model_schema=tuple(columns))
        ts = r.TrainingSpec(name=f"financial_rc_{h}d_training", dataset_name=ds.name,
                backend="lightgbm", backend_version=meta["library_versions"]["lightgbm"], objective="regression",
                parameters={"n_estimators": 300, "early_stopping_rounds": 20, "metric": "mse",
                  "colsample_bytree": .8879, "learning_rate": .0421, "subsample": .8789,
                  "lambda_l1": 205.6999, "lambda_l2": 580.9768, "max_depth": 8,
                  "num_leaves": 210, "num_threads": 8, "verbosity": -1,
                  "selected_iterations_metadata_anchor": metrics["selected_iterations"]},
                seed=42, preprocessing="input fill0/float32; median center; unscaled MAD median(abs(X-center)), zero scale->1; transform clip[-3,3] then fill0; evaluation fit=train-only; serving refit=all matured",
                fit_scope="all_matured_after_selection", fit_range=r.SessionRange(start=meta["train_start"], end=meta["train_end"]),
                selection_protocol="horizon-purged trailing40-session validation, early_stop20 selects tree count; serving refit fixed selected rounds; OOS not established",
                resources=r.ResourceSpec(threads=8, concurrent_folds=u("CONCURRENCY", "Original concurrency budget not recorded", "Explicit run resource plan"),
                                         memory_limit_mb=u("MEMORY_BUDGET", "Original memory budget not recorded", "Explicit run memory limit")))
        labels.append(label); datasets.append(ds); training.append(ts)
        anchors.append({"horizon": h, "metadata_ref": r.to_dict(evidence), "metadata": meta})
        evaluations.append(r.SignalEvaluationSpec(key=("security_id", "session"), label=label,
             signal_stage="final", metrics=("daily_rank_ic", "coverage"),
             reference_universe=BLOCKERS["PIT_BASELINE"], missing_policy="exclude invalid and report reasons/coverage",
             tie_policy="average ranks", maturity_policy="only realized labels available at evaluation cutoff",
             aggregation="DECISION: report daily series and unannualized mean; no backtest performance"))
    reference = "canonical same-day eligible scored current CSI800 snapshot before TopK; identical keyed rows for60/180; not historical PIT members"
    nodes = tuple(r.SignalNode(name=f"z{h}", op="daily_zscore", inputs=(f"pred_{h}",),
                    input_stages=("raw_prediction",), output_stage="daily_zscore", weights=(),
                    reference_universe=reference, missing_policy="reject nonfinite model predictions; reject std<1e-12",
                    parameters={"ddof": 0, "clip": None, "epsilon": 1e-12, "constant": "reject"}) for h in (60, 180))
    nodes += (r.SignalNode(name="final", op="weighted_combine", inputs=("z60", "z180"),
                    input_stages=("daily_zscore", "daily_zscore"), output_stage="final", weights=(.5, .5),
                    reference_universe=reference, missing_policy="both horizon scores required on identical eligible keys", parameters={}),)
    sp = r.SignalPlanSpec(name="financial_rc_equal_unclipped", key=("security_id", "session"),
         inputs=tuple(r.SignalInput(alias=f"pred_{h}", label=label, source_stage="raw_prediction",
                      score_semantics="LightGBM regression score trained on absolute forward adjusted return; not calibrated probability")
                      for h, label in zip((60, 180), labels)), nodes=nodes, output="final",
         join_policy="inner_on_security_session", score_semantics="0.5*z60 +0.5*z180; dimensionless standardized regression ranking score",
         available_time_semantics="Canonical decision T at18:00 uses snapshot previous_session(T), emits candidates next_session(T); original historical OOS availability unproven")
    core_rows = json.loads((ROOT / "reports/core-map.json").read_text())
    if isinstance(core_rows, dict): core_rows = core_rows["requirements"]
    core = r.CoreCapabilityRequirements(target="axiom-engine/core E0", requirements=tuple(
        r.Requirement(name=x["name"], required_by=tuple(x["required_by"]), input_semantics=x["input_semantics"],
                      output_semantics=x["output_semantics"], edge_cases=tuple(x["edge_cases"]),
                      reference_fixture="core_cases.json#" + x["name"], status="DECISION",
                      evidence=tuple(source(p["source_file"]) for p in x["evidence"])) for x in core_rows))
    recipe = r.StrategyRecipeDraft(name="Financial RC R0 frozen forensic definition", feature_release=feature,
                datasets=tuple(datasets), training=tuple(training), signal_plan=sp, evaluation=tuple(evaluations),
                core_requirements=core, evidence=(feature_evidence, strategy_evidence),
                correctness_blockers=tuple(BLOCKERS.values()))
    r.validate(recipe)
    try:
        r.validate(recipe, require_resolved=True)
    except r.ContractError:
        pass
    else:
        raise ValueError("Forensic recipe unexpectedly admitted as execution-ready")
    fixture = ROOT/"src/axiom_research/fixtures/financial_rc.json"
    write_checked(fixture, r.dumps(recipe))
    for name, value in (("data_requirements", dr), ("core_requirements", core),
                        ("label_60", labels[0]), ("label_180", labels[1])):
        write_checked(ROOT/"src/axiom_research/fixtures"/(name+".json"), r.dumps(value))
    evidence = {"run_identity": "research-r0-20260913", "source_head": "c969c74a66b148fa66396c3896e48760004683d4",
                "dirty_patch_digest": r.content_digest(dirty),
                "source_state_ref": str(ARCHIVE/"source_state/git_state.json"),
                "source_state_digest": r.content_digest((ARCHIVE/"source_state/git_state.json").read_bytes()),
                "model_anchors": anchors, "sources": sources,
                "recipe_semantic_identity": r.semantic_identity(recipe),
                "unresolved": [r.to_dict(x) for x in r.unresolved(recipe)]}
    write_checked(ROOT/"reports/evidence.json", json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps({"fixture": str(fixture), "columns": len(ordered), "unknowns": len(r.unresolved(recipe)),
                      "structural_validation": "PASS", "execution_readiness": "BLOCKED",
                      "identity": r.semantic_identity(recipe)}))


if __name__ == "__main__":
    build()

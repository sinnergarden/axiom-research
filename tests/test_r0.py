"""R0 contract tests only. Synthetic manifest examples are not research outputs."""
from dataclasses import replace, fields, is_dataclass
from pathlib import Path
import json
import tempfile
import unittest

import axiom_research as r
from axiom_research.api import TYPES


def ref(kind, name="synthetic"):
    return r.ArtifactRef(artifact_type=kind, artifact_id=name,
                         artifact_contract_version="1",
                         content_digest=r.content_digest((kind + name).encode()),
                         uri="fixture://" + name)


def interval(start, end):
    return r.SessionRange(start=start, end=end)


def unknown(name="SYNTHETIC_UNKNOWN"):
    return r.Unknown(unknown_id=name, reason="Synthetic missing correctness evidence",
                     required_evidence="Supply explicit evidence; no default")


def examples():
    """Small explicitly invented closure exercises every public contract."""
    req = r.Requirement(name="market", required_by=("ratio",), input_semantics="keyed facts",
                        output_semantics="fixed values", edge_cases=("missing facts reject",),
                        reference_fixture="synthetic-one-column", status="DECISION", evidence=())
    dr = r.DataRequirements(requirements=(req,), key=("security_id", "session"),
                            scope=interval("2019-01-01", "2022-12-31"), lookback_sessions=2,
                            label_extension_sessions=181, pit_policy="fixture_safe_v1",
                            public_view_binding=ref("FactView"))
    col = r.Column(name="ratio", dtype="float32", unit="ratio", stage="model_input")
    fd = r.FeatureDefinition(name="ratio", business_definition="Synthetic feature",
                             inputs=("a", "b"), formula="a / b", implementation_ref=ref("FeaturePlugin"),
                             window_sessions=2, lag_sessions=0, missing_policy="reject",
                             outlier_policy="none", reference_universe_policy="daily_fixture_members",
                             pit_policy=r.PITPolicy(source_publication_policy="synthetic close",
                                availability_dependency="both inputs at close",
                                report_period_update_semantics="daily observations",
                                exact_date_matching="allow_on_source_date",
                                original_materialization=ref("FactView")))
    plan = r.FeaturePlanSpec(key=("security_id", "session"), features=(fd,),
                             ordered_output_schema=(col,), data_requirements=dr,
                             history_policy="continuous", pit_policy="fixture_safe_v1",
                             cutoff_policy="prior_close_v1", execution_abi="fixture_abi_v1")
    feature = r.FeatureRelease(name="synthetic", business_definition="test only", plan=plan)
    labels, datasets, training = [], [], []
    split = r.SplitSpec(train=interval("2020-01-01", "2020-03-31"),
                         validation=interval("2021-01-01", "2021-02-28"),
                         oos=interval("2022-01-01", "2022-02-28"), purge="target_end_before_validation_v1",
                         maturity_cutoff="2021-12-31", fit_scope="train_only",
                         sample_alignment="keyed_security_session")
    fold = r.FoldSpec(fold_id="f0", split=split, allowed_prediction_interval=split.oos)
    for h in (60, 180):
        label = r.LabelSpec(name=f"synthetic_{h}", key=("security_id", "session"), horizon_sessions=h,
                             feature_session="f",
                             return_start_rule="next_session_close", return_end_rule="horizon_after_start",
                             price_basis="close_times_factor", benchmark_semantics="absolute_return",
                             corporate_action_semantics="supplier_cumulative_factor", normalization_policy="none",
                             maturity=r.MaturitySpec(rule="outcomes_available_strictly_before_cutoff",
                                                     calendar_policy="explicit_calendar", availability_rule="close_final"),
                             missing_delisting_policy="invalidate_and_report")
        ds = r.DatasetSpec(name=f"ds{h}", key=("security_id", "session"), feature_release=feature,
                            label=label, sample_universe="daily_fixture_members", sample_filter="valid",
                            split=split, rolling_folds=(fold,), rolling_rule="explicit_fold_schedule",
                            retrain_step_sessions=20, train_window_sessions=504, ordered_model_schema=(col,))
        ts = r.TrainingSpec(name=f"train{h}", dataset_name=ds.name, backend="lightgbm", backend_version="4.6.0",
                             objective="regression", parameters={"n_estimators": 300}, seed=42,
                             preprocessing="median_mad_train_clip3", fit_scope="train_only", fit_range=split.train,
                             selection_protocol="purged_validation", resources=r.ResourceSpec(threads=1,
                             concurrent_folds=1, memory_limit_mb=128))
        labels.append(label); datasets.append(ds); training.append(ts)
    nodes = tuple(r.SignalNode(name=f"z{h}", op="daily_zscore", inputs=(f"pred_{h}",),
                                input_stages=("raw_prediction",), output_stage="daily_zscore", weights=(),
                                reference_universe="daily_fixture_members", missing_policy="reject",
                                parameters={"ddof": 0, "clip": None, "epsilon": 1e-12, "constant": "reject"})
                  for h in (60, 180))
    final = r.SignalNode(name="final", op="weighted_combine", inputs=("z60", "z180"),
                          input_stages=("daily_zscore", "daily_zscore"), output_stage="final",
                          weights=(0.5, 0.5), reference_universe="daily_fixture_members", missing_policy="reject",
                          parameters={})
    sp = r.SignalPlanSpec(name="synthetic_equal", key=("security_id", "session"),
                           inputs=tuple(r.SignalInput(alias=f"pred_{h}", label=l, source_stage="raw_prediction",
                                                      score_semantics="regression_score")
                                        for h, l in zip((60, 180), labels)),
                           nodes=nodes+(final,), output="final", join_policy="inner_on_security_session",
                           score_semantics="standardized_return_prediction", available_time_semantics="prior_close")
    ev = r.SignalEvaluationSpec(key=("security_id", "session"), label=labels[0], signal_stage="final",
                                 metrics=("daily_rank_ic",), reference_universe="daily_fixture_members",
                                 missing_policy="exclude_report", tie_policy="average_rank",
                                 maturity_policy="target_end_available", aggregation="unannualized_daily_mean")
    core = r.CoreCapabilityRequirements(requirements=(req,), target="axiom-engine/core E0")
    recipe = r.StrategyRecipeDraft(name="synthetic", feature_release=feature, datasets=tuple(datasets),
                                    training=tuple(training), signal_plan=sp, evaluation=(ev,),
                                    core_requirements=core, evidence=(), correctness_blockers=())
    fb = r.FeatureBuildIdentity(data_refs=(ref("DataSnapshot"),), view_refs=(ref("FactView"),),
                                 feature_release=feature, scope=dr.scope, lookback_sessions=2,
                                 pit_policy=plan.pit_policy, cutoff_policy=plan.cutoff_policy,
                                 reference_universe=ref("Universe"), implementation_package=ref("CorePackage"))
    lb = r.LabelBuildIdentity(data_refs=(ref("DataSnapshot"),), label=labels[0], scope=dr.scope,
                               calendar_ref=ref("Calendar"), benchmark_semantics="absolute_return",
                               corporate_action_semantics="supplier_cumulative_factor")
    di = r.DatasetIdentity(feature_build=ref("FeatureBuild"), label_build=ref("LabelBuild"),
                             dataset_spec=datasets[0], fit_protocol="train_only")
    models = []
    for ds, ts in zip(datasets, training):
        mi = r.ModelIdentity(dataset=ref("TrainingDataset", ds.name), dataset_spec=ds,
                              training_spec=ts, fold=fold, environment=ref("Environment"))
        models.append(r.ModelReleaseManifest(identity=mi, model_bytes=ref("ModelBytes", ds.name),
                                              fitted_preprocessing_state=ref("TransformState", ds.name),
                                              ordered_schema=(col,), inference_package=ref("InferencePackage"),
                                              inference_abi="fixture_predict_v1"))
    si = r.SignalIdentity(plan=sp, bindings=tuple(r.SignalBinding(alias=f"pred_{h}", model=m,
                                                                  feature_build=ref("FeatureBuild"))
                                                 for h, m in zip((60, 180), models)),
                            allowed_dates=split.oos, implementation_package=ref("SignalPackage"))
    run = r.SignalRunManifest(identity=si, output_bytes=ref("SignalFrame"), key=("security_id", "session"),
                               output_stage="final", time_columns=("knowledge_cutoff", "simulated_available_at"),
                               validity_columns=("valid", "invalid_reason"))
    return recipe, fb, lb, di, run, unknown()


class R0Contracts(unittest.TestCase):
    def setUp(self):
        self.recipe, self.fb, self.lb, self.di, self.run, self.unknown = examples()

    def reject(self, value):
        with self.assertRaises(r.ContractError):
            r.validate(value)

    def test_R0_01_all_public_types_roundtrip(self):
        found = {}
        def collect(value):
            if isinstance(value, r.Contract):
                found[type(value).__name__] = value
                for f in fields(value):
                    collect(getattr(value, f.name))
            elif isinstance(value, tuple):
                for v in value: collect(v)
        for value in examples(): collect(value)
        self.assertEqual(set(found), set(TYPES))
        for name, value in found.items():
            with self.subTest(contract=name):
                restored = r.loads(r.dumps(value))
                self.assertEqual(value, restored)
                self.assertEqual(r.semantic_identity(value), r.semantic_identity(restored))
                self.assertIn(name, r.contract_schema(type(value))["$defs"])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"manifest.json"
            r.save(self.run, path)
            self.assertEqual(r.load(path), self.run)
            with self.assertRaises(FileExistsError): r.save(self.run, path)

    def test_R0_02_invalid_contracts(self):
        fp = self.recipe.feature_release.plan
        ds = self.recipe.datasets[0]
        sp = self.recipe.signal_plan
        bad = [replace(fp, ordered_output_schema=fp.ordered_output_schema*2),
               replace(fp, key=()), replace(ds.label, horizon_sessions=0),
               replace(ds.label, horizon_sessions=True), replace(ds.label, maturity=None),
               replace(ds.label, return_end_rule="start+5_sessions"),
               replace(ds.split, validation=ds.split.train),
               replace(ds.split, oos=interval("2019-01-01", "2019-02-01")),
               replace(ds, rolling_folds=[]),
               replace(self.run.identity.bindings[0].model.identity, training_spec=
                       replace(self.recipe.training[0], fit_range=ds.split.validation)),
               replace(self.run.identity.bindings[0].model, ordered_schema=()),
               replace(sp, join_policy="positional"),
               replace(sp, nodes=(replace(sp.nodes[0], inputs=("absent",)),)+sp.nodes[1:]),
               replace(sp, nodes=sp.nodes[:-1]+(replace(sp.nodes[-1], input_stages=()),)),
               replace(sp, nodes=sp.nodes[:-1]+(replace(sp.nodes[-1], input_stages=("raw_prediction",)*2),)),
               replace(sp, nodes=sp.nodes[:-1]+(replace(sp.nodes[-1], weights=(float("nan"), .5)),)),
               replace(self.recipe, training=(self.recipe.training[0],)*2),
               replace(self.run.identity, allowed_dates=interval("2021-01-01", "2021-02-01")),
               replace(ref("FactView"), artifact_id="latest"),
               replace(ref("FactView"), content_digest="sha256:bad"),
               replace(ds, rolling_folds=(replace(ds.rolling_folds[0], split=replace(ds.split,
                       oos=interval("2023-01-01", "2023-02-01")),
                       allowed_prediction_interval=interval("2023-01-01", "2023-02-01")),))]
        for index, obj in enumerate(bad):
            with self.subTest(case=index): self.reject(obj)
        raw = r.to_dict(ds.label)
        del raw["maturity"]
        with self.assertRaises(r.ContractError): r.from_dict(raw)
        for text in ('{"contract_type":"LabelSpec","contract_type":"LabelSpec"}', '{"x":NaN}'):
            with self.assertRaises(r.ContractError): r.loads(text)
        raw = r.to_dict(ds)
        raw["contract_version"] = "2"
        with self.assertRaises(r.ContractError): r.from_dict(raw)

    def test_R0_03_identity_semantic_changes(self):
        fp = self.fb.feature_release.plan
        def feature_changed(**kwargs):
            return replace(self.fb, feature_release=replace(self.fb.feature_release,
                           plan=replace(fp, features=(replace(fp.features[0], **kwargs),))))
        changed = [(self.fb, feature_changed(formula="a / (b + 1)")),
                   (replace(self.fb, lookback_sessions=3),
                    replace(feature_changed(lag_sessions=1), lookback_sessions=3)),
                   (self.lb, replace(self.lb, label=replace(self.lb.label, horizon_sessions=59))),
                   (self.fb, replace(self.fb, pit_policy="different_safe_v2", feature_release=
                       replace(self.fb.feature_release, plan=replace(fp, pit_policy="different_safe_v2")))),
                   (self.di, replace(self.di, dataset_spec=replace(self.di.dataset_spec,
                        split=replace(self.di.dataset_spec.split, purge="longer_purge_v2")))),
                   (self.di, replace(self.di, dataset_spec=replace(self.di.dataset_spec,
                        split=replace(self.di.dataset_spec.split,
                                      train=interval("2019-12-31", "2020-03-31"))))),
                   (self.run.identity.bindings[0].model.identity,
                        replace(self.run.identity.bindings[0].model.identity, training_spec=
                        replace(self.recipe.training[0], parameters={"n_estimators": 200}))),
                   (self.recipe.signal_plan, replace(self.recipe.signal_plan, nodes=
                        self.recipe.signal_plan.nodes[:-1]+(replace(self.recipe.signal_plan.nodes[-1],
                                                                  weights=(.4, .6)),))),
                   (self.run.identity, replace(self.run.identity, implementation_package=ref("SignalPackage", "v2")))]
        mi = self.run.identity.bindings[0].model.identity
        fold = replace(mi.fold, fold_id="f1")
        changed.append((mi, replace(mi, fold=fold, dataset_spec=replace(mi.dataset_spec, rolling_folds=(fold,)))))
        for before, after in changed:
            self.assertNotEqual(r.semantic_identity(before), r.semantic_identity(after))
        self.assertEqual(r.semantic_identity(self.fb), r.semantic_identity(replace(self.fb, metadata={"title": "pretty"})))
        a = ref("FactView")
        self.assertEqual(r.semantic_identity(a), r.semantic_identity(replace(a, uri="fixture://moved")))
        ts = self.recipe.training[0]
        self.assertNotEqual(r.semantic_identity(ts), r.semantic_identity(replace(ts, parameters={"metadata": {"loss": "different"}})))
        # Numeric and container normalization must not alter identity after roundtrip.
        single = replace(self.recipe.signal_plan.nodes[-1], inputs=("z60",),
                         input_stages=("daily_zscore",), weights=(1,))
        self.assertEqual(r.semantic_identity(single), r.semantic_identity(r.loads(r.dumps(single))))

    def test_R0_04_frozen_financial_rc(self):
        path = Path(r.__file__).parent/"fixtures"/"financial_rc.json"
        recipe = r.load(path)
        self.assertIsInstance(recipe, r.StrategyRecipeDraft)
        self.assertEqual(len(recipe.feature_release.plan.features), 96)
        self.assertEqual([d.label.horizon_sessions for d in recipe.datasets], [60, 180])
        self.assertEqual(recipe.signal_plan.nodes[-1].weights, (.5, .5))
        self.assertEqual(recipe.signal_plan.nodes[-1].input_stages, ("daily_zscore", "daily_zscore"))
        self.assertEqual(recipe.signal_plan.nodes[0].parameters["ddof"], 0)
        self.assertIsNone(recipe.signal_plan.nodes[0].parameters["clip"])
        self.assertTrue(all(isinstance(f.formula, str) for f in recipe.feature_release.plan.features))
        self.assertEqual(recipe, r.loads(r.dumps(recipe)))

    def test_R0_05_unknown_preservation_and_admission(self):
        fp = self.recipe.feature_release.plan
        unknown_plan = replace(fp, pit_policy=self.unknown)
        self.assertEqual(r.loads(r.dumps(unknown_plan)).pit_policy, self.unknown)
        with self.assertRaises(r.ContractError): r.validate(unknown_plan, require_resolved=True)
        self.reject(replace(self.fb, feature_release=replace(self.fb.feature_release, plan=unknown_plan)))
        req = replace(fp.data_requirements.requirements[0], status="UNKNOWN")
        dr = replace(fp.data_requirements, requirements=(req,))
        self.reject(replace(self.fb, feature_release=replace(self.fb.feature_release, plan=replace(fp, data_requirements=dr))))
        recipe = r.load(Path(r.__file__).parent/"fixtures"/"financial_rc.json")
        ids = {u.unknown_id for u in r.unresolved(recipe)}
        self.assertTrue({"DATA_CLOSURE", "PIT_BASELINE", "LABEL_LINEAGE", "OOS_LINEAGE", "CORE_ABI"} <= ids)
        with self.assertRaises(r.ContractError): r.validate(recipe, require_resolved=True)
        with self.assertRaises(r.ContractError): r.validate(replace(recipe, correctness_blockers=()), require_resolved=True)

    def test_R0_06_unknown_cannot_be_laundered_through_parameters(self):
        model = self.run.identity.bindings[0].model.identity
        # Exercise Any, lists and embedded contracts, not just typed Unknown unions.
        for nested in (self.unknown, {"inner": self.unknown}, {"metadata": {"nested": self.unknown}},
                       [{"inner": [self.unknown]}],
                       {"spec": replace(self.recipe.training[1],
                                        parameters={"nested": [self.unknown]})}):
            with self.subTest(nested=type(nested).__name__):
                draft = replace(self.recipe.training[0], parameters={"custom": nested})
                restored = r.loads(r.dumps(draft))
                self.assertEqual(restored, draft)
                self.assertEqual(r.unresolved(restored), (self.unknown,))
                self.assertEqual(r.unresolved(r.to_dict(restored)), (self.unknown,))
                with self.assertRaises(r.ContractError):
                    r.validate(restored, require_resolved=True)
                blocked = replace(model, training_spec=restored)
                with self.assertRaises(r.ContractError): r.validate(blocked)
                with self.assertRaises(r.ContractError): r.semantic_identity(blocked)
                raw = r.to_dict(model)
                raw["training_spec"] = r.to_dict(restored)
                with self.assertRaises(r.ContractError): r.from_dict(raw)
        # YAML parsers return ordinary dictionaries/lists, the same wire boundary.
        try:
            import yaml
        except ImportError:
            self.fail("Run this counterexample with PyYAML installed")
        draft = replace(self.recipe.training[0], parameters={"nested": [self.unknown]})
        tagged = replace(self.recipe.training[0], parameters={"nested": [r.to_dict(self.unknown)]})
        self.assertEqual(r.semantic_identity(draft), r.semantic_identity(tagged))
        restored = r.from_dict(yaml.safe_load(yaml.safe_dump(r.to_dict(draft))))
        self.assertEqual(r.unresolved(restored), (self.unknown,))
        self.reject(replace(model, training_spec=restored))
        malformed = r.to_dict(draft)
        del malformed["parameters"]["nested"][0]["reason"]
        with self.assertRaises(r.ContractError): r.from_dict(malformed)
        malformed = r.to_dict(draft)
        malformed["parameters"]["nested"][0]["contract_type"] = "UnregisteredUnknown"
        with self.assertRaises(r.ContractError): r.from_dict(malformed)

    def test_R0_07_label_has_one_authority(self):
        label = self.lb.label
        shorter = replace(label, horizon_sessions=59)
        r.validate(shorter)
        self.assertEqual((label.return_start_offset_sessions, label.return_end_offset_sessions), (1, 61))
        self.assertEqual((shorter.return_start_offset_sessions, shorter.return_end_offset_sessions), (1, 60))
        self.assertNotEqual(label.formula, shorter.formula)
        self.assertNotEqual(label.target_interval, shorter.target_interval)
        self.assertEqual(label.maturity_lag_sessions, 61)
        self.assertEqual(shorter.maturity_lag_sessions, 60)
        self.assertNotEqual(r.semantic_identity(label), r.semantic_identity(shorter))
        # Old independent executable authorities cannot enter the new wire contract.
        for field, value in (("formula", "A[f+61]/A[f+1]-1"),
                             ("return_end_offset_sessions", 60),
                             ("return_start_offset_sessions", 1)):
            raw = r.to_dict(shorter)
            raw[field] = value
            with self.subTest(field=field), self.assertRaises(r.ContractError): r.from_dict(raw)
        for field, value in (("return_end_rule", "start+60_sessions"),
                             ("return_start_rule", "f+2"),
                             ("price_basis", "unspecified"),
                             ("benchmark_semantics", "undocumented_excess_return"),
                             ("corporate_action_semantics", "undocumented_action"),
                             ("corporate_action_semantics", "none")):
            self.reject(replace(shorter, **{field: value}))
        raw = r.to_dict(shorter)
        raw["maturity"]["lag_sessions"] = 61
        with self.assertRaises(r.ContractError): r.from_dict(raw)
        evidence = replace(shorter, metadata={"legacy_formula": label.formula})
        self.assertEqual(evidence.formula, shorter.formula)
        self.assertEqual(r.semantic_identity(evidence), r.semantic_identity(shorter))
        self.assertEqual(r.loads(r.dumps(shorter)), shorter)

    def test_R0_08_feature_pit_is_semantic_and_required(self):
        feature = self.fb.feature_release.plan.features[0]
        policy = feature.pit_policy
        for field, value in (("availability_dependency", "maximum input availability plus one session"),
                             ("report_period_update_semantics", "only newer period innovations"),
                             ("exact_date_matching", "strictly_after_dependency_date")):
            changed = replace(feature, pit_policy=replace(policy, **{field: value}))
            with self.subTest(field=field):
                self.assertEqual(changed, r.loads(r.dumps(changed)))
                self.assertNotEqual(r.semantic_identity(feature), r.semantic_identity(changed))
        for field in ("availability_dependency", "report_period_update_semantics", "exact_date_matching"):
            raw = r.to_dict(feature)
            del raw["pit_policy"][field]
            with self.assertRaises(r.ContractError): r.from_dict(raw)
        self.reject(replace(policy, exact_date_matching="arbitrary"))
        self.reject(replace(policy, contract_version="unsupported"))
        self.reject(replace(policy, exact_date_matching="source_mode_dependent"))
        self.reject(replace(policy, availability_dependency=" "))
        self.reject(replace(policy, report_period_update_semantics=""))
        unresolved_feature = replace(feature, pit_policy=replace(policy, availability_dependency=self.unknown))
        with self.assertRaises(r.ContractError): r.validate(unresolved_feature, require_resolved=True)
        self.reject(replace(self.fb, feature_release=replace(self.fb.feature_release,
                    plan=replace(self.fb.feature_release.plan, features=(unresolved_feature,)))))
        recipe = r.load(Path(r.__file__).parent/"fixtures"/"financial_rc.json")
        self.assertEqual(len(r.unresolved(recipe)), 24)
        income = next(f for f in recipe.feature_release.plan.features if f.name == "ttm_revenue_yoy")
        self.assertIsInstance(income.pit_policy.availability_dependency, str)
        self.assertIsInstance(income.pit_policy.report_period_update_semantics, str)
        self.assertEqual(income.pit_policy.exact_date_matching, "source_mode_dependent")
        self.assertTrue(r.unresolved(income.pit_policy))


if __name__ == "__main__":
    unittest.main(verbosity=2)

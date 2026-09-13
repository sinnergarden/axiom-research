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
                             outlier_policy="none", reference_universe_policy="daily_fixture_members")
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
                             feature_session="f", formula=f"A[f+{h+1}]/A[f+1]-1",
                             return_start_rule="next_session_close", return_end_rule=f"start+{h}_sessions",
                             return_start_offset_sessions=1, return_end_offset_sessions=h+1,
                             price_basis="adjusted_close", benchmark_semantics="absolute",
                             corporate_action_semantics="fixture_adjustment", normalization_policy="none",
                             maturity=r.MaturitySpec(lag_sessions=h+1, rule="target_end_available_before_fit",
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
                               calendar_ref=ref("Calendar"), benchmark_semantics="absolute",
                               corporate_action_semantics="fixture_adjustment")
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
               replace(ds.label, return_end_offset_sessions=5),
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
                   (self.lb, replace(self.lb, label=replace(self.lb.label, horizon_sessions=59,
                                                          return_end_offset_sessions=60))),
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

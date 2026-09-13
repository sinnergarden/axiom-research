"""Research-owned definitions and manifests. No feature/model/signal execution.

All semantic fields are required. Unknown is an explicit draft value, never a
default. Nested definitions are embedded deliberately: a small portable closure
is sufficient for R0 and avoids a registry/resolver framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, kw_only=True)
class Contract:
    contract_version: Literal["1"] = "1"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class Unknown(Contract):
    unknown_id: str
    reason: str
    required_evidence: str


@dataclass(frozen=True, kw_only=True)
class ArtifactRef(Contract):
    artifact_type: str
    artifact_id: str
    artifact_contract_version: str
    content_digest: str
    uri: str  # locator only; excluded from semantic identity


@dataclass(frozen=True, kw_only=True)
class Column(Contract):
    name: str
    dtype: Literal["float32", "float64"]
    unit: str | Unknown
    stage: Literal["base", "cross_sectional", "model_input"]


@dataclass(frozen=True, kw_only=True)
class SessionRange(Contract):
    start: str
    end: str  # inclusive; membership interval conventions are separate


@dataclass(frozen=True, kw_only=True)
class PITPolicy(Contract):
    source_publication_policy: str | Unknown
    availability_dependency: str | Unknown
    report_period_update_semantics: str | Unknown
    exact_date_matching: Literal["allow_on_source_date", "strictly_after_dependency_date",
                                 "source_mode_dependent", "not_applicable"] | Unknown
    original_materialization: ArtifactRef | Unknown


@dataclass(frozen=True, kw_only=True)
class FeatureDefinition(Contract):
    contract_version: Literal["2"] = "2"
    name: str
    business_definition: str
    inputs: tuple[str, ...]
    formula: str | Unknown  # declarative text, never eval'ed
    implementation_ref: ArtifactRef | Unknown
    window_sessions: int | Unknown
    lag_sessions: int | Unknown
    missing_policy: str | Unknown
    outlier_policy: str | Unknown
    reference_universe_policy: str | Unknown
    pit_policy: PITPolicy


@dataclass(frozen=True, kw_only=True)
class Requirement(Contract):
    name: str
    required_by: tuple[str, ...]
    input_semantics: str
    output_semantics: str
    edge_cases: tuple[str, ...]
    reference_fixture: str
    status: Literal["FACT", "INFERENCE", "UNKNOWN", "DECISION"]
    evidence: tuple[ArtifactRef, ...]


@dataclass(frozen=True, kw_only=True)
class DataRequirements(Contract):
    requirements: tuple[Requirement, ...]
    key: tuple[str, ...]
    scope: SessionRange | Unknown
    lookback_sessions: int | Unknown
    label_extension_sessions: int | Unknown
    pit_policy: str | Unknown
    public_view_binding: ArtifactRef | Unknown


@dataclass(frozen=True, kw_only=True)
class CoreCapabilityRequirements(Contract):
    requirements: tuple[Requirement, ...]
    target: Literal["axiom-engine/core E0"]


@dataclass(frozen=True, kw_only=True)
class FeaturePlanSpec(Contract):
    key: tuple[str, ...]
    features: tuple[FeatureDefinition, ...]
    ordered_output_schema: tuple[Column, ...]
    data_requirements: DataRequirements
    history_policy: str | Unknown
    pit_policy: str | Unknown
    cutoff_policy: str | Unknown
    execution_abi: str | Unknown


@dataclass(frozen=True, kw_only=True)
class FeatureRelease(Contract):
    name: str
    business_definition: str
    plan: FeaturePlanSpec


@dataclass(frozen=True, kw_only=True)
class MaturitySpec(Contract):
    contract_version: Literal["2"] = "2"
    rule: Literal["outcomes_available_strictly_before_cutoff"] | Unknown
    calendar_policy: str
    availability_rule: str | Unknown


@dataclass(frozen=True, kw_only=True)
class LabelSpec(Contract):
    contract_version: Literal["2"] = "2"
    name: str
    key: tuple[str, ...]
    horizon_sessions: int | Unknown
    feature_session: str | Unknown
    return_start_rule: Literal["feature_session_close", "next_session_close"] | Unknown
    return_end_rule: Literal["horizon_after_start"] | Unknown
    price_basis: Literal["close", "close_times_factor"] | Unknown
    benchmark_semantics: Literal["absolute_return"] | Unknown
    corporate_action_semantics: Literal["none", "supplier_cumulative_factor"] | Unknown
    normalization_policy: str | Unknown
    maturity: MaturitySpec | Unknown
    missing_delisting_policy: str | Unknown

    @property
    def return_start_offset_sessions(self) -> int | Unknown:
        if isinstance(self.return_start_rule, Unknown):
            return self.return_start_rule
        return {"feature_session_close": 0, "next_session_close": 1}[self.return_start_rule]

    @property
    def return_end_offset_sessions(self) -> int | Unknown:
        for value in (self.return_end_rule, self.horizon_sessions,
                      self.return_start_offset_sessions):
            if isinstance(value, Unknown):
                return value
        return self.return_start_offset_sessions + self.horizon_sessions

    @property
    def target_interval(self) -> tuple[int | Unknown, int | Unknown]:
        return self.return_start_offset_sessions, self.return_end_offset_sessions

    @property
    def maturity_lag_sessions(self) -> int | Unknown:
        """Target-end offset; actual availability must still satisfy MaturitySpec."""
        return self.return_end_offset_sessions

    @property
    def formula(self) -> str | Unknown:
        start, end = self.target_interval
        for value in (start, end, self.price_basis, self.benchmark_semantics,
                      self.corporate_action_semantics):
            if isinstance(value, Unknown):
                return value
        price = {"close": "close", "close_times_factor": "close*factor"}[self.price_basis]
        return f"A[f+{end}]/A[f+{start}]-1; A={price}"


@dataclass(frozen=True, kw_only=True)
class SplitSpec(Contract):
    train: SessionRange | Unknown
    validation: SessionRange | Unknown
    oos: SessionRange | Unknown
    purge: str | Unknown
    maturity_cutoff: str | Unknown
    fit_scope: Literal["train_only"] | Unknown
    sample_alignment: Literal["keyed_security_session"]


@dataclass(frozen=True, kw_only=True)
class FoldSpec(Contract):
    fold_id: str
    split: SplitSpec
    allowed_prediction_interval: SessionRange | Unknown


@dataclass(frozen=True, kw_only=True)
class DatasetSpec(Contract):
    name: str
    key: tuple[str, ...]
    feature_release: FeatureRelease
    label: LabelSpec
    sample_universe: str | Unknown
    sample_filter: str | Unknown
    split: SplitSpec
    rolling_folds: tuple[FoldSpec, ...] | Unknown
    rolling_rule: str | Unknown
    retrain_step_sessions: int | Unknown
    train_window_sessions: int | Unknown
    ordered_model_schema: tuple[Column, ...]


@dataclass(frozen=True, kw_only=True)
class ResourceSpec(Contract):
    threads: int | Unknown
    concurrent_folds: int | Unknown
    memory_limit_mb: int | Unknown


@dataclass(frozen=True, kw_only=True)
class TrainingSpec(Contract):
    name: str
    dataset_name: str
    backend: str
    backend_version: str | Unknown
    objective: str | Unknown
    parameters: dict[str, Any]
    seed: int | Unknown
    preprocessing: str | Unknown
    fit_scope: Literal["train_only", "all_matured_after_selection"] | Unknown
    fit_range: SessionRange | Unknown
    selection_protocol: str | Unknown
    resources: ResourceSpec


@dataclass(frozen=True, kw_only=True)
class SignalInput(Contract):
    alias: str
    label: LabelSpec
    source_stage: Literal["raw_prediction"]
    score_semantics: str | Unknown


@dataclass(frozen=True, kw_only=True)
class SignalNode(Contract):
    name: str
    op: Literal["daily_zscore", "weighted_combine"]
    inputs: tuple[str, ...]
    input_stages: tuple[Literal["raw_prediction", "daily_zscore", "final"], ...]
    output_stage: Literal["daily_zscore", "final"]
    weights: tuple[float, ...]
    reference_universe: str | Unknown
    missing_policy: str | Unknown
    parameters: dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class SignalPlanSpec(Contract):
    name: str
    key: tuple[str, ...]
    inputs: tuple[SignalInput, ...]
    nodes: tuple[SignalNode, ...]
    output: str
    join_policy: Literal["inner_on_security_session", "outer_on_security_session"] | Unknown
    score_semantics: str | Unknown
    available_time_semantics: str | Unknown


@dataclass(frozen=True, kw_only=True)
class SignalEvaluationSpec(Contract):
    key: tuple[str, ...]
    label: LabelSpec
    signal_stage: Literal["raw_prediction", "daily_zscore", "final"]
    metrics: tuple[str, ...]
    reference_universe: str | Unknown
    missing_policy: str | Unknown
    tie_policy: str | Unknown
    maturity_policy: str | Unknown
    aggregation: str | Unknown


@dataclass(frozen=True, kw_only=True)
class StrategyRecipeDraft(Contract):
    name: str
    feature_release: FeatureRelease
    datasets: tuple[DatasetSpec, ...]
    training: tuple[TrainingSpec, ...]
    signal_plan: SignalPlanSpec
    evaluation: tuple[SignalEvaluationSpec, ...]
    core_requirements: CoreCapabilityRequirements
    evidence: tuple[ArtifactRef, ...]
    correctness_blockers: tuple[Unknown, ...]


@dataclass(frozen=True, kw_only=True)
class FeatureBuildIdentity(Contract):
    data_refs: tuple[ArtifactRef, ...]
    view_refs: tuple[ArtifactRef, ...]
    feature_release: FeatureRelease
    scope: SessionRange
    lookback_sessions: int
    pit_policy: str
    cutoff_policy: str
    reference_universe: ArtifactRef
    implementation_package: ArtifactRef


@dataclass(frozen=True, kw_only=True)
class LabelBuildIdentity(Contract):
    data_refs: tuple[ArtifactRef, ...]
    label: LabelSpec
    scope: SessionRange
    calendar_ref: ArtifactRef
    benchmark_semantics: str
    corporate_action_semantics: str


@dataclass(frozen=True, kw_only=True)
class DatasetIdentity(Contract):
    feature_build: ArtifactRef
    label_build: ArtifactRef
    dataset_spec: DatasetSpec
    fit_protocol: str


@dataclass(frozen=True, kw_only=True)
class ModelIdentity(Contract):
    dataset: ArtifactRef
    dataset_spec: DatasetSpec
    training_spec: TrainingSpec
    fold: FoldSpec
    environment: ArtifactRef


@dataclass(frozen=True, kw_only=True)
class ModelReleaseManifest(Contract):
    identity: ModelIdentity
    model_bytes: ArtifactRef
    fitted_preprocessing_state: ArtifactRef
    ordered_schema: tuple[Column, ...]
    inference_package: ArtifactRef
    inference_abi: str


@dataclass(frozen=True, kw_only=True)
class SignalBinding(Contract):
    alias: str
    model: ModelReleaseManifest
    feature_build: ArtifactRef


@dataclass(frozen=True, kw_only=True)
class SignalIdentity(Contract):
    plan: SignalPlanSpec
    bindings: tuple[SignalBinding, ...]
    allowed_dates: SessionRange
    implementation_package: ArtifactRef


@dataclass(frozen=True, kw_only=True)
class SignalRunManifest(Contract):
    identity: SignalIdentity
    output_bytes: ArtifactRef
    key: tuple[str, ...]
    output_stage: Literal["daily_zscore", "final"]
    time_columns: tuple[str, ...]
    validity_columns: tuple[str, ...]

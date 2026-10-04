# Axiom Research R0

Research 0.2.1 exports saved stock stage reports through
`export_stock_stage_report(experiment_path, timing_receipts=(), destination="reports/stage.json")`.
`destination` is a JSON file outside the saved experiment directory. Identical
reports reuse that file; different content cannot overwrite it. Explicit owner
receipt paths can supply current measurements and declared reused Feature timings.
Missing timings stay null. `load_stock_stage_report("reports/stage.json")` verifies
the report's stored hashes and references without upstream imports or calculations.
Neither entry point changes saved models, datasets, predictions or registrations.

Versioned research definitions, manifests and content identity contracts for
Financial RC 60d/180d. Python3.11+. Definition contracts use the standard library;
Data adapters use Core and the optional build runtime uses Data/Arrow.

R0 delivers the question of **what Research specifies and what E0 must execute**.
The Financial RC fixture is structurally valid forensic content. Original
Data/PIT/label/OOS proof obligations stay explicit and block execution admission.

## Public API

Definitions:
`FeatureRelease`, `FeaturePlanSpec`, `LabelSpec`, `DatasetSpec`, `TrainingSpec`,
`SignalPlanSpec`, `SignalEvaluationSpec`, `StrategyRecipeDraft`,
`DataRequirements`, `CoreCapabilityRequirements`.

Output manifests: `ModelReleaseManifest`, `SignalRunManifest`.

Build-request identity contracts: `FeatureBuildIdentity`, `LabelBuildIdentity`,
`DatasetIdentity`, `ModelIdentity`, `SignalIdentity`. These describe identity
requirements; constructing one is not evidence that a build ran or bytes exist.

Supporting types: `ArtifactRef`, `Unknown`, `Column`, `SessionRange`,
`FeatureDefinition`, `Requirement`, `MaturitySpec`, `SplitSpec`, `FoldSpec`,
`ResourceSpec`, `SignalInput`, `SignalNode`, `SignalBinding`, `Contract`.

Functions: `validate`, `unresolved`, `to_dict`, `from_dict`, `dumps`, `loads`,
`load`, `save`, `semantic_identity`, `content_digest`, `contract_schema`.
Validation errors use `ContractError`. The dataclass constructor is a value
constructor; call `validate` to enforce the public boundary. Serialization,
deserialization and identity APIs always validate. Validation returns normalized
immutable dataclasses (including tuple/number normalization); nested free JSON
parameter maps should be treated as values, not mutated in place.

```python
from axiom_research import load, validate, semantic_identity, unresolved

recipe = load("src/axiom_research/fixtures/financial_rc.json")
validate(recipe)                         # draft schema/semantic consistency
print(semantic_identity(recipe))         # definition identity, not ready status
print([u.unknown_id for u in unresolved(recipe)])
validate(recipe, require_resolved=True)  # raises: evidence is incomplete
```

All contracts carry type/version, explicit semantic fields and separate metadata.
JSON transport requires every field, including version/metadata. Unknown fields
or contract versions, duplicate JSON keys, nonfinite values and malformed
references are rejected. `contract_schema(Type)` returns JSON Schema for the
transport shape; `validate` is authoritative for cross-field constraints.

Definitions embed small dependency definitions; materialized inputs use immutable
Refs. This keeps R0 portable without introducing a registry/resolver service.
All joins use `(security_id, session)`; the recipe defines session as feature f,
and declares label-address/decision/availability shifts separately. The ordered
column schema describes the selected Feature input boundary (base or
cross_sectional per column); fitted scaler application remains separate in the
TrainingSpec/ModelRelease. Model schema must match that ordered input boundary.

## Files and verification

- `docs/FORENSIC.md`:20-question findings and version/variant boundaries.
- `reports/feature-map.json`:96 formulas and exact source locations.
- `src/axiom_research/fixtures/financial_rc.json`:complete frozen recipe draft.
- `src/axiom_research/fixtures/label_60.json`, `label_180.json`:effective
  next-session consumer LabelSpecs, distinct from LabelStore's t-based formula.
- `src/axiom_research/fixtures/data_requirements.json`, `core_requirements.json`:
  owner handoffs; `core_cases.json` supplies declarative E0 conformance cases.
- `reports/evidence.json`:source digests, actual pinned model metadata and blockers.
- `docs/IDENTITY.md`:identity, reuse and unresolved-data rules.
- `reports/tests.log`, `reports/REVIEW.md`:executed test and independent review evidence.

From this directory:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Tests contain explicitly synthetic manifest closures; none are model/signal
research outputs. R0-01 exercises every public contract type; R0-02 invalid
contracts; R0-03 semantic identity mutations; R0-04 Financial RC recipe validation;
R0-05 preservation/admission of UNKNOWN values.

The reviewed forensic tool `tools/freeze_financial_rc.py` reads the exact local
legacy source and frozen archive. It verifies HEAD/dirty patch and pinned payload
hashes, writes only definition/evidence files, and refuses differing existing
outputs. It does not import legacy Python or execute feature/model/signal code.
Readiness and the next implementation phase require a separate E0 review.

## Data consumers

`axiom_research.qlib_adapter.QlibView` reads Data's immutable P04 export through
the actual Qlib API. Explicit `activate()` selects the process-global provider;
`read(fields=...)` returns native exported fields and `reference` preserves the
originating Snapshot/query, units, cutoffs, identity mapping and numeric tolerance.
There is no automatic filling, normalization or strategy feature computation.
The direct DataBatch adapter remains the P02/Core path.

Qlib is an optional consumer runtime, separate from R0's standard-library
contracts. Install the Data repository's `requirements-qlib.lock` for the measured
Python3.12/macOS environment, or `axiom-data[qlib]` when resolving another environment.
See [Qlib contract and real tutorials](https://github.com/sinnergarden/axiom-docs/blob/main/docs/qlib-interface.md).

## Joint numeric inputs and persistent builds

`build_joint_features` accepts named `market_queries` and `financial_queries`;
each query supports multiple numeric fields. All daily inputs share a concrete
Snapshot, symbols, exchange sessions, PIT policy and per-session cutoffs, with
explicit membership. Financial queries bind their native endpoint/report type
and an inclusive multi-year report range. Data selects revisions at each cutoff;
each stream selects the latest report period not after the session, including
null/retracted values. Core executes identity, pct_change and asof.

Columns are `alias__field`, plus `alias__field__return` for each market numeric
field. `lag_sessions` counts the supplied exchange sessions; include the intended
lookback. Missing history remains null. The fixed recipe does not support arbitrary
FeaturePlans, TTM derivation, labels, training or a strategy/backtest platform.

Install the `build` extra with the reviewed local Data/Core packages. Pass an
immutable `implementation_package: ArtifactRef` covering actual Research and Core
versions, together with the queries and a Research-owned `destination`. A build
saves `panel.parquet`, `evidence.json` and `manifest.json` atomically. The identity
uses existing FeatureRelease/FeatureBuildIdentity contracts; no registry exists.
Same inputs/recipe/implementation reuse bytes without Data fact reads or Core
execution. `load_feature_build(path).read()` opens an existing panel in another
process. `evidence()` includes per-cell Core reasons, revision/receipt provenance
and normalized query contexts. Integrity failures raise and preserve old files.

Scoped tests cover multi-domain/key alignment, report-period switching, same-period
revision changes, retraction without fallback, unavailable reports, package/query
identity, disk-write failure, corrupt output, relocation and reuse without execution.
Real saved-Data examples and their measured limits live in
[axiom-docs](https://github.com/sinnergarden/axiom-docs). Twelve-year bulk and
many-year feature build performance have not been accepted.

## Saved experiment records and readonly browsing

Research 0.1.4 retains public `ExperimentStore` and `ExperimentReader`. The writer
registers immutable questions, versions and runs referencing existing artifacts;
question group/tag organization and run favorite/shelved updates use an expected
revision and preserve history. `update_run_organization(run_record_ref, ...)`
resolves the registration to the saved account's stable `saved_run_ref` (bound to
Engine `run_id`, `content_digest` and committed sequence). Replacing a saved
evaluation does not create another backtest or lose that account's markers.
`ExperimentReader(path).index(...)`, `.detail(question_id)` and
`.compare_versions(left_ref, right_ref)` only read and validate the small JSON
index. They do not create files, scan artifact directories or compute metrics.
Missing indexes raise FileNotFoundError; missing values remain explicit nulls.
The public Reader import is independent of the optional Data/Core build runtime.
Existing build/loader exports keep their import paths and load on demand.

Default `runs` contains one group per saved account with complete immutable
`registration_history`; its latest matching registration is navigation metadata.
`saved_backtest_count` and `registration_count` distinguish executions from
registrations. Records without an account remain `REGISTRATION_ONLY` items and
do not count as backtests. Version/status filters retain the matching associations
and the complete history. Use `run_favorite` / `run_shelved` for run filters;
the original `favorite` / `shelved` arguments keep their question-level meaning.
Older indexes get false run markers in memory without a disk migration.
Recent questions use `last_activity_at`, the latest saved question/version/run
timestamp, with stable ID ties. Filesystem times and marker edits do not change it.

Engine associations preserve `run_id`, `content_digest` and committed sequence.
An optional saved evaluation must refer to that same account result. Signal
references must match the saved run. Index validation checks metadata consistency;
the actual artifact is verified by its owner's public loader when opened.

`tools/register_rotation_experiment.py` explicitly registers an already saved
ETF experiment, optionally with already saved Engine results. It uses the public
loaders and writes only the specified Research index. It never reads Data facts,
builds features, trains or backtests. The caller supplies the hypothesis,
explanation and declared changes; results do not invent those texts.
It calls `ExperimentStore.register_saved_experiment(question=..., version=...,
run=...)` to save the complete registration atomically, so an invalid reason or
parent cannot leave a partial question/version pair.

The complete [synthetic projection](examples/experiment_projection.synthetic.json)
illustrates the public shape; its references and outcomes are synthetic and do
not locate real research results. The authoritative schema and ownership rules
are in [Research design](https://github.com/sinnergarden/axiom-docs/blob/60c2ec912693703b1b4b94a96bf498489ac3f89f/docs/design/05_axiom_research.md#experiment-records).

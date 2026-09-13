# Identity and cache contract v1

`semantic_identity` = SHA256 of canonical UTF-8 JSON containing contract type,
version and every semantic field. Dictionary keys are sorted; list/column/node
order is preserved. Typed tuple and numeric fields are normalized before hashing.
Parameter dictionaries retain their exact JSON semantics, including any key named
`metadata`; only actual Contract.metadata is excluded.

Contract metadata is exclusively presentation/provenance annotation. Correctness
state belongs in required semantic fields and structured Unknown/Requirement
status, never metadata. Clearing a recipe's summary blocker list does not clear
the unresolved values embedded in its Feature/Label/Dataset definitions.

ArtifactRef identity includes artifact type, immutable ID, referenced contract
version and content digest. URI only locates the bytes and is excluded from
semantic identity. Mutable current/latest refs and path-as-ID are rejected.
Moving files preserves semantic identity; moving a file does not prove its bytes
or declared support range. `content_digest(bytes)` is a separate exact byte hash.

`save` creates a new file exclusively, never overwrites. There is no cache engine,
pointer resolver, DAG invalidation or implicit artifact discovery.

| Contract | Required identity closure | Earliest changed artifact |
|---|---|---|
| FeatureBuildIdentity | Data/View refs, FeatureRelease incl ordered formulas/windows/lags/requirements, requested scope/lookback, PIT/cutoff, reference universe, implementation package | FeatureBuild |
| LabelBuildIdentity | Data refs, effective LabelSpec incl horizon/start/end/maturity/price/missing policies, scope, calendar ref, benchmark/action semantics | LabelBuild |
| DatasetIdentity | FeatureBuild+LabelBuild refs, DatasetSpec incl sample/filter/key/splits/purge/maturity/fold/fit protocol | TrainingDataset |
| ModelIdentity / ModelReleaseManifest | TrainingDataset ref, matching DatasetSpec/TrainingSpec, seed/params/resource config, explicit fold and environment; manifest adds model bytes, fitted state, expected ordered schema and inference package/ABI | Model |
| SignalIdentity / SignalRunManifest | SignalPlan, alias→ModelReleaseManifest+FeatureBuild bindings, each model's fold and allowed interval, requested allowed dates, signal implementation package; run adds output bytes, stage, keys/time/validity columns | Prediction/Signal |

Changes propagate by **passing the changed immutable ref into the dependent
contract**. R0 does not traverse a stored DAG or auto-rewrite dependent refs.
Examples: formula/lag/PIT change→FeatureBuild→Dataset→Model→Signal; label horizon
change→LabelBuild→Dataset→Model→Signal; split/purge change→Dataset→Model→Signal;
training recipe/seed/environment change→Model→Signal; fold or allowed prediction
dates change→Model/Signal; blend weights change→combined Signal only. An unrelated
label or immutable FeatureBuild remains reusable. Inference/Core implementation
changes are explicitly part of the ModelRelease/Signal identity closure.

## Admission versus reuse

Definition identity can be calculated for an explicitly unresolved draft.
`validate(..., require_resolved=True)` rejects any Unknown or required requirement
whose status is UNKNOWN. Build identity contracts and output manifests enforce
this automatically, including nested dependencies. Thus draft validity and R0
test success cannot manufacture certified Data, a trained model or OOS evidence.

Future R1 artifact readers must independently verify the manifest and payload
digests, reference closure, expected producer/type/version, actual keys/schema,
scope/coverage, temporal availability and terminal validator result. In
particular, a declared TrainingDataset ref must resolve to the same DatasetSpec,
and a Signal binding's FeatureBuild must resolve to the model's FeatureRelease.
R0 only defines that immutable closure; it does not read those large tables or
claim that a caller-supplied digest proves semantic correctness.

Identical identity + verified complete output closure permits exact reuse.
Filename, directory name, mtime, pointer position, zero exit status or equal row
count never suffices. Superset/prefix reuse requires a separately declared and
validated projection contract; none is supplied by R0. Output serialization
differences may change byte digest without changing definition identity.

## Fit and time boundaries

Dataset split dates are inclusive ISO sessions with ordered, disjoint train,
validation and OOS ranges. Every explicit fold prediction interval lies within
its fold OOS and overall Dataset OOS. A model's train_only fit interval must lie
within the fold training partition; an explicitly declared serving refit may
use all matured rows after selection. Fit end must precede allowed predictions.
The LabelSpec defines the full future target interval and maturity rule; R1 must
validate actual session/calendar and outcome availability for every admitted row.
An ISO-date ordering check alone is not a trading calendar or target-maturity
validator. Those missing original proofs keep Financial RC's draft blocked.

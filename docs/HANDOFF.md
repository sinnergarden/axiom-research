# Axiom Research R0

## 1. Forensic findings

FACT:96 ordered Feature columns and per-column source formulas are frozen;
73 outputs are base and23 are cross_sectional. Financial RC's pinned60d/180d
model/scaler bytes were verified against metadata. Metadata explicitly marks
historical training rows as current-CSI800, not PIT, and validation as a purged
holdout rather than full rolling OOS.

FACT: inspected LabelStore target is adjusted close `A[t+H]/A[t]-1`, where
`A=close*factor`; Financial RC trainer addresses the label at next_session(f),
so the effective target is `A[f+H+1]/A[f+1]-1`. The60d/180d canonical blend uses
unclipped per-horizon daily zscore, ddof0, then exact0.5/0.5 weights.

FACT:504 matured feature sessions, trailing40-session validation, H+1 purge,
early-stop tree selection and full matured serving refit. Historical rolling
studies separately used20-session steps and their own normalization stages.

UNKNOWN: original full Data/label/sample/OOS closure and future strict-PIT
admission.24 explicit unresolved obligations remain. Detailed20-question findings,
classifications, version differences and numerical anchors are in FORENSIC.md.

## 2. Implemented contracts

Version1: FeatureRelease, FeaturePlanSpec, LabelSpec, DatasetSpec, TrainingSpec,
ModelReleaseManifest, SignalPlanSpec, SignalRunManifest, SignalEvaluationSpec,
StrategyRecipeDraft, DataRequirements, CoreCapabilityRequirements.

FeatureBuildIdentity, LabelBuildIdentity, DatasetIdentity, ModelIdentity and
SignalIdentity define future artifact/cache identities. Public API includes
load/save, dict/JSON roundtrip, validate, unresolved, contract_schema,
semantic_identity and content_digest. Standard library only.

## 3. Frozen Financial RC recipe

```text
fixed public Data/View + calendar/units/PIT requirements (bindings unresolved)
→96 ordered legacy Feature definitions + per-column policies/source identity
→60d /180d next-session adjusted-close absolute-return LabelSpecs
→keyed Dataset; declared purged validation and unresolved original OOS
→LightGBM regression;504 matured sessions/40-session selection;
  median/MAD scaler, clip3, selected-tree full matured refit
→pred_60/pred_180 → daily zscore(ddof0, no clip) →0.5*z60+0.5*z180
```

Packaged financial_rc.json has definition identity:
`sha256:a74df99ae7644a4ff9f46a73fabbad5b25c6e320d1f2f18cdf69e031f5f41cbe`.
It passes draft contract validation and rejects execution admission while
unresolved obligations remain. No actual ModelRelease or SignalRun is fabricated
for these incomplete historical proofs; manifest roundtrip examples are synthetic.

## 4. Identity / cache rules

Formula/lag/PIT change begins at FeatureBuild; horizon/label policy at LabelBuild;
sample/split/purge at Dataset; training recipe/seed/environment at Model;
fold/allowed interval/inference implementation at Model/Signal; weights at combined
Signal. Updated dependency refs carry the change downstream. Display metadata and
locator relocation preserve semantic identity; exact payload bytes retain their
own digest. Cache reuse requires verified immutable manifest/payload closure.

## 5. CoreCapabilityRequirements

12 requirements, each with required_by, input/output semantics, edge cases and
a declarative reference fixture: key/schema/time validation; grouped shift and
finite rolling; arithmetic/missing primitives; explicit reference-CS transforms;
parameterized zscore; quantile winsorization; availability/as-of alignment;
frozen plan/plugin ABI; ordered schema and frozen preprocessing application;
saved-model inference ABI; keyed signal normalization/blend; deterministic replay
and semantic identity. Quarterly/TTM source preparation can be satisfied by Data's
stable facts. These are E0 inputs, not implemented operators.

## 6. Data blockers

Bind public immutable views with actual field units, continuous historical union
and calendar, price/factor/action and delisting coverage, financial/holder revision
visibility, membership/industry reference sets. Reconcile original label/sample
and model/fold evidence before historical OOS admission. Code formula and file
hash evidence cannot certify those claims. No other repo requires an R0 code edit.

## 7. Tests

| Acceptance | Result |
|---|---|
| R0-01 all public contract roundtrips | PASS |
| R0-02 invalid keys/schema/horizon/maturity/splits/model/stages | PASS |
| R0-03 identity mutations and metadata/locator invariance | PASS; strengthened lag/split targeted rerun PASS |
| R0-04 real96-column Financial RC frozen fixture | PASS as unresolved draft |
| R0-05 UNKNOWN preservation and fail-closed artifact admission | PASS |

Luna executed the tests after independent Astra code review. Evidence is in
reports/tests.log and tests-identity.log. Independent final review is recorded in
reports/REVIEW.md. Readiness for R1 remains blocked by the declared evidence and E0
requirements; this handoff completes R0 definitions and forensic scope only.

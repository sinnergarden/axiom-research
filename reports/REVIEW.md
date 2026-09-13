# Independent Research R0 acceptance

Run: `research-r0-20260913`  
Reviewer: independent GPT-6 Astra reviewer  
Date: 2026-09-13  
Decision: **PASS_R0**. No remaining blocking finding for the definition,
forensic evidence and contract deliverables. Parent task must deliver the
user-facing handoff before closing its persistent goal.

## Code gate and execution evidence

Reviewed the public contract definitions, serializer/validator/identity API,
package export, frozen-fixture authoring tool and R0 tests. The implementation
contains no Feature executor, inference executor, signal runtime or backtest.
The freeze tool reads source/config/artifact bytes and writes only declarative
fixtures and evidence inside axiom-research. It checks the reviewed source HEAD
and dirty patch, verifies pinned model/scaler hashes, and refuses to replace
differing output files.

Blocking review findings were corrected before successful freezing: explicit
keyed signal joins, named Dataset/Training association, canonical numeric and
container handling, fold OOS containment, UNKNOWN requirement admission,
inference/Core implementation identity, train-only fit-range containment,
accurate Feature stages and complete forensic source inventory. The first freeze
attempt found a typing.Union compatibility error before writing outputs. The
reviewed decoder/schema fix recognizes both Python union representations.

Luna executed the reviewed tool and tests. `freeze.log` records a structurally
valid draft with execution readiness BLOCKED. `tests.log` records all five
R0-01 through R0-05 test groups passing. After review strengthened the isolated
lag and actual split-date identity cases, Luna reran R0-03 successfully in
`tests-identity.log`; the original full-suite log remains intact. This reviewer
did not rerun tests or execute legacy calculations.

## Independent terminal artifact inspection

Acceptance includes direct JSON inspection, source reads and byte-digest
comparison, rather than relying on the process exit code.

| Deliverable | Verified result |
|---|---|
| `docs/FORENSIC.md` | All twenty questions have evidence-scoped classifications; serving, rolling and CSI1800 R3 families remain distinct. Inspected source definitions are not presented as proof of original-run correctness. |
| `reports/feature-map.json` and packaged `financial_rc.json` | 96 ordered names, formulas and input lists match; the fixture contains 73 base and 23 cross-sectional columns. No formula is replaced by an invented placeholder. |
| `label_60.json` and `label_180.json` | Exactly equal to their recipe-embedded LabelSpecs. Effective consumer formulas are A[f+61]/A[f+1]-1 and A[f+181]/A[f+1]-1, with explicit next-session start, adjusted-price basis, maturity and preserved lineage uncertainty. |
| Dataset and Training definitions | Two explicit dataset-name bindings; saved fit ranges and selected 21/91 iteration anchors match model metadata. Evaluation training-start and historical OOS remain Unknown. Train-only evaluation and full matured serving refit are separately described. |
| Signal and evaluation definitions | Explicit security/session keys and raw→daily-zscore→final stages; ddof 0, no clipping and equal 0.5/0.5 blend. Evaluation is label-conditioned research statistics, with universe qualification unresolved. |
| `data_requirements.json` | Exactly equals the embedded contract; six dependency groups cover identity/calendar, market/valuation, financial, margin, shareholder and membership/industry, with relevant edge cases. |
| `core_requirements.json`, `reports/core-map.json`, `core_cases.json` | Packaged requirements equal the embedded contract. All twelve capability names have corresponding declarative conformance cases. Numeric examples and key/schema/missing/time cases are specifications for E0, not executed E0 results. |
| `reports/evidence.json` | All 47 recorded source content digests match the referenced local bytes at review time. Pinned model metadata and current source are identified separately. Rolling manifest count/hash declarations and R3's 13 blocking certification exceptions agree with the cited source records. |
| `README.md` and `docs/IDENTITY.md` | Public entrypoint, serialization, identity closure, reuse conditions, fit boundaries and the responsibilities of future artifact readers are documented consistently with R0 scope. |
| `docs/HANDOFF.md` | All seven requested handoff sections are present. Formulas, stages, model/validation distinctions, identity, twelve Core requirements, Data blockers and five passing test groups agree with the inspected artifacts and logs. Completion is explicitly limited to R0. |
| `PLAN.md` | Records the validated artifacts and tests; leaves final user delivery with the parent task. No background job or bulk build is required to complete this R0 handoff. |

The independent recipe identity calculation, using canonical JSON with only
Contract metadata and ArtifactRef locators removed, produced:

`sha256:a74df99ae7644a4ff9f46a73fabbad5b25c6e320d1f2f18cdf69e031f5f41cbe`

This agrees with the actual evidence file and freeze log. The recipe file's
separate byte digest is:

`sha256:0fd1d5b86fc3775db502854cdc238914934e3a89dbe79d1ba5c0efd2518ec3ad`

## Scope of acceptance

The draft preserves **24 distinct unresolved obligations**, including Data
closure, historical PIT, Label lineage, OOS lineage, E0 ABI, evaluation training
bounds, event-history lookbacks and resource/retraining declarations. These
prevent admission to formal builds/runs and require evidence or explicit
decisions before R1. They are the intended R0 discovery output and do not leave
the R0 evidence/specification task unfinished.

Original pinned model/scaler bytes were inspected as forensic anchors. Saved
prediction row counts and payload hashes were checked as manifest declarations;
large prediction tables were not replayed, and their full payload correctness
is not certified here. No original strategy alpha, historical PIT baseline,
OOS performance or real execution readiness is approved by this review.

R0 validators enforce declarative type, schema, key, split/fit bounds and unknown
admission rules. R1 readers must still resolve immutable references, verify
payloads and actual support ranges, and prove per-row calendar/maturity and
availability conditions. Core capability cases remain unexecuted until E0.
The next implementation phase is Engine/Core E0 under its own review gate.

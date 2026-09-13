# Research R0 — run research-r0-20260913

Objective: freeze Financial RC 60/180 evidence and research definitions, implement
small versioned contracts and identity rules, deliver requirements for Data and
Engine/Core E0. Source evidence is read-only. No data build or research execution.

## Definition of Done

- All 20 forensic questions have evidence-scoped FACT/INFERENCE/UNKNOWN/DECISION
  classifications, source locations and digests. Correctness unknowns remain explicit.
- Public definitions and manifests roundtrip, reject invalid semantics, bind immutable
  identities and distinguish draft validity from readiness for a build/run.
- Financial RC fixture captures the evidenced recipe without inventing missing
  baseline semantics; Data/Core requirements link to actual recipe dependencies.
- R0-01 through R0-05 pass under Luna execution after independent Astra code review.
- Independent Astra terminal review checks source evidence, fixtures, contracts,
  test results and final handoff. Only axiom-research receives changes.

## Stages

1. Design/scope and existing work discovery — complete. Read design 01–06 and patch A;
   no axiom-research implementation or live R0 job existed. graphify-out has no graph.
2. Financial RC forensic — complete.96 ordered formulas, b242 model anchors versus
   c969 definition/dirty source and rolling variants recorded; original historical
   PIT/OOS/data closure limitations preserved. reports/evidence.json binds sources.
3. Contracts, frozen recipe, identity and Data/Core requirements — complete.
   Packaged financial_rc.json:96 columns,24 distinct unresolved obligations,
   definition identity sha256:a74df99ae7644a4ff9f46a73fabbad5b25c6e320d1f2f18cdf69e031f5f41cbe.
4. Independent code review, fixes and Luna tests R0-01–05 — complete.
   All5 groups PASS; first freeze caught typing.Union compatibility before output;
   reviewed fix passed retry. Logs: reports/freeze-attempt1.log, freeze.log, tests.log.
5. Independent artifact/deliverable validation and user handoff — complete.
   Independent Astra PASS_R0 in reports/REVIEW.md:47 source digests, standalone
   fixture equivalence,96 ordered formulas,73/23 stages,12 Core case references,
   independently calculated identity and all test logs verified. Seven-section
   docs/HANDOFF.md independently accepted and delivered to user.

Status: R0 terminal outcome complete.24 explicit proof obligations block subsequent
formal execution/R1, not R0 discovery delivery. NO_BULK_BUILD. No detached jobs.

Use case: Research R0 definition discovery. Public entrypoint: axiom_research
contract load/save/validate/identity API (new package explicitly requested).
Inputs: design documents and read-only legacy source/config/artifact evidence.
Outputs: contracts, JSON fixtures, forensic report and handoff requirements.
Lookahead: draft unknowns may serialize but cannot admit build/run artifacts.

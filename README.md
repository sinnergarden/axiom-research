# Axiom Research R0

Versioned research definitions, manifests and content identity contracts for
Financial RC 60d/180d. Python3.11+, standard library only.

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

# Independent R0 contract correctness review

Reviewer: independent GPT-6 Astra
Baseline: `2699fbc`
Repair branch: `fix/r0-contract-correctness`
Date: 2026-09-13

## Code gate

**PASS_CODE_GATE** for the three requested contract repairs. No remaining
blocking code finding within this scope. This gate is not merge approval.

Reviewed the full contract definitions and serialization/validation/identity API,
public exports, existing freeze entrypoint, R0-01–08 tests, changed identity and
forensic documentation, and the narrowly scoped temporary fixture migration.
The migration changes three packaged definition fixtures and their evidence
identity; the canonical freeze tool must independently reproduce those bytes.
It imports no legacy execution code and preserves the existing source and
payload guards. No bulk build, executor, Engine/Core, backtest or merge is part
of the reviewed action.

| Finding | Reviewed repair and counterexamples |
|---|---|
| Nested Unknown laundering | Encoding and decoding recurse through Any dictionaries, lists and embedded tagged contracts. The unresolved walker recognizes wire tags. Malformed tags fail closed; ModelIdentity validation, loading and identity computation reject nested blockers. Tests include JSON/YAML roundtrips, nested specs, plain parameter keys named metadata and equal identities for typed versus tagged Unknown. Actual Contract metadata remains presentation-only by the documented contract. |
| Multiple label authorities | Version 2 LabelSpec accepts horizon and enumerated start/end, price, benchmark and action rules. Formula, start/end offsets, target interval and target-end maturity lag are derived properties, absent from transport input. MaturitySpec version 2 adds qualification without an independent lag. Old or contradictory authorities and inconsistent price/action combinations are rejected. Tests cover the 60→59 horizon mutation and injected old fields. |
| Missing feature PIT semantics | Version 2 FeatureDefinition requires PITPolicy with publication, dependency availability, report-period updates, exact-date matching and original materialization. Fields are required, validated and hashed. Ambiguous source-mode-dependent matching requires Unknown materialization, so an artifact reference alone cannot promote it to a resolved build. Tests cover changed identities, missing/blank/invalid values and unresolved FeatureBuild admission. |

Reviewed code SHA-256 values:

| File | SHA-256 |
|---|---|
| `src/axiom_research/api.py` | `29101397af953bf2cd250666079525ae4decce9ef6510ee0b18f14186fb1c4fd` |
| `src/axiom_research/contracts.py` | `149c22d1793d4b84a58b065d6e383c1f88210655033e0996a998d71b5bfec357` |
| `tests/test_r0.py` | `f9bec76dec59804b749b22d32728aead969ad467f3beb75ccb923a9207c6d90b` |
| `tools/freeze_financial_rc.py` | `e5c90335e1e16282bd20c0a761a49780b668ddc3dd0a432ed59ad461d4889150` |
| Temporary reviewed fixture migration | `13583176c386f7bc5e2775bf646d3f7d2f6dfbf315e35d38ed5aff0506ae12e4` |

## Execution and terminal validation

**PASS_ARTIFACT_REVIEW**. Luna ran the reviewed fixture migration, all eight
R0-01–08 test groups and canonical freeze validation. Directly inspected
`tests-correctness.log`: eight groups passed in 3.266 seconds. The canonical
freeze log reports 96 columns, 24 Unknowns, structural PASS and execution BLOCKED.
The reviewed source/test hashes above still match the executed working tree.
The reviewer did not rerun tests or any legacy calculations.

Independent artifact inspection used JSON bytes and a separate canonical JSON
identity calculation, without importing the changed package:

- All 96 feature definitions retain their baseline nonmigration fields and now
  have version 2 plus required PITPolicy contracts.
- Both standalone label fixtures equal the recipe-embedded labels. Their
  structured rules derive intervals `(1, 61)` and `(1, 181)`. Formula/offset input
  fields and independent maturity lag are absent.
- Standalone Data and Core requirements equal the embedded definitions.
- All 24 distinct Unknown definitions are exactly preserved against `2699fbc`,
  including IDs, reasons and required evidence. Reordering in the evidence
  inventory reflects their earlier occurrence in the new feature PIT fields.
- Pinned evidence outside recipe identity and Unknown ordering is unchanged.
  All 47 recorded source byte digests still match their local referenced files.
- The independently computed recipe identity agrees with the evidence and
  canonical freeze log:
  `sha256:e7bee37026fdd935c2e154481ae524b425c769cbd01ee623ac901b17246e9d24`.

| Artifact | Byte SHA-256 |
|---|---|
| `financial_rc.json` | `410fb38ed9d03574c7fc5efee32358c28c6d1f1d0202e422095e11590b5fc7c2` |
| `label_60.json` | `e8df2ad881f4ddf1c79b1c2ec27422323fdc14a09d240456d1794b0c3c22e272` |
| `label_180.json` | `d4f1a16624a876da5b69429591f00c2a68c2be259b1760a0d9b085e9273a760a` |

Committed content and the remote repair-branch tip remain to be independently
verified after publication. Historical `reports/REVIEW.md` remains an account
of the original acceptance; this follow-up supersedes its three contract
correctness conclusions. Historical PIT, Data/Label/OOS lineage and execution
readiness remain blocked by the preserved proof obligations.

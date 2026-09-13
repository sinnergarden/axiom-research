# Financial RC R0 forensic

Scope: 2026-09-13, definitions and evidence only. Primary source checkout is
`/home/liuming/.openclaw/workspace/SysQ`, HEAD
`c969c74a66b148fa66396c3896e48760004683d4`, plus captured working changes.
Frozen config/artifact archive is
`../data/forensic/sysq-c969c74-20260904`. Content digests and exact source
locations are recorded in `reports/evidence.json`; feature formulas are in
`reports/feature-map.json` and the packaged recipe fixture.

FACT means a statement about the inspected code/config/bytes, not a claim that
those bytes establish historical correctness. INFERENCE is an interpretation;
UNKNOWN is an unresolved proof obligation; DECISION is an R0 contract choice.
Design 01–06 and patch A were read; patch A governs conflicting boundaries.

## Baseline families must remain distinct

FACT: `configs/strategies/financial_rc.yaml:46–141` pins the canonical 60/180
screening model pair, equal weights and `daily_cs_zscore_unclipped_ddof0`.
The selected model IDs are `e75f67aee783d789` (60d) and `b918966fd9fab47b`
(180d). Their metadata says source commit
`b242fbe5533d409b58bea12ffebfcc549ed0f5b1`, clean. That is not the inspected
c969c74 checkout. Model/center/scale/meta bytes are forensic anchors, not a
certified historical OOS pair. The named bundle's separate JSON was not found.

FACT: rolling v3 60d/180d configs specify 2021-01-01–2026-07-31, 504 feature
sessions, step 20, 300 maximum boosting rounds, maturity lag 61/181. Saved
experiment manifests record 68 windows and code identifier `c9b3a514`.
The `_pit` variants use `csi800_pit_union` and `csi800_pit_v2`; the unsuffixed
v3 uses `csi800`. CSI1800 terminal R3 is a separate 180d study, not a paired
60/180 baseline. Its certification report explicitly remains BLOCKED by 13
source-capability exceptions.

DECISION: R0 freezes the current canonical equal-weight definition, preserving
historical lineage/OOS unknowns. Rolling findings are a separate variant record;
no rolling results are assigned to the pinned serving models.

## Twenty required questions

| # | Classification | Finding and evidence scope |
|---|---|---|
| 1 | FACT | Frozen `configs/features/v3a_plus_liquidity_financial_rc.yaml:4–99` gives 96 ordered names, retained verbatim, including `$` prefixes. YAML SHA256 `740786fd59167cd913fefbabe3ee1db7650109664653f0ad82533d5bb7387fb9`. Actual model metadata carries its own ordered list; adapter aliases need explicit mapping. |
| 2 | FACT | Raw field passthrough plus formulas in `qsys/feature/groups/{fundamental_context,relative_strength,value_growth_v3a,liquidity,growth_confirmation_v0}.py`; per-column formulas and inputs are frozen. Fiscal/announcement operations are not implied by a feature's name. |
| 3 | FACT / UNKNOWN | Direct lag reaches 756 daily rows; nested windows must include dependency history. Adapter pads 1461 calendar days (`adapter.py:53,379–384`), which does not prove dense trading-session history. Research margin lag defaults to 0; canonical inference snapshot lag is 1. Source-specific publication/availability differs. |
| 4 | FACT | Zero denominators generally become NaN; composite features locally fill 0 or .5. Six selected liquidity columns are winsorized in place at daily reference 1%/99% (`builder.py:121–142`, `transforms.py:57–99`). Model input fill0 + float32, then robust transform. `is_profitable_ttm` maps missing TTM to 0. No universal base-layer fill or zscore. |
| 5 | FACT / UNKNOWN | `_pit_member` controls CS; absent mask means all loaded rows (`transforms.py:10–39`). PIT research loads union continuous history before filtering; canonical saved models record `current_constituents_snapshot`. Original historical membership/industry/calendar closure and PIT certification are UNKNOWN. |
| 6 | FACT / UNKNOWN | LabelStore formula is `A[t+60]/A[t]-1`; generator/trainer pairs feature f with LabelStore key next_session(f). Therefore inspected consumer target is `A[f+61]/A[f+1]-1`. Original run's exact consumed label bytes/producer lineage remain UNKNOWN. |
| 7 | FACT / UNKNOWN | Analogously 180d store formula `A[t+180]/A[t]-1`; inspected consumer target `A[f+181]/A[f+1]-1`. Same historical lineage limitation. |
| 8 | FACT | Start is next session close relative to feature f, end is H sessions after start in the inspected generator/trainer. This is a close-to-close proxy, not next-open execution return. `lightgbm_single_label.py:1265–1285`; `financial_rc_trainer.py:106–176`. |
| 9 | FACT / UNKNOWN | Current label implementation uses A=Qlib `$close * $factor`, casts float32 (`qsys/label/compute.py:115–201`). Original consumed price/factor snapshot, units and adjustment vintage are not proven merely by this formula. |
| 10 | FACT | Raw label is absolute forward adjusted return; no benchmark subtraction and no label zscore. “raw” denotes no normalization, not unadjusted price. |
| 11 | FACT / UNKNOWN | Target end is f+H+1. Rolling lag H+1 retreats from the session before prediction, giving H+2 calendar-session separation; trainer also requires H+2. Calendar helpers contain business-day fallbacks; whether original runs used them is UNKNOWN. Axiom requires explicit actual calendar and availability proof. |
| 12 | FACT / UNKNOWN | Canonical trainer reserves last 40 feature sessions for evaluation, purges earlier training and refits serving on all 504 matured sessions. Rolling current code uses complete label-date boundaries near legacy trailing 15% capped at 20k rows; its historic code predates current dirty fixes. No pinned-model allowed historical OOS interval is established. |
| 13 | FACT | Canonical evaluation train-end index = validation-start index − H − 2; H+1 intervening feature sessions purged (`financial_rc_trainer.py:179–197,975–1053`). Rolling maturity embargo is not evidence of equivalent validation-boundary purge. |
| 14 | FACT / UNKNOWN | Rolling windows step/predict=20, terminal partial window retained, train window 504 (`rolling_window.py:31–115`). Saved 68-window manifests are count anchors; actual validated per-fold sample-key closure and correspondence to pinned models remain UNKNOWN. |
| 15 | FACT / UNKNOWN | LightGBM regression/MSE, maximum300 rounds, early stopping20 in inspected trainer. Exact defaults and model metadata anchors are recorded in evidence; source/environment differs by historical artifact, so no implicit backend-version or seed defaults. |
| 16 | FACT | Fill0/float32; center=median, scale=median(abs(X−center)), scale0→1; transform clip±3 then fill0. No 1.4826 MAD correction. Evaluation scaler fit on pre-validation train, serving refit scaler on all matured fit rows (`alpha_v1/labels.py:17–24`, `training.py:254–379`). |
| 17 | FACT | Booster output is regression score from the absolute-return target, preserved as raw prediction. It is neither a calibrated probability nor a promised return; original training correctness still needs separate evidence. |
| 18 | FACT / UNKNOWN | Canonical inference zscores each horizon over that day's eligible scored cross-section, ddof0, no clipping. Rolling generator zscore clips±3; additional pipeline daily_zscore must be tracked as a distinct stage. Neither reference set may be inferred from the name CSI800 alone. |
| 19 | FACT | Canonical pinned config uses exactly0.5/0.5 after horizon-wise daily zscore. Other legacy workflows had different weights; no claim that all historical Financial RC runs used equal weights. |
| 20 | FACT / UNKNOWN | Existing pinned model/scaler/meta bytes and saved signal manifests provide forensic anchors listed below. No fully certified, reproducible paired historical Financial RC baseline is established by this inspection. |

## Forensic numerical anchors

FACT (metadata claims, not recomputed predictions): 60d selected21 trees,
training 2024-04-11–2026-05-13, purged48,450 rows /61 sessions. 180d selected91
trees, training 2023-10-13–2025-11-10, purged136,113 rows /181 sessions.
The training dates describe the full matured fit window, not a disjoint
evaluation training partition. Model SHA256 values are
`cc6f4febcb7d2e2f67339322f5ac05eb1c70d03540a09aa8eba94e3ddc30313c` and
`7e0450168ecd7d908dd75ffd54bfec971be9d24dc0969b80a7acdf0fb2d66fef`.

FACT: saved unsuffixed v3 signal manifests each declare 1,025,136 rows;
predictions SHA256 values are
`4d7f851073342b571f3b6e84511cab9452dde330518fed955b42c65cfa2bf39b`
and `0e63cf0894691eb3908a19d6ab91a7cb85e35fbd0376e047b4cbcd9779d66afb`.
CSI1800 terminal R3 separately declares2,431,524 rows and prediction SHA256
`52f1a9bcc6980cfc804e69da9d7d2061df90f3706df2a1b2f12839472eb588ba`.
Counts/hashes bind evidence, not alpha, PIT or OOS validity. No backtest metrics
are promoted as R0 quality evidence.

## Preserved proof obligations

- UNKNOWN DATA_CLOSURE: fixed public Data/View refs, required fields/units,
  actual source revisions, close/factor and effective calendar closure.
- FACT: pinned metadata explicitly says current-CSI800 historical rows are not PIT.
  UNKNOWN PIT_BASELINE concerns qualification for a new historical Research run,
  including membership/industry and financial/holder revision-specific visibility.
- UNKNOWN LABEL_LINEAGE: actual label-producer version, dense session alignment,
  adjustment and missing/delisting coverage for the original models/runs.
- UNKNOWN OOS_LINEAGE: sample-key manifests, original split/fit/allowed prediction
  dates and original code plus environment, linking models to actual OOS outputs.
- UNKNOWN CORE_ABI: compatible frozen E0 implementation package does not yet exist.

INFERENCE: missing→0 in some legacy features can turn unavailable evidence into
an economic-looking value. New policy decisions must create a new definition
identity and separate comparison, not silently repair the frozen forensic recipe.

DECISION: Axiom joins use `(security_id, session)`, with session=feature session;
next-session label addressing and signal availability are separately declared.
Legacy `instrument/trade_date/data_date` aliases need a validated adapter.
DECISION: preserved UNKNOWN values are legal draft content but cannot enter
FeatureBuild/LabelBuild/Dataset/Model/Signal artifact identity admission.

## Legacy test evidence and version comparison

FACT: inspected tests include `tests/model/test_financial_rc_trainer.py:62–100`
(maturity and purged target span), `:400–419` (pre-validation scaler and full
matured refit), and `tests/label/test_label_compute_pit_continuity.py:107–120`
(forward horizon must not jump across membership gaps). They support the stated
code intent; these legacy suites were not executed as R0 acceptance tests.

FACT: read-only `git show b242fbe:...` confirmed the pinned-model trainer's
next-session label pairing, H+2 maturity, H+1 purge, trailing40-session validation
and full matured serving refit. b242 alpha training has the same regression
defaults and median/MAD scaler. Comparing b242 to c969 shows added PIT masks for
cross-sectional formulas and audited income availability propagation, so source
formula similarity does not establish equivalent original feature materialization.

## Version 2 definition migration

DECISION: the packaged LabelSpec now stores enum start/end, price, benchmark and
action rules. Its formula, offsets and target-end maturity lag derive from horizon;
legacy formula text is retained only as evidence metadata. Maturity availability
and original label lineage remain explicit proof obligations.

FACT: the feature map records source publication rules for native provider fields,
shareholder announcements and income features. Inspected income code propagates
the maximum availability of each feature's dependencies, emits only newer report
periods per availability stream, and allows exact-date matches only in legacy mode.
The required per-feature PITPolicy preserves these scoped facts and both income
source modes. Fields absent from the feature map retain DATA_CLOSURE; original
materialization retains PIT_BASELINE. This migration does not establish which mode
produced the pinned models and does not resolve any of the existing 24 Unknown IDs.

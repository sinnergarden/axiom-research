# QSYS industry migration

The 23 industry inventory rows use SW L1 supplier intervals. They are explicitly
**best effort terminal history, not publication-time PIT**. `index_member_all`
returns L1/L2/L3 plus `in_date` and `out_date`; it does not supply a full public
revision clock or certify the membership row's historical taxonomy vintage.
References: https://tushare.pro/document/2?doc_id=335 and
https://tushare.pro/document/2?doc_id=181.

`Data.industry_intervals(snapshot=..., security_ids=...)` reads latest native
terminal revisions inside a pinned Snapshot, preserving actual receipts.
`extend_industry_feature_source` uses [in_date, out_date) as an explicit boundary
assumption. Missing intervals, invalid intervals and simultaneous distinct L1
codes remain unknown. Multiple rows with the same L1 are not a conflict at L1.
No current label is extended before its dated interval. Numeric `industry_code`
is the native code's digits, a categorical identifier with no ordinal meaning.

The industry input's selected interval evidence binds reference dependency
receipts. Changing membership/industry requires recomputation of affected CS
features through FeatureLibrary.ensure. `prepare_qsys_inputs` accepts the pinned
`industry_snapshot`; repeated preparation reuses existing bound industry inputs.

Industry rolling means operate on industry/session aggregates, then map back to
current stock membership. They never roll through a stock's changing labels.
Core's `rolling_industry` accepts inclusive industry `cs_mean` inputs. The
`stock_industry_ret_corr_60d` recipe retains QSYS's uncentered cosine formula,
not Pearson correlation. Return predicates preserve missing; the original
explicit top-stock return fill-zero remains in that recipe. These differences
from the old static-label pipeline are recorded as semantic version 2.0.0.

Two source inventory rows are retained for traceability but excluded by user:
`days_since_252d_high` and `margin_eligible`. Native margin balance is already
available and does not imply an eligibility label.

`tools/prepare_qsys_history.py --config <private-config.json>` stages existing
Raw offline into an independent candidate, prepares bounded physical inputs,
then calls the existing incremental FeatureLibrary for raw and normalized
layers. It never trains or collects supplier data. The private config pins the
scope receipt, source roots, parent Snapshot, benchmark and resource limits.
Output progress, domain/source checkpoints and FeatureView receipts remain on
disk. Missing inputs and resource limits are not completion evidence. Historical
cache acceptance requires examining the emitted coverage and missing reasons.

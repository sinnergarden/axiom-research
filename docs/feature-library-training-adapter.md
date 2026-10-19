# Feature library training adapter

The public workflow and evidence are maintained in
[Axiom Docs: Feature lifecycle](https://github.com/sinnergarden/axiom-docs/blob/main/docs/design/feature-lifecycle.md).

`build_feature_library_training_view` publishes a native binding consumed by the
existing `StockFeatureView` and batch owner. Selected FeatureView columns are
projected in memory; no permanent training matrix is copied. Publication checks
source compatibility once. Cold loading verifies the binding, Feature files and
source membership, without Data or full source ancestry admission.

`export_feature_library_training_projection` explicitly writes the old daily
interoperability format. Existing projection v1 files retain their original full
comparison loader. Both paths preserve clocks, validity and membership. The v3
model producer still requires normalized columns; raw is not implicitly normalized.

The version ledger records the candidate's one E2E outcome. Historical synthetic
tests and T01 timings do not validate this consumer or full-history capacity.


## Model missing values and independent labels (2026-10-18)

The saved-library training profile keeps finite physical payloads and validity masks.
Only model X/P projections replace invalid cells with NaN; observed zero remains zero.
A model row needs known, complete, PIT-visible research membership and at least one
valid selected feature. Every valid feature still needs finite value, lineage and a
visible clock. All-missing rows are excluded; damaged axes and future clocks fail
admission. Prediction uses the same rule without future labels.

Label cross-sectional normalization uses research membership, valid raw returns and
the configured maturity rule only. Feature subset or missing cells cannot change y.
Training uses one ordered intersection of valid normalized labels and model-usable
rows; X, y, keys, saved selector and cold loading share it. The binding v3 records
`target_validity_mask` separately from the training selector. New cohort and dataset
identities invalidate affected labels/folds/models without invalidating raw returns
or unrelated Feature partitions. Historical model migration is not provided.

QSYS v5 changes only TRD006 (3.0.0). Its virtual daily status preserves explicit
event unknowns. An absent event with a PIT-visible positive observed share volume
is evidence of non-full-day suspension. Known full-day suspension yields raw zero
through Core `where`, even without close. Unknown remains missing. Only TRD006
uses outer `constant=zero`; a normalized zero is distinct from the raw score zero.
Use existing `ensure_features` for that column, compose it with the other saved raw
columns, then use `normalize_saved` and the native training-view publication.
No other raw column needs recomputation. Old v4 remains installed for saved identities.

Unexplained financial missingness retains its original masks/reasons. This model
policy does not certify those gaps as normal and does not forward-fill financials.


## Shared owner consumption

`CompactState.operation()` borrows the native Feature library and membership
stores through their existing operation scopes. Native checks still run when
entering an external operation; nested helpers use immutable admitted buffers.
The native adapter retains typed per-column buffers and reason codes. Only
requested row projections materialize Python rows. Overlapping model selections
reuse columns still retained in the Feature window; trimming releases them.
Training dependency digests reuse column digests. A bounded binary packing pass
preserves the historical grouped source proof so saved prepared batches remain
eligible for exact validation, without a disk panel or wide Python rows.

`reuse_raw_from_batch` also finds an exactly matching normalized target in the
borrowed batch. Cohort, raw references, cutoff and normalization definition must
match; each model still produces its own selector. No Feature validity enters
the label cohort. Raw dependency derivation is cached on the existing source
owner and boundary-validated before reuse.

`load_stock_feature_view` binds independent membership from the saved source
descriptor. Older saved Features without that descriptor accept an explicit
`membership_source` descriptor from their original source. The example options
pass this argument through. Boolean Feature membership alone cannot establish
known/complete membership; missing evidence remains an error. Native format is
not required. `prepared_batch` continues through the configured public entry.

Feature-library progress emits only the current `core_work` item; final returned
statistics retain the full history. Timings from a fresh preparation root and
saved-batch consumption must be reported separately, including actual model
`cache_hit`, `train_calls`, fit and predict times.

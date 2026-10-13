# M1 public entry closeout — 2026-10-13

This review candidate closes A05 public entry gaps, rejects the A08 falsely declared raw training path, exposes A01 existing stage measurements and updates A07 navigation in the companion Docs change. It does not complete general M1.

## Change and compatibility

The public lazy namespace exports `load_feature_library_training_view`. `load_stock_model` and `predict_stock_model` accept v3 through the existing complete saved-fold validator, with an optional admitted batch. This verifies the original saved parent, LabelSpec, clock, selection and booster bindings; it is not a contract-name allowlist. Cold calls retain parent availability and admission costs. Historical refs are neither migrated nor re-signed.

All v3 producers reach the same saved-input boundary before cache admission or fit. That boundary now requires cross-sectional, dimensionless Feature schema because this profile declares normalized inputs. Previously accepted raw inputs could be falsely described as normalized; they are now rejected. Raw Feature library reads remain supported. This tightening does not normalize data, repair old raw artifacts or establish that a particular historical run was affected. Before reusing such a candidate, inspect its original Feature schema and stop if it is raw; explicit normalized preparation and affected model/downstream regeneration require a separately scoped decision. No historical result is changed by this patch.

The configured entry saves a small `metrics.sequential` receipt even when no metrics dictionary is supplied. Its stage durations partition shared preparation through frozen OOS; fold entries and existing owner timings are nested details, not extra time to add. Fold residual save/admission includes remaining conversion and persistence work. Exact HIT records zero fit/predict calls and time. CLI Feature and saved-batch cold admission are separate. Caller Data opening, receipt publication, signal statistics, accounts and UI are outside this total. Published immutable receipts are not rewritten for later timing.

Static review followed the common producer path, complete fold loader and save-before-publish flow. Timers observe existing operations; the change adds no parent proof scans or fit calls. A02 broad implementation compatibility, A03 additional universes/backends/shadow, A04 physical security-axis incremental extension and A06 remote UI remain deferred.

## Bounded verification

Four selected tests passed in one bounded verification session:

- `test_stock_feature_library.StockFeatureLibraryTests.test_second_experiment_adds_other_source_column_and_rejects_wrong_training_source`: actual public training-view loader import and library/source admission.
- `test_stock_prepared_reuse.PreparedReuseTests.test_changed_model_consumes_complete_batch_and_subset_reprepares_its_cohort`: two-fold first run, changed-parameter CLI reuse, public v3 shared/cold model load and two exact independent booster predictions; changed subset matches fresh preparation. Two fits per first/reuse/subset/control phase (eight total). Reuse performs zero Data selection, Raw/Core preparation or proof writes; saved prepared hashes and mtimes remain unchanged.
- `test_stock_prepared_reuse.PreparedReuseTests.test_raw_feature_stage_is_rejected_before_v3_fit`: configured v3 producer rejects raw schema with zero backend fits and no booster saved. Direct producer coverage is by static common-path review, not a second runtime test.
- `test_stock_shared_execution.SharedExecutionTests.test_real_public_two_fold_train_save_oos_and_exact_resume`: two fits/predictions initially, zero on exact HIT; zero HIT fit/predict seconds; immutable fold bytes unchanged. Stage sum equals sequential total and progress counts two folds.

The first invocation passed three tests, while the fourth stopped before fixture setup because its requested evidence directory did not exist. Only that fourth test was retried after creating the directory; it passed in 2.101 seconds. The initial four-test invocation took 5.491 seconds. No implementation changes followed execution. Arrow emitted sandbox CPU-cache inspection warnings but completed the selected paths. These durations are test-run observations, not a platform performance benchmark.

Runtime: Python 3.12.12, NumPy 2.4.1, pandas 2.3.3, LightGBM 4.6.0, PyArrow 22.0.0, PyYAML 6.0.3. This does **not** verify the optional dependency pin LightGBM 4.7.0. Source bases: Research `699fa6a706dbce074510d40b22eb48accc9aa4ba`, Engine `306b26d3ec513fa6422b5647ec6fa3e03dfacf1e`, Data `b743b704aba1232941f98dc2b338c6261fd3d8fb`. The local Data source used for tests is distinct from the newer remote audit head.

Private small receipts are retained in the task's `.artifacts/m1-closeout/verification.json`, `prepared/acceptance.json` and `shared/acceptance.json`; they are not committed production data. The library loader read and configured two-fold run use separate existing fixtures, not a new long Feature-library-to-account run. No full suite, production dataset, supplier collection, seven-year training, account replay or UI deployment was performed. Existing T01 account/UI and basis-registration evidence remains the evidence for those unchanged paths; this change does not revalidate or improve their measured performance.

The remaining critical path is parent review of the fixed Research and Docs commits. Broader M1 scope remains in the canonical Docs manual; these four fixes do not settle the deferred findings.

# T01 configuration

Read `proposal.md`, then `time.yaml`, `features.json`, `label.yaml`, `model.yaml`,
`strategies.json` and `evaluation.json`. The 88 ordered feature IDs, versions,
names, formulas and lookbacks come directly from the existing installed catalog.
They are fixed for this first round.

The existing `examples/a01_one_year.py` CLI remains the research entry. Assemble
its existing dataset/experiment/options files using a genuine frozen Data scope
and source. These public files contain no private snapshot or result path.
`time.yaml` is copied into the existing dataset `weekly_schedule` field,
using `stock_dataset_schedule_v2` and `compile_stock_weekly_folds`;
`evaluation.json` is copied into the existing options `evaluation` field.
`strategies.json` describes two calls to the existing Engine strategy factory.
Its account cash, frequency and shared-prediction conditions are orchestration
requirements rather than arguments to that factory.

The date range is 2019-01-02 through 2025-12-31. A lightweight inspection of
the saved exchange calendar compiled into 1,699 daily predictions and 359
actual ISO trading weeks, including short holiday weeks and year boundaries.
Each fold declares 543 preceding training feature dates, with 504 nominal
mature dates. Actual Label availability remains checked by its owner.
The first week trades 2019-01-02/03/04 and fits on 2018-12-28.
Features need 2016-10-13 through 2025-12-30; physical lookback starts
2015-09-28. The final prediction Feature date's H40 endpoint is 2026-03-06,
and evaluation uses a 2026-03-09 cutoff.

This is a calendar plan. Formal Data scope, historical member qualification,
source expansion, Feature production and the seven-year experiment remain
pending. It does not establish actual complete data or accepted results.

Use the existing `run_saved_stock_strategy` and
`evaluate_saved_stock_strategy` Research adapters over the sole Engine Runtime
account path. Supply the original frozen market/profile/native inputs, source
and account request. Attach both accounts with the same registration context.
No account executor or Feature mathematics is introduced by these configs.

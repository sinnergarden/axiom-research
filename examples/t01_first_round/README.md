# T01 configuration

Read `proposal.md`, then `time.yaml`, `features.json`, `label.yaml`, `model.yaml`,
`strategies.json` and `evaluation.json`. The 88 ordered feature IDs, versions,
names, formulas and lookbacks come directly from the existing installed catalog.
They are fixed for this first round.

`label.yaml` defines raw H40 returns and therefore says `normalization_policy:
none`. Training is wired through `dataset.yaml`'s existing
`preparation_options.normalization_backend: core_cs_batch_v1`. Its sequential
owner calls `forward_label_cs_zscore_v1` per mature feature-date cross-section
(ddof 0, epsilon 1e-12, constant/missing outcomes excluded), and feeds the
normalized target to fit. Signal evaluation retains raw returns. This is the
existing owner behavior, not a new preprocessing implementation.

The existing `examples/a01_one_year.py` CLI remains the research entry. Assemble
the supplied `dataset.yaml`, `experiment.yaml` and `options.json` templates
using a genuine frozen Data scope and source. Replace every `<...>` value with
an actual local path or saved identity before validation. These templates are
deliberately incomplete until Data supplies those artifacts; they contain no
private snapshot or result path.
`time.yaml` is copied into the existing dataset `weekly_schedule` field,
using `stock_dataset_schedule_v2` and `compile_stock_weekly_folds`;
`evaluation.json` is copied into the existing options `evaluation` field.
The templates already contain copies of those objects. If editing the separate
time or evaluation file, update its consumed copy before validation; the CLI
does not read those fragment files automatically.
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
Historical members and visibility use `best_effort_vendor_v1` and
`declared_vendor_assumption`; actual historical receipt times are unverified.

Use the existing `run_saved_stock_strategy` and
`evaluate_saved_stock_strategy` Research adapters over the sole Engine Runtime
account path. Supply the original frozen market/profile/native inputs, source
and account request. Attach both accounts with the same registration context.
No account executor or Feature mathematics is introduced by these configs.

The inspected existing profile declares 0.0003 commission, minimum CNY 5,
zero slippage bps and 10% daily-volume participation. Its fee schedule uses
sell stamp tax 0.001 before 2023-08-28 and 0.0005 thereafter; two-sided transfer
fees are 0.00002 before 2022-04-29 and 0.00001 thereafter. Those are saved model
assumptions and source-backed schedule values. The inspected profile's rules
cover the old short calendar and its verified fee window is only 2019-06-28
through 2024-06-28. A genuine seven-year profile and verified fee window must
be prepared by Engine; do not pass the old profile as complete seven-year
coverage. Report the actually admitted profile identity and fee values beside
both local account results. Gross dividends exclude personal dividend taxes.

After replacing the placeholders, use the unchanged formal Research CLI:

```sh
python examples/a01_one_year.py --options /absolute/path/options.json --stage validate
python examples/a01_one_year.py --options /absolute/path/options.json --stage features
python examples/a01_one_year.py --options /absolute/path/options.json --stage train
python examples/a01_one_year.py --options /absolute/path/options.json --stage evaluate
```

For an accepted Feature-library projection, build/export its full-period
training projection once, open its genuine `StockFeatureView` once, and pass
that handle to `build_configured_stock_sequential_experiment`. The existing
sequential preparation owner borrows it for all folds and projects each active
window. Do not invoke the projection builder or loader for every fold. The
current CLI recognizes the original Feature format; automatic dispatch for the
candidate projection remains an owner integration task.

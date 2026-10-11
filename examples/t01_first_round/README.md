# T01 configuration

Read `proposal.md`, then `time.yaml`, `features.json`, `label.yaml`, `model.yaml`,
`strategies.json` and `evaluation.json`. The 88 ordered feature IDs, versions,
names, formulas and lookbacks come directly from the existing installed catalog.
They are fixed for this first round.

The existing `examples/a01_one_year.py` CLI remains the research entry. Assemble
its existing dataset/experiment/options files using a genuine frozen Data scope
and source. These public files contain no private snapshot or result path.
`time.yaml` is copied into the existing dataset `periodic_schedule` field;
`evaluation.json` is copied into the existing options `evaluation` field.
`strategies.json` describes two calls to the existing Engine strategy factory.
Its account cash, frequency and shared-prediction conditions are orchestration
requirements rather than arguments to that factory.

For the frozen calendar checked locally, the annual plan has 242 trade dates
and 13 folds. Each fold declares 543 preceding training feature dates and has
504 nominal mature dates. Actual Label availability remains checked by its
owner. Features cover 2020-10-12 through 2023-12-28 and physical lookback starts
2019-09-23. The last prediction Feature date's H40 endpoint is 2024-03-04;
the evaluation cutoff is 2024-03-05. A calendar plan does not establish actual
complete data or completed annual research.

Use the existing `run_saved_stock_strategy` and
`evaluate_saved_stock_strategy` Research adapters over the sole Engine Runtime
account path. Supply the original frozen market/profile/native inputs, source
and account request. Attach both accounts with the same registration context.
No account executor or Feature mathematics is introduced by these configs.

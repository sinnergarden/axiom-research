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

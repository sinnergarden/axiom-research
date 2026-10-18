# Published inputs and ordinary research runs

Preparation publishes immutable Feature, membership, target and model artifacts.
It owns source interpretation, original-byte inspection and publication completeness.
An ordinary research run reads their manifests, selected partitions and buffers.
Source descriptors remain attached for traceability; consumers do not reconstruct
ancestral receipts, hash every file again, or scan past files at operation boundaries.
Missing files fail through normal I/O. An owner must not modify a published file in place.

The existing APIs use this behavior for native and older exported training views:
`normalize_saved`, `build_feature_library_training_view`,
`load_feature_library_training_view`, shared rolling batch consumption and saved
model loading. There is no alternate trusted loader or proof cache.

`normalize_saved` binds the published raw view and membership source once. Existing
completed coverage is reused through its original Core header's raw view, source,
reference identity and column semantics; its dependency IDs are retained unchanged.
New missing output partitions use `published_feature_normalization_inputs_v1` as
the input identity. This changes metadata identity, not the normalization arithmetic.
Only coverage entries for completed partitions are eligible. A completed training
view can be passed straight to the training loader without calling normalization.

Fold preparation now records `stock_training_binding_v4`: the published Feature
view, ordered feature selection, row selector and prepared target. It emits no
per-date Feature proof files. Older v3 bindings remain readable. New fold identities
can differ because their bindings and implementation references differ; existing
completed model outputs remain readable through the same loader.

PIT and cutoff selection, mature labels, actual membership, training visibility,
normalization definitions, missing-value handling and X/y row alignment remain
business rules. They are not replaced by content hashes. Saved model HIT reads the
completed model/prediction publication; OOS publication still selects its requested
evaluation labels. Production Core and target calculations are unchanged.

The focused saved-input comparison is supplied with the change handoff. It reads
one existing F88 fold under both revisions, compares X/y/P and ordered keys, and
compares saved model/prediction output. No Feature generation or model fitting is
needed. Execute once when the active long research run has released memory.

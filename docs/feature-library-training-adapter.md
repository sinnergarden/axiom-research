# Feature library training adapter

`stock_feature_library_adapter.build_feature_library_training_view` projects a
fixed Engine FeatureView into bounded daily arrays and the original Research
training row facts. It uses the explicit
`stock_feature_library_training_projection_v1` contract, retaining the library
view ref, original partition refs, fixed source descriptor, sample membership,
original cutoff, cell clocks and reasons. This export performs no Feature math.

`load_feature_library_training_view` verifies its actual files, compares the
exported cells with the original library projection, checks membership against
the pinned source buffer, and returns the existing StockFeatureView owner handle.
Both build and cold load verify the registered DAG/input semantics and every
column/day dependency receipt, including the reference cohort for CS. An explicit
compatibility receipt binds these checks to the requested source, snapshot/PIT
and cutoffs. Unchanged local inputs can reuse outputs from older source frames;
the adapter preserves their original refs and does not relabel their provenance.
Pass this handle to `prepare_stock_ml_batch_inputs` or
`build_configured_stock_sequential_experiment`; labels, maturity, structural
training proofs, fitting and signal freezing continue through their original
owners. The adapter does not alter experiment discovery or CLI registration.

For cold restoration, load this projection through the adapter first, then pass
the handle explicitly to `load_stock_ml_batch_inputs(..., feature_inputs=handle)`.
Within an existing preparation session, pass `source=admitted_source` to the
adapter loader to reuse that source owner's admission. A cold loader admits the
pinned source through its original owner before checking local compatibility.
Its new contract is not a legacy Feature producer artifact. The old Feature
loaders and already saved normalized matrices retain their existing identities.

The synthetic source test verifies packed-to-training-owner row equivalence.
Real training, strategy execution and performance evidence are still pending.

Engine's generic source interface supports new raw facts: register the new fact
semantics and derived DAG, fingerprint its selected data revisions, then request
the existing output range. A synthetic field backfill verifies only the new
dependent output is computed and every old content ref remains unchanged despite
the source manifest/catalog expansion. Reading source bytes and computing
Features are counted separately.

The current Research vendor source is still the fixed open/high/low/close/amount/
factor profile. Arbitrary additional vendor fields are not admitted by its v1
format. The smallest extension is a Data-owned typed field sidecar with its own
clock, missing/unit/precision contract and local revision evidence, composed by a
Research fact adapter alongside the unchanged six-field source. Its per-field/day
receipts and source bindings feed the existing Engine interface; the old physical
buffers and keys remain intact. No full-history job is started for this design.

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
Pass this handle to `prepare_stock_ml_batch_inputs` or
`build_configured_stock_sequential_experiment`; labels, maturity, structural
training proofs, fitting and signal freezing continue through their original
owners. The adapter does not alter experiment discovery or CLI registration.

For cold restoration, load this projection through the adapter first, then pass
the handle explicitly to `load_stock_ml_batch_inputs(..., feature_inputs=handle)`.
Its new contract is not a legacy Feature producer artifact. The old Feature
loaders and already saved normalized matrices retain their existing identities.

The synthetic source test verifies packed-to-training-owner row equivalence.
Real training, strategy execution and performance evidence are still pending.

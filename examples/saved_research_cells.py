"""Executable cells for the existing researcher notebook.

Load these definitions with runpy.run_path; call a cell with explicit local
inputs and a new destination. Importing this file performs no research or I/O.
Authoritative prose and the notebook remain in axiom-docs.
"""

# %% Saved Feature/Label inputs: change one explicit model parameter.
def learning_rate_variant(batch_manifest_path, *, fold_index, learning_rate,
                          destination, limits=None):
    import json
    from pathlib import Path
    from time import perf_counter
    from axiom_research import (load_stock_ml_batch_inputs,
        build_stock_ml_fold_from_saved_inputs)

    manifest = json.loads(Path(batch_manifest_path).read_text())
    if type(fold_index) is not int or not 0 <= fold_index < len(manifest['folds']):
        raise ValueError('explicit zero-based saved fold index required')
    item = manifest['folds'][fold_index]
    if manifest['contract_version'] not in ('stock_ml_batch_inputs_v3', 'stock_ml_batch_inputs_v4'):
        raise ValueError('this sequential cell requires compact v3/v4 saved inputs')
    started = perf_counter()
    with load_stock_ml_batch_inputs(manifest, residency='sequential', limits=limits) as batch:
        control_seconds = perf_counter() - started
        observations = []
        for _ in range(2):
            before = dict(batch.metrics)
            metrics = {}
            started = perf_counter()
            fold = build_stock_ml_fold_from_saved_inputs(item['input_manifest'],
                fold_spec=item['fold_spec'], batch=batch, destination=destination,
                training_options={'learning_rate': learning_rate}, metrics=metrics)
            elapsed = perf_counter() - started
            after = batch.metrics
            observations.append({
                'mode': 'cache_reuse' if metrics['cache_hit'] else 'saved_input_build',
                'caller_seconds': elapsed, 'builder_metrics': metrics,
                'owner_delta': {k: after.get(k, 0)-before.get(k, 0) for k in
                    ('source_bytes', 'file_hash_calls', 'json_decode_calls',
                     'feature_block_admissions', 'training_block_gathers')},
                'fold_ref': fold.identity})
        if observations[0]['fold_ref'] != observations[1]['fold_ref'] or not fold.reused:
            raise AssertionError('same definition must reuse the saved fold')
        if any(observations[1]['builder_metrics'][k] for k in ('train_calls', 'predict_calls')):
            raise AssertionError('exact HIT must not fit or predict')
        for observation in observations:
            if any(observation['builder_metrics'][k] for k in
                   ('data_read_calls', 'supplier_calls', 'core_calls', 'account_calls')):
                raise AssertionError('saved model variant crossed a preparation/account boundary')
        saved = fold.to_dict()
        result = {'scope': 'caller-supplied saved inputs; provenance remains in the owner artifacts',
            'batch_ref': batch.identity, 'fold_ref': fold.identity,
            'dataset_ref': saved['dataset_ref'], 'signal_run_ref': saved['signal_run_ref'],
            'parameters': fold.model()['parameters'],
            'num_boost_round': fold.model()['num_boost_round'],
            'control_admission_seconds': control_seconds, 'calls': observations}
    return result

# %% Read a saved engineering receipt; do not run the benchmark again.
def width_costs(receipt_path):
    import json
    from pathlib import Path
    saved = json.loads(Path(receipt_path).read_text())
    if saved['status'] != 'PASS_WIDTH_CONSUMPTION_ENGINEERING_CLOSED':
        raise ValueError('explicit accepted width-consumption receipt required')
    return {key: saved[key] for key in
        ('scope', 'seconds', 'warm_deltas', 'memory_bytes',
         'ratios_300_over_6', 'remaining_boundary')}

# %% Notebook calls (configure explicit local selectors in the notebook).
# cells = runpy.run_path(str(research_repo / 'examples/saved_research_cells.py'))
# model = cells['learning_rate_variant'](saved_batch_path, fold_index=0,
#     learning_rate=0.04, destination=new_model_destination, limits=owner_limits)
# top3_path = make_topk_account(saved_stock_account_path, 3, new_account_destination)
# The existing ML notebook owns make_topk_account and evaluate_saved_topk.
# costs = cells['width_costs'](saved_width_receipt_path)
# Model changes preserve prepared Feature/Label inputs; TopK executes a new
# account. Staged-clock prediction v2 still requires Engine account admission.
# Receipt costs describe explicit engineering columns, not real Alpha158
# computation or a five-year historical universe/coverage acceptance.

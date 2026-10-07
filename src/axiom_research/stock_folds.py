"""One bounded moving fold over saved stages; no Data/Feature/account calls."""
from copy import deepcopy
import errno
from pathlib import Path
import tempfile
import time

from .stock_artifacts import digest, file_digest, write_json
from .stock_fold_inputs import (project_saved_fold, seal, require, feature_available,
                                _instant, _finite, NORMALIZATION_SPEC)
from .stock_fold_artifacts import StockMLFold, load_stock_ml_fold, _load_stock_ml_fold
from .stock_ml import (_implementation, _environment, TARGET_SEMANTICS,
                       LGBM_PARAMETERS, TREES, _signal_evidence)

VERSION = 'axiom.stock_ml_fold/1'


def prediction_rows(features, spec, model, scores):
    """Publish the agreed v2 clocks without mutating original Feature rows."""
    rows = []
    for source in features['rows']:
        day = source['session']
        if day not in features['prediction_sessions']:
            continue
        key = source['security_id'], day
        score = scores.get(key); valid = _finite(score)
        reason = None if valid else 'NOT_MEMBER' if not source['member'] else (
            'FEATURE_AVAILABILITY_UNKNOWN' if feature_available(source) is None else 'FEATURE_MISSING')
        parent = features['parents_by_session'][day]
        rows.append({'security_id': key[0], 'session': day,
            'knowledge_cutoff': spec['inference_cutoff_by_session'][day],
            'available_at': spec['inference_cutoff_by_session'][day],
            'feature_knowledge_cutoff': source['knowledge_cutoff'],
            'feature_available_at': feature_available(source),
            'simulated_model_available_at': spec['simulated_model_available_at'],
            'score': float(score) if valid else None, 'valid': valid, 'invalid_reason': reason,
            'member': source['member'], 'source_refs': [features['feature_ref'], parent['feature_ref'],
                parent['qlib_view_ref'], model['model_ref'], features['catalog_ref']]})
    return rows


def build_stock_ml_fold_from_saved_inputs(input_manifest, *, fold_spec, destination, metrics=None, batch=None,
                                        training_options=None):
    """Fit/predict once; an exact, fully validated definition alone permits HIT.

    Parent paths are explicit and remain required. An initialized batch reuses
    verified common parents and a shared Feature matrix within this process.
    A separate public loader always verifies the complete saved closure.
    Saved matrix inputs also accept training_options containing only
    learning_rate and/or num_boost_round. All other existing profile values
    remain fixed; these options change the model identity, not prepared inputs.
    """
    begin = time.perf_counter()
    inputs, spec = deepcopy(input_manifest), deepcopy(fold_spec)
    if inputs.get('contract_version') in ('stock_ml_saved_inputs_v2','stock_ml_saved_inputs_v3'):
        from .stock_matrix_folds import build_matrix_fold
        return build_matrix_fold(inputs,spec=spec,destination=destination,metrics=metrics,batch=batch,
                                 training_options=training_options)
    require(training_options is None, 'training_options requires saved matrix inputs')
    if batch is not None:
        from .stock_batch import _data
        _data(batch)  # Reject a caller-created object or a skip-validation flag.
    projection = project_saved_fold(inputs, spec) if batch is None else batch._project(inputs, spec)
    features, labels, training, excluded, raw_refs, evaluation = projection
    compact = spec['contract_version'] == 'stock_ml_fold_spec_v2'
    from .feature_catalog import load_feature_catalog
    catalog = load_feature_catalog()
    require(inputs['catalog_ref'] == catalog.identity and inputs['ordered_features'] ==
            [x['id'] for x in catalog.select(inputs['feature_selection'])], 'current catalog selection mismatch')
    implementations = _implementation()
    definition = {'version': 'axiom.stock_ml_fold/2' if compact else VERSION,
        'input_manifest_ref': digest(inputs), 'input_manifest': inputs,
        'fold_spec_ref': digest(spec), 'fold_spec': spec, 'catalog_ref': inputs['catalog_ref'],
        'parameters': LGBM_PARAMETERS, 'num_boost_round': TREES, 'target_semantics': TARGET_SEMANTICS,
        'label_normalization': NORMALIZATION_SPEC, 'environment': _environment(),
        'implementation_sources': implementations, 'implementation_ref': digest(implementations)}
    definition_ref = digest(definition); target = Path(destination) / definition_ref[7:]
    zeros = dict(data_read_calls=0, supplier_calls=0, feature_core_calls=0, label_core_calls=0,
                 core_calls=0, account_calls=0, train_calls=0, predict_calls=0)
    if target.exists():
        loaded = (load_stock_ml_fold(target) if batch is None else
                  _load_stock_ml_fold(target, projection=projection))
        require(loaded.to_dict()['definition'] == definition, 'cached fold definition mismatch')
        if metrics is not None:
            metrics.update(zeros, cache_hit=True, total_seconds=time.perf_counter()-begin)
        return StockMLFold(target, True)
    require(len(training) >= 40, 'insufficient mature finite training rows')
    dataset = seal({'contract_version': 'stock_fold_dataset_v2' if compact else 'stock_fold_dataset_v1',
        'feature_ref': features['feature_ref'],
        'label_ref': labels['label_ref'], 'raw_label_refs': raw_refs, 'fold_spec_ref': digest(spec),
        'fit_cutoff': spec['fit_cutoff'], 'ordered_features': inputs['ordered_features'],
        'training_keys': [[r['security_id'], r['session']] for r in training],
        'training_rows_ref': digest(training), 'training_row_count': len(training), 'excluded': excluded,
        'target_semantics': TARGET_SEMANTICS, 'normalization': NORMALIZATION_SPEC,
        'validation': 'none_fixed_parameters_no_early_stopping'}, 'dataset_ref')
    from .stock_training import fit_predict_stock_model
    import numpy as np
    candidates = [r for r in features['rows'] if r['session'] in features['prediction_sessions'] and
        r['member'] and all(r['validity']) and all(_finite(v) for v in r['values']) and
        feature_available(r) is not None]
    stats = {**zeros, 'cache_hit': False, 'saved_input_validation_seconds': time.perf_counter()-begin}
    if batch is None:
        X = np.asarray([r['values'] for r in training], dtype=np.float64)
        y = np.asarray([r['label'] for r in training], dtype=np.float64)
        P = np.asarray([r['values'] for r in candidates], dtype=np.float64)
    else:
        X, y, P = batch._matrices(training, candidates)
    booster, scores = fit_predict_stock_model(X, y, P, ordered_features=inputs['ordered_features'],
        parameters=LGBM_PARAMETERS, num_boost_round=TREES, metrics=stats)
    require(len(scores) == len(candidates) and all(_finite(float(s)) for s in scores), 'invalid model scores')
    score_map = {(r['security_id'], r['session']): float(s) for r, s in zip(candidates, scores)}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stock-fold-', dir=target.parent) as temp:
        stage = Path(temp) / 'complete'; stage.mkdir(); booster.save_model(str(stage / 'booster.txt'))
        model = seal({'contract_version': 'stock_model_release_v2', 'dataset_ref': dataset['dataset_ref'],
            'feature_ref': features['feature_ref'], 'label_ref': labels['label_ref'], 'raw_label_refs': raw_refs,
            'fit_cutoff': spec['fit_cutoff'], 'simulated_available_at': spec['simulated_model_available_at'],
            'clock_basis': 'declared_simulation', 'ordered_features': inputs['ordered_features'],
            'feature_selection': inputs['feature_selection'], 'catalog_ref': inputs['catalog_ref'],
            'target_semantics': TARGET_SEMANTICS, 'parameters': LGBM_PARAMETERS, 'num_boost_round': TREES,
            'environment': definition['environment'], 'implementation_ref': definition['implementation_ref'],
            'booster_digest': file_digest(stage/'booster.txt'),
            'feature_normalization': 'same_date_visible_members_cs_zscore_no_fit',
            'label_normalization': NORMALIZATION_SPEC}, 'model_ref')
        predictions = seal({'contract_version': 'stock_prediction_run_v2', 'signal_stage': 'prediction_raw',
            'score_semantics': TARGET_SEMANTICS, 'score_unit': 'dimensionless', 'feature_ref': features['feature_ref'],
            'model_ref': model['model_ref'], 'fold_spec_ref': digest(spec), 'clock_basis': 'declared_simulation',
            'universe': inputs['universe'], 'rows': prediction_rows(features, spec, model, score_map),
            'limitations': ['Declared simulation publication clocks are not historical realtime completion evidence.',
                'Feature inputs retain their original best-effort historical availability and immutable refs.',
                'Scores are normalized-target predictions, not return percentages.',
                'Engine validation and Runtime account execution are not performed by this builder.']}, 'signal_run_ref')
        evidence = _signal_evidence(predictions, evaluation, spec['evaluation_cutoff'])
        refs = {key: value[key] for key, value in [('feature_ref', features), ('label_ref', labels),
            ('dataset_ref', dataset), ('model_ref', model), ('signal_run_ref', predictions), ('evidence_ref', evidence)]}
        fold = {'contract_version': 'stock_ml_fold_v2' if compact else 'stock_ml_fold_v1', 'definition': definition,
            'definition_ref': definition_ref, 'status': 'COMPLETE', **refs,
            'fold_ref': digest({'definition_ref': definition_ref, **refs}),
            'engine_admission': {'neutral_validation': 'NOT_PERFORMED_BY_BUILDER',
                                 'runtime': 'NOT_PERFORMED_BY_BUILDER'},
            'limitations': ['Explicit immutable parent paths remain required; relocation is not supported.',
                'Full parent proof parsing is bounded by caller resources, not streaming.',
                'No Data, supplier, Feature, label normalization or account execution in this builder.']}
        fold = seal(fold, 'content_digest')
        values = {'fold.json': fold, 'feature-slice.json': features, 'label-slice.json': labels,
            'dataset.json': dataset, 'model.json': model, 'predictions.json': predictions, 'signal-evidence.json': evidence}
        for name, value in values.items():
            write_json(stage/name, value)
        write_json(stage/'manifest.json', {'contract_version': 'stock_ml_fold_manifest_v1', 'fold_ref': fold['fold_ref'],
            'files': {n: file_digest(stage/n) for n in [*values, 'booster.txt']}})
        if batch is None:
            load_stock_ml_fold(stage)
        else:
            batch._check_sources()
            _load_stock_ml_fold(stage, projection=projection)
            batch._check_sources()
        try:
            stage.rename(target)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            loaded = (load_stock_ml_fold(target) if batch is None else
                      _load_stock_ml_fold(target, projection=projection))
            require(loaded.identity == fold['fold_ref'], 'concurrent fold conflict')
    stats.update(training_rows=len(training), training_sessions=len({r['session'] for r in training}),
        prediction_rows=len(predictions['rows']), valid_predictions=len(candidates),
        matrix_bytes=X.nbytes+y.nbytes+P.nbytes, artifact_bytes=sum(p.stat().st_size for p in target.iterdir()),
        total_seconds=time.perf_counter()-begin)
    if metrics is not None:
        metrics.update(stats)
    return StockMLFold(target)

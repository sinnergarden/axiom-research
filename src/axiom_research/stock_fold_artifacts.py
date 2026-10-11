"""Read-only saved fold closure. Imports only stdlib and saved-contract helpers."""
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from weakref import ref
import os

from .stock_artifacts import digest, file_digest, _read, _verify_ref
from .stock_fold_inputs import project_saved_fold, require, feature_available, _instant, _finite
from .stock_training import LGBM_PARAMETERS, TREES, TARGET_SEMANTICS
from .stock_label_contracts import NORMALIZATION_SPEC

OUTPUTS = {'feature-slice.json': 'feature_ref', 'label-slice.json': 'label_ref',
    'dataset.json': 'dataset_ref', 'model.json': 'model_ref', 'predictions.json': 'signal_run_ref',
    'signal-evidence.json': 'evidence_ref'}

# Compact output consumers keep the exact decoded documents admitted by their
# loader. Keys use object identity: dataclass equality must not let an unrelated
# caller-created path object borrow another object's verified payload.
_ADMITTED_FOLDS = {}


def _document(run, name):
    saved = _ADMITTED_FOLDS.get(id(run))
    if saved is not None and saved[0]() is run:
        require(saved[2]==os.getpid(), 'admitted compact fold belongs to another process')
        return deepcopy(saved[1][name])
    return _read(run.path/name)


def _owned_fold(path, documents, *, reused=False):
    run = StockMLFold(Path(path), reused)
    identity = id(run)
    def release(reference):
        saved = _ADMITTED_FOLDS.get(identity)
        if saved is not None and saved[0] is reference:
            _ADMITTED_FOLDS.pop(identity)
    _ADMITTED_FOLDS[identity] = (ref(run, release), documents, os.getpid())
    return run


def _repath_owned_fold(run, path, *, reused=False):
    saved = _ADMITTED_FOLDS.get(id(run))
    require(saved is not None and saved[0]() is run, 'admitted compact fold required')
    require(saved[2]==os.getpid(), 'admitted compact fold belongs to another process')
    return _owned_fold(path, saved[1], reused=reused)


@dataclass(frozen=True)
class StockMLFold:
    path: Path
    reused: bool = False

    def to_dict(self):
        return _document(self, 'fold.json')

    @property
    def identity(self):
        return self.to_dict()['fold_ref']

    def predictions(self):
        return _document(self, 'predictions.json')

    def model(self):
        return _document(self, 'model.json')

    def evidence(self):
        return _document(self, 'signal-evidence.json')


def load_stock_ml_fold(path, *, batch=None):
    """Validate saved bytes, parent closure and clocks, never run a stage.

    No current implementation/environment requirement is imposed on old saved
    reports. A fresh batch may reuse its fully admitted inputs in this process;
    all saved output checks remain. Without a batch, disk parents are rechecked.
    """
    return _load_stock_ml_fold(path, batch=batch)


def _load_stock_ml_fold(path, *, projection=None, batch=None):
    """Internal publication check may reuse this call's verified projection."""
    if batch is not None:
        from .stock_batch import _data
        _data(batch)
        require(projection is None, 'batch cannot accept a caller projection')
        batch._check_sources()
    path = Path(path).resolve()
    from .stock_compact_store import OwnedStore
    with OwnedStore() as ingress:
        manifest = ingress.read_json({'path':str(path/'manifest.json')})
        if manifest.get('contract_version') == 'stock_ml_fold_manifest_v2':
            from .stock_matrix_folds import _load_matrix_fold
            return _load_matrix_fold(path,projection=projection,batch=batch,ingress=ingress)
    require(manifest.get('contract_version') == 'stock_ml_fold_manifest_v1' and
            set(manifest.get('files', {})) == {*OUTPUTS, 'fold.json', 'booster.txt'}, 'unexpected saved fold files')
    for name, reference in manifest['files'].items():
        require(file_digest(path/name) == reference, 'saved fold file mismatch: '+name)
    fold = _read(path/'fold.json'); _verify_ref(fold, 'content_digest')
    require(fold['contract_version'] in ('stock_ml_fold_v1', 'stock_ml_fold_v2') and
            fold['status'] == 'COMPLETE', 'incomplete fold')
    definition = fold['definition']; spec = definition['fold_spec']; inputs = definition['input_manifest']
    compact = spec['contract_version'] == 'stock_ml_fold_spec_v2'
    require(fold['contract_version'] == ('stock_ml_fold_v2' if compact else 'stock_ml_fold_v1') and
            definition['version'] == ('axiom.stock_ml_fold/2' if compact else 'axiom.stock_ml_fold/1') and
            definition['parameters'] == LGBM_PARAMETERS and
            definition['num_boost_round'] == TREES and definition['target_semantics'] == TARGET_SEMANTICS and
            definition['label_normalization'] == NORMALIZATION_SPEC, 'unsupported saved fold profile')
    require(fold['definition_ref'] == digest(definition) and definition['input_manifest_ref'] == digest(inputs) and
            definition['fold_spec_ref'] == digest(spec) and
            definition['implementation_ref'] == digest(definition['implementation_sources']), 'fold definition mismatch')
    refs = {key: fold[key] for key in OUTPUTS.values()}
    require(fold['fold_ref'] == digest({'definition_ref': fold['definition_ref'], **refs}) == manifest['fold_ref'],
            'fold identity mismatch')
    saved = {}
    for name, key in OUTPUTS.items():
        value = _read(path/name); _verify_ref(value, key)
        require(value[key] == fold[key], 'fold stage reference mismatch')
        saved[name] = value
    if batch is not None:
        projection = batch._project(inputs, spec)
    features, labels, training, excluded, raw_refs, evaluation = (
        project_saved_fold(inputs, spec) if projection is None else projection)
    require(saved['feature-slice.json'] == features and saved['label-slice.json'] == labels,
            'saved slice/immutable parent mismatch')
    dataset, model = saved['dataset.json'], saved['model.json']
    predictions, evidence = saved['predictions.json'], saved['signal-evidence.json']
    require(dataset['contract_version'] == ('stock_fold_dataset_v2' if compact else 'stock_fold_dataset_v1') and
            model['contract_version'] == 'stock_model_release_v2', 'unsupported fold stage contract')
    require(len(training) >= 40 and dataset['training_keys'] == [[r['security_id'], r['session']] for r in training] and
            dataset['training_row_count'] == len(training) and dataset['training_rows_ref'] == digest(training) and
            dataset['excluded'] == excluded, 'saved training selection mismatch')
    for actual, expected in ((dataset['feature_ref'], features['feature_ref']), (dataset['label_ref'], labels['label_ref']),
        (dataset['fold_spec_ref'], digest(spec)), (dataset['fit_cutoff'], spec['fit_cutoff']),
        (dataset['raw_label_refs'], raw_refs), (dataset['ordered_features'], inputs['ordered_features']),
        (dataset['target_semantics'], definition['target_semantics']), (dataset['normalization'], definition['label_normalization']),
        (model['dataset_ref'], dataset['dataset_ref']), (model['feature_ref'], features['feature_ref']),
        (model['label_ref'], labels['label_ref']), (model['raw_label_refs'], raw_refs),
        (model['fit_cutoff'], spec['fit_cutoff']), (model['simulated_available_at'], spec['simulated_model_available_at']),
        (model['clock_basis'], 'declared_simulation'), (model['ordered_features'], inputs['ordered_features']),
        (model['feature_selection'], inputs['feature_selection']), (model['catalog_ref'], inputs['catalog_ref']),
        (model['parameters'], definition['parameters']), (model['num_boost_round'], definition['num_boost_round']),
        (model['environment'], definition['environment']), (model['implementation_ref'], definition['implementation_ref']),
        (model['booster_digest'], manifest['files']['booster.txt']),
        (model['target_semantics'], definition['target_semantics']), (model['label_normalization'], definition['label_normalization'])):
        require(actual == expected, 'saved fold model/dataset linkage mismatch')
    require(_instant(model['fit_cutoff']) < _instant(model['simulated_available_at']), 'model publication precedes fit')
    require(set(predictions) == {'contract_version', 'signal_run_ref', 'signal_stage', 'score_semantics', 'score_unit',
        'feature_ref', 'model_ref', 'limitations', 'universe', 'rows', 'fold_spec_ref', 'clock_basis'} and
        predictions['contract_version'] == 'stock_prediction_run_v2' and predictions['signal_stage'] == 'prediction_raw' and
        predictions['score_unit'] == 'dimensionless' and predictions['clock_basis'] == 'declared_simulation' and
        predictions['score_semantics'] == model['target_semantics'] and predictions['fold_spec_ref'] == digest(spec) and
        predictions['feature_ref'] == features['feature_ref'] and predictions['model_ref'] == model['model_ref'] and
        predictions['universe'] == inputs['universe'], 'prediction contract/stage mismatch')
    source = {(r['security_id'], r['session']): r for r in features['rows']}
    expected_keys = {(s, d) for d in features['prediction_sessions'] for s in inputs['universe']}; indexed = {}
    for row in predictions['rows']:
        require(set(row) == {'security_id', 'session', 'knowledge_cutoff', 'available_at', 'score', 'valid',
            'invalid_reason', 'source_refs', 'member', 'feature_knowledge_cutoff', 'feature_available_at',
            'simulated_model_available_at'}, 'prediction row fields mismatch')
        key = row['security_id'], row['session']
        require(key in expected_keys and key not in indexed, 'duplicate/unexpected prediction key')
        indexed[key] = row; original = source[key]; day = key[1]
        parent = features['parents_by_session'][day]
        require(row['feature_knowledge_cutoff'] == original['knowledge_cutoff'] and
                row['feature_available_at'] == feature_available(original) and
                row['member'] is original['member'] and row['simulated_model_available_at'] == model['simulated_available_at'] and
                row['source_refs'] == [features['feature_ref'], parent['feature_ref'], parent['qlib_view_ref'],
                    model['model_ref'], inputs['catalog_ref']], 'prediction original source/clock mismatch')
        infer = _instant(spec['inference_cutoff_by_session'][day])
        require(_instant(row['knowledge_cutoff']) == _instant(row['available_at']) == infer and
                _instant(row['simulated_model_available_at']) < infer and
                _instant(row['feature_knowledge_cutoff']) <= infer, 'prediction inference clock mismatch')
        valid = (original['member'] and all(original['validity']) and all(_finite(v) for v in original['values']) and
                 feature_available(original) is not None)
        reason = None if valid else 'NOT_MEMBER' if not original['member'] else (
            'FEATURE_AVAILABILITY_UNKNOWN' if feature_available(original) is None else 'FEATURE_MISSING')
        require(type(row['valid']) is bool and row['valid'] == valid and row['invalid_reason'] == reason and
                (_finite(row['score']) if valid else row['score'] is None), 'prediction validity/value mismatch')
    require(set(indexed) == expected_keys, 'complete prediction union required')
    require(evidence['contract_version'] == 'stock_signal_evidence_v1' and evidence['signal_ref'] == predictions['signal_run_ref'] and
            evidence['label_ref'] == evaluation['label_ref'] and evidence['evaluation_cutoff'] == spec['evaluation_cutoff'] and
            evidence['score_semantics'] == predictions['score_semantics'] and
            [r['session'] for r in evidence['series']] == features['prediction_sessions'], 'saved evidence input linkage mismatch')
    # Old artifacts retain the implementation-era Runtime capability marker.
    # Neither marker is an admission receipt for this particular saved Signal.
    require(fold['engine_admission'] in (
        {'neutral_validation': 'NOT_PERFORMED_BY_BUILDER', 'runtime': 'NOT_PERFORMED_BY_BUILDER'},
        {'neutral_validation': 'NOT_PERFORMED_BY_BUILDER', 'runtime': 'UNSUPPORTED_V2'}),
            'fold cannot claim Engine Runtime admission')
    if batch is not None:
        batch._check_sources()
    return StockMLFold(path)

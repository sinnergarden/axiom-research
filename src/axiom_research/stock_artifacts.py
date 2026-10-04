"""Saved stock ML documents; read-only loaders have no Data/ML/Core imports."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path


def digest(value):
    return 'sha256:' + sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return 'sha256:' + h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def _read(path):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out: raise ValueError('duplicate JSON key: ' + k)
            out[k] = v
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def _verify_ref(value, key):
    if value.get(key) != digest({k: v for k, v in value.items() if k != key}):
        raise ValueError('saved ' + key + ' mismatch')


@dataclass(frozen=True)
class StockMLExperiment:
    path: Path
    reused: bool = False

    def to_dict(self):
        return _read(self.path / 'experiment.json')

    @property
    def identity(self):
        return self.to_dict()['experiment_ref']

    def predictions(self):
        return _read(self.path / 'predictions.json')

    def evidence(self):
        return _read(self.path / 'signal-evidence.json')


def load_stock_ml_experiment(path):
    """Hash/ref verify frozen files. Never initialize a provider, train or replay."""
    path = Path(path)
    manifest = _read(path / 'manifest.json')
    if manifest.get('contract_version') != 'stock_ml_manifest_v1':
        raise ValueError('unsupported stock ML manifest')
    expected = {'experiment.json','features.json','feature-inputs.json','labels.json',
                'dataset.json','model.json','booster.txt','predictions.json','signal-evidence.json'}
    if set(manifest.get('files', {})) != expected: raise ValueError('unexpected saved stock files')
    for name, ref in manifest['files'].items():
        if file_digest(path / name) != ref: raise ValueError('saved stock file mismatch: ' + name)
    experiment = _read(path / 'experiment.json'); _verify_ref(experiment, 'experiment_ref')
    for name, key in (('features.json','feature_ref'), ('labels.json','label_ref'),
                      ('dataset.json','dataset_ref'),('model.json','model_ref'),
                      ('predictions.json','signal_run_ref'),('signal-evidence.json','evidence_ref')):
        value = _read(path / name); _verify_ref(value, key)
        if experiment[key] != value[key]: raise ValueError('stock experiment reference mismatch: ' + key)
    model = _read(path / 'model.json')
    features=_read(path/'features.json'); labels=_read(path/'labels.json')
    dataset=_read(path/'dataset.json'); predictions=_read(path/'predictions.json')
    evidence=_read(path/'signal-evidence.json')
    _verify_ref(labels['training'],'label_ref'); _verify_ref(labels['evaluation'],'label_ref')
    if features['input_evidence_ref']!=digest(_read(path/'feature-inputs.json')):
        raise ValueError('feature input evidence mismatch')
    for actual,expected in ((dataset['feature_ref'],features['feature_ref']),
            (dataset['label_ref'],labels['training']['label_ref']),
            (model['dataset_ref'],dataset['dataset_ref']),
            (model['feature_ref'],features['feature_ref']),
            (predictions['model_ref'],model['model_ref']),
            (predictions['feature_ref'],features['feature_ref']),
            (evidence['signal_ref'],predictions['signal_run_ref']),
            (evidence['label_ref'],labels['evaluation']['label_ref'])):
        if actual!=expected: raise ValueError('stock saved stage linkage mismatch')
    if (features['ordered_features']!=dataset['ordered_features'] or
            features['ordered_features']!=model['ordered_features'] or
            dataset['fit_cutoff']!=model['fit_cutoff']):
        raise ValueError('stock model dataset/schema mismatch')
    if model['booster_digest'] != manifest['files']['booster.txt']:
        raise ValueError('model booster reference mismatch')
    if manifest.get('experiment_ref') != experiment['experiment_ref']:
        raise ValueError('stock manifest identity mismatch')
    return StockMLExperiment(path)


def load_stock_model(path):
    """Return verified saved ModelRelease metadata, without importing LightGBM."""
    experiment = load_stock_ml_experiment(path)
    return _read(experiment.path / 'model.json')

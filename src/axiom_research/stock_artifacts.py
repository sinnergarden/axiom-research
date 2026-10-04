"""Saved stock ML documents; read-only loaders have no Data/ML/Core imports."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
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


def _canonical_file_ref(path):
    """Hash canonical writer bytes without its final LF, using bounded memory."""
    path=Path(path)
    with path.open('rb') as f:
        f.seek(-1,2)
        if f.read(1)!=b'\n': raise ValueError('saved canonical JSON requires final LF')
        remaining=f.tell()-1;f.seek(0);h=sha256()
        while remaining:
            chunk=f.read(min(1024*1024,remaining));h.update(chunk);remaining-=len(chunk)
    return 'sha256:'+h.hexdigest()


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


def _verify_normalized_labels(value):
    """Verify saved Core document bindings without loading its executor."""
    _verify_ref(value,'label_ref')
    sections=value['sections']
    groups=[value[k] for k in ('core_plan','core_facts','core_context','core_frames')]
    if any(len(items)!=len(sections) for items in groups):
        raise ValueError('normalized Core section count mismatch')
    for section,plan,facts,context,frame in zip(sections,*groups):
        definition={'contract_version':'stock_label_section_v1','feature_session':section['feature_session'],
            'eligible_keys':section['eligible_keys'],'raw_label_ref':value['raw_label_ref'],
            'feature_ref':value['feature_ref'],'cutoff':value['cutoff'],
            'normalization_spec':value['normalization_spec']}
        for actual,expected in ((section['section_ref'],digest(definition)),
                (section['core_plan_ref'],digest(plan)),(section['fact_ref'],digest(facts)),
                (section['core_context_ref'],digest(context)),(section['frame_ref'],digest(frame)),
                (frame['plan_identity'],section['core_plan_ref']),
                (frame['fact_identity'],section['fact_ref']),
                (frame['context_identity'],section['core_context_ref']),
                (frame['reference_ref'],section['section_ref']),
                (plan['reference_ref'],section['section_ref']),
                (context['reference_ref'],section['section_ref'])):
            if actual!=expected: raise ValueError('normalized Core section linkage mismatch')
    if value['frame_ref']!=digest([{'feature_session':s['feature_session'],'frame_ref':s['frame_ref']}
                                  for s in sections]):
        raise ValueError('normalized Core aggregate frame mismatch')


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
    training_labels=labels['training']
    normalized='normalized_training' in labels or 'normalized_evaluation' in labels
    if normalized:
        for name,raw_name,cutoff_name in (('normalized_training','training','fit_cutoff'),
                                         ('normalized_evaluation','evaluation','evaluation_cutoff')):
            value=labels[name];_verify_normalized_labels(value)
            cutoff=experiment['definition']['config'][cutoff_name]
            if (value.get('contract_version')!='stock_normalized_label_build_v1' or
                    value['raw_label_ref']!=labels[raw_name]['label_ref'] or
                    value['feature_ref']!=features['feature_ref'] or
                    datetime.fromisoformat(value['cutoff'].replace('Z','+00:00'))!=
                    datetime.fromisoformat(cutoff.replace('Z','+00:00'))):
                raise ValueError('stock normalized label input linkage mismatch')
        training_labels=labels['normalized_training']
        for actual,expected in ((dataset['raw_label_ref'],labels['training']['label_ref']),
                (model['raw_label_ref'],labels['training']['label_ref']),
                (model['label_ref'],training_labels['label_ref']),
                (model['feature_selection'],features['selection']),
                (model['catalog_ref'],features['catalog_ref']),
                (dataset['label_section_refs'],[s['section_ref'] for s in training_labels['sections']]),
                (dataset['normalization'],training_labels['normalization_spec']),
                (model['label_normalization'],training_labels['normalization_spec']),
                (experiment['definition']['label_normalization'],training_labels['normalization_spec']),
                (dataset['target_semantics'],model['target_semantics']),
                (predictions['score_semantics'],model['target_semantics']),
                (experiment['definition']['target_semantics'],model['target_semantics'])):
            if actual!=expected: raise ValueError('stock normalized model stage linkage mismatch')
    if features['input_evidence_ref']!=_canonical_file_ref(path/'feature-inputs.json'):
        raise ValueError('feature input evidence mismatch')
    for actual,expected in ((dataset['feature_ref'],features['feature_ref']),
            (dataset['label_ref'],training_labels['label_ref']),
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
    path=Path(path);model=_read(path/'model.json');_verify_ref(model,'model_ref')
    if model.get('contract_version')!='stock_model_release_v1':
        raise ValueError('unsupported stock model contract')
    columns=model.get('ordered_features')
    if type(columns) is not list or not columns or len(columns)!=len(set(columns)) or any(
            type(c) is not str or not c for c in columns):
        raise ValueError('saved model requires ordered feature IDs')
    if model.get('booster_digest')!=file_digest(path/'booster.txt'):
        raise ValueError('model booster reference mismatch')
    return model

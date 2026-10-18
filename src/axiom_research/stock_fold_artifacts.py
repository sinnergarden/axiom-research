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
        return saved[1][name]
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
    """Read a COMPLETE published fold without reconstructing its ancestors."""
    return _load_stock_ml_fold(path, batch=batch)


def _load_stock_ml_fold(path, *, projection=None, batch=None):
    """Internal publication check may reuse this call's verified projection."""
    path=Path(path).resolve()
    manifest=_read(path/'manifest.json')
    if manifest['contract_version']=='stock_ml_fold_manifest_v2':
        from .stock_matrix_folds import load_matrix_fold
        return load_matrix_fold(path,batch=batch)
    fold=_read(path/'fold.json')
    require(fold['status']=='COMPLETE' and fold['fold_ref']==manifest['fold_ref'],'incomplete saved fold')
    return _owned_fold(path,{'fold.json':fold,**{name:_read(path/name) for name in OUTPUTS}})

"""Axiom Research R0 public contract API."""
from .contracts import *
from .api import (
    ContractError, content_digest, contract_schema, dumps, from_dict, load, loads,
    save, semantic_identity, to_dict, unresolved, validate,
)
from .experiments import (ExperimentStore, ExperimentReader,
                          ExperimentRecordError, RevisionConflict)
from .stock_artifacts import (StockMLExperiment, load_stock_ml_experiment,
                              load_stock_model)
from .stock_stage_report import export_stock_stage_report, load_stock_stage_report
from .stock_fold_artifacts import StockMLFold, load_stock_ml_fold
from .stock_feature_inputs import StockFeatureInputs, load_stock_feature_inputs
from .stock_signal_evaluation import (StockSignalEvaluation, evaluate_stock_signal,
    evaluate_stock_signals, save_stock_signal_evaluation, load_stock_signal_evaluation)

__version__ = "0.2.5"

# Readonly metadata consumers do not import an optional Data/Core build runtime.
# Existing runtime names retain their public import paths and load when requested.
from importlib import import_module as _import_module

_RUNTIME_EXPORTS = {
    "ViewRef": ".view_ref",
    "FeatureBuild": ".joint_build", "build_joint_features": ".joint_build",
    "load_feature_build": ".joint_build",
    "RotationExperiment": ".rotation", "build_rotation_features": ".rotation",
    "build_rotation_experiment": ".rotation", "load_rotation_experiment": ".rotation",
    "build_stock_ml_experiment": ".stock_ml", "predict_stock_model": ".stock_ml",
    "build_stock_ml_from_saved_features": ".stock_ml",
    "build_stock_ml_fold_from_saved_inputs": ".stock_folds",
    "StockMLBatchInputs": ".stock_batch", "load_stock_ml_batch_inputs": ".stock_batch",
    "build_stock_feature_inputs": ".stock_feature_inputs",
    "prepare_stock_ml_batch_inputs": ".stock_matrix_prepare",
    "normalize_forward_labels": ".stock_label_normalization",
    "load_feature_catalog": ".feature_catalog", "build_feature_plan": ".feature_catalog",
    "build_forward_labels": ".labels", "mature_training_rows": ".labels",
}
__all__ = [name for name in globals() if not name.startswith("_")] + list(_RUNTIME_EXPORTS)


def __getattr__(name):
    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(name)
    value = getattr(_import_module(_RUNTIME_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_RUNTIME_EXPORTS))

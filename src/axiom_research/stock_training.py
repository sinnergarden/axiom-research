"""Shared native stock-model backend; no Feature, Data or account execution."""
from __future__ import annotations

import time
import math
from .stock_label_contracts import TARGET_SEMANTICS

LGBM_PARAMETERS = {'objective':'regression','boosting_type':'gbdt','learning_rate':0.05,
    'num_leaves':31,'max_depth':5,'min_data_in_leaf':20,'seed':42,'num_threads':1,
    'feature_fraction':1.0,'bagging_fraction':1.0,'deterministic':True,
    'force_col_wise':True,'verbosity':-1}
TREES = 100

def training_profile(options=None):
    """Resolve the two explicit knobs of the existing LightGBM profile."""
    if options is None: options = {}
    if type(options) is not dict or not set(options) <= {'learning_rate', 'num_boost_round'}:
        raise ValueError('training_options supports only learning_rate and num_boost_round')
    rate = options.get('learning_rate', LGBM_PARAMETERS['learning_rate'])
    rounds = options.get('num_boost_round', TREES)
    if type(rate) not in (int, float):
        raise ValueError('learning_rate must be a positive finite number')
    try: rate = float(rate)
    except OverflowError: raise ValueError('learning_rate must be a positive finite number') from None
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('learning_rate must be a positive finite number')
    if type(rounds) is not int or rounds <= 0:
        raise ValueError('num_boost_round must be a positive integer')
    return {**LGBM_PARAMETERS, 'learning_rate': rate}, rounds


def validate_training_profile(parameters, rounds):
    """Admit stored parameters without importing the native training library."""
    if type(parameters) is not dict:
        raise ValueError('stored LightGBM parameters must be an object')
    resolved, count = training_profile({'learning_rate': parameters.get('learning_rate'),
                                      'num_boost_round': rounds})
    if parameters != resolved or any(type(parameters[k]) is not type(v) for k,v in resolved.items()):
        raise ValueError('unsupported stored LightGBM parameters')
    return resolved, count



def fit_predict_stock_model(X, y, P, *, ordered_features, parameters,
                            num_boost_round, metrics):
    """Run the existing fixed LightGBM backend once for one declared fold.

    Date/maturity/input validation belongs to the calling Research builder.
    This function does not choose parameters, windows or portfolio policies.
    """
    import lightgbm as lgb

    begin = time.perf_counter()
    model = lgb.train(dict(parameters), lgb.Dataset(X, label=y,
                      feature_name=list(ordered_features)),
                      num_boost_round=num_boost_round)
    metrics['train_seconds'] = time.perf_counter() - begin
    metrics['train_calls'] = 1
    begin = time.perf_counter()
    scores = model.predict(P, num_threads=1) if len(P) else []
    metrics['predict_seconds'] = time.perf_counter() - begin
    metrics['predict_calls'] = 1
    return model, scores

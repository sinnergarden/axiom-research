"""Shared native stock-model backend; no Feature, Data or account execution."""
from __future__ import annotations

import time

LGBM_PARAMETERS = {'objective':'regression','boosting_type':'gbdt','learning_rate':0.05,
    'num_leaves':31,'max_depth':5,'min_data_in_leaf':20,'seed':42,'num_threads':1,
    'feature_fraction':1.0,'bagging_fraction':1.0,'deterministic':True,
    'force_col_wise':True,'verbosity':-1}
TREES = 100
TARGET_SEMANTICS = 'forward_5_session_cs_zscore_prediction'



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

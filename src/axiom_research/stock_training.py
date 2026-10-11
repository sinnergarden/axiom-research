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


def resolve_stock_training_spec(value):
    """Execute only explicitly supplied parameters of the existing backend."""
    from .contracts import TrainingSpec
    from .api import from_dict,validate,_encode,semantic_identity
    if type(value) is dict:
        from copy import deepcopy
        from .stock_experiment_config import _transport_defaults
        wire=_transport_defaults(deepcopy(value))
        value=from_dict(wire)
    if type(value) is not TrainingSpec:raise ValueError('typed TrainingSpec required')
    validate(value,require_resolved=True)
    if value.backend!='lightgbm' or value.objective!='regression' or value.fit_scope!='train_only':
        raise ValueError('stock training requires the declared LightGBM regression/train_only profile')
    if value.preprocessing not in ('none','same_date_visible_members_cs_zscore_no_fit') or value.selection_protocol!='fixed_parameters_no_validation_no_early_stopping':
        raise ValueError('this stock profile does not fit preprocessing or perform parameter selection')
    p=dict(value.parameters)
    if set(p)!={*LGBM_PARAMETERS,'n_estimators'}:raise ValueError('all supported LightGBM parameters must be explicit')
    rounds=p.pop('n_estimators');validate_explicit_training_profile(p,rounds)
    if p['seed']!=value.seed or p['objective']!=value.objective or p['num_threads']!=value.resources.threads or value.resources.concurrent_folds!=1 or type(value.resources.memory_limit_mb) is not int or value.resources.memory_limit_mb<=0:
        raise ValueError('conflicting training seed, objective or resource declaration')
    wire=_encode(value,True)
    return p,rounds,{'training_spec':wire,'training_spec_ref':semantic_identity(value)}


def validate_explicit_training_profile(parameters,rounds):
    """Readonly bounds for the explicit profile; never imports LightGBM."""
    if type(parameters) is not dict or set(parameters)!=set(LGBM_PARAMETERS) or type(rounds) is not int or rounds<=0:
        raise ValueError('unsupported explicit LightGBM parameter fields/rounds')
    p=parameters
    if p['objective']!='regression' or p['boosting_type']!='gbdt':raise ValueError('unsupported stock training objective/backend')
    for name in ('num_leaves','min_data_in_leaf','num_threads'):
        if type(p[name]) is not int or p[name]<=0:raise ValueError('positive integer training parameter required: '+name)
    if type(p['max_depth']) is not int or p['max_depth']==0 or p['max_depth'] < -1 or type(p['seed']) is not int or p['seed']<0:
        raise ValueError('invalid depth or seed')
    for name in ('learning_rate','feature_fraction','bagging_fraction'):
        try:finite=math.isfinite(p[name]) if type(p[name]) in (float,int) else False
        except OverflowError:finite=False
        if not finite or p[name]<=0 or name!='learning_rate' and p[name]>1:
            raise ValueError('invalid numerical training parameter: '+name)
    if p['deterministic'] is not True or p['force_col_wise'] is not True or type(p['verbosity']) is not int:
        raise ValueError('explicit stock model must retain deterministic column-wise execution')
    return dict(p),rounds



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
    scores = model.predict(P, num_threads=parameters['num_threads']) if len(P) else []
    metrics['predict_seconds'] = time.perf_counter() - begin
    metrics['predict_calls'] = 1
    return model, scores

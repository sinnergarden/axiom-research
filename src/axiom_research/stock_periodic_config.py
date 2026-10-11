"""Compile readable periodic configuration into the existing explicit folds.

No model, label or Feature mathematics lives here. The Label owner still
checks every outcome dependency at the actual fit cutoff. The declared window
includes the immature tail; its expected mature dates are recorded separately.
"""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

from .stock_artifacts import digest, file_digest
from .stock_fold_inputs import ordered, require, validate_spec
from .stock_label_contracts import _instant, _session
from .stock_weekly_config import _local_clock


PERIODIC_FIELDS = {'trade_start', 'trade_end_exclusive', 'mature_sessions',
    'label_horizon_sessions', 'retrain_every_sessions', 'timezone',
    'fit_time', 'model_time', 'inference_time', 'evaluation_cutoff'}


def compile_stock_periodic_folds(*, calendar, feature_sessions, trade_start,
        trade_end_exclusive, mature_sessions, label_horizon_sessions,
        retrain_every_sessions, timezone, fit_time, model_time, inference_time,
        evaluation_cutoff):
    """One fit per N sessions, fresh previous-session features every OOS day.

    For H-session endpoints, M+H-1 preceding feature dates contain M nominal
    mature dates when the fit-session endpoint is strictly available before fit.
    This is a calendar plan, not a proof of Label availability. Callers must
    inspect actual admitted mature dates before claiming M mature observations.
    """
    from zoneinfo import ZoneInfo
    ordered(calendar, 'calendar'); ordered(feature_sessions, 'Feature sessions')
    for d in calendar: _session(d)
    _session(trade_start); _session(trade_end_exclusive); _instant(evaluation_cutoff)
    require(trade_start < trade_end_exclusive, 'nonempty periodic OOS interval required')
    for n in (mature_sessions, label_horizon_sessions, retrain_every_sessions):
        require(type(n) is int and n > 0, 'positive periodic session counts required')
    try:
        zone = ZoneInfo(timezone)
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError('invalid periodic timezone') from exc
    positions = {d:i for i,d in enumerate(calendar)}
    trades = [d for d in calendar if trade_start <= d < trade_end_exclusive]
    require(bool(trades) and calendar[0] <= trade_start and
        trade_end_exclusive <= (date.fromisoformat(calendar[-1])+
        timedelta(days=1)).isoformat(), 'periodic OOS exceeds calendar')
    features = set(feature_sessions); folds = []
    length = mature_sessions + label_horizon_sessions - 1
    for start in range(0, len(trades), retrain_every_sessions):
        days = trades[start:start+retrain_every_sessions]
        require(positions[days[0]] > 0, 'missing strict previous fit session')
        fit = calendar[positions[days[0]]-1]
        predictions = [calendar[positions[d]-1] for d in days]
        spec = dict(contract_version='stock_ml_fold_spec_v3',
            training_window=dict(unit='feature_sessions', length=length,
                                 end='previous_fit_session'),
            fit_session=fit, fit_cutoff=_local_clock(fit,fit_time,zone),
            simulated_model_available_at=_local_clock(fit,model_time,zone),
            oos_trade_sessions=days,
            inference_cutoff_by_session={d:_local_clock(d,inference_time,zone) for d in predictions},
            evaluation_cutoff=evaluation_cutoff)
        training, inferred = validate_spec(spec,calendar)
        require(inferred == predictions and set(training+predictions) <= features,
                'periodic Feature coverage is incomplete')
        folds.append(spec)
    return folds


def nominal_mature_sessions(fold, calendar, horizon_sessions):
    """Expected calendar dates; actual row clocks must subsequently admit them."""
    training, _ = validate_spec(fold, calendar)
    fit = calendar.index(fold['fit_session'])
    return [d for d in training if calendar.index(d)+horizon_sessions <= fit]


def stock_evaluation_sessions(*, calendar, trade_sessions, frequency):
    """Select actual daily predictions without creating filled observations."""
    require(frequency in ('daily', 'weekly'), 'explicit daily or weekly evaluation required')
    ordered(calendar, 'calendar'); ordered(trade_sessions, 'trade sessions')
    positions = {d:i for i,d in enumerate(calendar)}
    require(all(d in positions and positions[d] > 0 for d in trade_sessions),
            'evaluation trade dates lack original predictions')
    first = {}
    for d in calendar:
        key = tuple(date.fromisoformat(d).isocalendar()[:2])
        first.setdefault(key, d)
    chosen = trade_sessions if frequency == 'daily' else [d for d in trade_sessions
        if first[tuple(date.fromisoformat(d).isocalendar()[:2])] == d]
    require(all(d in positions and positions[d] > 0 for d in chosen),
            'evaluation trade dates lack original predictions')
    return [calendar[positions[d]-1] for d in chosen]


def _expand_periodic_dataset(dataset, path):
    from .stock_experiment_config import _read_configuration_yaml
    from .stock_feature_inputs import SPEC_FIELDS
    require(type(dataset) is dict and set(dataset) == {'contract_version','axes_input',
        'periodic_schedule','preparation_options','model_feature_selection','signal_contexts'} and
        dataset['contract_version'] == 'stock_dataset_schedule_v3', 'exact periodic Dataset required')
    axes, rules = dataset['axes_input'], dataset['periodic_schedule']
    require(type(axes) is dict and set(axes) == {'path','file_digest'} and
            type(rules) is dict and set(rules) == PERIODIC_FIELDS, 'frozen axes/periodic rules required')
    axes_path = (Path(path).parent/axes['path']).resolve()
    source = _read_configuration_yaml(axes_path, expected_digest=axes['file_digest'])
    require(type(source) is dict and set(source) == SPEC_FIELDS,
            'periodic Dataset requires the original Feature spec')
    folds = compile_stock_periodic_folds(calendar=source['calendar'],
        feature_sessions=source['feature_sessions'], **rules)
    scope = dict(calendar=deepcopy(source['calendar']), universe=deepcopy(source['universe']),
        sessions=[d for f in folds for d in f['inference_cutoff_by_session']],
        evaluation_cutoff=rules['evaluation_cutoff'])
    effective = dict(contract_version='stock_dataset_schedule_v1', fold_specs=folds,
        scope=scope, **{k:deepcopy(dataset[k]) for k in
        ('preparation_options','model_feature_selection','signal_contexts')})
    compiler = digest(dict(compiler=file_digest(__file__),
        fold_validator=file_digest(Path(__file__).parent/'stock_fold_inputs.py')))
    compilation = dict(contract_version='stock_periodic_compilation_v1', compiler_ref=compiler,
        axes_input=dict(path=str(axes_path),file_digest=axes['file_digest']),
        periodic_schedule=deepcopy(rules),
        nominal_mature_sessions_by_fit={f['fit_session']:nominal_mature_sessions(f,
            source['calendar'], rules['label_horizon_sessions']) for f in folds})
    return effective, compilation

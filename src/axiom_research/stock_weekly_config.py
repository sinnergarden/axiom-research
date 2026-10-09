"""Pure calendar compilation into the existing explicit fold-v3 contract."""
from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone as utc_zone
from pathlib import Path
from zoneinfo import ZoneInfo

from .stock_artifacts import digest, file_digest
from .stock_fold_inputs import ordered, require, validate_spec
from .stock_label_contracts import _instant, _session

RULE_FIELDS = {'trade_start', 'trade_end_exclusive', 'training_window', 'timezone',
    'fit_time', 'model_time', 'inference_time', 'evaluation_cutoff'}


def _local_clock(day, clock, zone):
    require(type(clock) is str and bool(clock), 'explicit local clock string required')
    value = time.fromisoformat(clock)
    require(value.tzinfo is None, 'local clock must use the declared timezone')
    local = datetime.combine(date.fromisoformat(day), value, tzinfo=zone)
    require(local.replace(fold=0).utcoffset() == local.replace(fold=1).utcoffset() and
        local.astimezone(utc_zone.utc).astimezone(zone) == local,
        'ambiguous or nonexistent local clock is unsupported')
    return local.isoformat()


def compile_stock_weekly_folds(*, calendar, feature_sessions, trade_start, trade_end_exclusive,
    training_window, timezone, fit_time, model_time, inference_time, evaluation_cutoff):
    """Keep actual partial ISO weeks and strict previous-session predictions.

    This neither reads business sources nor changes Runtime's rebalance policy.
    The original validator owns window, leap-day and staged-clock semantics.
    """
    ordered(calendar, 'calendar'); ordered(feature_sessions, 'Feature sessions')
    for day in calendar: _session(day)
    for day in feature_sessions: _session(day)
    _session(trade_start); _session(trade_end_exclusive); _instant(evaluation_cutoff)
    require(trade_start < trade_end_exclusive, 'nonempty exclusive trading range required')
    require(calendar[0] <= trade_start and trade_end_exclusive <=
        (date.fromisoformat(calendar[-1])+timedelta(days=1)).isoformat(),
        'requested trading range exceeds the frozen calendar boundary')
    require(type(timezone) is str and bool(timezone), 'explicit IANA timezone required')
    try: zone = ZoneInfo(timezone)
    except (KeyError,ValueError) as error:
        raise ValueError('unknown explicit IANA timezone') from error
    require(type(training_window) is dict, 'explicit original training window required')
    require(set(feature_sessions) <= set(calendar), 'Feature dates outside frozen calendar')
    positions = {d:i for i,d in enumerate(calendar)}
    trades = [d for d in calendar if trade_start <= d < trade_end_exclusive]
    require(bool(trades), 'no actual trading sessions in the declared range')
    groups = {}
    for day in trades:
        require(positions[day] > 0, 'missing strict previous session')
        iso = date.fromisoformat(day).isocalendar()
        groups.setdefault((iso.year, iso.week), []).append(day)
    result = []; features = set(feature_sessions)
    for days in groups.values():
        fit = calendar[positions[days[0]]-1]
        predictions = [calendar[positions[d]-1] for d in days]
        spec = {'contract_version':'stock_ml_fold_spec_v3', 'training_window':deepcopy(training_window),
            'fit_session':fit, 'fit_cutoff':_local_clock(fit,fit_time,zone),
            'simulated_model_available_at':_local_clock(fit,model_time,zone), 'oos_trade_sessions':list(days),
            'inference_cutoff_by_session':{d:_local_clock(d,inference_time,zone) for d in predictions},
            'evaluation_cutoff':evaluation_cutoff}
        training, inferred = validate_spec(spec,calendar)
        require(inferred == predictions and set(training+predictions) <= features,
            'declared Feature sessions do not cover training and inference')
        result.append(spec)
    return result


def _expand_weekly_dataset(dataset, path):
    from .stock_experiment_config import _read_configuration_yaml
    from .stock_feature_inputs import SPEC_FIELDS
    require(type(dataset) is dict and set(dataset) == {'contract_version','axes_input','weekly_schedule',
        'preparation_options','model_feature_selection','signal_contexts'} and
        dataset['contract_version'] == 'stock_dataset_schedule_v2', 'exact short Dataset schedule required')
    axes = dataset['axes_input']; rules = dataset['weekly_schedule']
    require(type(axes) is dict and set(axes) == {'path','file_digest'} and type(axes['path']) is str and
        bool(axes['path']) and type(axes['file_digest']) is str and
        type(rules) is dict and set(rules) == RULE_FIELDS, 'exact frozen axes and weekly rules required')
    axes_path = (Path(path).parent/axes['path']).resolve()
    source = _read_configuration_yaml(axes_path, expected_digest=axes['file_digest'])
    require(type(source) is dict and set(source) == SPEC_FIELDS and
        source.get('contract_version') == 'stock_feature_inputs_spec_v1',
        'short Dataset requires the original frozen Feature spec')
    folds = compile_stock_weekly_folds(calendar=source['calendar'],feature_sessions=source['feature_sessions'],**rules)
    scope = {'calendar':deepcopy(source['calendar']), 'universe':deepcopy(source['universe']),
        'sessions':[day for fold in folds for day in fold['inference_cutoff_by_session']],
        'evaluation_cutoff':rules['evaluation_cutoff']}
    effective = {'contract_version':'stock_dataset_schedule_v1','fold_specs':folds,'scope':scope,
        **{k:deepcopy(dataset[k]) for k in ('preparation_options','model_feature_selection','signal_contexts')}}
    compiler = digest({'compiler':file_digest(__file__),
        'fold_validator':file_digest(Path(__file__).parent/'stock_fold_inputs.py')})
    compilation = {'contract_version':'stock_weekly_compilation_v1','compiler_ref':compiler,
        'axes_input':{'path':str(axes_path),'file_digest':axes['file_digest']},'weekly_schedule':deepcopy(rules)}
    return effective, compilation

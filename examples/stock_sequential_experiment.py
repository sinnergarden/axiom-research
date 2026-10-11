"""Notebook and worker share these concrete calls and explicit owner contexts.

No execution occurs on import. Supply approved frozen inputs and budgets; no
sample dates, model tuning, paths, credentials or initial capital are invented.
"""
from axiom_research import (build_stock_sequential_experiment,build_configured_stock_sequential_experiment,evaluate_stock_sequential_signals,
    stock_request_with_saved_predictions,run_saved_stock_strategy,evaluate_saved_stock_strategy)


def prepare_configured_models(data,*,snapshot,data_limits,configuration_path,saved_feature,output,metrics,progress):
    """Use the split YAML Dataset/model/Label/Signal values without defaults."""
    with data.open_column_source(snapshot=snapshot,limits=data_limits) as source:
        return build_configured_stock_sequential_experiment(data,configuration_path=configuration_path,
            feature_inputs=saved_feature,column_source=source,destination=output,metrics=metrics,progress=progress)


def prepare_models(data, *, snapshot, data_limits, configuration_paths, saved_feature,
    frozen_fold_specs, prepare_options, signal_scope, output, model_feature_selection,
    signal_contexts, metrics, progress):
    # The public source owns version selection and common-anchor adjustment.
    with data.open_column_source(snapshot=snapshot,limits=data_limits) as source:
        return build_stock_sequential_experiment(data,configuration_paths=configuration_paths,
            feature_inputs=saved_feature,fold_specs=frozen_fold_specs,column_source=source,
            preparation_options=prepare_options,scope=signal_scope,destination=output,
            model_feature_selection=model_feature_selection,signal_contexts=signal_contexts,
            metrics=metrics,progress=progress)


def evaluate_predictions(experiment, *, signal_scope, output):
    return evaluate_stock_sequential_signals(experiment,scope=signal_scope,destination=output)


def backtest_saved_signals(experiment, *, frozen_strategy_request, admitted_market,
    original_stock_source, output, block_sessions, runtime_limits, max_signal_bytes):
    # Portfolio/fees/clock/initial-account fields stay in the declared request.
    request=stock_request_with_saved_predictions(frozen_strategy_request,
        prediction_bindings=experiment['prediction_bindings'])
    return run_saved_stock_strategy(request,market=admitted_market,source=original_stock_source,
        destination=output,block_sessions=block_sessions,limits=runtime_limits,max_signal_bytes=max_signal_bytes)


def evaluate_account(run_path, *, projection_limits, benchmark, evaluation_spec,
    dividend_scope, output, artifact_reader):
    return evaluate_saved_stock_strategy(run_path,limits=projection_limits,benchmark=benchmark,
        spec=evaluation_spec,dividend_scope=dividend_scope,destination=output,artifact_reader=artifact_reader)

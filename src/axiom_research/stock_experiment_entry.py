"""Concrete Notebook/worker calls over the existing Research and Runtime owners.

These functions introduce no runner, process monitor, source selector or account
implementation. The caller declares frozen folds, scopes, original market input
and budgets, and owns authorization and the public Data/market/source contexts.
"""
from copy import deepcopy
from pathlib import Path

from .stock_fold_inputs import require
from .stock_artifacts import digest


def build_stock_sequential_experiment(data, *, configuration_paths, feature_inputs,
    fold_specs, column_source, preparation_options, scope, destination,
    model_feature_selection=None, signal_contexts=None, metrics=None, progress=None):
    """Prepare/build/freeze a sequence through one shared Feature/source owner.

    The executable configuration roles here are label/model and optional signal.
    Other R0 draft roles are rejected rather than silently treated as executable
    Dataset schedules or portfolio policies. Actual fold/window/clock contracts
    are explicit stock_ml_fold_spec_v3 (legacy v1/v2 remain valid).
    A configured signal is one declared raw input from each newly saved fold;
    multi-model combinations use build_stock_derived_signal with their original
    independently saved parents. No statistics or account execute in this call.
    """
    from .stock_experiment_config import load_stock_experiment_specs
    from .stock_sequential_api import open_stock_ml_batch_preparation,open_stock_signal_evaluation_freeze
    from .stock_folds import build_stock_ml_fold_from_saved_inputs
    from .stock_derived_signal import build_stock_derived_signal,_parent
    from .api import semantic_identity
    config=load_stock_experiment_specs(configuration_paths);specs=config['specs']
    require({'label','model'}<=set(specs)<= {'label','model','signal'},
        'this stock entry executes label/model/signal only; Dataset/Feature/strategy drafts require explicit owner contracts')
    require(column_source is not None,'caller-owned public ColumnSource required')
    plan=specs.get('signal')
    require((plan is None and signal_contexts is None) or
        (plan is not None and len(plan.inputs)==1 and type(signal_contexts) is dict and
         set(signal_contexts)=={digest(f) for f in fold_specs} and
         semantic_identity(plan.inputs[0].label)==config['spec_refs']['label']),
        'one original raw parent and complete declared Signal contexts required for this entry')
    root=Path(destination).resolve();folds=[];bindings=[];derived=[]
    with open_stock_ml_batch_preparation(data,feature_inputs=feature_inputs,fold_specs=fold_specs,
        destination=root/'prepared',preparation_options=preparation_options,label_spec=specs['label'],
        column_source=column_source,model_feature_selection=model_feature_selection,metrics=metrics,
        progress=progress) as owner:
        with open_stock_signal_evaluation_freeze(owner.batch,scope=scope,destination=root/'oos',signal_name='model') as writer:
            while (item:=owner.next_fold()) is not None:
                run=build_stock_ml_fold_from_saved_inputs(item['input_manifest'],fold_spec=item['fold_spec'],
                    destination=root/'folds',batch=owner.batch,training_spec=specs['model'])
                # Read the builder's actual saved outputs. No fabricated model or
                # synthetic combined prediction replaces the original fold refs.
                prediction,binding,_=_parent(run)
                if plan is not None:
                    saved=build_stock_derived_signal(plan,prediction_inputs={plan.inputs[0].alias:run.path},
                        context=signal_contexts[digest(item['fold_spec'])],destination=root/'derived',batch=owner.batch)
                    bindings.append(saved.engine_input_binding())
                    derived.append({'path':str(saved.path),'signal_run_ref':saved.identity})
                else:bindings.append(binding)
                folds.append({'path':str(run.path),'fold_ref':run.identity,
                    'signal_run_ref':prediction['signal_run_ref'],'fold_spec_ref':digest(item['fold_spec'])})
                run=prediction=binding=None
            batch=owner.finish();frozen=writer.finish()
    return {'configuration_ref':config['configuration_ref'],'spec_refs':config['spec_refs'],
        'batch_manifest':batch,'folds':folds,'derived_signals':derived,
        'prediction_bindings':bindings,'signal_evaluation_input':frozen}


def evaluate_stock_sequential_signals(experiment, *, scope, destination):
    """Saved raw OOS all/by-year statistics through the existing Core owner.

    Derived signals retain separate refs and stage. They are not substituted
    into this raw admission; a joint frozen comparison requires its own owner
    input contract. No preparation, fit, prediction or account is repeated.
    """
    from .stock_signal_evaluation_projection import evaluate_stock_signal_input_periods
    return evaluate_stock_signal_input_periods(experiment['signal_evaluation_input'],scope=scope,destination=destination)


def stock_request_with_saved_predictions(request, *, prediction_bindings):
    """Replace only a declared request's Signal inputs with original saved refs."""
    from axiom_engine.runtime import BacktestRequest
    from axiom_engine.runtime.stock_stream_contracts import logical_ref,validate_manifest
    plan=request.to_dict() if isinstance(request,BacktestRequest) else deepcopy(request)
    require(type(prediction_bindings) is list and bool(prediction_bindings),'original saved prediction bindings required')
    plan['prediction_input']={'contract_version':'stock_prediction_input_refs_v2','frames':deepcopy(prediction_bindings)}
    plan['prediction_input']['prediction_ref']=logical_ref(plan['prediction_input'],'prediction_ref')
    plan['request_ref']=logical_ref(plan,'request_ref')
    resolved=BacktestRequest.from_dict(plan);validate_manifest(resolved)
    return resolved


def run_saved_stock_strategy(request, *, market, source, destination, block_sessions,
    limits, max_signal_bytes):
    """Consume frozen inputs using the sole admitted-market/Runtime account path.

    The market and source remain caller-owned. A new strategy request must use
    a new destination; this call does not implement account crash recovery.
    """
    from axiom_engine.runtime import (bind_stock_prediction_inputs,StockResultSink,
        stock_run_id,run_stock_backtest,save_backtest_run)
    root=Path(destination).resolve()
    with bind_stock_prediction_inputs(market,request,source=source,limits=limits,
        max_signal_bytes=max_signal_bytes) as inputs:
        sink=StockResultSink(root/'parts',run_id=stock_run_id(request.to_dict()))
        run=run_stock_backtest(request,source=inputs,sink=sink,block_sessions=block_sessions,limits=limits)
        save_backtest_run(run,root/'run.json')
    return root/'run.json'


def evaluate_saved_stock_strategy(run_path, *, limits, benchmark, spec, dividend_scope,
    destination, artifact_reader):
    """Load saved committed results and call the original account evaluator."""
    from axiom_engine.runtime import load_stock_backtest_projection,evaluate_backtest,save_backtest_evaluation
    run=load_stock_backtest_projection(run_path,artifact_reader=artifact_reader,limits=limits)
    report=evaluate_backtest(run,benchmark=benchmark,spec=spec,dividend_scope=dividend_scope)
    path=Path(destination).resolve();save_backtest_evaluation(report,path)
    return path

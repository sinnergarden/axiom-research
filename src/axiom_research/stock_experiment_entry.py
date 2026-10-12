"""Concrete Notebook/worker calls over the existing Research and Runtime owners.

These functions introduce no runner, process monitor, source selector or account
implementation. The caller declares frozen folds, scopes, original market input
and budgets, and owns authorization and the public Data/market/source contexts.
"""
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path

from .stock_fold_inputs import require
from .stock_artifacts import digest


def build_configured_stock_sequential_experiment(data=None,*,configuration_path,feature_inputs,
    column_source=None,destination,reuse_raw_from_batch=None,metrics=None,progress=None,registration=None,
    prepared_destination=None,prepared_batch=None):
    """Configured entry with independent prepared inputs and model output location.

    Supply public Feature and COMPLETE batch owners for an unchanged Label,
    schedule and Feature selection; Data/ColumnSource are then unnecessary.
    A changed subset/cohort uses preparation with the
    existing reuse_raw_from_batch path; saved normalized targets are not relabeled.
    """
    from .stock_experiment_config import load_stock_sequential_configuration
    configuration=load_stock_sequential_configuration(configuration_path);dataset=configuration['dataset']
    from .result_registration import (begin_stock_result_registration, publish_stock_result,
        _save_stock_sequential_receipt, _record_stock_result_failure)
    root=Path(destination).resolve().parent
    context=registration or dict(request_path=root/'.result-request.json',
        roots={'research':str(Path(configuration_path).resolve().parent)},
        index_path=Path(configuration_path).resolve().parent/'index.json')
    if not Path(context['request_path']).exists():
        begin_stock_result_registration(configuration,**context)
    else:
        from .stock_result_projection import load_stock_result_configuration
        require(load_stock_result_configuration(context['request_path']).wire['configuration']['configuration_ref']
            == configuration['configuration_ref'],'retry configuration changed')
    publish_stock_result(**context)
    try:
        result=_build_stock_sequential_experiment(data,configuration=configuration['specification'],
            feature_inputs=feature_inputs,fold_specs=dataset['fold_specs'],column_source=column_source,
            preparation_options=dataset['preparation_options'],scope=dataset['scope'],destination=destination,
            model_feature_selection=dataset['model_feature_selection'],signal_contexts=dataset['signal_contexts'],
            reuse_raw_from_batch=reuse_raw_from_batch,metrics=metrics,progress=progress,
            prepared_destination=prepared_destination,prepared_batch=prepared_batch)
        result.update(configuration_ref=configuration['configuration_ref'],dataset_configuration_ref=configuration['dataset_ref'],
            configuration_files=configuration['configuration_files'])
        saved_path=_save_stock_sequential_receipt(result,destination=root,metrics=metrics)
        publish_stock_result(**context,experiment_result=saved_path)
        return result
    except Exception as exc:
        _record_stock_result_failure(**context,failure=type(exc).__name__+': '+str(exc))
        raise


def build_stock_sequential_experiment(data, *, configuration_paths, feature_inputs,
    fold_specs, column_source, preparation_options, scope, destination,
    model_feature_selection=None, signal_contexts=None, reuse_raw_from_batch=None,metrics=None, progress=None,
    prepared_destination=None,prepared_batch=None):
    """Load typed specs once and run the concrete shared preparation sequence."""
    from .stock_experiment_config import load_stock_experiment_specs
    return _build_stock_sequential_experiment(data,configuration=load_stock_experiment_specs(configuration_paths),
        feature_inputs=feature_inputs,fold_specs=fold_specs,column_source=column_source,
        preparation_options=preparation_options,scope=scope,destination=destination,
        model_feature_selection=model_feature_selection,signal_contexts=signal_contexts,
        reuse_raw_from_batch=reuse_raw_from_batch,metrics=metrics,progress=progress,
        prepared_destination=prepared_destination,prepared_batch=prepared_batch)


def _build_stock_sequential_experiment(data, *, configuration, feature_inputs,
    fold_specs, column_source, preparation_options, scope, destination,
    model_feature_selection=None, signal_contexts=None, reuse_raw_from_batch=None,metrics=None, progress=None,
    prepared_destination=None,prepared_batch=None):
    """Prepare/build/freeze a sequence through one shared Feature/source owner.

    The executable configuration roles here are label/model and optional signal.
    Other R0 draft roles are rejected rather than silently treated as executable
    Dataset schedules or portfolio policies. Actual fold/window/clock contracts
    are explicit stock_ml_fold_spec_v3 (legacy v1/v2 remain valid).
    A configured signal is one declared raw input from each newly saved fold;
    multi-model combinations use build_stock_derived_signal with their original
    independently saved parents. No statistics or account execute in this call.
    """
    from .stock_sequential_api import open_stock_ml_batch_preparation,open_stock_signal_evaluation_freeze
    from .stock_folds import build_stock_ml_fold_from_saved_inputs
    from .stock_derived_signal import build_stock_derived_signal,_parent
    from .api import semantic_identity
    config=configuration;specs=config['specs']
    require({'label','model'}<=set(specs)<= {'label','model','signal'},
        'this stock entry executes label/model/signal only; Dataset/Feature/strategy drafts require explicit owner contracts')
    require(prepared_batch is not None or column_source is not None,'caller-owned public ColumnSource required for preparation')
    require(prepared_batch is None or reuse_raw_from_batch is None,'choose COMPLETE prepared inputs or Raw reuse preparation')
    plan=specs.get('signal')
    require((plan is None and signal_contexts is None) or
        (plan is not None and len(plan.inputs)==1 and type(signal_contexts) is dict and
         set(signal_contexts)=={digest(f) for f in fold_specs} and
         semantic_identity(plan.inputs[0].label)==config['spec_refs']['label']),
        'one original raw parent and complete declared Signal contexts required for this entry')
    root=Path(destination).resolve();folds=[];bindings=[];derived=[]
    prepared=None
    if prepared_batch is not None:
        prepared=_compatible_prepared_batch(prepared_batch,feature_inputs,fold_specs,preparation_options,
            specs['label'],model_feature_selection)
        if metrics is not None:
            metrics.update(prepared_batch_reused=True,prepared_batch_ref=prepared['batch_ref'],
                data_read_calls=0,supplier_calls=0,feature_core_calls=0,label_core_calls=0,core_calls=0,
                raw_operator_calls=0,raw_cache_hits=0,normalized_cache_hits=0,account_calls=0)
    preparation=(open_stock_ml_batch_preparation(data,feature_inputs=feature_inputs,fold_specs=fold_specs,
        destination=prepared_destination if prepared_destination is not None else root/'prepared',
        preparation_options=preparation_options,label_spec=specs['label'],column_source=column_source,
        model_feature_selection=model_feature_selection,metrics=metrics,reuse_raw_from_batch=reuse_raw_from_batch,
        progress=progress) if prepared_batch is None else nullcontext(None))
    with preparation as owner:
        inputs=owner.batch if owner is not None else prepared_batch
        with open_stock_signal_evaluation_freeze(inputs,scope=scope,destination=root/'oos',signal_name='model') as writer:
            for ordinal in range(len(fold_specs)):
                with owner.operation() if owner is not None else inputs.operation():
                    item=owner.next_fold() if owner is not None else prepared['folds'][ordinal]
                    require(item is not None,'prepared fold coverage is incomplete')
                    run=build_stock_ml_fold_from_saved_inputs(item['input_manifest'],fold_spec=item['fold_spec'],
                        destination=root/'folds',batch=inputs,training_spec=specs['model'],
                        model_feature_selection=model_feature_selection)
                    # Read the builder's actual saved outputs. No fabricated model or
                    # synthetic combined prediction replaces the original fold refs.
                    prediction,binding,_=_parent(run)
                    if plan is not None:
                        saved=build_stock_derived_signal(plan,prediction_inputs={plan.inputs[0].alias:run.path},
                            context=signal_contexts[digest(item['fold_spec'])],destination=root/'derived',batch=inputs)
                        bindings.append(saved.engine_input_binding())
                        derived.append({'path':str(saved.path),'signal_run_ref':saved.identity})
                    else:bindings.append(binding)
                    folds.append({'path':str(run.path),'fold_ref':run.identity,
                        'signal_run_ref':prediction['signal_run_ref'],'fold_spec_ref':digest(item['fold_spec'])})
                    run=prediction=binding=None
            if owner is not None:
                require(owner.next_fold() is None,'prepared fold coverage exceeds requested schedule')
            batch=owner.finish() if owner is not None else prepared
            frozen=writer.finish()
    evaluation_input=frozen
    if derived:
        from .stock_signal_evaluation_derived import save_stock_derived_signal_evaluation_inputs
        evaluation_input=save_stock_derived_signal_evaluation_inputs(frozen,
            derived_inputs={'derived':[item['path'] for item in derived]},scope=scope,
            destination=root/'joint-oos',maximum_resident_bytes=preparation_options['maximum_resident_bytes'])
    return {'configuration_ref':config['configuration_ref'],'spec_refs':config['spec_refs'],
        'batch_manifest':batch,'folds':folds,'derived_signals':derived,
        'prediction_bindings':bindings,'raw_signal_evaluation_input':frozen,'signal_evaluation_input':evaluation_input}


def _compatible_prepared_batch(batch, feature, folds, options, label, selection):
    """Validate the original prepared definition before any new fit is started."""
    from .stock_batch import StockMLBatchInputs
    from .stock_compact_store import StockFeatureView, model_feature_binding
    from .stock_target_spec import resolve_stock_label_spec
    require(type(batch) is StockMLBatchInputs,'publicly admitted prepared batch required')
    require(type(feature) is StockFeatureView,'publicly admitted Feature owner required for prepared reuse')
    batch._check_sources(); manifest=batch.to_dict(); definition=manifest['definition']
    require(manifest['contract_version']=='stock_ml_batch_inputs_v5' and manifest['status']=='COMPLETE',
            'COMPLETE column prepared inputs required')
    require(definition['fold_specs']==folds and [f['fold_spec'] for f in manifest['folds']]==folds,
            'prepared fold schedule differs from requested inputs')
    require(definition['feature_view']==feature.to_dict(),'prepared Feature identity differs from requested inputs')
    require(definition.get('model_feature_selection')==model_feature_binding(feature,selection),
            'prepared Feature selection/cohort differs; prepare the new selection using original Raw targets')
    require(definition['target_spec']==resolve_stock_label_spec(label),'prepared Label definition differs from requested inputs')
    require(definition['preparation_options']=={**options,'control_layout':options.get('control_layout','fold_controls_v2')},
            'prepared options differ from requested inputs')
    return manifest


def evaluate_stock_sequential_signals(experiment, *, scope, destination,registration=None):
    """Saved Raw/Derived all/by-year statistics through the existing Core owner.

    An optional joint input retains separate stages and original parent refs.
    No preparation, fit, prediction, normalization or account is repeated.
    """
    from .stock_signal_evaluation_projection import evaluate_stock_signal_input_periods
    result=evaluate_stock_signal_input_periods(experiment['signal_evaluation_input'],scope=scope,destination=destination)
    if registration is not None:
        from .stock_artifacts import write_json
        from .result_registration import publish_stock_result
        path=Path(destination)/'evaluation-result.json'
        descriptors=lambda group:{name:{'path':str(report.path),'evidence_ref':report.identity}
                                  for name,report in group.items()}
        wire={'statistics':descriptors(result['all']),
              'by_year':{year:descriptors(group) for year,group in result['by_year'].items()}}
        write_json(path,wire)
        publish_stock_result(**registration,signal_evaluation=path)
    return result


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
    limits, max_signal_bytes, registration=None):
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
    if registration is not None:
        from .result_registration import attach_stock_account
        attach_stock_account(run_path=root/'run.json', **registration)
    return root/'run.json'


def evaluate_saved_stock_strategy(run_path, *, limits, benchmark, spec, dividend_scope,
    destination, artifact_reader, registration=None):
    """Load saved committed results and call the original account evaluator."""
    from axiom_engine.runtime import load_stock_backtest_projection,evaluate_backtest,save_backtest_evaluation
    run=load_stock_backtest_projection(run_path,artifact_reader=artifact_reader,limits=limits)
    report=evaluate_backtest(run,benchmark=benchmark,spec=spec,dividend_scope=dividend_scope)
    path=Path(destination).resolve();save_backtest_evaluation(report,path)
    if registration is not None:
        from .result_registration import attach_stock_account
        attach_stock_account(run_path=run_path, evaluation_paths=[path], **registration)
    return path

"""A concrete configuration-driven case over the existing public owners.

Run each stage explicitly. This file adds no executor or process supervisor.
The caller supplies local Data, immutable axes/config paths and byte budgets.
prepared_destination keeps prepared inputs separate from output_root. An
explicit prepared_batch path consumes a saved COMPLETE batch without preparing
Label/proof inputs again; changed selections use reuse_raw_from_batch instead.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--options', required=True, help='Local path/budget JSON')
    parser.add_argument('--stage', required=True, choices=('validate','features','train','evaluate'))
    args = parser.parse_args(); options = read(args.options)
    from axiom_research import load_stock_sequential_configuration
    configured = load_stock_sequential_configuration(options['experiment_file'])
    output = Path(options['output_root']); output.mkdir(parents=True,exist_ok=True)
    registration = None
    if args.stage != 'validate':
        from axiom_research import begin_stock_result_registration, publish_stock_result
        discovery = options.get('result_discovery', {})
        roots = {'research': discovery.get('research_root', str(output.parent.resolve()))}
        if discovery.get('engine_root'):
            roots['engine'] = discovery['engine_root']
        request_path = output/'.result-request.json'
        registration = dict(request_path=request_path,
            index_path=Path(roots['research'])/'index.json', roots=roots)
        begin_stock_result_registration(configured, **registration,
            question=discovery.get('question'), version=discovery.get('version'),
            required_stages=discovery.get('required_stages', ['training','signal_evaluation']),
            strategy_refs=discovery.get('strategy_refs', []),
            imported=args.stage == 'evaluate' and not request_path.exists())
        publish_stock_result(**registration)
    try:
        execute(args, options, configured, output, registration)
    except Exception as exc:
        if registration is not None:
            try:
                from axiom_research.result_registration import _record_stock_result_failure
                _record_stock_result_failure(**registration, failure=type(exc).__name__+': '+str(exc))
            except Exception as registration_error:
                import sys
                print('Result registration failed: '+str(registration_error), file=sys.stderr)
        raise


def execute(args, options, configured, output, registration):
    from axiom_research import (load_stock_sequential_configuration, to_dict,
        build_stock_feature_inputs, load_stock_feature_view,
        load_stock_ml_batch_inputs, load_stock_ml_batch_manifest,
        build_configured_stock_sequential_experiment,
        stock_evaluation_sessions, evaluate_stock_signal_input_analysis)
    from axiom_research.stock_artifacts import write_json
    if args.stage == 'validate':
        print(json.dumps({'configuration_ref':configured['configuration_ref'],
            'folds':len(configured['dataset']['fold_specs']),
            'prediction_dates':len(configured['dataset']['scope']['sessions'])}))
        return
    if args.stage == 'evaluate':
        from axiom_research.result_registration import _registered_sequential_path
        saved_path = _registered_sequential_path(**registration) if registration is not None else None
        saved_path = saved_path or Path(options['experiment_result'])
        saved = read(saved_path); scope = configured['dataset']['scope']
        trades = [d for f in configured['dataset']['fold_specs'] for d in f['oos_trade_sessions']]
        scope = {**scope, 'sessions':stock_evaluation_sessions(calendar=scope['calendar'],
            trade_sessions=trades,frequency=options['evaluation']['frequency'])}
        result = evaluate_stock_signal_input_analysis(saved['signal_evaluation_input'],
            scope=scope,destination=output/'evaluation',group_spec=options['evaluation']['group_spec'])
        result['statistics'] = {name:{'path':str(report.path),'evidence_ref':report.identity}
            for name,report in result['statistics'].items()}
        write_json(output/'evaluation-result.json',result)
        if registration is not None:
            from axiom_research import publish_stock_result
            publish_stock_result(**registration, experiment_result=saved_path,
                signal_evaluation=output/'evaluation-result.json')
        return
    if args.stage == 'features':
        from axiom_data import Data
        data = Data(options['data_root'])
        feature = build_stock_feature_inputs(data,spec=read(options['feature_spec']),
            destination=output/'features',storage_options=options['feature_storage_options'],
            column_source=options.get('feature_column_source'),
            progress=lambda event:print(json.dumps(event),flush=True))
        write_json(output/'feature-result.json',{'path':str(feature.path),
            'feature_inputs_ref':feature.identity,'reused':feature.reused})
        return
    preparation = configured['dataset']['preparation_options']
    limits = {'maximum_matrix_bytes':preparation['maximum_resident_bytes']}
    limits.update({name:preparation[name] for name in ('maximum_parent_bytes','maximum_source_bytes')
        if name in preparation})
    reuse_manifest = read(options['reuse_raw_from_batch']) if options.get('reuse_raw_from_batch') else None
    with ExitStack() as owners:
        if options.get('feature_library_training_view'):
            from axiom_research import load_feature_library_training_view
            feature=owners.enter_context(load_feature_library_training_view(options['feature_library_training_view'],limits=limits))
        else:
            feature_path = read(options['feature_result'])['path']
            feature=owners.enter_context(load_stock_feature_view(feature_path,limits=limits,residency='sequential'))
        reuse = owners.enter_context(load_stock_ml_batch_inputs(reuse_manifest,
            feature_inputs=feature,limits=limits,residency='sequential')) if reuse_manifest is not None else None
        prepared=owners.enter_context(load_stock_ml_batch_inputs(load_stock_ml_batch_manifest(options['prepared_batch']).wire,
            feature_inputs=feature,limits=limits,residency='sequential')) if options.get('prepared_batch') else None
        data=source=None
        if prepared is None:
            from axiom_data import Data
            data=Data(options['data_root'])
            snapshot=feature.to_dict()['spec']['snapshot']
            source=owners.enter_context(data.open_column_source(snapshot=snapshot,limits=options['data_limits']))
        metrics = {}
        result = build_configured_stock_sequential_experiment(data,
            configuration_path=options['experiment_file'],feature_inputs=feature,
            column_source=source,destination=output/'training',reuse_raw_from_batch=reuse,
            metrics=metrics,progress=lambda event:print(json.dumps(event),flush=True),
            registration=registration,prepared_destination=options.get('prepared_destination'),prepared_batch=prepared)
    # The configured producer has already saved and registered an immutable
    # receipt group. The CLI must not rewrite files referenced by its head.


if __name__ == '__main__': main()

"""One pending bounded saved-Feature preparation and shared four-fold run.

The existing monitor owns process/time/RSS limits. Observers forward the exact
owner arguments/results and never re-read a panel or compute a target/model.
No stage may launch until the parent releases the comparison and source review.
"""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN = Path(os.environ.get('AXIOM_FOURFOLD_PLAN', str(HERE / 'window.json')))
cfg = json.loads(PLAN.read_text())
loader = importlib.util.spec_from_file_location('existing_bounded_monitor', cfg['monitor_path'])
monitor = importlib.util.module_from_spec(loader)
loader.loader.exec_module(monitor)
monitor.__file__ = __file__
monitor.PLAN = PLAN
monitor.SOURCES = {k:(Path(cfg['source_paths'][k]), v) for k,v in cfg['source_refs'].items()}
read, save = monitor.read, monitor.save


class SemanticBoundaryError(ValueError):
    pass




def require(ok, why):
    if not ok:
        raise SemanticBoundaryError(why)


def digest(value):
    return 'sha256:' + sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def instant(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(isinstance(value, datetime) and value.tzinfo is not None, 'aware cutoff required')
    return value.astimezone(timezone.utc)


def released():
    auth = cfg['authorization']
    require(auth['status'] == 'APPROVED' and all(auth[k] for k in (
        'optimization_window_released', 'dual_source_review_complete', 'parent_window_authorized')),
        'parent release/source dual review/window authorization pending; no launch')
    require(monitor.file_ref(cfg['monitor_path']) == cfg['monitor_file_digest'], 'monitor bytes changed')


def inputs():
    sys.path[:0] = [str(root/'src') for root,_ in monitor.SOURCES.values()]
    require(monitor.file_ref(monitor.OLD_PLAN) == cfg['input_refs']['rolling_plan_file'], 'rolling plan changed')
    require(monitor.file_ref(monitor.OLD_SPEC) == cfg['input_refs']['feature_spec_file'], 'Feature spec changed')
    old = read(monitor.OLD_PLAN)
    require([f['fold_spec'] for f in old['folds']] == cfg['fold_specs'], 'fold specs changed')
    require([digest(s) for s in cfg['fold_specs']] == cfg['fold_spec_refs'], 'fold spec refs changed')
    require(old['parameters'] == cfg['parameters'] and old['trees'] == cfg['trees'] == 100,
            'fixed model parameters changed')
    return old


def phase(stage):
    out = Path(cfg['new_output'])
    started = time.monotonic()
    state = {'pid':os.getpid(), 'status':'RUNNING', 'stage':stage,
        'started_monotonic':started, 'stage_started_monotonic':started}
    def emit(**values):
        state.update(values, wall_seconds=time.monotonic()-started)
        save(out/'worker-progress.json', state)
    emit(component='initializing')
    return out, started, state, emit


def prepare_child(*, consume=False):
    """One v3 Feature admission through prepare, saved batch and four folds."""
    old = inputs()
    out, started, state, emit = phase('prepare')
    observations = {'feature_admission_calls':0, 'data_read_calls':0}
    query_log = []
    try:
        from axiom_data import Data
        from axiom_research import (load_stock_feature_view, prepare_stock_ml_batch_inputs,
                                    load_stock_ml_batch_inputs)
        data = Data(old['data_root'], cache_bytes=0)
        native_read = data.read
        limits = dict(cfg['batch_limits'])
        limits['maximum_matrix_bytes'] = cfg['prepare_options']['maximum_resident_bytes']
        tick = time.monotonic()
        with load_stock_feature_view(cfg['feature_path'], limits=limits) as feature:
            observations['feature_admission_calls'] = 1
            view = feature.to_dict()
            require(view['historical_feature_inputs_ref'] == cfg['input_refs']['feature_inputs_ref'] and
                    view['historical_content_digest'] == cfg['input_refs']['feature_content_digest'] and
                    view['spec'] == read(monitor.OLD_SPEC), 'original saved Feature identity/spec differs')
            save(out/'feature-admission-receipt.json', {'wall_seconds':time.monotonic()-tick,
                'public_api':'axiom_research.load_stock_feature_view', 'feature_view_ref':feature.identity,
                'historical_feature_inputs_ref':view['historical_feature_inputs_ref'],
                'source_index':view['source_index'], 'metrics':feature.metrics})
            def observed_read(*args, **kwargs):
                began = time.monotonic(); value = native_read(*args, **kwargs)
                observations['data_read_calls'] += 1
                query = value.context['query']
                require(kwargs['snapshot']==cfg['snapshot'] and query['purpose']=='label_outcomes' and
                        query['pit_policy']==cfg['pit_policy'] and query['symbols']==old['universe'] and
                        kwargs['query'].domain in ('market_daily','adjustment_factors'), 'Label Query scope differs')
                allowed = {instant(f['fit_cutoff']) for f in cfg['fold_specs']} | {instant(f['evaluation_cutoff']) for f in cfg['fold_specs']}
                require(len({instant(v) for v in query['cutoff_by_session'].values()})==1 and
                        {instant(v) for v in query['cutoff_by_session'].values()}<=allowed, 'Label Query cutoff differs')
                query_log.append({'call':len(query_log)+1,'query':deepcopy(query),'query_ref':digest(query),
                                  'wall_seconds':time.monotonic()-began,'row_count':int(value.frame.shape[0])})
                save(out/'data-query-receipts.json',query_log)
                return value
            data.read = observed_read
            metrics = {}
            def progress(value): emit(component=value['stage'],completed_folds=value['completed'])
            manifest = prepare_stock_ml_batch_inputs(data, feature_inputs=feature,
                fold_specs=cfg['fold_specs'], destination=out/'prepared',
                preparation_options=cfg['prepare_options'], metrics=metrics, progress=progress)
            require(manifest['contract_version']=='stock_ml_batch_inputs_v3' and manifest['status']=='COMPLETE' and
                    manifest['definition']['fold_specs']==cfg['fold_specs'], 'prepared v3 contract differs')
            path = out/'prepared'/manifest['definition_ref'][7:]/'batch.json'
            locator = {'path':str(path),'file_digest':monitor.file_ref(path),'batch_ref':manifest['batch_ref'],
                       'prepared_view_ref':manifest['prepared_view']['prepared_view_ref']}
            save(out/'batch-locator.json',locator)
            save(out/'prepare-receipt.json', {'status':'COMPLETE','wall_seconds':time.monotonic()-started,
                'batch':locator,'metrics':metrics,'observations':observations,
                'Data_calls':metrics['data_read_calls'],'Raw_calls':metrics['raw_operator_calls'],
                'Core_calls':metrics['core_calls'],'supplier_calls':0,'Feature_calls':0,
                'fit_calls':0,'predict_calls':0,'account_calls':0})
            emit(status='STAGE_COMPLETE',component='prepared_batch_saved')
            if consume:
                with load_stock_ml_batch_inputs(manifest, feature_inputs=feature, limits=cfg['batch_limits']) as batch:
                    folds_child(batch=batch)
                handoff=read(out/'handoff.json'); handoff['batch_closed']=True; save(out/'handoff.json',handoff)
    except BaseException as exc:
        emit(status='FAILED',error_type=type(exc).__name__,error=str(exc)); traceback.print_exc(); raise
    finally:
        save(out/'prepare-exit.json', {'status':state['status'],'wall_seconds':time.monotonic()-started,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'observations':observations})




def folds_child(*, batch=None):
    owns_batch = batch is None
    inputs()
    out, started, state, emit = phase('folds')
    observations = {'shared_batch_admissions':0, 'backend_calls':0}
    finished = []
    backend = None
    native_backend = None
    try:
        from axiom_research import (load_stock_ml_batch_inputs,
            build_stock_ml_fold_from_saved_inputs, load_stock_ml_fold)
        from axiom_research import stock_training as backend
        require(backend.LGBM_PARAMETERS==cfg['parameters'] and backend.TREES==cfg['trees'],
            'original native model parameters differ')
        locator = read(cfg['recovery']['prepared_locator'] or out/'batch-locator.json')
        require(monitor.file_ref(locator['path'])==locator['file_digest'], 'batch manifest changed')
        manifest = read(locator['path'])
        require(manifest['batch_ref']==locator['batch_ref'] and manifest['status']=='COMPLETE' and
            manifest['definition']['fold_specs']==cfg['fold_specs'] and
            manifest['definition']['feature_view']['historical_feature_inputs_ref']==cfg['input_refs']['feature_inputs_ref'],
            'prepared batch or original Feature identity differs')
        native_backend = backend.fit_predict_stock_model
        def observed_backend(*args, **kwargs):
            observations['backend_calls'] += 1
            require(observations['backend_calls']<=4 and kwargs['parameters']==cfg['parameters'] and
                kwargs['num_boost_round']==100, 'unexpected training count/parameters')
            return native_backend(*args, **kwargs)
        backend.fit_predict_stock_model = observed_backend
        emit(component='shared_batch_admission', completed_folds=0)
        began = time.monotonic()
        if batch is None:
            batch = load_stock_ml_batch_inputs(manifest, limits=cfg['batch_limits'])
        observations['shared_batch_admissions'] += 1
        require(batch.identity==locator['batch_ref'], 'shared batch identity differs')
        save(out/'shared-admission-receipt.json', {'status':'COMPLETE',
            'public_api':'axiom_research.load_stock_ml_batch_inputs',
            'wall_seconds':time.monotonic()-began, 'pid':os.getpid(),
            'batch_ref':batch.identity, 'limits':cfg['batch_limits'], 'metrics':batch.metrics,
            'os_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'Data_calls':0, 'Core_calls':0, 'fit_calls':0, 'predict_calls':0, 'account_calls':0})
        destination = Path(cfg['recovery']['fold_destination'] or out/'folds')
        for index, request in enumerate(manifest['folds'], 1):
            emit(component='fold_build', current_fold=index, completed_folds=len(finished))
            tick = time.monotonic(); metrics = {}
            result = build_stock_ml_fold_from_saved_inputs(request['input_manifest'],
                fold_spec=request['fold_spec'], destination=destination, metrics=metrics, batch=batch)
            build_seconds = time.monotonic()-tick
            value = result.to_dict()
            expected_calls = 0 if metrics['cache_hit'] else 1
            require(metrics['train_calls']==metrics['predict_calls']==expected_calls,
                'fold HIT/fresh execution counts differ')
            require(all(metrics[k]==0 for k in ('data_read_calls','supplier_calls',
                'feature_core_calls','label_core_calls','core_calls','account_calls')),
                'fold executed a forbidden owner stage')
            require(value['status']=='COMPLETE' and value['definition']['fold_spec']==request['fold_spec'] and
                value['definition']['fold_spec_ref']==cfg['fold_spec_refs'][index-1] and
                value['definition']['parameters']==cfg['parameters'] and
                value['definition']['num_boost_round']==100, 'saved fold contract differs')
            emit(component='public_fold_readback', current_fold=index)
            before = observations['backend_calls']; tick = time.monotonic()
            loaded = load_stock_ml_fold(result.path, batch=batch)
            load_seconds = time.monotonic()-tick
            require(loaded.identity==result.identity and loaded.to_dict()==value and
                observations['backend_calls']==before, 'public fold readback differs or trained')
            predictions = loaded.predictions()
            spec = request['fold_spec']
            # Universe is owned by the immutable original Feature spec. Fold
            # specs deliberately do not copy it, so use that original spec.
            securities = read(monitor.OLD_SPEC)['universe']
            expected = {(s,d) for d in spec['inference_cutoff_by_session'] for s in securities}
            require(predictions['signal_run_ref']==value['signal_run_ref'] and
                predictions['fold_spec_ref']==cfg['fold_spec_refs'][index-1] and
                predictions['universe']==securities and
                {(r['security_id'],r['session']) for r in predictions['rows']}==expected and
                len(predictions['rows'])==len(expected), 'complete saved prediction grid differs')
            output_manifest = read(result.path/'manifest.json')
            prediction_path = result.path/'predictions.json'
            item = {'index':index, 'status':'COMPLETE', 'path':str(result.path),
                'fold_ref':loaded.identity, 'content_digest':value['content_digest'],
                'fold_spec_ref':cfg['fold_spec_refs'][index-1], 'fold_spec':spec,
                'model_ref':value['model_ref'], 'signal_run_ref':value['signal_run_ref'],
                'feature_ref':value['feature_ref'], 'label_ref':value['label_ref'],
                'dataset_ref':value['dataset_ref'], 'implementation_ref':value['definition']['implementation_ref'],
                'prediction':{'path':str(prediction_path),
                    'file_digest':output_manifest['files']['predictions.json'],
                    'signal_run_ref':value['signal_run_ref'], 'contract_version':predictions['contract_version']},
                'files':output_manifest['files'],
                'manifest_file_digest':monitor.file_ref(result.path/'manifest.json'),
                'prediction_rows':len(predictions['rows']),
                'valid_predictions':sum(r['valid'] is True for r in predictions['rows']),
                'build_seconds':build_seconds, 'public_load_seconds':load_seconds,
                'public_loader':'axiom_research.load_stock_ml_fold(path,batch=batch)',
                'public_loader_status':'PASS', 'metrics':metrics, 'reader_metrics':batch.metrics,
                'os_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'cache_hit':metrics['cache_hit'], 'public_load_Data_Core_fit_predict_account_calls':0}
            finished.append(item)
            save(out/f'fold-{index}-receipt.json',item)
            save(out/'fold-handoff.partial.json', {'status':'PARTIAL', 'folds':finished,
                'batch_ref':batch.identity, 'feature_inputs_ref':cfg['input_refs']['feature_inputs_ref']})
            print(json.dumps({'event':'FOLD_COMPLETE', 'index':index, 'fold_ref':loaded.identity,
                'signal_run_ref':value['signal_run_ref'], 'training_rows':read(result.path/'dataset.json')['training_row_count'],
                'prediction_rows':len(predictions['rows']), 'build_seconds':build_seconds,
                'public_load_seconds':load_seconds}), flush=True)
            emit(component='fold_complete', completed_folds=len(finished))
            del result,loaded,value,predictions,item
        fresh_count = sum(not f['cache_hit'] for f in finished)
        require(observations['shared_batch_admissions']==1 and observations['backend_calls']==fresh_count and
            len(finished)==4, 'four-fold HIT/fresh execution counts differ')
        final_metrics = batch.metrics
        if owns_batch:
            batch.close(); batch = None
        save(out/'handoff.json', {'status':'COMPLETE', 'source_refs':cfg['source_refs'],
            'source_paths':cfg['source_paths'], 'input_refs':cfg['input_refs'],
            'feature_inputs_ref':cfg['input_refs']['feature_inputs_ref'],
            'feature_content_digest':cfg['input_refs']['feature_content_digest'],
            'batch':locator, 'batch_ref':locator['batch_ref'], 'folds':finished,
            'observations':observations, 'final_reader_metrics':final_metrics,
            'batch_closed':owns_batch, 'Data_calls':0, 'Core_calls':0, 'supplier_calls':0,
            'feature_builds':0, 'fit_calls':fresh_count, 'predict_calls':fresh_count,
            'reused_folds':4-fresh_count, 'account_calls':0,
            'Engine_neutral_or_account_validation':'NOT_EXECUTED_BY_RESEARCH_HARNESS'})
        emit(status='STAGE_COMPLETE', component='four_fold_handoff_saved', completed_folds=4)
    except BaseException as exc:
        emit(status='FAILED',error_type=type(exc).__name__,error=str(exc))
        traceback.print_exc()
        raise
    finally:
        if native_backend is not None: backend.fit_predict_stock_model = native_backend
        if owns_batch and batch is not None: batch.close()
        save(out/'folds-exit.json', {'status':state['status'],
            'wall_seconds':time.monotonic()-started,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'completed_folds':len(finished), 'observations':observations})


def worker_router():
    out = Path(cfg['new_output']); began = time.monotonic()
    status = 'FAILED'
    try:
        modes = ['--folds-child'] if cfg['recovery']['prepared_locator'] else ['--v3-child']
        for mode in modes:
            code = subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),mode]).returncode
            if code: raise RuntimeError(mode+' exited '+str(code)+'; no automatic retry')
        require(read(out/'handoff.json')['status']=='COMPLETE', 'completed owner handoff required')
        require(monitor.lock_sources()==read(out/'preflight.json')['source_blobs'], 'source bytes changed during window')
        status = 'COMPLETE'
    finally:
        exits = [read(out/name) for name in ('prepare-exit.json','folds-exit.json') if (out/name).exists()]
        save(out/'worker-exit.json', {'status':status, 'stage':read(out/'worker-progress.json').get('stage')
            if (out/'worker-progress.json').exists() else 'prepare',
            'wall_seconds':time.monotonic()-began,
            'peak_rss_bytes':max([resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                *[v['peak_rss_bytes'] for v in exits]]), 'child_exits':exits,
            'peak_basis':'maximum individual child OS high water; monitor separately samples whole process tree'})


if __name__=='__main__':
    if '--describe' in sys.argv:
        print(json.dumps({'status':'STATIC_DESCRIPTION', 'authorization':cfg['authorization'],
            'source_refs':cfg['source_refs'], 'budgets':cfg['budgets'],
            'feature_inputs_ref':cfg['input_refs']['feature_inputs_ref'],
            'fold_spec_refs':cfg['fold_spec_refs']},ensure_ascii=False))
    else:
        released()
        if '--v3-child' in sys.argv: prepare_child(consume=True)
        elif '--prepare-child' in sys.argv: prepare_child()
        elif '--folds-child' in sys.argv: folds_child()
        elif '--worker' in sys.argv: worker_router()
        else: raise SystemExit(monitor.monitor())

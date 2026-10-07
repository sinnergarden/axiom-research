"""One pending bounded saved-Feature preparation and shared four-fold run.

The existing monitor owns process/time/RSS limits. Observers forward the exact
owner arguments/results and never re-read a panel or compute a target/model.
No stage may launch until the parent releases the comparison and source review.
"""
from copy import deepcopy
from contextlib import contextmanager
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


class _FeatureAdmissionPhases:
    """Separate initial Feature admission from required saved-closure checks."""
    def __init__(self, observations):
        self.observations = observations
        self.phase = 'initial'
        observations['feature_admission_calls_by_phase'] = {}

    @contextmanager
    def scope(self, phase):
        previous = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = previous

    def completed(self):
        self.observations['feature_admission_calls'] += 1
        counts = self.observations['feature_admission_calls_by_phase']
        counts[self.phase] = counts.get(self.phase, 0) + 1
        return counts[self.phase]

    @property
    def initial_calls(self):
        return self.observations['feature_admission_calls_by_phase'].get('initial', 0)

    def complete_counts_match(self, cache_hit):
        counts = self.observations['feature_admission_calls_by_phase']
        return (self.initial_calls == 1
            and set(counts) <= {'initial','staged','prepared_saved_batch'}
            and self.observations['feature_admission_calls'] == sum(counts.values())
            and counts.get('staged',0) == (0 if cache_hit else 1)
            and (counts.get('prepared_saved_batch',0) == 1 if cache_hit
                 else counts.get('prepared_saved_batch',0) in (0,1)))


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


def prepare_child():
    old = inputs()
    out, started, state, emit = phase('prepare')
    query_log, raw_log, core_log = [], [], []
    observations = {'feature_admission_calls':0, 'data_read_calls':0,
        'snapshot_load_calls':0, 'snapshot_load_seconds':0.0,
        'verified_calls':0, 'verification_hash_calls':0, 'verification_hash_bytes':0}
    restores = []
    data = None
    feature_phases = _FeatureAdmissionPhases(observations)
    def replace(obj, name, value):
        restores.append((obj, name, getattr(obj,name)))
        setattr(obj, name, value)
    try:
        from axiom_data import Data
        from axiom_research import prepare_stock_ml_batch_inputs
        from axiom_research.stock_matrix_prepare import _label_work_plan
        from axiom_research import stock_matrix_reader as mr, labels
        from axiom_research import stock_batch as sb
        import axiom_engine.core as core
        from axiom_research.stock_label_contracts import NORMALIZATION_SPEC
        data = Data(old['data_root'], cache_bytes=0)
        native_load = mr.load_feature_matrix_index
        def feature_load(*args, **kwargs):
            phase_name = feature_phases.phase
            emit(component='feature_admission' if phase_name == 'initial' else phase_name+'_admission',
                feature_admission_phase=phase_name)
            tick = time.monotonic()
            value = native_load(*args, **kwargs)
            phase_call = feature_phases.completed()
            index = value.to_dict()
            require(value.identity == cfg['input_refs']['feature_inputs_ref'] and
                index['content_digest'] == cfg['input_refs']['feature_content_digest'] and
                index['definition']['spec'] == read(monitor.OLD_SPEC), 'saved Feature identity/spec differs')
            receipt_name = ('feature-admission-receipt.json' if phase_name == 'initial' and phase_call == 1
                else f'feature-admission-{phase_name}-{phase_call}.json')
            save(out/receipt_name, {'wall_seconds':time.monotonic()-tick,
                'feature_inputs_ref':value.identity, 'content_digest':index['content_digest'],
                'metrics':value.metrics, 'phase':phase_name, 'phase_call':phase_call,
                'feature_admission_calls':observations['feature_admission_calls']})
            emit(component='raw_labels' if phase_name == 'initial' else phase_name+'_admission',
                feature_admission_calls=observations['feature_admission_calls'],
                feature_initial_admission_calls=feature_phases.initial_calls)
            return value
        replace(mr, 'load_feature_matrix_index', feature_load)
        native_batch_load = sb.load_stock_ml_batch_inputs
        def saved_batch_load(*args, **kwargs):
            with feature_phases.scope('prepared_saved_batch'):
                return native_batch_load(*args, **kwargs)
        replace(sb, 'load_stock_ml_batch_inputs', saved_batch_load)
        native_snapshot = data.store.load_snapshot
        def snapshot_load(*args, **kwargs):
            tick = time.monotonic()
            try:
                return native_snapshot(*args, **kwargs)
            finally:
                observations['snapshot_load_calls'] += 1
                observations['snapshot_load_seconds'] += time.monotonic()-tick
        replace(data.store, 'load_snapshot', snapshot_load)
        native_verified = data.store._verified
        def verified(uri, expected):
            prior = data.store._hash_cache.get(uri)
            result = native_verified(uri, expected)
            observations['verified_calls'] += 1
            if data.store._hash_cache.get(uri) != prior:
                observations['verification_hash_calls'] += 1
                observations['verification_hash_bytes'] += result.stat().st_size
            return result
        replace(data.store, '_verified', verified)
        native_read = data.read
        def data_read(*args, **kwargs):
            tick = time.monotonic()
            observations['data_read_attempts'] = observations.get('data_read_attempts',0)+1
            value = native_read(*args, **kwargs)
            observations['data_read_calls'] += 1
            requested_domain = kwargs['query'].domain
            query = deepcopy(value.context['query'])
            require(kwargs.get('snapshot') == cfg['snapshot'] and query['purpose'] == 'label_outcomes' and
                query['pit_policy'] == cfg['pit_policy'] and set(query['symbols']) == set(old['universe']) and
                len(query['symbols']) == 389 and int(value.frame.shape[0]) == 389*len(query['sessions']) and
                int(value.frame.shape[0]) <= physical_plan['maximum_query_rows'], 'Label query scope differs')
            require(requested_domain in ('market_daily','adjustment_factors'), 'unexpected Data domain')
            require(set(query['fields']) == ({'open','close'} if requested_domain=='market_daily' else {'factor'}),
                    'unexpected Label fields')
            allowed = {instant(s['fit_cutoff']) for s in cfg['fold_specs']} | {instant(old['evaluation_cutoff'])}
            clocks = {instant(v) for v in query['cutoff_by_session'].values()}
            require(len(clocks)==1 and clocks <= allowed, 'Label native cutoff differs')
            query_log.append({'call':len(query_log)+1, 'wall_seconds':time.monotonic()-tick,
                'row_count':int(value.frame.shape[0]), 'query_ref':digest(query), 'query':query,
                'requested_domain':requested_domain})
            save(out/'data-query-receipts.json',query_log)
            return value
        replace(data, 'read', data_read)
        native_labels = labels.build_forward_labels
        def raw_labels(*args, **kwargs):
            tick = time.monotonic()
            value = native_labels(*args, **kwargs)
            require(value['label_spec']['label_id']=='forward_5_session_open_close_v1' and
                value['label_spec']['price_basis']=='common_anchor_adjusted_v1', 'Raw Label semantics differ')
            raw_log.append({'call':len(raw_log)+1, 'wall_seconds':time.monotonic()-tick,
                'rows':len(value['rows']), 'label_ref':value['label_ref'], 'label_spec':value['label_spec'],
                'valid_rows':sum(r['valid'] is True for r in value['rows'])})
            save(out/'raw-label-receipts.json',raw_log)
            return value
        replace(labels, 'build_forward_labels', raw_labels)
        native_core = core.execute_cs_zscore_batch
        def normalize(*args, **kwargs):
            emit(component='normalization', normalization_calls=len(core_log))
            require(kwargs['params']==NORMALIZATION_SPEC['params'] and len(core_log)<4, 'normalization params/count differ')
            tick = time.monotonic()
            value = native_core(*args, **kwargs)
            core_log.append({'call':len(core_log)+1, 'fold_spec_ref':cfg['fold_spec_refs'][len(core_log)],
                'wall_seconds':time.monotonic()-tick, 'rows':len(value['values']),
                'result_ref':value['metadata']['result_ref'],
                'os_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            save(out/'normalization-receipts.json',core_log)
            return value
        replace(core, 'execute_cs_zscore_batch', normalize)
        native_staged = mr._validate_staged_matrix_batch
        def staged(*args, **kwargs):
            emit(component='staged_joint_admission')
            tick = time.monotonic()
            with feature_phases.scope('staged'):
                value = native_staged(*args, **kwargs)
            save(out/'staged-admission-receipt.json', {'wall_seconds':time.monotonic()-tick, 'metrics':value})
            return value
        replace(mr, '_validate_staged_matrix_batch', staged)
        progress_log = []
        def progress(value):
            progress_log.append({'wall_seconds':time.monotonic()-started, 'public_progress':deepcopy(value)})
            save(out/'prepare-progress-receipts.json',progress_log)
            emit(component=value['stage'], completed_dates=value.get('completed',0),
                 logical_query_count=observations['data_read_calls'], normalization_calls=len(core_log))
        metrics = {}
        physical_plan = _label_work_plan(read(monitor.OLD_SPEC), cfg['fold_specs'], cfg['prepare_options']['row_block_sessions'])
        save(out/'label-physical-plan.json', physical_plan)
        manifest = prepare_stock_ml_batch_inputs(data, feature_inputs=cfg['feature_path'],
            fold_specs=cfg['fold_specs'], destination=out/'prepared',
            preparation_options=cfg['prepare_options'], metrics=metrics, progress=progress)
        require(manifest['status']=='COMPLETE' and manifest['definition']['fold_specs']==cfg['fold_specs'],
                'prepared batch contract differs')
        require(feature_phases.complete_counts_match(metrics['cache_hit']),
                'initial/staged/saved-batch Feature admission counts differ')
        if metrics['cache_hit']:
            require(feature_phases.initial_calls==1 and observations['data_read_calls']==0 and
                not raw_log and not core_log, 'prepared HIT performed owner work')
        else:
            require(feature_phases.initial_calls==1 and
                observations['data_read_calls']==physical_plan['data_calls'] and
                len(raw_log)==physical_plan['raw_calls'] and len(core_log)==physical_plan['core_calls'] and
                sum(x['rows'] for x in raw_log)==physical_plan['raw_rows'], 'preparation physical plan differs')
        path = out/'prepared'/manifest['definition_ref'][7:]/'batch.json'
        locator = {'path':str(path), 'file_digest':monitor.file_ref(path), 'batch_ref':manifest['batch_ref'],
            'prepared_view_ref':manifest['prepared_view']['prepared_view_ref']}
        save(out/'batch-locator.json',locator)
        save(out/'prepare-receipt.json', {'status':'COMPLETE', 'wall_seconds':time.monotonic()-started,
            'batch':locator, 'metrics':metrics, 'observations':observations, 'raw_requests':len(raw_log),
            'raw_rows':sum(x['rows'] for x in raw_log), 'normalization_calls':len(core_log),
            'cache_hit':metrics['cache_hit'], 'physical_plan':physical_plan,
            'feature_execution_calls':0, 'qlib_export_calls':0, 'supplier_calls':0,
            'fit_calls':0, 'predict_calls':0, 'account_calls':0})
        emit(status='STAGE_COMPLETE',component='prepared_batch_saved')
    except BaseException as exc:
        emit(status='FAILED',error_type=type(exc).__name__,error=str(exc))
        traceback.print_exc()
        raise
    finally:
        for obj,name,original in reversed(restores):
            setattr(obj,name,original)
        save(out/'prepare-exit.json', {'status':state['status'], 'wall_seconds':time.monotonic()-started,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'observations':observations})


def folds_child():
    inputs()
    out, started, state, emit = phase('folds')
    observations = {'shared_batch_admissions':0, 'backend_calls':0}
    finished = []
    backend = None
    native_backend = None
    batch = None
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
            manifest['definition']['feature_inputs']['feature_inputs_ref']==cfg['input_refs']['feature_inputs_ref'],
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
        batch.close(); batch = None
        save(out/'handoff.json', {'status':'COMPLETE', 'source_refs':cfg['source_refs'],
            'source_paths':cfg['source_paths'], 'input_refs':cfg['input_refs'],
            'feature_inputs_ref':cfg['input_refs']['feature_inputs_ref'],
            'feature_content_digest':cfg['input_refs']['feature_content_digest'],
            'batch':locator, 'batch_ref':locator['batch_ref'], 'folds':finished,
            'observations':observations, 'final_reader_metrics':final_metrics,
            'batch_closed':True, 'Data_calls':0, 'Core_calls':0, 'supplier_calls':0,
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
        if batch is not None: batch.close()
        save(out/'folds-exit.json', {'status':state['status'],
            'wall_seconds':time.monotonic()-started,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'completed_folds':len(finished), 'observations':observations})


def worker_router():
    out = Path(cfg['new_output']); began = time.monotonic()
    status = 'FAILED'
    try:
        modes = ['--folds-child'] if cfg['recovery']['prepared_locator'] else ['--prepare-child','--folds-child']
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
        if '--prepare-child' in sys.argv: prepare_child()
        elif '--folds-child' in sys.argv: folds_child()
        elif '--worker' in sys.argv: worker_router()
        else: raise SystemExit(monitor.monitor())

"""Bounded Qlib/Core/LightGBM orchestration; account execution stays in Engine."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
import errno
import importlib.metadata
import math
import json
from pathlib import Path
import shutil
import tempfile
import time

from .stock_artifacts import (StockMLExperiment, digest, file_digest, write_json,
                             load_stock_ml_experiment, load_stock_model)

VERSION = 'axiom.stock_ml/1'
LGBM_PARAMETERS = {'objective':'regression','boosting_type':'gbdt','learning_rate':0.05,
    'num_leaves':31,'max_depth':5,'min_data_in_leaf':20,'seed':42,'num_threads':1,
    'feature_fraction':1.0,'bagging_fraction':1.0,'deterministic':True,
    'force_col_wise':True,'verbosity':-1}
TREES = 100
TARGET_SEMANTICS = 'forward_5_session_cs_zscore_prediction'


def _instant(value):
    d = datetime.fromisoformat(value.replace('Z','+00:00'))
    if d.tzinfo is None: raise ValueError('timezone required')
    return d.astimezone(timezone.utc)


def _environment():
    import platform
    return {'python':platform.python_version(), 'packages':{
        p:importlib.metadata.version(p) for p in ('pyqlib','lightgbm','numpy','pandas','pyarrow')}}


def _implementation():
    import importlib.util
    root = Path(__file__).parent
    sources={'research':{p.name:file_digest(p) for p in sorted(root.glob('*.py'))}}
    for name,package in (('core','axiom_engine.core'),('data','axiom_data')):
        base=Path(next(iter(importlib.util.find_spec(package).submodule_search_locations)))
        sources[name]={str(p.relative_to(base)):file_digest(p) for p in sorted(base.rglob('*.py'))}
    return sources


def _seal(value, key):
    return {**value, key:digest(value)}


def _project_qlib(batch, values, view_ref):
    """Explicit float32 numeric projection. Original per-key Reader metadata stays bound.

    A later revision differing from the frozen Qlib query fails rather than
    borrowing earlier values and attaching a later revision's provenance.
    """
    import numpy as np
    from axiom_data import DataBatch
    wire = batch.to_json(); frame = batch.frame.copy()
    fields = wire['context']['query']['fields']
    for i, row in enumerate(wire['records']):
        key = (row['security_id'], row['session'])
        for field in fields:
            expected = row[field]; value = values[key][field]
            if expected is None:
                if value is not None: raise ValueError('Qlib/Reader missingness mismatch')
            elif value is None or float(np.float32(expected))!=value:
                raise ValueError('Qlib/Reader revision or value mismatch: '+str((key,field)))
            frame.at[frame.index[i],field] = value
    context = {**wire['context'], 'numeric_projection':{
        'contract_version':'research_qlib_native_projection_v1','view_ref':view_ref,
        'reader_batch_ref':digest(wire),'revision_admission':'exact_reader_float32_value',
        'dtype':'float32_values_promoted_to_float64'}}
    return DataBatch(frame, batch.field_meta, context)


def _adjust_feature(prices, factors, session):
    from axiom_data import DataBatch, adjust_prices
    adjusted = adjust_prices(prices,factors,fields=('open','high','low','close'),
                             factor_field='factor',anchor_session=session,decision_session=session)
    frame = adjusted.frame.merge(prices.frame[['security_id','session','amount_cny']],
                                on=['security_id','session'],validate='one_to_one')
    context = {**adjusted.context, 'query':{**adjusted.context['query'],
        'fields':['open','high','low','close','amount_cny']}, 'research_projection':{
        'operation':'keyed_join_native_amount','adjusted_ref':digest(adjusted.to_json()),
        'native_ref':digest(prices.to_json()), 'native_fields':['amount_cny']}}
    return DataBatch(frame,{**adjusted.field_meta,'amount_cny':prices.field_meta['amount_cny']},context)


def _validate_config(config):
    required = {'snapshot','universe_id','symbols','calendar','read_sessions','feature_sessions',
                'prediction_sessions','fit_cutoff','pit_policy','cutoff_by_session',
                'evaluation_cutoff','feature_selection','scope_ref','calendar_ref'}
    if set(config) != required: raise ValueError('explicit stock ML input config required')
    if config['snapshot'] in ('','current','latest'): raise ValueError('fixed Snapshot required')
    for name in ('symbols','calendar','read_sessions','feature_sessions','prediction_sessions'):
        v=config[name]
        if not v or len(v)!=len(set(v)) or v!=sorted(v): raise ValueError('ordered unique '+name+' required')
    if not set(config['read_sessions'])<=set(config['calendar']): raise ValueError('calendar scope mismatch')
    calendar=config['calendar']; selected=config['read_sessions']
    if selected!=calendar[calendar.index(selected[0]):calendar.index(selected[-1])+1]:
        raise ValueError('read sessions must preserve every actual exchange session')
    if not set(config['feature_sessions'])<=set(config['read_sessions']): raise ValueError('feature scope mismatch')
    if not set(config['prediction_sessions'])<=set(config['feature_sessions']): raise ValueError('prediction scope mismatch')
    if set(config['cutoff_by_session'])!=set(config['read_sessions']): raise ValueError('feature cutoffs required')
    times=[_instant(config['cutoff_by_session'][s]) for s in selected]
    if times!=sorted(times) or any(t.date().isoformat()<s for s,t in zip(selected,times)):
        raise ValueError('chronological feature cutoffs cannot precede their sessions')
    fit=_instant(config['fit_cutoff'])
    if any(_instant(config['cutoff_by_session'][s])<=fit for s in config['prediction_sessions']):
        raise ValueError('model fit cutoff must precede all prediction cutoffs')
    if _instant(config['evaluation_cutoff'])<=fit: raise ValueError('outcome cutoff must follow fit cutoff')


def predict_stock_model(path, feature_rows, *, ordered_features, feature_selection=None):
    """Independently load a saved native booster and predict ordered finite rows."""
    import lightgbm as lgb
    import numpy as np
    model=load_stock_model(path)
    columns=model['ordered_features']
    if list(ordered_features)!=columns: raise ValueError('prediction feature order differs from saved model')
    if 'feature_selection' in model and feature_selection!=model['feature_selection']:
        raise ValueError('prediction feature semantic versions differ from saved model')
    rows=list(feature_rows)
    for row in rows:
        if len(row['values'])!=len(columns) or any(v is None or not math.isfinite(v) for v in row['values']):
            raise ValueError('independent prediction requires finite ordered feature values')
    if not rows: return []
    booster=lgb.Booster(model_file=str(Path(path)/'booster.txt'))
    if booster.feature_name()!=columns: raise ValueError('saved booster feature schema mismatch')
    return [float(x) for x in booster.predict(np.asarray([r['values'] for r in rows],dtype=np.float64),num_threads=1)]


def _signal_evidence(predictions, labels, cutoff):
    import pandas as pd
    targets={(r['security_id'],r['feature_session']):r for r in labels['rows']}
    groups={}
    for row in predictions['rows']:
        label=targets.get((row['security_id'],row['session']))
        eligible=(row['valid'] and label and label['valid'] and
                  _instant(label['label_available_at'])<=_instant(cutoff))
        groups.setdefault(row['session'],[]).append((row,label,eligible))
    out=[]
    for session, rows in sorted(groups.items()):
        valid=[(r['score'],l['return']) for r,l,ok in rows if ok]
        reason=None; ic=rank_ic=None
        if len(valid)<20: reason='INSUFFICIENT_VALID_PAIRS'
        else:
            x=pd.Series([v[0] for v in valid]); y=pd.Series([v[1] for v in valid])
            if x.nunique()<2 or y.nunique()<2: reason='CONSTANT_SECTION'
            else:
                ic=float(x.corr(y)); rank_ic=float(x.rank(method='average').corr(y.rank(method='average')))
        out.append({'session':session,'valid_pair_count':len(valid),
            'prediction_valid_count':sum(r['valid'] for r,_,_ in rows),
            'excluded_pair_count':len(rows)-len(valid),'ic':ic,'rank_ic':rank_ic,'reason':reason})
    return _seal({'contract_version':'stock_signal_evidence_v1','signal_ref':predictions['signal_run_ref'],
        'label_ref':labels['label_ref'],'evaluation_cutoff':cutoff,'minimum_pairs':20,
        'score_semantics':predictions['score_semantics'],'label_semantics':'raw_forward_5_session_open_close_return',
        'rank_ties':'average','series':out,
        'limitations':['Forward-label statistics are not account returns.','No tuning or confidence claim.']},'evidence_ref')


def _prepare_stock_features(data, *, config, destination, catalog, chosen, progress):
    """Build decision features once; no outcome reads or fold label cutoff."""
    from .feature_catalog import build_feature_plan
    from axiom_data import QuerySpec
    from axiom_engine.core import execute_feature_plan
    from .data_adapter import adapt_decision_batch
    from .qlib_adapter import QlibView
    import numpy as np
    stats={'feature_core_calls':0,'core_calls':0,'data_read_calls':0,'feature_cache_hit':False}
    symbols=tuple(config['symbols']); sessions=tuple(config['read_sessions']); cutoffs=config['cutoff_by_session']
    price_query=QuerySpec('market_daily',('open','high','low','close','amount_cny'),symbols,
        sessions,config['pit_policy'],cutoffs)
    factor_query=replace(price_query,domain='adjustment_factors',fields=('factor',))
    member_query=replace(price_query,domain='universe_membership',fields=('is_member',),universe_id=config['universe_id'])
    begin=time.perf_counter()
    qlib_path=Path(destination)/'qlib'/digest({'snapshot':config['snapshot'],
        'prices':{k:str(v) for k,v in vars(price_query).items()},'scope_ref':config['scope_ref']})[7:]
    data.export_qlib(snapshot=config['snapshot'],queries=(price_query,factor_query),destination=qlib_path,
                     universe_query=member_query,universe_name=config['universe_id'])
    view=QlibView(qlib_path).activate(); native=view.read(fields=('open','high','low','close','amount_cny','factor'),symbols=symbols)
    view_reference=view.reference; view_id=view_reference['view_id']
    reverse={v:k for k,v in view_reference['instrument_map'].items()}; values={}
    for (instrument,day),row in native.iterrows():
        values[reverse[instrument],str(day.date())]={k:None if np.isnan(row['$'+k]) else float(row['$'+k])
            for k in ('open','high','low','close','amount_cny','factor')}
    stats['qlib_seconds']=time.perf_counter()-begin
    rows=[]; inputs=[]; columns=[f['id'] for f in chosen]; position={s:i for i,s in enumerate(sessions)}
    begin=time.perf_counter()
    for completed,session in enumerate(config['feature_sessions'],1):
        i=position[session]; history=sessions[max(0,i-20):i+1]
        window_cutoffs={s:cutoffs[session] for s in history}
        pq=replace(price_query,sessions=history,cutoff_by_session=window_cutoffs)
        fq=replace(factor_query,sessions=history,cutoff_by_session=window_cutoffs)
        rq=replace(member_query,sessions=history,cutoff_by_session=window_cutoffs)
        stats['data_read_calls']+=3
        price=_project_qlib(data.read(snapshot=config['snapshot'],query=pq),values,view_id)
        factor=_project_qlib(data.read(snapshot=config['snapshot'],query=fq),values,view_id)
        adjusted=_adjust_feature(price,factor,session)
        membership=data.members(snapshot=config['snapshot'],query=rq)
        adapted=adapt_decision_batch(adjusted,reference=membership,recipe_ref=catalog.recipe_ref(
            config['feature_selection'],normalized=True),output_keys=tuple((s,session) for s in symbols),
            source_granularity='batch_field')
        plan=build_feature_plan(adapted.plan,config['feature_selection'],catalog=catalog,normalized=True)
        frame=execute_feature_plan(plan,adapted.facts,adapted.context); stats['core_calls']+=1; stats['feature_core_calls']+=1
        frame_ref=frame.identity; plan_ref=plan.identity
        frame_wire=frame.to_dict(); member={r['security_id']:r['is_member'] for r in membership.to_json()['records'] if r['session']==session}
        for r in frame_wire['rows']:
            rows.append({'security_id':r['security_id'],'session':session,'values':r['values'],
                'availability':r['availability'],'validity':r['valid'],'reasons':r['reasons'],
                'member':member[r['security_id']],
                'knowledge_cutoff':cutoffs[session],'source_refs':[frame_ref,plan_ref]})
        inputs.append({'session':session,'core_frame_ref':frame_ref,'core_plan':plan.to_dict(),
            'fact_ref':adapted.facts.identity,'context_ref':adapted.context.identity,
            'sessions':list(history),'cutoffs':window_cutoffs,
            'adjusted_input_ref':digest(adjusted.to_json()),'membership_ref':digest(membership.to_json()),
            'source_evidence':{k:{**{a:b for a,b in v.items() if a!='provenance_by_key'},
                'provenance_by_key_ref':digest(v['provenance_by_key'])}
                for k,v in adapted.source_evidence.items()}})
        if progress is not None:
            progress({'stage':'features','completed':completed,'total':len(config['feature_sessions']),
                      'session':session,'seconds':time.perf_counter()-begin})
    stats['feature_seconds']=time.perf_counter()-begin
    features=_seal({'contract_version':'stock_feature_build_v1','catalog_ref':catalog.identity,
        'selection':config['feature_selection'],'ordered_features':columns,'qlib_view':view_reference,
        'input_evidence_ref':digest(inputs),'rows':rows},'feature_ref')
    return features, inputs, stats


def _prepare_stock_labels(data, *, config, training_sessions, metrics=None):
    """Read original outcome inputs separately at this fold's two cutoffs."""
    from axiom_data import QuerySpec,adjust_prices
    from .labels import build_forward_labels
    if not training_sessions: raise ValueError('training feature sessions required')
    symbols=tuple(config['symbols']); sessions=tuple(config['read_sessions'])
    price_query=QuerySpec('market_daily',('open','high','low','close','amount_cny'),symbols,
        sessions,config['pit_policy'],config['cutoff_by_session'])
    def outcome(cutoff, wanted):
        allowed=tuple(s for s in config['calendar'] if sessions[0]<=s<=_instant(cutoff).date().isoformat())
        if not allowed: raise ValueError('no actual label calendar at cutoff')
        q=replace(price_query,fields=('open','close'),sessions=allowed,
                  cutoff_by_session={s:cutoff for s in allowed},purpose='label_outcomes')
        f=replace(q,domain='adjustment_factors',fields=('factor',))
        if metrics is not None: metrics['data_read_calls']=metrics.get('data_read_calls',0)+2
        p=data.read(snapshot=config['snapshot'],query=q); factors=data.read(snapshot=config['snapshot'],query=f)
        adjusted=adjust_prices(p,factors,fields=('open','close'),anchor_session=allowed[-1],
                               decision_session=allowed[-1],factor_field='factor')
        return build_forward_labels(adjusted,calendar=config['calendar'],feature_sessions=wanted)
    train_labels=outcome(config['fit_cutoff'],training_sessions)
    evaluation_labels=outcome(config['evaluation_cutoff'],config['prediction_sessions'])
    return train_labels, evaluation_labels


def _prepare_stock_inputs(data, *, config, destination, catalog, chosen, progress):
    features, inputs, stats=_prepare_stock_features(data,config=config,destination=destination,
        catalog=catalog,chosen=chosen,progress=progress)
    cutoffs=config['cutoff_by_session']
    training_sessions=[s for s in config['feature_sessions'] if _instant(cutoffs[s])<=_instant(config['fit_cutoff'])]
    train_labels, evaluation_labels=_prepare_stock_labels(data,config=config,
        training_sessions=training_sessions,metrics=stats)
    return features, train_labels, evaluation_labels, inputs, stats


def build_stock_ml_experiment(data, *, config, destination, metrics=None, progress=None,
                              reuse_input_path=None):
    """One fixed fold, durable whole-experiment reuse and a saved native model.

    Reuse checks the manifest before touching Data/Qlib/Core/LightGBM execution.
    Runtime-neutral forecasts are saved for the Engine owner; this function does
    not select positions, price fills or calculate account NAV.
    """
    _validate_config(config)
    from .feature_catalog import load_feature_catalog, build_feature_plan
    catalog=load_feature_catalog(); chosen=catalog.select(config['feature_selection'])
    from .stock_label_normalization import NORMALIZATION_SPEC
    source=None;input_reuse=None
    if reuse_input_path is not None:
        source=load_stock_ml_experiment(reuse_input_path)
        previous=source.to_dict()
        if previous['definition']['config']!=config or previous['definition']['catalog_ref']!=catalog.identity:
            raise ValueError('saved feature/raw-label input config differs')
        input_reuse={'experiment_ref':source.identity,'feature_ref':previous['feature_ref'],
                     'raw_label_bundle_ref':previous['label_ref']}
    implementations=_implementation()
    definition={'version':VERSION,'config':config,'catalog_ref':catalog.identity,
                'implementation_ref':digest(implementations),'implementation_sources':implementations,
                'environment':_environment(),
                'parameters':LGBM_PARAMETERS,'num_boost_round':TREES,
                'label_normalization':NORMALIZATION_SPEC,'target_semantics':TARGET_SEMANTICS,
                'input_reuse':input_reuse}
    target=Path(destination)/digest(definition)[7:]
    if target.exists():
        result=load_stock_ml_experiment(target)
        if result.to_dict()['definition']!=definition: raise ValueError('cached stock definition mismatch')
        if metrics is not None: metrics.update(cache_hit=True, train_calls=0, core_calls=0, feature_core_calls=0, label_core_calls=0, data_read_calls=0, predict_calls=0)
        return StockMLExperiment(target,True)
    import lightgbm as lgb
    import numpy as np
    from .stock_label_normalization import normalize_forward_labels
    stats={'cache_hit':False,'train_calls':0,'core_calls':0,'predict_calls':0}
    if source is None:
        features,train_labels,evaluation_labels,inputs,prepared=_prepare_stock_inputs(
            data,config=config,destination=destination,catalog=catalog,chosen=chosen,progress=progress)
        stats.update(prepared)
    else:
        features=json.loads((source.path/'features.json').read_text())
        if (features['catalog_ref']!=catalog.identity or features['selection']!=config['feature_selection'] or
                features['ordered_features']!=[f['id'] for f in chosen]):
            raise ValueError('saved feature catalog/schema differs from input config')
        raw=json.loads((source.path/'labels.json').read_text())
        train_labels=raw['training'];evaluation_labels=raw['evaluation'];inputs=None
        stats.update(feature_cache_hit=True,feature_core_calls=0,core_calls=0,data_read_calls=0,
                     qlib_seconds=0,feature_seconds=0)
    rows=features['rows'];columns=features['ordered_features'];symbols=tuple(config['symbols'])
    cutoffs=config['cutoff_by_session'];view_reference=features['qlib_view']
    training_sessions=[s for s in config['feature_sessions'] if _instant(cutoffs[s])<=_instant(config['fit_cutoff'])]
    begin=time.perf_counter()
    normalized_training=normalize_forward_labels(train_labels,features=features,cutoff=config['fit_cutoff'])
    normalized_evaluation=normalize_forward_labels(evaluation_labels,features=features,cutoff=config['evaluation_cutoff'])
    stats['label_core_calls']=len(normalized_training['core_frames'])+len(normalized_evaluation['core_frames'])
    stats['core_calls']+=stats['label_core_calls']
    labels=_seal({'contract_version':'stock_label_bundle_v1','training':train_labels,'evaluation':evaluation_labels,
        'normalized_training':normalized_training,'normalized_evaluation':normalized_evaluation},'label_ref')
    normalized={(r['security_id'],r['feature_session']):r for r in normalized_training['rows']}
    training=[]; excluded={}
    for r in rows:
        if r['session'] not in training_sessions: continue
        key=(r['security_id'],r['session'])
        target_row=normalized[key]
        reason=target_row['invalid_reason'] if not target_row['valid'] else None
        if reason: excluded[reason]=excluded.get(reason,0)+1
        else: training.append({**r,'label':target_row['normalized_target'],
            'raw_return':target_row['raw_return'],'label_available_at':target_row['label_available_at'],
            'normalized_available_at':target_row['normalized_available_at']})
    if len(training)<40: raise ValueError('insufficient mature finite training rows: '+str(len(training)))
    dataset=_seal({'contract_version':'stock_training_dataset_v1','feature_ref':features['feature_ref'],
        'label_ref':normalized_training['label_ref'],'raw_label_ref':train_labels['label_ref'],
        'target_semantics':TARGET_SEMANTICS,'fit_cutoff':config['fit_cutoff'],
        'ordered_features':columns,'training_keys':[[r['security_id'],r['session']] for r in training],
        'training_rows_ref':digest(training),'training_row_count':len(training),'excluded':excluded,
        'validation':'none_fixed_parameters_no_early_stopping',
        'normalization':NORMALIZATION_SPEC,'label_section_refs':[s['section_ref'] for s in normalized_training['sections']]},'dataset_ref')
    stats['label_dataset_seconds']=time.perf_counter()-begin
    X=np.asarray([r['values'] for r in training],dtype=np.float64); y=np.asarray([r['label'] for r in training],dtype=np.float64)
    begin=time.perf_counter(); booster=lgb.train(LGBM_PARAMETERS,lgb.Dataset(X,label=y,feature_name=columns),num_boost_round=TREES)
    stats['train_seconds']=time.perf_counter()-begin; stats['train_calls']=1
    prediction_features=[r for r in rows if r['session'] in config['prediction_sessions'] and r['member'] and
                         all(v is not None for v in r['values']) and all(r['validity'])]
    begin=time.perf_counter()
    scores=booster.predict(np.asarray([r['values'] for r in prediction_features],dtype=np.float64),num_threads=1) if prediction_features else []
    stats['predict_seconds']=time.perf_counter()-begin; stats['predict_calls']=1
    score_map={(r['security_id'],r['session']):float(s) for r,s in zip(prediction_features,scores)}
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stock-ml-',dir=target.parent) as tmp:
        stage=Path(tmp)/'complete';stage.mkdir();booster.save_model(str(stage/'booster.txt'))
        model=_seal({'contract_version':'stock_model_release_v1','dataset_ref':dataset['dataset_ref'],
            'feature_ref':features['feature_ref'],'ordered_features':columns,'fit_cutoff':config['fit_cutoff'],
            'label_ref':normalized_training['label_ref'],'raw_label_ref':train_labels['label_ref'],
            'target_semantics':TARGET_SEMANTICS,'feature_selection':config['feature_selection'],'catalog_ref':catalog.identity,
            'parameters':LGBM_PARAMETERS,'num_boost_round':TREES,'environment':definition['environment'],
            'implementation_ref':definition['implementation_ref'],'booster_digest':file_digest(stage/'booster.txt'),
            'feature_normalization':'same_date_visible_members_cs_zscore_no_fit',
            'label_normalization':NORMALIZATION_SPEC},'model_ref')
        prediction_rows=[]
        for r in rows:
            if r['session'] not in config['prediction_sessions']: continue
            score=score_map.get((r['security_id'],r['session'])); valid=score is not None and math.isfinite(score)
            reason=None if valid else 'NOT_MEMBER' if not r['member'] else 'FEATURE_MISSING'
            avail=max([x for x in r['availability'] if x]+[config['fit_cutoff']],key=_instant)
            prediction_rows.append({'security_id':r['security_id'],'session':r['session'],
                'knowledge_cutoff':r['knowledge_cutoff'],'available_at':avail,'score':score if valid else None,
                'valid':valid,'invalid_reason':reason,'member':r['member'],'source_refs':[
                    features['feature_ref'],model['model_ref'],catalog.identity,view_reference['view_id']]})
        predictions=_seal({'contract_version':'stock_prediction_run_v1','signal_stage':'prediction_raw',
            'score_semantics':TARGET_SEMANTICS,'score_unit':'dimensionless','model_ref':model['model_ref'],
            'feature_ref':features['feature_ref'],'universe':list(symbols),'rows':prediction_rows,
            'limitations':['Model/signal OOS evidence only; stock account admission is separate.',
                'Historical membership is a visible dated supplier snapshot carried forward.',
                'Best-effort historical availability is not strict historical receipt evidence.',
                'Input digests retain fixed Snapshot/query recovery; saved-only load does not requery original facts.',
                'Scores predict normalized cross-sectional targets and are not return percentages.']},'signal_run_ref')
        evidence=_signal_evidence(predictions,evaluation_labels,config['evaluation_cutoff'])
        experiment=_seal({'contract_version':'stock_ml_experiment_v1','definition':definition,
            'feature_ref':features['feature_ref'],'label_ref':labels['label_ref'],'dataset_ref':dataset['dataset_ref'],
            'model_ref':model['model_ref'],'signal_run_ref':predictions['signal_run_ref'],'evidence_ref':evidence['evidence_ref'],
            'account_status':'BLOCKED_PENDING_STOCK_RUNTIME_ADMISSION',
            'account_reason':'Neutral ML/dynamic Top5 and stock execution/event profile require Engine owner admission.'},'experiment_ref')
        for name,value in (('experiment.json',experiment),('features.json',features),('feature-inputs.json',inputs),
                ('labels.json',labels),('dataset.json',dataset),('model.json',model),('predictions.json',predictions),('signal-evidence.json',evidence)):
            if name=='feature-inputs.json' and source is not None:
                shutil.copyfile(source.path/name,stage/name)
            else: write_json(stage/name,value)
        names=[p.name for p in stage.iterdir()]
        write_json(stage/'manifest.json',{'contract_version':'stock_ml_manifest_v1',
            'experiment_ref':experiment['experiment_ref'],'files':{n:file_digest(stage/n) for n in names}})
        try: stage.rename(target)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST,errno.ENOTEMPTY): raise
            if load_stock_ml_experiment(target).identity!=experiment['experiment_ref']: raise ValueError('concurrent artifact conflict')
    stats.update(training_rows=len(training),feature_rows=len(rows),prediction_rows=len(prediction_rows),
                 valid_predictions=sum(r['valid'] for r in prediction_rows),artifact_bytes=sum(p.stat().st_size for p in target.iterdir()))
    if metrics is not None: metrics.update(stats)
    return load_stock_ml_experiment(target)


def build_stock_ml_from_saved_features(path, *, destination, metrics=None):
    """Derive normalized labels/model using frozen inputs, with no Data calls."""
    source=load_stock_ml_experiment(path)
    return build_stock_ml_experiment(None,config=source.to_dict()['definition']['config'],
        destination=destination,metrics=metrics,reuse_input_path=source.path)

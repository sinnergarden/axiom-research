"""One Research Raw operator and existing Core CS; compact v3 publication."""
from array import array
from copy import deepcopy
from dataclasses import replace, fields as dataclass_fields
from collections.abc import Mapping
from datetime import datetime, timezone, timedelta
from pathlib import Path
import errno
import json
import tempfile
import time

from .stock_artifacts import digest, digest_array_rows, file_digest, write_json
from .stock_fold_inputs import require, seal, validate_spec
from .stock_label_contracts import (_eligible_reason, _instant, NORMALIZATION_SPEC,
    RAW_TARGET_SCHEMA, NORMALIZED_TARGET_SCHEMA, _normalization_sources, core_clock)
from .stock_matrix_storage import write_buffer, write_part, instant_us
from .stock_compact_store import (_view_data, StockFeatureView, load_stock_feature_view,
    iter_feature_rows, iter_feature_eligibility, set_feature_window, OwnedStore, limits, sealed, _size)


def _implementation():
    import inspect
    sources={name:file_digest(Path(__file__).with_name(name)) for name in (
        'labels.py','stock_label_contracts.py','stock_compact_labels.py',
        'stock_compact_store.py','stock_compact_batch.py','stock_batch.py',
        'stock_fold_inputs.py','stock_matrix_storage.py','stock_compact_controls.py')}
    sources['canonical_training_binding']=inspect.getsource(digest_array_rows)
    return digest(sources)


def _input_environment():
    """Only libraries used to prepare inputs; model versions are downstream."""
    import platform
    import importlib.metadata
    return {'python':platform.python_version(), 'packages':{
        p:importlib.metadata.version(p) for p in ('numpy','pandas','pyarrow')}}


def _raw_implementation():
    """Raw identity excludes Feature/cohort/normalization/storage code."""
    import inspect
    from . import labels
    return digest({'operator_version':'forward_5_session_open_close_v1',
        'price_basis':labels.PRICE_BASIS,'functions':{name:inspect.getsource(getattr(labels,name))
        for name in ('_session','_instant','_sessions','_query_context','_keyed','_endpoint','_forward_rows')}})


def _working(metrics, amount):
    """Measure each completed chunk once; never rescan an accumulated panel."""
    _sync_feature_charge(metrics)
    total=metrics['_feature_bytes']+metrics['_store'].resident_bytes+metrics['_retained_raw_bytes']+metrics.get('_price_view_bytes',0)+amount
    require(total<=metrics['_limits']['maximum_matrix_bytes'],'compact producer working byte budget exceeded')
    metrics['maximum_working_bytes']=max(metrics['maximum_working_bytes'],total)


def _measured(metrics,value):
    _sync_feature_charge(metrics)
    try:
        return _size(value,maximum=metrics['_limits']['maximum_matrix_bytes'],
            retained=metrics['_feature_bytes']+metrics['_store'].resident_bytes+metrics['_retained_raw_bytes']+metrics.get('_price_view_bytes',0))
    finally:
        value=None


def _sync_feature_charge(metrics):
    """A sequential Feature window changes the live shared byte charge."""
    feature_store=metrics['_feature_store']
    metrics['_feature_bytes']=feature_store.resident_bytes+metrics['_caller_bytes']
    metrics['_store'].shared_bytes=metrics['_feature_bytes']
    metrics['_store'].shared_source_bytes=feature_store.metrics['source_bytes']+metrics['_caller_source_bytes']


def _producer_feature_window(feature, offsets, metrics):
    feature_store=metrics['_feature_store']; store=metrics['_store']
    previous=(feature_store.limits,feature_store.shared_bytes,feature_store.shared_source_bytes)
    try:
        feature_store.limits={k:min(v,metrics['_limits'][k]) for k,v in previous[0].items()}
        feature_store.shared_bytes=max(previous[1],metrics['_caller_bytes']+store.resident_bytes+
            metrics['_retained_raw_bytes']+metrics.get('_price_view_bytes',0))
        feature_store.shared_source_bytes=max(previous[2],metrics['_caller_source_bytes']+store.metrics['source_bytes'])
        selection=metrics.get('_model_feature_selection')
        set_feature_window(feature,offsets,**({} if selection is None else {'model_feature_selection':selection}))
    finally:
        feature_store.limits,feature_store.shared_bytes,feature_store.shared_source_bytes=previous
    _sync_feature_charge(metrics)


def _clock(value):
    return (datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(microseconds=int(value))).isoformat().replace('+00:00','Z')


def _write(root, definition, rows, *, normalized=False, core_ref=None, cohort=None, final_root=None):
    """One typed target, no Raw rows/proof JSON copies at the next stages."""
    reasons=sorted({r['invalid_reason'] for r in rows if r['invalid_reason'] is not None})
    dictionary=[None,*reasons]; codes={r:i for i,r in enumerate(dictionary)}
    calendar=definition['calendar']; positions={d:i for i,d in enumerate(calendar)}
    columns={'values':('float64_le',[r['return'] if r['valid'] else 0.0 for r in rows]),
        'validity':('bool_u8',[r['valid'] for r in rows]),
        'availability':('int64_le',[instant_us(r['label_available_at']) if r['label_available_at'] else 0 for r in rows]),
        'availability_validity':('bool_u8',[r['label_available_at'] is not None for r in rows]),
        'reason_codes':('int32_le',[codes[r['invalid_reason']] for r in rows])}
    sources=[]
    if not normalized:
        columns['start_session']=('int32_le',[positions.get(r['start_session'],-1) for r in rows])
        columns['end_session']=('int32_le',[positions.get(r['end_session'],-1) for r in rows])
        sources=sorted({tuple(r['source_refs']) for r in rows}); source_codes={r:i for i,r in enumerate(sources)}
        columns['source_codes']=('int32_le',[source_codes[tuple(r['source_refs'])] for r in rows])
    buffers={name:write_buffer(root,values,dtype=dtype,shape=[len(rows)]) for name,(dtype,values) in columns.items()}
    if final_root is not None:
        for descriptor in buffers.values():
            descriptor['path']=str(Path(final_root)/Path(descriptor['path']).relative_to(Path(root).resolve()))
    value=seal({'contract_version':'stock_compact_normalized_v1' if normalized else 'stock_compact_raw_v1',
        'definition':definition,'definition_ref':digest(definition),'row_count':len(rows),
        'reason_dictionary':dictionary,'source_dictionary':[list(r) for r in sources],
        'buffers':buffers,'core_ref':core_ref,'cohort':cohort},'target_ref')
    path=Path(root)/'target.json'; write_json(path,value)
    return {'path':str(path.resolve()),'file_digest':file_digest(path),'target_ref':value['target_ref']}


def _publish(target,definition,rows,*,budgets,normalized=False,core_ref=None,cohort=None,store=None):
    """Only a fully checked target directory is made visible at its key."""
    from .stock_compact_batch import read_target
    target=Path(target).resolve(); target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.compact-target-',dir=target.parent) as temporary:
        stage=Path(temporary)/'complete'; stage.mkdir()
        desc=_write(stage,definition,rows,normalized=normalized,core_ref=core_ref,cohort=cohort,final_root=target)
        final={'path':str(target/'target.json'),'file_digest':file_digest(stage/'target.json'),'target_ref':desc['target_ref']}
        checker=store or OwnedStore(budgets); original=checker.resolve
        checker.resolve=lambda p:stage/Path(p).relative_to(target) if Path(p).is_relative_to(target) else original(p)
        try:
            read_target(checker,final,expected=definition)
            try: stage.rename(target)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST,errno.ENOTEMPTY): raise
                # An independently published inode requires independent byte
                # admission. Do not pin our unpublished stage to that winner.
                raise ValueError('concurrent compact target publication; fresh admission required') from exc
            checker.resolve=original; checker.check()
        finally:
            checker.resolve=original
            if store is None: checker.close()
        return final


def _source_view(records,meta,context):
    # Bind the actually selected input once. Never reconstruct the old Raw
    # native/canonical proof graph. The fixed public query is the audit recipe.
    keys=('snapshot_id','domain','contract_id','source_profile_id','reader_version','query','derivation','limitations')
    selected={k:deepcopy(context[k]) for k in keys if k in context}
    value={'contract_version':'stock_label_price_view_v1','context':selected,
        'records_ref':digest(records),'field_meta_ref':digest(meta)}
    value['price_view_ref']=digest(value)
    return value


def _query(spec,cutoff,days):
    from axiom_data import QuerySpec
    calendar=spec['calendar']; allowed=[d for d in calendar if d<=_instant(cutoff).date().isoformat()]
    require(bool(allowed),'no actual label calendar at cutoff'); anchor=allowed[-1]
    positions={d:i for i,d in enumerate(calendar)}; endpoints={anchor}
    for day in days:
        pos=positions[day]
        for offset in (1,5):
            if pos+offset<len(calendar) and calendar[pos+offset]<=anchor: endpoints.add(calendar[pos+offset])
    query=QuerySpec(domain='market_daily',fields=('open','close'),symbols=tuple(spec['universe']),
        sessions=tuple(sorted(endpoints)),pit_policy=spec['pit_policy'],
        cutoff_by_session={d:cutoff for d in sorted(endpoints)},purpose='label_outcomes',price_basis='unadjusted')
    return query,anchor


def _price_view(data,spec,cutoff,days,metrics):
    """Select one full public cutoff domain, never relabel sliced Data wires."""
    from .labels import _query_context
    from .data_adapter import _versioned
    from axiom_data import adjust_prices
    query,anchor=_query(spec,cutoff,days)
    query_wire={f.name:(dict(value) if isinstance(value,Mapping) else list(value) if isinstance(value,tuple) else value)
        for f in dataclass_fields(query) for value in (getattr(query,f.name),)}
    # Data owns its internal read allocations. Bound requested cells before
    # calling it, then charge the actual adjusted wire once at this boundary.
    _working(metrics,len(query.sessions)*len(spec['universe'])*4096+len(days)*len(spec['universe'])*1024)
    price=data.read(snapshot=spec['snapshot'],query=query)
    factors=data.read(snapshot=spec['snapshot'],query=replace(query,domain='adjustment_factors',fields=('factor',)))
    metrics['data_read_calls']+=2
    adjusted=adjust_prices(price,factors,fields=('open','close'),anchor_session=anchor,
                          decision_session=anchor,factor_field='factor')
    # The public adjuster returns a detached batch. Native inputs need not
    # overlap its JSON conversion, canonical hashes or graph accounting.
    del price,factors
    records,meta,context=_versioned(adjusted,'label_outcomes')
    del adjusted
    _working(metrics,2*_measured(metrics,[records,meta,context])+len(days)*len(spec['universe'])*1024)
    _query_context(context,spec['calendar']); source=_source_view(records,meta,context)
    metrics['_price_view_bytes']=_measured(metrics,[records,meta,context,source])
    metrics['price_domains'].append({'cutoff':cutoff,'query_ref':digest(query_wire),
        'query':query_wire,'price_view_ref':source['price_view_ref'],'data_read_calls':2,
        'resident_bytes':metrics['_price_view_bytes']})
    return records,meta,context,source


def _raw(spec,cutoff,days,cache,metrics,price_view):
    from .labels import _forward_rows
    records,meta,context,source=price_view
    definition={'price_view':source,'snapshot':spec['snapshot'],'cutoff':cutoff,
        'calendar':spec['calendar'],'universe':spec['universe'],'sessions':days,
        'formula':'close(f+5) / open(f+1) - 1','price_basis':'common_anchor_adjusted_v1',
        'horizon_sessions':5,'start_session_offset':1,'end_session_offset':5,
        'missing_policy':'invalid_null_preserve_grid','implementation_ref':_raw_implementation()}
    key=digest(definition); target=(Path(cache)/'raw'/key[7:]).resolve(); artifact=target/'target.json'
    if artifact.exists():
        from .stock_compact_batch import read_target
        store=metrics['_store']; descriptor={'path':str(artifact.absolute())}
        value,rows=read_target(store,descriptor,expected=definition)
        _working(metrics,len(rows)*2048); rows=list(rows)
        metrics['raw_cache_hits']+=1
        return {**descriptor,'file_digest':store.hashes[descriptor['path']],'target_ref':value['target_ref']},rows
    rows=list(_forward_rows(records,meta,context,calendar=spec['calendar'],features=days,
        horizon_sessions=5,source_ref=source['price_view_ref']))
    _working(metrics,_measured(metrics,rows))
    metrics['raw_operator_calls']+=1
    descriptor=_publish(target,definition,rows,budgets=metrics['_limits'],store=metrics['_store'])
    return descriptor,rows


def _normalized(raw_parts,rows,feature,cutoff,days,cache,metrics):
    from axiom_engine.core import execute_cs_zscore_batch
    from axiom_engine._implementation import IMPLEMENTATION_REF
    fd=_view_data(feature); spec=fd['definition']['spec']; width=len(spec['universe'])
    feature_ref=fd['definition']['feature_view_ref']
    positions={d:i for i,d in enumerate(spec['feature_sessions'])}
    offsets=[positions[d]*width+i for d in days for i in range(width)]
    _working(metrics,len(rows)*4096+len(spec['ordered_features'])*96+3072)
    reasons=[]; instant=_instant(cutoff); cutoff_us=instant_us(cutoff)
    selection=metrics.get('_model_feature_selection')
    from .stock_compact_store import model_feature_binding,model_eligibility_ref
    binding=model_feature_binding(feature,selection)
    feature_count=len(spec['ordered_features']) if binding is None else len(binding['ordered_features'])
    for row,facts in zip(rows,iter_feature_eligibility(feature,offsets,model_feature_selection=selection)):
        reason=_eligible_reason(row,facts,feature_count,instant)
        if reason is None:
            require(facts.maximum_available_at_utc_us is not None and
                    facts.maximum_available_at_utc_us<=cutoff_us,
                    'training Feature native clock exceeds fit')
        reasons.append(reason)
    cohort={'contract_version':'stock_compact_cohort_v1','feature_view_ref':feature_ref,'cutoff':cutoff,
        'sessions':days,'universe':spec['universe'],'raw_refs':[d['target_ref'] for d in raw_parts],
        'eligibility_reasons':reasons,'eligible_keys':[[r['security_id'],r['feature_session']]
            for r,reason in zip(rows,reasons) if reason is None]}
    if binding is not None: cohort['model_feature_eligibility_ref']=model_eligibility_ref(binding)
    cohort_ref=digest(cohort)
    definition={'calendar':spec['calendar'],'universe':spec['universe'],'sessions':days,'cutoff':cutoff,
        'raw_refs':cohort['raw_refs'],'cohort_ref':cohort_ref,'normalization_spec':NORMALIZATION_SPEC,
        'implementation_ref':_implementation(),'core_implementation_ref':IMPLEMENTATION_REF}
    key=digest(definition); target=(Path(cache)/'normalized'/key[7:]).resolve(); artifact=target/'target.json'
    if artifact.exists():
        from .stock_compact_batch import read_target
        store=metrics['_store']; desc={'path':str(artifact.absolute())}
        value,normalized=read_target(store,desc,expected=definition,raw_rows=rows)
        metrics['normalized_cache_hits']+=1
        fd['store'].check()
        return {**desc,'file_digest':store.hashes[desc['path']],'target_ref':value['target_ref']},list(normalized),value['core_ref'],cohort
    arrays={name:array(code) for name,code in {'values':'d','value_validity':'B','value_reason_codes':'i',
        'fact_available_at_utc_us':'q','reference_member':'B','reference_available_at_utc_us':'q',
        'selection_cutoff_utc_us':'q','fact_source_codes':'i','reference_source_codes':'i'}.items()}
    dictionary=[None,*sorted({r for r in reasons if r is not None})]; codes={r:i for i,r in enumerate(dictionary)}
    sources={}; calendar_ref=digest({'contract_version':'stock_label_calendar_v1','sessions':spec['calendar']})
    raw_ref=digest(cohort['raw_refs'])
    _working(metrics,_measured(metrics,cohort)+len(rows)*4096)
    for i,day in enumerate(days):
        eligible=[[security,day] for j,security in enumerate(spec['universe']) if reasons[i*width+j] is None]
        bindings=_normalization_sources(raw_ref,feature_ref,day,cutoff,eligible)
        sources[day]={'bindings':sorted(bindings,key=lambda r:r['id']),
                      'source_sets':[['offline_eligibility'],['raw_labels']]}
        arrays['selection_cutoff_utc_us'].append(instant_us(core_clock(cutoff)))
        for j in range(width):
            n=i*width+j; row=rows[n]; valid=reasons[n] is None
            arrays['values'].append(row['return'] if valid else 0.0); arrays['value_validity'].append(valid)
            arrays['value_reason_codes'].append(codes[reasons[n]])
            arrays['fact_available_at_utc_us'].append(instant_us(core_clock(row['label_available_at'] if valid else cutoff)))
            arrays['reference_member'].append(valid); arrays['reference_available_at_utc_us'].append(instant_us(core_clock(cutoff)))
            arrays['fact_source_codes'].append(1 if valid else 0); arrays['reference_source_codes'].append(0)
    # Core accepts immutable readonly buffer views; its arithmetic remains the
    # existing operator, not a Research vectorized substitute.
    carrier={'contract_version':'core_cs_zscore_batch_input_v1','calendar_ref':calendar_ref,
        'schema':RAW_TARGET_SCHEMA,'output_schema':NORMALIZED_TARGET_SCHEMA,'sessions':days,
        'security_ids':spec['universe'],'reason_dictionary':dictionary,'source_bindings_by_session':sources,
        **{k:memoryview(v.tobytes()).cast('?' if k in ('value_validity','reference_member') else v.typecode)
           for k,v in arrays.items()}}
    # Carrier bytes and per-session bindings are detached from these working
    # panels. Release them before Core takes its own immutable snapshot and
    # before normalized rows/cohort publication are allocated.
    del arrays
    try:
        result=execute_cs_zscore_batch(carrier,params=NORMALIZATION_SPEC['params'])
    finally:
        # A retained exception traceback must not turn the producer frame into
        # an owner of the detached Core input buffers.
        del carrier,sources,dictionary
    metrics['core_calls']+=1; metrics['label_core_calls']+=1
    core_ref=result['metadata']['result_ref']; normalized=[]
    for i,row in enumerate(rows):
        valid=bool(result['value_validity'][i])
        normalized.append({**row,'raw_return':row['return'],'raw_available_at':row['label_available_at'],
            'return':float(result['values'][i]) if valid else None,'valid':valid,
            'label_available_at':_clock(result['available_at_utc_us'][i]) if valid else None,
            'invalid_reason':None if valid else reasons[i] or 'NORMALIZATION_UNDEFINED'})
    # Neither the Core carrier nor its immutable result is a next-fold cache.
    # Keep only the logical normalized rows and small committed references.
    del result
    desc=_publish(target,definition,normalized,budgets=metrics['_limits'],normalized=True,core_ref=core_ref,cohort=cohort,
                  store=metrics['_store'])
    fd['store'].check()
    return desc,normalized,core_ref,cohort


def prepare_compact_batch(data, *, feature_inputs,fold_specs,destination,preparation_options,metrics=None,progress=None,
                          _caller_bytes=0,_caller_source_bytes=0,model_feature_selection=None,reuse_raw_from_batch=None):
    from .stock_compact_batch import load_compact_state
    begin=time.perf_counter(); own=type(feature_inputs) is not StockFeatureView
    require(not own or isinstance(feature_inputs,(str,Path)),'Feature input must be an owner handle or path')
    options=deepcopy(preparation_options)
    required={'row_block_sessions','column_block','maximum_resident_bytes','normalization_backend'}
    require(type(options) is dict and required<=set(options)<=required|{'maximum_source_bytes','maximum_parent_bytes','control_layout'} and
            options['normalization_backend']=='core_cs_batch_v1' and
            options.get('control_layout','fold_controls_v1') in ('fold_controls_v1','inline_v3') and
            all(type(v) is int and v>0 for k,v in options.items() if k not in ('normalization_backend','control_layout')),
            'fixed compact options and positive budgets required')
    compact=options.get('control_layout','fold_controls_v1')=='fold_controls_v1'
    require(model_feature_selection is None or compact,'model_feature_selection requires compact v4 controls')
    budgets=limits({'maximum_matrix_bytes':options['maximum_resident_bytes'],
        **{k:options[k] for k in ('maximum_source_bytes','maximum_parent_bytes') if k in options}})
    require(type(_caller_bytes) is int and _caller_bytes>=0 and type(_caller_source_bytes) is int and _caller_source_bytes>=0,
            'nonnegative owner audit accounting required')
    feature=load_stock_feature_view(feature_inputs,limits=budgets,residency='sequential') if own else feature_inputs
    store=None
    try:
        from .stock_compact_store import model_feature_binding,model_common
        fd=_view_data(feature); source_spec=fd['definition']['spec']; spec=source_spec; width=len(spec['universe'])
        model_binding=model_feature_binding(feature,model_feature_selection)
        if model_binding is not None:
            spec={**source_spec,'ordered_features':model_binding['ordered_features'],'feature_selection':model_binding['selection']}
        require(fd['store'].resident_bytes<=budgets['maximum_matrix_bytes'] and
                fd['store'].metrics['source_bytes']<=budgets['maximum_source_bytes'] and
                fd['store'].metrics['largest_parent_bytes']<=budgets['maximum_parent_bytes'],'borrowed Feature budget incompatible')
        definition={'version':'axiom.stock_ml_batch_inputs/4' if compact else 'axiom.stock_ml_batch_inputs/3','feature_view':feature.to_dict(),
            'fold_specs':deepcopy(fold_specs),'preparation_options':options,'implementation_ref':_implementation(),
            'environment':_input_environment()}
        if compact: definition['price_domain_plan']='fit_window_evaluation_calendar_blocks_v1'
        if model_binding is not None: definition['model_feature_selection']=model_binding
        require(reuse_raw_from_batch is None or compact,'Raw reuse requires compact v4 controls')
        definition_ref=digest(definition); target=Path(destination).absolute()/definition_ref[7:]
        store=OwnedStore(budgets,shared_bytes=fd['store'].resident_bytes+_caller_bytes,
            shared_source_bytes=fd['store'].metrics['source_bytes']+_caller_source_bytes)
        store.check_hook=fd['store'].check
    except BaseException:
        if store is not None: store.close()
        if own: feature.close()
        raise
    transferred=False
    stats={'cache_hit':False,'data_read_calls':0,'supplier_calls':0,'feature_core_calls':0,'core_calls':0,
        'label_core_calls':0,'raw_operator_calls':0,'raw_cache_hits':0,'normalized_cache_hits':0,
        'admitted_price_view_reuses':0,'account_calls':0,'train_calls':0,'predict_calls':0,'fold_queries':[],
        'price_domains':[],
        'maximum_working_bytes':fd['store'].resident_bytes,
        '_limits':budgets,'_feature_bytes':fd['store'].resident_bytes+_caller_bytes,'_retained_raw_bytes':0,'_store':store,
        '_feature_store':fd['store'],'_caller_bytes':_caller_bytes,'_caller_source_bytes':_caller_source_bytes,
        '_model_feature_selection':model_feature_selection}
    try:
        if (target/'batch.json').exists():
            value=json.loads((target/'batch.json').read_bytes()); require(value['definition']==definition,'cached compact definition mismatch')
            state=load_compact_state(value,feature_inputs=feature,limits=budgets,residency='sequential')
            try: state.verify_all()
            except BaseException:
                state.close(); raise
            if value['batch_ref'] in fd['prepared']: fd['prepared'][value['batch_ref']].close()
            fd['prepared'][value['batch_ref']]=state
            stats.update(cache_hit=True,total_seconds=time.perf_counter()-begin)
            if metrics is not None: metrics.update({k:v for k,v in stats.items() if not k.startswith('_')})
            return value
        require(type(fold_specs) is list and bool(fold_specs),'explicit ordered folds required')
        plans=[]; previous=None
        for fold in fold_specs:
            training,inference=validate_spec(fold,spec['calendar'])
            require((previous is None or fold['oos_trade_sessions'][0]>previous) and
                    set(training+inference)<=set(spec['feature_sessions']),'fold grid/chronology mismatch')
            previous=fold['oos_trade_sessions'][-1]; plans.append((fold,training,inference))
        cache=Path(destination).absolute()/'compact-cache'; records=[]; domains={}; raw_outputs=[]
        reused=None
        if reuse_raw_from_batch is not None:
            from .stock_batch import _data
            owner=_data(reuse_raw_from_batch)
            require(owner['manifest']['contract_version']=='stock_ml_batch_inputs_v4',
                    'Raw reuse requires an owner-loaded compact v4 batch')
            origin=owner['matrix_state']; origin.check()
            require(origin.compact and origin.active==0 and
                    origin.batch['definition']['feature_view']==feature.to_dict(),'Raw reuse Feature identity mismatch')
            require(origin.batch['definition']['preparation_options']['row_block_sessions']==options['row_block_sessions'],
                    'Raw reuse row-block query plan mismatch')
            by_spec={digest(f['fold_spec']):f for f in origin.batch['folds']}
            reused=[]
            for fold,_,_ in plans:
                require(digest(fold) in by_spec,'Raw reuse fold/cutoff mismatch')
                saved=by_spec[digest(fold)]; control=origin._parts(saved['input_manifest'],fold)
                reused.append({'raw_parts':deepcopy(control['raw_parts']),
                    'evaluation_parts':deepcopy(control['evaluation_parts'])})
            origin.check()
            stats['raw_reused_from_batch']=origin._batch_ref
        for i,(fold,training,inference) in enumerate(plans):
            if reused is not None:
                raw_outputs.append(reused[i]); stats['raw_cache_hits']+=len(reused[i]['raw_parts'])+len(reused[i]['evaluation_parts'])
                continue
            chunks=[training[n:n+options['row_block_sessions']] for n in range(0,len(training),options['row_block_sessions'])]
            raw_outputs.append({'raw_parts':[None]*len(chunks),
                **({'evaluation_parts':[]} if compact else {'evaluation':None})})
            stats['fold_queries'].append({'fold_spec_ref':digest(fold),'data_read_calls':0,
                'admitted_price_view_reuses':0,'price_domain_refs':[]})
            jobs=[('raw_parts',n,days,fold['fit_cutoff']) for n,days in enumerate(chunks)]
            if compact:
                positions={day:n for n,day in enumerate(spec['calendar'])}; evaluation={}
                for day in inference:
                    evaluation.setdefault(positions[day]//options['row_block_sessions'],[]).append(day)
                raw_outputs[i]['evaluation_parts']=[None]*len(evaluation)
                jobs.extend(('evaluation_parts',n,days,fold['evaluation_cutoff'])
                    for n,days in enumerate(evaluation.values()))
            else: jobs.append(('evaluation',None,inference,fold['evaluation_cutoff']))
            for role,n,days,cutoff in jobs:
                if not compact: key=_instant(cutoff)
                elif role=='evaluation_parts': key=(_instant(cutoff),'evaluation',positions[days[0]]//options['row_block_sessions'])
                else: key=(_instant(cutoff),'training')
                group=domains.setdefault(key,{'cutoff':cutoff,'days':set(),'jobs':[]})
                group['days'].update(days); group['jobs'].append((i,role,n,days,cutoff))
        # One full fit domain or fixed evaluation block; release it before
        # selecting the next domain. Raw children retain the full query
        # identity, never an invented sliced DataBatch contract.
        for group in domains.values():
            stats['_price_view_bytes']=0
            price_view=_price_view(data,spec,group['cutoff'],sorted(group['days']),stats)
            for use,(i,role,n,days,cutoff) in enumerate(group['jobs']):
                report=stats['fold_queries'][i]; domain_ref=price_view[3]['price_view_ref']
                if domain_ref not in report['price_domain_refs']: report['price_domain_refs'].append(domain_ref)
                if use: stats['admitted_price_view_reuses']+=1; report['admitted_price_view_reuses']+=1
                else: report['data_read_calls']+=2
                desc,chunk=_raw(spec,cutoff,days,cache,stats,price_view)
                if role in ('raw_parts','evaluation_parts'): raw_outputs[i][role][n]=desc
                else: raw_outputs[i][role]=desc
                del chunk
                # Published descriptors, rather than every historical typed
                # target, are the retained union between cutoff domains.
                store.release_payloads()
            del price_view
            stats['_price_view_bytes']=0
        from .stock_compact_batch import read_target
        positions={d:i for i,d in enumerate(spec['feature_sessions'])}
        for i,(fold,training,inference) in enumerate(plans):
            window=[positions[d]*width+j for d in training+inference for j in range(width)]
            _producer_feature_window(feature,window,stats)
            parts=raw_outputs[i]['raw_parts']; rows=[]
            for desc in parts:
                raw_value,raw_rows=read_target(store,desc); chunk=list(raw_rows)
                chunk_bytes=_measured(stats,chunk); _working(stats,chunk_bytes)
                stats['_retained_raw_bytes']+=chunk_bytes; rows.extend(chunk)
                del raw_value,raw_rows,chunk
                store.release_payloads()
            norm,nrows,core_ref,cohort=_normalized(parts,rows,feature,fold['fit_cutoff'],training,cache,stats)
            normalized_bytes=_measured(stats,[nrows,cohort]); _working(stats,normalized_bytes)
            stats['_retained_raw_bytes']+=normalized_bytes
            store.release_payloads()
            training_offsets=[positions[r['feature_session']]*width+spec['universe'].index(r['security_id'])
                for r in nrows if r['valid']]
            inference_offsets=[positions[d]*width+i for d in inference for i in range(width)]
            _working(stats,len(spec['ordered_features'])*96+3072)
            def joined_rows():
                selected=(r for r in nrows if r['valid'])
                for f,r in zip(iter_feature_rows(feature,training_offsets,
                        model_feature_selection=model_feature_selection),selected):
                    f.update(label=r['return'],raw_return=r['raw_return'],label_available_at=r['raw_available_at'],
                             normalized_available_at=r['label_available_at'])
                    yield f
            training_binding={'training_rows_ref':digest_array_rows(joined_rows()),'training_row_count':len(training_offsets),
                'training_keys_digest':digest_array_rows([spec['universe'][off%width],
                    spec['feature_sessions'][off//width]] for off in training_offsets)}
            record={'fold_spec':deepcopy(fold),'raw_parts':parts,'normalized':norm,
                'core_ref':core_ref,'cohort_ref':digest(cohort),'training_binding':training_binding}
            if compact:
                from .stock_compact_controls import selectors_for
                record.update(contract_version='stock_ml_fold_control_v1',evaluation_parts=raw_outputs[i]['evaluation_parts'])
                if model_binding is not None: record['model_feature_selection_ref']=model_binding['model_feature_selection_ref']
                selectors=selectors_for(record,spec,fd['row_index'],training_offsets)
                record=seal(record,'fold_control_ref')
                descriptor=write_part(cache/'fold-controls',record,'fold_control_ref')
                records.append({'fold_spec':record['fold_spec'],'fold_control':descriptor,'selectors':selectors,'core_ref':core_ref})
                del record,selectors,training_offsets,inference_offsets
            else:
                record.update(evaluation=raw_outputs[i]['evaluation'],training_offsets=training_offsets,inference_offsets=inference_offsets)
                records.append(record)
            if progress: progress({'stage':'compact_labels','completed':len(records),'total':len(plans)})
            # Completed fold facts live in typed targets; release their Python
            # working panels before the next cutoff-specific Data selection.
            del rows,nrows,cohort,joined_rows,window
            stats['_retained_raw_bytes']=0
            store.release_payloads()
            # Preserve already verified overlap for the next full window;
            # set_feature_window trims only partitions outside that window.
        if model_binding is None: _producer_feature_window(feature,[],stats)
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.compact-batch-',dir=target.parent) as temporary:
            stage=Path(temporary)/'complete'; stage.mkdir()
            common=model_common(source_spec,model_binding)
            view=seal({'contract_version':'stock_ml_prepared_view_v3' if compact else 'stock_ml_prepared_view_v2',
                'definition':common,'feature_view':feature.to_dict(),**({} if compact else {'fold_targets':records})},'prepared_view_ref')
            write_json(stage/'view.json',view)
            view_desc={'path':str(target/'view.json'),'file_digest':file_digest(stage/'view.json'),
                       'prepared_view_ref':view['prepared_view_ref']}
            folds=[]
            for record in records:
                selectors=record['selectors'] if compact else {k:None if k=='validation' else record['training_offsets' if k in
                    ('training','training_labels') else 'inference_offsets'] for k in
                    ('training','validation','inference','training_labels','evaluation_labels')}
                inputs=seal({'contract_version':'stock_ml_saved_inputs_v4' if compact else 'stock_ml_saved_inputs_v3','prepared_view':view_desc,
                    **({'fold_control':record['fold_control']} if compact else {}),
                    'fold_spec_ref':digest(record['fold_spec']),'selectors':selectors,
                    'core_result_refs':[record['core_ref']]},'input_ref')
                if model_binding is not None:
                    inputs=seal({**{k:v for k,v in inputs.items() if k!='input_ref'},
                        'model_feature_selection_ref':model_binding['model_feature_selection_ref']},'input_ref')
                folds.append({'input_manifest':inputs,'fold_spec':record['fold_spec']})
            batch={'contract_version':'stock_ml_batch_inputs_v4' if compact else 'stock_ml_batch_inputs_v3','definition':definition,'definition_ref':definition_ref,
                'prepared_view':view_desc,'folds':folds,'status':'COMPLETE'}
            batch['batch_ref']=digest(batch); batch=seal(batch,'content_digest'); write_json(stage/'batch.json',batch)
            state=load_compact_state(batch,feature_inputs=feature,limits=budgets,
                resolver=lambda p:stage/Path(p).relative_to(target) if Path(p).is_relative_to(target) else Path(p),
                publication_store=store,residency='sequential')
            # COMPLETE requires the same full saved closure validation, one
            # active fold at a time, before the directory becomes visible.
            try:
                state.verify_all()
                transferred=True
                stage.rename(target); state.store.resolve=lambda p:Path(p); state.check()
            except BaseException:
                state.close(); raise
            fd['prepared'][batch['batch_ref']]=state
        stats.update(total_seconds=time.perf_counter()-begin,batch_ref=batch['batch_ref'],
            feature_file_hash_calls=fd['store'].metrics['file_hash_calls'],
            label_file_hash_calls=store.metrics['file_hash_calls'],label_hash_bytes=store.metrics['hash_bytes'],
            label_released_buffer_bytes=store.metrics.get('released_buffer_bytes',0),
            legacy_ancestor_reads=0,legacy_native_hash_calls=0)
        if metrics is not None: metrics.update({k:v for k,v in stats.items() if not k.startswith('_')})
        return batch
    finally:
        if not transferred: store.close()
        if own: feature.close()
        if stats['data_read_calls'] and callable(getattr(data,'clear_cache',None)): data.clear_cache()

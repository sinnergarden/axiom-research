"""Scope and missing-rectangle reuse for the existing physical input format.

No new storage/registry: unchanged evidence and buffers keep their descriptors.
Only missing dates are read; changed numeric layouts are copied in
bounded rows, without recomputing the already prepared values.
"""
from copy import deepcopy
import json
from pathlib import Path

from .stock_artifacts import digest, file_digest, _verify_ref
from .stock_fold_inputs import require, seal
from .stock_matrix_storage import write_part


def source_descriptor(path):
    path=Path(path).resolve()
    m=json.loads(path.read_text());_verify_ref(m,'source_ref')
    return {'path':str(path),'file_digest':file_digest(path),'source_ref':m['source_ref']}


def open_physical_source(path):
    """Admit the header and hash/map each published buffer on its first use."""
    from .stock_feature_columns import FeatureColumnSource
    from .stock_feature_library import _SourceArrays
    from .stock_matrix_reader import VerifiedMatrixStore
    descriptor=source_descriptor(path)
    store=VerifiedMatrixStore(maximum_source_bytes=256*1024**3,maximum_matrix_bytes=2*1024**3)
    manifest=store.read_json(descriptor,'source_ref')
    source=FeatureColumnSource(manifest,_SourceArrays(manifest,store))
    source._history_store=store
    return source


def history_scope(data, *, snapshot, catalog, selection, output_sessions,
                  physical_universe, destination, universe_id='csi1800'):
    """Derive required prior sessions; calendar gaps fail before expensive work."""
    history=max(f['lookback'] for f in catalog.select(selection))-1
    request={'snapshot':snapshot,'universe_id':universe_id,
        'output_sessions':list(output_sessions),
        'cutoff_by_session':{d:d+'T20:30:00+08:00' for d in output_sessions},
        'pit_policy':'best_effort_vendor_v1','lookback_sessions':history,
        'held_symbols':list(physical_universe),'exchange':'SSE'}
    scope=data.plan_scope(**request)
    calendar_gaps=[g for g in scope['missing_reasons'] if g['domain']=='trading_calendar']
    require(not calendar_gaps,'required output/warmup calendar unavailable: '+str(calendar_gaps[:4]))
    # Keep missing membership explicitly in a separate receipt; never silently
    # convert unknown to an empty/complete reference pool.
    gaps=scope.pop('missing_reasons')
    from .feature_input_checkpoint import atomic_json
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'scope-gaps.json',{'count':len(gaps),'by_reason':{
        reason:sum(g['reason']==reason for g in gaps) for reason in sorted({g['reason'] for g in gaps})},
        'examples':gaps[:25]})
    proof={'source':'Data.plan_scope','snapshot':snapshot,'gap_count':len(gaps),
           'warmup_sessions':history,'physical_universe_preserved':list(physical_universe)}
    bundle=seal({'contract_version':'stock_scope_bundle_v1','purpose':'QSYS reusable Feature inputs',
        'request':request,'request_ref':digest(request),'result':scope,'result_ref':digest(scope),
        'source_proof':proof,'source_proof_ref':digest(proof)},'scope_bundle_ref')
    descriptor=write_part(root,bundle,'scope_bundle_ref')
    days=scope['read_sessions']
    return {'contract_version':'stock_feature_inputs_spec_v1','scope':descriptor,
        'snapshot':snapshot,'pit_policy':request['pit_policy'],'calendar':days,
        'read_sessions':days,'feature_sessions':list(output_sessions),'universe':scope['read_symbols'],
        'catalog_ref':catalog.identity,'feature_selection':selection,
        'ordered_features':[s['id'] for s in selection],
        'cutoff_by_session':{d:d+'T20:30:00+08:00' for d in days}}


def _subset_spec(spec, days, symbols, root):
    """Explicit missing rectangle, not a shortened requested Feature output."""
    value=deepcopy(spec);value.update(calendar=list(days),read_sessions=list(days),
        feature_sessions=list(days),universe=list(symbols),
        cutoff_by_session={d:spec['cutoff_by_session'][d] for d in days})
    original=json.loads(Path(spec['scope']['path']).read_text())
    _verify_ref(original,'scope_bundle_ref')
    request={**original['request'],'output_sessions':list(days),'lookback_sessions':0,
             'cutoff_by_session':value['cutoff_by_session']}
    result={**original['result'],'read_sessions':list(days),'read_symbols':list(symbols),
            'output_sessions':list(days),'warmup_sessions':[],
            'decision_membership_by_session':{d:original['result']['decision_membership_by_session'].get(d,[]) for d in days}}
    proof={'source':'missing_rectangle_of_public_scope','parent_scope':spec['scope'],
           'sessions':list(days),'security_ids':list(symbols)}
    bundle=seal({'contract_version':'stock_scope_bundle_v1','purpose':'missing physical cells only',
        'request':request,'request_ref':digest(request),'result':result,'result_ref':digest(result),
        'source_proof':proof,'source_proof_ref':digest(proof)},'scope_bundle_ref')
    value['scope']=write_part(root,bundle,'scope_bundle_ref')
    return value


def _project_chunks(chunks, days, symbols, root):
    """Keep whole matching proofs; trim only boundary/subset proof documents."""
    selected=set(days);wanted=set(symbols or ());result=[]
    for chunk in chunks:
        chosen=[d for d in chunk['sessions'] if d in selected]
        if not chosen:continue
        proofs=[]
        for desc in chunk['evidence']:
            if chosen==chunk['sessions'] and symbols is None:
                proofs.append(desc);continue
            require(file_digest(desc['path'])==desc['file_digest'],'reused evidence changed')
            child=json.loads(Path(desc['path']).read_text());_verify_ref(child,'evidence_ref')
            child['sessions']=chosen
            child['field_meta']={f:{**m,'by_key':[r for r in m['by_key']
                if r['session'] in selected and (symbols is None or r['security_id'] in wanted)]}
                for f,m in child['field_meta'].items()}
            child.pop('evidence_ref');child=seal(child,'evidence_ref')
            proofs.append(write_part(root,child,'evidence_ref'))
        result.append({'sessions':chosen,'evidence':proofs})
    return result


def reuse_history_source(data, *, source, spec, catalog, destination,
                         event_cache_bytes=64*1024**2, progress=None):
    """Reuse existing cells and prepare only missing date rectangles.

    Native daily domain identity must match before old cells can cross a
    Snapshot boundary. Physical axes may grow or project; Core math is untouched.
    """
    import numpy as np
    from .stock_feature_columns import build_feature_column_source, FeatureColumnSource
    from .qsys_preparation import prepare_qsys_inputs
    from .feature_input_buffers import allocate_feature_buffer,seal_feature_buffer
    from .feature_input_checkpoint import atomic_json
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    old=source.manifest['spec'];days=spec['read_sessions'];symbols=spec['universe']
    require(old['pit_policy']==spec['pit_policy'] and all(
        old['cutoff_by_session'][d]==spec['cutoff_by_session'][d] for d in set(days)&set(old['read_sessions'])),
        'existing input policy/cutoffs differ from request')
    domains=[d for d in source.manifest['source_profiles'] if not d.startswith('materialized_')]
    if old['snapshot']!=spec['snapshot']:
        prior=data.describe_snapshot(snapshot=old['snapshot'],domains=domains,include_domain_refs=True)
        current=data.describe_snapshot(snapshot=spec['snapshot'],domains=domains,include_domain_refs=True)
        require(prior['domains']==current['domains'],'existing native domain identity changed; explicit recomputation required')
        data.clear_cache()
    rectangles=[]
    missing_days=[d for d in days if d not in source.positions]
    missing_symbols=[s for s in symbols if s not in source.security_positions]
    require(not missing_symbols,'additional securities need explicit complete-axis input preparation; existing cells have not been changed')
    if missing_days:
        require(all(d['event_snapshot']==spec['snapshot'] for d in
                    source.manifest.get('financial_materializations',{}).values()),
                'financial date extension must retain its pinned event Snapshot; changed event facts need explicit preparation')
    if missing_days:rectangles.append((missing_days,symbols))
    overlap=[d for d in days if d in source.positions]
    sources=[source]
    for ordinal,(delta_days,delta_symbols) in enumerate(rectangles):
        target=root/f'gap-{ordinal}'
        delta_spec=_subset_spec(spec,delta_days,delta_symbols,target)
        delta,_=build_feature_column_source(data,spec=delta_spec,destination=target/'base',block_sessions=2,checkpoint=True,progress=progress)
        fields=set(source.manifest.get('daily_fields',{}))
        if fields:
            from .qsys_catalog import qsys_input_requirements
            req=qsys_input_requirements()
            delta,_=prepare_qsys_inputs(data,source=delta,names=spec['ordered_features'],
                input_fields=fields,destination=target/'added',
                event_snapshots={b['domain']:spec['snapshot'] for b in req['financial_fields'].values()},
                report_start='2010-01-01',report_end=days[-1],daily_snapshot=spec['snapshot'],
                benchmark_snapshot=spec['snapshot'],industry_snapshot=spec['snapshot'],
                benchmark_security_id=source.manifest['source_profiles'].get('materialized_benchmark_daily',{}).get('source_profile',{}).get('benchmark_security_id'),
                block_sessions=2,maximum_added_bytes=8*1024**3,maximum_event_cache_bytes=event_cache_bytes)
        require(set(delta.manifest['buffers'])==set(source.manifest['buffers']),'gap source fields differ from reused source')
        sources.append(delta)
    manifest=deepcopy(source.manifest);manifest['spec']=deepcopy(spec)
    arrays={};same_axes=days==old['read_sessions'] and symbols==old['universe']
    positions={d:i for i,d in enumerate(days)};security_positions={s:i for i,s in enumerate(symbols)}
    metrics={'reused_dates':len(overlap),'reused_securities':len(set(symbols)&set(old['universe'])),
             'missing_date_count':len(missing_days),'missing_security_count':len(missing_symbols),
             'old_value_recomputations':0,'buffer_copy_bytes':0}
    for name,desc in manifest['buffers'].items():
        if same_axes:
            arrays[name]=source.arrays[name];continue
        original=source.arrays[name]
        shape=(len(days),len(symbols),*original.shape[2:])
        target=allocate_feature_buffer(root,dtype=original.dtype,shape=shape)
        for part in sources:
            old_days=part.manifest['spec']['read_sessions'];old_symbols=part.manifest['spec']['universe']
            selected=[(i,positions[d]) for i,d in enumerate(old_days) if d in positions]
            cols=[(j,security_positions[s]) for j,s in enumerate(old_symbols) if s in security_positions]
            if not selected or not cols:continue
            src_cols,dst_cols=zip(*cols);value=part.arrays[name]
            for src_row,dst_row in selected:
                target[dst_row,list(dst_cols)]=value[src_row,list(src_cols)]
        manifest['buffers'][name],arrays[name]=seal_feature_buffer(root,target,dtype=desc['dtype'],shape=shape)
        metrics['buffer_copy_bytes']+=int(np.prod(shape))*original.dtype.itemsize
    # Reuse whole proofs when their date and security axes already match.
    manifest['chunks']=[]
    for part in sources:
        axis=part.manifest['spec']['universe']
        manifest['chunks'].extend(_project_chunks(part.manifest['chunks'],days,None if axis==symbols else symbols,root))
    manifest['chunks'].sort(key=lambda c:c['sessions'][0])
    extensions=[]
    for old_ext in source.manifest.get('daily_extensions',[]):
        fields=old_ext['fields'];chunks=[]
        for part in sources:
            for ext in part.manifest.get('daily_extensions',[]):
                if set(ext['fields'])==set(fields):chunks.extend(_project_chunks(ext['chunks'],days,None if part.manifest['spec']['universe']==symbols else symbols,root))
                elif set(ext['fields'])&set(fields):
                    raise ValueError('gap extension grouping differs; explicit field projection required')
        chunks.sort(key=lambda c:c['sessions'][0]);extensions.append({'fields':fields,'chunks':chunks})
    manifest['daily_extensions']=extensions
    for decl in manifest.get('financial_materializations',{}).values():
        decl['coverage_security_ids']=[s for s in decl['coverage_security_ids'] if s in set(symbols)]
    manifest.pop('source_ref');manifest=seal(manifest,'source_ref')
    atomic_json(root/'source.json',manifest)
    if progress is not None:progress({'stage':'input_reuse',**metrics})
    return FeatureColumnSource(manifest,arrays),metrics

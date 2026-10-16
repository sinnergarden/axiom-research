"""Offline QSYS preparation through existing Data, input and Feature owners.

Use --config. Reuse a published physical source and pinned candidate; prepare
only absent domains/cells/fields. No supplier calls, training, or automatic
full-history start. COMPLETE means actual requested Feature coverage, never a
marker filename. See the public Feature lifecycle workflow in axiom-docs.
"""
import argparse
import json
from pathlib import Path
import resource
import shutil
import time


def prepare_candidate(cfg, root, requirements, progress):
    from axiom_data import Data
    from axiom_data.storage import LocalStore
    from axiom_research.stock_artifacts import digest
    from axiom_research.feature_input_checkpoint import atomic_json
    store=LocalStore(root/'data')
    binding={'candidate_root':cfg['candidate_root'],'candidate_snapshot':cfg['candidate_snapshot'],
             'market_root':cfg['market_root'],'market_snapshot':cfg['market_snapshot'],
             'raw_roots':cfg.get('raw_roots',[])}
    required={'market_daily','adjustment_factors','universe_membership','trading_calendar',
              *[v['domain'] for v in requirements['daily_fields'].values()],
              *[v['domain'] for v in requirements['financial_fields'].values()]}
    if requirements['benchmark_fields']:required.add('benchmark_daily')
    if requirements['industry_fields']:required.add('industry_membership_events')
    marker=root/'candidate-snapshot.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved.get('request_ref')==digest(binding):
            manifest=store.load_snapshot(saved['snapshot_id'])
            if required<=set(manifest['domains']):
                return Data(root/'data',cache_bytes=cfg.get('event_cache_bytes',64*1024**2),max_readers=1),saved['snapshot_id']
    progress('candidate',required_domains=sorted(required))
    manifest=store.borrow_snapshot(cfg['candidate_root'],cfg['candidate_snapshot'])
    base=manifest['snapshot_id'];domains=dict(manifest['domains']);missing=required-set(domains)
    if missing:
        parent=store.borrow_snapshot(cfg['market_root'],cfg['market_snapshot'])
        restored=missing&set(parent['domains'])
        for domain in sorted(restored):domains[domain]=parent['domains'][domain]
        del parent
        if set(domains)!=set(manifest['domains']):
            # Publication verifies newly attached Raw IDs. Borrow their exact
            # receipts before attaching the restored canonical domain.
            store.borrow_raw([cfg['market_root'],*cfg.get('raw_roots',[])],domains=restored)
            base=store.publish_snapshot(domains,parent_snapshot=base,promote=False,
                build_context={'purpose':'retain all prior domains and restore required missing domains',
                               'original_roots':[cfg['candidate_root'],cfg['market_root']]})['snapshot_id']
    missing=required-set(domains);del manifest,domains
    data=Data(root/'data',cache_bytes=cfg.get('event_cache_bytes',64*1024**2),max_readers=1)
    if missing:
        ids=store.borrow_raw(cfg.get('raw_roots',[]),domains=missing)
        if not ids:
            raise ValueError('MISSING_RAW_REQUIRES_SEPARATE_COLLECTION: '+','.join(sorted(missing)))
        progress('missing_domain_rebuild',domains=sorted(missing),raw_receipts=len(ids))
        base=data.rebuild(base_snapshot=base,raw_batch_ids=ids,domains=sorted(missing),
            operation_id='qsys-missing-domains-'+digest([base,ids,sorted(missing)])[7:23],
            build_context={'purpose':'only missing required domains from existing Raw'},promote=False).snapshot_id
    data.describe_snapshot(snapshot=base,domains=sorted(required))
    data.clear_cache()
    atomic_json(marker,{'snapshot_id':base,'root':str(root/'data'),'request_ref':digest(binding)})
    return data,base


def run(cfg, *, observer=None):
    from axiom_research.feature_input_evidence import configure_input_disk_reserve
    from axiom_research.feature_input_checkpoint import atomic_json
    from axiom_research.qsys_history_inputs import (open_physical_source,history_scope,
                                                   reuse_history_source,source_descriptor)
    from axiom_research.qsys_catalog import load_qsys_feature_catalog,qsys_input_requirements
    from axiom_research.stock_feature_library import SavedColumnFeatureSource,CatalogFeatureLibrary
    from axiom_research.stock_feature_columns import extend_feature_column_source,build_feature_column_source
    from axiom_research.financial_feature_columns import extend_financial_feature_sources
    from axiom_research.benchmark_feature_columns import extend_benchmark_feature_source
    from axiom_research.industry_feature_columns import extend_industry_feature_source
    from axiom_research.stock_artifacts import digest
    from axiom_engine.feature_library import FeatureLibrary
    from axiom_data import EventQuery
    root=Path(cfg['destination']).resolve();root.mkdir(parents=True,exist_ok=True)
    configure_input_disk_reserve(cfg.get('minimum_free_bytes',15*1024**3))
    started=time.monotonic();timings={};previous=started;previous_stage=None
    def progress(stage,**extra):
        nonlocal previous,previous_stage
        now=time.monotonic()
        if stage!=previous_stage:
            if previous_stage is not None:timings[previous_stage]=timings.get(previous_stage,0)+now-previous
            previous=now;previous_stage=stage
        record={'stage':stage,'elapsed_seconds':now-started,
                'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'disk_free_bytes':shutil.disk_usage(root).free,**extra}
        atomic_json(root/'progress.json',record)
        print(json.dumps(record,default=str),flush=True)
        if observer is not None:observer(record)
        if record['disk_free_bytes']<cfg.get('minimum_free_bytes',15*1024**3):
            raise RuntimeError('DISK_RESERVE_REACHED')
        if record['peak_rss_bytes']>cfg.get('maximum_rss_bytes',12*1024**3):
            raise RuntimeError('RSS_LIMIT_REACHED')
    catalog=load_qsys_feature_catalog();selection=[s.to_dict() for s in catalog.default_selection]
    requirements=qsys_input_requirements()
    progress('candidate')
    data,base=prepare_candidate(cfg,root,requirements,progress)
    progress('scope')
    original=None if cfg.get('rebuild_inputs',False) else open_physical_source(cfg['reuse_source'])
    seed=json.loads(Path(cfg['scope_seed']).read_text()) if cfg.get('scope_seed') else {
        'output_sessions':original.manifest['spec']['feature_sessions'],
        'physical_universe':original.manifest['spec']['universe']}
    outputs=cfg.get('output_sessions')
    if outputs is None:
        outputs=list(seed['output_sessions'])
        end=cfg.get('output_end')
        if end and end not in outputs:outputs=sorted([*outputs,end])
    if outputs!=sorted(set(outputs)) or not outputs:raise ValueError('explicit sorted output sessions required')
    from axiom_research.stock_feature_inputs import _spec
    source_universe=None if original is None else _spec(original.manifest['spec'])
    universe_id=cfg.get('universe_id',source_universe)
    if not isinstance(universe_id,str) or not universe_id:raise ValueError('explicit configured universe_id required')
    if source_universe is not None and source_universe!=universe_id:raise ValueError('configured universe differs from reused membership')
    spec=history_scope(data,snapshot=base,catalog=catalog,selection=selection,universe_id=universe_id,
        output_sessions=outputs,physical_universe=seed['physical_universe'],destination=root/'scope')
    request_ref=digest(spec)
    if cfg.get('prepared_inputs_only'):
        if original is None:raise ValueError('published prepared source required')
        prior=original.manifest['spec']
        if ({k:v for k,v in prior.items() if k!='scope'}!={k:v for k,v in spec.items() if k!='scope'} or
                prior['scope']['scope_bundle_ref']!=spec['scope']['scope_bundle_ref']):
            raise ValueError('prepared-input continuation differs from frozen plan')
    atomic_json(root/'request.json',{'request_ref':request_ref,'spec':spec,
        'research_range':cfg.get('research_range',['2019-01-02','2025-12-31'])})
    progress('input_reuse')
    reuse_path=root/'inputs/reused/source.json'
    if reuse_path.exists():
        source=open_physical_source(reuse_path)
        if source.manifest['spec']!=spec:raise ValueError('saved input request differs; use a new destination and reuse_source')
        reuse_metrics={'reused_complete_source':True,'old_value_recomputations':0}
    elif original is None:
        source,reuse_metrics=build_feature_column_source(data,spec=spec,destination=reuse_path.parent,
            block_sessions=2,checkpoint=True,progress=lambda detail:progress('input_rebuild',detail=detail))
        reuse_metrics={**reuse_metrics,'explicit_full_input_rebuild':True}
    else:
        source,reuse_metrics=reuse_history_source(data,source=original,spec=spec,catalog=catalog,
            destination=reuse_path.parent,event_cache_bytes=cfg.get('event_cache_bytes',64*1024**2),
            progress=lambda detail:progress('input_reuse',detail=detail))
    source_path=reuse_path
    if cfg.get('prepared_inputs_only') and source.manifest['buffers']!=original.manifest['buffers']:
        raise ValueError('prepared-input continuation changed numeric buffer descriptors')
    if cfg.get('prepared_inputs_only'):
        required=set(requirements['daily_fields'])|set(requirements['benchmark_fields'])|set(requirements['industry_fields'])
        required.update(n for b in requirements['financial_fields'].values() for n in b['outputs'])
        if not required<=set(source.physical_fields()):raise ValueError('published continuation lacks required inputs')
    from axiom_research.financial_feature_columns import QUARTER_POLICY
    if any('quarters_back' in d and d['selection']!=QUARTER_POLICY
           for d in source.manifest.get('financial_materializations',{}).values()):
        raise ValueError('old quarterly inputs require preparation under the current quarter selection rule')
    atomic_json(root/'input-reuse.json',reuse_metrics)
    # Stage receipts bind both request and parent source. A path alone is never
    # evidence that the current request or newly requested fields are complete.
    def resume(path,binding):
        receipt=path.parent/'stage.json'
        if not path.exists():
            atomic_json(receipt,{'binding':digest(binding),'source_ref':None,'status':'PREPARING'})
            return None
        if not receipt.exists():raise ValueError('published stage lacks request binding: '+str(path))
        saved=json.loads(receipt.read_text())
        if saved['binding']!=digest(binding):raise ValueError('published stage request differs: '+str(path))
        result=open_physical_source(path)
        if saved['source_ref'] not in (None,result.identity) or result.manifest['spec']!=spec:
            raise ValueError('published stage source/axes mismatch')
        if saved['source_ref'] is None:
            inherited=source.manifest['buffers']
            if any(result.manifest['buffers'].get(k)!=v for k,v in inherited.items()):
                raise ValueError('recovered stage changed inherited input buffers')
            expected=set(binding.get('fields',{}))|set(binding.get('bundle',{}).get('outputs',{}))
            if not expected<=set(result.physical_fields()):raise ValueError('recovered stage lacks requested fields')
            atomic_json(receipt,{'binding':digest(binding),'source_ref':result.identity,'status':'COMPLETE','recovered_publication':True})
        return result
    def finish(path,binding,value,metrics):
        atomic_json(path.parent/'stage.json',{'binding':digest(binding),'source_ref':value.identity,'metrics':metrics})
    for domain in sorted({v['domain'] for v in requirements['daily_fields'].values()}):
        fields={k:v for k,v in requirements['daily_fields'].items() if v['domain']==domain and k not in source.physical_fields()}
        if not fields:continue
        path=root/'inputs'/('daily-'+domain)/'source.json';binding={'request':request_ref,'parent':source.identity,'fields':fields,'snapshot':base}
        cached=resume(path,binding)
        if cached is not None:source=cached;source_path=path;continue
        progress('daily_inputs',domain=domain)
        source,metrics=extend_feature_column_source(data,source=source,daily_fields=fields,
            snapshot=base,destination=path.parent,block_sessions=2,maximum_added_bytes=8*1024**3,
            progress=lambda detail:progress('daily_inputs',domain=domain,detail=detail))
        finish(path,binding,source,metrics);source_path=path
    event_bundles={key:bundle for key,bundle in requirements['financial_fields'].items()
        if any(field not in source.physical_fields() for field in bundle['outputs'])}
    if cfg.get('prepared_inputs_only') and event_bundles:
        raise ValueError('prepared continuation lacks required event inputs')
    if event_bundles:
        path=root/'inputs'/'events'/'source.json'
        binding={'request':request_ref,'parent':source.identity,'bundles':event_bundles,'snapshot':base,
                 'execution':'stock_block_event_spans_v1'}
        cached=resume(path,binding)
        if cached is not None:source=cached;source_path=path
        else:
            progress('financial_inputs',bundle='unified_events')
            source,metrics=extend_financial_feature_sources(data,source=source,bundles=event_bundles,
                destination=path.parent,event_snapshot=base,report_start=cfg.get('report_start','2010-01-01'),
                report_end=outputs[-1],maximum_added_bytes=8*1024**3,
                maximum_event_cache_bytes=cfg.get('event_cache_bytes',512*1024**2),
                maximum_group_bytes=cfg.get('event_group_bytes',256*1024**2),
                stock_block_limit=cfg.get('event_stock_block_limit'),
                progress=lambda detail:progress('financial_inputs',bundle='unified_events',detail=detail))
            finish(path,binding,source,metrics);source_path=path
    for kind,field,fn,kw in [('benchmark','benchmark_close',extend_benchmark_feature_source,
                             {'benchmark_security_id':cfg['benchmark_security_id']}),
                            ('industry','industry_code',extend_industry_feature_source,{})]:
        if field in source.physical_fields():continue
        path=root/'inputs'/kind/'source.json';binding={'request':request_ref,'parent':source.identity,
            'snapshot':base,'fields':{field:kind},**kw}
        cached=resume(path,binding)
        if cached is not None:source=cached;source_path=path;continue
        progress(kind+'_inputs')
        source,metrics=fn(data,source=source,snapshot=base,destination=path.parent,
            block_sessions=2,maximum_added_bytes=1024**3,
            progress=lambda detail:progress(kind+'_inputs',detail=detail),**kw)
        finish(path,binding,source,metrics);source_path=path
    atomic_json(root/'prepared-source.json',source_descriptor(source_path))
    # Release the Data Snapshot and mutable preparation objects before Core.
    data.clear_cache()
    if original is not None:original._history_store.close()
    del source,data,original
    import gc
    gc.collect()
    progress('features')
    library=FeatureLibrary(root/'library',maximum_working_bytes=cfg.get('working_bytes',2*1024**3),
                           maximum_source_bytes=cfg.get('maximum_source_bytes',256*1024**3))
    views=[]
    with library.execution_session(), SavedColumnFeatureSource(source_descriptor(source_path),maximum_source_bytes=cfg.get('maximum_source_bytes',256*1024**3),
                                 maximum_working_bytes=cfg.get('working_bytes',2*1024**3)) as admitted:
        owner=CatalogFeatureLibrary(library,admitted,catalog)
        block=cfg.get('output_block_sessions',5)
        for index in range(0,len(outputs),block):
            days=outputs[index:index+block]
            # Always ask Engine for real coverage. Existing immutable columns
            # are reused; changed block size cannot produce false COMPLETE.
            view,metrics=owner.ensure(selection=selection,output_sessions=days,
                normalized=cfg.get('normalized',False),
                progress=lambda detail:progress('features',first=days[0],detail=detail))
            path=root/'views'/(digest([request_ref,days,cfg.get('normalized',False)])[7:]+'.json')
            atomic_json(path,{'view':view,'metrics':metrics,'request_ref':request_ref,'sessions':days})
            views.append(view)
        progress('native_consumer_publication')
        # Reuse all saved tiles to publish the complete requested FeatureView;
        # this call does not accept marker-only completion.
        full,coverage=owner.ensure(selection=selection,output_sessions=outputs,normalized=cfg.get('normalized',False))
        from axiom_research.stock_feature_library_adapter import build_feature_library_training_view
        with library.open_feature_view(full) as view:
            training=build_feature_library_training_view(view,admitted,selection=selection,
                destination=root/'training-view',maximum_working_bytes=cfg.get('working_bytes',2*1024**3))
    progress('complete')
    result={'status':'COMPLETE_WITH_EXPLICIT_MISSING','catalog_ref':catalog.identity,'request_ref':request_ref,
        'dates':len(outputs),'bounds':[outputs[0],outputs[-1]],'columns':len(selection),'securities':len(spec['universe']),
        'read_dates':len(spec['read_sessions']),'warmup':max(f['lookback'] for f in catalog.select(selection))-1,
        'feature_view':full,'training_view':training,'source':source_descriptor(source_path),
        'coverage_owner_metrics':dict(library.coverage_metrics),'timings':timings,'elapsed_seconds':time.monotonic()-started,'input_reuse':reuse_metrics,
        'supplier_calls':0,'training_calls':0,'industry_policy':'sw_supplier_intervals_best_effort_v1',
        'capacity_target_bytes':200_000_000_000,'capacity_target_verified':False,'full_history_12h_verified':False}
    atomic_json(root/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    try:run(cfg)
    except Exception as exc:
        from axiom_research.feature_input_checkpoint import atomic_json
        import traceback
        atomic_json(Path(cfg['destination'])/'failure.json',{'error':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()

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
             'raw_roots':cfg.get('raw_roots',[]),
             'canonical_raw_stage':cfg.get('canonical_raw_stage'),
             'native_domains':cfg.get('native_domains',[])}
    raw_stage=None
    if cfg.get('canonical_raw_stage'):
        raw_stage=json.loads(Path(cfg['canonical_raw_stage']).read_text())
        binding['retained_stock_scope_ref']=digest(raw_stage)
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
    refresh=set(cfg.get('native_domains',[]))
    if missing or refresh:
        parent=store.borrow_snapshot(cfg['market_root'],cfg['market_snapshot'])
        restored=(missing|refresh)&set(parent['domains'])
        if refresh-set(parent['domains']):raise ValueError('requested frozen native domains are absent')
        for domain in sorted(restored):domains[domain]=parent['domains'][domain]
        del parent
        if restored:
            # Publication verifies newly attached Raw IDs. Borrow their exact
            # receipts before attaching the restored canonical domain.
            store.borrow_raw([cfg['market_root'],*cfg.get('raw_roots',[])],
                raw_batch_ids=sorted({bid for name in restored for bid in domains[name]['raw_batch_ids']}))
            base=store.publish_snapshot(domains,parent_snapshot=base,promote=False,
                build_context={'purpose':'retain prior domains and bind explicitly selected frozen native domains',
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
    if raw_stage is not None:
        # Original Raw retains the full market response. Rebuild the retained
        # historical candidate union once, rather than inherit a T01 subset.
        endpoint_domains={'income_vip':'financial_events','balancesheet_vip':'balance_sheet_events',
            'fina_indicator_vip':'financial_indicator_events','daily_basic':'valuation_daily',
            'moneyflow':'money_flow_daily','margin_detail':'margin_daily',
            'stk_holdernumber':'shareholder_count_events','forecast_vip':'forecast_events',
            'index_member_all':'industry_membership_events'}
        selected={ep:domain for ep,domain in endpoint_domains.items()
                  if domain in required and ep in raw_stage['endpoints']}
        codes=sorted(raw_stage['identity_map'])
        ids=list(dict.fromkeys(bid for ep in selected for bid in raw_stage['endpoints'][ep]))
        require_domains=set(selected.values())
        if not ids or not codes:raise ValueError('retained canonical scope has no Raw or stable identities')
        progress('canonical_scope_rebuild',domains=sorted(require_domains),securities=len(codes),raw_receipts=len(ids))
        store.borrow_raw([cfg['candidate_root'],cfg['market_root'],*cfg.get('raw_roots',[])],domains=require_domains)
        data.clear_cache()
        fingerprint=digest([base,ids,codes,sorted(require_domains)])
        base=data.rebuild(base_snapshot=base,raw_batch_ids=ids,domains=sorted(require_domains),
            domain_overrides={domain:{'canonical_symbols':codes} for domain in require_domains},
            operation_id='qsys-canonical-scope-'+fingerprint[7:27],promote=False,
            build_context={'purpose':'restore all retained historical candidates from unchanged original Raw',
                'bulk_chunk':{'job_fingerprint':fingerprint,'chunk_index':0}}).snapshot_id
        data.clear_cache()
        atomic_json(root/'canonical-scope.json',{'snapshot_id':base,'canonical_securities':len(codes),
            'identity_map_ref':digest(raw_stage['identity_map']),
            'canonical_security_ids':sorted(raw_stage['identity_map'].values()),'domains':sorted(require_domains),
            'raw_receipts':len(ids),'supplier_calls':0})
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
    from axiom_research.stock_feature_columns import build_feature_column_source
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
    catalog=load_qsys_feature_catalog()
    from axiom_research.qsys_catalog import select_qsys_features
    selection=[s.to_dict() for s in (catalog.default_selection if cfg.get('feature_names') is None else select_qsys_features(cfg['feature_names']))]
    catalog.select(selection)
    requirements=qsys_input_requirements(cfg.get('feature_names'))
    if cfg.get('prepared_inputs_only'):
        # A published source already owns its frozen scope and axes. Do not
        # rebuild membership or scan arrays in a disposable preparation owner.
        progress('input_reuse')
        source_path=Path(cfg['reuse_source']).resolve()
        original=open_physical_source(source_path)
        from axiom_research.financial_feature_columns import QUARTER_POLICY
        if any('quarters_back' in d and d['selection']!=QUARTER_POLICY
               for d in original.manifest.get('financial_materializations',{}).values()):
            raise ValueError('old quarterly inputs require preparation under the current quarter selection rule')
        spec=original.manifest['spec'];outputs=list(cfg.get('output_sessions',spec['feature_sessions']))
        if not outputs or outputs!=sorted(set(outputs)) or not set(outputs)<=set(spec['feature_sessions']):
            raise ValueError('prepared continuation outside published output scope')
        from axiom_research.stock_feature_inputs import _spec
        universe_id=_spec(spec)
        if cfg.get('universe_id',universe_id)!=universe_id:raise ValueError('configured universe differs from saved scope')
        if cfg.get('candidate_snapshot',spec['snapshot'])!=spec['snapshot']:raise ValueError('prepared Snapshot differs')
        required=set(requirements['daily_fields'])|set(requirements['benchmark_fields'])
        if requirements['industry_fields']:required.add('industry_code')
        required.update(n for bundle in requirements['financial_fields'].values() for n in bundle['outputs'])
        if not required<=set(original.physical_fields()):raise ValueError('published continuation lacks required inputs')
        request_ref=digest({'source':original.identity,'outputs':outputs,'selection':selection})
        reuse_metrics={'reused_dates':len(spec['read_sessions']),'reused_securities':len(spec['universe']),
            'missing_date_count':0,'missing_security_count':0,'old_value_recomputations':0,'buffer_copy_bytes':0,
            'reused_complete_source':True}
        atomic_json(root/'request.json',{'request_ref':request_ref,'spec':spec,'output_sessions':outputs})
        atomic_json(root/'input-reuse.json',reuse_metrics)
        atomic_json(root/'prepared-source.json',source_descriptor(source_path))
        original._history_store.close();del original
    else:
        progress('candidate')
        data,base=prepare_candidate(cfg,root,requirements,progress)
        progress('scope')
        original=None if cfg.get('rebuild_inputs',False) or not cfg.get('reuse_source') else open_physical_source(cfg['reuse_source'])
        seed=json.loads(Path(cfg['scope_seed']).read_text()) if cfg.get('scope_seed') else None
        outputs=cfg.get('output_sessions')
        if outputs is None and not cfg.get('research_range'):
            if seed is None and original is None:raise ValueError('explicit research_range or output sessions required')
            outputs=list(seed['output_sessions'] if seed else original.manifest['spec']['feature_sessions'])
        if outputs is not None and (outputs!=sorted(set(outputs)) or not outputs):
            raise ValueError('explicit sorted output sessions required')
        from axiom_research.stock_feature_inputs import _spec
        source_universe=None if original is None else _spec(original.manifest['spec'])
        universe_id=cfg.get('universe_id',source_universe)
        if not isinstance(universe_id,str) or not universe_id:raise ValueError('explicit configured universe_id required')
        if source_universe is not None and source_universe!=universe_id:raise ValueError('configured universe differs from reused membership')
        spec=history_scope(data,snapshot=base,catalog=catalog,selection=selection,universe_id=universe_id,
            output_sessions=outputs,physical_universe=seed['physical_universe'] if seed else (),destination=root/'scope',
            start_session=cfg.get('research_range',[None,None])[0],end_session=cfg.get('research_range',[None,None])[1],
            calendar_probe_security=cfg.get('calendar_probe_security','cnstock.600000.SH.19991110'))
        outputs=list(spec['feature_sessions'])
        canonical_receipt=Path(cfg['canonical_scope_receipt']) if cfg.get('canonical_scope_receipt') else root/'canonical-scope.json'
        if cfg.get('research_range') and not cfg.get('scope_seed'):
            if not canonical_receipt.exists():raise ValueError('independent production range requires an explicit canonical scope receipt')
            canonical_scope=json.loads(canonical_receipt.read_text())
            if canonical_scope['snapshot_id']!=base or not set(spec['universe'])<=set(canonical_scope['canonical_security_ids']):
                raise ValueError('canonical retained scope does not cover the planned historical security union')
        request_ref=digest(spec)
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
                block_sessions=cfg.get('input_block_sessions',16),checkpoint=True,progress=lambda detail:progress('input_rebuild',detail=detail))
            reuse_metrics={**reuse_metrics,'explicit_full_input_rebuild':True}
        else:
            source,reuse_metrics=reuse_history_source(data,source=original,spec=spec,catalog=catalog,
                destination=reuse_path.parent,event_cache_bytes=cfg.get('event_cache_bytes',64*1024**2),
                block_sessions=cfg.get('input_block_sessions',16),
                progress=lambda detail:progress('input_reuse',detail=detail))
        source_path=reuse_path
        from axiom_research.financial_feature_columns import QUARTER_POLICY
        if any('quarters_back' in d and d['selection']!=QUARTER_POLICY
               for d in source.manifest.get('financial_materializations',{}).values()):
            raise ValueError('old quarterly inputs require preparation under the current quarter selection rule')
        atomic_json(root/'input-reuse.json',reuse_metrics)
        # One missing-input orchestration serves full, field, date and security paths.
        from axiom_research.qsys_preparation import prepare_qsys_inputs
        source,preparation=prepare_qsys_inputs(data,source=source,names=cfg.get('feature_names'),
            destination=root/'inputs/added',event_snapshots={bundle['domain']:base
                for bundle in requirements['financial_fields'].values()},
            report_start=cfg.get('report_start','2010-01-01'),report_end=outputs[-1],
            daily_snapshot=base,benchmark_snapshot=base,industry_snapshot=base,
            benchmark_security_id=cfg['benchmark_security_id'],
            block_sessions=cfg.get('input_block_sessions',16),maximum_added_bytes=8*1024**3,
            maximum_event_cache_bytes=cfg.get('event_cache_bytes',512*1024**2),
            maximum_group_bytes=cfg.get('event_group_bytes',256*1024**2),
            stock_block_limit=cfg.get('event_stock_block_limit'),
            progress=lambda detail:progress(detail['stage'],**{k:v for k,v in detail.items() if k!='stage'}))
        if preparation['source_path'] is not None:source_path=Path(preparation['source_path'])
        atomic_json(root/'input-preparation.json',preparation)
        atomic_json(root/'prepared-source.json',source_descriptor(source_path))
        # Release the Data Snapshot and mutable preparation objects before Core.
        data.clear_cache()
        if original is not None:original._history_store.close()
        del source,data,original
        import gc
        gc.collect()
    progress('source_admission')
    library=FeatureLibrary(cfg.get('library_root',root/'library'),maximum_working_bytes=cfg.get('working_bytes',2*1024**3),
                           maximum_source_bytes=cfg.get('maximum_source_bytes',256*1024**3))
    views=[]
    with library.execution_session(), SavedColumnFeatureSource(source_descriptor(source_path),maximum_source_bytes=cfg.get('maximum_source_bytes',256*1024**3),
                                 maximum_working_bytes=cfg.get('working_bytes',2*1024**3)) as admitted:
        progress('features')
        owner=CatalogFeatureLibrary(library,admitted,catalog)
        block=cfg.get('output_block_sessions',len(outputs))
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
        'source_metrics':dict(admitted.metrics),'source_store_metrics':dict(admitted.store.metrics),'coverage_owner_metrics':dict(library.coverage_metrics),'timings':timings,'elapsed_seconds':time.monotonic()-started,'input_reuse':reuse_metrics,
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

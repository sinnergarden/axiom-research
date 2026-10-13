"""Resumable offline QSYS history preparation. No training or supplier calls.

Config pins existing Raw roots, parent Snapshot and public scope receipt. Work
is single-process. Existing source receipts and FeatureLibrary.ensure checkpoints
are reused; failures/coverage remain explicit. Run with --config /private/file.
"""
import argparse, json, os, shutil, time, hashlib, resource
from pathlib import Path
from collections import defaultdict


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());root=Path(cfg['destination']);root.mkdir(parents=True,exist_ok=True)
    from axiom_data import Data
    from axiom_data.storage import LocalStore
    from axiom_data.stock_input_sources import stock_input_source_spec, NORMALIZER
    from axiom_research.stock_artifacts import file_digest, write_json, digest
    from axiom_research.stock_fold_inputs import seal
    from axiom_research.stock_matrix_storage import write_part
    from axiom_research.stock_feature_columns import build_feature_column_source, extend_feature_column_source
    from axiom_research.stock_feature_library import SavedColumnFeatureSource,CatalogFeatureLibrary
    from axiom_research.qsys_catalog import load_qsys_feature_catalog,qsys_input_requirements
    from axiom_research.qsys_preparation import prepare_qsys_inputs
    from axiom_engine.feature_library import FeatureLibrary
    started=time.monotonic()
    def save(name,obj):
        tmp=root/(name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,default=str));tmp.replace(root/name)
    def progress(stage,**extra):
        free=shutil.disk_usage(root).free;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record={'stage':stage,'elapsed_seconds':time.monotonic()-started,'peak_rss_bytes':rss,'disk_free_bytes':free,**extra}
        save('progress.json',record);print(json.dumps(record,default=str),flush=True)
        if free<cfg.get('minimum_free_bytes',15*1024**3):raise RuntimeError('DISK_RESERVE_REACHED: completed checkpoints retained')
        if rss>cfg.get('maximum_rss_bytes',12*1024**3):raise RuntimeError('RSS_LIMIT_REACHED: completed checkpoints retained')
    def link(source,target):
        if target.exists():return
        target.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(source,target)
        except OSError:shutil.copyfile(source,target)
    def descriptor(path):
        path=path.resolve();return {'path':str(path),'file_digest':file_digest(path),'source_ref':json.loads(path.read_text())['source_ref']}
    def load_source(path):
        # Borrow mmap without repeatedly rebuilding saved physical inputs.
        import numpy as np
        from axiom_research.stock_feature_columns import FeatureColumnSource
        m=json.loads(path.read_text());arrays={};types={'float64_le':'<f8','int64_le':'<i8','bool_u8':'u1','uint8':'u1'}
        for name,b in m['buffers'].items():
            if file_digest(b['path'])!=b['file_digest']:raise ValueError('saved input hash mismatch')
            value=np.memmap(b['path'],mode='r',dtype=types[b['dtype']],shape=tuple(b['shape']))
            arrays[name]=value.view(bool) if b['dtype']=='bool_u8' else value
        return FeatureColumnSource(m,arrays)
    progress('loading_scope')
    scope_receipt=json.loads(Path(cfg['scope_receipt']).read_text());scope=scope_receipt['result'];request=scope_receipt['request']
    symbols=scope['read_symbols'];sessions=scope['read_sessions'];outputs=request['output_sessions']
    # Keep compact scope proof for the existing source owner; original receipt
    # stays untouched, with its full explicit membership/warmup gaps.
    gaps=scope.pop('missing_reasons',[])
    save('scope-gaps.json',{'count':len(gaps),'examples':gaps[:25],'original_receipt':cfg['scope_receipt']});del gaps
    proof={'source':'Data.plan_scope','receipt_file':cfg['scope_receipt'],'receipt_digest':file_digest(cfg['scope_receipt'])}
    bundle=seal({'contract_version':'stock_scope_bundle_v1','purpose':'QSYS CSI1800 all-feature cache',
        'request':request,'request_ref':digest(request),'result':scope,'result_ref':digest(scope),
        'source_proof':proof,'source_proof_ref':digest(proof)},'scope_bundle_ref')
    scope_descriptor=write_part(root,bundle,'scope_bundle_ref')
    candidate=root/'data';candidate.mkdir(exist_ok=True);store=LocalStore(candidate)
    checkpoint=root/'raw-stage.json'
    if checkpoint.exists():staged=json.loads(checkpoint.read_text())
    else:
        targets={'daily_basic','moneyflow','margin_detail','income_vip','balancesheet_vip','fina_indicator_vip','forecast_vip','stk_holdernumber','index_member_all'}
        staged={'endpoints':{},'identity_map':{},'next_open_session_by_date':{}};ids=set();by_endpoint=defaultdict(list)
        rawdir=candidate/'raw';rawdir.mkdir(exist_ok=True);tmp=rawdir/'fetches.jsonl.partial'
        with tmp.open('wb') as out:
            for rawroot in cfg['raw_roots']:
                old=LocalStore(rawroot);log=Path(rawroot)/'raw/fetches.jsonl'
                with log.open('rb') as f:
                    for line in f:
                        compact=json.loads(line);req=compact.get('request',{});ep=req.get('endpoint')
                        if ep not in targets or compact.get('status') not in ('success','empty'):continue
                        if not str(compact.get('operation_id','')).startswith('qsys'):continue
                        rec=old._decode_raw_line(line);bid=rec['batch_id']
                        if bid in ids:continue
                        ids.add(bid);out.write(line);link(Path(rawroot)/rec['payload_uri'],candidate/rec['payload_uri'])
                        profile=rec['source_profile']
                        for code,identity in profile.get('identity_map',{}).items():
                            prior=staged['identity_map'].setdefault(code,identity)
                            if prior!=identity:raise ValueError('conflicting stock identities')
                        calendar=profile.get('next_open_session_by_date',profile.get('availability',{}).get('next_open_session_by_date',{}))
                        for day,next_day in calendar.items():
                            prior=staged['next_open_session_by_date'].setdefault(day,next_day)
                            if prior!=next_day:raise ValueError('conflicting supplier calendar')
                        by_endpoint[ep].append(bid)
                        if len(ids)%250==0:progress('raw_stage',receipts=len(ids))
        tmp.replace(rawdir/'fetches.jsonl');staged['endpoints']=dict(by_endpoint);save('raw-stage.json',staged)
    progress('raw_staged',receipts=sum(map(len,staged['endpoints'].values())))
    # Existing native market domains are borrowed as immutable objects. The
    # original Raw roots remain required for their older source audit closure.
    base_marker=root/'base-snapshot.json'
    if base_marker.exists():base=json.loads(base_marker.read_text())['snapshot_id']
    else:
        oldstore=LocalStore(cfg['market_root']);parent=oldstore.load_snapshot(cfg['market_snapshot'])
        (candidate/'snapshots').mkdir(exist_ok=True)
        shutil.copyfile(Path(cfg['market_root'])/'snapshots'/f"{cfg['market_snapshot']}.json",candidate/'snapshots'/f"{cfg['market_snapshot']}.json")
        keep=('market_daily','adjustment_factors','universe_membership','trading_calendar','security_master','price_limits','security_status','benchmark_daily','top_holders_reports')
        domains={k:parent['domains'][k] for k in keep}
        for domain in domains.values():
            for part in domain['partitions']:link(Path(cfg['market_root'])/part['uri'],candidate/part['uri'])
        base=store.publish_snapshot(domains,parent_snapshot=cfg['market_snapshot'],build_context={'purpose':'QSYS native-domain frozen reuse','original_raw_root':cfg['market_root']},promote=False)['snapshot_id']
        save('base-snapshot.json',{'snapshot_id':base});del parent,domains
    data=Data(candidate,cache_bytes=64*1024**2,max_readers=1)
    selected_codes=sorted(k for k,v in staged['identity_map'].items() if v in set(symbols))
    for ep,ids in sorted(staged['endpoints'].items()):
        marker=root/(ep+'-snapshot.json')
        if marker.exists():base=json.loads(marker.read_text())['snapshot_id'];continue
        progress('rebuild',endpoint=ep,raw_receipts=len(ids))
        spec=stock_input_source_spec(ep,identity_map=staged['identity_map'],next_open_session_by_date=staged['next_open_session_by_date'],
            lending_sell_unit_by_security={v:'shares' for v in staged['identity_map'].values()})
        override={k:spec[k] for k in ('contract','source_profile')};override.update(normalizer=NORMALIZER,canonical_symbols=selected_codes)
        result=data.rebuild(base_snapshot=base,raw_batch_ids=ids,domains=[spec['domain']],
            operation_id='qsys-full-offline-v1-'+ep,build_context={'purpose':'offline original Raw replay, no source calls','scope_receipt':proof},
            promote=False,domain_overrides={spec['domain']:override})
        base=result.snapshot_id;save(marker.name,{'snapshot_id':base});data.clear_cache()
    save('candidate-snapshot.json',{'snapshot_id':base,'root':str(candidate)})
    # Selected rebuilds never alter scope's market/calendar/membership domains.
    # Rebind their already computed axes explicitly to the derived Snapshot.
    scope['snapshot_id']=base;request={**request,'snapshot':base}
    proof={**proof,'projection':'unchanged native trading_calendar and universe_membership domains',
           'original_snapshot':cfg['market_snapshot'],'derived_snapshot':base}
    bundle=seal({'contract_version':'stock_scope_bundle_v1','purpose':'QSYS CSI1800 all-feature cache',
        'request':request,'request_ref':digest(request),'result':scope,'result_ref':digest(scope),
        'source_proof':proof,'source_proof_ref':digest(proof)},'scope_bundle_ref')
    scope_descriptor=write_part(root,bundle,'scope_bundle_ref')
    catalog=load_qsys_feature_catalog();selection=[vars(s) for s in catalog.default_selection]
    spec={'contract_version':'stock_feature_inputs_spec_v1','scope':scope_descriptor,'snapshot':base,'pit_policy':'best_effort_vendor_v1',
          'calendar':sessions,'read_sessions':sessions,'feature_sessions':outputs,'universe':symbols,'catalog_ref':catalog.identity,
          'feature_selection':selection,'ordered_features':[s['id'] for s in selection],
          'cutoff_by_session':{d:d+'T20:30:00+08:00' for d in sessions}}
    source_path=root/'inputs/base/source.json'
    if source_path.exists():source=load_source(source_path)
    else:
        progress('base_inputs',sessions=len(sessions),securities=len(symbols))
        source,_=build_feature_column_source(data,spec=spec,destination=source_path.parent,block_sessions=2,
            progress=lambda p:progress('base_inputs',detail=p))
    requirements=qsys_input_requirements();daily=requirements['daily_fields']
    # Each domain checkpoints independently instead of restarting 21 columns.
    for domain in sorted({d['domain'] for d in daily.values()}):
        path=root/'inputs'/('daily-'+domain)/'source.json'
        if path.exists():source=load_source(path);continue
        progress('daily_inputs',domain=domain)
        source,_=extend_feature_column_source(data,source=source,daily_fields={k:v for k,v in daily.items() if v['domain']==domain},
            destination=path.parent,block_sessions=2,maximum_added_bytes=8*1024**3)
    # One public subset per financial bundle provides durable input checkpoints.
    from axiom_data import EventQuery
    from axiom_research.financial_feature_columns import extend_financial_feature_source
    for key,bundle in requirements['financial_fields'].items():
        path=root/'inputs'/key/'source.json'
        if path.exists():source=load_source(path);continue
        progress('financial_inputs',bundle=key)
        filters={'endpoint':bundle['endpoint']} if bundle['endpoint'] else {}
        if bundle['report_type']:filters['report_type']=bundle['report_type']
        query=EventQuery(bundle['domain'],(bundle['field'],),tuple(symbols),cfg.get('report_start','2010-01-01'),outputs[-1],
            outputs[-1]+'T20:30:00+08:00','best_effort_vendor_v1','report_period',filters)
        source,_=extend_financial_feature_source(data,source=source,query=query,field=bundle['field'],outputs=bundle['outputs'],unit=bundle['unit'],
            destination=path.parent,event_snapshot=base,session_timezone=bundle['session_timezone'],block_sessions=2,
            maximum_added_bytes=2*1024**3,maximum_event_cache_bytes=64*1024**2)
    from axiom_research.benchmark_feature_columns import extend_benchmark_feature_source
    from axiom_research.industry_feature_columns import extend_industry_feature_source
    for kind,fn,kw in [('benchmark',extend_benchmark_feature_source,{'benchmark_security_id':cfg['benchmark_security_id']}),('industry',extend_industry_feature_source,{})]:
        path=root/'inputs'/kind/'source.json'
        if path.exists():source=load_source(path);continue
        progress(kind+'_inputs');source,_=fn(data,source=source,snapshot=base,destination=path.parent,block_sessions=2,maximum_added_bytes=512*1024**2,**kw)
    source_path=root/'inputs/industry/source.json';del source;data.clear_cache()
    # Existing private cache can be safely imported as immutable objects. Any
    # differing reference cohort or dependency identity is recomputed by ensure.
    libroot=root/'library';libroot.mkdir(exist_ok=True)
    if cfg.get('existing_library') and not (root/'library-import.json').exists():
        old=Path(cfg['existing_library']);count=0
        for path in old.rglob('*'):
            if not path.is_file() or path.name.startswith('.') or 'lock' in path.name:continue
            target=libroot/path.relative_to(old)
            if target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            if path.suffix in ('.bin','.buffer'):link(path,target)
            else:shutil.copyfile(path,target)
            count+=1
        save('library-import.json',{'source':str(old),'files':count})
    library=FeatureLibrary(libroot,maximum_working_bytes=cfg.get('working_bytes',2*1024**3))
    receipts=root/'views';receipts.mkdir(exist_ok=True)
    with SavedColumnFeatureSource(descriptor(source_path),maximum_source_bytes=64*1024**3,maximum_working_bytes=cfg.get('working_bytes',2*1024**3)) as admitted:
        owner=CatalogFeatureLibrary(library,admitted,catalog)
        for index in range(0,len(outputs),cfg.get('output_block_sessions',5)):
            days=outputs[index:index+cfg.get('output_block_sessions',5)];marker=receipts/(days[0]+'.json')
            if marker.exists():continue
            progress('features',first=days[0],last=days[-1],completed_dates=index,total_dates=len(outputs),columns=len(selection))
            view,metrics=owner.ensure(selection=selection,output_sessions=days,normalized=True,
                progress=lambda p:progress('features',first=days[0],detail=p))
            with library.open_feature_view(view) as opened:
                values=opened.project();valid=values['validity'];coverage={'valid':int(valid.sum()),'total':int(valid.size)}
            write_json(marker,{'view':view,'metrics':metrics,'coverage':coverage,'catalog_ref':catalog.identity})
    progress('complete',dates=len(outputs),columns=len(selection),views=str(receipts),supplier_calls=0,training_calls=0)
    save('result.json',{'status':'COMPLETE_WITH_EXPLICIT_MISSING','catalog_ref':catalog.identity,'dates':len(outputs),'columns':len(selection),'views':str(receipts),'scope_gaps':str(root/'scope-gaps.json'),'industry_policy':'sw_supplier_intervals_best_effort_v1'})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)

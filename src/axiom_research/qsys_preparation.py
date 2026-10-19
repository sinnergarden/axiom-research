"""One missing-input orchestration for full, field and physical-axis preparation."""
from pathlib import Path

from .qsys_catalog import qsys_input_requirements
from .stock_fold_inputs import require


def prepare_qsys_inputs(data, *, source, names, destination, event_snapshots,
                        report_start, report_end, security_ids=None,
                        daily_snapshot=None, industry_snapshot=None, benchmark_snapshot=None,
                        benchmark_security_id=None, block_sessions=16,
                        maximum_added_bytes=512*1024**2,
                        maximum_event_cache_bytes=64*1024**2, input_fields=None,
                        maximum_group_bytes=256*1024**2, stock_block_limit=None,
                        progress=None):
    """Prepare absent fields through the same bounded producers and checkpoints.

    Existing numeric leaves retain their descriptors. Daily reads borrow the
    fixed Data owner's coverage inside the producer; no public read changes.
    A completed stage is admitted by its parent/parameters and immutable source,
    so a full run and an axis extension resume the same implementation.
    Nothing collects suppliers or computes Feature mathematics here.
    """
    from axiom_data import EventQuery
    from .stock_feature_columns import extend_feature_column_source
    from .financial_feature_columns import extend_financial_feature_source,extend_financial_feature_sources
    from .stock_artifacts import digest
    from .feature_input_checkpoint import atomic_json
    from .qsys_history_inputs import open_physical_source
    import json

    require(type(block_sessions) is int and 0<block_sessions<=32,'bounded input block required')
    requirements=qsys_input_requirements(names)
    wanted=None if input_fields is None else set(input_fields)
    spec=source.manifest['spec']
    symbols=sorted(spec['universe'] if security_ids is None else security_ids)
    require(bool(symbols) and set(symbols)<=set(spec['universe']),
            'requested securities outside physical source')
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    metrics={'daily':{},'financial':{},'reused_fields':[],'source_path':None}

    def announce(name, detail):
        if progress is None:return
        phase='daily_inputs' if name.startswith('daily-') else 'financial_inputs' if name.startswith(('events-','financial-')) else name+'_inputs'
        progress({'stage':phase,'input_stage':name,**detail})

    def stage(name, fields, parameters, producer):
        nonlocal source
        path=root/name/'source.json';receipt=path.parent/'stage.json'
        binding=digest({'parent_source_ref':source.identity,'fields':sorted(fields),
                        'parameters':parameters,'block_sessions':block_sessions})
        saved=json.loads(receipt.read_text()) if receipt.exists() else None
        require(saved is None or saved['binding']==binding,'saved input stage binding differs: '+name)
        if path.exists():
            require(saved is not None,'published input stage lacks request binding: '+name)
            value=open_physical_source(path)
            expected_spec=source.manifest['spec'];actual_spec=value.manifest['spec']
            require({k:v for k,v in actual_spec.items() if k!='snapshot'}==
                    {k:v for k,v in expected_spec.items() if k!='snapshot'} and
                    actual_spec['snapshot'] in (expected_spec['snapshot'],parameters.get('snapshot')) and
                    set(fields)<=set(value.physical_fields()) and
                    all(value.manifest['buffers'].get(k)==v for k,v in source.manifest['buffers'].items()),
                    'saved input stage changes inherited fields or scope: '+name)
            require(saved.get('source_ref') in (None,value.identity),'saved stage source identity differs')
            detail={'reused_complete_stage':True,'data_read_calls':0}
        else:
            atomic_json(receipt,{'binding':binding,'source_ref':None,'status':'PREPARING'})
            announce(name,{'status':'PREPARING'})
            value,detail=producer(path.parent,lambda d:announce(name,d))
        atomic_json(receipt,{'binding':binding,'source_ref':value.identity,'status':'COMPLETE','metrics':detail})
        source=value;metrics['source_path']=str(path)
        announce(name,{'status':'COMPLETE','metrics':detail})
        return detail

    present=set(source.physical_fields())
    daily={name:decl for name,decl in requirements['daily_fields'].items()
           if name not in present and (wanted is None or name in wanted)}
    metrics['reused_fields'].extend(sorted(set(requirements['daily_fields']) & present))
    for domain in sorted({v['domain'] for v in daily.values()}):
        fields={k:v for k,v in daily.items() if v['domain']==domain}
        metrics['daily'][domain]=stage('daily-'+domain,fields,{'snapshot':daily_snapshot,'fields':fields},
            lambda path,emit:extend_feature_column_source(data,source=source,daily_fields=fields,
                destination=path,snapshot=daily_snapshot,block_sessions=block_sessions,
                maximum_added_bytes=maximum_added_bytes,progress=emit))

    # One existing EventReadOwner traversal for all compatible financial leaves.
    if symbols==spec['universe']:
        groups={}
        for key,bundle in requirements['financial_fields'].items():
            outputs={n:p for n,p in bundle['outputs'].items()
                     if n not in source.physical_fields() and (wanted is None or n in wanted)}
            if outputs:groups.setdefault(event_snapshots[bundle['domain']],{})[key]={**bundle,'outputs':outputs}
        for ordinal,(snapshot,bundles) in enumerate(groups.items()):
            fields=[n for bundle in bundles.values() for n in bundle['outputs']]
            metrics['financial']['unified_'+str(ordinal)]=stage('events-'+str(ordinal),fields,
                {'snapshot':snapshot,'bundles':bundles,'report_start':report_start,'report_end':report_end},
                lambda path,emit:extend_financial_feature_sources(data,source=source,bundles=bundles,
                    destination=path,event_snapshot=snapshot,report_start=report_start,report_end=report_end,
                    maximum_added_bytes=maximum_added_bytes,maximum_event_cache_bytes=maximum_event_cache_bytes,
                    maximum_group_bytes=maximum_group_bytes,stock_block_limit=stock_block_limit,progress=emit))
    # Preserve the existing explicit partial-security financial recipe route.
    for key,bundle in requirements['financial_fields'].items():
        present=set(source.physical_fields());reused=set(bundle['outputs']) & present
        source.require_coverage(reused,symbols)
        metrics['reused_fields'].extend(sorted(reused))
        outputs={name:p for name,p in bundle['outputs'].items()
                 if name not in present and (wanted is None or name in wanted)}
        if not outputs:continue
        filters={'endpoint':bundle['endpoint']} if bundle['endpoint'] is not None else {}
        if bundle['report_type'] is not None:filters['report_type']=bundle['report_type']
        query=EventQuery(bundle['domain'],(bundle['field'],),tuple(symbols),report_start,report_end,
            max(spec['cutoff_by_session'].values()),spec['pit_policy'],'report_period',filters)
        metrics['financial'][key]=stage('financial-'+key,outputs,
            {'snapshot':event_snapshots[bundle['domain']],'bundle':bundle,'symbols':symbols,
             'report_start':report_start,'report_end':report_end},
            lambda path,emit:extend_financial_feature_source(data,source=source,query=query,
                field=bundle['field'],outputs=outputs,unit=bundle['unit'],destination=path,
                event_snapshot=event_snapshots[bundle['domain']],session_timezone=bundle['session_timezone'],
                block_sessions=block_sessions,maximum_added_bytes=maximum_added_bytes,
                maximum_event_cache_bytes=maximum_event_cache_bytes,progress=emit))

    if requirements['benchmark_fields'] and (wanted is None or 'benchmark_close' in wanted):
        if 'benchmark_close' in source.physical_fields():
            existing=source.manifest['source_profiles']['materialized_benchmark_daily']['source_profile']['benchmark_security_id']
            require(benchmark_security_id==existing,'existing benchmark identity differs from request')
            metrics['reused_fields'].append('benchmark_close')
        else:
            from .benchmark_feature_columns import extend_benchmark_feature_source
            metrics['benchmark']=stage('benchmark',['benchmark_close'],
                {'snapshot':benchmark_snapshot,'benchmark_security_id':benchmark_security_id},
                lambda path,emit:extend_benchmark_feature_source(data,source=source,
                    snapshot=benchmark_snapshot,benchmark_security_id=benchmark_security_id,
                    destination=path,block_sessions=block_sessions,maximum_added_bytes=maximum_added_bytes,progress=emit))
    if requirements['industry_fields'] and (wanted is None or 'industry_code' in wanted):
        if 'industry_code' in source.physical_fields():
            existing=source.manifest['source_profiles']['materialized_industry_intervals']['source_profile']['snapshot_id']
            require(existing==industry_snapshot,'existing industry snapshot differs from request')
            metrics['reused_fields'].append('industry_code')
        else:
            from .industry_feature_columns import extend_industry_feature_source
            metrics['industry']=stage('industry',['industry_code'],{'snapshot':industry_snapshot},
                lambda path,emit:extend_industry_feature_source(data,source=source,snapshot=industry_snapshot,
                    destination=path,block_sessions=block_sessions,maximum_added_bytes=maximum_added_bytes,progress=emit))
    return source,metrics

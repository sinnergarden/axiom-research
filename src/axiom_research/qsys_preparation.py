"""Incremental QSYS input preparation through the existing public adapters."""
from pathlib import Path

from .qsys_catalog import qsys_input_requirements
from .stock_fold_inputs import require


def prepare_qsys_inputs(data, *, source, names, destination, event_snapshots,
                        report_start, report_end, security_ids=None,
                        daily_snapshot=None, benchmark_snapshot=None, benchmark_security_id=None, block_sessions=20,
                        maximum_added_bytes=512*1024**2,
                        maximum_event_cache_bytes=64*1024**2):
    """Append only inputs absent from a saved physical source for a subset.

    The caller freezes report range and each financial domain's Snapshot.
    Existing leaves and their evidence are reused without source reads. Feature
    computation remains FeatureLibrary.ensure; this entrypoint never trains or
    contacts suppliers. Existing financial leaves must cover the requested IDs.
    """
    from axiom_data import EventQuery
    from .stock_feature_columns import extend_feature_column_source
    from .financial_feature_columns import extend_financial_feature_source

    requirements=qsys_input_requirements(names)
    spec=source.manifest['spec']
    symbols=sorted(spec['universe'] if security_ids is None else security_ids)
    require(bool(symbols) and set(symbols)<=set(spec['universe']),
            'requested securities outside physical source')
    root=Path(destination)
    present=set(source.physical_fields())
    metrics={'daily':None,'financial':{},'reused_fields':[]}
    daily={name:decl for name,decl in requirements['daily_fields'].items() if name not in present}
    metrics['reused_fields'].extend(sorted(set(requirements['daily_fields']) & present))
    if daily:
        source,metrics['daily']=extend_feature_column_source(data,source=source,
            daily_fields=daily,destination=root/'daily',snapshot=daily_snapshot,
            block_sessions=block_sessions,maximum_added_bytes=maximum_added_bytes)
    for key,bundle in requirements['financial_fields'].items():
        present=set(source.physical_fields())
        reused=set(bundle['outputs']) & present
        source.require_coverage(reused,symbols)
        outputs={name:projection for name,projection in bundle['outputs'].items() if name not in present}
        metrics['reused_fields'].extend(sorted(reused))
        if not outputs:
            continue
        filters={'endpoint':bundle['endpoint']}
        if bundle['report_type'] is not None:
            filters['report_type']=bundle['report_type']
        query=EventQuery(bundle['domain'],(bundle['field'],),tuple(symbols),
            report_start,report_end,max(spec['cutoff_by_session'].values()),
            spec['pit_policy'],'report_period',filters)
        source,metrics['financial'][key]=extend_financial_feature_source(data,
            source=source,query=query,field=bundle['field'],outputs=outputs,
            unit=bundle['unit'],destination=root/key,
            event_snapshot=event_snapshots[bundle['domain']],
            session_timezone=bundle['session_timezone'],block_sessions=block_sessions,
            maximum_added_bytes=maximum_added_bytes,
            maximum_event_cache_bytes=maximum_event_cache_bytes)
    if requirements['benchmark_fields']:
        if 'benchmark_close' in source.physical_fields():
            existing=source.manifest['source_profiles']['materialized_benchmark_daily']['source_profile']['benchmark_security_id']
            require(benchmark_security_id==existing,'existing benchmark identity differs from request')
            metrics['reused_fields'].append('benchmark_close')
        else:
            from .benchmark_feature_columns import extend_benchmark_feature_source
            source,metrics['benchmark']=extend_benchmark_feature_source(data,source=source,
                snapshot=benchmark_snapshot,benchmark_security_id=benchmark_security_id,
                destination=root/'benchmark',block_sessions=block_sessions,
                maximum_added_bytes=maximum_added_bytes)
    return source,metrics
